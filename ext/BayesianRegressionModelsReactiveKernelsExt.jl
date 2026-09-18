module BayesianRegressionModelsReactiveKernelsExt

using BayesianRegressionModels
using LogDensityProblems
using ReactiveKernels
using ReactiveKernelsPPL

const BRM = BayesianRegressionModels

# Boundary serializer: BRM-side `_RKStructuralPlan` (plain data, no RK types)
# to the thin-layer `StructuralPlan` (contract v1+v2). The core emitter gates
# slice-1 admission, so every map below translates and never admits: an
# unmapped key is an internal contractor error, while a
# `ContractValidationError` from `build_kernel` propagates as the hard
# spec-build-time error the contract requires.

const _RK_PPL_FAMILY = Dict{Symbol,ReactiveKernelsPPL.LikelihoodFamily}(
    :gaussian => ReactiveKernelsPPL.GaussianFam,
    :bernoulli_logit => ReactiveKernelsPPL.BernoulliLogitFam,
    :poisson_log => ReactiveKernelsPPL.PoissonLogFam,
    :binomial_logit => ReactiveKernelsPPL.BinomialLogitFam,
    :nb2_log => ReactiveKernelsPPL.NegativeBinomial2Fam,
    :gamma_log => ReactiveKernelsPPL.GammaLogFam,
)

const _RK_PPL_LINK = Dict{Symbol,ReactiveKernelsPPL.LinkFunction}(
    :identity => ReactiveKernelsPPL.IdentityLink,
    :logit => ReactiveKernelsPPL.LogitLink,
    :log => ReactiveKernelsPPL.LogLink,
)

const _RK_PPL_TERM = Dict{Symbol,ReactiveKernelsPPL.TermKind}(
    :intercept => ReactiveKernelsPPL.InterceptTerm,
    :continuous => ReactiveKernelsPPL.ContinuousTerm,
    :factor => ReactiveKernelsPPL.FactorTerm,
    :offset => ReactiveKernelsPPL.OffsetTerm,
)

const _RK_PPL_SAMPLED_FAMILY = Dict{Symbol,Symbol}(
    :Normal => :normal,
    :Cauchy => :cauchy,
    :Exponential => :exponential,
    :Gamma => :gamma,
    :LogNormal => :lognormal,
    :Beta => :beta,
    :InverseGamma => :inverse_gamma,
    :Flat => :flat,
)

function _rk_ppl_mapped(map::AbstractDict, key::Symbol, what::String,
        label::Symbol)
    haskey(map, key) ||
        error("RK backend: internal: unmapped $what `$key` at `$label` " *
              "(the emitter plan already gates $what)")
    map[key]
end

function _rk_ppl_evidence(evidence::BRM._RKResponseEvidence)
    ResponseEvidence(evidence.kind, evidence.lower, evidence.upper)
end

function _rk_ppl_response(spec::BRM._RKLikelihoodSpec)
    LikelihoodSpec(
        _rk_ppl_mapped(_RK_PPL_FAMILY, spec.family, "family", spec.label),
        _rk_ppl_mapped(_RK_PPL_LINK, spec.link, "link", spec.label),
        spec.response, spec.predictor, spec.scale, spec.weights,
        _rk_ppl_evidence(spec.evidence), spec.label, spec.trials, nothing)
end

function _rk_ppl_term(term::BRM._RKTermSpec)
    TermSpec(
        _rk_ppl_mapped(_RK_PPL_TERM, term.kind, "term kind", term.label),
        term.columns, term.options, term.addressee, term.label)
end

function _rk_ppl_predictor(spec::BRM._RKPredictorSpec)
    PredictorSpec(spec.name,
        _rk_ppl_mapped(_RK_PPL_LINK, spec.link, "link", spec.label),
        _rk_ppl_term.(spec.terms), spec.label)
end

function _rk_ppl_prior(prior::BRM._RKPopulationPrior)
    PopulationPrior(prior.predictor, prior.addressee, prior.location,
        prior.scale)
end

function _rk_ppl_parameter(parameter::BRM._RKSampledParameter)
    family = _rk_ppl_mapped(_RK_PPL_SAMPLED_FAMILY, parameter.family,
        "sampled family", parameter.label)
    args = NamedTuple(
        Symbol(:arg, i) => value for (i, value) in enumerate(parameter.args))
    SampledParameter(parameter.name, family, args,
        parameter.support_override, parameter.label)
end

function _rk_ppl_assignment(spec::BRM._RKAssignmentSpec)
    AssignmentSpec(spec.name,
        BRM._rk_lower_assignment_expr(spec.expression, spec.name), spec.label)
end

function _rk_ppl_derived(spec::BRM._RKDerivedSpec)
    VectorAssignmentSpec(spec.name, spec.expression, spec.label)
end

# The LevelMap subset a factor term's options select: full cover, or every
# observed position but the reference drop (edge drops as ranges, middle
# drops as index lists — the same literals the AST surface emits).
function _rk_ppl_levelsubset(options::NamedTuple, K::Int)
    options.coding === :fullrank && return Colon()
    p = options.drop
    p == 1 && return UnitRange(2, K)
    p == K && return UnitRange(1, K - 1)
    return Vector{Int}([1:p-1; p+1:K])
end

function _rk_ppl_levelmaps(plan::BRM._RKStructuralPlan)
    maps = LevelMap[]
    for predictor in plan.predictors, term in predictor.terms
        term.kind === :factor || continue
        col = only(term.columns)
        K = length(BRM._rk_grouping_levels(plan.columns[col]))
        push!(maps, LevelMap(predictor.name, col, [], :levels,
            _rk_ppl_levelsubset(term.options, K)))
    end
    maps
end

function _rk_ppl_structural_plan(plan::BRM._RKStructuralPlan)
    StructuralPlan(
        _rk_ppl_response.(plan.responses),
        _rk_ppl_predictor.(plan.predictors),
        _rk_ppl_prior.(plan.population_priors),
        _rk_ppl_parameter.(plan.parameters),
        _rk_ppl_assignment.(plan.assignments),
        plan.columns, plan.n_obs;
        derived=_rk_ppl_derived.(plan.derived),
        levelmaps=_rk_ppl_levelmaps(plan))
end

# The executable `model` of an `RKBRMI` is the thin-layer `(; spec, layout)`
# pair: the `KernelSpec` callable after `prepare`, plus its `LayoutTable`.
# Expressible plans route through the `@rkppl` AST surface
# (`lower_rkppl` + `bind_data`, the same function the macro lowers
# through); the inexpressible subset (offset-only predictors,
# predictor/data name overlap) keeps the legacy direct serializer.
function BRM._brm_rk_model(plan::BRM._RKStructuralPlan)
    ast = BRM._rk_emit_ast(plan)
    translated = if ast === nothing
        bind_data(_rk_ppl_structural_plan(plan), plan.columns)
    else
        unbound = lower_rkppl(ast, Tuple(sort!(collect(keys(plan.columns)))))
        bind_data(unbound, plan.columns)
    end
    build_kernel(translated)
end

function BRM.RKBRMI(brmi::BRM.BRMI)
    plan = BRM._brm_rk_plan(brmi)
    BRM.RKBRMI(brmi, plan, BRM._brm_rk_model(plan))
end

# LogDensityProblems shim over the thin-layer sampler query: the packed
# unconstrained coordinates are the sampler space, so no transform sits
# between the sampler and the kernel. The `StructuralPlan` is re-serialized
# from the BRM-side plan (pure and cheap) because `prepare_sampler` derives
# its have/bound boundary from it; `model` stays exactly `build_kernel`
# output.
struct RKLogDensityProblem{Q<:SamplerQuery}
    query::Q
end

function BRM.rk_logdensity_problem(backend::BRM.RKBRMI;
        ad_backend,
        u0=zeros(Float64, backend.model.layout.total))
    translated = _rk_ppl_structural_plan(backend.plan)
    query = prepare_sampler(backend.model, translated, u0; backend=ad_backend)
    RKLogDensityProblem(query)
end

LogDensityProblems.capabilities(::Type{<:RKLogDensityProblem}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(problem::RKLogDensityProblem) =
    problem.query.layout.total
LogDensityProblems.logdensity(problem::RKLogDensityProblem,
        position::AbstractVector) = problem.query(position)
function LogDensityProblems.logdensity_and_gradient(
        problem::RKLogDensityProblem, position::AbstractVector)
    u = position isa Vector{Float64} ? position : Vector{Float64}(position)
    gradient = Vector{Float64}(undef, length(u))
    value, _ = sampler_value_and_gradient!(problem.query, gradient, u)
    value, gradient
end

function BRM.rk_restore_draws(backend::BRM.RKBRMI, U::AbstractMatrix)
    restore_draws(backend.model.layout, U)
end

end
