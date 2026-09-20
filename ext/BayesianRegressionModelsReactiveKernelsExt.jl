module BayesianRegressionModelsReactiveKernelsExt

using BayesianRegressionModels
using LogDensityProblems
using ReactiveKernels
using ReactiveKernelsPPL

const BRM = BayesianRegressionModels

# Sole emission path: BRM-side plan (plain data, no RK types) → `@rkppl`
# program (`BRM._rk_emit_ast`, total over admitted plans: submodel defs
# + main block) → thin-layer `StructuralPlan` via `lower_rkppl` +
# `bind_data` (the same function the macro lowers through). Model
# execution and the sampler boundary both derive from this one
# lowering, so the boundary plan is definitionally the plan the model
# was built from — there is no parallel direct serializer to drift
# (the retired one did: factor term options and the preserved Binomial
# triple-3 are both rejected from hand-built plans yet accepted from
# the AST route). Ordinal extras are the one plan-level patch: the
# surface spells `Ordinal` with three positionals only, so after
# lowering the extension rebuilds ordinal responses carrying extras,
# translates modeled-scale predictors (which the AST skips — they have
# no response use-site), and appends the per-threshold coefficient
# vectors; `bind_data` then validates the patched plan with the full
# thin-layer suite. Kernel plans ride the same route; until
# the thin-layer KernelPlate reader lands, `lower_rkppl`/`build_kernel`
# fail closed with thin-layer attribution (leaf 14bv4nq).
const _RK_PLAN_TYPES = Union{BRM._RKStructuralPlan,BRM._RKKernelPlan}

# Population term kinds a modeled ordinal scale admits (the planner gates
# the same set; anything else is an internal error here).
const _RK_PPL_TERM = Dict{Symbol,TermKind}(
    :intercept => InterceptTerm,
    :continuous => ContinuousTerm,
    :factor => FactorTerm,
    :offset => OffsetTerm,
)

# The LevelMap subset a scale factor term's options select: full cover,
# or every observed position but the reference drop (edge drops as
# ranges, middle drops as index lists — the same values the surface
# parses from the AST subset literals).
function _rk_ppl_levelsubset(options::NamedTuple, K::Int)
    options.coding === :fullrank && return Colon()
    p = options.drop
    p == 1 && return UnitRange(2, K)
    p == K && return UnitRange(1, K - 1)
    return Vector{Int}([1:p-1; p+1:K])
end

# Discrimination symbols naming a planned predictor (predictor-first,
# mirroring the planner): the modeled scales, deduped in plan order.
function _rk_ordinal_scale_names(plan::BRM._RKStructuralPlan)
    scales = Symbol[]
    for response in plan.responses
        d = response.discrimination
        d isa Symbol || continue
        any(p -> p.name === d, plan.predictors) || continue
        d in scales || push!(scales, d)
    end
    scales
end

_rk_response_wants_extras(response::BRM._RKLikelihoodSpec) =
    response.discrimination !== nothing ||
    !isempty(response.threshold_columns) ||
    response.threshold_coefs !== nothing

function _rk_patch_ordinal_response(lowered::LikelihoodSpec,
        planned::BRM._RKLikelihoodSpec)
    LikelihoodSpec(lowered.family, lowered.link, lowered.response,
        lowered.predictor, lowered.scale, lowered.weights, lowered.evidence,
        lowered.label, lowered.trials, lowered.range;
        n_levels = lowered.n_levels, thresholds = lowered.thresholds,
        extra_predictors = lowered.extra_predictors,
        count_columns = lowered.count_columns,
        ordinal_structure = lowered.ordinal_structure,
        discrimination = planned.discrimination,
        threshold_columns = planned.threshold_columns,
        threshold_coefs = planned.threshold_coefs)
end

function _rk_patch_scale_predictor!(predictors::Vector{PredictorSpec},
        priors::Vector{PopulationPrior}, levelmaps::Vector{LevelMap},
        plan::BRM._RKStructuralPlan, sname::Symbol)
    any(p -> p.name === sname, predictors) &&
        error("RK backend: internal: scale predictor `$sname` already " *
              "lowered (a modeled scale feeds no response slot)")
    spec = only(p for p in plan.predictors if p.name === sname)
    spec.link === :log ||
        error("RK backend: internal: scale predictor `$sname` has link " *
              "`$(spec.link)` (the planner gates `log`)")
    terms = map(spec.terms) do term
        kind = get(_RK_PPL_TERM, term.kind, nothing)
        kind === nothing &&
            error("RK backend: internal: scale predictor `$sname` term " *
                  "kind `$(term.kind)` (the planner gates population terms)")
        # Factor sizing lives in the LevelMap; terms take no options.
        TermSpec(kind, term.columns, NamedTuple(), term.addressee, term.label)
    end
    push!(predictors, PredictorSpec(spec.name, LogLink, terms, spec.label))
    for prior in plan.population_priors
        prior.predictor === sname || continue
        push!(priors, PopulationPrior(prior.predictor, prior.addressee,
            prior.location, prior.scale))
    end
    for term in spec.terms
        term.kind === :factor || continue
        col = only(term.columns)
        K = length(BRM._rk_grouping_levels(plan.columns[col]))
        push!(levelmaps, LevelMap(sname, col, [], :levels,
            _rk_ppl_levelsubset(term.options, K)))
    end
    nothing
end

function _rk_patch_threshold_coefs!(vectors::Vector{VectorParameter},
        plan::BRM._RKStructuralPlan, response::BRM._RKLikelihoodSpec)
    response.threshold_coefs === nothing && return nothing
    spec = only(v for v in plan.vector_parameters
        if v.name === response.threshold_coefs)
    spec.family === :vector_normal ||
        error("RK backend: internal: threshold coefs `$(spec.name)` " *
              "family `$(spec.family)` (the planner gates `:vector_normal`)")
    any(p -> p.name === spec.name, vectors) &&
        error("RK backend: internal: threshold coefs `$(spec.name)` " *
              "already lowered")
    args = NamedTuple(
        Symbol(:arg, i) => value for (i, value) in enumerate(spec.args))
    push!(vectors, VectorParameter(
        spec.name, spec.family, args, spec.size, spec.label))
    nothing
end

function _rk_patch_ordinal_extras(unbound::StructuralPlan,
        plan::BRM._RKStructuralPlan)
    scales = _rk_ordinal_scale_names(plan)
    any(_rk_response_wants_extras, plan.responses) || begin
        isempty(scales) ||
            error("RK backend: internal: scales without extras")
        return unbound
    end
    by_response = Dict{Symbol,BRM._RKLikelihoodSpec}(
        spec.response => spec for spec in plan.responses)
    responses = map(unbound.responses) do lowered
        planned = get(by_response, lowered.response, nothing)
        planned === nothing &&
            error("RK backend: internal: lowered response " *
                  "`$(lowered.response)` matches no planned response")
        _rk_response_wants_extras(planned) || return lowered
        _rk_patch_ordinal_response(lowered, planned)
    end
    predictors = copy(unbound.predictors)
    priors = copy(unbound.population_priors)
    levelmaps = copy(unbound.levelmaps)
    for sname in scales
        _rk_patch_scale_predictor!(predictors, priors, levelmaps, plan, sname)
    end
    vectors = copy(unbound.vector_parameters)
    for spec in plan.responses
        _rk_patch_threshold_coefs!(vectors, plan, spec)
    end
    StructuralPlan(responses, predictors, priors, unbound.parameters,
        unbound.assignments, unbound.columns, unbound.n_obs;
        roles = unbound.roles, derived = unbound.derived,
        levelmaps = levelmaps, plate_parameters = unbound.plate_parameters,
        scans = unbound.scans, ranef_buckets = unbound.ranef_buckets,
        vector_parameters = vectors, spline_bases = unbound.spline_bases,
        spline_vectors = unbound.spline_vectors,
        hsgp_bases = unbound.hsgp_bases,
        kernel_plates = unbound.kernel_plates)
end

# Evaluate the emitted submodel defs through `@rkppl` in a FRESH module
# per lowering. Defs are canonical lattice-named (same name means same
# body across all programs), so a shared module would be safe too; the
# fresh module is retained as evaluation hygiene. The macrocall `Expr`
# is exactly the parser's shape for `@rkppl sm(args...) = begin ... end`.
function _rk_emit_module(emitted::BRM._RKEmittedProgram)
    mod = Module(gensym(:RKEmittedModels))
    Core.eval(mod, :(using ReactiveKernelsPPL))
    for d in emitted.defs
        Core.eval(mod, Expr(:macrocall, Symbol("@rkppl"),
            LineNumberNode(0), d))
    end
    mod
end

function _rk_translated_plan(plan::BRM._RKStructuralPlan)
    emitted = BRM._rk_emit_ast(plan)
    unbound = lower_rkppl(emitted.main,
        Tuple(sort!(collect(keys(plan.columns)))); mod=_rk_emit_module(emitted))
    bind_data(_rk_patch_ordinal_extras(unbound, plan), plan.columns)
end

# Kernel plans additionally bind the plate dims (subjects/timepoints) the
# `subjects=...` key names; the counts live on the kernel spec.
function _rk_translated_plan(plan::BRM._RKKernelPlan)
    emitted = BRM._rk_emit_ast(plan)
    unbound = lower_rkppl(emitted.main,
        Tuple(sort!(collect(keys(plan.columns)))); mod=_rk_emit_module(emitted))
    bind_data(unbound, plan.columns; dims=BRM._rk_kernel_bind_dims(plan.kernel))
end

# The executable `model` of an `RKBRMI` is the thin-layer `(; spec, layout)`
# pair: the `KernelSpec` callable after `prepare`, plus its `LayoutTable`.
function BRM._brm_rk_model(plan::_RK_PLAN_TYPES)
    build_kernel(_rk_translated_plan(plan))
end

function BRM.RKBRMI(brmi::BRM.BRMI)
    plan = BRM._brm_rk_plan(brmi)
    BRM.RKBRMI(brmi, plan, BRM._brm_rk_model(plan))
end

# LogDensityProblems shim over the thin-layer sampler query: the packed
# unconstrained coordinates are the sampler space, so no transform sits
# between the sampler and the kernel. The boundary plan re-derives from
# the emission route (pure lowering, no kernel compile) because
# `prepare_sampler` derives its have/bound boundary from it; `model`
# stays exactly `build_kernel` output.
struct RKLogDensityProblem{Q<:SamplerQuery}
    query::Q
end

function BRM.rk_logdensity_problem(backend::BRM.RKBRMI;
        ad_backend,
        u0=zeros(Float64, backend.model.layout.total))
    translated = _rk_translated_plan(backend.plan)
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
