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
# thin-layer suite. Kernel plans ride the same route. Varying-source
# innovations likewise enter the AST as native-read arguments, so
# `lower_rkppl` sees their names as data; after lowering the extension
# removes exactly those plate slices, rebuilds the plate, and appends
# the typed `VectorParameter` ports. Final binding never treats them
# as raw columns.
const _RK_PLAN_TYPES =
    Union{BRM._RKStructuralPlan,BRM._RKKernelPlan,BRM._RKVaryingSourcePlan}

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

# BRM plan family (CamelCase) to thin-layer `POPULATION_FAMILIES` token.
const _RK_THIN_POPULATION_FAMILIES = Dict{Symbol,Symbol}(
    :Normal => :normal, :Cauchy => :cauchy, :Laplace => :laplace,
    :Logistic => :logistic, :StudentT => :student_t, :Flat => :flat)

# Plan prior to thin-layer `PopulationPrior`: the family crosses by
# table, the arg shape by arity — a 3-tuple is `(nu, mu, sigma)` and
# rotates to `(mu, sigma, nu)`; an empty tuple is `Flat` (whose
# location/scale/nu the thin layer ignores); anything else passes
# through as `(location, scale)`.
function _rk_thin_population_prior(prior::BRM._RKPopulationPrior)
    family = _RK_THIN_POPULATION_FAMILIES[prior.family]
    args = prior.args
    length(args) == 3 &&
        return PopulationPrior(prior.predictor, prior.addressee, family,
            args[2], args[3], args[1])
    isempty(args) &&
        return PopulationPrior(prior.predictor, prior.addressee, family,
            0.0, 1.0, NaN)
    PopulationPrior(prior.predictor, prior.addressee, family,
        args[1], args[2])
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
        # Main-predictor priors cross via the AST, never here.
        push!(priors, _rk_thin_population_prior(prior))
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

# Varying-source innovation vectors enter the AST as native-read
# arguments, so they must be declared as data for the first lowering.
# After lowering, convert their planned `:vector_normal` specs to typed
# IR ports and remove their inferred plate slices. The plan deliberately
# carries no data columns for these names: they stay latent parameters
# at final `bind_data`.
function _rk_patch_varyingsource_vector_ports(unbound::StructuralPlan,
        plan::BRM._RKVaryingSourcePlan)
    specs = [spec for spec in plan.vector_parameters
             if spec.family === :vector_normal]
    isempty(specs) && return unbound
    ports = Set(spec.name for spec in specs)
    kp = only(unbound.kernel_plates)
    removed = count(x -> x[2] in ports, kp.slices)
    removed == length(ports) || error(
        "RK backend: internal: varying-source innovation ports " *
        "expected slices for `$(sort!(collect(ports)))`, got $removed")
    plate = KernelPlate(kp.result, kp.subjects, kp.timepoints,
        [slice for slice in kp.slices if !(slice[2] in ports)],
        kp.assignments, kp.obs, kp.collected, kp.label, kp.lp_args,
        kp.schedules)
    vectors = copy(unbound.vector_parameters)
    for spec in specs
        any(p -> p.name === spec.name, vectors) && error(
            "RK backend: internal: innovation port `$(spec.name)` " *
            "already lowered")
        args = NamedTuple(
            Symbol(:arg, i) => value for (i, value) in enumerate(spec.args))
        push!(vectors, VectorParameter(
            spec.name, spec.family, args, spec.size, spec.label))
    end
    StructuralPlan(unbound.responses, unbound.predictors,
        unbound.population_priors, unbound.parameters, unbound.assignments,
        unbound.columns, unbound.n_obs; roles = unbound.roles,
        derived = unbound.derived, levelmaps = unbound.levelmaps,
        plate_parameters = unbound.plate_parameters, scans = unbound.scans,
        dar_paths = unbound.dar_paths,
        varying_draws = unbound.varying_draws,
        varying_slices = unbound.varying_slices,
        vector_parameters = vectors, spline_bases = unbound.spline_bases,
        spline_vectors = unbound.spline_vectors,
        hsgp_bases = unbound.hsgp_bases,
        kernel_plates = [plate], r2d2_priors = unbound.r2d2_priors,
        horseshoe_priors = unbound.horseshoe_priors,
        matrices = unbound.matrices)
end

# `mi()` plan-level patch (no surface syntax in v1, decision 05aemvx
# P4): the AST lowers the ordinary response statement, and the planned
# `Jobs` column rides `mi_jobs` onto the thin-layer spec here — the
# same plan-level route as the ordinal extras. Plans without `mi()`
# responses pass through untouched.
function _rk_patch_mi_response(lowered::LikelihoodSpec,
        planned::BRM._RKLikelihoodSpec)
    LikelihoodSpec(lowered.family, lowered.link, lowered.response,
        lowered.predictor, lowered.scale, lowered.weights, lowered.evidence,
        lowered.label, lowered.trials, lowered.range;
        n_levels = lowered.n_levels, thresholds = lowered.thresholds,
        extra_predictors = lowered.extra_predictors,
        count_columns = lowered.count_columns,
        ordinal_structure = lowered.ordinal_structure,
        discrimination = lowered.discrimination,
        threshold_columns = lowered.threshold_columns,
        threshold_coefs = lowered.threshold_coefs,
        extra_responses = lowered.extra_responses,
        factor_scales = lowered.factor_scales,
        factor_corr = lowered.factor_corr,
        glm_alpha = lowered.glm_alpha, glm_beta = lowered.glm_beta,
        mi_jobs = planned.mi_jobs)
end

function _rk_patch_mi_jobs(unbound::StructuralPlan,
        plan::BRM._RKStructuralPlan)
    any(r -> r.mi_jobs !== nothing, plan.responses) || return unbound
    by_response = Dict{Symbol,BRM._RKLikelihoodSpec}(
        spec.response => spec for spec in plan.responses)
    responses = map(unbound.responses) do lowered
        planned = get(by_response, lowered.response, nothing)
        planned === nothing &&
            error("RK backend: internal: lowered response " *
                  "`$(lowered.response)` matches no planned response")
        planned.mi_jobs === nothing && return lowered
        _rk_patch_mi_response(lowered, planned)
    end
    StructuralPlan(responses, unbound.predictors, unbound.population_priors,
        unbound.parameters, unbound.assignments, unbound.columns,
        unbound.n_obs; roles = unbound.roles, derived = unbound.derived,
        levelmaps = unbound.levelmaps,
        plate_parameters = unbound.plate_parameters, scans = unbound.scans,
        dar_paths = unbound.dar_paths,
        varying_draws = unbound.varying_draws,
        varying_slices = unbound.varying_slices,
        vector_parameters = unbound.vector_parameters,
        spline_bases = unbound.spline_bases,
        spline_vectors = unbound.spline_vectors,
        hsgp_bases = unbound.hsgp_bases,
        kernel_plates = unbound.kernel_plates,
        r2d2_priors = unbound.r2d2_priors, matrices = unbound.matrices)
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
        scans = unbound.scans, dar_paths = unbound.dar_paths,
        varying_draws = unbound.varying_draws,
        varying_slices = unbound.varying_slices,
        vector_parameters = vectors, spline_bases = unbound.spline_bases,
        spline_vectors = unbound.spline_vectors,
        hsgp_bases = unbound.hsgp_bases,
        kernel_plates = unbound.kernel_plates,
        r2d2_priors = unbound.r2d2_priors,
        horseshoe_priors = unbound.horseshoe_priors,
        matrices = unbound.matrices)
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
    bind_data(_rk_patch_mi_jobs(
        _rk_patch_ordinal_extras(unbound, plan), plan), plan.columns)
end

# Kernel plans additionally bind the plate dims (subjects/timepoints) the
# `subjects=...` key names; the counts live on the kernel spec.
function _rk_translated_plan(plan::BRM._RKKernelPlan)
    emitted = BRM._rk_emit_ast(plan)
    unbound = lower_rkppl(emitted.main,
        Tuple(sort!(collect(keys(plan.columns)))); mod=_rk_emit_module(emitted))
    bind_data(unbound, plan.columns; dims=BRM._rk_kernel_bind_dims(plan.kernel))
end

# Varying-source twin plans: flat program (no defs lattice, no
# ordinal/MI patching — the plate carries its own observation); the
# subject count binds as dims like kernel plans.
function _rk_translated_plan(plan::BRM._RKVaryingSourcePlan)
    emitted = BRM._rk_emit_ast(plan)
    names = union(keys(plan.columns),
        (spec.name for spec in plan.vector_parameters
         if spec.family === :vector_normal))
    unbound = lower_rkppl(emitted.main,
        Tuple(sort!(collect(names))); mod=_rk_emit_module(emitted))
    bound = bind_data(_rk_patch_varyingsource_vector_ports(unbound, plan),
        plan.columns;
        dims=Dict(plan.spec.subject_count => plan.spec.n_subjects))
    bound
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
