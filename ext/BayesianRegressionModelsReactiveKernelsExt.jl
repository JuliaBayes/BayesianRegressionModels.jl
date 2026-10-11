module BayesianRegressionModelsReactiveKernelsExt

using BayesianRegressionModels
using LogDensityProblems
using ReactiveKernels
using ReactiveKernelsPPL

const BRM = BayesianRegressionModels

include("rk_statistical_gp.jl")

# The emitted source is the executable contract. Every statistical prior and
# observation role lowers through the public RKPPL surface; binding supplies
# the data, with no ordinal or missing-response plan mutations.
const _RK_PLAN_TYPES = Union{BRM._RKStructuralPlan,BRM._RKKernelPlan,BRM._RKValuePlan,BRM._RKHeldOutPlan,BRM._RKUnconditionedPlan}

# Evaluate native functions and explicit `@kernel`/`@rkppl` definitions in a fresh module
# per lowering. Each build owns its definition namespace even when different
# programs use the same authored names. The macrocall `Expr`
# is exactly the parser's shape for `@rkppl sm(args...) = begin ... end`.
function _rk_emit_module(emitted::BRM._RKEmittedProgram)
    BRM._rk_validate_source_definitions(emitted)
    mod = Module(gensym(:RKEmittedModels))
    Core.eval(mod, :(using ReactiveKernelsPPL))
    Core.eval(mod, :(import ReactiveKernels))
    Core.eval(mod, :(import BayesianRegressionModels))
    # Callable bindings are named around exactly these imports.
    for (source, names) in pairs(BRM._RK_SOURCE_IMPORTS)
        Core.eval(mod, Expr(:import, Expr(:(:), Expr(:., source),
            (Expr(:., name) for name in names)...)))
    end
    for (name, value) in emitted.bindings
        Core.eval(mod, Expr(:const, Expr(:(=), name, QuoteNode(value))))
    end
    for d in emitted.defs
        if BRM._rk_source_definition(d).kind in (:function, :kernel)
            Core.eval(mod, d)
        else
            Core.eval(mod, Expr(:macrocall, Symbol("@rkppl"),
                LineNumberNode(0), d))
        end
    end
    mod
end

function _rk_translate_from_emitted(plan::_RK_PLAN_TYPES,
        emitted::BRM._RKEmittedProgram)
    columns = BRM._rk_source_data_columns(plan, emitted)
    unbound = lower_rkppl(emitted.main,
        columns; mod=_rk_emit_module(emitted),
        conditioned=BRM._rk_observed_names(plan))
    bind_data(unbound, columns)
end

function _rk_translated_plan(plan::_RK_PLAN_TYPES)
    _rk_translate_from_emitted(plan, BRM._rk_emit_ast(plan))
end

const _RK_STATISTICAL_MODEL_MODULE = Ref{Union{Nothing,Module}}(nothing)
const _RK_STATISTICAL_MODEL_LOCK = ReentrantLock()

function _rk_statistical_model_module()
    lock(_RK_STATISTICAL_MODEL_LOCK) do
        cached = _RK_STATISTICAL_MODEL_MODULE[]
        cached === nothing || return cached
        prep = BRM.StatisticalPreparation
        helpers = (:tps_basis, :cr_basis, :t2_basis, :hsgp_basis, :hsgp_periodic_basis,
            :hsgp_matern_sqrt_spd, :hsgp_sqrt_spd, :hsgp_grouped_sqrt_spd,
            :hsgp_periodic_sqrt_spd, :hsgp_periodic_grouped_sqrt_spd,
            :hsgp_rho_floors, :hsgp_periodic_rho_floor)
        bindings = Pair{Symbol,Any}[name => getproperty(prep, name) for name in helpers]
        mod = _rk_emit_module(BRM._RKEmittedProgram(collect(values(BRM._BRM_STATISTICAL_MODELS)),
            Expr(:block), bindings))
        _RK_STATISTICAL_MODEL_MODULE[] = mod
        mod
    end
end

BRM.rkppl_model(name::Symbol) = hasproperty(BRM._BRM_STATISTICAL_MODELS, name) ?
    getproperty(_rk_statistical_model_module(), name) :
    throw(ArgumentError("unknown BRM statistical model `$name`"))

"""
    rk_translate_artifact(artifact) -> bound `StructuralPlan`

Translate a v4 append artifact (`BRM.emit_rk_artifact` shape) through the
production route, using the same definition module and data bindings as
live `RKBRMI` builds. Fails closed on shape/version skew.
"""
function BRM.rk_translate_artifact(artifact)
    keys(artifact) == (:case_id, :ast, :defs, :bindings, :plan, :meta) || error(
        "RK artifact: not a v4 artifact (keys $(keys(artifact)))")
    artifact.meta.generator_version == BRM.rk_artifact_version() || error(
        "RK artifact: case `$(artifact.case_id)` has generator_version " *
        "$(artifact.meta.generator_version); this BRM translates " *
        "$(BRM.rk_artifact_version())")
    artifact.plan isa BRM._RK_ARTIFACT_PLAN_TYPES || error(
        "RK artifact: case `$(artifact.case_id)` carries a " *
        "$(typeof(artifact.plan)), not an RK plan")
    emitted = BRM._RKEmittedProgram(
        artifact.defs, artifact.ast, artifact.bindings)
    return _rk_translate_from_emitted(artifact.plan, emitted)
end

# The executable `model` of an `RKBRMI` is the thin-layer `(; spec, layout)`
# pair: the `KernelSpec` callable after `prepare`, plus its `LayoutTable`.
function BRM._brm_rk_model(plan::_RK_PLAN_TYPES)
    build_kernel(_rk_translated_plan(plan))
end

# Planning and building read the model as syntax; only wrapping the retained
# original `brmi` constructs a model-specific `RKBRMI` type.
Base.@nospecializeinfer function BRM.RKBRMI(@nospecialize(brmi::BRM.BRMI); held_out=())
    plan = BRM._brm_rk_plan(brmi; held_out)
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

# Cross-backend coordinate transport (src/coordinate_transport.jl).
BRM._rk_layout_coordinate_names(backend::BRM.RKBRMI) =
    coordinate_names(backend.model.layout)
# Each packed coordinate's layout transform, in coordinate order.
function BRM._rk_layout_coordinate_transforms(backend::BRM.RKBRMI)
    layout = backend.model.layout
    transforms = fill(:unassigned, layout.total)
    for entry in layout.entries
        transforms[entry.offset:(entry.offset + entry.size - 1)] .= entry.transform
    end
    any(==(:unassigned), transforms) && error(
        "brm_coordinate_transport: internal: the RK layout leaves coordinates " *
        "without an entry")
    transforms
end
BRM._rk_constrained_values(backend::BRM.RKBRMI, u::AbstractVector) =
    constrain(backend.model.layout, u)

# NLME evaluation (src/nlme_view.jl): the prepared `:pointwise` query and its
# evaluation at a packed unconstrained point, and the sampler query whose
# reverse sweep also retains those pointwise densities (reverse-mode Enzyme
# only; RK refuses other backends when preparing it).
BRM._rk_pointwise_query(backend::BRM.RKBRMI) =
    prepare_query(backend.model, _rk_translated_plan(backend.plan), :pointwise)
BRM._rk_pointwise_values(query, u::AbstractVector) = Base.invokelatest(query, u)
BRM._rk_retained_pointwise_sampler(backend::BRM.RKBRMI; ad_backend,
        u0=zeros(Float64, backend.model.layout.total)) =
    prepare_sampler(backend.model, _rk_translated_plan(backend.plan), u0;
        backend=ad_backend, retain=(:pointwise,))
function BRM._rk_value_gradient_and_pointwise!(query::SamplerQuery,
        gradient::Vector{Float64}, u::Vector{Float64})
    value, gradient, retained = sampler_value_gradient_and_retained!(query, gradient, u)
    value, gradient, retained.pointwise
end

end
