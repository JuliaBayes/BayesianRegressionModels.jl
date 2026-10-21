module BayesianRegressionModelsReactiveKernelsExt

using BayesianRegressionModels
using LogDensityProblems
using ReactiveKernels
using ReactiveKernelsPPL

const BRM = BayesianRegressionModels

include("rk_statistical_gp.jl")

# Stan builtins a kernel cell calls, bound to their Julia equivalents. Stan's
# conversions between vectors and one-dimensional arrays flatten column-major,
# as `vec` does. Its ODE solvers run RK's native bridges, which return the
# solution with one row per output time and one column per state, so
# `sol[i, k]` and `sol[:, k]` read as in Stan. The untolerated forms take
# Stan's defaults. Each bridge needs its solver package loaded with SciMLBase
# and SciMLSensitivity (reactivekernels-use §4g).
const _STAN_BUILTIN = BRM.StanBlocks.stan.builtin
BRM._rk_kernel_stan_builtin(::typeof(_STAN_BUILTIN.to_vector)) = vec
BRM._rk_kernel_stan_builtin(::typeof(_STAN_BUILTIN.to_array_1d)) = vec
BRM._rk_kernel_stan_builtin(::typeof(_STAN_BUILTIN.ode_rk45_tol)) =
    ReactiveKernels.rk_ode_rk45_tol
BRM._rk_kernel_stan_builtin(::typeof(_STAN_BUILTIN.ode_bdf_tol)) =
    ReactiveKernels.rk_ode_bdf_tol
BRM._rk_kernel_stan_builtin(::typeof(_STAN_BUILTIN.ode_rk45)) = _rk_ode_rk45
BRM._rk_kernel_stan_builtin(::typeof(_STAN_BUILTIN.ode_bdf)) = _rk_ode_bdf
BRM._rk_kernel_function_arguments(::Union{typeof(_STAN_BUILTIN.ode_rk45),
    typeof(_STAN_BUILTIN.ode_rk45_tol), typeof(_STAN_BUILTIN.ode_bdf),
    typeof(_STAN_BUILTIN.ode_bdf_tol)}) = (1,)
_rk_ode_rk45(f::F, y0, t0, ts, args::Vararg{Any,N}) where {F,N} =
    ReactiveKernels.rk_ode_rk45_tol(f, y0, t0, ts, 1e-6, 1e-6, 10^6, args...)
_rk_ode_bdf(f::F, y0, t0, ts, args::Vararg{Any,N}) where {F,N} =
    ReactiveKernels.rk_ode_bdf_tol(f, y0, t0, ts, 1e-6, 1e-6, 10^6, args...)

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

# A translated program: its emitted source and the module its definitions
# are evaluated in. One program serves every binding of it.
struct _RKProgram
    emitted::BRM._RKEmittedProgram
    mod::Module
end
_RKProgram(emitted::BRM._RKEmittedProgram) = _RKProgram(emitted, _rk_emit_module(emitted))

# Lowering reads the data values' shapes (rkppl-use §1), so each binding
# lowers; the definitions are evaluated once per program.
function _rk_lower_program(program::_RKProgram, plan::_RK_PLAN_TYPES)
    columns = BRM._rk_source_data_columns(plan, program.emitted)
    unbound = lower_rkppl(program.emitted.main,
        columns; mod=program.mod,
        conditioned=BRM._rk_observed_names(plan))
    unbound, columns
end
function _rk_bind_program(program::_RKProgram, plan::_RK_PLAN_TYPES)
    unbound, columns = _rk_lower_program(program, plan)
    bind_data(unbound, columns)
end

_rk_translate_from_emitted(plan::_RK_PLAN_TYPES, emitted::BRM._RKEmittedProgram) =
    _rk_bind_program(_RKProgram(emitted), plan)

function _rk_translated_plan(plan::_RK_PLAN_TYPES)
    _rk_translate_from_emitted(plan, BRM._rk_emit_ast(plan))
end

# A backend's program, translated once and shared with its replays.
_rk_program(backend::BRM.RKBRMI) =
    BRM._rk_once!(() -> _RKProgram(BRM._rk_emit_ast(backend.plan)), backend.program)
# A backend's binding of that program to its own data, bound once. Every
# query of the backend reads it.
_rk_translated(backend::BRM.RKBRMI) =
    BRM._rk_once!(() -> _rk_bind_program(_rk_program(backend), backend.plan),
        backend.bound)
BRM._rk_retained_program(backend::BRM.RKBRMI) = _rk_program(backend).emitted

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
    brmi = BRM._brm_resolve_optional_priors(brmi)
    plan = BRM._brm_rk_plan(brmi; held_out)
    program = _RKProgram(BRM._rk_emit_ast(plan))
    bound = _rk_bind_program(program, plan)
    BRM.RKBRMI(brmi, plan, build_kernel(bound);
        program=BRM._RKOnce(program), bound=BRM._RKOnce(bound))
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
    translated = _rk_translated(backend)
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

BRM.rk_bound_plan(backend::BRM.RKBRMI) = _rk_translated(backend)

BRM.rk_bound_data(backend::BRM.RKBRMI) = _rk_bound_columns(BRM.rk_bound_plan(backend).columns)
function _rk_bound_columns(columns)
    names = Tuple(keys(columns))
    NamedTuple{names}(Tuple(columns[name] for name in names))
end

# A design binder's lowering for one kind of design: a replay's, which every
# later design of that kind binds into (lowering reads only the data's names,
# numbers and arrays per index, rkppl-use §1).
BRM._rk_design_lowering(replay::BRM.RKBRMI) =
    first(_rk_lower_program(_rk_program(replay), replay.plan))
BRM._rk_bind_design(lowering, columns) = _rk_bound_columns(bind_data(lowering, columns).columns)

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
BRM._rk_bound_coordinate_names(backend::BRM.RKBRMI) =
    coordinate_names(assign_layout(_rk_translated(backend)))
# Each packed coordinate's constrained lower bound under a `:floored`
# transform (`lo + exp(u)`); `NaN` for other transforms or an unresolved bound.
function BRM._rk_layout_coordinate_floors(backend::BRM.RKBRMI)
    layout = backend.model.layout
    floors = fill(NaN, layout.total)
    for entry in layout.entries
        entry.transform === :floored && entry.lo isa Real &&
            (floors[entry.offset:(entry.offset + entry.size - 1)] .= entry.lo)
    end
    floors
end
BRM._rk_constrained_values(backend::BRM.RKBRMI, u::AbstractVector) =
    constrain(backend.model.layout, u)
BRM._rk_unconstrained_point(backend::BRM.RKBRMI, values::NamedTuple) =
    unconstrain(backend.model.layout, values)

# NLME evaluation (src/nlme_view.jl): the prepared `:pointwise` query and its
# evaluation at a packed unconstrained point, and the sampler query whose
# reverse sweep also retains those pointwise densities (reverse-mode Enzyme
# only; RK refuses other backends when preparing it).
BRM._rk_pointwise_query(backend::BRM.RKBRMI) =
    prepare_query(backend.model, _rk_translated(backend), :pointwise)
BRM._rk_pointwise_values(query, u::AbstractVector) = Base.invokelatest(query, u)
BRM._rk_nlme_cell_query(backend::BRM.RKBRMI, observations::Tuple) =
    prepare_cell_query(backend.model, _rk_translated(backend), observations)
BRM._rk_nlme_cell_value(query::CellQuery, point::BRM._BRMNLMEPoint, row::Int) =
    query(point, row)
BRM._rk_retained_pointwise_sampler(backend::BRM.RKBRMI; ad_backend,
        u0=zeros(Float64, backend.model.layout.total)) =
    prepare_sampler(backend.model, _rk_translated(backend), u0;
        backend=ad_backend, retain=(:pointwise,))
function BRM._rk_value_gradient_and_pointwise!(query::SamplerQuery,
        gradient::Vector{Float64}, u::Vector{Float64})
    value, gradient, retained = sampler_value_gradient_and_retained!(query, gradient, u)
    value, gradient, retained.pointwise
end

# The prior and its coordinate Jacobian are separate RK graph nodes. Preparing
# them independently of the likelihood lets NLME evaluate the population prior
# once per objective, without running any subject observation cells. Prepare
# them on first use so a conditional-only NLME fit keeps its construction cost.
function BRM._rk_nlme_population_prior(backend::BRM.RKBRMI,
        unit::Vector{Float64}, ad_backend)
    (; backend, unit, ad_backend, cached=Ref{Any}(nothing), lock=ReentrantLock())
end

function _nlme_population_prior_bundle(state)
    lock(state.lock) do
        state.cached[] === nothing || return state.cached[]
        backend, unit, ad_backend = state.backend, state.unit, state.ad_backend
        plan = _rk_translated(backend)
        built = backend.model
        value = prepare_query(built, plan, :prior)
        jacobian = prepare_query(built, plan, :log_jacobian)
        gradient = prepare_query_ad(prepare_ad, built, plan, :prior, ad_backend, unit)
        jacobian_gradient = prepare_query_ad(prepare_ad, built, plan,
            :log_jacobian, ad_backend, unit)
        state.cached[] = (; value, jacobian, gradient, jacobian_gradient)
    end
end

function BRM._rk_nlme_population_prior_value(state, u::Vector{Float64})
    bundle = _nlme_population_prior_bundle(state)
    Base.invokelatest(bundle.value, u) + Base.invokelatest(bundle.jacobian, u)
end

function BRM._rk_nlme_population_prior_value_gradient(state, u::Vector{Float64})
    bundle = _nlme_population_prior_bundle(state)
    value, gradient = ad_value_and_gradient(bundle.gradient, u)
    jacobian, jacobian_gradient = ad_value_and_gradient(bundle.jacobian_gradient, u)
    value + jacobian, gradient + jacobian_gradient
end

# The Gaussian NLME view uses one primal observations query and one set of
# colored η directions. Direction k moves margin k for every subject; their
# responses are conditionally independent, so each output sees only its own
# subject's component of that direction.
BRM._rk_observation_query(backend::BRM.RKBRMI) =
    prepare_query(backend.model, _rk_translated(backend), :observations)
BRM._rk_observation_values(query, u::AbstractVector) = Base.invokelatest(query, u)
function BRM._rk_observation_pushforward(backend::BRM.RKBRMI,
        view::BRM.BRMNLMEView, unit::Vector{Float64}, ad_backend)
    neta = maximum(length, view.subject_coordinates)
    directions = ntuple(neta) do k
        direction = zeros(Float64, length(unit))
        for block in view.subject_coordinates
            k <= length(block) && (direction[block[k]] = 1.0)
        end
        direction
    end
    query = prepare_query_ad(prepare_ad_pushforward, backend.model,
        _rk_translated(backend), :observations, ad_backend,
        directions, unit)
    (; query, directions)
end
BRM._rk_observation_value_and_pushforward(bundle, u::AbstractVector) =
    ad_value_and_pushforward(bundle.query, bundle.directions, u)

end
