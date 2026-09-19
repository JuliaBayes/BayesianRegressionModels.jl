module BayesianRegressionModelsReactiveKernelsExt

using BayesianRegressionModels
using LogDensityProblems
using ReactiveKernels
using ReactiveKernelsPPL

const BRM = BayesianRegressionModels

# Sole emission path: BRM-side plan (plain data, no RK types) → `@rkppl`
# AST (`BRM._rk_emit_ast`, total over admitted plans) → thin-layer
# `StructuralPlan` via `lower_rkppl` + `bind_data` (the same function the
# macro lowers through). Model execution and the sampler boundary both
# derive from this one lowering, so the boundary plan is definitionally
# the plan the model was built from — there is no parallel direct
# serializer to drift (the retired one did: factor term options and the
# preserved Binomial triple-3 are both rejected from hand-built plans yet
# accepted from the AST route). Kernel plans ride the same route; until
# the thin-layer KernelPlate reader lands, `lower_rkppl`/`build_kernel`
# fail closed with thin-layer attribution (leaf 14bv4nq).
const _RK_PLAN_TYPES = Union{BRM._RKStructuralPlan,BRM._RKKernelPlan}
function _rk_translated_plan(plan::BRM._RKStructuralPlan)
    ast = BRM._rk_emit_ast(plan)
    unbound = lower_rkppl(ast, Tuple(sort!(collect(keys(plan.columns)))))
    bind_data(unbound, plan.columns)
end

# Kernel plans additionally bind the plate dims (subjects/timepoints) the
# `subjects=...` key names; the counts live on the kernel spec.
function _rk_translated_plan(plan::BRM._RKKernelPlan)
    ast = BRM._rk_emit_ast(plan)
    unbound = lower_rkppl(ast, Tuple(sort!(collect(keys(plan.columns)))))
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
# the AST route (pure lowering, no kernel compile) because
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
