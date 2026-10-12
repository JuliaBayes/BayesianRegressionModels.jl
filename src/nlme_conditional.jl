# The same compiled Stan target, conditioned on the fixed NLME coordinates of
# one checked joint start: only the standardized subject draws move.

"""
    BRMNLMEConditionalProblem
    brm_nlme_conditional_problem(map::BRMNLMEJointMap, start)

Condition `map.stan_problem`, the exact compiled model of `map`, on the fixed
coordinates of `start`, a value returned by [`brm_nlme_joint_start`](@ref) for
the same map. The returned problem implements the LogDensityProblems interface
over the free coordinates only: the standardized subject draws, including one
block per nested unit, in full Stan unconstrained order. Population
coefficients, scalar parameters, shared latent coordinates and the
random-effect scales and correlation factors stay at `start.fixed.values`.

The log density at a free point `x` is the full model's log density at
`start.stan` with `start.free.indices` replaced by `x`; it differs from the
conditional posterior of the subject draws only by a constant. The gradient,
and the Hessian when the compiled problem provides one, are the full ones
restricted to the free coordinates. Every evaluation works on a fresh full
vector, so one problem can serve several threads.

Fields:
- `map` — the joint map; `map.stan_problem` is the evaluated target.
- `point` — the complete unconstrained Stan point of the start.
- `free`, `fixed` — full Stan indices of the free and fixed coordinates.
- `names` — Stan names of the free coordinates, in problem order.
- `subjects`, `margins` — for each free coordinate, its subject label and the
  view's `(; block, predictor, coefficient, level)` margin record.
- `position` — the initial free point, `point[free]`.

Construction checks that `start` has this map's dimension, names and
fixed/free partition, that its recorded values equal `start.stan`, that the
point passes the physical RK/Stan coordinate check, and that its log density
is finite. A map without subject draws, or without fixed coordinates, is
refused. No caller input is mutated. Use
[`brm_nlme_conditional_point`](@ref) to turn free draws back into full Stan
points; sample the unconditioned model with `map.stan_problem` and
`start.stan` instead.
"""
struct BRMNLMEConditionalProblem{M<:BRMNLMEJointMap}
    map::M
    point::Vector{Float64}
    free::Vector{Int}
    fixed::Vector{Int}
    names::Vector{String}
    subjects::Vector{Any}
    margins::Vector{NamedTuple}
    position::Vector{Float64}
end

# The partition is the map's, never the caller's: subject draws are free and
# every other coordinate is fixed.
function _brm_nlme_conditional_partition(map::BRMNLMEJointMap)
    role = map.view.role
    free = Int[]
    fixed = Int[]
    for (j, k) in enumerate(map.transport.permutation)
        push!(role[k] === :subject_effect ? free : fixed, j)
    end
    free, fixed
end

function brm_nlme_conditional_problem(map::BRMNLMEJointMap, start::NamedTuple)
    t, view = map.transport, map.view
    n = length(t.stan_names)
    point = Vector{Float64}(start.stan)
    length(point) == n || throw(DimensionMismatch(
        "brm_nlme_conditional_problem: start.stan has $(length(point)) " *
        "coordinates for $n compiled parameters"))
    LogDensityProblems.dimension(map.stan_problem) == n || _brm_nlme_error(
        "brm_nlme_conditional_problem: the compiled problem no longer has the " *
        "map's $n unconstrained coordinates")
    all(isfinite, point) || _brm_nlme_error(
        "brm_nlme_conditional_problem: start.stan must be finite")
    free, fixed = _brm_nlme_conditional_partition(map)
    isempty(free) && _brm_nlme_error(
        "brm_nlme_conditional_problem: the model has no standardized subject " *
        "draws to sample")
    isempty(fixed) && _brm_nlme_error(
        "brm_nlme_conditional_problem: the model has no fixed coordinates; " *
        "sample map.stan_problem from start.stan instead")
    for (field, expected) in ((:fixed, fixed), (:free, free))
        coords = getproperty(start, field)
        collect(coords.indices) == expected || _brm_nlme_error(
            "brm_nlme_conditional_problem: start.$field.indices do not match " *
            "this map's fixed/free partition")
        collect(coords.names) == t.stan_names[expected] || _brm_nlme_error(
            "brm_nlme_conditional_problem: start.$field.names do not match " *
            "the compiled parameter names")
        collect(coords.values) == point[expected] || _brm_nlme_error(
            "brm_nlme_conditional_problem: start.$field.values disagree " *
            "with start.stan")
    end
    brm_check_coordinate_transport(t, map.rk, map.stan_model,
        brm_stan_point_to_rk(t, point))
    value = LogDensityProblems.logdensity(map.stan_problem, point)
    isfinite(value) || _brm_nlme_error(
        "brm_nlme_conditional_problem: the log density at start.stan is $value")
    subjects = Any[]
    margins = NamedTuple[]
    for j in free
        k = t.permutation[j]
        i = view.subject[k]
        push!(subjects, view.levels[i])
        push!(margins, view.subject_margins[i][
            only(findall(==(k), view.subject_coordinates[i]))])
    end
    BRMNLMEConditionalProblem(map, point, free, fixed, t.stan_names[free],
        subjects, margins, point[free])
end

"""
    brm_nlme_conditional_point(problem::BRMNLMEConditionalProblem, x)

Return the complete unconstrained Stan point of the compiled model for the
free point `x`: `problem.point` with `problem.free` replaced by `x`. A matrix
whose columns are free draws, as WarmupHMC returns them, gives one complete
point per column. The result is fresh; `x` is not mutated. Read it with
[`brm_nlme_joint_values`](@ref) or `BridgeStan.param_constrain`.
"""
function brm_nlme_conditional_point(problem::BRMNLMEConditionalProblem,
        x::AbstractVector)
    length(x) == length(problem.free) || throw(DimensionMismatch(
        "brm_nlme_conditional_point: $(length(x)) free coordinates, expected " *
        "$(length(problem.free))"))
    u = copy(problem.point)
    u[problem.free] = x
    u
end

function brm_nlme_conditional_point(problem::BRMNLMEConditionalProblem,
        X::AbstractMatrix)
    size(X, 1) == length(problem.free) || throw(DimensionMismatch(
        "brm_nlme_conditional_point: $(size(X, 1)) free coordinates per draw, " *
        "expected $(length(problem.free))"))
    U = repeat(problem.point, 1, size(X, 2))
    U[problem.free, :] = X
    U
end

LogDensityProblems.capabilities(::Type{<:BRMNLMEConditionalProblem{M}}) where {M} =
    LogDensityProblems.capabilities(fieldtype(M, :stan_problem))

LogDensityProblems.dimension(problem::BRMNLMEConditionalProblem) =
    length(problem.free)

LogDensityProblems.logdensity(problem::BRMNLMEConditionalProblem, x) =
    LogDensityProblems.logdensity(problem.map.stan_problem,
        brm_nlme_conditional_point(problem, x))

function LogDensityProblems.logdensity_and_gradient(
        problem::BRMNLMEConditionalProblem, x)
    value, gradient = LogDensityProblems.logdensity_and_gradient(
        problem.map.stan_problem, brm_nlme_conditional_point(problem, x))
    value, gradient[problem.free]
end

function LogDensityProblems.logdensity_gradient_and_hessian(
        problem::BRMNLMEConditionalProblem, x)
    value, gradient, hessian = LogDensityProblems.logdensity_gradient_and_hessian(
        problem.map.stan_problem, brm_nlme_conditional_point(problem, x))
    value, gradient[problem.free], hessian[problem.free, problem.free]
end
