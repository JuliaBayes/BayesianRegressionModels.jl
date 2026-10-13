# One NLME model's full target, conditioned on the fixed coordinates of one
# checked joint start: only the standardized subject draws move. The compiled
# Stan model and the RK backend evaluate the same conditional density in their
# own coordinate frames.

"""
    BRMNLMEConditionalProblem
    brm_nlme_conditional_problem(map::BRMNLMEJointMap, start;
        backend=map.sb, ad_backend)

Condition one backend's full target on the fixed coordinates of `start`, a
value returned by [`brm_nlme_joint_start`](@ref) for the same map. The
returned problem implements the LogDensityProblems interface over the free
coordinates only: the standardized subject draws, including one block per
nested unit. Population coefficients, scalar parameters, shared latent
coordinates and the random-effect scales and correlation factors stay at
`start`.

`backend` selects the evaluated target and its coordinate frame:
- `map.sb` (default): `map.stan_problem`, the exact compiled Stan model, in
  Stan unconstrained order. No `ad_backend` is taken.
- `map.rk`: `rk_logdensity_problem(map.rk; ad_backend)`, the ReactiveKernels
  program of the same BRMI, in RK packed order. `ad_backend` is a
  DifferentiationInterface backend such as reverse-mode `AutoEnzyme`.

The log density at a free point `x` is the target's log density at the start
point with the free coordinates replaced by `x`; it differs from the
conditional posterior of the subject draws only by a constant. The gradient,
and the Hessian when the target provides one, are the target's restricted to
the free coordinates. Each evaluation builds a fresh full point. `deepcopy`
gives an RK copy its own prepared automatic-differentiation workspace and a
Stan copy its own native model instance. WarmupHMC's multi-chain scalar-density
entry copies the problem once per chain.

Fields:
- `map`, `backend` — the joint map and the selected backend.
- `target` — the evaluated full LogDensityProblems target.
- `point` — the complete start point in the target's frame.
- `free`, `fixed` — the target's indices of the free and fixed coordinates.
- `names` — the target's names of the free coordinates, in problem order.
- `subjects`, `margins` — for each free coordinate, its subject label and the
  view's `(; block, predictor, coefficient, level)` margin record.
- `position` — the initial free point, `point[free]`.

Construction checks that `start` has this map's dimension, names and
fixed/free partition, that its recorded values equal `start.stan`, that the
point passes the physical RK/Stan coordinate check, and that the target's log
density there is finite. A map without subject draws, or without fixed
coordinates, is refused. No caller input is mutated. Use
[`brm_nlme_conditional_point`](@ref) for complete points in the target's
frame and `brm_nlme_joint_values(problem, x)` for the NLME values of a free
point; sample the unconditioned model with `map.stan_problem` and `start.stan`
instead.
"""
struct BRMNLMEConditionalProblem{M<:BRMNLMEJointMap,B,P}
    map::M
    backend::B
    target::P
    point::Vector{Float64}
    free::Vector{Int}
    fixed::Vector{Int}
    names::Vector{String}
    subjects::Vector{Any}
    margins::Vector{NamedTuple}
    position::Vector{Float64}
end

# A conditional RK target owns a prepared AD workspace. StanProblem instead
# wraps a BridgeStan.StanModel with a finalizer-managed native pointer: Julia's
# structural deepcopy would duplicate that pointer without constructing a new
# native model. Reopen the already compiled library and copy the wrapper using
# the replacement model, preserving its log-density options.
_brm_nlme_copy_conditional_target(::RKBRMI, target, stack) =
    Base.deepcopy_internal(target, stack)
function _brm_nlme_copy_conditional_target(::SBBRMI, target, stack)
    model = target.model
    copied = BridgeStan.StanModel(Libdl.dlpath(model.lib), model.data,
        model.seed; warn=false)
    stack[model] = copied
    Base.deepcopy_internal(target, stack)
end

function Base.deepcopy_internal(problem::BRMNLMEConditionalProblem, stack::IdDict)
    haskey(stack, problem) && return stack[problem]
    target = _brm_nlme_copy_conditional_target(problem.backend, problem.target, stack)
    # The map and backend are read-only provenance. Their compiled program and
    # coordinate records are shared; the evaluation target and point buffers
    # belong to this copy alone.
    copied = BRMNLMEConditionalProblem(problem.map, problem.backend, target,
        Base.deepcopy_internal(problem.point, stack),
        Base.deepcopy_internal(problem.free, stack),
        Base.deepcopy_internal(problem.fixed, stack),
        Base.deepcopy_internal(problem.names, stack),
        Base.deepcopy_internal(problem.subjects, stack),
        Base.deepcopy_internal(problem.margins, stack),
        Base.deepcopy_internal(problem.position, stack))
    stack[problem] = copied
    copied
end

# A start's own fixed/free records are in Stan order: subject draws are free
# and every other coordinate is fixed.
function _brm_nlme_conditional_partition(map::BRMNLMEJointMap)
    role = map.view.role
    free = Int[]
    fixed = Int[]
    for (j, k) in enumerate(map.transport.permutation)
        push!(role[k] === :subject_effect ? free : fixed, j)
    end
    free, fixed
end

# Each backend's frame of one Stan start point: its target, the point in its
# coordinates, the RK coordinate behind each of its coordinates, and its names.
function _brm_nlme_conditional_frame(sb::SBBRMI, map::BRMNLMEJointMap, stan)
    sb === map.sb || _brm_nlme_error(
        "brm_nlme_conditional_problem: backend is not this map's SBBRMI")
    (; target=map.stan_problem, point=stan, rk=map.transport.permutation,
       names=map.transport.stan_names)
end

function _brm_nlme_conditional_frame(rk::RKBRMI, map::BRMNLMEJointMap, stan;
        ad_backend)
    rk === map.rk || _brm_nlme_error(
        "brm_nlme_conditional_problem: backend is not this map's RKBRMI")
    point = brm_stan_point_to_rk(map.transport, stan)
    (; target=rk_logdensity_problem(rk; ad_backend, u0=copy(point)), point,
       rk=collect(eachindex(point)), names=String.(map.view.coordinates))
end

function brm_nlme_conditional_problem(map::BRMNLMEJointMap, start::NamedTuple;
        backend=map.sb, kwargs...)
    t, view = map.transport, map.view
    n = length(t.stan_names)
    stan = Vector{Float64}(start.stan)
    length(stan) == n || throw(DimensionMismatch(
        "brm_nlme_conditional_problem: start.stan has $(length(stan)) " *
        "coordinates for $n compiled parameters"))
    all(isfinite, stan) || _brm_nlme_error(
        "brm_nlme_conditional_problem: start.stan must be finite")
    start_free, start_fixed = _brm_nlme_conditional_partition(map)
    isempty(start_free) && _brm_nlme_error(
        "brm_nlme_conditional_problem: the model has no standardized subject " *
        "draws to sample")
    isempty(start_fixed) && _brm_nlme_error(
        "brm_nlme_conditional_problem: the model has no fixed coordinates; " *
        "sample map.stan_problem from start.stan instead")
    for (field, expected) in ((:fixed, start_fixed), (:free, start_free))
        coords = getproperty(start, field)
        collect(coords.indices) == expected || _brm_nlme_error(
            "brm_nlme_conditional_problem: start.$field.indices do not match " *
            "this map's fixed/free partition")
        collect(coords.names) == t.stan_names[expected] || _brm_nlme_error(
            "brm_nlme_conditional_problem: start.$field.names do not match " *
            "the compiled parameter names")
        collect(coords.values) == stan[expected] || _brm_nlme_error(
            "brm_nlme_conditional_problem: start.$field.values disagree " *
            "with start.stan")
    end
    brm_check_coordinate_transport(t, map.rk, map.stan_model,
        brm_stan_point_to_rk(t, stan))
    frame = _brm_nlme_conditional_frame(backend, map, stan; kwargs...)
    LogDensityProblems.dimension(frame.target) == n || _brm_nlme_error(
        "brm_nlme_conditional_problem: the target no longer has the map's " *
        "$n unconstrained coordinates")
    value = LogDensityProblems.logdensity(frame.target, frame.point)
    isfinite(value) || _brm_nlme_error(
        "brm_nlme_conditional_problem: the log density at the start is $value")
    free = findall(k -> view.role[k] === :subject_effect, frame.rk)
    fixed = findall(k -> view.role[k] !== :subject_effect, frame.rk)
    subjects = Any[]
    margins = NamedTuple[]
    for k in frame.rk[free]
        i = view.subject[k]
        push!(subjects, view.levels[i])
        push!(margins, view.subject_margins[i][
            only(findall(==(k), view.subject_coordinates[i]))])
    end
    BRMNLMEConditionalProblem(map, backend, frame.target, frame.point, free,
        fixed, frame.names[free], subjects, margins, frame.point[free])
end

"""
    brm_nlme_conditional_point(problem::BRMNLMEConditionalProblem, x)

Return the complete point of `problem.target` for the free point `x`:
`problem.point` with `problem.free` replaced by `x`, in the target's frame
(Stan unconstrained order for `map.sb`, RK packed order for `map.rk`). A
matrix whose columns are free draws, as WarmupHMC returns them, gives one
complete point per column. The result is fresh; `x` is not mutated.
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

_brm_nlme_stan_point(::SBBRMI, map::BRMNLMEJointMap, u) = u
_brm_nlme_stan_point(::RKBRMI, map::BRMNLMEJointMap, u) =
    brm_rk_point_to_stan(map.transport, u)

"""
    brm_nlme_joint_values(problem::BRMNLMEConditionalProblem, x)

Read the free point `x` of a conditional problem back as
`(; theta, sigma, omega, eta, subjects, shared)`, exactly as
`brm_nlme_joint_values(problem.map, stan_u)` reads the corresponding complete
Stan point, whichever backend `problem` evaluates.
"""
brm_nlme_joint_values(problem::BRMNLMEConditionalProblem, x::AbstractVector) =
    brm_nlme_joint_values(problem.map, _brm_nlme_stan_point(problem.backend,
        problem.map, brm_nlme_conditional_point(problem, x)))

LogDensityProblems.capabilities(
        ::Type{<:BRMNLMEConditionalProblem{M,B,P}}) where {M,B,P} =
    LogDensityProblems.capabilities(P)

LogDensityProblems.dimension(problem::BRMNLMEConditionalProblem) =
    length(problem.free)

LogDensityProblems.logdensity(problem::BRMNLMEConditionalProblem, x) =
    LogDensityProblems.logdensity(problem.target,
        brm_nlme_conditional_point(problem, x))

function LogDensityProblems.logdensity_and_gradient(
        problem::BRMNLMEConditionalProblem, x)
    value, gradient = LogDensityProblems.logdensity_and_gradient(
        problem.target, brm_nlme_conditional_point(problem, x))
    value, gradient[problem.free]
end

function LogDensityProblems.logdensity_gradient_and_hessian(
        problem::BRMNLMEConditionalProblem, x)
    value, gradient, hessian = LogDensityProblems.logdensity_gradient_and_hessian(
        problem.target, brm_nlme_conditional_point(problem, x))
    value, gradient[problem.free], hessian[problem.free, problem.free]
end
