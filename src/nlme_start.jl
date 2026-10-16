# A complete point of the same model -> NLMEEstimation starting values.
# The inverse direction of nlme_joint_start.jl: the NLME partition and the RK
# record inventory remain the only sources of the coordinate layout.

# Names follow the model's NLME coordinates: packed RK names for θ and σ, and
# `<block>:<predictor>:<coefficient>` margins for η (prefixed by the grouping
# and its level in a nested model).
_brm_nlme_parameter_names(m::BRMNLMEModel) = (
    theta=string.(m.view.coordinates[m.theta]),
    sigma=string.(m.view.coordinates[m.sigma]),
    eta=_brm_nlme_ragged(m.view) ?
        [[string(b.block.group, "[", b.level, "]:", b.predictor, ":", b.coefficient)
            for b in margins] for margins in m.view.subject_margins] :
        [string(b.predictor, ":", b.coefficient) for b in m.view.block_margins])

# NLMEEstimation owns the ω packing and the population-prior measure, so its
# extension converts the natural covariance and checks the prior.
_brm_nlme_start_coordinates(m, θ, σ, Ω, u) = error(
    "brm_nlme_start: ω is NLMEEstimation's covariance coordinate; load " *
    "NLMEEstimation to activate BRM's NLMEEstimation extension")

"""
    brm_nlme_start(m::BRMNLMEModel, u::AbstractVector)
    brm_nlme_start(m::BRMNLMEModel, map::BRMNLMEJointMap, stan_u::AbstractVector)
    brm_nlme_start(m::BRMNLMEModel, inits::NamedTuple; base=m.unit)

Read one complete point of the *same* model as NLMEEstimation starting values
for `m`. `u` is a full RK packed unconstrained point of `m.backend`'s layout.
`stan_u` is a full unconstrained point of `map.stan_problem`; `map` must come
from the same BRMI instance and NLME partition as `m`, and its validated
semantic transport converts `stan_u` to RK coordinates first. `inits` are
initial values keyed by prior address, as [`brm_start`](@ref) takes them; the
point is `brm_start(m.backend, inits; base)`, whose default base is `m`'s unit
point (zero subject effects, and the fixed values of a fixed-shared model).
Requires NLMEEstimation to be loaded.

Returns a named tuple:

- `theta`, `sigma` — `m`'s unconstrained θ and σ, read at their positions in
  the point. When `m` estimates shared latent coordinates, θ includes them in
  packed order.
- `omega` — NLMEEstimation's log-Cholesky covariance vector ω for
  `nlme_layout(m)`, the value `saem`, `focei` and `empirical_bayes` take.
- `omega_matrix` — the natural covariance `diag(scale) L Lᵀ diag(scale)` at the
  point: one block-diagonal matrix for a flat model, one matrix per
  random-effect family for a nested model.
- `eta` — natural subject effects `diag(scale) L z`, in `m`'s native shape: a
  `neta × nsubjects` matrix, or one compact vector per subject for a nested
  model. Pass it as `η0`.
- `subjects` — the grouping levels, in the subject order of `eta` and `m`.
- `names` — `(; theta, sigma, eta)` coordinate names, as
  `NLMEEstimation.parameter_names(m)` reports them.
- `theta_roles` — `:population` or `:shared_latent` for each θ entry.
- `shared` — `(; names, values, estimated)` for the shared latent coordinates:
  `estimated` is true when they are part of θ. A model built with
  `shared_point` fixes them, and then the point's shared values must equal
  that model's fixed values exactly; a different point is refused.
- `loglikelihoods` — each subject's conditional log-likelihood at the start.
- `population_logprior` — the authored population priors in NLME coordinates.
- `max_error` — the largest difference between BRM's covariance coordinates in
  the point and those mapped back from `omega`.

Every covariance block must be positive definite, and the conditional
likelihoods and population prior must be finite. No input is mutated.
"""
function brm_nlme_start(m::BRMNLMEModel, u::AbstractVector)
    view = m.view
    length(u) == length(view.coordinates) || throw(DimensionMismatch(
        "brm_nlme_start: point has length $(length(u)), the RK layout has " *
        "$(length(view.coordinates)) coordinates"))
    all(isfinite, u) || _brm_nlme_error(
        "brm_nlme_start: the point must be finite")
    point = Vector{Float64}(u)
    shared = findall(==(:shared_latent), view.role)
    estimated = all(i -> i in m.theta, shared)
    if !estimated
        moved = [view.coordinates[i] for i in shared if !isequal(point[i], m.unit[i])]
        isempty(moved) || _brm_nlme_error(
            "brm_nlme_start: the model fixes its shared latent coordinates, and the " *
            "point differs at $(join(string.(moved), ", ")). Build the model with " *
            "this point as `shared_point`, or start from a point at its fixed values")
    end
    θ, σ = point[m.theta], point[m.sigma]
    families = _brm_nlme_families(m.backend, view)
    natural = _brm_nlme_natural(m.backend, view, families, point)
    Ω, H = natural.omega, natural.eta
    for (; family, record, indices) in families
        block = _brm_nlme_ragged(view) ? Ω[family] : Ω[indices, indices]
        isposdef(Symmetric(block)) || _brm_nlme_error(
            "brm_nlme_start: the covariance of `$(record.group)` random effects " *
            "is not positive definite at the point")
    end
    loglikelihoods = brm_nlme_loglikelihoods(m, θ, σ, H)
    bad = findall(!isfinite, loglikelihoods)
    isempty(bad) || _brm_nlme_error(
        "brm_nlme_start: conditional log-likelihoods are not finite for subjects " *
        "$(join(repr.(view.levels[bad]), ", "))")
    coordinates = _brm_nlme_start_coordinates(m, θ, σ, Ω, point)
    (; theta=θ, sigma=σ, coordinates.omega, omega_matrix=Ω, eta=H,
       subjects=copy(view.levels), names=_brm_nlme_parameter_names(m),
       theta_roles=view.role[m.theta],
       shared=(; names=view.coordinates[shared], values=point[shared], estimated),
       loglikelihoods, coordinates.population_logprior, coordinates.max_error)
end

brm_nlme_start(m::BRMNLMEModel, inits::NamedTuple; base=m.unit) =
    brm_nlme_start(m, brm_start(m.backend, inits; base))

function brm_nlme_start(m::BRMNLMEModel, map::BRMNLMEJointMap, stan_u::AbstractVector)
    parent(map.rk) === parent(m.backend) || _brm_nlme_error(
        "brm_nlme_start: the joint map and the NLME model come from different " *
        "BRMI instances")
    a, b = map.view, m.view
    a.coordinates == b.coordinates && a.role == b.role && a.levels == b.levels &&
        a.subject_coordinates == b.subject_coordinates &&
        a.block_margins == b.block_margins && a.family_sizes == b.family_sizes &&
        a.subject_families == b.subject_families &&
        a.nested_groups == b.nested_groups || _brm_nlme_error(
        "brm_nlme_start: the joint map's NLME partition differs from the model's")
    length(stan_u) == length(map.transport.stan_names) || throw(DimensionMismatch(
        "brm_nlme_start: $(length(stan_u)) Stan coordinates for " *
        "$(length(map.transport.stan_names)) compiled parameters"))
    all(isfinite, stan_u) || _brm_nlme_error(
        "brm_nlme_start: Stan coordinates must be finite")
    brm_nlme_start(m, brm_stan_point_to_rk(map.transport, stan_u))
end
