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
    brm_nlme_start(m::BRMNLMEModel)
    brm_nlme_start(m::BRMNLMEModel, u::AbstractVector)
    brm_nlme_start(m::BRMNLMEModel, map::BRMNLMEJointMap, stan_u::AbstractVector)

Read one complete point of the *same* model as NLMEEstimation starting values
for `m`. `u` is a full RK packed unconstrained point of `m.backend`'s layout.
`stan_u` is a full unconstrained point of `map.stan_problem`; `map` must come
from the same BRMI instance and NLME partition as `m`, and its validated
semantic transport converts `stan_u` to RK coordinates first. Requires
NLMEEstimation to be loaded.

Without a point, the start is the model's **prior medians**: every population
coefficient, sampled scalar and random-effect scale starts at the median of
the prior law the model declares for it (laws whose arguments name another
scalar use that scalar's own median), the random-effect correlations at the
identity, and every subject effect at zero. A coordinate whose law has no
median (`Flat()`, or an argument without a value) and every other estimated
coordinate (shared latents, simplexes, thresholds) keeps the zero
unconstrained value. The result adds `start_source = (; prior_median, zero)`,
the packed names of the θ, σ and random-effect scale coordinates in each
group.

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

# ── Prior medians ────────────────────────────────────────────────────────────
#
# Beside each population coefficient, sampled scalar and random-effect scale it
# declares, the RK emitter records the law it wrote (`prior`, `scale_priors`).
# The default start reads those records, so it follows the emitted program.

# A recorded law as a Distributions.jl distribution. Names resolve through
# `value`: another declaration's own start, or a bound scalar. `nothing` when
# the law has no median (`Flat()`) or an argument has no value.
_brm_prior_law(x::Real, _value) = Float64(x)
_brm_prior_law(x::Symbol, value) = value(x)
function _brm_prior_law(x::Expr, value)
    x.head === :call && first(x.args) isa Symbol || return nothing
    args = Any[_brm_prior_law(a, value) for a in x.args[2:end]]
    any(isnothing, args) && return nothing
    _brm_prior_law(Val(first(x.args)), args...)
end
_brm_prior_law(_x, _value) = nothing

_brm_prior_law(::Val{:Flat}) = nothing
_brm_prior_law(::Val{:HalfNormal}, s::Real) = truncated(Normal(0, s), 0, Inf)
_brm_prior_law(::Val{:HalfCauchy}, s::Real) = truncated(Cauchy(0, s), 0, Inf)
# `restricted` keeps its family's kernel on the support, so its median within
# the bounds is the normalized truncation's.
_brm_prior_law(::Val{:restricted}, d::UnivariateDistribution, lo::Real, hi::Real) =
    truncated(d, lo, hi)
_brm_prior_law(::Val{:truncated}, d::UnivariateDistribution, lo::Real, hi::Real) =
    truncated(d, lo, hi)
# RKPPL's three-argument Student t is located and scaled.
_brm_prior_law(::Val{:StudentT}, nu::Real, mu::Real, s::Real) = mu + s * TDist(nu)
for op in (:+, :-, :*, :/, :^, :sqrt, :exp, :log, :abs)
    @eval _brm_prior_law(::Val{$(QuoteNode(op))}, args::Real...) = $op(args...)
end
# Any other law names its Distributions.jl family.
function _brm_prior_law(::Val{F}, args...) where {F}
    isdefined(Distributions, F) || return nothing
    family = getfield(Distributions, F)
    family isa Type && family <: UnivariateDistribution &&
        all(a -> a isa Real, args) || return nothing
    family(args...)
end

_brm_prior_median(law::UnivariateDistribution) = Float64(median(law))
_brm_prior_median(_law) = nothing

_brm_nlme_scalar_data(plan::_RKHeldOutPlan, name) = _brm_nlme_scalar_data(plan.parent, name)
_brm_nlme_scalar_data(plan, name) = let x = get(plan.columns, name, nothing)
    x isa Real ? Float64(x) : nothing
end

# One declaration with its element at `index` (all elements for `()`) set to x.
function _brm_nlme_with_element(values, declaration, index, x)
    current = _brm_rk_declaration_value(values, declaration)
    value = current isa AbstractArray ?
        (isempty(index) ? fill(x, size(current)) :
            (new = Array{Float64}(current); new[index...] = x; new)) : x
    _brm_rk_with_declaration(values, declaration, value)
end

# The packed names a record's element addresses: `<declaration>.<index>`, or
# every coordinate of the declaration.
_brm_nlme_element_names(view, declaration, index) = isempty(index) ?
    [n for n in view.coordinates if _brm_nlme_owned(n, declaration)] :
    [Symbol(declaration, ".", join(index, "."))]

# Each record kind contributes the declarations its recorded laws start.
# Returns `(values, names)` with the names of the coordinates it set.
_brm_nlme_prior_start(::Val, values, record, view, value) = (values, Symbol[])
function _brm_nlme_prior_start(::Val{:population}, values, record, view, value)
    x = _brm_prior_median(_brm_prior_law(get(record, :prior, nothing), value))
    x === nothing && return (values, Symbol[])
    index = get(record, :index, ())
    _brm_nlme_with_element(values, record.declaration, index, x),
        _brm_nlme_element_names(view, record.declaration, index)
end
_brm_nlme_prior_start(::Val{:population_block}, values, record, view, value) =
    _brm_nlme_prior_start(Val(:population), values, record, view, value)
function _brm_nlme_prior_start(::Val{:scalar}, values, record, view, value)
    x = value(record.declaration)
    x === nothing && return (values, Symbol[])
    _brm_nlme_with_element(values, record.declaration, (), x),
        _brm_nlme_element_names(view, record.declaration, ())
end
function _brm_nlme_prior_start(::Val{:ranef}, values, record, view, value)
    laws = get(record, :scale_priors, nothing)
    laws === nothing && return (values, Symbol[])
    names = Symbol[]
    for (k, law) in enumerate(laws)
        x = _brm_prior_median(_brm_prior_law(law, value))
        x === nothing && continue
        declaration, index = isnothing(record.scales) ? (record.scale, (k,)) :
            (record.scales[k], ())
        values = _brm_nlme_with_element(values, declaration, index, x)
        append!(names, _brm_nlme_element_names(view, declaration, index))
    end
    values, names
end

function brm_nlme_start(m::BRMNLMEModel)
    view, backend = m.view, m.backend
    records = _brm_nlme_records(backend.plan)
    scalars = Dict(r.declaration => r for r in records if r.kind === :scalar)
    medians = Dict{Symbol,Union{Nothing,Float64}}()
    pending = Set{Symbol}()
    # A scalar's start is its own prior median, resolved when a law reads it.
    function value(name::Symbol)
        haskey(medians, name) && return medians[name]
        record = get(scalars, name, nothing)
        record === nothing && return _brm_nlme_scalar_data(backend.plan, name)
        name in pending && return nothing
        push!(pending, name)
        x = _brm_prior_median(_brm_prior_law(get(record, :prior, nothing), value))
        delete!(pending, name)
        medians[name] = x
    end
    values = _rk_constrained_values(backend, m.unit)
    set = Symbol[]
    for record in records
        values, names = _brm_nlme_prior_start(Val(record.kind), values, record, view, value)
        append!(set, names)
    end
    position = Dict(n => i for (i, n) in enumerate(view.coordinates))
    chosen = sort!(unique!([position[n] for n in set]))
    u = copy(m.unit)
    u[chosen] = _rk_unconstrained_values(backend, values)[chosen]
    estimated = sort!(vcat(m.theta, m.sigma, findall(==(:subject_scale), view.role)))
    start = brm_nlme_start(m, u)
    merge(start, (; start_source=(; prior_median=view.coordinates[chosen],
        zero=view.coordinates[setdiff(estimated, chosen)])))
end
