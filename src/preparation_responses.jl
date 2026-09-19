# Response coding and implicit ordinal parameters are formula semantics. Both
# emitters consume these records; ordinary distribution calls pass unchanged.
struct _BRMResponseLevels{L}
    levels::L
    n_levels::Int
end

function _brm_response_levels(target, raw; training=nothing, prefix="BRM")
    raw isa AbstractVector || error(
        "$prefix: response `$target` must be an observed vector")
    levels = isnothing(training) ? _brm_fit_levels(raw) : training.levels
    fit = _BRMResponseLevels(levels, length(levels))
    (; response=_brm_apply_fitted_levels(levels, raw; prefix), fit)
end

# This is the density kernel of independent standard normals on an optional
# ordered domain. Its prior draw sorts iid normals; its density deliberately
# has no factorial normalizer, matching an ordered Stan declaration.
struct _BRMThresholdPrior{Ordered} <: ContinuousMultivariateDistribution
    n::Int
end
Base.length(d::_BRMThresholdPrior) = d.n
Base.size(d::_BRMThresholdPrior) = (d.n,)
Base.eltype(::_BRMThresholdPrior) = Float64
Distributions.insupport(d::_BRMThresholdPrior{O}, x::AbstractVector) where {O} =
    length(x) == d.n && all(isfinite, x) &&
    (!O || all(x[i] < x[i + 1] for i in 1:length(x)-1))
function Distributions._logpdf(d::_BRMThresholdPrior, x::AbstractVector)
    insupport(d, x) || return oftype(sum(x), -Inf)
    sum(value -> logpdf(Normal(), value), x; init=zero(eltype(x)))
end
function Distributions._rand!(rng::AbstractRNG, d::_BRMThresholdPrior{O},
                               x::AbstractVector) where {O}
    randn!(rng, x)
    O && sort!(x)
    x
end

_brm_ordinal_tag(value::S, ::Type{T}; prefix="BRM") where {T,S<:T} = value
_brm_ordinal_tag(::Type{S}, ::Type{T}; prefix="BRM") where {T,S<:T} = S()
function _brm_ordinal_tag(value::ExprColumn, expected; prefix="BRM")
    isempty(getargs(value)) && isempty(getkwargs(value)) || error(
        "$prefix: ordinal structure and link tags take no arguments")
    _brm_ordinal_tag(getf(value), expected; prefix)
end
_brm_ordinal_tag(value, expected; prefix="BRM") = error(
    "$prefix: ordinal tag must be a $expected, got $(typeof(value))")

_brm_ordinal_has_fixed_intercept(x::Real) = !iszero(x)
_brm_ordinal_has_fixed_intercept(x::NamedColumn) =
    _brm_ordinal_has_fixed_intercept_parent(parent(x))
_brm_ordinal_has_fixed_intercept(_) = false
function _brm_ordinal_has_fixed_intercept_parent(p::ExprColumn)
    getf(p) === (~) || return false
    _, rhs = getargs(p, 2)
    any(term -> term isa Integer && term == 1, _brm_additive_terms(rhs))
end
_brm_ordinal_has_fixed_intercept_parent(_) = false

function _brm_threshold_predictors(raw, n_obs; prefix="BRM")
    raw isa Tuple || error(
        "$prefix: `per_threshold` expects a tuple of raw numeric columns")
    for term in raw
        term isa NamedColumn && parent(term) isa DataColumn || error(
            "$prefix: `per_threshold` expects raw numeric data columns")
        values = parent(parent(term))
        values isa AbstractVector{<:Real} || error(
            "$prefix: ordinal threshold predictor `$(name(term))` must be numeric")
        length(values) == n_obs || error(
            "$prefix: ordinal threshold predictor `$(name(term))` has " *
            "$(length(values)) rows; expected $n_obs")
        all(isfinite, values) || error(
            "$prefix: ordinal threshold predictor `$(name(term))` contains non-finite values")
    end
    raw
end

function _brm_ordinal_discrimination(raw; prefix="BRM")
    if raw isa Real
        isfinite(raw) && raw > 0 || error(
            "$prefix: ordinal discrimination must be finite and strictly positive")
    elseif raw isa NamedColumn && parent(raw) isa DataColumn
        values = parent(parent(raw))
        values isa AbstractVector{<:Real} && all(x -> isfinite(x) && x > 0, values) ||
            error("$prefix: ordinal discrimination data must contain only finite positive values")
    end
    raw
end

# Stage-major packing: flat[(k-1)*p+j] is the stage-k, term-j
# coefficient, matching SB-Stan's `array[n_cut] vector[n_terms]` runtime
# layout. A plain `reshape(beta, n_cut, p)` would read term-major and
# diverge from SB for p>1 (snag brm-threshold-et-0f516c0c).
_brm_threshold_eta(eta, columns, beta, n_cut) =
    eta .+ permutedims(reshape(beta, length(columns), n_cut)) * collect(columns)

_brm_prepare_response(target, rhs::ExprColumn, raw; training=nothing) =
    _brm_prepare_response(getf(rhs), target, rhs, raw; training)
_brm_observation_rows(raw, _shape) = raw
_brm_observation_rows(raw, shape::Tuple) = _brm_observation_rows(raw, first(shape))
_brm_observation_rows(raw::AbstractMatrix, ::Type{Distributions.Multivariate}) =
    [collect(row) for row in eachrow(raw)]
_brm_observation_rows(raw::AbstractMatrix, ::Type{Distributions.Matrixvariate}) = [raw]
_brm_prepare_response(_family, _target, rhs, raw; training=nothing) =
    (; distribution=rhs, response=_brm_observation_rows(raw, _brm_distribution_shape(rhs)),
       parameters=(), fit=nothing)

function _brm_prepare_response(::Type{<:CategoricalLogit}, target, rhs, raw;
                                training=nothing)
    prepared = _brm_response_levels(target, raw; training)
    n_args = length(getargs(rhs))
    prepared.fit.n_levels == n_args + 1 || error(
        "BRM: `CategoricalLogit($target)` has $(prepared.fit.n_levels) fitted " *
        "outcome levels but $n_args non-reference predictors")
    (; distribution=rhs, response=prepared.response, parameters=(), fit=prepared.fit)
end

function _brm_prepare_response(::Type{<:OrderedLogistic}, target, rhs, raw;
                                training=nothing)
    length(getargs(rhs)) == 1 || return (
        ; distribution=rhs, response=raw, parameters=(), fit=nothing)
    all(x -> x isa Real && isinteger(x) && x >= 1, raw) || error(
        "BRM: `OrderedLogistic($target)` expects positive integer outcome data")
    n_levels = isnothing(training) ? maximum(Int, raw) : training.n_levels
    all(<=(n_levels), raw) || error(
        "BRM: `OrderedLogistic($target)` contains an outcome beyond its fitted levels")
    fit = _BRMResponseLevels(1:n_levels, n_levels)
    name = Symbol(target, :_cutpoints)
    prior = ExprColumn(_BRMThresholdPrior{true}, n_levels - 1)
    ref = NamedColumn(name, MissingColumn())
    distribution = ExprColumn(OrderedLogistic, only(getargs(rhs)), ref)
    (; distribution, response=Int.(raw), parameters=(name => prior,), fit)
end

function _brm_prepare_response(::Type{<:Ordinal}, target, rhs, raw;
                                training=nothing)
    args = getargs(rhs)
    length(args) == 3 || return (
        ; distribution=rhs, response=raw, parameters=(), fit=nothing)
    structure = _brm_ordinal_tag(args[1], OrdinalStructure)
    link = _brm_ordinal_tag(args[2], OrdinalLink)
    eta = args[3]
    _brm_ordinal_has_fixed_intercept(eta) && error(
        "BRM: `Ordinal($target)` cannot include a fixed intercept in `eta`; " *
        "the estimated thresholds already supply the location. Use `eta ~ 0 + ...`.")
    prepared = _brm_response_levels(target, raw; training)
    n_cut = prepared.fit.n_levels - 1
    kwargs = getkwargs(rhs)
    columns = _brm_threshold_predictors(get(kwargs, :per_threshold, ()), length(raw))
    structure isa Cumulative && !isempty(columns) && error(
        "BRM: unrestricted `per_threshold` effects can make cumulative probabilities non-monotone")
    name = Symbol(target, :_thresholds)
    prior = ExprColumn(_BRMThresholdPrior{structure isa Cumulative}, n_cut)
    parameters = (name => prior,)
    if !isempty(columns)
        beta_name = Symbol(target, :_threshold_beta)
        beta_prior = ExprColumn(_BRMThresholdPrior{false}, n_cut * length(columns))
        parameters = (parameters..., beta_name => beta_prior)
        eta = ExprColumn(_brm_threshold_eta, eta, columns,
                         NamedColumn(beta_name, MissingColumn()), n_cut)
    end
    discrimination = _brm_ordinal_discrimination(get(kwargs, :discrimination, 1.0))
    distribution = ExprColumn(Ordinal, structure, link, eta,
        NamedColumn(name, MissingColumn()); discrimination)
    (; distribution, response=prepared.response, parameters, fit=prepared.fit)
end
