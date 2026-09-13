# Executable Julia contracts for public likelihood surfaces whose formula
# lowering is implemented by sbimpl.  The @brm parser keeps the distribution
# type/function and its expression arguments separate, so these numeric
# constructors do not change formula syntax.

"""Supertype for the ordinal probability construction used by [`Ordinal`](@ref)."""
abstract type OrdinalStructure end

"""Cumulative-link ordinal probabilities, `P(Y <= k) = F(d * (cut[k] - eta))`."""
struct Cumulative <: OrdinalStructure end

"""Stopping-ratio ordinal probabilities, `P(Y = k | Y >= k) = F(d * (cut[k] - eta[k]))`."""
struct StoppingRatio <: OrdinalStructure end

"""Supertype for typed ordinal inverse-link choices used by [`Ordinal`](@ref)."""
abstract type OrdinalLink end

"""Logistic inverse link for [`Ordinal`](@ref)."""
struct LogitLink <: OrdinalLink end

"""Standard-normal inverse link for [`Ordinal`](@ref)."""
struct ProbitLink <: OrdinalLink end

"""Complementary-log-log inverse link for [`Ordinal`](@ref)."""
struct CloglogLink <: OrdinalLink end

"""
    Ordinal(structure, link, eta, thresholds; discrimination=1)

Executable ordinal distribution with independently typed probability
`structure` and inverse `link`. `structure` is [`Cumulative`](@ref) or
[`StoppingRatio`](@ref); `link` is [`LogitLink`](@ref), [`ProbitLink`](@ref),
or [`CloglogLink`](@ref). `discrimination` is finite and strictly positive.

For a cumulative model, `thresholds` must be strictly increasing and `eta`
must be scalar. For a stopping-ratio model the thresholds are stage-specific
intercepts and need not be ordered; `eta` may be either one shared scalar or a
vector with one value per non-terminal stage. The support is
`1:(length(thresholds) + 1)`.

Inside `@brm`, omit `thresholds`: for example,
`y ~ Ordinal(Cumulative(), ProbitLink(), eta)`. BRM prepares the fitted outcome
levels and threshold parameters. Formula-only keywords `discrimination=` and
`per_threshold=` are documented in the likelihood guide.
"""
struct Ordinal{S<:OrdinalStructure,L<:OrdinalLink,T<:Real,E} <:
       Distributions.DiscreteUnivariateDistribution
    structure::S
    link::L
    eta::E
    thresholds::Vector{T}
    discrimination::T
end

_ordinal_eta_values(eta::Real) = (eta,)
_ordinal_eta_values(eta::AbstractVector{<:Real}) = Tuple(eta)
_ordinal_eta_values(eta) = throw(ArgumentError(
    "Ordinal: `eta` must be a real scalar or a real vector, got $(typeof(eta))"))

function Ordinal(structure::S, link::L, eta,
                 thresholds::AbstractVector{<:Real}, discrimination::Real;
                 check_args::Bool=true) where {S<:OrdinalStructure,L<:OrdinalLink}
    raw_eta = _ordinal_eta_values(eta)
    T = promote_type(map(typeof, map(float,
        (raw_eta..., thresholds..., discrimination)))...)
    eta_p = eta isa Real ? convert(T, float(eta)) :
            collect(T, map(float, eta))
    cuts = collect(T, map(float, thresholds))
    disc = convert(T, float(discrimination))

    Distributions.@check_args(
        Ordinal,
        (eta_p, all(isfinite, raw_eta), "eta must be finite"),
        (cuts, all(isfinite, cuts), "thresholds must be finite"),
        (disc, isfinite(disc) && disc > zero(disc),
         "discrimination must be finite and strictly positive"),
    )
    if structure isa Cumulative
        eta_p isa Real || throw(ArgumentError(
            "Ordinal(Cumulative(), ...): `eta` must be scalar"))
        all(cuts[i] < cuts[i + 1] for i in 1:length(cuts)-1) ||
            throw(DomainError(cuts,
                "Ordinal(Cumulative(), ...): thresholds must be strictly increasing"))
    elseif eta_p isa AbstractVector
        length(eta_p) == length(cuts) || throw(DimensionMismatch(
            "Ordinal(StoppingRatio(), ...): vector eta has length $(length(eta_p)); " *
            "expected $(length(cuts)), one value per threshold"))
    end
    Ordinal{S,L,T,typeof(eta_p)}(structure, link, eta_p, cuts, disc)
end

Ordinal(structure::OrdinalStructure, link::OrdinalLink, eta,
        thresholds::AbstractVector{<:Real}; discrimination::Real=1,
        check_args::Bool=true) =
    Ordinal(structure, link, eta, thresholds, discrimination; check_args)

Distributions.params(d::Ordinal) =
    (d.structure, d.link, d.eta, d.thresholds, d.discrimination)
Distributions.partype(::Ordinal{S,L,T}) where {S,L,T} = T
Distributions.@distr_support Ordinal 1 (length(d.thresholds) + 1)

_ordinal_link_cdf(::LogitLink, z) = logistic(z)
_ordinal_link_logcdf(::LogitLink, z) = loglogistic(z)
_ordinal_link_logccdf(::LogitLink, z) = log1mlogistic(z)

const _ORDINAL_STANDARD_NORMAL = Normal()
_ordinal_link_cdf(::ProbitLink, z) = cdf(_ORDINAL_STANDARD_NORMAL, z)
_ordinal_link_logcdf(::ProbitLink, z) = logcdf(_ORDINAL_STANDARD_NORMAL, z)
_ordinal_link_logccdf(::ProbitLink, z) = logccdf(_ORDINAL_STANDARD_NORMAL, z)

_ordinal_link_cdf(::CloglogLink, z) = -expm1(-exp(z))
_ordinal_link_logcdf(::CloglogLink, z) = log1mexp(-exp(z))
_ordinal_link_logccdf(::CloglogLink, z) = -exp(z)

_ordinal_stage_eta(eta::Real, _k) = eta
_ordinal_stage_eta(eta::AbstractVector, k) = eta[k]

function Distributions.probs(d::Ordinal{<:Cumulative})
    isempty(d.thresholds) && return [one(partype(d))]
    cumulative = map(d.thresholds) do cut
        _ordinal_link_cdf(d.link, d.discrimination * (cut - d.eta))
    end
    [first(cumulative); diff(cumulative);
     one(eltype(cumulative)) - last(cumulative)]
end

function Distributions.probs(d::Ordinal{<:StoppingRatio})
    T = partype(d)
    rv = Vector{T}(undef, length(d.thresholds) + 1)
    remaining = one(T)
    for k in eachindex(d.thresholds)
        q = _ordinal_link_cdf(
            d.link,
            d.discrimination *
            (d.thresholds[k] - _ordinal_stage_eta(d.eta, k)),
        )
        rv[k] = remaining * q
        remaining *= one(T) - q
    end
    rv[end] = remaining
    rv
end

function Distributions.logpdf(d::Ordinal{<:Cumulative}, k::Real)
    K = length(d.thresholds) + 1
    (!isinteger(k) || k < 1 || k > K) &&
        return oftype(d.discrimination, -Inf)
    K == 1 && return zero(partype(d))
    i = Int(k)
    z(j) = d.discrimination * (d.thresholds[j] - d.eta)
    i == 1 && return _ordinal_link_logcdf(d.link, z(1))
    i == K && return _ordinal_link_logccdf(d.link, z(K - 1))
    logsubexp(
        _ordinal_link_logcdf(d.link, z(i)),
        _ordinal_link_logcdf(d.link, z(i - 1)),
    )
end

function Distributions.logpdf(d::Ordinal{<:StoppingRatio}, k::Real)
    K = length(d.thresholds) + 1
    (!isinteger(k) || k < 1 || k > K) &&
        return oftype(d.discrimination, -Inf)
    i = Int(k)
    rv = zero(partype(d))
    for j in 1:min(i, K - 1)
        z = d.discrimination *
            (d.thresholds[j] - _ordinal_stage_eta(d.eta, j))
        rv += j == i ? _ordinal_link_logcdf(d.link, z) :
                       _ordinal_link_logccdf(d.link, z)
    end
    rv
end

Random.rand(rng::Random.AbstractRNG, d::Ordinal) =
    rand(rng, Categorical(Distributions.probs(d)))

"""
    OrderedLogistic(eta, cutpoints)

Ordered-logistic distribution with location `eta` and strictly increasing
`cutpoints`.  The support is `1:(length(cutpoints) + 1)`.

Inside `@brm`, `y ~ OrderedLogistic(eta)` remains the cumulative-link formula
shorthand: BRM prepares and estimates the cutpoints. A standalone numeric
distribution must supply them explicitly.
"""
struct OrderedLogistic{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    eta::T
    cutpoints::Vector{T}

    OrderedLogistic{T}(eta::T, cutpoints::Vector{T}) where {T<:Real} =
        new{T}(eta, cutpoints)
end

function OrderedLogistic(eta::Real, cutpoints::AbstractVector{<:Real};
                         check_args::Bool=true)
    values = promote(float(eta), map(float, cutpoints)...)
    eta_p = first(values)
    cuts = collect(typeof(eta_p), Base.tail(values))
    Distributions.@check_args(OrderedLogistic,
        (eta_p, isfinite(eta_p), "eta must be finite"),
        (cuts, all(isfinite, cuts), "cutpoints must be finite"),
        (cuts, all(cuts[i] < cuts[i + 1] for i in 1:length(cuts)-1),
         "cutpoints must be strictly increasing"),
    )
    OrderedLogistic{typeof(eta_p)}(eta_p, cuts)
end

Distributions.params(d::OrderedLogistic) = (d.eta, d.cutpoints)
Distributions.partype(::OrderedLogistic{T}) where {T} = T
Distributions.@distr_support OrderedLogistic 1 (length(d.cutpoints) + 1)

function Distributions.probs(d::OrderedLogistic)
    isempty(d.cutpoints) && return [one(partype(d))]
    cumulative = logistic.(d.cutpoints .- d.eta)
    [first(cumulative); diff(cumulative); one(eltype(cumulative)) - last(cumulative)]
end

function Distributions.logpdf(d::OrderedLogistic, k::Real)
    K = length(d.cutpoints) + 1
    (!isinteger(k) || k < 1 || k > K) && return oftype(float(d.eta), -Inf)
    K == 1 && return zero(partype(d))
    i = Int(k)
    i == 1 && return loglogistic(first(d.cutpoints) - d.eta)
    i == K && return log1mlogistic(last(d.cutpoints) - d.eta)
    logsubexp(
        loglogistic(d.cutpoints[i] - d.eta),
        loglogistic(d.cutpoints[i - 1] - d.eta),
    )
end

Random.rand(rng::Random.AbstractRNG, d::OrderedLogistic) =
    rand(rng, Categorical(Distributions.probs(d)))

"""
    CategoricalLogit(eta2, eta3, ...)
    CategoricalLogit(@brm(formula))

Reference-class categorical distribution.  Class 1 has logit zero and each
argument is the logit for one subsequent class.  The support is
`1:(length(params(d)) + 1)`.

Inside an outer `@brm` model, the concise nested-`@brm` form is normalised to
one ordinary scalar linear predictor per non-reference outcome class. Those
predictors share formula structure while fitting distinct coefficients.
"""
struct CategoricalLogit{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    nonreference_logits::Vector{T}

    CategoricalLogit{T}(eta::Vector{T}) where {T<:Real} = new{T}(eta)
end

function CategoricalLogit(eta::Real...; check_args::Bool=true)
    isempty(eta) && return CategoricalLogit{Float64}(Float64[])
    promoted = promote(map(float, eta)...)
    logits = collect(promoted)
    Distributions.@check_args(
        CategoricalLogit,
        (logits, all(isfinite, logits), "logits must be finite"),
    )
    CategoricalLogit{eltype(logits)}(logits)
end

CategoricalLogit(eta::AbstractVector{<:Real}; check_args::Bool=true) =
    CategoricalLogit(eta...; check_args=check_args)

Distributions.params(d::CategoricalLogit) = Tuple(d.nonreference_logits)
Distributions.partype(::CategoricalLogit{T}) where {T} = T
Distributions.ncategories(d::CategoricalLogit) = length(d.nonreference_logits) + 1
Distributions.@distr_support CategoricalLogit 1 Distributions.ncategories(d)

_categorical_logit_values(d::CategoricalLogit{T}) where {T} =
    [zero(T); d.nonreference_logits]

function Distributions.probs(d::CategoricalLogit)
    logits = _categorical_logit_values(d)
    m = maximum(logits)
    weights = exp.(logits .- m)
    weights ./ sum(weights)
end

function Distributions.logpdf(d::CategoricalLogit, k::Real)
    K = Distributions.ncategories(d)
    (!isinteger(k) || k < 1 || k > K) &&
        return oftype(zero(partype(d)), -Inf)
    logits = _categorical_logit_values(d)
    logits[Int(k)] - logsumexp(logits)
end

Random.rand(rng::Random.AbstractRNG, d::CategoricalLogit) =
    rand(rng, Categorical(Distributions.probs(d)))

"""
    CircularVonMises(mu, kappa; interval=(-pi, pi))

Von-Mises distribution represented on one fixed half-open principal interval.
`interval` must be a finite pair `(lo, hi)` with `hi - lo == 2pi`.  The density
is exactly Distributions.jl's `VonMises(mu, kappa)` after mapping both the mean
and observation to equivalent angles, while `rand` maps native von-Mises draws
back into `[lo, hi)`.

This surface is deliberately distinct from Distributions' `VonMises`, whose support
moves with `mu`.  In an `@brm` formula the StanBlocks backend uses the same
fixed-interval contract and lowers density and RNG evaluation to Stan's native
`von_mises_lpdf` and `von_mises_rng`.
"""
struct CircularVonMises{T<:Real,S<:Real} <:
       Distributions.ContinuousUnivariateDistribution
    mu::T
    kappa::T
    lo::S
    hi::S

    CircularVonMises{T,S}(mu::T, kappa::T, lo::S, hi::S) where {T<:Real,S<:Real} =
        new{T,S}(mu, kappa, lo, hi)
end

function CircularVonMises(mu::Real, kappa::Real;
                          interval=(-Float64(pi), Float64(pi)),
                          check_args::Bool=true)
    interval isa Tuple && length(interval) == 2 || throw(ArgumentError(
        "CircularVonMises: `interval` must be a 2-tuple `(lo, hi)`"))
    lo_raw, hi_raw = interval
    lo_raw isa Real && hi_raw isa Real || throw(ArgumentError(
        "CircularVonMises: `interval` endpoints must be real numbers"))

    mu_p, kappa_p = promote(float(mu), float(kappa))
    lo, hi = promote(float(lo_raw), float(hi_raw))
    width = hi - lo
    interval_tol = 8eps(typeof(width))
    circumference = 2 * oftype(width, pi)
    Distributions.@check_args(CircularVonMises,
        (mu_p, isfinite(mu_p), "mu must be finite"),
        (kappa_p, kappa_p > zero(kappa_p), "kappa must be positive"),
        ((lo, hi), all(isfinite, (lo, hi)) && lo < hi,
         "interval endpoints must be finite with lo < hi"),
        (width, isapprox(width, circumference;
                         rtol=interval_tol, atol=interval_tol),
         "interval must have length 2pi"),
    )
    CircularVonMises{typeof(mu_p),typeof(lo)}(mu_p, kappa_p, lo, hi)
end

CircularVonMises(mu::Real, kappa::Real, interval::Tuple;
                 check_args::Bool=true) =
    CircularVonMises(mu, kappa; interval, check_args)

Distributions.params(d::CircularVonMises) =
    (d.mu, d.kappa, (d.lo, d.hi))
Distributions.partype(::CircularVonMises{T,S}) where {T,S} = promote_type(T, S)
Base.minimum(d::CircularVonMises) = d.lo
Base.maximum(d::CircularVonMises) = d.hi
Distributions.insupport(d::CircularVonMises, x::Real) = d.lo <= x < d.hi

_circular_von_mises_mean(d::CircularVonMises) =
    d.lo + mod(d.mu - d.lo, d.hi - d.lo)

function _circular_von_mises_base(d::CircularVonMises)
    VonMises(_circular_von_mises_mean(d), d.kappa; check_args=false)
end

function _circular_von_mises_representative(d::CircularVonMises, x::Real)
    wrapped_mu = _circular_von_mises_mean(d)
    moving_lo = wrapped_mu - oftype(wrapped_mu, pi)
    moving_lo + mod(x - moving_lo, 2 * oftype(wrapped_mu, pi))
end

function Distributions.logpdf(d::CircularVonMises, x::Real)
    Distributions.insupport(d, x) ||
        return oftype(float(d.mu + d.kappa + d.lo + d.hi), -Inf)
    logpdf(_circular_von_mises_base(d),
           _circular_von_mises_representative(d, x))
end

function Random.rand(rng::Random.AbstractRNG, d::CircularVonMises)
    draw = rand(rng, _circular_von_mises_base(d))
    d.lo + mod(draw - d.lo, d.hi - d.lo)
end

"""
    SkewDoubleExponential(mu, sigma, tau)

Asymmetric double-exponential distribution with the exact parameterization of
Stan's native `skew_double_exponential(mu, sigma, tau)`. `sigma` is the native
Stan scale and `tau` is the probability mass at or below `mu`, so
`cdf(d, mu) == tau`. At `tau == 0.5`, this is exactly `Laplace(mu, sigma)`.

Distributions.jl represents the same family as
`SkewedExponentialPower(mu, sigma_sepd, 1, tau)`, with
`sigma == 4sigma_sepd*tau*(1-tau)`. BRM uses that identity as the executable
Julia receipt while lowering this type directly to Stan's native family.
"""
struct SkewDoubleExponential{T<:Real} <:
       Distributions.ContinuousUnivariateDistribution
    mu::T
    sigma::T
    tau::T

    SkewDoubleExponential{T}(mu::T, sigma::T, tau::T) where {T<:Real} =
        new{T}(mu, sigma, tau)
end

function SkewDoubleExponential(mu::Real, sigma::Real, tau::Real;
                               check_args::Bool=true)
    mu_p, sigma_p, tau_p = promote(float(mu), float(sigma), float(tau))
    Distributions.@check_args(SkewDoubleExponential,
        (mu_p, isfinite(mu_p), "mu must be finite"),
        (sigma_p, isfinite(sigma_p) && sigma_p > zero(sigma_p),
         "sigma must be finite and positive"),
        (tau_p, isfinite(tau_p) && zero(tau_p) < tau_p < one(tau_p),
         "tau must lie strictly between zero and one"),
    )
    SkewDoubleExponential{typeof(mu_p)}(mu_p, sigma_p, tau_p)
end

Distributions.params(d::SkewDoubleExponential) = (d.mu, d.sigma, d.tau)
Distributions.partype(::SkewDoubleExponential{T}) where {T} = T
Distributions.location(d::SkewDoubleExponential) = d.mu
Distributions.scale(d::SkewDoubleExponential) = d.sigma
Distributions.@distr_support SkewDoubleExponential -Inf Inf

_skew_double_exponential_sepd(d::SkewDoubleExponential) =
    SkewedExponentialPower(
        d.mu, d.sigma / (4 * d.tau * (one(d.tau) - d.tau)),
        one(d.tau), d.tau; check_args=false)

Distributions.logpdf(d::SkewDoubleExponential, x::Real) =
    logpdf(_skew_double_exponential_sepd(d), x)
Distributions.cdf(d::SkewDoubleExponential, x::Real) =
    cdf(_skew_double_exponential_sepd(d), x)
Distributions.logcdf(d::SkewDoubleExponential, x::Real) =
    logcdf(_skew_double_exponential_sepd(d), x)
Distributions.quantile(d::SkewDoubleExponential, q::Real) =
    quantile(_skew_double_exponential_sepd(d), q)
Random.rand(rng::Random.AbstractRNG, d::SkewDoubleExponential) =
    rand(rng, _skew_double_exponential_sepd(d))

"""
    ZeroInflatedPoisson(lambda, zi)

Mixture of a point mass at zero (probability `zi`) and `Poisson(lambda)`.
"""
struct ZeroInflatedPoisson{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    lambda::T
    zi::T

    ZeroInflatedPoisson{T}(lambda::T, zi::T) where {T<:Real} =
        new{T}(lambda, zi)
end

function ZeroInflatedPoisson(lambda::Real, zi::Real; check_args::Bool=true)
    lambda_p, zi_p = promote(float(lambda), float(zi))
    Distributions.@check_args(ZeroInflatedPoisson,
        (lambda_p, lambda_p >= zero(lambda_p), "lambda must be nonnegative"),
        (zi_p, zero(zi_p) <= zi_p <= one(zi_p),
         "zero-inflation probability must lie in [0, 1]"),
    )
    ZeroInflatedPoisson{typeof(lambda_p)}(lambda_p, zi_p)
end

Distributions.params(d::ZeroInflatedPoisson) = (d.lambda, d.zi)
Distributions.partype(::ZeroInflatedPoisson{T}) where {T} = T
Distributions.@distr_support ZeroInflatedPoisson 0 Inf

function Distributions.logpdf(d::ZeroInflatedPoisson, k::Real)
    (!isinteger(k) || k < 0) && return oftype(float(d.lambda), -Inf)
    poisson_lp = logpdf(Poisson(d.lambda), Int(k))
    k == 0 && return logaddexp(log(d.zi), log1p(-d.zi) + poisson_lp)
    log1p(-d.zi) + poisson_lp
end

Random.rand(rng::Random.AbstractRNG, d::ZeroInflatedPoisson) =
    rand(rng) < d.zi ? 0 : rand(rng, Poisson(d.lambda))

"""
    NegativeBinomial2(mu, phi)

Negative-binomial distribution parameterised by positive mean `mu` and
positive shape/precision `phi`, matching Stan's `neg_binomial_2(mu, phi)`.
"""
struct NegativeBinomial2{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    mu::T
    phi::T

    NegativeBinomial2{T}(mu::T, phi::T) where {T<:Real} = new{T}(mu, phi)
end

function NegativeBinomial2(mu::Real, phi::Real; check_args::Bool=true)
    mu_p, phi_p = promote(float(mu), float(phi))
    Distributions.@check_args(NegativeBinomial2,
        (mu_p, mu_p > zero(mu_p), "mu must be positive"),
        (phi_p, phi_p > zero(phi_p), "phi must be positive"),
    )
    NegativeBinomial2{typeof(mu_p)}(mu_p, phi_p)
end

Distributions.params(d::NegativeBinomial2) = (d.mu, d.phi)
Distributions.partype(::NegativeBinomial2{T}) where {T} = T
Distributions.@distr_support NegativeBinomial2 0 Inf

_negative_binomial2_base(d::NegativeBinomial2) =
    NegativeBinomial(d.phi, d.phi / (d.phi + d.mu))
Distributions.logpdf(d::NegativeBinomial2, k::Real) =
    logpdf(_negative_binomial2_base(d), k)
Random.rand(rng::Random.AbstractRNG, d::NegativeBinomial2) =
    rand(rng, _negative_binomial2_base(d))

"""
    BetaBinomial2(trials, mean, precision)

Beta-binomial distribution parameterised by success probability `mean` and
positive concentration `precision`.
"""
struct BetaBinomial2{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    trials::Int
    mean::T
    precision::T

    BetaBinomial2{T}(trials::Int, mean::T, precision::T) where {T<:Real} =
        new{T}(trials, mean, precision)
end

function BetaBinomial2(trials::Integer, mean::Real, precision::Real;
                       check_args::Bool=true)
    mean_p, precision_p = promote(float(mean), float(precision))
    n = Int(trials)
    Distributions.@check_args(BetaBinomial2,
        (n, n >= 0, "trials must be nonnegative"),
        (mean_p, zero(mean_p) < mean_p < one(mean_p),
         "mean must lie strictly between zero and one"),
        (precision_p, precision_p > zero(precision_p),
         "precision must be positive"),
    )
    BetaBinomial2{typeof(mean_p)}(n, mean_p, precision_p)
end

Distributions.params(d::BetaBinomial2) = (d.trials, d.mean, d.precision)
Distributions.partype(::BetaBinomial2{T}) where {T} = T
Distributions.@distr_support BetaBinomial2 0 d.trials

_beta_binomial2_base(d::BetaBinomial2) = BetaBinomial(
    d.trials, d.mean * d.precision, (one(d.mean) - d.mean) * d.precision)
Distributions.logpdf(d::BetaBinomial2, k::Real) = logpdf(_beta_binomial2_base(d), k)
Random.rand(rng::Random.AbstractRNG, d::BetaBinomial2) =
    rand(rng, _beta_binomial2_base(d))

"""
    BinomialLogit(n, logitp)

Binomial distribution parameterised on the logit scale.  Both BRM backends
use a logit-native log-pmf and avoid an inverse-logit round trip for density
evaluation.
"""
struct BinomialLogit{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    n::Int
    logitp::T

    BinomialLogit{T}(n::Int, logitp::T) where {T<:Real} = new{T}(n, logitp)
end

function BinomialLogit(n::Integer, logitp::Real; check_args::Bool=true)
    eta = float(logitp)
    n_int = Int(n)
    Distributions.@check_args(BinomialLogit,
        (n_int, n_int >= 0, "n must be nonnegative"),
        (eta, isfinite(eta), "logitp must be finite"),
    )
    BinomialLogit{typeof(eta)}(n_int, eta)
end

Distributions.params(d::BinomialLogit) = (d.n, d.logitp)
Distributions.partype(::BinomialLogit{T}) where {T} = T
Distributions.ntrials(d::BinomialLogit) = d.n
Distributions.succprob(d::BinomialLogit) = logistic(d.logitp)
Distributions.@distr_support BinomialLogit 0 d.n

function Distributions.logpdf(d::BinomialLogit, k::Real)
    (!isinteger(k) || k < 0 || k > d.n) &&
        return oftype(float(d.logitp), -Inf)
    k_int = Int(k)
    SpecialFunctions.logabsbinomial(d.n, k_int)[1] +
        k_int * loglogistic(d.logitp) +
        (d.n - k_int) * log1mlogistic(d.logitp)
end

Random.rand(rng::Random.AbstractRNG, d::BinomialLogit) =
    rand(rng, Binomial(d.n, logistic(d.logitp)))

"""
    TruncatedNormal(mu, sigma, lloq, uloq)

Historical BRM spelling for the **censored** normal observation model.
Numeric arguments construct Distributions.jl's exact
`Censored(Normal(mu, sigma), lloq, uloq)` distribution, including its boundary
atoms.  The name is retained for formula compatibility.
"""
function TruncatedNormal end

function TruncatedNormal(mu::Real, sigma::Real, lloq::Real, uloq::Real;
                         check_args::Bool=true)
    lower, upper = promote(float(lloq), float(uloq))
    base = Normal(mu, sigma; check_args=check_args)
    Distributions.Censored(base, lower, upper; check_args=check_args)
end
