# A declaration bound retains the ordinary density kernel. Sampling and the
# unconstrained transform use the corresponding normalized bounded measure.
# Explicit `truncated(...)` remains an ordinary normalized distribution call.
using LinearAlgebra: Cholesky
struct _BRMConstrainedKernel{D,L,U} <: ContinuousUnivariateDistribution
    base::D
    lower::L
    upper::U
end

function _brm_constrained_kernel(base::ContinuousUnivariateDistribution;
                                 lower=nothing, upper=nothing)
    distribution = _BRMConstrainedKernel(base, lower, upper)
    minimum(distribution) < maximum(distribution) || throw(ArgumentError(
        "declaration bounds have empty intersection with the prior support"))
    distribution
end
Base.minimum(d::_BRMConstrainedKernel) = isnothing(d.lower) ? minimum(d.base) :
    max(minimum(d.base), d.lower)
Base.maximum(d::_BRMConstrainedKernel) = isnothing(d.upper) ? maximum(d.base) :
    min(maximum(d.base), d.upper)
Distributions.insupport(d::_BRMConstrainedKernel, x::Real) =
    minimum(d) <= x <= maximum(d) && Distributions.insupport(d.base, x)
logpdf(d::_BRMConstrainedKernel, x::Real) =
    Distributions.insupport(d, x) ? logpdf(d.base, x) : oftype(float(x), -Inf)
rand(rng::AbstractRNG, d::_BRMConstrainedKernel) =
    rand(rng, truncated(d.base; lower=d.lower, upper=d.upper))
Turing.Bijectors.bijector(d::_BRMConstrainedKernel) =
    Turing.Bijectors.TruncatedBijector(minimum(d), maximum(d))

Turing.@model function _brm_lkj_covariance_prior(K, scale_prior, shape)
    scales ~ product_distribution(fill(
        _brm_constrained_kernel(scale_prior; lower=0), K))
    L_corr ~ LKJCholesky(K, shape)
    Diagonal(scales) * L_corr.L
end

BRM._brm_turing_covariance_prior(K; scale_prior=Exponential(1.0), shape=1.0) =
    _brm_lkj_covariance_prior(K, scale_prior, shape)

# Parameter emission uses the same retained call as an observation, with
# extra syntax only for declaration support and composite latent geometry.
_brm_turing_parameter_ast(parameter, callables) =
    _brm_turing_prior_ast(parameter.prior, callables)
_brm_turing_prior_ast(prior::BRM._BRMPreparedExpr, callables) =
    _brm_turing_prior_ast(prior.callable, prior, callables)
function _brm_turing_prior_ast(constructor, prior, callables)
    isnothing(BRM.brm_distribution_type(constructor)) ?
        _brm_prepared_ast(prior, callables) :
        _brm_turing_bounded_prior_ast(prior, callables)
end
_brm_turing_prior_ast(::Type{<:Distributions.Distribution}, prior, callables) =
    _brm_turing_bounded_prior_ast(prior, callables)
_brm_turing_prior_ast(::Type{<:Distributions.LocationScale}, prior, callables) =
    _brm_turing_bounded_prior_ast(prior, callables)

function _brm_turing_bounded_prior_ast(prior, callables)
    bounds = (; (key => value for (key, value) in pairs(prior.kwargs)
                 if key in (:lower, :upper))...)
    isempty(bounds) && return _brm_prepared_ast(prior, callables)
    ordinary = BRM._BRMPreparedExpr(prior.callable, prior.args,
        (; (key => value for (key, value) in pairs(prior.kwargs)
            if !(key in (:lower, :upper)))...))
    base = _brm_prepared_ast(ordinary, callables)
    keywords = Expr(:parameters, (Expr(:kw, key,
        _brm_prepared_ast(value, callables)) for (key, value) in pairs(bounds))...)
    Expr(:call, :_brm_constrained_kernel, keywords, base)
end

function _brm_turing_prior_ast(::typeof(BRM.LKJCovarianceFactor), prior, callables)
    native = BRM._BRMPreparedExpr(BRM._brm_turing_covariance_prior,
                                  prior.args, prior.kwargs)
    Expr(:call, :to_submodel, _brm_prepared_ast(native, callables))
end

# The BRM marker specifies a covariance factor, whereas MvNormal's matrix
# constructor takes the covariance itself. Retain the supplied factor without
# squaring it and factoring the resulting matrix again.
_brm_mvn_cholesky(mean, factor) = MvNormal(mean,
    Distributions.PDMat(Cholesky(factor, 'L', 0)))
_brm_ast_call(::typeof(BRM.MvNormalCholesky), args, kwargs, callables) =
    _brm_ast_call(_brm_mvn_cholesky, args, kwargs, callables)

Turing.@model function _brm_horseshoe_prior(local_scale, global_scale)
    raw ~ Normal()
    lambda ~ _brm_constrained_kernel(Distributions.Cauchy(0, local_scale); lower=0)
    tau ~ _brm_constrained_kernel(Distributions.Cauchy(0, global_scale); lower=0)
    raw * lambda * tau
end
function BRM._brm_turing_horseshoe_prior(; kwargs...)
    spec = BRM._brm_horseshoe_spec(:horseshoe, (), (; kwargs...))
    _brm_horseshoe_prior(spec.local_scale, spec.global_scale)
end
function _brm_turing_prior_ast(::Type{BRM.Horseshoe}, prior, callables)
    BRM._brm_horseshoe_spec(:horseshoe, prior.args, prior.kwargs)
    native = BRM._BRMPreparedExpr(BRM._brm_turing_horseshoe_prior,
                                  prior.args, prior.kwargs)
    Expr(:call, :to_submodel, _brm_prepared_ast(native, callables))
end
