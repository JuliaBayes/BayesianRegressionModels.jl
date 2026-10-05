# GP statistical construction adopted from public RK 4d4b7608478a389c472dc14282c4053330517650.
# Dense factorization is a general numerical operation supplied by RK proper.
function _gp_indices(x::AbstractVector)
    isempty(x) && throw(ArgumentError("GP covariance needs at least one location"))
    collect(eachindex(x))
end

function _gp_positive(value::Real, name)
    value > 0 || throw(ArgumentError("GP $name must be positive, got $value"))
    Float64(value)
end

function _gp_period(period::Real)
    isfinite(period) && period > 0 || throw(ArgumentError(
        "GP period must be finite and positive, got $period"))
    Float64(period)
end

function _gp_jitter(jitter::Real)
    isfinite(jitter) && jitter >= 0 || throw(ArgumentError(
        "GP jitter must be finite and nonnegative, got $jitter"))
    Float64(jitter)
end


"""
    gp_exp_quad_cov(x, sigma, rho, jitter)

Squared-exponential covariance plus diagonal jitter. A vector holds
one-dimensional locations; each row of a matrix is one multi-dimensional
location. `rho` is a positive scalar or, for matrix locations, one length
scale per column. The kernel is
`sigma^2 * exp(-sum(((x[i,:]-x[j,:])./rho).^2)/2)`.
"""
function gp_exp_quad_cov(x::AbstractVector, sigma::Real, rho::Real,
        jitter::Real)
    _gp_exp_quad_cov_vector(x, sigma, rho, jitter)
end

function gp_exp_quad_cov(x::AbstractMatrix, sigma::Real,
        rho::Union{Real,AbstractVector}, jitter::Real)
    _gp_cov_args(x, sigma, rho, jitter)
    delta = _gp_differences(x) ./ _gp_axis_scale(rho)
    distance2 = dropdims(sum(delta .* delta; dims=3); dims=3)
    return sigma^2 .* exp.(-0.5 .* distance2) .+ _gp_jitter(size(x,1), jitter)
end

"""
    gp_periodic_cov(x, sigma, rho, period, jitter)

Isotropic periodic covariance plus diagonal jitter. A vector holds one
axis; each row of a matrix is one multi-dimensional location. Uses the
Euclidean distance `r = norm(x[i,:]-x[j,:])`, as in Stan's periodic
covariance: `sigma^2 * exp(-2sin(pi*r/period)^2/rho^2)`.
`rho` and `period` are positive scalars.
"""
function gp_periodic_cov(x::AbstractVector, sigma::Real, rho::Real,
        period::Real, jitter::Real)
    _gp_periodic_cov_vector(x, sigma, rho, period, jitter)
end

function gp_periodic_cov(x::AbstractMatrix, sigma::Real, rho::Real,
        period::Real, jitter::Real)
    _gp_cov_args(x, sigma, rho, jitter)
    isfinite(period) && period > 0 || throw(ArgumentError(
        "gp_periodic_cov period must be finite and positive"))
    delta = _gp_differences(x)
    distance = sqrt.(dropdims(sum(delta .* delta; dims=3); dims=3))
    sine = sin.((pi / period) .* distance)
    return sigma^2 .* exp.((-2 / rho^2) .* sine .* sine) .+
        _gp_jitter(size(x,1), jitter)
end

# Broadcast/reduction kernels: no body replication with location count or
# dimension. All arrays are fresh, and caller-owned locations stay read-only.
_gp_differences(x::AbstractMatrix) =
    reshape(x, size(x,1), 1, size(x,2)) .-
        reshape(x, 1, size(x,1), size(x,2))
_gp_axis_scale(rho::Real) = rho
_gp_axis_scale(rho::AbstractVector) = reshape(rho, 1, 1, :)
_gp_jitter(n, jitter) = jitter .* (collect(1:n) .== permutedims(collect(1:n)))

function _gp_cov_args(x, sigma, rho, jitter)
    size(x,1) > 0 && size(x,2) > 0 || throw(ArgumentError("GP locations must be nonempty"))
    all(isfinite, x) || throw(ArgumentError("GP locations must be finite"))
    isfinite(sigma) && sigma > 0 || throw(ArgumentError("GP sigma must be finite and positive"))
    _gp_rho_args(rho, size(x,2))
    isfinite(jitter) && jitter >= 0 || throw(ArgumentError("GP jitter must be finite and nonnegative"))
    return nothing
end
function _gp_rho_args(rho::Real, d)
    isfinite(rho) && rho > 0 || throw(ArgumentError("GP rho must be finite and positive"))
    return nothing
end
function _gp_rho_args(rho::AbstractVector, d)
    length(rho) == d || throw(DimensionMismatch("GP requires one length scale per location axis"))
    all(r -> isfinite(r) && r > 0, rho) || throw(ArgumentError("GP length scales must be finite and positive"))
    return nothing
end


# Vector covariance evaluation uses the same transparent graphs exposed by
# rk_model after loading the ReactiveKernels extension.
function _gp_exp_quad_cov_vector end
function _gp_periodic_cov_vector end
function _gp_cholesky_lower end

"""
    gp_chol_latent(K, z)

Multiply the lower Cholesky factor of covariance `K` by standard-normal
coordinates `z`. Only the lower triangle is read. Inputs are converted to
Float64, matching the adopted covariance/latent helper contract. General
factorization uses `ReactiveKernels.rk_cholesky_lower` through the RK extension.
"""
function gp_chol_latent(K::AbstractMatrix, z::AbstractVector)
    n = size(K, 1)
    size(K, 2) == n || throw(ArgumentError(
        "gp_chol_latent needs a square covariance, got $(size(K))"))
    length(z) == n || throw(ArgumentError(
        "gp_chol_latent z has length $(length(z)) for a $n×$n covariance"))
    L = try
        _gp_cholesky_lower(K)
    catch err
        err isa PosDefException || rethrow()
        throw(ArgumentError(
            "gp_chol_latent covariance is not positive definite " *
            "(non-positive pivot at $(err.info) — increase jitter)"))
    end
    f = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        acc = 0.0
        for j in 1:i
            acc += L[i, j] * Float64(z[j])
        end
        f[i] = acc
    end
    f
end

"""
    prepare_dual_hsgp(data; want=:posterior)

Prepare the exact dual k=20 HSGP model on 133 observations, with domain
half-width 1.5. Only packed coordinates `q` (44 values) and live centeredness
`c` (40 values) remain inputs; all bound basis/frequency work is folded.
Requires the ReactiveKernels extension. See `BayesianRegressionModels.rk_model`.
"""
function prepare_dual_hsgp end

include("statistical_gp_models.jl")
