# Posterior-preserving sum-to-zero (S2Z) geometry kernel.
#
# Pure-Julia implementation of Sean's brms PR #1919 construction, shared by
# SBBRMI emission, the offline Fisher-candidate selector and audits. Reference
# points: pinned brms `73cf607889879cb2a55f50b88d8141d76ff43279`
# (`R/stan-predictor.R` S2Z emission, `R/re-autocenter-fit.R` precursor),
# captured pupil Stan
# (`research/pupil_total_effects/results/student_mixture/s2z_auto/clean.stan`),
# and the staged design in `research/s2z_design/README.md`.
#
# Scope of this first kernel: scalar (conditionally independent) blocks with
# J >= 2 levels. Correlated blocks follow the same restricted-determinant
# identity (`research/s2z_design/README.md` §3) and are not implemented here.

# Orthonormal Helmert contrasts, applied implicitly in O(J). The dense form is
# Q[j,k] = 1/sqrt(k*(k+1)) for j <= k, Q[k+1,k] = -k/sqrt(k*(k+1)), zero below.
# This matches Stan's `sum_to_zero_constrain` basis orientation used by brms.
function _s2z_helmert_mul(v::AbstractVector{<:Real})
    J = length(v) + 1
    J >= 2 || throw(ArgumentError("S2Z Helmert map needs at least one contrast (J >= 2)"))
    T = promote_type(eltype(v), Float64)
    out = Vector{T}(undef, J)
    suffix = zero(T)
    @inbounds for k in (J - 1):-1:1
        s = inv(sqrt(T(k * (k + 1))))
        suffix += T(v[k]) * s
        out[k + 1] = suffix - T(k + 1) * T(v[k]) * s
    end
    out[1] = suffix
    out
end

function _s2z_helmert_transpose_mul(u::AbstractVector{<:Real})
    J = length(u)
    J >= 2 || throw(ArgumentError("S2Z Helmert transpose needs at least two levels (J >= 2)"))
    T = promote_type(eltype(u), Float64)
    out = Vector{T}(undef, J - 1)
    prefix = zero(T)
    @inbounds for k in 1:J-1
        prefix += T(u[k])
        out[k] = (prefix - T(k) * T(u[k + 1])) * inv(sqrt(T(k * (k + 1))))
    end
    out
end

function _s2z_check_partial_args(
        tau::Real, rho::AbstractVector{<:Real}, J::Integer)
    isfinite(tau) && tau > 0 ||
        throw(ArgumentError("S2Z partial map needs a finite positive scale, got $tau"))
    length(rho) == J ||
        throw(DimensionMismatch("S2Z partial map needs one weight per level"))
    all(r -> isfinite(r) && 0 <= r <= 1, rho) ||
        throw(ArgumentError("S2Z centering weights must lie in [0, 1]"))
    nothing
end

# Per-level interpolation scales of Sean's projected partial transform:
# d_j = 1 - rho_j + rho_j * tau. At rho = 0 the map is fully scaled, at
# rho = 1 it is the identity on the zero-sum subspace.
_s2z_partial_scales(tau::Real, rho::AbstractVector{<:Real}) =
    [1 - r + r * tau for r in rho]

"""
    _s2z_partial_forward(u, tau, rho) -> delta

Map a zero-sum level vector `u` to zero-sum deviations `delta` under Sean's
partial S2Z transform: `delta = tau * (w .- mean(w))` with `w = u ./ d` and
`d = 1 .- rho .+ rho .* tau`. At `rho = 0` this is `tau .* u`; at `rho = 1`
it is `u`.
"""
function _s2z_partial_forward(
        u::AbstractVector{<:Real}, tau::Real, rho::AbstractVector{<:Real})
    J = length(u)
    J >= 2 || throw(ArgumentError("S2Z partial map needs at least two levels (J >= 2)"))
    _s2z_check_partial_args(tau, rho, J)
    d = _s2z_partial_scales(tau, rho)
    all(>(0), d) || throw(ArgumentError("S2Z partial scales must be positive"))
    w = u ./ d
    tau .* (w .- sum(w) / J)
end

"""
    _s2z_partial_inverse(delta, tau, rho) -> u

Invert [`_s2z_partial_forward`](@ref) on the zero-sum subspace:
`u = d .* (z .- dot(d, z) / sum(d))` with `z = delta ./ tau`. The weighted
subtraction is the coupling an independent per-level rescaling would miss.
"""
function _s2z_partial_inverse(
        delta::AbstractVector{<:Real}, tau::Real, rho::AbstractVector{<:Real})
    J = length(delta)
    J >= 2 || throw(ArgumentError("S2Z partial map needs at least two levels (J >= 2)"))
    _s2z_check_partial_args(tau, rho, J)
    d = _s2z_partial_scales(tau, rho)
    all(>(0), d) || throw(ArgumentError("S2Z partial scales must be positive"))
    z = delta ./ tau
    t = dot(d, z) / sum(d)
    d .* (z .- t)
end

"""
    _s2z_partial_logjac(tau, rho) -> Real

Restricted log-Jacobian of the partial S2Z map on its `J - 1` free dimensions:
`(J-1)*log(tau) - sum(log.(d)) + log(mean(d))`. Follows from
`det(Q' * Diagonal(inv.(d)) * Q) = prod(inv.(d)) * mean(d)`; the shared
log-mean term is essential whenever level weights differ.
"""
function _s2z_partial_logjac(tau::Real, rho::AbstractVector{<:Real})
    J = length(rho)
    J >= 2 || throw(ArgumentError("S2Z partial map needs at least two levels (J >= 2)"))
    _s2z_check_partial_args(tau, rho, J)
    d = _s2z_partial_scales(tau, rho)
    all(>(0), d) || throw(ArgumentError("S2Z partial scales must be positive"))
    logd = log.(d)
    shift = maximum(logd)
    (J - 1) * log(tau) - sum(logd) + shift + log(sum(exp.(logd .- shift)) / J)
end

"""
    _s2z_fisher_candidate(infos, sd) -> Matrix

Per-(group, coefficient) posterior-vs-prior reliability fractions from one
draw's expected observation information, matching brms's
`rho_center_candidate` generated quantity (pinned PR #1919). `infos[j]` is
group `j`'s `M x M` expected-information matrix
(`sum_n obs_prec[n] * design[n] * design[n]'`) and `sd` the `M` prior scales.

Each group's whitened posterior covariance `W_j = (D*info_j*D + I)^{-1}` is
projected onto the S2Z constraint (`W_j - W_j * S^{-1} * W_j` with
`S = sum(W_j)`); the diagonal ratio against the restricted prior fraction
`1 - 1/J` gives a raw reliability in `[0, 1]`, rescaled by
`raw / (raw + (1 - raw) * sd[k])` into the interpolation-weight
parameterization the partial map consumes.
"""
function _s2z_fisher_candidate(
        infos::AbstractVector{<:AbstractMatrix{<:Real}}, sd::AbstractVector{<:Real})
    J = length(infos)
    J >= 2 || throw(ArgumentError("S2Z Fisher candidates need at least two groups (J >= 2)"))
    M = length(sd)
    M >= 1 || throw(ArgumentError("S2Z Fisher candidates need at least one coefficient"))
    all(s -> isfinite(s) && s > 0, sd) ||
        throw(ArgumentError("S2Z Fisher candidates need finite positive scales"))
    for (j, info) in enumerate(infos)
        size(info) == (M, M) || throw(DimensionMismatch(
            "S2Z Fisher info for group $j has size $(size(info)), expected ($M, $M)"))
        all(isfinite, info) ||
            throw(ArgumentError("S2Z Fisher info for group $j is not finite"))
    end
    scales = Vector{Float64}(vec(sd))
    white = Vector{Matrix{Float64}}(undef, J)
    total = zeros(Float64, M, M)
    for j in 1:J
        K = Symmetric(Matrix{Float64}(infos[j]))
        K = Symmetric(Diagonal(scales) * K * Diagonal(scales))
        K = Symmetric((K + K') / 2 + I)
        factor = cholesky!(Symmetric(Matrix{Float64}(K)))
        W = Symmetric(factor \ Matrix{Float64}(I, M, M))
        white[j] = Matrix{Float64}(W)
        total .+= white[j]
    end
    total = Symmetric((total + total') / 2)
    L = cholesky!(Symmetric(Matrix{Float64}(total))).L
    prior_fraction = 1 - inv(J)
    out = Matrix{Float64}(undef, J, M)
    for j in 1:J, k in 1:M
        G = L \ white[j]
        restricted = white[j][k, k] - dot(view(G, :, k), view(G, :, k))
        raw = clamp(1 - restricted / prior_fraction, 0.0, 1.0)
        out[j, k] = raw / (raw + (1 - raw) * scales[k])
    end
    out
end
