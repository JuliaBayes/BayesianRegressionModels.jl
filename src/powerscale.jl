# Prior/likelihood power-scaling sensitivity (Kallioinen et al., 2024).
#
# This file is BRM's equivalent of the ArviZ `psense_summary` / R priorsense
# `powerscale_sensitivity` analysis that the bambi `prior_sensitivity` notebook
# runs: perturb the prior (or likelihood) by raising it to a power alpha near
# 1, reweight the fitted draws by Pareto-smoothed importance sampling, and
# measure how far each posterior marginal moves (cumulative Jensen-Shannon
# distance gradient). Large prior movement means the posterior leans on the
# prior; large movement under BOTH perturbations flags prior-data conflict.
#
# Three layers, bottom-up:
#   1. `brm_psis_weights` — dependency-free Pareto smoothing (Vehtari et al.,
#      PSIS.jl conventions). No new Project dependency on purpose: the GPD tail
#      fit is ~100 lines, and `test/powerscale.jl` cross-checks it against
#      PSIS.jl, which already sits in the test environment.
#   2. `brm_cjs_dist` — the priorsense cumulative Jensen-Shannon distance,
#      ported exactly (base-2 logs, bound normalisation, unsigned max).
#   3. `brm_powerscale_sensitivity` — the per-variable summary over caller
#      supplied `(draws, log_prior, log_lik)` triples, plus
#      `brm_powerscale_inputs(::SBBRMI, ...)` which assembles those triples for
#      a fitted SBBRMI. The likelihood half comes from the emitted pointwise
#      log-likelihood twins; the prior half is evaluated in Julia for the
#      covered surface below, and anything outside it fails closed LOUDLY
#      rather than returning a silently incomplete joint prior.
#
# Covered prior surface (v1): scalar `name ~ Normal/Exponential(...)` priors
# with constant arguments, population `Normal` coefficients (default or
# `effect(...)` overrides), and categorical contrast/cell-mean blocks (default
# or `effect(...)` overrides). Covered likelihoods (v1): `Normal` and
# `BernoulliLogit` — response families that introduce no latent parameters of
# their own. Random effects, Gaussian processes, smooths, R2D2, mixtures, and
# every other prior/likelihood family error by name. Extending the surface is
# additive: admit one family plus its BridgeStan identity case in
# `test/powerscale.jl`.

# ---- Part 1: Pareto-smoothed importance weights ------------------------------

# Tail length rule, exactly as PSIS.jl `tail_length`: at most a fifth of the
# draws, at least ~3*sqrt(S/reff), whichever binds first.
function _brm_psis_tail_length(S::Int; reff::Real=1.0)
    max_length = cld(S, 5)
    (isfinite(reff) && reff > 0) || return max_length
    min_length = ceil(Int, 3 * sqrt(S / reff))
    min(max_length, min_length)
end

_brm_gpd_quantile(μ::Float64, σ::Float64, k::Float64, p::Float64) =
    μ + σ * (abs(k) < eps(Float64) ? -log1p(-p) : expm1(k * -log1p(-p)) / k)

# MLE of (sigma, k) given location mu and theta = k/sigma (Zhang & Stephens).
function _brm_gpd_mle_mu_theta(x::AbstractVector{Float64}, μ::Float64, θ::Float64)
    k = Statistics.mean(xi -> log1p(θ * (xi - μ)), x)
    (σ=k / θ, k)
end

# Log joint likelihood p(x | mu, theta) with k the MLE given theta and x.
function _brm_gpd_profile_loglik(μ::Float64, θ::Float64,
                                 x::AbstractVector{Float64}, n::Int)
    (; σ, k) = _brm_gpd_mle_mu_theta(x, μ, θ)
    -n * (log(σ) + k + 1)
end

# Posterior-mean theta-hat by quadrature over the empirical prior
# (Zhang & Stephens 2009, as in PSIS.jl `_fit_gpd_theta_empirical_bayes`).
function _brm_gpd_theta_eb(μ::Float64, xsorted::Vector{Float64})
    n = length(xsorted)
    μ_star = -inv(xsorted[n] - μ)
    σ_star = inv(6 * (xsorted[max(fld(n + 2, 4), 1)] - μ))
    npoints = 30 + floor(Int, sqrt(n))
    thetas = Vector{Float64}(undef, npoints)
    lls = Vector{Float64}(undef, npoints)
    for j in 1:npoints
        θ = _brm_gpd_quantile(μ_star, σ_star, 0.5, (j - 0.5) / npoints)
        thetas[j] = θ
        lls[j] = _brm_gpd_profile_loglik(μ, θ, xsorted, n)
    end
    total = LogExpFunctions.logsumexp(lls)
    sum(thetas[j] * exp(lls[j] - total) for j in 1:npoints)
end

# Empirical-Bayes GPD fit at location 0 with the weakly informative shape
# prior (PSIS.jl `fit_gpd` with `prior_adjusted=true`, `sorted=true`).
function _brm_fit_gpd(xsorted::Vector{Float64})
    xmin, xmax = first(xsorted), last(xsorted)
    if xmin ≈ xmax
        # Near-point support: the uniform (k = -1) solution.
        return (σ=xmax, k=-1.0)
    end
    θ_hat = _brm_gpd_theta_eb(0.0, xsorted)
    (; σ, k) = _brm_gpd_mle_mu_theta(xsorted, 0.0, θ_hat)
    n = length(xsorted)
    (σ=σ, k=(n * k + 10 * 0.5) / (n + 10))
end

# Smooth the ASCENDING largest-M+1-excluded tail in place (PSIS.jl `psis_tail!`
# with logu the cutoff). Returns the fitted shape k (NaN when the fit fails,
# exactly as PSIS.jl, leaving the tail unsmoothed).
function _brm_psis_smooth_tail!(tail::Vector{Float64}, logμ::Float64)
    logw_max = tail[end]
    μ_scaled = exp(logμ - logw_max)
    w_scaled = exp.(tail .- logw_max) .- μ_scaled
    (; σ, k) = _brm_fit_gpd(w_scaled)
    if isfinite(k)
        dist_σ, dist_k = σ, k
        for i in eachindex(tail)
            q = _brm_gpd_quantile(0.0, dist_σ, dist_k, (i - 0.5) / length(tail))
            tail[i] = min(log(q + μ_scaled), 0.0) + logw_max
        end
    end
    k
end

"""
    brm_psis_weights(log_ratios; reff=1.0)

Pareto-smooth importance-sampling log-ratios (Vehtari et al., JMLR 2021),
following PSIS.jl conventions. Returns a named tuple with normalized
`smoothed` weights in the input draw order:

- `log_weights` — Pareto-smoothed, log-normalised log weights;
- `weights` — `exp.(log_weights)`, summing to 1;
- `pareto_k` — the fitted generalised-Pareto shape; `k > 0.7` means the
  smoothed estimates are unreliable (NaN when the tail fit itself fails);
- `tail_length` — how many of the largest ratios were smoothed;
- `ess` — `reff / sum(weights .^ 2)`.

`reff` is the relative MCMC efficiency of the log-ratios (1.0 for
independent draws). Non-finite ratios are an input violation and error, as
does a draw count too small to fit a tail (fewer than 21 draws).
"""
function brm_psis_weights(log_ratios::AbstractVector; reff::Real=1.0)
    S = length(log_ratios)
    S > 0 || error("brm_psis_weights needs at least one log-ratio")
    all(isfinite, log_ratios) || error(
        "brm_psis_weights needs finite log-ratios; non-finite values mean " *
        "the log-density they came from is broken, so smooth nothing.")
    (isfinite(reff) && reff > 0) ||
        error("brm_psis_weights needs a finite positive `reff`, got $reff")
    logw = Float64.(vec(collect(log_ratios)))
    M = _brm_psis_tail_length(S; reff=Float64(reff))
    M >= 5 || error(
        "brm_psis_weights needs at least 21 draws to fit the Pareto tail " *
        "(got $S, tail length $M). Moment matching is not implemented; " *
        "collect more draws.")
    perm = partialsortperm(logw, (S - M):S)
    logμ = logw[perm[1]]
    tail_inds = perm[2:(M + 1)]
    tail = logw[tail_inds]
    k = _brm_psis_smooth_tail!(tail, logμ)
    logw[tail_inds] = tail
    logw .-= LogExpFunctions.logsumexp(logw)
    weights = exp.(logw)
    (; log_weights=logw, weights, pareto_k=k, tail_length=M,
       ess=Float64(reff) / sum(abs2, weights))
end

# ---- Part 2: cumulative Jensen-Shannon distance ------------------------------

# Weighted ECDF of ASCENDING `sx` with aligned weights at `v` (right
# continuous, matching priorsense `ewcdf`). `cw` is the cumulative weight
# vector, so each query is a binary search, not a scan.
_brm_ewcdf_at(sx::Vector{Float64}, cw::Vector{Float64}, total::Float64, v::Float64) =
    (i = searchsortedlast(sx, v); i == 0 ? 0.0 : cw[i] / total)

# One-sided CJS(P || Q) + CJS(Q || P), bound-normalised and square-rooted —
# the priorsense `.cjs_dist` body. `wx`/`wy` are normalised inside, exactly
# as there. A zero-width support (every draw identical on both sides) is a
# point mass compared against itself: distance 0. R priorsense returns NaN
# there (0/0 over an empty bin set); 0.0 is the mathematically correct value.
function _brm_cjs_onesided(sx::Vector{Float64}, wx::Vector{Float64},
                           sy::Vector{Float64}, wy::Vector{Float64})
    wx_sum, wy_sum = sum(wx), sum(wy)
    if sx == sy
        widths = diff(sx)
        all(iszero, widths) && return 0.0
        px = @view cumsum(wx ./ wx_sum)[1:(end - 1)]
        qx = @view cumsum(wy ./ wy_sum)[1:(end - 1)]
        bins = widths
    else
        nbins = max(length(sx), length(sy))
        lo = min(first(sx), first(sy))
        hi = max(last(sx), last(sy))
        step = (hi - lo) / (nbins - 1)
        cw_x = cumsum(wx)
        cw_y = cumsum(wy)
        px = [_brm_ewcdf_at(sx, cw_x, wx_sum, lo + (b - 1) * step)
              for b in 1:nbins]
        qx = [_brm_ewcdf_at(sy, cw_y, wy_sum, lo + (b - 1) * step)
              for b in 1:nbins]
        bins = fill(step, nbins)
    end
    px_int = sum(px .* bins)
    qx_int = sum(qx .* bins)
    cjs_pq = 0.5 / log(2) * (qx_int - px_int)
    cjs_qp = 0.5 / log(2) * (px_int - qx_int)
    for i in eachindex(px, qx, bins)
        pxi, qxi, w = px[i], qx[i], bins[i]
        # Terms with a zero outer mass are NaN (`0 * -Inf`) and dropped in R
        # by `na.rm = TRUE`; skip them rather than special-casing the log.
        pxi > 0 && (cjs_pq += w * pxi * (log2(pxi) - log2((pxi + qxi) / 2)))
        qxi > 0 && (cjs_qp += w * qxi * (log2(qxi) - log2((qxi + pxi) / 2)))
    end
    bound = px_int + qx_int
    # Zero bound happens only on the identical-support path, where it means
    # both weight vectors concentrate on the final draw: two point masses at
    # the same point, distance 0. (R priorsense NaNs here, like the constant
    # case above.)
    bound > 0 || return 0.0
    # The sum is mathematically nonnegative (it is a divergence), but
    # near-identical inputs can push it a few ulps below zero; R priorsense
    # NaNs there via sqrt. Clamp: a divergence of -1e-16 is 0.
    sqrt(max((cjs_pq + cjs_qp) / bound, 0.0))
end

function _brm_cjs_weights(w, n::Int, what::String)
    isnothing(w) && return fill(1.0 / n, n)
    wv = Float64.(vec(collect(w)))
    length(wv) == n || throw(DimensionMismatch(
        "brm_cjs_dist: $what has $(length(wv)) weights for $n draws"))
    all(isfinite, wv) || error("brm_cjs_dist: $what must be finite")
    all(>=(0), wv) || error("brm_cjs_dist: $what must be nonnegative")
    sum(wv) > 0 || error(
        "brm_cjs_dist: $what is all zero — the perturbation destroyed every draw")
    wv
end

"""
    brm_cjs_dist(x, y; x_weights=nothing, y_weights=nothing)

Cumulative Jensen-Shannon distance between two samples, ported exactly from R
priorsense `cjs_dist` (base-2 logs, upper-bound normalisation, square-root
metric, unsigned max over `x`/`-x`). Weights default to uniform. `x` and `y`
are usually the SAME posterior draws with uniform versus power-scaling
weights, in which case the weighted-ECDF fast path applies.

Inputs must be finite with at least two draws each; weights must be finite,
nonnegative, and not all zero. A point mass compared against itself is
distance 0.0 (R priorsense returns NaN there, an arithmetic artifact).
"""
function brm_cjs_dist(x::AbstractVector, y::AbstractVector;
                      x_weights=nothing, y_weights=nothing)
    xv = Float64.(vec(collect(x)))
    yv = Float64.(vec(collect(y)))
    length(xv) >= 2 || error("brm_cjs_dist needs at least two draws in `x`")
    length(yv) >= 2 || error("brm_cjs_dist needs at least two draws in `y`")
    all(isfinite, xv) || error("brm_cjs_dist needs finite draws in `x`")
    all(isfinite, yv) || error("brm_cjs_dist needs finite draws in `y`")
    wx = _brm_cjs_weights(x_weights, length(xv), "`x_weights`")
    wy = _brm_cjs_weights(y_weights, length(yv), "`y_weights`")
    # The unsigned max needs its OWN sort: negating a sorted vector reverses
    # it, and the one-sided body requires ascending inputs (priorsense
    # re-sorts inside each call for the same reason).
    fwd = _brm_cjs_sorted(xv, wx, yv, wy)
    bwd = _brm_cjs_sorted(-xv, wx, -yv, wy)
    max(_brm_cjs_onesided(fwd...), _brm_cjs_onesided(bwd...))
end

function _brm_cjs_sorted(xv::Vector{Float64}, wx::Vector{Float64},
                          yv::Vector{Float64}, wy::Vector{Float64})
    ox, oy = sortperm(xv), sortperm(yv)
    (xv[ox], wx[ox], yv[oy], wy[oy])
end

# ---- Part 3: power-scaling perturbations and the sensitivity summary ---------

"""
    brm_powerscale_weights(log_component; alpha, reff=1.0)

Pareto-smoothed importance weights for power-scaling one posterior component
(the joint log-prior or joint log-likelihood, one value per draw) by `alpha`:
`w ∝ exp((alpha - 1) * log_component)`. Returns the
[`brm_psis_weights`](@ref) named tuple plus the `alpha` it was computed at.

`alpha == 1` is the base posterior: uniform weights with `pareto_k = -Inf`,
exactly as priorsense. A constant component errors (there is nothing to
scale — priorsense errors here too).
"""
function brm_powerscale_weights(log_component::AbstractVector; alpha::Real,
                                reff::Real=1.0)
    alpha = Float64(alpha)
    alpha >= 0 || error("brm_powerscale_weights needs `alpha >= 0`, got $alpha")
    ℓ = Float64.(vec(collect(log_component)))
    isempty(ℓ) && error("brm_powerscale_weights needs at least one draw")
    all(isfinite, ℓ) || error(
        "brm_powerscale_weights needs finite log-densities; non-finite " *
        "values mean the fit they came from is broken, so scale nothing.")
    if alpha == 1
        n = length(ℓ)
        u = fill(1.0 / n, n)
        return (; log_weights=log.(u), weights=u, pareto_k=-Inf,
                  tail_length=0, ess=Float64(n), alpha)
    end
    # priorsense `is_constant` on the log-ratios (tolerance eps).
    maximum(ℓ) - minimum(ℓ) < eps(Float64) && error(
        "brm_powerscale_weights: the log-component is constant, so " *
        "power-scaling cannot move the posterior. A uniform prior (or a " *
        "likelihood that ignores the data) is the usual cause — power-scaling " *
        "is undefined there.")
    out = brm_psis_weights((alpha - 1) .* ℓ; reff)
    (; out..., alpha)
end

"""
    BRMPowerscaleSensitivity

Per-variable power-scaling sensitivity summary, the
[`brm_powerscale_sensitivity`](@ref) value. `prior[i]` / `likelihood[i]` are
the CJS-distance gradients of `variables[i]` under prior / likelihood
power-scaling; `diagnosis[i]` is one of `:none`, `:prior_data_conflict`
(both at or above `threshold`), or `:strong_prior_weak_likelihood` (prior
above, likelihood below). `pareto_k` carries the four smoothing diagnostics
`(prior_lower, prior_upper, likelihood_lower, likelihood_upper)`; any value
above 0.7 (or NaN) makes that component's row untrustworthy, reported by
`reliable` per component.
"""
struct BRMPowerscaleSensitivity
    variables::Vector{Symbol}
    prior::Vector{Float64}
    likelihood::Vector{Float64}
    diagnosis::Vector{Symbol}
    pareto_k::NamedTuple
    reliable::NamedTuple
    threshold::Float64
    lower_alpha::Float64
    upper_alpha::Float64
end

function Base.show(io::IO, ::MIME"text/plain", s::BRMPowerscaleSensitivity)
    println(io, "Power-scaling sensitivity (CJS gradient; threshold " *
                "$(s.threshold), α ∈ [$(s.lower_alpha), $(s.upper_alpha)])")
    println(io, rpad("variable", 24), rpad("prior", 10), rpad("likelihood", 12),
            "diagnosis")
    for i in eachindex(s.variables, s.prior, s.likelihood, s.diagnosis)
        diag = s.diagnosis[i] === :prior_data_conflict ?
            "potential prior-data conflict" :
            s.diagnosis[i] === :strong_prior_weak_likelihood ?
            "potential strong prior / weak likelihood" : "-"
        println(io, rpad(string(s.variables[i]), 24),
                rpad(string(round(s.prior[i]; digits=4)), 10),
                rpad(string(round(s.likelihood[i]; digits=4)), 12), diag)
    end
    k = s.pareto_k
    print(io, "Pareto-k: prior (lo $(round(k.prior_lower; digits=2)), " *
              "hi $(round(k.prior_upper; digits=2)))" *
              (s.reliable.prior ? "" : " UNRELIABLE") *
              ", likelihood (lo $(round(k.likelihood_lower; digits=2)), " *
              "hi $(round(k.likelihood_upper; digits=2)))" *
              (s.reliable.likelihood ? "" : " UNRELIABLE"))
end

_brm_powerscale_select(names, variables) =
    Symbol.(isnothing(variables) ? names : variables)

function _brm_powerscale_indices(names, variables)
    strings = string.(names)
    have = _brm_powerscale_select(names, variables)
    isempty(have) && error("brm_powerscale_sensitivity needs at least one variable")
    idx = map(have) do v
        i = findfirst(==(string(v)), strings)
        isnothing(i) && error(
            "brm_powerscale_sensitivity: variable `$v` is not among the " *
            "draw columns. Available: " * join(strings[1:min(end, 12)], ", ") *
            (length(strings) > 12 ? ", … ($(length(strings)) total)" : ""))
        i
    end
    length(unique(idx)) == length(idx) ||
        error("brm_powerscale_sensitivity: duplicate variable selection")
    have, idx
end

"""
    brm_powerscale_sensitivity(draws, log_prior, log_lik, names; variables=nothing,
                               lower_alpha=0.99, upper_alpha=1.01, threshold=0.05)

Prior/likelihood power-scaling sensitivity over fitted posterior draws —
BRM's equivalent of ArviZ `psense_summary` / priorsense
`powerscale_sensitivity`. `draws` is draws × coordinates (the repository
convention, NOT WarmupHMC's orientation), `names` its column names, and
`log_prior` / `log_lik` the per-draw JOINT log-prior / log-likelihood.
`variables` selects columns by name (default: all).

For each variable and each component, the draws are reweighted by
[`brm_powerscale_weights`](@ref) at `lower_alpha` and `upper_alpha`, the
[`brm_cjs_dist`](@ref) distance to the base marginal is measured on both
sides, and the reported sensitivity is the mean one-sided gradient
(`D_hi / log2(upper_alpha)` and `-D_lo / log2(lower_alpha)` averaged),
exactly as priorsense `powerscale_sensitivity`. Values at or above
`threshold` (default 0.05, the Kallioinen et al. cutoff) diagnose as
documented on [`BRMPowerscaleSensitivity`](@ref).

Hierarchical models need selective power-scaling (top-level priors only —
see the EABM sensitivity chapter). This summary always scales the JOINT
densities it is given; pass joint densities restricted to the levels you
mean, or use the documented refit comparison for hierarchical priors until
selective scaling ships.
"""
function brm_powerscale_sensitivity(draws::AbstractMatrix,
                                    log_prior::AbstractVector,
                                    log_lik::AbstractVector,
                                    names;
                                    variables=nothing,
                                    lower_alpha::Real=0.99,
                                    upper_alpha::Real=1.01,
                                    threshold::Real=0.05)
    _brm_check_draw_names(draws, names)
    lower_alpha = Float64(lower_alpha)
    upper_alpha = Float64(upper_alpha)
    threshold = Float64(threshold)
    0 < lower_alpha < 1 ||
        error("brm_powerscale_sensitivity needs `0 < lower_alpha < 1`")
    upper_alpha > 1 ||
        error("brm_powerscale_sensitivity needs `upper_alpha > 1`")
    threshold >= 0 ||
        error("brm_powerscale_sensitivity needs `threshold >= 0`")
    S = size(draws, 1)
    ℓp = Float64.(vec(collect(log_prior)))
    ℓl = Float64.(vec(collect(log_lik)))
    length(ℓp) == S || throw(DimensionMismatch(
        "brm_powerscale_sensitivity: `log_prior` has $(length(ℓp)) values " *
        "for $S draws"))
    length(ℓl) == S || throw(DimensionMismatch(
        "brm_powerscale_sensitivity: `log_lik` has $(length(ℓl)) values " *
        "for $S draws"))
    have, idx = _brm_powerscale_indices(names, variables)
    # Weights depend on the component and alpha only — compute the four
    # vectors once, then score every variable against them.
    w = Dict{Tuple{Symbol,Float64},Any}()
    khat = Dict{Tuple{Symbol,Float64},Float64}()
    for (comp, ℓ) in ((:prior, ℓp), (:likelihood, ℓl))
        for α in (lower_alpha, upper_alpha)
            out = brm_powerscale_weights(ℓ; alpha=α)
            w[(comp, α)] = out.weights
            khat[(comp, α)] = out.pareto_k
        end
    end
    log_lo = log2(lower_alpha)
    log_hi = log2(upper_alpha)
    prior = Vector{Float64}(undef, length(idx))
    likelihood = Vector{Float64}(undef, length(idx))
    for (j, i) in enumerate(idx)
        x = Float64.(vec(collect(@view draws[:, i])))
        for (comp, store) in ((:prior, prior), (:likelihood, likelihood))
            d_lo = brm_cjs_dist(x, x; y_weights=w[(comp, lower_alpha)])
            d_hi = brm_cjs_dist(x, x; y_weights=w[(comp, upper_alpha)])
            store[j] = ((-d_lo / log_lo) + (d_hi / log_hi)) / 2
        end
    end
    # priorsense's exact comparisons, asymmetry included: conflict needs both
    # AT the threshold, strong-prior needs prior strictly ABOVE it.
    diagnosis = map(eachindex(have)) do j
        p, l = prior[j], likelihood[j]
        (p >= threshold && l >= threshold) ? :prior_data_conflict :
        (p > threshold && l < threshold) ? :strong_prior_weak_likelihood : :none
    end
    k_tuple = (; prior_lower=khat[(:prior, lower_alpha)],
                 prior_upper=khat[(:prior, upper_alpha)],
                 likelihood_lower=khat[(:likelihood, lower_alpha)],
                 likelihood_upper=khat[(:likelihood, upper_alpha)])
    reliable = (prior=all(k -> isfinite(k) && k <= 0.7,
                           (k_tuple.prior_lower, k_tuple.prior_upper)),
                likelihood=all(k -> isfinite(k) && k <= 0.7,
                               (k_tuple.likelihood_lower, k_tuple.likelihood_upper)))
    BRMPowerscaleSensitivity(have, prior, likelihood, diagnosis, k_tuple,
                             reliable, threshold, lower_alpha, upper_alpha)
end

# ---- Part 4: SBBRMI assembly -------------------------------------------------

# Covered surface (v1). Scalar priors admit exactly these families; likelihoods
# admit exactly these response families (both introduce no latent parameters of
# their own, so the Julia-side joint prior is COMPLETE for an accepted model —
# up to an additive constant, which importance weights normalise away). Term
# heads that own sampled parameters outside this evaluator are rejected by
# name; deterministic transforms (log/exp/*, offset, center/zscale/...) pass.
const _BRM_POWERSCALE_PRIOR_FAMILIES = (Normal, Exponential)
const _BRM_POWERSCALE_LIKELIHOODS = (Normal, BernoulliLogit)
const _BRM_POWERSCALE_REJECT_HEADS =
    ((|), doublepipe, gr, mm, gp, hsgp, s, t2, ar, dar, rw, cdar, mo, mo1,
     kernel, ragged, mi, me, interval_censored, LKJCovarianceFactor,
     MvNormalCholesky, effect, r2d2)

function _brm_powerscale_collect_heads!(heads::Set, node)
    if node isa ExprColumn
        push!(heads, getf(node))
        for a in getargs(node)
            _brm_powerscale_collect_heads!(heads, a)
        end
        for v in values(getkwargs(node))
            _brm_powerscale_collect_heads!(heads, v)
        end
    elseif node isa MultiMembershipTerm
        push!(heads, mm)
    elseif node isa JointResponseColumn
        push!(heads, JointResponseColumn)
    end
    heads
end

_brm_powerscale_head_name(h) =
    h isa Type ? string(nameof(h)) :
    h isa Function ? string(nameof(h)) : string(h)

# Fail-closed coverage inventory over the whole BRMI. Every `~` operation is
# classified (likelihood / prior address / scalar prior / linear predictor)
# and anything the Julia evaluator below cannot price errors by NAME. An
# accepted model has a provably complete joint prior; the BridgeStan identity
# battery in `test/powerscale.jl` proves it per admitted shape.
function _brm_powerscale_gate!(brmi::BRMI)
    for spec in r2d2_priors(brmi)
        error("brm_powerscale_inputs: `$(spec.predictor)` carries an `r2d2` " *
              "variance decomposition, whose joint prior the Julia evaluator " *
              "does not cover yet. Drop the decomposition or use the refit " *
              "comparison from brm-use until selective scaling ships.")
    end
    for spec in ranef_effect_priors(brmi)
        error("brm_powerscale_inputs: `$(spec.class)(:, $(spec.id))` " *
              "addresses a shared random-effect covariance the Julia " *
              "evaluator does not cover yet (and hierarchical priors need " *
              "selective top-level-only scaling — see the EABM sensitivity " *
              "chapter). Use the refit comparison from brm-use instead.")
    end
    isempty(term_priors(brmi)) || error(
        "brm_powerscale_inputs: term-addressed priors " *
        "(`sd`/`ar`/`simplex`/`latent`/`length_scale` on s/t2/gp/hsgp/...) " *
        "are outside the covered surface. Use the refit comparison instead.")
    for spec in effect_priors(brmi)
        if spec.family !== Normal || !isempty(spec.keywords)
            error("brm_powerscale_inputs: `effect($(spec.predictor), " *
                  "$(spec.coefficient))` carries a non-Normal (or keyword) " *
                  "prior the Julia evaluator does not cover. Only " *
                  "`effect(...) ~ Normal(location, scale)` is supported.")
        end
    end
    for (key, op_nc) in pairs(brmi.operations)
        op = _named_op(op_nc)
        isnothing(op) && continue
        getf(op) === (~) || continue
        lhs, rhs = getargs(op, 2)
        if !isnothing(_observed_lhs_or_nothing(lhs))
            _brm_powerscale_gate_likelihood!(key, lhs, rhs)
        elseif lhs isa ExprColumn && getf(lhs) === effect
            continue
        elseif _brm_is_prior_declaration(brmi, key)
            _brm_powerscale_gate_scalar!(key, lhs, rhs)
        else
            _brm_powerscale_gate_predictor!(key, rhs)
        end
    end
    nothing
end

function _brm_powerscale_gate_likelihood!(key::Symbol, lhs, rhs)
    plain = lhs isa NamedColumn && parent(lhs) isa DataColumn
    plain || error("brm_powerscale_inputs: response `$key` has a decorated " *
                   "LHS (link / `mi` / `ragged` / joint), outside the " *
                   "covered surface. Use the refit comparison instead.")
    rhs isa ExprColumn && getf(rhs) in _BRM_POWERSCALE_LIKELIHOODS || error(
        "brm_powerscale_inputs: response `$key` uses " *
        "`$(_brm_powerscale_head_name(rhs isa ExprColumn ? getf(rhs) : rhs))`, " *
        "outside the covered likelihoods (`Normal`, `BernoulliLogit`). " *
        "Use the refit comparison instead.")
    nothing
end

function _brm_powerscale_gate_scalar!(key::Symbol, lhs, rhs)
    lhs isa NamedColumn || error(
        "brm_powerscale_inputs: prior `$key` has a non-bare LHS, outside " *
        "the covered surface.")
    rhs isa ExprColumn && getf(rhs) in _BRM_POWERSCALE_PRIOR_FAMILIES || error(
        "brm_powerscale_inputs: prior `$key` uses " *
        "`$(_brm_powerscale_head_name(rhs isa ExprColumn ? getf(rhs) : rhs))`, " *
        "outside the covered scalar families (`Normal`, `Exponential`).")
    for k in keys(getkwargs(rhs))
        k in (:lower, :upper) || error(
            "brm_powerscale_inputs: prior `$key` carries keyword `$k`; only " *
            "support bounds (`lower`/`upper`) are covered.")
        isnothing(_brm_numeric_constant(getkwargs(rhs)[k])) && error(
            "brm_powerscale_inputs: prior `$key` has a non-constant `$k` " *
            "bound; only constant support bounds are covered.")
    end
    for a in getargs(rhs)
        isnothing(_brm_numeric_constant(a)) && error(
            "brm_powerscale_inputs: prior `$key` has a non-constant argument " *
            "`$a` (data- or parameter-dependent). Hierarchical priors need " *
            "selective scaling; data-dependent scales are not resolved yet. " *
            "Use the refit comparison instead.")
    end
    nothing
end

function _brm_powerscale_gate_predictor!(key::Symbol, rhs)
    heads = _brm_powerscale_collect_heads!(Set{Any}(), rhs)
    for h in _BRM_POWERSCALE_REJECT_HEADS
        h in heads && error(
            "brm_powerscale_inputs: predictor `$key` contains " *
            "`$(_brm_powerscale_head_name(h))`, whose parameters the Julia " *
            "evaluator does not cover yet. Use the refit comparison from " *
            "brm-use instead.")
    end
    JointResponseColumn in heads && error(
        "brm_powerscale_inputs: predictor `$key` contains a joint response, " *
        "outside the covered surface.")
    nothing
end

# ---- Part 4b: the Julia-side joint prior -------------------------------------

# Numeric `(location, scale)` from a Normal effect/scalar RHS, mirroring the
# emission-side arg handling (`_brm_normal_effect_args` defaults, constant
# folding). Anything non-constant or non-positive-scale fails closed.
function _brm_powerscale_normal_args(rhs::ExprColumn, what::String)
    _sb_is_normal_effect_prior(rhs) || error(
        "brm_powerscale_inputs: $what is not a `Normal` prior " *
        "(internal coverage error — the gate should have rejected it)")
    raw = _brm_normal_effect_args(rhs; prefix="brm_powerscale_inputs")
    loc = _brm_numeric_constant(raw[1])
    scale = _brm_numeric_constant(raw[2])
    (isnothing(loc) || isnothing(scale)) && error(
        "brm_powerscale_inputs: $what has a non-constant location/scale; " *
        "only constant `Normal(location, scale)` is covered.")
    isfinite(loc) && isfinite(scale) && scale > 0 || error(
        "brm_powerscale_inputs: $what has invalid location/scale " *
        "($loc, $scale)")
    Float64(loc), Float64(scale)
end

function _brm_powerscale_scalar_dist(rhs::ExprColumn, key::Symbol)
    F = getf(rhs)
    args = getargs(rhs)
    if F === Normal
        length(args) <= 2 || error(
            "brm_powerscale_inputs: prior `$key` has $(length(args)) " *
            "arguments; `Normal` takes at most location and scale.")
        loc = isempty(args) ? 0.0 : _brm_numeric_constant(args[1])
        scale = length(args) < 2 ? 1.0 : _brm_numeric_constant(args[2])
        (isnothing(loc) || isnothing(scale)) && error(
            "brm_powerscale_inputs: prior `$key` has a non-constant " *
            "location/scale (internal coverage error)")
        isfinite(loc) && isfinite(scale) && scale > 0 || error(
            "brm_powerscale_inputs: prior `$key` has invalid location/scale " *
            "($loc, $scale)")
        dist = Normal(Float64(loc), Float64(scale))
    elseif F === Exponential
        length(args) <= 1 || error(
            "brm_powerscale_inputs: prior `$key` has $(length(args)) " *
            "arguments; `Exponential` takes at most a scale.")
        scale = isempty(args) ? 1.0 : _brm_numeric_constant(only(args))
        isnothing(scale) && error(
            "brm_powerscale_inputs: prior `$key` has a non-constant scale " *
            "(internal coverage error)")
        isfinite(scale) && scale > 0 || error(
            "brm_powerscale_inputs: prior `$key` has invalid scale ($scale)")
        # Julia parameterisation (mean `scale`); Stan lowering translates to
        # rate, so this matches the fitted density exactly.
        dist = Exponential(Float64(scale))
    else
        error("brm_powerscale_inputs: prior `$key` uses `$F` (internal " *
              "coverage error — the gate should have rejected it)")
    end
    kw = getkwargs(rhs)
    for k in (:lower, :upper)
        haskey(kw, k) || continue
        v = _brm_numeric_constant(kw[k])
        (isnothing(v) || !isfinite(v)) && error(
            "brm_powerscale_inputs: prior `$key` has a non-constant `$k` " *
            "bound (internal coverage error)")
    end
    lower = haskey(kw, :lower) ? Float64(_brm_numeric_constant(kw[:lower])) : -Inf
    upper = haskey(kw, :upper) ? Float64(_brm_numeric_constant(kw[:upper])) : Inf
    dist, lower, upper
end

# Per-contrast `(location, scale)` for one categorical block: the shared
# override (treatment), the per-level vector (cell means), or the `Normal(0,1)`
# default. `coords` order is the fitted level order on both paths.
function _brm_powerscale_cat_priors(override, n::Int, col::Symbol, lp::Symbol)
    if isnothing(override)
        return fill((0.0, 1.0), n)
    elseif override isa ExprColumn
        loc, scale = _brm_powerscale_normal_args(
            override, "categorical block `$col` on `$lp`")
        return fill((loc, scale), n)
    elseif override isa AbstractVector
        length(override) == n || error(
            "brm_powerscale_inputs: cell-mean block `$col` on `$lp` has $n " *
            "coordinates but $(length(override)) prior cells — descriptor " *
            "drift; re-reflect the model that produced the draws.")
        return map(enumerate(override)) do (level, cell)
            isnothing(cell) ? (0.0, 1.0) :
                _brm_powerscale_normal_args(
                    cell, "cell-mean level $level of `$col` on `$lp`")
        end
    end
    error("brm_powerscale_inputs: categorical block `$col` on `$lp` has an " *
          "unrecognised prior value (internal coverage error)")
end

function _brm_powerscale_log_prior(brmi::BRMI, d::BRMDescriptor,
                                   draws::AbstractMatrix, names)
    S = size(draws, 1)
    acc = zeros(Float64, S)
    for (key, op_nc) in pairs(brmi.operations)
        _brm_is_prior_declaration(brmi, key) || continue
        op = _named_op(op_nc)
        lhs, rhs = getargs(op, 2)
        lhs isa NamedColumn || error(
            "brm_powerscale_inputs: prior `$key` has a non-bare LHS " *
            "(internal coverage error — the gate should have rejected it)")
        target = name(lhs)
        coords = brm_output_coordinates(d, target, names)
        length(coords) == 1 || error(
            "brm_powerscale_inputs: prior `$target` resolves to " *
            "$(length(coords)) coordinates; only scalar priors are covered.")
        dist, lower, upper = _brm_powerscale_scalar_dist(rhs, key)
        col = @view draws[:, only(coords)]
        for i in 1:S
            x = col[i]
            acc[i] += (x < lower || x > upper) ? -Inf : logpdf(dist, x)
        end
    end
    effect_overrides = _sb_effect_prior_overrides(brmi)
    for entry in _brm_population_effect_entries(brmi)
        lp = entry.logical
        labels = try
            popcoefnames(brmi, lp)
        catch err
            error("brm_powerscale_inputs: cannot name the population " *
                  "columns of `$lp`: $(sprint(showerror, err))")
        end
        isnothing(labels) && error(
            "brm_powerscale_inputs: `$lp` has no population columns " *
            "(internal coverage error)")
        col_overrides = _sb_pop_effect_overrides(effect_overrides, lp)
        if !isnothing(col_overrides) && length(col_overrides) != length(labels)
            error("brm_powerscale_inputs: `$lp` has $(length(labels)) " *
                  "population labels but $(length(col_overrides)) prior " *
                  "cells (internal coverage error)")
        end
        for (i, label) in enumerate(labels)
            cell = isnothing(col_overrides) ? nothing : col_overrides[i]
            loc, scale = isnothing(cell) ? (0.0, 1.0) :
                _brm_powerscale_normal_args(cell, "`effect($lp, $label)`")
            res = brm_population_effect_coordinates(d, lp, names;
                                                   coefficient=label)
            length(res.coordinates) == 1 || error(
                "brm_powerscale_inputs: coefficient `$label` on `$lp` " *
                "resolves to $(length(res.coordinates)) coordinates; " *
                "re-reflect the model that produced the draws.")
            acc .+= logpdf.(Normal(loc, scale),
                            @view draws[:, only(res.coordinates)])
        end
        cat_map = try
            _sb_cat_address_map(brmi, lp)
        catch err
            error("brm_powerscale_inputs: cannot map the categorical blocks " *
                  "of `$lp`: $(sprint(showerror, err))")
        end
        block_overrides = _sb_cat_effect_overrides(effect_overrides, lp)
        for (col, block) in cat_map
            res = brm_population_effect_coordinates(d, lp, names; coefficient=col)
            priors = _brm_powerscale_cat_priors(
                get(block_overrides, block, nothing),
                length(res.coordinates), col, lp)
            for (j, c) in enumerate(res.coordinates)
                loc, scale = priors[j]
                acc .+= logpdf.(Normal(loc, scale), @view draws[:, c])
            end
        end
    end
    acc
end

# ---- Part 4c: joint log-likelihood from the pointwise twins ------------------

function _brm_powerscale_log_lik(d::BRMDescriptor, draws::AbstractMatrix, names)
    S = size(draws, 1)
    twins = brm_outputs(d; role=:pointwise_loglik)
    isempty(twins) && error(
        "brm_powerscale_inputs: this model emits no pointwise log-likelihood " *
        "twins, so the joint log-likelihood cannot be assembled. Models " *
        "whose likelihood was fully held out have nothing to scale.")
    acc = zeros(Float64, S)
    seen = Set{Symbol}()
    for t in twins
        logical = isnothing(t.logical) ? t.source : t.logical
        isnothing(logical) && error(
            "brm_powerscale_inputs: pointwise twin `$(t.name)` has no " *
            "logical response; re-reflect the model.")
        logical in seen && continue
        push!(seen, logical)
        coords = try
            brm_output_coordinates(d, logical, names; role=:pointwise_loglik)
        catch err
            error("brm_powerscale_inputs: cannot resolve the pointwise " *
                  "log-likelihood twin for `$logical` in the draw columns " *
                  "— the fit must SAVE generated quantities " *
                  "(`BridgeStan.param_names(...; include_gq=true)` plus the " *
                  "matching draws). Underlying error: $(sprint(showerror, err))")
        end
        for c in coords
            acc .+= @view draws[:, c]
        end
    end
    acc
end

# Default sensitivity variables: every draw column except the predictive and
# pointwise twins (they are per-observation quantities, not parameters) and
# Stan sampler diagnostics (`lp__`, ...). Transformed parameters stay in —
# priorsense likewise scores every variable except its log-density groups.
function _brm_powerscale_default_variables(d::BRMDescriptor, names)
    twin_idx = Set{Int}()
    for role in (:posterior_predictive, :pointwise_loglik)
        for t in brm_outputs(d; role)
            logical = isnothing(t.logical) ? t.source : t.logical
            isnothing(logical) && continue
            coords = try
                brm_output_coordinates(d, logical, names; role)
            catch
                continue
            end
            union!(twin_idx, coords)
        end
    end
    map(Symbol, (string(names[i]) for i in eachindex(names)
                if !(i in twin_idx) && !endswith(string(names[i]), "__")))
end

"""
    brm_powerscale_inputs(sb::SBBRMI, constrained_draws, constrained_names)

Assemble the `(draws, log_prior, log_lik)` triple
[`brm_powerscale_sensitivity`](@ref) needs from a fitted SBBRMI. Returns
`(; log_prior, log_lik, variables, descriptor)` where `log_prior` /
`log_lik` are per-draw joint densities, `variables` the default sensitivity
selection (every parameter column — twins and `__` diagnostics excluded),
and `descriptor` the `brm_descriptor(sb)` the assembly ran through.

`constrained_draws` is draws × coordinates with `constrained_names` from the
SAME transformed-parameter / generated-quantity settings, and the fit MUST
have saved generated quantities (the pointwise log-likelihood twins live
there). Models outside the covered surface in `src/powerscale.jl` (random
effects, GPs, smooths, R2D2, non-Normal/non-Exponential priors, …) error by
name — a silently incomplete joint prior would be a wrong sensitivity, so
there is no partial credit. Pair with the refit comparison from brm-use for
those models.
"""
function brm_powerscale_inputs(sb::SBBRMI, constrained_draws::AbstractMatrix,
                               constrained_names)
    _brm_check_draw_names(constrained_draws, constrained_names)
    brmi = sb.parent
    _brm_powerscale_gate!(brmi)
    d = brm_descriptor(sb)
    log_prior = _brm_powerscale_log_prior(brmi, d, constrained_draws,
                                          constrained_names)
    log_lik = _brm_powerscale_log_lik(d, constrained_draws, constrained_names)
    variables =
        _brm_powerscale_default_variables(d, constrained_names)
    (; log_prior, log_lik, variables, descriptor=d)
end
