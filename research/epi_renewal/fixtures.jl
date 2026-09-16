# Pure-Julia fixtures for translating the Wren.jl PR #16 multi-patch renewal
# model (docs/src/examples/multi-patch-renewal.md @ epi-example 6e3fd026) into
# native @brm. No BRM / StanBlocks here.
#
# Two jobs:
#   1. SIMULATE data from the PR's own generative process with FIXED truth
#      parameters (deterministic given the seeds), single-patch and six-patch.
#   2. Keep the PR's process functions (ported near-verbatim) as the Julia
#      REFERENCE, so the Stan @deffun translations can be cross-checked against
#      them value-for-value.
#
# Delay PMFs: the PR builds them with CensoredDistributions.double_interval_censored;
# that package is not in the BRM test env, so `censored_pmf` evaluates the same
# doubly-interval-censored mass function directly (quadrature over the primary
# event time). The reporting delay is taken at the PR's TRUE LogNormal(1.5, 0.5)
# rather than the PR's linelist-fit posterior means (which recover it); the
# generation time is the PR's Gamma(6.5, 0.62), D = 14, day-zero mass dropped.
#
# Run: julia --project=test research/epi_renewal/fixtures.jl   (prints a summary)

using Distributions, LinearAlgebra, Random, Statistics

# ── delay distributions ──────────────────────────────────────────────────────

"""
    censored_pmf(dist; D, drop_zero=false, nq=400)

Doubly-interval-censored daily mass function — the semantics of
`pdf(double_interval_censored(dist; upper=D, interval=1.0), d)` for d = 0..D-1:

    p_d = P(d <= P + T < d + 1 | P + T <= D),   P ~ Uniform(0, 1),  T ~ dist,

where P is the primary event's time within its day and T the delay. `P(P+T <= x)`
is `∫_0^1 F_T(x - p) dp`, evaluated by midpoint quadrature (`nq` points).
`drop_zero` removes the day-0 mass and renormalises (generation time: a same-day
term would put I_t on both sides of the renewal equation). Returns a `Vector` of
length `D` (or `D - 1` with `drop_zero`), index `t` = mass on day `t - 1`
(or day `t` with `drop_zero`).
"""
function censored_pmf(dist; D, drop_zero=false, nq=400)
    ps = ((1:nq) .- 0.5) ./ nq                                     # midpoints of [0, 1]
    FS(x) = x <= 0 ? 0.0 : mean(cdf(dist, max(x - p, 0.0)) for p in ps)   # P(P + T <= x)
    pmf = [(FS(Float64(t)) - FS(Float64(t - 1))) / FS(Float64(D)) for t in 1:D]  # mass on [t-1, t)
    drop_zero && (pmf = pmf[2:end] ./ sum(pmf[2:end]))
    return pmf
end

gen_pmf(; D=14) = censored_pmf(Gamma(6.5, 0.62); D, drop_zero=true)        # lags 1..13
delay_pmf(; mu=1.5, sigma=0.5, D=15) = censored_pmf(LogNormal(mu, sigma); D)  # lags 0..14

# ── the PR's process functions (Julia reference) ─────────────────────────────

"Growth rate r implied by R through the generation time: R * sum_s g_s e^{-r s} = 1 (Newton)."
function R_to_r(R, gen_pmf; newton_steps=2)
    mean_gen = zero(eltype(gen_pmf))
    for (s, g) in enumerate(gen_pmf)
        mean_gen += s * g
    end
    r = clamp((R - 1) / (R * mean_gen), -2.0, 2.0)
    for _ in 1:newton_steps
        f = zero(r)
        df = zero(r)
        for (s, g) in enumerate(gen_pmf)
            e = exp(-r * s)
            f += g * e
            df -= s * g * e
        end
        r = clamp(r - (R * f - 1) / (R * df), -2.0, 2.0)
    end
    return r
end

"Sum of `pmf`-weighted lags of `x` at `t`, reaching into the seeded history x0 * e^{r u}, u <= 0."
function lagged_sum(x, t, pmf, x0, r, first_lag)
    acc = zero(eltype(x))
    for (i, w) in enumerate(pmf)
        s = first_lag + i - 1
        past = t - s >= 1 ? x[t - s] : x0 * exp(r * (t - s))
        acc += w * past
    end
    return acc
end

function renew(I0, R::AbstractVector, gen_pmf, r)
    I_t = similar(R)
    for t in eachindex(R)
        I_t[t] = min(R[t] * lagged_sum(I_t, t, gen_pmf, I0, r, 1), 1e15)
    end
    return I_t
end

function delay_convolve(x::AbstractVector, pmf, x0, r)
    y = similar(x)
    for t in eachindex(x)
        y[t] = lagged_sum(x, t, pmf, x0, r, 0)
    end
    return y
end

"NegBin with mean mu and variance mu + alpha mu^2 (alpha = cluster^2), as in EpiNow2."
function NegativeBinomialMeanClust(mu, alpha)
    var = mu + alpha * mu^2
    return NegativeBinomial(1 / alpha, mu / var; check_args=false)
end

growth_rates(R::AbstractVector, gen_pmf) = R_to_r(R[1], gen_pmf)

function expected_renewal_given_Rt(R::AbstractVector, I0, gen_pmf, delay_pmf)
    r0 = growth_rates(R, gen_pmf)
    I_t = renew(I0, R, gen_pmf, r0)
    return (; I_t, Y_t=delay_convolve(I_t, delay_pmf, I0, r0))
end

# multi-patch ------------------------------------------------------------------

"Row-normalised gravity mixing matrix K[g,h] ∝ p̃_g p̃_h / d_gh^γ (g ≠ h), K[g,g] ∝ within."
function gravity(pop, dist; gamma, within=1.0)
    n = length(pop)
    mean_pop = sum(pop) / n
    K = [g == h ? within : (pop[g] / mean_pop) * (pop[h] / mean_pop) / dist[g, h]^gamma
         for g in 1:n, h in 1:n]
    return K ./ sum(K; dims=2)
end

growth_rates(R::AbstractMatrix, gen_pmf) = [growth_rates(view(R, :, h), gen_pmf) for h in axes(R, 2)]

function renew(I0::AbstractVector, R::AbstractMatrix, gen_pmf, K::AbstractMatrix, growth)
    T, P = size(R)
    I_t = similar(R)
    lambda = similar(I0)
    for t in 1:T
        for h in 1:P
            lambda[h] = lagged_sum(view(I_t, :, h), t, gen_pmf, I0[h], growth[h], 1)
        end
        for g in 1:P
            pressure = zero(eltype(R))
            for h in 1:P
                pressure += K[g, h] * lambda[h]
            end
            I_t[t, g] = min(R[t, g] * pressure, 1e15)
        end
    end
    return I_t
end

function delay_convolve(x::AbstractMatrix, pmf, x0, growth)
    y = similar(x)
    for g in axes(x, 2), t in axes(x, 1)
        y[t, g] = lagged_sum(view(x, :, g), t, pmf, x0[g], growth[g], 0)
    end
    return y
end

function expected_renewal_given_Rt(R::AbstractMatrix, I0, gen_pmf, delay_pmf, K)
    growth = growth_rates(R, gen_pmf)
    I_t = renew(I0, R, gen_pmf, K, growth)
    return (; I_t, Y_t=delay_convolve(I_t, delay_pmf, I0, growth))
end

"Damped weekly random walk of patch deviations with spatially correlated innovations (L L' = C)."
function spatial_deviations(L_corr, eta::AbstractMatrix; sigma, rho)
    n, n_weeks = size(eta)
    delta = similar(eta)
    prev = sigma .* (L_corr * eta[:, 1])
    delta[:, 1] = prev
    scale = sigma * sqrt(1 - rho^2)
    for w in 2:n_weeks
        prev = rho .* prev .+ scale .* (L_corr * eta[:, w])
        delta[:, w] = prev
    end
    return delta
end

"Expand a P × W weekly matrix to T × P daily rows: out[t, g] = delta[g, week(t)], week(t) = fld1(t, 7)."
function expand_weeks(delta_w::AbstractMatrix, T)
    P, W = size(delta_w)
    out = similar(delta_w, T, P)
    for g in 1:P, t in 1:T
        out[t, g] = delta_w[g, fld1(t, 7)]
    end
    return out
end

# ── simulated data with fixed truth ──────────────────────────────────────────

"""Single-patch epidemic: T days, fixed truth (init, sigma, log_I0, cluster), seeded innovations."""
function simulate_single(; T=56, seed=132, init=log(1.3), sigma=0.05, log_I0=log(50.0),
                          cluster=0.1, gen=gen_pmf(), delay=delay_pmf())
    rng = Xoshiro(seed)
    eps = randn(rng, T - 1)
    Z = vcat(init, init .+ sigma .* cumsum(eps))
    R = exp.(Z)
    (; I_t, Y_t) = expected_renewal_given_Rt(R, exp(log_I0), gen, delay)
    cases = [rand(rng, NegativeBinomialMeanClust(Y_t[t], cluster^2)) for t in 1:T]
    return (; T, init, sigma, eps, Z, R, log_I0, cluster, gen_pmf=gen, delay_pmf=delay,
              I_t, Y_t, cases, time=collect(1.0:T))
end

"""Six coupled patches: PR geography (seed 42), fixed truth (sigma, sigma_d, rho, gamma, cluster),
seeded innovations, outbreak seeded in the least populous patch."""
function simulate_patches(; T=56, n_patches=6, seed=42, sim_seed=8, gen=gen_pmf(), delay=delay_pmf(),
                           init=log(1.2), sigma=0.03, sigma_d=0.15, rho=0.8, gamma=1.5, cluster=0.1,
                           ell=30.0)
    # truth kept inside the priors but off the heavy tail (init=log 1.3, sigma=0.05, sigma_d=0.2
    # gave a sustained R≈2 and ~1e8 cumulative cases in the origin patch — the renewal model has
    # no depletion, so the PR's own prior-predictive heavy tail is real; a milder truth keeps the
    # simulated data epidemiologically sane for later recovery checks)
    rng = Xoshiro(seed)
    coords = [100 .* (rand(rng), rand(rng)) for _ in 1:n_patches]
    pop = rand(rng, LogNormal(log(100_000), 0.75), n_patches)
    dist = [hypot(p[1] - q[1], p[2] - q[2]) for p in coords, q in coords]
    K = gravity(pop, dist; gamma)
    C = exp.(-dist ./ ell)
    L_corr = Matrix(cholesky(Symmetric(C)).L)
    n_weeks = cld(T, 7)
    srng = Xoshiro(sim_seed)
    eps = randn(srng, T - 1)
    Z = vcat(init, init .+ sigma .* cumsum(eps))
    eta = randn(srng, n_patches, n_weeks)
    delta = spatial_deviations(L_corr, eta; sigma=sigma_d, rho)
    log_R = Z .+ expand_weeks(delta, T)                       # T × P
    R = exp.(log_R)
    origin = argmin(pop)
    seed_mean = [g == origin ? log(50.0) : log(0.05) for g in 1:n_patches]
    log_I0 = seed_mean .+ 0.5 .* randn(srng, n_patches)      # MvNormal(seed_mean, 0.25 I): sd 0.5
    growth = growth_rates(R, gen)
    I_t = renew(exp.(log_I0), R, gen, K, growth)
    Y_t = delay_convolve(I_t, delay, exp.(log_I0), growth)
    cases = [rand(srng, NegativeBinomialMeanClust(Y_t[t, g], cluster^2)) for t in 1:T, g in 1:n_patches]
    return (; T, n_patches, coords, pop, dist, C, K, L_corr, n_weeks, ell, gamma, init, sigma, sigma_d,
              rho, cluster, eps, Z, eta, delta, log_R, R, origin, seed_mean, log_I0, growth, I_t, Y_t,
              cases, gen_pmf=gen, delay_pmf=delay, time=collect(1.0:T), week=[fld1(t, 7) for t in 1:T])
end

function main()
    g = gen_pmf(); d = delay_pmf()
    println("gen_pmf   (lags 1..$(length(g))): sum=", round(sum(g); digits=6), "  mean lag=", round(sum((1:length(g)) .* g); digits=3))
    println("delay_pmf (lags 0..$(length(d)-1)): sum=", round(sum(d); digits=6), "  mean lag=", round(sum((0:length(d)-1) .* d); digits=3))
    s = simulate_single()
    println("single: T=$(s.T)  r0=", round(growth_rates(s.R, s.gen_pmf); digits=4),
            "  cases range=", extrema(s.cases), "  Y_t[1]=", round(s.Y_t[1]; digits=2), "  Y_t[end]=", round(s.Y_t[end]; digits=2))
    p = simulate_patches()
    println("patches: T=$(p.T) P=$(p.n_patches) origin=$(p.origin)  pop=", round.(p.pop ./ 1e3; digits=1), "k",
            "  K row sums=", round.(vec(sum(p.K; dims=2)); digits=6))
    println("  per-patch total cases=", vec(sum(p.cases; dims=1)), "  growth=", round.(p.growth; digits=3))
    println("  delta range=", round.(extrema(p.delta); digits=3))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
