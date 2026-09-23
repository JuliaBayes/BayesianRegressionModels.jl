# Wren.jl PR #16 (multi-patch renewal model, `epi-example @ 6e3fd026`) in native @brm —
# NOTE (2026-09-23, snag sbbrmi-brmi-mod-a97bf761): the `"prior"` section below
# uses `held_out=:all`, which no longer exists — the prior spelling is the same
# model with the response column omitted (fixed_param, dim 0). Kept as the
# historical record; re-running that section stops at the loud refusal.
# SELF-CONTAINED: everything needed to simulate, define, fit and summarise the models is in
# this one file (no includes). It is the consolidated form of research/epi_renewal/
# {fixtures,single_patch,multi_patch,recover,prior_predictive,delay_model,forecast,vector_params}.jl,
# on the formula surface that `x ~ MvNormal(...)` top-level vector parameters made possible
# (sbimpl c54cd0c, decision 187g4va).
#
#   single patch    log R_t random walk (init ~ N(log 1.3, 0.1), sigma ~ N+(0, 0.05)), renewal recursion
#                   with growth-rate seeding (2-step Newton), reporting-delay convolution,
#                   cases ~ NegBin(mean Y, var Y + c^2 Y^2), c ~ N+(0, 0.1), log I0 ~ N(log 50, 0.5)
#   six patches     log R_{g,t} = Z_t + delta_{g,w(t)}: shared walk + AR(1)-damped weekly deviations with
#                   spatially correlated innovations (L L' = exp(-d/30)), gravity mixing K(gamma),
#                   per-patch seeds log I0_g ~ N(seed_mean_g, 0.5), coupled recursion
#   delay model     doubly-interval-censored, per-stratum right-truncated LogNormal on a linelist
#   forecast        the single-patch model observed through day 42 only; the prior tail is the forecast
#
# Run (test env, ~12 min on strato2):  julia --project=test research/epi_renewal/wren16.jl [n_draws]
# Writes research/epi_renewal/results/*.json (quantile summaries + receipt) for plots.jl.

using BayesianRegressionModels, StanBlocks, LogDensityProblems, BridgeStan, WarmupHMC
using Distributions, LinearAlgebra, Random, Statistics, Printf, JSON
using MCMCDiagnosticTools

const RESULTS = joinpath(@__DIR__, "results")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Delay distributions → daily PMFs (the PR's CensoredDistributions semantics)
#    p_d = P(d <= P + T < d + 1 | P + T <= D),  P ~ U(0, 1),  T ~ dist
# ═══════════════════════════════════════════════════════════════════════════════
function censored_pmf(dist; D, drop_zero=false, nq=400)
    midpoints = ((1:nq) .- 0.5) ./ nq
    FS(x) = mean(cdf(dist, max(x - p, 0.0)) for p in midpoints)      # P(P + T < x)
    pmf = [(FS(t) - FS(t - 1)) / FS(D) for t in 1:D]                # lags 0 .. D-1
    drop_zero ? pmf[2:end] ./ sum(pmf[2:end]) : pmf
end
gen_pmf(; D=14) = censored_pmf(Gamma(6.5, 0.62); D, drop_zero=true)        # lags 1..13
delay_pmf(; mu=1.5, sigma=0.5, D=15) = censored_pmf(LogNormal(mu, sigma); D)  # lags 0..14

# ═══════════════════════════════════════════════════════════════════════════════
# 2. The renewal process in Julia (simulation truth)
# ═══════════════════════════════════════════════════════════════════════════════
function R_to_r(R, gen_pmf)
    r = 0.0
    for _ in 1:2
        f = sum(g * exp(-r * s) for (s, g) in enumerate(gen_pmf)) - 1 / R
        df = -sum(s * g * exp(-r * s) for (s, g) in enumerate(gen_pmf))
        r -= f / df
    end
    clamp(r, -2.0, 2.0)
end
function lagged_sum(x, t, pmf, x0, r, first_lag)
    acc = 0.0
    for (i, w) in enumerate(pmf)
        lag = first_lag + i - 1
        s = t - lag
        acc += w * (s >= 1 ? x[s] : x0 * exp(r * (s - 1)))           # seeded exponential history
    end
    acc
end
function simulate_single(; T=56, seed=132, init=log(1.3), sigma=0.05, log_I0=log(50.0), cluster=0.1,
                          gen=gen_pmf(), delay=delay_pmf())
    rng = Xoshiro(seed)
    eps = randn(rng, T - 1)
    Z = vcat(init, init .+ sigma .* cumsum(eps))
    R = exp.(Z)
    I0 = exp(log_I0); r0 = R_to_r(R[1], gen)
    I_t = zeros(T); Y_t = zeros(T)
    for t in 1:T
        I_t[t] = min(R[t] * lagged_sum(I_t, t, gen, I0, r0, 1), 1e15)
    end
    for t in 1:T
        Y_t[t] = lagged_sum(I_t, t, delay, I0, r0, 0)
    end
    cases = [rand(rng, NegativeBinomial(1 / cluster^2, 1 / (1 + cluster^2 * y))) for y in Y_t]
    (; T, init, sigma, eps, Z, R, log_I0, cluster, gen_pmf=gen, delay_pmf=delay, I_t, Y_t, cases, time=collect(1.0:T))
end
function gravity(pop, dist; gamma, within=1.0)
    P = length(pop); mean_pop = mean(pop)
    K = [g == h ? within : (pop[g] / mean_pop) * (pop[h] / mean_pop) / dist[g, h]^gamma for g in 1:P, h in 1:P]
    K ./ sum(K; dims=2)
end
function simulate_patches(; T=56, n_patches=6, seed=42, sim_seed=8, init=log(1.2), sigma=0.03, sigma_d=0.15,
                           rho=0.8, gamma=1.5, cluster=0.1, ell=30.0, gen=gen_pmf(), delay=delay_pmf())
    rng = Xoshiro(seed)
    coords = 100 .* rand(rng, n_patches, 2)
    pop = rand(rng, LogNormal(log(100_000), 0.75), n_patches)
    dist = [g == h ? 0.0 : hypot(coords[g, 1] - coords[h, 1], coords[g, 2] - coords[h, 2]) for g in 1:n_patches, h in 1:n_patches]
    C = exp.(-dist ./ ell); L = cholesky(Symmetric(C)).L
    K = gravity(pop, dist; gamma)
    n_weeks = cld(T, 7); week = [fld1(t, 7) for t in 1:T]
    srng = Xoshiro(sim_seed)
    eps = randn(srng, T - 1); Z = vcat(init, init .+ sigma .* cumsum(eps))
    eta = randn(srng, n_patches, n_weeks)
    delta = similar(eta)
    delta[:, 1] = sigma_d .* (L * eta[:, 1])
    for w in 2:n_weeks
        delta[:, w] = rho .* delta[:, w-1] .+ sigma_d * sqrt(1 - rho^2) .* (L * eta[:, w])
    end
    R = [exp(Z[t] + delta[g, week[t]]) for t in 1:T, g in 1:n_patches]
    origin = argmin(pop)
    seed_mean = [g == origin ? log(50.0) : log(0.05) for g in 1:n_patches]
    log_I0 = seed_mean .+ 0.5 .* randn(srng, n_patches)
    I0 = exp.(log_I0); growth = [R_to_r(R[1, h], gen) for h in 1:n_patches]
    I_t = zeros(T, n_patches); Y_t = zeros(T, n_patches)
    for t in 1:T
        lam = [lagged_sum(view(I_t, :, h), t, gen, I0[h], growth[h], 1) for h in 1:n_patches]
        for g in 1:n_patches
            I_t[t, g] = min(R[t, g] * sum(K[g, h] * lam[h] for h in 1:n_patches), 1e15)
        end
    end
    for g in 1:n_patches, t in 1:T
        Y_t[t, g] = lagged_sum(view(I_t, :, g), t, delay, I0[g], growth[g], 0)
    end
    cases = [rand(srng, NegativeBinomial(1 / cluster^2, 1 / (1 + cluster^2 * y))) for y in Y_t]
    (; T, n_patches, coords, pop, dist, C, K, L_corr=L, n_weeks, ell, init, sigma, sigma_d, rho, gamma, cluster,
       eps, Z, eta, delta, R, origin, seed_mean, log_I0, growth, I_t, Y_t, cases, gen_pmf=gen, delay_pmf=delay,
       time=collect(1.0:T), week)
end
function simulate_linelist(; true_delay=LogNormal(1.5, 0.5), analysis_day=21, n_events=250, seed=2)
    rng = Xoshiro(seed)
    w = exp.(0.1 .* (0:20)); w ./= sum(w)
    p_days = rand(rng, Categorical(w), n_events) .- 1
    counts = Dict{Tuple{Int,Int},Int}()
    for p in p_days
        D = analysis_day - p
        x = rand(rng) + rand(rng, true_delay)
        while x > D
            x = rand(rng) + rand(rng, true_delay)
        end
        counts[(floor(Int, x), D)] = get(counts, (floor(Int, x), D), 0) + 1
    end
    strata = sort!(collect(keys(counts)))
    (; delay=first.(strata), window=last.(strata), n=[counts[k] for k in strata], p_days)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. The same process as Stan functions (@deffun) and the observation families (@lpxf)
# ═══════════════════════════════════════════════════════════════════════════════
StanBlocks.@deffun begin
    clamp2(x::real, lo::real, hi::real)::real = x < lo ? lo : (x > hi ? hi : x)
    R_to_r(R::real, gen_pmf::vector[G])::real = begin
        r = 0.0
        for iter in 1:2
            f = -1.0 / R
            df = 0.0
            for s in 1:G
                f += gen_pmf[s] * exp(-r * s)
                df -= s * gen_pmf[s] * exp(-r * s)
            end
            r = r - f / df
        end
        clamp2(r, -2.0, 2.0)
    end
    lagged_sum(x::vector[T], t::int, pmf::vector[G], x0::real, r::real, first_lag::int)::real = begin
        acc = 0.0
        for i in 1:G
            s = t - (first_lag + i - 1)
            acc += pmf[i] * (s >= 1 ? x[s] : x0 * exp(r * (s - 1)))
        end
        acc
    end
    expected_cases(log_R::vector[T], log_I0::real, gen_pmf::vector[G], delay_pmf::vector[D])::vector[T] = begin
        I0 = exp(log_I0)
        r0 = R_to_r(exp(log_R[1]), gen_pmf)
        I::vector[T]
        for t in 1:T
            I[t] = clamp2(exp(log_R[t]) * lagged_sum(I, t, gen_pmf, I0, r0, 1), 0.0, 1e15)
        end
        Y::vector[T]
        for t in 1:T
            Y[t] = lagged_sum(I, t, delay_pmf, I0, r0, 0)
        end
        Y
    end
    rw_path(init::real, sig::real, eps::vector[K])::vector[K + 1] =
        append_row(init, init + sig * cumulative_sum(eps))
    # cases ~ NegBin(mean Y, var Y + c^2 Y^2) == neg_binomial_2(Y, 1/c^2)
    @lhs @lpxf nb_clust_lpmf(cases::int[N], Y::vector[N], cluster::real)::real = begin
        lp = 0.0
        for t in 1:N
            lp += neg_binomial_2_lpmf(cases[t], Y[t], 1.0 / (cluster * cluster))
        end
        lp
    end
    nb_clust_lpmfs(cases::int[N], Y::vector[N], cluster::real)::vector[N] = begin
        lp::vector[N]
        for t in 1:N
            lp[t] = neg_binomial_2_lpmf(cases[t], Y[t], 1.0 / (cluster * cluster))
        end
        lp
    end
    nb_clust_rng(int[N], Y::vector[N], cluster::real)::int[N] = begin
        out::int[N]
        for t in 1:N
            out[t] = neg_binomial_2_rng(Y[t], 1.0 / (cluster * cluster))
        end
        out
    end
    # the same family scoring only the first n_obs[1] days (the forecast fit); the RNG draws all N
    @lhs @lpxf nb_clust_upto_lpmf(cases::int[N], Y::vector[N], cluster::real, n_obs::int[1])::real = begin
        lp = 0.0
        for t in 1:n_obs[1]
            lp += neg_binomial_2_lpmf(cases[t], Y[t], 1.0 / (cluster * cluster))
        end
        lp
    end
    nb_clust_upto_lpmfs(cases::int[N], Y::vector[N], cluster::real, n_obs::int[1])::vector[N] = begin
        lp::vector[N]
        for t in 1:N
            lp[t] = t <= n_obs[1] ? neg_binomial_2_lpmf(cases[t], Y[t], 1.0 / (cluster * cluster)) : 0.0
        end
        lp
    end
    nb_clust_upto_rng(int[N], Y::vector[N], cluster::real, n_obs::int[1])::int[N] = begin
        out::int[N]
        for t in 1:N
            out[t] = neg_binomial_2_rng(Y[t], 1.0 / (cluster * cluster))
        end
        out
    end
    # ── six patches ──
    dist_at(dist_flat::vector[PP], g::int, h::int, P::int)::real = dist_flat[g + (h - 1) * P]
    gravity_K(pop::vector[P], dist_flat::vector[PP], gamma::real)::matrix[P, P] = begin
        mean_pop = sum(pop) / P
        K::matrix[P, P]
        for g in 1:P
            rs = 0.0
            for h in 1:P
                K[g, h] = g == h ? 1.0 : (pop[g] / mean_pop) * (pop[h] / mean_pop) / exp(gamma * log(dist_at(dist_flat, g, h, P)))
                rs += K[g, h]
            end
            for h in 1:P
                K[g, h] = K[g, h] / rs
            end
        end
        K
    end
    corr_chol(dist_flat::vector[PP], P::int, ell::real)::matrix[P, P] = begin
        C::matrix[P, P]
        for g in 1:P
            for h in 1:P
                C[g, h] = exp(-dist_at(dist_flat, g, h, P) / ell)
            end
        end
        cholesky_decompose(C)
    end
    deviations(eta::vector[PW], L::matrix[P, P], sig_d::real, rho::real, W::int)::matrix[P, W] = begin
        E = to_matrix(eta, P, W)
        delta::matrix[P, W]
        prev = sig_d * (L * col(E, 1))
        delta[:, 1] = prev
        scale = sig_d * sqrt(1.0 - rho * rho)
        for w in 2:W
            prev = rho * prev + scale * (L * col(E, w))
            delta[:, w] = prev
        end
        delta
    end
    patch_delta(eta::vector[PW], pop::vector[P], dist_flat::vector[PP], sig_d::real, rho::real, wgrid::vector[W])::matrix[P, W] =
        deviations(eta, corr_chol(dist_flat, P, 30.0), sig_d, rho, W)
    patch_R(z::vector[Tm1], delta::matrix[P, W], init::real, sig::real, week::int[T])::matrix[T, P] = begin
        Z = append_row(init, init + sig * cumulative_sum(z))
        R::matrix[T, P]
        for g in 1:P
            for t in 1:T
                R[t, g] = exp(Z[t] + delta[g, week[t]])
            end
        end
        R
    end
    patch_I(R::matrix[T, P], K::matrix[P, P], log_I0::vector[P], gen_pmf::vector[G])::matrix[T, P] = begin
        I0 = exp(log_I0)
        growth::vector[P]
        for h in 1:P
            growth[h] = R_to_r(R[1, h], gen_pmf)
        end
        I::matrix[T, P]
        lambda::vector[P]
        for t in 1:T
            for h in 1:P
                lambda[h] = lagged_sum(col(I, h), t, gen_pmf, I0[h], growth[h], 1)
            end
            for g in 1:P
                pressure = 0.0
                for h in 1:P
                    pressure += K[g, h] * lambda[h]
                end
                I[t, g] = clamp2(R[t, g] * pressure, 0.0, 1e15)
            end
        end
        I
    end
    patch_Y(I::matrix[T, P], R::matrix[T, P], log_I0::vector[P], gen_pmf::vector[G], delay_pmf::vector[D])::vector[T * P] = begin
        I0 = exp(log_I0)
        Y::matrix[T, P]
        for g in 1:P
            growth = R_to_r(R[1, g], gen_pmf)
            for t in 1:T
                Y[t, g] = lagged_sum(col(I, g), t, delay_pmf, I0[g], growth, 0)
            end
        end
        to_vector(Y)
    end
    # ── the delay model: closed-form primary-censored LogNormal CDF, F_pc(x) = G(x) - G(x - 1) ──
    lnorm_G(a::real, mu::real, sigma::real)::real =
        a <= 0.0 ? 0.0 : a * Phi((log(a) - mu) / sigma) - exp(mu + 0.5 * sigma * sigma) * Phi((log(a) - mu) / sigma - sigma)
    lnorm_FS(x::real, mu::real, sigma::real)::real = lnorm_G(x, mu, sigma) - lnorm_G(x - 1.0, mu, sigma)
    dic_lognormal_lp1(d::int, mu::real, sigma::real, D::int)::real =
        log(lnorm_FS(d + 1.0, mu, sigma) - lnorm_FS(d + 0.0, mu, sigma)) - log(lnorm_FS(D + 0.0, mu, sigma))
    @lhs @lpxf dic_lognormal_lpmf(d::int[N], mu::real, sigma::real, window::int[N])::real = begin
        lp = 0.0
        for j in 1:N
            lp += dic_lognormal_lp1(d[j], mu, sigma, window[j])
        end
        lp
    end
    dic_lognormal_lpmfs(d::int[N], mu::real, sigma::real, window::int[N])::vector[N] = begin
        lp::vector[N]
        for j in 1:N
            lp[j] = dic_lognormal_lp1(d[j], mu, sigma, window[j])
        end
        lp
    end
    dic_lognormal_rng(int[N], mu::real, sigma::real, window::int[N])::int[N] = begin
        out::int[N]
        for j in 1:N
            K = window[j]
            probs::vector[K]
            for k in 1:K
                probs[k] = lnorm_FS(k + 0.0, mu, sigma) - lnorm_FS(k - 1.0, mu, sigma)
            end
            out[j] = categorical_rng(probs / sum(probs)) - 1
        end
        out
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4. The @brm models — the PR's statements on the formula surface
# ═══════════════════════════════════════════════════════════════════════════════
single_rw(d) = @brm d begin
    log_I0  ~ Normal(log(50.0), 0.5)
    cluster ~ Normal(0.0, 0.1; lower=0.0)
    sig     ~ Normal(0.0, 0.05; lower=0.0)
    init    ~ Normal(log(1.3), 0.1)
    eps     ~ MvNormal(zeros(length(time) - 1), 1.0)       # vector[T-1] innovations
    log_R   = rw_path(init, sig, eps)
    Y       = expected_cases(log_R, log_I0, gen_pmf, delay_pmf)
    cases   ~ nb_clust(Y, cluster)
end
single_rw_fc(d) = @brm d begin                              # observed through day n_obs[1]; prior tail = forecast
    log_I0  ~ Normal(log(50.0), 0.5)
    cluster ~ Normal(0.0, 0.1; lower=0.0)
    sig     ~ Normal(0.0, 0.05; lower=0.0)
    init    ~ Normal(log(1.3), 0.1)
    eps     ~ MvNormal(zeros(length(time) - 1), 1.0)
    log_R   = rw_path(init, sig, eps)
    Y       = expected_cases(log_R, log_I0, gen_pmf, delay_pmf)
    cases   ~ nb_clust_upto(Y, cluster, n_obs)
end
patch_model(d) = @brm d begin
    init    ~ Normal(log(1.3), 0.1)
    sig     ~ Normal(0.0, 0.05; lower=0.0)
    sig_d   ~ Normal(0.0, 0.2; lower=0.0)
    rho     ~ Normal(0.8, 0.1; lower=0.0, upper=1.0)
    gamma   ~ Normal(1.5, 0.5; lower=0.0)
    cluster ~ Normal(0.0, 0.1; lower=0.0)
    z       ~ MvNormal(zeros(length(time) - 1), 1.0)                 # trend innovations (T-1)
    eta     ~ MvNormal(zeros(length(pop) * length(wgrid)), 1.0)      # deviation innovations (P·W)
    log_I0  ~ MvNormal(seed_mean, 0.5)                               # per-patch seeds
    K_mix   = gravity_K(pop, dist_flat, gamma)                       # matrix[P, P]
    delta   = patch_delta(eta, pop, dist_flat, sig_d, rho, wgrid)    # matrix[P, W]
    R       = patch_R(z, delta, init, sig, week)                     # matrix[T, P]
    I_t     = patch_I(R, K_mix, log_I0, gen_pmf)                     # matrix[T, P]
    Yf      = patch_Y(I_t, R, log_I0, gen_pmf, delay_pmf)            # vector[T*P]
    cases_flat ~ nb_clust(Yf, cluster)
end
delay_model(d) = @brm d begin
    mu    ~ Normal(1.0, 0.5; lower=0.0, upper=3.0)
    sigma ~ Normal(0.5, 0.25; lower=0.1, upper=2.0)
    delay ~ dic_lognormal(mu, sigma, window)
end

single_data(s) = (; time=s.time, cases=s.cases, gen_pmf=s.gen_pmf, delay_pmf=s.delay_pmf)
patch_data(p) = (; time=p.time, cases_flat=vec(p.cases), pop=p.pop, dist_flat=vec(p.dist), week=p.week,
                   wgrid=collect(1.0:p.n_weeks), seed_mean=p.seed_mean, gen_pmf=p.gen_pmf, delay_pmf=p.delay_pmf)
expand_linelist(ll) = (; delay=reduce(vcat, (fill(d, k) for (d, k) in zip(ll.delay, ll.n))),
                         window=reduce(vcat, (fill(w, k) for (w, k) in zip(ll.window, ll.n))))

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Fit (NUTS via WarmupHMC on the BridgeStan target), constrain, summarise
# ═══════════════════════════════════════════════════════════════════════════════
function build(buildfn, d; held_out=())
    brmi = buildfn(d)
    sb = SBBRMI(brmi; mod=@__MODULE__, held_out)
    code = StanBlocks.stan_code(sb.model)
    stanc = StanBlocks.stanc_check(code; warn_pedantic=false)
    stanc.ok || error("stanc rejected the model:\n" * stanc.output)
    (; brmi, sb, problem=StanBlocks.stan_instantiate(sb.model))
end
# An unmixed single chain (min ESS < 50 or split-R̂ > 1.05) is refit on the next seed, up to `attempts`
# times; every attempt's diagnostics are kept in the receipt, and the LAST attempt is the one reported.
function fit(name, problem; n_draws=1000, seed=1, target_acceptance_rate=0.9, tolerate_gq_failures=false, attempts=3, log=nothing)
    r = nothing
    for k in 0:attempts-1
        r = fit_once(name, problem; n_draws, seed=seed + k, target_acceptance_rate, tolerate_gq_failures)
        log === nothing || push!(log, Dict(string(k2) => v for (k2, v) in pairs(r.diag)))
        (r.diag.min_ess >= 50 && r.diag.max_rhat <= 1.05) && return r
        println("  chain not mixed (min ESS ", round(Int, r.diag.min_ess), ", max Rhat ", round(r.diag.max_rhat; digits=3), ") — refitting on seed ", seed + k + 1)
    end
    println("  WARNING: ", name, " did not mix in ", attempts, " attempts; reporting the last one")
    r
end
function fit_once(name, problem; n_draws=1000, seed=1, target_acceptance_rate=0.9, tolerate_gq_failures=false)
    seconds = @elapsed f = WarmupHMC.adaptive_warmup_mcmc(Xoshiro(seed), problem; n_draws, target_acceptance_rate, progress=nothing)
    q = convert(Matrix{Float64}, f.posterior_position)
    ess, rhat = MCMCDiagnosticTools.ess_rhat(reshape(permutedims(q), size(q, 2), 1, size(q, 1)))
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    rng = BridgeStan.StanRNG(problem.model, seed)
    cons = Matrix{Float64}(undef, length(names), size(q, 2)); failed = 0
    for j in 1:size(q, 2)
        try
            cons[:, j] = BridgeStan.param_constrain(problem.model, q[:, j]; include_tp=true, include_gq=true, rng)
        catch e
            tolerate_gq_failures || rethrow()
            failed += 1; cons[:, j] .= NaN
        end
    end
    diag = (; name, seed, draws=size(q, 2), seconds, divergences=f.n_divergent_samples, dim=size(q, 1),
              min_ess=minimum(ess), max_rhat=maximum(rhat), target_acceptance_rate, gq_failed=failed)
    println(@sprintf("%s (seed %d): %d draws in %.0fs, divergent=%d, dim=%d, min ESS=%.0f, max Rhat=%.3f%s", name, seed, diag.draws, seconds,
                     diag.divergences, diag.dim, diag.min_ess, diag.max_rhat, failed > 0 ? " ($failed GQ-failed draws)" : ""))
    flush(stdout)
    (; names, cons, diag)
end
stem(n) = first(split(n, ['.', '[']))
rows_for(r, want) = [i for (i, n) in enumerate(r.names) if stem(n) == want]
finite(x) = filter(isfinite, x)
const PROBS = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)
const QNAMES = (:q025, :q10, :q25, :q50, :q75, :q90, :q975)
qs(x) = (; (QNAMES[i] => quantile(finite(x), PROBS[i]) for i in eachindex(PROBS))...)
# quantile rows of a vector-valued quantity, in Stan's (column-major) element order, with truth
function band_rows(r, want, truth; transform=identity, keys=NamedTuple())
    idx = rows_for(r, want)
    length(idx) == length(truth) || error("$want: $(length(idx)) rows vs $(length(truth)) truths")
    [merge((; element=j), keys isa Function ? keys(j) : keys, qs(transform.(r.cons[idx[j], :])), (; truth=truth[j])) for j in eachindex(truth)]
end
scalar_row(r, want, truth, model) = merge((; model, name=want, truth), qs(r.cons[only(rows_for(r, want)), :]))
coverage(rows) = count(row -> row.q05 <= row.truth <= row.q95, rows) / length(rows)
cov90(rows) = count(row -> row.q025 <= row.truth <= row.q975, rows) / length(rows)
save_json(name, obj) = (isdir(RESULTS) || mkpath(RESULTS); open(io -> JSON.print(io, obj, 1), joinpath(RESULTS, name), "w"); joinpath(RESULTS, name))

function main(; n_draws=1000, sections=("single", "prior", "forecast", "delay", "patches"))
    receipt_path = joinpath(RESULTS, "receipt.json")
    receipt = isfile(receipt_path) ? JSON.parsefile(receipt_path) : Dict{String,Any}()
    receipt["n_draws"] = n_draws
    fits = get!(receipt, "fits", Any[])
    scalars_path = joinpath(RESULTS, "scalars.json")
    scalars = isfile(scalars_path) ? Any[(; (Symbol(k) => v for (k, v) in r)...) for r in JSON.parsefile(scalars_path)] : Any[]
    drop_model!(m) = filter!(r -> r.model != m, scalars)
    drop_fits!(prefix) = filter!(d -> !startswith(d["name"], prefix), fits)
    s = simulate_single()
    if "single" in sections
    drop_model!("single"); drop_fits!("single_rw (")
    t1 = build(single_rw, single_data(s))
    r1 = fit("single_rw", t1.problem; n_draws, log=fits)
    append!(scalars, Any[scalar_row(r1, nm, tr, "single") for (nm, tr) in (("init", s.init), ("sig", s.sigma), ("log_I0", s.log_I0), ("cluster", s.cluster))])
    R_rows = band_rows(r1, "log_R", s.R; transform=exp, keys=j -> (; day=j))
    Y_rows = band_rows(r1, "Y", s.Y_t; keys=j -> (; day=j, observed=s.cases[j]))
    C_rows = band_rows(r1, "cases_gen", Float64.(s.cases); keys=j -> (; day=j))
    println(@sprintf("  single: R 95%% coverage %.0f%%, Y %.0f%%", 100cov90(R_rows), 100cov90(Y_rows)))
    save_json("single_R.json", R_rows); save_json("single_Y.json", Y_rows); save_json("single_cases_gen.json", C_rows)
    end
    if "prior" in sections
    drop_fits!("single_rw prior")
    tp = build(single_rw, single_data(s); held_out=:all)
    rp = fit("single_rw prior", tp.problem; n_draws, target_acceptance_rate=0.8, tolerate_gq_failures=true, log=fits)
    PP_rows = band_rows(rp, "cases_gen", Float64.(s.cases); keys=j -> (; day=j))
    PR_rows = band_rows(rp, "log_R", s.R; transform=exp, keys=j -> (; day=j))
    save_json("single_prior_cases.json", PP_rows); save_json("single_prior_R.json", PR_rows)
    end
    if "forecast" in sections
    drop_model!("single, 42 days"); drop_fits!("single_rw_fc")
    n_obs = 42
    tf = build(single_rw_fc, merge(single_data(s), (; n_obs=[n_obs])))
    rf = fit("single_rw_fc (obs 1..42)", tf.problem; n_draws, log=fits)
    F_rows = band_rows(rf, "Y", s.Y_t; keys=j -> (; day=j, observed=s.cases[j], phase=j <= n_obs ? "fitted" : "forecast"))
    FC_rows = band_rows(rf, "cases_gen", Float64.(s.cases); keys=j -> (; day=j, phase=j <= n_obs ? "fitted" : "forecast"))
    FR_rows = band_rows(rf, "log_R", s.R; transform=exp, keys=j -> (; day=j, phase=j <= n_obs ? "fitted" : "forecast"))
    save_json("forecast_Y.json", F_rows); save_json("forecast_cases.json", FC_rows); save_json("forecast_R.json", FR_rows)
    append!(scalars, Any[scalar_row(rf, nm, tr, "single, 42 days") for (nm, tr) in (("init", s.init), ("sig", s.sigma), ("log_I0", s.log_I0), ("cluster", s.cluster))])
    end
    if "delay" in sections
    drop_model!("delay"); drop_fits!("delay_model")
    ll = simulate_linelist()
    td = build(delay_model, expand_linelist(ll))
    rd = fit("delay_model", td.problem; n_draws, log=fits)
    mu_hat = mean(rd.cons[only(rows_for(rd, "mu")), :]); sigma_hat = mean(rd.cons[only(rows_for(rd, "sigma")), :])
    append!(scalars, Any[scalar_row(rd, "mu", 1.5, "delay"), scalar_row(rd, "sigma", 0.5, "delay")])
    # PMF uncertainty: the daily masses over posterior draws
    mus = rd.cons[only(rows_for(rd, "mu")), :]; sigmas = rd.cons[only(rows_for(rd, "sigma")), :]
    pmf_draws = reduce(hcat, (delay_pmf(; mu=mus[j], sigma=sigmas[j]) for j in 1:min(200, length(mus))))
    truth_pmf = delay_pmf()
    D_rows = [merge((; lag=l - 1, truth=truth_pmf[l], at_posterior_means=delay_pmf(; mu=mu_hat, sigma=sigma_hat)[l]), qs(pmf_draws[l, :])) for l in 1:length(truth_pmf)]
    save_json("delay_pmf.json", D_rows)
    save_json("delay_linelist.json", [(; delay=d, window=w, n) for (d, w, n) in zip(ll.delay, ll.window, ll.n)])
    println(@sprintf("  delay: mu %.3f, sigma %.3f (truth 1.5, 0.5); naive mean delay %.2f d", mu_hat, sigma_hat, sum(ll.n .* ll.delay) / sum(ll.n)))
    end
    if "patches" in sections
    drop_model!("six patches"); drop_fits!("patch_model")
    p = simulate_patches()
    t2 = build(patch_model, patch_data(p))
    r2 = fit("patch_model", t2.problem; n_draws, log=fits)
    append!(scalars, Any[scalar_row(r2, nm, tr, "six patches") for (nm, tr) in
        (("init", p.init), ("sig", p.sigma), ("sig_d", p.sigma_d), ("rho", p.rho), ("gamma", p.gamma), ("cluster", p.cluster))])
    T, P, W = p.T, p.n_patches, p.n_weeks
    PR = band_rows(r2, "R", vec(p.R); transform=identity, keys=j -> (; day=(j - 1) % T + 1, patch=(j - 1) ÷ T + 1))
    PI = band_rows(r2, "I_t", vec(p.I_t); keys=j -> (; day=(j - 1) % T + 1, patch=(j - 1) ÷ T + 1))
    PY = band_rows(r2, "Yf", vec(p.Y_t); keys=j -> (; day=(j - 1) % T + 1, patch=(j - 1) ÷ T + 1, observed=vec(p.cases)[j]))
    PK = band_rows(r2, "K_mix", vec(p.K); keys=j -> (; from=(j - 1) % P + 1, to=(j - 1) ÷ P + 1))
    PD = band_rows(r2, "delta", vec(p.delta); keys=j -> (; patch=(j - 1) % P + 1, week=(j - 1) ÷ P + 1))
    PS = band_rows(r2, "log_I0", p.log_I0; keys=j -> (; patch=j))
    println(@sprintf("  six patches: 95%% coverage R %.0f%%, I_t %.0f%%, Yf %.0f%%, K %.0f%%, delta %.0f%%",
                     100cov90(PR), 100cov90(PI), 100cov90(PY), 100cov90(PK), 100cov90(PD)))
    save_json("patch_R.json", PR); save_json("patch_I.json", PI); save_json("patch_Y.json", PY)
    save_json("patch_K.json", PK); save_json("patch_delta.json", PD); save_json("patch_seeds.json", PS)
    save_json("patches.json", [(; patch=g, x=p.coords[g, 1], y=p.coords[g, 2], pop=p.pop[g], origin=g == p.origin) for g in 1:P])
    receipt["truth_patches"] = Dict("init" => p.init, "sigma" => p.sigma, "sigma_d" => p.sigma_d, "rho" => p.rho, "gamma" => p.gamma, "cluster" => p.cluster)
    end
    save_json("scalars.json", scalars)
    receipt["seeds"] = Dict("single" => 132, "patches" => [42, 8], "linelist" => 2, "sampler_first" => 1)
    receipt["mixing_rule"] = "a fit with min ESS < 50 or split-Rhat > 1.05 is refit on the next seed (up to 3 attempts); all attempts listed in fits, the last one reported"
    receipt["truth_single"] = Dict("init" => s.init, "sigma" => s.sigma, "log_I0" => s.log_I0, "cluster" => s.cluster)
    save_json("receipt.json", receipt)
    println("results written to ", RESULTS)
end

if abspath(PROGRAM_FILE) == @__FILE__
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1000
    secs = length(ARGS) >= 2 ? Tuple(split(ARGS[2], ',')) : ("single", "prior", "forecast", "delay", "patches")
    main(; n_draws=n, sections=secs)
end
