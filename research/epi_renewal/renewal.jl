# Renewal-equation epidemic models on the `@brm` formula surface.
#
# This file is the executable source of the documentation page "Epidemic renewal models"
# (docs/src/renewal.md): the page shows the Stan functions and the model declarations below
# verbatim, and its figures are drawn from the summaries that `main()` writes. It is
# self-contained: it simulates the data, declares the Stan functions and the models, fits them
# and writes the summaries.
#
#   reporting delay   doubly interval-censored, right-truncated LogNormal delay on a linelist
#   one population    log R_t is a random walk; renewal recursion; reporting-delay convolution;
#                     negative-binomial daily counts; rows can be masked (forecast)
#   six patches       log R_{g,t} = shared walk + spatially correlated, damped weekly deviations;
#                     gravity mixing between patches; one seed per patch
#
#   Run:  julia --project=test research/epi_renewal/renewal.jl [sections]
#         sections: comma-separated subset of delay,single,prior,forecast,patches (default: all)
#   Then: julia --project=<env with AlgebraOfVega> research/epi_renewal/renewal_figures.jl

using BayesianRegressionModels, StanBlocks, Distributions
using LinearAlgebra, Random, Statistics

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Delay distributions as daily probability mass functions
#    p_d = P(d <= P + T < d + 1 | P + T <= D),  P ~ Uniform(0, 1),  T ~ dist
#    (an event happens at a uniform time P within its day and is reported T later)
# ═══════════════════════════════════════════════════════════════════════════════
function censored_pmf(dist; D, drop_zero=false, nq=400)
    midpoints = ((1:nq) .- 0.5) ./ nq
    FS(x) = mean(cdf(dist, max(x - p, 0.0)) for p in midpoints)      # P(P + T < x)
    pmf = [(FS(t) - FS(t - 1)) / FS(D) for t in 1:D]                # lags 0 .. D-1
    drop_zero ? pmf[2:end] ./ sum(pmf[2:end]) : pmf
end
generation_pmf(; D=14) = censored_pmf(Gamma(6.5, 0.62); D, drop_zero=true)             # lags 1..13
reporting_pmf(; mu=1.5, sigma=0.5, D=15) = censored_pmf(LogNormal(mu, sigma); D)      # lags 0..14

# ═══════════════════════════════════════════════════════════════════════════════
# 2. The data-generating process in Julia (the truth the fits are compared with)
# ═══════════════════════════════════════════════════════════════════════════════
# growth rate r implied by a reproduction number R (Euler–Lotka, two Newton steps from r = 0)
function growth_rate(R, gen_pmf)
    r = 0.0
    for _ in 1:2
        f = sum(g * exp(-r * s) for (s, g) in enumerate(gen_pmf)) - 1 / R
        df = -sum(s * g * exp(-r * s) for (s, g) in enumerate(gen_pmf))
        r -= f / df
    end
    clamp(r, -2.0, 2.0)
end
# sum_i pmf[i] * x[t - lag_i]; before day 1 the history is x0 * exp(r * (s - 1))
function lagged_sum(x, t, pmf, x0, r, first_lag)
    acc = 0.0
    for (i, w) in enumerate(pmf)
        s = t - (first_lag + i - 1)
        acc += w * (s >= 1 ? x[s] : x0 * exp(r * (s - 1)))
    end
    acc
end
function simulate_single(; T=56, seed=132, init=log(1.3), sigma=0.05, log_I0=log(50.0), cluster=0.1,
                          gen=generation_pmf(), delay=reporting_pmf())
    rng = Xoshiro(seed)
    Z = vcat(init, init .+ sigma .* cumsum(randn(rng, T - 1)))
    R = exp.(Z)
    I0 = exp(log_I0); r0 = growth_rate(R[1], gen)
    I_t = zeros(T); Y_t = zeros(T)
    for t in 1:T
        I_t[t] = min(R[t] * lagged_sum(I_t, t, gen, I0, r0, 1), 1e15)
    end
    for t in 1:T
        Y_t[t] = lagged_sum(I_t, t, delay, I0, r0, 0)
    end
    cases = [rand(rng, NegativeBinomial(1 / cluster^2, 1 / (1 + cluster^2 * y))) for y in Y_t]
    (; T, init, sigma, R, log_I0, cluster, gen_pmf=gen, delay_pmf=delay, I_t, Y_t, cases, time=collect(1.0:T))
end
function gravity(pop, dist; gamma)
    P = length(pop); mean_pop = mean(pop)
    K = [g == h ? 1.0 : (pop[g] / mean_pop) * (pop[h] / mean_pop) / dist[g, h]^gamma for g in 1:P, h in 1:P]
    K ./ sum(K; dims=2)
end
function simulate_patches(; T=56, n_patches=6, seed=42, sim_seed=8, init=log(1.2), sigma=0.03, sigma_d=0.15,
                           rho=0.8, gamma=1.5, cluster=0.1, ell=30.0, gen=generation_pmf(), delay=reporting_pmf())
    rng = Xoshiro(seed)
    coords = 100 .* rand(rng, n_patches, 2)
    pop = rand(rng, LogNormal(log(100_000), 0.75), n_patches)
    dist = [g == h ? 0.0 : hypot(coords[g, 1] - coords[h, 1], coords[g, 2] - coords[h, 2]) for g in 1:n_patches, h in 1:n_patches]
    C = exp.(-dist ./ ell); L = cholesky(Symmetric(C)).L
    K = gravity(pop, dist; gamma)
    n_weeks = cld(T, 7); week = [fld1(t, 7) for t in 1:T]
    srng = Xoshiro(sim_seed)
    Z = vcat(init, init .+ sigma .* cumsum(randn(srng, T - 1)))
    eta = randn(srng, n_patches, n_weeks)
    delta = similar(eta)
    delta[:, 1] = sigma_d .* (L * eta[:, 1])
    for w in 2:n_weeks
        delta[:, w] = rho .* delta[:, w-1] .+ sigma_d * sqrt(1 - rho^2) .* (L * eta[:, w])
    end
    R = [exp(Z[t] + delta[g, week[t]]) for t in 1:T, g in 1:n_patches]
    origin = argmin(pop)                                   # the outbreak starts in the smallest patch
    seed_mean = [g == origin ? log(50.0) : log(0.05) for g in 1:n_patches]
    log_I0 = seed_mean .+ 0.5 .* randn(srng, n_patches)
    I0 = exp.(log_I0); growth = [growth_rate(R[1, h], gen) for h in 1:n_patches]
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
    (; T, n_patches, coords, pop, dist, C, K, n_weeks, init, sigma, sigma_d, rho, gamma, cluster, delta, R, origin,
       seed_mean, log_I0, I_t, Y_t, cases, gen_pmf=gen, delay_pmf=delay, time=collect(1.0:T), week)
end
# a linelist seen on `analysis_day`: an event on day p is in the data only if its delay is < analysis_day - p
function simulate_linelist(; true_delay=LogNormal(1.5, 0.5), analysis_day=21, n_events=250, seed=2)
    rng = Xoshiro(seed)
    w = exp.(0.1 .* (0:20)); w ./= sum(w)                  # a growing epidemic: recent days hold more events
    event_days = rand(rng, Categorical(w), n_events) .- 1
    delay = Int[]; window = Int[]
    for p in event_days
        D = analysis_day - p
        x = rand(rng) + rand(rng, true_delay)
        while x > D
            x = rand(rng) + rand(rng, true_delay)
        end
        push!(delay, floor(Int, x)); push!(window, D)
    end
    (; delay, window)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. The same process as Stan functions, and the observation families
# ═══════════════════════════════════════════════════════════════════════════════
StanBlocks.@deffun begin
    # ── renewal recursion: one population ──
    clamp2(x::real, lo::real, hi::real)::real = x < lo ? lo : (x > hi ? hi : x)
    # growth rate r implied by a reproduction number R (Euler–Lotka, two Newton steps from r = 0)
    growth_rate(R::real, gen_pmf::vector[G])::real = begin
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
    # sum_i pmf[i] * x[t - lag_i]; before day 1 the history is x0 * exp(r * (s - 1))
    lagged_sum(x::vector[T], t::int, pmf::vector[G], x0::real, r::real, first_lag::int)::real = begin
        acc = 0.0
        for i in 1:G
            s = t - (first_lag + i - 1)
            acc += pmf[i] * (s >= 1 ? x[s] : x0 * exp(r * (s - 1)))
        end
        acc
    end
    # infections I_t = R_t * sum_s g_s I_{t-s}, then expected reported cases Y_t = sum_d pi_d I_{t-d}
    expected_cases(log_R::vector[T], log_I0::real, gen_pmf::vector[G], delay_pmf::vector[D])::vector[T] = begin
        I0 = exp(log_I0)
        r0 = growth_rate(exp(log_R[1]), gen_pmf)
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
    # ── observation family: negative-binomial counts on the rows marked observed ──
    # cases ~ NegBin(mean Y, variance Y + cluster^2 Y^2); a row with observed == 0 is not scored,
    # and the generator draws every row, so an unobserved row's draw is its forecast
    @lhs @lpxf nb_cases_lpmf(cases::int[N], Y::vector[N], cluster::real, observed::int[N])::real = begin
        lp = 0.0
        for i in 1:N
            if observed[i] == 1
                lp += neg_binomial_2_lpmf(cases[i], Y[i], 1.0 / (cluster * cluster))
            end
        end
        lp
    end
    nb_cases_lpmfs(cases::int[N], Y::vector[N], cluster::real, observed::int[N])::vector[N] = begin
        lp::vector[N]
        for i in 1:N
            lp[i] = observed[i] == 1 ? neg_binomial_2_lpmf(cases[i], Y[i], 1.0 / (cluster * cluster)) : 0.0
        end
        lp
    end
    nb_cases_rng(int[N], Y::vector[N], cluster::real, observed::int[N])::int[N] = begin
        out::int[N]
        for i in 1:N
            out[i] = neg_binomial_2_rng(Y[i], 1.0 / (cluster * cluster))
        end
        out
    end
    # ── renewal recursion: coupled patches ──
    # rows of the long (day, patch) frame are ordered by patch, then day: row i = t + (g - 1) * T,
    # so `to_matrix(x, T, P)` puts patch g in column g
    dist_at(dist_flat::vector[PP], g::int, h::int, P::int)::real = dist_flat[g + (h - 1) * P]
    # gravity mixing: K[g, h] is the share of patch g's infection pressure that comes from patch h
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
    # I[t, g] = R[t, g] * sum_h K[g, h] * lambda[h, t],  lambda[h, t] = sum_s g_s I[t - s, h]
    patch_infections(R::matrix[T, P], K::matrix[P, P], log_I0::vector[P], gen_pmf::vector[G])::matrix[T, P] = begin
        I0 = exp(log_I0)
        growth::vector[P]
        for h in 1:P
            growth[h] = growth_rate(R[1, h], gen_pmf)
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
    # the seed predictor is per row and constant within a patch: read each patch's first row
    patch_seeds(log_I0_row::vector[N], pop::vector[P])::vector[P] = begin
        T = N / P
        out::vector[P]
        for g in 1:P
            out[g] = log_I0_row[(g - 1) * T + 1]
        end
        out
    end
    patch_expected_cases(log_R::vector[N], log_I0::vector[P], gamma::real, pop::vector[P], dist_flat::vector[PP],
                         gen_pmf::vector[G], delay_pmf::vector[D])::vector[N] = begin
        T = N / P
        R = to_matrix(exp(log_R), T, P)
        I = patch_infections(R, gravity_K(pop, dist_flat, gamma), log_I0, gen_pmf)
        I0 = exp(log_I0)
        Y::matrix[T, P]
        for g in 1:P
            growth = growth_rate(R[1, g], gen_pmf)
            for t in 1:T
                Y[t, g] = lagged_sum(col(I, g), t, delay_pmf, I0[g], growth, 0)
            end
        end
        to_vector(Y)
    end
    # ── reporting delay: doubly interval-censored, right-truncated LogNormal ──
    # F(x) = P(P + T < x) for P ~ Uniform(0, 1), T ~ LogNormal(mu, sigma), in closed form: G(x) - G(x - 1)
    lnorm_G(a::real, mu::real, sigma::real)::real =
        a <= 0.0 ? 0.0 : a * Phi((log(a) - mu) / sigma) - exp(mu + 0.5 * sigma * sigma) * Phi((log(a) - mu) / sigma - sigma)
    lnorm_F(x::real, mu::real, sigma::real)::real = lnorm_G(x, mu, sigma) - lnorm_G(x - 1.0, mu, sigma)
    # log P(delay = d | delay < window)
    censored_delay_lp1(d::int, mu::real, sigma::real, window::int)::real =
        log(lnorm_F(d + 1.0, mu, sigma) - lnorm_F(d + 0.0, mu, sigma)) - log(lnorm_F(window + 0.0, mu, sigma))
    @lhs @lpxf censored_delay_lpmf(d::int[N], mu::real, sigma::real, window::int[N])::real = begin
        lp = 0.0
        for j in 1:N
            lp += censored_delay_lp1(d[j], mu, sigma, window[j])
        end
        lp
    end
    censored_delay_lpmfs(d::int[N], mu::real, sigma::real, window::int[N])::vector[N] = begin
        lp::vector[N]
        for j in 1:N
            lp[j] = censored_delay_lp1(d[j], mu, sigma, window[j])
        end
        lp
    end
    censored_delay_rng(int[N], mu::real, sigma::real, window::int[N])::int[N] = begin
        out::int[N]
        for j in 1:N
            W = window[j]
            probs::vector[W]
            for k in 1:W
                probs[k] = lnorm_F(k + 0.0, mu, sigma) - lnorm_F(k - 1.0, mu, sigma)
            end
            out[j] = categorical_rng(probs / sum(probs)) - 1
        end
        out
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4. The data each model reads
# ═══════════════════════════════════════════════════════════════════════════════
reporting_delay_data() = simulate_linelist()

# one row per day; `observed_through = 42` masks days 43.. (their posterior predictive draws are the forecast)
function renewal_single_data(; observed_through=nothing)
    s = simulate_single()
    last_day = something(observed_through, s.T)
    (; time=s.time, cases=s.cases, observed=Int.((1:s.T) .<= last_day), gen_pmf=s.gen_pmf, delay_pmf=s.delay_pmf)
end

# one row per (day, patch), ordered by patch, then day
function renewal_patch_data()
    p = simulate_patches()
    T, P = p.T, p.n_patches
    (; time=repeat(p.time; outer=P), week=repeat(p.week; outer=P), patch=repeat(1:P; inner=T),
       cases=vec(p.cases), observed=ones(Int, T * P),
       seed_mean=repeat(p.seed_mean; inner=T),             # prior mean of each row's patch seed
       C=p.C,                                              # P × P correlation of the weekly innovations
       pop=p.pop, dist_flat=vec(p.dist), gen_pmf=p.gen_pmf, delay_pmf=p.delay_pmf)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 5. The models
# ═══════════════════════════════════════════════════════════════════════════════
function reporting_delay_model(data = reporting_delay_data())
    @brm data begin
        mu    ~ Normal(1.0, 0.5; lower=0.0, upper=3.0)
        sigma ~ Normal(0.5, 0.25; lower=0.1, upper=2.0)
        delay ~ censored_delay(mu, sigma, window)
    end
end

function renewal_single_model(data = renewal_single_data())
    @brm data begin
        log_I0  ~ Normal(log(50.0), 0.5)                    # log initial infections (scale of the seeded history)
        cluster ~ Normal(0.0, 0.1; lower=0.0)               # overdispersion of the counts
        log_R   ~ 1 + rw(time)                              # log R_t: a random walk over days
        effect(log_R, Intercept) ~ Normal(log(1.3), 0.1)    # ... its first value, log R_1
        sd(:, rw(time)) ~ Normal(0.0, 0.05)                 # ... its daily innovation scale
        Y       = expected_cases(log_R, log_I0, gen_pmf, delay_pmf)
        cases   ~ nb_cases(Y, cluster, observed)
    end
end

function renewal_patch_model(data = renewal_patch_data())
    @brm data begin
        gamma   ~ Normal(1.5, 0.5; lower=0.0)               # distance decay of the gravity mixing
        cluster ~ Normal(0.0, 0.1; lower=0.0)               # overdispersion of the counts
        log_R   ~ 1 + rw(time) + cdar(week; by=patch, cor=C)   # shared walk + correlated weekly patch deviations
        effect(log_R, Intercept) ~ Normal(log(1.3), 0.1)    # the walk's first value
        sd(:, rw(time)) ~ Normal(0.0, 0.05)                 # the walk's daily innovation scale
        sd(:, cdar(week)) ~ Normal(0.0, 0.2)                # the deviations' stationary scale
        ar(:, cdar(week)) ~ Normal(0.8, 0.1)                # the deviations' week-to-week persistence
        log_I0  ~ 0 + offset(seed_mean) + factor(patch)     # one seed per patch: prior mean + a cell mean
        effect(log_I0, patch) ~ Normal(0.0, 0.5)
        seeds   = patch_seeds(log_I0, pop)
        Y       = patch_expected_cases(log_R, seeds, gamma, pop, dist_flat, gen_pmf, delay_pmf)
        cases   ~ nb_cases(Y, cluster, observed)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Fit (NUTS via WarmupHMC on the BridgeStan log density), constrain, summarise
# ═══════════════════════════════════════════════════════════════════════════════
using LogDensityProblems, BridgeStan, WarmupHMC, MCMCDiagnosticTools, JSON, Printf

const RESULTS = joinpath(@__DIR__, "results", "renewal")

function build(brmi; held_out=())
    sb = SBBRMI(brmi; mod=@__MODULE__, held_out)
    code = BayesianRegressionModels.stan_code(sb)
    stanc = StanBlocks.stanc_check(code; warn_pedantic=false)
    stanc.ok || error("stanc rejected the model:\n" * stanc.output)
    (; sb, code, problem=StanBlocks.stan_instantiate(sb.model))
end
function fit(name, problem; n_draws=1000, seed=1, target_acceptance_rate=0.9, tolerate_gq_failures=false)
    seconds = @elapsed f = WarmupHMC.adaptive_warmup_mcmc(Xoshiro(seed), problem; n_draws, target_acceptance_rate, progress=nothing)
    q = convert(Matrix{Float64}, f.posterior_position)
    ess, rhat = MCMCDiagnosticTools.ess_rhat(reshape(permutedims(q), size(q, 2), 1, size(q, 1)))
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    rng = BridgeStan.StanRNG(problem.model, seed)
    cons = Matrix{Float64}(undef, length(names), size(q, 2)); failed = 0
    for j in 1:size(q, 2)
        try
            cons[:, j] = BridgeStan.param_constrain(problem.model, q[:, j]; include_tp=true, include_gq=true, rng)
        catch
            tolerate_gq_failures || rethrow()               # a prior draw can overflow the count generator
            failed += 1; cons[:, j] .= NaN
        end
    end
    diag = (; name, seed, draws=size(q, 2), seconds, divergences=f.n_divergent_samples, dim=size(q, 1),
              min_ess=minimum(ess), max_rhat=maximum(rhat), target_acceptance_rate, gq_failed=failed)
    println(@sprintf("%s: %d draws in %.0fs, divergent=%d, dim=%d, min ESS=%.0f, max Rhat=%.3f%s", name, diag.draws, seconds,
                     diag.divergences, diag.dim, diag.min_ess, diag.max_rhat, failed > 0 ? " ($failed draws without generated quantities)" : ""))
    flush(stdout)
    (; names, cons, diag)
end
stem(n) = first(split(n, ['.', '[']))
rows_for(r, want) = [i for (i, n) in enumerate(r.names) if stem(n) == want]
const PROBS = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)
const QNAMES = (:q025, :q10, :q25, :q50, :q75, :q90, :q975)
qs(x) = (; (QNAMES[i] => quantile(filter(isfinite, x), PROBS[i]) for i in eachindex(PROBS))...)
# quantile rows of a vector-valued quantity, in Stan's element order, next to the truth
function band_rows(r, want, truth; transform=identity, keys=j -> (;))
    idx = rows_for(r, want)
    length(idx) == length(truth) || error("$want: $(length(idx)) rows vs $(length(truth)) truths")
    [merge(keys(j), qs(transform.(r.cons[idx[j], :])), (; truth=truth[j])) for j in eachindex(truth)]
end
scalar_row(r, want, label, truth, model) = merge((; model, name=label, truth), qs(r.cons[only(rows_for(r, want)), :]))
coverage95(rows) = count(row -> row.q025 <= row.truth <= row.q975, rows) / length(rows)
save_json(name, obj) = (mkpath(RESULTS); open(io -> JSON.print(io, obj, 1), joinpath(RESULTS, name), "w"))
function load_rows(name)
    path = joinpath(RESULTS, name)
    isfile(path) ? Any[(; (Symbol(k) => v for (k, v) in row)...) for row in JSON.parsefile(path)] : Any[]
end

const MODEL_ORDER = ("reporting delay", "one population", "one population, prior only", "one population, days 1-42", "six patches")
by_model(rows) = sort(rows; by=r -> findfirst(==(r.model), MODEL_ORDER))      # stable: keeps each model's row order

const SINGLE_SCALARS = (("pop_log_R_beta_pop", "log R_1  (intercept of log_R)"), ("rw_log_R_time_sigma", "walk scale  sd(:, rw(time))"),
                        ("log_I0", "log_I0"), ("cluster", "cluster"))

# Sampler settings per fit (one chain each). The one-population posterior couples the walk scale with
# 55 innovations that the counts pin down tightly; it needs a small step size, hence the high target
# acceptance. The prior-only fit is easy and only feeds the prior predictive figure.
const SAMPLER = (delay=(; n_draws=1000, target_acceptance_rate=0.9),
                 single=(; n_draws=2000, target_acceptance_rate=0.995),
                 prior=(; n_draws=1000, target_acceptance_rate=0.8),
                 patches=(; n_draws=2000, target_acceptance_rate=0.9))

function main(; sections=("delay", "single", "prior", "forecast", "patches"))
    scalars = load_rows("scalars.json"); fits = load_rows("fits.json")
    replace_model!(rows, model) = filter!(r -> r.model != model, rows)
    record!(model, r, rows) = (replace_model!(scalars, model); append!(scalars, rows);
                               replace_model!(fits, model); push!(fits, merge((; model), r.diag)))
    s = simulate_single()
    single_truth = (s.init, s.sigma, s.log_I0, s.cluster)
    if "delay" in sections
        model = "reporting delay"
        ll = simulate_linelist()
        r = fit(model, build(reporting_delay_model(ll)).problem; SAMPLER.delay...)
        mus = r.cons[only(rows_for(r, "mu")), :]; sigmas = r.cons[only(rows_for(r, "sigma")), :]
        pmf_draws = reduce(hcat, (reporting_pmf(; mu=mus[j], sigma=sigmas[j]) for j in 1:min(400, length(mus))))
        truth_pmf = reporting_pmf()
        naive = [count(==(l - 1), ll.delay) / length(ll.delay) for l in eachindex(truth_pmf)]
        save_json("delay_pmf.json", [merge((; lag=l - 1, truth=truth_pmf[l], naive=naive[l]), qs(pmf_draws[l, :])) for l in eachindex(truth_pmf)])
        record!(model, r, Any[scalar_row(r, "mu", "mu", 1.5, model), scalar_row(r, "sigma", "sigma", 0.5, model)])
        println(@sprintf("  mean delay: linelist %.2f d, truth %.2f d", mean(ll.delay .+ 0.5), sum((0:14) .* truth_pmf) + 0.5))
    end
    if "single" in sections
        model = "one population"
        r = fit(model, build(renewal_single_model()).problem; SAMPLER.single...)
        R_rows = band_rows(r, "log_R", s.R; transform=exp, keys=j -> (; day=j))
        C_rows = band_rows(r, "cases_gen", s.Y_t; keys=j -> (; day=j, observed=s.cases[j]))
        println(@sprintf("  95%% interval coverage over days: R %.0f%%", 100coverage95(R_rows)))
        save_json("single_R.json", R_rows); save_json("single_cases.json", C_rows)
        record!(model, r, Any[scalar_row(r, nm, label, tr, model) for ((nm, label), tr) in zip(SINGLE_SCALARS, single_truth)])
    end
    if "prior" in sections
        model = "one population, prior only"
        # NOTE (2026-09-23, snag sbbrmi-brmi-mod-a97bf761): `held_out=:all` no
        # longer exists; the prior spelling is the same model with `cases`
        # omitted (fixed_param, dim 0 — this NUTS `fit` needs a fixed_param
        # sampler when the section is re-run). The checked-in summaries below
        # are the record of the held-out prior fit.
        r = fit(model, build(renewal_single_model(); held_out=:all).problem; SAMPLER.prior..., tolerate_gq_failures=true)
        save_json("prior_R.json", band_rows(r, "log_R", s.R; transform=exp, keys=j -> (; day=j)))
        save_json("prior_cases.json", band_rows(r, "cases_gen", s.Y_t; keys=j -> (; day=j, observed=s.cases[j])))
        replace_model!(fits, model); push!(fits, merge((; model), r.diag))
    end
    if "forecast" in sections
        model = "one population, days 1-42"
        n_obs = 42
        r = fit(model, build(renewal_single_model(renewal_single_data(; observed_through=n_obs))).problem; SAMPLER.single...)
        phase(j) = j <= n_obs ? "fitted" : "held out"
        save_json("forecast_R.json", band_rows(r, "log_R", s.R; transform=exp, keys=j -> (; day=j, phase=phase(j))))
        save_json("forecast_cases.json", band_rows(r, "cases_gen", s.Y_t; keys=j -> (; day=j, observed=s.cases[j], phase=phase(j))))
        record!(model, r, Any[scalar_row(r, nm, label, tr, model) for ((nm, label), tr) in zip(SINGLE_SCALARS, single_truth)])
    end
    if "patches" in sections
        model = "six patches"
        p = simulate_patches()
        T, P = p.T, p.n_patches
        r = fit(model, build(renewal_patch_model()).problem; SAMPLER.patches...)
        cell(j) = (; day=(j - 1) % T + 1, patch=(j - 1) ÷ T + 1)
        R_rows = band_rows(r, "log_R", vec(p.R); transform=exp, keys=cell)
        C_rows = band_rows(r, "cases_gen", vec(p.Y_t); keys=j -> merge(cell(j), (; observed=vec(p.cases)[j])))
        S_rows = band_rows(r, "seeds", p.log_I0; keys=j -> (; patch=j, prior_mean=p.seed_mean[j]))
        println(@sprintf("  95%% interval coverage: R %.0f%%, seeds %d/%d", 100coverage95(R_rows), count(x -> x.q025 <= x.truth <= x.q975, S_rows), P))
        save_json("patch_R.json", R_rows); save_json("patch_cases.json", C_rows); save_json("patch_seeds.json", S_rows)
        save_json("patches.json", [(; patch=g, x=p.coords[g, 1], y=p.coords[g, 2], population=p.pop[g], origin=g == p.origin) for g in 1:P])
        record!(model, r, Any[scalar_row(r, nm, label, tr, model) for (nm, label, tr) in
            (("pop_log_R_beta_pop", "log R_1  (intercept of log_R)", p.init), ("rw_log_R_time_sigma", "walk scale  sd(:, rw(time))", p.sigma),
             ("cdar_log_R_week_sigma", "deviation scale  sd(:, cdar(week))", p.sigma_d), ("cdar_log_R_week_rho", "persistence  ar(:, cdar(week))", p.rho),
             ("gamma", "gamma", p.gamma), ("cluster", "cluster", p.cluster))])
    end
    save_json("scalars.json", by_model(scalars)); save_json("fits.json", by_model(fits))
    println("summaries written to ", RESULTS)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; sections=length(ARGS) >= 1 ? Tuple(split(ARGS[1], ',')) : ("delay", "single", "prior", "forecast", "patches"))
end
