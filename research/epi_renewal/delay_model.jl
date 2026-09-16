# Wren.jl PR #16 (epi-example @ 6e3fd026) — the REPORTING-DELAY sub-model in native @brm.
#
# The PR fits a doubly-interval-censored, per-stratum right-truncated LogNormal to an
# aggregated linelist, then fixes the reporting-delay PMF at the posterior means:
#
#   p_d(D) = P(d <= P + T < d + 1 | P + T <= D),   P ~ U(0, 1),  T ~ LogNormal(mu, sigma)
#   mu ~ N(1, 0.5) on [0, 3],   sigma ~ N(0.5, 0.25) on [0.1, 2]
#   stratum j: observed delay d_j, own truncation window D_j (primary day to analysis day),
#              likelihood weighted by its count n_j
#
# The primary-censored CDF F_pc(x) = P(P + T < x) = ∫_0^1 F(x - p) dp has a CLOSED FORM for
# LogNormal T: with G(a) = ∫_0^a F(u) du = a Φ((ln a - mu)/sigma) - e^{mu + sigma²/2} Φ((ln a - mu)/sigma - sigma)
# (G(a) = 0 for a <= 0), F_pc(x) = G(x) - G(x - 1). The PR evaluates the same masses through
# CensoredDistributions.jl; fixtures.jl evaluates them by quadrature; this file checks the closed
# form against the quadrature and then uses it in Stan.
#
# Gate: @brm build -> SBBRMI -> stan_code -> stanc_check -> stan_instantiate -> finite log-density
# + gradient, then NUTS (WarmupHMC) and the two-stage hand-off: the PMF at the posterior means
# is the `delay=` fixture argument of the renewal simulations, exactly as the PR does.
#   Run: julia --project=test research/epi_renewal/delay_model.jl [n_draws]
#
# VERIFIED (strato2, StanBlocks pin 9a958f97): closed form == quadrature to 2.4e-8 / 6.0e-8 / 1.3e-8
#   on three (mu, sigma, D); linelist 250 events → 80 strata, naive mean 3.23 d vs true 5.08 d (the
#   PR's truncation bias); delay_brm_expanded gate dim 2, finite gradient; NUTS 1000 draws in 17s,
#   0 divergences, min ESS 359, R̂ 1.002; mu 1.548 [1.463, 1.636], sigma 0.493 [0.442, 0.555]
#   (truth 1.5, 0.5); PMF at the posterior means: mean lag 5.17 d (true-parameter PMF 4.96 d),
#   max|diff| 0.016; simulate_single(; delay=pmf_hat) runs (cases 31..314).
#   `weighted(dic_lognormal(...), fweights(n))` — the aggregated spelling — fails at transpile:
#   snag delay-weighted-d-7c37c642 on BayesianRegressionModels.

include(joinpath(@__DIR__, "fixtures.jl"))   # censored_pmf (quadrature reference), simulate_* (take delay=)

using StanBlocks, BayesianRegressionModels, LogDensityProblems, BridgeStan, WarmupHMC
using Distributions, Random, Statistics, Printf
using MCMCDiagnosticTools

# ── closed-form primary-censored CDF, Julia reference ─────────────────────────
Φ(z) = cdf(Normal(), z)
lnorm_G(a, mu, sigma) = a <= 0 ? 0.0 : a * Φ((log(a) - mu) / sigma) - exp(mu + sigma^2 / 2) * Φ((log(a) - mu) / sigma - sigma)
lnorm_FS(x, mu, sigma) = lnorm_G(x, mu, sigma) - lnorm_G(x - 1, mu, sigma)   # P(P + T < x)
closed_pmf(mu, sigma, D) = [(lnorm_FS(t, mu, sigma) - lnorm_FS(t - 1, mu, sigma)) / lnorm_FS(D, mu, sigma) for t in 1:D]

# ── the PR's simulated linelist (same construction; a different RNG stream, so different strata) ──
function simulate_linelist(; true_delay=LogNormal(1.5, 0.5), analysis_day=21, n_events=250, seed=2)
    rng = Xoshiro(seed)
    w = exp.(0.1 .* (0:20)); w ./= sum(w)
    p_days = rand(rng, Categorical(w), n_events) .- 1                    # primary day 0..20, growing incidence
    counts = Dict{Tuple{Int,Int},Float64}()
    for p in p_days
        D = analysis_day - p                                             # this event's own truncation window (integer days)
        x = rand(rng) + rand(rng, true_delay)                            # exact draw from the truncated censored process
        while x > D
            x = rand(rng) + rand(rng, true_delay)
        end
        key = (floor(Int, x), D)
        counts[key] = get(counts, key, 0.0) + 1.0
    end
    strata = sort!(collect(keys(counts)))
    (; delay=first.(strata), window=last.(strata), n=[Int(counts[k]) for k in strata], p_days)
end

# ── Stan side: the same closed form and the observation family ────────────────
StanBlocks.@deffun begin
    lnorm_G(a::real, mu::real, sigma::real)::real =
        a <= 0.0 ? 0.0 : a * Phi((log(a) - mu) / sigma) - exp(mu + 0.5 * sigma * sigma) * Phi((log(a) - mu) / sigma - sigma)
    lnorm_FS(x::real, mu::real, sigma::real)::real = lnorm_G(x, mu, sigma) - lnorm_G(x - 1.0, mu, sigma)
    # log mass of one observed integer delay d under right truncation at window D
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
    # draw: the support under window D is d = 0 .. D-1 with the truncated masses
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

# ── the @brm model: the PR's delay_model, with its count weights as BRM frequency weights ──
delay_brm(d) = @brm d begin
    mu    ~ Normal(1.0, 0.5; lower=0.0, upper=3.0)
    sigma ~ Normal(0.5, 0.25; lower=0.1, upper=2.0)
    delay ~ weighted(dic_lognormal(mu, sigma, window), fweights(n))   # the PR's count weights
end

# The same likelihood on the UNAGGREGATED linelist (each stratum repeated n times): the
# frequency weights are repetition counts, so this is byte-for-byte the same density. Used
# while `weighted(<custom family>, fweights(n))` is snagged (delay-weighted-d-7c37c642).
delay_brm_expanded(d) = @brm d begin
    mu    ~ Normal(1.0, 0.5; lower=0.0, upper=3.0)
    sigma ~ Normal(0.5, 0.25; lower=0.1, upper=2.0)
    delay ~ dic_lognormal(mu, sigma, window)
end
expand_linelist(ll) = (; delay=reduce(vcat, (fill(d, k) for (d, k) in zip(ll.delay, ll.n))),
                         window=reduce(vcat, (fill(w, k) for (w, k) in zip(ll.window, ll.n))))

function gate_delay(d; build=delay_brm)
    brmi = build(d)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    stanc = StanBlocks.stanc_check(code; warn_pedantic=false)
    stanc.ok || error("stanc rejected delay_brm:\n" * stanc.output)
    problem = StanBlocks.stan_instantiate(sb.model)
    dim = LogDensityProblems.dimension(problem)
    q = [0.1 * ((i % 7) - 3) for i in 1:dim]
    lp, g = LogDensityProblems.logdensity_and_gradient(problem, q)
    println(@sprintf("delay_brm gate: dim=%d lp=%.4f finite grad=%s", dim, lp, all(isfinite, g)))
    (; brmi, sb, problem, code)
end

function main_delay(; n_draws=1000)
    # 1. closed form == quadrature (the fixtures' PMFs are the PR's PMFs)
    for (mu, sigma, D) in ((1.5, 0.5, 15), (1.0, 0.8, 21), (2.0, 0.3, 10))
        err = maximum(abs.(closed_pmf(mu, sigma, D) .- censored_pmf(LogNormal(mu, sigma); D)))
        println(@sprintf("closed-form vs quadrature PMF  mu=%.2f sigma=%.2f D=%d  max|diff|=%.2e", mu, sigma, D, err))
    end
    # 2. linelist + fit
    ll = simulate_linelist()
    println("linelist: ", length(ll.p_days), " events → ", length(ll.delay), " strata; naive mean delay = ",
            round(sum(ll.n .* ll.delay) / sum(ll.n); digits=3), " (true mean ", round(mean(LogNormal(1.5, 0.5)); digits=3), ")")
    d = expand_linelist(ll)                      # see delay_brm_expanded
    println("expanded linelist rows: ", length(d.delay))
    t = gate_delay(d; build=delay_brm_expanded)
    seconds = @elapsed fit = WarmupHMC.adaptive_warmup_mcmc(Xoshiro(1), t.problem; n_draws, progress=nothing)
    q = convert(Matrix{Float64}, fit.posterior_position)
    ess, rhat = MCMCDiagnosticTools.ess_rhat(reshape(permutedims(q), size(q, 2), 1, size(q, 1)))
    names = BridgeStan.param_names(t.problem.model)
    cons = reduce(hcat, (BridgeStan.param_constrain(t.problem.model, q[:, j]) for j in 1:size(q, 2)))
    println(@sprintf("delay fit: %d draws in %.0fs, divergent=%d, min ESS=%.0f, max Rhat=%.3f", size(q, 2), seconds, fit.n_divergent_samples, minimum(ess), maximum(rhat)))
    for (i, nm) in enumerate(names)
        x = cons[i, :]
        println(@sprintf("  %-6s post mean=%.4f  90%% [%.4f, %.4f]", nm, mean(x), quantile(x, 0.05), quantile(x, 0.95)))
    end
    mu_hat = mean(cons[findfirst(==("mu"), names), :])
    sigma_hat = mean(cons[findfirst(==("sigma"), names), :])
    # 3. the hand-off: PMF at the posterior means feeds the renewal simulations (the PR's two-stage workflow)
    pmf_hat = censored_pmf(LogNormal(mu_hat, sigma_hat); D=15)
    pmf_true = delay_pmf()
    println(@sprintf("delay_pmf at posterior means: mean lag %.3f (true-parameter PMF %.3f), max|diff| %.4f",
                     sum((0:14) .* pmf_hat), sum((0:14) .* pmf_true), maximum(abs.(pmf_hat .- pmf_true))))
    s = simulate_single(; delay=pmf_hat)
    println("simulate_single(; delay=pmf_hat): cases range ", extrema(s.cases))
    (; ll, fit=t, mu_hat, sigma_hat, pmf_hat)
end

if abspath(PROGRAM_FILE) == @__FILE__
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1000
    main_delay(; n_draws=n)
end
