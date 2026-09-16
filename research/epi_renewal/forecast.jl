# Wren.jl PR #16 (epi-example @ 6e3fd026) — the FORECAST variant in native @brm.
#
# The PR refits the single-patch model to the first 42 of 56 days and simulates the remaining
# 14 days forward: the fitted latents come `FromParams`, the 14 unobserved random-walk
# innovations `FromPrior` (its `random_walk_forecast` draws innovations element-wise so a
# shorter fit can partially fill the walk). In @brm that is ONE model: the kernel cell still
# declares all T-1 innovations over the full 56-day time axis, but the observation family
# scores only the first `n_obs` days — so the posterior of innovations 42..55 is their prior,
# and the named in-cell expectation `Y` (and the auto `cases_gen` twin) beyond day 42 IS the
# forecast, with the fitted uncertainty and the prior tail propagated through the same
# renewal + delay map.
#
# Gate + fit + check: Y[43:56] and cases_gen[43:56] 90% intervals against the held-out truth.
#   Run: julia --project=test research/epi_renewal/forecast.jl [n_draws]
#
# VERIFIED (strato2, StanBlocks pin 9a958f97, δ=0.8, 1000 draws): gate dim 59; 21s, 260 divergences,
#   min ESS 24, R̂ 1.087 (the single-patch kernel geometry, worse on 42 days — see recover.jl);
#   init, log_I0, cluster inside 90% intervals, sig below (0.021 [0.002, 0.043] vs 0.05);
#   Y: fitted days 98% coverage (rel. width 0.14), forecast days 57% (rel. width 0.95);
#   cases_gen: fitted 95% (0.51), forecast 71% (1.13); Z (= log R): fitted 64%, forecast 43%.
#   The forecast bands widen ~2× past day 42, as the prior tail should; their calibration is
#   limited by this fit's geometry, not by the spelling — the same forecast on the top-level
#   vector spelling (`single_rw_fc`, δ=0.9): 28s, 2 divergences, min ESS 105, R̂ 1.010; all four
#   scalars inside 90% intervals (sig 0.026 [0.004, 0.058]); Y fitted 100% (rel. width 0.15) /
#   forecast 64% (1.05); cases_gen 88% (0.51) / 71% (1.21); log R fitted 81% / forecast 43% (0.48).

include(joinpath(@__DIR__, "vector_params.jl"))   # recover.jl (models, fixtures, fit helpers) + the vector-parameter spellings; mains guarded

# ── observation family that scores only the first n_obs elements; the RNG twin draws all N ──
StanBlocks.@deffun begin
    @lhs @lpxf nb_clust_upto_lpmf(cases::int[N], Y::vector[N], cluster::real, n_obs::int)::real = begin
        lp = 0.0
        for t in 1:n_obs
            lp += neg_binomial_2_lpmf(cases[t], Y[t], 1.0 / (cluster * cluster))
        end
        lp
    end
    nb_clust_upto_lpmfs(cases::int[N], Y::vector[N], cluster::real, n_obs::int)::vector[N] = begin
        lp::vector[N]
        for t in 1:N
            lp[t] = t <= n_obs ? neg_binomial_2_lpmf(cases[t], Y[t], 1.0 / (cluster * cluster)) : 0.0
        end
        lp
    end
    nb_clust_upto_rng(int[N], Y::vector[N], cluster::real, n_obs::int)::int[N] = begin
        out::int[N]
        for t in 1:N
            out[t] = neg_binomial_2_rng(Y[t], 1.0 / (cluster * cluster))
        end
        out
    end
end

# the exact random walk over the FULL time axis, observed through day n_obs only
single_kernel_rw_fc(d) = @brm d begin
    log_I0 ~ Normal(log(50.0), 0.5)
    cluster ~ Normal(0.0, 0.1; lower=0.0)
    sig ~ Normal(0.0, 0.05; lower=0.0)
    init ~ Normal(log(1.3), 0.1)
    g0 ~ 0 + (1 | series)
    Z ~ kernel(time, cases, gen_pmf, delay_pmf, n_obs, g0) do ts, cs, gp, dp, no, gd
        z::vector[dims(ts)[1] - 1] ~ std_normal()
        logR = append_row(init, init + sig * cumulative_sum(z))
        Y = expected_cases(logR, log_I0, gp, dp)
        cs ~ nb_clust_upto(Y, cluster, no[1])
        logR
    end
end

# ── the same forecast on the top-level-vector spelling (vector_params.jl `single_rw`) ──
# `n_obs` rides as a 1-element data vector; the family reads `n_obs[1]`.
StanBlocks.@deffun begin
    @lhs @lpxf nb_clust_upto_v_lpmf(cases::int[N], Y::vector[N], cluster::real, n_obs::int[1])::real =
        nb_clust_upto_lpmf(cases, Y, cluster, n_obs[1])
    nb_clust_upto_v_lpmfs(cases::int[N], Y::vector[N], cluster::real, n_obs::int[1])::vector[N] =
        nb_clust_upto_lpmfs(cases, Y, cluster, n_obs[1])
    # (a sized-token `_rng` cannot be delegated to from another @deffun — the tracer does not
    #  emit the callee for that call shape — so the body is spelled out)
    nb_clust_upto_v_rng(int[N], Y::vector[N], cluster::real, n_obs::int[1])::int[N] = begin
        out::int[N]
        for t in 1:N
            out[t] = neg_binomial_2_rng(Y[t], 1.0 / (cluster * cluster))
        end
        out
    end
end

single_rw_fc(d) = @brm d begin
    log_I0  ~ Normal(log(50.0), 0.5)
    cluster ~ Normal(0.0, 0.1; lower=0.0)
    sig     ~ Normal(0.0, 0.05; lower=0.0)
    init    ~ Normal(log(1.3), 0.1)
    eps     ~ MvNormal(zeros(length(time) - 1), 1.0)
    log_R   = rw_path(init, sig, eps)
    Y       = expected_cases(log_R, log_I0, gen_pmf, delay_pmf)
    cases   ~ nb_clust_upto_v(Y, cluster, n_obs)
end

function report_forecast(r, s, n_obs; paths)
    fitted = 1:n_obs; tail = (n_obs + 1):s.T
    for (want, truth, tf) in paths
        idx = rows_for(r, want)
        if length(idx) != s.T
            println("  ", want, ": ", length(idx), " rows (expected ", s.T, ")"); continue
        end
        cov(rows) = count(j -> (x = tf.(r.cons[idx[j], :]); q05(x) <= truth[j] <= q95(x)), rows) / length(rows)
        width(rows) = mean(j -> (x = tf.(r.cons[idx[j], :]); (q95(x) - q05(x)) / max(truth[j], 1e-9)), rows)
        println(@sprintf("  %-9s fitted days 90%% coverage %.0f%% (rel. width %.2f) | forecast days coverage %.0f%% (rel. width %.2f)",
                         want, 100cov(fitted), width(fitted), 100cov(tail), width(tail)))
    end
end

function main_forecast_vector(; n_draws=1000, n_obs=42)
    s = simulate_single()
    d = (; time=s.time, cases=s.cases, gen_pmf=s.gen_pmf, delay_pmf=s.delay_pmf, n_obs=[n_obs])
    gate("single_rw_fc", single_rw_fc, d)
    t = instantiate(single_rw_fc, d)
    r = fit_model("single_rw_fc (obs 1..$n_obs of $(s.T), vector spelling)", t.problem; n_draws, target_acceptance_rate=0.9)
    println("  stems: ", join(unique(stem.(r.names)), " "))
    report_scalar(r, "init", s.init); report_scalar(r, "sig", s.sigma)
    report_scalar(r, "log_I0", s.log_I0); report_scalar(r, "cluster", s.cluster)
    report_forecast(r, s, n_obs; paths=(("Y", s.Y_t, identity), ("log_R", s.R, exp), ("cases_gen", s.cases, identity)))
    r
end

function main_forecast(; n_draws=1000, n_obs=42)
    s = simulate_single()
    d = (; series=["all"], time=[s.time], cases=[s.cases], gen_pmf=[s.gen_pmf], delay_pmf=[s.delay_pmf], n_obs=[[n_obs]])
    t = instantiate(single_kernel_rw_fc, d)
    r = fit_model("single_kernel_rw_fc (obs 1..$n_obs of $(s.T))", t.problem; n_draws)
    println("  stems: ", join(unique(stem.(r.names)), " "))
    report_scalar(r, "init", s.init)
    report_scalar(r, "sig", s.sigma)
    report_scalar(r, "log_I0", s.log_I0)
    report_scalar(r, "cluster", s.cluster)
    fitted = 1:n_obs; tail = (n_obs + 1):s.T
    for (want, truth, tf) in (("Y", s.Y_t, identity), ("Z", s.R, exp), ("cases_gen", s.cases, identity))
        idx = rows_for(r, want)
        if length(idx) != s.T
            println("  ", want, ": ", length(idx), " rows (expected ", s.T, ")"); continue
        end
        cov(rows) = count(j -> (x = tf.(r.cons[idx[j], :]); q05(x) <= truth[j] <= q95(x)), rows) / length(rows)
        width(rows) = mean(j -> (x = tf.(r.cons[idx[j], :]); (q95(x) - q05(x)) / max(truth[j], 1e-9)), rows)
        println(@sprintf("  %-9s fitted days 90%% coverage %.0f%% (rel. width %.2f) | forecast days coverage %.0f%% (rel. width %.2f)",
                         want, 100cov(fitted), width(fitted), 100cov(tail), width(tail)))
    end
    r
end

if abspath(PROGRAM_FILE) == @__FILE__
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1000
    mode = length(ARGS) >= 2 ? ARGS[2] : "all"
    mode in ("all", "kernel") && main_forecast(; n_draws=n)
    mode in ("all", "vector") && main_forecast_vector(; n_draws=n)
end
