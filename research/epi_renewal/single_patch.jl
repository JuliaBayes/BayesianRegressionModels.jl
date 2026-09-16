# Wren.jl PR #16 (epi-example @ 6e3fd026) — the SINGLE-PATCH renewal model in
# native @brm.
#
#   log R_t : random walk, Z_1 ~ N(log 1.3, 0.1), Z_t = Z_{t-1} + sigma eps_t, sigma ~ N+(0, 0.05)
#   I_t     = R_t * sum_s g_s I_{t-s}        (seeded history I_u = I_0 e^{r u}, u <= 0; r from R_1)
#   Y_t     = sum_d p_d I_{t-d}              (reporting-delay convolution)
#   cases_t ~ NegBin(mean Y_t, var Y_t + c^2 Y_t^2),  c ~ N+(0, 0.1),  log I_0 ~ N(log 50, 0.5)
#
# The mechanistic map (renewal + convolution + NegBin) is ONE custom StanBlocks
# `@lpxf` family, `renewal_negbin`, used directly as the @brm RESPONSE — the
# custom-@lpxf-family bridge (`df1196e`, canonical) that `research/ema_ctsem/ema_ekf.jl`
# established. The random walk is where the formula surface is probed:
#
#   single_dar        — trend via the native `dar(time; p=1)` term (intercept = initial
#                       level; its AR coefficient beta ∈ [0,1] is SAMPLED, so a tight
#                       `ar(:, dar(time)) ~ Normal(0, 0.01)` pins beta ≈ 0 ≈ pure RW).
#                       Pure formula surface, no kernel. NOT exactly the PR's RW.
#   single_kernel_rw  — the EXACT RW: in-cell `z ~ std_normal()` + `cumulative_sum`
#                       inside a ONE-CELL `kernel(...)`. The kernel requires a
#                       ranef-bearing LP for its grouping, so a zero-mean dummy
#                       `g0 ~ 0 + (1 | series)` (1 level) is passed and left unused;
#                       it reaches no likelihood, so activity analysis lowers it to
#                       generated quantities (no nuisance sampler coordinates).
#                       The cell computes the expected cases and observes the ragged
#                       INTEGER `cases` slice IN-cell with the `nb_clust` family. That
#                       needs the test env's StanBlocks pin >= 9a958f97 (the integer
#                       ragged carrier, snag ragged-int-obser-771dd259): on the earlier
#                       pin bec23bc3 it failed with "family ... is discrete ... A
#                       RaggedVector stores its groups in a real vector", and the
#                       lossless escape was to observe the flat Vector{Int} at top
#                       level against `Z.mem` (research/ema_ctsem/ema_brm.jl's Kfull3).
#
# Every model is gated: @brm build -> SBBRMI -> stan_code -> stanc_check ->
# stan_instantiate -> LogDensityProblems.logdensity_and_gradient finite.
# Run: julia --project=test research/epi_renewal/single_patch.jl
#
# VERIFIED (strato2, test env from test/setup_env.jl, StanBlocks pin 9a958f97; the same verdicts held on
# the earlier pin bec23bc3 with the top-level `.mem` observation, and on canonical 8b4ec2f6):
#   probe   top-level vector parameter in @brm:
#           `eps::vector[5] ~ std_normal()`   REJECTED — parser: "Don't know how to handle xassignable(eps::vector[5])"
#           `eps ~ MvNormal(zeros(5), 1.0)`   REJECTED — sbimpl: "distribution MvNormal has no Stan translation"
#   single_dar        OK  dim=60  (55 innovations + dar beta + dar sigma + init + log_I0 + cluster), finite gradient
#   single_kernel_rw  OK  dim=59  (55 innovations + sig + init + log_I0 + cluster; the dummy ranef is
#                     GQ-lowered and costs no sampler coordinate; ragged integer obs IN-cell), finite gradient

using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Normal
include(joinpath(@__DIR__, "fixtures.jl"))

# ── the renewal process as Stan functions (ports of fixtures.jl) ─────────────
StanBlocks.@deffun begin
    clamp2(x::real, lo::real, hi::real)::real = begin
        y = x
        if y < lo
            y = lo
        end
        if y > hi
            y = hi
        end
        y
    end
    # growth rate r implied by R through the generation time: R * sum_s g_s e^{-r s} = 1
    R_to_r(R::real, gen_pmf::vector[G])::real = begin
        mean_gen = 0.0
        for s in 1:G
            mean_gen += s * gen_pmf[s]
        end
        r = clamp2((R - 1.0) / (R * mean_gen), -2.0, 2.0)
        for k in 1:2
            f = 0.0
            df = 0.0
            for s in 1:G
                e = exp(-r * s)
                f += gen_pmf[s] * e
                df -= s * gen_pmf[s] * e
            end
            r = clamp2(r - (R * f - 1.0) / (R * df), -2.0, 2.0)
        end
        r
    end
    # sum of pmf-weighted lags of x at t, reaching into the seeded history x0 e^{r u}
    lagged_sum(x::vector[T], t::int, pmf::vector[G], x0::real, r::real, first_lag::int)::real = begin
        acc = 0.0
        for i in 1:G
            s = first_lag + i - 1
            acc += pmf[i] * (t - s >= 1 ? x[t - s] : x0 * exp(r * (t - s)))
        end
        acc
    end
    renew(I0::real, R::vector[T], gen_pmf::vector[G], r::real)::vector[T] = begin
        I_t::vector[T]
        for t in 1:T
            I_t[t] = clamp2(R[t] * lagged_sum(I_t, t, gen_pmf, I0, r, 1), 0.0, 1e15)
        end
        I_t
    end
    delay_convolve(x::vector[T], pmf::vector[D], x0::real, r::real)::vector[T] = begin
        y::vector[T]
        for t in 1:T
            y[t] = lagged_sum(x, t, pmf, x0, r, 0)
        end
        y
    end
    # expected reported cases Y_t from the log-R path and the seed
    expected_cases(log_R::vector[T], log_I0::real, gen_pmf::vector[G], delay_pmf::vector[D])::vector[T] = begin
        R = exp(log_R)
        I0 = exp(log_I0)
        r0 = R_to_r(R[1], gen_pmf)
        delay_convolve(renew(I0, R, gen_pmf, r0), delay_pmf, I0, r0)
    end
end

# ── the observation model as a custom family: cases ~ renewal_negbin(...) ────
# NegBin(mean mu, var mu + alpha mu^2) with alpha = cluster^2  ==  Stan neg_binomial_2(mu, phi), phi = 1/cluster^2
StanBlocks.@deffun begin
    @lhs @lpxf renewal_negbin_lpmf(cases::int[T], log_R::vector[T], log_I0::real, cluster::real,
                                   gen_pmf::vector[G], delay_pmf::vector[D])::real = begin
        Y = expected_cases(log_R, log_I0, gen_pmf, delay_pmf)
        phi = 1.0 / (cluster * cluster)
        lp = 0.0
        for t in 1:T
            lp += neg_binomial_2_lpmf(cases[t], Y[t], phi)
        end
        lp
    end
    renewal_negbin_lpmfs(cases::int[T], log_R::vector[T], log_I0::real, cluster::real,
                         gen_pmf::vector[G], delay_pmf::vector[D])::vector[T] = begin
        Y = expected_cases(log_R, log_I0, gen_pmf, delay_pmf)
        phi = 1.0 / (cluster * cluster)
        out::vector[T]
        for t in 1:T
            out[t] = neg_binomial_2_lpmf(cases[t], Y[t], phi)
        end
        out
    end
    renewal_negbin_rng(int[T], log_R::vector[T], log_I0::real, cluster::real,
                       gen_pmf::vector[G], delay_pmf::vector[D])::int[T] = begin
        Y = expected_cases(log_R, log_I0, gen_pmf, delay_pmf)
        phi = 1.0 / (cluster * cluster)
        out::int[T]
        for t in 1:T
            out[t] = neg_binomial_2_rng(Y[t], phi)
        end
        out
    end
end

# ── @brm models ──────────────────────────────────────────────────────────────

# The observation family on a PRECOMPUTED expectation, for cells that compute Y themselves:
# cases ~ nb_clust(Y, cluster)  ==  neg_binomial_2(Y, 1/cluster^2), with the pointwise + sized-RNG twins.
StanBlocks.@deffun begin
    @lhs @lpxf nb_clust_lpmf(cases::int[N], Y::vector[N], cluster::real)::real = begin
        phi = 1.0 / (cluster * cluster)
        lp = 0.0
        for i in 1:N
            lp += neg_binomial_2_lpmf(cases[i], Y[i], phi)
        end
        lp
    end
    nb_clust_lpmfs(cases::int[N], Y::vector[N], cluster::real)::vector[N] = begin
        phi = 1.0 / (cluster * cluster)
        out::vector[N]
        for i in 1:N
            out[i] = neg_binomial_2_lpmf(cases[i], Y[i], phi)
        end
        out
    end
    nb_clust_rng(int[N], Y::vector[N], cluster::real)::int[N] = begin
        phi = 1.0 / (cluster * cluster)
        out::int[N]
        for i in 1:N
            out[i] = neg_binomial_2_rng(Y[i], phi)
        end
        out
    end
end

# (1) trend via the native `dar` term: pure formula surface.
single_dar(d) = @brm d begin
    log_I0 ~ Normal(log(50.0), 0.5)
    cluster ~ Normal(0.0, 0.1; lower=0.0)
    log_R ~ 1 + dar(time; p=1)
    effect(log_R, Intercept) ~ Normal(log(1.3), 0.1)   # Z_1
    sd(:, dar(time)) ~ Normal(0.0, 0.05)               # sigma (half-normal)
    ar(:, dar(time)) ~ Normal(0.0, 0.01)               # beta ≈ 0 -> pure random walk (approx.)
    cases ~ renewal_negbin(log_R, log_I0, cluster, gen_pmf, delay_pmf)
end

# (2) the exact RW inside a one-cell kernel: the cell declares the T-1 innovations.
single_kernel_rw(d) = @brm d begin
    log_I0 ~ Normal(log(50.0), 0.5)
    cluster ~ Normal(0.0, 0.1; lower=0.0)
    sig ~ Normal(0.0, 0.05; lower=0.0)
    init ~ Normal(log(1.3), 0.1)                     # Z_1, exactly the PR's prior
    g0 ~ 0 + (1 | series)                            # grouping dummy for the one-cell kernel (unused -> GQ)
    Z ~ kernel(time, cases, gen_pmf, delay_pmf, g0) do ts, cs, gp, dp, gd
        z::vector[dims(ts)[1] - 1] ~ std_normal()    # the T-1 innovations
        logR = append_row(init, init + sig * cumulative_sum(z))
        Y = expected_cases(logR, log_I0, gp, dp)      # expected reported cases (named -> posterior-addressable)
        cs ~ nb_clust(Y, cluster)                     # the ragged INTEGER observation, in-cell
        logR
    end
end

# ── gate ─────────────────────────────────────────────────────────────────────
function gate(name, buildfn, d)
    print(rpad(name, 44))
    local brmi, sb, code
    try brmi = buildfn(d) catch e; println("@brm FAIL: ", first(sprint(showerror, e), 200)); return nothing end
    try sb = SBBRMI(brmi; mod=@__MODULE__) catch e; println("SBBRMI FAIL: ", first(sprint(showerror, e), 200)); return nothing end
    try code = StanBlocks.stan_code(sb.model) catch e; println("transpile FAIL: ", first(sprint(showerror, e), 200)); return nothing end
    r = StanBlocks.stanc_check(code; warn_pedantic=false)
    r.ok || (println("stanc FAIL: ", first(string(r), 300)); return nothing)
    cache = joinpath(tempdir(), "brm-epi-renewal"); isdir(cache) || mkpath(cache)
    prob = StanBlocks.stan_instantiate(sb.model; path=joinpath(cache, string(hash(code)) * ".stan"))
    dim = LogDensityProblems.dimension(prob)
    q = [0.05 * ((i % 7) - 3) for i in 1:dim]
    lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
    println("OK  dim=", dim, "  lp=", round(lp; digits=2), "  finite_grad=", all(isfinite, g))
    return (; brmi, sb, prob)
end

# C-probe: does @brm admit a TOP-LEVEL vector parameter? (measured, not assumed)
function probe_toplevel_vector(d)
    println("── probe: top-level vector parameter in @brm ──")
    for body in ("eps::vector[5] ~ std_normal(); mu ~ 1; y ~ Normal(mu + sum(eps), 1.0)",
                 "eps ~ MvNormal(zeros(5), 1.0); mu ~ 1; y ~ Normal(mu + sum(eps), 1.0)")
        print(rpad(body, 78))
        try
            brmi = Core.eval(@__MODULE__, _brm(body; df=d))
            sb = SBBRMI(brmi; mod=@__MODULE__)
            println("ACCEPTED (dim probe): ", LogDensityProblems.dimension(StanBlocks.stan_instantiate(sb.model)))
        catch e
            println("REJECTED: ", first(sprint(showerror, e), 220))
        end
    end
end

function main()
    s = simulate_single()
    d_flat = (; time=s.time, cases=s.cases, gen_pmf=s.gen_pmf, delay_pmf=s.delay_pmf)
    d_cell = (; series=["all"], time=[s.time], cases=[s.cases], gen_pmf=[s.gen_pmf], delay_pmf=[s.delay_pmf])
    println("single-patch renewal: T=", s.T, "  cases range=", extrema(s.cases))
    probe_toplevel_vector((; y=randn(5)))
    println("── single-patch @brm ──")
    gate("single_dar (native dar trend)", single_dar, d_flat)
    gate("single_kernel_rw (exact RW, one-cell kernel)", single_kernel_rw, d_cell)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
