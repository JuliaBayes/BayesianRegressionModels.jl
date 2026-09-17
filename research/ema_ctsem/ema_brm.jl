# How far does the pure `@brm` formula surface carry a continuous-time state-space
# model, and where does the `kernel(...)` cell take over? -- an incremental build-up
# towards the hierarchical EMA model of ctsem (Charles Driver's R package,
# https://github.com/cdriveraus/ctsem): two latent processes (stress, mood), three
# indicators, time-dependent and time-independent predictors.
#
#   Part A -- the MEASUREMENT + FACTOR layer on the pure `@brm` formula surface.
#             Two formula-surface boundaries show why the dynamics need a cell:
#             `dar(time)` is a single strictly increasing series (not a panel), and
#             mutually coupled latents are a cyclic declaration.
#   Part B -- the DYNAMICS via the `kernel(...)` do-block + one `@deffun` scan.
#
# The complete model these increments lead to is ema_sampled.jl.
#
# Every model is gated on the full pipeline: @brm build -> SBBRMI lowering ->
# transpile -> stanc -> BridgeStan finite log-density + gradient. Run with
#   julia --project=test research/ema_ctsem/ema_brm.jl

using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Normal, Exponential

# ============================================================================ #
# Fixtures (synthetic; no real data)
# ============================================================================ #

"""Long-format EMA data: one row per ping. Subject covariates repeat within
subject. Used by the formula-surface increments (Part A)."""
function ema_long(; n_subjects = 6, nt = 12, seed = 1)
    rng = seed
    rnd() = (rng = (1103515245 * rng + 12345) % 2^31; rng / 2^31)
    subject = String[]; time = Float64[]; workload = Float64[]
    age = Float64[]; treatment = Float64[]
    stressReport = Float64[]; moodReport = Float64[]; smoked = Int[]
    for i in 1:n_subjects
        agei = (i - n_subjects/2)/n_subjects; trti = Float64(i % 2)
        t = 0.0; s = 0.0; m = 0.0
        for _ in 1:nt
            t += 0.4 + rnd(); w = rnd() - 0.5
            s = 0.7s + 0.3m + 0.5w + 0.2*(rnd()-0.5)
            m = 0.6m + 0.2s + 0.2*(rnd()-0.5)
            push!(subject, "s$(i)"); push!(time, t); push!(workload, w)
            push!(age, agei); push!(treatment, trti)
            push!(stressReport, s + 0.1*(rnd()-0.5)); push!(moodReport, m + 0.1*(rnd()-0.5))
            push!(smoked, s > 0.3 ? 1 : 0)
        end
    end
    (; subject, time, workload, age, treatment, stressReport, moodReport, smoked)
end

"""Ragged per-subject EMA data (one 'row' per subject; ragged vectors). `dt_grid`
= per-step intervals (dt[1]=first time), `sqrt_dt_grid` = its elementwise sqrt
(precomputed to sidestep an in-cell broadcast limitation). `srmr` = stacked
[stress; mood] response per subject. Used by the kernel-form increments (Part B)."""
function ema_ragged(; n_subjects = 6, nt = 10, seed = 1)
    rng = seed
    rnd() = (rng = (1103515245 * rng + 12345) % 2^31; rng / 2^31)
    subject = String[]; age = Float64[]; treatment = Float64[]
    t_grid = Vector{Float64}[]; dt_grid = Vector{Float64}[]; sqrt_dt_grid = Vector{Float64}[]
    workload = Vector{Float64}[]; stressReport = Vector{Float64}[]
    moodReport = Vector{Float64}[]; smoked = Vector{Int}[]
    for i in 1:n_subjects
        push!(subject, "s$(i)"); push!(age, (i - n_subjects/2)/n_subjects); push!(treatment, Float64(i % 2))
        ts = Float64[]; dts = Float64[]; sdts = Float64[]; wl = Float64[]
        sr = Float64[]; mr = Float64[]; sm = Int[]
        t = 0.0; s = 0.0; m = 0.0
        for _ in 1:nt
            dv = 0.4 + rnd(); t += dv; push!(ts, t); push!(dts, dv); push!(sdts, sqrt(dv))
            w = rnd() - 0.5; push!(wl, w)
            s = 0.7s + 0.3m + 0.5w + 0.2*(rnd()-0.5)
            m = 0.6m + 0.2s + 0.2*(rnd()-0.5)
            push!(sr, s + 0.1*(rnd()-0.5)); push!(mr, m + 0.1*(rnd()-0.5)); push!(sm, s > 0.3 ? 1 : 0)
        end
        push!(t_grid, ts); push!(dt_grid, dts); push!(sqrt_dt_grid, sdts); push!(workload, wl)
        push!(stressReport, sr); push!(moodReport, mr); push!(smoked, sm)
    end
    srmr = [vcat(stressReport[i], moodReport[i]) for i in 1:n_subjects]
    smoked_flat = reduce(vcat, smoked)   # flat int[total], same order as the ragged latent
    (; subject, age, treatment, t_grid, dt_grid, sqrt_dt_grid, workload,
       stressReport, moodReport, smoked, srmr, smoked_flat)
end

# ============================================================================ #
# Part A — the measurement + factor layer, pure `@brm` formula surface
# ============================================================================ #

# BRM idiom: a data-column response takes an explicit `response ~ Distribution(
# linpred, ...)`; the linear predictor is a SEPARATE named `~` statement. (BRM
# does not use brms's implicit `y ~ 1 + x`.)

# 1. Hierarchical measurement + covariates + random slope.  OK, dim 20.
inc1(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + workload + age + treatment + (1 + workload | subject)
    stressReport ~ Normal(mu, sigma)
end

# 2. One latent AR(1) trend, observed with Normal error.  OK, dim 83.
#    (A single GLOBAL AR process over the time column + per-subject intercepts.)
inc2_ar(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + ar(time; p = 1) + (1 | subject)
    stressReport ~ Normal(mu, sigma)
end

# 2'. dAR trend — WALL on a multi-subject panel: "dar time axis must be strictly
#     increasing" (the time column resets per subject).
inc2_dar(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + dar(time; p = 1) + (1 | subject)
    stressReport ~ Normal(mu, sigma)
end

# 3. Two INDEPENDENT latent trends -> two Normal indicators.  OK, dim 166.
inc3(d) = @brm d begin
    sigma_s ~ Exponential(1); sigma_m ~ Exponential(1)
    mu_s ~ 1 + ar(time; p = 1) + (1 | subject)
    mu_m ~ 1 + ar(time; p = 1) + (1 | subject)
    stressReport ~ Normal(mu_s, sigma_s)
    moodReport   ~ Normal(mu_m, sigma_m)
end

# 4a. ONE latent shared across two MIXED-family responses.  OK, dim 83.
inc4a(d) = @brm d begin
    sigma ~ Exponential(1)
    stress ~ 1 + ar(time; p = 1) + (1 | subject)
    stressReport ~ Normal(stress, sigma)
    smoked ~ BernoulliLogit(stress)
end

# 4b. Shared latent + FREE loading + threshold (a factor / LAMBDA row).  OK, dim 85.
inc4b(d) = @brm d begin
    sigma ~ Exponential(1)
    l31 ~ Normal(0, 1); smoke_threshold ~ Normal(0, 1)
    stress ~ 1 + ar(time; p = 1) + (1 | subject)
    stressReport ~ Normal(stress, sigma)
    smoked ~ BernoulliLogit(l31 * stress + smoke_threshold)
end

# 5. Coupling probe — WALL: "cyclic model declarations". A mutual same-time
#    reference (VAR / transition matrix) is a cycle on the formula DAG; the
#    coupling is a time-LAGGED recurrence, which the formula surface has no term
#    for. Crossed in Part B via the kernel + a @deffun scan.
inc5_coupled(d) = @brm d begin
    sigma_s ~ Exponential(1); sigma_m ~ Exponential(1)
    stress ~ 1 + mood + ar(time; p = 1) + (1 | subject)
    mood   ~ 1 + stress + ar(time; p = 1) + (1 | subject)
    stressReport ~ Normal(stress, sigma_s)
    moodReport   ~ Normal(mood, sigma_m)
end

# ============================================================================ #
# Part B — the dynamics via the kernel(...) do-block + one @deffun scan
# ============================================================================ #

# K0b: the cell DECLARES in-cell sampled innovations and builds a per-subject
#      random walk via cumulative_sum — NO @deffun.  OK, dim 70.
K0b(d) = @brm d begin
    sigma ~ Exponential(1); sd ~ Exponential(1)
    s0 ~ 1 + (1 | subject)
    stress ~ kernel(t_grid, stressReport, s0) do ts, sR, ls0
        z::vector[dims(ts)[1]] ~ std_normal()
        traj = ls0 .+ sd .* cumulative_sum(z)
        sR ~ normal(traj, sigma)
        traj
    end
end

# K1: continuous-time RW — scale innovations by sqrt(dt). sqrt_dt is precomputed
#     as DATA (in-cell `sqrt.(dt)` broadcast trips the StanBlocks tracer).  OK, dim 70.
K1(d) = @brm d begin
    sigma ~ Exponential(1); sd ~ Exponential(1)
    s0 ~ 1 + (1 | subject)
    stress ~ kernel(sqrt_dt_grid, stressReport, s0) do sdt, sR, ls0
        z::vector[dims(sdt)[1]] ~ std_normal()
        traj = ls0 .+ sd .* cumulative_sum(sdt .* z)
        sR ~ normal(traj, sigma)
        traj
    end
end

# The coupled/nonlinear recurrence is the ONE piece that must be a @deffun.
# It returns [stress; mood] STACKED as vector[2*nt] (a per-cell `matrix` is not
# yet supported by the plate — "scalar or vector[K] only (MVP)").
StanBlocks.@deffun begin
    ema_coupled(dt::vector[nt], a_ss::real, a_sm::real, a_ms::real, a_mm::real,
                s0::real, m0::real, sd_s::real, sd_m::real,
                zs::vector[nt], zm::vector[nt])::vector[2 * nt] = begin
        out::vector[2 * nt]
        s = s0; m = m0
        out[1] = s; out[nt + 1] = m
        for t in 2:nt
            ds = (a_ss * s + a_sm * m) * dt[t] + sd_s * sqrt(dt[t]) * zs[t]
            dm = (a_ms * s + a_mm * m) * dt[t] + sd_m * sqrt(dt[t]) * zm[t]
            s = s + ds; m = m + dm
            out[t] = s; out[nt + t] = m
        end
        out
    end
end

# K2: COUPLED linear dynamics (the VAR the formula rejected as cyclic).  OK, dim 143.
K2(d) = @brm d begin
    sigma ~ Exponential(1)
    a_ss ~ Normal(-0.5, 0.3); a_sm ~ Normal(0, 0.3); a_ms ~ Normal(0, 0.3); a_mm ~ Normal(-0.5, 0.3)
    sd_s ~ Exponential(1); sd_m ~ Exponential(1)
    s0 ~ 1 + (1 | subject)
    m0 ~ 1 + (1 | subject)
    stress ~ kernel(dt_grid, srmr, s0, m0) do dt, y, ls0, lm0
        zs::vector[dims(dt)[1]] ~ std_normal()
        zm::vector[dims(dt)[1]] ~ std_normal()
        traj = ema_coupled(dt, a_ss, a_sm, a_ms, a_mm, ls0, lm0, sd_s, sd_m, zs, zm)
        y ~ normal(traj, sigma)   # y = [stressReport; moodReport] stacked
        traj
    end
end

# The FULL nonlinear, hierarchical, three-indicator model built on these increments
# is ema_sampled.jl (latent states sampled); its marginalized counterparts are
# ema_kernel_kalman.jl, ema_kernel_marginalized.jl and ema_state_dependent.jl.

# ============================================================================ #
# Reproducible gate: build -> stanc -> BridgeStan finite density/gradient.
# ============================================================================ #

function gate(name, buildfn, d)
    print(rpad(name, 42))
    local brmi, sb, code
    try brmi = buildfn(d) catch e; println("@brm FAIL: ", first(sprint(showerror, e), 90)); return end
    try sb = SBBRMI(brmi; mod=@__MODULE__) catch e; println("SBBRMI FAIL: ", first(sprint(showerror, e), 90)); return end
    try code = StanBlocks.stan_code(sb.model) catch e; println("transpile FAIL: ", first(sprint(showerror, e), 90)); return end
    r = StanBlocks.stanc_check(code; warn_pedantic=false)
    r.ok || (println("stanc FAIL"); return)
    cache = joinpath(tempdir(), "brm-ema"); isdir(cache) || mkpath(cache)
    prob = StanBlocks.stan_instantiate(sb.model; path=joinpath(cache, string(hash(code))*".stan"))
    dim = LogDensityProblems.dimension(prob)
    q = [0.1*((i%5)-2) for i in 1:dim]
    lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
    println("OK  dim=", dim, "  lp=", round(lp; digits=2), "  finite_grad=", all(isfinite, g))
end

function main()
    L = ema_long(); R = ema_ragged()
    println("── Part A: formula surface ──")
    for (n, f) in (("inc1", inc1), ("inc2_ar", inc2_ar), ("inc2_dar (WALL)", inc2_dar),
                   ("inc3", inc3), ("inc4a", inc4a), ("inc4b", inc4b), ("inc5_coupled (WALL)", inc5_coupled))
        gate(n, f, L)
    end
    println("── Part B: kernel + one @deffun ──")
    for (n, f) in (("K0b", K0b), ("K1", K1), ("K2", K2))
        gate(n, f, R)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
