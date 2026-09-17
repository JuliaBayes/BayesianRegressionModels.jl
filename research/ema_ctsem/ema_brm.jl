# Translating the ctsem "EMA" continuous-time state-space demo to `@brm`.
#
# Companion to the note prepared for Charles Driver (ctsem author). The ctsem
# demo (ctModelLatex render) is a nonlinear, hierarchical, continuous-time SDE
# state-space model of two latent processes (stress, mood) measured by three
# indicators, with time-dependent and time-independent predictors.
#
# This file is the runnable, self-contained translation, built incrementally:
#   Part A — the MEASUREMENT + FACTOR layer on the pure `@brm` formula surface.
#   Part B — the DYNAMICS via the `kernel(...)` do-block + one `@deffun` scan.
#
# Every model is gated on the full pipeline: @brm build -> SBBRMI lowering ->
# transpile -> stanc -> BridgeStan finite log-density + gradient. Run with
#   julia --project=test research/ema_ctsem/ema_brm.jl
# on a host with stanc + a BridgeStan toolchain (see test/README.md).
#
# VERIFIED (strato2, StanBlocks bec23bc3c523):
#   Part A: inc1 dim20, inc2_ar dim83, inc3 dim166, inc4a dim83, inc4b dim85  -> all OK
#           inc2_dar -> WALL "dar time axis must be strictly increasing" (panel)
#           inc5_coupled -> WALL "cyclic model declarations" (VAR/transition matrix)
#   Part B: K0b dim70, K1 dim70, K2 dim143, Kfull2cont dim174, Kfull3 dim191 -> all OK
#           Kfull (binary AS A RAGGED KERNEL-CELL obs) -> fails: integer ragged
#             observation has no carrier in StanBlocks (snag ragged-int-obser-771dd259).
#           Kfull3 is the COMPLETE, FAITHFUL 3-indicator model: the four subject-
#             varying params (b0,q0,cint_mood,wl_stress) share one correlated
#             (1|p|subject) block (= ctsem's 4x4 rawPCov), a process-noise
#             correlation diff21 and manifest means mm_stress/mm_mood are included,
#             and the binary is observed at top level against the ragged latent's
#             flat backing (stress.mem). So the ENTIRE ctsem EMA demo builds.
#   The marginalized (competitive) counterparts are ema_kernel_marginalized.jl
#   (faithful EKF, states integrated out) and ema_kernel_kalman.jl (linear-Gaussian).
#
# The states are SAMPLED (dim grows with subjects x pings x processes) — the
# "expressive but not competitive" regime. Marginalizing them (Kalman/filter)
# and a structured metric are cross-package (StanBlocks + WarmupHMC) work.

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
    # Full EMA generator: nonlinear state-dependent drift + input-dependent
    # diffusion, with a free process-noise CORRELATION diff21 (fisher-z): the mood
    # innovation is correlated with the stress innovation, matching the spec's
    # DIFFUSION off-diagonal. (Single Euler step per occasion; the substepped mesh
    # lives in the marginalized EKF, where refining Δt adds no parameters.)
    ema_full(dt::vector[nt], wl::vector[nt],
             b0::real, bm::real, a12::real, a21::real, a22::real,
             cm::real, wls::real, q0::real, qw::real, diffm::real, diff21::real,
             s0::real, m0::real, zs::vector[nt], zm::vector[nt])::vector[2 * nt] = begin
        out::vector[2 * nt]
        s = s0 + wls * wl[1]; m = m0           # TDPREDEFFECT = IMPULSE at each observation (ctsem)
        corr = tanh(diff21)
        out[1] = s; out[nt + 1] = m
        for t in 2:nt
            drift_s = -log1p(exp(b0 + bm * m)) * s + a12 * m
            drift_m = a21 * s + a22 * m + cm
            gs = exp(q0 + qw * wl[t])              # DIFFUSION reads tdpreds[rowi]: the CURRENT row
            zc = corr * zs[t] + sqrt(1 - corr * corr) * zm[t]
            s = s + drift_s * dt[t] + gs * sqrt(dt[t]) * zs[t] + wls * wl[t]   # impulse at row t
            m = m + drift_m * dt[t] + diffm * sqrt(dt[t]) * zc
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

# Kfull2cont: the FULL nonlinear coupled continuous-time hierarchical EMA model
#   with covariates on parameters + the TWO continuous indicators.  OK, dim 173.
Kfull2cont(d) = @brm d begin
    sigma_s ~ Exponential(1); sigma_m ~ Exponential(1)
    bm ~ Normal(0, 0.5); a12 ~ Normal(0, 0.5); a21 ~ Normal(0, 0.5); a22 ~ Normal(-0.5, 0.3)
    wls ~ Normal(0, 0.5); qw ~ Normal(0, 0.5); diffm ~ Exponential(1); diff21 ~ Normal(0, 0.5)
    b0 ~ 1 + age + treatment + (1 | subject)
    q0 ~ 1 + (1 | subject)
    cm ~ 1 + age + treatment + (1 | subject)
    s0 ~ 1 + (1 | subject)
    m0 ~ 1 + (1 | subject)
    stress ~ kernel(dt_grid, workload, stressReport, moodReport,
                    b0, q0, cm, s0, m0) do dt, wl, sR, mR, lb0, lq0, lcm, ls0, lm0
        zs::vector[dims(dt)[1]] ~ std_normal()
        zm::vector[dims(dt)[1]] ~ std_normal()
        traj = ema_full(dt, wl, lb0, bm, a12, a21, a22, lcm, wls, lq0, qw, diffm, diff21, ls0, lm0, zs, zm)
        st = traj[1:dims(dt)[1]]
        mo = traj[(dims(dt)[1] + 1):(2 * dims(dt)[1])]
        sR ~ normal(st, sigma_s)
        mR ~ normal(mo, sigma_m)
        st
    end
end

# Kfull: as Kfull2cont but adds the BINARY indicator AS A RAGGED KERNEL-CELL obs.
#   Fails: integer ragged observation has no carrier in StanBlocks
#   (snag ragged-int-obser-771dd259). See Kfull3 for the working spelling.
Kfull(d) = @brm d begin
    sigma_s ~ Exponential(1); sigma_m ~ Exponential(1)
    bm ~ Normal(0, 0.5); a12 ~ Normal(0, 0.5); a21 ~ Normal(0, 0.5); a22 ~ Normal(-0.5, 0.3)
    wls ~ Normal(0, 0.5); qw ~ Normal(0, 0.5); diffm ~ Exponential(1); diff21 ~ Normal(0, 0.5)
    l31 ~ Normal(0, 1); smoke_threshold ~ Normal(0, 1)
    b0 ~ 1 + age + treatment + (1 | subject)
    q0 ~ 1 + (1 | subject)
    cm ~ 1 + age + treatment + (1 | subject)
    s0 ~ 1 + (1 | subject)
    m0 ~ 1 + (1 | subject)
    stress ~ kernel(dt_grid, workload, stressReport, moodReport, smoked,
                    b0, q0, cm, s0, m0) do dt, wl, sR, mR, smk, lb0, lq0, lcm, ls0, lm0
        zs::vector[dims(dt)[1]] ~ std_normal()
        zm::vector[dims(dt)[1]] ~ std_normal()
        traj = ema_full(dt, wl, lb0, bm, a12, a21, a22, lcm, wls, lq0, qw, diffm, diff21, ls0, lm0, zs, zm)
        st = traj[1:dims(dt)[1]]
        mo = traj[(dims(dt)[1] + 1):(2 * dims(dt)[1])]
        sR ~ normal(st, sigma_s)
        mR ~ normal(mo, sigma_m)
        smk ~ bernoulli_logit(l31 .* st .+ smoke_threshold)
        st
    end
end

# Kfull3: the COMPLETE 3-indicator EMA model. The binary indicator is observed
#   at TOP LEVEL against the ragged latent's flat backing (`stress.mem`), the
#   lossless workaround for the integer-ragged carrier gap (snag handler
#   verified: identical density/gradient — bernoulli_logit factorises
#   elementwise, so the ragged grouping is pure bookkeeping).  OK, dim 175.
Kfull3(d) = @brm d begin
    sigma_s ~ Exponential(1); sigma_m ~ Exponential(1)
    bm ~ Normal(0, 0.5); a12 ~ Normal(0, 0.5); a21 ~ Normal(0, 0.5); a22 ~ Normal(-0.5, 0.3)
    qw ~ Normal(0, 0.5); diffm ~ Exponential(1); diff21 ~ Normal(0, 0.5)   # + process-noise corr
    l31 ~ Normal(0, 1); smoke_threshold ~ Normal(0, 1)
    mm_s ~ Normal(0, 0.5); mm_m ~ Normal(0, 0.5)                            # manifest means
    # the four subject-varying params (b0, q0, cint_mood, wl_stress) share ONE
    # correlated block (brms (1|p|subject) = ctsem's free 4x4 rawPCov).
    b0  ~ 1 + age + treatment + (1 | p | subject)
    q0  ~ 1 +                   (1 | p | subject)
    cm  ~ 1 + age + treatment + (1 | p | subject)
    wls ~ 1 +                   (1 | p | subject)                           # wl_stress per-subject
    s0 ~ 1 + (1 | subject)
    m0 ~ 1 + (1 | subject)
    stress ~ kernel(dt_grid, workload, stressReport, moodReport,
                    b0, q0, cm, wls, s0, m0) do dt, wl, sR, mR, lb0, lq0, lcm, lwls, ls0, lm0
        zs::vector[dims(dt)[1]] ~ std_normal()
        zm::vector[dims(dt)[1]] ~ std_normal()
        traj = ema_full(dt, wl, lb0, bm, a12, a21, a22, lcm, lwls, lq0, qw, diffm, diff21, ls0, lm0, zs, zm)
        st = traj[1:dims(dt)[1]]
        mo = traj[(dims(dt)[1] + 1):(2 * dims(dt)[1])]
        sR ~ normal(mm_s .+ st, sigma_s)
        mR ~ normal(mm_m .+ mo, sigma_m)
        st
    end
    # discrete indicator: top-level, against the ragged latent's flat backing.
    smoked_flat ~ BernoulliLogit(l31 * stress.mem + smoke_threshold)
end

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
    for (n, f) in (("K0b", K0b), ("K1", K1), ("K2", K2),
                   ("Kfull2cont", Kfull2cont),
                   ("Kfull (binary as ragged cell obs, FAILS)", Kfull),
                   ("Kfull3 (full 3-indicator, binary via .mem)", Kfull3))
        gate(n, f, R)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
