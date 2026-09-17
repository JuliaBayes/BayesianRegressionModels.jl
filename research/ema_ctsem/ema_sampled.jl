# A hierarchical continuous-time state-space model with SAMPLED latent states.
#
# The model is the ecological-momentary-assessment (EMA) demonstration model of
# ctsem (Charles Driver's R package for hierarchical continuous-time dynamic
# modelling, https://github.com/cdriveraus/ctsem): two latent processes, stress
# and mood, follow a nonlinear stochastic differential equation and are measured
# at irregular times by two continuous reports and one binary indicator.
#
#   d stress = (-softplus(b0 + bm * mood) * stress + a12 * mood) dt + exp(q0 + qw * workload) dW1
#   d mood   = (a21 * stress + a22 * mood + cint_mood) dt           + diffm dW2,   corr(dW1, dW2) = tanh(diff21)
#   stress  += wl_stress * workload            at each observation (an input impulse)
#
# Division of labour, the same in every file of this directory:
#   formula surface  the population model -- covariates and (correlated) random
#                    effects on the subject-level parameters;
#   kernel(...) cell one subject: that subject's series, that subject's parameters;
#   @deffun          the recurrence, which needs a loop with carried state.
#
# Here the latent path is a deterministic function of standard-normal innovations
# that are PARAMETERS of the cell, so the model's dimension grows with
# subjects x occasions x processes. The marginalized counterparts, which integrate
# the path out instead, are ema_kernel_kalman.jl, ema_kernel_marginalized.jl and
# ema_state_dependent.jl.
#
# Run: julia --project=test research/ema_ctsem/ema_sampled.jl

using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Normal, Exponential

StanBlocks.@deffun begin
    # Euler-Maruyama path of the coupled SDE, driven by the innovations zs, zm.
    # Returns [stress; mood] stacked.
    ema_path(dt::vector[nt], wl::vector[nt],
             b0::real, bm::real, a12::real, a21::real, a22::real,
             cm::real, wls::real, q0::real, qw::real, diffm::real, diff21::real,
             s0::real, m0::real, zs::vector[nt], zm::vector[nt])::vector[2 * nt] = begin
        out::vector[2 * nt]
        s = s0 + wls * wl[1]; m = m0           # workload acts as an impulse at each observation
        corr = tanh(diff21)
        out[1] = s; out[nt + 1] = m
        for t in 2:nt
            drift_s = -log1p(exp(b0 + bm * m)) * s + a12 * m
            drift_m = a21 * s + a22 * m + cm
            gs = exp(q0 + qw * wl[t])              # stress volatility depends on the workload input
            zc = corr * zs[t] + sqrt(1 - corr * corr) * zm[t]
            s = s + drift_s * dt[t] + gs * sqrt(dt[t]) * zs[t] + wls * wl[t]
            m = m + drift_m * dt[t] + diffm * sqrt(dt[t]) * zc
            out[t] = s; out[nt + t] = m
        end
        out
    end
end

"""Synthetic EMA panel: one row per subject, ragged per-subject series."""
function ema_sampled_fixture(; n_subjects = 6, nt = 10, seed = 1)
    rng = seed
    rnd() = (rng = (1103515245 * rng + 12345) % 2^31; rng / 2^31)
    subject = String[]; age = Float64[]; treatment = Float64[]
    dt_grid = Vector{Float64}[]; workload = Vector{Float64}[]
    stressReport = Vector{Float64}[]; moodReport = Vector{Float64}[]; smoked = Vector{Int}[]
    for i in 1:n_subjects
        push!(subject, "s$(i)"); push!(age, (i - n_subjects/2)/n_subjects); push!(treatment, Float64(i % 2))
        dts = Float64[]; wl = Float64[]; sr = Float64[]; mr = Float64[]; sm = Int[]
        s = 0.0; m = 0.0
        for _ in 1:nt
            push!(dts, 0.4 + rnd()); w = rnd() - 0.5; push!(wl, w)
            s = 0.7s + 0.3m + 0.5w + 0.2*(rnd()-0.5)
            m = 0.6m + 0.2s + 0.2*(rnd()-0.5)
            push!(sr, s + 0.1*(rnd()-0.5)); push!(mr, m + 0.1*(rnd()-0.5)); push!(sm, s > 0.3 ? 1 : 0)
        end
        push!(dt_grid, dts); push!(workload, wl)
        push!(stressReport, sr); push!(moodReport, mr); push!(smoked, sm)
    end
    (; subject, age, treatment, dt_grid, workload, stressReport, moodReport, smoked)
end

function ema_sampled_model(data = ema_sampled_fixture())
    @brm data begin
        sigma_s ~ Exponential(1); sigma_m ~ Exponential(1)          # measurement sds
        bm ~ Normal(0, 0.5); a12 ~ Normal(0, 0.5); a21 ~ Normal(0, 0.5); a22 ~ Normal(-0.5, 0.3)
        qw ~ Normal(0, 0.5); diffm ~ Exponential(1); diff21 ~ Normal(0, 0.5)
        l31 ~ Normal(0, 1); smoke_threshold ~ Normal(0, 1)          # binary indicator: loading, threshold
        mm_s ~ Normal(0, 0.5); mm_m ~ Normal(0, 0.5)                # manifest means
        # subject-level parameters: covariates + ONE correlated random-effect block
        b0  ~ 1 + age + treatment + (1 | p | subject)
        q0  ~ 1 +                   (1 | p | subject)
        cm  ~ 1 + treatment +       (1 | p | subject)
        wls ~ 1 +                   (1 | p | subject)
        s0 ~ 1 + (1 | subject)
        m0 ~ 1 + (1 | subject)
        stress ~ kernel(dt_grid, workload, stressReport, moodReport, smoked,
                        b0, q0, cm, wls, s0, m0) do dt, wl, sR, mR, smk, lb0, lq0, lcm, lwls, ls0, lm0
            zs::vector[dims(dt)[1]] ~ std_normal()                  # the innovations ARE parameters
            zm::vector[dims(dt)[1]] ~ std_normal()
            path = ema_path(dt, wl, lb0, bm, a12, a21, a22, lcm, lwls, lq0, qw, diffm, diff21, ls0, lm0, zs, zm)
            st = path[1:dims(dt)[1]]
            mo = path[(dims(dt)[1] + 1):(2 * dims(dt)[1])]
            sR ~ normal(mm_s .+ st, sigma_s)
            mR ~ normal(mm_m .+ mo, sigma_m)
            smk ~ bernoulli_logit(l31 .* st .+ smoke_threshold)
            st
        end
    end
end

function main()
    data = ema_sampled_fixture()
    sb = SBBRMI(ema_sampled_model(data); mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    @assert StanBlocks.stanc_check(code; warn_pedantic=false).ok "stanc failed"
    prob = StanBlocks.stan_instantiate(sb.model; path=joinpath(tempdir(), "ema_sampled_$(hash(code)).stan"))
    dim = LogDensityProblems.dimension(prob)
    q = [0.1*((i % 5) - 2) for i in 1:dim]
    lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
    println("hierarchical EMA, latent states SAMPLED")
    println("  subjects = ", length(data.subject), "  occasions = ", length(data.dt_grid[1]))
    println("  dim = ", dim, "  lp = ", round(lp; digits=2), "  finite_grad = ", all(isfinite, g))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
