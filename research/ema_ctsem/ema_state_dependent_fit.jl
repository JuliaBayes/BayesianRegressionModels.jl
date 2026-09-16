# A REAL fit + recovery of the state-dependent-diffusion EMA (Charles Driver's
# fitDemo model) IN THE KERNEL — the faithful `indvarying = FALSE` multi-subject
# shape, sampled with WarmupHMC via native BRM/SB (no AdvancedHMC, no Turing).
#
# Data is generated from the KNOWN truth (a panel of independent subjects sharing
# all parameters); the kernel marginalizes each subject's latent path with the
# state-dependent-diffusion EKF (`ema_sd`), and the shared params are recovered.
# Sampler: WarmupHMC.adaptive_warmup_mcmc on the BridgeStan problem (BridgeStan
# supplies the gradient; the model has no random effects, so it is sampled
# directly rather than through adaptive_centering_problem).
#
# Illustrative recovery (strato2, StanBlocks bec23bc, 20 subjects x 30, 600 draws,
# seed 1) — z = (est - true)/sd. With the SUBSTEPPED EKF (nsub=8) matched to
# fine-grid generation, NEARLY ALL params land within ~1 sd. In particular the
# coupling term `a21` (z 7.9 -> -1.3) and the state-dependent shock correlation
# `cz` (= ctsem's `rs`, z ~0.1, est ~0.74 vs 0.70) both RECOVER: single-step Euler
# was the integration-path cost Driver flags (fitDemo.Rmd 288-296); substepping
# closes it. (Earlier single-step runs left a21 grossly biased.)
#
# Run: julia --project=test research/ema_ctsem/ema_state_dependent_fit.jl

using BayesianRegressionModels, StanBlocks, LogDensityProblems, BridgeStan, Random, WarmupHMC
using Distributions: Normal, Exponential
import Statistics

include(joinpath(@__DIR__, "ema_state_dependent.jl"))   # ema_sd triad, fixture, ema_state_dependent

function main()
    panel = fixture(n=20, nt=30, seed=20260916)          # 20 independent subjects, shared truth
    sb = SBBRMI(ema_state_dependent(panel); mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    prob = StanBlocks.stan_instantiate(sb.model; path=joinpath(tempdir(), "ema_sd_fit_$(hash(code)).stan"))
    println("state-dependent EMA in the kernel, dim=", LogDensityProblems.dimension(prob),
            " subjects=", length(panel.subject), " — WarmupHMC.adaptive_warmup_mcmc")

    fit = WarmupHMC.adaptive_warmup_mcmc(Xoshiro(1), prob; n_draws=600, progress=nothing)
    unc = fit.posterior_position
    cn = BridgeStan.param_names(prob.model)
    C = reduce(hcat, [BridgeStan.param_constrain(prob.model, collect(Float64, unc[:,i]))
                      for i in 1:size(unc,2)])
    mean_(s) = Statistics.mean(C[findfirst(==(s), cn), :])
    sd_(s)   = Statistics.std(C[findfirst(==(s), cn), :])
    truth = (; b0=0.5, bm=0.4, a12=-0.25, a21=-0.30, a22=-0.60, cintm=0.3, qd0=-0.2,
              qd1=0.3, cz=0.7, sdm=0.6, l31=1.2, thr=-1.0, r1=0.3, r2=0.3, s0=0.0, m0=0.5)
    println("draws=", size(unc,2))
    println(rpad("param",7), rpad("true",8), rpad("est",9), rpad("sd",8), "z=(est-true)/sd")
    for k in keys(truth)
        s=String(k); m=mean_(s); sd=sd_(s); z=(m-truth[k])/sd
        println(rpad(s,7), rpad(truth[k],8), rpad(round(m;digits=3),9), rpad(round(sd;digits=3),8), round(z;digits=2))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
