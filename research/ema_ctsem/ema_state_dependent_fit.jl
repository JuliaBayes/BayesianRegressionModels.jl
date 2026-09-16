# A REAL fit + recovery of the state-dependent-diffusion EMA (Charles Driver's
# fitDemo model), sampled with WarmupHMC via native BRM/SB — no AdvancedHMC, no
# Turing. Generates one long series from the KNOWN truth and recovers it.
#
# This is the single-series form (top-level `@lpxf` response, no kernel), which
# is identifiable from one long series and sidesteps the no-random-effects
# multi-subject kernel gap (snag no-ranef-kernel-7947153a). The state-dependent
# EKF and fixture are reused from ema_state_dependent.jl.
#
# Sampler: WarmupHMC.adaptive_warmup_mcmc on the BridgeStan problem (BridgeStan
# supplies the gradient; no external AD). The model has no random effects, so it
# is sampled directly rather than through adaptive_centering_problem.
#
# Illustrative recovery (strato2, StanBlocks bec23bc, 250 occasions, 600 draws,
# seed 1) — z = (est - true)/sd:
#   most params land within ~1-2 sd; the STATE-DEPENDENT SHOCK CORRELATION `cz`
#   (= ctsem's `rs`) has by far the widest posterior (sd ~0.28) and the coupling
#   term `a21` is the largest miss. This is exactly the delicacy Charles flags in
#   fitDemo.Rmd (lines 288-296): a hand-rolled Euler filter under-recovered `rs`
#   (0.41 vs 0.70), an integration-path artefact only a particle filter settles.
#   Our single-step EKF is such an approximate filter, so it inherits that cost;
#   a substepped mesh (as in ema_kernel_marginalized.jl) with matched fine-grid
#   generation is the faithful refinement.
#
# Run: julia --project=test research/ema_ctsem/ema_state_dependent_fit.jl

using BayesianRegressionModels, StanBlocks, LogDensityProblems, BridgeStan, Random, WarmupHMC
using Distributions: Normal, Exponential
import Statistics

include(joinpath(@__DIR__, "ema_state_dependent.jl"))   # ema_sd triad + fixture

function main()
    gen = fixture(n=1, nt=250, seed=20260916)           # one long series from truth
    d1 = (; stressReport=gen.stressReport[1], moodReport=gen.moodReport[1],
           smoked=gen.smoked[1], dt=gen.dt[1])

    single(d) = @brm d begin
        b0~Normal(0.5,0.5); bm~Normal(0.4,0.5); a12~Normal(-0.25,0.5)
        a21~Normal(-0.30,0.5); a22~Normal(-0.60,0.3); cintm~Normal(0.3,0.5)
        qd0~Normal(-0.2,0.5); qd1~Normal(0.3,0.5); cz~Normal(0.7,0.5); sdm~Exponential(1.0)
        l31~Normal(1.2,0.5); thr~Normal(-1.0,0.5); r1~Exponential(1.0); r2~Exponential(1.0)
        s0~Normal(0.,1.); m0~Normal(0.5,1.)
        stressReport ~ ema_sd(moodReport, smoked, dt, b0,bm,a12,a21,a22,cintm,qd0,qd1,cz,sdm,
                              l31,thr,r1,r2,s0,m0,0.6,0.5)
    end
    sb = SBBRMI(single(d1); mod=@__MODULE__)
    prob = StanBlocks.stan_instantiate(sb.model; path=joinpath(tempdir(),"sd_fit.stan"))
    println("state-dependent EMA recovery, dim=", LogDensityProblems.dimension(prob),
            " — WarmupHMC.adaptive_warmup_mcmc")

    fit = WarmupHMC.adaptive_warmup_mcmc(Xoshiro(1), prob; n_draws=600, progress=nothing)
    unc = fit.posterior_position                         # coords × draws
    cnames = BridgeStan.param_names(prob.model)
    C = reduce(hcat, [BridgeStan.param_constrain(prob.model, collect(Float64, unc[:,i]))
                      for i in 1:size(unc,2)])           # nparam × ndraws
    mean_(s) = Statistics.mean(C[findfirst(==(s), cnames), :])
    sd_(s)   = Statistics.std(C[findfirst(==(s), cnames), :])
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
