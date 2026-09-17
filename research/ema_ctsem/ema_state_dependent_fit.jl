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
# Recovery on Charles's EXACT design (strato2, StanBlocks bec23bc; 100 subjects x 30,
# meas. sd 0.3, T0 draws + burnin=10, 600 draws, seed 1, ~4 min) -- z = (est - true)/sd:
#   - every parameter but one recovers within ~2 sd, and our posterior sds match the
#     standard errors in Driver's own fitDemo recovery table to a median ratio of 1.10
#     (his fit is ML + Hessian draws; ours a full NUTS posterior).
#   - the exception ON THIS DATASET is the state-dependent shock correlation `cz` (= ctsem's
#     `rs`): 0.458 +/- 0.097 with the first-order predict (gh=0), 0.483 +/- 0.108 with the
#     moment-matched predict (gh=3), against a true 0.70 (z = -2.5 / -2.0). That is NOT a
#     bias of the filter: over 8 independent panels of the same design (generator drawn from
#     Xoshiro instead of this fixture's LCG) the gh=0 estimates are 0.40 ... 0.86 with mean
#     0.664 (se 0.057). One parameter in 14 at |z| ~ 2 is what chance produces.
#   - Driver reports rs ~ 0.41 on hand-rolled Euler-Maruyama data vs 0.641 +/- 0.092 on
#     ctGenerate data (fitDemo.Rmd 288-296). Those are different data-generating processes:
#     with no max timestep set, ctsem's generator AND filter take ONE step per observation
#     interval (state_sampling.jl / substep_mesh.jl), freezing the state-dependent cells at
#     the interval start; Euler-Maruyama at small steps lets them track the state. Our filter
#     substeps (nsub), so it matches fine-grid data.
#   - s0, m0 and T0VAR are not compared with truth: after the burn-in the first observed
#     state is not distributed as T0MEANS/T0VAR (Driver's table omits them too).
#
# Run: julia --project=test research/ema_ctsem/ema_state_dependent_fit.jl [gh]   (gh = 0 | 3 | 5)

using BayesianRegressionModels, StanBlocks, LogDensityProblems, BridgeStan, Random, WarmupHMC
using Distributions: Normal, Exponential
import Statistics

include(joinpath(@__DIR__, "ema_state_dependent.jl"))   # ema_sd triad, fixture, ema_state_dependent

function main(; gh=3)                                    # gh=0: ctsem's first-order predict; 3: moment-matched
    panel = with_filter(fixture(n=100, nt=30, seed=20260916); nsub=8, gh)   # Charles's design: 100 subjects x 30
    sb = SBBRMI(ema_state_dependent(panel); mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    prob = StanBlocks.stan_instantiate(sb.model; path=joinpath(tempdir(), "ema_sd_fit_$(hash(code)).stan"))
    println("state-dependent EMA in the kernel, dim=", LogDensityProblems.dimension(prob),
            " subjects=", length(panel.subject), " nsub=", panel.nsub, " gh=", panel.gh, " — WarmupHMC.adaptive_warmup_mcmc")

    fit = WarmupHMC.adaptive_warmup_mcmc(Xoshiro(1), prob; n_draws=600, progress=nothing)
    unc = fit.posterior_position
    cn = BridgeStan.param_names(prob.model)
    C = reduce(hcat, [BridgeStan.param_constrain(prob.model, collect(Float64, unc[:,i]))
                      for i in 1:size(unc,2)])
    mean_(s) = Statistics.mean(C[findfirst(==(s), cn), :])
    sd_(s)   = Statistics.std(C[findfirst(==(s), cn), :])
    truth = (; b0=0.5, bm=0.4, a12=-0.25, a21=-0.30, a22=-0.60, cintm=0.3, qd0=-0.2,
              qd1=0.3, cz=0.7, sdm=0.6, l31=1.2, thr=-1.0, r1=0.3, r2=0.3)   # r1,r2 are SDs
    println("draws=", size(unc,2))
    println(rpad("param",7), rpad("true",8), rpad("est",9), rpad("sd",8), "z=(est-true)/sd")
    for k in keys(truth)
        s=String(k); m=mean_(s); sd=sd_(s); z=(m-truth[k])/sd
        println(rpad(s,7), rpad(truth[k],8), rpad(round(m;digits=3),9), rpad(round(sd;digits=3),8), round(z;digits=2))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; gh = isempty(ARGS) ? 3 : parse(Int, ARGS[1]))
end
