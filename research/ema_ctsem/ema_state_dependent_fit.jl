# Fit the state-dependent-diffusion model (ema_state_dependent.jl) to a panel simulated from
# known parameter values, and compare the posterior with those values.
#
# Design: 100 subjects x 30 irregularly spaced occasions, all parameters shared across
# subjects. Each subject's latent path is marginalized by the filter in the kernel cell, so
# the posterior has 19 dimensions whatever the panel size. Sampler:
# WarmupHMC.adaptive_warmup_mcmc on the BridgeStan problem that StanBlocks instantiates
# from the `@brm` model (the model has no random effects, so it is sampled directly).
#
# Filter precision (nsub substeps, gh predict order) is an argument. The default is the
# setting that the importance-sampling check in ema_state_dependent_psis.jl certifies for
# this panel (16 substeps, 3x3 Gauss-Hermite predict; about 50 minutes). `8 0` -- 8 substeps,
# first-order predict -- runs in about 5 minutes and lands within 0.54 posterior sds.
#
# Recorded result (600 draws, seed 1, nsub=16, gh=3), z = (mean - true) / sd:
#   param  true    mean     sd      z
#   b0     0.5     0.886    0.185   2.08
#   bm     0.4     0.187    0.153   -1.39
#   a12    -0.25   -0.258   0.045   -0.17
#   a21    -0.3    -0.326   0.043   -0.61
#   a22    -0.6    -0.652   0.05    -1.05
#   cintm  0.3     0.327    0.027   0.99
#   qd0    -0.2    -0.171   0.097   0.3
#   qd1    0.3     0.329    0.056   0.53
#   cz     0.7     0.474    0.115   -1.96
#   sdm    0.6     0.659    0.037   1.6
#   l31    1.2     1.16     0.12    -0.33
#   thr    -1.0    -0.988   0.043   0.28
#   r1     0.3     0.299    0.058   -0.01
#   r2     0.3     0.247    0.033   -1.59
# The initial means and covariance are not compared with generating values: after the
# generator's burn-in, the first observed state is not distributed as T0MEANS / T0VAR.
# Sampling variability of `cz` over independent panels: ema_state_dependent_replicate.jl.
#
# Run: julia --project=test research/ema_ctsem/ema_state_dependent_fit.jl [nsub] [gh]

using BayesianRegressionModels, StanBlocks, LogDensityProblems, BridgeStan, Random, WarmupHMC
using Distributions: Normal, Exponential
import Statistics

include(joinpath(@__DIR__, "ema_state_dependent.jl"))   # filter family, fixture, with_filter, model

generating_values() = (; b0=0.5, bm=0.4, a12=-0.25, a21=-0.30, a22=-0.60, cintm=0.3, qd0=-0.2,
                         qd1=0.3, cz=0.7, sdm=0.6, l31=1.2, thr=-1.0, r1=0.3, r2=0.3)

function main(; nsub=16, gh=3)
    panel = with_filter(ema_state_dependent_fixture(n=100, nt=30); nsub, gh)
    sb = SBBRMI(ema_state_dependent_model(panel); mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    prob = StanBlocks.stan_instantiate(sb.model; path=joinpath(tempdir(), "ema_sd_fit_$(hash(code)).stan"))
    println("state-dependent model, dim=", LogDensityProblems.dimension(prob),
            " subjects=", length(panel.subject), " nsub=", nsub, " gh=", gh)

    fit = WarmupHMC.adaptive_warmup_mcmc(Xoshiro(1), prob; n_draws=600, progress=nothing)
    names = BridgeStan.param_names(prob.model)
    C = reduce(hcat, [BridgeStan.param_constrain(prob.model, collect(Float64, c))
                      for c in eachcol(fit.posterior_position)])
    truth = generating_values()
    println("draws=", size(C, 2))
    println(rpad("param",7), rpad("true",8), rpad("mean",9), rpad("sd",8), "z")
    for k in keys(truth)
        v = C[findfirst(==(String(k)), names), :]
        m, sd = Statistics.mean(v), Statistics.std(v)
        println(rpad(k,7), rpad(truth[k],8), rpad(round(m;digits=3),9), rpad(round(sd;digits=3),8),
                round((m-truth[k])/sd;digits=2))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; nsub = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 16,
           gh   = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 3)
end
