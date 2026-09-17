# How much does a recovered parameter vary from panel to panel? -- refit the
# state-dependent-diffusion model (ema_state_dependent.jl) on independently simulated
# panels of the same design (100 subjects x 30) and collect the posterior summaries.
#
# One panel is one draw from the sampling distribution of the estimator: a posterior mean
# two posterior sds from the generating value says little by itself when 14 parameters are
# compared. Replication separates sampling variability from systematic error.
#
# Generators (first argument):
#   xo8    Xoshiro(seed) random stream,  8 Euler-Maruyama steps per observation interval
#   xo64   Xoshiro(seed) random stream, 64 steps per interval (closer to the SDE)
#   lcg8   the fixture's dependency-free LCG stream (seed offset), 8 steps per interval
# Every fit uses the 8-substep first-order filter (about 5 minutes per panel).
#
# RESULT for the state-dependent shock correlation `cz` (generating value 0.70; mean
# posterior sd over panels 0.115) -- results/replication.tsv holds all 14 parameters:
#   generator   panels   mean     se
#   xo8          8       0.664    0.057
#   xo64         6       0.648    0.023
#   lcg8         7       0.576    0.053     (includes the default panel of the fit script)
#   all         21       0.630    0.029
#
# Run one panel:  julia --project=test research/ema_ctsem/ema_state_dependent_replicate.jl xo8 3 [outfile]
# Panels are independent, so run them in parallel as separate processes.

using BayesianRegressionModels, StanBlocks, LogDensityProblems, BridgeStan, Random, WarmupHMC
using Distributions: Normal, Exponential
import Statistics

include(joinpath(@__DIR__, "ema_state_dependent.jl"))   # filter family, fixture, with_filter, model

function panel_for(generator, seed)
    generator == "xo8"  && return ema_state_dependent_fixture(n=100, nt=30, rng=Xoshiro(seed))
    generator == "xo64" && return ema_state_dependent_fixture(n=100, nt=30, rng=Xoshiro(seed), ng=64)
    generator == "lcg8" && return ema_state_dependent_fixture(n=100, nt=30, seed=20260916 + seed)
    error("unknown generator `$generator` (xo8 | xo64 | lcg8)")
end

function main(generator, seed, outfile=nothing)
    panel = with_filter(panel_for(generator, seed); nsub=8, gh=0)
    sb = SBBRMI(ema_state_dependent_model(panel); mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    # one compiled copy per process: concurrent processes must not share an instantiation path
    prob = StanBlocks.stan_instantiate(sb.model;
        path=joinpath(tempdir(), "ema_sd_rep_$(generator)_$(seed)_$(hash(code)).stan"))
    fit = WarmupHMC.adaptive_warmup_mcmc(Xoshiro(1), prob; n_draws=600, progress=nothing)
    names = BridgeStan.param_names(prob.model)
    C = reduce(hcat, [BridgeStan.param_constrain(prob.model, collect(Float64, c))
                      for c in eachcol(fit.posterior_position)])
    rows = String[]
    for k in ("b0","bm","a12","a21","a22","cintm","qd0","qd1","cz","sdm","l31","thr","r1","r2")
        v = C[findfirst(==(k), names), :]
        push!(rows, join((generator, seed, k, round(Statistics.mean(v); digits=4),
                          round(Statistics.std(v); digits=4)), '\t'))
    end
    foreach(println, rows)
    outfile === nothing || open(io -> foreach(r -> println(io, r), rows), outfile, "a")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1], parse(Int, ARGS[2]), length(ARGS) > 2 ? ARGS[3] : nothing)
end
