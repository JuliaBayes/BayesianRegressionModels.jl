# Bambi replication: https://bambinos.github.io/bambi/notebooks/sleepstudy.html
# Run: julia --project=. sleepstudy_pilot.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Sleepstudy pilot: verify BRM build + sample pipeline on bambi's sleepstudy data.
# Usage: julia --project=<nb env> sleepstudy_pilot.jl
using Random, Statistics, Distributions, CSV, DataFrames
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)

df = CSV.read(joinpath(SCRATCH, "data", "sleepstudy.csv"), DataFrame)
codes(v) = (u = sort(unique(v)); m = Dict(x => i for (i, x) in enumerate(u)); [m[x] for x in v])
data = (;
    Reaction = Float64.(df.Reaction) ./ 100,
    Days = Float64.(df.Days),
    Subject = codes(df.Subject),
)

builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + Days + (1 + Days | Subject)
    Reaction ~ Normal(mu, sigma)
end
brmi = builder(data)
sb = SBBRMI(brmi; mod=@__MODULE__)
stan_path = joinpath(SCRATCH, ".out", "stan", "sleepstudy.stan")
mkpath(dirname(stan_path))
problem = StanBlocks.stan_instantiate(sb.model; path=stan_path)
println("COMPILED dim=", LogDensityProblems.dimension(problem))
flush(stdout)

names = BridgeStan.param_unc_names(problem.model)
println("N_UNC_PARAMS=", length(names))

rp = BRM.adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.0)
d0 = LogDensityProblems.dimension(rp)
init = zeros(d0)
fit = adaptive_warmup_mcmc(
    Xoshiro(1211), rp; init=init, n_draws=200,
    nonlinear_adapt=true, monitor_ess=false,
)
println("DRAWS size=", size(fit.posterior_position))
println("DIVERGENCES=", fit.n_divergent_samples)
flush(stdout)
