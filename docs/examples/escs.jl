# Bambi replication: https://bambinos.github.io/bambi/notebooks/ESCS_multiple_regression.html
# Run: julia --project=. escs.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Bambi ESCS_multiple_regression replication with BRM.
# Bambi: drugs ~ o + c + e + a + n (gaussian), tune=2000 draws=2000.
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "escs")
mkpath(OUT)

df = CSV.read(joinpath(SCRATCH, "data", "ESCS.csv"), DataFrame)
data = (;
    drugs=Float64.(df.drugs), o=Float64.(df.o), c=Float64.(df.c),
    e=Float64.(df.e), a=Float64.(df.a), n=Float64.(df.n),
)
builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + o + c + e + a + n
    drugs ~ Normal(mu, sigma)
end
d = BRM.brm_descriptor(builder, data; mod=@__MODULE__, name=:escs)
sb = SBBRMI(builder(data); mod=@__MODULE__)
stan_path = joinpath(SCRATCH, ".out", "stan", "escs.stan")
mkpath(dirname(stan_path))
problem = StanBlocks.stan_instantiate(sb.model; path=stan_path)
# No random effects: sample the bare StanProblem (no adaptive-centering wrap).
fit = adaptive_warmup_mcmc(
    Xoshiro(2608), problem; init=zeros(LogDensityProblems.dimension(problem)),
    n_draws=1000, nonlinear_adapt=false, monitor_ess=false,
)
P = fit.posterior_position
con_names = BridgeStan.param_names(problem.model)
C = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[:, i])) for i in axes(P, 2)]...))
rows = map(eachindex(con_names)) do i
    x = C[:, i]
    Dict("name" => string(con_names[i]), "mean" => mean(x), "sd" => std(x),
        "q05" => quantile(x, 0.05), "q50" => quantile(x, 0.5), "q95" => quantile(x, 0.95))
end
open(joinpath(OUT, "summary.json"), "w") do io
    JSON.print(io, Dict("n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples, "params" => rows))
end
println("ESCS_DONE draws=", size(P, 2))
