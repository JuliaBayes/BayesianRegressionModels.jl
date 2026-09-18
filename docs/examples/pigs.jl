# Bambi replication: https://bambinos.github.io/bambi/notebooks/multi-level_regression.html
# Run: julia --project=. pigs.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Bambi multi-level_regression (pigs/dietox) replication with BRM.
# Bambi: Weight ~ Time + (Time|Pig) on geepack dietox.
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "pigs")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

df = CSV.read(joinpath(SCRATCH, "data", "dietox.csv"), DataFrame)
dropmissing!(df, [:Weight, :Time, :Pig])
codes(v) = (u = sort(unique(v)); m = Dict(x => i for (i, x) in enumerate(u)); [m[x] for x in v])
# Raw-Time model in bambi's frame, with wide DATA-DRIVEN population priors
# (bambi's own default strategy): N(0,1) defaults would sit 15sd out on the
# intercept and drag it along the -0.97 int-slope ridge (measured 15.7->10.8).
# Normal(60,30)/Normal(0,30) are flat over every plausible value.
Tmean = sum(Float64.(df.Time)) / nrow(df)
data = (;
    Weight=Float64.(df.Weight), Time=Float64.(df.Time), Pig=codes(df.Pig),
)
println("N=", length(data.Weight), " PIGS=", maximum(data.Pig), " Tmean=", Tmean)
open(joinpath(OUT, "tmean.json"), "w") do io
    JSON.print(io, Dict("tmean" => Tmean))
end

builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + Time + (1 + Time | Pig)
    effect(mu, Intercept) ~ Normal(60, 30)
    effect(mu, Time) ~ Normal(0, 30)
    Weight ~ Normal(mu, sigma)
end
sb = SBBRMI(builder(data); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "pigs.stan"))
rp = BRM.adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.0)
fit = adaptive_warmup_mcmc(
    Xoshiro(2611), rp; init=missing,
    n_draws=1000, nonlinear_adapt=true, monitor_ess=false,
)
P = fit.posterior_position
con_names = BridgeStan.param_names(problem.model)
C = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[:, i])) for i in axes(P, 2)]...))
pops = findall(n -> occursin("pop_", string(n)) || occursin("tau", string(n)) || occursin("_L.", string(n)) || string(n) in ("Intercept", "Time", "sigma"), con_names)
rows = map(pops) do i
    v = C[:, i]
    Dict("name" => string(con_names[i]), "mean" => mean(v), "sd" => std(v),
        "q05" => quantile(v, 0.05), "q50" => quantile(v, 0.5), "q95" => quantile(v, 0.95))
end
open(joinpath(OUT, "summary.json"), "w") do io
    JSON.print(io, Dict("n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples, "params" => rows))
end
# Raw-Time frame: intercept/slope read straight off the joint draws.
i1 = findfirst(==("pop_mu_beta_pop.1"), string.(con_names))
i2 = findfirst(==("pop_mu_beta_pop.2"), string.(con_names))
raw_int0 = C[:, i1]
raw_slope = C[:, i2]
open(joinpath(OUT, "intercept0.json"), "w") do io
    JSON.print(io, Dict(
        "int_mean" => mean(raw_int0), "int_sd" => std(raw_int0),
        "int_q05" => quantile(raw_int0, 0.05), "int_q50" => quantile(raw_int0, 0.5), "int_q95" => quantile(raw_int0, 0.95),
        "slope_mean" => mean(raw_slope), "slope_sd" => std(raw_slope),
        "slope_q05" => quantile(raw_slope, 0.05), "slope_q50" => quantile(raw_slope, 0.5), "slope_q95" => quantile(raw_slope, 0.95)))
end
println("PIGS_DONE")
