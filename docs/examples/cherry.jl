# Bambi replication: https://bambinos.github.io/bambi/notebooks/splines_cherry_blossoms.html
# Run: julia --project=. cherry.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Bambi splines_cherry_blossoms replication with BRM.
# Bambi: doy ~ bs(year, knots, intercept=True) (B-spline basis, McElreath).
# BRM: doy ~ s(year) (penalized thin-plate-style smooth) — same model class
# (smooth regression of bloom day on year), different basis construction.
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "cherry")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

df = CSV.read(joinpath(SCRATCH, "data", "cherry_blossoms.csv"), DataFrame;
    missingstring="NA")
dropmissing!(df, [:year, :doy])
yr = Float64.(df.year); dy = Float64.(df.doy)
println("N=", length(yr), " year range=", extrema(yr), " doy mean=", sum(dy) / length(dy))
# O(1) response (WHMC discipline); center year for a stable null space.
data = (; doy=dy ./ 100, year=(yr .- sum(yr) / length(yr)) ./ 100)

builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + s(year)
    doy ~ Normal(mu, sigma)
end
sb = SBBRMI(builder(data); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "cherry.stan"))
# Smooth-only model: no ranef blocks, so no adaptive-centering wrap.
fit = adaptive_warmup_mcmc(
    Xoshiro(5452), problem; init=missing, n_draws=800,
    nonlinear_adapt=false, monitor_ess=false,
)
P = fit.posterior_position
println("DRAWS=", size(P), " DIV=", fit.n_divergent_samples)
con_names = BridgeStan.param_names(problem.model; include_tp=true)
C = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[:, i]); include_tp=true) for i in axes(P, 2)]...))
d = BRM.brm_descriptor(builder, data; mod=@__MODULE__, name=:cherry)
mu_draws = BRM.brm_output_draws(d, C, con_names; logical=:mu)
println("MU=", size(mu_draws))
open(joinpath(OUT, "mu_quantiles.json"), "w") do io
    JSON.print(io, Dict(
        "year" => yr, "doy" => dy,
        "q05" => [quantile(mu_draws[:, i], 0.05) * 100 for i in axes(mu_draws, 2)],
        "q50" => [quantile(mu_draws[:, i], 0.50) * 100 for i in axes(mu_draws, 2)],
        "q95" => [quantile(mu_draws[:, i], 0.95) * 100 for i in axes(mu_draws, 2)],
    ))
end
ix = findfirst(==("sigma"), string.(con_names))
sv = C[:, ix] .* 100
open(joinpath(OUT, "summary.json"), "w") do io
    JSON.print(io, Dict("n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples,
        "sigma" => Dict("mean" => mean(sv), "q05" => quantile(sv, 0.05), "q95" => quantile(sv, 0.95))))
end
println("CHERRY_DONE")
