# Bambi replication: BRM extra: no bambi counterpart
# Run: julia --project=. hsgp1d.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Bambi hsgp_1d replication with BRM.
# Bambi: y ~ 0 + hsgp(x, m=10, c=2) on simulated spline truth + N(0, 0.15).
# BRM: y ~ 0 + hsgp(x; k=10, c=2). Truth simulated analogously in Julia
# (smooth sinusoid mixture; patsy bs basis unavailable — noted in brief).
using Random, Statistics, Distributions, LinearAlgebra, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "hsgp1d")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

rng = Xoshiro(121195)
n = 100
x = collect(range(0, 50; length=n))
f = @. 1.5 * sin(2pi * x / 25) + 0.05 * x  # period-25 only: resolvable at k=10 like bambi's bs truth
y = f .+ randn(rng, n) .* 0.15
data = (; x=x, y=y)

builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 0 + hsgp(x; k=10, c=2)
    y ~ Normal(mu, sigma)
end
sb = SBBRMI(builder(data); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "hsgp1d.stan"))
rp = BRM.adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.0)
fit = adaptive_warmup_mcmc(
    Xoshiro(121195), rp; init=missing, n_draws=800,
    nonlinear_adapt=true, monitor_ess=false,
)
P = fit.posterior_position
println("DRAWS=", size(P), " DIV=", fit.n_divergent_samples)
con_names = BridgeStan.param_names(problem.model; include_tp=true)
C = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[:, i]); include_tp=true) for i in axes(P, 2)]...))
d = BRM.brm_descriptor(builder, data; mod=@__MODULE__, name=:hsgp1d)
mu_draws = BRM.brm_output_draws(d, C, con_names; logical=:mu)
open(joinpath(OUT, "mu_quantiles.json"), "w") do io
    JSON.print(io, Dict(
        "x" => x, "y" => y, "f" => f,
        "q05" => [quantile(mu_draws[:, i], 0.05) for i in axes(mu_draws, 2)],
        "q50" => [quantile(mu_draws[:, i], 0.50) for i in axes(mu_draws, 2)],
        "q95" => [quantile(mu_draws[:, i], 0.95) for i in axes(mu_draws, 2)],
    ))
end
println("HSGP1D_DONE")
flush(stdout)
