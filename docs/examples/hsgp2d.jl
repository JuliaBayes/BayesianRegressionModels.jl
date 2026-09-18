# Bambi replication: BRM extra: no bambi counterpart
# Run: julia --project=. hsgp2d.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Bambi hsgp_2d replication with BRM (plain 2-D variant first).
# Bambi: outcome ~ 0 + hsgp(x, y, c=1.5, m=10) on a 12x12 ExpQuad grid.
# BRM: outcome ~ 0 + hsgp(x, y; k=10, c=1.5) with LogNormal hyperpriors.
using Random, Statistics, Distributions, LinearAlgebra, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "hsgp2d")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

rng = Xoshiro(1234)
g1 = collect(range(0, 10; length=12))
X = [[x, y] for y in g1 for x in g1]
n = length(X)
D2 = [sum((X[i] .- X[j]) .^ 2) for i in 1:n, j in 1:n]
K = 1.2 .* exp.(-D2 ./ (2 * 2.0^2)) + 1e-8 * I
f = cholesky(Hermitian(K)).L * randn(rng, n)
xs = [p[1] for p in X]; ys = [p[2] for p in X]
data = (; x=xs, y=ys, outcome=f)

builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 0 + hsgp(x, y; k=10, c=1.5)
    length_scale(mu, hsgp(x, y)) ~ LogNormal(0, 1)
    sd(mu, hsgp(x, y)) ~ LogNormal(0, 1)
    outcome ~ Normal(mu, sigma)
end
sb = SBBRMI(builder(data); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "hsgp2d.stan"))
# No adaptive wrap: iso/grouped HSGPs are outside its supported shapes
# (HSGP latents are noncentered by construction).
fit = adaptive_warmup_mcmc(
    Xoshiro(1234), problem; init=missing, n_draws=600,
    nonlinear_adapt=false, monitor_ess=false,
)
P = fit.posterior_position
println("DRAWS=", size(P), " DIV=", fit.n_divergent_samples)
con_names = BridgeStan.param_names(problem.model; include_tp=true)
C = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[:, i]); include_tp=true) for i in axes(P, 2)]...))
d = BRM.brm_descriptor(builder, data; mod=@__MODULE__, name=:hsgp2d)
mu_draws = BRM.brm_output_draws(d, C, con_names; logical=:mu)
open(joinpath(OUT, "mu_quantiles.json"), "w") do io
    JSON.print(io, Dict(
        "x" => xs, "y" => ys, "f" => f,
        "q05" => [quantile(mu_draws[:, i], 0.05) for i in axes(mu_draws, 2)],
        "q50" => [quantile(mu_draws[:, i], 0.50) for i in axes(mu_draws, 2)],
        "q95" => [quantile(mu_draws[:, i], 0.95) for i in axes(mu_draws, 2)],
    ))
end
println("HSGP2D_DONE")
flush(stdout)
