# Bambi replication: https://bambinos.github.io/bambi/notebooks/polynomial_regression.html
# Run: julia --project=. polynomial.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Bambi polynomial_regression replication with BRM (falling-body gravity).
# Bambi: x ~ I(t**2) + 1. BRM has no poly() term; the squared column is
# precomputed (same model, bambi's `{t**2}` variation does exactly this).
using Random, Statistics, Distributions, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "polynomial")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

g = -9.81
t = collect(range(0, 2; length=100))
x_true = 0.5 .* g .* t .^ 2 .+ 50
rng = Xoshiro(1234)
x_obs = x_true .+ randn(rng, 100) .* 0.3
# O(1) response scale (WHMC-catalogue convention): truth becomes
# intercept 1.0, t2-slope -0.0981, sigma 0.006 — all inside default priors.
ys = x_obs ./ 50
data = (; t=t, t2=t .^ 2, x=ys)

builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + t2
    x ~ Normal(mu, sigma)
end
sb = SBBRMI(builder(data); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "poly.stan"))
# No random effects: sample the bare StanProblem.
# Data-driven init (unc order: log sigma, Intercept, t2-slope), mirroring
# bambi's data-scaled default prior means; init=zeros self-traps (big
# residuals -> big sigma -> flat likelihood, measured sigma=32 on raw scale).
init = [log(std(ys)), mean(ys), 0.0]
fit = adaptive_warmup_mcmc(
    Xoshiro(1234), problem; init=init,
    n_draws=1000, nonlinear_adapt=false, monitor_ess=false,
)
P = fit.posterior_position
con_names = BridgeStan.param_names(problem.model)
C = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[:, i])) for i in axes(P, 2)]...))
rows = map(eachindex(con_names)) do i
    v = C[:, i]
    Dict("name" => string(con_names[i]), "mean" => mean(v), "sd" => std(v),
        "q05" => quantile(v, 0.05), "q50" => quantile(v, 0.5), "q95" => quantile(v, 0.95))
end
open(joinpath(OUT, "summary.json"), "w") do io
    JSON.print(io, Dict("n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples,
        "true_g" => g, "true_h0" => 50, "params" => rows))
end
println("POLY_DONE")
