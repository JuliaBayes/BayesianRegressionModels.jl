# Bambi replication: https://bambinos.github.io/bambi/notebooks/t_regression.html
# Run: julia --project=. t_regression.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Bambi t_regression replication with BRM (gaussian vs Student-t robust fit).
# Design mirrors the notebook: y = 1 + 2x + N(0, 0.5), 3 outliers at
# (0.1, 8), (0.15, 6), (0.2, 9). RNG differs (Julia vs numpy) — noted in brief.
using Random, Statistics, Distributions, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "t_regression")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

rng = Xoshiro(1211)
n = 100
x = collect(range(0, 1; length=n))
y = 1 .+ 2 .* x .+ randn(rng, n) .* 0.5
data = (; x=vcat(x, [0.1, 0.15, 0.2]), y=vcat(y, [8.0, 6.0, 9.0]))

function sample(builder_fn, stan_name, seed)
    sb = SBBRMI(builder_fn(data); mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", stan_name))
    # No random effects: sample the bare StanProblem.
    fit = adaptive_warmup_mcmc(
        Xoshiro(seed), problem; init=zeros(LogDensityProblems.dimension(problem)),
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
    Dict("n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples, "params" => rows)
end

if !isfile(joinpath(OUT, "gauss.json"))
    gauss_builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    open(joinpath(OUT, "gauss.json"), "w") do io
        JSON.print(io, sample(gauss_builder, "treg_gauss.stan", 1211))
    end
    println("TREG_GAUSS_DONE")
    flush(stdout)
else
    println("TREG_GAUSS_CACHED")
end

t_builder = @brm begin
    sigma ~ Exponential(1)
    nu ~ Gamma(2, 0.1)
    mu ~ 1 + x
    y ~ LocationScale(mu, sigma, TDist(nu))
end
open(joinpath(OUT, "studentt.json"), "w") do io
    JSON.print(io, sample(t_builder, "treg_studentt.stan", 1212))
end
println("TREG_STUDENTT_DONE")
