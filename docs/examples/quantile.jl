# Bambi replication: https://bambinos.github.io/bambi/notebooks/quantile_regression.html
# Run: julia --project=. quantile.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# bambi quantile_regression: bmi ~ s(age), AsymmetricLaplace likelihood.
# BRM's native SkewDoubleExponential(mu, sigma, tau) IS this family, with tau
# the quantile directly (bambi fixes kappa = sqrt(q/(1-q)) per fit instead).
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "quantile")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

df = CSV.read(joinpath(SCRATCH, "data", "bmi.csv"), DataFrame)
age = Float64.(df.age); bmi = Float64.(df.bmi)
println("N=", length(age), " age range=", extrema(age))
mage = sum(age) / length(age)
sage = sqrt(sum((age .- mage) .^ 2) / length(age))
agez = (age .- mage) ./ sage
open(joinpath(OUT, "quant_scale.json"), "w") do io
    JSON.print(io, Dict("mean" => mage, "std" => sage))
end

# Named numeric constants ride in data (snag bambi-quantile-r-36bd8217): one
# builder, `tau` supplied per fit through the data container. A bare scope
# name would be shadowed by the builder's data-side binding and rejected at
# trace time, so the earlier Core.eval literal-splice is gone.
builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + s(agez)
    bmi ~ SkewDoubleExponential(mu, sigma, tau)
end
for tau in (0.1, 0.5, 0.9)
    data = (; bmi=bmi ./ 10, agez=agez, tau=tau)
    sb = SBBRMI(builder(data); mod=@__MODULE__)
    tag = replace(string(tau), "." => "p")
problem = StanBlocks.stan_instantiate(sb.model;
        path=joinpath(SCRATCH, ".out", "stan", "quant_$tag.stan"))
    fit = adaptive_warmup_mcmc(
        Xoshiro(9000 + Int(round(tau * 100))), problem; init=missing,
        n_draws=800, nonlinear_adapt=false, monitor_ess=false,
    )
    P = fit.posterior_position
    println("TAU=$tau DRAWS=", size(P), " DIV=", fit.n_divergent_samples)
    con_names = BridgeStan.param_names(problem.model; include_tp=true)
    C = permutedims(hcat([BridgeStan.param_constrain(problem.model,
        collect(P[:, i]); include_tp=true) for i in axes(P, 2)]...))
    d = BRM.brm_descriptor(builder, data; mod=@__MODULE__, name=Symbol("quant_$tag"))
    mu_draws = BRM.brm_output_draws(d, C, con_names; logical=:mu)
    ix = findfirst(==("sigma"), string.(con_names))
    sv = C[:, ix] .* 10
    open(joinpath(OUT, "quant_$tag.json"), "w") do io
        JSON.print(io, Dict(
            "n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples,
            "tau" => tau, "age" => age, "bmi" => bmi,
            "q05" => [quantile(mu_draws[:, i], 0.05) * 10 for i in axes(mu_draws, 2)],
            "q50" => [quantile(mu_draws[:, i], 0.50) * 10 for i in axes(mu_draws, 2)],
            "q95" => [quantile(mu_draws[:, i], 0.95) * 10 for i in axes(mu_draws, 2)],
            "sigma" => Dict("mean" => mean(sv), "q05" => quantile(sv, 0.05),
                "q95" => quantile(sv, 0.95))))
    end
end
println("QUANT_DONE")
flush(stdout)
