# Bambi replication: https://bambinos.github.io/bambi/notebooks/circular_regression.html
# Run: julia --project=. circular.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# bambi circular_regression: direction ~ distance, family=vonmises.
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "circular")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

function sample_fit(sb, problem, seed; draws=800)
    rp = try
        BRM.adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.0)
    catch
        problem
    end
    fit = adaptive_warmup_mcmc(
        Xoshiro(seed), rp; init=missing, n_draws=draws,
        nonlinear_adapt=false, monitor_ess=false,
    )
    P = fit.posterior_position
    con_names = BridgeStan.param_names(problem.model)
    C = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[:, i])) for i in axes(P, 2)]...))
    rows = map(eachindex(con_names)) do i
        v = C[:, i]
        Dict("name" => string(con_names[i]), "mean" => mean(v), "sd" => std(v),
            "q05" => quantile(v, 0.05), "q50" => quantile(v, 0.5), "q95" => quantile(v, 0.95))
    end
    Dict("n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples, "params" => rows,
        "names" => string.(con_names), "draws" => [C[:, i] for i in eachindex(con_names)])
end

pdf = CSV.read(joinpath(SCRATCH, "data", "periwinkles.csv"), DataFrame)
dirc = Float64.(pdf.direction)
dist0 = Float64.(pdf.distance)
md = sum(dist0) / length(dist0)
sd_ = sqrt(sum((dist0 .- md) .^ 2) / length(dist0))
distz = (dist0 .- md) ./ sd_
open(joinpath(OUT, "circ_scale.json"), "w") do io
    JSON.print(io, Dict("mean" => md, "std" => sd_))
end
b_c = @brm begin
    mu ~ 1 + distz
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, distz) ~ Normal(0, 2.5)
    log(kappa) ~ 1
    direction ~ CircularVonMises(mu, kappa; interval=(-pi, pi))
end
sb = SBBRMI(b_c((; direction=dirc, distz=distz)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "circ.stan"))
f = sample_fit(sb, problem, 9001)
open(joinpath(OUT, "circ.json"), "w") do io
    JSON.print(io, f)
end
println("CIRC_DONE")
flush(stdout)
