# Bambi replication: https://bambinos.github.io/bambi/notebooks/zero_inflated_regression.html
# Run: julia --project=. hurdle_only.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Batch5: bambi zero_inflated_regression (ZIP + hurdle halves; native HurdlePoisson).
# bambi psi = P(non-structural zero complement) = 1 - zi_BRM: all five fitted
# zi-submodel coefs match bambi's psi coefs in magnitude with flipped signs
# (BRM zi = P(extra zero), verified in the emitted zero_inflated_poisson_lpmf).
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme
using LogExpFunctions: logit

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "zip")
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

zsc(v) = (v .- sum(v) / length(v)) ./ sqrt(sum((v .- sum(v) / length(v)) .^ 2) / length(v))
fdf = CSV.read(joinpath(SCRATCH, "data", "fish.csv"), DataFrame)
fdf = fdf[fdf.count .< 60, :]
yf = Int.(fdf.count)
lb = Float64.(fdf.livebait); camp = Float64.(fdf.camper)
pers0 = Float64.(fdf.persons); ch0 = Float64.(fdf.child)
persz = zsc(pers0); chz = zsc(ch0)
open(joinpath(OUT, "zip_scale.json"), "w") do io
    JSON.print(io, Dict(
        "persons" => Dict("mean" => sum(pers0) / length(pers0),
            "std" => sqrt(sum((pers0 .- sum(pers0) / length(pers0)) .^ 2) / length(pers0))),
        "child" => Dict("mean" => sum(ch0) / length(ch0),
            "std" => sqrt(sum((ch0 .- sum(ch0) / length(ch0)) .^ 2) / length(ch0))),
        "n" => length(yf)))
end

# --- model C: hurdle (native HurdlePoisson) ---
# BRM p_zero = P(Y=0) = 1 - psi_bambi: p_zero coefs compare vs NEGATED bambi psi.
b_h = @brm begin
    log(lambda) ~ 1 + livebait + camper + persz + chz
    effect(lambda, Intercept) ~ Normal(0, 5)
    effect(lambda, livebait) ~ Normal(0, 2.5)
    effect(lambda, camper) ~ Normal(0, 2.5)
    effect(lambda, persz) ~ Normal(0, 2.5)
    effect(lambda, chz) ~ Normal(0, 2.5)
    logit(p_zero) ~ 1 + livebait + camper + persz + chz
    effect(p_zero, Intercept) ~ Normal(0, 2)
    effect(p_zero, livebait) ~ Normal(0, 1)
    effect(p_zero, camper) ~ Normal(0, 1)
    effect(p_zero, persz) ~ Normal(0, 1)
    effect(p_zero, chz) ~ Normal(0, 1)
    count ~ HurdlePoisson(lambda, p_zero)
end
sb = SBBRMI(b_h((; count=yf, livebait=lb, camper=camp, persz=persz, chz=chz)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "zip_h.stan"))
f = sample_fit(sb, problem, 4003)
open(joinpath(OUT, "zip_h.json"), "w") do io
    JSON.print(io, f)
end
println("ZIP_H_DONE")
flush(stdout)
