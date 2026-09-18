# Bambi replication: https://bambinos.github.io/bambi/notebooks/wald_gamma_glm.html
# Run: julia --project=. wald_only.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Batch5 rates: bambi wald_gamma_glm (gamma half) + beta_regression (4 models).
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme
using LogExpFunctions: logit

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "ratebeta")
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
# --- carclaims gamma (claim/1000; Wald half blocked, snag filed) ---
cdf = CSV.read(joinpath(SCRATCH, "data", "carclaims.csv"), DataFrame)
cdf = cdf[cdf.claimcst0 .> 0, :]
ck = Float64.(cdf.claimcst0) ./ 1000.0
age = Int.(cdf.agecat)
dummies = Dict{Symbol,Vector{Float64}}()
for k in 2:6
    dummies[Symbol("age$k")] = Float64.(age .== k)
end
dummies[:male] = Float64.(string.(cdf.gender) .== "M")
for a in ["B", "C", "D", "E", "F"]
    dummies[Symbol("area$a")] = Float64.(string.(cdf.area) .== a)
end
b_w = @brm begin
    eta ~ 1 + age2 + age3 + age4 + age5 + age6 + male + areaB + areaC + areaD + areaE + areaF
    effect(eta, Intercept) ~ Normal(0, 2.5)
    effect(eta, age2) ~ Normal(0, 2.5)
    effect(eta, age3) ~ Normal(0, 2.5)
    effect(eta, age4) ~ Normal(0, 2.5)
    effect(eta, age5) ~ Normal(0, 2.5)
    effect(eta, age6) ~ Normal(0, 2.5)
    effect(eta, male) ~ Normal(0, 2.5)
    effect(eta, areaB) ~ Normal(0, 2.5)
    effect(eta, areaC) ~ Normal(0, 2.5)
    effect(eta, areaD) ~ Normal(0, 2.5)
    effect(eta, areaE) ~ Normal(0, 2.5)
    effect(eta, areaF) ~ Normal(0, 2.5)
    lambda ~ LogNormal(-0.3, 1.0)
    claim ~ InverseGaussian(exp(eta), lambda)
end
sb = SBBRMI(b_w((; claim=ck, dummies...)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "wald_claim.stan"))
f = sample_fit(sb, problem, 3006)
open(joinpath(OUT, "wald.json"), "w") do io
    JSON.print(io, f)
end
println("WALD_DONE")
flush(stdout)
