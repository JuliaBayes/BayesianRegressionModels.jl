# Bambi replication: https://bambinos.github.io/bambi/notebooks/beta_regression.html
# Run: julia --project=. beta_only.jl   (from docs/examples/, after Pkg.instantiate())
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
# --- beta sim: Beta(1000,1000), intercept-only ---
rng = Xoshiro(11)
psim = rand(rng, Beta(1000, 1000), 10_000)
b_b0 = @brm begin
    logit(mu) ~ 1
    effect(mu, Intercept) ~ Normal(0, 2.5)
    kappa ~ Gamma(2.0, 1000.0)
    p ~ Beta(mu * kappa, (1 - mu) * kappa)
end
sb = SBBRMI(b_b0((; p=psim)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "beta_sim.stan"))
f = sample_fit(sb, problem, 3002)
open(joinpath(OUT, "beta_sim.json"), "w") do io
    JSON.print(io, f)
end
println("BETA_SIM_DONE")
flush(stdout)

# --- beta dirt DGP (seeded): alpha = 1000 + 5*dh - 5*dt ---
rng = Xoshiro(12)
halfnorm(s) = abs.(randn(rng, 1000) .* s)
dh = round.(halfnorm(25)); dt = round.(halfnorm(25))
alpha_d = 1000.0 .+ 5.0 .* dh .- 5.0 .* dt
pdirt = [rand(rng, Beta(a, 1000.0)) for a in alpha_d]
delta = Float64.(dh .- dt)
ddz = zsc(delta)
open(joinpath(OUT, "dirt_scale.json"), "w") do io
    JSON.print(io, Dict("mean" => sum(delta) / length(delta),
        "std" => sqrt(sum((delta .- sum(delta) / length(delta)) .^ 2) / length(delta))))
end
b_bd = @brm begin
    logit(mu) ~ 1 + ddz
    effect(mu, Intercept) ~ Normal(0, 2.5)
    effect(mu, ddz) ~ Normal(0, 2.5)
    kappa ~ Gamma(2.0, 1000.0)
    p ~ Beta(mu * kappa, (1 - mu) * kappa)
end
sb = SBBRMI(b_bd((; p=pdirt, ddz=ddz)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "beta_dirt.stan"))
f = sample_fit(sb, problem, 3003)
open(joinpath(OUT, "beta_dirt.json"), "w") do io
    JSON.print(io, f)
end
println("BETA_DIRT_DONE")
flush(stdout)

# --- batting averages (deterministic data) ---
bdf = CSV.read(joinpath(SCRATCH, "data", "Batting.csv"), DataFrame)
bdf.bavg = Float64.(bdf.H) ./ Float64.(bdf.AB)
bdf = bdf[(bdf.AB .> 100) .& (bdf.yearID .> 1990) .& (bdf.yearID .< 2018), :]
sort!(bdf, [:playerID, :yearID, :stint])
bdf.bavg_shift = [i == 1 ? missing : (bdf.playerID[i] == bdf.playerID[i-1] ? bdf.bavg[i-1] : missing) for i in 1:nrow(bdf)]
bdf0 = bdf[.!ismissing.(bdf.bavg_shift), :]
sh0 = Float64.(bdf0.bavg_shift)
shz = zsc(sh0)
open(joinpath(OUT, "bat_scale.json"), "w") do io
    JSON.print(io, Dict("mean" => sum(sh0) / length(sh0),
        "std" => sqrt(sum((sh0 .- sum(sh0) / length(sh0)) .^ 2) / length(sh0)),
        "n_full" => nrow(bdf), "n_shift" => nrow(bdf0)))
end
b_ba = @brm begin
    logit(mu) ~ 1
    effect(mu, Intercept) ~ Normal(0, 2.5)
    kappa ~ Gamma(2.0, 1000.0)
    bavg ~ Beta(mu * kappa, (1 - mu) * kappa)
end
sb = SBBRMI(b_ba((; bavg=Float64.(bdf.bavg))); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "beta_bat.stan"))
f = sample_fit(sb, problem, 3004)
open(joinpath(OUT, "beta_bat.json"), "w") do io
    JSON.print(io, f)
end
println("BETA_BAT_DONE")
flush(stdout)

b_bl = @brm begin
    logit(mu) ~ 1 + shz
    effect(mu, Intercept) ~ Normal(0, 2.5)
    effect(mu, shz) ~ Normal(0, 2.5)
    kappa ~ Gamma(2.0, 1000.0)
    bavg ~ Beta(mu * kappa, (1 - mu) * kappa)
end
sb = SBBRMI(b_bl((; bavg=Float64.(bdf0.bavg), shz=shz)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "beta_lag.stan"))
f = sample_fit(sb, problem, 3005)
open(joinpath(OUT, "beta_lag.json"), "w") do io
    JSON.print(io, f)
end
println("BETA_LAG_DONE")
flush(stdout)
