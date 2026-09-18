# Bambi replication: https://bambinos.github.io/bambi/notebooks/Strack_RRR_re_analysis.html
# Run: julia --project=. strack.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# bambi Strack_RRR_re_analysis: facial-feedback RRR, naive
# value ~ condition + (1|uid) vs maximal
# value ~ condition + age + gender + (1|uid) + (condition|study) + (condition|stimulus).
# Answer: no discernible condition effect either way.
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "strack")
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

df = CSV.read(joinpath(SCRATCH, "data", "rrr_long.csv"), DataFrame)
codes(v) = (u = sort(unique(v)); m = Dict(x => i for (i, x) in enumerate(u)); [m[x] for x in v])
f64(v) = Float64.(coalesce.(v, NaN))  # CSV columns may be Union{Missing,Float64}
value = f64(df.value)
cond = f64(df.condition)
uid = codes(f64(df.uid))
# model 1: drop rows with missing value (notebook: 9/6940)
m1 = .!isnan.(value)
println("M1_N=", sum(m1), " NUID=", length(unique(uid[m1])))
b1 = @brm begin
    mu ~ 1 + condition + (1 | uid)
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, condition) ~ Normal(0, 2.5)
    sigma ~ Exponential(1)
    value ~ Normal(mu, sigma)
end
sb = SBBRMI(b1((; value=value[m1], condition=cond[m1], uid=uid[m1])); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "strack_m1.stan"))
f1 = sample_fit(sb, problem, 36001)
open(joinpath(OUT, "strack_m1.json"), "w") do io
    JSON.print(io, f1)
end
println("STRACK_M1_DONE"); flush(stdout)

# model 2: + age + gender + (condition|study) + (condition|stimulus), dropna (33/6940)
age = f64(df.age); gender = f64(df.gender)
mage = sum(filter(isfinite, age)) / sum(isfinite, age)
agez = (age .- mage) ./ sqrt(sum((filter(isfinite, age) .- mage) .^ 2) / sum(isfinite, age))
open(joinpath(OUT, "strack_scale.json"), "w") do io
    JSON.print(io, Dict("mean" => mage))
end
m2 = .!isnan.(value) .& .!isnan.(age) .& .!isnan.(gender)
println("M2_N=", sum(m2), " DROPPED=", length(m2) - sum(m2))
study = codes(string.(df.study)); stim = codes(string.(df.stimulus))
b2 = @brm begin
    mu ~ 1 + condition + agez + gender + (1 | uid) + (condition | study) + (condition | stimulus)
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, condition) ~ Normal(0, 2.5)
    effect(mu, agez) ~ Normal(0, 2.5)
    effect(mu, gender) ~ Normal(0, 2.5)
    sigma ~ Exponential(1)
    value ~ Normal(mu, sigma)
end
sb = SBBRMI(b2((; value=value[m2], condition=cond[m2], agez=agez[m2],
    gender=gender[m2], uid=uid[m2], study=study[m2], stimulus=stim[m2])); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "strack_m2.stan"))
f2 = sample_fit(sb, problem, 36002)
open(joinpath(OUT, "strack_m2.json"), "w") do io
    JSON.print(io, f2)
end
println("STRACK_M2_DONE"); flush(stdout)
