# Bambi replication: https://bambinos.github.io/bambi/notebooks/logistic_regression.html
# Run: julia --project=. logistic.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Bambi logistic_regression replication with BRM (SBBRMI).
# Bambi: vote['clinton'] ~ party_id + party_id:age, ANES pilot, bernoulli.
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "logistic")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

df = CSV.read(joinpath(SCRATCH, "data", "ANES_2016_pilot.csv"), DataFrame)
df = df[in.(string.(df.vote), Ref(["clinton", "trump"])), :]
codes(v) = (u = sort(unique(v)); m = Dict(x => i for (i, x) in enumerate(u)); [m[x] for x in v])
# Int response (SBBRMI Bernoulli requirement); O(1)-ish age via /10.
pid = codes(string.(df.party_id))
age10 = Float64.(df.age) ./ 10
# Bambi `party_id:age` keeps one age slope per level (no age main effect);
# BRM `&` is treatment-coded (K-1). Precompute the three level slopes exactly.
data = (;
    vote=[v == "clinton" ? 1 : 0 for v in string.(df.vote)],
    age_dem=[p == 1 ? a : 0.0 for (p, a) in zip(pid, age10)],
    age_ind=[p == 2 ? a : 0.0 for (p, a) in zip(pid, age10)],
    age_rep=[p == 3 ? a : 0.0 for (p, a) in zip(pid, age10)],
    party_id=pid,
)
println("N=", length(data.vote), " rate=", sum(data.vote) / length(data.vote),
    " parties=", sort(unique(data.party_id)))

builder = @brm begin
    mu ~ 1 + party_id + age_dem + age_ind + age_rep
    vote ~ BernoulliLogit(mu)
end
sb = SBBRMI(builder(data); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "logistic.stan"))
fit = adaptive_warmup_mcmc(
    Xoshiro(6126), problem; init=missing, n_draws=800,
    nonlinear_adapt=false, monitor_ess=false,
)
P = fit.posterior_position
println("DRAWS=", size(P), " DIV=", fit.n_divergent_samples)
con_names = BridgeStan.param_names(problem.model)
C = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[:, i])) for i in axes(P, 2)]...))
keep = findall(n -> (s = string(n); occursin("beta_pop", s) || occursin("cat_", s) || s == "sigma"), con_names)
rows = map(keep) do i
    v = C[:, i]
    Dict("name" => string(con_names[i]), "mean" => mean(v), "sd" => std(v),
        "q05" => quantile(v, 0.05), "q50" => quantile(v, 0.5), "q95" => quantile(v, 0.95))
end
open(joinpath(OUT, "summary.json"), "w") do io
    JSON.print(io, Dict("n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples, "params" => rows))
end
println("LOGISTIC_DONE")
flush(stdout)
