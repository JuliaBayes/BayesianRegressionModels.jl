# Bambi replication: https://bambinos.github.io/bambi/notebooks/predict_new_groups.html
# Run: julia --project=. newgroups.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Bambi predict_new_groups replication with BRM.
# Bambi: fvc ~ 0 + weeks + smoking_status + (0 + weeks | patient) on normalized
# pulmonary-fibrosis data; predict for an UNSEEN patient (sample_new_groups).
# BRM route: reprocess(resample_groups=...) + predictive replay.
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "newgroups")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

df = CSV.read(joinpath(SCRATCH, "data", "pulmonary_fibrosis.csv"), DataFrame)
rename!(df, lowercase.(names(df)))
rename!(df, :smokingstatus => :smoking_status)
codes(v) = (u = sort(unique(v)); m = Dict(x => i for (i, x) in enumerate(u)); [m[x] for x in v])
w = Float64.(df.weeks); f = Float64.(df.fvc)
wn = (w .- minimum(w)) ./ (maximum(w) - minimum(w))
fn = (f .- minimum(f)) ./ (maximum(f) - minimum(f))
smoke = codes(string.(df.smoking_status))
data = (; fvc=fn, weeks=wn, smoking_status=smoke, patient=codes(string.(df.patient)))
println("N=", length(data.fvc), " PATIENTS=", maximum(data.patient),
    " SMOKE_LEVELS=", sort(unique(smoke)))

builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 0 + weeks + smoking_status + (0 + weeks | patient)
    fvc ~ Normal(mu, sigma)
end
sb = SBBRMI(builder(data); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "newgroups.stan"))
fit = adaptive_warmup_mcmc(
    Xoshiro(42), BRM.adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.0);
    init=missing, n_draws=800, nonlinear_adapt=true, monitor_ess=false,
)
P = fit.posterior_position
con_names = BridgeStan.param_names(problem.model; include_tp=true)
C = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[:, i]); include_tp=true) for i in axes(P, 2)]...))
keep = findall(n -> (s = string(n); startswith(s, "sigma") || startswith(s, "cat_mu") || startswith(s, "pop_mu") || occursin("tau", s)), con_names)
rows = map(keep) do i
    v = C[:, i]
    Dict("name" => string(con_names[i]), "mean" => mean(v), "sd" => std(v),
        "q05" => quantile(v, 0.05), "q50" => quantile(v, 0.5), "q95" => quantile(v, 0.95))
end
open(joinpath(OUT, "summary.json"), "w") do io
    JSON.print(io, Dict("n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples, "params" => rows))
end
println("NEWGROUPS_FIT_DONE draws=", size(P, 2))
flush(stdout)
