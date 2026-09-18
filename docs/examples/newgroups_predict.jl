# Bambi replication: https://bambinos.github.io/bambi/notebooks/predict_new_groups.html
# Run: julia --project=. newgroups_predict.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Bambi predict_new_groups replication with BRM, part 2: unseen-patient prediction.
# Fits the base model, replays the declaration on a new-patient cohort via
# brm_execute(d, :replay, new_df), and predicts with the fitted draws.
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
pat = codes(string.(df.patient))
data = (; fvc=fn, weeks=wn, smoking_status=smoke, patient=pat)

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
println("FIT draws=", size(P), " div=", fit.n_divergent_samples)
flush(stdout)

# New cohort: one unseen patient (code beyond fitted range) reusing an existing
# patient's schedule, mirroring the notebook's patient-39 -> 176 construction.
d = BRM.brm_descriptor(builder, data; mod=@__MODULE__, name=:newgroups)
src_patient = 39
rows = findall(==(src_patient), pat)
new_code = maximum(pat) + 1
new_df = (; fvc=fn[rows], weeks=wn[rows], smoking_status=smoke[rows],
    patient=fill(new_code, length(rows)))
d2 = BRM.brm_execute(d, :replay, new_df)
println("REPLAY groups: n=", length(new_df.weeks))
prob2 = BRM.brm_execute(d2, :fit)
unc = permutedims(P)  # draws x coords
unc_names = BridgeStan.param_unc_names(problem.model)
unc_new_names = BridgeStan.param_unc_names(prob2.model)
moved = BRM.transport_draws(sb, d2.plan, unc, unc_names, unc_new_names; rng=Xoshiro(176))
pred = BRM.brm_predictive_draws(d2, moved; problem=prob2, seed=176)
println("PREDICT keys=", keys(pred))
flush(stdout)
yv = pred[filter(k -> occursin("gen", string(k)) || occursin("fvc", string(k)), keys(pred))...]
open(joinpath(OUT, "newpatient.json"), "w") do io
    JSON.print(io, Dict(
        "weeks" => wn[rows], "fvc" => fn[rows],
        "q05" => [quantile(yv[:, i], 0.05) for i in axes(yv, 2)],
        "q50" => [quantile(yv[:, i], 0.50) for i in axes(yv, 2)],
        "q95" => [quantile(yv[:, i], 0.95) for i in axes(yv, 2)],
    ))
end
println("NEWGROUPS_PREDICT_DONE")
