# Bambi replication: https://bambinos.github.io/bambi/notebooks/sleepstudy.html
# Run: julia --project=. sleepstudy.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Bambi sleepstudy replication with BRM.
# Bambi: Reaction ~ 1 + Days + (Days | Subject), sleepstudy data, draws=2000.
# Usage: julia --project=<nb env> sleepstudy.jl
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "sleepstudy")
mkpath(OUT)

df = CSV.read(joinpath(SCRATCH, "data", "sleepstudy.csv"), DataFrame)
codes(v) = (u = sort(unique(v)); m = Dict(x => i for (i, x) in enumerate(u)); [m[x] for x in v])
subj = codes(df.Subject)
data = (;
    Reaction = Float64.(df.Reaction) ./ 100,
    Days = Float64.(df.Days),
    Subject = subj,
)

builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + Days + (1 + Days | Subject)
    Reaction ~ Normal(mu, sigma)
end
d = BRM.brm_descriptor(builder, data; mod=@__MODULE__, name=:sleepstudy)
println("FORMULA: ", d.formula)

sb = SBBRMI(builder(data); mod=@__MODULE__)
stan_path = joinpath(SCRATCH, ".out", "stan", "sleepstudy.stan")
mkpath(dirname(stan_path))
problem = StanBlocks.stan_instantiate(sb.model; path=stan_path)
unc_names = BridgeStan.param_unc_names(problem.model)
println("UNC_DIM=", length(unc_names))

rp = BRM.adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.0)
fit = adaptive_warmup_mcmc(
    Xoshiro(1211), rp; init=zeros(LogDensityProblems.dimension(rp)),
    n_draws=1000, nonlinear_adapt=true, monitor_ess=false,
)
P = fit.posterior_position  # coords x draws
println("DRAWS=", size(P), " DIVERGENCES=", fit.n_divergent_samples)

# Constrain to the model frame (draws x coords), WITH transformed parameters
# so the `mu` linear predictor is available to brm_output_draws.
con_names = BridgeStan.param_names(problem.model; include_tp=true)
C = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[:, i]); include_tp=true) for i in axes(P, 2)]...))

# Scalar summaries for population coefficients + sigma.
# Constrained names: pop_mu_beta_pop.1 = Intercept, .2 = Days;
# r_mu_Subject_tau.1/.2 = Intercept/Days ranef SDs; sigma = residual SD.
function summarize(idx)
    x = C[:, idx]
    (mean=mean(x), sd=std(x), q05=quantile(x, 0.05), q50=quantile(x, 0.5), q95=quantile(x, 0.95))
end
summary = Dict{String,Any}()
for (label, pat) in ("Intercept" => "pop_mu_beta_pop.1", "Days" => "pop_mu_beta_pop.2",
        "sigma" => "sigma", "sd_intercept" => "r_mu_Subject_tau.1",
        "sd_days" => "r_mu_Subject_tau.2")
    hits = findall(n -> string(n) == pat, con_names)
    summary[label] = [merge((name=string(con_names[i]),), summarize(i)) for i in hits]
end
open(joinpath(OUT, "summary.json"), "w") do io
    JSON.print(io, Dict("n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples, "params" => summary))
end

# Linear-predictor draws via the descriptor (draws x observations).
mu_draws = BRM.brm_output_draws(d, C, con_names; logical=:mu)
println("MU_DRAWS=", size(mu_draws))
open(joinpath(OUT, "mu_quantiles.json"), "w") do io
    JSON.print(io, Dict(
        "days" => data.Days, "subject" => data.Subject, "reaction" => data.Reaction,
        "q05" => [quantile(mu_draws[:, i], 0.05) for i in axes(mu_draws, 2)],
        "q50" => [quantile(mu_draws[:, i], 0.50) for i in axes(mu_draws, 2)],
        "q95" => [quantile(mu_draws[:, i], 0.95) for i in axes(mu_draws, 2)],
    ))
end
println("SLEEPSTUDY_DONE")
