# Bambi replication: https://bambinos.github.io/bambi/notebooks/prior_sensitivity.html
# Run: julia --project=. prior_sensitivity.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Body-fat siri ~ 13 mean-centered covariates, two prior settings (notebook
# 00 informative StudentT-family / 01 defaults). Power-scaling sensitivity via
# brm_powerscale_inputs/sensitivity (needs GQs saved); refit comparison where
# the helper's v1 surface does not reach (StudentT priors).
using Random, Statistics, Distributions, LinearAlgebra, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
const BS = BridgeStan
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "priorsens")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

df = CSV.read(joinpath(SCRATCH, "data", "body_fat.csv"), DataFrame)
y = Float64.(df.siri)
covars = [n for n in names(df) if n != "siri"]
X = hcat([Float64.(df[!, c]) .- sum(Float64.(df[!, c])) / nrow(df) for c in covars]...)
println("N=", nrow(df), " P=", length(covars))
data = merge((; siri=y), NamedTuple(Symbol(c) => X[:, i] for (i, c) in enumerate(covars)))

function sample_gq(builder, dat, tag, seed; draws=800)
    # BRM trace entries re-enter via invokelatest internally, so in-function
    # build + instantiate is call-site safe (snag two-sbbrmi-fits-f2beca06).
    sb = SBBRMI(builder(dat); mod=@__MODULE__)
    problem = BRM.stan_instantiate(sb;
        path=joinpath(SCRATCH, ".out", "stan", "psens_$tag.stan"))
    rp = try
        BRM.adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.0)
    catch
        problem
    end
    fit = adaptive_warmup_mcmc(Xoshiro(seed), rp; init=missing, n_draws=draws,
        nonlinear_adapt=false, monitor_ess=false)
    P = fit.posterior_position
    names = BS.param_names(problem.model; include_tp=true, include_gq=true)
    C = permutedims(hcat([BS.param_constrain(problem.model, collect(P[:, i]);
        include_tp=true, include_gq=true,
        rng=BS.new_rng(problem.model, seed + i)) for i in axes(P, 2)]...))
    println("TAG=$tag DRAWS=", size(P), " DIV=", fit.n_divergent_samples,
        " C=", size(C), " NAMES=", length(names))
    return sb, C, names, fit.n_divergent_samples
end

# ---------- fit B: defaults (helper-covered surface) ----------
b_B = @brm begin
    mu ~ 1 + age + weight + height + neck + chest + abdomen + hip + thigh + knee + ankle + biceps + forearm + wrist
    sigma ~ Exponential(1)
    siri ~ Normal(mu, sigma)
end
sbB, CB, namesB, divB = sample_gq(b_B, data, "def", 41001)
inputsB = BRM.brm_powerscale_inputs(sbB, CB, namesB)
sB = BRM.brm_powerscale_sensitivity(CB, inputsB.log_prior, inputsB.log_lik, namesB;
    variables=inputsB.variables)
rows = [Dict("variable" => string(v), "prior" => sB.prior[i],
    "likelihood" => sB.likelihood[i], "diagnosis" => string(sB.diagnosis[i]))
        for (i, v) in enumerate(inputsB.variables)]
open(joinpath(OUT, "psense_def.json"), "w") do io
    JSON.print(io, Dict("divergences" => divB, "rows" => rows,
        "pareto_k" => Dict("prior_lower" => sB.pareto_k.prior_lower,
            "prior_upper" => sB.pareto_k.prior_upper,
            "likelihood_lower" => sB.pareto_k.likelihood_lower,
            "likelihood_upper" => sB.pareto_k.likelihood_upper),
        "reliable" => Dict("prior" => sB.reliable.prior, "likelihood" => sB.reliable.likelihood)))
end
println("PSENS_DEF_DONE")

# wrist trajectory under alpha in {0.8, 1, 1.25}, prior and likelihood halves
wi = findfirst(==("wrist"), string.(BRM.popcoefnames(sbB.parent, :mu)))
wcol = let
    hits = [j for (j, nm) in enumerate(string.(namesB)) if nm == "pop_mu_beta_pop.$wi"]
    CB[:, only(hits)]
end
traj = Dict{String,Any}[]
for (lab, ell) in (("prior", inputsB.log_prior), ("likelihood", inputsB.log_lik))
    for alpha in (0.8, 1.0, 1.25)
        w = BRM.brm_powerscale_weights(ell; alpha=alpha).weights
        m = sum(w .* wcol)
        v = sum(w .* (wcol .- m) .^ 2)
        push!(traj, Dict("half" => lab, "alpha" => alpha, "mean" => m, "sd" => sqrt(v)))
    end
end
open(joinpath(OUT, "wrist_traj.json"), "w") do io
    JSON.print(io, traj)
end

# derived quantities on fit B: profile predictions, R2, log score
beta_cols = [CB[:, j] for (j, nm) in enumerate(string.(namesB)) if occursin(r"^pop_mu_beta_pop\.\d+$", nm)]
pnames = string.(BRM.popcoefnames(sbB.parent, :mu))
@assert length(beta_cols) == length(pnames) == 14
Beta = hcat(beta_cols...)  # draws x 14, col1 = Intercept
prof = Dict("min" => [minimum(X[:, i]) for i in 1:size(X, 2)],
    "median" => [Statistics.median(X[:, i]) for i in 1:size(X, 2)],
    "max" => [maximum(X[:, i]) for i in 1:size(X, 2)])
derived = Dict{String,Vector{Float64}}()
for (k, p) in prof
    xv = vcat(1.0, p)
    derived["at_$k"] = [dot(Beta[d, :], xv) for d in axes(Beta, 1)]
end
sigcol = CB[:, findfirst(==("sigma"), string.(namesB))]
X1 = hcat(ones(size(X, 1)), X)
mu_draws = X1 * permutedims(Beta)  # nobs x draws
# bayesian R2 per draw (Gelman et al.): var(mu)/(var(mu) + sigma^2)
derived["r2_score"] = [let v = var(mu_draws[:, d]); v / (v + sigcol[d]^2) end
    for d in axes(mu_draws, 2)]
derived["log_score"] = vec(inputsB.log_lik)  # joint loglik per draw
dnames = collect(keys(derived))
D = hcat([derived[k] for k in dnames]...)
sD = BRM.brm_powerscale_sensitivity(D, inputsB.log_prior, inputsB.log_lik, dnames)
drows = [Dict("variable" => dnames[i], "prior" => sD.prior[i],
    "likelihood" => sD.likelihood[i], "diagnosis" => string(sD.diagnosis[i]))
        for i in eachindex(dnames)]
open(joinpath(OUT, "derived_sens.json"), "w") do io
    JSON.print(io, Dict("rows" => drows))
end
println("PSENS_DERIVED_DONE")

# ---------- fit A: notebook-00 informative priors + refit comparison ----------
b_A = @brm begin
    mu ~ 1 + age + weight + height + neck + chest + abdomen + hip + thigh + knee + ankle + biceps + forearm + wrist
    effect(mu, Intercept) ~ LocationScale(0, 9.2, TDist(3))
    effect(mu, age) ~ Normal(0, 1)
    effect(mu, weight) ~ Normal(0, 1)
    effect(mu, height) ~ Normal(0, 1)
    effect(mu, neck) ~ Normal(0, 1)
    effect(mu, chest) ~ Normal(0, 1)
    effect(mu, abdomen) ~ Normal(0, 1)
    effect(mu, hip) ~ Normal(0, 1)
    effect(mu, thigh) ~ Normal(0, 1)
    effect(mu, knee) ~ Normal(0, 1)
    effect(mu, ankle) ~ Normal(0, 1)
    effect(mu, biceps) ~ Normal(0, 1)
    effect(mu, forearm) ~ Normal(0, 1)
    effect(mu, wrist) ~ Normal(0, 1)
    sigma ~ truncated(LocationScale(0, 9.2, TDist(3)); lower=0)
    siri ~ Normal(mu, sigma)
end
sbA, CA, namesA, divA = sample_gq(b_A, data, "info", 41002)
helper_note = try
    inA = BRM.brm_powerscale_inputs(sbA, CA, namesA)
    "helper-accepted"
catch e
    "helper-gate: " * sprint(showerror, e)[1:min(200, length(sprint(showerror, e)))]
end
println("HELPER_A: ", helper_note)
# refit comparison: 5x-wider common prior, compare wrist marginals
b_Aw = @brm begin
    mu ~ 1 + age + weight + height + neck + chest + abdomen + hip + thigh + knee + ankle + biceps + forearm + wrist
    effect(mu, Intercept) ~ LocationScale(0, 9.2, TDist(3))
    effect(mu, age) ~ Normal(0, 5)
    effect(mu, weight) ~ Normal(0, 5)
    effect(mu, height) ~ Normal(0, 5)
    effect(mu, neck) ~ Normal(0, 5)
    effect(mu, chest) ~ Normal(0, 5)
    effect(mu, abdomen) ~ Normal(0, 5)
    effect(mu, hip) ~ Normal(0, 5)
    effect(mu, thigh) ~ Normal(0, 5)
    effect(mu, knee) ~ Normal(0, 5)
    effect(mu, ankle) ~ Normal(0, 5)
    effect(mu, biceps) ~ Normal(0, 5)
    effect(mu, forearm) ~ Normal(0, 5)
    effect(mu, wrist) ~ Normal(0, 5)
    sigma ~ truncated(LocationScale(0, 9.2, TDist(3)); lower=0)
    siri ~ Normal(mu, sigma)
end
sbAw, CAw, namesAw, divAw = sample_gq(b_Aw, data, "infowide", 41003)
function wrist_marginal(C, names, sb)
    wi = findfirst(==("wrist"), string.(BRM.popcoefnames(sb.parent, :mu)))
    C[:, only([j for (j, nm) in enumerate(string.(names)) if nm == "pop_mu_beta_pop.$wi"])]
end
wA, wAw = wrist_marginal(CA, namesA, sbA), wrist_marginal(CAw, namesAw, sbAw)
open(joinpath(OUT, "refit_A.json"), "w") do io
    JSON.print(io, Dict("divergences" => [divA, divAw], "helper" => helper_note,
        "wrist_base" => Dict("mean" => mean(wA), "sd" => std(wA)),
        "wrist_wide" => Dict("mean" => mean(wAw), "sd" => std(wAw))))
end
println("PSENS_DONE")
flush(stdout)
