# Bambi replication: https://bambinos.github.io/bambi/notebooks/model_comparison.html
# Run: julia --project=. model_comparison.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# bambi model_comparison: adults income, 3 Bernoulli logistic models
# (linear / quadratic / cubic in scaled age+hours) compared by PSIS-LOO
# (elpd; az.compare equivalent) from BRM pointwise likelihoods + PSIS.jl,
# then P(>50K) over age for model 2.
using Random, Statistics, Distributions, CSV, DataFrames, JSON, LogExpFunctions
using PSIS
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "modelcomp")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

df = CSV.read(joinpath(SCRATCH, "data", "adults.csv"), DataFrame)
keep = (string.(df.race) .== "Black") .| (string.(df.race) .== "White")
df = df[keep, :]
println("N=", nrow(df))
y = Int.(string.(df.income) .== ">50K")
male = Float64.(string.(df.sex) .== "Male")
white = Float64.(string.(df.race) .== "White")
zsc(v) = (m = sum(v) / length(v); s = sqrt(sum((v .- m) .^ 2) / length(v)); ((v .- m) ./ s, m, s))
age0 = Float64.(df.age); hrs0 = Float64.(df.hs_week)
agez, agem, ages = zsc(age0); hrsz, hrsm, hrss = zsc(hrs0)
open(joinpath(OUT, "mc_scale.json"), "w") do io
    JSON.print(io, Dict("age" => Dict("mean" => agem, "std" => ages),
        "hrs" => Dict("mean" => hrsm, "std" => hrss)))
end
age2 = agez .^ 2; hrs2 = hrsz .^ 2; age3 = agez .^ 3; hrs3 = hrsz .^ 3

function fit_model(builder, data, tag, seed; draws=800)
    sb = SBBRMI(builder(data); mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model;
        path=joinpath(SCRATCH, ".out", "stan", "mc_$tag.stan"))
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
    # pointwise loglik for PSIS-LOO (generated quantities included)
    gq_names = BridgeStan.param_names(problem.model; include_gq=true)
    srng = BridgeStan.StanRNG(problem.model, seed)
    G = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[:, i]); include_gq=true, rng=srng) for i in axes(P, 2)]...))
    d = BRM.brm_descriptor(builder, data; mod=@__MODULE__, name=Symbol("mc_$tag"))
    LL = BRM.brm_output_draws(d, G, string.(gq_names); logical=:income, role=:pointwise_loglik)
    println("POINTWISE_", uppercase(tag), "=", size(LL))
    Dict("n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples,
        "params" => rows, "names" => string.(con_names),
        "draws" => [C[:, i] for i in eachindex(con_names)], "ll" => LL)
end

function psis_loo(ll)
    res = psis(-ll; normalize=true, warn=false)
    lw = res.log_weights
    pointwise = vec(logsumexp(ll .+ lw; dims=1))
    lpd = vec(logsumexp(ll; dims=1) .- log(size(ll, 1)))
    elpd = sum(pointwise)
    se = sqrt(length(pointwise) * var(pointwise))
    Dict("elpd" => elpd, "se" => se, "p_loo" => sum(lpd) - elpd,
        "kmax" => maximum(res.pareto_shape), "kbad" => sum(res.pareto_shape .> 0.7),
        "pointwise" => pointwise)
end

b1 = @brm begin
    eta ~ 1 + male + white + agez + hrsz
    effect(eta, Intercept) ~ Normal(0, 2.5)
    effect(eta, male) ~ Normal(0, 2.5)
    effect(eta, white) ~ Normal(0, 2.5)
    effect(eta, agez) ~ Normal(0, 2.5)
    effect(eta, hrsz) ~ Normal(0, 2.5)
    income ~ BernoulliLogit(eta)
end
b2 = @brm begin
    eta ~ 1 + male + white + agez + age2 + hrsz + hrs2
    effect(eta, Intercept) ~ Normal(0, 2.5)
    effect(eta, male) ~ Normal(0, 2.5)
    effect(eta, white) ~ Normal(0, 2.5)
    effect(eta, agez) ~ Normal(0, 2.5)
    effect(eta, age2) ~ Normal(0, 2.5)
    effect(eta, hrsz) ~ Normal(0, 2.5)
    effect(eta, hrs2) ~ Normal(0, 2.5)
    income ~ BernoulliLogit(eta)
end
b3 = @brm begin
    eta ~ 1 + male + white + agez + age2 + age3 + hrsz + hrs2 + hrs3
    effect(eta, Intercept) ~ Normal(0, 2.5)
    effect(eta, male) ~ Normal(0, 2.5)
    effect(eta, white) ~ Normal(0, 2.5)
    effect(eta, agez) ~ Normal(0, 2.5)
    effect(eta, age2) ~ Normal(0, 2.5)
    effect(eta, age3) ~ Normal(0, 2.5)
    effect(eta, hrsz) ~ Normal(0, 2.5)
    effect(eta, hrs2) ~ Normal(0, 2.5)
    effect(eta, hrs3) ~ Normal(0, 2.5)
    income ~ BernoulliLogit(eta)
end
data = (; income=y, male=male, white=white, agez=agez, hrsz=hrsz,
    age2=age2, hrs2=hrs2, age3=age3, hrs3=hrs3)
f1 = fit_model(b1, data, "m1", 35001)
println("MC_M1_DONE"); flush(stdout)
f2 = fit_model(b2, data, "m2", 35002)
println("MC_M2_DONE"); flush(stdout)
f3 = fit_model(b3, data, "m3", 35003)
println("MC_M3_DONE"); flush(stdout)
loo = Dict("m1" => psis_loo(f1["ll"]), "m2" => psis_loo(f2["ll"]), "m3" => psis_loo(f3["ll"]))
for (k, v) in loo
    println("LOO $k elpd=", round(v["elpd"]; digits=1), " se=", round(v["se"]; digits=1),
        " p_loo=", round(v["p_loo"]; digits=1), " kmax=", round(v["kmax"]; digits=2),
        " kbad=", v["kbad"])
end
for (f, t) in ((f1, "m1"), (f2, "m2"), (f3, "m3"))
    g = copy(f); delete!(g, "ll")  # 800x31k too big to keep; LOO already reduced
    open(joinpath(OUT, "$t.json"), "w") do io
        JSON.print(io, g)
    end
end
open(joinpath(OUT, "loo.json"), "w") do io
    loo2 = Dict(k => Dict(j => v for (j, v) in vv if j != "pointwise") for (k, vv) in loo)
    JSON.print(io, loo2)
end
# elpd differences with SE of the difference (paired)
for (a, b, la, lb) in (("m3", "m2", "m3-m2", "m3-m2"), ("m2", "m1", "m2-m1", "m2-m1"))
    d = loo[a]["pointwise"] .- loo[b]["pointwise"]
    println("ELPD_DIFF $la=", round(sum(d); digits=1), " se=", round(sqrt(length(d) * var(d)); digits=1))
end
println("MC_LOO_DONE"); flush(stdout)
# NOTE: model-2 probability curves over age x sex x race are built in Python
# from m2.json draws + mc_scale.json (no Stan replay needed for fixed effects).
