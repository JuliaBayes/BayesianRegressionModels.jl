# Bambi replication: https://bambinos.github.io/bambi/notebooks/model_comparison.html
# Run: julia --project=. modelcomp_loo.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# PSIS-LOO for the saved model_comparison fits (no refit): rebuild descriptors,
# recover unconstrained draws by round-trip, read pointwise loglik, smooth with
# PSIS.jl in (draws, chains=1, params) orientation.
using Random, Statistics, Distributions, CSV, DataFrames, JSON, LogExpFunctions
using PSIS
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "modelcomp")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

df = CSV.read(joinpath(SCRATCH, "data", "adults.csv"), DataFrame)
keep = (string.(df.race) .== "Black") .| (string.(df.race) .== "White")
df = df[keep, :]
y = Int.(string.(df.income) .== ">50K")
male = Float64.(string.(df.sex) .== "Male")
white = Float64.(string.(df.race) .== "White")
zsc(v) = (m = sum(v) / length(v); s = sqrt(sum((v .- m) .^ 2) / length(v)); ((v .- m) ./ s, m, s))
agez, _, _ = zsc(Float64.(df.age)); hrsz, _, _ = zsc(Float64.(df.hs_week))
age2 = agez .^ 2; hrs2 = hrsz .^ 2; age3 = agez .^ 3; hrs3 = hrsz .^ 3
data = (; income=y, male=male, white=white, agez=agez, hrsz=hrsz,
    age2=age2, hrs2=hrs2, age3=age3, hrs3=hrs3)

b1 = @brm begin
    eta ~ 1 + male + white + agez + hrsz
    income ~ BernoulliLogit(eta)
end
b2 = @brm begin
    eta ~ 1 + male + white + agez + age2 + hrsz + hrs2
    income ~ BernoulliLogit(eta)
end
b3 = @brm begin
    eta ~ 1 + male + white + agez + age2 + age3 + hrsz + hrs2 + hrs3
    income ~ BernoulliLogit(eta)
end

function psis_loo(ll)
    # ll: draws x obs -> (draws, 1 chain, obs params)
    ll3 = reshape(ll, size(ll, 1), 1, size(ll, 2))
    res = psis(-ll3; normalize=true, warn=false)
    lw = res.log_weights
    pointwise = vec(dropdims(logsumexp(ll3 .+ lw; dims=(1, 2)); dims=(1, 2)))
    lpd = vec(dropdims(logsumexp(ll3; dims=(1, 2)) .- log(size(ll, 1)); dims=(1, 2)))
    elpd = sum(pointwise)
    se = sqrt(length(pointwise) * var(pointwise))
    Dict("elpd" => elpd, "se" => se, "p_loo" => sum(lpd) - elpd,
        "kmax" => maximum(res.pareto_shape), "kbad" => sum(res.pareto_shape .> 0.7),
        "pointwise" => pointwise)
end

loos = Dict{String,Any}()
for (b, t, seed) in ((b1, "m1", 35001), (b2, "m2", 35002), (b3, "m3", 35003))
    sb = SBBRMI(b(data); mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model;
        path=joinpath(SCRATCH, ".out", "stan", "mc_$t.stan"))
    f = JSON.parsefile(joinpath(OUT, "$t.json"))
    Cc = hcat([Float64.(f["draws"][i]) for i in eachindex(f["names"])]...)
    P = permutedims(hcat([BridgeStan.param_unconstrain(problem.model, collect(Cc[i, :]))
        for i in axes(Cc, 1)]...))
    gq_names = BridgeStan.param_names(problem.model; include_gq=true)
    srng = BridgeStan.StanRNG(problem.model, seed)
    G = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[i, :]); include_gq=true, rng=srng) for i in axes(P, 1)]...))
    d = BRM.brm_descriptor(b, data; mod=@__MODULE__, name=Symbol("mc_$t"))
    LL = BRM.brm_output_draws(d, G, string.(gq_names); logical=:income, role=:pointwise_loglik)
    loos[t] = psis_loo(Matrix(LL))
    v = loos[t]
    println("LOO $t elpd=", round(v["elpd"]; digits=1), " se=", round(v["se"]; digits=1),
        " p_loo=", round(v["p_loo"]; digits=1), " kmax=", round(v["kmax"]; digits=2),
        " kbad=", v["kbad"])
end
for (a, b) in (("m3", "m2"), ("m2", "m1"))
    dd = loos[a]["pointwise"] .- loos[b]["pointwise"]
    println("ELPD_DIFF $a-$b=", round(sum(dd); digits=1), " se=",
        round(sqrt(length(dd) * var(dd)); digits=1))
end
open(joinpath(OUT, "loo.json"), "w") do io
    loo2 = Dict(k => Dict(j => v for (j, v) in vv if j != "pointwise") for (k, vv) in loos)
    JSON.print(io, loo2)
end
println("MC_LOO_FIXED_DONE")
flush(stdout)
