# Bambi replication: https://bambinos.github.io/bambi/notebooks/kulprit.html
# Run: julia --project=. kulprit.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: none (simulated). Outputs: .out/results/kulprit/ + .out/stan/ (gitignored).
# The notebook is a STUB (no model/data/fit): for few variables it recommends
# PSIS-LOO-CV, for many variables projective predictive inference via Kulprit.
# BRM counterpart: the small-p half is replicated (PSIS-LOO over nested models
# + full-fit coefficient ranking on simulated sparse truth); projective
# predictive inference has NO BRM counterpart (grep-clean in src/) and is
# recorded as a capability gap in the brief, not worked around here.
using Random, Statistics, Distributions, JSON, LogExpFunctions
using PSIS
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "kulprit")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

# sparse truth: 8 candidate predictors, 3 signals
rng = Xoshiro(7)
n, p = 200, 8
beta_true = [1.5, -2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
X = randn(rng, n, p)
y = vec(X * beta_true) + randn(rng, n)
data = merge((; y=y), NamedTuple(Symbol("x$i") => X[:, i] for i in 1:p))

function sample_fit(builder, dat, tag, seed; draws=800)
    sb = SBBRMI(builder(dat); mod=@__MODULE__)
    problem = BRM.stan_instantiate(sb;
        path=joinpath(SCRATCH, ".out", "stan", "kulp_$tag.stan"))
    rp = try
        BRM.adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.0)
    catch
        problem
    end
    fit = adaptive_warmup_mcmc(Xoshiro(seed), rp; init=missing, n_draws=draws,
        nonlinear_adapt=false, monitor_ess=false)
    P = fit.posterior_position
    names = BridgeStan.param_names(problem.model; include_tp=true)
    C = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[:, i]);
        include_tp=true) for i in axes(P, 2)]...))
    open(joinpath(OUT, "$tag.json"), "w") do io
        JSON.print(io, Dict("names" => string.(names),
            "draws" => [C[:, i] for i in eachindex(names)],
            "divergences" => fit.n_divergent_samples))
    end
    println("FIT_DONE tag=$tag draws=", size(C))
    return sb, problem
end

function psis_loo(ll)
    ll3 = reshape(ll, size(ll, 1), 1, size(ll, 2))
    res = psis(-ll3; normalize=true, warn=false)
    lw = res.log_weights
    pointwise = vec(dropdims(logsumexp(ll3 .+ lw; dims=(1, 2)); dims=(1, 2)))
    lpd = vec(dropdims(logsumexp(ll3; dims=(1, 2)) .- log(size(ll, 1)); dims=(1, 2)))
    elpd = sum(pointwise)
    Dict("elpd" => elpd, "se" => sqrt(length(pointwise) * var(pointwise)),
        "p_loo" => sum(lpd) - elpd, "kmax" => maximum(res.pareto_shape),
        "pointwise" => pointwise)
end

function loo_of(builder, dat, tag, seed)
    f = JSON.parsefile(joinpath(OUT, "$tag.json"))
    sb = SBBRMI(builder(dat); mod=@__MODULE__)
    problem = BRM.stan_instantiate(sb;
        path=joinpath(SCRATCH, ".out", "stan", "kulp_$tag.stan"))
    Cc = hcat([Float64.(f["draws"][i]) for i in eachindex(f["names"])]...)
    P = permutedims(hcat([BridgeStan.param_unconstrain(problem.model, collect(Cc[i, :]))
        for i in axes(Cc, 1)]...))
    gq_names = BridgeStan.param_names(problem.model; include_gq=true)
    srng = BridgeStan.StanRNG(problem.model, seed)
    G = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[i, :]);
        include_gq=true, rng=srng) for i in axes(P, 1)]...))
    d = BRM.brm_descriptor(builder, dat; mod=@__MODULE__, name=Symbol("kulp_$tag"))
    LL = BRM.brm_output_draws(d, G, string.(gq_names); logical=:y, role=:pointwise_loglik)
    psis_loo(Matrix(LL))
end

b0 = @brm begin
    mu ~ 1
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
btrue = @brm begin
    mu ~ 1 + x1 + x2 + x5
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
bfull = @brm begin
    mu ~ 1 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end

sample_fit(b0, data, "m0", 43001)
sample_fit(btrue, data, "mtrue", 43002)
sbfull, _ = sample_fit(bfull, data, "mfull", 43003)

loos = Dict(t => loo_of(b, data, t, s) for (b, t, s) in
    ((b0, "m0", 43001), (btrue, "mtrue", 43002), (bfull, "mfull", 43003)))
for (t, l) in loos
    println("LOO tag=$t elpd=", round(l["elpd"]; digits=1),
        " se=", round(l["se"]; digits=1), " p_loo=", round(l["p_loo"]; digits=1),
        " kmax=", round(l["kmax"]; digits=2))
end
@assert loos["mtrue"]["elpd"] > loos["m0"]["elpd"] + 20
@assert loos["mfull"]["elpd"] > loos["m0"]["elpd"] + 20
@assert abs(loos["mfull"]["elpd"] - loos["mtrue"]["elpd"]) < 25

# full-fit coefficient ranking: |mean|/sd per slope
f = JSON.parsefile(joinpath(OUT, "mfull.json"))
C = hcat([Float64.(f["draws"][i]) for i in eachindex(f["names"])]...)
nms = string.(f["names"])
pnames = string.(BRM.popcoefnames(sbfull.parent, :mu))
@assert length(pnames) == 9
col(j) = C[:, findfirst(==("pop_mu_beta_pop.$j"), nms)]
rank = sort([(pnames[j], abs(mean(col(j))) / std(col(j)))
    for j in 2:9]; by=last, rev=true)
println("RANK=", [(r[1], round(r[2]; digits=2)) for r in rank])
top4 = Set(first.(rank[1:4]))
@assert issubset(Set(["x1", "x2", "x5"]), top4) "ranking missed a signal: $rank"

open(joinpath(OUT, "kulp_loo.json"), "w") do io
    JSON.print(io, Dict("loo" => Dict(t => Dict("elpd" => l["elpd"], "se" => l["se"],
        "p_loo" => l["p_loo"], "kmax" => l["kmax"]) for (t, l) in loos),
        "rank" => [Dict("var" => r[1], "z" => r[2]) for r in rank],
        "beta_true" => beta_true))
end
println("KULPRIT_DONE")
flush(stdout)
