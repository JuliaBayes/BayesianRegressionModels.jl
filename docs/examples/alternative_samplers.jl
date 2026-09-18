# Bambi replication: https://bambinos.github.io/bambi/notebooks/alternative_samplers.html
# Run: julia --project=. alternative_samplers.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: none (simulated, notebook generative process). Outputs: .out/results/altsamp/ + .out/stan/ (gitignored).
# Notebook fits y ~ x (n=100 simulated) under three NUTS backends (blackjax,
# numpyro, nutpie). BRM counterpart: the same model under three WarmupHMC
# drivers (adaptive single-chain, cooperative + clustered multi-chain) on ONE
# shared Stan program, comparing posteriors and wall time. BRM has no
# non-MCMC inference (no VI/Pathfinder standalone), so the backends compared
# are all adaptive-warmup NUTS drivers.
using Random, Statistics, Distributions, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "altsamp")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

# notebook generative process (seed 42; RNG differs from numpy by design)
rng = Xoshiro(42)
n, beta_true, sigma_true = 100, randn(rng, 1), 1.0
x = randn(rng, n)
y = vec(x .* beta_true') + sigma_true * randn(rng, n)
println("BETA_TRUE=", beta_true[1])
data = (; y=y, x=x)

b = @brm begin
    mu ~ 1 + x
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
sb = SBBRMI(b(data); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model;
    path=joinpath(SCRATCH, ".out", "stan", "altsamp.stan"))
rp = try
    BRM.adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.0)
catch
    problem
end

function constrain_all(P)
    names = BridgeStan.param_names(problem.model; include_tp=true)
    C = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[:, i]);
        include_tp=true) for i in axes(P, 2)]...))
    return C, string.(names)
end
function summarize(C, names, tag)
    bu = BRM.popcoefnames(sb.parent, :mu)
    slope = C[:, findfirst(==("pop_mu_beta_pop.2"), names)]
    inter = C[:, findfirst(==("pop_mu_beta_pop.1"), names)]
    sig = C[:, findfirst(==("sigma"), names)]
    @assert length(bu) == 2
    Dict("tag" => tag, "n_draws" => size(C, 1),
        "slope" => Dict("mean" => mean(slope), "sd" => std(slope)),
        "intercept" => Dict("mean" => mean(inter), "sd" => std(inter)),
        "sigma" => Dict("mean" => mean(sig), "sd" => std(sig)),
        "slope_draws" => slope)
end

results = Dict{String,Any}()

# driver 1: adaptive single chain (the default, like the notebook's PyMC NUTS)
t = time()
f1 = adaptive_warmup_mcmc(Xoshiro(42001), rp; init=missing, n_draws=800,
    nonlinear_adapt=false, monitor_ess=false)
t1 = time() - t
C1, n1 = constrain_all(f1.posterior_position)
results["adaptive"] = merge(summarize(C1, n1, "adaptive"),
    Dict("wall_s" => t1, "divergences" => f1.n_divergent_samples))
println("ADAPTIVE_DONE draws=", size(C1), " wall_s=", round(t1; digits=1))

# driver 2: cooperative multi-chain (pooled-ESS stop gate)
t = time()
f2 = cooperative_warmup_mcmc([Xoshiro(42002 + i) for i in 1:4], rp;
    n_draws=400, target_ess=800, nonlinear_adapt=false)
t2 = time() - t
P2 = hcat([r.posterior_position for r in f2.results
    if size(r.posterior_position, 2) > 0]...)
C2, n2 = constrain_all(P2)
results["cooperative"] = merge(summarize(C2, n2, "cooperative"),
    Dict("wall_s" => t2, "n_used" => f2.n_used,
        "divergences" => sum(r.n_divergent_samples for r in f2.results)))
println("COOP_DONE draws=", size(C2), " wall_s=", round(t2; digits=1))

# driver 3: clustered multi-chain
t = time()
f3 = clustered_warmup_mcmc([Xoshiro(42010 + i) for i in 1:4], rp;
    n_draws=200)
t3 = time() - t
P3 = hcat([r.posterior_position for r in f3.results
    if size(r.posterior_position, 2) > 0]...)
C3, n3 = constrain_all(P3)
results["clustered"] = merge(summarize(C3, n3, "clustered"),
    Dict("wall_s" => t3,
        "divergences" => sum(r.n_divergent_samples for r in f3.results)))
println("CLUSTER_DONE draws=", size(C3), " wall_s=", round(t3; digits=1))

# all three drivers must agree on the slope (the notebook's cross-backend point)
m = [results[k]["slope"]["mean"] for k in ("adaptive", "cooperative", "clustered")]
@assert maximum(m) - minimum(m) < 0.15 "driver disagreement: $m"
open(joinpath(OUT, "altsamp.json"), "w") do io
    JSON.print(io, merge(results, Dict("beta_true" => beta_true[1])))
end
println("ALTSAMP_DONE")
flush(stdout)
