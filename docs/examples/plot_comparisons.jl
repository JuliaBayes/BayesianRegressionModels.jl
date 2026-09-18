# Bambi replication: https://bambinos.github.io/bambi/notebooks/plot_comparisons.html
# Run: julia --project=. plot_comparisons.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/plotcomp/ + .out/stan/ (gitignored).
# UCLA fish zero-inflated Poisson: group comparisons over prediction grids via
# the native engine (brm_prediction_grid/brm_conditional_draws/brm_contrast_draws,
# snag replicate-bambi-e064da10): persons 4-vs-1 diffs over a child x livebait
# grid, and the livebait 1-vs-0 contrast over a persons x child grid.
# ZIP response means are unmapped in engine v1 (fail-closed), so contrasts run
# on the sanctioned response-scale path: target=:predictive draws.
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "plotcomp")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

df = CSV.read(joinpath(SCRATCH, "data", "fish.csv"), DataFrame)
livebait = Int.(df.livebait); camper = Int.(df.camper)
persons = Int.(df.persons); child = Int.(df.child); count = Int.(df.count)
data = (; count=count, livebait=livebait, camper=camper, persons=persons, child=child)

b = @brm begin
    log(lambda) ~ 1 + livebait + camper + persons + child
    effect(lambda, Intercept) ~ Normal(0, 5)
    effect(lambda, livebait) ~ Normal(0, 2.5)
    effect(lambda, camper) ~ Normal(0, 2.5)
    effect(lambda, persons) ~ Normal(0, 2.5)
    effect(lambda, child) ~ Normal(0, 2.5)
    zi0 ~ Beta(2, 2)
    count ~ ZeroInflatedPoisson(lambda, zi0)
end
sb = SBBRMI(b(data); mod=@__MODULE__)
problem = BRM.stan_instantiate(sb;
    path=joinpath(SCRATCH, ".out", "stan", "plotcomp.stan"))
rp = try
    BRM.adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.0)
catch
    problem
end
fit = adaptive_warmup_mcmc(Xoshiro(44001), rp; init=missing, n_draws=800,
    nonlinear_adapt=false, monitor_ess=false)
P = fit.posterior_position
names = BridgeStan.param_names(problem.model; include_tp=true)
C = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[:, i]);
    include_tp=true) for i in axes(P, 2)]...))
println("FIT_DONE draws=", size(C), " div=", fit.n_divergent_samples)

# round-trip to unconstrained draws (rows) for the engine
U = permutedims(hcat([BridgeStan.param_unconstrain(problem.model, collect(C[i, :]))
    for i in axes(C, 1)]...))
d = BRM.brm_descriptor(b, data; mod=@__MODULE__, name=:plotcomp)

# --- contrast 1: persons [1,4] over child x livebait (notebook cell 9/12) ---
g1 = BRM.brm_prediction_grid(d;
    focal=[:persons => [1, 4], :child => [0, 1, 2], :livebait => [0, 1]])
@assert length(g1.persons) == 12
c1 = BRM.brm_conditional_draws(d, U, g1; focal=[:persons, :child, :livebait],
    target=:predictive, seed=21)
@assert minimum(c1.draws) >= 0.0
k1 = BRM.brm_contrast_draws(c1; by=:persons, pairs=[(4, 1)], how=:diff)
s1 = BRM.brm_summarize_draws(k1.draws; probs=[0.94])
@assert all(r -> r.mean > 0, s1) "persons 4-vs-1 must be positive: $(getproperty.(s1, :mean))"
println("CONTRAST1_DONE labels=", k1.labels, " cells=", length(s1))

# --- contrast 2: livebait 1-vs-0 over persons x child (notebook cell 18/20) ---
g2 = BRM.brm_prediction_grid(d; focal=[:livebait, :persons, :child], n=10)
c2 = BRM.brm_conditional_draws(d, U, g2; focal=[:livebait, :persons, :child],
    target=:predictive, seed=22)
k2 = BRM.brm_contrast_draws(c2; by=:livebait, pairs=:reference, how=:diff)
s2 = BRM.brm_summarize_draws(k2.draws; probs=[0.94])
@assert all(r -> r.mean > 0, s2) "livebait 1-vs-0 must be positive"
println("CONTRAST2_DONE labels=", k2.labels, " cells=", length(s2))

function rows(s, sub)
    [Dict("element" => r.element, "mean" => r.mean, "sd" => r.sd,
        "lo" => r.lower_1, "hi" => r.upper_1) for r in s]
end
open(joinpath(OUT, "plotcomp.json"), "w") do io
    JSON.print(io, Dict(
        "persons_4v1" => Dict("labels" => k1.labels, "rows" => rows(s1, k1.subgrid),
            "child" => collect(k1.subgrid.child), "livebait" => collect(k1.subgrid.livebait)),
        "livebait_1v0" => Dict("labels" => k2.labels, "rows" => rows(s2, k2.subgrid),
            "persons" => collect(k2.subgrid.persons), "child" => collect(k2.subgrid.child)),
        "divergences" => fit.n_divergent_samples))
end
println("PLOTCOMP_DONE")
flush(stdout)
