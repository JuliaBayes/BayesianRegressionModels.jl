# Bambi replication: https://bambinos.github.io/bambi/notebooks/hierarchical_binomial_bambi.html
# Run: julia --project=. hierbinomial.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# bambi hierarchical_binomial_bambi: p(H, AB) ~ 0 + playerID / 1 + (1|playerID).
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme
using LogExpFunctions: logit

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "hierbin")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

function sample_fit(sb, problem, seed; draws=800)
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
    Dict("n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples, "params" => rows,
        "names" => string.(con_names), "draws" => [C[:, i] for i in eachindex(con_names)])
end

bdf = CSV.read(joinpath(SCRATCH, "data", "Batting.csv"), DataFrame)
dropmissing!(bdf, [:H, :AB, :playerID, :yearID])
bdf = bdf[bdf.AB .> 0, :]
bdf = bdf[bdf.yearID .>= 2016, :]
bdf = first(bdf, 15)
Hv = Int.(bdf.H); ABv = Int.(bdf.AB)
players = string.(bdf.playerID)
uniq = unique(players)
D = hcat([Float64.(players .== u) for u in uniq]...)
pc = [findfirst(==(u), uniq) for u in players]
open(joinpath(OUT, "players.json"), "w") do io
    JSON.print(io, Dict("players" => players, "H" => Hv, "AB" => ABv))
end

# non-hierarchical cell means (15 float dummies)
cols = Dict(Symbol("d$k") => D[:, k] for k in 1:15)
b_nh = @brm begin
    logit(p) ~ 0 + d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9 + d10 + d11 + d12 + d13 + d14 + d15
    H ~ Binomial(ABv, p)
end
sb = SBBRMI(b_nh((; H=Hv, ABv=ABv, cols...)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "hb_nonhier.stan"))
f = sample_fit(sb, problem, 6001)
open(joinpath(OUT, "hb_nonhier.json"), "w") do io
    JSON.print(io, f)
end
println("HB_NONHIER_DONE")
flush(stdout)

# hierarchical varying intercept (bambi priors: Intercept N(0,1); group sd default)
b_h = @brm begin
    logit(p) ~ 1 + (1 | player)
    effect(p, Intercept) ~ Normal(0, 1)
    H ~ Binomial(ABv, p)
end
sb = SBBRMI(b_h((; H=Hv, ABv=ABv, player=pc)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "hb_hier.stan"))
f = sample_fit(sb, problem, 6002)
open(joinpath(OUT, "hb_hier.json"), "w") do io
    JSON.print(io, f)
end
println("HB_HIER_DONE")
flush(stdout)
