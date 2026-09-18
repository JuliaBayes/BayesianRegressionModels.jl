# Bambi replication: https://bambinos.github.io/bambi/notebooks/per_parameter_noncentered.html
# Run: julia --project=. distsleep.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Bambi per_parameter_noncentered replication with BRM (distributional sleepstudy).
# Bambi: Reaction ~ 1 + (1|Subject), sigma ~ 1 + (1|Subject), with mu
# noncentered + sigma centered (m_mixed), plus pure noncentered/centered.
# BRM: centered_groups is whole-group, so the mixed case maps to the adaptive
# wrapper (per-effect continuous adaptation); static cases map directly.
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "distsleep")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

df = CSV.read(joinpath(SCRATCH, "data", "sleepstudy.csv"), DataFrame)
codes(v) = (u = sort(unique(v)); m = Dict(x => i for (i, x) in enumerate(u)); [m[x] for x in v])
data = (;
    Reaction=Float64.(df.Reaction) ./ 100,
    Subject=codes(df.Subject),
)

builder = @brm begin
    mu ~ 1 + (1 | Subject)
    log_sigma ~ 1 + (1 | Subject)
    Reaction ~ Normal(mu, exp(log_sigma))
end

# Lowering happens in the caller (top level); see snag brm-totals-lower-a47c6227.
function sample_fit(sb, problem, seed)
    rp = BRM.adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.0)
    fit = adaptive_warmup_mcmc(
        Xoshiro(seed), rp; init=missing, n_draws=600,
        nonlinear_adapt=true, monitor_ess=false,
    )
    P = fit.posterior_position
    con_names = BridgeStan.param_names(problem.model)
    C = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[:, i])) for i in axes(P, 2)]...))
    keep = findall(n -> (s = string(n); occursin("beta_pop", s) || occursin("tau", s) || occursin("log_scale", s) || s == "sigma"), con_names)
    rows = map(keep) do i
        v = C[:, i]
        Dict("name" => string(con_names[i]), "mean" => mean(v), "sd" => std(v),
            "q05" => quantile(v, 0.05), "q50" => quantile(v, 0.5), "q95" => quantile(v, 0.95))
    end
    Dict("n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples, "params" => rows)
end

for (tag, cg, seed) in (("nc", [], 991), ("c", [:Subject], 992))
    sb = SBBRMI(builder(data); mod=@__MODULE__, total_groups=(), centered_groups=cg)
    problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "distsleep_$tag.stan"))
    open(joinpath(OUT, "$tag.json"), "w") do io
        JSON.print(io, sample_fit(sb, problem, seed))
    end
    println("DISTSLEEP_$(uppercase(tag))_DONE")
    flush(stdout)
end
