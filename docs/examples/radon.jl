# Bambi replication: https://bambinos.github.io/bambi/notebooks/radon_example.html
# Run: julia --project=. radon.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Bambi radon_example replication with BRM (partial pooling ladder).
# Bambi models: log_radon ~ 1 + (1|county); ~ 0 + floor + (1|county);
# ~ 0 + floor + (0+floor|county); + log_u county uranium covariate.
# Data: pymc radon.csv (919 rows, 85 counties).
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "radon")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

df = CSV.read(joinpath(SCRATCH, "data", "radon.csv"), DataFrame)
codes(v) = (u = sort(unique(v)); m = Dict(x => i for (i, x) in enumerate(u)); [m[x] for x in v])
# Notebook maps floor 0/1 -> Basement/Floor strings; bambi auto-categorizes.
# BRM cell-means-codes bare integer columns under `0 +`, so codes 1/2 give the
# same cell-means model (Basement=1, Floor=2).
data = (;
    log_radon=Float64.(df.log_radon), floor=[x == 0 ? 1 : 2 for x in df.floor],
    county=codes(df.county_code), log_u=log.(Float64.(df.Uppm)),
)

# NOTE (snag brm-totals-lower-a47c6227): totals-path SBBRMI lowering fails when
# the builder is FIRST invoked inside a function. Lower + instantiate every
# model here at top level; sample() only samples the prebuilt problem.
function sample(sb, problem, seed; draws=800)
    rp = BRM.adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.0)
    fit = adaptive_warmup_mcmc(
        Xoshiro(seed), rp; init=zeros(LogDensityProblems.dimension(rp)),
        n_draws=draws, nonlinear_adapt=true, monitor_ess=false,
    )
    P = fit.posterior_position
    con_names = BridgeStan.param_names(problem.model; include_tp=true)
    C = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[:, i]); include_tp=true) for i in axes(P, 2)]...))
    pops = eachindex(con_names)
    rows = map(pops) do i
        v = C[:, i]
        Dict("name" => string(con_names[i]), "mean" => mean(v), "sd" => std(v),
            "q05" => quantile(v, 0.05), "q50" => quantile(v, 0.5), "q95" => quantile(v, 0.95))
    end
    Dict("n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples, "params" => rows)
end

b_partial = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + (1 | county)
    log_radon ~ Normal(mu, sigma)
end
b_varyint = @brm begin
    sigma ~ Exponential(1)
    mu ~ 0 + floor + (1 | county)
    log_radon ~ Normal(mu, sigma)
end
b_varyslope = @brm begin
    sigma ~ Exponential(1)
    mu ~ 0 + floor + (0 + floor | county)
    log_radon ~ Normal(mu, sigma)
end

for (tag, b, seed) in (("partial", b_partial, 2251), ("varyint", b_varyint, 2252), ("varyslope", b_varyslope, 2253))
    if isfile(joinpath(OUT, "$tag.json"))
        println("RADON_$(uppercase(tag))_CACHED")
        continue
    end
    sb = SBBRMI(b(data); mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "radon_$tag.stan"))
    open(joinpath(OUT, "$tag.json"), "w") do io
        JSON.print(io, sample(sb, problem, seed))
    end
    println("RADON_$(uppercase(tag))_DONE")
    flush(stdout)
end
