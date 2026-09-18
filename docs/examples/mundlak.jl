# Bambi replication: https://bambinos.github.io/bambi/notebooks/fixed_random.html
# Run: julia --project=. mundlak.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Bambi fixed_random replication with BRM (Mundlak ladder, simulated CRE logit).
# Design mirrors the notebook: 30 groups, 2000 obs, a0=-2, bxy=0, bzy=1,
# ug ~ N(0, 1.5^2)/group, x|ug ~ N(ug[g], 1), z ~ N(0,1)/group,
# y ~ Bernoulli(logit^-1(-2 + ug[g] + z[g])). RNG differs (Xoshiro vs numpy).
using Random, Statistics, Distributions, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "mundlak")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

rng = Xoshiro(42)
G, N = 30, 2000
g = rand(rng, 1:G, N)
ug = randn(rng, G) .* 1.5
z = randn(rng, G)
x = [randn(rng) + ug[g[i]] for i in 1:N]
p = @. 1 / (1 + exp(2 - ug[g] - z[g]))  # a0=-2, bxy=0, bzy=1
y = [rand(rng, Bernoulli(pi_)) ? 1 : 0 for pi_ in p]  # Int 0/1: SBBRMI Bernoulli needs int (snag brm-sbbrmi-berno-18aeccfe)
xg = zeros(N)
for gg in 1:G
    xg[g .== gg] .= sum(x[g .== gg]) / sum(g .== gg)
end
data = (; y=y, x=Float64.(x), z=Float64.(z[g]), xbar=xg, group=g)
println("rate=", sum(y) / N)

builders = Dict(
    "naive" => @brm(begin
        mu ~ 1 + x + z
        y ~ BernoulliLogit(mu)
    end),
    "fe" => @brm(begin
        mu ~ 0 + group + x + z
        y ~ BernoulliLogit(mu)
    end),
    "multilevel" => @brm(begin
        mu ~ 1 + x + z + (1 | group)
        y ~ BernoulliLogit(mu)
    end),
    "mundlak" => @brm(begin
        mu ~ 1 + x + z + xbar + (1 | group)
        y ~ BernoulliLogit(mu)
    end),
)

function sample_fit(sb, problem, seed; draws=600)
    # Models without ordinary ranef blocks (naive, fe) sample the bare problem.
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
    keep = findall(n -> (s = string(n); occursin("beta_pop", s) || occursin("tau", s) || occursin("_L.", s)), con_names)
    rows = map(keep) do i
        v = C[:, i]
        Dict("name" => string(con_names[i]), "mean" => mean(v), "sd" => std(v),
            "q05" => quantile(v, 0.05), "q50" => quantile(v, 0.5), "q95" => quantile(v, 0.95))
    end
    Dict("n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples, "params" => rows)
end

# Lower + instantiate at top level (snag brm-totals-lower-a47c6227).
for (i, (tag, b)) in enumerate(sort!(collect(builders); by=first))
    out = joinpath(OUT, "$tag.json")
    if isfile(out)
        println("MUNDLAK_$(uppercase(tag))_CACHED")
        continue
    end
    sb = SBBRMI(b(data); mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "mundlak_$tag.stan"))
    open(out, "w") do io
        JSON.print(io, sample_fit(sb, problem, 4200 + i))
    end
    println("MUNDLAK_$(uppercase(tag))_DONE")
    flush(stdout)
end
