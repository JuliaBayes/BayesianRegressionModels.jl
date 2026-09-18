# Bambi replication: https://bambinos.github.io/bambi/notebooks/plot_slopes.html
# Run: julia --project=. plot_slopes.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/plotslopes/ + .out/stan/ (gitignored).
# Wells arsenic-switching Bernoulli: marginal slopes wrt arsenic over
# dist100 x educ4 grids via the native engine (brm_slope_draws, snag
# replicate-bambi-e064da10), plain and interacted specifications.
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "plotslopes")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

df = CSV.read(joinpath(SCRATCH, "data", "Wells.csv"), DataFrame)
switch = Int.(string.(df.switch) .== "yes")
dist100 = Float64.(df.distance) ./ 100
educ4 = Float64.(df.education) ./ 4
arsenic = Float64.(df.arsenic)
data = (; switch=switch, dist100=dist100, educ4=educ4, arsenic=arsenic)

function sample_engine(builder, dat, tag, seed; draws=800)
    sb = SBBRMI(builder(dat); mod=@__MODULE__)
    problem = BRM.stan_instantiate(sb;
        path=joinpath(SCRATCH, ".out", "stan", "plotsl_$tag.stan"))
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
    U = permutedims(hcat([BridgeStan.param_unconstrain(problem.model, collect(C[i, :]))
        for i in axes(C, 1)]...))
    d = BRM.brm_descriptor(builder, dat; mod=@__MODULE__, name=Symbol("plotsl_$tag"))
    println("FIT_DONE tag=$tag draws=", size(C), " div=", fit.n_divergent_samples)
    return d, U, fit.n_divergent_samples
end

b1 = @brm begin
    eta ~ 1 + dist100 + arsenic + educ4
    effect(eta, Intercept) ~ Normal(0, 5)
    effect(eta, dist100) ~ Normal(0, 2.5)
    effect(eta, arsenic) ~ Normal(0, 2.5)
    effect(eta, educ4) ~ Normal(0, 2.5)
    switch ~ BernoulliLogit(eta)
end
d1, U1, div1 = sample_engine(b1, data, "plain", 44011)

# slopes wrt arsenic at 1.3 over the notebook's 3x3 grid (cell 10)
g1 = BRM.brm_prediction_grid(d1; focal=[:arsenic => [1.3],
    :dist100 => [0.20, 0.50, 0.80], :educ4 => [1.00, 1.20, 2.00]])
s1 = BRM.brm_slope_draws(d1, U1, g1; wrt=:arsenic, focal=[:dist100, :educ4])
t1 = BRM.brm_summarize_draws(s1.draws; probs=[0.94])
@assert all(isfinite.(s1.draws))
@assert all(r -> r.mean > 0, t1) "arsenic slopes must be positive: $(getproperty.(t1, :mean))"
println("SLOPES1_DONE cells=", length(t1),
    " mean=", round(mean(getproperty.(t1, :mean)); digits=4))

b2 = @brm begin
    eta ~ 1 + dist100 + arsenic + educ4 + dist100 & educ4 + arsenic & educ4
    effect(eta, Intercept) ~ Normal(0, 5)
    effect(eta, dist100) ~ Normal(0, 2.5)
    effect(eta, arsenic) ~ Normal(0, 2.5)
    effect(eta, educ4) ~ Normal(0, 2.5)
    effect(eta, dist100 & educ4) ~ Normal(0, 2.5)
    effect(eta, arsenic & educ4) ~ Normal(0, 2.5)
    switch ~ BernoulliLogit(eta)
end
d2, U2, div2 = sample_engine(b2, data, "interact", 44012)

# slopes over dist100 linspace x educ4 levels at median arsenic (cell 19)
amed = Statistics.median(arsenic)
g2 = BRM.brm_prediction_grid(d2; focal=[:arsenic => [amed],
    :dist100 => collect(range(0, 4; length=25)), :educ4 => [0.0, 1.0, 2.0, 3.0, 4.0]])
s2 = BRM.brm_slope_draws(d2, U2, g2; wrt=:arsenic, focal=[:dist100, :educ4])
t2 = BRM.brm_summarize_draws(s2.draws; probs=[0.94])
m2 = getproperty.(t2, :mean)
@assert all(isfinite.(s2.draws))
@assert maximum(m2) - minimum(m2) > 0.01 "interaction must move slopes across educ4"
println("SLOPES2_DONE cells=", length(t2))

open(joinpath(OUT, "plotslopes.json"), "w") do io
    JSON.print(io, Dict(
        "plain" => [Dict("dist100" => g1.dist100[i], "educ4" => g1.educ4[i],
            "mean" => t1[i].mean, "sd" => t1[i].sd,
            "lo" => t1[i].lower_1, "hi" => t1[i].upper_1)
            for i in eachindex(t1)],
        "interact" => [Dict("dist100" => g2.dist100[i], "educ4" => g2.educ4[i],
            "mean" => t2[i].mean, "sd" => t2[i].sd,
            "lo" => t2[i].lower_1, "hi" => t2[i].upper_1)
            for i in eachindex(t2)],
        "divergences" => [div1, div2]))
end
println("PLOTSLOPES_DONE")
flush(stdout)
