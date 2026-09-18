# Bambi replication: https://bambinos.github.io/bambi/notebooks/survival_continuous_time_notebook.html
# Run: julia --project=. surv_cont.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# bambi survival_continuous_time_notebook: Weibull AFT sim (truth recovery) +
# retention applied example (Weibull / Exponential / frailty).
# Right-censoring via the sanctioned clamp surface: censored(Weibull/Exponential;
# upper=u) with a per-row upper column (Inf = observed -> density, t = censored
# -> CCDF). DGP: notebook structure, own seed.
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "survcont")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

function sample_fit(sb, problem, seed; draws=800)
    rp = try
        BRM.adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.0)
    catch
        problem
    end
    fit = adaptive_warmup_mcmc(
        Xoshiro(seed), rp; init=0.2, n_draws=draws,
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

# ================= Weibull AFT sim =================
# (skip when already fitted: restart-safe)
if isfile(joinpath(OUT, "sc_sim.json"))
    println("SC_SIM_CACHED")
else
rng = Xoshiro(1234)
n = 1000
treatment = Float64.(rand(rng, Bernoulli(0.5), n))
age = randn(rng, n)
times = Float64[]; events = Int[]
for i in 1:n
    s = exp(3.0 + 0.6 * treatment[i] - 0.4 * age[i])
    ti = rand(rng, Weibull(2.0, s))
    if ti <= 50
        push!(times, ti); push!(events, 1)
    else
        push!(times, 50.0); push!(events, 0)
    end
end
println("SIM_EVENTS=", sum(events))
u = [events[i] == 1 ? 1.0e300 : times[i] for i in 1:n]  # huge, not Inf: JSON data round-trip
b_sim = @brm begin
    log(mu) ~ 1 + treatment + age
    effect(mu, Intercept) ~ Normal(0, 3.5)
    effect(mu, treatment) ~ Normal(0, 5)
    effect(mu, age) ~ Normal(0, 2.5)
    alpha ~ Gamma(2.0, 1.0)
    time ~ censored(Weibull(alpha, mu); upper=u)
end
sb = SBBRMI(b_sim((; time=times, treatment=treatment, age=age, u=u)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "sc_sim.stan"))
f = sample_fit(sb, problem, 32001)
open(joinpath(OUT, "sc_sim.json"), "w") do io
    JSON.print(io, f)
end
println("SC_SIM_DONE")
flush(stdout)
end

# ================= retention =================
rdf = CSV.read(joinpath(SCRATCH, "data", "retention.csv"), DataFrame)
println("RET_N=", nrow(rdf), " EVENTS=", sum(rdf.left))
month = Float64.(rdf.month)
lev = string.(rdf.level); gen = string.(rdf.gender); fld = string.(rdf.field)
ulv = sort(unique(lev)); ugn = sort(unique(gen)); ufl = sort(unique(fld))
println("LEVELS=", ulv, " GENDER=", ugn, " FIELDS=", ufl)
female = Float64.(gen .== ugn[1])
lvlM = Float64.(lev .== "Medium"); lvlH = Float64.(lev .== "High")
fcols = Dict(Symbol("f$(k-1)") => Float64.(fld .== ufl[k]) for k in 2:length(ufl))
sent = Float64.(rdf.sentiment); intent = Float64.(rdf.intention)
ur = [rdf.left[i] == 1 ? 1.0e300 : month[i] for i in 1:nrow(rdf)]
fids = Dict(x => i for (i, x) in enumerate(ufl))
fieldid = [fids[x] for x in fld]

function retention_block(dist, frailty)
    if dist == :weibull && !frailty
        return @brm begin
            log(mu) ~ 1 + female + lvlM + lvlH + f1 + f2 + f3 + f4 + f5 + sent + intent
            effect(mu, Intercept) ~ Normal(0, 5)
            effect(mu, female) ~ Normal(0, 2.5)
            effect(mu, lvlM) ~ Normal(0, 2.5)
            effect(mu, lvlH) ~ Normal(0, 2.5)
            effect(mu, f1) ~ Normal(0, 2.5)
            effect(mu, f2) ~ Normal(0, 2.5)
            effect(mu, f3) ~ Normal(0, 2.5)
            effect(mu, f4) ~ Normal(0, 2.5)
            effect(mu, f5) ~ Normal(0, 2.5)
            effect(mu, sent) ~ Normal(0, 1.5)
            effect(mu, intent) ~ Normal(0, 1.5)
            alpha ~ Gamma(2.0, 1.0)
            month ~ censored(Weibull(alpha, mu); upper=ur)
        end
    elseif dist == :exponential
        return @brm begin
            log(mu) ~ 1 + female + lvlM + lvlH + f1 + f2 + f3 + f4 + f5 + sent + intent
            effect(mu, Intercept) ~ Normal(0, 5)
            effect(mu, female) ~ Normal(0, 2.5)
            effect(mu, lvlM) ~ Normal(0, 2.5)
            effect(mu, lvlH) ~ Normal(0, 2.5)
            effect(mu, f1) ~ Normal(0, 2.5)
            effect(mu, f2) ~ Normal(0, 2.5)
            effect(mu, f3) ~ Normal(0, 2.5)
            effect(mu, f4) ~ Normal(0, 2.5)
            effect(mu, f5) ~ Normal(0, 2.5)
            effect(mu, sent) ~ Normal(0, 1.5)
            effect(mu, intent) ~ Normal(0, 1.5)
            month ~ censored(Exponential(mu); upper=ur)
        end
    else
        return @brm begin
            log(mu) ~ 1 + female + lvlM + lvlH + sent + intent + (1 | field)
            effect(mu, Intercept) ~ Normal(0, 5)
            effect(mu, female) ~ Normal(0, 2.5)
            effect(mu, lvlM) ~ Normal(0, 2.5)
            effect(mu, lvlH) ~ Normal(0, 2.5)
            effect(mu, sent) ~ Normal(0, 1.5)
            effect(mu, intent) ~ Normal(0, 1.5)
            alpha ~ Gamma(2.0, 1.0)
            month ~ censored(Weibull(alpha, mu); upper=ur)
        end
    end
end

basecols = (; month=month, female=female, lvlM=lvlM, lvlH=lvlH, sent=sent,
    intent=intent, ur=ur, f1=fcols[:f1], f2=fcols[:f2], f3=fcols[:f3],
    f4=fcols[:f4], f5=fcols[:f5], field=fieldid)
for (tag, dist, frailty, seed) in (("weibull", :weibull, false, 32002),
        ("exp", :exponential, false, 32003), ("frailty", :weibull, true, 32004))
    sb = SBBRMI(retention_block(dist, frailty)(basecols); mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model;
        path=joinpath(SCRATCH, ".out", "stan", "sc_ret_$tag.stan"))
    fr = sample_fit(sb, problem, seed)
    open(joinpath(OUT, "ret_$tag.json"), "w") do io
        JSON.print(io, fr)
    end
    println("SC_RET_$(uppercase(tag))_DONE")
    flush(stdout)
end
