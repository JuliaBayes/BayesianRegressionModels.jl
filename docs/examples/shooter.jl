# Bambi replication: https://bambinos.github.io/bambi/notebooks/shooter_crossed_random_ANOVA.html
# Run: julia --project=. shooter.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# bambi shooter_crossed_random_ANOVA: response rates (Gaussian) with
# subject-only vs subject+stimulus crossed random effects, then the
# shoot/don't-shoot Bernoulli with the full crossed structure.
# S() contrast codes (+-1) precomputed (notebook: formulae S()).
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "shooter")
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

df = CSV.read(joinpath(SCRATCH, "data", "shooter.csv"), DataFrame)
codes(v) = (u = sort(unique(v)); m = Dict(x => i for (i, x) in enumerate(u)); [m[x] for x in v])
race = string.(df.race); object = string.(df.object); resp = string.(df.response)
Srace = Float64.([r == "black" ? 1.0 : -1.0 for r in race])
Sobj = Float64.([o == "gun" ? 1.0 : -1.0 for o in object])
subject = codes(Int.(df.subject)); tgt = codes(string.(df.target))
# rate = responses/sec; timeout rows have no time -> drop (notebook: 98/3600)
tstr = string.(df.time)
ok = [try parse(Float64, s); true catch; false end for s in tstr]
println("RATE_N=", sum(ok))
rate = [ok[i] ? 1000.0 / parse(Float64, tstr[i]) : NaN for i in eachindex(tstr)]
m = ok

b_subj = @brm begin
    mu ~ 1 + Srace + Sobj + Srace & Sobj + (Srace + Sobj + Srace & Sobj | subject)
    effect(mu, Intercept) ~ Normal(0, 2.5)
    effect(mu, Srace) ~ Normal(0, 1)
    effect(mu, Sobj) ~ Normal(0, 1)
    effect(mu, Srace & Sobj) ~ Normal(0, 1)
    sigma ~ Exponential(1)
    rate ~ Normal(mu, sigma)
end
sb = SBBRMI(b_subj((; rate=rate[m], Srace=Srace[m], Sobj=Sobj[m],
    subject=subject[m])); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "shooter_subj.stan"))
f = sample_fit(sb, problem, 37001)
open(joinpath(OUT, "shooter_subj.json"), "w") do io
    JSON.print(io, f)
end
println("SHOOTER_SUBJ_DONE"); flush(stdout)

b_stim = @brm begin
    mu ~ 1 + Srace + Sobj + Srace & Sobj + (Srace + Sobj + Srace & Sobj | subject) + (Sobj | tgt)
    effect(mu, Intercept) ~ Normal(0, 2.5)
    effect(mu, Srace) ~ Normal(0, 1)
    effect(mu, Sobj) ~ Normal(0, 1)
    effect(mu, Srace & Sobj) ~ Normal(0, 1)
    sigma ~ Exponential(1)
    rate ~ Normal(mu, sigma)
end
sb = SBBRMI(b_stim((; rate=rate[m], Srace=Srace[m], Sobj=Sobj[m],
    subject=subject[m], tgt=tgt[m])); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "shooter_stim.stan"))
f = sample_fit(sb, problem, 37002)
open(joinpath(OUT, "shooter_stim.json"), "w") do io
    JSON.print(io, f)
end
println("SHOOTER_STIM_DONE"); flush(stdout)

# shoot/don't-shoot recode
function shootcode(o, r)
    o == "gun" ? (r == "correct" ? 1 : r == "incorrect" ? 0 : -1) :
        (r == "correct" ? 0 : r == "incorrect" ? 1 : -1)
end
shoot = [shootcode(object[i], resp[i]) for i in eachindex(resp)]
m2 = shoot .>= 0
println("SHOOT_N=", sum(m2))
b_resp = @brm begin
    eta ~ 1 + Srace + Sobj + Srace & Sobj + (Srace + Sobj + Srace & Sobj | subject) + (Sobj | tgt)
    effect(eta, Intercept) ~ Normal(0, 2.5)
    effect(eta, Srace) ~ Normal(0, 1)
    effect(eta, Sobj) ~ Normal(0, 1)
    effect(eta, Srace & Sobj) ~ Normal(0, 1)
    shoot ~ BernoulliLogit(eta)
end
sb = SBBRMI(b_resp((; shoot=shoot[m2], Srace=Srace[m2], Sobj=Sobj[m2],
    subject=subject[m2], tgt=tgt[m2])); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "shooter_resp.stan"))
f = sample_fit(sb, problem, 37003)
open(joinpath(OUT, "shooter_resp.json"), "w") do io
    JSON.print(io, f)
end
println("SHOOTER_RESP_DONE"); flush(stdout)
