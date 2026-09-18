# Bambi replication: https://bambinos.github.io/bambi/notebooks/survival_discrete_time_notebook.html
# Run: julia --project=. surv_disc.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# bambi survival_discrete_time_notebook: simulated cloglog DGP (truth recovery)
# + child data via the notebook's own Poisson-offset aggregated variant.
# (Binomial-cloglog child + spline variants await snag brm-tracer-inv-c-a19794bc.)
using Random, Statistics, Distributions, CSV, DataFrames, Dates, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "survdisc")
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

# ================= sim DGP (notebook structure, own seed) =================
rng = Xoshiro(98)
n = 2000
max_time = 8
treatment = Float64.(rand(rng, Bernoulli(0.5), n))
age_raw = randn(rng, n) .* 10.0 .+ 50.0
age = (age_raw .- sum(age_raw) / n) ./ sqrt(sum((age_raw .- sum(age_raw) / n) .^ 2) / n)
bh = [0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
ftime = Int[]; find_ = Int[]
for i in 1:n
    lp = 0.5 * treatment[i] + 0.3 * age[i]
    ft = max_time + 1; ev = 0
    for t in 1:max_time
        p = 1 - exp(-exp(log(bh[t]) + lp))
        if rand(rng) < p
            ft = t; ev = 1
            break
        end
    end
    push!(ftime, min(ft, max_time)); push!(find_, ev)
end
# person-period expansion
pp_ev = Int[]; pp_tr = Float64[]; pp_age = Float64[]; pp_per = Int[]
for i in 1:n
    for t in 1:ftime[i]
        push!(pp_ev, (t == ftime[i] && find_[i] == 1) ? 1 : 0)
        push!(pp_tr, treatment[i]); push!(pp_age, age[i]); push!(pp_per, t)
    end
end
println("SIM_PP_ROWS=$(length(pp_ev)) SIM_EVENTS=$(sum(pp_ev))")
# Exact cloglog via inline Bernoulli expression. (A 2-level ordinal
# Cumulative+Cloglog is loglog-shaped -- verified in brm_ordinal_lpmf -- so it
# cannot stand in here. Direct inv_cloglog awaits snag brm-tracer-inv-c-a19794bc.)
# No threshold exists, so all 8 period cell-means are identified.
P8 = hcat([Float64.(pp_per .== k) for k in 1:8]...)
b_sd = @brm begin
    eta ~ 0 + treatment + age + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8
    effect(eta, treatment) ~ Normal(0, 1)
    effect(eta, age) ~ Normal(0, 1)
    effect(eta, p1) ~ Normal(0, 5)
    effect(eta, p2) ~ Normal(0, 5)
    effect(eta, p3) ~ Normal(0, 5)
    effect(eta, p4) ~ Normal(0, 5)
    effect(eta, p5) ~ Normal(0, 5)
    effect(eta, p6) ~ Normal(0, 5)
    effect(eta, p7) ~ Normal(0, 5)
    effect(eta, p8) ~ Normal(0, 5)
    event ~ Bernoulli(1 - exp(-exp(eta)))
end
sb = SBBRMI(b_sd((; event=pp_ev, treatment=pp_tr, age=pp_age,
    p1=P8[:, 1], p2=P8[:, 2], p3=P8[:, 3], p4=P8[:, 4], p5=P8[:, 5], p6=P8[:, 6],
    p7=P8[:, 7], p8=P8[:, 8])); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "sd_sim.stan"))
f = sample_fit(sb, problem, 7001)
open(joinpath(OUT, "sd_sim.json"), "w") do io
    JSON.print(io, f)
end
println("SD_SIM_DONE")
flush(stdout)

# ================= child Poisson-offset aggregated =================
cdf = CSV.read(joinpath(SCRATCH, "data", "child.csv"), DataFrame)
cdf.birth_year = year.(Date.(string.(cdf.birthdate)))
cdf.birth_decade = (cdf.birth_year .÷ 10) .* 10
# person-period long frame
lp_ev = Int[]; lp_per = Int[]; lp_sex = String[]; lp_soc = String[]
lp_il = String[]; lp_bd = Int[]
for r in eachrow(cdf)
    T = Int(ceil(Float64(r.exit)))
    ev = Int(r.event)
    for t in 1:T
        push!(lp_ev, (t == T && ev == 1) ? 1 : 0)
        push!(lp_per, t); push!(lp_sex, string(r.sex)); push!(lp_soc, string(r.socBranch))
        push!(lp_il, string(r.illeg)); push!(lp_bd, r.birth_decade)
    end
end
ldf = DataFrame(period=lp_per, sex=lp_sex, soc=lp_soc, illeg=lp_il, bd=lp_bd, event=lp_ev)
gdf = combine(groupby(ldf, [:period, :sex, :soc, :illeg, :bd]),
    :event => sum => :events, nrow => :at_risk)
println("CHILD_STRATA=$(nrow(gdf))")
bd0 = Float64.(gdf.bd)
bdz = (bd0 .- sum(bd0) / length(bd0)) ./ sqrt(sum((bd0 .- sum(bd0) / length(bd0)) .^ 2) / length(bd0))
open(joinpath(OUT, "child_scale.json"), "w") do io
    JSON.print(io, Dict("mean" => sum(bd0) / length(bd0),
        "std" => sqrt(sum((bd0 .- sum(bd0) / length(bd0)) .^ 2) / length(bd0))))
end
female = Float64.(gdf.sex .== "female")
socB = Float64.(gdf.soc .== "business"); socF = Float64.(gdf.soc .== "farming"); socW = Float64.(gdf.soc .== "worker")
illy = Float64.(gdf.illeg .== "yes")
pers = sort(unique(gdf.period))
np_ = length(pers) - 1  # last period = reference (threshold absorbs location)
pcols = Dict(Symbol("q$k") => Float64.(gdf.period .== pers[k]) for k in 1:np_)
lexp = log.(Float64.(gdf.at_risk))
b_cp = @brm begin
    log(mu) ~ 1 + female + socB + socF + socW + illy + bdz + q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8 + q9 + q10 + q11 + q12 + q13 + q14 + offset(lexp)
    events ~ Poisson(mu)
end
# NOTE: period count is data-dependent; the formula above assumes 15 periods.
sb = SBBRMI(b_cp((; events=Int.(gdf.events), female=female, socB=socB, socF=socF, socW=socW,
    illy=illy, bdz=bdz, lexp=lexp, pcols...)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "sd_child.stan"))
f = sample_fit(sb, problem, 7002)
open(joinpath(OUT, "sd_child.json"), "w") do io
    JSON.print(io, f)
end
println("SD_CHILD_DONE")
flush(stdout)
