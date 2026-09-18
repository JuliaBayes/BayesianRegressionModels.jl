# Bambi replication: https://bambinos.github.io/bambi/notebooks/mister_p.html
# Run: julia --project=. mister_p.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# bambi mister_p (MRP): biased CCES sample -> hierarchical binomial ->
# post-stratified state estimates. Cleaning pipeline mirrors the notebook
# exactly (recodes + lookups); the biased 5000-subsample uses own RNG.
# CCES input is vendored gzipped; unpacked to .out/tmp/ on first run.
using Random, Statistics, Distributions, CSV, DataFrames, JSON, LogExpFunctions
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "misterp")
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

states = ["AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY"]
lkup_states = Dict(i => states[i] for i in 1:50)
ethnicity = ["White","Black","Hispanic","Asian","Native American","Mixed","Other","Middle Eastern"]
lkup_eth = Dict(i => ethnicity[i] for i in 1:8)
edu = ["No HS","HS","Some college","Associates","4-Year College","Post-grad"]
lkup_edu = Dict(i => edu[i] for i in 1:6)

cces_csv = joinpath(SCRATCH, ".out", "tmp", "cces18.csv")
if !isfile(cces_csv)
    mkpath(dirname(cces_csv))
    run(pipeline(`gunzip -c $(joinpath(SCRATCH, "data", "mr_p_cces18_common_vv.csv.gz"))`, stdout=cces_csv))
end
raw = CSV.read(cces_csv, DataFrame)
println("CCES_N=", nrow(raw))
maybeint(x) = ismissing(x) ? missing :
    (try Int(x) catch; parse(Int, string(x)) end)
abortion = [ismissing(v) ? missing : abs(maybeint(v) - 2) for v in raw.CC18_321d]
state = [ismissing(x) ? missing : get(lkup_states, maybeint(x), missing) for x in raw.inputstate]
male = [ismissing(x) ? missing : abs(Float64(x) - 2) - 0.5 for x in raw.gender]
eth = [ismissing(x) ? missing : get(lkup_eth, maybeint(x), missing) for x in raw.race]
collapse = Set(["Asian","Other","Middle Eastern","Mixed","Native American"])
eth = [ismissing(e) ? missing : (e in collapse ? "Other" : e) for e in eth]
agey = [ismissing(x) ? missing : 2018 - maybeint(x) for x in raw.birthyr]
function ageband(a)
    a < 30 ? "18-29" : a < 40 ? "30-39" : a < 50 ? "40-49" :
        a < 60 ? "50-59" : a < 70 ? "60-69" : "70+"
end
age = [ismissing(a) ? missing : ageband(a) for a in agey]
ed = [ismissing(x) ? missing : get(lkup_edu, maybeint(x), missing) for x in raw.educ]
ed = [ismissing(e) ? missing : (e in ("Some college","Associates") ? "Some college" : e) for e in ed]
keep = .!ismissing.(state) .& .!ismissing.(eth) .& .!ismissing.(ed) .&
    .!ismissing.(abortion) .& isfinite.(male)
println("CLEAN_N=", sum(keep))
cdf = DataFrame(abortion=abortion[keep], state=state[keep], eth=eth[keep],
    male=male[keep], age=age[keep], edu=ed[keep])

slp = CSV.read(joinpath(SCRATCH, "data", "mr_p_statelevel_predictors.csv"), DataFrame)
repvote = Dict(string(r.state) => Float64(r.repvote) for r in eachrow(slp))
cdf.repvote = [repvote[s] for s in cdf.state]
# NOTE: notebook's `(male == 1) * 20` term is a no-op (male is +-0.5); omitted.
w = 5 .* cdf.repvote .+
    (cdf.age .== "18-29") .* 0.5 .+ (cdf.age .== "30-39") .* 1.0 .+
    (cdf.age .== "40-49") .* 2.0 .+ (cdf.age .== "50-59") .* 4.0 .+
    (cdf.age .== "60-69") .* 6.0 .+ (cdf.age .== "70+") .* 8.0 .+
    (cdf.eth .== "White") .* 1.05
rng = Xoshiro(1000)
samp = sample(rng, 1:nrow(cdf), Weights(w), 5000; replace=false)
sdf = cdf[samp, :]
println("SAMPLE_N=", nrow(sdf))

# aggregate to strata
gdf = combine(groupby(sdf, [:state, :eth, :male, :age, :edu]),
    :abortion => sum => :abortion, nrow => :n,
    :repvote => first => :repvote)
println("STRATA=", nrow(gdf))
codes(v) = (u = sort(unique(v)); m = Dict(x => i for (i, x) in enumerate(u)); ([m[x] for x in v], u))
st, ust = codes(gdf.state); et, uet = codes(gdf.eth)
ed_, ued = codes(gdf.edu)
me, ume = codes(string.(gdf.male) .* ":" .* gdf.eth)
ea, uea = codes(gdf.edu .* ":" .* gdf.age)
ee, uee = codes(gdf.edu .* ":" .* gdf.eth)
println("LEVELS state=", length(ust), " eth=", length(uet), " edu=", length(ued),
    " male:eth=", length(ume), " edu:age=", length(uea), " edu:eth=", length(uee))

b_hier = @brm begin
    logit(p) ~ 1 + male + repvote + (1 | state) + (1 | eth) + (1 | edu) +
        (1 | male_eth) + (1 | edu_age) + (1 | edu_eth)
    effect(p, Intercept) ~ Normal(0, 2.5)
    effect(p, male) ~ Normal(0, 2.5)
    effect(p, repvote) ~ Normal(0, 2.5)
    abortion ~ Binomial(n, p)
end
data = (; abortion=Int.(gdf.abortion), n=Int.(gdf.n),
    male=Float64.(gdf.male), repvote=Float64.(gdf.repvote),
    state=st, eth=et, edu=ed_, male_eth=me, edu_age=ea, edu_eth=ee)
sb = SBBRMI(b_hier(data); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "mrp_hier.stan"))
if isfile(joinpath(OUT, "mrp_hier.json"))
    println("MRP_HIER_CACHED")
    f = JSON.parsefile(joinpath(OUT, "mrp_hier.json"))
else
    f = sample_fit(sb, problem, 34001)
    open(joinpath(OUT, "mrp_hier.json"), "w") do io
        JSON.print(io, f)
    end
    println("MRP_HIER_DONE")
    flush(stdout)
end

# ================= post-stratification =================
# Established route (newgroups_predict.jl): replay the declaration on the
# census cells, transport draws (unseen combos resampled natively), read
# per-cell p through the descriptor — no manual coefficient mapping.
ps = CSV.read(joinpath(SCRATCH, "data", "mr_p_poststrat_df.csv"), DataFrame)
rename!(ps, :educ => :edu)
ps = ps[ps.n .> 0, :]  # drop zero-n cells (notebook: 11,040 of 12,000)
println("POSTSTRAT_NONZERO=", nrow(ps))
ps.repvote = [repvote[string(s)] for s in ps.state]
mksub(u) = Dict(x => i for (i, x) in enumerate(u))
ms, me_, md, mme, mea, mee = mksub(ust), mksub(uet), mksub(ued),
    mksub(ume), mksub(uea), mksub(uee)
# replay codes: fit index where seen, else fresh code past the fitted range
function replaycode(m, xs)
    base = length(m)
    extra = Dict{String,Int}()
    [let k = string(x)
         haskey(m, k) ? m[k] : get!(extra, k, base + length(extra) + 1)
     end for x in xs]
end
nc = nrow(ps)
pstate = replaycode(ms, ps.state)
peth = replaycode(me_, ps.eth)
pedu = replaycode(md, ps.edu)
pme = replaycode(mme, string.(ps.male) .* ":" .* string.(ps.eth))
pea = replaycode(mea, string.(ps.edu) .* ":" .* string.(ps.age))
pee = replaycode(mee, string.(ps.edu) .* ":" .* string.(ps.eth))
println("POSTSTRAT_CELLS=", nc, " maxcodes=",
    maximum(pstate), ",", maximum(peth), ",", maximum(pedu), ",",
    maximum(pme), ",", maximum(pea), ",", maximum(pee),
    " fitcodes=", length(ust), ",", length(uet), ",", length(ued), ",",
    length(ume), ",", length(uea), ",", length(uee))
new_df = (; abortion=zeros(Int, nc), n=Int.(ps.n),
    male=Float64.(ps.male), repvote=Float64.(ps.repvote),
    state=pstate, eth=peth, edu=pedu, male_eth=pme, edu_age=pea, edu_eth=pee)
d = BRM.brm_descriptor(b_hier, data; mod=@__MODULE__, name=:mrp)
d2 = BRM.brm_execute(d, :replay, new_df)
prob2 = BRM.brm_execute(d2, :fit)
unc_names = BridgeStan.param_unc_names(problem.model)
unc_new_names = BridgeStan.param_unc_names(prob2.model)
# unconstrained draws via exact constrain round-trip (fit object is gone).
# Cc/P are draws x params (transport takes draws x uparams, per newgroups).
Cc = hcat([Float64.(f["draws"][i]) for i in eachindex(f["names"])]...)
P = permutedims(hcat([BridgeStan.param_unconstrain(problem.model, collect(Cc[i, :]))
    for i in axes(Cc, 1)]...))
rt = BridgeStan.param_constrain(problem.model, collect(P[1, :]))
println("MRP_ROUNDTRIP_MAXERR=", maximum(abs.(rt .- Cc[1, :])))
moved = BRM.transport_draws(sb, d2.plan, P, unc_names,
    unc_new_names; rng=Xoshiro(34002))
println("MRP_TRANSPORT=", size(moved))
Pm = try
    Cn = BridgeStan.param_names(prob2.model; include_gq=true)
    Cm = permutedims(hcat([BridgeStan.param_constrain(prob2.model, collect(moved[i, :]); include_gq=true)
        for i in axes(moved, 1)]...))
    BRM.brm_output_draws(d2, Cm, string.(Cn); logical=:p)
catch e
    println("MRP_P_FALLBACK: ", split(sprint(showerror, e), "\n")[1])
    pred = BRM.brm_predictive_draws(d2, moved; problem=prob2, seed=34003)
    pk = only(filter(k -> occursin("abortion", string(k)), keys(pred)))
    pred[pk] ./ reshape(Float64.(ps.n), 1, :)
end
println("MRP_P=", size(Pm))
# state shares: weight cells by census share within state
states_ps = string.(ps.state)
share = zeros(nc)
for s in unique(states_ps)
    ix = findall(==(s), states_ps)
    tot = sum(ps.n[ix])
    share[ix] = Float64.(ps.n[ix]) ./ tot
end
st_tab = map(sort(unique(states_ps))) do s
    ix = findall(==(s), states_ps)
    wv = share[ix] ./ sum(share[ix])
    est = vec(Pm[:, ix] * wv)
    Dict("state" => s, "est" => mean(est),
        "q05" => quantile(est, 0.05), "q95" => quantile(est, 0.95),
        "cells" => length(ix))
end
# biased-sample raw shares per state (the "before adjustment" series)
raw_tab = map(sort(unique(states_ps))) do s
    ix = findall(==(s), string.(sdf.state))
    Dict("state" => s,
        "raw" => isempty(ix) ? missing : sum(sdf.abortion[ix]) / length(ix),
        "n" => length(ix))
end
open(joinpath(OUT, "mrp_states.json"), "w") do io
    JSON.print(io, Dict("mrp" => st_tab, "raw" => raw_tab))
end
println("MRP_STATES_DONE")
flush(stdout)
