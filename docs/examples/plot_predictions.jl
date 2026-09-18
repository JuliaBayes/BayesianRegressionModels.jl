# Bambi replication: https://bambinos.github.io/bambi/notebooks/plot_predictions.html
# Run: julia --project=. plot_predictions.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/plotpred/ + .out/stan/ (gitignored).
# Conditional adjusted predictions via the native engine (brm_prediction_grid /
# brm_conditional_draws, snag replicate-bambi-e064da10) on the notebook's three
# models: mtcars Gaussian, students negative-binomial interaction, movies
# Bernoulli. Categoricals are hand dummies (counts.jl pattern) with intercept
# form so grid typicals are valid cells; the notebook's `0 +` full-dummy form
# gives identical predictions. Link/response self-consistency asserted per family.
# Gaussian target=:mean is snagged upstream (brm-conditional-bf9d2446: GLM
# emission lacks the LP carrier), so the mtcars curves use target=:predictive.
using Random, Statistics, Distributions, CSV, DataFrames, JSON, ReadStatTables
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "plotpred")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

function sample_engine(builder, dat, tag, seed; draws=800)
    sb = SBBRMI(builder(dat); mod=@__MODULE__)
    problem = BRM.stan_instantiate(sb;
        path=joinpath(SCRATCH, ".out", "stan", "plotpr_$tag.stan"))
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
    d = BRM.brm_descriptor(builder, dat; mod=@__MODULE__, name=Symbol("plotpr_$tag"))
    println("FIT_DONE tag=$tag draws=", size(C), " div=", fit.n_divergent_samples)
    return d, U, fit.n_divergent_samples
end

res = Dict{String,Any}()

# ---------- 1. mtcars Gaussian ----------
mt = CSV.read(joinpath(SCRATCH, "data", "mtcars.csv"), DataFrame)
mpg = Float64.(mt.mpg); hp = Float64.(mt.hp); wt = Float64.(mt.wt)
cylm = Float64.(Int.(mt.cyl) .== 6); cylh = Float64.(Int.(mt.cyl) .== 8)
gearb = Float64.(Int.(mt.gear) .== 4); gearc = Float64.(Int.(mt.gear) .== 5)
dmt = (; mpg=mpg, hp=hp, wt=wt, cylm=cylm, cylh=cylh, gearb=gearb, gearc=gearc)
bmt = @brm begin
    mu ~ 1 + hp + wt + hp & wt + cylm + cylh + gearb + gearc
    effect(mu, Intercept) ~ Normal(20, 10)
    effect(mu, hp) ~ Normal(0, 1)
    effect(mu, wt) ~ Normal(0, 5)
    effect(mu, hp & wt) ~ Normal(0, 1)
    effect(mu, cylm) ~ Normal(0, 5)
    effect(mu, cylh) ~ Normal(0, 5)
    effect(mu, gearb) ~ Normal(0, 5)
    effect(mu, gearc) ~ Normal(0, 5)
    sigma ~ Exponential(1)
    mpg ~ Normal(mu, sigma)
end
dd1, U1, divmt = sample_engine(bmt, dmt, "mtcars", 44021)
# hp curve at typicals (low/A cell): must decline like the notebook
ghp = BRM.brm_prediction_grid(dd1; focal=:hp, n=25)
chp = BRM.brm_conditional_draws(dd1, U1, ghp; focal=:hp, target=:predictive, seed=31)
shp = BRM.brm_summarize_draws(chp.draws; probs=[0.94])
mh = getproperty.(shp, :mean)
@assert mh[end] < mh[1] - 2.0 "mpg must fall over the hp grid"
# hp x wt factorial (notebook cell 15, 50x5 -> 25x5 here)
gwt = BRM.brm_prediction_grid(dd1;
    focal=[:hp => collect(range(50, 350; length=25)), :wt => collect(range(1, 6; length=5))])
cwt = BRM.brm_conditional_draws(dd1, U1, gwt; focal=[:hp, :wt], target=:predictive, seed=32)
swt = BRM.brm_summarize_draws(cwt.draws; probs=[0.94])
# gear x cyl one-hot panel (notebook cell 17)
mpgfill = Statistics.median(mpg); hpm = sum(hp) / length(hp); wtm = sum(wt) / length(wt)
cells = [(cm, ch, gb, gc) for cm in (0.0, 1.0) for ch in (0.0, 1.0)
    for gb in (0.0, 1.0) for gc in (0.0, 1.0)
    if cm + ch <= 1.0 && gb + gc <= 1.0]
gcat = (; mpg=fill(mpgfill, 9), hp=fill(hpm, 9), wt=fill(wtm, 9),
    cylm=[c[1] for c in cells], cylh=[c[2] for c in cells],
    gearb=[c[3] for c in cells], gearc=[c[4] for c in cells])
ccat = BRM.brm_conditional_draws(dd1, U1, gcat; focal=[:cylm, :cylh, :gearb, :gearc],
    target=:predictive, seed=33)
scat = BRM.brm_summarize_draws(ccat.draws; probs=[0.94])
println("MTCARS_DONE hpgrid=", length(shp))
res["mtcars_hp"] = [Dict("hp" => ghp.hp[i], "mean" => shp[i].mean,
    "lo" => shp[i].lower_1, "hi" => shp[i].upper_1) for i in eachindex(shp)]

# ---------- 2. students negative-binomial interaction ----------
sdf = DataFrame(readstat(joinpath(SCRATCH, "data", "nb_data.dta")))
prog = Int.(sdf.prog)  # 1=General, 2=Academic, 3=Vocational
gen = Float64.(prog .== 1); voc = Float64.(prog .== 3)
math0 = Float64.(sdf.math); mathz = (math0 .- sum(math0) / length(math0)) ./
    sqrt(sum((math0 .- sum(math0) / length(math0)) .^ 2) / length(math0))
ysc = Int.(sdf.daysabs)
dnb = (; daysabs=ysc, gen=gen, voc=voc, mathz=mathz)
bnb = @brm begin
    log(mu) ~ 1 + gen + voc + mathz + gen & mathz + voc & mathz
    effect(mu, Intercept) ~ Normal(0, 2.5)
    effect(mu, gen) ~ Normal(0, 2.5)
    effect(mu, voc) ~ Normal(0, 2.5)
    effect(mu, mathz) ~ Normal(0, 2.5)
    effect(mu, gen & mathz) ~ Normal(0, 2.5)
    effect(mu, voc & mathz) ~ Normal(0, 2.5)
    phi ~ Exponential(1)
    daysabs ~ NegativeBinomial2(mu, phi)
end
dd2, U2, divnb = sample_engine(bnb, dnb, "nbint", 44022)
# math curves per program (academic baseline + pinned cells)
curves = Dict{String,Any}()
for (lab, gv, vv) in (("Academic", 0.0, 0.0), ("General", 1.0, 0.0), ("Vocational", 0.0, 1.0))
    g = BRM.brm_prediction_grid(dd2; focal=:mathz, n=25, fixed=(; gen=gv, voc=vv))
    c = BRM.brm_conditional_draws(dd2, U2, g; focal=:mathz)
    s = BRM.brm_summarize_draws(c.draws; probs=[0.94])
    m = getproperty.(s, :mean)
    @assert m[end] < m[1] "daysabs must fall over math for $lab"
    curves[lab] = [Dict("mathz" => g.mathz[i], "mean" => s[i].mean,
        "lo" => s[i].lower_1, "hi" => s[i].upper_1) for i in eachindex(s)]
end
# log link: response == exp(link)
g0 = BRM.brm_prediction_grid(dd2; focal=:mathz, n=9)
cr = BRM.brm_conditional_draws(dd2, U2, g0; focal=:mathz)
cl = BRM.brm_conditional_draws(dd2, U2, g0; focal=:mathz, scale=:link)
# :link is the predictor value the likelihood sees, which for LHS-linked
# `log(mu)` is already response-mapped: link and response coincide here
# (engine docstring). The genuine link/response split is asserted on movies.
@assert maximum(abs.(cr.draws .- cl.draws) ./ cr.draws) < 1e-6
# predictive target is wider than the mean (notebook cell 11/12 point)
cp = BRM.brm_conditional_draws(dd2, U2, g0; focal=:mathz, target=:predictive, seed=34)
sr = BRM.brm_summarize_draws(cr.draws; probs=[0.94])
sp = BRM.brm_summarize_draws(cp.draws; probs=[0.94])
wmean = mean([r.upper_1 - r.lower_1 for r in sr])
wpred = mean([r.upper_1 - r.lower_1 for r in sp])
@assert wpred > 2 * wmean
println("NBINT_DONE wmean=", round(wmean; digits=2), " wpred=", round(wpred; digits=2))
res["nb_predw"] = Dict("mean" => wmean, "predictive" => wpred)
res["nb_curves"] = curves

# ---------- 3. movies Bernoulli ----------
mv_gz = joinpath(SCRATCH, "data", "movies.csv.gz")
mv_csv = joinpath(SCRATCH, ".out", "tmp", "movies.csv")
if !isfile(mv_csv)
    mkpath(dirname(mv_csv))
    run(pipeline(`gunzip -c $mv_gz`, stdout=mv_csv))
end
mv = CSV.read(mv_csv, DataFrame)
style = [r == 1 ? "Action" : (c == 1 ? "Comedy" : (dr == 1 ? "Drama" : "Other"))
    for (r, c, dr) in zip(Int.(mv.Action), Int.(mv.Comedy), Int.(mv.Drama))]
fresh = Int.(Float64.(mv.rating) .>= 8)
len = Float64.(mv.length)
keep = len .< 240
style, fresh, len = style[keep], fresh[keep], len[keep]
isact = Float64.(style .== "Action"); iscom = Float64.(style .== "Comedy")
isdra = Float64.(style .== "Drama")
lenz = (len .- sum(len) / length(len)) ./
    sqrt(sum((len .- sum(len) / length(len)) .^ 2) / length(len))
dmv = (; fresh=fresh, lenz=lenz, isact=isact, iscom=iscom, isdra=isdra)
bmv = @brm begin
    eta ~ 1 + lenz + isact + iscom + isdra + lenz & isact + lenz & iscom + lenz & isdra
    effect(eta, Intercept) ~ Normal(0, 2.5)
    effect(eta, lenz) ~ Normal(0, 2.5)
    effect(eta, isact) ~ Normal(0, 2.5)
    effect(eta, iscom) ~ Normal(0, 2.5)
    effect(eta, isdra) ~ Normal(0, 2.5)
    effect(eta, lenz & isact) ~ Normal(0, 2.5)
    effect(eta, lenz & iscom) ~ Normal(0, 2.5)
    effect(eta, lenz & isdra) ~ Normal(0, 2.5)
    fresh ~ BernoulliLogit(eta)
end
dd3, U3, divmv = sample_engine(bmv, dmv, "movies", 44023)
gm = BRM.brm_prediction_grid(dd3; focal=:lenz, n=25)
cm_resp = BRM.brm_conditional_draws(dd3, U3, gm; focal=:lenz)
cm_link = BRM.brm_conditional_draws(dd3, U3, gm; focal=:lenz, scale=:link)
@assert minimum(cm_resp.draws) >= 0.0 && maximum(cm_resp.draws) <= 1.0
@assert maximum(abs.(cm_resp.draws .- 1.0 ./ (1.0 .+ exp.(-cm_link.draws)))) < 1e-6
sm = BRM.brm_summarize_draws(cm_resp.draws; probs=[0.94])
println("MOVIES_DONE n=", length(fresh))
res["movies_curve"] = [Dict("lenz" => gm.lenz[i], "mean" => sm[i].mean,
    "lo" => sm[i].lower_1, "hi" => sm[i].upper_1) for i in eachindex(sm)]
res["divergences"] = [divmt, divnb, divmv]

open(joinpath(OUT, "plotpred.json"), "w") do io
    JSON.print(io, res)
end
println("PLOTPRED_DONE")
flush(stdout)
