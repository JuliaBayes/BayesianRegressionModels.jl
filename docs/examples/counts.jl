# Bambi replication: https://bambinos.github.io/bambi/notebooks/count_roaches.html
# Run: julia --project=. counts.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Batch5 counts: bambi negative_binomial (students) + count_roaches (Poisson + NB).
using Random, Statistics, Distributions, CSV, DataFrames, JSON, ReadStatTables
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "counts")
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

zsc(v) = (v .- sum(v) / length(v)) ./ sqrt(sum((v .- sum(v) / length(v)) .^ 2) / length(v))

# --- students absence (negative binomial, cell means, no intercept) ---
sdf = DataFrame(readstat(joinpath(SCRATCH, "data", "nb_data.dta")))
prog = Int.(sdf.prog)  # 1=General, 2=Academic, 3=Vocational
acad = Float64.(prog .== 2); gen = Float64.(prog .== 1); voc = Float64.(prog .== 3)
math0 = Float64.(sdf.math)
mathz = zsc(math0)
ysc = Int.(sdf.daysabs)
open(joinpath(OUT, "nb_scale.json"), "w") do io
    JSON.print(io, Dict("mean" => sum(math0) / length(math0),
        "std" => sqrt(sum((math0 .- sum(math0) / length(math0)) .^ 2) / length(math0))))
end
b_nb1 = @brm begin
    log(mu) ~ 0 + acad + gen + voc + mathz
    effect(mu, acad) ~ Normal(0, 2.5)
    effect(mu, gen) ~ Normal(0, 2.5)
    effect(mu, voc) ~ Normal(0, 2.5)
    effect(mu, mathz) ~ Normal(0, 2.5)
    phi ~ Exponential(1)
    daysabs ~ NegativeBinomial2(mu, phi)
end
sb = SBBRMI(b_nb1((; daysabs=ysc, acad=acad, gen=gen, voc=voc, mathz=mathz)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "nb_add.stan"))
f = sample_fit(sb, problem, 2001)
open(joinpath(OUT, "nb_add.json"), "w") do io
    JSON.print(io, f)
end
println("NB_ADD_DONE")
flush(stdout)

gen_m = gen .* mathz; voc_m = voc .* mathz
b_nb2 = @brm begin
    log(mu) ~ 0 + acad + gen + voc + mathz + gen_m + voc_m
    effect(mu, acad) ~ Normal(0, 2.5)
    effect(mu, gen) ~ Normal(0, 2.5)
    effect(mu, voc) ~ Normal(0, 2.5)
    effect(mu, mathz) ~ Normal(0, 2.5)
    effect(mu, gen_m) ~ Normal(0, 2.5)
    effect(mu, voc_m) ~ Normal(0, 2.5)
    phi ~ Exponential(1)
    daysabs ~ NegativeBinomial2(mu, phi)
end
sb = SBBRMI(b_nb2((; daysabs=ysc, acad=acad, gen=gen, voc=voc, mathz=mathz, gen_m=gen_m, voc_m=voc_m)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "nb_int.stan"))
f = sample_fit(sb, problem, 2002)
open(joinpath(OUT, "nb_int.json"), "w") do io
    JSON.print(io, f)
end
println("NB_INT_DONE")
flush(stdout)

# --- roaches (Poisson + NB with offset) ---
rdf = CSV.read(joinpath(SCRATCH, "data", "roaches.csv"), DataFrame)
yr = Int.(rdf.y)
# notebook rescales roach1 by 1/100 before fitting; match it exactly (no z-scoring)
r1h = Float64.(rdf.roach1) ./ 100.0
trt = Float64.(rdf.treatment); sen = Float64.(rdf.senior)
lexp = log.(Float64.(rdf.exposure2))
b_rp = @brm begin
    log(mu) ~ 1 + r1h + treatment + senior + offset(lexp)
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, r1h) ~ Normal(0, 2.5)
    effect(mu, treatment) ~ Normal(0, 2.5)
    effect(mu, senior) ~ Normal(0, 2.5)
    y ~ Poisson(mu)
end
sb = SBBRMI(b_rp((; y=yr, r1h=r1h, treatment=trt, senior=sen, lexp=lexp)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "roach_pois.stan"))
f = sample_fit(sb, problem, 2003)
open(joinpath(OUT, "roach_pois.json"), "w") do io
    JSON.print(io, f)
end
println("ROACH_POIS_DONE")
flush(stdout)

b_rnb = @brm begin
    log(mu) ~ 1 + r1h + treatment + senior + offset(lexp)
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, r1h) ~ Normal(0, 2.5)
    effect(mu, treatment) ~ Normal(0, 2.5)
    effect(mu, senior) ~ Normal(0, 2.5)
    phi ~ Exponential(1)
    y ~ NegativeBinomial2(mu, phi)
end
sb = SBBRMI(b_rnb((; y=yr, r1h=r1h, treatment=trt, senior=sen, lexp=lexp)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "roach_nb.stan"))
f = sample_fit(sb, problem, 2004)
open(joinpath(OUT, "roach_nb.json"), "w") do io
    JSON.print(io, f)
end
println("ROACH_NB_DONE")
flush(stdout)
