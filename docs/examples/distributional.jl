# Bambi replication: https://bambinos.github.io/bambi/notebooks/distributional_models.html
# Run: julia --project=. distributional.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# bambi distributional_models: Gamma constant/varying alpha (synthetic DGP,
# notebook structure, own seed) + bikes NegativeBinomial with splines on mu
# and alpha.
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "distributional")
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
    con_names = BridgeStan.param_names(problem.model; include_tp=true)
    C = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[:, i]); include_tp=true) for i in axes(P, 2)]...))
    rows = map(eachindex(con_names)) do i
        v = C[:, i]
        Dict("name" => string(con_names[i]), "mean" => mean(v), "sd" => std(v),
            "q05" => quantile(v, 0.05), "q50" => quantile(v, 0.5), "q95" => quantile(v, 0.95))
    end
    Dict("n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples, "params" => rows,
        "names" => string.(con_names), "draws" => [C[:, i] for i in eachindex(con_names)])
end

# ================= synthetic Gamma DGP (notebook structure) =================
rng = Xoshiro(121195)
N = 200
xg = rand(rng, N) .* 3.0 .- 1.5
shapeg = exp.(0.3 .+ 0.5 .* xg .+ randn(rng, N) .* 0.1)
mug = exp.(0.5 .+ 1.1 .* xg)
yg = [rand(rng, Gamma(shapeg[i], mug[i] / shapeg[i])) for i in 1:N]

# ---- constant alpha ----
b_gc = @brm begin
    log(mu) ~ 1 + x
    effect(mu, Intercept) ~ Normal(0, 2.5)
    effect(mu, x) ~ Normal(0, 2.5)
    alpha ~ Exponential(1)
    y ~ Gamma(alpha, mu / alpha)
end
sb = SBBRMI(b_gc((; y=yg, x=xg)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "dist_gamma_const.stan"))
fc = sample_fit(sb, problem, 31001)
open(joinpath(OUT, "gamma_const.json"), "w") do io
    JSON.print(io, fc)
end
println("DIST_GCONST_DONE")
flush(stdout)

# ---- varying alpha ----
b_gv = @brm begin
    log(mu) ~ 1 + x
    effect(mu, Intercept) ~ Normal(0, 2.5)
    effect(mu, x) ~ Normal(0, 2.5)
    log(alpha) ~ 1 + x
    effect(alpha, Intercept) ~ Normal(0, 1)
    effect(alpha, x) ~ Normal(0, 1)
    y ~ Gamma(alpha, mu / alpha)
end
sb = SBBRMI(b_gv((; y=yg, x=xg)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "dist_gamma_vary.stan"))
fv = sample_fit(sb, problem, 31002)
open(joinpath(OUT, "gamma_vary.json"), "w") do io
    JSON.print(io, fv)
end
println("DIST_GVARY_DONE")
flush(stdout)

# posterior-mean mu + predictive band on grid, both models (draws subset for bands)
xgrid = collect(range(-1.5, 1.5; length=50))
function gammacurve(f, varying)
    mu_i = findall(n -> occursin("beta_pop", n) && occursin("mu", n), f["names"])
    length(mu_i) == 2 || error("mu beta shape: $(f["names"])")
    bu = hcat([f["draws"][i] for i in mu_i]...)
    al_i = findall(n -> occursin("beta_pop", n) && occursin("alpha", n), f["names"])
    ai = findfirst(n -> n == "alpha", f["names"])
    ba = isempty(al_i) ? nothing : hcat([f["draws"][i] for i in al_i]...)
    rows = map(xgrid) do g
        mu_d = exp.(bu[:, 1] .+ bu[:, 2] .* g)
        al_d = varying ? exp.(ba[:, 1] .+ ba[:, 2] .* g) : f["draws"][ai]
        mu_m = mean(mu_d)
        # predictive quantiles: mean over draws of per-draw Gamma quantiles
        sub = 1:4:length(mu_d)
        lo = mean(quantile.(Gamma.(al_d[sub], mu_d[sub] ./ al_d[sub]), 0.025))
        hi = mean(quantile.(Gamma.(al_d[sub], mu_d[sub] ./ al_d[sub]), 0.975))
        Dict("x" => g, "mu" => mu_m, "lo" => lo, "hi" => hi)
    end
    rows
end
open(joinpath(OUT, "gamma_curves.json"), "w") do io
    JSON.print(io, Dict("const" => gammacurve(fc, false),
        "vary" => gammacurve(fv, true),
        "obs" => [Dict("x" => xg[i], "y" => yg[i]) for i in 1:N]))
end

# ================= bikes NB, splines on mu and alpha =================
# Native double smooth (snag distributional-m-3f3a622f landed): same
# variable smoothed on both linear predictors. Basis/knots/priors are
# BRM's s() defaults rather than the notebook's bs(hour, 8) + Normal(0, 5/1)
# columns, so fitted curves agree qualitatively, not numerically.
bdf = CSV.read(joinpath(SCRATCH, "data", "bike_sharing.csv"), DataFrame)
bdf = bdf[1:50:nrow(bdf), :]
hour = Float64.(bdf.hour); cnt = Int.(bdf.count)
println("BIKE_N=", length(hour))
b_bk = @brm begin
    log(mu) ~ 1 + s(hour)
    log(alpha) ~ 1 + s(hour)
    count ~ NegativeBinomial2(mu, alpha)
end
sb = SBBRMI(b_bk((; count=cnt, hour=hour)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "dist_bikes.stan"))
fb = sample_fit(sb, problem, 31003)
open(joinpath(OUT, "bikes.json"), "w") do io
    JSON.print(io, fb)
end
println("DIST_BIKES_DONE")
flush(stdout)
# posterior mu/alpha curves at fitted points through the descriptor (grid
# evaluation has no s() helper; fitted hours are dense over 0..23).
d_bk = BRM.brm_descriptor(b_bk, (; count=cnt, hour=hour); mod=@__MODULE__,
    name=:dist_bikes)
Cb = hcat(fb["draws"]...)
lmu = BRM.brm_output_draws(d_bk, Cb, fb["names"]; logical=:mu)
lal = BRM.brm_output_draws(d_bk, Cb, fb["names"]; logical=:alpha)
sub = 1:20:size(lmu, 1)  # thin for the predictive bands (exact NB quantiles)
brows = map(sortperm(hour)) do i
    mu_d = lmu[:, i]  # carriers are response-scale already (mu = exp(log_mu))
    al_d = lal[:, i]
    qs = quantile(mu_d, [0.05, 0.5, 0.95])
    # NB quantiles stall far into the tail: cap at sane magnitudes and report
    # the excluded tail-draw fraction rather than hanging on sampler garbage.
    ok = isfinite.(mu_d[sub]) .& isfinite.(al_d[sub]) .& (al_d[sub] .> 1e-3) .&
        (al_d[sub] .< 1e7) .& (mu_d[sub] .> 0) .& (mu_d[sub] .< 1e7)
    n_excl = count(.!ok)
    safeq(p) = begin
        any(ok) || return NaN
        v = try
            mean(quantile.(NegativeBinomial.(al_d[sub][ok], al_d[sub][ok] ./ (al_d[sub][ok] .+ mu_d[sub][ok])), p))
        catch
            NaN
        end
        isfinite(v) ? v : NaN
    end
    yq95, yq05 = safeq(0.95), safeq(0.05)
    Dict("hour" => hour[i], "mu" => mean(mu_d),
        "mu_q05" => qs[1], "mu_q50" => qs[2], "mu_q95" => qs[3],
        "y_q05" => (isnan(yq05) ? nothing : yq05),
        "y_q95" => (isnan(yq95) ? nothing : yq95), "n_excluded" => n_excl,
        "n_band_draws" => length(sub))
end
open(joinpath(OUT, "bikes_curves.json"), "w") do io
    JSON.print(io, Dict("grid" => brows,
        "obs" => [Dict("hour" => hour[i], "count" => cnt[i]) for i in eachindex(hour)],
        "note" => "y bands missing (null) where tail draws exceeded sane magnitudes"))
end
println("DIST_CURVES_DONE")
flush(stdout)
