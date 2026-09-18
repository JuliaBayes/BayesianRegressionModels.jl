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
# Notebook-faithful explicit B-spline basis (bambi: bs(hour, 8)), NOT s():
# the same variable smoothed twice collides in transpile
# (snag distributional-m-3f3a622f); explicit columns are also what the
# notebook fits.
function bspline_basis(x::AbstractVector{<:Real}, df::Int=8, degree::Int=3)
    nint = df - degree - 1
    qs = quantile(x, range(0, 1; length=nint + 2))[2:end-1]
    lo, hi = minimum(x), maximum(x)
    knots = vcat(fill(lo, degree + 1), qs, fill(hi, degree + 1))
    nc = length(knots) - degree - 1
    @assert nc == df
    function col(j, d, v)
        if d == 0
            return (knots[j] <= v < knots[j+1]) ||
                   (v == hi && j == nc) ? 1.0 : 0.0
        end
        a = knots[j+d] == knots[j] ? 0.0 :
            (v - knots[j]) / (knots[j+d] - knots[j]) * col(j, d - 1, v)
        b = knots[j+d+1] == knots[j+1] ? 0.0 :
            (knots[j+d+1] - v) / (knots[j+d+1] - knots[j+1]) * col(j + 1, d - 1, v)
        return a + b
    end
    B = zeros(length(x), nc)
    for i in eachindex(x), j in 1:nc
        B[i, j] = col(j, degree, x[i])
    end
    return B, knots
end
function bspline_predict(x::AbstractVector{<:Real}, knots, degree::Int=3)
    nc = length(knots) - degree - 1
    lo, hi = knots[1], knots[end]
    function col(j, d, v)
        if d == 0
            return (knots[j] <= v < knots[j+1]) ||
                   (v == hi && j == nc) ? 1.0 : 0.0
        end
        a = knots[j+d] == knots[j] ? 0.0 :
            (v - knots[j]) / (knots[j+d] - knots[j]) * col(j, d - 1, v)
        b = knots[j+d+1] == knots[j+1] ? 0.0 :
            (knots[j+d+1] - v) / (knots[j+d+1] - knots[j+1]) * col(j + 1, d - 1, v)
        return a + b
    end
    B = zeros(length(x), nc)
    for i in eachindex(x), j in 1:nc
        B[i, j] = col(j, degree, x[i])
    end
    return B
end

bdf = CSV.read(joinpath(SCRATCH, "data", "bike_sharing.csv"), DataFrame)
bdf = bdf[1:50:nrow(bdf), :]
hour = Float64.(bdf.hour); cnt = Int.(bdf.count)
println("BIKE_N=", length(hour))
Bm, knots = bspline_basis(hour)
println("BS_POU=", maximum(abs.(sum(Bm; dims=2) .- 1.0)))
mcols = Dict(Symbol("m$k") => Bm[:, k] for k in 1:8)
acols = Dict(Symbol("a$k") => Bm[:, k] for k in 1:8)
open(joinpath(OUT, "bike_knots.json"), "w") do io
    JSON.print(io, Dict("knots" => knots))
end
b_bk = @brm begin
    log(mu) ~ 0 + m1 + m2 + m3 + m4 + m5 + m6 + m7 + m8
    effect(mu, m1) ~ Normal(0, 5)
    effect(mu, m2) ~ Normal(0, 5)
    effect(mu, m3) ~ Normal(0, 5)
    effect(mu, m4) ~ Normal(0, 5)
    effect(mu, m5) ~ Normal(0, 5)
    effect(mu, m6) ~ Normal(0, 5)
    effect(mu, m7) ~ Normal(0, 5)
    effect(mu, m8) ~ Normal(0, 5)
    log(alpha) ~ 0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8
    effect(alpha, a1) ~ Normal(0, 1)
    effect(alpha, a2) ~ Normal(0, 1)
    effect(alpha, a3) ~ Normal(0, 1)
    effect(alpha, a4) ~ Normal(0, 1)
    effect(alpha, a5) ~ Normal(0, 1)
    effect(alpha, a6) ~ Normal(0, 1)
    effect(alpha, a7) ~ Normal(0, 1)
    effect(alpha, a8) ~ Normal(0, 1)
    count ~ NegativeBinomial2(mu, alpha)
end
sb = SBBRMI(b_bk((; count=cnt, mcols..., acols...)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "dist_bikes.stan"))
fb = sample_fit(sb, problem, 31003)
open(joinpath(OUT, "bikes.json"), "w") do io
    JSON.print(io, fb)
end
println("DIST_BIKES_DONE")
flush(stdout)
# posterior-mean mu + intervals on grid (bambi: 200 pts over 0..23)
hgrid = collect(range(0, 23; length=200))
Bg = bspline_predict(hgrid, knots)
mu_i = findall(n -> occursin("beta_pop", n) && occursin("_mu_", n), fb["names"])
al_i = findall(n -> occursin("beta_pop", n) && occursin("alpha", n), fb["names"])
println("BIKE_MU_N=", length(mu_i), " BIKE_ALPHA_N=", length(al_i))
Bmu = hcat([fb["draws"][i] for i in mu_i]...)
Bal = hcat([fb["draws"][i] for i in al_i]...)
sub = 1:4:size(Bmu, 1)
brows = map(eachindex(hgrid)) do gi
    mu_d = exp.(Bmu * Bg[gi, :])
    al_d = exp.(Bal * Bg[gi, :])
    qs = quantile(mu_d, [0.05, 0.5, 0.95])
    NB = NegativeBinomial.(al_d[sub], al_d[sub] ./ (al_d[sub] .+ mu_d[sub]))
    yq = quantile.(NB, 0.95)
    Dict("hour" => hgrid[gi], "mu" => mean(mu_d),
        "mu_q05" => qs[1], "mu_q50" => qs[2], "mu_q95" => qs[3],
        "y_q05" => mean(quantile.(NB, 0.05)), "y_q95" => mean(yq))
end
open(joinpath(OUT, "bikes_curves.json"), "w") do io
    JSON.print(io, Dict("grid" => brows,
        "obs" => [Dict("hour" => hour[i], "count" => cnt[i]) for i in eachindex(hour)]))
end
println("DIST_CURVES_DONE")
flush(stdout)
