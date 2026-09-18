# Bambi replication: https://bambinos.github.io/bambi/notebooks/orthogonal_polynomial_reg.html
# Run: julia --project=. orthogonal.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# bambi orthogonal_polynomial_reg: projectile x = -4.905 t^2 + 7 t + 1.5 + N(0, 0.2).
# Raw vs orthogonal-quadratic fits (notebook's own DGP structure, own seed).
using Random, Statistics, Distributions, LinearAlgebra, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "orthog")
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

rng = Xoshiro(31337)
t = collect(range(0.0, 2.0; length=100))
xtrue = -4.905 .* t .^ 2 .+ 7.0 .* t .+ 1.5
x = xtrue .+ randn(rng, 100) .* 0.2
keep = x .>= 0
t = t[keep]; x = x[keep]; xtrue = xtrue[keep]
println("ORTH_N=$(length(x))")

b_raw = @brm begin
    mu ~ 1 + t + t2
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, t) ~ Normal(0, 2.5)
    effect(mu, t2) ~ Normal(0, 2.5)
    sigma ~ Exponential(1)
    x ~ Normal(mu, sigma)
end
sb = SBBRMI(b_raw((; x=x, t=t, t2=t .^ 2)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "orth_raw.stan"))
f = sample_fit(sb, problem, 8001)
open(joinpath(OUT, "orth_raw.json"), "w") do io
    JSON.print(io, f)
end
println("ORTH_RAW_DONE")
flush(stdout)

# orthogonal basis via QR (bambi poly(t, 2) equivalent: orthonormal, no intercept col)
F = qr(hcat(ones(length(t)), t, t .^ 2))
Q = Matrix(F.Q)
R = F.R
q1 = Q[:, 2]; q2 = Q[:, 3]
b_orth = @brm begin
    mu ~ 1 + q1 + q2
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, q1) ~ Normal(0, 2.5)
    effect(mu, q2) ~ Normal(0, 2.5)
    sigma ~ Exponential(1)
    x ~ Normal(mu, sigma)
end
sb = SBBRMI(b_orth((; x=x, q1=q1, q2=q2)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "orth_q.stan"))
f = sample_fit(sb, problem, 8002)
# back-transform orthogonal coefs to raw (t^2, t, intercept) per draw
bi = findall(n -> occursin("beta_pop", n), f["names"])
B = hcat([f["draws"][i] for i in bi]...)
R22 = R[2:3, 2:3]; r1 = R[1, 2:3]; r11 = R[1, 1]
rawdraws = map(1:size(B, 1)) do d
    bq = B[d, 2:3]
    brow = R22 \ bq  # [t coef, t^2 coef]
    # raw = R^{-1} [b0*R11, b1, b2]: intercept keeps b0, minus R[1,2:3]*brow/R11
    icept = B[d, 1] - (r1[1] * brow[1] + r1[2] * brow[2]) / r11
    [icept, brow[1], brow[2]]
end
M = hcat(rawdraws...)'
raw = Dict("intercept" => Dict("mean" => mean(M[:, 1]), "q05" => quantile(M[:, 1], 0.05), "q95" => quantile(M[:, 1], 0.95)),
    "t" => Dict("mean" => mean(M[:, 2]), "q05" => quantile(M[:, 2], 0.05), "q95" => quantile(M[:, 2], 0.95)),
    "t2" => Dict("mean" => mean(M[:, 3]), "q05" => quantile(M[:, 3], 0.05), "q95" => quantile(M[:, 3], 0.95)))
open(joinpath(OUT, "orth_q.json"), "w") do io
    JSON.print(io, Dict("fit" => Dict("n_draws" => f["n_draws"],
        "divergences" => f["divergences"], "params" => f["params"]), "raw" => raw))
end
# posterior-mean fitted curves on a shared grid + observed series (for briefs)
tg = collect(range(minimum(t), maximum(t); length=61))
fr = JSON.parsefile(joinpath(OUT, "orth_raw.json"))
bri = findall(n -> occursin("beta_pop", n), fr["names"])
Br = hcat([fr["draws"][i] for i in bri]...)
curve_rows = map(tg) do g
    mu_r = Br[:, 1] .+ Br[:, 2] .* g .+ Br[:, 3] .* g .^ 2
    mu_q = M[:, 1] .+ M[:, 2] .* g .+ M[:, 3] .* g .^ 2
    Dict("t" => g, "raw_mean" => mean(mu_r),
        "raw_q05" => quantile(mu_r, 0.05), "raw_q95" => quantile(mu_r, 0.95),
        "q_mean" => mean(mu_q),
        "q_q05" => quantile(mu_q, 0.05), "q_q95" => quantile(mu_q, 0.95))
end
open(joinpath(OUT, "orth_curves.json"), "w") do io
    JSON.print(io, Dict("grid" => curve_rows,
        "obs" => [Dict("t" => t[i], "x" => x[i]) for i in eachindex(t)]))
end
println("ORTH_Q_DONE")
flush(stdout)
