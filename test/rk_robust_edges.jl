# test/rk_robust_edges.jl — SB reference battery for the kernel-lane robustness
# leg (pair 5, BayesianRegressionModels:rk:kernel:robust).
#
# Edge-axis inputs over the matrix-a slice families that pair 1 does NOT
# cover: degenerate shapes (single group, unbalanced n=(1,5), n=1, nsub=1/T=1
# kernel plates, n=0 empty) and boundary link/family combos (Bernoulli
# complete separation, extreme predictors). Each case pins the SBBRMI
# BridgeStan value (propto=false, jacobian=true) at u=zeros(dim) against an
# independently derived Distributions.jl oracle, plus dimension and an
# FD gradient self-check. These numbers back RK parity re-verification
# after hardening fixes.
#
# RUN: `julia --project=test test/rk_robust_edges.jl`
# BRM_ROBUST_RUNTIME=0 skips the BridgeStan layer (emission + stanc only).

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Binomial, Cauchy, Exponential, LocationScale, LogNormal,
    MixtureModel, MvNormal, Normal, Poisson, TDist, logpdf, pdf
using LinearAlgebra
using LogExpFunctions: logit
import BridgeStan as BS

const BRM = BayesianRegressionModels
const ROBUST_CACHE = joinpath(tempdir(), "brm-robust-edges")
const RUN_BRIDGESTAN = get(ENV, "BRM_ROBUST_RUNTIME", "1") != "0"

# World-age-safe trace entries throughout (brm-use: scripted code must use
# the BRM entries, never direct StanBlocks.stan_* calls on a fresh build).
function robust_build(brmi; sb_kwargs...)
    sb = SBBRMI(brmi; mod=@__MODULE__, sb_kwargs...)
    @test BRM.transpiles(sb)
    code = BRM.stan_code(sb)
    checked = StanBlocks.stanc_check(code; warn_pedantic=false)
    checked.ok || @error "stanc rejected robust edge case" output=checked.output
    @test checked.ok
    return sb
end

function robust_value(sb, tag)
    isdir(ROBUST_CACHE) || mkpath(ROBUST_CACHE)
    path = joinpath(ROBUST_CACHE, replace(tag, "/" => "_") * ".stan")
    problem = BRM.stan_instantiate(sb; path)
    d = LogDensityProblems.dimension(problem)
    q = zeros(d)
    g = zeros(d)
    lp, _ = BS.log_density_gradient!(
        problem.model, q, g; propto=false, jacobian=true)
    return d, lp, g, problem
end

function robust_findiff(problem, d)
    q = zeros(d)
    h = cbrt(eps(Float64))
    fd = zeros(d)
    for i in eachindex(q)
        qp, qm = copy(q), copy(q)
        qp[i] += h
        qm[i] -= h
        fp = BS.log_density(problem.model, qp; propto=false, jacobian=true)
        fm = BS.log_density(problem.model, qm; propto=false, jacobian=true)
        fd[i] = (fp - fm) / (2h)
    end
    return fd
end

# Oracle inputs shared by the hierarchical cases. Plain `(1 | g)` ranef
# selects exact total coefficients (brm-use): totals T are sampled under
# the collapsed marginal T ~ N(A*loc, tau^2*I + A*diag(1/prec)*A') with
# beta integrated out; the totals scale default for plain scalar
# intercepts is LogNormal(0,1) (src/total_effects_plan.jl). At u=0 all
# Jacobians are 0 (identity/lb/exp at the origin).
const _Y1 = [0.5, -0.2, 0.1, 0.9]
const _LL1 = sum(logpdf.(Normal(0, 1), _Y1))
const _Y2 = [0.5, -0.2, 0.1, 0.9, 1.4, 1.1]
const _LL2 = sum(logpdf.(Normal(0, 1), _Y2))
const _STDN0 = logpdf(Normal(0, 1), 0.0)
const _EXP1 = logpdf(Exponential(1), 1.0)

@testset "robust edges G1 single group (default totals)" begin
    brmi = @brm (y=_Y1, g=[1, 1, 1, 1]) begin
        mu ~ 1 + (1 | g)
        y ~ Normal(mu, sigma)
        sigma ~ Exponential(1)
    end
    sb = robust_build(brmi)
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "E-hier-1/G1-default")
        @test d == 3
        oracle = _LL1 + logpdf(Normal(0, sqrt(2)), 0.0) +
            logpdf(LogNormal(0, 1), 1.0) + _EXP1
        @test lp ≈ oracle atol = 1e-9
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end

@testset "robust edges G1 single group (centered)" begin
    brmi = @brm (y=_Y1, g=[1, 1, 1, 1]) begin
        mu ~ 1 + (1 | g)
        y ~ Normal(mu, sigma)
        sigma ~ Exponential(1)
    end
    sb = robust_build(brmi; centered_groups=[:g])
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "E-hier-1/G1-centered")
        @test d == 4
        # beta ~ N(0,1), log_scale ~ N(0,1), xi ~ N(0, exp(0)).
        oracle = _LL1 + 3 * _STDN0 + _EXP1
        @test lp ≈ oracle atol = 1e-9
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end

@testset "robust edges unbalanced G=2 n=(1,5)" begin
    brmi = @brm (y=_Y2, g=[1, 2, 2, 2, 2, 2]) begin
        mu ~ 1 + (1 | g)
        y ~ Normal(mu, sigma)
        sigma ~ Exponential(1)
    end
    sb = robust_build(brmi)
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "E-hier-2/unbalanced")
        @test d == 4
        oracle = _LL2 +
            logpdf(MvNormal(zeros(2), [2.0 1.0; 1.0 2.0]), zeros(2)) +
            logpdf(LogNormal(0, 1), 1.0) + _EXP1
        @test lp ≈ oracle atol = 1e-9
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end

@testset "robust edges Bernoulli complete separation" begin
    x6 = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    for (tag, y) in (("E-bern-1/all0", [0, 0, 0, 0, 0, 0]),
            ("E-bern-1/all1", [1, 1, 1, 1, 1, 1]))
        brmi = @brm (y=y, x=x6) begin
            mu ~ 1 + x
            y ~ BernoulliLogit(mu)
        end
        sb = robust_build(brmi)
        if RUN_BRIDGESTAN
            d, lp, g, problem = robust_value(sb, tag)
            @test d == 2
            # eta=0 at beta=0: n*log(0.5) + two standard-Normal priors.
            @test lp ≈ 6 * log(0.5) + 2 * _STDN0 atol = 1e-9
            @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
        end
    end
end

@testset "robust edges Bernoulli extreme predictors" begin
    brmi = @brm (y=[0, 1, 0, 1, 0, 1],
        x=[1000.0, -1000.0, 500.0, -500.0, 0.0, 1.0]) begin
        mu ~ 1 + x
        y ~ BernoulliLogit(mu)
    end
    sb = robust_build(brmi)
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "E-bern-2/bigx")
        @test d == 2
        @test lp ≈ 6 * log(0.5) + 2 * _STDN0 atol = 1e-9
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end

@testset "robust edges Gaussian n=1" begin
    brmi = @brm (y=[0.5], x=[1.0]) begin
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
        sigma ~ Exponential(1)
    end
    sb = robust_build(brmi)
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "E-glm-1/n1")
        @test d == 3
        oracle = logpdf(Normal(0, 1), 0.5) + 2 * _STDN0 + _EXP1
        @test lp ≈ oracle atol = 1e-9
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end

@testset "robust edges kernel plate nsub=1 T=1" begin
    brmi = @brm (; t=[[1.0]], y=[[1.0]], intercept=[0.2], slope=[0.1]) begin
        sigma ~ Exponential(1)
        pred ~ kernel(t, intercept, slope, y) do ts, a, b, yy
            mu = a + b * ts
            yy ~ normal(mu, sigma)
            mu
        end
    end
    sb = robust_build(brmi)
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "E-k1/nsub1-T1")
        @test d == 1
        oracle = logpdf(Normal(0.3, 1.0), 1.0) + _EXP1
        @test lp ≈ oracle atol = 1e-9
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end

@testset "robust edges kernel plate nsub=1 all-scalar" begin
    brmi = @brm (x=[0.1], y=[0.5], intercept=[0.3]) begin
        sigma ~ Exponential(1)
        pred ~ kernel(x, intercept, y) do xx, a, yy
            mu = a + 2xx
            yy ~ normal(mu, sigma)
            mu
        end
    end
    sb = robust_build(brmi)
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "E-k2/nsub1")
        @test d == 1
        oracle = logpdf(Normal(0.5, 1.0), 0.5) + _EXP1
        @test lp ≈ oracle atol = 1e-9
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end

@testset "robust edges standardize degenerate columns reject loudly" begin
    @test_throws "requires nonzero sample variance" SBBRMI(
        (@brm (y=_Y2, x=fill(2.0, 6)) begin
            mu ~ 1 + standardize(x)
            y ~ Normal(mu, sigma)
            sigma ~ Exponential(1)
        end); mod=@__MODULE__)
    @test_throws "requires at least two training values" SBBRMI(
        (@brm (y=[0.5], x=[1.0]) begin
            mu ~ 1 + standardize(x)
            y ~ Normal(mu, sigma)
            sigma ~ Exponential(1)
        end); mod=@__MODULE__)
end

@testset "robust edges n=0 empty data evaluates prior-only" begin
    # Current behavior under audit (verdict brief): empty data emits and
    # evaluates to exactly the prior. Pinned for change-detection; if the
    # owner decides n=0 must reject, this becomes a @test_throws.
    brmi = @brm (y=Float64[], x=Float64[]) begin
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
        sigma ~ Exponential(1)
    end
    sb = robust_build(brmi)
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "F1/empty")
        @test d == 3
        @test lp ≈ 2 * _STDN0 + _EXP1 atol = 1e-9
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end

# General kernel oracles on the SB side use the same public linear panel
# fixtures as RK parity. Historical PK implementations remain in Git and
# belong to downstream RKPPLBench. SB cells use undotted SLIC arithmetic.
# Convention: value at constrained sigma=1.0 (u=[0.0])
# equals the oracle; the unconstrained gradient equals the constrained
# oracle gradient + 1 (exp-Jacobian).
const _VECTOR_CELL_COLS = (;
    t=[[0.5, 1.0, 2.0, 4.0] for _ in 1:3],
    y=[[0.1, 0.4, -0.2, 0.8] for _ in 1:3],
    intercept=[0.2, -0.3, 0.7], slope=[0.1, -0.2, 0.05],
)
const _SCALAR_CELL_COLS = (;
    x=[0.1, 0.2, 0.15, 0.25],
    intercept=[0.3, -0.2, 0.6, 0.4], y=[0.5, 1.2, 0.2, -0.3],
)

@testset "robust baseline SB ordinary vector-cell oracle" begin
    brmi = @brm _VECTOR_CELL_COLS begin
        sigma ~ Exponential(1)
        pred ~ kernel(t, intercept, slope, y) do ts, a, b, yy
            mu = a + b * ts
            yy ~ normal(mu, sigma)
            mu
        end
    end
    sb = robust_build(brmi)
    # Independent scalar-loop oracle at sigma=1.0 (constrained d/dσ).
    ll, sq, n = 0.0, 0.0, 0
    for s in 1:3
        a, b = _VECTOR_CELL_COLS.intercept[s], _VECTOR_CELL_COLS.slope[s]
        for (t, y) in zip(_VECTOR_CELL_COLS.t[s], _VECTOR_CELL_COLS.y[s])
            mu = a + b * t
            ll += logpdf(Normal(mu, 1.0), y)
            sq += abs2(y - mu)
            n += 1
        end
    end
    hand_v, hand_g = ll + _EXP1, sq - n - 1.0
    @test hand_v ≈ -15.465074898456072 atol = 1e-9
    @test hand_g ≈ -6.124375 atol = 1e-8
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "SB-ordinary-vector")
        @test d == 1
        @test lp ≈ hand_v atol = 1e-9
        @test g[1] ≈ hand_g + 1.0 atol = 1e-8
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end

@testset "robust baseline SB ordinary scalar-cell oracle" begin
    brmi = @brm _SCALAR_CELL_COLS begin
        sigma ~ Exponential(1)
        pred ~ kernel(x, intercept, y) do xx, a, yy
            mu = a + 2xx
            yy ~ normal(mu, sigma)
            mu
        end
    end
    sb = robust_build(brmi)
    ll, sq, n = 0.0, 0.0, 0
    for i in eachindex(_SCALAR_CELL_COLS.y)
        mu = _SCALAR_CELL_COLS.intercept[i] + 2 * _SCALAR_CELL_COLS.x[i]
        y = _SCALAR_CELL_COLS.y[i]
        ll += logpdf(Normal(mu, 1.0), y)
        sq += abs2(y - mu)
        n += 1
    end
    hand_v, hand_g = ll + _EXP1, sq - n - 1.0
    @test hand_v ≈ -6.140754132818691 atol = 1e-9
    @test hand_g ≈ -2.07 atol = 1e-8
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "SB-ordinary-scalar")
        @test d == 1
        @test lp ≈ hand_v atol = 1e-9
        @test g[1] ≈ hand_g + 1.0 atol = 1e-8
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end

# Matrix-b edges: HSGP validity floor (k=1), few-points HSGP, Poisson
# all-zero/n=1, Binomial extreme-eta/single-trial, single-component and
# few-obs mixtures, single-slope horseshoe, Poisson-GLMM single group.
#
# Truncation convention (verified empirically 2026-09-27): a bare Stan `~`
# with declaration bounds does NOT normalize (only explicit `T[,]` does),
# so these SB bounded-prior references use plain densities (sans-log2).
# BRM's explicit RK declarations normalize positive/truncated priors; a
# mapped comparison must account for their stated normalization constants.
#
# Mapping rule: unbounded location-scale priors map u=0 to constrained 0,
# NOT to the prior location (matters for the mixture means below).

@testset "robust edges HSGP k=1 validity floor" begin
    x6 = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    brmi = @brm (y=_Y2, x=x6) begin
        mu ~ 1 + hsgp(x; k=1)
        y ~ Normal(mu, sigma)
        sigma ~ Exponential(1)
    end
    sb = robust_build(brmi)
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "B-hsgp-1/k1")
        @test d == 5
        # k=1 floors rho_lower to exactly 0.0 (guarded).
        @test BRM.stan_data(sb)[:rho_lower_hsgp_x] == 0.0
        oracle = _LL2 + _STDN0 + logpdf(LogNormal(0, 1), 1.0) +
            logpdf(LogNormal(0, 1), 1.0) + _STDN0 + _EXP1
        @test lp ≈ oracle atol = 1e-9
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end

@testset "robust edges HSGP k=5 n=2" begin
    brmi = @brm (y=[0.5, -0.2], x=[0.5, -1.0]) begin
        mu ~ 1 + hsgp(x; k=5)
        y ~ Normal(mu, sigma)
        sigma ~ Exponential(1)
    end
    sb = robust_build(brmi)
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "B-hsgp-2/n2")
        @test d == 9
        rl = BRM.stan_data(sb)[:rho_lower_hsgp_x]
        ll = sum(logpdf.(Normal(0, 1), [0.5, -0.2]))
        oracle = ll + _STDN0 + logpdf(LogNormal(0, 1), rl + 1.0) +
            logpdf(LogNormal(0, 1), 1.0) + 5 * _STDN0 + _EXP1
        @test lp ≈ oracle atol = 1e-9
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end

@testset "robust edges Poisson all-zero and n=1" begin
    for (tag, y, x) in (("B-pois-1/allzero", [0, 0, 0, 0, 0, 0],
            [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]),
            ("B-pois-2/n1", [3], [1.0]))
        brmi = @brm (y=y, x=x) begin
            log(lambda) ~ 1 + x
            y ~ Poisson(lambda)
        end
        sb = robust_build(brmi)
        if RUN_BRIDGESTAN
            d, lp, g, problem = robust_value(sb, tag)
            @test d == 2
            # beta=0 -> lambda=1.
            @test lp ≈ sum(logpdf.(Poisson(1), y)) + 2 * _STDN0 atol = 1e-9
            @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
        end
    end
end

@testset "robust edges Binomial extreme-eta and single-trial" begin
    for (tag, b, n, x) in (
            ("B-binom-1/bigx", [0, 1, 0, 1, 0, 1], fill(10, 6),
                [1000.0, -1000.0, 500.0, -500.0, 0.0, 1.0]),
            ("B-binom-2/n1trials", [0, 1, 0, 1], fill(1, 4),
                [0.5, -1.0, 1.5, 0.0]))
        brmi = @brm (b=b, n=n, x=x) begin
            logit(p) ~ 1 + x
            b ~ Binomial(n, p)
        end
        sb = robust_build(brmi)
        if RUN_BRIDGESTAN
            d, lp, g, problem = robust_value(sb, tag)
            @test d == 2
            # beta=0 -> p=0.5.
            @test lp ≈ sum(logpdf.(Binomial.(n, 0.5), b)) + 2 * _STDN0 atol = 1e-9
            @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
        end
    end
end

@testset "robust edges single-component mixture K=1" begin
    y = [-2.0, -1.8, 1.9, 2.2]
    brmi = @brm (y=y,) begin
        mu1 ~ Normal(-2, 0.1)
        log(sigma) ~ 1
        y ~ MixtureModel([Normal(mu1, sigma)], [1.0])
    end
    sb = robust_build(brmi)
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "B-mix-1/K1")
        @test d == 2
        # mu1@u=0 is 0 (identity map), prior N(-2,0.1) evaluated at 0.
        oracle = sum(logpdf.(Normal(0, 1), y)) +
            logpdf(Normal(-2, 0.1), 0.0) + _STDN0
        @test lp ≈ oracle atol = 1e-9
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end

@testset "robust edges mixture n=2" begin
    y = [-2.0, 2.2]
    brmi = @brm (y=y,) begin
        mu1 ~ Normal(-2, 0.1)
        mu2 ~ Normal(2, 0.1)
        log(sigma) ~ 1
        y ~ MixtureModel([Normal(mu1, sigma), Normal(mu2, sigma)], [0.4, 0.6])
    end
    sb = robust_build(brmi)
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "B-mix-2/n2")
        @test d == 3
        ll = sum(yi -> log(0.4 * pdf(Normal(0, 1), yi) +
                           0.6 * pdf(Normal(0, 1), yi)), y)
        oracle = ll + logpdf(Normal(-2, 0.1), 0.0) +
            logpdf(Normal(2, 0.1), 0.0) + _STDN0
        @test lp ≈ oracle atol = 1e-9
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end

@testset "robust edges single-slope horseshoe" begin
    x6 = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    brmi = @brm (y=_Y2, x=x6) begin
        mu ~ 0 + x
        effect(mu, x) ~ Horseshoe()
        y ~ Normal(mu, sigma)
        sigma ~ Exponential(1)
    end
    sb = robust_build(brmi)
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "B-hs-1/p1")
        @test d == 4
        # beta = raw*lambda*tau = 0; half-Cauchy terms are PLAIN (see the
        # truncation-convention note above).
        oracle = _LL2 + _STDN0 + 2 * logpdf(Cauchy(0, 1), 1.0) + _EXP1
        @test lp ≈ oracle atol = 1e-9
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end

@testset "robust edges Poisson-GLMM single group" begin
    y = [2, 0, 1, 3]
    brmi = @brm (y=y, g=[1, 1, 1, 1]) begin
        log(lambda) ~ 1 + (1 | g)
        y ~ Poisson(lambda)
    end
    sb = robust_build(brmi)
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "B-poisg-1/G1")
        @test d == 2
        # totals on the log scale: lambda=1, collapsed T ~ N(0,2).
        oracle = sum(logpdf.(Poisson(1), y)) +
            logpdf(Normal(0, sqrt(2)), 0.0) + logpdf(LogNormal(0, 1), 1.0)
        @test lp ≈ oracle atol = 1e-9
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end

# KernelPlate edges: degenerate (nsub=1 / T=1) variants of pair 3's
# joint-2a/2b fixtures (Poisson + Bernoulli plates, nsub=2/T=3 ragged).
# Same cell spellings (undotted calls, dotted operators); u=[0] (b0=0).

@testset "robust edges Poisson plate degenerate shapes" begin
    for (tag, t, dose, obs, ys) in (
            ("P-2a-nsub1", [[0.5, 1.0, 2.0]], [1.0], [[1, 0, 2]], [1, 0, 2]),
            ("P-2a-T1", [[1.0], [2.0]], [1.0, 2.0], [[1], [3]], [1, 3]),
            ("P-2a-nsub1T1", [[1.0]], [1.0], [[2]], [2]))
        brmi = @brm (t=t, dose=dose, obs=obs) begin
            b0 ~ Normal(0, 1)
            pred ~ kernel(t, dose, obs) do ts, dd, yy
                mu = exp(b0 * dd * ts)
                yy ~ poisson(mu)
                mu
            end
        end
        sb = robust_build(brmi)
        if RUN_BRIDGESTAN
            d, lp, g, problem = robust_value(sb, tag)
            @test d == 1
            # b0=0 -> mu=1.
            @test lp ≈ sum(logpdf.(Poisson(1), ys)) + _STDN0 atol = 1e-9
            @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
        end
    end
end

@testset "robust edges Bernoulli plate degenerate shapes" begin
    for (tag, t, dose, obs, n) in (
            ("P-2b-nsub1", [[0.5, 1.0, 2.0]], [1.0], [[1, 0, 1]], 3),
            ("P-2b-T1", [[1.0], [2.0]], [1.0, 2.0], [[1], [0]], 2),
            ("P-2b-nsub1T1", [[2.0]], [2.0], [[0]], 1))
        brmi = @brm (t=t, dose=dose, obs=obs) begin
            b0 ~ Normal(0, 1)
            pred ~ kernel(t, dose, obs) do ts, dd, yy
                eta = b0 * dd * ts
                p = 1.0 ./ (1.0 .+ exp(-eta))
                yy ~ bernoulli(p)
                p
            end
        end
        sb = robust_build(brmi)
        if RUN_BRIDGESTAN
            d, lp, g, problem = robust_value(sb, tag)
            @test d == 1
            # b0=0 -> p=0.5.
            @test lp ≈ n * log(0.5) + _STDN0 atol = 1e-9
            @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
        end
    end
end
