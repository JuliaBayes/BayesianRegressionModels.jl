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
using Distributions: Exponential, LogNormal, MvNormal, Normal, logpdf
using LinearAlgebra
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
    brmi = @brm (; t=[[1.0]], dose=[100.0], dv=[[1.0]],
        CL=[5.0], Vc=[50.0], Ka=[1.0]) begin
        sigma ~ Exponential(1)
        pred ~ kernel(t, dose, dv, CL, Vc, Ka) do ts, d, yy, CLi, Vci, Kai
            ke = CLi / Vci
            mu = d * Kai / (Vci * (Kai - ke)) * (exp(-ke * ts) - exp(-Kai * ts))
            yy ~ normal(mu, sigma)
            mu
        end
    end
    sb = robust_build(brmi)
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "E-k1/nsub1-T1")
        @test d == 1
        mu_k1 = 100.0 * 1.0 / (50.0 * (1.0 - 0.1)) * (exp(-0.1) - exp(-1.0))
        oracle = logpdf(Normal(mu_k1, 1.0), 1.0) + _EXP1
        @test lp ≈ oracle atol = 1e-9
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end

@testset "robust edges kernel plate nsub=1 all-scalar" begin
    brmi = @brm (dose=[100.0], dv=[0.5], ls=[0.1]) begin
        sigma ~ Exponential(1)
        pred ~ kernel(dose, dv, ls) do dd, yy, lsi
            mu = (dd / 10.0) * exp(lsi)
            yy ~ normal(mu, sigma)
            mu
        end
    end
    sb = robust_build(brmi)
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "E-k2/nsub1")
        @test d == 1
        oracle = logpdf(Normal(10.0 * exp(0.1), 1.0), 0.5) + _EXP1
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

# P2 kernel oracles re-verified on the SB side (baseline for KernelPlate
# re-verification). The SB cell keeps its undotted SLIC spelling
# (`normal`, scalar `exp`) per decision 0qf6vi2 — the dotted neutral body
# is RK-only. Convention: value at constrained sigma=1.0 (u=[0.0])
# equals the oracle; the unconstrained gradient equals the constrained
# oracle gradient + 1 (exp-Jacobian).
const _PK1CMT_COLS = (;
    t=[[0.5, 1.0, 2.0, 4.0] for _ in 1:3],
    dose=fill(100.0, 3),
    dv=[[1.0, 2.0, 1.5, 0.8] for _ in 1:3],
    CL=[5.0, 6.0, 4.5],
    Vc=[50.0, 55.0, 48.0],
    Ka=[1.0, 1.2, 0.9],
)
const _DOSEPLATE_COLS = (;
    dose=fill(100.0, 4),
    dv=[0.5, 1.2, 2.1, 3.3],
    ls=[0.1, 0.2, 0.15, 0.25],
)

@testset "robust baseline SB Ex1 pk1cmt oracle" begin
    brmi = @brm _PK1CMT_COLS begin
        sigma ~ Exponential(1)
        pred ~ kernel(t, dose, dv, CL, Vc, Ka) do ts, d, yy, CLi, Vci, Kai
            ke = CLi / Vci
            mu = d * Kai / (Vci * (Kai - ke)) * (exp(-ke * ts) - exp(-Kai * ts))
            yy ~ normal(mu, sigma)
            mu
        end
    end
    sb = robust_build(brmi)
    # Independent Bateman hand oracle at sigma=1.0 (constrained d/dσ).
    ll, sq, n = 0.0, 0.0, 0
    for s in 1:3
        CL, Vc, Ka, d =
            _PK1CMT_COLS.CL[s], _PK1CMT_COLS.Vc[s], _PK1CMT_COLS.Ka[s],
            _PK1CMT_COLS.dose[s]
        ke = CL / Vc
        for (t, y) in zip(_PK1CMT_COLS.t[s], _PK1CMT_COLS.dv[s])
            mu = d * Ka / (Vc * (Ka - ke)) * (exp(-ke * t) - exp(-Ka * t))
            ll += logpdf(Normal(mu, 1.0), y)
            sq += abs2(y - mu)
            n += 1
        end
    end
    hand_v, hand_g = ll + _EXP1, sq - n - 1.0
    @test hand_v ≈ -13.703526816545866 atol = 1e-9
    @test hand_g ≈ -9.647471163820416 atol = 1e-8
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "SB-Ex1/pk1cmt")
        @test d == 1
        @test lp ≈ -13.703526816545866 atol = 1e-9
        @test g[1] ≈ -9.647471163820416 + 1.0 atol = 1e-8
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end

@testset "robust baseline SB Ex2 doseplate oracle" begin
    brmi = @brm _DOSEPLATE_COLS begin
        sigma ~ Exponential(1)
        pred ~ kernel(dose, dv, ls) do dd, yy, lsi
            mu = (dd / 10.0) * exp(lsi)
            yy ~ normal(mu, sigma)
            mu
        end
    end
    sb = robust_build(brmi)
    ll, sq, n = 0.0, 0.0, 0
    for i in eachindex(_DOSEPLATE_COLS.dv)
        mu = (_DOSEPLATE_COLS.dose[i] / 10.0) * exp(_DOSEPLATE_COLS.ls[i])
        y = _DOSEPLATE_COLS.dv[i]
        ll += logpdf(Normal(mu, 1.0), y)
        sq += abs2(y - mu)
        n += 1
    end
    hand_v, hand_g = ll + _EXP1, sq - n - 1.0
    @test hand_v ≈ -211.80708530040758 atol = 1e-9
    @test hand_g ≈ 409.2626623351777 atol = 1e-8
    if RUN_BRIDGESTAN
        d, lp, g, problem = robust_value(sb, "SB-Ex2/doseplate")
        @test d == 1
        @test lp ≈ -211.80708530040758 atol = 1e-9
        @test g[1] ≈ 409.2626623351777 + 1.0 atol = 1e-8
        @test g ≈ robust_findiff(problem, d) rtol = 1e-5 atol = 1e-7
    end
end
