# test/rk_parity.jl — BRM→RK end-to-end parity on ranef models.
#
# Run: julia --project=test test/rk_parity.jl
#
# Each case routes an `@brm` model through the FULL build path
# (`_brm_rk_plan` → `_rk_emit_ast` → `lower_rkppl` → `bind_data` →
# `build_kernel`, i.e. `RKBRMI(brmi)`), then checks likelihood / prior /
# posterior values against independent SB-shape references and the Enzyme
# posterior gradient (via the `rk_logdensity_problem` shim) against
# central differences.
#
# Trust chain for the references (all joint-validated, none assumed):
# - SB emission: verified statement-by-statement from the SBBRMI-emitted
#   Stan dump (population intercept only, `lkj_corr_cholesky(1.0)`,
#   half-normal tau with no truncation renormalizer, column-major
#   `z_flat`, `b = (diag_pre_multiply(tau,L)*z)'`).
# - LKJ math: the thin-layer `lkj_corr_cholesky_logpdf` is Stan-verbatim
#   (peer-tested); the K=2 reference below re-derives eta=1 from the
#   Beta integral, independently of the emitter's LKJ09 port.
# - Joint anchors: the K=2 case pins the Stage-C joint values
#   (likelihood bit-exact, prior 1 ulp in the live exchange).
#
# Requires the ReactiveKernels bootstrap pin (test/setup_env.jl); the
# plan/AST halves stay dependency-free in test/rk_emitter.jl and
# test/rk_ast.jl.

using Test
using BayesianRegressionModels
using DifferentiationInterface: AutoEnzyme
using Distributions: Exponential, Normal, logpdf
using Enzyme
using LogDensityProblems
using ReactiveKernels: prepare
using ReactiveKernelsPPL: constrain, logjac
using SpecialFunctions: loggamma

const BRM = BayesianRegressionModels
const _PARITY_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

# A thin-layer value query against a built BRM backend: the bound data
# comes straight from the structural plan columns (same mapping the
# extension's `bind_data` route uses).
function _rk_query(backend::BRM.RKBRMI, want::Symbol, u)
    names = sort!(collect(keys(backend.plan.columns)))
    bound =
        NamedTuple{Tuple(names)}(Tuple(backend.plan.columns[k] for k in names))
    kern = prepare(backend.model.spec;
        have = (:unconstrained, names...), want = want, bound = bound)
    return kern(u)
end

function _findiff_grad(f, u; h = cbrt(eps(Float64)))
    g = similar(u, Float64)
    for i in eachindex(u)
        up = copy(u)
        up[i] += h
        dn = copy(u)
        dn[i] -= h
        g[i] = (f(up) - f(dn)) / (2h)
    end
    return g
end

# Posterior value through the sampler shim + its Enzyme gradient vs
# central differences of the direct posterior query.
function _check_parity_gradient(backend::BRM.RKBRMI, u)
    problem = BRM.rk_logdensity_problem(backend;
        ad_backend = _PARITY_BACKEND, u0 = u)
    @test LogDensityProblems.dimension(problem) == length(u)
    value, grad = LogDensityProblems.logdensity_and_gradient(problem, u)
    @test value ≈ _rk_query(backend, :posterior, u)
    @test all(isfinite, grad)
    @test grad ≈
        _findiff_grad(w -> _rk_query(backend, :posterior, w), u) rtol = 1e-5 atol = 1e-7
    return value
end

_group_index(gcol) = [findfirst(==(v), sort!(unique(gcol))) for v in gcol]

# SB `exp(log_scale) * xi[idx]` shape, explicit levels/order.
_ref_intercept_r(gcol, log_scale, xi) =
    exp(log_scale) .* xi[_group_index(gcol)]

# SB `tau * (xi[idx] .* Z)` association (`:column` and `:dummy` Z alike).
_ref_slope_r(gcol, tau, xi, Z) = tau .* (xi[_group_index(gcol)] .* Z)

# SB `(diag(tau)*L*z)'` shape with explicit per-margin/per-group loops
# (never the fused form), over the global margin subset `js` with Z
# columns `Zs` (`Zs[j]` is the j-th GLOBAL margin's column).
function _ref_corr_r(gcol, L, tau, zflat, Zs, js)
    K = length(Zs)
    idx = _group_index(gcol)
    r = zeros(Float64, length(idx))
    for m in eachindex(idx)
        g = idx[m]
        for j in js
            acc = 0.0
            for s in 1:j
                acc += tau[j] * L[j, s] * zflat[s + (g - 1) * K]
            end
            r[m] += Zs[j][m] * acc
        end
    end
    return r
end

# K=2 LKJ at eta=1 from the Beta-integral closed form (independent of
# the LKJ09-theorem-5 `lkj_logconst` port the emitter inlines).
function _ref_lkj_k2_eta1(L)
    c = loggamma(1.5) - loggamma(1.0) - 0.5 * log(pi)
    return c # + (2*1-2) * log(L[2, 2]) == c; L kept for the call shape
end

# K=2 spherical-Cholesky theta Jacobian: theta = pi*sigmoid(t).
function _lkj2_theta_jac(t)
    s = 1 / (1 + exp(-t))
    return log(pi) + log(s) + log1p(-s)
end

function _layout_signature(layout)
    return [(e.kind, e.name, e.size, e.transform) for e in layout.entries]
end

_parity_cols = (;
    g = [1, 2, 1, 3, 2, 3],
    x = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
    y = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
)
_parity_cols_multi = merge(_parity_cols,
    (; y2 = [0.5, 1.5, 1.0, 2.0, 2.5, 1.5]))
_parity_cols_dummy = merge(_parity_cols, (; c = [1, 2, 2, 1, 2, 1]))

@testset "rk parity K=1 intercept" begin
    brmi = @brm _parity_cols begin
        mu ~ 1 + (1 | g)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 6
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 1, :identity),
        (:sampled, :sigma, 1, :exp),
        (:sampled, :log_scale_g, 1, :identity),
        (:ranef, :xi_g, 3, :identity),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    r = _ref_intercept_r(_parity_cols.g, nt.log_scale_g, nt.xi_g)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), _parity_cols.y))
    pr = logpdf(Normal(0, 5), nt.mu[1]) +
        logpdf(Exponential(1), nt.sigma) +
        logpdf(Normal(0, 1), nt.log_scale_g) +
        sum(logpdf.(Normal(0, 1), nt.xi_g))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # Jacobian: sigma's exp only (log_scale/xi ride identity).
    @test logjac(layout, u) ≈ u[2]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[2]
    _check_parity_gradient(backend, u)
end

@testset "rk parity K=1 slope" begin
    brmi = @brm _parity_cols begin
        mu ~ 1 + (0 + x | g)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 6
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 1, :identity),
        (:sampled, :sigma, 1, :exp),
        (:sampled, :tau_g, 1, :exp),
        (:ranef, :xi_g, 3, :identity),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    r = _ref_slope_r(_parity_cols.g, nt.tau_g[1], nt.xi_g, _parity_cols.x)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), _parity_cols.y))
    # No +log(2): SB Stan-convention tau (joint Stage-B note).
    pr = logpdf(Normal(0, 5), nt.mu[1]) +
        logpdf(Exponential(1), nt.sigma) +
        logpdf(Normal(0, 1), nt.tau_g[1]) +
        sum(logpdf.(Normal(0, 1), nt.xi_g))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[2] + u[3]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[2] + u[3]
    _check_parity_gradient(backend, u)
end

@testset "rk parity K=2 correlated (+ joint anchor)" begin
    brmi = @brm _parity_cols begin
        mu ~ 1 + (1 + x | g)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 11
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 1, :identity),
        (:sampled, :sigma, 1, :exp),
        (:ranef_corr, :L_g, 1, :lkj),
        (:ranef, :tau_g, 2, :exp),
        (:ranef, :z_flat_g, 6, :identity),
    ]
    # The joint Stage-C point: the peer built its constrained case from
    # exactly this u, so the live-exchange values pin this leg.
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    r = _ref_corr_r(_parity_cols.g, nt.L_g, nt.tau_g, nt.z_flat_g,
        [ones(6), _parity_cols.x], 1:2)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), _parity_cols.y))
    pr = logpdf(Normal(0, 5), nt.mu[1]) +
        logpdf(Exponential(1), nt.sigma) +
        _ref_lkj_k2_eta1(nt.L_g) +
        sum(logpdf.(Normal(0, 1), nt.tau_g)) +
        sum(logpdf.(Normal(0, 1), nt.z_flat_g))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test _rk_query(backend, :likelihood, u) ≈ -34.557661700821690 atol = 1e-12
    @test _rk_query(backend, :prior, u) ≈ -12.267527341929741 atol = 1e-12
    jac = u[2] + u[4] + u[5] + _lkj2_theta_jac(u[3])
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@testset "rk parity multislice ID" begin
    brmi = @brm _parity_cols_multi begin
        mu1 ~ 1 + (1 | ID | g)
        mu2 ~ 1 + (0 + x | ID | g)
        y ~ Normal(mu1, s)
        y2 ~ Normal(mu2, s)
        effect(mu1, Intercept) ~ Normal(0, 5)
        effect(mu2, Intercept) ~ Normal(0, 5)
        s ~ Exponential(1)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 12
    @test _layout_signature(layout) == [
        (:coefficient, :mu1_coef, 1, :identity),
        (:coefficient, :mu2_coef, 1, :identity),
        (:sampled, :s, 1, :exp),
        (:ranef_corr, :L_ID_g, 1, :lkj),
        (:ranef, :tau_ID_g, 2, :exp),
        (:ranef, :z_flat_ID_g, 6, :identity),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    Zs = [ones(6), _parity_cols_multi.x]
    r1 = _ref_corr_r(_parity_cols_multi.g, nt.L_ID_g, nt.tau_ID_g,
        nt.z_flat_ID_g, Zs, 1:1)
    r2 = _ref_corr_r(_parity_cols_multi.g, nt.L_ID_g, nt.tau_ID_g,
        nt.z_flat_ID_g, Zs, 2:2)
    ll = sum(logpdf.(Normal.(nt.mu1[1] .+ r1, nt.s), _parity_cols_multi.y)) +
        sum(logpdf.(Normal.(nt.mu2[1] .+ r2, nt.s), _parity_cols_multi.y2))
    pr = logpdf(Normal(0, 5), nt.mu1[1]) +
        logpdf(Normal(0, 5), nt.mu2[1]) +
        logpdf(Exponential(1), nt.s) +
        _ref_lkj_k2_eta1(nt.L_ID_g) +
        sum(logpdf.(Normal(0, 1), nt.tau_ID_g)) +
        sum(logpdf.(Normal(0, 1), nt.z_flat_ID_g))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = u[3] + u[5] + u[6] + _lkj2_theta_jac(u[4])
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@testset "rk parity treatment-dummy correlated" begin
    brmi = @brm _parity_cols_dummy begin
        mu ~ 1 + (1 + c | g)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    # Two observed levels → intercept + one treatment dummy (SB rule).
    bucket = only(BRM._brm_rk_plan(brmi).ranef_buckets)
    @test bucket.kind === :correlated
    @test [m.coefficient for m in bucket.margins] == [:Intercept, :c_dummy_2]
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 11
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 1, :identity),
        (:sampled, :sigma, 1, :exp),
        (:ranef_corr, :L_g, 1, :lkj),
        (:ranef, :tau_g, 2, :exp),
        (:ranef, :z_flat_g, 6, :identity),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    Z2 = Float64.([v == 2 for v in _parity_cols_dummy.c])
    r = _ref_corr_r(_parity_cols_dummy.g, nt.L_g, nt.tau_g, nt.z_flat_g,
        [ones(6), Z2], 1:2)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), _parity_cols_dummy.y))
    pr = logpdf(Normal(0, 5), nt.mu[1]) +
        logpdf(Exponential(1), nt.sigma) +
        _ref_lkj_k2_eta1(nt.L_g) +
        sum(logpdf.(Normal(0, 1), nt.tau_g)) +
        sum(logpdf.(Normal(0, 1), nt.z_flat_g))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = u[2] + u[4] + u[5] + _lkj2_theta_jac(u[3])
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

