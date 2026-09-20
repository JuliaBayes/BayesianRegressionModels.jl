# test/rk_parity.jl — BRM→RK end-to-end parity (ranef + P2 kernel +
# ordinal-extras models).
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
# - Correlated outcomes: the K=2 LKJ(eta) closed form
#   `-logbeta(1/2,eta) + 2(eta-1)log L22` is re-derived from the LKJ
#   definition (not imported); the per-row MvNormal ref is Stan
#   `multi_normal_cholesky_lpdf` verbatim, row constant included.
# - Joint anchors: the K=2 case pins the Stage-C joint values
#   (likelihood bit-exact, prior 1 ulp in the live exchange).
#
# Requires the ReactiveKernels bootstrap pin (test/setup_env.jl); the
# plan/AST halves stay dependency-free in test/rk_emitter.jl and
# test/rk_ast.jl.

using Test
using BayesianRegressionModels
using CategoricalArrays: categorical, levelcode
using DifferentiationInterface: AutoEnzyme
using Distributions: Beta, Dirichlet, Exponential, Normal, logcdf, logccdf,
                     logpdf
using Enzyme
using LogDensityProblems
using ReactiveKernels: prepare
using ReactiveKernelsPPL: constrain, logjac
using SpecialFunctions: logbeta, loggamma

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
    # theta = pi*sigmoid(t); the (i-j) = 1 Gram exponent keeps one log-sin
    # term (Omega volume element — peer fix a5b810a, RK 86e5265; the old
    # (i-1-j) = 0 exponent dropped it, pinning the buggy Jacobian).
    s = 1 / (1 + exp(-t))
    return log(sin(pi * s)) + log(pi) + log(s) + log1p(-s)
end

function _layout_signature(layout)
    return [(e.kind, e.name, e.size, e.transform) for e in layout.entries]
end

# SB `ar1_recurse` verbatim (`u[1] = eps[1]`,
# `u[t] = phi*u[t-1] + eps[t]`), over the constrained innovations.
function _ref_ar1_path(phi, eps)
    u = Vector{Float64}(undef, length(eps))
    u[1] = eps[1]
    for t in 2:length(eps)
        u[t] = phi * u[t-1] + eps[t]
    end
    return u
end

_parity_cols = (;
    g = [1, 2, 1, 3, 2, 3],
    x = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
    y = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
)
_parity_cols_multi = merge(_parity_cols,
    (; y2 = [0.5, 1.5, 1.0, 2.0, 2.5, 1.5]))
_parity_cols_dummy = merge(_parity_cols, (; c = [1, 2, 2, 1, 2, 1]))
_parity_cols_xz = merge(_parity_cols, (; z = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]))
_parity_cols_mo = (; c = [1, 2, 3, 1, 2, 3], y = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0])
_parity_cols_r2d2 = merge(_parity_cols,
    (; z = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
_parity_cols_corr = (;
    y1 = [0.5, -0.2, 0.1, 0.9, 1.4, 1.1],
    y2 = [0.1, 0.3, -0.4, 0.2, 0.8, -0.1],
    y3 = [-0.3, 0.7, 0.2, -0.1, 0.4, 0.6],
    x = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
)

# Sample variance, N−1 normalization (Stan `variance()`); dummy
# variance without materializing the dummy (SB `brm_cat_variances`).
function _ref_sample_variance(col)
    n = length(col)
    m = sum(col) / n
    return sum((x - m)^2 for x in col) / (n - 1)
end
function _ref_dummy_variance(col, lvl)
    n = length(col)
    m = count(==(lvl), col)
    return m * (n - m) / (n * (n - 1))
end

# Default-ordered categorical grouping: `categorical` sorts levels, so
# `CA.levels` order == the thin layer's bind-derived sort order (P1).
_parity_cols_cat = (;
    g = categorical(["a", "b", "a", "c", "b", "c"]),
    x = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
    y = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
)

# SB `_sb_mo` contrast `cumsum([0; incr])[idx]`, explicit loop (never the
# thin-layer gather recipe).
function _ref_mo_contrast(incr, idx)
    K = length(incr) + 1
    cum = zeros(Float64, K)
    for j in 2:K
        cum[j] = cum[j - 1] + incr[j - 1]
    end
    return [cum[i] for i in idx]
end

# Stick-breaking log-Jacobian over the packed increments (the
# thin-layer-owned parameterization, NOT Stan's ILR — `Σ [log(r) +
# log(z) + log1p(-z)]` with `z[j] = σ(u[j] + log(d-j))`, per the layout
# docs; re-derived here, not imported).
function _ref_simplex_logjac(u)
    d = length(u) + 1
    jac = 0.0
    remaining = 1.0
    for j in 1:(d - 1)
        z = 1 / (1 + exp(-(u[j] + log(d - j))))
        jac += log(remaining) + log(z) + log1p(-z)
        remaining *= 1 - z
    end
    return jac
end

# Per-row MvNormalCholesky log-density by forward substitution, full
# normalizer (Stan `multi_normal_cholesky_lpdf` form). RK keeps the row
# constant Stan's model-block `~` drops for data hyperparameters — the
# mo Dirichlet-normalizer precedent: the RK posterior exceeds Stan's by
# exactly that constant while every gradient agrees.
function _ref_mvn_chol_row(y, m, L)
    K = length(y)
    z = zeros(Float64, K)
    for i in 1:K
        acc = y[i] - m[i]
        for j in 1:(i - 1)
            acc -= L[i, j] * z[j]
        end
        z[i] = acc / L[i, i]
    end
    return -0.5 * K * log(2pi) - sum(log(abs(L[i, i])) for i in 1:K) -
           0.5 * sum(abs2, z)
end

# K=2 LKJ(eta) from the definition: L = [1 0; rho sqrt(1-rho^2)]
# with a unit-Jacobian free element rho, so log p =
# -log B(1/2,eta) + (eta-1)*log(1-rho^2) =
# -log B(1/2,eta) + 2*(eta-1)*log(L[2,2]). Derived here, not
# imported from the PPL.
_ref_lkj_k2(eta, L) = -logbeta(0.5, eta) + 2 * (eta - 1) * log(L[2, 2])

@testset "rk parity mo monotonic" begin
    brmi = @brm _parity_cols_mo begin
        mu ~ 1 + mo(c)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 4
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 2, :identity),
        (:sampled, :s, 1, :exp),
        (:vector, :mo_c_simplex_incr, 1, :simplex),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    contrast = _ref_mo_contrast(nt.mo_c_simplex_incr, _parity_cols_mo.c)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ nt.mu[2] .* contrast, nt.s),
        _parity_cols_mo.y))
    # Full Dirichlet logpdf (normalizer included): the thin layer keeps
    # the log-multivariate-Beta constant Stan drops for data alpha, so the
    # RK posterior exceeds Stan's by exactly that constant (peer-verified
    # core parity is modulo it) while every gradient agrees.
    pr = logpdf(Normal(0, 1), nt.mu[1]) +
        logpdf(Normal(0, 1), nt.mu[2]) +
        logpdf(Exponential(1), nt.s) +
        logpdf(Dirichlet(ones(2)), nt.mo_c_simplex_incr)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = u[3] + _ref_simplex_logjac(u[4:4])
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@testset "rk parity mo1 summand (override alpha)" begin
    brmi = @brm _parity_cols_mo begin
        mu ~ 1 + mo1(c)
        simplex(mu, mo1(c)) ~ Dirichlet(1, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 1, :identity),
        (:sampled, :s, 1, :exp),
        (:vector, :mo1_c_simplex_incr, 1, :simplex),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    contrast = _ref_mo_contrast(nt.mo1_c_simplex_incr, _parity_cols_mo.c)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ contrast, nt.s), _parity_cols_mo.y))
    pr = logpdf(Normal(0, 1), nt.mu[1]) +
        logpdf(Exponential(1), nt.s) +
        logpdf(Dirichlet([1.0, 2.0]), nt.mo1_c_simplex_incr)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = u[2] + _ref_simplex_logjac(u[3:3])
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@testset "rk parity correlated outcomes K=2" begin
    brmi = @brm _parity_cols_corr begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior = Exponential(1), shape = 2)
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 7
    @test _layout_signature(layout) == [
        (:coefficient, :mu1_coef, 2, :identity),
        (:coefficient, :mu2_coef, 2, :identity),
        (:vector, :L_res_scales, 2, :exp),
        (:cholesky_corr, :L_res_L_corr, 1, :lkj),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    cols = _parity_cols_corr
    m1 = nt.mu1[1] .+ nt.mu1[2] .* cols.x
    m2 = nt.mu2[1] .+ nt.mu2[2] .* cols.x
    L = [nt.L_res_scales[i] * nt.L_res_L_corr[i, j] for i in 1:2, j in 1:2]
    ll = sum(_ref_mvn_chol_row([cols.y1[r], cols.y2[r]], [m1[r], m2[r]], L)
        for r in 1:6)
    pr = sum(logpdf(Normal(0, 1), c) for c in (nt.mu1..., nt.mu2...)) +
        sum(logpdf(Exponential(1), s) for s in nt.L_res_scales) +
        _ref_lkj_k2(2.0, nt.L_res_L_corr)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = u[5] + u[6] + _lkj2_theta_jac(u[7])
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@testset "rk parity correlated outcomes K=3 sampled scale" begin
    brmi = @brm _parity_cols_corr begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        mu3 ~ 1 + x
        tau ~ Exponential(1)
        L3 ~ LKJCovarianceFactor(3; scale_prior = Exponential(tau), shape = 1.5)
        [y1, y2, y3] ~ MvNormalCholesky([mu1, mu2, mu3], L3)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 13
    @test _layout_signature(layout) == [
        (:coefficient, :mu1_coef, 2, :identity),
        (:coefficient, :mu2_coef, 2, :identity),
        (:coefficient, :mu3_coef, 2, :identity),
        (:sampled, :tau, 1, :exp),
        (:vector, :L3_scales, 3, :exp),
        (:cholesky_corr, :L3_L_corr, 3, :lkj),
    ]
    u = collect(range(-0.3, 0.3; length = layout.total))
    nt = constrain(layout, u)
    cols = _parity_cols_corr
    m = [nt.mu1[1] .+ nt.mu1[2] .* cols.x,
        nt.mu2[1] .+ nt.mu2[2] .* cols.x,
        nt.mu3[1] .+ nt.mu3[2] .* cols.x]
    L = [nt.L3_scales[i] * nt.L3_L_corr[i, j] for i in 1:3, j in 1:3]
    ll = sum(_ref_mvn_chol_row([cols.y1[r], cols.y2[r], cols.y3[r]],
            [m[1][r], m[2][r], m[3][r]], L) for r in 1:6)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    # The K=3 LKJ normalizer is peer-tested thin-side; the hand-checked
    # part here is the likelihood wiring plus the full posterior
    # gradient (same stem as the K=2 case above).
    @test isfinite(_rk_query(backend, :prior, u))
    _check_parity_gradient(backend, u)
end

@testset "rk parity r2d2 flat" begin
    brmi = @brm _parity_cols_r2d2 begin
        mu ~ 1 + x + z
        effect(mu, :) ~ r2d2(R2=Beta(2, 5), alpha=0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 7
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 3, :identity),
        (:sampled, :s, 1, :exp),
        (:sampled, :r2d2_mu_R2, 1, :logistic),
        (:sampled, :r2d2_mu_tau_bsv, 1, :exp),
        (:vector, :r2d2_mu_phi, 1, :simplex),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    R2, tau, phi = nt.r2d2_mu_R2, nt.r2d2_mu_tau_bsv, nt.r2d2_mu_phi
    cols = _parity_cols_r2d2
    sx = sqrt(phi[1] * R2 * tau^2 / _ref_sample_variance(cols.x))
    sz = sqrt(phi[2] * R2 * tau^2 / _ref_sample_variance(cols.z))
    mu_hat = nt.mu[1] .+ nt.mu[2] .* cols.x .+ nt.mu[3] .* cols.z
    ll = sum(logpdf.(Normal.(mu_hat, nt.s), cols.y))
    # The sampled tau carries the thin-layer `:positive` half
    # renormalizer (+log 2, peer-blessed in `test_r2d2.jl`) over SB's
    # Stan-convention unnormalized half-normal — a constant the RK
    # posterior exceeds SB's by, while every gradient agrees (the mo
    # Dirichlet-normalizer precedent).
    pr = logpdf(Normal(0, 1), nt.mu[1]) +
        logpdf(Normal(0, sx), nt.mu[2]) +
        logpdf(Normal(0, sz), nt.mu[3]) +
        logpdf(Exponential(1), nt.s) +
        logpdf(Beta(2, 5), R2) +
        logpdf(Normal(0, 1), tau) + log(2) +
        logpdf(Dirichlet([0.5, 0.5]), phi)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = u[4] + (log(R2) + log1p(-R2)) + u[6] + _ref_simplex_logjac(u[7:7])
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@testset "rk parity r2d2 override + tau literal" begin
    brmi = @brm _parity_cols_r2d2 begin
        mu ~ 1 + x + z
        effect(mu, :) ~ r2d2(tau_bsv=2.0)
        effect(mu, x) ~ Normal(0, 3)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 5
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 3, :identity),
        (:sampled, :s, 1, :exp),
        (:sampled, :r2d2_mu_R2, 1, :logistic),
        (:vector, :r2d2_mu_phi, 0, :simplex),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    R2, phi = nt.r2d2_mu_R2, nt.r2d2_mu_phi
    @test phi ≈ [1.0]
    cols = _parity_cols_r2d2
    sz = sqrt(phi[1] * R2 * 2.0^2 / _ref_sample_variance(cols.z))
    mu_hat = nt.mu[1] .+ nt.mu[2] .* cols.x .+ nt.mu[3] .* cols.z
    ll = sum(logpdf.(Normal.(mu_hat, nt.s), cols.y))
    pr = logpdf(Normal(0, 1), nt.mu[1]) +
        logpdf(Normal(0, 3), nt.mu[2]) +
        logpdf(Normal(0, sz), nt.mu[3]) +
        logpdf(Exponential(1), nt.s) +
        logpdf(Beta(1, 1), R2) +
        logpdf(Dirichlet([1.0]), phi)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = u[4] + (log(R2) + log1p(-R2))
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@testset "rk parity r2d2 factor join" begin
    brmi = @brm _parity_cols begin
        mu ~ 0 + g
        effect(mu, :) ~ r2d2()
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 8
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 3, :identity),
        (:sampled, :s, 1, :exp),
        (:sampled, :r2d2_mu_R2, 1, :logistic),
        (:sampled, :r2d2_mu_tau_bsv, 1, :exp),
        (:vector, :r2d2_mu_phi, 2, :simplex),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    R2, tau, phi = nt.r2d2_mu_R2, nt.r2d2_mu_tau_bsv, nt.r2d2_mu_phi
    cols = _parity_cols
    sc = [sqrt(phi[k] * R2 * tau^2 / _ref_dummy_variance(cols.g, k))
        for k in 1:3]
    mu_hat = nt.mu[_group_index(cols.g)]
    ll = sum(logpdf.(Normal.(mu_hat, nt.s), cols.y))
    pr = sum(logpdf(Normal(0, sc[k]), nt.mu[k]) for k in 1:3) +
        logpdf(Exponential(1), nt.s) +
        logpdf(Beta(1, 1), R2) +
        logpdf(Normal(0, 1), tau) + log(2) +
        logpdf(Dirichlet(ones(3)), phi)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = u[4] + (log(R2) + log1p(-R2)) + u[6] + _ref_simplex_logjac(u[7:8])
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

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

@testset "rk parity K=1 intercept categorical grouping" begin
    brmi = @brm _parity_cols_cat begin
        mu ~ 1 + (1 | g)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 6
    # SB numbering is CA.levels positions (levelcodes) — independent of
    # the thin layer's sorted-strings `_declared_codes` encoder.
    idx = levelcode.(_parity_cols_cat.g)
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    r = exp(nt.log_scale_g) .* nt.xi_g[idx]
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), _parity_cols_cat.y))
    pr = logpdf(Normal(0, 5), nt.mu[1]) +
        logpdf(Exponential(1), nt.sigma) +
        logpdf(Normal(0, 1), nt.log_scale_g) +
        sum(logpdf.(Normal(0, 1), nt.xi_g))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
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

@testset "rk parity ranef interaction" begin
    brmi = @brm _parity_cols_xz begin
        mu ~ 1 + (1 + x & z | g)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    # The cross margin gathers the in-graph derived product (SB: `x .* z`).
    bucket = only(BRM._brm_rk_plan(brmi).ranef_buckets)
    @test bucket.kind === :correlated
    @test [m.coefficient for m in bucket.margins] == [:Intercept, :int_x_x_z]
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
    r = _ref_corr_r(_parity_cols_xz.g, nt.L_g, nt.tau_g, nt.z_flat_g,
        [ones(6), _parity_cols_xz.x .* _parity_cols_xz.z], 1:2)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), _parity_cols_xz.y))
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

# P2 kernel(...) execution parity (peer KernelPlate reader, RK @ 86e5265).
# Ex1/Ex2 neutral translations from parent todo 14bv4nq; SB oracles
# re-verified on c13f41c (same numbers the peer pins in PPL test_kernel.jl).
# Convention: posterior at constrained sigma = 1.0 (u = [0.0]) == SB oracle
# (logjac(0) = 0); the unconstrained gradient == oracle constrained d/dσ + 1
# (exp-Jacobian). Kernel-built specs expose no direct :posterior prepare
# query, so the gradient cross-check findiffs the sampler value itself.
_kernel_pk1cmt_cols = (;
    t    = [[0.5, 1.0, 2.0, 4.0] for _ in 1:3],
    dose = fill(100.0, 3),
    dv   = [[1.0, 2.0, 1.5, 0.8] for _ in 1:3],
    CL   = [5.0, 6.0, 4.5],
    Vc   = [50.0, 55.0, 48.0],
    Ka   = [1.0, 1.2, 0.9],
)
_kernel_doseplate_cols = (;
    dose = fill(100.0, 4),
    dv   = [0.5, 1.2, 2.1, 3.3],
    ls   = [0.1, 0.2, 0.15, 0.25],
)

function _check_kernel_parity(backend::BRM.RKBRMI, u, val_oracle, grad_oracle;
        grad_atol = 1e-8)
    problem = BRM.rk_logdensity_problem(backend;
        ad_backend = _PARITY_BACKEND, u0 = u)
    @test LogDensityProblems.dimension(problem) == length(u)
    value, grad = LogDensityProblems.logdensity_and_gradient(problem, u)
    @test value ≈ val_oracle atol = 1e-9
    @test grad[1] ≈ grad_oracle + 1.0 atol = grad_atol
    @test all(isfinite, grad)
    @test grad ≈ _findiff_grad(
        w -> LogDensityProblems.logdensity(problem, w), u) rtol = 1e-5 atol = 1e-7
    return value
end

@testset "rk parity ar(1) latent path" begin
    ar_cols = (;
        t=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        y=[0.5, -0.2, 0.1, 0.9, 1.4, 1.1],
    )
    brmi = @brm ar_cols begin
        mu ~ 1 + ar(t; p=1)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 10
    # Submodel expansion inlines the popefs body at the call site, which
    # follows the top-level preamble — so the preamble's phi_raw precedes
    # the expanded beta in sampled order.
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 1, :identity),
        (:sampled, :phi_raw_ar_mu_t, 1, :identity),
        (:sampled, :mu_b2, 1, :identity),
        (:sampled, :sigma, 1, :exp),
        (:scan, :_ppl_scan_z_ar_mu_t, 6, :identity),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    phi = tanh(nt.phi_raw_ar_mu_t)
    path = _ref_ar1_path(phi, nt._ppl_scan_z_ar_mu_t)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ nt.mu_b2 .* path, nt.sigma),
        ar_cols.y))
    pr = logpdf(Normal(0, 5), nt.mu[1]) +
        logpdf(Normal(0, 1), nt.mu_b2) +
        logpdf(Normal(0, 1), nt.phi_raw_ar_mu_t) +
        logpdf(Exponential(1), nt.sigma) +
        sum(logpdf.(Normal(0, 1), nt._ppl_scan_z_ar_mu_t))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # Jacobian: sigma's exp only (betas/innovations ride identity).
    @test logjac(layout, u) ≈ u[4]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[4]
    _check_parity_gradient(backend, u)
end

# `me(x, sd)` measurement-error parity (thin-layer PlateParameter
# surface, re-cut 8a6c36c, RK @ d5e8bed). SB shape per `_sb_me`
# (src/sbimpl.jl): a length-N latent true covariate with the
# `latent(...)` Normal prior, the LP riding it through popefs's free
# beta, and the self-contained observation likelihood
# `x_obs ~ normal(x_true, sd)`. The synthetic observation response
# lowers with a width-0 `x_loc_coef` block (no free location
# coefficient — the plate IS the mean); both betas stay in `mu_coef`.
@testset "rk parity me(x, sd) latent" begin
    me_cols = (;
        x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        y=[1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
    )
    brmi = @brm me_cols begin
        mu ~ 1 + me(x, 0.5)
        latent(mu, me(x)) ~ Normal(0.5, 1.5)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 9
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 2, :identity),
        (:coefficient, :x_loc_coef, 0, :identity),
        (:sampled, :sigma, 1, :exp),
        (:plate, :me_x, 6, :identity),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    b = Vector(nt.mu)
    xt = Vector(nt.me_x)
    @test isempty(nt.x_loc)
    lp = b[1] .+ b[2] .* xt
    ll = sum(logpdf.(Normal.(lp, nt.sigma), me_cols.y)) +
        sum(logpdf.(Normal.(xt, 0.5), me_cols.x))
    pr = logpdf(Normal(0, 1), b[1]) +
        logpdf(Normal(0, 1), b[2]) +
        logpdf(Exponential(1), nt.sigma) +
        sum(logpdf.(Normal(0.5, 1.5), xt))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # Jacobian: sigma's exp only (betas ride identity, the plate
    # carries its Normal args directly with no transform).
    @test logjac(layout, u) ≈ u[3]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[3]
    _check_parity_gradient(backend, u)
end

@testset "rk parity kernel Ex1 pk1cmt" begin
    brmi = @brm _kernel_pk1cmt_cols begin
        sigma ~ Exponential(1)
        pred ~ kernel(t, dose, dv, CL, Vc, Ka) do ts, d, yy, CLi, Vci, Kai
            ke = CLi / Vci
            mu = d * Kai / (Vci * (Kai - ke)) .* (exp.(-ke .* ts) .- exp.(-Kai .* ts))
            yy ~ Normal(mu, sigma)
            mu
        end
    end
    plan = BRM._brm_rk_plan(brmi)
    @test plan isa BRM._RKKernelPlan
    @test plan.kernel.n_subjects == 3
    @test plan.kernel.data_columns == [:t, :dose, :dv, :CL, :Vc, :Ka]
    @test plan.kernel.slice_kinds == [:vector, :scalar, :vector, :scalar, :scalar, :scalar]
    @test plan.kernel.n_timepoints == 4
    @test plan.obs.family === :gaussian
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 1
    @test _layout_signature(layout) == [(:sampled, :sigma, 1, :exp)]
    _check_kernel_parity(backend, [0.0], -13.703526816545866, -9.647471163820416)
end

@testset "rk parity kernel Ex2 doseplate" begin
    brmi = @brm _kernel_doseplate_cols begin
        sigma ~ Exponential(1)
        pred ~ kernel(dose, dv, ls) do dd, yy, lsi
            mu = (dd ./ 10.0) .* exp.(lsi)
            yy ~ Normal(mu, sigma)
            mu
        end
    end
    plan = BRM._brm_rk_plan(brmi)
    @test plan isa BRM._RKKernelPlan
    @test plan.kernel.n_subjects == 4
    @test plan.kernel.data_columns == [:dose, :dv, :ls]
    @test plan.kernel.slice_kinds == [:scalar, :scalar, :scalar]
    @test plan.kernel.n_timepoints === nothing
    @test plan.obs.family === :gaussian
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 1
    @test _layout_signature(layout) == [(:sampled, :sigma, 1, :exp)]
    _check_kernel_parity(backend, [0.0], -211.80708530040758, 409.2626623351777;
        grad_atol = 1e-7)
end

# Ordinal discrimination / per_threshold parity (thin-layer 37a1213
# surface, RK @ 86e5265). The `_ref_ordinal` oracle re-derives SB's
# `brm_ordinal_*` Stan math (src/sbimpl.jl): cumulative cells take
# `F(d*(c-eta))` differences, stopping-ratio stages take
# `d*(c-eta-E)` with the stage effect inside the scaled argument, and a
# modeled scale is `exp` over its log-link predictor (verified against
# SBBRMI-emitted Stan: `vector disc = exp(log_disc)`).
_ord_cols = (;
    y=[1, 2, 3, 2, 1, 3, 2, 1, 3],
    x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0, -0.25, 0.75, -1.25],
    g=[1, 2, 3, 1, 2, 3, 1, 2, 3],
    z1=[0.11, 0.23, 0.37, 0.41, 0.53, 0.62, 0.71, 0.83, 0.97],
    z2=[0.91, 0.82, 0.73, 0.64, 0.55, 0.46, 0.37, 0.28, 0.19],
    d=[0.5, 1.0, 1.5, 2.0, 1.0, 0.8, 1.2, 0.9, 1.1],
)
_ord_cols1 = merge(_ord_cols, (; y=ones(Int, 9)))

_ref_log_inv_logit(z) = z >= 0 ? -log1p(exp(-z)) : z - log1p(exp(z))
function _ref_ord_logF(z, link)
    link === :logit && return _ref_log_inv_logit(z)
    link === :probit && return logcdf(Normal(), z)
    return log(-expm1(-exp(z)))
end
function _ref_ord_logCC(z, link)
    link === :logit && return _ref_log_inv_logit(-z)
    link === :probit && return logccdf(Normal(), z)
    return -exp(z)
end
_ref_log_diff_exp(a, b) = a + log1p(-exp(b - a))

# SB `brm_ordinal_lpmf` scalar mirror: `y` in 1..K, scalar `eta`, the K-1
# thresholds `t`, the positive scale `d`, and the K-1 stage effects `E`
# (stopping only; `nothing` without per_threshold).
function _ref_ordinal(y, eta, t, d, structure, link, E=nothing)
    K = length(t) + 1
    if structure === :cumulative
        y == 1 && return _ref_ord_logF(d * (t[1] - eta), link)
        y == K && return _ref_ord_logCC(d * (t[K-1] - eta), link)
        hi = _ref_ord_logF(d * (t[y] - eta), link)
        lo = _ref_ord_logF(d * (t[y-1] - eta), link)
        return _ref_log_diff_exp(hi, lo)
    else
        ll = 0.0
        for j in 1:K-1
            eff = E === nothing ? 0.0 : E[j]
            z = d * (t[j] - eta - eff)
            j < y && (ll += _ref_ord_logCC(z, link))
            j == y && (ll += _ref_ord_logF(z, link))
        end
        return ll
    end
end

@testset "rk parity ordinal cumulative literal scale" begin
    brmi = @brm _ord_cols begin
        eta ~ 0 + x
        y ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=2.0)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    @test _layout_signature(layout) == [
        (:coefficient, :eta_coef, 1, :identity),
        (:vector, :y_thresholds, 2, :ordered),
    ]
    u = [0.4, -0.2, 0.25]
    nt = constrain(layout, u)
    eta = only(Vector(nt.eta)) .* _ord_cols.x
    t = Vector(nt.y_thresholds)
    ll = sum(_ref_ordinal(y, e, t, 2.0, :cumulative, :logit)
        for (y, e) in zip(_ord_cols.y, eta))
    pr = logpdf(Normal(), only(nt.eta)) + sum(logpdf.(Normal(), t))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[3]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[3]
    _check_parity_gradient(backend, u)
end

@testset "rk parity ordinal cumulative modeled scale" begin
    brmi = @brm _ord_cols begin
        eta ~ 0 + x
        log(disc) ~ 1 + x
        y ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=disc)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 5
    @test _layout_signature(layout) == [
        (:coefficient, :eta_coef, 1, :identity),
        (:coefficient, :disc_coef, 2, :identity),
        (:vector, :y_thresholds, 2, :ordered),
    ]
    u = [0.4, 0.1, -0.2, 0.25, -0.15]
    nt = constrain(layout, u)
    eta = only(Vector(nt.eta)) .* _ord_cols.x
    a = Vector(nt.disc)
    d = exp.(a[1] .+ a[2] .* _ord_cols.x)
    t = Vector(nt.y_thresholds)
    ll = sum(_ref_ordinal(y, e, t, di, :cumulative, :logit)
        for (y, e, di) in zip(_ord_cols.y, eta, d))
    pr = logpdf(Normal(), only(nt.eta)) + sum(logpdf.(Normal(), a)) +
        sum(logpdf.(Normal(), t))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[5]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[5]
    _check_parity_gradient(backend, u)
end

@testset "rk parity ordinal cumulative grouping scale" begin
    brmi = @brm _ord_cols begin
        eta ~ 0 + x
        log(disc) ~ 0 + g
        effect(disc, g) ~ Normal(0, 1)
        y ~ Ordinal(Cumulative(), CloglogLink(), eta; discrimination=disc)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 6
    @test _layout_signature(layout) == [
        (:coefficient, :eta_coef, 1, :identity),
        (:coefficient, :disc_coef, 3, :identity),
        (:vector, :y_thresholds, 2, :ordered),
    ]
    u = [0.4, 0.1, -0.2, 0.3, 0.25, -0.15]
    nt = constrain(layout, u)
    eta = only(Vector(nt.eta)) .* _ord_cols.x
    levs = sort(unique(_ord_cols.g))
    C = Float64.(_ord_cols.g .== permutedims(levs))
    d = exp.(C * Vector(nt.disc))
    t = Vector(nt.y_thresholds)
    ll = sum(_ref_ordinal(y, e, t, di, :cumulative, :cloglog)
        for (y, e, di) in zip(_ord_cols.y, eta, d))
    pr = logpdf(Normal(), only(nt.eta)) +
        sum(logpdf.(Normal(), nt.disc)) + sum(logpdf.(Normal(), t))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[6]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[6]
    _check_parity_gradient(backend, u)
end

@testset "rk parity ordinal cumulative column scale" begin
    brmi = @brm _ord_cols begin
        eta ~ 0 + x
        y ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=d)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    @test _layout_signature(layout) == [
        (:coefficient, :eta_coef, 1, :identity),
        (:vector, :y_thresholds, 2, :ordered),
    ]
    u = [0.4, -0.2, 0.25]
    nt = constrain(layout, u)
    eta = only(Vector(nt.eta)) .* _ord_cols.x
    t = Vector(nt.y_thresholds)
    ll = sum(_ref_ordinal(y, e, t, di, :cumulative, :logit)
        for (y, e, di) in zip(_ord_cols.y, eta, _ord_cols.d))
    pr = logpdf(Normal(), only(nt.eta)) + sum(logpdf.(Normal(), t))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[3]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[3]
    _check_parity_gradient(backend, u)
end

@testset "rk parity ordinal stopping per_threshold p=1" begin
    brmi = @brm _ord_cols begin
        eta ~ 0 + x
        y ~ Ordinal(StoppingRatio(), LogitLink(), eta; per_threshold=(z1,))
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 5
    @test _layout_signature(layout) == [
        (:coefficient, :eta_coef, 1, :identity),
        (:vector, :y_thresholds, 2, :identity),
        (:vector, :y_threshold_beta, 2, :identity),
    ]
    u = [0.4, 0.1, -0.2, 0.25, -0.15]
    nt = constrain(layout, u)
    eta = only(Vector(nt.eta)) .* _ord_cols.x
    t = Vector(nt.y_thresholds)
    beta = Vector(nt.y_threshold_beta)
    E = [_ord_cols.z1[i] * beta[j] for i in 1:9, j in 1:2]
    ll = sum(_ref_ordinal(y, e, t, 1.0, :stopping, :logit, E[i, :])
        for (i, (y, e)) in enumerate(zip(_ord_cols.y, eta)))
    pr = logpdf(Normal(), only(nt.eta)) + sum(logpdf.(Normal(), t)) +
        sum(logpdf.(Normal(), beta))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@testset "rk parity ordinal stopping per_threshold p=2" begin
    brmi = @brm _ord_cols begin
        eta ~ 0 + x
        y ~ Ordinal(StoppingRatio(), ProbitLink(), eta;
            per_threshold=(z1, z2))
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 7
    @test _layout_signature(layout) == [
        (:coefficient, :eta_coef, 1, :identity),
        (:vector, :y_thresholds, 2, :identity),
        (:vector, :y_threshold_beta, 4, :identity),
    ]
    u = [0.4, 0.1, -0.2, 0.25, -0.15, 0.05, -0.1]
    nt = constrain(layout, u)
    eta = only(Vector(nt.eta)) .* _ord_cols.x
    t = Vector(nt.y_thresholds)
    beta = Vector(nt.y_threshold_beta)
    X = hcat(_ord_cols.z1, _ord_cols.z2)
    E = [sum(X[i, c] * beta[(j-1)*2+c] for c in 1:2)
        for i in 1:9, j in 1:2]
    ll = sum(_ref_ordinal(y, e, t, 1.0, :stopping, :probit, E[i, :])
        for (i, (y, e)) in enumerate(zip(_ord_cols.y, eta)))
    pr = logpdf(Normal(), only(nt.eta)) + sum(logpdf.(Normal(), t)) +
        sum(logpdf.(Normal(), beta))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@testset "rk parity ordinal stopping scale plus stage" begin
    brmi = @brm _ord_cols begin
        eta ~ 0 + x
        log(disc) ~ 1 + x
        y ~ Ordinal(StoppingRatio(), LogitLink(), eta;
            discrimination=disc, per_threshold=(z1,))
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 7
    @test _layout_signature(layout) == [
        (:coefficient, :eta_coef, 1, :identity),
        (:coefficient, :disc_coef, 2, :identity),
        (:vector, :y_thresholds, 2, :identity),
        (:vector, :y_threshold_beta, 2, :identity),
    ]
    u = [0.4, 0.1, -0.2, 0.25, -0.15, 0.05, -0.1]
    nt = constrain(layout, u)
    eta = only(Vector(nt.eta)) .* _ord_cols.x
    a = Vector(nt.disc)
    d = exp.(a[1] .+ a[2] .* _ord_cols.x)
    t = Vector(nt.y_thresholds)
    beta = Vector(nt.y_threshold_beta)
    E = [_ord_cols.z1[i] * beta[j] for i in 1:9, j in 1:2]
    ll = sum(_ref_ordinal(_ord_cols.y[i], eta[i], t, d[i], :stopping,
        :logit, E[i, :]) for i in 1:9)
    pr = logpdf(Normal(), only(nt.eta)) + sum(logpdf.(Normal(), a)) +
        sum(logpdf.(Normal(), t)) + sum(logpdf.(Normal(), beta))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@testset "rk parity ordinal K=1 modeled scale" begin
    brmi = @brm _ord_cols1 begin
        eta ~ 0 + x
        log(disc) ~ 1 + x
        y ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=disc)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    @test _layout_signature(layout) == [
        (:coefficient, :eta_coef, 1, :identity),
        (:coefficient, :disc_coef, 2, :identity),
        (:vector, :y_thresholds, 0, :ordered),
    ]
    u = [0.4, 0.1, -0.2]
    nt = constrain(layout, u)
    # Zero-information likelihood (SB's K=1 degeneration), live prior.
    @test _rk_query(backend, :likelihood, u) == 0.0
    pr = logpdf(Normal(), only(nt.eta)) +
        sum(logpdf.(Normal(), nt.disc))
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ pr
    _check_parity_gradient(backend, u)
end

@testset "rk parity ordinal K=1 per_threshold" begin
    brmi = @brm _ord_cols1 begin
        eta ~ 0 + x
        y ~ Ordinal(StoppingRatio(), LogitLink(), eta; per_threshold=(z1,))
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 1
    @test _layout_signature(layout) == [
        (:coefficient, :eta_coef, 1, :identity),
        (:vector, :y_thresholds, 0, :identity),
        (:vector, :y_threshold_beta, 0, :identity),
    ]
    u = [0.4]
    nt = constrain(layout, u)
    # Zero stages: zero-information likelihood, eta prior only.
    @test _rk_query(backend, :likelihood, u) == 0.0
    pr = logpdf(Normal(), only(nt.eta))
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ pr
    _check_parity_gradient(backend, u)
end

