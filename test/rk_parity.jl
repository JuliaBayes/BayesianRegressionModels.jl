# test/rk_parity.jl — BRM→RK end-to-end parity (ranef + P2 kernel +
# ordinal-extras + hsgp models).
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
using Distributions: Beta, Cauchy, Dirichlet, Exponential, Gamma,
                     LocationScale, LogNormal, MixtureModel, Normal, Poisson,
                     TDist, cdf, logcdf, logccdf, logpdf
using Enzyme
using LogDensityProblems
using LogExpFunctions: logistic, logit
using ReactiveKernels: prepare
using ReactiveKernelsPPL: constrain, coordinate_names, logjac
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

# K=2 Stan partial-correlation-vine Jacobian (Digest-2 RK pin 45f765e):
# the single packed coordinate is z = tanh(t), with logjac
# log(1 - z^2).
function _lkj2_vine_logjac(t)
    z = tanh(t)
    return log1p(-z^2)
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

# SB `differenced_ar1_path` verbatim (`x` zero-started, `d[0] = 0`,
# `d[t] = beta*d[t-1] + sigma*z[t]`, `x[t+1] = x[t] + d[t]`), over the
# T-1 constrained innovations.
function _ref_dar_path(beta, sigma, z)
    n = length(z)
    x = zeros(Float64, n + 1)
    inc = 0.0
    for t in 1:n
        inc = beta * inc + sigma * z[t]
        x[t + 1] = x[t] + inc
    end
    return x
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
_parity_cols_dar = (; t = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    y = [0.5, -0.2, 0.1, 0.9, 1.4, 1.1])
# Corpus-56 horseshoe data (thin-layer `test_horseshoe.jl` mirror).
_parity_cols_hs = (;
    x1 = [0.5, -1.0, 1.5, 0.0],
    x2 = [1.0, 0.5, -0.5, 2.0],
    y = [1.0, 2.0, 1.5, 2.5],
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
    jac = u[5] + u[6] + _lkj2_vine_logjac(u[7])
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

@testset "rk parity horseshoe flat" begin
    # Thin-layer corpus-56 mirror: per-coefficient triples, no
    # `:coefficient` block (every coordinate derives in-graph).
    brmi = @brm _parity_cols_hs begin
        mu ~ 1 + x1 + x2
        effect(mu, x1) ~ Horseshoe()
        effect(mu, x2) ~ Horseshoe(local_scale=0.5, global_scale=0.25)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 8
    @test _layout_signature(layout) == [
        (:sampled, :sigma, 1, :exp),
        (:sampled, :horseshoe_mu_Intercept_normal, 1, :identity),
        (:sampled, :horseshoe_mu_x1_raw, 1, :identity),
        (:sampled, :horseshoe_mu_x1_lambda, 1, :exp),
        (:sampled, :horseshoe_mu_x1_tau, 1, :exp),
        (:sampled, :horseshoe_mu_x2_raw, 1, :identity),
        (:sampled, :horseshoe_mu_x2_lambda, 1, :exp),
        (:sampled, :horseshoe_mu_x2_tau, 1, :exp),
    ]
    # Corpus-56 probe by coordinate name (peer's `u` order pinned in
    # `test_horseshoe.jl`; the map keeps this test order-robust).
    byname = Dict(
        :sigma => 0.5, :horseshoe_mu_Intercept_normal => 0.1,
        :horseshoe_mu_x1_raw => -0.2, :horseshoe_mu_x1_lambda => 0.3,
        :horseshoe_mu_x1_tau => 0.4, :horseshoe_mu_x2_raw => 0.15,
        :horseshoe_mu_x2_lambda => -0.35, :horseshoe_mu_x2_tau => 0.25)
    u = Float64[byname[n] for n in coordinate_names(layout)]
    nt = constrain(layout, u)
    sig = nt.sigma
    b1 = nt.horseshoe_mu_x1_raw * nt.horseshoe_mu_x1_lambda *
        nt.horseshoe_mu_x1_tau
    b2 = nt.horseshoe_mu_x2_raw * nt.horseshoe_mu_x2_lambda *
        nt.horseshoe_mu_x2_tau
    cols = _parity_cols_hs
    mu_hat = nt.horseshoe_mu_Intercept_normal .+ b1 .* cols.x1 .+
        b2 .* cols.x2
    ll = sum(logpdf.(Normal.(mu_hat, sig), cols.y))
    # Stan-kernel halves: unnormalized Cauchy + log-Jacobian, NO
    # truncation renormalizer (SB-settled; the R2D2 `+log(2)` divergence
    # does NOT apply to the `:positive_stan` horseshoe triples).
    pr = logpdf(Normal(0, 1), nt.horseshoe_mu_Intercept_normal) +
        logpdf(Normal(0, 1), nt.horseshoe_mu_x1_raw) +
        logpdf(Cauchy(0, 1), nt.horseshoe_mu_x1_lambda) +
        logpdf(Cauchy(0, 1), nt.horseshoe_mu_x1_tau) +
        logpdf(Normal(0, 1), nt.horseshoe_mu_x2_raw) +
        logpdf(Cauchy(0, 0.5), nt.horseshoe_mu_x2_lambda) +
        logpdf(Cauchy(0, 0.25), nt.horseshoe_mu_x2_tau) +
        logpdf(Exponential(1), sig)
    jac = log(sig) + log(nt.horseshoe_mu_x1_lambda) +
        log(nt.horseshoe_mu_x1_tau) + log(nt.horseshoe_mu_x2_lambda) +
        log(nt.horseshoe_mu_x2_tau)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    # SB anchor: BridgeStan full posterior (`propto=false`, Jacobian) at
    # this point is `-20.29984607289203` (BRM SB structured emission,
    # StanBlocks `24578c3`, handoff numbers 23:00Z).
    @test _rk_query(backend, :posterior, u) ≈ -20.29984607289203
    _check_parity_gradient(backend, u)
end

@testset "rk parity dar default" begin
    brmi = @brm _parity_cols_dar begin
        mu ~ 1 + dar(t)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 9
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 1, :identity),
        (:sampled, :dar_mu_t_beta, 1, :interval),
        (:sampled, :dar_mu_t_sigma, 1, :exp),
        (:sampled, :s, 1, :exp),
        (:scan, :_ppl_dar_z_dar_mu, 5, :identity),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    beta, sigma = nt.dar_mu_t_beta, nt.dar_mu_t_sigma
    z = nt._ppl_dar_z_dar_mu
    path = _ref_dar_path(beta, sigma, z)
    @test path[1] == 0.0
    cols = _parity_cols_dar
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ path, nt.s), cols.y))
    # Truncated persistence WITH the Stan truncation renormalizer; the
    # HalfNormal scale carries the thin-layer `:positive` half
    # renormalizer (+log 2, the R2D2-tau precedent) over SB's
    # Stan-convention unnormalized half-normal.
    zn = Normal(0.5, 0.2)
    pr = logpdf(Normal(0, 1), nt.mu[1]) +
        logpdf(Exponential(1), nt.s) +
        (logpdf(zn, beta) - log(cdf(zn, 1) - cdf(zn, 0))) +
        (logpdf(Normal(0, 0.2), sigma) + log(2)) +
        sum(logpdf.(Normal(0, 1), z))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = (log(beta) + log1p(-beta)) + u[3] + u[4]
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@testset "rk parity dar prior overrides" begin
    brmi = @brm _parity_cols_dar begin
        mu ~ 1 + dar(t)
        ar(mu, dar(t)) ~ Normal(0.6, 0.1)
        sd(mu, dar(t)) ~ Normal(0, 0.3)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 9
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 1, :identity),
        (:sampled, :dar_mu_t_beta, 1, :interval),
        (:sampled, :dar_mu_t_sigma, 1, :exp),
        (:sampled, :s, 1, :exp),
        (:scan, :_ppl_dar_z_dar_mu, 5, :identity),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    beta, sigma = nt.dar_mu_t_beta, nt.dar_mu_t_sigma
    z = nt._ppl_dar_z_dar_mu
    path = _ref_dar_path(beta, sigma, z)
    @test path[1] == 0.0
    cols = _parity_cols_dar
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ path, nt.s), cols.y))
    zn = Normal(0.6, 0.1)
    pr = logpdf(Normal(0, 1), nt.mu[1]) +
        logpdf(Exponential(1), nt.s) +
        (logpdf(zn, beta) - log(cdf(zn, 1) - cdf(zn, 0))) +
        (logpdf(Normal(0, 0.3), sigma) + log(2)) +
        sum(logpdf.(Normal(0, 1), z))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = (log(beta) + log1p(-beta)) + u[3] + u[4]
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
        (:varying, :xi_g, 3, :identity),
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
        (:varying, :xi_g, 3, :identity),
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
        (:varying_corr, :L_g, 1, :lkj),
        (:varying, :tau_g, 2, :exp),
        (:varying, :z_flat_g, 6, :identity),
    ]
    # The joint Stage-C point was re-anchored at the Digest-2 vine LKJ
    # pin; the self-contained references above still verify the wiring.
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
    @test _rk_query(backend, :likelihood, u) ≈ -34.484707540969616 atol = 1e-12
    @test _rk_query(backend, :prior, u) ≈ -12.267527341929741 atol = 1e-12
    jac = u[2] + u[4] + u[5] + _lkj2_vine_logjac(u[3])
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
        (:varying_corr, :L_g, 1, :lkj),
        (:varying, :tau_g, 2, :exp),
        (:varying, :z_flat_g, 6, :identity),
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
    jac = u[2] + u[4] + u[5] + _lkj2_vine_logjac(u[3])
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
    # Draws mint group-suffixed names (binding on repeats), so the ID
    # bucket surfaces plain group names.
    @test _layout_signature(layout) == [
        (:coefficient, :mu1_coef, 1, :identity),
        (:coefficient, :mu2_coef, 1, :identity),
        (:sampled, :s, 1, :exp),
        (:varying_corr, :L_g, 1, :lkj),
        (:varying, :tau_g, 2, :exp),
        (:varying, :z_flat_g, 6, :identity),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    Zs = [ones(6), _parity_cols_multi.x]
    r1 = _ref_corr_r(_parity_cols_multi.g, nt.L_g, nt.tau_g,
        nt.z_flat_g, Zs, 1:1)
    r2 = _ref_corr_r(_parity_cols_multi.g, nt.L_g, nt.tau_g,
        nt.z_flat_g, Zs, 2:2)
    ll = sum(logpdf.(Normal.(nt.mu1[1] .+ r1, nt.s), _parity_cols_multi.y)) +
        sum(logpdf.(Normal.(nt.mu2[1] .+ r2, nt.s), _parity_cols_multi.y2))
    pr = logpdf(Normal(0, 5), nt.mu1[1]) +
        logpdf(Normal(0, 5), nt.mu2[1]) +
        logpdf(Exponential(1), nt.s) +
        _ref_lkj_k2_eta1(nt.L_g) +
        sum(logpdf.(Normal(0, 1), nt.tau_g)) +
        sum(logpdf.(Normal(0, 1), nt.z_flat_g))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = u[3] + u[5] + u[6] + _lkj2_vine_logjac(u[4])
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
        (:varying_corr, :L_g, 1, :lkj),
        (:varying, :tau_g, 2, :exp),
        (:varying, :z_flat_g, 6, :identity),
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
    jac = u[2] + u[4] + u[5] + _lkj2_vine_logjac(u[3])
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

@testset "rk parity student-t sampled nu" begin
    t_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        y=[0.5, -0.2, 0.1, 2.9, 1.4, -1.1],
    )
    brmi = @brm t_cols begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        nu ~ Gamma(2, 0.1)
        y ~ LocationScale(mu, sigma, TDist(nu))
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 4
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 2, :identity),
        (:sampled, :sigma, 1, :exp),
        (:sampled, :nu, 1, :exp),
    ]
    u = [-0.4, 0.3, -0.2, 1.1]
    nt = constrain(layout, u)
    b = Vector(nt.mu)
    lp = b[1] .+ b[2] .* t_cols.x
    ll = sum(logpdf.(LocationScale.(lp, nt.sigma, TDist(nt.nu)), t_cols.y))
    pr = logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2]) +
        logpdf(Exponential(1), nt.sigma) + logpdf(Gamma(2, 0.1), nt.nu)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # Jacobian: sigma's + nu's exp (betas ride identity).
    @test logjac(layout, u) ≈ u[3] + u[4]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[3] + u[4]
    _check_parity_gradient(backend, u)
end

@testset "rk parity student-t literal nu" begin
    t_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        y=[0.5, -0.2, 0.1, 2.9, 1.4, -1.1],
    )
    brmi = @brm t_cols begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        y ~ LocationScale(mu, sigma, TDist(4.0))
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 2, :identity),
        (:sampled, :sigma, 1, :exp),
    ]
    u = [-0.4, 0.3, -0.2]
    nt = constrain(layout, u)
    b = Vector(nt.mu)
    lp = b[1] .+ b[2] .* t_cols.x
    ll = sum(logpdf.(LocationScale.(lp, nt.sigma, TDist(4.0)), t_cols.y))
    pr = logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2]) +
        logpdf(Exponential(1), nt.sigma)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[3]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[3]
    _check_parity_gradient(backend, u)
end

@testset "rk parity hurdle-poisson hu submodel" begin
    h_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        c=[0, 1, 3, 0, 4, 2],
    )
    brmi = @brm h_cols begin
        log(lambda) ~ 1 + x
        effect(lambda, Intercept) ~ Normal(0, 5)
        effect(lambda, x) ~ Normal(0, 2.5)
        logit(p_zero) ~ 1 + x
        effect(p_zero, Intercept) ~ Normal(0, 2)
        effect(p_zero, x) ~ Normal(0, 1)
        c ~ HurdlePoisson(lambda, p_zero)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 4
    @test _layout_signature(layout) == [
        (:coefficient, :lambda_coef, 2, :identity),
        (:coefficient, :p_zero_coef, 2, :identity),
    ]
    u = [0.5, -0.25, 0.1, 0.2]
    nt = constrain(layout, u)
    bl = Vector(nt.lambda)
    bh = Vector(nt.p_zero)
    lam = exp.(bl[1] .+ bl[2] .* h_cols.x)
    p = logistic.(bh[1] .+ bh[2] .* h_cols.x)
    ll = sum(logpdf.(HurdlePoisson.(lam, p), h_cols.c))
    pr = logpdf(Normal(0, 5), bl[1]) + logpdf(Normal(0, 2.5), bl[2]) +
        logpdf(Normal(0, 2), bh[1]) + logpdf(Normal(0, 1), bh[2])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # All-identity layout: no Jacobian.
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@testset "rk parity hurdle-poisson scalar p0" begin
    h_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        c=[0, 1, 3, 0, 4, 2],
    )
    brmi = @brm h_cols begin
        log(lambda) ~ 1 + x
        effect(lambda, Intercept) ~ Normal(0, 5)
        effect(lambda, x) ~ Normal(0, 2.5)
        p0 ~ Beta(2, 2)
        c ~ HurdlePoisson(lambda, p0)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    @test _layout_signature(layout) == [
        (:coefficient, :lambda_coef, 2, :identity),
        (:sampled, :p0, 1, :logistic),
    ]
    u = [0.5, -0.25, 0.3]
    nt = constrain(layout, u)
    bl = Vector(nt.lambda)
    lam = exp.(bl[1] .+ bl[2] .* h_cols.x)
    ll = sum(logpdf.(HurdlePoisson.(lam, nt.p0), h_cols.c))
    pr = logpdf(Normal(0, 5), bl[1]) + logpdf(Normal(0, 2.5), bl[2]) +
        logpdf(Beta(2, 2), nt.p0)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # Jacobian: the logit-constrained p0 only (betas ride identity).
    jac = log(nt.p0 * (1 - nt.p0))
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@testset "rk parity hurdle-poisson literal p0" begin
    h_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        c=[0, 1, 3, 0, 4, 2],
    )
    brmi = @brm h_cols begin
        log(lambda) ~ 1 + x
        effect(lambda, Intercept) ~ Normal(0, 5)
        effect(lambda, x) ~ Normal(0, 2.5)
        c ~ HurdlePoisson(lambda, 0.35)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 2
    @test _layout_signature(layout) == [
        (:coefficient, :lambda_coef, 2, :identity),
    ]
    u = [0.5, -0.25]
    nt = constrain(layout, u)
    bl = Vector(nt.lambda)
    lam = exp.(bl[1] .+ bl[2] .* h_cols.x)
    ll = sum(logpdf.(HurdlePoisson.(lam, 0.35), h_cols.c))
    pr = logpdf(Normal(0, 5), bl[1]) + logpdf(Normal(0, 2.5), bl[2])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@testset "rk parity ZIP sampled zi" begin
    z_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        c=[0, 2, 0, 3, 1, 0],
    )
    brmi = @brm z_cols begin
        log(lambda) ~ 1 + x
        zi ~ Beta(2, 2)
        c ~ ZeroInflatedPoisson(lambda, zi)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    @test _layout_signature(layout) == [
        (:coefficient, :lambda_coef, 2, :identity),
        (:sampled, :zi, 1, :logistic),
    ]
    u = [0.2, -0.3, 0.5]
    nt = constrain(layout, u)
    b = Vector(nt.lambda)
    lp = exp.(b[1] .+ b[2] .* z_cols.x)
    ll = sum(logpdf.(BRM.ZeroInflatedPoisson.(lp, nt.zi), z_cols.c))
    pr = logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2]) +
        logpdf(Beta(2, 2), nt.zi)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # Jacobian: zi's logistic (betas ride identity).
    @test logjac(layout, u) ≈ log(nt.zi) + log1p(-nt.zi)
    @test _rk_query(backend, :posterior, u) ≈
        ll + pr + log(nt.zi) + log1p(-nt.zi)
    _check_parity_gradient(backend, u)
end

@testset "rk parity ZIP literal zi" begin
    z_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        c=[0, 2, 0, 3, 1, 0],
    )
    brmi = @brm z_cols begin
        log(lambda) ~ 1 + x
        c ~ ZeroInflatedPoisson(lambda, 0.25)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 2
    @test _layout_signature(layout) == [
        (:coefficient, :lambda_coef, 2, :identity),
    ]
    u = [0.2, -0.3]
    nt = constrain(layout, u)
    b = Vector(nt.lambda)
    lp = exp.(b[1] .+ b[2] .* z_cols.x)
    ll = sum(logpdf.(BRM.ZeroInflatedPoisson.(lp, 0.25), z_cols.c))
    pr = logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
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

# HSGP emission parity (thin-layer Stage B: in-graph basis + floored
# scales + matmul summand). References re-derive the SB shapes with
# explicit loops in SB op order (never the thin-layer expressions):
# `_brm_fit_hsgp` fits, `_brm_apply_hsgp` trig columns,
# `CartesianIndices` tensor products, `brm_hsgp_sqrt_spd` folds, and
# the `_sb_hsgp`/`_sb_hsgp_aniso` prior block (scalar lognormals +
# std-normal beta plate, no truncation normalizer on the floored rhos).
_parity_cols_hsgp = (;
    x = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
    z = [-0.25, 0.75, -1.25, 0.5, 1.0, -0.5],
    y = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
)

const _HSGP_SQRT2PI = 2.5066282746310002

# One axis's 1D trig basis (SB `_brm_apply_hsgp` element order verbatim).
function _ref_hsgp_axis_basis(x, k, c)
    n = length(x)
    mu = sum(x) / n
    L = c * maximum(abs.(x .- mu))
    lam = [(kk * pi / (2 * L))^2 for kk in 1:k]
    P = zeros(n, k)
    for kk in 1:k, i in 1:n
        P[i, kk] = (1 / sqrt(L)) * sin(sqrt(lam[kk]) * (x[i] - mu + L))
    end
    return P, lam
end

# Full smooth vector: tensor-product basis + SB `brm_hsgp_sqrt_spd`
# folds (left-assoc, SB order) + the `PHI * (s .* beta)` summand.
function _ref_hsgp_muv(xs, Ks, cs, rhos, sigh, beta; iso)
    d = length(xs)
    n = length(first(xs))
    bases = [_ref_hsgp_axis_basis(xs[j], Ks[j], cs[j]) for j in 1:d]
    K = Tuple(Ks)
    M = prod(K)
    PHI = zeros(n, M)
    o2 = zeros(M, d)
    for (b, I) in enumerate(CartesianIndices(K))
        for i in 1:n
            v = 1.0
            for j in 1:d
                v *= bases[j][1][i, I[j]]
            end
            PHI[i, b] = v
        end
        for j in 1:d
            o2[b, j] = bases[j][2][I[j]]
        end
    end
    rr = iso ? fill(rhos[1], d) : rhos
    scale = sigh
    for j in 1:d
        scale *= sqrt(rr[j] * _HSGP_SQRT2PI)
    end
    s = Vector{Float64}(undef, M)
    for b in 1:M
        ex = 0.0
        for j in 1:d
            ex += rr[j] * rr[j] * o2[b, j]
        end
        s[b] = scale * exp(-0.25 * ex)
    end
    return PHI * (s .* beta)
end

function _ref_hsgp_prior(a, sig, rhos, sigh, beta)
    return logpdf(Normal(0, 5), a) + logpdf(Exponential(1), sig) +
        sum(logpdf(LogNormal(0, 1), r) for r in rhos) +
        logpdf(LogNormal(0, 1), sigh) + sum(logpdf.(Normal(0, 1), beta))
end

@testset "rk parity hsgp 1d" begin
    brmi = @brm _parity_cols_hsgp begin
        mu ~ 1 + hsgp(x; k = 4)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 8
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 1, :identity),
        (:sampled, :sigma, 1, :exp),
        (:sampled, :rho_hsgp_x, 1, :floored),
        (:sampled, :sigma_hsgp_x, 1, :exp),
        (:hsgp, :beta_raw_hsgp_x, 4, :identity),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    f = _ref_hsgp_muv([_parity_cols_hsgp.x], [4], [1.5], [nt.rho_hsgp_x],
        nt.sigma_hsgp_x, Vector(nt.beta_raw_hsgp_x); iso = true)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ f, nt.sigma), _parity_cols_hsgp.y))
    pr = _ref_hsgp_prior(nt.mu[1], nt.sigma, [nt.rho_hsgp_x], nt.sigma_hsgp_x,
        Vector(nt.beta_raw_hsgp_x))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = u[2] + u[3] + u[4]
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@testset "rk parity hsgp aniso" begin
    brmi = @brm _parity_cols_hsgp begin
        mu ~ 1 + hsgp(x, z; k = (4, 3), c = (1.5, 2.0), iso = false)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 17
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 1, :identity),
        (:sampled, :sigma, 1, :exp),
        (:sampled, :rho_hsgp_x_z_1, 1, :floored),
        (:sampled, :rho_hsgp_x_z_2, 1, :floored),
        (:sampled, :sigma_hsgp_x_z, 1, :exp),
        (:hsgp, :beta_raw_hsgp_x_z, 12, :identity),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    rhos = [nt.rho_hsgp_x_z_1, nt.rho_hsgp_x_z_2]
    f = _ref_hsgp_muv([_parity_cols_hsgp.x, _parity_cols_hsgp.z], [4, 3],
        [1.5, 2.0], rhos, nt.sigma_hsgp_x_z, Vector(nt.beta_raw_hsgp_x_z);
        iso = false)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ f, nt.sigma), _parity_cols_hsgp.y))
    pr = _ref_hsgp_prior(nt.mu[1], nt.sigma, rhos, nt.sigma_hsgp_x_z,
        Vector(nt.beta_raw_hsgp_x_z))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = u[2] + u[3] + u[4] + u[5]
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@testset "rk parity hsgp k1" begin
    brmi = @brm _parity_cols_hsgp begin
        mu ~ 1 + hsgp(x; k = 1)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 5
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 1, :identity),
        (:sampled, :sigma, 1, :exp),
        (:sampled, :rho_hsgp_x, 1, :exp),
        (:sampled, :sigma_hsgp_x, 1, :exp),
        (:hsgp, :beta_raw_hsgp_x, 1, :identity),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    f = _ref_hsgp_muv([_parity_cols_hsgp.x], [1], [1.5], [nt.rho_hsgp_x],
        nt.sigma_hsgp_x, Vector(nt.beta_raw_hsgp_x); iso = true)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ f, nt.sigma), _parity_cols_hsgp.y))
    pr = _ref_hsgp_prior(nt.mu[1], nt.sigma, [nt.rho_hsgp_x], nt.sigma_hsgp_x,
        Vector(nt.beta_raw_hsgp_x))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = u[2] + u[3] + u[4]
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

# Mixture parity cases (pair fam-mixture, RK 49ebaf1): the committed
# oracles are Distributions.jl loops over the constrained point; the
# SB-point comparison (same models, SB brief values) rides the verdict
# probe, not the committed suite.
@testset "rk parity mixture gaussian" begin
    df = (; y=[-2.0, -1.8, 1.9, 2.2])
    brmi = @brm df begin
        mu1 ~ Normal(-2, 0.1)
        mu2 ~ Normal(2, 0.1)
        log(sigma) ~ 1
        y ~ MixtureModel([Normal(mu1, sigma), Normal(mu2, sigma)], [0.4, 0.6])
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    @test _layout_signature(layout) == [
        (:coefficient, :sigma_coef, 1, :identity),
        (:sampled, :mu1, 1, :identity),
        (:sampled, :mu2, 1, :identity),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    mixture = MixtureModel(
        [Normal(nt.mu1, exp(nt.sigma[1])), Normal(nt.mu2, exp(nt.sigma[1]))],
        [0.4, 0.6])
    ll = sum(logpdf.(mixture, df.y))
    pr = logpdf(Normal(-2, 0.1), nt.mu1) + logpdf(Normal(2, 0.1), nt.mu2) +
        logpdf(Normal(), nt.sigma[1])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@testset "rk parity mixture poisson" begin
    df = (; y=[0, 1, 3, 5, 2])
    brmi = @brm df begin
        lambda1 ~ Exponential(1)
        lambda2 ~ Exponential(1)
        y ~ MixtureModel([Poisson(lambda1), Poisson(lambda2)], [0.3, 0.7])
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 2
    @test _layout_signature(layout) == [
        (:sampled, :lambda1, 1, :exp),
        (:sampled, :lambda2, 1, :exp),
    ]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    mixture = MixtureModel([Poisson(nt.lambda1), Poisson(nt.lambda2)],
        [0.3, 0.7])
    ll = sum(logpdf.(mixture, df.y))
    pr = logpdf(Exponential(1), nt.lambda1) +
        logpdf(Exponential(1), nt.lambda2)
    jac = u[1] + u[2]
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@testset "rk parity mi() missing-response obs-only likelihood" begin
    # Case A (decision 05aemvx): packed obs slices; the likelihood sees
    # observed rows only while predictors stay full-length. Reference is
    # an independent Distributions.jl hand oracle; the plain obs-only twin
    # (GLM-fused, remapped probe) cross-checks across lowering paths.
    cols = (; x=[-1.0, 0.5, 2.0, 0.25],
              y=Union{Missing,Float64}[0.2, missing, -0.4, missing])
    brmi = @brm cols begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        mi(y) ~ Normal(mu, sigma)
    end
    backend = BRM.RKBRMI(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    @test _layout_signature(layout) == [
        (:coefficient, :mu_coef, 2, :identity),
        (:sampled, :sigma, 1, :exp),
    ]
    u = [0.0, 0.05, -0.05]
    nt = constrain(layout, u)
    mu_o = nt.mu[1] .+ nt.mu[2] .* [-1.0, 2.0]
    ll = sum(logpdf.(Normal.(mu_o, nt.sigma), [0.2, -0.4]))
    pr = logpdf(Normal(0, 1), nt.mu[1]) +
        logpdf(Normal(0, 1), nt.mu[2]) +
        logpdf(Exponential(2), nt.sigma)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[3]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[3]
    _check_parity_gradient(backend, u)
    twin = BRM.RKBRMI(@brm (; x=[-1.0, 2.0], y=[0.2, -0.4]) begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end)
    # The plain twin fuses to the GLM object (layout [beta, sigma, alpha]);
    # remap the probe and compare across lowering paths (fused reduction vs
    # plate sum agree to 1 ulp, not bit-exact).
    u_twin = [u[2], u[3], u[1]]
    @test _rk_query(twin, :posterior, u_twin) ≈
        _rk_query(backend, :posterior, u)
end
