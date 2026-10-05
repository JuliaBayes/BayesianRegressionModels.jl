# test/rk_parity.jl — BRM→RK end-to-end parity (ranef + P2 kernel +
# ordinal-extras + hsgp models).
#
# Run: julia --project=test test/rk_parity.jl
#
# Chunked runs (this file OOMs single-process on squeezed hosts): pass
# substring filters matching `@stestset` names as trailing args, or
# comma-separated via `BRM_TEST_FILTER`. Empty filter runs everything.
#
#     julia --project=test test/rk_parity.jl "von-Mises" "mixture"
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
                     InverseGaussian, Laplace, LKJCholesky, LocationScale,
                     LogNormal,
                     MixtureModel, NegativeBinomial,
                     Normal, Poisson, TDist, Uniform, VonMises, Weibull, cdf,
                     logcdf,
                     logccdf, logpdf, truncated
using Enzyme
using LinearAlgebra: cholesky, Symmetric
using LogDensityProblems
using LogExpFunctions: logistic, logit
import ReactiveKernels
using ReactiveKernels: prepare
using ReactiveKernelsPPL: constrain, coordinate_names, logjac, prepare_query
using Random: Xoshiro, randn
using SpecialFunctions: besselix, logbeta, loggamma

# Substring subset contract for chunked runs (see testset_filter.jl): blocks
# below are `@stestset`, selectable via trailing ARGS or `BRM_TEST_FILTER`.
include(joinpath(@__DIR__, "testset_filter.jl"))
include(joinpath(@__DIR__, "spline_parity_models.jl"))

const BRM = BayesianRegressionModels
const _PARITY_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
include(joinpath(@__DIR__, "rk_source_roundtrip.jl"))

_parity_backend(brmi) = check_rk_source_roundtrip(BRM.RKBRMI(brmi))

# Read the explicitly named population coordinates in formula order. The
# density references below retain their independent predictor/prior algebra.
function _parity_population(nt, backend, name)
    hasproperty(nt, name) && return getproperty(nt, name)
    plan = backend.plan isa BRM._RKValuePlan ? backend.plan.regression : backend.plan
    predictor = only(p for p in plan.predictors if p.name === name)
    coefficients = Float64[]
    for term in predictor.terms
        term.kind in (:intercept, :continuous, :factor, :monotonic, :ar, :me) || continue
        value = getproperty(nt, Symbol(name, "_", term.addressee))
        value isa Number ? push!(coefficients, value) : append!(coefficients, value)
    end
    coefficients
end

function _parity_ranef(nt, backend)
    plan = backend.plan isa BRM._RKValuePlan ? backend.plan.regression : backend.plan
    bucket = only(plan.ranef_buckets)
    suffix = bucket.id === nothing ? string(bucket.group) :
        string(bucket.id, "_", bucket.group)
    stem = "ranef_draws_" * suffix
    draws = getproperty(nt, Symbol(stem))
    sd, z = draws.sd, draws.z
    L = length(bucket.margins) == 1 ? ones(1, 1) : draws.L
    (; sd, z, L, zflat=vec(permutedims(z)))
end

# Query the same bound ordinary source as the production density route,
# including prepared data-only definitions and explicit observation roles.
function _rk_query(backend::BRM.RKBRMI, want::Symbol, u)
    translated = _rk_translated(backend)
    preset = want === :posterior ? :sampler : want
    return Base.invokelatest(prepare_query(backend.model, translated, preset), u)
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

# Obtain the production lowering, including prepared data-only definitions.
function _rk_translated(backend::BRM.RKBRMI)
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    return ext._rk_translated_plan(backend.plan)
end

function _rk_query_translated(backend::BRM.RKBRMI, translated, want::Symbol, u)
    kern = prepare_query(backend.model, translated, want)
    return Base.invokelatest(kern, Vector{Float64}(u))
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
# term-multimembership fixtures (same 6-row literals as the SB M1–M5 /
# S1–S2 probes; union [a,b,c], G = 3, M = 2; gr groups [s1..s4] over
# strata [A,A,B,B] with no straddle).
_parity_cols_mm = (;
    g1 = ["a", "a", "b", "c", "b", "a"],
    g2 = ["b", "c", "c", "a", "a", "b"],
    w1 = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0],
    w2 = [1.0, 1.0, 3.0, 2.0, 1.0, 1.0],
    x = [0.2, -0.1, 0.4, 0.3, -0.5, 0.1],
    y = [0.1, 0.2, 0.3, -0.2, 0.15, 0.05],
)
_parity_cols_gr = (;
    g = ["s1", "s1", "s2", "s3", "s3", "s4"],
    b = ["A", "A", "A", "B", "B", "B"],
    x = [0.2, -0.1, 0.4, 0.3, -0.5, 0.1],
    y = [0.1, 0.2, 0.3, -0.2, 0.15, 0.05],
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

@stestset "rk parity mo monotonic" begin
    brmi = @brm _parity_cols_mo begin
        mu ~ 1 + mo(c)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 4
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    contrast = _ref_mo_contrast(nt.mo_mu_c_contrast.simplex_incr, _parity_cols_mo.c)
    ll = sum(logpdf.(Normal.(_parity_population(nt, backend, :mu)[1] .+ _parity_population(nt, backend, :mu)[2] .* contrast, nt.s),
        _parity_cols_mo.y))
    # Full Dirichlet logpdf (normalizer included): the thin layer keeps
    # the log-multivariate-Beta constant Stan drops for data alpha, so the
    # RK posterior exceeds Stan's by exactly that constant (peer-verified
    # core parity is modulo it) while every gradient agrees.
    pr = logpdf(Normal(0, 1), _parity_population(nt, backend, :mu)[1]) +
        logpdf(Normal(0, 1), _parity_population(nt, backend, :mu)[2]) +
        logpdf(Exponential(1), nt.s) +
        logpdf(Dirichlet(ones(2)), nt.mo_mu_c_contrast.simplex_incr)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = u[3] + _ref_simplex_logjac(u[4:4])
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@stestset "rk parity mo1 summand (override alpha)" begin
    brmi = @brm _parity_cols_mo begin
        mu ~ 1 + mo1(c)
        simplex(mu, mo1(c)) ~ Dirichlet(1, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    contrast = _ref_mo_contrast(nt.mo1_mu_c_contrast.simplex_incr, _parity_cols_mo.c)
    ll = sum(logpdf.(Normal.(_parity_population(nt, backend, :mu)[1] .+ contrast, nt.s), _parity_cols_mo.y))
    pr = logpdf(Normal(0, 1), _parity_population(nt, backend, :mu)[1]) +
        logpdf(Exponential(1), nt.s) +
        logpdf(Dirichlet([1.0, 2.0]), nt.mo1_mu_c_contrast.simplex_incr)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = u[2] + _ref_simplex_logjac(u[3:3])
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@stestset "rk parity correlated outcomes K=2" begin
    brmi = @brm _parity_cols_corr begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior = Exponential(1), shape = 2)
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 7
    @test coordinate_names(layout) == [:mu1_Intercept, :mu1_x, :mu2_Intercept, :mu2_x, Symbol("L_res_scales.1"),
        Symbol("L_res_scales.2"), Symbol("L_res_L_corr.1")]
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    cols = _parity_cols_corr
    m1 = _parity_population(nt, backend, :mu1)[1] .+ _parity_population(nt, backend, :mu1)[2] .* cols.x
    m2 = _parity_population(nt, backend, :mu2)[1] .+ _parity_population(nt, backend, :mu2)[2] .* cols.x
    L = [nt.L_res_scales[i] * nt.L_res_L_corr[i, j] for i in 1:2, j in 1:2]
    ll = sum(_ref_mvn_chol_row([cols.y1[r], cols.y2[r]], [m1[r], m2[r]], L)
        for r in 1:6)
    pr = sum(logpdf(Normal(0, 1), c) for c in (_parity_population(nt, backend, :mu1)..., _parity_population(nt, backend, :mu2)...)) +
        sum(logpdf(Exponential(1), s) for s in nt.L_res_scales) +
        _ref_lkj_k2(2.0, nt.L_res_L_corr)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = u[5] + u[6] + _lkj2_vine_logjac(u[7])
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@stestset "rk parity correlated outcomes K=3 sampled scale" begin
    brmi = @brm _parity_cols_corr begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        mu3 ~ 1 + x
        tau ~ Exponential(1)
        L3 ~ LKJCovarianceFactor(3; scale_prior = Exponential(tau), shape = 1.5)
        [y1, y2, y3] ~ MvNormalCholesky([mu1, mu2, mu3], L3)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 13
    u = collect(range(-0.3, 0.3; length = layout.total))
    nt = constrain(layout, u)
    cols = _parity_cols_corr
    m = [_parity_population(nt, backend, :mu1)[1] .+ _parity_population(nt, backend, :mu1)[2] .* cols.x,
        _parity_population(nt, backend, :mu2)[1] .+ _parity_population(nt, backend, :mu2)[2] .* cols.x,
        _parity_population(nt, backend, :mu3)[1] .+ _parity_population(nt, backend, :mu3)[2] .* cols.x]
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

@stestset "rk parity r2d2 flat" begin
    brmi = @brm _parity_cols_r2d2 begin
        mu ~ 1 + x + z
        effect(mu, :) ~ r2d2(R2=Beta(2, 5), alpha=0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 7
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    R2, tau, phi = nt.r2d2_mu_R2, nt.r2d2_mu_tau_bsv, nt.r2d2_mu_phi
    cols = _parity_cols_r2d2
    sx = sqrt(phi[1] * R2 * tau^2 / _ref_sample_variance(cols.x))
    sz = sqrt(phi[2] * R2 * tau^2 / _ref_sample_variance(cols.z))
    mu_hat = _parity_population(nt, backend, :mu)[1] .+ _parity_population(nt, backend, :mu)[2] .* cols.x .+ _parity_population(nt, backend, :mu)[3] .* cols.z
    ll = sum(logpdf.(Normal.(mu_hat, nt.s), cols.y))
    # The sampled tau carries the thin-layer `:positive` half
    # renormalizer (+log 2, peer-blessed in `test_r2d2.jl`) over SB's
    # Stan-convention unnormalized half-normal — a constant the RK
    # posterior exceeds SB's by, while every gradient agrees (the mo
    # Dirichlet-normalizer precedent).
    pr = logpdf(Normal(0, 1), _parity_population(nt, backend, :mu)[1]) +
        logpdf(Normal(0, sx), _parity_population(nt, backend, :mu)[2]) +
        logpdf(Normal(0, sz), _parity_population(nt, backend, :mu)[3]) +
        logpdf(Exponential(1), nt.s) +
        logpdf(Beta(2, 5), R2) +
        logpdf(Normal(0, 1), tau) + log(2) +
        logpdf(Dirichlet([0.5, 0.5]), phi)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = log(nt.s) + log(R2) + log1p(-R2) + log(tau) + _ref_simplex_logjac(u[findall(n -> startswith(string(n), "r2d2_mu_phi."), coordinate_names(layout))])
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@stestset "rk parity r2d2 override + tau literal" begin
    brmi = @brm _parity_cols_r2d2 begin
        mu ~ 1 + x + z
        effect(mu, :) ~ r2d2(tau_bsv=2.0)
        effect(mu, x) ~ Normal(0, 3)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 5
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    R2, phi = nt.r2d2_mu_R2, nt.r2d2_mu_phi
    @test phi ≈ [1.0]
    cols = _parity_cols_r2d2
    sz = sqrt(phi[1] * R2 * 2.0^2 / _ref_sample_variance(cols.z))
    mu_hat = _parity_population(nt, backend, :mu)[1] .+ _parity_population(nt, backend, :mu)[2] .* cols.x .+ _parity_population(nt, backend, :mu)[3] .* cols.z
    ll = sum(logpdf.(Normal.(mu_hat, nt.s), cols.y))
    pr = logpdf(Normal(0, 1), _parity_population(nt, backend, :mu)[1]) +
        logpdf(Normal(0, 3), _parity_population(nt, backend, :mu)[2]) +
        logpdf(Normal(0, sz), _parity_population(nt, backend, :mu)[3]) +
        logpdf(Exponential(1), nt.s) +
        logpdf(Beta(1, 1), R2) +
        logpdf(Dirichlet([1.0]), phi)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = log(nt.s) + log(R2) + log1p(-R2)
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@stestset "rk parity r2d2 factor join" begin
    brmi = @brm _parity_cols begin
        mu ~ 0 + g
        effect(mu, :) ~ r2d2()
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 8
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    R2, tau, phi = nt.r2d2_mu_R2, nt.r2d2_mu_tau_bsv, nt.r2d2_mu_phi
    cols = _parity_cols
    sc = [sqrt(phi[k] * R2 * tau^2 / _ref_dummy_variance(cols.g, k))
        for k in 1:3]
    mu_hat = _parity_population(nt, backend, :mu)[_group_index(cols.g)]
    ll = sum(logpdf.(Normal.(mu_hat, nt.s), cols.y))
    pr = sum(logpdf(Normal(0, sc[k]), _parity_population(nt, backend, :mu)[k]) for k in 1:3) +
        logpdf(Exponential(1), nt.s) +
        logpdf(Beta(1, 1), R2) +
        logpdf(Normal(0, 1), tau) + log(2) +
        logpdf(Dirichlet(ones(3)), phi)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = log(nt.s) + log(R2) + log1p(-R2) + log(tau) + _ref_simplex_logjac(u[findall(n -> startswith(string(n), "r2d2_mu_phi."), coordinate_names(layout))])
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@stestset "rk parity horseshoe flat" begin
    brmi = @brm _parity_cols_hs begin
        mu ~ 1 + x1 + x2
        effect(mu, x1) ~ Horseshoe()
        effect(mu, x2) ~ Horseshoe(local_scale=0.5, global_scale=0.25)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 7
    @test coordinate_names(layout) == [:mu_tau, :mu_Intercept,
        Symbol("mu_x1.lambda"), Symbol("mu_x1.raw"), Symbol("mu_x2.lambda"), Symbol("mu_x2.raw"), :sigma]
    byname = Dict(:sigma=>0.5, :mu_Intercept=>0.1, :mu_tau=>0.4,
        Symbol("mu_x1.raw")=>-0.2, Symbol("mu_x1.lambda")=>0.3,
        Symbol("mu_x2.raw")=>0.15, Symbol("mu_x2.lambda")=>-0.35)
    u = Float64[byname[n] for n in coordinate_names(layout)]
    nt = constrain(layout, u)
    b1 = nt.mu_x1.raw * nt.mu_x1.lambda * nt.mu_tau
    b2 = nt.mu_x2.raw * nt.mu_x2.lambda * nt.mu_tau * 0.25
    cols = _parity_cols_hs
    mu_hat = nt.mu_Intercept .+ b1 .* cols.x1 .+ b2 .* cols.x2
    ll = sum(logpdf.(Normal.(mu_hat, nt.sigma), cols.y))
    # One shared global scale and two local scales, all normalized halves.
    pr = logpdf(Normal(), nt.mu_Intercept) +
        logpdf(Normal(), nt.mu_x1.raw) + logpdf(Normal(), nt.mu_x2.raw) +
        logpdf(Cauchy(0, 1), nt.mu_tau) +
        logpdf(Cauchy(0, 1), nt.mu_x1.lambda) +
        logpdf(Cauchy(0, 0.5), nt.mu_x2.lambda) + 3log(2) +
        logpdf(Exponential(1), nt.sigma)
    jac = log(nt.sigma) + log(nt.mu_tau) +
        log(nt.mu_x1.lambda) + log(nt.mu_x2.lambda)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@stestset "rk parity dar default" begin
    brmi = @brm _parity_cols_dar begin
        mu ~ 1 + dar(t)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 9
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    beta, sigma = nt.dar_mu_t_level.beta, nt.dar_mu_t_level.sigma
    z = getproperty(nt.dar_mu_t_level, :_ppl_scan_z_level)
    path = _ref_dar_path(beta, sigma, z)
    @test path[1] == 0.0
    cols = _parity_cols_dar
    ll = sum(logpdf.(Normal.(_parity_population(nt, backend, :mu)[1] .+ path, nt.s), cols.y))
    # The emitted persistence and scale have normalized truncated priors.
    zn = Normal(0.5, 0.2)
    pr = logpdf(Normal(0, 1), _parity_population(nt, backend, :mu)[1]) +
        logpdf(Exponential(1), nt.s) +
        logpdf(truncated(zn, 0, 1), beta) +
        logpdf(Normal(0, 0.2), sigma) + log(2) +
        sum(logpdf.(Normal(0, 1), z))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = log(beta) + log1p(-beta) + log(sigma) + log(nt.s)
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@stestset "rk parity dar prior overrides" begin
    brmi = @brm _parity_cols_dar begin
        mu ~ 1 + dar(t)
        ar(mu, dar(t)) ~ Normal(0.6, 0.1)
        sd(mu, dar(t)) ~ Normal(0, 0.3)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 9
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    beta, sigma = nt.dar_mu_t_level.beta, nt.dar_mu_t_level.sigma
    z = getproperty(nt.dar_mu_t_level, :_ppl_scan_z_level)
    path = _ref_dar_path(beta, sigma, z)
    @test path[1] == 0.0
    cols = _parity_cols_dar
    ll = sum(logpdf.(Normal.(_parity_population(nt, backend, :mu)[1] .+ path, nt.s), cols.y))
    zn = Normal(0.6, 0.1)
    pr = logpdf(Normal(0, 1), _parity_population(nt, backend, :mu)[1]) +
        logpdf(Exponential(1), nt.s) +
        logpdf(truncated(zn, 0, 1), beta) +
        logpdf(Normal(0, 0.3), sigma) + log(2) +
        sum(logpdf.(Normal(0, 1), z))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = log(beta) + log1p(-beta) + log(sigma) + log(nt.s)
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@stestset "rk parity K=1 intercept" begin
    brmi = @brm _parity_cols begin
        mu ~ 1 + (1 | g)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 6
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    r = only(_parity_ranef(nt, backend).sd) .* _parity_ranef(nt, backend).zflat[_group_index(_parity_cols.g)]
    ll = sum(logpdf.(Normal.(_parity_population(nt, backend, :mu)[1] .+ r, nt.sigma), _parity_cols.y))
    pr = logpdf(Normal(0, 5), _parity_population(nt, backend, :mu)[1]) +
        logpdf(Exponential(1), nt.sigma) +
        logpdf(LogNormal(0, 1), only(_parity_ranef(nt, backend).sd)) +
        sum(logpdf.(Normal(0, 1), _parity_ranef(nt, backend).zflat))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # Both positive scales contribute their exponential Jacobians.
    @test logjac(layout, u) ≈ (log(nt.sigma) + log(only(_parity_ranef(nt, backend).sd)))
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + (log(nt.sigma) + log(only(_parity_ranef(nt, backend).sd)))
    _check_parity_gradient(backend, u)
end

@stestset "rk parity K=1 intercept categorical grouping" begin
    brmi = @brm _parity_cols_cat begin
        mu ~ 1 + (1 | g)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 6
    # SB numbering is CA.levels positions (levelcodes) — independent of
    # the thin layer's sorted-strings `_declared_codes` encoder.
    idx = levelcode.(_parity_cols_cat.g)
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    r = only(_parity_ranef(nt, backend).sd) .* _parity_ranef(nt, backend).zflat[idx]
    ll = sum(logpdf.(Normal.(_parity_population(nt, backend, :mu)[1] .+ r, nt.sigma), _parity_cols_cat.y))
    pr = logpdf(Normal(0, 5), _parity_population(nt, backend, :mu)[1]) +
        logpdf(Exponential(1), nt.sigma) +
        logpdf(LogNormal(0, 1), only(_parity_ranef(nt, backend).sd)) +
        sum(logpdf.(Normal(0, 1), _parity_ranef(nt, backend).zflat))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ (log(nt.sigma) + log(only(_parity_ranef(nt, backend).sd)))
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + (log(nt.sigma) + log(only(_parity_ranef(nt, backend).sd)))
    _check_parity_gradient(backend, u)
end

@stestset "rk parity K=1 slope" begin
    brmi = @brm _parity_cols begin
        mu ~ 1 + (0 + x | g)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 6
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    r = _ref_slope_r(_parity_cols.g, _parity_ranef(nt, backend).sd[1], _parity_ranef(nt, backend).zflat, _parity_cols.x)
    ll = sum(logpdf.(Normal.(_parity_population(nt, backend, :mu)[1] .+ r, nt.sigma), _parity_cols.y))
    # The original positive Normal kernel is restricted without renormalization.
    pr = logpdf(Normal(0, 5), _parity_population(nt, backend, :mu)[1]) +
        logpdf(Exponential(1), nt.sigma) +
        logpdf(Normal(0, 1), _parity_ranef(nt, backend).sd[1]) +
        sum(logpdf.(Normal(0, 1), _parity_ranef(nt, backend).zflat))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[2] + u[3]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[2] + u[3]
    _check_parity_gradient(backend, u)
end

@stestset "rk parity correlated categorical declared and unused levels" begin
    data = (; x=[-0.8, 0.2, 0.7, -0.3, 0.4, 1.1],
        y=[0.1, -0.2, 0.5, 0.3, -0.1, 0.6],
        g=categorical(["a", "b", "a", "c", "b", "c"];
            levels=["b", "a", "c", "unused"]))
    saved = deepcopy(data)
    brmi = @brm data begin
        mu ~ 1 + (1 + x | g)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 13
    @test BRM._rk_grouping_levels(backend.plan.columns[:g]) ==
        ["b", "a", "c", "unused"]
    u = collect(range(-0.35, 0.35; length=layout.total))
    nt = constrain(layout, u)
    effects = _parity_ranef(nt, backend)
    @test size(effects.z) == (4, 2)
    # The pool's declared order, including its unobserved level, defines
    # latent rows. Observation positions are independent of sorted labels.
    positions = [2, 1, 2, 3, 1, 3]
    contribution = [effects.sd[1] * effects.z[j, 1] + data.x[i] *
        effects.sd[2] * (effects.L[2, 1] * effects.z[j, 1] +
            effects.L[2, 2] * effects.z[j, 2]) for (i, j) in enumerate(positions)]
    ll = sum(logpdf.(Normal.(nt.mu_Intercept .+ contribution, nt.sigma), data.y))
    pr = logpdf(Normal(0, 5), nt.mu_Intercept) +
        logpdf(Exponential(1), nt.sigma) +
        _ref_lkj_k2_eta1(effects.L) +
        sum(logpdf.(Normal(0, 1), effects.sd)) +
        sum(logpdf.(Normal(0, 1), effects.z))
    jac = log(nt.sigma) + sum(log, effects.sd) + _lkj2_vine_logjac(u[end])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
    @test isequal(data, saved)
end

@stestset "rk parity K=2 correlated (+ joint anchor)" begin
    brmi = @brm _parity_cols begin
        mu ~ 1 + (1 + x | g)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 11
    # The joint Stage-C point was re-anchored at the Digest-2 vine LKJ
    # pin; the self-contained references above still verify the wiring.
    legacy_point = collect(range(-0.4, 0.4; length=11))
    u = legacy_point[[1, 2, 4, 5, 6, 8, 10, 7, 9, 11, 3]]
    nt = constrain(layout, u)
    r = _ref_corr_r(_parity_cols.g, _parity_ranef(nt, backend).L, _parity_ranef(nt, backend).sd, _parity_ranef(nt, backend).zflat,
        [ones(6), _parity_cols.x], 1:2)
    ll = sum(logpdf.(Normal.(_parity_population(nt, backend, :mu)[1] .+ r, nt.sigma), _parity_cols.y))
    pr = logpdf(Normal(0, 5), _parity_population(nt, backend, :mu)[1]) +
        logpdf(Exponential(1), nt.sigma) +
        _ref_lkj_k2_eta1(_parity_ranef(nt, backend).L) +
        sum(logpdf.(Normal(0, 1), _parity_ranef(nt, backend).sd)) +
        sum(logpdf.(Normal(0, 1), _parity_ranef(nt, backend).zflat))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test _rk_query(backend, :likelihood, u) ≈ -34.484707540969616 atol = 1e-12
    @test _rk_query(backend, :prior, u) ≈ -12.267527341929741 atol = 1e-12
    jac = log(nt.sigma) + sum(log, _parity_ranef(nt, backend).sd) + _lkj2_vine_logjac(u[end])
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@stestset "rk parity ranef interaction" begin
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
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 11
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    r = _ref_corr_r(_parity_cols_xz.g, _parity_ranef(nt, backend).L, _parity_ranef(nt, backend).sd, _parity_ranef(nt, backend).zflat,
        [ones(6), _parity_cols_xz.x .* _parity_cols_xz.z], 1:2)
    ll = sum(logpdf.(Normal.(_parity_population(nt, backend, :mu)[1] .+ r, nt.sigma), _parity_cols_xz.y))
    pr = logpdf(Normal(0, 5), _parity_population(nt, backend, :mu)[1]) +
        logpdf(Exponential(1), nt.sigma) +
        _ref_lkj_k2_eta1(_parity_ranef(nt, backend).L) +
        sum(logpdf.(Normal(0, 1), _parity_ranef(nt, backend).sd)) +
        sum(logpdf.(Normal(0, 1), _parity_ranef(nt, backend).zflat))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = log(nt.sigma) + sum(log, _parity_ranef(nt, backend).sd) + _lkj2_vine_logjac(u[end])
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@stestset "rk parity multislice ID" begin
    brmi = @brm _parity_cols_multi begin
        mu1 ~ 1 + (1 | ID | g)
        mu2 ~ 1 + (0 + x | ID | g)
        y ~ Normal(mu1, s)
        y2 ~ Normal(mu2, s)
        effect(mu1, Intercept) ~ Normal(0, 5)
        effect(mu2, Intercept) ~ Normal(0, 5)
        s ~ Exponential(1)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 12
    # Draws mint group-suffixed names (binding on repeats), so the ID
    # bucket surfaces plain group names.
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    Zs = [ones(6), _parity_cols_multi.x]
    r1 = _ref_corr_r(_parity_cols_multi.g, _parity_ranef(nt, backend).L, _parity_ranef(nt, backend).sd,
        _parity_ranef(nt, backend).zflat, Zs, 1:1)
    r2 = _ref_corr_r(_parity_cols_multi.g, _parity_ranef(nt, backend).L, _parity_ranef(nt, backend).sd,
        _parity_ranef(nt, backend).zflat, Zs, 2:2)
    ll = sum(logpdf.(Normal.(_parity_population(nt, backend, :mu1)[1] .+ r1, nt.s), _parity_cols_multi.y)) +
        sum(logpdf.(Normal.(_parity_population(nt, backend, :mu2)[1] .+ r2, nt.s), _parity_cols_multi.y2))
    pr = logpdf(Normal(0, 5), _parity_population(nt, backend, :mu1)[1]) +
        logpdf(Normal(0, 5), _parity_population(nt, backend, :mu2)[1]) +
        logpdf(Exponential(1), nt.s) +
        _ref_lkj_k2_eta1(_parity_ranef(nt, backend).L) +
        sum(logpdf.(Normal(0, 1), _parity_ranef(nt, backend).sd)) +
        sum(logpdf.(Normal(0, 1), _parity_ranef(nt, backend).zflat))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = log(nt.s) + sum(log, _parity_ranef(nt, backend).sd) + _lkj2_vine_logjac(u[end])
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@stestset "rk parity treatment-dummy correlated" begin
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
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 11
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    Z2 = Float64.([v == 2 for v in _parity_cols_dummy.c])
    r = _ref_corr_r(_parity_cols_dummy.g, _parity_ranef(nt, backend).L, _parity_ranef(nt, backend).sd, _parity_ranef(nt, backend).zflat,
        [ones(6), Z2], 1:2)
    ll = sum(logpdf.(Normal.(_parity_population(nt, backend, :mu)[1] .+ r, nt.sigma), _parity_cols_dummy.y))
    pr = logpdf(Normal(0, 5), _parity_population(nt, backend, :mu)[1]) +
        logpdf(Exponential(1), nt.sigma) +
        _ref_lkj_k2_eta1(_parity_ranef(nt, backend).L) +
        sum(logpdf.(Normal(0, 1), _parity_ranef(nt, backend).sd)) +
        sum(logpdf.(Normal(0, 1), _parity_ranef(nt, backend).zflat))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = log(nt.sigma) + sum(log, _parity_ranef(nt, backend).sd) + _lkj2_vine_logjac(u[end])
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

# General panel-cell coverage belongs here. PK-specific models from the
# historical corpus are downstream RKPPLBench inputs (preserved in Git).
_kernel_vector_cols = (;
    t=[[0.5, 1.0, 2.0, 4.0] for _ in 1:3],
    intercept=[0.2, -0.3, 0.7], slope=[0.1, -0.2, 0.05],
    y=[[0.1, 0.4, -0.2, 0.8] for _ in 1:3])
_kernel_scalar_cols = (; x=[0.1, 0.2, 0.15, 0.25],
    intercept=[0.3, -0.2, 0.6, 0.4], y=[0.5, 1.2, 0.2, -0.3])

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

@stestset "rk parity ar(1) latent path" begin
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
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 10
    # Explicit scan innovations and named population coefficients share
    # the authored model's coordinate layout.
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    phi = tanh(nt.ar_mu_t.phi_raw)
    path = _ref_ar1_path(phi, getproperty(nt.ar_mu_t, :_ppl_scan_z_state))
    ll = sum(logpdf.(Normal.(_parity_population(nt, backend, :mu)[1] .+ nt.mu_ar_mu_t .* path, nt.sigma),
        ar_cols.y))
    pr = logpdf(Normal(0, 5), _parity_population(nt, backend, :mu)[1]) +
        logpdf(Normal(0, 1), nt.mu_ar_mu_t) +
        logpdf(Normal(0, 1), nt.ar_mu_t.phi_raw) +
        logpdf(Exponential(1), nt.sigma) +
        sum(logpdf.(Normal(0, 1), getproperty(nt.ar_mu_t, :_ppl_scan_z_state)))
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
# `latent(...)` Normal prior, the LP's named free coefficient, and the
# self-contained observation likelihood
# `x_obs ~ normal(x_true, sd)`. The synthetic observation response
# uses the latent array as its mean; both population coefficients retain
# their explicit names.
@stestset "rk parity me(x, sd) latent" begin
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
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 9
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
    xt = Vector(nt.me_x)
    @test !hasproperty(nt, :x_loc)
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

@stestset "rk parity student-t sampled nu" begin
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
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 4
    u = [-0.4, 0.3, -0.2, 1.1]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
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

@stestset "rk parity student-t literal nu" begin
    t_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        y=[0.5, -0.2, 0.1, 2.9, 1.4, -1.1],
    )
    brmi = @brm t_cols begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        y ~ LocationScale(mu, sigma, TDist(4.0))
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    u = [-0.4, 0.3, -0.2]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
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

# Modeled-nu twins (pair nuisance-nu, RK 99d278db): the N1/N1b probe
# shapes with the term-nuisance SB pins committed (SB brief values at
# BRM 97bb538 / SB 24578c3, reproduced bit-exact at lane tip).
@stestset "rk parity student-t modeled nu" begin
    nu_cols = (;
        x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        z=[1.0, 0.5, -0.5, 1.5, 0.0, -1.0],
        y=[1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
    )
    brmi = @brm nu_cols begin
        mu ~ 1 + x
        log(nu) ~ 1 + z
        y ~ LocationScale(mu, 2.0, TDist(nu))
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 4
    u = [0.5, -0.25, 1.2, 0.2]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
    c = _parity_population(nt, backend, :nu)
    lp = b[1] .+ b[2] .* nu_cols.x
    nu = exp.(c[1] .+ c[2] .* nu_cols.z)
    ll = sum(logpdf.(LocationScale.(lp, 2.0, TDist.(nu)), nu_cols.y))
    pr = sum(logpdf.(Normal(0, 1), u))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    @test _rk_query(backend, :posterior, u) ≈ -17.041416962231473
    _check_parity_gradient(backend, u)
end

@stestset "rk parity student-t modeled nu sampled scale" begin
    nu_cols = (;
        x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        z=[1.0, 0.5, -0.5, 1.5, 0.0, -1.0],
        y=[1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
    )
    brmi = @brm nu_cols begin
        mu ~ 1 + x
        s ~ Exponential(1)
        log(nu) ~ 1 + z
        y ~ LocationScale(mu, s, TDist(nu))
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 5
    u = [0.5, -0.25, 1.2, 0.2, 0.7]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
    c = _parity_population(nt, backend, :nu)
    lp = b[1] .+ b[2] .* nu_cols.x
    nu = exp.(c[1] .+ c[2] .* nu_cols.z)
    ll = sum(logpdf.(LocationScale.(lp, nt.s, TDist.(nu)), nu_cols.y))
    pr = sum(logpdf.(Normal(0, 1), u[1:4])) + logpdf(Exponential(1), nt.s)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[5]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[5]
    @test _rk_query(backend, :posterior, u) ≈ -18.367541834521536
    _check_parity_gradient(backend, u)
end

@stestset "rk parity hurdle-poisson hu submodel" begin
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
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 4
    u = [0.5, -0.25, 0.1, 0.2]
    nt = constrain(layout, u)
    bl = Vector(_parity_population(nt, backend, :lambda))
    bh = Vector(_parity_population(nt, backend, :p_zero))
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

@stestset "rk parity hurdle-poisson Z2 two-column hu submodel" begin
    # Z2 shape (term-nuisance spec brief 1mcop44, pair
    # nuisance-hurdle-p0): the hu submodel rides a DISTINCT column
    # from the rate predictor, under default Normal(0, 1) popefs
    # priors. The pinned posterior is the SB full-posterior value at
    # the spec u (SB leg of this pair's verdict brief).
    z2_cols = (;
        x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        z=[1.0, 0.5, -0.5, 1.5, 0.0, -1.0],
        c=[0, 1, 3, 0, 2, 1],
    )
    brmi = @brm z2_cols begin
        log(lambda) ~ 1 + x
        logit(p_zero) ~ 1 + z
        c ~ HurdlePoisson(lambda, p_zero)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 4
    u = [0.2, -0.1, -0.5, 0.3]
    nt = constrain(layout, u)
    bl = Vector(_parity_population(nt, backend, :lambda))
    bh = Vector(_parity_population(nt, backend, :p_zero))
    lam = exp.(bl[1] .+ bl[2] .* z2_cols.x)
    p = logistic.(bh[1] .+ bh[2] .* z2_cols.z)
    ll = sum(logpdf.(HurdlePoisson.(lam, p), z2_cols.c))
    pr = sum(logpdf(Normal(0, 1), c) for c in u)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # All-identity layout: no Jacobian.
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    @test _rk_query(backend, :posterior, u) ≈ -11.954631019953592
    _check_parity_gradient(backend, u)
end

@stestset "rk parity hurdle-poisson scalar p0" begin
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
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    u = [0.5, -0.25, 0.3]
    nt = constrain(layout, u)
    bl = Vector(_parity_population(nt, backend, :lambda))
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

@stestset "rk parity hurdle-poisson literal p0" begin
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
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 2
    u = [0.5, -0.25]
    nt = constrain(layout, u)
    bl = Vector(_parity_population(nt, backend, :lambda))
    lam = exp.(bl[1] .+ bl[2] .* h_cols.x)
    ll = sum(logpdf.(HurdlePoisson.(lam, 0.35), h_cols.c))
    pr = logpdf(Normal(0, 5), bl[1]) + logpdf(Normal(0, 2.5), bl[2])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@stestset "rk parity ZIP sampled zi" begin
    z_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        c=[0, 2, 0, 3, 1, 0],
    )
    brmi = @brm z_cols begin
        log(lambda) ~ 1 + x
        zi ~ Beta(2, 2)
        c ~ ZeroInflatedPoisson(lambda, zi)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    u = [0.2, -0.3, 0.5]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :lambda))
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

@stestset "rk parity wald sampled lambda" begin
    w_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        z=[1.2, 0.8, 1.1, 2.3, 0.7, 1.9],
    )
    brmi = @brm w_cols begin
        log(mu) ~ 1 + x
        lam ~ Exponential(1)
        z ~ InverseGaussian(mu, lam)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    u = [0.5, -0.25, 0.3]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
    mm = exp.(b[1] .+ b[2] .* w_cols.x)
    ll = sum(logpdf.(InverseGaussian.(mm, nt.lam), w_cols.z))
    pr = logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2]) +
        logpdf(Exponential(1), nt.lam)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[3]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[3]
    _check_parity_gradient(backend, u)
end

@stestset "rk parity ZIP literal zi" begin
    z_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        c=[0, 2, 0, 3, 1, 0],
    )
    brmi = @brm z_cols begin
        log(lambda) ~ 1 + x
        c ~ ZeroInflatedPoisson(lambda, 0.25)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 2
    u = [0.2, -0.3]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :lambda))
    lp = exp.(b[1] .+ b[2] .* z_cols.x)
    ll = sum(logpdf.(BRM.ZeroInflatedPoisson.(lp, 0.25), z_cols.c))
    pr = logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@stestset "rk parity ZIP zi submodel" begin
    # Z1 columns (spec brief 1mcop44): the posterior pin below is the
    # SB number (brief ff47x8), so this entry is the e2e SB-parity
    # check at the bumped RK pin. Hurdle hu-submodel twin.
    z_cols = (;
        x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        z=[1.0, 0.5, -0.5, 1.5, 0.0, -1.0],
        c=[0, 1, 3, 0, 2, 1],
    )
    brmi = @brm z_cols begin
        log(lambda) ~ 1 + x
        logit(zi) ~ 1 + z
        c ~ ZeroInflatedPoisson(lambda, zi)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 4
    u = [0.2, -0.1, -0.5, 0.3]
    nt = constrain(layout, u)
    bl = Vector(_parity_population(nt, backend, :lambda))
    bz = _parity_population(nt, backend, :zi)
    lam = exp.(bl[1] .+ bl[2] .* z_cols.x)
    p = logistic.(bz[1] .+ bz[2] .* z_cols.z)
    ll = sum(logpdf.(BRM.ZeroInflatedPoisson.(lam, p), z_cols.c))
    pr = sum(logpdf.(Normal(0, 1), u))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # All-identity layout: no Jacobian.
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    @test _rk_query(backend, :posterior, u) ≈ -12.817555472510582
    _check_parity_gradient(backend, u)
end

@stestset "rk parity negative-binomial sampled p" begin
    nb_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        c=[1, 3, 0, 2, 5, 1],
    )
    brmi = @brm nb_cols begin
        log(r) ~ 1 + x
        p ~ Beta(2, 2)
        c ~ NegativeBinomial(r, p)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    u = [0.2, -0.3, 0.5]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :r))
    rr = exp.(b[1] .+ b[2] .* nb_cols.x)
    ll = sum(logpdf.(NegativeBinomial.(rr, nt.p), nb_cols.c))
    pr = logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2]) +
        logpdf(Beta(2, 2), nt.p)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # Jacobian: p's logistic (betas ride identity).
    @test logjac(layout, u) ≈ log(nt.p) + log1p(-nt.p)
    @test _rk_query(backend, :posterior, u) ≈
        ll + pr + log(nt.p) + log1p(-nt.p)
    _check_parity_gradient(backend, u)
end

@stestset "rk parity negative-binomial literal p" begin
    nb_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        c=[1, 3, 0, 2, 5, 1],
    )
    brmi = @brm nb_cols begin
        log(r) ~ 1 + x
        c ~ NegativeBinomial(r, 0.4)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 2
    u = [0.2, -0.3]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :r))
    rr = exp.(b[1] .+ b[2] .* nb_cols.x)
    ll = sum(logpdf.(NegativeBinomial.(rr, 0.4), nb_cols.c))
    pr = logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@stestset "rk parity negative-binomial modeled p" begin
    # Pair nuisance-nb1p B1 shape: `log(r) ~ 1+x`, `logit(p) ~ 1+z`
    # over the term-nuisance probe columns (the hurdle hu-submodel
    # twin: p rides the scale-predictor slot under `logistic.`).
    nb_cols = (;
        x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        z=[1.0, 0.5, -0.5, 1.5, 0.0, -1.0],
        c=[3, 1, 6, 2, 1, 4],
    )
    brmi = @brm nb_cols begin
        log(r) ~ 1 + x
        logit(p) ~ 1 + z
        c ~ NegativeBinomial(r, p)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 4
    u = [0.2, -0.1, -0.5, 0.3]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :r))
    g = _parity_population(nt, backend, :p)
    rr = exp.(b[1] .+ b[2] .* nb_cols.x)
    pp = logistic.(g[1] .+ g[2] .* nb_cols.z)
    ll = sum(logpdf.(NegativeBinomial.(rr, pp), nb_cols.c))
    pr = logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2]) +
        logpdf(Normal(0, 1), g[1]) + logpdf(Normal(0, 1), g[2])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # All-identity layout: no Jacobian.
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@stestset "rk parity wald literal lambda" begin
    w_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        z=[1.2, 0.8, 1.1, 2.3, 0.7, 1.9],
    )
    brmi = @brm w_cols begin
        log(mu) ~ 1 + x
        z ~ InverseGaussian(mu, 2.0)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 2
    u = [0.5, -0.25]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
    mm = exp.(b[1] .+ b[2] .* w_cols.x)
    ll = sum(logpdf.(InverseGaussian.(mm, 2.0), w_cols.z))
    pr = logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@stestset "rk parity wald modeled lambda" begin
    # Pair nuisance-lam I1 (spec: term-nuisance verdict brief
    # 1mcop44): `log(mu) ~ 1+x` + `log(lam) ~ 1+z` over the shared
    # N=6 probe columns. The posterior pin is the SB full-posterior
    # value at u (propto=false, jacobian=true), re-verified fresh on
    # this lane's SB leg.
    w_cols = (;
        x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        z=[1.0, 0.5, -0.5, 1.5, 0.0, -1.0],
        y=[1.2, 0.8, 2.1, 1.5, 0.6, 1.9],
    )
    brmi = @brm w_cols begin
        log(mu) ~ 1 + x
        log(lam) ~ 1 + z
        y ~ InverseGaussian(mu, lam)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 4
    u = [0.2, -0.1, 0.5, 0.1]
    nt = constrain(layout, u)
    b_mu = Vector(_parity_population(nt, backend, :mu))
    b_lam = _parity_population(nt, backend, :lam)
    mm = exp.(b_mu[1] .+ b_mu[2] .* w_cols.x)
    ll_lam = exp.(b_lam[1] .+ b_lam[2] .* w_cols.z)
    ll = sum(logpdf.(InverseGaussian.(mm, ll_lam), w_cols.y))
    pr = sum(logpdf(Normal(0, 1), b) for b in
        (b_mu[1], b_mu[2], b_lam[1], b_lam[2]))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    @test _rk_query(backend, :posterior, u) ≈ -10.804159781573498
    _check_parity_gradient(backend, u)
end

@stestset "rk parity exponential" begin
    e_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        z=[1.2, 0.8, 1.1, 2.3, 0.7, 1.9],
    )
    brmi = @brm e_cols begin
        log(mu) ~ 1 + x
        z ~ Exponential(mu)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 2
    u = [0.5, -0.25]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
    mm = exp.(b[1] .+ b[2] .* e_cols.x)
    ll = sum(logpdf.(Exponential.(mm), e_cols.z))
    pr = logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@stestset "rk parity beta-binomial sampled precision" begin
    bb_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        n=[10, 8, 12, 6, 9, 11],
        c=[3, 5, 7, 2, 6, 4],
    )
    brmi = @brm bb_cols begin
        logit(mu) ~ 1 + x
        effect(mu, Intercept) ~ Normal(0, 5)
        effect(mu, x) ~ Normal(0, 2.5)
        phi ~ Gamma(2, 0.1)
        c ~ BetaBinomial2(n, mu, phi)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    u = [0.5, -0.25, 1.1]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
    mu = logistic.(b[1] .+ b[2] .* bb_cols.x)
    ll = sum(logpdf.(BetaBinomial2.(bb_cols.n, mu, nt.phi), bb_cols.c))
    pr = logpdf(Normal(0, 5), b[1]) + logpdf(Normal(0, 2.5), b[2]) +
        logpdf(Gamma(2, 0.1), nt.phi)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # Jacobian: phi's exp (betas ride identity).
    @test logjac(layout, u) ≈ u[3]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[3]
    _check_parity_gradient(backend, u)
end

@stestset "rk parity beta-binomial literal precision" begin
    bb_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        c=[3, 5, 7, 2, 6, 4],
    )
    brmi = @brm bb_cols begin
        logit(mu) ~ 1
        effect(mu, Intercept) ~ Normal(0, 5)
        c ~ BetaBinomial2(10, mu, 5.0)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 1
    u = [0.5]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
    mu = logistic.(b[1] .+ bb_cols.x .* 0.0)
    ll = sum(logpdf.(BetaBinomial2.(10, mu, 5.0), bb_cols.c))
    pr = logpdf(Normal(0, 5), b[1])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@stestset "rk parity beta-binomial column trials literal precision" begin
    bb_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        n=[10, 8, 12, 6, 9, 11],
        c=[3, 5, 7, 2, 6, 4],
    )
    brmi = @brm bb_cols begin
        logit(mu) ~ 1 + x
        effect(mu, Intercept) ~ Normal(0, 5)
        effect(mu, x) ~ Normal(0, 2.5)
        c ~ BetaBinomial2(n, mu, 5.0)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 2
    u = [0.5, -0.25]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
    mu = logistic.(b[1] .+ b[2] .* bb_cols.x)
    ll = sum(logpdf.(BetaBinomial2.(bb_cols.n, mu, 5.0), bb_cols.c))
    pr = logpdf(Normal(0, 5), b[1]) + logpdf(Normal(0, 2.5), b[2])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@stestset "rk parity beta-binomial modeled precision" begin
    # P3 (pair nuisance-precision, spec 1mcop44): logit-link mean +
    # log-link precision submodel over column trials; default
    # Normal(0, 1) popefs priors.
    bb_cols = (;
        x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        z=[1.0, 0.5, -0.5, 1.5, 0.0, -1.0],
        n=[8, 6, 10, 5, 4, 9],
        c=[3, 1, 6, 2, 1, 4],
    )
    brmi = @brm bb_cols begin
        logit(mean) ~ 1 + x
        log(precision) ~ 1 + z
        c ~ BetaBinomial2(n, mean, precision)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 4
    u = [0.2, -0.1, 1.0, 0.15]
    nt = constrain(layout, u)
    bm = Vector(_parity_population(nt, backend, :mean))
    bp = Vector(_parity_population(nt, backend, :precision))
    mu = logistic.(bm[1] .+ bm[2] .* bb_cols.x)
    phi = exp.(bp[1] .+ bp[2] .* bb_cols.z)
    ll = sum(logpdf.(BetaBinomial2.(bb_cols.n, mu, phi), bb_cols.c))
    pr = sum(logpdf(Normal(0, 1), ui) for ui in u)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # All-identity layout: no Jacobian.
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@stestset "rk parity kernel ordinary vector cells" begin
    brmi = @brm _kernel_vector_cols begin
        sigma ~ Exponential(1)
        pred ~ kernel(t, intercept, slope, y) do ts, a, b, yy
            mu = a .+ b .* ts
            yy ~ Normal(mu, sigma)
            mu
        end
    end
    plan = BRM._brm_rk_plan(brmi)
    @test plan isa BRM._RKKernelPlan
    @test plan.kernel.n_subjects == 3
    @test plan.kernel.data_columns == [:t, :intercept, :slope, :y]
    @test plan.kernel.slice_kinds == [:vector, :scalar, :scalar, :vector]
    @test plan.kernel.n_timepoints == 4
    @test plan.obs.family === :gaussian
    backend = _parity_backend(brmi)
    @test backend.model.layout.total == 1
    means = [a .+ b .* t for (a, b, t) in zip(_kernel_vector_cols.intercept,
        _kernel_vector_cols.slope, _kernel_vector_cols.t)]
    residuals = reduce(vcat, _kernel_vector_cols.y .- means)
    value = sum(logpdf.(Normal(), residuals)) + logpdf(Exponential(1), 1.0)
    derivative = sum(abs2, residuals) - length(residuals) - 1
    _check_kernel_parity(backend, [0.0], value, derivative)
end

@stestset "rk parity kernel ordinary scalar cells" begin
    brmi = @brm _kernel_scalar_cols begin
        sigma ~ Exponential(1)
        pred ~ kernel(x, intercept, y) do xx, a, yy
            mu = a + 2xx
            yy ~ Normal(mu, sigma)
            mu
        end
    end
    plan = BRM._brm_rk_plan(brmi)
    @test plan isa BRM._RKKernelPlan
    @test plan.kernel.n_subjects == 4
    @test plan.kernel.data_columns == [:x, :intercept, :y]
    @test plan.kernel.slice_kinds == [:scalar, :scalar, :scalar]
    @test plan.kernel.n_timepoints === nothing
    @test plan.obs.family === :gaussian
    backend = _parity_backend(brmi)
    @test backend.model.layout.total == 1
    residuals = _kernel_scalar_cols.y .- (_kernel_scalar_cols.intercept .+ 2 .* _kernel_scalar_cols.x)
    value = sum(logpdf.(Normal(), residuals)) + logpdf(Exponential(1), 1.0)
    derivative = sum(abs2, residuals) - length(residuals) - 1
    _check_kernel_parity(backend, [0.0], value, derivative)
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

@stestset "rk parity ordinal cumulative literal scale" begin
    brmi = @brm _ord_cols begin
        eta ~ 0 + x
        y ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=2.0)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    u = [0.4, -0.2, 0.25]
    nt = constrain(layout, u)
    eta = only(Vector(_parity_population(nt, backend, :eta))) .* _ord_cols.x
    t = Vector(nt.y_thresholds)
    ll = sum(_ref_ordinal(y, e, t, 2.0, :cumulative, :logit)
        for (y, e) in zip(_ord_cols.y, eta))
    pr = logpdf(Normal(), only(_parity_population(nt, backend, :eta))) + sum(logpdf.(Normal(), t))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[3]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[3]
    _check_parity_gradient(backend, u)
end

@stestset "rk parity ordinal cumulative modeled scale" begin
    brmi = @brm _ord_cols begin
        eta ~ 0 + x
        log(disc) ~ 1 + x
        y ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=disc)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 5
    u = [0.4, 0.1, -0.2, 0.25, -0.15]
    nt = constrain(layout, u)
    eta = only(Vector(_parity_population(nt, backend, :eta))) .* _ord_cols.x
    a = Vector(_parity_population(nt, backend, :disc))
    d = exp.(a[1] .+ a[2] .* _ord_cols.x)
    t = Vector(nt.y_thresholds)
    ll = sum(_ref_ordinal(y, e, t, di, :cumulative, :logit)
        for (y, e, di) in zip(_ord_cols.y, eta, d))
    pr = logpdf(Normal(), only(_parity_population(nt, backend, :eta))) + sum(logpdf.(Normal(), a)) +
        sum(logpdf.(Normal(), t))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[5]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[5]
    _check_parity_gradient(backend, u)
end

@stestset "rk parity ordinal cumulative grouping scale" begin
    brmi = @brm _ord_cols begin
        eta ~ 0 + x
        log(disc) ~ 0 + g
        effect(disc, g) ~ Normal(0, 1)
        y ~ Ordinal(Cumulative(), CloglogLink(), eta; discrimination=disc)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 6
    u = [0.4, 0.1, -0.2, 0.3, 0.25, -0.15]
    nt = constrain(layout, u)
    eta = only(Vector(_parity_population(nt, backend, :eta))) .* _ord_cols.x
    levs = sort(unique(_ord_cols.g))
    C = Float64.(_ord_cols.g .== permutedims(levs))
    d = exp.(C * Vector(_parity_population(nt, backend, :disc)))
    t = Vector(nt.y_thresholds)
    ll = sum(_ref_ordinal(y, e, t, di, :cumulative, :cloglog)
        for (y, e, di) in zip(_ord_cols.y, eta, d))
    pr = logpdf(Normal(), only(_parity_population(nt, backend, :eta))) +
        sum(logpdf.(Normal(), _parity_population(nt, backend, :disc))) + sum(logpdf.(Normal(), t))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = log(t[2] - t[1])
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@stestset "rk parity ordinal cumulative column scale" begin
    brmi = @brm _ord_cols begin
        eta ~ 0 + x
        y ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=d)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    u = [0.4, -0.2, 0.25]
    nt = constrain(layout, u)
    eta = only(Vector(_parity_population(nt, backend, :eta))) .* _ord_cols.x
    t = Vector(nt.y_thresholds)
    ll = sum(_ref_ordinal(y, e, t, di, :cumulative, :logit)
        for (y, e, di) in zip(_ord_cols.y, eta, _ord_cols.d))
    pr = logpdf(Normal(), only(_parity_population(nt, backend, :eta))) + sum(logpdf.(Normal(), t))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[3]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[3]
    _check_parity_gradient(backend, u)
end

@stestset "rk parity ordinal stopping per_threshold p=1" begin
    brmi = @brm _ord_cols begin
        eta ~ 0 + x
        y ~ Ordinal(StoppingRatio(), LogitLink(), eta; per_threshold=(z1,))
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 5
    u = [0.4, 0.1, -0.2, 0.25, -0.15]
    nt = constrain(layout, u)
    eta = only(Vector(_parity_population(nt, backend, :eta))) .* _ord_cols.x
    t = Vector(nt.y_thresholds)
    beta = vec(nt.y_threshold_beta)
    E = [_ord_cols.z1[i] * beta[j] for i in 1:9, j in 1:2]
    ll = sum(_ref_ordinal(y, e, t, 1.0, :stopping, :logit, E[i, :])
        for (i, (y, e)) in enumerate(zip(_ord_cols.y, eta)))
    pr = logpdf(Normal(), only(_parity_population(nt, backend, :eta))) + sum(logpdf.(Normal(), t)) +
        sum(logpdf.(Normal(), beta))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@stestset "rk parity ordinal stopping per_threshold p=2" begin
    brmi = @brm _ord_cols begin
        eta ~ 0 + x
        y ~ Ordinal(StoppingRatio(), ProbitLink(), eta;
            per_threshold=(z1, z2))
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 7
    u = [0.4, 0.1, -0.2, 0.25, -0.15, 0.05, -0.1]
    nt = constrain(layout, u)
    eta = only(Vector(_parity_population(nt, backend, :eta))) .* _ord_cols.x
    t = Vector(nt.y_thresholds)
    beta = vec(nt.y_threshold_beta)
    X = hcat(_ord_cols.z1, _ord_cols.z2)
    E = [sum(X[i, c] * beta[(j-1)*2+c] for c in 1:2)
        for i in 1:9, j in 1:2]
    ll = sum(_ref_ordinal(y, e, t, 1.0, :stopping, :probit, E[i, :])
        for (i, (y, e)) in enumerate(zip(_ord_cols.y, eta)))
    pr = logpdf(Normal(), only(_parity_population(nt, backend, :eta))) + sum(logpdf.(Normal(), t)) +
        sum(logpdf.(Normal(), beta))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@stestset "rk parity ordinal stopping scale plus stage" begin
    brmi = @brm _ord_cols begin
        eta ~ 0 + x
        log(disc) ~ 1 + x
        y ~ Ordinal(StoppingRatio(), LogitLink(), eta;
            discrimination=disc, per_threshold=(z1,))
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 7
    u = [0.4, 0.1, -0.2, 0.25, -0.15, 0.05, -0.1]
    nt = constrain(layout, u)
    eta = only(Vector(_parity_population(nt, backend, :eta))) .* _ord_cols.x
    a = Vector(_parity_population(nt, backend, :disc))
    d = exp.(a[1] .+ a[2] .* _ord_cols.x)
    t = Vector(nt.y_thresholds)
    beta = vec(nt.y_threshold_beta)
    E = [_ord_cols.z1[i] * beta[j] for i in 1:9, j in 1:2]
    ll = sum(_ref_ordinal(_ord_cols.y[i], eta[i], t, d[i], :stopping,
        :logit, E[i, :]) for i in 1:9)
    pr = logpdf(Normal(), only(_parity_population(nt, backend, :eta))) + sum(logpdf.(Normal(), a)) +
        sum(logpdf.(Normal(), t)) + sum(logpdf.(Normal(), beta))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@stestset "rk parity ordinal K=1 modeled scale" begin
    brmi = @brm _ord_cols1 begin
        eta ~ 0 + x
        log(disc) ~ 1 + x
        y ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=disc)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    u = [0.4, 0.1, -0.2]
    nt = constrain(layout, u)
    # Zero-information likelihood (SB's K=1 degeneration), live prior.
    @test _rk_query(backend, :likelihood, u) == 0.0
    pr = logpdf(Normal(), only(_parity_population(nt, backend, :eta))) +
        sum(logpdf.(Normal(), _parity_population(nt, backend, :disc)))
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ pr
    _check_parity_gradient(backend, u)
end

@stestset "rk parity ordinal K=1 per_threshold" begin
    brmi = @brm _ord_cols1 begin
        eta ~ 0 + x
        y ~ Ordinal(StoppingRatio(), LogitLink(), eta; per_threshold=(z1,))
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 1
    u = [0.4]
    nt = constrain(layout, u)
    # Zero stages: zero-information likelihood, eta prior only.
    @test _rk_query(backend, :likelihood, u) == 0.0
    pr = logpdf(Normal(), only(_parity_population(nt, backend, :eta)))
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

function _ref_hsgp_floors(xs, Ks, cs; iso)
    floors = map(eachindex(xs)) do j
        Ks[j] == 1 && return 0.0
        centered = xs[j] .- sum(xs[j]) / length(xs[j])
        L = cs[j] * maximum(abs, centered)
        (4L / pi) * sqrt(log(100) / (Ks[j]^2 - 1))
    end
    iso ? [maximum(floors)] : floors
end

function _ref_hsgp_periodic_floor(K)
    K == 1 && return 0.0
    lo, hi = 1e-4, 1e6
    for _ in 1:160
        rho = sqrt(lo * hi)
        a = inv(rho^2)
        ratio = besselix(K, a) / besselix(1, a)
        ratio > 1e-4 ? (lo = rho) : (hi = rho)
    end
    sqrt(lo * hi)
end

function _ref_hsgp_prior(a, sig, rhos, sigh, beta, floors)
    return logpdf(Normal(0, 5), a) + logpdf(Exponential(1), sig) +
        sum(logpdf(LogNormal(), r) for r in rhos) +
        logpdf(LogNormal(0, 1), sigh) + sum(logpdf.(Normal(0, 1), beta))
end

# Periodic smooth vector: cos/sine columns at harmonics of `2pi/P`
# (SB `_brm_apply_hsgp_periodic` element order verbatim) + the Stan
# `brm_hsgp_periodic_sqrt_spd` weights in the never-overflow scaled
# form `sigma*sqrt(2*besselix(j, a))`, `a = 1/rho^2`.
function _ref_hsgp_periodic_muv(x, K, period, rho, sigh, beta)
    n = length(x)
    w0 = 2pi / period
    PHI = zeros(n, 2K)
    for j in 1:K, i in 1:n
        PHI[i, j] = cos(w0 * j * x[i])
        PHI[i, K + j] = sin(w0 * j * x[i])
    end
    a = 1 / (rho * rho)
    s = [sigh * sqrt(2 * besselix(j, a)) for j in [1:K; 1:K]]
    return PHI * (s .* beta)
end

@stestset "rk parity hsgp 1d" begin
    brmi = @brm _parity_cols_hsgp begin
        mu ~ 1 + hsgp(x; k = 4)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 8
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    f = _ref_hsgp_muv([_parity_cols_hsgp.x], [4], [1.5], [nt.hsgp_x.rho],
        nt.hsgp_x.sigma, Vector(nt.hsgp_x.z); iso = true)
    ll = sum(logpdf.(Normal.(_parity_population(nt, backend, :mu)[1] .+ f, nt.sigma), _parity_cols_hsgp.y))
    floors = _ref_hsgp_floors([_parity_cols_hsgp.x], [4], [1.5]; iso=true)
    pr = _ref_hsgp_prior(_parity_population(nt, backend, :mu)[1], nt.sigma, [nt.hsgp_x.rho], nt.hsgp_x.sigma,
        Vector(nt.hsgp_x.z), floors)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = log(nt.hsgp_x.rho - only(floors)) + log(nt.hsgp_x.sigma) + log(nt.sigma)
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@stestset "rk parity hsgp aniso" begin
    brmi = @brm _parity_cols_hsgp begin
        mu ~ 1 + hsgp(x, z; k = (4, 3), c = (1.5, 2.0), iso = false)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 17
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    rhos = [nt.hsgp_x_z.rho1, nt.hsgp_x_z.rho2]
    f = _ref_hsgp_muv([_parity_cols_hsgp.x, _parity_cols_hsgp.z], [4, 3],
        [1.5, 2.0], rhos, nt.hsgp_x_z.sigma, Vector(nt.hsgp_x_z.z);
        iso = false)
    ll = sum(logpdf.(Normal.(_parity_population(nt, backend, :mu)[1] .+ f, nt.sigma), _parity_cols_hsgp.y))
    floors = _ref_hsgp_floors([_parity_cols_hsgp.x, _parity_cols_hsgp.z],
        [4, 3], [1.5, 2.0]; iso=false)
    pr = _ref_hsgp_prior(_parity_population(nt, backend, :mu)[1], nt.sigma, rhos, nt.hsgp_x_z.sigma,
        Vector(nt.hsgp_x_z.z), floors)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = sum(log.(rhos .- floors)) + log(nt.hsgp_x_z.sigma) + log(nt.sigma)
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@stestset "rk parity hsgp k1" begin
    brmi = @brm _parity_cols_hsgp begin
        mu ~ 1 + hsgp(x; k = 1)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 5
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    f = _ref_hsgp_muv([_parity_cols_hsgp.x], [1], [1.5], [nt.hsgp_x.rho],
        nt.hsgp_x.sigma, Vector(nt.hsgp_x.z); iso = true)
    ll = sum(logpdf.(Normal.(_parity_population(nt, backend, :mu)[1] .+ f, nt.sigma), _parity_cols_hsgp.y))
    floors = _ref_hsgp_floors([_parity_cols_hsgp.x], [1], [1.5]; iso=true)
    pr = _ref_hsgp_prior(_parity_population(nt, backend, :mu)[1], nt.sigma, [nt.hsgp_x.rho], nt.hsgp_x.sigma,
        Vector(nt.hsgp_x.z), floors)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = log(nt.hsgp_x.rho) + log(nt.hsgp_x.sigma) + log(nt.sigma)
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

@stestset "rk parity hsgp periodic" begin
    brmi = @brm _parity_cols_hsgp begin
        mu ~ 1 + hsgp(x; k = 4, cov = :periodic, period = 2.0)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 12
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    f = _ref_hsgp_periodic_muv(_parity_cols_hsgp.x, 4, 2.0, nt.hsgp_x.rho,
        nt.hsgp_x.sigma, Vector(nt.hsgp_x.z))
    ll = sum(logpdf.(Normal.(_parity_population(nt, backend, :mu)[1] .+ f, nt.sigma), _parity_cols_hsgp.y))
    floors = [_ref_hsgp_periodic_floor(4)]
    pr = _ref_hsgp_prior(_parity_population(nt, backend, :mu)[1], nt.sigma, [nt.hsgp_x.rho], nt.hsgp_x.sigma,
        Vector(nt.hsgp_x.z), floors)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    jac = log(nt.hsgp_x.rho - only(floors)) + log(nt.hsgp_x.sigma) + log(nt.sigma)
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    _check_parity_gradient(backend, u)
end

# Spline density references retain the independently measured Stan points.
# Default scales retain the original support-restricted Normal density.
@stestset "rk parity spline s default" begin
    # Pair-agreed Xoshiro(7207) n=80 probe, default k=10 basis.
    (; brmi, x, y) = spline_s_parity_case()
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 12
    # Probe in layout order (pinned by the signature above):
    # [mu.Intercept, log(sigma), b_fixed.1, b_raw.1..8, log(sd)].
    # Fold SB's flat constant coefficient (-0.2) into its intercept (-0.3).
    u = [-0.5, -0.25, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1,
        0.15, 0.2, 0.25, 0.3]
    # Values route through the bound translated plan (see the helper):
    # the kernel takes host-materialized basis columns.
    u = u[[12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    translated = _rk_translated(backend)
    nt = constrain(layout, u)
    sig = nt.sigma
    sd = nt.s_x.sd1
    Xnull, Zpen = BRM._brm_apply_spline(BRM._brm_fit_spline(x; k=10), x)
    mu_hat = _parity_population(nt, backend, :mu)[1] .+ Xnull[:, 2:2] * nt.s_x.fixed .+ Zpen * (sd .* nt.s_x.raw1)
    ll = sum(logpdf.(Normal.(mu_hat, sig), y))
    pr = logpdf(Normal(0, 5), _parity_population(nt, backend, :mu)[1]) + logpdf(Normal(), sd) +
        sum(logpdf.(Normal(), nt.s_x.raw1)) + logpdf(Exponential(1), sig)
    jac = log(sd) + log(sig)
    @test _rk_query_translated(backend, translated, :likelihood, u) ≈ ll
    @test _rk_query_translated(backend, translated, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ jac
    @test _rk_query_translated(backend, translated, :sampler, u) ≈ ll + pr + jac
    # Independent BridgeStan anchor at SB's original 13-coordinate point
    # (test/spline_sb_parity.jl). Folding its constant into the intercept
    # preserves the likelihood; only the Normal(0,5) intercept prior moves.
    intercept_bridge = logpdf(Normal(0, 5), nt.mu_Intercept) - logpdf(Normal(0, 5), -0.3)
    @test _rk_query_translated(backend, translated, :sampler, u) ≈
        SPLINE_S_SB_ANCHOR + intercept_bridge atol=1e-9
    problem = BRM.rk_logdensity_problem(backend;
        ad_backend = _PARITY_BACKEND, u0 = u)
    value, grad = LogDensityProblems.logdensity_and_gradient(problem, u)
    @test value ≈ _rk_query_translated(backend, translated, :sampler, u)
    @test all(isfinite, grad)
    @test grad ≈
        _findiff_grad(w -> LogDensityProblems.logdensity(problem, w), u) rtol = 1e-5 atol = 1e-7
end

@stestset "rk parity spline t2 k33" begin
    # Pair-agreed Xoshiro(7208) n=80 probe, k=(3,3) tensor basis.
    (; brmi, x, z, y) = spline_t2_parity_case()
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 13
    # Probe in layout order (pinned by the signature above).
    u = [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1,
        0.15, 0.2, 0.25, 0.3]
    # Values route through the bound translated plan (see the helper):
    # the kernel takes host-materialized basis columns.
    u = u[[11, 12, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    translated = _rk_translated(backend)
    nt = constrain(layout, u)
    sig = nt.sigma
    sd = [nt.t2_x_z.sd1, nt.t2_x_z.sd2, nt.t2_x_z.sd3]
    Xfixed, Zrr, Zrn, Znr = BRM._brm_apply_t2(
        BRM._brm_fit_t2(x, z; k=(3, 3)), x, z)
    mu_hat = _parity_population(nt, backend, :mu)[1] .+ Xfixed * nt.t2_x_z.fixed .+
        Zrr * (sd[1] .* nt.t2_x_z.raw1) .+
        Zrn * (sd[2] .* nt.t2_x_z.raw2) .+
        Znr * (sd[3] .* nt.t2_x_z.raw3)
    ll = sum(logpdf.(Normal.(mu_hat, sig), y))
    pr = logpdf(Normal(0, 5), _parity_population(nt, backend, :mu)[1]) + sum(logpdf.(Normal(), sd)) +
        sum(logpdf.(Normal(), nt.t2_x_z.raw1)) +
        sum(logpdf.(Normal(), nt.t2_x_z.raw2)) +
        sum(logpdf.(Normal(), nt.t2_x_z.raw3)) +
        logpdf(Exponential(1), sig)
    jac = sum(log.(sd)) + log(sig)
    @test _rk_query_translated(backend, translated, :likelihood, u) ≈ ll
    @test _rk_query_translated(backend, translated, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ jac
    @test _rk_query_translated(backend, translated, :sampler, u) ≈ ll + pr + jac
    # Re-derived with canonical signs by test/spline_sb_parity.jl; t2 has
    # the same layout in both backends and needs no intercept-prior bridge.
    @test _rk_query_translated(backend, translated, :sampler, u) ≈
        SPLINE_T2_SB_ANCHOR atol=1e-9
    problem = BRM.rk_logdensity_problem(backend;
        ad_backend = _PARITY_BACKEND, u0 = u)
    value, grad = LogDensityProblems.logdensity_and_gradient(problem, u)
    @test value ≈ _rk_query_translated(backend, translated, :sampler, u)
    @test all(isfinite, grad)
    @test grad ≈
        _findiff_grad(w -> LogDensityProblems.logdensity(problem, w), u) rtol = 1e-5 atol = 1e-7
end

# Exact-GP parity cases (pair term-gp): the committed oracles are
# Distributions.jl loops over the constrained point; the SB-point
# comparison (same models, SB brief values) rides the verdict probe,
# not the committed suite.
_parity_cols_gp = (;
    x = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
    y = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
)

# Exact-GP reference: Stan `gp_exp_quad_cov` + diagonal jitter verbatim
# (1d iso): K[i,j] = s^2 exp(-(x_i-x_j)^2/(2 rho^2)), f = L z.
function _ref_gp_exp_quad_f(x, rho, sigma_gp, z; jitter=1e-9)
    n = length(x)
    K = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        K[i, j] = sigma_gp^2 * exp(-(x[i] - x[j])^2 / (2 * rho^2))
    end
    for i in 1:n
        K[i, i] += jitter
    end
    return cholesky(Symmetric(K)).L * z
end

@stestset "rk parity exact gp iso" begin
    brmi = @brm _parity_cols_gp begin
        mu ~ 1 + gp(x)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 10
    # The formula intercept, covariance parameters and latent vector each
    # have ordinary explicit declarations in the emitted source.
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    @test !hasproperty(nt, :y_loc)
    f = _ref_gp_exp_quad_f(_parity_cols_gp.x, nt.f_gp.rho, nt.f_gp.sigma,
        Vector(nt.f_gp.z))
    ll = sum(logpdf.(Normal.(nt.mu_Intercept .+ f, nt.sigma), _parity_cols_gp.y))
    pr = logpdf(Normal(0, 5), nt.mu_Intercept) +
        logpdf(Exponential(1), nt.sigma) +
        logpdf(LogNormal(0, 1), nt.f_gp.rho) +
        logpdf(LogNormal(0, 1), nt.f_gp.sigma) +
        sum(logpdf.(Normal(0, 1), nt.f_gp.z))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # Jacobian: rho/sigma_gp/sigma exps (u[1], u[2], u[4]); the empty
    # coefficient and identity mu_b1/z_gp contribute nothing.
    @test logjac(layout, u) ≈ u[1] + u[2] + u[4]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[1] + u[2] + u[4]
    _check_parity_gradient(backend, u)
end

# Periodic-GP reference: Stan `gp_periodic_cov` + diagonal jitter verbatim
# (1d iso): K[i,j] = s^2 exp(-2 sin^2(pi |x_i-x_j|/period)/rho^2), f = L z.
function _ref_gp_periodic_f(x, rho, sigma_gp, z; period=1.0, jitter=1e-9)
    n = length(x)
    K = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        d = abs(x[i] - x[j])
        K[i, j] = sigma_gp^2 * exp(-2 * sin(pi * d / period)^2 / rho^2)
    end
    for i in 1:n
        K[i, i] += jitter
    end
    return cholesky(Symmetric(K)).L * z
end

@stestset "rk parity periodic gp iso" begin
    brmi = @brm _parity_cols_gp begin
        mu ~ 1 + gp(x; cov=:periodic, period=1.0)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 10
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    @test !hasproperty(nt, :y_loc)
    f = _ref_gp_periodic_f(_parity_cols_gp.x, nt.f_gp.rho, nt.f_gp.sigma,
        Vector(nt.f_gp.z); period=1.0)
    ll = sum(logpdf.(Normal.(nt.mu_Intercept .+ f, nt.sigma), _parity_cols_gp.y))
    pr = logpdf(Normal(0, 5), nt.mu_Intercept) +
        logpdf(Exponential(1), nt.sigma) +
        logpdf(LogNormal(0, 1), nt.f_gp.rho) +
        logpdf(LogNormal(0, 1), nt.f_gp.sigma) +
        sum(logpdf.(Normal(0, 1), nt.f_gp.z))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[1] + u[2] + u[4]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[1] + u[2] + u[4]
    _check_parity_gradient(backend, u)
end

# Mixture parity cases (pair fam-mixture, RK 49ebaf1): the committed
# oracles are Distributions.jl loops over the constrained point; the
# SB-point comparison (same models, SB brief values) rides the verdict
# probe, not the committed suite.
@stestset "rk parity mixture gaussian" begin
    df = (; y=[-2.0, -1.8, 1.9, 2.2])
    brmi = @brm df begin
        mu1 ~ Normal(-2, 0.1)
        mu2 ~ Normal(2, 0.1)
        log(sigma) ~ 1
        y ~ MixtureModel([Normal(mu1, sigma), Normal(mu2, sigma)], [0.4, 0.6])
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    mixture = MixtureModel(
        [Normal(_parity_population(nt, backend, :mu1), exp(_parity_population(nt, backend, :sigma)[1])), Normal(_parity_population(nt, backend, :mu2), exp(_parity_population(nt, backend, :sigma)[1]))],
        [0.4, 0.6])
    ll = sum(logpdf.(mixture, df.y))
    pr = logpdf(Normal(-2, 0.1), _parity_population(nt, backend, :mu1)) + logpdf(Normal(2, 0.1), _parity_population(nt, backend, :mu2)) +
        logpdf(Normal(), _parity_population(nt, backend, :sigma)[1])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@stestset "rk parity mixture poisson" begin
    df = (; y=[0, 1, 3, 5, 2])
    brmi = @brm df begin
        lambda1 ~ Exponential(1)
        lambda2 ~ Exponential(1)
        y ~ MixtureModel([Poisson(lambda1), Poisson(lambda2)], [0.3, 0.7])
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 2
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

@stestset "rk parity mi() missing-response obs-only likelihood" begin
    # Case A (decision 05aemvx): explicit observed slices; the likelihood sees
    # observed rows only while predictors stay full-length. Reference is
    # an independent Distributions.jl hand oracle; the plain obs-only twin
    # cross-checks the same physical coefficients and observation values.
    cols = (; x=[-1.0, 0.5, 2.0, 0.25],
              y=Union{Missing,Float64}[0.2, missing, -0.4, missing])
    brmi = @brm cols begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        mi(y) ~ Normal(mu, sigma)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    u = [0.0, 0.05, -0.05]
    nt = constrain(layout, u)
    mu_o = _parity_population(nt, backend, :mu)[1] .+ _parity_population(nt, backend, :mu)[2] .* [-1.0, 2.0]
    ll = sum(logpdf.(Normal.(mu_o, nt.sigma), [0.2, -0.4]))
    pr = logpdf(Normal(0, 1), _parity_population(nt, backend, :mu)[1]) +
        logpdf(Normal(0, 1), _parity_population(nt, backend, :mu)[2]) +
        logpdf(Exponential(2), nt.sigma)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[3]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[3]
    _check_parity_gradient(backend, u)
    twin = _parity_backend(@brm (; x=[-1.0, 2.0], y=[0.2, -0.4]) begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end)
    # The observed-only twin retains the same coordinate names and point.
    u_twin = copy(u)
    @test _rk_query(twin, :posterior, u_twin) ≈
        _rk_query(backend, :posterior, u)
end

@stestset "rk parity mi() + horseshoe keeps the shrinkage rows" begin
    # The `mi()` plan patch once rebuilt the thin-layer plan from a
    # hand-copied keyword list that omitted `horseshoe_priors`, so bind
    # failed with `[plan] no prior for (mu, Intercept)` (snag
    # rk-ext-patch-dro-46d976f3). Reference: the obs-only twin with the
    # same horseshoe, compared by coordinate name.
    cols = (; x=[-1.0, -0.5, 0.0, 0.5, 1.0], z=[0.3, -0.2, 0.9, -0.7, 0.1],
              y=Union{Missing,Float64}[0.2, missing, -0.4, missing, 0.7])
    brmi = @brm cols begin
        sigma ~ Exponential(2)
        mu ~ 1 + x + z
        effect(mu, x) ~ Horseshoe()
        mi(y) ~ Normal(mu, sigma)
    end
    backend = _parity_backend(brmi)
    translated = _rk_translated(backend)
    source = sprint(Base.show_unquoted, BRM._rk_emit_ast(backend.plan).main)
    @test occursin("mu_tau ~ HalfCauchy", source)
    @test occursin("y[Jobs_y]", source)
    obs = [1, 3, 5]
    twin = _parity_backend(@brm (; x=cols.x[obs], z=cols.z[obs],
            y=Float64[0.2, -0.4, 0.7]) begin
        sigma ~ Exponential(2)
        mu ~ 1 + x + z
        effect(mu, x) ~ Horseshoe()
        y ~ Normal(mu, sigma)
    end)
    names = coordinate_names(backend.model.layout)
    twin_names = coordinate_names(twin.model.layout)
    @test sort(names) == sort(twin_names)
    byname = Dict(n => 0.1 * i - 0.25 for (i, n) in enumerate(sort(names)))
    u = Float64[byname[n] for n in names]
    u_twin = Float64[byname[n] for n in twin_names]
    @test _rk_query(backend, :posterior, u) ≈
        _rk_query(twin, :posterior, u_twin)
    _check_parity_gradient(backend, u)

end

@stestset "rk parity interval-gaussian literal upper" begin
    # Interval-censored Gaussian: each row contributes
    # log(Phi(hi) - Phi(y)) (the response is the lower endpoint).
    # Independent Distributions.jl cdf-difference oracle.
    cols = (;
        x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        y=[0.2, 0.8, 1.1, 0.4, 1.5, 0.9],
    )
    brmi = @brm cols begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ interval_censored(Normal(mu, s); upper=2.0)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    u = [0.2, -0.3, 0.1]
    nt = constrain(layout, u)
    mu_o = _parity_population(nt, backend, :mu)[1] .+ _parity_population(nt, backend, :mu)[2] .* cols.x
    ll = sum(log.(cdf.(Normal.(mu_o, nt.s), 2.0) .-
        cdf.(Normal.(mu_o, nt.s), cols.y)))
    pr = logpdf(Normal(0, 1), _parity_population(nt, backend, :mu)[1]) +
        logpdf(Normal(0, 1), _parity_population(nt, backend, :mu)[2]) +
        logpdf(Exponential(1), nt.s)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[3]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[3]
    _check_parity_gradient(backend, u)
end

@stestset "rk parity interval-gaussian column upper" begin
    # Rowwise upper endpoints from a data column.
    cols = (;
        x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        y=[0.2, 0.8, 1.1, 0.4, 1.5, 0.9],
        hi=[2.0, 1.5, 2.5, 1.0, 2.0, 1.2],
    )
    brmi = @brm cols begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ interval_censored(Normal(mu, s); upper=hi)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    u = [0.2, -0.3, 0.1]
    nt = constrain(layout, u)
    mu_o = _parity_population(nt, backend, :mu)[1] .+ _parity_population(nt, backend, :mu)[2] .* cols.x
    ll = sum(log.(cdf.(Normal.(mu_o, nt.s), cols.hi) .-
        cdf.(Normal.(mu_o, nt.s), cols.y)))
    pr = logpdf(Normal(0, 1), _parity_population(nt, backend, :mu)[1]) +
        logpdf(Normal(0, 1), _parity_population(nt, backend, :mu)[2]) +
        logpdf(Exponential(1), nt.s)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[3]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[3]
    _check_parity_gradient(backend, u)
end

@stestset "rk parity interval-poisson literal upper" begin
    # Interval-censored Poisson: each row contributes
    # log(F(hi) - F(c)) — the response is the OPEN lower endpoint of
    # (c, hi] (brm-use contract; SB's `interval_evidence_impl_poisson`
    # computes `log_diff_exp(poisson_lcdf(hi), poisson_lcdf(lo))`
    # with no -1 shift).
    cols = (;
        x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        c=[0, 2, 1, 3, 0, 1],
    )
    brmi = @brm cols begin
        log(lambda) ~ 1 + x
        c ~ interval_censored(Poisson(lambda); upper=5)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 2
    u = [0.2, -0.3]
    nt = constrain(layout, u)
    lam = exp.(_parity_population(nt, backend, :lambda)[1] .+ _parity_population(nt, backend, :lambda)[2] .* cols.x)
    ll = sum(log.(cdf.(Poisson.(lam), 5) .-
        cdf.(Poisson.(lam), cols.c)))
    pr = logpdf(Normal(0, 1), _parity_population(nt, backend, :lambda)[1]) +
        logpdf(Normal(0, 1), _parity_population(nt, backend, :lambda)[2])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@stestset "rk parity interval-poisson column upper" begin
    # Rowwise integer-valued upper endpoints from a data column.
    cols = (;
        x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        c=[0, 2, 1, 3, 0, 1],
        hi=[3, 4, 2, 5, 1, 3],
    )
    brmi = @brm cols begin
        log(lambda) ~ 1 + x
        c ~ interval_censored(Poisson(lambda); upper=hi)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 2
    u = [0.2, -0.3]
    nt = constrain(layout, u)
    lam = exp.(_parity_population(nt, backend, :lambda)[1] .+ _parity_population(nt, backend, :lambda)[2] .* cols.x)
    ll = sum(log.(cdf.(Poisson.(lam), cols.hi) .-
        cdf.(Poisson.(lam), cols.c)))
    pr = logpdf(Normal(0, 1), _parity_population(nt, backend, :lambda)[1]) +
        logpdf(Normal(0, 1), _parity_population(nt, backend, :lambda)[2])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@stestset "rk parity von-Mises circular kappa submodel" begin
    v_cols = (;
        x=[-1.0, -0.25, 0.5, 1.0],
        y=[-2.8, -0.4, 1.1, 2.9],
    )
    brmi = @brm v_cols begin
        mu ~ 1 + x
        log(kappa) ~ 1
        y ~ CircularVonMises(mu, kappa; interval=(-pi, pi))
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    u = [0.5, -0.25, 0.3]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
    mu = b[1] .+ b[2] .* v_cols.x
    kap = exp(only(_parity_population(nt, backend, :kappa)))
    ll = sum(logpdf.(BRM.CircularVonMises.(mu, kap;
        interval=(-Float64(pi), Float64(pi))), v_cols.y))
    pr = logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2]) +
        logpdf(Normal(0, 1), only(_parity_population(nt, backend, :kappa)))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # All-identity layout: no Jacobian.
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@stestset "rk parity von-Mises exact sampled kappa" begin
    v_cols = (;
        x=[-1.0, -0.25, 0.5, 1.0],
        y=[-2.0, 0.3, 1.5, -1.2],
    )
    brmi = @brm v_cols begin
        mu ~ 1 + x
        k ~ LogNormal(0, 1)
        y ~ VonMises(mu, k)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    u = [0.5, -0.25, 0.2]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
    mu = b[1] .+ b[2] .* v_cols.x
    ll = sum(logpdf.(VonMises.(mu, nt.k), v_cols.y))
    pr = logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2]) +
        logpdf(LogNormal(0, 1), nt.k)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[3]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[3]
    _check_parity_gradient(backend, u)
end

@stestset "rk parity von-Mises circular literal kappa" begin
    v_cols = (;
        x=[-1.0, -0.25, 0.5, 1.0],
        y=[0.5, 2.0, 6.0, 1.0],
    )
    brmi = @brm v_cols begin
        mu ~ 1 + x
        y ~ CircularVonMises(mu, 1.7; interval=(0.0, 6.283185307179586))
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 2
    u = [1.0, -0.5]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
    mu = b[1] .+ b[2] .* v_cols.x
    ll = sum(logpdf.(BRM.CircularVonMises.(mu, 1.7;
        interval=(0.0, 6.283185307179586)), v_cols.y))
    pr = logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # All-identity layout: no Jacobian.
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

@stestset "rk parity beta modeled kappa" begin
    # The nuisance-kappa P2 shape: logit-link location + log-link kappa
    # submodel over the shared N=6 probe columns.
    p2_cols = (;
        x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        z=[1.0, 0.5, -0.5, 1.5, 0.0, -1.0],
        prop=[0.2, 0.7, 0.4, 0.6, 0.3, 0.5],
    )
    brmi = @brm p2_cols begin
        logit(mu) ~ 1 + x
        log(kappa) ~ 1 + z
        prop ~ Beta(mu * kappa, (1 - mu) * kappa)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 4
    u = [0.2, -0.1, 0.3, 0.15]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
    c = _parity_population(nt, backend, :kappa)
    mu = logistic.(b[1] .+ b[2] .* p2_cols.x)
    kap = exp.(c[1] .+ c[2] .* p2_cols.z)
    ll = sum(logpdf.(Beta.(mu .* kap, (1 .- mu) .* kap), p2_cols.prop))
    pr = sum(logpdf.(Normal(0, 1), u))
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # All-identity layout: no Jacobian.
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end
# term-multimembership oracles (re-derived from the SB submodels, never
# the emitter): flat row-major mm index/weights over the sort-ordered
# union [a,b,c]; per-group gr strata [1,1,2,2].
_ref_mm_gidx() = [1, 2, 1, 3, 2, 3, 3, 1, 2, 1, 1, 2]
_ref_mm_weights() =
    [2 / 3, 1 / 3, 1 / 2, 1 / 2, 0.0, 1.0, 1 / 3, 2 / 3, 3 / 4, 1 / 4, 1 / 2, 1 / 2]
_ref_mm_weights_raw() =
    [2.0, 1.0, 1.0, 1.0, 0.0, 3.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0]
_ref_mm_gather(b, gidx, w) =
    [w[2i-1] * b[gidx[2i-1]] + w[2i] * b[gidx[2i]] for i in 1:6]

@stestset "rk parity mm intercept (equal weights)" begin
    brmi = @brm _parity_cols_mm begin
        loc ~ 1 + (1 | mm(g1, g2))
        y ~ Normal(loc, sigma)
        effect(loc, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 6
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    beta, sigma = _parity_population(nt, backend, :loc)[1], nt.sigma
    b = only(_parity_ranef(nt, backend).sd) .* _parity_ranef(nt, backend).zflat
    r = _ref_mm_gather(b, _ref_mm_gidx(), fill(0.5, 12))
    ll = sum(logpdf(Normal(beta + r[i], sigma), _parity_cols_mm.y[i])
        for i in 1:6)
    pr = logpdf(Normal(0, 5), beta) + logpdf(Exponential(1), sigma) +
        logpdf(LogNormal(), only(_parity_ranef(nt, backend).sd)) +
        sum(logpdf(Normal(0, 1), v) for v in _parity_ranef(nt, backend).zflat)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + logjac(layout, u)
    _check_parity_gradient(backend, u)
end

@stestset "rk parity mm intercept (weighted)" begin
    brmi = @brm _parity_cols_mm begin
        loc ~ 1 + (1 | mm(g1, g2; weights = (w1, w2)))
        y ~ Normal(loc, sigma)
        effect(loc, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 6
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    beta, sigma = _parity_population(nt, backend, :loc)[1], nt.sigma
    b = only(_parity_ranef(nt, backend).sd) .*
        _parity_ranef(nt, backend).zflat
    r = _ref_mm_gather(b, _ref_mm_gidx(), _ref_mm_weights())
    ll = sum(logpdf(Normal(beta + r[i], sigma), _parity_cols_mm.y[i])
        for i in 1:6)
    pr = logpdf(Normal(0, 5), beta) + logpdf(Exponential(1), sigma) +
        logpdf(LogNormal(), only(_parity_ranef(nt, backend).sd)) +
        sum(logpdf(Normal(0, 1), v) for v in _parity_ranef(nt, backend).zflat)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + logjac(layout, u)
    _check_parity_gradient(backend, u)
end

@stestset "rk parity mm intercept (raw weights)" begin
    brmi = @brm _parity_cols_mm begin
        loc ~ 1 + (1 | mm(g1, g2; weights = (w1, w2), normalize = false))
        y ~ Normal(loc, sigma)
        effect(loc, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 6
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    beta, sigma = _parity_population(nt, backend, :loc)[1], nt.sigma
    b = only(_parity_ranef(nt, backend).sd) .*
        _parity_ranef(nt, backend).zflat
    r = _ref_mm_gather(b, _ref_mm_gidx(), _ref_mm_weights_raw())
    ll = sum(logpdf(Normal(beta + r[i], sigma), _parity_cols_mm.y[i])
        for i in 1:6)
    pr = logpdf(Normal(0, 5), beta) + logpdf(Exponential(1), sigma) +
        logpdf(LogNormal(), only(_parity_ranef(nt, backend).sd)) +
        sum(logpdf(Normal(0, 1), v)
            for v in _parity_ranef(nt, backend).zflat)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + logjac(layout, u)
    _check_parity_gradient(backend, u)
end

@stestset "rk parity mm correlated" begin
    brmi = @brm _parity_cols_mm begin
        loc ~ 1 + (1 + x | mm(g1, g2; weights = (w1, w2)))
        y ~ Normal(loc, sigma)
        effect(loc, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 11
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    beta, sigma = _parity_population(nt, backend, :loc)[1], nt.sigma
    L = _parity_ranef(nt, backend).L
    tau = _parity_ranef(nt, backend).sd
    zf = _parity_ranef(nt, backend).zflat
    zm = reshape(zf, 2, 3)
    b = Matrix{Float64}(undef, 3, 2)
    for g in 1:3, k in 1:2
        b[g, k] = tau[k] * (L[k, 1] * zm[1, g] + L[k, 2] * zm[2, g])
    end
    gidx, w = _ref_mm_gidx(), _ref_mm_weights()
    Z = hcat(ones(6), _parity_cols_mm.x)
    r = [w[2i-1] * (Z[i, 1] * b[gidx[2i-1], 1] + Z[i, 2] * b[gidx[2i-1], 2]) +
         w[2i] * (Z[i, 1] * b[gidx[2i], 1] + Z[i, 2] * b[gidx[2i], 2])
        for i in 1:6]
    ll = sum(logpdf(Normal(beta + r[i], sigma), _parity_cols_mm.y[i])
        for i in 1:6)
    pr = logpdf(Normal(0, 5), beta) + logpdf(Exponential(1), sigma) +
        _ref_lkj_k2_eta1(L) + sum(logpdf(Normal(0, 1), v) for v in tau) +
        sum(logpdf(Normal(0, 1), v) for v in zf)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + logjac(layout, u)
    _check_parity_gradient(backend, u)
end

@stestset "rk parity mm slope (vacuous LKJ)" begin
    # SB routes even a lone mm slope through the correlated draws
    # (vacuous 1x1 LKJ + normalizer) — never the plain slope geometry.
    brmi = @brm _parity_cols_mm begin
        loc ~ 1 + (0 + x | mm(g1, g2; weights = (w1, w2)))
        y ~ Normal(loc, sigma)
        effect(loc, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 6
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    beta, sigma = _parity_population(nt, backend, :loc)[1], nt.sigma
    tau = only(_parity_ranef(nt, backend).sd)
    zf = _parity_ranef(nt, backend).zflat
    b = tau .* zf
    x = _parity_cols_mm.x
    gidx, w = _ref_mm_gidx(), _ref_mm_weights()
    r = [x[i] * (w[2i-1] * b[gidx[2i-1]] + w[2i] * b[gidx[2i]]) for i in 1:6]
    ll = sum(logpdf(Normal(beta + r[i], sigma), _parity_cols_mm.y[i])
        for i in 1:6)
    # The vacuous 1x1 LKJ contributes 0 (pinned by the value match).
    pr = logpdf(Normal(0, 5), beta) + logpdf(Exponential(1), sigma) +
        logpdf(Normal(0, 1), tau) +
        sum(logpdf(Normal(0, 1), v) for v in zf)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + logjac(layout, u)
    _check_parity_gradient(backend, u)
end

@stestset "rk parity gr intercept (stratified)" begin
    # Thin-layer `constrain` refuses stratified draws by design
    # (log-density-only slice), so the oracle unconstrains by hand from
    # the documented layout order [beta, sigma, tau_s1, tau_s2, z_g].
    brmi = @brm _parity_cols_gr begin
        loc ~ 1 + (1 | gr(g, by = b))
        y ~ Normal(loc, sigma)
        effect(loc, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 8
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    beta, sigma = nt.loc_Intercept, nt.sigma
    draws = _parity_ranef(nt, backend)
    tau, z = vec(draws.sd), vec(draws.z)
    sidx = [1, 1, 2, 2]
    gidx = [1, 1, 2, 3, 3, 4]
    b = [tau[sidx[g]] * z[g] for g in 1:4]
    r = [b[gidx[i]] for i in 1:6]
    ll = sum(logpdf(Normal(beta + r[i], sigma), _parity_cols_gr.y[i])
        for i in 1:6)
    pr = logpdf(Normal(0, 5), beta) + logpdf(Exponential(1), sigma) +
        sum(logpdf(Normal(0, 1), v) for v in tau) +
        sum(logpdf(Normal(0, 1), v) for v in z)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + logjac(layout, u)
    _check_parity_gradient(backend, u)
end

@stestset "rk parity gr correlated (stratified)" begin
    brmi = @brm _parity_cols_gr begin
        loc ~ 1 + (1 + x | gr(g, by = b))
        y ~ Normal(loc, sigma)
        effect(loc, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 16
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    beta, sigma = nt.loc_Intercept, nt.sigma
    draws = _parity_ranef(nt, backend)
    L1, L2 = draws.L[:, :, 1], draws.L[:, :, 2]
    tau1, tau2 = draws.sd[1, :], draws.sd[2, :]
    zm = permutedims(draws.z)
    sidx = [1, 1, 2, 2]
    gidx = [1, 1, 2, 3, 3, 4]
    Ls, taus = (L1, L2), (tau1, tau2)
    b = zeros(4, 2)
    for g in 1:4
        s = sidx[g]
        for k in 1:2
            b[g, k] =
                taus[s][k] * (Ls[s][k, 1] * zm[1, g] + Ls[s][k, 2] * zm[2, g])
        end
    end
    Z = hcat(ones(6), _parity_cols_gr.x)
    r = [Z[i, 1] * b[gidx[i], 1] + Z[i, 2] * b[gidx[i], 2] for i in 1:6]
    ll = sum(logpdf(Normal(beta + r[i], sigma), _parity_cols_gr.y[i])
        for i in 1:6)
    pr = logpdf(Normal(0, 5), beta) + logpdf(Exponential(1), sigma) +
        _ref_lkj_k2_eta1(L1) + _ref_lkj_k2_eta1(L2) +
        sum(logpdf(Normal(0, 1), v) for v in tau1) +
        sum(logpdf(Normal(0, 1), v) for v in tau2) +
        sum(logpdf(Normal(0, 1), v) for v in draws.z)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + logjac(layout, u)
    _check_parity_gradient(backend, u)
end

# Prior-vocab v1 corpus (P1-P5): per-addressee population families +
# widened sampled vocabulary through the full BRM→RK build path. SB
# literals pinned from brief 8yw5i7 (BRM ff5e589 / StanBlocks 24578c3 /
# BridgeStan 2.9.0, propto=false); the RK legs were compared there to
# ≤2e-15 (P3/P4 bit-exact).

@stestset "rk parity prior vocab P1 mixed StudentT+Laplace" begin
    p_cols = (;
        x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        y=[1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
    )
    brmi = @brm p_cols begin
        mu ~ 1 + x
        effect(mu, Intercept) ~ LocationScale(0, 2, TDist(4))
        effect(mu, x) ~ Laplace(0, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test coordinate_names(layout) ==
        [:mu_Intercept, :mu_x, :s]
    u = [0.5, -0.25, log(1.3)]
    nt = constrain(layout, u)
    a, b, s = _parity_population(nt, backend, :mu)[1], _parity_population(nt, backend, :mu)[2], nt.s
    lp = a .+ b .* p_cols.x
    ll = sum(logpdf.(Normal.(lp, s), p_cols.y))
    pr = logpdf(LocationScale(0, 2, TDist(4)), a) +
        logpdf(Laplace(0, 1), b) + logpdf(Exponential(1), s)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ log(s)
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + log(s)
    @test _rk_query(backend, :posterior, u) ≈ -15.676861749966013
    _check_parity_gradient(backend, u)
end

@stestset "rk parity prior vocab P2 Cauchy+Flat" begin
    p_cols = (;
        x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        y=[1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
    )
    brmi = @brm p_cols begin
        mu ~ 1 + x
        effect(mu, Intercept) ~ Cauchy(0, 1)
        effect(mu, x) ~ Flat()
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test coordinate_names(layout) ==
        [:mu_Intercept, :mu_x, :s]
    u = [0.5, -0.25, log(1.3)]
    nt = constrain(layout, u)
    a, b, s = _parity_population(nt, backend, :mu)[1], _parity_population(nt, backend, :mu)[2], nt.s
    lp = a .+ b .* p_cols.x
    ll = sum(logpdf.(Normal.(lp, s), p_cols.y))
    # Flat contributes exactly 0.0.
    pr = logpdf(Cauchy(0, 1), a) + 0.0 + logpdf(Exponential(1), s)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ log(s)
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + log(s)
    @test _rk_query(backend, :posterior, u) ≈ -14.388851106658095
    _check_parity_gradient(backend, u)
end

@stestset "rk parity prior vocab P3 factor StudentT" begin
    p_cols = (;
        x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        y=[1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
        g=[1, 2, 1, 3, 2, 3],
    )
    brmi = @brm p_cols begin
        mu ~ 0 + g + x
        effect(mu, g) ~ LocationScale(0, 2, TDist(3))
        effect(mu, x) ~ Cauchy(0, 1)
        y ~ Normal(mu, 1.5)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    point = Dict(Symbol("mu_g.1") => 0.3, Symbol("mu_g.2") => -0.4,
        Symbol("mu_g.3") => 0.1, :mu_x => 0.75)
    @test Set(coordinate_names(layout)) == Set(keys(point))
    u = [point[name] for name in coordinate_names(layout)]
    nt = constrain(layout, u)
    c, b = _parity_population(nt, backend, :mu)[1:3], _parity_population(nt, backend, :mu)[4]
    lp = [c[gi] + b * xi for (gi, xi) in zip(p_cols.g, p_cols.x)]
    ll = sum(logpdf.(Normal.(lp, 1.5), p_cols.y))
    t3 = LocationScale(0, 2, TDist(3))
    pr = sum(logpdf(t3, ci) for ci in c) + logpdf(Cauchy(0, 1), b)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    @test _rk_query(backend, :posterior, u) ≈ -21.633064049357102
    _check_parity_gradient(backend, u)
end

@stestset "rk parity prior vocab P4 Uniform scale" begin
    # The Uniform sampled scale rides the
    # affine-logit interval, same as SB's declared bounds.
    p_cols = (;
        x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        y=[1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
    )
    brmi = @brm p_cols begin
        mu ~ 1 + x
        s ~ Uniform(0.5, 1.5)
        y ~ Normal(mu, s)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test coordinate_names(layout) ==
        [:mu_Intercept, :mu_x, :s]
    lo, hi = 0.5, 1.5
    u = [0.5, -0.25, log((1.3 - lo) / (hi - 1.3))]
    nt = constrain(layout, u)
    b, s, a = nt.mu_x, nt.s, nt.mu_Intercept
    lp = a .+ b .* p_cols.x
    ll = sum(logpdf.(Normal.(lp, s), p_cols.y))
    pr = logpdf(Normal(0, 1), a) + logpdf(Normal(0, 1), b) +
        logpdf(Uniform(lo, hi), s)
    jac = log(s - lo) + log(hi - s)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ jac
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + jac
    @test _rk_query(backend, :posterior, u) ≈ -15.810050464119632
    _check_parity_gradient(backend, u)
end

@stestset "rk parity prior vocab P5 half-StudentT scale" begin
    # The half-StudentT sampled scale rides the
    # exact +log(2) `:positive` leg (SB `truncated(; lower)` matches).
    p_cols = (;
        x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        y=[1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
    )
    brmi = @brm p_cols begin
        mu ~ 1 + x
        s ~ truncated(LocationScale(0, 1, TDist(4)); lower=0.0)
        y ~ Normal(mu, s)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test coordinate_names(layout) ==
        [:mu_Intercept, :mu_x, :s]
    u = [0.5, -0.25, log(1.3)]
    nt = constrain(layout, u)
    b, s, a = nt.mu_x, nt.s, nt.mu_Intercept
    lp = a .+ b .* p_cols.x
    ll = sum(logpdf.(Normal.(lp, s), p_cols.y))
    pr = logpdf(Normal(0, 1), a) + logpdf(Normal(0, 1), b) +
        logpdf(LocationScale(0, 1, TDist(4)), s) + log(2)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ log(s)
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + log(s)
    @test _rk_query(backend, :posterior, u) ≈ -14.883826525901483
    _check_parity_gradient(backend, u)
end

@stestset "rk parity lognormal sampled sigma" begin
    ln_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        z=[1.2, 0.8, 1.1, 2.3, 0.7, 1.9],
    )
    brmi = @brm ln_cols begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        z ~ LogNormal(mu, sigma)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    u = [0.5, -0.25, 0.3]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
    mm = b[1] .+ b[2] .* ln_cols.x
    ll = sum(logpdf.(LogNormal.(mm, nt.sigma), ln_cols.z))
    pr = logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2]) +
        logpdf(Exponential(1), nt.sigma)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ u[3]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[3]
    _check_parity_gradient(backend, u)
end

@stestset "rk parity lognormal literal sigma" begin
    ln_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        z=[1.2, 0.8, 1.1, 2.3, 0.7, 1.9],
    )
    brmi = @brm ln_cols begin
        mu ~ 1 + x
        z ~ LogNormal(mu, 0.5)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 2
    u = [0.5, -0.25]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
    mm = b[1] .+ b[2] .* ln_cols.x
    ll = sum(logpdf.(LogNormal.(mm, 0.5), ln_cols.z))
    pr = logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end

# Weibull twins (pair fam-weibull): tail placement is permanent
# (the lognormal precedent) — a red mid-file testset would abort
# the tail, so family twins append here.
@stestset "rk parity weibull sampled shape" begin
    w_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        y=[0.5, 1.2, 0.8, 2.1, 1.7, 0.3],
    )
    brmi = @brm w_cols begin
        log(mu) ~ 1 + x
        k ~ LogNormal(0, 0.3)
        y ~ Weibull(k, mu)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 3
    u = [0.2, -0.3, 0.5]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
    th = exp.(b[1] .+ b[2] .* w_cols.x)
    ll = sum(logpdf.(Weibull.(nt.k, th), w_cols.y))
    pr = logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2]) +
        logpdf(LogNormal(0, 0.3), nt.k)
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    # Jacobian: k's exp (betas ride identity).
    @test logjac(layout, u) ≈ u[3]
    @test _rk_query(backend, :posterior, u) ≈ ll + pr + u[3]
    _check_parity_gradient(backend, u)
end

@stestset "rk parity weibull literal shape" begin
    w_cols = (;
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        y=[0.5, 1.2, 0.8, 2.1, 1.7, 0.3],
    )
    brmi = @brm w_cols begin
        log(mu) ~ 1 + x
        y ~ Weibull(2.0, mu)
    end
    backend = _parity_backend(brmi)
    layout = backend.model.layout
    @test layout.total == 2
    u = [0.2, -0.3]
    nt = constrain(layout, u)
    b = Vector(_parity_population(nt, backend, :mu))
    th = exp.(b[1] .+ b[2] .* w_cols.x)
    ll = sum(logpdf.(Weibull.(2.0, th), w_cols.y))
    pr = logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2])
    @test _rk_query(backend, :likelihood, u) ≈ ll
    @test _rk_query(backend, :prior, u) ≈ pr
    @test logjac(layout, u) ≈ 0.0
    @test _rk_query(backend, :posterior, u) ≈ ll + pr
    _check_parity_gradient(backend, u)
end
