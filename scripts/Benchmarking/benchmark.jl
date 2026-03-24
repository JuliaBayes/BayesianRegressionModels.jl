using DataFrames, Chairmarks
using LogDensityProblems: LogDensityProblems, logdensity, logdensity_and_gradient, dimension, capabilities
using PythonCall, RCall, BridgeStan

import ForwardDiff, Enzyme, Mooncake
using LinearAlgebra: dot, mul!
using Statistics: mean
import DifferentiationInterface as DI
using ADTypes: AutoForwardDiff, AutoEnzyme, AutoMooncake
using Turing
import DynamicPPL
using Distributions: TDist, Normal, logpdf

# ── Shared helpers ────────────────────────────────────────────────────────────

function _parse_formula(formula::String)
    lhs, rhs = strip.(split(formula, "~"))
    return Symbol(lhs), Symbol.(strip.(split(rhs, "+")))
end

function _center_predictors(data::DataFrame, predictors)
    raw = hcat([Vector{Float64}(data[!, p]) for p in predictors]...)
    col_means = vec(mean(raw, dims=1))
    return raw .- col_means', col_means
end

# ── PyMC (bambi) problem ───────────────────────────────────────────────────────
#
# bambi builds a PyMC model under the hood.  We compile a single pytensor
# function that returns (logp, flat_gradient) in one forward+backward pass,
# then wrap it in a Julia struct that implements the LogDensityProblems interface.
#
# Construction note: bambi's formula parser (formulae) captures the Python
# calling frame to resolve variable names.  When called directly from Julia
# there is no Python frame, so we define a thin Python wrapper function and
# call that instead.

# Cached numpy module — lazily initialized to avoid errors when Python is not available
const _NP = Ref{Py}()
function get_np()
    if !isassigned(_NP)
        _NP[] = pyimport("numpy")
    end
    return _NP[]
end

# Python helper — lazily initialized
const _PYMC_HELPERS = Ref{Py}()
function get_pymc_helpers()
    isassigned(_PYMC_HELPERS) && return _PYMC_HELPERS[]
    g = pydict()
    pyexec("""
import numpy as np
import pytensor
import pytensor.tensor as pt

def build_bambi(formula, data):
    \"\"\"Build a bambi model inside a Python frame so formulae can capture it.\"\"\"
    import bambi as bmb
    return bmb.Model(formula, data)

def make_logp_dlogp(pm_model, pytensor_mode=None):
    \"\"\"
    Compile two pytensor functions for a PyMC model that accept a single flat
    np.ndarray of unconstrained parameters directly (no dict conversion per call).

    Builds the symbolic graph by replacing each value_var with a slice of a single
    flat input vector, then differentiates w.r.t. that vector.  This eliminates
    the Python-level to_dict() call that would otherwise run on every evaluation.

    Returns (logp_fn, logp_dlogp_fn, q0_flat).
    Pass pytensor_mode=\"JAX\" to compile with the JAX backend.
    \"\"\"
    compile_kwargs = {"mode": pytensor_mode} if pytensor_mode is not None else {}

    q0_dict = pm_model.initial_point()
    keys    = list(q0_dict.keys())
    shapes  = {k: np.shape(q0_dict[k]) for k in keys}
    sizes   = {k: int(np.prod(shapes[k])) if shapes[k] else 1 for k in keys}
    q0_flat = np.concatenate(
        [np.atleast_1d(np.asarray(q0_dict[k])).ravel() for k in keys]
    )

    # Single symbolic flat-vector input
    q_sym = pt.vector("q", dtype=pytensor.config.floatX)

    # Map each value_var to its corresponding slice of q_sym
    replacements = {}
    i = 0
    for vv, k in zip(pm_model.value_vars, keys):
        s = sizes[k]
        replacements[vv] = q_sym[i:i+s].reshape(shapes[k]) if shapes[k] else q_sym[i]
        i += s

    logp_expr = pm_model.logp(jacobian=True)
    logp_sub  = pytensor.clone_replace(logp_expr, replacements)
    grad_sym  = pytensor.gradient.grad(logp_sub, q_sym)

    compiled_logp = pytensor.function([q_sym], logp_sub, **compile_kwargs)
    compiled_fn   = pytensor.function([q_sym], [logp_sub, grad_sym], **compile_kwargs)

    def call_logp(q_flat):
        return float(compiled_logp(q_flat))

    def call_logp_dlogp(q_flat):
        lp, grad = compiled_fn(q_flat)
        return float(lp), np.asarray(grad)

    return call_logp, call_logp_dlogp, q0_flat
""", g)
    _PYMC_HELPERS[] = g
    return g
end

struct PyMCProblem
    _logp_fn::Py         # (np.ndarray) -> float
    _fn::Py              # (np.ndarray) -> (float, np.ndarray)
    _q0::Vector{Float64} # initial unconstrained parameter vector
end

"""
    PyMCProblem(formula, data; compiler=:pytensor) -> PyMCProblem

Build a bambi/PyMC model from `formula` and `data`, compile a combined
logp+gradient pytensor function, and return a `PyMCProblem` that implements
the `LogDensityProblems` interface.

`compiler` controls the pytensor backend:
- `:pytensor` (default) — standard C-compiled pytensor
- `:jax`                — JAX backend (requires `jax` installed); faster after JIT warmup
"""
function PyMCProblem(formula::String, data::DataFrame; compiler::Symbol = :pytensor)
    helpers = get_pymc_helpers()
    build_bambi     = helpers["build_bambi"]
    make_logp_dlogp = helpers["make_logp_dlogp"]

    bm_model  = build_bambi(formula, pytable(data))
    bm_model.build()
    pm_model  = bm_model.backend.model

    pytensor_mode = compiler === :jax ? "JAX" : pybuiltins.None
    py_logp_fn, py_fn, py_q0 = make_logp_dlogp(pm_model, pytensor_mode)
    q0 = pyconvert(Vector{Float64}, py_q0)
    p  = PyMCProblem(py_logp_fn, py_fn, q0)

    # JAX functions are JIT-compiled on first call — run one warmup so benchmark
    # times reflect steady-state performance, not compilation.
    if compiler === :jax
        q0_np = get_np().asarray(q0)
        py_logp_fn(q0_np)
        py_fn(q0_np)
    end

    return p
end

LogDensityProblems.dimension(p::PyMCProblem) = length(p._q0)
LogDensityProblems.capabilities(::Type{PyMCProblem}) = LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity(p::PyMCProblem, q::AbstractVector{<:Real})
    np = pyimport("numpy")
    return pyconvert(Float64, p._logp_fn(np.asarray(q)))
end

function LogDensityProblems.logdensity_and_gradient(p::PyMCProblem, q::AbstractVector{<:Real})
    np   = pyimport("numpy")
    result = p._fn(np.asarray(q))
    lp   = pyconvert(Float64, result[0])
    grad = pyconvert(Vector{Float64}, result[1])
    return lp, grad
end

# ── Stan (brms) problem ───────────────────────────────────────────────────────
#
# brms generates Stan code and data; BridgeStan compiles it to a shared library
# that exposes log_density_gradient with a flat unconstrained parameter vector.
# BridgeStan.StanModel already supports log_density_gradient, so we wrap it in
# a thin struct to keep a symmetric interface with PyMCProblem.

# Sanitize a formula string for use as a directory name:
#   "drugs ~ o + c + e + a + n"  →  "drugs_o_c_e_a_n"
function _formula_to_path(formula::String)
    s = replace(formula, r"[^a-zA-Z0-9]+" => "_")
    return strip(s, '_')
end

struct StanProblem
    _model::BridgeStan.StanModel
    _q0::Vector{Float64}
end

"""
    StanProblem(formula, data; source, key, models_dir) -> StanProblem

Generate Stan code and data via `brms`, compile with BridgeStan, and return a
`StanProblem` that implements the `LogDensityProblems` interface.

Stan source and data files are written to a persistent directory:
  `{models_dir}/{source}/{key}/{sanitized_formula}/`
and reused on subsequent calls if they already exist.
"""
function StanProblem(
    formula::String,
    data::DataFrame;
    source::Union{Symbol,Nothing} = nothing,
    key::Union{Symbol,Nothing}    = nothing,
    models_dir::String            = joinpath(@__DIR__, "models"),
)
    formula_dir = _formula_to_path(formula)
    stan_dir = if source !== nothing && key !== nothing
        joinpath(models_dir, string(source), string(key), formula_dir)
    else
        joinpath(models_dir, formula_dir)
    end
    mkpath(stan_dir)

    stan_file = joinpath(stan_dir, "model.stan")
    data_file = joinpath(stan_dir, "data.json")

    if !isfile(stan_file) || !isfile(data_file)
        @rput data formula
        stan_code      = rcopy(String, R"brms::make_stancode(as.formula(formula), data=data)")
        stan_data_json = rcopy(String, R"""
            jsonlite::toJSON(
                brms::make_standata(as.formula(formula), data=data),
                auto_unbox=TRUE
            )
        """)
        write(stan_file, stan_code)
        write(data_file, stan_data_json)
    end

    sm = BridgeStan.StanModel(stan_file, data_file; warn=false)
    q0 = zeros(BridgeStan.param_unc_num(sm))
    return StanProblem(sm, q0)
end

"""
    StanProblem(stan_file, data_file) -> StanProblem

Compile a pre-written Stan model with BridgeStan.  Unlike the `(formula, data)`
constructor this skips brms code generation entirely.
"""
function StanProblem(stan_file::String, data_file::String)
    sm = BridgeStan.StanModel(stan_file, data_file; warn=false)
    q0 = zeros(BridgeStan.param_unc_num(sm))
    return StanProblem(sm, q0)
end

LogDensityProblems.dimension(p::StanProblem) = length(p._q0)
LogDensityProblems.capabilities(::Type{StanProblem}) = LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity(p::StanProblem, q::AbstractVector{<:Real})
    # propto=false avoids autodiff-type overhead that propto=true incurs even
    # when no gradient is being computed (see BridgeStan internals docs).
    return BridgeStan.log_density(p._model, q; propto=false, jacobian=true)
end

function LogDensityProblems.logdensity_and_gradient(p::StanProblem, q::AbstractVector{<:Real})
    return BridgeStan.log_density_gradient(p._model, q)
end

# ── Hand-written Julia problem ────────────────────────────────────────────────
#
# A pure Julia implementation of the same linear regression model that brms
# generates.  The log-density is hand-coded; gradients are computed via
# DifferentiationInterface with a user-selected AD backend.

struct JuliaProblem{F}
    _logdensity::F
    _q0::Vector{Float64}
end

"""
    JuliaProblem(formula, data; intercept_prior, sigma_prior) -> JuliaProblem

Build a hand-written Julia log-density function matching the brms-generated
Stan model (centered predictors, student-t priors).

`intercept_prior` and `sigma_prior` are `(df, loc, scale)` tuples matching the
`student_t_lpdf` calls in the generated Stan code.
"""
function JuliaProblem(
    formula::String, data::DataFrame;
    intercept_prior::NTuple{3,Float64} = (3.0, 2.1, 2.5),
    sigma_prior::NTuple{3,Float64}     = (3.0, 0.0, 2.5),
    kwargs...,
)
    response, predictors = _parse_formula(formula)
    Y  = Vector{Float64}(data[!, response])
    N  = length(Y)
    Kc = length(predictors)
    Xc, _ = _center_predictors(data, predictors)

    int_df, int_loc, int_scale = intercept_prior
    sig_df, sig_loc, sig_scale = sigma_prior

    q0 = zeros(Kc + 2)

    # Pre-compute constants
    half_N_log2π = N / 2 * log(2π)
    log_half      = log(0.5)
    int_dist      = TDist(int_df)
    sig_dist      = TDist(sig_df)
    inv_int_scale = inv(int_scale)
    log_int_scale = log(int_scale)
    inv_sig_scale = inv(sig_scale)
    log_sig_scale = log(sig_scale)

    function ℓ(q)
        b     = @view q[1:Kc]
        α     = q[Kc + 1]
        log_σ = q[Kc + 2]
        σ     = exp(log_σ)

        # Likelihood: Normal(Y | Xc*b + α, σ)
        r  = Y .- (Xc * b .+ α)
        lp = -half_N_log2π - N * log_σ - dot(r, r) / (2 * σ^2)

        # student_t prior on Intercept
        lp += logpdf(int_dist, (α - int_loc) * inv_int_scale) - log_int_scale

        # half-student_t prior on sigma
        lp += logpdf(sig_dist, (σ - sig_loc) * inv_sig_scale) - log_sig_scale - log_half

        # Jacobian for exp transform
        lp += log_σ

        return lp
    end

    return JuliaProblem(ℓ, q0)
end

struct JuliaProblem2{F}
    _logdensity::F
    _q0::Vector{Float64}
end

"""
    JuliaProblem2(formula, data; intercept_prior, sigma_prior) -> JuliaProblem2

Build a hand-written Julia log-density function matching the brms-generated
Stan model (centered predictors, student-t priors).

`intercept_prior` and `sigma_prior` are `(df, loc, scale)` tuples matching the
`student_t_lpdf` calls in the generated Stan code.
"""
function JuliaProblem2(
    formula::String, data::DataFrame;
    intercept_prior::NTuple{3,Float64} = (3.0, 2.1, 2.5),
    sigma_prior::NTuple{3,Float64}     = (3.0, 0.0, 2.5),
    kwargs...,
)
    response, predictors = _parse_formula(formula)
    Y  = Vector{Float64}(data[!, response])
    N  = length(Y)
    Kc = length(predictors)
    Xc, _ = _center_predictors(data, predictors)
    XcT = Matrix(Xc')'

    int_df, int_loc, int_scale = intercept_prior
    sig_df, sig_loc, sig_scale = sigma_prior

    q0 = zeros(Kc + 2)

    # Pre-compute constants
    half_N_log2π = N / 2 * log(2π)
    log_half      = log(0.5)
    int_dist      = TDist(int_df)
    sig_dist      = TDist(sig_df)
    inv_int_scale = inv(int_scale)
    log_int_scale = log(int_scale)
    inv_sig_scale = inv(sig_scale)
    log_sig_scale = log(sig_scale)

    function ℓ(q)
        b     = @view q[1:Kc]
        α     = q[Kc + 1]
        log_σ = q[Kc + 2]
        σ     = exp(log_σ)

        # Likelihood: Normal(Y | Xc*b + α, σ)
        r  = Base.broadcasted(-,Y, Base.broadcasted(+, α, Base.broadcasted(dot, eachrow(XcT), Ref(b))))
        #     Y .- (Xc * b .+ α)
        # )
        lp = -half_N_log2π - N * log_σ - mysum(abs2, r) / (2 * σ^2)

        # student_t prior on Intercept
        lp += logpdf(int_dist, (α - int_loc) * inv_int_scale) - log_int_scale

        # half-student_t prior on sigma
        lp += logpdf(sig_dist, (σ - sig_loc) * inv_sig_scale) - log_sig_scale - log_half

        # Jacobian for exp transform
        lp += log_σ

        return lp
    end

    return JuliaProblem2(ℓ, q0)
end

struct JuliaProblem3{F}
    _logdensity::F
    _q0::Vector{Float64}
end

"""
    JuliaProblem3(formula, data; intercept_prior, sigma_prior) -> JuliaProblem3

Like JuliaProblem but pre-allocates a buffer for `Xc * b` to avoid allocation
in the hot loop.
"""
function JuliaProblem3(
    formula::String, data::DataFrame;
    intercept_prior::NTuple{3,Float64} = (3.0, 2.1, 2.5),
    sigma_prior::NTuple{3,Float64}     = (3.0, 0.0, 2.5),
    kwargs...,
)
    response, predictors = _parse_formula(formula)
    Y  = Vector{Float64}(data[!, response])
    N  = length(Y)
    Kc = length(predictors)
    Xc, _ = _center_predictors(data, predictors)

    int_df, int_loc, int_scale = intercept_prior
    sig_df, sig_loc, sig_scale = sigma_prior

    q0 = zeros(Kc + 2)
    mu = Vector{Float64}(undef, N)  # pre-allocated buffer

    # Pre-compute constants
    half_N_log2π = N / 2 * log(2π)
    log_half      = log(0.5)
    int_dist      = TDist(int_df)
    sig_dist      = TDist(sig_df)
    inv_int_scale = inv(int_scale)
    log_int_scale = log(int_scale)
    inv_sig_scale = inv(sig_scale)
    log_sig_scale = log(sig_scale)

    function ℓ(q)
        b     = @view q[1:Kc]
        α     = q[Kc + 1]
        log_σ = q[Kc + 2]
        σ     = exp(log_σ)

        # Likelihood: Normal(Y | Xc*b + α, σ)
        mul!(mu, Xc, b)           # mu = Xc * b  (no allocation)
        ss = zero(eltype(q))
        @inbounds @simd for i in eachindex(Y)
            r = Y[i] - (mu[i] + α)
            ss += r * r
        end
        lp = -half_N_log2π - N * log_σ - ss / (2 * σ^2)

        # student_t prior on Intercept
        lp += logpdf(int_dist, (α - int_loc) * inv_int_scale) - log_int_scale

        # half-student_t prior on sigma
        lp += logpdf(sig_dist, (σ - sig_loc) * inv_sig_scale) - log_sig_scale - log_half

        # Jacobian for exp transform
        lp += log_σ

        return lp
    end

    return JuliaProblem3(ℓ, q0)
end

struct JuliaProblem5{F}
    _logdensity::F
    _q0::Vector{Float64}
end

"""
    JuliaProblem5(formula, data; ...) -> JuliaProblem5

Like JuliaProblem3 but uses 5-arg `mul!` to fuse `r = Y - Xc*b` into one
BLAS call, then handles the intercept shift in the sum-of-squares loop.
"""
function JuliaProblem5(
    formula::String, data::DataFrame;
    intercept_prior::NTuple{3,Float64} = (3.0, 2.1, 2.5),
    sigma_prior::NTuple{3,Float64}     = (3.0, 0.0, 2.5),
    kwargs...,
)
    response, predictors = _parse_formula(formula)
    Y  = Vector{Float64}(data[!, response])
    N  = length(Y)
    Kc = length(predictors)
    Xc, _ = _center_predictors(data, predictors)

    int_df, int_loc, int_scale = intercept_prior
    sig_df, sig_loc, sig_scale = sigma_prior

    q0 = zeros(Kc + 2)
    r_buf = Vector{Float64}(undef, N)  # pre-allocated residual buffer

    # Pre-compute constants
    half_N_log2π = N / 2 * log(2π)
    log_half      = log(0.5)
    int_dist      = TDist(int_df)
    sig_dist      = TDist(sig_df)
    inv_int_scale = inv(int_scale)
    log_int_scale = log(int_scale)
    inv_sig_scale = inv(sig_scale)
    log_sig_scale = log(sig_scale)

    function ℓ(q)
        b     = @view q[1:Kc]
        α     = q[Kc + 1]
        log_σ = q[Kc + 2]
        σ     = exp(log_σ)

        # r_buf = Y - Xc*b  (one BLAS call via 5-arg mul!)
        copyto!(r_buf, Y)
        mul!(r_buf, Xc, b, -1.0, 1.0)

        # Subtract intercept and accumulate sum of squares
        ss = zero(eltype(q))
        @inbounds @simd for i in eachindex(r_buf)
            ri = r_buf[i] - α
            ss += ri * ri
        end
        lp = -half_N_log2π - N * log_σ - ss / (2 * σ^2)

        # student_t prior on Intercept
        lp += logpdf(int_dist, (α - int_loc) * inv_int_scale) - log_int_scale

        # half-student_t prior on sigma
        lp += logpdf(sig_dist, (σ - sig_loc) * inv_sig_scale) - log_sig_scale - log_half

        # Jacobian for exp transform
        lp += log_σ

        return lp
    end

    return JuliaProblem5(ℓ, q0)
end

struct JuliaProblem6{F}
    _logdensity::F
    _q0::Vector{Float64}
end

"""
    JuliaProblem6(formula, data; ...) -> JuliaProblem6

Manual gemv: computes `mu = Xc * b` column-by-column with explicit loops
instead of calling BLAS, then fuses the residual + sum-of-squares in one pass.
"""
function JuliaProblem6(
    formula::String, data::DataFrame;
    intercept_prior::NTuple{3,Float64} = (3.0, 2.1, 2.5),
    sigma_prior::NTuple{3,Float64}     = (3.0, 0.0, 2.5),
    kwargs...,
)
    response, predictors = _parse_formula(formula)
    Y  = Vector{Float64}(data[!, response])
    N  = length(Y)
    Kc = length(predictors)
    Xc, _ = _center_predictors(data, predictors)

    int_df, int_loc, int_scale = intercept_prior
    sig_df, sig_loc, sig_scale = sigma_prior

    q0 = zeros(Kc + 2)
    r_buf = Vector{Float64}(undef, N)

    # Pre-compute constants
    half_N_log2π = N / 2 * log(2π)
    log_half      = log(0.5)
    int_dist      = TDist(int_df)
    sig_dist      = TDist(sig_df)
    inv_int_scale = inv(int_scale)
    log_int_scale = log(int_scale)
    inv_sig_scale = inv(sig_scale)
    log_sig_scale = log(sig_scale)

    function ℓ(q)
        b     = @view q[1:Kc]
        α     = q[Kc + 1]
        log_σ = q[Kc + 2]
        σ     = exp(log_σ)

        # Manual gemv: r_buf = Y - α - Xc * b  (column-by-column axpy)
        @inbounds @simd for i in 1:N
            r_buf[i] = Y[i] - α
        end
        @inbounds for j in 1:Kc
            bj = b[j]
            @simd for i in 1:N
                r_buf[i] -= Xc[i, j] * bj
            end
        end

        # Sum of squares
        ss = zero(eltype(q))
        @inbounds @simd for i in 1:N
            ss += r_buf[i] * r_buf[i]
        end
        lp = -half_N_log2π - N * log_σ - ss / (2 * σ^2)

        # student_t prior on Intercept
        lp += logpdf(int_dist, (α - int_loc) * inv_int_scale) - log_int_scale

        # half-student_t prior on sigma
        lp += logpdf(sig_dist, (σ - sig_loc) * inv_sig_scale) - log_sig_scale - log_half

        # Jacobian for exp transform
        lp += log_σ

        return lp
    end

    return JuliaProblem6(ℓ, q0)
end

struct JuliaProblem4{F}
    _logdensity::F
    _q0::Vector{Float64}
end

"""
    JuliaProblem4(formula, data; ...) -> JuliaProblem4

Like JuliaProblem2 (broadcasted row-dot) but materializes the broadcasted
residual into a pre-allocated vector before computing the sum of squares.
"""
function JuliaProblem4(
    formula::String, data::DataFrame;
    intercept_prior::NTuple{3,Float64} = (3.0, 2.1, 2.5),
    sigma_prior::NTuple{3,Float64}     = (3.0, 0.0, 2.5),
    kwargs...,
)
    response, predictors = _parse_formula(formula)
    Y  = Vector{Float64}(data[!, response])
    N  = length(Y)
    Kc = length(predictors)
    Xc, _ = _center_predictors(data, predictors)
    XcT = Matrix(Xc')'

    int_df, int_loc, int_scale = intercept_prior
    sig_df, sig_loc, sig_scale = sigma_prior

    q0 = zeros(Kc + 2)
    r_buf = Vector{Float64}(undef, N)  # pre-allocated residual buffer

    # Pre-compute constants
    half_N_log2π = N / 2 * log(2π)
    log_half      = log(0.5)
    int_dist      = TDist(int_df)
    sig_dist      = TDist(sig_df)
    inv_int_scale = inv(int_scale)
    log_int_scale = log(int_scale)
    inv_sig_scale = inv(sig_scale)
    log_sig_scale = log(sig_scale)

    function ℓ(q)
        b     = @view q[1:Kc]
        α     = q[Kc + 1]
        log_σ = q[Kc + 2]
        σ     = exp(log_σ)

        # Materialize broadcasted residual into pre-allocated buffer
        r_buf .= Base.broadcasted(-, Y, Base.broadcasted(+, α, Base.broadcasted(dot, eachrow(XcT), Ref(b))))
        lp = -half_N_log2π - N * log_σ - dot(r_buf, r_buf) / (2 * σ^2)

        # student_t prior on Intercept
        lp += logpdf(int_dist, (α - int_loc) * inv_int_scale) - log_int_scale

        # half-student_t prior on sigma
        lp += logpdf(sig_dist, (σ - sig_loc) * inv_sig_scale) - log_sig_scale - log_half

        # Jacobian for exp transform
        lp += log_σ

        return lp
    end

    return JuliaProblem4(ℓ, q0)
end

@inline function mysum(f, bc::Base.Broadcast.Broadcasted)
    bc′ = Base.Broadcast.preprocess(nothing, Base.Broadcast.instantiate(bc))
    rv = zero(Float64)
    @simd for I in eachindex(bc′)
        @inbounds rv += f(bc′[I])
    end
    rv
end
@inline function mysum(f, x)
    rv = zero(Float64)
    @simd for xi in x
        @inbounds rv += f(xi)
    end
    rv
end

# ── Split-argument version for fine-grained Enzyme annotations ───────────────
#
# The closure only captures constant data (Y, Xc, priors).  The mutable buffer
# is a separate argument so it can be annotated Duplicated while the closure
# stays Const.  This avoids shadow allocation for the large data arrays.

struct JuliaProblem7{F,G}
    _logdensity::F    # ℓ(q)  — for primal / non-Enzyme AD
    _logdensity_split::G  # ℓ_inner(mu, q) — for Enzyme with split annotations
    _q0::Vector{Float64}
    _mu::Vector{Float64}  # pre-allocated buffer (passed as arg, not captured)
end

function JuliaProblem7(
    formula::String, data::DataFrame;
    intercept_prior::NTuple{3,Float64} = (3.0, 2.1, 2.5),
    sigma_prior::NTuple{3,Float64}     = (3.0, 0.0, 2.5),
    kwargs...,
)
    response, predictors = _parse_formula(formula)
    Y  = Vector{Float64}(data[!, response])
    N  = length(Y)
    Kc = length(predictors)
    Xc, _ = _center_predictors(data, predictors)

    int_df, int_loc, int_scale = intercept_prior
    sig_df, sig_loc, sig_scale = sigma_prior

    q0 = zeros(Kc + 2)
    mu = Vector{Float64}(undef, N)

    half_N_log2π = N / 2 * log(2π)
    log_half      = log(0.5)
    int_dist      = TDist(int_df)
    sig_dist      = TDist(sig_df)
    inv_int_scale = inv(int_scale)
    log_int_scale = log(int_scale)
    inv_sig_scale = inv(sig_scale)
    log_sig_scale = log(sig_scale)

    # Split version: mu is an argument, closure captures only const data
    function ℓ_inner(mu, q)
        b     = @view q[1:Kc]
        α     = q[Kc + 1]
        log_σ = q[Kc + 2]
        σ     = exp(log_σ)

        mul!(mu, Xc, b)
        ss = zero(eltype(q))
        @inbounds @simd for i in eachindex(Y)
            r = Y[i] - (mu[i] + α)
            ss += r * r
        end
        lp = -half_N_log2π - N * log_σ - ss / (2 * σ^2)

        lp += logpdf(int_dist, (α - int_loc) * inv_int_scale) - log_int_scale
        lp += logpdf(sig_dist, (σ - sig_loc) * inv_sig_scale) - log_sig_scale - log_half
        lp += log_σ

        return lp
    end

    # Wrapper for primal / non-Enzyme use
    ℓ(q) = ℓ_inner(mu, q)

    return JuliaProblem7(ℓ, ℓ_inner, q0, mu)
end

function _benchmark_enzyme_split(p::JuliaProblem7, mode::Symbol; enzyme_mode=Enzyme.Reverse)
    q0     = p._q0
    mu     = p._mu
    dmu    = zeros(length(mu))
    ℓ_inner = p._logdensity_split
    if mode === :primal
        return @be $(p._logdensity)($q0)
    elseif mode === :gradient
        grad = zeros(length(q0))
        if enzyme_mode === Enzyme.Reverse || enzyme_mode isa Enzyme.ReverseMode
            return @be begin
                fill!($dmu, 0.0)
                fill!($grad, 0.0)
                Enzyme.autodiff(
                    $enzyme_mode,
                    Enzyme.Const($ℓ_inner),
                    Enzyme.Active,
                    Enzyme.Duplicated($mu, $dmu),
                    Enzyme.Duplicated($q0, $grad),
                )
            end
        else
            # Forward mode: use Duplicated return, batch over parameters
            Kc = length(q0)
            seeds = [zeros(Kc) for _ in 1:Kc]
            for i in 1:Kc; seeds[i][i] = 1.0; end
            dmus = [zeros(length(mu)) for _ in 1:Kc]
            batch_seeds = Enzyme.BatchDuplicated(q0, ntuple(i -> seeds[i], Kc))
            batch_dmus  = Enzyme.BatchDuplicated(mu, ntuple(i -> dmus[i], Kc))
            return @be begin
                for s in $seeds; fill!(s, 0.0); end
                for i in 1:$Kc; $(seeds)[i][i] = 1.0; end
                for d in $dmus; fill!(d, 0.0); end
                Enzyme.autodiff(
                    $enzyme_mode,
                    Enzyme.Const($ℓ_inner),
                    Enzyme.BatchDuplicated($mu, $(ntuple(i -> dmus[i], Kc))),
                    Enzyme.BatchDuplicated($q0, $(ntuple(i -> seeds[i], Kc))),
                )
            end
        end
    else
        throw(ArgumentError("Unknown mode :$mode."))
    end
end

const AnyJuliaProblem = Union{JuliaProblem, JuliaProblem2, JuliaProblem3, JuliaProblem4, JuliaProblem5, JuliaProblem6, JuliaProblem7}

LogDensityProblems.dimension(p::AnyJuliaProblem) = length(p._q0)
LogDensityProblems.capabilities(::Type{<:AnyJuliaProblem}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(p::AnyJuliaProblem, q::AbstractVector{<:Real}) = p._logdensity(q)

function LogDensityProblems.logdensity_and_gradient(p::AnyJuliaProblem, q::AbstractVector{<:Real})
    grad = similar(q, Float64)
    prep = DI.prepare_gradient(p._logdensity, AutoForwardDiff(), q)
    val, _ = DI.value_and_gradient!(p._logdensity, grad, prep, AutoForwardDiff(), q)
    return val, grad
end

function _benchmark(p::AnyJuliaProblem, mode::Symbol; ad=AutoForwardDiff())
    q0 = p._q0
    ℓ  = p._logdensity
    if mode === :primal
        return @be $ℓ($q0)
    elseif mode === :gradient
        prep = DI.prepare_gradient(ℓ, ad, q0)
        grad = similar(q0)
        return @be DI.value_and_gradient!($ℓ, $grad, $prep, $ad, $q0)
    else
        throw(ArgumentError("Unknown mode :$mode. Choose :primal or :gradient."))
    end
end

# ── Turing (DynamicPPL) problem ──────────────────────────────────────────────
#
# Uses DynamicPPL's @model macro to define the same regression model.
# Gradients are computed via DifferentiationInterface with a user-selected
# AD backend.

@model function _turing_brms_linear(Y, Xc, Kc, int_prior, sig_prior)
    # Flat prior on regression coefficients (matches brms default)
    b ~ filldist(Flat(), Kc)

    # student_t prior on Intercept
    Intercept ~ int_prior[2] + int_prior[3] * TDist(int_prior[1])

    # half-student_t prior on sigma
    sigma ~ truncated(sig_prior[2] + sig_prior[3] * TDist(sig_prior[1]), lower=0.0)

    # Likelihood
    mu = Xc * b .+ Intercept
    for i in eachindex(Y)
        Y[i] ~ Normal(mu[i], sigma)
    end
end

@model function _turing_brms_linear_addlogprob(Y, Xc, Kc, int_prior, sig_prior)
    N = length(Y)

    # Flat prior on regression coefficients (matches brms default)
    b ~ filldist(Flat(), Kc)

    # student_t prior on Intercept
    Intercept ~ int_prior[2] + int_prior[3] * TDist(int_prior[1])

    # half-student_t prior on sigma
    sigma ~ truncated(sig_prior[2] + sig_prior[3] * TDist(sig_prior[1]), lower=0.0)

    # Likelihood via @addlogprob! — bypasses per-observation tilde processing
    dmu = Y .- (Xc * b .+ Intercept)
    Turing.@addlogprob! -N/2 * log(2π) - N * log(sigma) - sum(abs2, dmu) / (2 * sigma^2)
end

struct TuringProblem{L}
    _logdensity::L      # DynamicPPL.LogDensityFunction
    _q0::Vector{Float64}
end

"""
    TuringProblem(formula, data; intercept_prior, sigma_prior) -> TuringProblem

Build a DynamicPPL model matching the brms-generated Stan model and wrap it
in a `LogDensityFunction` that implements the `LogDensityProblems` interface.
"""
function TuringProblem(
    formula::String, data::DataFrame;
    intercept_prior::NTuple{3,Float64} = (3.0, 2.1, 2.5),
    sigma_prior::NTuple{3,Float64}     = (3.0, 0.0, 2.5),
    kwargs...,
)
    response, predictors = _parse_formula(formula)
    Y  = Vector{Float64}(data[!, response])
    Kc = length(predictors)
    Xc, _ = _center_predictors(data, predictors)

    model = _turing_brms_linear(Y, Xc, Kc, intercept_prior, sigma_prior)
    ℓ = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, DynamicPPL.LinkAll())

    q0 = zeros(LogDensityProblems.dimension(ℓ))
    return TuringProblem(ℓ, q0)
end

"""
    TuringProblem with `@addlogprob!` — uses `:turing2` backend.

Same priors as the standard Turing model, but the likelihood is computed
manually and added via `Turing.@addlogprob!`, bypassing per-observation
tilde processing.
"""
function TuringProblem2(
    formula::String, data::DataFrame;
    intercept_prior::NTuple{3,Float64} = (3.0, 2.1, 2.5),
    sigma_prior::NTuple{3,Float64}     = (3.0, 0.0, 2.5),
    kwargs...,
)
    response, predictors = _parse_formula(formula)
    Y  = Vector{Float64}(data[!, response])
    Kc = length(predictors)
    Xc, _ = _center_predictors(data, predictors)

    model = _turing_brms_linear_addlogprob(Y, Xc, Kc, intercept_prior, sigma_prior)
    ℓ = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, DynamicPPL.LinkAll())

    q0 = zeros(LogDensityProblems.dimension(ℓ))
    return TuringProblem(ℓ, q0)
end

LogDensityProblems.dimension(p::TuringProblem) = length(p._q0)
LogDensityProblems.capabilities(::Type{<:TuringProblem}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(p::TuringProblem, q::AbstractVector{<:Real}) =
    LogDensityProblems.logdensity(p._logdensity, q)

function LogDensityProblems.logdensity_and_gradient(p::TuringProblem, q::AbstractVector{<:Real})
    f    = Base.Fix1(LogDensityProblems.logdensity, p._logdensity)
    grad = similar(q, Float64)
    prep = DI.prepare_gradient(f, AutoForwardDiff(), q)
    val, _ = DI.value_and_gradient!(f, grad, prep, AutoForwardDiff(), q)
    return val, grad
end

function _benchmark(p::TuringProblem, mode::Symbol; ad=AutoForwardDiff())
    q0     = p._q0
    ℓ_dppl = p._logdensity
    f      = Base.Fix1(LogDensityProblems.logdensity, ℓ_dppl)
    if mode === :primal
        return @be $f($q0)
    elseif mode === :gradient
        prep = DI.prepare_gradient(f, ad, q0)
        grad = similar(q0)
        return @be DI.value_and_gradient!($f, $grad, $prep, $ad, $q0)
    else
        throw(ArgumentError("Unknown mode :$mode. Choose :primal or :gradient."))
    end
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    make_problem(formula, data, backend) -> PyMCProblem | StanProblem | JuliaProblem | TuringProblem

Build a log-density problem for `backend` (`:bambi`, `:brms`, `:julia`, or `:turing`).
The returned object implements the `LogDensityProblems` interface:
`logdensity_and_gradient(problem, q)` takes a flat `Vector{Float64}` of
unconstrained parameters and returns `(logp::Float64, grad::Vector{Float64})`.
"""
function make_problem(
    formula::String,
    data::DataFrame,
    backend::Symbol;
    compiler::Symbol              = :pytensor,
    source::Union{Symbol,Nothing} = nothing,
    key::Union{Symbol,Nothing}    = nothing,
    models_dir::String            = joinpath(@__DIR__, "models"),
    intercept_prior::NTuple{3,Float64} = (3.0, 2.1, 2.5),
    sigma_prior::NTuple{3,Float64}     = (3.0, 0.0, 2.5),
)
    if backend === :bambi
        return PyMCProblem(formula, data; compiler)
    elseif backend === :brms
        return StanProblem(formula, data; source, key, models_dir)
    elseif backend === :julia
        return JuliaProblem(formula, data; intercept_prior, sigma_prior)
    elseif backend === :julia2
        return JuliaProblem2(formula, data; intercept_prior, sigma_prior)
    elseif backend === :julia3
        return JuliaProblem3(formula, data; intercept_prior, sigma_prior)
    elseif backend === :julia4
        return JuliaProblem4(formula, data; intercept_prior, sigma_prior)
    elseif backend === :julia5
        return JuliaProblem5(formula, data; intercept_prior, sigma_prior)
    elseif backend === :julia6
        return JuliaProblem6(formula, data; intercept_prior, sigma_prior)
    elseif backend === :julia7
        return JuliaProblem7(formula, data; intercept_prior, sigma_prior)
    elseif backend === :turing
        return TuringProblem(formula, data; intercept_prior, sigma_prior)
    elseif backend === :turing2
        return TuringProblem2(formula, data; intercept_prior, sigma_prior)
    else
        throw(ArgumentError("Unknown backend :$backend. Choose :bambi, :brms, :julia, or :turing."))
    end
end

# ── Low-overhead benchmark helpers ───────────────────────────────────────────
#
# These bypass the LogDensityProblems interface to eliminate interop overhead:
#
# PyMCProblem: pre-converts q0 to a numpy array with $ interpolation, so each
#   timed call is purely the pytensor computation (no Julia→Python conversion).
#
# StanProblem: pre-allocates the gradient buffer (log_density_gradient!) and
#   uses $ interpolation for the model and q0, so each timed call is the C
#   computation only (no alloc for the gradient vector).
#   For the primal, uses propto=false (avoids autodiff-type overhead when no
#   gradient is needed; see BridgeStan internals docs).

function _benchmark(p::PyMCProblem, mode::Symbol; kwargs...)
    q0_np = get_np().asarray(p._q0)
    if mode === :primal
        fn = p._logp_fn
        return @be $fn($q0_np)
    elseif mode === :gradient
        fn = p._fn
        return @be $fn($q0_np)
    else
        throw(ArgumentError("Unknown mode :$mode. Choose :primal or :gradient."))
    end
end

function _benchmark(p::StanProblem, mode::Symbol; kwargs...)
    q0   = p._q0
    if mode === :primal
        return @be BridgeStan.log_density($p._model, $q0; propto=false, jacobian=true)
    elseif mode === :gradient
        grad = zeros(length(q0))
        return @be BridgeStan.log_density_gradient!($p._model, $q0, $grad)
    else
        throw(ArgumentError("Unknown mode :$mode. Choose :primal or :gradient."))
    end
end

"""
    benchmark_model(formula, data, backend; mode, ad) -> Chairmarks.Sample

Build a log-density problem for the given backend, then use Chairmarks.@be to
benchmark a single evaluation at the initial unconstrained parameter vector.

`mode` controls what is benchmarked:
- `:gradient` (default) — logp + gradient
- `:primal`             — logp only, without gradient

`ad` selects the AD backend for gradient computation (`:julia` / `:turing` only):
- `AutoForwardDiff()` (default), `AutoEnzyme()`, or `AutoMooncake()`

Model compilation / pytensor tracing is excluded from the timing.
Interop overhead (Julia↔Python array conversion, gradient allocation) is
eliminated via \$ interpolation and pre-allocated buffers.
"""
function benchmark_model(
    formula::String,
    data::DataFrame,
    backend::Symbol;
    mode::Symbol                  = :gradient,
    ad                            = AutoForwardDiff(),
    compiler::Symbol              = :pytensor,
    source::Union{Symbol,Nothing} = nothing,
    key::Union{Symbol,Nothing}    = nothing,
    models_dir::String            = joinpath(@__DIR__, "models"),
    intercept_prior::NTuple{3,Float64} = (3.0, 2.1, 2.5),
    sigma_prior::NTuple{3,Float64}     = (3.0, 0.0, 2.5),
)
    problem = make_problem(formula, data, backend; compiler, source, key, models_dir, intercept_prior, sigma_prior)
    return _benchmark(problem, mode; ad)
end
