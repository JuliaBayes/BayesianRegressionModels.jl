_brm_gp_matrix(axes::Tuple) = Matrix{Float64}(hcat(axes...))

function _brm_axis_option(label::Symbol, key::Symbol, value, n_axes::Int, pred, expectation::String)
    values = value isa Tuple || value isa AbstractVector ? Tuple(value) : ntuple(_ -> value, n_axes)
    length(values) == n_axes || error(
        "sbimpl: `$label(...; $key=...)` needs one value per axis ($n_axes), got $(length(values))")
    all(pred, values) || error(
        "sbimpl: `$label(...; $key=...)` expects $expectation, got $values")
    values
end

_brm_hsgp_options(kw, n_axes::Int) = begin
    K = _brm_axis_option(:hsgp, :k, get(kw, :k, 20), n_axes,
        x -> x isa Integer && !(x isa Bool) && x >= 1, "positive integers")
    c = _brm_axis_option(:hsgp, :c, get(kw, :c, 1.5), n_axes,
        x -> x isa Real && isfinite(x) && x > 1, "finite real values greater than 1")
    Tuple(Int(x) for x in K), Tuple(Float64(x) for x in c)
end

# Partial-centering coordinates are defined per tensor-product basis weight.
# A scalar is convenient for the two endpoints while a vector lets a pilot fit
# choose a different geometry for every spectral frequency.
function _brm_hsgp_centeredness(kw, n_basis::Int)
    raw = get(kw, :centeredness, 0.0)
    if raw isa NamedColumn
        backing = parent(raw)
        backing isa DataColumn || error(
            "hsgp: a named `centeredness` value must be backed by model data")
        raw = parent(backing)
    end
    values = if raw isa Real && !(raw isa Bool)
        fill(Float64(raw), n_basis)
    elseif raw isa Tuple || raw isa AbstractVector
        length(raw) == n_basis || error(
            "hsgp: `centeredness` needs one value per basis weight " *
            "($n_basis), got $(length(raw))")
        all(x -> x isa Real && !(x isa Bool), raw) || error(
            "hsgp: `centeredness` values must be real numbers")
        Float64[raw...]
    else
        error("hsgp: `centeredness` expects a real scalar or vector, got $(typeof(raw))")
    end
    all(x -> isfinite(x) && 0 <= x <= 1, values) || error(
        "hsgp: `centeredness` values must be finite and lie in [0, 1]")
    values
end

# `domain` is the actual compact interval used by the HSGP eigenfunctions,
# unlike `c`, which expands a domain inferred from raw training data. A latent
# axis has no Julia-time values from which to infer that interval, so it must
# supply one explicitly. Variadic raw HSGPs may supply one pair per axis.
function _brm_hsgp_domain_fits(kw, n_axes::Int; required::Bool=false)
    if !haskey(kw, :domain)
        required && error(
            "sbimpl: `hsgp(...)` over a model-derived axis requires an explicit " *
            "fixed `domain=(lower, upper)`; its sampled values do not exist " *
            "while the HSGP basis is being configured")
        return nothing
    end
    haskey(kw, :c) && error(
        "sbimpl: `hsgp(...; domain=...)` fixes the approximation boundary " *
        "directly and cannot also specify the data-derived expansion factor `c`")

    raw = kw[:domain]
    is_pair(x) = (x isa Tuple || x isa AbstractVector) && length(x) == 2 &&
                 all(v -> v isa Real && isfinite(v), x)
    pairs = if n_axes == 1 && is_pair(raw)
        (raw,)
    elseif (raw isa Tuple || raw isa AbstractVector) && length(raw) == n_axes &&
           all(is_pair, raw)
        Tuple(raw)
    else
        expectation = n_axes == 1 ? "`(lower, upper)`" :
            "one `(lower, upper)` pair per axis"
        error("sbimpl: `hsgp(...; domain=...)` expects $expectation, got $(repr(raw))")
    end

    ntuple(n_axes) do j
        lower, upper = Float64.(pairs[j])
        lower < upper || error(
            "sbimpl: `hsgp(...; domain=...)` needs lower < upper on axis $j, " *
            "got ($lower, $upper)")
        ((lower + upper) / 2, (upper - lower) / 2)
    end
end

function _brm_hsgp_orthogonal_to(kw, n_axes::Int)
    value = get(kw, :orthogonal_to, nothing)
    (isnothing(value) || value === :linear) || error(
        "sbimpl: `hsgp(...; orthogonal_to=...)` supports only `:linear`, got $(repr(value))")
    value === :linear && n_axes != 1 && error(
        "sbimpl: `hsgp(...; orthogonal_to=:linear)` currently supports exactly " *
        "one predictor axis, got $n_axes")
    value
end

_brm_gp_iso(kw, label::Symbol) = begin
    iso = get(kw, :iso, true)
    iso isa Bool || error("sbimpl: `$label(...; iso=...)` expects Bool, got $(typeof(iso))")
    iso
end

const _BRM_GP_COVARIANCES = (:exp_quad, :periodic)

_brm_gp_cov(kw, label::Symbol) = begin
    cov = get(kw, :cov, :exp_quad)
    cov in _BRM_GP_COVARIANCES || error(
        "sbimpl: `$label(...; cov=...)` supports " *
        join(("`$(repr(c))`" for c in _BRM_GP_COVARIANCES), " and ") *
        ", got $(repr(cov))")
    cov
end

# `period` is the periodic kernel's formula constant: required with
# `cov=:periodic`, meaningless (and refused) otherwise.
function _brm_gp_period(kw, label::Symbol, cov::Symbol)
    if cov !== :periodic
        haskey(kw, :period) && error(
            "sbimpl: `$label(...; period=...)` is meaningful only with " *
            "`cov=:periodic` (got `cov=$(repr(cov))`)")
        return nothing
    end
    haskey(kw, :period) || error(
        "sbimpl: `$label(...; cov=:periodic)` requires a numeric `period=` " *
        "formula constant (the kernel's period on the axis's own scale)")
    period = kw[:period]
    (period isa Real && !(period isa Bool) && isfinite(period) && period > 0) || error(
        "sbimpl: `$label(...; period=...)` expects a finite positive numeric " *
        "formula constant, got $(repr(period))")
    Float64(period)
end



# HSGP fit/apply split. The scalar methods reproduce the historical 1D basis;
# the tuple methods form its tensor product for variadic `hsgp(x...)`.
_brm_fit_hsgp(raw::AbstractVector{<:Real}, K::Integer, c::Real) = begin
    K >= 1 || error("hsgp: k must be >= 1 (got $K)")
    c > 1  || error("hsgp: c must be > 1 (got $c)")
    mu = sum(raw) / length(raw)
    L = c * maximum(abs, raw .- mu)
    L > 0 || error("hsgp: degenerate input (all x equal)")
    (mu, L)
end
_brm_apply_hsgp(c::Tuple, raw::AbstractVector{<:Real}, K::Integer) = begin
    mu, L = c
    x_c = raw .- mu
    lambda = [(k * pi / (2 * L))^2 for k in 1:K]
    PHI = zeros(length(raw), K)
    inv_sqrt_L = 1 / sqrt(L)
    for k in 1:K, i in eachindex(x_c)
        PHI[i, k] = inv_sqrt_L * sin(sqrt(lambda[k]) * (x_c[i] + L))
    end
    PHI, lambda
end

_brm_fit_hsgp(axes::Tuple, K::Tuple, c::Tuple) =
    ntuple(j -> _brm_fit_hsgp(axes[j], K[j], c[j]), length(axes))

# The weight threshold `w` in the validity bound below. 100 is the value the
# reference port uses; it is not reachable from the formula.
const _BRM_HSGP_WEIGHT_THRESHOLD = 100.0

# Riutort-Mayol et al. (2022) bound where the Hilbert-space approximation stops
# representing the kernel: with `k` basis functions on a domain of half-width
# `L`, a length scale below
#
#     (4L/pi) * sqrt(log(w) / (k^2 - 1))
#
# is not approximated, and the model silently becomes a GP nobody asked for --
# it still transpiles, still samples, still returns finite draws. Both inputs
# are known here, so `hsgp` declares `rho` with this as its lower bound by
# DEFAULT (decision 13keyez).
#
# It is passed as DATA rather than baked into the emitted Stan because `L`
# comes from the covariate: a `reprocess` with `freeze_constants=false`
# re-fits the basis on new data, and a literal would leave the bound describing
# the OLD basis while `PHI`/`omega2` describe the new one.
#
# `k == 1` has no usable floor (`k^2 - 1 == 0` puts the bound at infinity), so
# that degenerate basis stays unbounded rather than emitting an
# impossible-to-satisfy declaration.
_brm_hsgp_rho_lower(fit::Tuple, K::Integer) = begin
    K > 1 || return 0.0
    _, L = fit
    (4 * L / pi) * sqrt(log(_BRM_HSGP_WEIGHT_THRESHOLD) / (K^2 - 1))
end

# Per-axis bounds for the anisotropic spelling; the isotropic one shares a
# single `rho` across every axis, so it must satisfy the STRICTEST of them.
_brm_hsgp_rho_lowers(fits::Tuple, K::Tuple) =
    [_brm_hsgp_rho_lower(fits[j], K[j]) for j in eachindex(fits)]

_brm_hsgp_rho_lower_data(fits::Tuple, K::Tuple, iso::Bool) =
    iso ? maximum(_brm_hsgp_rho_lowers(fits, K)) : _brm_hsgp_rho_lowers(fits, K)

function _brm_apply_hsgp(fits::Tuple, axes::Tuple, K::Tuple)
    n_axes = length(axes)
    length(fits) == n_axes == length(K) || error("hsgp: internal axis-count mismatch")
    axis_basis = ntuple(j -> _brm_apply_hsgp(fits[j], axes[j], K[j]), n_axes)
    n_obs = length(first(axes))
    n_basis = prod(K)
    PHI = Matrix{Float64}(undef, n_obs, n_basis)
    omega2 = Matrix{Float64}(undef, n_basis, n_axes)
    for (b, I) in enumerate(CartesianIndices(K))
        for i in 1:n_obs
            value = 1.0
            for axis in 1:n_axes
                value *= axis_basis[axis][1][i, I[axis]]
            end
            PHI[i, b] = value
        end
        for axis in 1:n_axes
            omega2[b, axis] = axis_basis[axis][2][I[axis]]
        end
    end
    PHI, omega2
end

# Periodic Hilbert-space basis: `k` harmonics of the fundamental angular
# frequency `2pi / period`, `2k` columns -- cosines first, then sines -- with
# no centering, boundary factor, or domain. `harmonics` is the per-column
# harmonic index the Stan-side spectral weight reads.
function _brm_apply_hsgp_periodic(period::Real, raw::AbstractVector{<:Real},
                                 K::Integer)
    K >= 1 || error("hsgp: k must be >= 1 (got $K)")
    period > 0 || error("hsgp: period must be positive (got $period)")
    w0 = 2pi / period
    PHI = Matrix{Float64}(undef, length(raw), 2K)
    for j in 1:K, i in eachindex(raw)
        angle = w0 * j * raw[i]
        PHI[i, j] = cos(angle)
        PHI[i, K + j] = sin(angle)
    end
    PHI
end

_brm_hsgp_periodic_harmonics(K::Integer) =
    Float64[repeat(1:K, 2)...]

# The periodic analogue of `_brm_hsgp_rho_lower`, by the SAME amplitude-ratio
# rule: the exp-quad floor is the length scale at which the k-th basis
# function's spectral amplitude (sqrt of the spectral density) has fallen to
# 1/w of the first's -- `S(omega_k)/S(omega_1) = w^-2`, which is exactly what
# `(4L/pi) sqrt(log(w)/(k^2-1))` solves. The periodic basis has amplitude
# q_j = sigma sqrt(2 exp(-a) I_j(a)) with a = 1/rho^2, so the same rule reads
# `I_k(a)/I_1(a) = w^-2`. That ratio is monotone in `a` and has no closed
# form, so it is solved by bisection on the exponentially scaled Bessel
# functions (`besselix`, which never overflows). Unlike exp-quad the floor
# depends on `k` alone: there is no data-derived domain, so `reprocess` with
# either `freeze_constants` reproduces it exactly. `k == 1` stays unbounded
# for the same reason as the exp-quad case (no truncated harmonic to bound).
function _brm_hsgp_periodic_rho_lower(K::Integer)
    K > 1 || return 0.0
    target = _BRM_HSGP_WEIGHT_THRESHOLD^-2
    ratio(loga) = let a = exp(loga)
        SpecialFunctions.besselix(K, a) / SpecialFunctions.besselix(1, a) - target
    end
    # AMOS refuses |z| beyond ~1e8 (argument-reduction accuracy), and the
    # solution sits at a ~ (k^2 - 1) / (2 log(w^2)) — far inside this bracket
    # for any usable k.
    lo, hi = log(1e-12), log(1e7)
    ratio(lo) < 0 < ratio(hi) || error(
        "hsgp: internal periodic validity-floor bracket failed for k=$K")
    for _ in 1:200
        mid = (lo + hi) / 2
        ratio(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    1 / sqrt(exp((lo + hi) / 2))
end

function _brm_orthogonalize_hsgp_linear(PHI::AbstractMatrix,
                                       x::AbstractVector{<:Real})
    size(PHI, 1) == length(x) || error(
        "hsgp: internal orthogonalization row-count mismatch")
    xc = collect(Float64, x)
    xc .-= sum(xc) / length(xc)
    ss = sum(abs2, xc)
    out = Matrix{Float64}(undef, size(PHI))
    for b in axes(PHI, 2)
        phi = collect(Float64, @view PHI[:, b])
        phi .-= sum(phi) / length(phi)
        ss > 0 && (phi .-= xc .* (dot(xc, phi) / ss))
        out[:, b] = phi
    end
    out
end


# ==============================================================================
