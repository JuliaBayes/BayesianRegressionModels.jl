# Adopted from RKPPL smooth_bases.jl at e005f231c1feb93e4eb1c636f238ccb02fc1cb9e.
# Input SHA256: 17be19680d104c30182cda3cd5cb4229a35d5e7cb6ace81af8fe76b4d033cd39.
# Model preparation belongs to BRM; shared fitted spline primitives below are
# the same ones used by BRM's Stan and Turing backends.
module StatisticalPreparation
using LinearAlgebra, Statistics, SpecialFunctions
const BRM = parentmodule(@__MODULE__)
const _HSGP_SQRT2PI = 2.5066282746310002
_rk_spline_fail(message) = error("BRM statistical preparation: " * message)
_rk_fit_spline(args...; kwargs...) = BRM._brm_fit_spline(args...; kwargs...)
function _rk_apply_spline(fit, x)
    X, Z = BRM._brm_apply_spline(fit, x)
    X[:, 2:2], Z
end
_rk_fit_cr_spline(args...; kwargs...) = BRM._brm_fit_cr_spline(args...; kwargs...)
_rk_apply_cr_spline(args...) = BRM._brm_apply_cr_spline(args...)
_rk_canonical_column_signs(M) = BRM._brm_canonical_column_signs(M)
_rk_row_tensor(A, B) = BRM._brm_row_tensor(A, B)
_hsgp_axis_fit(x, c, _name, _where) = BRM._brm_fit_hsgp(x, 1, c)
_hsgp_lambda(k, L) = (k * pi / (2L))^2
_hsgp_axis_floor(k, L) = BRM._brm_hsgp_rho_lower((0.0, L), k)
_hsgp_periodic_rho_lower(k) = BRM._brm_hsgp_periodic_rho_lower(k)
_grouping_levels(g) = BRM._rk_grouping_levels(g)
_declared_codes(g, pool) = Int[findfirst(isequal(x), pool) for x in g]
function _rk_tps_projection(E, T, k)
    q = size(T, 2)
    eig_E = eigen(Symmetric(E))
    keep = sortperm(abs.(eig_E.values); rev=true)[1:k]
    U = eig_E.vectors[:, keep]
    D = eig_E.values[keep]

    Z = nullspace(transpose(T) * U)
    size(Z, 2) == k - q || _rk_spline_fail(
        "could not isolate the $q-dimensional TPS null space")

    S = Symmetric(transpose(Z) * Diagonal(D) * Z)
    eig_S = eigen(S)
    penalty_scale = maximum(abs, eig_S.values)
    penalty_scale > 0 || _rk_spline_fail("`s(x)` produced a zero range-space penalty")
    tol = penalty_scale * eps(Float64) * 100
    minimum(eig_S.values) >= -tol || _rk_spline_fail(
        "`s(x)` produced a non-positive range-space penalty")
    penalty_values = max.(eig_S.values, tol)
    penalty_whitener = eig_S.vectors * Diagonal(inv.(sqrt.(penalty_values)))
    range_projection = _rk_canonical_column_signs(U * Z * penalty_whitener)

    return range_projection
end

function _rk_t2_raw_blocks(margins, axes_...)
    d = length(axes_)
    parts = map(_rk_apply_cr_spline, margins, axes_)
    NN = reduce(_rk_row_tensor, first.(parts))
    # Range before null, with the last margin varying fastest. At d=2
    # this is exactly the historical rr, rn, nr ordering.
    blocks = map((2^d - 1):-1:1) do mask
        selected = map(1:d) do j
            parts[j][iszero(mask & (1 << (d - j))) ? 1 : 2]
        end
        reduce(_rk_row_tensor, selected)
    end
    return (Matrix(NN[:, 2:end]), blocks...)
end

_rk_block_center(A::AbstractMatrix) = vec(sum(A; dims=1)) ./ size(A, 1)
_rk_center_block(A::AbstractMatrix, center) = A .- reshape(center, 1, :)

function _rk_fit_t2(axes_::AbstractVector{<:Real}...;
                    k::Tuple=ntuple(_ -> 5, length(axes_)))
    length(k) == length(axes_) || _rk_spline_fail("one k per tensor margin is required")
    all(x -> length(x) == length(first(axes_)), axes_) ||
        _rk_spline_fail("tensor margins must have equal lengths")
    margins = map((x, kk) -> _rk_fit_cr_spline(x; k=kk), axes_, k)
    raw = _rk_t2_raw_blocks(margins, axes_...)
    fixed_center = _rk_block_center(first(raw))
    (; margins, fixed_center, k)
end

function _rk_apply_t2(fit, axes_::AbstractVector{<:Real}...)
    raw = _rk_t2_raw_blocks(fit.margins, axes_...)
    (_rk_center_block(first(raw), fit.fixed_center), Base.tail(raw)...)
end

"""
    tps_basis(x; k = 10) -> (X, Z)

Thin-plate regression spline basis over the data vector `x` with basis
dimension `k` (`k > 2`, at least `k` distinct values of `x`).

- `X` (`n × 1`): the unpenalized null space, the centered linear column.
  The constant column is left out; the model's intercept owns it.
- `Z` (`n × (k - 2)`): the penalized range space, whitened so that an iid
  `Normal(0, sd)` coefficient prior is the smoothing penalty.

These are the columns the built-in `spline_basis(:id, x; k)` binds as
`<id>_Xnull_1` and `<id>_Zpen_j`. Use them with
[`penalized_smooth`](@ref).
"""
function tps_basis(x::AbstractVector{<:Real}; k::Integer = 10)
    X, Z = _rk_apply_spline(_rk_fit_spline(x; k = Int(k)), x)
    return (Matrix{Float64}(X), Matrix{Float64}(Z))
end

"""
    tps_basis(x, z; k = 10) -> (X, Z)

Isotropic two-dimensional thin-plate regression spline. The radial kernel
is `r^2 log(r) / (8pi)` (zero at `r = 0`), with null space `(1, x, z)`.
`X` contains the two centered linear columns; `Z` has `k - 3` whitened
penalized columns. Requires `k > 3`, at least `k` distinct locations and
a full-rank polynomial null space. Use with [`penalized_smooth`](@ref).
"""
function tps_basis(x::AbstractVector{<:Real}, z::AbstractVector{<:Real};
        k::Integer = 10)
    length(x) == length(z) || throw(DimensionMismatch("TPS axes must have equal lengths"))
    points = hcat(Float64.(x), Float64.(z))
    all(isfinite, points) || _rk_spline_fail("TPS locations must be finite")
    k > 3 || _rk_spline_fail("two-axis TPS needs k > 3")
    length(unique(eachrow(points))) >= k || _rk_spline_fail("TPS needs at least k distinct locations")
    centered = points .- mean(points; dims=1)
    T = hcat(ones(length(x)), centered)
    rank(T) == 3 || _rk_spline_fail("two-axis TPS needs non-collinear locations")
    E = Matrix{Float64}(undef, length(x), length(x))
    for j in axes(E, 2), i in axes(E, 1)
        r2 = sum(abs2, centered[i, :] .- centered[j, :])
        E[i, j] = iszero(r2) ? 0.0 : r2 * log(r2) / (16pi)
    end
    projection = _rk_tps_projection(E, T, Int(k))
    return (centered, E * projection)
end

"""
    cr_basis(x; k = 10) -> (X, Z)

Natural cubic regression spline, using quantile knots and the integrated
squared-second-derivative penalty. `X` is the centered linear null column
(the constant is omitted); `Z` has `k - 2` whitened range columns.
"""
function cr_basis(x::AbstractVector{<:Real}; k::Integer = 10)
    N, R = _rk_apply_cr_spline(_rk_fit_cr_spline(x; k=Int(k)), x)
    return (Matrix(N[:, 2:2]), Matrix(R))
end

"""
    t2_basis(axes...; k = 5) -> (X, penalized_blocks...)

Tensor-product (`t2`) cubic regression spline basis over one or more data
vectors. `k` is one integer for every margin, or one per margin, each `> 2`.
For `d` margins, `X` has `2^d - 1` centered null-space products (the
constant product is omitted). The `2^d - 1` penalized blocks follow
descending binary masks, with range = 1 and null = 0, the first margin
the most significant bit. Each block takes its own smoothing sd; within
a block the last margin varies fastest. One margin returns `(X, Z)`.

For two margins:

- `X` (`n × 3`): the unpenalized null-space products, column-centered (the
  constant product is left out; the model's intercept owns it).
- `Zrr`, `Zrn`, `Znr`: the three penalized blocks (range × range, range ×
  null, null × range), of widths `(k1-2)(k2-2)`, `2(k1-2)` and `2(k2-2)`,
  each penalized by its own smoothing sd.

For two margins these are the columns the built-in
`spline_basis(:id, x, z; k)` binds. Use them with [`t2_smooth`](@ref).
"""
function t2_basis(axes_::AbstractVector{<:Real}...; k = 5)
    isempty(axes_) && throw(ArgumentError("t2_basis takes at least one margin"))
    kk = Tuple(Int.(_per_axis(k, length(axes_), :k)))
    return _rk_apply_t2(_rk_fit_t2(axes_...; k=kk), axes_...)
end

# One value per axis: a scalar applies to every axis, a tuple or vector
# gives one entry per axis.
_per_axis(v::Number, d::Int, what) = fill(v, d)
function _per_axis(v::Union{Tuple,AbstractVector}, d::Int, what)
    length(v) == d || throw(ArgumentError(
        "`$what` takes one value or one per axis ($d), got " *
        repr(v)))
    return collect(v)
end

"""
    hsgp_basis(x...; k = 20, c = 1.5, domain = nothing, by = nothing) -> (PHI, lambda)

Hilbert-space approximate GP basis (exponentiated-quadratic kernel) over one
or more data vectors `x...` (the axes).

- `k`: basis functions per axis (an integer, or one per axis); the basis has
  `M = prod(k)` functions, the tensor products of the per-axis sines in
  `CartesianIndices(k)` order.
- `c`: the boundary factor (`> 1`, or one per axis). Each axis is centered at
  its mean and the domain half-width is `L = c * maximum(abs.(x .- mean(x)))`.
- `domain`: a fixed `(lo, hi)` (one axis) or one pair per axis instead of the
  data-derived domain (`c` is then unused); every value must lie inside.
- `by`: a grouping vector. The basis then repeats once per group level
  (`levels(by)` order), each copy zero outside its group: `PHI` has
  `G * M` columns, group fastest (column `(m - 1) * G + g` is basis
  function `m` on group `g`).

Returns `PHI` (`n × M`, or `n × G*M` with `by`) and `lambda` (`M × d`): the
per-axis eigenvalues `(k_j * pi / (2 L_j))^2` of each basis function.
[`hsgp_sqrt_spd`](@ref) turns `lambda` and the kernel's scale and length
scale into the coefficient weights, and [`hsgp_rho_floors`](@ref) gives the
smallest length scale the basis resolves. Use them with
[`hsgp_effect`](@ref) or [`hsgp_grouped_effect`](@ref). For one length scale
per axis, write the explicit statements documented under [`hsgp_effect`](@ref).
"""
function hsgp_basis(axes_::AbstractVector{<:Real}...; k = 20, c = 1.5,
        domain = nothing, by = nothing)
    d = length(axes_)
    d >= 1 || throw(ArgumentError("hsgp_basis takes at least one axis"))
    n = length(first(axes_))
    all(a -> length(a) == n, axes_) || throw(ArgumentError(
        "hsgp_basis: axes have different lengths $(map(length, axes_))"))
    K = Int.(_per_axis(k, d, :k))
    all(>(0), K) || throw(ArgumentError(
        "hsgp_basis: `k` must be positive, got $(repr(k))"))
    fits = Vector{Tuple{Float64,Float64}}(undef, d)
    if domain === nothing
        C = Float64.(_per_axis(c, d, :c))
        all(>(1), C) || throw(ArgumentError(
            "hsgp_basis: `c` must exceed 1, got $(repr(c))"))
        for j in 1:d
            fits[j] = _hsgp_axis_fit(axes_[j], C[j], :hsgp_basis,
                "hsgp_basis axis $j")
        end
    else
        pairs = d == 1 && domain isa Tuple{Real,Real} ? [domain] :
            _per_axis(domain, d, :domain)
        for j in 1:d
            lo, hi = Float64.(pairs[j])
            isfinite(lo) && isfinite(hi) && lo < hi || throw(ArgumentError(
                "hsgp_basis: domain $j must be finite with lo < hi, got " *
                repr(pairs[j])))
            all(v -> lo <= v <= hi, axes_[j]) || throw(ArgumentError(
                "hsgp_basis: axis $j has values outside its domain " *
                "($lo, $hi)"))
            fits[j] = ((lo + hi) / 2, (hi - lo) / 2)
        end
    end
    # Per-axis sines (SB `_brm_apply_hsgp`, the built-in's in-graph form):
    # `inv(sqrt(L)) * sin(sqrt(lambda_k) * (x - mu + L))`.
    cols = map(1:d) do j
        mu, L = fits[j]
        inv_sqrt_L = 1.0 / sqrt(L)
        xs = Float64.(axes_[j])
        [inv_sqrt_L .* sin.(sqrt(_hsgp_lambda(kk, L)) .* (xs .- mu .+ L))
            for kk in 1:K[j]]
    end
    idx = vec(collect(CartesianIndices(Tuple(K))))
    M = length(idx)
    PHI = Matrix{Float64}(undef, n, M)
    lambda = Matrix{Float64}(undef, M, d)
    for (b, I) in enumerate(idx)
        col = copy(cols[1][I[1]])
        for j in 2:d
            col .*= cols[j][I[j]]
        end
        PHI[:, b] = col
        for j in 1:d
            lambda[b, j] = _hsgp_lambda(I[j], fits[j][2])
        end
    end
    by === nothing && return (PHI, lambda)
    return (_grouped_basis(PHI, by), lambda)
end

# The basis repeated per group level, zero outside the group, group
# fastest: column `(m - 1) * G + g` is `PHI[:, m]` on the rows of level `g`.
function _grouped_basis(PHI::AbstractMatrix, by::AbstractVector)
    size(PHI, 1) == length(by) || throw(ArgumentError(
        "hsgp_basis: `by` has $(length(by)) values for $(size(PHI, 1)) rows"))
    levels = _grouping_levels(by)
    codes = _declared_codes(by, levels)
    G, M = length(levels), size(PHI, 2)
    out = zeros(Float64, size(PHI, 1), G * M)
    for m in 1:M, i in axes(PHI, 1)
        out[i, (m - 1) * G + codes[i]] = PHI[i, m]
    end
    return out
end

"""
    hsgp_periodic_basis(x; k = 20, period) -> (PHI, harmonics)

Hilbert-space approximate GP basis for the periodic kernel with the given
`period` (on the scale of `x`): `k` harmonics of the angular frequency
`2pi / period` as `2k` columns, the cosines then the sines. `harmonics`
(length `2k`) is the harmonic index of each column (`1..k, 1..k`).
[`hsgp_periodic_sqrt_spd`](@ref) turns it into the coefficient weights. Use
them with [`hsgp_periodic_effect`](@ref).
"""
function hsgp_periodic_basis(x::AbstractVector{<:Real}; k::Integer = 20,
        period::Real, by = nothing)
    k > 0 || throw(ArgumentError("hsgp_periodic_basis: `k` must be positive"))
    isfinite(period) && period > 0 || throw(ArgumentError(
        "hsgp_periodic_basis: `period` must be finite and positive"))
    all(isfinite, x) || throw(ArgumentError(
        "hsgp_periodic_basis: x must be finite"))
    xs = Float64.(x)
    w0 = 2.0 * pi / Float64(period)
    PHI = hcat((cos.((w0 * j) .* xs) for j in 1:k)...,
        (sin.((w0 * j) .* xs) for j in 1:k)...)
    basis = by === nothing ? Matrix{Float64}(PHI) : _grouped_basis(PHI, by)
    return (basis, vcat(collect(1:Int(k)), collect(1:Int(k))))
end

"""
    hsgp_periodic_basis(x, z, axes...; k = 20, period, by = nothing)

Separable periodic Fourier basis over multiple axes. `k` and `period` are
scalar or per-axis. Includes constant, cosine and sine factors on each
axis and omits only the all-constant product, leaving `prod(2k .+ 1)-1`
columns. The returned harmonic matrix has one row per column and one
column per axis, including zero for a constant factor. `by` uses the
same group-fastest ordering as [`hsgp_basis`](@ref).
"""
function hsgp_periodic_basis(x::AbstractVector{<:Real},
        z::AbstractVector{<:Real}, rest::AbstractVector{<:Real}...;
        k = 20, period, by = nothing)
    axes_ = (x, z, rest...)
    d, n = length(axes_), length(x)
    all(a -> length(a) == n, axes_) || throw(DimensionMismatch("periodic axes must have equal lengths"))
    K = Int.(_per_axis(k, d, :k))
    periods = _per_axis(period, d, :period)
    margins = map(1:d) do j
        P, h = hsgp_periodic_basis(axes_[j]; k=K[j], period=periods[j])
        (hcat(ones(n), P), vcat(0, h))
    end
    indices = vec(collect(CartesianIndices(Tuple(2 .* K .+ 1))))[2:end]
    PHI = Matrix{Float64}(undef, n, length(indices))
    harmonics = Matrix{Int}(undef, length(indices), d)
    for (m, I) in enumerate(indices)
        col = copy(margins[1][1][:, I[1]])
        for j in 2:d
            col .*= margins[j][1][:, I[j]]
        end
        PHI[:, m] = col
        harmonics[m, :] = [margins[j][2][I[j]] for j in 1:d]
    end
    return (by === nothing ? PHI : _grouped_basis(PHI, by), harmonics)
end

"""
    hsgp_matern_sqrt_spd(lambda, sigma, rho, nu)

Square root of the d-dimensional Matérn spectral density, evaluated at
the Laplacian eigenvalues from [`hsgp_basis`](@ref). `nu` is explicitly
positive (`1.5` for Matérn 3/2, `2.5` for Matérn 5/2); `rho` is a scalar
or one length scale per axis. Angular-frequency convention of Solin &
Särkkä (2020), https://doi.org/10.1007/s11222-019-09886-w.
"""
function hsgp_matern_sqrt_spd(lambda::AbstractMatrix, sigma::Number, rho, nu::Real)
    nu > 0 && isfinite(nu) || throw(ArgumentError("Matérn nu must be finite and positive"))
    d = size(lambda, 2)
    r = _matern_rho(rho, d)
    c = d * log(2.0) + (d/2) * log(pi) +
        SpecialFunctions.loggamma(nu + d/2) - SpecialFunctions.loggamma(nu) + nu * log(2nu)
    return sigma .* exp.(0.5 .* (c + sum(log.(r)) .-
        (nu + d/2) .* log.(2nu .+ lambda * (r .* r))))
end
_matern_rho(rho::Number, d) = fill(rho, d)
function _matern_rho(rho::AbstractVector, d)
    length(rho) == d || throw(DimensionMismatch("one Matérn length scale per axis is required"))
    return rho
end

"""
    hsgp_sqrt_spd(lambda, sigma, rho) -> Vector

Coefficient weights of an [`hsgp_basis`](@ref): the square root of the
exponentiated-quadratic kernel's spectral density at each basis function's
eigenvalues `lambda` (`M × d`), for marginal scale `sigma` and length scale
`rho` (a scalar, isotropic, or one per axis):
`sigma * prod_j sqrt(rho_j * sqrt(2pi)) * exp(-rho_j^2 * lambda_j / 4)`.
"""
function hsgp_sqrt_spd(lambda::AbstractMatrix, sigma::Number, rho::Number)
    scale = sigma * sqrt(rho * _HSGP_SQRT2PI)^size(lambda, 2)
    return scale .* exp.(-0.25 .* (rho * rho) .* vec(sum(lambda; dims = 2)))
end

function hsgp_sqrt_spd(lambda::AbstractMatrix, sigma::Number,
        rho::AbstractVector)
    length(rho) == size(lambda, 2) || throw(ArgumentError(
        "hsgp_sqrt_spd: one length scale per axis ($(size(lambda, 2))), " *
        "got $(length(rho))"))
    scale = sigma * prod(sqrt.(rho .* _HSGP_SQRT2PI))
    return scale .* exp.(-0.25 .* (lambda * (rho .* rho)))
end

"""
    hsgp_grouped_sqrt_spd(lambda, sigma, rho) -> Vector

Coefficient weights of a grouped [`hsgp_basis`](@ref) (`by = g`): one
isotropic exponentiated-quadratic spectral weight per basis function and
group, group fastest (the grouped basis's column order). `sigma` and `rho`
are each one value per group (`length(levels(g))`) or one shared scalar.
"""
function hsgp_grouped_sqrt_spd(lambda::AbstractMatrix, sigma, rho)
    d = size(lambda, 2)
    scale = sigma .* sqrt.(rho .* _HSGP_SQRT2PI) .^ d
    E = exp.(-0.25 .* (rho .* rho) .* permutedims(vec(sum(lambda; dims = 2))))
    return vec(scale .* E)
end

"""
    hsgp_periodic_sqrt_spd(harmonics, sigma, rho) -> Vector

Coefficient weights of an [`hsgp_periodic_basis`](@ref): for harmonic `j`,
`sigma * sqrt(2 * exp(-a) * besseli(j, a))` with `a = 1 / rho^2`, evaluated
through the exponentially scaled `besselix` so it never overflows.
"""
function hsgp_periodic_sqrt_spd(harmonics::AbstractVector, sigma::Number,
        rho::Number)
    a = 1.0 / (rho * rho)
    return sigma .* exp.(0.5 .* (0.6931471805599453 .+
        log.(SpecialFunctions.besselix.(harmonics, a))))
end

function hsgp_periodic_sqrt_spd(harmonics::AbstractMatrix, sigma::Number, rho)
    r = _matern_rho(rho, size(harmonics, 2))
    a = permutedims(1.0 ./ (r .* r))
    logweights = log.(SpecialFunctions.besselix.(harmonics, a)) .+
        log(2.0) .* (harmonics .!= 0)
    return sigma .* exp.(0.5 .* vec(sum(logweights; dims=2)))
end

"""
    hsgp_periodic_grouped_sqrt_spd(harmonics, sigma, rho)

Periodic spectral weights with one scale and isotropic length scale per
group (or shared scalars), flattened in group-fastest basis order.
"""
function hsgp_periodic_grouped_sqrt_spd(harmonics::AbstractVector, sigma, rho)
    a = 1.0 ./ (rho .* rho)
    return vec(sigma .* sqrt.(2.0 .* SpecialFunctions.besselix.(permutedims(harmonics), a)))
end
function hsgp_periodic_grouped_sqrt_spd(harmonics::AbstractMatrix, sigma, rho)
    a = reshape(1.0 ./ (_group_vector(rho) .* _group_vector(rho)), :, 1, 1)
    h = reshape(harmonics, 1, size(harmonics)...)
    logweights = log.(SpecialFunctions.besselix.(h, a)) .+ log(2.0) .* (h .!= 0)
    return vec(_group_vector(sigma) .* exp.(0.5 .* dropdims(sum(logweights; dims=3); dims=3)))
end
_group_vector(x::Number) = [x]
_group_vector(x::AbstractVector) = x

"""
    hsgp_rho_floors(lambda) -> Vector{Float64}

Per-axis validity floors of an [`hsgp_basis`](@ref) from its eigenvalues
`lambda` (`M × d`): the length scale below which axis `j`'s `K_j`-term
approximation degrades, `(4 L_j / pi) * sqrt(log(100) / (K_j^2 - 1))`
(`0.0` when `K_j == 1`). The isotropic floor is their `maximum`.
"""
function hsgp_rho_floors(lambda::AbstractMatrix)
    return map(axes(lambda, 2)) do j
        lo, hi = extrema(view(lambda, :, j))
        lo > 0 || throw(ArgumentError(
            "hsgp_rho_floors: eigenvalues must be positive"))
        _hsgp_axis_floor(round(Int, sqrt(hi / lo)), pi / (2 * sqrt(lo)))
    end
end

"""
    hsgp_periodic_rho_floor(harmonics) -> Float64

Validity floor of an [`hsgp_periodic_basis`](@ref): the length scale at
which the highest harmonic's spectral weight falls to 1/100 of the first's
(`0.0` for a single harmonic).
"""
hsgp_periodic_rho_floor(harmonics::AbstractVector) =
    _hsgp_periodic_rho_lower(maximum(harmonics))
hsgp_periodic_rho_floor(harmonics::AbstractMatrix) =
    maximum(_hsgp_periodic_rho_lower(maximum(col)) for col in eachcol(harmonics))

end # StatisticalPreparation
