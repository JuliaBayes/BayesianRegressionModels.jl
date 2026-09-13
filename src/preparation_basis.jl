# Fit/apply split for Wood's rank-k thin-plate regression spline (d=1, m=2).
# The radial kernel is eta(r)=r^3/12 (Wood 2003, eq. 7). We keep the k largest-
# magnitude eigenvectors of the full kernel matrix, impose the T' * delta = 0
# side constraint, and diagonalize the resulting range-space penalty. Applying
# the fitted object to new x values needs only the frozen training centers,
# shift, and penalty-whitened range projection.
function _brm_tps_kernel(x::AbstractVector{<:Real}, centers::AbstractVector{<:Real})
    E = Matrix{Float64}(undef, length(x), length(centers))
    for j in eachindex(centers), i in eachindex(x)
        E[i, j] = abs(Float64(x[i]) - Float64(centers[j]))^3 / 12
    end
    E
end

function _brm_fit_spline(x::AbstractVector{<:Real}; k::Int=10)
    k > 2 || error("sbimpl: `s(x)` needs basis dimension k > 2 (got $k)")
    xs = collect(Float64, x)
    all(isfinite, xs) || error("sbimpl: `s(x)` requires finite numeric data")
    length(unique(xs)) >= k || error(
        "sbimpl: `s(x)` needs at least $k unique x values for the default ",
        "thin-plate basis (got $(length(unique(xs))))")

    shift = sum(xs) / length(xs)
    centers = xs .- shift
    E = _brm_tps_kernel(centers, centers)
    eig_E = eigen(Symmetric(E))
    keep = sortperm(abs.(eig_E.values); rev=true)[1:k]
    U = eig_E.vectors[:, keep]
    D = eig_E.values[keep]

    T = hcat(ones(Float64, length(xs)), centers)
    Z = nullspace(transpose(T) * U)
    size(Z, 2) == k - 2 || error(
        "sbimpl: `s(x)` could not isolate the two-dimensional TPS null space")

    S = Symmetric(transpose(Z) * Diagonal(D) * Z)
    eig_S = eigen(S)
    penalty_scale = maximum(abs, eig_S.values)
    penalty_scale > 0 || error("sbimpl: `s(x)` produced a zero range-space penalty")
    tol = penalty_scale * eps(Float64) * 100
    minimum(eig_S.values) >= -tol || error(
        "sbimpl: `s(x)` produced a non-positive range-space penalty")
    penalty_values = max.(eig_S.values, tol)
    penalty_whitener = eig_S.vectors * Diagonal(inv.(sqrt.(penalty_values)))
    range_projection = U * Z * penalty_whitener

    (; shift, centers, range_projection, k)
end

function _brm_apply_spline(fit, x::AbstractVector{<:Real})
    xs = collect(Float64, x)
    all(isfinite, xs) || error("sbimpl: `s(x)` requires finite numeric data")
    centered = xs .- fit.shift
    Xnull = hcat(ones(Float64, length(xs)), centered)
    Zpen = _brm_tps_kernel(centered, fit.centers) * fit.range_projection
    Xnull, Zpen
end

_brm_spline_basis_tps(x::AbstractVector{<:Real}; k::Int=10) =
    _brm_apply_spline(_brm_fit_spline(x; k), x)

# Cubic regression spline margin used by `t2`. Knots follow R's default
# quantile algorithm (type 7) over the sorted unique training values. `F` maps
# knot values to the natural cubic spline's second derivatives, while `S` is
# the integrated-squared-second-derivative penalty. The positive eigenspace of
# `S` is penalty-whitened without mixing in the null space, so tensoring the
# marginal range/null pieces preserves the three `t2(full=false)` penalties.
function _brm_type7_knots(x::AbstractVector{<:Real}, k::Int)
    values = sort!(unique(collect(Float64, x)))
    length(values) >= k || error(
        "sbimpl: `t2` margin needs at least $k unique values (got $(length(values)))")
    n = length(values)
    knots = Vector{Float64}(undef, k)
    for i in 1:k
        pos = 1 + (n - 1) * (i - 1) / (k - 1)
        lo = clamp(floor(Int, pos), 1, n)
        hi = clamp(ceil(Int, pos), 1, n)
        weight = pos - lo
        knots[i] = (1 - weight) * values[lo] + weight * values[hi]
    end
    all(diff(knots) .> 0) || error(
        "sbimpl: `t2` margin produced non-distinct cubic-regression-spline knots")
    knots
end

function _brm_cr_second_derivative_map(knots::AbstractVector{<:Real})
    k = length(knots)
    h = diff(knots)
    all(h .> 0) || error("sbimpl: `t2` cubic-regression-spline knots must increase")
    D = zeros(Float64, k - 2, k)
    B = zeros(Float64, k - 2, k - 2)
    for i in 1:(k - 2)
        D[i, i] = inv(h[i])
        D[i, i + 1] = -inv(h[i]) - inv(h[i + 1])
        D[i, i + 2] = inv(h[i + 1])
        B[i, i] = (h[i] + h[i + 1]) / 3
        if i < k - 2
            B[i, i + 1] = h[i + 1] / 6
            B[i + 1, i] = B[i, i + 1]
        end
    end
    interior = B \ D
    F = zeros(Float64, k, k)
    F[2:(k - 1), :] .= interior
    F, transpose(D) * interior
end

function _brm_cr_basis(knots, F, x::AbstractVector{<:Real})
    k = length(knots)
    X = zeros(Float64, length(x), k)
    for (i, raw_x) in enumerate(x)
        xi = Float64(raw_x)
        if xi < knots[1]
            h = knots[2] - knots[1]
            xik = xi - knots[1]
            cjm = -xik * h / 3
            cjp = -xik * h / 6
            for q in 1:k
                X[i, q] = cjm * F[1, q] + cjp * F[2, q]
            end
            X[i, 1] += 1 - xik / h
            X[i, 2] += xik / h
        elseif xi > knots[k]
            h = knots[k] - knots[k - 1]
            xik = xi - knots[k]
            cjm = xik * h / 6
            cjp = xik * h / 3
            for q in 1:k
                X[i, q] = cjm * F[k - 1, q] + cjp * F[k, q]
            end
            X[i, k - 1] -= xik / h
            X[i, k] += 1 + xik / h
        else
            j = clamp(searchsortedlast(knots, xi), 1, k - 1)
            h = knots[j + 1] - knots[j]
            ajm = knots[j + 1] - xi
            ajp = xi - knots[j]
            cjm = ajm * (ajm * ajm / h - h) / 6
            cjp = ajp * (ajp * ajp / h - h) / 6
            for q in 1:k
                X[i, q] = cjm * F[j, q] + cjp * F[j + 1, q]
            end
            X[i, j] += ajm / h
            X[i, j + 1] += ajp / h
        end
    end
    X
end

function _brm_fit_cr_spline(x::AbstractVector{<:Real}; k::Int=5)
    k > 2 || error("sbimpl: `t2` basis dimensions must be integers greater than 2 (got $k)")
    xs = collect(Float64, x)
    isempty(xs) && error("sbimpl: `t2` cannot use an empty margin")
    all(isfinite, xs) || error("sbimpl: `t2` margins require finite numeric data")
    shift = sum(xs) / length(xs)
    scale = maximum(xs) - minimum(xs)
    scale > 0 || error("sbimpl: `t2` margin is degenerate (all values equal)")
    normalized = (xs .- shift) ./ scale
    knots = _brm_type7_knots(normalized, k)
    F, penalty = _brm_cr_second_derivative_map(knots)

    eig_penalty = eigen(Symmetric(penalty))
    order = sortperm(eig_penalty.values; rev=true)
    keep = order[1:(k - 2)]
    penalty_scale = maximum(abs, eig_penalty.values)
    tol = penalty_scale * eps(Float64) * 100
    minimum(eig_penalty.values) >= -tol || error(
        "sbimpl: `t2` cubic-regression-spline penalty is not positive semidefinite")
    minimum(eig_penalty.values[keep]) > tol || error(
        "sbimpl: `t2` could not isolate the two-dimensional marginal null space")
    range_projection = eig_penalty.vectors[:, keep] *
                       Diagonal(inv.(sqrt.(eig_penalty.values[keep])))

    null_const_scale = inv(sqrt(length(xs)))
    slope_norm = norm(normalized)
    slope_norm > 0 || error("sbimpl: `t2` margin has a zero linear null-space norm")

    (; shift, scale, knots, F, range_projection, null_const_scale, slope_norm, k)
end

function _brm_apply_cr_spline(fit, x::AbstractVector{<:Real})
    xs = collect(Float64, x)
    all(isfinite, xs) || error("sbimpl: `t2` margins require finite numeric data")
    normalized = (xs .- fit.shift) ./ fit.scale
    Xnull = hcat(fill(fit.null_const_scale, length(xs)),
                 normalized ./ fit.slope_norm)
    range = _brm_cr_basis(fit.knots, fit.F, normalized) * fit.range_projection
    Xnull, range
end

function _brm_row_tensor(A::AbstractMatrix, B::AbstractMatrix)
    size(A, 1) == size(B, 1) || error(
        "sbimpl: `t2` marginal basis row counts differ ($(size(A, 1)) vs $(size(B, 1)))")
    out = Matrix{Float64}(undef, size(A, 1), size(A, 2) * size(B, 2))
    for i in axes(out, 1), a in axes(A, 2), b in axes(B, 2)
        out[i, (a - 1) * size(B, 2) + b] = A[i, a] * B[i, b]
    end
    out
end

function _brm_t2_raw_blocks(margins, x, z)
    N1, R1 = _brm_apply_cr_spline(margins[1], x)
    N2, R2 = _brm_apply_cr_spline(margins[2], z)
    NN = _brm_row_tensor(N1, N2)
    (fixed=Matrix(NN[:, 2:end]), rr=_brm_row_tensor(R1, R2),
     rn=_brm_row_tensor(R1, N2), nr=_brm_row_tensor(N1, R2))
end

_brm_block_center(A::AbstractMatrix) = vec(sum(A; dims=1)) ./ size(A, 1)
_brm_center_block(A::AbstractMatrix, center) = A .- reshape(center, 1, :)

function _brm_fit_t2(x::AbstractVector{<:Real}, z::AbstractVector{<:Real};
                    k::Tuple{Int,Int}=(5, 5))
    length(x) == length(z) || error(
        "sbimpl: `t2(x, z)` margins must have equal lengths ($(length(x)) vs $(length(z)))")
    margins = (_brm_fit_cr_spline(x; k=k[1]), _brm_fit_cr_spline(z; k=k[2]))
    raw = _brm_t2_raw_blocks(margins, x, z)
    fixed_center = _brm_block_center(raw.fixed)
    (; margins, fixed_center, k)
end

function _brm_apply_t2(fit, x::AbstractVector{<:Real}, z::AbstractVector{<:Real})
    length(x) == length(z) || error(
        "sbimpl: `t2(x, z)` margins must have equal lengths ($(length(x)) vs $(length(z)))")
    raw = _brm_t2_raw_blocks(fit.margins, x, z)
    (_brm_center_block(raw.fixed, fit.fixed_center), raw.rr, raw.rn, raw.nr)
end

function _brm_t2_options(kw)
    kval = get(kw, :k, (5, 5))
    kval isa Tuple && length(kval) == 2 || error(
        "t2: `k` must be a 2-tuple of integers greater than 2, got $(repr(kval))")
    all(x -> x isa Integer && !(x isa Bool) && x > 2, kval) || error(
        "t2: `k` must be a 2-tuple of integers greater than 2, got $(repr(kval))")

    basis = get(kw, :basis, (:cr, :cr))
    basis isa Tuple && length(basis) == 2 || error(
        "t2: `basis` must be a 2-tuple; only `(:cr, :cr)` is currently supported")
    basis == (:cr, :cr) || error(
        "t2: only cubic-regression-spline margins `basis=(:cr, :cr)` are currently supported, got $(repr(basis))")

    full = get(kw, :full, false)
    full isa Bool || error("t2: `full` must be Bool, got $(typeof(full))")
    full && error("t2: `full=true` is not supported yet; use `full=false`")
    (Tuple(Int(x) for x in kval), basis, full)
end
