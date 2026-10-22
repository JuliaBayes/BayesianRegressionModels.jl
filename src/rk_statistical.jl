# Statistical preparation and algebra owned by BRM. The emitted model spells
# every parameter and prior; these ordinary Julia functions supply fitted
# data bases and differentiable covariance/spectral calculations.
# A basis is fitted to the fit-role columns (`x_fit`: the column itself when
# the model is built, the training column on a frozen `reprocess`) and
# evaluated at the bound rows.
function brm_tps_basis(x, x_fit, k)
    X, Z = _brm_apply_spline(_brm_fit_spline(x_fit; k), x)
    # RK's smooth convention keeps the centered linear null-space column;
    # the formula intercept owns the constant. Shared SB preparation retains
    # both columns, including its documented intercept-prior bridge.
    X[:, 2:2], Z
end
brm_tps_basis(x, k) = brm_tps_basis(x, x, k)

brm_t2_basis(x, z, x_fit, z_fit, k) = _brm_apply_t2(_brm_fit_t2(x_fit, z_fit; k), x, z)
brm_t2_basis(x, z, k) = brm_t2_basis(x, z, x, z, k)
# A `t2` basis fitted to its fit-role axes, as one numeric vector (the shape of
# data a program binds): each margin's shift, scale, null-space constants,
# knots, second-derivative map and range projection, then the fixed block's
# center. The emitted program evaluates the basis at the rows from it.
function brm_t2_fit(x_fit, z_fit, k)
    fit = _brm_fit_t2(x_fit, z_fit; k)
    vcat((_brm_pack_cr_fit(margin) for margin in fit.margins)..., fit.fixed_center)
end
brm_t2_basis(x, z, fit::AbstractVector{<:Real}, k) =
    _brm_apply_t2(_brm_unpack_t2_fit(fit, k), x, z)

_brm_pack_cr_fit(fit) = vcat(fit.shift, fit.scale, fit.null_const_scale, fit.slope_norm,
    fit.knots, vec(fit.F), vec(fit.range_projection))
function _brm_unpack_t2_fit(packed, k)
    offset = Ref(0)
    take(n) = (values = packed[offset[]+1:offset[]+n]; offset[] += n; values)
    margins = map(k) do m
        shift, scale, null_const_scale, slope_norm = take(4)
        knots = take(m)
        F = reshape(take(m^2), m, m)
        range_projection = reshape(take(m * (m - 2)), m, m - 2)
        (; shift, scale, knots, F, range_projection, null_const_scale, slope_norm, k=m)
    end
    (; margins, fixed_center=packed[offset[]+1:end], k)
end

_brm_covariate_var(values) = var(values)

function brm_hsgp_basis(axes, k, c, iso)
    K = k isa Tuple ? k : (k,)
    C = c isa Tuple ? c : (c,)
    state = _brm_hsgp_basis_state(axes, K, :exp_quad, iso, nothing;
        fits=_brm_fit_hsgp(axes, K, C))
    state.PHI, state.omega2, state.rho_lower
end

brm_hsgp_periodic_basis(x, k, period) = _brm_rk_periodic_basis(x, k, period, nothing, nothing)
# An anchored (`anchor::Real`) and/or row-masked (`mask`, the bound 0/1 rows)
# periodic basis (decision 1fk9b29).
brm_hsgp_periodic_basis(x, k, period, anchor::Real) =
    _brm_rk_periodic_basis(x, k, period, (anchor,), nothing)
brm_hsgp_periodic_basis(x, k, period, mask::AbstractVector) =
    _brm_rk_periodic_basis(x, k, period, nothing, mask .!= 0)
brm_hsgp_periodic_basis(x, k, period, anchor::Real, mask::AbstractVector) =
    _brm_rk_periodic_basis(x, k, period, (anchor,), mask .!= 0)

function _brm_rk_periodic_basis(x, k, period, anchor, mask)
    state = _brm_hsgp_basis_state((x,), (k,), :periodic, true, period; anchor, mask)
    state.PHI, state.harmonics, state.rho_lower
end

function brm_hsgp_sqrt_spd(omega2, sigma, rho::Real)
    scale = sigma * (rho * sqrt(2pi))^(size(omega2, 2) / 2)
    result = Vector{promote_type(typeof(sigma), typeof(rho))}(undef, size(omega2, 1))
    for b in axes(omega2, 1)
        exponent = zero(rho)
        for j in axes(omega2, 2)
            exponent += rho^2 * omega2[b, j]
        end
        result[b] = scale * exp(-0.25exponent)
    end
    result
end

function brm_hsgp_sqrt_spd(omega2, sigma, rhos)
    scale = sigma
    for rho in rhos
        scale *= sqrt(rho * sqrt(2pi))
    end
    result = Vector{promote_type(typeof(sigma), eltype(rhos))}(undef, size(omega2, 1))
    for b in axes(omega2, 1)
        exponent = zero(sigma)
        for j in axes(omega2, 2)
            exponent += rhos[j]^2 * omega2[b, j]
        end
        result[b] = scale * exp(-0.25exponent)
    end
    result
end

function brm_hsgp_periodic_sqrt_spd(harmonics, sigma, rho)
    result = Vector{promote_type(typeof(sigma), typeof(rho))}(undef, length(harmonics))
    for j in eachindex(harmonics)
        result[j] = exp(_brm_hsgp_periodic_log_scale(harmonics[j], sigma, rho))
    end
    result
end

# Legacy emitter bindings delegate to the adopted construction family.
# All backends importing these names use one statistical implementation.
brm_gp_covariance(x, sigma, rho, period, jitter) = iszero(period) ?
    StatisticalPreparation.gp_exp_quad_cov(x, sigma, rho, jitter) :
    StatisticalPreparation.gp_periodic_cov(x, sigma, rho, period, jitter)

brm_gp_latent(covariance, z) = StatisticalPreparation.gp_chol_latent(covariance, z)

brm_level_indices(labels, source) = _rk_value_level_indices(labels, source)
# A grouping's fitted levels: reading them as a grouping column gives the same
# levels back (`_rk_grouping_levels`), and a categorical column keeps its
# level order. The emitted program reads them in place of the fit-role column.
brm_fit_levels(source::CA.CategoricalArray) = CA.categorical(CA.levels(source);
    levels=CA.levels(source), ordered=CA.isordered(source))
brm_fit_levels(source::AbstractVector) = collect(CA.levels(source))
# No longer emitted: the gather is written inline as `draws[indices, margin]`.
# Kept for RK sources printed before that change, which import it by name.
brm_ranef_column(draws, indices, margin) = draws[indices, margin]
brm_dummy(values, level) = _rk_value_dummy(values, level)

# A rectangular panel's ordinary Julia slice, preserving subject order.
brm_panel_slice(column, timepoints, subject) =
    column[((subject - 1) * timepoints + 1):(subject * timepoints)]

# Per-subject cells concatenated in subject order with one allocation of their
# total length. `reduce(vcat, cells; init=…)` would fold one pairwise `vcat`
# per cell, reallocating the accumulator each time: quadratic in the number of
# cells, on every evaluation and reverse pass. Base's specialized
# `reduce(vcat, cells)` refuses zero cells, so that case is explicit. Scalar
# cells already hold one value per subject.
brm_flatten_cells(cells::AbstractVector{<:AbstractVector}) =
    isempty(cells) ? eltype(eltype(cells))[] : reduce(vcat, cells)
brm_flatten_cells(cells::AbstractVector{<:Number}) = collect(cells)
brm_flatten_cells(cells::AbstractVector) = collect(Iterators.flatten(cells))
