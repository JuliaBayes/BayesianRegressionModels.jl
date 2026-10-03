# Statistical preparation and algebra owned by BRM. The emitted model spells
# every parameter and prior; these ordinary Julia functions supply fitted
# data bases and differentiable covariance/spectral calculations.
function brm_tps_basis(x, k)
    X, Z = _brm_spline_basis_tps(x; k)
    # RK's smooth convention keeps the centered linear null-space column;
    # the formula intercept owns the constant. Shared SB preparation retains
    # both columns, including its documented intercept-prior bridge.
    X[:, 2:2], Z
end

brm_t2_basis(x, z, k) = _brm_apply_t2(_brm_fit_t2(x, z; k), x, z)

function brm_hsgp_basis(axes, k, c, iso)
    K = k isa Tuple ? k : (k,)
    C = c isa Tuple ? c : (c,)
    state = _brm_hsgp_basis_state(axes, K, :exp_quad, iso, nothing;
        fits=_brm_fit_hsgp(axes, K, C))
    state.PHI, state.omega2, state.rho_lower
end

function brm_hsgp_periodic_basis(x, k, period)
    state = _brm_hsgp_basis_state((x,), (k,), :periodic, true, period)
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

function brm_gp_covariance(x, sigma, rho, period, jitter)
    n = length(x)
    T = promote_type(eltype(x), typeof(sigma), typeof(rho))
    covariance = Matrix{T}(undef, n, n)
    for j in 1:n, i in 1:n
        exponent = iszero(period) ?
            0.5 * ((x[i] - x[j]) / rho)^2 :
            2sinpi(abs(x[i] - x[j]) / period)^2 / rho^2
        covariance[i, j] = sigma^2 * exp(-exponent)
    end
    covariance + jitter * I
end

brm_gp_latent(covariance, z) = cholesky(Symmetric(covariance)).L * z

brm_level_indices(labels, source) = _rk_value_level_indices(labels, source)
brm_dummy(values, level) = _rk_value_dummy(values, level)

# A rectangular panel's ordinary Julia slice, preserving subject order.
brm_panel_slice(column, timepoints, subject) =
    column[((subject - 1) * timepoints + 1):(subject * timepoints)]
