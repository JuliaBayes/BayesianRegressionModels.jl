# Native Turing submodels for shared fitted GP/HSGP records.

function _brm_gp_covariance(state, sigma, rho)
    n, d = size(state.X)
    rhos = rho isa Real ? fill(rho, d) : rho
    T = promote_type(eltype(state.X), typeof(sigma), eltype(rhos))
    K = Matrix{T}(undef, n, n)
    for j in 1:n, i in 1:n
        exponent = if state.cov === :periodic
            2sinpi(abs(state.X[i, 1] - state.X[j, 1]) / state.period)^2 / rho^2
        else
            0.5sum(((state.X[i, k] - state.X[j, k]) / rhos[k])^2 for k in 1:d)
        end
        K[i, j] = sigma^2 * exp(-exponent)
    end
    K + state.jitter * I
end

Turing.@model function _brm_turing_gp_term(state)
    rho_distribution = _brm_constrained_kernel(
        _brm_term_distribution(state.rho_prior); lower=0)
    rho ~ state.iso ? rho_distribution :
        product_distribution(fill(rho_distribution, size(state.X, 2)))
    sigma ~ _brm_constrained_kernel(
        _brm_term_distribution(state.sigma_prior); lower=0)
    z ~ product_distribution(fill(Normal(), size(state.X, 1)))
    effect = cholesky(Symmetric(_brm_gp_covariance(state, sigma, rho))).L * z
    (; effect, rho, sigma, z)
end

function _brm_hsgp_sqrt_spd(state, sigma, rho::Real)
    if state.cov === :periodic
        a = inv(rho^2)
        return [sigma * sqrt(2 * BRM.SpecialFunctions.besselix(Int(j), a))
                for j in state.harmonics]
    end
    scale = sigma * sqrt(rho * sqrt(2pi))
    rho_squared = rho^2
    [scale * exp(-0.25 * rho_squared * state.omega2[b, 1])
     for b in axes(state.omega2, 1)]
end

function _brm_hsgp_sqrt_spd(state, sigma, rhos)
    scale = sigma * prod(sqrt.(rhos .* sqrt(2pi)))
    [scale * exp(-0.25sum(rhos .^ 2 .* state.omega2[b, :]))
     for b in axes(state.omega2, 1)]
end

function _brm_hsgp_exp_quad_log_sqrt_spd(state, sigma, rhos)
    log_scale = log(sigma) +
        0.5sum(log(rhos[j]) + 0.5log(2pi) for j in eachindex(rhos))
    [log_scale - 0.25sum(rhos .^ 2 .* state.omega2[b, :])
     for b in axes(state.omega2, 1)]
end

function _brm_hsgp_exp_quad_log_sqrt_spd(state, sigma, rho::Real)
    log_scale = log(sigma) + 0.5 * (log(rho) + 0.5log(2pi))
    rho_squared = rho^2
    [log_scale - 0.25 * rho_squared * state.omega2[b, 1]
     for b in axes(state.omega2, 1)]
end

function _brm_hsgp_log_sqrt_spd(state, sigma, rho)
    if state.cov === :periodic
        a = inv(rho^2)
        return [log(sigma) + 0.5 * (log(2) - a +
                    log(BRM.SpecialFunctions.besselix(Int(j), a)))
                for j in state.harmonics]
    end
    _brm_hsgp_exp_quad_log_sqrt_spd(state, sigma, rho)
end

_brm_hsgp_centered_log_scale(log_scale, c) =
    iszero(c) ? zero(log_scale) : c * log_scale
_brm_hsgp_remaining_log_scale(log_scale, c) =
    isone(c) ? zero(log_scale) : (one(c) - c) * log_scale

Turing.@model function _brm_turing_hsgp_partial_iso_term(state)
    rho ~ _brm_constrained_kernel(
        _brm_term_distribution(state.rho_prior);
        lower=state.rho_lower)
    sigma ~ _brm_constrained_kernel(
        _brm_term_distribution(state.sigma_prior); lower=0)
    log_sqrt_spd = _brm_hsgp_exp_quad_log_sqrt_spd(state, sigma, rho)
    centered_log_scale = [_brm_hsgp_centered_log_scale(log_sqrt_spd[b],
                            state.centeredness[b]) for b in eachindex(log_sqrt_spd)]
    log_floor = log(floatmin(Float64))
    if !(all(isfinite, centered_log_scale) &&
         minimum(centered_log_scale) >= log_floor)
        Turing.@addlogprob! -Inf
    end
    safe_centered_log_scale = max.(centered_log_scale, log_floor)
    remaining_log_scale = [_brm_hsgp_remaining_log_scale(log_sqrt_spd[b],
                             state.centeredness[b]) for b in eachindex(log_sqrt_spd)]
    if isnothing(state.by)
        beta_partial ~ arraydist([
            Normal(0, exp(safe_centered_log_scale[b]))
            for b in eachindex(safe_centered_log_scale)])
        weights = exp.(remaining_log_scale) .* beta_partial
        effect = state.PHI * weights
        return (; effect, rho, sigma, beta_partial, log_sqrt_spd,
                  centeredness=state.centeredness, weights)
    end
    n_levels = length(state.by.levels)
    n_basis = length(safe_centered_log_scale)
    beta_partial ~ arraydist([Normal(0, exp(safe_centered_log_scale[b]))
                              for _g in 1:n_levels for b in 1:n_basis])
    by_weights = reshape(beta_partial, n_basis, n_levels)
    weights = exp.(remaining_log_scale) .* by_weights
    effect = [dot(state.PHI[i, :], weights[:, state.by.idx[i]])
              for i in axes(state.PHI, 1)]
    (; effect, rho, sigma, beta_partial, log_sqrt_spd,
       centeredness=state.centeredness, weights)
end

Turing.@model function _brm_turing_hsgp_partial_aniso_term(state)
    rho ~ product_distribution([
        _brm_constrained_kernel(_brm_term_distribution(state.rho_prior);
                                lower=state.rho_lower[j])
        for j in eachindex(state.rho_lower)])
    sigma ~ _brm_constrained_kernel(
        _brm_term_distribution(state.sigma_prior); lower=0)
    log_sqrt_spd = _brm_hsgp_exp_quad_log_sqrt_spd(state, sigma, rho)
    centered_log_scale = [_brm_hsgp_centered_log_scale(log_sqrt_spd[b],
                            state.centeredness[b]) for b in eachindex(log_sqrt_spd)]
    # A positive centeredness cannot represent a coordinate whose scale has
    # rounded to zero. Reject that geometry explicitly while keeping c=0 safe
    # even when the physical high-frequency weight itself underflows.
    log_floor = log(floatmin(Float64))
    if !(all(isfinite, centered_log_scale) &&
         minimum(centered_log_scale) >= log_floor)
        Turing.@addlogprob! -Inf
    end
    safe_centered_log_scale = max.(centered_log_scale, log_floor)
    remaining_log_scale = [_brm_hsgp_remaining_log_scale(log_sqrt_spd[b],
                             state.centeredness[b]) for b in eachindex(log_sqrt_spd)]
    if isnothing(state.by)
        beta_partial ~ arraydist([
            Normal(0, exp(safe_centered_log_scale[b]))
            for b in eachindex(safe_centered_log_scale)])
        weights = exp.(remaining_log_scale) .* beta_partial
        effect = state.PHI * weights
        return (; effect, rho, sigma, beta_partial, log_sqrt_spd,
                  centeredness=state.centeredness, weights)
    end
    n_levels = length(state.by.levels)
    n_basis = length(safe_centered_log_scale)
    beta_partial ~ arraydist([Normal(0, exp(safe_centered_log_scale[b]))
                              for _g in 1:n_levels for b in 1:n_basis])
    by_weights = reshape(beta_partial, n_basis, n_levels)
    weights = exp.(remaining_log_scale) .* by_weights
    effect = [dot(state.PHI[i, :], weights[:, state.by.idx[i]])
              for i in axes(state.PHI, 1)]
    (; effect, rho, sigma, beta_partial, log_sqrt_spd,
       centeredness=state.centeredness, weights)
end

Turing.@model function _brm_turing_hsgp_term(state)
    rho ~ state.iso ? _brm_constrained_kernel(
        _brm_term_distribution(state.rho_prior); lower=state.rho_lower) :
        product_distribution([
            _brm_constrained_kernel(_brm_term_distribution(state.rho_prior);
                                    lower=state.rho_lower[j])
            for j in eachindex(state.rho_lower)])
    sigma ~ _brm_constrained_kernel(
        _brm_term_distribution(state.sigma_prior); lower=0)
    sqrt_spd = _brm_hsgp_sqrt_spd(state, sigma, rho)
    if isnothing(state.by)
        beta_raw ~ filldist(Normal(), size(state.PHI, 2))
        effect = state.PHI * (sqrt_spd .* beta_raw)
        return (; effect, rho, sigma, beta_raw, sqrt_spd)
    end
    n_levels = length(state.by.levels)
    beta_raw ~ product_distribution(fill(Normal(), n_levels * size(state.PHI, 2)))
    weights = reshape(beta_raw, size(state.PHI, 2), n_levels)
    effect = [dot(state.PHI[i, :], sqrt_spd .* weights[:, state.by.idx[i]])
              for i in axes(state.PHI, 1)]
    (; effect, rho, sigma, beta_raw, sqrt_spd)
end

function _brm_hsgp_latent_basis(state, x)
    T = promote_type(eltype(x), Float64)
    PHI = Matrix{T}(undef, length(x), only(state.K))
    inv_sqrt_L = inv(sqrt(state.L))
    for k in axes(PHI, 2), i in eachindex(x)
        frequency = sqrt(state.omega2[k, 1])
        PHI[i, k] = inv_sqrt_L * sin(frequency * (x[i] - state.center + state.L))
    end
    state.orthogonal === :linear || return PHI
    xc = x .- sum(x) / length(x)
    ss = sum(abs2, xc)
    for k in axes(PHI, 2)
        column = @view PHI[:, k]
        column .-= sum(column) / length(column)
        ss > 0 && (column .-= xc .* (dot(xc, column) / ss))
    end
    PHI
end

Turing.@model function _brm_turing_hsgp_latent_term(state, x)
    rho ~ _brm_constrained_kernel(
        _brm_term_distribution(state.rho_prior);
        lower=state.rho_lower)
    sigma ~ _brm_constrained_kernel(
        _brm_term_distribution(state.sigma_prior); lower=0)
    sqrt_spd = _brm_hsgp_sqrt_spd(state, sigma, rho)
    beta_raw ~ product_distribution(fill(Normal(), only(state.K)))
    PHI = _brm_hsgp_latent_basis(state, x)
    effect = PHI * (sqrt_spd .* beta_raw)
    (; effect, rho, sigma, beta_raw, sqrt_spd)
end

_brm_term_rows(term::BRM._BRMPreparedTerm{typeof(BRM.gp)}) = size(term.state.X, 1)
_brm_term_rows(term::BRM._BRMPreparedTerm{typeof(BRM.hsgp)}) =
    get(term.state, :latent, false) ? nothing : size(term.state.PHI, 1)

function BRM._brm_turing_term_model(
        term::BRM._BRMPreparedTerm{typeof(BRM.gp)}, nobs)
    _brm_checked_term_model(term, nobs, _brm_turing_gp_term(term.state))
end

function BRM._brm_turing_term_model(
        term::BRM._BRMPreparedTerm{typeof(BRM.hsgp)}, nobs)
    model = any(!iszero, term.state.centeredness) ?
        (term.state.iso ? _brm_turing_hsgp_partial_iso_term(term.state) :
                          _brm_turing_hsgp_partial_aniso_term(term.state)) :
        _brm_turing_hsgp_term(term.state)
    _brm_checked_term_model(term, nobs, model)
end

function BRM._brm_turing_term_model(
        term::BRM._BRMPreparedTerm{typeof(BRM.gp)}, nobs, priors)
    prepared = _brm_term_with_priors(term,
        (; rho_prior=priors.rho, sigma_prior=priors.sigma))
    BRM._brm_turing_term_model(prepared, nobs)
end
function BRM._brm_turing_term_model(
        term::BRM._BRMPreparedTerm{typeof(BRM.hsgp)}, nobs, priors)
    prepared = _brm_term_with_priors(term,
        (; rho_prior=priors.rho, sigma_prior=priors.sigma))
    BRM._brm_turing_term_model(prepared, nobs)
end

function BRM._brm_turing_term_model(
        term::BRM._BRMPreparedTerm{typeof(BRM.hsgp)}, nobs, priors, inputs)
    prepared = _brm_term_with_priors(term,
        (; rho_prior=priors.rho, sigma_prior=priors.sigma))
    get(prepared.state, :latent, false) ||
        return BRM._brm_turing_term_model(prepared, nobs)
    x = getproperty(inputs, prepared.state.axis_source)
    length(x) == nobs || error("Turing backend: hsgp term has $(length(x)) rows; expected $nobs")
    _brm_turing_hsgp_latent_term(prepared.state, x)
end

function _brm_turing_hsgp_ncp_model(term, nobs, priors, _inputs)
    prepared = _brm_term_with_priors(term,
        (; rho_prior=priors.rho, sigma_prior=priors.sigma))
    _brm_checked_term_model(
        prepared, nobs, _brm_turing_hsgp_term(prepared.state))
end

function _brm_turing_hsgp_partial_iso_model(term, nobs, priors, _inputs)
    prepared = _brm_term_with_priors(term,
        (; rho_prior=priors.rho, sigma_prior=priors.sigma))
    _brm_checked_term_model(
        prepared, nobs, _brm_turing_hsgp_partial_iso_term(prepared.state))
end

function _brm_turing_hsgp_partial_aniso_model(term, nobs, priors, _inputs)
    prepared = _brm_term_with_priors(term,
        (; rho_prior=priors.rho, sigma_prior=priors.sigma))
    _brm_checked_term_model(
        prepared, nobs, _brm_turing_hsgp_partial_aniso_term(prepared.state))
end

function _brm_turing_hsgp_latent_model(term, nobs, priors, inputs)
    prepared = _brm_term_with_priors(term,
        (; rho_prior=priors.rho, sigma_prior=priors.sigma))
    x = getproperty(inputs, prepared.state.axis_source)
    length(x) == nobs || error(
        "Turing backend: hsgp term has $(length(x)) rows; expected $nobs")
    _brm_turing_hsgp_latent_term(prepared.state, x)
end

function _brm_turing_term_call_ast(
        term::BRM._BRMPreparedTerm{typeof(BRM.hsgp)}, term_ast, nobs,
        priors, inputs)
    model = if get(term.state, :latent, false)
        :_brm_turing_hsgp_latent_model
    elseif any(!iszero, term.state.centeredness)
        term.state.iso ? :_brm_turing_hsgp_partial_iso_model :
                         :_brm_turing_hsgp_partial_aniso_model
    else
        :_brm_turing_hsgp_ncp_model
    end
    :($model($term_ast, $nobs, $priors, $inputs))
end
