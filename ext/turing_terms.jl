# Turing submodels for backend-neutral fitted formula terms.

_brm_term_distribution(prior::BRM.ExprColumn) =
    BRM.getf(prior)(BRM.getargs(prior)...; BRM.getkwargs(prior)...)
_brm_term_distribution(prior::Distribution) = prior

Turing.@model function _brm_turing_s_term(state)
    b_fixed ~ product_distribution(fill(Turing.Flat(), size(state.Xnull, 2)))
    sd_pen ~ _brm_constrained_kernel(
        _brm_term_distribution(state.sd_prior); lower=0)
    b_pen_raw ~ product_distribution(fill(Normal(), size(state.Zpen, 2)))
    effect = state.Xnull * b_fixed + state.Zpen * (sd_pen .* b_pen_raw)
    (; effect, b_fixed, sd_pen, b_pen_raw)
end

Turing.@model function _brm_turing_t2_term(state)
    b_fixed ~ product_distribution(fill(Turing.Flat(), size(state.Xfixed, 2)))
    sd_pen ~ product_distribution([
        _brm_constrained_kernel(_brm_term_distribution(prior); lower=0)
        for prior in state.sd_priors])
    b_rr_raw ~ product_distribution(fill(Normal(), size(state.Zrr, 2)))
    b_rn_raw ~ product_distribution(fill(Normal(), size(state.Zrn, 2)))
    b_nr_raw ~ product_distribution(fill(Normal(), size(state.Znr, 2)))
    effect = state.Xfixed * b_fixed +
             state.Zrr * (sd_pen[1] .* b_rr_raw) +
             state.Zrn * (sd_pen[2] .* b_rn_raw) +
             state.Znr * (sd_pen[3] .* b_nr_raw)
    (; effect, b_fixed, sd_pen, b_rr_raw, b_rn_raw, b_nr_raw)
end

Turing.@model function _brm_turing_me_term(state)
    n = length(state.x_obs)
    x_true ~ product_distribution(fill(
        _brm_term_distribution(state.latent_prior), n))
    beta ~ Normal()
    for i in eachindex(state.x_obs)
        state.x_obs[i] ~ Normal(x_true[i], state.sd_x)
    end
    effect = beta .* x_true
    (; effect, x_true, beta)
end

Turing.@model function _brm_turing_monotonic_term(state, scaled)
    simplex_incr ~ BRM._brm_simplex_prior(
        _brm_term_distribution(state.simplex_prior), length(state.levels) - 1)
    contrast = cumsum(vcat(0.0, simplex_incr))[state.idx]
    if scaled
        beta ~ Normal()
        return (; effect=beta .* contrast, beta, simplex_incr)
    end
    (; effect=contrast, simplex_incr)
end

Turing.@model function _brm_turing_interval_term(state)
    base = _brm_term_distribution(state.latent_prior)
    bounds = [truncated(base; lower=state.x_lower[i], upper=state.x_upper[i])
              for i in eachindex(state.x_lower)]
    x_interval ~ product_distribution(bounds)
    x_true = zeros(length(state.Jexact) + length(state.Jinterval))
    x_true[state.Jexact] = state.x_exact
    x_true[state.Jinterval] = x_interval
    beta ~ Normal()
    (; effect=beta .* x_true, beta, x_true, x_interval)
end

Turing.@model function _brm_turing_ar_term(state)
    phi_raw ~ Normal()
    epsilon ~ product_distribution(fill(Normal(), length(state.time)))
    beta ~ Normal()
    phi = tanh(phi_raw)
    path = similar(epsilon)
    path[1] = epsilon[1]
    for i in 2:length(path)
        path[i] = phi * path[i - 1] + epsilon[i]
    end
    (; effect=beta .* path, beta, phi_raw, phi, epsilon, path)
end

Turing.@model function _brm_turing_dar_term(state)
    beta ~ _brm_constrained_kernel(
        _brm_term_distribution(state.ar_prior); lower=0, upper=1)
    sigma ~ _brm_constrained_kernel(
        _brm_term_distribution(state.sd_prior); lower=0)
    z ~ product_distribution(fill(Normal(), length(state.time) - 1))
    path = zeros(length(state.time))
    increment = 0.0
    for i in eachindex(z)
        increment = beta * increment + sigma * z[i]
        path[i + 1] = path[i] + increment
    end
    (; effect=path, beta, sigma, z, path)
end

# `rw`: the same walk with the increments' persistence fixed at zero.
Turing.@model function _brm_turing_rw_term(state)
    sigma ~ _brm_constrained_kernel(
        _brm_term_distribution(state.sd_prior); lower=0)
    z ~ product_distribution(fill(Normal(), state.n_steps - 1))
    path = zeros(eltype(z), state.n_steps)
    for i in eachindex(z)
        path[i + 1] = path[i] + sigma * z[i]
    end
    (; effect=path[state.time_idx], sigma, z, path)
end

# `cdar`: per-group deviations, damped over steps, innovations correlated across
# groups by the fitted Cholesky factor; each row reads its (group, step) cell.
Turing.@model function _brm_turing_cdar_term(state)
    sigma ~ _brm_constrained_kernel(
        _brm_term_distribution(state.sd_prior); lower=0)
    rho ~ _brm_constrained_kernel(
        _brm_term_distribution(state.ar_prior); lower=0, upper=1)
    eta ~ product_distribution(fill(Normal(), state.n_groups * state.n_steps))
    E = reshape(eta, state.n_groups, state.n_steps)
    delta = zeros(eltype(eta), state.n_groups, state.n_steps)
    delta[:, 1] = sigma .* (state.L * E[:, 1])
    scale = sigma * sqrt(1 - rho^2)
    for w in 2:state.n_steps
        delta[:, w] = rho .* delta[:, w - 1] .+ scale .* (state.L * E[:, w])
    end
    effect = [delta[state.group_idx[i], state.step_idx[i]] for i in eachindex(state.group_idx)]
    (; effect, sigma, rho, eta, delta)
end

function _brm_term_rows(term::BRM._BRMPreparedTerm{typeof(BRM.s)})
    size(term.state.Xnull, 1)
end

function _brm_term_rows(term::BRM._BRMPreparedTerm{typeof(BRM.t2)})
    size(term.state.Xfixed, 1)
end

function _brm_term_rows(term::BRM._BRMPreparedTerm{typeof(BRM.me)})
    length(term.state.x_obs)
end
_brm_term_rows(term::BRM._BRMPreparedTerm{typeof(BRM.mo)}) = length(term.state.idx)
_brm_term_rows(term::BRM._BRMPreparedTerm{typeof(BRM.mo1)}) = length(term.state.idx)
_brm_term_rows(term::BRM._BRMPreparedTerm{typeof(BRM.interval_censored)}) =
    term.state.nobs
_brm_term_rows(term::BRM._BRMPreparedTerm{typeof(BRM.ar)}) = length(term.state.time)
_brm_term_rows(term::BRM._BRMPreparedTerm{typeof(BRM.dar)}) = length(term.state.time)
_brm_term_rows(term::BRM._BRMPreparedTerm{typeof(BRM.rw)}) = length(term.state.time)
_brm_term_rows(term::BRM._BRMPreparedTerm{typeof(BRM.cdar)}) = length(term.state.group_idx)

function _brm_checked_term_model(term, nobs, model)
    rows = _brm_term_rows(term)
    rows == nobs || error(
        "Turing backend: prepared `$(nameof(term.callable))` term has $rows rows, " *
        "but its predictor has $nobs observations")
    model
end


function BRM._brm_turing_term_model(
        term::BRM._BRMPreparedTerm{typeof(BRM.s)}, nobs)
    _brm_checked_term_model(term, nobs, _brm_turing_s_term(term.state))
end


function BRM._brm_turing_term_model(
        term::BRM._BRMPreparedTerm{typeof(BRM.t2)}, nobs)
    _brm_checked_term_model(term, nobs, _brm_turing_t2_term(term.state))
end


function BRM._brm_turing_term_model(
        term::BRM._BRMPreparedTerm{typeof(BRM.me)}, nobs)
    _brm_checked_term_model(term, nobs, _brm_turing_me_term(term.state))
end

function BRM._brm_turing_term_model(
        term::BRM._BRMPreparedTerm{typeof(BRM.mo)}, nobs)
    _brm_checked_term_model(term, nobs,
        _brm_turing_monotonic_term(term.state, true))
end

function BRM._brm_turing_term_model(
        term::BRM._BRMPreparedTerm{typeof(BRM.mo1)}, nobs)
    _brm_checked_term_model(term, nobs,
        _brm_turing_monotonic_term(term.state, false))
end

function BRM._brm_turing_term_model(
        term::BRM._BRMPreparedTerm{typeof(BRM.interval_censored)}, nobs)
    _brm_checked_term_model(term, nobs,
        _brm_turing_interval_term(term.state))
end
function BRM._brm_turing_term_model(
        term::BRM._BRMPreparedTerm{typeof(BRM.ar)}, nobs)
    _brm_checked_term_model(term, nobs, _brm_turing_ar_term(term.state))
end
function BRM._brm_turing_term_model(
        term::BRM._BRMPreparedTerm{typeof(BRM.dar)}, nobs)
    _brm_checked_term_model(term, nobs, _brm_turing_dar_term(term.state))
end
function BRM._brm_turing_term_model(
        term::BRM._BRMPreparedTerm{typeof(BRM.rw)}, nobs)
    _brm_checked_term_model(term, nobs, _brm_turing_rw_term(term.state))
end
function BRM._brm_turing_term_model(
        term::BRM._BRMPreparedTerm{typeof(BRM.cdar)}, nobs)
    _brm_checked_term_model(term, nobs, _brm_turing_cdar_term(term.state))
end


function _brm_term_with_priors(term, replacements)
    BRM._BRMPreparedTerm(term.callable, term.source,
        merge(term.state, replacements), term.dependencies)
end

BRM._brm_turing_term_model(term::BRM._BRMPreparedTerm{typeof(BRM.s)}, nobs,
                           priors) = BRM._brm_turing_term_model(
    _brm_term_with_priors(term, (; sd_prior=priors.sd)), nobs)
BRM._brm_turing_term_model(term::BRM._BRMPreparedTerm{typeof(BRM.t2)}, nobs,
                           priors) = BRM._brm_turing_term_model(
    _brm_term_with_priors(term, (; sd_priors=priors.sd)), nobs)
BRM._brm_turing_term_model(term::BRM._BRMPreparedTerm{typeof(BRM.me)}, nobs,
                           priors) = BRM._brm_turing_term_model(
    _brm_term_with_priors(term, (; latent_prior=priors.latent)), nobs)
function BRM._brm_turing_term_model(
        term::BRM._BRMPreparedTerm{F}, nobs, priors) where {F<:Union{
            typeof(BRM.mo),typeof(BRM.mo1)}}
    BRM._brm_turing_term_model(
        _brm_term_with_priors(term, (; simplex_prior=priors.simplex)), nobs)
end
BRM._brm_turing_term_model(
        term::BRM._BRMPreparedTerm{typeof(BRM.interval_censored)}, nobs,
        priors) = BRM._brm_turing_term_model(
    _brm_term_with_priors(term, (; latent_prior=priors.latent)), nobs)
BRM._brm_turing_term_model(term::BRM._BRMPreparedTerm{typeof(BRM.dar)}, nobs,
                           priors) = BRM._brm_turing_term_model(
    _brm_term_with_priors(term,
        (; ar_prior=priors.ar, sd_prior=priors.sd)), nobs)
BRM._brm_turing_term_model(term::BRM._BRMPreparedTerm{typeof(BRM.rw)}, nobs,
                           priors) = BRM._brm_turing_term_model(
    _brm_term_with_priors(term, (; sd_prior=priors.sd)), nobs)
BRM._brm_turing_term_model(term::BRM._BRMPreparedTerm{typeof(BRM.cdar)}, nobs,
                           priors) = BRM._brm_turing_term_model(
    _brm_term_with_priors(term,
        (; ar_prior=priors.ar, sd_prior=priors.sd)), nobs)
BRM._brm_turing_term_model(term::BRM._BRMPreparedTerm{typeof(BRM.ar)}, nobs,
                           _priors) = BRM._brm_turing_term_model(term, nobs)
