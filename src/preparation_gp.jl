# Shared fitted preparation for exact and Hilbert-space Gaussian-process terms.

_brm_prepares_term(term::ExprColumn{typeof(gp)}) = true
_brm_prepares_term(term::ExprColumn{typeof(hsgp)}) = true

_brm_term_prior_expressions(term::_BRMPreparedTerm{typeof(gp)}) =
    (; rho=term.state.rho_prior, sigma=term.state.sigma_prior)
_brm_term_prior_expressions(term::_BRMPreparedTerm{typeof(hsgp)}) =
    (; rho=term.state.rho_prior, sigma=term.state.sigma_prior)

function _brm_gp_axes(term, context, label)
    args = getargs(term)
    isempty(args) && error("BRM term preparation: `$label` needs at least one axis")
    sources = Symbol[]
    axes = Vector{Vector{Float64}}()
    for arg in args
        source, raw = _brm_term_data(label, arg, context)
        values = collect(Float64, raw)
        isempty(values) && error("BRM term preparation: `$label` axis cannot be empty")
        all(isfinite, values) || error("BRM term preparation: `$label` axes must be finite")
        push!(sources, source); push!(axes, values)
    end
    all(axis -> length(axis) == length(first(axes)), axes) || error(
        "BRM term preparation: `$label` axes must have equal lengths")
    Tuple(sources), Tuple(axes)
end

function _brm_gp_priors(term, target, context)
    rho_prior = _brm_term_prior_expression(term, target, context,
        :term_length_scale; default=ExprColumn(LogNormal, 0.0, 1.0))
    sigma_prior = _brm_term_prior_expression(term, target, context,
        :term_sd; default=ExprColumn(LogNormal, 0.0, 1.0))
    (; rho_prior, sigma_prior)
end

function _brm_hsgp_basis_state(axes, K, cov, iso, period;
                               fits=nothing, orthogonal=nothing)
    if cov === :periodic
        PHI = _brm_apply_hsgp_periodic(period, only(axes), only(K))
        return (; axes, PHI, harmonics=_brm_hsgp_periodic_harmonics(only(K)),
                rho_lower=_brm_hsgp_periodic_rho_lower(only(K)), fits=nothing,
                omega2=nothing)
    end
    isnothing(fits) && error("BRM term preparation: HSGP domain fit is missing")
    PHI, omega2 = _brm_apply_hsgp(fits, axes, K)
    orthogonal === :linear &&
        (PHI = _brm_orthogonalize_hsgp_linear(PHI, only(axes)))
    (; axes, PHI, harmonics=nothing,
       rho_lower=_brm_hsgp_rho_lower_data(fits, K, iso), fits, omega2)
end

function _brm_prepare_term(term::ExprColumn{typeof(gp)}, target::Symbol, context)
    sources, axes = _brm_gp_axes(term, context, :gp)
    kw = getkwargs(term)
    cov = _brm_gp_cov(kw, :gp)
    iso = _brm_gp_iso(kw, :gp)
    period = _brm_gp_period(kw, :gp, cov)
    cov === :periodic && (length(axes) == 1 && iso) || cov !== :periodic ||
        error("BRM term preparation: periodic gp requires one isotropic axis")
    jitter = Float64(get(kw, :jitter, 1e-9))
    jitter >= 0 && isfinite(jitter) || error(
        "BRM term preparation: gp jitter must be finite and nonnegative")
    state = (; target, X=_brm_gp_matrix(axes), cov, iso, period, jitter,
             _brm_gp_priors(term, target, context)...)
    _BRMPreparedTerm(gp, sources, state, sources)
end

function _brm_prepare_term(term::ExprColumn{typeof(hsgp)}, target::Symbol, context)
    kw = getkwargs(term)
    args = getargs(term)
    latent_args = Tuple(arg for arg in args if arg isa NamedColumn &&
        !haskey(context.data, name(arg)) && parent(arg) isa ExprColumn)
    if !isempty(latent_args)
        length(args) == 1 && length(latent_args) == 1 || error(
            "BRM term preparation: model-derived hsgp supports exactly one axis")
        iso = _brm_gp_iso(kw, :hsgp)
        iso || error(
            "BRM term preparation: model-derived hsgp requires iso=true")
        haskey(kw, :by) && error(
            "BRM term preparation: model-derived hsgp does not support by")
        cov = _brm_gp_cov(kw, :hsgp)
        cov === :periodic && error(
            "BRM term preparation: model-derived periodic hsgp is unsupported")
        K, c = _brm_hsgp_options(kw, 1)
        fits = _brm_hsgp_domain_fits(kw, 1; required=true)
        orthogonal = _brm_hsgp_orthogonal_to(kw, 1)
        center, L = only(fits)
        _, omega2 = _brm_apply_hsgp(fits, ([center],), K)
        source = name(only(latent_args))
        state = (; target, latent=true, explicit_domain=true, axis_source=source, K, c, cov, iso,
                 period=nothing, fits, center, L, omega2,
                 rho_lower=_brm_hsgp_rho_lower_data(fits, K, true),
                 orthogonal, by=nothing,
                 _brm_gp_priors(term, target, context)...)
        return _BRMPreparedTerm(hsgp, (source,), state, (source,))
    end
    sources, axes = _brm_gp_axes(term, context, :hsgp)
    cov = _brm_gp_cov(kw, :hsgp)
    iso = _brm_gp_iso(kw, :hsgp)
    K, c = _brm_hsgp_options(kw, length(axes))
    period = _brm_gp_period(kw, :hsgp, cov)
    by = get(kw, :by, nothing)
    by_state = if isnothing(by)
        nothing
    else
        source, raw = _brm_term_data(:hsgp, by, context)
        levels = _brm_fit_levels(raw)
        (; source, levels, idx=_brm_apply_levels(levels, raw))
    end
    if cov === :periodic
        length(axes) == 1 && iso || error(
            "BRM term preparation: periodic hsgp requires one isotropic axis")
        isnothing(by_state) || error(
            "BRM term preparation: periodic hsgp does not support by")
        basis = _brm_hsgp_basis_state(axes, K, cov, iso, period)
        state = (; target, latent=false, explicit_domain=false, K, c, cov, iso, period, basis..., by=by_state,
                 _brm_gp_priors(term, target, context)...)
    else
        domain_fits = _brm_hsgp_domain_fits(kw, length(axes))
        fits = isnothing(domain_fits) ? _brm_fit_hsgp(axes, K, c) : domain_fits
        orthogonal = _brm_hsgp_orthogonal_to(kw, length(axes))
        basis = _brm_hsgp_basis_state(axes, K, cov, iso, period;
                                      fits, orthogonal)
        state = (; target, latent=false, explicit_domain=!isnothing(domain_fits), K, c, cov, iso, period, basis...,
                 orthogonal, by=by_state,
                 _brm_gp_priors(term, target, context)...)
    end
    deps = isnothing(by_state) ? sources : (sources..., by_state.source)
    _BRMPreparedTerm(hsgp, sources, state, deps)
end

function _brm_replay_gp_axes(training, context, label)
    axes = Tuple(collect(Float64, context.data[source]) for source in training.source)
    all(!isempty, axes) || error("BRM term replay: `$label` axes cannot be empty")
    all(axis -> all(isfinite, axis), axes) || error(
        "BRM term replay: `$label` axes must be finite")
    all(axis -> length(axis) == length(first(axes)), axes) || error(
        "BRM term replay: `$label` axes must have equal lengths")
    axes
end

function _brm_replay_term(::typeof(gp), training,
                          fresh::Union{ExprColumn,_BRMPreparedTerm}, context)
    axes = _brm_replay_gp_axes(training, context, :gp)
    state = merge(training.state, (; X=_brm_gp_matrix(axes)))
    _BRMPreparedTerm(gp, training.source, state, training.dependencies)
end

function _brm_replay_term(::typeof(hsgp), training,
                          fresh::Union{ExprColumn,_BRMPreparedTerm}, context)
    get(training.state, :latent, false) && return _BRMPreparedTerm(
        hsgp, training.source, training.state, training.dependencies)
    axes = _brm_replay_gp_axes(training, context, :hsgp)
    get(training.state, :explicit_domain, false) &&
        _brm_check_hsgp_domain(training.state.fits, axes; prefix="BRM term replay")
    state = if training.state.cov === :periodic
        merge(training.state, (; axes,
            PHI=_brm_apply_hsgp_periodic(training.state.period, only(axes),
                                         only(training.state.K))))
    else
        PHI, omega2 = _brm_apply_hsgp(training.state.fits, axes, training.state.K)
        get(training.state, :orthogonal, nothing) === :linear &&
            (PHI = _brm_orthogonalize_hsgp_linear(PHI, only(axes)))
        merge(training.state, (; axes, PHI, omega2))
    end
    if !isnothing(training.state.by)
        by = training.state.by
        idx = _brm_apply_levels(by.levels, context.data[by.source])
        state = merge(state, (; by=merge(by, (; idx))))
    end
    _BRMPreparedTerm(hsgp, training.source, state, training.dependencies)
end
