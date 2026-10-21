# Shared fitted preparation for exact and Hilbert-space Gaussian-process terms.

_brm_prepares_term(term::ExprColumn{typeof(gp)}) = true
_brm_prepares_term(term::ExprColumn{typeof(hsgp)}) = true

_brm_term_prior_expressions(term::_BRMPreparedTerm{typeof(gp)}) =
    (; rho=term.state.rho_prior, sigma=term.state.sigma_prior)
_brm_term_prior_expressions(term::_BRMPreparedTerm{typeof(hsgp)}) =
    (; rho=term.state.rho_prior, sigma=term.state.sigma_prior)

# `allow_empty` admits a zero-row axis for a basis that needs no data: an HSGP
# with an explicit `domain=` or a periodic one (decision 1fk9b29).
function _brm_gp_axes(term, context, label; allow_empty::Bool=false, mask=nothing)
    args = getargs(term)
    isempty(args) && error("BRM term preparation: `$label` needs at least one axis")
    sources = Symbol[]
    axes = Vector{Vector{Float64}}()
    for arg in args
        source, raw = _brm_term_data(label, arg, context)
        values = _brm_axis_values(raw, mask,
            "BRM term preparation: `$label` axis `$source`")
        isempty(values) && !allow_empty && error(
            "BRM term preparation: `$label` axis cannot be empty" *
            (label === :hsgp ? "; an HSGP over an axis that can be empty needs " *
             "an explicit `domain=(lower, upper)` (or `cov=:periodic`)" : ""))
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

_brm_zero_rho_lower(lower::Real) = zero(lower)
_brm_zero_rho_lower(lower::AbstractArray) = zero.(lower)

function _brm_hsgp_declared_rho_lower(term, target, context, default_lower,
                                      geometry, remedy)
    # An explicit length-scale prior replaces the whole declaration, including
    # the approximation-validity floor supplied by the default HSGP prior --
    # unless it names that floor through `length_scale_floor` (decision 1fk9b29).
    spec = _brm_term_prior_spec(term, target, context, :term_length_scale)
    isnothing(spec) && return default_lower
    _brm_has_floor_marker(spec.expression) ||
        return _brm_zero_rho_lower(default_lower)
    _brm_check_hsgp_floor_upper(spec.expression, default_lower,
        _brm_term_prior_spelling(spec), geometry, remedy;
        prefix="BRM term preparation")
    default_lower
end

# The basis of an HSGP term without its evaluated rows: harmonics, validity
# floor and fits. A backend that evaluates the basis itself (RK, in its graph)
# prepares terms with this alone.
function _brm_hsgp_basis_geometry(axes, K, cov, iso, period;
                                  fits=nothing, orthogonal=nothing,
                                  anchor=nothing, mask=nothing)
    cov === :periodic && return (; axes,
        harmonics=_brm_hsgp_periodic_harmonics(only(K)),
        rho_lower=_brm_hsgp_periodic_rho_lower(only(K)), fits=nothing)
    isnothing(fits) && error("BRM term preparation: HSGP domain fit is missing")
    (; axes, harmonics=nothing, rho_lower=_brm_hsgp_rho_lower_data(fits, K, iso), fits)
end

# `anchor` subtracts the basis row at the anchor from every row, so the term is
# `f(x) - f(x0)` for the same weights; `mask` zeroes the rows that carry no
# predictor value (decision 1fk9b29). Both are data-only transforms of `PHI`.
function _brm_hsgp_basis_state(axes, K, cov, iso, period;
                               fits=nothing, orthogonal=nothing,
                               anchor=nothing, mask=nothing)
    geometry = _brm_hsgp_basis_geometry(axes, K, cov, iso, period; fits)
    if cov === :periodic
        PHI = _brm_apply_hsgp_periodic(period, only(axes), only(K))
        isnothing(anchor) || (PHI = PHI .- _brm_apply_hsgp_periodic(
            period, [only(anchor)], only(K)))
        isnothing(mask) || (PHI[.!mask, :] .= 0)
        return (; axes, PHI, geometry.harmonics, geometry.rho_lower, fits=nothing,
                omega2=nothing)
    end
    PHI, omega2 = _brm_apply_hsgp(fits, axes, K)
    orthogonal === :linear &&
        (PHI = _brm_orthogonalize_hsgp_linear(PHI, only(axes)))
    isnothing(anchor) || (PHI = PHI .- first(_brm_apply_hsgp(
        fits, Tuple([value] for value in anchor), K)))
    isnothing(mask) || (PHI[.!mask, :] .= 0)
    (; axes, PHI, geometry.harmonics, geometry.rho_lower, fits, omega2)
end

# Parse the replay/anchor/mask options once and check their combinations; the
# prepared-term path and the Stan emitter share this (decision 1fk9b29).
function _brm_hsgp_extended_options(kw, n_axes::Int, cov::Symbol;
                                    latent::Bool=false)
    extrapolate = _brm_hsgp_extrapolate(kw, cov)
    anchor = _brm_hsgp_anchor(kw, n_axes)
    rows = _brm_hsgp_rows_column(kw)
    if latent
        for key in (:anchor, :rows, :extrapolate)
            haskey(kw, key) && error(
                "sbimpl: `hsgp(...; $key=...)` is not built for a model-derived " *
                "axis; it is available on raw-data axes")
        end
    end
    orthogonal = get(kw, :orthogonal_to, nothing)
    orthogonal === :linear && !isnothing(anchor) && error(
        "sbimpl: `hsgp(...; anchor=...)` cannot be combined with " *
        "`orthogonal_to=:linear`: the projection centers every basis column, " *
        "so the term would no longer vanish at the anchor")
    orthogonal === :linear && !isnothing(rows) && error(
        "sbimpl: `hsgp(...; rows=...)` with `orthogonal_to=:linear` is not " *
        "built: the projection would have to be restricted to the masked rows")
    (; extrapolate, anchor, rows)
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

function _brm_prepare_term(term::ExprColumn{typeof(hsgp)}, target::Symbol, context;
                           basis=_brm_hsgp_basis_state)
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
        _brm_hsgp_extended_options(kw, 1, cov; latent=true)
        K, c = _brm_hsgp_options(kw, 1)
        centeredness = _brm_hsgp_centeredness(kw, prod(K))
        any(!iszero, centeredness) && error(
            "BRM term preparation: partial centering currently requires a " *
            "raw-data HSGP axis")
        fits = _brm_hsgp_domain_fits(kw, 1; required=true)
        orthogonal = _brm_hsgp_orthogonal_to(kw, 1)
        center, L = only(fits)
        _, omega2 = _brm_apply_hsgp(fits, ([center],), K)
        source = name(only(latent_args))
        rho_lower = _brm_hsgp_declared_rho_lower(
            term, target, context,
            _brm_hsgp_rho_lower_data(fits, K, true),
            _brm_hsgp_floor_geometry(K, c, fits, true),
            _brm_hsgp_floor_remedy(true),
        )
        state = (; target, latent=true, explicit_domain=true, axis_source=source,
                 K, c, centeredness, cov, iso,
                 period=nothing, fits, center, L, omega2,
                 rho_lower,
                 orthogonal, by=nothing,
                 _brm_gp_priors(term, target, context)...)
        return _BRMPreparedTerm(hsgp, (source,), state, (source,))
    end
    cov = _brm_gp_cov(kw, :hsgp)
    options = _brm_hsgp_extended_options(kw, length(getargs(term)), cov)
    (; extrapolate, anchor) = options
    rows = _brm_hsgp_rows(kw, term, context)
    mask = isnothing(rows) ? nothing : rows.mask
    sources, axes = _brm_gp_axes(term, context, :hsgp;
        allow_empty=cov === :periodic || haskey(kw, :domain), mask)
    iso = _brm_gp_iso(kw, :hsgp)
    K, c = _brm_hsgp_options(kw, length(axes))
    period = _brm_gp_period(kw, :hsgp, cov)
    fitted_axes = _brm_hsgp_masked_axes(axes, mask)
    by = get(kw, :by, nothing)
    by_state = if isnothing(by)
        nothing
    else
        source, raw = _brm_term_data(:hsgp, by, context)
        # A frozen replay keeps the training groups (`_brm_fit_source`).
        levels = _brm_fit_levels(_brm_fit_source(source, raw))
        (; source, levels, idx=_brm_apply_levels(levels, raw))
    end
    n_basis = cov === :periodic ? 2 * only(K) : prod(K)
    centeredness = _brm_hsgp_centeredness(kw, n_basis)
    if any(!iszero, centeredness)
        cov === :exp_quad || error(
            "BRM term preparation: partial centering currently supports the " *
            "exp_quad HSGP spectrum")
        isnothing(by_state) || error(
            "BRM term preparation: partial centering is an ungrouped HSGP " *
            "weight geometry and cannot be combined with `by=`")
    end
    if cov === :periodic
        length(axes) == 1 && iso || error(
            "BRM term preparation: periodic hsgp requires one isotropic axis")
        isnothing(by_state) || error(
            "BRM term preparation: periodic hsgp does not support by")
        prepared = basis(axes, K, cov, iso, period; anchor, mask)
        prepared = merge(prepared, (; rho_lower=_brm_hsgp_declared_rho_lower(
            term, target, context, prepared.rho_lower,
            _brm_hsgp_periodic_floor_geometry(K, period),
            _BRM_HSGP_PERIODIC_FLOOR_REMEDY,
        )))
        state = (; target, latent=false, explicit_domain=false, K, c, centeredness,
                 cov, iso, period, prepared..., by=by_state,
                 anchor, extrapolate, support=nothing, rows,
                 _brm_gp_priors(term, target, context)...)
    else
        domain_fits = _brm_hsgp_domain_fits(kw, length(axes))
        # A frozen replay fits the interval to the training axes and refuses
        # replay values past it (unless `extrapolate=:clamp`).
        training_axes = _brm_hsgp_training_axes(sources, axes, rows, fitted_axes)
        fits = isnothing(domain_fits) ?
            _brm_fit_hsgp(training_axes, K, c; anchor) : domain_fits
        training_axes === fitted_axes || extrapolate === :clamp ||
            _brm_hsgp_replay_axes(fitted_axes, fits, nothing, :error;
                prefix="RK reprocess", names=sources,
                fixed_domain=!isnothing(domain_fits))
        isnothing(domain_fits) ||
            _brm_check_hsgp_domain(domain_fits, fitted_axes;
                                   prefix="BRM term preparation")
        _brm_check_hsgp_anchor(domain_fits, anchor; prefix="BRM term preparation")
        orthogonal = _brm_hsgp_orthogonal_to(kw, length(axes))
        prepared = basis(axes, K, cov, iso, period; fits, orthogonal, anchor, mask)
        prepared = merge(prepared, (; rho_lower=_brm_hsgp_declared_rho_lower(
            term, target, context, prepared.rho_lower,
            _brm_hsgp_floor_geometry(K, c, fits, !isnothing(domain_fits)),
            _brm_hsgp_floor_remedy(!isnothing(domain_fits)),
        )))
        state = (; target, latent=false, explicit_domain=!isnothing(domain_fits),
                 K, c, centeredness, cov, iso, period, prepared...,
                 orthogonal, by=by_state, anchor, extrapolate,
                 support=_brm_hsgp_support(training_axes, fits), rows,
                 _brm_gp_priors(term, target, context)...)
    end
    deps = (sources..., (isnothing(by_state) ? () : (by_state.source,))...,
            (isnothing(rows) ? () : (rows.source,))...)
    _BRMPreparedTerm(hsgp, sources, state, deps)
end

function _brm_replay_gp_axes(training, context, label; allow_empty::Bool=false,
                             mask=nothing)
    axes = Tuple(_brm_axis_values(context.data[source], mask,
                     "BRM term replay: `$label` axis `$source`")
                 for source in training.source)
    allow_empty || all(!isempty, axes) ||
        error("BRM term replay: `$label` axes cannot be empty")
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
    st = training.state
    periodic = st.cov === :periodic
    rows = get(st, :rows, nothing)
    mask = isnothing(rows) ? nothing : _brm_hsgp_row_mask(
        context.data[rows.source], length(context.data[first(training.source)]),
        rows.source)
    rows = isnothing(rows) ? nothing : merge(rows, (; mask))
    axes = _brm_replay_gp_axes(training, context, :hsgp;
        allow_empty=periodic || get(st, :explicit_domain, false), mask)
    anchor = get(st, :anchor, nothing)
    state = if periodic
        basis = _brm_hsgp_basis_state(axes, st.K, :periodic, true, st.period;
                                      anchor, mask)
        merge(st, (; axes, PHI=basis.PHI, rows))
    else
        evaluated = _brm_hsgp_replay_axes(axes, st.fits, st.support,
            get(st, :extrapolate, :error), mask;
            prefix="BRM term replay", names=training.source,
            fixed_domain=get(st, :explicit_domain, false))
        basis = _brm_hsgp_basis_state(evaluated, st.K, :exp_quad, st.iso, nothing;
            fits=st.fits, orthogonal=get(st, :orthogonal, nothing), anchor, mask)
        merge(st, (; axes, PHI=basis.PHI, omega2=basis.omega2, rows))
    end
    if !isnothing(st.by)
        by = st.by
        idx = _brm_apply_levels(by.levels, context.data[by.source])
        state = merge(state, (; by=merge(by, (; idx))))
    end
    _BRMPreparedTerm(hsgp, training.source, state, training.dependencies)
end
