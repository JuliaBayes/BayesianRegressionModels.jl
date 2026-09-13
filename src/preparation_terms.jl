# Backend-neutral fitted preparation for formula terms that own latent state.

struct _BRMPreparedTerm{F,S,ST,D}
    callable::F
    source::S
    state::ST
    dependencies::D
end

function _brm_turing_term_model end

_brm_prepare_term(_term, _target, _context) = nothing
_brm_prepares_term(_term) = false
_brm_prepares_term(term::ExprColumn{typeof(s)}) = true
_brm_prepares_term(term::ExprColumn{typeof(t2)}) = true
_brm_prepares_term(term::ExprColumn{typeof(me)}) = true
_brm_prepares_term(term::ExprColumn{typeof(mo)}) = true
_brm_prepares_term(term::ExprColumn{typeof(mo1)}) = true
_brm_prepares_term(term::ExprColumn{typeof(interval_censored)}) = true
_brm_prepares_term(term::ExprColumn{typeof(ar)}) = true
_brm_prepares_term(term::ExprColumn{typeof(dar)}) = true

_brm_term_arg_name(value::NamedColumn) = name(value)
_brm_term_arg_name(_value) = nothing
_brm_prepared_term_key(term::ExprColumn) = Symbol(nameof(getf(term)), "(",
    join((value for value in (_brm_term_arg_name(arg) for arg in getargs(term))
          if !isnothing(value)), ","), ")")

_brm_term_prior_rank(spec) = isnothing(spec.predictor) ? 0 : 1
_brm_term_prior_matches(spec, key, target, class, component) =
    spec.class === class && spec.term === key && spec.component === component &&
    (isnothing(spec.predictor) || spec.predictor === target)
_brm_term_prior_targets(spec, targets, has_term) = isnothing(spec.predictor) ?
    Symbol[target for target in targets if has_term(target, spec.term)] :
    (has_term(spec.predictor, spec.term) ? Symbol[spec.predictor] : Symbol[])

function _brm_select_term_prior(specs, key, target, class, component=nothing)
    candidates = filter(spec ->
        _brm_term_prior_matches(spec, key, target, class, component), specs)
    isempty(candidates) && return nothing
    rank = maximum(_brm_term_prior_rank, candidates)
    winners = filter(spec -> _brm_term_prior_rank(spec) == rank, candidates)
    length(winners) == 1 || error(
        "BRM term preparation: duplicate prior for `$key` in `$target`")
    only(winners)
end

function _brm_term_prior_spec(term, target, context, class; component=nothing)
    brmi = hasproperty(context, :parent) ? context.parent :
           hasproperty(context, :brmi) ? context.brmi : nothing
    isnothing(brmi) && return nothing
    key = _brm_prepared_term_key(term)
    _brm_select_term_prior(term_priors(brmi), key, target, class, component)
end

_brm_term_prior_expression(term, target, context, class; component=nothing,
                           default=ExprColumn(Normal, 0.0, 1.0)) =
    something(_brm_term_prior_spec(term, target, context, class; component),
              (; expression=default)).expression

function _brm_term_data(term_name::Symbol, value, context)
    value isa NamedColumn || error(
        "BRM term preparation: `$term_name` expects a data-column argument")
    key = name(value)
    haskey(context.data, key) || error(
        "BRM term preparation: `$term_name` data column `$key` is unavailable")
    key, context.data[key]
end

function _brm_prepare_term(term::ExprColumn{typeof(s)}, target::Symbol,
                           context)
    args = getargs(term)
    length(args) == 1 || error(
        "BRM term preparation: `s(x)` expects one positional argument")
    isempty(getkwargs(term)) || error(
        "BRM term preparation: `s(x)` does not support keyword arguments")
    source, raw = _brm_term_data(:s, only(args), context)
    raw isa AbstractVector{<:Real} || error(
        "BRM term preparation: `s($source)` requires numeric data")
    fit = _brm_fit_spline(raw)
    Xnull, Zpen = _brm_apply_spline(fit, raw)
    sd_prior = _brm_term_prior_expression(term, target, context, :term_sd)
    _BRMPreparedTerm(s, source, (; target, fit, Xnull, Zpen, sd_prior), (source,))
end

function _brm_prepare_term(term::ExprColumn{typeof(t2)}, target::Symbol,
                           context)
    args = getargs(term)
    length(args) == 2 || error(
        "BRM term preparation: `t2(x, z)` expects two positional margins")
    k, _, _ = _brm_t2_options(getkwargs(term))
    first_source, x = _brm_term_data(:t2, args[1], context)
    second_source, z = _brm_term_data(:t2, args[2], context)
    (x isa AbstractVector{<:Real} && z isa AbstractVector{<:Real}) || error(
        "BRM term preparation: `t2` margins must be numeric data columns")
    fit = _brm_fit_t2(x, z; k)
    Xfixed, Zrr, Zrn, Znr = _brm_apply_t2(fit, x, z)
    sources = (first_source, second_source)
    sd_priors = ntuple(3) do index
        _brm_term_prior_expression(term, target, context, :term_sd;
            component=(:rr, :rn, :nr)[index])
    end
    state = (; target, fit, Xfixed, Zrr, Zrn, Znr, sd_priors)
    _BRMPreparedTerm(t2, sources, state, sources)
end

function _brm_prepare_term(term::ExprColumn{typeof(me)}, target::Symbol,
                           context)
    args = getargs(term)
    length(args) == 2 || error(
        "BRM term preparation: `me(x, sd)` expects two positional arguments")
    isempty(getkwargs(term)) || error(
        "BRM term preparation: `me(x, sd)` does not accept keywords")
    source, raw = _brm_term_data(:me, args[1], context)
    raw isa AbstractVector{<:Real} || error(
        "BRM term preparation: `me($source, sd)` requires numeric observations")
    sd_x = args[2]
    (sd_x isa Real && !(sd_x isa Bool) && isfinite(sd_x) && sd_x > 0) || error(
        "BRM term preparation: `me($source, sd)` requires finite numeric sd > 0")
    latent_prior = _brm_term_prior_expression(
        term, target, context, :term_latent)
    state = (; target, x_obs=collect(Float64, raw), sd_x=Float64(sd_x),
             latent_prior)
    _BRMPreparedTerm(me, source, state, (source,))
end

function _brm_simplex_alpha(term, target, context, n_levels)
    prior = _brm_term_prior_expression(term, target, context, :term_simplex;
        default=ExprColumn(Dirichlet, 1.0))
    args = getargs(prior)
    length(args) == 1 || error(
        "BRM term preparation: monotonic simplex prior needs one concentration argument")
    raw = only(args)
    alpha = raw isa Real ? fill(Float64(raw), n_levels - 1) : collect(Float64, raw)
    length(alpha) == n_levels - 1 || error(
        "BRM term preparation: monotonic simplex prior has wrong dimension")
    all(x -> isfinite(x) && x > 0, alpha) || error(
        "BRM term preparation: monotonic simplex concentrations must be positive")
    alpha
end

function _brm_prepare_monotonic(term, target, context)
    args = getargs(term)
    length(args) == 1 || error("BRM term preparation: monotonic term needs one argument")
    isempty(getkwargs(term)) || error(
        "BRM term preparation: monotonic term does not accept keywords")
    source, raw = _brm_term_data(nameof(getf(term)), only(args), context)
    levels = _brm_fit_levels(raw)
    idx = _brm_apply_levels(levels, raw)
    simplex_prior = _brm_term_prior_expression(term, target, context,
        :term_simplex; default=ExprColumn(Dirichlet, 1.0))
    alpha = _brm_simplex_alpha(term, target, context, length(levels))
    _BRMPreparedTerm(getf(term), source,
        (; target, levels, idx, alpha, simplex_prior), (source,))
end


_brm_term_prior_expressions(term::_BRMPreparedTerm{typeof(s)}) =
    (; sd=term.state.sd_prior)
_brm_term_prior_expressions(term::_BRMPreparedTerm{typeof(t2)}) =
    (; sd=term.state.sd_priors)
_brm_term_prior_expressions(term::_BRMPreparedTerm{typeof(me)}) =
    (; latent=term.state.latent_prior)
_brm_term_prior_expressions(term::_BRMPreparedTerm{typeof(mo)}) =
    (; simplex=term.state.simplex_prior)
_brm_term_prior_expressions(term::_BRMPreparedTerm{typeof(mo1)}) =
    (; simplex=term.state.simplex_prior)
_brm_term_prior_expressions(
    term::_BRMPreparedTerm{typeof(interval_censored)}) =
    (; latent=term.state.latent_prior)
_brm_term_prior_expressions(term::_BRMPreparedTerm{typeof(ar)}) = NamedTuple()
_brm_term_prior_expressions(term::_BRMPreparedTerm{typeof(dar)}) =
    (; ar=term.state.ar_prior, sd=term.state.sd_prior)
_brm_prepare_term(term::ExprColumn{typeof(mo)}, target::Symbol, context) =
    _brm_prepare_monotonic(term, target, context)
_brm_prepare_term(term::ExprColumn{typeof(mo1)}, target::Symbol, context) =
    _brm_prepare_monotonic(term, target, context)

function _brm_interval_predictor_plan(x_raw, upper_raw, lower::Real)
    x = collect(Float64, x_raw)
    upper = collect(Float64, upper_raw)
    length(x) == length(upper) || error(
        "BRM term preparation: interval predictor bounds have unequal lengths")
    all(isfinite, x) && all(isfinite, upper) && isfinite(lower) || error(
        "BRM term preparation: interval predictor data must be finite")
    any(x .< upper) && error(
        "BRM term preparation: quantified interval predictor values must not be below upper bounds")
    Jinterval = findall(x .== upper)
    isempty(Jinterval) && error(
        "BRM term preparation: interval predictor has no interval-censored rows")
    all(lower .< upper[Jinterval]) || error(
        "BRM term preparation: interval predictor lower bounds must be below upper bounds")
    Jexact = findall(x .> upper)
    (; x_exact=x[Jexact], x_lower=fill(Float64(lower), length(Jinterval)),
     x_upper=upper[Jinterval], Jexact=collect(Int, Jexact),
     Jinterval=collect(Int, Jinterval), nobs=length(x))
end

function _brm_prepare_term(term::ExprColumn{typeof(interval_censored)},
                           target::Symbol, context)
    args, kw = getargs(term), getkwargs(term)
    length(args) == 1 && haskey(kw, :upper) || error(
        "BRM term preparation: interval_censored predictor needs x and upper")
    all(key -> key in (:lower, :upper), keys(kw)) || error(
        "BRM term preparation: unsupported interval_censored predictor keyword")
    source, x = _brm_term_data(:interval_censored, only(args), context)
    upper_source, upper = _brm_term_data(:interval_censored, kw.upper, context)
    lower = get(kw, :lower, 0.0)
    lower isa Real || error("BRM term preparation: interval lower bound must be numeric")
    plan = _brm_interval_predictor_plan(x, upper, lower)
    latent_prior = _brm_term_prior_expression(
        term, target, context, :term_latent)
    _BRMPreparedTerm(interval_censored, (source, upper_source),
        (; target, plan..., lower=Float64(lower), latent_prior),
        (source, upper_source))
end

function _brm_prepare_ar_term(term, target, context, differenced)
    args, kw = getargs(term), getkwargs(term)
    length(args) == 1 || error("BRM term preparation: AR term needs one time axis")
    all(key -> key === :p, keys(kw)) && get(kw, :p, 1) == 1 || error(
        "BRM term preparation: AR terms currently require p=1")
    source, raw = _brm_term_data(nameof(getf(term)), only(args), context)
    time = collect(Float64, raw)
    isempty(time) && error("BRM term preparation: AR time axis cannot be empty")
    all(isfinite, time) || error("BRM term preparation: AR time axis must be finite")
    differenced && !all(>(0), diff(time)) && error(
        "BRM term preparation: dar time axis must be strictly increasing")
    ar_prior = differenced ? _brm_term_prior_expression(
        term, target, context, :term_ar;
        default=ExprColumn(Normal, 0.5, 0.2)) : nothing
    sd_prior = differenced ? _brm_term_prior_expression(
        term, target, context, :term_sd;
        default=ExprColumn(Normal, 0.0, 0.2)) : nothing
    _BRMPreparedTerm(getf(term), source,
        (; target, time, ar_prior, sd_prior), (source,))
end

function _brm_replay_ar_time(raw, differenced)
    time = collect(Float64, raw)
    isempty(time) && error("BRM term replay: AR time axis cannot be empty")
    all(isfinite, time) || error("BRM term replay: AR time axis must be finite")
    differenced && !all(>(0), diff(time)) && error(
        "BRM term replay: dar time axis must be strictly increasing")
    time
end
_brm_prepare_term(term::ExprColumn{typeof(ar)}, target::Symbol, context) =
    _brm_prepare_ar_term(term, target, context, false)
_brm_prepare_term(term::ExprColumn{typeof(dar)}, target::Symbol, context) =
    _brm_prepare_ar_term(term, target, context, true)

function _brm_replay_term(training::_BRMPreparedTerm,
                          fresh::ExprColumn, context::_BRMBackendContext)
    getf(fresh) === training.callable || error(
        "BRM term replay: fitted `$(nameof(training.callable))` term changed callable")
    _brm_replay_term(Val(nameof(training.callable)), training, fresh, context)
end

function _brm_replay_term(training::_BRMPreparedTerm,
                          fresh::_BRMPreparedTerm, context::_BRMBackendContext)
    training.callable === fresh.callable || error(
        "BRM term replay: fitted term changed callable")
    training.source == fresh.source || error(
        "BRM term replay: fitted term source changed")
    if training.callable === s
        Xnull, Zpen = _brm_apply_spline(
            training.state.fit, context.data[training.source])
        state = merge(training.state, (; Xnull, Zpen))
    elseif training.callable === t2
        sx, sz = training.source
        Xfixed, Zrr, Zrn, Znr = _brm_apply_t2(
            training.state.fit, context.data[sx], context.data[sz])
        state = merge(training.state, (; Xfixed, Zrr, Zrn, Znr))
    elseif training.callable === me
        state = merge(training.state,
            (; x_obs=collect(Float64, context.data[training.source])))
    elseif training.callable === mo || training.callable === mo1
        state = merge(training.state,
            (; idx=_brm_apply_levels(training.state.levels,
                                     context.data[training.source])))
    elseif training.callable === interval_censored
        sx, su = training.source
        state = merge(training.state, _brm_interval_predictor_plan(
            context.data[sx], context.data[su], training.state.lower))
    elseif training.callable === ar || training.callable === dar
        state = merge(training.state,
            (; time=_brm_replay_ar_time(context.data[training.source],
                                        training.callable === dar)))
    else
        error("BRM term replay: unsupported prepared callable")
    end
    _BRMPreparedTerm(training.callable, training.source, state,
                     training.dependencies)
end

function _brm_replay_term(::Val{:s}, training, fresh, context)
    source, raw = _brm_term_data(:s, only(getargs(fresh)), context)
    source === training.source || error("BRM term replay: `s` source changed")
    Xnull, Zpen = _brm_apply_spline(training.state.fit, raw)
    _BRMPreparedTerm(s, source,
        merge(training.state, (; Xnull, Zpen)), training.dependencies)
end

function _brm_replay_term(::Val{:t2}, training, fresh, context)
    args = getargs(fresh)
    length(args) == 2 || error("BRM term replay: `t2` needs two margins")
    sx, x = _brm_term_data(:t2, args[1], context)
    sz, z = _brm_term_data(:t2, args[2], context)
    (sx, sz) == training.source || error("BRM term replay: `t2` sources changed")
    Xfixed, Zrr, Zrn, Znr = _brm_apply_t2(training.state.fit, x, z)
    state = merge(training.state, (; Xfixed, Zrr, Zrn, Znr))
    _BRMPreparedTerm(t2, training.source, state, training.dependencies)
end

function _brm_replay_term(::Val{:me}, training, fresh, context)
    args = getargs(fresh)
    length(args) == 2 || error("BRM term replay: `me` needs two arguments")
    source, raw = _brm_term_data(:me, args[1], context)
    source === training.source || error("BRM term replay: `me` source changed")
    args[2] == training.state.sd_x || error(
        "BRM term replay: `me` measurement sd changed from its fitted value")
    state = merge(training.state, (; x_obs=collect(Float64, raw)))
    _BRMPreparedTerm(me, source, state, training.dependencies)
end

function _brm_replay_term(::Val{F}, training, fresh, context) where {F}
    if F in (:ar, :dar)
        source, raw = _brm_term_data(F, only(getargs(fresh)), context)
        source === training.source || error("BRM term replay: AR source changed")
        time = _brm_replay_ar_time(raw, F === :dar)
        return _BRMPreparedTerm(training.callable, source,
            merge(training.state, (; time)), training.dependencies)
    end
    F in (:mo, :mo1) || error("BRM term replay: unsupported fitted term `$F`")
    source, raw = _brm_term_data(F, only(getargs(fresh)), context)
    source === training.source || error("BRM term replay: monotonic source changed")
    idx = _brm_apply_levels(training.state.levels, raw)
    _BRMPreparedTerm(training.callable, source,
        merge(training.state, (; idx)), training.dependencies)
end

function _brm_replay_term(::Val{:interval_censored}, training, fresh, context)
    args, kw = getargs(fresh), getkwargs(fresh)
    source, x = _brm_term_data(:interval_censored, only(args), context)
    upper_source, upper = _brm_term_data(:interval_censored, kw.upper, context)
    (source, upper_source) == training.source || error(
        "BRM term replay: interval predictor sources changed")
    lower = Float64(get(kw, :lower, 0.0))
    lower == training.state.lower || error(
        "BRM term replay: interval predictor lower bound changed")
    plan = _brm_interval_predictor_plan(x, upper, lower)
    _BRMPreparedTerm(interval_censored, training.source,
        merge(training.state, plan), training.dependencies)
end
