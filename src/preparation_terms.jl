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
_brm_prepares_term(term::ExprColumn{typeof(rw)}) = true
_brm_prepares_term(term::ExprColumn{typeof(cdar)}) = true

_brm_term_arg_name(value::NamedColumn) = name(value)
_brm_term_arg_name(_value) = nothing
_brm_prepared_term_key(term::ExprColumn) = Symbol(nameof(getf(term)), "(",
    join((value for value in (_brm_term_arg_name(arg) for arg in getargs(term))
          if !isnothing(value)), ","), ")")

function _brm_term_prior_spec(term, target, context, class; component=nothing)
    brmi = hasproperty(context, :parent) ? context.parent :
           hasproperty(context, :brmi) ? context.brmi : nothing
    isnothing(brmi) && return nothing
    key = _brm_prepared_term_key(term)
    resolved = hasproperty(context, :term_priors) ? context.term_priors :
               _brm_resolve_term_priors(brmi)
    per_term = get(get(resolved, target, Dict()), key, Dict())
    slots = filter(slot -> slot.class === class && slot.component === component,
                   _brm_term_prior_slots(getf(term)))
    isempty(slots) && return nothing
    entry = get(per_term, only(slots).name, nothing)
    isnothing(entry) ? nothing : entry.spec
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

_brm_simplex_alpha_metadata(_constructor, _prior) = nothing
function _brm_simplex_alpha_metadata(::Type{<:Dirichlet}, prior)
    alpha = only(getargs(prior))
    alpha isa AbstractVector{<:Real} ? alpha : nothing
end

function _brm_prepare_monotonic(term, target, context)
    args = getargs(term)
    length(args) == 1 || error("BRM term preparation: monotonic term needs one argument")
    isempty(getkwargs(term)) || error(
        "BRM term preparation: monotonic term does not accept keywords")
    source, raw = _brm_term_data(nameof(getf(term)), only(args), context)
    levels = _brm_fit_levels(raw)
    idx = _brm_apply_levels(levels, raw)
    simplex_prior = _brm_normalize_simplex_prior(_brm_term_prior_expression(
        term, target, context, :term_simplex; default=ExprColumn(Dirichlet, 1.0)),
        length(levels) - 1)
    alpha = _brm_simplex_alpha_metadata(getf(simplex_prior), simplex_prior)
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
_brm_term_prior_expressions(term::_BRMPreparedTerm{typeof(rw)}) =
    (; sd=term.state.sd_prior)
_brm_term_prior_expressions(term::_BRMPreparedTerm{typeof(cdar)}) =
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

# `differenced`: the term integrates its increments (dar, rw) and needs a
# strictly increasing axis; `persistence`: the increments carry a sampled AR(1)
# coefficient (dar) — a random walk (rw) has none, so it owns no `ar` slot.
function _brm_prepare_ar_term(term, target, context, differenced;
                              persistence::Bool=differenced)
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
    ar_prior = persistence ? _brm_term_prior_expression(
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
# `rw(time)`: a zero-started random walk over the SORTED DISTINCT values of `time`;
# each row reads the point of its own time value, so rows may share times (a long
# frame with several groups per day gets one shared walk). Unique increasing times
# reduce to the plain path.
function _brm_prepare_term(term::ExprColumn{typeof(rw)}, target::Symbol, context)
    args, kw = getargs(term), getkwargs(term)
    length(args) == 1 || error("BRM term preparation: `rw(time)` needs one time axis")
    isempty(kw) || error("BRM term preparation: `rw(time)` takes no keyword arguments")
    source, raw = _brm_term_data(:rw, only(args), context)
    time = collect(Float64, raw)
    isempty(time) && error("BRM term preparation: `rw` time axis cannot be empty")
    all(isfinite, time) || error("BRM term preparation: `rw` time axis must be finite")
    steps = _brm_cdar_levels(time)
    time_idx = [searchsortedfirst(steps, v) for v in time]
    sd_prior = _brm_term_prior_expression(term, target, context, :term_sd;
                                          default=ExprColumn(Normal, 0.0, 0.2))
    _BRMPreparedTerm(rw, source, (; target, time, steps, n_steps=length(steps), time_idx, sd_prior),
                     (source,))
end
function _brm_replay_rw(training, raw)
    time = collect(Float64, raw)
    all(isfinite, time) || error("BRM term replay: `rw` time axis must be finite")
    steps = _brm_cdar_levels(vcat(training.state.steps, time))
    time_idx = [searchsortedfirst(steps, v) for v in time]
    _BRMPreparedTerm(rw, training.source,
        merge(training.state, (; time, steps, n_steps=length(steps), time_idx)),
        training.dependencies)
end
function _brm_replay_term(::typeof(rw), training, fresh::ExprColumn, context)
    source, raw = _brm_term_data(:rw, only(getargs(fresh)), context)
    source === training.source || error("BRM term replay: `rw` source changed")
    _brm_replay_rw(training, raw)
end
_brm_replay_term(::typeof(rw), training, fresh::_BRMPreparedTerm, context) =
    _brm_replay_rw(training, context.data[training.source])

# `cdar(step; by=group, cor=C)`: per-group deviations that follow a damped walk
# over the sorted distinct `step` values with innovations correlated across
# groups by the Cholesky factor of `C`. The row contribution is the deviation
# of the row's group at the row's step, so the term needs both index maps.
_brm_cdar_levels(values) = sort!(unique(collect(values)))
function _brm_cdar_levels(values::AbstractVector{<:AbstractString})
    sort!(unique(collect(String, values)))
end
function _brm_cdar_cor(value, context)
    if value isa NamedColumn
        key, raw = _brm_term_data(:cdar, value, context)
        return key, _brm_cdar_cor_matrix(raw)
    end
    nothing, _brm_cdar_cor_matrix(value)
end
_brm_cdar_cor_matrix(C::AbstractMatrix{<:Real}) = Matrix{Float64}(C)
function _brm_cdar_cor_matrix(v::AbstractVector{<:Real})
    P = isqrt(length(v))
    P * P == length(v) || error(
        "BRM term preparation: `cdar(...; cor=...)` got a vector of length $(length(v)), " *
        "which is not a flattened square matrix")
    Matrix{Float64}(reshape(collect(Float64, v), P, P))
end
_brm_cdar_cor_matrix(x) = error(
    "BRM term preparation: `cdar(...; cor=...)` needs a `P × P` numeric matrix " *
    "(a data field or a literal), got $(typeof(x))")
function _brm_cdar_factor(C::AbstractMatrix{<:Real})
    size(C, 1) == size(C, 2) || error(
        "BRM term preparation: `cdar(...; cor=...)` needs a square matrix, got $(size(C))")
    all(isfinite, C) || error("BRM term preparation: `cdar(...; cor=...)` must be finite")
    maximum(abs.(C .- C')) <= 1e-10 * max(1.0, maximum(abs.(C))) || error(
        "BRM term preparation: `cdar(...; cor=...)` must be symmetric")
    F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Matrix{Float64}(C)); check=false)
    LinearAlgebra.issuccess(F) || error(
        "BRM term preparation: `cdar(...; cor=...)` must be positive definite")
    Matrix{Float64}(F.L)
end
function _brm_cdar_indices(step_raw, steps, group_raw, groups; prefix="BRM term preparation")
    step_idx = Vector{Int}(undef, length(step_raw))
    for (i, s) in enumerate(step_raw)
        j = searchsortedfirst(steps, s)
        (j <= length(steps) && steps[j] == s) || error(
            "$prefix: `cdar` step value $s is not on the step grid")
        step_idx[i] = j
    end
    group_idx = Vector{Int}(undef, length(group_raw))
    for (i, g) in enumerate(group_raw)
        j = findfirst(==(g), groups)
        isnothing(j) && error("$prefix: `cdar` group `$g` is not a fitted group level")
        group_idx[i] = j
    end
    step_idx, group_idx
end
function _brm_prepare_term(term::ExprColumn{typeof(cdar)}, target::Symbol, context)
    args, kw = getargs(term), getkwargs(term)
    length(args) == 1 || error(
        "BRM term preparation: `cdar(step; by=group, cor=C)` needs exactly one step axis")
    haskey(kw, :by) || error("BRM term preparation: `cdar(step; by=group, cor=C)` needs `by=`")
    haskey(kw, :cor) || error("BRM term preparation: `cdar(step; by=group, cor=C)` needs `cor=`")
    extra = setdiff(keys(kw), (:by, :cor))
    isempty(extra) || error(
        "BRM term preparation: `cdar` takes only `by=` and `cor=`, got $(collect(extra))")
    source, step_raw = _brm_term_data(:cdar, only(args), context)
    group_source, group_raw = _brm_term_data(:cdar, kw.by, context)
    cor_source, C = _brm_cdar_cor(kw.cor, context)
    all(isfinite, step_raw) || error("BRM term preparation: `cdar` step axis must be finite")
    steps = _brm_cdar_levels(step_raw)
    groups = _brm_cdar_levels(group_raw)
    P, W = length(groups), length(steps)
    size(C) == (P, P) || error(
        "BRM term preparation: `cdar(...; cor=...)` is $(size(C)) but `by` has $P levels")
    L = _brm_cdar_factor(C)
    step_idx, group_idx = _brm_cdar_indices(step_raw, steps, group_raw, groups)
    ar_prior = _brm_term_prior_expression(term, target, context, :term_ar;
                                          default=ExprColumn(Normal, 0.5, 0.2))
    sd_prior = _brm_term_prior_expression(term, target, context, :term_sd;
                                          default=ExprColumn(Normal, 0.0, 0.2))
    sources = isnothing(cor_source) ? (source, group_source) : (source, group_source, cor_source)
    _BRMPreparedTerm(cdar, (source, group_source),
        (; target, steps, groups, n_groups=P, n_steps=W, cor=C, L, step_idx, group_idx,
           ar_prior, sd_prior), sources)
end

function _brm_replay_term(training::_BRMPreparedTerm,
                          fresh::ExprColumn, context::_BRMBackendContext)
    getf(fresh) === training.callable || error(
        "BRM term replay: fitted `$(nameof(training.callable))` term changed callable")
    _brm_replay_term(training.callable, training, fresh, context)
end

function _brm_replay_term(training::_BRMPreparedTerm,
                          fresh::_BRMPreparedTerm, context::_BRMBackendContext)
    training.callable === fresh.callable || error(
        "BRM term replay: fitted term changed callable")
    training.source == fresh.source || error(
        "BRM term replay: fitted term source changed")
    _brm_replay_term(training.callable, training, fresh, context)
end

function _brm_replay_term(::typeof(s), training, fresh::ExprColumn, context)
    source, raw = _brm_term_data(:s, only(getargs(fresh)), context)
    source === training.source || error("BRM term replay: `s` source changed")
    Xnull, Zpen = _brm_apply_spline(training.state.fit, raw)
    _BRMPreparedTerm(s, source,
        merge(training.state, (; Xnull, Zpen)), training.dependencies)
end

function _brm_replay_term(::typeof(s), training, fresh::_BRMPreparedTerm, context)
    Xnull, Zpen = _brm_apply_spline(
        training.state.fit, context.data[training.source])
    _BRMPreparedTerm(s, training.source,
        merge(training.state, (; Xnull, Zpen)), training.dependencies)
end

function _brm_replay_term(::typeof(t2), training, fresh::ExprColumn, context)
    args = getargs(fresh)
    length(args) == 2 || error("BRM term replay: `t2` needs two margins")
    sx, x = _brm_term_data(:t2, args[1], context)
    sz, z = _brm_term_data(:t2, args[2], context)
    (sx, sz) == training.source || error("BRM term replay: `t2` sources changed")
    Xfixed, Zrr, Zrn, Znr = _brm_apply_t2(training.state.fit, x, z)
    state = merge(training.state, (; Xfixed, Zrr, Zrn, Znr))
    _BRMPreparedTerm(t2, training.source, state, training.dependencies)
end


function _brm_replay_term(::typeof(t2), training, fresh::_BRMPreparedTerm, context)
    sx, sz = training.source
    Xfixed, Zrr, Zrn, Znr = _brm_apply_t2(
        training.state.fit, context.data[sx], context.data[sz])
    state = merge(training.state, (; Xfixed, Zrr, Zrn, Znr))
    _BRMPreparedTerm(t2, training.source, state, training.dependencies)
end

function _brm_replay_term(::typeof(me), training, fresh::ExprColumn, context)
    args = getargs(fresh)
    length(args) == 2 || error("BRM term replay: `me` needs two arguments")
    source, raw = _brm_term_data(:me, args[1], context)
    source === training.source || error("BRM term replay: `me` source changed")
    args[2] == training.state.sd_x || error(
        "BRM term replay: `me` measurement sd changed from its fitted value")
    state = merge(training.state, (; x_obs=collect(Float64, raw)))
    _BRMPreparedTerm(me, source, state, training.dependencies)
end


function _brm_replay_term(::typeof(me), training, fresh::_BRMPreparedTerm, context)
    state = merge(training.state,
        (; x_obs=collect(Float64, context.data[training.source])))
    _BRMPreparedTerm(me, training.source, state, training.dependencies)
end

function _brm_replay_monotonic(callable, training, fresh::ExprColumn, context)
    source, raw = _brm_term_data(nameof(callable), only(getargs(fresh)), context)
    source === training.source || error("BRM term replay: monotonic source changed")
    idx = _brm_apply_levels(training.state.levels, raw)
    _BRMPreparedTerm(callable, source,
        merge(training.state, (; idx)), training.dependencies)
end
_brm_replay_term(callable::Union{typeof(mo),typeof(mo1)}, training,
                 fresh::ExprColumn, context) =
    _brm_replay_monotonic(callable, training, fresh, context)
function _brm_replay_term(callable::Union{typeof(mo),typeof(mo1)}, training,
                          fresh::_BRMPreparedTerm, context)
    idx = _brm_apply_levels(training.state.levels,
                            context.data[training.source])
    _BRMPreparedTerm(callable, training.source,
        merge(training.state, (; idx)), training.dependencies)
end

function _brm_replay_ar(callable, training, fresh::ExprColumn, context)
    source, raw = _brm_term_data(nameof(callable), only(getargs(fresh)), context)
    source === training.source || error("BRM term replay: AR source changed")
    time = _brm_replay_ar_time(raw, callable !== ar)
    _BRMPreparedTerm(callable, source,
        merge(training.state, (; time)), training.dependencies)
end
_brm_replay_term(callable::Union{typeof(ar),typeof(dar)}, training,
                 fresh::ExprColumn, context) =
    _brm_replay_ar(callable, training, fresh, context)
function _brm_replay_term(callable::Union{typeof(ar),typeof(dar)}, training,
                          fresh::_BRMPreparedTerm, context)
    time = _brm_replay_ar_time(context.data[training.source], callable !== ar)
    _BRMPreparedTerm(callable, training.source,
        merge(training.state, (; time)), training.dependencies)
end

# cdar replay: group levels and the correlation factor are frozen from the fit;
# the step grid may grow (new steps append), which is how a forecast extends it.
function _brm_replay_cdar(training, step_raw, group_raw)
    all(isfinite, step_raw) || error("BRM term replay: `cdar` step axis must be finite")
    steps = _brm_cdar_levels(vcat(training.state.steps, collect(step_raw)))
    step_idx, group_idx = _brm_cdar_indices(step_raw, steps, group_raw,
                                            training.state.groups; prefix="BRM term replay")
    _BRMPreparedTerm(cdar, training.source,
        merge(training.state, (; steps, n_steps=length(steps), step_idx, group_idx)),
        training.dependencies)
end
function _brm_replay_term(::typeof(cdar), training, fresh::ExprColumn, context)
    args, kw = getargs(fresh), getkwargs(fresh)
    source, step_raw = _brm_term_data(:cdar, only(args), context)
    group_source, group_raw = _brm_term_data(:cdar, kw.by, context)
    (source, group_source) == training.source || error(
        "BRM term replay: `cdar` step or group source changed")
    _brm_replay_cdar(training, step_raw, group_raw)
end
function _brm_replay_term(::typeof(cdar), training, fresh::_BRMPreparedTerm, context)
    source, group_source = training.source
    _brm_replay_cdar(training, context.data[source], context.data[group_source])
end

function _brm_replay_term(::typeof(interval_censored), training,
                          fresh::ExprColumn, context)
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


function _brm_replay_term(::typeof(interval_censored), training,
                          fresh::_BRMPreparedTerm, context)
    sx, su = training.source
    state = merge(training.state, _brm_interval_predictor_plan(
        context.data[sx], context.data[su], training.state.lower))
    _BRMPreparedTerm(interval_censored, training.source, state,
                     training.dependencies)
end

function _brm_replay_term(callable, training, fresh, context)
    error("BRM term replay: unsupported prepared callable `$(nameof(callable))`")
end
