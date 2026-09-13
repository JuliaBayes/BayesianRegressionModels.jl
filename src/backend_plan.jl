# Shared BRMI analysis used before a concrete backend emits or executes a model.
# Nothing in this file may depend on StanBlocks (or on any other backend): package
# extensions must be able to consume this layer after all backends become weak
# dependencies.

struct _BRMGroupDeclaration{E,EF,G,D}
    predictor::Symbol
    expression::E
    effects::EF
    raw_group::G
    explicit_id::Union{Nothing,Symbol}
    id::Union{Nothing,Symbol}
    uncorrelated::Bool
    descriptor::D
end

"""
    _BRMBackendContext

Backend-neutral facts collected from a data-bound [`BRMI`](@ref): raw data,
likelihood-decorator prepass state, and the observation row axis associated with
each referenced target. It also caches grouped-term source declarations in
formula order. Concrete backends own all later representation choices.
"""
struct _BRMBackendContext{P<:BRMI,D<:AbstractDict,PP<:AbstractDict,TO<:AbstractDict,TP,GD}
    parent::P
    data::D
    prepass::PP
    target_obs::TO
    term_priors::TP
    group_declarations::GD
end

# Backend-neutral construction replay. Concrete backends consume the same
# rebound BRMI rather than each maintaining a tree walker over formula values.
function reprocess end

_brm_df_has_column(df, key::Symbol) = hasproperty(df, key)
function _brm_df_column(df, key::Symbol)
    column = getproperty(Data(df), key)
    data = parent(column)
    data isa DataColumn || error(
        "BRM replay: new data has no column `$key`")
    parent(data)
end

_brm_rebind_value(x, _df) = x
_brm_rebind_value(x::Tuple, df) =
    map(value -> _brm_rebind_value(value, df), x)
_brm_rebind_value(x::NamedTuple, df) = NamedTuple{keys(x)}(
    map(value -> _brm_rebind_value(value, df), values(x)))
_brm_rebind_value(x::NestedPredictorFormula, df) =
    NestedPredictorFormula(_brm_rebind_value(parent(x), df))
_brm_rebind_value(x::LikelihoodColumn, df) = LikelihoodColumn(
    _brm_rebind_value(parent(x), df), _brm_rebind_value(rhs(x), df))
_brm_rebind_value(x::JointResponseColumn, df) = JointResponseColumn(
    map(value -> _brm_rebind_value(value, df), joint_response_columns(x)))
function _brm_rebind_value(x::NamedColumn, df)
    column_name = name(x)
    payload = parent(x)
    if payload isa DataColumn || payload isa MissingColumn
        if _brm_df_has_column(df, column_name)
            return NamedColumn(
                column_name, DataColumn(_brm_df_column(df, column_name)))
        end
        payload isa DataColumn && error(
            "BRM replay: new data has no column `$column_name`, which was " *
            "data-backed in the fitted BRMI")
        return NamedColumn(column_name, MissingColumn())
    end
    NamedColumn(column_name, _brm_rebind_value(payload, df))
end
function _brm_rebind_value(x::ExprColumn, df)
    args = map(value -> _brm_rebind_value(value, df), getargs(x))
    kwargs = getkwargs(x)
    rebound_kwargs = NamedTuple{keys(kwargs)}(
        map(value -> _brm_rebind_value(value, df), values(kwargs)))
    ExprColumn(getf(x), args...; rebound_kwargs...)
end
function _brm_rebind_value(x::MultiMembershipTerm, df)
    groups = map(
        value -> _brm_rebind_value(value, df), getfield(x, :groups))
    old_weights = getfield(x, :weights)
    weights = isnothing(old_weights) ? nothing : map(
        value -> _brm_rebind_value(value, df), old_weights)
    MultiMembershipTerm(
        groups...; weights, normalize=getfield(x, :normalize))
end
function _brm_rebind_brmi(brmi::BRMI, df)
    operations = brmi.operations
    BRMI(NamedTuple{keys(operations)}(
        map(value -> _brm_rebind_value(value, df), values(operations))))
end

# Materialise a user data vector without silently deleting missing rows. A
# likelihood decorator may deliberately keep its response out of this generic
# data dict and model the missing values itself; that decision is made by the
# prepass below.
function _brm_data_vec(col_name::Symbol, raw)
    if eltype(raw) >: Missing
        any(ismissing, raw) && error(
            "BRM backend lowering: data column `$col_name` contains `missing` " *
            "values. BRM never silently drops rows. Either drop/impute them " *
            "before building the model, or use a response decorator that " *
            "models missing observations explicitly.")
        return collect(nonmissingtype(eltype(raw)), raw)
    end
    raw
end

# Generic likelihood-LHS prepass. Each decorator claims its own bucket in the
# context; no concrete backend or walker needs to know the decorator list.
_brm_visit_op!(_, _) = nothing
_brm_visit_op!(ctx, op::NamedColumn) = _brm_visit_op!(ctx, parent(op))
_brm_visit_op!(ctx, op::ExprColumn{typeof(~)}) =
    _brm_register_sampling_lhs!(ctx, first(getargs(op, 2)))

_brm_register_sampling_lhs!(_, _) = nothing
_brm_register_sampling_lhs!(ctx::AbstractDict,
                            lhs::ExprColumn{typeof(mi)}) =
    _brm_register_skipped_inner!(ctx, only(getargs(lhs)))
_brm_register_sampling_lhs!(ctx::AbstractDict,
                            lhs::ExprColumn{typeof(ragged)}) =
    _brm_register_skipped_inner!(ctx, first(getargs(lhs)))

_brm_register_skipped_inner!(_, _) = nothing
_brm_register_skipped_inner!(ctx::AbstractDict, inner::NamedColumn) =
    (push!(get!(ctx, :skip_data, Set{Symbol}()), name(inner)); nothing)

# Multi-membership-only sources are Julia-side preprocessing inputs, not raw
# backend data. A source also used elsewhere remains ordinary model data.
_brm_collect_mm_sources!(_, _) = nothing
_brm_collect_mm_sources!(out, x::NamedColumn) =
    _brm_collect_mm_sources!(out, parent(x))
function _brm_collect_mm_sources!(out, x::ExprColumn)
    foreach(a -> _brm_collect_mm_sources!(out, a), getargs(x))
    foreach(v -> _brm_collect_mm_sources!(out, v), values(getkwargs(x)))
end
function _brm_collect_mm_sources!(out, x::MultiMembershipTerm)
    foreach(g -> push!(out, name(g)), getargs(x))
    weights = getfield(x, :weights)
    isnothing(weights) || foreach(w -> push!(out, name(w)), weights)
end
function _brm_collect_mm_sources!(out, x::JointResponseColumn)
    foreach(a -> _brm_collect_mm_sources!(out, a), joint_response_columns(x))
end

_brm_collect_non_mm_sources!(_, _) = nothing
function _brm_collect_non_mm_sources!(out, x::NamedColumn)
    parent(x) isa DataColumn && push!(out, name(x))
    nothing
end
function _brm_collect_non_mm_sources!(out, x::ExprColumn)
    foreach(a -> _brm_collect_non_mm_sources!(out, a), getargs(x))
    foreach(v -> _brm_collect_non_mm_sources!(out, v), values(getkwargs(x)))
end
_brm_collect_non_mm_sources!(_, ::MultiMembershipTerm) = nothing
function _brm_collect_non_mm_sources!(out, x::JointResponseColumn)
    foreach(a -> _brm_collect_non_mm_sources!(out, a), joint_response_columns(x))
end

_brm_collect_data!(_data, _x; skip=Set{Symbol}()) = nothing
function _brm_collect_data!(data, x::NamedColumn; skip=Set{Symbol}())
    backing = parent(x)
    if backing isa DataColumn && !(name(x) in skip)
        data[name(x)] = _brm_data_vec(name(x), parent(backing))
    end
    _brm_collect_data!(data, backing; skip)
end
function _brm_collect_data!(data, x::ExprColumn; skip=Set{Symbol}())
    foreach(a -> _brm_collect_data!(data, a; skip), getargs(x))
    foreach(v -> _brm_collect_data!(data, v; skip), values(getkwargs(x)))
end
function _brm_collect_data!(data, x::MultiMembershipTerm; skip=Set{Symbol}())
    foreach(a -> _brm_collect_data!(data, a; skip), getargs(x))
    weights = getfield(x, :weights)
    isnothing(weights) || foreach(a -> _brm_collect_data!(data, a; skip), weights)
end
function _brm_joint_response_values(x::JointResponseColumn;
                                    prefix="BRM backend lowering")
    columns = joint_response_columns(x)
    names = joint_response_names(x)
    all(c -> parent(c) isa DataColumn, columns) || error(
        "$prefix: joint response $(collect(names)) requires every outcome to " *
        "be a data-backed column")
    raw = map(columns) do column
        values = parent(parent(column))
        values isa AbstractVector || error(
            "$prefix: joint outcome `$(name(column))` must be a vector, got " *
            "$(typeof(values))")
        value_type = nonmissingtype(eltype(values))
        value_type <: Real || error(
            "$prefix: joint outcome `$(name(column))` must be real-valued, got " *
            "element type $(eltype(values))")
        any(ismissing, values) && error(
            "$prefix: joint outcome `$(name(column))` contains `missing`. " *
            "`MvNormalCholesky` uses an aligned complete-row likelihood and " *
            "never silently drops or factorizes missing outcome patterns. " *
            "Supply complete aligned rows or model the outcomes separately.")
        collected = collect(Float64, values)
        all(isfinite, collected) || error(
            "$prefix: joint outcome `$(name(column))` contains non-finite values")
        collected
    end
    lengths = map(length, raw)
    all(==(first(lengths)), lengths) || error(
        "$prefix: joint outcomes $(collect(names)) must be row-aligned and have " *
        "equal lengths, got $(collect(lengths))")
    first(lengths) > 0 || error(
        "$prefix: joint outcomes $(collect(names)) must contain at least one row")
    # StanBlocks' top-level ragged-observation contract is the representation
    # that preserves one JOINT density contribution per aligned row.  All cells
    # have the same width here, but keeping the outer row collection explicit
    # prevents a dense matrix from being interpreted as elementwise observations.
    [Float64[values[row] for values in raw] for row in eachindex(first(raw))]
end

# Pack one ordered response vector per aligned row before concrete emission.
# The target-to-observation map sizes intercept-only mean predictors from the
# explicit scalar row count, while StanBlocks sees a top-level RaggedVector and
# therefore emits one joint density / predictive vector per row.
function _brm_collect_data!(data, x::JointResponseColumn; skip=Set{Symbol}())
    values = _brm_joint_response_values(x)
    data[_joint_response_data_key(x)] = values
    data[_joint_response_n_key(x)] = length(values)
    nothing
end

_brm_is_nothing_column(x::NamedColumn) =
    name(x) === :nothing && parent(x) isa MissingColumn

_brm_observation_name(_) = nothing
_brm_observation_name(lhs::NamedColumn) =
    parent(lhs) isa DataColumn ? name(lhs) : nothing
_brm_observation_name(lhs::ExprColumn{typeof(ragged)}) =
    _brm_observation_name(first(getargs(lhs)))
_brm_observation_name(lhs::JointResponseColumn) = _joint_response_data_key(lhs)
function _brm_observation_name(lhs::ExprColumn)
    args = getargs(lhs)
    length(args) == 1 ? _brm_observation_name(only(args)) : nothing
end

_brm_collect_rhs_refs!(_target_obs, _x, _obs_name) = nothing
function _brm_collect_rhs_refs!(target_obs, x::NamedColumn, obs_name)
    !_brm_is_nothing_column(x) && get!(target_obs, name(x), obs_name)
    nothing
end
function _brm_collect_rhs_refs!(target_obs, x::ExprColumn, obs_name)
    foreach(a -> _brm_collect_rhs_refs!(target_obs, a, obs_name), getargs(x))
    foreach(v -> _brm_collect_rhs_refs!(target_obs, v, obs_name),
            values(getkwargs(x)))
end
function _brm_collect_rhs_refs!(target_obs, x::Union{Tuple,AbstractVector}, obs_name)
    foreach(a -> _brm_collect_rhs_refs!(target_obs, a, obs_name), x)
end

function _brm_collect_target_obs(brmi::BRMI)
    target_obs = Dict{Symbol,Symbol}()
    for op_nc in values(brmi.operations)
        op_nc isa NamedColumn || continue
        op = parent(op_nc)
        op isa ExprColumn || continue
        getf(op) === (~) || continue
        lhs, rhs = getargs(op, 2)
        obs_name = lhs isa JointResponseColumn ?
            _joint_response_n_key(lhs) : _brm_observation_name(lhs)
        isnothing(obs_name) && continue
        get!(target_obs, obs_name, obs_name)
        _brm_collect_rhs_refs!(target_obs, rhs, obs_name)
    end
    target_obs
end

# ---- shared response composition -----------------------------------------

"""
    _BRMResponseModifierPlan

Backend-neutral syntax and materialized bounds for one response wrapper. The
`base` remains a formula distribution expression; concrete backends decide how
to represent its density. Bounds are raw formula values in the syntax plan and
scalars/vectors after materialization.
"""
struct _BRMResponseModifierPlan{B,L,U}
    kind::Symbol
    base::B
    lower::L
    upper::U
end

struct _BRMObservationWeightPlan{D,V<:AbstractVector}
    kind::Symbol
    distribution::D
    source::Symbol
    values::V
end

"""
    _BRMMissingResponsePlan

Backend-neutral materialization of `mi(y)`: the original partly-missing row
axis, the observed and missing row indices, and the concrete observed values.
Concrete backends decide how missing rows become latent variables and how the
merged response is exposed.
"""
struct _BRMMissingResponsePlan{Y<:AbstractVector,O<:AbstractVector{Int},
                               M<:AbstractVector{Int},V<:AbstractVector}
    source::Symbol
    values::Y
    observed_indices::O
    missing_indices::M
    observed_values::V
end

function _brm_missing_response_plan(lhs; prefix="BRM backend lowering")
    lhs isa ExprColumn && getf(lhs) === mi || return nothing
    isempty(getkwargs(lhs)) || error(
        "$prefix: `mi(response)` accepts no keywords")
    args = getargs(lhs)
    length(args) == 1 || error(
        "$prefix: `mi(response)` expects exactly one argument")
    inner = only(args)
    inner isa NamedColumn || error(
        "$prefix: `mi(...)` expects one named response column, got " *
        "$(typeof(inner))")
    backing = parent(inner)
    backing isa DataColumn || error(
        "$prefix: `mi($(name(inner)))` requires a raw data column with " *
        "missing values, got backing $(typeof(backing))")
    raw = parent(backing)
    raw isa AbstractVector || error(
        "$prefix: `mi($(name(inner)))` requires a vector response")
    Missing <: eltype(raw) || error(
        "$prefix: `mi($(name(inner)))` requires a column whose element type " *
        "admits `missing` (got $(eltype(raw))); drop `mi(...)` when there are no NAs")
    value_type = nonmissingtype(eltype(raw))
    value_type <: Real || error(
        "$prefix: `mi($(name(inner)))` currently requires a real-valued response; " *
        "got non-missing element type $value_type")
    observed_indices = findall(!ismissing, raw)
    missing_indices = findall(ismissing, raw)
    isempty(missing_indices) && error(
        "$prefix: `mi($(name(inner)))` found no missing values; drop the wrapper")
    observed_values = collect(value_type, raw[observed_indices])
    _BRMMissingResponsePlan(
        name(inner), collect(raw), observed_indices, missing_indices,
        observed_values)
end

_brm_observation_weight_kind(f) =
    f === aweights || f === AnalyticWeights ? :analytic :
    f === fweights || f === FrequencyWeights ? :frequency :
    f === weights || f === Weights ? :power :
    f === pweights || f === ProbabilityWeights ? :probability :
    f === uweights || f === UnitWeights ? :unit : nothing

function _brm_prepare_observation_weight_values(
        kind::Symbol, raw, nobs::Integer, target::Symbol, source::Symbol;
        prefix="BRM backend lowering")
    raw isa AbstractVector{<:Real} || error(
        "$prefix: weight column `$source` for response `$target` must be a real " *
        "vector, got $(typeof(raw))")
    length(raw) == nobs || error(
        "$prefix: weight column `$source` has length $(length(raw)) but response " *
        "`$target` has length $nobs")
    values = collect(Float64, raw)
    all(isfinite, values) || error(
        "$prefix: weight column `$source` for response `$target` contains " *
        "non-finite values")
    if kind === :analytic
        all(>(0), values) || error(
            "$prefix: analytic/precision weights for response `$target` must be " *
            "strictly positive")
    elseif kind === :frequency
        all(x -> x >= 0 && isinteger(x), values) || error(
            "$prefix: frequency weights for response `$target` must be " *
            "nonnegative integer-valued counts")
    elseif kind === :power
        all(>=(0), values) || error(
            "$prefix: power-likelihood weights for response `$target` must be " *
            "nonnegative")
    else
        error("$prefix: internal unsupported observation-weight kind `$kind`")
    end
    values
end

function _brm_observation_weight_plan(rhs, target::Symbol, response;
                                      prefix="BRM backend lowering")
    rhs isa ExprColumn && getf(rhs) === weighted || return nothing
    response isa AbstractVector || error(
        "$prefix: weighted response `$target` must be a one-dimensional " *
        "observation vector; got $(typeof(response))")
    isempty(getkwargs(rhs)) || error(
        "$prefix: `weighted(distribution, weights)` accepts no keywords")
    args = getargs(rhs)
    length(args) == 2 || error(
        "$prefix: `weighted(distribution, weights)` expects exactly two arguments")
    distribution, weight = args
    distribution isa ExprColumn || error(
        "$prefix: first argument of `weighted` for response `$target` must be a " *
        "distribution call, got $(typeof(distribution))")
    weight isa ExprColumn || error(
        "$prefix: second argument of `weighted` must be a StatsBase weight " *
        "constructor, got $(typeof(weight))")
    isempty(getkwargs(weight)) || error(
        "$prefix: `weighted(..., $(getf(weight))(...))` does not accept weight " *
        "constructor keywords")
    kind = _brm_observation_weight_kind(getf(weight))
    isnothing(kind) && error(
        "$prefix: unsupported weight constructor `$(getf(weight))`; use " *
        "`aweights`, `fweights`, or `weights`")
    kind === :probability && error(
        "$prefix: `ProbabilityWeights` sampling-weight semantics are not " *
        "implemented; they are not interchangeable with likelihood weights")
    kind === :unit && error(
        "$prefix: omit `weighted(...)` for unit weights; write the base " *
        "distribution directly")
    weight_args = getargs(weight)
    length(weight_args) == 1 || error(
        "$prefix: `weighted` expects a one-column StatsBase weight constructor " *
        "such as `aweights(k)`, `fweights(n)`, or `weights(w)`; got " *
        "$(length(weight_args)) arguments for response `$target`")
    source_column = only(weight_args)
    source_column isa NamedColumn && parent(source_column) isa DataColumn || error(
        "$prefix: weights for response `$target` must be built from one raw " *
        "dataframe column, got $(typeof(source_column))")
    source = name(source_column)
    raw = parent(parent(source_column))
    values = _brm_prepare_observation_weight_values(
        kind, raw, length(response), target, source; prefix)
    _BRMObservationWeightPlan(kind, distribution, source, values)
end

_brm_normalize_response_bound(::Nothing) = nothing
_brm_normalize_response_bound(x::NamedColumn) =
    _brm_is_nothing_column(x) ? nothing : x
_brm_normalize_response_bound(x) = x

function _brm_response_modifier_plan(wrapper, args, kwargs::NamedTuple;
                                     prefix="BRM backend lowering")
    kind = wrapper === truncated || wrapper === :truncated ? :truncated :
           wrapper === censored || wrapper === :censored ? :censored :
           wrapper === interval_censored || wrapper === :interval_censored ?
               :interval_censored : nothing
    isnothing(kind) && return nothing

    if kind === :interval_censored
        length(args) == 1 || error(
            "$prefix: `interval_censored` expects one base distribution argument")
        keys(kwargs) == (:upper,) || error(
            "$prefix: `interval_censored` requires exactly the `upper` keyword; " *
            "the response column is the interval lower endpoint")
        base = only(args)
        return _BRMResponseModifierPlan(
            kind, base, nothing,
            _brm_normalize_response_bound(kwargs.upper))
    end

    length(args) in (1, 3) || error(
        "$prefix: `$kind` expects a base distribution and either keyword " *
        "bounds or positional `(lower, upper)` bounds, got $(length(args)) arguments")
    lower, upper = if length(args) == 3
        isempty(kwargs) || error(
            "$prefix: `$kind` cannot mix positional bounds with keyword bounds")
        (_brm_normalize_response_bound(args[2]),
         _brm_normalize_response_bound(args[3]))
    else
        unknown = setdiff(collect(keys(kwargs)), [:lower, :upper])
        isempty(unknown) || error(
            "$prefix: `$kind` accepts only `lower` and `upper` keywords, got $unknown")
        (_brm_normalize_response_bound(get(kwargs, :lower, nothing)),
         _brm_normalize_response_bound(get(kwargs, :upper, nothing)))
    end
    isnothing(lower) && isnothing(upper) && error(
        "$prefix: `$kind` needs at least one non-`nothing` bound")
    _BRMResponseModifierPlan(kind, first(args), lower, upper)
end

function _brm_response_modifier_plan(rhs;
                                     prefix="BRM backend lowering")
    rhs isa ExprColumn || return nothing
    _brm_response_modifier_plan(
        getf(rhs), getargs(rhs), getkwargs(rhs); prefix)
end

function _brm_materialize_response_bound(bound, data::AbstractDict, n::Integer,
                                         label::Symbol;
                                         prefix="BRM backend lowering")
    isnothing(bound) && return nothing
    raw = if bound isa NamedColumn && parent(bound) isa DataColumn
        get(data, name(bound), parent(parent(bound)))
    else
        _brm_materialize_data_expression(bound)
    end
    raw isa Real || raw isa AbstractVector{<:Real} || error(
        "$prefix: response $label bound must be a numeric scalar or a pure " *
        "expression over raw numeric data columns")
    raw isa Real && return raw
    length(raw) == n || error(
        "$prefix: response $label bound has $(length(raw)) rows; expected $n")
    collect(raw)
end

_brm_response_bound_at(bound::Real, _i) = bound
_brm_response_bound_at(bound::AbstractVector, i) = bound[i]

function _brm_materialize_bounded_response(
        spec::_BRMResponseModifierPlan, target::Symbol,
        response::AbstractVector, data::AbstractDict;
        support_kind::Symbol=:continuous,
        prefix="BRM backend lowering")
    spec.kind in (:truncated, :censored) || error(
        "$prefix: internal response modifier `$(spec.kind)` is not a bounded " *
        "response distribution")
    n = length(response)
    lower = _brm_materialize_response_bound(
        spec.lower, data, n, :lower; prefix)
    upper = _brm_materialize_response_bound(
        spec.upper, data, n, :upper; prefix)
    support_kind in (:continuous, :discrete) || error(
        "$prefix: internal unsupported bounded-response support `$support_kind`")
    if support_kind === :discrete
        all(x -> x isa Integer && !(x isa Bool), response) || error(
            "$prefix: `$(spec.kind)` discrete base family requires an integer response")
        for (label, bound) in ((:lower, lower), (:upper, upper))
            isnothing(bound) && continue
            values = bound isa AbstractVector ? bound : (bound,)
            all(x -> x isa Integer && !(x isa Bool), values) || error(
                "$prefix: `$(spec.kind)` discrete $label bounds must be integers")
        end
    end
    for i in eachindex(response)
        lo = isnothing(lower) ? nothing : _brm_response_bound_at(lower, i)
        hi = isnothing(upper) ? nothing : _brm_response_bound_at(upper, i)
        (isnothing(lo) || isnothing(hi) || lo <= hi) || error(
            "$prefix: `$(spec.kind)` lower bounds must not exceed upper bounds")
        (isnothing(lo) || lo <= response[i]) &&
        (isnothing(hi) || response[i] <= hi) || error(
            "$prefix: `$(spec.kind)` response `$target` contains values outside its bounds")
    end
    _BRMResponseModifierPlan(spec.kind, spec.base, lower, upper)
end

function _brm_materialize_interval_response(
        spec::_BRMResponseModifierPlan, target::Symbol,
        response::AbstractVector, data::AbstractDict;
        support_kind::Symbol=:continuous,
        prefix="BRM backend lowering")
    spec.kind === :interval_censored || error(
        "$prefix: internal response modifier `$(spec.kind)` is not interval evidence")
    n = length(response)
    upper = _brm_materialize_response_bound(
        spec.upper, data, n, :upper; prefix)
    isnothing(upper) && error(
        "$prefix: `interval_censored` requires an upper bound")
    support_kind in (:continuous, :discrete) || error(
        "$prefix: internal unsupported interval-response support `$support_kind`")
    if support_kind === :discrete
        all(x -> x isa Integer && !(x isa Bool), response) || error(
            "$prefix: `interval_censored` discrete base family requires integer " *
            "lower endpoints")
        values = upper isa AbstractVector ? upper : (upper,)
        all(x -> x isa Integer && !(x isa Bool), values) || error(
            "$prefix: `interval_censored` discrete upper bounds must be integers")
    end
    all(eachindex(response)) do i
        response[i] < _brm_response_bound_at(upper, i)
    end || error(
        "$prefix: `interval_censored` lower endpoints must be strictly below " *
        "upper endpoints for response `$target`")
    _BRMResponseModifierPlan(:interval_censored, spec.base, nothing, upper)
end

function _brm_backend_context(brmi::BRMI;
                              data::AbstractDict=Dict{Symbol,Any}(),
                              retain_mm_sources::Bool=false)
    prepass = Dict{Symbol,Any}()
    for op in values(brmi.operations)
        _brm_visit_op!(prepass, op)
    end

    skip_data = copy(get(prepass, :skip_data, Set{Symbol}()))
    mm_sources = Set{Symbol}()
    non_mm_sources = Set{Symbol}()
    for op in values(brmi.operations)
        payload = op isa NamedColumn ? parent(op) : op
        _brm_collect_mm_sources!(mm_sources, payload)
        _brm_collect_non_mm_sources!(non_mm_sources, payload)
    end
    retain_mm_sources || union!(skip_data, setdiff(mm_sources, non_mm_sources))
    prepass[:skip_data] = skip_data

    for op in values(brmi.operations)
        _brm_collect_data!(data, op; skip=skip_data)
    end

    _BRMBackendContext(brmi, data, prepass, _brm_collect_target_obs(brmi),
                       _brm_resolve_term_priors(brmi),
                       _brm_group_declarations(brmi))
end

# ---- narrow shared population design --------------------------------------

struct _BRMPopulationPreprocess{C,R,D<:Tuple}
    kind::Symbol
    const_::C
    raw_ref::R
    dependencies::D
end
_BRMPopulationPreprocess(kind::Symbol, const_, raw_ref) =
    _BRMPopulationPreprocess(kind, const_, raw_ref, ())

struct _BRMPopulationColumn{A<:Tuple,V<:AbstractVector,P}
    label::Symbol
    effect_addresses::A
    effect_block::Symbol
    source::Union{Nothing,Symbol}
    values::V
    preprocess::P
end

struct _BRMPopulationFixedTerm{V<:AbstractVector,P}
    label::Symbol
    source::Symbol
    values::V
    preprocess::P
end

struct _BRMPopulationDesign{C<:Tuple,M<:AbstractMatrix,T<:Tuple,V<:AbstractVector}
    target::Symbol
    columns::C
    matrix::M
    row_source::Symbol
    fixed_terms::T
    fixed::V
end

_brm_replay_expression(value::Number, _data) = value
_brm_replay_column(data, key) = data[key]
_brm_replay_column(lookup::Function, key) = lookup(key)
_brm_replay_expression(value::NamedColumn, data) = _brm_replay_column(data, name(value))
function _brm_broadcast_data_call(f, args::Tuple, kwargs::NamedTuple)
    isempty(kwargs) && return broadcast(f, args...)
    nargs = length(args)
    names = keys(kwargs)
    broadcast(args..., values(kwargs)...) do values...
        f(values[1:nargs]...; NamedTuple{names}(values[nargs+1:end])...)
    end
end
function _brm_replay_expression(value::ExprColumn, data)
    _brm_broadcast_data_call(
        getf(value),
        map(argument -> _brm_replay_expression(argument, data),
            getargs(value)),
        map(argument -> _brm_replay_expression(argument, data),
            getkwargs(value)))
end

_brm_population_preproc_entry(preprocess::_BRMPopulationPreprocess) =
    _BRMPreprocEntry(
        preprocess.kind, preprocess.const_, preprocess.raw_ref,
        preprocess.kind === :population_factor_dummy)

_brm_population_replay_input(::Union{
        Val{:zscale},Val{:standardize},Val{:center}},
        column, preprocess, data, _cache) =
    preprocess.raw_ref isa AbstractColumn ?
        _brm_replay_expression(preprocess.raw_ref, data) : data[column.source]

_brm_population_replay_input(::Val{:protect}, _column, preprocess, data, _cache) =
    _brm_replay_expression(preprocess.raw_ref, data)

_brm_population_replay_input(::Val{:population_factor_dummy},
        column, _preprocess, data, _cache) = data[column.source]

function _brm_population_replay_input(::Val{:interaction}, _column,
                                      preprocess, data, cache)
    for dependency in preprocess.dependencies
        _brm_replay_population_column(dependency, data, cache)
    end
    left, right = preprocess.raw_ref
    left_values = get(cache, left, get(data, left, nothing))
    right_values = get(cache, right, get(data, right, nothing))
    isnothing(left_values) && error(
        "BRM replay: interaction input `$left` is unavailable")
    isnothing(right_values) && error(
        "BRM replay: interaction input `$right` is unavailable")
    (left_values, right_values)
end

_brm_population_replay_input(::Val{Kind}, _column, _preprocess, _data,
                             _cache) where {Kind} = error(
    "BRM replay: unsupported population preprocessing kind `$(Kind)`")

function _brm_replay_population_column(column, data, cache)
    haskey(cache, column.label) && return cache[column.label]
    preprocess = column.preprocess
    values = if isnothing(column.source)
        nothing
    elseif isnothing(preprocess)
        collect(Float64, data[column.source])
    else
        kind = Val(preprocess.kind)
        input = _brm_population_replay_input(
            kind, column, preprocess, data, cache)
        replay = _brm_replay_preprocess(
            _brm_population_preproc_entry(preprocess), input;
            freeze=true, prefix="BRM replay")
        replay.values.primary
    end
    isnothing(values) || (cache[column.label] = values)
    values
end

function _brm_replay_population_design(
        training::_BRMPopulationDesign, context::_BRMBackendContext)
    data = context.data
    haskey(data, training.row_source) || error(
        "BRM replay: row-axis source `$(training.row_source)` is absent")
    n = length(data[training.row_source])
    cache = Dict{Symbol,Vector{Float64}}()
    columns = map(training.columns) do column
        values = _brm_replay_population_column(column, data, cache)
        isnothing(values) && (values = ones(Float64, n))
        length(values) == n || error(
            "BRM replay: population column `$(column.label)` has " *
            "$(length(values)) rows; expected $n")
        _BRMPopulationColumn(
            column.label, column.effect_addresses, column.effect_block,
            column.source, values, column.preprocess)
    end
    fixed_terms = map(training.fixed_terms) do term
        values = collect(
            Float64, _brm_replay_expression(term.preprocess.raw_ref, data))
        length(values) == n || error(
            "BRM replay: fixed term `$(term.label)` has $(length(values)) " *
            "rows; expected $n")
        _BRMPopulationFixedTerm(
            term.label, term.source, values, term.preprocess)
    end
    fixed = zeros(Float64, n)
    foreach(term -> fixed .+= term.values, fixed_terms)
    matrix = hcat((column.values for column in columns)...)
    _BRMPopulationDesign(
        training.target, Tuple(columns), matrix, training.row_source,
        Tuple(fixed_terms), fixed)
end

_brm_mm_group_names(term::MultiMembershipTerm) =
    Tuple(name(group) for group in getargs(term))
_brm_mm_weight_names(term::MultiMembershipTerm) = begin
    weights = getfield(term, :weights)
    isnothing(weights) ? nothing : Tuple(name(weight) for weight in weights)
end
function _brm_mm_suffix(term::MultiMembershipTerm)
    groups = join(String.(_brm_mm_group_names(term)), "__")
    weights = _brm_mm_weight_names(term)
    weighted = isnothing(weights) ? "" : "__w__$(join(String.(weights), "__"))"
    raw = getfield(term, :normalize) ? "" : "__raw"
    Symbol("mm__", groups, weighted, raw)
end

_brm_mm_values(raw::CA.CategoricalVector) = CA.unwrap.(raw)
_brm_mm_values(raw::AbstractVector) = raw
_brm_mm_level_pool(raw::CA.CategoricalVector) = collect(CA.levels(raw))
_brm_mm_level_pool(raw::AbstractVector) = collect(unique(raw))

function _brm_mm_fit_levels(raw_groups; prefix="BRM backend lowering")
    pooled = Any[]
    for raw in raw_groups
        append!(pooled, _brm_mm_level_pool(raw))
    end
    unique!(pooled)
    try
        sort!(pooled)
    catch
        error("$prefix: `mm(...)` grouping levels must be mutually orderable " *
              "so one shared level index can be fitted (got $(repr(pooled)))")
    end
    pooled
end

function _brm_prepare_mm(raw_groups::Tuple, raw_weights, normalize::Bool;
                         levels=nothing,
                         group_names=ntuple(i -> Symbol(:group_, i),
                                            length(raw_groups)),
                         weight_names=nothing,
                         prefix="BRM backend lowering")
    n_memberships = length(raw_groups)
    n_memberships >= 2 || error(
        "$prefix: internal — multi-membership preprocessing received " *
        "$n_memberships groups")
    all(group -> group isa AbstractVector, raw_groups) || error(
        "$prefix: every `mm(...)` group must resolve to a vector column")
    n_obs = length(first(raw_groups))
    group_values = map(_brm_mm_values, raw_groups)
    for membership in 1:n_memberships
        values = group_values[membership]
        length(values) == n_obs || error(
            "$prefix: `mm(...)` group column `$(group_names[membership])` " *
            "has $(length(values)) rows; expected $n_obs to match " *
            "`$(group_names[1])`")
        missing_row = findfirst(ismissing, values)
        isnothing(missing_row) || error(
            "$prefix: `mm(...)` group column `$(group_names[membership])` " *
            "is missing at row $missing_row")
    end

    fitted_levels = isnothing(levels) ?
        _brm_mm_fit_levels(raw_groups; prefix) : collect(levels)
    isempty(fitted_levels) && error(
        "$prefix: `mm(...)` grouping columns contain no levels")
    level_map = Dict(level => index
                     for (index, level) in enumerate(fitted_levels))
    group_idx = Vector{Int}(undef, n_obs * n_memberships)
    for observation in 1:n_obs, membership in 1:n_memberships
        level = group_values[membership][observation]
        haskey(level_map, level) || error(
            "$prefix: replay: `mm(...)` group column " *
            "`$(group_names[membership])` has unseen level `$(level)` at row " *
            "$observation (training levels: $(fitted_levels)). Re-fit with " *
            "`freeze_constants=false` or resample every membership column " *
            "to derive a new shared level set.")
        group_idx[(observation - 1) * n_memberships + membership] =
            level_map[level]
    end

    weights = Vector{Float64}(undef, n_obs * n_memberships)
    if isnothing(raw_weights)
        fill!(weights, inv(Float64(n_memberships)))
    else
        length(raw_weights) == n_memberships || error(
            "$prefix: internal — `mm(...)` has $n_memberships groups but " *
            "$(length(raw_weights)) weights")
        names = isnothing(weight_names) ?
            ntuple(i -> Symbol(:weight_, i), n_memberships) : weight_names
        for membership in 1:n_memberships
            values = raw_weights[membership]
            values isa AbstractVector || error(
                "$prefix: `mm(...)` weight `$(names[membership])` must " *
                "resolve to a vector column")
            length(values) == n_obs || error(
                "$prefix: `mm(...)` weight column `$(names[membership])` " *
                "has $(length(values)) rows; expected $n_obs")
            for observation in 1:n_obs
                value = values[observation]
                value isa Real || error(
                    "$prefix: `mm(...)` weight `$(names[membership])` at row " *
                    "$observation must be real; got $(repr(value))")
                isfinite(value) || error(
                    "$prefix: `mm(...)` weight `$(names[membership])` at row " *
                    "$observation must be finite; got $(repr(value))")
                value >= 0 || error(
                    "$prefix: `mm(...)` weight `$(names[membership])` at row " *
                    "$observation must be nonnegative; got $(repr(value))")
                converted = Float64(value)
                isfinite(converted) || error(
                    "$prefix: `mm(...)` weight `$(names[membership])` at row " *
                    "$observation cannot be represented as finite Float64")
                weights[(observation - 1) * n_memberships + membership] =
                    converted
            end
        end
        for observation in 1:n_obs
            row = ((observation - 1) * n_memberships + 1):(observation * n_memberships)
            total = sum(@view weights[row])
            isfinite(total) || error(
                "$prefix: `mm(...)` weights have a non-finite total at row " *
                "$observation")
            total > 0 || error(
                "$prefix: `mm(...)` weights must have a positive total at " *
                "row $observation")
            normalize && (@views weights[row] ./= total)
        end
    end
    (; levels=fitted_levels, group_idx, weights, n_obs, n_memberships)
end

struct _BRMRandomEffectPlan{L<:AbstractVector,I<:AbstractVector{Int},
                            S<:AbstractVector,
                            C<:Tuple,M<:AbstractMatrix,
                            SP<:AbstractVector,
                            SF<:AbstractVector{Int},SR<:AbstractVector{Float64}}
    predictor::Symbol
    id::Union{Nothing,Symbol}
    group::Symbol
    by::Union{Nothing,Symbol}
    levels::L
    strata::S
    indices::I
    stratum_indices::Vector{Int}
    group_strata::Vector{Int}
    columns::C
    matrix::M
    intercept_only::Bool
    zero_correlation::Bool
    centered::Bool
    sd_prior::SP
    sd_family::SF
    sd_rate::SR
    lkj_eta::Float64
end

struct _BRMMultiMembershipPlan{L<:AbstractVector,I<:AbstractVector{Int},
                               C<:Tuple,M<:AbstractMatrix,
                               W<:AbstractVector{Float64}}
    predictor::Symbol
    id::Nothing
    group::Symbol
    by::Nothing
    levels::L
    strata::Vector{Any}
    indices::I
    stratum_indices::Vector{Int}
    group_strata::Vector{Int}
    columns::C
    matrix::M
    intercept_only::Bool
    zero_correlation::Bool
    centered::Bool
    sd_prior::Vector{Any}
    sd_family::Vector{Int}
    sd_rate::Vector{Float64}
    lkj_eta::Float64
    groups::Tuple
    weight_sources::Union{Nothing,Tuple}
    weights::W
    n_obs::Int
    n_memberships::Int
    normalize::Bool
end

function _brm_replay_random_effect_plan(
        training::_BRMRandomEffectPlan, context::_BRMBackendContext)
    raw_groups = context.data[training.group]
    lookup = Dict(level => index for (index, level) in pairs(training.levels))
    unknown = unique(
        [level for level in raw_groups if !haskey(lookup, level)])
    isempty(unknown) || error(
        "BRM replay: grouping factor `$(training.group)` contains unseen " *
        "level(s) $(collect(unknown)); fitted levels are " *
        "$(collect(training.levels))")
    indices = Int[lookup[level] for level in raw_groups]
    stratum_indices = Int[]
    group_strata = Int[]
    if !isnothing(training.by)
        raw_strata = context.data[training.by]
        length(raw_strata) == length(raw_groups) || error(
            "BRM replay: grouping factor `$(training.group)` and stratum " *
            "column `$(training.by)` have different row counts")
        stratum_lookup = Dict(
            level => index for (index, level) in pairs(training.strata))
        unknown_strata = unique(
            [level for level in raw_strata if !haskey(stratum_lookup, level)])
        isempty(unknown_strata) || error(
            "BRM replay: stratum `$(training.by)` contains unseen level(s) " *
            "$(collect(unknown_strata)); fitted strata are " *
            "$(collect(training.strata))")
        stratum_indices = Int[stratum_lookup[level] for level in raw_strata]
        observed_mapping = _brm_group_strata(
            indices, stratum_indices, length(training.levels),
            training.group, training.by)
        for group_index in eachindex(observed_mapping)
            observed_mapping[group_index] == 0 && continue
            observed_mapping[group_index] == training.group_strata[group_index] ||
                error("BRM replay: group `$(training.group)` level " *
                      "$(training.levels[group_index]) changed stratum in " *
                      "`$(training.by)`")
        end
        group_strata = training.group_strata
    end
    n = length(indices)
    cache = Dict{Symbol,Vector{Float64}}()
    columns = map(training.columns) do column
        values = _brm_replay_population_column(column, context.data, cache)
        isnothing(values) && (values = ones(Float64, n))
        length(values) == n || error(
            "BRM replay: random-effect column `$(column.label)` and grouping " *
            "factor `$(training.group)` have different row counts")
        _BRMPopulationColumn(
            column.label, column.effect_addresses, column.effect_block,
            column.source, values, column.preprocess)
    end
    matrix = hcat((column.values for column in columns)...)
    _BRMRandomEffectPlan(
        training.predictor, training.id, training.group, training.by,
        training.levels, training.strata, indices, stratum_indices,
        group_strata, Tuple(columns), matrix, training.intercept_only,
        training.zero_correlation, training.centered, training.sd_prior,
        training.sd_family,
        training.sd_rate, training.lkj_eta)
end

function _brm_replay_random_effect_plan(
        training::_BRMMultiMembershipPlan, context::_BRMBackendContext)
    raw_groups = Tuple(context.data[group] for group in training.groups)
    raw_weights = isnothing(training.weight_sources) ? nothing :
        Tuple(context.data[weight] for weight in training.weight_sources)
    prepared = _brm_prepare_mm(
        raw_groups, raw_weights, training.normalize;
        levels=training.levels, group_names=training.groups,
        weight_names=training.weight_sources, prefix="BRM replay")
    cache = Dict{Symbol,Vector{Float64}}()
    columns = map(training.columns) do column
        values = _brm_replay_population_column(column, context.data, cache)
        isnothing(values) && (values = ones(Float64, prepared.n_obs))
        length(values) == prepared.n_obs || error(
            "BRM replay: random-effect column `$(column.label)` and " *
            "multi-membership rows have different row counts")
        _BRMPopulationColumn(
            column.label, column.effect_addresses, column.effect_block,
            column.source, values, column.preprocess)
    end
    matrix = hcat((column.values for column in columns)...)
    _BRMMultiMembershipPlan(
        training.predictor, nothing, training.group, nothing,
        training.levels, Any[], prepared.group_idx, Int[], Int[],
        Tuple(columns), matrix, training.intercept_only, false, false,
        training.sd_prior, training.sd_family, training.sd_rate, training.lkj_eta,
        training.groups, training.weight_sources, prepared.weights,
        prepared.n_obs, prepared.n_memberships, training.normalize)
end

_brm_replay_random_effect_plans(training::Tuple, context::_BRMBackendContext) =
    Tuple(_brm_replay_random_effect_plan(block, context) for block in training)

_brm_additive_terms(rhs) = begin
    terms = Any[]
    _brm_collect_additive_terms!(terms, rhs)
    terms
end
_brm_collect_additive_terms!(terms, x::ExprColumn{typeof(+)}) =
    foreach(a -> _brm_collect_additive_terms!(terms, a), getargs(x))
function _brm_collect_additive_terms!(terms, x::Integer)
    x == 0 || push!(terms, x)
    nothing
end
_brm_collect_additive_terms!(terms, x) = push!(terms, x)

_brm_is_grouped_term(term::ExprColumn) =
    getf(term) === (|) || getf(term) === doublepipe
_brm_is_grouped_term(_term) = false

_brm_group_id(::Nothing) = nothing
_brm_group_id(value::Symbol) = value
_brm_group_id(value::AbstractString) = Symbol(value)
_brm_group_id(value) = error(
    "BRM backend lowering: `gr(...; id=...)` expects a Symbol or String; " *
    "got $(typeof(value))")

function _brm_random_effect_group(raw)
    if raw isa NamedColumn && parent(raw) isa DataColumn
        return (; group=raw, by=nothing, id=nothing)
    end
    raw isa ExprColumn && getf(raw) === gr || error(
        "BRM backend lowering: random-effect group must be one raw data " *
        "column or `gr(group; by=..., id=...)`; got `$(repr(raw))`")
    args = getargs(raw)
    length(args) == 1 || error(
        "BRM backend lowering: `gr(...)` expects exactly one positional " *
        "group; got $(length(args))")
    group = only(args)
    group isa NamedColumn && parent(group) isa DataColumn || error(
        "BRM backend lowering: `gr(...)` group must be one raw data column; " *
        "got `$(repr(group))`")
    kwargs = getkwargs(raw)
    unknown = setdiff(Set(keys(kwargs)), Set((:by, :id)))
    isempty(unknown) || error(
        "BRM backend lowering: `gr(...)` has unsupported keyword(s) " *
        "$(sort!(collect(unknown)))")
    by = get(kwargs, :by, nothing)
    if !isnothing(by)
        by isa NamedColumn && parent(by) isa DataColumn || error(
            "BRM backend lowering: `gr(...; by=...)` must name one raw data " *
            "column; got `$(repr(by))`")
    end
    (; group, by, id=_brm_group_id(get(kwargs, :id, nothing)))
end

function _brm_group_declaration(predictor::Symbol, term::ExprColumn)
    args = getargs(term)
    length(args) in (2, 3) || error(
        "BRM backend lowering: grouped term `$(repr(term))` does not have " *
        "the expected `(effects | group)` or `(effects | ID | group)` shape")
    inner = first(args)
    explicit_id = length(args) == 3 ? args[2] : nothing
    if !isnothing(explicit_id) && !(explicit_id isa Symbol)
        error("BRM backend lowering: shared random-effect ID must be a " *
              "symbol; got `$(repr(explicit_id))`")
    end
    raw_group = last(args)
    uncorrelated = getf(term) === doublepipe

    if raw_group isa MultiMembershipTerm
        descriptor = raw_group
        id = explicit_id
    else
        parsed = _brm_random_effect_group(raw_group)
        if !isnothing(explicit_id) && !isnothing(parsed.id)
            error("BRM backend lowering: `(effects | ID | group)` cannot " *
                  "also carry `gr(...; id=...)`")
        end
        id = isnothing(explicit_id) ? parsed.id : explicit_id
        descriptor = isnothing(parsed.by) ? parsed.group :
            (parsed.group, parsed.by)
    end
    _BRMGroupDeclaration(
        predictor, term, Tuple(_brm_additive_terms(inner)), raw_group,
        explicit_id, id, uncorrelated, descriptor)
end

function _brm_group_declarations(brmi::BRMI)
    declarations = _BRMGroupDeclaration[]
    for (predictor, operation) in pairs(brmi.operations)
        expression = operation isa NamedColumn ? parent(operation) : operation
        expression isa ExprColumn && getf(expression) === (~) || continue
        _, rhs = getargs(expression, 2)
        for term in _brm_additive_terms(rhs)
            _brm_is_grouped_term(term) || continue
            push!(declarations, _brm_group_declaration(predictor, term))
        end
    end
    Tuple(declarations)
end

function _brm_group_strata(group_indices, stratum_indices, n_groups,
                           group::Symbol, by::Symbol)
    length(group_indices) == length(stratum_indices) || error(
        "BRM backend lowering: group `$group` and stratum `$by` have " *
        "different row counts")
    mapping = zeros(Int, n_groups)
    for (group_index, stratum_index) in zip(group_indices, stratum_indices)
        if mapping[group_index] == 0
            mapping[group_index] = stratum_index
        elseif mapping[group_index] != stratum_index
            error("BRM backend lowering: gr($group, by=$by): group level " *
                  "$group_index straddles multiple strata " *
                  "($(mapping[group_index]) vs $stratum_index)")
        end
    end
    mapping
end

function _brm_simple_random_effect_plans(
        brmi::BRMI, target::Symbol, context::_BRMBackendContext;
        required::Bool=false)
    grouped = filter(declaration -> declaration.predictor === target,
                     context.group_declarations)
    plans = ()
    seen = Set{Any}()
    for declaration in grouped
        term = declaration.expression
        zero_correlation = declaration.uncorrelated
        group_raw = declaration.raw_group
        multi_membership = group_raw isa MultiMembershipTerm
        prepared_mm = nothing
        groups = ()
        weight_sources = nothing
        normalize = true
        if multi_membership
            isnothing(declaration.id) || error(
                "BRM backend lowering: `mm(...)` cannot carry a shared " *
                "random-effect ID")
            zero_correlation && error(
                "BRM backend lowering: `mm(...)` is one shared correlated " *
                "membership block and does not support `||`")
            groups = _brm_mm_group_names(group_raw)
            weight_sources = _brm_mm_weight_names(group_raw)
            normalize = getfield(group_raw, :normalize)
            key = (:mm, groups, weight_sources, normalize)
            key in seen && error(
                "BRM backend lowering: predictor `$target` repeats " *
                "multi-membership block `mm($(join(groups, ", ")))`")
            push!(seen, key)
            raw_groups = Tuple(context.data[group_name]
                               for group_name in groups)
            raw_weights = isnothing(weight_sources) ? nothing :
                Tuple(context.data[weight] for weight in weight_sources)
            prepared_mm = _brm_prepare_mm(
                raw_groups, raw_weights, normalize;
                group_names=groups, weight_names=weight_sources)
            id = nothing
            group = _brm_mm_suffix(group_raw)
            by = nothing
            levels = prepared_mm.levels
            indices = prepared_mm.group_idx
            strata = Any[]
            stratum_indices = Int[]
            group_strata = Int[]
        else
            descriptor = declaration.descriptor
            id = declaration.id
            group_column = descriptor isa Tuple ? first(descriptor) : descriptor
            by_column = descriptor isa Tuple ? last(descriptor) : nothing
            group = name(group_column)
            by = isnothing(by_column) ? nothing : name(by_column)
            if zero_correlation && !isnothing(id)
                required && error(
                    "BRM backend lowering: shared `(effects | ID | group)` " *
                    "blocks are correlated; `||` cannot carry an ID")
                return nothing
            end
            zero_correlation && !isnothing(by) && error(
                "BRM backend lowering: zero-correlation `||` is not " *
                "supported for stratified `gr($group, by=$by)` blocks")
            key = (id, group, by)
            key in seen && error(
                "BRM backend lowering: predictor `$target` repeats " *
                "random-effect block " *
                "$(isnothing(id) ? "" : "ID `$id`, ")group `$group`")
            push!(seen, key)
            raw = context.data[group]
            raw isa AbstractVector || error(
                "BRM backend lowering: grouping column `$group` must be a " *
                "vector")
            levels = collect(_brm_fit_levels(raw))
            n_levels, indices = _brm_level_index(raw)
            length(levels) == n_levels || error(
                "BRM backend lowering: inconsistent fitted levels for " *
                "group `$group`")
            strata = Any[]
            stratum_indices = Int[]
            group_strata = Int[]
            if !isnothing(by)
                raw_strata = context.data[by]
                raw_strata isa AbstractVector || error(
                    "BRM backend lowering: stratum column `$by` must be a " *
                    "vector")
                length(raw_strata) == length(raw) || error(
                    "BRM backend lowering: group `$group` and stratum `$by` " *
                    "have different row counts")
                strata = collect(_brm_fit_levels(raw_strata))
                n_strata, raw_stratum_indices = _brm_level_index(raw_strata)
                length(strata) == n_strata || error(
                    "BRM backend lowering: inconsistent fitted levels for " *
                    "stratum `$by`")
                stratum_indices = collect(Int, raw_stratum_indices)
                group_strata = _brm_group_strata(
                    indices, stratum_indices, n_levels, group, by)
            end
        end
        raw_columns = Any[]
        for inner_term in declaration.effects
            columns = _brm_random_effect_columns(inner_term)
            if isnothing(columns) || any(column ->
                    !isnothing(column.source) &&
                    !(column.values isa AbstractVector{<:Real}), columns)
                required && error(
                    "BRM backend lowering: random-slope designs support the " *
                    "backend-neutral population column surface (continuous, " *
                    "categorical, transformed, and interaction columns); " *
                    "got `$(repr(inner_term))` in `$(repr(term))`")
                return nothing
            end
            append!(raw_columns, columns)
        end
        isempty(raw_columns) && error(
            "BRM backend lowering: random-effect term `$(repr(term))` has no columns")
        n = multi_membership ? prepared_mm.n_obs : length(indices)
        columns = map(raw_columns) do column
            values = isnothing(column.source) ? ones(Float64, n) : column.values
            length(values) == n || error(
                "BRM backend lowering: random-effect predictor `$target` and " *
                "group `$group` have different row counts")
            _BRMPopulationColumn(
                column.label, column.effect_addresses, column.effect_block,
                column.source, values, column.preprocess)
        end
        matrix = hcat((column.values for column in columns)...)
        intercept_only = length(columns) == 1 &&
                         only(columns).label === :Intercept
        n_terms = length(columns)
        if multi_membership
            plans = (plans..., _BRMMultiMembershipPlan(
                target, nothing, group, nothing, levels, Any[], indices,
                Int[], Int[], Tuple(columns), matrix, intercept_only, false,
                false, Any[nothing for _ in 1:n_terms], zeros(Int, n_terms),
                ones(Float64, n_terms), 1.0,
                groups, weight_sources, prepared_mm.weights,
                prepared_mm.n_obs, prepared_mm.n_memberships, normalize))
        else
            plans = (plans..., _BRMRandomEffectPlan(
                target, id, group, by, levels, strata, indices,
                stratum_indices, group_strata, Tuple(columns), matrix,
                intercept_only, zero_correlation, false,
                Any[nothing for _ in 1:n_terms], zeros(Int, n_terms),
                ones(Float64, n_terms), 1.0))
        end
    end
    plans
end

function _brm_population_column(term::Integer)
    term == 1 || return nothing
    (; label=:Intercept, effect_addresses=(:Intercept,),
       effect_block=:Intercept, source=nothing,
       values=nothing, preprocess=nothing)
end
function _brm_population_column(term::NamedColumn)
    backing = parent(term)
    backing isa DataColumn || return nothing
    raw = parent(backing)
    raw isa AbstractVector{<:Real} || return nothing
    eltype(raw) <: Integer && return nothing
    (; label=name(term), effect_addresses=(name(term),),
       effect_block=name(term), source=name(term),
       values=collect(Float64, raw), preprocess=nothing)
end

_brm_fit_zscale(v::AbstractVector{<:Real}) = let
    fit = _brm_fit_zscale_numeric(v, :predictor,
        message -> ArgumentError("BRM preparation: $message"))
    (fit.mean, fit.scale)
end
_brm_apply_zscale(c::Tuple, v::AbstractVector{<:Real}) =
    (v .- c[1]) ./ c[2]
_brm_fit_center(v::AbstractVector{<:Real}) = sum(v) / length(v)
_brm_apply_center(mu::Real, v::AbstractVector{<:Real}) = v .- mu

_brm_data_expression_sources!(sources, term::NamedColumn) = begin
    parent(term) isa DataColumn && push!(sources, name(term))
    sources
end
_brm_data_expression_sources!(sources, term::ExprColumn) = begin
    foreach(arg -> _brm_data_expression_sources!(sources, arg), getargs(term))
    foreach(arg -> _brm_data_expression_sources!(sources, arg), values(getkwargs(term)))
    sources
end
_brm_data_expression_sources!(sources, _term) = sources
_brm_data_expression_sources(term) =
    _brm_data_expression_sources!(Symbol[], term)

# A BRM DSL term head (`mo`, `me`, `s`, `t2`, `gp`, `hsgp`, `ar`, `dar`, `mo1`) is a
# parameter-owning special term, NOT a pure data function: it carries no method
# that accepts a materialized data value, so broadcasting it as if it were
# `log`/`*` throws a bare `MethodError` (`mo(::Int64)`). Such a term is never
# materialisable as a population data column — reuse the canonical `_TERM_HEADS`
# set so a new special term is covered without editing here too.
_brm_is_term_head(f) = f isa Function && nameof(f) in _TERM_HEADS

_brm_materialize_data_expression(x::Number) = x
_brm_materialize_data_expression(x::NamedColumn) = begin
    backing = parent(x)
    backing isa DataColumn ? parent(backing) : nothing
end
function _brm_materialize_data_expression(x::ExprColumn)
    _brm_is_term_head(getf(x)) && return nothing
    args = map(_brm_materialize_data_expression, getargs(x))
    kwargs = map(_brm_materialize_data_expression, getkwargs(x))
    any(isnothing, args) && return nothing
    any(isnothing, values(kwargs)) && return nothing
    _brm_broadcast_data_call(getf(x), args, kwargs)
end
_brm_materialize_data_expression(_x) = nothing

function _brm_materialize_count_argument(argument, n::Integer, role::AbstractString;
                                         prefix="BRM backend lowering")
    raw = _brm_materialize_data_expression(argument)
    isnothing(raw) && error(
        "$prefix: $role must be a nonnegative integer constant or a pure " *
        "expression over raw numeric data columns")
    values = raw isa AbstractVector ? raw : fill(raw, n)
    length(values) == n || error(
        "$prefix: $role has $(length(values)) rows; expected $n")
    all(x -> x isa Integer && !(x isa Bool) && x >= 0, values) || error(
        "$prefix: $role must contain only nonnegative integers")
    Int.(values)
end

function _brm_validate_binomial_response(response::AbstractVector, trials,
                                         target::Symbol;
                                         prefix="BRM backend lowering")
    length(response) == length(trials) || error(
        "$prefix: Binomial response `$target` has $(length(response)) rows; " *
        "expected $(length(trials))")
    all(eachindex(response)) do i
        y = response[i]
        y isa Integer && !(y isa Bool) && 0 <= y <= trials[i]
    end || error(
        "$prefix: Binomial response `$target` must contain integer counts " *
        "between zero and its row's trial count")
    Int.(response)
end

_brm_wrapper_col_name(prefix::Symbol, inner::NamedColumn) =
    Symbol(prefix, :_, name(inner))
_brm_wrapper_col_name(prefix::Symbol, inner) =
    Symbol(prefix, :_expr_, string(hash(inner); base=16)[1:8])

function _brm_population_column(term::ExprColumn)
    f = getf(term)
    kind = f === zscale ? :zscale :
           f === standardize ? :standardize :
           f === center ? :center : nothing
    if isnothing(kind)
        values = _brm_materialize_data_expression(term)
        values isa AbstractVector{<:Real} || return nothing
        sources = _brm_data_expression_sources(term)
        isempty(sources) && return nothing
        label = _brm_wrapper_col_name(Symbol(getf(term)), term)
        preprocess = _BRMPopulationPreprocess(
            :protect, nothing, term)
        return (; label, effect_addresses=(label,), effect_block=label,
                source=first(sources), values=collect(Float64, values),
                preprocess)
    end
    args = getargs(term)
    length(args) == 1 || return nothing
    inner = only(args)
    raw = _brm_materialize_data_expression(inner)
    raw isa AbstractVector{<:Real} || return nothing
    sources = _brm_data_expression_sources(inner)
    isempty(sources) && return nothing

    values = collect(Float64, raw)
    const_ = kind === :center ? _brm_fit_center(values) :
                               _brm_fit_zscale(values)
    transformed = kind === :center ? _brm_apply_center(const_, values) :
                                     _brm_apply_zscale(const_, values)
    label = _brm_wrapper_col_name(kind, inner)
    preprocess = _BRMPopulationPreprocess(kind, const_, inner)
    (; label, effect_addresses=(label,), effect_block=label,
       source=first(sources), values=transformed, preprocess)
end
_brm_population_column(_term) = nothing

function _brm_population_fixed_term(term)
    term isa ExprColumn && getf(term) === offset || return nothing
    args = getargs(term)
    length(args) == 1 || error(
        "BRM backend lowering: `offset(x)` expects exactly one positional " *
        "argument, got $(length(args))")
    isempty(getkwargs(term)) || error(
        "BRM backend lowering: `offset(x)` does not accept keyword arguments")
    inner = only(args)
    values = _brm_materialize_data_expression(inner)
    values isa AbstractVector{<:Real} || error(
        "BRM backend lowering: direct-BRMI `offset(...)` currently requires a " *
        "pure expression over raw numeric data columns")
    sources = _brm_data_expression_sources(inner)
    isempty(sources) && error(
        "BRM backend lowering: `offset(...)` has no raw data row axis")
    label = _brm_wrapper_col_name(:offset, inner)
    preprocess = _BRMPopulationPreprocess(:protect, nothing, inner)
    _BRMPopulationFixedTerm(
        label, first(sources), collect(Float64, values), preprocess)
end

_brm_fit_levels(raw::CA.CategoricalVector) = CA.levels(raw)
_brm_apply_levels(levels, raw::AbstractVector) =
    _brm_apply_fitted_levels(levels, raw)
_brm_level_index(raw::CA.CategoricalVector) =
    length(CA.levels(raw)), Int.(CA.levelcode.(raw))
_brm_level_index(raw::AbstractVector) = begin
    levels = _brm_fit_levels(raw)
    lookup = Dict(level => i for (i, level) in enumerate(levels))
    length(levels), Int[lookup[level] for level in raw]
end

function _brm_categorical_population_columns(
        raw, source::Symbol, block::Symbol,
        effect_addresses::Tuple=(block,); ref::Integer=1)
    is_categorical = raw isa CA.CategoricalVector ||
                     (raw isa AbstractVector && eltype(raw) <: Integer)
    is_categorical || return nothing
    n_levels, indices = _brm_level_index(raw)
    n_levels >= 2 || return nothing
    levels = _brm_fit_levels(raw)
    Tuple(begin
        label = Symbol(block, :_lvl_, level)
        values = Float64[index == level ? 1.0 : 0.0 for index in indices]
        preprocess = _BRMPopulationPreprocess(
            :population_factor_dummy,
            (; levels, level, n_levels, ref), source)
        (; label, effect_addresses, effect_block=block, source, values,
           preprocess)
    end for level in 2:n_levels)
end

function _brm_population_columns(term::NamedColumn)
    backing = parent(term)
    categorical = backing isa DataColumn ?
        _brm_categorical_population_columns(
            parent(backing), name(term), name(term)) : nothing
    !isnothing(categorical) && return categorical
    column = _brm_population_column(term)
    isnothing(column) ? nothing : (column,)
end

function _brm_population_columns(term::ExprColumn{typeof(factor)})
    args = getargs(term)
    length(args) == 1 || return nothing
    inner = only(args)
    inner isa NamedColumn || return nothing
    backing = parent(inner)
    backing isa DataColumn || return nothing
    raw = parent(backing)
    raw isa AbstractVector || return nothing
    raw_values = if raw isa CA.CategoricalVector
        levels = CA.levels(raw)
        eltype(levels) <: Integer || return nothing
        [levels[code] for code in Int.(CA.levelcode.(raw))]
    else
        eltype(raw) <: Integer || return nothing
        raw
    end
    kwargs = getkwargs(term)
    all(k -> k === :ref, keys(kwargs)) || return nothing
    ref_raw = get(kwargs, :ref, 1)
    ref_raw isa Integer || return nothing
    1 <= ref_raw <= maximum(raw_values) || error(
        "BRM backend lowering: `factor($(name(inner)); ref=$ref_raw)` ref " *
        "out of range (max level $(maximum(raw_values)))")

    source = name(inner)
    block = ref_raw == 1 ? source : Symbol(source, :__ref_, ref_raw)
    recoded = ref_raw == 1 ? raw_values :
        Int[value == ref_raw ? 1 : value == 1 ? ref_raw : value
            for value in raw_values]
    addresses = ref_raw == 1 ? (source,) : (block, source)
    _brm_categorical_population_columns(
        recoded, source, block, addresses; ref=ref_raw)
end
_brm_population_columns(term) = let column = _brm_population_column(term)
    isnothing(column) ? nothing : (column,)
end

function _brm_random_categorical_column(column)
    _brm_population_column_is_categorical(column) || return column
    level = column.preprocess.const_.level
    label = Symbol(column.source, :_dummy_, level)
    _BRMPopulationColumn(
        label, (label,), column.effect_block, column.source, column.values,
        column.preprocess)
end

function _brm_random_effect_columns(term::ExprColumn{typeof(&)})
    args = getargs(term)
    length(args) == 2 || return nothing
    left = _brm_random_effect_columns(args[1])
    right = _brm_random_effect_columns(args[2])
    (isnothing(left) || isnothing(right) ||
     any(c -> isnothing(c.source), left) ||
     any(c -> isnothing(c.source), right)) && return nothing
    Tuple(_brm_interaction_population_column(l, r)
          for l in left for r in right)
end
function _brm_random_effect_columns(term)
    columns = _brm_population_columns(term)
    isnothing(columns) && return nothing
    Tuple(_brm_random_categorical_column(column) for column in columns)
end

_brm_population_column_is_categorical(column) =
    !isnothing(column.preprocess) &&
    column.preprocess.kind === :population_factor_dummy

function _brm_interaction_population_column(left, right)
    left_cat = _brm_population_column_is_categorical(left)
    right_cat = _brm_population_column_is_categorical(right)
    # StanBlocks names mixed interactions with the continuous operand first,
    # independent of surface order. Preserve that exact public effect address.
    if left_cat && !right_cat
        return _brm_interaction_population_column(right, left)
    end
    length(left.values) == length(right.values) || error(
        "BRM backend lowering: interaction `$(left.label) & $(right.label)` " *
        "mixes row axes of lengths $(length(left.values)) and " *
        "$(length(right.values))")

    label = Symbol(:int_, left.label, :_x_, right.label)
    values = left.values .* right.values
    dependencies = Tuple(c for c in (left, right)
                         if !isnothing(c.preprocess))
    preprocess = _BRMPopulationPreprocess(
        :interaction, nothing, (left.label, right.label), dependencies)
    (; label, effect_addresses=(label,), effect_block=label,
       source=left.source, values, preprocess)
end

function _brm_population_columns(term::ExprColumn{typeof(&)})
    args = getargs(term)
    length(args) == 2 || return nothing
    left = _brm_population_columns(args[1])
    right = _brm_population_columns(args[2])
    (isnothing(left) || isnothing(right) ||
     any(c -> isnothing(c.source), left) ||
     any(c -> isnothing(c.source), right)) && return nothing
    Tuple(_brm_interaction_population_column(l, r)
          for l in left for r in right)
end

"""
    _brm_simple_population_design(target, rhs, data, obs_name; required=false)

Materialise the backend-neutral population design for the first common surface:
an additive intercept, continuous raw-data columns, pure numeric data
expressions, fitted `zscale`/`standardize`/`center` columns, pairwise
continuous/categorical interactions, treatment contrasts for integer or
`CategoricalVector` columns, and pure data-derived fixed `offset(...)` terms.
Plain grouped terms are separated into `_BRMRandomEffectPlan` values rather
than being mistaken for population columns.
Returns `nothing` for a term requiring richer lowering unless `required=true`,
in which case it fails loudly. SBBRMI and Turing consume the shared
coefficient-bearing representation; Turing also consumes the materialized
fixed-offset vector. Categorical columns reuse SBBRMI's established ordered
level-coding primitive.
"""
function _brm_simple_population_design(target::Symbol, rhs,
                                       data::AbstractDict,
                                       obs_name::Union{Nothing,Symbol};
                                       required::Bool=false)
    _brm_population_design(target, Tuple(_brm_additive_terms(rhs)), data,
                           obs_name; required)
end

# A backend may lower structured terms separately while sharing exactly the
# same fitted population columns. Taking the selected terms explicitly avoids
# treating a mixed formula as an unsupported population design.
function _brm_population_design(target::Symbol, terms::Tuple,
                                data::AbstractDict,
                                obs_name::Union{Nothing,Symbol};
                                required::Bool=false,
                                row_source::Union{Nothing,Symbol}=nothing)
    raw_columns = Any[]
    fixed_terms = _BRMPopulationFixedTerm[]
    for term in terms
        # Grouped terms have their own backend-neutral geometry. They are not
        # coefficient-bearing population columns and are planned separately.
        _brm_is_grouped_term(term) && continue
        fixed = _brm_population_fixed_term(term)
        if !isnothing(fixed)
            push!(fixed_terms, fixed)
            continue
        end
        columns = _brm_population_columns(term)
        if isnothing(columns)
            required && error(
                "BRM backend lowering: predictor `$target` contains unsupported " *
                "population term `$(repr(term))`; the shared initial surface " *
                "supports `1`, continuous raw-data columns, pure numeric data " *
                "expressions and fixed data-derived `offset(...)` terms, plus " *
                "`zscale`, `standardize`, and `center` of one numeric column, " *
                "pairwise continuous/categorical interactions, and ordered " *
                "treatment contrasts for integer or `CategoricalVector` columns")
            return nothing
        end
        append!(raw_columns, columns)
    end
    isempty(raw_columns) && begin
        required && error("BRM backend lowering: predictor `$target` has no terms")
        return nothing
    end

    concrete = filter(c -> !isnothing(c.source), raw_columns)
    row_sources = Symbol[c.source for c in concrete]
    append!(row_sources, (term.source for term in fixed_terms))
    row_source = if !isnothing(row_source)
        row_source
    elseif isempty(concrete)
        if !isempty(row_sources)
            first(row_sources)
        elseif isnothing(obs_name)
            required && error(
                "BRM backend lowering: intercept-only predictor `$target` has no " *
                "observed response from which to determine its row axis")
            return nothing
        else
            obs_name
        end
    else
        first(concrete).source
    end
    haskey(data, row_source) || begin
        required && error(
            "BRM backend lowering: row-axis source `$row_source` for predictor " *
            "`$target` is absent from materialized data")
        return nothing
    end
    row_value = data[row_source]
    n = row_value isa Integer && !(row_value isa Bool) ?
        Int(row_value) : length(row_value)

    columns = map(raw_columns) do column
        values = isnothing(column.source) ? ones(Float64, n) : column.values
        length(values) == n || error(
            "BRM backend lowering: predictor `$target` mixes row axes of lengths " *
            "$n and $(length(values))")
        _BRMPopulationColumn(
            column.label, column.effect_addresses, column.effect_block,
            column.source, values, column.preprocess)
    end
    matrix = hcat((c.values for c in columns)...)
    fixed = zeros(Float64, n)
    for term in fixed_terms
        length(term.values) == n || error(
            "BRM backend lowering: predictor `$target` mixes row axes of " *
            "lengths $n and $(length(term.values)) in fixed offset `$(term.label)`")
        fixed .+= term.values
    end
    _BRMPopulationDesign(
        target, Tuple(columns), matrix, row_source, Tuple(fixed_terms), fixed)
end

struct _BRMPopulationPredictor{F,D<:_BRMPopulationDesign}
    name::Symbol
    link_lhs_fn::F
    emitted_name::Symbol
    design::D
end

function _brm_replay_population_predictor(
        training::_BRMPopulationPredictor, context::_BRMBackendContext)
    design = _brm_replay_population_design(training.design, context)
    _BRMPopulationPredictor(
        training.name, training.link_lhs_fn, training.emitted_name, design)
end

_brm_lp_emitted_name(name::Symbol, link_lhs_fn) =
    link_lhs_fn === identity ? name : Symbol(nameof(link_lhs_fn), :_, name)

function _brm_simple_population_predictor(brmi::BRMI, target::Symbol,
                                          context::_BRMBackendContext;
                                          required::Bool=false)
    op = linear_predictor_op(brmi, target)
    if isnothing(op) || getf(op) !== (~)
        required && error(
            "BRM backend lowering: predictor `$target` has no population formula")
        return nothing
    end
    lhs, rhs = getargs(op, 2)
    peeled = _peel_lp_lhs(lhs)
    if isnothing(peeled) || last(peeled) !== target
        required && error(
            "BRM backend lowering: predictor `$target` has an unsupported LHS")
        return nothing
    end
    link_lhs_fn, name = peeled
    design = _brm_simple_population_design(
        name, rhs, context.data, get(context.target_obs, name, nothing);
        required)
    isnothing(design) && return nothing
    _BRMPopulationPredictor(
        name, link_lhs_fn, _brm_lp_emitted_name(name, link_lhs_fn), design)
end

# ---- shared population-effect prior semantics -----------------------------

# This is a representation test, not a Stan lowering decision. Keep it in the
# backend-neutral layer so a weak-dependency backend can validate distribution
# declarations without loading SBBRMI.
_as_distribution_type(::Type{T}) where {T<:Distribution} = T
_as_distribution_type(_) = nothing

_brm_effect_specificity(spec) =
    (spec.predictor === _EFFECT_COLON ? 0 : 1) +
    (spec.coefficient === _EFFECT_COLON ? 0 : 1)
_brm_effect_spelling(spec) =
    "effect($(spec.predictor), $(spec.coefficient))"

function _brm_validate_population_effect_spec(spec;
                                               prefix="BRM backend lowering")
    spec.expression isa ExprColumn || error(
        "$prefix: population `effect(...)` RHS must be a callable prior expression")
    nothing
end

# One precedence engine is shared by the simple direct-BRMI plan and SBBRMI's
# richer population/categorical resolver. A cell stores both the exact parsed
# RHS and the specificity that won it; formula order never decides a tie.
function _brm_claim_effect_prior!(get_cell, set_cell!, spec, what;
                                  prefix="BRM backend lowering")
    rank = _brm_effect_specificity(spec)
    spelling = _brm_effect_spelling(spec)
    held = get_cell()
    if isnothing(held) || rank > held.rank
        set_cell!((; expression=spec.expression, rank, spelling))
    elseif rank == held.rank
        error("$prefix: `$spelling` and `$(held.spelling)` are equally specific " *
              "and both set the prior for $what. Neither wins — make one of " *
              "them more specific, or drop it.")
    end
    nothing
end

_brm_is_array_value(::AbstractArray) = true
_brm_is_array_value(_) = false
function _brm_is_array_value(value::NamedColumn)
    backing = parent(value)
    backing isa DataColumn && return parent(backing) isa AbstractArray
    if backing isa ExprColumn{typeof(~)}
        shape = _brm_distribution_shape(last(getargs(backing)))
        return !isnothing(shape) && first(shape) !== Distributions.Univariate
    end
    false
end

function _brm_normal_effect_args(rhs::ExprColumn;
                                 prefix="BRM backend lowering")
    args = getargs(rhs)
    length(args) <= 2 || error(
        "$prefix: `effect(...) ~ Normal(...)` must lower to exactly location " *
        "and scale arguments, got $(length(args))")
    any(_brm_is_array_value, args) && error(
        "$prefix: each coefficient's Normal prior needs scalar location and scale; " *
        "index or reduce an array-valued argument explicitly")
    isempty(args) && return (0.0, 1.0)
    length(args) == 1 && return (only(args), 1.0)
    args
end

_brm_numeric_constant(x::Real) = Float64(x)
function _brm_numeric_constant(x::ExprColumn)
    isempty(getkwargs(x)) || return nothing
    values = map(_brm_numeric_constant, getargs(x))
    any(isnothing, values) && return nothing
    value = try
        getf(x)(values...)
    catch
        return nothing
    end
    value isa Real ? Float64(value) : nothing
end
_brm_numeric_constant(_x) = nothing

# ---- shared random-effect prior resolution ---------------------------------

function _brm_ranef_sd_prior(spec, spelling::AbstractString;
                             prefix="BRM backend lowering")
    spec.expression isa ExprColumn || error(
        "$prefix: `$spelling` RHS must be a callable distribution expression")
    spec.expression
end

# Temporary compatibility metadata for consumers which have not migrated to
# `sd_prior`. Unknown families deliberately return `nothing`: semantic prior
# resolution must remain open, and a legacy backend may then diagnose its own
# missing lowering rather than constraining the common representation.
function _brm_legacy_ranef_sd_prior(prior::ExprColumn)
    T = _as_distribution_type(getf(prior))
    args = getargs(prior)
    if !isnothing(T) && T <: Exponential && length(args) in (0, 1)
        scale = isempty(args) ? 1.0 : _brm_numeric_constant(only(args))
        !isnothing(scale) && isfinite(scale) && scale > 0 &&
            return (1, 1.0 / scale)
    elseif !isnothing(T) && T <: Normal && length(args) <= 2
        location = isempty(args) ? 0.0 : _brm_numeric_constant(args[1])
        scale = length(args) < 2 ? 1.0 : _brm_numeric_constant(args[2])
        !isnothing(location) && !isnothing(scale) && location == 0 &&
            isfinite(scale) && scale > 0 && return (2, scale)
    end
    nothing
end

function _brm_ranef_lkj_eta(spec, n_terms::Int;
                            prefix="BRM backend lowering")
    T = _as_distribution_type(spec.family)
    (!isnothing(T) && T <: LKJCholesky) || error(
        "$prefix: `cor(:, ID)` expects `LKJCholesky(K, eta)`; got `$(spec.family)`")
    isempty(spec.keywords) || error(
        "$prefix: `cor(:, ID) ~ LKJCholesky(...)` does not accept keywords")
    length(spec.arguments) == 2 || error(
        "$prefix: `LKJCholesky` correlation priors require dimension K and eta")
    k = _brm_numeric_constant(spec.arguments[1])
    eta = _brm_numeric_constant(spec.arguments[2])
    isnothing(k) && error(
        "$prefix: `LKJCholesky(K, eta)` dimension K must be an integer formula constant")
    isinteger(k) || error(
        "$prefix: `LKJCholesky(K, eta)` dimension K must be an integer formula constant")
    Int(k) == n_terms || error(
        "$prefix: `LKJCholesky($(Int(k)), ...)` does not match the addressed " *
        "random-effect block width $n_terms")
    isnothing(eta) && error(
        "$prefix: LKJ eta must be a numeric formula constant")
    isfinite(eta) && eta > 0 || error(
        "$prefix: LKJ eta must be finite and strictly positive, got $eta")
    eta
end

function _brm_ranef_margin_claim(spec, margins;
                                 prefix="BRM backend lowering")
    spelling = "sd($(isnothing(spec.predictor) ? ":" : spec.predictor), " *
               "$(spec.id)" *
               (isnothing(spec.coefficient) ? "" : ", $(spec.coefficient)") * ")"
    if isnothing(spec.predictor) && isnothing(spec.coefficient)
        return nothing
    elseif isnothing(spec.predictor)
        hits = findall(m -> m.coefficient === spec.coefficient, margins)
        isempty(hits) && error(
            "$prefix: `$spelling` matches no random-effect margin")
        return (hits, 1)
    elseif isnothing(spec.coefficient)
        hits = findall(m -> m.predictor === spec.predictor, margins)
        isempty(hits) && error(
            "$prefix: `$spelling` matches no random-effect margin")
        length(hits) == 1 || error(
            "$prefix: `$spelling` is ambiguous because that predictor " *
            "contributes $(length(hits)) margins; name a coefficient")
        return (hits, 1)
    end
    hits = findall(m -> m.predictor === spec.predictor &&
                        m.coefficient === spec.coefficient, margins)
    isempty(hits) && error(
        "$prefix: `$spelling` matches no random-effect margin")
    length(hits) == 1 || error(
        "$prefix: `$spelling` matches $(length(hits)) margins and is ambiguous")
    (hits, 2)
end

"""
    _brm_resolve_ranef_effect_overrides(specs, bucket_margins)

Resolve public `sd(...)` and `cor(...)` statements against backend-supplied
ordered `(predictor, coefficient)` margin axes. Both SBBRMI and direct Turing
use this precedence and validation engine; only their executable distributions
differ.
"""
function _brm_resolve_ranef_effect_overrides(
        specs, bucket_margins; prefix="BRM backend lowering")
    isempty(specs) && return Dict{Any,NamedTuple}()
    states = Dict{Any,Dict{Symbol,Any}}()
    for spec in specs
        matches = Any[key for key in keys(bucket_margins)
                      if first(key) === spec.id]
        isempty(matches) && error(
            "$prefix: `$(spec.class)(:, $(spec.id))` matches no shared " *
            "`|$(spec.id)|` random-effect block")
        length(matches) == 1 || error(
            "$prefix: public `|$(spec.id)|` addresses $(length(matches)) blocks " *
            "with different grouping factors; use a unique ID")
        key = only(matches)
        margins = bucket_margins[key]
        state = get!(states, key) do
            Dict{Symbol,Any}(
                :margins => margins,
                :sd_default => nothing,
                :sd_overrides => Dict{Int,Tuple{ExprColumn,Int}}(),
                :lkj_eta => nothing)
        end
        if spec.class === :cor
            state[:lkj_eta] === nothing || error(
                "$prefix: duplicate correlation prior for `cor(:, $(spec.id))`")
            state[:lkj_eta] = _brm_ranef_lkj_eta(
                spec, length(margins); prefix)
            continue
        end
        prior = _brm_ranef_sd_prior(spec, "sd(:, $(spec.id))"; prefix)
        claim = _brm_ranef_margin_claim(spec, margins; prefix)
        if isnothing(claim)
            state[:sd_default] === nothing || error(
                "$prefix: duplicate block SD prior for `sd(:, $(spec.id))`")
            state[:sd_default] = prior
            continue
        end
        indices, rank = claim
        for index in indices
            held = get(state[:sd_overrides], index, nothing)
            if isnothing(held) || rank > held[2]
                state[:sd_overrides][index] =
                    (prior, rank)
            elseif rank == held[2]
                error("$prefix: two SD prior statements are equally specific " *
                      "and both set margin $(margins[index]) of `|$(spec.id)|`")
            end
        end
    end
    out = Dict{Any,NamedTuple}()
    for (key, state) in states
        margins = state[:margins]
        default_prior = state[:sd_default]
        sd_prior = Any[default_prior for _ in margins]
        for (index, (prior, _)) in state[:sd_overrides]
            sd_prior[index] = prior
        end
        legacy = map(sd_prior) do prior
            isnothing(prior) ? (0, 1.0) : _brm_legacy_ranef_sd_prior(prior)
        end
        sd_family = Int[isnothing(x) ? -1 : first(x) for x in legacy]
        sd_rate = Float64[isnothing(x) ? NaN : last(x) for x in legacy]
        out[key] = (; sd_prior, sd_family, sd_rate,
                    has_sd=(!isnothing(default_prior) ||
                            !isempty(state[:sd_overrides])),
                    lkj_eta=isnothing(state[:lkj_eta]) ? 1.0 : state[:lkj_eta],
                    margins)
    end
    out
end

"""
    _brm_simple_population_effect_overrides(
        brmi, design; available_predictors=(design.target,))

Resolve formula-level `effect(...) ~ Normal(...)` statements against a narrow
shared population design. The output is aligned 1:1 with `design.columns` and
contains the winning parsed RHS (or `nothing` for the default Normal(0, 1)).
An `effect(lp, categorical_column)` address fans out over that column's K-1
treatment contrasts, matching SBBRMI's one-prior-per-contrast-block contract.
For a distributional likelihood, `available_predictors` names its complete
predictor set: a prior targeting a peer is ignored by this component, while a
prior targeting no member still fails loudly.
"""
function _brm_simple_population_effect_overrides(brmi::BRMI,
                                                 design::_BRMPopulationDesign;
                                                 prefix="BRM backend lowering",
                                                 available_predictors=(design.target,))
    specs = effect_priors(brmi)
    isempty(specs) && return nothing

    labels = Symbol[c.label for c in design.columns]
    address_blocks = Dict{Symbol,Set{Symbol}}()
    for column in design.columns, address in column.effect_addresses
        push!(get!(address_blocks, address, Set{Symbol}()), column.effect_block)
    end
    ambiguous = Dict(address => blocks for (address, blocks) in address_blocks
                     if length(blocks) > 1)
    available = Symbol[address for (address, blocks) in address_blocks
                       if length(blocks) == 1]
    cells = Any[nothing for _ in labels]
    for spec in specs
        _brm_validate_population_effect_spec(spec; prefix)
        all_predictors = spec.predictor === _EFFECT_COLON
        if !all_predictors && spec.predictor !== design.target
            spec.predictor in available_predictors && continue
            error("$prefix: `$(_brm_effect_spelling(spec))` names no linear " *
                  "predictor in this backend plan. Available predictors: " *
                  "$(join(available_predictors, ", ")).")
        end

        indices = if spec.coefficient === _EFFECT_COLON
            eachindex(labels)
        else
            if haskey(ambiguous, spec.coefficient)
                blocks = sort!(collect(ambiguous[spec.coefficient]))
                error("$prefix: `$(spec.coefficient)` ambiguously names " *
                      "categorical contrast blocks $(join(blocks, ", ")). " *
                      "Address an emitted block name explicitly.")
            end
            idxs = findall(c -> spec.coefficient in c.effect_addresses,
                           design.columns)
            isempty(idxs) && error(
                "$prefix: `$(spec.coefficient)` is not a population coefficient " *
                "of `$(design.target)`. Available labels: " *
                "$(join(sort!(available), ", ")).")
            idxs
        end
        for idx in indices
            _brm_claim_effect_prior!(
                () -> cells[idx], v -> (cells[idx] = v), spec,
                "`$(design.target)`'s `$(labels[idx])` column"; prefix)
        end
    end
    Any[isnothing(cell) ? nothing : cell.expression for cell in cells]
end

function _brm_materialize_normal_effect_priors(overrides, n::Integer;
                                               prefix="BRM backend lowering")
    isnothing(overrides) && return (zeros(Float64, n), ones(Float64, n))
    length(overrides) == n || error(
        "$prefix: internal effect-prior alignment error: $(length(overrides)) " *
        "priors for $n population columns")
    location = zeros(Float64, n)
    scale = ones(Float64, n)
    for i in eachindex(overrides)
        isnothing(overrides[i]) && continue
        raw_location, raw_scale = _brm_normal_effect_args(overrides[i]; prefix)
        loc = _brm_numeric_constant(raw_location)
        sd = _brm_numeric_constant(raw_scale)
        isnothing(loc) && error(
            "$prefix: population-effect Normal location must be a numeric constant")
        isnothing(sd) && error(
            "$prefix: population-effect Normal scale must be a numeric constant")
        isfinite(loc) || error(
            "$prefix: population-effect Normal location must be finite")
        isfinite(sd) && sd > 0 || error(
            "$prefix: population-effect Normal scale must be finite and positive")
        location[i] = loc
        scale[i] = sd
    end
    location, scale
end

# Operation keys for the declarations consumed by the resolver above. These
# keys are model statements, not stray operations, and must be admitted by a
# strict direct-BRMI backend after their semantics have been validated.
function _brm_population_effect_operation_keys(brmi::BRMI)
    out = Set{Symbol}()
    for (key, op_nc) in pairs(brmi.operations)
        op = _named_op(op_nc)
        isnothing(op) && continue
        getf(op) === (~) || continue
        lhs, rhs = getargs(op, 2)
        lhs_e = _as_expr_column(lhs)
        rhs_e = _as_expr_column(rhs)
        isnothing(lhs_e) && continue
        isnothing(rhs_e) && continue
        getf(lhs_e) === effect || continue
        address = getargs(lhs_e)
        length(address) == 2 || continue
        first(address) in _NON_EFFECT_CLASSES && continue
        push!(out, key)
    end
    out
end

function _brm_ranef_effect_operation_keys(brmi::BRMI)
    out = Set{Symbol}()
    for (key, op_nc) in pairs(brmi.operations)
        op = _named_op(op_nc)
        isnothing(op) && continue
        getf(op) === (~) || continue
        lhs, rhs = getargs(op, 2)
        lhs_e = _as_expr_column(lhs)
        rhs_e = _as_expr_column(rhs)
        (isnothing(lhs_e) || isnothing(rhs_e) || getf(lhs_e) !== effect) &&
            continue
        address = getargs(lhs_e)
        isempty(address) && continue
        first(address) in (:sd, :cor) && push!(out, key)
    end
    out
end
