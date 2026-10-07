# Missing covariates retain their subject rows. The design describes their
# geometry and fixed observed-only anchors; executable columns use completed
# model values. Missing entries remain missing in geometry, never fake data.
struct _RKMissingValueSpec
    source::Symbol
    observed::Symbol
    observed_rows::Symbol
    missing_rows::Symbol
    observed_component::Symbol
    missing_lookup::Symbol
    missing_mask::Symbol
    nmissing::Int
    distribution::_BRMPreparedExpr
end

function _rk_mi_predictor_plan(term::NamedColumn)
    operation = parent(term)
    operation isa ExprColumn && getf(operation) === (~) || return nothing
    _brm_missing_response_plan(first(getargs(operation)); prefix="RK backend")
end
_rk_mi_predictor_plan(_) = nothing

# Formula geometry needs the declared row axis, not a numerical evaluation at
# invented parameter values. Track pointwise assignment dependencies back to
# their actual data/observation axes; scalar sampled parents add no row axis.
_rk_model_value_axis(::Number, _context) = ()
_rk_model_value_axis(_value, _context=nothing) = nothing
_rk_model_value_axis(value::NamedColumn, context) =
    _rk_model_value_axis(value, parent(value), context)
function _rk_model_value_axis(value::NamedColumn, backing::DataColumn, _context)
    raw = parent(backing)
    raw isa Real && return ()
    raw isa AbstractVector{<:Union{Missing,Real}} || return nothing
    (source=name(value), nrows=length(raw))
end
_rk_model_value_axis(value::NamedColumn, backing::ExprColumn{typeof(assign)}, context) =
    _rk_model_value_axis(last(getargs(backing)), context)
function _rk_model_value_axis(value::NamedColumn, backing::ExprColumn{typeof(~)}, context)
    lhs, rhs = getargs(backing)
    completion = _brm_missing_response_plan(lhs; prefix="RK backend")
    completion === nothing || return (
        source=completion.source, nrows=length(completion.values))
    _brm_observation_name(lhs) === nothing || return _rk_model_value_axis(lhs, context)
    _brm_prior_expression(rhs) && return (
        _brm_parameter_reference_axis(rhs) === :scalar ? () : nothing)
    context === nothing && return nothing
    # Reuse the predictor's actual geometry, including declared consumers and
    # group rows. The number of group levels is a coefficient axis, not this
    # predictor's row axis; no parameter values are invented to resolve it.
    geometry = _rk_model_predictor_geometry(context.parent, context, name(value);
        available_predictors=Tuple(p.name for p in linear_predictors(context.parent)),
        tolerant_default=true)
    (; source=geometry.design.row_source, nrows=size(geometry.design.matrix, 1))
end
_rk_model_value_axis(::NamedColumn, _backing, _context) = nothing
function _rk_model_value_axis(value::ExprColumn, context)
    callable = getf(value)
    (haskey(_RK_DERIVED_BINOPS, callable) ||
     haskey(_RK_DERIVED_MATH, callable) ||
     haskey(_RK_DERIVED_CMP, callable)) && isempty(getkwargs(value)) || return nothing
    axes = map(argument -> _rk_model_value_axis(argument, context), getargs(value))
    any(isnothing, axes) && return nothing
    rows = filter(!isempty, axes)
    isempty(rows) && return ()
    shape = Base.Broadcast.broadcast_shape(((row.nrows,) for row in rows)...)
    rows[findfirst(row -> row.nrows == only(shape), rows)]
end

function _rk_named_model_population_column(term::NamedColumn, context)
    parent(term) isa DataColumn && return nothing
    axis = _rk_model_value_axis(term, context)
    (axis === nothing || isempty(axis)) && return nothing
    label = name(term)
    # Missing geometry marks values unavailable until graph execution. It is
    # never bound as data: the term consumes the emitted assignment by name.
    (; label, effect_addresses=(label,), effect_block=label, source=axis.source,
        values=fill(missing, axis.nrows),
        preprocess=_BRMPopulationPreprocess(:model_value, nothing, term),
        runtime_expression=label)
end
_rk_named_model_population_column(_term, _context) = nothing

function _rk_model_population_column(term, context=nothing)
    inner = term
    kind = :model_value
    if term isa ExprColumn && getf(term) in (standardize, zscale, center)
        length(getargs(term)) == 1 && isempty(getkwargs(term)) || return nothing
        inner = only(getargs(term))
        kind = nameof(getf(term))
    end
    plan = _rk_mi_predictor_plan(inner)
    plan === nothing && return kind === :model_value ?
        _rk_named_model_population_column(inner, context) : nothing
    label = kind === :model_value ? name(inner) : _brm_wrapper_col_name(kind, inner)
    raw = plan.values
    make_error = message -> ArgumentError("RK backend: missing covariate transform: $message")
    observed = Float64.(plan.observed_values)
    fit = kind === :model_value ? nothing : kind === :center ?
        (; mean=_brm_fit_mean_numeric(observed, :predictor, :center, make_error), scale=1.0) :
        _brm_fit_zscale_numeric(observed, :predictor, make_error)
    values = fit === nothing ? raw : (raw .- fit.mean) ./ fit.scale
    anchors = Expr(:call, :_brm_observed_values, plan.source)
    expression = fit === nothing ? name(inner) :
        Expr(:call, :./,
            Expr(:call, :.-, name(inner),
                Expr(:_rk_data_preparation, :brm_covariate_mean, anchors)),
            kind === :center ? 1.0 :
                Expr(:_rk_data_preparation, :brm_covariate_sd, anchors))
    preprocess = _BRMPopulationPreprocess(kind,
        fit === nothing ? nothing : (fit.mean, fit.scale), inner)
    (; label, effect_addresses=(label,), effect_block=label,
        source=plan.source, values, preprocess, runtime_expression=expression)
end

# The marker is planning metadata, resolved to the actual packed observation
# port before source is lowered. Its mean/std remain executable graph math.
function _rk_observed_anchor_source(value::Expr, observations)
    if Meta.isexpr(value, :call, 2) && first(value.args) === :_brm_observed_values
        return observations[value.args[2]]
    end
    Expr(value.head, map(arg -> _rk_observed_anchor_source(arg, observations), value.args)...)
end
_rk_observed_anchor_source(value, _observations) = value

function _rk_model_population_columns(term; cellmeans=false, context=nothing)
    column = _rk_model_population_column(term, context)
    column === nothing ? _brm_population_columns(term; cellmeans) : (column,)
end

function _rk_model_random_effect_columns(term; cellmeans=false, context)
    column = _rk_model_population_column(term, context)
    column === nothing ? _brm_random_effect_columns(term; cellmeans) : (column,)
end

Base.@nospecializeinfer function _rk_model_predictor_geometry(@nospecialize(brmi::BRMI), context, target; kwargs...)
    data = copy(context.data)
    geometry_context = _BRMBackendContext(context.parent, data, context.prepass,
        context.target_obs, context.target_axes, context.term_priors,
        context.group_declarations)
    function population_columns(term; cellmeans=false)
        columns = _rk_model_population_columns(term; cellmeans, context=geometry_context)
        if columns !== nothing
            for column in columns
                column.source === nothing && continue
                # A wholly modeled covariate has no materialized source.
                # Its declared row axis is metadata, not executable values.
                haskey(data, column.source) ||
                    (data[column.source] = Base.OneTo(length(column.values)))
            end
        end
        columns
    end
    _brm_prepare_predictor_geometry(brmi, geometry_context, target; kwargs...,
        population_columns,
        random_effect_columns=(term; cellmeans=false) ->
            _rk_model_random_effect_columns(term; cellmeans, context=geometry_context))
end

function _rk_mi_downstream(program, key, source)
    any(operation -> operation.name !== key &&
        _rk_refs_name(operation.expression, source), program.operations)
end

# A distribution parameter on the response row axis is gathered separately
# for observed and missing rows. Scalar parameters retain scalar broadcasting.
_rk_mi_gather(value, _rows) = value
function _rk_mi_gather(value::_BRMPreparedRef, rows)
    value.axis in (:observation, :observation_row) || return value
    indices = rows isa Symbol ? _BRMPreparedRef(rows, :whole) : rows
    _BRMPreparedExpr(getindex, (value, indices), (;))
end
function _rk_mi_gather(value::_BRMPreparedExpr, rows)
    _BRMPreparedExpr(value.callable,
        map(argument -> _rk_mi_gather(argument, rows), value.args),
        map(argument -> _rk_mi_gather(argument, rows), value.kwargs))
end

function _rk_prepare_missing_value!(columns, taken, observation, program, completions, derived)
    plan = observation.missing_response
    plan === nothing && return observation
    observation.modifier === nothing || error(
        "RK backend: missing response `$(observation.name)` with response modifiers requires a supported completion law")
    observed = _rk_ast_fresh_name(string(plan.source, "_obs"), taken)
    jobs = _rk_ast_fresh_name(string("Jobs_", plan.source), taken)
    raw = _rk_ast_fresh_name(string(plan.source, "_raw"), taken)
    columns[raw] = plan.values
    columns[observed], columns[jobs] = plan.observed_values, plan.observed_indices
    push!(derived, _RKDerivedSpec(observed,
        Expr(:_rk_data_preparation, :brm_covariate_observed, raw), observed))
    push!(derived, _RKDerivedSpec(jobs,
        Expr(:_rk_data_preparation, :brm_covariate_observed_rows, raw), jobs))
    delete!(columns, plan.source)
    delete!(columns, observation.name)
    if _rk_mi_downstream(program, observation.name, plan.source)
        jmis = _rk_ast_fresh_name(string("Jmis_", plan.source), taken)
        component = _rk_ast_fresh_name(string(plan.source, "_observed_component"), taken)
        lookup = _rk_ast_fresh_name(string(plan.source, "_missing_lookup"), taken)
        mask = _rk_ast_fresh_name(string(plan.source, "_missing_mask"), taken)
        # Row partitions describe the declared covariate geometry. The
        # observed contribution, missing lookup and mask are graph values.
        columns[jmis] = plan.missing_indices
        push!(derived, _RKDerivedSpec(jmis,
            Expr(:_rk_data_preparation, :brm_covariate_missing_rows, raw), jmis))
        push!(completions, _RKMissingValueSpec(plan.source, observed,
            jobs, jmis, component, lookup, mask, length(plan.missing_indices), observation.distribution))
    end
    _BRMPreparedObservation(observed, observation.lhs,
        _rk_mi_gather(observation.distribution, jobs), plan.observed_values,
        nothing, observation.weight, nothing)
end

function _rk_emit_missing_value!(definitions, statements, bindings, taken, completion)
    geometry = _rk_ast_statistical_call!(definitions, taken,
        :brm_covariate_geometry, completion.observed, completion.observed_rows,
        completion.missing_rows; kernel=true)
    push!(statements, Expr(:(=), Expr(:tuple, completion.observed_component,
        completion.missing_lookup, completion.missing_mask), geometry))
    if completion.nmissing == 0
        push!(statements, Expr(:(=), completion.source, completion.observed_component))
        return nothing
    end
    distribution = _rk_mi_gather(completion.distribution, completion.missing_rows)
    law = _rk_ast_value_distribution(distribution, bindings, taken)
    # One block, as StanBlocks' `_sb_mi_response`: the missing entries with
    # their law, completed onto the covariate's original row axis. The
    # declared dimensions belong to the missing-row axis; unlike an
    # observation broadcast, the statement samples that independent array.
    block = _rk_block_body(law)
    observed = _rk_block_argument!(block, :observed, completion.observed_component)
    rows = _rk_block_argument!(block, :rows, completion.missing_rows)
    lookup = _rk_block_argument!(block, :lookup, completion.missing_lookup)
    mask = _rk_block_argument!(block, :mask, completion.missing_mask)
    missing = _rk_block_local!(block, :y_mis)
    push!(block.statements, Expr(:call, :.~,
        Expr(:ref, missing, Expr(:call, :axes, rows, 1)), law))
    # Completion is a whole covariate value on its original row axis. Keep
    # the gather and mask in an explicit graph when downstream likelihoods
    # have different observation axes, rather than inferring a scalar recipe.
    completed = _rk_block_local!(block, :completed)
    push!(block.statements, Expr(:(=), completed, _rk_ast_statistical_call!(definitions,
        taken, :brm_completed_covariate, observed, missing, lookup, mask; kernel=true)))
    push!(statements, Expr(:call, :~, completion.source,
        _rk_ast_block_call!(definitions, taken, "brm_missing_covariate", block,
            completed)))
end

function _rk_source_loop_indices!(names, statement)
    statement isa Expr || return names
    if statement.head === :for
        iterator = first(statement.args)
        Meta.isexpr(iterator, :(=)) && _rk_source_lhs!(names, first(iterator.args))
    end
    foreach(argument -> _rk_source_loop_indices!(names, argument), statement.args)
    names
end

# A block call `lhs ~ block(...)` also reads the caller names its submodel
# body mentions freely (for example a prior's hyperparameter or a missing
# covariate's law): every body symbol that is not an argument, a local
# output or a loop index.
function _rk_block_free_reads(definitions)
    reads = Dict{Symbol,Set{Symbol}}()
    for definition in definitions
        Meta.isexpr(definition, :(=), 2) || continue
        call, body = definition.args
        Meta.isexpr(call, :call) && Meta.isexpr(body, :block) || continue
        names = setdiff!(_rk_source_symbols!(Set{Symbol}(), body), call.args[2:end])
        for statement in body.args
            setdiff!(names, _rk_source_outputs!(Set{Symbol}(), statement))
            setdiff!(names, _rk_source_loop_indices!(Set{Symbol}(), statement))
        end
        reads[first(call.args)] = names
    end
    reads
end

# Completion, formula columns, sampled priors and readers may depend on one
# another. Order ordinary emitted statements by their declared outputs before
# PPL authoring, preserving independent source order and bound data ownership.
function _rk_order_value_statements(statements, data_names, definitions=(); observed=())
    block_reads = _rk_block_free_reads(definitions)
    outputs = map(statements) do statement
        names = _rk_source_outputs!(Set{Symbol}(), statement)
        Meta.isexpr(statement, :call) && first(statement.args) in (:~, :.~) &&
            setdiff!(names, observed)
        setdiff!(names, data_names)
        setdiff!(names, _rk_source_loop_indices!(Set{Symbol}(), statement))
    end
    producers = Dict{Symbol,Int}()
    for (index, names) in enumerate(outputs), name in names
        haskey(producers, name) && error("RK backend: multiple emitted definitions of `$name`")
        producers[name] = index
    end
    dependencies = map(eachindex(statements)) do index
        references = _rk_source_symbols!(Set{Symbol}(), statements[index])
        for name in collect(references)
            haskey(block_reads, name) && union!(references, block_reads[name])
        end
        Set(producers[name] for name in references if haskey(producers, name) &&
            producers[name] != index)
    end
    pending, finished, ordered = collect(eachindex(statements)), Set{Int}(), Expr[]
    while !isempty(pending)
        next = findfirst(index -> issubset(dependencies[index], finished), pending)
        next === nothing && error("RK backend: cyclic completed-value source dependencies")
        index = popat!(pending, next)
        push!(finished, index)
        push!(ordered, statements[index])
    end
    ordered
end
