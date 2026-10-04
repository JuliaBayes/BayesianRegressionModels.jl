# Missing covariates retain their subject rows. The design describes their
# geometry and fixed observed-only anchors; executable columns use completed
# model values. Missing entries remain missing in geometry, never fake data.
struct _RKMissingValueSpec
    source::Symbol
    observed::Symbol
    missing::Symbol
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

function _rk_model_population_column(term)
    inner = term
    kind = :model_value
    if term isa ExprColumn && getf(term) in (standardize, zscale, center)
        length(getargs(term)) == 1 && isempty(getkwargs(term)) || return nothing
        inner = only(getargs(term))
        kind = nameof(getf(term))
    end
    plan = _rk_mi_predictor_plan(inner)
    plan === nothing && return nothing
    label = kind === :model_value ? name(inner) : _brm_wrapper_col_name(kind, inner)
    raw = plan.values
    make_error = message -> ArgumentError("RK backend: missing covariate transform: $message")
    observed = Float64.(plan.observed_values)
    fit = kind === :model_value ? nothing : kind === :center ?
        (; mean=_brm_fit_mean_numeric(observed, :predictor, :center, make_error), scale=1.0) :
        _brm_fit_zscale_numeric(observed, :predictor, make_error)
    values = fit === nothing ? raw : (raw .- fit.mean) ./ fit.scale
    expression = fit === nothing ? name(inner) :
        Expr(:call, :./, Expr(:call, :.-, name(inner), fit.mean), fit.scale)
    preprocess = _BRMPopulationPreprocess(kind,
        fit === nothing ? nothing : (fit.mean, fit.scale), inner)
    (; label, effect_addresses=(label,), effect_block=label,
        source=plan.source, values, preprocess, runtime_expression=expression)
end

function _rk_model_population_columns(term; cellmeans=false)
    column = _rk_model_population_column(term)
    column === nothing ? _brm_population_columns(term; cellmeans) : (column,)
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

function _rk_prepare_missing_value!(columns, taken, observation, program, completions)
    plan = observation.missing_response
    plan === nothing && return observation
    observation.modifier === nothing || error(
        "RK backend: missing response `$(observation.name)` with response modifiers requires a supported completion law")
    observed = _rk_ast_fresh_name(string(plan.source, "_obs"), taken)
    jobs = _rk_ast_fresh_name(string("Jobs_", plan.source), taken)
    columns[observed], columns[jobs] = plan.observed_values, plan.observed_indices
    delete!(columns, plan.source)
    delete!(columns, observation.name)
    if _rk_mi_downstream(program, observation.name, plan.source)
        missing = _rk_ast_fresh_name(string(plan.source, "_y_mis"), taken)
        jmis = _rk_ast_fresh_name(string("Jmis_", plan.source), taken)
        component = _rk_ast_fresh_name(string(plan.source, "_observed_component"), taken)
        lookup = _rk_ast_fresh_name(string(plan.source, "_missing_lookup"), taken)
        mask = _rk_ast_fresh_name(string(plan.source, "_missing_mask"), taken)
        # These are the two contributions of the completion map, not filled
        # covariate data. Only actual observed values enter the observed law.
        nrows = length(plan.values)
        fixed = zeros(Float64, nrows)
        fixed[plan.observed_indices] = plan.observed_values
        indices, selected = ones(Int, nrows), zeros(Float64, nrows)
        indices[plan.missing_indices] = eachindex(plan.missing_indices)
        selected[plan.missing_indices] .= 1.0
        columns[component] = fixed
        if !isempty(plan.missing_indices)
            columns[jmis], columns[lookup], columns[mask] = plan.missing_indices, indices, selected
        end
        push!(completions, _RKMissingValueSpec(plan.source, observed, missing,
            jobs, jmis, component, lookup, mask, length(plan.missing_indices), observation.distribution))
    end
    _BRMPreparedObservation(observed, observation.lhs,
        _rk_mi_gather(observation.distribution, jobs), plan.observed_values,
        nothing, observation.weight, nothing)
end

function _rk_emit_missing_value!(definitions, statements, bindings, taken, completion)
    if completion.nmissing == 0
        push!(statements, Expr(:(=), completion.source, completion.observed_component))
        return nothing
    end
    distribution = _rk_mi_gather(completion.distribution, completion.missing_rows)
    law = _rk_ast_value_distribution(distribution, bindings, taken)
    # Declared array dimensions belong to the missing-row axis. Unlike an
    # observation broadcast, this statement samples that independent array.
    shape = Expr(:call, :axes, completion.missing_rows, 1)
    push!(statements, Expr(:call, Symbol(".~"),
        Expr(:ref, completion.missing, shape), law))
    # Completion is a whole covariate value on its original row axis. Keep
    # the gather and mask in an explicit graph when downstream likelihoods
    # have different observation axes, rather than inferring a scalar recipe.
    expression = _rk_ast_statistical_call!(definitions, taken,
        :brm_completed_covariate, completion.observed_component,
        completion.missing, completion.missing_lookup, completion.missing_mask;
        kernel=true)
    push!(statements, Expr(:(=), completion.source, expression))
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

# Completion, formula columns, sampled priors and readers may depend on one
# another. Order ordinary emitted statements by their declared outputs before
# PPL authoring, preserving independent source order and bound data ownership.
function _rk_order_value_statements(statements, data_names)
    outputs = map(statements) do statement
        names = _rk_source_outputs!(Set{Symbol}(), statement)
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
