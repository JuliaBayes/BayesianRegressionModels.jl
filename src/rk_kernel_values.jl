# Whole-array source graphs for group-local deterministic kernels. Each
# positional input retains its own inner axis; latent inputs are sliced from
# their original predictor, and explicit ragged joins use prepared row indices.
struct _RKPreparedKernelAssignment
    name::Symbol
    params::Vector{Symbol}
    body::Vector{Any}
    collected::Any
    scope::Module
    inputs::Tuple
    globals::Vector{Symbol}
    columns::Dict{Symbol,Any}
    group_values::Any
    observations::Tuple
    # The graph value holding the collected value of each subject, which a
    # per-subject observation reads entry by entry.
    cells::Symbol
    # Responses the cell's `@scan` blocks observe entry by entry, as
    # `(; param, source)`: the plate observes them in place (`y[i][t] ~ …`).
    scan_observed::Tuple
    # The raw column whose rows label the cells (a random-effect grouping or a
    # `ragged(x, group => cells)` label column), or `nothing`.
    cell_column::Union{Nothing,Symbol}
end

# The authored cell's lexical namespace is a stable callable authority, not
# per-chain state. Julia deliberately refuses to deepcopy a Module; copy the
# plan's mutable payload while retaining that namespace by identity.
function Base.deepcopy_internal(x::_RKPreparedKernelAssignment, stack::IdDict)
    _RKPreparedKernelAssignment(x.name,
        Base.deepcopy_internal(x.params, stack),
        Base.deepcopy_internal(x.body, stack),
        Base.deepcopy_internal(x.collected, stack),
        x.scope,
        Base.deepcopy_internal(x.inputs, stack),
        Base.deepcopy_internal(x.globals, stack),
        Base.deepcopy_internal(x.columns, stack),
        Base.deepcopy_internal(x.group_values, stack),
        Base.deepcopy_internal(x.observations, stack),
        x.cells,
        Base.deepcopy_internal(x.scan_observed, stack),
        x.cell_column)
end
_rk_kernel_column_name(column::NamedColumn) = name(column)

# One observed array per subject. RKPPL observes it in nested subject and entry
# plates: `cells` maps each value read per subject to the graph value holding
# its per-subject cells; every other value is shared by all subjects.
struct _RKNestedObservation{O}
    observation::O
    cells::Dict{Symbol,Symbol}
end
Base.getproperty(observation::_RKNestedObservation, field::Symbol) =
    field in (:observation, :cells) ? getfield(observation, field) :
        getproperty(getfield(observation, :observation), field)

# A response a kernel cell's `@scan` observes entry by entry. RKPPL computes
# its densities in the cell (rkppl-use §5 "Observing a carried array"): the
# program conditions on it by name and emits no observation statement for it.
struct _RKScanObservation{Y}
    name::Symbol
    response::Y
end

_rk_observation_statement(observation, base, taken) =
    Expr(:call, :.~, observation.name, base)
function _rk_observation_statement(observation::_RKNestedObservation, base, taken)
    # Every plate reuses the authored loop name `i` unless the cell reads it.
    read = union!(_rk_source_symbols!(Set{Symbol}(), base), values(observation.cells))
    index = :i
    while index in taken || index in read
        index = Symbol(index, :_)
    end
    cell = Expr(:call, :.~, Expr(:ref, observation.name, index),
        _rk_ast_index_cells(base, observation.cells, index))
    Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
        Expr(:for, Expr(:(=), index, Expr(:call, :eachindex, observation.name)),
            Expr(:block, cell)))
end
_rk_ast_index_cells(value::Symbol, cells, index) =
    haskey(cells, value) ? Expr(:ref, cells[value], index) : value
_rk_ast_index_cells(value::Expr, cells, index) =
    Expr(value.head, (_rk_ast_index_cells(arg, cells, index) for arg in value.args)...)
_rk_ast_index_cells(value, cells, index) = value

# A nested observation reads its own per-subject cells, so the row alignment
# of a flattened response does not apply.
_rk_align_kernel_observation_arguments!(defs, statements, bindings, taken, plan,
    observation::_RKNestedObservation, distribution) = distribution

function _rk_kernel_observation_callee(scope, expression, kernel)
    Meta.isexpr(expression, :call) || error(
        "RK backend: kernel `$kernel` observation needs a constructor call")
    head = first(expression.args)
    (head isa Symbol || head isa GlobalRef || Meta.isexpr(head, :.)) || error(
        "RK backend: kernel `$kernel` observation needs a named constructor")
    _rk_kernel_binding(scope, head)
end

# SLIC's unqualified built-in family tokens need not be exported into the
# caller module. A caller's actual binding still takes precedence.
# The name is read, never evaluated: while a package image is generated, every
# module but the package being precompiled is closed to `eval`, and a model
# may be defined in one of its dependencies.
function _rk_kernel_binding(scope, head)
    head isa Symbol && !isdefined(scope, head) && isdefined(StanBlocks.stan, head) &&
        return getfield(StanBlocks.stan, head)
    value = _sb_cell_binding(scope, head)
    value === nothing && error("RK backend: `$head` names nothing in module `$scope`")
    value
end

# In-cell distribution combinators take a family token as their first
# positional (`weighted(normal, w, mu, sigma)`), never a data argument.
_rk_kernel_weighted(callable) = callable === weighted || callable === StanBlocks.stan.weighted
_rk_kernel_bounded(callable) = _rk_kernel_bounded_kind(callable) !== nothing
_rk_kernel_bounded_kind(callable) =
    callable in (censored, StanBlocks.stan.censored) ? :censored :
    callable in (truncated, StanBlocks.stan.truncated) ? :truncated :
    callable in (interval_censored, StanBlocks.stan.interval_censored) ? :interval_censored :
    nothing

function _rk_kernel_observation_family(scope, expression, kernel)
    callable = _rk_kernel_observation_callee(scope, expression, kernel)
    _rk_kernel_weighted(callable) && error(
        "RK backend: kernel `$kernel` observation nests `weighted` in its family; " *
        "write one `weighted(family, weight, args...)`")
    _rk_kernel_bounded(callable) && error(
        "RK backend: kernel `$kernel` in-cell `$(nameof(callable))(family, ...)` " *
        "supports `censored`, `truncated` or `interval_censored` of a `Normal(...)` " *
        "family with `lower=`/`upper=` bounds only; use `_rk_observation_source!` " *
        "for other bounded laws")
    callable === StanBlocks.normal && return Normal
    callable
end

# A cell writes the same bounded Normal law as a formula response — censored,
# truncated or interval-censored — but its bounds may read cell locals and
# grouped inputs. Keep those expressions beside the constructor arguments so
# the authored plate computes them on its axis.
function _rk_kernel_observation_law(scope, expression, kernel)
    callable = _rk_kernel_observation_callee(scope, expression, kernel)
    kind = _rk_kernel_bounded_kind(callable)
    kind === nothing && return expression, nothing
    args = expression.args[2:end]
    parameters = filter(arg -> Meta.isexpr(arg, :parameters), args)
    length(parameters) <= 1 || error(
        "RK backend: kernel `$kernel` `$kind` has repeated keyword groups")
    bounds = Dict{Symbol,Any}()
    for keyword in (isempty(parameters) ? () : only(parameters).args)
        Meta.isexpr(keyword, :kw) && first(keyword.args) in (:lower, :upper) || error(
            "RK backend: kernel `$kernel` `$kind` accepts only `lower` and `upper` bounds")
        haskey(bounds, first(keyword.args)) && error(
            "RK backend: kernel `$kernel` `$kind` repeats `$(first(keyword.args))`")
        bounds[first(keyword.args)] = last(keyword.args)
    end
    positional = filter(arg -> !Meta.isexpr(arg, :parameters), args)
    isempty(positional) && error(
        "RK backend: kernel `$kernel` `$kind` needs a Normal family")
    family = first(positional)
    base = if Meta.isexpr(family, :call) && length(positional) == 1
        family
    elseif (family isa Symbol || family isa GlobalRef || Meta.isexpr(family, :.)) &&
            length(positional) > 1
        Expr(:call, positional...)
    else
        error("RK backend: kernel `$kernel` `$kind` needs a family call or token")
    end
    _rk_kernel_observation_family(scope, base, kernel) === Normal || error(
        "RK backend: kernel `$kernel` in-cell `$kind` currently supports Normal only")
    lower = get(bounds, :lower, nothing)
    upper = get(bounds, :upper, nothing)
    lower === :nothing && (lower = nothing)
    upper === :nothing && (upper = nothing)
    lower === nothing && upper === nothing && error(
        "RK backend: kernel `$kernel` `$kind` needs a non-nothing bound")
    # The observed value is an interval's lower endpoint.
    kind === :interval_censored && (lower === nothing && upper !== nothing || error(
        "RK backend: kernel `$kernel` `interval_censored` takes `upper=` only; the " *
        "observed value is the interval's lower endpoint"))
    base, (; kind, lower, upper)
end

# SLIC's observation weighting `weighted(family, weight, args...)` is a power
# likelihood: each row's `family(args...)` log density is scaled by its
# weight. StanBlocks' call form `weighted(family(args...), weight, extra...)`
# splices the family call's arguments after the remaining positionals, so it
# names the same observation as `weighted(family, weight, extra..., args...)`.
# A bounded combinator as the family weights that combinator's law, and its
# bounds stay its own: on the family call in `weighted(censored(Normal(m, s);
# lower=l), w)`, on `weighted` in the token form `weighted(censored, w, normal,
# m, s; lower=l)`. `_rk_kernel_observation_law` then reads the bounded law.
function _rk_kernel_observation_weight(scope, expression, kernel)
    _rk_kernel_weighted(_rk_kernel_observation_callee(scope, expression, kernel)) ||
        return expression, nothing
    keyword(arg) = Meta.isexpr(arg, (:parameters, :kw))
    positional(args) = any(keyword, args) && error(
        "RK backend: kernel `$kernel` `weighted(family, weight, args...)` " *
        "takes positional arguments only")
    args = expression.args[2:end]
    keywords = filter(keyword, args)
    values = filter(!keyword, args)
    length(values) >= 2 || error(
        "RK backend: kernel `$kernel` observation needs `weighted(family, weight, args...)`")
    family, weight, rest = values[1], values[2], values[3:end]
    call = Meta.isexpr(family, :call)
    if _rk_kernel_bounded(_rk_kernel_binding(scope, call ? first(family.args) : family))
        call || return Expr(:call, family, _rk_kernel_keyword_group(keywords)..., rest...), weight
        isempty(rest) && isempty(keywords) || error(
            "RK backend: kernel `$kernel` `weighted($(first(family.args))(...), weight)` " *
            "takes no further arguments; bounds belong on `$(first(family.args))`")
        return family, weight
    end
    positional(keywords)
    call || return Expr(:call, family, rest...), weight
    positional(family.args[2:end])
    Expr(:call, first(family.args), rest..., family.args[2:end]...), weight
end

# One `;` keyword group from call keywords written after `;` or after a comma.
_rk_kernel_keyword_group(keywords) = isempty(keywords) ? () :
    (Expr(:parameters, (arg for keyword in keywords
        for arg in (Meta.isexpr(keyword, :parameters) ? keyword.args : (keyword,)))...),)

function _rk_kernel_observation_distribution(observation, names=observation.argument_names)
    arguments = map(name -> _BRMPreparedRef(name, :whole), names)
    # The original Stan scalar family is emitted through RKPPL's exact
    # StudentT(nu, location, scale) spelling, retaining its argument order.
    if observation.callable === StanBlocks.student_t
        length(arguments) == 3 || error("RK backend: student_t needs nu, location and scale")
        nu, location, scale = arguments
        return _BRMPreparedExpr(LocationScale,
            (location, scale, _BRMPreparedExpr(TDist, (nu,), (;))), (;))
    end
    _BRMPreparedExpr(observation.callable, arguments, (;))
end

function _rk_kernel_observation_modifier(observation, distribution, bound_names)
    observation.modifier === nothing && return nothing
    function bound(label)
        index = findfirst(entry -> entry.label === label, observation.bounds)
        index === nothing ? nothing : _BRMPreparedRef(bound_names[index], :whole)
    end
    _BRMResponseModifierPlan(observation.modifier.kind, distribution,
        bound(:lower), bound(:upper))
end

# Values grouped per subject by a row partition. An absent replay response
# stays absent.
_rk_gather_rows(values, rows) = [values[r] for r in rows]
_rk_gather_rows(values::_BRMAbsentResponse, rows) = values

# A bounded in-cell observation is checked against its data when the model is
# built, as a formula response is: every entry of the response within its
# bounds, and an interval's lower endpoint below its upper one. A bound the
# cell computes from parameters is the law's own business at evaluation.
_rk_validate_cell_bounds(kernel, observation, ::_BRMAbsentResponse, data) = nothing
function _rk_validate_cell_bounds(kernel, observation, response, data)
    observation.modifier === nothing && return
    response isa AbstractVector{<:AbstractVector} || return
    values = Dict{Symbol,Any}()
    function bound(label)
        index = findfirst(entry -> entry.label === label, observation.bounds)
        index === nothing && return nothing
        expression = observation.bounds[index].expression
        expression isa Real && return expression
        expression isa Symbol || return missing
        slice = _rk_kernel_value_slice(kernel, expression)
        if isempty(slice.inputs)
            value = get(data, expression, nothing)
            return value isa Real ? value : missing
        end
        isempty(slice.body) && length(slice.inputs) == 1 || return missing
        input = only(slice.inputs)
        flat = get(kernel.columns, input.source, nothing)
        flat === nothing && return missing
        cells = input.kind === :gather ? _rk_gather_rows(flat, kernel.columns[input.rows]) : flat
        cells isa AbstractVector{<:AbstractVector} &&
            length.(cells) == length.(response) || return missing
        key = Symbol(:bound_, label)
        values[key] = brm_flatten_cells(cells)
        NamedColumn(key, DataColumn(values[key]))
    end
    lower, upper = bound(:lower), bound(:upper)
    (lower === missing || upper === missing) && return
    spec = _BRMResponseModifierPlan(observation.modifier.kind, nothing, lower, upper)
    validate = spec.kind === :interval_censored ? _brm_materialize_interval_response :
        _brm_materialize_bounded_response
    validate(spec, observation.source, brm_flatten_cells(response), values;
        prefix="RK backend")
    nothing
end

# How the graph computes a ragged join's row partition from its bound raw
# columns (`brm_ragged_rows`), or `nothing` when no column labels the cells
# (a `groups(g)` plate): the partition is then a planned input. The axis
# columns checked against `group` are an LP's data columns, or the joined
# column itself unless it is a response, which a design may omit.
function _rk_kernel_partition_source!(columns, value, group, cell_column, order,
        is_lp::Bool, observed::Bool)
    cell_column === nothing && return nothing
    raw(column) = parent(parent(column))
    axes = is_lp ? _brm_kernel_lp_data_columns(value) :
        observed || !(parent(value) isa DataColumn) ? Tuple{Symbol,Any}[] :
        Tuple{Symbol,Any}[(name(value), raw(value))]
    order_name = order === nothing ? nothing : name(order)
    for (column, values) in ((name(group), raw(group)), (name(cell_column), raw(cell_column)),
            (order === nothing ? () : ((order_name, raw(order)),))..., axes...)
        get!(columns, column, values)
    end
    (; x=name(value), group=name(group), cell=name(cell_column), order=order_name,
        axes=Symbol[first(axis) for axis in axes])
end

# Each data input's subject count; an absent replay response has none.
_rk_push_outer_length!(lengths, values) = push!(lengths, length(values))
_rk_push_outer_length!(lengths, ::_BRMAbsentResponse) = lengths

Base.@nospecializeinfer function _rk_prepare_kernel_value(@nospecialize(brmi::BRMI), program, name, rhs)
    parts = _sb_kernel_lambda_parts(first(getargs(rhs)))
    parts === nothing && error("RK backend: kernel `$name` requires an inline cell body")
    params, raw_body = parts
    # A cell ending in `nothing` has no collected value (`collected === nothing`).
    resultless = _brm_cell_has_no_result(raw_body)
    resultless && (raw_body = _brm_cell_without_result(raw_body))
    arguments = getargs(rhs)[2:end]
    length(params) == length(arguments) || error(
        "RK backend: kernel `$name` cell and positional argument counts disagree")
    isempty(getkwargs(rhs)) || error("RK backend: kernel `$name` has unsupported control keywords")
    scope = _brm_inline_scope(first(getargs(rhs)))
    direct_lps = [argument for argument in arguments
        if argument isa NamedColumn && parent(argument) isa ExprColumn]
    ragged_cols = [argument for argument in arguments
        if argument isa ExprColumn && getf(argument) === ragged]
    buckets = _sb_kernel_lp_buckets(direct_lps; ragged_cols)
    groups = [bucket for bucket in buckets if bucket !== nothing]
    # The column whose rows label the plate's cells, when one does: ragged
    # joins then compute their row partitions in the graph from it.
    cell_column = nothing
    group_values = if isempty(groups)
        # Without a subject predictor, `ragged(x, group => cells)` inputs label
        # the cells by their explicit cell label column (district totals over
        # facility predictors are the shape), and a `groups(g)` plate by the
        # labels of `g`.
        if isempty(ragged_cols)
            nothing
        else
            cells = _brm_kernel_plate_cells(ragged_cols; prefix="RK backend")
            cells.distinct || (cell_column = cells.column)
            cells.values
        end
    else
        length(unique(group.key for group in groups)) == 1 || error(
            "RK backend: kernel `$name` predictor inputs disagree on subject grouping")
        column = first(groups).column
        cell_column = column
        _brm_kernel_check_cell_labels(ragged_cols, _rk_kernel_column_name(column); prefix="RK backend")
        _brm_kernel_subject_values(parent(parent(column)), _rk_kernel_column_name(column); prefix="RK backend")
    end
    columns = Dict{Symbol,Any}()
    inputs = Any[]
    outer_lengths = Int[]
    observed = _rk_kernel_observed_columns(brmi, params, arguments, raw_body)
    scanned = _rk_scan_observed_params(raw_body, params)
    scan_observed = Any[]
    for (i, argument) in enumerate(arguments)
        if argument isa ExprColumn && getf(argument) === ragged
            value, group = _brm_kernel_ragged_args(argument; prefix="RK backend")
            value isa NamedColumn || error("RK backend: ragged input must name a column or predictor")
            (; into, order) = _brm_kernel_ragged_options(argument; prefix="RK backend")
            partition = _brm_kernel_ragged_rows(value, group, group_values;
                prefix="RK backend", order)
            if params[i] in scanned && parent(value) isa DataColumn
                # A response a scan observes is bound one series per subject
                # and read in place, as RKPPL observes it.
                column = _rk_kernel_scan_response(name, value, observed)
                columns[column] = _rk_gather_rows(parent(parent(value)), partition.rows)
                push!(inputs, (; source=column, kind=:element, rows=nothing, column))
                push!(scan_observed, (; param=params[i], source=column))
                continue
            end
            rows = Symbol(name, :_rows_, i)
            columns[rows] = partition.rows
            column = _rk_kernel_column_name(value)
            source = !isnothing(into) ? Symbol(name, :_index_, column) :
                column in observed ? Symbol(name, :_input_, column) : column
            if !isnothing(into)
                # Rebased positions under their own port: the raw row numbers
                # stay available to any other reader of the column.
                columns[source] = _brm_kernel_ragged_index_values(
                    value, group, into; prefix="RK backend")
            elseif parent(value) isa DataColumn
                columns[source] = parent(parent(value))
            end
            push!(inputs, (; source, kind=:gather, rows, column=nothing,
                partition=_rk_kernel_partition_source!(columns, value, group,
                    cell_column, order, partition.is_lp, column in observed)))
        elseif argument isa NamedColumn && parent(argument) isa DataColumn
            # A data input is read through its own column. Only a response,
            # whose column is flattened for the likelihood under its own name,
            # keeps its original nested values in a separate port; that port
            # is also the response's raw source (`_rk_kernel_input_port`).
            column = _rk_kernel_column_name(argument)
            scan = params[i] in scanned
            source = scan ? _rk_kernel_scan_response(name, argument, observed) :
                column in observed ? Symbol(name, :_input_, column) : column
            values = parent(parent(argument))
            values isa AbstractVector || error("RK backend: kernel input `$source` must be an array")
            columns[source] = values
            _rk_push_outer_length!(outer_lengths, values)
            push!(inputs, (; source, kind=:element, rows=nothing, column))
            scan && push!(scan_observed, (; param=params[i], source))
        elseif argument isa NamedColumn && parent(argument) isa ExprColumn
            push!(inputs, (; source=_rk_kernel_column_name(argument), kind=:element,
                rows=nothing, column=nothing))
        elseif argument isa NamedColumn && parent(argument) isa MissingColumn &&
                (params[i] in scanned || any(statement -> Meta.isexpr(statement, :call) &&
                    length(statement.args) == 3 && first(statement.args) === :~ &&
                    statement.args[2] === params[i], raw_body))
            # An unbound in-cell response is generated by Stan, but contributes
            # no RK likelihood. Keep its formal position so slices of the
            # deterministic cell can omit it without shifting later inputs.
            push!(inputs, (; source=_rk_kernel_column_name(argument), kind=:omitted,
                rows=nothing, column=nothing))
        else
            error("RK backend: kernel `$name` input must be a data column, predictor or ragged join")
        end
    end
    # The cell's subject plate iterates these inputs in lockstep. Without a
    # random-effect LP or a pre-grouped data input, the per-subject columns the
    # LPs read supply the count.
    group_values === nothing && isempty(outer_lengths) && foreach(
        lp -> append!(outer_lengths, last.(_brm_kernel_lp_data_lengths(lp))), direct_lps)
    nsubjects = group_values === nothing ?
        (isempty(outer_lengths) ? error("RK backend: kernel `$name` needs a subject input") : first(outer_lengths)) :
        length(group_values)
    all(==(nsubjects), outer_lengths) || error(
        "RK backend: kernel `$name` positional inputs disagree on subject count")
    for (lp, bucket) in zip(direct_lps, buckets)
        bucket === nothing && _brm_kernel_check_subject_frame_lp(
            lp, nsubjects; prefix="RK backend")
    end
    body = Any[]
    observations = Any[]
    # The observation statements the cell keeps; one whose response is
    # omitted reads nothing for the program.
    observed = Any[]
    collected = nothing
    for statement in raw_body
        statement isa LineNumberNode && continue
        if Meta.isexpr(statement, :call) && length(statement.args) == 3 &&
                first(statement.args) === :~
            lhs, distribution = statement.args[2:end]
            index = findfirst(==(lhs), params)
            index === nothing && error(
                "RK backend: kernel `$name` sampled cell declarations need a statistical submodel")
            argument = arguments[index]
            argument isa NamedColumn && parent(argument) isa MissingColumn && continue
            # The emitter calls the AR(1) submodel in the cell, which observes
            # the response in place (`scan_observed`).
            _is_brm_ar1_normal(statement) && (push!(body, statement); continue)
            source = argument isa NamedColumn ? _rk_kernel_column_name(argument) :
                (argument isa ExprColumn && getf(argument) === ragged ?
                    _rk_kernel_column_name(first(getargs(argument))) : nothing)
            source === nothing && error("RK backend: kernel `$name` observed cell needs a named response")
            if _rk_kernel_weighted(_rk_kernel_observation_callee(scope, distribution, name))
                # A weight may call a cell-local helper that `scope` does not bind.
                binding(head) = head isa Symbol && !isdefined(scope, head) &&
                    !isdefined(StanBlocks.stan, head) ? nothing : _rk_kernel_binding(scope, head)
                distribution = _brm_cell_typed_weight(binding,
                    distribution, "RK backend: kernel `$name`";
                    normal=c -> c === Normal || c === StanBlocks.normal)
            end
            distribution, weight = _rk_kernel_observation_weight(scope, distribution, name)
            distribution, modifier = _rk_kernel_observation_law(scope, distribution, name)
            callable = _rk_kernel_observation_family(scope, distribution, name)
            values = Tuple(distribution.args[2:end])
            any(value -> Meta.isexpr(value, :parameters), values) && error(
                "RK backend: kernel `$name` observation constructor keywords need explicit argument lowering")
            argument_names = Tuple(Symbol(name, :_argument_, source, :_, i)
                for i in eachindex(values))
            weight_name = weight === nothing ? nothing : Symbol(name, :_weight_, source)
            bounds = modifier === nothing ? () : Tuple((;
                name=Symbol(name, :_, label, :_, source), expression=bound, label)
                for (label, bound) in pairs(modifier) if label !== :kind && bound !== nothing)
            # A response holding one array per subject is observed per subject
            # (RKPPL nested plates).
            input = inputs[index]
            nested = input.kind === :gather || (input.kind === :element &&
                get(columns, input.source, nothing) isa AbstractVector{<:AbstractVector})
            push!(observations, (; source, param=lhs, callable,
                arguments=values, argument_names, weight, weight_name, nested,
                modifier, bounds, input))
            push!(observed, statement)
            continue
        end
        push!(body, statement)
        resultless && continue
        collected = Meta.isexpr(statement, :(=)) ? first(statement.args) :
            Meta.isexpr(statement, :return) ? only(statement.args) : statement
    end
    resultless || collected !== nothing || error("RK backend: kernel `$name` has no collected value")
    # The final value expression is returned once. An assignment stays in the
    # body and returns its newly bound name.
    !resultless && !isempty(body) && !Meta.isexpr(last(body), :(=)) && pop!(body)
    locals = Set{Symbol}(params)
    for statement in body
        _rk_source_outputs!(locals, statement)
    end
    # The names the cell reads for the program: an omitted response's
    # observation is not one, so a predictor only it reads is unreached (and a
    # shared `|ID|` block narrows to the margins observed likelihoods read).
    referenced = Set{Symbol}()
    foreach(statement -> _brm_cell_value_refs!(referenced, statement), [body; observed])
    collected === nothing || _brm_cell_value_refs!(referenced, collected)
    available = union(Set(keys(program.context.data)), Set(op.name for op in program.operations))
    globals = sort!(collect(intersect(setdiff(referenced, locals), available)))
    cells = _rk_ast_fresh_name(string(name, "_cells"), union(available, keys(columns)))
    _RKPreparedKernelAssignment(name, params, body, collected, scope, Tuple(inputs),
        globals, columns, group_values, Tuple(observations), cells, Tuple(scan_observed),
        cell_column === nothing ? nothing : _rk_kernel_column_name(cell_column))
end

# A formula observation of kernel output whose response holds one array per
# subject is observed per subject when every value its law and bounds read is
# either per subject (a kernel's cells, grouped data with one array per
# subject) or shared (a number, scalar data, a scalar parameter), through BRM's
# elementwise arithmetic and row-local callables. Returns the per-subject
# cells, or `nothing` to keep the flattened route: weights, missing entries,
# keyword laws and other callables.
function _rk_nested_kernel_cells(observation, layout, kernels, columns, parameters)
    layout.lengths === nothing && return nothing
    observation.weight === nothing && observation.missing_response === nothing ||
        return nothing
    modifier = observation.modifier
    distribution = modifier === nothing ? observation.distribution :
        _brm_prepare_expr(modifier.base)
    distribution isa _BRMPreparedExpr && isempty(distribution.kwargs) || return nothing
    cells = Dict{Symbol,Symbol}()
    context = (; kernels, columns, parameters, name=observation.name,
        lengths=layout.lengths)
    _rk_nested_law_cells!(cells, distribution.callable, distribution, context) || return nothing
    modifier === nothing && return cells
    all(bound -> bound === nothing ||
            _rk_nested_cells!(cells, _brm_prepare_expr(bound), context),
        (modifier.lower, modifier.upper)) || return nothing
    cells
end

# Data bounds read per subject are validated as the flattened route validates
# them, on their concatenated values. `response` holds one array of observed
# values per subject, in the bounds' row order.
# An absent replay response has no values to compare with its bounds; RKPPL
# still validates the bounds themselves.
_rk_validate_nested_bounds(observation, ::_BRMAbsentResponse, data) = nothing
function _rk_validate_nested_bounds(observation, response, data)
    modifier = observation.modifier
    modifier === nothing && return
    bounds = (modifier.lower, modifier.upper)
    all(b -> b === nothing || b isa Real || (b isa NamedColumn && parent(b) isa DataColumn),
        bounds) || return
    flat = Dict{Symbol,Any}(name(b) => brm_flatten_cells(parent(parent(b))) for b in bounds
        if b isa NamedColumn && parent(parent(b)) isa AbstractVector{<:AbstractVector})
    materialize = modifier.kind === :interval_censored ?
        _brm_materialize_interval_response : _brm_materialize_bounded_response
    materialize(modifier, observation.name, brm_flatten_cells(response),
        merge(Dict{Symbol,Any}(data), flat); prefix="RK backend")
    nothing
end

_rk_nested_law_cells!(cells, callable, distribution, context) =
    all(argument -> _rk_nested_cells!(cells, argument, context), distribution.args)
# Student-t reads its degrees of freedom through `TDist(nu)`.
function _rk_nested_law_cells!(cells, ::Type{LocationScale}, distribution, context)
    length(distribution.args) == 3 || return false
    location, scale, base = distribution.args
    base isa _BRMPreparedExpr && base.callable === TDist && length(base.args) == 1 &&
        all(argument -> _rk_nested_cells!(cells, argument, context),
            (location, scale, only(base.args)))
end

_rk_nested_cells!(cells, value, context) = false
_rk_nested_cells!(cells, value::Number, context) = true
function _rk_nested_cells!(cells, value::_BRMPreparedRef, context)
    index = findfirst(kernel -> kernel.name === value.name, context.kernels)
    if index !== nothing
        cells[value.name] = context.kernels[index].cells
        return true
    end
    data = get(context.columns, value.name, nothing)
    if data isa AbstractVector{<:AbstractVector}
        _rk_check_argument_groups(context.name, value.name, data, context.lengths)
        cells[value.name] = value.name
        return true
    end
    data isa Real || (value.axis === :scalar && value.name in context.parameters)
end
function _rk_nested_cells!(cells, value::_BRMPreparedExpr, context)
    callable = value.callable
    isempty(value.kwargs) && (haskey(_RK_DERIVED_BINOPS, callable) ||
        haskey(_RK_DERIVED_CMP, callable) || haskey(_RK_DERIVED_MATH, callable) ||
        _rk_rowwise_callable(callable)) &&
        all(argument -> _rk_nested_cells!(cells, argument, context), value.args)
end

# A callable whose output entry `k` depends only on entry `k` of its
# row-aligned inputs (and on shared values), so a subject cell may evaluate it
# on that subject's rows alone. Opt in by method (`addprop`, `src/sbimpl.jl`).
_rk_rowwise_callable(callable) = false

# The data columns observed by a likelihood: every authored response, and
# every kernel input this cell observes. Each is flattened under its own name.
# The cell parameters whose entries a `@scan` block observes (`y[1] ~ …`,
# `y[t] ~ …` with `y` a cell slice): RKPPL observes such a carried array in
# place, one series per subject (rkppl-use §5 "Observing a carried array").
function _rk_scan_observed_params(raw_body, params)
    observed = Symbol[]
    for statement in raw_body
        # `y ~ AR1Normal(...)` observes `y` through the AR(1) submodel's scan.
        if _is_brm_ar1_normal(statement)
            param = statement.args[2]
            param in params && !(param in observed) && push!(observed, param)
            continue
        end
        _is_brm_scan(statement) || continue
        for s in _brm_plate_flat_statements(Any[statement])
            isxcall(s, :~) && Meta.isexpr(s.args[2], :ref) || continue
            param = first(s.args[2].args)
            param in params && !(param in observed) && push!(observed, param)
        end
    end
    observed
end

# The `@rkppl` submodels `AR1Normal` calls in a cell (`ar1_normal.jl`): the
# exact AR(1) density over one series, given one mean, scale and persistence
# per step, at the recorded integer times or one step apart. The carried
# array they return is the response, which the call observes in place.
const _RK_AR1_NORMAL_SUBMODELS = (
    timed=:(brm_ar1_normal(mu, sigma, phi, time) = begin
        @scan begin
            y[1] ~ Normal(mu[1], sigma[1] / sqrt(1 - phi[1] ^ 2))
            for t in 2:length(time)
                k = time[t] - time[t - 1]
                y[t] ~ Normal(mu[t] + phi[t] ^ k * (y[t - 1] - mu[t - 1]),
                    sigma[t] * sqrt((1 - phi[t] ^ (2 * k)) / (1 - phi[t] ^ 2)))
            end
        end
        return y
    end),
    stepped=:(brm_ar1_normal_steps(mu, sigma, phi) = begin
        @scan begin
            y[1] ~ Normal(mu[1], sigma[1] / sqrt(1 - phi[1] ^ 2))
            for t in 2:length(mu)
                y[t] ~ Normal(mu[t] + phi[t] * (y[t - 1] - mu[t - 1]), sigma[t])
            end
        end
        return y
    end),
)

# A cell statement in the emitted program: its calls bound, and an
# `AR1Normal` observation calling the AR(1) submodel, added to the program's
# definitions once.
_rk_kernel_cell_statement(statement, kernel, defs, bindings, taken) =
    _is_brm_ar1_normal(statement) ?
        _rk_kernel_ar1_normal_call!(statement, kernel, defs, bindings, taken) :
        _rk_kernel_bind_calls(statement, kernel.scope, bindings, taken)

function _rk_kernel_ar1_normal_call!(statement, kernel, defs, bindings, taken)
    defs === nothing && error(
        "RK backend: internal: kernel `$(kernel.name)` emits an `AR1Normal` cell " *
        "without the program's definitions")
    spec = _brm_kernel_ar1_normal(statement, kernel.params, kernel.body)
    submodel = _rk_ar1_normal_submodel!(defs, taken,
        _RK_AR1_NORMAL_SUBMODELS[spec.time === nothing ? :stepped : :timed])
    arguments = Any[_brm_ar1_normal_series(spec, :fill)...]
    spec.time === nothing || push!(arguments, spec.time)
    bound = Any[_rk_kernel_bind_calls(argument, kernel.scope, bindings, taken)
        for argument in arguments]
    Expr(:call, :~, spec.response, Expr(:call, submodel, bound...))
end

# The name of a definition in `defs` with the template's signature and body,
# adding it under a name free in the program when absent.
function _rk_ar1_normal_submodel!(defs, taken, template)
    call, body = template.args
    for definition in defs
        Meta.isexpr(definition, :(=), 2) || continue
        signature = first(definition.args)
        Meta.isexpr(signature, :call) && isequal(signature.args[2:end], call.args[2:end]) &&
            isequal(last(definition.args), body) && return first(signature.args)
    end
    name = _rk_ast_fresh_name(string(first(call.args)), taken)
    push!(defs, Expr(:(=), Expr(:call, name, call.args[2:end]...), body))
    name
end

# A scan-observed response is conditioned under its own column name, which the
# cell reads in place, so nothing else may observe it.
function _rk_kernel_scan_response(kernel, value, observed)
    column = _rk_kernel_column_name(value)
    column in observed && error(
        "RK backend: kernel `$kernel` observes response `$column` in a `@scan` " *
        "block and again in another statement; observe each response once")
    column
end

function _rk_kernel_observed_columns(brmi, params, arguments, raw_body)
    observed = Set{Symbol}()
    for (_, node) in _brm_operation_entries(brmi)
        node isa NamedColumn && parent(node) isa ExprColumn{typeof(~)} || continue
        response = _brm_observation_name(first(getargs(parent(node))))
        response === nothing || push!(observed, response)
    end
    for statement in raw_body
        Meta.isexpr(statement, :call) && length(statement.args) == 3 &&
            first(statement.args) === :~ || continue
        # The AR(1) submodel observes its response in place (`scan_observed`).
        _is_brm_ar1_normal(statement) && continue
        index = findfirst(==(statement.args[2]), params)
        index === nothing && continue
        argument = arguments[index]
        argument isa NamedColumn && !(parent(argument) isa MissingColumn) &&
            push!(observed, _rk_kernel_column_name(argument))
        if argument isa ExprColumn && getf(argument) === ragged
            value = first(getargs(argument))
            value isa NamedColumn && parent(value) isa DataColumn &&
                push!(observed, _rk_kernel_column_name(value))
        end
    end
    observed
end

# The separate port a kernel reads a response's nested values from, if any.
function _rk_kernel_input_port(kernels, column)
    ports = unique(Symbol[input.source for kernel in kernels for input in kernel.inputs
        if input.kind === :element && input.column === column && input.source !== column])
    isempty(ports) ? nothing : first(ports)
end

# A kernel-observed response: its flat values, and its per-subject lengths
# when it holds one array per subject.
_rk_observed_layout(response) = (; values=response, lengths=nothing)
_rk_observed_layout(response::AbstractVector{<:AbstractVector}) =
    (; values=brm_flatten_cells(response), lengths=length.(response))
# An absent array-per-index response keeps the fitted nested route; each cell
# takes its shape from the data it reads, so it has no lengths to check.
_rk_observed_layout(response::_BRMAbsentResponse{<:AbstractVector}) =
    (; values=response, lengths=response)

# The per-subject lengths fix likelihood geometry; the flat values are produced
# from the original response port by the emitted numerical graph. A flat bound
# of a per-subject response is one group of rows. A kernel port already holding
# the raw response (`port`) is that source, so the nested values enter the
# graph once.
function _rk_prepare_kernel_observed_values!(columns, taken, derived, name, layout, raw;
        port=nothing)
    columns[name] = layout.values
    layout.lengths === nothing && return
    source = port !== nothing && isequal(get(columns, port, nothing), raw) ? port :
        _rk_ast_fresh_name(string(name, "_raw_response"), taken)
    columns[source] = raw
    expression = if raw isa AbstractVector{<:AbstractVector}
        Expr(:_rk_data_preparation, :brm_flatten_response, source)
    else
        groups = _rk_ast_fresh_name(string(name, "_rows"), taken)
        columns[groups] = [collect(eachindex(raw))]
        Expr(:_rk_data_preparation, :brm_gather_response, source, groups)
    end
    push!(derived, _RKDerivedSpec(name, expression, name))
    nothing
end

# The bound port whose groups fix a kernel-observed response's rows: its
# per-subject cells. Argument readers derive their row geometry from it, so new
# data of the same body rebinds it.
function _rk_observation_geometry_port(plan, observation)
    expressions = [spec.expression for spec in plan.regression.derived
        if spec.name === observation.name && spec.expression.head === :_rk_data_preparation]
    length(expressions) == 1 || error(
        "RK backend: internal: response `$(observation.name)` needs one row-geometry preparation")
    recipe, source = only(expressions).args
    recipe === :brm_flatten_response && return source
    error("RK backend: internal: response `$(observation.name)` row geometry " *
        "comes from unknown preparation `$recipe`")
end

# Only the likelihood receives these row views. Keep original data ports for
# kernels, readers and other responses, and perform the gather in printed RK
# source rather than replacing their bound values during preparation.
_rk_observation_argument_rows!(defs, statements, taken, observation, layout,
    geometry, argument, raw) = argument

# Grouped argument data broadcasts against its response subject by subject: one
# array per subject, each as long as the response's or a single value.
_rk_check_argument_groups(response, argument, raw, ::_BRMAbsentResponse) = nothing
function _rk_check_argument_groups(response, argument, raw, expected)
    lengths = length.(raw)
    length(lengths) == length(expected) &&
        all(pair -> first(pair) == last(pair) || first(pair) == 1,
            zip(lengths, expected)) || error(
        "RK backend: response `$response` argument `$argument` " *
        "has group lengths $lengths; expected $expected")
    nothing
end

function _rk_observation_argument_rows!(defs, statements, taken, observation, layout,
        geometry, argument, raw::AbstractVector{<:AbstractVector})
    lengths = length.(raw)
    _rk_check_argument_groups(observation.name, argument.name, raw, layout.lengths)
    name = _rk_ast_fresh_name("$(observation.name)_rows_$(argument.name)", taken)
    source = argument.name
    # Arguments as long as the response flatten in the argument kernel; a
    # singleton per subject repeats over its rows in a subject plate.
    if lengths == layout.lengths
        push!(statements, :($name = brm_flatten_cells($source)))
    else
        reader = _rk_ast_fresh_name("$(name)_reader", taken)
        push!(defs, :(ReactiveKernels.@kernel $reader(raw, groups) = begin
            cells = ReactiveKernels.plate(raw, groups) do value, rows
                ones(length(rows)) .* value
            end
            values = brm_flatten_cells(cells)
            return values
        end))
        push!(statements, Expr(:(=), name, Expr(:call, reader, source, geometry())))
    end
    _BRMPreparedRef(name, :whole)
end

_rk_align_observation_argument!(defs, statements, taken, columns, observation, layout,
    geometry, aligned, argument) = argument

function _rk_align_observation_argument!(defs, statements, taken, columns, observation,
        layout, geometry, aligned, argument::_BRMPreparedRef)
    argument.axis in (:observation, :observation_row) || return argument
    argument.name === observation.name && return argument
    get!(aligned, argument.name) do
        _rk_observation_argument_rows!(defs, statements, taken, observation, layout,
            geometry, argument, get(columns, argument.name, nothing))
    end
end

function _rk_align_observation_argument!(defs, statements, taken, columns, observation,
        layout, geometry, aligned, argument::_BRMPreparedExpr)
    # BRM arithmetic is elementwise. Whole-array reader calls retain their
    # original input axes and remain responsible for their returned row values.
    callable = argument.callable
    (haskey(_RK_DERIVED_BINOPS, callable) || haskey(_RK_DERIVED_CMP, callable) ||
        haskey(_RK_DERIVED_MATH, callable)) || return argument
    args = map(argument.args) do value
        _rk_align_observation_argument!(defs, statements, taken, columns, observation,
            layout, geometry, aligned, value)
    end
    _BRMPreparedExpr(callable, args, argument.kwargs)
end

function _rk_align_kernel_observation_arguments!(defs, statements, bindings, taken, plan,
        observation, distribution)
    kernels = Tuple(a for a in plan.assignments if a isa _RKPreparedKernelAssignment)
    isempty(kernels) && return distribution
    layout = _rk_observed_layout(observation.response)
    layout.lengths === nothing && return distribution
    # Prepare the constructor's arguments together in authored source. A
    # data-only model assignment is evaluated by RKPPL at binding; keeping
    # these operations in one argument kernel retains the complete source
    # graph beside the live location/scale and avoids that preprocessing path.
    inputs = Any[]
    params = Symbol[]
    columns = Dict{Symbol,Any}()
    refs = Dict{Symbol,_BRMPreparedRef}()
    function local_argument(argument)
        if argument isa _BRMPreparedExpr &&
                (haskey(_RK_DERIVED_BINOPS, argument.callable) ||
                 haskey(_RK_DERIVED_CMP, argument.callable) ||
                 haskey(_RK_DERIVED_MATH, argument.callable))
            return _BRMPreparedExpr(argument.callable,
                map(local_argument, argument.args), argument.kwargs)
        elseif argument isa _BRMPreparedRef || argument isa _BRMPreparedExpr
            argument isa _BRMPreparedRef && haskey(refs, argument.name) &&
                return refs[argument.name]
            param = _rk_ast_fresh_name("$(observation.name)_input_$(length(params) + 1)", taken)
            push!(params, param)
            push!(inputs, _rk_value_expr!(bindings, argument, taken))
            axis = argument isa _BRMPreparedRef ? argument.axis : :whole
            local_ref = _BRMPreparedRef(param, axis)
            if argument isa _BRMPreparedRef
                columns[param] = get(plan.columns, argument.name, nothing)
                argument.name === observation.name &&
                    (local_ref = _BRMPreparedRef(param, :whole))
                refs[argument.name] = local_ref
            end
            return local_ref
        end
        argument
    end
    local_args = map(local_argument, distribution.args)
    # The argument kernel's row geometry enters as one more input, on first use.
    geometry_param = nothing
    function geometry()
        geometry_param === nothing || return geometry_param
        geometry_param = _rk_ast_fresh_name(
            "$(observation.name)_input_$(length(params) + 1)", taken)
        push!(params, geometry_param)
        push!(inputs, _rk_observation_geometry_port(plan, observation))
        geometry_param
    end
    body = Any[]
    aligned = Dict{Symbol,_BRMPreparedRef}()
    args = map(local_args) do argument
        _rk_align_observation_argument!(defs, body, taken, columns,
            observation, layout, geometry, aligned, argument)
    end
    isempty(body) && return distribution
    prepared = map(args) do argument
        value = _rk_ast_fresh_name("$(observation.name)_prepared_argument", taken)
        push!(body, Expr(:(=), value, _rk_value_expr!(bindings, argument, taken)))
        value
    end
    reader = _rk_ast_fresh_name("$(observation.name)_observation_arguments", taken)
    returned = length(prepared) == 1 ? only(prepared) : Expr(:tuple, prepared...)
    definition = Expr(:(=), Expr(:call, reader, params...),
        Expr(:block, body..., Expr(:return, returned)))
    push!(defs, Expr(:macrocall,
        Expr(:., :ReactiveKernels, QuoteNode(Symbol("@kernel"))),
        LineNumberNode(0), definition))
    names = [_rk_ast_fresh_name("$(observation.name)_argument_$i", taken)
        for i in eachindex(args)]
    target = length(names) == 1 ? only(names) : Expr(:tuple, names...)
    push!(statements, Expr(:(=), target, Expr(:call, reader, inputs...)))
    outputs = map(name -> _BRMPreparedRef(name, :whole), names)
    _BRMPreparedExpr(distribution.callable, Tuple(outputs), distribution.kwargs)
end

function _rk_kernel_response_modifier!(columns, taken, derived, observation, layout)
    modifier = observation.modifier
    (modifier === nothing || layout.lengths === nothing) && return modifier
    function gather(bound, label)
        bound isa NamedColumn && parent(bound) isa DataColumn || return bound
        raw = parent(parent(bound))
        raw isa Real && return bound
        grouped = if raw isa AbstractVector{<:AbstractVector}
            length.(raw) == layout.lengths || error(
                "RK backend: response `$(observation.name)` $label bound `$(name(bound))` " *
                "has group lengths $(length.(raw)); expected $(layout.lengths)")
            convert(Vector{Float64}, brm_flatten_cells(raw))
        else
            raw isa AbstractVector{<:Real} || error(
                "RK backend: response `$(observation.name)` $label bound must be numeric")
            length(raw) == length(layout.values) || error(
                "RK backend: response `$(observation.name)` $label bound has " *
                "$(length(raw)) rows; expected $(length(layout.values))")
            collect(raw)
        end
        # Keep the flat bound available on its original axis for other formula
        # terms. Only this likelihood consumes the gathered bound column.
        key = _rk_ast_fresh_name(
            "$(observation.name)_$(label)_$(name(bound))_grouped", taken)
        bound_layout = (; values=grouped, lengths=layout.lengths)
        _rk_prepare_kernel_observed_values!(columns, taken, derived, key, bound_layout, raw)
        NamedColumn(key, DataColumn(grouped))
    end
    _BRMResponseModifierPlan(modifier.kind, modifier.base,
        gather(modifier.lower, :lower), gather(modifier.upper, :upper))
end

Base.@nospecializeinfer function _brm_rk_composed_kernel_plan(@nospecialize(brmi::BRMI))
    program = _brm_prepare_program(brmi;
        context=_brm_backend_context(brmi; retain_mm_sources=true))
    kernels = Tuple(_rk_prepare_kernel_value(brmi, program, name, rhs)
        for (name, rhs) in _rk_kernel_ops(brmi))
    direct = Any[(; key, lhs=getargs(parent(node))[1], rhs=getargs(parent(node))[2])
        for (key, node) in _brm_operation_entries(brmi) if node isa NamedColumn &&
            parent(node) isa ExprColumn{typeof(~)} &&
            _brm_observation_name(first(getargs(parent(node)))) !== nothing]
    submodels = _rk_prepare_submodel_values(program)
    plan = _brm_rk_value_plan(brmi, program, direct; kernels, submodels)
    observations = Any[plan.observations...]
    taken = union(Set{Symbol}(keys(plan.columns)),
        Set(spec.name for spec in plan.regression.derived),
        Set(assignment.name for assignment in plan.assignments),
        Set(kernel.cells for kernel in kernels))
    for kernel in kernels, observation in kernel.observations
        raw = program.context.data[observation.source]
        distribution = _rk_kernel_observation_distribution(observation)
        weight = observation.weight === nothing ? nothing :
            _BRMPreparedRef(observation.weight_name, :whole)
        if observation.nested
            # The response stays one array per subject. Each argument and the
            # weight is read per subject or shared, as the cell reads it.
            response = observation.input.kind === :gather ?
                _rk_gather_rows(raw, kernel.columns[observation.input.rows]) : raw
            _rk_validate_cell_bounds(kernel, observation, response, program.context.data)
            plan.columns[observation.source] = response
            routes = [_rk_kernel_cell_argument(kernel, name, argument) for (name, argument)
                in zip(observation.argument_names, observation.arguments)]
            bound_routes = [_rk_kernel_cell_argument(kernel, entry.name, entry.expression)
                for entry in observation.bounds]
            observation.weight === nothing || push!(routes,
                _rk_kernel_cell_argument(kernel, observation.weight_name, observation.weight))
            arguments = Tuple(route.ref for route in routes[eachindex(observation.arguments)])
            distribution = _rk_kernel_observation_distribution(observation, arguments)
            modifier = _rk_kernel_observation_modifier(observation, distribution,
                Tuple(route.ref for route in bound_routes))
            weight = observation.weight === nothing ? nothing :
                _BRMPreparedRef(last(routes).ref, :whole)
            prepared = _BRMPreparedObservation(observation.source,
                NamedColumn(observation.source, DataColumn(response)), distribution, response,
                modifier, weight)
            push!(observations, _RKNestedObservation(prepared, Dict{Symbol,Symbol}(
                route.ref => route.ref for route in (routes..., bound_routes...)
                if route.route in (:input, :plate))))
            continue
        end
        layout = raw isa AbstractVector{<:AbstractVector} ?
            (; values=brm_flatten_cells(raw), rows=nothing, lengths=length.(raw)) :
            (; values=raw, rows=nothing, lengths=nothing)
        _rk_prepare_kernel_observed_values!(plan.columns, taken, plan.regression.derived,
            observation.source, layout, raw;
            port=_rk_kernel_input_port((kernel,), observation.source))
        modifier = _rk_kernel_observation_modifier(observation, distribution,
            Tuple(entry.name for entry in observation.bounds))
        push!(observations, _BRMPreparedObservation(observation.source,
            NamedColumn(observation.source, DataColumn(plan.columns[observation.source])),
            distribution, raw, modifier, weight))
    end
    for kernel in kernels, entry in kernel.scan_observed
        push!(observations, _RKScanObservation(entry.source, plan.columns[entry.source]))
    end
    isempty(observations) && error("RK backend: kernel program needs at least one observed likelihood")
    _RKValuePlan(plan.regression, plan.assignments, Tuple(observations), plan.columns,
        plan.completions)
end

_rk_kernel_inv_logit(x) = logistic(x)
_rk_kernel_inv_logit(x::AbstractArray) = logistic.(x)

# StanBlocks declares Stan's builtins as functions without Julia methods. A
# cell that calls one through its StanBlocks binding runs its Julia equivalent
# on RK (the table is in the ReactiveKernels extension); every other callable
# runs as bound. `_rk_kernel_function_arguments` names the positional
# arguments a builtin takes as functions.
_rk_kernel_stan_builtin(callable) = callable
_rk_kernel_function_arguments(callable) = ()
_rk_kernel_named(value) = value isa Symbol || value isa GlobalRef || Meta.isexpr(value, :.)

# Stan broadcasts scalar addition/subtraction over vectors and matrices.
# Julia's undotted operators require matching array operands, so bind those
# Stan cases at the cell edge while leaving all other dispatch unchanged.
# Unary `+`/`-` mean the same in both languages and keep their authored
# spelling; a splatted call may still reach the one-operand methods.
_rk_kernel_stan_add(x) = +x
_rk_kernel_stan_add(x, y) = x + y
_rk_kernel_stan_add(x::Number, y::AbstractArray) = x .+ y
_rk_kernel_stan_add(x::AbstractArray, y::Number) = x .+ y
_rk_kernel_stan_add(x, y, z, rest...) =
    _rk_kernel_stan_add(_rk_kernel_stan_add(x, y), z, rest...)
_rk_kernel_stan_sub(x) = -x
_rk_kernel_stan_sub(x, y) = x - y
_rk_kernel_stan_sub(x::Number, y::AbstractArray) = x .- y
_rk_kernel_stan_sub(x::AbstractArray, y::Number) = x .- y
_rk_kernel_stan_sub(x, y, z, rest...) =
    _rk_kernel_stan_sub(_rk_kernel_stan_sub(x, y), z, rest...)

# One plain operand: `-x`, or the broadcast `(-).(x)`.
function _rk_kernel_unary_call(value)
    operands = value.head === :call ? value.args[2:end] : value.args[2].args
    length(operands) == 1 && !Meta.isexpr(only(operands), :...)
end

# Indices and range endpoints are integer arithmetic: Stan's scalar-array
# broadcasting does not apply there, and RKPPL recognises a scan's lag reads
# (`h[t - 1]`) and loop starts by their authored operators. `index` marks
# such a position.
function _rk_kernel_bind_calls(value, scope, bindings, taken; index=false)
    value isa Expr || return value
    integer = value.head === :ref || (value.head === :call && first(value.args) === :(:))
    args = Any[_rk_kernel_bind_calls(arg, scope, bindings, taken;
        index=index || (integer && i > 1)) for (i, arg) in enumerate(value.args)]
    # A broadcast `f.(args...)` carries its callee in the same first slot.
    if value.head === :call || _brm_is_broadcast_call(value)
        head = first(value.args)
        if head === :rep_vector
            args[1] = :fill
        elseif head === :+ && !index && !_rk_kernel_unary_call(value)
            args[1] = _rk_value_callee!(bindings, _rk_kernel_stan_add, taken)
        elseif head === :- && !index && !_rk_kernel_unary_call(value)
            args[1] = _rk_value_callee!(bindings, _rk_kernel_stan_sub, taken)
        elseif head === :inv_logit && !isdefined(scope, head)
            # Stan's vectorized primitive need not be defined in the cell's
            # Julia scope. Bind its scalar and array Julia equivalent.
            args[1] = _rk_value_callee!(bindings, _rk_kernel_inv_logit, taken)
        elseif head isa Symbol && (Base.isoperator(head) ||
                (startswith(string(head), ".") &&
                 Base.isoperator(Symbol(string(head)[2:end]))))
            args[1] = head
        elseif head isa Symbol && isdefined(Base, head) &&
                isdefined(scope, head) && getfield(scope, head) === getfield(Base, head)
            args[1] = head
        elseif _rk_kernel_named(head)
            callable = _rk_kernel_binding(scope, head)
            args[1] = _rk_value_callee!(bindings, _rk_kernel_stan_builtin(callable), taken)
            # A function passed by name (an ODE right-hand side) is bound as a
            # callable, not read as a value of the program.
            for position in _rk_kernel_function_arguments(callable)
                position + 1 <= length(args) &&
                    !any(arg -> Meta.isexpr(arg, :...), args[2:position + 1]) &&
                    _rk_kernel_named(args[position + 1]) || continue
                args[position + 1] = _rk_value_callee!(bindings,
                    _rk_kernel_binding(scope, args[position + 1]), taken)
            end
        end
    end
    Expr(value.head, args...)
end

# The authored cell, once, as an RKPPL `@plate for` in the model body. Its
# statements are sliced to what the program reads from it, and each such value
# is one indexed write `<cells>[i] = value`, so every subject's cell runs once
# per evaluation however many values the program reads. Each per-subject input
# is read at the subject index (`t[i]`; a ragged join gathers its subject's
# rows, `x[rows[i]]`), and the model values the cell reads stay shared.
#
# The plate iterates the authored range input unless that is a live value
# beside a bound data input: then it iterates the data input, the same subjects
# in the same order, because RK prepares a cell's data-only work once only when
# bound data fix the plate's domain (rkppl-use §13 "Arrays per index").
function _rk_emit_kernel_plate!(statements, bindings, taken, kernel, outputs;
        rendered=(), defs=nothing)
    # `rendered` outputs are already emitted program expressions (hoisted
    # observation arguments): their calls are bound, not cell source.
    written = Any[outputs..., rendered...]
    # A response a scan observes is observed in the cell itself.
    scanned = Any[entry.param for entry in kernel.scan_observed]
    # Nothing written or observed per subject (a cell without a collected
    # value whose observations read only shared values): there is no cell to
    # run.
    isempty(written) && isempty(scanned) && return
    slice = _rk_kernel_value_slice(kernel, Expr(:tuple, last.(written)..., scanned...))
    any(input -> input.kind === :omitted, slice.inputs) && error(
        "RK backend: kernel `$(kernel.name)` reads an unbound in-cell response outside its observation")
    cell = Any[[_rk_kernel_cell_statement(statement, slice, defs, bindings, taken)
        for statement in slice.body];
        [Expr(:(=), cells, _rk_kernel_bind_calls(value, kernel.scope, bindings, taken))
            for (cells, value) in outputs];
        [Expr(:(=), cells, value) for (cells, value) in rendered]]
    bound(input) = input.kind === :gather || haskey(kernel.columns, input.source)
    authored = first(input for input in kernel.inputs if input.kind !== :omitted)
    data = filter(bound, Any[slice.inputs..., kernel.inputs...])
    axis = bound(authored) || isempty(data) ? authored : first(data)
    inputs = Dict(zip(slice.params, slice.inputs))
    # A ragged join computes its row partition in the graph from bound raw
    # columns, and the plate then iterates the bound cell label column. A join
    # without a cell label column (a `groups(g)` plate) keeps its partition as
    # a bound input.
    graph_partition(input) = input.kind === :gather &&
        get(input, :partition, nothing) !== nothing
    for input in values(inputs)
        graph_partition(input) && _rk_emit_kernel_partition!(statements, input)
    end
    range = axis.kind !== :gather ? axis.source :
        graph_partition(axis) ? axis.partition.cell : axis.rows
    # Plate-scoped cell locals keep their names unless a model name, which the
    # plate also reads or writes, already spells them.
    locals = Set{Symbol}()
    foreach(statement -> _rk_source_outputs!(locals, statement), slice.body)
    setdiff!(locals, slice.params)
    model = union!(Set{Symbol}(), taken, _rk_source_symbols!(Set{Symbol}(), statements),
        (input.kind === :gather ? input.rows : input.source for input in values(inputs)),
        (input.source for input in values(inputs)), slice.globals, first.(written), (range,))
    cell_names = _rk_source_symbols!(Set{Symbol}(slice.params), cell)
    avoid = union(model, cell_names)
    reads = Dict{Symbol,Any}(l => _rk_ast_fresh_name(string(l, "_cell"), avoid)
        for l in sort!(collect(intersect(locals, model))))
    union!(taken, (get(reads, l, l) for l in locals))
    index = :i
    while index in avoid
        index = Symbol(index, :_)
    end
    for (param, input) in inputs
        reads[param] = input.kind === :gather ?
            Expr(:ref, input.source, Expr(:ref, input.rows, index)) :
            Expr(:ref, input.source, index)
    end
    body = Any[_rk_kernel_cell_reads(statement, reads) for statement in cell]
    for (k, (cells, _)) in enumerate(written)
        body[end - length(written) + k].args[1] = Expr(:ref, cells, index)
    end
    push!(statements, Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
        Expr(:for, Expr(:(=), index, Expr(:call, :eachindex, range)),
            Expr(:block, body...))))
end

# `<rows> = brm_ragged_rows([x, group, cell, axes...], group, cell, axes...)`
# (`brm_ragged_rows_ordered` with `order=`), once per program. A partition
# without a graph source stays a planned input.
function _rk_emit_kernel_partition!(statements, input)
    source = get(input, :partition, nothing)
    source === nothing && return
    any(s -> Meta.isexpr(s, :(=)) && first(s.args) === input.rows, statements) && return
    ordered = source.order !== nothing
    names = Symbol[source.x, source.group, source.cell,
        (ordered ? (source.order,) : ())..., source.axes...]
    push!(statements, Expr(:(=), input.rows, Expr(:call,
        ordered ? :brm_ragged_rows_ordered : :brm_ragged_rows,
        Expr(:vect, QuoteNode.(names)...), source.group, source.cell,
        (ordered ? (source.order,) : ())..., source.axes...)))
end

# A cell statement as the plate reads it: each name in `reads` (a per-subject
# formal, a renamed local) is replaced in value positions only. Call heads,
# field and keyword names, and names an inner function or generator binds keep
# their meaning.
_rk_kernel_cell_reads(value, reads) = value
_rk_kernel_cell_reads(value::Symbol, reads) = get(reads, value, value)
function _rk_kernel_cell_reads(value::Expr, reads)
    recur(arg) = _rk_kernel_cell_reads(arg, reads)
    args = value.args
    if value.head === :call || _brm_is_broadcast_call(value)
        return Expr(value.head, first(args), map(recur, args[2:end])...)
    elseif value.head === :. && length(args) == 2 && last(args) isa QuoteNode
        return Expr(:., recur(first(args)), last(args))
    elseif value.head === :kw
        return Expr(:kw, first(args), recur(last(args)))
    elseif value.head === :->
        bound = _rk_source_symbols!(Set{Symbol}(), first(args))
        inner = filter(pair -> !(first(pair) in bound), reads)
        return Expr(:->, first(args), _rk_kernel_cell_reads(last(args), inner))
    elseif value.head === :generator
        specs = args[2:end]
        bound = Set{Symbol}()
        for spec in specs, s in (Meta.isexpr(spec, :filter) ? spec.args[2:end] : (spec,))
            Meta.isexpr(s, :(=)) && _rk_source_symbols!(bound, first(s.args))
        end
        inner = filter(pair -> !(first(pair) in bound), reads)
        ranges(spec) = Meta.isexpr(spec, :(=)) ?
            Expr(:(=), first(spec.args), recur(last(spec.args))) :
            Meta.isexpr(spec, :filter) ? Expr(:filter,
                _rk_kernel_cell_reads(first(spec.args), inner), map(ranges, spec.args[2:end])...) :
            spec
        return Expr(:generator, _rk_kernel_cell_reads(first(args), inner), map(ranges, specs)...)
    end
    Expr(value.head, map(recur, args)...)
end

# How a per-subject observation reads one in-cell argument (`name` holds its
# value): a model value or whole-model expression the cell reads is shared by
# every subject; a per-subject input read as is stays its bound port; anything
# else is written per subject by the cell's plate.
function _rk_kernel_cell_argument(kernel, name, argument)
    slice = _rk_kernel_value_slice(kernel, argument)
    if isempty(slice.params) && isempty(slice.body)
        return (; route=argument isa Symbol ? :shared : :value,
            ref=argument isa Symbol ? argument : name)
    end
    if argument isa Symbol && isempty(slice.body) && only(slice.inputs).kind === :element
        return (; route=:input, ref=only(slice.inputs).source)
    end
    (; route=:plate, ref=name)
end

# The kernel's value is its subject cells; formula terms read them
# concatenated, a per-subject observation reads them cell by cell. The cell's
# plate also writes every in-cell observation argument the observations read
# per subject: a per-subject observation reads it cell by cell, a flattened one
# reads one value per response row.
function _rk_emit_value_assignment!(defs, statements, bindings, taken,
        kernel::_RKPreparedKernelAssignment; arguments=nothing)
    arguments === nothing || (kernel = _rk_kernel_with_arguments(kernel, arguments))
    # A cell without a collected value writes only what its observations read.
    outputs, after = kernel.collected === nothing ? (Any[], Any[]) : (
        Any[(kernel.cells, kernel.collected)],
        Any[Expr(:(=), kernel.name, Expr(:call, :brm_flatten_cells, kernel.cells))])
    for observation in kernel.observations
        named = Any[zip(observation.argument_names, observation.arguments)...]
        append!(named, ((entry.name, entry.expression) for entry in observation.bounds))
        observation.weight === nothing ||
            push!(named, (observation.weight_name, observation.weight))
        for (name, argument) in named
            if observation.nested
                # Each cell broadcasts against its own response array.
                route = _rk_kernel_cell_argument(kernel, name, argument)
                if route.route === :plate
                    push!(outputs, (name, argument))
                elseif route.route === :value
                    push!(after, Expr(:(=), name,
                        _rk_kernel_bind_calls(argument, kernel.scope, bindings, taken)))
                end
                continue
            end
            cells = _rk_ast_fresh_name(string(name, "_cells"), taken)
            push!(outputs, (cells, Expr(:call, :.*,
                Expr(:call, :ones, Expr(:call, :length, observation.param)), argument)))
            push!(after, Expr(:(=), name, Expr(:call, :brm_flatten_cells, cells)))
        end
    end
    _rk_emit_kernel_plate!(statements, bindings, taken, kernel, outputs;
        rendered=arguments === nothing ? () : arguments.outputs, defs)
    append!(statements, after)
end

# A per-subject observation reads its kernel's cells entry by entry, and RK's
# cell query runs the kernel's plate at one subject only while every reader of
# those cells reads them at that subject. RKPPL lowers a law argument computed
# from per-subject values (`addprop(pred, a, b)`, `pred - offset`) as a
# separate plate that reads them whole, so such an argument is computed in the
# kernel's plate instead, as an in-cell observation's arguments are, and the
# observation reads it cell by cell. In the plate the kernel's own value is
# bound once, and each per-subject data column the argument reads is one more
# input read at the subject. An argument reading several kernels, or a shared
# name the cell also binds, keeps its place in the observation.
function _rk_nested_kernel_arguments(plan, bindings, taken)
    kernels = Dict{Symbol,_RKPreparedKernelAssignment}(a.name => a
        for a in plan.assignments if a isa _RKPreparedKernelAssignment)
    homes = Dict{Symbol,Any}()
    reads = Dict{Symbol,Dict{Int,Symbol}}()
    for observation in plan.observations
        observation isa _RKNestedObservation || continue
        law = _rk_observation_law(observation)
        law isa _BRMPreparedExpr || continue
        owners = sort!([n for n in keys(observation.cells) if haskey(kernels, n)])
        isempty(owners) && continue
        positions = law.callable === LocationScale ? (1, 2) : eachindex(law.args)
        for k in positions
            argument = law.args[k]
            argument isa _BRMPreparedExpr || continue
            names = _rk_prepared_refs!(Set{Symbol}(), argument)
            owned = filter(n -> haskey(kernels, n), names)
            data = filter(n -> haskey(observation.cells, n) && !haskey(kernels, n), names)
            isempty(owned) && isempty(data) && continue
            length(owned) > 1 && continue
            home = isempty(owned) ? first(owners) : only(owned)
            kernel = kernels[home]
            state = get!(homes, home) do
                locals = Set{Symbol}(kernel.params)
                foreach(statement -> _rk_source_outputs!(locals, statement), kernel.body)
                value = kernel.collected isa Symbol && kernel.collected in locals ?
                    kernel.collected :
                    _rk_ast_fresh_name(string(home, "_value"), union(taken, locals))
                (; value, locals, params=Symbol[], inputs=Any[], outputs=Any[],
                    sources=Dict{Symbol,Symbol}())
            end
            isdisjoint(setdiff(names, owned, data), state.locals) || continue
            renames = Dict{Symbol,Any}(home => state.value)
            for n in data
                source = observation.cells[n]
                renames[n] = get!(state.sources, source) do
                    param = _rk_ast_fresh_name(string(source, "_cell"),
                        union(taken, state.locals, state.params))
                    push!(state.params, param)
                    push!(state.inputs, (; source, kind=:element, rows=nothing, column=nothing))
                    param
                end
            end
            name = _rk_ast_fresh_name(string(home, "_argument_", observation.name, "_", k), taken)
            push!(state.outputs, (name, _rk_kernel_cell_reads(
                _rk_value_expr!(bindings, argument, taken), renames)))
            get!(reads, observation.name, Dict{Int,Symbol}())[k] = name
        end
    end
    (; kernels=homes, reads)
end

# The law a value-plan observation is emitted from (a bounded response's base
# law), and the names its prepared values read.
_rk_observation_law(observation) = observation.modifier === nothing ?
    observation.distribution : _brm_prepare_expr(observation.modifier.base)
_rk_prepared_refs!(names, value) = names
_rk_prepared_refs!(names, value::_BRMPreparedRef) = push!(names, value.name)
function _rk_prepared_refs!(names, value::_BRMPreparedExpr)
    foreach(arg -> _rk_prepared_refs!(names, arg), value.args)
    foreach(arg -> _rk_prepared_refs!(names, arg), values(value.kwargs))
    names
end

# The observation as it reads its hoisted arguments: each by its kernel output,
# cell by cell.
function _rk_nested_observation_reads(observation, distribution, reads)
    hoisted = get(reads, observation.name, nothing)
    hoisted === nothing && return observation, distribution
    args = Tuple(haskey(hoisted, k) ? _BRMPreparedRef(hoisted[k], :whole) : argument
        for (k, argument) in enumerate(distribution.args))
    cells = merge(observation.cells, Dict{Symbol,Symbol}(n => n for n in values(hoisted)))
    _RKNestedObservation(observation.observation, cells),
        _BRMPreparedExpr(distribution.callable, args, distribution.kwargs)
end

# The program names a kernel depends on, for the plan's reachability: its own
# dependencies except the positional sources only an omitted response's
# observation reads (a source another kept statement reads stays one).
function _rk_kernel_dependencies(kernel::_RKPreparedKernelAssignment, dependencies)
    reads = Set{Symbol}()
    foreach(statement -> _brm_cell_value_refs!(reads, statement), kernel.body)
    kernel.collected === nothing || _brm_cell_value_refs!(reads, kernel.collected)
    for observation in kernel.observations
        push!(reads, observation.param)
        foreach(argument -> _brm_cell_value_refs!(reads, argument), observation.arguments)
        observation.weight === nothing || _brm_cell_value_refs!(reads, observation.weight)
        foreach(bound -> _brm_cell_value_refs!(reads, bound.expression), observation.bounds)
    end
    unread = Set{Symbol}(input.source for (param, input) in zip(kernel.params, kernel.inputs)
        if !(param in reads))
    setdiff!(unread, reads)
    union(setdiff(Set{Symbol}(dependencies), unread), kernel.globals)
end

# The kernel with its value bound to the plate local the hoisted arguments
# read, and the per-subject data they read as more inputs.
function _rk_kernel_with_arguments(kernel::_RKPreparedKernelAssignment, arguments)
    body = kernel.collected === arguments.value ? kernel.body :
        Any[kernel.body..., Expr(:(=), arguments.value, kernel.collected)]
    _RKPreparedKernelAssignment(kernel.name, [kernel.params; arguments.params], body,
        arguments.value, kernel.scope, (kernel.inputs..., arguments.inputs...),
        kernel.globals, kernel.columns, kernel.group_values, kernel.observations,
        kernel.cells, kernel.scan_observed, kernel.cell_column)
end

# The cell restricted to what `value` reads: the body statements producing
# its names, and the positional inputs and globals those statements read.
# Only an assignment whose every recognized output is unread is left out; any
# other statement is kept with everything it reads.
function _rk_kernel_value_slice(kernel::_RKPreparedKernelAssignment, value)
    needed = _brm_cell_value_refs!(Set{Symbol}(), value)
    body = Any[]
    for statement in Iterators.reverse(kernel.body)
        outputs = Meta.isexpr(statement, :(=)) ?
            _rk_source_outputs!(Set{Symbol}(), statement) :
            _is_brm_scan(statement) ? _rk_scan_writes(statement) : Set{Symbol}()
        !isempty(outputs) && isdisjoint(outputs, needed) && continue
        pushfirst!(body, statement)
        _brm_cell_value_refs!(needed, statement)
    end
    kept = [i for (i, param) in enumerate(kernel.params) if param in needed]
    _RKPreparedKernelAssignment(kernel.name, kernel.params[kept], body, value,
        kernel.scope, kernel.inputs[kept], filter(in(needed), kernel.globals),
        kernel.columns, kernel.group_values, (), kernel.cells, (), kernel.cell_column)
end

# The arrays and step locals a `@scan` block writes, and the responses it
# observes (`y[t] ~ …` writes `y`'s entries).
function _rk_scan_writes(scan)
    names = Set{Symbol}()
    for statement in _brm_plate_flat_statements(Any[scan])
        Meta.isexpr(statement, :(=), 2) || isxcall(statement, :~) || continue
        lhs = Meta.isexpr(statement, :(=)) ? first(statement.args) : statement.args[2]
        Meta.isexpr(lhs, :ref) && (lhs = first(lhs.args))
        lhs isa Symbol && push!(names, lhs)
    end
    names
end
