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
        x.cells)
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
    # SLIC's unqualified built-in family tokens need not be exported into
    # the caller module. A caller's actual binding still takes precedence.
    head isa Symbol && !isdefined(scope, head) && isdefined(StanBlocks.stan, head) ?
        getfield(StanBlocks.stan, head) : Core.eval(scope, head)
end

# In-cell distribution combinators take a family token as their first
# positional (`weighted(normal, w, mu, sigma)`), never a data argument.
_rk_kernel_weighted(callable) = callable === weighted || callable === StanBlocks.stan.weighted
_rk_kernel_bounded(callable) = callable in (censored, truncated, interval_censored,
    StanBlocks.stan.censored, StanBlocks.stan.truncated, StanBlocks.stan.interval_censored)

function _rk_kernel_observation_family(scope, expression, kernel)
    callable = _rk_kernel_observation_callee(scope, expression, kernel)
    _rk_kernel_weighted(callable) && error(
        "RK backend: kernel `$kernel` observation nests `weighted` in its family; " *
        "write one `weighted(family, weight, args...)`")
    _rk_kernel_bounded(callable) && error(
        "RK backend: kernel `$kernel` in-cell `$(nameof(callable))(family, ...)` " *
        "supports `censored(Normal(...); lower=..., upper=...)` only; " *
        "use `_rk_observation_source!` for other bounded laws")
    callable === StanBlocks.normal && return Normal
    callable
end

# A cell writes the same censored Normal law as a formula response, but its
# bounds may read cell locals and ragged inputs. Keep those expressions beside
# the constructor arguments so the authored plate computes them on its axis.
function _rk_kernel_observation_law(scope, expression, kernel)
    callable = _rk_kernel_observation_callee(scope, expression, kernel)
    callable in (censored, StanBlocks.stan.censored) ||
        return expression, nothing
    args = expression.args[2:end]
    parameters = filter(arg -> Meta.isexpr(arg, :parameters), args)
    length(parameters) <= 1 || error(
        "RK backend: kernel `$kernel` `censored` has repeated keyword groups")
    bounds = Dict{Symbol,Any}()
    for keyword in (isempty(parameters) ? () : only(parameters).args)
        Meta.isexpr(keyword, :kw) && first(keyword.args) in (:lower, :upper) || error(
            "RK backend: kernel `$kernel` `censored` accepts only `lower` and `upper` bounds")
        haskey(bounds, first(keyword.args)) && error(
            "RK backend: kernel `$kernel` `censored` repeats `$(first(keyword.args))`")
        bounds[first(keyword.args)] = last(keyword.args)
    end
    positional = filter(arg -> !Meta.isexpr(arg, :parameters), args)
    isempty(positional) && error(
        "RK backend: kernel `$kernel` `censored` needs a Normal family")
    family = first(positional)
    base = if Meta.isexpr(family, :call) && length(positional) == 1
        family
    elseif (family isa Symbol || family isa GlobalRef || Meta.isexpr(family, :.)) &&
            length(positional) > 1
        Expr(:call, positional...)
    else
        error("RK backend: kernel `$kernel` `censored` needs a family call or token")
    end
    _rk_kernel_observation_family(scope, base, kernel) === Normal || error(
        "RK backend: kernel `$kernel` in-cell `censored` currently supports Normal only")
    lower = get(bounds, :lower, nothing)
    upper = get(bounds, :upper, nothing)
    lower === :nothing && (lower = nothing)
    upper === :nothing && (upper = nothing)
    lower === nothing && upper === nothing && error(
        "RK backend: kernel `$kernel` `censored` needs a non-nothing bound")
    base, (; kind=:censored, lower, upper)
end

# SLIC's observation weighting `weighted(family, weight, args...)` is a power
# likelihood: each row's `family(args...)` log density is scaled by its
# weight. StanBlocks' call form `weighted(family(args...), weight, extra...)`
# splices the family call's arguments after the remaining positionals, so it
# names the same observation as `weighted(family, weight, extra..., args...)`.
function _rk_kernel_observation_weight(scope, expression, kernel)
    _rk_kernel_weighted(_rk_kernel_observation_callee(scope, expression, kernel)) ||
        return expression, nothing
    positional(args) = any(arg -> Meta.isexpr(arg, (:parameters, :kw)), args) && error(
        "RK backend: kernel `$kernel` `weighted(family, weight, args...)` " *
        "takes positional arguments only")
    args = expression.args[2:end]
    positional(args)
    length(args) >= 2 || error(
        "RK backend: kernel `$kernel` observation needs `weighted(family, weight, args...)`")
    family, weight, rest = args[1], args[2], args[3:end]
    Meta.isexpr(family, :call) || return Expr(:call, family, rest...), weight
    positional(family.args[2:end])
    Expr(:call, first(family.args), rest..., family.args[2:end]...), weight
end

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

Base.@nospecializeinfer function _rk_prepare_kernel_value(@nospecialize(brmi::BRMI), program, name, rhs)
    parts = _sb_kernel_lambda_parts(first(getargs(rhs)))
    parts === nothing && error("RK backend: kernel `$name` requires an inline cell body")
    params, raw_body = parts
    arguments = getargs(rhs)[2:end]
    length(params) == length(arguments) || error(
        "RK backend: kernel `$name` cell and positional argument counts disagree")
    isempty(getkwargs(rhs)) || error("RK backend: kernel `$name` has unsupported control keywords")
    scope = _brm_inline_scope(first(getargs(rhs)))
    direct_lps = [argument for argument in arguments
        if argument isa NamedColumn && parent(argument) isa ExprColumn]
    groups = [_sb_kernel_lp_bucket(lp) for lp in direct_lps]
    group_values = if isempty(groups)
        nothing
    else
        length(unique(group[2] for group in groups)) == 1 || error(
            "RK backend: kernel `$name` predictor inputs disagree on subject grouping")
        column = first(groups)[3]
        _brm_kernel_subject_values(parent(parent(column)), _rk_kernel_column_name(column); prefix="RK backend")
    end
    columns = Dict{Symbol,Any}()
    inputs = Any[]
    outer_lengths = Int[]
    observed = _rk_kernel_observed_columns(brmi, params, arguments, raw_body)
    for (i, argument) in enumerate(arguments)
        if argument isa ExprColumn && getf(argument) === ragged
            group_values === nothing && error(
                "RK backend: kernel `$name` ragged inputs require a subject predictor")
            length(getargs(argument)) == 2 || error("RK backend: ragged input needs value and group")
            value, group = getargs(argument)
            value isa NamedColumn || error("RK backend: ragged input must name a column or predictor")
            partition = _brm_kernel_ragged_rows(value, group, group_values; prefix="RK backend")
            rows = Symbol(name, :_rows_, i)
            columns[rows] = partition.rows
            column = _rk_kernel_column_name(value)
            source = column in observed ? Symbol(name, :_input_, column) : column
            if parent(value) isa DataColumn
                columns[source] = parent(parent(value))
            end
            push!(inputs, (; source, kind=:gather, rows,
                column=nothing))
        elseif argument isa NamedColumn && parent(argument) isa DataColumn
            # A data input is read through its own column. Only a response,
            # whose column is flattened for the likelihood under its own name,
            # keeps its original nested values in a separate port; that port
            # is also the response's raw source (`_rk_kernel_input_port`).
            column = _rk_kernel_column_name(argument)
            source = column in observed ? Symbol(name, :_input_, column) : column
            values = parent(parent(argument))
            values isa AbstractVector || error("RK backend: kernel input `$source` must be an array")
            columns[source] = values
            push!(outer_lengths, length(values))
            push!(inputs, (; source, kind=:element, rows=nothing, column))
        elseif argument isa NamedColumn && parent(argument) isa ExprColumn
            push!(inputs, (; source=_rk_kernel_column_name(argument), kind=:element,
                rows=nothing, column=nothing))
        elseif argument isa NamedColumn && parent(argument) isa MissingColumn &&
                any(statement -> Meta.isexpr(statement, :call) &&
                    length(statement.args) == 3 && first(statement.args) === :~ &&
                    statement.args[2] === params[i], raw_body)
            # An unbound in-cell response is generated by Stan, but contributes
            # no RK likelihood. Keep its formal position so slices of the
            # deterministic cell can omit it without shifting later inputs.
            push!(inputs, (; source=_rk_kernel_column_name(argument), kind=:omitted,
                rows=nothing, column=nothing))
        else
            error("RK backend: kernel `$name` input must be a data column, predictor or ragged join")
        end
    end
    # The cell's subject plate iterates these inputs in lockstep.
    nsubjects = group_values === nothing ?
        (isempty(outer_lengths) ? error("RK backend: kernel `$name` needs a subject input") : first(outer_lengths)) :
        length(group_values)
    all(==(nsubjects), outer_lengths) || error(
        "RK backend: kernel `$name` positional inputs disagree on subject count")
    body = Any[]
    observations = Any[]
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
            source = argument isa NamedColumn ? _rk_kernel_column_name(argument) :
                (argument isa ExprColumn && getf(argument) === ragged ?
                    _rk_kernel_column_name(first(getargs(argument))) : nothing)
            source === nothing && error("RK backend: kernel `$name` observed cell needs a named response")
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
            continue
        end
        push!(body, statement)
        collected = Meta.isexpr(statement, :(=)) ? first(statement.args) :
            Meta.isexpr(statement, :return) ? only(statement.args) : statement
    end
    collected === nothing && error("RK backend: kernel `$name` has no collected value")
    # The final value expression is returned once. An assignment stays in the
    # body and returns its newly bound name.
    !isempty(body) && !Meta.isexpr(last(body), :(=)) && pop!(body)
    locals = Set{Symbol}(params)
    for statement in body
        _rk_source_outputs!(locals, statement)
    end
    referenced = Set{Symbol}()
    foreach(statement -> _brm_cell_value_refs!(referenced, statement), raw_body)
    available = union(Set(keys(program.context.data)), Set(op.name for op in program.operations))
    globals = sort!(collect(intersect(setdiff(referenced, locals), available)))
    cells = _rk_ast_fresh_name(string(name, "_cells"), union(available, keys(columns)))
    _RKPreparedKernelAssignment(name, params, body, collected, scope, Tuple(inputs),
        globals, columns, group_values, Tuple(observations), cells)
end

# A formula observation of kernel output whose response holds one array per
# subject is observed per subject when every value its law and bounds read is
# either per subject (a kernel's cells, grouped data with one array per
# subject) or shared (a number, scalar data, a scalar parameter), through BRM's
# elementwise arithmetic and row-local callables. A ragged join (`ragged(y, g)`
# over one flat column) is observed per subject too: its response, and every
# data column with one value per joined row, are grouped by the join's row
# partition (`gathered`). Returns the per-subject cells and those gathered
# columns, or `nothing` to keep the flattened route: weights, missing entries,
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
    gathered = Set{Symbol}()
    context = (; kernels, columns, parameters, name=observation.name,
        lengths=layout.lengths, joined=layout.rows === nothing ? nothing :
            length(layout.values), gathered)
    _rk_nested_law_cells!(cells, distribution.callable, distribution, context) || return nothing
    modifier === nothing && return (; cells, gathered)
    all(bound -> bound === nothing ||
            _rk_nested_cells!(cells, _brm_prepare_expr(bound), context),
        (modifier.lower, modifier.upper)) || return nothing
    (; cells, gathered)
end

# Bind a ragged join's per-subject cells: the response and each gathered
# column are grouped by the join's row partition, as data. The partition is
# BRM's own preparation of the authored rows, so new data rebinds through the
# same planning; RKPPL observes a bounded (censored, truncated) response only
# from a bound column, never from a derived definition.
function _rk_prepare_nested_join!(columns, taken, observation, layout, nested, raw)
    gather(values) = [values[rows] for rows in layout.rows]
    columns[observation.name] = gather(raw)
    for name in sort!(collect(nested.gathered))
        key = _rk_ast_fresh_name(string(observation.name, "_cells_", name), taken)
        columns[key] = gather(columns[name])
        nested.cells[name] = key
    end
    nested.cells
end

# Data bounds read per subject are validated as the flattened route validates
# them, on their concatenated values. `response` holds the observed values in
# the bounds' row order: one array per subject, or a join's flat column.
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
    # One value per joined row: grouped by the join's partition.
    if context.joined !== nothing && data isa AbstractVector && length(data) != 1 &&
            length(data) == context.joined
        push!(context.gathered, value.name)
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

function _rk_kernel_observed_layout(observation, kernels)
    lhs = observation.lhs
    if lhs isa ExprColumn && getf(lhs) === ragged
        value, group = getargs(lhs)
        matches = [kernel for kernel in kernels if kernel.group_values !== nothing &&
            kernel.name in _brm_prepared_references(observation.distribution)]
        length(matches) == 1 || error(
            "RK backend: response `$(observation.name)` needs one kernel subject axis for its ragged join")
        partition = _brm_kernel_ragged_rows(value, group, only(matches).group_values; prefix="RK backend")
        raw = parent(parent(value))
        values = raw[brm_flatten_cells(partition.rows)]
        return (; values, rows=partition.rows, lengths=length.(partition.rows))
    end
    response = observation.response
    if response isa AbstractVector{<:AbstractVector}
        return (; values=brm_flatten_cells(response),
            rows=nothing, lengths=length.(response))
    end
    (; values=response, rows=nothing, lengths=nothing)
end

# The partition fixes likelihood geometry; its values are produced from the
# original response port by the emitted numerical graph. A ragged join's row
# partition (one group of response rows per kernel subject) is data, so it is
# a bound port like a kernel's ragged input rows, never a literal in source.
# A kernel port already holding the raw response (`port`) is that source, so
# the nested values enter the graph once.
function _rk_prepare_kernel_observed_values!(columns, taken, derived, name, layout, raw;
        port=nothing)
    columns[name] = layout.values
    layout.lengths === nothing && return
    source = port !== nothing && isequal(get(columns, port, nothing), raw) ? port :
        _rk_ast_fresh_name(string(name, "_raw_response"), taken)
    columns[source] = raw
    expression = if layout.rows === nothing && raw isa AbstractVector{<:AbstractVector}
        Expr(:_rk_data_preparation, :brm_flatten_response, source)
    else
        groups = _rk_ast_fresh_name(string(name, "_rows"), taken)
        columns[groups] = layout.rows === nothing ? [collect(eachindex(raw))] : layout.rows
        Expr(:_rk_data_preparation, :brm_gather_response, source, groups)
    end
    push!(derived, _RKDerivedSpec(name, expression, name))
    nothing
end

# The bound port whose groups fix a kernel-observed response's rows: its
# per-subject cells, or its ragged-join partition. Argument readers derive
# their row geometry from it, so new data of the same body rebinds it.
function _rk_observation_geometry_port(plan, observation)
    expressions = [spec.expression for spec in plan.regression.derived
        if spec.name === observation.name && spec.expression.head === :_rk_data_preparation]
    length(expressions) == 1 || error(
        "RK backend: internal: response `$(observation.name)` needs one row-geometry preparation")
    recipe, source, groups... = only(expressions).args
    recipe === :brm_flatten_response && return source
    recipe === :brm_gather_response && return only(groups)
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

function _rk_observation_argument_rows!(defs, statements, taken, observation, layout,
        geometry, argument, raw::AbstractVector)
    layout.rows === nothing && return argument
    length(raw) == 1 && return argument
    length(raw) == length(layout.values) || error(
        "RK backend: response `$(observation.name)` argument `$(argument.name)` " *
        "has $(length(raw)) rows; expected $(length(layout.values))")
    name = _rk_ast_fresh_name("$(observation.name)_rows_$(argument.name)", taken)
    # The original response join partitions these rows; the bound partition
    # port orders them and the argument gather stays in the graph.
    push!(statements, Expr(:(=), name, Expr(:ref, argument.name,
        Expr(:call, :brm_flatten_cells, geometry()))))
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
    layout = _rk_kernel_observed_layout(observation, kernels)
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
            layout.rows === nothing ? collect(raw) :
                raw[brm_flatten_cells(layout.rows)]
        end
        # Keep the flat bound available on its original axis for other formula
        # terms. Only this likelihood consumes the gathered bound column.
        key = _rk_ast_fresh_name(
            "$(observation.name)_$(label)_$(name(bound))_grouped", taken)
        bound_layout = (; values=grouped,
            rows=raw isa AbstractVector{<:AbstractVector} ? nothing : layout.rows,
            lengths=layout.lengths)
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
                [raw[rows] for rows in kernel.columns[observation.input.rows]] : raw
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
    isempty(observations) && error("RK backend: kernel program needs at least one observed likelihood")
    _RKValuePlan(plan.regression, plan.assignments, Tuple(observations), plan.columns,
        plan.completions)
end

_rk_kernel_inv_logit(x) = logistic(x)
_rk_kernel_inv_logit(x::AbstractArray) = logistic.(x)

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

function _rk_kernel_bind_calls(value, scope, bindings, taken)
    value isa Expr || return value
    args = map(arg -> _rk_kernel_bind_calls(arg, scope, bindings, taken), value.args)
    # A broadcast `f.(args...)` carries its callee in the same first slot.
    if value.head === :call || _brm_is_broadcast_call(value)
        head = first(value.args)
        if head === :rep_vector
            args[1] = :fill
        elseif head === :+ && !_rk_kernel_unary_call(value)
            args[1] = _rk_value_callee!(bindings, _rk_kernel_stan_add, taken)
        elseif head === :- && !_rk_kernel_unary_call(value)
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
        elseif head isa Symbol || head isa GlobalRef || Meta.isexpr(head, :.)
            callable = Core.eval(scope, head)
            args[1] = _rk_value_callee!(bindings, callable, taken)
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
        rendered=())
    # `rendered` outputs are already emitted program expressions (hoisted
    # observation arguments): their calls are bound, not cell source.
    written = Any[outputs..., rendered...]
    slice = _rk_kernel_value_slice(kernel, Expr(:tuple, last.(written)...))
    any(input -> input.kind === :omitted, slice.inputs) && error(
        "RK backend: kernel `$(kernel.name)` reads an unbound in-cell response outside its observation")
    cell = Any[[_rk_kernel_bind_calls(statement, kernel.scope, bindings, taken)
        for statement in slice.body];
        [Expr(:(=), cells, _rk_kernel_bind_calls(value, kernel.scope, bindings, taken))
            for (cells, value) in outputs];
        [Expr(:(=), cells, value) for (cells, value) in rendered]]
    bound(input) = input.kind === :gather || haskey(kernel.columns, input.source)
    authored = first(input for input in kernel.inputs if input.kind !== :omitted)
    data = filter(bound, Any[slice.inputs..., kernel.inputs...])
    axis = bound(authored) || isempty(data) ? authored : first(data)
    range = axis.kind === :gather ? axis.rows : axis.source
    inputs = Dict(zip(slice.params, slice.inputs))
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
    outputs = Any[(kernel.cells, kernel.collected)]
    after = Any[Expr(:(=), kernel.name, Expr(:call, :brm_flatten_cells, kernel.cells))]
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
        rendered=arguments === nothing ? () : arguments.outputs)
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

# The kernel with its value bound to the plate local the hoisted arguments
# read, and the per-subject data they read as more inputs.
function _rk_kernel_with_arguments(kernel::_RKPreparedKernelAssignment, arguments)
    body = kernel.collected === arguments.value ? kernel.body :
        Any[kernel.body..., Expr(:(=), arguments.value, kernel.collected)]
    _RKPreparedKernelAssignment(kernel.name, [kernel.params; arguments.params], body,
        arguments.value, kernel.scope, (kernel.inputs..., arguments.inputs...),
        kernel.globals, kernel.columns, kernel.group_values, kernel.observations,
        kernel.cells)
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
            _rk_source_outputs!(Set{Symbol}(), statement) : Set{Symbol}()
        !isempty(outputs) && isdisjoint(outputs, needed) && continue
        pushfirst!(body, statement)
        _brm_cell_value_refs!(needed, statement)
    end
    kept = [i for (i, param) in enumerate(kernel.params) if param in needed]
    _RKPreparedKernelAssignment(kernel.name, kernel.params[kept], body, value,
        kernel.scope, kernel.inputs[kept], filter(in(needed), kernel.globals),
        kernel.columns, kernel.group_values, (), kernel.cells)
end
