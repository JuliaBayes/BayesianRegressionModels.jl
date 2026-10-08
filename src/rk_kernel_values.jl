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
end
_rk_kernel_column_name(column::NamedColumn) = name(column)

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
        "observations are not lowered on the RK backend yet; observe the response " *
        "with an unbounded in-cell family, or supply the bounded law through " *
        "`_rk_observation_source!` for a caller-owned family")
    callable === StanBlocks.normal && return Normal
    callable
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

function _rk_kernel_observation_distribution(observation)
    arguments = map(name -> _BRMPreparedRef(name, :whole), observation.argument_names)
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
            if parent(value) isa DataColumn
                columns[_rk_kernel_column_name(value)] = parent(parent(value))
            end
            push!(inputs, (; source=_rk_kernel_column_name(value), kind=:gather, rows,
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
        else
            error("RK backend: kernel `$name` input must be a data column, predictor or ragged join")
        end
    end
    # The reader's subject plate iterates these inputs in lockstep.
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
            source = argument isa NamedColumn ? _rk_kernel_column_name(argument) :
                (argument isa ExprColumn && getf(argument) === ragged ?
                    _rk_kernel_column_name(first(getargs(argument))) : nothing)
            source === nothing && error("RK backend: kernel `$name` observed cell needs a named response")
            distribution, weight = _rk_kernel_observation_weight(scope, distribution, name)
            callable = _rk_kernel_observation_family(scope, distribution, name)
            values = Tuple(distribution.args[2:end])
            any(value -> Meta.isexpr(value, :parameters), values) && error(
                "RK backend: kernel `$name` observation constructor keywords need explicit argument lowering")
            argument_names = Tuple(Symbol(name, :_argument_, source, :_, i)
                for i in eachindex(values))
            weight_name = weight === nothing ? nothing : Symbol(name, :_weight_, source)
            push!(observations, (; source, param=lhs, callable,
                arguments=values, argument_names, weight, weight_name))
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
    _RKPreparedKernelAssignment(name, params, body, collected, scope, Tuple(inputs),
        globals, columns, group_values, Tuple(observations))
end

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
        argument isa NamedColumn && push!(observed, _rk_kernel_column_name(argument))
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

function _rk_observation_argument_rows!(defs, statements, taken, observation, layout,
        geometry, argument, raw::AbstractVector{<:AbstractVector})
    lengths = length.(raw)
    length(lengths) == length(layout.lengths) &&
        all(pair -> first(pair) == last(pair) || first(pair) == 1,
            zip(lengths, layout.lengths)) || error(
        "RK backend: response `$(observation.name)` argument `$(argument.name)` " *
        "has group lengths $(length.(raw)); expected $(layout.lengths)")
    name = _rk_ast_fresh_name("$(observation.name)_rows_$(argument.name)", taken)
    source = argument.name
    reader = _rk_ast_fresh_name("$(name)_reader", taken)
    if lengths == layout.lengths
        push!(defs, :(ReactiveKernels.@kernel $reader(raw) = begin
            values = brm_flatten_cells(raw)
            return values
        end))
        push!(statements, :($name = $reader($source)))
    else
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
    reader = _rk_ast_fresh_name("$(name)_reader", taken)
    push!(defs, :(ReactiveKernels.@kernel $reader(raw, groups) = begin
        values = raw[brm_flatten_cells(groups)]
        return values
    end))
    push!(statements, Expr(:(=), name, Expr(:call, reader, argument.name, geometry())))
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
    # these operations in the argument readers retains the complete source
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
    # The readers' row geometry enters as one more input, on first use.
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
    outputs = map(eachindex(args)) do i
        reader = _rk_ast_fresh_name("$(observation.name)_observation_argument_$i", taken)
        definition = Expr(:(=), Expr(:call, reader, params...),
            Expr(:block, body..., Expr(:return, prepared[i])))
        push!(defs, Expr(:macrocall,
            Expr(:., :ReactiveKernels, QuoteNode(Symbol("@kernel"))),
            LineNumberNode(0), definition))
        value = _rk_ast_fresh_name("$(observation.name)_argument_$i", taken)
        push!(statements, Expr(:(=), value, Expr(:call, reader, inputs...)))
        _BRMPreparedRef(value, :whole)
    end
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
        Set(assignment.name for assignment in plan.assignments))
    for kernel in kernels, observation in kernel.observations
        raw = program.context.data[observation.source]
        layout = raw isa AbstractVector{<:AbstractVector} ?
            (; values=brm_flatten_cells(raw), rows=nothing, lengths=length.(raw)) :
            (; values=raw, rows=nothing, lengths=nothing)
        _rk_prepare_kernel_observed_values!(plan.columns, taken, plan.regression.derived,
            observation.source, layout, raw;
            port=_rk_kernel_input_port((kernel,), observation.source))
        distribution = _rk_kernel_observation_distribution(observation)
        weight = observation.weight === nothing ? nothing :
            _BRMPreparedRef(observation.weight_name, :whole)
        push!(observations, _BRMPreparedObservation(observation.source,
            NamedColumn(observation.source, DataColumn(plan.columns[observation.source])),
            distribution, raw, nothing, weight))
    end
    isempty(observations) && error("RK backend: kernel program needs at least one observed likelihood")
    _RKValuePlan(plan.regression, plan.assignments, Tuple(observations), plan.columns,
        plan.completions)
end

function _rk_kernel_bind_calls(value, scope, bindings, taken)
    value isa Expr || return value
    args = map(arg -> _rk_kernel_bind_calls(arg, scope, bindings, taken), value.args)
    # A broadcast `f.(args...)` carries its callee in the same first slot.
    if value.head === :call || _brm_is_broadcast_call(value)
        head = first(value.args)
        if head === :rep_vector
            args[1] = :fill
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

# The reader is the authored cell, sliced to what its value reads, as one
# subject plate. Data inputs are plate operands; when a cell also reads a live
# vector, a bound data axis indexes that vector from an atomic operand. A
# ragged join iterates its subject row indices and gathers from its shared
# values; model values read as globals stay shared. The subject cells are then
# flattened in subject order. A value reading only data stays data-only.
function _rk_emit_kernel_reader!(defs, statements, bindings, taken, kernel, name, collected)
    reader_name = _rk_ast_fresh_name(string(name, "_reader"), taken)
    slice = _rk_kernel_value_slice(kernel, collected)
    cell = Any[[_rk_kernel_bind_calls(statement, kernel.scope, bindings, taken)
        for statement in slice.body];
        _rk_kernel_bind_calls(collected, kernel.scope, bindings, taken)]
    inputs = collect(zip(slice.params, slice.inputs))
    # A cell reading no per-subject input still iterates one for its subject axis.
    isempty(inputs) && push!(inputs, (first(kernel.params), first(kernel.inputs)))
    # A live vector in a zip plate can change its domain after binding, which
    # prevents preparation of the cell's data-only recipes. A bound input fixes the
    # subject axis; read live elements by index from atomic operands instead.
    data_axis = findfirst(inputs) do (_, input)
        input.kind === :gather || haskey(kernel.columns, input.source)
    end
    indexed = data_axis !== nothing && any(inputs) do (_, input)
        input.kind === :element && !haskey(kernel.columns, input.source)
    end
    globals = slice.globals
    gathered = unique(Symbol[input.source for (_, input) in inputs if input.kind === :gather])
    ports = unique(Symbol[[input.kind === :gather ? input.rows : input.source
        for (_, input) in inputs]; gathered; globals])
    # The cell runs inline in the plate's closure. Every name the reader adds
    # avoids the cell's own names, so the cell neither captures nor rebinds a
    # reader local. Globals keep their names because the cell reads them so.
    cell_names = _rk_source_symbols!(Set{Symbol}(kernel.params), cell)
    used = union(cell_names, ports)
    fresh(base) = _rk_ast_fresh_name(string(base), used)
    rows = Dict(param => fresh("$(param)_rows") for (param, input) in inputs
        if input.kind === :gather)
    shared = Dict(source => source in cell_names ? fresh("$(source)_values") : source
        for source in gathered)
    subject = indexed ? fresh("$(name)_subject") : nothing
    live_values = Dict(param => fresh("$(param)_values") for (param, input) in inputs
        if indexed && input.kind === :element && !haskey(kernel.columns, input.source))
    formals = Symbol[[input.kind === :gather ? rows[param] :
        get(live_values, param, param) for (param, input) in inputs];
        [shared[source] for source in gathered]; globals]
    # A reader port no formal shadows stays clear of the cell's names; a
    # renamed shared source keeps one name as port and formal.
    port = Dict(p => haskey(shared, p) ? shared[p] :
        p in cell_names && !(p in formals) ? fresh("$(p)_port") : p for p in ports)
    operands = Any[[haskey(live_values, param) ?
        Expr(:call, :Ref, port[input.source]) :
        port[input.kind === :gather ? input.rows : input.source]
        for (param, input) in inputs];
        [Expr(:call, :Ref, port[source]) for source in [gathered; globals]]]
    if indexed
        axis = last(inputs[data_axis])
        axis_port = port[axis.kind === :gather ? axis.rows : axis.source]
        pushfirst!(operands, Expr(:call, :(Base.eachindex), axis_port))
        pushfirst!(formals, subject)
    end
    indexed_values = [Expr(:(=), param, Expr(:ref, live_values[param], subject))
        for (param, _) in inputs if haskey(live_values, param)]
    gathers = [Expr(:(=), param, Expr(:ref, shared[input.source], rows[param]))
        for (param, input) in inputs if input.kind === :gather]
    plate = Expr(:do, Expr(:call, Expr(:., :ReactiveKernels, QuoteNode(:plate)), operands...),
        Expr(:->, Expr(:tuple, formals...), Expr(:block, indexed_values..., gathers..., cell...)))
    cells, values = fresh(:cells), fresh(:values)
    body = Expr(:block,
        Expr(:(=), cells, plate),
        Expr(:(=), values, Expr(:call, :brm_flatten_cells, cells)),
        Expr(:return, values))
    reader_definition = Expr(:(=), Expr(:call, reader_name, (port[p] for p in ports)...), body)
    push!(defs, Expr(:macrocall,
        Expr(:., :ReactiveKernels, QuoteNode(Symbol("@kernel"))),
        LineNumberNode(0), reader_definition))
    push!(statements, Expr(:(=), name, Expr(:call, reader_name, ports...)))
end

function _rk_emit_value_assignment!(defs, statements, bindings, taken,
        kernel::_RKPreparedKernelAssignment)
    _rk_emit_kernel_reader!(defs, statements, bindings, taken, kernel,
        kernel.name, kernel.collected)
    for observation in kernel.observations
        vectorize(value) = Expr(:call, :.*, Expr(:call, :ones,
            Expr(:call, :length, observation.param)), value)
        for (name, argument) in zip(observation.argument_names, observation.arguments)
            _rk_emit_kernel_reader!(defs, statements, bindings, taken, kernel,
                name, vectorize(argument))
        end
        # A data weight's reader reads only data, so it stays a data-only
        # definition for RKPPL's `weighted`.
        observation.weight === nothing && continue
        _rk_emit_kernel_reader!(defs, statements, bindings, taken, kernel,
            observation.weight_name, vectorize(observation.weight))
    end
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
        kernel.columns, kernel.group_values, ())
end
