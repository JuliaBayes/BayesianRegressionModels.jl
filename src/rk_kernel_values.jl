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
    count::Symbol
    columns::Dict{Symbol,Any}
    group_values::Any
    observations::Tuple
end
_rk_kernel_column_name(column::NamedColumn) = name(column)
_rk_flatten_kernel_response(cells) = reduce(vcat, cells; init=eltype(eltype(cells))[])

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
            push!(inputs, (; source=_rk_kernel_column_name(value), kind=:gather, rows))
        elseif argument isa NamedColumn && parent(argument) isa DataColumn
            # A separate port preserves the original nested input when its
            # observation counterpart is flattened for the likelihood.
            source = Symbol(name, :_input_, _rk_kernel_column_name(argument))
            values = parent(parent(argument))
            values isa AbstractVector || error("RK backend: kernel input `$source` must be an array")
            columns[source] = values
            push!(outer_lengths, length(values))
            push!(inputs, (; source, kind=:element, rows=nothing))
        elseif argument isa NamedColumn && parent(argument) isa ExprColumn
            push!(inputs, (; source=_rk_kernel_column_name(argument), kind=:element, rows=nothing))
        else
            error("RK backend: kernel `$name` input must be a data column, predictor or ragged join")
        end
    end
    count = Symbol(name, :_subject_count)
    nsubjects = group_values === nothing ?
        (isempty(outer_lengths) ? error("RK backend: kernel `$name` needs a subject input") : first(outer_lengths)) :
        length(group_values)
    all(==(nsubjects), outer_lengths) || error(
        "RK backend: kernel `$name` positional inputs disagree on subject count")
    columns[count] = nsubjects
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
        globals, count, columns, group_values, Tuple(observations))
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
        values = reduce(vcat, (raw[rows] for rows in partition.rows); init=eltype(raw)[])
        return (; values, rows=partition.rows, lengths=length.(partition.rows))
    end
    response = observation.response
    if response isa AbstractVector{<:AbstractVector}
        return (; values=_rk_flatten_kernel_response(response),
            rows=nothing, lengths=length.(response))
    end
    (; values=response, rows=nothing, lengths=nothing)
end

# The partition fixes likelihood geometry; its values are produced from the
# original response port by the emitted numerical graph. A ragged join's row
# partition (one group of response rows per kernel subject) is data, so it is
# a bound port like a kernel's ragged input rows, never a literal in source.
function _rk_prepare_kernel_observed_values!(columns, taken, derived, name, layout, raw)
    columns[name] = layout.values
    layout.lengths === nothing && return
    source = _rk_ast_fresh_name(string(name, "_raw_response"), taken)
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
            values = reduce(vcat, raw; init=eltype(eltype(raw))[])
            return values
        end))
        push!(statements, :($name = $reader($source)))
    else
        push!(defs, :(ReactiveKernels.@kernel $reader(raw, groups) = begin
            cells = ReactiveKernels.plate(eachindex(groups), Ref(raw), Ref(groups)) do group, raw, groups
                ones(length(groups[group])) .* raw[group]
            end
            values = reduce(vcat, cells; init=Float64[])
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
        values = raw[reduce(vcat, groups; init=Int[])]
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
            reduce(vcat, raw; init=Float64[])
        else
            raw isa AbstractVector{<:Real} || error(
                "RK backend: response `$(observation.name)` $label bound must be numeric")
            length(raw) == length(layout.values) || error(
                "RK backend: response `$(observation.name)` $label bound has " *
                "$(length(raw)) rows; expected $(length(layout.values))")
            layout.rows === nothing ? collect(raw) :
                reduce(vcat, (raw[rows] for rows in layout.rows); init=eltype(raw)[])
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
            (; values=_rk_flatten_kernel_response(raw), rows=nothing, lengths=length.(raw)) :
            (; values=raw, rows=nothing, lengths=nothing)
        _rk_prepare_kernel_observed_values!(plan.columns, taken, plan.regression.derived,
            observation.source, layout, raw)
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

function _rk_emit_kernel_reader!(defs, statements, bindings, taken, kernel, name, collected)
    source_names = unique(Symbol[kernel.count;
        [input.source for input in kernel.inputs];
        [input.rows for input in kernel.inputs if input.rows !== nothing]; kernel.globals])
    cell_name = _rk_ast_fresh_name(string(name, "_cell"), taken)
    reader_name = _rk_ast_fresh_name(string(name, "_reader"), taken)
    cell_params = [kernel.params; kernel.globals]
    cell_body = [_rk_kernel_bind_calls(statement, kernel.scope, bindings, taken)
        for statement in kernel.body]
    push!(cell_body, Expr(:return,
        _rk_kernel_bind_calls(collected, kernel.scope, bindings, taken)))
    cell_definition = Expr(:(=), Expr(:call, cell_name, cell_params...),
        Expr(:block, cell_body...))
    push!(defs, Expr(:macrocall,
        Expr(:., :ReactiveKernels, QuoteNode(Symbol("@kernel"))),
        LineNumberNode(0), cell_definition))
    subject = _rk_ast_fresh_name("subject", Set(source_names))
    plate_names = Set([source_names; subject])
    cell_args = [_rk_ast_fresh_name("cell_input_$i", plate_names)
        for i in eachindex(kernel.inputs)]
    cell_inputs = [Expr(:(=), argument, input.kind === :gather ?
        Expr(:ref, input.source, Expr(:ref, input.rows, subject)) :
        Expr(:ref, input.source, subject))
        for (argument, input) in zip(cell_args, kernel.inputs)]
    call = Expr(:call, cell_name, cell_args..., kernel.globals...)
    # The subject plate and its child cell are authored graph recipes. A
    # downstream @kernel entry's scan remains inside this retained child.
    sequence = Expr(:call, :(:), 1, kernel.count)
    plate_args = [Expr(:call, :Ref, source) for source in source_names]
    plate = Expr(:do, Expr(:call,
        Expr(:., :ReactiveKernels, QuoteNode(:plate)), sequence, plate_args...),
        Expr(:->, Expr(:tuple, subject, source_names...),
            Expr(:block, cell_inputs..., call)))
    reader_names = Set(source_names)
    values = _rk_ast_fresh_name("cell_values", reader_names)
    result = _rk_ast_fresh_name("result", reader_names)
    body = Expr(:block,
        Expr(:(=), values, plate),
        :($result = convert(Vector{Float64}, reduce(vcat, $values; init=Float64[]))),
        Expr(:return, result))
    reader_definition = Expr(:(=), Expr(:call, reader_name, source_names...), body)
    push!(defs, Expr(:macrocall,
        Expr(:., :ReactiveKernels, QuoteNode(Symbol("@kernel"))),
        LineNumberNode(0), reader_definition))
    push!(statements, Expr(:(=), name, Expr(:call, reader_name, source_names...)))
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
        # The weight reader reads only what the weight depends on, so a data
        # weight stays a data-only definition for RKPPL's `weighted`.
        observation.weight === nothing && continue
        weight = vectorize(observation.weight)
        _rk_emit_kernel_reader!(defs, statements, bindings, taken,
            _rk_kernel_value_slice(kernel, weight), observation.weight_name, weight)
    end
end

# The cell restricted to what `value` reads: the body statements producing
# its names, and the positional inputs and globals those statements read.
function _rk_kernel_value_slice(kernel::_RKPreparedKernelAssignment, value)
    needed = _brm_cell_value_refs!(Set{Symbol}(), value)
    body = Any[]
    for statement in Iterators.reverse(kernel.body)
        isdisjoint(_rk_source_outputs!(Set{Symbol}(), statement), needed) && continue
        pushfirst!(body, statement)
        _brm_cell_value_refs!(needed, statement)
    end
    kept = [i for (i, param) in enumerate(kernel.params) if param in needed]
    _RKPreparedKernelAssignment(kernel.name, kernel.params[kept], body, value,
        kernel.scope, kernel.inputs[kept], filter(in(needed), kernel.globals),
        kernel.count, kernel.columns, kernel.group_values, ())
end
