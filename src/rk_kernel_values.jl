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

function _rk_kernel_observation_family(scope, expression, kernel)
    Meta.isexpr(expression, :call) || error(
        "RK backend: kernel `$kernel` observation needs a constructor call")
    head = first(expression.args)
    (head isa Symbol || head isa GlobalRef || Meta.isexpr(head, :.)) || error(
        "RK backend: kernel `$kernel` observation needs a named constructor")
    # SLIC's unqualified built-in family tokens need not be exported into
    # the caller module. A caller's actual binding still takes precedence.
    callable = head isa Symbol && !isdefined(scope, head) &&
            isdefined(StanBlocks.stan, head) ?
        getfield(StanBlocks.stan, head) : Core.eval(scope, head)
    callable === StanBlocks.normal && return Normal
    callable
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

function _rk_kernel_value_refs!(names, value)
    value isa Symbol && (push!(names, value); return names)
    value isa Expr || return names
    if value.head in (:call, :macrocall)
        foreach(arg -> _rk_kernel_value_refs!(names, arg), value.args[2:end])
    elseif value.head in (:kw, :.)
        _rk_kernel_value_refs!(names, value.args[value.head === :kw ? 2 : 1])
    else
        foreach(arg -> _rk_kernel_value_refs!(names, arg), value.args)
    end
    names
end

function _rk_prepare_kernel_value(brmi, program, name, rhs)
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
            callable = _rk_kernel_observation_family(scope, distribution, name)
            values = Tuple(distribution.args[2:end])
            any(value -> Meta.isexpr(value, :parameters), values) && error(
                "RK backend: kernel `$name` observation constructor keywords need explicit argument lowering")
            argument_names = Tuple(Symbol(name, :_argument_, source, :_, i)
                for i in eachindex(values))
            push!(observations, (; source, param=lhs, callable,
                arguments=values, argument_names))
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
    foreach(statement -> _rk_kernel_value_refs!(referenced, statement), raw_body)
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

function _rk_kernel_response_modifier!(columns, taken, observation, layout)
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
        columns[key] = grouped
        NamedColumn(key, DataColumn(grouped))
    end
    _BRMResponseModifierPlan(modifier.kind, modifier.base,
        gather(modifier.lower, :lower), gather(modifier.upper, :upper))
end

function _brm_rk_composed_kernel_plan(brmi)
    program = _brm_prepare_program(brmi;
        context=_brm_backend_context(brmi; retain_mm_sources=true))
    kernels = Tuple(_rk_prepare_kernel_value(brmi, program, name, rhs)
        for (name, rhs) in _rk_kernel_ops(brmi))
    direct = Tuple((; key, lhs=getargs(parent(node))[1], rhs=getargs(parent(node))[2])
        for (key, node) in pairs(brmi.operations) if node isa NamedColumn &&
            parent(node) isa ExprColumn{typeof(~)} &&
            _brm_observation_name(first(getargs(parent(node)))) !== nothing)
    submodels = _rk_prepare_submodel_values(program)
    plan = _brm_rk_value_plan(brmi, program, direct; kernels, submodels)
    observations = Any[plan.observations...]
    for kernel in kernels, observation in kernel.observations
        raw = program.context.data[observation.source]
        plan.columns[observation.source] = raw isa AbstractVector{<:AbstractVector} ?
            _rk_flatten_kernel_response(raw) : raw
        distribution = _rk_kernel_observation_distribution(observation)
        push!(observations, _BRMPreparedObservation(observation.source,
            NamedColumn(observation.source, DataColumn(plan.columns[observation.source])),
            distribution, plan.columns[observation.source], nothing, nothing))
    end
    isempty(observations) && error("RK backend: kernel program needs at least one observed likelihood")
    _RKValuePlan(plan.regression, plan.assignments, Tuple(observations), plan.columns,
        plan.completions)
end

function _rk_kernel_bind_calls(value, scope, bindings, taken)
    value isa Expr || return value
    args = map(arg -> _rk_kernel_bind_calls(arg, scope, bindings, taken), value.args)
    if value.head === :call
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
    end
end
