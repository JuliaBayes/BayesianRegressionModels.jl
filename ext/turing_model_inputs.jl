# Bind prepared inputs as ordinary model arguments. This is an AST pass over
# static field/index accesses, never an evaluator for user expressions. Runtime
# row indexing, sampled variables and callable bodies stay in the model.
struct _BRMTuringASTContext
    callables::Vector{Any}
    row::Symbol
    data_sources::Dict{Symbol,Int}
end
Base.push!(context::_BRMTuringASTContext, value) = push!(context.callables, value)
Base.length(context::_BRMTuringASTContext) = length(context.callables)
Base.iterate(context::_BRMTuringASTContext, state...) = iterate(context.callables, state...)
_brm_row_symbol(context::_BRMTuringASTContext) = context.row

function _brm_reference_ast(name::Symbol, context::_BRMTuringASTContext)
    index = get(context.data_sources, name, nothing)
    isnothing(index) ? name :
        :(multi.plans[$index].context.data[$(QuoteNode(name))])
end

function _brm_model_binding_names(plans)
    names = Set{Symbol}()
    for plan in plans
        union!(names, keys(plan.context.data))
        union!(names, (p.name for p in plan.parameters))
        union!(names, (a.name for a in plan.assignments))
        union!(names, (p.predictor.name for p in plan.predictors))
    end
    names
end

function _brm_fresh_model_name(preferred, used)
    name, index = preferred, 1
    while name in used
        name = Symbol(preferred, :_, index)
        index += 1
    end
    name
end

function _brm_response_symbols(plans; single=false)
    used = _brm_model_binding_names(plans)
    setdiff!(used, (plan.response_name for plan in plans))
    names = Symbol[]
    for (index, plan) in enumerate(plans)
        preferred = single ? :y : Symbol(:y_, index)
        name = preferred in used ? _brm_fresh_model_name(plan.response_name, used) : preferred
        push!(used, name)
        push!(names, name)
    end
    names
end

function _brm_qualify_generated_calls!(node)
    node isa Expr || return node
    if node.head === :call && first(node.args) isa Symbol
        name = first(node.args)
        if name !== :~ && isdefined(@__MODULE__, name)
            node.args[1] = GlobalRef(Base.binding_module(@__MODULE__, name), name)
        end
    end
    foreach(_brm_qualify_generated_calls!, node.args)
    node
end

function _brm_input_path(node)
    node in (:multi, :callables) && return (node,)
    node isa Expr || return nothing
    if node.head === :. && length(node.args) == 2 && node.args[2] isa QuoteNode
        path = _brm_input_path(node.args[1])
        isnothing(path) && return nothing
        return (path..., (:property, node.args[2].value))
    elseif node.head === :ref
        path = _brm_input_path(first(node.args))
        isnothing(path) && return nothing
        indices = map(node.args[2:end]) do index
            index isa QuoteNode ? index.value : index
        end
        all(index -> index isa Integer || index isa AbstractString ||
                     (index isa Symbol && any(arg -> arg isa QuoteNode &&
                         arg.value === index, node.args[2:end])), indices) || return nothing
        return (path..., (:index, Tuple(indices)))
    end
    nothing
end

function _brm_input_value(path, multi, callables)
    value = first(path) === :multi ? multi : callables
    for (kind, key) in Base.tail(path)
        value = kind === :property ? getproperty(value, key) : getindex(value, key...)
    end
    value
end

function _brm_input_name(path, multi)
    first(path) === :callables && return Symbol(:callable_, path[2][2][1])
    length(path) >= 3 && path[2] == (:property, :plans) || return :model_input
    plan = multi.plans[only(path[3][2])]
    tail = path[4:end]
    if length(tail) >= 2 && first(tail) == (:property, :predictors)
        component = plan.predictors[only(tail[2][2])]
        predictor = component.predictor.name
        fields = tail[3:end]
        fields == ((:property, :design), (:property, :matrix)) &&
            return Symbol(:X_, predictor)
        fields == ((:property, :design), (:property, :fixed)) &&
            return Symbol(:offset_, predictor)
        if length(fields) == 2 && fields[2][1] === :index
            fields[1] == (:property, :random_effects) && return Symbol(
                :group_effects_, predictor, :_, only(fields[2][2]))
            return Symbol(fields[1][2], :_, predictor, :_, only(fields[2][2]))
        end
        return Symbol(:predictor_, predictor)
    elseif tail == ((:property, :observation_weight), (:property, :values))
        return Symbol(:weights_, plan.response_name)
    elseif length(tail) == 2 && first(tail) == (:property, :response_modifier)
        return Symbol(last(tail)[2], :_, plan.response_name)
    elseif length(tail) == 3 && tail[1:2] ==
            ((:property, :context), (:property, :data))
        return only(last(tail)[2])
    end
    :model_input
end

function _brm_ast_symbols!(symbols, node)
    node isa Symbol && push!(symbols, node)
    node isa Expr && foreach(arg -> _brm_ast_symbols!(symbols, arg), node.args)
    symbols
end

function _brm_model_inputs!(body, multi, callables, response_symbols)
    _brm_qualify_generated_calls!(body)
    names = copy(response_symbols)
    values = Any[plan.response for plan in multi.plans]
    reserved = _brm_ast_symbols!(Set{Symbol}(names), body)
    bindings = Dict{Any,Symbol}()
    for (index, name) in enumerate(response_symbols)
        bindings[(:multi, (:property, :plans), (:index, (index,)),
                  (:property, :response))] = name
    end
    function bind(path)
        haskey(bindings, path) && return bindings[path]
        candidate = _brm_fresh_model_name(_brm_input_name(path, multi), reserved)
        push!(reserved, candidate)
        push!(names, candidate)
        push!(values, _brm_input_value(path, multi, callables))
        bindings[path] = candidate
        candidate
    end
    function rewrite(node)
        path = _brm_input_path(node)
        (isnothing(path) || length(path) == 1) || return bind(path)
        node isa Expr || return node
        Expr(node.head, map(rewrite, node.args)...)
    end
    # Prepared references, rather than textual symbol counts, identify the raw
    # inputs. A design-only column cannot accidentally condition a same-named
    # generated latent site (for example a column named `beta_pop`).
    body.args = map(rewrite, body.args)
    Base.remove_linenums!(body)
    NamedTuple{Tuple(names)}(Tuple(values))
end
