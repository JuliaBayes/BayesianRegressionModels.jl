"""
    _BRMPreparedRef(name, axis)

A logical reference in a prepared model expression. `axis` is `:scalar`,
`:observation`, or `:whole`; it makes row indexing a property of the formula
reference rather than a guess based on the value's Julia container type.
"""
struct _BRMPreparedRef
    name::Symbol
    axis::Symbol
    function _BRMPreparedRef(name::Symbol, axis::Symbol)
        axis in (:scalar, :observation, :observation_row, :whole) || throw(ArgumentError(
            "prepared reference axis must be :scalar, :observation, :observation_row, or :whole"))
        new(name, axis)
    end
end

"""A callable expression retained from an `ExprColumn`, including keywords."""
struct _BRMPreparedExpr{F,A<:Tuple,K<:NamedTuple}
    callable::F
    args::A
    kwargs::K
end

struct _BRMPreparedParameter{P,S}
    name::Symbol
    prior::P
    support::S
    normalized::Bool
end

struct _BRMPreparedPredictor{P,B,R,D}
    name::Symbol
    population::P
    beta_prior::B
    random_effects::R
    dependencies::D
end

struct _BRMPreparedAssignment{E,D}
    name::Symbol
    expression::E
    dependencies::D
end

struct _BRMPreparedObservation{L,E,Y,M,W}
    name::Symbol
    lhs::L
    distribution::E
    response::Y
    modifier::M
    weight::W
end

"""
Backend-neutral semantic program. Backends may prepare only the operation
classes they consume; `source_operations` always retains the complete ordered
BRMI program so unsupported backend features are not discarded.
"""
struct _BRMPreparedModel{G,P,A,O,S}
    program::G
    parameters::P
    predictors::A
    assignments::O
    observations::S
end

function Base.getproperty(model::_BRMPreparedModel, field::Symbol)
    field === :context && return getfield(model, :program).context
    field === :order && return getfield(model, :program).order
    field === :source_operations &&
        return getfield(model, :program).context.parent.operations
    getfield(model, field)
end


"""
    _brm_prepare_model(brmi; context=_brm_backend_context(brmi))

Prepare the backend-neutral ordered expression graph without deciding which
operations a backend can execute. This intentionally retains exotic terms and
joint responses in `source_operations`; backend-specific preparation can add
geometry while sharing the same identities and dependency order.
"""
function _brm_prepare_model(brmi::BRMI;
                            program=_brm_prepare_program(brmi))
    context = program.context
    axes = Dict{Symbol,Symbol}(key => :observation for key in keys(context.data))
    parameters = Any[]
    assignments = Any[]
    observations = Any[]
    operations_by_name = Dict(operation.name => operation
                              for operation in program.operations)
    for operation_name in program.order
        prepared_operation = operations_by_name[operation_name]
        key = prepared_operation.name
        operation = prepared_operation.expression
        if prepared_operation.role === :assignment
            lhs, rhs = getargs(operation, 2)
            target = lhs isa NamedColumn ? name(lhs) : key
            prepared = _brm_prepare_expr(rhs, axes)
            refs = _brm_prepared_references(prepared)
            axes[target] = any(ref -> get(axes, ref, :scalar) in
                                      (:observation, :observation_row), refs) ?
                           :observation : :scalar
            push!(assignments, _BRMPreparedAssignment(
                target, prepared, refs))
            continue
        end
        operation isa ExprColumn{typeof(~)} || continue
        lhs, rhs = getargs(operation, 2)
        observation_name = _brm_observation_name(lhs)
        if !isnothing(observation_name)
            prepared = _brm_prepare_expr(rhs, axes)
            response = get(context.data, observation_name, nothing)
            push!(observations, _BRMPreparedObservation(
                observation_name, lhs, prepared, response, nothing, nothing))
            continue
        end
        if prepared_operation.role === :parameter
            lhs isa NamedColumn || continue
            target = name(lhs)
            axes[target] = _brm_parameter_reference_axis(rhs)
            prior_axes = copy(axes)
            foreach(key -> prior_axes[key] = :whole, keys(context.data))
            push!(parameters, _BRMPreparedParameter(
                target, _brm_prepare_expr(rhs, prior_axes), :distribution, true))
            continue
        end
        peeled = _peel_lp_lhs(lhs)
        isnothing(peeled) && continue
        target = last(peeled)
        axes[target] = :observation
        prepared = _brm_prepare_expr(rhs, axes)
        push!(assignments, _BRMPreparedAssignment(
            target, prepared, _brm_prepared_references(prepared)))
    end
    _BRMPreparedModel(
        program, Tuple(parameters), (), Tuple(assignments), Tuple(observations))
end

function _brm_prepared_references!(out::Set{Symbol}, x)
    x isa _BRMPreparedRef && (push!(out, x.name); return out)
    x isa _BRMPreparedExpr || return out
    foreach(arg -> _brm_prepared_references!(out, arg), x.args)
    foreach(arg -> _brm_prepared_references!(out, arg), values(x.kwargs))
    out
end
function _brm_prepared_references!(out::Set{Symbol}, values::Tuple)
    foreach(value -> _brm_prepared_references!(out, value), values)
    out
end
_brm_prepared_references(x) =
    Tuple(sort!(collect(_brm_prepared_references!(Set{Symbol}(), x))))

function _brm_prepare_expr(x, axes::AbstractDict{Symbol,Symbol})
    if x isa NamedColumn
        key = name(x)
        axis = get(axes, key, parent(x) isa DataColumn ? :observation : :scalar)
        if axis === :observation && parent(x) isa DataColumn &&
           parent(parent(x)) isa AbstractMatrix
            axis = :observation_row
        end
        return _BRMPreparedRef(key, axis)
    elseif x isa ExprColumn
        return _brm_prepare_call_expr(getf(x), x, axes)
    elseif x isa Tuple
        return map(value -> _brm_prepare_expr(value, axes), x)
    elseif x isa AbstractArray
        elements = Tuple(_brm_prepare_expr(value, axes) for value in x)
        vector = _BRMPreparedExpr(Base.vect, elements, (;))
        return ndims(x) == 1 ? vector :
            _BRMPreparedExpr(reshape, (vector, size(x)...), (;))
    end
    x
end

_brm_prepare_expr(x; axes=Dict{Symbol,Symbol}()) = _brm_prepare_expr(x, axes)

function _brm_prepare_call_expr(callable, expression, axes)
    args = map(arg -> _brm_prepare_expr(arg, axes), getargs(expression))
    kwargs = map(value -> _brm_prepare_expr(value, axes), getkwargs(expression))
    _BRMPreparedExpr(callable, args, kwargs)
end

# A data vector used as a mathematical vector argument is one whole value.
# Matrices supply one vector per observation row; parameter references keep
# their resolved shape and expressions retain their explicit construction.
function _brm_prepare_vector_argument(argument, axes)
    if argument isa NamedColumn && parent(argument) isa DataColumn &&
       parent(parent(argument)) isa AbstractVector{<:Real}
        return _BRMPreparedRef(name(argument), :whole)
    end
    _brm_prepare_expr(argument, axes)
end

function _brm_prepare_call_expr(callable::Type{<:Multinomial}, expression, axes)
    raw = getargs(expression)
    args = Tuple(index == 2 ? _brm_prepare_vector_argument(argument, axes) :
                 _brm_prepare_expr(argument, axes) for (index, argument) in enumerate(raw))
    _BRMPreparedExpr(callable, args,
        map(value -> _brm_prepare_expr(value, axes), getkwargs(expression)))
end

function _brm_prepare_call_expr(callable::Type{<:MvNormal}, expression, axes)
    args = Tuple(if index > 1 && argument isa NamedColumn &&
                    parent(argument) isa DataColumn &&
                    parent(parent(argument)) isa AbstractMatrix
                     _BRMPreparedRef(name(argument), :whole)
                 else
                     _brm_prepare_vector_argument(argument, axes)
                 end for (index, argument) in enumerate(getargs(expression)))
    _BRMPreparedExpr(callable, args,
        map(value -> _brm_prepare_expr(value, axes), getkwargs(expression)))
end

# This evaluator is for preparation/replay diagnostics. Executable backends
# lower `_BRMPreparedExpr` to native code and do not call it in density loops.
_brm_eval_prepared_expr(x, _values, _row=nothing) = x
function _brm_eval_prepared_expr(x::_BRMPreparedRef, values, row=nothing)
    value = getproperty(values, x.name)
    x.axis === :observation_row ? view(value, row, :) :
        x.axis === :observation ? value[row] : value
end
function _brm_eval_prepared_expr(x::_BRMPreparedExpr, values, row=nothing)
    args = map(arg -> _brm_eval_prepared_expr(arg, values, row), x.args)
    kwargs = map(arg -> _brm_eval_prepared_expr(arg, values, row), x.kwargs)
    x.callable(args...; kwargs...)
end
