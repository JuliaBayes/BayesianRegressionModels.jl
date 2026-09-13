# Source-level preparation is independent of any backend's supported term set.
# Every operation survives this pass, including extension-owned formula nodes.
struct _BRMPreparedOperation{E,D}
    name::Symbol
    role::Symbol
    expression::E
    dependencies::D
end

struct _BRMPreparedProgram{C,O,N}
    context::C
    operations::O
    order::N
end

"""
    brm_distribution_type(constructor)

Describe the distribution returned by a callable used in a BRM formula.
Distribution types describe themselves. A factory function can extend this
trait, for example `brm_distribution_type(::typeof(my_prior)) = Normal`, so
BRM recognizes a sampled declaration and its scalar/vector shape without
executing the factory before its sampled arguments exist. The original call,
including its keywords, is retained. Return `nothing` for non-distribution
callables. Stan emission additionally requires the factory's Stan translation.
"""
brm_distribution_type(::Type{D}) where {D<:Distribution} = D
brm_distribution_type(::Type{D}) where {D<:LocationScale} = D
brm_distribution_type(_) = nothing

_brm_prior_constructor(constructor) = !isnothing(brm_distribution_type(constructor))
_brm_prior_constructor(::Type{Horseshoe}) = true
_brm_prior_constructor(::typeof(LKJCovarianceFactor)) = true
_brm_prior_expression(x) = false
_brm_prior_expression(x::ExprColumn) = _brm_prior_constructor(getf(x))

function _brm_operation_role(op::ExprColumn{typeof(~)})
    lhs, rhs = getargs(op, 2)
    isnothing(_brm_observation_name(lhs)) || return :observation
    lhs isa ExprColumn && getf(lhs) === effect && return :prior_modifier
    lhs isa NamedColumn && _brm_prior_expression(rhs) ? :parameter : :predictor
end
_brm_operation_role(::ExprColumn{typeof(assign)}) = :assignment
_brm_operation_role(_) = :extension

_brm_operation_references!(refs, _) = refs
_brm_operation_references!(refs, node::NamedColumn) = (push!(refs, name(node)); refs)
function _brm_operation_references!(refs, node::ExprColumn)
    foreach(arg -> _brm_operation_references!(refs, arg), getargs(node))
    foreach(arg -> _brm_operation_references!(refs, arg), values(getkwargs(node)))
    refs
end
function _brm_operation_references!(refs, values::Union{Tuple,AbstractArray})
    foreach(value -> _brm_operation_references!(refs, value), values)
    refs
end
function _brm_operation_references!(refs, values::NamedTuple)
    foreach(value -> _brm_operation_references!(refs, value), values)
    refs
end
function _brm_operation_references!(refs, mapping::AbstractDict)
    foreach(value -> _brm_operation_references!(refs, value), values(mapping))
    refs
end

# Backends that allocate group/structured blocks before their ordinary walk
# still consume model values in the common graph's dependency order.
function _brm_prior_value_dependencies(program::_BRMPreparedProgram, roots)
    byname = Dict(operation.name => operation for operation in program.operations)
    required = Set{Symbol}()
    function visit(name)
        name in required && return
        haskey(byname, name) || error("BRM preparation: prior references unknown model value `$name`")
        operation = byname[name]
        operation.role === :extension && haskey(program.context.data, name) && return
        operation.role in (:parameter, :assignment) || error(
            "BRM preparation: model-level prior references `$name`, a $(operation.role); " *
            "it requires a sampled parameter, deterministic model value, or data")
        push!(required, name)
        foreach(visit, operation.dependencies)
    end
    foreach(visit, roots)
    Tuple(name for name in program.order if name in required)
end

function _brm_operation_dependencies(op, names, key, role)
    role === :prior_modifier && return ()
    op isa ExprColumn || return ()
    args = getargs(op)
    length(args) == 2 || return ()
    refs = _brm_operation_references!(Set{Symbol}(), last(args))
    # Preserve source order between independent nodes. References retain their
    # callable expression in the operation; this list only orders declarations.
    Tuple(name for name in names if name != key && name in refs)
end

function _brm_operation_order(operations)
    available = Set{Symbol}()
    remaining = collect(operations)
    order = Symbol[]
    while !isempty(remaining)
        index = findfirst(op -> all(in(available), op.dependencies), remaining)
        isnothing(index) && error(
            "BRM preparation: cyclic model declarations: " *
            join((string(op.name) for op in remaining), ", "))
        op = popat!(remaining, index)
        push!(order, op.name)
        push!(available, op.name)
    end
    Tuple(order)
end

"""
    _brm_prepare_program(brmi; data=Dict{Symbol,Any}())

Collect shared data/row-axis facts and a stable dependency order over the full
BRMI source. Formula terms and distribution calls remain typed `ExprColumn`
values: this pass neither selects a likelihood family nor emits backend code.
"""
function _brm_prepare_program(brmi::BRMI; data=Dict{Symbol,Any}(),
                              context=_brm_backend_context(brmi; data))
    names = keys(brmi.operations)
    operations = map(names) do key
        raw = brmi.operations[key]
        op = _named_op(raw)
        isnothing(op) && (op = raw)
        role = _brm_operation_role(op)
        _BRMPreparedOperation(key, role, op,
            _brm_operation_dependencies(op, names, key, role))
    end
    order = _brm_operation_order(operations)
    byname = Dict(op.name => op for op in operations)
    # An assignment preserves its consumers' observation axis. Propagate this
    # through assignments only: structured predictor terms may intentionally
    # change axes (for example subject-local kernels over ragged events).
    for key in reverse(order)
        operation = byname[key]
        operation.role === :assignment || continue
        observation = get(context.target_obs, key, nothing)
        isnothing(observation) && continue
        for dependency in operation.dependencies
            get!(context.target_obs, dependency, observation)
        end
    end
    program = _BRMPreparedProgram(context, operations, order)
    claims = Pair{Symbol,Tuple}[
        target => Tuple(entry.spec.expression
            for per_term in values(terms) for entry in values(per_term))
        for (target, terms) in context.term_priors]
    isempty(claims) ? program : _brm_with_prior_dependencies(program, claims)
end

function _brm_is_prior_declaration(brmi::BRMI, key::Symbol)
    haskey(brmi.operations, key) || return false
    op = _named_op(brmi.operations[key])
    !isnothing(op) && _brm_operation_role(op) === :parameter
end

"""
Attach the dependencies of resolved coefficient, group, or term priors to the
operation that owns their sample sites. Callers supply the expressions selected
by common prior-address resolution; this pass never repeats wildcard matching.
"""
function _brm_with_prior_dependencies(program::_BRMPreparedProgram, claims)
    names = Tuple(operation.name for operation in program.operations)
    additions = Dict{Symbol,Set{Symbol}}()
    for (owner, priors) in claims
        owner in names || error("BRM preparation: prior owner `$owner` has no model operation")
        refs = get!(additions, owner, Set{Symbol}())
        _brm_operation_references!(refs, priors)
    end
    operations = map(program.operations) do operation
        refs = union(Set(operation.dependencies), get(additions, operation.name, Set{Symbol}()))
        dependencies = Tuple(name for name in names if name in refs)
        _BRMPreparedOperation(operation.name, operation.role,
                              operation.expression, dependencies)
    end
    _BRMPreparedProgram(program.context, operations, _brm_operation_order(operations))
end
