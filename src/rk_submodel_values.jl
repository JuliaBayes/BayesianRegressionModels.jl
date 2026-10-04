"""
    _rk_submodel_rhs!(definitions, statements, data, bindings, target, f, rhs)

Native source extension for `target ~ f(...)` deterministic or statistical
submodel bindings. Return non-`nothing` to claim the binding, or `nothing` to
retain ordinary population-formula preparation. The caller owns all four
mutable collections. `rhs` retains the original positional and keyword column
references; parameter-dependent values must remain named source references.

Append ordinary `@rkppl` definitions (`name(args...) = begin ... end`) and/or
ordinary Julia function definitions, then ordinary main statements producing
`target`. `data` holds preparation-only inputs, and `bindings` holds exact
callable identities. All declarations, priors and observations must be visible
in this source; the extension never receives a lowered PPL plan or derivative.
An emitted entry owns its callable name: bind its existing leaf functions under
separate names, or bind an existing entry without defining it again. Defining
and binding the same name is rejected before any definitions are evaluated.
Extend this method with `import BayesianRegressionModels: _rk_submodel_rhs!`.
Model-specific downstream mathematics stays in the downstream extension.
"""
_rk_submodel_rhs!(definitions, statements, data, bindings, target, f, rhs) = nothing

struct _RKPreparedSubmodelAssignment
    name::Symbol
    definitions::Vector{Expr}
    statements::Vector{Expr}
    columns::Dict{Symbol,Any}
    bindings::Vector{Pair{Symbol,Any}}
    globals::Vector{Symbol}
end

function _rk_prepare_submodel_values(program)
    submodels = Any[]
    for operation in program.operations
        operation.role === :predictor || continue
        rhs = last(getargs(operation.expression))
        rhs isa ExprColumn || continue
        getf(rhs) === kernel && continue
        definitions, statements = Expr[], Expr[]
        columns, bindings = Dict{Symbol,Any}(), Pair{Symbol,Any}[]
        claimed = _rk_submodel_rhs!(definitions, statements, columns, bindings,
            operation.name, getf(rhs), rhs)
        claimed === nothing && continue
        outputs = Set{Symbol}()
        foreach(statement -> _rk_source_outputs!(outputs, statement), statements)
        operation.name in outputs || error(
            "RK backend: native submodel `$(operation.name)` did not emit its result binding")
        referenced = Set{Symbol}()
        foreach(statement -> _rk_source_symbols!(referenced, statement), statements)
        available = Set(op.name for op in program.operations)
        globals = sort!(collect(intersect(
            setdiff(referenced, outputs, Set(keys(columns)), Set(first.(bindings))),
            available)))
        push!(submodels, _RKPreparedSubmodelAssignment(operation.name, definitions,
            statements, columns, bindings, globals))
    end
    Tuple(submodels)
end

function _rk_emit_value_assignment!(defs, statements, bindings, taken,
        submodel::_RKPreparedSubmodelAssignment)
    for (name, value) in submodel.bindings
        previous = findfirst(pair -> first(pair) === name, bindings)
        previous === nothing || last(bindings[previous]) === value || error(
            "RK backend: native submodel `$(submodel.name)` callable `$name` collides")
        previous === nothing && push!(bindings, name => value)
    end
    append!(defs, submodel.definitions)
    append!(statements, submodel.statements)
end
