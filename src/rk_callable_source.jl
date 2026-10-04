"""
    _rk_callable_source!(definitions, bindings, entry, f)

Source provider for an exact ordinary callable identity used by an emitted
expression, including calls nested inside an anonymous kernel. Return
`nothing` to keep the existing callable binding. To claim it, append an
ordinary Julia `function entry(...) ... end` or an explicit
`ReactiveKernels.@kernel entry(...) = begin ... end` compatible with the emitted
call's inputs, optionally append helper definitions and separately named
leaf bindings, and return a non-`nothing` value. The emitted entry replaces
the original binding; existing call sites retain their arguments. Graph
definitions may expose named plate/scan recipes; ordinary Julia functions do
not make their internal loops visible to RK. Domain source may choose a new
scientifically equivalent API rather than retain a historical argument count.
Provider graph definitions precede their emitted reader/cell graphs; place
helper graph definitions before entries that compose them.

The provider receives no values, lowered plan or derivative. It supplies its
own native mathematics under the caller's source namespace. An entry cannot
also be bound, and rebinding the original callable as its own leaf is refused.
Extend with `import BayesianRegressionModels: _rk_callable_source!`.
"""
_rk_callable_source!(definitions, bindings, entry, f) = nothing

function _rk_resolve_callable_sources(emitted::_RKEmittedProgram)
    definitions = copy(emitted.defs)
    pending = copy(emitted.bindings)
    bindings = Pair{Symbol,Any}[]
    claimed = IdDict{Any,Symbol}()
    while !isempty(pending)
        entry, callable = popfirst!(pending)
        haskey(claimed, callable) && error(
            "RK source: provider for `$(claimed[callable])` rebound its original callable as `$entry`")
        provider_definitions = Expr[]
        provider_bindings = Pair{Symbol,Any}[]
        result = _rk_callable_source!(provider_definitions, provider_bindings, entry, callable)
        if result === nothing
            isempty(provider_definitions) && isempty(provider_bindings) || error(
                "RK source: unclaimed provider for `$entry` changed its source collections")
            push!(bindings, entry => callable)
            continue
        end
        names = Symbol[]
        for definition in provider_definitions
            source = _rk_source_definition(definition)
            source.kind in (:function, :kernel) || error(
                "RK source: callable provider for `$entry` needs ordinary Julia functions or explicit @kernel definitions")
            push!(names, source.name)
        end
        entry in names || error("RK source: callable provider did not define its entry `$entry`")
        claimed[callable] = entry
        prepend!(definitions, provider_definitions)
        append!(pending, provider_bindings)
    end
    resolved = _RKEmittedProgram(definitions, emitted.main, bindings)
    _rk_validate_source_definitions(resolved)
    resolved
end

_rk_source_program(definitions, main, bindings) =
    _rk_resolve_callable_sources(_RKEmittedProgram(definitions, main, bindings))
