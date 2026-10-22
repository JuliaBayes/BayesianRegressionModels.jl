"""
    _rk_observation_source!(definitions, bindings, entry, family)
    _rk_observation_source!(definitions, bindings, entry, family, ::Val{N})

Optional source provider for the exact scientific observation constructor
`family`. Return `nothing` to retain its ordinary constructor route. To claim
the scalar graph route, define a fresh explicit `ReactiveKernels.@kernel`
`entry(value, args...)` returning the normalized scalar log density, append
any preceding child graph definitions and separately named numerical leaves,
and return a non-`nothing` value. `args` are the original constructor's
arguments on the observation cell's axis. A shared scalar broadcasts; grouped
row arguments follow their response's subject partition. Original data ports
remain available to other consumers. Prepare history-dependent row values
before invoking this entry. The provider receives no model values or AD state.

BRM calls the provider once per observation statement, each time with a fresh
`entry`, and always through the second form: `N` is the number of constructor
arguments that observation writes (`y ~ family(args...)` has `N =
length(args)`; the response and an in-cell `weighted` weight are not counted).
Its default forwards to the first form, so a family observed with one argument
count needs only that. A family with several argument counts (Stan overloads
of one `@lpxf` name) defines one `::Val{N}` method per count, each emitting
its own law. The entry must take exactly `N + 1` ports; a mismatch is an error.

The emitted observation `y .~ LogDensity.(entry, args...)` composes this graph
inside RKPPL's observation plate. It adds no sampled coordinates and keeps the
original observed response. Ordinary constructors and deterministic callable-source providers
remain available; neither certifies expansion of a density callback's body.
Extend with `import BayesianRegressionModels: _rk_observation_source!`.
"""
_rk_observation_source!(definitions, bindings, entry, family) = nothing
_rk_observation_source!(definitions, bindings, entry, family, ::Val) =
    _rk_observation_source!(definitions, bindings, entry, family)

# The ports of a provider's `@kernel entry(ports...) = body` definition.
function _rk_kernel_entry_ports(definition)
    body = only(arg for arg in definition.args[2:end] if !(arg isa LineNumberNode))
    first(body.args).args[2:end]
end

function _rk_emit_observation_source!(defs, statements, bindings, taken,
        observation, distribution, response)
    entry = _rk_ast_fresh_name(string(observation.name, "_scalar_logdensity"), taken)
    supplied_defs = Expr[]
    supplied_bindings = Pair{Symbol,Any}[]
    arity = length(distribution.args)
    claimed = _rk_observation_source!(supplied_defs, supplied_bindings,
        entry, distribution.callable, Val(arity))
    if claimed === nothing
        isempty(supplied_defs) && isempty(supplied_bindings) || error(
            "RK source: unclaimed observation provider changed its source collections")
        callable = distribution.callable
        if callable isa StanBlocks.ValueFamily ||
                (callable isa Function && isempty(methods(callable)))
            family = callable isa StanBlocks.ValueFamily ? callable.label : nameof(callable)
            error("RK source: `$(observation.name) ~ $family(...)` needs a scalar " *
                "`_rk_observation_source!` provider on the RK backend; " *
                "this Stan-only family has no scalar Julia constructor")
        end
        return false
    end
    observation.modifier === nothing || error(
        "RK source: observation graph provider must include the exact response modifier")
    sources = _rk_source_definition.(supplied_defs)
    position = findfirst(source -> source.name === entry && source.kind === :kernel, sources)
    position === nothing && error(
        "RK source: observation provider must define its scalar @kernel entry `$entry`")
    ports = length(_rk_kernel_entry_ports(supplied_defs[position]))
    family = nameof(distribution.callable)
    ports == arity + 1 || error(
        "RK source: `$(observation.name) ~ $family(...)` writes $arity constructor " *
        "arguments, but its observation provider's entry `$entry` takes $ports ports " *
        "instead of the value plus $arity; define `_rk_observation_source!(definitions, " *
        "bindings, entry, ::typeof($family), ::Val{$arity})` for this argument count")
    any(pair -> first(pair) === entry || last(pair) === distribution.callable,
        supplied_bindings) && error(
        "RK source: observation provider cannot bind its entry or original constructor")
    append!(defs, supplied_defs)
    append!(bindings, supplied_bindings)
    union!(taken, (source.name for source in sources), first.(supplied_bindings))

    # The scalar law is observed directly: RKPPL composes the explicit graph
    # inside the observation plate, one cell per observed value, with the
    # constructor's arguments broadcast on the response's axis.
    arguments = Any[_rk_value_expr!(bindings, argument, taken) for argument in distribution.args]
    entry = _rk_weighted_observation_entry!(defs, arguments, bindings, taken,
        observation.weight, entry)
    push!(statements, response isa AbstractVector ?
        _rk_observation_statement(observation,
            _rk_ast_dotted(:LogDensity, entry, arguments...), taken) :
        Expr(:call, :~, observation.name, Expr(:call, :LogDensity, entry, arguments...)))
    true
end

# An in-cell `weighted(family, weight, args...)` power likelihood scales the
# caller's normalized scalar law by its row weight. A fresh entry composes the
# law graph, so the weight is one more row-aligned port of the same plate.
_rk_weighted_observation_entry!(defs, arguments, bindings, taken, ::Nothing, entry) = entry
function _rk_weighted_observation_entry!(defs, arguments, bindings, taken,
        weight::_BRMPreparedRef, entry)
    weighted_entry = _rk_ast_fresh_name(string(entry, "_weighted"), taken)
    ports = [Symbol(:argument_, i) for i in eachindex(arguments)]
    push!(defs, :(ReactiveKernels.@kernel $weighted_entry(value, weight, $(ports...)) = begin
        density = $entry(value, $(ports...))
        weighted_density = weight * density
        return weighted_density
    end))
    pushfirst!(arguments, _rk_value_expr!(bindings, weight, taken))
    weighted_entry
end
