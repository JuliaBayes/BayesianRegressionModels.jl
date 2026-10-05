"""
    _rk_observation_source!(definitions, bindings, entry, family)

Optional source provider for the exact scientific observation constructor
`family`. Return `nothing` to retain its ordinary constructor route. To claim
the scalar graph route, define a fresh explicit `ReactiveKernels.@kernel`
`entry(value, args...)` returning the normalized scalar log density, append
any preceding child graph definitions and separately named numerical leaves,
and return a non-`nothing` value. `args` are the original constructor's
arguments on the observation cell's axis. A shared scalar broadcasts; grouped
row arguments follow their response's subject partition, and flat row arguments
follow an explicit ragged response join. Original data ports remain available
to other consumers. Prepare history-dependent row values
before invoking this entry. The provider receives no model values or AD state.

The emitted observation `y .~ LogDensity.(entry, args...)` composes this graph
inside RKPPL's observation plate. It adds no sampled coordinates and keeps the
original observed response. Ordinary constructors and deterministic callable-source providers
remain available; neither certifies expansion of a density callback's body.
Extend with `import BayesianRegressionModels: _rk_observation_source!`.
"""
_rk_observation_source!(definitions, bindings, entry, family) = nothing

function _rk_emit_observation_source!(defs, statements, bindings, taken,
        observation, distribution, response)
    entry = _rk_ast_fresh_name(string(observation.name, "_scalar_logdensity"), taken)
    supplied_defs = Expr[]
    supplied_bindings = Pair{Symbol,Any}[]
    claimed = _rk_observation_source!(supplied_defs, supplied_bindings,
        entry, distribution.callable)
    if claimed === nothing
        isempty(supplied_defs) && isempty(supplied_bindings) || error(
            "RK source: unclaimed observation provider changed its source collections")
        return false
    end
    observation.modifier === nothing || error(
        "RK source: observation graph provider must include the exact response modifier")
    sources = _rk_source_definition.(supplied_defs)
    any(source -> source.name === entry && source.kind === :kernel, sources) || error(
        "RK source: observation provider must define its scalar @kernel entry `$entry`")
    any(pair -> first(pair) === entry || last(pair) === distribution.callable,
        supplied_bindings) && error(
        "RK source: observation provider cannot bind its entry or original constructor")
    append!(defs, supplied_defs)
    append!(bindings, supplied_bindings)
    union!(taken, (source.name for source in sources), first.(supplied_bindings))

    # The scalar law is observed directly: RKPPL composes the explicit graph
    # inside the observation plate, one cell per observed value, with the
    # constructor's arguments broadcast on the response's axis.
    arguments = [_rk_value_expr!(bindings, argument, taken) for argument in distribution.args]
    push!(statements, response isa AbstractVector ?
        Expr(:call, :.~, observation.name, _rk_ast_dotted(:LogDensity, entry, arguments...)) :
        Expr(:call, :~, observation.name, Expr(:call, :LogDensity, entry, arguments...)))
    true
end
