"""
    _rk_observation_source!(definitions, bindings, entry, family)

Optional source provider for the exact scientific observation constructor
`family`. Return `nothing` to retain its ordinary constructor route. To claim
the scalar graph route, define a fresh explicit `ReactiveKernels.@kernel`
`entry(value, args...)` returning the normalized scalar log density, append
any preceding child graph definitions and separately named numerical leaves,
and return a non-`nothing` value. `args` are the original constructor's
arguments on the observation cell's axis. A subject-shared scalar broadcasts;
a row vector keeps its original order. Prepare history-dependent row values
before invoking this entry. The provider receives no model values or AD state.

The emitted observation plate composes this graph before scoring its returned
log density. It adds no sampled coordinates and keeps the original observed
response. Ordinary constructors and deterministic callable-source providers
remain available; neither certifies expansion of a density callback's body.
Extend with `import BayesianRegressionModels: _rk_observation_source!`.
"""
_rk_observation_source!(definitions, bindings, entry, family) = nothing

function _rk_emit_observation_source!(defs, statements, bindings, taken,
        observation, distribution)
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

    values = _rk_ast_fresh_name(string(observation.name, "_law_values"), taken)
    row_values = observation.response isa AbstractVector ? observation.name :
        Expr(:vect, observation.name)
    push!(statements, Expr(:(=), values, row_values))
    arguments = Symbol[]
    for (i, argument) in enumerate(distribution.args)
        name = _rk_ast_fresh_name(string(observation.name, "_law_argument_", i), taken)
        # This is ordinary broadcast alignment, not a new packed model axis.
        expression = _rk_value_expr!(bindings, argument, taken)
        push!(statements, Expr(:(=), name, Expr(:call, :.*,
            Expr(:call, :ones, Expr(:call, :length, values)), expression)))
        push!(arguments, name)
    end
    ports = [values; arguments]
    index = _rk_ast_fresh_name("observation", Set(ports))
    cell_call = Expr(:call, entry, (Expr(:ref, port, index) for port in ports)...)
    plate = Expr(:do, Expr(:call,
        Expr(:., :ReactiveKernels, QuoteNode(:plate)),
        Expr(:call, :eachindex, values),
        (Expr(:call, :Ref, port) for port in ports)...),
        Expr(:->, Expr(:tuple, index, ports...), Expr(:block, cell_call)))
    reader = _rk_ast_fresh_name(string(observation.name, "_logdensity_reader"), taken)
    pointwise = _rk_ast_fresh_name("pointwise", Set(ports))
    definition = Expr(:(=), Expr(:call, reader, ports...),
        Expr(:block, Expr(:(=), pointwise, plate), Expr(:return, pointwise)))
    push!(defs, Expr(:macrocall,
        Expr(:., :ReactiveKernels, QuoteNode(Symbol("@kernel"))),
        LineNumberNode(0), definition))
    scores = _rk_ast_fresh_name(string(observation.name, "_logdensity"), taken)
    push!(statements, Expr(:(=), scores, Expr(:call, reader, ports...)))

    # All scientific algebra has already been composed into the numerical
    # graph. This explicit scalar scoring adapter only returns that value.
    score = _rk_ast_fresh_name("brm_logdensity_value", taken)
    push!(defs, :(function $score(observed, logdensity)
        return logdensity
    end))
    push!(statements, Expr(:call, :.~, observation.name,
        _rk_ast_dotted(:LogDensity, score, scores)))
    true
end
