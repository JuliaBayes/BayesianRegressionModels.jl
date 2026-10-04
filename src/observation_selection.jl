# Response withholding is one semantic contract shared by backend adapters.
_brm_held_out_all_redirect() =
    " Holding out every observation leaves nothing to fit, and held-out " *
    "likelihoods are not the prior mechanism. For prior draws keep the model " *
    "identical and omit the response column from the data — the program " *
    "lowers to generated quantities automatically. To cross-validate, hold " *
    "out a strict subset of the responses."

function _brm_held_out_request(held_out; prefix="BRM")
    (held_out === nothing || held_out === ()) && return (; names=Set{Symbol}())
    held_out === :all && error(
        "$prefix: `held_out=:all` is not supported." * _brm_held_out_all_redirect())
    held_out isa AbstractString && error(
        "$prefix: `held_out` expects a response Symbol or a collection of response " *
        "Symbols; got $(repr(held_out))")
    values = held_out isa Symbol ? (held_out,) : try
        collect(held_out)
    catch
        error("$prefix: `held_out` expects a response Symbol or a collection of " *
              "response Symbols; got $(repr(held_out))")
    end
    all(x -> x isa Symbol, values) || error(
        "$prefix: every `held_out` response must be a Symbol; got $(repr(values))")
    names = Set{Symbol}(values)
    :all in names && error(
        "$prefix: `held_out=:all` is not supported." * _brm_held_out_all_redirect())
    (; names)
end

function _brm_resolve_held_out(request, aliases, sources, unbound; prefix="BRM")
    isempty(request.names) && return Set{Symbol}()
    if isempty(sources)
        isempty(unbound) && error(
            "$prefix: `held_out` was requested, but this BRMI emits no observation likelihoods")
        error("$prefix: every observation (`$(join(sort!(unbound), "`, `"))`) is " *
              "unbound (response omitted from the data); there is no data to hold " *
              "out. Omit `held_out`: the program already lowers to generated " *
              "quantities.")
    end
    unknown = sort!(collect(setdiff(request.names, Set(keys(aliases)))))
    if !isempty(unknown)
        unbound_hit = sort!(Symbol[n for n in unknown if n in unbound])
        hint = isempty(unbound_hit) ? "" :
            " (`$(join(unbound_hit, "`, `"))` is unbound (response omitted), not holdable.)"
        error("$prefix: `held_out` names unknown response(s) $(unknown). Available " *
              "responses: $(sort!(collect(keys(aliases)))).$hint")
    end
    ambiguous = sort!(Symbol[name for name in request.names if length(aliases[name]) > 1])
    isempty(ambiguous) || error(
        "$prefix: `held_out` alias(es) $(ambiguous) each resolve to several " *
        "response data sources. Name the dataframe response column instead.")
    selected = reduce(union, (aliases[name] for name in request.names); init=Set{Symbol}())
    selected == sources && error(
        "$prefix: holding out $(join(sort!(collect(selected)), ", ")) covers every " *
        "observation." * _brm_held_out_all_redirect())
    selected
end
