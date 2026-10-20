# Response withholding is one semantic contract shared by backend adapters.
_brm_held_out_all_redirect() =
    " Holding out every observation leaves nothing to fit, and held-out " *
    "likelihoods are not the prior mechanism. For prior draws keep the model " *
    "identical and omit the response column from the data — the program " *
    "lowers to generated quantities automatically. To cross-validate, hold " *
    "out a strict subset of the responses."

# `held_out` names whole responses (`:y`, `[:y, :z]`) and per-row hold-outs
# (`(; y = mask)`, `:y => mask`), in any mix. `rows` keeps each mask as given,
# keyed by the name it was given under; a backend validates it against the
# response's bound values (`_brm_held_out_mask`).
function _brm_held_out_request(held_out; prefix="BRM")
    names = Set{Symbol}()
    rows = Dict{Symbol,Any}()
    (held_out === nothing || held_out === ()) && return (; names, rows)
    held_out === :all && error(
        "$prefix: `held_out=:all` is not supported." * _brm_held_out_all_redirect())
    held_out isa AbstractString && error(_brm_held_out_shape_error(held_out, prefix))
    entries = held_out isa Symbol ? (held_out,) :
        held_out isa NamedTuple ? collect(pairs(held_out)) : try
        collect(held_out)
    catch
        error(_brm_held_out_shape_error(held_out, prefix))
    end
    foreach(entry -> _brm_held_out_entry!(names, rows, entry, prefix), entries)
    :all in names && error(
        "$prefix: `held_out=:all` is not supported." * _brm_held_out_all_redirect())
    both = sort!(collect(intersect(names, keys(rows))))
    isempty(both) || error(
        "$prefix: `held_out` names $(both) both as a whole response and with a row mask")
    (; names, rows)
end

_brm_held_out_shape_error(held_out, prefix) =
    "$prefix: `held_out` expects a response Symbol, `response => row_mask`, or a " *
    "collection or NamedTuple of those; got $(repr(held_out))"

_brm_held_out_entry!(names, rows, entry::Symbol, prefix) = push!(names, entry)
function _brm_held_out_entry!(names, rows, entry::Pair, prefix)
    name = first(entry)
    name isa Symbol || error(
        "$prefix: a `held_out` row mask is keyed by a response Symbol; got $(repr(name))")
    haskey(rows, name) && error("$prefix: `held_out` gives `$name` two row masks")
    rows[name] = last(entry)
    rows
end
_brm_held_out_entry!(names, rows, entry, prefix) = error(
    "$prefix: every `held_out` entry must be a response Symbol or " *
    "`response => row_mask`; got $(repr(entry))")

# A row mask has the shape of the response it holds out: one `Bool` per value
# of a flat response, one `Bool` vector per array of a response bound as one
# array per subject (an observation inside `@plate for`). `true` holds a row
# out. Returns the mask as fresh `Vector{Bool}` / `Vector{Vector{Bool}}`.
# `replay` names the replay's own `held_out=` in a shape error: a replay
# re-applies the fit's masks by default, which fit only data of the fit's shape.
function _brm_held_out_mask(mask, response, name::Symbol; prefix="BRM",
                            replay::Bool=false)
    describe = "the row mask for `held_out` response `$name`"
    hint = replay ? _brm_held_out_replay_hint(name) : ""
    response isa AbstractVector || error(
        "$prefix: $describe needs a response bound as a vector; `$name` is a " *
        "$(typeof(response))")
    mask isa AbstractVector || error(
        "$prefix: $describe must be a vector shaped like the response; got " *
        "$(typeof(mask))")
    length(mask) == length(response) || error(
        "$prefix: $describe has length $(length(mask)), but the response has " *
        "$(length(response)) $(_brm_held_out_nested(response) ? "arrays" : "values")." *
        hint)
    _brm_held_out_nested(response) || return _brm_held_out_flat_mask(mask, describe, prefix)
    map(enumerate(zip(mask, response))) do (i, (inner, values))
        inner isa AbstractVector && length(inner) == length(values) || error(
            "$prefix: $describe entry $i must be a Bool vector of length " *
            "$(length(values)), like response array $i; got $(repr(inner))." * hint)
        _brm_held_out_flat_mask(inner, describe, prefix)
    end
end
_brm_held_out_replay_hint(name) =
    " A replay keeps the fit's row masks, which fit only data of the fit's " *
    "shape: pass the replay data's own mask, `held_out=(; $name = mask)`, or " *
    "`held_out=()` to hold nothing out."
_brm_held_out_nested(response) =
    !isempty(response) && all(value -> value isa AbstractVector, response)
function _brm_held_out_flat_mask(mask, describe, prefix)
    all(value -> value isa Bool, mask) || error(
        "$prefix: $describe must hold `Bool` values (`true` holds a row out); got " *
        "$(eltype(mask))")
    collect(Bool, mask)
end
_brm_held_out_all_rows(mask::Vector{Bool}) = all(mask)
_brm_held_out_all_rows(mask::Vector{Vector{Bool}}) = all(all, mask)
_brm_held_out_no_rows(mask::Vector{Bool}) = !any(mask)
_brm_held_out_no_rows(mask::Vector{Vector{Bool}}) = !any(any, mask)

# Resolve a request to response data sources: `whole` holds every row of a
# source out, `rows` maps a source to its validated row mask (neither all rows
# nor none). `response(source)` returns the source's bound values for mask
# validation.
function _brm_resolve_held_out(request, aliases, sources, unbound; prefix="BRM",
                               cover_all::Bool=false, response=nothing,
                               replay::Bool=false)
    requested = union(request.names, keys(request.rows))
    rows = Dict{Symbol,Any}()
    isempty(requested) && return (; whole=Set{Symbol}(), rows)
    if isempty(sources)
        isempty(unbound) && error(
            "$prefix: `held_out` was requested, but this BRMI emits no observation likelihoods")
        error("$prefix: every observation (`$(join(sort!(unbound), "`, `"))`) is " *
              "unbound (response omitted from the data); there is no data to hold " *
              "out. Omit `held_out`: the program already lowers to generated " *
              "quantities.")
    end
    unknown = sort!(collect(setdiff(requested, Set(keys(aliases)))))
    if !isempty(unknown)
        unbound_hit = sort!(Symbol[n for n in unknown if n in unbound])
        hint = isempty(unbound_hit) ? "" :
            " (`$(join(unbound_hit, "`, `"))` is unbound (response omitted), not holdable.)"
        error("$prefix: `held_out` names unknown response(s) $(unknown). Available " *
              "responses: $(sort!(collect(keys(aliases)))).$hint")
    end
    ambiguous = sort!(Symbol[name for name in requested if length(aliases[name]) > 1])
    isempty(ambiguous) || error(
        "$prefix: `held_out` alias(es) $(ambiguous) each resolve to several " *
        "response data sources. Name the dataframe response column instead.")
    whole = reduce(union, (aliases[name] for name in request.names); init=Set{Symbol}())
    for (name, mask) in request.rows
        source = only(aliases[name])
        source in whole && error(
            "$prefix: `held_out` holds response `$name` out both whole and by rows")
        haskey(rows, source) && error(
            "$prefix: `held_out` gives response `$name` a second row mask")
        response === nothing && error(
            "$prefix: a per-row `held_out` mask is not supported by this backend yet")
        validated = _brm_held_out_mask(mask, response(source), name; prefix, replay)
        if _brm_held_out_all_rows(validated)
            push!(whole, source)
        elseif !_brm_held_out_no_rows(validated)
            rows[source] = validated
        end
    end
    whole == sources && !cover_all && error(
        "$prefix: holding out $(join(sort!(collect(whole)), ", ")) covers every " *
        "observation." * _brm_held_out_all_redirect())
    (; whole, rows)
end
