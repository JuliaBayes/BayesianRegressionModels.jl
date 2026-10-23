# A term owns its addressable parameter slots. Resolution is shared by every
# backend; the selected expression is never reconstructed from a family name.
struct _BRMTermPriorSlot
    name::Symbol
    class::Symbol
    component::Union{Nothing,Symbol}
    support::Symbol
end
_BRMTermPriorSlot(name, class, support; component=nothing) =
    _BRMTermPriorSlot(name, class, component, support)

_brm_term_prior_slots(_callable) = ()
_brm_term_prior_slots(::typeof(s)) =
    (_BRMTermPriorSlot(:sd, :term_sd, :positive),)
_brm_term_prior_slots(::typeof(t2)) = Tuple(
    _BRMTermPriorSlot(Symbol(:sd_, block), :term_sd, :positive; component=block)
    for block in (:rr, :rn, :nr))
_brm_term_prior_slots(::typeof(gp)) = (
    _BRMTermPriorSlot(:sigma, :term_sd, :positive),
    _BRMTermPriorSlot(:length_scale, :term_length_scale, :positive))
_brm_term_prior_slots(::typeof(hsgp)) = _brm_term_prior_slots(gp)
_brm_term_prior_slots(::typeof(dar)) = (
    _BRMTermPriorSlot(:sigma, :term_sd, :positive),
    _BRMTermPriorSlot(:ar, :term_ar, :unit_interval))
# a random walk owns only its innovation scale: `ar(lp, rw(t))` is refused here
_brm_term_prior_slots(::typeof(rw)) =
    (_BRMTermPriorSlot(:sigma, :term_sd, :positive),)
_brm_term_prior_slots(::typeof(cdar)) = (
    _BRMTermPriorSlot(:sigma, :term_sd, :positive),
    _BRMTermPriorSlot(:ar, :term_ar, :unit_interval))
# `car` owns its scale and, under `type=:bym2`, the spatial mixing share; an
# `:icar` term refuses a `mixing(...)` address with the term's own message.
_brm_term_prior_slots(::typeof(car)) = (
    _BRMTermPriorSlot(:sigma, :term_sd, :positive),
    _BRMTermPriorSlot(:mixing, :term_mixing, :unit_interval))
_brm_term_prior_slots(::typeof(mo)) =
    (_BRMTermPriorSlot(:simplex, :term_simplex, :simplex),)
_brm_term_prior_slots(::typeof(mo1)) = _brm_term_prior_slots(mo)
_brm_term_prior_slots(::typeof(me)) =
    (_BRMTermPriorSlot(:latent, :term_latent, :real),)
_brm_term_prior_slots(::typeof(interval_censored)) = _brm_term_prior_slots(me)

_brm_term_prior_rank(spec) = isnothing(spec.predictor) ? 0 : 1

_brm_term_prior_class(::Val{:term_sd}) = "scale"
_brm_term_prior_class(::Val{:term_ar}) = "bounded persistence coefficient"
_brm_term_prior_class(::Val{:term_length_scale}) = "length scale"
_brm_term_prior_class(::Val{:term_simplex}) = "simplex"
_brm_term_prior_class(::Val{:term_latent}) = "latent covariate"
_brm_term_prior_class(::Val{:term_mixing}) = "mixing share"
_brm_term_prior_label(::Val{:term_sd}) = "sd"
_brm_term_prior_label(::Val{:term_ar}) = "ar"
_brm_term_prior_label(::Val{:term_length_scale}) = "length_scale"
_brm_term_prior_label(::Val{:term_simplex}) = "simplex"
_brm_term_prior_label(::Val{:term_latent}) = "latent"
_brm_term_prior_label(::Val{:term_mixing}) = "mixing"
function _brm_term_prior_spelling(spec)
    head = _brm_term_prior_label(Val(spec.class))
    lp = isnothing(spec.predictor) ? ":" : string(spec.predictor)
    component = isnothing(spec.component) ? "" : ", $(spec.component)"
    qualifiers = get(spec, :qualifiers, (;))
    term = isempty(qualifiers) ? string(spec.term) :
        string(chop(string(spec.term)), "; ",
               join(("$k=$(repr(v))" for (k, v) in pairs(qualifiers)), ", "), ")")
    "$head($lp, $term$component)"
end

# ---- keyword-qualified term addresses (decision 1fk9b29) --------------------
#
# Several terms over the same axes in one predictor share a term key. An
# address may name the term's literal keywords to reach only the terms that
# spell them (`length_scale(:, hsgp(x; cov=:periodic))`); its winning priors are stored
# under the term's FULL key, which also spells every keyword, and every reader
# looks that key up first. A predictor with one such term keeps the plain key.
_brm_term_key_value(value::NamedColumn) = name(value)
_brm_term_key_value(value::Tuple) = map(_brm_term_key_value, value)
_brm_term_key_value(value) = value
function _brm_term_full_key(term::ExprColumn)
    kwargs = getkwargs(term)
    isempty(kwargs) && return _brm_prepared_term_key(term)
    spelled = sort!([string(k, "=", repr(_brm_term_key_value(v))) for (k, v) in pairs(kwargs)])
    Symbol(chop(string(_brm_prepared_term_key(term))), ";", join(spelled, ","), ")")
end

# A qualifier compares with the keyword's effective value, so the default
# `cov=:exp_quad` names an `hsgp(t)` that does not spell it. The parsers own
# the defaults.
_brm_term_effective_keyword(term, key) = get(getkwargs(term), key, nothing)
function _brm_term_effective_keyword(term::ExprColumn{F}, key) where {F<:Union{typeof(gp),typeof(hsgp)}}
    label = nameof(getf(term))
    key === :cov && return _brm_gp_cov(getkwargs(term), label)
    key === :iso && return _brm_gp_iso(getkwargs(term), label)
    get(getkwargs(term), key, nothing)
end

function _brm_term_matches_qualifiers(term::ExprColumn, qualifiers)
    all(pairs(qualifiers)) do (key, value)
        held = _brm_term_effective_keyword(term, key)
        held isa Tuple && value isa Tuple ?
            length(held) == length(value) && all(map(==, held, value)) :
            held == value
    end
end

"""The resolved priors of `term` in one predictor's resolved-term map."""
function _brm_term_override(per_target, term)
    term isa ExprColumn || return get(per_target, _brm_prepared_term_key(term), nothing)
    full = _brm_term_full_key(term)
    haskey(per_target, full) && return per_target[full]
    get(per_target, _brm_prepared_term_key(term), nothing)
end

# ---- the HSGP validity floor inside its own length-scale prior --------------
#
# `length_scale(:, hsgp(x)) ~ Uniform(length_scale_floor, 24.0)` keeps BRM's
# derived validity floor as the lower end of an explicit prior (decision
# 1fk9b29). Resolution replaces the name with this marker; each backend lowers
# it to the term's own floor data, so a fresh `reprocess` re-derives it.
struct _BRMLengthScaleFloor end
const _BRM_LENGTH_SCALE_FLOOR = :length_scale_floor
Base.show(io::IO, ::_BRMLengthScaleFloor) = print(io, _BRM_LENGTH_SCALE_FLOOR)

_brm_substitute_floor(value, spelling) = value
function _brm_substitute_floor(value::NamedColumn, spelling)
    name(value) === _BRM_LENGTH_SCALE_FLOOR || return value
    parent(value) isa DataColumn && error(
        "BRM preparation: `$spelling` names `$(_BRM_LENGTH_SCALE_FLOOR)`, which " *
        "is both a data column and the HSGP validity-floor marker; rename the column")
    _BRMLengthScaleFloor()
end
function _brm_substitute_floor(value::ExprColumn, spelling)
    args = map(arg -> _brm_substitute_floor(arg, spelling), getargs(value))
    kwargs = getkwargs(value)
    rebuilt = NamedTuple{keys(kwargs)}(
        map(arg -> _brm_substitute_floor(arg, spelling), values(kwargs)))
    ExprColumn(getf(value), args...; rebuilt...)
end
_brm_substitute_floor(value::Tuple, spelling) =
    map(arg -> _brm_substitute_floor(arg, spelling), value)
_brm_substitute_floor(value::NamedTuple, spelling) =
    NamedTuple{keys(value)}(_brm_substitute_floor(values(value), spelling))

_brm_has_floor_marker(::_BRMLengthScaleFloor) = true
_brm_has_floor_marker(value::ExprColumn) =
    any(_brm_has_floor_marker, getargs(value)) ||
    any(_brm_has_floor_marker, values(getkwargs(value)))
_brm_has_floor_marker(_value) = false

# The numeric upper end of a length-scale prior whose lower end is the floor
# marker: `Uniform(length_scale_floor, b)` or
# `truncated(D; lower=length_scale_floor, upper=b)`; `nothing` otherwise.
_brm_floor_prior_upper(_prior) = nothing
_brm_floor_prior_upper(prior::ExprColumn) =
    _brm_floor_prior_upper(getf(prior), getargs(prior), getkwargs(prior))
_brm_floor_prior_upper(_f, _args, _kwargs) = nothing
_brm_floor_prior_upper(::Type{<:Uniform}, args, _kwargs) =
    length(args) == 2 && first(args) isa _BRMLengthScaleFloor &&
        last(args) isa Real ? Float64(last(args)) : nothing
_brm_floor_prior_upper(::typeof(truncated), _args, kwargs) =
    get(kwargs, :lower, nothing) isa _BRMLengthScaleFloor &&
        get(kwargs, :upper, nothing) isa Real ? Float64(kwargs.upper) : nothing

# A floor-marked prior with an upper bound at or below the fitted floor
# declares `real<lower=floor, upper=b>` with no admissible value: Stan fails
# only at initialization and RK with an unrelated domain error. The fitted
# floor is data, so this is checked wherever it is computed (decision 0ddmtf8),
# like the literal `Uniform(a, b)` bounds `_sb_gp_scale_prior` already checks.
function _brm_check_hsgp_floor_upper(prior, floor, spelling, geometry, remedy;
                                     prefix="BRM preparation")
    upper = _brm_floor_prior_upper(prior)
    isnothing(upper) || _brm_check_hsgp_floor_below(
        floor, upper, "$spelling ~ $prior", geometry, remedy; prefix)
    nothing
end

# `statement` is the authored prior; a fresh replay re-derives `floor` and
# re-checks it against the recorded `upper`.
function _brm_check_hsgp_floor_below(floor, upper, statement, geometry, remedy;
                                     prefix="BRM preparation")
    floors = floor isa Real ? (floor,) : Tuple(floor)
    axis = findfirst(>=(upper), floors)
    isnothing(axis) && return nothing
    which = length(floors) == 1 ? "fitted length-scale floor" :
        "fitted length-scale floor of axis $axis"
    error("$prefix: `$statement`: the $which ",
          "$(round(floors[axis]; sigdigits=4)) ($geometry) is not below the ",
          "upper bound $upper, so the length scale has no admissible value; ",
          "$remedy, or raise the upper bound")
end

_brm_hsgp_floor_values(values) = join(string.(values), ", ")
_brm_hsgp_floor_values(values::Tuple{Any}) = string(only(values))
function _brm_hsgp_floor_geometry(K, c, fits, explicit_domain::Bool)
    L = _brm_hsgp_floor_values(Tuple(last(fit) for fit in fits))
    k = _brm_hsgp_floor_values(Tuple(K))
    explicit_domain ? "k=$k, domain half-width L=$L" :
        "k=$k, c=$(_brm_hsgp_floor_values(Tuple(c))), L=$L"
end
_brm_hsgp_floor_remedy(explicit_domain::Bool) =
    explicit_domain ? "raise k, narrow the domain" : "raise k, lower c"
_brm_hsgp_periodic_floor_geometry(K, period) =
    "k=$(_brm_hsgp_floor_values(Tuple(K))), period=$period"
const _BRM_HSGP_PERIODIC_FLOOR_REMEDY = "raise k"

function _brm_floor_prior_spec(spec, term, slot, spelling)
    substituted = _brm_substitute_floor(spec.expression, spelling)
    _brm_has_floor_marker(substituted) || return spec
    slot.class === :term_length_scale && getf(term) === hsgp || error(
        "BRM preparation: `$spelling` uses `$(_BRM_LENGTH_SCALE_FLOOR)`, which " *
        "names an `hsgp` term's validity floor and is only valid in its own " *
        "`length_scale(...)` prior")
    merge(spec, (; expression=substituted,
                 arguments=_brm_substitute_floor(spec.arguments, spelling),
                 keywords=_brm_substitute_floor(spec.keywords, spelling)))
end

_brm_term_slot_ambiguity(_callable) = "this term has several parameter components"
_brm_term_slot_ambiguity(::typeof(t2)) = "a tensor smooth has three independent smoothing scales"
function _brm_resolve_term_slot(term, spec; prefix="BRM preparation")
    callable = getf(term)
    spelling = _brm_term_prior_spelling(spec)
    slots = filter(slot -> slot.class === spec.class, _brm_term_prior_slots(callable))
    isempty(slots) && error(
        "$prefix: `$spelling` — `$(nameof(callable))` has no " *
        "$(_brm_term_prior_class(Val(spec.class))) to configure")
    matches = filter(slot -> slot.component === spec.component, slots)
    if isempty(matches)
        isnothing(spec.component) && error(
            "$prefix: `$spelling` is ambiguous — $(_brm_term_slot_ambiguity(callable)). " *
            "Name one of " * join(("`$(slot.component)`" for slot in slots), ", ") * ".")
        all(slot -> isnothing(slot.component), slots) && error(
            "$prefix: `$spelling` names a component, but this parameter takes no component slot")
        error("$prefix: `$spelling` names no penalty block or parameter component; " *
              "valid components are " * join(("`$(slot.component)`" for slot in slots), ", "))
    end
    only(matches)
end

Base.@nospecializeinfer function _brm_term_address_map(@nospecialize(brmi::BRMI), target::Symbol)
    result = Dict{Symbol,Vector{Any}}()
    _brm_is_prior_declaration(brmi, target) && return result
    operation = linear_predictor_op(brmi, target)
    isnothing(operation) && return result
    for term in _brm_additive_terms(last(getargs(operation, 2)))
        term isa ExprColumn || continue
        isempty(_brm_term_prior_slots(getf(term))) && continue
        push!(get!(result, _brm_prepared_term_key(term), Any[]), term)
    end
    result
end

"""Resolve and account for every term-prior statement before backend lowering."""
Base.@nospecializeinfer function _brm_resolve_term_priors(@nospecialize(brmi::BRMI); prefix="BRM preparation")
    specs = term_priors(brmi)
    result = Dict{Symbol,Dict{Symbol,Dict{Symbol,Any}}}()
    isempty(specs) && return result
    targets = Symbol[predictor.name for predictor in linear_predictors(brmi)]
    addresses = Dict(target => _brm_term_address_map(brmi, target) for target in targets)
    for spec in specs
        spelling = _brm_term_prior_spelling(spec)
        candidates = isnothing(spec.predictor) ? targets : [spec.predictor]
        carrying = [target for target in candidates
                    if haskey(get(addresses, target, Dict()), spec.term)]
        if isempty(carrying)
            location = isnothing(spec.predictor) ? "any linear predictor" : "`$(spec.predictor)`"
            error("$prefix: `$spelling` matches no `$(spec.term)` term in $location")
        end
        # A qualified address reaches only the terms whose keywords agree.
        qualifiers = get(spec, :qualifiers, (;))
        matching(target) = filter(term -> _brm_term_matches_qualifiers(term, qualifiers),
                                  addresses[target][spec.term])
        reached = filter(target -> !isempty(matching(target)), carrying)
        isempty(reached) && error(
            "$prefix: `$spelling` matches none of the `$(spec.term)` terms in " *
            join(("`$target`" for target in carrying), ", ") *
            ": no such term spells those keyword values")
        for target in reached
            shared = addresses[target][spec.term]
            terms = matching(target)
            length(terms) == 1 || error(
                "$prefix: `$spelling` is ambiguous — `$target` carries $(length(terms)) " *
                "terms spelled `$(spec.term)`; qualify the address with a keyword " *
                "that tells them apart, as in `hsgp(x; cov=:periodic)`")
            term = only(terms)
            slot = _brm_resolve_term_slot(term, spec; prefix)
            resolved = _brm_floor_prior_spec(spec, term, slot, spelling)
            key = length(shared) > 1 ? _brm_term_full_key(term) : spec.term
            per_term = get!(get!(result, target, Dict{Symbol,Dict{Symbol,Any}}()),
                            key, Dict{Symbol,Any}())
            held = get(per_term, slot.name, nothing)
            rank = _brm_term_prior_rank(resolved)
            if isnothing(held) || rank > _brm_term_prior_rank(held.spec)
                per_term[slot.name] = (; spec=resolved, term)
            elseif rank == _brm_term_prior_rank(held.spec)
                error("$prefix: `$spelling` and `$(_brm_term_prior_spelling(held.spec))` " *
                      "are equally specific and both set the same parameter of `$(spec.term)` in `$target`")
            end
        end
    end
    result
end

_brm_normalize_simplex_prior(prior::ExprColumn, n::Int) =
    _brm_normalize_simplex_prior(getf(prior), prior, n)
_brm_normalize_simplex_prior(_constructor, prior, _n) = prior
function _brm_normalize_simplex_prior(::Type{<:Dirichlet}, prior, n)
    isempty(getkwargs(prior)) || error("BRM preparation: Dirichlet concentration shorthand does not accept keywords")
    args = getargs(prior)
    isempty(args) && error("BRM preparation: simplex Dirichlet prior needs a concentration")
    if length(args) != 1
        length(args) == n || error(
            "BRM preparation: simplex Dirichlet prior expects either one concentration or $n of them")
        alpha = collect(args)
    else
        alpha = only(args)
    end
    concentration = _brm_simplex_concentration_expression(alpha, n)
    ExprColumn(getf(prior), concentration)
end

function _brm_simplex_concentration_expression(alpha::Real, n)
    isfinite(alpha) && alpha > 0 || error("BRM preparation: simplex concentrations must be positive")
    fill(Float64(alpha), n)
end
function _brm_simplex_concentration_expression(alpha::AbstractVector, n)
    length(alpha) == n || error("BRM preparation: simplex prior has wrong dimension")
    all(value -> !(value isa Real) || (isfinite(value) && value > 0), alpha) || error(
        "BRM preparation: simplex concentrations must be positive")
    map(value -> value isa Real ? Float64(value) : value, alpha)
end
function _brm_simplex_concentration_expression(alpha::NamedColumn, n)
    backing = parent(alpha)
    if backing isa DataColumn && parent(backing) isa AbstractVector
        _brm_simplex_concentration_expression(parent(backing), n)
        return alpha
    end
    ExprColumn(fill, alpha, n)
end
_brm_simplex_concentration_expression(alpha, n) = ExprColumn(fill, alpha, n)

# The mathematical parameter is a simplex. Keep the exact prior density and
# RNG; its coordinate transform is supplied by the backend's simplex adapter.
struct _BRMSimplexPrior{D} <: ContinuousMultivariateDistribution
    base::D
    n::Int
end
function _brm_simplex_prior(prior::ContinuousMultivariateDistribution, n::Int)
    length(prior) == n || error("BRM preparation: simplex prior has wrong dimension")
    _BRMSimplexPrior(prior, n)
end
_brm_simplex_prior(prior, n::Int) = error(
    "BRM preparation: simplex prior must be a continuous multivariate distribution, got $(typeof(prior))")
Base.length(d::_BRMSimplexPrior) = d.n
Base.size(d::_BRMSimplexPrior) = (d.n,)
Base.eltype(::_BRMSimplexPrior) = Float64
Distributions.insupport(d::_BRMSimplexPrior, x::AbstractVector) =
    length(x) == d.n && all(>=(0), x) && isapprox(sum(x), one(sum(x))) &&
    insupport(d.base, x)
Distributions._logpdf(d::_BRMSimplexPrior, x::AbstractVector) =
    insupport(d, x) ? logpdf(d.base, x) : oftype(sum(x), -Inf)
function Random.rand(rng::AbstractRNG, d::_BRMSimplexPrior)
    draw = rand(rng, d.base)
    insupport(d, draw) || error("BRM preparation: configured simplex prior drew a value outside the simplex")
    draw
end
