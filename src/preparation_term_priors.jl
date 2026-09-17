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
_brm_term_prior_label(::Val{:term_sd}) = "sd"
_brm_term_prior_label(::Val{:term_ar}) = "ar"
_brm_term_prior_label(::Val{:term_length_scale}) = "length_scale"
_brm_term_prior_label(::Val{:term_simplex}) = "simplex"
_brm_term_prior_label(::Val{:term_latent}) = "latent"
function _brm_term_prior_spelling(spec)
    head = _brm_term_prior_label(Val(spec.class))
    lp = isnothing(spec.predictor) ? ":" : string(spec.predictor)
    component = isnothing(spec.component) ? "" : ", $(spec.component)"
    "$head($lp, $(spec.term)$component)"
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

function _brm_term_address_map(brmi::BRMI, target::Symbol)
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
function _brm_resolve_term_priors(brmi::BRMI; prefix="BRM preparation")
    specs = term_priors(brmi)
    result = Dict{Symbol,Dict{Symbol,Dict{Symbol,Any}}}()
    isempty(specs) && return result
    targets = Symbol[predictor.name for predictor in linear_predictors(brmi)]
    addresses = Dict(target => _brm_term_address_map(brmi, target) for target in targets)
    for spec in specs
        spelling = _brm_term_prior_spelling(spec)
        candidates = isnothing(spec.predictor) ? targets : [spec.predictor]
        reached = [target for target in candidates
                   if haskey(get(addresses, target, Dict()), spec.term)]
        if isempty(reached)
            location = isnothing(spec.predictor) ? "any linear predictor" : "`$(spec.predictor)`"
            error("$prefix: `$spelling` matches no `$(spec.term)` term in $location")
        end
        for target in reached
            terms = addresses[target][spec.term]
            length(terms) == 1 || error(
                "$prefix: `$spelling` is ambiguous — `$target` carries $(length(terms)) " *
                "terms spelled `$(spec.term)`")
            term = only(terms)
            slot = _brm_resolve_term_slot(term, spec; prefix)
            per_term = get!(get!(result, target, Dict{Symbol,Dict{Symbol,Any}}()),
                            spec.term, Dict{Symbol,Any}())
            held = get(per_term, slot.name, nothing)
            rank = _brm_term_prior_rank(spec)
            if isnothing(held) || rank > _brm_term_prior_rank(held.spec)
                per_term[slot.name] = (; spec, term)
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
