# src/nlme_view.jl — the nonlinear-mixed-effects reading of an RK-lowered model.
#
# NLME estimators (posthoc/EBE, FOCEI, SAEM; the separate NLMEEstimation
# package) see a population model as: population coordinates θ, subject-level
# covariance coordinates Ω, other scalars σ, and one random-effect block η_i per
# subject, with subjects conditionally independent given (θ, Ω, σ). The RK
# emitter already files one semantic record per sampled declaration it can
# address (`_rk_coordinate_record!`, shared with the cross-backend coordinate
# transport), so the partition is read from those records, never from
# coordinate positions or name guessing. Every packed coordinate must be claimed
# by exactly one record; anything unclaimed or outside the per-subject structure
# is refused by name.

"""
    BRMNLMEViewError <: Exception

A model that has no nonlinear-mixed-effects reading in [`brm_nlme_view`](@ref):
no subject-level random effects, more than one grouping factor, a non-plain
grouping (`gr(...)`, `mm(...)`), a shared latent quantity (HSGP weights, imputed
values, submodel coordinates), or a coordinate the semantic inventory does not
cover. `message` names the offending declaration or coordinates.
"""
struct BRMNLMEViewError <: Exception
    message::String
end
Base.showerror(io::IO, err::BRMNLMEViewError) = print(io, err.message)
_brm_nlme_error(message) = throw(BRMNLMEViewError(message))

"""
    BRMNLMEView

The nonlinear-mixed-effects partition of one [`RKBRMI`](@ref), returned by
[`brm_nlme_view`](@ref). All positions index the backend's packed
unconstrained coordinate vector (`coordinates`, in packed order).

- `group` — the subject grouping column; `levels[i]` is subject `i`'s label,
  in the RK level order.
- `role[c]` — `:population` (population coefficients, including categorical
  contrasts, monotonic simplexes, ordinal thresholds and vector parameters),
  `:scalar` (other sampled scalars such as residual scales),
  `:subject_scale` / `:subject_correlation` (the random-effect scales and
  Cholesky correlation factors, i.e. Ω), or `:subject_effect` /
  `:subject_mixing` (subject `subject[c]`'s standardized draw or Student-t
  mixing weight).
- `subject_coordinates[i]` — subject `i`'s block, in a fixed order shared by
  every subject: for each random-effect block, its standardized draws in margin
  order, then its mixing weight when the block is Student-t.
- `block_margins[k]` — `(; block, predictor, coefficient)` for position `k` of
  every subject block (`coefficient === :mixing` for a mixing weight).
- `mu_references` — one `(; predictor, coefficient, link, population, subject)`
  per subject margin whose predictor also carries a population coefficient of
  the same name: `population` is that coefficient's packed position and
  `subject` the margin's position inside a subject block. The individual
  parameter is `link⁻¹(Xβ + Zη)` on the predictor's link scale.
- `rows[i]` — the rows of the grouping column that belong to subject `i`.

Subject draws are standardized (`η = diag(scale) L z`, `z ~ N(0, I)`): the
subject block holds `z`, and Ω is read through the scale and correlation
coordinates. Because the map from `z` to `η` is linear given Ω, Laplace and
FOCE objectives agree in either coordinate system.
"""
struct BRMNLMEView
    group::Symbol
    levels::Vector{Any}
    coordinates::Vector{Symbol}
    role::Vector{Symbol}
    subject::Vector{Int}
    subject_coordinates::Vector{Vector{Int}}
    block_margins::Vector{NamedTuple}
    mu_references::Vector{NamedTuple}
    rows::Vector{Vector{Int}}
end

Base.show(io::IO, v::BRMNLMEView) = print(io, "BRMNLMEView(",
    length(v.levels), " subjects of `", v.group, "`, ",
    length(v.block_margins), " coordinates per subject, ",
    count(==(:population), v.role), " population, ",
    count(r -> r === :subject_scale || r === :subject_correlation, v.role),
    " Ω and ", count(==(:scalar), v.role), " scalar coordinates)")

# The coordinate records this view reads. Kernel (`kernel(...)` / panel) plans
# have no semantic inventory yet; the refusal names that boundary instead of
# the transport's wording.
_brm_nlme_records(plan::Union{_RKStructuralPlan,_RKValuePlan,_RKHeldOutPlan}) =
    _rk_coordinate_records(plan)
_brm_nlme_records(plan) = _brm_nlme_error(
    "brm_nlme_view: an RK plan of kind `$(nameof(typeof(plan)))` has no " *
    "semantic coordinate inventory, so its NLME partition cannot be read. " *
    "The legacy panel `kernel(...)` route is not covered; `@plate for` cells " *
    "lower to value plans, which are.")

_brm_nlme_owned(name::Symbol, declaration::Symbol) = name === declaration ||
    startswith(String(name), string(declaration, "."))

"""
    brm_nlme_view(backend::RKBRMI) -> BRMNLMEView

Read the nonlinear-mixed-effects partition of an RK-lowered model: population
coordinates, subject-level covariance coordinates, other scalars, and one
random-effect block per subject. See [`BRMNLMEView`](@ref).

Refuses with [`BRMNLMEViewError`](@ref) when the model has no such reading:
no random effects, random effects on more than one grouping column, a non-plain
grouping, a shared latent quantity, or any coordinate the semantic inventory
does not cover.
"""
function brm_nlme_view(backend::RKBRMI)
    names = Vector{Symbol}(_rk_layout_coordinate_names(backend))
    position = Dict{Symbol,Int}(n => i for (i, n) in enumerate(names))
    records = _brm_nlme_records(backend.plan)
    ranefs = [r for r in records if r.kind === :ranef]
    isempty(ranefs) && _brm_nlme_error(
        "brm_nlme_view: the model has no subject-level random effects; an NLME " *
        "reading needs at least one `(... | group)` block")
    groups = unique(r.group for r in ranefs)
    length(groups) == 1 || _brm_nlme_error(
        "brm_nlme_view: random effects use $(length(groups)) grouping columns " *
        "($(join(string.(groups), ", "))); subjects are conditionally " *
        "independent only under a single grouping")
    group = only(groups)
    levels = Any[_rk_grouping_levels(backend.plan.columns[group])...]
    n = length(levels)
    role = fill(:unclaimed, length(names))
    subject = zeros(Int, length(names))
    claim!(i, r) = (role[i] === :unclaimed || error(
        "brm_nlme_view: internal: coordinate `$(names[i])` is claimed twice " *
        "($(role[i]) and $r)"); role[i] = r; nothing)
    function claim_declaration!(declaration, r)
        hits = [i for (i, name) in enumerate(names) if _brm_nlme_owned(name, declaration)]
        isempty(hits) && error("brm_nlme_view: internal: declaration " *
            "`$declaration` owns no packed coordinate")
        foreach(i -> claim!(i, r), hits)
        hits
    end
    coordinate(name) = get(position, name) do
        error("brm_nlme_view: internal: coordinate `$name` is not packed")
    end
    blocks = [Int[] for _ in 1:n]
    block_margins = NamedTuple[]
    population = Dict{Tuple{Symbol,Symbol},Int}()
    for record in records
        kind = record.kind
        if kind === :ranef
            for declaration in (isnothing(record.scales) ? (record.scale,) : record.scales)
                claim_declaration!(declaration, :subject_scale)
            end
            isnothing(record.L) || claim_declaration!(record.L, :subject_correlation)
            mixing = get(record, :mixing, nothing)
            block = (; group=record.group, id=record.id)
            for (k, (predictor, coefficient)) in enumerate(record.margins)
                push!(block_margins, (; block, predictor, coefficient))
                for g in 1:n
                    i = coordinate(Symbol(record.z, ".", g, ".", k))
                    claim!(i, :subject_effect)
                    subject[i] = g
                    push!(blocks[g], i)
                end
            end
            if !isnothing(mixing)
                push!(block_margins, (; block, predictor=:none, coefficient=:mixing))
                for g in 1:n
                    i = coordinate(Symbol(mixing, ".", g))
                    claim!(i, :subject_mixing)
                    subject[i] = g
                    push!(blocks[g], i)
                end
            end
        elseif kind === :population
            index = get(record, :index, ())
            if isempty(index)
                hits = claim_declaration!(record.declaration, :population)
                length(hits) == 1 &&
                    (population[(record.predictor, record.coefficient)] = only(hits))
            else
                i = coordinate(Symbol(record.declaration, ".", join(index, ".")))
                claim!(i, :population)
                population[(record.predictor, record.coefficient)] = i
            end
        elseif kind in (:population_block, :monotonic, :threshold_coefficients, :vector)
            claim_declaration!(record.declaration, :population)
        elseif kind === :scalar
            claim_declaration!(record.declaration, :scalar)
        elseif kind === :hsgp
            _brm_nlme_error(
                "brm_nlme_view: predictor `$(record.predictor)` carries the shared " *
                "Gaussian-process term `$(record.declaration)`; its latent weights are " *
                "shared across subjects, which an NLME reading does not cover")
        elseif kind === :missing_value || kind === :submodel
            _brm_nlme_error(
                "brm_nlme_view: `$(record.target)` carries $(kind === :missing_value ?
                    "imputed values" : "submodel coordinates") that are not part of " *
                "the per-subject structure; an NLME reading does not cover them")
        else
            _brm_nlme_error("brm_nlme_view: coordinate record kind `$kind` has no " *
                "NLME role")
        end
    end
    unclaimed = names[role .=== :unclaimed]
    isempty(unclaimed) || _brm_nlme_error(
        "brm_nlme_view: $(length(unclaimed)) coordinate(s) are not covered by the " *
        "semantic inventory: $(join(string.(unclaimed), ", ")). Random effects " *
        "must use a plain `(... | group)` grouping, not `gr(...)` or `mm(...)`.")
    mu_references = NamedTuple[]
    for (k, m) in enumerate(block_margins)
        m.coefficient === :mixing && continue
        pop = get(population, (m.predictor, m.coefficient), 0)
        pop == 0 && continue
        push!(mu_references, (; predictor=m.predictor, coefficient=m.coefficient,
            link=_rk_predictor_link(parent(backend), m.predictor),
            population=pop, subject=k))
    end
    labels = map(_brm_transport_level, backend.plan.columns[group])
    rows = [findall(x -> isequal(x, _brm_transport_level(level)), labels) for level in levels]
    BRMNLMEView(group, levels, names, role, subject, blocks, block_margins,
        mu_references, rows)
end
