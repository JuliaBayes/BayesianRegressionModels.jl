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

# ── Evaluation for NLME estimators ───────────────────────────────────────────
#
# Subjects are conditionally independent given (θ, Ω, σ), so one evaluation of
# the joint RK program serves all subjects in lockstep: the pointwise densities
# summed by subject give every subject's conditional log-likelihood, and the
# joint gradient restricted to a subject's block is that subject's conditional
# gradient. This layer therefore offers all-subject evaluation only: RK has no
# per-subject evaluation, and a per-subject entry point would cost a full pass
# per call, O(N²) per sweep over subjects.

# The packed unconstrained point at which every random-effect block has unit
# scales and an identity correlation factor, so that each block's standardized
# draws ARE the natural-scale random effects η_i. Verified on the constrained
# values rather than assumed from the transforms.
function _brm_nlme_unit_point(backend::RKBRMI, view::BRMNLMEView)
    u = zeros(Float64, length(view.coordinates))
    values = _rk_constrained_values(backend, u)
    for record in _brm_nlme_records(backend.plan)
        record.kind === :ranef || continue
        for declaration in (isnothing(record.scales) ? (record.scale,) : record.scales)
            scale = _brm_rk_declaration_value(values, declaration)
            all(isequal(1), scale) || _brm_nlme_error(
                "brm_nlme_model: random-effect scale `$declaration` is $(scale), not 1, " *
                "at the zero unconstrained point; its transform has no unit point here")
        end
        isnothing(record.L) && continue
        L = _brm_rk_declaration_value(values, record.L)
        L == I || _brm_nlme_error(
            "brm_nlme_model: correlation factor `$(record.L)` is not the identity at " *
            "the zero unconstrained point")
    end
    u
end

# Which subject each pointwise density element belongs to, per observed name:
# one element per row of the grouping column (flat rows on the grouping axis,
# or one in-cell array per plate cell), or a `ragged(y, g)` axis whose labels
# are subject labels. Anything else is refused rather than attributed by
# position.
_brm_nlme_ragged_labels(plan, name) = nothing
_brm_nlme_ragged_labels(plan::_RKHeldOutPlan, name) =
    _brm_nlme_ragged_labels(plan.parent, name)
function _brm_nlme_ragged_labels(plan::_RKValuePlan, name)
    for o in plan.observations
        o.name === name || continue
        lhs = o.lhs
        lhs isa ExprColumn && getf(lhs) === ragged || return nothing
        return parent(parent(getargs(lhs)[2]))
    end
    nothing
end

function _brm_nlme_attribution(backend::RKBRMI, view::BRMNLMEView, fields)
    L = sum(length, view.rows)
    row_subject = zeros(Int, L)
    for (i, rows) in enumerate(view.rows), r in rows
        row_subject[r] = i
    end
    any(iszero, row_subject) && error(
        "brm_nlme_model: internal: a row of `$(view.group)` belongs to no subject")
    index = Dict(_brm_transport_level(l) => i for (i, l) in enumerate(view.levels))
    attribution = Pair{Symbol,Vector{Int}}[]
    for (name, values) in pairs(fields)
        m = length(values)
        if m == L
            push!(attribution, name => row_subject)
            continue
        end
        labels = _brm_nlme_ragged_labels(backend.plan, name)
        labels === nothing && _brm_nlme_error(
            "brm_nlme_model: observation `$name` has $m pointwise densities, neither " *
            "one per row of `$(view.group)` ($L) nor a `ragged(..., group)` axis; its " *
            "densities cannot be attributed to subjects")
        length(labels) == m || _brm_nlme_error(
            "brm_nlme_model: observation `$name` has $m pointwise densities but its " *
            "ragged grouping column has $(length(labels)) labels")
        subjects = map(labels) do label
            get(index, _brm_transport_level(label)) do
                _brm_nlme_error("brm_nlme_model: label $(repr(label)) of observation " *
                    "`$name` is not a level of `$(view.group)`")
            end
        end
        push!(attribution, name => subjects)
    end
    attribution
end

# Implemented by `BayesianRegressionModelsReactiveKernelsExt`: the prepared
# `:pointwise` query of a backend and its evaluation at a packed point.
function _rk_pointwise_query end
function _rk_pointwise_values end

"""
    BRMNLMEModel

An RK-lowered population model prepared for NLME estimators, returned by
[`brm_nlme_model`](@ref). Its coordinates follow the NLME convention:

- `θ` — the population coordinates (`view.role .=== :population`), in packed order;
- `σ` — the other sampled scalars (`view.role .=== :scalar`), in packed order;
- `η_i` — subject `i`'s random effects on the natural scale, one column of a
  `neta × nsubjects` matrix, in the subject-block order of [`BRMNLMEView`](@ref).

All three are in the model's unconstrained coordinates. The random-effect
covariance Ω is not a model coordinate here: estimators own it, and the model
evaluates the conditional likelihood `log p(y_i | θ, σ, η_i)` without the
random-effect prior.
"""
struct BRMNLMEModel{B<:RKBRMI,P,Q}
    view::BRMNLMEView
    backend::B
    problem::P
    pointwise::Q
    attribution::Vector{Pair{Symbol,Vector{Int}}}
    theta::Vector{Int}
    sigma::Vector{Int}
    eta_blocks::Vector{Int}
    unit::Vector{Float64}
end

Base.show(io::IO, m::BRMNLMEModel) = print(io, "BRMNLMEModel(",
    length(m.view.levels), " subjects, θ: ", length(m.theta), ", σ: ",
    length(m.sigma), ", η: ", sum(m.eta_blocks), " in blocks ", m.eta_blocks, ")")

"""
    brm_nlme_model(backend::RKBRMI; ad_backend) -> BRMNLMEModel

Prepare an RK-lowered population model for NLME estimators: its
[`brm_nlme_view`](@ref) partition, the backend's log-density problem
(`rk_logdensity_problem(backend; ad_backend)`, first-order), its pointwise
densities, and the subject attribution of every observed density.

Refuses ([`BRMNLMEViewError`](@ref)) everything `brm_nlme_view` refuses, Student-t
random-effect blocks (the estimators own a Gaussian η prior), and observed
densities that cannot be attributed to subjects.
"""
function brm_nlme_model(backend::RKBRMI; ad_backend)
    view = brm_nlme_view(backend)
    any(==(:subject_mixing), view.role) && _brm_nlme_error(
        "brm_nlme_model: Student-t random-effect blocks are not covered; NLME " *
        "estimators own a Gaussian random-effect prior")
    unit = _brm_nlme_unit_point(backend, view)
    problem = rk_logdensity_problem(backend; ad_backend)
    pointwise = _rk_pointwise_query(backend)
    attribution = _brm_nlme_attribution(backend, view,
        _rk_pointwise_values(pointwise, unit))
    eta_blocks = Int[]
    previous = nothing
    for margin in view.block_margins
        margin.block == previous ? (eta_blocks[end] += 1) : push!(eta_blocks, 1)
        previous = margin.block
    end
    BRMNLMEModel(view, backend, problem, pointwise, attribution,
        findall(==(:population), view.role), findall(==(:scalar), view.role),
        eta_blocks, unit)
end

function _brm_nlme_point(m::BRMNLMEModel, θ, σ, H)
    n = length(m.view.levels)
    length(θ) == length(m.theta) || throw(DimensionMismatch(
        "θ has length $(length(θ)), the model has $(length(m.theta)) population coordinates"))
    length(σ) == length(m.sigma) || throw(DimensionMismatch(
        "σ has length $(length(σ)), the model has $(length(m.sigma)) scalar coordinates"))
    size(H) == (sum(m.eta_blocks), n) || throw(DimensionMismatch(
        "η matrix has size $(size(H)), the model needs ($(sum(m.eta_blocks)), $n)"))
    u = Vector{promote_type(Float64, eltype(θ), eltype(σ), eltype(H))}(m.unit)
    u[m.theta] .= θ
    u[m.sigma] .= σ
    for (i, block) in enumerate(m.view.subject_coordinates)
        u[block] .= view(H, :, i)
    end
    u
end

function _brm_nlme_sum_by_subject(m::BRMNLMEModel, fields)
    out = zeros(Float64, length(m.view.levels))
    for (name, subjects) in m.attribution
        values = getproperty(fields, name)
        for (k, i) in enumerate(subjects)
            out[i] += sum(values[k])
        end
    end
    out
end

"""
    brm_nlme_loglikelihoods(m::BRMNLMEModel, θ, σ, H) -> Vector{Float64}

Every subject's conditional log-likelihood `log p(y_i | θ, σ, η_i)`, with
`η_i = H[:, i]`, from one evaluation of the model. See [`BRMNLMEModel`](@ref)
for the coordinates.
"""
brm_nlme_loglikelihoods(m::BRMNLMEModel, θ, σ, H) =
    _brm_nlme_sum_by_subject(m, _rk_pointwise_values(m.pointwise, _brm_nlme_point(m, θ, σ, H)))

"""
    brm_nlme_loglikelihoods_and_gradients(m::BRMNLMEModel, θ, σ, H) -> (values, G)

[`brm_nlme_loglikelihoods`](@ref) together with `G[:, i] = ∇_{η_i} log p(y_i | θ, σ, η_i)`
for every subject, from one pointwise and one gradient evaluation. Subjects are
conditionally independent, so the joint gradient restricted to subject `i`'s
block is its conditional gradient once the standard-normal draw prior is removed.
"""
function brm_nlme_loglikelihoods_and_gradients(m::BRMNLMEModel, θ, σ, H)
    u = _brm_nlme_point(m, θ, σ, H)
    values = _brm_nlme_sum_by_subject(m, _rk_pointwise_values(m.pointwise, u))
    _, g = LogDensityProblems.logdensity_and_gradient(m.problem, u)
    G = Matrix{Float64}(undef, size(H))
    for (i, block) in enumerate(m.view.subject_coordinates)
        G[:, i] .= view(g, block) .+ view(H, :, i)
    end
    values, G
end
