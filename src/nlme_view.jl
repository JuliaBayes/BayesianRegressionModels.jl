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
- `mu_references` — one `(; predictor, coefficient, link, population, subject,
  design)` per population coefficient that can be absorbed into a subject
  margin. `population` is the coefficient's packed position, `subject` the
  margin's position in a subject block, and `design[i]` its multiplier for
  subject `i`. The conditional predictor is unchanged by adding `δ` to the
  population coefficient and subtracting `design[i] * δ` from that margin.
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

_brm_nlme_regression(plan::_RKStructuralPlan) = plan
_brm_nlme_regression(plan::_RKValuePlan) = plan.regression
_brm_nlme_regression(plan::_RKHeldOutPlan) = _brm_nlme_regression(plan.parent)

# Read the same scalar design columns that the RK affine emitter multiplies by
# its population coefficients. Other coefficient families need a separate
# coordinate-to-column record and are deliberately not inferred from names.
function _brm_nlme_population_column(plan::_RKStructuralPlan, predictor, coefficient, nrows)
    priors = [p for p in plan.population_priors if p.predictor === predictor &&
        p.addressee === coefficient]
    # MuReferencing shifts unconstrained θ additively. Bounded/positive priors
    # transform that coordinate before the predictor sees it, so their
    # population design cannot certify this invariance.
    length(priors) == 1 && only(priors).family in
        (:Normal, :Cauchy, :Laplace, :Logistic, :StudentT, :Flat) || return nothing
    specs = [term for p in plan.predictors if p.name === predictor
        for term in p.terms if term.addressee === coefficient &&
        term.kind in (:intercept, :continuous)]
    length(specs) == 1 || return nothing
    term = only(specs)
    term.kind === :intercept && return ones(Float64, nrows)
    values = plan.columns[only(term.columns)]
    length(values) == nrows && all(x -> x isa Real && isfinite(x), values) || return nothing
    Float64.(values)
end

# A population column is mu-referenced through a random slope when its actual
# emitted Z column is identical. A subject-constant column can instead be
# absorbed through an intercept, including a covariate slope without a random
# slope. Both cases preserve the whole predictor row by row.
function _brm_nlme_mu_references(backend, block_margins, population, rows, nrows)
    plan = _brm_nlme_regression(backend.plan)
    references = NamedTuple[]
    for ((predictor, coefficient), position) in sort!(collect(population); by=x -> last(x))
        column = _brm_nlme_population_column(plan, predictor, coefficient, nrows)
        column === nothing && continue
        link = _rk_predictor_link(parent(backend), predictor)
        # A matching random slope takes precedence, since it also works when
        # the observed covariate varies among a subject's rows.
        matched = false
        for (k, margin) in enumerate(block_margins)
            margin.predictor === predictor && margin.coefficient === coefficient || continue
            recipes = [r.z for bucket in plan.ranef_buckets
                if bucket.group === margin.block.group && bucket.id === margin.block.id
                for r in bucket.margins if r.predictor === predictor &&
                    r.coefficient === coefficient]
            length(recipes) == 1 || continue
            recipe = only(recipes)
            z = recipe.kind === :ones ? ones(Float64, nrows) :
                recipe.kind === :column ? get(plan.columns, recipe.column, nothing) : nothing
            z === nothing && continue
            length(z) == nrows && all(isequal.(column, z)) || continue
            push!(references, (; predictor, coefficient, link, population=position,
                subject=k, design=ones(Float64, length(rows))))
            matched = true
            break
        end
        matched && continue
        for (k, margin) in enumerate(block_margins)
            margin.predictor === predictor && margin.coefficient === :Intercept || continue
            values = Float64[]
            for subject_rows in rows
                isempty(subject_rows) && break
                value = column[first(subject_rows)]
                all(r -> isequal(column[r], value), subject_rows) || break
                push!(values, value)
            end
            length(values) == length(rows) || continue
            push!(references, (; predictor, coefficient, link, population=position,
                subject=k, design=values))
            break
        end
    end
    references
end

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
        elseif kind === :monotonic && get(record, :column, nothing) !== nothing
            # One predictor's column of a joint monotonic block: its simplex
            # column `<block>.simplex_incr.<j>.<t>` and its coefficient entry.
            t = string(record.column)
            prefix = string(record.declaration, ".simplex_incr.")
            for (i, name) in enumerate(names)
                parts = split(String(name), '.')
                startswith(String(name), prefix) && length(parts) == 4 &&
                    parts[end] == t && claim!(i, :population)
            end
            # The coefficient entry exists whether or not it pairs with Stan.
            beta = get(record, :beta_declaration, nothing)
            if beta !== nothing
                index = get(record, :beta_index, ())
                claim!(coordinate(isempty(index) ? beta :
                    Symbol(beta, ".", join(index, "."))), :population)
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
    labels = map(_brm_transport_level, backend.plan.columns[group])
    rows = [findall(x -> isequal(x, _brm_transport_level(level)), labels) for level in levels]
    mu_references = _brm_nlme_mu_references(backend, block_margins, population,
        rows, length(labels))
    BRMNLMEView(group, levels, names, role, subject, blocks, block_margins,
        mu_references, rows)
end

# ── Evaluation for NLME estimators ───────────────────────────────────────────
#
# Subjects are conditionally independent given (θ, Ω, σ). The lockstep path
# serves all subjects in one RK evaluation. A subject-indexed observation plate
# also permits a cell query that evaluates just one subject at a time.

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
# a `ragged(y, g)` axis whose labels are subject labels, or else one element
# per row of the grouping column (flat rows on the grouping axis, or one in-cell
# array per plate cell). Anything else is refused rather than attributed by
# position.
_brm_nlme_ragged_labels(plan, name) = nothing
_brm_nlme_ragged_labels(plan::_RKHeldOutPlan, name) =
    _brm_nlme_ragged_labels(plan.parent, name)
# A ragged join gathers its response cell by cell (`_rk_kernel_observed_layout`),
# so its densities follow the row partition, not the flat row order: the labels
# are gathered by the same partition.
function _brm_nlme_ragged_labels(plan::_RKValuePlan, name)
    for o in plan.observations
        o.name === name || continue
        lhs = o.lhs
        lhs isa ExprColumn && getf(lhs) === ragged || return nothing
        kernels = Tuple(a for a in plan.assignments if a isa _RKPreparedKernelAssignment)
        layout = _rk_kernel_observed_layout(o, kernels)
        return parent(parent(getargs(lhs)[2]))[brm_flatten_cells(layout.rows)]
    end
    nothing
end

_brm_nlme_gaussian_plan(plan::_RKHeldOutPlan) =
    isempty(plan.held_out) && _brm_nlme_gaussian_plan(plan.parent)
_brm_nlme_gaussian_plan(plan::_RKStructuralPlan) =
    !isempty(plan.responses) && all(o -> o.family === :gaussian && o.weights === nothing &&
        o.interval === nothing && o.mi_jobs === nothing, plan.responses)
function _brm_nlme_gaussian_plan(plan::_RKValuePlan)
    !isempty(plan.observations) && all(o -> begin
        o.distribution.callable in (Normal, StanBlocks.normal) &&
            o.modifier === nothing && o.weight === nothing &&
            o.missing_response === nothing
    end, plan.observations)
end
_brm_nlme_gaussian_plan(plan) = false

function _brm_nlme_observed(backend::RKBRMI, attribution, nsubjects)
    observed = [Float64[] for _ in 1:nsubjects]
    scalar_attribution = Pair{Symbol,Vector{Int}}[]
    for (name, subjects) in attribution
        # Value plans have already gathered ragged response values into the
        # pointwise query's order in their bound `columns[name]`.
        values = backend.plan.columns[name]
        length(values) == length(subjects) || error(
            "brm_nlme_model: response `$name` has $(length(values)) values for " *
            "$(length(subjects)) attributed pointwise entries")
        expanded = Int[]
        for (value, i) in zip(values, subjects)
            items = value isa AbstractArray ? vec(value) : (value,)
            for y in items
                y isa Real && isfinite(y) || _brm_nlme_error(
                    "brm_nlme_model: Gaussian response `$name` contains a non-finite " *
                    "or non-numeric observation $(repr(y))")
                push!(observed[i], Float64(y))
                push!(expanded, i)
            end
        end
        push!(scalar_attribution, name => expanded)
    end
    observed, scalar_attribution
end

function _brm_nlme_observation_argument(value, sizes, name, field)
    n = sum(sizes)
    values = if value isa Number
        fill(Float64(value), n)
    elseif value isa AbstractVector && length(value) == length(sizes)
        out = Float64[]
        sizehint!(out, n)
        for (part, size) in zip(value, sizes)
            if part isa Number
                append!(out, Iterators.repeated(Float64(part), size))
            elseif part isa AbstractArray
                length(part) == size || throw(DimensionMismatch(
                    "brm_nlme_model: `$name.$field` has $(length(part)) entries " *
                    "for a plate cell with $size observations"))
                append!(out, Float64.(vec(part)))
            else
                _brm_nlme_error(
                    "brm_nlme_model: `$name.$field` is neither a scalar nor an array " *
                    "in a plate cell")
            end
        end
        out
    else
        Float64.(brm_flatten_cells(value))
    end
    length(values) == n || throw(DimensionMismatch(
        "brm_nlme_model: `$name.$field` has $(length(values)) entries for $n observations"))
    values
end

function _brm_nlme_moments(m, arguments, tangents=())
    nsubjects = length(m.view.levels)
    neta = sum(m.eta_blocks)
    means = [Float64[] for _ in 1:nsubjects]
    variances = [Float64[] for _ in 1:nsubjects]
    mean_rows = [Vector{Float64}[] for _ in 1:nsubjects]
    variance_rows = [Vector{Float64}[] for _ in 1:nsubjects]
    for (name, subjects) in m.observation_attribution
        n = length(subjects)
        sizes = [value isa AbstractArray ? length(value) : 1
            for value in m.backend.plan.columns[name]]
        sum(sizes) == n || error(
            "brm_nlme_model: response `$name` has $(sum(sizes)) observed values " *
            "for $n attributed pointwise entries")
        field = getproperty(arguments, name)
        locations = _brm_nlme_observation_argument(field.location, sizes, name, :location)
        scales = _brm_nlme_observation_argument(field.scale, sizes, name, :scale)
        all(s -> isfinite(s) && s > 0, scales) || _brm_nlme_error(
            "brm_nlme_model: Gaussian response `$name` has a non-finite or " *
            "non-positive scale")
        dlocations = [_brm_nlme_observation_argument(getproperty(t, name).location,
            sizes, name, :location) for t in tangents]
        dscales = [_brm_nlme_observation_argument(getproperty(t, name).scale,
            sizes, name, :scale) for t in tangents]
        for (k, i) in enumerate(subjects)
            push!(means[i], locations[k])
            push!(variances[i], abs2(scales[k]))
            if !isempty(tangents)
                push!(mean_rows[i], [d[k] for d in dlocations])
                push!(variance_rows[i], [2 * scales[k] * d[k] for d in dscales])
            end
        end
    end
    for i in 1:nsubjects
        length(means[i]) == length(m.observed[i]) || error(
            "brm_nlme_model: subject $i has $(length(means[i])) observation moments " *
            "for $(length(m.observed[i])) observed values")
    end
    matrix(rows) = isempty(rows) ? zeros(Float64, 0, neta) :
        reduce(vcat, (permutedims(row) for row in rows))
    isempty(tangents) ? collect(zip(means, variances)) :
        [(means[i], variances[i], matrix(mean_rows[i]), matrix(variance_rows[i]))
            for i in 1:nsubjects]
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
        # A ragged axis is attributed by its labels even when its length
        # happens to equal the grouping column's.
        labels = _brm_nlme_ragged_labels(backend.plan, name)
        if labels === nothing
            m == L || _brm_nlme_error(
                "brm_nlme_model: observation `$name` has $m pointwise densities, neither " *
                "one per row of `$(view.group)` ($L) nor a `ragged(..., group)` axis; its " *
                "densities cannot be attributed to subjects")
            push!(attribution, name => row_subject)
            continue
        end
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
# `:pointwise` query of a backend and its evaluation at a packed point, and the
# sampler query that retains the pointwise densities of its reverse sweep, with
# `(value, gradient, pointwise)` from one evaluation.
function _rk_pointwise_query end
function _rk_pointwise_values end
function _rk_retained_pointwise_sampler end
function _rk_value_gradient_and_pointwise! end
function _rk_eta_hvp_query end
function _rk_eta_hvp end
function _rk_observation_query end
function _rk_observation_pushforward end
function _rk_observation_values end
function _rk_observation_value_and_pushforward end
function _rk_nlme_cell_query end
function _rk_nlme_cell_value end
function _rk_nlme_compact_cell_sampler end
function _rk_nlme_compact_cell_value_and_gradient! end
_brm_nlme_forward_backend(ad_backend) = nothing

# A cell query still addresses the original packed coordinate space. Overlay
# only the shared coordinates and the selected subject's η onto the read-only
# unit point; constructing a packed Vector would make each call scale with the
# number of other subjects.
struct _BRMNLMEPoint{T,S,E} <: AbstractVector{Float64}
    unit::Vector{Float64}
    theta_positions::Vector{Int}
    theta::T
    sigma_positions::Vector{Int}
    sigma::S
    eta_positions::Vector{Int}
    eta::E
end
Base.size(p::_BRMNLMEPoint) = size(p.unit)
Base.IndexStyle(::Type{<:_BRMNLMEPoint}) = IndexLinear()
function Base.getindex(p::_BRMNLMEPoint, j::Int)
    k = findfirst(==(j), p.theta_positions)
    k === nothing || return p.theta[k]
    k = findfirst(==(j), p.sigma_positions)
    k === nothing || return p.sigma[k]
    k = findfirst(==(j), p.eta_positions)
    k === nothing || return p.eta[k]
    p.unit[j]
end

# Only a subject-indexed observation plate has a single query cell containing
# exactly that subject's densities. A flat grouped observation, including a
# ragged join, remains available through lockstep evaluation but cannot use a
# cell as its conditional subject density.
function _brm_nlme_cell_rows(backend::RKBRMI, view::BRMNLMEView, attribution)
    plan = backend.plan
    isempty(attribution) && return nothing
    plan isa _RKValuePlan || return nothing
    plate_observations = Set(o.source for a in plan.assignments
        if a isa _RKPreparedKernelAssignment for o in a.observations)
    all(name -> name in plate_observations, first.(attribution)) || return nothing
    all(rows -> length(rows) == 1, view.rows) || return nothing
    n = length(view.levels)
    rows = [only(group_rows) for group_rows in view.rows]
    all(subjects -> length(subjects) == n &&
        all(i -> subjects[rows[i]] == i, 1:n), last.(attribution)) || return nothing
    rows
end

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

When every fitted response is an independent, unweighted Normal observation
supported by RK's `:observations` preset, `observed` holds subject observation
vectors in the same order as the prepared query. `observations_pushforward`
uses one colored direction per random-effect margin across all subjects.
"""
struct BRMNLMEModel{B<:RKBRMI,S,Q,V,O,J,C,G}
    view::BRMNLMEView
    backend::B
    sampler::S
    pointwise::Q
    eta_hvp::V
    attribution::Vector{Pair{Symbol,Vector{Int}}}
    theta::Vector{Int}
    sigma::Vector{Int}
    eta_blocks::Vector{Int}
    unit::Vector{Float64}
    observations_query::O
    observations_pushforward::J
    observed::Vector{Vector{Float64}}
    observation_attribution::Vector{Pair{Symbol,Vector{Int}}}
    cell_query::C
    cell_gradient::G
    cell_rows::Union{Nothing,Vector{Int}}
end

Base.show(io::IO, m::BRMNLMEModel) = print(io, "BRMNLMEModel(",
    length(m.view.levels), " subjects, θ: ", length(m.theta), ", σ: ",
    length(m.sigma), ", η: ", sum(m.eta_blocks), " in blocks ", m.eta_blocks, ")")

"""
    brm_nlme_model(backend::RKBRMI; ad_backend, observation_ad_backend=nothing) -> BRMNLMEModel

Prepare an RK-lowered population model for NLME estimators: its
[`brm_nlme_view`](@ref) partition, the backend's sampler query retaining its
pointwise densities (first-order, reverse-mode Enzyme: `ad_backend` must be
`AutoEnzyme(; mode=Enzyme.Reverse)`), a reverse-over-forward sampler HVP query,
its pointwise densities, and the subject attribution of every observed density.

`observation_ad_backend` selects the forward-mode backend for Gaussian moment
Jacobians. With the NLMEEstimation/Enzyme extension loaded, an `AutoEnzyme`
reverse backend automatically selects Enzyme forward mode. Other callers can
pass a forward backend explicitly.

When the observations belong to one subject-indexed `@plate for` cell each,
the model also prepares a cell query for
[`brm_nlme_loglikelihood`](@ref) and
[`brm_nlme_loglikelihood_and_gradient`](@ref).

Refuses ([`BRMNLMEViewError`](@ref)) everything `brm_nlme_view` refuses, Student-t
random-effect blocks (the estimators own a Gaussian η prior), and observed
densities that cannot be attributed to subjects.
"""
function brm_nlme_model(backend::RKBRMI; ad_backend, observation_ad_backend=nothing)
    view = brm_nlme_view(backend)
    any(==(:subject_mixing), view.role) && _brm_nlme_error(
        "brm_nlme_model: Student-t random-effect blocks are not covered; NLME " *
        "estimators own a Gaussian random-effect prior")
    unit = _brm_nlme_unit_point(backend, view)
    sampler = _rk_retained_pointwise_sampler(backend; ad_backend)
    pointwise = _rk_pointwise_query(backend)
    eta_hvp = _rk_eta_hvp_query(backend, unit)
    attribution = _brm_nlme_attribution(backend, view,
        _rk_pointwise_values(pointwise, unit))
    eta_blocks = Int[]
    previous = nothing
    for margin in view.block_margins
        margin.block == previous ? (eta_blocks[end] += 1) : push!(eta_blocks, 1)
        previous = margin.block
    end
    gaussian = _brm_nlme_gaussian_plan(backend.plan)
    observations_query = gaussian ? _rk_observation_query(backend) : nothing
    forward_backend = observation_ad_backend === nothing ?
        _brm_nlme_forward_backend(ad_backend) : observation_ad_backend
    observations_pushforward = gaussian && forward_backend !== nothing ?
        _rk_observation_pushforward(backend, view, unit, forward_backend) : nothing
    observed, observation_attribution = gaussian ?
        _brm_nlme_observed(backend, attribution, length(view.levels)) :
        (Vector{Vector{Float64}}(), Pair{Symbol,Vector{Int}}[])
    cell_rows = _brm_nlme_cell_rows(backend, view, attribution)
    cell_query = cell_rows === nothing ? nothing :
        _rk_nlme_cell_query(backend, Tuple(first.(attribution)))
    cell_gradient = if cell_rows === nothing
        nothing
    else
        selected = sort!(vcat(findall(==(:population), view.role),
            findall(==(:scalar), view.role), first(view.subject_coordinates)))
        (sampler=_rk_nlme_compact_cell_sampler(backend,
            Tuple(first.(attribution)), unit, selected), lock=ReentrantLock())
    end
    BRMNLMEModel(view, backend, sampler, pointwise, eta_hvp, attribution,
        findall(==(:population), view.role), findall(==(:scalar), view.role),
        eta_blocks, unit, observations_query, observations_pushforward,
        observed, observation_attribution, cell_query, cell_gradient, cell_rows)
end

function _brm_nlme_cell_point(m::BRMNLMEModel, i::Integer, θ, σ, η)
    rows = m.cell_rows
    rows === nothing && _brm_nlme_error(
        "brm_nlme_loglikelihood: per-subject evaluation requires a " *
        "subject-indexed `@plate for` observation; flat-row grouped and " *
        "ragged-join models use the lockstep methods")
    1 <= i <= length(rows) || throw(BoundsError(rows, i))
    length(θ) == length(m.theta) || throw(DimensionMismatch(
        "θ has length $(length(θ)), expected $(length(m.theta))"))
    length(σ) == length(m.sigma) || throw(DimensionMismatch(
        "σ has length $(length(σ)), expected $(length(m.sigma))"))
    block = m.view.subject_coordinates[i]
    length(η) == length(block) || throw(DimensionMismatch(
        "η has length $(length(η)), expected $(length(block))"))
    point = _BRMNLMEPoint(m.unit, m.theta, Vector{Float64}(θ), m.sigma,
        Vector{Float64}(σ), block, Vector{Float64}(η))
    point, rows[i]
end

"""
    brm_nlme_loglikelihood(m::BRMNLMEModel, i, θ, σ, η_i) -> Float64

One subject's conditional log-likelihood from its observation plate cell.
The work depends on that cell's observations, not on other subjects. Models
with flat grouped observations use [`brm_nlme_loglikelihoods`](@ref) instead.
"""
function brm_nlme_loglikelihood(m::BRMNLMEModel, i::Integer, θ, σ, η)
    point, row = _brm_nlme_cell_point(m, i, θ, σ, η)
    _rk_nlme_cell_value(m.cell_query, point, row)
end

function _brm_nlme_cell_value_and_selected_gradient(m::BRMNLMEModel, i::Integer, θ, σ, η)
    point, row = _brm_nlme_cell_point(m, i, θ, σ, η)
    indices = sort!(vcat(m.theta, m.sigma, point.eta_positions))
    active = point[indices]
    selected_gradient = similar(active)
    # A prepared RK sampler owns mutable AD workspace. Protect it when the
    # model is shared across concurrent subject requests.
    value = lock(m.cell_gradient.lock) do
        first(_rk_nlme_compact_cell_value_and_gradient!(
            m.cell_gradient.sampler, selected_gradient, m.unit,
            active, indices, row))
    end
    value, indices, selected_gradient, point.eta_positions
end

"""
    brm_nlme_loglikelihood_and_gradient(m::BRMNLMEModel, i, θ, σ, η_i)

Return one subject's conditional log-likelihood and its η_i gradient. RK's
compact cell sampler differentiates selected coordinates in one fused call,
without a packed gradient buffer.
"""
function brm_nlme_loglikelihood_and_gradient(m::BRMNLMEModel, i::Integer, θ, σ, η)
    value, indices, selected_gradient, eta_positions =
        _brm_nlme_cell_value_and_selected_gradient(m, i, θ, σ, η)
    value, [selected_gradient[searchsortedfirst(indices, j)]
        for j in eta_positions]
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
for every subject, from one reverse sweep that also retains the pointwise
densities. Subjects are
conditionally independent, so the joint gradient restricted to subject `i`'s
block is its conditional gradient once the standard-normal draw prior is removed.
"""
function brm_nlme_loglikelihoods_and_gradients(m::BRMNLMEModel, θ, σ, H)
    u = _brm_nlme_point(m, θ, σ, H)
    g = Vector{Float64}(undef, length(u))
    _, g, pointwise = _rk_value_gradient_and_pointwise!(m.sampler, g, u)
    values = _brm_nlme_sum_by_subject(m, pointwise)
    G = Matrix{Float64}(undef, size(H))
    for (i, block) in enumerate(m.view.subject_coordinates)
        G[:, i] .= view(g, block) .+ view(H, :, i)
    end
    values, G
end

# One direction per position within a subject block colors that position in
# every subject. Conditional independence makes its HVP the corresponding
# column of each subject's Hessian. The sampler also includes z ~ N(0, I): at
# the unit point z = η, so adding I removes that prior's Hessian.
function _brm_nlme_hessian_blocks(m::BRMNLMEModel, θ, σ, H)
    u = _brm_nlme_point(m, θ, σ, H)
    nη = size(H, 1)
    blocks = [Matrix{Float64}(undef, nη, nη) for _ in m.view.levels]
    for k in 1:nη
        direction = zeros(Float64, length(u))
        for coordinates in m.view.subject_coordinates
            direction[coordinates[k]] = 1.0
        end
        hv = _rk_eta_hvp(m.eta_hvp, direction, u)
        for (i, coordinates) in enumerate(m.view.subject_coordinates)
            blocks[i][:, k] .= view(hv, coordinates)
            blocks[i][k, k] += 1.0
        end
    end
    blocks
end
