# src/coordinate_transport.jl — the cross-backend coordinate transport.
#
# One BRMI lowers to an `RKBRMI` and an `SBBRMI` whose sampled coordinates live
# in different namespaces and shapes (RK `b_<g>.z.<level>.<margin>`
# versus Stan `<binding>_z_flat.<margin + K(level-1)>`, RK `pop_mu.beta_pop.2`
# versus Stan `pop_mu_beta_pop.2`). Both emitters already know what every declaration
# means, so the correspondence is built from their records instead of from the
# names: the RK emitter files one semantic record per sampled declaration it can
# address (`_rk_coordinate_record!`, threaded through `_rk_emit_ast`), and the
# SB side is read from its own binding metadata, random-effect blocks and
# descriptor carriers. The two inventories join on semantic addresses —
# predictor and coefficient label, grouping level and margin — and the result
# must be a bijection over BOTH coordinate lists. Anything either side cannot
# address is refused by name; nothing is matched by position or dimension. The
# one spelling rule is StanBlocks' own submodel naming: a provider declaration
# `<target>.<path>` pairs only with an SB output the same `<target>`
# declaration owns, named `<target>_<path>`.

"""
    BRMCoordinateTransportError <: Exception

A known boundary of the semantic [`brm_coordinate_transport`](@ref).
`reason` is `:unsupported_coverage` when the semantic inventory does not cover
the model, or `:parameterization_mismatch` when a known correspondence is not
one of the transport's relations (for example, centered effects or exact-total
blocks). `message::String` retains the diagnostic rendered by `showerror`.

Neither reason returns a partial map or permits a matched timing comparison.
Other failures, including invalid inputs and failed physical checks, propagate
normally; do not classify arbitrary exceptions as missing transport coverage.
"""
struct BRMCoordinateTransportError <: Exception
    reason::Symbol
    message::String
end

Base.showerror(io::IO, err::BRMCoordinateTransportError) = print(io, err.message)
_brm_transport_error(reason, message) = throw(BRMCoordinateTransportError(reason, message))

_rk_coordinate_record!(::Nothing, record) = nothing
_rk_coordinate_record!(records::AbstractVector, record) = (push!(records, record); nothing)

_rk_coordinate_records(plan::Union{_RKStructuralPlan,_RKValuePlan,_RKHeldOutPlan}) =
    let records = Any[]
        _rk_emit_ast(plan; coordinates=records)
        records
    end
_rk_coordinate_records(plan) = _brm_transport_error(:unsupported_coverage,
    "brm_coordinate_transport: an RK plan of kind `$(nameof(typeof(plan)))` has " *
    "no cross-backend coordinate inventory")

# Implemented by `BayesianRegressionModelsReactiveKernelsExt`: the built
# layout's coordinate names, each coordinate's layout transform, and the named
# constrained values at a point.
function _rk_layout_coordinate_names end
function _rk_layout_coordinate_transforms end
function _rk_constrained_values end

"""
    BRMCoordinatePair

One sampled coordinate shared by an [`RKBRMI`](@ref) and an [`SBBRMI`](@ref)
of the same [`BRMI`](@ref):

- `address` — the semantic address both backends agree on, e.g.
  `(; kind=:population, predictor=:mu, coefficient=:x)` or
  `(; kind=:ranef_z, group=:subject, id=:effect, level=3, margin=(:mu, :x))`;
- `rk` / `stan` — the RK layout coordinate name and the Stan unconstrained
  parameter name;
- `rk_declaration` / `rk_index` — the RK declaration's complete scope path
  (e.g. `pop_mu.beta_pop`) and the element of its
  constrained value that the coordinate parameterizes;
- `relation` — how the constrained values correspond: `:identity` (equal),
  `:exp` (the RK value is `exp` of the Stan value, e.g. a random-intercept
  scale against Stan's `log_scale`), `:cholesky` (a correlation-factor
  coordinate; the whole factors are compared, see
  [`brm_check_coordinate_transport`](@ref)), or `:simplex` (one free
  coordinate of a simplex block; see below).

In the `:identity`, `:exp` and `:cholesky` relations the two UNCONSTRAINED
values are equal, so those coordinates move by a permutation. A `:simplex`
block has the same physical simplex on both sides but different free
coordinates: RK samples stick-breaking fractions `σ(u[j] + log(K-j))`, while
Stan samples the inverse isometric log-ratio transform. Its `j`-th RK and Stan
coordinates are paired only as block positions; the block moves through the
nonlinear map recorded in [`BRMCoordinateTransport`](@ref)`.simplexes`.
"""
struct BRMCoordinatePair
    address::NamedTuple
    rk::Symbol
    stan::String
    rk_declaration::Symbol
    rk_index::Tuple
    relation::Symbol
end

"""
    BRMCoordinateTransport

The validated coordinate correspondence returned by
[`brm_coordinate_transport`](@ref). `rk_names` and `stan_names` are the two
complete coordinate lists, `pairs` holds one [`BRMCoordinatePair`](@ref) per
coordinate in RK order, and `permutation` matches positions:
`stan_names[i]` is paired with `rk_names[permutation[i]]`. `correlations` lists
each paired correlation factor (`rk_declaration`, `stan_carrier`, dimension
`K`) for the whole-factor physical check. A random-effect factor also carries
its `group`, `id` and ordered predictor/coefficient `margins`; a factor
declared by a submodel provider carries its `target` and `declaration`.

`simplexes` lists each `:simplex` block: its semantic `address`,
`rk_declaration`, `stan_carrier`, simplex length `K`, and the block's
`rk_positions` / `stan_positions` (the `K-1` free coordinates, in element
order). `logdensity_offset` is the constant
`-sum(log(K) / 2 for each simplex)`: at transported points the RK log density
equals Stan's normalized log density with Jacobian
(`propto=false, jacobian=true`) plus this offset, because the two simplex
transforms' log-Jacobians are `sum(log, x)` and `sum(log, x) + log(K)/2`.

With no simplex block, `stan_u == rk_u[permutation]` for points and gradients
alike ([`brm_rk_to_stan`](@ref), [`brm_stan_to_rk`](@ref)) and
`logdensity_offset == 0`. With a simplex block, points and gradients move
differently: use [`brm_rk_point_to_stan`](@ref),
[`brm_stan_point_to_rk`](@ref) and [`brm_stan_gradient_to_rk`](@ref).
"""
struct BRMCoordinateTransport
    rk_names::Vector{Symbol}
    stan_names::Vector{String}
    pairs::Vector{BRMCoordinatePair}
    permutation::Vector{Int}
    correlations::Vector{NamedTuple}
    simplexes::Vector{NamedTuple}
    logdensity_offset::Float64
end

# A transport without simplex blocks is a pure permutation.
BRMCoordinateTransport(rk_names, stan_names, pairs, permutation, correlations) =
    BRMCoordinateTransport(rk_names, stan_names, pairs, permutation, correlations,
        NamedTuple[], 0.0)

Base.length(t::BRMCoordinateTransport) = length(t.pairs)
Base.show(io::IO, t::BRMCoordinateTransport) = print(io,
    "BRMCoordinateTransport(", length(t.pairs), " coordinates, ",
    length(t.correlations), " correlation factors, ",
    length(t.simplexes), " simplexes)")

"""
    brm_coordinate_transport(rk::RKBRMI, sb::SBBRMI, stan_names) -> BRMCoordinateTransport

Pair every sampled coordinate of `rk` (its built layout's `coordinate_names`)
with exactly one unconstrained Stan coordinate of `sb`. `stan_names` is the
compiled model's `BridgeStan.param_unc_names(model)`, in its order. Both
backends must come from the same [`BRMI`](@ref).

The correspondence is semantic. Population coefficients pair by predictor and
coefficient label; categorical coefficients by their fitted level; scalar
parameters by their authored name; random-effect blocks by grouping column and
`|ID|`, with scales by margin, standardized draws by (grouping level, margin)
and correlation-factor coordinates in Stan's packing. A grouped
`levels × margins` RK draw matrix therefore reaches Stan's flat
`margin + K(level - 1)` vector through the level and margin labels, never
through matching dimensions.

Covered in this version: scalar parameters, ordinary-prior population
coefficients (including shared-design and separate categorical blocks),
plain non-centered random-effect blocks (`(1 | g)`, `(0 + x | g)`,
`(1 + x | g)`, `(… | id | g)`) with sampled scales, ungrouped HSGP terms
(isotropic or per-axis length scales, marginal scale and basis weights, paired
by their formula axes and parameter roles), monotonic `mo`/`mo1` terms (the
coefficient as a population coefficient, the increments as a `:simplex`
block), and the declarations a downstream submodel provider scopes under its
target (`_rk_submodel_rhs!` / `_sb_submodel_rhs!`). A provider declaration
`<target>.<path>` pairs with the sampled SB output that the same `<target>`
declaration owns under StanBlocks' submodel name `<target>_<path>` (path
segments joined by `_`), element by element; a simplex declaration pairs as a
`:simplex` block and an LKJ Cholesky factor as a correlation factor. Name both
providers' declarations identically for this pairing. Any coordinate outside
that coverage — on either side — is an error naming it; there is no partial
transport. An `SBBRMI` built with exact total-effect blocks or centered groups
has no sampled counterpart for the absorbed or centered coordinates; build it
with `total_groups=()` and without `centered_groups` for this comparison.

Every non-simplex pair has equal unconstrained values on both sides, and the
log-Jacobian of each backend's transform is evaluated at the same physical
point. Without simplex blocks, points and gradients move by the same
permutation ([`brm_rk_to_stan`](@ref), [`brm_stan_to_rk`](@ref)). With them,
use [`brm_rk_point_to_stan`](@ref), [`brm_stan_point_to_rk`](@ref) and
[`brm_stan_gradient_to_rk`](@ref), and add `logdensity_offset` to the Stan
density. Confirm the physical agreement at concrete points with
[`brm_check_coordinate_transport`](@ref).

For an already retained artifact build, use `RKBRMI(brmi, artifact.plan, built)`
with the original `brmi`, artifact plan and `built = build_kernel(bound)`.
That constructor wraps the exact values without translating or building again.
Use the same `held_out` selection for both backends. Known coverage boundaries
and incompatible parameterizations throw [`BRMCoordinateTransportError`](@ref)
with distinct `reason` values; other failures propagate normally.
"""
function brm_coordinate_transport(rk::RKBRMI, sb::SBBRMI, stan_names::AbstractVector)
    parent(rk) === parent(sb) || error(
        "brm_coordinate_transport: build both backends from the same BRMI instance")
    rk_names = Vector{Symbol}(_rk_layout_coordinate_names(rk))
    stan = Vector{String}(String.(stan_names))
    records = _rk_coordinate_records(rk.plan)
    d = brm_descriptor(sb)
    stan_pos = Dict{String,Int}(n => i for (i, n) in enumerate(stan))
    length(stan_pos) == length(stan) || error(
        "brm_coordinate_transport: Stan unconstrained names are not unique")
    totals = total_effect_blocks(d.plan)
    isempty(totals) || _brm_transport_error(:parameterization_mismatch,
        "brm_coordinate_transport: SB exact total-effect blocks for predictors " *
        "$(Tuple(b.predictor for b in totals)) absorb or mix population and " *
        "group coefficients; their coordinates are not a permutation of RK's " *
        "standardized draws. Build the SBBRMI with `total_groups=()` for this comparison.")
    pairs = BRMCoordinatePair[]
    correlations = NamedTuple[]
    simplexes = NamedTuple[]
    for record in records
        _brm_transport_pairs!(pairs, correlations, simplexes, Val(record.kind),
            record, rk, d, stan, stan_pos)
    end
    _brm_transport_validate(rk_names, stan, pairs, correlations, simplexes)
end

function _brm_transport_validate(rk_names, stan, pairs, correlations, simplexes)
    rk_pos = Dict{Symbol,Int}(n => i for (i, n) in enumerate(rk_names))
    length(rk_pos) == length(rk_names) || error(
        "brm_coordinate_transport: RK coordinate names are not unique")
    stan_pos = Dict{String,Int}(n => i for (i, n) in enumerate(stan))
    seen_rk = Dict{Symbol,BRMCoordinatePair}()
    seen_stan = Dict{String,BRMCoordinatePair}()
    seen_address = Dict{NamedTuple,BRMCoordinatePair}()
    for pair in pairs
        haskey(rk_pos, pair.rk) || error(
            "brm_coordinate_transport: the RK emission records coordinate " *
            "`$(pair.rk)` ($(pair.address)) that the built layout does not " *
            "carry; the emission and the build disagree")
        haskey(seen_rk, pair.rk) && error(
            "brm_coordinate_transport: RK coordinate `$(pair.rk)` is claimed by " *
            "both $(seen_rk[pair.rk].address) and $(pair.address)")
        haskey(seen_stan, pair.stan) && error(
            "brm_coordinate_transport: Stan coordinate `$(pair.stan)` is claimed " *
            "by both $(seen_stan[pair.stan].address) and $(pair.address)")
        haskey(seen_address, pair.address) && error(
            "brm_coordinate_transport: semantic address $(pair.address) is " *
            "claimed by both `$(seen_address[pair.address].rk)` and `$(pair.rk)`")
        seen_rk[pair.rk] = pair
        seen_stan[pair.stan] = pair
        seen_address[pair.address] = pair
    end
    missing_rk = [n for n in rk_names if !haskey(seen_rk, n)]
    missing_stan = [n for n in stan if !haskey(seen_stan, n)]
    if !isempty(missing_rk) || !isempty(missing_stan)
        lines = String[]
        isempty(missing_rk) || push!(lines,
            "RK coordinates with no semantic Stan counterpart: " *
            join(string.(missing_rk), ", "))
        isempty(missing_stan) || push!(lines,
            "Stan coordinates with no semantic RK counterpart: " *
            join(missing_stan, ", "))
        _brm_transport_error(:unsupported_coverage,
              "brm_coordinate_transport: the two lowerings do not pair " *
              "completely, so no transport is returned.\n  " * join(lines, "\n  ") *
              "\n  Covered: scalar parameters, ordinary-prior population and " *
              "categorical coefficients, plain non-centered random-effect " *
              "blocks with sampled scales, ungrouped HSGP terms, monotonic " *
              "terms, and declarations a submodel provider scopes under its " *
              "target (RK `<target>.<path>` with SB `<target>_<path>`). Other " *
              "constructs (R2D2/horseshoe priors, smooths, GP, grouped or " *
              "hyper-predicted HSGP, AR, measurement error, multi-membership " *
              "and stratified groups, centered or exact-total SB emissions, " *
              "completed covariates) are not paired yet.")
    end
    ordered = sort(pairs; by=pair -> rk_pos[pair.rk])
    permutation = Vector{Int}(undef, length(stan))
    for pair in ordered
        permutation[stan_pos[pair.stan]] = rk_pos[pair.rk]
    end
    blocks = NamedTuple[(; block...,
            rk_positions=[rk_pos[n] for n in block.rk],
            stan_positions=[stan_pos[n] for n in block.stan])
        for block in simplexes]
    offset = -sum((log(block.K) / 2 for block in blocks); init=0.0)
    BRMCoordinateTransport(rk_names, stan, ordered, permutation, correlations,
        blocks, offset)
end

_brm_stan_name(stan_pos, name, address) = begin
    haskey(stan_pos, name) || error(
        "brm_coordinate_transport: Stan coordinate `$name` for $(address) is " *
        "absent from the supplied unconstrained names")
    name
end

# Scalar parameters keep their authored name on both sides.
function _brm_transport_pairs!(pairs, correlations, simplexes, ::Val{:scalar}, record,
        rk, d, stan, stan_pos)
    name = String(record.declaration)
    address = (; kind=:scalar, name=record.declaration)
    haskey(stan_pos, name) || error(
        "brm_coordinate_transport: scalar parameter `$(record.declaration)` has " *
        "no Stan coordinate of the same authored name")
    push!(pairs, BRMCoordinatePair(address, record.declaration, name,
        record.declaration, (), :identity))
end

# The SB population binding of one predictor and its coefficient carrier.
function _brm_transport_population_binding(d, predictor::Symbol)
    bindings = hasproperty(d.plan, :bindings) ? d.plan.bindings : Dict()
    keys_ = [key for (key, b) in pairs(bindings)
        if b.role === :population_effect && b.logical === predictor]
    isempty(keys_) && return nothing
    length(keys_) == 1 || error(
        "brm_coordinate_transport: predictor `$predictor` owns $(length(keys_)) " *
        "SB population blocks; expected one")
    key = only(keys_)
    binding = bindings[key]
    kind = hasproperty(binding, :prior_scheme) ? binding.prior_scheme.kind : :ordinary
    idxs = _brm_carrier_indices(d.outputs,
        o -> o.role === :population_effect && !isnothing(o.declaration) &&
             o.declaration.target === key && o.name !== key)
    length(idxs) == 1 || error(
        "brm_coordinate_transport: SB population block `$key` owns " *
        "$(length(idxs)) coefficient carriers; expected one")
    (; key, binding, kind, output=d.outputs[only(idxs)],
       labels=Symbol[c.label for c in binding.design_columns])
end

function _brm_transport_population_stan(d, predictor, coefficient, stan, stan_pos)
    block = _brm_transport_population_binding(d, predictor)
    isnothing(block) && return nothing
    position = findall(==(coefficient), block.labels)
    isempty(position) && return nothing
    length(position) == 1 || error(
        "brm_coordinate_transport: coefficient `$coefficient` occurs " *
        "$(length(position)) times in SB population block `$(block.key)`")
    block.kind === :ordinary || _brm_transport_error(:unsupported_coverage,
        "brm_coordinate_transport: SB population block `$(block.key)` uses a " *
        "`$(block.kind)` prior scheme; only ordinary coefficient priors pair")
    coordinates = _brm_element_coordinates(block.output, stan)
    length(coordinates) == length(block.labels) || error(
        "brm_coordinate_transport: SB population carrier `$(block.output.name)` " *
        "has $(length(block.labels)) labels but $(length(coordinates)) Stan " *
        "coordinates")
    stan[coordinates[only(position)]]
end

function _brm_transport_pairs!(pairs, correlations, simplexes, ::Val{:population}, record,
        rk, d, stan, stan_pos)
    address = (; kind=:population, predictor=record.predictor,
        coefficient=record.coefficient)
    name = _brm_transport_population_stan(d, record.predictor,
        record.coefficient, stan, stan_pos)
    if isnothing(name)
        _brm_transport_error(:unsupported_coverage,
            "brm_coordinate_transport: population coefficient $(address) has no " *
            "sampled SB coordinate. The semantic inventory does not cover this coefficient.")
    end
    index = get(record, :index, ())
    rk_name = isempty(index) ? record.declaration :
        Symbol(record.declaration, ".", join(index, "."))
    push!(pairs, BRMCoordinatePair(address, rk_name, name,
        record.declaration, index, :identity))
end

_brm_transport_level(x) = x isa CA.CategoricalValue ? CA.unwrap(x) : x

# A categorical block: in a shared SB design the columns pair by their shared
# label; a separate SB contrast block pairs by fitted level.
function _brm_transport_pairs!(pairs, correlations, simplexes, ::Val{:population_block},
        record, rk, d, stan, stan_pos)
    shared = _brm_transport_population_binding(d, record.predictor)
    if !isnothing(shared) && all(label -> label in shared.labels, record.labels)
        for (i, label) in enumerate(record.labels)
            address = (; kind=:population, predictor=record.predictor,
                coefficient=label)
            name = _brm_transport_population_stan(d, record.predictor, label,
                stan, stan_pos)
            push!(pairs, BRMCoordinatePair(address,
                Symbol(record.declaration, ".", i), name, record.declaration,
                (i,), :identity))
        end
        return
    end
    resolved = brm_population_effect_coordinates(d, record.predictor, stan;
        coefficient=record.coefficient)
    hasproperty(resolved, :nonreference_levels) || error(
        "brm_coordinate_transport: categorical term `$(record.coefficient)` of " *
        "predictor `$(record.predictor)` resolves to no SB categorical block")
    sb_cellmeans = resolved.coding === :cellmeans
    rk_cellmeans = record.coding === :fullrank
    sb_cellmeans == rk_cellmeans || _brm_transport_error(:parameterization_mismatch,
        "brm_coordinate_transport: categorical term `$(record.coefficient)` of " *
        "predictor `$(record.predictor)` is coded differently by the two " *
        "backends (RK $(record.coding), SB $(resolved.coding))")
    sb_levels = map(_brm_transport_level, resolved.nonreference_levels)
    length(sb_levels) == length(record.level_values) || error(
        "brm_coordinate_transport: categorical term `$(record.coefficient)` of " *
        "predictor `$(record.predictor)` has $(length(record.level_values)) RK " *
        "and $(length(sb_levels)) SB coefficients")
    for (i, level) in enumerate(record.level_values)
        j = findall(l -> isequal(l, _brm_transport_level(level)), sb_levels)
        length(j) == 1 || error(
            "brm_coordinate_transport: categorical level $(repr(level)) of " *
            "`$(record.coefficient)` matches $(length(j)) SB coefficients " *
            "(SB levels $(repr(sb_levels)))")
        address = (; kind=:categorical, predictor=record.predictor,
            coefficient=record.coefficient, level=_brm_transport_level(level))
        push!(pairs, BRMCoordinatePair(address,
            Symbol(record.declaration, ".", i), stan[resolved.coordinates[only(j)]],
            record.declaration, (i,), :identity))
    end
end

# The SB random-effect block an RK bucket lowered to, and its ordered margins.
function _brm_transport_ranef_block(d, record)
    plan = d.plan
    brmi = plan.parent
    blocks = [b for b in ranef_blocks(plan)
        if b.group === record.group && b.id === record.id]
    if isnothing(record.id)
        predictors = unique(first.(record.margins))
        blocks = [b for b in blocks if begin
            binding = get(plan.bindings, b.binding, nothing)
            !isnothing(binding) && hasproperty(binding, :predictor) &&
                binding.predictor in predictors
        end]
    end
    address = (; group=record.group, id=record.id)
    length(blocks) == 1 || error(
        "brm_coordinate_transport: random-effect block $(address) of RK " *
        "margins $(record.margins) matches $(length(blocks)) SB blocks")
    block = only(blocks)
    block.generated && _brm_transport_error(:parameterization_mismatch,
        "brm_coordinate_transport: SB block `$(block.binding)` is drawn in " *
        "generated quantities and has no sampled coordinates")
    block.noncentered || _brm_transport_error(:parameterization_mismatch,
        "brm_coordinate_transport: SB block `$(block.binding)` is centered; its " *
        "coordinates are the effects, not the standardized draws RK samples. " *
        "Build the SBBRMI without `centered_groups` for this comparison.")
    margins = if isnothing(record.id)
        binding = plan.bindings[block.binding]
        [(binding.predictor, c) for c in binding.columns]
    else
        [(m.predictor, m.coefficient) for m in ranefcoefnames(brmi, record.id)]
    end
    length(margins) == block.n_terms || error(
        "brm_coordinate_transport: SB block `$(block.binding)` reports " *
        "$(length(margins)) margins for $(block.n_terms) terms")
    (; block, margins)
end

function _brm_transport_pairs!(pairs, correlations, simplexes, ::Val{:ranef}, record,
        rk, d, stan, stan_pos)
    (; block, margins) = _brm_transport_ranef_block(d, record)
    K = length(record.margins)
    K == block.n_terms || error(
        "brm_coordinate_transport: random-effect block `$(block.binding)` has " *
        "$K RK and $(block.n_terms) SB margins")
    sb_margin = map(collect(record.margins)) do margin
        j = findall(==(margin), margins)
        length(j) == 1 || error(
            "brm_coordinate_transport: RK margin $margin of block " *
            "`$(block.binding)` matches $(length(j)) SB margins $(Tuple(margins))")
        only(j)
    end
    base = (; group=record.group, id=record.id)
    # Scales: one per margin.
    spec = _RANEF_FAMILIES[block.family]
    tau = get(spec, :tau, nothing)
    for (k, margin) in enumerate(record.margins)
        address = (; kind=:ranef_scale, base..., margin)
        rk_name, rk_decl, rk_index = isnothing(record.scales) ?
            (Symbol(record.scale, ".", k), record.scale, (k,)) :
            (record.scales[k], record.scales[k], ())
        if !isnothing(tau)
            carrier = string(block.binding, "_", tau)
            name = string(carrier, ".", sb_margin[k])
            haskey(stan_pos, name) || (K == 1 && haskey(stan_pos, carrier)) || error(
                "brm_coordinate_transport: SB scale `$carrier` of block " *
                "`$(block.binding)` has no unconstrained coordinate for margin $margin")
            push!(pairs, BRMCoordinatePair(address, rk_name,
                haskey(stan_pos, name) ? name : carrier, rk_decl, rk_index, :identity))
        else
            carrier = string(block.binding, "_log_scale")
            K == 1 && haskey(stan_pos, carrier) || error(
                "brm_coordinate_transport: SB block `$(block.binding)` " *
                "(`$(block.family)`) has no per-margin scale coordinate to pair")
            push!(pairs, BRMCoordinatePair(address, rk_name, carrier, rk_decl,
                rk_index, :exp))
        end
    end
    # Standardized draws: one per (level, margin), paired by level label.
    rk_levels = _rk_grouping_levels(rk.plan.columns[record.group])
    length(rk_levels) == block.n_groups || error(
        "brm_coordinate_transport: grouping `$(record.group)` has " *
        "$(length(rk_levels)) RK and $(block.n_groups) SB levels")
    positions = ranef_coordinates(block, stan)
    for (g, level) in enumerate(rk_levels)
        h = findall(l -> isequal(l, level), block.levels)
        length(h) == 1 || error(
            "brm_coordinate_transport: level $(repr(level)) of grouping " *
            "`$(record.group)` matches $(length(h)) SB levels")
        for (k, margin) in enumerate(record.margins)
            address = (; kind=:ranef_z, base..., level, margin)
            push!(pairs, BRMCoordinatePair(address,
                Symbol(record.z, ".", g, ".", k),
                stan[positions[sb_margin[k], only(h)]], record.z, (g, k), :identity))
        end
    end
    # Correlation factor: Stan's packed coordinates, in the same margin order.
    isnothing(record.L) && return
    sb_margin == collect(1:K) || _brm_transport_error(:parameterization_mismatch,
        "brm_coordinate_transport: block `$(block.binding)` orders its margins " *
        "$(Tuple(margins)) on SB and $(record.margins) on RK; a reordered " *
        "correlation factor is not a coordinate permutation")
    carrier = string(block.binding, "_L")
    for m in 1:(K * (K - 1)) ÷ 2
        address = (; kind=:ranef_correlation, base..., margins=record.margins, index=m)
        push!(pairs, BRMCoordinatePair(address, Symbol(record.L, ".", m),
            _brm_stan_name(stan_pos, string(carrier, ".", m), address),
            record.L, (m,), :cholesky))
    end
    push!(correlations, (; base..., margins=record.margins, rk_declaration=record.L,
        stan_carrier=carrier, K))
end

# The unconstrained Stan names of one SB carrier, in element order.
function _brm_transport_elements(output, stan, address)
    positions = _brm_element_coordinates(output, stan)
    isempty(positions) && error(
        "brm_coordinate_transport: SB carrier `$(output.name)` for $(address) has " *
        "no coordinate among the supplied unconstrained names")
    stan[positions]
end

_brm_transport_is_simplex(output) = string(output.type) == "simplex"

# One simplex block: the K-1 free RK coordinates `rk_names` of
# `rk_declaration` with the K-1 free coordinates of SB carrier `output`.
function _brm_transport_simplex!(pairs, simplexes, address, rk_names,
        rk_declaration, output, stan, plan)
    _brm_transport_is_simplex(output) || error(
        "brm_coordinate_transport: RK simplex $(address) pairs with SB carrier " *
        "`$(output.name)` of type `$(output.type)`, not a simplex")
    names = _brm_transport_elements(output, stan, address)
    K = length(rk_names) + 1
    length(names) == K - 1 || error(
        "brm_coordinate_transport: simplex $(address) has $(K - 1) free RK and " *
        "$(length(names)) free Stan coordinates")
    count = _brm_output_coordinate_count(plan, output)
    isnothing(count) || count == K || error(
        "brm_coordinate_transport: SB simplex `$(output.name)` has $count " *
        "elements; RK's has $K")
    for (j, (rk_name, stan_name)) in enumerate(zip(rk_names, names))
        push!(pairs, BRMCoordinatePair((; address..., index=j), rk_name, stan_name,
            rk_declaration, (j,), :simplex))
    end
    push!(simplexes, (; address, rk_declaration,
        stan_carrier=String(output.name), K, rk=collect(Symbol, rk_names),
        stan=collect(String, names)))
end

# A monotonic component: its increments are a simplex block resolved through
# the formula term's `:simplex` role, and its coefficient is the population
# column that SB labels with the same term's owning carrier.
function _brm_transport_pairs!(pairs, correlations, simplexes, ::Val{:monotonic},
        record, rk, d, stan, stan_pos)
    head = record.head === :mo ? mo : mo1
    entry = _brm_term_entry(d, record.predictor,
        e -> getf(e.value) === head &&
            name(_sb_named_inner(record.head, only(getargs(e.value)))) === record.source,
        "`$(record.head)($(record.source))`")
    output = _brm_term_parameter_output(d, record.predictor, entry, :simplex)
    address = (; kind=:simplex, predictor=record.predictor, term=entry.term)
    increments = Symbol(record.declaration, ".simplex_incr")
    free = length(_brm_transport_elements(output, stan, address))
    _brm_transport_simplex!(pairs, simplexes, address,
        [Symbol(increments, ".", j) for j in 1:free], increments, output, stan, d.plan)
    record.beta || return
    coefficient = (; kind=:population, predictor=record.predictor,
        coefficient=entry.term)
    stan_name = _brm_transport_population_stan(d, record.predictor,
        output.declaration.target, stan, stan_pos)
    isnothing(stan_name) && _brm_transport_error(:unsupported_coverage,
        "brm_coordinate_transport: monotonic coefficient $(coefficient) has no " *
        "sampled SB population coordinate")
    beta = Symbol(record.declaration, ".beta")
    push!(pairs, BRMCoordinatePair(coefficient, beta, stan_name, beta, (), :identity))
end

# An ungrouped HSGP component, paired by its formula axes and parameter roles.
function _brm_transport_pairs!(pairs, correlations, simplexes, ::Val{:hsgp},
        record, rk, d, stan, stan_pos)
    term_axes(t) = Tuple(name(_sb_named_inner(:hsgp, a)) for a in getargs(t))
    entry = _brm_term_entry(d, record.predictor,
        e -> getf(e.value) === hsgp && !haskey(getkwargs(e.value), :by) &&
            term_axes(e.value) == record.axes,
        "ungrouped `hsgp$(record.axes)`")
    base = (; kind=:hsgp, predictor=record.predictor, term=entry.term)
    role_names(parameter) = _brm_transport_elements(
        _brm_term_parameter_output(d, record.predictor, entry, parameter), stan,
        (; base..., parameter))
    scale = record.declaration
    rho = role_names(:length_scale)
    rk_rho = record.iso ? [Symbol(scale, ".rho_iso")] :
        [Symbol(scale, ".rho_", j) for j in eachindex(record.axes)]
    length(rk_rho) == length(rho) || error(
        "brm_coordinate_transport: HSGP $(base) has $(length(rk_rho)) RK and " *
        "$(length(rho)) SB length scales")
    for (j, (rk_name, stan_name)) in enumerate(zip(rk_rho, rho))
        push!(pairs, BRMCoordinatePair((; base..., parameter=:length_scale, index=j),
            rk_name, stan_name, rk_name, (), :identity))
    end
    sigma = Symbol(scale, ".sigma")
    push!(pairs, BRMCoordinatePair((; base..., parameter=:sd, index=1), sigma,
        only(role_names(:sd)), sigma, (), :identity))
    weights = Symbol(scale, ".beta_raw")
    for (j, stan_name) in enumerate(role_names(:basis_weights))
        push!(pairs, BRMCoordinatePair((; base..., parameter=:basis_weights, index=j),
            Symbol(weights, ".", j), stan_name, weights, (j,), :identity))
    end
end

# The RK coordinate `<target>.<path>[.<index>...]` split into its declaration
# path and integer element index; `nothing` when a segment after the first
# integer is not one.
function _brm_transport_scoped(name::Symbol)
    parts = split(String(name), '.')
    k = findfirst(p -> !isnothing(tryparse(Int, p)), parts)
    isnothing(k) && return (Symbol(name), ())
    index = map(p -> tryparse(Int, p), parts[k:end])
    any(isnothing, index) && return nothing
    (Symbol(join(parts[1:(k - 1)], ".")), Tuple(index))
end

# A downstream submodel provider's declarations. RK scopes each one under the
# provider's target (`loc.inner.slope`); StanBlocks inlines the same submodel
# under `_`-joined names (`loc_inner_slope`) owned by the same target
# declaration. Only outputs that declaration owns are candidates, and an RK
# declaration without one stays unpaired, so the validator names it.
function _brm_transport_pairs!(pairs, correlations, simplexes, ::Val{:submodel},
        record, rk, d, stan, stan_pos)
    target = record.target
    owned = Dict{String,BRMOutput}(String(o.name) => o for o in d.outputs
        if !isnothing(o.declaration) && o.declaration.target === target &&
            o.kind === :parameter)
    prefix = string(target, ".")
    declarations = Symbol[]
    entries = Dict{Symbol,Vector{Any}}()
    for (rk_name, transform) in zip(_rk_layout_coordinate_names(rk),
            _rk_layout_coordinate_transforms(rk))
        startswith(String(rk_name), prefix) || continue
        scoped = _brm_transport_scoped(rk_name)
        isnothing(scoped) && continue
        declaration, index = scoped
        haskey(entries, declaration) || push!(declarations, declaration)
        push!(get!(entries, declaration, Any[]), (; name=rk_name, index, transform))
    end
    for declaration in declarations
        carrier = replace(String(declaration), '.' => '_')
        output = get(owned, carrier, nothing)
        isnothing(output) && continue
        elements = sort(entries[declaration]; by=e -> reverse(e.index))
        transforms = unique(e.transform for e in elements)
        length(transforms) == 1 || error(
            "brm_coordinate_transport: RK declaration `$declaration` mixes layout " *
            "transforms $(Tuple(transforms))")
        address = (; kind=:submodel, target, declaration)
        if only(transforms) === :simplex
            _brm_transport_simplex!(pairs, simplexes, address,
                [e.name for e in elements], declaration, output, stan, d.plan)
        elseif only(transforms) === :lkj
            names = _brm_transport_elements(output, stan, address)
            length(names) == length(elements) || error(
                "brm_coordinate_transport: correlation factor $(address) has " *
                "$(length(elements)) RK and $(length(names)) Stan coordinates")
            K = (1 + isqrt(1 + 8length(names))) ÷ 2
            for (m, (element, stan_name)) in enumerate(zip(elements, names))
                push!(pairs, BRMCoordinatePair((; address..., index=(m,)),
                    element.name, stan_name, declaration, (m,), :cholesky))
            end
            push!(correlations, (; target, declaration, rk_declaration=declaration,
                stan_carrier=carrier, K))
        else
            _brm_transport_is_simplex(output) && error(
                "brm_coordinate_transport: SB simplex `$carrier` pairs with RK " *
                "declaration `$declaration` of layout transform `$(only(transforms))`")
            positions = Set(_brm_element_coordinates(output, stan))
            for element in elements
                stan_name = isempty(element.index) ? carrier :
                    string(carrier, ".", join(element.index, "."))
                get(stan_pos, stan_name, 0) in positions || continue
                push!(pairs, BRMCoordinatePair((; address..., index=element.index),
                    element.name, stan_name, declaration, element.index, :identity))
            end
        end
    end
end

function _brm_transport_pairs!(pairs, correlations, simplexes, ::Val{kind}, record,
        rk, d, stan, stan_pos) where {kind}
    error("brm_coordinate_transport: internal: no pairing for RK record kind `$kind`")
end

"""
    brm_rk_to_stan(t::BRMCoordinateTransport, v) -> Vector
    brm_stan_to_rk(t::BRMCoordinateTransport, v) -> Vector

Move an unconstrained point, or a gradient with respect to it, between the RK
and Stan coordinate orders of a transport without simplex blocks. Both are the
same permutation: each pair has equal unconstrained values, so a Stan gradient
maps back to RK order exactly as the point does. A transport with simplex
blocks refuses both, since its points and gradients move differently; use
[`brm_rk_point_to_stan`](@ref), [`brm_stan_point_to_rk`](@ref) and
[`brm_stan_gradient_to_rk`](@ref). Neither function mutates `v`.
"""
function brm_rk_to_stan(t::BRMCoordinateTransport, v::AbstractVector)
    _brm_transport_permutation_only(t, "brm_rk_to_stan")
    _brm_rk_order_to_stan(t, v)
end

function brm_stan_to_rk(t::BRMCoordinateTransport, v::AbstractVector)
    _brm_transport_permutation_only(t, "brm_stan_to_rk")
    _brm_stan_order_to_rk(t, v)
end

_brm_transport_permutation_only(t, caller) = isempty(t.simplexes) || error(
    "$caller: this transport pairs $(length(t.simplexes)) simplex block(s), whose " *
    "points and gradients do not move by one permutation. Use " *
    "`brm_rk_point_to_stan` / `brm_stan_point_to_rk` for points and " *
    "`brm_stan_gradient_to_rk` for gradients.")

function _brm_rk_order_to_stan(t, v)
    length(v) == length(t.rk_names) || error(
        "brm_coordinate_transport: got $(length(v)) values for " *
        "$(length(t.rk_names)) RK coordinates")
    v[t.permutation]
end

function _brm_stan_order_to_rk(t, v)
    length(v) == length(t.stan_names) || error(
        "brm_coordinate_transport: got $(length(v)) values for " *
        "$(length(t.stan_names)) Stan coordinates")
    out = similar(v)
    for (stan_i, rk_i) in enumerate(t.permutation)
        out[rk_i] = v[stan_i]
    end
    out
end

# RK's stick-breaking simplex (ReactiveKernelsPPL `simplex_constrain`):
# `z[j] = σ(u[j] + log(K-j))`, `x[j] = z[j] * prod(1 - z[1:j-1])`. Returns
# `log.(x)` and the break fractions `z`, in log space throughout.
function _brm_rk_simplex_log(u::AbstractVector)
    K = length(u) + 1
    T = float(eltype(u))
    logx = Vector{T}(undef, K)
    z = Vector{T}(undef, K - 1)
    remaining = zero(T)
    for j in 1:(K - 1)
        a = u[j] + log(K - j)
        z[j] = logistic(a)
        logx[j] = remaining - log1pexp(-a)
        remaining -= log1pexp(a)
    end
    logx[K] = remaining
    logx, z
end

# The inverse: `u[j] = log(x[j]) - log(sum(x[j+1:K])) - log(K-j)`.
_brm_rk_simplex_free(logx::AbstractVector) =
    [logx[j] - logsumexp(view(logx, (j + 1):length(logx))) - log(length(logx) - j)
     for j in 1:(length(logx) - 1)]

# Stan's simplex (Stan 2.37+ `simplex_free`, the isometric log-ratio):
# `y[i] = (sum(log.(x[1:i])) - i * log(x[i+1])) / sqrt(i * (i + 1))`.
function _brm_stan_simplex_free(logx::AbstractVector)
    cumulative = zero(eltype(logx))
    map(1:(length(logx) - 1)) do i
        cumulative += logx[i]
        (cumulative - i * logx[i + 1]) / sqrt(i * (i + 1))
    end
end

# Its inverse (`simplex_constrain`, `softmax(sum_to_zero_constrain(y))`), as
# `log.(x)`.
function _brm_stan_simplex_log(y::AbstractVector)
    N = length(y)
    z = zeros(float(eltype(y)), N + 1)
    total = zero(eltype(z))
    for i in N:-1:1
        w = y[i] / sqrt(i * (i + 1))
        total += w
        z[i] += total
        z[i + 1] -= w * i
    end
    z .- logsumexp(z)
end

# `J' * g` for `J = d(stan free) / d(RK free)` at RK free coordinates `u`.
# With `v = B' * g` for the isometric log-ratio rows `B`, and
# `d log(x[k]) / d u[i]` equal to `1 - z[i]` (`k == i`), `-z[i]` (`k > i`) and
# `0` (`k < i`): `(J' * g)[i] = v[i] - z[i] * sum(v[i:K])`.
function _brm_simplex_pullback(u::AbstractVector, g::AbstractVector)
    K = length(u) + 1
    _, z = _brm_rk_simplex_log(u)
    v = zeros(promote_type(float(eltype(u)), eltype(g)), K)
    for i in 1:(K - 1)
        scale = g[i] / sqrt(i * (i + 1))
        for j in 1:i
            v[j] += scale
        end
        v[i + 1] -= i * scale
    end
    tail = zero(eltype(v))
    out = similar(v, K - 1)
    for i in (K - 1):-1:1
        tail += v[i]
        i == K - 1 && (tail += v[K])
        out[i] = v[i] - z[i] * tail
    end
    out
end

"""
    brm_rk_point_to_stan(t::BRMCoordinateTransport, u_rk) -> Vector
    brm_stan_point_to_rk(t::BRMCoordinateTransport, u_stan) -> Vector

Move an unconstrained POINT between the RK and Stan coordinate frames. Paired
coordinates move by `t.permutation`; each `t.simplexes` block goes through the
physical simplex, from RK's stick-breaking coordinates to Stan's isometric
log-ratio coordinates or back. The two functions are inverses and both
describe the same physical parameter point. They equal
[`brm_rk_to_stan`](@ref) / [`brm_stan_to_rk`](@ref) for a transport without
simplex blocks. Neither mutates its input. Use
[`brm_stan_gradient_to_rk`](@ref) for gradients.
"""
function brm_rk_point_to_stan(t::BRMCoordinateTransport, u::AbstractVector)
    out = _brm_rk_order_to_stan(t, u)
    for block in t.simplexes
        logx, _ = _brm_rk_simplex_log(u[block.rk_positions])
        out[block.stan_positions] = _brm_stan_simplex_free(logx)
    end
    out
end

function brm_stan_point_to_rk(t::BRMCoordinateTransport, y::AbstractVector)
    out = _brm_stan_order_to_rk(t, y)
    for block in t.simplexes
        out[block.rk_positions] =
            _brm_rk_simplex_free(_brm_stan_simplex_log(y[block.stan_positions]))
    end
    out
end

"""
    brm_stan_gradient_to_rk(t::BRMCoordinateTransport, u_rk, stan_gradient) -> Vector

Pull back a gradient with respect to Stan's unconstrained coordinates, taken at
`brm_rk_point_to_stan(t, u_rk)`, to RK's coordinates at `u_rk`. Paired
coordinates move by `t.permutation`; a simplex block multiplies by the
transpose of the exact Jacobian of its stick-breaking-to-log-ratio map. Since
`t.logdensity_offset` is constant, the pulled-back gradient of Stan's
normalized log density with Jacobian equals RK's log-density gradient. Neither
input is mutated.
"""
function brm_stan_gradient_to_rk(t::BRMCoordinateTransport, u::AbstractVector,
        g::AbstractVector)
    length(u) == length(t.rk_names) || error(
        "brm_stan_gradient_to_rk: got $(length(u)) point values for " *
        "$(length(t.rk_names)) RK coordinates")
    out = _brm_stan_order_to_rk(t, g)
    for block in t.simplexes
        out[block.rk_positions] =
            _brm_simplex_pullback(u[block.rk_positions], g[block.stan_positions])
    end
    out
end

_brm_physical(::Val{:identity}, stan_value) = stan_value
_brm_physical(::Val{:exp}, stan_value) = exp(stan_value)

# The emitter records a declaration's complete scope path. Constrained values
# expose statistical submodels as nested named tuples, while older declarations
# remain top-level properties.
function _brm_rk_declaration_value(values, declaration::Symbol)
    hasproperty(values, declaration) && return getproperty(values, declaration)
    for field in split(String(declaration), '.')
        values = getproperty(values, Symbol(field))
    end
    values
end

"""
    brm_check_coordinate_transport(t, rk::RKBRMI, stan_model, u_rk; atol=1e-10, rtol=1e-10)

Confirm at the RK unconstrained point `u_rk` that both backends describe the
same physical parameter point: every paired constrained value agrees under its
`relation`, every correlation factor agrees as a whole matrix, including
its structurally fixed unit diagonal norm and zero upper triangle, and every
simplex agrees as a whole vector of `K` elements. The Stan point is
`brm_rk_point_to_stan(t, u_rk)`. `stan_model` is the compiled
`BridgeStan.StanModel` whose `param_unc_names` built `t`.
Returns `(; pairs::Int, factors::Int, simplexes::Int, max_error::Float64)`:
`pairs` counts checked elements outside correlation factors and simplexes,
`factors` counts whole correlation matrices, `simplexes` whole simplexes, and
`max_error` is the largest absolute physical difference over these elements
and every matrix and simplex entry. Every comparison must meet
`atol + rtol * max(abs(rk_value), abs(stan_value))`. Any disagreement is an error
naming the coordinate. Densities and gradients are not compared here.
"""
function brm_check_coordinate_transport(t::BRMCoordinateTransport, rk::RKBRMI,
        stan_model, u_rk::AbstractVector; atol::Real=1e-10, rtol::Real=1e-10)
    Vector{Symbol}(_rk_layout_coordinate_names(rk)) == t.rk_names || error(
        "brm_check_coordinate_transport: the RK layout no longer matches the transport")
    String.(BridgeStan.param_unc_names(stan_model)) == t.stan_names || error(
        "brm_check_coordinate_transport: the Stan model's unconstrained names " *
        "do not match the transport")
    u = Vector{Float64}(u_rk)
    rk_values = _rk_constrained_values(rk, u)
    stan_u = brm_rk_point_to_stan(t, u)
    stan_values = BridgeStan.param_constrain(stan_model, stan_u)
    stan_index = Dict{String,Int}(String(n) => i
        for (i, n) in enumerate(BridgeStan.param_names(stan_model)))
    stan_value(name) = begin
        haskey(stan_index, name) || error(
            "brm_check_coordinate_transport: Stan constrained value `$name` is absent")
        stan_values[stan_index[name]]
    end
    max_error = 0.0
    function compare(what, a, b)
        err = abs(a - b)
        err <= atol + rtol * max(abs(a), abs(b)) || error(
            "brm_check_coordinate_transport: $what disagrees: RK $a, Stan $b")
        max_error = max(max_error, err)
    end
    checked = 0
    for pair in t.pairs
        pair.relation in (:cholesky, :simplex) && continue
        value = _brm_rk_declaration_value(rk_values, pair.rk_declaration)
        rk_value = isempty(pair.rk_index) ? value : value[pair.rk_index...]
        compare("$(pair.address) (`$(pair.rk)` / `$(pair.stan)`)", rk_value,
            _brm_physical(Val(pair.relation), stan_value(pair.stan)))
        checked += 1
    end
    for factor in t.correlations
        L = _brm_rk_declaration_value(rk_values, factor.rk_declaration)
        size(L) == (factor.K, factor.K) || error(
            "brm_check_coordinate_transport: RK factor `$(factor.rk_declaration)` " *
            "has size $(size(L)), expected $((factor.K, factor.K))")
        for i in 1:factor.K, j in 1:factor.K
            compare("correlation factor $(factor.stan_carrier)[$i,$j]", L[i, j],
                stan_value(string(factor.stan_carrier, ".", i, ".", j)))
        end
    end
    for block in t.simplexes
        x = _brm_rk_declaration_value(rk_values, block.rk_declaration)
        length(x) == block.K || error(
            "brm_check_coordinate_transport: RK simplex `$(block.rk_declaration)` " *
            "has $(length(x)) elements, expected $(block.K)")
        for i in 1:block.K
            compare("simplex $(block.stan_carrier)[$i]", x[i],
                stan_value(string(block.stan_carrier, ".", i)))
        end
    end
    (; pairs=checked, factors=length(t.correlations),
       simplexes=length(t.simplexes), max_error)
end
