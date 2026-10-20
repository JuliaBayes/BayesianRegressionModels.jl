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
one of the transport's relations (for example, exact-total blocks or a
reordered correlation factor). `message::String` retains the diagnostic
rendered by `showerror`.

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

# The program names a record's declarations live under: scalar `c`, block
# `b_g` of `b_g.z`, component `pop_x` of `pop_x.beta_pop`, submodel target `t`
# of `t.<path>`.
const _RK_RECORD_DECLARATION_FIELDS =
    (:declaration, :scale, :scales, :z, :L, :mixing, :target, :beta_declaration)
_rk_record_root(name::Symbol) = Symbol(first(split(String(name), '.')))
_rk_record_roots!(roots, ::Nothing) = roots
_rk_record_roots!(roots, name::Symbol) = push!(roots, _rk_record_root(name))
_rk_record_roots!(roots, names::Tuple) =
    (foreach(name -> _rk_record_roots!(roots, name), names); roots)
_rk_record_roots(record) = foldl(_RK_RECORD_DECLARATION_FIELDS; init=Set{Symbol}()) do roots, field
    _rk_record_roots!(roots, get(record, field, nothing))
end

# The fitted pass removes the declarations no response reads; on the Stan route
# they are generated draws, so neither backend packs a coordinate for them.
# A record leaves with them only when every program name it declares is an
# output of a removed statement and of no kept one; any other record stays, so
# a genuine emission/build disagreement still fails by name downstream.
_rk_drop_unfitted_records!(::Nothing, outputs, kept) = nothing
function _rk_drop_unfitted_records!(records::AbstractVector, outputs, kept)
    fitted = reduce(union!, outputs[kept]; init=Set{Symbol}())
    removed = setdiff!(reduce(union!, outputs[.!kept]; init=Set{Symbol}()), fitted)
    isempty(removed) && return nothing
    filter!(records) do record
        roots = _rk_record_roots(record)
        isempty(roots) || !issubset(roots, removed)
    end
    nothing
end

_rk_coordinate_records(plan::Union{_RKStructuralPlan,_RKValuePlan,_RKHeldOutPlan}) =
    let records = Any[]
        _rk_emit_ast(plan; coordinates=records)
        records
    end
_rk_coordinate_records(plan) = _brm_transport_error(:unsupported_coverage,
    "brm_coordinate_transport: an RK plan of kind `$(nameof(typeof(plan)))` has " *
    "no cross-backend coordinate inventory")

# Implemented by `BayesianRegressionModelsReactiveKernelsExt`: the built
# layout's coordinate names, each coordinate's layout transform and the
# constrained lower bound of a `:floored` one, and the named constrained values
# at a point.
function _rk_layout_coordinate_names end
function _rk_layout_coordinate_transforms end
function _rk_layout_coordinate_floors end
function _rk_constrained_values end

"""
    BRMCoordinatePair

One sampled coordinate shared by an [`RKBRMI`](@ref) and an [`SBBRMI`](@ref)
of the same [`BRMI`](@ref):

- `address` — the semantic address both backends agree on, e.g.
  `(; kind=:population, predictor=:mu, coefficient=:x)` or
  `(; kind=:ranef_z, group=:subject, id=:effect, level=3, margin=(:mu, :x))`,
  or a Student-t block's per-level weight `(; kind=:ranef_mixing, …, level=3)`;
- `rk` / `stan` — the RK layout coordinate name and the Stan unconstrained
  parameter name;
- `rk_declaration` / `rk_index` — the RK declaration's complete scope path
  (e.g. `pop_mu.beta_pop`) and the element of its
  constrained value that the coordinate parameterizes;
- `relation` — how the constrained values correspond: `:identity` (equal),
  `:exp` (the RK value is `exp` of the Stan value, e.g. a random-intercept
  scale against Stan's `log_scale`), `:cholesky` (a correlation-factor
  coordinate; the whole factors are compared, see
  [`brm_check_coordinate_transport`](@ref)), `:simplex` (one free
  coordinate of a simplex block; see below), or `:centered` (one standardized
  draw of a centered random-effect block; see below);
- `stan_value` — the Stan constrained parameter name holding the physical
  value this coordinate parameterizes. It equals `stan` except inside a Stan
  array of vectors, whose unconstrained values BridgeStan stores array-major
  while `param_unc_names` lists the same block first-index-fastest, so the
  name at a position can belong to another element.

In the `:identity`, `:exp` and `:cholesky` relations the two UNCONSTRAINED
values are equal, so those coordinates move by a permutation. A `:simplex`
block has the same physical simplex on both sides but different free
coordinates: RK samples stick-breaking fractions `σ(u[j] + log(K-j))`, while
Stan samples the inverse isometric log-ratio transform. Its `j`-th RK and Stan
coordinates are paired only as block positions; the block moves through the
nonlinear map recorded in [`BRMCoordinateTransport`](@ref)`.simplexes`.
A `:centered` pair joins RK's standardized draw `z` of one (level, margin) with
the effect `b` of the same (level, margin) that a centered SB block
(`centered_groups`) samples. At each point `b = diag(tau) * L * z` per level,
with that point's scales `tau` and correlation factor `L`, so the block moves
through the map recorded in [`BRMCoordinateTransport`](@ref)`.centered`.
"""
struct BRMCoordinatePair
    address::NamedTuple
    rk::Symbol
    stan::String
    rk_declaration::Symbol
    rk_index::Tuple
    relation::Symbol
    stan_value::String
end

BRMCoordinatePair(address, rk, stan, rk_declaration, rk_index, relation) =
    BRMCoordinatePair(address, rk, stan, rk_declaration, rk_index, relation, stan)

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

`centered` lists each centered SB random-effect block: its `group`, `id`,
ordered `margins`, SB `binding` and effect carrier `stan_carrier`, margin
count `K`, level count `G`, the RK coordinates of its scales (`rk_scales`, one
per margin, log scales) and of its correlation factor (`rk_cholesky`, empty
for `K == 1`), and the `K × G` matrices `rk` / `stan` of standardized-draw and
effect coordinates by (margin, SB level), with their positions
(`scale_positions`, `cholesky_positions`, `rk_positions`, `stan_positions`).
`stan_values` names each effect's constrained Stan value. The block's density
offset depends on the point: at transported points the RK log density equals
Stan's plus `logdensity_offset` plus `G * log(det(diag(tau) * L))` per
centered block, the log-Jacobian of `z -> b`; use
[`brm_stan_logdensity_to_rk`](@ref). Reading `logdensity_offset` from a
transport with a centered block is an error naming that function.

With no simplex or centered block, `stan_u == rk_u[permutation]` for points
and gradients alike ([`brm_rk_to_stan`](@ref), [`brm_stan_to_rk`](@ref)) and
`logdensity_offset == 0`. Otherwise points and gradients move differently:
use [`brm_rk_point_to_stan`](@ref), [`brm_stan_point_to_rk`](@ref),
[`brm_stan_gradient_to_rk`](@ref) and [`brm_stan_logdensity_to_rk`](@ref).
"""
struct BRMCoordinateTransport
    rk_names::Vector{Symbol}
    stan_names::Vector{String}
    pairs::Vector{BRMCoordinatePair}
    permutation::Vector{Int}
    correlations::Vector{NamedTuple}
    simplexes::Vector{NamedTuple}
    logdensity_offset::Float64
    centered::Vector{NamedTuple}
end

# A transport without simplex or centered blocks is a pure permutation.
BRMCoordinateTransport(rk_names, stan_names, pairs, permutation, correlations) =
    BRMCoordinateTransport(rk_names, stan_names, pairs, permutation, correlations,
        NamedTuple[], 0.0)
BRMCoordinateTransport(rk_names, stan_names, pairs, permutation, correlations,
        simplexes, logdensity_offset) =
    BRMCoordinateTransport(rk_names, stan_names, pairs, permutation, correlations,
        simplexes, logdensity_offset, NamedTuple[])

# A centered block's density offset depends on the point, so the constant
# field alone would silently mis-state it.
function Base.getproperty(t::BRMCoordinateTransport, field::Symbol)
    field === :logdensity_offset && !isempty(getfield(t, :centered)) && error(
        "BRMCoordinateTransport: this transport maps $(length(getfield(t, :centered))) " *
        "centered random-effect block(s), whose log-density offset depends on the " *
        "point. Use `brm_stan_logdensity_to_rk(transport, rk_u, stan_value)`.")
    getfield(t, field)
end

Base.length(t::BRMCoordinateTransport) = length(t.pairs)
Base.show(io::IO, t::BRMCoordinateTransport) = print(io,
    "BRMCoordinateTransport(", length(t.pairs), " coordinates, ",
    length(t.correlations), " correlation factors, ",
    length(t.simplexes), " simplexes",
    isempty(t.centered) ? "" : ", $(length(t.centered)) centered blocks", ")")

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

Covered in this version: scalar parameters, vector parameters (a response's
`OrderedLogistic` cutpoints and `Ordinal` cumulative or stopping-ratio
thresholds, element by element under the shared ordered or unconstrained
transform, and authored `Dirichlet` simplexes as `:simplex` blocks; all keep
their declaration name on both backends), `Ordinal` per-threshold
coefficients by (stage, term), ordinary-prior population
coefficients (including shared-design and separate categorical blocks),
plain random-effect blocks (`(1 | g)`, `(0 + x | g)`, `(1 + x | g)`,
`(… | id | g)`) with sampled scales, non-centered or centered on the SB side
(see below), ungrouped HSGP terms
(isotropic or per-axis length scales, marginal scale and basis weights, paired
by their formula axes and parameter roles), monotonic `mo`/`mo1` terms (the
coefficient as a population coefficient, the increments as a `:simplex`
block), and the declarations a downstream submodel provider scopes under its
target (`_rk_submodel_rhs!` / `_sb_submodel_rhs!`). A provider declaration
`<target>.<path>` pairs with the sampled SB output that the same `<target>`
declaration owns under StanBlocks' submodel name `<target>_<path>` (path
segments joined by `_`), element by element; a simplex declaration pairs as a
`:simplex` block and an LKJ Cholesky factor as a correlation factor. Name both
providers' declarations identically for this pairing. Missing-value completions
pair RK `<column>.y_mis.<i>` with Stan `<column>_y_mis.<i>` when the completed
column feeds a downstream likelihood. Any coordinate outside
that coverage — on either side — is an error naming it; there is no partial
transport. An `SBBRMI` built with exact total-effect blocks has no sampled
counterpart for the absorbed coordinates; build it with `total_groups=()` for
this comparison.

A block in `centered_groups` samples the effects `b` themselves, with
`diag(tau) * L` as their covariance factor, while RK samples standardized
draws `z`. Its effects pair with RK's draws by (level, margin) as `:centered`
pairs, its scales and correlation factor pair as for a non-centered block, and
the block moves through the exact per-point map `b = diag(tau) * L * z`, using
that point's `tau` and `L` (`t.centered`). Its SB margins must be in RK's
margin order.

Every non-simplex, non-centered pair has equal unconstrained values on both
sides, and the log-Jacobian of each backend's transform is evaluated at the
same physical point. Without simplex or centered blocks, points and gradients
move by the same permutation ([`brm_rk_to_stan`](@ref),
[`brm_stan_to_rk`](@ref)). With them, use [`brm_rk_point_to_stan`](@ref),
[`brm_stan_point_to_rk`](@ref) and [`brm_stan_gradient_to_rk`](@ref), and move
the Stan density with [`brm_stan_logdensity_to_rk`](@ref). Confirm the
physical agreement at concrete points with
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
    centered = NamedTuple[]
    for record in records
        _brm_transport_pairs!(pairs, correlations, simplexes, centered,
            Val(record.kind), record, rk, d, stan, stan_pos)
    end
    t = _brm_transport_validate(rk_names, stan, pairs, correlations, simplexes,
        centered)
    _brm_transport_centered_scales(rk, t)
    t
end

# The centered map rebuilds each block's scales as `exp` of their coordinates:
# RK's `:exp` transform, or `:floored` (`lo + exp(u)`) at a zero floor.
function _brm_transport_centered_scales(rk, t)
    isempty(t.centered) && return nothing
    transforms = _rk_layout_coordinate_transforms(rk)
    floors = _rk_layout_coordinate_floors(rk)
    for block in t.centered, i in block.scale_positions
        transforms[i] === :exp || transforms[i] === :floored && iszero(floors[i]) ||
            _brm_transport_error(:unsupported_coverage,
                "brm_coordinate_transport: centered block `$(block.binding)` scale " *
                "`$(t.rk_names[i])` has RK transform `$(repr(transforms[i]))`" *
                (transforms[i] === :floored ? " at floor $(floors[i])" : "") *
                "; the centered map rebuilds scales sampled as `exp` of their " *
                "coordinate only")
    end
    nothing
end

function _brm_transport_validate(rk_names, stan, pairs, correlations, simplexes,
        centered=NamedTuple[])
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
              "\n  Covered: scalar parameters, response cutpoints and " *
              "thresholds, `Ordinal` per-threshold coefficients, authored " *
              "`Dirichlet` simplexes, ordinary-prior " *
              "population and categorical coefficients, plain non-centered or " *
              "centered random-effect " *
              "blocks with sampled scales, ungrouped HSGP terms, monotonic " *
              "terms, submodel provider declarations, and missing-value " *
              "completions (RK `<column>.y_mis` with SB `<column>_y_mis`). Other " *
              "constructs (R2D2/horseshoe priors, smooths, GP, grouped or " *
              "hyper-predicted HSGP, AR, measurement error, multi-membership " *
              "and stratified groups, exact-total SB emissions) " *
              "are not paired yet.")
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
    centered_blocks = NamedTuple[(; block...,
            scale_positions=[rk_pos[n] for n in block.rk_scales],
            cholesky_positions=[rk_pos[n] for n in block.rk_cholesky],
            rk_positions=map(n -> rk_pos[n], block.rk),
            stan_positions=map(n -> stan_pos[n], block.stan))
        for block in centered]
    BRMCoordinateTransport(rk_names, stan, ordered, permutation, correlations,
        blocks, offset, centered_blocks)
end

_brm_stan_name(stan_pos, name, address) = begin
    haskey(stan_pos, name) || error(
        "brm_coordinate_transport: Stan coordinate `$name` for $(address) is " *
        "absent from the supplied unconstrained names")
    name
end

# Scalar parameters keep their authored name on both sides.
function _brm_transport_pairs!(pairs, correlations, simplexes, centered,
        ::Val{:scalar}, record, rk, d, stan, stan_pos)
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

function _brm_transport_pairs!(pairs, correlations, simplexes, centered,
        ::Val{:population}, record, rk, d, stan, stan_pos)
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
function _brm_transport_pairs!(pairs, correlations, simplexes, centered,
        ::Val{:population_block}, record, rk, d, stan, stan_pos)
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

# The known marginal scales of a known-value SB block, in its margin order
# (`nothing` for a sampled margin).
_brm_transport_known_scales(d, block) =
    collect(d.plan.bindings[block.binding].known_scales)

# The SB random-effect block an RK bucket (or `(; group, id)`) lowered to, and
# its ordered margins; for a marginalized `|ID|` block, its sampled active part.
# `caller` names the public entry in errors.
function _brm_sb_ranef_block(d, record; caller="brm_coordinate_transport")
    plan = d.plan
    brmi = plan.parent
    all_blocks = ranef_blocks(d)
    blocks = [b for b in all_blocks
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
        "$caller: random-effect block $(address) of RK " *
        "margins $(record.margins) matches $(length(blocks)) SB blocks")
    block = only(blocks)
    # A marginalized `|ID|` block samples its likelihood-active margins in a
    # carrier (src/sb_marginal_ranef.jl); RK plans that same reduced block.
    completion = get(plan.bindings, block.binding, NamedTuple())
    active = hasproperty(completion, :active) ?
        collect(plan.bindings[completion.active].margins) : nothing
    isnothing(active) ||
        (block = only(b for b in all_blocks if b.binding === completion.active))
    margins = if isnothing(record.id)
        binding = plan.bindings[block.binding]
        [(binding.predictor, c) for c in binding.columns]
    else
        shared = [(m.predictor, m.coefficient) for m in ranefcoefnames(brmi, record.id)]
        isnothing(active) ? shared : shared[active]
    end
    length(margins) == block.n_terms || error(
        "$caller: SB block `$(block.binding)` reports " *
        "$(length(margins)) margins for $(block.n_terms) terms")
    (; block, margins)
end

function _brm_transport_ranef_block(d, record)
    (; block, margins) = _brm_sb_ranef_block(d, record)
    block.generated && _brm_transport_error(:parameterization_mismatch,
        "brm_coordinate_transport: SB block `$(block.binding)` is drawn in " *
        "generated quantities and has no sampled coordinates")
    (; block, margins)
end

function _brm_transport_pairs!(pairs, correlations, simplexes, centered,
        ::Val{:ranef}, record, rk, d, stan, stan_pos)
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
    # A centered block's effects are `diag(tau) * L * z` in SB margin order.
    block.noncentered || sb_margin == collect(1:K) || _brm_transport_error(
        :parameterization_mismatch,
        "brm_coordinate_transport: centered block `$(block.binding)` orders its " *
        "margins $(Tuple(margins)) on SB and $(record.margins) on RK; its " *
        "effects cannot be mapped margin by margin")
    base = (; group=record.group, id=record.id)
    # Scales: one per margin.
    rk_scales = Vector{Symbol}(undef, K)
    rk_scale_values = Vector{Tuple{Symbol,Tuple}}(undef, K)
    spec = _RANEF_FAMILIES[block.family]
    tau = get(spec, :tau, nothing)
    # A known-value block (`sd(...) = value`) samples only its free scales,
    # under `<binding>_tau_free` when some scale is known; a known scale is a
    # constant on both backends and pairs with nothing.
    known_scales = get(spec, :known, false) ?
        _brm_transport_known_scales(d, block) : fill(nothing, block.n_terms)
    rk_known = get(record, :known_scales, nothing)
    free_margins = findall(isnothing, known_scales)
    for (k, margin) in enumerate(record.margins)
        sb_known = !isnothing(known_scales[sb_margin[k]])
        sb_known == (!isnothing(rk_known) && !isnothing(rk_known[k])) ||
            _brm_transport_error(:parameterization_mismatch,
                "brm_coordinate_transport: margin $margin of block " *
                "`$(block.binding)` has a known scale on " *
                "$(sb_known ? "SB" : "RK") only")
        sb_known && continue
        address = (; kind=:ranef_scale, base..., margin)
        rk_name, rk_decl, rk_index = isnothing(record.scales) ?
            (Symbol(record.scale, ".", k), record.scale, (k,)) :
            (record.scales[k], record.scales[k], ())
        rk_scales[k] = rk_name
        rk_scale_values[k] = (rk_decl, rk_index)
        if length(free_margins) < block.n_terms
            carrier = string(block.binding, "_tau_free")
            name = string(carrier, ".", only(findall(==(sb_margin[k]), free_margins)))
            single = length(free_margins) == 1 && haskey(stan_pos, carrier)
            haskey(stan_pos, name) || single || error(
                "brm_coordinate_transport: SB free scale `$carrier` of block " *
                "`$(block.binding)` has no unconstrained coordinate for margin $margin")
            push!(pairs, BRMCoordinatePair(address, rk_name,
                haskey(stan_pos, name) ? name : carrier, rk_decl, rk_index, :identity))
        elseif !isnothing(tau)
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
    # Standardized draws: one per (level, margin), paired by level label. A
    # centered block pairs each with the effect of the same (level, margin).
    rk_levels = _rk_grouping_levels(rk.plan.columns[record.group])
    G = length(rk_levels)
    G == block.n_groups || error(
        "brm_coordinate_transport: grouping `$(record.group)` has " *
        "$G RK and $(block.n_groups) SB levels")
    positions = ranef_coordinates(block, stan)
    layout = spec.layout
    rk_z = Matrix{Symbol}(undef, K, G)
    stan_b = Matrix{String}(undef, K, G)
    stan_values = Matrix{String}(undef, K, G)
    rk_level = Vector{Int}(undef, G)
    for (g, level) in enumerate(rk_levels)
        h = findall(l -> isequal(l, level), block.levels)
        length(h) == 1 || error(
            "brm_coordinate_transport: level $(repr(level)) of grouping " *
            "`$(record.group)` matches $(length(h)) SB levels")
        h = only(h)
        rk_level[h] = g
        for (k, margin) in enumerate(record.margins)
            address = (; kind=:ranef_z, base..., level, margin)
            rk_name = Symbol(record.z, ".", g, ".", k)
            stan_name = stan[positions[sb_margin[k], h]]
            if block.noncentered
                push!(pairs, BRMCoordinatePair(address, rk_name, stan_name,
                    record.z, (g, k), :identity))
            else
                # An array-of-vectors carrier's unconstrained name at a
                # position can belong to another element (`ranef_coordinates`).
                value = _ranef_coord_name(block, layout, sb_margin[k], h)
                push!(pairs, BRMCoordinatePair(address, rk_name, stan_name,
                    record.z, (g, k), :centered, value))
                rk_z[k, h], stan_b[k, h], stan_values[k, h] = rk_name, stan_name, value
            end
        end
    end
    block.noncentered || push!(centered, (; base..., margins=record.margins,
        binding=block.binding, stan_carrier=String(block.z), K, G, rk_scales,
        rk_cholesky=isnothing(record.L) ? Symbol[] :
            [Symbol(record.L, ".", m) for m in 1:(K * (K - 1)) ÷ 2],
        rk=rk_z, stan=stan_b, stan_values, rk_declaration=record.z, rk_level,
        rk_scale_values, rk_L=record.L))
    # Student-t mixing weights: one per level, paired by level label. Both
    # backends must agree that the block is Student-t.
    rk_mixing = get(record, :mixing, nothing)
    stan_mixing = _ranef_mixing_coordinates(block, stan)
    isnothing(rk_mixing) == isnothing(stan_mixing) || _brm_transport_error(
        :parameterization_mismatch,
        "brm_coordinate_transport: block `$(block.binding)` is Student-t on " *
        "$(isnothing(rk_mixing) ? "SB" : "RK") only")
    if !isnothing(rk_mixing)
        for (g, level) in enumerate(rk_levels)
            h = only(findall(l -> isequal(l, level), block.levels))
            address = (; kind=:ranef_mixing, base..., level)
            push!(pairs, BRMCoordinatePair(address, Symbol(rk_mixing, ".", g),
                stan[stan_mixing[h]], rk_mixing, (g,), :identity))
        end
    end
    # Correlation factor: Stan's packed coordinates, in the same margin order.
    # A known correlation (`cor(:, ID) = R`) is data on SB and a constant on RK.
    sb_cor_known = get(spec, :known, false) &&
        !isnothing(d.plan.bindings[block.binding].known_cor)
    sb_cor_known == !isnothing(get(record, :known_cor, nothing)) || _brm_transport_error(
        :parameterization_mismatch,
        "brm_coordinate_transport: block `$(block.binding)` has a known " *
        "correlation on $(sb_cor_known ? "SB" : "RK") only")
    isnothing(record.L) && return
    sb_margin == collect(1:K) || _brm_transport_error(:parameterization_mismatch,
        "brm_coordinate_transport: block `$(block.binding)` orders its margins " *
        "$(Tuple(margins)) on SB and $(record.margins) on RK; a reordered " *
        "correlation factor is not a coordinate permutation")
    carrier = _sb_ranef_correlation_carrier(block)
    for m in 1:(K * (K - 1)) ÷ 2
        address = (; kind=:ranef_correlation, base..., margins=record.margins, index=m)
        push!(pairs, BRMCoordinatePair(address, Symbol(record.L, ".", m),
            _brm_stan_name(stan_pos, string(carrier, ".", m), address),
            record.L, (m,), :cholesky))
    end
    push!(correlations, (; base..., margins=record.margins, rk_declaration=record.L,
        stan_carrier=carrier, K))
end

# The Stan carrier of a random-effect block's correlation Cholesky factor.
_sb_ranef_correlation_carrier(block) = string(block.binding, "_L")

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
# `rk_declaration` with the K-1 free coordinates of SB carrier `output`. A
# `column` names one simplex column of a two-axis RK declaration.
function _brm_transport_simplex!(pairs, simplexes, address, rk_names,
        rk_declaration, output, stan; column=nothing)
    _brm_transport_is_simplex(output) || error(
        "brm_coordinate_transport: RK simplex $(address) pairs with SB carrier " *
        "`$(output.name)` of type `$(output.type)`, not a simplex")
    names = _brm_transport_elements(output, stan, address)
    K = length(rk_names) + 1
    length(names) == K - 1 || error(
        "brm_coordinate_transport: simplex $(address) has $(K - 1) free RK and " *
        "$(length(names)) free Stan coordinates")
    count = _brm_output_coordinate_count(output)
    isnothing(count) || count == K || error(
        "brm_coordinate_transport: SB simplex `$(output.name)` has $count " *
        "elements; RK's has $K")
    for (j, (rk_name, stan_name)) in enumerate(zip(rk_names, names))
        push!(pairs, BRMCoordinatePair((; address..., index=j), rk_name, stan_name,
            rk_declaration, column === nothing ? (j,) : (j, column), :simplex))
    end
    push!(simplexes, (; address, rk_declaration, column,
        stan_carrier=String(output.name), K, rk=collect(Symbol, rk_names),
        stan=collect(String, names)))
end

# A monotonic component: its increments are a simplex block resolved through
# the formula term's `:simplex` role, and its coefficient is the population
# column that SB labels with the same term's public label (`mo_<c>`, whichever
# carrier the occurrence emitted). In a joint block (`column`), the predictor
# owns one simplex column and one coefficient entry.
function _brm_transport_pairs!(pairs, correlations, simplexes, centered,
        ::Val{:monotonic}, record, rk, d, stan, stan_pos)
    head = record.head === :mo ? mo : mo1
    entry = _brm_term_entry(d, record.predictor,
        e -> getf(e.value) === head &&
            name(_sb_named_inner(record.head, only(getargs(e.value)))) === record.source,
        "`$(record.head)($(record.source))`")
    output = _brm_term_parameter_output(d, record.predictor, entry, :simplex)
    address = (; kind=:simplex, predictor=record.predictor, term=entry.term)
    increments = Symbol(record.declaration, ".simplex_incr")
    free = length(_brm_transport_elements(output, stan, address))
    column = get(record, :column, nothing)
    rk_names = column === nothing ? [Symbol(increments, ".", j) for j in 1:free] :
        [Symbol(increments, ".", j, ".", column) for j in 1:free]
    _brm_transport_simplex!(pairs, simplexes, address, rk_names, increments, output,
        stan; column)
    record.beta || return
    coefficient = (; kind=:population, predictor=record.predictor,
        coefficient=entry.term)
    stan_name = _brm_transport_population_stan(d, record.predictor,
        entry.term, stan, stan_pos)
    isnothing(stan_name) && _brm_transport_error(:unsupported_coverage,
        "brm_coordinate_transport: monotonic coefficient $(coefficient) has no " *
        "sampled SB population coordinate")
    beta = something(get(record, :beta_declaration, nothing), Symbol(record.declaration, ".beta"))
    index = get(record, :beta_index, ())
    rk_name = isempty(index) ? beta : Symbol(beta, ".", join(index, "."))
    push!(pairs, BRMCoordinatePair(coefficient, rk_name, stan_name, beta, index, :identity))
end

# An ungrouped HSGP component, paired by its formula axes and parameter roles.
function _brm_transport_pairs!(pairs, correlations, simplexes, centered,
        ::Val{:hsgp}, record, rk, d, stan, stan_pos)
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

# A provider's or missing completion's scoped declarations. RK scopes each
# under its target (`loc.inner.slope`, `age.y_mis`); StanBlocks inlines the
# submodel under `_`-joined names owned by that target declaration. Only
# sampled outputs that declaration owns are candidates; the validator names
# any RK declaration without a matching sampled SB output.
function _brm_transport_scoped_pairs!(pairs, correlations, simplexes,
        kind::Symbol, target::Symbol, rk, d, stan, stan_pos;
        only_declaration::Union{Nothing,Symbol}=nothing)
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
        !isnothing(only_declaration) && declaration !== only_declaration && continue
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
        address = (; kind, target, declaration)
        if only(transforms) === :simplex
            _brm_transport_simplex!(pairs, simplexes, address,
                [e.name for e in elements], declaration, output, stan)
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

function _brm_transport_pairs!(pairs, correlations, simplexes, centered,
        ::Val{:submodel}, record, rk, d, stan, stan_pos)
    _brm_transport_scoped_pairs!(pairs, correlations, simplexes,
        :submodel, record.target, rk, d, stan, stan_pos)
end

function _brm_transport_pairs!(pairs, correlations, simplexes, centered,
        ::Val{:missing_value}, record, rk, d, stan, stan_pos)
    _brm_transport_scoped_pairs!(pairs, correlations, simplexes,
        :missing_value, record.target, rk, d, stan, stan_pos;
        only_declaration=Symbol(record.target, ".y_mis"))
end

# The Stan declaration type and RK layout transform of each vector family the
# shared emitter declares. Within a family both backends use the same
# transform (an ordered vector's first element, then log increments), so
# its free coordinates are equal; a simplex moves as a `:simplex` block.
const _BRM_TRANSPORT_VECTOR_FAMILIES = Dict(
    :ordered_normal => ("ordered", :ordered),
    :vector_normal => ("vector", :identity),
    :simplex_dirichlet => ("simplex", :simplex))

# A vector parameter — a response's cutpoints or thresholds, or an authored
# `Dirichlet` simplex — keeps its declaration name on both backends, like a
# scalar, and pairs element by element with the same-named SB draw carrier.
# A declaration with no such carrier stays unpaired, so the validator names it.
function _brm_transport_pairs!(pairs, correlations, simplexes, centered,
        ::Val{:vector}, record, rk, d, stan, stan_pos)
    declaration = record.declaration
    idxs = _brm_carrier_indices(d.outputs,
        o -> o.name === declaration && !isnothing(o.declaration) &&
            o.declaration.target === declaration)
    isempty(idxs) && return
    length(idxs) == 1 || error(
        "brm_coordinate_transport: vector `$declaration` matches $(length(idxs)) " *
        "SB draw carriers; expected one")
    output = d.outputs[only(idxs)]
    stan_type, transform = _BRM_TRANSPORT_VECTOR_FAMILIES[record.family]
    address = (; kind=:vector, declaration)
    string(output.type) == stan_type || _brm_transport_error(:parameterization_mismatch,
        "brm_coordinate_transport: RK `$(record.family)` vector `$declaration` " *
        "pairs with SB `$(output.type)` carrier `$(output.name)`; the two " *
        "transforms differ, so their coordinates are not a permutation")
    elements = Pair{Int,Symbol}[]
    for (rk_name, rk_transform) in zip(_rk_layout_coordinate_names(rk),
            _rk_layout_coordinate_transforms(rk))
        scoped = _brm_transport_scoped(rk_name)
        !isnothing(scoped) && first(scoped) === declaration &&
            length(last(scoped)) == 1 || continue
        rk_transform === transform || error(
            "brm_coordinate_transport: internal: RK `$(record.family)` vector " *
            "`$declaration` has layout transform `$rk_transform`")
        push!(elements, only(last(scoped)) => rk_name)
    end
    sort!(elements; by=first)
    first.(elements) == 1:length(elements) || error(
        "brm_coordinate_transport: RK vector `$declaration` has element " *
        "coordinates $(Tuple(first.(elements))), not 1:$(length(elements))")
    rk_names = last.(elements)
    # A one-element simplex or an empty threshold vector has no free coordinate.
    isempty(rk_names) && isempty(_brm_element_coordinates(output, stan)) && return
    if transform === :simplex
        _brm_transport_simplex!(pairs, simplexes, address, rk_names, declaration,
            output, stan)
        return
    end
    names = stan[_brm_element_coordinates(output, stan)]
    length(names) == length(rk_names) || error(
        "brm_coordinate_transport: vector $(address) has $(length(rk_names)) RK " *
        "and $(length(names)) Stan coordinates")
    for (j, (rk_name, stan_name)) in enumerate(zip(rk_names, names))
        push!(pairs, BRMCoordinatePair((; address..., index=j), rk_name, stan_name,
            declaration, (j,), :identity))
    end
end

# `Ordinal(...; per_threshold=...)` coefficients: RK's `terms × stages` matrix
# and SB's `stages`-array of `terms`-vectors hold the same independent
# coefficient per (stage, term), paired by those positions. Stan stores the
# array stage-major — element `[s][t]` at block offset `(s-1)T + t` — which is
# also RK's column-major `(t, s)` order, while BridgeStan's names enumerate the
# block stage-fastest; each pair keeps the unconstrained name at its true
# position and the constrained `<carrier>.<s>.<t>` for the physical check.
function _brm_transport_pairs!(pairs, correlations, simplexes, centered,
        ::Val{:threshold_coefficients}, record, rk, d, stan, stan_pos)
    declaration = record.declaration
    idxs = _brm_carrier_indices(d.outputs,
        o -> o.name === declaration && !isnothing(o.declaration) &&
            o.declaration.target === declaration)
    isempty(idxs) && return
    length(idxs) == 1 || error(
        "brm_coordinate_transport: threshold coefficients `$declaration` match " *
        "$(length(idxs)) SB draw carriers; expected one")
    output = d.outputs[only(idxs)]
    T, S = length(record.terms), record.stages
    string(output.type) == "vector" && length(output.size) == 2 ||
        _brm_transport_error(:parameterization_mismatch,
            "brm_coordinate_transport: RK threshold coefficients `$declaration` " *
            "pair with SB `$(output.type)$(output.size)` carrier, not a " *
            "stages-array of term vectors")
    positions = sort(_brm_element_coordinates(output, stan))
    n = S * T
    n == 0 && isempty(positions) && return
    positions == positions[1]:(positions[1] + n - 1) || error(
        "brm_coordinate_transport: SB threshold coefficients `$declaration` " *
        "occupy Stan positions $(positions); expected $n contiguous ones " *
        "($S stages × $T terms)")
    for s in 1:S, t in 1:T
        address = (; kind=:threshold_coefficient, response=record.response,
            stage=s, term=record.terms[t])
        push!(pairs, BRMCoordinatePair(address, Symbol(declaration, ".", t, ".", s),
            stan[positions[1] + (s - 1) * T + t - 1], declaration, (t, s), :identity,
            string(declaration, ".", s, ".", t)))
    end
end

function _brm_transport_pairs!(pairs, correlations, simplexes, centered,
        ::Val{kind}, record, rk, d, stan, stan_pos) where {kind}
    error("brm_coordinate_transport: internal: no pairing for RK record kind `$kind`")
end

"""
    brm_rk_to_stan(t::BRMCoordinateTransport, v) -> Vector
    brm_stan_to_rk(t::BRMCoordinateTransport, v) -> Vector

Move an unconstrained point, or a gradient with respect to it, between the RK
and Stan coordinate orders of a transport without simplex or centered blocks.
Both are the same permutation: each pair has equal unconstrained values, so a
Stan gradient maps back to RK order exactly as the point does. A transport with
simplex or centered blocks refuses both, since its points and gradients move
differently; use [`brm_rk_point_to_stan`](@ref), [`brm_stan_point_to_rk`](@ref)
and [`brm_stan_gradient_to_rk`](@ref). Neither function mutates `v`.
"""
function brm_rk_to_stan(t::BRMCoordinateTransport, v::AbstractVector)
    _brm_transport_permutation_only(t, "brm_rk_to_stan")
    _brm_rk_order_to_stan(t, v)
end

function brm_stan_to_rk(t::BRMCoordinateTransport, v::AbstractVector)
    _brm_transport_permutation_only(t, "brm_stan_to_rk")
    _brm_stan_order_to_rk(t, v)
end

_brm_transport_permutation_only(t, caller) =
    isempty(t.simplexes) && isempty(t.centered) || error(
    "$caller: this transport pairs $(length(t.simplexes)) simplex block(s) and " *
    "$(length(t.centered)) centered block(s), whose points and gradients do not " *
    "move by one permutation. Use `brm_rk_point_to_stan` / " *
    "`brm_stan_point_to_rk` for points, `brm_stan_gradient_to_rk` for gradients " *
    "and `brm_stan_logdensity_to_rk` for log densities.")

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

# A centered block's effect factor `C = diag(tau) * L` at RK coordinates `u`,
# with `tau` and `L`: the scales are `exp` of their coordinates, and RK and
# Stan share the `cholesky_factor_corr` transform of the free values.
function _brm_centered_factor(block, u::AbstractVector)
    tau = exp.(u[block.scale_positions])
    L = _adaptive_cholesky_corr(u[block.cholesky_positions], block.K)
    tau .* L, tau, L
end

# `log |det(d b / d z)|` of a centered block at RK coordinates `u`: each of its
# `G` levels maps `z -> C * z` with a triangular `C`.
_brm_centered_logjacobian(block, u::AbstractVector) =
    block.G * sum(log, diag(first(_brm_centered_factor(block, u))))

# `J' * GL` for `L = _adaptive_cholesky_corr(raw, K)`, where `GL` is a gradient
# with respect to the entries of `L`. Row `i` holds `L[i, j] = s[j] * z[j]` for
# `j < i` and `L[i, i] = s[i]`, with `z[j] = tanh(raw[p])`, `s[1] = 1` and
# `s[j + 1] = s[j] * sqrt(1 - z[j]^2)`; each row is reversed from its diagonal.
function _brm_cholesky_corr_pullback(raw::AbstractVector, K::Int, GL::AbstractMatrix)
    out = zeros(promote_type(float(eltype(raw)), eltype(GL)), length(raw))
    p = 0
    for i in 2:K
        z = tanh.(raw[(p + 1):(p + i - 1)])
        w = sqrt.(1 .- z .* z)
        s = cumprod(vcat(one(eltype(w)), w))
        gs = GL[i, i]
        for j in (i - 1):-1:1
            out[p + j] = (GL[i, j] * w[j] - gs * z[j]) * s[j] * w[j]
            gs = gs * w[j] + GL[i, j] * z[j]
        end
        p += i - 1
    end
    out
end

# Adds a centered block's pullback to `out`, the permuted Stan gradient `g`:
# the effects `B = C * Z` pull back to `C' * dB` on the standardized draws and
# to the block's scale and correlation coordinates, plus the gradient of its
# log-Jacobian `G * (sum(log, tau) + sum(log, diag(L)))`.
function _brm_centered_pullback!(out, block, u::AbstractVector, g::AbstractVector)
    C, tau, L = _brm_centered_factor(block, u)
    Z = u[block.rk_positions]
    dB = g[block.stan_positions]
    out[block.rk_positions] = transpose(C) * dB
    # `d B[k, h] / d log(tau[k]) == B[k, h]`.
    out[block.scale_positions] .+= vec(sum(dB .* (C * Z); dims=2)) .+ block.G
    isempty(block.cholesky_positions) && return out
    # `d B[i, h] / d L[i, j] == tau[i] * Z[j, h]`; `d log(L[i, i]) == 1 / L[i, i]`.
    GL = (tau .* dB) * transpose(Z)
    for i in 1:block.K
        GL[i, i] += block.G / L[i, i]
    end
    out[block.cholesky_positions] .+= _brm_cholesky_corr_pullback(
        u[block.cholesky_positions], block.K, GL)
    out
end

"""
    brm_rk_point_to_stan(t::BRMCoordinateTransport, u_rk) -> Vector
    brm_stan_point_to_rk(t::BRMCoordinateTransport, u_stan) -> Vector

Move an unconstrained POINT between the RK and Stan coordinate frames. Paired
coordinates move by `t.permutation`; each `t.simplexes` block goes through the
physical simplex, from RK's stick-breaking coordinates to Stan's isometric
log-ratio coordinates or back; each `t.centered` block's standardized draws
`z` become the effects `b = diag(tau) * L * z` per level, or back, with the
point's own scales `tau` and correlation factor `L`. The two functions are
inverses and both describe the same physical parameter point. They equal
[`brm_rk_to_stan`](@ref) / [`brm_stan_to_rk`](@ref) for a transport without
simplex or centered blocks. Neither mutates its input. Use
[`brm_stan_gradient_to_rk`](@ref) for gradients and
[`brm_stan_logdensity_to_rk`](@ref) for log densities.
"""
function brm_rk_point_to_stan(t::BRMCoordinateTransport, u::AbstractVector)
    out = _brm_rk_order_to_stan(t, u)
    for block in t.simplexes
        logx, _ = _brm_rk_simplex_log(u[block.rk_positions])
        out[block.stan_positions] = _brm_stan_simplex_free(logx)
    end
    for block in t.centered
        out[block.stan_positions] =
            first(_brm_centered_factor(block, u)) * u[block.rk_positions]
    end
    out
end

function brm_stan_point_to_rk(t::BRMCoordinateTransport, y::AbstractVector)
    out = _brm_stan_order_to_rk(t, y)
    for block in t.simplexes
        out[block.rk_positions] =
            _brm_rk_simplex_free(_brm_stan_simplex_log(y[block.stan_positions]))
    end
    # A block's scales and correlation factor are already in RK order.
    for block in t.centered
        out[block.rk_positions] = LowerTriangular(first(_brm_centered_factor(block, out))) \
            y[block.stan_positions]
    end
    out
end

"""
    brm_stan_gradient_to_rk(t::BRMCoordinateTransport, u_rk, stan_gradient) -> Vector

Pull back the gradient of Stan's normalized log density with Jacobian
(`propto=false, jacobian=true`), taken at `brm_rk_point_to_stan(t, u_rk)`, to
RK's coordinates at `u_rk`; the result is RK's log-density gradient. Paired
coordinates move by `t.permutation`; a simplex block multiplies by the
transpose of the exact Jacobian of its stick-breaking-to-log-ratio map, whose
log-Jacobian difference is the constant `t.logdensity_offset`. A centered
block multiplies by the transpose of the Jacobian of `z -> b`, which also
moves its scale and correlation coordinates, and adds the gradient of that
map's log-Jacobian (see [`brm_stan_logdensity_to_rk`](@ref)). Neither input
is mutated.
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
    for block in t.centered
        _brm_centered_pullback!(out, block, u, g)
    end
    out
end

"""
    brm_stan_logdensity_to_rk(t::BRMCoordinateTransport, u_rk, stan_value) -> Real

RK's log density at `u_rk`, from Stan's normalized log density with Jacobian
(`propto=false, jacobian=true`) `stan_value` at
`brm_rk_point_to_stan(t, u_rk)`: `stan_value + t.logdensity_offset` (the
constant simplex offset) plus, for each centered block, the log-Jacobian
`G * log(det(diag(tau) * L))` of its map `z -> b` over its `G` levels at
`u_rk`. Works for every transport; without centered blocks it equals
`stan_value + t.logdensity_offset`.
"""
function brm_stan_logdensity_to_rk(t::BRMCoordinateTransport, u::AbstractVector,
        stan_value::Real)
    length(u) == length(t.rk_names) || error(
        "brm_stan_logdensity_to_rk: got $(length(u)) point values for " *
        "$(length(t.rk_names)) RK coordinates")
    stan_value + getfield(t, :logdensity_offset) +
        sum((_brm_centered_logjacobian(block, u) for block in t.centered); init=0.0)
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
its structurally fixed unit diagonal norm and zero upper triangle, every
simplex agrees as a whole vector of `K` elements, and every centered block's
Stan effects equal `diag(tau) * L * z` computed from RK's own constrained
scales, correlation factor and standardized draws. The Stan point is
`brm_rk_point_to_stan(t, u_rk)`. `stan_model` is the compiled
`BridgeStan.StanModel` whose `param_unc_names` built `t`.
Returns `(; pairs::Int, factors::Int, simplexes::Int, centered::Int,
max_error::Float64)`: `pairs` counts checked elements outside correlation
factors, simplexes and centered blocks, `factors` counts whole correlation
matrices, `simplexes` whole simplexes, `centered` whole centered blocks (all
`K × G` effects), and `max_error` is the largest absolute physical difference
over these elements and every matrix, simplex and effect entry. Every
comparison must meet
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
        pair.relation in (:cholesky, :simplex, :centered) && continue
        value = _brm_rk_declaration_value(rk_values, pair.rk_declaration)
        rk_value = isempty(pair.rk_index) ? value : value[pair.rk_index...]
        compare("$(pair.address) (`$(pair.rk)` / `$(pair.stan_value)`)", rk_value,
            _brm_physical(Val(pair.relation), stan_value(pair.stan_value)))
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
        column = get(block, :column, nothing)
        column === nothing || (x = x[:, column])
        length(x) == block.K || error(
            "brm_check_coordinate_transport: RK simplex `$(block.rk_declaration)` " *
            "has $(length(x)) elements, expected $(block.K)")
        for i in 1:block.K
            compare("simplex $(block.stan_carrier)[$i]", x[i],
                stan_value(string(block.stan_carrier, ".", i)))
        end
    end
    for block in t.centered
        z = _brm_rk_declaration_value(rk_values, block.rk_declaration)
        tau = map(block.rk_scale_values) do (declaration, index)
            value = _brm_rk_declaration_value(rk_values, declaration)
            isempty(index) ? value : value[index...]
        end
        L = isnothing(block.rk_L) ? Matrix{Float64}(I, block.K, block.K) :
            _brm_rk_declaration_value(rk_values, block.rk_L)
        for h in 1:block.G, k in 1:block.K
            effect = tau[k] * sum(L[k, j] * z[block.rk_level[h], j] for j in 1:k)
            compare("centered effect `$(block.stan_values[k, h])`", effect,
                stan_value(block.stan_values[k, h]))
        end
    end
    (; pairs=checked, factors=length(t.correlations),
       simplexes=length(t.simplexes), centered=length(t.centered), max_error)
end
