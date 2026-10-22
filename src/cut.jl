# src/cut.jl — cut posteriors on the RK backend (todo `0fdr8mz`; shape decided
# on `1bgtn3d`). A cut fits an upstream part of a model first, then one
# downstream conditional per upstream draw. Every conditional is a binding of
# one compiled RK program, prepared once per draw, so nothing is compiled per
# draw. Two front ends build the conditionals:
#
# - the coordinate cut pins the joint model's upstream declarations at each
#   draw (RKPPL `merge` pins); RK's partial evaluation then folds what depends
#   only on them;
# - the explicit cut binds a downstream model to stage-2 data the modeler
#   computes from each draw, for a joint model whose efficient plan fuses what
#   the cut separates.
#
# Sampling is the caller's (`brm_cut_fit(sampler, family)`); BRM supplies the
# targets, starts and features a family sampler reads.

"""
    BRMCutFamily

The downstream conditionals of a cut posterior, one per upstream draw, on one
compiled RK program (from [`brm_cut_family`](@ref)).

- `targets[k]`: a LogDensityProblems target (density and gradient) in the
  downstream coordinates `names`, conditional on upstream draw `k`.
- `upstream[:, k]`: that draw, in the upstream fit's unconstrained
  coordinates `upstream_names`.
- `starts[k]`, `features[k]`: a start point for target `k`, and the draw's
  coordinates as a vector a family sampler can compare conditionals by.
- `joint_names`: the joint model's coordinates, when the family can place its
  draws there (`nothing` otherwise).
"""
struct BRMCutFamily{T}
    targets::Vector{T}
    names::Vector{Symbol}
    upstream::Matrix{Float64}
    upstream_names::Vector{Symbol}
    starts::Vector{Vector{Float64}}
    features::Vector{Vector{Float64}}
    joint_names::Union{Nothing,Vector{Symbol}}
    # Each joint coordinate's source: `(:upstream, i)` or `(:downstream, i)`.
    joint_sources::Union{Nothing,Vector{Tuple{Symbol,Int}}}
end
Base.length(f::BRMCutFamily) = length(f.targets)
Base.show(io::IO, f::BRMCutFamily) = print(io, "BRMCutFamily(", length(f),
    " conditionals over ", length(f.names), " downstream coordinates",
    f.joint_names === nothing ? "" : ", joint $(length(f.joint_names))", ")")

"""
    brm_cut_family(joint::RKBRMI, upstream::RKBRMI, U; ad_backend, start=nothing)
    brm_cut_family(inputs, downstream::RKBRMI, data, upstream::RKBRMI, U;
                   ad_backend, joint=nothing, start=nothing)

One downstream conditional per upstream draw `U[:, k]`: the columns of `U` are
draws in `upstream`'s unconstrained coordinates (`rk_restore_draws` order),
for example the posterior positions of a fit of `upstream`.

**Coordinate cut** (the default). `upstream` is `joint`'s model on the data
without the downstream responses, so the declarations it samples are the
upstream part (an omitted response is not conditioned on, and what no
remaining likelihood reaches leaves the sampler). Each conditional is `joint`
with those declarations pinned at draw `k`: the downstream coordinates of
`joint`, with density equal to `joint`'s up to a per-draw constant. RK folds
what depends only on the pinned values into constants of that draw's kernel.
A declaration only partly upstream (a shared `(… | p | g)` block whose margins
split across the stages) is refused.

**Explicit cut.** For a joint model whose efficient computation fuses what the
cut separates, `downstream` is a model of stage 2 alone that reads the
declared stage-1 outputs as data, built once on its data. Draw `k`'s
conditional is `downstream` on `merge(data, inputs(U[:, k]))`: `inputs` takes
the upstream draw (a vector in `upstream`'s unconstrained coordinates) and
returns the stage-2 columns, and may spend any compute preparing them.
Preprocessing stays frozen at `downstream`'s own data, as `reprocess` freezes
it. `inputs` may be a tuple of upstream value names instead, such as
`(:conc,)`: each draw then binds those graph values of `upstream` under the
same names. Pass the joint model as `joint=` to place the draws in its
coordinates: every joint coordinate must be an upstream or a downstream
coordinate of the same name.

`ad_backend` is the gradient backend of the targets, as in
[`rk_logdensity_problem`](@ref). `start` is every target's start point in the
downstream coordinates (default zeros).
"""
function brm_cut_family end

function brm_cut_family(joint::RKBRMI, upstream::RKBRMI, U::AbstractMatrix;
        ad_backend, start=nothing)
    prefix = "brm_cut_family"
    up_names = _rk_layout_coordinate_names(upstream)
    U = _brm_cut_draws(U, up_names; prefix)
    joint_names = _rk_layout_coordinate_names(joint)
    up_entries = _brm_cut_entries(upstream, up_names)
    joint_entries = _brm_cut_entries(joint, joint_names)
    for (name, positions) in up_entries
        haskey(joint_entries, name) && joint_names[joint_entries[name]] == up_names[positions] &&
            continue
        error("$prefix: the upstream declaration `$name` is not a declaration of the " *
            "joint model with the same coordinates. A declaration only partly " *
            "upstream (a shared |ID| block whose margins split across the stages) " *
            "cannot be pinned yet; for a model whose stages fuse, build the " *
            "explicit cut, brm_cut_family(inputs, downstream, data, upstream, U).")
    end
    pinned = Set{Int}(i for name in keys(up_entries) for i in joint_entries[name])
    free = [i for i in eachindex(joint_names) if !(i in pinned)]
    restored = rk_restore_draws(upstream, U)
    pins = [(; _brm_cut_pins(restored, k)...) for k in axes(U, 2)]
    lowered = _rk_cut_lowering(joint, first(pins))
    plans = [_rk_cut_bind(lowered, p) for p in pins]
    built = _rk_build_plan(first(plans))
    names = _rk_built_coordinate_names(built)
    names == joint_names[free] || error(
        "$prefix: internal: the pinned program's coordinates $(names) are not the " *
        "joint model's free coordinates $(joint_names[free])")
    targets = [_rk_cut_target(built, plan; ad_backend) for plan in plans]
    joint_sources = _brm_cut_joint_sources(joint_names, up_names, names; prefix)
    BRMCutFamily(targets, names, U, up_names, _brm_cut_starts(start, names, U),
        _brm_cut_features(U), joint_names, joint_sources)
end

function brm_cut_family(inputs, downstream::RKBRMI, data, upstream::RKBRMI,
        U::AbstractMatrix; ad_backend, joint=nothing, start=nothing)
    prefix = "brm_cut_family"
    up_names = _rk_layout_coordinate_names(upstream)
    U = _brm_cut_draws(U, up_names; prefix)
    names = _rk_layout_coordinate_names(downstream)
    binder = rk_design_binder(downstream)
    targets = map(axes(U, 2)) do k
        stage2 = inputs(U[:, k])
        stage2 isa NamedTuple || throw(ArgumentError(
            "$prefix: `inputs` must return the stage-2 columns as a NamedTuple; " *
            "got $(typeof(stage2)) for upstream draw $k"))
        plan = _rk_design_plan(binder, merge(data, stage2))
        _rk_cut_target(downstream.model, plan; ad_backend)
    end
    joint_names, joint_sources = joint === nothing ? (nothing, nothing) :
        (collect(Symbol, _rk_layout_coordinate_names(joint)),
         _brm_cut_joint_sources(_rk_layout_coordinate_names(joint), up_names, names; prefix))
    BRMCutFamily(targets, collect(Symbol, names), U, up_names,
        _brm_cut_starts(start, names, U), _brm_cut_features(U), joint_names, joint_sources)
end

function brm_cut_family(interface::Tuple{Vararg{Symbol}}, downstream::RKBRMI, data,
        upstream::RKBRMI, U::AbstractMatrix; kwargs...)
    isempty(interface) && throw(ArgumentError(
        "brm_cut_family: the interface names no upstream value"))
    brm_cut_family(_rk_value_reader(upstream, interface), downstream, data, upstream, U;
        kwargs...)
end

"""
    brm_cut_fit(sampler, family::BRMCutFamily) -> NamedTuple

Sample every conditional of `family` with `sampler(targets, starts, features)`,
which returns one coordinates × draws matrix per target (a family sampler, or
independent fits), and place the draws:

- `downstream[k]`: target `k`'s draws in `family.names`;
- `joint`: when the family knows the joint model, its coordinates × draws
  matrix; every column takes its upstream coordinates from the draw its
  conditional was built on, `family.upstream[:, source[j]]`;
- `joint_names`, `source`.
"""
function brm_cut_fit(sampler, family::BRMCutFamily)
    prefix = "brm_cut_fit"
    downstream = sampler(family.targets, family.starts, family.features)
    length(downstream) == length(family) || error(
        "$prefix: the sampler returned $(length(downstream)) results for " *
        "$(length(family)) conditionals")
    downstream = map(enumerate(downstream)) do (k, D)
        D isa AbstractMatrix && size(D, 1) == length(family.names) || throw(DimensionMismatch(
            "$prefix: the sampler's result $k is not a $(length(family.names)) × draws " *
            "matrix (the conditionals' coordinates); got $(summary(D))"))
        Matrix{Float64}(D)
    end
    source = reduce(vcat, [fill(k, size(D, 2)) for (k, D) in enumerate(downstream)]; init=Int[])
    family.joint_sources === nothing && return (; downstream, joint=nothing,
        joint_names=nothing, source)
    columns = [D[:, j] for D in downstream for j in axes(D, 2)]
    joint = _brm_cut_assemble(family.joint_sources, family.upstream, source, columns)
    (; downstream, joint, joint_names=family.joint_names, source)
end

"""
    brm_cut_point(U) -> Matrix

The poor cut's upstream point: the coordinate-wise median of the upstream
draws (the columns of `U`) in their unconstrained coordinates, as a one-column
matrix to build a one-conditional family on.
"""
brm_cut_point(U::AbstractMatrix) =
    reshape([Statistics.median(view(U, i, :)) for i in axes(U, 1)], :, 1)

"""
    brm_cut_propagate(fit, family::BRMCutFamily, U; rng=Random.default_rng()) -> NamedTuple

The poor cut's draws. `family` has one conditional, at the upstream point
(`brm_cut_point(U)`), and `fit = brm_cut_fit(sampler, family)` its downstream
draws. Each upstream draw `U[:, k]` is paired with downstream draw `σ(k)`, a
draw without replacement while the fit has enough draws, and placed in the
joint coordinates: `joint` (joint coordinates × size(U, 2)), `joint_names`,
and `pairing` (σ).
"""
function brm_cut_propagate(fit, family::BRMCutFamily, U::AbstractMatrix;
        rng::Random.AbstractRNG=Random.default_rng())
    prefix = "brm_cut_propagate"
    length(family) == 1 || throw(ArgumentError(
        "$prefix: the poor cut's family has one conditional, at the upstream point; " *
        "this one has $(length(family))"))
    family.joint_sources === nothing && throw(ArgumentError(
        "$prefix: the family does not know the joint model; build it with `joint=`"))
    U = _brm_cut_draws(U, family.upstream_names; prefix)
    D = only(fit.downstream)
    n = size(D, 2)
    n > 0 || throw(ArgumentError("$prefix: the fit has no downstream draws"))
    pairing = n >= size(U, 2) ? Random.randperm(rng, n)[1:size(U, 2)] :
        rand(rng, 1:n, size(U, 2))
    joint = _brm_cut_assemble(family.joint_sources, U, collect(axes(U, 2)),
        [D[:, j] for j in pairing])
    (; joint, joint_names=family.joint_names, pairing)
end

"""
    rk_draw_values(rk::RKBRMI, U, names) -> NamedTuple

The graph values `names` of `rk` (formula values, `=` assignments, `@plate`
cell outputs) at each draw, the columns of `U` in `rk`'s unconstrained
coordinates: one vector of per-draw values per name, from one prepared value
kernel.
"""
function rk_draw_values(rk::RKBRMI, U::AbstractMatrix, names)
    names = Tuple(Symbol.(names))
    U = _brm_cut_draws(U, _rk_layout_coordinate_names(rk); prefix="rk_draw_values")
    read = _rk_value_reader(rk, names)
    values = [read(U[:, k]) for k in axes(U, 2)]
    NamedTuple{names}(Tuple([v[name] for v in values] for name in names))
end

function _brm_cut_draws(U, names; prefix)
    size(U, 1) == length(names) || throw(DimensionMismatch(
        "$prefix: the draws have $(size(U, 1)) rows; the model has " *
        "$(length(names)) coordinates (draws are columns)"))
    size(U, 2) > 0 || throw(ArgumentError("$prefix: no upstream draws"))
    Matrix{Float64}(U)
end

# A model's declarations by layout name (scoped `b_p_subject.tau`), with the
# positions of their coordinates.
function _brm_cut_entries(rk::RKBRMI, names)
    entries = Dict{Symbol,Vector{Int}}()
    for entry in rk.model.layout.entries
        root = string(entry.name)
        entries[entry.name] = [i for (i, n) in enumerate(names)
            if n === entry.name || startswith(string(n), root * ".")]
    end
    entries
end

# Draw `k` of one restored declaration: numeric arrays hold draws in their last
# axis, otherwise one array per draw.
_brm_cut_draw(v::AbstractVector{<:Number}, k) = v[k]
_brm_cut_draw(v::AbstractMatrix{<:Number}, k) = v[:, k]
_brm_cut_draw(v::AbstractVector{<:AbstractArray}, k) = v[k]

# Pins of draw `k`, by the scoped names RKPPL pins read (`b_p_subject.tau`).
function _brm_cut_pins(restored::NamedTuple, k; prefix="")
    pins = Pair{Symbol,Any}[]
    for (name, value) in pairs(restored)
        scoped = Symbol(prefix, name)
        value isa NamedTuple ? append!(pins, _brm_cut_pins(value, k; prefix=string(scoped, "."))) :
            push!(pins, scoped => _brm_cut_draw(value, k))
    end
    pins
end

function _brm_cut_joint_sources(joint_names, up_names, names; prefix)
    upstream = Dict(n => i for (i, n) in enumerate(up_names))
    downstream = Dict(n => i for (i, n) in enumerate(names))
    both = sort!(collect(intersect(keys(upstream), keys(downstream))))
    isempty(both) || error("$prefix: coordinates $both are in both the upstream and " *
        "the downstream model")
    map(joint_names) do n
        haskey(downstream, n) && return (:downstream, downstream[n])
        haskey(upstream, n) && return (:upstream, upstream[n])
        error("$prefix: the joint coordinate `$n` is in neither the upstream nor the " *
            "downstream model")
    end
end

function _brm_cut_assemble(sources, U, upstream_columns, downstream_columns)
    joint = Matrix{Float64}(undef, length(sources), length(downstream_columns))
    for (j, (k, d)) in enumerate(zip(upstream_columns, downstream_columns)),
            (i, (from, at)) in enumerate(sources)
        joint[i, j] = from === :upstream ? U[at, k] : d[at]
    end
    joint
end

_brm_cut_starts(::Nothing, names, U) = [zeros(length(names)) for _ in axes(U, 2)]
function _brm_cut_starts(start, names, U)
    length(start) == length(names) || throw(DimensionMismatch(
        "brm_cut_family: `start` has $(length(start)) entries; the conditionals have " *
        "$(length(names)) coordinates"))
    [collect(Float64, start) for _ in axes(U, 2)]
end
_brm_cut_features(U) = [U[:, k] for k in axes(U, 2)]

# Implemented by `BayesianRegressionModelsReactiveKernelsExt`.
function _rk_cut_lowering end
function _rk_cut_bind end
function _rk_build_plan end
function _rk_built_coordinate_names end
function _rk_cut_target end
function _rk_value_reader end
