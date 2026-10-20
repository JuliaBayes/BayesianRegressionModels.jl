using LogExpFunctions, InverseFunctions, Distributions, ElasticArrays, LogDensityProblems, LinearAlgebra, SpecialFunctions
import CategoricalArrays as CA

"""
    VBRMI(brmi::BRMI) -> VBRMI

Vectorised-Julia backend: materialises `brmi` into a
`LogDensityProblems`-compatible struct over a flat unconstrained vector.
Holds `meta.blocks` (per-block tuples of [`Part`](@ref)s) and
`meta.materialized` (per-name materialised columns).

```julia
brmi  = @brm df (y ~ 1 + a)
vbrmi = VBRMI(brmi)
d     = LogDensityProblems.dimension(vbrmi)
ld    = LogDensityProblems.logdensity(vbrmi, randn(d))
```

For Stan-grade sampling use [`SBBRMI`](@ref) instead; `VBRMI` is the
pure-Julia path (Mooncake AD compatible, no BridgeStan required).
"""
struct VBRMI{P<:BRMI,M<:NamedTuple}
    parent::P
    meta::M
end
function VBRMI(p::BRMI)
    # Reject the backend capability at the model boundary before walking any
    # earlier composite-prior declaration. Otherwise the formula-order prior
    # fails first with a misleading marker-constructor MethodError.
    for value in values(p.operations)
        value isa NamedColumn || continue
        op = parent(value)
        op isa ExprColumn && getf(op) === (~) || continue
        lhs, _ = getargs(op, 2)
        lhs isa JointResponseColumn || continue
        error(
            "VBRMI: correlated joint outcomes $(joint_response_names(lhs)) are " *
            "supported by the StanBlocks backend only. Construct `SBBRMI(brmi)`; " *
            "VBRMI cannot silently replace one multivariate density with " *
            "independent scalar terms.")
    end
    VBRMI(p, finalize(foldl(
        vmeta, p.operations; init=(;materialized=(;), blocks=(;)))))
end

"""
    Part{F<:Function,D<:NamedTuple}

Bundles a marker function `func` (the sole dispatch tag for per-part behavior —
`nparams`, `lprior!`, `Base.show`) with a NamedTuple `data` of per-kind state.
Each part owns its buffer and knows how to consume a slice of the unconstrained
vector, constrain it into the buffer, and return its log-prior contribution.

`meta.blocks` is a NamedTuple keyed by coalescing Symbol (`:__population__`, or
a grouping-factor name like `:g1`); each value is a tuple of parts.
"""
struct Part{F<:Function,D<:NamedTuple}
    func::F
    data::D
end

"""IID-normal population slots."""
function normal end
"""Grouped column set, correlated via a shared L."""
function grouped_normal end
"""LKJ-Cholesky owning the shared L for a grouped block."""
function chol end
"""Stratified grouped-normal: per-row L picked from `Ls[:,:,stratum_idx[i]]`. Implements brms `gr(g, by=b)` — each stratum gets its own covariance structure."""
function grouped_normal_by end
"""Array-of-LKJ-Cholesky owning per-stratum Ls for a `grouped_normal_by` block."""
function chol_by end
"""Unit simplex owning `values`; `length(values)` entries sum to 1, parameterized by `length(values)-1` unconstrained reals via stick-breaking."""
function simplex end

"""
    GrGroup(group::NamedColumn, by::NamedColumn)

Walker-side grouping tag for `gr(group, by=strata)`. `growblock!!` /
`_gc_idx` dispatch on it to allocate a `grouped_normal_by` block whose
per-row covariance is keyed off `by`.
"""
struct GrGroup
    group
    by
end

"""
    finalize(meta) / finalize(parts) / finalize(part, acc)

Threaded finalize: fold over each block's parts, each part emitting zero or
more replacement parts. `acc` is visible to later parts so a part can observe
what already landed (e.g. a `grouped_normal` could skip re-allocating L if a
`chol` sibling is already present — useful for the future multi-grouped_normal
case). Default per-part method is a pass-through; specializations emit their
own expansion (e.g. `grouped_normal` prepends a `chol` sharing the same L).
"""
finalize(x::NamedTuple) = merge(x, (; blocks = map(finalize, x.blocks)))
finalize(parts::Tuple) = foldl((acc, p) -> (acc..., finalize(p, acc)...), parts; init=())
finalize(p::Part, _) = (p,)
finalize(p::Part{typeof(grouped_normal)}, _) = let n = size(p.data.values, 2), L = zeros(n, n)
    (Part(chol, (; L)), Part(grouped_normal, merge(p.data, (; L))))
end
finalize(p::Part{typeof(grouped_normal_by)}, _) = let
    n = size(p.data.values, 2)
    Ls = zeros(n, n, p.data.n_strata)
    (Part(chol_by, (; Ls)), Part(grouped_normal_by, merge(p.data, (; Ls))))
end
rmerge(x::NamedTuple, y::NamedTuple) = begin
    xykeys = (intersect(keys(x), keys(y))...,)
    merge(x, y, map(rmerge, NamedTuple{xykeys}(x), NamedTuple{xykeys}(y)))
end
rmerge(::AbstractDict, ::AbstractDict) = error("Can only rmerge NamedTuples for now!")
rmerge(x, y) = y
vmeta(meta, x::NamedColumn) = begin
    meta, m = vmeta(meta, parent(x))::Tuple
    rmerge(meta, (;materialized=(;name(x)=>m)))
end
vmeta(meta, x::DataColumn) = meta, x
vmeta(meta, x::ExprColumn{typeof(assign)}) = vmeta_assignment(meta, getargs(x)...; getkwargs(x)...)
vmeta(meta, x::ExprColumn{typeof(~)}) = vmeta_sampling(meta, getargs(x)...; getkwargs(x)...)

vmeta_assignment(meta, ::Symbol, x) = meta, vmaterialize(vbroadcasted(x; meta))
"""
    vbroadcasted(x; meta) -> Broadcasted

vimpl's column materialiser. Resolves a column-tree value to a
broadcastable expression by walking [`NamedColumn`](@ref),
[`DataColumn`](@ref), [`ExprColumn`](@ref) leaves. Used inside
[`vmeta_sampling_rhs`](@ref) method bodies to resolve argument columns
before scaling by a fresh β or wrapping in a likelihood. Add a method
for a custom term to expose it under generic broadcast/arithmetic.
"""
vbroadcasted(;kwargs...) = (args...)->vbroadcasted(args...; kwargs...)
vbroadcasted(x::NamedColumn{<:Any,<:DataColumn}; meta) = parent(meta.materialized[name(x)])
vbroadcasted(x::NamedColumn; meta) = meta.materialized[name(x)]
vbroadcasted(x::ExprColumn; meta) = Base.broadcasted(getf(x), map(vbroadcasted(;meta), getargs(x))...)
# `protect(expr)` is brms's `I()` literal-escape. In our DSL every call is
# already a first-class `ExprColumn` node, so `protect` just needs to unwrap to
# its inner argument.
vbroadcasted(x::ExprColumn{typeof(protect)}; meta) = vbroadcasted(only(getargs(x)); meta)
# `zscale(x)` / `center(x)` / `standardize(x)` z-transform the inner column at
# VBRMI-materialization time: they materialize the inner broadcast once, apply
# the transform, and pass the resulting plain vector back up to the predictor
# pipeline. Because this fires inside `vbroadcasted`, they compose with every
# downstream consumer (pop predictor, ranefs, link transforms, ...).
vbroadcasted(x::ExprColumn{typeof(center)}; meta) = let raw = Base.materialize(vbroadcasted(only(getargs(x)); meta))
    raw .- _mean(raw)
end
vbroadcasted(x::ExprColumn{typeof(zscale)}; meta) = let raw = Base.materialize(vbroadcasted(only(getargs(x)); meta))
    mu = _mean(raw); sd = _std(raw, mu)
    sd > 0 || error("zscale: zero variance in `$(only(getargs(x)))`")
    (raw .- mu) ./ sd
end
vbroadcasted(x::ExprColumn{typeof(standardize)}; meta) = vbroadcasted(ExprColumn(zscale, getargs(x)...); meta)

_mean(xs) = sum(xs) / length(xs)
_std(xs, mu=_mean(xs)) = sqrt(sum(abs2, xs .- mu) / (length(xs) - 1))

# Literals (Int, Float, etc.) inside formula expressions like `a^2` pass
# through unchanged — they get broadcasted as scalars by Base.broadcasted.
vbroadcasted(x::Number; meta) = x
getinverse(x::ExprColumn{<:Any,<:Tuple{<:Any}}) = inverse(getf(x))
getinverse(x::ExprColumn{<:Any,<:Tuple{<:ExprColumn}}) = inverse(getf(x)) ∘ getinverse(getargs(x, 1)[1]) 
vmeta_sampling(meta, lhs::ExprColumn, rhs) = begin
    meta, o = vmeta_sampling_rhs(meta, rhs; group=:__population__)
    meta, vmaterialize(Base.broadcasted(getinverse(lhs), o))
end
vmeta_sampling(meta, ::NamedColumn{<:Any,MissingColumn}, rhs) = begin
    meta, o = vmeta_sampling_rhs(meta, rhs; group=:__population__)
    meta, vmaterialize(o)
end
vmeta_sampling(meta, lhs::NamedColumn{<:Any,<:DataColumn}, rhs) = begin
    meta, o = vmeta_sampling_rhs(meta, rhs; group=:__population__)
    meta, LikelihoodColumn(parent(parent(lhs)), o)
end
vmeta_sampling(_meta, lhs::JointResponseColumn, _rhs) = error(
    "VBRMI: correlated joint outcomes $(joint_response_names(lhs)) are supported " *
    "by the StanBlocks backend only. Construct `SBBRMI(brmi)`; VBRMI cannot " *
    "silently replace one multivariate density with independent scalar terms.")
"""
    vmeta_sampling_rhs(meta, rhs; group) -> (meta', broadcasted)

vimpl's per-term sampling-RHS walker. Dispatches on the type of `rhs`
([`ExprColumn`](@ref) of `+` / `|` / `&` / custom term, scalar `Int`,
[`NamedColumn`](@ref), …) and returns an updated `meta` plus a
broadcasted expression that feeds the linear predictor. `group` is the
current ranef-grouping symbol (`:__population__` for fixed effects).

**Extension hook**: add a method on this function (typically dispatched
on `::ExprColumn{typeof(your_term)}`) to register a new formula term
with the vimpl backend. The method allocates any [`Part`](@ref)s into
`meta.blocks` via [`push_parts!!`](@ref) and returns the per-row
contribution. See `mo` / `mo1` / `gp` / `offset` in `vimpl.jl` for
worked examples.
"""
vmeta_sampling_rhs(;kwargs...) = (args...)->vmeta_sampling_rhs(args...; kwargs...)
vmeta_sampling_rhs(meta, x::ExprColumn{typeof(+)}; kwargs...) = begin
    meta, args = foldl(getargs(x); init=(meta, ())) do (_meta, _args), _arg
        _meta, _arg = vmeta_sampling_rhs(_meta, _arg; kwargs...)
        _meta, (_args..., _arg)
    end
    meta, Base.broadcasted(+, args...)
end
vmeta_sampling_rhs(meta, x::ExprColumn; kwargs...) = vmeta_sampling_rhs(meta, vbroadcasted(x; meta); kwargs...)
vmeta_sampling_rhs(meta, x::ExprColumn{typeof(*)}; kwargs...) = error("NOT IMPLEMENTED")
vmeta_sampling_rhs(meta, x::ExprColumn{typeof(|)}; kwargs...) = begin
    lhs, rhs = getargs(x, 2)
    vmeta_sampling_rhs(meta, lhs; group=_normalize_group(rhs))
end

_as_named_column(x::NamedColumn) = x
_as_named_column(_) = nothing
_split_sum_terms(x::ExprColumn{typeof(+)}) = getargs(x)
_split_sum_terms(x) = (x,)
_as_real_only_vec(v::AbstractVector{<:Real}) = eltype(v) <: Integer ? nothing : v
_as_real_only_vec(_) = nothing
_as_any_real_vec(v::AbstractVector{<:Real}) = v
_as_any_real_vec(_) = nothing

# `(a + b + ... || g)` -- zero-correlation random effects. Each term on the LHS
# gets its own 1-column block keyed by a synthetic `__nocor__N` suffix, so no
# `chol` sibling is ever allocated between them. Same total direct-parameter
# count as the correlated variant, minus the Cholesky factor parameters.
vmeta_sampling_rhs(meta, x::ExprColumn{typeof(doublepipe)}; kwargs...) = begin
    lhs, rhs = getargs(x, 2)
    isnothing(_as_named_column(rhs)) && error("zerocorr `||`: RHS must be a NamedColumn group, got $(typeof(rhs))")
    terms = _split_sum_terms(lhs)
    meta, args = foldl(enumerate(terms); init=(meta, ())) do (mm, aa), (i, term)
        nocor_key = NamedColumn(Symbol(name(rhs), :__nocor__, i), parent(rhs))
        mm, arg = vmeta_sampling_rhs(mm, term; group=nocor_key)
        mm, (aa..., arg)
    end
    meta, length(args) == 1 ? only(args) : Base.broadcasted(+, args...)
end

# `a & b` -- continuous x continuous interaction (parallels StatsModels.jl).
# Single slot on the elementwise product. Categorical operands (integer /
# CategoricalVector) would need K-1 dummy expansion; errors out for now with
# a pointer to the 2.1 TODO. `&` was chosen over R's `:` because Julia parses
# `:` as lower-precedence than `+`, breaking formula conventions.
vmeta_sampling_rhs(meta, x::ExprColumn{typeof(&)}; group) = begin
    length(getargs(x)) == 2 || error("interaction `&` expects exactly two operands, got $(length(getargs(x)))")
    lhs, rhs = getargs(x, 2)
    _check_cont_interaction(lhs); _check_cont_interaction(rhs)
    lbc = vbroadcasted(lhs; meta); rbc = vbroadcasted(rhs; meta)
    _scale_by_beta(meta, Base.broadcasted(*, lbc, rbc); group)
end
_check_cont_interaction(t) = nothing
_check_cont_interaction(t::NamedColumn) =
    isnothing(_as_real_only_vec(parent(parent(t)))) &&
        error("interaction `&`: only continuous x continuous is supported for now (got `$(name(t))`); see 2.1 TODO for cat expansions")

# `(... | rhs)` -> walker-side group tag. Bare NamedColumn stays as-is; `gr(g)`
# with no kwargs normalizes to the inner NamedColumn (so it coalesces with
# plain `(... | g)` blocks); `gr(g; by=b)` becomes a `GrGroup` so growblock!!
# and _gc_idx can allocate the stratified structure.
_normalize_group(g) = g
_normalize_group(g::NamedColumn) = g
_normalize_group(g::ExprColumn{typeof(gr)}) = begin
    args = getargs(g); kw = getkwargs(g)
    length(args) == 1 || error("gr(...) expects exactly one positional group, got $(length(args))")
    group = args[1]
    isnothing(_as_named_column(group)) && error("gr(...) expects a NamedColumn group, got $(typeof(group))")
    by = get(kw, :by, nothing)
    by === nothing && return group
    isnothing(_as_named_column(by)) && error("gr(...; by=...) expects a NamedColumn for `by`, got $(typeof(by))")
    GrGroup(group, by)
end
vmeta_sampling_rhs(meta, n::Int; group) = begin
    n == 0 && return meta, 0  # `0` suppresses the intercept (no parameter allocated)
    meta, p = growblock!!(meta, group, 1)
    meta, _re_lookup(p, _gc_idx(group))
end
vmeta_sampling_rhs(meta, x::NamedColumn; kwargs...) = vmeta_sampling_rhs(meta, meta.materialized[name(x)]; kwargs...)
vmeta_sampling_rhs(meta, x::DataColumn; kwargs...) = vmeta_sampling_rhs(meta, parent(x); kwargs...)
vmeta_sampling_rhs(meta, x::AbstractVector{<:AbstractFloat}; group) = _scale_by_beta(meta, x; group)
FBroadcasted{F,Style<:Union{Nothing, Base.Broadcast.BroadcastStyle},Axes} = Base.Broadcast.Broadcasted{Style,Axes,F}
vmeta_sampling_rhs(meta, x::FBroadcasted; group) = _scale_by_beta(meta, x; group)
vmeta_sampling_rhs(meta, x::AbstractVector{<:Integer}; group) = begin
    # Treatment-coded categorical predictor: level 1 is the reference (drops out),
    # remaining k-1 levels each get their own coefficient slot in the block.
    K, c_idx = _level_index(x)
    meta, p = growblock!!(meta, group, K - 1)
    meta, _cat_broadcast(p, _gc_idx(group), c_idx)
end
# Population case (gc_idx === nothing): p is (1, k-1), look up the (level-1)-th
# column via linear indexing.
_cat_broadcast(p, ::Nothing, dense) =
    Base.broadcasted(_cat_lookup, Ref(p), dense)
# Grouped case: p is (n_levels, k-1), look up p[gc_idx[i], level - 1].
_cat_broadcast(p, gc_idx::AbstractVector{Int}, dense) =
    Base.broadcasted(_cat_re_lookup, Ref(p), gc_idx, dense)
_cat_lookup(p, level) = level == 1 ? zero(eltype(p)) : p[level - 1]
_cat_re_lookup(p, gc, level) = level == 1 ? zero(eltype(p)) : p[gc, level - 1]

# Group code index: maps each row of the data to its row in the block matrix
# `values`. Returns `nothing` for the special :__population__ marker so that the
# population-level path can pass through `_re_lookup` unchanged.
_gc_idx(::Symbol) = nothing
_gc_idx(group::NamedColumn) = _level_index(parent(parent(group)))[2]

"""
    _level_index(raw) -> (K, c_idx)

Return the number of distinct levels `K` and a dense `c_idx::Vector{Int}`
mapping each row to its 1-based level index. `CategoricalArrays.CategoricalVector`
is handled for free via `levels` + `levelcode`; plain vectors fall back to
building a `Dict` on the fly and warn once per call site.
"""
_level_index(raw::CA.CategoricalVector) = length(CA.levels(raw)), CA.levelcode.(raw)
_level_index(raw::AbstractVector) = begin
    @warn "Building categorical level index from a plain vector; wrap in `categorical(…)` to avoid the per-call Dict." maxlog=1 _id=:vimpl_level_index
    lvls = sort(unique(raw))
    lm = Dict(l => i for (i, l) in enumerate(lvls))
    length(lvls), [lm[l] for l in raw]
end

"""
    push_parts!!(meta, group::Symbol, parts::Part...) -> meta

Append `parts` to `meta.blocks[group]`, creating the block if it didn't exist.
"""
push_parts!!(meta, group::Symbol, new::Part...) =
    rmerge(meta, (; blocks = (; group => (get(meta.blocks, group, ())..., new...))))

"""
    _scale_by_beta(meta, bcast; group) -> (meta, scaled_bcast)

Allocate one normal slot via `growblock!!(meta, group, 1)` and return a
broadcasted expression that multiplies `bcast` by that β. Works at both the
population (scalar β) and group (one β per level, looked up via `_re_lookup`)
levels. Shared backbone of `AbstractVector{<:AbstractFloat}`, `FBroadcasted`,
and `mo(c)` parsers.
"""
_scale_by_beta(meta, bcast; group) = begin
    meta, p = growblock!!(meta, group, 1)
    meta, Base.broadcasted(*, bcast, _re_lookup(p, _gc_idx(group)))
end

# Wrap a (m, 1) growblock view as a length-N broadcasted lookup keyed by
# `gc_idx`. For population groups (`gc_idx === nothing`) the (1, 1) view
# already broadcasts as a scalar against length-N data, so it passes through
# unchanged. Only handles n=1 (single column per growblock!! call) — multi-term
# random specs like `(1 + x | group)` decompose into separate growblock!!
# calls inside the parent `+` foldl, so each call hits this with n=1.
_re_lookup(p, ::Nothing) = p
_re_lookup(p, gc_idx::AbstractVector{Int}) =
    Base.broadcasted(getindex, Ref(p), gc_idx, 1)
vmeta_sampling_rhs(meta, x::FBroadcasted{<:Type{<:Distribution}}; group) = meta, x
vmaterialize(x) = MaterializedColumn(Base.materialize(x), x)

"""
    MaterializedColumn(parent, broadcast)

vimpl's already-materialised storage: `parent` is the live buffer
refreshed each draw, `broadcast` is the lazy expression that fills it.
Used inside `VBRMI.meta.materialized` to thread per-draw recomputation
of named intermediates (`loc ~ 1 + a`, etc.).
"""
struct MaterializedColumn{P,B} <: AbstractColumn
    parent::P
    broadcast::B
end
Base.parent(x::MaterializedColumn) = getfield(x, :parent)
"""
    getbroadcast(x::MaterializedColumn) -> Broadcasted

Access the lazy broadcast expression that refreshes
[`MaterializedColumn`](@ref)`(x)`'s `parent` buffer each draw.
"""
getbroadcast(x::MaterializedColumn) = getfield(x, :broadcast)
Base.broadcastable(x::MaterializedColumn) = Base.broadcastable(parent(x))
# A name bound via `~` (e.g. `loc1 ~ 1 + a`) lands in meta.materialized as a
# MaterializedColumn whose parent is the live buffer refreshed each draw.
# When it reappears on a later `~` RHS, scale the buffer by a fresh beta.
vmeta_sampling_rhs(meta, x::MaterializedColumn; group) = _scale_by_beta(meta, parent(x); group)

n_levels(group::NamedColumn) = length(unique(parent(parent(group))))

_append!(values, m, n) = begin
    append!(values, zeros(m, n))
    view(values, :, size(values, 2)-n+1:size(values, 2))
end

# Append a fresh part with an (m, n) buffer; always valid, no coalescing.
_push_part(parts::Tuple, f, m, n) = let values = ElasticMatrix(zeros(m, n))
    (parts..., Part(f, (; values))), view(values, :, 1:n)
end

# Coalesce-if-same-kind: extend trailing part's ElasticMatrix by `n` cols when
# its func type matches, else append fresh. Dispatches on `(last_part, f)`.
_grow_or_push(parts::Tuple{}, f, m, n) = _push_part(parts, f, m, n)
_grow_or_push(parts::Tuple, f, m, n) = _grow_or_push_tail(parts, last(parts), f, m, n)
_grow_or_push_tail(parts::Tuple, tail::Part{F}, ::F, m, n) where F =
    (parts, _append!(tail.data.values, m, n))
_grow_or_push_tail(parts::Tuple, _, f, m, n) = _push_part(parts, f, m, n)

# Stratified variant: Part.data carries `(values, stratum_idx, n_strata)`.
# Coalescing is valid only when the trailing part uses the same stratification
# (same stratum_idx + n_strata) — i.e. repeated `(… | gr(g, by=b))` with
# identical (g, b). Ls is allocated later by finalize once the final column
# count is known.
_push_part_by(parts::Tuple, f, m, n, stratum_idx, n_strata) = let
    values = ElasticMatrix(zeros(m, n))
    (parts..., Part(f, (; values, stratum_idx, n_strata))), view(values, :, 1:n)
end
_grow_or_push_by(parts::Tuple{}, f, m, n, n_strata, stratum_idx) =
    _push_part_by(parts, f, m, n, stratum_idx, n_strata)
_grow_or_push_by(parts::Tuple, f, m, n, n_strata, stratum_idx) =
    _grow_or_push_tail_by(parts, last(parts), f, m, n, n_strata, stratum_idx)
_grow_or_push_tail_by(parts::Tuple, tail::Part{F}, ::F, m, n, n_strata, stratum_idx) where F =
    (tail.data.n_strata == n_strata && tail.data.stratum_idx == stratum_idx) ?
        (parts, _append!(tail.data.values, m, n)) :
        _push_part_by(parts, f, m, n, stratum_idx, n_strata)
_grow_or_push_tail_by(parts::Tuple, _, f, m, n, n_strata, stratum_idx) =
    _push_part_by(parts, f, m, n, stratum_idx, n_strata)

growblock!!(meta, ::Symbol, n) = begin
    parts = get(meta.blocks, :__population__, ())
    parts, p = _grow_or_push(parts, normal, 1, n)
    rmerge(meta, (; blocks = (; __population__ = parts))), p
end
growblock!!(meta, group::NamedColumn, n) = begin
    key, m = name(group), n_levels(group)
    parts = get(meta.blocks, key, ())
    parts, p = _grow_or_push(parts, grouped_normal, m, n)
    rmerge(meta, (; blocks = (; key => parts))), p
end
growblock!!(meta, group::GrGroup, n) = begin
    # Stratified (gr(g, by=b)) block. Keyed distinctly from plain (_|g) so
    # the two don't coalesce, even if the same `g` shows up in both forms.
    m = n_levels(group.group)
    n_strata = n_levels(group.by)
    stratum_idx = _stratum_idx(group.group, group.by)
    key = Symbol(name(group.group), :__by__, name(group.by))
    parts = get(meta.blocks, key, ())
    parts, p = _grow_or_push_by(parts, grouped_normal_by, m, n, n_strata, stratum_idx)
    rmerge(meta, (; blocks = (; key => parts))), p
end

# For each level of `group`, find the level of `by` it belongs to. Each group
# level must map to exactly one stratum (same stratum on every row of that
# group); otherwise the covariance structure is ill-defined.
_stratum_idx(group::NamedColumn, by::NamedColumn) = begin
    _, g_idx = _level_index(parent(parent(group)))
    _, b_idx = _level_index(parent(parent(by)))
    m_groups = maximum(g_idx)
    mapping = zeros(Int, m_groups)
    for (gi, bi) in zip(g_idx, b_idx)
        if mapping[gi] == 0
            mapping[gi] = bi
        elseif mapping[gi] != bi
            error("gr($(name(group)), by=$(name(by))): group level $gi straddles multiple strata ($(mapping[gi]) vs $bi)")
        end
    end
    mapping
end

# Extend `_re_lookup` / `_gc_idx` to the stratified group. The per-row lookup
# into `values` is still keyed by the primary group column — stratum handling
# lives inside `lprior!(::Part{grouped_normal_by})`, not in the lookup path.
n_levels(g::GrGroup) = n_levels(g.group)
_gc_idx(g::GrGroup) = _level_index(parent(parent(g.group)))[2]

Base.show(io::IO, (;parent, broadcast)::MaterializedColumn) = print(io, eltype(parent), "[...] .= ", broadcast)
Base.show(io::IO, (;parent, rhs)::LikelihoodColumn) = print(io, eltype(parent), "[...] .~ ", rhs)

Base.show(io::IO, p::Part{typeof(normal)}) = print(io, "Part{normal}", size(p.data.values))
Base.show(io::IO, p::Part{typeof(grouped_normal)}) = print(io, "Part{grouped_normal}", size(p.data.values))
Base.show(io::IO, p::Part{typeof(grouped_normal_by)}) = print(io, "Part{grouped_normal_by}", size(p.data.values), " x ", p.data.n_strata, " strata")
Base.show(io::IO, p::Part{typeof(chol)}) = print(io, "Part{chol}(", size(p.data.L, 1), ")")
Base.show(io::IO, p::Part{typeof(chol_by)}) = print(io, "Part{chol_by}(", size(p.data.Ls, 1), " x ", size(p.data.Ls, 3), " strata)")
Base.show(io::IO, p::Part{typeof(simplex)}) = print(io, "Part{simplex}(", length(p.data.values), ")")
Base.show(io::IO, p::Part) = print(io, "Part{", p.func, "}")

Base.show(io::IO, vbrm::VBRMI) = begin
    (; parent, meta) = vbrm
    print(io, parent)
    print(io, "dim: ", LogDensityProblems.dimension(vbrm), "\n")
    print(io, "materialized:\n")
    for (key, value) in pairs(meta.materialized)
        print(io, "  ", key, ": ", value, "\n")
    end
    print(io, "blocks:\n")
    for (key, parts) in pairs(meta.blocks)
        print(io, "  ", key, ":\n")
        for part in parts
            print(io, "    ", part, "\n")
        end
    end
end

LogDensityProblems.dimension(vbrm::VBRMI) = nparams(vbrm.meta.blocks)

"""
    nparams(x) -> Int

Number of unconstrained reals consumed by `x` (a [`Part`](@ref), a
tuple/NamedTuple of parts, or `meta.blocks`). Defines the slice length
each part owns in the flat unconstrained vector. Extension methods
register the dimension of a new marker function's parameter block.
"""
nparams(x::Union{Tuple,NamedTuple}) = sum(nparams, x; init=0)
nparams(p::Part{typeof(normal)}) = size(p.data.values, 2)
nparams(p::Part{typeof(grouped_normal)}) = let (m, n) = size(p.data.values); m * n end
nparams(p::Part{typeof(grouped_normal_by)}) = let (m, n) = size(p.data.values); m * n end
nparams(p::Part{typeof(chol)}) = let n = size(p.data.L, 1); n * (n + 1) ÷ 2 end
nparams(p::Part{typeof(chol_by)}) = let n = size(p.data.Ls, 1), S = size(p.data.Ls, 3); S * n * (n + 1) ÷ 2 end
nparams(p::Part{typeof(simplex)}) = length(p.data.values) - 1

@inline advance!!(x, pos) = x[pos+1], pos+1
@inline advance!!(x, pos, n) = view(x, pos+1:pos+n), pos+n

"""
    lprior!(container, x) -> lp

Recursive shell: each child (block-container, `Part`, …) is handed a view of
exactly `nparams(child)` unconstrained reals; bottoms out on `Part` methods,
each of which writes its constrained buffer and returns its log-prior +
Jacobian contribution.
"""
@inline lprior!(vbrmi::VBRMI, x::AbstractVector) = lprior!(vbrmi.meta.blocks, x)
@inline lprior!(xs::Union{Tuple,NamedTuple}, x::AbstractVector) =
    foldl(xs; init=(0.0, 0)) do (total, pos), child
        xi, pos = advance!!(x, pos, nparams(child))
        total + lprior!(child, xi), pos
    end |> first

@inline lprior!(p::Part{typeof(normal)}, x) = begin
    p.data.values[1, :] .= x
    sum(Base.Fix1(logpdf, Normal()), x)
end

log_abs_tanh(x) = begin
    z = -2*abs(x)
    (log1mexp(z) - log1pexp(z))
end
log_square_tanh(x) = 2 * log_abs_tanh(x)

"""
    lprior!(p::Part{typeof(chol)}, x; eta=1.0) -> lp

Either wrong or better LKJCholesky unconstraining + prior. Writes `p.data.L`
in place from `x` (length n(n+1)/2) and returns the log-prior + Jacobian
contribution. `eta` is the LKJ shape parameter.
"""
@inline lprior!(p::Part{typeof(chol)}, x; eta=1.0) = _chol_lprior!(p.data.L, x; eta)

# Shared LKJ-Cholesky unconstraining body. Writes `L` in place from the first
# n(n+1)/2 entries of `x` and returns the log-prior + Jacobian contribution.
# Used by both `chol` (one shared L) and `chol_by` (per-stratum slice).
@inline _chol_lprior!(L, x; eta=1.0) = begin
    n = LinearAlgebra.checksquare(L)
    pos = 0
    lprior = 0.0
    log_scale, pos = advance!!(x, pos)
    lprior += logpdf(Normal(), log_scale)
    L[1, 1] = exp(log_scale)
    for i in 2:n
        log_scale, pos = advance!!(x, pos)
        lprior += logpdf(Normal(), log_scale)
        xi, pos = advance!!(x, pos)
        tmp = log_abs_tanh(xi / sqrt(n-1))
        L[i, 1] = sign(xi) * exp(log_scale + tmp)
        log_sos = 2 * tmp
        lprior += log1mexp(log_sos)
        for j in 2:i-1
            xi, pos = advance!!(x, pos)
            tmp1 = .5 * log1mexp(log_sos)
            lprior += tmp1
            tmp2 = log_abs_tanh(xi / sqrt(n-j))
            lprior += log1mexp(2*tmp2)
            tmp = tmp1 + tmp2
            L[i, j] = sign(xi) * exp(log_scale + tmp)
            log_sos = logaddexp(log_sos, 2*tmp)
        end
        L[i, i] = exp(log_scale + .5 * log1mexp(log_sos))
        lprior += (n - i + 2*eta-2) * .5 * log1mexp(log_sos)
    end
    lprior
end

# Array-of-Cholesky: one L per stratum. Unconstrains n_strata independent LKJ
# factors, writing each slice `Ls[:, :, s]` in place. Same eta for every
# stratum (brms `gr(g, by=b)` shares the prior hyperparameters).
@inline lprior!(p::Part{typeof(chol_by)}, x; eta=1.0) = begin
    (; Ls) = p.data
    n = size(Ls, 1)
    n_strata = size(Ls, 3)
    n_per = n * (n + 1) ÷ 2
    lprior = 0.0
    pos = 0
    for s in 1:n_strata
        xi, pos = advance!!(x, pos, n_per)
        lprior += _chol_lprior!(view(Ls, :, :, s), xi; eta)
    end
    lprior
end

@inline lprior!(p::Part{typeof(grouped_normal)}, x) = begin
    (; values, L) = p.data
    _, n = size(values)
    lprior = 0.0
    pos = 0
    for vi in eachrow(values)
        xi, pos = advance!!(x, pos, n)
        mul!(vi, LowerTriangular(L), xi)
        lprior += sum(Base.Fix1(logpdf, Normal()), xi)
    end
    lprior
end

# Stratified grouped-normal: per-row pick the stratum's Cholesky slice.
@inline lprior!(p::Part{typeof(grouped_normal_by)}, x) = begin
    (; values, Ls, stratum_idx) = p.data
    _, n = size(values)
    lprior = 0.0
    pos = 0
    for (i, vi) in enumerate(eachrow(values))
        xi, pos = advance!!(x, pos, n)
        s = stratum_idx[i]
        mul!(vi, LowerTriangular(view(Ls, :, :, s)), xi)
        lprior += sum(Base.Fix1(logpdf, Normal()), xi)
    end
    lprior
end

"""
    lprior!(p::Part{typeof(simplex)}, x; alpha=1.0) -> lp

Log-scale stick-breaking with logistic slices: constrain `length(values)-1`
unconstrained reals `x` into the simplex `p.data.values` in place and return
`log|J| + logpdf(Dirichlet(K, α), values)` in one pass. Fully log-scale —
never forms `1 - Σ values_<i` on the linear scale, so it stays stable even
when simplex entries get tiny. `α == 1.0` reduces to the Jacobian-only
transform (plus the constant `loggamma(K)`).

Ported from `blog/posts/simplex/transforms/stickbreakingLogistic.stan`.
"""
@inline lprior!(p::Part{typeof(simplex)}, x; alpha=1.0) = begin
    values = p.data.values
    K = length(values)
    lp = 0.0
    log_cum_prod = 0.0
    for i in 1:K-1
        log_zi = -log1pexp(-(x[i] - log(K - i)))
        log_xi = log_cum_prod + log_zi
        values[i] = exp(log_xi)
        lp += alpha * log_xi
        log_cum_prod += log1mexp(log_zi)
    end
    values[K] = exp(log_cum_prod)
    lp += alpha * log_cum_prod
    lp += loggamma(K * alpha) - K * loggamma(alpha)
    lp
end

@inline llikelihood!((;meta)::VBRMI) = foldl(meta.materialized; init=0.) do llikelihood, m
    llikelihood + llikelihood!(m)
end
@inline llikelihood!(::DataColumn) = 0.
@inline llikelihood!(x::MaterializedColumn) = (Base.materialize!(parent(x), getbroadcast(x)); 0.)
# llikelihood!(x::LikelihoodColumn) = sum(Base.broadcasted(logpdf, rhs(x), parent(x)); init=0.)
# The below is faster for some reason?
@inline llikelihood!(x::LikelihoodColumn) = ssum(Base.broadcasted(logpdf, rhs(x), parent(x)); init=0.)
llikelihood!(x) = error(typeof(x))

ssum(args...; kwargs...) = sum(args...; kwargs...)
ssum(x::Base.Broadcast.Broadcasted; init) = begin
    rv = init
    for xi in x
        rv += xi
    end
    rv
end

Distributions.logpdf(vbrmi::VBRMI, x::AbstractVector) = lprior!(vbrmi, x) + llikelihood!(vbrmi)
LogDensityProblems.logdensity(vbrmi::VBRMI, x::AbstractVector) = lprior!(vbrmi, x) + llikelihood!(vbrmi)

# ==============================================================================
# Template: user-defined terms `mo1(c)` / `mo(c)` — brms-style monotonic effects
# ------------------------------------------------------------------------------
# Every block below is what a user adding a new term would need to touch. Use
# this as the canonical template when sketching additional user-model terms.
#
# 1. Marker function    — pure dispatch tag, never called.
# 2. Parser method      — `vmeta_sampling_rhs(meta, ::ExprColumn{typeof(mo1)};…)`
#                         allocates the backing buffers, pushes Part(s) into
#                         `meta.blocks`, returns a broadcasted expression that
#                         feeds the linear predictor.
# 3. `nparams` method   — how many unconstrained reals this Part consumes.
# 4. `lprior!` method   — refresh derived buffers / return log-prior + Jacobian.
# 5. `Base.show` method — pretty-printed entry in the VBRMI summary.
#
# `mo1` piggy-backs on the library `simplex` Part: the parser pushes a
# `Part(simplex, (; values))` that owns the unit simplex, followed by a
# zero-parameter `Part(mo1, (; values, contrast, c_idx))` that just refreshes
# `contrast = vcat(0, cumsum(values))` each draw.
#
# `mo` is the brms-style free-β variant: it reuses the `mo1` pipeline for the
# monotonic contrast and multiplies by a fresh β slot (one extra normal slot
# allocated via `growblock!!`) — so `mo(c)` ≡ β · mo1(c).
# ==============================================================================

"""Monotonic effect with β fixed to 1 (brms-style mo, β=1); shares a `simplex` sibling."""
function mo1 end

"""Monotonic effect with free β coefficient (brms-style mo); reuses `mo1` + one normal slot."""
function mo end

"""
    _mo_contrast!(meta, raw) -> (meta, broadcasted)

Shared core of `mo1` and `mo`: for a categorical column `raw` with K levels,
emits `(Part(simplex, …), Part(mo1, …))` into the population block and returns
a broadcasted `getindex(contrast, c_idx)` expression.
"""
_mo_contrast!(meta, raw) = begin
    K, c_idx = _level_index(raw)
    K >= 2 || error("mo/mo1: categorical must have ≥ 2 levels (got $K)")
    values   = zeros(K - 1)
    contrast = zeros(K)
    meta = push_parts!!(meta, :__population__,
        Part(simplex, (; values)),
        Part(mo1,     (; values, contrast, c_idx)),
    )
    meta, Base.broadcasted(getindex, Ref(contrast), c_idx)
end

"""
    vmeta_sampling_rhs(meta, x::ExprColumn{typeof(mo1)}; group) -> (meta, broadcasted)

Parse a `mo1(c)` term: brms-style monotonic effect with β fixed to 1.
Population-level only for now.
"""
vmeta_sampling_rhs(meta, x::ExprColumn{typeof(mo1)}; group) = begin
    group === :__population__ || error("mo1: only population-level supported for now (got group=$group)")
    _mo_contrast!(meta, vbroadcasted(getargs(x, 1)[1]; meta))
end

"""
    vmeta_sampling_rhs(meta, x::ExprColumn{typeof(mo)}; group) -> (meta, broadcasted)

Parse a `mo(c)` term: brms-style monotonic effect with free β. Delegates to
`_mo_contrast!` for the monotonic contrast and multiplies by a fresh β slot
allocated via `growblock!!`. Population-level only for now.
"""
vmeta_sampling_rhs(meta, x::ExprColumn{typeof(mo)}; group) = begin
    group === :__population__ || error("mo: only population-level supported for now (got group=$group)")
    meta, contrast_bcast = _mo_contrast!(meta, vbroadcasted(getargs(x, 1)[1]; meta))
    _scale_by_beta(meta, contrast_bcast; group)
end

nparams(p::Part{typeof(mo1)}) = 0

"""
    lprior!(p::Part{typeof(mo1)}, _) -> 0.0

Zero-parameter follow-up to a `simplex` sibling: refresh
`contrast = vcat(0, cumsum(values))` in place so downstream broadcasts see
up-to-date contrast values. Contributes nothing to the log-prior.
"""
@inline lprior!(p::Part{typeof(mo1)}, _) = begin
    (; values, contrast) = p.data
    contrast[1] = 0.0
    for k in 2:length(contrast)
        contrast[k] = contrast[k-1] + values[k-1]
    end
    0.0
end

Base.show(io::IO, p::Part{typeof(mo1)}) = print(io, "Part{mo1}(K=", length(p.data.contrast), ")")

# ==============================================================================
# End of `mo1` / `mo` template
# ==============================================================================

# ==============================================================================
# User-term: `gp(x; k=K, c=C)` -- 1D Hilbert-space approx GP (squared-exp kernel)
# ------------------------------------------------------------------------------
# Standard Riutort-Mayol et al. (2020) parameterization. Julia-side precompute:
#   x_c       = x .- mean(x)         (center)
#   L         = c * maximum(abs, x_c)
#   lambda[k] = (k * pi / (2L))^2    for k = 1..K   (eigenvalues)
#   PHI[i,k]  = (1/sqrt(L)) * sin(sqrt(lambda[k]) * (x_c[i] + L))
#
# Per-draw state (inside `lprior!`):
#   log_rho, log_sigma, beta_1..beta_K   <- K + 2 unconstrained params
#   rho, sigma           = exp(log_rho), exp(log_sigma)
#   sqrt_spd[k]          = sigma * (2*pi)^(1/4) * sqrt(rho) * exp(-0.25 * rho^2 * lambda[k])
#   contrib              = PHI * (sqrt_spd .* beta)        # length-n vector
#
# Priors: log_rho, log_sigma ~ Normal(0, 1) directly on the unconstrained scale
# (=> rho, sigma ~ LogNormal(0, 1), no Jacobian needed since we parameterize on
# log scale); beta_k ~ Normal(0, 1). Scope: population-level, data-x only.
# ==============================================================================

"""
    vmeta_sampling_rhs(meta, x::ExprColumn{typeof(offset)}; group) -> (meta, broadcasted)

Parse an `offset(x)` term: fixed-slope (no beta) contribution to the linear
predictor. brms-compatible — adds `x` directly to the predictor without
allocating any parameters. Population-level only.
"""
vmeta_sampling_rhs(meta, x::ExprColumn{typeof(offset)}; group) = begin
    group === :__population__ || error("offset: only population-level supported (got group=$group)")
    length(getargs(x)) == 1 || error("offset: expects exactly one positional argument, got $(length(getargs(x)))")
    meta, vbroadcasted(getargs(x, 1)[1]; meta)
end

"""Hilbert-space approx GP marker; population-level, 1D squared-exp kernel."""
function hsgp end

"""
    _hsgp_basis(raw, K, c) -> (PHI, lambda)

Precompute the Riutort-Mayol eigen-basis from raw x values. `PHI` is (n, K),
`lambda` is length-K. Center-then-scale-by-boundary-factor-c is standard; the
fit and the basis are the shared `_brm_fit_hsgp` / `_brm_apply_hsgp` law.
"""
_hsgp_basis(raw::AbstractVector{<:Real}, K::Integer, c::Real) =
    _brm_apply_hsgp(_brm_fit_hsgp(raw, K, c), raw, K)

"""
    vmeta_sampling_rhs(meta, x::ExprColumn{typeof(gp)}; group) -> (meta, broadcasted)

Parse a `gp(x; k=K, c=C)` term. Precomputes `PHI` + `lambda` from raw data,
pushes a `Part(hsgp, …)` into the population block, and returns a broadcast
wrapper over the per-draw `contrib` buffer (refreshed in `lprior!`).
"""
vmeta_sampling_rhs(meta, x::ExprColumn{typeof(gp)}; group) = begin
    group === :__population__ || error("gp: only population-level supported for now (got group=$group)")
    args = getargs(x); kw = getkwargs(x)
    length(args) == 1 || error("gp: expects exactly one positional argument, got $(length(args))")
    inner = args[1]
    isnothing(_as_named_column(inner)) && error("gp: expects a NamedColumn argument, got $(typeof(inner))")
    raw_in = parent(parent(inner))
    raw = _as_any_real_vec(raw_in)
    isnothing(raw) && error("gp: `$(name(inner))` must be a real-valued vector (got $(typeof(raw_in)))")
    K = get(kw, :k, 20)
    c = get(kw, :c, 1.5)
    PHI, lambda = _hsgp_basis(raw, K, c)
    sqrt_spd = zeros(K)
    beta = zeros(K)
    contrib = zeros(length(raw))
    meta = push_parts!!(meta, :__population__,
        Part(hsgp, (; PHI, lambda, sqrt_spd, beta, contrib)),
    )
    meta, Base.broadcasted(identity, contrib)
end

nparams(p::Part{typeof(hsgp)}) = length(p.data.lambda) + 2

"""
    lprior!(p::Part{typeof(hsgp)}, x) -> lp

Refresh `sqrt_spd` + `contrib` in place from the `K+2` unconstrained reals
(ordering: log_rho, log_sigma, beta_1..beta_K). Returns Normal(0,1) log-priors
on all three component groups (no Jacobian because we parameterize on the
log-scale directly).
"""
@inline lprior!(p::Part{typeof(hsgp)}, x) = begin
    (; PHI, lambda, sqrt_spd, beta, contrib) = p.data
    K = length(lambda)
    pos = 0
    log_rho,   pos = advance!!(x, pos)
    log_sigma, pos = advance!!(x, pos)
    beta_x,    pos = advance!!(x, pos, K)
    beta .= beta_x
    rho   = exp(log_rho)
    sigma = exp(log_sigma)
    # sqrt(S(sqrt(lambda[k]); rho, sigma)) for the 1D squared-exp spectral density.
    c_base = sigma * (2 * pi)^(1/4) * sqrt(rho)
    for k in 1:K
        sqrt_spd[k] = c_base * exp(-0.25 * rho^2 * lambda[k])
    end
    mul!(contrib, PHI, sqrt_spd .* beta)
    lp = logpdf(Normal(), log_rho) + logpdf(Normal(), log_sigma)
    lp += sum(Base.Fix1(logpdf, Normal()), beta_x)
    lp
end

Base.show(io::IO, p::Part{typeof(hsgp)}) = print(io, "Part{hsgp}(K=", length(p.data.lambda), ", n=", length(p.data.contrib), ")")
