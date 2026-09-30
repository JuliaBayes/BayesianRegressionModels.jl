# ---------------------------------------------------------------------------
# Julianic NativePPL — a "run-the-body" authoring surface.
#
# Goal (user, 2026-08-04): the `@model` body should read as valid SSA Julia that
# actually computes the log density, with `~` the ONLY magic — and `~` reducing
# to "prep + logpdf": constrain the unconstrained coordinate (+ log-Jacobian),
# then accumulate the distribution's log density.
#
# This is the additive Option-1 path (decision `0tuhucl`, resolved "use your
# judgment"): a new `@jmodel` entry point that lowers `~` and RUNS the body as
# the primal density `θ -> Real`, verified model-by-model against the existing
# declarative executor as a differential oracle. Once the gallery reaches
# parity, `@model` itself flips onto this lowering.
#
# Design
# ------
# * Only `~` / `.~` statements are rewritten; every other statement is kept
#   VERBATIM — that is exactly "valid Julia except ~".
# * `sym ~ dist` lowers to ONE operation — prep the value, accumulate its
#   log-density contribution — with the value coming from the conditioned data
#   when `sym` was passed to `jcondition` and from a fresh slice of theta when it
#   was not. The dot lifts that operation elementwise and nothing more, so
#   `@. y ~ D(a, b)` and `y .~ D.(a, b)` are the same statement (Julia's own
#   `Base.Broadcast.__dot__` performs the normalization).
# * The body runs twice against a context `ctx`:
#     - TRACE (compile): discover the total unconstrained dimension by counting
#       the coordinates each latent `~` consumes. Returns in-support placeholder
#       values so deterministic code and observations run without error.
#     - PRIMAL (density): a cursor walks `θ`; each latent `~` consumes its slice,
#       preps it (constrain + logjac), and accumulates prior logpdf; each
#       observation accumulates its data logpdf. The scalar `ctx.buffers.acc` is the
#       primal `θ -> Real` that Enzyme differentiates (same AD path the
#       declarative executor already uses — DifferentiationInterface + AutoEnzyme
#       straight through the primal kernel).
#
# Latent-vs-observation is decided by CONDITIONING, never by spelling. That is
# what data-vs-parameter means, it is what the architecture contract specifies
# ("free-coordinate allocation after composition and conditioning"), and it is
# what the declarative graph layer already does (`AbstractSiteActivity` in
# `native_ppl_authoring.jl`, assigned `hasproperty(conditions, name)`).
#
# An earlier version of this file decided it syntactically — broadcast meant
# observation, scalar meant latent — and needed two fail-closed errors to police
# the disagreement with the conditioned data. Both are gone: the check that used
# to throw now selects a branch instead, and it still folds to a compile-time
# constant, so the primal pays nothing for it either way.
#
# Supported latent site families, each one `_sample!` method pair (trace +
# primal) — the surface grows by ADDING methods, never by touching the lowering:
#   * scalar `Normal` (real support, identity) and `Exponential` (positive, exp);
#   * real-support multivariate (`MvNormal`, `product_distribution` of Normals) —
#     one contiguous block, identity transform;
#   * positive-support multivariate (`product_distribution` of `Exponential`s) —
#     per-coordinate exp transform, for a vector of grouped scales;
#   * `LKJCholesky` — the constrained correlation manifold, K(K-1)/2 raw coords.
# Observations take any `Distributions` likelihood in broadcast form. A
# multivariate family with NO method of its own hits the real-support probe and
# is REFUSED rather than scored against the wrong measure.
#
# Gradients run under the plain `DI.AutoEnzyme()` the declarative executor uses:
# the conditioned data is threaded through the body as a separate `DI.Constant`
# argument rather than stored on the differentiated context, so STATIC activity
# analysis suffices and no `set_runtime_activity` workaround is needed.
# ---------------------------------------------------------------------------

# NOTE: `Normal`/`Exponential`/`StandardNormal` are ALSO names inside NativePPL
# (prior/declaration constructors), so we must NOT `using Distributions`-import
# them here — fully qualify the Distributions types/functions instead. In the
# macro-generated model body, `Normal`/`logpdf` deliberately resolve in the
# CALLER's scope (the user's `using Distributions`), keeping the body real Julia.
import Distributions
# `weighted`'s typed weight-vector methods dispatch on the StatsBase weight
# kinds, and `rand` on a wrapper distribution needs the RNG abstract type.
import StatsBase
import Random

# 0-alloc buffer-threading substrate (user directive 2026-08-04): every
# array-valued intermediate in the run-the-body primal is filled IN PLACE into a
# preallocated buffer via MutatingFunctions' `apply!!`, so the differentiated
# kernel allocates nothing per gradient eval — the same discipline the declarative
# FactorWorkspace uses (preallocated `FactorBuffers` threaded as a DI.Cache).
using MutatingFunctions: apply!!
# Destination SHAPES for those in-place fills, declared rather than discovered by a
# pilot run: `broadcast_size` for broadcast nodes (function-first, so it is its own
# entry point) and `output_size(getindex, …)` for gathers.
using OutputSignatures: broadcast_eltype, broadcast_size, output_size

# --- Activity-analysis types ----------------------------------------------
#
# The macro records a static per-statement SKELETON (pure syntax: what each
# statement reads and writes, and whether it is a `~`). At `jprepare` — after
# conditioning is known — `_julianic_activity` runs ONE forward + ONE backward
# sweep over that skeleton to partition every statement into data /
# needed-for-sampling / generated-quantity, with NO values (the SSA statement
# order IS a topological sort, so no graph need be built). The partition is a
# function of (skeleton, conditioning), so it can move a site between `:sampled`
# and `:gq` when the same model is conditioned differently.

# One statement's syntactic facts, built at macro time. `write` is the name it
# binds — a `~` binds its site name; `Symbol("")` marks a bare expression that
# binds nothing. `reads` are the free names of its right-hand side, OVER-collected:
# a name with no model binding (`exp`, `Normal`) simply resolves to no dependency
# edge, so precision is unnecessary and only false-negatives (a missed read) would
# be unsound.
struct JulianicStmt
    write::Symbol
    reads::Vector{Symbol}
    is_sample::Bool
    broadcast::Bool
end

# One statement's assigned role, for `jactivity` introspection:
#   :observation — a conditioned `~` (a likelihood term; always in the density)
#   :sampled     — an unconditioned `~` reaching the likelihood (consumes a
#                  coordinate, prior scored in the density)
#   :gq          — an unconditioned `~`, or a parameter-dependent deterministic,
#                  that does NOT reach the likelihood (drawn / computed only in
#                  postprocessing; consumes NO coordinate)
#   :density     — a parameter-dependent deterministic on the likelihood path
#   :data        — depends on no parameter (constant across θ; computable once)
struct JulianicActivity
    write::Symbol
    role::Symbol
    is_sample::Bool
end

# The role sets a body's names fall into, each carried as a TYPE PARAMETER so its
# per-name predicate folds to a literal at the statement's call site — exactly the
# trick `_julianic_is_conditioned` plays on the data type. Three sets drive the
# lowering:
#   G  — generated-quantity LATENT sites (drawn only in postprocessing; no coordinate)
#   GD — generated-quantity DETERMINISTIC writes (a parameter-dependent deterministic
#        that reaches no likelihood term — skipped in the density, computed only where
#        a downstream gq draw needs it, i.e. postprocessing)
#   D  — data-only writes (constant across θ — precomputed once at `jprepare`, fetched
#        in the density rather than recomputed)
# An EMPTY set makes its branch dead code: a model with no gq latent, no gq
# deterministic and no precomputed data (the entire pre-AA gallery) lowers
# IDENTICALLY to the pre-AA surface — so both the 0-alloc primal and the oracle
# parity are preserved by construction. The pruning is Julia's own constant-branch
# folding at `run`'s specialization on this type, NOT LLVM dead-code elimination of
# an emitted-but-unused store: the losing branch is gone before LLVM sees the body.
struct JulianicRoles{G,GD,D} end
@inline _julianic_is_gq(::JulianicRoles{G,GD,D}, ::Val{name}) where {G,GD,D,name} = name in G
@inline _julianic_is_gq_det(::JulianicRoles{G,GD,D}, ::Val{name}) where {G,GD,D,name} = name in GD
@inline _julianic_is_data(::JulianicRoles{G,GD,D}, ::Val{name}) where {G,GD,D,name} = name in D

# --- Buffer pool ----------------------------------------------------------
#
# The pool holds one reusable flat `Vector{T}` per array-valued intermediate the
# body produces, walked by a cursor in body-execution order. It is threaded through
# the primal kernel as a `DI.Cache` (Enzyme shadows it; the in-place fills are
# differentiated through), exactly mirroring the executor's `FactorBuffers`.
#
# Storage is deliberately FLAT and concretely typed (`Vector{Vector{T}}`): a pool of
# heterogeneously-shaped arrays would be an abstract container and cost a dynamic
# dispatch per access. An N-D intermediate (`tau .* (L*z)` is K×ngroups) is served by
# `reshape`ing its flat buffer to the traced shape — a view, so still no allocation,
# and MutatingFunctions' `apply!!(cache::AbstractArray, broadcast, …)` fills any-rank
# destinations in place (snag `apply-0alloc-ppl-fb5b821f`, landed `0134a6c`).
#
# Sizing: a TRACE pass runs the body once with in-support placeholders and records
# each intermediate's SHAPE (`buffer_sizes::Vector{Dims}`); `jworkspace` then
# preallocates the pool at the workspace eltype. `_next_buffer!` grows the pool on
# demand for the unprepared path (`jlogdensity` without a workspace), so a
# preallocated pool never grows inside a differentiated call.
# It also carries the primal's own mutable state — the θ cursor and the density
# accumulator. That state is preallocated with the pool rather than constructed per
# call for the same reason the executor keeps everything in `FactorWorkspace`: a
# per-call mutable context ESCAPES into the body closure, so Julia must heap-allocate
# it, and one such box per gradient eval is exactly the allocation this whole piece
# exists to remove. `JulianicPrimal` is therefore an immutable view over this struct.
mutable struct JulianicBuffers{T}
    vectors::Vector{Vector{T}}
    vcursor::Int
    cursor::Int         # position in θ
    acc::T              # the accumulated log density
end
JulianicBuffers{T}() where {T} =
    JulianicBuffers{T}(Vector{Vector{T}}(), 0, 0, zero(T))
function JulianicBuffers{T}(sizes::AbstractVector) where {T}
    JulianicBuffers{T}([zeros(T, prod(s)) for s in sizes], 0, 0, zero(T))
end

# The next flat buffer, grown to `n` elements if the pool was not presized.
@inline function _next_flat!(pool::JulianicBuffers{T}, n::Int) where {T}
    pool.vcursor += 1
    if pool.vcursor > length(pool.vectors)
        push!(pool.vectors, zeros(T, n))    # grow lazily (unprepared eval only)
    end
    buffer = @inbounds pool.vectors[pool.vcursor]
    length(buffer) == n || resize!(buffer, n)
    return buffer
end

# The next buffer, shaped to `shape`. A 1-D shape returns the flat vector itself;
# an N-D shape returns a `reshape` view of it (no allocation either way).
@inline _next_buffer!(pool::JulianicBuffers, shape::Tuple{Int}) =
    _next_flat!(pool, @inbounds shape[1])
@inline _next_buffer!(pool::JulianicBuffers, shape::Dims) =
    reshape(_next_flat!(pool, prod(shape)), shape)

# --- Contexts -------------------------------------------------------------

# PRIMAL: walks θ with a cursor and accumulates the log density. Both the cursor
# and the accumulator live on the preallocated `buffers` (see `JulianicBuffers`),
# so this context is IMMUTABLE — two fields, both references — and costs nothing to
# construct per call. A mutable context here would be heap-boxed on every gradient
# eval, because it escapes into the body closure.
#
# The observation DATA is deliberately NOT a field here: it is a run-time
# constant, and mixing a constant array with the active `theta`/`acc` inside one
# struct is exactly what forced Enzyme's `set_runtime_activity`. Instead the body's
# `run` closure takes the data as a SEPARATE argument (passed through DI as a
# `Constant`), so this struct holds only active state and STATIC activity
# suffices — no runtime-activity workaround.
# A context that WALKS θ: it constrains each site's coordinates and returns the
# natural-space value, so every deterministic in the body computes for real.
# `JulianicPrimal` accumulates the log density while doing it; the postprocessing
# contexts (below) do the identical walk and record instead. Sharing the abstract
# type is what makes postprocessing structurally unable to disagree with the
# density — there is one walk, not a walk and a reimplementation of it.
abstract type JulianicRun end

# The contexts that WALK θ. The per-family `_sample!` table is keyed on this, not
# on `JulianicRun`, because reading a coordinate and applying its constraining
# transform is exactly what the families are FOR — and `JulianicSimulate` (below)
# runs the same body with no position at all, so it must not inherit them.
abstract type JulianicWalk <: JulianicRun end

struct JulianicPrimal{TH<:AbstractVector,B} <: JulianicWalk
    theta::TH
    buffers::B          # JulianicBuffers{eltype(theta)} — the preallocated pool
end

# TRACE: runs the body once to size the unconstrained coordinate vector AND record
# each array-valued intermediate's SHAPE (`buffer_sizes`), in body order, so the
# primal pool can be preallocated to match.
mutable struct JulianicTrace
    dim::Int
    buffer_sizes::Vector{Dims}
    captured::Vector{Pair{Symbol,Any}}   # data-only writes, computed once here
end
JulianicTrace() = JulianicTrace(0, Dims[], Pair{Symbol,Any}[])

# POSTPROCESSING: the same θ walk as the primal, recording instead of scoring.
# It reuses `JulianicBuffers` verbatim — including the cursor and the (unread)
# accumulator — so every `_sample!` method above works here with no second
# implementation. That is the whole point: a postprocessing pass that re-derived
# what a site means could disagree with the density, and this one cannot.
#
# `sites` collects `name => constrained value` in body order, and `observations`
# collects whatever the observation hook produced per observed site. No pooled
# buffer discipline: postprocessing is not on the gradient hot path, and the
# 0-alloc constraint is deliberately scoped to logdensity + gradient (user,
# 2026-08-04). Values are COPIED out, because a `_sample!` returning a `@view`
# into θ (the multivariate path) would otherwise alias a caller's position — and
# a pooled observation buffer is overwritten by the next site.
struct JulianicRecord{TH<:AbstractVector,B,K,R<:Random.AbstractRNG} <: JulianicWalk
    theta::TH
    buffers::B
    kernel::K           # what an observation site computes (logpdf / rand / …)
    rng::R              # for GQ latents: they have no coordinate, so they are DRAWN
    sites::Vector{Pair{Symbol,Any}}
    observations::Vector{Pair{Symbol,Any}}
end
JulianicRecord(theta::AbstractVector, buffers, kernel, rng::Random.AbstractRNG) =
    JulianicRecord(theta, buffers, kernel, rng,
                   Pair{Symbol,Any}[], Pair{Symbol,Any}[])

# SIMULATE (the prior / prior-predictive pass): the same body again, but with no
# position to walk — each latent `~` DRAWS from its own prior instead of reading
# θ, and each observation resamples. It is the one pass that does not share the
# `_sample!` family table, and it needs almost none of it: a prior draw is
# `rand(dist)` for every family at once, precisely BECAUSE it never touches the
# unconstrained parametrization. Only `LKJCholesky` needs an adapter, since its
# `rand` yields a `Cholesky` where the body wants the plain factor matrix.
struct JulianicSimulate{R<:Random.AbstractRNG,B,K} <: JulianicRun
    rng::R
    buffers::B
    kernel::K
    sites::Vector{Pair{Symbol,Any}}
    observations::Vector{Pair{Symbol,Any}}
end
JulianicSimulate(rng::Random.AbstractRNG, buffers, kernel) =
    JulianicSimulate(rng, buffers, kernel,
                     Pair{Symbol,Any}[], Pair{Symbol,Any}[])

@inline _sample!(ctx::JulianicSimulate, ::Val, dist) = rand(ctx.rng, dist)
@inline _sample!(ctx::JulianicSimulate, ::Val, dist::Distributions.LKJCholesky) =
    Matrix(rand(ctx.rng, dist).L)

# "Records what each site produced" is a SECOND axis, crossing the walk/simulate
# split: `JulianicRecord` walks θ and records, `JulianicSimulate` draws and
# records. Single inheritance cannot express both, so this axis is a Union — it
# dispatches exactly like an abstract type and stays more specific than
# `JulianicRun`, which is what lets the recording `_obs_*` hooks override the
# pooled ones.
const JulianicRecording = Union{JulianicRecord,JulianicSimulate}

# The recording hook on the latent path. Inlined to the identity for every
# context that is not recording, so the primal keeps its 0-alloc property.
@inline _record_site!(_ctx, ::Val, value) = value
@inline function _record_site!(ctx::JulianicRecording, ::Val{name}, value) where {name}
    push!(ctx.sites, name => _julianic_recorded(value))
    return value
end
# `copy` on anything that is a view of, or a pooled buffer over, live state.
@inline _julianic_recorded(value) = value
@inline _julianic_recorded(value::AbstractArray) = copy(value)

# Does THIS context need a generated-quantity DETERMINISTIC evaluated? Only the
# recording passes do — a gq deterministic reaches no likelihood term, so nothing
# on the density path (primal, trace) ever reads it; but a gq LATENT drawn in
# postprocessing may, so `JulianicRecord`/`JulianicSimulate` compute it. The
# density and trace get a `nothing` placeholder for it instead, and because this
# predicate folds to a compile-time constant, that placeholder branch is the ONLY
# code generated there — the deterministic's work (and its pooled buffer slot)
# is never emitted, not emitted-then-eliminated.
@inline _julianic_computes_det(_ctx) = false
@inline _julianic_computes_det(::JulianicRecording) = true

# Data-only precompute. A `:data` write is constant across θ, so it is computed
# EXACTLY ONCE — during the trace pass, which already runs the body once at
# `jprepare` — and every later pass FETCHES it from a `store` NamedTuple threaded
# as `run`'s 4th argument (a Constant on `prepared`, exactly like the observation
# data). Only the trace "captures"; the density, gradient and postprocessing fetch.
# Because a `:data` write is fetched (never `_cached_broadcast!`-ed) in the density
# AND computed by a plain broadcast (never pooled) in the trace, NEITHER pass gives
# it a pooled buffer slot — so the trace's slot list and the primal's cursor stay
# in lockstep with the `:data` array simply absent from both. The fetched array is
# part of the `prepared` Constant, so the gradient never shadows or differentiates
# it: the density does no work on it at all.
@inline _julianic_captures_data(_ctx) = false
@inline _julianic_captures_data(::JulianicTrace) = true
@inline _julianic_record_data!(ctx::JulianicTrace, ::Val{name}, value) where {name} =
    (push!(ctx.captured, name => value); value)
@inline _julianic_fetch_data(store, ::Val{name}) where {name} = getfield(store, name)

# --- `~` runtime: prep + logpdf ------------------------------------------
#
# Each latent method advances the cursor by the site's unconstrained dimension,
# applies the support's transform (prep), accumulates prior logpdf + logjac, and
# returns the CONSTRAINED value (so downstream body code sees the natural-space
# quantity). Trace methods only grow `dim` and return an in-support placeholder.

# Is `name` bound to conditioned data? Both arguments are compile-time
# constants (the site name rides a `Val`, the data is a `NamedTuple` whose field
# names are in its type), so this folds to a literal `true`/`false` and the
# guarded branch disappears from the primal.
@inline _julianic_is_conditioned(::NamedTuple{names}, ::Val{name}) where {names,name} =
    name in names
@inline _julianic_is_conditioned(_data, ::Val) = false

# A latent `~`: prep the site's coordinate(s), accumulate its density, return the
# natural-space value. `data` is NOT consulted for a kind check — conditioning
# selects between this and the observation path at the call site
# (`_julianic_lower_sample`), as a compile-time constant, so a conditioned name
# never reaches here. It stays in the signature because the scalar and broadcast
# forms share one call shape.
@inline _sample_site!(ctx, _data, name::Val, dist) =
    _record_site!(ctx, name, _sample!(ctx, name, dist))

# Elementwise latent: `b .~ Normal.(0, tau)` is k INDEPENDENT scalar sites by
# ordinary broadcasting — no `product_distribution`, no special multivariate form.
# `dists` arrives LAZY (a `Broadcasted`, never materialized), so the k
# distributions cost no allocation.
#
# The k sites consume k CONTIGUOUS coordinates, so the walk takes the whole slice
# ONCE and constrains + scores it with FUSED broadcasts into pooled buffers —
# never a per-element loop. That is not a micro-optimization: a scalar loop that
# writes a Cache-nested pool buffer element-by-element BOXES under Enzyme reverse
# (~34 B / coordinate, measured), while the identical fused `.=` the observation
# path uses is 0-alloc. Taking the slice up front also makes evaluation ORDER
# irrelevant — coordinate `i` pairs with marginal `i` by index, not by iteration
# order — which is what the per-element cursor walk needed the loop to guarantee.
@inline _julianic_instantiate(dists) = dists
@inline _julianic_instantiate(dists::Base.Broadcast.Broadcasted) =
    Base.Broadcast.instantiate(dists)

# Constrain a whole coordinate slice `u` into `dest` for one marginal FAMILY,
# returning the TOTAL log-Jacobian. Fused counterparts of the scalar `_sample!`
# transforms; the family is read off a representative element so the two passes
# agree. Real support → identity (0 Jacobian); positive support → exp (Jacobian
# `u`). A family without a method here is refused at TRACE time (prepare), the
# mirror of the multivariate `_julianic_require_real_support` guard.
@inline function _broadcast_constrain!(::Distributions.Normal, dest, u)
    dest .= u
    return zero(eltype(dest))
end
@inline function _broadcast_constrain!(::Distributions.Exponential, dest, u)
    dest .= exp.(u)
    return sum(u)
end
@noinline _broadcast_constrain!(dist, _dest, _u) = throw(ArgumentError(
    "julianic @jmodel: an elementwise latent `b .~ D.(...)` supports `Normal` " *
    "and `Exponential` marginals (real and positive support); got `" *
    string(nameof(typeof(dist))) * "`. A constrained-manifold marginal needs " *
    "its own `_broadcast_constrain!` method, as the scalar/LKJ sites do."))

# WALK (primal + record): slice, constrain (fused), score (fused, pooled), one
# accumulate. Both pooled buffers are presized by the matching TRACE method below,
# in the same push order.
@inline function _sample_broadcast!(ctx::JulianicWalk, name::Val, dists)
    lazy = _julianic_instantiate(dists)
    n = length(lazy)
    lo = ctx.buffers.cursor + 1
    ctx.buffers.cursor += n
    u = @view ctx.theta[lo:ctx.buffers.cursor]
    b = _site_buffer!(ctx, n)
    logjac = _broadcast_constrain!(lazy[1], b, u)
    lp = _site_buffer!(ctx, n)
    lp .= Distributions.logpdf.(lazy, b)
    ctx.buffers.acc += sum(lp) + logjac
    return _record_site!(ctx, name, b)
end
# TRACE: count k coordinates, record the two buffer sizes in the SAME order the
# walk pushes them, and hand back an in-support placeholder so downstream body
# code runs. The placeholder slice is zeros, constrained the same way (`0` for
# real support, `exp(0)=1` for positive) so it is always in the marginal's domain.
@inline function _sample_broadcast!(ctx::JulianicTrace, name::Val, dists)
    lazy = _julianic_instantiate(dists)
    n = length(lazy)
    ctx.dim += n
    b = _site_buffer!(ctx, n)
    _broadcast_constrain!(lazy[1], b, zeros(eltype(b), n))
    _site_buffer!(ctx, n)                       # the logpdf scratch's size
    return b
end
# SIMULATE: draw each marginal from its prior (no position to walk). Not on the
# gradient path, so the plain loop is fine.
@inline function _sample_broadcast!(ctx::JulianicSimulate, name::Val, dists)
    lazy = _julianic_instantiate(dists)
    n = length(lazy)
    b = _site_buffer!(ctx, n)
    for i in 1:n
        b[i] = rand(ctx.rng, lazy[i])
    end
    return _record_site!(ctx, name, b)
end

# --- Generated-quantity latents (a `~` that reaches no likelihood term) ----
#
# The activity analysis routes an unconstrained-latent `~` here when its site
# does NOT reach the likelihood (a new-group random effect with no data, say).
# It consumes NO unconstrained coordinate and contributes nothing to the density;
# it exists only to be DRAWN in postprocessing (the julianic counterpart of a
# Stan `generated quantities` draw). The branch is compile-time-dead for every
# model with no gq latent, which is the entire current gallery.

# DENSITY / TRACE: return a cheap in-support placeholder, consume no θ, score
# nothing. Nothing on the likelihood path reads a gq value — by construction of
# the partition — so the placeholder is never scored, and for a scalar site it is
# a literal the primal's dead-code elimination removes. (`0.0`/`1.0` are the same
# in-support values the TRACE `_sample!` methods hand back.)
@inline _julianic_gq_placeholder(::Distributions.Normal) = 0.0
@inline _julianic_gq_placeholder(::Distributions.Exponential) = 1.0
@inline _julianic_gq_placeholder(dist::Distributions.MultivariateDistribution) =
    zeros(length(dist))
@inline _julianic_gq_placeholder(
        dist::Distributions.Product{Distributions.Continuous,<:Distributions.Exponential}) =
    ones(length(dist))
@inline function _julianic_gq_placeholder(dist::Distributions.LKJCholesky)
    K = dist.d
    L = zeros(K, K)
    for i in 1:K
        L[i, i] = 1.0
    end
    return L
end

# POSTPROCESSING: a gq latent has no posterior coordinate, so it is drawn from its
# prior (conditioned on the other latents' current values — which the record walk
# has already read from θ) and recorded, using the recording context's RNG.
@inline _julianic_gq_draw(ctx, dist) = rand(ctx.rng, dist)
@inline _julianic_gq_draw(ctx, dist::Distributions.LKJCholesky) =
    Matrix(rand(ctx.rng, dist).L)

@inline _sample_gq!(::JulianicPrimal, ::Val, dist) = _julianic_gq_placeholder(dist)
@inline _sample_gq!(::JulianicTrace, ::Val, dist) = _julianic_gq_placeholder(dist)
@inline _sample_gq!(ctx::JulianicRecording, name::Val, dist) =
    _record_site!(ctx, name, _julianic_gq_draw(ctx, dist))

# Elementwise gq latent (`b .~ Normal.(0, tau)` with `b` a gq site): the same, per
# marginal. DENSITY/TRACE hand back an in-support placeholder vector (allocating,
# but dead in the primal — a gq broadcast site is unread on the likelihood path);
# POSTPROCESSING draws each marginal. Consumes no θ and pushes no pooled buffer.
@inline function _sample_broadcast_gq!(ctx::Union{JulianicPrimal,JulianicTrace},
                                       name::Val, dists)
    lazy = _julianic_instantiate(dists)
    return _julianic_gq_placeholder(lazy[1]) isa Number ?
        fill(_julianic_gq_placeholder(lazy[1]), length(lazy)) :
        _julianic_gq_placeholder(lazy[1])
end
@inline function _sample_broadcast_gq!(ctx::JulianicRecording, name::Val, dists)
    lazy = _julianic_instantiate(dists)
    n = length(lazy)
    b = _site_buffer!(ctx, n)
    for i in 1:n
        b[i] = rand(ctx.rng, lazy[i])
    end
    return _record_site!(ctx, name, b)
end

# Normal: real support, identity transform (log-Jacobian 0). Covers `Normal()`
# (standard normal) and `Normal(mu, sigma)` with latent-dependent parameters.
# NOTE: no `@inbounds` on the cursor read. The trace and the primal must agree
# on how many coordinates each site consumes; if they ever drift, an elided
# bounds check turns that into an out-of-bounds READ (silent garbage under
# Enzyme) instead of a `BoundsError`. The check is free next to a `logpdf`.
@inline function _sample!(ctx::JulianicWalk, ::Val, dist::Distributions.Normal)
    ctx.buffers.cursor += 1
    v = ctx.theta[ctx.buffers.cursor]
    ctx.buffers.acc += Distributions.logpdf(dist, v)
    return v
end
@inline _sample!(ctx::JulianicTrace, ::Val, ::Distributions.Normal) = (ctx.dim += 1; 0.0)

# Exponential: positive support, exp transform. constrain u -> exp(u); the
# log-Jacobian of that transform is exactly u, so accumulate logpdf + u.
@inline function _sample!(ctx::JulianicWalk, ::Val, dist::Distributions.Exponential)
    ctx.buffers.cursor += 1
    u = ctx.theta[ctx.buffers.cursor]
    v = exp(u)
    ctx.buffers.acc += Distributions.logpdf(dist, v) + u
    return v
end
@inline _sample!(ctx::JulianicTrace, ::Val, ::Distributions.Exponential) = (ctx.dim += 1; 1.0)

# Multivariate real-support latent (e.g. MvNormal, `product_distribution` of
# real-support univariates): identity transform on every coordinate (support is
# all of R^k, so the log-Jacobian is 0). Consumes `length(dist)` contiguous
# coordinates and returns the value vector for the body to index. This covers
# grouped/blocked latents authored as `b ~ product_distribution(fill(D, k))`
# or `b ~ MvNormal(...)`, then referenced as `b[group]`.
@inline function _sample!(ctx::JulianicWalk, ::Val, dist::Distributions.MultivariateDistribution)
    k = length(dist)
    lo = ctx.buffers.cursor + 1
    ctx.buffers.cursor += k
    v = @view ctx.theta[lo:ctx.buffers.cursor]   # view, not a copy — avoids an allocation
    ctx.buffers.acc += Distributions.logpdf(dist, v)
    return v
end
@inline function _sample!(ctx::JulianicTrace, name::Val, dist::Distributions.MultivariateDistribution)
    k = length(dist)
    _julianic_require_real_support(name, dist, k)
    ctx.dim += k
    return zeros(k)
end

# The `MultivariateDistribution` primal method above dispatches on the whole
# abstract type but implements only the REAL-support case (identity transform,
# zero log-Jacobian). A constrained multivariate — `Dirichlet`, `MvLogNormal` —
# would silently get the wrong measure: no constraining transform and no
# log-Jacobian, exactly the asymmetry the scalar `Exponential` method exists to
# avoid. Probe the support once, in TRACE mode (so the primal pays nothing), and
# refuse rather than mis-score. Constrained families that DO have their own
# `_sample!` (positive vectors, LKJ, below) never reach this method at all — a
# more specific signature wins dispatch — so the guard only ever fires on a
# family nobody has implemented yet.
@noinline function _julianic_require_real_support(::Val{name}, dist, k) where {name}
    Distributions.insupport(dist, fill(-1.0, k)) && return nothing
    throw(ArgumentError(
        "julianic @jmodel: multivariate site `" * String(name) * "` has a " *
        "distribution of type " * string(nameof(typeof(dist))) * " whose " *
        "support is not all of R^" * string(k) * ". The run-the-body `~` " *
        "applies an identity transform with zero log-Jacobian to multivariate " *
        "sites, which would score the wrong measure. Constrained-manifold " *
        "sites (LKJ, simplex, positive) need their own `_sample!` method."))
end

# Multivariate POSITIVE-support latent: a `product_distribution` of `Exponential`
# marginals — used for a vector of grouped scales `tau ~
# product_distribution(fill(Exponential(1), K))`. Positive support → exp
# transform per coordinate (constrain u -> exp(u), log-Jacobian u), so accumulate
# per-marginal logpdf + sum(u). More specific than the real-support method above,
# so it wins for the `Product{…,Exponential}` type (a product of Normals resolves
# to `DiagNormal <: MvNormal`, which correctly stays on the identity path).
@inline function _sample!(ctx::JulianicWalk, ::Val,
        dist::Distributions.Product{Distributions.Continuous,<:Distributions.Exponential})
    k = length(dist)
    lo = ctx.buffers.cursor + 1
    ctx.buffers.cursor += k
    u = @view ctx.theta[lo:ctx.buffers.cursor]   # view, not a copy
    v = exp.(u)
    ctx.buffers.acc += sum(Distributions.logpdf.(dist.v, v)) + sum(u)
    return v
end
@inline function _sample!(ctx::JulianicTrace, ::Val,
        dist::Distributions.Product{Distributions.Continuous,<:Distributions.Exponential})
    k = length(dist)
    ctx.dim += k
    return ones(k)
end

# LKJ-Cholesky correlation latent: `L ~ LKJCholesky(K, eta)` returns the K×K
# lower-triangular Cholesky factor of a correlation matrix, so the body can write
# the non-centered effect `b = diag(tau) * (L * z)` in ordinary Julia. The `~`
# magic reduces to prep + logpdf exactly as everywhere else — it just prepares a
# constrained-manifold coordinate:
#   * prep: consume K(K-1)/2 unconstrained coords, build L via the tanh/sech
#     parametrization (row-normalized Cholesky-of-correlation).
#   * logpdf (fused with the manifold log-Jacobian): the LKJ density in raw
#     coords, `sum_j (log_normalizer_j + alpha_j * logsech2(raw))`.
# This reuses the executor's exact leaf math (`_factor_logsech2` / `_factor_sech`
# / the `LKJCholeskySiteFactor` normalizer formula), so it matches bit-for-bit —
# but with RUNTIME K (no `Val{K}` / `@generated`) so the primal kernel stays
# type-stable when K comes from the distribution object's `d` field.

# LKJ log-density in raw coords (mirrors `_factor_lkj_logdensity`, runtime K).
@inline function _julianic_lkj_logdensity(K::Int, eta, coords::AbstractVector{T}) where {T}
    density = zero(T)
    offset = 0
    for column in 1:(K - 1)
        alpha = eta + (K - column - 1) / 2
        logc = BRM.loggamma(alpha + 0.5) - BRM.loggamma(alpha) - 0.5 * log(BRM.pi)
        for _row in (column + 1):K
            raw = coords[begin + offset]
            density += logc + alpha * _factor_logsech2(raw)
            offset += 1
        end
    end
    return density
end

# Build the K×K lower-triangular correlation Cholesky factor from the raw coords
# (mirrors the implicit construction in `_factor_grouped_affine_value`): column s
# of row c is `(prod_{t<s} sech(raw_ct)) * tanh(raw_cs)`, diagonal is the residual
# `prod_{t<c} sech(raw_ct)`. Coord addressing matches `_factor_correlation_raw`.
@inline function _julianic_corr_cholesky(K::Int, coords::AbstractVector{T}) where {T}
    L = zeros(T, K, K)
    @inbounds L[1, 1] = one(T)
    @inbounds for c in 2:K
        residual = one(T)
        for s in 1:(c - 1)
            raw = coords[(s - 1) * K - (s - 1) * s ÷ 2 + c - s]
            L[c, s] = residual * tanh(raw)
            residual = residual * _factor_sech(raw)
        end
        L[c, c] = residual
    end
    return L
end

@inline function _sample!(ctx::JulianicWalk, ::Val, dist::Distributions.LKJCholesky)
    K = dist.d
    m = K * (K - 1) ÷ 2
    lo = ctx.buffers.cursor + 1
    ctx.buffers.cursor += m
    coords = @view ctx.theta[lo:lo + m - 1]   # view, not a copy
    ctx.buffers.acc += _julianic_lkj_logdensity(K, dist.η, coords)
    return _julianic_corr_cholesky(K, coords)
end
@inline function _sample!(ctx::JulianicTrace, name::Val, dist::Distributions.LKJCholesky)
    K = dist.d
    _julianic_require_lower_uplo(name, dist)
    ctx.dim += K * (K - 1) ÷ 2
    L = zeros(Float64, K, K)                # in-support placeholder (L = I)
    for i in 1:K
        L[i, i] = 1.0
    end
    return L
end

# `LKJCholesky` carries an `uplo` field, and `LKJCholesky(K, eta, :U)` promises
# the UPPER factor — but `_julianic_corr_cholesky` always builds the LOWER one,
# and hands the body a bare `Matrix` that cannot carry the distinction. The body
# would then compute `L * z` against a silently transposed factor. The density is
# `uplo`-invariant (it is a function of the raw coords), so this can never
# surface as a wrong number at the `~` itself — only as wrong DOWNSTREAM math,
# which is the hardest kind to notice. Checked once in TRACE mode, like the
# real-support probe.
@noinline function _julianic_require_lower_uplo(::Val{name}, dist) where {name}
    dist.uplo === 'L' && return nothing
    throw(ArgumentError(
        "julianic @jmodel: LKJ site `" * String(name) * "` was declared with " *
        "uplo=" * repr(dist.uplo) * ", but the run-the-body `~` returns the " *
        "LOWER-triangular correlation factor as a plain matrix — the body " *
        "would silently use a transposed factor. Declare it as " *
        "`LKJCholesky(" * string(dist.d) * ", " * string(dist.η) * ")` " *
        "(uplo='L') and transpose in the body if you need the upper factor."))
end

# Observation log density. The `~` magic (not the user) supplies `logpdf`, so we
# route through this runtime helper — the model body needs no `logpdf` in scope,
# only the distribution constructor the user actually wrote. Broadcast-safe.
@inline _obs_logpdf(dist, x) = Distributions.logpdf(dist, x)

# --- Observation weights --------------------------------------------------
#
# The declarative surface spells a weighted observation
# `y ~ weighted(Normal(mu, sigma), fweights(n))`, and the weight KIND — carried
# by the StatsBase constructor's NAME — selects the semantics at lowering time:
# `:analytic` scales precision (`Normal(mu, sigma/sqrt(w))`), `:frequency` and
# `:power` MULTIPLY the log density by `w`, `:unit` is a no-op. That taxonomy is
# a FORMULA-surface device: a declaration cannot say `sigma/sqrt(w)`, so the
# weight type has to say it for the author.
#
# A julianic body is ordinary Julia, so it says it directly:
#
#   analytic:  `@. y ~ Normal(mu, sigma / sqrt(w))`     — already expressible
#   unit:      `@. y ~ Normal(mu, sigma)`               — just omit the weight
#
# What plain Julia has no spelling for is the MULTIPLY: `logpdf` returns a
# number, and there is no distribution whose density is another's raised to a
# power. That is the one real gap, and this closes it — as a runtime wrapper
# distribution, per decision `06le2au`. `weighted` is BRM's existing formula
# marker (`src/macro.jl`, declared with no methods); this gives it an executable
# method, so ONE name means the same thing on both surfaces.
#
# The result is a genuine `Distributions.Distribution` — the point of julianic
# is that the body is real Julia, so `logpdf(weighted(d, 2.0), y)` must work
# outside the macro too. It is deliberately NOT normalized: `w * logpdf` is a
# pseudo-likelihood for any `w != 1`, exactly as the declarative `:frequency` /
# `:power` kinds are.
struct WeightedDistribution{VF<:Distributions.VariateForm,
                            VS<:Distributions.ValueSupport,
                            D<:Distributions.Distribution{VF,VS},
                            W<:Real} <: Distributions.Distribution{VF,VS}
    distribution::D
    weight::W
end

Distributions.params(d::WeightedDistribution) =
    (Distributions.params(d.distribution)..., d.weight)

# The whole contract, both variate forms. `logpdf` on the base distribution is
# whatever the author's distribution defines, so this composes with every family
# — including the `truncated`/`censored` response evidence.
@inline Distributions.logpdf(d::WeightedDistribution{Distributions.Univariate},
                             x::Real) =
    d.weight * Distributions.logpdf(d.distribution, x)
@inline Distributions.logpdf(d::WeightedDistribution{<:Union{Distributions.Multivariate,
                                                             Distributions.Matrixvariate}},
                             x::AbstractArray) =
    d.weight * Distributions.logpdf(d.distribution, x)

# Sampling is UNWEIGHTED on purpose, matching the declarative backend: a
# frequency/power weight scales the log-density contribution but leaves the base
# distribution's predictive RNG intact (`src/macro.jl`'s `weighted` docstring).
Base.rand(rng::Random.AbstractRNG, d::WeightedDistribution) =
    rand(rng, d.distribution)
Base.length(d::WeightedDistribution) = length(d.distribution)
Distributions.insupport(d::WeightedDistribution, x) =
    Distributions.insupport(d.distribution, x)

"""
    weighted(distribution, weight::Real)

A distribution whose log density is `weight * logpdf(distribution, x)` — the
executable form of BRM's observation-weight marker, for a julianic `@jmodel`
body:

    @. y ~ weighted(Poisson(exp(eta)), n)

This is the `:frequency` / `:power` semantics (the weight MULTIPLIES the log
density). The other two kinds need no wrapper, because a julianic body writes
them directly: `:analytic` weights scale precision — `@. y ~ Normal(mu, sigma /
sqrt(w))` — and `:unit` weights are the unweighted model.

`weight * logpdf` is a pseudo-likelihood, not a normalized density, for any
`weight != 1`. `rand` is deliberately UNWEIGHTED, so the predictive RNG stays
the base distribution's.

!!! note
    Passing a `StatsBase` weight VECTOR (`weighted(d, fweights(n))`) applies
    elementwise and is checked: `aweights` is rejected with the precision
    spelling above, because broadcasting it here would silently multiply. Under
    `@.` the vector is broadcast to plain numbers BEFORE this function sees it,
    so that check cannot fire — write the weight kind you mean.
"""
BRM.weighted(distribution::Distributions.Distribution, weight::Real) =
    WeightedDistribution(distribution, weight)

# The un-dotted vector spellings, kept identical to the declarative surface so a
# model ports across unchanged. Each maps to the kind's real semantics, and the
# two that this wrapper CANNOT express fail closed rather than silently
# multiplying.
BRM.weighted(distribution::Distributions.Distribution,
             weight::StatsBase.AbstractWeights) =
    WeightedDistribution.(distribution, weight)
@noinline function BRM.weighted(::Distributions.Distribution,
                                ::StatsBase.AnalyticWeights)
    throw(ArgumentError(
        "julianic `weighted`: analytic weights scale PRECISION, not the log " *
        "density, so they are not a `weighted(...)` wrapper here — write them " *
        "directly, e.g. `@. y ~ Normal(mu, sigma / sqrt(w))`. Use " *
        "`weighted(dist, w)` only for frequency/power weights, which multiply " *
        "the log density."))
end
@noinline function BRM.weighted(::Distributions.Distribution,
                                ::StatsBase.ProbabilityWeights)
    throw(ArgumentError(
        "julianic `weighted`: ProbabilityWeights semantics are not implemented " *
        "(the declarative surface rejects them too)."))
end

# Accumulate an already-computed observation log-density contribution.
@inline _accumulate!(ctx::JulianicRun, contribution) =
    (ctx.buffers.acc += contribution; nothing)
@inline _accumulate!(::JulianicTrace, _contribution) = nothing

# Fetch conditioned data for an observation site. The data is passed to the body
# as a separate argument (not stored in the differentiated context — see
# `JulianicPrimal`), so this reads straight from that constant `data` value.
# Available in both modes, because tracing happens after conditioning.
#
# No fail-closed check here any more: this is only reached from the branch that
# `_julianic_is_conditioned` already selected, so the name is conditioned by
# construction. The two errors that used to police the latent/observation split
# are gone with the split itself.
@inline _observation(data, ::Val{sitename}) where {sitename} =
    getproperty(data, sitename)

# --- Cached array ops (the 0-alloc rewrite targets) -----------------------
#
# The macro rewrites each array-valued intermediate in the body into one of these,
# decomposing a fused broadcast per operation node. PRIMAL fills a pooled buffer in
# place via `apply!!`; TRACE runs the op allocating (once, at prepare) and records
# the result SHAPE so the pool can be presized. Both passes execute the same lowered
# body, so the push order (trace) and the walk order (primal) stay in lockstep.
#
# Shapes come from OutputSignatures — `broadcast_size` for a broadcast node (it is
# function-first, so it has its own entry point rather than the `x`-first
# `output_size`) and `output_size(getindex, …)` for a gather — which is exactly what
# that package is for: the destination shape from immediately-available information,
# no pilot run (snag `broadcast-gather-3d8a0849`, landed `121de31`).
#
# ONLY floating-point intermediates may go through the pool. The pool is a flat
# `Vector{T}` at the workspace/AD eltype, so anything else routed through it comes
# back RETYPED: an index vector `[2, 4]` returns as `[2.0, 4.0]` and is no longer a
# valid index, and a `Bool` mask returns as `[1.0, 0.0, …]` and silently stops being
# a mask. Both are ordinary things to compute in a model body (`selector[group]`,
# a mask sliced out of data), and the failure is invisible to `jprepare` — the TRACE
# pass returns the real `getindex`/`broadcast` result, so only the PRIMAL breaks, at
# the first density evaluation.
#
# The predicate must give the SAME answer in both passes or the trace's recorded
# shapes and the primal's cursor walk desynchronise. It does: it reads only element
# TYPES, and the two passes differ solely in float precision (`Float64` vs an Enzyme
# dual) — both `<: AbstractFloat`, while an integer or boolean intermediate is
# neither, in either pass.
@inline _julianic_poolable(::Type{E}) where {E} = E <: AbstractFloat

# `broadcast` a single operator/function `f` over already-materialized args:
@inline function _cached_broadcast!(ctx::JulianicRun, f::F, args...) where {F}
    _julianic_poolable(broadcast_eltype(f, args...)) || return broadcast(f, args...)
    buffer = _next_buffer!(ctx.buffers, broadcast_size(args...))
    apply!!(buffer, broadcast, f, args...)
end
@inline function _cached_broadcast!(ctx::JulianicTrace, f::F, args...) where {F}
    _julianic_poolable(broadcast_eltype(f, args...)) || return broadcast(f, args...)
    push!(ctx.buffer_sizes, broadcast_size(args...))
    broadcast(f, args...)
end

# `getindex` gather `A[idx]` (a whole-vector index, e.g. `b[group]`):
@inline function _cached_gather!(ctx::JulianicRun, A, idx)
    _julianic_poolable(eltype(A)) || return getindex(A, idx)
    buffer = _next_buffer!(ctx.buffers, output_size(getindex, size(A), idx))
    apply!!(buffer, getindex, A, idx)
end
@inline function _cached_gather!(ctx::JulianicTrace, A, idx)
    _julianic_poolable(eltype(A)) || return getindex(A, idx)
    push!(ctx.buffer_sizes, output_size(getindex, size(A), idx))
    getindex(A, idx)
end

# A length-n destination buffer for a whole-vector site — serves BOTH a broadcast
# observation (the `@. buf = logpdf(...)` fill) and an elementwise latent (the
# `b .~ Normal.(0, tau)` per-coordinate walk), because both want the same thing: a
# right-sized destination presized from the trace. PRIMAL hands back a pooled
# buffer (so the term is 0-alloc — the distribution vector stays lazy, never
# materialized); TRACE records the length and hands back a throwaway scratch so the
# same fill/walk is harmless.
@inline _site_buffer!(ctx::JulianicRun, n::Int) = _next_buffer!(ctx.buffers, (n,))
@inline function _site_buffer!(ctx::JulianicTrace, n::Int)
    push!(ctx.buffer_sizes, (n,))
    Vector{Float64}(undef, n)           # one-shot trace scratch
end
@inline _obs_end!(ctx::JulianicPrimal, ::Val, buf) = (ctx.buffers.acc += sum(buf); nothing)
@inline _obs_end!(::JulianicTrace, ::Val, _buf) = nothing
@inline function _obs_end!(ctx::JulianicRecording, ::Val{name}, buf) where {name}
    push!(ctx.observations, name => identity.(buf))   # narrows `Any` to the real eltype
    return nothing
end

# WHAT an observation site computes, chosen by the context rather than baked into
# the lowered body. The body's fused fill calls this function elementwise over
# `(distribution, datum)`, so one body serves the density, the pointwise
# log-likelihood and the posterior predictive with no second lowering — the
# observation distribution is built by the author's own code either way.
@inline _obs_kernel(_ctx) = _obs_logpdf
@inline _obs_kernel(ctx::JulianicRecording) = ctx.kernel

# Posterior-predictive kernel: ignores the observed datum — redrawing it is the
# point — and carries its own RNG so a predictive pass is reproducible.
struct JulianicPredictKernel{R<:Random.AbstractRNG}
    rng::R
end
@inline (kernel::JulianicPredictKernel)(dist, _x) = rand(kernel.rng, dist)

# Postprocessing collects into a FRESH `Any` buffer rather than the pooled one:
# the pool is typed at θ's eltype, which would silently coerce a discrete
# predictive draw (a `Poisson` count, a `Bernoulli` outcome) to `Float64`.
# `_obs_end!` narrows it back to the concrete eltype the kernel actually
# produced. Nothing here is on the gradient path, so the allocation is free of
# consequence — the 0-alloc constraint is scoped to logdensity + gradient.
@inline _site_buffer!(::JulianicRecording, n::Int) = Vector{Any}(undef, n)

# --- Model objects --------------------------------------------------------

# A julianic model: the lowered body (`run(ctx, data, roles)`, closing over the
# model's input arguments), the input argument names (introspection/error text),
# and the static AA skeleton (one `JulianicStmt` per body statement).
struct JulianicModel{R}
    run::R
    input_names::Tuple{Vararg{Symbol}}
    statements::Vector{JulianicStmt}
end

# After conditioning: the run closure + the conditioned observation data + the
# skeleton (carried through so `jprepare` can run the analysis on it).
struct JulianicConditioned{R,D}
    run::R
    data::D
    statements::Vector{JulianicStmt}
end

# After the trace: adds the discovered unconstrained dimension, the ordered
# array-valued intermediate buffer lengths (so a primal pool can be preallocated
# — see `JulianicBuffers`), the compile-time-constant `roles` carrier threaded
# into `run`, and the per-statement `activity` partition (for `jactivity`).
struct JulianicPrepared{R,D,RL,S}
    run::R
    data::D
    dimension::Int
    buffer_sizes::Vector{Dims}
    roles::RL
    activity::Vector{JulianicActivity}
    store::S            # data-only writes, precomputed once by the trace pass
end

"""
    jcondition(model::JulianicModel; observations...)

Bind observed data to a julianic model, yielding a `JulianicConditioned`.
"""
jcondition(model::JulianicModel; observations...) =
    JulianicConditioned(model.run, NamedTuple(observations), model.statements)

# The two-pass activity analysis (§ Activity-analysis types). Runs over the static
# skeleton with the conditioned names — no values — so it can run at `jprepare`,
# after conditioning, before any position exists.
#
#   * FORWARD (parameter taint): a statement produces a parameter-dependent value
#     if it is an unconditioned `~` (introduces a latent) or reads a name whose
#     defining statement is parameter-tainted. A conditioned `~` produces the
#     observed DATA value (not parameter-tainted).
#   * BACKWARD (affects-likelihood): seed every conditioned `~` (a likelihood
#     term); a statement affects the likelihood if a downstream affecting
#     statement reads what it writes. SSA + top-to-bottom means a write is only
#     read later, so a single reverse sweep is exact.
#   * DISTRIBUTE: unconditioned `~` → :sampled if it affects the likelihood else
#     :gq; deterministic → :density if parameter ∧ affecting, :gq if parameter ∧
#     not, :data otherwise; conditioned `~` → :observation.
#
# Returns the per-statement activity (introspection) and the gq LATENT site names
# (threaded as `JulianicRoles`). O(V·E) in the reverse sweep, which is nothing for
# the handful of statements a model body has.
function _julianic_activity(statements::Vector{JulianicStmt}, condnames)
    n = length(statements)
    defindex = Dict{Symbol,Int}()
    for i in 1:n
        w = statements[i].write
        w === Symbol("") || (defindex[w] = i)
    end
    is_cond(name) = name in condnames
    is_obs = Bool[s.is_sample && is_cond(s.write) for s in statements]
    is_latent = Bool[s.is_sample && !is_cond(s.write) for s in statements]

    is_param = falses(n)
    for i in 1:n
        if is_latent[i]
            is_param[i] = true
        elseif is_obs[i]
            is_param[i] = false
        else
            is_param[i] = any(statements[i].reads) do r
                j = get(defindex, r, 0)
                j != 0 && is_param[j]
            end
        end
    end

    affects = Bool[is_obs[i] for i in 1:n]
    # With no likelihood term at all (a prior-only model), the sampling target IS
    # the joint prior, so every latent is part of it. Seed the backward sweep from
    # the latent sites instead of the (empty) observation set — otherwise no latent
    # would "affect the likelihood", every one would collapse to a generated
    # quantity, and the model would prepare with dimension 0 (unsamplable). A
    # generated quantity is meaningful only relative to a likelihood; with none,
    # there is nothing to generate a quantity against.
    any(is_obs) || (affects .= is_latent)
    for i in n:-1:1
        (affects[i] || statements[i].write === Symbol("")) && continue
        w = statements[i].write
        for j in (i + 1):n
            if affects[j] && (w in statements[j].reads)
                affects[i] = true
                break
            end
        end
    end

    activity = Vector{JulianicActivity}(undef, n)
    gq = Symbol[]        # gq LATENT sites (drawn in postprocessing; consume no coordinate)
    gq_det = Symbol[]    # gq DETERMINISTIC writes (skipped in the density)
    data = Symbol[]      # data-only writes (precomputed once at prepare)
    named(w) = w !== Symbol("")   # a bare expression writes nothing to gate on
    for i in 1:n
        s = statements[i]
        role = if is_obs[i]
            :observation
        elseif is_latent[i]
            affects[i] ? :sampled : (push!(gq, s.write); :gq)
        elseif !is_param[i]
            named(s.write) && push!(data, s.write)
            :data
        elseif affects[i]
            :density
        else
            named(s.write) && push!(gq_det, s.write)
            :gq
        end
        activity[i] = JulianicActivity(s.write, role, s.is_sample)
    end
    return activity, Tuple(gq), Tuple(gq_det), Tuple(data)
end

"""
    jprepare(conditioned::JulianicConditioned)

Run the activity analysis (partition the body into data / needed-for-sampling /
generated-quantity), then run the body once in TRACE mode to discover the
unconstrained dimension — which now counts ONLY the sampled latents, so a
generated-quantity latent consumes no coordinate.
"""
function jprepare(conditioned::JulianicConditioned)
    activity, gq_sites, gq_det, data_names = _julianic_activity(
        conditioned.statements, propertynames(conditioned.data))
    roles = JulianicRoles{gq_sites,gq_det,data_names}()
    # The trace pass DOUBLES as the capture pass: it runs the body once here, sizing
    # the buffer pool AND computing every `:data` write into `trace.captured`. No
    # store exists yet, so pass an empty one — the trace captures, it never fetches.
    trace = JulianicTrace()
    conditioned.run(trace, conditioned.data, roles, NamedTuple())
    store = NamedTuple(trace.captured)
    return JulianicPrepared(
        conditioned.run, conditioned.data, trace.dim, trace.buffer_sizes,
        roles, activity, store)
end

"""
    jactivity(prepared) -> Vector{JulianicActivity}

The activity partition assigned to each body statement at `jprepare`: per
statement, the name it binds and its role (`:observation`, `:sampled`, `:gq`,
`:density`, `:data`). The partition is a function of the model AND the
conditioning, so re-conditioning the same model can move a site between
`:sampled` and `:gq`.
"""
jactivity(prepared::JulianicPrepared) = prepared.activity

dimension(prepared::JulianicPrepared) = prepared.dimension

# The primal kernel `θ -> Real`. Top-level (not a closure over θ) so the AD
# backend sees `prepared` as a constant, `theta` as the sole active input, and the
# preallocated `buffers` pool as scratch (a DI.Cache in the gradient path). The
# body fills those buffers in place, so no per-eval allocation.
function _julianic_kernel(theta::AbstractVector{T}, prepared::JulianicPrepared,
                          buffers::JulianicBuffers{T}) where {T}
    buffers.vcursor = 0                 # rewind the pool + the primal's own state
    buffers.cursor = 0
    buffers.acc = zero(T)
    prepared.run(JulianicPrimal(theta, buffers), prepared.data, prepared.roles, prepared.store)
    return buffers.acc
end

@noinline _julianic_dimension_error(prepared::JulianicPrepared, theta) =
    throw(DimensionMismatch(
        "julianic @jmodel: model has " * string(prepared.dimension) *
        " unconstrained coordinate(s) but the position has length " *
        string(length(theta))))

@inline _julianic_check_dimension(prepared::JulianicPrepared, theta::AbstractVector) =
    length(theta) == prepared.dimension || _julianic_dimension_error(prepared, theta)

"""
    jlogdensity(prepared::JulianicPrepared, theta::AbstractVector)

Evaluate the log density by running the body as the primal.
"""
function jlogdensity(prepared::JulianicPrepared, theta::AbstractVector)
    _julianic_check_dimension(prepared, theta)
    buffers = JulianicBuffers{eltype(theta)}(prepared.buffer_sizes)
    return _julianic_kernel(theta, prepared, buffers)
end

# Gradient plumbing. DifferentiationInterface lives in a weak-dep extension, so
# these are declared here and given methods in
# ext/BayesianRegressionModelsDifferentiationInterfaceExt.jl.
function _julianic_prepare_gradient end
function _julianic_value_and_gradient! end

struct JulianicWorkspace{P,G,B}
    di_preparation::P
    gradient::G
    buffers::B          # the preallocated pool, threaded as a DI.Cache
end

"""
    jworkspace(prepared, T, backend)

Allocate a gradient workspace (AD preparation + gradient buffer + the preallocated
0-alloc buffer pool) for a julianic prepared model at element type `T`.

`backend` is the plain `DI.AutoEnzyme()` the declarative executor uses — the
conditioned data reaches the body as a separate `DI.Constant` argument instead
of riding on the differentiated context, so static activity analysis resolves
the primal and no `set_runtime_activity` mode is required.
"""
function jworkspace(prepared::JulianicPrepared, ::Type{T}, backend) where {T<:AbstractFloat}
    position = zeros(T, prepared.dimension)
    buffers = JulianicBuffers{T}(prepared.buffer_sizes)
    di_preparation = _julianic_prepare_gradient(prepared, backend, position, buffers)
    JulianicWorkspace(di_preparation, zeros(T, prepared.dimension), buffers)
end

"""
    jlogdensity!(workspace, prepared, theta)

Log density evaluated against the workspace's preallocated pool — the 0-alloc
primal, and the exact counterpart of the declarative executor's
`logdensity!(workspace, …)`. `jlogdensity(prepared, theta)` computes the same
number but allocates a pool per call, so it is the convenience entry point, not
the one to put in a sampler's inner loop.
"""
function jlogdensity!(
    workspace::JulianicWorkspace, prepared::JulianicPrepared, theta::AbstractVector)
    _julianic_check_dimension(prepared, theta)
    return _julianic_kernel(theta, prepared, workspace.buffers)
end

"""
    jlogdensity_and_gradient!(workspace, prepared, theta) -> (density, gradient)

Value + gradient of the log density via Enzyme straight through the run-the-body
primal, with all array intermediates filled in place into the preallocated pool.
"""
function jlogdensity_and_gradient!(
    workspace::JulianicWorkspace, prepared::JulianicPrepared, theta::AbstractVector)
    _julianic_check_dimension(prepared, theta)
    return _julianic_value_and_gradient!(
        prepared, workspace.di_preparation, workspace.gradient, theta,
        workspace.buffers)
end

# --- Postprocessing ---------------------------------------------------------
#
# The same body, the same θ walk, a different sink. Everything a consumer asks of
# a fitted model — what are the parameters in natural space, how well is each
# observation predicted, what would new data look like — is answered by RUNNING
# the model rather than by a parallel description of it. The declarative surface
# needs `output`/`outputs`/`simulate` machinery precisely because its executor
# discarded the body; julianic still has the body.
# `rng` is used only to DRAW gq latents (sites the analysis found unreachable from
# the likelihood); a model with none never touches it, so it defaults.
function _julianic_record(prepared::JulianicPrepared, theta::AbstractVector, kernel,
                          rng::Random.AbstractRNG=Random.default_rng())
    _julianic_check_dimension(prepared, theta)
    # A GROWABLE pool, not one presized to `prepared.buffer_sizes` — the density's
    # pool. Postprocessing's set of pooled intermediates genuinely DIFFERS from the
    # density's: it COMPUTES the generated-quantity deterministics the density skips
    # (a downstream gq draw may read them), so a presized pool would be sized for the
    # density's slots and the gq-det broadcast would land on the wrong one. Sizing on
    # demand is exactly right here and costs nothing that matters — postprocessing is
    # off the 0-alloc hot path (that is scoped to logdensity + gradient).
    buffers = JulianicBuffers{eltype(theta)}()
    context = JulianicRecord(theta, buffers, kernel, rng)
    prepared.run(context, prepared.data, prepared.roles, prepared.store)
    return context
end

"""
    jconstrained(prepared, theta) -> NamedTuple

Every latent site's value in NATURAL (constrained) space, at the unconstrained
position `theta`, keyed by site name in body order.

This is the same constraining transform the density applies — one walk, not a
reimplementation — so a value here can never disagree with the log density that
scored it. A generated-quantity latent (one the analysis found unreachable from
the likelihood) has no coordinate to constrain, so it is DRAWN from its prior
using `rng`, conditioned on this position's other latents.
"""
jconstrained(prepared::JulianicPrepared, theta::AbstractVector;
             rng::Random.AbstractRNG=Random.default_rng()) =
    NamedTuple(_julianic_record(prepared, theta, _obs_logpdf, rng).sites)

"""
    jpointwise(prepared, theta) -> NamedTuple

Per-observation log-likelihood at `theta`, keyed by observation site — the
elementwise terms whose sum the density accumulates, which is what PSIS-LOO and
WAIC consume.
"""
jpointwise(prepared::JulianicPrepared, theta::AbstractVector;
           rng::Random.AbstractRNG=Random.default_rng()) =
    NamedTuple(_julianic_record(prepared, theta, _obs_logpdf, rng).observations)

"""
    jpredict([rng], prepared, theta) -> NamedTuple

One posterior-predictive draw per observation site at `theta`: the body's own
observation distributions, resampled instead of scored. Discrete families keep
their natural element type.
"""
jpredict(rng::Random.AbstractRNG, prepared::JulianicPrepared, theta::AbstractVector) =
    NamedTuple(_julianic_record(
        prepared, theta, JulianicPredictKernel(rng), rng).observations)
jpredict(prepared::JulianicPrepared, theta::AbstractVector) =
    jpredict(Random.default_rng(), prepared, theta)

"""
    jsimulate([rng], prepared) -> (; latents, observations)

Draw the whole model FORWARD from its priors, with no position: every latent is
sampled from its own prior and every observation is generated from the resulting
distribution. `latents` are natural-space site values, `observations` the prior
predictive — the julianic counterpart of the declarative `simulate_prior`.

Both halves come back because a prior predictive check needs them together: the
generated data is only interpretable against the parameters that generated it,
and returning them separately would mean running the model twice and getting two
different draws.

Note this is the ONE pass that does not go through the `_sample!` family table.
It does not need to: a prior draw is `rand(dist)` for every family at once,
precisely because forward simulation never touches the unconstrained
parametrization that the family methods exist to handle.
"""
function jsimulate(rng::Random.AbstractRNG, prepared::JulianicPrepared)
    # Growable, not presized — the forward pass computes generated-quantity
    # deterministics the density skips, so its pooled set differs from the density's
    # `buffer_sizes` (see `_julianic_record`). Off the 0-alloc hot path.
    buffers = JulianicBuffers{Float64}()
    context = JulianicSimulate(rng, buffers, JulianicPredictKernel(rng))
    prepared.run(context, prepared.data, prepared.roles, prepared.store)
    return (; latents=NamedTuple(context.sites),
              observations=NamedTuple(context.observations))
end
jsimulate(prepared::JulianicPrepared) = jsimulate(Random.default_rng(), prepared)

# --- Draws ------------------------------------------------------------------
#
# The batch forms. `positions` is a matrix of unconstrained draws with ROWS =
# draws, matching the declarative `evaluate_draws!` convention so a sampler's
# output feeds either surface unchanged.
#
# Each site's draws are stacked into one dense array whose FIRST axis is the
# draw, so a scalar site becomes an `ndraws` vector, a length-k site an
# `ndraws × k` matrix, and the LKJ factor an `ndraws × K × K` array. That layout
# is what a labelled container wants (`TreeData(store, :draw, …)` — see the
# TreeArrays extension) and what a plain reduction over draws wants.
@inline _julianic_draw_store(value::Number, ndraws::Int) =
    Vector{typeof(value)}(undef, ndraws)
@inline _julianic_draw_store(value::AbstractArray, ndraws::Int) =
    Array{eltype(value)}(undef, ndraws, size(value)...)

@inline _julianic_store_draw!(store, draw::Int, value::Number) =
    (store[draw] = value; nothing)
@inline _julianic_store_draw!(store, draw::Int, value::AbstractArray) =
    (selectdim(store, 1, draw) .= value; nothing)

@noinline _julianic_draw_shape_error(draw::Int, expected, got) =
    throw(ArgumentError(
        "julianic draws: draw " * string(draw) * " recorded sites " *
        string(got) * " but draw 1 recorded " * string(expected) * ". The body " *
        "takes a different path per position, so the draws cannot be stacked."))

function _julianic_draws(prepared::JulianicPrepared, positions::AbstractMatrix,
                         kernel, selector::F,
                         rng::Random.AbstractRNG=Random.default_rng()) where {F}
    ndraws = size(positions, 1)
    ndraws >= 1 || throw(ArgumentError(
        "julianic draws: `positions` needs at least one row (rows are draws)"))
    recorded = selector(
        _julianic_record(prepared, view(positions, 1, :), kernel, rng))
    names = Tuple(pair.first for pair in recorded)
    stores = Tuple(_julianic_draw_store(pair.second, ndraws) for pair in recorded)
    for (store, pair) in zip(stores, recorded)
        _julianic_store_draw!(store, 1, pair.second)
    end
    for draw in 2:ndraws
        recorded = selector(
            _julianic_record(prepared, view(positions, draw, :), kernel, rng))
        # A body whose site set depends on the position would stack garbage under
        # the first draw's names, so check rather than trust the shape.
        Tuple(pair.first for pair in recorded) == names ||
            _julianic_draw_shape_error(
                draw, names, Tuple(pair.first for pair in recorded))
        for (store, pair) in zip(stores, recorded)
            _julianic_store_draw!(store, draw, pair.second)
        end
    end
    return NamedTuple{names}(stores)
end

_julianic_sites(context::JulianicRecording) = context.sites
_julianic_observations(context::JulianicRecording) = context.observations

"""
    jconstrained_draws(prepared, positions) -> NamedTuple

`jconstrained` over a matrix of unconstrained draws (ROWS = draws). Each site's
draws are stacked with the draw as the FIRST axis.
"""
jconstrained_draws(prepared::JulianicPrepared, positions::AbstractMatrix) =
    _julianic_draws(prepared, positions, _obs_logpdf, _julianic_sites)

"""
    jpointwise_draws(prepared, positions) -> NamedTuple

`jpointwise` over a matrix of unconstrained draws (ROWS = draws), stacked as
`ndraws × nobservations` per observation site — the layout PSIS-LOO expects.
"""
jpointwise_draws(prepared::JulianicPrepared, positions::AbstractMatrix) =
    _julianic_draws(prepared, positions, _obs_logpdf, _julianic_observations)

"""
    jpredict_draws([rng], prepared, positions) -> NamedTuple

`jpredict` over a matrix of unconstrained draws (ROWS = draws): the posterior
predictive, one redrawn replicate per draw per observation.
"""
jpredict_draws(rng::Random.AbstractRNG, prepared::JulianicPrepared,
               positions::AbstractMatrix) =
    _julianic_draws(prepared, positions, JulianicPredictKernel(rng),
                    _julianic_observations)
jpredict_draws(prepared::JulianicPrepared, positions::AbstractMatrix) =
    jpredict_draws(Random.default_rng(), prepared, positions)

# Labelled-container forms, given methods in
# ext/BayesianRegressionModelsTreeArraysExt.jl. They are a weak-dep extension
# rather than core because a labelled container is a POSTPROCESSING convenience:
# what the axes mean, attached to the very arrays the functions above already
# return. A consumer who only wants the density never loads TreeArrays.
function jconstrained_tree end
function jpointwise_tree end
function jpredict_tree end

# --- LogDensityProblems -----------------------------------------------------
#
# The interface every downstream consumer actually reaches for (WarmupHMC and
# the samplers take a `LogDensityProblem`, not a `Prepared`), so julianic cannot
# stand in for the declarative surface without it. Deliberately the same shape as
# `FactorLogDensityProblem`: the problem OWNS its workspace, so `logdensity` and
# `logdensity_and_gradient` hit the 0-alloc pooled paths rather than allocating
# per call.
struct JulianicLogDensityProblem{P,W}
    prepared::P
    workspace::W
end

"""
    JulianicLogDensityProblem(prepared, backend; eltype=Float64)

Wrap a prepared julianic model as a `LogDensityProblems` problem of order 1,
allocating the gradient workspace (and the 0-alloc buffer pool) once.
"""
function JulianicLogDensityProblem(prepared::JulianicPrepared, backend;
                                   eltype::Type{T}=Float64) where {T<:AbstractFloat}
    work = jworkspace(prepared, T, backend)
    JulianicLogDensityProblem{typeof(prepared),typeof(work)}(prepared, work)
end

Base.eltype(problem::JulianicLogDensityProblem) = eltype(problem.workspace.gradient)
BRM.LogDensityProblems.capabilities(::Type{<:JulianicLogDensityProblem}) =
    BRM.LogDensityProblems.LogDensityOrder{1}()
BRM.LogDensityProblems.dimension(problem::JulianicLogDensityProblem) =
    problem.prepared.dimension
BRM.LogDensityProblems.logdensity(problem::JulianicLogDensityProblem,
                                  position::AbstractVector) =
    jlogdensity!(problem.workspace, problem.prepared, position)
# `copy` because the workspace's gradient buffer is reused on the next call —
# the same contract the declarative problem keeps.
function BRM.LogDensityProblems.logdensity_and_gradient(
        problem::JulianicLogDensityProblem, position::AbstractVector)
    density, gradient = jlogdensity_and_gradient!(
        problem.workspace, problem.prepared, position)
    density, copy(gradient)
end

# --- Macro ----------------------------------------------------------------

# Detect a sampling statement in ANY spelling, and normalize it.
#
# There is ONE rule for `~`; the dot carries only its ordinary Julia meaning
# (lift elementwise) and never decides what the statement MEANS. So `@. y ~ D(l, s)`
# and `y .~ D.(l, s)` are the same statement, which is exactly the identity Julia
# itself gives them — and the normalization is done BY Julia: `Base.Broadcast.__dot__`
# is the function behind `@.`, so the `@.` spelling is dotted by the same code that
# dots every other `@.` in the language rather than by a rewriter of mine.
#
# Binary `.~` has no meaning in Base, so matching it cannot shadow ordinary Julia
# (unary `~x` has `length(args) == 2` and is left alone).
function _julianic_sample_statement(statement)
    statement isa Expr || return nothing
    if statement.head === :macrocall
        macro_name = statement.args[1]
        (macro_name === Symbol("@__dot__") || macro_name === Symbol("@.")) ||
            return nothing
        inner = statement.args[end]
        inner isa Expr && inner.head === :call && length(inner.args) == 3 &&
            inner.args[1] === :~ || return nothing
        return (lhs=inner.args[2], rhs=Base.Broadcast.__dot__(inner.args[3]),
                broadcast=true)
    end
    statement.head === :call && length(statement.args) == 3 || return nothing
    statement.args[1] === :~ &&
        return (lhs=statement.args[2], rhs=statement.args[3], broadcast=false)
    statement.args[1] === :.~ &&
        return (lhs=statement.args[2], rhs=statement.args[3], broadcast=true)
    return nothing
end

# Rewrite dotted syntax into its LAZY form: `f.(a, b)` -> `broadcasted(f, a, b)`,
# `a .+ b` -> `broadcasted(+, a, b)`, recursively. `broadcasted` is what `f.(x)`
# builds before `materialize` runs, so this is Julia's own fusion machinery with
# the final materialize withheld — which is the whole point: a lazy vector of
# distributions is never built, so an elementwise latent costs no allocation.
#
# One exception, stated rather than hidden: `broadcasted` accepts no KEYWORD
# arguments, so a dotted call carrying them (`censored.(D.(mu, s); lower, upper)`)
# cannot be made lazy and is left materializing. That costs an allocation per
# evaluation — but only for an UNCONDITIONED kwarg site, because for a conditioned
# one this branch is eliminated before it runs.
_julianic_lazy(ex) = ex
function _julianic_lazy(ex::Expr)
    if ex.head === :. && length(ex.args) == 2 && ex.args[2] isa Expr &&
            ex.args[2].head === :tuple
        any(a -> a isa Expr && a.head === :parameters, ex.args[2].args) && return ex
        return Expr(:call, Base.Broadcast.broadcasted, ex.args[1],
                    (_julianic_lazy(a) for a in ex.args[2].args)...)
    end
    if ex.head === :call
        operator = _julianic_dotted_operator(ex.args[1])
        operator === nothing || return Expr(:call, Base.Broadcast.broadcasted, operator,
                                            (_julianic_lazy(a) for a in ex.args[2:end])...)
    end
    return ex
end

# Lower one `~`, whatever dots surround it. The statement is prep + accumulate:
#
#   * the value comes from the CONDITIONED DATA if the name was passed to
#     `jcondition`, and from a fresh slice of theta otherwise;
#   * the log-density contribution is accumulated either way.
#
# Which one is decided by `_julianic_is_conditioned`, whose arguments are both
# compile-time constants (the name rides a `Val`, the data's field names are in
# its type), so the branch folds to a literal and the dead half is eliminated —
# no runtime test, and both halves stay inline generated code rather than
# closures, which is what keeps Enzyme's activity analysis clean.
function _julianic_lower_sample(lhs, rhs, broadcast::Bool, ctx, data, roles)
    lhs isa Symbol || throw(ArgumentError(
        "julianic @jmodel: `~` left-hand side must be a name; got `$lhs`"))
    name = Val(lhs)
    data_var = gensym(lhs)
    buffer_var = gensym(:obs)
    kernel_var = gensym(:kernel)
    # The observation half. `_obs_kernel` picks WHAT each site computes from the
    # context (logpdf for the density, a redraw for the posterior predictive), so
    # one lowered body serves every query. A scalar observation is just the n=1
    # case — same buffer, same accumulate — rather than its own code path.
    fill = broadcast ?
        :($buffer_var .= $kernel_var.($rhs, $data_var)) :
        :($buffer_var[1] = $kernel_var($rhs, $data_var))
    size = broadcast ? :(length($data_var)) : 1
    observed = quote
        $kernel_var = $(_obs_kernel)($ctx)
        $data_var = $(_observation)($data, $name)
        $buffer_var = $(_site_buffer!)($ctx, $size)
        $fill
        $(_obs_end!)($ctx, $name, $buffer_var)
        $data_var
    end
    # A latent `~` splits three ways, each guard a compile-time constant:
    #   conditioned → observation (above);
    #   generated-quantity (unconditioned, unreachable from the likelihood) →
    #     no coordinate, drawn only in postprocessing;
    #   sampled (the default) → consume a coordinate, score the prior.
    # `_julianic_is_gq` is `false` for every model with no gq latent, so the
    # `elseif` folds away and the lowering is identical to the pre-AA surface.
    sampled = broadcast ?
        :($(_sample_broadcast!)($ctx, $name, $(_julianic_lazy(rhs)))) :
        :($(_sample_site!)($ctx, $data, $name, $rhs))
    gq = broadcast ?
        :($(_sample_broadcast_gq!)($ctx, $name, $(_julianic_lazy(rhs)))) :
        :($(_sample_gq!)($ctx, $name, $rhs))
    return :($lhs = if $(_julianic_is_conditioned)($data, $name)
                 $observed
             elseif $(_julianic_is_gq)($roles, $name)
                 $gq
             else
                 $sampled
             end)
end

# --- 0-alloc rewrite of deterministic array expressions -------------------
#
# Every array-valued intermediate the body computes is routed through a pooled
# buffer instead of a fresh allocation, so the differentiated kernel is literally
# 0-alloc (user directive 2026-08-04; decision `08w0buk` option B). The rewrite is
# per-NODE, bottom-up:
#
#   `beta[1] .+ beta[2] .* x .+ b[group]`
#     ->  t1 = _cached_gather!(ctx, b, group)             # b[group]
#         t2 = _cached_broadcast!(ctx, *, beta[2], x)     # beta[2] .* x
#         t3 = _cached_broadcast!(ctx, +, beta[1], t2)    # ... .+ ...
#         t4 = _cached_broadcast!(ctx, +, t3, t1)
#
# which is exactly `apply!!(buffer, broadcast|getindex, …)` — MutatingFunctions'
# registered 0-alloc forms — with the buffers presized from the trace pass.
# Scalar-only expressions are left VERBATIM (nothing to buffer), so the body stays
# ordinary Julia wherever buffering buys nothing.

# A dotted operator (`.+`) or dot-call (`exp.(v)`) is a broadcast node; `A[idx]`
# with a non-literal index is a gather node. Anything else is left alone.
_julianic_dotted_operator(f) =
    f isa Symbol && length(string(f)) > 1 && startswith(string(f), ".") ?
        Symbol(string(f)[2:end]) : nothing

# Rewrite one expression bottom-up, emitting pooled-buffer ops into `prelude`.
# Returns the expression to use in place of `ex`.
function _julianic_buffer_expression!(prelude, ex, ctx)
    ex isa Expr || return ex
    # `exp.(v)` / `f.(a, b)` — an explicit dot-call.
    if ex.head === :. && length(ex.args) == 2 && ex.args[2] isa Expr &&
            ex.args[2].head === :tuple
        f = ex.args[1]
        args = [_julianic_buffer_expression!(prelude, a, ctx) for a in ex.args[2].args]
        tmp = gensym(:bc)
        push!(prelude, :($tmp = $(_cached_broadcast!)($ctx, $f, $(args...))))
        return tmp
    end
    if ex.head === :call
        operator = _julianic_dotted_operator(ex.args[1])
        if operator !== nothing
            args = [_julianic_buffer_expression!(prelude, a, ctx) for a in ex.args[2:end]]
            tmp = gensym(:bc)
            push!(prelude, :($tmp = $(_cached_broadcast!)($ctx, $operator, $(args...))))
            return tmp
        end
        # Non-dotted call (`L * z`, `reshape(z, 2, n)`, `maximum(group)`, …) —
        # recurse into the arguments but keep the call itself verbatim. `*` on
        # arrays already has a registered mutating form, but it needs an
        # output-shaped (possibly matrix) buffer, so it is handled by the
        # matrix-aware path in a later milestone rather than forced through the
        # vector pool here.
        return Expr(:call, ex.args[1],
                    (_julianic_buffer_expression!(prelude, a, ctx)
                     for a in ex.args[2:end])...)
    end
    # `b[group]` — a gather when the index is not a literal scalar.
    if ex.head === :ref && length(ex.args) == 2 && !(ex.args[2] isa Integer)
        A = _julianic_buffer_expression!(prelude, ex.args[1], ctx)
        idx = _julianic_buffer_expression!(prelude, ex.args[2], ctx)
        tmp = gensym(:gather)
        push!(prelude, :($tmp = $(_cached_gather!)($ctx, $A, $idx)))
        return tmp
    end
    return ex
end

# Lower one statement. Only `~` statements change MEANING; deterministic
# statements keep their semantics exactly and are only re-routed through pooled
# buffers (still ordinary Julia, just allocation-free).
# `ctx` names the (mutable) sampling context; `data` names the conditioned
# observation data — both are parameters of the generated `run` closure.
function _julianic_lower_statement(statement, ctx, data, roles, store)
    # `end`/`begin` inside `A[...]` are POSITIONAL markers — they mean nothing
    # outside the indexing expression that encloses them. The buffer rewrite
    # hoists a gather into its own statement (`t = _cached_gather!(ctx, A, idx)`),
    # which carries the index OUT of that context, so `z[1:2:end]` — ordinary
    # Julia the surface promises to run — would lower to a bare undefined `end`.
    # Resolve them to `lastindex`/`firstindex` calls FIRST, exactly as Base's own
    # `@views`/`@view` do, so the index is a self-contained expression before
    # anything moves it.
    statement = Base.replace_ref_begin_end!(statement)
    sample = _julianic_sample_statement(statement)
    if sample !== nothing
        return _julianic_lower_sample(
            sample.lhs, sample.rhs, sample.broadcast, ctx, data, roles)
    end
    # Deterministic assignment `lhs = rhs`: buffer every array-valued node in rhs,
    # then GATE the whole computation on `lhs`'s role. Every guard folds to a
    # compile-time constant at `run`'s specialization, so a `:density` write folds
    # straight to the final else branch — identical to the pre-AA lowering, buffer
    # ops and all — while the other roles prune the density's work away entirely:
    #   * data-only — computed ONCE in the trace by a PLAIN (un-pooled) broadcast, so
    #     it takes no buffer slot there; the density and every other pass FETCH it
    #     from `store`, so it takes no slot there either. The cursors stay aligned
    #     with the `:data` array simply absent from both slot lists.
    #   * gq deterministic — computed only in the recording passes (a gq draw may
    #     read it); the density and trace get `nothing`, so its buffer ops are never
    #     emitted and it consumes no pool slot.
    if statement isa Expr && statement.head === :(=)
        target = statement.args[1]
        plain = statement.args[2]           # the raw rhs — plain Julia, no pooled ops
        prelude = Any[]
        rewritten = _julianic_buffer_expression!(prelude, statement.args[2], ctx)
        compute = isempty(prelude) ? rewritten : Expr(:block, prelude..., rewritten)
        # Only a plain `name = …` write carries a role. The SSA check has already
        # rejected index/destructuring/mutation targets, so a non-Symbol target here
        # is unreachable; keep the un-gated form for it defensively.
        target isa Symbol || return isempty(prelude) ? statement :
            Expr(:block, prelude..., Expr(:(=), target, rewritten))
        name = Val(target)
        gated = :(if $(_julianic_is_data)($roles, $name)
                      if $(_julianic_captures_data)($ctx)
                          $(_julianic_record_data!)($ctx, $name, $plain)
                      else
                          $(_julianic_fetch_data)($store, $name)
                      end
                  elseif $(_julianic_is_gq_det)($roles, $name)
                      $(_julianic_computes_det)($ctx) ? $compute : nothing
                  else
                      $compute
                  end)
        return Expr(:(=), target, gated)
    end
    return statement
end

# Any `~` still present AFTER lowering is a sampling statement nested inside
# control flow (`for` / `if` / `begin`), which milestone 1 does not lower — it
# would survive verbatim and fail at run time with an `UndefVarError` naming the
# site, far from its cause. Reject it at macro-expansion time instead.
function _julianic_assert_lowered(expr)
    expr isa Expr || return nothing
    if _julianic_sample_statement(expr) !== nothing
        throw(ArgumentError(
            "julianic @jmodel: `~` is only lowered at the TOP LEVEL of the " *
            "model body, but found a nested sampling statement `" *
            string(expr) * "`. Hoist it to the top level; a vector of " *
            "independent sites is written elementwise (`b .~ Normal.(0, tau)`) " *
            "and indexed afterwards in ordinary Julia."))
    end
    for argument in expr.args
        _julianic_assert_lowered(argument)
    end
    return nothing
end

# --- Body restrictions (what makes the activity analysis sound) -----------
#
# The activity analysis (below, at `jprepare`) reads the body as a STATIC
# single-assignment statement list: the statement ORDER is a topological sort of
# the dependency graph, so one forward + one backward sweep over the names each
# statement reads and writes computes the data / needed-for-sampling /
# generated-quantity partition with NO values. Two properties make that sound, and
# each is enforced here with a loud macro-expansion error rather than left as a
# silent assumption:
#
#   * NO value-dependent control flow. A branch or loop makes "which statements
#     run" depend on the position, so one static statement list could no longer
#     stand for the graph. Same constraint StanBlocks imposes.
#   * SINGLE ASSIGNMENT. Each name is written once, so "the statement that defines
#     `n`" is unambiguous — that is exactly the edge the analysis follows. In-place
#     mutation (`x[i] = v`, `x .= v`, `x += v`) and reassignment both break it.
#
# Both NARROW the surface the walking skeleton accepted; the gallery uses neither
# (verified: no body has control flow, in-place mutation, or a repeated
# left-hand side), so nothing regresses.

const _JULIANIC_CONTROL_FLOW_HEADS = Dict{Symbol,String}(
    :for => "a `for` loop", :while => "a `while` loop",
    :if => "an `if` / `?:` branch", :elseif => "an `elseif` branch",
    :(&&) => "an `&&` short-circuit", :(||) => "an `||` short-circuit",
    :comprehension => "a comprehension", :generator => "a generator",
    :flatten => "a nested generator", :do => "a `do` block",
    :try => "a `try` block", :break => "a `break`", :continue => "a `continue`",
    :let => "a `let` block",
)

@noinline _julianic_control_flow_error(kind, statement) = throw(ArgumentError(
    "julianic @jmodel: the model body must be single-assignment Julia with no " *
    "value-dependent control flow — the activity analysis reads it as a static " *
    "statement list — but found " * kind * " in `" * string(statement) * "`. " *
    "Rewrite without control flow: a vector of independent sites is " *
    "`b .~ Normal.(0, tau)`, a conditional mean is arithmetic or indexing."))

# Reject any control-flow / scope-introducing construct anywhere in a RAW
# statement (before lowering, so the `if` the lowering itself emits is never seen).
function _julianic_assert_no_control_flow(root)
    root isa Expr || return nothing
    kind = get(_JULIANIC_CONTROL_FLOW_HEADS, root.head, nothing)
    kind === nothing || _julianic_control_flow_error(kind, root)
    for argument in root.args
        _julianic_assert_no_control_flow(argument)
    end
    return nothing
end

# The Expr heads ending in `=` are exactly `:(=)` (plain assignment) and the
# augmented / broadcast assignments (`:+=`, `:.+=`, `:.=`, …); the comparison
# operators (`<=`, `==`) are `:call` heads, never Expr heads. So "ends in `=`,
# is not plain `=`" cleanly identifies an in-place update.
_julianic_is_mutation_head(h) = h isa Symbol && h !== :(=) && endswith(string(h), "=")

@noinline _julianic_ssa_error(statement, reason) = throw(ArgumentError(
    "julianic @jmodel: the model body must be single-assignment, but `" *
    string(statement) * "` " * reason * ". Bind a NEW name instead — the activity " *
    "analysis follows each name to ONE defining statement, which in-place " *
    "mutation and destructuring make ambiguous."))

# The name a raw statement writes, or `nothing` for a bare expression. Throws on
# any non-single-assignment form (compound / broadcast assign, setindex,
# destructuring), which the SSA restriction forbids. `sample` is the precomputed
# `_julianic_sample_statement` result (a `~` writes its site name).
function _julianic_statement_write(statement, sample)
    # A `~` writes its site name — but only a Symbol lhs is a valid write. A
    # non-Symbol lhs (`x[1] ~ …`) is left for `_julianic_lower_sample` to reject
    # with its precise "lhs must be a name" error, so return `nothing` here rather
    # than track a non-Symbol in the (Symbol-typed) written set.
    sample === nothing || return sample.lhs isa Symbol ? sample.lhs : nothing
    statement isa Expr || return nothing
    _julianic_is_mutation_head(statement.head) &&
        _julianic_ssa_error(statement, "mutates in place (`x .= v` / `x += v`)")
    statement.head === :(=) || return nothing          # bare expression: no write
    target = statement.args[1]
    target isa Symbol && return target
    target isa Expr && target.head === :ref &&
        _julianic_ssa_error(statement, "assigns into an index (`x[i] = v`)")
    _julianic_ssa_error(statement, "is not a plain `name = value` assignment")
end

@noinline _julianic_reassignment_error(name, statement) = throw(ArgumentError(
    "julianic @jmodel: `" * string(name) * "` is assigned more than once (at `" *
    string(statement) * "`). The body must be single-assignment so the activity " *
    "analysis follows each name to one defining statement; use a new name."))

# The free NAMES an expression reads, for the static skeleton. Over-collected on
# purpose: a symbol with no model binding (`exp`, `Normal`, a range `end`) simply
# gets no `defindex` entry in the analysis, so it contributes no dependency edge.
# A MISSED read would be unsound (a hidden edge), so the walk errs toward
# collecting: the only things skipped are things that are provably NOT variable
# reads — a `.field` name, a keyword-argument key, a macro name.
_julianic_collect_symbols!(_acc, _x) = nothing
_julianic_collect_symbols!(acc, s::Symbol) = (push!(acc, s); nothing)
function _julianic_collect_symbols!(acc, ex::Expr)
    if ex.head === :.                               # `a.field` vs broadcast `f.(x)`
        _julianic_collect_symbols!(acc, ex.args[1])
        if length(ex.args) >= 2 && ex.args[2] isa Expr && ex.args[2].head === :tuple
            for a in ex.args[2].args                # broadcast args ARE reads
                _julianic_collect_symbols!(acc, a)
            end
        end                                         # a QuoteNode field name is not
        return nothing
    elseif ex.head === :kw                          # `key = val` — only val is read
        _julianic_collect_symbols!(acc, ex.args[2])
        return nothing
    elseif ex.head === :macrocall                   # skip the macro name (args[1])
        for a in ex.args[2:end]
            _julianic_collect_symbols!(acc, a)
        end
        return nothing
    end
    for a in ex.args
        _julianic_collect_symbols!(acc, a)
    end
    return nothing
end
function _julianic_free_symbols(ex)
    collected = Symbol[]
    _julianic_collect_symbols!(collected, ex)
    seen = Set{Symbol}()
    reads = Symbol[]
    for s in collected
        s in seen || (push!(seen, s); push!(reads, s))
    end
    return reads
end

# The skeleton statement for one lowered body statement — the syntactic facts the
# analysis runs on. Built from the RAW statement (a `~` reads its distribution's
# free names; a `lhs = rhs` reads rhs's; a bare expression writes nothing).
function _julianic_skeleton_statement(statement, sample, write)
    # `write` is `nothing` for a bare expression, and also for a `~` with a
    # non-Symbol lhs that lowering is about to reject — coerce both to `Symbol("")`
    # so the skeleton is well-formed even for the statement whose lowering throws.
    w = write isa Symbol ? write : Symbol("")
    if sample !== nothing
        return JulianicStmt(w, _julianic_free_symbols(sample.rhs), true,
                            sample.broadcast)
    end
    reads = statement isa Expr && statement.head === :(=) ?
        _julianic_free_symbols(statement.args[2]) : Symbol[]
    return JulianicStmt(w, reads, false, false)
end

function _julianic_model_syntax(definition)
    definition isa Expr && definition.head === :function ||
        throw(ArgumentError("julianic @jmodel must wrap a function definition"))
    # Index lowering rewrites nested Expr nodes. Own the complete syntax tree,
    # including the signature returned below, before touching any of them.
    definition = deepcopy(definition)
    signature, body = definition.args
    signature isa Expr && signature.head === :call ||
        throw(ArgumentError("julianic @jmodel requires a named function definition"))
    function_name = first(signature.args)
    arguments = signature.args[2:end]
    input_names = Tuple(Symbol[_julianic_argument_name(a) for a in arguments])

    statements = body isa Expr && body.head === :block ? body.args : Any[body]
    ctx = gensym(:ctx)
    data = gensym(:data)
    roles = gensym(:roles)
    store = gensym(:store)
    lowered = Any[]
    skeleton = JulianicStmt[]
    # Function-argument names are pre-bound; a statement re-writing one is a
    # single-assignment violation, so they seed the written-name set.
    written = Set{Symbol}(input_names)
    for statement in statements
        statement isa LineNumberNode && (push!(lowered, statement); continue)
        # Drop an explicit `return` — the density is the accumulator, not the
        # body's return value (the return only named outputs in `@model`).
        if statement isa Expr && statement.head === :return
            continue
        end
        # Enforce the body restrictions BEFORE lowering, so the checks see the
        # author's syntax rather than the `if`/`.=` the lowering introduces.
        _julianic_assert_no_control_flow(statement)
        sample = _julianic_sample_statement(statement)
        write = _julianic_statement_write(statement, sample)
        if write !== nothing
            write in written && _julianic_reassignment_error(write, statement)
            push!(written, write)
        end
        # Record the static skeleton BEFORE lowering mutates the statement
        # (`replace_ref_begin_end!`) — the analysis wants the author's own reads.
        push!(skeleton, _julianic_skeleton_statement(statement, sample, write))
        lowered_statement = _julianic_lower_statement(statement, ctx, data, roles, store)
        _julianic_assert_lowered(lowered_statement)
        push!(lowered, lowered_statement)
    end

    run_closure = Expr(:function, Expr(:call, gensym(:run), ctx, data, roles, store),
                       Expr(:block, lowered..., nothing))
    # The skeleton is real data built here; splice it in as a literal object so
    # `jprepare` can run the analysis over it at runtime.
    model_expr = :($(JulianicModel)($run_closure, $(input_names), $(skeleton)))
    return Expr(:function, signature, Expr(:block, model_expr))
end

_julianic_argument_name(argument::Symbol) = argument
function _julianic_argument_name(argument::Expr)
    argument.head === :(::) && return _julianic_argument_name(argument.args[1])
    argument.head === :kw && return _julianic_argument_name(argument.args[1])
    throw(ArgumentError("julianic @jmodel unsupported argument form `$argument`"))
end

"""
    @jmodel function name(inputs...)
        sigma ~ Exponential(2.0)    # prep + accumulate
        b .~ Normal.(0, tau)        # k independent sites, by ordinary broadcasting
        mu = f(b)                   # ordinary Julia, kept verbatim
        @. y ~ Normal(mu, sigma)    # identical to `y .~ Normal.(mu, sigma)`
    end

Julianic NativePPL model: the body is valid SSA Julia except `~`, which reduces
to prep + accumulate. Calling `name(inputs...)` returns a `JulianicModel`;
condition it with `jcondition`, `jprepare` it, then evaluate with `jlogdensity` /
`jlogdensity_and_gradient!` (see `jworkspace` for the required AD backend).

Every `~` is ONE operation, whatever dots surround it:

  * **prep** — if the name was passed to `jcondition` the value is that data; if
    not, it is a fresh slice of the unconstrained position, constrained with its
    log-Jacobian.
  * **accumulate** — the log-density contribution, either way.

So **conditioning decides what is data**, not spelling: `y ~ Normal(mu, sigma)`
scores an observation when `y` is conditioned and declares a latent when it is
not. The dot carries only its ordinary Julia meaning — lift elementwise — which
is why `@. y ~ D(a, b)` and `y .~ D.(a, b)` are the same statement, and why
`b .~ Normal.(0, tau)` is simply k independent sites.

`~` is lowered at the TOP LEVEL of the body only; a nested one is a
macro-expansion error rather than an opaque runtime failure.
"""
macro jmodel(definition)
    esc(_julianic_model_syntax(definition))
end
