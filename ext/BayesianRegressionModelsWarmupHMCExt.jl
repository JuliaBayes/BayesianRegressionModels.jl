module BayesianRegressionModelsWarmupHMCExt

using BayesianRegressionModels
using WarmupHMC

const BRM = BayesianRegressionModels

# Random-effect block states: ordinary blocks (`BRMAdaptiveCenteringState`) and
# R2D2 blocks (`BRMR2D2AdaptiveCenteringState`) share one cell layout and one
# triangular map; they differ only in where `tau` comes from (`_block_tau`).
# `N` is the innovation capacity `_block_location` keeps on the stack: the
# widest block's K-1, and at least 1, so every K<=2 plan shares one concrete
# type (the exact pullback in BayesianRegressionModelsWarmupHMCEnzymeExt
# dispatches on `N == 1`).
abstract type _BRMRanefCenteringState{N} end

_innovation_capacity(blocks) =
    max(1, maximum((b.ranef.n_terms for b in blocks); init=1) - 1)

mutable struct BRMAdaptiveCenteringState{N} <: _BRMRanefCenteringState{N}
    blocks::Vector{BRM.AdaptiveCenteringBlock}
    pair_blocks::Vector{Int}
    pair_terms::Vector{Int}
    pair_groups::Vector{Int}
    pair_mixing::Vector{Int}
    pair_lookup::Vector{Matrix{Int}}
    sources::Vector{Float64}
    effect_indices::BitSet
end

# Pairs are block-major, then group-major, then term-major. `pair_mixing[p]`
# is the coordinate `log w[g]` of a Student-t pair's level, `0` for a Gaussian
# block: accessors read that integer once per call, never the block's `mixing`
# vector inside their loops (see `_pair_mixing_scale`).
function _ranef_pair_layout(blocks)
    pair_blocks = Int[]
    pair_terms = Int[]
    pair_groups = Int[]
    pair_mixing = Int[]
    pair_lookup = [zeros(Int, b.ranef.n_terms, b.ranef.n_groups) for b in blocks]
    sources = Float64[]
    for (bi, block) in enumerate(blocks), g in 1:block.ranef.n_groups,
            k in 1:block.ranef.n_terms
        push!(pair_blocks, bi)
        push!(pair_terms, k)
        push!(pair_groups, g)
        push!(pair_mixing, _block_mixed(block) ? block.mixing[g] : 0)
        push!(sources, block.target_c)
        pair_lookup[bi][k, g] = length(sources)
    end
    effect_indices = BitSet(Iterators.flatten(vec(b.effects) for b in blocks))
    (pair_blocks, pair_terms, pair_groups, pair_mixing, pair_lookup, sources,
     effect_indices)
end

BRMAdaptiveCenteringState(blocks) =
    BRMAdaptiveCenteringState{_innovation_capacity(blocks)}(
        collect(blocks), _ranef_pair_layout(blocks)...,
    )

# Derived-scale blocks — R2D2 blocks, and ordinary blocks inside the joint
# wrapper — keep their own state and argument types for the same reason HSGP
# and cdar cells do: the exact K<=2 Enzyme reverse pass dispatches on the
# ordinary state's concrete pair type and must never see a derived scale.
mutable struct BRMR2D2AdaptiveCenteringState{N} <: _BRMRanefCenteringState{N}
    blocks::Vector{BRM.R2D2AdaptiveCenteringBlock}
    pair_blocks::Vector{Int}
    pair_terms::Vector{Int}
    pair_groups::Vector{Int}
    pair_mixing::Vector{Int}
    pair_lookup::Vector{Matrix{Int}}
    sources::Vector{Float64}
    effect_indices::BitSet
end

BRMR2D2AdaptiveCenteringState(blocks) =
    BRMR2D2AdaptiveCenteringState{_innovation_capacity(blocks)}(
        collect(blocks), _ranef_pair_layout(blocks)...,
    )

# A block's marginal scale `tau[i]`: a sampled coordinate for an ordinary
# block, the emitted model's derived scale for an R2D2 block.
_block_tau(x, block::BRM.AdaptiveCenteringBlock, i) = exp(x[block.log_scales[i]])
_block_tau(x, block::BRM.R2D2AdaptiveCenteringBlock, i) =
    exp(BRM._adaptive_log_scale(x, block.table, i))

# A Student-t block (`mixing` non-empty) scales level `g`'s factor by
# `sqrt(w[g])`, the level's mixing weight read from its coordinate `log w[g]`.
_block_mixed(::BRM.AdaptiveCenteringBlock) = false
_block_mixed(block::BRM.R2D2AdaptiveCenteringBlock) = !isempty(block.mixing)
_block_mixing_scale(x, ::BRM.AdaptiveCenteringBlock, g) = one(eltype(x))
_block_mixing_scale(x, block::BRM.R2D2AdaptiveCenteringBlock, g) =
    exp(x[block.mixing[g]] / 2)

# Pair `p`'s level scale `sqrt(w[g])`, read once per accessor call from the
# state's flat `pair_mixing`; one for a Gaussian block, whose operations the
# product `tau * one(T)` leaves bit-identical. The differentiated accessors
# never read the block's `mixing` vector: on Julia 1.10, adding that read (and
# its branch) to `_block_location`'s row loop corrupted Julia's GC under
# Enzyme's reverse pass in `test/adaptive_centering_threads.jl`, even for
# blocks without weights.
function _pair_mixing_scale(x, state, p)
    j = state.pair_mixing[p]
    j == 0 ? one(eltype(x)) : exp(x[j] / 2)
end

# Canonical partial correlation `p` of a block's correlation factor, in Stan's
# `cholesky_factor_corr` coordinate order: `tanh` of its coordinate, or the
# constant of a known correlation (`cor(:, ID) = R`), which has no coordinates.
_block_cpc(x, block::BRM.AdaptiveCenteringBlock, p) = tanh(x[block.cholesky_free[p]])
_block_cpc(x, block::BRM.R2D2AdaptiveCenteringBlock, p) =
    isempty(block.cholesky_cpc) ? tanh(x[block.cholesky_free[p]]) :
        convert(eltype(x), block.cholesky_cpc[p])

# The kind is a type parameter so Enzyme sees only the accessor branch it is
# differentiating.  Every pair still has the same concrete
# `(location, log_scale)` argument-tuple type, keeping WarmupHMC's pair vector
# concrete.
struct BRMAdaptiveCenteringArgument{KIND,N} <: Function
    state::BRMAdaptiveCenteringState{N}
    pair_number::Int
end

function BRMAdaptiveCenteringArgument(
    state::BRMAdaptiveCenteringState{N}, pair_number, kind::Symbol,
) where {N}
    kind in (:location, :log_scale) || error(
        "unknown BRM adaptive-centering argument kind $kind",
    )
    BRMAdaptiveCenteringArgument{kind,N}(state, pair_number)
end

function _pair_location(state, pair_number)
    (state.pair_blocks[pair_number],
     state.pair_terms[pair_number],
     state.pair_groups[pair_number])
end

function _pair_index(state, pair_number)
    bi, k, g = _pair_location(state, pair_number)
    state.blocks[bi].effects[k, g]
end

# Return one entry of `C = diag(tau) * L` without allocating, for any K.  Stan's
# `cholesky_factor_corr` stick recursion is walked inline along row `i` instead
# of materializing `L`, so this is one nonrecursive loop over that row's free
# coordinates.  The spelling is bit-identical to the legacy `_adaptive_cholesky_corr`
# + `tau .* L` matrix graph, including its multiplication association: first
# `stick * z`, then `tau * L[i,j]`.  WarmupHMC's discrete window winners can
# change after a one-ulp gradient drift, so operation order — not just the
# limiting value — is the contract.
function _block_cholesky_entry(x, block, i, j)
    tau = _block_tau(x, block, i)
    stick = one(eltype(x))
    offset = (i - 1) * (i - 2) ÷ 2
    for l in 1:i-1
        z = _block_cpc(x, block, offset + l)
        l == j && return tau * (stick * z)
        stick *= sqrt(one(eltype(x)) - z * z)
    end
    j == i || throw(BoundsError((i, j)))
    tau * stick
end

# One arbitrary-K location accessor, differentiated by the ordinary Enzyme path.
#
# `u[k] = c[k] * sum(C[k,l] * z[l] for l < k) + C[k,k]^c[k] * z[k]` is triangular,
# so recovering `z` is a forward substitution: row `i` needs the `i-1`
# innovations below it.  That O(k) state is irreducible — the whole `K x K` `L`
# and `C` rebuild per pair is gone, replaced by the same inline stick walk as
# `_block_cholesky_entry`.  The innovations live in an `NTuple{N}` sized by the
# state's capacity `N`, not a heap vector: on Julia 1.10, Enzyme's reverse pass
# over a function that allocates a small work vector and reads it back is a
# use-after-free once another thread runs GC — `getindex` segfaults or the GC
# aborts with "probable corruption" (measured on Enzyme 0.13.210-0.13.216; Julia
# 1.12 and single-threaded runs are clean).  Concurrent completion/multi-chain
# workers hit exactly that.  `N` belongs to the state, so every pair keeps one
# concrete element type whatever the block widths.  The buffer cannot be cached
# on the state, because accessors run under `Enzyme.Const`.  Recursion is
# deliberately avoided: a recursive scalar callable here corrupted Julia's GC
# after ~1,750 sustained gradients (fixed by ade46ac/bd9ca2d/b5f4a86).
function _block_location(x, state::_BRMRanefCenteringState{N}, pair_number) where {N}
    bi, k, g = _pair_location(state, pair_number)
    T = eltype(x)
    k == 1 && return zero(T)
    block = state.blocks[bi]
    lookup = state.pair_lookup[bi]
    r = _pair_mixing_scale(x, state, pair_number)
    innovations = ntuple(_ -> zero(T), Val(N))
    m = zero(T)
    for i in 1:k
        m = zero(T)
        tau = _block_tau(x, block, i) * r
        stick = one(T)
        offset = (i - 1) * (i - 2) ÷ 2
        for l in 1:i-1
            z = _block_cpc(x, block, offset + l)
            m += (tau * (stick * z)) * innovations[l]
            stick *= sqrt(one(T) - z * z)
        end
        if i < k
            c = state.sources[lookup[i, g]]
            u = x[block.effects[i, g]]
            s = tau * stick
            innovations = Base.setindex(innovations, (u - c * m) / s^c, i)
        end
    end
    m
end

function (arg::BRMAdaptiveCenteringArgument{:location})(x)
    _block_location(x, arg.state, arg.pair_number)
end

function (arg::BRMAdaptiveCenteringArgument{:log_scale})(x)
    bi, k, _ = _pair_location(arg.state, arg.pair_number)
    block = arg.state.blocks[bi]
    log(_block_cholesky_entry(x, block, k, k))
end

struct BRMR2D2AdaptiveCenteringArgument{KIND,N} <: Function
    state::BRMR2D2AdaptiveCenteringState{N}
    pair_number::Int
end

function BRMR2D2AdaptiveCenteringArgument(
    state::BRMR2D2AdaptiveCenteringState{N}, pair_number, kind::Symbol,
) where {N}
    kind in (:location, :log_scale) || error(
        "unknown BRM R2D2 adaptive-centering argument kind $kind",
    )
    BRMR2D2AdaptiveCenteringArgument{kind,N}(state, pair_number)
end

function (arg::BRMR2D2AdaptiveCenteringArgument{:location})(x)
    _block_location(x, arg.state, arg.pair_number)
end

function (arg::BRMR2D2AdaptiveCenteringArgument{:log_scale})(x)
    bi, k, _ = _pair_location(arg.state, arg.pair_number)
    log(_block_cholesky_entry(x, arg.state.blocks[bi], k, k) *
        _pair_mixing_scale(x, arg.state, arg.pair_number))
end

function _sync_sources!(state, ir)
    length(ir.pairs) == length(state.sources) || throw(DimensionMismatch(
        "BRM adaptive-centering plan has $(length(state.sources)) cells but the " *
        "WarmupHMC reparametrizer has $(length(ir.pairs)) pairs",
    ))
    for (i, (idx, value)) in enumerate(ir.pairs)
        expected = _pair_index(state, i)
        idx == expected || throw(ArgumentError(
            "BRM adaptive-centering pair $i addresses raw coordinate $idx, " *
            "but the model metadata requires $expected; pair ordering changed",
        ))
        state.sources[i] = Float64(value.source.c)
    end
    ir
end

struct BRMAdaptiveCenteringFrame{T}
    source::Vector{T}
    location::Vector{T}
    scale::Vector{T}
    innovation::Vector{T}
    invariant_gradient::Vector{T}
end

function _prepare_frame(state::_BRMRanefCenteringState, ir, position, gradient)
    _sync_sources!(state, ir)
    _centering_frame(state, position, gradient)
end

# `ordered` is a `BRMOrderedCenteringReparametrization`, whose source frame
# orders every group (`_source_group_factor`); `nothing` is the declared order,
# whose frame is unchanged.
function _centering_frame(state::_BRMRanefCenteringState{N}, position, gradient,
                          ordered=nothing) where {N}
    T = promote_type(eltype(position), eltype(gradient), Float64)
    n = length(state.sources)
    source = T.(state.sources)
    location = Vector{T}(undef, n)
    scale = Vector{T}(undef, n)
    innovation = Vector{T}(undef, n)
    invariant_gradient = Vector{T}(undef, n)

    for (bi, block) in enumerate(state.blocks), g in 1:block.ranef.n_groups
        K = block.ranef.n_terms
        order = isnothing(ordered) ? (1:K) : view(ordered.orders[bi], :, g)
        C = isnothing(ordered) ? BRM._adaptive_block_cholesky(position, block, g) :
            _unpacked(_source_group_factor(position, ordered, bi, g), K)
        A = zeros(T, K, K)
        z = Vector{T}(undef, K)
        for j in 1:K
            k = order[j]
            p = state.pair_lookup[bi][k, g]
            c = source[p]
            m = zero(T)
            for l in 1:j-1
                m += C[j, l] * z[l]
                A[j, l] = c * C[j, l]
            end
            s = C[j, j]
            u = position[block.effects[k, g]]
            z[j] = (u - c * m) / s^c
            A[j, j] = s^c
            location[p] = m
            scale[p] = s
            innovation[p] = z[j]
        end
        gu = gradient[block.effects[order, g]]
        h = transpose(A) * gu
        for j in 1:K
            invariant_gradient[state.pair_lookup[bi][order[j], g]] = h[j]
        end
    end
    BRMAdaptiveCenteringFrame(
        source, location, scale, innovation, invariant_gradient,
    )
end

function _score_candidate(frame, pair_number, idx, value, candidate)
    1 <= pair_number <= length(frame.source) || throw(BoundsError(frame.source, pair_number))
    t = candidate.c
    c = frame.source[pair_number]
    m = frame.location[pair_number]
    s = frame.scale[pair_number]
    z = frame.innovation[pair_number]
    h = frame.invariant_gradient[pair_number]
    ((t - c) * log(s), t * m + s^t * z, h / s^t)
end

function _problem_unc_names(problem)
    hasproperty(problem, :model) || error(
        "BRM adaptive centering: `unc_names` was omitted, but problem type " *
        "$(typeof(problem)) has no `.model` BridgeStan handle. Pass the compiled " *
        "model's unconstrained parameter names explicitly.",
    )
    BRM.StanBlocks.BridgeStan.param_unc_names(getproperty(problem, :model))
end

function _adaptive_centering_reparametrizer(blocks)
    state = BRMAdaptiveCenteringState(blocks)
    pairs = [begin
        block = state.blocks[state.pair_blocks[p]]
        k = state.pair_terms[p]
        g = state.pair_groups[p]
        idx = block.effects[k, g]
        target = WarmupHMC.PartiallyCentered(block.target_c)
        source = WarmupHMC.PartiallyCentered(block.target_c)
        loc = BRMAdaptiveCenteringArgument(state, p, :location)
        log_scale = BRMAdaptiveCenteringArgument(state, p, :log_scale)
        idx => WarmupHMC.Reparametrization(target, source, loc, log_scale)
    end for p in eachindex(state.sources)]
    state, WarmupHMC.IndexedReparametrization(pairs)
end

# HSGP basis weights are conditionally independent given their spectral
# scales, so they use a scalar zero-location frame.  Keep this state and its
# callable distinct from `BRMAdaptiveCenteringState`: the latter's concrete
# type parameter selects the exact K<=2 Enzyme reverse pass for ordinary random
# effects, and HSGP support must not perturb that dispatch.
mutable struct BRMHSGPAdaptiveCenteringState
    blocks::Vector{BRM._HSGPAdaptiveCenteringBlock}
    pair_blocks::Vector{Int}
    pair_basis::Vector{Int}
    sources::Vector{Float64}
    effect_indices::BitSet
end

function BRMHSGPAdaptiveCenteringState(blocks)
    pair_blocks = Int[]
    pair_basis = Int[]
    sources = Float64[]
    for (bi, block) in enumerate(blocks), basis in eachindex(block.effects)
        push!(pair_blocks, bi)
        push!(pair_basis, basis)
        push!(sources, block.target_c[basis])
    end
    effect_indices = BitSet(Iterators.flatten(b.effects for b in blocks))
    BRMHSGPAdaptiveCenteringState(
        collect(blocks), pair_blocks, pair_basis, sources, effect_indices,
    )
end

struct BRMHSGPAdaptiveCenteringArgument{KIND} <: Function
    state::BRMHSGPAdaptiveCenteringState
    pair_number::Int
end

function BRMHSGPAdaptiveCenteringArgument(state, pair_number, kind::Symbol)
    kind in (:location, :log_scale) || error(
        "unknown BRM HSGP adaptive-centering argument kind $kind",
    )
    BRMHSGPAdaptiveCenteringArgument{kind}(state, pair_number)
end

function _hsgp_pair_location(state, pair_number)
    (state.pair_blocks[pair_number], state.pair_basis[pair_number])
end

function _hsgp_pair_index(state, pair_number)
    bi, basis = _hsgp_pair_location(state, pair_number)
    state.blocks[bi].effects[basis]
end

function (::BRMHSGPAdaptiveCenteringArgument{:location})(x)
    zero(eltype(x))
end

function (arg::BRMHSGPAdaptiveCenteringArgument{:log_scale})(x)
    bi, basis = _hsgp_pair_location(arg.state, arg.pair_number)
    BRM._adaptive_hsgp_log_scale(x, arg.state.blocks[bi], basis)
end

function _sync_sources!(state::BRMHSGPAdaptiveCenteringState, ir)
    length(ir.pairs) == length(state.sources) || throw(DimensionMismatch(
        "BRM HSGP adaptive-centering plan has $(length(state.sources)) basis " *
        "weights but the WarmupHMC reparametrizer has $(length(ir.pairs)) pairs",
    ))
    for (p, (idx, value)) in enumerate(ir.pairs)
        expected = _hsgp_pair_index(state, p)
        idx == expected || throw(ArgumentError(
            "BRM HSGP adaptive-centering pair $p addresses raw coordinate $idx, " *
            "but the model metadata requires $expected; pair ordering changed",
        ))
        state.sources[p] = Float64(value.source.c)
    end
    ir
end

function _prepare_frame(state::BRMHSGPAdaptiveCenteringState,
                        ir, position, gradient)
    _sync_sources!(state, ir)
    _centering_frame(state, position, gradient)
end

function _centering_frame(state::BRMHSGPAdaptiveCenteringState, position, gradient)
    T = promote_type(eltype(position), eltype(gradient), Float64)
    n = length(state.sources)
    source = T.(state.sources)
    location = zeros(T, n)
    scale = Vector{T}(undef, n)
    innovation = Vector{T}(undef, n)
    invariant_gradient = Vector{T}(undef, n)
    for p in eachindex(state.sources)
        bi, basis = _hsgp_pair_location(state, p)
        block = state.blocks[bi]
        idx = block.effects[basis]
        log_scale = BRM._adaptive_hsgp_log_scale(position, block, basis)
        s = exp(log_scale)
        c = source[p]
        # A floored scale is (numerically) the same at every observation, so
        # every candidate frame `s^t * z` is a fixed rescaling of the others:
        # the losses tie and rounding would choose the control. Off the floor
        # that control couples the weight to a log-scale up to ~150 larger, a
        # funnel no chain crosses (snag adaptive-centeri-eca5a77c). A floored
        # observation therefore scores every candidate in the non-centred frame
        # (scale 1), so a window spent on the floor ties exactly and keeps the
        # first candidate, `c = 0`. The transform still reads the floored `s`.
        scale[p] = log_scale > BRM._ADAPTIVE_HSGP_LOG_SCALE_EXACT ? s : one(s)
        innovation[p] = position[idx] / s^c
        invariant_gradient[p] = s^c * gradient[idx]
    end
    BRMAdaptiveCenteringFrame(
        source, location, scale, innovation, invariant_gradient,
    )
end

function _adaptive_hsgp_centering_reparametrizer(blocks)
    state = BRMHSGPAdaptiveCenteringState(blocks)
    pairs = [begin
        bi, basis = _hsgp_pair_location(state, p)
        block = state.blocks[bi]
        target = WarmupHMC.PartiallyCentered(block.target_c[basis])
        source = WarmupHMC.PartiallyCentered(block.target_c[basis])
        location = BRMHSGPAdaptiveCenteringArgument(state, p, :location)
        log_scale = BRMHSGPAdaptiveCenteringArgument(state, p, :log_scale)
        block.effects[basis] => WarmupHMC.Reparametrization(
            target, source, location, log_scale,
        )
    end for p in eachindex(state.sources)]
    state, WarmupHMC.IndexedReparametrization(pairs)
end

# Scalar cells with a constant location and a raw log-scale coordinate: exact
# totals (location = prior total, compiled at c = 1) and S2Z free contrasts
# (location 0, compiled at c = 0 or 1). Cells enumerate totals first, then S2Z
# contrasts, each in its family's cell order (`BRM._total_centering_cells`,
# `BRM._s2z_centering_cells`), which the post-hoc selectors share.
mutable struct BRMScalarCenteringState
    indices::Vector{Int}
    scales::Vector{Int}
    locations::Vector{Float64}
    sources::Vector{Float64}
end

struct BRMScalarCenteringArgument{KIND} <: Function
    state::BRMScalarCenteringState
    pair_number::Int
end
(arg::BRMScalarCenteringArgument{:location})(x) = arg.state.locations[arg.pair_number]
(arg::BRMScalarCenteringArgument{:log_scale})(x) = x[arg.state.scales[arg.pair_number]]

function _sync_sources!(state::BRMScalarCenteringState,ir)
    length(state.sources) == length(ir.pairs) || throw(DimensionMismatch("scalar centering pair count changed"))
    for (p,(index,value)) in enumerate(ir.pairs)
        index == state.indices[p] || throw(ArgumentError("scalar centering pair order changed"))
        state.sources[p] = value.source.c
    end
    ir
end

function _prepare_frame(state::BRMScalarCenteringState,ir,position,gradient)
    _sync_sources!(state,ir)
    _centering_frame(state,position,gradient)
end

function _centering_frame(state::BRMScalarCenteringState,position,gradient)
    source = copy(state.sources)
    location = copy(state.locations)
    scale,innovation,invariant_gradient = similar(source),similar(source),similar(source)
    for p in eachindex(source)
        scale[p] = exp(position[state.scales[p]])
        factor = scale[p]^source[p]
        innovation[p] = (position[state.indices[p]]-source[p]*location[p])/factor
        invariant_gradient[p] = factor*gradient[state.indices[p]]
    end
    BRMAdaptiveCenteringFrame(source,location,scale,innovation,invariant_gradient)
end

# Every scalar cell of `model`: sampled exact totals, then S2Z contrasts.
function _scalar_centering_cells(model,names)
    totals = BRM._total_centering_cells(model,names)
    s2z = BRM._s2z_centering_cells(model,names)
    (;indices=vcat(totals.indices,s2z.indices),scales=vcat(totals.scales,s2z.scales),
      locations=vcat(totals.locations,s2z.locations),targets=vcat(totals.targets,s2z.targets))
end

_adaptive_scalar_centering_reparametrizer(model,names) =
    _scalar_centering_reparametrizer(_scalar_centering_cells(model,names)...)

_scalar_centering_state(cells) =
    BRMScalarCenteringState(cells.indices,cells.scales,cells.locations,copy(cells.targets))

function _scalar_centering_reparametrizer(indices,scales,locations,targets)
    state = _scalar_centering_state((;indices,scales,locations,targets))
    pairs = [indices[p] => WarmupHMC.Reparametrization(
        WarmupHMC.PartiallyCentered(targets[p]),WarmupHMC.PartiallyCentered(targets[p]),
        BRMScalarCenteringArgument{:location}(state,p),
        BRMScalarCenteringArgument{:log_scale}(state,p)) for p in eachindex(indices)]
    state,WarmupHMC.IndexedReparametrization(pairs)
end

# Areal car cells (`BRM._adaptive_car_centering_blocks`): per block its `:icar`
# `phi` cells or its `:bym2` `theta` cells, one per area. A `theta` cell is
# centred on its area's total `intercept + a * phi`, read from coordinates no
# cell rewrites (`phi` stays the emitted standardized coordinate), so the cells
# are independent and map in any order. Like HSGP and cdar cells they keep their
# own state and argument types, away from the ordinary K<=2 Enzyme dispatch.
# Their accessors read only `cells`, a flat table of isbits
# `BRM._CARAdaptiveCell`s, and never `sources`. A first version that read the
# blocks from the joint wrapper's accessor aborted
# `test/adaptive_centering_threads.jl` with "GC error (probable corruption)" in
# its `joint` case, which has no car cell at all (Julia 1.10.12, Enzyme
# 0.13.216; base passed the same gate), and so did accessors that pass a cell to
# an outlined call: every one here is `@inline` (`BRM._CARAdaptiveCell`).
# `blocks`, `pair_blocks` and `pair_cells` serve only the controls report.
mutable struct BRMCARAdaptiveCenteringState
    blocks::Vector{BRM._CARAdaptiveCenteringBlock}
    pair_blocks::Vector{Int}
    pair_cells::Vector{Int}
    cells::Vector{BRM._CARAdaptiveCell}
    sources::Vector{Float64}
    effect_indices::BitSet
end

function BRMCARAdaptiveCenteringState(blocks)
    pair_blocks, pair_cells = Int[], Int[]
    cells = BRM._CARAdaptiveCell[]
    for (bi, block) in enumerate(blocks), cell in 1:BRM._adaptive_car_n_cells(block)
        push!(pair_blocks, bi)
        push!(pair_cells, cell)
        push!(cells, BRM._adaptive_car_cell(block, cell))
    end
    effect_indices = BitSet(c.index for c in cells)
    BRMCARAdaptiveCenteringState(collect(blocks), pair_blocks, pair_cells, cells,
                                 zeros(length(cells)), effect_indices)
end

_car_pair_location(state, p) = (state.pair_blocks[p], state.pair_cells[p])
_car_pair_index(state, p) = state.cells[p].index
@inline _car_log_scale(x, state, p) = BRM._adaptive_car_log_scale(x, state.cells[p])
@inline _car_location(x, state, p) = BRM._adaptive_car_location(x, state.cells[p])

struct BRMCARAdaptiveCenteringArgument{KIND} <: Function
    state::BRMCARAdaptiveCenteringState
    pair_number::Int
end

function BRMCARAdaptiveCenteringArgument(state, pair_number, kind::Symbol)
    kind in (:location, :log_scale) || error(
        "unknown BRM car adaptive-centering argument kind $kind",
    )
    BRMCARAdaptiveCenteringArgument{kind}(state, pair_number)
end

(arg::BRMCARAdaptiveCenteringArgument{:location})(x) =
    _car_location(x, arg.state, arg.pair_number)
(arg::BRMCARAdaptiveCenteringArgument{:log_scale})(x) =
    _car_log_scale(x, arg.state, arg.pair_number)

function _sync_sources!(state::BRMCARAdaptiveCenteringState, ir)
    length(ir.pairs) == length(state.sources) || throw(DimensionMismatch(
        "BRM car adaptive-centering plan has $(length(state.sources)) cells but " *
        "the WarmupHMC reparametrizer has $(length(ir.pairs)) pairs",
    ))
    for (p, (idx, value)) in enumerate(ir.pairs)
        expected = _car_pair_index(state, p)
        idx == expected || throw(ArgumentError(
            "BRM car adaptive-centering pair $p addresses raw coordinate $idx, " *
            "but the model metadata requires $expected; pair ordering changed",
        ))
        state.sources[p] = Float64(value.source.c)
    end
    ir
end

function _prepare_frame(state::BRMCARAdaptiveCenteringState, ir, position, gradient)
    _sync_sources!(state, ir)
    _centering_frame(state, position, gradient)
end

# Each cell's frame `x = c * m + s^c * z`: its location `m` and scale `s` read
# coordinates no cell rewrites, so the innovation gradient is `s^c * g`.
function _centering_frame(state::BRMCARAdaptiveCenteringState, position, gradient)
    T = promote_type(eltype(position), eltype(gradient), Float64)
    n = length(state.sources)
    source = T.(state.sources)
    location = Vector{T}(undef, n)
    scale = Vector{T}(undef, n)
    innovation = Vector{T}(undef, n)
    invariant_gradient = Vector{T}(undef, n)
    for p in 1:n
        idx = _car_pair_index(state, p)
        c = source[p]
        s = exp(_car_log_scale(position, state, p))
        m = convert(T, _car_location(position, state, p))
        scale[p] = s
        location[p] = m
        innovation[p] = (position[idx] - c * m) / s^c
        invariant_gradient[p] = s^c * gradient[idx]
    end
    BRMAdaptiveCenteringFrame(
        source, location, scale, innovation, invariant_gradient,
    )
end

function _adaptive_car_centering_reparametrizer(blocks)
    state = BRMCARAdaptiveCenteringState(blocks)
    pairs = [_car_pair_index(state, p) => WarmupHMC.Reparametrization(
                 WarmupHMC.PartiallyCentered(0.0), WarmupHMC.PartiallyCentered(0.0),
                 BRMCARAdaptiveCenteringArgument(state, p, :location),
                 BRMCARAdaptiveCenteringArgument(state, p, :log_scale))
             for p in eachindex(state.sources)]
    state, WarmupHMC.IndexedReparametrization(pairs)
end

# Joint ordinary + R2D2 + HSGP + scalar (totals/S2Z) + car adaptation.
# `IndexedReparametrization` requires one concrete pair element type, so the
# families cannot contribute their own accessor closures to one vector. Every
# joint pair therefore carries the same `BRMJointAdaptiveCenteringArgument`,
# which selects by pair-number range: pairs `1:n_ranef` are ordinary cells, the
# next `n_r2d2` are R2D2 cells, then `n_hsgp` HSGP basis-weight cells, then
# `n_scalar` scalar total/S2Z cells, and the rest are car cells. A model without
# R2D2 blocks has `n_r2d2 = 0` and one without scalar or car cells has none, so
# each keeps the historical ordinary-then-HSGP pair order.
#
# Ordinary and R2D2 cells share one derived-scale state: an ordinary block
# joins it in derived-scale form (`BRM._adaptive_derived_block`), whose `tau` is
# `exp(0.0 + 1.0 * x[i])`, bit-identical to `exp(x[i])`. On Julia 1.10, a
# differentiated joint accessor that reaches two `_block_location`
# instantiations (ordinary and R2D2) corrupts Julia's GC under Enzyme's reverse
# pass once another thread runs GC — even when the R2D2 branch never executes —
# while one instantiation is clean (see `_block_location`). HSGP and car cells
# keep their scalar accessors.
struct BRMJointAdaptiveCenteringState{N}
    ranef::BRMR2D2AdaptiveCenteringState{N}
    hsgp::BRMHSGPAdaptiveCenteringState
    scalar::BRMScalarCenteringState
    car::BRMCARAdaptiveCenteringState
    n_ranef::Int
    n_r2d2::Int
    n_hsgp::Int
    n_scalar::Int
end

struct BRMJointAdaptiveCenteringArgument{KIND,N} <: Function
    state::BRMJointAdaptiveCenteringState{N}
    pair_number::Int
end

function BRMJointAdaptiveCenteringArgument(
        state::BRMJointAdaptiveCenteringState{N}, pair_number, kind::Symbol,
    ) where {N}
    kind in (:location, :log_scale) || error(
        "unknown BRM joint adaptive-centering argument kind $kind",
    )
    BRMJointAdaptiveCenteringArgument{kind,N}(state, pair_number)
end

function (arg::BRMJointAdaptiveCenteringArgument{:location})(x)
    p = arg.pair_number
    s = arg.state
    p <= s.n_ranef + s.n_r2d2 && return _block_location(x, s.ranef, p)
    q = p - s.n_ranef - s.n_r2d2 - s.n_hsgp
    q > s.n_scalar && return _car_location(x, s.car, q - s.n_scalar)
    q > 0 && return convert(eltype(x), s.scalar.locations[q])
    zero(eltype(x))
end

function (arg::BRMJointAdaptiveCenteringArgument{:log_scale})(x)
    p = arg.pair_number
    s = arg.state
    if p <= s.n_ranef + s.n_r2d2
        bi, k, _ = _pair_location(s.ranef, p)
        log(_block_cholesky_entry(x, s.ranef.blocks[bi], k, k) *
            _pair_mixing_scale(x, s.ranef, p))
    elseif p <= s.n_ranef + s.n_r2d2 + s.n_hsgp
        bi, basis = _hsgp_pair_location(s.hsgp, p - s.n_ranef - s.n_r2d2)
        BRM._adaptive_hsgp_log_scale(x, s.hsgp.blocks[bi], basis)
    elseif p <= s.n_ranef + s.n_r2d2 + s.n_hsgp + s.n_scalar
        x[s.scalar.scales[p - s.n_ranef - s.n_r2d2 - s.n_hsgp]]
    else
        _car_log_scale(x, s.car, p - s.n_ranef - s.n_r2d2 - s.n_hsgp - s.n_scalar)
    end
end

# The family state owning joint pair `i`, and the pair's number within it.
function _joint_pair_family(state::BRMJointAdaptiveCenteringState, i)
    i <= state.n_ranef + state.n_r2d2 && return (:ranef, i)
    i -= state.n_ranef + state.n_r2d2
    i <= state.n_hsgp && return (:hsgp, i)
    i -= state.n_hsgp
    i <= state.n_scalar && return (:scalar, i)
    (:car, i - state.n_scalar)
end

_joint_pair_index(state, family, j) =
    family === :hsgp ? _hsgp_pair_index(state.hsgp, j) :
    family === :scalar ? state.scalar.indices[j] :
    family === :car ? _car_pair_index(state.car, j) : _pair_index(state.ranef, j)

function _sync_sources!(state::BRMJointAdaptiveCenteringState, ir)
    n_cells = state.n_ranef + state.n_r2d2 + state.n_hsgp + state.n_scalar +
              length(state.car.sources)
    length(ir.pairs) == n_cells || throw(DimensionMismatch(
        "BRM joint adaptive-centering plan has $n_cells cells but the " *
        "WarmupHMC reparametrizer has $(length(ir.pairs)) pairs",
    ))
    for (i, (idx, value)) in enumerate(ir.pairs)
        family, j = _joint_pair_family(state, i)
        expected = _joint_pair_index(state, family, j)
        idx == expected || throw(ArgumentError(
            "BRM joint adaptive-centering pair $i addresses raw coordinate $idx, " *
            "but the model metadata requires $expected; pair ordering changed",
        ))
        getfield(state, family).sources[j] = Float64(value.source.c)
    end
    ir
end

function _prepare_frame(state::BRMJointAdaptiveCenteringState, ir, position, gradient)
    _sync_sources!(state, ir)
    _centering_frame(state, position, gradient)
end

function _centering_frame(state::BRMJointAdaptiveCenteringState, position, gradient,
                          ordered=nothing)
    frames = (_centering_frame(state.ranef, position, gradient, ordered),
              _centering_frame(state.hsgp, position, gradient),
              _centering_frame(state.scalar, position, gradient),
              _centering_frame(state.car, position, gradient))
    BRMAdaptiveCenteringFrame(
        (reduce(vcat, getfield.(frames, f)) for f in fieldnames(BRMAdaptiveCenteringFrame))...,
    )
end

const _NO_SCALAR_CELLS = (; indices=Int[], scales=Int[], locations=Float64[],
                           targets=Float64[])

const _NO_CAR_BLOCKS = BRM._CARAdaptiveCenteringBlock[]

# The joint state of ordinary, R2D2, HSGP, scalar and car cells, after refusing
# every coordinate overlap that would make the transform inexact. Shared by the
# per-pair wrapper below and the learned-order wrapper
# (`_adaptive_ordered_centering_reparametrizer`).
function _joint_centering_state(blocks, hsgp_blocks, r2d2_blocks, scalar_cells,
                                car_blocks=_NO_CAR_BLOCKS)
    hsgp_state = BRMHSGPAdaptiveCenteringState(hsgp_blocks)
    scalar_state = _scalar_centering_state(scalar_cells)
    car_state = BRMCARAdaptiveCenteringState(car_blocks)
    block_effects(bs) = BitSet(Iterators.flatten(vec(b.effects) for b in bs))
    effects = (block_effects(blocks), block_effects(r2d2_blocks),
               hsgp_state.effect_indices, BitSet(scalar_state.indices),
               car_state.effect_indices)
    sum(length, effects) == length(union(effects...)) || error(
        "BRM joint adaptive centering: ordinary, R2D2, HSGP, total/S2Z and car " *
        "cells claim overlapping unconstrained coordinates; refusing an " *
        "ambiguous transform.",
    )
    for block in car_blocks
        reads = filter(!=(0), vcat(block.sigma, block.rho, block.intercept,
                                   block.bym2 ? block.structured : Int[]))
        isdisjoint(reads, union(effects...)) || error(
            "BRM joint adaptive centering: car `$(block.term)` reads its scale, " *
            "intercept or `phi` from a coordinate another cell rewrites; refusing " *
            "an inexact transform.")
    end
    for block in r2d2_blocks
        read = BRM._adaptive_read_coordinates(block)
        isdisjoint(read, car_state.effect_indices) ||
            error("BRM joint adaptive centering: block " *
                  "`$(block.ranef.binding)` derives its scale from a car field " *
                  "coordinate this transform rewrites; refusing an inexact " *
                  "transform.")
        isdisjoint(read, hsgp_state.effect_indices) ||
            error("BRM joint adaptive centering: block " *
                  "`$(block.ranef.binding)` derives its scale from an HSGP " *
                  "basis weight this transform rewrites; refusing an inexact " *
                  "transform.")
        isdisjoint(read, scalar_state.indices) ||
            error("BRM joint adaptive centering: block " *
                  "`$(block.ranef.binding)` derives its scale from a total/S2Z " *
                  "coordinate this transform rewrites; refusing an inexact " *
                  "transform.")
    end
    isdisjoint(scalar_state.scales, union(effects...)) || error(
        "BRM joint adaptive centering: a total/S2Z scale coordinate is rewritten " *
        "by another cell; refusing an inexact transform.")
    # Block order fixes pair order, which is load-bearing across
    # checkpoint/resume (positional restore): ordinary cells first, then R2D2
    # cells, each in the family's own block/group/term order, then HSGP
    # basis-weight cells, then scalar total and S2Z cells
    # (`_scalar_centering_cells` order), then car cells (per block, in
    # `_adaptive_car_cell` order).
    ranef_state = BRMR2D2AdaptiveCenteringState(vcat(
        BRM.R2D2AdaptiveCenteringBlock[BRM._adaptive_derived_block(b) for b in blocks],
        r2d2_blocks,
    ))
    n_ranef = sum(b -> b.ranef.n_terms * b.ranef.n_groups, blocks; init=0)
    n_r2d2 = length(ranef_state.sources) - n_ranef
    n_hsgp = length(hsgp_state.sources)
    BRMJointAdaptiveCenteringState(ranef_state, hsgp_state, scalar_state, car_state,
                                   n_ranef, n_r2d2, n_hsgp,
                                   length(scalar_state.sources))
end

# The compiled frame of every joint cell, in pair order. Car fields are always
# emitted standardized (`c=0`).
function _joint_targets(state::BRMJointAdaptiveCenteringState, scalar_cells)
    ranef = [state.ranef.blocks[state.ranef.pair_blocks[p]].target_c
             for p in eachindex(state.ranef.sources)]
    hsgp = [state.hsgp.blocks[bi].target_c[basis]
            for (bi, basis) in zip(state.hsgp.pair_blocks, state.hsgp.pair_basis)]
    vcat(ranef, hsgp, Float64.(scalar_cells.targets), zeros(length(state.car.sources)))
end

_joint_cell_index(state::BRMJointAdaptiveCenteringState, p) =
    _joint_pair_index(state, _joint_pair_family(state, p)...)

function _adaptive_joint_centering_reparametrizer(
        blocks, hsgp_blocks, r2d2_blocks=BRM.R2D2AdaptiveCenteringBlock[],
        scalar_cells=_NO_SCALAR_CELLS, car_blocks=_NO_CAR_BLOCKS)
    state = _joint_centering_state(blocks, hsgp_blocks, r2d2_blocks, scalar_cells,
                                   car_blocks)
    pairs = [_joint_cell_index(state, p) => WarmupHMC.Reparametrization(
                 WarmupHMC.PartiallyCentered(c), WarmupHMC.PartiallyCentered(c),
                 BRMJointAdaptiveCenteringArgument(state, p, :location),
                 BRMJointAdaptiveCenteringArgument(state, p, :log_scale))
             for (p, c) in enumerate(_joint_targets(state, scalar_cells))]
    state, WarmupHMC.IndexedReparametrization(pairs)
end

# ---- Learned triangular order (`ranef_order = :learned`) --------------------
#
# A correlated block's triangular map centres margin `k` on its conditional mean
# given the margins before it, and the per-pair wrappers above take that order
# from the formula declaration order. A data-poor margin declared before
# data-rich ones can then never be "non-centred given the data-rich margins",
# for any controls (snag correlated-kreis-15564811). The learned order re-derives
# each group's order from that group's controls (`_learned_order`): descending
# `c`, ties in declaration order. Centred margins come first; every other margin
# is centred to its own degree around its conditional mean given the margins
# ahead of it.
#
# The conditioning order is a permutation `P` of the block covariance
# `Sigma = C * C'`, `C = diag(tau) * L`: position `j` holds margin `P[j]`, and
# `Cp = cholesky(P * Sigma * P')` replaces `C`. Source coordinate `x[P[j]]` is
# `c * m[j] + Cp[j,j]^c * w[j]` with `m[j] = sum(Cp[j,i] * w[i] for i < j)`; the
# effects are `b[P[j]] = m[j] + Cp[j,j] * w[j]`, and the compiled coordinates are
# the declared-order frame of `b` at the block's compiled `target_c`. For a block
# compiled non-centred, source -> compiled is then a rotation rather than
# per-coordinate pairs, so this wrapper is a custom reparametrization that a
# `WindowSelectionPlan` adapts (`_ordered_select!`); HSGP, total, S2Z and car
# cells keep their per-cell maps inside it. Its controls are the joint wrapper's cell
# controls in the joint pair order, and the order is a pure function of them, so
# checkpoints carry nothing else.
#
# The block's own correlation coordinates are reordered too. Stan's
# `cholesky_factor_corr` parametrizes `L` by canonical partial correlations
# taken in declaration order, and that geometry alone left a data-poor margin
# declared first mixing about 4x worse with identical effect frames (synthetic
# `rho_sr` bulk ESS 120 against 479). So the block has one order of its own,
# its margins by mean control (`_derive_orders!`), and the source frame's
# Cholesky coordinates are the partial correlations of the correlation matrix
# in that order. The compiled ones are recovered through the declared-order
# factor, and the Jacobian is the difference of the two correlation
# transforms (`_correlation_logjac`). Each group's ties follow the block order,
# so most groups share it and need no rotations. At uniform controls every order
# is the declared one, so the wrapper starts in the compiled frame.
#
# The map runs under Enzyme on every gradient, so it follows `_block_location`'s
# Julia 1.10 rules: per-block factors live in `NTuple`s sized by the state's
# capacity, never in heap work vectors, and the random-effect loops read only
# flat integer arrays.
struct BRMOrderedCenteringReparametrization{N} <: WarmupHMC.AbstractReparametrization
    state::BRMJointAdaptiveCenteringState{N}
    # The compiled frame of every cell, in joint pair order.
    targets::Vector{Float64}
    # Per ranef block: `orders[bi][j, g]` is the margin at position `j` of group
    # `g`. The distinct orders are `perms[bi][:, q]`, and the groups sharing
    # order `q` are `groups[bi][starts[bi][q]:starts[bi][q+1]-1]`, so the
    # permuted factor is built once per distinct order.
    orders::Vector{Matrix{Int}}
    perms::Vector{Matrix{Int}}
    starts::Vector{Vector{Int}}
    groups::Vector{Vector{Int}}
    # Per ranef block: its own order (margins by mean control), each margin's
    # position in it, and, per distinct group order `q`, that order's rows of
    # the block-order factor (`rows[bi][j, q]`) and its kind: `0` the declared
    # order, `1` the block order, `2` any other.
    block_orders::Vector{Vector{Int}}
    positions::Vector{Vector{Int}}
    rows::Vector{Matrix{Int}}
    kinds::Vector{Vector{Int}}
    # `1`: source -> compiled; `-1`: compiled -> source.
    direction::Int
end

# `Base.setindex(t, v, i)` with a runtime `i`, as a branch-free select over every
# entry. Base's recursive version falls off a cliff past 32 entries (measured:
# 1.9 us and an 80 KB heap allocation per write at 49 entries, against 6 ns and
# none here), and a heap allocation inside an Enzyme-differentiated map is the
# Julia 1.10 GC hazard `_block_location` describes. The square Givens buffer
# reaches 49 entries at `K = 7`, the packed factor 36 at `K = 8`.
@inline _tset(t::NTuple{M,T}, v, i) where {M,T} = ntuple(k -> ifelse(k == i, T(v), t[k]), Val(M))

# Packed lower-triangular storage: entry `(i, j)`, `j <= i`.
@inline _tri(i, j) = (i * (i - 1)) ÷ 2 + j
# Row-major `n x n` storage.
@inline _sq(i, j, n) = (i - 1) * n + j

_packed_length(::Val{K}) where {K} = Val((K * (K + 1)) ÷ 2)

# `C = diag(tau[order]) * L`, packed, where `L` is built from the block's
# Cholesky coordinates and row `i` holds margin `order[i]`; the declared order
# gives `diag(tau) * L` with the same operations and association as
# `_block_cholesky_entry`.
function _packed_block_cholesky(x, block, order, ::Val{KMAX}) where {KMAX}
    T = eltype(x)
    C = ntuple(_ -> zero(T), _packed_length(Val(KMAX)))
    for i in 1:block.ranef.n_terms
        tau = _block_tau(x, block, order[i])
        stick = one(T)
        offset = (i - 1) * (i - 2) ÷ 2
        for l in 1:i-1
            z = _block_cpc(x, block, offset + l)
            C = _tset(C, tau * (stick * z), _tri(i, l))
            stick *= sqrt(one(T) - z * z)
        end
        C = _tset(C, tau * stick, _tri(i, i))
    end
    C
end

# `cholesky(P * C * C' * P')`, packed, for the order `order[1:K]`: the LQ factor
# of the row-permuted `P * C`, by Givens rotations of its columns. Each diagonal
# is the norm of a residual row, never a difference of squares, so a nearly
# singular correlation cannot round a conditional variance below zero (the
# difference-of-squares Cholesky did, mid-warm-up). `A` is `KMAX x KMAX`,
# row-major.
function _permuted_cholesky(C::NTuple{M,T}, order, K, ::Val{KMAX}) where {M,T,KMAX}
    A = ntuple(_ -> zero(T), Val(KMAX * KMAX))
    for j in 1:K
        for l in 1:order[j]
            A = _tset(A, C[_tri(order[j], l)], _sq(j, l, KMAX))
        end
    end
    for j in 1:K
        for l in j+1:K
            a, b = A[_sq(j, j, KMAX)], A[_sq(j, l, KMAX)]
            r = sqrt(a * a + b * b)
            iszero(r) && continue
            c, s = a / r, b / r
            for i in j:K
                u, v = A[_sq(i, j, KMAX)], A[_sq(i, l, KMAX)]
                A = _tset(A, c * u + s * v, _sq(i, j, KMAX))
                A = _tset(A, c * v - s * u, _sq(i, l, KMAX))
            end
        end
        if A[_sq(j, j, KMAX)] < zero(T)
            for i in j:K
                A = _tset(A, -A[_sq(i, j, KMAX)], _sq(i, j, KMAX))
            end
        end
    end
    Cp = ntuple(_ -> zero(T), Val(M))
    for j in 1:K, i in 1:j
        Cp = _tset(Cp, A[_sq(j, i, KMAX)], _tri(j, i))
    end
    Cp
end

_unpacked(C, K) = [j <= i ? C[_tri(i, j)] : zero(eltype(C)) for i in 1:K, j in 1:K]

# A packed factor times the scalar `r`: a Student-t level's `sqrt(w[g]) * C`,
# and its permuted factor, since `cholesky(P * (r^2 * Sigma) * P') =
# r * cholesky(P * Sigma * P')`.
_tscale(C::NTuple, r) = map(v -> r * v, C)

# Stan's `cholesky_factor_corr` log-Jacobian from its unconstrained coordinates
# to the correlation matrix, at the packed factor `F` (rows may carry scales):
# `sum(log(1 - z^2)) + sum(log(stick)) + sum((K - i) * log(L[i,i]))`, the tanh,
# stick-breaking and `L -> L * L'` terms. With `n[i,l]` the norm of row `i` from
# column `l` on, `1 - z[i,l]^2 = (n[i,l+1] / n[i,l])^2`, `stick[i,l] =
# n[i,l] / n[i,1]` and `L[i,i] = n[i,i] / n[i,1]`; everything is a ratio of row
# norms, so no difference of squares can cancel.
function _correlation_logjac(F, K)
    T = eltype(F)
    total = zero(T)
    for i in 2:K
        suffix = zero(T)
        log_ii = zero(T)
        log_next = zero(T)
        acc = zero(T)
        for l in i:-1:1
            suffix += F[_tri(i, l)] * F[_tri(i, l)]
            log_n = log(suffix) / 2
            if l == i
                log_ii = log_n
            else
                acc += 2 * log_next - log_n
            end
            log_next = log_n
        end
        total += acc - (i - 1) * log_next + (K - i) * (log_ii - log_next)
    end
    total
end

# Writes the unconstrained Cholesky coordinates of the packed factor `F`,
# `atanh(z)` with `z[i,l] = F[i,l] / n[i,l]` (`n` as above), spelled
# `sign(F) * log((n[i,l] + |F|) / n[i,l+1])`: rounding `z` to `±1` first would
# turn a nearly singular correlation into an infinite coordinate.
function _write_cholesky_coordinates!(y, block, F, K)
    for i in 2:K
        offset = (i - 1) * (i - 2) ÷ 2
        suffix = F[_tri(i, i)] * F[_tri(i, i)]
        for l in i-1:-1:1
            f = F[_tri(i, l)]
            rest = sqrt(suffix)
            suffix += f * f
            y[block.cholesky_free[offset + l]] = copysign(log((sqrt(suffix) + abs(f)) / rest), f)
        end
    end
    y
end

function _is_declared_order(order, K)
    for j in 1:K
        order[j] == j || return false
    end
    true
end

# Margins in descending control; ties by `tiebreak` (default: declaration
# order).
_learned_order(controls, tiebreak=eachindex(controls)) =
    sortperm(collect(eachindex(controls)); by=k -> (-controls[k], tiebreak[k]))

_known_correlation(::BRM.AdaptiveCenteringBlock) = false
_known_correlation(block::BRM.R2D2AdaptiveCenteringBlock) = !isempty(block.cholesky_known)

function _derive_orders!(ir::BRMOrderedCenteringReparametrization)
    state = ir.state.ranef
    for (bi, block) in enumerate(state.blocks)
        K, G = block.ranef.n_terms, block.ranef.n_groups
        lookup = state.pair_lookup[bi]
        controls = [state.sources[lookup[k, g]] for k in 1:K, g in 1:G]
        # A known correlation has no Cholesky coordinates to reorder, so its
        # block keeps the declared order; its groups still learn their own.
        block_order = _known_correlation(block) ? collect(1:K) :
            _learned_order(vec(sum(controls; dims=2)) ./ G)
        position = invperm(block_order)
        order = reduce(hcat, [_learned_order(controls[:, g], position) for g in 1:G];
                       init=zeros(Int, K, 0))
        members = Dict{Vector{Int},Vector{Int}}()
        distinct = Vector{Int}[]
        for g in 1:G
            key = order[:, g]
            haskey(members, key) || push!(distinct, key)
            push!(get!(members, key, Int[]), g)
        end
        ir.orders[bi] = order
        ir.perms[bi] = reduce(hcat, distinct; init=zeros(Int, K, 0))
        ir.starts[bi] = cumsum(vcat(1, [length(members[key]) for key in distinct]))
        ir.groups[bi] = reduce(vcat, (members[key] for key in distinct); init=Int[])
        ir.block_orders[bi] = block_order
        ir.positions[bi] = position
        ir.rows[bi] = reduce(hcat, [position[key] for key in distinct];
                             init=zeros(Int, K, 0))
        ir.kinds[bi] = [key == 1:K ? 0 : key == block_order ? 1 : 2 for key in distinct]
    end
    ir
end

function _adaptive_ordered_centering_reparametrizer(
        blocks, hsgp_blocks, r2d2_blocks=BRM.R2D2AdaptiveCenteringBlock[],
        scalar_cells=_NO_SCALAR_CELLS, car_blocks=_NO_CAR_BLOCKS)
    state = _joint_centering_state(blocks, hsgp_blocks, r2d2_blocks, scalar_cells,
                                   car_blocks)
    n = length(state.ranef.blocks)
    matrices() = [Matrix{Int}(undef, 0, 0) for _ in 1:n]
    vectors() = [Int[] for _ in 1:n]
    ir = BRMOrderedCenteringReparametrization(state, _joint_targets(state, scalar_cells),
        matrices(), matrices(), vectors(), vectors(), vectors(), vectors(), matrices(),
        vectors(), 1)
    state, _derive_orders!(ir)
end

# One group of one block, source -> compiled. Every source value is read before
# any output is written, so `y === x` is safe.
function _ordered_group_forward!(y, x, state, block, lookup, g, C, Cp, perms, q,
                                 declared, ::Val{KMAX}) where {KMAX}
    T = eltype(x)
    K = block.ranef.n_terms
    t = block.target_c
    u = ntuple(_ -> zero(T), Val(KMAX))
    for k in 1:K
        u = _tset(u, x[block.effects[k, g]], k)
    end
    w = ntuple(_ -> zero(T), Val(KMAX))
    b = ntuple(_ -> zero(T), Val(KMAX))
    ljac = zero(T)
    for j in 1:K
        a = perms[j, q]
        c = state.sources[lookup[a, g]]
        m = zero(T)
        for i in 1:j-1
            m += Cp[_tri(j, i)] * w[i]
        end
        s = Cp[_tri(j, j)]
        w = _tset(w, (u[a] - c * m) / s^c, j)
        if declared
            # WarmupHMC's per-pair spelling (`warmuphmc-use` §5c), so the
            # declared order reproduces the per-pair wrapper.
            log_scale = log(s)
            y[block.effects[a, g]] = t * m + (u[a] - c * m) * exp(log_scale * (t - c))
            ljac += log_scale * (t - c)
        else
            b = _tset(b, m + s * w[j], a)
            ljac += (1 - c) * log(s)
        end
    end
    declared && return ljac
    # Effects -> the compiled declared-order frame at `target_c`.
    z = ntuple(_ -> zero(T), Val(KMAX))
    for k in 1:K
        m = zero(T)
        for l in 1:k-1
            m += C[_tri(k, l)] * z[l]
        end
        s = C[_tri(k, k)]
        z = _tset(z, (b[k] - m) / s, k)
        y[block.effects[k, g]] = t * m + s^t * z[k]
        ljac += (t - 1) * log(s)
    end
    ljac
end

# One group of one block, compiled -> source; the exact inverse of
# `_ordered_group_forward!`.
function _ordered_group_inverse!(y, x, state, block, lookup, g, C, Cp, perms, q,
                                 declared, ::Val{KMAX}) where {KMAX}
    T = eltype(x)
    K = block.ranef.n_terms
    t = block.target_c
    v = ntuple(_ -> zero(T), Val(KMAX))
    for k in 1:K
        v = _tset(v, x[block.effects[k, g]], k)
    end
    # The compiled frame -> innovations `z` and effects `b`, declared order.
    z = ntuple(_ -> zero(T), Val(KMAX))
    b = ntuple(_ -> zero(T), Val(KMAX))
    ljac = zero(T)
    for k in 1:K
        m = zero(T)
        for l in 1:k-1
            m += C[_tri(k, l)] * z[l]
        end
        s = C[_tri(k, k)]
        z = _tset(z, (v[k] - t * m) / s^t, k)
        if declared
            c = state.sources[lookup[k, g]]
            log_scale = log(s)
            y[block.effects[k, g]] = c * m + (v[k] - t * m) * exp(log_scale * (c - t))
            ljac += log_scale * (c - t)
        else
            b = _tset(b, m + s * z[k], k)
            ljac += (1 - t) * log(s)
        end
    end
    declared && return ljac
    w = ntuple(_ -> zero(T), Val(KMAX))
    for j in 1:K
        a = perms[j, q]
        c = state.sources[lookup[a, g]]
        m = zero(T)
        for i in 1:j-1
            m += Cp[_tri(j, i)] * w[i]
        end
        s = Cp[_tri(j, j)]
        w = _tset(w, (b[a] - m) / s, j)
        y[block.effects[a, g]] = c * m + s^c * w[j]
        ljac += (c - 1) * log(s)
    end
    ljac
end

function _ordered_block_map!(y, x, ir::BRMOrderedCenteringReparametrization{N},
                             bi) where {N}
    state = ir.state.ranef
    block = state.blocks[bi]
    K = block.ranef.n_terms
    lookup = state.pair_lookup[bi]
    perms = ir.perms[bi]
    starts = ir.starts[bi]
    groups = ir.groups[bi]
    rows = ir.rows[bi]
    kinds = ir.kinds[bi]
    block_order = ir.block_orders[bi]
    reordered = !_is_declared_order(block_order, K)
    # `C`: the declared-order factor of the compiled frame; `Cs`: the block-order
    # factor of the source frame, whose Cholesky coordinates are permuted.
    if ir.direction > 0
        Cs = _packed_block_cholesky(x, block, block_order, Val(N + 1))
        C = reordered ? _permuted_cholesky(Cs, ir.positions[bi], K, Val(N + 1)) : Cs
    else
        C = _packed_block_cholesky(x, block, Base.OneTo(K), Val(N + 1))
        Cs = reordered ? _permuted_cholesky(C, block_order, K, Val(N + 1)) : C
    end
    ljac = zero(eltype(x))
    if reordered
        from, to = ir.direction > 0 ? (Cs, C) : (C, Cs)
        _write_cholesky_coordinates!(y, block, to, K)
        ljac += _correlation_logjac(from, K) - _correlation_logjac(to, K)
    end
    for q in axes(perms, 2)
        declared = kinds[q] == 0
        Cp = declared ? C : kinds[q] == 1 ? Cs :
            _permuted_cholesky(Cs, view(rows, :, q), K, Val(N + 1))
        for r in starts[q]:starts[q + 1] - 1
            g = groups[r]
            # A Student-t level's factors carry its `sqrt(w[g])`; the block's
            # correlation coordinates and their Jacobian above are scale-free.
            # The scale comes from the flat `pair_mixing` (`_pair_mixing_scale`).
            p = lookup[1, g]
            Cg, Cpg = state.pair_mixing[p] == 0 ? (C, Cp) :
                (_tscale(C, _pair_mixing_scale(x, state, p)),
                 _tscale(Cp, _pair_mixing_scale(x, state, p)))
            ljac += ir.direction > 0 ?
                _ordered_group_forward!(y, x, state, block, lookup, g, Cg, Cpg,
                                        perms, q, declared, Val(N + 1)) :
                _ordered_group_inverse!(y, x, state, block, lookup, g, Cg, Cpg,
                                        perms, q, declared, Val(N + 1))
        end
    end
    ljac
end

# WarmupHMC's per-pair partially centred map (`warmuphmc-use` §5c),
# `y = t*loc + (x - c*loc) * exp(log_scale * (t - c))`; its inverse swaps `t`
# and `c`.
function _cell_map!(y, x, idx, location, log_scale, target, source, direction)
    from, to = direction > 0 ? (source, target) : (target, source)
    y[idx] = to * location + (x[idx] - from * location) * exp(log_scale * (to - from))
    log_scale * (to - from)
end

function WarmupHMC.with_logabsdet_jacobian!(y::AbstractVector,
        ir::BRMOrderedCenteringReparametrization, x::AbstractVector)
    s = ir.state
    ljac = zero(eltype(x))
    for bi in eachindex(s.ranef.blocks)
        ljac += _ordered_block_map!(y, x, ir, bi)
    end
    offset = s.n_ranef + s.n_r2d2
    for p in eachindex(s.hsgp.sources)
        bi, basis = _hsgp_pair_location(s.hsgp, p)
        block = s.hsgp.blocks[bi]
        ljac += _cell_map!(y, x, block.effects[basis], zero(eltype(x)),
            BRM._adaptive_hsgp_log_scale(x, block, basis), ir.targets[offset + p],
            s.hsgp.sources[p], ir.direction)
    end
    offset += s.n_hsgp
    for p in eachindex(s.scalar.sources)
        ljac += _cell_map!(y, x, s.scalar.indices[p], s.scalar.locations[p],
            x[s.scalar.scales[p]], ir.targets[offset + p], s.scalar.sources[p],
            ir.direction)
    end
    offset += s.n_scalar
    ljac += _ordered_car_map!(y, x, s.car, ir.targets, offset, ir.direction)
    ljac, y
end

# Car cells inside the learned-order wrapper; `targets[offset + p]` is car pair
# `p`'s compiled frame. Every location and scale reads coordinates no cell
# rewrites, so the cells map in any order, even when `y === x`.
function _ordered_car_map!(y, x, state::BRMCARAdaptiveCenteringState, targets,
                           offset, direction)
    ljac = zero(eltype(x))
    for p in eachindex(state.sources)
        ljac += _cell_map!(y, x, _car_pair_index(state, p), _car_location(x, state, p),
            _car_log_scale(x, state, p), targets[offset + p], state.sources[p],
            direction)
    end
    ljac
end

WarmupHMC.InverseFunctions.inverse(ir::BRMOrderedCenteringReparametrization) =
    BRMOrderedCenteringReparametrization(ir.state, ir.targets, ir.orders, ir.perms,
        ir.starts, ir.groups, ir.block_orders, ir.positions, ir.rows, ir.kinds,
        -ir.direction)

WarmupHMC.reparam_controls(ir::BRMOrderedCenteringReparametrization) =
    vcat(ir.state.ranef.sources, ir.state.hsgp.sources, ir.state.scalar.sources,
         ir.state.car.sources)

function WarmupHMC.restore_reparam_controls!(ir::BRMOrderedCenteringReparametrization,
                                             controls)
    s = ir.state
    sources = (s.ranef.sources, s.hsgp.sources, s.scalar.sources, s.car.sources)
    n = sum(length, sources)
    controls isa AbstractVector && length(controls) == n || throw(ArgumentError(
        "BRM learned-order centering controls must be one value per adaptive " *
        "cell ($n), in the joint pair order"))
    all(c -> c isa Real && isfinite(c) && 0 <= c <= 1, controls) ||
        throw(ArgumentError("centeredness values must lie in [0,1]"))
    offset = 0
    for v in sources
        v .= controls[offset .+ eachindex(v)]
        offset += length(v)
    end
    _derive_orders!(ir)
end

# Every cell's frame at one source-frame draw, in joint pair order: ranef cells
# in the current learned order, the rest in their per-cell frames.
_ordered_frame(ir::BRMOrderedCenteringReparametrization, position, gradient) =
    _centering_frame(ir.state, position, gradient, ir)

# Group `g`'s permuted factor at a source-frame position, packed (the map's
# `Cp`, outside AD).
function _source_group_factor(position, ir::BRMOrderedCenteringReparametrization{N},
                              bi, g) where {N}
    block = ir.state.ranef.blocks[bi]
    K = block.ranef.n_terms
    Cs = _packed_block_cholesky(position, block, ir.block_orders[bi], Val(N + 1))
    rows = ir.positions[bi][ir.orders[bi][:, g]]
    Cp = rows == 1:K ? Cs : _permuted_cholesky(Cs, rows, K, Val(N + 1))
    _block_mixed(block) ? _tscale(Cp, _block_mixing_scale(position, block, g)) : Cp
end

const _CENTERING_CANDIDATES = range(0.0, 1.0; length=11)

# The learned-order wrapper's window rule. Each cell scores WarmupHMC's 11
# candidates `c = 0:0.1:1` with WarmupHMC's default criterion, the
# position-gradient correlation `Cor(q_t, g_t)` over the window's equally
# weighted evidence (`warmuphmc-use` §5d), from the same source-invariant frame
# the per-pair wrapper scores (`_score_candidate`), random-effect cells in their
# current learned order. The lowest finite loss wins; a cell with no finite loss
# keeps its control. The order is then re-derived from the new controls.
function _ordered_select!(ir::BRMOrderedCenteringReparametrization, positions,
                          gradients)
    n = size(positions, 2)
    n > 2 || return false
    old = WarmupHMC.reparam_controls(ir)
    candidates = [WarmupHMC.PartiallyCentered(t) for t in _CENTERING_CANDIDATES]
    shape = (length(candidates), length(old))
    mean_q, mean_g = zeros(shape), zeros(shape)
    qq, gg, qg = zeros(shape), zeros(shape), zeros(shape)
    # Two passes over the evidence: means, then centred moments.
    for pass in 1:2, col in 1:n
        frame = _ordered_frame(ir, positions[:, col], gradients[:, col])
        for p in eachindex(old), (i, t) in enumerate(candidates)
            _, q, g = _score_candidate(frame, p, 0, nothing, t)
            if pass == 1
                mean_q[i, p] += q / n
                mean_g[i, p] += g / n
            else
                dq, dg = q - mean_q[i, p], g - mean_g[i, p]
                qq[i, p] += dq * dq
                gg[i, p] += dg * dg
                qg[i, p] += dq * dg
            end
        end
    end
    new = copy(old)
    for p in eachindex(old)
        losses = [qg[i, p] / sqrt(qq[i, p] * gg[i, p]) for i in axes(qg, 1)]
        finite = findall(isfinite, losses)
        isempty(finite) && continue
        new[p] = _CENTERING_CANDIDATES[finite[argmin(losses[finite])]]
    end
    new == old && return false
    WarmupHMC.restore_reparam_controls!(ir, new)
    true
end

function _ordered_centering_problem(ir, problem, ad_backend, centeredness)
    controls = _initial_controls(centeredness, length(WarmupHMC.reparam_controls(ir)))
    isnothing(controls) || WarmupHMC.restore_reparam_controls!(ir, controls)
    plan = WarmupHMC.WindowSelectionPlan(_ordered_select!; synchronize! = _derive_orders!)
    WarmupHMC.ReparametrizedProblem(ir, problem, ad_backend; scoring_plan=plan)
end

# cdar walk cells are scalar zero-location cells like HSGP basis weights, so
# they reuse that frame shape one-to-one: per-cell marginal spread, diagonal
# transport, linear cost. They keep their own state/argument types for the same
# reason — the ordinary-block exact small-block dispatch must not see them —
# and they form their own plan rather than joining the joint ordinary+HSGP
# wrapper, whose pair order and checkpoint contract are already settled.
mutable struct BRMCDARAdaptiveCenteringState
    blocks::Vector{BRM._CDARAdaptiveCenteringBlock}
    pair_blocks::Vector{Int}
    pair_cells::Vector{Int}
    sources::Vector{Float64}
    effect_indices::BitSet
end

function BRMCDARAdaptiveCenteringState(blocks)
    pair_blocks = Int[]
    pair_cells = Int[]
    sources = Float64[]
    for (bi, block) in enumerate(blocks), cell in eachindex(block.effects)
        push!(pair_blocks, bi)
        push!(pair_cells, cell)
        push!(sources, 0.0)
    end
    effect_indices = BitSet(Iterators.flatten(b.effects for b in blocks))
    BRMCDARAdaptiveCenteringState(
        collect(blocks), pair_blocks, pair_cells, sources, effect_indices,
    )
end

struct BRMCDARAdaptiveCenteringArgument{KIND} <: Function
    state::BRMCDARAdaptiveCenteringState
    pair_number::Int
end

function BRMCDARAdaptiveCenteringArgument(state, pair_number, kind::Symbol)
    kind in (:location, :log_scale) || error(
        "unknown BRM cdar adaptive-centering argument kind $kind",
    )
    BRMCDARAdaptiveCenteringArgument{kind}(state, pair_number)
end

function _cdar_pair_location(state, pair_number)
    (state.pair_blocks[pair_number], state.pair_cells[pair_number])
end

function _cdar_pair_index(state, pair_number)
    bi, cell = _cdar_pair_location(state, pair_number)
    state.blocks[bi].effects[cell]
end

function (::BRMCDARAdaptiveCenteringArgument{:location})(x)
    zero(eltype(x))
end

function (arg::BRMCDARAdaptiveCenteringArgument{:log_scale})(x)
    bi, cell = _cdar_pair_location(arg.state, arg.pair_number)
    BRM._adaptive_cdar_log_scale(x, arg.state.blocks[bi], cell)
end

function _sync_sources!(state::BRMCDARAdaptiveCenteringState, ir)
    length(ir.pairs) == length(state.sources) || throw(DimensionMismatch(
        "BRM cdar adaptive-centering plan has $(length(state.sources)) walk " *
        "cells but the WarmupHMC reparametrizer has $(length(ir.pairs)) pairs",
    ))
    for (p, (idx, value)) in enumerate(ir.pairs)
        expected = _cdar_pair_index(state, p)
        idx == expected || throw(ArgumentError(
            "BRM cdar adaptive-centering pair $p addresses raw coordinate $idx, " *
            "but the model metadata requires $expected; pair ordering changed",
        ))
        state.sources[p] = Float64(value.source.c)
    end
    ir
end

function _prepare_frame(state::BRMCDARAdaptiveCenteringState,
                        ir, position, gradient)
    _sync_sources!(state, ir)
    _centering_frame(state, position, gradient)
end

function _centering_frame(state::BRMCDARAdaptiveCenteringState, position, gradient)
    T = promote_type(eltype(position), eltype(gradient), Float64)
    n = length(state.sources)
    source = T.(state.sources)
    location = zeros(T, n)
    scale = Vector{T}(undef, n)
    innovation = Vector{T}(undef, n)
    invariant_gradient = Vector{T}(undef, n)
    for p in eachindex(state.sources)
        bi, cell = _cdar_pair_location(state, p)
        block = state.blocks[bi]
        idx = block.effects[cell]
        s = exp(BRM._adaptive_cdar_log_scale(position, block, cell))
        c = source[p]
        scale[p] = s
        innovation[p] = position[idx] / s^c
        invariant_gradient[p] = s^c * gradient[idx]
    end
    BRMAdaptiveCenteringFrame(
        source, location, scale, innovation, invariant_gradient,
    )
end

function _adaptive_cdar_centering_reparametrizer(blocks)
    state = BRMCDARAdaptiveCenteringState(blocks)
    pairs = [begin
        bi, cell = _cdar_pair_location(state, p)
        block = state.blocks[bi]
        target = WarmupHMC.PartiallyCentered(0.0)
        source = WarmupHMC.PartiallyCentered(0.0)
        location = BRMCDARAdaptiveCenteringArgument(state, p, :location)
        log_scale = BRMCDARAdaptiveCenteringArgument(state, p, :log_scale)
        block.effects[cell] => WarmupHMC.Reparametrization(
            target, source, location, log_scale,
        )
    end for p in eachindex(state.sources)]
    state, WarmupHMC.IndexedReparametrization(pairs)
end

# Sean's projected partial map over S2Z contrast coordinates (brms PR #1919),
# as a coupled block reparametrization: every group of a coefficient has its
# own weight, all J weights act on the J-1 contrasts together, so it cannot be
# written as per-coordinate pairs. The compiled model is any S2Z frame
# (`target`, its `s2z_rho`); the sampler works under the adapted `source`
# weights. Source contrasts map to physical deviations with the source weights
# and back to target contrasts with the target weights; the (J-1)*log(tau)
# terms of the two restricted Jacobians cancel.
struct BRMS2ZMapBlock
    contrasts::Matrix{Int}
    scales::Vector{Int}
end

struct BRMS2ZMapReparametrization <: WarmupHMC.AbstractReparametrization
    blocks::Vector{BRMS2ZMapBlock}
    source::Vector{Matrix{Float64}}
    target::Vector{Matrix{Float64}}
    direction::Int
end

# Helmert contrast `u[j]` of coefficient `k` (`_s2z_helmert_mul` of its J-1
# contrast coordinates), streamed in descending `j` from `j = J`: `suf` carries
# the suffix sum that function accumulates, so every step is O(1) and every pass
# reproduces `u` bit for bit.
@inline function _s2z_helmert_step(x, contrasts, k, j, suf, ::Type{T}) where {T}
    j == 1 && return suf, suf
    v = T(x[contrasts[j - 1, k]])
    c = inv(sqrt(T((j - 1) * j)))
    suf += v * c
    suf - T(j) * v * c, suf
end

# Level `j` of the target contrasts before the Helmert transpose:
# `d_to * (w - mean(w) - shift)` with `w = u / d_from`.
@inline function _s2z_target_deviation(u, j, k, tau, from, to, mean_w, shift)
    d_to = 1 - to[j, k] + to[j, k] * tau
    d_to * (u / (1 - from[j, k] + from[j, k] * tau) - mean_w - shift)
end

# One coefficient's contrasts from the `from` frame to the `to` frame, written
# into `y`; returns the log-Jacobian. Same math as `_s2z_partial_forward` /
# `_s2z_partial_inverse` / `_s2z_partial_logjac`, written as scalar loops over
# the weight matrices: this runs under Enzyme on every gradient, and views of
# the (constant) weights stored into active temporaries trip its activity
# analysis. It also keeps no heap work vectors — on Julia 1.10 one is a
# use-after-free in Enzyme's reverse pass once another thread runs GC (see
# `_block_location`) — so O(J) passes stream the Helmert contrasts instead, and
# the transpose's prefix sums are the total minus the descending suffix. Each
# `x` contrast is read before its `y` slot is written, so `y === x` is safe.
function _s2z_map_column!(y, x, contrasts::Matrix{Int}, k::Int, tau,
                          from::Matrix{Float64}, to::Matrix{Float64})
    J = size(from, 1)
    T = promote_type(eltype(x), typeof(tau))
    # The restricted Jacobians need only the level weights.
    log_from = zero(T); sum_from = zero(T); log_to = zero(T); sum_to = zero(T)
    for j in 1:J
        d = 1 - from[j, k] + from[j, k] * tau
        log_from += log(d); sum_from += d
        d = 1 - to[j, k] + to[j, k] * tau
        log_to += log(d); sum_to += d
    end
    # Physical deviations over tau: w .- mean(w), w = u ./ d_from.
    sum_w = zero(T)
    suf = zero(T)
    for j in J:-1:1
        u, suf = _s2z_helmert_step(x, contrasts, k, j, suf, T)
        sum_w += u / (1 - from[j, k] + from[j, k] * tau)
    end
    mean_w = sum_w / J
    # Target contrasts: d_to .* (z .- dot(d_to, z) / sum(d_to)).
    weighted = zero(T)
    suf = zero(T)
    for j in J:-1:1
        u, suf = _s2z_helmert_step(x, contrasts, k, j, suf, T)
        weighted += (1 - to[j, k] + to[j, k] * tau) *
            (u / (1 - from[j, k] + from[j, k] * tau) - mean_w)
    end
    shift = weighted / sum_to
    total = zero(T)
    suf = zero(T)
    for j in J:-1:1
        u, suf = _s2z_helmert_step(x, contrasts, k, j, suf, T)
        total += _s2z_target_deviation(u, j, k, tau, from, to, mean_w, shift)
    end
    suffix = zero(T)
    suf = zero(T)
    for j in J:-1:2
        u, suf = _s2z_helmert_step(x, contrasts, k, j, suf, T)
        deviation = _s2z_target_deviation(u, j, k, tau, from, to, mean_w, shift)
        suffix += deviation
        r = j - 1
        y[contrasts[r, k]] =
            ((total - suffix) - T(r) * deviation) * inv(sqrt(T(r * (r + 1))))
    end
    (log(sum_from / J) - log_from) - (log(sum_to / J) - log_to)
end

function WarmupHMC.with_logabsdet_jacobian!(y::AbstractVector,
        t::BRMS2ZMapReparametrization, x::AbstractVector)
    ljac = zero(eltype(x))
    for (b, block) in enumerate(t.blocks)
        from, to = t.direction > 0 ? (t.source[b], t.target[b]) : (t.target[b], t.source[b])
        for k in axes(block.contrasts, 2)
            tau = exp(x[block.scales[k]])
            ljac += _s2z_map_column!(y, x, block.contrasts, k, tau, from, to)
        end
    end
    ljac, y
end

WarmupHMC.InverseFunctions.inverse(t::BRMS2ZMapReparametrization) =
    BRMS2ZMapReparametrization(t.blocks, t.source, t.target, -t.direction)
WarmupHMC.reparam_controls(t::BRMS2ZMapReparametrization) = [copy(m) for m in t.source]
WarmupHMC.snapshot_reparametrization(t::BRMS2ZMapReparametrization) =
    BRMS2ZMapReparametrization(t.blocks, [copy(m) for m in t.source], t.target, t.direction)
function WarmupHMC.restore_reparam_controls!(t::BRMS2ZMapReparametrization, controls)
    controls isa AbstractVector && length(controls) == length(t.source) &&
        all(i -> controls[i] isa AbstractMatrix && size(controls[i]) == size(t.source[i]),
            eachindex(t.source)) || throw(ArgumentError(
        "S2Z centering controls must be one J-by-K weight matrix per S2Z block"))
    all(c -> all(w -> isfinite(w) && 0 <= w <= 1, c), controls) ||
        throw(ArgumentError("S2Z centering weights must lie in [0, 1]"))
    foreach((m, c) -> m .= c, t.source, controls)
    t
end

# Base-name index of BridgeStan's constrained output, e.g. "mu.3" -> :mu.
function _s2z_constrained_index(model)
    names = BRM.StanBlocks.BridgeStan.param_names(model; include_tp=true)
    index = Dict{Symbol,Vector{Int}}()
    for (i, n) in enumerate(names)
        push!(get!(index, Symbol(first(split(n, '.'))), Int[]), i)
    end
    index
end

function _s2z_select!(ir::BRMS2ZMapReparametrization, positions, plans, model, data, index)
    size(positions, 2) > 0 || return false
    draws = [Matrix{Float64}[] for _ in plans]
    for x in eachcol(positions)
        _, y = ir(collect(Float64, x))   # BridgeStan takes a plain Vector{Float64}
        values = try
            BRM.StanBlocks.BridgeStan.param_constrain(model, y; include_tp=true)
        catch err
            err isa MethodError && rethrow()
            continue
        end
        lookup(name) = haskey(index, name) ? (length(index[name]) == 1 ?
            values[only(index[name])] : values[index[name]]) :
            haskey(data, name) ? data[name] :
            throw(ArgumentError("S2Z Fisher selection: no value named `$name`"))
        for (b, plan) in enumerate(plans)
            tau = exp.(y[ir.blocks[b].scales])
            rho = try
                BRM._s2z_fisher_draw(plan, BRM._s2z_row_information(plan, lookup), tau)
            catch err
                err isa Union{ArgumentError,DomainError,BRM.LinearAlgebra.PosDefException} ||
                    rethrow()
                continue
            end
            all(isfinite, rho) && push!(draws[b], rho)
        end
    end
    changed = false
    for (b, rhos) in enumerate(draws)
        isempty(rhos) && continue
        new = [BRM.Statistics.median(r[j, k] for r in rhos) for j in axes(first(rhos), 1),
               k in axes(first(rhos), 2)]
        clamp!(new, 0.0, 1.0)
        changed |= new != ir.source[b]
        ir.source[b] .= new
    end
    changed
end

function _adaptive_s2z_fisher_problem(model, problem, ad_backend, names; centeredness)
    all(b -> b.coordinates === :contrasts, BRM.s2z_effect_blocks(model)) ||
        throw(ArgumentError("S2Z Fisher selection applies Sean's map to contrast " *
            "coordinates; compile without `s2z_coordinates=:groups`"))
    hasproperty(problem, :model) || throw(ArgumentError(
        "S2Z Fisher selection evaluates per-row information through BridgeStan; " *
        "pass StanBlocks' StanProblem"))
    plans = BRM._s2z_information_plans(model)
    blocks = map(plans) do plan
        coords = BRM._s2z_coordinates(model, plan.block, names)
        BRMS2ZMapBlock(coords.contrasts, coords.scales)
    end
    target = [copy(plan.block.rho) for plan in plans]
    source = if isnothing(centeredness)
        [copy(t) for t in target]
    elseif centeredness isa Real
        [fill(Float64(centeredness), size(t)) for t in target]
    else
        [Matrix{Float64}(c) for c in centeredness]
    end
    ir = BRMS2ZMapReparametrization(blocks, [copy(t) for t in target], target, 1)
    WarmupHMC.restore_reparam_controls!(ir, source)
    stan = problem.model
    index = _s2z_constrained_index(stan)
    data = model.data
    plan = WarmupHMC.WindowSelectionPlan(
        (ir_, positions, _) -> _s2z_select!(ir_, positions, plans, stan, data, index))
    WarmupHMC.ReparametrizedProblem(ir, problem, ad_backend; scoring_plan=plan)
end

function _initial_controls(centeredness, n)
    isnothing(centeredness) && return nothing
    values = centeredness isa Real ? fill(centeredness,n) : collect(centeredness)
    length(values) == n || throw(DimensionMismatch("one centering value is required per adaptive coordinate"))
    all(c -> c isa Real && isfinite(c) && 0 <= c <= 1,values) ||
        throw(ArgumentError("centeredness values must lie in [0,1]"))
    values
end

function _initial_centering!(state,ir,centeredness)
    values = _initial_controls(centeredness,length(ir.pairs))
    isnothing(values) && return ir
    ir.pairs .= [index => WarmupHMC.Reparametrization(value.target,
        WarmupHMC.PartiallyCentered(Float64(c)),value.args...)
        for ((index,value),c) in zip(ir.pairs,values)]
    _sync_sources!(state,ir)
end

"""
    adaptive_centering_problem(model, problem, ad_backend;
        unc_names=nothing, centeredness=nothing, s2z_rule=:cells,
        ranef_order=:declared)

Wrap a compiled BRM log-density in WarmupHMC's strictly-online adaptive
centering for exact total-coefficient blocks, S2Z free contrasts, ordinary
scalar or correlated random-effect blocks, R2D2-scaled random-effect blocks,
HSGP basis weights (squared-exponential ungrouped or grouped, or ungrouped
periodic), areal `car` fields (`:icar` or `:bym2`), or `cdar` correlated-walk
cells. Ordinary, R2D2, HSGP, total, S2Z contrast and car cells adapt together
in one wrapper; `cdar` cells form a separate plan and mix with none of them.

`model` is the `SBBRMI` or `GenerativePlan` that emitted `problem`. When
`problem` is StanBlocks' `StanProblem`, unconstrained names are read from its
BridgeStan model; otherwise pass `unc_names` explicitly. `ad_backend` is the
DifferentiationInterface backend WarmupHMC uses for the exact source-to-model
transport.

By default the sampler starts in the compiled model's own frame; an explicit
`centeredness` scalar or vector selects its initial controls. It independently scores
the 11 `c=0:0.1:1` candidates for each `(group, term)` cell from a one-pass
source-invariant innovation frame. The selected score is a fixed-frame proxy;
the transform applied to the model and its hyperparameter gradients remain
exact. Literal endpoints are preserved: `c=0` is BRM's standardised draw and
`c=1` is the model-scale correlated effect.

For exact totals, `c=1` is the sampled group total and `c=0` is the total
scaled around its prior location. The exact marginal prior remains correlated
at either endpoint. Each group/term cell receives its own control automatically.
A total block emitted in generated quantities (no observed likelihood reaches
it) has no sampler coordinate, so it has no cell and does not count as a totals
block when choosing the wrapper. The same holds for every other family: an
ordinary or R2D2 random-effect block, an HSGP term, a car field or a cdar walk
emitted in generated quantities is skipped (see [`adaptive_centering_blocks`](@ref)). A
model left with no cell at all raises, naming what was skipped.

For an S2Z block compiled with `s2z_coordinates=:groups`, each group of each
coefficient is one scalar cell with zero location and scale `tau_k`: `c=0` is
the standard-normal group coordinate `w_j` and `c=1` the centered `tau_k * w_j`.
The compiled per-group `s2z_rho` is the starting frame. This is the per-group
analogue of Sean's partial map; the extra auxiliary mean dimension is sampled
too but never reaches the likelihood. With the default contrast coordinates,
each of the `J-1` free Helmert contrasts is a cell instead. There the compiled
model must be an endpoint frame (`s2z_rho` uniformly `0` or `1` per
coefficient), and controls mix groups. The collapsed population coefficients
stay untouched, and `recover_s2z_draws` applies to the returned compiled-frame
draws. Totals cells precede S2Z cells in the pair order, and both follow any
ordinary, R2D2 and HSGP cells of the same model (an S2Z grouping beside further
groupings or an `hsgp` term on its predictor). Neither shares a wrapper with
cdar cells.

`s2z_rule=:fisher` instead applies Sean's rule (brms PR #1919) at every
restarting warm-up window. The sampler uses Sean's projected per-group partial
map over the contrasts, with one weight per group and coefficient. At each
window boundary, every evidence draw WarmupHMC hands over (the retained pool by
default; with `nonlinear_evidence=:nuts_weighted` or `:all_good_leaves`, an
equally sized weight-proportional sample of the window's leaves) is mapped to
the compiled frame. There
the per-row expected information is evaluated at the draw's fitted values
through BridgeStan, accumulated per group, and turned into Sean's per-draw
weights, rescaled at that draw's `tau`. The new weights are the per-cell median.
Any compiled `s2z_rho` is the target frame and the starting point; `centeredness`
(a scalar or one J-by-K matrix per block) overrides the start. Supported
likelihoods are Gaussian identity, Bernoulli/binomial logit and Poisson log.
This rule cannot yet share a wrapper with totals, ordinary, HSGP or cdar blocks.

For an HSGP, each basis weight is one scalar cell with zero location and
per-basis scale `brm_hsgp_sqrt_spd(omega2, sigma, rho)[basis]`
(`brm_hsgp_periodic_sqrt_spd(harmonics, sigma, rho)[basis]` for a periodic
term); `c=0` is the emitted standardized coordinate, while `c=1` is its
literal spectral/model-scale coefficient. A compiled fixed-partial model
starts at its declared per-basis `centeredness` values, not at zero. Grouped
HSGPs adapt one cell per (group, basis) weight around the same shared
per-basis frame; periodic terms contribute one cell per cosine/sine weight
and join the same HSGP pair order. Ordinary random-effect cells, R2D2 cells
and HSGP basis-weight cells adapt together in one wrapper: pairs enumerate
ordinary cells first, then R2D2 cells, then HSGP cells, then total and S2Z
cells, then car cells, each in the family's own deterministic order (that
order is load-bearing across checkpoint/resume; a model without R2D2 blocks,
total/S2Z cells or car fields keeps the historical ordinary-then-HSGP order).

An R2D2 block (`sd(...) ~ r2d2(...)` in any of its block-wide, per-margin ICC
or joint `include=` forms, or the whole-predictor `effect(lp, :) ~ r2d2(...)`)
adapts exactly like an ordinary block — the same cells, the same triangular map
and endpoints — except that `tau` is the emitted model's derived scale,
re-derived from the `R2`, Dirichlet share and reference/total-scale coordinates
on every evaluation (see [`adaptive_centering_blocks`](@ref)). Those
coordinates are read, never transformed, so the wrapped density stays exact.

A Student-t block (`ranef(:, ID) ~ TDist(nu)`, either emission) adapts the same
way with level `g`'s factor `sqrt(w[g]) * diag(tau) * L`, `w[g]` being the
level's mixing weight: `c=0` is the emitted standardised draw and `c=1` the
effect `b[g]`, which is the compiled `centered_groups` frame. Its cells join
the R2D2 cells, and its weights are read, never transformed.

For a `car(area; edges, type)` field, every area is one cell. Under `:icar` it
is the area's `phi` (areas with neighbours): location 0, scale `sigma`, so
`c=1` is its effect `effects[k]`. Under `:bym2` it is the area's `theta`:
scale `sigma * sqrt(1 - rho)` (`sigma` for an island) and location
`intercept + sigma * sqrt(rho / s_c) * phi`, so `c=1` is the area's total
`intercept + effects[k]`; `phi` stays the emitted standardized coordinate.
`c=0` is the emitted frame. Car cells follow every other joint cell in the pair
order: `:icar` cells in emitted `phi` order, `:bym2` cells in sorted area order.

For a `cdar` walk, each of the `P * W` innovations is one scalar cell with
zero location and its marginal prior spread
`sigma * sqrt(C[p, p] * (1 - rho^(2w)) / (1 - rho^2))`; `c=0` is the emitted
`eta` frame. Walk cells form their own plan rather than joining the joint
wrapper, whose pair order and checkpoint contract are already settled: a model
mixing cdar cells with ordinary or HSGP cells fails before construction.

A correlated block's triangular map centres each margin on its conditional mean
given the margins before it. With the default `ranef_order=:declared`, that order
is the order the block's margins are declared in the formula, so a data-poor
margin declared before data-rich ones can never be non-centred *given* them,
whatever its control. `ranef_order=:learned` instead re-derives every group's
order from that group's controls: margins in descending `c`, ties in
declaration order, so centred margins come first and each other margin is
centred around its conditional mean given the margins ahead of it. One group
may centre a margin that another keeps non-centred. Both compiled endpoints
are supported; the density, Jacobian and priors stay exact, and returned draws
stay in the compiled frame. Because the composite map is not per-coordinate,
this wrapper is selected at each restarting warm-up window
(`WarmupHMC.WindowSelectionPlan`) instead of WarmupHMC's online per-cell
scoring. Every cell, HSGP, total and S2Z cells included, then scores the same
11 candidates with WarmupHMC's default position-gradient correlation over the
window's evidence (see `nonlinear_evidence` in warmuphmc-use), and keeps the
lowest finite loss. Controls, `centeredness` and checkpoints keep the joint pair
order above; the order is a function of the controls. A model without a
correlated block (at least two margins) has nothing to order and gets exactly
the `:declared` wrapper. [`adaptive_centering_controls`](@ref) reports every
cell's control and, for random-effect cells, its place in the current order.

This changes coordinates, not the statistical model or its priors. Conditional
on a block's `C = diag(tau) * L`, an intermediate source coordinate is Gaussian
with covariance `A(c) * A(c)'` whenever the block innovation is standard normal;
the wrapped density and Jacobian still represent the original BRM prior exactly.
"""
function BRM.adaptive_centering_problem(model, problem, ad_backend; unc_names=nothing,
                                       centeredness=nothing, s2z_rule=:cells,
                                       ranef_order=:declared)
    names = isnothing(unc_names) ? _problem_unc_names(problem) : unc_names
    s2z_rule in (:cells, :fisher) ||
        throw(ArgumentError("s2z_rule must be :cells or :fisher"))
    ranef_order in (:declared, :learned) ||
        throw(ArgumentError("ranef_order must be :declared or :learned"))
    if s2z_rule === :fisher
        isempty(BRM.s2z_effect_blocks(model)) &&
            throw(ArgumentError("s2z_rule=:fisher needs S2Z blocks"))
        isempty(BRM._total_blocks_by_placement(model).sampled) &&
            isempty(BRM.adaptive_centering_blocks(model, names)) &&
            isempty(BRM._adaptive_hsgp_centering_blocks(model, names)) &&
            isempty(BRM._adaptive_cdar_centering_blocks(model, names)) &&
            isempty(BRM._adaptive_car_centering_blocks(model, names)) ||
            throw(ArgumentError("s2z_rule=:fisher cannot yet be combined with " *
                "totals, ordinary, HSGP, car or cdar blocks"))
        return _adaptive_s2z_fisher_problem(model, problem, ad_backend, names; centeredness)
    end
    blocks = BRM.adaptive_centering_blocks(model, names)
    hsgp_blocks = BRM._adaptive_hsgp_centering_blocks(model, names)
    cdar_blocks = BRM._adaptive_cdar_centering_blocks(model, names)
    car_blocks = BRM._adaptive_car_centering_blocks(model, names)
    total_blocks = BRM._total_blocks_by_placement(model).sampled
    s2z_blocks = BRM.s2z_effect_blocks(model)
    scalar = !isempty(total_blocks) || !isempty(s2z_blocks)
    if scalar
        isempty(cdar_blocks) || throw(ArgumentError(
            "adaptive total coefficients and S2Z contrasts cannot yet be mixed with cdar blocks; use total_groups=() and s2z_groups=() for the conventional model"))
    end
    if scalar && isempty(blocks) && isempty(hsgp_blocks) && isempty(car_blocks)
        state,ir = _adaptive_scalar_centering_reparametrizer(model,names)
        _initial_centering!(state,ir,centeredness)
        scoring = WarmupHMC.CandidateScoringPlan(
            (ir_,q,g) -> _prepare_frame(state,ir_,q,g), _score_candidate;
            synchronize! = ir_ -> _sync_sources!(state,ir_))
        return WarmupHMC.ReparametrizedProblem(ir,problem,ad_backend;scoring_plan=scoring)
    end
    if isempty(blocks) && isempty(hsgp_blocks) && isempty(cdar_blocks) &&
            isempty(car_blocks)
        generated = BRM._adaptive_generated_labels(model)
        error("BRM adaptive centering: this model has no supported random-effect " *
            "blocks (ordinary or R2D2), squared-exponential or periodic HSGPs, " *
            "areal car fields, or cdar correlated walks" * (isempty(generated) ? "." :
            " with sampler coordinates. Emitted in generated quantities, " *
            "because no observed likelihood reaches them (for example, their " *
            "response is omitted): `" * join(generated, "`, `") * "`. Stan " *
            "re-draws those, so there is nothing to adapt; bind the response " *
            "that reads them, or sample this model without adaptive centering."))
    end
    ordinary = BRM.AdaptiveCenteringBlock[
        b for b in blocks if b isa BRM.AdaptiveCenteringBlock]
    r2d2_blocks = BRM.R2D2AdaptiveCenteringBlock[
        b for b in blocks if b isa BRM.R2D2AdaptiveCenteringBlock]
    # cdar cells never share a wrapper; the branch below refuses the mix.
    if ranef_order === :learned && isempty(cdar_blocks) &&
            any(b -> b.ranef.n_terms > 1, blocks)
        _, ir = _adaptive_ordered_centering_reparametrizer(ordinary, hsgp_blocks,
            r2d2_blocks, scalar ? _scalar_centering_cells(model, names) : _NO_SCALAR_CELLS,
            car_blocks)
        return _ordered_centering_problem(ir, problem, ad_backend, centeredness)
    end
    state, ir = if !isempty(cdar_blocks)
        (isempty(blocks) && isempty(hsgp_blocks) && isempty(car_blocks)) || error(
            "BRM adaptive centering: a single online plan cannot yet mix " *
            "cdar walk cells with random-effect, HSGP or car cells. Build a " *
            "model with one supported adaptive geometry family.",
        )
        _adaptive_cdar_centering_reparametrizer(cdar_blocks)
    elseif scalar
        _adaptive_joint_centering_reparametrizer(ordinary, hsgp_blocks, r2d2_blocks,
            _scalar_centering_cells(model, names), car_blocks)
    elseif !isempty(r2d2_blocks)
        _adaptive_joint_centering_reparametrizer(ordinary, hsgp_blocks, r2d2_blocks,
            _NO_SCALAR_CELLS, car_blocks)
    elseif !isempty(car_blocks) && isempty(ordinary) && isempty(hsgp_blocks)
        _adaptive_car_centering_reparametrizer(car_blocks)
    elseif !isempty(car_blocks) || (!isempty(ordinary) && !isempty(hsgp_blocks))
        _adaptive_joint_centering_reparametrizer(ordinary, hsgp_blocks,
            BRM.R2D2AdaptiveCenteringBlock[], _NO_SCALAR_CELLS, car_blocks)
    elseif !isempty(hsgp_blocks)
        _adaptive_hsgp_centering_reparametrizer(hsgp_blocks)
    else
        _adaptive_centering_reparametrizer(ordinary)
    end
    _initial_centering!(state,ir,centeredness)
    plan = WarmupHMC.CandidateScoringPlan(
        (ir_, position, gradient) -> _prepare_frame(state, ir_, position, gradient),
        _score_candidate;
        synchronize! = ir_ -> _sync_sources!(state, ir_),
    )
    WarmupHMC.ReparametrizedProblem(
        ir, problem, ad_backend; scoring_plan=plan,
    )
end

# ---- Controls report --------------------------------------------------------

_centering_state(ir::WarmupHMC.IndexedReparametrization) =
    isempty(ir.pairs) ? nothing : first(ir.pairs).second.args[1].state
_centering_state(ir::BRMOrderedCenteringReparametrization) = ir.state
_centering_state(::BRMS2ZMapReparametrization) = throw(ArgumentError(
    "adaptive_centering_controls: an `s2z_rule=:fisher` wrapper has one J-by-K " *
    "weight matrix per S2Z block, not per-cell controls; read them with " *
    "`WarmupHMC.reparam_controls(adaptive.reparametrizer)`"))

_centering_orders(::WarmupHMC.IndexedReparametrization) = nothing
_centering_orders(ir::BRMOrderedCenteringReparametrization) = ir.orders

_centering_values(ir::WarmupHMC.IndexedReparametrization) =
    ([last(p).source.c for p in ir.pairs], [last(p).target.c for p in ir.pairs])
_centering_values(ir::BRMOrderedCenteringReparametrization) =
    (WarmupHMC.reparam_controls(ir), ir.targets)

_control_row(cell, index, names, family; binding=nothing, group=nothing,
             margin=nothing, level=nothing, term=nothing, rank=nothing, c, target) =
    (; cell, index, name=names[index], family, binding, group, margin, level, term,
       rank, c=Float64(c), target=Float64(target))

function _control_rows!(rows, state::_BRMRanefCenteringState, orders, offset, values,
                        names, margins)
    controls, targets = values
    for p in eachindex(state.sources)
        bi, k, g = _pair_location(state, p)
        ranef = state.blocks[bi].ranef
        labels = get(margins, ranef.binding, ())
        push!(rows, _control_row(offset + p, state.blocks[bi].effects[k, g], names,
            :ranef; binding=ranef.binding, group=ranef.group,
            margin=length(labels) == ranef.n_terms ? labels[k] : nothing,
            level=ranef.levels[g], term=k,
            rank=isnothing(orders) ? k : findfirst(==(k), view(orders[bi], :, g)),
            c=controls[offset + p], target=targets[offset + p]))
    end
    offset + length(state.sources)
end

function _control_rows!(rows, state::BRMHSGPAdaptiveCenteringState, _, offset, values,
                        names, _)
    controls, targets = values
    for p in eachindex(state.sources)
        bi, basis = _hsgp_pair_location(state, p)
        block = state.blocks[bi]
        push!(rows, _control_row(offset + p, block.effects[basis], names, :hsgp;
            binding=block.logical, margin=(; predictor=block.logical,
            coefficient=block.term), term=basis, c=controls[offset + p],
            target=targets[offset + p]))
    end
    offset + length(state.sources)
end

function _control_rows!(rows, state::BRMScalarCenteringState, _, offset, values,
                        names, _)
    controls, targets = values
    for p in eachindex(state.sources)
        push!(rows, _control_row(offset + p, state.indices[p], names, :scalar;
            c=controls[offset + p], target=targets[offset + p]))
    end
    offset + length(state.sources)
end

function _control_rows!(rows, state::BRMCDARAdaptiveCenteringState, _, offset, values,
                        names, _)
    controls, targets = values
    for p in eachindex(state.sources)
        bi, cell = _cdar_pair_location(state, p)
        block = state.blocks[bi]
        push!(rows, _control_row(offset + p, block.effects[cell], names, :cdar;
            binding=block.logical, margin=(; predictor=block.logical,
            coefficient=block.term), term=cell, c=controls[offset + p],
            target=targets[offset + p]))
    end
    offset + length(state.sources)
end

# `term` is 1 for an `:icar` area's structured `phi` cell and 2 for a `:bym2`
# area's unstructured `theta` cell, the field's position in the emitted law.
function _control_rows!(rows, state::BRMCARAdaptiveCenteringState, _, offset, values,
                        names, _)
    controls, targets = values
    for p in eachindex(state.sources)
        bi, cell = _car_pair_location(state, p)
        block = state.blocks[bi]
        push!(rows, _control_row(offset + p, BRM._adaptive_car_index(block, cell),
            names, :car; binding=block.logical, margin=(; predictor=block.logical,
            coefficient=block.term), level=block.areas[BRM._adaptive_car_area(block, cell)],
            term=block.bym2 ? 2 : 1, c=controls[offset + p], target=targets[offset + p]))
    end
    offset + length(state.sources)
end

function _control_rows!(rows, state::BRMJointAdaptiveCenteringState, orders, offset,
                        values, names, margins)
    offset = _control_rows!(rows, state.ranef, orders, offset, values, names, margins)
    offset = _control_rows!(rows, state.hsgp, nothing, offset, values, names, margins)
    offset = _control_rows!(rows, state.scalar, nothing, offset, values, names, margins)
    _control_rows!(rows, state.car, nothing, offset, values, names, margins)
end

_control_margins(model::Union{BRM.GenerativePlan,BRM.SBBRMI}) =
    Dict(m.block.binding => m.margins
         for m in BRM._brmd_ranef_metadata(BRM.brm_descriptor(model)))
_control_margins(_) = Dict{Symbol,Tuple}()

function BRM.adaptive_centering_controls(adaptive::WarmupHMC.ReparametrizedProblem,
                                         model; unc_names=nothing)
    ir = adaptive.reparametrizer
    state = _centering_state(ir)
    rows = NamedTuple[]
    isnothing(state) && return rows
    names = isnothing(unc_names) ? _problem_unc_names(adaptive.problem) : unc_names
    _control_rows!(rows, state, _centering_orders(ir), 0, _centering_values(ir), names,
                   _control_margins(model))
    rows
end

end # module
