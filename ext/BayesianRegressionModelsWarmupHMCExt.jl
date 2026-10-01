module BayesianRegressionModelsWarmupHMCExt

using BayesianRegressionModels
using WarmupHMC

const BRM = BayesianRegressionModels

# Random-effect block states: ordinary blocks (`BRMAdaptiveCenteringState`) and
# R2D2 blocks (`BRMR2D2AdaptiveCenteringState`) share one cell layout and one
# triangular map; they differ only in where `tau` comes from (`_block_tau`).
abstract type _BRMRanefCenteringState end

mutable struct BRMAdaptiveCenteringState{SMALL_BLOCKS} <: _BRMRanefCenteringState
    blocks::Vector{BRM.AdaptiveCenteringBlock}
    pair_blocks::Vector{Int}
    pair_terms::Vector{Int}
    pair_groups::Vector{Int}
    pair_lookup::Vector{Matrix{Int}}
    sources::Vector{Float64}
    effect_indices::BitSet
end

# Pairs are block-major, then group-major, then term-major.
function _ranef_pair_layout(blocks)
    pair_blocks = Int[]
    pair_terms = Int[]
    pair_groups = Int[]
    pair_lookup = [zeros(Int, b.ranef.n_terms, b.ranef.n_groups) for b in blocks]
    sources = Float64[]
    for (bi, block) in enumerate(blocks), g in 1:block.ranef.n_groups,
            k in 1:block.ranef.n_terms
        push!(pair_blocks, bi)
        push!(pair_terms, k)
        push!(pair_groups, g)
        push!(sources, block.target_c)
        pair_lookup[bi][k, g] = length(sources)
    end
    effect_indices = BitSet(Iterators.flatten(vec(b.effects) for b in blocks))
    (pair_blocks, pair_terms, pair_groups, pair_lookup, sources, effect_indices)
end

function BRMAdaptiveCenteringState(blocks)
    small_blocks = all(b.ranef.n_terms <= 2 for b in blocks)
    BRMAdaptiveCenteringState{small_blocks}(
        collect(blocks), _ranef_pair_layout(blocks)...,
    )
end

# R2D2 blocks keep their own state and argument types for the same reason HSGP
# and cdar cells do: the exact K<=2 Enzyme reverse pass dispatches on the
# ordinary state's concrete pair type and must never see a derived scale.
mutable struct BRMR2D2AdaptiveCenteringState <: _BRMRanefCenteringState
    blocks::Vector{BRM.R2D2AdaptiveCenteringBlock}
    pair_blocks::Vector{Int}
    pair_terms::Vector{Int}
    pair_groups::Vector{Int}
    pair_lookup::Vector{Matrix{Int}}
    sources::Vector{Float64}
    effect_indices::BitSet
end

BRMR2D2AdaptiveCenteringState(blocks) = BRMR2D2AdaptiveCenteringState(
    collect(blocks), _ranef_pair_layout(blocks)...,
)

# A block's marginal scale `tau[i]`: a sampled coordinate for an ordinary
# block, the emitted model's derived scale for an R2D2 block.
_block_tau(x, block::BRM.AdaptiveCenteringBlock, i) = exp(x[block.log_scales[i]])
_block_tau(x, block::BRM.R2D2AdaptiveCenteringBlock, i) =
    exp(BRM._adaptive_log_scale(x, block.scales[i]))

# The kind is a type parameter so Enzyme sees only the accessor branch it is
# differentiating.  Every pair still has the same concrete
# `(location, log_scale)` argument-tuple type, keeping WarmupHMC's pair vector
# concrete.
struct BRMAdaptiveCenteringArgument{KIND,SMALL_BLOCKS} <: Function
    state::BRMAdaptiveCenteringState{SMALL_BLOCKS}
    pair_number::Int
end

function BRMAdaptiveCenteringArgument(
    state::BRMAdaptiveCenteringState{SMALL_BLOCKS}, pair_number, kind::Symbol,
) where {SMALL_BLOCKS}
    kind in (:location, :log_scale) || error(
        "unknown BRM adaptive-centering argument kind $kind",
    )
    BRMAdaptiveCenteringArgument{kind,SMALL_BLOCKS}(state, pair_number)
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
        z = tanh(x[block.cholesky_free[offset + l]])
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
# innovations below it.  That O(k) state is irreducible, and it is the only
# allocation left here — the whole `K x K` `L` and `C` rebuild per pair is gone,
# replaced by the same inline stick walk as `_block_cholesky_entry`.  The
# innovation buffer cannot become an `NTuple`: `IndexedReparametrization` needs a
# concrete pair element type, so making K a type parameter would widen the pair
# vector's `eltype` and throw on any plan mixing block widths.  It cannot be
# cached on the state either, because accessors run under `Enzyme.Const`.
# Recursion is deliberately avoided: a recursive scalar callable here corrupted
# Julia's GC after ~1,750 sustained gradients (fixed by ade46ac/bd9ca2d/b5f4a86).
function _block_location(x, state::_BRMRanefCenteringState, pair_number)
    bi, k, g = _pair_location(state, pair_number)
    T = eltype(x)
    k == 1 && return zero(T)
    block = state.blocks[bi]
    lookup = state.pair_lookup[bi]
    innovations = Vector{T}(undef, k - 1)
    m = zero(T)
    for i in 1:k
        m = zero(T)
        tau = _block_tau(x, block, i)
        stick = one(T)
        offset = (i - 1) * (i - 2) ÷ 2
        for l in 1:i-1
            z = tanh(x[block.cholesky_free[offset + l]])
            m += (tau * (stick * z)) * innovations[l]
            stick *= sqrt(one(T) - z * z)
        end
        if i < k
            c = state.sources[lookup[i, g]]
            u = x[block.effects[i, g]]
            s = tau * stick
            innovations[i] = (u - c * m) / s^c
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

struct BRMR2D2AdaptiveCenteringArgument{KIND} <: Function
    state::BRMR2D2AdaptiveCenteringState
    pair_number::Int
end

function BRMR2D2AdaptiveCenteringArgument(state, pair_number, kind::Symbol)
    kind in (:location, :log_scale) || error(
        "unknown BRM R2D2 adaptive-centering argument kind $kind",
    )
    BRMR2D2AdaptiveCenteringArgument{kind}(state, pair_number)
end

function (arg::BRMR2D2AdaptiveCenteringArgument{:location})(x)
    _block_location(x, arg.state, arg.pair_number)
end

function (arg::BRMR2D2AdaptiveCenteringArgument{:log_scale})(x)
    bi, k, _ = _pair_location(arg.state, arg.pair_number)
    log(_block_cholesky_entry(x, arg.state.blocks[bi], k, k))
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

function _centering_frame(state::_BRMRanefCenteringState, position, gradient)
    T = promote_type(eltype(position), eltype(gradient), Float64)
    n = length(state.sources)
    source = T.(state.sources)
    location = Vector{T}(undef, n)
    scale = Vector{T}(undef, n)
    innovation = Vector{T}(undef, n)
    invariant_gradient = Vector{T}(undef, n)

    for (bi, block) in enumerate(state.blocks), g in 1:block.ranef.n_groups
        K = block.ranef.n_terms
        C = BRM._adaptive_block_cholesky(position, block)
        A = zeros(T, K, K)
        z = Vector{T}(undef, K)
        for k in 1:K
            p = state.pair_lookup[bi][k, g]
            c = source[p]
            m = zero(T)
            for l in 1:k-1
                m += C[k, l] * z[l]
                A[k, l] = c * C[k, l]
            end
            s = C[k, k]
            u = position[block.effects[k, g]]
            z[k] = (u - c * m) / s^c
            A[k, k] = s^c
            location[p] = m
            scale[p] = s
            innovation[p] = z[k]
        end
        gu = gradient[block.effects[:, g]]
        h = transpose(A) * gu
        for k in 1:K
            invariant_gradient[state.pair_lookup[bi][k, g]] = h[k]
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
        s = exp(BRM._adaptive_hsgp_log_scale(position, block, basis))
        c = source[p]
        scale[p] = s
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

# Joint ordinary + R2D2 + HSGP adaptation. `IndexedReparametrization`
# requires one concrete pair element type, so the families cannot contribute
# their own accessor closures to one vector. Every joint pair therefore carries
# the same `BRMJointAdaptiveCenteringArgument`, which selects the family's
# exact existing accessor by pair-number range: pairs `1:n_ranef` are ordinary
# cells, the next `n_r2d2` are R2D2 cells, the rest are HSGP basis-weight
# cells. No accessor arithmetic is duplicated or altered — every branch calls
# the family functions verbatim. A model without R2D2 blocks has `n_r2d2 = 0`
# and the historical ordinary-then-HSGP pair order.
struct BRMJointAdaptiveCenteringState{SB}
    ranef::BRMAdaptiveCenteringState{SB}
    r2d2::BRMR2D2AdaptiveCenteringState
    hsgp::BRMHSGPAdaptiveCenteringState
    n_ranef::Int
    n_r2d2::Int
end

struct BRMJointAdaptiveCenteringArgument{KIND,SB} <: Function
    state::BRMJointAdaptiveCenteringState{SB}
    pair_number::Int
end

function BRMJointAdaptiveCenteringArgument(
        state::BRMJointAdaptiveCenteringState{SB}, pair_number, kind::Symbol,
    ) where {SB}
    kind in (:location, :log_scale) || error(
        "unknown BRM joint adaptive-centering argument kind $kind",
    )
    BRMJointAdaptiveCenteringArgument{kind,SB}(state, pair_number)
end

function (arg::BRMJointAdaptiveCenteringArgument{:location})(x)
    p = arg.pair_number
    s = arg.state
    p <= s.n_ranef && return _block_location(x, s.ranef, p)
    p <= s.n_ranef + s.n_r2d2 && return _block_location(x, s.r2d2, p - s.n_ranef)
    zero(eltype(x))
end

function (arg::BRMJointAdaptiveCenteringArgument{:log_scale})(x)
    p = arg.pair_number
    s = arg.state
    if p <= s.n_ranef
        bi, k, _ = _pair_location(s.ranef, p)
        log(_block_cholesky_entry(x, s.ranef.blocks[bi], k, k))
    elseif p <= s.n_ranef + s.n_r2d2
        bi, k, _ = _pair_location(s.r2d2, p - s.n_ranef)
        log(_block_cholesky_entry(x, s.r2d2.blocks[bi], k, k))
    else
        bi, basis = _hsgp_pair_location(s.hsgp, p - s.n_ranef - s.n_r2d2)
        BRM._adaptive_hsgp_log_scale(x, s.hsgp.blocks[bi], basis)
    end
end

# The family owning joint pair `i`, and the pair's number within it.
function _joint_pair_family(state::BRMJointAdaptiveCenteringState, i)
    i <= state.n_ranef && return (:ranef, i)
    i <= state.n_ranef + state.n_r2d2 && return (:r2d2, i - state.n_ranef)
    (:hsgp, i - state.n_ranef - state.n_r2d2)
end

function _sync_sources!(state::BRMJointAdaptiveCenteringState, ir)
    n_cells = state.n_ranef + state.n_r2d2 + length(state.hsgp.sources)
    length(ir.pairs) == n_cells || throw(DimensionMismatch(
        "BRM joint adaptive-centering plan has $n_cells cells but the " *
        "WarmupHMC reparametrizer has $(length(ir.pairs)) pairs",
    ))
    for (i, (idx, value)) in enumerate(ir.pairs)
        family, j = _joint_pair_family(state, i)
        expected = family === :hsgp ? _hsgp_pair_index(state.hsgp, j) :
                                      _pair_index(getfield(state, family), j)
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

function _centering_frame(state::BRMJointAdaptiveCenteringState, position, gradient)
    frames = (_centering_frame(state.ranef, position, gradient),
              _centering_frame(state.r2d2, position, gradient),
              _centering_frame(state.hsgp, position, gradient))
    BRMAdaptiveCenteringFrame(
        (reduce(vcat, getfield.(frames, f)) for f in fieldnames(BRMAdaptiveCenteringFrame))...,
    )
end

function _adaptive_joint_centering_reparametrizer(
        blocks, hsgp_blocks, r2d2_blocks=BRM.R2D2AdaptiveCenteringBlock[])
    ranef_state = BRMAdaptiveCenteringState(blocks)
    r2d2_state = BRMR2D2AdaptiveCenteringState(r2d2_blocks)
    hsgp_state = BRMHSGPAdaptiveCenteringState(hsgp_blocks)
    effects = (ranef_state.effect_indices, r2d2_state.effect_indices,
               hsgp_state.effect_indices)
    sum(length, effects) == length(union(effects...)) || error(
        "BRM joint adaptive centering: ordinary, R2D2 and HSGP blocks claim " *
        "overlapping unconstrained coordinates; refusing an ambiguous transform.",
    )
    for block in r2d2_state.blocks, s in block.scales
        isdisjoint(BRM._adaptive_scale_coordinates(s), hsgp_state.effect_indices) ||
            error("BRM joint adaptive centering: R2D2 block " *
                  "`$(block.ranef.binding)` derives its scale from an HSGP " *
                  "basis weight this transform rewrites; refusing an inexact " *
                  "transform.")
    end
    n_ranef = length(ranef_state.sources)
    n_r2d2 = length(r2d2_state.sources)
    state = BRMJointAdaptiveCenteringState(
        ranef_state, r2d2_state, hsgp_state, n_ranef, n_r2d2)
    pair(p, idx, c) = idx => WarmupHMC.Reparametrization(
        WarmupHMC.PartiallyCentered(c), WarmupHMC.PartiallyCentered(c),
        BRMJointAdaptiveCenteringArgument(state, p, :location),
        BRMJointAdaptiveCenteringArgument(state, p, :log_scale),
    )
    # Pair order is load-bearing across checkpoint/resume (positional restore):
    # ordinary cells first, then R2D2 cells, then HSGP basis-weight cells, each
    # in the family's own deterministic order.
    pairs = [pair(p, _pair_index(ranef_state, p),
                  ranef_state.blocks[ranef_state.pair_blocks[p]].target_c)
             for p in 1:n_ranef]
    append!(pairs, [pair(n_ranef + j, _pair_index(r2d2_state, j),
                         r2d2_state.blocks[r2d2_state.pair_blocks[j]].target_c)
                    for j in 1:n_r2d2])
    append!(pairs, [begin
        bi, basis = _hsgp_pair_location(hsgp_state, j)
        pair(n_ranef + n_r2d2 + j, _hsgp_pair_index(hsgp_state, j),
             hsgp_state.blocks[bi].target_c[basis])
    end for j in eachindex(hsgp_state.sources)])
    state, WarmupHMC.IndexedReparametrization(pairs)
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

function _adaptive_scalar_centering_reparametrizer(model,names)
    totals = BRM._total_centering_cells(model,names)
    s2z = BRM._s2z_centering_cells(model,names)
    indices = vcat(totals.indices,s2z.indices)
    targets = vcat(totals.targets,s2z.targets)
    state = BRMScalarCenteringState(indices,vcat(totals.scales,s2z.scales),
        vcat(totals.locations,s2z.locations),copy(targets))
    pairs = [indices[p] => WarmupHMC.Reparametrization(
        WarmupHMC.PartiallyCentered(targets[p]),WarmupHMC.PartiallyCentered(targets[p]),
        BRMScalarCenteringArgument{:location}(state,p),
        BRMScalarCenteringArgument{:log_scale}(state,p)) for p in eachindex(indices)]
    state,WarmupHMC.IndexedReparametrization(pairs)
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

# One coefficient's contrasts from the `from` frame to the `to` frame, and the
# log-Jacobian. Same math as `_s2z_partial_forward` / `_s2z_partial_inverse` /
# `_s2z_partial_logjac`, written as scalar loops over the weight matrices: this
# runs under Enzyme on every gradient, and views of the (constant) weights
# stored into active temporaries trip its activity analysis.
function _s2z_map_column(x, contrasts::Matrix{Int}, k::Int, tau,
                         from::Matrix{Float64}, to::Matrix{Float64})
    J = size(from, 1)
    T = promote_type(eltype(x), typeof(tau))
    v = Vector{T}(undef, J - 1)
    for r in 1:J-1
        v[r] = x[contrasts[r, k]]
    end
    u = BRM._s2z_helmert_mul(v)
    # Physical deviations over tau: w .- mean(w), w = u ./ d_from.
    w = Vector{T}(undef, J)
    sum_w = zero(T); log_from = zero(T); sum_from = zero(T)
    for j in 1:J
        d = 1 - from[j, k] + from[j, k] * tau
        w[j] = u[j] / d
        sum_w += w[j]; log_from += log(d); sum_from += d
    end
    mean_w = sum_w / J
    # Target contrasts: d_to .* (z .- dot(d_to, z) / sum(d_to)).
    d_to = Vector{T}(undef, J)
    weighted = zero(T); log_to = zero(T); sum_to = zero(T)
    for j in 1:J
        d_to[j] = 1 - to[j, k] + to[j, k] * tau
        weighted += d_to[j] * (w[j] - mean_w)
        log_to += log(d_to[j]); sum_to += d_to[j]
    end
    shift = weighted / sum_to
    u_to = Vector{T}(undef, J)
    for j in 1:J
        u_to[j] = d_to[j] * (w[j] - mean_w - shift)
    end
    ljac = (log(sum_from / J) - log_from) - (log(sum_to / J) - log_to)
    BRM._s2z_helmert_transpose_mul(u_to), ljac
end

function WarmupHMC.with_logabsdet_jacobian!(y::AbstractVector,
        t::BRMS2ZMapReparametrization, x::AbstractVector)
    ljac = zero(eltype(x))
    for (b, block) in enumerate(t.blocks)
        from, to = t.direction > 0 ? (t.source[b], t.target[b]) : (t.target[b], t.source[b])
        for k in axes(block.contrasts, 2)
            tau = exp(x[block.scales[k]])
            z, l = _s2z_map_column(x, block.contrasts, k, tau, from, to)
            for r in eachindex(z)
                y[block.contrasts[r, k]] = z[r]
            end
            ljac += l
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

function _initial_centering!(state,ir,centeredness)
    isnothing(centeredness) && return ir
    values = centeredness isa Real ? fill(centeredness,length(ir.pairs)) : collect(centeredness)
    length(values) == length(ir.pairs) || throw(DimensionMismatch("one centering value is required per adaptive coordinate"))
    all(c -> c isa Real && isfinite(c) && 0 <= c <= 1,values) ||
        throw(ArgumentError("centeredness values must lie in [0,1]"))
    ir.pairs .= [index => WarmupHMC.Reparametrization(value.target,
        WarmupHMC.PartiallyCentered(Float64(c)),value.args...)
        for ((index,value),c) in zip(ir.pairs,values)]
    _sync_sources!(state,ir)
end

"""
    adaptive_centering_problem(model, problem, ad_backend;
        unc_names=nothing, centeredness=nothing)

Wrap a compiled BRM log-density in WarmupHMC's strictly-online adaptive
centering for exact total-coefficient blocks, S2Z free contrasts, ordinary
scalar or correlated random-effect blocks, R2D2-scaled random-effect blocks,
squared-exponential HSGP basis weights (ungrouped or grouped), or `cdar`
correlated-walk cells. Ordinary, R2D2 and HSGP cells adapt together in one
wrapper, as do totals and S2Z contrasts; `cdar` cells form a separate plan and
mix with neither family.

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
draws. Totals cells precede S2Z cells in the pair order. Neither can currently
share one wrapper with ordinary, HSGP, or cdar cells.

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
per-basis scale `brm_hsgp_sqrt_spd(omega2, sigma, rho)[basis]`; `c=0` is the
emitted standardized coordinate, while `c=1` is its literal
spectral/model-scale coefficient. A compiled fixed-partial model starts at its
declared per-basis `centeredness` values, not at zero. Grouped HSGPs adapt one
cell per (group, basis) weight around the same shared per-basis frame;
periodic HSGPs fail before construction. Ordinary random-effect cells, R2D2
cells and HSGP basis-weight cells adapt together in one wrapper: pairs
enumerate ordinary cells first, then R2D2 cells, then HSGP cells, each in the
family's own deterministic order (that order is load-bearing across
checkpoint/resume; a model without R2D2 blocks keeps the historical
ordinary-then-HSGP order).

An R2D2 block (`sd(...) ~ r2d2(...)` in any of its block-wide, per-margin ICC
or joint `include=` forms, or the whole-predictor `effect(lp, :) ~ r2d2(...)`)
adapts exactly like an ordinary block — the same cells, the same triangular map
and endpoints — except that `tau` is the emitted model's derived scale,
re-derived from the `R2`, Dirichlet share and reference/total-scale coordinates
on every evaluation (see [`adaptive_centering_blocks`](@ref)). Those
coordinates are read, never transformed, so the wrapped density stays exact.

For a `cdar` walk, each of the `P * W` innovations is one scalar cell with
zero location and its marginal prior spread
`sigma * sqrt(C[p, p] * (1 - rho^(2w)) / (1 - rho^2))`; `c=0` is the emitted
`eta` frame. Walk cells form their own plan rather than joining the joint
wrapper, whose pair order and checkpoint contract are already settled: a model
mixing cdar cells with ordinary or HSGP cells fails before construction.

This changes coordinates, not the statistical model or its priors. Conditional
on a block's `C = diag(tau) * L`, an intermediate source coordinate is Gaussian
with covariance `A(c) * A(c)'` whenever the block innovation is standard normal;
the wrapped density and Jacobian still represent the original BRM prior exactly.
"""
function BRM.adaptive_centering_problem(model, problem, ad_backend; unc_names=nothing,
                                       centeredness=nothing, s2z_rule=:cells)
    names = isnothing(unc_names) ? _problem_unc_names(problem) : unc_names
    s2z_rule in (:cells, :fisher) ||
        throw(ArgumentError("s2z_rule must be :cells or :fisher"))
    if s2z_rule === :fisher
        isempty(BRM.s2z_effect_blocks(model)) &&
            throw(ArgumentError("s2z_rule=:fisher needs S2Z blocks"))
        isempty(BRM.total_effect_blocks(model)) &&
            isempty(BRM.adaptive_centering_blocks(model, names)) &&
            isempty(BRM._adaptive_hsgp_centering_blocks(model, names)) &&
            isempty(BRM._adaptive_cdar_centering_blocks(model, names)) ||
            throw(ArgumentError("s2z_rule=:fisher cannot yet be combined with " *
                "totals, ordinary, HSGP or cdar blocks"))
        return _adaptive_s2z_fisher_problem(model, problem, ad_backend, names; centeredness)
    end
    blocks = BRM.adaptive_centering_blocks(model, names)
    hsgp_blocks = BRM._adaptive_hsgp_centering_blocks(model, names)
    cdar_blocks = BRM._adaptive_cdar_centering_blocks(model, names)
    total_blocks = BRM.total_effect_blocks(model)
    s2z_blocks = BRM.s2z_effect_blocks(model)
    if !isempty(total_blocks) || !isempty(s2z_blocks)
        isempty(blocks) && isempty(hsgp_blocks) && isempty(cdar_blocks) || throw(ArgumentError(
            "adaptive total coefficients and S2Z contrasts cannot yet be mixed with random-effect (ordinary or R2D2), HSGP, or cdar blocks; use total_groups=() and s2z_groups=() for the conventional model"))
        state,ir = _adaptive_scalar_centering_reparametrizer(model,names)
        _initial_centering!(state,ir,centeredness)
        scoring = WarmupHMC.CandidateScoringPlan(
            (ir_,q,g) -> _prepare_frame(state,ir_,q,g), _score_candidate;
            synchronize! = ir_ -> _sync_sources!(state,ir_))
        return WarmupHMC.ReparametrizedProblem(ir,problem,ad_backend;scoring_plan=scoring)
    end
    isempty(blocks) && isempty(hsgp_blocks) && isempty(cdar_blocks) && error(
        "BRM adaptive centering: this model has no supported random-effect " *
        "blocks (ordinary or R2D2), squared-exponential HSGPs, or cdar " *
        "correlated walks.",
    )
    ordinary = BRM.AdaptiveCenteringBlock[
        b for b in blocks if b isa BRM.AdaptiveCenteringBlock]
    r2d2_blocks = BRM.R2D2AdaptiveCenteringBlock[
        b for b in blocks if b isa BRM.R2D2AdaptiveCenteringBlock]
    state, ir = if !isempty(cdar_blocks)
        (isempty(blocks) && isempty(hsgp_blocks)) || error(
            "BRM adaptive centering: a single online plan cannot yet mix " *
            "cdar walk cells with random-effect or HSGP cells. Build a model " *
            "with one supported adaptive geometry family.",
        )
        _adaptive_cdar_centering_reparametrizer(cdar_blocks)
    elseif !isempty(r2d2_blocks)
        _adaptive_joint_centering_reparametrizer(ordinary, hsgp_blocks, r2d2_blocks)
    elseif !isempty(ordinary) && !isempty(hsgp_blocks)
        _adaptive_joint_centering_reparametrizer(ordinary, hsgp_blocks)
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

end # module
