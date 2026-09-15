module BayesianRegressionModelsWarmupHMCExt

using BayesianRegressionModels
using WarmupHMC

const BRM = BayesianRegressionModels

mutable struct BRMAdaptiveCenteringState{SMALL_BLOCKS}
    blocks::Vector{BRM.AdaptiveCenteringBlock}
    pair_blocks::Vector{Int}
    pair_terms::Vector{Int}
    pair_groups::Vector{Int}
    pair_lookup::Vector{Matrix{Int}}
    sources::Vector{Float64}
    effect_indices::BitSet
end

function BRMAdaptiveCenteringState(blocks)
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
    small_blocks = all(b.ranef.n_terms <= 2 for b in blocks)
    effect_indices = BitSet(Iterators.flatten(vec(b.effects) for b in blocks))
    BRMAdaptiveCenteringState{small_blocks}(
        collect(blocks), pair_blocks, pair_terms, pair_groups, pair_lookup, sources,
        effect_indices,
    )
end

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
    tau = exp(x[block.log_scales[i]])
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
function _block_location(x, state::BRMAdaptiveCenteringState, pair_number)
    bi, k, g = _pair_location(state, pair_number)
    T = eltype(x)
    k == 1 && return zero(T)
    block = state.blocks[bi]
    lookup = state.pair_lookup[bi]
    innovations = Vector{T}(undef, k - 1)
    m = zero(T)
    for i in 1:k
        m = zero(T)
        tau = exp(x[block.log_scales[i]])
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

function _prepare_frame(state, ir, position, gradient)
    _sync_sources!(state, ir)
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

mutable struct BRMTotalCenteringState
    indices::Vector{Int}
    scales::Vector{Int}
    locations::Vector{Float64}
    sources::Vector{Float64}
end

struct BRMTotalCenteringArgument{KIND} <: Function
    state::BRMTotalCenteringState
    pair_number::Int
end
(arg::BRMTotalCenteringArgument{:location})(x) = arg.state.locations[arg.pair_number]
(arg::BRMTotalCenteringArgument{:log_scale})(x) = x[arg.state.scales[arg.pair_number]]

function _sync_sources!(state::BRMTotalCenteringState,ir)
    length(state.sources) == length(ir.pairs) || throw(DimensionMismatch("total centering pair count changed"))
    for (p,(index,value)) in enumerate(ir.pairs)
        index == state.indices[p] || throw(ArgumentError("total centering pair order changed"))
        state.sources[p] = value.source.c
    end
    ir
end

function _prepare_frame(state::BRMTotalCenteringState,ir,position,gradient)
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

function _adaptive_total_centering_reparametrizer(model,blocks,names)
    indices,scales,locations = Int[],Int[],Float64[]
    for block in blocks
        coords = BRM._total_coordinates(model,block,names)
        mu = block.A*block.location
        for k in axes(coords.totals,2), g in axes(coords.totals,1)
            push!(indices,coords.totals[g,k])
            push!(scales,coords.scales[k])
            push!(locations,mu[k])
        end
    end
    state = BRMTotalCenteringState(indices,scales,locations,ones(length(indices)))
    pairs = [indices[p] => WarmupHMC.Reparametrization(
        WarmupHMC.PartiallyCentered(1.),WarmupHMC.PartiallyCentered(1.),
        BRMTotalCenteringArgument{:location}(state,p),
        BRMTotalCenteringArgument{:log_scale}(state,p)) for p in eachindex(indices)]
    state,WarmupHMC.IndexedReparametrization(pairs)
end

"""
    adaptive_centering_problem(model, problem, ad_backend; unc_names=nothing)

Wrap a compiled BRM log-density in WarmupHMC's strictly-online adaptive
centering for every ordinary scalar or correlated random-effect block, or for
every ungrouped squared-exponential HSGP basis weight.

`model` is the `SBBRMI` or `GenerativePlan` that emitted `problem`. When
`problem` is StanBlocks' `StanProblem`, unconstrained names are read from its
BridgeStan model; otherwise pass `unc_names` explicitly. `ad_backend` is the
DifferentiationInterface backend WarmupHMC uses for the exact source-to-model
transport.

The sampler starts in the compiled model's own frame and independently scores
the 11 `c=0:0.1:1` candidates for each `(group, term)` cell from a one-pass
source-invariant innovation frame. The selected score is a fixed-frame proxy;
the transform applied to the model and its hyperparameter gradients remain
exact. Literal endpoints are preserved: `c=0` is BRM's standardised draw and
`c=1` is the model-scale correlated effect.

For an HSGP, each basis weight is one scalar cell with zero location and
per-basis scale `brm_hsgp_sqrt_spd(omega2, sigma, rho)[basis]`; `c=0` is the
emitted `beta_raw`, while `c=1` is its literal spectral/model-scale
coefficient. A compiled fixed-partial model starts at its declared per-basis
`centeredness` values, not at zero. Grouped or periodic HSGPs and models mixing HSGP cells with
ordinary random-effect cells fail before construction in this first contract.

This changes coordinates, not the statistical model or its priors. Conditional
on a block's `C = diag(tau) * L`, an intermediate source coordinate is Gaussian
with covariance `A(c) * A(c)'` whenever the block innovation is standard normal;
the wrapped density and Jacobian still represent the original BRM prior exactly.
"""
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

function BRM.adaptive_centering_problem(model, problem, ad_backend; unc_names=nothing,
                                       centeredness=nothing)
    names = isnothing(unc_names) ? _problem_unc_names(problem) : unc_names
    blocks = BRM.adaptive_centering_blocks(model, names)
    hsgp_blocks = BRM._adaptive_hsgp_centering_blocks(model, names)
    total_blocks = BRM.total_effect_blocks(model)
    if !isempty(total_blocks)
        isempty(blocks) && isempty(hsgp_blocks) || throw(ArgumentError(
            "adaptive total coefficients cannot yet be mixed with ordinary or HSGP blocks; use total_groups=() for the conventional model"))
        state,ir = _adaptive_total_centering_reparametrizer(model,total_blocks,names)
        _initial_centering!(state,ir,centeredness)
        scoring = WarmupHMC.CandidateScoringPlan(
            (ir_,q,g) -> _prepare_frame(state,ir_,q,g), _score_candidate;
            synchronize! = ir_ -> _sync_sources!(state,ir_))
        return WarmupHMC.ReparametrizedProblem(ir,problem,ad_backend;scoring_plan=scoring)
    end
    !isempty(blocks) && !isempty(hsgp_blocks) && error(
        "BRM adaptive centering: a single online plan cannot yet mix ordinary " *
        "random-effect cells with HSGP basis-weight cells. Build a model with " *
        "one supported adaptive geometry family for this first contract.",
    )
    isempty(blocks) && isempty(hsgp_blocks) && error(
        "BRM adaptive centering: this model has no supported ordinary " *
        "random-effect blocks or ungrouped squared-exponential HSGPs.",
    )
    state, ir = isempty(hsgp_blocks) ?
        _adaptive_centering_reparametrizer(blocks) :
        _adaptive_hsgp_centering_reparametrizer(hsgp_blocks)
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
