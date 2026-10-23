# Concurrent reverse-mode gradients through the adaptive-centering
# reparametrizers, the learned-order wrapper included. On Julia 1.10, Enzyme's
# reverse pass corrupts Julia's GC once another thread collects while a
# differentiated accessor allocates a heap work vector, reads a struct holding
# vectors inside its row loop, or reaches two
# `_block_location` instantiations (see `_block_location` in
# BayesianRegressionModelsWarmupHMCExt). Completion and multi-chain workers do
# exactly that; `threaded_gradient_stress.jl` runs the stress in a child Julia.
#
#     julia --project=test test/adaptive_centering_threads.jl
#
# Synthetic toy blocks only; no compiled Stan model is involved.
using BayesianRegressionModels, WarmupHMC, Enzyme, LogDensityProblems, Random
using DifferentiationInterface: AutoEnzyme
include("threaded_gradient_stress.jl")

const BRM = BayesianRegressionModels
const AC_EXT = Base.get_extension(BRM, :BayesianRegressionModelsWarmupHMCExt)
const _GRADIENTS_PER_WORKER = 40_000

struct _StdNormal
    d::Int
end
LogDensityProblems.dimension(p::_StdNormal) = p.d
LogDensityProblems.capabilities(::Type{_StdNormal}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::_StdNormal, y) = -sum(abs2, y) / 2
LogDensityProblems.logdensity_and_gradient(::_StdNormal, y) =
    (-sum(abs2, y) / 2, -collect(y))

# Correlated block `K x G` at `offset`: log-scales, then Cholesky factors, then
# effects; returns the block and the next free coordinate.
function _synthetic_block(name, K, G, offset)
    nchol = K * (K - 1) ÷ 2
    ranef = BRM.RanefBlock(name, :correlated, :g, nothing, nothing, collect(1:G),
                           K, G, Symbol(:z_, name), true, false)
    block = BRM.AdaptiveCenteringBlock(ranef, 0.0,
        reshape(collect(offset + K + nchol .+ (1:K*G)), K, G),
        collect(offset + K .+ (1:nchol)), collect(offset .+ (1:K)))
    block, offset + K + nchol + K * G
end

function _ordinary_blocks(offset=0)
    wide, offset = _synthetic_block(:r_wide, 7, 10, offset)
    narrow, offset = _synthetic_block(:r_narrow, 3, 10, offset)
    [wide, narrow], offset
end

function _with_controls(state, ir, d)
    controls = rand(Xoshiro(1), length(ir.pairs))
    ir.pairs .= map(ir.pairs, controls) do (idx, value), c
        idx => WarmupHMC.Reparametrization(
            value.target, WarmupHMC.PartiallyCentered(c), value.args...)
    end
    AC_EXT._sync_sources!(state, ir)
    WarmupHMC.ReparametrizedProblem(ir, _StdNormal(d), AutoEnzyme()), d
end

# The learned-order wrapper's controls also fix every group's triangular order;
# random controls permute most groups.
function _with_controls(_, ir::AC_EXT.BRMOrderedCenteringReparametrization, d)
    controls = rand(Xoshiro(1), length(WarmupHMC.reparam_controls(ir)))
    WarmupHMC.restore_reparam_controls!(ir, controls)
    WarmupHMC.ReparametrizedProblem(ir, _StdNormal(d), AutoEnzyme()), d
end

function _ordinary_problem()
    blocks, d = _ordinary_blocks()
    _with_controls(AC_EXT._adaptive_centering_reparametrizer(blocks)..., d)
end

# Ordinary + R2D2 + squared-exponential HSGP cells in one joint wrapper. Every
# R2D2 term's log-scale carries a linear, a unit-interval and a simplex atom.
# `scalar=true` appends twelve total/S2Z cells sharing three log-scales;
# `learned=true` builds the `ranef_order=:learned` wrapper over the same cells.
function _joint_problem(; scalar=false, learned=false)
    blocks, o = _ordinary_blocks()
    K, G = 3, 10
    reference, r2, simplex = o + 1, o + 2, collect(o + 3:o + 5)
    cholesky = collect(o + 6:o + 5 + K * (K - 1) ÷ 2)
    effects = reshape(collect(last(cholesky) .+ (1:K*G)), K, G)
    scales = [BRM._AdaptiveLogScale(0.0, [(1.0, reference)], [(0.5, 0.5, r2)],
                  [BRM._AdaptiveSimplexLog(0.5, simplex, k)]) for k in 1:K]
    ranef = BRM.RanefBlock(:r_r2d2, :correlated, :g, nothing, nothing, collect(1:G),
                           K, G, :z_r2d2, true, false)
    r2d2 = BRM.R2D2AdaptiveCenteringBlock(ranef, 0.0, effects, cholesky, scales)
    o = last(effects)
    M = 12
    hsgp = BRM._HSGPAdaptiveCenteringBlock(:hsgp, :gp_x, zeros(M), collect(o .+ (1:M)),
        [o + M + 1], [0.0], [Inf], o + M + 2, 0.0, Inf,
        reshape([(pi * j / 8)^2 for j in 1:M], M, 1), Float64[])
    d = o + M + 2
    n = scalar ? 12 : 0
    cells = (; indices=collect(d .+ (1:n)), scales=d + n .+ mod1.(1:n, 3),
               locations=collect(range(-1, 1; length=12))[1:n], targets=zeros(n))
    build = learned ? AC_EXT._adaptive_ordered_centering_reparametrizer :
        AC_EXT._adaptive_joint_centering_reparametrizer
    _with_controls(build(blocks, [hsgp], [r2d2], cells)..., d + (scalar ? n + 3 : 0))
end

function _s2z_problem()
    J, K = 8, 6
    contrasts = reshape(collect(1:(J - 1) * K), J - 1, K)
    scales = collect((J - 1) * K .+ (1:K))
    rng = Xoshiro(2)
    s2z = AC_EXT.BRMS2ZMapReparametrization([AC_EXT.BRMS2ZMapBlock(contrasts, scales)],
        [rand(rng, J, K)], [rand(rng, J, K)], 1)
    WarmupHMC.ReparametrizedProblem(s2z, _StdNormal(last(scales)), AutoEnzyme()), last(scales)
end

# HSGP-only wrapper: a two-axis squared-exponential block and a periodic block.
function _hsgp_problem()
    M = 10
    se = BRM._HSGPAdaptiveCenteringBlock(:mu, :gp_x, zeros(M), collect(1:M),
        [M + 1, M + 2], [0.0, 0.0], [Inf, Inf], M + 3, 0.0, Inf,
        hcat([(pi * j / 8)^2 for j in 1:M], [(pi * j / 6)^2 for j in 1:M]), Float64[])
    o = M + 3
    periodic = BRM._HSGPAdaptiveCenteringBlock(:sigma, :gp_t, fill(0.5, 2M),
        collect(o .+ (1:2M)), [o + 2M + 1], [0.0], [Inf], o + 2M + 2, 0.0, Inf,
        zeros(0, 0), vcat(1.0:M, 1.0:M))
    _with_controls(AC_EXT._adaptive_hsgp_centering_reparametrizer([se, periodic])...,
        o + 2M + 2)
end

# cdar walk cells: 4 groups x 6 steps, bounded persistence.
function _cdar_problem()
    P, W = 4, 6
    block = BRM._CDARAdaptiveCenteringBlock(:mu, :cdar_t, P, W, collect(1:P*W),
        P * W + 1, 0.0, Inf, P * W + 2, 0.0, 1.0, [1.0, 1.5, 0.7, 1.2])
    _with_controls(AC_EXT._adaptive_cdar_centering_reparametrizer([block])..., P * W + 2)
end

# Areal car cells: a BYM2 field over 7 areas (two components and an island,
# whose `theta` cells read their intercept and `phi` cell's coordinates) and an
# ICAR field.
# `offset` shifts every coordinate past another family's.
function _car_blocks(offset=0)
    struct_idx = [1, 2, 3, 0, 4, 5, 6]
    bym2 = BRM._CARAdaptiveCenteringBlock(:mu, :car_mu_area, true,
        offset .+ collect(1:6), offset .+ collect(7:13), struct_idx, [1, 2, 3, 5, 6, 7],
        log.([1.4, 1.4, 1.4, 4.0, 4.0, 4.0]), offset + 14, 0.0, Inf, offset + 15, 0.0, 1.0,
        offset + 22, collect(Any, 1:7))
    icar = BRM._CARAdaptiveCenteringBlock(:sigma, :car_sigma_area, false,
        offset .+ collect(16:20), Int[], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], zeros(5),
        offset + 21, 0.0, Inf, 0, 0.0, 1.0, 0, collect(Any, 1:5))
    [bym2, icar], offset + 22
end

function _car_problem()
    blocks, d = _car_blocks()
    _with_controls(AC_EXT._adaptive_car_centering_reparametrizer(blocks)..., d)
end

# Car cells after ordinary, R2D2, HSGP and scalar cells in the joint wrapper.
function _joint_car_problem(; learned=false)
    blocks, o = _ordinary_blocks()
    M = 12
    hsgp = BRM._HSGPAdaptiveCenteringBlock(:hsgp, :gp_x, zeros(M), collect(o .+ (1:M)),
        [o + M + 1], [0.0], [Inf], o + M + 2, 0.0, Inf,
        reshape([(pi * j / 8)^2 for j in 1:M], M, 1), Float64[])
    d = o + M + 2
    n = 12
    cells = (; indices=collect(d .+ (1:n)), scales=d + n .+ mod1.(1:n, 3),
               locations=collect(range(-1, 1; length=n)), targets=zeros(n))
    car, d = _car_blocks(d + n + 3)
    build = learned ? AC_EXT._adaptive_ordered_centering_reparametrizer :
        AC_EXT._adaptive_joint_centering_reparametrizer
    _with_controls(build(blocks, [hsgp], BRM.R2D2AdaptiveCenteringBlock[], cells, car)...,
                   d)
end

# Scalar total / S2Z cells: twelve cells sharing three log-scale coordinates.
function _scalar_problem()
    n = 12
    _with_controls(AC_EXT._scalar_centering_reparametrizer(collect(1:n),
        n .+ mod1.(1:n, 3), collect(range(-1, 1; length=n)), zeros(n))..., n + 3)
end

const _CASES = ("ordinary" => _ordinary_problem, "joint" => _joint_problem,
                "joint_scalar" => () -> _joint_problem(; scalar=true),
                "learned_order" => () -> _joint_problem(; scalar=true, learned=true),
                "s2z" => _s2z_problem, "hsgp" => _hsgp_problem,
                "cdar" => _cdar_problem, "scalar" => _scalar_problem,
                "car" => _car_problem, "joint_car" => _joint_car_problem,
                "learned_order_car" => () -> _joint_car_problem(; learned=true))

run_threaded_gradient_gate(@__FILE__, _CASES, _GRADIENTS_PER_WORKER)
