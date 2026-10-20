# `adaptive_centering_problem(...; ranef_order=:learned)`: each group's
# triangular order follows its learned controls instead of the formula
# declaration order (snag correlated-kreis-15564811, decision 11w47g1).
#
#     julia --project=test test/adaptive_centering_learned_order.jl
#
# The synthetic testsets need no Stan model: they check the map against a dense
# LinearAlgebra reference, its inverse, Jacobian and Enzyme gradient, and the
# window rule against an independent re-implementation. The BridgeStan testsets
# check both compiled endpoints against BridgeStan's own constrained factor, the
# controls report, and a short checkpointed and resumed WarmupHMC run.
using Test
using BayesianRegressionModels
using WarmupHMC, Enzyme, BridgeStan, LogDensityProblems, LinearAlgebra, Random
using Serialization: deserialize
using DifferentiationInterface: AutoEnzyme
using Distributions: Normal, Exponential
import StanBlocks

const BRM = BayesianRegressionModels
const AC_EXT = Base.get_extension(BRM, :BayesianRegressionModelsWarmupHMCExt)

struct _LearnedStdNormal
    d::Int
end
LogDensityProblems.dimension(p::_LearnedStdNormal) = p.d
LogDensityProblems.capabilities(::Type{_LearnedStdNormal}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::_LearnedStdNormal, y) = -sum(abs2, y) / 2
LogDensityProblems.logdensity_and_gradient(::_LearnedStdNormal, y) =
    (-sum(abs2, y) / 2, -collect(y))

# Correlated block `K x G` at `offset`: log-scales, Cholesky factors, effects.
function _learned_block(name, K, G, offset; target=0.0)
    nchol = K * (K - 1) ÷ 2
    ranef = BRM.RanefBlock(name, :correlated, :g, nothing, nothing, collect(1:G),
                           K, G, Symbol(:z_, name), true, false)
    block = BRM.AdaptiveCenteringBlock(ranef, target,
        reshape(collect(offset + K + nchol .+ (1:K*G)), K, G),
        collect(offset + K .+ (1:nchol)), collect(offset .+ (1:K)))
    block, offset + K + nchol + K * G
end

# A block's declared scales and the unit-row factor `L` its Cholesky
# coordinates spell, read without BRM's wrapper.
_learned_source(x, block) = (BRM._adaptive_block_taus(x, block),
    BRM._adaptive_cholesky_corr(x[block.cholesky_free], block.ranef.n_terms))

# Stan's `cholesky_factor_corr` log-Jacobian from its unconstrained coordinates
# to the correlation matrix: tanh, stick-breaking, then `L -> L * L'`.
function _stan_correlation_logjac(free, K)
    L = BRM._adaptive_cholesky_corr(free, K)
    z = tanh.(free)
    logjac = sum(log1p.(-z .^ 2); init=0.0)
    for i in 2:K
        logjac += sum(log(sqrt(1 - sum(abs2, L[i, 1:l-1]; init=0.0))) for l in 1:i-1)
        logjac += (K - i) * log(L[i, i])
    end
    logjac
end

# The unconstrained coordinates of a unit-row lower-triangular factor.
_stan_correlation_free(L, K) =
    [atanh(L[i, l] / sqrt(1 - sum(abs2, L[i, 1:l-1]; init=0.0))) for i in 2:K for l in 1:i-1]

# The orders the learned wrapper must use: the block's margins by mean control
# (ties in declaration order), each group's by control (ties in block order).
function _learned_orders(c)
    K, G = size(c)
    block = sortperm(vec(sum(c; dims=2)) ./ G; rev=true)
    position = invperm(block)
    block, position, [sortperm(1:K; by=k -> (-c[k, g], position[k])) for g in 1:G]
end

# Dense reference of the learned-order map. The source frame's Cholesky
# coordinates spell the correlation in the block order; the compiled ones are
# those of the declared-order factor. Each group's effects go through
# `Cp = cholesky(P Sigma P')` in its own order, then the declared-order
# compiled frame at `target_c`.
function _learned_reference(x, blocks, controls; source=_learned_source)
    y = copy(x)
    ljac = 0.0
    offset = 0
    for block in blocks
        K, G = size(block.effects)
        t = block.target_c
        c = reshape(controls[offset .+ (1:K*G)], K, G)
        order, position, groups = _learned_orders(c)
        tau, Ls = source(x, block)
        Cs = Diagonal(tau[order]) * Ls
        Sigma = (Cs * Cs')[position, position]
        C = cholesky(Symmetric(Sigma)).L
        if order != 1:K
            y[block.cholesky_free] = _stan_correlation_free(C ./ sqrt.(sum(abs2, C; dims=2)), K)
            ljac += _stan_correlation_logjac(x[block.cholesky_free], K) -
                _stan_correlation_logjac(y[block.cholesky_free], K)
        end
        for g in 1:G
            perm = groups[g]
            Cp = cholesky(Symmetric(Sigma[perm, perm])).L
            w, b = zeros(K), zeros(K)
            for j in 1:K
                a = perm[j]
                m = sum(Cp[j, i] * w[i] for i in 1:j-1; init=0.0)
                w[j] = (x[block.effects[a, g]] - c[a, g] * m) / Cp[j, j]^c[a, g]
                b[a] = m + Cp[j, j] * w[j]
                ljac += (1 - c[a, g]) * log(Cp[j, j])
            end
            z = zeros(K)
            for k in 1:K
                m = sum(C[k, l] * z[l] for l in 1:k-1; init=0.0)
                z[k] = (b[k] - m) / C[k, k]
                y[block.effects[k, g]] = t * m + C[k, k]^t * z[k]
                ljac += (t - 1) * log(C[k, k])
            end
        end
        offset += K * G
    end
    ljac, y
end

# Independent window rule: every cell's frame in its group's learned order from
# dense algebra, the 11 candidates' position-gradient correlation over equally
# weighted columns, the lowest finite loss.
function _learned_reference_select(positions, gradients, blocks, controls)
    candidates = 0.0:0.1:1.0
    n = size(positions, 2)
    q = zeros(length(controls), length(candidates), n)
    gq = similar(q)
    for col in 1:n
        x, grad = positions[:, col], gradients[:, col]
        offset = 0
        for block in blocks
            K, G = size(block.effects)
            c = reshape(controls[offset .+ (1:K*G)], K, G)
            order, position, groups = _learned_orders(c)
            tau, Ls = _learned_source(x, block)
            Cs = Diagonal(tau[order]) * Ls
            Sigma = (Cs * Cs')[position, position]
            for g in 1:G
                perm = groups[g]
                Cp = cholesky(Symmetric(Sigma[perm, perm])).L
                w, m, A = zeros(K), zeros(K), zeros(K, K)
                for j in 1:K
                    a = perm[j]
                    m[j] = sum(Cp[j, i] * w[i] for i in 1:j-1; init=0.0)
                    w[j] = (x[block.effects[a, g]] - c[a, g] * m[j]) / Cp[j, j]^c[a, g]
                    A[j, 1:j-1] = c[a, g] .* Cp[j, 1:j-1]
                    A[j, j] = Cp[j, j]^c[a, g]
                end
                h = A' * grad[block.effects[perm, g]]
                for j in 1:K, (i, t) in enumerate(candidates)
                    p = offset + (g - 1) * K + perm[j]
                    s = Cp[j, j]
                    q[p, i, col] = t * m[j] + s^t * w[j]
                    gq[p, i, col] = h[j] / s^t
                end
            end
            offset += K * G
        end
    end
    map(eachindex(controls)) do p
        losses = [_learned_cor(q[p, i, :], gq[p, i, :]) for i in eachindex(candidates)]
        finite = findall(isfinite, losses)
        isempty(finite) ? controls[p] : candidates[finite[argmin(losses[finite])]]
    end
end

function _learned_cor(a, b)
    da, db = a .- sum(a) / length(a), b .- sum(b) / length(b)
    sum(da .* db) / sqrt(sum(da .^ 2) * sum(db .^ 2))
end

function _learned_fd_jacobian(f, x; h=1e-6)
    J = zeros(length(x), length(x))
    for i in eachindex(x)
        plus, minus = copy(x), copy(x)
        plus[i] += h
        minus[i] -= h
        J[:, i] = (last(f(plus)) - last(f(minus))) / (2h)
    end
    J
end

function _learned_fd_gradient(problem, x; h=1e-6)
    map(eachindex(x)) do i
        plus, minus = copy(x), copy(x)
        plus[i] += h
        minus[i] -= h
        (LogDensityProblems.logdensity(problem, plus) -
         LogDensityProblems.logdensity(problem, minus)) / (2h)
    end
end

_set_pair_controls!(state, ir, controls) = begin
    ir.pairs .= [i => WarmupHMC.Reparametrization(v.target,
                     WarmupHMC.PartiallyCentered(c), v.args...)
                 for ((i, v), c) in zip(ir.pairs, controls)]
    AC_EXT._sync_sources!(state, ir)
end

@testset "learned order follows descending controls, ties by the given order" begin
    @test AC_EXT._learned_order([0.3, 0.7, 0.3, 1.0]) == [4, 2, 1, 3]
    @test AC_EXT._learned_order(zeros(3)) == 1:3
    @test AC_EXT._learned_order(ones(3)) == 1:3
    @test AC_EXT._learned_order([0.3, 0.7, 0.3], [3, 1, 2]) == [2, 3, 1]
end

@testset "permuted factor stays real near a singular correlation" begin
    # A difference-of-squares Cholesky rounded a conditional variance to
    # -1.8e-18 here and threw a DomainError mid-warm-up; Givens rotations
    # keep every diagonal a norm.
    rng = Xoshiro(5)
    for K in (3, 4), trial in 1:25
        block, d = _learned_block(:near, K, 1, 0)
        x = randn(rng, d)
        x[block.cholesky_free] .= rand(rng, [-1, 1], length(block.cholesky_free)) .*
            (8 .+ 11 .* rand(rng, length(block.cholesky_free)))
        x[block.log_scales] .= 4 .* randn(rng, K)
        tau, L = _learned_source(x, block)
        C = Diagonal(tau) * L
        packed = AC_EXT._packed_block_cholesky(x, block, 1:K, Val(K))
        @test AC_EXT._unpacked(packed, K) == C
        for perm in (randperm(rng, K), reverse(1:K))
            Cp = AC_EXT._unpacked(AC_EXT._permuted_cholesky(packed, perm, K, Val(K)), K)
            @test all(isfinite, Cp)
            @test all(>=(0), diag(Cp))
            @test istril(Cp)
            Sigma = (C * C')[perm, perm]
            @test Cp * Cp' ≈ Sigma rtol=1e-12 atol=1e-12 * maximum(abs, Sigma)
        end
        _, ir = AC_EXT._adaptive_ordered_centering_reparametrizer([block],
            BRM._HSGPAdaptiveCenteringBlock[])
        WarmupHMC.restore_reparam_controls!(ir, rand(rng, 0:0.1:1, K))
        ljac, y = ir(x)
        @test isfinite(ljac) && all(isfinite, y)
    end
end

@testset "synthetic learned-order map is exact at both endpoints" begin
    rng = Xoshiro(20261020)
    b1, o = _learned_block(:r3, 3, 5, 0)
    b2, o = _learned_block(:r2_centred, 2, 4, o; target=1.0)
    b3, o = _learned_block(:r4, 4, 3, o)
    blocks = [b1, b2, b3]
    d = o
    ncell = sum(length(b.effects) for b in blocks)
    x = 0.6 .* randn(rng, d)
    _, ir = AC_EXT._adaptive_ordered_centering_reparametrizer(blocks,
        BRM._HSGPAdaptiveCenteringBlock[])
    @test WarmupHMC.reparam_controls(ir) == [b.target_c for b in blocks
        for _ in 1:length(b.effects)]
    for trial in 1:6
        controls = rand(rng, 0:0.1:1, ncell)
        WarmupHMC.restore_reparam_controls!(ir, controls)
        @test WarmupHMC.reparam_controls(ir) == controls
        offset = 0
        for (bi, block) in enumerate(blocks)
            K, G = size(block.effects)
            order, _, groups = _learned_orders(reshape(controls[offset .+ (1:K*G)], K, G))
            @test ir.block_orders[bi] == order
            @test ir.orders[bi] == reduce(hcat, groups)
            offset += K * G
        end
        ljac, y = ir(x)
        reference_ljac, reference = _learned_reference(x, blocks, controls)
        @test y ≈ reference atol=1e-12
        @test ljac ≈ reference_ljac atol=1e-11
        inverse_ljac, roundtrip = WarmupHMC.InverseFunctions.inverse(ir)(y)
        @test roundtrip ≈ x atol=1e-12
        @test inverse_ljac ≈ -ljac atol=1e-11
        @test first(logabsdet(_learned_fd_jacobian(ir, x))) ≈ ljac atol=1e-6
        problem = WarmupHMC.ReparametrizedProblem(ir, _LearnedStdNormal(d), AutoEnzyme())
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, x)
        @test lp ≈ ljac + LogDensityProblems.logdensity(_LearnedStdNormal(d), y) atol=1e-12
        @test gradient ≈ _learned_fd_gradient(problem, x) rtol=1e-6 atol=1e-6
    end
    # Aliased output: every group's sources are read before its outputs land.
    controls = rand(rng, 0:0.1:1, ncell)
    WarmupHMC.restore_reparam_controls!(ir, controls)
    aliased = copy(x)
    @test WarmupHMC.with_logabsdet_jacobian!(aliased, ir, aliased) == ir(x)
end

@testset "declared orders reproduce the per-pair joint wrapper (ordinary, R2D2, HSGP, scalar)" begin
    rng = Xoshiro(7)
    b1, o = _learned_block(:r3, 3, 4, 0)
    b2, o = _learned_block(:r1, 1, 5, o)
    K, G = 3, 4
    reference, r2, simplex = o + 1, o + 2, collect(o + 3:o + 5)
    cholesky = collect(o + 6:o + 5 + K * (K - 1) ÷ 2)
    effects = reshape(collect(last(cholesky) .+ (1:K*G)), K, G)
    scales = [BRM._AdaptiveLogScale(0.0, [(1.0, reference)], [(0.5, 0.5, r2)],
                  [BRM._AdaptiveSimplexLog(0.5, simplex, k)]) for k in 1:K]
    ranef = BRM.RanefBlock(:r_r2d2, :correlated, :g, nothing, nothing, collect(1:G),
                           K, G, :z_r2d2, true, false)
    r2d2 = BRM.R2D2AdaptiveCenteringBlock(ranef, 0.0, effects, cholesky, scales)
    o = last(effects)
    M = 6
    hsgp = BRM._HSGPAdaptiveCenteringBlock(:hsgp, :gp_x, fill(0.5, M), collect(o .+ (1:M)),
        [o + M + 1], [0.0], [Inf], o + M + 2, 0.0, Inf,
        reshape([(pi * j / 8)^2 for j in 1:M], M, 1), Float64[])
    o += M + 2
    n = 4
    cells = (; indices=collect(o .+ (1:n)), scales=o + n .+ mod1.(1:n, 2),
               locations=collect(range(-1, 1; length=n)), targets=[0.0, 1.0, 0.0, 1.0])
    d = o + n + 2
    x = 0.5 .* randn(rng, d)
    args = ([b1, b2], [hsgp], [r2d2], cells)
    state, ir = AC_EXT._adaptive_ordered_centering_reparametrizer(args...)
    joint_state, joint = AC_EXT._adaptive_joint_centering_reparametrizer(args...)
    @test WarmupHMC.reparam_controls(ir) == [last(p).source.c for p in joint.pairs]
    @test ir.targets == [last(p).target.c for p in joint.pairs]
    ncell = length(joint.pairs)
    nranef = length(state.ranef.sources)
    # Controls that leave every group in declaration order: equal within a
    # group, or decreasing along it.
    decreasing = reduce(vcat, [repeat([0.9 - 0.2 * (k - 1) for k in 1:b.ranef.n_terms],
                                      b.ranef.n_groups) for b in state.ranef.blocks])
    for ranef_controls in (zeros(nranef), ones(nranef), fill(0.4, nranef), decreasing)
        controls = vcat(ranef_controls, rand(rng, 0:0.1:1, ncell - nranef))
        WarmupHMC.restore_reparam_controls!(ir, controls)
        @test all(o -> all(==(1:size(o, 1)), eachcol(o)), ir.orders)
        @test all(o -> o == 1:length(o), ir.block_orders)
        _set_pair_controls!(joint_state, joint, controls)
        ljac, y = ir(x)
        joint_ljac, joint_y = joint(x)
        @test y ≈ joint_y atol=1e-13
        @test ljac ≈ joint_ljac atol=1e-12
        _, back = WarmupHMC.InverseFunctions.inverse(ir)(y)
        @test back ≈ x atol=1e-12
        # Same frame, so the same per-cell scores.
        frame = AC_EXT._ordered_frame(ir, x, -x)
        joint_frame = AC_EXT._centering_frame(joint_state, x, -x)
        for f in fieldnames(typeof(frame))
            @test getfield(frame, f) ≈ getfield(joint_frame, f) atol=1e-13
        end
    end
    # Permuted R2D2 groups and block orders against the dense reference.
    blocks = state.ranef.blocks
    controls = vcat(rand(rng, 0:0.1:1, nranef), ir.targets[nranef+1:end])
    WarmupHMC.restore_reparam_controls!(ir, controls)
    @test any(o -> o != 1:length(o), ir.block_orders)
    reference_ljac, reference = _learned_reference(x, blocks, controls[1:nranef])
    ljac, y = ir(x)
    ranef_indices = reduce(vcat, [vcat(vec(b.effects), b.cholesky_free) for b in blocks])
    @test y[ranef_indices] ≈ reference[ranef_indices] atol=1e-12
    @test ljac ≈ reference_ljac atol=1e-11
    problem = WarmupHMC.ReparametrizedProblem(ir, _LearnedStdNormal(d), AutoEnzyme())
    @test last(LogDensityProblems.logdensity_and_gradient(problem, x)) ≈
        _learned_fd_gradient(problem, x) rtol=1e-6 atol=1e-6
end

@testset "window rule matches an independent re-implementation" begin
    rng = Xoshiro(11)
    b1, o = _learned_block(:r3, 3, 6, 0)
    b2, o = _learned_block(:r4, 4, 4, o; target=1.0)
    blocks = [b1, b2]
    d = o
    ncell = sum(length(b.effects) for b in blocks)
    _, ir = AC_EXT._adaptive_ordered_centering_reparametrizer(blocks,
        BRM._HSGPAdaptiveCenteringBlock[])
    for trial in 1:3
        controls = rand(rng, 0:0.1:1, ncell)
        WarmupHMC.restore_reparam_controls!(ir, controls)
        positions = 0.7 .* randn(rng, d, 40)
        gradients = -positions .+ 0.3 .* randn(rng, d, 40)
        expected = _learned_reference_select(positions, gradients, blocks, controls)
        changed = AC_EXT._ordered_select!(ir, positions, gradients)
        @test changed === (expected != controls)
        @test WarmupHMC.reparam_controls(ir) == expected
        @test all(o -> all(isperm, eachcol(o)), ir.orders)
    end
    before = WarmupHMC.reparam_controls(ir)
    @test AC_EXT._ordered_select!(ir, zeros(d, 2), zeros(d, 2)) === false
    @test WarmupHMC.reparam_controls(ir) == before
end

@testset "controls interface: validation, snapshot and inverse share" begin
    b1, d = _learned_block(:r3, 3, 4, 0)
    _, ir = AC_EXT._adaptive_ordered_centering_reparametrizer([b1],
        BRM._HSGPAdaptiveCenteringBlock[])
    @test_throws ArgumentError WarmupHMC.restore_reparam_controls!(ir, zeros(11))
    @test_throws ArgumentError WarmupHMC.restore_reparam_controls!(ir, fill(1.5, 12))
    @test_throws ArgumentError WarmupHMC.restore_reparam_controls!(ir, fill(NaN, 12))
    controls = collect(range(0.0, 1.0; length=12))
    WarmupHMC.restore_reparam_controls!(ir, controls)
    snap = WarmupHMC.snapshot_reparametrization(ir)
    inv = WarmupHMC.InverseFunctions.inverse(ir)
    WarmupHMC.restore_reparam_controls!(ir, reverse(controls))
    @test WarmupHMC.reparam_controls(snap) == controls
    @test snap.orders != ir.orders
    @test WarmupHMC.reparam_controls(inv) == reverse(controls)
    @test inv.orders === ir.orders
    x = randn(Xoshiro(1), d)
    @test last(inv(last(ir(x)))) ≈ x atol=1e-12
end

# ---- Compiled models ---------------------------------------------------------

const LEARNED_DATA = (
    group = repeat(["a", "b", "c"], inner=2),
    x = [-0.8, 0.2, -0.3, 0.9, 0.4, -0.6],
    t = [0.1, 0.7, -0.5, 0.2, -0.9, 0.6],
    y = [-0.4, 0.5, -0.1, 0.8, 0.3, -0.2],
)
const LEARNED_CORRELATED = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x + t | p | group)
    y ~ Normal(mu, sigma)
end
const LEARNED_SCALAR = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | group)
    y ~ Normal(mu, sigma)
end

# BridgeStan's own scales and `L`, independent of BRM's accessors.
function _learned_bridgestan_source(problem, block, q)
    K = block.ranef.n_terms
    names = BridgeStan.param_names(problem.model)
    values = BridgeStan.param_constrain(problem.model, q)
    binding = block.ranef.binding
    value(name) = values[only(findall(==(name), names))]
    [value("$(binding)_tau.$k") for k in 1:K],
        [value("$(binding)_L.$i.$j") for i in 1:K, j in 1:K]
end

@testset "compiled K=3 block, target=$(Int(centered))" for centered in (false, true)
    sb = SBBRMI(LEARNED_CORRELATED(LEARNED_DATA); total_groups=(),
                centered_groups=centered ? [:group] : (), mod=@__MODULE__)
    problem = stan_instantiate(sb)
    names = BridgeStan.param_unc_names(problem.model)
    block = only(adaptive_centering_blocks(sb, names))
    @test block.target_c == Float64(centered)
    K, G = size(block.effects)
    q = collect(range(-0.4, 0.6; length=length(names)))
    q[block.log_scales] .= log.([0.7, 1.25, 0.9])
    backend = AutoEnzyme()

    default = adaptive_centering_problem(sb, problem, backend; ranef_order=:learned)
    ir = WarmupHMC.reparametrizer(default)
    @test ir isa AC_EXT.BRMOrderedCenteringReparametrization
    @test WarmupHMC.reparam_controls(ir) == fill(block.target_c, K * G)
    lp0, g0 = LogDensityProblems.logdensity_and_gradient(problem, q)
    lp, g = LogDensityProblems.logdensity_and_gradient(default, q)
    @test lp ≈ lp0 atol=1e-10
    @test g ≈ g0 atol=1e-10

    # Every group in a different order, one group with a tie broken by the
    # block order (mean controls 0.43, 0.5, 0.5: margins 2, 3, 1).
    controls = [0.0, 1.0, 0.6,  0.3, 0.3, 0.9,  1.0, 0.2, 0.0]
    wrapped = adaptive_centering_problem(sb, problem, backend; ranef_order=:learned,
                                         centeredness=controls)
    ir = WarmupHMC.reparametrizer(wrapped)
    @test only(ir.block_orders) == [2, 3, 1]
    @test only(ir.orders) == reshape([2, 3, 1,  3, 2, 1,  1, 2, 3], K, G)
    ljac, mapped = ir(q)
    reference_ljac, reference = _learned_reference(q, [block], controls;
        source=(x, b) -> _learned_bridgestan_source(problem, b, x))
    @test mapped ≈ reference atol=1e-12
    @test ljac ≈ reference_ljac atol=1e-12
    untouched = setdiff(eachindex(names), vec(block.effects), block.cholesky_free)
    @test mapped[untouched] == q[untouched]
    inverse_ljac, roundtrip = WarmupHMC.InverseFunctions.inverse(ir)(mapped)
    @test roundtrip ≈ q atol=1e-12
    @test inverse_ljac ≈ -ljac atol=1e-12
    lp, gradient = LogDensityProblems.logdensity_and_gradient(wrapped, q)
    @test lp ≈ ljac + LogDensityProblems.logdensity(problem, mapped) atol=1e-10
    @test gradient ≈ _learned_fd_gradient(wrapped, q; h=1e-5) atol=2e-5 rtol=2e-5

    # The controls report, against the block metadata and BRM's margin labels.
    rows = adaptive_centering_controls(wrapped, sb)
    @test length(rows) == K * G
    @test [r.index for r in rows] == vec(block.effects)
    @test [r.name for r in rows] == names[vec(block.effects)]
    @test all(r -> r.family === :ranef && r.binding === block.ranef.binding, rows)
    @test [r.c for r in rows] == controls
    @test all(r -> r.target == block.target_c, rows)
    @test [r.level for r in rows] == [block.ranef.levels[g] for g in 1:G for _ in 1:K]
    @test [r.term for r in rows] == repeat(1:K, G)
    @test [r.rank for r in rows] == [3, 1, 2,  3, 2, 1,  1, 2, 3]
    constrained = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    draw = BridgeStan.param_constrain(problem.model, q; include_tp=true, include_gq=true,
                                      rng=BridgeStan.StanRNG(problem.model, 1))
    margins = only(brm_ranef_draws(brm_descriptor(sb), permutedims(draw),
                                   constrained)).margins
    @test [r.margin for r in rows] == repeat(collect(margins), G)
    declared_rows = adaptive_centering_controls(
        adaptive_centering_problem(sb, problem, backend; centeredness=controls), sb)
    @test [r.rank for r in declared_rows] == repeat(1:K, G)
    @test [r.c for r in declared_rows] == controls

    # A restored copy matches a fresh construction and leaves the original.
    copied = deepcopy(wrapped)
    WarmupHMC.restore_reparam_controls!(WarmupHMC.reparametrizer(copied), reverse(controls))
    fresh = adaptive_centering_problem(sb, problem, backend; ranef_order=:learned,
                                       centeredness=reverse(controls))
    @test WarmupHMC.reparametrizer(copied)(q) == WarmupHMC.reparametrizer(fresh)(q)
    @test LogDensityProblems.logdensity_and_gradient(copied, q) ==
        LogDensityProblems.logdensity_and_gradient(fresh, q)
    @test WarmupHMC.reparametrizer(wrapped)(q) == (ljac, mapped)
end

# The snag's shape: one `|kreis|` block shared by three predictors, `s`
# observed in two of six groups. The two formulas differ only in line order.
const ORDER_DATA = let rng = Xoshiro(15564811), G = 6, NS = 2
    alpha = [-1.0, 0.5, 0.2] .+ 0.4 .* randn(rng, 3, G)
    (; g=["K$d" for d in 1:G], s_kreis=collect(1:NS),
       y_s=[alpha[1, d] .+ 0.5 .* randn(rng, 3) for d in 1:NS],
       off_s=[zeros(3) for _ in 1:NS],
       y_r=[alpha[2, d] .+ 0.25 .* randn(rng, 4) for d in 1:G],
       off_r=[zeros(4) for _ in 1:G],
       y_i=[alpha[3, d] .+ 0.25 .* randn(rng, 4) for d in 1:G],
       off_i=[zeros(4) for _ in 1:G])
end
const ORDER_S_FIRST = @brm begin
    alpha_s ~ 1 + (1 | kreis | g)
    alpha_r ~ 1 + (1 | kreis | g)
    alpha_i ~ 1 + (1 | kreis | g)
    sd(:, kreis) ~ Exponential(0.5)
    cor(:, kreis) ~ LKJCholesky(3, 2)
    sigma_s ~ Exponential(1)
    sigma_r ~ Exponential(1)
    sigma_i ~ Exponential(1)
    @plate for k in eachindex(y_s)
        y_s[k] ~ Normal(off_s[k] .+ alpha_s[s_kreis[k]], sigma_s)
    end
    @plate for d in eachindex(alpha_r)
        y_r[d] ~ Normal(off_r[d] .+ alpha_r[d], sigma_r)
        y_i[d] ~ Normal(off_i[d] .+ alpha_i[d], sigma_i)
    end
end
const ORDER_S_LAST = @brm begin
    alpha_r ~ 1 + (1 | kreis | g)
    alpha_i ~ 1 + (1 | kreis | g)
    alpha_s ~ 1 + (1 | kreis | g)
    sd(:, kreis) ~ Exponential(0.5)
    cor(:, kreis) ~ LKJCholesky(3, 2)
    sigma_s ~ Exponential(1)
    sigma_r ~ Exponential(1)
    sigma_i ~ Exponential(1)
    @plate for k in eachindex(y_s)
        y_s[k] ~ Normal(off_s[k] .+ alpha_s[s_kreis[k]], sigma_s)
    end
    @plate for d in eachindex(alpha_r)
        y_r[d] ~ Normal(off_r[d] .+ alpha_r[d], sigma_r)
        y_i[d] ~ Normal(off_i[d] .+ alpha_i[d], sigma_i)
    end
end

@testset "the learned order removes the declaration order from the sampled density" begin
    # Same controls per margin and group. Margin `s` ranks last in the block
    # and in every group but the first, where it ranks second.
    control(margin, g) = margin === :alpha_r ? 1.0 : margin === :alpha_i ? 0.8 :
        g == 1 ? 0.9 : g == 2 ? 0.5 : 0.0
    models = map((ORDER_S_FIRST, ORDER_S_LAST)) do builder
        sb = SBBRMI(builder(ORDER_DATA); mod=@__MODULE__)
        problem = stan_instantiate(sb)
        names = BridgeStan.param_unc_names(problem.model)
        block = only(adaptive_centering_blocks(sb, names))
        K, G = size(block.effects)
        margins = [row.margin.predictor for row in adaptive_centering_controls(
            adaptive_centering_problem(sb, problem, AutoEnzyme()), sb)[1:K]]
        controls = [control(margins[k], g) for g in 1:G for k in 1:K]
        wrapped = adaptive_centering_problem(sb, problem, AutoEnzyme();
            ranef_order=:learned, centeredness=controls)
        (; problem, names, block, margins, wrapped)
    end
    first_, last_ = models
    @test first_.margins == [:alpha_s, :alpha_r, :alpha_i]
    @test last_.margins == [:alpha_r, :alpha_i, :alpha_s]
    @test only(WarmupHMC.reparametrizer(first_.wrapped).block_orders) == [2, 3, 1]
    @test only(WarmupHMC.reparametrizer(last_.wrapped).block_orders) == [1, 2, 3]
    # The s-first model's coordinate `i` is the s-last model's `into[i]`: the
    # block's scales and effects by margin, its Cholesky coordinates by slot
    # (both spell the correlation in the order r, i, s), the rest by name.
    into = zeros(Int, length(first_.names))
    for (k, margin) in enumerate(first_.margins)
        k_last = findfirst(==(margin), last_.margins)
        into[first_.block.log_scales[k]] = last_.block.log_scales[k_last]
        into[first_.block.effects[k, :]] = last_.block.effects[k_last, :]
    end
    into[first_.block.cholesky_free] = last_.block.cholesky_free
    for (i, name) in enumerate(first_.names)
        into[i] == 0 && (into[i] = only(findall(==(name), last_.names)))
    end
    @test isperm(into)
    rng = Xoshiro(2)
    points = [0.4 .* randn(rng, length(into)) for _ in 1:3]
    evaluated = map(points) do x
        y = similar(x)
        y[into] = x
        (LogDensityProblems.logdensity_and_gradient(first_.wrapped, x),
         LogDensityProblems.logdensity_and_gradient(last_.wrapped, y))
    end
    for ((lp_first, g_first), (lp_last, g_last)) in evaluated
        @test g_first ≈ g_last[into] rtol=1e-9 atol=1e-9
        @test lp_first - first(first(evaluated))[1] ≈
            lp_last - last(first(evaluated))[1] atol=1e-9
    end
end

@testset "a model without a correlated block gets the declared wrapper" begin
    sb = SBBRMI(LEARNED_SCALAR(LEARNED_DATA); total_groups=(), mod=@__MODULE__)
    problem = stan_instantiate(sb)
    learned = adaptive_centering_problem(sb, problem, AutoEnzyme(); ranef_order=:learned,
                                         centeredness=0.4)
    declared = adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.4)
    @test WarmupHMC.reparametrizer(learned) isa WarmupHMC.IndexedReparametrization
    @test WarmupHMC.reparam_sources(learned) == WarmupHMC.reparam_sources(declared)
    q = collect(range(-0.3, 0.5; length=LogDensityProblems.dimension(problem)))
    @test LogDensityProblems.logdensity_and_gradient(learned, q) ==
        LogDensityProblems.logdensity_and_gradient(declared, q)
    @test_throws ArgumentError adaptive_centering_problem(sb, problem, AutoEnzyme();
                                                          ranef_order=:sorted)
end

@testset "WarmupHMC runs, checkpoints and resumes the learned order" begin
    sb = SBBRMI(LEARNED_CORRELATED(LEARNED_DATA); total_groups=(), mod=@__MODULE__)
    problem = stan_instantiate(sb)
    names = BridgeStan.param_unc_names(problem.model)
    dir = mktempdir()
    settings = (; n_evaluations=120, stepsize_adaptation_limit=20, max_tree_depth=7,
                progress=nothing, monitor_ess=false, checkpoint_dir=dir)
    online = adaptive_centering_problem(sb, problem, AutoEnzyme(); ranef_order=:learned)
    fit = adaptive_warmup_mcmc(Xoshiro(0x15564811), online; n_draws=20, settings...)
    controls = WarmupHMC.reparam_controls(WarmupHMC.reparametrizer(online))
    @test length(controls) == 9
    @test all(in(0.0:0.1:1.0), controls)
    @test all(isfinite, fit.posterior_position)
    payload = deserialize(joinpath(dir, "cp_latest.jls"))
    @test payload.reparam_sources == controls
    # Returned draws are in the compiled frame: BridgeStan constrains them.
    @test all(x -> all(isfinite, BridgeStan.param_constrain(problem.model, collect(x))),
              eachcol(fit.posterior_position))
    resumed = adaptive_centering_problem(sb, problem, AutoEnzyme(); ranef_order=:learned)
    refit = adaptive_warmup_mcmc(Xoshiro(1), resumed; n_draws=40, resume=true, settings...)
    @test size(refit.posterior_position, 2) >= 40
    @test all(in(0.0:0.1:1.0), WarmupHMC.reparam_controls(WarmupHMC.reparametrizer(resumed)))
    rows = adaptive_centering_controls(resumed, sb)
    @test [r.c for r in rows] ==
        WarmupHMC.reparam_controls(WarmupHMC.reparametrizer(resumed))
    _, _, groups = _learned_orders(reshape([r.c for r in rows], 3, :))
    @test [r.rank for r in rows] == reduce(vcat, invperm.(groups))
end
