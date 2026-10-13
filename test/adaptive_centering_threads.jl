# Concurrent reverse-mode gradients through the adaptive-centering
# reparametrizers. On Julia 1.10, Enzyme's reverse pass corrupts Julia's GC once
# another thread collects while a differentiated accessor allocates a heap work
# vector, reads a struct holding vectors inside its row loop, or reaches two
# `_block_location` instantiations (see `_block_location` in
# BayesianRegressionModelsWarmupHMCExt). Completion and multi-chain workers do
# exactly that, and a regression kills the whole process — a segfault in
# `getindex` or "GC error (probable corruption)" — so the stress runs in a child
# Julia with four threads. This file asserts the child's clean exit and that
# every concurrent gradient bitwise-equals its single-threaded reference.
#
#     julia --project=test test/adaptive_centering_threads.jl
#
# Synthetic toy blocks only; no compiled Stan model is involved.
using BayesianRegressionModels, WarmupHMC, Enzyme, LogDensityProblems, Random
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
const AC_EXT = Base.get_extension(BRM, :BayesianRegressionModelsWarmupHMCExt)
const _STRESS_WORKER = "--stress-worker" in ARGS
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

function _ordinary_problem()
    blocks, d = _ordinary_blocks()
    _with_controls(AC_EXT._adaptive_centering_reparametrizer(blocks)..., d)
end

# Ordinary + R2D2 + squared-exponential HSGP cells in one joint wrapper. Every
# R2D2 term's log-scale carries a linear, a unit-interval and a simplex atom.
function _joint_problem()
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
    _with_controls(AC_EXT._adaptive_joint_centering_reparametrizer(
        blocks, [hsgp], [r2d2])..., d)
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

# Every worker differentiates its own deepcopy, as completion workers do, while
# another task churns the allocator and forces collections.
function _stress(problem, d)
    points = [0.3 .* randn(Xoshiro(10 + i), d) for i in 1:8]
    reference = [LogDensityProblems.logdensity_and_gradient(problem, x) for x in points]
    stop = Threads.Atomic{Bool}(false)
    churn = Threads.@spawn while !stop[]
        junk = [rand(64) for _ in 1:100]
        rand() < 0.01 && GC.gc(false)
        yield()
    end
    mismatches = Threads.Atomic{Int}(0)
    workers = map(1:Threads.nthreads() - 1) do t
        Threads.@spawn begin
            local_problem = deepcopy(problem)
            for i in 1:_GRADIENTS_PER_WORKER
                j = mod1(i + t, length(points))
                value = LogDensityProblems.logdensity_and_gradient(local_problem, points[j])
                isequal(value, reference[j]) || Threads.atomic_add!(mismatches, 1)
            end
        end
    end
    foreach(wait, workers)
    stop[] = true
    wait(churn)
    (Threads.nthreads() - 1) * _GRADIENTS_PER_WORKER, mismatches[]
end

if _STRESS_WORKER
    Threads.nthreads() >= 2 || (println(stderr, "stress worker needs --threads >= 2"); exit(2))
    for (name, build) in (("ordinary", _ordinary_problem), ("joint", _joint_problem),
                          ("s2z", _s2z_problem))
        gradients, mismatches = _stress(build()...)
        println("STRESS ", name, " gradients=", gradients, " mismatches=", mismatches)
        flush(stdout)
    end
    exit(0)
end

using Test

@testset "adaptive-centering gradients stay exact under concurrent GC" begin
    project = "--project=$(Base.active_project())"
    cmd = `$(Base.julia_cmd()) --startup-file=no --threads=4 $project $(@__FILE__) --stress-worker`
    out = IOBuffer()
    process = run(pipeline(ignorestatus(cmd); stdout=out, stderr=out))
    log = String(take!(out))
    clean = process.exitcode == 0 && process.termsignal == 0
    clean || println(log)
    @test clean
    for name in ("ordinary", "joint", "s2z")
        line = match(Regex("STRESS $name gradients=(\\d+) mismatches=(\\d+)"), log)
        @test !isnothing(line)
        isnothing(line) && continue
        @test parse(Int, line[1]) == 3 * _GRADIENTS_PER_WORKER
        @test parse(Int, line[2]) == 0
    end
end
