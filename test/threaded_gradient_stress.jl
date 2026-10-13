# Shared harness for the concurrent-GC gradient gates
# (`adaptive_centering_threads.jl`, `rk_enzyme_threads.jl`).
#
# Enzyme's reverse pass can turn a differentiated function into a use-after-free
# once another thread runs GC — measured on Julia 1.10 and 1.12 (upstream,
# unfixed; see `_block_location` in BayesianRegressionModelsWarmupHMCExt). WarmupHMC's
# multi-chain and completion samplers differentiate one `deepcopy` of the density
# per chain on `Threads.@threads`, and a regression kills the whole process — a
# segfault or "GC error (probable corruption)" — so each gate runs its stress in a
# child Julia with four threads and asserts the child's clean exit and that every
# concurrent gradient bitwise-equals its single-threaded reference.
#
# A gate file defines `cases`, a tuple of `name => build` pairs where `build()`
# returns `(problem, dimension)`, or `(problem, dimension, worker_copy)` when a
# worker's own copy cannot be a `deepcopy`, and ends with
#
#     run_threaded_gradient_gate(@__FILE__, cases, budget)
#
# `budget` is gradients per worker (an `Integer`) or seconds per case (a float):
# slow, rarely crashing paths need wall time rather than a fixed count.
#
# A path that still corrupts for a reason outside BRM passes
# `known_broken = "<why>"`: the gate then records a broken test naming that
# reason instead of running, unless `BRM_STRESS_KNOWN_BROKEN=1` asks for the
# stress (the acceptance check of the upstream fix).
using LogDensityProblems, Random, Test

# Why a path's gate is known broken when the cause is outside BRM.
const ENZYME_CONCURRENT_GC = "Enzyme's reverse pass corrupts memory when another " *
    "thread runs GC, on Julia 1.10 and 1.12 (Enzyme snag julia-1-10-concu-152a4415)."

# Every worker differentiates its own copy, as the samplers' chains do, while
# another task churns the allocator and forces collections.
_keep_going(budget::Integer, i, deadline) = i < budget
_keep_going(budget::AbstractFloat, i, deadline) = i == 0 || time() < deadline

function threaded_gradient_stress(problem, d, budget, worker_copy=deepcopy)
    points = [0.3 .* randn(Xoshiro(10 + i), d) for i in 1:8]
    reference = [LogDensityProblems.logdensity_and_gradient(problem, x) for x in points]
    stop = Threads.Atomic{Bool}(false)
    churn = Threads.@spawn while !stop[]
        junk = [rand(64) for _ in 1:100]
        rand() < 0.01 && GC.gc(false)
        yield()
    end
    gradients = Threads.Atomic{Int}(0)
    mismatches = Threads.Atomic{Int}(0)
    workers = map(1:Threads.nthreads() - 1) do t
        Threads.@spawn begin
            local_problem = worker_copy(problem)
            # A rebuilt copy compiles on first use; keep that out of the budget.
            first_value = LogDensityProblems.logdensity_and_gradient(local_problem, points[t])
            isequal(first_value, reference[t]) || Threads.atomic_add!(mismatches, 1)
            deadline = time() + (budget isa AbstractFloat ? budget : 0.0)
            i = 0
            while _keep_going(budget, i, deadline)
                i += 1
                j = mod1(i + t, length(points))
                value = LogDensityProblems.logdensity_and_gradient(local_problem, points[j])
                Threads.atomic_add!(gradients, 1)
                isequal(value, reference[j]) || Threads.atomic_add!(mismatches, 1)
            end
        end
    end
    foreach(wait, workers)
    stop[] = true
    wait(churn)
    gradients[], mismatches[]
end

# In the child (`--stress-worker`), stress every case and exit; in the parent,
# launch that child and check its report.
function run_threaded_gradient_gate(file, cases, budget; known_broken=nothing)
    if !isnothing(known_broken) &&
            get(ENV, "BRM_STRESS_KNOWN_BROKEN", "") != "1" && !("--stress-worker" in ARGS)
        @testset "$(basename(file)): known broken" begin
            @info "Not stressed: $known_broken " *
                "Set BRM_STRESS_KNOWN_BROKEN=1 to run it."
            @test_broken false
        end
        return
    end
    if "--stress-worker" in ARGS
        Threads.nthreads() >= 2 ||
            (println(stderr, "stress worker needs --threads >= 2"); exit(2))
        for (name, build) in cases
            built = build()
            gradients, mismatches = threaded_gradient_stress(
                built[1], built[2], budget, get(built, 3, deepcopy))
            println("STRESS ", name, " gradients=", gradients, " mismatches=", mismatches)
            flush(stdout)
        end
        exit(0)
    end
    @testset "$(basename(file)): gradients stay exact under concurrent GC" begin
        project = "--project=$(Base.active_project())"
        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=4 $project $file --stress-worker`
        out = IOBuffer()
        process = run(pipeline(ignorestatus(cmd); stdout=out, stderr=out))
        log = String(take!(out))
        clean = process.exitcode == 0 && process.termsignal == 0
        clean || println(log)
        @test clean
        for (name, _) in cases
            line = match(Regex("STRESS $name gradients=(\\d+) mismatches=(\\d+)"), log)
            @test !isnothing(line)
            isnothing(line) && continue
            gradients = parse(Int, line[1])
            budget isa Integer ? @test(gradients == 3 * budget) : @test(gradients > 0)
            @test parse(Int, line[2]) == 0
        end
    end
end
