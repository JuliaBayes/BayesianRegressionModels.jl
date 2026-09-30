# test/sbimpl_cache_concurrency.jl — gate for synchronized generated-family
# cache misses in sbimpl (original snag sbimpl-generated-54b0af30).
#
# `_sb_vector_prior_family` / `_sb_mixture_family`
# now construct ValueFamily values without installing Julia bindings or
# methods. Every lookup and miss remains synchronized by `_SB_GEN_LOCK`, so
# one task constructs the family and the rest read the same cached value.
# Horseshoe submodels now construct private AST values without registration.
#
# The race needs true thread parallelism to manifest; the hammer below only
# stresses it with more than one thread:
#   julia --threads=4 --project=test test/sbimpl_cache_concurrency.jl
# Single-threaded it still validates cache identity (same family object).

using Test
using BayesianRegressionModels
using Distributions: Exponential

const BRM = BayesianRegressionModels

# Run `f` on `n` tasks released at once so every task misses the (fresh-key)
# cache simultaneously; return all fetched results.
function sbcc_hammer(f, n)
    ready = Threads.Atomic{Int}(0)
    go = Threads.Atomic{Bool}(false)
    tasks = map(1:n) do _
        Threads.@spawn begin
            Threads.atomic_add!(ready, 1)
            while !go[]
                yield()
            end
            f()
        end
    end
    while ready[] < n
        yield()
    end
    go[] = true
    fetch.(tasks)
end

@testset "sbimpl generated-cache concurrency" begin
@testset "generated caches are guarded by one reentrant lock" begin
    @test isdefined(BRM, :_SB_GEN_LOCK)
    if isdefined(BRM, :_SB_GEN_LOCK)
        @test BRM._SB_GEN_LOCK isa ReentrantLock
    end
end

@testset "concurrent vector-prior miss defines the family once" begin
    # Fresh key in this process: a heterogeneous Exponential pair with
    # distinctive rates no other testset in this file touches.
    priors = [ExprColumn(Exponential, 2.5), ExprColumn(Exponential, 0.25)]
    n = 8 * max(Threads.nthreads(), 1)
    results = sbcc_hammer(() -> BRM._sb_vector_prior_family(priors), n)
    families = first.(results)
    @test all(f -> f === first(families), families)
    # A sequential repeat call hits the cache and returns the identical family.
    family_again, _ = BRM._sb_vector_prior_family(priors)
    @test family_again === first(families)
end

@testset "concurrent mixture miss defines the family once" begin
    # Fresh key in this process: 2-component continuous Normal mixture.
    args = (:normal, 2, (), 2, false)
    n = 8 * max(Threads.nthreads(), 1)
    families = sbcc_hammer(() -> BRM._sb_mixture_family(args...), n)
    @test all(f -> f === first(families), families)
    @test BRM._sb_mixture_family(args...) === first(families)
end
end
