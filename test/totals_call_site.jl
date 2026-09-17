using Test, Random, Distributions
using BayesianRegressionModels, StanBlocks
const BRM = BayesianRegressionModels

# Regression gate for snag brm-totals-lower-a47c6227.
#
# A totals-path model registers its composed scale-prior family
# (`brm_vector_prior_*` @deffun triad + `lpxf_expr` hook) via `Core.eval`
# DURING `SBBRMI` construction. Tracing the model in the SAME compiled caller
# frame resolves methods at that frame's world age, so the fresh hooks are
# invisible there and tracing dies with
# "`brm_vector_prior_*` is missing `lpxf_expr`" — while an identical top-level
# trace succeeds. The supported trace/compile entries (`stan_code`,
# `stan_model`, `stan_instantiate` on the `SBBRMI`) re-enter the compiler in
# the current world via `Base.invokelatest`; direct
# `StanBlocks.stan_code(sb.model)` / `StanBlocks.stan_instantiate(sb.model)`
# from inside such a frame remains a world-age error by Julia semantics.
#
# Order is load-bearing: the in-function traces below MUST run before any
# top-level trace of this shape in this process. A prior top-level trace
# populates `_SB_VECTOR_PRIOR_CACHE`, so no new family is registered and the
# staleness never manifests.

function build_radonlike()
    rng = Xoshiro(1234)
    n, J = 919, 85
    weights = rand(rng, J)
    weights ./= sum(weights)
    county = Vector{Int}(undef, n)
    for i in 1:n
        r = rand(rng)
        c = 1
        acc = weights[1]
        while acc < r && c < J
            c += 1
            acc += weights[c]
        end
        county[i] = c
    end
    floor = Float64.(rand(rng, n) .< 0.7)
    log_radon = randn(rng, n) .+ 1.0
    (; log_radon=Float64.(log_radon), floor=Float64.(floor), county=county)
end

builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + (1 | county)
    log_radon ~ Normal(mu, sigma)
end

# Build + trace entirely inside compiled frames: the FIRST lowering of this
# shape in the process. Returns the traced sources for the call-site
# byte-identity check below.
function build_and_trace_in_function(builder, data)
    sb = SBBRMI(builder(data); mod=@__MODULE__)
    @test !isempty(total_effect_blocks(sb))
    src = BRM.stan_code(sb)
    traced = BRM.stan_model(sb)
    (; sb, src, traced)
end

@testset "totals lowering is call-site independent" begin
    data = build_radonlike()
    infunc = build_and_trace_in_function(builder, data)
    @test infunc.src isa String
    @test !isempty(infunc.src)
    @test occursin("brm_vector_prior_", infunc.src)
    @test infunc.traced isa StanBlocks.StanModel
    # Compile entry exists on the SBBRMI (same invokelatest pattern as the
    # trace entries; the C++ compile itself is covered by
    # test/total_effects_integration.jl at top level).
    @test applicable(BRM.stan_instantiate, infunc.sb)

    # Same builder + same data at top level: identical Stan source.
    sb_top = SBBRMI(builder(data); mod=@__MODULE__)
    @test BRM.stan_code(sb_top) == infunc.src
    @test StanBlocks.stan_code(sb_top.model) == infunc.src
end
