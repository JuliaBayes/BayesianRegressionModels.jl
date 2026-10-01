using Test, Random, Distributions
using BayesianRegressionModels, StanBlocks
const BRM = BayesianRegressionModels

# Regression gate for snag brm-totals-lower-a47c6227.
#
# Previously, totals construction installed a scale-prior family through
# Core.eval, and a same-frame trace could miss its fresh dispatch hooks.
# Composed priors now use ValueFamily data. Both direct StanBlocks tracing
# and BRM's supported newest-world wrappers must work in the constructing
# frame, including the very first use of a positive vector-prior shape.
#
# Order is load-bearing: the in-function traces below MUST run before any
# top-level trace of this shape in this process. A prior top-level trace
# populates `_SB_VECTOR_PRIOR_CACHE` and would miss a cold-construction
# regression.

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
    direct_src = StanBlocks.stan_code(sb.model)
    src = BRM.stan_code(sb)
    @test direct_src == src
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
