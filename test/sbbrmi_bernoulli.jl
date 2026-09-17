# test/sbbrmi_bernoulli.jl — SBBRMI Bernoulli/BernoulliLogit regression gate.
#
# Snag brm-sbbrmi-berno-18aeccfe: `y ~ BernoulliLogit(mu)` with a float 0/1
# response (`y = [0.0, 1.0]`) died inside the StanBlocks tracer with
# `tracetype not defined for bernoulli_logit_rng(::array[] tokenof, ::vector)`
# because the real observation produced a non-`int` sized-token GQ draw that
# matches no `bernoulli_logit_rng` overload. SBBRMI now coerces 0/1 responses
# (Bool/Integer/float) to `int` data at lowering, so density, pointwise
# log-lik and RNG paths lower exactly as for integer responses.
# Transpile + stanc + Stan-shape gate (mirrors the family_surfaces SBBRMI
# testset); no BridgeStan compile.

using Test
using BayesianRegressionModels
using StanBlocks
import StanBlocks.stan: transpiles

berno_data(; y=[0.0, 1.0, 1.0, 0.0]) =
    (; y=y, x=[-0.5, 0.5, -0.25, 0.75])
berno_builder(df) = (@brm begin
    mu ~ 1 + x
    y ~ BernoulliLogit(mu)
end)(df)
berno_prob_builder(df) = (@brm begin
    mu ~ 1 + x
    y ~ Bernoulli(mu)
end)(df)
berno_cellmeans_builder(df) = (@brm begin
    mu ~ 0 + gg + x
    y ~ BernoulliLogit(mu)
end)(merge(df, (; gg=[1, 2, 1, 2])))

berno_code(brmi) = begin
    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)
    @test transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    code
end

@testset "SBBRMI BernoulliLogit regression lowers density, pointwise and RNG" begin
    float_code = berno_code(berno_builder(berno_data()))
    int_code = berno_code(berno_builder(berno_data(; y=[0, 1, 1, 0])))
    bool_code = berno_code(berno_builder(berno_data(; y=[false, true, true, false])))

    # Coercion is Stan-preserving: float/Bool 0/1 lower to the identical
    # program as integer 0/1.
    @test float_code == int_code
    @test bool_code == int_code

    # Density + pointwise log-lik + RNG paths are all present.
    @test occursin("y ~ bernoulli_logit(mu);", float_code)
    @test occursin("bernoulli_logit_lpmf(", float_code)
    @test occursin("bernoulli_logit_rng(", float_code)
    @test occursin("y_gen", float_code)
    @test occursin("y_likelihood", float_code)

    # The probability-form family lowers the same triad.
    prob_code = berno_code(berno_prob_builder(berno_data(; y=[0, 1, 1, 0])))
    prob_float_code = berno_code(berno_prob_builder(berno_data()))
    @test prob_float_code == prob_code
    @test occursin("y ~ bernoulli(mu);", prob_code)
    @test occursin("bernoulli_rng(", prob_code)
    @test occursin("y_gen", prob_code)
    @test occursin("y_likelihood", prob_code)

    # Cell-means form lowers too (same likelihood path, different predictor).
    cell_code = berno_code(berno_cellmeans_builder(berno_data()))
    @test occursin("y ~ bernoulli_logit(mu);", cell_code)
    @test occursin("bernoulli_logit_rng(", cell_code)
    @test StanBlocks.stanc_check(cell_code; warn_pedantic=false).ok
end

@testset "SBBRMI Bernoulli responses outside 0/1 fail loudly" begin
    bad = berno_data(; y=[0.0, 0.5, 1.0, 0.0])
    @test_throws "must contain only 0/1 values" SBBRMI(berno_builder(bad); mod=@__MODULE__)
    bad_prob = berno_data(; y=[0, 2, 1, 0])
    @test_throws "must contain only 0/1 values" SBBRMI(berno_prob_builder(bad_prob); mod=@__MODULE__)
end
