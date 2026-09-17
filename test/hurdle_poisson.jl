# test/hurdle_poisson.jl — exact Julia and Stan contracts for HurdlePoisson.
#
# Run on a capable host:
#   julia --startup-file=no --project=. test/hurdle_poisson.jl

using Test
using BayesianRegressionModels
using Distributions
using LogDensityProblems
using LogExpFunctions: logit
using Random
using Statistics
using StanBlocks

@testset "HurdlePoisson is an executable Distributions.jl distribution" begin
    lambda = 2.3
    p_zero = 0.35
    d = HurdlePoisson(lambda, p_zero)

    @test d isa DiscreteUnivariateDistribution
    @test params(d) == (lambda, p_zero)
    @test minimum(d) == 0
    @test maximum(d) == Inf
    @test logpdf(d, 0) == log(p_zero)
    for y in 1:8
        expected = log1p(-p_zero) + logpdf(Poisson(lambda), y) -
                   log1p(-exp(-lambda))
        @test logpdf(d, y) ≈ expected atol=1e-14
    end
    @test logpdf(d, -1) == -Inf
    @test logpdf(d, 1.5) == -Inf
    @test sum(pdf(d, y) for y in 0:100) ≈ 1.0 atol=1e-14

    @test_throws DomainError HurdlePoisson(0.0, p_zero)
    @test_throws DomainError HurdlePoisson(-1.0, p_zero)
    @test_throws DomainError HurdlePoisson(Inf, p_zero)
    @test_throws DomainError HurdlePoisson(lambda, -0.01)
    @test_throws DomainError HurdlePoisson(lambda, 1.01)
    @test_throws DomainError HurdlePoisson(lambda, NaN)

    only_positive = HurdlePoisson(lambda, 0.0)
    only_zero = HurdlePoisson(lambda, 1.0)
    rng = MersenneTwister(20260729)
    @test all(rand(rng, only_positive, 256) .> 0)
    @test all(iszero, rand(rng, only_zero, 256))

    draws = rand(rng, d, 20_000)
    @test mean(iszero, draws) ≈ p_zero atol=0.015
    positives = filter(!iszero, draws)
    @test mean(positives) ≈ lambda / (1 - exp(-lambda)) atol=0.05
end

semantic_df = (; y=[0, 1, 2, 5, 0, 3])
semantic_builder = @brm begin
    y ~ HurdlePoisson(2.3, 0.35)
end

@testset "HurdlePoisson lowering is exact on every Stan path" begin
    vbrmi = VBRMI(semantic_builder(semantic_df))
    @test LogDensityProblems.dimension(vbrmi) == 0
    @test LogDensityProblems.logdensity(vbrmi, Float64[]) ≈
          sum(logpdf.(HurdlePoisson(2.3, 0.35), semantic_df.y)) atol=1e-10

    descriptor = brm_descriptor(semantic_builder, semantic_df; mod=@__MODULE__)
    code = brm_execute(descriptor, :transpile)

    @test occursin("y ~ hurdle_poisson(2.3, 0.35);", code)
    @test occursin("poisson_lpmf", code)
    @test occursin("poisson_lccdf", code)
    @test occursin("bernoulli_rng", code)
    @test occursin("poisson_rng", code)
    @test occursin("while", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    plan = generative_plan(semantic_builder, semantic_df; mod=@__MODULE__)
    declaration = only(d for d in plan.declarations if d.target === :y)
    @test declaration.family === :hurdle_poisson
    @test declaration.role === :observation
    @test !isnothing(declaration.draw)

    cache = joinpath(tempdir(), "brm-hurdle-poisson")
    isdir(cache) || mkpath(cache)
    problem = brm_execute(
        descriptor, :instantiate;
        path=joinpath(cache, string(descriptor.id) * ".stan"))
    pointwise = brm_execute(
        descriptor, :pointwise_loglik;
        problem, draws=Float64[], seed=20260729)
    expected = logpdf.(HurdlePoisson(2.3, 0.35), semantic_df.y)
    @test pointwise.y_likelihood ≈ expected atol=1e-10

    predictions = brm_execute(
        descriptor, :predict;
        problem, draws=zeros(0, 4096), seed=20260729)
    draws = vec(predictions.y_gen)
    @test mean(iszero, draws) ≈ 0.35 atol=0.015
    positives = filter(!iszero, draws)
    @test all(>(0), positives)
    @test mean(positives) ≈ 2.3 / (1 - exp(-2.3)) atol=0.05
end

regression_df = (;
    x=[-1.0, -0.25, 0.5, 1.0, 1.5, 2.0],
    y=[0, 1, 2, 0, 4, 3],
)
regression_builder = @brm begin
    log(lambda) ~ 1 + x
    logit(p_zero) ~ 1
    y ~ HurdlePoisson(lambda, p_zero)
end

@testset "HurdlePoisson accepts distributional lambda and p_zero" begin
    descriptor = brm_descriptor(regression_builder, regression_df; mod=@__MODULE__)
    code = brm_execute(descriptor, :transpile)
    @test occursin("y ~ hurdle_poisson(lambda, p_zero);", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end
