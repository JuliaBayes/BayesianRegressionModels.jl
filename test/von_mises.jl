# test/von_mises.jl — exact Distributions.jl and principal-interval contracts.
#
# Run on a capable host:
#   julia --startup-file=no --project=. test/von_mises.jl

using Test
using BayesianRegressionModels
using Distributions
using Random
using StanBlocks

const BRM = BayesianRegressionModels

@testset "VonMises constructor semantics stay Julia-native" begin
    @test BRM._sb_stan_dist_name(VonMises) === nothing
    @test BRM._sb_stan_dist_args(VonMises, (:kappa,)) == (0.0, :kappa)
    @test BRM._sb_stan_dist_args(VonMises, (:mu, :kappa)) == (:mu, :kappa)
end

@testset "CircularVonMises has an executable Julia distribution contract" begin
    circular = CircularVonMises(7.0, 2.1; interval=(0.0, 2pi))
    @test circular isa ContinuousUnivariateDistribution
    @test params(circular) == (7.0, 2.1, (0.0, 2pi))
    @test CircularVonMises(params(circular)...) isa CircularVonMises
    @test minimum(circular) == 0.0
    @test maximum(circular) == 2pi
    @test insupport(circular, 0.0)
    @test !insupport(circular, 2pi)
    @test logpdf(circular, 2pi) == -Inf

    wrapped_mu = mod(7.0, 2pi)
    moving_lo = wrapped_mu - pi
    observations = [0.0, 2.0, 6.0]
    representatives = moving_lo .+ mod.(observations .- moving_lo, 2pi)
    @test logpdf.(Ref(circular), observations) ≈
          logpdf.(VonMises(wrapped_mu, 2.1), representatives)

    draws = rand(MersenneTwister(20260729), circular, 128)
    @test all(insupport.(Ref(circular), draws))
    @test_throws DomainError CircularVonMises(0.0, 0.0)
    @test_throws DomainError CircularVonMises(0.0, 1.0; interval=(0.0, 1.0))
    @test_throws ArgumentError CircularVonMises(0.0, 1.0; interval=0.0)
end

semantic_df = (;
    y_exact=[0.3 - pi, 0.3, 0.3 + pi],
    y_zero=[-pi, 0.0, pi],
    y_circular=[0.0, 2.0, 6.0],
)

semantic_builder = @brm begin
    y_exact ~ VonMises(0.3, 1.7)
    y_zero ~ VonMises(1.7)
    y_circular ~ CircularVonMises(7.0, 2.1;
        interval=(0.0, 6.283185307179586))
end

@testset "exact and principal-interval lowering use native Stan faithfully" begin
    descriptor = brm_descriptor(semantic_builder, semantic_df; mod=@__MODULE__)
    code = brm_execute(descriptor, :transpile)

    @test occursin("von_mises_lpdf", code)
    @test occursin("von_mises_rng", code)
    @test occursin(
        "y_exact ~ brm_von_mises(0.3, 1.7, 0.0, 0.0, 0);", code)
    @test occursin(
        "y_zero ~ brm_von_mises(0.0, 1.7, 0.0, 0.0, 0);", code)
    @test occursin(
        "y_circular ~ brm_von_mises(7.0, 2.1, 0.0, 6.283185307179586, 1);",
        code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    cache = joinpath(tempdir(), "brm-von-mises-semantics")
    isdir(cache) || mkpath(cache)
    problem = brm_execute(
        descriptor, :instantiate;
        path=joinpath(cache, string(descriptor.id) * ".stan"))
    pointwise = brm_execute(
        descriptor, :pointwise_loglik;
        problem, draws=Float64[], seed=20260729)

    @test pointwise.y_exact_likelihood ≈
          logpdf.(VonMises(0.3, 1.7), semantic_df.y_exact) atol=1e-10
    @test pointwise.y_zero_likelihood ≈
          logpdf.(VonMises(1.7), semantic_df.y_zero) atol=1e-10

    wrapped_mu = mod(7.0, 2pi)
    moving_lo = wrapped_mu - pi
    circular_representatives =
        moving_lo .+ mod.(semantic_df.y_circular .- moving_lo, 2pi)
    @test pointwise.y_circular_likelihood ≈
          logpdf.(VonMises(wrapped_mu, 2.1), circular_representatives) atol=1e-10

    predictions = brm_execute(
        descriptor, :predict;
        problem, draws=zeros(0, 64), seed=20260729)
    @test all((0.3 - pi) .<= predictions.y_exact_gen .< (0.3 + pi))
    @test all(-pi .<= predictions.y_zero_gen .< pi)
    @test all(0.0 .<= predictions.y_circular_gen .< 2pi)
end

regression_df = (;
    x=[-1.0, -0.25, 0.5, 1.0],
    y=[-2.8, -0.4, 1.1, 2.9],
)
regression_builder = @brm begin
    mu ~ 1 + x
    log(kappa) ~ 1
    y ~ CircularVonMises(mu, kappa; interval=(-pi, pi))
end

@testset "CircularVonMises is a regression family" begin
    code = brm_execute(
        brm_descriptor(regression_builder, regression_df; mod=@__MODULE__),
        :transpile)
    @test occursin("y ~ brm_von_mises(mu, kappa", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "von-Mises validation is early and explicit" begin
    circular = @brm begin
        y ~ CircularVonMises(0.0, 1.0; interval=(-pi, pi))
    end
    @test_throws ErrorException brm_descriptor(
        circular, (; y=[pi]); mod=@__MODULE__)

    exact = @brm begin
        y ~ VonMises(0.0, 1.0)
    end
    @test_throws ErrorException brm_descriptor(
        exact, (; y=[pi + 0.01]); mod=@__MODULE__)

    zero_kappa = @brm begin
        y ~ VonMises(0.0, 0.0)
    end
    @test_throws ErrorException brm_descriptor(
        zero_kappa, (; y=[0.0]); mod=@__MODULE__)

    bad_width = @brm begin
        y ~ CircularVonMises(0.0, 1.0; interval=(0.0, 1.0))
    end
    @test_throws ErrorException bad_width((; y=[0.5]))

    dynamic_interval = @brm begin
        y ~ CircularVonMises(0.0, 1.0; interval=(lo, hi))
    end
    @test_throws ErrorException dynamic_interval((; y=[0.5]))

    bad_keyword = @brm begin
        y ~ CircularVonMises(0.0, 1.0; period=2pi)
    end
    @test_throws ErrorException bad_keyword((; y=[0.5]))

    bad_arity_keyword = @brm begin
        y ~ CircularVonMises(1.0; interval=(-pi, pi))
    end
    @test_throws ErrorException SBBRMI(
        bad_arity_keyword((; y=[0.5])); mod=@__MODULE__)

    prior = @brm begin
        theta ~ VonMises(0.0, 1.0)
    end
    prior_model = SBBRMI(prior((; dummy=[1.0])); mod=@__MODULE__)
    @test StanBlocks.stanc_check(BRM.stan_code(prior_model)).ok
end
