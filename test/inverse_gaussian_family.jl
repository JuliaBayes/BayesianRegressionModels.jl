# test/inverse_gaussian_family.jl — Wald (inverse-Gaussian) likelihood + prior contracts.
#
# Run on a capable host:
#   julia --startup-file=no --project=test test/inverse_gaussian_family.jl

using Test
using BayesianRegressionModels
using Distributions
using Statistics
using StanBlocks
using Turing

const BRM = BayesianRegressionModels

@testset "InverseGaussian constructor semantics stay Julia-native" begin
    # Deliberately NOT in `_sb_stan_dist_name` (same reasoning as `VonMises`):
    # no StanBlocks `inverse_gaussian` builtin exists, so the bespoke
    # likelihood and prior methods emit the `brm_inverse_gaussian` density.
    @test BRM._sb_stan_dist_name(InverseGaussian) === nothing
    @test BRM._sb_stan_dist_args(InverseGaussian, ()) == params(InverseGaussian())
    @test BRM._sb_stan_dist_args(InverseGaussian, (:mu,)) == (:mu, 1.0)
    @test BRM._sb_stan_dist_args(InverseGaussian, (:mu, :lambda)) == (:mu, :lambda)
end

semantic_df = (;
    y_wald=[0.7, 1.4, 2.6],
    y_unit=[0.5, 1.0, 3.0],
)

semantic_builder = @brm begin
    y_wald ~ InverseGaussian(2.0, 3.0)
    y_unit ~ InverseGaussian(1.5)
end

@testset "wald lowering matches Distributions.jl exactly" begin
    descriptor = brm_descriptor(semantic_builder, semantic_df; mod=@__MODULE__)
    code = brm_execute(descriptor, :transpile)

    @test occursin("brm_inverse_gaussian_lpdf", code)
    @test occursin("brm_inverse_gaussian_rng", code)
    @test occursin("y_wald ~ brm_inverse_gaussian(2.0, 3.0);", code)
    @test occursin("y_unit ~ brm_inverse_gaussian(1.5, 1.0);", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    cache = joinpath(tempdir(), "brm-inverse-gaussian-semantics")
    isdir(cache) || mkpath(cache)
    problem = brm_execute(
        descriptor, :instantiate;
        path=joinpath(cache, string(descriptor.id) * ".stan"))
    pointwise = brm_execute(
        descriptor, :pointwise_loglik;
        problem, draws=Float64[], seed=20260729)

    @test pointwise.y_wald_likelihood ≈
          logpdf.(InverseGaussian(2.0, 3.0), semantic_df.y_wald) atol=1e-10
    @test pointwise.y_unit_likelihood ≈
          logpdf.(InverseGaussian(1.5), semantic_df.y_unit) atol=1e-10

    # Exercise the Michael–Schucane–Haas RNG on the exact same translated
    # arguments. A deterministic large generated-quantities batch checks the
    # Julia mean and variance contracts without requiring the two libraries'
    # RNG streams to be byte-identical. Tolerances sit at least 5 standard
    # errors out: for InverseGaussian(2, 3), SE(mean) ≈ 0.026 and
    # SE(var) ≈ 0.15 over 4096 draws; for InverseGaussian(1.5, 1),
    # SE(mean) ≈ 0.029 and SE(var) ≈ 0.27.
    predictions = brm_execute(
        descriptor, :predict;
        problem, draws=zeros(0, 4096), seed=20260729)
    @test all(predictions.y_wald_gen .> 0)
    @test all(predictions.y_unit_gen .> 0)
    for (draws, dist, mean_atol, var_atol) in (
        (predictions.y_wald_gen, InverseGaussian(2.0, 3.0), 0.25, 1.5),
        (predictions.y_unit_gen, InverseGaussian(1.5, 1.0), 0.25, 1.5),
    )
        @test mean(draws) ≈ mean(dist) atol=mean_atol
        @test var(vec(draws)) ≈ var(dist) atol=var_atol
    end
end

# The reporting shape: a log-link Wald GLM (bambi `wald_gamma_glm`-style
# `claimcst0 ~ age/gender/area` with `family=wald, link=log`), spelled as
# explicit distributional composition exactly like the Gamma half.
regression_df = (;
    x=[-1.0, 0.0, 1.0],
    y=[1.2, 0.8, 1.1],
)
regression_builder = @brm begin
    lambda ~ LogNormal(0, 0.3)
    eta ~ 1 + x
    y ~ InverseGaussian(exp(eta), lambda)
end

@testset "InverseGaussian is a log-link regression family" begin
    code = brm_execute(
        brm_descriptor(regression_builder, regression_df; mod=@__MODULE__),
        :transpile)
    @test occursin("y ~ brm_inverse_gaussian(exp(eta), lambda", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "Turing backend agrees the model is well-formed" begin
    backend = TuringBRMI(regression_builder(regression_df))
    params = (; lambda=1.0, beta_pop=[0.1, 0.2])
    @test isfinite(Turing.logjoint(backend.model, params))
    @test occursin("InverseGaussian", string(turing_model_source(backend)))
end

@testset "inverse-Gaussian validation is early and explicit" begin
    nonpositive = @brm begin
        y ~ InverseGaussian(1.0, 2.0)
    end
    @test_throws ErrorException brm_descriptor(
        nonpositive, (; y=[1.2, 0.0, 1.1]); mod=@__MODULE__)
    @test_throws ErrorException brm_descriptor(
        nonpositive, (; y=[1.2, -0.5, 1.1]); mod=@__MODULE__)
    @test_throws ErrorException brm_descriptor(
        nonpositive, (; y=[1.2, Inf, 1.1]); mod=@__MODULE__)

    zero_mu = @brm begin
        y ~ InverseGaussian(0.0, 2.0)
    end
    @test_throws ErrorException brm_descriptor(
        zero_mu, (; y=[1.2, 0.8, 1.1]); mod=@__MODULE__)

    negative_lambda = @brm begin
        y ~ InverseGaussian(1.0, -2.0)
    end
    @test_throws ErrorException brm_descriptor(
        negative_lambda, (; y=[1.2, 0.8, 1.1]); mod=@__MODULE__)

    bad_arity = @brm begin
        y ~ InverseGaussian(1.0, 2.0, 3.0)
    end
    @test_throws ErrorException SBBRMI(
        bad_arity((; y=[1.2, 0.8, 1.1])); mod=@__MODULE__)

    no_args = @brm begin
        y ~ InverseGaussian()
    end
    @test_throws ErrorException SBBRMI(
        no_args((; y=[1.2, 0.8, 1.1])); mod=@__MODULE__)

    prior = @brm begin
        sigma ~ InverseGaussian(1.0, 2.0)
    end
    prior_model = SBBRMI(prior((; dummy=[1.0])); mod=@__MODULE__)
    @test StanBlocks.stanc_check(BRM.stan_code(prior_model)).ok

    prior_short = @brm begin
        sigma ~ InverseGaussian(1.5)
    end
    @test StanBlocks.stanc_check(
        BRM.stan_code(SBBRMI(prior_short((; dummy=[1.0])); mod=@__MODULE__))).ok

    prior_unit = @brm begin
        sigma ~ InverseGaussian()
    end
    @test StanBlocks.stanc_check(
        BRM.stan_code(SBBRMI(prior_unit((; dummy=[1.0])); mod=@__MODULE__))).ok

    prior_bad_arity = @brm begin
        sigma ~ InverseGaussian(1.0, 2.0, 3.0)
    end
    @test_throws ErrorException SBBRMI(
        prior_bad_arity((; dummy=[1.0])); mod=@__MODULE__)
end
