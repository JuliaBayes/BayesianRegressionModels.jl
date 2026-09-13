using Test
using BayesianRegressionModels
using Distributions
using Turing
using LinearAlgebra
using Random

const BRM = BayesianRegressionModels
const Ext = Base.get_extension(BRM, :BayesianRegressionModelsTuringExt)

@testset "parameter axes are distribution shapes" begin
    @test BRM._brm_parameter_reference_axis(ExprColumn(Normal, 0, 1)) === :scalar
    @test BRM._brm_parameter_reference_axis(ExprColumn(Dirichlet, 3, 1)) === :whole
    @test BRM._brm_parameter_reference_axis(ExprColumn(MvNormal, zeros(2), I)) === :whole
    @test BRM._brm_parameter_reference_axis(
        ExprColumn(LKJCovarianceFactor, 2)) === :whole
end

@testset "simplex parameters stay whole through rowwise indexing" begin
    data = (; category=[1, 3, 2], y=[0.1, 0.7, 0.4])
    backend = TuringBRMI((@brm begin
        share ~ Dirichlet(3, 2.0)
        y ~ Normal(share[category], 0.2)
    end)(data))
    parameters = (; share=[0.2, 0.3, 0.5])
    expected = logpdf(Dirichlet(3, 2.0), parameters.share) +
        sum(logpdf.(Normal.(parameters.share[data.category], 0.2), data.y))
    @test Turing.logjoint(backend.model, parameters) ≈ expected
    @test only(backend.plan.parameters).prior.callable === Dirichlet
    @test Turing.returned(backend.model, parameters).share == parameters.share
end

@testset "bounded hierarchical priors preserve their declared kernel" begin
    data = (; y=[0.1, 0.7, -0.2])
    backend = TuringBRMI((@brm begin
        location ~ Normal(0, 1)
        width ~ Normal(location, 0.7; lower=0)
        y ~ Normal(0, width)
    end)(data))
    for location in (-0.8, 0.4)
        parameters = (; location, width=0.6)
        expected = logpdf(Normal(), location) +
            logpdf(Normal(location, 0.7), parameters.width)
        @test Turing.logprior(backend.model, parameters) ≈ expected
        @test Turing.loglikelihood(backend.model, parameters) ≈
            sum(logpdf.(Normal(0, parameters.width), data.y))
    end
end

@testset "joint outcomes retain one density and predictive vector per row" begin
    data = (; x=[-1.0, 0.2, 1.0], y1=[0.2, 0.4, -0.1], y2=[0.8, 1.2, 0.7])
    backend = TuringBRMI((@brm begin
        L_res ~ LKJCovarianceFactor(2; scale_prior=Normal(0.3, 0.8), shape=2.0)
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end)(data))
    scales = [0.5, 1.2]
    L_corr = cholesky(Symmetric([1.0 0.2; 0.2 1.0]))
    parameters = (; L_res=(; scales, L_corr),
                    beta_pop=[0.1, -0.2], beta_pop_mu2=[0.8, 0.3])
    mean1 = backend.plan.predictors[1].design.matrix * parameters.beta_pop
    mean2 = backend.plan.predictors[2].design.matrix * parameters.beta_pop_mu2
    factor = Diagonal(scales) * L_corr.L
    expected = [logpdf(MvNormal([mean1[i], mean2[i]], factor * factor'),
                      [data.y1[i], data.y2[i]]) for i in eachindex(data.x)]
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    pointwise = turing_pointwise_loglikelihoods(backend, parameters)
    @test only(values(pointwise)) ≈ expected
    @test length(only(values(pointwise))) == length(data.x)
    returned = Turing.returned(backend.model, parameters)
    @test returned.L_res ≈ factor
    predictive = turing_posterior_predictive(Xoshiro(20260913), backend, parameters)
    @test length(only(values(predictive))) == length(data.x)
    @test all(row -> length(row) == 2, only(values(predictive)))
end

@testset "declaration support preserves parameter-dependent density kernels" begin
    for location in (-1.0, 0.4, 1.3)
        base = Normal(location, 0.7)
        bounded = Ext._brm_constrained_kernel(base; lower=0)
        @test logpdf(bounded, 0.6) == logpdf(base, 0.6)
        @test logpdf(bounded, -0.1) == -Inf
        @test logpdf(bounded, 0.6) != logpdf(truncated(base; lower=0), 0.6)
        @test all(>=(0), rand(MersenneTwister(4), bounded, 32))
        @test isfinite(Turing.Bijectors.bijector(bounded)(0.6))
    end
end

@testset "covariance factor submodel retains general scale kernels" begin
    L_corr = cholesky(Symmetric([1.0 0.2; 0.2 1.0]))
    scales = [0.5, 1.2]
    for location in (-0.5, 0.7)
        prior = Normal(location, 0.8)
        model = BRM._brm_turing_covariance_prior(2; scale_prior=prior, shape=2.0)
        expected = sum(logpdf.(prior, scales)) +
                   logpdf(LKJCholesky(2, 2.0), L_corr)
        @test Turing.logjoint(model, (; scales, L_corr)) ≈ expected atol=1e-12
        @test Turing.returned(model, (; scales, L_corr)) ≈ Diagonal(scales) * L_corr.L
    end
end
