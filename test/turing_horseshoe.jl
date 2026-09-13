using Test
using BayesianRegressionModels
using Distributions
using Turing
using StanBlocks
using LinearAlgebra

const BRM = BayesianRegressionModels

@testset "horseshoe preparation and hierarchical scale calls are shared" begin
    data = (; y=[-0.4, 0.1, 0.7])
    builder = @brm begin
        prior_scale ~ Exponential(0.3)
        beta ~ Horseshoe(local_scale=1 / 4, global_scale=prior_scale)
        y ~ Normal(beta, 1.0)
    end
    brmi = builder(data)
    backend = TuringBRMI(brmi)
    parameters = (; prior_scale=0.2, beta=(; raw=0.4, lambda=0.3, tau=0.15))
    coefficient = prod(values(parameters.beta))
    expected_prior = logpdf(Exponential(0.3), parameters.prior_scale) +
        logpdf(Normal(), parameters.beta.raw) +
        logpdf(Cauchy(0, 0.25), parameters.beta.lambda) +
        logpdf(Cauchy(0, parameters.prior_scale), parameters.beta.tau)
    @test Turing.logprior(backend.model, parameters) ≈ expected_prior
    @test Turing.loglikelihood(backend.model, parameters) ≈
        sum(logpdf.(Normal(coefficient, 1), data.y))
    @test Turing.returned(backend.model, parameters).beta ≈ coefficient
    code = BRM.stan_code(SBBRMI(brmi))
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("beta_tau ~ cauchy(0.0, prior_scale);", code)
    @test occursin("beta_lambda ~ cauchy(0.0, 0.25);", code)
end

@testset "covariance shape may depend on an earlier sampled declaration" begin
    data = (; y1=[0.2, 0.4], y2=[0.8, 1.1])
    builder = @brm begin
        concentration ~ Exponential(2.0)
        L_res ~ LKJCovarianceFactor(2; shape=concentration)
        [y1, y2] ~ MvNormalCholesky([0.0, 0.5], L_res)
    end
    brmi = builder(data)
    sb = SBBRMI(brmi)
    code = BRM.stan_code(sb)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("lkj_corr_cholesky(concentration)", code)
    backend = TuringBRMI(brmi)
    L_corr = cholesky(Symmetric([1.0 0.2; 0.2 1.0]))
    parameters = (; concentration=1.7, L_res=(; scales=[0.5, 1.1], L_corr))
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(Exponential(2.0), parameters.concentration) +
        sum(logpdf.(Exponential(1.0), parameters.L_res.scales)) +
        logpdf(LKJCholesky(2, parameters.concentration), L_corr)
end
