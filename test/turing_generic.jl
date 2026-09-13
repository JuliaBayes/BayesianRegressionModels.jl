using Test
using BayesianRegressionModels
using Distributions
using Turing
using Random
using LinearAlgebra

const BRM = BayesianRegressionModels

generic_normal_prior(mu; width=1.0) = Normal(mu, width)
BRM.brm_distribution_type(::typeof(generic_normal_prior)) = Normal
generic_simplex_prior(alpha; power=1.0) = Dirichlet(alpha .^ power)
BRM.brm_distribution_type(::typeof(generic_simplex_prior)) = Dirichlet

struct ShiftedLaplace{T<:Real} <: ContinuousUnivariateDistribution
    location::T
    scale::T
end

@testset "generic callable prior factories retain kwargs and shape" begin
    data = (; y=zeros(2), alpha=[2.0, 3.0, 4.0])
    backend = TuringBRMI((@brm begin
        location ~ generic_normal_prior(0.25; width=1.5, lower=0)
        share ~ generic_simplex_prior(alpha; power=1.0)
        y ~ Normal(location, 1)
    end)(data))
    # Globally named factories must stay in the generated AST. Routing them
    # through the runtime callable tuple makes Enzyme abort while differentiating
    # the corresponding DynamicPPL log-density function.
    @test isempty(backend.model.args.callables)
    parameters = (; location=0.7, share=[0.2, 0.3, 0.5])
    expected = logpdf(Normal(0.25, 1.5), parameters.location) +
        logpdf(Dirichlet(data.alpha), parameters.share)
    @test Turing.logprior(backend.model, parameters) ≈ expected
    @test Turing.loglikelihood(backend.model, parameters) ≈
          sum(logpdf.(Normal(parameters.location, 1), data.y))
end

@testset "shared-ID predictors use one covariance block" begin
    data = (; x=[-1.0, 0.5, 2.0], g=["a", "b", "a"], y=[0, 1, 3])
    backend = TuringBRMI((@brm begin
        log(mu) ~ 1 + x + (1 | joint | g)
        log(phi) ~ 1 + x + (1 | joint | g)
        y ~ BRM.NegativeBinomial2(mu, phi)
    end)(data))
    L = cholesky(Symmetric([1.0 0.2; 0.2 1.0]))
    parameters = (; beta_pop=[0.2, -0.1], beta_pop_phi=[0.4, 0.15],
        shared_group_1=(; L, tau=[0.5, 0.3], z_flat=[-0.2, 0.4, 0.1, -0.3]))
    first_plan, second_plan = backend.plan.predictors
    coefficients = transpose(Diagonal(parameters.shared_group_1.tau) *
        Matrix(L.L) * reshape(parameters.shared_group_1.z_flat, 2, 2))
    mu_eta = first_plan.design.matrix * parameters.beta_pop +
        coefficients[first_plan.random_effects[1].indices, 1]
    phi_eta = second_plan.design.matrix * parameters.beta_pop_phi +
        coefficients[second_plan.random_effects[1].indices, 2]
    prior = sum(logpdf.(Normal(), parameters.beta_pop)) +
        sum(logpdf.(Normal(), parameters.beta_pop_phi)) +
        logpdf(LKJCholesky(2, 1), L) +
        sum(logpdf.(Normal(), parameters.shared_group_1.tau)) +
        sum(logpdf.(Normal(), parameters.shared_group_1.z_flat))
    likelihood = sum(logpdf.(BRM.NegativeBinomial2.(exp.(mu_eta), exp.(phi_eta)),
                             data.y))
    @test Turing.logjoint(backend.model, parameters) ≈ prior + likelihood
    returned = Turing.DynamicPPL.returned(backend.model, parameters)
    @test returned.mu ≈ exp.(mu_eta)
    @test returned.phi ≈ exp.(phi_eta)
    names = string.(keys(rand(backend.model).data))
    @test any(startswith("shared_group_1"), names)
    @test !any(startswith("group_1_1"), names)
end

@testset "shared-ID effects retain response-specific rows and level order" begin
    extension = Base.get_extension(BRM, :BayesianRegressionModelsTuringExt)
    levels = ("a", "b", "c")
    first_block = (; matrix=reshape([1.0, 2.0, -1.0, 0.5], :, 1),
        indices=[1, 2, 1, 3], levels, by=nothing, centered=false,
        lkj_eta=1.0, strata=(), group_strata=Int[])
    second_block = (; matrix=reshape([3.0, -2.0, 1.5], :, 1),
        indices=[3, 1, 2], levels, by=nothing, centered=false,
        lkj_eta=1.0, strata=(), group_strata=Int[])
    model = extension._brm_shared_group_effect_model(
        (first_block, second_block), (Normal(), Normal()),
        (nothing, nothing))
    L = cholesky(Symmetric([1.0 0.25; 0.25 1.0]))
    parameters = (; L, tau=[0.4, 0.7],
        z_flat=[-0.2, 0.3, 0.5, -0.4, 0.1, 0.6])
    coefficients = transpose(Diagonal(parameters.tau) * Matrix(L.L) *
        reshape(parameters.z_flat, 2, 3))
    returned = Turing.DynamicPPL.returned(model, parameters)
    @test returned.effects[1] ≈ [
        first_block.matrix[i, 1] * coefficients[first_block.indices[i], 1]
        for i in axes(first_block.matrix, 1)]
    @test returned.effects[2] ≈ [
        second_block.matrix[i, 1] * coefficients[second_block.indices[i], 2]
        for i in axes(second_block.matrix, 1)]
    expected_prior = logpdf(LKJCholesky(2, 1), L) +
        sum(logpdf.(Normal(), parameters.tau)) +
        sum(logpdf.(Normal(), parameters.z_flat))
    @test Turing.logprior(model, parameters) ≈ expected_prior
end

@testset "data references in parameter priors use whole values" begin
    data = (; alpha_data=[1.5, 2.0, 3.0], y=[0.1, -0.2])
    backend = TuringBRMI((@brm begin
        share ~ Dirichlet(alpha_data)
        y ~ Normal(0, 1)
    end)(data))
    parameters = (; share=[0.2, 0.3, 0.5])
    @test Turing.logprior(backend.model, parameters) ≈
          logpdf(Dirichlet(data.alpha_data), parameters.share)
end

@testset "standalone declarations remain sampled" begin
    data = (; y=[0.1, -0.2], y2=[0.3, 0.4])
    single = TuringBRMI((@brm begin
        unused ~ Normal(0, 2)
        y ~ Normal(0, 1)
    end)(data))
    parameters = (; unused=0.7)
    @test Turing.logprior(single.model, parameters) ≈ logpdf(Normal(0, 2), 0.7)
    @test :unused in keys(rand(single.model).data)

    multi = TuringBRMI((@brm begin
        unused ~ Normal(0, 2)
        y ~ Normal(0, 1)
        y2 ~ Normal(0, 1)
    end)(data))
    @test Turing.logprior(multi.model, parameters) ≈ logpdf(Normal(0, 2), 0.7)
    @test keys(rand(multi.model).data) == (:unused,)
end

@testset "random-effect priors use generated dependencies" begin
    data = (; g=["a", "b", "a"], y=[0.2, -0.1, 0.4])
    backend = TuringBRMI((@brm begin
        hyper ~ Exponential(1)
        mu ~ 1 + (1 | rid | g)
        sd(mu, rid) ~ Normal(hyper, 1)
        y ~ Normal(mu, 1)
    end)(data))
    parameters = (;
        hyper=0.8, beta_pop=[0.25],
        group_1_1=(; scale=0.6, z=[-0.2, 0.4]))
    block = only(backend.plan.random_effects)
    mu = fill(0.25, 3) + 0.6 .* parameters.group_1_1.z[block.indices]
    prior = logpdf(Exponential(1), 0.8) + logpdf(Normal(), 0.25) +
            logpdf(Normal(0.8, 1), 0.6) +
            sum(logpdf.(Normal(), parameters.group_1_1.z))
    likelihood = sum(logpdf.(Normal.(mu, 1), data.y))
    @test Turing.logprior(backend.model, parameters) ≈ prior
    @test Turing.logjoint(backend.model, parameters) ≈ prior + likelihood
end
Distributions.logpdf(d::ShiftedLaplace, x::Real) =
    -log(2d.scale) - abs(x - d.location) / d.scale
Base.rand(rng::Random.AbstractRNG, d::ShiftedLaplace) =
    d.location + rand(rng, Laplace(0, d.scale))

shifted_laplace(location, scale; shift=0) =
    ShiftedLaplace(location + shift, scale)

@testset "generic Turing likelihood retains callable and keywords" begin
    data = (; x=[-1.0, 0.0, 2.0], y=[0.2, -0.3, 1.1])
    brmi = (@brm begin
        width ~ Exponential(1.5)
        location ~ 1 + x
        y ~ shifted_laplace(location, width; shift=0.25)
    end)(data)
    backend = TuringBRMI(brmi)
    @test isempty(backend.model.args.callables)
    parameters = (; beta_pop=[0.4, -0.2], width=0.7)
    location_values = backend.plan.design.matrix * parameters.beta_pop
    expected = sum(logpdf.(Normal(), parameters.beta_pop)) +
        logpdf(Exponential(1.5), parameters.width) +
        sum(logpdf.(shifted_laplace.(location_values, parameters.width; shift=0.25),
                    data.y))
    @test Turing.logjoint(backend.model, parameters) ≈ expected
    @test backend.plan.distribution.callable === shifted_laplace
    @test backend.plan.distribution.kwargs.shift == 0.25
    @test BRM.turing_model_source(backend) isa Expr

    shifted_again = TuringBRMI((@brm begin
        width ~ Exponential(1.5)
        location ~ 1 + x
        y ~ shifted_laplace(location, width; shift=1.25)
    end)(data))
    expected_again = sum(logpdf.(shifted_laplace.(
        location_values, parameters.width; shift=1.25), data.y))
    @test Turing.loglikelihood(shifted_again.model, parameters) ≈ expected_again
    @test Turing.loglikelihood(shifted_again.model, parameters) !=
          Turing.loglikelihood(backend.model, parameters)
end

@testset "generic Turing population priors and parameter names" begin
    data = (; x=[-1.0, 0.5, 2.0], outcome=[0.2, 1.1, -0.4])
    backend = TuringBRMI((@brm begin
        noise ~ truncated(Cauchy(0, 1); lower=0)
        center_value ~ 1 + x
        effect(center_value, x) ~ Cauchy(0, 0.5)
        outcome ~ Normal(center_value, noise)
    end)(data))
    parameters = (; beta_pop=[0.25, -0.5], noise=0.8)
    center_value = backend.plan.design.matrix * parameters.beta_pop
    expected_prior = logpdf(Normal(), parameters.beta_pop[1]) +
        logpdf(Cauchy(0, 0.5), parameters.beta_pop[2]) +
        logpdf(truncated(Cauchy(0, 1); lower=0), parameters.noise)
    @test Turing.logprior(backend.model, parameters) ≈ expected_prior
    @test Turing.loglikelihood(backend.model, parameters) ≈
        sum(logpdf.(Normal.(center_value, parameters.noise), data.outcome))
end
