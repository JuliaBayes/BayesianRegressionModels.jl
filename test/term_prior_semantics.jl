using Test, BayesianRegressionModels, Distributions, Turing, Random
const BRM = BayesianRegressionModels

# A valid simplex density whose parameters cannot be substituted into a
# Dirichlet constructor without changing the model.
struct ReversedSimplexPrior <: ContinuousMultivariateDistribution
    alpha::Vector{Float64}
end

@testset "standalone priors retain deterministic dependencies" begin
    model = TuringBRMI((@brm begin
        log_scale ~ Normal(0, 1)
        prior_scale = exp(log_scale)
        theta ~ Normal(0, prior_scale)
        y ~ Normal(0, 1)
    end)((; y=[0.1, -0.2])))
    p = (; log_scale=0.4, theta=-0.7)
    @test Turing.logprior(model.model, p) ≈
        logpdf(Normal(), p.log_scale) + logpdf(Normal(0, exp(p.log_scale)), p.theta)
    @test Tuple(node.name for node in model.plan.assignments) == (:prior_scale,)
end
Distributions.params(d::ReversedSimplexPrior) = (d.alpha,)
Base.length(d::ReversedSimplexPrior) = length(d.alpha)
Base.size(d::ReversedSimplexPrior) = (length(d),)
Distributions.logpdf(d::ReversedSimplexPrior, x::AbstractVector) =
    logpdf(Dirichlet(reverse(d.alpha)), x)
Distributions.insupport(d::ReversedSimplexPrior, x) =
    insupport(Dirichlet(reverse(d.alpha)), x)
Random.rand(rng::AbstractRNG, d::ReversedSimplexPrior) =
    rand(rng, Dirichlet(reverse(d.alpha)))

simplex_factory(alpha; power=1.0) = ReversedSimplexPrior(alpha .^ power)
BRM.brm_distribution_type(::typeof(simplex_factory)) = ReversedSimplexPrior

const prior_data = (; x=collect(range(-1., 1.; length=12)),
                     grade=repeat(1:4, 3), y=zeros(12))

@testset "both backends account for every term-prior address" begin
    wrong_address = @brm begin
            sd(mu, s(misspelled)) ~ Exponential(3)
            mu ~ 1 + s(x)
            y ~ Normal(mu, 1)
    end
    wrong_slot = @brm begin
            simplex(mu, s(x)) ~ Dirichlet(1)
            mu ~ 1 + s(x)
            y ~ Normal(mu, 1)
    end
    ambiguous = @brm begin
            sd(mu, s(x)) ~ Exponential(3)
            mu ~ 1 + s(x) + s(x)
            y ~ Normal(mu, 1)
    end
    builders = (wrong_address, wrong_slot, ambiguous)
    for builder in builders, backend in (SBBRMI, TuringBRMI)
        @test_throws ErrorException backend(builder(prior_data))
    end
    valid = (@brm begin
        sd(:, s(x)) ~ Exponential(3)
        sd(mu, s(x)) ~ Exponential(2)
        mu ~ 1 + s(x)
        y ~ Normal(mu, 1)
    end)(prior_data)
    context = BRM._brm_backend_context(valid)
    slot = context.term_priors[:mu][Symbol("s(x)")][:sd]
    @test getargs(slot.spec.expression) == (2,)
    @test length(TuringBRMI(valid).plan.predictors) == 1
end

@testset "simplex priors retain callable, keywords, density, and RNG" begin
    model = TuringBRMI((@brm begin
        simplex(mu, mo(grade)) ~ simplex_factory([1., 2., 3.]; power=2.)
        mu ~ 1 + mo(grade)
        y ~ Normal(mu, 1)
    end)(prior_data))
    parameters = (; beta_pop=[0.2],
        term_mu_1=(; beta=-0.7, simplex_incr=[0.2,0.3,0.5]))
    prior = simplex_factory([1.,2.,3.]; power=2.)
    expected = logpdf(Normal(),0.2) + logpdf(Normal(),-0.7) +
               logpdf(prior, parameters.term_mu_1.simplex_incr)
    @test Turing.logprior(model.model, parameters) ≈ expected
    geometry = BRM._brm_simplex_prior(prior, 3)
    @test logpdf(geometry, parameters.term_mu_1.simplex_incr) ==
          logpdf(prior, parameters.term_mu_1.simplex_incr)
    @test rand(MersenneTwister(31), geometry) == rand(MersenneTwister(31), prior)
    @test !insupport(geometry, [0.1,0.1,0.1])
    @test_throws ErrorException BRM._brm_simplex_prior(prior, 2)
    @test Turing.Bijectors.bijector(geometry) ==
          Turing.Bijectors.bijector(Dirichlet(ones(3)))
    draw = rand(MersenneTwister(4), model.model)
    @test all(isfinite, Turing.DynamicPPL.returned(model.model, draw.data).mu)
end

@testset "Dirichlet shorthand and sampled concentrations are normalized once" begin
    builder = @brm begin
        simplex(mu, mo(grade)) ~ Dirichlet(1,2,3)
        mu ~ 1 + mo(grade)
        y ~ Normal(mu,1)
    end
    model = TuringBRMI(builder(prior_data))
    parameters = (; beta_pop=[0.2],
        term_mu_1=(; beta=-0.7, simplex_incr=[0.2,0.3,0.5]))
    @test Turing.logprior(model.model, parameters) ≈
        logpdf(Normal(),0.2) + logpdf(Normal(),-0.7) +
        logpdf(Dirichlet([1.,2.,3.]), parameters.term_mu_1.simplex_incr)
    symbolic = TuringBRMI((@brm begin
        concentration ~ Exponential(1)
        simplex(mu, mo(grade)) ~ Dirichlet(concentration)
        mu ~ 1 + mo(grade)
        y ~ Normal(mu,1)
    end)(prior_data))
    p = merge(parameters, (; concentration=2.5))
    @test Turing.logprior(symbolic.model,p) ≈
        logpdf(Exponential(1),2.5) + logpdf(Normal(),0.2) +
        logpdf(Normal(),-0.7) + logpdf(Dirichlet(fill(2.5,3)),p.term_mu_1.simplex_incr)
end
