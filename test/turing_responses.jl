using Test, BayesianRegressionModels, Distributions, Turing, Random
import StanBlocks
const BRM = BayesianRegressionModels
const BRMOrderedLogistic = BRM.OrderedLogistic

@testset "fitted categorical coding and generic class-logit expressions" begin
    data = (; x=[-0.8, 0.2, 0.7, 1.0], y=[10, 30, 20, 10])
    builder = @brm begin
        location ~ Normal(0, 1)
        eta ~ 0 + x
        y ~ CategoricalLogit(location + 0.2, eta)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; location=0.3, beta_pop=[0.4])
    expected = [logpdf(CategoricalLogit(0.5, 0.4x), y)
                for (x, y) in zip(data.x, [1, 3, 2, 1])]
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    replay = reprocess(backend, (; x=[0.3, -0.2], y=[30, 10]))
    @test replay.plan.response == [3, 1]
    @test replay.plan.response_fit.levels == backend.plan.response_fit.levels
    @test_throws "not a training level" reprocess(backend, (; x=[0.2], y=[40]))
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "implicit ordered thresholds are shared formula semantics" begin
    data = (; x=[-0.8, 0.2, 0.7, 1.0], y=[1, 3, 2, 1])
    builder = @brm begin
        eta ~ 0 + x
        y ~ BRMOrderedLogistic(eta)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; beta_pop=[0.4], y_cutpoints=[-0.5, 0.8])
    expected = [logpdf(BRMOrderedLogistic(0.4x, parameters.y_cutpoints), y)
                for (x, y) in zip(data.x, data.y)]
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(Normal(), 0.4) + sum(logpdf.(Normal(), parameters.y_cutpoints))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    @test all(in(1:3), turing_posterior_predictive(Xoshiro(42), backend, parameters).y)
    @test reprocess(backend, (; x=[0.1], y=[1])).plan.response_fit.n_levels == 3
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "typed ordinal links and threshold effects" begin
    data = (; x=[-0.8, 0.2, 0.7, 1.0], treat=[0.0, 1.0, 0.0, 1.0],
             y=[10, 30, 20, 10])
    for structure in (Cumulative(), StoppingRatio()), link in (LogitLink(), ProbitLink(), CloglogLink())
        S, L = typeof(structure), typeof(link)
        builder = @brm begin
            eta ~ 0 + x
            discrimination ~ Exponential(1)
            y ~ Ordinal(S(), L(), eta; discrimination)
        end
        backend = TuringBRMI(builder(data))
        parameters = (; beta_pop=[0.4], discrimination=1.2, y_thresholds=[-0.5, 0.8])
        expected = [logpdf(Ordinal(structure, link, 0.4x,
                            parameters.y_thresholds; discrimination=1.2), y)
                    for (x, y) in zip(data.x, [1, 3, 2, 1])]
        @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
        @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
        @test all(in(1:3), turing_posterior_predictive(Xoshiro(42), backend, parameters).y)
    end
    builder = @brm begin
        eta ~ 0 + x
        y ~ Ordinal(StoppingRatio(), ProbitLink(), eta; per_threshold=(treat,))
    end
    backend = TuringBRMI(builder(data))
    parameters = (; beta_pop=[0.4], y_thresholds=[-0.5, 0.8], y_threshold_beta=[0.3, -0.2])
    expected = [logpdf(Ordinal(StoppingRatio(), ProbitLink(),
                    0.4x .+ treat .* parameters.y_threshold_beta,
                    parameters.y_thresholds), y)
                for (x, treat, y) in zip(data.x, data.treat, [1, 3, 2, 1])]
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "one outcome level has zero-information likelihood" begin
    for distribution in (CategoricalLogit(), BRMOrderedLogistic(0.3, Float64[]),
                         Ordinal(Cumulative(), LogitLink(), 0.3, Float64[]),
                         Ordinal(StoppingRatio(), ProbitLink(), 0.3, Float64[]))
        @test logpdf(distribution, 1) == 0
        @test logpdf(distribution, 2) == -Inf
        @test Distributions.probs(distribution) == [1.0]
        @test rand(Xoshiro(42), distribution) == 1
    end
    data = (; y=[1, 1])
    backend = TuringBRMI((@brm begin
        y ~ BRMOrderedLogistic(0.0)
    end)(data))
    @test Turing.logjoint(backend.model, (; y_cutpoints=Float64[])) == 0
end
