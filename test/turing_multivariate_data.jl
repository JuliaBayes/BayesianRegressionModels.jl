using Test, BayesianRegressionModels, Distributions, Turing, Random
const BRM = BayesianRegressionModels

@testset "multinomial response rows preserve the joint count density" begin
    counts = [2 3 1; 1 1 4; 3 2 1]
    ntrial = vec(sum(counts; dims=2))
    original_data = (; counts, ntrial)
    for probability in ([0.2, 0.5, 0.3], [0.2 0.5 0.3; 0.1 0.4 0.5; 0.3 0.4 0.3])
        data = (; original_data..., probability)
        backend = TuringBRMI((@brm begin
            counts ~ Multinomial(ntrial, probability)
        end)(data))
        probabilities = data.probability isa AbstractMatrix ? eachrow(data.probability) :
            fill(data.probability, size(data.counts, 1))
        expected = [logpdf(Multinomial(n, p), y)
                    for (n, p, y) in zip(data.ntrial, probabilities, eachrow(data.counts))]
        @test Turing.loglikelihood(backend.model, (;)) ≈ sum(expected)
        @test turing_pointwise_loglikelihoods(backend, (;)).counts ≈ expected
        @test length(backend.plan.response) == size(data.counts, 1)
        predictive = turing_posterior_predictive(Xoshiro(42), backend, (;)).counts
        @test length(predictive) == 3
        @test sum.(predictive) == data.ntrial
    end
end

@testset "objective weights multiply each joint row once" begin
    data = (; counts=[2 3 1; 1 1 4], probability=[0.2, 0.5, 0.3], power=[0.5, 2.0])
    backend = TuringBRMI((@brm begin
        counts ~ weighted(Multinomial(6, probability), weights(power))
    end)(data))
    expected = data.power .* [logpdf(Multinomial(6, data.probability), row)
                              for row in eachrow(data.counts)]
    @test Turing.loglikelihood(backend.model, (;)) ≈ sum(expected)
    @test turing_pointwise_loglikelihoods(backend, (;)).counts ≈ expected
    @test all(==(6), sum.(turing_posterior_predictive(Xoshiro(42), backend, (;)).counts))
end

@testset "multivariate Gaussian data arguments retain their mathematical shapes" begin
    outcomes = [0.1 0.5; -0.2 1.0; 0.3 0.7]
    covariance = [1.0 0.2; 0.2 0.6]
    original_data = (; outcomes, covariance)
    for locations in ([0.0, 0.8], [0.0 0.8; 0.1 0.7; -0.2 1.0])
        data = (; original_data..., locations)
        backend = TuringBRMI((@brm begin
            outcomes ~ MvNormal(locations, covariance)
        end)(data))
        means = data.locations isa AbstractMatrix ? eachrow(data.locations) : fill(data.locations, 3)
        expected = [logpdf(MvNormal(mu, data.covariance), y)
                    for (mu, y) in zip(means, eachrow(data.outcomes))]
        @test Turing.loglikelihood(backend.model, (;)) ≈ sum(expected)
        @test turing_pointwise_loglikelihoods(backend, (;)).outcomes ≈ expected
        @test length(turing_posterior_predictive(Xoshiro(42), backend, (;)).outcomes) == 3
    end
end
