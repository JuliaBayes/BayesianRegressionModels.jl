using Test
using BayesianRegressionModels
using Distributions
using Turing
using StanBlocks

const BRM = BayesianRegressionModels

@testset "Dirichlet concentration calls retain sampled hyperparameters" begin
    data = (; category=[1, 3, 2], y=[0.1, 0.7, 0.4])
    symmetric = @brm begin
        concentration ~ Exponential(2.0)
        share ~ Dirichlet(3, concentration)
        y ~ Normal(share[category], 0.2)
    end
    heterogeneous = @brm begin
        concentration ~ Exponential(2.0)
        share ~ Dirichlet([concentration, 1.0, concentration + 1.0])
        y ~ Normal(share[category], 0.2)
    end
    builders = (symmetric, heterogeneous)
    params = (; concentration=1.7, share=[0.2, 0.3, 0.5])
    for (i, builder) in enumerate(builders)
        brmi = builder(data)
        code = BRM.stan_code(SBBRMI(brmi))
        @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
        @test occursin("simplex[", code)
        backend = TuringBRMI(brmi)
        alpha = i == 1 ? fill(params.concentration, 3) :
                        [params.concentration, 1.0, params.concentration + 1.0]
        @test Turing.logprior(backend.model, params) ≈
            logpdf(Exponential(2.0), params.concentration) +
            logpdf(Dirichlet(alpha), params.share)
    end
end
