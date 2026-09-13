using Test
using BayesianRegressionModels
using Distributions
using StanBlocks
using LogDensityProblems

const BRM = BayesianRegressionModels

@testset "affine composition retains the complete base distribution" begin
    builder = @brm begin
        y_normal ~ LocationScale(0.1, 2.0, Normal(5.0, 0.3))
        y_laplace ~ LocationScale(-0.2, 1.5, Laplace(0.7, 0.4))
    end
    data = (; y_normal=[9.7, 10.2, 10.8], y_laplace=[0.2, 0.7, 1.1])
    descriptor = brm_descriptor(builder, data; mod=@__MODULE__, highlights=())
    code = BRM.stan_code(descriptor.plan)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    cache = joinpath(tempdir(), "brm-affine-distributions")
    mkpath(cache)
    problem = brm_execute(descriptor, :instantiate;
        path=joinpath(cache, string(descriptor.id) * ".stan"))
    @test LogDensityProblems.dimension(problem) == 0
    pointwise = brm_execute(descriptor, :pointwise_loglik;
        problem, draws=Float64[], seed=20260913)
    @test pointwise.y_normal_likelihood ≈
        logpdf.(LocationScale(0.1, 2.0, Normal(5.0, 0.3)), data.y_normal) atol=1e-10
    @test pointwise.y_laplace_likelihood ≈
        logpdf.(LocationScale(-0.2, 1.5, Laplace(0.7, 0.4)), data.y_laplace) atol=1e-10
end

@testset "affine prior uses the same transformation and Jacobian" begin
    builder = @brm begin
        beta ~ LocationScale(0.1, 2.0, Normal(5.0, 0.3))
        y ~ Normal(beta, 1.0)
    end
    descriptor = brm_descriptor(builder, (; y=[9.8, 10.6]);
        mod=@__MODULE__, highlights=())
    code = BRM.stan_code(descriptor.plan)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    cache = joinpath(tempdir(), "brm-affine-distributions")
    mkpath(cache)
    problem = brm_execute(descriptor, :instantiate;
        path=joinpath(cache, string(descriptor.id) * ".stan"))
    @test LogDensityProblems.dimension(problem) == 1
    for beta in (9.9, 10.7)
        _, gradient = LogDensityProblems.logdensity_and_gradient(problem, [beta])
        expected = -(beta - 10.1) / 0.6^2 + (9.8 - beta) + (10.6 - beta)
        @test only(gradient) ≈ expected atol=1e-10
    end
end
