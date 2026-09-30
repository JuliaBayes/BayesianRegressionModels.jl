# Run with --threads=4 after setup_env.jl installs the concurrent-build RK pin.
using Test, BayesianRegressionModels, Distributions, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
import Enzyme, ReactiveKernels, ReactiveKernelsPPL

const BRM = BayesianRegressionModels
include("concurrent_builds.jl")

const RK_SMALL = @brm begin
    mu ~ 1 + x
    s ~ Exponential(1)
    y ~ Normal(mu, s)
end
const RK_LARGE = @brm begin
    mu ~ 1 + x + z
    s ~ Exponential(1)
    y ~ Normal(mu, s)
end

rk_construction_result(index) = rk_construction_result_for(
    isodd(index) ? RK_SMALL : RK_LARGE, index)

@noinline function rk_construction_result_for(builder, index)
    data = (; x=[-1.0, 0.0, 1.0], z=[1.0, -2.0, 1.0],
        y=[-0.4, 0.1, 0.7] .+ 0.05index)
    snapshot = deepcopy(data)
    backend = BRM.RKBRMI(builder(data))
    layout = backend.model.layout
    position = collect(range(-0.3, 0.2; length=layout.total))
    parameters = ReactiveKernelsPPL.constrain(layout, position)
    # These all-Normal population priors use BRM's fused GLM lowering:
    # intercept and slope vector are separate constrained parameters.
    mu = parameters.mu_alpha .+ parameters.mu_beta[1] .* data.x
    iseven(index) && (mu .+= parameters.mu_beta[2] .* data.z)
    expected = sum(logpdf.(Normal.(mu, parameters.s), data.y)) +
        logpdf(Normal(), parameters.mu_alpha) +
        sum(logpdf.(Normal(), parameters.mu_beta)) +
        logpdf(Exponential(1), parameters.s) + log(parameters.s)

    # Construction and first execution share a compiled frame. Each task owns
    # its sampler query; BRM's supported shim supplies the world-age boundary.
    problem = BRM.rk_logdensity_problem(backend;
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), u0=position)
    value = LogDensityProblems.logdensity(problem, position)
    (; dimension=LogDensityProblems.dimension(problem), value, expected,
       input_unchanged=data == snapshot)
end

@testset "independent RK construction and first execution preserve density" begin
    results = concurrent_builds(rk_construction_result, 1:8)
    @test all(result.dimension == (isodd(index) ? 3 : 4)
              for (index, result) in enumerate(results))
    @test all(result.value ≈ result.expected for result in results)
    @test all(result.input_unchanged for result in results)
end
