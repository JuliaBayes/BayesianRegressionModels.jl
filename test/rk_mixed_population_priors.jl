# Independent normalized control for the delivered mixed-family component.
# Population prior arguments retain their existing numeric-constant boundary.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Distributions, Enzyme
using DifferentiationInterface: AutoEnzyme
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "rk_source_roundtrip.jl"))
const DATA = (; x=[-.6, -.1, .4, .9], y=[.2, -.3, .5, .1])

@testset "mixed population components retain independent prior families" begin
    brmi = @brm DATA begin
        tau ~ Exponential(1)
        mu ~ 1 + x
        effect(mu, Intercept) ~ Normal(.1, .8)
        effect(mu, x) ~ Laplace(-.2, .6)
        y ~ Normal(mu, tau)
    end
    backend = check_rk_source_roundtrip(RKBRMI(brmi))
    layout = backend.model.layout
    @test layout.total == 3
    println("mixed-prior coordinates: ", coordinate_names(layout))
    function oracle(u)
        p = constrain(layout, u)
        a, b = p.pop_mu.beta_pop_1, p.pop_mu.beta_pop_2
        logpdf(Exponential(), p.tau) + logpdf(Normal(.1, .8), a) +
            logpdf(Laplace(-.2, .6), b) + log(p.tau) +
            sum(logpdf.(Normal.(a .+ b .* DATA.x, p.tau), DATA.y))
    end
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    bound = ext._rk_translated_plan(backend.plan)
    query = prepare_sampler(backend.model, bound, zeros(3);
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for u in (zeros(3), fill(.13, 3), collect(range(-.3, .25; length=3)))
        gradient = similar(u)
        value, _ = sampler_value_and_gradient!(query, gradient, u)
        @test value ≈ oracle(u) atol=2e-12 rtol=2e-12
        h = 2e-5
        independent = map(eachindex(u)) do j
            plus, minus = copy(u), copy(u)
            plus[j] += h
            minus[j] -= h
            (oracle(plus)-oracle(minus))/(2h)
        end
        @test gradient ≈ independent atol=2e-7 rtol=2e-7
    end
end

@testset "population prior arguments retain the numeric-constant boundary" begin
    brmi = @brm DATA begin
        tau ~ Exponential(1)
        mu ~ 1 + x
        effect(mu, Intercept) ~ Normal(.1, tau)
        effect(mu, x) ~ Laplace(-.2, tau)
        y ~ Normal(mu, tau)
    end
    @test_throws "must be a numeric constant" RKBRMI(brmi)
end
