# BRM statistical bodies expressed as ordinary RKPPL declarations.
# Run: julia --project=test test/rk_plain.jl [testset substring ...]
using Test, BayesianRegressionModels, Distributions
using ReactiveKernels, ReactiveKernelsPPL
include(joinpath(@__DIR__, "testset_filter.jl"))
const BRM = BayesianRegressionModels

function check_printed_roundtrip(backend)
    emitted = BRM._rk_emit_ast(backend.plan)
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    printed = sprint(Base.show_unquoted, emitted.main)
    parsed = Meta.parse(printed)
    rebound = ext._rk_translate_from_emitted(backend.plan,
        BRM._RKEmittedProgram(Meta.parse.(sprint.(Base.show_unquoted, emitted.defs)), parsed,
            emitted.bindings))
    original = ext._rk_translated_plan(backend.plan)
    built = Base.invokelatest(build_kernel, rebound)
    @test coordinate_names(built.layout) == coordinate_names(backend.model.layout)
    N = length(coordinate_names(backend.model.layout))
    for u in (zeros(N), fill(0.13, N), collect(range(-0.2, 0.3; length=N)))
        for preset in (:sampler, :prior, :likelihood)
            a = Base.invokelatest(prepare_query(backend.model, original, preset), u)
            b = Base.invokelatest(prepare_query(built, rebound, preset), u)
            @test isequal(a, b)
        end
    end
    printed
end

@stestset "flat named coefficient priors and source round trip" begin
    data = (; x=[-0.5, 0.2, 0.8], y=[0.4, -0.1, 0.6])
    backend = RKBRMI(@brm data begin
        mu ~ 1 + x
        effect(mu, Intercept) ~ Normal(0.3, 1.2)
        effect(mu, x) ~ Laplace(-0.1, 0.7)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end)
    emitted = BRM._rk_emit_ast(backend.plan)
    @test isempty(emitted.defs)
    source = check_printed_roundtrip(backend)
    @test occursin("mu_Intercept ~ Normal(0.3, 1.2)", source)
    @test occursin("mu_x ~ Laplace(-0.1, 0.7)", source)
    @test !occursin("popefs", source)
    u = fill(0.13, length(coordinate_names(backend.model.layout)))
    values = constrain(backend.model.layout, u)
    expected = sum(logpdf.(Normal.(values.mu_Intercept .+ values.mu_x .* data.x,
        values.sigma), data.y)) + logpdf(Normal(0.3, 1.2), values.mu_Intercept) +
        logpdf(Laplace(-0.1, 0.7), values.mu_x) +
        logpdf(Exponential(1), values.sigma) + logjac(backend.model.layout, u)
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    plan = ext._rk_translated_plan(backend.plan)
    actual = Base.invokelatest(prepare_query(backend.model, plan, :sampler), u)
    @test actual ≈ expected
end
