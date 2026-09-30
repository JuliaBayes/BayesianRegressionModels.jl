# Run with --threads=4 to exercise simultaneous construction in a fresh process.
using Test, BayesianRegressionModels, Turing, Distributions

const BRM = BayesianRegressionModels
const NP = BRM.NativePPL
include("concurrent_builds.jl")

@testset "Turing source expressions belong to each caller" begin
    data = (; y=[0.1, 0.5], z=[0.2, -0.1])
    parameters = (; beta=0.3)
    single = @brm begin
        beta ~ Normal(0.2, 0.7)
        y ~ Normal(beta, 0.5)
    end
    multi = @brm begin
        beta ~ Normal(0.2, 0.7)
        y ~ Normal(beta, 0.5)
        z ~ Cauchy(beta + 0.1, 0.8)
    end
    expected = logpdf(Normal(0.2, 0.7), parameters.beta) +
        sum(logpdf.(Normal(parameters.beta, 0.5), data.y))
    for (builder, density) in ((single, expected),
            (multi, expected + sum(logpdf.(Cauchy(0.4, 0.8), data.z))))
        # Cold identical structure: the compiled evaluator may be shared, but
        # its mutable source must not escape into independently built models.
        backends = concurrent_builds(_ -> TuringBRMI(builder(data)), 1:16)
        snapshots = [turing_model_source(backend) for backend in backends]
        @test all(source == first(snapshots) for source in snapshots)
        @test all(source !== first(snapshots) for source in snapshots[2:end])

        source = turing_model_source(first(backends))
        # Mutate a nested node, so a shallow copy would fail this regression.
        body = last(last(source.args).args)
        push!(body.args, :(error("caller edited source")))
        @test source != first(snapshots)
        @test all(turing_model_source(backend) == snapshot
                  for (backend, snapshot) in zip(backends, snapshots))
        @test all(Turing.logjoint(backend.model, parameters) ≈ density
                  for backend in backends)
    end
end

@testset "Julianic lowering owns its syntax before rewriting indices" begin
    definition = :(function indexed_source(selector)
        z ~ product_distribution(fill(Normal(0.0, 1.0), 4))
        total = sum(z[selector[begin:end]])
        y ~ Normal(total, 1.0)
    end)
    original = deepcopy(definition)
    expansions = concurrent_builds(_ -> NP._julianic_model_syntax(definition), 1:16)
    @test definition == original
    @test all(occursin("lastindex", sprint(show, expansion))
              for expansion in expansions)
    @test all(occursin("firstindex", sprint(show, expansion))
              for expansion in expansions)

    # The returned signature also belongs to the expansion, not the input.
    push!(first(expansions).args[1].args, :extra_argument)
    @test definition == original
    @test all(length(expansion.args[1].args) == 2
              for expansion in expansions[2:end])
    @test_throws ArgumentError NP._julianic_model_syntax(:(x + 1))
    @test definition == original
end
