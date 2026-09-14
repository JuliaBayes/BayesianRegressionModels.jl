using Test

repo = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(repo, "docs", "backend_comparisons.jl"))
source = read(joinpath(repo, "docs", "src", "feature-atlas.md"), String)

@testset "Exact deck source examples" begin
    for name in ("gaussian", "population_pk")
        marker = "\n" * name * " = (@brm begin"
        first_match = only(findall(marker, source))
        first_index = first(first_match) + 1
        terminator = findnext("\"\"\"", source, first_index)
        code = source[first_index:prevind(source, first(terminator))]
        example_mod = Module(Symbol("Deck_", name))
        Core.eval(example_mod, :(using BayesianRegressionModels, Distributions))
        model = BRMDocsComparisons.evaluate_source(example_mod, code)
        @test model isa BRMDocsComparisons.BRM.BRMI
        sb, stan = BRMDocsComparisons.stan_emissions(model, example_mod; required=true)
        @test !isempty(sb)
        @test occursin("model", stan)
        turing = BRMDocsComparisons.turing_emission(model)
        if name == "gaussian"
            @test !occursin("unsupported for this BRM example", turing)
            println("verified example=gaussian StanBlocks+Stan+Turing emission")
        else
            @test occursin("Turing unsupported for this BRM example", turing)
            @test occursin("response decorators other than `mi(response)` and response links are not yet supported", turing)
            println("verified example=population_pk StanBlocks+Stan emission; expected Turing ragged-response rejection")
        end
    end
end
println("DECK_EXAMPLES_CHECK_COMPLETE")
