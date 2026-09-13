using Test, Markdown, BayesianRegressionModels, Distributions, Turing
import StanBlocks
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "docs", "backend_comparisons.jl"))

@testset "four-pane comparisons execute one model through both backends" begin
    sources = (
        """
        data = (; x=[-1.0, 0.0, 1.0], y=[0.2, 0.4, 0.7])
        model = (@brm begin
            sigma ~ LogNormal(-0.2, 0.3)
            mu ~ 1 + x
            effect(mu, x) ~ Cauchy(0, 0.5)
            y ~ Normal(mu, sigma)
        end)(data)
        """,
        """
        data = (; x=[-1.0, 0.0, 1.0], y=[1.2, 0.8, 1.1])
        model = (@brm begin
            shape ~ LogNormal(0, 0.3)
            eta ~ 1 + x
            y ~ Gamma(shape, exp(eta))
        end)(data)
        """,
        """
        data = (; x=[-1.0, 0.0, 1.0], y=[0.2, 0.4, 0.7], y2=[0.1, 0.8, 0.2])
        model = (@brm begin
            sigma ~ LogNormal(-0.2, 0.3)
            mu ~ 1 + x
            y ~ Normal(mu, sigma)
            y2 ~ Normal(mu, sigma)
        end)(data)
        """,
    )
    for (i, source) in pairs(sources)
        mod = BRMDocsComparisons.example_module(Symbol(:refactoring_comparison_, i))
        panes = BRMDocsComparisons.comparison(mod, source, :model; require_stan=true)
        @test length(panes.content) == 5
        @test getfield.(panes.content, :language) ==
              ["brm-comparison", "julia", "julia", "stan", "julia"]
        @test panes.content[2].code == strip(source, '\n')
        @test StanBlocks.stanc_check(panes.content[4].code).ok
        turing_source = panes.content[5].code
        @test !occursin("unsupported", turing_source)
        @test occursin("@model", turing_source)
        @test occursin("~", turing_source)
        backend = TuringBRMI(Core.eval(mod, :model))
        params = i == 2 ? (; shape=1.3, beta_pop=[0.1, 0.2]) :
                          (; sigma=0.8, beta_pop=[0.1, 0.2])
        @test isfinite(Turing.logjoint(backend.model, params))
        @test turing_model_source(backend) isa Expr
        if i == 3
            @test occursin("y_1", turing_source)
            @test occursin("y_2", turing_source)
            @test length(findall("@model", turing_source)) == 1
        end
    end
end
