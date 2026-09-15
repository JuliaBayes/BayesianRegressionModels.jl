# Run with --project=test; no sampling or figure generation.
# Evaluate each required comparison exactly as Documenter does: in a fresh
# module, not Main, so consumer-defined prior families retain their owner.
using Test, Markdown, StanBlocks

include(joinpath(@__DIR__, "..", "docs", "backend_comparisons.jl"))

const CASE_PAGES = isempty(ARGS) ? [
    "adaptive-centering.md", "eight-schools-centering.md", "radon-centering.md",
] : ARGS

@testset "case-study backend panes in isolated page modules" begin
    for page in CASE_PAGES
        @testset "$page" begin
            path = joinpath(@__DIR__, "..", "docs", "src", page)
            source = read(path, String)
            blocks = [m.captures[1] for m in eachmatch(
                r"(?ms)^```@eval[^\n]*\n(.*?)^```\s*$", source)
                if occursin("Main.BRMDocsComparisons.comparison(", m.captures[1])]
            @test !isempty(blocks)
            page_module = Module(gensym(:CaseStudyDocs))
            for block in blocks
                @test occursin("require_stan=true", block)
                rendered = Core.eval(page_module, Meta.parse(block))
                @test rendered isa Markdown.MD
                @test length(rendered.content) == 5
                slic = rendered.content[3]
                @test slic isa Markdown.Code && slic.language == "julia"
                @test !occursin("SlicModel(untraced", slic.code)
                if page == "adaptive-centering.md"
                    @test occursin("configured submodels:", slic.code)
                    @test occursin("Base.merge(", slic.code)
                    @test occursin("_configured_1", slic.code)
                end
                stan = rendered.content[4]
                @test stan isa Markdown.Code && stan.language == "stan"
                @test StanBlocks.stanc_check(stan.code; warn_pedantic=false).ok
                println("case_study_stan_pane\t", page, "\t", sizeof(stan.code))
            end
        end
    end
end
