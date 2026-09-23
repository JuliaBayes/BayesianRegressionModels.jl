# Run with --project=test; no sampling or figure generation.
#
# The "Epidemic renewal models" docs page (docs/src/renewal.md) is built from
# research/epi_renewal/renewal.jl. This file evaluates every build-time `@eval`
# block of the page in order, in a fresh module, as Documenter does, and checks
# that
#   - each `source_code_region` / `source_function` marker still resolves,
#   - each backend comparison carries exactly one Stan pane that `stanc` accepts,
#     and a Turing pane that refuses the Stan-only callees,
#   - the tables generated from the checked-in summaries render,
# and that every model of the source file has a finite BridgeStan log density
# and gradient.

using Test, Markdown, Random
using BayesianRegressionModels, StanBlocks
import LogDensityProblems

include(joinpath(@__DIR__, "..", "docs", "backend_comparisons.jl"))

const RENEWAL_PAGE = joinpath(@__DIR__, "..", "docs", "src", "renewal.md")
const RENEWAL_SOURCE = joinpath(@__DIR__, "..", "research", "epi_renewal", "renewal.jl")

@testset "renewal docs page: validators" begin
    @test BRMDocsComparisons.validate_no_bypasses([RENEWAL_PAGE]) === nothing
    @test BRMDocsComparisons.validate_generated_templates([RENEWAL_PAGE]) === nothing
end

@testset "renewal docs page: build-time blocks evaluate in order" begin
    source = read(RENEWAL_PAGE, String)
    blocks = [m.captures[1] for m in eachmatch(r"(?ms)^```@eval[^\n]*\n(.*?)^```\s*$", source)]
    @test count(b -> occursin("Main.BRMDocsComparisons.comparison(", b), blocks) == 5
    page_module = Module(gensym(:RenewalDocs))
    comparisons = 0
    for block in blocks
        rendered = nothing
        for expression in Meta.parseall(block).args
            expression isa LineNumberNode && continue
            rendered = Core.eval(page_module, expression)
        end
        if occursin("Main.BRMDocsComparisons.comparison(", block)
            comparisons += 1
            @test occursin("require_stan=true", block)
            @test rendered isa Markdown.MD
            stan_panes = [pane for pane in rendered.content
                          if pane isa Markdown.Code && pane.language == "stan"]
            @test length(stan_panes) == 1
            @test StanBlocks.stanc_check(only(stan_panes).code; warn_pedantic=false).ok
            # `@deffun` / `@lpxf` callees are Stan-only: the Turing pane must refuse
            # them by name, which is what the page tells its reader.
            turing_pane = last(rendered.content)
            @test turing_pane isa Markdown.Code
            @test occursin("Turing unsupported for this BRM example", turing_pane.code)
        elseif occursin("source_code_region", block)
            @test rendered isa Markdown.MD
            @test only(rendered.content) isa Markdown.Code
        elseif occursin("Markdown.parse", block)
            @test rendered isa Markdown.MD
            @test any(part -> part isa Markdown.Table, rendered.content)
        end
    end
    @test comparisons == 5
end

@testset "renewal models: finite BridgeStan density and gradient" begin
    models = Module(gensym(:RenewalModels))
    Base.include(models, RENEWAL_SOURCE)                 # `main()` is guarded by PROGRAM_FILE
    # The prior-only program is the same model with the response column
    # omitted — a custom `@lpxf` family forward-simulates through its `_rng`
    # companion exactly like a Distributions.jl family.
    prior_df = Base.invokelatest(() -> begin
        d = models.renewal_single_data()
        (; time=d.time, observed=d.observed, gen_pmf=d.gen_pmf, delay_pmf=d.delay_pmf)
    end)
    cases = (
        ("reporting delay", () -> models.reporting_delay_model(), 2),
        ("one population", () -> models.renewal_single_model(), 59),
        ("one population, prior only",
         () -> models.renewal_single_model(prior_df), 0),
        ("one population, days 1-42",
         () -> models.renewal_single_model(models.renewal_single_data(; observed_through=42)), 59),
        ("six patches", () -> models.renewal_patch_model(), 115),
    )
    for (name, construct, dimension) in cases
        @testset "$name" begin
            built = Base.invokelatest(() -> models.build(construct()))
            @test LogDensityProblems.dimension(built.problem) == dimension
            q = 0.1 .* randn(Xoshiro(1), dimension)
            if dimension == 0
                @test isfinite(LogDensityProblems.logdensity(built.problem, q))
            else
                lp, gradient = LogDensityProblems.logdensity_and_gradient(built.problem, q)
                @test isfinite(lp)
                @test all(isfinite, gradient)
            end
        end
    end
    # The prior-only program is the fixed_param regime: empty parameters, the
    # custom family re-drawn in generated quantities.
    prior_built = Base.invokelatest(
        () -> models.build(models.renewal_single_model(prior_df)))
    @test occursin(r"parameters\s*\{\s*\}", prior_built.code)
    @test occursin("nb_cases_rng", prior_built.code)
    # masking rows removes exactly their likelihood terms: the masked density is larger
    full = Base.invokelatest(() -> models.build(models.renewal_single_model()))
    masked = Base.invokelatest(() -> models.build(
        models.renewal_single_model(models.renewal_single_data(; observed_through=42))))
    q = 0.1 .* randn(Xoshiro(2), 59)
    @test LogDensityProblems.logdensity(masked.problem, q) > LogDensityProblems.logdensity(full.problem, q)
end
