using Markdown, Test
include("backend_comparisons.jl")
include("centering_examples.jl")
out=only(ARGS);mkpath(out)
@testset "Four generated automatic-total model examples" begin
    for (which,name,blocks,dimensions) in (
        (:pupil_numeric,:pupil_numeric_brm_model,1,45),
        (:pupil_hierarchical,:pupil_hierarchical_brm_model,2,65),
        (:air_intercept,:air_intercept_brm_model,1,10),
        (:air_independent,:air_independent_brm_model,1,16))
        mod=Module(which)
        code=BRMCenteringExamples.authoring(which)
        rendered=BRMDocsComparisons.comparison(mod,code,name;require_stan=true)
        file=joinpath(out,string(which)*".md")
        open(io->show(io,MIME"text/markdown"(),rendered),file,"w")
        @test BRMDocsComparisons.validate_required_stan_outputs(file,(name,))===nothing
        brmi=Base.invokelatest(getfield(mod,name))
        sb=BRMDocsComparisons.BRM.SBBRMI(brmi;mod)
        @test length(BRMDocsComparisons.BRM.total_effect_blocks(sb))==blocks
        @test occursin("brm_total_lpdf",read(file,String))
        println("AUTOMATIC_MODEL_PANES_COMPLETE ",which," expected_sampling_dimension=",dimensions)
        flush(stdout)
    end
end
