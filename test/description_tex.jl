using Test, BayesianRegressionModels, Distributions
import StanBlocks

# Synthetic public models exercise standard, affine and embedded Student-t
# equations, including the compact-expression rendering path.
const TEX_DESCRIPTION_DATA=(;x=[-0.4,0.2,0.8],y=[0.1,0.3,0.7])
const TEX_DESCRIPTION_MODELS=(
    standard=@brm(TEX_DESCRIPTION_DATA, begin
        theta ~ TDist(7.0)
        y ~ TDist(4.0)
    end),
    affine=@brm(TEX_DESCRIPTION_DATA, begin
        mu ~ 1+x
        sigma ~ Exponential(1)
        y ~ LocationScale(mu,sigma,TDist(5.0))
    end),
    wide=@brm(TEX_DESCRIPTION_DATA, begin
        mu ~ 1+x
        sigma ~ Exponential(1)
        y ~ LocationScale(mu,
            exp(sigma)+exp(sigma)^2+exp(sigma)^3+exp(sigma)^4+exp(sigma)^5,
            TDist(8.0))
    end),
    embedded=@brm((;times=[[0.1,0.3],[0.2,0.4,0.6]],
                    readings=[[0.2,0.4],[0.3,0.5,0.7]]), begin
        predicted ~ kernel(times,readings) do ts,yy
            yy ~ student_t(6.0,ts,0.4)
            ts
        end
    end),
)

let descriptions=Pair{Symbol,String}[]
    @testset "Student-t descriptions preserve laws and TeX token boundaries" begin
        for (name,model) in pairs(TEX_DESCRIPTION_MODELS)
            sb=SBBRMI(model;mod=@__MODULE__,total_groups=())
            descriptor=brm_descriptor(sb)
            source=stan_code(sb); data=deepcopy(sb.data)
            result=brm_description(descriptor)
            markdown=brm_description_markdown(result;prefix="tex-"*string(name))
            @test result.complete
            @test isempty(result.diagnostics)
            @test result.model_id==descriptor.id
            @test !occursin(raw"\simt",markdown)
            @test occursin("Student-t",markdown)
            @test occursin("student_t(",source)
            @test stan_code(sb)==source
            @test isequal(sb.data,data)
            @test brm_descriptor(sb).id==descriptor.id
            if name===:standard
                @test any(e->occursin(raw"t_{4.0}(0,1)",e),result.equations)
                @test occursin(raw"t_{7.0}(0,1)",markdown)
                @test only(filter(p->p.id==(:parameter,:theta),result.priors)).distribution.arguments==(7.0,0,1)
            elseif name in (:affine,:wide)
                @test any(e->startswith(e,"Z") && occursin(raw"t_{",e),result.equations)
                @test any(e->occursin("f_Z",e) && occursin(raw"\frac{1}",e),result.equations)
                @test any(p->occursin("ν/(ν−2)",p),result.prose)
                name===:wide && @test any(e->startswith(e,raw"\begin{aligned}"),result.equations)
            else
                @test any(e->occursin(raw"t_{6.0}(",e) && occursin(",0.4)",e),result.equations)
                @test only(filter(c->c.kind===:observation,brm_description_components(result))).provenance.observation_role===:conditioned
            end
            push!(descriptions,name=>markdown)
        end
    end

    # PDF compilation is an explicit acceptance mode, so the ordinary Julia check
    # stays usable on hosts without a TeX installation. --pdf requires the tools;
    # failures propagate, including from LuaLaTeX rather than only Pandoc-to-TeX.
    if "--pdf" in ARGS
        @testset "public Markdown compiles through Quarto and LuaLaTeX" begin
            @test !isnothing(Sys.which("quarto"))
            output=get(()->mktempdir(),ENV,"BRM_DESCRIPTION_TEX_OUTPUT")
            mkpath(output)
            document="---\ntitle: Student-t description regression\nformat:\n  pdf:\n    pdf-engine: lualatex\n    keep-tex: true\n---\n\n"*
                join(("# "*string(name)*"\n\n"*markdown for (name,markdown) in descriptions),"\n\n")
            input=joinpath(output,"description.qmd")
            write(input,document)
            run(`quarto render $input --to pdf`)
            @test isfile(joinpath(output,"description.pdf"))
            @test filesize(joinpath(output,"description.pdf"))>0
        end
    end
end
