using Test, BayesianRegressionModels, Distributions
@testset "unbounded censoring and distinct covariance blocks" begin
    data=(;x=[-1.,0.,1.,2.,3.,4.],g=[1,1,2,2,3,3],y=[.2,.3,.5,.6,.9,1.1])
    m=@brm data begin
        mu ~ 1+(1|a|g)+(1+x|b|g)
        y ~ censored(Normal(mu,1.0);lower=0.0)
    end
    r=brm_description(SBBRMI(m;mod=@__MODULE__,total_groups=()))
    @test r.complete
    @test !any(e->occursin("y=\\infty",e) || occursin("textbackslash{}infty",e),r.equations)
    @test any(e->occursin("f(y)&0.0<y",e),r.equations)
    vectors=filter(e->startswith(e,"\\mathbf b_{"),r.equations)
    @test length(vectors)==2
    @test length(unique(first(split(e,"\\sim")) for e in vectors))==2
    @test any(e->occursin("\\mathcal N_{1}",e),vectors)
    @test any(e->occursin("\\mathcal N_{2}",e),vectors)
end
