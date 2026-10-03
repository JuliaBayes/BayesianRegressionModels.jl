using Test, BayesianRegressionModels, Distributions
@testset "weighted pooled multi-membership axes" begin
    data=(;g1=["a","a","b"],g2=["b","c","c"],
        w1=[2.,1.,0.],w2=[1.,1.,3.],y=[.2,.5,.7])
    normalized=@brm data begin
        mu ~ 1+(1|mm(g1,g2;weights=(w1,w2)))
        y ~ Normal(mu,1.0)
    end
    r=brm_description(SBBRMI(normalized;mod=@__MODULE__,total_groups=()))
    @test r.complete
    @test any(e->occursin("\\sum_m\\widetilde w_{jm}",e),r.equations)
    @test any(e->occursin("w_{jm}/\\sum_h w_{jh}",e),r.equations)
    block=only(filter(c->c.kind===:random_effect,r.components))
    @test block.keywords.n_groups==3
    @test block.keywords.group==(:g1,:g2)
    @test block.keywords.margins==((predictor=:mu,coefficient=:Intercept),)
    raw=@brm data begin
        mu ~ 1+(1|mm(g1,g2;weights=(w1,w2),normalize=false))
        y ~ Normal(mu,1.0)
    end
    rr=brm_description(SBBRMI(raw;mod=@__MODULE__,total_groups=()))
    @test rr.complete
    @test "\\widetilde w_{jm}=w_{jm}" in rr.equations
end
