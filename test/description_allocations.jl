using Test, BayesianRegressionModels, Distributions
import StanBlocks

const ALLOCATION_DATA=(;x=[-1.,0.,1.,2.,3.,4.],g=[1,1,2,2,3,3],y=[.2,.3,.5,.6,.9,1.1])
@testset "allocation gaps retain actual conditional and hyperpriors" begin
    m=@brm ALLOCATION_DATA begin
        mu ~ 1+x+(1|g)
        effect(mu,:) ~ r2d2(R2=Beta(2,3),tau_bsv=0.5)
        y ~ Normal(mu,1.0)
    end
    r=brm_description(SBBRMI(m;mod=@__MODULE__,total_groups=()))
    @test !r.complete
    @test any(c->c.kind===:prior_allocation,r.components)
    p=only(filter(p->p.id==(:population,:mu,:x),r.priors))
    @test p.source.kind===:conditional
    @test p.distribution.arguments[2] isa BRMDescriptionReference
    @test any(p->p.distribution.callable===StanBlocks.stan.builtin.beta &&
        p.distribution.arguments==(2,3),r.priors)
    hs=@brm ALLOCATION_DATA begin
        mu ~ 1+x
        effect(mu,x) ~ Horseshoe(local_scale=0.5,global_scale=0.2)
        y ~ Normal(mu,1.0)
    end
    sb=SBBRMI(hs;mod=@__MODULE__,total_groups=())
    before=stan_code(sb)
    hr=brm_description(sb)
    @test !hr.complete
    @test any(c->c.kind===:prior_allocation,hr.components)
    scales=filter(p->p.distribution.callable===StanBlocks.stan.builtin.cauchy,hr.priors)
    @test length(scales)==2
    @test Set(p.distribution.arguments for p in scales)==Set([(0.,0.5),(0.,0.2)])
    @test all(p->p.support.lower==0.0,scales)
    @test any(p->last(p.id)===:raw,hr.priors)
    @test stan_code(sb)==before
end

@testset "exact GP and periodic spectral conventions" begin
    m=@brm ALLOCATION_DATA begin
        mu ~ 1+gp(x;jitter=0.002)
        y ~ Normal(mu,1.0)
    end
    r=brm_description(SBBRMI(m;mod=@__MODULE__,total_groups=()))
    @test r.complete
    @test any(e->occursin("0.002",e) && occursin("\\ell_r^2",e),r.equations)
    periodic=@brm ALLOCATION_DATA begin
        mu ~ 1+hsgp(x;k=3,cov=:periodic,period=4.0)
        y ~ Normal(mu,1.0)
    end
    pr=brm_description(SBBRMI(periodic;mod=@__MODULE__,total_groups=()))
    @test pr.complete
    @test any(e->occursin("I_k(a)",e),pr.equations)
    @test any(e->occursin("/4.0",e),pr.equations)
end
