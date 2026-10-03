using Test, BayesianRegressionModels, Distributions
import StanBlocks

const ALLOCATION_DATA=(;x=[-1.,0.,1.,2.,3.,4.],g=[1,1,2,2,3,3],y=[.2,.3,.5,.6,.9,1.1])
@testset "R2D2 allocations and unsupported Horseshoe retain actual priors" begin
    m=@brm ALLOCATION_DATA begin
        mu ~ 1+x+(1|g)
        effect(mu,:) ~ r2d2(R2=Beta(2,3),tau_bsv=0.5)
        y ~ Normal(mu,1.0)
    end
    r=brm_description(SBBRMI(m;mod=@__MODULE__,total_groups=()))
    @test r.complete
    @test any(c->c.kind===:prior_allocation,r.components)
    p=only(filter(p->p.id==(:population,:mu,:x),r.priors))
    @test p.source.kind===:conditional
    @test p.distribution.arguments[2] isa BRMDescriptionReference
    @test any(p->p.distribution.callable===StanBlocks.stan.builtin.beta &&
        p.distribution.arguments==(2,3),r.priors)
    @test any(e->occursin("R^2_{mu}",e) && occursin("\\operatorname{beta}(2.0,3.0)",e),r.equations)
    @test any(e->startswith(e,"s_{mu,x}=") && occursin("V_{mu,x}",e),r.equations)
    @test any(e->startswith(e,"T_{mu}=0.5"),r.equations)
    @test any(e->startswith(e,"\\mathrm{SD}_{") && occursin("R^2_{mu}",e),r.equations)
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

@testset "empty allocations and excluded population priors" begin
    empty_model=@brm ALLOCATION_DATA begin
        mu ~ 1+(1|g)
        effect(mu,:) ~ r2d2(tau_bsv=0.6)
        y ~ Normal(mu,1.0)
    end
    empty_result=brm_description(SBBRMI(empty_model;mod=@__MODULE__,total_groups=()))
    @test empty_result.complete
    @test !any(e->startswith(e,"R^2_") || startswith(e,"\\phi_"),empty_result.equations)
    @test any(e->startswith(e,"\\mathrm{SD}_{") && occursin("T_{mu}",e),empty_result.equations)
    excluded_data=merge(ALLOCATION_DATA,(;z=[.3,.5,.2,.8,.4,.9]))
    excluded=@brm excluded_data begin
        mu ~ 1+x+z+(1|g)
        effect(mu,:) ~ r2d2(tau_bsv=0.5)
        effect(mu,x) ~ Normal(0.2,0.4)
        y ~ Normal(mu,1.0)
    end
    er=brm_description(SBBRMI(excluded;mod=@__MODULE__,total_groups=()))
    @test er.complete
    xprior=only(filter(p->p.id==(:population,:mu,:x),er.priors))
    @test xprior.source.kind===:selector && xprior.distribution.arguments==(0.2,0.4)
    @test !any(e->startswith(e,"s_{mu,x}="),er.equations)
end

@testset "shared-block and joint R2D2 budgets bind actual scales" begin
    data=merge(ALLOCATION_DATA,(;tier=[1,1,2,2,1,2],z=[.3,.5,.2,.8,.4,.9]))
    shared=@brm data begin
        mu ~ 1+x+(1+x|p|g)
        sd(:,p) ~ r2d2(R2=Uniform(0.1,0.8),reference_scale=0.7)
        y ~ Normal(mu,1.0)
    end
    r=brm_description(SBBRMI(shared;mod=@__MODULE__,total_groups=()))
    @test r.complete
    @test isempty(r.diagnostics)
    @test any(e->occursin("\\operatorname{Uniform}(0.1,0.8)",e),r.equations)
    @test any(e->startswith(e,"\\mathrm{SD}_{") && occursin("0.7",e) && occursin("\\frac",e),r.equations)
    @test any(p->p.source.kind===:declaration && p.distribution.callable===StanBlocks.stan.builtin.dirichlet &&
        p.distribution.arguments==((1.0,1.0),),r.priors)
    joint=@brm data begin
        sigma ~ Exponential(1.0)
        a ~ 1+x+tier+(1|p|g)
        b ~ 1+x+(1|p|g)
        sd(:,p) ~ r2d2(mean_R2=0.4,prec_R2=5.0,concentration=0.3,
            reference_scale=sigma,include=(:population,:contrasts))
        y ~ Normal(a,sigma)
        z ~ Normal(b,sigma)
    end
    sb=SBBRMI(joint;mod=@__MODULE__,total_groups=())
    before=stan_code(sb)
    jr=brm_description(sb)
    @test jr.complete
    @test isempty(jr.diagnostics)
    @test stan_code(sb)==before
    @test any(p->p.distribution.callable===StanBlocks.stan.builtin.dirichlet &&
        p.distribution.arguments==((0.3,0.3,0.3,0.3,0.3),),jr.priors)
    @test any(e->startswith(e,"s_{a,x}=") && occursin("\\frac",e) && occursin("R^2_{1,1}",e),jr.equations)
    @test any(e->startswith(e,"s_{b,x}=") && occursin("R^2_{1,1}",e),jr.equations)
    contrasts=filter(p->length(p.id)>=5 && p.id[2]===:a && p.source.kind===:conditional,jr.priors)
    @test length(contrasts)==1
    @test any(e->startswith(e,brm_description_math(first(jr.components),only(contrasts).distribution.arguments[2])*"="),jr.equations)
    @test any(e->endswith(e,"=\\frac{m(n-m)}{n(n-1)}"),jr.equations)
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
