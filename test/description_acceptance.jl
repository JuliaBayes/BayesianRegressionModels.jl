# Independent public fixtures for the scientific reporting contract. No fit or
# executable model build is needed; the selected SBBRMI descriptor is reused.
using Test, Statistics, BayesianRegressionModels, Distributions
using BayesianRegressionModels: aweights, fweights
if !isdefined(@__MODULE__,:DESCRIPTION_DATA)
    const DESCRIPTION_DATA=(;x=[-2.0,-0.5,0.7,2.5,3.0,4.2],g=[1,1,2,2,3,3],
        y=[0.4,0.1,1.1,1.7,2.4,2.8],z=[1.4,1.1,2.1,2.7,3.4,3.8])
end

@testset "fitted design, contrasts, interactions and explicit link" begin
    m=@brm DESCRIPTION_DATA begin
        sigma ~ Exponential(2)
        log(mu) ~ 0 + factor(g;ref=2,cmc=false) + zscale(x) + x & g
        y ~ Normal(mu,sigma)
    end
    sb=SBBRMI(m;total_groups=())
    d=brm_descriptor(sb)
    r=brm_description(d)
    @test r.complete
    @test isempty(r.diagnostics)
    predictor=only(filter(c->c.kind===:predictor,r.components))
    @test any(c->c.label===:zscale_x,predictor.provenance.design_columns)
    @test isapprox(only(filter(k->k.kind===:zscale,predictor.fitted_constants)).value[1],mean(DESCRIPTION_DATA.x))
    @test any(e->occursin("\\log",e) && occursin("\\frac",e),r.equations)
    @test any(e->occursin("int\\_x\\_x\\_g\\_lvl",e) && occursin("\\mathbf1",e),r.equations)
    cats=filter(p->get(p.source,:predictor,nothing)===:g,r.priors)
    @test length(cats)==2
    @test all(p->p.source.reference==2 && p.source.coding===:treatment,cats)
    @test only(filter(p->p.id==(:parameter,:sigma),r.priors)).support.lower==0.0
    frozen=deepcopy(sb.data)
    brm_description(d;labels=Dict(:mu=>(;meaning="positive mean",unit="supplied unit")))
    @test isequal(sb.data,frozen)
    # Keep the ref= contrast fixture separate: its reprocessing currently
    # requires an emitted factor carrier, unrelated to description.
    simple=@brm DESCRIPTION_DATA begin
        mu ~ 1 + zscale(x)
        y ~ Normal(mu,1)
    end
    prepared=SBBRMI(simple;total_groups=())
    original=brm_description(prepared)
    fresh=reprocess(prepared,merge(DESCRIPTION_DATA,(;x=DESCRIPTION_DATA.x .+ 10));freeze_constants=false)
    @test brm_description(fresh).equations!=original.equations
    replay=reprocess(prepared,merge(DESCRIPTION_DATA,(;x=DESCRIPTION_DATA.x .+ 10)))
    @test brm_description(replay).equations==original.equations
end

@testset "multiple outcomes, held-out and unconditioned observations" begin
    builder=@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x + (1 | shared | g)
        nu ~ 1 + x + (1 | shared | g)
        y ~ Normal(mu,sigma)
        z ~ Normal(nu,sigma)
    end
    sb=SBBRMI(builder(DESCRIPTION_DATA);held_out=:z,total_groups=())
    r=brm_description(sb)
    @test r.complete
    obs=filter(c->c.kind===:observation,r.components)
    @test only(filter(c->c.provenance.owner===:y,obs)).provenance.observation_role===:conditioned
    @test only(filter(c->c.provenance.owner===:z,obs)).provenance.observation_role===:held_out
    @test count(c->c.kind===:random_effect,r.components)==1
    @test any(e->occursin("D\\Omega D",e),r.equations)
    prior_only=builder((;x=DESCRIPTION_DATA.x,g=DESCRIPTION_DATA.g))
    pr=brm_description(SBBRMI(prior_only;total_groups=()))
    @test pr.complete
    @test all(c->c.provenance.observation_role===:unconditioned,filter(c->c.kind===:observation,pr.components))
    @test any(p->p.id==(:population,:mu,:x),pr.priors)
end

@testset "observation evidence and weight type" begin
    df=merge(DESCRIPTION_DATA,(;w=[1.,2.,1.,3.,2.,1.],y_upper=DESCRIPTION_DATA.z .+ 0.5))
    m=@brm df begin
        mu ~ 1 + x
        y ~ weighted(Normal(mu,1.5),aweights(w))
        z ~ interval_censored(Normal(mu,1.0);upper=y_upper)
    end
    r=brm_description(SBBRMI(m;total_groups=()))
    @test r.complete
    @test isempty(r.diagnostics)
    @test any(e->occursin("\\sigma^2/",e),r.equations)
    @test any(e->occursin("F(L_j-1)",e),r.equations)
    wrappers=@brm DESCRIPTION_DATA begin
        mu ~ 1 + x
        y ~ censored(Normal(mu,1.0);lower=0.0,upper=3.0)
        z ~ truncated(Normal(mu,1.0);lower=0.0)
    end
    wr=brm_description(SBBRMI(wrappers;total_groups=()))
    @test wr.complete
    @test any(e->occursin("\\begin{cases}",e),wr.equations)
    @test any(e->occursin("\\frac{f(y)",e),wr.equations)
    generic=@brm df begin
        mu ~ 1 + x
        y ~ weighted(Normal(mu,1.5),fweights(w))
    end
    gr=brm_description(SBBRMI(generic;total_groups=()))
    @test gr.complete
    @test any(e->occursin("\\log\\mathcal L=",e),gr.equations)
end

@testset "smooth, missing covariate and generated prior inventory" begin
    smoothdata=(;x=collect(range(-2.,4.;length=12)),y=sin.(range(-2.,4.;length=12)))
    smooth=@brm smoothdata begin
        mu ~ 0 + s(x)
        y ~ Normal(mu,1.0)
    end
    r=brm_description(SBBRMI(smooth;total_groups=()))
    @test r.complete
    @test isempty(r.diagnostics)
    @test any(p->p.source.kind===:flat && p.source.proper===false,r.priors)
    @test any(p->last(p.id)===:b_pen_raw,r.priors)
    @test any(p->last(p.id)===:sd_pen && p.support.lower==0.0,r.priors)
    @test any(c->any(k->k.kind===:spline,c.fitted_constants),r.components)
    missingdf=merge(DESCRIPTION_DATA,(;x=Union{Missing,Float64}[-2.,missing,.7,2.5,missing,4.2]))
    missingmodel=@brm missingdf begin
        mi(x) ~ Normal(0,1)
        mu ~ 1 + center(x)
        y ~ Normal(mu,1.0)
    end
    mr=brm_description(SBBRMI(missingmodel;total_groups=()))
    @test mr.complete
    @test isempty(mr.diagnostics)
    @test any(c->c.callable===mi,brm_description_components(mr))
    @test any(p->get(p.source,:dimension,())!=(),mr.priors)
end
