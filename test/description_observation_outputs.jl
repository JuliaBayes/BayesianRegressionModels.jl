using Test, BayesianRegressionModels, Distributions, CategoricalArrays
import StanBlocks

@testset "embedded response predictive outputs and conditioning" begin
    data=(;ts=[[0.1,0.4],[0.2,0.5,0.7]],response=[[0.8,1.1],[0.9,1.2,1.4]],other=[1.1,1.4])
    m=@brm data begin
        pred ~ kernel(ts,response) do t,yy
            signal=t .+ 0.7
            yy ~ normal(signal,0.2)
            signal
        end
        other ~ Normal(1.0,0.5)
    end
    for held in ((),(:response,))
        d=brm_descriptor(SBBRMI(m;mod=@__MODULE__,held_out=held,total_groups=()))
        r=brm_description(d)
        @test r.complete
        expected=only(filter(o->o.logical===:response && o.role===:posterior_predictive,d.outputs))
        obs=only(filter(c->c.kind===:observation && c.provenance.owner===:yy,brm_description_components(r)))
        @test any(o->o.name===expected.name && o.logical===:response && o.role===:posterior_predictive,obs.outputs)
        @test obs.provenance.observation_role== (isempty(held) ? :conditioned : :held_out)
        law=only(filter(c->c.callable===StanBlocks.stan.builtin.normal,obs.children))
        @test any(o->o.name===expected.name && o.role===:posterior_predictive,law.outputs)
        @test law.provenance.observation_role==obs.provenance.observation_role
    end
end

@testset "ragged observed aliases retain conditioning provenance" begin
    data=(;subject=["A","B"],event_subject=["A","A","B","B"],
        time=[.1,.3,.2,.4],response=[.2,.5,.3,.6],other=[.1,.2])
    model=@brm data begin
        eta ~ 1+(1|subject)
        state ~ kernel(ragged(time,event_subject),ragged(response,event_subject),eta) do ts,yy,m
            signal=ts.+m
            yy ~ normal(signal,0.2)
            signal
        end
        other ~ Normal(0,1)
    end
    for held in ((),(:yy,))
        d=brm_descriptor(SBBRMI(model;mod=@__MODULE__,held_out=held,total_groups=()))
        r=brm_description(d)
        @test r.complete
        obs=only(filter(c->c.kind===:observation && c.provenance.owner===:yy,brm_description_components(r)))
        @test obs.provenance.observation_sources==(:response,)
        @test obs.provenance.observation_role== (isempty(held) ? :conditioned : :held_out)
        @test all(p->!occursin("`yy` is unconditioned",p),r.prose)
        law=only(filter(c->c.callable===StanBlocks.stan.builtin.normal,obs.children))
        @test law.provenance.observation_role===obs.provenance.observation_role
    end
end

@testset "categorical snapshots retain labels rather than pool state" begin
    data=(;tier=categorical(["A","B","A","B","A","B"];ordered=true),
        y=[.3,.6,.4,.7,.5,.8])
    m=@brm data begin
        mu ~ 1+tier
        y ~ Normal(mu,1.0)
    end
    r=brm_description(SBBRMI(m;mod=@__MODULE__,total_groups=()))
    @test r.complete
    md=brm_description_markdown(r)
    @test occursin("A",md) && occursin("B",md)
    @test !occursin("levelsinds",md) && !occursin("pool =",md) && !occursin("Ptr{",md)
    contrast=only(filter(p->get(p.source,:predictor,nothing)===:tier,r.priors))
    @test contrast.source.level=="B" && contrast.source.reference=="A"
end
