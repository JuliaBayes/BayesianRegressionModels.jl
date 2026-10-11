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

# An omitted in-cell response has no `_gen` twin: its forward-simulated carrier
# is claimed under the cell-local declaration (brm-use "Prior draws"). Its
# embedded observation context lists that carrier exactly as an omitted
# top-level response lists its twin (snag describe-a-joint-8a53c3b9).
@testset "omitted embedded responses list their forward-simulated carriers" begin
    legacy=@brm begin
        pred ~ kernel(ts,response) do t,yy
            signal=t .+ 0.7
            yy ~ normal(signal,0.2)
            signal
        end
        other ~ Normal(level,0.5)
    end
    plate=@brm begin
        @plate for i in eachindex(ts)
            signal=ts[i] .+ 0.7
            response[i] ~ normal(signal,0.2)
            pred[i]=signal
        end
        other ~ Normal(level,0.5)
    end
    ts=[[0.1,0.4],[0.2,0.5,0.7]]; level=[1.0,1.3]
    for (builder,cell) in ((legacy,:yy),(plate,:response)),
        (data,top) in (((;ts,level,other=[1.1,1.4]),:conditioned),((;ts,level),:unconditioned))
        d=brm_descriptor(SBBRMI(builder(data);mod=@__MODULE__,total_groups=()))
        r=brm_description(d)
        @test r.complete
        components=brm_description_components(r)
        expected=brm_output(d,cell;role=:posterior_predictive)
        obs=only(filter(c->c.kind===:observation && c.provenance.owner===cell,components))
        @test obs.provenance.observation_role===:unconditioned
        @test any(o->o.name===expected.name && o.logical===cell && o.role===:posterior_predictive,obs.outputs)
        law=only(filter(c->c.callable===StanBlocks.stan.builtin.normal,obs.children))
        @test any(o->o.name===expected.name && o.role===:posterior_predictive,law.outputs)
        @test law.provenance.observation_role===:unconditioned
        # The top-level response beside it: bound, or omitted with its twin.
        other=only(filter(c->c.kind===:observation && c.provenance.owner===:other,components))
        @test other.provenance.observation_role===top
        twin=brm_output(d,:other;role=:posterior_predictive)
        @test any(o->o.name===twin.name,other.outputs)
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
