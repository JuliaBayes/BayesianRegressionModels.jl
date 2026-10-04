using Test, BayesianRegressionModels, Distributions
import StanBlocks
module DescriptionHigherOrder
using StanBlocks
@deffun derivative(t::real,state::vector[n])::vector[n] = -state
end
const derivative=DescriptionHigherOrder.derivative

@testset "higher-order callable arguments retain identity" begin
    data=(;sample_times=[[0.1,0.4],[0.2,0.5,0.8]],y=[0.9,0.8])
    m=@brm data begin
        pred ~ kernel(sample_times) do ts
            trajectory = ode_rk45(derivative,[1.0],0.0,to_array_1d(ts))
            sum(trajectory[1])
        end
        y ~ Normal(pred,1.0)
    end
    d=brm_descriptor(SBBRMI(m;mod=@__MODULE__,total_groups=()))
    r=brm_description(d)
    ode=only(filter(c->c.callable===StanBlocks.stan.builtin.ode_rk45,
        brm_description_components(r)))
    @test first(ode.arguments)===derivative
    @test brm_description_math(ode,first(ode.arguments))=="\\mathrm{derivative}"
    @test brm_description_binding(ode,:ts).value.name===:sample_times
    @test !r.complete
    @test any(c->c.id==ode.id && c.status===:unsupported,r.coverage)
    @test any(c->c.callable===StanBlocks.stan.builtin.to_array_1d,ode.children)
    partial=brm_description(d;hooks=(ode.callable=>c->BRMDescriptionFragment(covers=(c.id,)),))
    @test !partial.complete
end
