using Test, BayesianRegressionModels, Distributions
import Markdown
import StanBlocks
module DescriptionStateCoupling
using StanBlocks
@deffun opaque_state(t::vector[n])::vector[n] = t .* t .+ 0.4
end
const opaque_state=DescriptionStateCoupling.opaque_state
@testset "explicit grouped state-to-response links" begin
    data=(;t_grid=[[0.1,0.3,0.6],[0.2,0.5]],index_grid=[[1,2],[1,2]],
        y_grid=[[1.4,1.6],[2.5,2.8]],base=[1.,2.],amp=[2.,3.])
    m=@brm data begin
        pred ~ kernel(t_grid,index_grid,y_grid,base,amp) do ts,lookup_idx,y,baseline,amplitude
            state=opaque_state(ts)
            signal=state[lookup_idx]
            mean=baseline+amplitude.*signal
            scale=0.1+inv_logit(amplitude)
            y ~ normal(mean,scale)
            state
        end
    end
    d=brm_descriptor(SBBRMI(m;mod=@__MODULE__,total_groups=()))
    @test !brm_description(d).complete
    hook=c->BRMDescriptionFragment(equations=(brm_description_symbol(c,:state)*"="*
        brm_description_math(c,only(c.arguments))*"^2+0.4",),covers=(c.id,))
    r=brm_description(d;hooks=(opaque_state=>hook,))
    @test r.complete
    @test isempty(r.diagnostics)
    @test any(e->startswith(e,"\\mathrm{pred.signal}=") && occursin("{\\mathrm{pred.state}}_{\\mathrm{index\\_grid}}",e),r.equations)
    @test any(e->startswith(e,"\\mathrm{pred.mean}=") && occursin("\\mathrm{base}",e) &&
        occursin("\\mathrm{amp}",e) && occursin("\\mathrm{pred.signal}",e),r.equations)
    @test any(e->startswith(e,"\\mathrm{pred.scale}=") && occursin("\\frac{1}{1+\\exp",e),r.equations)
    @test any(e->occursin("\\mathrm{pred.mean}",e) && occursin("{\\mathrm{pred.scale}}^{2}",e),r.equations)
    @test any(e->startswith(e,"\\mathcal K_{pred,g_j}") && endswith(e,"=\\mathrm{pred.state}"),r.equations)
    @test all(e->!occursin("\\mathrm{block}",e) && !occursin("\\mathrm{->}",e),r.equations)
    statecall=only(filter(c->c.callable===opaque_state,brm_description_components(r)))
    @test only(statecall.outputs).logical===:state
    @test any(c->c.callable===StanBlocks.stan.builtin.inv_logit,brm_description_components(r))
    observation=only(filter(c->c.kind===:observation,brm_description_components(r)))
    mean=observation.arguments[2].arguments[1]
    @test mean isa BRMDescriptionReference && mean.axis===:cell && mean.name===:mean
    @test brm_description_math(observation,mean)=="\\mathrm{pred.mean}"
    binding_prose=only(filter(p->occursin("cell input bindings",p),r.prose))
    markdown=Markdown.parse(binding_prose)
    @test occursin("<code>lookup_idx</code>",Markdown.html(markdown))
    @test occursin(raw"$\mathrm{index\_grid}$",Markdown.latex(markdown))
end
