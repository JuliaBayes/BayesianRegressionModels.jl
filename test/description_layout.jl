using Test, Statistics, BayesianRegressionModels, Distributions
import StanBlocks, Markdown

module DescriptionWideCell
using StanBlocks
@deffun combine_state(t::vector[n],baseline::real,amplitude::real,
    lower_bound::real,upper_bound::real,decay_rate::real,
    onset_time::real,normalization::real)::vector[n] =
    baseline .+ amplitude .* t .+ lower_bound .+ upper_bound .+
    decay_rate .+ onset_time .+ normalization
end

@testset "wide prior laws keep complete definitions outside index cells" begin
    data=(;y=[.2,.4,.6])
    model=@brm data begin
        location_component ~ Normal(0,1)
        scale_component ~ Exponential(1)
        wide_parameter ~ Normal(
            exp(location_component)+exp(location_component)^2+exp(location_component)^3,
            sqrt(scale_component+scale_component^2+scale_component^3))
        y ~ Normal(wide_parameter,1)
    end
    sb=SBBRMI(model;mod=@__MODULE__,total_groups=())
    code=stan_code(sb); frozen=deepcopy(sb.data)
    id=(:parameter,:wide_parameter)
    r=brm_description(sb;prior_anchors=Dict(id=>"https://example.org/prior"))
    @test r.complete
    latent=only(filter(c->c.provenance.owner===:wide_parameter,r.components))
    @test latent.provenance.observation_role===:latent_parameter
    @test any(o->o.role===:parameter,latent.outputs)
    @test any(p->startswith(p,"`wide_parameter` is an unobserved sampled parameter"),r.prose)
    index=only(findall(p->p.id==id,r.priors))
    md=brm_description_markdown(r;prefix="wide prior law")
    prior_rows=filter(row->startswith(row,"| <a id="),split(md,'\n'))
    @test length(prior_rows)==length(r.priors)
    @test all(row->!occursin("\$",row) && length(row)<240,prior_rows)
    @test occursin("[prior listing](https://example.org/prior)",md)
    @test occursin("\\xi_{P"*string(index)*",",md)
    @test occursin("\\begin{aligned}",md)
    @test all(name->occursin("\\mathrm{"*replace(string(name),"_"=>"\\_")*"}",md),
        (:location_component,:scale_component))
    @test occursin("\\mathcal N\\left(\\xi_",md) && occursin("}^{2}",md)
    @test all(p->count("<a id=\""*brm_description_prior_anchor(p;prefix="wide prior law")*"\"",md)==1,r.priors)
    @test stan_code(sb)==code && isequal(sb.data,frozen)
    explicit=brm_description(sb;labels=Dict(:y=>(;meaning="y",unit="mg/day")))
    @test occursin("Meaning: y\n",brm_description_markdown(explicit))
end
const combine_state=DescriptionWideCell.combine_state

@testset "wide grouped relations retain all bindings and child coverage" begin
    data=(;event_times=[[.1,.2],[.3,.4]],baseline_values=[1.,2.],
        amplitude_values=[.2,.3],lower_boundary=[0.,0.],upper_boundary=[4.,5.],
        decay_rate_values=[.5,.7],event_onset_times=[0.,.1],normalization_values=[1.,1.],
        observed_values=[[1.,2.],[2.,3.]])
    model=@brm data begin
        prediction ~ kernel(event_times,baseline_values,amplitude_values,
            lower_boundary,upper_boundary,decay_rate_values,event_onset_times,
            normalization_values,observed_values) do times,baseline,amplitude,lower,upper,decay,onset,norm,observed
            state=combine_state(times,baseline,amplitude,lower,upper,decay,onset,norm)
            observed ~ normal(state,1.0)
            state
        end
    end
    sb=SBBRMI(model;mod=@__MODULE__,total_groups=())
    code=stan_code(sb); prepared=deepcopy(sb.data)
    @test !brm_description(sb).complete
    @test !brm_description(sb;hooks=(kernel=>c->BRMDescriptionFragment(covers=(c.id,)),)).complete
    r=brm_description(sb;hooks=(combine_state=>c->BRMDescriptionFragment(
        prose=("The public toy combines its eight supplied inputs.",),covers=(c.id,)),))
    @test r.complete
    call=only(filter(c->c.callable===combine_state,brm_description_components(r)))
    @test length(call.arguments)==8
    @test Tuple(brm_description_binding(call,k).value.name for k in
        (:times,:baseline,:amplitude,:lower,:upper,:decay,:onset,:norm))==keys(data)[1:8]
    relation=only(filter(e->occursin("combine\\_state",e),r.equations))
    @test occursin("\\begin{aligned}",relation)
    @test count(line->occursin("&\\quad",line),split(relation,"\n"))==8
    for argument in call.arguments
        @test occursin(brm_description_math(call,argument),relation)
    end
    map=only(filter(e->startswith(e,"\\mathcal K_"),r.equations))
    @test map=="\\mathcal K_{prediction,g_j}=\\mathrm{prediction.state}"
    @test all(e->!occursin("\\mathcal K_{prediction,g_j}\\left(",e),r.equations)
    bindings=join(filter(p->startswith(p,"Cell input `"),r.prose),"\n\n")
    html=Markdown.html(Markdown.parse(bindings))
    @test all(k->occursin(string(k),html),(:times,:baseline,:amplitude,:lower,:upper,:decay,:onset,:norm))
    @test stan_code(sb)==code && isequal(sb.data,prepared)
end

@testset "multivariate standardization has named fitted coordinates" begin
    data=(;exposure_measure=[.2,.4,.7,1.1,1.6],
        baseline_measure=[1.4,1.3,1.2,1.1,1.0],
        duration_measure=[2.,3.,4.,6.,8.],
        y=[.1,.2,.3,.5,.7])
    model=@brm data begin
        mu ~ 1+zscale(exposure_measure)+zscale(baseline_measure)+zscale(duration_measure)
        y ~ Normal(mu,1.0)
    end
    sb=SBBRMI(model;total_groups=())
    code=stan_code(sb); prepared=deepcopy(sb.data)
    r=brm_description(sb)
    @test r.complete
    predictor=only(filter(c->c.kind===:predictor,r.components))
    population=only(filter(e->occursin("\\mathrm{mu}&=",e),r.equations))
    @test occursin("\\begin{aligned}",population)
    @test !occursin("\\frac",population)
    @test count(line->occursin("+",line),split(population,"\n"))==3
    for (k,column) in enumerate(predictor.provenance.design_columns)
        column.label===:Intercept && continue
        prepared_column=only(filter(n->n.name==(:prepared_column,:mu,column.label),r.notation))
        @test occursin(prepared_column.symbol,population)
        definition=only(filter(e->startswith(e,prepared_column.symbol*"="),r.equations))
        @test occursin(brm_description_math(predictor,column.preprocess.raw_ref),definition)
        fitted=column.preprocess.const_
        values_=fitted isa NamedTuple ? Tuple(values(fitted)) : fitted
        for (role,value) in zip((:center,:scale),values_)
            n=only(filter(n->n.name==(:fitted,:mu,column.label,role),r.notation))
            @test n.symbol*"="*string(value) in r.equations
        end
        @test (:population,:mu,column.label) in Tuple(p.id for p in r.priors)
    end
    @test stan_code(sb)==code && isequal(sb.data,prepared)
    @test brm_description(sb).equations==r.equations
end

@testset "long likelihood arguments have exact separately defined quantities" begin
    data=(;event_times=[[.1,.2],[.3,.4]],baseline_values=[1.,2.],
        amplitude_values=[.2,.3],lower_boundary=[0.,0.],upper_boundary=[4.,5.],
        decay_rate_values=[.5,.7],event_onset_times=[0.,.1],normalization_values=[1.,1.],
        observed_values=[[1.,2.],[2.,3.]])
    model=@brm data begin
        prediction ~ kernel(event_times,baseline_values,amplitude_values,
            lower_boundary,upper_boundary,decay_rate_values,event_onset_times,
            normalization_values,observed_values) do ts,b,a,l,u,d,o,n,yy
            yy ~ normal(combine_state(ts,b,a,l,u,d,o,n),
                addprop(combine_state(ts,b,a,l,u,d,o,n),.1,.2)+
                addprop(combine_state(ts,b,a,l,u,d,o,n),.2,.3))
            ts
        end
    end
    sb=SBBRMI(model;mod=@__MODULE__,total_groups=())
    code=stan_code(sb); frozen=deepcopy(sb.data)
    @test !brm_description(sb).complete
    hook=c->BRMDescriptionFragment(covers=(c.id,))
    r=brm_description(sb;hooks=(combine_state=>hook,))
    @test r.complete
    likelihood=only(filter(e->occursin("\\sim",e),r.equations))
    @test length(likelihood)<160
    @test occursin("\\xi_{",likelihood) && occursin("}^{2}",likelihood)
    quantities=filter(n->n.name isa Tuple && first(n.name)===:description_expression,r.notation)
    @test !isempty(quantities)
    @test length(unique(n.symbol for n in quantities))==length(quantities)
    @test all(n->any(e->occursin(n.symbol*"=",e) || occursin(n.symbol*"&=",e),r.equations),quantities)
    definitions=join(r.equations,"\n")
    @test all(key->occursin(replace(string(key),"_"=>"\\_"),definitions),keys(data)[1:8])
    @test any(e->occursin("combine\\_state",e) && occursin("\\begin{aligned}",e),r.equations)
    @test !any(e->occursin("BRMDescription",e),r.equations)
    @test stan_code(sb)==code && isequal(sb.data,frozen)
end

@testset "prior rows wrap support and retain complete long logical IDs" begin
    data=(;y=[.2,.4,.6])
    model=@brm data begin
        extraordinarily_long_authored_parameter_name ~ Exponential(.7)
        y ~ Normal(extraordinarily_long_authored_parameter_name,1)
    end
    r=brm_description(SBBRMI(model;mod=@__MODULE__,total_groups=()))
    @test r.complete
    prior=only(r.priors)
    md=brm_description_markdown(r;prefix="wide priors")
    target=brm_description_prior_anchor(r,prior.id;prefix="wide priors")
    rows=filter(row->startswith(row,"| <a id="),split(md,'\n'))
    @test length(rows)==1 && occursin("></a>P1 |",only(rows))
    @test count(target,md)>=1
    @test count("<a id=\""*target*"\"",md)==1
    @test occursin("Prior definitions (complete logical IDs",md)
    @test all(part->occursin("`"*string(part)*"`",md),prior.id)
    lawcell=split(only(rows)," | ")[2]
    @test occursin("definition P1 below",lawcell) && !occursin("\$",lawcell)
    @test occursin("\\mathrm{lower}=0",md)
    rate=only(prior.distribution.arguments)
    @test rate isa BRMDescriptionComponent && rate.callable===(/)
    @test rate.arguments==(1,.7)
    @test occursin("\\operatorname{Exponential}_{\\mathrm{rate}}",md)
    @test occursin(brm_description_math(prior.distribution,rate),md)
    @test !occursin(string(last(prior.id)),lawcell)
    parsed=Markdown.parse(md)
    @test occursin("extraordinarily_long_authored_parameter_name",Markdown.html(parsed))
    @test !occursin("\\sim",lawcell)
    @test occursin("[P1](#"*target*")",md)
    @test !occursin("| `(:",md)
    supplied=brm_description(SBBRMI(model;mod=@__MODULE__,total_groups=());
        labels=Dict(:extraordinarily_long_authored_parameter_name=>
            (;meaning="Measured response anchor.",unit="mg/day")))
    supplied_md=brm_description_markdown(supplied)
    @test occursin("Meaning: Measured response anchor.",supplied_md)
    @test occursin("Supplied unit: mg/day",supplied_md)
    @test !occursin("| Quantity |",supplied_md)
    @test !occursin("Meaning: y",supplied_md)
end
