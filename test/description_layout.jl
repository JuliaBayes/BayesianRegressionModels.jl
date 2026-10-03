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
    bindings=only(filter(p->occursin("cell input bindings",p),r.prose))
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
