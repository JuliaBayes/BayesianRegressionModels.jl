using Test, BayesianRegressionModels, Distributions
import StanBlocks

module DescriptionNamedRecords
using BayesianRegressionModels, StanBlocks
function record_term end
@deffun opaque_record(record_values,input::vector[n])::vector[n] =
    record_values.offset .+ record_values.multiplier .* input
const cell=@slic begin
    offset ~ normal(0,1)
    multiplier=2.0
    lower=0.1
    upper=5.0
    parameters=(;lower,upper,offset,multiplier,prepared_values=input,
        weights=exp(offset),normalizer=0.0)
    state=opaque_record((;lower,upper,offset,multiplier,prepared_values=input,
        weights=exp(offset),normalizer=0.0),input)
    return state
end
function BayesianRegressionModels._sb_submodel_rhs!(stmts,data,target::Symbol,::typeof(record_term),rhs)
    input=only(getargs(rhs)); key=name(input)
    data[key]=collect(parent(parent(input)))
    push!(stmts,Expr(:call,:~,target,Expr(:call,cell,Expr(:parameters,Expr(:kw,:input,key)))))
    :done
end
end
const record_term=DescriptionNamedRecords.record_term

@testset "named included records expose authoritative public fields" begin
    data=(;input=[.1,.4,.8],y=[.2,.5,.9])
    model=@brm data begin
        mu ~ record_term(input)
        y ~ Normal(mu,1)
    end
    sb=SBBRMI(model;mod=@__MODULE__,total_groups=())
    code=stan_code(sb); frozen=deepcopy(sb.data)
    simple=c->BRMDescriptionFragment(covers=(c.id,))
    gap=brm_description(sb;hooks=(record_term=>simple,))
    @test !gap.complete
    r=brm_description(sb;hooks=(record_term=>simple,DescriptionNamedRecords.opaque_record=>simple))
    @test r.complete
    context=only(filter(c->c.callable===DescriptionNamedRecords.opaque_record,brm_description_components(r)))
    fields=brm_description_record(context,first(context.arguments))
    @test keys(fields)==(:lower,:upper,:offset,:multiplier,:prepared_values,:weights,:normalizer)
    @test brm_description_math(context,fields.lower)=="0.1"
    @test brm_description_math(context,fields.upper)=="5.0"
    @test brm_description_math(context,fields.multiplier)=="2.0"
    @test fields.normalizer==0
    @test brm_description_record(context,(;normalizer=0)).normalizer===0
    @test fields.offset.logical==(:parameter,:mu,:offset)
    @test brm_description_math(context,fields.prepared_values)=="\\mathrm{input}"
    @test fields.weights isa BRMDescriptionComponent && fields.weights.callable===exp
    @test only(brm_description_binding(context,:offset).prior_ids) in brm_description_prior_references(context)
    @test brm_description_record(context,fields)===fields
    @test occursin("\\mathrm{callable}=\\mathrm{exp}",brm_description_math(context,(;callable=exp,scale=2)))
    @test_throws ArgumentError brm_description_record(context,context.arguments[2])
    relation=only(filter(e->startswith(e,"\\begin{aligned}\\mathrm{parameters}&="),r.equations))
    @test all(name->occursin(replace(string(name),"_"=>"\\_"),relation),keys(fields))
    call_relation=only(filter(e->occursin("opaque\\_record",e),r.equations))
    @test occursin("\\xi_{",call_relation)
    @test !occursin("\\mathrm{prepared\\_values}=",call_relation)
    record_definition=only(filter(e->occursin("\\xi_{",e) && occursin("\\mathrm{prepared\\_values}=",e),r.equations))
    @test occursin("\\begin{aligned}",record_definition)
    @test all(name->occursin(replace(string(name),"_"=>"\\_"),record_definition),keys(fields))
    @test !any(e->occursin("\\mathrm{parameters}\\left(",e) || occursin("\\mathrm{kw}",e),r.equations)
    @test stan_code(sb)==code && isequal(sb.data,frozen)
    @test StanBlocks.stanc_check(code;warn_pedantic=false).ok
    cyclebindings=(
        (;name=:left,role=:alias,value=BRMDescriptionReference(:right,:local),prior_ids=()),
        (;name=:right,role=:alias,value=BRMDescriptionReference(:left,:local),prior_ids=()))
    cycle=BRMDescriptionComponent(context.id,context.kind,context.callable,
        context.arguments,context.keywords,context.axes,context.outputs,context.priors,
        context.fitted_constants,context.children,context.provenance,context.notation,cyclebindings)
    @test_throws ArgumentError brm_description_record(cycle,BRMDescriptionReference(:left,:local))
    key=(:description_expression,context.id,1)
    colliding=BRMDescriptionComponent(context.id,context.kind,context.callable,
        context.arguments,context.keywords,context.axes,context.outputs,context.priors,
        context.fitted_constants,context.children,context.provenance,
        ((;name=key,symbol="E",meaning="Explicitly bound intermediate"),),
        ((;name=:intermediate_expression,role=:alias,value=2,prior_ids=()),))
    @test brm_description_math(colliding,BRMDescriptionReference(:intermediate_expression,:local,key))=="E"
end
