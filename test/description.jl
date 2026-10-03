using Test
using BayesianRegressionModels
import StanBlocks
using Distributions

const DESCRIPTION_DATA=(; x=[-2.0,-0.5,0.7,2.5,3.0,4.2],
    g=[1,1,2,2,3,3], y=[0.4,0.1,1.1,1.7,2.4,2.8],
    z=[1.4,1.1,2.1,2.7,3.4,3.8])

module DescriptionScientificComponent
using BayesianRegressionModels, StanBlocks
@deffun curve(x::vector[n])::vector[n] = x .* x .+ 0.6
function toy_term end
const toy_cell=@slic begin
    offset ~ normal(anchor,1.0)
    lower_bound = 1.0
    return curve(input .+ offset .+ lower_bound)
end
function BayesianRegressionModels._sb_submodel_rhs!(stmts,data,target::Symbol,::typeof(toy_term),rhs)
    input=only(getargs(rhs))
    input_key=name(input)
    data[input_key]=collect(parent(parent(input)))
    anchor_key=Symbol(target,:_fitted_anchor)
    data[anchor_key]=getkwargs(rhs).anchor
    push!(stmts,Expr(:call,:~,target,Expr(:call,toy_cell,
        Expr(:parameters,Expr(:kw,:input,input_key),Expr(:kw,:anchor,anchor_key)))))
    :done
end
end
const curve=DescriptionScientificComponent.curve
const toy_term=DescriptionScientificComponent.toy_term

@testset "canonical variadic products retain scientific quantities" begin
    model=@brm DESCRIPTION_DATA begin
        b ~ Normal(0,1)
        r ~ Uniform(-1,1)
        s ~ Exponential(1)
        u ~ Exponential(1)
        z ~ Normal(0,1)
        mu = b+r*(s/u)*z*x
        y ~ Normal(mu,1)
    end
    sb=SBBRMI(model;mod=@__MODULE__,total_groups=())
    code=stan_code(sb); frozen=deepcopy(sb.data)
    @test StanBlocks.stanc_check(code;warn_pedantic=false).ok
    result=brm_description(sb)
    @test result.complete
    assignment=only(filter(c->c.kind===:assignment && c.provenance.owner===:mu,result.components))
    product=only(filter(c->c.callable===(*) && length(c.arguments)==4,brm_description_components(assignment)))
    @test brm_description_math(product,product.arguments[1])=="\\mathrm{r}"
    @test brm_description_math(product,product.arguments[3])=="\\mathrm{z}"
    @test brm_description_math(product,product.arguments[4])=="\\mathrm{x}"
    @test Set(p.id for p in result.priors)==Set((:parameter,k) for k in (:b,:r,:s,:u,:z))
    @test any(e->occursin("\\frac{\\mathrm{s}}{\\mathrm{u}}",e) &&
        occursin("\\mathrm{r}",e) && occursin("\\mathrm{z}",e) && occursin("\\mathrm{x}",e),result.equations)
    @test stan_code(sb)==code && isequal(sb.data,frozen)
end

@testset "public reference-valued keywords terminate alias cycles" begin
    data=merge(DESCRIPTION_DATA,(;input=DESCRIPTION_DATA.x))
    m=@brm data begin
        mu ~ toy_term(input;anchor=1.7,input=input)
        y ~ Normal(mu,0.5)
    end
    r=brm_description(SBBRMI(m;mod=@__MODULE__,total_groups=()))
    c=only(filter(c->c.callable===toy_term,brm_description_components(r)))
    @test c.keywords.input isa BRMDescriptionReference
    @test brm_description_binding(c,:input).value.name===:input
    @test brm_description_math(c,c.keywords.input)=="\\mathrm{input}"
    bindings=c.bindings
    withbindings(bs)=BRMDescriptionContext(c.id,c.kind,c.callable,c.arguments,c.keywords,
        c.axes,c.outputs,c.priors,c.fitted_constants,c.children,c.provenance,c.notation,bs)
    a=BRMDescriptionReference(:a,:local)
    b=BRMDescriptionReference(:b,:local)
    cyclic=withbindings(((;name=:a,role=:alias,value=b,prior_ids=()),
        (;name=:b,role=:alias,value=a,prior_ids=())))
    @test brm_description_math(cyclic,a)=="\\mathrm{a}"
    expression=BRMDescriptionComponent((:cycle,),:call,+,(b,1),NamedTuple(),(),(),(),(),(),c.provenance,(),())
    composed=withbindings(((;name=:a,role=:deterministic,value=expression,prior_ids=()),
        (;name=:b,role=:alias,value=a,prior_ids=())))
    @test brm_description_math(composed,a)=="\\mathrm{a}"
    @test brm_description_math(composed,brm_description_binding(composed,:a).value)=="\\left(\\mathrm{a} + 1\\right)"
    @test c.bindings===bindings
end

@testset "prepared hierarchical description and effective priors" begin
    m=@brm DESCRIPTION_DATA begin
        sigma ~ Exponential(2)
        mu ~ 1 + x + (1 + x | q | g)
        effect(:,x) ~ Normal(0,0.8)
        effect(mu,x) ~ Normal(1.2,0.4)
        y ~ Normal(mu,sigma)
    end
    sb=SBBRMI(m;total_groups=())
    d=brm_descriptor(sb)
    source=stan_code(sb)
    data=deepcopy(sb.data)
    r=brm_description(d;prior_anchors=Dict((:population,:mu,:x)=>"#slope-prior"))
    @test r isa BRMDescription
    @test r.model_id==d.id
    @test r.complete
    @test isempty(r.diagnostics)
    @test all(c -> c.status===:covered,r.coverage)
    @test any(e -> occursin("\\Omega_{",e),r.equations)
    @test any(e -> occursin("\\mathcal N",e) && occursin("sigma",e) && occursin("}^{2}",e),r.equations)
    ps=only(filter(p -> p.id===(:population,:mu,:x),r.priors))
    @test ps.distribution.arguments==(1.2,0.4)
    @test ps.anchor=="#slope-prior"
    @test ps.source.kind===:selector
    @test ps.source.selector=="effect(mu, x)"
    @test ps.source.specificity==2
    @test count(p -> :sd in p.id,r.priors)==2
    @test count(p -> :correlation in p.id,r.priors)==1
    @test all(p -> p.support.lower==0.0,filter(p -> :sd in p.id,r.priors))
    @test occursin("[prior listing](#slope-prior)",brm_description_markdown(r))
    anchor=brm_description_prior_anchor(r,ps.id)
    @test occursin("id=\""*anchor*"\"",brm_description_markdown(r))
    @test occursin("](#"*anchor*")",brm_description_markdown(r))
    @test brm_description_prior_anchor(ps)==anchor
    predictor=only(filter(c->c.kind===:predictor,r.components))
    @test ps.id in brm_description_prior_references(predictor)
    @test brm_description_prior_anchor(predictor,ps.id)==anchor
    @test brm_description_prior_anchor(r,ps.id;prefix="model-A")!=
          brm_description_prior_anchor(r,ps.id;prefix="model-B")
    @test occursin(brm_description_prior_anchor(r,ps.id;prefix="model-A"),
        brm_description_markdown(r;prefix="model-A"))
    @test stan_code(sb)==source
    @test isequal(sb.data,data)
    @test brm_descriptor(sb).id==d.id
    @test brm_description(d).equations==r.equations
end

@testset "included scientific submodel parameters and calls" begin
    m=@brm DESCRIPTION_DATA begin
        mu ~ toy_term(x;anchor=1.7)
        y ~ Normal(mu,0.5)
    end
    d=brm_descriptor(SBBRMI(m;mod=@__MODULE__,total_groups=()))
    gap=brm_description(d)
    @test !gap.complete
    @test any(c->c.callable===curve,brm_description_components(gap))
    hook=c -> begin
        binding=brm_description_binding(c,:offset)
        @test binding.value.logical==(:parameter,:mu,:offset)
        prior=brm_description_prior(c,only(binding.prior_ids))
        @test prior.distribution.arguments==(1.7,1.0)
        bound=brm_description_binding(c,:lower_bound)
        @test bound.role===:constant
        @test bound.value===1.0
        @test isempty(bound.prior_ids)
        @test brm_description_math(c,bound.value)=="1.0"
        BRMDescriptionFragment(equations=("m=x+"*brm_description_math(c,binding.value)*"+"*brm_description_math(c,bound.value),),covers=(c.id,))
    end
    partial=brm_description(d;hooks=(toy_term=>hook,))
    @test !partial.complete
    full=brm_description(d;hooks=(toy_term=>hook,curve=>c->BRMDescriptionFragment(
        equations=("q="*brm_description_math(c,only(c.arguments))*"^2+0.6",),covers=(c.id,))))
    @test full.complete
    @test isempty(full.diagnostics)
    custom_priors=filter(p->p.id==(:parameter,:mu,:offset),full.priors)
    @test length(custom_priors)==1
end

@testset "kernel aliases, local parameters and composed scientific calls" begin
    curve=DescriptionScientificComponent.curve
    data=(; g=[1,2],t_grid=[[0.1,0.4],[0.2,0.5,0.8]],
            y_grid=[[0.2,0.4],[0.3,0.6,0.9]])
    m=@brm data begin
        eta ~ 1 + (1 | g)
        pred ~ kernel(t_grid,y_grid,eta) do ts,yy,e
            inner_sd ~ exponential(1.0)
            q = curve(curve(ts)) .+ e
            yy ~ normal(q,inner_sd)
            q
        end
    end
    d=brm_descriptor(SBBRMI(m;mod=@__MODULE__,total_groups=()))
    gap=brm_description(d)
    scientific=filter(c -> c.callable===curve,brm_description_components(gap))
    @test length(scientific)==2
    @test !gap.complete
    @test any(c -> !isempty(c.axes),scientific)
    root_only=c -> BRMDescriptionFragment(covers=(c.id,))
    @test !brm_description(d;hooks=(kernel=>root_only,)).complete
    contexts=BRMDescriptionContext[]
    hook=c -> begin
        push!(contexts,c)
        sd=brm_description_binding(c,:inner_sd)
        @test sd.role===:parameter
        @test sd.value.logical==only(sd.prior_ids)
        @test brm_description_prior(c,only(sd.prior_ids)).support.lower==0.0
        @test brm_description_binding(c,:ts).value.name===:t_grid
        @test brm_description_binding(c,:e).value.name===:eta
        @test any(o -> o.logical===:q,c.outputs)
        BRMDescriptionFragment(equations=("q="*brm_description_math(c,only(c.arguments))*"^2+0.6",),covers=(c.id,))
    end
    r=brm_description(d;hooks=(curve=>hook,))
    @test r.complete
    @test isempty(r.diagnostics)
    @test length(contexts)==2
    @test all(e->!occursin("\\mathrm{block}",e) && !occursin("\\mathrm{->}",e),r.equations)
    @test occursin("t\\_grid",brm_description_math(last(contexts),only(last(contexts).arguments)))
    unrelated=first(filter(c -> c.kind===:parameter || c.kind===:predictor,r.components))
    @test_throws ArgumentError brm_description(d;hooks=(curve=>c->BRMDescriptionFragment(covers=(unrelated.id,)),))
end

@testset "independent versus shared group covariance" begin
    make(terms)=nothing
    independent=@brm DESCRIPTION_DATA begin
        sigma ~ Exponential(2)
        mu ~ 1 + x + (1 + x || g)
        y ~ Normal(mu,sigma)
    end
    r=brm_description(SBBRMI(independent;total_groups=()))
    @test r.complete
    @test !any(e -> occursin("\\Omega_{",e),r.equations)
    @test count(c -> c.kind===:random_effect,r.components)==2
    @test !any(p -> :correlation in p.id,r.priors)
end

@testset "custom callable hooks and public notation" begin
    curve=DescriptionScientificComponent.curve
    m=@brm DESCRIPTION_DATA begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        y ~ Normal(curve(mu),sigma)
    end
    d=brm_descriptor(SBBRMI(m;mod=@__MODULE__,total_groups=()))
    gap=brm_description(d)
    @test !gap.complete
    @test any(c -> c.callable===curve,brm_description_components(gap))
    @test occursin("Incomplete model description",brm_description_markdown(gap))
    received=Ref{Any}()
    hook=c -> begin
        received[]=c
        argument=brm_description_math(c,only(c.arguments))
        BRMDescriptionFragment(prose=("The mean is a quadratic response.",),
            equations=("q="*argument*"^2+0.6",),covers=(c.id,),
            notation=((;name=:q,meaning="quadratic mean",axis=:observation),))
    end
    r=brm_description(d;hooks=(curve=>hook,),labels=Dict(:mu=>(;meaning="location",symbol="\\mu")))
    @test r.complete
    @test received[].arguments[1] isa BRMDescriptionReference
    @test received[].arguments[1].name===:mu
    @test received[].keywords isa NamedTuple
    @test received[].children isa Tuple
    @test received[].fitted_constants isa Tuple
    @test received[].outputs isa Tuple
    @test brm_description_symbol(received[],:mu)=="\\mu"
    @test any(e -> e=="q=\\mu^2+0.6",r.equations)
    @test any(n -> n.name===:q,r.notation)
    @test brm_description_symbol(received[],:a_b)=="\\mathrm{a\\_b}"
    # refused: claims must refer to this prepared semantic inventory.
    @test_throws ArgumentError brm_description(d;hooks=(curve=>c->BRMDescriptionFragment(covers=((:invented,),)),))
end
