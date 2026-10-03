using Test, BayesianRegressionModels, Distributions
import StanBlocks

module DescriptionNestedBinding
using BayesianRegressionModels, StanBlocks
function nested_term end
const leaf=@slic begin
    offset ~ normal(anchor,1.0)
    "The padding"
    lower_bound=2.0
    "Number of terms"
    n_terms=8
    plain=0.5
    return offset+input+lower_bound+n_terms+count+plain
end
const outer=@slic begin
    inner ~ leaf(;input=input,anchor=anchor,count=6)
    return inner
end
function BayesianRegressionModels._sb_submodel_rhs!(stmts,data,target::Symbol,::typeof(nested_term),rhs)
    input=only(getargs(rhs)); input_key=name(input)
    data[input_key]=collect(parent(parent(input)))
    anchor_key=Symbol(target,:_anchor); data[anchor_key]=0.7
    push!(stmts,Expr(:call,:~,target,Expr(:call,outer,Expr(:parameters,
        Expr(:kw,:input,input_key),Expr(:kw,:anchor,anchor_key)))))
    :done
end
end
const nested_term=DescriptionNestedBinding.nested_term

module DescriptionKeywordDefault
using BayesianRegressionModels, StanBlocks
function default_term end
const cell=@slic begin
    "Number of functions"
    n_functions=8
    offsets::vector[n_functions] ~ std_normal()
    dimension=n_functions+1
    return offsets[1]+input+dimension
end
function BayesianRegressionModels._sb_submodel_rhs!(stmts,data,target::Symbol,::typeof(default_term),rhs)
    input=only(getargs(rhs)); key=name(input)
    data[key]=collect(parent(parent(input)))
    count=get(getkwargs(rhs),:n_functions,8)
    push!(stmts,Expr(:call,:~,target,Expr(:call,cell,Expr(:parameters,
        Expr(:kw,:input,key),Expr(:kw,:n_functions,count)))))
    :done
end
end
const default_term=DescriptionKeywordDefault.default_term

@testset "supplied included keyword overrides documented fixed default" begin
    data=(;input=[.2,.5,1.2],y=[.3,.6,1.1])
    model=@brm data begin
        mu ~ default_term(input;n_functions=3)
        y ~ Normal(mu,1.0)
    end
    sb=SBBRMI(model;mod=@__MODULE__,total_groups=())
    # Inventory/render without a scientific hook must return precise gaps,
    # rather than throwing on a duplicate fixed/default name.
    gap=brm_description(sb)
    @test !gap.complete
    @test occursin("Incomplete",brm_description_markdown(gap))
    hook=c->begin
        @test brm_description_binding(c,:n_functions).value==3
        BRMDescriptionFragment(covers=(c.id,))
    end
    r=brm_description(sb;hooks=(default_term=>hook,))
    @test r.complete
    cell=only(filter(c->c.kind===:submodel,brm_description_components(r)))
    @test count(b->b.name===:n_functions,cell.bindings)==1
    @test brm_description_math(cell,BRMDescriptionReference(:n_functions,:local))=="3"
    prior=brm_description_prior(cell,only(brm_description_binding(cell,:offsets).prior_ids))
    @test prior.source.dimension==(3,)
    @test any(e->startswith(e,"\\mathrm{dimension}=") && occursin("3 + 1",e),r.equations)
    @test !any(e->startswith(e,"\\mathrm{n\\_functions}="),r.equations)
    @test StanBlocks.stanc_check(stan_code(sb);warn_pedantic=false).ok
end

@testset "nested authored bindings and identical formal aliases" begin
    data=(;input=[0.2,0.5,1.2],y=[0.3,0.6,1.1])
    m=@brm data begin
        mu ~ nested_term(input)
        y ~ Normal(mu,1.0)
    end
    r=brm_description(SBBRMI(m;mod=@__MODULE__,total_groups=());
        hooks=(nested_term=>c->BRMDescriptionFragment(covers=(c.id,)),))
    @test r.complete
    root=only(filter(c->c.callable===nested_term,brm_description_components(r)))
    binding=brm_description_binding(root,(:inner,:offset))
    @test binding.value.logical==(:parameter,:mu,:inner,:offset)
    @test brm_description_prior(root,only(binding.prior_ids)).distribution.arguments==(0.7,1.0)
    leaf=only(filter(c->c.kind===:submodel && c.callable===DescriptionNestedBinding.leaf,
        brm_description_components(r)))
    @test brm_description_binding(leaf,:input).value.name===:input
    @test brm_description_binding(leaf,:lower_bound).value==2.0
    @test brm_description_binding(leaf,:n_terms).value==8
    @test brm_description_binding(leaf,:count).value==6
    @test brm_description_binding(leaf,:plain).value==0.5
    @test !any(c->c.callable===:macrocall,brm_description_components(r))
    @test brm_description_math(leaf,brm_description_binding(leaf,:input).value)=="\\mathrm{input}"
    @test binding.prior_ids[1] in brm_description_prior_references(root)
end

@testset "compact distinct covariance block symbols" begin
    data=(;x=[-1.,0.,1.,2.,3.,4.],g=[1,1,2,2,3,3],y=[.2,.3,.5,.6,.9,1.1])
    m=@brm data begin
        mu ~ 1+(1|first_block|g)+(1+x|second_block|g)
        y ~ Normal(mu,1.0)
    end
    r=brm_description(SBBRMI(m;mod=@__MODULE__,total_groups=()))
    @test r.complete
    blocks=filter(c->c.kind===:random_effect,r.components)
    notes=filter(n->n.axis===:covariance_block,r.notation)
    @test Set(n.name for n in notes)==Set(c.id for c in blocks)
    @test Set(n.symbol for n in notes)==Set(("1","2"))
    @test all(n->occursin("Covariance block "*n.symbol,n.meaning) && occursin("margins",n.meaning),notes)
    @test all(c->any(e->startswith(e,"\\mathbf b_{"*brm_description_symbol(c,c.id)*",i}"),r.equations),blocks)
    @test all(e->!occursin("random\\_effect",e) && !occursin("first\\_block",e) && !occursin("second\\_block",e),r.equations)
    @test any(e->occursin("\\mathbf b_{1,g_j}",e) && occursin("\\mathbf b_{2,g_j}",e),r.equations)
    md=brm_description_markdown(r)
    rows=filter(row->startswith(row,"| <a id="),split(md,'\n'))
    @test length(rows)==length(r.priors)
    @test all(row->!occursin("\\sim",split(row," | ")[2]),rows)
    @test all(p->all(part->occursin("`"*string(part)*"`",md),p.id),r.priors)
end

@testset "stratified and centered group covariance" begin
    data=(;x=[-1.,0.,1.,2.,3.,4.],g=[1,1,2,2,3,3],
        stratum=[1,1,1,1,2,2],y=[0.2,0.3,0.5,0.6,0.9,1.1])
    m=@brm data begin
        mu ~ 1 + (1+x | shared | gr(g;by=stratum))
        y ~ Normal(mu,1.0)
    end
    r=brm_description(SBBRMI(m;mod=@__MODULE__,total_groups=()))
    @test r.complete
    @test any(e->occursin(",s(i)}",e),r.equations)
    @test count(p->p.distribution.callable===StanBlocks.stan.builtin.lkj_corr_cholesky,r.priors)==1
    ordinary=@brm data begin
        mu ~ 1+(1+x | shared | g)
        y ~ Normal(mu,1.0)
    end
    centered=brm_description(SBBRMI(ordinary;mod=@__MODULE__,centered_groups=(:g,),total_groups=()))
    @test centered.complete
    @test any(p->last(p.id)===:deviations && p.source.kind===:conditional,centered.priors)
    @test !any(p->last(p.id)===:standardized_deviations,centered.priors)
end
