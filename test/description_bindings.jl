using Test, BayesianRegressionModels, Distributions
import StanBlocks

module DescriptionNestedBinding
using BayesianRegressionModels, StanBlocks
function nested_term end
const leaf=@slic begin
    offset ~ normal(anchor,1.0)
    lower_bound=2.0
    return offset+input+lower_bound
end
const outer=@slic begin
    inner ~ leaf(;input=input,anchor=anchor)
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
    @test brm_description_math(leaf,brm_description_binding(leaf,:input).value)=="\\mathrm{input}"
    @test binding.prior_ids[1] in brm_description_prior_references(root)
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
