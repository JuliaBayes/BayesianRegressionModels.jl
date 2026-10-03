using Test, BayesianRegressionModels, Distributions
import StanBlocks
module DescriptionArrayConversions
using BayesianRegressionModels, StanBlocks
function matrix_term end
@deffun opaque_matrix(x::matrix[n,2])::vector[n] = x[:,1]
const cell=@slic begin
    count=3
    indices=range(1,count)
    colon_indices=1:count
    flat=to_vector(input)
    repeated=rep_matrix(flat,2)
    transposed=repeated'
    vectorized=to_vector(transposed)
    matrixized=to_matrix(vectorized,count,2)
    scalar_matrix=rep_matrix(0.5,count,2)
    return opaque_matrix(matrixized+scalar_matrix)
end
function BayesianRegressionModels._sb_submodel_rhs!(stmts,data,target::Symbol,::typeof(matrix_term),rhs)
    input=only(getargs(rhs)); key=name(input)
    data[key]=collect(parent(parent(input)))
    push!(stmts,Expr(:call,:~,target,Expr(:call,cell,Expr(:parameters,Expr(:kw,:input,key)))))
    :done
end
end
const matrix_term=DescriptionArrayConversions.matrix_term
@testset "included array conversion and range semantics" begin
    data=(;input=[.2,.3,.4],y=[.1,.2,.3])
    model=@brm data begin
        mu ~ matrix_term(input)
        y ~ Normal(mu,1)
    end
    sb=SBBRMI(model;mod=@__MODULE__,total_groups=())
    before=stan_code(sb); frozen=deepcopy(sb.data)
    opaque=DescriptionArrayConversions.opaque_matrix
    root=c->BRMDescriptionFragment(covers=(c.id,))
    gap=brm_description(sb;hooks=(matrix_term=>root,))
    @test !gap.complete
    @test all(c->c.callable===opaque,filter(c->c.id in Tuple(g.id for g in gap.coverage if g.status===:unsupported),brm_description_components(gap)))
    r=brm_description(sb;hooks=(matrix_term=>root,opaque=>root))
    @test r.complete
    nodes=brm_description_components(r)
    for f in (StanBlocks.stan.builtin.to_vector,StanBlocks.stan.builtin.to_matrix,
        StanBlocks.stan.builtin.rep_matrix,Base.range,Base.adjoint)
        @test any(c->c.callable===f,nodes)
    end
    @test any(e->occursin("\\operatorname{vec}_{\\mathrm{col}}",e),r.equations)
    @test any(e->occursin("\\operatorname{reshape}_{3\\times 2",e),r.equations)
    @test any(e->occursin("^{\\mathsf T}",e),r.equations)
    @test any(e->occursin("0.5\\,\\mathbf1_{3\\times 2}",e),r.equations)
    @test stan_code(sb)==before && isequal(sb.data,frozen)
    context=only(filter(c->c.kind===:observation,r.components))
    record=(;columns=(BRMDescriptionReference(:left,:observation),BRMDescriptionReference(:right,:observation)),impute=true)
    @test brm_description_math(context,record)=="\\left[\\mathrm{left}, \\mathrm{right}\\right]"
    @test !occursin("BRMDescription",brm_description_math(context,(;values_=record.columns)))
end

@testset "joint covariance and missing completion conditioning" begin
    data=(;u=[-.2,.4,.8,1.1],x=Union{Missing,Float64}[.2,missing,.5,missing],
        z=Union{Missing,Float64}[1.,.7,missing,missing],y=[.1,.3,.4,.6])
    builder=@brm begin
        L ~ LKJCovarianceFactor(2;scale_prior=Exponential(.7),shape=2)
        x_loc ~ 1+u
        z_loc ~ 1+u
        mi([x,z]) ~ MvNormalCholesky([x_loc,z_loc],L)
        mu ~ 1+x+z
        y ~ Normal(mu,1)
    end
    complete=@brm begin
        L ~ LKJCovarianceFactor(2;scale_prior=Exponential(.7),shape=2)
        x_loc ~ 1+u
        z_loc ~ 1+u
        [x,z] ~ MvNormalCholesky([x_loc,z_loc],L)
        mu ~ 1+x+z
        y ~ Normal(mu,1)
    end
    for (model,role,fixed,missing_) in (
        (builder(data),:partially_observed,4,4),
        (complete(merge(data,(;x=[.2,.4,.5,.6],z=[1.,.7,.9,.8]))),:conditioned,8,0))
        sb=SBBRMI(model;mod=@__MODULE__,total_groups=())
        before=stan_code(sb)
        r=brm_description(sb)
        @test r.complete
        joint=only(filter(c->c.kind===:observation && c.provenance.observation_sources==(:x,:z),r.components))
        @test joint.provenance.observation_role===role
        @test joint.provenance.observed_entries==fixed && joint.provenance.missing_entries==missing_
        factor=only(filter(c->c.callable===LKJCovarianceFactor,brm_description_components(r)))
        refs=brm_description_prior_references(factor)
        @test Set(refs)==Set(((:parameter,:L_scales),(:parameter,:L_L_corr)))
        scales=brm_description_prior(factor,(:parameter,:L_scales))
        correlation=brm_description_prior(factor,(:parameter,:L_L_corr))
        @test only(scales.distribution.arguments)≈inv(.7)
        @test scales.support.lower==0 && correlation.distribution.arguments==(2.0,)
        @test any(e->occursin("\\operatorname{diag}",e) && occursin("\\mathrm{L}",e),r.equations)
        @test all(e->!occursin("BRMDescriptionReference",e) && !occursin("BRMDescriptionComponent",e),r.equations)
        @test all(p->!occursin("brm_joint_x__z` is unconditioned",p),r.prose)
        @test any(p->occursin("partially observed",p),r.prose)==(missing_>0)
        @test any(p->p.distribution.callable===StanBlocks.stan.builtin.dummy,r.priors)==(missing_>0)
        @test occursin("no additional density",brm_description_markdown(r))==(missing_>0)
        @test stan_code(sb)==before
    end
end
