using Test, BayesianRegressionModels, Distributions
import StanBlocks

module DescriptionPreparedInputs
using BayesianRegressionModels, StanBlocks
function prepared_term end
@deffun opaque_mix(x::vector[n], mask::vector[n], observed::vector[n])::vector[n] = x .+ mask .+ observed
const captured_mix=opaque_mix
const cell=@slic begin
    offset ~ normal(anchor,1.0)
    largest=max(mask)
    n_input=dims(log_input)[1]
    return captured_mix(log_input .+ offset .+ largest .+ n_input,mask,observed)
end
function BayesianRegressionModels._sb_submodel_rhs!(stmts,data,target::Symbol,::typeof(prepared_term),rhs)
    input,observed=getargs(rhs)
    input_key=name(input); observed_key=name(observed)
    raw=collect(parent(parent(input)))
    data[input_key]=raw
    data[observed_key]=collect(parent(parent(observed)))
    mask_key=Symbol(target,:_mask); log_key=Symbol(target,:_log_input)
    data[mask_key]=Float64.(raw .> 1.0)
    data[log_key]=log.(raw)
    push!(stmts,Expr(:call,:~,target,Expr(:call,cell,Expr(:parameters,
        Expr(:kw,:log_input,log_key),Expr(:kw,:mask,mask_key),
        Expr(:kw,:observed,observed_key),Expr(:kw,:anchor,0.8)))))
    :done
end
end
const prepared_term=DescriptionPreparedInputs.prepared_term

@testset "included prepared vectors retain public identities" begin
    data=(;input=collect(range(0.2,2.2;length=40)),observed=collect(range(1.,3.;length=40)),
        y=collect(range(1.2,2.4;length=40)))
    m=@brm data begin
        mu ~ prepared_term(input,observed)
        y ~ Normal(mu,1.0)
    end
    d=brm_descriptor(SBBRMI(m;mod=@__MODULE__,total_groups=()))
    simple=c->BRMDescriptionFragment(covers=(c.id,))
    r=brm_description(d;hooks=(prepared_term=>simple,DescriptionPreparedInputs.opaque_mix=>simple))
    @test r.complete
    # The prepared declaration owns a copied SlicModel payload; its authored
    # namespace and contained scientific callable identities stay intact.
    cell=only(filter(c->c.kind===:submodel,brm_description_components(r)))
    @test cell.callable.mod===DescriptionPreparedInputs
    mask=brm_description_binding(cell,:mask).value
    log_input=brm_description_binding(cell,:log_input).value
    observed=brm_description_binding(cell,:observed).value
    @test mask isa BRMDescriptionReference && first(mask.logical)===:prepared_data
    @test log_input isa BRMDescriptionReference && first(log_input.logical)===:prepared_data
    @test observed isa BRMDescriptionReference && observed.name===:observed
    @test brm_description_binding(cell,:anchor).value==0.8
    opaque=only(filter(c->c.callable===DescriptionPreparedInputs.opaque_mix,brm_description_components(r)))
    @test brm_description_math(opaque,opaque.arguments[2])==brm_description_math(cell,mask)
    @test brm_description_math(opaque,opaque.arguments[3])=="\\mathrm{observed}"
    @test length(brm_description_math(opaque,opaque.arguments[1]))<220
    @test !occursin("\\left[",brm_description_math(opaque,opaque.arguments[2]))
    @test brm_description_math(cell,(0.5,1.5))=="\\left[0.5, 1.5\\right]"
    @test any(c->c.callable===StanBlocks.stan.builtin.max,brm_description_components(r))
    @test any(e->startswith(e,"\\mathrm{largest}=") && occursin("\\operatorname{max}",e),r.equations)
    @test any(c->c.callable===StanBlocks.stan.builtin.dims,brm_description_components(r))
    @test any(e->startswith(e,"\\mathrm{n\\_input}=") && occursin("\\operatorname{shape}",e) && endswith(e,"}_{1}"),r.equations)
    # Exercise a captured global value in the same prepared included context.
    globalref=GlobalRef(DescriptionPreparedInputs,:opaque_mix)
    @test BayesianRegressionModels._brmd_snapshot(globalref)===globalref
    @test BayesianRegressionModels._brmd_value(globalref,
        (;mod=DescriptionPreparedInputs),(:public_global,))===DescriptionPreparedInputs.opaque_mix
end
