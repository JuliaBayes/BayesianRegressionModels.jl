using Test, BayesianRegressionModels, StanBlocks, BridgeStan, Distributions
using Statistics, LinearAlgebra
include(joinpath(@__DIR__,"testset_filter.jl"))
const BRM=BayesianRegressionModels

function mask_values(sb,problem,q,seed,logical)
    full=BridgeStan.param_constrain(problem.model,q;include_tp=true,include_gq=true,
        rng=BridgeStan.StanRNG(problem.model,seed))
    names=BridgeStan.param_names(problem.model;include_tp=true,include_gq=true)
    selected=Set(member for record in modeled_covariates(sb) if record.resampled for member in record.members)
    descriptor=brm_descriptor(sb)
    Dict(name=>full[brm_output_coordinates(descriptor,name,names;
        role=name in selected ? :covariate_draw : nothing)] for name in logical)
end

@stestset "fresh covariates cover fully observed and fully missing fitted declarations" begin
    for observed in (true,false),family in (Normal,LogNormal)
        x=observed ? [.3,.6,.9,1.2] : Union{Missing,Float64}[missing,missing,missing,missing]
        data=(;x,y=[.1,.2,-.1,.3])
        scalar_lhs=observed ? :x : :(mi(x))
        scalar_model=Core.eval(@__MODULE__,:(@brm begin
            mx ~ Normal(0,1)
            $scalar_lhs ~ $family(mx,.7)
            physical=exp(x)
            mu=x
            y ~ Normal(mu,1)
        end))
        fitted=SBBRMI(scalar_model(data);mod=@__MODULE__,total_groups=())
        prediction=reprocess(fitted,data;resample_covariates=[:x])
        source=StanBlocks.stan_instantiate(fitted.model)
        target=StanBlocks.stan_instantiate(prediction.model)
        from_names=BridgeStan.param_unc_names(source.model)
        to_names=BridgeStan.param_unc_names(target.model)
        q=fill(.1,length(from_names))
        moved=transport_draws(fitted,prediction,permutedims(q),from_names,to_names)
        @test to_names==["mx"]
        @test moved[1,1]==q[findfirst(==("mx"),from_names)]
        a=mask_values(prediction,target,moved[1,:],41,(:x,:physical,:mx))
        b=mask_values(prediction,target,moved[1,:],42,(:x,:physical,:mx))
        @test length(a[:x])==4 && a[:x]!=b[:x]
        @test a[:physical]≈exp.(a[:x])
        @test family===Normal || all(>(0),a[:x])
        @test BridgeStan.log_density(target.model,moved[1,:];propto=false)≈logpdf(Normal(),.1)
        @test only(modeled_covariates(prediction)).missing==(observed ? 0 : 4,)
    end
    joint_builder=@brm begin
        L ~ LKJCovarianceFactor(2;scale_prior=Exponential(1))
        mi([x,z]) ~ MvNormalCholesky([.2,.4],L)
        physical=exp(x)
        mu=x+z
        y ~ Normal(mu,1)
    end
    for observed in (true,false)
        x=observed ? [.3,.6,.9,1.2] : Union{Missing,Float64}[missing,missing,missing,missing]
        z=observed ? [.4,.5,.7,.8] : Union{Missing,Float64}[missing,missing,missing,missing]
        data=(;x,z,y=[.1,.2,-.1,.3])
        fitted=SBBRMI(joint_builder(data);mod=@__MODULE__,total_groups=())
        prediction=reprocess(fitted,data;resample_covariates=[:z])
        source=StanBlocks.stan_instantiate(fitted.model)
        target=StanBlocks.stan_instantiate(prediction.model)
        from_names=BridgeStan.param_unc_names(source.model)
        to_names=BridgeStan.param_unc_names(target.model)
        q=fill(.1,length(from_names))
        moved=transport_draws(fitted,prediction,permutedims(q),from_names,to_names)
        @test length(to_names)==3
        @test all(in(from_names),to_names)
        for (j,name) in enumerate(to_names)
            @test moved[1,j]==q[findfirst(==(name),from_names)]
        end
        a=mask_values(prediction,target,moved[1,:],41,(:x,:z,:physical,:mu))
        b=mask_values(prediction,target,moved[1,:],42,(:x,:z,:physical,:mu))
        @test length(a[:x])==4 && a[:x]!=b[:x] && a[:z]!=b[:z]
        @test a[:physical]≈exp.(a[:x])
        @test a[:mu]≈a[:x]+a[:z]
        @test only(modeled_covariates(prediction)).members==(:x,:z)
        @test only(modeled_covariates(prediction)).missing==((observed ? 0 : 4),(observed ? 0 : 4))
    end
end

@stestset "fresh covariates feed the original native kernel observations" begin
    builder=@brm begin
        mx ~ Normal(0,1)
        mi(x) ~ Normal(mx,.7)
        physical=exp(x)
        mu ~ 1+standardize(x)+physical+(1|subject)
        out ~ kernel(y,mu) do yy,mm
            yy ~ normal(mm,1.)
            mm
        end
    end
    data=(;x=Union{Missing,Float64}[.2,missing,.8,missing],
        subject=[1,1,2,2],y=[.1,.2,-.1,.3])
    fitted=SBBRMI(builder(data);mod=@__MODULE__,total_groups=())
    prediction=reprocess(fitted,data;resample_covariates=[:x],resample_groups=[:subject])
    source=StanBlocks.stan_instantiate(fitted.model)
    target=StanBlocks.stan_instantiate(prediction.model)
    from_names=BridgeStan.param_unc_names(source.model)
    to_names=BridgeStan.param_unc_names(target.model)
    q=fill(.1,length(from_names))
    moved=transport_draws(fitted,prediction,permutedims(q),from_names,to_names)
    @test all(in(from_names),to_names)
    @test only(ranef_blocks(prediction)).generated
    a=mask_values(prediction,target,moved[1,:],41,(:x,:physical,:mu,:out))
    b=mask_values(prediction,target,moved[1,:],42,(:x,:physical,:mu,:out))
    @test a[:physical]≈exp.(a[:x])
    @test a[:out]≈a[:mu]
    @test a[:x]!=b[:x] && a[:out]!=b[:out]
    @test brm_output(brm_descriptor(prediction),:y;role=:posterior_predictive).source===:y
end
