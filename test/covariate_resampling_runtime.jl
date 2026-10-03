using Test, BayesianRegressionModels, StanBlocks, BridgeStan
using Distributions, Statistics, LinearAlgebra, Random
include(joinpath(@__DIR__,"testset_filter.jl"))
include(joinpath(@__DIR__,"covariate_resampling_fixtures.jl"))
const BRM=BayesianRegressionModels

function fresh_outputs(sb,problem,q,seed,logical)
    names=BridgeStan.param_names(problem.model;include_tp=true,include_gq=true)
    full=BridgeStan.param_constrain(problem.model,q;include_tp=true,include_gq=true,
        rng=BridgeStan.StanRNG(problem.model,seed))
    descriptor=brm_descriptor(sb)
    selected=Set(member for record in modeled_covariates(sb) if record.resampled for member in record.members)
    Dict(name=>full[brm_output_coordinates(descriptor,name,names;
        role=name in selected ? :covariate_draw : nothing)] for name in logical)
end

@stestset "compiled covariate population draws preserve posterior carriers and conditional laws" begin
    for (label,builder) in ((:scalar,FRESH_SCALAR_BUILDER),(:joint,FRESH_JOINT_BUILDER))
        original=deepcopy(FRESH_SCALAR_DATA)
        fitted=SBBRMI(builder(FRESH_SCALAR_DATA);mod=@__MODULE__,total_groups=())
        selected=label===:scalar ? [:x,:z] : [:z]
        future=(;x=fill(missing,6),z=fill(missing,6),u=collect(range(-.7,.9;length=6)),
            subject=[101,101,202,202,303,303],y=zeros(6))
        prediction=reprocess(fitted,future;resample_covariates=selected,resample_groups=[:subject])
        source=StanBlocks.stan_instantiate(fitted.model)
        target=StanBlocks.stan_instantiate(prediction.model)
        from_names=BridgeStan.param_unc_names(source.model)
        to_names=BridgeStan.param_unc_names(target.model)
        println("COMPILED_FRESH=",label," source=",from_names," target=",to_names)
        q=[.03*(i-3) for i in eachindex(from_names)]
        if label===:joint
            correlation=only(findall(name->occursin("L_L_corr.",name),from_names))
            q[correlation]=.5
        end
        draws=permutedims(q)
        untouched=copy(draws)
        moved=transport_draws(fitted,prediction,draws,from_names,to_names)
        @test draws==untouched
        @test all(in(from_names),to_names)
        for (j,name) in enumerate(to_names)
            @test moved[1,j]==q[findfirst(==(name),from_names)]
        end
        @test length(to_names)<length(from_names)
        @test all(block->block.generated,ranef_blocks(prediction))
        logical=label===:scalar ? (:x,:z,:physical,:log_physical,:mu,:mx,:sx,:zloc,:zscale) :
            (:x,:z,:physical,:log_physical,:mu,:xloc,:zloc,:L)
        source_values=fresh_outputs(fitted,source,q,41,logical)
        observed_x=findall(!ismissing,FRESH_SCALAR_DATA.x)
        observed_z=findall(!ismissing,FRESH_SCALAR_DATA.z)
        @test source_values[:x][observed_x]==FRESH_SCALAR_DATA.x[observed_x]
        @test source_values[:z][observed_z]==FRESH_SCALAR_DATA.z[observed_z]
        @test source_values==fresh_outputs(fitted,source,q,42,logical)
        samples=[fresh_outputs(prediction,target,moved[1,:],seed,logical) for seed in 1:200]
        first_draw=first(samples)
        @test length(first_draw[:x])==6 && length(first_draw[:z])==6
        @test all(sample->sample[:physical]≈exp.(sample[:x]),samples)
        @test all(sample->sample[:log_physical]≈sample[:x],samples)
        @test first_draw[:x]!=samples[2][:x] && first_draw[:z]!=samples[2][:z]
        @test first_draw[:mu]!=samples[2][:mu]
        if label===:scalar
            mx,sx=only(first_draw[:mx]),only(first_draw[:sx])
            xs=reduce(vcat,[sample[:x] for sample in samples])
            @test mean(xs)≈mx atol=.1*sx
            @test std(xs)≈sx rtol=.1
            @test all(sample->sample[:zloc]≈.2 .+.3 .*sample[:x],samples)
            @test all(sample->sample[:zscale]≈exp.(.1 .+.2 .*sample[:x]),samples)
            residual=reduce(vcat,[(log.(sample[:z]).-(.2 .+.3 .*sample[:x]))./sample[:zscale] for sample in samples])
            @test abs(mean(residual))<.1
            @test std(residual)≈1 rtol=.1
            covariate_lp=sum(logpdf.(Normal(mx,sx),source_values[:x]))+
                sum(logpdf.(LogNormal.(source_values[:zloc],source_values[:zscale]),source_values[:z]))
            missing_jacobian=sum(log,source_values[:z][findall(ismissing,FRESH_SCALAR_DATA.z)])
        else
            L=reshape(first_draw[:L],2,2)
            residual=reduce(vcat,[hcat(sample[:x]-sample[:xloc],
                sample[:z]-sample[:zloc]) for sample in samples])
            @test vec(mean(residual;dims=1))≈zeros(2) atol=.1*maximum(abs,L)
            @test cov(residual)≈L*L' atol=.15*maximum(abs,L*L')
            source_L=reshape(source_values[:L],2,2)
            covariate_lp=sum(logpdf(MvNormal([source_values[:xloc][i],source_values[:zloc][i]],
                Symmetric(source_L*source_L')),[source_values[:x][i],source_values[:z][i]])
                for i in eachindex(FRESH_SCALAR_DATA.y))
            missing_jacobian=0.
        end
        endpoint_lp=sum(logpdf.(Normal.(source_values[:mu],1),FRESH_SCALAR_DATA.y))
        source_lp=BridgeStan.log_density(source.model,q;propto=false,jacobian=true)
        target_lp=BridgeStan.log_density(target.model,moved[1,:];propto=false,jacobian=true)
        removed_effect_lp=sum(logpdf.(Normal(),q[vec(ranef_coordinates(only(ranef_blocks(fitted)),from_names))]))
        # Only retained prior density remains: both covariate and endpoint
        # observations are predictive, and fitted missing cells leave the frame.
        @test target_lp≈source_lp-covariate_lp-endpoint_lp-missing_jacobian-removed_effect_lp atol=1e-9
        @test isequal(original,FRESH_SCALAR_DATA)
        @test BRM.stan_code(reprocess(prediction,future))==BRM.stan_code(prediction)
    end
end
