using Test,BayesianRegressionModels,StanBlocks,Distributions,Statistics
include(joinpath(@__DIR__,"testset_filter.jl"))
const BRM=BayesianRegressionModels

include(joinpath(@__DIR__,"covariate_resampling_fixtures.jl"))

@stestset "modeled covariate logical inventory and selection" begin
    scalar=SBBRMI(FRESH_SCALAR_BUILDER(FRESH_SCALAR_DATA);mod=@__MODULE__,total_groups=())
    inventory=modeled_covariates(scalar)
    @test map(r->r.members,inventory)==((:x,),(:z,))
    @test map(r->r.family,inventory)==(Normal,LogNormal)
    @test all(r->r.rows==5 && r.supported && !r.resampled,inventory)
    @test inventory[2].dependencies==(:zloc,:zscale)
    @test_throws "modeled covariate member" reprocess(scalar,FRESH_SCALAR_DATA;resample_covariates=[:y])
    @test_throws "include dependent modeled declaration" reprocess(scalar,FRESH_SCALAR_DATA;resample_covariates=[:x])
    joint=SBBRMI(FRESH_JOINT_BUILDER(FRESH_SCALAR_DATA);mod=@__MODULE__,total_groups=())
    @test only(modeled_covariates(joint)).members==(:x,:z)
end

@stestset "fresh scalar and joint covariate source preserves training anchors" begin
    for builder in (FRESH_SCALAR_BUILDER,FRESH_JOINT_BUILDER)
        before=deepcopy(FRESH_SCALAR_DATA)
        source=SBBRMI(builder(FRESH_SCALAR_DATA);mod=@__MODULE__,total_groups=())
        selected=builder===FRESH_SCALAR_BUILDER ? [:x,:z] : [:x]
        target=reprocess(source,FRESH_SCALAR_DATA;resample_covariates=selected)
        code=BRM.stan_code(target)
        println("FRESH_SOURCE=",builder===FRESH_SCALAR_BUILDER ? "scalar" : "joint","\n",code)
        @test StanBlocks.stanc_check(code;warn_pedantic=false).ok
        @test all(r->r.resampled,modeled_covariates(target))
        descriptor=brm_descriptor(target)
        for record in modeled_covariates(target),member in record.members
            @test brm_output(descriptor,member;role=record.value_role).kind===:generated_quantity
        end
        @test !occursin("_y_mis;",split(code,"model {")[1])
        @test occursin("dummy_lpdf",code)
        @test any(o->o.kind===:parameter,StanBlocks.stan_descriptor(target.model).outputs)
        for (key,entry) in source.preproc
            startswith(String(entry.kind),"missing_") && entry.kind!==:missing_response || continue
            @test target.data[key]==source.data[key]
        end
        @test isequal(before,FRESH_SCALAR_DATA)
        replay=reprocess(target,FRESH_SCALAR_DATA)
        @test BRM.stan_code(replay)==code
        cv=reprocess(source,FRESH_SCALAR_DATA;resample_covariates=selected,resample_groups=[:subject])
        @test StanBlocks.stanc_check(BRM.stan_code(cv);warn_pedantic=false).ok
        @test BRM.stan_code(reprocess(cv,FRESH_SCALAR_DATA))==BRM.stan_code(cv)
        @test only(ranef_blocks(reprocess(cv,FRESH_SCALAR_DATA))).generated
        plan_target=reprocess(generative_plan(source),FRESH_SCALAR_DATA;
            resample_covariates=selected,resample_groups=[:subject])
        @test BRM.stan_code(plan_target)==BRM.stan_code(cv)
        @test modeled_covariates(plan_target)==modeled_covariates(cv)
        @test restan_data(source,FRESH_SCALAR_DATA;resample_covariates=selected)==BRM.stan_data(target)
        @test_throws "frozen observed training anchors" reprocess(source,FRESH_SCALAR_DATA;
            resample_covariates=selected,freeze_constants=false)
    end
end

@stestset "complete observed modeled covariate becomes a runtime draw" begin
    data=(;x=[.2,.5,.8,1.1],y=[.1,.3,-.2,.4])
    source=SBBRMI(@brm(data,begin
        mx ~ Normal(0,1)
        sx ~ LogNormal(0,.3)
        x ~ Normal(mx,sx)
        mu ~ 1+standardize(x)
        y ~ Normal(mu,1)
    end);mod=@__MODULE__,total_groups=())
    target=reprocess(source,merge(data,(;x=[10.,20.,30.,40.]));resample_covariates=[:x])
    @test target.data[:standardize_x_mean] ≈ mean(data.x)
    @test target.data[:standardize_x_scale] ≈ std(data.x)
    @test StanBlocks.stanc_check(BRM.stan_code(target);warn_pedantic=false).ok
    @test !haskey(target.data,:standardize_x)
    @test only(modeled_covariates(source)).members==(:x,)
    @test_throws "expects a Symbol" reprocess(source,data;resample_covariates="x")
end

@stestset "selected source masks and unchanged fitted completion contracts" begin
    scalar_builder=@brm begin
        mx ~ Normal(0,1)
        mi(x) ~ Normal(mx,1)
        physical=exp(x)
        mu=physical
        y ~ Normal(mu,1)
    end
    for new_x in (Union{Missing,Float64}[.2,missing,.8,missing,1.1],
                  Union{Missing,Float64}[.2,.5,.8,.9,1.1],
                  Union{Missing,Float64}[missing,missing,missing,missing,missing])
        sb=SBBRMI(scalar_builder(FRESH_SCALAR_DATA);mod=@__MODULE__,total_groups=())
        future=merge(FRESH_SCALAR_DATA,(;x=new_x))
        target=reprocess(sb,future;resample_covariates=[:x])
        @test StanBlocks.stanc_check(BRM.stan_code(target);warn_pedantic=false).ok
        @test only(modeled_covariates(target)).missing==(count(ismissing,new_x),)
        @test brm_output(brm_descriptor(target),:x;role=:covariate_draw).kind===:generated_quantity
    end
    observed=(;x=Union{Missing,Float64}[.2,.5,.8,.9,1.1],y=FRESH_SCALAR_DATA.y)
    @test_throws "found no missing values" SBBRMI(scalar_builder(observed);mod=@__MODULE__,total_groups=())
    before=deepcopy(FRESH_SCALAR_DATA)
    fitted=SBBRMI(FRESH_SCALAR_BUILDER(FRESH_SCALAR_DATA);mod=@__MODULE__,total_groups=())
    @test BRM.stan_code(reprocess(fitted,FRESH_SCALAR_DATA))==BRM.stan_code(fitted)
    @test all(r->!r.resampled,modeled_covariates(reprocess(fitted,FRESH_SCALAR_DATA)))
    @test isequal(before,FRESH_SCALAR_DATA)
    centered=SBBRMI(FRESH_SCALAR_BUILDER(FRESH_SCALAR_DATA);mod=@__MODULE__,
        total_groups=(),centered_groups=[:subject])
    @test_throws "non-centered, non-CV geometry" reprocess(centered,FRESH_SCALAR_DATA;
        resample_covariates=[:x,:z])
end
