include("common.jl")
using Test
out=only(ARGS);mkpath(out)
tm=total_model(out)
q0=total_initial(tm)
write_json(joinpath(out,"ordinary-init.json"),ordinary_init())
ordinary=stan_model("ordinary_ncp")
nq=BridgeStan.param_unconstrain_json(ordinary,JSON.json(ordinary_init()))
@test length(nq)==66
@test ordinary_physical(ordinary,nq).totals ≈ ols_initial().totals
@testset "Post-4 exact target and gradients" begin
    for (model,q,ref,label) in ((tm.model,q0,x->total_reference(tm,x),"totals"),
                              (ordinary,nq,x->ordinary_reference(ordinary,x),"ordinary_ncp"))
        for trial in 0:3
            point=q .+ (trial==0 ? zeros(length(q)) :
                0.01max.(1.,abs.(q)).*randn(Xoshiro(20+trial),length(q)))
            lp,g=BridgeStan.log_density_gradient(model,point;propto=false)
            # BRM's lower-bounded vector SD prior omits three constant
            # half-Student-t normalizers. The posterior and gradients agree.
            offset=label=="totals" ? 3log(2.) : 0.
            @test lp+offset ≈ ref(point) atol=2e-8 rtol=1e-12
            fd=finite_gradient(ref,point)
            @test maximum(abs.(g-fd)./max.(1.,abs.(g))) < 2e-5
        end
        println("DENSITY_GRADIENT_CHECKED ",label);flush(stdout)
    end
    for c in (0.,0.5,1.)
        counted=BrmsPupilProblem(tm.model,Ref(0))
        rp=adaptive_centering_problem(tm.sb,counted,AutoEnzyme();unc_names=tm.names,centeredness=c)
        ir=WarmupHMC.reparametrizer(rp)
        _,x=WarmupHMC._inverse_with_logabsdet_jacobian(ir,q0)
        jac,y=ir(x);@test y≈q0
        lp,g=LogDensityProblems.logdensity_and_gradient(rp,x)
        @test lp+3log(2.) ≈ total_reference(tm,y)+jac atol=2e-8
        fd=finite_gradient(z->LogDensityProblems.logdensity(rp,z),x)
        @test maximum(abs.(g-fd)./max.(1.,abs.(g))) < 3e-5
    end
    recovered=recover_population_draws(tm.sb,repeat(q0',5000,1),tm.names;rng=Xoshiro(42))
    for b in total_effect_blocks(tm.sb)
        c=tm.coords[b.predictor];cond=BRM._total_conditional(b,c,q0)
        v=recovered[b.predictor]
        @test maximum(abs.(vec(mean(v.population;dims=1))-cond.mean)./sqrt.(diag(inv(cond.factor)))) < 0.05
        @test maximum(abs,v.totals.-v.deviations.-reshape(v.population*b.A',5000,1,size(b.A,1))) < 1e-10
    end
end
write_json(joinpath(out,"audit.json"),Dict("status"=>"passed","ordinary_dimension"=>length(nq),
    "total_dimension"=>length(q0),"total_blocks"=>[string(b.predictor) for b in total_effect_blocks(tm.sb)],
    "group_levels"=>J,"observations"=>length(Y),
    "brm_density_constant_to_brms"=>3log(2.)))
serialize(joinpath(out,"initial.jls"),(;q0,nq,names=tm.names))
println("PUPIL4_AUDIT_COMPLETE");flush(stdout)
if haskey(ENV,"PUPIL4_START_TOTAL_OUTPUT")
    include("total_whmc.jl")
    Base.invokelatest(main,ENV["PUPIL4_START_TOTAL_OUTPUT"],out)
end
