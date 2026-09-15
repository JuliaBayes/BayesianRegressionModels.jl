include("common.jl")
out=only(ARGS);mkpath(out);tm=total_model(out);q0=total_initial(tm)
@testset "Post-3 built-in totals versus independent analytic target" begin
    for trial in 0:6
        q=q0.+(trial==0 ? zeros(45) : .01max.(1.,abs.(q0)).*randn(Xoshiro(380+trial),45))
        lp,g=BridgeStan.log_density_gradient(tm.model,q;propto=false)
        expected,gradient=total_reference(tm,q)
        @test lp+2log(2.)≈expected atol=1e-7 rtol=1e-12
        @test g[tm.to_manual]≈gradient atol=1e-7 rtol=1e-8
    end
    for c in (0.,.5,1.)
        raw=BrmsPupilProblem(tm.model,Ref(0))
        rp=adaptive_centering_problem(tm.sb,raw,AutoEnzyme();unc_names=tm.names,centeredness=c)
        ir=WarmupHMC.reparametrizer(rp);_,x=WarmupHMC._inverse_with_logabsdet_jacobian(ir,q0)
        jac,q=ir(x);@test q≈q0
        lp,g=LogDensityProblems.logdensity_and_gradient(rp,x)
        @test lp+2log(2.)≈first(total_reference(tm,q))+jac atol=1e-7
        # Analytic chain rule from the independently differentiated manual
        # target avoids finite-difference cancellation at this OLS start.
        expected=zeros(45);expected[tm.to_manual]=last(total_reference(tm,q))
        transformed=copy(expected)
        for margin in 1:2, j in 1:J
            i=tm.coords.totals[j,margin];s=tm.coords.scales[margin]
            location=margin==1 ? 5651.9 : 0.
            transformed[i]=expected[i]*exp((1-c)*q[s])
            transformed[s]+=(1-c)*(expected[i]*(q[i]-location)+1)
        end
        @test g≈transformed atol=1e-7 rtol=1e-8
    end
    rec=recover_population_draws(tm.sb,repeat(q0',5000,1),tm.names;rng=Xoshiro(42))[:mu]
    cond=BRM._total_conditional(tm.block,tm.coords,q0)
    @test maximum(abs.(vec(mean(rec.population;dims=1))-cond.mean)./sqrt.(diag(inv(cond.factor))))<.05
    @test maximum(abs,rec.totals.-rec.deviations.-reshape(rec.population*tm.block.A',5000,1,2))<1e-9
end
open(joinpath(out,"audit.json"),"w") do io
    JSON.print(io,Dict("status"=>"passed","model"=>"post3","dimension"=>45,"brm_constant_offset"=>2log(2.)))
end
println("PUPIL3_BUILTIN_AUDIT_COMPLETE");flush(stdout)
if haskey(ENV,"PUPIL3_START_TOTAL_OUTPUT")
    include("total_whmc.jl")
    Base.invokelatest(main,ENV["PUPIL3_START_TOTAL_OUTPUT"],out)
end
