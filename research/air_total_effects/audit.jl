include("model.jl")
using .AIRTotals, Test, Random, JSON, BridgeStan, Serialization, LogDensityProblems
using WarmupHMC
using DifferentiationInterface: AutoEnzyme
grouping,hierarchy,out=ARGS
d=AIRTotals.load_data(grouping,hierarchy);m=AIRTotals.model(d,out)
q=AIRTotals.total_initial(d,m)
ordinary=BridgeStan.StanModel(joinpath(d.dir,"ordinary_ncp.stan"),joinpath(d.dir,"ordinary_ncp.json");warn=false)
init=AIRTotals.ordinary_init(d)
nq=BridgeStan.param_unconstrain_json(ordinary,JSON.json(init))
@testset "Exact AIR target and automatic coordinate audit" begin
    for (target,start,reference,offset) in ((m.model,q,x->AIRTotals.total_reference(d,m,x),(d.K+1)*log(2.)),
            (ordinary,nq,x->AIRTotals.ordinary_reference(d,ordinary,x),0.))
        for trial in 0:3
            point=start.+(trial==0 ? zeros(length(start)) : .02randn(Xoshiro(trial),length(start)))
            lp,g=BridgeStan.log_density_gradient(target,point;propto=false)
            @test isapprox(lp+offset,reference(point);atol=1e-7,rtol=1e-12)
            fd=AIRTotals.finite_gradient(reference,point)
            @test maximum(abs.(g-fd)./max.(1.,abs.(g)))<2e-5
        end
    end
    for c in (0.,.5,1.)
        raw=AIRTotals.BrmsPupilProblem(m.model,Ref(0))
        rp=AIRTotals.BRM.adaptive_centering_problem(m.sb,raw,AutoEnzyme();unc_names=m.names,centeredness=c)
        ir=WarmupHMC.reparametrizer(rp);_,x=WarmupHMC._inverse_with_logabsdet_jacobian(ir,q)
        jac,y=ir(x);@test y≈q
        lp,g=LogDensityProblems.logdensity_and_gradient(rp,x)
        @test lp+(d.K+1)*log(2.)≈AIRTotals.total_reference(d,m,y)+jac
        @test maximum(abs.(g-AIRTotals.finite_gradient(z->LogDensityProblems.logdensity(rp,z),x))./max.(1.,abs.(g)))<3e-5
    end
end
open(io->JSON.print(io,Dict("status"=>"passed","grouping"=>grouping,"hierarchy"=>hierarchy,
    "groups"=>d.J,"observations"=>length(d.y),"ordinary_dimension"=>length(nq),
    "total_dimension"=>length(q),"constant_offset"=>(d.K+1)*log(2.)),2),joinpath(out,"audit.json"),"w")
open(io->JSON.print(io,init),joinpath(out,"ordinary-init.json"),"w")
serialize(joinpath(out,"initial.jls"),(;q,nq,names=m.names))
println("AIR_AUTOMATIC_AUDIT_COMPLETE ",grouping," ",hierarchy)
if haskey(ENV,"AIR_START_OUTPUT")
    include("run.jl")
    Base.invokelatest(run_matrix,d,m,ENV["AIR_START_OUTPUT"],ordinary)
end
