include("run.jl")
record=deserialize(joinpath(OUTPUT,REQUEST,"transport-failure.jls"))
out=joinpath(OUTPUT,REQUEST,"rejection-audit");mkpath(out)
stan=make_stan(out)
adaptive=adaptive_centering_problem(stan.sb,stan.density,ENZYME_BACKEND)
old_sources=[idx=>r.source for (idx,r) in record.old_ir.pairs]
WarmupHMC.restore_reparam_sources!(adaptive,old_sources)
point=WarmupHMC.DynamicHMC.evaluate_ℓ(adaptive,record.old_position[:,end])
WarmupHMC.restore_reparam_sources!(adaptive,[idx=>r.source for (idx,r) in record.new_ir.pairs])
positions=copy(record.old_position);gradients=copy(record.old_gradient)
@testset "captured HSGP invalid transport is rejected atomically" begin
    result=WarmupHMC._validated_transport!(
        adaptive,record.old_ir,record.old_position,record.old_gradient,positions,gradients,point)
    @test result === point
    @test positions == record.old_position
    @test gradients == record.old_gradient
    @test WarmupHMC.reparam_sources(adaptive) == old_sources
    value,g=LogDensityProblems.logdensity_and_gradient(adaptive,result.q)
    @test value == result.ℓq
    @test g == result.∇ℓq
    scale=Diagonal(ones(length(g)))
    @test isfinite(WarmupHMC.update_loss!(scale,positions,gradients))
    @test all(isfinite,scale.diag) && all(>(0),scale.diag)
end
println("CAPTURED_HSGP_REJECTION_AUDIT_COMPLETE")
