include("run.jl")
record=deserialize(joinpath(OUTPUT,REQUEST,"transport-failure.jls"))
out=joinpath(OUTPUT,REQUEST,"transport-analysis");mkpath(out)
stan=make_stan(out)
adaptive=adaptive_centering_problem(stan.sb,stan.density,ENZYME_BACKEND)
WarmupHMC.restore_reparam_sources!(adaptive,[idx=>r.source for (idx,r) in record.new_ir.pairs])
direct=WarmupHMC.IndexedReparametrization([idx=>WarmupHMC.Reparametrization(old.source,new.source,new.args...)
    for ((idx,old),(_,new)) in zip(record.old_ir.pairs,record.new_ir.pairs)])
reverse_direct=WarmupHMC.IndexedReparametrization([idx=>WarmupHMC.Reparametrization(new.source,old.source,new.args...)
    for ((idx,old),(_,new)) in zip(record.old_ir.pairs,record.new_ir.pairs)])
objective(x,ir,g)=let (jac,old)=ir(x);jac+dot(g,old);end
bad=unique(i[2] for i in findall(!isfinite,record.gradient))
println("BAD_COLUMNS ",bad," pairs=",length(record.old_ir.pairs))
rows=NamedTuple[]
for j in unique(vcat(bad,round.(Int,range(1,size(record.position,2);length=12))))
    x=record.position[:,j];old=record.old_position[:,j];g=record.old_gradient[:,j]
    ljac,q=direct(x)
    new=last(reverse_direct(old))
    _,direct_g=WarmupHMC.value_and_gradient(objective,ENZYME_BACKEND,x,
        WarmupHMC.Constant(direct),WarmupHMC.Constant(collect(g)))
    value,target_g=LogDensityProblems.logdensity_and_gradient(adaptive,x)
    row=(;column=j,bad=j in bad,old_point_error=maximum(abs.(q-old)./(1 .+abs.(old))),
        new_point_error=maximum(abs.(new-x)./(1 .+abs.(x))),
        direct_gradient_finite=all(isfinite,direct_g),target_gradient_finite=all(isfinite,target_g),
        gradient_error=maximum(abs.(direct_g-target_g)./(1 .+abs.(target_g))),
        direct_bad=join(findall(!isfinite,direct_g),","),target_bad=join(findall(!isfinite,target_g),","))
    push!(rows,row);println("DIRECT_TRANSPORT ",row)
end
write_tsv(joinpath(out,"comparison.tsv"),rows)
println("TRANSPORT_ANALYSIS_COMPLETE")
