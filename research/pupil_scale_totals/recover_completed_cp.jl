using WarmupHMC, Serialization, JSON

# One-time recovery of the completed fit whose subsequent BridgeStan array-view
# conversion failed. No fitting occurs here. This plain Stan target has no
# reparametrizer, and finalize_warmup! performs no density/gradient evaluation.
out,logpath=ARGS
payload=deserialize(joinpath(out,"s2z_cp-checkpoints","cp_latest.jls"))
log=read(logpath,String)
matches=collect(eachmatch(r"BOUNDARY s2z_cp window window=(\d+) gradients=(\d+)",log))
last_boundary=last(matches)
@assert parse(Int,last_boundary.captures[1])==payload.outer_counter==7
@assert size(payload.posterior_position)==(65,2000)
@assert occursin("MethodError: no method matching param_constrain",log)
calls=parse(Int,last_boundary.captures[2])
@assert calls==104514>payload.total_evaluation_counter>=payload.sampling_evaluation_counter
@assert isempty(payload.reparam_sources)
base=(;positions=Matrix(payload.posterior_position),
    sampling_gradients=payload.sampling_evaluation_counter,
    all_gradient_calls=calls,total_gradient_calls=calls,pilot_gradient_calls=0,
    divergences=payload.n_divergent_samples,fit_seconds=missing,numerical_rejections=missing)
path=joinpath(out,"s2z_cp-raw.jls")
@assert !isfile(path) "Completed raw record already exists"
serialize(path,base)
open(joinpath(out,"s2z_cp-checkpoint-recovery.json"),"w") do io
    JSON.print(io,Dict("checkpoint"=>joinpath(out,"s2z_cp-checkpoints","cp_latest.jls"),
        "log"=>logpath,"draws"=>2000,"all_gradient_calls"=>calls,
        "sampling_gradient_calls"=>base.sampling_gradients,"refitted"=>false,
        "counter_evidence"=>"Last window callback immediately before non-evaluating finalization; postprocessing subsequently failed."),2)
end
println("COMPLETED_CP_RECOVERED draws=2000 gradients=",calls)
