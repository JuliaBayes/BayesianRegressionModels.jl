include("common.jl")
using DelimitedFiles
root,out=ARGS
totaldir=joinpath(root,"pupil3-builtin-totals-v2")
brmsdir=joinpath(root,"pupil3-builtin-brms-summary-v1")
tm=total_model(joinpath(totaldir,"model"))
values,header=readdlm(joinpath(out,"comparison.tsv"),'\t',Any;header=true)
columns=Dict(String(k)=>i for (i,k) in enumerate(vec(header)))
labels=String.(values[:,columns["arm"]]);@assert length(labels)==15
arrays=Dict(label=>deserialize(joinpath(startswith(label,"total_") ? totaldir : brmsdir,
    label*"-qois.jls")) for label in labels)
moments=Dict(label=>begin
    @assert size(q)==(2000,46) && all(isfinite,q)
    cube=reshape(q,2000,1,46)
    (;mean=vec(mean(q;dims=1)),mcse=vec(MCMCDiagnosticTools.mcse(cube;kind=mean)),
        ess=vec(MCMCDiagnosticTools.ess(cube;kind=:bulk)))
end for (label,q) in arrays)
names=qoi_names();checks=NamedTuple[];details=NamedTuple[];seeds=NamedTuple[]
baseline=moments["ordinary_ncp_native"];reference=moments["total_cp"]
for (i,label) in enumerate(labels)
    m=moments[label];z=abs.(m.mean-reference.mean)./sqrt.(m.mcse.^2+reference.mcse.^2)
    invariant=minimum(m.ess[3:end])
    push!(checks,(;arm=label,max_mean_difference_mcse=maximum(z),largest_difference=names[argmax(z)],
        quantities_above_3mcse=count(>(3),z),invariant_min_ess=invariant,
        invariant_relative_sampling=(invariant/Float64(values[i,columns["sampling_gradients"]]))/
            (minimum(baseline.ess[3:end])/Float64(values[1,columns["sampling_gradients"]])),
        invariant_relative_total=(invariant/Float64(values[i,columns["total_gradients"]]))/
            (minimum(baseline.ess[3:end])/Float64(values[1,columns["total_gradients"]]))))
    for k in 1:46
        push!(details,(;arm=label,parameter=names[k],mean=m.mean[k],mcse=m.mcse[k],bulk_ess=m.ess[k],mean_difference_mcse=z[k]))
    end
    startswith(label,"total_")||continue
    fit=deserialize(joinpath(totaldir,label*".jls"))
    for seed in 101:110
        q=total_qois(tm,fit.positions;seed)
        @assert q[:,3:end]==arrays[label][:,3:end]
        ess=vec(MCMCDiagnosticTools.ess(reshape(q,2000,1,46);kind=:bulk))
        push!(seeds,(;arm=label,recovery_seed=seed,min_bulk_ess=minimum(ess),limiting_qoi=names[argmin(ess)]))
    end
end
write_tsv(joinpath(out,"saved_draw_checks.tsv"),checks)
write_tsv(joinpath(out,"per_quantity_mcse.tsv"),details)
write_tsv(joinpath(out,"recovery_seed_sensitivity.tsv"),seeds)
println("PUPIL3_SAVED_VALIDATION_COMPLETE")
