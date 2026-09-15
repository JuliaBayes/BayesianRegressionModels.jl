using Serialization, Statistics, DelimitedFiles
include("diagnostics.jl")

function validate_saved(root,out)
    table,header=readdlm(joinpath(out,"comparison.tsv"),'\t';header=true)
    columns=Dict(String(k)=>i for (i,k) in enumerate(vec(header)))
    labels=String.(table[:,columns["arm"]])
    function load_qois(label)
        dir=startswith(label,"total_") ? "pupil4-totals-v1" :
            endswith(label,"_native") ? "pupil4-summary-v1" : "pupil4-brms-whmc-v1"
        name=endswith(label,"_whmc") ? chop(label;tail=5) : label
        deserialize(joinpath(root,dir,name*"-qois.jls"))
    end
    arrays=Dict(label=>load_qois(label) for label in labels)
    moments=Dict(label=>begin
        q=arrays[label];@assert size(q)==(2000,66) && all(isfinite,q)
        cube=reshape(q,2000,1,66)
        (;mean=vec(mean(q;dims=1)),mcse=vec(MCMCDiagnosticTools.mcse(cube;kind=mean)),
          ess=vec(MCMCDiagnosticTools.ess(cube;kind=:bulk)))
    end for label in labels)
    reference=moments["total_online_position"]
    details=NamedTuple[];checks=NamedTuple[]
    names=qoi_names();baseline=moments["ordinary_ncp_native"]
    baseline_sampling=Float64(table[1,columns["sampling_gradients"]])
    baseline_total=Float64(table[1,columns["total_gradients"]])
    for (i,label) in enumerate(labels)
        m=moments[label]
        z=abs.(m.mean .- reference.mean) ./ sqrt.(m.mcse .^ 2 .+ reference.mcse .^ 2)
        for k in eachindex(names)
            push!(details,(;arm=label,quantity=names[k],mean=m.mean[k],mcse=m.mcse[k],
                bulk_ess=m.ess[k],mean_difference_mcse=z[k]))
        end
        invariant_min=minimum(m.ess[4:end])
        sampling=Float64(table[i,columns["sampling_gradients"]])
        total=Float64(table[i,columns["total_gradients"]])
        push!(checks,(;arm=label,max_mean_difference_mcse=maximum(z),
            largest_difference=names[argmax(z)],quantities_above_3mcse=count(>(3),z),
            invariant63_min_ess=invariant_min,
            invariant_relative_sampling=(invariant_min/sampling)/(minimum(baseline.ess[4:end])/baseline_sampling),
            invariant_relative_total=(invariant_min/total)/(minimum(baseline.ess[4:end])/baseline_total)))
    end
    write_tsv(joinpath(out,"per_quantity_mcse.tsv"),details)
    write_tsv(joinpath(out,"saved_draw_checks.tsv"),checks)
    println("SAVED_DRAW_VALIDATION_COMPLETE arms=",length(labels))
    for c in checks;println(c);end
end
validate_saved(ARGS...)
