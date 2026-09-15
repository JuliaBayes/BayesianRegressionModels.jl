using MCMCDiagnosticTools, Statistics
const N_SUBJECTS = 20

function write_tsv(path,rows)
    keys=propertynames(first(rows))
    open(path,"w") do io
        println(io,join(keys,'\t'))
        for row in rows;println(io,join((getproperty(row,k) for k in keys),'\t'));end
    end
end

function qoi_names()
    vcat(["population_mean_intercept_at_mean_load","population_mean_load_slope","population_log_sigma_intercept",
          "mean_intercept_group_sd","mean_slope_group_sd","log_sigma_group_sd"],
         ["subject_$(700+j)_total_intercept" for j in 1:N_SUBJECTS],
         ["subject_$(700+j)_total_load_slope" for j in 1:N_SUBJECTS],
         ["subject_$(700+j)_residual_sd" for j in 1:N_SUBJECTS])
end

function scientific_diagnostics(label,qois,sampling,total,divergences,out)
    @assert size(qois,2)==66 && all(isfinite,qois)
    cube=reshape(qois,size(qois,1),1,size(qois,2))
    ess=vec(MCMCDiagnosticTools.ess(cube;kind=:bulk))
    tail=vec(MCMCDiagnosticTools.ess(cube;kind=:tail))
    rh=vec(MCMCDiagnosticTools.rhat(cube))
    names=qoi_names()
    write_tsv(joinpath(out,label*"-qois.tsv"),[(;parameter=names[j],mean=mean(qois[:,j]),sd=std(qois[:,j]),
        bulk_ess=ess[j],tail_ess=tail[j],split_rhat=rh[j]) for j in eachindex(names)])
    summary=(;arm=label,draws=size(qois,1),min_bulk_ess=minimum(ess),limiting_qoi=names[argmin(ess)],
        sampling_gradients=sampling,total_gradients=total,
        ess_per_1000_sampling_gradients=1000minimum(ess)/sampling,
        ess_per_1000_total_gradients=1000minimum(ess)/total,
        max_split_rhat=maximum(rh),divergences)
    write_tsv(joinpath(out,label*"-summary.tsv"),[summary])
    println("SCIENTIFIC_RESULT ",summary);flush(stdout)
    summary
end
