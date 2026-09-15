using Serialization, Statistics, MCMCDiagnosticTools, DelimitedFiles

function validate_case(d,m,totaldir,brmsdir,summarydir,out)
    files=vcat(readdir(totaldir;join=true),readdir(brmsdir;join=true),readdir(summarydir;join=true))
    summaries=Dict{String,Any}()
    for file in filter(p->endswith(p,"-summary.tsv"),files)
        values,header=readdlm(file,'\t',Any;header=true)
        row=Dict(String(k)=>values[1,i] for (i,k) in enumerate(vec(header)))
        summaries[String(row["arm"])]=row
    end
    @assert length(summaries)==14
    function load_qoi(label)
        dir=endswith(label,"_native") ? summarydir : startswith(label,"s2z_") ? brmsdir : totaldir
        file=startswith(label,"s2z_")&&endswith(label,"_whmc") ? chop(label;tail=5) : label
        deserialize(joinpath(dir,file*"-qois.jls"))
    end
    arrays=Dict(label=>load_qoi(label) for label in keys(summaries))
    names=qoi_names(d);n=length(names)
    moments=Dict(label=>begin
        @assert size(q)==(2000,n) && all(isfinite,q)
        cube=reshape(q,2000,1,n)
        (;mean=vec(mean(q;dims=1)),mcse=vec(MCMCDiagnosticTools.mcse(cube;kind=mean)),
            ess=vec(MCMCDiagnosticTools.ess(cube;kind=:bulk)))
    end for (label,q) in arrays)
    reference=moments["total_cp"];baseline=moments["ordinary_ncp_native"]
    invariant=setdiff(1:n,d.K==1 ? [1] : [1,2])
    b=summaries["ordinary_ncp_native"];checks=NamedTuple[];details=NamedTuple[];seeds=NamedTuple[]
    for label in sort(collect(keys(arrays)))
        values=moments[label];cost=summaries[label]
        z=abs.(values.mean-reference.mean)./sqrt.(values.mcse.^2+reference.mcse.^2)
        minimum_invariant=minimum(values.ess[invariant])
        push!(checks,(;arm=label,max_mean_difference_mcse=maximum(z),largest_difference=names[argmax(z)],
            quantities_above_3mcse=count(>(3),z),invariant_min_ess=minimum_invariant,
            invariant_relative_sampling=(minimum_invariant/cost["sampling_gradients"])/(minimum(baseline.ess[invariant])/b["sampling_gradients"]),
            invariant_relative_total=(minimum_invariant/cost["total_gradients"])/(minimum(baseline.ess[invariant])/b["total_gradients"])))
        for k in 1:n
            push!(details,(;arm=label,parameter=names[k],mean=values.mean[k],mcse=values.mcse[k],bulk_ess=values.ess[k],mean_difference_mcse=z[k]))
        end
        startswith(label,"total_")||continue
        fit=deserialize(joinpath(totaldir,label*".jls"))
        for seed in 101:110
            q=copy(arrays[label])
            population=AIRTotals.BRM.recover_population_draws(m.sb,permutedims(fit.positions),m.names;rng=Xoshiro(seed))[:mu].population
            q[:,d.K==1 ? [1] : [1,2]]=population
            @assert q[:,invariant]==arrays[label][:,invariant]
            ess=vec(MCMCDiagnosticTools.ess(reshape(q,2000,1,n);kind=:bulk))
            push!(seeds,(;arm=label,recovery_seed=seed,min_bulk_ess=minimum(ess),limiting_qoi=names[argmin(ess)]))
        end
    end
    write_air_tsv(joinpath(out,"saved_draw_checks.tsv"),checks)
    write_air_tsv(joinpath(out,"per_quantity_mcse.tsv"),details)
    write_air_tsv(joinpath(out,"recovery_seed_sensitivity.tsv"),seeds)
    println("AIR_SAVED_VALIDATION_COMPLETE ",d.hierarchy)
end
