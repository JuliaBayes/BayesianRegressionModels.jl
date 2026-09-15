include("recovery.jl")

function compare_matrix(root,output)
    mkpath(output)
    data=PTE.load_data()
    specs=[
        ("brms NCP","Native Stan","pupil-student-native-v1/native_ncp.jls",0),
        ("brms NCP","WHMC","pupil-student-brms-v1/brms_ncp.jls",0),
        ("brms CP","WHMC","pupil-ordinary-cp-whmc-v1/brms_cp.jls",0)]
    for (label,name) in (("CP","s2z_cp"),("NCP","s2z_ncp"),("auto","s2z_auto"))
        base="pupil-s2z-whmc-v3/"*name
        push!(specs,("brms S2Z "*label,"Native Stan",base*"/native_analysis/fit.jls",0))
        push!(specs,("brms S2Z "*label,"WHMC",base*"/fit.jls",0))
    end
    pilot=deserialize(joinpath(root,"pupil-student-total-v1/scaled_total.jls"))
    append!(specs,[
        ("total coefficients CP","WHMC","pupil-student-total-cp-v1/centered_total.jls",0),
        ("total coefficients NCP","WHMC","pupil-student-total-v1/scaled_total.jls",0),
        ("total coefficients ACP","WHMC","pupil-student-total-v1/partial.jls",pilot.all_gradient_calls)])
    rows=NamedTuple[];parameters=NamedTuple[];agreement=NamedTuple[]
    reference=deserialize(joinpath(root,"pupil-student-total-v1/partial.jls"))
    ref=reference.positions[1:44,:]
    ref_ds=diagnostics(ref)
    names=PTE.coordinate_names(data)
    for (label,sampler,file,pilot_cost) in specs
        record=deserialize(joinpath(root,file))
        common=record.positions[1:44,:]
        shared=diagnostics(common)
        precursor=get(record,:precursor_gradient_calls,0)
        total=record.all_gradient_calls+precursor+pilot_cost
        samples=permutedims(reshape(common,44,size(common,2),1),(2,3,1))
        rhat=vec(MCMCDiagnosticTools.rhat(samples))
        tail=vec(MCMCDiagnosticTools.ess(samples;kind=:tail))
        scopes=[("common44",common)]
        if hasproperty(record,:original_physical_positions)
            physical=record.original_physical_positions
        elseif hasproperty(record,:original_named_positions)
            physical=baseline_original(record,data)
        else
            recovery_dir=label=="total coefficients CP" ? "pupil-student-total-cp-v1" : "pupil-student-recovery-v1"
            arm=label=="total coefficients CP" ? "centered_total" : label=="total coefficients NCP" ? "scaled_total" : "partial"
            physical=deserialize(joinpath(root,recovery_dir,arm*"_recovered_seed101.jls")).positions
        end
        append!(scopes,[("original_physical46",physical[1:46,:]),
                        ("original_NCP46",original_ncp(physical)[1:46,:])])
        if size(record.positions,1)==45
            push!(scopes,("common44_plus_mixture",record.positions))
        end
        for (scope,values) in scopes
            ds=diagnostics(values)
            scope_names=scope=="common44" ? names : scope=="common44_plus_mixture" ?
                PTE.coordinate_names(data,PTE.StudentMixtureMean()) : original_names(data)[1:46]
            k=argmin(ds.bulk)
            push!(rows,(;model=label,sampler,scope,total_gradients=total,
                min_ess_per_sampling_gradient=ds.bulk[k]/record.sampling_gradients,
                min_ess_per_total_gradient=ds.bulk[k]/total,
                min_bulk_ess=ds.bulk[k],limiting_parameter=scope_names[k],
                sampling_gradients=record.sampling_gradients,fit_total_gradients=record.all_gradient_calls,
                precursor_gradients=precursor,pilot_gradients=pilot_cost,
                divergences=record.divergences,draws=size(values,2),
                common_max_split_rhat=maximum(rhat),common_min_tail_ess=minimum(tail),
                saved_fit=file))
        end
        for j in 1:44
            push!(parameters,(;model=label,sampler,parameter=names[j],mean=mean(common[j,:]),
                sd=std(common[j,:]),bulk_ess=shared.bulk[j],mcse=shared.mcse[j],split_rhat=rhat[j],tail_ess=tail[j]))
            difference=mean(common[j,:])-mean(ref[j,:])
            combined=sqrt(shared.mcse[j]^2+ref_ds.mcse[j]^2)
            push!(agreement,(;model=label,sampler,parameter=names[j],difference,
                combined_mcse=combined,difference_in_combined_mcse=difference/combined))
        end
    end
    write_tsv(joinpath(output,"all_scopes.tsv"),rows)
    primary=filter(r->r.scope=="common44",rows)
    write_tsv(joinpath(output,"common44.tsv"),primary)
    write_tsv(joinpath(output,"common_parameters.tsv"),parameters)
    write_tsv(joinpath(output,"posterior_agreement.tsv"),agreement)
    open(joinpath(output,"table.md"),"w") do io
        println(io,"| Model | Sampler | Total gradients | Min ESS / sampling gradient | Min ESS / total gradient |")
        println(io,"|---|---|---:|---:|---:|")
        for r in primary
            println(io,"| ",r.model," | ",r.sampler," | ",r.total_gradients," | ",
                round(r.min_ess_per_sampling_gradient;sigdigits=4)," | ",
                round(r.min_ess_per_total_gradient;sigdigits=4)," |")
        end
    end
    println(read(joinpath(output,"table.md"),String))
    println("MATRIX_COMPLETE\t",output)
end

if abspath(PROGRAM_FILE)==@__FILE__
    compare_matrix(ARGS...)
end
