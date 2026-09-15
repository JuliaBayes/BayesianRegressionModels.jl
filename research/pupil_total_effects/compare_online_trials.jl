include("recovery.jl")

function online_qois(record,data)
    if hasproperty(record,:original_named_positions)
        beta=baseline_original(record,data)[1:2,:]
    elseif hasproperty(record,:original_physical_positions)
        beta=record.original_physical_positions[1:2,:]
    else
        prior=get(record,:mean_prior,PTE.GaussianMean())
        rng=Xoshiro(101)
        beta=reduce(hcat,(begin
            mu,covariance=population_conditional(q,data,prior)
            mu+cholesky(covariance).L*randn(rng,2)
        end for q in eachcol(record.positions)))
    end
    vcat(beta,record.positions[1:44,:])
end

function compare_online_trials(root,output)
    mkpath(output);data=PTE.load_data()
    rows=NamedTuple[];parameters=NamedTuple[]
    names=vcat(original_names(data)[1:2],PTE.coordinate_names(data))
    specifications=[
        ("Gaussian","brms NCP / Native Stan","pupil-gaussian-native-v1/native_ncp.jls",nothing),
        ("Gaussian","brms NCP / WHMC","pupil-brms-baseline-v3/brms_ncp.jls",nothing),
        ("Gaussian","Totals NCP / WHMC","pupil-total-v1/scaled_total.jls",nothing),
        ("Gaussian","Post-hoc position / WHMC","pupil-total-v1/partial.jls","pupil-total-v1/scaled_total.jls"),
        ("Gaussian","Post-hoc gradient / WHMC","pupil-gaussian-offline-gradient-v1/partial_gradient.jls","pupil-total-v1/scaled_total.jls"),
        ("Gaussian","Historical broken online / WHMC","pupil-total-v1/online.jls",nothing),
        ("Gaussian","Online position / WHMC","pupil-online-position-v1/online_position.jls",nothing),
        ("Gaussian","Online gradient / WHMC","pupil-online-gradient-v1/online_gradient.jls",nothing),
        ("Student-t","brms NCP / Native Stan","pupil-student-native-v1/native_ncp.jls",nothing),
        ("Student-t","brms NCP / WHMC","pupil-student-brms-v1/brms_ncp.jls",nothing),
        ("Student-t","Totals NCP / WHMC","pupil-student-total-v1/scaled_total.jls",nothing),
        ("Student-t","Totals CP / WHMC","pupil-student-total-cp-v1/centered_total.jls",nothing),
        ("Student-t","brms S2Z auto / WHMC","pupil-s2z-whmc-v3/s2z_auto/fit.jls",nothing),
        ("Student-t","Post-hoc position / WHMC","pupil-student-total-v1/partial.jls","pupil-student-total-v1/scaled_total.jls"),
        ("Student-t","Post-hoc gradient / WHMC","pupil-student-offline-gradient-v1/partial_gradient.jls","pupil-student-total-v1/scaled_total.jls"),
        ("Student-t","Online position / WHMC","pupil-student-online-position-v1/online_position.jls",nothing),
        ("Student-t","Online gradient / WHMC","pupil-student-online-gradient-v1/online_gradient.jls",nothing),
    ]
    for (prior,arm,path,pilot) in specifications
        isfile(joinpath(root,path)) || continue
        record=deserialize(joinpath(root,path))
        values=online_qois(record,data)
        @test size(values)==(46,2000)
        ds=diagnostics(values);j=argmin(ds.bulk)
        samples=permutedims(reshape(values,46,2000,1),(2,3,1))
        rhat=vec(MCMCDiagnosticTools.rhat(samples));tail=vec(MCMCDiagnosticTools.ess(samples;kind=:tail))
        total=record.all_gradient_calls+get(record,:precursor_gradient_calls,0)+
              (isnothing(pilot) ? 0 : deserialize(joinpath(root,pilot)).all_gradient_calls)
        push!(rows,(;prior,arm,total_gradients=total,sampling_gradients=record.sampling_gradients,
            min_bulk_ess=ds.bulk[j],limiting_qoi=names[j],sampling_efficiency=ds.bulk[j]/record.sampling_gradients,
            total_efficiency=ds.bulk[j]/total,divergences=record.divergences,
            max_split_rhat=maximum(rhat),min_tail_ess=minimum(tail)))
        for k in 1:46
            push!(parameters,(;prior,arm,qoi=names[k],bulk_ess=ds.bulk[k],mcse=ds.mcse[k],
                              mean=mean(values[k,:]),sd=std(values[k,:])))
        end
    end
    relative=map(rows) do row
        base=only(filter(r->r.prior==row.prior&&r.arm=="brms NCP / Native Stan",rows))
        merge(row,(;relative_sampling_efficiency=row.sampling_efficiency/base.sampling_efficiency,
                   relative_total_efficiency=row.total_efficiency/base.total_efficiency))
    end
    write_tsv(joinpath(output,"qoi46.tsv"),relative)
    write_tsv(joinpath(output,"parameters.tsv"),parameters)
    for row in relative
        println(row)
    end
    println("ONLINE_QOI_COMPARISON_COMPLETE rows=",length(relative))
end

if abspath(PROGRAM_FILE)==@__FILE__
    compare_online_trials(ARGS...)
end
