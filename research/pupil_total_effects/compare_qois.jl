#= One scientific scope: population coefficients, scales, and subject totals.

Read saved matrix fits and saved conditional recoveries; run no HMC transitions.
=#
include("recovery.jl")
using DelimitedFiles, Printf

function compare_qois(root,output)
    mkpath(output)
    input,header=readdlm(joinpath(@__DIR__,"results/student_mixture/matrix/common44.tsv"),'\t',Any;header=true)
    columns=Dict(String(name)=>j for (j,name) in enumerate(vec(header)))
    data=PTE.load_data()
    names=vcat(original_names(data)[1:2],PTE.coordinate_names(data))
    rows=NamedTuple[];details=NamedTuple[]
    for row in eachrow(input)
        value(name)=row[columns[name]]
        label=String(value("model"));sampler=String(value("sampler"))
        record=deserialize(joinpath(root,String(value("saved_fit"))))
        if hasproperty(record,:original_physical_positions)
            beta=record.original_physical_positions[1:2,:]
        elseif hasproperty(record,:original_named_positions)
            idx(name)=only(findall(==(name),record.original_named_names))
            beta=record.original_named_positions[idx.(["Intercept","b.1"]),:]
        else
            recovery_dir=label=="total coefficients CP" ? "pupil-student-total-cp-v1" : "pupil-student-recovery-v1"
            arm=label=="total coefficients CP" ? "centered_total" : label=="total coefficients NCP" ? "scaled_total" : "partial"
            beta=deserialize(joinpath(root,recovery_dir,arm*"_recovered_seed101.jls")).positions[1:2,:]
        end
        quantities=vcat(beta,record.positions[1:44,:])
        @test size(quantities)==(46,2000)
        ds=diagnostics(quantities);j=argmin(ds.bulk)
        total=Int(value("total_gradients"));sampling=record.sampling_gradients
        push!(rows,(;model=label,sampler,total_gradients=total,sampling_gradients=sampling,
            min_ess=ds.bulk[j],limiting_qoi=names[j],sampling_efficiency=ds.bulk[j]/sampling,
            total_efficiency=ds.bulk[j]/total))
        for k in 1:46
            push!(details,(;model=label,sampler,qoi=names[k],bulk_ess=ds.bulk[k],mcse=ds.mcse[k]))
        end
    end
    baseline=only(filter(r->r.model=="brms NCP"&&r.sampler=="Native Stan",rows))
    relative=[merge(r,(;relative_sampling_efficiency=r.sampling_efficiency/baseline.sampling_efficiency,
        relative_total_efficiency=r.total_efficiency/baseline.total_efficiency)) for r in rows]
    @test length(relative)==12 && length(details)==12*46
    write_tsv(joinpath(output,"qoi46.tsv"),relative)
    write_tsv(joinpath(output,"parameters.tsv"),details)
    metrics=NamedTuple[]
    for r in relative
        for (metric,value) in (("1. Total gradients ↓",Float64(r.total_gradients)),
                              ("2. Sampling efficiency ↑",r.relative_sampling_efficiency),
                              ("3. Total efficiency ↑",r.relative_total_efficiency))
            push!(metrics,(;model=replace(r.model,"total coefficients"=>"Our totals"),
                           sampler=r.sampler,metric,value))
        end
    end
    write_tsv(joinpath(output,"efficiency_plot.tsv"),metrics)
    open(joinpath(output,"table.md"),"w") do io
        println(io,"| Model | Sampler | Total gradients | Relative sampling efficiency | Relative total efficiency |")
        println(io,"|---|---|---:|---:|---:|")
        for r in relative
            @printf(io,"| %s | %s | %d | %.3g× | %.3g× |\n",replace(r.model,"total coefficients"=>"Our totals"),r.sampler,r.total_gradients,r.relative_sampling_efficiency,r.relative_total_efficiency)
        end
    end
    println(read(joinpath(output,"table.md"),String))
    println("QOI46_COMPLETE baseline_min_ess=",baseline.min_ess," limiter=",baseline.limiting_qoi)
end

if abspath(PROGRAM_FILE)==@__FILE__
    compare_qois(ARGS...)
end
