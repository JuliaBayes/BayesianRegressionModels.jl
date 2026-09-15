include("common.jl")
using DelimitedFiles

function reuse(root,out)
    mkpath(out)
    table,header=readdlm(joinpath(@__DIR__,"..","pupil_total_effects","results",
        "student_mixture","matrix","common44.tsv"),'\t',Any;header=true)
    columns=Dict(String(k)=>i for (i,k) in enumerate(vec(header)))
    labels=Dict("brms NCP"=>"ordinary_ncp","brms CP"=>"ordinary_cp",
        "brms S2Z CP"=>"s2z_cp","brms S2Z NCP"=>"s2z_ncp","brms S2Z auto"=>"s2z_auto")
    records=NamedTuple[]
    for row in eachrow(table)
        value(k)=row[columns[k]]
        model=String(value("model"));haskey(labels,model)||continue
        sampler=String(value("sampler"));label=labels[model]*(sampler=="WHMC" ? "_whmc" : "_native")
        path=joinpath(root,String(value("saved_fit")));fit=deserialize(path)
        beta=if hasproperty(fit,:original_physical_positions)
            fit.original_physical_positions[1:2,:]
        else
            idx(n)=only(findall(==(n),fit.original_named_names))
            fit.original_named_positions[idx.(["Intercept","b.1"]),:]
        end
        qois=permutedims(vcat(beta,exp.(fit.positions[1:2,:]),fit.positions[3:44,:]))
        serialize(joinpath(out,label*"-qois.jls"),qois)
        scientific_diagnostics(label,qois,fit.sampling_gradients,Int(value("total_gradients")),
            Int(value("divergences")),out)
        push!(records,(;arm=label,saved_fit=String(value("saved_fit")),draws=size(qois,1)))
    end
    @assert length(records)==9
    write_tsv(joinpath(out,"reused_brms_fits.tsv"),records)
    println("PUPIL3_REUSED_BRMS_COMPLETE")
end
reuse(ARGS...)
