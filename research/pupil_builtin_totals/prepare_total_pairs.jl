using Serialization, Statistics, Printf
include("common.jl")
input,out=ARGS;mkpath(out)
fit=deserialize(joinpath(input,"total_posthoc_position.jls"))
selected=deserialize(joinpath(input,"selected_position.jls"))
names=fit.names;idx=Dict(n=>i for (i,n) in enumerate(names))
controls=selected.centeredness
chosen=[argmin(controls)]
unused=setdiff(eachindex(controls),chosen)
push!(chosen,unused[argmin(abs.(controls[unused].-.5))])
unused=setdiff(eachindex(controls),chosen)
push!(chosen,unused[argmax(controls[unused])])
rows=NamedTuple[];selection=NamedTuple[];error=0.
for (panelnum,k) in enumerate(chosen)
    i=selected.indices[k];name=names[i]
    matchname=match(r"^total_(mu|logsigma)\.(\d+)\.(\d+)$",name)
    @assert !isnothing(matchname)
    block,subject,margin=matchname.captures
    j=parse(Int,subject);m=parse(Int,margin)
    scale=idx["total_scale_$(block)_tau.$m"]
    location=block=="mu" && m==1 ? 5651.9 : 0.
    term=block=="logsigma" ? "log residual SD" : m==1 ? "mean intercept" : "load slope"
    shortterm=block=="logsigma" ? "log SD" : m==1 ? "intercept" : "slope"
    panel=@sprintf("%d: %d %s, c=%.2f",panelnum,700+j,shortterm,controls[k])
    push!(selection,(;panel,parameter=name,centeredness=controls[k]))
    for (column,c) in (("1. CP",1.),("2. NCP",0.),("3. ACP (position)",controls[k]))
        for s in axes(fit.positions,2)
            ell=fit.positions[scale,s];value=fit.positions[i,s]
            coordinate=c*location+(value-location)*exp((c-1)*ell)
            column=="3. ACP (position)" && (global error=max(error,abs(coordinate-fit.source_positions[i,s])))
            push!(rows,(;panel,column,draw=s,log_group_sd=ell,coordinate))
        end
    end
end
@assert error<1e-8 && length(rows)==18000
write_tsv(joinpath(out,"total_pairs.tsv"),rows)
write_tsv(joinpath(out,"total_pairs_selection.tsv"),selection)
write_tsv(joinpath(out,"total_pairs_audit.tsv"),[(;draws=2000,rows=length(rows),source_coordinate_error=error)])
println("TOTAL_PAIRS_COMPLETE source_coordinate_error=",error)
