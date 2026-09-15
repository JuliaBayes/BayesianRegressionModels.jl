using BridgeStan, Serialization, JSON, Statistics, LinearAlgebra, Printf
include("diagnostics.jl")
include("s2z.jl")
const J=20
const XBAR=mean(Float64.(JSON.parsefile(joinpath(@__DIR__,"reference","ordinary_ncp.json"))["Z_1_2"]))

function prepare_s2z_pairs(fitroot,nativeroot,totalroot,out)
    fit=deserialize(joinpath(fitroot,"s2z_auto.jls"))
    names=deserialize(joinpath(totalroot,"total_ncp.jls")).names
    idx=Dict(n=>i for (i,n) in enumerate(names))
    coords=Dict(key=>(;totals=[idx["total_$key.$j.$k"] for j in 1:J,k in 1:count],
        scales=[idx["total_scale_$(key)_tau.$k"] for k in 1:count],
        mixture=[idx["total_mixture_$(key).1"]]) for (key,count) in ((:mu,2),(:logsigma,1)))
    tm=(;names,coords)
    dir=joinpath(nativeroot,"s2z_auto")
    model=BridgeStan.StanModel(joinpath(dir,"clean.stan"),joinpath(dir,"resolved-data.json");warn=false)
    l=s2z_layout(model,"s2z_auto",JSON.parsefile(joinpath(dir,"resolved-data.json")))
    controls=vec(l.rho);chosen=[argmin(controls)]
    unused=setdiff(eachindex(controls),chosen)
    push!(chosen,unused[argmin(abs.(controls[unused].-.5))])
    unused=setdiff(eachindex(controls),chosen)
    push!(chosen,unused[argmax(controls[unused])])
    n=size(fit.positions,2);@assert n==2000
    cp=zeros(60,n);ncp=similar(cp);acp=similar(cp)
    source_error=0.;zero_sum_error=0.;total_error=0.
    for draw in 1:n
        source=BridgeStan.param_constrain(model,fit.positions[:,draw])
        q=fit.total_positions[:,draw]
        total_error=max(total_error,maximum(abs.(first(s2z_to_total(source,l,tm)).-q)))
        totals=hcat(q[coords[:mu].totals],q[coords[:logsigma].totals])
        for k in 1:3
            ids=(k-1)*J+1:k*J;r=totals[:,k].-mean(totals[:,k]);tau=source[l.tau[k]]
            scale=1 .-l.rho[:,k].+l.rho[:,k].*tau
            shift=-sum(r.*scale)/sum(scale)
            subject_source=(r.+shift).*scale./tau
            source_error=max(source_error,maximum(abs.(subject_source-l.Q*source[l.z[:,k]])))
            zero_sum_error=max(zero_sum_error,abs(sum(subject_source)),abs(sum(r)))
            cp[ids,draw]=r;ncp[ids,draw]=r./tau;acp[ids,draw]=subject_source
        end
    end
    @assert source_error<1e-8 && zero_sum_error<1e-7 && total_error<1e-8
    rows=NamedTuple[];selection=NamedTuple[]
    scales=vcat(coords[:mu].scales,coords[:logsigma].scales)
    for (panelnum,index) in enumerate(chosen)
        margin=cld(index,J);subject=700+mod1(index,J)
        term=("intercept","slope","log SD")[margin]
        panel=@sprintf("%d: %d %s, ρ=%.2f",panelnum,subject,term,controls[index])
        push!(selection,(;panel,subject,term,rho=controls[index]))
        for (column,values) in (("1. CP",cp),("2. NCP",ncp),("3. ACP (brms auto)",acp)),draw in 1:n
            push!(rows,(;panel,column,draw,log_group_sd=fit.total_positions[scales[margin],draw],coordinate=values[index,draw]))
        end
    end
    @assert length(rows)==18000
    write_tsv(joinpath(out,"s2z_pairs.tsv"),rows)
    write_tsv(joinpath(out,"s2z_pairs_selection.tsv"),selection)
    write_tsv(joinpath(out,"s2z_pairs_audit.tsv"),[(;draws=n,source_error,zero_sum_error,total_error)])
    println("S2Z_PAIRS_COMPLETE source_error=",source_error)
end
prepare_s2z_pairs(ARGS...)
