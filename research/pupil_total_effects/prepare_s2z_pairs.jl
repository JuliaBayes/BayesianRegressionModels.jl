#= Visualize the same saved S2Z-auto draws in CP/NCP/auto subject coordinates.

Qz has J labelled entries but J-1 independent directions per term. The auto
column is Qz reconstructed from physical contrasts, not a per-subject shortcut
that ignores the centering correction.
=#
include("s2z_whmc.jl")
using Printf

function prepare_s2z_pairs(root,output)
    mkpath(output)
    fit=deserialize(joinpath(root,"pupil-s2z-whmc-v3/s2z_auto/fit.jls"))
    standata=JSON.parsefile(joinpath(root,"pupil-s2z-native-v1/s2z_auto/resolved-data.json"))
    l=s2z_layout(fit.source_named_names,"s2z_auto",standata)
    data=PTE.load_data();rho=vec(l.rho)
    selected=[argmin(rho)]
    unused=setdiff(1:40,selected)
    push!(selected,unused[argmin(abs.(rho[unused].-0.5))])
    unused=setdiff(1:40,selected)
    push!(selected,unused[argmax(rho[unused])])
    n=size(fit.positions,2)
    cp=zeros(40,n);ncp=similar(cp);acp=similar(cp)
    source_error=0.0;zero_sum_error=0.0;total_error=0.0
    for s in 1:n
        q=fit.positions[:,s]
        source=fit.source_named_positions[:,s]
        total_error=max(total_error,maximum(abs.(first(s2z_to_total(source,l,data)).-q)))
        for k in 1:2
            ids=(k-1)*20+1:k*20
            values=q[4 .+ ids]
            r=values.-mean(values)
            tau=exp(q[k]);scale=1 .-l.rho[:,k].+l.rho[:,k].*tau
            shift=-sum(r.*scale)/sum(scale)
            subject_source=(r.+shift).*scale./tau
            actual=l.Q*source[l.z[(k-1)*19+1:k*19]]
            source_error=max(source_error,maximum(abs.(subject_source.-actual)))
            zero_sum_error=max(zero_sum_error,abs(sum(subject_source)),abs(sum(r)))
            @test l.Q'subject_source ≈ source[l.z[(k-1)*19+1:k*19]] atol=1e-8 rtol=1e-10
            cp[ids,s]=r;ncp[ids,s]=r./tau;acp[ids,s]=subject_source
        end
    end
    @test source_error<1e-8 && zero_sum_error<1e-7 && total_error<1e-8
    rows=NamedTuple[];selection=NamedTuple[]
    for (k,j) in enumerate(selected)
        kind=j<=20 ? "a" : "b"
        subject=data.ids[mod1(j,20)]
        panel=@sprintf("%d: %s%d (ρ=%.3f)",k,kind,subject,rho[j])
        push!(selection,(;selection=k,term=kind,subject,coordinate=j,rho=rho[j]))
        for (column,values) in (("1. CP",cp),("2. NCP",ncp),("3. ACP (brms auto)",acp))
            for s in 1:n
                push!(rows,(;panel,column,draw=s,log_group_sd=fit.positions[j<=20 ? 1 : 2,s],
                             coordinate=values[j,s]))
            end
        end
    end
    @test n==2000 && length(rows)==18000
    @test all(r->isfinite(r.coordinate)&&isfinite(r.log_group_sd),rows)
    write_tsv(joinpath(output,"s2z_pairs.tsv"),rows)
    write_tsv(joinpath(output,"s2z_pairs_selection.tsv"),selection)
    write_tsv(joinpath(output,"s2z_pairs_audit.tsv"),[(;draws=n,source_error,zero_sum_error,total_error)])
    println("S2Z_PAIRS_COMPLETE rows=",length(rows)," max_source_error=",source_error)
end

if abspath(PROGRAM_FILE)==@__FILE__
    prepare_s2z_pairs(ARGS...)
end
