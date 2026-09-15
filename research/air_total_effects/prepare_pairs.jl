include("brms_whmc.jl")
using Printf

function extreme_indices(controls)
    selected=[argmin(controls)]
    remaining=setdiff(eachindex(controls),selected)
    push!(selected,remaining[argmin(abs.(controls[remaining].-.5))])
    remaining=setdiff(eachindex(controls),selected)
    push!(selected,remaining[argmax(controls[remaining])])
end

function total_pairs(d,m,input,out)
    fit=deserialize(joinpath(input,"total_posthoc_position.jls"))
    selected=deserialize(joinpath(input,"selected_position.jls"))
    chosen=extreme_indices(selected.centeredness);rows=NamedTuple[];labels=NamedTuple[];error=0.
    location=m.block.A*m.block.location
    for (panel,k) in enumerate(chosen)
        i=selected.indices[k];j,margin=Tuple(only(findall(==(i),m.coords.totals)))
        c=selected.centeredness[k];scale=m.coords.scales[margin]
        title=@sprintf("%d: region %d %s, c=%.2f",panel,j,margin==1 ? "intercept" : "slope",c)
        push!(labels,(;panel=title,region=j,margin,centeredness=c,parameter=fit.names[i]))
        for (column,weight) in (("1. CP",1.),("2. NCP",0.),("3. ACP (position)",c)), s in axes(fit.positions,2)
            ell=fit.positions[scale,s];value=fit.positions[i,s]
            coordinate=weight*location[margin]+(value-location[margin])*exp((weight-1)*ell)
            weight==c && column=="3. ACP (position)" &&
                (error=max(error,abs(coordinate-fit.source_positions[i,s])))
            push!(rows,(;panel=title,column,draw=s,log_group_sd=ell,coordinate))
        end
    end
    @assert error<1e-8 && length(rows)==18000
    write_air_tsv(joinpath(out,"total_pairs.tsv"),rows)
    write_air_tsv(joinpath(out,"total_pairs_selection.tsv"),labels)
    write_air_tsv(joinpath(out,"total_pairs_audit.tsv"),[(;draws=2000,source_coordinate_error=error)])
end

function s2z_pairs(d,m,input,native,out)
    fit=deserialize(joinpath(input,"s2z_auto.jls"))
    target,data,_=target_model(d,"s2z_auto",native);l=s2z_layout(d,target,"s2z_auto",data)
    chosen=extreme_indices(vec(l.rho));rows=NamedTuple[];labels=NamedTuple[];error=0.
    for (panel,k) in enumerate(chosen)
        j,margin=Tuple(CartesianIndices(l.rho)[k]);rho=l.rho[j,margin]
        title=@sprintf("%d: region %d %s, rho=%.2f",panel,j,margin==1 ? "intercept" : "slope",rho)
        push!(labels,(;panel=title,region=j,margin,centeredness=rho))
        for s in axes(fit.positions,2)
            x=BridgeStan.param_constrain(target,collect(fit.positions[:,s]))
            tau=x[l.tau[margin]];source=l.Q*x[l.z[:,margin]]
            scale=1 .-l.rho[:,margin].+l.rho[:,margin].*tau
            w=tau.*source./scale;r=w.-mean(w)
            shift=-sum(r.*scale)/sum(scale);reconstructed=(r.+shift).*scale./tau
            error=max(error,maximum(abs,reconstructed.-source))
            for (column,coordinate) in (("1. CP",r[j]),("2. NCP",r[j]/tau),("3. ACP (brms auto)",source[j]))
                push!(rows,(;panel=title,column,draw=s,log_group_sd=log(tau),coordinate))
            end
        end
    end
    @assert error<1e-8 && length(rows)==18000
    write_air_tsv(joinpath(out,"s2z_pairs.tsv"),rows)
    write_air_tsv(joinpath(out,"s2z_pairs_selection.tsv"),labels)
    write_air_tsv(joinpath(out,"s2z_pairs_audit.tsv"),[(;draws=2000,source_coordinate_error=error)])
end

function prepare(grouping,hierarchy,totals,brms,native,out)
    mkpath(out);d=AIRTotals.load_data(grouping,hierarchy)
    m=AIRTotals.model(d,joinpath(totals,"model"))
    total_pairs(d,m,totals,out);s2z_pairs(d,m,brms,native,out)
    println("AIR_PAIRS_COMPLETE ",hierarchy)
end
if abspath(PROGRAM_FILE)==@__FILE__
    prepare(ARGS...)
end
