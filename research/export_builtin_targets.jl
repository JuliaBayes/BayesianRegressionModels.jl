using JSON, StanBlocks, BridgeStan

function export_target(tm,q,out)
    data=StanBlocks.stan_data(tm.sb.model)
    converted=Dict(k=>(v isa AbstractMatrix ? [collect(row) for row in eachrow(v)] : v) for (k,v) in data)
    path=joinpath(out,"automatic_totals.json")
    open(io->JSON.print(io,converted,2),path,"w")
    target=BridgeStan.StanModel(joinpath(out,"automatic_totals.stan"),path;warn=false)
    lp,g=BridgeStan.log_density_gradient(tm.model,q;propto=false)
    other,og=BridgeStan.log_density_gradient(target,q;propto=false)
    @assert lp==other && g==og
    open(io->JSON.print(io,Dict("status"=>"passed","density_error"=>abs(lp-other),
        "gradient_error"=>maximum(abs.(g-og))),2),joinpath(out,"export-audit.json"),"w")
    println("BUILTIN_EXPORT_EXACT ",out);flush(stdout)
end

kind,out=ARGS
if kind=="pupil3"
    include("pupil_builtin_totals/common.jl")
    tm=total_model(out);export_target(tm,total_initial(tm),out)
elseif kind=="air"
    include("air_total_effects/model.jl")
    for hierarchy in ("intercept_only","independent")
        dir=joinpath(out,hierarchy);d=AIRTotals.load_data("cluster_region",hierarchy)
        tm=AIRTotals.model(d,dir);export_target(tm,AIRTotals.total_initial(d,tm),dir)
    end
else
    error("Unknown study")
end
