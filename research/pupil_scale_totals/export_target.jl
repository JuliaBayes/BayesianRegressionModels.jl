include("common.jl")
out=only(ARGS);mkpath(out)
tm=total_model(out)
data_path=joinpath(out,"automatic_totals.json")
data=StanBlocks.stan_data(tm.sb.model)
# JSON.jl serializes Julia matrices by columns; Stan JSON expects rows.
stan_json=Dict(k=>(v isa AbstractMatrix ? [collect(row) for row in eachrow(v)] : v) for (k,v) in data)
write_json(data_path,stan_json)
exported=BridgeStan.StanModel(joinpath(out,"automatic_totals.stan"),data_path;warn=false)
q=total_initial(tm)
lp,g=BridgeStan.log_density_gradient(tm.model,q;propto=false)
elp,eg=BridgeStan.log_density_gradient(exported,q;propto=false)
println("EXPORTED_DENSITY_ERROR ",abs(lp-elp)," GRADIENT_ERROR ",maximum(abs.(g-eg)),
    " RELATIVE_GRADIENT_ERROR ",maximum(abs.(g-eg)./max.(1.,abs.(g))));flush(stdout)
@assert isapprox(lp,elp;rtol=1e-13,atol=1e-9) && isapprox(g,eg;rtol=1e-12,atol=1e-9)
write_json(joinpath(out,"export-audit.json"),Dict("status"=>"passed",
    "density_error"=>abs(lp-elp),"gradient_error"=>maximum(abs.(g-eg))))
println("TOTAL_TARGET_EXPORT_COMPLETE")
