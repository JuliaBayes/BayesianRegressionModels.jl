include("replay_failure.jl")
source=read(joinpath(pkgdir(WarmupHMC),"src","Reparametrizations.jl"),String)
first_index=findfirst("function _jointly_transport_halo!",source).start
last_index=findnext("# A centering switch",source,first_index).start-1
body=replace(source[first_index:last_index],"_jointly_transport_halo!"=>"_diagnostic_original_transport!")
Base.include_string(WarmupHMC,body,"unchanged-transport-capture")
function WarmupHMC._jointly_transport_halo!(lpdf,old_ir,old_position,old_gradient,position,gradient)
    WarmupHMC._diagnostic_original_transport!(lpdf,old_ir,old_position,old_gradient,position,gradient)
    if !all(isfinite,gradient) || !all(isfinite,position)
        println("NONFINITE_TRANSPORT positions=",count(!isfinite,position)," gradients=",count(!isfinite,gradient),
            " old_positions=",count(!isfinite,old_position)," old_gradients=",count(!isfinite,old_gradient))
        serialize(joinpath(OUTPUT,REQUEST,"transport-failure.jls"),
            (;old_ir,new_ir=WarmupHMC.reparametrizer(lpdf),old_position,old_gradient,position,gradient))
    end
    gradient
end
replay_failure()
