include("run.jl")
cp=deserialize(joinpath(OUTPUT,REQUEST,"checkpoints","cp_latest.jls"))
println("CHECKPOINT_FIELDS ",propertynames(cp))
for key in propertynames(cp)
    x=getproperty(cp,key)
    if x isa Array{<:Number}
        println(key," size=",size(x)," nonfinite=",count(!isfinite,x))
    elseif x isa Number
        println(key,"=",x)
    elseif x isa NamedTuple
        println(key," fields=",keys(x))
    else
        println(key," type=",typeof(x))
    end
end
println("ACTIVE_TRANSFORMATION ",cp.active_transformation)
function inspect_finite(x,path)
    if x isa Array{<:Number}
        println(path," size=",size(x)," nonfinite=",findall(!isfinite,x))
    elseif x isa Number
        isfinite(x) || println(path,"=",x)
    elseif x isa NamedTuple || isstructtype(typeof(x))
        for key in propertynames(x)
            inspect_finite(getproperty(x,key),path*"."*string(key))
        end
    end
end
inspect_finite(cp.scale_options,"scale_options")
inspect_finite(cp.linear_recorder,"linear_recorder")
