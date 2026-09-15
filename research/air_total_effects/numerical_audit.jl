using Serialization, JSON, LinearAlgebra, Statistics, Test
input,out=ARGS
data=JSON.parsefile(joinpath(@__DIR__,"reference","cluster_region","independent","ordinary_ncp.json"))
xbar=mean(Float64.(getindex.(data["X"],2)))
function precision(q,names,::Type{T}) where T
    index=Dict(n=>i for (i,n) in enumerate(names))
    tau=exp.(T.([q[index["total_scale_mu_tau.$k"]] for k in 1:2]))
    p=exp(T(q[index["total_mixture_mu.1"]]))/T(2.5)^2
    A=T[1 -xbar;0 1];Q=Matrix(Diagonal(T[p,0]))
    for a in 1:2,b in 1:2,c in 1:2
        Q[a,b]+=6*A[c,a]*A[c,b]/tau[c]^2
    end
    Q
end
rows=[]
@testset "AIR extreme trial precision: exact SPD, lost double precision" begin
    for arm in ("total_posthoc_position","total_posthoc_gradient")
        fit=deserialize(joinpath(input,arm*".jls"));q=fit.first_spd_point
        @test !isnothing(q) && all(isfinite,q) && fit.numerical_rejections==1
        Q=precision(q,fit.names,Float64)
        high=precision(q,fit.names,BigFloat)
        @test isposdef(Symmetric(high))
        condition_proxy=Float64(tr(high)^2/det(high))
        @test condition_proxy>inv(eps(Float64))
        push!(rows,Dict("arm"=>arm,"double_precision_determinant"=>det(Q),
            "high_precision_determinant"=>string(det(high)),"condition_proxy"=>condition_proxy,
            "log_residual_sd"=>q[only(findall(==("sigma"),fit.names))],
            "double_precision_isposdef"=>isposdef(Symmetric(Q)),"numerical_rejections"=>fit.numerical_rejections))
    end
end
open(io->JSON.print(io,rows,2),out,"w")
println("AIR_NUMERICAL_WITNESS_VERIFIED ",rows)
