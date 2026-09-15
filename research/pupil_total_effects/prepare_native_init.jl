include("model.jl")
using JSON,Statistics
data = PupilTotalEffects.load_data()
q = PupilTotalEffects.initial_position(data)
A,B = q[5:24],q[25:44]
init = Dict("Intercept"=>mean(A)+data.xbar*mean(B),"b"=>[mean(B)],
    "Intercept_sigma"=>q[3],"b_sigma"=>[q[4]],"sd_1"=>exp.(q[1:2]),
    "z_1"=>[(A.-mean(A))./exp(q[1]),(B.-mean(B))./exp(q[2])])
open(io->JSON.print(io,init),only(ARGS),"w")
