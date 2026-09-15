include("diagnostics.jl")
using DelimitedFiles, Serialization
input,out=ARGS;mkpath(out)
v,h=readdlm(joinpath(input,"sampling.tsv"),'\t',Float64;header=true)
names=String.(vec(h));idx=Dict(n=>i for (i,n) in enumerate(names))
column(n)=v[:,idx[n]]
qois=hcat(column("Intercept"),column("b.1"),column("Intercept_sigma"),
    column("sd_1.1"),column("sd_1.2"),column("sd_2.1"),
    [column("b_Intercept").+column("r_1_1.$j") for j in 1:20]...,
    [column("b.1").+column("r_1_2.$j") for j in 1:20]...,
    [exp.(column("Intercept_sigma").+column("r_2_sigma_1.$j")) for j in 1:20]...)
c,ch=readdlm(joinpath(input,"gradient_counts.tsv"),'\t',Float64;header=true)
costs=Dict(String(k)=>Int(c[1,i]) for (i,k) in enumerate(vec(ch)))
serialize(joinpath(out,"ordinary_ncp_native-qois.jls"),qois)
scientific_diagnostics("ordinary_ncp_native",qois,costs["sampling_gradients"],
    costs["workflow_gradient_calls"],costs["divergences"],out)
