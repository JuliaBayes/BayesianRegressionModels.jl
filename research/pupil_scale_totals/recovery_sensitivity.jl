using Serialization, Statistics, LinearAlgebra, Random, JSON
include("diagnostics.jl")

function sensitivity(root,out)
    source_names=deserialize(joinpath(root,"pupil4-totals-v1","total_ncp.jls")).names
    index=Dict(name=>i for (i,name) in enumerate(source_names))
    mu=[index["total_mu.$j.$k"] for j in 1:20,k in 1:2]
    logsigma=[index["total_logsigma.$j.1"] for j in 1:20]
    scales=[index["total_scale_mu_tau.1"],index["total_scale_mu_tau.2"],index["total_scale_logsigma_tau.1"]]
    mixtures=[index["total_mixture_mu.1"],index["total_mixture_logsigma.1"]]
    xbar=mean(Float64.(JSON.parsefile(joinpath(@__DIR__,"reference","ordinary_ncp.json"))["Z_1_2"]))
    A=[1. -xbar;0. 1.];rows=NamedTuple[]
    for label in ("total_ncp","total_cp","total_posthoc_position","total_posthoc_gradient",
                  "total_online_position","total_online_gradient","s2z_cp_native","s2z_ncp_native",
                  "s2z_auto_native","s2z_cp_whmc","s2z_ncp_whmc","s2z_auto_whmc")
        native=endswith(label,"_native");total=startswith(label,"total_")
        dir=joinpath(root,total ? "pupil4-totals-v1" : native ? "pupil4-summary-v1" : "pupil4-brms-whmc-v1")
        name=endswith(label,"_whmc") ? chop(label;tail=5) : label
        positions=if native
            deserialize(joinpath(dir,label*"-positions.jls"))
        else
            fit=deserialize(joinpath(dir,name*".jls"));total ? fit.positions : fit.total_positions
        end
        original=deserialize(joinpath(dir,name*"-qois.jls"))
        n=size(positions,2);means=zeros(n,3);factors=Matrix{Float64}[];residual_sd=zeros(n)
        for s in 1:n
            q=positions[:,s];tau=exp.(q[scales]);lambda=exp.(q[mixtures])
            D=Diagonal(inv.(tau[1:2].^2));precision=[lambda[1]/2026.1^2,0.]
            Q=Symmetric(Diagonal(precision)+20A'*D*A)
            rhs=precision.*[5651.9,0.]+A'*D*vec(sum(q[mu];dims=1))
            means[s,1:2]=Q\rhs;push!(factors,Matrix(cholesky(Symmetric(inv(Q))).L))
            p=lambda[2]/2.5^2+20/tau[3]^2
            means[s,3]=sum(q[logsigma])/tau[3]^2/p;residual_sd[s]=inv(sqrt(p))
        end
        for seed in 101:110
            rng=Xoshiro(seed);q=copy(original)
            for s in 1:n
                q[s,1:2]=means[s,1:2]+factors[s]*randn(rng,2)
                q[s,3]=means[s,3]+residual_sd[s]*randn(rng)
            end
            @assert q[:,4:end]==original[:,4:end]
            ess=vec(MCMCDiagnosticTools.ess(reshape(q,n,1,66);kind=:bulk))
            push!(rows,(;arm=label,recovery_seed=seed,min_bulk_ess=minimum(ess),limiter=qoi_names()[argmin(ess)]))
        end
    end
    write_tsv(joinpath(out,"recovery_seed_sensitivity.tsv"),rows)
    println("RECOVERY_SENSITIVITY_COMPLETE rows=",length(rows))
end
sensitivity(ARGS...)
