function helmert(j)
    q=zeros(j,j-1)
    for k in 1:j-1
        q[1:k,k].=inv(sqrt(k*(k+1)))
        q[k+1,k]=-k/sqrt(k*(k+1))
    end
    q
end

function s2z_layout(model,label,data)
    names=BridgeStan.param_names(model);idx=Dict(v=>i for (i,v) in enumerate(names))
    rho=label=="s2z_auto" ?
        hcat(permutedims(reduce(hcat,data["rho_s2z_1"])),Float64.(only.(data["rho_s2z_2"]))) :
        fill(label=="s2z_cp" ? 1. : 0.,J,3)
    @assert size(rho)==(J,3)
    (;names,theta=[idx["theta_s2z.1"],idx["theta_s2z.2"],idx["theta_s2z_sigma.1"]],
      tau=[idx["sd_1.1"],idx["sd_1.2"],idx["sd_2.1"]],
      z=hcat([idx["z_s2z_1.$k"] for k in 1:J-1],[idx["z_s2z_1.$k"] for k in J:2(J-1)],
             [idx["z_s2z_2.$k"] for k in 1:J-1]),
      u=[idx["udf_b_s2z_1"],idx["udf_b_s2z_sigma_1"]],rho,Q=helmert(J))
end

function s2z_to_total(x,l,tm)
    tau=x[l.tau];contrasts=zeros(J,3)
    jac=1.5log(J)
    for k in 1:3
        scale=1 .-l.rho[:,k].+l.rho[:,k].*tau[k]
        w=tau[k].*(l.Q*x[l.z[:,k]])./scale
        contrasts[:,k]=w.-mean(w)
        jac+=(J-1)*log(tau[k])-sum(log.(scale))+log(mean(scale))
    end
    theta=x[l.theta]
    totals=contrasts.+[theta[1]-XBAR*theta[2],theta[2],theta[3]]'
    q=zeros(length(tm.names));m=tm.coords[:mu];s=tm.coords[:logsigma]
    q[vec(m.totals)]=vec(totals[:,1:2]);q[vec(s.totals)]=totals[:,3]
    q[vcat(m.scales,s.scales)]=log.(tau)
    q[vcat(m.mixture,s.mixture)]=-log(3.).-log.(x[l.u])
    q,jac
end

function s2z_from_total(q,l,tm)
    m=tm.coords[:mu];s=tm.coords[:logsigma]
    totals=hcat(q[m.totals],q[s.totals]);tau=exp.(q[vcat(m.scales,s.scales)])
    x=zeros(length(l.names));x[l.tau]=tau
    x[l.theta]=[mean(totals[:,1])+XBAR*mean(totals[:,2]),mean(totals[:,2]),mean(totals[:,3])]
    x[l.u]=exp.(-q[vcat(m.mixture,s.mixture)])./3
    for k in 1:3
        r=totals[:,k].-mean(totals[:,k])
        scale=1 .-l.rho[:,k].+l.rho[:,k].*tau[k]
        shift=-sum(r.*scale)/sum(scale)
        x[l.z[:,k]]=l.Q'*((r.+shift).*scale./tau[k])
    end
    x
end

function audit_s2z(model,l,tm,label,out)
    rng=Xoshiro(712);results=NamedTuple[]
    reference(q)=begin
        physical,jac=s2z_to_total(BridgeStan.param_constrain(model,Vector{Float64}(q)),l,tm)
        total_reference(tm,physical)+jac
    end
    for trial in 1:4
        physical=total_initial(tm)
        physical .+=0.01max.(1.,abs.(physical)).*randn(rng,length(physical))
        x=s2z_from_total(physical,l,tm)
        q=BridgeStan.param_unconstrain(model,x)
        @assert first(s2z_to_total(x,l,tm))≈physical
        lp,g=BridgeStan.log_density_gradient(model,q;propto=false)
        err=abs(lp-reference(q));gerr=maximum(abs.(g-finite_gradient(reference,q))./max.(1.,abs.(g)))
        @assert err<2e-7 "S2Z density mismatch: $err"
        @assert gerr<5e-5 "S2Z gradient mismatch: $gerr"
        names=BridgeStan.param_names(model;include_tp=true,include_gq=true)
        v=BridgeStan.param_constrain(model,q;include_tp=true,include_gq=true,rng=BridgeStan.StanRNG(model,trial))
        idx=Dict(n=>i for (i,n) in enumerate(names))
        recovered=hcat(v[idx["Intercept"]]-XBAR*v[idx["b.1"]].+v[[idx["r_1_1.$j"] for j in 1:J]],
            v[idx["b.1"]].+v[[idx["r_1_2.$j"] for j in 1:J]],
            v[idx["Intercept_sigma"]].+v[[idx["r_2_sigma_1.$j"] for j in 1:J]])
        @assert recovered≈hcat(physical[tm.coords[:mu].totals],physical[tm.coords[:logsigma].totals])
        push!(results,(;trial,density_error=err,gradient_error=gerr))
    end
    write_tsv(joinpath(out,label*"-audit.tsv"),results)
    println("S2Z_AUDIT_PASS ",label);flush(stdout)
end
