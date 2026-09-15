function helmert(J)
    Q=zeros(J,J-1)
    for k in 1:J-1
        Q[1:k,k].=inv(sqrt(k*(k+1)));Q[k+1,k]=-k/sqrt(k*(k+1))
    end
    Q
end

function s2z_layout(d,target,label,data)
    names=BridgeStan.param_names(target);idx=Dict(n=>i for (i,n) in enumerate(names))
    rho=if label=="s2z_auto"
        permutedims(reduce(hcat,data["rho_s2z_1"]))
    else
        fill(label=="s2z_cp" ? 1. : 0.,d.J,d.K)
    end
    theta=d.K==1 ? [idx["theta_s2z_active.1"]] : [idx["theta_s2z.$k"] for k in 1:d.K]
    (;names,theta,rho,slope=d.K==1 ? idx["fixed_s2z.1"] : nothing,
        tau=[idx["sd_1.$k"] for k in 1:d.K],sigma=idx["sigma"],u=idx["udf_b_s2z_1"],
        z=reshape([idx["z_s2z_1.$k"] for k in 1:d.K*(d.J-1)],d.J-1,d.K),Q=helmert(d.J))
end

function s2z_to_total(d,m,l,x)
    tau=x[l.tau];contrasts=zeros(d.J,d.K);jac=d.K*.5log(d.J)
    for k in 1:d.K
        scale=1 .-l.rho[:,k].+l.rho[:,k].*tau[k]
        w=tau[k].*(l.Q*x[l.z[:,k]])./scale
        contrasts[:,k]=w.-mean(w)
        jac+=(d.J-1)*log(tau[k])-sum(log.(scale))+log(mean(scale))
    end
    theta=x[l.theta]
    location=d.K==1 ? theta : [theta[1]-d.xbar*theta[2],theta[2]]
    totals=contrasts.+location'
    q=zeros(length(m.names));c=m.coords
    q[vec(c.totals)]=vec(totals);q[c.scales]=log.(tau);q[c.mixture].=-log(3.)-log(x[l.u])
    q[m.sigma]=log(x[l.sigma]);d.K==1 && (q[m.slope]=x[l.slope])
    q,jac
end

function s2z_from_total(d,m,l,q)
    c=m.coords;T=q[c.totals];tau=exp.(q[c.scales]);average=vec(mean(T;dims=1))
    x=zeros(length(l.names));x[l.tau]=tau;x[l.sigma]=exp(q[m.sigma]);x[l.u]=exp(-only(q[c.mixture]))/3
    x[l.theta]=d.K==1 ? average : [average[1]+d.xbar*average[2],average[2]]
    d.K==1 && (x[l.slope]=q[m.slope])
    for k in 1:d.K
        r=T[:,k].-average[k];scale=1 .-l.rho[:,k].+l.rho[:,k].*tau[k]
        shift=-sum(r.*scale)/sum(scale)
        x[l.z[:,k]]=l.Q'*((r.+shift).*scale./tau[k])
    end
    x
end

function audit_s2z(d,m,target,l,out,label)
    rows=NamedTuple[]
    reference(q)=begin
        physical,jac=s2z_to_total(d,m,l,BridgeStan.param_constrain(target,collect(q)))
        AIRTotals.total_reference(d,m,physical)+jac
    end
    for trial in 1:4
        physical=AIRTotals.total_initial(d,m).+.02randn(Xoshiro(700+trial),length(m.names))
        x=s2z_from_total(d,m,l,physical);q=BridgeStan.param_unconstrain(target,x)
        @assert first(s2z_to_total(d,m,l,x))≈physical
        lp,g=BridgeStan.log_density_gradient(target,q;propto=false)
        error=abs(lp-reference(q));gerror=maximum(abs.(g-AIRTotals.finite_gradient(reference,q))./max.(1.,abs.(g)))
        @assert error<1e-7 && gerror<3e-5
        gqnames=BridgeStan.param_names(target;include_tp=true,include_gq=true)
        values=BridgeStan.param_constrain(target,q;include_tp=true,include_gq=true,
            rng=BridgeStan.StanRNG(target,trial))
        idx=Dict(n=>i for (i,n) in enumerate(gqnames))
        beta=values[[idx["Intercept"],idx["b.1"]]]
        recovered=reshape(beta[1].+values[[idx["r_1_1.$j"] for j in 1:d.J]],d.J,1)
        if d.K==2
            recovered[:,1].-=d.xbar*beta[2]
            recovered=hcat(recovered,beta[2].+values[[idx["r_1_2.$j"] for j in 1:d.J]])
        end
        recovery_error=maximum(abs,recovered.-physical[m.coords.totals])
        @assert recovery_error<1e-9
        push!(rows,(;trial,density_error=error,gradient_error=gerror,recovery_error))
    end
    write_air_tsv(joinpath(out,label*"-audit.tsv"),rows)
    println("AIR_S2Z_AUDIT_PASS ",label);flush(stdout)
end
