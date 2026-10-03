include(joinpath(@__DIR__, "rk_consumer_support.jl"))
include(joinpath(@__DIR__, "rk_structured_fixture.jl"))
using LinearAlgebra

@stestset "downstream prepared structured block primal density Stan and source" begin
    data=(;x=[-.5,0.,.5,1.],g=[1,1,2,2],y=[-1.,.1,1.,.4])
    before=deepcopy(data)
    model=@brm data begin
        mu ~ 1 + group_line(x;group=g)
        y ~ Normal(mu,1)
    end
    backend,problem=consumer_problem(model)
    names=coordinate_names(backend.model.layout)
    index(n)=something(findfirst(==(Symbol(n)),names))
    a=index("mu_Intercept")
    tau=[index("b_line_g_sd.$k") for k in 1:2]
    z=[index("b_line_g_z.$g.$k") for g in 1:2,k in 1:2]
    l=only(setdiff(eachindex(names),[a;tau;vec(z)]))
    @test length(names)==8
    oracle(u)=begin
        scales=exp.(u[tau]); corr=tanh(u[l])
        L=[1. 0.; corr sqrt(1-corr^2)]
        block=u[z]*transpose(Diagonal(scales)*L)
        mu=[u[a]+block[data.g[i],1]+block[data.g[i],2]*data.x[i] for i in eachindex(data.y)]
        sum(logpdf.(Normal.(mu,1),data.y)) + logpdf(Normal(),u[a]) +
            sum(logpdf.(Normal(),u[z])) +
            sum(logpdf.(truncated(Normal(),0,Inf),scales)) + sum(u[tau]) -
            log(2) + log1p(-corr^2)
    end
    for u in (zeros(8),fill(.13,8),collect(range(-.2,.3;length=8)))
        @test LogDensityProblems.logdensity(problem,u) ≈ oracle(u) atol=2e-11 rtol=2e-11
    end
    @test isequal(data,before)
    stan=consumer_stan(model,"downstream-group-line")
    mapping=[:mu_Intercept=>"pop_mu_beta_pop.1",
        Symbol("b_line_g_sd.1")=>"b_line_g_tau.1",
        Symbol("b_line_g_sd.2")=>"b_line_g_tau.2",
        names[l]=>"b_line_g_L.1",
        Symbol("b_line_g_z.1.1")=>"b_line_g_z_flat.1",
        Symbol("b_line_g_z.1.2")=>"b_line_g_z_flat.2",
        Symbol("b_line_g_z.2.1")=>"b_line_g_z_flat.3",
        Symbol("b_line_g_z.2.2")=>"b_line_g_z_flat.4"]
    # USER 0m1j3iz chose normalized positive laws. Historical constrained
    # Stan Normal kernels omit the two half-Normal log(2) constants.
    for u in (zeros(8),collect(range(-.2,.3;length=8)))
        permutation=BRM.resolve_sb_map(mapping,names,BridgeStan.param_unc_names(stan.model);
            case_id="downstream-group-line")
        su=BRM.apply_sb_map(u,permutation); sg=similar(su)
        sv,_=BridgeStan.log_density_gradient!(stan.model,su,sg;propto=false,jacobian=true)
        rv=LogDensityProblems.logdensity(problem,u)
        @test rv ≈ sv+2log(2) atol=2e-11 rtol=2e-11
        h=1e-5
        independent_gradient=map(eachindex(u)) do j
            plus,minus=copy(u),copy(u);plus[j]+=h;minus[j]-=h
            (oracle(plus)-oracle(minus))/(2h)
        end
        @test independent_gradient ≈ BRM.unmap_sb_grad(sg,permutation) atol=2e-8 rtol=2e-8
    end
end

@stestset "downstream original native hook ordinary Enzyme reverse acceptance" begin
    # Strict delivery gate: the consumer's exact authored hook is preserved.
    # Generic Enzyme activity support is tracked by generic-array-re-b298e503.
    data=(;x=[-.5,0.,.5,1.],g=[1,1,2,2],y=[-1.,.1,1.,.4])
    model=@brm data begin
        mu ~ 1 + group_line(x;group=g)
        y ~ Normal(mu,1)
    end
    backend,problem=consumer_problem(model)
    u=zeros(backend.model.layout.total);before=copy(u)
    value,gradient=LogDensityProblems.logdensity_and_gradient(problem,u)
    @test isfinite(value)
    @test all(isfinite,gradient)
    @test isequal(u,before)
end

function independent_hsgp_basis(x,k,c)
    center=sum(x)/length(x); L=c*maximum(abs.(x.-center))
    frequencies=[(j*pi/(2L))^2 for j in 1:k]
    phi=[sin(sqrt(frequencies[j])*(x[i]-center+L))/sqrt(L)
        for i in eachindex(x),j in 1:k]
    floor=k==1 ? 0. : 4L/pi*sqrt(log(100)/(k^2-1))
    phi,frequencies,floor
end

const grouped_hsgp_default_builder=@brm begin
    mu ~ 1 + hsgp(x;k=3,by=g)
    y ~ Normal(mu,1)
end
const grouped_hsgp_hyper_builder=@brm begin
    mu ~ 1 + hsgp(x;k=3,by=g)
    log(length_scale(hsgp(x))) ~ 1 + (1|g)
    log(sd(hsgp(x))) ~ 1 + (1|g)
    y ~ Normal(mu,1)
end
const grouped_hsgp_explicit_builder=@brm begin
    mu ~ 1 + hsgp(x;k=3,by=g)
    length_scale(:,hsgp(x)) ~ Exponential(.7)
    sd(:,hsgp(x)) ~ Exponential(1.3)
    y ~ Normal(mu,1)
end

@stestset "grouped HSGP shared and authored hyper predictors native acceptance" begin
    for (label,builder) in (("default",grouped_hsgp_default_builder),
            ("hyper",grouped_hsgp_hyper_builder),("explicit",grouped_hsgp_explicit_builder))
        data=(;x=[-.5,0.,.5,1.],g=[1,1,2,2],y=[-1.,.1,1.,.4])
        before=deepcopy(data)
        model=builder(data)
        backend,problem=consumer_problem(model)
        names=coordinate_names(backend.model.layout)
        index(n)=something(findfirst(==(Symbol(n)),names))
        a=index("mu_Intercept")
        weights=[index("hsgp_x_z.$g.$k") for g in 1:2,k in 1:3]
        phi,frequencies,floor=independent_hsgp_basis(data.x,3,1.5)
        # Prepared geometry supplies only its immutable fitted c, never the
        # statistical density or a backend result used as a reference.
        term=only(filter(t->t.kind===:hsgp,only(backend.plan.predictors).terms))
        phi,frequencies,floor=independent_hsgp_basis(data.x,3,term.options.c)
        stan=consumer_stan(model,"grouped-hsgp-"*label)
        mapping=[:mu_Intercept=>"pop_mu_beta_pop.1"]
        for g in 1:2,k in 1:3
            push!(mapping,names[weights[g,k]]=>"zflat_hsgpw_x_g.$((g-1)*3+k)")
        end
        for (stem,stan_stem) in (("rho","rho_iso"),("sigma","sigma"))
            if label=="hyper"
                push!(mapping,Symbol("hsgp_x_$(stem)_Intercept")=>"hsgp_x_by_g_beta0_$stem")
                push!(mapping,Symbol("hsgp_x_$(stem)_sd")=>"hsgp_x_by_g_sd_$stem")
                for g in 1:2
                    push!(mapping,Symbol("hsgp_x_$(stem)_z.$g")=>"hsgp_x_by_g_z_$stem.$g")
                end
            else
                push!(mapping,Symbol("hsgp_x_$stem")=>"hsgp_x_by_g_$stan_stem")
            end
        end
        oracle(u)=begin
            prior=logpdf(Normal(),u[a])+sum(logpdf.(Normal(),u[weights]))
            hypers=map(("rho","sigma")) do stem
                if label=="hyper"
                    beta=index("hsgp_x_$(stem)_Intercept")
                    sd=index("hsgp_x_$(stem)_sd")
                    zs=[index("hsgp_x_$(stem)_z.$g") for g in 1:2]
                    prior+=logpdf(Normal(),u[beta])+sum(logpdf.(Normal(),u[zs]))+
                        logpdf(truncated(Normal(),0,Inf),exp(u[sd]))+u[sd]
                    value=exp.(u[beta].+exp(u[sd]).*u[zs])
                    stem=="rho" ? max.(value,floor) : value
                else
                    q=index("hsgp_x_$(stem)")
                    bound=label=="default" && stem=="rho" ? floor : 0.
                    value=bound+exp(u[q])
                    law=label=="explicit" ? Exponential(stem=="rho" ? .7 : 1.3) :
                        (stem=="rho" ? truncated(LogNormal(),floor,Inf) : LogNormal())
                    prior+=logpdf(law,value)+u[q]
                    fill(value,2)
                end
            end
            rho,sigma=hypers
            mu=[u[a]+sum(phi[i,k]*u[weights[data.g[i],k]]*
                sigma[data.g[i]]*sqrt(rho[data.g[i]]*sqrt(2pi))*
                exp(-rho[data.g[i]]^2*frequencies[k]/4) for k in 1:3)
                for i in eachindex(data.y)]
            prior+sum(logpdf.(Normal.(mu,1),data.y))
        end
        for u in (zeros(length(names)),fill(.13,length(names)),
                collect(range(-.2,.3;length=length(names))))
            check_consumer_point(problem,u,oracle)
            # Historical Stan lower bounds omit conditional normalizers.
            offset=label=="hyper" ? 2log(2) : label=="default" ?
                -logccdf(LogNormal(),floor) : 0.
            check_consumer_stan(problem,stan,mapping,backend,u;density_offset=offset)
        end
        @test isequal(data,before)
    end
end
