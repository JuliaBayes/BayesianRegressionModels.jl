using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using Distributions, LinearAlgebra, Statistics, Random, JSON, Serialization
using WarmupHMC, Enzyme, MCMCDiagnosticTools
using DifferentiationInterface: AutoEnzyme
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__,"..","pupil_total_effects","stan_target.jl"))
const REFERENCE = joinpath(@__DIR__,"reference")
const REF = JSON.parsefile(joinpath(REFERENCE,"ordinary_ncp.json"))
const Y = Float64.(REF["Y"])
const X = Float64.(REF["Z_1_2"])
const GROUP = Int.(REF["J_1"])
const J = Int(REF["N_1"])
const XBAR = mean(X)
const DATA = (;p_size=Y, load=X, subj=GROUP)
const BUILDER = @brm begin
    mu ~ 1 + center(load) + (1 | mean_intercept | subj) + (0 + load | mean_slope | subj)
    logsigma ~ 1 + (1 | scale_intercept | subj)
    effect(mu, Intercept) ~ LocationScale(5651.9,2026.1,TDist(3))
    effect(mu, center_load) ~ Flat()
    effect(logsigma, Intercept) ~ LocationScale(0.,2.5,TDist(3))
    sd(:, mean_intercept) ~ LocationScale(0.,2026.1,TDist(3))
    sd(:, mean_slope) ~ LocationScale(0.,2026.1,TDist(3))
    sd(:, scale_intercept) ~ LocationScale(0.,2026.1,TDist(3))
    p_size ~ Normal(mu,exp(logsigma))
end

function total_model(out)
    mkpath(out)
    sb = SBBRMI(BUILDER(DATA);mod=@__MODULE__)
    blocks = Dict(b.predictor=>b for b in total_effect_blocks(sb))
    @assert Set(keys(blocks)) == Set([:mu,:logsigma])
    p = Base.invokelatest(StanBlocks.stan_instantiate,sb.model;path=joinpath(out,"automatic_totals.stan"))
    names = BridgeStan.param_unc_names(p.model)
    coords = Dict(k=>BRM._total_coordinates(sb,b,names) for (k,b) in blocks)
    @assert length(names)==65
    @assert blocks[:mu].A ≈ [1. -XBAR;0. 1.]
    @assert blocks[:logsigma].A == ones(1,1)
    (;sb,model=p.model,names,blocks,coords)
end

function stan_model(label;source_dir=REFERENCE)
    source=joinpath(source_dir,label*".stan")
    data=joinpath(source_dir,label*".json")
    BridgeStan.StanModel(source,data;warn=false)
end

function ols_initial()
    totals = zeros(J,3)
    for j in 1:J
        rows=findall(==(j),GROUP)
        design=hcat(ones(length(rows)),X[rows])
        beta=design\Y[rows]
        totals[j,1:2]=beta
        totals[j,3]=log(sqrt(mean(abs2,Y[rows]-design*beta)))
    end
    tau=vec(std(totals;dims=1))
    beta=[mean(totals[:,1])+XBAR*mean(totals[:,2]),mean(totals[:,2]),mean(totals[:,3])]
    # Zero deviations give every arm the same physical starting point even
    # when brms resolves automatic centering weights after its pilot.
    totals .= mean(totals;dims=1)
    (;totals,tau,beta)
end

function total_initial(tm, init=ols_initial())
    q=zeros(length(tm.names))
    for (key, cols) in ((:mu,1:2),(:logsigma,3:3))
        c=tm.coords[key]
        q[vec(c.totals)]=vec(init.totals[:,cols])
        q[c.scales]=log.(init.tau[cols])
        q[c.mixture].=0. # lambda=1 for each Student-t population intercept
    end
    q
end

function ordinary_init(init=ols_initial())
    b=init.beta;t=init.totals;tau=init.tau
    Dict("b"=>[b[2]],"Intercept"=>b[1],"Intercept_sigma"=>b[3],
        "sd_1"=>tau[1:2],"sd_2"=>tau[3:3],
        "z_1"=>[zeros(J),zeros(J)],"z_2"=>[zeros(J)])
end

function likelihood(totals)
    sum(eachindex(Y)) do n
        j=GROUP[n]
        logpdf(Normal(totals[j,1]+X[n]*totals[j,2],exp(totals[j,3])),Y[n])
    end
end

scale_prior(tau)=sum(logpdf(LocationScale(0.,2026.1,TDist(3)),t)+log(2)+log(t) for t in tau)

# Independent density reference: evaluate the full Gaussian joint at an
# arbitrary population coefficient and divide by its normalized conditional.
# This uses dense Gaussian algebra, independent of BRM's emitted functions.
function integrated_gaussian(totals,tau,A,location,precision)
    ng,k=size(totals);p=length(location)
    D=Diagonal(inv.(tau.^2))
    Q=Symmetric(Diagonal(precision)+ng*A'*D*A)
    rhs=precision.*location+A'*D*vec(sum(totals;dims=1))
    conditional_mean=Q\rhs
    beta=conditional_mean .+ sqrt.(diag(inv(Q))).*(0.1 .*collect(1:p))
    joint=sum(logpdf(Normal((A*beta)[c],tau[c]),totals[j,c]) for j in 1:ng,c in 1:k)
    joint+=sum(precision[a]>0 ? logpdf(Normal(location[a],inv(sqrt(precision[a]))),beta[a]) : 0. for a in 1:p)
    joint-logpdf(MvNormal(conditional_mean,Symmetric(inv(Q))),beta)
end

function total_reference(tm,q)
    c=tm.coords[:mu];s=tm.coords[:logsigma]
    T=hcat(q[c.totals],q[s.totals]);tau=exp.(vcat(q[c.scales],q[s.scales]))
    lm=only(q[c.mixture]);ls=only(q[s.mixture])
    likelihood(T)+scale_prior(tau)+
        integrated_gaussian(T[:,1:2],tau[1:2],[1. -XBAR;0. 1.],[5651.9,0.],[exp(lm)/2026.1^2,0.])+
        integrated_gaussian(T[:,3:3],tau[3:3],ones(1,1),[0.],[exp(ls)/2.5^2])+
        logpdf(Gamma(1.5,1/1.5),exp(lm))+lm+logpdf(Gamma(1.5,1/1.5),exp(ls))+ls
end

function ordinary_physical(model,q)
    # Stan's array-of-vector unconstrained name order is not its input-vector
    # order. Let Stan perform the conversion, then use constrained names.
    n=BridgeStan.param_names(model;include_tp=true);idx=Dict(v=>i for (i,v) in enumerate(n))
    x=BridgeStan.param_constrain(model,Vector{Float64}(q);include_tp=true)
    b=[x[idx["Intercept"]],x[idx["b.1"]],x[idx["Intercept_sigma"]]]
    tau=x[[idx["sd_1.1"],idx["sd_1.2"],idx["sd_2.1"]]]
    deviations=hcat(x[[idx["r_1_1.$j"] for j in 1:J]],x[[idx["r_1_2.$j"] for j in 1:J]],
                    x[[idx["r_2_sigma_1.$j"] for j in 1:J]])
    z=deviations./tau'
    totals=z.*tau' .+ [b[1]-XBAR*b[2],b[2],b[3]]'
    (;b,tau,z,totals)
end

function ordinary_reference(model,q)
    v=ordinary_physical(model,q)
    likelihood(v.totals)+scale_prior(v.tau)+sum(logpdf.(Normal(),v.z))+
        logpdf(LocationScale(5651.9,2026.1,TDist(3)),v.b[1])+
        logpdf(LocationScale(0.,2.5,TDist(3)),v.b[3])
end

function finite_gradient(f,q)
    [begin
        h=1e-5*max(1,abs(q[i])); plus=copy(q);minus=copy(q)
        plus[i]+=h;minus[i]-=h
        (f(plus)-f(minus))/(2h)
    end for i in eachindex(q)]
end

function write_json(path,obj)
    open(path,"w") do io; JSON.print(io,obj,2); end
end

include("diagnostics.jl")

function total_qois(tm,positions;seed=404)
    rows=permutedims(positions)
    rec=recover_population_draws(tm.sb,rows,tm.names;rng=Xoshiro(seed))
    m=tm.coords[:mu];s=tm.coords[:logsigma]
    hcat(rec[:mu].population,rec[:logsigma].population,
         exp.(rows[:,vcat(m.scales,s.scales)]),rows[:,m.totals[:,1]],
         rows[:,m.totals[:,2]],exp.(rows[:,s.totals[:,1]]))
end

function ordinary_qois(model,positions)
    permutedims(hcat([begin
        v=ordinary_physical(model,q)
        vcat(v.b,v.tau,v.totals[:,1],v.totals[:,2],exp.(v.totals[:,3]))
    end for q in eachcol(positions)]...))
end

function online_loss!(kind)
    kind in (:position,:gradient) || error("Unknown loss")
    source=read(joinpath(pkgdir(WarmupHMC),"src","Reparametrizations.jl"),String)
    for (start,finish) in (("reparametrization_loss((;ljac, cov)::OnlineReparametrizationLoss", "scale_estimate(loss::OnlineReparametrizationLoss)"),
                           ("function reparametrization_loss(loss::WeightedReparametrizationLoss","scale_estimate(loss::WeightedReparametrizationLoss)"))
        a=first(findfirst(start,source));b=first(findnext(finish,source,a))-1
        code=source[a:b];@assert occursin("w1=0",code)
        Base.include_string(WarmupHMC,replace(code,"w1=0"=>(kind===:position ? "w1=1" : "w1=0")),"pupil4-loss-selection")
    end
end
