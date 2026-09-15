module AIRTotals
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using Distributions, LinearAlgebra, Statistics, Random, JSON, Serialization
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme
const BRM=BayesianRegressionModels
include(joinpath(@__DIR__,"..","pupil_total_effects","stan_target.jl"))
include("target.jl")

const INTERCEPT = @brm begin
    mu ~ 1 + center(log_sat) + (1 | regional_intercept | region)
    effect(mu, Intercept) ~ LocationScale(2.8,2.5,TDist(3))
    effect(mu, center_log_sat) ~ Flat()
    sd(:, regional_intercept) ~ LocationScale(0.,2.5,TDist(3))
    sigma ~ LocationScale(0.,2.5,TDist(3);lower=0.)
    log_pm25 ~ Normal(mu,sigma)
end
const INDEPENDENT = @brm begin
    mu ~ 1 + center(log_sat) + (1 | regional_intercept | region) + (0 + log_sat | regional_slope | region)
    effect(mu, Intercept) ~ LocationScale(2.8,2.5,TDist(3))
    effect(mu, center_log_sat) ~ Flat()
    sd(:, regional_intercept) ~ LocationScale(0.,2.5,TDist(3))
    sd(:, regional_slope) ~ LocationScale(0.,2.5,TDist(3))
    sigma ~ LocationScale(0.,2.5,TDist(3);lower=0.)
    log_pm25 ~ Normal(mu,sigma)
end

function load_data(grouping,hierarchy)
    hierarchy in ("intercept_only","independent") || error("Unknown hierarchy")
    dir=joinpath(@__DIR__,"reference",grouping,hierarchy)
    reference=JSON.parsefile(joinpath(dir,"ordinary_ncp.json"))
    x=Float64.(getindex.(reference["X"],2));y=Float64.(reference["Y"])
    group=Int.(reference["J_1"]);J=Int(reference["N_1"]);K=Int(reference["M_1"])
    @assert length(y)==6003 && K==(hierarchy=="independent" ? 2 : 1)
    (;dir,x,y,group,J,K,xbar=mean(x),hierarchy,grouping)
end

function model(d,out)
    mkpath(out)
    builder=d.K==1 ? INTERCEPT : INDEPENDENT
    sb=SBBRMI(builder((;log_pm25=d.y,log_sat=d.x,region=d.group));mod=@__MODULE__)
    block=only(total_effect_blocks(sb));@assert block.predictor==:mu
    p=Base.invokelatest(StanBlocks.stan_instantiate,sb.model;path=joinpath(out,"automatic_totals.stan"))
    names=BridgeStan.param_unc_names(p.model)
    coords=BRM._total_coordinates(sb,block,names)
    sigma=only(findall(==("sigma"),names))
    slope=d.K==1 ? only(findall(==("pop_mu_beta_pop.1"),names)) : nothing
    @assert block.A ≈ (d.K==1 ? ones(1,1) : [1. -d.xbar;0. 1.])
    (;sb,block,model=p.model,names,coords,sigma,slope)
end

function initial_values(d)
    design=hcat(ones(length(d.x)),d.x.-d.xbar)
    beta=design\d.y
    effects=zeros(d.J,d.K);residual=zeros(length(d.y))
    for j in 1:d.J
        rows=findall(==(j),d.group)
        if d.K==1
            effects[j,1]=mean(d.y[rows].-beta[2].*(d.x[rows].-d.xbar))
            residual[rows]=d.y[rows].-effects[j,1].-beta[2].*(d.x[rows].-d.xbar)
        else
            regional=hcat(ones(length(rows)),d.x[rows])\d.y[rows]
            effects[j,:]=regional
            residual[rows]=d.y[rows].-regional[1].-regional[2].*d.x[rows]
        end
    end
    tau=max.(vec(std(effects;dims=1)),.05)
    totals=d.K==1 ? fill(beta[1],d.J,1) : repeat([beta[1]-d.xbar*beta[2] beta[2]],d.J,1)
    (;beta,tau,totals,sigma=sqrt(mean(abs2,residual)))
end

function total_initial(d,m)
    init=initial_values(d);q=zeros(length(m.names));c=m.coords
    q[vec(c.totals)]=vec(init.totals);q[c.scales]=log.(init.tau);q[c.mixture].=0
    q[m.sigma]=log(init.sigma)
    d.K==1 && (q[m.slope]=init.beta[2])
    q
end

function ordinary_init(d)
    i=initial_values(d)
    Dict("Intercept"=>i.beta[1],"b"=>[i.beta[2]],"sd_1"=>i.tau,
         "sigma"=>i.sigma,"z_1"=>[zeros(d.J) for _ in 1:d.K])
end

function integrated_gaussian(T,tau,A,location,precision)
    J,K=size(T);D=Diagonal(inv.(tau.^2))
    Q=Symmetric(Diagonal(precision)+J*A'*D*A)
    conditional=Q\(precision.*location+A'*D*vec(sum(T;dims=1)))
    beta=conditional.+sqrt.(diag(inv(Q))).*.1
    joint=sum(logpdf(Normal((A*beta)[k],tau[k]),T[j,k]) for j in 1:J,k in 1:K)
    joint+=sum(precision[k]>0 ? logpdf(Normal(location[k],inv(sqrt(precision[k]))),beta[k]) : 0. for k in eachindex(beta))
    joint-logpdf(MvNormal(conditional,Symmetric(inv(Q))),beta)
end

half_t_prior(t)=logpdf(LocationScale(0.,2.5,TDist(3)),t)+log(2.)+log(t)

function total_reference(d,m,q)
    c=m.coords;T=q[c.totals];tau=exp.(q[c.scales]);sigma=exp(q[m.sigma]);lm=only(q[c.mixture])
    mean_values=d.K==1 ? T[d.group,1].+q[m.slope].*(d.x.-d.xbar) : T[d.group,1].+T[d.group,2].*d.x
    location=d.K==1 ? [2.8] : [2.8,0.]
    precision=d.K==1 ? [exp(lm)/2.5^2] : [exp(lm)/2.5^2,0.]
    sum(logpdf.(Normal.(mean_values,sigma),d.y))+sum(half_t_prior,tau)+half_t_prior(sigma)+
        integrated_gaussian(T,tau,m.block.A,location,precision)+logpdf(Gamma(1.5,1/1.5),exp(lm))+lm
end

function ordinary_physical(d,m,q)
    names=BridgeStan.param_names(m;include_tp=true);idx=Dict(n=>i for (i,n) in enumerate(names))
    p=BridgeStan.param_constrain(m,collect(q);include_tp=true)
    beta=p[[idx["Intercept"],idx["b.1"]]];tau=p[[idx["sd_1.$k"] for k in 1:d.K]]
    sigma=p[idx["sigma"]]
    deviations=hcat([p[[idx["r_1_$k.$j"] for j in 1:d.J]] for k in 1:d.K]...)
    means=beta[1].+beta[2].*(d.x.-d.xbar).+deviations[d.group,1]
    d.K==2 && (means .+= deviations[d.group,2].*d.x)
    (;beta,tau,sigma,deviations,means)
end

function ordinary_reference(d,m,q)
    p=ordinary_physical(d,m,q)
    sum(logpdf.(Normal.(p.means,p.sigma),d.y))+sum(half_t_prior,p.tau)+half_t_prior(p.sigma)+
        logpdf(LocationScale(2.8,2.5,TDist(3)),p.beta[1])+sum(logpdf.(Normal(),p.deviations./p.tau'))
end

function finite_gradient(f,q)
    [begin
        h=1e-5*max(1,abs(q[k]));plus=copy(q);minus=copy(q)
        plus[k]+=h;minus[k]-=h;(f(plus)-f(minus))/(2h)
    end for k in eachindex(q)]
end
end
