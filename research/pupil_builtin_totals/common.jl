using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using Distributions, LinearAlgebra, Statistics, Random, Serialization, JSON, Test
using WarmupHMC, Enzyme, MCMCDiagnosticTools
using DifferentiationInterface: AutoEnzyme
const BRM=BayesianRegressionModels
include(joinpath(@__DIR__,"..","pupil_total_effects","model.jl"))
include(joinpath(@__DIR__,"..","pupil_total_effects","stan_target.jl"))
const PTE=PupilTotalEffects
const DATA0=PTE.load_data()
const J=length(DATA0.ids)
const DATA=(;p_size=DATA0.y,load=DATA0.x,subj=DATA0.group,
    subject_id=DATA0.ids[DATA0.group])

# The post-3 data contain numeric subject IDs. Preserve the original residual
# scale regression exactly; this is not a model with one residual SD per subject.
const BUILDER=@brm begin
    mu ~ 1 + center(load) + (1 | mean_intercept | subj) + (0 + load | mean_slope | subj)
    logsigma ~ 1 + center(subject_id)
    effect(mu, Intercept) ~ LocationScale(5651.9,2026.1,TDist(3))
    effect(mu, center_load) ~ Flat()
    effect(logsigma, Intercept) ~ LocationScale(0.,2.5,TDist(3))
    effect(logsigma, center_subject_id) ~ Flat()
    sd(:, mean_intercept) ~ LocationScale(0.,2026.1,TDist(3))
    sd(:, mean_slope) ~ LocationScale(0.,2026.1,TDist(3))
    p_size ~ Normal(mu,exp(logsigma))
end

function total_model(out)
    mkpath(out)
    sb=SBBRMI(BUILDER(DATA);mod=@__MODULE__)
    block=only(total_effect_blocks(sb));@assert block.predictor==:mu
    p=Base.invokelatest(StanBlocks.stan_instantiate,sb.model;path=joinpath(out,"automatic_totals.stan"))
    names=BridgeStan.param_unc_names(p.model)
    c=BRM._total_coordinates(sb,block,names)
    gamma=[only(findall(==("pop_logsigma_beta_pop.$k"),names)) for k in 1:2]
    # This permutation makes the independent analytic reference directly
    # comparable to the emitted target and its gradient.
    to_manual=vcat(c.scales,gamma,vec(c.totals),c.mixture)
    @assert sort(to_manual)==collect(1:45)
    @assert block.A≈[1. -DATA0.xbar;0. 1.]
    (;sb,model=p.model,names,block,coords=c,gamma,to_manual)
end

function total_initial(tm)
    q=zeros(45)
    q[tm.to_manual]=PTE.initial_position(DATA0,PTE.StudentMixtureMean())
    q
end

const REFERENCE=PTE.PupilProblem(DATA0,PTE.StudentMixtureMean())
total_reference(tm,q)=PTE.evaluate(REFERENCE,q[tm.to_manual])

function total_qois(tm,positions;seed=101)
    rows=permutedims(positions)
    beta=recover_population_draws(tm.sb,rows,tm.names;rng=Xoshiro(seed))[:mu].population
    hcat(beta,exp.(rows[:,tm.coords.scales]),rows[:,tm.gamma],
        rows[:,tm.coords.totals[:,1]],rows[:,tm.coords.totals[:,2]])
end

function write_tsv(path,rows)
    keys=propertynames(first(rows))
    open(path,"w") do io
        println(io,join(keys,'\t'))
        for row in rows;println(io,join((getproperty(row,k) for k in keys),'\t'));end
    end
end

function qoi_names()
    vcat(["population_mean_intercept_at_mean_load","population_mean_load_slope",
        "mean_intercept_group_sd","mean_slope_group_sd","population_log_sigma_intercept",
        "population_log_sigma_subject_id_slope"],
        ["subject_$(700+j)_total_intercept" for j in 1:J],
        ["subject_$(700+j)_total_load_slope" for j in 1:J])
end

function scientific_diagnostics(label,qois,sampling,total,divergences,out)
    @assert size(qois)==(2000,46) && all(isfinite,qois)
    cube=reshape(qois,2000,1,46)
    ess=vec(MCMCDiagnosticTools.ess(cube;kind=:bulk))
    rh=vec(MCMCDiagnosticTools.rhat(cube));mcse=vec(MCMCDiagnosticTools.mcse(cube;kind=mean))
    names=qoi_names()
    write_tsv(joinpath(out,label*"-qois.tsv"),[(;parameter=names[j],mean=mean(qois[:,j]),
        sd=std(qois[:,j]),bulk_ess=ess[j],split_rhat=rh[j],mcse=mcse[j]) for j in 1:46])
    summary=(;arm=label,draws=2000,min_bulk_ess=minimum(ess),limiting_qoi=names[argmin(ess)],
        sampling_gradients=sampling,total_gradients=total,
        ess_per_1000_sampling_gradients=1000minimum(ess)/sampling,
        ess_per_1000_total_gradients=1000minimum(ess)/total,max_split_rhat=maximum(rh),divergences)
    write_tsv(joinpath(out,label*"-summary.tsv"),[summary]);println("SCIENTIFIC_RESULT ",summary);flush(stdout)
end

function online_loss!(kind)
    source=read(joinpath(pkgdir(WarmupHMC),"src","Reparametrizations.jl"),String)
    for (start,finish) in (("reparametrization_loss((;ljac, cov)::OnlineReparametrizationLoss","scale_estimate(loss::OnlineReparametrizationLoss)"),
            ("function reparametrization_loss(loss::WeightedReparametrizationLoss","scale_estimate(loss::WeightedReparametrizationLoss)"))
        a=first(findfirst(start,source));b=first(findnext(finish,source,a))-1
        code=source[a:b];@assert occursin("w1=0",code)
        Base.include_string(WarmupHMC,replace(code,"w1=0"=>(kind==:position ? "w1=1" : "w1=0")),"pupil3-loss-selection")
    end
end
