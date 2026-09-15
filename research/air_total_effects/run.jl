isdefined(@__MODULE__,:AIRTotals) || include("model.jl")
using .AIRTotals, Serialization, Statistics, LinearAlgebra, Random, JSON, BridgeStan
using WarmupHMC, LogDensityProblems, MCMCDiagnosticTools
using DifferentiationInterface: AutoEnzyme

function write_air_tsv(path,rows)
    keys=propertynames(first(rows))
    open(path,"w") do io
        println(io,join(keys,'\t'))
        for row in rows;println(io,join((getproperty(row,k) for k in keys),'\t'));end
    end
end

function qoi_names(d)
    vcat(["population_intercept_at_mean_log_sat","population_log_sat_slope"],
        ["group_sd_$k" for k in 1:d.K],["residual_sd"],
        ["region_$(j)_total_intercept" for j in 1:d.J],
        d.K==2 ? ["region_$(j)_total_slope" for j in 1:d.J] : String[])
end

function total_qois(d,m,positions)
    rows=permutedims(positions)
    recovered=AIRTotals.BRM.recover_population_draws(m.sb,rows,m.names;rng=Xoshiro(404))[:mu]
    population=d.K==1 ? hcat(recovered.population[:,1],rows[:,m.slope]) : recovered.population
    intercept=rows[:,m.coords.totals[:,1]]
    d.K==1 && (intercept=intercept.-d.xbar.*population[:,2])
    values=hcat(population,exp.(rows[:,m.coords.scales]),exp.(rows[:,m.sigma]),intercept)
    d.K==2 ? hcat(values,rows[:,m.coords.totals[:,2]]) : values
end

function ordinary_qois(d,target,positions)
    permutedims(hcat([begin
        v=AIRTotals.ordinary_physical(d,target,q)
        intercept=v.beta[1].-d.xbar*v.beta[2].+v.deviations[:,1]
        vcat(v.beta,v.tau,v.sigma,intercept,d.K==2 ? v.beta[2].+v.deviations[:,2] : Float64[])
    end for q in eachcol(positions)]...))
end

function diagnostics(d,label,qois,fit,out)
    names=qoi_names(d);@assert size(qois)==(2000,length(names)) && all(isfinite,qois)
    cube=reshape(qois,2000,1,length(names))
    ess=vec(MCMCDiagnosticTools.ess(cube;kind=:bulk));rh=vec(MCMCDiagnosticTools.rhat(cube))
    mcse=vec(MCMCDiagnosticTools.mcse(cube;kind=mean))
    write_air_tsv(joinpath(out,label*"-qois.tsv"),[(;parameter=names[k],mean=mean(qois[:,k]),
        sd=std(qois[:,k]),bulk_ess=ess[k],split_rhat=rh[k],mcse=mcse[k]) for k in eachindex(names)])
    s=(;arm=label,draws=2000,min_bulk_ess=minimum(ess),limiting_qoi=names[argmin(ess)],
        sampling_gradients=fit.sampling_gradients,total_gradients=fit.total_gradient_calls,
        ess_per_1000_sampling_gradients=1000minimum(ess)/fit.sampling_gradients,
        ess_per_1000_total_gradients=1000minimum(ess)/fit.total_gradient_calls,
        divergences=fit.divergences,max_split_rhat=maximum(rh))
    write_air_tsv(joinpath(out,label*"-summary.tsv"),[s]);println("AIR_SCIENTIFIC_RESULT ",s);flush(stdout)
end

function run_arm(d,m,label,out;centeredness=0.,online=false,pilot_cost=0,ordinary=nothing)
    target=isnothing(ordinary) ? m.model : ordinary
    raw=AIRTotals.AirStanProblem(target)
    rp=isnothing(ordinary) ? AIRTotals.BRM.adaptive_centering_problem(m.sb,raw,AutoEnzyme();unc_names=m.names,centeredness) : raw
    init=if isnothing(ordinary)
        last(WarmupHMC._inverse_with_logabsdet_jacobian(WarmupHMC.reparametrizer(rp),AIRTotals.total_initial(d,m)))
    else
        BridgeStan.param_unconstrain_json(target,JSON.json(AIRTotals.ordinary_init(d)))
    end
    checkpoint_dir=joinpath(out,label*"-checkpoints")
    callback=(state,stage)->begin
        println("BOUNDARY ",label," ",stage," window=",state.outer_counter," gradients=",raw.gradient_calls[])
        flush(stdout);isfile(joinpath(out,"STOP"))
    end
    path=joinpath(out,label*".jls")
    record=if isfile(path)
        deserialize(path)
    else
        raw.gradient_calls[]=0
        timed=@timed adaptive_warmup_mcmc(Xoshiro(1),rp;init,n_draws=2000,monitor_ess=true,
            nonlinear_adapt=online,checkpoint_dir,callback)
        fit=timed.value;cp=deserialize(joinpath(checkpoint_dir,"cp_latest.jls"))
        @assert size(fit.posterior_position,2)==2000
        @assert raw.gradient_calls[]>=fit.total_evaluation_counter>=fit.sampling_evaluation_counter>0
        @assert WarmupHMC.back_transform(cp,rp,cp.posterior_position)≈fit.posterior_position
        value=(;positions=Matrix(fit.posterior_position),source_positions=Matrix(cp.posterior_position),
            source_gradients=Matrix(cp.posterior_gradient),names=isnothing(ordinary) ? m.names : BridgeStan.param_unc_names(target),
            controls=[last(p).c for p in WarmupHMC.reparam_sources(rp)],
            sampling_gradients=fit.sampling_evaluation_counter,all_gradient_calls=raw.gradient_calls[],
            total_gradient_calls=raw.gradient_calls[]+pilot_cost,pilot_gradient_calls=pilot_cost,
            divergences=fit.n_divergent_samples,fit_seconds=timed.time,
            numerical_rejections=raw.numerical_rejections[],first_spd_point=raw.first_spd_point[])
        serialize(path,value);value
    end
    qois=isnothing(ordinary) ? total_qois(d,m,record.positions) : ordinary_qois(d,target,record.positions)
    serialize(joinpath(out,label*"-qois.jls"),qois);diagnostics(d,label,qois,record,out)
    record
end

function online_loss!(kind)
    source=read(joinpath(pkgdir(WarmupHMC),"src","Reparametrizations.jl"),String)
    for (start,finish) in (("reparametrization_loss((;ljac, cov)::OnlineReparametrizationLoss","scale_estimate(loss::OnlineReparametrizationLoss)"),
            ("function reparametrization_loss(loss::WeightedReparametrizationLoss","scale_estimate(loss::WeightedReparametrizationLoss)"))
        a=first(findfirst(start,source));b=first(findnext(finish,source,a))-1
        code=source[a:b];@assert occursin("w1=0",code)
        Base.include_string(WarmupHMC,replace(code,"w1=0"=>(kind==:position ? "w1=1" : "w1=0")),"air-loss-selection")
    end
end

function run_matrix(d,m,out,ordinary)
    mkpath(out);BLAS.set_num_threads(1)
    pilot=run_arm(d,m,"total_ncp",out)
    source=AIRTotals.BRM.adaptive_centering_problem(m.sb,AIRTotals.BrmsPupilProblem(m.model,Ref(0)),AutoEnzyme();unc_names=m.names,centeredness=0.)
    target=AIRTotals.BRM.adaptive_centering_problem(m.sb,AIRTotals.BrmsPupilProblem(m.model,Ref(0)),AutoEnzyme();unc_names=m.names,centeredness=1.)
    physical=similar(pilot.positions);gradients=similar(pilot.positions)
    WarmupHMC._jointly_transport_halo!(target,WarmupHMC.reparametrizer(source),pilot.source_positions,pilot.source_gradients,physical,gradients)
    @assert physical≈pilot.positions
    for draw in (1,1000,2000)
        @assert isapprox(gradients[:,draw],last(BridgeStan.log_density_gradient(m.model,physical[:,draw];propto=false));rtol=1e-8,atol=1e-7)
    end
    serialize(joinpath(out,"pilot-model-gradients.jls"),gradients)
    for criterion in (:position,:gradient)
        selected=AIRTotals.BRM.select_total_centeredness(m.sb,permutedims(physical),m.names;
            criterion,gradients=permutedims(gradients))
        serialize(joinpath(out,"selected_$(criterion).jls"),selected)
        run_arm(d,m,"total_posthoc_$(criterion)",out;centeredness=selected.centeredness,pilot_cost=pilot.all_gradient_calls)
    end
    run_arm(d,m,"total_cp",out;centeredness=1.)
    for criterion in (:position,:gradient)
        online_loss!(criterion)
        Base.invokelatest(run_arm,d,m,"total_online_$(criterion)",out;online=true)
    end
    run_arm(d,m,"ordinary_ncp_whmc",out;ordinary)
    println("AIR_TOTAL_MATRIX_COMPLETE ",d.grouping," ",d.hierarchy);flush(stdout)
end

if abspath(PROGRAM_FILE)==@__FILE__
    grouping,hierarchy,out,audit_dir=ARGS
    @assert JSON.parsefile(joinpath(audit_dir,"audit.json"))["status"]=="passed"
    d=AIRTotals.load_data(grouping,hierarchy);m=AIRTotals.model(d,joinpath(out,"model"))
    ordinary=BridgeStan.StanModel(joinpath(d.dir,"ordinary_ncp.stan"),joinpath(d.dir,"ordinary_ncp.json");warn=false)
    run_matrix(d,m,out,ordinary)
end
