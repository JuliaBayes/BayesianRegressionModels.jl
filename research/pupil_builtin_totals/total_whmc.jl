isdefined(@__MODULE__,:BUILDER) || include("common.jl")
using Test

function run_total(label,tm,out;centeredness=0.,online=false,pilot_cost=0)
    raw=BrmsPupilProblem(tm.model,Ref(0);reject_numerical_errors=true)
    rp=adaptive_centering_problem(tm.sb,raw,AutoEnzyme();unc_names=tm.names,centeredness)
    _,initial=WarmupHMC._inverse_with_logabsdet_jacobian(WarmupHMC.reparametrizer(rp),total_initial(tm))
    checkpoint_dir=joinpath(out,label*"-checkpoints")
    callback=(state,stage)->begin
        println("BOUNDARY ",label," ",stage," window=",state.outer_counter," gradients=",raw.gradient_calls[])
        flush(stdout);isfile(joinpath(out,"STOP"))
    end
    raw.gradient_calls[]=0
    timed=@timed adaptive_warmup_mcmc(Xoshiro(1),rp;init=initial,n_draws=2000,
        monitor_ess=true,nonlinear_adapt=online,checkpoint_dir,callback)
    fit=timed.value;calls=raw.gradient_calls[]
    @test size(fit.posterior_position,2)>=2000
    cp=deserialize(joinpath(checkpoint_dir,"cp_latest.jls"))
    @test cp.sampling_evaluation_counter==fit.sampling_evaluation_counter
    @test calls>=fit.total_evaluation_counter>=fit.sampling_evaluation_counter>0
    @test WarmupHMC.back_transform(cp,rp,cp.posterior_position) ≈ fit.posterior_position
    record=(;positions=Matrix(fit.posterior_position),source_positions=Matrix(cp.posterior_position),
        source_gradients=Matrix(cp.posterior_gradient),names=tm.names,
        controls=[last(p).c for p in WarmupHMC.reparam_sources(rp)],
        sampling_gradients=fit.sampling_evaluation_counter,all_gradient_calls=calls,
        total_gradient_calls=calls+pilot_cost,pilot_gradient_calls=pilot_cost,
        divergences=fit.n_divergent_samples,fit_seconds=timed.time,numerical_rejections=raw.numerical_rejections[])
    serialize(joinpath(out,label*".jls"),record)
    qois=total_qois(tm,record.positions)
    serialize(joinpath(out,label*"-qois.jls"),qois)
    scientific_diagnostics(label,qois,record.sampling_gradients,record.total_gradient_calls,record.divergences,out)
    record
end

function main(out,audit_dir)
    @assert JSON.parsefile(joinpath(audit_dir,"audit.json"))["status"]=="passed"
    ispath(out) && error("Preserve previous fits; choose a fresh directory")
    mkpath(out);BLAS.set_num_threads(1)
    tm=total_model(joinpath(out,"model"))
    cp(@__DIR__,joinpath(out,"harness"))
    pilot=run_total("total_ncp",tm,out)
    source=adaptive_centering_problem(tm.sb,BrmsPupilProblem(tm.model,Ref(0)),AutoEnzyme();unc_names=tm.names,centeredness=0.)
    target=adaptive_centering_problem(tm.sb,BrmsPupilProblem(tm.model,Ref(0)),AutoEnzyme();unc_names=tm.names,centeredness=1.)
    physical=similar(pilot.positions);gradients=similar(pilot.positions)
    WarmupHMC._jointly_transport_halo!(target,WarmupHMC.reparametrizer(source),
        pilot.source_positions,pilot.source_gradients,physical,gradients)
    @test physical≈pilot.positions
    for s in (1,1000,2000)
        direct=last(BridgeStan.log_density_gradient(tm.model,physical[:,s];propto=false))
        @test gradients[:,s]≈direct rtol=1e-8 atol=1e-7
    end
    serialize(joinpath(out,"pilot-model-gradients.jls"),gradients)
    for criterion in (:position,:gradient)
        selected=select_total_centeredness(tm.sb,permutedims(pilot.positions),tm.names;
            criterion,gradients=permutedims(gradients))
        serialize(joinpath(out,"selected_$(criterion).jls"),selected)
        run_total("total_posthoc_$(criterion)",tm,out;centeredness=selected.centeredness,
            pilot_cost=pilot.all_gradient_calls)
    end
    run_total("total_cp",tm,out;centeredness=1.)
    for criterion in (:position,:gradient)
        online_loss!(criterion)
        Base.invokelatest(run_total,"total_online_$(criterion)",tm,out;online=true)
    end
    println("PUPIL3_BUILTIN_TOTAL_MATRIX_COMPLETE");flush(stdout)
end
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end
