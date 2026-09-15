# Post-hoc position-gradient centering, using gradients already saved by the NCP
# pilot. The grid matches the existing post-hoc position/Jacobian arm (step .01).
include("run.jl")

function offline_gradient_trial(input,output)
    ispath(output) && error("Preserve previous outputs")
    mkpath(output);BLAS.set_num_threads(1)
    data=PTE.load_data()
    pilot=deserialize(joinpath(input,"scaled_total.jls"))
    prior=get(pilot,:mean_prior,PTE.GaussianMean())
    cp=deserialize(joinpath(input,"checkpoints-scaled_total/cp_latest.jls"))
    controls=[last(pair).c for pair in cp.reparam_sources]
    target=problem(controls,data;mean_prior=prior)
    @test WarmupHMC.back_transform(cp,target.rp,cp.posterior_position) ≈ pilot.positions
    for s in (1,1000,2000)
        _,g=LogDensityProblems.logdensity_and_gradient(target.rp,cp.posterior_position[:,s])
        @test cp.posterior_gradient[:,s] ≈ g rtol=1e-10 atol=1e-8
    end
    rows=NamedTuple[];selected=zeros(40);grid=collect(0.0:.01:1.0)
    for j in 1:40
        ell=pilot.positions[j<=20 ? 1 : 2,:]
        physical=pilot.positions[4+j,:]
        loc=j<=20 ? PTE.INTERCEPT_MEAN : 0.0
        source_gradient=cp.posterior_gradient[4+j,:]
        losses=map(grid) do c
            candidate=c*loc .+(physical.-loc).*exp.((c-1).*ell)
            candidate_gradient=source_gradient.*exp.((controls[j]-c).*ell)
            cor(candidate,candidate_gradient)
        end
        @test all(isfinite,losses)
        selected[j]=grid[argmin(losses)]
        append!(rows,[(;coordinate=j,candidate=c,loss=losses[k]) for (k,c) in enumerate(grid)])
    end
    write_tsv(joinpath(output,"selection_losses.tsv"),rows)
    write_tsv(joinpath(output,"selected_controls.tsv"),[(;coordinate=j,c=selected[j]) for j in 1:40])
    fit,summary=run_arm("partial_gradient",selected,data,output;mean_prior=prior)
    write_tsv(joinpath(output,"workflow.tsv"),[(;criterion="position-gradient correlation",
        pilot_gradient_calls=pilot.all_gradient_calls,refit_gradient_calls=fit.all_gradient_calls,
        workflow_gradient_calls=pilot.all_gradient_calls+fit.all_gradient_calls,
        scoring_gradient_calls=0,scoring_evidence="stored NCP pilot positions and source gradients",
        validation_gradient_calls=3,grid_step=.01)])
    println("OFFLINE_GRADIENT_COMPLETE\t",summary)
end

if abspath(PROGRAM_FILE)==@__FILE__
    offline_gradient_trial(ARGS...)
end
