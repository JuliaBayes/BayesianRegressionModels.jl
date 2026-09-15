# Process-local experiment at WarmupHMC deeea1d128d5. No dependency files change.
# The two replacements preserve the active physical state when centering changes.
include("audit_online_boundary.jl")

function install_online_trial!(criterion)
    criterion in ("position","gradient") || error("Unknown criterion $criterion")
    path=joinpath(pkgdir(WarmupHMC),"src","Reparametrizations.jl")
    source=read(path,String)
    before="DynamicHMC.evaluate_ℓ(lpdf, position_and_gradient.q; strict=false)"
    after="""
    _, active_model = old_ir(position_and_gradient.q)
    _, active_source = _inverse_with_logabsdet_jacobian(reparametrizer(lpdf), active_model)
    DynamicHMC.evaluate_ℓ(lpdf, active_source; strict=false)
    """
    if !occursin("function _transport_active_evaluation",source)
        first_hook=findfirst("find_reparametrization!(lpdf, halo_position",source).start
        last_hook=findfirst("# Draws are stored in the SAMPLING",source).start-1
        hooks=source[first_hook:last_hook]
        @test count(before,hooks)==2
        Base.include_string(WarmupHMC,replace(hooks,before=>after),"online-active-state-trial")
    end
    if criterion=="position"
        # Existing loss formulas, with only the default mixture weight changed.
        # w1=1 is position log-SD minus mean forward log-Jacobian, as offline ACP.
        first_loss=findfirst("reparametrization_loss((;ljac, cov)::OnlineReparametrizationLoss",source).start
        last_loss=findnext("scale_estimate(loss::OnlineReparametrizationLoss)",source,first_loss).start-1
        loss=source[first_loss:last_loss]
        @test occursin("w1=0",loss)
        Base.include_string(WarmupHMC,replace(loss,"w1=0"=>"w1=1"),"online-position-loss-trial")
        first_weighted=findfirst("function reparametrization_loss(loss::WeightedReparametrizationLoss",source).start
        last_weighted=findnext("scale_estimate(loss::WeightedReparametrizationLoss)",source,first_weighted).start-1
        loss=source[first_weighted:last_weighted]
        @test occursin("w1=0",loss)
        Base.include_string(WarmupHMC,replace(loss,"w1=0"=>"w1=1"),"online-weighted-position-loss-trial")
    end
end

function main_online_trial(criterion,input,output)
    ispath(output) && error("Preserve existing trial; choose a new output")
    mkpath(output)
    BLAS.set_num_threads(1)
    install_online_trial!(criterion)
    # New methods installed at runtime must be called from the latest world.
    Base.invokelatest(audit_online_boundary,input,joinpath(output,"audit");repaired=true)
    data=PTE.load_data()
    prior=get(ENV,"PUPIL_INTERCEPT_PRIOR","gaussian")=="student_mixture" ? PTE.StudentMixtureMean() : PTE.GaussianMean()
    fit,summary=Base.invokelatest(run_arm,"online_"*criterion,zeros(40),data,output;
        online=true,mean_prior=prior)
    write_tsv(joinpath(output,"trial.tsv"),[(;criterion,transport="preserve active physical state",
        warmuphmc_base_sha="deeea1d128d5235ad0ecb2fd911a6d881f1ac2c2",
        package_source=pkgdir(WarmupHMC),position_loss_override=criterion=="position",summary...)])
    println("ONLINE_TRIAL_COMPLETE criterion=",criterion)
end

if abspath(PROGRAM_FILE)==@__FILE__
    main_online_trial(ARGS...)
end
