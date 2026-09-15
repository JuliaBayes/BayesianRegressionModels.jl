# Reproduce a centering-boundary transition using saved pupil draws, without HMC.
include("run.jl")

function audit_online_boundary(input,output; repaired=false)
    mkpath(output)
    data=PTE.load_data()
    fit=deserialize(joinpath(input,"scaled_total.jls"))
    physical=fit.positions[:,1:8:end]
    controls=zeros(40)
    target=problem(controls,data;mean_prior=get(fit,:mean_prior,PTE.GaussianMean()))
    source=reduce(hcat,(PTE.to_source(q,controls,data) for q in eachcol(physical)))
    gradients=reduce(hcat,(last(LogDensityProblems.logdensity_and_gradient(target.rp,q)) for q in eachcol(source)))
    old_source=copy(source[:,end])
    old_point=WarmupHMC.DynamicHMC.evaluate_ℓ(target.rp,old_source)
    old_physical=PTE.to_model(old_source,controls,data)
    old_lp=first(PTE.evaluate(target.raw,old_physical))
    next_point=WarmupHMC.find_reparametrization!(target.rp,source,gradients,old_point)
    learned=[last(pair).c for pair in WarmupHMC.reparam_sources(target.rp)]
    next_physical=PTE.to_model(next_point.q,learned,data)
    expected_source=PTE.to_source(old_physical,learned,data)
    expected_point=WarmupHMC.DynamicHMC.evaluate_ℓ(target.rp,expected_source)
    transported=reduce(hcat,(PTE.to_model(q,learned,data) for q in eachcol(source)))
    halo_error=maximum(abs.(transported.-physical))
    gradient_error=maximum(maximum(abs.(g-last(LogDensityProblems.logdensity_and_gradient(target.rp,q)))./
        (1 .+abs.(g))) for (q,g) in zip(eachcol(source),eachcol(gradients)))
    changed=count(!iszero,learned)
    active_error=maximum(abs.(next_physical.-old_physical))
    @test changed>0
    @test halo_error<1e-8 && gradient_error<1e-8
    if repaired
        @test active_error<1e-8
        @test next_point.q ≈ expected_source
    else
        @test next_point.q==old_source
        @test active_error>1.0
    end
    @test source[:,end] ≈ expected_source
    @test PTE.to_model(expected_point.q,learned,data) ≈ old_physical
    row=(;changed_controls=changed,halo_physical_error=halo_error,halo_gradient_error=gradient_error,
        active_physical_error=active_error,source_left_unchanged=next_point.q==old_source,
        physical_logdensity_before=old_lp,
        physical_logdensity_after=first(PTE.evaluate(target.raw,next_physical)),
        physical_logdensity_after_correct_transport=first(PTE.evaluate(target.raw,
            PTE.to_model(expected_point.q,learned,data))))
    write_tsv(joinpath(output,"boundary_reproducer.tsv"),[row])
    write_tsv(joinpath(output,"selected_controls.tsv"),[(;coordinate=j,c=learned[j]) for j in 1:40])
    println("BOUNDARY_REPRODUCER\t",row)
    # Check recorded restart boundaries too. Dropped draws retain the old
    # source frame, whereas reparam_sources describes the newly selected frame.
    checkpoints=joinpath(input,"checkpoints-online")
    if !isdir(checkpoints)
        println("ONLINE_BOUNDARY_AUDIT_COMPLETE (no historical online checkpoints for this prior)")
        return
    end
    previous=deserialize(joinpath(checkpoints,"cp_init.jls"))
    history=NamedTuple[]
    files=sort(filter(n->occursin(r"^cp_window_\d+\.jls$",n),readdir(checkpoints));
               by=n->parse(Int,match(r"\d+",n).match))
    for file in files
        cp=deserialize(joinpath(checkpoints,file))
        if cp.restart && cp.n_samples==0 && size(cp.dropped_posterior_position,2)>0
            before_c=[last(pair).c for pair in previous.reparam_sources]
            after_c=[last(pair).c for pair in cp.reparam_sources]
            # The last retained transition is the active state at restart.
            last_old=cp.dropped_posterior_position[:,end]
            active=cp.position_and_gradient.q
            same=active==last_old
            before=PTE.to_model(last_old,before_c,data)
            after=PTE.to_model(active,after_c,data)
            push!(history,(;window=cp.outer_counter,changed_controls=count(before_c.!=after_c),
                dropped_draws=size(cp.dropped_posterior_position,2),active_equals_last_old_source=same,
                physical_error=maximum(abs.(after.-before)),
                physical_logdensity_before=first(PTE.evaluate(target.raw,before)),
                physical_logdensity_after=first(PTE.evaluate(target.raw,after))))
        end
        previous=cp
    end
    if !isempty(history)
        write_tsv(joinpath(output,"recorded_boundaries.tsv"),history)
        foreach(println,history)
    end
    println("ONLINE_BOUNDARY_AUDIT_COMPLETE")
end

if abspath(PROGRAM_FILE)==@__FILE__
    audit_online_boundary(ARGS...)
end
