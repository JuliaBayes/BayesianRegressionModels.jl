include("run.jl")

function audit_boundaries(arm)
    out=joinpath(OUTPUT,arm,"boundary-audit"); mkpath(out)
    stan=make_stan(out)
    adaptive=adaptive_centering_problem(stan.sb,stan.density,ENZYME_BACKEND)
    dir=joinpath(OUTPUT,arm,"checkpoints")
    previous=deserialize(joinpath(dir,"cp_init.jls"))
    files=sort(filter(n->occursin(r"^cp_window_\d+\.jls$",n),readdir(dir));
        by=n->parse(Int,match(r"\d+",n).match))
    rows=NamedTuple[]
    for file in files
        checkpoint=deserialize(joinpath(dir,file))
        WarmupHMC.restore_reparam_sources!(adaptive,checkpoint.reparam_sources)
        active=checkpoint.position_and_gradient
        value,gradient=LogDensityProblems.logdensity_and_gradient(adaptive,active.q)
        @test isfinite(value) && all(isfinite,gradient)
        @test active.ℓq ≈ value
        @test active.∇ℓq ≈ gradient
        error=0.0
        if checkpoint.restart && checkpoint.n_samples==0 && size(checkpoint.dropped_posterior_position,2)>0
            after=last(WarmupHMC.reparametrizer(adaptive)(active.q))
            WarmupHMC.restore_reparam_sources!(adaptive,previous.reparam_sources)
            before=last(WarmupHMC.reparametrizer(adaptive)(checkpoint.dropped_posterior_position[:,end]))
            error=maximum(abs.(after-before)./(1 .+abs.(before)))
            @test error < 1e-9
        end
        push!(rows,(;window=checkpoint.outer_counter,restart=checkpoint.restart,physical_error=error,
            active_logdensity=value,gradient_error=maximum(abs.(active.∇ℓq-gradient)./(1 .+abs.(gradient)))))
        previous=checkpoint
    end
    write_tsv(joinpath(out,"boundaries.tsv"),rows)
    println("BOUNDARY_AUDIT_COMPLETE ",CASE," ",arm," ",rows)
end
for arm in split(REQUEST,',')
    audit_boundaries(arm)
end
