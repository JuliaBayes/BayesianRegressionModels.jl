include("run.jl")

function audit_online(output)
    data = PTE.load_data()
    fit = deserialize(joinpath(output,"online.jls"))
    p = problem(fit.controls,data)
    cp = deserialize(joinpath(output,"checkpoints-online","cp_latest.jls"))
    points = round.(Int,range(1,size(cp.posterior_position,2),length=24))
    rows = NamedTuple[]
    for s in points
        r = cp.posterior_position[:,s]
        q = PTE.to_model(r,fit.controls,data)
        value,g = LogDensityProblems.logdensity_and_gradient(p.rp,r)
        expected_gradient = copy(last(PTE.evaluate(p.raw,q)))
        for j in 1:40
            scaleidx = j<=20 ? 1 : 2
            loc = j<=20 ? PTE.INTERCEPT_MEAN : 0.0
            expected_gradient[scaleidx] += (1-fit.controls[j])*(q[4+j]-loc)*expected_gradient[4+j]
            expected_gradient[scaleidx] += 1-fit.controls[j]
            expected_gradient[4+j] *= exp((1-fit.controls[j])*r[scaleidx])
        end
        transform_error = maximum(abs.(g-expected_gradient)./(1 .+ abs.(expected_gradient)))
        checkpoint_error = maximum(abs.(g-cp.posterior_gradient[:,s])./(1 .+ abs.(g)))
        @test transform_error < 1e-10
        @test checkpoint_error < 1e-10
        push!(rows,(; draw=s,transform_error,checkpoint_error))
    end
    write_tsv(joinpath(output,"online_gradient_frame_audit.tsv"),rows)
    baseline = deserialize(joinpath(output,"scaled_total.jls"))
    indices = 1:4:size(baseline.positions,2)
    physical = baseline.positions[:,indices]
    source = reduce(hcat,(PTE.to_source(q,fit.controls,data) for q in eachcol(physical)))
    gradients = reduce(hcat,(last(LogDensityProblems.logdensity_and_gradient(p.rp,q)) for q in eachcol(source)))
    if isdefined(WarmupHMC,:candidate_scoring_losses)
        native = WarmupHMC.candidate_scoring_losses(p.rp,source,gradients)
        write_tsv(joinpath(output,"online_retrospective_native_losses.tsv"),native)
        physical_gradients = reduce(hcat,(last(PTE.evaluate(p.raw,q)) for q in eachcol(physical)))
        comparisons = map(native) do row
            j, c = row.index-4, row.candidate
            ell = physical[j<=20 ? 1 : 2,:]
            loc = j<=20 ? PTE.INTERCEPT_MEAN : 0.0
            candidate = c*loc .+ (physical[row.index,:].-loc).*exp.((c-1).*ell)
            candidate_gradient = physical_gradients[row.index,:].*exp.((1-c).*ell)
            expected_loss = cor(candidate,candidate_gradient)
            (; coordinate=j,candidate=c,native_loss=row.loss,expected_loss,
                error=abs(row.loss-expected_loss))
        end
        @test maximum(row.error for row in comparisons) < 1e-9
        write_tsv(joinpath(output,"online_candidate_score_audit.tsv"),comparisons)
        println("max_candidate_score_error=",maximum(row.error for row in comparisons))
        println("native_retrospective_rows=",length(native))
    else
        println("native_retrospective_query_unavailable_at_recorded_pin")
    end
    println("max_transform_error=",maximum(r.transform_error for r in rows))
    println("max_checkpoint_gradient_error=",maximum(r.checkpoint_error for r in rows))
end

if abspath(PROGRAM_FILE) == @__FILE__
    audit_online(only(ARGS))
end
