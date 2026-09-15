isdefined(@__MODULE__,:PTE) || include("run.jl")

function prepare_pairs(output)
    fit = deserialize(joinpath(output,"partial.jls"))
    data = PTE.load_data()
    controls = fit.controls
    selected = [argmin(controls)]
    unused = setdiff(1:40,selected)
    push!(selected,unused[argmin(abs.(controls[unused].-0.5))])
    unused = setdiff(1:40,selected)
    push!(selected,unused[argmax(controls[unused])])
    names = PTE.coordinate_names(data)
    rows = NamedTuple[]
    selected_rows = NamedTuple[]
    labels = ("Minimum c","Nearest 0.5 (distinct)","Maximum c (distinct)")
    # All columns visualize the same 2000 retained partial-refit draws.
    for (k,j) in enumerate(selected)
        parameter = names[4+j]
        push!(selected_rows,(;selection=labels[k],coordinate=j,parameter,centeredness=controls[j]))
        kind = j<=20 ? "A" : "B"
        subject = data.ids[mod1(j,20)]
        panel = "$k: $kind$subject (c=$(controls[j]))"
        for (column,c) in (("1. CP",1.0),("2. NCP",0.0),("3. ACP (offline)",controls[j]))
            for s in axes(fit.positions,2)
                ell = fit.positions[j<=20 ? 1 : 2,s]
                value = fit.positions[4+j,s]
                loc = j<=20 ? PTE.INTERCEPT_MEAN : 0.0
                source = c*loc + (value-loc)*exp((c-1)*ell)
                push!(rows,(;panel,column,draw=s,log_group_sd=ell,coordinate=source))
            end
        end
    end
    @test length(rows)==9size(fit.positions,2)
    @test all(r->isfinite(r.coordinate)&&isfinite(r.log_group_sd),rows)
    write_tsv(joinpath(output,"pairs.tsv"),rows)
    write_tsv(joinpath(output,"pairs_selection.tsv"),selected_rows)
    println("PAIRS_COMPLETE\t",output,"\trows=",length(rows))
end

if abspath(PROGRAM_FILE)==@__FILE__
    prepare_pairs(only(ARGS))
end
