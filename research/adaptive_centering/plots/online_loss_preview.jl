using BayesianRegressionModels, AlgebraOfVega, CSV, Tables, JSON, CairoMakie

"""Plot the exported native WarmupHMC query, not a reconstructed loss formula."""
function online_loss_preview(input_dir)
    all_rows = collect(Tables.namedtupleiterator(CSV.File(
        joinpath(input_dir, "retrospective_online_losses.tsv"); delim='\t')))
    labels = Dict("NCP" => "1 NCP", "Post-hoc" => "2 Post-hoc", "Online" => "3 Online")
    rows = NamedTuple[]
    unavailable = 0
    for configuration in ("NCP", "Post-hoc", "Online"), predictor in ("mu", "log_sigma"),
        basis in (1, 2, 19, 20)
        curve = sort(filter(r -> r.configuration == configuration &&
            r.predictor == predictor && r.basis == basis, all_rows); by=r -> r.centeredness)
        length(curve) == 11 || error("Incomplete native scoring curve")
        segment = 1
        for r in curve
            r.evidence == "retrospective_saved_draws" &&
                r.objective == "position_gradient_correlation_w1_0" ||
                error("Unexpected loss provenance")
            if ismissing(r.loss) || !isfinite(r.loss)
                unavailable += 1
                segment += 1
                continue
            end
            push!(rows, (; configuration=labels[configuration],
                predictor=predictor == "mu" ? "Mean GP" : "Log-SD GP",
                basis_label="Basis $(lpad(basis, 2, '0'))",
                centeredness=r.centeredness, loss=r.loss,
                segment="$configuration-$predictor-$basis-$segment"))
        end
    end
    title = "Online objective replayed on saved draws — not warmup history"
    plot = brm_centering_lossplot(rows; title, configurations=true,
                                  ylabel="Position–gradient correlation (w₁ = 0)")
    spec = to_vegalite(plot; interactive=false)
    base = "8dfe41253af3043482cb3270cf513b50a1de5437"
    envelope = Dict("schema" => "kb-aov/v1", "title" => title,
        "alt" => "Three independently sampled fits in columns, two GPs in rows, four basis functions per panel. Native WarmupHMC candidate scores, unit-weight replay on all 10,000 saved draws. $unavailable nonfinite or missing candidate values are gaps, retained in the source table.",
        "spec" => spec, "provenance" => Dict(
            "producer" => "BayesianRegressionModels:docs:adaptive-centering",
            "mode" => "preliminary", "base_commit" => base,
            "run" => "native-candidate-scoring-losses-saved-draws-v1",
            "references" => [Dict("kind" => "spec", "label" => "Published full-fit BRM harness; native loss-query extension is work in progress",
                "url" => "https://github.com/nsiccha/BayesianRegressionModels.jl/blob/$base/research/adaptive_centering/reproduce.jl",
                "commit" => base, "path" => "research/adaptive_centering/reproduce.jl")]))
    open(joinpath(input_dir, "online-loss.preview.md"), "w") do io
        print(io, "```kb-aov\n", JSON.json(envelope), "\n```\n")
    end
    fig = Figure(size=(1550, 850), fontsize=15)
    Label(fig[0, 1], title; fontsize=21, font=:bold)
    sdraw!(fig[1, 1], plot)
    save(joinpath(input_dir, "online-loss.png"), fig; px_per_unit=1.5)
    println("native_loss_preview\trows=", length(rows), "\tunavailable=", unavailable,
            "\tspec_bytes=", sizeof(JSON.json(spec)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Usage: online_loss_preview.jl DIAGNOSTICS_DIR")
    online_loss_preview(only(ARGS))
end
