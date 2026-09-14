using BayesianRegressionModels, AlgebraOfVega, CSV, Tables, JSON, SHA, CairoMakie

# Full saved-draw diagnostics. Do not thin posterior data to fit a UI budget.
# KB delivery requires the approved large-payload renderer; generation does not.
function gradient_preview(input_dir)
    source = joinpath(input_dir, "coordinate_gradients.tsv")
    all_rows = collect(Tables.namedtupleiterator(CSV.File(source; delim='\t')))
    labels = Dict("NCP" => "1 NCP", "Post-hoc" => "2 Post-hoc", "Online" => "3 Online")
    # Canonical/public base of these saved fits, not this unpublished preview tip.
    base = "8dfe41253af3043482cb3270cf513b50a1de5437"
    output = String[]
    plots = Pair[]
    for (predictor, title) in (("mu", "Mean GP: position versus gradient"),
                               ("log_sigma", "Log-SD GP: position versus gradient"))
        rows = [(; coordinate=r.coordinate, gradient=r.gradient,
                   configuration=labels[r.configuration], basis_label="Basis $(lpad(r.basis, 2, '0'))")
                for r in all_rows if r.predictor == predictor && r.basis in (1, 2, 19, 20)]
        @assert length(rows) == 120_000 && all(r -> isfinite(r.coordinate) && isfinite(r.gradient), rows)
        for configuration in values(labels), basis in (1, 2, 19, 20)
            @assert count(r -> r.configuration == configuration &&
                r.basis_label == "Basis $(lpad(basis, 2, '0'))", rows) == 10_000
        end
        plot = brm_gradientplot(rows; title, opacity=0.12, markersize=2)
        spec = to_vegalite(plot; interactive=false)
        function check_bounded(value)
            if value isa AbstractDict
                for (key, child) in value
                    key in ("params", "selection", "transform", "test", "expr", "signal", "url", "href") &&
                        error("KB preview contains forbidden field $key")
                    check_bounded(child)
                end
            elseif value isa AbstractVector
                foreach(check_bounded, value)
            end
        end
        check_bounded(spec)
        body = JSON.json(spec)
        open(joinpath(input_dir, "gradient-$predictor.vl.json"), "w") do io
            print(io, body)
        end
        envelope = Dict(
            "schema" => "kb-aov/v1", "title" => title,
            "alt" => "Three configurations in columns; bases 1, 2, 19, 20 in rows. Both axes are independent in every panel. All 10,000 saved draws per panel, no new sampling.",
            "spec" => spec,
            "provenance" => Dict(
                "producer" => "BayesianRegressionModels:docs:adaptive-centering",
                "mode" => "preliminary", "base_commit" => base,
                "run" => "saved-10000-draw-gradient-preview-v2-unthinned",
                "references" => [Dict("kind" => "spec", "label" => "Original full-fit BRM reproduction harness; preview extension is work in progress",
                    "url" => "https://github.com/nsiccha/BayesianRegressionModels.jl/blob/$base/research/adaptive_centering/reproduce.jl",
                    "commit" => base, "path" => "research/adaptive_centering/reproduce.jl")]))
        fence = "```kb-aov\n" * JSON.json(envelope) * "\n```"
        open(joinpath(input_dir, "gradient-$predictor.preview.md"), "w") do io
            print(io, fence)
        end
        push!(output, fence)
        push!(plots, predictor => plot)
        println("AoV preview\t", predictor, "\trows=", length(rows), "\tspec_bytes=", sizeof(body))
    end
    open(joinpath(input_dir, "preview-fences.md"), "w") do io
        print(io, join(output, "\n\n"))
    end
    println("source_sha256\t", bytes2hex(sha256(read(source))))
    for (predictor, plot) in plots
        fig = Figure(size=(1650, 1450), fontsize=15)
        Label(fig[0, 1], "$(predictor == "mu" ? "Mean" : "Log-SD") GP: coordinate–gradient geometry";
              fontsize=23, font=:bold)
        sdraw!(fig[1, 1], plot)
        save(joinpath(input_dir, "gradient-$predictor.png"), fig; px_per_unit=1.5)
        println("rendered\tgradient-", predictor, ".png")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Usage: gradient_preview.jl DIAGNOSTICS_DIR")
    gradient_preview(only(ARGS))
end
