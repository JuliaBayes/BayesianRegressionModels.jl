using BayesianRegressionModels, AlgebraOfVega, CSV, Tables, JSON, SHA, CairoMakie

# Display thinning only: all 10,000 saved draws were evaluated and remain in
# coordinate_gradients.tsv. Each facet uses the same 50 evenly spaced indices.
function gradient_preview(input_dir)
    source = joinpath(input_dir, "coordinate_gradients.tsv")
    all_rows = collect(Tables.namedtupleiterator(CSV.File(source; delim='\t')))
    indices = Set(round.(Int, range(1, 10_000; length=50)))
    labels = Dict("NCP" => "1 NCP", "Post-hoc" => "2 Post-hoc", "Online" => "3 Online")
    base = "8dfe41253af3043482cb3270cf513b50a1de5437"
    output = String[]
    plots = Pair[]
    for (predictor, title) in (("mu", "Mean GP: position versus gradient"),
                               ("log_sigma", "Log-SD GP: position versus gradient"))
        rows = [(; coordinate=r.coordinate, gradient=r.gradient,
                   configuration=labels[r.configuration], basis_label="Basis $(lpad(r.basis, 2, '0'))")
                for r in all_rows if r.predictor == predictor && r.basis in (1, 20) && r.draw in indices]
        @assert length(rows) == 300 && all(r -> isfinite(r.coordinate) && isfinite(r.gradient), rows)
        plot = brm_gradientplot(rows; title, opacity=0.6, markersize=4)
        spec = to_vegalite(plot)
        # KB v1 permits a noninteractive AoV figure, not zoom/selection programs.
        # Do not change data, scales, mappings, or marks during this projection.
        function remove_interactivity!(value)
            if value isa AbstractDict
                pop!(value, "params", nothing)
                foreach(remove_interactivity!, values(value))
            elseif value isa AbstractVector
                foreach(remove_interactivity!, value)
            end
            value
        end
        remove_interactivity!(spec)
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
        @assert sizeof(body) <= 48 * 1024
        open(joinpath(input_dir, "gradient-$predictor.vl.json"), "w") do io
            print(io, body)
        end
        envelope = Dict(
            "schema" => "kb-aov/v1", "title" => title,
            "alt" => "Three configurations in columns; low and high frequency basis rows. Both axes are independent in every panel. Fifty saved draws per panel, no new sampling.",
            "spec" => spec,
            "provenance" => Dict(
                "producer" => "BayesianRegressionModels:docs:adaptive-centering",
                "mode" => "preliminary", "base_commit" => base,
                "run" => "saved-10000-draw-gradient-preview-v1",
                "references" => [Dict("kind" => "spec", "label" => "Original full-fit BRM reproduction harness; preview extension is work in progress",
                    "url" => "https://github.com/nsiccha/BayesianRegressionModels.jl/blob/$base/research/adaptive_centering/reproduce.jl",
                    "commit" => base, "path" => "research/adaptive_centering/reproduce.jl")]))
        fence = "```kb-aov\n" * JSON.json(envelope) * "\n```"
        @assert sizeof(fence) <= 64 * 1024
        push!(output, fence)
        push!(plots, predictor => plot)
        println("AoV preview\t", predictor, "\trows=", length(rows), "\tspec_bytes=", sizeof(body))
    end
    open(joinpath(input_dir, "preview-fences.md"), "w") do io
        print(io, join(output, "\n\n"))
    end
    println("source_sha256\t", bytes2hex(sha256(read(source))))
    for (predictor, plot) in plots
        sdraw_file(plot, joinpath(input_dir, "gradient-$predictor.png"))
        println("rendered\tgradient-", predictor, ".png")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Usage: gradient_preview.jl DIAGNOSTICS_DIR")
    gradient_preview(only(ARGS))
end
