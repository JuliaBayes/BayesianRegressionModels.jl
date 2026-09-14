using BayesianRegressionModels, AlgebraOfVega, CSV, Tables, JSON, TOML, CairoMakie
import AlgebraOfGraphics

"""Plot the exported native WarmupHMC query, not a reconstructed loss formula."""
function online_loss_preview(input_dir)
    all_rows = collect(Tables.namedtupleiterator(CSV.File(
        joinpath(input_dir, "retrospective_online_losses.tsv"); delim='\t')))
    rows = NamedTuple[]
    unavailable = 0
    segment_id = 0
    # One common physical reference sample, matching the offline pilot loss.
    # Source coordinates are not a dimension of the global candidate landscape.
    for predictor in ("mu", "log_sigma"),
        basis in (1, 2, 19, 20)
        curve = sort(filter(r -> r.configuration == "NCP" &&
            r.predictor == predictor && r.basis == basis, all_rows); by=r -> r.centeredness)
        length(curve) == 11 || error("Incomplete native scoring curve")
        segment_id += 1
        for r in curve
            r.evidence == "retrospective_saved_draws" &&
                r.objective == "position_gradient_correlation_w1_0" ||
                error("Unexpected loss provenance")
            if ismissing(r.loss) || !isfinite(r.loss)
                unavailable += 1
                segment_id += 1
                continue
            end
            push!(rows, (; predictor=predictor == "mu" ? "Mean GP" : "Log-SD GP",
                basis_label="Basis $(lpad(basis, 2, '0'))",
                centeredness=r.centeredness, loss=r.loss,
                segment=string(segment_id)))
        end
    end
    title = "Online correlation objective — common pilot reference"
    outside = filter(r -> !( -1 <= r.loss <= 0), rows)
    raw_range = extrema(r.loss for r in rows)
    plot = brm_centering_lossplot(rows; title, normalization=:none, ylimits=(-1, 0),
        ylabel="Position–gradient correlation (w₁ = 0)")
    spec = to_vegalite(plot; interactive=false)
    provenance = TOML.parsefile(joinpath(input_dir, "diagnostics_provenance.toml"))
    base = provenance["pilot"]["brm_commit"]
    envelope = Dict("schema" => "kb-aov/v1", "title" => title,
        "alt" => "Two panels: Mean GP and Log-SD GP. Four basis functions per panel. Native WarmupHMC candidate scores use one common reference of all 10,000 pilot posterior draws, with unit weights. Source coordinates do not change the loss landscape. This is retrospective, not recorded warmup history. Raw correlations, with fixed y limits [-1,0]; no min-max scaling. $unavailable nonfinite or missing candidate values are gaps, retained in the source table.",
        "spec" => spec, "provenance" => Dict(
            "producer" => "BayesianRegressionModels:docs:adaptive-centering",
            "mode" => "preliminary", "base_commit" => base,
            "run" => "$(provenance["coordinate_gradients_sha256"])-common-pilot-raw",
            "references" => [Dict("kind" => "spec", "label" => "BRM reproduction harness that produced this pilot",
                "url" => "https://github.com/nsiccha/BayesianRegressionModels.jl/blob/$base/research/adaptive_centering/reproduce.jl",
                "commit" => base, "path" => "research/adaptive_centering/reproduce.jl")]))
    open(joinpath(input_dir, "online-loss.preview.md"), "w") do io
        print(io, "```kb-aov\n", JSON.json(envelope), "\n```\n")
    end
    fig = Figure(size=(1400, 500), fontsize=15)
    Label(fig[0, 1:2], title; fontsize=21, font=:bold, tellwidth=false)
    grid = sdraw!(fig[1, 1], plot)
    AlgebraOfGraphics.legend!(fig[1, 2], grid)
    save(joinpath(input_dir, "online-loss.png"), fig; px_per_unit=1.5)
    println("native_loss_preview\trows=", length(rows), "\tunavailable=", unavailable,
            "\toutside_ylim=", length(outside), "\traw_range=", raw_range,
            "\tspec_bytes=", sizeof(JSON.json(spec)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Usage: online_loss_preview.jl DIAGNOSTICS_DIR")
    online_loss_preview(only(ARGS))
end
