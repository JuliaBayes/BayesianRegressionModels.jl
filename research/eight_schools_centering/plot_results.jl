using BayesianRegressionModels, Statistics
import Pkg
const SOURCE_DRAWS = 10_000
const SOURCE_SEED = 1
using AlgebraOfVega
using CSV
using Tables
using CairoMakie
import AlgebraOfGraphics
using Serialization
using SHA
using TOML

const CONFIGURATIONS = ("NCP", "Centered", "Post-hoc", "Online")
rows(path) = collect(Tables.namedtupleiterator(CSV.File(path; delim='\t')))
school_label(j) = "School $j"

function save_spec(output, name, spec; title, size=(960, 550), legend=true)
    # Without the legend column, facet row strips would run past the canvas
    # edge, so legend-free figures carry explicit right padding instead.
    padding = legend ? (5, 5, 5, 5) : (5, 45, 5, 5)
    fig = Figure(; size, fontsize=18, figure_padding=padding)
    Label(fig[0, 1:(legend ? 2 : 1)], title; fontsize=23, font=:bold, tellwidth=false)
    grid = sdraw!(fig[1, 1], spec)
    # Facet rows already label every school; a second color legend repeats
    # those labels, so some figures opt out (keeping their row strips).
    legend && AlgebraOfGraphics.legend!(fig[1, 2], grid)
    path = joinpath(output, "$name.png")
    save(path, fig; px_per_unit=1.5)
    println("figure\t", path)
    path
end

function fit_matrix(fit_dir, label)
    table = rows(joinpath(fit_dir, "$(label == "partial" ? "partial" : label)_coordinates.tsv"))
    matrix = Matrix{Float64}(undef, SOURCE_DRAWS, 8)
    for row in table
        matrix[row.draw, row.school] = row.theta_effect
    end
    all(isfinite, matrix) || error("non-finite $label school effects")
    matrix
end

function intervals(input; x=:school, xlabel="School", ylabel="Treatment effect", facets=(;))
    summary = [merge(r, (; lo90=r.q50-r.q05, hi90=r.q95-r.q50,
                           lo50=r.q50-r.q25, hi50=r.q75-r.q50)) for r in input]
    base = data(summary)
    wide = base * mapping(x => xlabel, :q50 => ylabel, :lo90, :hi90; facets...) *
        visual(Errorbars; color="#79b8de", linewidth=2, whiskerwidth=5)
    narrow = base * mapping(x => xlabel, :q50 => ylabel, :lo50, :hi50; facets...) *
        visual(Errorbars; color="#19679a", linewidth=5, whiskerwidth=0)
    medians = base * mapping(x => xlabel, :q50 => ylabel; facets...) *
        visual(Scatter; color="#19679a", markersize=6)
    wide + narrow + medians
end

function posterior_figures(fit_dir, output)
    input = NamedTuple[]
    for (label, configuration) in (("noncentered", "1 Noncentered"),
                                    ("centered", "2 Fully centered"))
        matrix = fit_matrix(fit_dir, label)
        for school in 1:8
            qs = quantile(view(matrix, :, school), [0.05, 0.25, 0.5, 0.75, 0.95])
            push!(input, (; school, configuration, q05=qs[1], q25=qs[2],
                q50=qs[3], q75=qs[4], q95=qs[5]))
        end
    end
    spec = intervals(input; facets=(; col=:configuration)) *
        config(axis=(; xticks=1:8))
    save_spec(output, "posterior_theta", spec; title="Posterior school effects",
              size=(960, 460), legend=false)
end

function ppc_figure(diagnostics_dir, output)
    input = rows(joinpath(diagnostics_dir, "ppc_intervals.tsv"))
    observed = data(input) * mapping(:school => "School", :observed => "Reported estimate") *
        visual(Scatter; color="#bd3d2a", markersize=12)
    spec = (intervals(input; ylabel="Reported estimate") + observed) *
        config(axis=(; xticks=1:8))
    save_spec(output, "posterior_predictive_check", spec;
        title="Posterior predictive intervals and observed estimates",
        size=(960, 480), legend=false)
end

function pair_figures(diagnostics_dir, output)
    all_rows = rows(joinpath(diagnostics_dir, "coordinate_pairs.tsv"))
    for (name, configurations, title) in (
            ("centered_vs_noncentered", ("Centered", "NCP"), "One pilot, two coordinate systems"),
            ("selected_vs_online", ("Centered", "Post-hoc", "Online"), "Centered reference and selected centering"))
        selected = [(; r.hyperparameter, r.coordinate,
            configuration=string(findfirst(==(r.configuration), configurations), " ", r.configuration),
            school=school_label(r.school)) for r in all_rows if r.configuration in configurations]
        length(selected) == length(configurations) * 8 * SOURCE_DRAWS || error("incomplete pair rows")
        spec = data(selected) * mapping(:hyperparameter => "Random-effect SD τ",
            :coordinate => "School coordinate"; col=:configuration, row=:school) *
            visual(Scatter; color="#19679a", opacity=0.12, markersize=3) *
            config(facet=(; linkxaxes=:all, linkyaxes=:none))
        save_spec(output, name, spec; title, size=(1080, 1500), legend=false)
    end
end

const SCHOOL_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
                       "#D55E00", "#56B4E9", "#777777", "#332288"]

function centering_figure(fit_dir, output)
    table = rows(joinpath(fit_dir, "centeredness.tsv"))
    input = vcat([(; x=r.school-0.08, centeredness=r.offline, method="Offline") for r in table],
                 [(; x=r.school+0.08, centeredness=r.online, method="Online") for r in table])
    spec = data(input) * mapping(:x => "School", :centeredness => "Centeredness";
        color=:method => "Selection") * visual(Scatter; markersize=10) *
        config(axis=(; xticks=1:8, limits=(nothing, (0,1))))
    save_spec(output, "selected_centeredness", spec;
        title="Selected centering for each school", size=(960, 470))
end

function loss_figure(input, output, name, title, ylabel; limits=nothing)
    spec = data(input) * mapping(:centeredness => "Candidate centeredness", :loss => ylabel;
        color=:school => "School") * visual(Lines; linewidth=2) *
        config(axis=(; limits=(nothing, limits)),
            scales=scales(Color=(; categories=school_label.(1:8), palette=SCHOOL_COLORS)))
    save_spec(output, name, spec; title, size=(960,550))
end

function offline_loss_figure(fit_dir, output)
    table = rows(joinpath(fit_dir, "offline_loss_profiles.tsv"))
    input = NamedTuple[]
    for school in 1:8
        selected = filter(r -> r.school == school, table)
        lo, hi = extrema(r.loss for r in selected)
        append!(input, [(; r.centeredness, loss=(r.loss-lo)/(hi-lo),
                          school=school_label(school)) for r in selected])
    end
    loss_figure(input, output, "offline_loss_profiles", "Offline selection objective",
                "Loss (per-school min–max)")
end

function online_loss_figure(diagnostics_dir, output)
    table = rows(joinpath(diagnostics_dir, "retrospective_online_losses.tsv"))
    input = [(; r.centeredness, r.loss, school=school_label(r.school)) for r in table]
    all(r -> isfinite(r.loss), input) || error("non-finite online losses")
    loss_figure(input, output, "online_loss", "Online objective evaluated on the pilot",
                "Position–gradient correlation"; limits=(-1,0))
end

function gradient_figure(diagnostics_dir, output)
    table = rows(joinpath(diagnostics_dir, "gradient_scatter.tsv"))
    for (name, configurations) in (("gradient_centered_vs_noncentered", ("Centered", "NCP")),
                                    ("gradient_selected_vs_online", ("Centered", "Post-hoc", "Online")))
        input = [(; r.coordinate, r.gradient,
            configuration=string(findfirst(==(r.configuration), configurations), " ", r.configuration),
            school=school_label(r.school)) for r in table if r.configuration in configurations]
        length(input) == length(configurations) * 8_000 || error("incomplete gradient points")
        spec = data(input) * mapping(:coordinate => "School coordinate",
            :gradient => "Log-density gradient"; col=:configuration, row=:school) *
            visual(Scatter; color="#19679a", opacity=0.25, markersize=4) *
            config(facet=(; linkxaxes=:none, linkyaxes=:none))
        save_spec(output, name, spec; title="Position and gradient in displayed coordinates",
                  size=(1080, 1500), legend=false)
    end
end

function write_tsv(path, table)
    names = propertynames(first(table))
    open(path, "w") do io
        println(io, join(names, '\t'))
        for row in table
            println(io, join((getproperty(row, key) for key in names), '\t'))
        end
    end
end

function render_dependencies()
    selected = ("BayesianRegressionModels", "AlgebraOfVega", "AlgebraOfGraphics",
                "CairoMakie", "Makie", "DynamicObjects", "HTMXObjects", "Treebars")
    [Dict("name" => p.name, "version" => string(p.version), "source" => p.source,
          "tree_hash" => string(p.tree_hash),
          "git_sha" => ispath(joinpath(p.source, ".git")) ?
              strip(read(`git -C $(p.source) rev-parse HEAD`, String)) : "")
     for p in sort!(collect(values(Pkg.dependencies())); by=p -> p.name)
     if p.name in selected]
end

function figure_provenance(fit_dir, diagnostics_dir, output)
    result = NamedTuple[]
    for path in sort(filter(f -> endswith(f, ".png"), readdir(output; join=true)))
        push!(result, (; figure=basename(path), sha256=bytes2hex(sha256(read(path))),
            bytes=filesize(path), fit_directory=fit_dir,
            diagnostics_directory=diagnostics_dir))
    end
    write_tsv(joinpath(output, "figure_provenance.tsv"), result)
    open(joinpath(output, "figure_provenance.toml"), "w") do io
        TOML.print(io, Dict(
            "fit_producer_sha256" => bytes2hex(sha256(
                read(joinpath(fit_dir, "eight-schools-model.stan")))),
            "pair_plot_draws_per_configuration" => 10_000,
            "gradient_display_draws_per_facet" => 1_000,
            "loss_rows" => "all 10,000 retained pilot draws",
            "renderer" => "native AlgebraOfVega specifications with CairoMakie layout",
            "render_dependencies" => render_dependencies(),
            "plot_script_sha256" => bytes2hex(sha256(read(@__FILE__))),
        ))
    end
end

function plot_results(fit_dir, diagnostics_dir, output=joinpath(fit_dir, "figures"))
    mkpath(output)
    dependencies_before = render_dependencies()
    posterior_figures(fit_dir, output)
    ppc_figure(diagnostics_dir, output)
    pair_figures(diagnostics_dir, output)
    centering_figure(fit_dir, output)
    offline_loss_figure(fit_dir, output)
    online_loss_figure(diagnostics_dir, output)
    gradient_figure(diagnostics_dir, output)
    dependencies_before == render_dependencies() || error("render dependency changed")
    figure_provenance(fit_dir, diagnostics_dir, output)
    println("eight_schools_figures_complete\t", output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    2 <= length(ARGS) <= 3 ||
        error("usage: plot_results.jl FULL_FIT_DIRECTORY DIAGNOSTICS_DIRECTORY [OUTPUT_DIRECTORY]")
    plot_results(ARGS...)
end
