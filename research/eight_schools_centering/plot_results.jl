include(joinpath(@__DIR__, "reproduce.jl"))
using AlgebraOfVega
using CSV
using Tables
using CairoMakie
import AlgebraOfGraphics
using Serialization
using SHA
using TOML

const CONFIGURATIONS = ("NCP", "Post-hoc", "Online")
rows(path) = collect(Tables.namedtupleiterator(CSV.File(path; delim='\t')))
school_label(j) = "School $j"

function save_spec(output, name, spec; title, size=(1500, 650), legend=true)
    # Without the legend column, facet row strips would run past the canvas
    # edge, so legend-free figures carry explicit right padding instead.
    padding = legend ? (5, 5, 5, 5) : (5, 45, 5, 5)
    fig = Figure(; size, fontsize=15, figure_padding=padding)
    Label(fig[0, 1:(legend ? 2 : 1)], title; fontsize=21, font=:bold, tellwidth=false)
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
    fit = deserialize(joinpath(fit_dir, "$label.jls"))
    table = rows(joinpath(fit_dir, "$(label == "partial" ? "partial" : label)_coordinates.tsv"))
    matrix = Matrix{Float64}(undef, size(fit.posterior_position, 2), 8)
    for row in table
        matrix[row.draw, row.school] = row.theta_effect
    end
    all(isfinite, matrix) || error("non-finite $label school effects")
    matrix
end

function posterior_figures(fit_dir, output)
    specs = NamedTuple[]
    for (label, title) in (("noncentered", "NCP pilot"),
                           ("partial", "Selected-partial refit"),
                           ("online", "Online adaptive fit"))
        matrix = fit_matrix(fit_dir, label)
        push!(specs, (; label, title,
            spec=brm_posteriorplot(matrix; x=string.(1:8),
                probs=[0.9, 0.8, 0.5], xlabel="School",
                ylabel="Treatment effect")))
    end
    fig = Figure(size=(2100, 620), fontsize=15)
    Label(fig[0, 1:3], "School treatment effects"; fontsize=23, font=:bold, tellwidth=false)
    for (i, panel) in enumerate(specs)
        slot = fig[1, i] = GridLayout()
        Label(slot[0, 1], panel.title; fontsize=18, font=:bold, tellwidth=false)
        # Ribbon panels carry no color mapping, so no legend slot is reserved.
        sdraw!(slot[1, 1], panel.spec)
    end
    path = joinpath(output, "posterior_theta.png")
    save(path, fig; px_per_unit=1.5)
    println("figure\t", path)
end

function ppc_figure(fit_dir, output)
    fit = deserialize(joinpath(fit_dir, "partial_target.jls"))
    stan = stan_density("plot", output)
    read(joinpath(output, "eight-schools-plot.stan")) ==
        read(joinpath(fit_dir, "eight-schools-model.stan")) ||
        error("plot target differs from fit producer")
    descriptor = brm_descriptor(stan.sb)
    spec = brm_ppcplot(descriptor, permutedims(fit.posterior_position);
        problem=stan.density, seed=SOURCE_SEED, response=:y, x=string.(1:8))
    save_spec(output, "posterior_predictive_check", spec;
        title="Selected-partial posterior predictive check", size=(1450, 650),
        legend=false)
end

function pair_figures(diagnostics_dir, output)
    all_rows = rows(joinpath(diagnostics_dir, "coordinate_pairs.tsv"))
    for configuration in CONFIGURATIONS
        selected = map(filter(row -> row.configuration == configuration, all_rows)) do row
            (; row.hyperparameter, row.coordinate,
               parameter="Random-effect SD", basis_label=school_label(row.school))
        end
        isempty(selected) && error("missing $configuration pair rows")
        length(selected) == 80_000 || error("unexpected $configuration pair-row count")
        spec = brm_pairplot(selected)
        title = configuration == "NCP" ? "Original noncentered pilot coordinates" :
            configuration == "Post-hoc" ? "Post-hoc selected partial coordinates" :
            "Online-adaptive selected coordinates"
        save_spec(output, "$(lowercase(replace(configuration, " " => "-")))_scatter", spec;
            title, size=(1800, 1350), legend=false)
    end
end

function centering_figure(fit_dir, output)
    table = rows(joinpath(fit_dir, "centeredness.tsv"))
    selected = [(; basis=Int(row.school), predictor="School effect",
                  centeredness=row.offline, configuration="Post-hoc") for row in table]
    append!(selected, [(; basis=Int(row.school), predictor="School effect",
                         centeredness=row.online, configuration="Online") for row in table])
    save_spec(output, "selected_centeredness",
        brm_centerednessplot(selected; compare=true, xlabel="School");
        title="Offline and online selected centering", size=(1250, 600))
end

function offline_loss_figure(fit_dir, output)
    table = rows(joinpath(fit_dir, "offline_loss_profiles.tsv"))
    input = [(; centeredness=row.centeredness, loss=row.loss,
               predictor="School", basis_label=school_label(Int(row.school)),
               segment=string(row.school)) for row in table]
    save_spec(output, "offline_loss_profiles",
        brm_centering_lossplot(input; normalization=:minmax);
        title="Offline log-scale selection profiles (display normalized)", size=(1800, 850))
end

function online_loss_figure(diagnostics_dir, output)
    table = rows(joinpath(diagnostics_dir, "retrospective_online_losses.tsv"))
    input = NamedTuple[]
    unavailable = 0
    for row in table
        if ismissing(row.loss) || !isfinite(row.loss)
            unavailable += 1
            continue
        end
        push!(input, (; row.centeredness, row.loss, predictor="School",
            basis_label=school_label(Int(row.school)), segment=string(row.school)))
    end
    spec = brm_centering_lossplot(input; normalization=:none, ylimits=(-1, 0))
    save_spec(output, "online_loss", spec;
        title="Online correlation objective — common NCP pilot reference",
        size=(1800, 850))
    println("online_loss_rows\t", length(input), "\tunavailable=", unavailable)
end

function gradient_figure(diagnostics_dir, output)
    table = rows(joinpath(diagnostics_dir, "gradient_scatter.tsv"))
    input = [(; row.coordinate, row.gradient, row.configuration,
               basis_label=school_label(Int(row.school))) for row in table]
    length(input) == 24_000 || error("expected 24,000 gradient display points")
    save_spec(output, "gradient_scatter", brm_gradientplot(input; markersize=12);
        title="Coordinate position versus exact log-density gradient",
        size=(2400, 1500), legend=false)
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
            "renderer" => "BRM native AlgebraOfVega specifications with CairoMakie layout",
        ))
    end
end

function plot_results(fit_dir, diagnostics_dir, output=joinpath(fit_dir, "figures"))
    mkpath(output)
    posterior_figures(fit_dir, output)
    ppc_figure(fit_dir, output)
    pair_figures(diagnostics_dir, output)
    centering_figure(fit_dir, output)
    offline_loss_figure(fit_dir, output)
    online_loss_figure(diagnostics_dir, output)
    gradient_figure(diagnostics_dir, output)
    figure_provenance(fit_dir, diagnostics_dir, output)
    println("eight_schools_figures_complete\t", output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    2 <= length(ARGS) <= 3 ||
        error("usage: plot_results.jl FULL_FIT_DIRECTORY DIAGNOSTICS_DIRECTORY [OUTPUT_DIRECTORY]")
    plot_results(ARGS...)
end
