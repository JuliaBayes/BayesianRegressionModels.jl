using BayesianRegressionModels, AlgebraOfVega, CSV, Tables, CairoMakie, JSON, SHA, TOML
import Pkg
import AlgebraOfGraphics

const ROLES = (:intercept, :slope)
role_label(role) = String(role) == "intercept" ? "County intercept" : "County slope"
# Same categorical palette as the BRM AlgebraOfVega extension, so the native
# county panels below match the shared loss/pair/gradient panels above.
const CENTEREDNESS_COLORS = ["#0B7BEC", "#E67E22", "#16877A", "#984EA3"]
table(dir, name) = collect(Tables.namedtupleiterator(
    CSV.File(joinpath(dir, name); delim='\t')))

function display_rows(source, representatives, role, configurations)
    chosen = Dict(r.county => r for r in representatives if Symbol(r.role) == role)
    [(; r..., panel=string(findfirst(==(r.configuration), configurations), " ",
             r.configuration == "2 centered" ? "Centered" :
             r.configuration == "1 NCP" ? "Noncentered" :
             r.configuration == "3 post-hoc selected" ? "Selected (pilot)" :
             r.configuration == "4 post-hoc fit" ? "Selected (refit)" :
             r.configuration == "5 online learned" ? "Online (refit)" :
             replace(r.configuration, r"^\d+ " => "")),
        cell=string(chosen[r.county].rank, " County ", lpad(r.county,3,'0'),
                    " · c=", chosen[r.county].centeredness))
     for r in source if Symbol(r.role) == role && r.configuration in configurations]
end

function save_panel(output, name, plot; title, size=(1080, 600), legend=true)
    fig = Figure(; size, fontsize=18, figure_padding=legend ? (5,5,5,5) : (5,45,5,5))
    Label(fig[0, 1:(legend ? 2 : 1)], title; fontsize=23, font=:bold, tellwidth=false)
    grid = sdraw!(fig[1, 1], plot)
    # Panels whose facet strips already identify every series pass
    # legend=false: a color legend would only duplicate the strips.
    legend && AlgebraOfGraphics.legend!(fig[1, 2], grid)
    path = joinpath(output, "$name.png")
    save(path, fig; px_per_unit=1.5)
    spec_path = joinpath(output, "$name.vl.json")
    open(spec_path, "w") do io
        JSON.print(io, to_vegalite(plot; interactive=false))
    end
    println("figure\t", path, "\tspec=", spec_path)
    path
end

function ppc_plot(diagnostics)
    source = table(diagnostics, "ppc_curves.tsv")
    [r.index for r in source] == collect(1:length(source)) || error("PPC row order changed")
    counts = Dict(j => count(r -> r.county == j, source) for j in unique(r.county for r in source))
    eligible = [j for j in keys(counts) if
        count(r -> r.county == j && r.floor == 0, source) >= 5 &&
        count(r -> r.county == j && r.floor == 1, source) >= 5]
    sort!(eligible; by=j -> (counts[j], j))
    counties = unique([first(eligible), eligible[cld(length(eligible), 2)], last(eligible)])
    rows = [merge(r, (; county_label="County $(r.county) (n=$(counts[r.county]))",
        floor_label="Floor code $(Int(r.floor))",
        lo90=r.q50-r.q05, hi90=r.q95-r.q50,
        lo50=r.q50-r.q25, hi50=r.q75-r.q50)) for r in source if r.county in counties]
    all(r -> all(isfinite, (r.q05, r.q25, r.q50, r.q75, r.q95)), rows) ||
        error("PPC intervals are non-finite")
    base = data(rows)
    wide = base * mapping(:index => "Original data row", :q50 => "Log radon",
        :lo90, :hi90; row=:county_label) *
        visual(Errorbars; color="#aac9df", linewidth=1, whiskerwidth=0)
    narrow = base * mapping(:index => "Original data row", :q50 => "Log radon",
        :lo50, :hi50; row=:county_label) *
        visual(Errorbars; color="#5789af", linewidth=2, whiskerwidth=0)
    observed = base * mapping(:index => "Original data row", :observation => "Log radon";
        row=:county_label, color=:floor_label => "Observed floor code") *
        visual(Scatter; markersize=6, strokewidth=0.4, strokecolor=:black)
    (wide + narrow + observed) * config(facet=(; linkxaxes=:none, linkyaxes=:all),
        scales=scales(Color=(; categories=["Floor code $j" for j in (0,1,2,3,9)],
            palette=["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#444444"])))
end

function centering_rows(offline, online)
    selected = table(offline, "selected_centeredness.tsv")
    learned = table(online, "online_centeredness.tsv")
    rows = vcat([merge(r, (; predictor=role_label(r.role),
                           configuration="Post-hoc pilot")) for r in selected],
                [merge(r, (; predictor=role_label(r.role),
                           configuration="Online warmup")) for r in learned])
    sort(rows; by=r -> (r.configuration, r.predictor, r.county))
end

# County-index centeredness comparison. The shared brm_centerednessplot maps a
# `:basis` carrier-frequency axis, which is wrong for county cells, so this
# panel composes the scatter algebra natively with county labels. Counties are
# unordered, so adjacent-county connecting lines would imply false continuity:
# scatter only.
function centeredness_plot(rows)
    data(rows) * mapping(:county => "County index",
        :centeredness => "Centeredness";
        col=:predictor, color=:configuration => "Selection") *
        visual(Scatter; markersize=5) *
        config(width=460, height=300,
               axis=(; limits=(nothing, (0, 1))),
               scales=scales(Color=(; palette=CENTEREDNESS_COLORS)))
end

function offline_loss_rows(offline, representatives)
    source = table(offline, "offline_loss_profiles.tsv")
    rows = NamedTuple[]
    for cell in representatives
        role, county = Symbol(cell.role), cell.county
        curve = sort(filter(r -> String(r.role) == String(role) && r.county == county, source);
                     by=r -> r.centeredness)
        length(curve) == 101 || error("offline loss profile is incomplete")
        segment = 1
        for r in curve
            isfinite(r.loss) && r.admissible || begin
                segment += 1
                continue
            end
            push!(rows, (; predictor=role_label(role),
                basis_label=string(cell.rank, " ", cell.criterion),
                segment="$role-$county-$segment", centeredness=r.centeredness,
                loss=r.loss))
        end
    end
    rows
end

function online_loss_rows(offline, representatives)
    source = table(offline, "retrospective_online_losses.tsv")
    selected = Dict((Symbol(r.role),r.county) => r for r in representatives)
    [(; predictor=role_label(r.role),
       basis_label=string(selected[(Symbol(r.role),r.county)].rank, " ",
                          selected[(Symbol(r.role),r.county)].criterion),
       segment="$(r.role)-$(r.county)", centeredness=r.centeredness, loss=r.loss)
     for r in source if haskey(selected,(Symbol(r.role),r.county))]
end

function pair_plot(diagnostics, representatives, role, configurations)
    rows = display_rows(table(diagnostics, "coordinate_pairs.tsv"),
                        representatives, role, configurations)
    length(rows) == length(configurations) * 3 * 10_000 || error("incomplete pair rows")
    all(r -> isfinite(r.hyperparameter) && r.hyperparameter > 0 && isfinite(r.coordinate), rows) ||
        error("pair plot contains invalid coordinates or scales")
    data(rows) * mapping(:hyperparameter => "County-effect SD",
        :coordinate => "County coordinate"; col=:panel, row=:cell) *
        visual(Scatter; color="#19679a", opacity=0.12, markersize=3) *
        config(facet=(; linkxaxes=:all, linkyaxes=:none),
               scales=scales(X=(; scale=log10)))
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

function plot_results(offline, online, diagnostics;
                      output=joinpath(diagnostics, "figures"))
    mkpath(output)
    dependencies_before = render_dependencies()
    provenance = TOML.parsefile(joinpath(diagnostics, "diagnostics_provenance.toml"))
    representatives = table(diagnostics, "representative_coordinates.tsv")
    paths = String[]
    push!(paths, save_panel(output, "data-ppc", ppc_plot(diagnostics);
        title="Observed log radon and predictive intervals by county", size=(1050, 1000)))
    push!(paths, save_panel(output, "selected-centeredness",
        centeredness_plot(centering_rows(offline, online));
        title="Per-county selected centering", size=(1050, 520)))
    push!(paths, save_panel(output, "offline-loss-profiles",
        brm_centering_lossplot(offline_loss_rows(offline, representatives);
            normalization=:minmax,
            ylabel="Offline loss (per-county min–max)");
        title="Offline KL/log-scale proxy on the pilot", size=(1050, 500)))
    push!(paths, save_panel(output, "online-loss-profiles",
        brm_centering_lossplot(online_loss_rows(offline, representatives);
            normalization=:none, ylimits=(-1, 0),
            ylabel="Position–gradient correlation (w₁ = 0)");
        title="Common-pilot retrospective online objective", size=(1050, 500)))
    # Each role has its own three selected coordinates. The same centered
    # pilot reference appears first in both the pilot and fresh-fit figures.
    for role in ROLES
        for (family, configurations, title) in (
            ("pilot", ("2 centered", "1 NCP", "3 post-hoc selected"),
                "$(role_label(role)): one pilot, three coordinate systems"),
            ("fits", ("2 centered", "4 post-hoc fit", "5 online learned"),
                "$(role_label(role)): centered reference and fresh fits"))
            push!(paths, save_panel(output, "pair-$family-$role",
                pair_plot(diagnostics, representatives, role, configurations);
                title, size=(1080, 1050), legend=false))
        end
    end
    gradients = table(diagnostics, "coordinate_gradients.tsv")
    length(gradients) == 18_000 || error("gradient display table is incomplete")
    all(r -> isfinite(r.coordinate) && isfinite(r.gradient), gradients) ||
        error("gradient display contains non-finite values")
    for role in ROLES
        rows = display_rows(gradients, representatives, role,
                            ("1 Centered", "2 post-hoc", "3 online"))
        length(rows) == 9_000 || error("incomplete $role gradient rows")
        plot = data(rows) * mapping(:coordinate => "County coordinate",
            :gradient => "Log-density gradient"; col=:panel, row=:cell) *
            visual(Scatter; color="#19679a", opacity=0.25, markersize=4) *
            config(facet=(; linkxaxes=:none, linkyaxes=:none))
        push!(paths, save_panel(output, "position-gradient-$role", plot;
            title="$(role_label(role)): position and displayed gradient",
            size=(1080, 1050), legend=false))
    end
    manifest = [(; figure=splitext(basename(path))[1], png=path,
        png_sha256=bytes2hex(sha256(read(path))),
        spec_sha256=bytes2hex(sha256(read(joinpath(output, "$(splitext(basename(path))[1]).vl.json")))))
        for path in paths]
    open(joinpath(output, "figure_manifest.tsv"), "w") do io
        println(io, "figure\tpng\tpng_sha256\tspec_sha256")
        for row in manifest
            println(io, join((row.figure, row.png, row.png_sha256, row.spec_sha256), '\t'))
        end
    end
    dependencies_before == render_dependencies() || error("render dependency changed")
    open(joinpath(output, "render_provenance.toml"), "w") do io
        TOML.print(io, Dict("render_dependencies" => dependencies_before,
            "plot_script_sha256" => bytes2hex(sha256(read(@__FILE__)))))
    end
    println("render_complete\tfigures=", length(paths), "\tbackend=AlgebraOfVega/CairoMakie")
    paths
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || error("usage: plot_results.jl OFFLINE_DIR ONLINE_DIR DIAGNOSTICS_DIR")
    plot_results(ARGS...)
end
