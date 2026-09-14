using BayesianRegressionModels, AlgebraOfVega, CSV, Tables, CairoMakie, JSON, SHA, TOML
import AlgebraOfGraphics

const ROLES = (:intercept, :slope)
role_label(role) = String(role) == "intercept" ? "County intercept" : "County slope"
# Same categorical palette as the BRM AlgebraOfVega extension, so the native
# county panels below match the shared loss/pair/gradient panels above.
const CENTEREDNESS_COLORS = ["#0B7BEC", "#E67E22", "#16877A", "#984EA3"]
table(dir, name) = collect(Tables.namedtupleiterator(
    CSV.File(joinpath(dir, name); delim='\t')))

function representative_counties(diagnostics)
    values = diagnostics["representative_counties"]
    values isa AbstractVector || error("diagnostics provenance lacks representative counties")
    Int.(values)
end

function save_panel(output, name, plot; title, size=(1300, 500))
    fig = Figure(; size, fontsize=15)
    Label(fig[0, 1], title; fontsize=21, font=:bold, tellwidth=false)
    grid = sdraw!(fig[1, 1], plot)
    AlgebraOfGraphics.legend!(fig[1, 2], grid)
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
    rows = sort(table(diagnostics, "ppc_curves.tsv"); by=r -> r.floor)
    all(r -> all(isfinite, (r.q05, r.q10, r.q25, r.q50, r.q75, r.q90, r.q95)), rows) ||
        error("PPC intervals are non-finite")
    observations = [(; floor=r.floor, response=r.observation) for r in rows]
    brm_posteriorplot(rows; x=:floor, xlabel="Floor measurement",
        ylabel="Log radon", observations, observed_y=:response,
        title="Observed log radon and posterior predictive intervals")
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

function offline_loss_rows(offline, counties)
    source = table(offline, "offline_loss_profiles.tsv")
    rows = NamedTuple[]
    for role in ROLES, county in counties
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
                basis_label="County $(lpad(county, 3, '0'))",
                segment="$role-$county-$segment", centeredness=r.centeredness,
                loss=r.loss))
        end
    end
    rows
end

function online_loss_rows(offline, counties)
    source = table(offline, "retrospective_online_losses.tsv")
    rows = filter(r -> String(r.role) in String.(ROLES) && r.county in counties, source)
    [(; predictor=role_label(r.role),
       basis_label="County $(lpad(r.county, 3, '0'))",
       segment="$(r.role)-$(r.county)", centeredness=r.centeredness,
       loss=r.loss) for r in rows]
end

function pair_plot(diagnostics, configurations)
    rows = filter(r -> r.configuration in configurations,
                  table(diagnostics, "coordinate_pairs.tsv"))
    all(r -> isfinite(r.hyperparameter) && r.hyperparameter > 0 &&
             isfinite(r.coordinate), rows) ||
        error("pair plot contains invalid coordinates or scales")
    brm_pairplot(rows)
end

function plot_results(offline, online, diagnostics;
                      output=joinpath(diagnostics, "figures"))
    mkpath(output)
    provenance = TOML.parsefile(joinpath(diagnostics, "diagnostics_provenance.toml"))
    counties = representative_counties(provenance)
    paths = String[]
    push!(paths, save_panel(output, "data-ppc", ppc_plot(diagnostics);
        title="Data and native posterior predictive check", size=(1300, 500)))
    push!(paths, save_panel(output, "selected-centeredness",
        centeredness_plot(centering_rows(offline, online));
        title="Per-county selected centering", size=(1050, 520)))
    push!(paths, save_panel(output, "offline-loss-profiles",
        brm_centering_lossplot(offline_loss_rows(offline, counties);
            normalization=:minmax,
            ylabel="Offline loss (per-county min–max)");
        title="Offline KL/log-scale proxy on the pilot", size=(1050, 500)))
    push!(paths, save_panel(output, "online-loss-profiles",
        brm_centering_lossplot(online_loss_rows(offline, counties);
            normalization=:none, ylimits=(-1, 0),
            ylabel="Position–gradient correlation (w₁ = 0)");
        title="Common-pilot retrospective online objective", size=(1050, 500)))
    # One figure per geometry: the shared pair algebra has no configuration
    # facet, so combining geometries would overplot indistinguishable clouds.
    for (name, configuration, title) in (
            ("pair-pilot-ncp", "1 NCP", "Noncentered pilot coordinates"),
            ("pair-pilot-centered", "2 centered",
                "Centered pilot coordinates (transformed draws)"),
            ("pair-pilot-posthoc", "3 post-hoc selected",
                "Post-hoc selected pilot coordinates (transformed draws)"),
            ("pair-fresh-posthoc", "4 post-hoc fit",
                "Fresh post-hoc partial-fit coordinates"),
            ("pair-fresh-online", "5 online learned",
                "Fresh online-fit coordinates (learned geometry)"))
        push!(paths, save_panel(output, name,
            pair_plot(diagnostics, (configuration,)); title, size=(1600, 780)))
    end
    gradients = [merge(row, (;
        basis_label="$(uppercasefirst(String(row.role))) / $(row.basis_label)"))
        for row in table(diagnostics, "coordinate_gradients.tsv")]
    length(gradients) == 3 * 2 * length(counties) * 1000 ||
        error("gradient display table is incomplete")
    all(r -> isfinite(r.coordinate) && isfinite(r.gradient), gradients) ||
        error("gradient display contains non-finite values")
    # One scatter figure per hierarchical role: stacking both roles in a single
    # panel crowds the long role/county facet strips.
    for (name, role, title) in (
            ("position-gradient-intercept", :intercept,
                "Position versus log-density gradient: county intercepts"),
            ("position-gradient-slope", :slope,
                "Position versus log-density gradient: county slopes"))
        rows = filter(r -> Symbol(r.role) == role, gradients)
        length(rows) == 3 * length(counties) * 1000 ||
            error("gradient display rows are incomplete for role $role")
        push!(paths, save_panel(output, name,
            brm_gradientplot(rows; opacity=0.25, markersize=8);
            title, size=(1800, 520)))
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
    println("render_complete\tfigures=", length(paths), "\tbackend=AlgebraOfVega/CairoMakie")
    paths
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || error("usage: plot_results.jl OFFLINE_DIR ONLINE_DIR DIAGNOSTICS_DIR")
    plot_results(ARGS...)
end
