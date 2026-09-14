# Recreate the case-study figures from saved full-fit exports, using Julia/AoV.
# No fitting or posterior subsampling occurs here. Makie provides figure layout;
# all data marks are AlgebraOfVega specifications. HSGP coordinate changes use BRM.
using BayesianRegressionModels, AlgebraOfVega, CSV, Tables, CairoMakie

const CASE_BASES = (1, 2, 19, 20)
const CASE_COLORS = ["#0B7BEC", "#E67E22", "#16877A", "#984EA3"]
const CASE_GPS = ("mu", "log_sigma")
gp_label(p) = p == "mu" ? "Mean GP" : "Log-SD GP"
basis_label(b) = "Basis $(lpad(b, 2, '0'))"
case_table(dir, name) = collect(Tables.namedtupleiterator(
    CSV.File(joinpath(dir, name); delim='\t')))

function save_case_panels(output, name, panels; title, size=(1300, 460), subtitles=nothing)
    fig = Figure(; size, fontsize=15)
    Label(fig[0, 1:length(panels)], title; fontsize=21, font=:bold)
    for (i, panel) in enumerate(panels)
        slot = fig[1, i] = GridLayout()
        isnothing(subtitles) || Label(slot[0, 1], subtitles[i]; fontsize=18, font=:bold)
        sdraw!(slot[1, 1], panel)
    end
    path = joinpath(output, "$name.png")
    save(path, fig; px_per_unit=1.5)
    println("figure\t", path)
    path
end

function case_basis_panels()
    rows = [(; x, value=sin(pi / 3 * (x + 1.5) * j) / sqrt(1.5),
               basis=basis_label(j)) for j in CASE_BASES for x in range(-1.5, 1.5; length=501)]
    basis = (data(rows) * mapping(:x => "Scaled time", :value => "Basis function";
                 color=:basis) * visual(Lines; linewidth=1.5) +
             data((; x=[-1.0, 1.0])) * mapping(:x => "Scaled time") *
                 visual(VLines; color="#777777", linestyle=:dash)) *
        config(axis=(; title="HSGP basis functions"),
               scales=scales(Color=(; palette=CASE_COLORS)))
    # Introductory mathematical illustration, not posterior reconstruction.
    rows = [(; frequency=j,
               sd=exp(-0.25 * (j * exp(r) * pi / 3)^2 + 0.45946926660233633 + 0.5r),
               length="log length = $r") for r in (-2, -1, 0)
            for j in range(1, 20; length=201)]
    spectrum = data(rows) * mapping(:frequency => "Basis frequency",
        :sd => "Prior spectral SD"; color=:length) * visual(Lines; linewidth=2) *
        config(axis=(; title="Squared-exponential spectrum"),
               scales=scales(Color=(; palette=CASE_COLORS)))
    [basis, spectrum]
end

function case_posterior_panels(input, label, observed)
    curves = case_table(input, "$(label)_curves.tsv")
    map(CASE_GPS) do predictor
        rows = sort(filter(r -> r.predictor == predictor, curves); by=r -> r.time)
        isempty(rows) && error("Missing $predictor posterior curves")
        noise = predictor == "log_sigma"
        all(r -> all(isfinite, (r.q05, r.q50, r.q95)) && (!noise || r.q05 > 0), rows) ||
            error("Non-finite posterior interval or non-positive conditional SD")
        brm_posteriorplot(rows; xlabel="Time after impact (ms)",
            ylabel=noise ? "Conditional SD (scaled)" : "Acceleration (scaled)",
            observations=noise ? nothing : observed, observed_y=:acceleration_scaled,
            logscale=noise)
    end
end

"""Adapt exported draw tables to BRM's native frame transport; never thin draws."""
function case_pair_rows(input, label, predictor; from=zeros(20), to=from)
    rows = case_table(input, "$(label)_$(predictor)_weights.tsv")
    columns = [sort(filter(r -> r.basis == b, rows); by=r -> r.draw) for b in CASE_BASES]
    indices = getproperty.(first(columns), :draw)
    isempty(indices) && error("No saved coordinates")
    length(unique(indices)) == length(indices) || error("Duplicate saved draw IDs")
    all(col -> getproperty.(col, :draw) == indices, columns) ||
        error("Basis columns do not contain identical saved draws")
    coordinates = hcat([getproperty.(col, :coordinate) for col in columns]...)
    log_scales = hcat([getproperty.(col, :log_spectral_scale) for col in columns]...)
    transformed = hsgp_transform_draws(coordinates, log_scales;
        from=from[collect(CASE_BASES)], to=to[collect(CASE_BASES)])
    all(transformed.finite) || error("Non-finite transformed coordinates; no points were dropped")
    output = [(; basis_label=basis_label(b),
               parameter=hyper == :sigma ? "Marginal SD" : "Length scale",
               hyperparameter=getproperty(columns[j][i], hyper),
               coordinate=transformed.coordinates[i, j])
              for (j, b) in enumerate(CASE_BASES) for hyper in (:sigma, :rho)
              for i in eachindex(indices)]
    all(r -> isfinite(r.hyperparameter) && r.hyperparameter > 0, output) ||
        error("Invalid hyperparameters for logarithmic pair-plot axis")
    println("pair_data\t", label, "\t", predictor, "\tdraws_per_panel=", length(indices),
            "\trows=", length(output))
    output
end

function case_loss_rows(input)
    loss = case_table(input, "loss_profiles.tsv")
    output = NamedTuple[]
    for predictor in CASE_GPS, b in CASE_BASES
        curve = sort(filter(r -> r.predictor == predictor && r.basis == b, loss);
                     by=r -> r.centeredness)
        admissible(r) = isfinite(r.loss) && r.admissible == true
        ok = filter(admissible, curve)
        isempty(ok) && error("No admissible losses for $predictor basis $b")
        low, high = extrema(getproperty.(ok, :loss))
        segment = 1
        for r in curve
            if !admissible(r)
                segment += 1 # Never connect across an inadmissible interval.
                continue
            end
            push!(output, (; predictor=gp_label(predictor), basis_label=basis_label(b),
                segment="$predictor-$b-$segment", centeredness=r.centeredness,
                loss=high == low ? 0.0 : (r.loss - low) / (high - low)))
        end
    end
    output
end

function plot_results(input; online_input=input, output=joinpath(input, "figures"))
    mkpath(output)
    observed = case_table(input, "observations.tsv")
    selected = sort(case_table(input, "centeredness.tsv"); by=r -> r.basis)
    getproperty.(selected, :basis) == collect(1:20) || error("Expected all 20 selected basis entries")
    profiles = Dict("mu" => getproperty.(selected, :mean),
                    "log_sigma" => getproperty.(selected, :log_scale))
    paths = String[]
    push!(paths, save_case_panels(output, "hsgp_basis", case_basis_panels();
        title="HSGP basis and prior spectrum"))
    for (label, title, dir) in (("noncentered", "Noncentered pilot", input),
                               ("partial", "Fresh selected-partial fit", input),
                               ("online", "Online adaptive centering", online_input))
        isfile(joinpath(dir, "$(label)_curves.tsv")) || continue
        push!(paths, save_case_panels(output, "$(label)_posterior",
            case_posterior_panels(dir, label, observed); title))
    end
    for (label, geometry, title) in (
            ("noncentered", "noncentered", "Noncentered pilot coordinates"),
            ("noncentered", "centered", "Centered geometry — transformed pilot draws, not another fit"),
            ("noncentered", "optimal", "Selected partial geometry — transformed pilot draws"),
            ("partial", "partial", "Selected-partial refit — newly sampled coordinates"))
        isfile(joinpath(input, "$(label)_mu_weights.tsv")) || continue
        panels = map(CASE_GPS) do predictor
            from = label == "partial" ? profiles[predictor] : zeros(20)
            to = geometry == "centered" ? ones(20) :
                 geometry == "optimal" ? profiles[predictor] : from
            brm_pairplot(case_pair_rows(input, label, predictor; from, to))
        end
        push!(paths, save_case_panels(output, "$(geometry)_scatter", panels;
            title, size=(1500, 1350), subtitles=gp_label.(CASE_GPS)))
    end
    push!(paths, save_case_panels(output, "loss_profiles",
        [brm_centering_lossplot(case_loss_rows(input);
             ylabel="Offline loss (each curve rescaled to [0, 1])")];
        title="Post-hoc pilot selection — offline objective"))
    centeredness = [(; basis=r.basis, predictor=gp_label(p),
                      centeredness=profiles[p][r.basis]) for p in CASE_GPS for r in selected]
    push!(paths, save_case_panels(output, "selected_centeredness",
        [brm_centerednessplot(centeredness)]; title="One centering per basis function",
        size=(1000, 470)))
    if isfile(joinpath(online_input, "online_centeredness.tsv"))
        online = case_table(online_input, "online_centeredness.tsv")
        learned = Dict(p => getproperty.(sort(filter(r -> r.predictor == p, online);
                                             by=r -> r.basis), :centeredness) for p in CASE_GPS)
        all(p -> length(learned[p]) == 20, CASE_GPS) || error("Incomplete online centering profile")
        compared = vcat([merge(r, (; configuration="Post-hoc pilot")) for r in centeredness],
            [(; basis=r.basis, predictor=gp_label(r.predictor), centeredness=r.centeredness,
                configuration="Online warmup") for r in online])
        push!(paths, save_case_panels(output, "online_centeredness",
            [brm_centerednessplot(compared; compare=true)];
            title="Online and post-hoc selected centeredness"))
        panels = map(CASE_GPS) do p
            # WarmupHMC returns final draws in the original NCP frame.
            brm_pairplot(case_pair_rows(online_input, "online", p; to=learned[p]))
        end
        push!(paths, save_case_panels(output, "online_scatter", panels;
            title="Online fit — coordinates in learned geometry", size=(1500, 1350),
            subtitles=gp_label.(CASE_GPS)))
    end
    println("render_complete\tfigures=", length(paths), "\tbackend=AlgebraOfVega/CairoMakie")
    paths
end

if abspath(PROGRAM_FILE) == @__FILE__
    1 <= length(ARGS) <= 3 || error("Usage: plot_results.jl OFFLINE_DIR [ONLINE_DIR [OUTPUT_DIR]]")
    plot_results(ARGS[1]; online_input=length(ARGS) >= 2 ? ARGS[2] : ARGS[1],
                 output=length(ARGS) == 3 ? ARGS[3] : joinpath(ARGS[1], "figures"))
end
