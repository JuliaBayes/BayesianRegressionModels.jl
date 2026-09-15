include(joinpath(@__DIR__, "reproduce.jl"))

function read_centeredness(path)
    rows = split.(readlines(path)[2:end], '\t')
    Dict((Symbol(row[1]), parse(Int, row[2])) => parse(Float64, row[3]) for row in rows)
end

function representative_coordinates(selected)
    rows = NamedTuple[]
    for role in (:intercept, :slope)
        counties = collect(1:RADON_DATA.J)
        by_c = sort(counties; by=j -> (selected[(role,j)], j))
        minimum_county = first(by_c)
        maximum_county = first(sort(counties; by=j -> (-selected[(role,j)], j)))
        middle_county = first(sort(filter(j -> j ∉ (minimum_county, maximum_county), counties);
                                  by=j -> (abs(selected[(role,j)] - 0.5), j)))
        for (rank, county) in enumerate((minimum_county, middle_county, maximum_county))
            push!(rows, (; role, rank, county, centeredness=selected[(role,county)],
                criterion=("Minimum", "Nearest 0.5", "Maximum")[rank]))
        end
    end
    rows
end
representative_counties(representatives, role) = [r.county for r in representatives if r.role == role]

function display_coordinate_and_gradient(value, gradient, log_scale, from, to)
    rule = WarmupHMC.Reparametrization(
        WarmupHMC.PartiallyCentered(to), WarmupHMC.PartiallyCentered(from),
        0.0, log_scale)
    _, coordinate, displayed_gradient = WarmupHMC.reparam(
        rule, value, gradient, [value, gradient, log_scale])
    coordinate, displayed_gradient
end

function frame_by_index(centeredness, blocks)
    out = Dict{Int,Float64}()
    for entry in blocks, county in axes(entry.block.effects, 2)
        out[entry.block.effects[1, county]] = centeredness[(entry.role, county)]
    end
    out
end

# WarmupHMC return values are model coordinates. Checkpoints retain sampler
# source coordinates and the controls needed to reconstruct the model frame.
function assert_stored_frames(pilot, partial, online)
    pilot.nonlinear_adapt && !partial.nonlinear_adapt && online.nonlinear_adapt ||
        error("unexpected centering adaptation configuration")
end

function assert_scalar_blocks(entries)
    for entry in entries
        size(entry.block.effects, 1) == 1 || error(
            "adaptive display math assumes single-margin blocks with zero " *
            "location; block $(entry.role) has $(size(entry.block.effects, 1)) terms")
    end
end

function map_refit_to_model(problem, raw)
    mapped = copy(raw)
    WarmupHMC.reparametrize!(problem, mapped)
    mapped
end

# (a) Every mapped coordinate against an independent scalar re-derivation
# from the raw saved source draws (not via reparametrize!).
function verify_mapped_against_raw(entries, selected, raw, mapped)
    for entry in entries, county in axes(entry.block.effects, 2)
        index = entry.block.effects[1, county]
        log_index = only(entry.block.log_scales)
        c = selected[index]
        u = vec(raw[index, :])
        ls = vec(raw[log_index, :])
        expected = u .* exp.(-c .* ls)
        maximum(abs.(vec(mapped[index, :]) .- expected)) < 1e-9 || error(
            "mapped model coordinate disagrees with the scalar frame formula " *
            "for $(entry.role) county $county")
    end
end

# (b) Every refit pair-table coordinate against the raw saved source draws.
# Pair rows carry no draw field, so groups are recovered in construction
# order (entries x representative counties x draws 1:N), asserting each
# row's labels as we go.
function verify_refit_pairs(pairs, entries, raw, representatives)
    per_config = length(representatives) * N_DRAWS
    offset = 3 * per_config
    i = offset
    for entry in entries, county in representative_counties(representatives, entry.role)
        index = entry.block.effects[1, county]
        label = "County $(lpad(county, 3, '0'))"
        u = vec(raw[index, :])
        for draw in 1:N_DRAWS
            i += 1
            row = pairs[i]
            row.configuration == "4 post-hoc fit" ||
                error("refit pair block misaligned at row $i")
            row.role == entry.role && row.basis_label == label ||
                error("refit pair labels misaligned at row $i")
            abs(row.coordinate - u[draw]) < 1e-9 || error(
                "refit pair coordinate disagrees with raw source draws at row $i")
        end
    end
    i == offset + per_config || error("refit pair block has unexpected length")
end

# (c) Every refit gradient-table value against the fixed-partial density
# evaluated at the actual saved source draws. The display configuration
# equals the frozen source frame, so displayed gradients must equal fixed
# source gradients cellwise (the same relationship density_invariants
# proves at the pilot point).
function verify_refit_gradients(gradients, fixed, entries, raw, representatives)
    cells = Dict{Tuple{Symbol,String},Int}()
    for entry in entries, county in representative_counties(representatives, entry.role)
        label = "County $(lpad(county, 3, '0'))"
        cells[(entry.role, label)] = entry.block.effects[1, county]
    end
    by_draw = Dict{Int,Vector{Float64}}()
    for row in gradients
        row.configuration == "2 post-hoc" || continue
        startswith(row.basis_label, "County ") ||
            error("unexpected gradient basis label $(row.basis_label)")
        index = cells[(Symbol(row.role), row.basis_label)]
        g = get!(by_draw, row.draw) do
            _, gg = LogDensityProblems.logdensity_and_gradient(
                fixed, collect(vec(raw[:, row.draw])))
            Vector{Float64}(gg)
        end
        abs(row.gradient - g[index]) < 1e-8 || error(
            "refit displayed gradient disagrees with fixed-partial density " *
            "at draw $(row.draw), cell $index")
    end
    isempty(by_draw) && error("no refit gradient rows found for verification")
end

# (d) Fixed-partial density plus Jacobian at saved refit draws: the fixed
# problem at raw source u must equal the NCP problem at mapped z plus the
# transport log-Jacobian, draw by draw.
function verify_refit_density(fixed, ncp_density, raw, mapped)
    for draw in round.(Int, range(1, size(raw, 2); length=25))
        u = collect(vec(raw[:, draw]))
        value, _ = LogDensityProblems.logdensity_and_gradient(fixed, u)
        rule_ljac, ymapped = WarmupHMC.reparametrizer(fixed)(u)
        maximum(abs.(ymapped .- vec(mapped[:, draw]))) < 1e-9 ||
            error("fixed-partial source did not round-trip at refit draw $draw")
        target = LogDensityProblems.logdensity(ncp_density, collect(vec(mapped[:, draw])))
        abs(value - (target + rule_ljac)) < 1e-7 ||
            error("fixed-partial density/Jacobian mismatch at refit draw $draw")
    end
end

# Reconstruct Stan's propto target: retain scale-dependent terms and Jacobians.
function verify_physical_target(ncp_density, unc_names, entries, mapped)
    original_of = Dict{String,String}()
    for line in readlines(joinpath(RESEARCH_DIR, "results", "source_coordinate_map.tsv"))[2:end]
        brm, original = split(line, '\t')
        original_of[original] = brm
    end
    pos = Dict(name => i for (i, name) in enumerate(unc_names))
    at(original) = pos[original_of[original]]
    i_mu_a = at("mu_alpha")
    i_mu_b = at("mu_beta")
    i_sig_a = at("sigma_alpha")
    i_sig_b = at("sigma_beta")
    i_sig_y = at("sigma_y")
    for draw in round.(Int, range(1, size(mapped, 2); length=10))
        q = vec(mapped[:, draw])
        mu_a, mu_b = q[i_mu_a], q[i_mu_b]
        sig_a, sig_b, sig_y = exp(q[i_sig_a]), exp(q[i_sig_b]), exp(q[i_sig_y])
        total = -0.5abs2(mu_a / 10) - 0.5abs2(mu_b / 10) +
            (-0.5abs2(sig_a) + q[i_sig_a]) +
            (-0.5abs2(sig_b) + q[i_sig_b]) +
            (-0.5abs2(sig_y) + q[i_sig_y])
        mu_vec = mu_a .+ mu_b .* RADON_DATA.floor_measure
        for entry in entries
            z_index = [entry.block.effects[1, county] for county in 1:RADON_DATA.J]
            sig = entry.role === :intercept ? sig_a : sig_b
            w = sig .* q[z_index]
            total += sum(-0.5abs2.(q[z_index]))
            if entry.role === :intercept
                mu_vec .+= w[RADON_DATA.county_idx]
            else
                mu_vec .+= w[RADON_DATA.county_idx] .* RADON_DATA.floor_measure
            end
        end
        resid = (RADON_DATA.log_radon .- mu_vec) ./ sig_y
        total += sum(-0.5abs2.(resid)) - length(resid) * log(sig_y)
        actual = LogDensityProblems.logdensity(ncp_density, collect(q))
        abs(total - actual) / max(1.0, abs(actual)) < 1e-9 || error(
            "hand-rebuilt physical target disagrees with Stan at refit draw $draw")
    end
end

function pair_rows(configuration, evidence, posterior, display_c, entries, representatives)
    rows = NamedTuple[]
    for entry in entries, county in representative_counties(representatives, entry.role)
        index = entry.block.effects[1, county]
        log_index = only(entry.block.log_scales)
        scale = exp.(vec(posterior[log_index, :]))
        z = vec(posterior[index, :])
        coordinate = [display_coordinate_and_gradient(
            z[i], 0.0, log(scale[i]), 0.0, display_c[index])[1] for i in eachindex(z)]
        label = "County $(lpad(county, 3, '0'))"
        append!(rows, [(; configuration, evidence, role=entry.role, county,
            basis_label=label, parameter=entry.role === :intercept ?
                "Intercept scale" : "Slope scale",
            hyperparameter=scale[i], coordinate=coordinate[i])
            for i in eachindex(z)])
    end
    rows
end

function gradient_rows(configuration, evidence, posterior, display_c, entries, representatives)
    displayed = round.(Int, range(1, size(posterior, 2); length=1000))
    length(unique(displayed)) == 1000 ||
        error("display draw selection duplicated an index")
    rows = NamedTuple[]
    for draw in displayed
        q = posterior[:, draw]
        _, gradient = LogDensityProblems.logdensity_and_gradient(
            DIAGNOSTIC_DENSITY[], collect(q))
        all(isfinite, gradient) || error("non-finite diagnostic gradient at draw $draw")
        for entry in entries, county in representative_counties(representatives, entry.role)
            index = entry.block.effects[1, county]
            log_index = only(entry.block.log_scales)
            label = "County $(lpad(county, 3, '0'))"
            coordinate, displayed_gradient = display_coordinate_and_gradient(
                q[index], gradient[index], q[log_index], 0.0, display_c[index])
            push!(rows, (; configuration, evidence, role=entry.role, county,
                basis_label=label, draw, coordinate, gradient=displayed_gradient))
        end
    end
    rows
end

const DIAGNOSTIC_DENSITY = Ref{Any}()

function density_invariants(entries, selected, output_dir, representatives)
    fixed = fixed_partial_problem(
        DIAGNOSTIC_STAN[].sb, DIAGNOSTIC_DENSITY[], selected)
    target = DIAGNOSTIC_TARGET[][:, 1]
    target_gradient = last(LogDensityProblems.logdensity_and_gradient(
        DIAGNOSTIC_DENSITY[], collect(target)))
    source = copy(target)
    expected_ljac = 0.0
    for entry in entries, county in 1:RADON_DATA.J
        index = entry.block.effects[1, county]
        log_scale = target[only(entry.block.log_scales)]
        c = selected[index]
        source[index] = display_coordinate_and_gradient(
            target[index], 0.0, log_scale, 0.0, c)[1]
        expected_ljac -= c * log_scale
    end
    source_value, source_gradient = LogDensityProblems.logdensity_and_gradient(
        fixed, collect(source))
    target_value = LogDensityProblems.logdensity(DIAGNOSTIC_DENSITY[], collect(target))
    rule_ljac, mapped_target = WarmupHMC.reparametrizer(fixed)(collect(source))
    maximum(abs.(mapped_target .- target)) < 1e-9 ||
        error("fixed-partial source did not round-trip to the target frame")
    abs(rule_ljac - expected_ljac) < 1e-8 ||
        error("fixed-partial Jacobian disagrees with the scalar frame formula")
    checks = NamedTuple[]
    density_error = abs(source_value - (target_value + rule_ljac))
    density_error < 1e-7 || error("fixed-partial density/Jacobian mismatch")
    for entry in entries, county in representative_counties(representatives, entry.role)
        index = entry.block.effects[1, county]
        log_index = only(entry.block.log_scales)
        log_scale = target[log_index]
        c = selected[index]
        displayed, displayed_gradient = display_coordinate_and_gradient(
            target[index], target_gradient[index], log_scale, 0.0, c)
        gradient_error = abs(source_gradient[index] - displayed_gradient)
        h = 1e-5 / max(1.0, abs(displayed_gradient))
        function displayed_density(value)
            q = copy(source)
            q[index] = value
            LogDensityProblems.logdensity(fixed, collect(q))
        end
        fd = (displayed_density(displayed + h) - displayed_density(displayed - h)) / 2h
        gradient_fd_error = abs(fd - displayed_gradient)
        isfinite(source_value) || error("non-finite fixed-partial density")
        gradient_error < 1e-8 || error("fixed-partial target gradient mismatch")
        gradient_fd_error / max(1.0, abs(displayed_gradient)) < 1e-5 ||
            error("displayed-gradient finite-difference mismatch")
        push!(checks, (; role=entry.role, county, draw=1, centeredness=c,
            density_absolute_error=density_error,
            target_gradient_absolute_error=gradient_error,
            displayed_gradient_fd_error=gradient_fd_error,
            roundtrip_absolute_error=maximum(abs.(mapped_target .- target))))
    end
    write_tsv(joinpath(output_dir, "density_jacobian_gradient_invariants.tsv"), checks)
    checks
end

function export_ppc(output_dir)
    descriptor = brm_descriptor(DIAGNOSTIC_STAN[].sb)
    predicted = brm_predictive_draws(
        descriptor, permutedims(DIAGNOSTIC_TARGET[]); problem=DIAGNOSTIC_DENSITY[], seed=SEED)
    matrix = predicted.log_radon
    size(matrix) == (size(DIAGNOSTIC_TARGET[], 2), RADON_DATA.N) ||
        error("native PPC returned an unexpected shape: $(size(matrix))")
    all(isfinite, matrix) || error("native PPC returned non-finite draws")
    rows = map(eachindex(RADON_DATA.floor_measure)) do i
        values = view(matrix, :, i)
        q05, q10, q25, q50, q75, q90, q95 = quantile(values, (
            0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95))
        (; index=i, county=RADON_DATA.county_idx[i],
           floor=RADON_DATA.floor_measure[i], observation=RADON_DATA.log_radon[i],
           q05, q10, q25, q50, q75, q90, q95)
    end
    write_tsv(joinpath(output_dir, "ppc_curves.tsv"), rows)
    rows
end

const DIAGNOSTIC_STAN = Ref{Any}()
const DIAGNOSTIC_TARGET = Ref{Matrix{Float64}}()

function prepare_diagnostics(offline_dir, online_dir, output_dir)
    mkpath(output_dir)
    pilot = deserialize(joinpath(offline_dir, "noncentered.jls"))
    partial = deserialize(joinpath(offline_dir, "partial.jls"))
    online = deserialize(joinpath(online_dir, "online.jls"))
    all(f -> f.complete && size(f.posterior_position, 2) == N_DRAWS,
        (pilot, partial, online)) || error("an input fit is incomplete")
    assert_stored_frames(pilot, partial, online)
    # Fail closed on a producer/checkout mismatch BEFORE any output write.
    # `stan_density` below already writes the diagnostics Stan source and its
    # compiled target, so the guard must precede it — a stale regeneration
    # checkout must never rewrite anything first.
    producer = TOML.parsefile(joinpath(offline_dir, "provenance.toml"))
    checkout_script = bytes2hex(sha256(read(joinpath(RESEARCH_DIR, "reproduce.jl"))))
    producer_script = producer["script_sha256"]
    checkout_script == producer_script || error(
        "regeneration checkout reproduce.jl differs from the producer script " *
        "that saved these draws (producer sha256 $producer_script); rerun " *
        "this script from a checkout carrying that producer source instead " *
        "— do not resample the fits")
    stan = stan_density("diagnostics", output_dir)
    read(joinpath(output_dir, "radon-diagnostics.stan")) ==
        read(joinpath(offline_dir, "radon-noncentered.stan")) ||
        error("offline producer Stan source differs")
    read(joinpath(online_dir, "radon-online.stan")) ==
        read(joinpath(offline_dir, "radon-noncentered.stan")) ||
        error("online producer Stan source differs")
    DIAGNOSTIC_STAN[] = stan
    DIAGNOSTIC_DENSITY[] = stan.density
    DIAGNOSTIC_TARGET[] = pilot.posterior_position
    unc_names = String.(BS.param_unc_names(stan.density.model))
    entries = effect_blocks(stan.sb, unc_names)
    assert_scalar_blocks(entries)
    offline_selected = read_centeredness(joinpath(offline_dir, "selected_centeredness.tsv"))
    representatives = representative_coordinates(offline_selected)
    write_tsv(joinpath(output_dir, "representative_coordinates.tsv"), representatives)
    online_selected = read_centeredness(joinpath(online_dir, "online_centeredness.tsv"))
    offline_by_index = frame_by_index(offline_selected, entries)
    online_by_index = frame_by_index(online_selected, entries)
    zero_by_index = Dict(index => 0.0 for index in keys(offline_by_index))
    one_by_index = Dict(index => 1.0 for index in keys(offline_by_index))
    pilot_model = pilot.posterior_position
    online_model = online.posterior_position
    raw_partial = deserialize(joinpath(offline_dir, "checkpoints-partial", "cp_latest.jls")).posterior_position
    refit_problem = fixed_partial_problem(stan.sb, stan.density, offline_by_index)
    refit_model = partial.posterior_position
    checkpoint_model = map_refit_to_model(refit_problem, raw_partial)
    maximum(abs.(checkpoint_model .- refit_model)) < 1e-9 ||
        error("returned model positions disagree with the checkpoint source mapping")
    serialize(joinpath(output_dir, "partial_model_frame.jls"),
        (; posterior_position=refit_model,
           mapped_from="partial.jls returned model coordinates",
           mapping="identity; checked against the final checkpoint source coordinates",
           n_divergent_samples=partial.n_divergent_samples))

    pairs = vcat(
        pair_rows("1 NCP", "pilot transformed display", pilot_model, zero_by_index, entries, representatives),
        pair_rows("2 centered", "pilot transformed display", pilot_model, one_by_index, entries, representatives),
        pair_rows("3 post-hoc selected", "pilot transformed display", pilot_model,
                  offline_by_index, entries, representatives),
        pair_rows("4 post-hoc fit", "fresh fit transformed display", refit_model,
                  offline_by_index, entries, representatives),
        pair_rows("5 online learned", "fresh fit transformed display", online_model,
                  online_by_index, entries, representatives))
    gradients = vcat(
        gradient_rows("1 Centered", "pilot transformed display", pilot_model, one_by_index, entries, representatives),
        gradient_rows("2 post-hoc", "fresh fit", refit_model, offline_by_index, entries, representatives),
        gradient_rows("3 online", "fresh fit", online_model, online_by_index, entries, representatives))
    verify_mapped_against_raw(entries, offline_by_index, raw_partial, refit_model)
    verify_refit_pairs(pairs, entries, raw_partial, representatives)
    verify_refit_gradients(gradients, refit_problem, entries, raw_partial, representatives)
    verify_refit_density(refit_problem, stan.density, raw_partial, refit_model)
    verify_physical_target(stan.density, unc_names, entries, refit_model)
    write_tsv(joinpath(output_dir, "coordinate_pairs.tsv"), pairs)
    write_tsv(joinpath(output_dir, "coordinate_gradients.tsv"), gradients)
    density_invariants(entries, offline_by_index, output_dir, representatives)
    ppc = export_ppc(output_dir)
    # Diagnostics use the same model coordinates for every arm.
    corrected = [diagnostics("noncentered", pilot),
        diagnostics("selected_partial",
            (; posterior_position=refit_model,
               n_divergent_samples=partial.n_divergent_samples)),
        diagnostics("online", online)]
    write_tsv(joinpath(offline_dir, "diagnostics.tsv"), corrected)
    # The producer/checkout equality was already enforced before any output
    # write above; checkout_script and producer_script are reused here.
    open(joinpath(output_dir, "diagnostics_provenance.toml"), "w") do io
        TOML.print(io, Dict(
            "posteriordb_revision" => POSTERIORDB_REVISION,
            "posterior_name" => POSTERIOR_NAME,
            "diagnostics_commit" => strip(read(`git -C $RESEARCH_DIR rev-parse HEAD`, String)),
            "diagnostics_script_sha256" => bytes2hex(sha256(read(@__FILE__))),
            "reproduce_checkout_sha256" => checkout_script,
            "reproduce_checkout_note" => "hash of reproduce.jl in this checkout, not provenance by itself",
            "reproduce_producer_sha256" => producer_script,
            "reproduce_producer_note" => "script_sha256 from the immutable run provenance; must equal the checkout hash above",
            "refit_stored_frame" => "model NCP z (partial.jls)",
            "refit_model_frame" => "NCP z, independently verified against checkpoint partial-u",
            "refit_diagnostics_frame" => "model-frame ESS on returned draws",
            "draws_per_configuration" => N_DRAWS,
            "gradient_display_draws_per_facet" => 1000,
            "representative_coordinates" => [Dict(string(k) => v isa Symbol ? string(v) : v for (k,v) in Base.pairs(r)) for r in representatives],
            "coordinate_selection" => "per role: minimum offline centeredness, nearest 0.5, maximum; ties by county index; distinct coordinates",
            "visual_baseline" => "centered transformation of NCP pilot; sampling and cost baseline stays NCP",
            "pair_evidence_modes" => ["pilot_transformed_display", "fresh_fit_transformed_display"],
            "native_ppc_seed" => SEED,
            "ppc_rows" => length(ppc),
        ))
    end
    println("radon_diagnostics_complete\t", output_dir,
            "\tpair_rows=", length(pairs),
            "\tgradient_rows=", length(gradients),
            "\tinvariants=", 6,
            "\tppc_rows=", length(ppc))
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || error("usage: prepare_diagnostics.jl OFFLINE_DIR ONLINE_DIR OUTPUT_DIR")
    prepare_diagnostics(ARGS...)
end
