include(joinpath(@__DIR__, "reproduce.jl"))

function read_centeredness(path)
    rows = split.(readlines(path)[2:end], '\t')
    Dict((Symbol(row[1]), parse(Int, row[2])) => parse(Float64, row[3]) for row in rows)
end

function representative_counties()
    counties = collect(1:RADON_DATA.J)
    middle = cld(RADON_DATA.J, 2)
    unique!(sort!([first(counties), middle, last(counties)]))
end

function display_coordinate_and_gradient(value, gradient, log_scale, from, to)
    rule = WarmupHMC.Reparametrization(
        WarmupHMC.PartiallyCentered(to), WarmupHMC.PartiallyCentered(from),
        0.0, log_scale)
    _, coordinate, displayed_gradient = WarmupHMC.reparam(
        rule, value, gradient, [value, gradient, log_scale])
    coordinate, displayed_gradient
) end

function frame_by_index(centeredness, blocks)
    out = Dict{Int,Float64}()
    for entry in blocks, county in axes(entry.block.effects, 2)
        out[entry.block.effects[1, county]] = centeredness[(entry.role, county)]
    end
    out
end

function pair_rows(configuration, evidence, fit, display_c, entries)
    rows = NamedTuple[]
    for entry in entries, county in representative_counties()
        index = entry.block.effects[1, county]
        log_index = only(entry.block.log_scales)
        scale = exp.(vec(fit.posterior_position[log_index, :]))
        z = vec(fit.posterior_position[index, :])
        coordinate = [display_coordinate_and_gradient(
            z[i], 0.0, log(scale[i]), 0.0, display_c[index])[1] for i in eachindex(z)]
        label = "County $(lpad(county, 3, '0'))"
        append!(rows, [(; configuration, evidence, role=entry.role,
            basis_label=label, parameter=entry.role === :intercept ?
                "Intercept scale" : "Slope scale",
            hyperparameter=scale[i], coordinate=coordinate[i])
            for i in eachindex(z)])
    end
    rows
end

function gradient_rows(configuration, evidence, fit, display_c, entries)
    displayed = round.(Int, range(1, size(fit.posterior_position, 2); length=1000))
    length(unique(displayed)) == 1000 ||
        error("display draw selection duplicated an index")
    rows = NamedTuple[]
    for draw in displayed
        q = fit.posterior_position[:, draw]
        _, gradient = LogDensityProblems.logdensity_and_gradient(
            DIAGNOSTIC_DENSITY[], collect(q))
        all(isfinite, gradient) || error("non-finite diagnostic gradient at draw $draw")
        for entry in entries, county in representative_counties()
            index = entry.block.effects[1, county]
            log_index = only(entry.block.log_scales)
            label = "County $(lpad(county, 3, '0'))"
            coordinate, displayed_gradient = display_coordinate_and_gradient(
                q[index], gradient[index], q[log_index], 0.0, display_c[index])
            push!(rows, (; configuration, evidence, role=entry.role,
                basis_label=label, draw, coordinate, gradient=displayed_gradient))
        end
    end
    rows
end

const DIAGNOSTIC_DENSITY = Ref{Any}()

function density_invariants(entries, selected, output_dir)
    fixed = fixed_partial_problem(
        DIAGNOSTIC_STAN[].sb, DIAGNOSTIC_DENSITY[], selected)
    checks = NamedTuple[]
    draw = 1
    for entry in entries, county in representative_counties()
        index = entry.block.effects[1, county]
        log_index = only(entry.block.log_scales)
        target = DIAGNOSTIC_TARGET[:, draw]
        target_gradient = last(LogDensityProblems.logdensity_and_gradient(
            DIAGNOSTIC_DENSITY[], collect(target)))
        log_scale = target[log_index]
        c = selected[(entry.role, county)]
        displayed, displayed_gradient = display_coordinate_and_gradient(
            target[index], target_gradient[index], log_scale, 0.0, c)
        source = copy(target)
        source[index] = displayed
        source_value, source_gradient = LogDensityProblems.logdensity_and_gradient(
            fixed, collect(source))
        target_value = LogDensityProblems.logdensity(DIAGNOSTIC_DENSITY[], collect(target))
        ljac = log_scale * (0.0 - c)
        density_error = abs(source_value - (target_value + ljac))
        gradient_error = abs(source_gradient[index] - displayed_gradient)
        h = 1e-5 / max(1.0, abs(displayed_gradient))
        function displayed_density(value)
            mapped = copy(target)
            mapped[index] = display_coordinate_and_gradient(
                value, 0.0, log_scale, 0.0, c)[1]
            LogDensityProblems.logdensity(DIAGNOSTIC_DENSITY[], collect(mapped)) +
                ljac
        end
        fd = (displayed_density(displayed + h) - displayed_density(displayed - h)) / 2h
        gradient_fd_error = abs(fd - displayed_gradient)
        roundtrip = display_coordinate_and_gradient(
            displayed, 0.0, log_scale, c, 0.0)[1]
        isfinite(source_value) || error("non-finite fixed-partial density")
        density_error < 1e-8 || error("fixed-partial density/Jacobian mismatch")
        gradient_error < 1e-8 || error("fixed-partial target gradient mismatch")
        gradient_fd_error / max(1.0, abs(displayed_gradient)) < 1e-5 ||
            error("displayed-gradient finite-difference mismatch")
        abs(roundtrip - target[index]) < 1e-10 || error("centering roundtrip failed")
        push!(checks, (; role=entry.role, county, draw, centeredness=c,
            density_absolute_error=density_error,
            target_gradient_absolute_error=gradient_error,
            displayed_gradient_fd_error=gradient_fd_error,
            roundtrip_absolute_error=abs(roundtrip - target[index])))
    end
    write_tsv(joinpath(output_dir, "density_jacobian_gradient_invariants.tsv"), checks)
    checks
end

function export_ppc(output_dir)
    descriptor = brm_descriptor(last(DIAGNOSTIC_STAN).sb)
    predicted = brm_predictive_draws(
        descriptor, permutedims(DIAGNOSTIC_TARGET); problem=DIAGNOSTIC_DENSITY[], seed=SEED)
    matrix = predicted.log_radon
    size(matrix) == (size(DIAGNOSTIC_TARGET, 2), RADON_DATA.N) ||
        error("native PPC returned an unexpected shape: $(size(matrix))")
    all(isfinite, matrix) || error("native PPC returned non-finite draws")
    rows = map(eachindex(RADON_DATA.floor_measure)) do i
        values = view(matrix, :, i)
        q05, q10, q25, q50, q75, q90, q95 = quantile(values, (
            0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95))
        (; floor=RADON_DATA.floor_measure[i], observation=RADON_DATA.log_radon[i],
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
    offline_selected = read_centeredness(joinpath(offline_dir, "selected_centeredness.tsv"))
    online_selected = read_centeredness(joinpath(online_dir, "online_centeredness.tsv"))
    offline_by_index = frame_by_index(offline_selected, entries)
    online_by_index = frame_by_index(online_selected, entries)
    zero_by_index = Dict(index => 0.0 for index in keys(offline_by_index))
    one_by_index = Dict(index => 1.0 for index in keys(offline_by_index))

    pairs = vcat(
        pair_rows("1 NCP", "pilot transformed display", pilot, zero_by_index, entries),
        pair_rows("2 centered", "pilot transformed display", pilot, one_by_index, entries),
        pair_rows("3 post-hoc selected", "pilot transformed display", pilot,
                  offline_by_index, entries),
        pair_rows("4 post-hoc fit", "fresh fit transformed display", partial,
                  offline_by_index, entries),
        pair_rows("5 online learned", "fresh fit transformed display", online,
                  online_by_index, entries))
    write_tsv(joinpath(output_dir, "coordinate_pairs.tsv"), pairs)
    gradients = vcat(
        gradient_rows("1 NCP", "pilot", pilot, zero_by_index, entries),
        gradient_rows("2 post-hoc", "fresh fit", partial, offline_by_index, entries),
        gradient_rows("3 online", "fresh fit", online, online_by_index, entries))
    write_tsv(joinpath(output_dir, "coordinate_gradients.tsv"), gradients)
    density_invariants(entries, offline_by_index, output_dir)
    ppc = export_ppc(output_dir)
    open(joinpath(output_dir, "diagnostics_provenance.toml"), "w") do io
        TOML.print(io, Dict(
            "posteriordb_revision" => POSTERIORDB_REVISION,
            "posterior_name" => POSTERIOR_NAME,
            "diagnostics_commit" => strip(read(`git -C $RESEARCH_DIR rev-parse HEAD`, String)),
            "diagnostics_script_sha256" => bytes2hex(sha256(read(@__FILE__))),
            "pilot_script_sha256" => bytes2hex(sha256(read(joinpath(RESEARCH_DIR, "reproduce.jl")))),
            "draws_per_configuration" => N_DRAWS,
            "gradient_display_draws_per_facet" => 1000,
            "representative_counties" => representative_counties(),
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
