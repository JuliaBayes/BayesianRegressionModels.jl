include(joinpath(@__DIR__, "reproduce.jl"))
using LogDensityProblems
using Serialization
using Statistics
using Test

function table(path)
    lines = split.(readlines(path), '\t')
    header = first(lines)
    Dict(name => [row[j] for row in lines[2:end]]
         for (j, name) in enumerate(header))
end

function number_column(table, name)
    values = table[name]
    all(v -> v != "missing", values) || return Union{Missing,Float64}[
        v == "missing" ? missing : parse(Float64, v) for v in values]
    parse.(Float64, values)
end

function fit_record(dir, name; checkpoint=name)
    fit = deserialize(joinpath(dir, "$name.jls"))
    @test fit.complete && fit.seed == SOURCE_SEED
    @test fit.requested_draws == SOURCE_DRAWS
    @test size(fit.posterior_position) == (10, 10_000)
    @test all(isfinite, fit.posterior_position)
    @test 0 < fit.sampling_gradient_evaluations <= fit.total_gradient_evaluations
    checkpoint = deserialize(joinpath(dir, "checkpoints-$checkpoint", "cp_latest.jls"))
    @test checkpoint.total_evaluation_counter == fit.total_gradient_evaluations
    @test checkpoint.sampling_evaluation_counter == fit.sampling_gradient_evaluations
    @test size(checkpoint.posterior_position, 2) == size(fit.posterior_position, 2)
    fit
end

function coordinate_table(dir, label)
    rows = table(joinpath(dir, "$(label == "selected_partial" ? "partial" : label)_coordinates.tsv"))
    mu = Vector{Float64}(undef, 10_000)
    tau = Vector{Float64}(undef, 10_000)
    z = Matrix{Float64}(undef, 10_000, 8)
    physical = Matrix{Float64}(undef, 10_000, 8)
    for row in eachindex(rows["draw"])
        draw = parse(Int, rows["draw"][row])
        school = parse(Int, rows["school"][row])
        z[draw, school] = parse(Float64, rows["noncentered_coordinate"][row])
        mu[draw] = parse(Float64, rows["mu"][row])
        tau[draw] = parse(Float64, rows["tau"][row])
        physical[draw, school] = parse(Float64, rows["theta_effect"][row])
    end
    @test all(isfinite, z) && all(isfinite, tau) && all(isfinite, mu)
    @test physical ≈ mu .+ z .* tau atol=1e-10
    (; mu, z, tau, physical)
end

function validate_results(fit_dir, diagnostics_dir)
    source = table(joinpath(fit_dir, "saved_source_density_gradient_audit.tsv"))
    diagnostics = table(joinpath(fit_dir, "diagnostics.tsv"))
    costs = table(joinpath(fit_dir, "fit_costs.tsv"))
    workflow = table(joinpath(fit_dir, "workflow_costs.tsv"))
    offline_centeredness = table(joinpath(fit_dir, "offline_centeredness.tsv"))
    offline_losses = table(joinpath(fit_dir, "offline_loss_profiles.tsv"))
    online_centeredness = table(joinpath(fit_dir, "online_centeredness.tsv"))
    gradient_checks = table(joinpath(diagnostics_dir, "gradient_checks.tsv"))
    frame_checks = table(joinpath(diagnostics_dir, "frame_invariants.tsv"))
    online_losses = table(joinpath(diagnostics_dir, "retrospective_online_losses.tsv"))

    pilot = fit_record(fit_dir, "noncentered")
    partial_source = fit_record(fit_dir, "partial")
    partial_target = fit_record(fit_dir, "partial_target"; checkpoint="partial")
    online = fit_record(fit_dir, "online")
    pilot_coordinates = coordinate_table(fit_dir, "noncentered")

    @testset "Eight-schools full-run receipts and literal selection" begin
        @test maximum(number_column(source, "density_absolute_error")) < 1e-9
        @test maximum(number_column(source, "max_gradient_absolute_error")) < 1e-8
        @test length(source["draw"]) == 16

        candidates = collect(0.0:0.01:1.0)
        @test length(offline_losses["school"]) == 8 * 101
        for school in 1:8
            selected = parse(Float64, offline_centeredness["centeredness"][school])
            z = pilot_coordinates.z[:, school]
            logtau = log.(pilot_coordinates.tau)
            raw = [log(std(z .* exp.(c .* logtau))) - mean(c .* logtau)
                   for c in candidates]
            @test all(isfinite, raw)
            @test candidates[argmin(raw)] == selected
            rows = findall(i -> parse(Int, offline_losses["school"][i]) == school,
                           eachindex(offline_losses["school"]))
            @test length(rows) == 101
            for (candidate_index, row) in enumerate(rows)
                @test parse(Float64, offline_losses["centeredness"][row]) ==
                    candidates[candidate_index]
                @test parse(Float64, offline_losses["loss"][row]) ≈
                    raw[candidate_index] atol=1e-10 rtol=1e-10
                @test offline_losses["admissible"][row] == "true"
            end
        end

        # The fresh partial fit's serialized source and target positions must
        # obey the selected scalar transform exactly.
        for school in 1:8
            c = parse(Float64, offline_centeredness["centeredness"][school])
            @test partial_source.posterior_position[school+1, :] ≈
                partial_target.posterior_position[school+1, :] .*
                exp.(c .* partial_target.posterior_position[1, :]) atol=1e-10
        end
        # The partial coordinates table is exported from the back-transformed
        # (model-frame) fit, so physical theta is mu + tau*z there. The same
        # physical effects must equal mu + tau^(1-c)*u against the
        # source-frame binary: one quantity, both frames, both binaries.
        partial_coordinates = coordinate_table(fit_dir, "partial")
        for school in 1:8
            c = parse(Float64, offline_centeredness["centeredness"][school])
            u = partial_source.posterior_position[school + 1, :]
            @test partial_coordinates.physical[:, school] ≈
                partial_coordinates.mu .+
                u .* partial_coordinates.tau .^ (1 - c) atol=1e-8
        end
        @test all(0 .<= parse.(Float64, online_centeredness["centeredness"]) .<= 1)

        for (row, fit) in zip(eachindex(diagnostics["fit"]),
                              (pilot, partial_target, online))
            @test parse(Int, diagnostics["retained_draws"][row]) == 10_000
            @test parse(Int, diagnostics["divergences"][row]) ==
                fit.n_divergent_samples
            @test parse(Int, diagnostics["total_gradient_evaluations"][row]) ==
                fit.total_gradient_evaluations
            @test parse(Int, diagnostics["sampling_gradient_evaluations"][row]) ==
                fit.sampling_gradient_evaluations
            @test diagnostics["ess_coordinate_scope"][row] ==
                "mu, log(tau), theta_effect_1:8"
        end
        @test parse(Int, workflow["total_evaluation_counter"][1]) ==
            pilot.total_gradient_evaluations + partial_target.total_gradient_evaluations
        @test parse(Int, workflow["sampling_evaluation_counter"][1]) ==
            pilot.sampling_gradient_evaluations +
            partial_target.sampling_gradient_evaluations
        @test length(costs["fit"]) == 3

        @test length(gradient_checks["configuration"]) == 72
        @test maximum(number_column(gradient_checks, "relative_error")) < 1e-4
        @test maximum(number_column(frame_checks, "position_gradient_product_error")) < 1e-12
        @test maximum(number_column(frame_checks, "physical_effect_error")) < 1e-12
        @test length(online_losses["school"]) == 88
        @test all(online_losses["objective"] .== "position_gradient_correlation_w1_0")
        @test all(online_losses["evidence"] .== "retrospective_saved_pilot_draws")
        @test all(online_losses["weights"] .== "unit")
    end
    # Displayed rows must come from their NAMED fit, not from a shared pilot
    # passed through coordinate transforms. The .jls binaries are the
    # independent anchor here: row 1 holds log(tau), rows 2-9 the eight
    # school coordinates in school order, row 10 mu (same layout the
    # source/target transform check above relies on).
    pairs = table(joinpath(diagnostics_dir, "coordinate_pairs.tsv"))
    scatters = table(joinpath(diagnostics_dir, "gradient_scatter.tsv"))
    @testset "Plotted rows match their named fits" begin
        @test length(pairs["configuration"]) == 240_000
        @test length(scatters["configuration"]) == 24_000
        learned = [parse(Float64, online_centeredness["centeredness"][j])
                   for j in 1:8]
        selected_c = [parse(Float64, offline_centeredness["centeredness"][j])
                      for j in 1:8]
        named = Dict("NCP" => pilot, "Online" => online)
        pair_key = Dict{Tuple{String,Int,Int},Float64}()
        for config in ("NCP", "Post-hoc", "Online")
            rows = findall(==(config), pairs["configuration"])
            @test length(rows) == 80_000
            for row in rows
                draw = parse(Int, pairs["draw"][row])
                school = parse(Int, pairs["school"][row])
                coordinate = parse(Float64, pairs["coordinate"][row])
                if config == "Post-hoc"
                    # The displayed refit coordinate is the source draw u,
                    # which must also equal tau^c times the back-transformed
                    # model draw z: both binaries, both frames.
                    u = partial_source.posterior_position[school + 1, draw]
                    z = partial_target.posterior_position[school + 1, draw]
                    tau = exp(partial_target.posterior_position[1, draw])
                    @test parse(Float64, pairs["hyperparameter"][row]) ≈ tau
                    @test coordinate ≈ u
                    @test coordinate ≈ tau^selected_c[school] * z
                else
                    fit = named[config]
                    tau = exp(fit.posterior_position[1, draw])
                    @test parse(Float64, pairs["hyperparameter"][row]) ≈ tau
                    expected = if config == "Online"
                        tau^learned[school] *
                            fit.posterior_position[school + 1, draw]
                    else
                        fit.posterior_position[school + 1, draw]
                    end
                    @test coordinate ≈ expected
                end
                pair_key[(config, draw, school)] = coordinate
            end
        end
        # Every gradient-scatter row must sit on its pair-table coordinate:
        # this binds the gradient display to the fit-anchored pairs without
        # rebuilding any target.
        for row in eachindex(scatters["configuration"])
            key = (scatters["configuration"][row],
                   parse(Int, scatters["draw"][row]),
                   parse(Int, scatters["school"][row]))
            @test parse(Float64, scatters["coordinate"][row]) ≈ pair_key[key]
        end
    end
    @testset "Refit gradients match the partial problem at source draws" begin
        # Explicit frame check on a small sample: rebuild the fixed
        # selected-partial problem and differentiate it at the ACTUAL
        # source-frame draws from partial.jls. This pins the frame the bulk
        # test above cannot see — written gradients must be du-gradients at
        # u, not gz-gradients at z.
        mktempdir() do tmp
            stan = stan_density("validate", tmp)
            selected = [parse(Float64, offline_centeredness["centeredness"][j])
                        for j in 1:8]
            problem = fixed_partial_problem(stan, selected)
            shown = Dict{Tuple{Int,Int},Float64}()
            for row in eachindex(scatters["configuration"])
                scatters["configuration"][row] == "Post-hoc" || continue
                shown[(parse(Int, scatters["draw"][row]),
                       parse(Int, scatters["school"][row]))] =
                    parse(Float64, scatters["gradient"][row])
            end
            @test length(shown) == 8_000
            present = sort!(unique!(first.(keys(shown))))
            for draw in (first(present), present[length(present) ÷ 2],
                         last(present)), school in (1, 8)
                q = collect(partial_source.posterior_position[:, draw])
                _value, gradient = LogDensityProblems.logdensity_and_gradient(
                    problem, q)
                @test shown[(draw, school)] ≈ gradient[school + 1] atol=1e-10
            end
        end
    end
    println("eight_schools_results_verified\t", fit_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 ||
        error("usage: validate_results.jl FULL_FIT_DIRECTORY DIAGNOSTICS_DIRECTORY")
    validate_results(ARGS...)
end
