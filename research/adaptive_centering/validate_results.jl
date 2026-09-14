# Independent check of saved full-run receipts and the source's literal loss.
# No sampling, native library loading, or dependence on the BRM selector.
using Serialization
using Statistics
using Test

function table(path)
    lines = split.(readlines(path), '\t')
    Dict(first(lines)[j] => [r[j] for r in lines[2:end]] for j in eachindex(first(lines)))
end

function validate_results(output_dir, audit_dir)
    mapping = table(joinpath(audit_dir, "source_coordinate_map.tsv"))
    index = Dict(mapping["brm"] .=> eachindex(mapping["brm"]))
    profiles = table(joinpath(output_dir, "loss_profiles.tsv"))
    selected = table(joinpath(output_dir, "centeredness.tsv"))
    pilot = deserialize(joinpath(output_dir, "noncentered.jls"))
    refit = deserialize(joinpath(output_dir, "partial.jls"))
    @testset "Full source-fit receipts and literal pilot-selection equality" begin
        for fit in (pilot, refit)
            @test fit.complete
            @test fit.seed == 1
            @test fit.requested_draws == 10_000
            @test size(fit.posterior_position, 1) == 44
            @test size(fit.posterior_position, 2) >= 10_000
            @test all(isfinite, fit.posterior_position)
        end
        for (name, prefix, selected_column) in (("mu", "hsgp_x", "mean"), ("log_sigma", "hsgp_log_sigma_x", "log_scale"))
            log_rho = pilot.posterior_position[index["$(prefix)_rho_iso"], :]
            log_sigma = pilot.posterior_position[index["$(prefix)_sigma"], :]
            for j in 1:20
                z = pilot.posterior_position[index["$(prefix)_beta_raw.$j"], :]
                # Literal source model formula, not the runner's spectral helper.
                log_s = @. -.25 * (j * exp(log_rho) * pi / 3)^2 +
                    log_sigma + .45946926660233633 + .5 * log_rho
                cs = collect(0:0.01:1)
                raw = [log(std(z .* exp.(c .* log_s))) - mean(c .* log_s) for c in cs]
                @test all(isfinite, raw)
                @test cs[argmin(raw)] == parse(Float64, selected[selected_column][j])
                rows = findall(i -> profiles["predictor"][i] == name && parse(Int, profiles["basis"][i]) == j,
                    eachindex(profiles["basis"]))
                @test length(rows) == 101
                for (ci, row) in enumerate(rows)
                    @test parse(Float64, profiles["centeredness"][row]) == cs[ci]
                    # Masked candidates are recorded, never presented as finite scores.
                    if profiles["admissible"][row] == "true"
                        @test isapprox(parse(Float64, profiles["loss"][row]), raw[ci]; atol=1e-10, rtol=1e-10)
                    end
                end
            end
        end
        for name in ("hsgp_basis", "noncentered_posterior", "noncentered_scatter",
                     "centered_scatter", "optimal_scatter", "loss_profiles",
                     "selected_centeredness", "partial_posterior", "partial_scatter")
            path = joinpath(output_dir, "figures", "$name.png")
            @test isfile(path) && filesize(path) > 1000
        end
    end
    println("saved_results_verified\t", output_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: validate_results.jl FIT_OUTPUT SOURCE_AUDIT_OUTPUT")
    validate_results(ARGS...)
end
