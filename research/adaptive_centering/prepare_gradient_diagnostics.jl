include(joinpath(@__DIR__, "reproduce.jl"))

"""Evaluate saved fits; this entry point never invokes a sampler."""
function prepare_gradient_diagnostics(offline_dir, online_dir, output_dir)
    mkpath(output_dir)
    table = split.(readlines(joinpath(offline_dir, "centeredness.tsv"))[2:end], '\t')
    cm = parse.(Float64, getindex.(table, 2))
    cs = parse.(Float64, getindex.(table, 3))
    online_table = split.(readlines(joinpath(online_dir, "online_centeredness.tsv"))[2:end], '\t')
    online_c = Dict(name => [parse(Float64, row[3]) for row in online_table if row[1] == name]
                    for name in ("mu", "log_sigma"))
    rows = NamedTuple[]
    checks = NamedTuple[]
    losses = NamedTuple[]
    for (label, configuration, dir, partial) in (
            ("noncentered", "NCP", offline_dir, false),
            ("partial", "Post-hoc", offline_dir, true),
            ("online", "Online", online_dir, false))
        fit = deserialize(joinpath(dir, "$label.jls"))
        @assert fit.complete && size(fit.posterior_position, 2) == 10_000
        data = partial ? prepared_data(; c_mu=cm, c_sigma=cs) : prepared_data()
        stan = stan_density(build_brmi(data; partial), "gradient-$label", output_dir)
        read(joinpath(output_dir, "motorcycle-gradient-$label.stan")) ==
            read(joinpath(dir, "motorcycle-$label.stan")) || error(
                "Generated $label Stan source differs from the saved fit producer; refusing coordinate reuse")
        descriptor = brm_descriptor(stan.sb)
        names, constrained = constrained_draws(stan, fit)
        unconstrained_names = BS.param_unc_names(stan.density.model)
        gradients = permutedims(reduce(hcat, [
            last(LogDensityProblems.logdensity_and_gradient(stan.density, collect(q)))
            for q in eachcol(fit.posterior_position)]))
        @assert all(isfinite, gradients)
        # Replay the public online scorer on full saved draws. Initial source
        # and target frames coincide with this compiled model, including the
        # partial refit's nonzero target centeredness. These are not the
        # historical NUTS trajectory groups/weights used during adaptation.
        adaptive = adaptive_centering_problem(stan.sb, stan.density, ENZYME_BACKEND)
        scored = candidate_scoring_losses(adaptive, fit.posterior_position,
                                          permutedims(gradients))
        for (name, predictor) in (("mu", :mu), ("log_sigma", :sigma))
            binding = brm_term_coordinates(descriptor, predictor, unconstrained_names;
                                           term=:hsgp_x, parameter=:basis_weights)
            for (basis, index) in enumerate(binding.coordinates), score in scored
                score.index == index || continue
                push!(losses, (; configuration, predictor=name, basis,
                    centeredness=score.candidate, loss=score.loss,
                    groups=score.groups, effective_n=score.effective_n,
                    evidence="retrospective_saved_draws", weights="unit",
                    objective="position_gradient_correlation_w1_0"))
            end
            display_c = label == "online" ? online_c[name] : nothing
            gp = hsgp_coordinate_draws(descriptor, permutedims(constrained), names;
                predictor, term=:hsgp_x, centeredness=display_c,
                basis_gradients=gradients[:, binding.coordinates])
            @assert all(gp.finite)
            # Independent finite differences of the compiled target in the
            # displayed frame, including its (basis-constant) log Jacobian.
            for i in (1, 5000, 10000), b in (1, 2, 19, 20)
                factor = exp((gp.sampled_centeredness[b] - gp.centeredness[b]) * gp.log_scales[i, b])
                u = gp.coordinates[i, b]
                q = copy(fit.posterior_position[:, i])
                f(v) = begin
                    pos = copy(q)
                    pos[binding.coordinates[b]] = v * factor
                    LogDensityProblems.logdensity(stan.density, pos) + log(factor)
                end
                h = 1e-5 / max(1.0, abs(gp.gradients[i, b]))
                fd = (f(u + h) - f(u - h)) / (2h)
                actual = gp.gradients[i, b]
                relative_error = abs(fd - actual) / max(1.0, abs(actual))
                @assert relative_error < 1e-4
                push!(checks, (; configuration, predictor=name, draw=i, basis=b,
                                gradient=actual, finite_difference=fd, relative_error))
            end
            for b in (1, 2, 19, 20), i in axes(gp.coordinates, 1)
                push!(rows, (; configuration, predictor=name, draw=i, basis=b,
                    centeredness=gp.centeredness[b], coordinate=gp.coordinates[i, b],
                    gradient=gp.gradients[i, b]))
            end
        end
        println("gradient_evaluation\t", label, "\tdraws=10000\tfinite=true")
        flush(stdout)
    end
    write_tsv(joinpath(output_dir, "coordinate_gradients.tsv"), rows)
    write_tsv(joinpath(output_dir, "gradient_checks.tsv"), checks)
    @assert length(losses) == 3 * 2 * DEFAULT_K * 11
    write_tsv(joinpath(output_dir, "retrospective_online_losses.tsv"), losses)
    write_tsv(joinpath(output_dir, "diagnostic_packages.tsv"), package_snapshot())
    open(joinpath(output_dir, "diagnostics_provenance.toml"), "w") do io
        TOML.print(io, Dict(
            "pilot" => TOML.parsefile(joinpath(offline_dir, "provenance.toml")),
            "online" => TOML.parsefile(joinpath(online_dir, "provenance.toml")),
            "diagnostics_commit" => strip(read(`git -C $RESEARCH_DIR rev-parse HEAD`, String)),
            "coordinate_gradients_sha256" => bytes2hex(sha256(read(joinpath(output_dir, "coordinate_gradients.tsv")))),
        ))
    end
    println("retrospective_online_losses\t", length(losses),
            "\tmissing=", count(r -> ismissing(r.loss), losses),
            "\tnonfinite=", count(r -> !ismissing(r.loss) && !isfinite(r.loss), losses))
    println("gradient_checks\t", length(checks), "\tmax_relative_error=",
            maximum(row.relative_error for row in checks))
    println("gradient_rows\t", length(rows), "\t", output_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || error("Usage: prepare_gradient_diagnostics.jl OFFLINE_DIR ONLINE_DIR OUTPUT_DIR")
    prepare_gradient_diagnostics(ARGS...)
end
