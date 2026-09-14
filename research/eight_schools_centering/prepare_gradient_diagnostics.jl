include(joinpath(@__DIR__, "reproduce.jl"))
using Serialization
using SHA
using TOML

function read_centeredness(path, column)
    rows = split.(readlines(path)[2:end], '\t')
    Dict(parse(Int, row[1]) => parse(Float64, row[column]) for row in rows)
end

function gradient_matrix(problem, positions)
    output = Matrix{Float64}(undef, size(positions, 1), size(positions, 2))
    for draw in axes(positions, 2)
        value, gradient = LogDensityProblems.logdensity_and_gradient(
            problem, collect(positions[:, draw]))
        isfinite(value) && all(isfinite, gradient) ||
            error("non-finite target gradient at retained draw $draw")
        output[:, draw] = gradient
    end
    output
end

function display_draws(count)
    round.(Int, range(1, count; length=min(count, 1_000)))
end

function transformed_coordinates(positions, gradients, controls, layout)
    q = copy(positions)
    g = copy(gradients)
    product_error = 0.0
    physical_error = 0.0
    for draw in axes(q, 2)
        scale = exp(q[layout.scale, draw])
        for (school, index) in enumerate(layout.effects)
            z = q[index, draw]
            gz = g[index, draw]
            c = controls[school]
            coordinate = scale^c * z
            gradient = scale^-c * gz
            product_error = max(product_error, abs(z * gz - coordinate * gradient))
            physical_error = max(physical_error,
                abs((q[layout.population, draw] + scale * z) -
                    (q[layout.population, draw] + scale^(1 - c) * coordinate)))
            q[index, draw] = coordinate
            g[index, draw] = gradient
            g[layout.scale, draw] -= c * gz * z
        end
    end
    (; positions=q, gradients=g, product_error, physical_error)
end

function prepare_gradient_diagnostics(fit_dir, output_dir)
    mkpath(output_dir)
    stan = stan_density("gradient", output_dir)
    read(joinpath(output_dir, "eight-schools-gradient.stan")) ==
        read(joinpath(fit_dir, "eight-schools-model.stan")) ||
        error("generated target differs from the saved full-fit producer")
    pilot = deserialize(joinpath(fit_dir, "noncentered.jls"))
    partial = deserialize(joinpath(fit_dir, "partial_target.jls"))
    online = deserialize(joinpath(fit_dir, "online.jls"))
    for (label, fit) in (("noncentered", pilot), ("selected_partial", partial),
                         ("online", online))
        fit.complete && size(fit.posterior_position) == (10, SOURCE_DRAWS) ||
            error("$label is not a completed 10-coordinate, 10,000-draw fit")
        all(isfinite, fit.posterior_position) || error("non-finite $label draws")
        0 < fit.sampling_gradient_evaluations <= fit.total_gradient_evaluations ||
            error("invalid counters in $label")
    end

    positions = pilot.posterior_position
    gradients = gradient_matrix(stan.density, positions)
    layout = brm_layout(BS.param_unc_names(stan.density.model))
    offline = read_centeredness(joinpath(fit_dir, "centeredness.tsv"), 2)
    online_c = read_centeredness(joinpath(fit_dir, "centeredness.tsv"), 3)
    configurations = (
        (; label="NCP", controls=fill(0.0, 8)),
        (; label="Post-hoc", controls=[offline[j] for j in 1:8]),
        (; label="Online", controls=[online_c[j] for j in 1:8]),
    )

    # The public online objective is replayed on ONE common physical reference:
    # all retained NCP pilot draws and their exact target gradients, with unit
    # observation weights. It is retrospective, not recorded warmup history.
    adaptive = adaptive_centering_problem(stan.sb, stan.density, ENZYME_BACKEND)
    scored = WarmupHMC.candidate_scoring_losses(
        adaptive, positions, gradients)
    loss_rows = NamedTuple[]
    for score in scored
        school = only(findall(==(score.index), layout.effects))
        push!(loss_rows, (; school, centeredness=score.candidate,
            loss=score.loss, groups=score.groups, effective_n=score.effective_n,
            evidence="retrospective_saved_pilot_draws", weights="unit",
            objective="position_gradient_correlation_w1_0"))
    end
    write_tsv(joinpath(output_dir, "retrospective_online_losses.tsv"), loss_rows)

    gradient_rows = NamedTuple[]
    pair_rows = NamedTuple[]
    checks = NamedTuple[]
    invariants = NamedTuple[]
    for configuration in configurations
        moved = transformed_coordinates(positions, gradients,
            configuration.controls, layout)
        all(isfinite, moved.positions) && all(isfinite, moved.gradients) ||
            error("non-finite transformed coordinates in $(configuration.label)")
        push!(invariants, (; configuration=configuration.label,
            position_gradient_product_error=moved.product_error,
            physical_effect_error=moved.physical_error,
            retained_draws=size(positions, 2)))

        for draw in eachindex(display_draws(size(positions, 2)))
            actual_draw = display_draws(size(positions, 2))[draw]
            scale = exp(positions[layout.scale, actual_draw])
            for (school, index) in enumerate(layout.effects)
                c = configuration.controls[school]
                coordinate = moved.positions[index, actual_draw]
                gradient = moved.gradients[index, actual_draw]
                push!(gradient_rows, (; configuration=configuration.label,
                    draw=actual_draw, school, coordinate, gradient))
            end
        end

        for draw in axes(positions, 2)
            scale = exp(positions[layout.scale, draw])
            for (school, index) in enumerate(layout.effects)
                push!(pair_rows, (; configuration=configuration.label, draw,
                    school, hyperparameter=scale,
                    coordinate=moved.positions[index, draw],
                    parameter="Random-effect SD"))
            end
        end

        # Finite-difference the compiled target through the displayed frame,
        # including its scalar coordinate Jacobian.
        for draw in (1, 5_000, 10_000), (school, index) in enumerate(layout.effects)
            c = configuration.controls[school]
            log_scale = positions[layout.scale, draw]
            factor = exp(-c * log_scale)
            u = moved.positions[index, draw]
            q = collect(positions[:, draw])
            objective = v -> begin
                probe = copy(q)
                probe[index] = v * factor
                LogDensityProblems.logdensity(stan.density, probe) + log(factor)
            end
            h = 1e-5 / max(1.0, abs(moved.gradients[index, draw]))
            finite_difference = (objective(u + h) - objective(u - h)) / 2h
            actual = moved.gradients[index, draw]
            relative_error = abs(finite_difference - actual) / max(1.0, abs(actual))
            relative_error < 1e-4 || error(
                "$(configuration.label) school $school gradient check failed")
            push!(checks, (; configuration=configuration.label, draw, school,
                gradient=actual, finite_difference, relative_error))
        end
        println("gradient_frame\t", configuration.label, "\tdraws=10000\tchecks=24")
    end
    write_tsv(joinpath(output_dir, "gradient_scatter.tsv"), gradient_rows)
    write_tsv(joinpath(output_dir, "coordinate_pairs.tsv"), pair_rows)
    write_tsv(joinpath(output_dir, "gradient_checks.tsv"), checks)
    write_tsv(joinpath(output_dir, "frame_invariants.tsv"), invariants)

    metadata = Dict(
        "pilot" => Dict(
            "brm_commit" => strip(read(`git -C $(RESEARCH_DIR) rev-parse HEAD`, String)),
            "retained_draws" => size(positions, 2),
            "seed" => pilot.seed,
            "producer_sha256" => bytes2hex(sha256(
                read(joinpath(fit_dir, "eight-schools-model.stan")))),
        ),
        "script_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "gradient_draws" => "all 10,000 retained pilot draws",
        "scatter_display_draws_per_facet" => 1000,
        "online_loss_reference" => "common NCP pilot; unit weights; retrospective",
    )
    open(joinpath(output_dir, "diagnostics_provenance.toml"), "w") do io
        TOML.print(io, metadata)
    end
    println("gradient_diagnostics_complete\t", output_dir)
    (; stan, positions, gradients, loss_rows, output_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 ||
        error("usage: prepare_gradient_diagnostics.jl FULL_FIT_DIRECTORY OUTPUT_DIRECTORY")
    prepare_gradient_diagnostics(ARGS...)
end
