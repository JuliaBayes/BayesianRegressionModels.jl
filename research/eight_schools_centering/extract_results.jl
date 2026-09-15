include(joinpath(@__DIR__, "reproduce.jl"))

"""Extract model and sampler coordinates from complete returns and checkpoints."""
function extract_results(input, output)
    ispath(output) && error("use a fresh extraction directory")
    mkpath(output)
    for name in ("observations.tsv", "offline_centeredness.tsv", "offline_loss_profiles.tsv",
                 "online_centeredness.tsv", "centeredness.tsv", "timing_repetitions.tsv",
                 "priming_costs.tsv", "packages.tsv", "eight-schools-model.stan")
        cp(joinpath(input, name), joinpath(output, name))
    end
    provenance = TOML.parsefile(joinpath(input, "provenance.toml"))
    producer = joinpath(input, "reproduce.jl")
    bytes2hex(sha256(read(producer))) == provenance["script_sha256"] || error("producer source hash mismatch")
    cp(producer, joinpath(output, "sampling_producer.jl"))
    provenance["sampling_directory"] = abspath(input)
    provenance["extraction_script_sha256"] = bytes2hex(sha256(read(@__FILE__)))
    provenance["returned_frame"] = "model"
    provenance["checkpoint_frame"] = "source"
    open(joinpath(output, "provenance.toml"), "w") do io
        TOML.print(io, provenance)
    end
    stan = stan_density("extract", output)
    read(stan.path) == read(joinpath(input, "eight-schools-model.stan")) || error("target differs")
    layout = brm_layout(BS.param_unc_names(stan.density.model))
    controls = split.(readlines(joinpath(input, "centeredness.tsv"))[2:end], '\t')
    selected = parse.(Float64, getindex.(controls, 2))
    learned = parse.(Float64, getindex.(controls, 3))
    rows = NamedTuple[]
    checks = NamedTuple[]
    for label in ("noncentered", "centered", "partial", "online")
        record = deserialize(joinpath(input, "$label.jls"))
        cp(joinpath(input, "checkpoints-$label"), joinpath(output, "checkpoints-$label"))
        checkpoint = deserialize(joinpath(output, "checkpoints-$label", "cp_latest.jls"))
        c = label == "noncentered" ? zeros(8) : label == "centered" ? ones(8) : label == "partial" ? selected : learned
        actual_controls = Dict(i => p.c for (i,p) in checkpoint.reparam_sources)
        if label != "noncentered"
            [actual_controls[i] for i in layout.effects] == c || error("checkpoint controls differ")
        end
        # Independent scalar map: do not use the library transport helper.
        expected = copy(checkpoint.posterior_position)
        for (j, index) in enumerate(layout.effects)
            expected[index,:] .*= exp.(-c[j] .* expected[layout.scale,:])
        end
        error_value = maximum(abs.(expected .- record.posterior_position))
        error_value < 1e-9 || error("$label returned/checkpoint coordinate mismatch")
        push!(checks, (; fit=label, max_coordinate_error=error_value))
        model_record = merge(record, (; stored_frame="model"))
        serialize(joinpath(output, "$label.jls"), model_record)
        serialize(joinpath(output, "$(label)_source.jls"),
            merge(record, (; posterior_position=copy(checkpoint.posterior_position), stored_frame="source")))
        label in ("centered", "partial") && serialize(joinpath(output, "$(label)_target.jls"), model_record)
        export_coordinates(label, stan, model_record, output)
        d = diagnostics(label == "partial" ? "selected_partial" : label, model_record)
        push!(rows, merge(d, (; total_gradient_evaluations=record.total_gradient_evaluations,
            sampling_gradient_evaluations=record.sampling_gradient_evaluations,
            fit_seconds=record.fit_seconds, julia_compile_seconds=record.julia_compile_seconds,
            gc_seconds=record.gc_seconds,
            min_bulk_ess_per_total_gradient=d.min_bulk_ess/record.total_gradient_evaluations,
            min_bulk_ess_per_sampling_gradient=d.min_bulk_ess/record.sampling_gradient_evaluations,
            min_tail_ess_per_total_gradient=d.min_tail_ess/record.total_gradient_evaluations,
            min_tail_ess_per_sampling_gradient=d.min_tail_ess/record.sampling_gradient_evaluations)))
    end
    write_tsv(joinpath(output, "returned_checkpoint_frames.tsv"), checks)
    write_tsv(joinpath(output, "diagnostics.tsv"), rows)
    write_tsv(joinpath(output, "fit_costs.tsv"), [(; fit=r.fit, elapsed_seconds=r.fit_seconds,
        julia_compile_seconds=r.julia_compile_seconds, gc_seconds=r.gc_seconds,
        total_evaluation_counter=r.total_gradient_evaluations,
        sampling_evaluation_counter=r.sampling_gradient_evaluations,
        counter_scope="NUTS run total excludes Pathfinder/setup; sampling counts retained appended transitions") for r in rows])
    pilot, partial = rows[1], rows[3]
    write_tsv(joinpath(output, "workflow_costs.tsv"), [(; configuration="pilot_then_selected_partial_refit",
        elapsed_seconds=pilot.fit_seconds+partial.fit_seconds,
        total_evaluation_counter=pilot.total_gradient_evaluations+partial.total_gradient_evaluations,
        sampling_evaluation_counter=pilot.sampling_gradient_evaluations+partial.sampling_gradient_evaluations)])
    cp(joinpath(input, "saved_source_density_gradient_audit.tsv"), joinpath(output, "saved_source_density_gradient_audit.tsv"))
    println("eight_schools_extraction_complete\t", output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: extract_results.jl SAMPLING_DIRECTORY OUTPUT_DIRECTORY")
    extract_results(ARGS...)
end
