include(joinpath(@__DIR__, "reproduce.jl"))

"""Recover native counters from original checkpoints; never resample or invent timing."""
function report_costs(offline, online, output)
    mkpath(output)
    rows = NamedTuple[]
    for (label, dir) in (("noncentered", offline), ("partial", offline), ("online", online))
        fit = deserialize(joinpath(dir, "$label.jls"))
        checkpoint = joinpath(dir, "checkpoints-$label", "cp_latest.jls")
        payload = deserialize(checkpoint)
        @assert fit.complete && size(fit.posterior_position) == size(payload.posterior_position)
        @assert fit.n_divergent_samples == payload.n_divergent_samples
        total = payload.total_evaluation_counter
        @assert total > 0
        hasproperty(fit, :total_gradient_evaluations) &&
            @assert fit.total_gradient_evaluations == total
        stats = diagnostics(label, fit)
        push!(rows, (; fit=label, total_nuts_gradient_evaluations=total,
            # Producing 131e742 does not retain these two measurements.
            sampling_gradient_evaluations=missing,
            fit_seconds=get(fit, :fit_seconds, missing),
            min_bulk_ess=stats.min_bulk_ess, min_tail_ess=stats.min_tail_ess,
            min_bulk_ess_per_total_gradient=stats.min_bulk_ess / total,
            min_tail_ess_per_total_gradient=stats.min_tail_ess / total,
            min_bulk_ess_per_sampling_gradient=missing,
            min_tail_ess_per_sampling_gradient=missing,
            parameter_scope="44_sampled_model_coordinates",
            counter_scope="NUTS_steps_including_warmup_and_discarded_epochs_excluding_initializer",
            evidence="final_original_checkpoint_not_sum_of_windows"))
    end
    write_tsv(joinpath(output, "fit_costs.tsv"), rows)
    foreach(println, rows)
    rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || error("Usage: report_costs.jl OFFLINE_DIR ONLINE_DIR OUTPUT_DIR")
    report_costs(ARGS...)
end
