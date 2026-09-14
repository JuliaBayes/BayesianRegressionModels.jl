include(joinpath(@__DIR__, "reproduce.jl"))

function recorded_sampling_cost(fit, checkpoint, total)
    count = fit.sampling_gradient_evaluations
    count isa Integer && 0 < count <= total ||
        throw(ArgumentError("Invalid recorded sampling-gradient count: $count"))
    checkpoint.sampling_evaluation_counter == count ||
        throw(ArgumentError("Fit and final checkpoint sampling-gradient counts disagree"))
    count
end

function report_costs(offline, online, output)
    mkpath(output)
    rows = NamedTuple[]
    for (label, dir) in (("noncentered", offline), ("partial", offline), ("online", online))
        fit = deserialize(joinpath(dir, "$label.jls"))
        checkpoint = deserialize(joinpath(dir, "checkpoints-$label", "cp_latest.jls"))
        @assert fit.complete && size(fit.posterior_position) == size(checkpoint.posterior_position)
        @assert fit.n_divergent_samples == checkpoint.n_divergent_samples
        total = checkpoint.total_evaluation_counter
        @assert total > 0 && fit.total_gradient_evaluations == total
        sampling = recorded_sampling_cost(fit, checkpoint, total)
        stats = diagnostics(label, fit)
        push!(rows, (; fit=label,
            total_nuts_gradient_evaluations=total,
            sampling_gradient_evaluations=sampling,
            fit_seconds=fit.fit_seconds,
            min_bulk_ess=stats.min_bulk_ess,
            min_tail_ess=stats.min_tail_ess,
            min_bulk_ess_per_total_gradient=stats.min_bulk_ess / total,
            min_tail_ess_per_total_gradient=stats.min_tail_ess / total,
            min_bulk_ess_per_sampling_gradient=stats.min_bulk_ess / sampling,
            min_tail_ess_per_sampling_gradient=stats.min_tail_ess / sampling,
            parameter_scope="all_777_unconstrained_model_coordinates",
            counter_scope="NUTS_steps_including_warmup_and_discarded_epochs_excluding_initializer",
            evidence="final_checkpoint_and_returned_result_agree"))
    end
    pilot = only(filter(r -> r.fit == "noncentered", rows))
    partial = only(filter(r -> r.fit == "partial", rows))
    workflow = (; fit="posthoc_workflow_pilot_plus_refit",
        total_nuts_gradient_evaluations=pilot.total_nuts_gradient_evaluations +
                                       partial.total_nuts_gradient_evaluations,
        sampling_gradient_evaluations=pilot.sampling_gradient_evaluations +
                                      partial.sampling_gradient_evaluations,
        fit_seconds=pilot.fit_seconds + partial.fit_seconds,
        min_bulk_ess=partial.min_bulk_ess,
        min_tail_ess=partial.min_tail_ess,
        min_bulk_ess_per_total_gradient=partial.min_bulk_ess /
            (pilot.total_nuts_gradient_evaluations + partial.total_nuts_gradient_evaluations),
        min_tail_ess_per_total_gradient=partial.min_tail_ess /
            (pilot.total_nuts_gradient_evaluations + partial.total_nuts_gradient_evaluations),
        min_bulk_ess_per_sampling_gradient=partial.min_bulk_ess /
            (pilot.sampling_gradient_evaluations + partial.sampling_gradient_evaluations),
        min_tail_ess_per_sampling_gradient=partial.min_tail_ess /
            (pilot.sampling_gradient_evaluations + partial.sampling_gradient_evaluations),
        parameter_scope="refit_minimum_over_all_777_coordinates_charged_for_pilot_plus_refit",
        counter_scope="NUTS_steps_including_warmup_and_discarded_epochs_excluding_initializer",
        evidence="charge_the_pilot_when_assessing_the_posthoc_workflow")
    push!(rows, workflow)
    write_tsv(joinpath(output, "fit_costs.tsv"), rows)
    foreach(println, rows)
    rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || error("Usage: report_costs.jl OFFLINE_DIR ONLINE_DIR OUTPUT_DIR")
    report_costs(ARGS...)
end
