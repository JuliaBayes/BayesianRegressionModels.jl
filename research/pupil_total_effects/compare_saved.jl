include("run.jl")

function compare_saved(integrated_dir,baseline_dir)
    baseline = deserialize(joinpath(baseline_dir,"brms_ncp.jls"))
    partial = deserialize(joinpath(integrated_dir,"partial.jls"))
    control = deserialize(joinpath(integrated_dir,"scaled_total.jls"))
    data = PTE.load_data()
    samples(q) = permutedims(reshape(q[1:44,:],44,size(q,2),1),(2,3,1))
    q1,q2 = baseline.positions,partial.positions
    mcse1 = vec(MCMCDiagnosticTools.mcse(samples(q1);kind=mean))
    mcse2 = vec(MCMCDiagnosticTools.mcse(samples(q2);kind=mean))
    names = PTE.coordinate_names(data)
    rows = map(1:44) do j
        mean1,mean2 = mean(q1[j,:]),mean(q2[j,:])
        combined_mcse = hypot(mcse1[j],mcse2[j])
        (;parameter=names[j],baseline_mean=mean1,partial_mean=mean2,
            baseline_sd=std(q1[j,:]),partial_sd=std(q2[j,:]),
            baseline_mcse=mcse1[j],partial_mcse=mcse2[j],
            standardized_mean_difference=(mean2-mean1)/combined_mcse)
    end
    # Descriptive smoke diagnostic, not proof of convergence or equivalence.
    write_tsv(joinpath(integrated_dir,"posterior_agreement.tsv"),rows)
    _,ds1 = diagnostic_rows("brms_ncp",baseline,data)
    _,ds2 = diagnostic_rows("partial",partial,data)
    workflow_cost = control.all_gradient_calls + partial.all_gradient_calls
    workflow_efficiency = ds2.min_bulk_ess/workflow_cost*1000
    comparison = (;sampling_ratio=ds2.min_common_bulk_ess_per_1000_gradients/
        ds1.min_common_bulk_ess_per_1000_gradients,
        pilot_plus_refit_gradient_calls=workflow_cost,
        pilot_plus_refit_min_ess_per_1000_gradients=workflow_efficiency,
        workflow_ratio=workflow_efficiency/ds1.min_bulk_ess_per_1000_all_gradients,
        max_abs_mean_difference_in_combined_mcse=maximum(abs(r.standardized_mean_difference) for r in rows),
        coordinates_above_3_mcse=count(r->abs(r.standardized_mean_difference)>3,rows))
    write_tsv(joinpath(integrated_dir,"baseline_comparison.tsv"),[comparison])
    println(comparison)
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==2 || error("Expected integrated and conventional baseline directories")
    compare_saved(ARGS...)
end
