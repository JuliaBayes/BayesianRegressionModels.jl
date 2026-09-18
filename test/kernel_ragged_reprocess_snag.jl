# test/kernel_ragged_reprocess_snag.jl — snag `joint-ragged-ref-40368a11`
# (reported by a downstream PKPD consumer against a joint PK/QT kernel model).
#
# Locks two facts established while handling the snag:
#
#   1. TWO independent TOP-LEVEL ragged observations reflect through the
#      descriptor / generated-output path WITHOUT the historical
#      `MethodError -(::Nothing, ::Int64)`. (Failure 1 in the snag — already
#      fixed by landed top-level-ragged work; this pins it so it can't regress.)
#
#   2. `reprocess(sb, new_df; freeze_constants=true)` of a kernel(...) model with
#      `ragged(...)` inputs regenerates the derived design for fitted groups.
#      The gathered/index columns carry provenance, unseen groups fail loudly,
#      and genuinely new groups retain the explicit formula-rebuild routes
#      (`generative_plan(plan, new_schedule)`, `brm_descriptor :replay`). A
#      legacy SBBRMI missing the new provenance gets one deterministic upgrade
#      error rather than an incidental Dict-order-dependent failure.
#
# Run: julia --project=test test/kernel_ragged_reprocess_snag.jl

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

@testset "two independent top-level ragged outputs reflect (snag Failure 1)" begin
    joint_df = (;
        subject    = ["s1", "s2", "s3"],
        weight     = [60.0, 75.0, 90.0],
        pk_subject = ["s1", "s1", "s2", "s3", "s3"],
        pk_time    = [0.5, 1.0, 0.5, 0.5, 1.5],
        pk_y       = [1.0, 0.8, 1.2, 0.9, 0.4],
        qt_subject = ["s1", "s2", "s2", "s3"],
        qt_time    = [0.3, 0.3, 0.9, 0.6],
        qt_y       = [10.0, 11.0, 9.5, 10.5],
    )
    builder = @brm begin
        sigma_pk ~ Exponential(1)
        sigma_qt ~ Exponential(1)
        log_CL ~ 1 + weight + (1 | p | subject)
        pred_pk ~ kernel(ragged(pk_time, pk_subject), log_CL) do ts, lCL
            exp(-exp(lCL) .* ts)
        end
        pred_qt ~ kernel(ragged(qt_time, qt_subject), log_CL) do ts, lCL
            400.0 .- exp(lCL) .* ts
        end
        ragged(pk_y, pk_subject) ~ Normal(pred_pk, sigma_pk)
        ragged(qt_y, qt_subject) ~ Normal(pred_qt, sigma_qt)
    end

    # Build + BOTH reflection paths must not throw (the historical failure was a
    # MethodError raised *inside* descriptor reflection, not at build).
    local sb, sdesc, bdesc
    @test (sb = SBBRMI(builder(joint_df); mod=@__MODULE__)) isa SBBRMI
    @test (sdesc = StanBlocks.stan_descriptor(sb.model)) !== nothing
    @test (bdesc = brm_descriptor(builder, joint_df; mod=@__MODULE__, name=:joint)) !== nothing

    # Each ragged response axis keeps its own generated-draw + pointwise-loglik
    # twin — the pair that the offset MethodError used to prevent emitting.
    names = Set(String(o.name) for o in sdesc.outputs)
    @test "pk_y_gen" in names
    @test "qt_y_gen" in names
    @test "pk_y_likelihood" in names
    @test "qt_y_likelihood" in names
end

@testset "kernel ragged reprocess: frozen replay + legacy guard (snag Failure 2)" begin
    subj_df(; subs = ["s1", "s2", "s3"]) = (;
        subject = subs,
        t_obs   = [abs.(sin.(1:4)) .+ 0.5 for _ in eachindex(subs)],
        dv      = [abs.(cos.(1:4)) .+ 0.1 for _ in eachindex(subs)],
        weight  = collect(range(60.0, 90.0; length = length(subs))),
        dose_subject     = vcat([fill(s, 2) for s in subs]...),
        dose_amount_flat = repeat([100.0, 50.0], length(subs)),
        log_dose         = log.(repeat([100.0, 50.0], length(subs))),
    )
    builder = @brm begin
        sigma  ~ Exponential(1)
        log_F  ~ 1 + hsgp(log_dose; k = 5)
        log_CL ~ 1 + weight + (1 | p | subject)
        pred   ~ kernel(t_obs, dv, ragged(dose_amount_flat, dose_subject),
                        ragged(log_F, dose_subject), log_CL) do ts, yy, doses, lF, lCL
            effective_dose = sum(doses .* exp(lF))
            mu = effective_dose .* exp(-exp(lCL) .* ts)
            yy ~ normal(mu, sigma)
            mu
        end
    end
    # `total_groups=()`: this testset exercises the CONVENTIONAL GQ resample
    # path, and the default `:auto` would integrate the single-margin subject
    # bucket into totals, which `resample_groups` refuses (same pin as
    # test/resample_groups.jl, 2026-09-18).
    sb = SBBRMI(builder(subj_df()); mod=@__MODULE__, total_groups=())

    # The gathered kernel ragged columns now carry explicit regeneration
    # provenance, superseding the historical incidental error.
    ragged_keys = Symbol[k for k in keys(sb.data)
        if startswith(String(k), "kernel_") && endswith(String(k), "_ragged")]
    @test !isempty(ragged_keys)
    @test all(haskey(sb.preproc, k) for k in ragged_keys)

    replayed = reprocess(sb, subj_df(); freeze_constants=true)
    @test replayed.data == sb.data
    @test StanBlocks.stan_code(replayed.model) == StanBlocks.stan_code(sb.model)
    @test_throws "unseen level" reprocess(
        sb, subj_df(; subs=["s1", "s2", "new-subject"]))

    # The RESAMPLE twin deliberately admits a wholly new subject population,
    # but freezes the trained event-axis HSGP fit while rebuilding both ragged
    # partitions and moving the subject draw to generated quantities.
    # A fixed future regimen legitimately makes the event-axis HSGP covariate
    # constant even though its TRAINING domain was non-degenerate. Frozen
    # replay must evaluate that constant against the training basis; it must
    # not try to fit a new basis while constructing the CV template.
    new_population = merge(
        subj_df(; subs=["n1", "n2", "n3", "n4"]),
        (; dose_amount_flat=fill(120.0, 8), log_dose=fill(log(120.0), 8)))
    resampled = reprocess(
        sb, new_population; resample_groups=[:subject])
    @test resampled.preproc[:subject_idx].const_.levels ==
          ["n1", "n2", "n3", "n4"]
    @test resampled.data[:kernel_nsub_pred] == 4
    @test resampled.preproc[:PHI_hsgp_log_dose].const_.fits ==
          sb.preproc[:PHI_hsgp_log_dose].const_.fits
    @test size(resampled.data[:PHI_hsgp_log_dose]) == (8, 5)
    @test all(isfinite, resampled.data[:PHI_hsgp_log_dose])
    @test all(r -> r == resampled.data[:PHI_hsgp_log_dose][1, :],
              eachrow(resampled.data[:PHI_hsgp_log_dose]))
    @test resampled.data[:omega2_hsgp_log_dose] ==
          sb.data[:omega2_hsgp_log_dose]

    # Same-subject frozen replay was the pre-existing reference path. Its
    # evaluation of 120 mg must agree with the new-population path above.
    fitted_population = merge(
        subj_df(),
        (; dose_amount_flat=fill(120.0, 6), log_dose=fill(log(120.0), 6)))
    fitted_replay = reprocess(sb, fitted_population)
    @test resampled.data[:PHI_hsgp_log_dose][1, :] ==
          fitted_replay.data[:PHI_hsgp_log_dose][1, :]

    # Fresh-fit semantics still have no domain on which to fit this HSGP.
    @test_throws "hsgp: degenerate input" reprocess(
        sb, new_population; freeze_constants=false,
        resample_groups=[:subject])
    resampled_code = StanBlocks.stan_code(resampled.model)
    resampled_params = resampled_code[
        first(findfirst("parameters {", resampled_code)):first(findfirst(
            "model {", resampled_code))]
    resampled_gq = resampled_code[
        first(findfirst("generated quantities {", resampled_code)):end]
    @test !occursin("b_p_subject_z_flat", resampled_params)
    @test occursin("b_p_subject_z_flat = std_normal_vector_rng", resampled_gq)

    # A short-lived older artifact can still have the same derived data without
    # the provenance. Preserve one deterministic upgrade diagnosis for it.
    foreach(k -> delete!(sb.preproc, k), ragged_keys)
    err = try
        reprocess(sb, subj_df(); freeze_constants=true)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    msg = sprint(showerror, err)
    @test occursin("ragged", msg)
    @test occursin("Rebuild the SBBRMI", msg)
    @test occursin("generative_plan", msg)

    # The documented carry-to-new-design routes DO work (they rebuild + refit).
    plan = generative_plan(builder, subj_df(); mod=@__MODULE__)
    rebuilt = generative_plan(plan, subj_df(; subs = ["n1", "n2", "n3", "n4"]))
    @test any(String(k) == "kernel_nsub_pred" for k in keys(rebuilt.data))

    d = brm_descriptor(builder, subj_df(); mod=@__MODULE__, name=:pk)
    @test :replay in Set(op.name for op in d.operations)
    @test :reprocess in Set(op.name for op in d.operations)
    @test brm_execute(d, :replay, subj_df(; subs = ["n1", "n2", "n3"])) !== nothing
end
