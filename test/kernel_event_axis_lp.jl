# test/kernel_event_axis_lp.jl — `ragged(x, group)`: take a FLAT secondary row
# axis and the column naming each of its rows' subject, and hand `kernel(...)`
# the ragged per-subject view (snag `a-bioavailabilit-28aeb46c`).
#
# Before this, `kernel(...)` admitted exactly two positional kinds, both length
# n_subjects: a raw data column and a per-subject LP. An LP over a DOSE-EVENT
# frame fits neither, and `_sb_kernel_lp_bucket` rejected it outright — an
# event-axis population LP has no ranef, so there was nothing to derive a
# grouping from. The only shipped spelling for event-level effects was one
# scalar prior per non-reference contrast addressed lexically in the cell, which
# cannot express `mo()` or `hsgp()` over dose.
#
# `x` may be a LINEAR PREDICTOR or a RAW FLAT COLUMN — one operation either way.
# Only the realization differs, and the difference is forced: a predictor is a
# Stan parameter, so it cannot be gathered Julia-side (the plate takes an index
# column and the cell fancy-indexes `log_F[rows]`); a data column is Julia data,
# so it is gathered into a `Vector{Vector{T}}` StanBlocks ingests natively —
# exactly the hand-prepared per-subject view a consumer would otherwise build
# before calling `kernel(...)` at all.
#
# WHAT THE LOAD-BEARING ASSERTIONS ARE:
#
#   1. `log_F` must be ABSENT from `sb.data` and the BridgeStan GRADIENT finite.
#      Registering an LP as data transpiles, passes stanc and samples — while
#      silently shadowing the sampled parameter with a constant (the trap
#      `probe_v2_lp_arg.jl` documents for the per-subject case).
#   2. `X_log_F`'s intercept must be sized off the EVENT axis. The intercept
#      length probe used to fall through to the consuming likelihood's N or to
#      hash order whenever every population term was wrapped (`1 + mo(diet) +
#      hsgp(x)` has no top-level data-backed peer), which put a SUBJECT-length
#      `rep_vector` inside an event-length `X`. Both extents are runtime in
#      Stan, so stanc accepts it and it dies at instantiation instead — hence
#      the runtime gate below, not just a text check.
#   3. A subject's event rows need NOT be contiguous in the event frame. The
#      join is by LABEL against the kernel's per-subject row order, so a
#      block-permuted event frame must give the same density at the same point.
#      Within one subject, event-frame order is what pairs with the ragged data
#      columns the cell walks.
#   4. Grouping a flat DATA column must land the same partition the predictor
#      gets, and must reproduce the hand-built ragged column byte for byte —
#      same density, same gradient, same dimension as the hand-built fixture.

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

const EV_RUN_BRIDGESTAN = get(ENV, "BRM_KERNEL_RUNTIME", "1") != "0"
const EV_CACHE = joinpath(tempdir(), "brm-kernel-event-axis")

# 3 subjects, 7 dose events (2, 3, 2). `sub_order` permutes the per-subject
# frame; `ev_order` permutes the event frame. Both must leave the density
# unchanged: the first because levels sort independently of dataframe order,
# the second because the event join is by label.
function ev_df(; sub_order = collect(1:3), ev_order = collect(1:7),
                 dose_amount = [[100.0, 100.0], [50.0, 50.0, 50.0], [200.0, 200.0]],
                 dose_subject = ["s1", "s1", "s2", "s2", "s2", "s3", "s3"])
    subj = (;
        subject = ["s1", "s2", "s3"],
        t_obs   = [abs.(sin.(1:4)) .+ 0.5 for _ in 1:3],
        dv      = [abs.(cos.(1:4)) .+ 0.1 for _ in 1:3],
        weight  = [60.0, 75.0, 90.0],
        dose_amount = dose_amount,
    )
    ev = (;
        dose_subject = dose_subject,
        vessel   = [1, 2, 1, 2, 2, 1, 2],
        diet     = [1, 2, 1, 3, 2, 3, 1],
        log_dose = log.([100.0, 100.0, 50.0, 50.0, 50.0, 200.0, 200.0]),
        # The same amounts as `dose_amount`, but FLAT on the event axis — what a
        # consumer's dose table actually holds before anyone groups it by hand.
        dose_amount_flat = [100.0, 100.0, 50.0, 50.0, 50.0, 200.0, 200.0],
    )
    (; (k => v[sub_order] for (k, v) in pairs(subj))...,
       (k => v[ev_order]  for (k, v) in pairs(ev))...)
end

ev_model(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_F  ~ 1 + factor(vessel) + mo(diet) + hsgp(log_dose; k = 5)
    log_CL ~ 1 + weight + (1 | p | subject)
    pred   ~ kernel(t_obs, dv, dose_amount, ragged(log_F, dose_subject), log_CL) do ts, yy, doses, lF, lCL
        effective_dose = sum(doses .* exp(lF))
        mu = effective_dose .* exp(-exp(lCL) .* ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

# Same model with a wider event-axis `hsgp` basis — the dimension delta is what
# pins the event-axis blocks as live parameters (see the runtime testset).
ev_wide_model(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_F  ~ 1 + factor(vessel) + mo(diet) + hsgp(log_dose; k = 8)
    log_CL ~ 1 + weight + (1 | p | subject)
    pred   ~ kernel(t_obs, dv, dose_amount, ragged(log_F, dose_subject), log_CL) do ts, yy, doses, lF, lCL
        effective_dose = sum(doses .* exp(lF))
        mu = effective_dose .* exp(-exp(lCL) .* ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

# No per-subject LP at all: the subject axis still has to come from somewhere,
# so an event-axis LP alone must NOT satisfy the grouping requirement.
ev_only_model(df) = @brm df begin
    sigma ~ Exponential(1)
    log_F ~ 1 + mo(diet)
    pred  ~ kernel(t_obs, dv, dose_amount, ragged(log_F, dose_subject)) do ts, yy, doses, lF
        mu = sum(doses .* exp(lF)) .* exp(-ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

# `ragged(...)` over an ALREADY-ragged per-subject column: nothing to group.
ev_preragged_model(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    pred   ~ kernel(t_obs, dv, ragged(dose_amount, dose_subject), log_CL) do ts, yy, doses, lCL
        mu = sum(doses) .* exp(-exp(lCL) .* ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

# `ragged(...)` over a FLAT raw data column — the same grouping the LP gets, but
# realized Julia-side. `dose_amount_flat` is the event frame's own amount column;
# grouping it here is exactly the hand-prepared per-subject view the consumer
# would otherwise have had to build before calling `kernel(...)` at all.
ev_flatcol_model(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_F  ~ 1 + factor(vessel) + mo(diet) + hsgp(log_dose; k = 5)
    log_CL ~ 1 + weight + (1 | p | subject)
    pred   ~ kernel(t_obs, dv, ragged(dose_amount_flat, dose_subject),
                    ragged(log_F, dose_subject), log_CL) do ts, yy, doses, lF, lCL
        effective_dose = sum(doses .* exp(lF))
        mu = effective_dose .* exp(-exp(lCL) .* ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

# Kernel zips only the OUTER subject cells. The PK index axis is already grouped
# per subject, while `ragged(qt_fixed, ecg_subject)` constructs an independent
# ECG value for each outer cell. Their inner segment lengths differ ([4, 3, 0]
# vs [3, 2, 2]); even their matching flat total is irrelevant to kernel.
function ev_independent_axes_df()
    (;
        subject = ["s1", "s2", "s3"],
        weight = [60.0, 75.0, 90.0],
        pk_idx = [collect(1.0:4.0), collect(1.0:3.0), Float64[]],
        pk_y = [fill(0.2, 4), fill(0.3, 3), Float64[]],
        ecg_subject = ["s1", "s1", "s1", "s2", "s2", "s3", "s3"],
        qt_x = collect(range(-1.0, 1.0; length = 7)),
    )
end

ev_independent_axes_model(df) = @brm df begin
    sigma ~ Exponential(1)
    qt_fixed ~ 1 + qt_x
    log_CL ~ 1 + weight + (1 | p | subject)
    pred ~ kernel(pk_idx, ragged(qt_fixed, ecg_subject), log_CL) do idx, qt, lCL
        rep_vector(sum(qt) + lCL, dims(idx)[1])
    end
    pk_y ~ Normal(pred, sigma)
end

# Grouping column on the WRONG axis (3 subject rows vs a 7-row predictor).
ev_wrong_axis_model(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_F  ~ 1 + mo(diet)
    log_CL ~ 1 + weight + (1 | p | subject)
    pred   ~ kernel(t_obs, dv, dose_amount, ragged(log_F, subject), log_CL) do ts, yy, doses, lF, lCL
        mu = sum(doses .* exp(lF)) .* exp(-exp(lCL) .* ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

ev_arity_model(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_F  ~ 1 + mo(diet)
    log_CL ~ 1 + weight + (1 | p | subject)
    pred   ~ kernel(t_obs, dv, dose_amount, ragged(log_F), log_CL) do ts, yy, doses, lF, lCL
        mu = sum(doses .* exp(lF)) .* exp(-exp(lCL) .* ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

ev_kwarg_model(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_F  ~ 1 + mo(diet)
    log_CL ~ 1 + weight + (1 | p | subject)
    pred   ~ kernel(t_obs, dv, dose_amount, ragged(log_F, dose_subject; by = subject), log_CL) do ts, yy, doses, lF, lCL
        mu = sum(doses .* exp(lF)) .* exp(-exp(lCL) .* ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

ev_problem(sb, tag) = begin
    isdir(EV_CACHE) || mkpath(EV_CACHE)
    code = StanBlocks.stan_code(sb.model)
    StanBlocks.stan_instantiate(sb.model; path = joinpath(EV_CACHE, "$(tag)_$(hash(code)).stan"))
end

@testset "kernel(...) — secondary row axis via ragged(x, group)" begin
    df = ev_df()
    sb = SBBRMI(ev_model(df); mod = @__MODULE__)
    descriptor = brm_descriptor(
        sb; name = :joint_pk_qt_brm2_direct_linear, highlights = ())

    @testset "the LP stays a parameter; the plate gets an index column" begin
        @test !haskey(sb.data, :log_F)
        # Row indices into the event frame, one group per SUBJECT ROW.
        @test sb.data[:kernel_pred_log_F_ragged] == [[1, 2], [3, 4, 5], [6, 7]]
        # String labels join rows on the Julia side only; Stan cannot type them.
        @test !haskey(sb.data, :dose_subject)
        @test !haskey(sb.data, :subject)

        # The fitted event-axis predictor is a public posterior quantity. Its
        # formula name, role, categorical design and HSGP basis survive as one
        # semantic address; consumers do not select its emitted Stan spelling.
        log_F = brm_output(descriptor, :log_F; role = :linear_predictor)
        @test log_F.name === :log_F
        @test log_F.logical === :log_F
        @test log_F.declaration === nothing
    end

    @testset "transpile + stanc" begin
        @test StanBlocks.stan.transpiles(sb.model)
        code = StanBlocks.stan_code(sb.model)
        # One Stan fancy-index, not a per-cell copy of the predictor.
        @test occursin(r"log_F\[\s*kernel_pred_log_F_ragged\.1\[", code)
        # The event-axis intercept must be sized off the EVENT axis (`diet`),
        # never off the subject axis (`weight`) — see header note 2.
        @test occursin("X_log_F = hcat(rep_vector(1.0, num_elements(diet))", code)
        # ...and the per-subject LP's own intercept must be UNCHANGED by that
        # probe change: it still resolves through the narrow tier-1 probe.
        @test occursin(r"X_log_CL\s*=\s*hcat\(", code)
        @test occursin("weight", code)
        @test StanBlocks.stanc_check(code; warn_pedantic = false).ok
    end

    @testset "frozen same-group replay regenerates the joint event design" begin
        # Same fitted subjects, reordered; nine future dose events on a changed
        # dose axis.  The kernel partition, event-level factors, and HSGP basis
        # all have to move while the trained group and basis coordinates stay
        # fixed.
        replay_df = ev_df(; sub_order = [3, 1, 2])
        replay_df = merge(replay_df, (;
            dose_amount = [[210.0, 190.0, 180.0, 170.0],
                           [110.0, 100.0, 90.0],
                           [60.0, 55.0]],
            dose_subject = ["s1", "s3", "s2", "s1", "s3", "s3", "s2", "s1", "s3"],
            vessel = [1, 2, 1, 2, 1, 2, 2, 1, 1],
            diet = [1, 3, 2, 2, 1, 3, 3, 1, 2],
            log_dose = log.([110.0, 210.0, 60.0, 100.0, 190.0,
                             180.0, 55.0, 90.0, 170.0]),
            dose_amount_flat = [110.0, 210.0, 60.0, 100.0, 190.0,
                                180.0, 55.0, 90.0, 170.0],
        ))

        replayed = reprocess(sb, replay_df)
        fresh = SBBRMI(ev_model(replay_df); mod = @__MODULE__)
        @test replayed.data[:kernel_nsub_pred] == 3
        @test replayed.data[:kernel_pred_log_F_ragged] ==
              [[2, 5, 6, 9], [1, 4, 8], [3, 7]]
        @test size(replayed.data[:PHI_hsgp_log_dose]) == (9, 5)
        @test replayed.preproc[:PHI_hsgp_log_dose].const_.fits ==
              sb.preproc[:PHI_hsgp_log_dose].const_.fits
        @test fresh.preproc[:PHI_hsgp_log_dose].const_.fits !=
              sb.preproc[:PHI_hsgp_log_dose].const_.fits
        @test replayed.data[:omega2_hsgp_log_dose] ==
              sb.data[:omega2_hsgp_log_dose]
        @test StanBlocks.stan_code(replayed.model) == StanBlocks.stan_code(sb.model)
        @test reprocess(sb, df).data == sb.data
        @test brm_output(
            brm_descriptor(replayed; name = :joint_pk_qt_replay, highlights = ()),
            :log_F; role = :linear_predictor).logical === :log_F

        unseen = merge(replay_df, (; subject = ["s3", "s1", "s9"]))
        @test_throws "label(s) Any[\"s2\"] in `dose_subject` name no subject" reprocess(
            sb, unseen)

        stray = merge(replay_df, (;
            dose_subject = ["s1", "s3", "s2", "s1", "s9", "s3", "s2", "s1", "s3"],
        ))
        @test_throws "name no subject" reprocess(sb, stray)
    end

    @testset "BridgeStan runtime — the event-axis LP is a live parameter" begin
        if EV_RUN_BRIDGESTAN
            using LogDensityProblems
            prob = ev_problem(sb, "base")
            dim = LogDensityProblems.dimension(prob)
            q = [0.1 * ((i % 5) - 2) for i in 1:dim]
            lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
            @test isfinite(lp)
            @test length(g) == dim
            @test all(isfinite, g)

            # The semantic selector and the former consumer workaround select
            # the exact same constrained posterior values. The workaround is
            # retained only as the independent acceptance oracle here.
            constrained_names = StanBlocks.BridgeStan.param_names(
                prob.model; include_tp = true, include_gq = false)
            semantic_coordinates = brm_output_coordinates(
                descriptor, :log_F, constrained_names;
                role = :linear_predictor)
            workaround_coordinates = findall(
                n -> n == "log_F" || startswith(n, "log_F."),
                constrained_names)
            @test semantic_coordinates == workaround_coordinates
            @test length(semantic_coordinates) == length(df.log_dose)
            constrained = StanBlocks.BridgeStan.param_constrain(
                prob.model, q; include_tp = true, include_gq = false)
            @test constrained[semantic_coordinates] ==
                  constrained[workaround_coordinates]

            # The event-axis blocks are REAL parameters, not data folded into a
            # constant: widening the event-axis `hsgp` basis by 3 must widen the
            # unconstrained dimension by exactly 3. A shadowed `log_F` would
            # leave the dimension untouched while still transpiling and sampling.
            wide = SBBRMI(ev_wide_model(df); mod = @__MODULE__)
            @test LogDensityProblems.dimension(ev_problem(wide, "wide")) == dim + 3

            # Permuting the per-subject frame leaves the density unchanged.
            sub_perm = SBBRMI(ev_model(ev_df(; sub_order = [3, 1, 2])); mod = @__MODULE__)
            sub_lp, sub_g = LogDensityProblems.logdensity_and_gradient(
                ev_problem(sub_perm, "subperm"), q)
            @test isapprox(sub_lp, lp; atol = 1e-8, rtol = 0)
            @test isapprox(sub_g, g; atol = 1e-8, rtol = 0)

            # ...and so does BLOCK-PERMUTING the event frame. `[6,1,3,2,4,7,5]`
            # interleaves the three subjects while preserving each subject's
            # internal event order, so no subject's rows are contiguous any
            # more. Nothing about the fix requires them to be.
            ev_perm_df = ev_df(; ev_order = [6, 1, 3, 2, 4, 7, 5])
            ev_perm = SBBRMI(ev_model(ev_perm_df); mod = @__MODULE__)
            @test ev_perm.data[:kernel_pred_log_F_ragged] == [[2, 4], [3, 5, 7], [1, 6]]
            ev_lp, ev_g = LogDensityProblems.logdensity_and_gradient(
                ev_problem(ev_perm, "evperm"), q)
            @test isapprox(ev_lp, lp; atol = 1e-8, rtol = 0)
            @test isapprox(ev_g, g; atol = 1e-8, rtol = 0)
        else
            @info "Skipping BridgeStan runtime gate (BRM_KERNEL_RUNTIME=0)"
        end
    end

    @testset "the contract fails loudly, not silently" begin
        # `ragged(...)` does not supply the kernel's subject axis.
        @test_throws "needs a subject grouping to join its event rows" SBBRMI(
            ev_only_model(df); mod = @__MODULE__)
        # An ALREADY-ragged per-subject column has nothing left to group.
        @test_throws "ALREADY a ragged per-subject column" SBBRMI(
            ev_preragged_model(df); mod = @__MODULE__)
        # The grouping column must cover the predictor's OWN frame.
        @test_throws "row axis" SBBRMI(ev_wrong_axis_model(df); mod = @__MODULE__)
        # Arity and keywords are rejected at @brm construction / lowering.
        @test_throws "exactly two positional args" SBBRMI(
            ev_arity_model(df); mod = @__MODULE__)
        @test_throws "ragged(...) takes no keywords" ev_kwarg_model(df)

        # A label naming no subject the kernel walks.
        stray = ev_df(; dose_subject = ["s1", "s1", "s2", "s2", "s9", "s3", "s3"])
        @test_throws "name no subject" SBBRMI(ev_model(stray); mod = @__MODULE__)

    end

    @testset "kernel constrains only the outer subject axis" begin
        axes = ev_independent_axes_df()
        @test length.(axes.pk_idx) == [4, 3, 0]
        @test [count(==(s), axes.ecg_subject) for s in axes.subject] == [3, 2, 2]
        @test sum(length, axes.pk_idx) == length(axes.ecg_subject) == 7

        independent = SBBRMI(ev_independent_axes_model(axes); mod = @__MODULE__)
        @test length(independent.data[:pk_idx]) ==
              length(independent.data[:kernel_pred_qt_fixed_ragged]) ==
              length(axes.subject)
        @test length.(independent.data[:pk_idx]) == [4, 3, 0]
        @test length.(independent.data[:kernel_pred_qt_fixed_ragged]) == [3, 2, 2]
        @test StanBlocks.stan.transpiles(independent.model)
        @test StanBlocks.stanc_check(
            StanBlocks.stan_code(independent.model); warn_pedantic = false).ok

        if EV_RUN_BRIDGESTAN
            using LogDensityProblems
            prob = ev_problem(independent, "independent_axes")
            dim = LogDensityProblems.dimension(prob)
            q = [0.05 * ((i % 5) - 2) for i in 1:dim]
            lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
            @test isfinite(lp)
            @test length(g) == dim
            @test all(isfinite, g)
        end
    end

    # `ragged(x, group)` is ONE operation — group a flat axis by a key — and `x`
    # being a parameter or data changes only WHERE that happens. This testset
    # pins the data half: same grouping, gathered Julia-side into a ragged column
    # instead of indexed in Stan, and equal to the hand-prepared view it replaces.
    @testset "ragged(...) groups a flat DATA column too, not just a predictor" begin
        flat = SBBRMI(ev_flatcol_model(df); mod = @__MODULE__)

        # Gathered here, so it IS data — unlike the predictor, which must not be.
        @test flat.data[:kernel_pred_dose_amount_flat_ragged] ==
              [[100.0, 100.0], [50.0, 50.0, 50.0], [200.0, 200.0]]
        # ...and it is exactly the hand-built per-subject view in the fixture.
        @test flat.data[:kernel_pred_dose_amount_flat_ragged] == df.dose_amount
        @test !haskey(flat.data, :log_F)

        # Wrapping a column for the cell must NOT consume the flat original: the
        # event-axis `hsgp(log_dose)` still names its own axis, and `X_log_F` is
        # still sized off it.
        code = StanBlocks.stan_code(flat.model)
        @test occursin("X_log_F = hcat(rep_vector(1.0, num_elements(diet))", code)
        @test StanBlocks.stanc_check(code; warn_pedantic = false).ok

        # Same grouping key, so the gathered column and the predictor's index
        # column must describe the same per-subject split.
        @test length.(flat.data[:kernel_pred_dose_amount_flat_ragged]) ==
              length.(flat.data[:kernel_pred_log_F_ragged])

        # Replay rebuilds the Julia-gathered half too.  Both axes are
        # independently reordered here: the subject frame selects the outer
        # ragged order, while the event frame selects each inner row sequence.
        flat_replay_df = ev_df(
            sub_order = [3, 1, 2], ev_order = [6, 1, 3, 2, 4, 7, 5])
        flat_replayed = reprocess(flat, flat_replay_df)
        @test flat_replayed.data[:kernel_pred_dose_amount_flat_ragged] ==
              flat_replay_df.dose_amount
        @test flat_replayed.data[:kernel_pred_log_F_ragged] ==
              [[1, 6], [2, 4], [3, 5, 7]]
        @test StanBlocks.stan_code(flat_replayed.model) == code

        if EV_RUN_BRIDGESTAN
            # The flat-column spelling is the SAME MODEL as the hand-grouped one:
            # `ev_model` passes `dose_amount` pre-grouped, `ev_flatcol_model` lets
            # BRM group `dose_amount_flat`. Identical density at the same point.
            base = ev_problem(SBBRMI(ev_model(df); mod = @__MODULE__), "base")
            dim = LogDensityProblems.dimension(base)
            q = [0.1 * ((i % 5) - 2) for i in 1:dim]
            b_lp, b_g = LogDensityProblems.logdensity_and_gradient(base, q)
            f_prob = ev_problem(flat, "flatcol")
            @test LogDensityProblems.dimension(f_prob) == dim
            f_lp, f_g = LogDensityProblems.logdensity_and_gradient(f_prob, q)
            @test isapprox(f_lp, b_lp; atol = 1e-8, rtol = 0)
            @test isapprox(f_g, b_g; atol = 1e-8, rtol = 0)
        end
    end
end

# ---- random effects on a kernel-delivered ragged LP (snag kernel-ragged-lp) ----
#
# A `ragged(x, group)` LP is an ordinary linear predictor over its OWN frame,
# so it may carry `(1 | ID | g)` terms — including a grouping FINER than the
# ragged `group` (per-lesion intercepts inside a per-subject tumor vector).
# Rows sharing a lesion share the drawn value; the cell receives the
# per-subject vector through the same fancy-index as a fixed-effect ragged LP.
#
# The one rule the per-subject case never teaches: EVERY grouping column must
# live on the LP's own (secondary) frame. The subject-level term uses the
# event-frame subject key (`tgi_obs_subject`), never the subject-frame column
# (`subject`) — the latter names a 3-row column against a 7-row LP and is
# refused loudly by the ragged length check, not silently misaligned.
#
# WHAT THE LOAD-BEARING ASSERTIONS ARE:
#
#   1. Rows 1 and 4 share lesion `s1:L1` NON-contiguously, and the emitted
#      `subject_lesion_idx` must map both to 1 — the lesion join is by label,
#      exactly like the subject join. The constrained `tgi_kg[1]` and
#      `tgi_kg[4]` are then EXACTLY equal at any parameter value (same
#      subject, same lesion, no event-varying population terms), while
#      `tgi_kg[1]` and `tgi_kg[2]` differ.
#   2. A sixth lesion level must widen the unconstrained dimension by exactly
#      1 — the lesion block is a live parameter, not data folded into a
#      constant (the same trap note 1 guards for the fixed-effect case).
#   3. Block-permuting the tumor frame leaves the density unchanged: the
#      lesion levels sort independently of event-frame order, so the same
#      point gives the same density and gradient.

function ev_lesion_df(; sub_order = collect(1:3), ev_order = collect(1:7))
    subj = (;
        subject = ["s1", "s2", "s3"],
        t_obs   = [abs.(sin.(1:4)) .+ 0.5 for _ in 1:3],
        dv      = [abs.(cos.(1:4)) .+ 0.1 for _ in 1:3],
        weight  = [60.0, 75.0, 90.0],
    )
    # 7 tumor rows; lesions nested in subjects (s1: L1, L2; s2: L3; s3: L4,
    # L5). Rows 1 and 4 share L1 with rows 2-3 of other lesions between
    # them, so contiguity can never explain a shared value.
    ev = (;
        tgi_obs_subject = ["s1", "s1", "s2", "s1", "s2", "s3", "s3"],
        subject_lesion  = ["s1:L1", "s1:L2", "s2:L3", "s1:L1",
                           "s2:L3", "s3:L4", "s3:L5"],
    )
    (; (k => v[sub_order] for (k, v) in pairs(subj))...,
       (k => v[ev_order]  for (k, v) in pairs(ev))...)
end

ev_lesion_model(df) = @brm df begin
    sigma     ~ Exponential(1)
    log(tgi_kg) ~ 1 + (1 | tg | tgi_obs_subject) + (1 | tgl | subject_lesion)
    log_CL    ~ 1 + weight + (1 | p | subject)
    pred ~ kernel(t_obs, dv, ragged(tgi_kg, tgi_obs_subject), log_CL) do ts, yy, kg, lCL
        # `tgi_kg` arrives on the RESPONSE scale (`tgi_kg = exp(log_tgi_kg)`):
        # no `exp` in the cell (snag `linked-lp-kernel-66e54eca`).
        kg_total = sum(kg)
        mu = kg_total .* exp(-exp(lCL) .* ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

# The subject-level ranef names the SUBJECT-frame column (3 rows) against the
# 7-row tumor frame — the spelling a per-subject habit reaches for first.
ev_lesion_subjectcol_model(df) = @brm df begin
    sigma     ~ Exponential(1)
    log(tgi_kg) ~ 1 + (1 | tg | subject) + (1 | tgl | subject_lesion)
    log_CL    ~ 1 + weight + (1 | p | subject)
    pred ~ kernel(t_obs, dv, ragged(tgi_kg, tgi_obs_subject), log_CL) do ts, yy, kg, lCL
        kg_total = sum(kg)
        mu = kg_total .* exp(-exp(lCL) .* ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

# Same 7 tumor rows, but row 4 is its own sixth lesion instead of sharing L1
# with row 1 — the dimension delta pins the lesion block as live parameters.
ev_lesion_wide_df() = merge(ev_lesion_df(), (;
    subject_lesion = ["s1:L1", "s1:L2", "s2:L3", "s1:L6",
                      "s2:L3", "s3:L4", "s3:L5"],
))

@testset "kernel(...) — random effects on a ragged(x, group) LP" begin
    df = ev_lesion_df()
    sb = SBBRMI(ev_lesion_model(df); mod = @__MODULE__)
    descriptor = brm_descriptor(
        sb; name = :joint_pk_tgi_brm1_direct_linear, highlights = ())

    @testset "the lesion ranef lowers onto the tumor frame" begin
        @test !haskey(sb.data, :tgi_kg)
        @test sb.data[:kernel_pred_tgi_kg_ragged] == [[1, 2, 4], [3, 5], [6, 7]]
        # Rows 1 and 4 share lesion 1 although rows 2-3 sit between them.
        @test sb.data[:subject_lesion_idx] == [1, 2, 3, 1, 3, 4, 5]
        @test sb.data[:n_subject_lesion] == 5
        @test sb.data[:tgi_obs_subject_idx] == [1, 1, 2, 1, 2, 3, 3]

        tgi_kg = brm_output(descriptor, :tgi_kg; role = :linear_predictor)
        @test tgi_kg.name === :tgi_kg
        @test tgi_kg.logical === :tgi_kg
    end

    @testset "transpile + stanc" begin
        @test StanBlocks.stan.transpiles(sb.model)
        code = StanBlocks.stan_code(sb.model)
        # The lesion block indexes the tumor frame, and the cell reads the
        # bare LHS-link name through the ragged fancy-index.
        @test occursin("b_tgl_subject_lesion[subject_lesion_idx, 1]", code)
        @test occursin(r"tgi_kg\[\s*kernel_pred_tgi_kg_ragged\.1\[", code)
        @test StanBlocks.stanc_check(code; warn_pedantic = false).ok
    end

    @testset "the subject-frame column is refused loudly, not misaligned" begin
        @test_throws "row axis" SBBRMI(
            ev_lesion_subjectcol_model(df); mod = @__MODULE__)
    end

    @testset "BridgeStan runtime — lesions share one drawn value" begin
        if EV_RUN_BRIDGESTAN
            using LogDensityProblems
            prob = ev_problem(sb, "lesion")
            dim = LogDensityProblems.dimension(prob)
            q = [0.1 * ((i % 5) - 2) for i in 1:dim]
            lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
            @test isfinite(lp)
            @test length(g) == dim
            @test all(isfinite, g)

            constrained_names = StanBlocks.BridgeStan.param_names(
                prob.model; include_tp = true, include_gq = false)
            tgi_idx = findall(
                n -> n == "tgi_kg" || startswith(n, "tgi_kg."),
                constrained_names)
            @test length(tgi_idx) == 7
            constrained = StanBlocks.BridgeStan.param_constrain(
                prob.model, q; include_tp = true, include_gq = false)
            # Same subject, same lesion, no event-varying population terms:
            # rows 1 and 4 are the SAME drawn value; row 2 is another lesion.
            @test constrained[tgi_idx][1] == constrained[tgi_idx][4]
            @test constrained[tgi_idx][1] != constrained[tgi_idx][2]

            # A sixth lesion level widens the unconstrained dimension by
            # exactly 1 — the lesion block is live, not a folded constant.
            wide = SBBRMI(ev_lesion_model(ev_lesion_wide_df()); mod = @__MODULE__)
            @test LogDensityProblems.dimension(ev_problem(wide, "lesion_wide")) ==
                  dim + 1

            # Block-permuting the tumor frame leaves the density unchanged:
            # `[6,1,3,2,4,7,5]` interleaves the subjects while the lesion
            # levels sort the same, so the same point still applies.
            perm = SBBRMI(
                ev_lesion_model(ev_lesion_df(; ev_order = [6, 1, 3, 2, 4, 7, 5]));
                mod = @__MODULE__)
            @test perm.data[:kernel_pred_tgi_kg_ragged] == [[2, 4, 5], [3, 7], [1, 6]]
            perm_lp, perm_g = LogDensityProblems.logdensity_and_gradient(
                ev_problem(perm, "lesion_perm"), q)
            @test isapprox(perm_lp, lp; atol = 1e-8, rtol = 0)
            @test isapprox(perm_g, g; atol = 1e-8, rtol = 0)
        else
            @info "Skipping BridgeStan runtime gate (BRM_KERNEL_RUNTIME=0)"
        end
    end
end
