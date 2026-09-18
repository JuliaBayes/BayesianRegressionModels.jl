# test/ragged_formula_lhs.jl — group a flat observation axis at the formula LHS.
#
# Run: julia --project=test test/ragged_formula_lhs.jl

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal
using LogDensityProblems

flat_builder = @brm begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    pred ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
        exp(lCL) .+ 0.0 .* idxs
    end
    ragged(obs_y, obs_subject) ~ Normal(pred, sigma)
end

grouped_builder = @brm begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    pred ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
        exp(lCL) .+ 0.0 .* idxs
    end
    obs_y ~ Normal(pred, sigma)
end

no_kernel_builder = @brm begin
    mu ~ 1 + x
    ragged(y, g) ~ Normal(mu, 1.0)
end

censored_flat_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    loc ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
        exp(lCL) .+ 0.0 .* idxs
    end
    ragged(pk_conc, obs_subject) ~ censored(
        Normal(loc, addprop(loc, sigma_add, sigma_prop)); lower=pk_lloq)
end

# The pre-fix workaround: callers had to group both the response and its bound
# themselves, even though the kernel still consumed the original flat frame.
censored_grouped_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    loc ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
        exp(lCL) .+ 0.0 .* idxs
    end
    pk_conc ~ censored(
        Normal(loc, addprop(loc, sigma_add, sigma_prop)); lower=pk_lloq)
end

censored_scalar_bound_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    loc ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
        exp(lCL) .+ 0.0 .* idxs
    end
    ragged(pk_conc, obs_subject) ~ censored(
        Normal(loc, addprop(loc, sigma_add, sigma_prop)); lower=0.10)
end

truncated_two_bound_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    loc ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
        exp(lCL) .+ 0.0 .* idxs
    end
    ragged(pk_conc, obs_subject) ~ truncated(
        Normal(loc, addprop(loc, sigma_add, sigma_prop));
        lower=pk_lloq, upper=pk_uloq)
end

interval_flat_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    loc ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
        exp(lCL) .+ 0.0 .* idxs
    end
    ragged(pk_conc, obs_subject) ~ interval_censored(
        Normal(loc, addprop(loc, sigma_add, sigma_prop)); upper=pk_uloq)
end

explicit_bound_grouping_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    loc ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
        exp(lCL) .+ 0.0 .* idxs
    end
    ragged(pk_conc, obs_subject) ~ censored(
        Normal(loc, addprop(loc, sigma_add, sigma_prop));
        lower=ragged(pk_lloq, obs_subject))
end

flat_data = (;
    subject=["s2", "s1", "s3"],
    obs_subject=["s1", "s2", "s3", "s2", "s3", "s3"],
    obs_idx=[1.0, 1.0, 1.0, 2.0, 2.0, 3.0],
    obs_y=[0.11, 0.21, 0.31, 0.22, 0.32, 0.33],
)
expected = [[0.21, 0.22], [0.11], [0.31, 0.32, 0.33]]
grouped_data = merge(flat_data, (; obs_y=expected))

censored_flat_data = (;
    subject=["s2", "s1"],
    obs_subject=["s1", "s2", "s1", "s2", "s2", "s1", "s2"],
    obs_idx=collect(1.0:7.0),
    pk_conc=[0.15, 0.20, 0.10, 0.25, 0.10, 0.30, 0.18],
    pk_lloq=fill(0.10, 7),
    pk_uloq=fill(0.40, 7),
)
censored_rows = [[2, 4, 5, 7], [1, 3, 6]]
censored_grouped_data = merge(censored_flat_data, (;
    pk_conc=[censored_flat_data.pk_conc[r] for r in censored_rows],
    pk_lloq=[censored_flat_data.pk_lloq[r] for r in censored_rows],
    pk_uloq=[censored_flat_data.pk_uloq[r] for r in censored_rows],
))

@testset "ragged observation LHS — flat formula boundary" begin
    brmi = flat_builder(flat_data)
    @test :obs_y in keys(brmi.operations)
    @test only(outcomes(brmi)).response === :obs_y

    flat = SBBRMI(brmi; mod=@__MODULE__)
    grouped = SBBRMI(grouped_builder(grouped_data); mod=@__MODULE__)

    @test flat.data[:obs_y] == expected
    @test flat.data[:obs_y] == grouped.data[:obs_y]
    @test BayesianRegressionModels.stan_code(flat) ==
          BayesianRegressionModels.stan_code(grouped)

    @test StanBlocks.stan.transpiles(flat.model)
    code = BayesianRegressionModels.stan_code(flat)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    d = brm_descriptor(flat_builder, flat_data; mod=@__MODULE__)
    draw = brm_output(d, :obs_y; role=:posterior_predictive)
    loglik = brm_output(d, :obs_y; role=:pointwise_loglik)
    @test draw.source === :obs_y && loglik.source === :obs_y
    @test draw.logical === :obs_y && loglik.logical === :obs_y
    @test draw.segments == [2, 3, 6]
    @test loglik.segments == draw.segments

    flat_problem = brm_execute(d, :fit)
    grouped_problem = brm_execute(
        brm_descriptor(grouped_builder, grouped_data; mod=@__MODULE__), :fit)
    dim = LogDensityProblems.dimension(flat_problem)
    @test LogDensityProblems.dimension(grouped_problem) == dim
    q = [0.05 * ((i % 7) - 3) for i in 1:dim]
    flat_lp, flat_grad = LogDensityProblems.logdensity_and_gradient(flat_problem, q)
    grouped_lp, grouped_grad =
        LogDensityProblems.logdensity_and_gradient(grouped_problem, q)
    @test flat_lp == grouped_lp
    @test flat_grad == grouped_grad

    @testset "invalid axes fail before Stan" begin
        mismatched = merge(flat_data, (; obs_subject=flat_data.obs_subject[1:end-1]))
        @test_throws "grouping column must name" SBBRMI(
            flat_builder(mismatched); mod=@__MODULE__)

        unknown = merge(flat_data, (;
            obs_subject=["s1", "s2", "s3", "s2", "s3", "not-a-subject"],
        ))
        @test_throws "name no subject" SBBRMI(flat_builder(unknown); mod=@__MODULE__)

        already_grouped = merge(flat_data, (; obs_y=expected))
        @test_throws "ALREADY-ragged response" SBBRMI(
            flat_builder(already_grouped); mod=@__MODULE__)

        @test_throws "needs a `kernel(...)` result" SBBRMI(
            no_kernel_builder((; x=[1.0, 2.0], y=[0.1, 0.2], g=[1, 1]));
            mod=@__MODULE__)
    end
end

@testset "ragged observation LHS — observed censoring bound" begin
    flat = SBBRMI(censored_flat_builder(censored_flat_data); mod=@__MODULE__)
    grouped = SBBRMI(
        censored_grouped_builder(censored_grouped_data); mod=@__MODULE__)

    expected_response = censored_grouped_data.pk_conc
    expected_lower = censored_grouped_data.pk_lloq
    @test flat.data[:pk_conc] == expected_response == grouped.data[:pk_conc]
    @test flat.data[:pk_conc_lower_pk_lloq_ragged] ==
          expected_lower == grouped.data[:pk_lloq]
    @test flat.data[:pk_lloq] == censored_flat_data.pk_lloq

    # The exact reported formula, including BRM's add+proportional residual
    # helper over the ragged kernel output, must survive both compiler gates.
    @test StanBlocks.stan.transpiles(flat.model)
    @test StanBlocks.stan.transpiles(grouped.model)
    scalar_bound = SBBRMI(
        censored_scalar_bound_builder(censored_flat_data); mod=@__MODULE__)
    @test StanBlocks.stan.transpiles(scalar_bound.model)
    two_bound = SBBRMI(
        truncated_two_bound_builder(censored_flat_data); mod=@__MODULE__)
    interval = SBBRMI(interval_flat_builder(censored_flat_data); mod=@__MODULE__)
    @test two_bound.data[:pk_conc_lower_pk_lloq_ragged] == expected_lower
    @test two_bound.data[:pk_conc_upper_pk_uloq_ragged] ==
          censored_grouped_data.pk_uloq
    @test interval.data[:pk_conc_upper_pk_uloq_ragged] ==
          censored_grouped_data.pk_uloq
    for brmi in (flat, grouped)
        code = BayesianRegressionModels.stan_code(brmi)
        @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    end

    # Diff the fix against the old manual-grouping workaround at the executable
    # boundary. The derived bound has a different Stan data name, but the model
    # density and gradient must be exactly the same at the same unconstrained q.
    flat_problem = brm_execute(
        brm_descriptor(censored_flat_builder, censored_flat_data; mod=@__MODULE__),
        :fit,
    )
    grouped_problem = brm_execute(
        brm_descriptor(
            censored_grouped_builder, censored_grouped_data; mod=@__MODULE__),
        :fit,
    )
    dim = LogDensityProblems.dimension(flat_problem)
    @test LogDensityProblems.dimension(grouped_problem) == dim
    q = fill(-0.2, dim)
    flat_lp_grad = LogDensityProblems.logdensity_and_gradient(flat_problem, q)
    grouped_lp_grad =
        LogDensityProblems.logdensity_and_gradient(grouped_problem, q)
    @test flat_lp_grad == grouped_lp_grad
    @test isfinite(first(flat_lp_grad))
    @test all(isfinite, last(flat_lp_grad))

    # The authoritative grouping lives at the formula LHS. Asking a bound to
    # perform a second, independent grouping remains unsupported and loud.
    @test_throws "bounds must be numeric literals or observed data columns" SBBRMI(
        explicit_bound_grouping_builder(censored_flat_data); mod=@__MODULE__)

    short_bound = merge(censored_flat_data, (;
        pk_lloq=censored_flat_data.pk_lloq[1:end-1],
    ))
    @test_throws "has 6 rows but the flat response has 7" SBBRMI(
        censored_flat_builder(short_bound); mod=@__MODULE__)

    wrong_segments = merge(censored_flat_data, (;
        pk_lloq=[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
    ))
    @test_throws "group lengths [3, 4]; expected [4, 3]" SBBRMI(
        censored_flat_builder(wrong_segments); mod=@__MODULE__)
end

# snag reprocess-ragged-ea8c7e42 (reported by a downstream PKPD consumer against
# joint_brm2). The gathered ragged RESPONSE and its data-backed censoring BOUND
# had no preprocessing record, so `reprocess`/`resample_groups` errored on the
# derived bound (`pk_conc_lower_pk_lloq_ragged`) and would have silently kept the
# stale/flat response. Both now carry `:ragged_gather` provenance and are
# re-gathered from their flat columns into the kernel's per-subject row order. A
# fixed future regimen makes the event-axis HSGP covariate constant, so
# rebuilding the model refits a degenerate basis — the frozen reprocess is the
# only correct route, exactly as the reporter needed.
hsgp_censored_builder = @brm begin
    sigma  ~ Exponential(1)
    log_F  ~ 1 + hsgp(log_dose; k = 5)
    log_CL ~ 1 + weight + (1 | p | subject)
    pred   ~ kernel(ragged(obs_time, obs_subject),
                    ragged(dose_amount_flat, dose_subject),
                    ragged(log_F, dose_subject), log_CL) do ts, doses, lF, lCL
        sum(doses .* exp(lF)) .* exp(-exp(lCL) .* ts)
    end
    ragged(pk_conc, obs_subject) ~ censored(Normal(pred, sigma); lower = pk_lloq)
end

hsgp_censored_df(; subs, dose_amt) = (;
    subject          = subs,
    weight           = collect(range(60.0, 90.0; length = length(subs))),
    obs_subject      = vcat([fill(s, 4) for s in subs]...),
    obs_time         = repeat([0.5, 1.0, 1.5, 2.0], length(subs)),
    pk_conc          = repeat([1.2, 0.8, 0.5, 0.3], length(subs)),
    pk_lloq          = fill(0.1, 4 * length(subs)),
    dose_subject     = vcat([fill(s, 2) for s in subs]...),
    dose_amount_flat = repeat(dose_amt, length(subs)),
    log_dose         = log.(repeat(dose_amt, length(subs))),
)

@testset "ragged observation LHS — reprocess/resample regenerates response + bound" begin
    train = hsgp_censored_df(; subs = ["s1", "s2", "s3"], dose_amt = [100.0, 50.0])
    # `total_groups=()`: this testset exercises the CONVENTIONAL GQ resample
    # path (see the resample call below), and the default `:auto` would
    # integrate the single-margin subject bucket into totals, which
    # `resample_groups` refuses (same pin as test/resample_groups.jl).
    sb = SBBRMI(hsgp_censored_builder(train); mod = @__MODULE__, total_groups = ())

    # (0) The gathered response and its data-backed bound now carry provenance.
    @test sb.preproc[:pk_conc].kind === :ragged_gather
    @test sb.preproc[:pk_conc_lower_pk_lloq_ragged].kind === :ragged_gather
    @test sb.preproc[:pk_conc_lower_pk_lloq_ragged].raw_ref === :pk_lloq

    # (1) Same-subject frozen replay: byte-identical Stan, correctly re-gathered.
    same = hsgp_censored_df(; subs = ["s1", "s2", "s3"], dose_amt = [120.0, 120.0])
    replayed = reprocess(sb, same; freeze_constants = true)
    @test StanBlocks.stan_code(replayed.model) == StanBlocks.stan_code(sb.model)
    @test replayed.data[:pk_conc] == [[1.2, 0.8, 0.5, 0.3] for _ in 1:3]
    @test replayed.data[:pk_conc_lower_pk_lloq_ragged] == [[0.1, 0.1, 0.1, 0.1] for _ in 1:3]

    # (2) New-population resample onto a fixed 120 mg future: succeeds, freezes
    #     the trained HSGP basis, and re-gathers response + bound for the new
    #     subjects (the reporter's exact call).
    future = hsgp_censored_df(; subs = ["n1", "n2", "n3", "n4"], dose_amt = [120.0, 120.0])
    resampled = reprocess(sb, future; resample_groups = [:subject])
    @test resampled.preproc[:subject_idx].const_.levels == ["n1", "n2", "n3", "n4"]
    @test resampled.data[:kernel_nsub_pred] == 4
    @test resampled.preproc[:PHI_hsgp_log_dose].const_.fits ==
          sb.preproc[:PHI_hsgp_log_dose].const_.fits            # HSGP basis frozen
    @test resampled.data[:pk_conc] == [[1.2, 0.8, 0.5, 0.3] for _ in 1:4]
    @test resampled.data[:pk_conc_lower_pk_lloq_ragged] == [[0.1, 0.1, 0.1, 0.1] for _ in 1:4]
    code = StanBlocks.stan_code(resampled.model)
    params = code[first(findfirst("parameters {", code)):first(findfirst("model {", code))]
    gq = code[first(findfirst("generated quantities {", code)):end]
    @test !occursin("b_p_subject_z_flat", params)
    @test occursin("b_p_subject_z_flat = std_normal_vector_rng", gq)

    # (2b) snag reprocess-resamp-14c5ef20 (reported by a downstream PKPD consumer against
    #      joint_brm2): the resampled program above must be stanc-valid. It was
    #      not — under resample the kernel result `pred` moves to generated
    #      quantities, but StanBlocks left the held-out top-level ragged density
    #      loop `ragged(pk_conc,…) ~ …(pred,…)` in the MODEL block, referencing
    #      `pred__pl_mem_1` (GQ-scoped) out of scope ("Identifier pred__pl_mem_1
    #      not in scope"). Root cause was a missing cv-taint propagation in
    #      StanBlocks' top-level ragged-obs forward path
    #      (`_forward_ragged_obs_broadcast!`), fixed in StanBlocks (snag
    #      ragged-obs-not-c-5b1180c7, StanBlocks devibe 9ec5e97): the ragged
    #      response is now marked cv when its density arg is cv (mirroring the
    #      scalar/indexed obs paths), so `distribution_blocks` routes the
    #      per-group density loop to generated quantities. This locks that
    #      resample of a top-level ragged-response kernel model stays stanc-valid.
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    # (3) COST: the fixed future axis is degenerate, so REBUILDING fails at
    #     construction — the frozen-basis reprocess above is genuinely required.
    @test_throws "hsgp: degenerate input" SBBRMI(
        hsgp_censored_builder(future); mod = @__MODULE__)
    @test_throws "hsgp: degenerate input" reprocess(
        sb, future; freeze_constants = false, resample_groups = [:subject])

    # (4) Unseen fitted subject still fails loudly on the frozen same-group path.
    @test_throws "unseen level" reprocess(
        sb, hsgp_censored_df(; subs = ["s1", "s2", "zzz"], dose_amt = [100.0, 50.0]))

    # (5) A bound-free top-level ragged response is also re-gathered now (the
    #     response gather was un-provenanced before, independent of any bound).
    plain = SBBRMI(flat_builder(flat_data); mod = @__MODULE__)
    @test plain.preproc[:obs_y].kind === :ragged_gather
    reprocessed = reprocess(plain, flat_data; freeze_constants = true)
    @test reprocessed.data[:obs_y] == plain.data[:obs_y]
    @test StanBlocks.stan_code(reprocessed.model) == StanBlocks.stan_code(plain.model)
end
