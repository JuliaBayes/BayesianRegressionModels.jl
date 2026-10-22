# test/plate_grouped_response.jl — a flat response observed per subject in the
# cell that computes its prediction, through a `groups` binding.
#
# Run: julia --startup-file=no --project=test test/plate_grouped_response.jl
#
# `obs = groups(obs_subject => subject)` binds each subject's observation
# indices; the cell observes `y[obs[i]] ~ Family(pred[i], …)`, as the same
# Julia loop would (todo `1nfukuv`, decision `0ivno7u`). The formula-level
# `ragged(y, group) ~ …` it replaces is refused. The flat response must give
# the same density and gradient as the same response grouped by hand, bounds
# read through the binding included, and replay (`reprocess`) must regather
# both for new data.

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal
using LogDensityProblems

flat_builder = @brm begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        pred[i] = exp(log_CL[i]) .+ 0.0 .* obs_idx[obs[i]]
        obs_y[obs[i]] ~ Normal(pred[i], sigma)
    end
end

# The same model on a response grouped by hand, in `subject` order.
grouped_builder = @brm begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        pred[i] = exp(log_CL[i]) .+ 0.0 .* obs_idx[obs[i]]
        obs_y[i] ~ Normal(pred[i], sigma)
    end
end

censored_flat_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i]) .+ 0.0 .* obs_idx[obs[i]]
        pk_conc[obs[i]] ~ censored(
            Normal(loc[i], addprop(loc[i], sigma_add, sigma_prop)); lower=pk_lloq[obs[i]])
    end
end

censored_grouped_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i]) .+ 0.0 .* obs_idx[obs[i]]
        pk_conc[i] ~ censored(
            Normal(loc[i], addprop(loc[i], sigma_add, sigma_prop)); lower=pk_lloq[i])
    end
end

censored_scalar_bound_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i]) .+ 0.0 .* obs_idx[obs[i]]
        pk_conc[obs[i]] ~ censored(
            Normal(loc[i], addprop(loc[i], sigma_add, sigma_prop)); lower=0.10)
    end
end

truncated_two_bound_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i]) .+ 0.0 .* obs_idx[obs[i]]
        pk_conc[obs[i]] ~ truncated(
            Normal(loc[i], addprop(loc[i], sigma_add, sigma_prop));
            lower=pk_lloq[obs[i]], upper=pk_uloq[obs[i]])
    end
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

function same_density(a, a_data, b, b_data; scale=0.05)
    pa = brm_execute(brm_descriptor(a, a_data; mod=@__MODULE__), :fit)
    pb = brm_execute(brm_descriptor(b, b_data; mod=@__MODULE__), :fit)
    dim = LogDensityProblems.dimension(pa)
    @test LogDensityProblems.dimension(pb) == dim
    q = [scale * ((i % 7) - 3) for i in 1:dim]
    lp_a, grad_a = LogDensityProblems.logdensity_and_gradient(pa, q)
    lp_b, grad_b = LogDensityProblems.logdensity_and_gradient(pb, q)
    @test isfinite(lp_a) && all(isfinite, grad_a)
    @test lp_a ≈ lp_b atol=1e-12 rtol=1e-12
    @test grad_a ≈ grad_b atol=1e-12 rtol=1e-12
end

@testset "a flat response observed in the cell through a binding" begin
    @test groups(flat_data.obs_subject => flat_data.subject) == [[2, 4], [1], [3, 5, 6]]
    @test [flat_data.obs_y[r] for r in groups(flat_data.obs_subject => flat_data.subject)] ==
        expected
    flat = SBBRMI(flat_builder(flat_data); mod=@__MODULE__)
    @test StanBlocks.stanc_check(BayesianRegressionModels.stan_code(flat);
        warn_pedantic=false).ok
    same_density(flat_builder, flat_data, grouped_builder, grouped_data)

    @testset "invalid axes fail before Stan" begin
        mismatched = merge(flat_data, (; obs_subject=flat_data.obs_subject[1:end-1]))
        @test_throws "grouping column must name" SBBRMI(
            flat_builder(mismatched); mod=@__MODULE__)
        unknown = merge(flat_data, (;
            obs_subject=["s1", "s2", "s3", "s2", "s3", "not-a-subject"],
        ))
        @test_throws "name no subject" SBBRMI(flat_builder(unknown); mod=@__MODULE__)
        already_grouped = merge(flat_data, (; obs_y=expected))
        @test_throws "ALREADY a ragged per-subject column" SBBRMI(
            flat_builder(already_grouped); mod=@__MODULE__)
    end
end

@testset "observed bounds read through the binding" begin
    flat = SBBRMI(censored_flat_builder(censored_flat_data); mod=@__MODULE__)
    grouped = SBBRMI(censored_grouped_builder(censored_grouped_data); mod=@__MODULE__)
    for brmi in (flat, grouped,
                 SBBRMI(censored_scalar_bound_builder(censored_flat_data); mod=@__MODULE__),
                 SBBRMI(truncated_two_bound_builder(censored_flat_data); mod=@__MODULE__))
        @test StanBlocks.stanc_check(BayesianRegressionModels.stan_code(brmi);
            warn_pedantic=false).ok
    end
    same_density(censored_flat_builder, censored_flat_data,
        censored_grouped_builder, censored_grouped_data; scale=-0.07)

    short_bound = merge(censored_flat_data, (;
        pk_lloq=censored_flat_data.pk_lloq[1:end-1],
    ))
    @test_throws "grouping column must name" SBBRMI(
        censored_flat_builder(short_bound); mod=@__MODULE__)
end

@testset "the formula-level `ragged(y, group) ~ …` is retired" begin
    retired = :(@brm begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + (1 | p | subject)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(log_CL)
            pred[i] = exp(log_CL[i]) .+ 0.0 .* obs_idx[obs[i]]
        end
        ragged(obs_y, obs_subject) ~ Normal(pred, sigma)
    end)
    @test_throws "are retired" macroexpand(@__MODULE__, retired)
    @test_throws "y[rows[i]] ~" macroexpand(@__MODULE__, retired)
    # refused: wrapping the retired LHS does not reach a backend (todo `1nfukuv`).
    for lhs in (:(mi(ragged(obs_y, obs_subject))), :(log(ragged(obs_y, obs_subject))))
        wrapped = :(@brm begin
            sigma ~ Exponential(1)
            $lhs ~ Normal(0, sigma)
        end)
        @test_throws "are retired" macroexpand(@__MODULE__, wrapped)
    end
end

# snag reprocess-ragged-ea8c7e42 (reported by a downstream PKPD consumer): a
# grouped response and its data-backed censoring bound are regathered from
# their flat vectors on replay. A fixed future regimen makes the event-axis
# HSGP covariate constant, so rebuilding the model refits a degenerate basis —
# the frozen reprocess is the only correct route. Replay doses stay inside the
# trained HSGP interval (`1ec23de2` refuses values past it).
hsgp_censored_builder = @brm begin
    sigma  ~ Exponential(1)
    log_F  ~ 1 + hsgp(log_dose; k = 5)
    log_CL ~ 1 + weight + (1 | p | subject)
    obs = groups(obs_subject => subject)
    doses = groups(dose_subject => subject)
    @plate for i in eachindex(log_CL)
        pred[i] = sum(dose_amount_flat[doses[i]] .* exp.(log_F[doses[i]])) .*
            exp.(-exp(log_CL[i]) .* obs_time[obs[i]])
        pk_conc[obs[i]] ~ censored(Normal(pred[i], sigma); lower = pk_lloq[obs[i]])
    end
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

# The data a cell reads for `name`, whatever its prepared key.
cell_data(sb, name) = only(v for (k, v) in sb.data
    if occursin(string(name), string(k)) && endswith(string(k), "_ragged"))

@testset "a grouped response and its bound replay and resample" begin
    train = hsgp_censored_df(; subs = ["s1", "s2", "s3"], dose_amt = [100.0, 50.0])
    # `total_groups=()`: this testset exercises the conventional GQ resample
    # path; the default `:auto` would integrate the single-margin subject
    # bucket into totals, which `resample_groups` refuses.
    sb = SBBRMI(hsgp_censored_builder(train); mod = @__MODULE__, total_groups = ())

    # Same-subject frozen replay: identical Stan, correctly regathered.
    same = hsgp_censored_df(; subs = ["s1", "s2", "s3"], dose_amt = [80.0, 60.0])
    replayed = reprocess(sb, same; freeze_constants = true)
    @test StanBlocks.stan_code(replayed.model) == StanBlocks.stan_code(sb.model)
    @test cell_data(replayed, :pk_conc) == [[1.2, 0.8, 0.5, 0.3] for _ in 1:3]
    @test cell_data(replayed, :pk_lloq) == [[0.1, 0.1, 0.1, 0.1] for _ in 1:3]

    # New-population resample onto a fixed 90 mg future: succeeds, freezes
    # the trained HSGP basis, and regathers response and bound for the new
    # subjects; the resampled program is stanc-valid.
    future = hsgp_censored_df(; subs = ["n1", "n2", "n3", "n4"], dose_amt = [90.0, 90.0])
    resampled = reprocess(sb, future; resample_groups = [:subject])
    @test resampled.preproc[:subject_idx].const_.levels == ["n1", "n2", "n3", "n4"]
    @test resampled.preproc[:PHI_hsgp_log_dose].const_.fits ==
          sb.preproc[:PHI_hsgp_log_dose].const_.fits
    @test cell_data(resampled, :pk_conc) == [[1.2, 0.8, 0.5, 0.3] for _ in 1:4]
    @test cell_data(resampled, :pk_lloq) == [[0.1, 0.1, 0.1, 0.1] for _ in 1:4]
    @test StanBlocks.stanc_check(StanBlocks.stan_code(resampled.model);
        warn_pedantic=false).ok

    # The fixed future vector is degenerate, so rebuilding fails at
    # construction — the frozen-basis reprocess above is genuinely required.
    @test_throws "hsgp: degenerate input" SBBRMI(
        hsgp_censored_builder(future); mod = @__MODULE__)
    # An unseen fitted subject still fails loudly on the frozen same-group path.
    @test_throws "unseen level" reprocess(
        sb, hsgp_censored_df(; subs = ["s1", "s2", "zzz"], dose_amt = [100.0, 50.0]))
end
