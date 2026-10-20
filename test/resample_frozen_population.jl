# test/resample_frozen_population.jl — `reprocess(...; resample_groups)` keeps
# every fitted population-design constant (snag `resample-reproce-b5f9c8dd`).
#
# A resample replay re-emits the model on the replay frame. That re-emission
# used to refit `zscale`/`standardize`/`center` and string level sets on the
# replay rows before the frozen training values replaced them, so a frame with
# one new subject (or tied covariates, or only some levels) raised. Emitted
# names of expression columns hashed the data vector's identity, so a
# re-emission could not find their fitted records at all.
#
# Run: julia --startup-file=no --project=test test/resample_frozen_population.jl

using Test
using BayesianRegressionModels
using StanBlocks
using Statistics: mean, std

const BRM = BayesianRegressionModels

# One row per subject: a per-subject linear predictor of standardized covariates.
subject_builder = @brm begin
    mu ~ 1 + standardize(age) + zscale(wt) + center(ht) + (1 | p | subject)
    y ~ Normal(mu, 1.0)
end
subject_train = (; subject=[1, 2, 3, 4], age=[30.0, 45.0, 60.0, 52.0],
                   wt=[60.0, 70.0, 82.0, 75.0], ht=[160.0, 170.0, 180.0, 175.0],
                   y=zeros(4))
one_subject = (; subject=[101], age=[41.0], wt=[66.0], ht=[168.0], y=[0.0])
tied_subjects = (; subject=[101, 102], age=[41.0, 41.0], wt=[66.0, 66.0],
                   ht=[168.0, 168.0], y=zeros(2))
two_subjects = (; subject=[101, 102], age=[41.0, 50.0], wt=[66.0, 71.0],
                  ht=[168.0, 172.0], y=zeros(2))

stanc_ok(sb) = StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok

@testset "one new subject replays on the training transform" begin
    fit = SBBRMI(subject_builder(subject_train); mod=@__MODULE__, total_groups=())
    reference = reprocess(fit, two_subjects; resample_groups=[:subject])
    for frame in (one_subject, tied_subjects)
        replay = reprocess(fit, frame; resample_groups=[:subject])
        @test BRM.stan_code(replay) == BRM.stan_code(reference)
        @test stanc_ok(replay)
        for key in (:standardize_age, :zscale_wt, :center_ht)
            @test replay.preproc[key].const_ == fit.preproc[key].const_
        end
        data = StanBlocks.stan_data(replay.model)
        @test data[:standardize_age] ≈
            (frame.age .- mean(subject_train.age)) ./ std(subject_train.age)
        @test data[:zscale_wt] ≈
            (frame.wt .- mean(subject_train.wt)) ./ std(subject_train.wt)
        @test data[:center_ht] ≈ frame.ht .- mean(subject_train.ht)
        @test replay.data[:n_subject] == length(frame.subject)
        @test restan_data(fit, frame; resample_groups=[:subject]) == data
        plan = reprocess(generative_plan(fit), frame; resample_groups=[:subject])
        @test StanBlocks.stan_data(plan.model) == data
    end
    # refused: a fresh fit of a sample SD on one value is undefined, and
    # `freeze_constants=false` asks for exactly that fit.
    @test_throws "requires at least two training values" reprocess(
        fit, one_subject; freeze_constants=false, resample_groups=[:subject])
end

@testset "a random slope on a standardized covariate replays one subject" begin
    builder = @brm begin
        mu ~ 1 + zscale(age) + (1 + zscale(age) | p | subject)
        y ~ Normal(mu, 1.0)
    end
    train = (; subject=repeat([1, 2, 3], inner=2),
               age=[30.0, 35.0, 45.0, 50.0, 60.0, 66.0], y=zeros(6))
    fit = SBBRMI(builder(train); mod=@__MODULE__, total_groups=())
    frame = (; subject=[101, 101], age=[41.0, 41.0], y=zeros(2))
    replay = reprocess(fit, frame; resample_groups=[:subject])
    @test stanc_ok(replay)
    @test replay.preproc[:zscale_age].const_ == fit.preproc[:zscale_age].const_
    @test StanBlocks.stan_data(replay.model)[:zscale_age] ≈
        (frame.age .- mean(train.age)) ./ std(train.age)
end

@testset "expression columns are named by the formula, not the data" begin
    builder = @brm begin
        mu ~ 1 + zscale(log(age)) + log(wt) + (1 | p | subject)
        y ~ Normal(mu, 1.0)
    end
    fit = SBBRMI(builder(subject_train); mod=@__MODULE__, total_groups=())
    copied = SBBRMI(builder(deepcopy(subject_train)); mod=@__MODULE__, total_groups=())
    @test BRM.stan_code(copied) == BRM.stan_code(fit)
    labels = sort!([key for key in keys(fit.preproc) if occursin("_expr_", String(key))])
    @test length(labels) == 2
    @test labels == sort!([key for key in keys(copied.preproc)
                           if occursin("_expr_", String(key))])
    for frame in (two_subjects, one_subject)
        replay = reprocess(fit, frame; resample_groups=[:subject])
        @test stanc_ok(replay)
        zkey = only(filter(key -> startswith(String(key), "zscale_expr_"), labels))
        @test replay.preproc[zkey].const_ == fit.preproc[zkey].const_
        @test StanBlocks.stan_data(replay.model)[zkey] ≈
            (log.(frame.age) .- mean(log.(subject_train.age))) ./
            std(log.(subject_train.age))
    end
end

@testset "string covariates keep their fitted levels" begin
    train = merge(subject_train, (; sex=["f", "m", "f", "m"]))
    treatment = @brm begin
        mu ~ 1 + sex + zscale(age) + zscale(age) & sex + (1 | p | subject)
        y ~ Normal(mu, 1.0)
    end
    cellmeans = @brm begin
        mu ~ 0 + sex + (1 | p | subject)
        y ~ Normal(mu, 1.0)
    end
    for (builder, keys_) in ((treatment, (:sex_lvl_2, :int_zscale_age_x_sex_lvl_2)),
                             (cellmeans, (:sex_lvl_1, :sex_lvl_2)))
        fit = SBBRMI(builder(train); mod=@__MODULE__, total_groups=())
        reference = reprocess(fit, merge(two_subjects, (; sex=["f", "m"]));
                              resample_groups=[:subject])
        for sex in (["m"], ["f"])
            frame = merge(one_subject, (; sex))
            replay = reprocess(fit, frame; resample_groups=[:subject])
            @test BRM.stan_code(replay) == BRM.stan_code(reference)
            @test stanc_ok(replay)
            data = StanBlocks.stan_data(replay.model)
            for key in keys_
                @test replay.preproc[key].const_ == fit.preproc[key].const_
            end
            @test data[:sex_lvl_2] == (sex == ["m"] ? [1.0] : [0.0])
        end
        # refused: a frozen replay has no fitted parameter for an unseen level.
        @test_throws "is not a training level" reprocess(
            fit, merge(one_subject, (; sex=["x"])); resample_groups=[:subject])
    end
end

@testset "a releveled integer factor on a frame without its reference" begin
    builder = @brm begin
        mu ~ 1 + factor(sex; ref=2) + (1 | p | subject)
        y ~ Normal(mu, 1.0)
    end
    fit = SBBRMI(builder(merge(subject_train, (; sex=[1, 2, 1, 2])));
                 mod=@__MODULE__, total_groups=())
    # Capability gap, not a refusal: `factor(...; ref)` validates its
    # reference against the replay frame during term lowering, before the
    # frozen level set applies (follow-up of snag `resample-reproce-b5f9c8dd`).
    @test_broken stanc_ok(reprocess(fit, merge(one_subject, (; sex=[1]));
                                    resample_groups=[:subject]))
end
