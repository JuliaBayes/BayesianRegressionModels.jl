using Test
using BayesianRegressionModels
using Distributions, Turing

const BRM = BayesianRegressionModels

@testset "shared fitted population preprocessing replay" begin
    future = fill(4.0, 3)
    population = BRM._BRMPopulationPreprocess(:zscale, (2.0, 2.0), :x)
    sb_entry = BRM._BRMPreprocEntry(:zscale, (2.0, 2.0), :x, false)

    population_entry = BRM._brm_population_preproc_entry(population)
    population_replay = BRM._brm_replay_preprocess(
        population_entry, future; freeze=true)
    sb_replay = BRM._brm_replay_preprocess(sb_entry, future; freeze=true)

    @test population_replay.values == sb_replay.values
    @test population_replay.values.primary == ones(3)
    @test population_replay.entry.const_ == (2.0, 2.0)
    @test sb_replay.entry.const_ == (2.0, 2.0)
end

replay_calibration(x; shift=0.0, gain=1.0) = gain * x + shift

@testset "shared callable data expressions retain keywords during fit and replay" begin
    builder = @brm begin
        mu ~ 1 + replay_calibration(x; shift=shift, gain=2.0) +
                 zscale(replay_calibration(x; shift=shift))
        y ~ Normal(mu, 1)
    end
    training = (; x=[-1.0, 0.0, 2.0, 3.0],
                 shift=[0.2, -0.1, 0.3, 0.5], y=zeros(4))
    future = (; x=[0.5, 4.0], shift=[0.1, -0.4], y=zeros(2))
    fitted = BRM._brm_fit_zscale(training.x .+ training.shift)
    expected(data) = (2 .* data.x .+ data.shift,
                     BRM._brm_apply_zscale(fitted, data.x .+ data.shift))
    sb = SBBRMI(builder(training); mod=@__MODULE__)
    native = TuringBRMI(builder(training))
    for (data, stan, turing) in ((training, sb, native),
            (future, reprocess(sb, future), reprocess(native, future)))
        plain, scaled = expected(data)
        @test turing.plan.design.matrix[:, 2] ≈ plain
        @test turing.plan.design.matrix[:, 3] ≈ scaled
        plain_key = only(key for (key, entry) in stan.preproc if entry.kind === :protect)
        scale_key = only(key for (key, entry) in stan.preproc if entry.kind === :zscale)
        @test stan.data[plain_key] ≈ plain
        @test stan.data[scale_key] ≈ scaled
        params = (; beta_pop=[0.1, 0.2, -0.3])
        mu = 0.1 .+ 0.2 .* plain .- 0.3 .* scaled
        @test Turing.loglikelihood(turing.model, params) ≈ sum(logpdf.(Normal.(mu, 1), data.y))
    end
end

@testset "shared fitted preprocessing refit and compound values" begin
    centered = BRM._BRMPreprocEntry(:center, 100.0, :x, false)
    replay = BRM._brm_replay_preprocess(
        centered, [1.0, 2.0, 3.0]; freeze=false)
    @test replay.entry.const_ == 2.0
    @test replay.values.primary == [-1.0, 0.0, 1.0]

    interaction = BRM._BRMPreprocEntry(
        :interaction, nothing, (:left, :right), false)
    product = BRM._brm_replay_preprocess(
        interaction, ([1.0, 2.0], [3.0, 4.0]); freeze=true)
    @test product.values.primary == [3.0, 8.0]

    factor = BRM._BRMPreprocEntry(
        :population_factor_dummy,
        (; levels=[1, 2, 3], level=2, n_levels=3, ref=1), :group, true)
    dummy = BRM._brm_replay_preprocess(
        factor, [3, 1, 2, 3]; freeze=true)
    @test dummy.values.primary == [0.0, 0.0, 1.0, 0.0]
    @test_throws ErrorException BRM._brm_replay_preprocess(
        factor, [1, 4]; freeze=true)
end
