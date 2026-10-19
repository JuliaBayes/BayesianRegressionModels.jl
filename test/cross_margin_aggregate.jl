# test/cross_margin_aggregate.jl — cross-margin correlated blocks across
# likelihood families, and district-total (sum) observations over facility
# predictors via a ragged-anchored @plate (snag brm-cross-margin-e5153770).
#
# Run:
#   julia --startup-file=no --project=test test/cross_margin_aggregate.jl

using Test
using BayesianRegressionModels
using DifferentiationInterface: AutoEnzyme
using Distributions
using Enzyme: Enzyme
using LogDensityProblems
using ReactiveKernels
using ReactiveKernelsPPL
using StanBlocks

const BRM = BayesianRegressionModels

cross_df = (;
    facility = [1, 1, 2, 2, 3, 3, 4, 4],
    intervention = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    y_count = [3, 5, 2, 4, 1, 3, 4, 6],
    hours = [10.0, 12.0, 9.0, 11.0, 8.0, 10.5, 9.5, 13.0],
)

cross_builder = @brm begin
    sigma_h ~ Exponential(1)
    log(lambda) ~ 1 + intervention + (1 + intervention | fac | facility)
    log_mu ~ 1 + (1 | fac | facility)
    sd(:, fac) ~ Exponential(1)
    cor(:, fac) ~ LKJCholesky(3, 2)
    y_count ~ Poisson(lambda)
    hours ~ LogNormal(log_mu, sigma_h)
end

@testset "cross-margin block spans Poisson and LogNormal predictors" begin
    brmi = cross_builder(cross_df)
    @test BRM.ranefcoefnames(brmi, :fac) == [
        (predictor = :lambda, coefficient = :Intercept),
        (predictor = :lambda, coefficient = :intervention),
        (predictor = :log_mu, coefficient = :Intercept),
    ]
    sb = SBBRMI(brmi; mod = @__MODULE__)
    code = BRM.stan_code(sb)
    @test occursin("lkj_corr_cholesky(2.0)", code)
    @test occursin("diag_pre_multiply", code)
    @test occursin("poisson", lowercase(code))
    @test StanBlocks.stanc_check(code; warn_pedantic = false).ok
end

@testset "cross-margin block on RK needs the link spelling" begin
    rk = RKBRMI(cross_builder(cross_df))
    prob = BRM.rk_logdensity_problem(rk; ad_backend = AutoEnzyme(; mode = Enzyme.Reverse))
    n = LogDensityProblems.dimension(prob)
    @test n > 0
    @test isfinite(LogDensityProblems.logdensity(prob, zeros(n)))
end

agg_df = (;
    facility = [1, 2, 3, 4],
    fac_district = [1, 1, 2, 2],
    Y = [10, 7],
)

agg_builder = @brm begin
    log_lambda ~ 1 + (1 | fac | facility)
    ragged(log_lambda, fac_district)
    @plate for d in eachindex(Y)
        lam_d = sum(exp.(log_lambda[d]))
        Y[d] ~ Poisson(lam_d)
    end
end

@testset "district total observes the summed facility rates" begin
    brmi = agg_builder(agg_df)
    sb = SBBRMI(brmi; mod = @__MODULE__)
    code = BRM.stan_code(sb)
    @test occursin("poisson", lowercase(code))
    @test occursin("sum(", code)
    @test StanBlocks.stanc_check(code; warn_pedantic = false).ok

    rk = RKBRMI(brmi)
    prob = BRM.rk_logdensity_problem(rk; ad_backend = AutoEnzyme(; mode = Enzyme.Reverse))
    n = LogDensityProblems.dimension(prob)
    # Intercept + one scale + four facility effects.
    @test n == 6
    @test isfinite(LogDensityProblems.logdensity(prob, zeros(n)))
end

@testset "ragged-anchored plate replays on new districts" begin
    sb = SBBRMI(agg_builder(agg_df); mod = @__MODULE__)
    future = (;
        facility = [1, 2, 3, 4],
        fac_district = [1, 1, 2, 3],
        Y = [5, 9, 4],
    )
    replay = reprocess(sb, future)
    # Three district cells on the new frame; Stan source is unchanged.
    @test BRM.stan_code(replay) == BRM.stan_code(sb)
    nsub_keys = [k for k in keys(replay.data) if startswith(string(k), "kernel_nsub_")]
    @test length(nsub_keys) == 1
    @test replay.data[only(nsub_keys)] == 3
end

@testset "ragged-anchored plate refusals stay loud" begin
    # Per-cell totals must carry one entry per distinct district label.
    bad_len = @brm begin
        log_lambda ~ 1 + (1 | fac | facility)
        ragged(log_lambda, fac_district)
        @plate for d in eachindex(Y)
            Y[d] ~ Poisson(sum(exp.(log_lambda[d])))
        end
    end
    @test_throws "one entry per cell" SBBRMI(
        bad_len((; facility = [1, 2, 3, 4], fac_district = [1, 1, 2, 2], Y = [10])); mod = @__MODULE__)

    # Two ragged views over different grouping columns cannot anchor a plate.
    two_groups = @brm begin
        log_lambda ~ 1 + (1 | fac | facility)
        log_other ~ 1 + (1 | fac2 | facility)
        ragged(log_lambda, fac_district)
        ragged(log_other, other_district)
        @plate for d in eachindex(Y)
            Y[d] ~ Poisson(sum(exp.(log_lambda[d])) + sum(exp.(log_other[d])))
        end
    end
    df2 = (;
        facility = [1, 2, 3, 4], fac_district = [1, 1, 2, 2],
        other_district = [1, 2, 1, 2], Y = [10, 7],
    )
    @test_throws "share one grouping" SBBRMI(two_groups(df2); mod = @__MODULE__)
end
