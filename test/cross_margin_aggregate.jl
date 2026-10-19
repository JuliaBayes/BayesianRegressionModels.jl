# test/cross_margin_aggregate.jl — cross-margin correlated blocks across
# likelihood families, and district-total (sum) observations over facility
# predictors via a label-anchored @plate (snag brm-cross-margin-e5153770).
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

rkprob(brmi) = BRM.rk_logdensity_problem(RKBRMI(brmi);
    ad_backend = AutoEnzyme(; mode = Enzyme.Reverse))
rkdensity(brmi, x = nothing) = let p = rkprob(brmi)
    LogDensityProblems.logdensity(p, something(x, zeros(LogDensityProblems.dimension(p))))
end
ragged_data(sb) = only(v for (k, v) in sb.data if endswith(string(k), "_ragged"))

agg_df = (;
    facility = [1, 2, 3, 4],
    fac_district = [1, 1, 2, 2],
    district = [1, 2],
    Y = [10, 7],
)

agg_builder = @brm begin
    log_lambda ~ 1 + (1 | fac | facility)
    ragged(log_lambda, fac_district => district)
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

    prob = rkprob(brmi)
    n = LogDensityProblems.dimension(prob)
    # Intercept + one scale + four facility effects.
    @test n == 6
    @test isfinite(LogDensityProblems.logdensity(prob, zeros(n)))
end

@testset "totals join their district by label, not by position" begin
    # Facilities 1-3 sit in district "b", facility 4 in "a"; the totals frame
    # lists the districts in either order.
    frame(district, Y) = (; facility = [1, 2, 3, 4],
        fac_district = ["b", "b", "b", "a"], district, Y)
    ba = frame(["b", "a"], [10, 7])
    ab = frame(["a", "b"], [7, 10])
    @test ragged_data(SBBRMI(agg_builder(ba); mod = @__MODULE__)) == [[1, 2, 3], [4]]
    @test ragged_data(SBBRMI(agg_builder(ab); mod = @__MODULE__)) == [[4], [1, 2, 3]]
    x = [0.3, -0.2, 0.1, 0.4, -0.5, 0.2]
    @test rkdensity(agg_builder(ba), x) ≈ rkdensity(agg_builder(ab), x)
    # At zero every facility rate is 1, so "b" has rate 3 and "a" rate 1.
    swapped = frame(["b", "a"], [7, 10])
    @test rkdensity(agg_builder(ba)) - rkdensity(agg_builder(swapped)) ≈
        (logpdf(Poisson(3), 10) + logpdf(Poisson(1), 7)) -
        (logpdf(Poisson(3), 7) + logpdf(Poisson(1), 10))
end

@testset "ragged views over different groupings share one set of cells" begin
    two_views = @brm begin
        log_lambda ~ 1 + (1 | fac | facility)
        log_site ~ 1 + (1 | site | site)
        ragged(log_lambda, fac_district => district)
        ragged(log_site, site_district => district)
        @plate for d in eachindex(Y)
            Y[d] ~ Poisson(sum(exp.(log_lambda[d])) + sum(exp.(log_site[d])))
        end
    end
    df = (; facility = [1, 2, 3, 4], fac_district = [1, 1, 2, 2],
        site = [1, 2, 3], site_district = [2, 1, 2], district = [1, 2], Y = [10, 7])
    sb = SBBRMI(two_views(df); mod = @__MODULE__)
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic = false).ok
    @test isfinite(rkdensity(two_views(df)))
end

@testset "label-anchored plate replays on new districts" begin
    sb = SBBRMI(agg_builder(agg_df); mod = @__MODULE__)
    future = (;
        facility = [1, 2, 3, 4],
        fac_district = [1, 1, 2, 3],
        district = [3, 1, 2],
        Y = [4, 5, 9],
    )
    replay = reprocess(sb, future)
    # Three district cells on the new frame; Stan source is unchanged.
    @test BRM.stan_code(replay) == BRM.stan_code(sb)
    nsub_keys = [k for k in keys(replay.data) if startswith(string(k), "kernel_nsub_")]
    @test length(nsub_keys) == 1
    @test replay.data[only(nsub_keys)] == 3
    @test ragged_data(replay) == [[4], [1, 2], [3]]
end

@testset "label-anchored plate refusals stay loud" begin
    df = (; facility = [1, 2, 3, 4], fac_district = [1, 1, 2, 2], Y = [10, 7])
    # refused: without a cell label column the totals could bind to cells only
    # by position, which nothing can check (user direction 2026-10-19,
    # BayesianRegressionModels:snag.brm-cross-margin-e5153770 messages 09:01).
    unlabelled = @brm begin
        log_lambda ~ 1 + (1 | fac | facility)
        ragged(log_lambda, fac_district)
        @plate for d in eachindex(Y)
            Y[d] ~ Poisson(sum(exp.(log_lambda[d])))
        end
    end
    @test_throws "no cell labels" SBBRMI(unlabelled(df); mod = @__MODULE__)
    @test_throws "no cell labels" rkprob(unlabelled(df))

    # refused: a total needs exactly one district row to bind to.
    @test_throws "per-cell column" SBBRMI(
        agg_builder(merge(df, (; district = [1, 2], Y = [10]))); mod = @__MODULE__)
    @test_throws "disagree on subject count" rkprob(
        agg_builder(merge(df, (; district = [1, 2], Y = [10]))))
    @test_throws "unique subject per row" SBBRMI(
        agg_builder(merge(df, (; district = [1, 1]))); mod = @__MODULE__)
    # refused: a facility whose district has no row would be dropped silently.
    @test_throws "name no subject" SBBRMI(
        agg_builder(merge(df, (; district = [1, 3]))); mod = @__MODULE__)

    # refused: one plate has one set of cells.
    two_cells = @brm begin
        log_lambda ~ 1 + (1 | fac | facility)
        log_other ~ 1 + (1 | fac2 | facility)
        ragged(log_lambda, fac_district => district)
        ragged(log_other, fac_district => region)
        @plate for d in eachindex(Y)
            Y[d] ~ Poisson(sum(exp.(log_lambda[d])) + sum(exp.(log_other[d])))
        end
    end
    @test_throws "different cell label columns" SBBRMI(
        two_cells(merge(df, (; district = [1, 2], region = [1, 2]))); mod = @__MODULE__)

    # refused: a random-effect predictor already labels the cells by its own
    # grouping, so a different cell label column contradicts it.
    ranef_cells = @brm begin
        log_lambda ~ 1 + (1 | fac | facility)
        log_d ~ 1 + (1 | dist | district)
        ragged(log_lambda, fac_district => region)
        @plate for d in eachindex(log_d)
            Y[d] ~ Poisson(exp(log_d[d]) + sum(exp.(log_lambda[d])))
        end
    end
    @test_throws "already labels them" SBBRMI(
        ranef_cells(merge(df, (; district = [1, 2], region = [1, 2]))); mod = @__MODULE__)
end
