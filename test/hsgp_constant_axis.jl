# A data-derived HSGP basis interval needs two distinct axis values, so a
# fresh fit refuses a constant axis at every row count. The refusal used to
# read the computed half-width, and the mean of n equal values rounds an ulp
# away from them for most n: a constant axis then passed with a half-width of
# rounding noise and a basis scaled by 1/sqrt(L) (about 3e7). A frozen replay
# keeps the fitted interval, and an explicit `domain=` fixes one, so both still
# evaluate a constant axis.
using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal
const BRM = BayesianRegressionModels

const LOG_DOSE = log(100.0)
const MAX_ROWS = 64
# Row counts whose mean of LOG_DOSE rounds away from LOG_DOSE itself.
const ROUNDING_ROWS = filter(n -> sum(fill(LOG_DOSE, n)) / n != LOG_DOSE, 1:MAX_ROWS)

constant_axis_df(n) = (; log_dose=fill(LOG_DOSE, n), y=zeros(n))
const CONSTANT_AXIS_TRAIN =
    (; log_dose=log.([25.0, 50.0, 100.0, 200.0, 400.0, 800.0]), y=zeros(6))

constant_axis_builder = @brm begin
    mu ~ 1 + hsgp(log_dose; k=5)
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end

domain_axis_builder = @brm begin
    mu ~ 1 + hsgp(log_dose; k=5, domain=(2.0, 8.0))
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end

constant_axis_model(builder, df) = SBBRMI(builder(df); mod=@__MODULE__)

@testset "the shared fit refuses a constant axis at every row count" begin
    # Positive control: the rows that used to slip through exist.
    @test !isempty(ROUNDING_ROWS)
    for n in 1:MAX_ROWS
        axis = fill(LOG_DOSE, n)
        @test_throws "hsgp: degenerate input" BRM._brm_fit_hsgp(axis, 5, 1.5)
        @test_throws "hsgp: degenerate input" BRM._brm_fit_hsgp(
            (axis, collect(range(0.0, 1.0; length=n + 1))[1:n]), (5, 3), (1.5, 1.5))
        # The vectorised backend's basis is the same law.
        @test_throws "hsgp: degenerate input" BRM._hsgp_basis(axis, 5, 1.5)
    end
    # An anchor at the constant value adds no width; another anchor does.
    @test_throws "hsgp: degenerate input" BRM._brm_fit_hsgp(
        fill(LOG_DOSE, first(ROUNDING_ROWS)), 5, 1.5; anchor=LOG_DOSE)
    _, anchored = BRM._brm_fit_hsgp(fill(LOG_DOSE, first(ROUNDING_ROWS)), 5, 1.5;
                                    anchor=0.0)
    @test anchored ≈ 1.5 * LOG_DOSE
    # Two distinct values fit the interval they span.
    center, half_width = BRM._brm_fit_hsgp([LOG_DOSE, log(200.0)], 5, 1.5)
    @test center ≈ (LOG_DOSE + log(200.0)) / 2
    @test half_width ≈ 1.5 * log(2.0) / 2
    message = try
        BRM._brm_fit_hsgp(fill(LOG_DOSE, 3), 5, 1.5)
    catch err
        sprint(showerror, err)
    end
    @test occursin("domain=(lower, upper)", message)
    @test occursin("freeze_constants=true", message)
end

@testset "a fresh refit onto a constant axis refuses; a frozen replay evaluates it" begin
    fitted = constant_axis_model(constant_axis_builder, CONSTANT_AXIS_TRAIN)
    fits = fitted.preproc[:PHI_hsgp_log_dose].const_.fits
    for n in ROUNDING_ROWS[1:3]
        slice = constant_axis_df(n)
        @test_throws "hsgp: degenerate input" reprocess(
            fitted, slice; freeze_constants=false)
        @test_throws "hsgp: degenerate input" constant_axis_model(
            constant_axis_builder, slice)
        frozen = reprocess(fitted, slice)
        @test frozen.preproc[:PHI_hsgp_log_dose].const_.fits == fits
        PHI = frozen.data[:PHI_hsgp_log_dose]
        @test size(PHI) == (n, 5)
        @test all(r -> r == PHI[1, :], eachrow(PHI))
        # Every basis entry is bounded by 1/sqrt(L) of the fitted interval.
        @test maximum(abs, PHI) <= 1 / sqrt(only(fits)[2])
        @test frozen.data[:omega2_hsgp_log_dose] == fitted.data[:omega2_hsgp_log_dose]
    end
end

@testset "an explicit domain fits a constant axis, fresh or frozen" begin
    slice = constant_axis_df(first(ROUNDING_ROWS))
    built = constant_axis_model(domain_axis_builder, slice)
    @test built.preproc[:PHI_hsgp_log_dose].const_.fits == ((5.0, 3.0),)
    @test maximum(abs, built.data[:PHI_hsgp_log_dose]) <= 1 / sqrt(3.0)
    fitted = constant_axis_model(domain_axis_builder, CONSTANT_AXIS_TRAIN)
    fresh = reprocess(fitted, slice; freeze_constants=false)
    @test fresh.data[:PHI_hsgp_log_dose] == built.data[:PHI_hsgp_log_dose]
    @test fresh.data[:omega2_hsgp_log_dose] == fitted.data[:omega2_hsgp_log_dose]
end
