using Test
using BayesianRegressionModels
using StanBlocks
using Distributions
using LogDensityProblems
import StanBlocks.stan: transpiles

const BRM = BayesianRegressionModels

const HYPER_RUN_BRIDGESTAN = get(ENV, "BRM_GP_RUNTIME", "1") != "0"
const HYPER_CACHE = joinpath(tempdir(), "brm-hyper-predictors")

function hyper_df()
    x = collect(range(-1.0, 1.0; length=8))
    y = sin.(x)
    g = repeat(["a", "b"], inner=4)
    (; x, y, g)
end

hyper_grouped_model(df) = @brm df begin
    loc ~ 1 + hsgp(x; k=4, by=g)
    log(length_scale(hsgp(x))) ~ 1 + (1 | g)
    log(sd(hsgp(x))) ~ 1 + (1 | g)
    y ~ Normal(loc, 1)
end

hyper_smooth_model(df) = @brm df begin
    loc ~ 1 + hsgp(x; k=4, by=g)
    log(length_scale(hsgp(x))) ~ 1 + s(x)
    y ~ Normal(loc, 1)
end

hyper_ungrouped_ranef_model(df) = @brm df begin
    loc ~ 1 + hsgp(x; k=4)
    log(length_scale(hsgp(x))) ~ 1 + (1 | g)
    y ~ Normal(loc, 1)
end

hyper_unmatched_model(df) = @brm df begin
    loc ~ 1 + hsgp(x; k=4, by=g)
    log(length_scale(hsgp(z))) ~ 1 + (1 | g)
    y ~ Normal(loc, 1)
end

hyper_prior_rhs_model(df) = @brm df begin
    loc ~ 1 + hsgp(x; k=4, by=g)
    log(length_scale(hsgp(x))) ~ LogNormal(0, 1)
    y ~ Normal(loc, 1)
end

hyper_ungrouped_model(df) = @brm df begin
    loc ~ 1 + hsgp(x; k=4)
    log(length_scale(hsgp(x))) ~ 1
    y ~ Normal(loc, 1)
end

hyper_intercept_prior_model(df) = @brm df begin
    loc ~ 1 + hsgp(x; k=4, by=g)
    log(length_scale(hsgp(x))) ~ 1 + (1 | g)
    log(sd(hsgp(x))) ~ 1 + (1 | g)
    length_scale(:, hsgp(x)) ~ Normal(0, 0.5)
    sd(:, hsgp(x)) ~ Normal(0, 0.5)
    y ~ Normal(loc, 1)
end

hyper_mixed_model(df) = @brm df begin
    loc ~ 1 + hsgp(x; k=4, by=g)
    log(length_scale(hsgp(x))) ~ 1 + (1 | g)
    y ~ Normal(loc, 1)
end

@testset "hyper-predictor spelling builds" begin
    df = hyper_df()
    sb = SBBRMI(hyper_grouped_model(df); mod=@__MODULE__)
    @test sb isa SBBRMI
    @test transpiles(sb.model)
    code = BRM.stan_code(sb)
    # Predicted hypers replace the shared sampled scalars.
    @test occursin("beta0_rho", code)
    @test occursin("beta0_sigma", code)
    @test !occursin("rho_iso", code)
    @test !occursin("lognormal", code)
end

@testset "hyper-predictor ungrouped scalar" begin
    df = hyper_df()
    sb = SBBRMI(hyper_ungrouped_model(df); mod=@__MODULE__)
    @test sb isa SBBRMI
    @test transpiles(sb.model)
    code = BRM.stan_code(sb)
    @test occursin("beta0_rho", code)
    @test occursin("lognormal", code)
end

@testset "hyper-predictor intercept priors retarget" begin
    df = hyper_df()
    sb = SBBRMI(hyper_intercept_prior_model(df); mod=@__MODULE__)
    @test transpiles(sb.model)
    code = BRM.stan_code(sb)
    @test occursin("beta0_rho", code)
    @test occursin("beta0_sigma", code)
    @test occursin("0.5", code)
    @test !occursin("rho_iso", code)
    @test !occursin("lognormal", code)
end

@testset "hyper-predictor mixed sampled-predicted" begin
    df = hyper_df()
    sb = SBBRMI(hyper_mixed_model(df); mod=@__MODULE__)
    @test transpiles(sb.model)
    code = BRM.stan_code(sb)
    @test occursin("beta0_rho", code)
    @test !occursin("rho_iso", code)
    @test occursin("sigma", code)
    @test occursin("lognormal", code)
end

@testset "hyper-predictor refusals stay loud" begin
    # Refusals fire at SBBRMI construction, where the term context exists —
    # the `@brm` macro only normalises the spelling.
    df = hyper_df()
    @test_throws "smooth" SBBRMI(hyper_smooth_model(df); mod=@__MODULE__)
    @test_throws "without grouping" SBBRMI(hyper_ungrouped_ranef_model(df); mod=@__MODULE__)
    @test_throws "matches no" SBBRMI(hyper_unmatched_model(df); mod=@__MODULE__)
    @test_throws "predictor formula" SBBRMI(hyper_prior_rhs_model(df); mod=@__MODULE__)
end
