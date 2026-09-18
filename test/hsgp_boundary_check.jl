# test/hsgp_boundary_check.jl — post-fit HSGP boundary verdict + hyper
# floor-binding exposure (hyper follow-ups Q1+Q2, owner discretion).
#
# Run: julia --project=test test/hsgp_boundary_check.jl
#
# All draws are synthetic: the check reads carriers through
# `brm_term_coordinates` against hand-built constrained axes (the
# `test/prior_only_coordinates.jl` precedent), so no BridgeStan compile is
# needed — the geometry under test is drawn, not sampled.
using Test
using BayesianRegressionModels
using Distributions: LogNormal, Normal

const BRM = BayesianRegressionModels

function boundary_df()
    x = collect(range(-1.0, 1.0; length=8))
    x2 = x .^ 2 .+ 0.1 .* x
    y = sin.(x)
    g = repeat(["a", "b"], inner=4)
    (; x, x2, y, g)
end

# Auto-fit margin on this grid: L = 1.5 * 1.0, margin = (c-1)/c * L = 0.5.
const BOUNDARY_MARGIN = 0.5

bare_model(df) = @brm df begin
    loc ~ 1 + hsgp(x; k=4)
    y ~ Normal(loc, 1)
end

explicit_prior_model(df) = @brm df begin
    loc ~ 1 + hsgp(x; k=4)
    length_scale(:, hsgp(x)) ~ LogNormal(0, 1)
    y ~ Normal(loc, 1)
end

hyper_model(df) = @brm df begin
    loc ~ 1 + hsgp(x; k=4, by=g)
    log(length_scale(hsgp(x))) ~ 1 + (1 | g)
    log(sd(hsgp(x))) ~ 1 + (1 | g)
    y ~ Normal(loc, 1)
end

hyper_prior_model(df) = @brm df begin
    loc ~ 1 + hsgp(x; k=4, by=g)
    log(length_scale(hsgp(x))) ~ 1 + (1 | g)
    log(sd(hsgp(x))) ~ 1 + (1 | g)
end

@testset "boundary check reports ratios on a bare term" begin
    df = boundary_df()
    sb = SBBRMI(bare_model(df); mod=@__MODULE__)
    d = brm_descriptor(sb)
    names = ["hsgp_x_rho_iso"]
    # Healthy boundary (0.3/0.5 = 0.6), fully sub-floor: the two verdicts
    # are independent by design.
    healthy = hsgp_boundary_check(d, fill(0.3, (6, 1)), names;
                                  predictor=:loc, term=:hsgp_x)
    @test healthy.n_groups == 1
    @test healthy.margin ≈ BOUNDARY_MARGIN
    @test healthy.mean_rho ≈ [0.3]
    @test healthy.ratios ≈ [0.6]
    @test healthy.flagged == false
    @test healthy.floor_binding ≈ [1.0]
    # Suspect boundary (0.8/0.5 = 1.6).
    suspect = hsgp_boundary_check(d, fill(0.8, (6, 1)), names;
                                  predictor=:loc, term=:hsgp_x)
    @test suspect.ratios ≈ [1.6]
    @test suspect.flagged == true
    @test suspect.floor_binding ≈ [1.0]
    # Above the floor the binding clears while the flag stays.
    @test hsgp_boundary_check(d, fill(2.0, (6, 1)), names;
                              predictor=:loc, term=:hsgp_x).floor_binding ≈ [0.0]
end

@testset "boundary check resolves per-group hyper values" begin
    df = boundary_df()
    sb = SBBRMI(hyper_model(df); mod=@__MODULE__)
    d = brm_descriptor(sb)
    names = ["hsgp_x_by_g_beta0_rho", "hsgp_x_by_g_sd_rho",
             "hsgp_x_by_g_z_rho.1", "hsgp_x_by_g_z_rho.2"]
    # beta0 = log(0.4), sd = 0.1, z = [0.5, -0.5]: both groups healthy.
    draws = repeat([log(0.4) 0.1 0.5 -0.5], 7, 1)
    healthy = hsgp_boundary_check(d, draws, names;
                                  predictor=:loc, term=:hsgp_x_by_g)
    @test healthy.n_groups == 2
    @test healthy.margin ≈ BOUNDARY_MARGIN
    expected = exp.(log(0.4) .+ 0.1 .* [0.5, -0.5])
    @test healthy.mean_rho ≈ expected
    @test healthy.ratios ≈ expected ./ BOUNDARY_MARGIN
    @test healthy.flagged == false
    @test healthy.floor_binding ≈ [1.0, 1.0]
    # beta0 = log(1.2), sd = 0.5, z = [0.5, -1.5]: group 1 suspect and
    # above the floor, group 2 healthy and bound — mixed verdicts.
    draws2 = repeat([log(1.2) 0.5 0.5 -1.5], 7, 1)
    mixed = hsgp_boundary_check(d, draws2, names;
                                predictor=:loc, term=:hsgp_x_by_g)
    expected2 = exp.(log(1.2) .+ 0.5 .* [0.5, -1.5])
    @test mixed.mean_rho ≈ expected2
    @test mixed.ratios ≈ expected2 ./ BOUNDARY_MARGIN
    @test mixed.flagged == true
    @test mixed.floor_binding ≈ [0.0, 1.0]
end

@testset "boundary check measures explicit-prior sub-floor mass" begin
    df = boundary_df()
    sb = SBBRMI(explicit_prior_model(df); mod=@__MODULE__)
    d = brm_descriptor(sb)
    names = ["hsgp_x_rho_iso"]
    # The explicit prior drops the declaration floor, but the validity
    # floor data still exists — the binding stays measurable.
    @test hsgp_boundary_check(d, fill(0.5, (6, 1)), names;
                              predictor=:loc, term=:hsgp_x).floor_binding ≈ [1.0]
    @test hsgp_boundary_check(d, fill(2.0, (6, 1)), names;
                              predictor=:loc, term=:hsgp_x).floor_binding ≈ [0.0]
end

@testset "boundary check survives the prior regime" begin
    df = boundary_df()
    sb = SBBRMI(hyper_prior_model(df); mod=@__MODULE__)
    d = brm_descriptor(sb)
    names = ["hsgp_x_by_g_beta0_rho", "hsgp_x_by_g_sd_rho",
             "hsgp_x_by_g_z_rho.1", "hsgp_x_by_g_z_rho.2"]
    draws = repeat([log(0.4) 0.1 0.5 -0.5], 7, 1)
    prior = hsgp_boundary_check(d, draws, names;
                                predictor=:loc, term=:hsgp_x_by_g)
    @test prior.n_groups == 2
    @test prior.flagged == false
    @test prior.floor_binding ≈ [1.0, 1.0]
end

@testset "boundary check refusals stay loud" begin
    df = boundary_df()
    sb = SBBRMI(bare_model(df); mod=@__MODULE__)
    d = brm_descriptor(sb)
    names = ["hsgp_x_rho_iso"]
    draws = fill(0.3, (6, 1))
    @test_throws "occurs 0 times" hsgp_boundary_check(
        d, draws, names; predictor=:loc, term=:hsgp_missing)
    periodic = SBBRMI((@brm df begin
        loc ~ 1 + hsgp(x; k=4, cov=:periodic, period=2.0)
        y ~ Normal(loc, 1)
    end); mod=@__MODULE__)
    @test_throws "non-`exp_quad`" hsgp_boundary_check(
        brm_descriptor(periodic), draws, names; predictor=:loc, term=:hsgp_x)
    multiaxis = SBBRMI((@brm df begin
        loc ~ 1 + hsgp(x, x2; k=(3, 3))
        y ~ Normal(loc, 1)
    end); mod=@__MODULE__)
    @test_throws "one-dimensional" hsgp_boundary_check(
        brm_descriptor(multiaxis), draws, names; predictor=:loc, term=:hsgp_x_x2)
    aniso = SBBRMI((@brm df begin
        loc ~ 1 + hsgp(x; k=4, iso=false)
        y ~ Normal(loc, 1)
    end); mod=@__MODULE__)
    @test_throws "anisotropic" hsgp_boundary_check(
        brm_descriptor(aniso), draws, names; predictor=:loc, term=:hsgp_x)
    fixed_domain = SBBRMI((@brm df begin
        loc ~ 1 + hsgp(x; k=4, domain=(-2.0, 2.0))
        y ~ Normal(loc, 1)
    end); mod=@__MODULE__)
    @test_throws "explicit `domain=`" hsgp_boundary_check(
        brm_descriptor(fixed_domain), draws, names; predictor=:loc, term=:hsgp_x)
    derived = SBBRMI((@brm df begin
        log(xlat) ~ 1
        loc ~ 1 + hsgp(xlat; k=3, domain=(0.1, 5.0))
        y ~ Normal(loc, 1)
    end); mod=@__MODULE__)
    @test_throws "model-derived axis" hsgp_boundary_check(
        brm_descriptor(derived), draws, names; predictor=:loc, term=:hsgp_xlat)
end
