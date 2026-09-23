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

hyper_prior_model(df) = @brm df begin
    loc ~ 1 + hsgp(x; k=4, by=g)
    log(length_scale(hsgp(x))) ~ 1 + (1 | g)
    log(sd(hsgp(x))) ~ 1 + (1 | g)
    y ~ Normal(loc, 1)
end

# Hand-built constrained axes, as in test/prior_only_coordinates.jl: the
# descriptor never parses these names, it only matches the carriers it owns.
hyper_grouped_names() = [
    "hsgp_x_by_g_beta0_rho",
    "hsgp_x_by_g_sd_rho",
    "hsgp_x_by_g_z_rho.1", "hsgp_x_by_g_z_rho.2",
    "hsgp_x_by_g_beta0_sigma",
    "hsgp_x_by_g_sd_sigma",
    "hsgp_x_by_g_z_sigma.1", "hsgp_x_by_g_z_sigma.2",
]

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
    # The validity floor binds per group inside the basis deffun.
    @test occursin("brm_hsgp_by_hyper_S", code)
    @test occursin("fmax", code)
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

@testset "hyper-predictor descriptor coordinates" begin
    df = hyper_df()
    # Grouped: all six hyper roles resolve with counts (1, 1, G=2).
    sb = SBBRMI(hyper_grouped_model(df); mod=@__MODULE__)
    d = brm_descriptor(sb)
    names = hyper_grouped_names()
    rho_beta = brm_term_coordinates(d, :loc, names;
        term=:hsgp_x_by_g, parameter=:length_scale_intercept)
    @test rho_beta.coordinates == [1]
    @test rho_beta.output.name === :hsgp_x_by_g_beta0_rho
    rho_sd = brm_term_coordinates(d, :loc, names;
        term=:hsgp_x_by_g, parameter=:length_scale_ranef_sd)
    @test rho_sd.coordinates == [2]
    @test rho_sd.output.name === :hsgp_x_by_g_sd_rho
    rho_z = brm_term_coordinates(d, :loc, names;
        term=:hsgp_x_by_g, parameter=:length_scale_ranef_z)
    @test rho_z.coordinates == [3, 4]
    @test rho_z.output.name === :hsgp_x_by_g_z_rho
    @test brm_term_coordinates(d, :loc, names;
        term=:hsgp_x_by_g, parameter=:sd_intercept).coordinates == [5]
    @test brm_term_coordinates(d, :loc, names;
        term=:hsgp_x_by_g, parameter=:sd_ranef_sd).coordinates == [6]
    @test brm_term_coordinates(d, :loc, names;
        term=:hsgp_x_by_g, parameter=:sd_ranef_z).coordinates == [7, 8]
    # The replaced hyper roles redirect loudly instead of drifting.
    @test_throws "no sampled carrier" brm_term_coordinates(
        d, :loc, names; term=:hsgp_x_by_g, parameter=:length_scale)
    @test_throws "no sampled carrier" brm_term_coordinates(
        d, :loc, names; term=:hsgp_x_by_g, parameter=:sd)
    # Mixed: the planned hyper redirects, the bare hyper still resolves.
    sbm = SBBRMI(hyper_mixed_model(df); mod=@__MODULE__)
    dm = brm_descriptor(sbm)
    namesm = [names[1:4]; "hsgp_x_by_g_sigma"]
    @test brm_term_coordinates(dm, :loc, namesm;
        term=:hsgp_x_by_g, parameter=:length_scale_intercept).coordinates == [1]
    @test_throws "no sampled carrier" brm_term_coordinates(
        dm, :loc, namesm; term=:hsgp_x_by_g, parameter=:length_scale)
    sd_bare = brm_term_coordinates(dm, :loc, namesm;
        term=:hsgp_x_by_g, parameter=:sd)
    @test sd_bare.coordinates == [5]
    @test sd_bare.output.name === :hsgp_x_by_g_sigma
    # Ungrouped: intercept-only plan; `:length_scale` redirects (its twin is
    # a transformed-parameter deterministic, never a draw carrier).
    sbu = SBBRMI(hyper_ungrouped_model(df); mod=@__MODULE__)
    du = brm_descriptor(sbu)
    namesu = ["hsgp_x_beta0_rho", "hsgp_x_sigma"]
    @test brm_term_coordinates(du, :loc, namesu;
        term=:hsgp_x, parameter=:length_scale_intercept).coordinates == [1]
    @test_throws "no sampled carrier" brm_term_coordinates(
        du, :loc, namesu; term=:hsgp_x, parameter=:length_scale)
    @test brm_term_coordinates(du, :loc, namesu;
        term=:hsgp_x, parameter=:sd).coordinates == [2]
end

@testset "hyper-predictor BridgeStan finite density and gradient" begin
    if HYPER_RUN_BRIDGESTAN
        isdir(HYPER_CACHE) || mkpath(HYPER_CACHE)
        df = hyper_df()
        for (label, sb) in (
                (:grouped, SBBRMI(hyper_grouped_model(df); mod=@__MODULE__)),
                (:ungrouped, SBBRMI(hyper_ungrouped_model(df); mod=@__MODULE__)),
                (:mixed, SBBRMI(hyper_mixed_model(df); mod=@__MODULE__)))
            code = BRM.stan_code(sb)
            problem = StanBlocks.stan_instantiate(
                sb.model;
                path=joinpath(HYPER_CACHE,
                              string(label) * "-" * string(hash(code)) * ".stan"))
            dimension = LogDensityProblems.dimension(problem)
            q = [0.05 * ((i % 5) - 2) for i in 1:dimension]
            lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
            @test isfinite(lp)
            @test length(gradient) == dimension
            @test all(isfinite, gradient)
        end
    else
        @info "Skipping BridgeStan hyper-predictor runtime gate (BRM_GP_RUNTIME=0)"
    end
end

@testset "hyper-predictor per-group values follow the hyper linear predictor" begin
    if HYPER_RUN_BRIDGESTAN
        isdir(HYPER_CACHE) || mkpath(HYPER_CACHE)
        df = hyper_df()
        sb = SBBRMI(hyper_grouped_model(df); mod=@__MODULE__)
        code = BRM.stan_code(sb)
        problem = StanBlocks.stan_instantiate(
            sb.model;
            path=joinpath(HYPER_CACHE, "values-" * string(hash(code)) * ".stan"))
        dimension = LogDensityProblems.dimension(problem)
        q = [0.05 * ((i % 5) - 2) for i in 1:dimension]
        names = StanBlocks.BridgeStan.param_names(
            problem.model; include_tp=true, include_gq=false)
        constrained = StanBlocks.BridgeStan.param_constrain(
            problem.model, q; include_tp=true, include_gq=false)
        d = brm_descriptor(sb)
        # The VALUE carriers carry no descriptor role by design (deterministic
        # transforms, not draws); the test reads them off the artifact while
        # every coefficient address goes through the public resolver.
        at(name) = only(findall(==(name), names))
        coef(role) = constrained[brm_term_coordinates(
            d, :loc, names; term=:hsgp_x_by_g, parameter=role).coordinates]
        beta0_rho = only(coef(:length_scale_intercept))
        sd_rho = only(coef(:length_scale_ranef_sd))
        z_rho = coef(:length_scale_ranef_z)
        rho_vec = [constrained[at("hsgp_x_by_g_rho_vec.1")],
                   constrained[at("hsgp_x_by_g_rho_vec.2")]]
        # `rho_vec` itself is unfloored: the validity floor binds downstream,
        # per group, inside `brm_hsgp_by_hyper_S` (the `fmax` the spelling
        # testset pins below), exactly as the bare term declares it upstream.
        @test rho_vec ≈ exp.(beta0_rho .+ sd_rho .* z_rho) atol=1e-10
        @test rho_vec[1] != rho_vec[2]
        beta0_sigma = only(coef(:sd_intercept))
        sd_sigma = only(coef(:sd_ranef_sd))
        z_sigma = coef(:sd_ranef_z)
        sigma_vec = [constrained[at("hsgp_x_by_g_sigma_vec.1")],
                     constrained[at("hsgp_x_by_g_sigma_vec.2")]]
        @test sigma_vec ≈ exp.(beta0_sigma .+ sd_sigma .* z_sigma) atol=1e-10
        @test sigma_vec[1] != sigma_vec[2]
    else
        @info "Skipping BridgeStan hyper-predictor values gate (BRM_GP_RUNTIME=0)"
    end
end

@testset "hyper-predictor recovery sizing rule" begin
    # Executable pin for the Hyper-predictors recovery guidance
    # (docs/src/formula-terms.md), on the snag hyper-predictor-d453ecf9 public
    # grid (log of [0.5, 1, 2, 4, 7, 14], truths rho 1.3/2.1): the default
    # basis violates the margin rule while reproducing the reported floor, a
    # small-k basis violates the floor rule, and the guided basis satisfies
    # both. Guards the floor/L computation against drift — the floor below is
    # the independently recomputed 0.6955 from that snag.
    grid = log.([0.5, 1.0, 2.0, 4.0, 7.0, 14.0])
    half_range = maximum(abs, grid .- sum(grid) / length(grid))
    rho_min, rho_max = 1.3, 2.5
    bad = BRM._brm_fit_hsgp(grid, 10, 1.5)
    @test bad[2] ≈ 2.5325 atol=1e-3
    @test BRM._brm_hsgp_rho_lower(bad, 10) ≈ 0.6955 atol=1e-4
    @test bad[2] - half_range < 2 * rho_max
    smallk = BRM._brm_fit_hsgp(grid, 5, 1.5)
    @test BRM._brm_hsgp_rho_lower(smallk, 5) ≈ 1.4125 atol=1e-3
    @test BRM._brm_hsgp_rho_lower(smallk, 5) > rho_min
    good = BRM._brm_fit_hsgp(grid, 24, 4.0)
    @test good[2] - half_range > 2 * rho_max
    @test BRM._brm_hsgp_rho_lower(good, 24) ≈ 0.7695 atol=1e-3
    @test BRM._brm_hsgp_rho_lower(good, 24) < rho_min
end

@testset "hyper-predictor prior-regime coordinates survive" begin
    # The prior regime is the same model with the response column omitted.
    df = hyper_df()
    sb = SBBRMI(hyper_prior_model((; df.x, df.g)); mod=@__MODULE__)
    @test transpiles(sb.model)
    d = brm_descriptor(sb)
    names = hyper_grouped_names()
    # Every carrier moved to generated quantities under the same constrained
    # name; the hyper roles resolve unchanged.
    @test brm_term_coordinates(d, :loc, names;
        term=:hsgp_x_by_g, parameter=:length_scale_intercept).coordinates == [1]
    @test brm_term_coordinates(d, :loc, names;
        term=:hsgp_x_by_g, parameter=:length_scale_ranef_z).coordinates == [3, 4]
    @test brm_term_coordinates(d, :loc, names;
        term=:hsgp_x_by_g, parameter=:sd_ranef_z).coordinates == [7, 8]
    @test_throws "no sampled carrier" brm_term_coordinates(
        d, :loc, names; term=:hsgp_x_by_g, parameter=:length_scale)
end
