using Test
using BayesianRegressionModels
using StanBlocks
using Distributions
using LinearAlgebra
using LogDensityProblems

const T2_TERM_N = 30
const T2_RUN_BRIDGESTAN = get(ENV, "BRM_T2_RUNTIME", "1") != "0"
const T2_TERM_CACHE = joinpath(tempdir(), "brm-t2")

function t2_term_df(; x_shift=0.0, x_scale=1.0, z_shift=0.0, z_scale=1.0)
    base = collect(range(-2.0, 2.0; length=T2_TERM_N))
    x = x_shift .+ x_scale .* base
    z0 = sin.(base) .+ 0.05 .* collect(1:T2_TERM_N)
    z = z_shift .+ z_scale .* z0
    y = @. 0.3 * cos(base) + 0.1 * sin(z0)
    (; x, z, y)
end

t2_default_model(df) = @brm df begin
    loc ~ 1 + t2(x, z)
    y ~ Normal(loc, 1)
end

t2_custom_model(df) = @brm df begin
    loc ~ 1 + t2(x, z; k=(4, 6), basis=(:cr, :cr), full=false)
    y ~ Normal(loc, 1)
end

t2_distributional_model(df) = @brm df begin
    loc ~ 1 + t2(x, z)
    log(err) ~ 1 + t2(x, z; k=(4, 4))
    y ~ Normal(loc, err)
end

t2_one_margin(df) = @brm df begin
    loc ~ t2(x)
    y ~ Normal(loc, 1)
end

t2_three_margins(df) = @brm df begin
    loc ~ t2(x, z, y)
    y ~ Normal(loc, 1)
end

t2_bad_k_vector(df) = @brm df begin
    loc ~ t2(x, z; k=[5, 5])
    y ~ Normal(loc, 1)
end

t2_bad_k_value(df) = @brm df begin
    loc ~ t2(x, z; k=(2, 5))
    y ~ Normal(loc, 1)
end

t2_bad_basis(df) = @brm df begin
    loc ~ t2(x, z; basis=(:tp, :cr))
    y ~ Normal(loc, 1)
end

t2_bad_full(df) = @brm df begin
    loc ~ t2(x, z; full=true)
    y ~ Normal(loc, 1)
end

t2_unknown_kw(df) = @brm df begin
    loc ~ t2(x, z; bs=(:cr, :cr))
    y ~ Normal(loc, 1)
end

@testset "tensor-product smooth t2" begin
    df = t2_term_df()

    @testset "public surface and eager keyword validation" begin
        default = t2_default_model(df)
        @test popcoefnames(default, :loc) == [:Intercept]
        @test_throws "exactly 2 positional margins" SBBRMI(t2_one_margin(df); mod=@__MODULE__)
        three_margin_error = try
            SBBRMI(t2_three_margins(df); mod=@__MODULE__)
            nothing
        catch error
            error
        end
        @test !isnothing(three_margin_error)
        @test any(message -> occursin(message, sprint(showerror, three_margin_error)),
                  ("exactly 2 positional margins", "cyclic model declarations"))
        @test_throws "2-tuple" t2_bad_k_vector(df)
        @test_throws "greater than 2" t2_bad_k_value(df)
        @test_throws "only cubic-regression-spline" t2_bad_basis(df)
        @test_throws "full=true" t2_bad_full(df)
        @test_throws "unsupported keyword" t2_unknown_kw(df)
    end

    @testset "cubic marginal and tensor geometry" begin
        margin = BayesianRegressionModels._sb_fit_cr_spline(df.x; k=5)
        cardinal = BayesianRegressionModels._sb_cr_basis(
            margin.knots, margin.F, margin.knots)
        @test cardinal ≈ Matrix{Float64}(I, 5, 5) atol=1e-12

        _, penalty = BayesianRegressionModels._sb_cr_second_derivative_map(margin.knots)
        @test transpose(margin.range_projection) * penalty * margin.range_projection ≈
              Matrix{Float64}(I, 3, 3) atol=1e-10
        Xnull, Zpen = BayesianRegressionModels._sb_apply_cr_spline(margin, df.x)
        @test rank(hcat(Xnull, Zpen)) == 5

        fit = BayesianRegressionModels._sb_fit_t2(df.x, df.z)
        blocks = BayesianRegressionModels._sb_apply_t2(fit, df.x, df.z)
        @test size.(blocks) == ((T2_TERM_N, 3), (T2_TERM_N, 9),
                               (T2_TERM_N, 6), (T2_TERM_N, 6))
        @test maximum(abs, vec(sum(blocks[1]; dims=1))) < 1e-10

        affine = t2_term_df(; x_shift=11.0, x_scale=7.0,
                            z_shift=-3.0, z_scale=2.5)
        affine_fit = BayesianRegressionModels._sb_fit_t2(affine.x, affine.z)
        affine_blocks = BayesianRegressionModels._sb_apply_t2(
            affine_fit, affine.x, affine.z)
        @test all(isapprox.(blocks, affine_blocks; atol=1e-10, rtol=1e-10))
    end

    sb = SBBRMI(t2_default_model(df); mod=@__MODULE__)
    sb_custom = SBBRMI(t2_custom_model(df); mod=@__MODULE__)
    sb_distributional = SBBRMI(t2_distributional_model(df); mod=@__MODULE__)

    @testset "data blocks and descriptor" begin
        @test size(sb.data[:Xfixed_t2_loc_x_z]) == (T2_TERM_N, 3)
        @test size(sb.data[:Zrr_t2_loc_x_z]) == (T2_TERM_N, 9)
        @test size(sb.data[:Zrn_t2_loc_x_z]) == (T2_TERM_N, 6)
        @test size(sb.data[:Znr_t2_loc_x_z]) == (T2_TERM_N, 6)
        @test size(sb_custom.data[:Xfixed_t2_loc_x_z]) == (T2_TERM_N, 3)
        @test size(sb_custom.data[:Zrr_t2_loc_x_z]) == (T2_TERM_N, 8)
        @test size(sb_custom.data[:Zrn_t2_loc_x_z]) == (T2_TERM_N, 4)
        @test size(sb_custom.data[:Znr_t2_loc_x_z]) == (T2_TERM_N, 8)
        @test size(sb_distributional.data[:Zrr_t2_loc_x_z]) == (T2_TERM_N, 9)
        @test size(sb_distributional.data[:Zrr_t2_log_err_x_z]) == (T2_TERM_N, 4)

        input = only(i for i in brm_descriptor(sb).inputs
                     if i.transform === :tensor_spline)
        @test input.name === :Xfixed_t2_loc_x_z
        @test input.column === nothing
    end

    @testset "frozen and fresh replay" begin
        replay_df = t2_term_df(; x_shift=0.4, x_scale=1.2,
                               z_shift=-0.3, z_scale=0.8)
        frozen = reprocess(sb, replay_df)
        fresh = reprocess(sb, replay_df; freeze_constants=false)
        code = StanBlocks.stan_code(sb.model)
        @test StanBlocks.stan_code(frozen.model) == code
        @test StanBlocks.stan_code(fresh.model) == code
        @test frozen.data[:Xfixed_t2_loc_x_z] != sb.data[:Xfixed_t2_loc_x_z]
        @test fresh.data[:Xfixed_t2_loc_x_z] != frozen.data[:Xfixed_t2_loc_x_z]
        @test all(k -> size(frozen.data[k]) == size(sb.data[k]),
                  (:Xfixed_t2_loc_x_z, :Zrr_t2_loc_x_z,
                   :Zrn_t2_loc_x_z, :Znr_t2_loc_x_z))
        @test maximum(abs, vec(sum(fresh.data[:Xfixed_t2_loc_x_z]; dims=1))) < 1e-10
    end

    @testset "transpile and stanc" begin
        for candidate in (sb, sb_custom, sb_distributional)
            @test StanBlocks.stan.transpiles(candidate.model)
            code = StanBlocks.stan_code(candidate.model)
            # One `vector[3]` in (rr, rn, nr) order rather than three scalars,
            # so a per-block `sd(<lp|:>, t2(x, z), <block>)` statement can
            # configure any subset through its semantic prior expression.
            @test occursin("vector<lower=0.0>[3] t2_loc_x_z_sd_pen;", code)
            @test occursin("t2_loc_x_z_sd_pen ~ std_normal();", code)
            @test occursin("vector[3] t2_loc_x_z_b_fixed;", code)
            @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
        end
    end

    @testset "BridgeStan finite density and gradient" begin
        if T2_RUN_BRIDGESTAN
            isdir(T2_TERM_CACHE) || mkpath(T2_TERM_CACHE)
            code = StanBlocks.stan_code(sb.model)
            problem = StanBlocks.stan_instantiate(
                sb.model; path=joinpath(T2_TERM_CACHE, string(hash(code)) * ".stan"))
            dimension = LogDensityProblems.dimension(problem)
            q = [0.03 * ((i % 7) - 3) for i in 1:dimension]
            lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
            @test isfinite(lp)
            @test length(gradient) == dimension
            @test all(isfinite, gradient)
        else
            @info "Skipping BridgeStan t2 runtime gate (BRM_T2_RUNTIME=0)"
        end
    end
end
