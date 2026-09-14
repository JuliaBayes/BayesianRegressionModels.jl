using Test
using BayesianRegressionModels
using LinearAlgebra
using Distributions: Exponential, Normal

builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x + t | subject)
    y ~ Normal(mu, sigma)
end

intercept_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject)
    y ~ Normal(mu, sigma)
end

slope_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | subject)
    y ~ Normal(mu, sigma)
end

single_bucket_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | p | subject)
    y ~ Normal(mu, sigma)
end

df = (; subject = repeat([11, 12], inner=3),
      x = collect(range(-1.0, 1.0, length=6)),
      t = repeat([0.0, 1.0, 2.0], 2),
      y = zeros(6))

const UNC_NONCENTERED = [
    "sigma",
    "pop_mu_beta_pop.1",
    "pop_mu_beta_pop.2",
    "r_mu_subject_L.1",
    "r_mu_subject_L.2",
    "r_mu_subject_L.3",
    "r_mu_subject_tau.1",
    "r_mu_subject_tau.2",
    "r_mu_subject_tau.3",
    "r_mu_subject_z_flat.1",
    "r_mu_subject_z_flat.2",
    "r_mu_subject_z_flat.3",
    "r_mu_subject_z_flat.4",
    "r_mu_subject_z_flat.5",
    "r_mu_subject_z_flat.6",
]

@testset "adaptive centering block metadata" begin
    sb = SBBRMI(builder(df); mod=@__MODULE__)
    block = only(adaptive_centering_blocks(sb, UNC_NONCENTERED))
    @test block.ranef.binding === :r_mu_subject
    @test block.target_c == 0.0
    @test block.effects == reshape(10:15, 3, 2)
    @test block.cholesky_free == [4, 5, 6]
    @test block.log_scales == [7, 8, 9]

    centered = SBBRMI(builder(df); mod=@__MODULE__, centered_groups=[:subject])
    unc_centered = vcat(
        UNC_NONCENTERED[1:9],
        ["r_mu_subject_b.$g.$k" for g in 1:2 for k in 1:3],
    )
    centered_block = only(adaptive_centering_blocks(centered, unc_centered))
    @test centered_block.target_c == 1.0
    @test centered_block.effects == reshape(10:15, 3, 2)

    missing = filter(!=("r_mu_subject_L.2"), UNC_NONCENTERED)
    err = try
        adaptive_centering_blocks(sb, missing)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("r_mu_subject_L.2", err.msg)
end

@testset "adaptive centering includes scalar random intercepts" begin
    names = [
        "r_mu_subject_log_scale",
        "r_mu_subject_xi.1",
        "r_mu_subject_xi.2",
    ]
    sb = SBBRMI(intercept_builder(df); mod=@__MODULE__)
    block = only(adaptive_centering_blocks(sb, names))
    @test block.ranef.family === :ranef_intercept
    @test block.ranef.n_terms == 1
    @test block.target_c == 0.0
    @test block.effects == reshape(2:3, 1, 2)
    @test isempty(block.cholesky_free)
    @test block.log_scales == [1]

    centered = SBBRMI(
        intercept_builder(df); mod=@__MODULE__, centered_groups=[:subject],
    )
    centered_block = only(adaptive_centering_blocks(centered, names))
    @test centered_block.ranef.family === :ranef_intercept_centered
    @test centered_block.target_c == 1.0
    @test centered_block.effects == block.effects
    @test isempty(centered_block.cholesky_free)
    @test centered_block.log_scales == [1]

    C = BayesianRegressionModels._adaptive_block_cholesky(
        [log(2.5), -0.3, 0.7], block,
    )
    @test C ≈ reshape([2.5], 1, 1) atol=1e-15

    err = try
        adaptive_centering_blocks(sb, names[2:3])
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("r_mu_subject_log_scale", err.msg)
end

@testset "random-effect pilot selection uses the scalar criterion" begin
    z = reshape([0.2 -0.4; 1.1 0.6; -0.7 0.9], 3, 2)
    logs = reshape([log(0.2) log(4.0); log(0.2) log(4.0); log(0.2) log(4.0)], 3, 2)
    selected = select_ranef_centeredness(z, logs; candidates=0:0.1:1)
    @test selected.centeredness == [0.7, 0.0]
    @test all(selected.admissible[:, 1])
    @test all(selected.admissible[:, 2])
end

@testset "adaptive centering includes one-term correlated buckets" begin
    names = [
        "b_p_subject_tau.1",
        "b_p_subject_z_flat.1",
        "b_p_subject_z_flat.2",
    ]
    sb = SBBRMI(single_bucket_builder(df); mod=@__MODULE__)
    block = only(adaptive_centering_blocks(sb, names))
    @test block.ranef.family === :ranef_correlated_draws
    @test block.ranef.n_terms == 1
    @test block.target_c == 0.0
    @test block.effects == reshape(2:3, 1, 2)
    @test isempty(block.cholesky_free)
    @test block.log_scales == [1]

    centered_names = [
        "b_p_subject_tau.1",
        "b_p_subject_b.1.1",
        "b_p_subject_b.2.1",
    ]
    centered = SBBRMI(
        single_bucket_builder(df); mod=@__MODULE__, centered_groups=[:subject],
    )
    centered_block = only(adaptive_centering_blocks(centered, centered_names))
    @test centered_block.ranef.family === :ranef_correlated_draws_centered
    @test centered_block.target_c == 1.0
    @test centered_block.effects == reshape(2:3, 1, 2)
    @test isempty(centered_block.cholesky_free)
    @test centered_block.log_scales == [1]
end

@testset "Stan cholesky_factor_corr reconstruction" begin
    raw = [0.2, -0.4, 0.7]
    L = BayesianRegressionModels._adaptive_cholesky_corr(raw, 3)
    z21, z31, z32 = tanh.(raw)
    expected = [
        1.0 0.0 0.0
        z21 sqrt(1-z21^2) 0.0
        z31 sqrt(1-z31^2)*z32 sqrt(1-z31^2)*sqrt(1-z32^2)
    ]
    @test L ≈ expected atol=1e-15
    @test L ≈ [
        1.0 0.0 0.0
        0.197375320224904 0.9803279976447253 0.0
        -0.3799489622552249 0.5590446975250928 0.7369584874674117
    ] atol=1e-15
    @test all(diag(L) .> 0)
    @test all(sum(abs2, @view(L[i, 1:i])) ≈ 1 for i in 1:3)
end

@testset "stratified correlated blocks refuse the single-frame contract" begin
    by_df = merge(df, (; stratum=repeat([1, 2], inner=3)))
    by_builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | gr(subject, by=stratum))
        y ~ Normal(mu, sigma)
    end
    by_sb = SBBRMI(by_builder(by_df); mod=@__MODULE__)
    err = try
        adaptive_centering_blocks(by_sb, String[])
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("stratified", lowercase(err.msg))
    @test occursin("r_mu_subject__by__stratum", err.msg)
end
