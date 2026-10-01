# Adaptive-centering coverage for R2D2 and multi-membership blocks
# (snag adaptive-centeri-953d87e0).
#
# R2D2 blocks derive their marginal scales (`tau[j] = reference_scale[j] *
# sqrt(phi[j] * R2 / (1 - R2))`) instead of sampling them, so the online
# wrapper's unconstrained `tau`/`log_scale` coordinates do not exist for them.
# Leaving such a block out of `adaptive_centering_blocks` silently is a wrong
# answer: a joint model would adapt every other cell and leave the R2D2 block
# at its compiled endpoint with nothing saying so. These blocks refuse loudly,
# naming the block — the same contract as stratified `gr(g, by=b)`.
#
# Multi-membership intercepts sample their scale (`log_scale`) exactly like an
# ordinary `(1 | g)` — only the downstream gather differs, and it is linear —
# so they adapt through the ordinary scalar path. Multi-term `mm` already
# shares the `:ranef_correlated_draws` emission and adapts; that is locked in
# here rather than left as an undocumented accident.
#
# Run: julia --project=test test/adaptive_r2d2_mm.jl

using Test
using BayesianRegressionModels
using Distributions: Beta, Exponential, LKJCholesky, Normal

# --- R2D2 fixtures ------------------------------------------------------

r2d2_bucket_builder = @brm begin
    sigma ~ Exponential(1)
    log_Vc ~ 1 + (1 | p | subject)
    log_k10 ~ 1 + (1 | p | subject)
    sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=2, concentration=1,
                    reference_scale=sigma)
    cor(:, p) ~ LKJCholesky(2, 2)
    y ~ Normal(log_Vc + log_k10, sigma)
end

r2d2_joint_builder = @brm begin
    sigma ~ Exponential(1)
    log_Vc ~ 1 + (1 | p | subject) + (1 | site)
    log_k10 ~ 1 + (1 | p | subject)
    sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=2, concentration=1,
                    reference_scale=sigma)
    cor(:, p) ~ LKJCholesky(2, 2)
    y ~ Normal(log_Vc + log_k10, sigma)
end

r2d2_flat_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject)
    effect(mu, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5, alpha=1)
    y ~ Normal(mu, sigma)
end

r2d2_df = (;
    subject=[1, 1, 2, 2, 3, 3],
    site=["a", "b", "a", "b", "a", "b"],
    x=collect(range(-1.0, 1.0, length=6)),
    y=zeros(6),
)

function adaptive_refusal(sb)
    err = try
        adaptive_centering_blocks(sb, String[])
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    err
end

@testset "R2D2 correlated bucket refuses adaptive centering loudly" begin
    sb = SBBRMI(r2d2_bucket_builder(r2d2_df); total_groups=(), mod=@__MODULE__)
    block = only(ranef_blocks(sb))
    @test block.family === :ranef_correlated_draws_r2d2
    err = adaptive_refusal(sb)
    @test occursin("r2d2", lowercase(err.msg))
    @test occursin(string(block.binding), err.msg)
end

@testset "R2D2 joint model names the R2D2 block, not the ordinary one" begin
    sb = SBBRMI(r2d2_joint_builder(r2d2_df); total_groups=(), mod=@__MODULE__)
    blocks = ranef_blocks(sb)
    @test length(blocks) == 2
    r2d2_block = only(b for b in blocks if occursin("r2d2", string(b.family)))
    @test r2d2_block.family === :ranef_correlated_draws_r2d2
    err = adaptive_refusal(sb)
    @test occursin("r2d2", lowercase(err.msg))
    @test occursin(string(r2d2_block.binding), err.msg)
end

@testset "R2D2 flat-form intercept refuses adaptive centering loudly" begin
    sb = SBBRMI(r2d2_flat_builder(r2d2_df); total_groups=(), mod=@__MODULE__)
    block = only(ranef_blocks(sb))
    @test block.family === :ranef_intercept_r2d2
    err = adaptive_refusal(sb)
    @test occursin("r2d2", lowercase(err.msg))
    @test occursin(string(block.binding), err.msg)
end

# --- multi-membership fixtures -------------------------------------------

mm_df = (;
    g1=["a", "a", "b"],
    g2=["b", "c", "c"],
    w1=[2.0, 1.0, 0.0],
    w2=[1.0, 1.0, 3.0],
    x=[0.2, -0.1, 0.4],
    y=[0.1, 0.2, 0.3],
)

mm_intercept_builder = @brm begin
    sigma ~ Exponential(1)
    loc ~ 1 + (1 | mm(g1, g2; weights=(w1, w2)))
    y ~ Normal(loc, sigma)
end

mm_slope_builder = @brm begin
    sigma ~ Exponential(1)
    loc ~ 1 + (1 + x | mm(g1, g2; weights=(w1, w2)))
    y ~ Normal(loc, sigma)
end

@testset "one-term mm blocks adapt like ordinary intercepts" begin
    sb = SBBRMI(mm_intercept_builder(mm_df); total_groups=(), mod=@__MODULE__)
    rblock = only(ranef_blocks(sb))
    @test rblock.family === :ranef_intercept_draws
    @test (rblock.n_terms, rblock.n_groups) == (1, 3)
    unc = vcat(
        ["$(rblock.binding)_log_scale"],
        ["$(rblock.z).$g" for g in 1:rblock.n_groups],
    )
    block = only(adaptive_centering_blocks(sb, unc))
    @test block.ranef.binding === rblock.binding
    @test block.target_c == 0.0
    @test block.effects == reshape(2:4, 1, 3)
    @test block.log_scales == [1]
    @test isempty(block.cholesky_free)
end

@testset "multi-term mm blocks adapt like ordinary correlated blocks" begin
    sb = SBBRMI(mm_slope_builder(mm_df); total_groups=(), mod=@__MODULE__)
    rblock = only(ranef_blocks(sb))
    @test rblock.family === :ranef_correlated_draws
    K, G = rblock.n_terms, rblock.n_groups
    @test (K, G) == (2, 3)
    n_cholesky = K * (K - 1) ÷ 2
    unc = vcat(
        ["$(rblock.binding)_L.$i" for i in 1:n_cholesky],
        ["$(rblock.binding)_tau.$i" for i in 1:K],
        ["$(rblock.z).$i" for i in 1:(K * G)],
    )
    block = only(adaptive_centering_blocks(sb, unc))
    @test block.ranef.binding === rblock.binding
    @test block.target_c == 0.0
    @test block.cholesky_free == collect(1:n_cholesky)
    @test block.log_scales == collect(n_cholesky .+ (1:K))
    @test block.effects == reshape(n_cholesky + K .+ (1:(K * G)), K, G)
end
