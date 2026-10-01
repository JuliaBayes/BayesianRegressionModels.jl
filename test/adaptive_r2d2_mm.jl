# Adaptive-centering coverage for R2D2 and multi-membership blocks
# (snag adaptive-centeri-953d87e0, derived-tau follow-up on decision
# 2026-10-01T11-33-23-658-13ne7ae).
#
# R2D2 blocks derive their marginal scales instead of sampling them, so there
# is no unconstrained `tau`/`log_scale` coordinate to read. They resolve to
# `R2D2AdaptiveCenteringBlock`s whose per-term scale is compiled from the
# emitted assignment into a log-linear normal form over the model's own R2,
# Dirichlet-share and reference/total-scale coordinates. This file checks that
# metadata against hand-spelled unconstrained names (no BridgeStan); the
# compiled-model exactness checks live in `adaptive_centering_bridgestan.jl`.
#
# Multi-membership intercepts sample their scale (`log_scale`) exactly like an
# ordinary `(1 | g)` — only the downstream gather differs, and it is linear —
# so they adapt through the ordinary scalar path. An `mm` block with any slope
# term — multi-term, or a slope-only `(0 + x | mm(...))` — already shares the
# `:ranef_correlated_draws` emission and adapts; that is locked in here rather
# than left as an undocumented accident.
#
# Run: julia --project=test test/adaptive_r2d2_mm.jl

using Test
using BayesianRegressionModels
using Distributions: Beta, Exponential, LKJCholesky, Normal

const BRM = BayesianRegressionModels

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

r2d2_icc_builder = @brm begin
    sigma ~ Exponential(1)
    log_Vc ~ 1 + (1 | p | subject)
    log_k10 ~ 1 + (1 | p | subject)
    sd(log_Vc, p) ~ r2d2(reference_scale=sigma)
    cor(:, p) ~ LKJCholesky(2, 2)
    y ~ Normal(log_Vc + log_k10, sigma)
end

r2d2_flat_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject)
    effect(mu, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5, alpha=1)
    y ~ Normal(mu, sigma)
end

r2d2_unbounded_reference_builder = @brm begin
    sigma ~ Exponential(1)
    s0 ~ Normal(0, 1)
    log_Vc ~ 1 + (1 | p | subject)
    log_k10 ~ 1 + (1 | p | subject)
    sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=2, concentration=1,
                    reference_scale=s0)
    cor(:, p) ~ LKJCholesky(2, 2)
    y ~ Normal(log_Vc + log_k10, sigma)
end

r2d2_df = (;
    subject=[1, 1, 2, 2, 3, 3],
    site=["a", "b", "a", "b", "a", "b"],
    x=collect(range(-1.0, 1.0, length=6)),
    y=zeros(6),
)

# The compiled bucket's unconstrained names (BridgeStan order; only the set
# matters to resolution, the order fixes the expected indices below).
bucket_names(extra...) = vcat(
    ["sigma", "b_p_subject_r2d2_1_R2", "b_p_subject_r2d2_1_phi.1",
     "b_p_subject_L.1"],
    ["b_p_subject_z_flat.$i" for i in 1:6],
    ["pop_log_Vc_beta_pop.1", "pop_log_k10_beta_pop.1"],
    collect(extra),
)

simplex_atoms(s) = [(a.weight, a.coordinates, a.entry) for a in s.simplex]
logistic(y) = inv(1 + exp(-y))

@testset "R2D2 bucket resolves its derived scales" begin
    sb = SBBRMI(r2d2_bucket_builder(r2d2_df); total_groups=(), mod=@__MODULE__)
    rblock = only(ranef_blocks(sb))
    @test rblock.family === :ranef_correlated_draws_r2d2
    block = only(adaptive_centering_blocks(sb, bucket_names()))
    @test block isa R2D2AdaptiveCenteringBlock
    @test block.ranef.binding === rblock.binding
    @test block.target_c == 0.0
    @test block.effects == reshape(5:10, 2, 3)
    @test block.cholesky_free == [4]
    # tau[j] = sigma * sqrt(phi[j] * R2 / (1 - R2))
    for (j, s) in enumerate(block.scales)
        @test s.constant == 0.0
        @test s.linear == [(1.0, 1)]
        @test s.unit == [(0.5, -0.5, 2)]
        @test simplex_atoms(s) == [(0.5, [3], j)]
    end
    # Two-entry Stan simplex: phi = softmax([w, -w]), w = y / sqrt(2).
    x = [0.3, -0.4, 0.7, 0.1, zeros(8)...]
    phi1 = logistic(sqrt(2) * x[3])
    expected = exp(x[1]) .* sqrt.([phi1, 1 - phi1] .* logistic(x[2]) ./
                                  (1 - logistic(x[2])))
    @test BRM._adaptive_block_taus(x, block) ≈ expected rtol=1e-14
    C = BRM._adaptive_block_cholesky(x, block)
    @test [C[1, 1], hypot(C[2, 1], C[2, 2])] ≈ expected rtol=1e-14
end

@testset "R2D2 bucket and an ordinary block resolve together" begin
    sb = SBBRMI(r2d2_joint_builder(r2d2_df); total_groups=(), mod=@__MODULE__)
    names = bucket_names("r_log_Vc_site_log_scale", "r_log_Vc_site_xi.1",
                         "r_log_Vc_site_xi.2")
    blocks = adaptive_centering_blocks(sb, names)
    @test length(blocks) == 2
    @test count(b -> b isa R2D2AdaptiveCenteringBlock, blocks) == 1
    ordinary = only(b for b in blocks if b isa AdaptiveCenteringBlock)
    @test ordinary.ranef.group === :site
    @test ordinary.log_scales == [13]
    @test ordinary.effects == reshape(14:15, 1, 2)
end

@testset "R2D2 partial ICC leaves the unaddressed margin a free scale" begin
    sb = SBBRMI(r2d2_icc_builder(r2d2_df); total_groups=(), mod=@__MODULE__)
    names = vcat(
        ["sigma", "b_p_subject_r2d2_1_R2", "b_p_subject_r2d2_free_tau_2",
         "b_p_subject_L.1"],
        ["b_p_subject_z_flat.$i" for i in 1:6],
        ["pop_log_Vc_beta_pop.1", "pop_log_k10_beta_pop.1"],
    )
    block = only(adaptive_centering_blocks(sb, names))
    @test block isa R2D2AdaptiveCenteringBlock
    # One-margin ICC: a `simplex[1]` share is identically 1 (no coordinates).
    @test simplex_atoms(block.scales[1]) == [(0.5, Int[], 1)]
    @test block.scales[1].linear == [(1.0, 1)]
    @test block.scales[1].unit == [(0.5, -0.5, 2)]
    @test block.scales[2].linear == [(1.0, 3)]
    @test isempty(block.scales[2].unit) && isempty(block.scales[2].simplex)
    x = [0.2, 0.5, -0.3, zeros(9)...]
    @test BRM._adaptive_block_taus(x, block) ≈
          [exp(0.2) * sqrt(logistic(0.5) / (1 - logistic(0.5))), exp(-0.3)] rtol=1e-14
end

@testset "R2D2 flat-form intercept resolves sqrt((1 - R2) * tau_bsv^2)" begin
    sb = SBBRMI(r2d2_flat_builder(r2d2_df); total_groups=(), mod=@__MODULE__)
    rblock = only(ranef_blocks(sb))
    @test rblock.family === :ranef_intercept_r2d2
    names = ["r2d2_mu_R2", "sigma", "pop_mu_beta_pop.1", "pop_mu_beta_pop.2",
             "r_mu_subject_xi.1", "r_mu_subject_xi.2", "r_mu_subject_xi.3"]
    block = only(adaptive_centering_blocks(sb, names))
    @test block isa R2D2AdaptiveCenteringBlock
    @test block.effects == reshape(5:7, 1, 3)
    @test isempty(block.cholesky_free)
    s = only(block.scales)
    @test s.constant ≈ log(0.5)
    @test isempty(s.linear) && isempty(s.simplex)
    @test s.unit == [(0.0, 0.5, 1)]
    x = [0.4, zeros(6)...]
    @test only(BRM._adaptive_block_taus(x, block)) ≈
          sqrt((1 - logistic(0.4)) * 0.5^2) rtol=1e-14
end

@testset "R2D2 with an unbounded reference scale refuses naming it" begin
    sb = SBBRMI(r2d2_unbounded_reference_builder(r2d2_df); total_groups=(),
                mod=@__MODULE__)
    err = try
        adaptive_centering_blocks(sb, String[])
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("b_p_subject", err.msg)
    @test occursin("s0", err.msg)
    @test occursin("bounds", err.msg)
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

mm_slope_only_builder = @brm begin
    sigma ~ Exponential(1)
    loc ~ 1 + (0 + x | mm(g1, g2; weights=(w1, w2)))
    y ~ Normal(loc, sigma)
end

@testset "intercept-only mm blocks adapt like ordinary intercepts" begin
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

@testset "$label mm blocks adapt like ordinary correlated blocks" for (label, mm_builder, expected_K) in (
    ("multi-term", mm_slope_builder, 2),
    ("slope-only", mm_slope_only_builder, 1),
)
    sb = SBBRMI(mm_builder(mm_df); total_groups=(), mod=@__MODULE__)
    rblock = only(ranef_blocks(sb))
    @test rblock.family === :ranef_correlated_draws
    K, G = rblock.n_terms, rblock.n_groups
    @test (K, G) == (expected_K, 3)
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
