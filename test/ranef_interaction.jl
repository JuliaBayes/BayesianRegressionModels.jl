# test/ranef_interaction.jl — `&` interactions in random-effects design matrices.
#
# Run: julia --project=test test/ranef_interaction.jl
#
# A random slope may be an interaction term: `(Srace + Sobj + Srace & Sobj |
# subject)` lowers the `&` through the SAME treatment-coded expander as the
# population path (cont×cont / cont×cat / cat×cat), so the Z column is the
# elementwise product (or dummy products) and shared-`|ID|` margins / `sd()`
# addresses use the exact emitted `int_…` symbols. Without the ranef intercept
# the term fell through to the protect-style materializer, which broadcasts
# `&` over raw vectors and dies with
# `MethodError: no method matching &(::Float64, ::Float64)`
# (snag `srace-sobj-srace-7bfe3a11`).
using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

const BRM = BayesianRegressionModels

const RI_N = 24
const RI_SRACE = Float64.([isodd(i) ? 1.0 : -1.0 for i in 1:RI_N])
const RI_SOBJ = Float64.([i % 4 < 2 ? 1.0 : -1.0 for i in 1:RI_N])
const RI_SUBJECT = Int.(mod.(1:RI_N, 6) .+ 1)
const RI_RATE = Float64.(collect(1:RI_N))
const RI_DF = (;
    rate=RI_RATE, Srace=RI_SRACE, Sobj=RI_SOBJ, subject=RI_SUBJECT)

@testset "ranef cont×cont interaction (the reported shape)" begin
    brmi = (@brm begin
        mu ~ 1 + Srace + Sobj + (Srace + Sobj + Srace & Sobj | subject)
        sigma ~ Exponential(1)
        rate ~ Normal(mu, sigma)
    end)(RI_DF)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    @test sb.data[:int_Srace_x_Sobj] ≈ RI_SRACE .* RI_SOBJ
    @test sb.preproc[:int_Srace_x_Sobj].kind === :interaction
    code = BRM.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    # The precomputed-column workaround this fix retires emits the same math:
    # identical Z values, identical Stan up to the column's own name.
    workaround = (@brm begin
        mu ~ 1 + Srace + Sobj + (Srace + Sobj + inter | subject)
        sigma ~ Exponential(1)
        rate ~ Normal(mu, sigma)
    end)((; RI_DF..., inter=RI_SRACE .* RI_SOBJ))
    sb_workaround = SBBRMI(workaround; mod=@__MODULE__, total_groups=())
    @test sb.data[:int_Srace_x_Sobj] ≈ sb_workaround.data[:inter]
    @test replace(code, "int_Srace_x_Sobj" => "inter") ==
          BRM.stan_code(sb_workaround)
end

@testset "ranef cont×cat and cat×cat interactions" begin
    df = (;
        x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        g=[1, 2, 3, 1, 2, 3],
        h=[1, 1, 2, 2, 1, 2],
        subj=[1, 1, 2, 2, 3, 3],
        y=zeros(6),
    )
    brmi = (@brm begin
        mu ~ 1 + (x + x & g + g & h | subj)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end)(df)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    @test sb.data[:int_x_x_g_lvl_2] == df.x .* (df.g .== 2)
    @test sb.data[:int_x_x_g_lvl_3] == df.x .* (df.g .== 3)
    @test sb.data[:int_g_lvl_2_x_h_lvl_2] == (df.g .== 2) .* (df.h .== 2)
    @test sb.data[:int_g_lvl_3_x_h_lvl_2] == (df.g .== 3) .* (df.h .== 2)
    code = BRM.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "ranef interaction in a shared |ID| bucket" begin
    df = (;
        x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        g=[1, 2, 3, 1, 2, 3],
        subj=[1, 1, 2, 2, 3, 3],
        y=zeros(6),
    )
    brmi = (@brm begin
        mu ~ 1 + (1 + x & g | p | subj)
        sd(mu, p, int_x_x_g_lvl_2) ~ Exponential(2)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end)(df)
    margins = BRM.ranefcoefnames(brmi, :p)
    @test [m.coefficient for m in margins] ==
          [:Intercept, :int_x_x_g_lvl_2, :int_x_x_g_lvl_3]
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    @test sb.data[:int_x_x_g_lvl_2] == df.x .* (df.g .== 2)
    code = BRM.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "ranef interaction reprocess" begin
    brmi = (@brm begin
        mu ~ 1 + Srace + Sobj + (Srace + Sobj + Srace & Sobj | subject)
        sigma ~ Exponential(1)
        rate ~ Normal(mu, sigma)
    end)(RI_DF)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    future = (;
        rate=zeros(2),
        Srace=[1.0, -1.0],
        Sobj=[-1.0, -1.0],
        subject=[1, 2],
    )
    replay = BRM.reprocess(sb, future)
    @test replay.data[:int_Srace_x_Sobj] ≈ future.Srace .* future.Sobj
    @test BRM.stan_code(replay) == BRM.stan_code(sb)
end

@testset "ranef interaction refuses non-data operands" begin
    df = (;
        x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        subj=[1, 1, 2, 2, 3, 3],
        y=zeros(6),
    )
    bad = (@brm begin
        mu ~ 1 + (sigma & x | subj)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end)(df)
    @test_throws "must be a raw data column" SBBRMI(
        bad; mod=@__MODULE__, total_groups=())
end
