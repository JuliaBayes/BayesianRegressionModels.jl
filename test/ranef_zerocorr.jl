# test/ranef_zerocorr.jl — `||` (zerocorr) uncorrelated random-effect blocks,
# with the drop-intercept `0` marker.
#
# `(t1 + t2 + … || g)` expands into one INDEPENDENT `(t_i | g__nocor__i)` scalar
# ranef per term (no shared correlation). The intercept-suppressing `0` in
# `(0 + x || g)` is the standard formula-language drop-intercept marker and
# contributes NO term, so the expander must drop it BEFORE splitting into
# per-term nocor groups — exactly as the correlated `|` path drops `0` in
# `_sb_collect_terms!(::Int)`.
#
# Regression: before `ac2db5b` the `0` claimed its own empty `g__nocor__1`
# group, which `_sb_emit_ranefs!` then errored on with
# "ranef (… | g__nocor__1) has no terms after dropping 0" — so `(0 + x || g)`,
# a single uncorrelated random slope and a documented feature-atlas example,
# failed to transpile instead of emitting that lone slope. The correlated
# `(0 + x | g)` form always worked; the two shapes must agree.

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Normal, Exponential

stanc_ok(code) = StanBlocks.stanc_check(code; warn_pedantic=false).ok
codeof(builder, df) = StanBlocks.stan_code(SBBRMI(builder(df); mod=@__MODULE__).model)

const zc_df = (; x=[-1.0, 0.5, 2.0, 0.25], g=[2, 1, 2, 3], y=[0.2, 1.1, -0.4, 0.7])

@testset "`(0 + x || g)` lone uncorrelated slope transpiles" begin
    zc = @brm begin
        mu ~ 1 + x + (0 + x || g)
        y ~ Normal(mu, 1.0)
    end
    cor = @brm begin
        mu ~ 1 + x + (0 + x | g)
        y ~ Normal(mu, 1.0)
    end

    zc_code  = codeof(zc, zc_df)
    cor_code = codeof(cor, zc_df)

    # The bug was a hard error at emission — the lone slope now transpiles.
    @test stanc_ok(zc_code)
    @test !occursin("lkj_corr_cholesky", zc_code)
    # The dropped `0` claims NO group: exactly ONE synthetic nocor group, not a
    # spurious empty second one.
    @test occursin("g__nocor__1", zc_code)
    @test !occursin("g__nocor__2", zc_code)
    # Correlation is vacuous with a single term, so the lone `||` slope emits
    # Stan byte-identical to the single-term correlated `|` modulo the synthetic
    # group name.
    @test replace(zc_code, "g__nocor__1" => "g") == cor_code
end

@testset "multi-term `||` still splits into independent nocor groups" begin
    # The `0`-drop must not collapse a genuine multi-term block: `1 + x` keeps
    # the intercept term and expands into TWO independent nocor groups.
    multi = @brm begin
        mu ~ 1 + (1 + x || g)
        y ~ Normal(mu, 1.0)
    end
    code = codeof(multi, zc_df)
    @test stanc_ok(code)
    @test occursin("g__nocor__1", code)
    @test occursin("g__nocor__2", code)
end

@testset "zerocorr synthetic groups retain their raw grouping factor" begin
    zerocorr = @brm begin
        mu ~ 1 + x + (1 + x || g)
        y ~ Normal(mu, 1.0)
    end
    sb = SBBRMI(zerocorr(zc_df); mod=@__MODULE__)
    blocks = ranef_blocks(sb)
    @test length(blocks) == 2
    @test [b.family for b in blocks] == [:ranef_intercept, :ranef_slope]
    @test [b.group for b in blocks] == [:g, :g]
    @test all(b -> b.n_terms == 1 && b.n_groups == 3, blocks)
    @test all(b -> b.levels == [1, 2, 3], blocks)
    unc = String[]
    for b in blocks
        if b.family === :ranef_intercept
            push!(unc, "$(b.binding)_log_scale")
            append!(unc, ["$(b.binding)_xi.$g" for g in 1:3])
        else
            push!(unc, "$(b.binding)_tau.1")
            append!(unc, ["$(b.binding)_xi.$g" for g in 1:3])
        end
    end
    adaptive = adaptive_centering_blocks(sb, unc)
    @test length(adaptive) == 2
    @test all(b -> size(b.effects) == (1, 3), adaptive)
end

@testset "`(… || g)` with no surviving term errors clearly" begin
    # Only the `0` marker, nothing else: dropping it leaves an empty block, and
    # the expander must reject that loudly rather than emit a term-less ranef.
    empty = @brm begin
        mu ~ 1 + (0 || g)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "has no terms after dropping" codeof(empty, zc_df)
end

@testset "feature-atlas zerocorr examples transpile" begin
    # docs/src/feature-atlas.md `crossed_groups`: a lone `(0 + x || site)`
    # alongside correlated and intercept-only blocks.
    cg = @brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x + (1 + x | subject) + (1 | item) + (0 + x || site)
        outcome ~ Normal(mu, sigma)
    end
    cg_df = (; x=[-1.0, 0.5, 2.0, 0.25], subject=[2, 1, 2, 3],
               item=[2, 1, 1, 2], site=[10, 10, 20, 20],
               outcome=[0.2, 1.1, -0.4, 0.7])
    @test stanc_ok(codeof(cg, cg_df))

    # docs/src/feature-atlas.md `grouped_negative_binomial`: a multi-term
    # `(1 + x || subject)` and a lone `(0 + z || batch)` across two predictors.
    gnb = @brm begin
        log(mu) ~ 1 + x + (1 + x || subject)
        log(phi) ~ 1 + z + (0 + z || batch)
        y ~ NegativeBinomial2(mu, phi)
    end
    gnb_df = (; x=[-1.0, 0.5, 2.0, 0.25], z=[0.0, 1.0, -0.5, 0.75],
                subject=[2, 1, 2, 3], batch=[2, 1, 1, 2], y=[0, 2, 5, 1])
    @test stanc_ok(codeof(gnb, gnb_df))
end

# A user's own data column may literally carry a `__nocor__N` name. The suffix
# alone must never reassign it: the emitted index vector arbitrates.
const lit_df = (; x=[-1.0, 0.5, 2.0, 0.25], g__nocor__1=[1, 1, 2, 2],
    y=[0.2, 1.1, -0.4, 0.7])

@testset "literal `g__nocor__1` column keeps its own name" begin
    # No `g` column exists at all: suffix-stripping has nothing to resolve to.
    lit = @brm begin
        mu ~ 1 + x + (1 | g__nocor__1)
        y ~ Normal(mu, 1.0)
    end
    sb = SBBRMI(lit(lit_df); mod=@__MODULE__)
    blocks = ranef_blocks(sb)
    @test length(blocks) == 1
    @test only(blocks).group === :g__nocor__1
    @test only(blocks).levels == [1, 2]
    @test only(blocks).n_groups == 2
end

# Same level SET, different membership: `g = [2,1,2,3]` vs
# `g__nocor__1 = [1,1,2,3]`. Levels alone cannot tell them apart, so the
# emitted index vector must arbitrate — otherwise the literal block silently
# inherits `g`'s membership.
const both_df = (; x=[-1.0, 0.5, 2.0, 0.25], g=[2, 1, 2, 3],
    g__nocor__1=[1, 1, 2, 3], y=[0.2, 1.1, -0.4, 0.7])

@testset "coexisting `g` and literal `g__nocor__1` keep their memberships" begin
    both = @brm begin
        mu ~ 1 + x + (1 | g) + (1 | g__nocor__1)
        y ~ Normal(mu, 1.0)
    end
    sb = SBBRMI(both(both_df); mod=@__MODULE__)
    blocks = ranef_blocks(sb)
    @test length(blocks) == 2
    @test [b.group for b in blocks] == [:g, :g__nocor__1]
    @test blocks[1].levels == [1, 2, 3]
    @test blocks[2].levels == [1, 2, 3]
    # The labels are not interchangeable: each block's own factor must
    # reproduce its own emitted index vector.
    plan = BayesianRegressionModels.generative_plan(sb)
    for b in blocks
        raw = BayesianRegressionModels.column_data(plan.parent, b.group)
        levels = collect(BayesianRegressionModels._sb_fit_levels(raw))
        idx = plan.data[Symbol(b.group, :_idx)]
        @test all(indexin(raw, levels) .== idx)
    end
end

const mm_lit_df = (; x=[0.2, -0.1, 0.4], g__nocor__1=["a", "a", "b"],
    h=["b", "c", "c"], w1=[2.0, 1.0, 0.0], w2=[1.0, 1.0, 3.0],
    y=[0.1, 0.2, 0.3])

@testset "multi-membership over a literal suffix-bearing column" begin
    mm_lit = @brm begin
        sigma ~ Exponential(1)
        loc ~ 1 + (1 | mm(g__nocor__1, h; weights=(w1, w2)))
        y ~ Normal(loc, sigma)
    end
    sb = SBBRMI(mm_lit(mm_lit_df); mod=@__MODULE__)
    blocks = ranef_blocks(sb)
    @test length(blocks) == 1
    @test only(blocks).group == (:g__nocor__1, :h)
end
