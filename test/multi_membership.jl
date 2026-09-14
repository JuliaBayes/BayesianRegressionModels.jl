# Focused acceptance for the typed `mm(...)` random-effect term.
#
# Run (transpile + stanc/data checks; no BridgeStan compile):
#   julia --project=. test/multi_membership.jl

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

weighted_builder = @brm begin
    sigma ~ Exponential(1)
    loc ~ 1 + (1 | mm(g1, g2; weights=(w1, w2)))
    y ~ Normal(loc, sigma)
end

raw_builder = @brm begin
    sigma ~ Exponential(1)
    loc ~ 1 + (1 | mm(g1, g2; weights=(w1, w2), normalize=false))
    y ~ Normal(loc, sigma)
end

equal_builder = @brm begin
    sigma ~ Exponential(1)
    loc ~ 1 + (1 | mm(g1, g2))
    y ~ Normal(loc, sigma)
end

slope_builder = @brm begin
    sigma ~ Exponential(1)
    loc ~ 1 + (1 + x | mm(g1, g2; weights=(w1, w2)))
    y ~ Normal(loc, sigma)
end

shared_source_builder = @brm begin
    sigma ~ Exponential(1)
    loc ~ 1 + w1 + (1 | mm(g1, g2; weights=(w1, w2)))
    y ~ Normal(loc, sigma)
end

df = (
    g1 = ["a", "a", "b"],
    g2 = ["b", "c", "c"],
    w1 = [2.0, 1.0, 0.0],
    w2 = [1.0, 1.0, 3.0],
    x = [0.2, -0.1, 0.4],
    y = [0.1, 0.2, 0.3],
)

function mm_preproc(sb)
    entries = [(k, e) for (k, e) in sb.preproc if e.kind === :multi_membership]
    only(entries)
end

@testset "mm parses to a typed structural term" begin
    brmi = weighted_builder(df)
    _, rhs = getargs(linear_predictor_op(brmi, :loc), 2)
    ranef = last(getargs(rhs))
    term = last(getargs(ranef))
    @test term isa BayesianRegressionModels.MultiMembershipTerm
    @test Tuple(name(g) for g in getargs(term)) == (:g1, :g2)
    @test getkwargs(term).normalize
    @test Tuple(name(w) for w in getkwargs(term).weights) == (:w1, :w2)
    @test grouping_factors(brmi, :loc) == [:g1, :g2]
    @test dependencies(brmi, :loc).data == [:g1, :g2, :w1, :w2]

    one_group = @brm begin loc ~ 1 + (1 | mm(g1)) end
    wrong_arity = @brm begin loc ~ 1 + (1 | mm(g1, g2; weights=(w1,))) end
    wrong_policy = @brm begin loc ~ 1 + (1 | mm(g1, g2; normalize=1)) end
    @test_throws ArgumentError one_group(df)
    @test_throws ArgumentError wrong_arity(df)
    @test_throws ArgumentError wrong_policy(df)
end

@testset "fit materializes one shared level set and strict weights" begin
    sb = SBBRMI(weighted_builder(df); mod=@__MODULE__)
    idx_key, entry = mm_preproc(sb)
    @test entry.raw_ref.groups == (:g1, :g2)
    @test entry.raw_ref.weights == (:w1, :w2)
    @test entry.const_.levels == ["a", "b", "c"]
    @test sb.data[idx_key] == [1, 2, 1, 3, 2, 3]
    @test sb.data[entry.const_.weight_key] ≈ [2/3, 1/3, 1/2, 1/2, 0, 1]
    @test sb.data[entry.const_.n_groups_key] == 3
    @test sb.data[entry.const_.n_obs_key] == 3
    @test sb.data[entry.const_.n_memberships_key] == 2
    @test all(!haskey(sb.data, k) for k in (:g1, :g2, :w1, :w2))
    intercept_code = BayesianRegressionModels.stan_code(sb)
    @test occursin("multi_membership_intercept", intercept_code)
    @test StanBlocks.stanc_check(intercept_code; warn_pedantic=false).ok

    block = only(ranef_blocks(sb))
    @test block.family === :ranef_intercept_draws
    @test block.group == (:g1, :g2)
    @test block.levels == ["a", "b", "c"]
    @test (block.n_terms, block.n_groups) == (1, 3)

    # Either membership-column spelling selects the ONE shared block.
    unc = ["$(block.z).$g" for g in 1:block.n_groups]
    draws = fill(4.0, 2, length(unc))
    @test all(iszero, population_draws(sb, draws, unc; groups=:g1))
    @test all(iszero, population_draws(sb, draws, unc; groups=:g2))

    raw_sb = SBBRMI(raw_builder(df); mod=@__MODULE__)
    _, raw_entry = mm_preproc(raw_sb)
    @test raw_sb.data[raw_entry.const_.weight_key] == [2, 1, 1, 1, 0, 3]

    equal_sb = SBBRMI(equal_builder(df); mod=@__MODULE__)
    _, equal_entry = mm_preproc(equal_sb)
    @test equal_sb.data[equal_entry.const_.weight_key] == fill(0.5, 6)

    slope_sb = SBBRMI(slope_builder(df); mod=@__MODULE__)
    slope_block = only(ranef_blocks(slope_sb))
    @test slope_block.family === :ranef_correlated_draws
    @test slope_block.n_terms == 2
    code = BayesianRegressionModels.stan_code(slope_sb)
    @test occursin("multi_membership_correlated", code)
    @test occursin(string(slope_block.z), code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    shared_sb = SBBRMI(shared_source_builder(df); mod=@__MODULE__)
    @test haskey(shared_sb.data, :w1) # still needed by the population design
    @test !haskey(shared_sb.data, :w2)
    @test StanBlocks.stanc_check(BayesianRegressionModels.stan_code(shared_sb);
                                 warn_pedantic=false).ok
end

@testset "reprocess repeats validation and normalization" begin
    sb = SBBRMI(weighted_builder(df); mod=@__MODULE__)
    idx_key, entry = mm_preproc(sb)
    replay = reprocess(sb, df)
    @test replay.data == sb.data

    changed = merge(df, (; w1=[9.0, 1.0, 1.0], w2=[1.0, 3.0, 1.0]))
    changed_sb = reprocess(sb, changed)
    @test changed_sb.data[entry.const_.weight_key] ≈ [0.9, 0.1, 0.25, 0.75, 0.5, 0.5]
    @test changed_sb.data[idx_key] == sb.data[idx_key]

    @test_throws ErrorException reprocess(sb, merge(df, (; w1=[-1.0, 1.0, 1.0])))
    @test_throws ErrorException reprocess(sb, merge(df, (; w1=[NaN, 1.0, 1.0])))
    @test_throws ErrorException reprocess(sb, merge(df, (; w1=[0.0, 1.0, 1.0], w2=[0.0, 1.0, 1.0])))
    @test_throws ErrorException reprocess(sb, merge(df, (; g1=Union{Missing,String}[missing, "a", "b"])))
    @test_throws ErrorException reprocess(sb, merge(df, (; g2=["b", "c", "new"])))

    expanded = reprocess(sb, merge(df, (; g2=["b", "c", "new"]));
                         freeze_constants=false)
    _, expanded_entry = mm_preproc(expanded)
    @test expanded_entry.const_.levels == ["a", "b", "c", "new"]
    @test expanded.data[expanded_entry.const_.n_groups_key] == 4

    raw_sb = SBBRMI(raw_builder(df); mod=@__MODULE__)
    @test_throws ErrorException reprocess(raw_sb,
        merge(df, (; w1=[0.0, 1.0, 1.0], w2=[0.0, 1.0, 1.0])))
end

mixed_builder = @brm begin
    sigma ~ Exponential(1)
    loc ~ 1 + (1 | g1) + (1 | mm(g1, g2; weights=(w1, w2)))
    y ~ Normal(loc, sigma)
end

@testset "mixed ordinary and multi-membership block metadata" begin
    sb = SBBRMI(mixed_builder(df); mod=@__MODULE__)
    blocks = ranef_blocks(sb)
    @test length(blocks) == 2
    ordinary, mm = blocks
    @test ordinary.family === :ranef_intercept
    @test ordinary.group === :g1
    @test ordinary.n_groups == 2
    # The multi-membership branch must bind its own raw factor tuple: before
    # the explicit assignment it read the ordinary block's stale `:g1`.
    @test mm.family === :ranef_intercept_draws
    @test mm.group == (:g1, :g2)
    @test mm.levels == ["a", "b", "c"]
    @test (mm.n_terms, mm.n_groups) == (1, 3)
    @test StanBlocks.stanc_check(BayesianRegressionModels.stan_code(sb);
                                 warn_pedantic=false).ok
end
