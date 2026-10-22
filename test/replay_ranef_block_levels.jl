# test/replay_ranef_block_levels.jl — REGRESSION for snag ranef-blocks-rep-c8fc0683.
#
# A replay frame may carry only some fitted levels of a group that is NOT
# resampled: `reprocess` codes its rows through the frozen fitted labels and
# sizes the block by the fitted count. `ranef_blocks` used to re-derive the
# block's labels from the replay rows instead, so it (and `transport_draws`,
# which calls it first) refused every such target with
# `grouping factor `h` has 1 training levels but block … was emitted with
# n_groups = 3`. The labels now come from the `:group_index` record the index
# was coded through.
#
# Run: julia --project=test test/replay_ranef_block_levels.jl
# (the compiled half needs BridgeStan through StanBlocks; set
# BRM_PRED_RUNTIME=0 for the declaration-level half only.)

using Test
using BayesianRegressionModels
using StanBlocks
using Random
using Distributions: Normal, Exponential

const PRED_RUNTIME = get(ENV, "BRM_PRED_RUNTIME", "1") != "0"
const BS = StanBlocks.BridgeStan

const N = 30
const train = (; y=collect(range(-1.0, 1.0; length=N)), x=collect(range(2.0, -2.0; length=N)),
    g=[string('a' + (i - 1) % 10) for i in 1:N], h=[["p", "q", "r"][(i - 1) % 3 + 1] for i in 1:N])
const G_LEVELS = string.('a':'j')
const H_LEVELS = ["p", "q", "r"]
# A new `g` level and one fitted `h` level, not the first, so neither the
# replay's own label set nor a prefix of the fitted one is the answer.
const new_g = (; y=[0.0], x=[0.0], g=["z"], h=["q"])
# Fitted levels only, every one of them a strict subset of its fitted set.
const same = (; y=[0.0, 0.5], x=[0.0, 1.0], g=["c", "e"], h=["r", "r"])

block(blocks, group) = only(b for b in blocks if b.group === group)

mu_model = @brm begin
    mu ~ 1 + x + (1 | g) + (1 | h)
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
sigma_model = @brm begin
    mu ~ 1 + x + (1 | g)
    log(sigma) ~ 1 + (1 | h)
    y ~ Normal(mu, sigma)
end

fits = (; mu=SBBRMI(mu_model(train); mod=@__MODULE__, total_groups=()),
    sigma=SBBRMI(sigma_model(train); mod=@__MODULE__, total_groups=()))

@testset "fit: labels are the fitted training labels ($k)" for (k, sb) in pairs(fits)
    blocks = ranef_blocks(sb)
    @test block(blocks, :g).levels == G_LEVELS
    @test block(blocks, :h).levels == H_LEVELS
    @test !any(b -> b.generated, blocks)
end

@testset "resample replay keeps a non-resampled group's fitted labels ($k)" for (k, sb) in pairs(fits)
    rp = reprocess(sb, new_g; resample_groups=[:g])
    blocks = ranef_blocks(rp)
    g, h = block(blocks, :g), block(blocks, :h)
    @test g.generated
    @test (g.levels, g.n_groups) == (["z"], 1)
    @test !h.generated
    @test (h.levels, h.n_groups) == (H_LEVELS, 3)
    # The descriptor resolves the same blocks.
    d = brm_descriptor(rp)
    @test [(b.group, b.levels) for b in ranef_blocks(d)] ==
          [(b.group, b.levels) for b in blocks]
end

@testset "frozen replay of fitted levels keeps every fitted label ($k)" for (k, sb) in pairs(fits)
    rp = reprocess(sb, same)
    blocks = ranef_blocks(rp)
    @test (block(blocks, :g).levels, block(blocks, :g).n_groups) == (G_LEVELS, 10)
    @test (block(blocks, :h).levels, block(blocks, :h).n_groups) == (H_LEVELS, 3)
    @test !any(b -> b.generated, blocks)
end

@testset "shared |ID| bucket replays with a subset of its levels" begin
    bucket = @brm begin
        mu ~ 1 + x + (1 | g) + (1 | p | h)
        log(sigma) ~ 1 + (1 | p | h)
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(bucket(train); mod=@__MODULE__, total_groups=())
    for rp in (reprocess(sb, new_g; resample_groups=[:g]), reprocess(sb, same))
        h = block(ranef_blocks(rp), :h)
        @test h.id === :p
        @test (h.levels, h.n_groups, h.n_terms) == (H_LEVELS, 3, 2)
    end
end

@testset "a suffix-bearing group name replays with a subset of its levels" begin
    # `h__nocor__1` reads like a zerocorr margin of `h`, so the block's origin
    # is recovered from the record: the column whose rows carry the recorded
    # label each index points at. A replay's rows carry a subset of them.
    nc = (; train..., h__nocor__1=[["u", "v", "w"][(i - 1) % 3 + 1] for i in 1:N])
    lit = @brm begin
        mu ~ 1 + x + (1 | g) + (1 | h) + (1 | h__nocor__1)
        y ~ Normal(mu, 1.0)
    end
    sb = SBBRMI(lit(nc); mod=@__MODULE__, total_groups=())
    @test block(ranef_blocks(sb), :h__nocor__1).levels == ["u", "v", "w"]
    for rp in (reprocess(sb, (; new_g..., h__nocor__1=["w"]); resample_groups=[:g]),
               reprocess(sb, (; same..., h__nocor__1=["v", "v"])))
        blocks = ranef_blocks(rp)
        @test (block(blocks, :h).levels, block(blocks, :h).n_groups) == (H_LEVELS, 3)
        lb = block(blocks, :h__nocor__1)
        @test (lb.levels, lb.n_groups) == (["u", "v", "w"], 3)
    end
end

@testset "zerocorr `||` replay (todo `12u0ozx`)" begin
    zc = @brm begin
        mu ~ 1 + x + (1 | g) + (1 + x || h)
        y ~ Normal(mu, 1.0)
    end
    sb = SBBRMI(zc(train); mod=@__MODULE__, total_groups=())
    @test [b.levels for b in ranef_blocks(sb) if b.group === :h] == [H_LEVELS, H_LEVELS]
    # Not built: `reprocess` looks the margins' `:group_index` records up under
    # their synthetic `h__nocor__k` names, which no replay frame carries
    # (`type NamedTuple has no field h__nocor__1`), on full frames too.
    @test_broken (reprocess(sb, train); true)
    @test_broken (reprocess(sb, new_g; resample_groups=[:g]); true)
end

@testset "a block sized by its index addresses that prefix of the fitted labels" begin
    # `cv_groups` sizes `h`'s block by `max(h_idx)` (a Stan-side local), so a
    # frozen replay whose rows reach only fitted code 2 has two columns: the
    # first two fitted labels, whichever of them the rows carry.
    sb = SBBRMI(mu_model(train); mod=@__MODULE__, total_groups=(), cv_groups=[:h])
    @test block(ranef_blocks(sb), :h).levels == H_LEVELS
    h = block(ranef_blocks(reprocess(sb, (; y=[0.0], x=[0.0], g=["a"], h=["q"]))), :h)
    @test (h.levels, h.n_groups) == (["p", "q"], 2)
end

if !PRED_RUNTIME
    @info "BRM_PRED_RUNTIME=0 — skipping the compiled-model half"
else

# Every target coordinate of `moved` is the source coordinate of the same
# name: the fitted `h` effects are copied by label, the new `g` level is
# generated Stan-side, and nothing is drawn Julia-side.
function check_named_copy(sb, rp)
    from_p = StanBlocks.stan_instantiate(sb.model)
    to_p = StanBlocks.stan_instantiate(rp.model)
    unc_from = BS.param_unc_names(from_p.model)
    unc_to = BS.param_unc_names(to_p.model)
    draws = randn(MersenneTwister(3), 5, length(unc_from))
    moved = transport_draws(sb, rp, draws, unc_from, unc_to; rng=MersenneTwister(5))
    pos_from = Dict(String(n) => i for (i, n) in enumerate(unc_from))
    @test all(j -> moved[:, j] == draws[:, pos_from[String(unc_to[j])]], eachindex(unc_to))
    h_from, h_to = ranef_coordinates.((block(ranef_blocks(sb), :h), block(ranef_blocks(rp), :h)),
                                      (unc_from, unc_to))
    @test moved[:, vec(h_to)] == draws[:, vec(h_from)]
    @test all(r -> isfinite(BS.log_density(to_p.model, moved[r, :])), axes(moved, 1))
    unc_to
end

@testset "transport_draws onto a replay with a subset of a fitted group ($k)" for (k, sb) in pairs(fits)
    unc_to = check_named_copy(sb, reprocess(sb, new_g; resample_groups=[:g]))
    @test !any(n -> occursin("_g_z", n), unc_to)   # g re-draws in generated quantities
    check_named_copy(sb, reprocess(sb, same))
end

end # PRED_RUNTIME
