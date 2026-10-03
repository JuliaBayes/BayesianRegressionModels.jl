using Test
using BayesianRegressionModels
using StanBlocks
using Distributions
using Statistics
using BridgeStan
using LogDensityProblems
include(joinpath(@__DIR__, "testset_filter.jl"))

const MISSING_TOY = (;
    subject=[1, 2, 3, 4, 5],
    x=Union{Missing,Float64}[0.8, missing, 2.1, missing, 1.4],
    z=Union{Missing,Float64}[missing, 1.1, 0.9, 1.6, missing],
    y=[-0.2, 0.7, 1.2, -0.4, 0.5],
)
const MISSING_PLAIN = @brm begin
    mx ~ Normal(0, 1)
    sx ~ LogNormal(0, 0.3)
    mi(x) ~ LogNormal(mx, sx)
    mu ~ 1 + x
    y ~ Normal(mu, 1)
end

@stestset "missing covariate replay preserves fitted row positions" begin
    original = deepcopy(MISSING_TOY)
    sb = SBBRMI(MISSING_PLAIN(MISSING_TOY); mod=@__MODULE__)
    same_mask = merge(MISSING_TOY, (;
        x=Union{Missing,Float64}[1.3, missing, 2.7, missing, 1.8],))
    replay = reprocess(sb, same_mask)
    @test replay.data[:x_obs] == [1.3, 2.7, 1.8]
    @test replay.data[:Jobs_x] == [1, 3, 5]
    @test replay.data[:Jmis_x] == [2, 4]
    @test BayesianRegressionModels.stan_code(replay) == BayesianRegressionModels.stan_code(sb)
    changed = merge(MISSING_TOY, (;
        x=Union{Missing,Float64}[missing, 0.6, 2.1, missing, 1.4],))
    # Fitted draws address rows, so a changed frozen mask would reassign them.
    @test_throws r"same fitted missing-row positions" reprocess(sb, changed)
    fresh = reprocess(sb, changed; freeze_constants=false)
    @test fresh.data[:Jobs_x] == [2, 3, 5]
    @test fresh.data[:Jmis_x] == [1, 4]
    @test fresh.data[:x_obs] == [0.6, 2.1, 1.4]
    @test fresh.preproc[:x_obs].const_.missing_indices == [1, 4]
    @test isequal(MISSING_TOY, original)
    @test sb.data[:x_obs] == [0.8, 2.1, 1.4]
end
