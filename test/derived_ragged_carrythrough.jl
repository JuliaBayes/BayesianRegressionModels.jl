# test/derived_ragged_carrythrough.jl — snag `reprocess-derive-765e8d2d`.
#
# A data-only kernel argument can be a per-group ragged column. When it is
# held out, BRM stores StanBlocks' `(mem, ends)` carrier. `reprocess` must
# still accept the replay design's RAW vector-of-vectors column and let the
# SLIC retrace make that carrier again, for both same-group replay and
# generated-quantity group resampling.
#
# Run: julia --project=test test/derived_ragged_carrythrough.jl

using Test
using BayesianRegressionModels
using StanBlocks

@testset "derived ragged carry-through" begin
    replay_df(; subs=["s1", "s2", "s3"]) = (;
        subject = subs,
        weight = collect(range(60.0, 90.0; length=length(subs))),
        tgi_bin = [[0.0, 1.0], [1.0], [0.0, 1.0, 1.0]],
    )
    builder = @brm begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + weight + (1 | p | subject)
        pred ~ kernel(tgi_bin, log_CL) do ys, lCL
            mu = sum(ys) .* exp(lCL)
            ys ~ normal(mu, sigma)
            mu
        end
    end
    # `total_groups=()` pins the conventional GQ resample path, matching the
    # existing kernel replay tests.
    train = replay_df()
    sb = SBBRMI(builder(train); mod=@__MODULE__, total_groups=(),
                held_out=:tgi_bin)
    fitted = StanBlocks.getvalue(sb.data[:tgi_bin])
    @test fitted == (; mem=[0.0, 1.0, 1.0, 0.0, 1.0, 1.0], ends=[2, 3, 6])
    @test !haskey(sb.preproc, :tgi_bin)
    @test sb.held_out == Set([:tgi_bin])
    @test !isempty(sb.bindings)

    replayed = reprocess(sb, train; freeze_constants=true)
    @test StanBlocks.getvalue(replayed.data[:tgi_bin]) == fitted
    @test StanBlocks.stan_code(replayed.model) == StanBlocks.stan_code(sb.model)
    @test replayed.held_out == sb.held_out
    @test replayed.bindings == sb.bindings

    without_tgi = (; subject=train.subject, weight=train.weight)
    @test_throws "data key `tgi_bin`" reprocess(sb, without_tgi)

    future = replay_df(; subs=["n1", "n2", "n3", "n4"])
    future = merge(future, (; tgi_bin=[[0.0], [1.0, 0.0], [1.0], [0.0, 1.0, 1.0]]))
    resampled = reprocess(sb, future; freeze_constants=true,
                          resample_groups=[:subject])
    @test StanBlocks.getvalue(resampled.data[:tgi_bin]) ==
        (; mem=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0], ends=[1, 3, 4, 7])
    @test resampled.held_out == sb.held_out
    @test resampled.bindings == sb.bindings

    code = StanBlocks.stan_code(resampled.model)
    @test code != StanBlocks.stan_code(sb.model)
    @test occursin("tgi_bin_mem_n", code)
    @test occursin("tgi_bin_ends_n", code)
    @test occursin("std_normal_vector_rng", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end
