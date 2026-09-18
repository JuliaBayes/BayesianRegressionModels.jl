# test/smooth_distributional.jl — same-column `s()` smooth in two linear predictors.
#
# Run: julia --project=test test/smooth_distributional.jl
#
# Snag distributional-m-3f3a622f: `log(mu) ~ 1 + s(h)` together with
# `log(alpha) ~ 1 + s(h)` (bambi bikes-style distributional model) emitted two
# identically-named `s_h` carriers and died in StanBlocks transpile with
# `AssertionError: name ∉ keys(info)`. Carriers now disambiguate like
# `gp`/`hsgp`: the first `s(x)` keeps the historical `s_<x>` names while a
# repeat of the same column takes `s_<target>_<x>` (+ serial).
using Test
using BayesianRegressionModels
using StanBlocks

const SMOOTH_DISTRIBUTIONAL_N = 30

function smooth_distributional_df(; h_shift=0.0, h_scale=1.0)
    h = h_shift .+ h_scale .* collect(range(0.0, 10.0; length=SMOOTH_DISTRIBUTIONAL_N))
    count = [2 + mod(i, 7) + mod(i * 3, 5) for i in 1:SMOOTH_DISTRIBUTIONAL_N]
    y = [0.5 + 0.1 * i for i in 1:SMOOTH_DISTRIBUTIONAL_N]
    (; h, count, y)
end

# The reported case: one column smoothed in both distributional predictors.
smooth_distributional_model(df) = @brm df begin
    log(mu) ~ 1 + s(h)
    log(alpha) ~ 1 + s(h)
    count ~ NegativeBinomial2(mu, alpha)
end

# Single use keeps the historical carrier names byte-identically.
smooth_single_model(df) = @brm df begin
    mu ~ 1 + s(h)
    y ~ Normal(mu, 1)
end

# A same-predictor repeat disambiguates by the same mechanism. (Statistically
# unidentified — two identical bases — but BRM's contract is distinct carriers,
# exactly as for a repeated `gp` axis; the assertion below pins the mechanism.)
smooth_duplicate_model(df) = @brm df begin
    mu ~ 1 + s(h) + s(h)
    y ~ Normal(mu, 1)
end

# Per-predictor smoothing-scale priors still bind their own smooth.
smooth_distributional_priors_model(df) = @brm df begin
    log(mu) ~ 1 + s(h)
    log(alpha) ~ 1 + s(h)
    count ~ NegativeBinomial2(mu, alpha)
    sd(mu, s(h)) ~ Exponential(2)
    sd(alpha, s(h)) ~ Exponential(3)
end

@testset "same-column smooth across distributional predictors" begin
    df = smooth_distributional_df()
    n = SMOOTH_DISTRIBUTIONAL_N

    sb_dist = SBBRMI(smooth_distributional_model(df); mod=@__MODULE__)
    sb_single = SBBRMI(smooth_single_model(df); mod=@__MODULE__)
    sb_dup = SBBRMI(smooth_duplicate_model(df); mod=@__MODULE__)

    @testset "distinct carriers, identical bases" begin
        @test size(sb_dist.data[:Xnull_h]) == (n, 2)
        @test size(sb_dist.data[:Zpen_h]) == (n, 8)
        @test size(sb_dist.data[:Xnull_log_alpha_h]) == (n, 2)
        @test size(sb_dist.data[:Zpen_log_alpha_h]) == (n, 8)
        # Same column, same fit — only the carrier names differ.
        @test sb_dist.data[:Xnull_h] == sb_dist.data[:Xnull_log_alpha_h]
        @test sb_dist.data[:Zpen_h] == sb_dist.data[:Zpen_log_alpha_h]
        # Same-predictor repeat: first keeps `s_h`, second takes the target.
        @test size(sb_dup.data[:Xnull_h]) == (n, 2)
        @test size(sb_dup.data[:Xnull_mu_h]) == (n, 2)
    end

    @testset "single use keeps the historical names" begin
        @test haskey(sb_single.data, :Xnull_h)
        @test haskey(sb_single.data, :Zpen_h)
        @test !haskey(sb_single.data, :Xnull_mu_h)
        code = StanBlocks.stan_code(sb_single.model)
        @test occursin("s_h", code)
        @test !occursin("s_mu_h", code)
    end

    @testset "transpile and stanc" begin
        for candidate in (sb_dist, sb_dup)
            @test StanBlocks.stan.transpiles(candidate.model)
            code = StanBlocks.stan_code(candidate.model)
            @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
        end
        dist_code = StanBlocks.stan_code(sb_dist.model)
        @test occursin("s_h", dist_code)
        @test occursin("s_log_alpha_h", dist_code)
    end

    @testset "frozen and fresh replay re-evaluate both bases" begin
        replay_df = smooth_distributional_df(; h_shift=0.4, h_scale=1.2)
        frozen = reprocess(sb_dist, replay_df)
        fresh = reprocess(sb_dist, replay_df; freeze_constants=false)
        code = StanBlocks.stan_code(sb_dist.model)
        @test StanBlocks.stan_code(frozen.model) == code
        @test StanBlocks.stan_code(fresh.model) == code
        for key in (:Xnull_h, :Zpen_h, :Xnull_log_alpha_h, :Zpen_log_alpha_h)
            @test frozen.data[key] != sb_dist.data[key]
            @test fresh.data[key] != frozen.data[key]
            @test size(frozen.data[key]) == size(sb_dist.data[key])
        end
    end

    @testset "descriptor enumerates both spline inputs" begin
        inputs = [i for i in brm_descriptor(sb_dist).inputs if i.transform === :spline]
        @test sort!([i.name for i in inputs]) == [:Xnull_h, :Xnull_log_alpha_h]
        @test all(i -> i.column === :h, inputs)
    end

    @testset "per-predictor smoothing priors bind their own smooth" begin
        sb_priors = SBBRMI(smooth_distributional_priors_model(df); mod=@__MODULE__)
        code = StanBlocks.stan_code(sb_priors.model)
        # Distributions.Exponential is scale-parameterized, Stan's is rate:
        # Exponential(2) -> 0.5, Exponential(3) -> 1/3. The shared hash names
        # the vector-prior template; the rate argument carries the value.
        @test occursin(r"s_h_sd_pen ~ brm_vector_prior_[0-9a-f]+\(0\.5\);", code)
        @test occursin(
            r"s_log_alpha_h_sd_pen ~ brm_vector_prior_[0-9a-f]+\(0\.3333333333333333\);",
            code)
        @test StanBlocks.stan.transpiles(sb_priors.model)
        @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    end
end
