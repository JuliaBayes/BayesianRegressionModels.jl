# test/likelihood_weights.jl — typed formula-level observation weights.
#
# Run on a capable host:
#   julia --startup-file=no --project=. test/likelihood_weights.jl

using Test
using BayesianRegressionModels
using Distributions
using LogDensityProblems
using Statistics
using StanBlocks

const BRM = BayesianRegressionModels

analytic_df = (;
    x=[-1.0, -0.25, 0.5, 1.25],
    y=[-0.4, 0.1, 0.8, 1.5],
    replicate_k=[1, 2, 4, 3],
)

analytic_builder = @brm begin
    mu ~ 1 + x
    y ~ weighted(Normal(mu, 1.5), aweights(replicate_k))
end

@testset "weighted(distribution, weights) is retained in the BRMI" begin
    brmi = analytic_builder(analytic_df)
    y_op = parent(brmi.operations.y)
    _, rhs = getargs(y_op, 2)
    @test getf(rhs) === weighted
    distribution, weight = getargs(rhs, 2)
    @test getf(distribution) === Normal
    @test getf(weight) === aweights
    @test name(only(getargs(weight))) === :replicate_k
end

@testset "Normal analytic weights lower to sigma / sqrt(weight)" begin
    sb = SBBRMI(analytic_builder(analytic_df))
    code = BRM.stan_code(sb)
    @test sb.data[:brm_weight_y] == Float64.(analytic_df.replicate_k)
    @test occursin("sqrt(brm_weight_y)", code)
    # What this pins is the SCALE: analytic weights rescale sigma and hand the
    # result to the ordinary Gaussian, so no `weighted_*` HOF is involved (the
    # frequency/power testset below is the contrast). WHICH Gaussian primitive
    # carries that scale is not part of the contract -- this model is also pure
    # population, so `_sb_fuse_normal_id_glm!` (sbimpl.jl) lowers it to Stan's
    # fused `normal_id_glm`, and the pointwise-loglik twin follows the density.
    @test occursin("(1.5 ./ sqrt(brm_weight_y))", code)
    @test occursin(r"y ~ normal(_id_glm)?\(", code)
    @test occursin(r"normal(_id_glm)?_lpdfs", code)
    @test occursin("normal_rng", code)
    @test !occursin("weighted_normal", code)

    new_df = (;
        x=[-0.5, 0.25, 1.0],
        y=[-0.2, 0.4, 1.1],
        replicate_k=[5, 1, 2],
    )
    replay = reprocess(sb, new_df)
    @test replay.data[:brm_weight_y] == Float64.(new_df.replicate_k)
    @test BRM.stan_code(replay) == code
end

frequency_builder = @brm begin
    mu ~ 1
    y ~ weighted(Normal(mu, 1.0), fweights(repeats))
end

power_builder = @brm begin
    log(rate) ~ 1
    counts ~ weighted(Poisson(rate), weights(power))
end

@testset "frequency and power weights use the StanBlocks weighted HOF" begin
    freq_df = (; y=[-0.2, 0.1, 0.7], repeats=[1, 3, 2])
    freq_sb = SBBRMI(frequency_builder(freq_df))
    freq_code = BRM.stan_code(freq_sb)
    @test freq_sb.data[:brm_weight_y] == [1.0, 3.0, 2.0]
    @test occursin("weighted_normal_lpdf", freq_code)
    @test occursin("weighted_normal_lpdfs", freq_code)
    @test occursin("weighted_vector_normal_rng", freq_code)

    power_df = (; counts=[0, 1, 3], power=[0.25, 1.0, 2.5])
    power_sb = SBBRMI(power_builder(power_df))
    power_code = BRM.stan_code(power_sb)
    @test power_sb.data[:brm_weight_counts] == power_df.power
    @test occursin("weighted_poisson_lpmf", power_code)
    @test occursin("weighted_poisson_lpmfs", power_code)
    @test occursin("weighted_int_poisson_rng", power_code)
end

@testset "invalid or unsupported weight semantics fail loudly" begin
    bad_analytic = (; analytic_df..., replicate_k=[1, 0, 2, 3])
    @test_throws ErrorException SBBRMI(analytic_builder(bad_analytic))

    bad_frequency = (; y=[0.1, 0.2], repeats=[1.0, 1.5])
    @test_throws ErrorException SBBRMI(frequency_builder(bad_frequency))

    mismatched = (; y=[0.1, 0.2, 0.3], repeats=[1, 2])
    @test_throws ErrorException SBBRMI(frequency_builder(mismatched))

    probability_builder = @brm begin
        y ~ weighted(Normal(0.0, 1.0), pweights(sampling_weight))
    end
    @test_throws ErrorException SBBRMI(probability_builder((;
        y=[0.1, 0.2], sampling_weight=[1.0, 2.0])))

    analytic_poisson = @brm begin
        counts ~ weighted(Poisson(1.0), aweights(precision))
    end
    @test_throws ErrorException SBBRMI(analytic_poisson((;
        counts=[0, 1], precision=[1.0, 2.0])))
end

@testset "compiled pointwise and predictive semantics follow the weight type" begin
    runtime_builder = @brm begin
        y_analytic ~ weighted(Normal(0.3, 1.2), aweights(precision))
        y_power ~ weighted(Normal(-0.2, 0.8), weights(power))
    end
    runtime_df = (;
        y_analytic=[0.7], precision=[4.0],
        y_power=[0.1], power=[2.5],
    )
    descriptor = brm_descriptor(runtime_builder, runtime_df; mod=@__MODULE__)
    code = brm_execute(descriptor, :transpile)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    cache = joinpath(tempdir(), "brm-likelihood-weights")
    isdir(cache) || mkpath(cache)
    problem = brm_execute(
        descriptor, :instantiate;
        path=joinpath(cache, string(descriptor.id) * ".stan"))
    theta = Float64[]
    @test LogDensityProblems.dimension(problem) == 0

    pointwise = brm_execute(
        descriptor, :pointwise_loglik; problem, draws=theta, seed=20260729)
    @test pointwise.y_analytic_likelihood ≈
          [logpdf(Normal(0.3, 1.2 / sqrt(4.0)), 0.7)] atol=1e-10
    @test pointwise.y_power_likelihood ≈
          [2.5 * logpdf(Normal(-0.2, 0.8), 0.1)] atol=1e-10

    predictions = brm_execute(
        descriptor, :predict;
        problem, draws=zeros(0, 4096), seed=20260729)
    @test mean(predictions.y_analytic_gen) ≈ 0.3 atol=0.04
    @test var(vec(predictions.y_analytic_gen)) ≈ (1.2 / sqrt(4.0))^2 atol=0.04
    @test mean(predictions.y_power_gen) ≈ -0.2 atol=0.05
    @test var(vec(predictions.y_power_gen)) ≈ 0.8^2 atol=0.06
end

@testset "frequency weights match explicit row replication" begin
    weighted_repeated_builder = @brm begin
        mu ~ 1
        log(sigma) ~ 1
        y ~ weighted(Normal(mu, sigma), fweights(repeats))
    end
    repeated_builder = @brm begin
        mu ~ 1
        log(sigma) ~ 1
        y ~ Normal(mu, sigma)
    end
    freq_df = (; y=[-0.2, 0.1, 0.7], repeats=[1, 3, 2])
    repeated_df = (; y=reduce(vcat,
        fill(freq_df.y[i], freq_df.repeats[i]) for i in eachindex(freq_df.y)))
    cache = mktempdir()

    weighted_problem = StanBlocks.stan_instantiate(
        SBBRMI(weighted_repeated_builder(freq_df)).model;
        path=joinpath(cache, "brm-frequency-weighted.stan"))
    repeated_problem = StanBlocks.stan_instantiate(
        SBBRMI(repeated_builder(repeated_df)).model;
        path=joinpath(cache, "brm-frequency-repeated.stan"))
    @test LogDensityProblems.dimension(weighted_problem) ==
          LogDensityProblems.dimension(repeated_problem) == 2

    lp_offsets = Float64[]
    for theta in ([-0.7, -0.3], [0.2, 0.1], [1.1, 0.5])
        weighted_lp, weighted_grad =
            LogDensityProblems.logdensity_and_gradient(weighted_problem, theta)
        repeated_lp, repeated_grad =
            LogDensityProblems.logdensity_and_gradient(repeated_problem, theta)
        push!(lp_offsets, weighted_lp - repeated_lp)
        @test weighted_grad ≈ repeated_grad atol=1e-8
    end
    # Stan's propto target may omit a parameter-independent normalizing constant;
    # posterior equivalence requires the same log-density shape and gradient.
    @test all(offset -> isapprox(offset, first(lp_offsets); atol=1e-8), lp_offsets)
end

@testset "power and frequency weights scale a bounded observation" begin
    # `weighted` over censored / truncated / interval_censored: each row's
    # likelihood is its weight times the bounded law's, while the prediction
    # is the bounded law's own unweighted draw.
    bounded_builder = @brm begin
        y_censored ~ weighted(censored(Normal(0.3, 1.2); lower=lloq), weights(w))
        y_truncated ~ weighted(truncated(Normal(-0.2, 0.8), -1.0, 1.0), fweights(n))
        y_interval ~ weighted(interval_censored(Normal(0.1, 0.5); upper=y_upper),
            weights(w))
    end
    bounded_df = (;
        y_censored=[0.0, 0.9, -0.1], lloq=[0.0, 0.0, -0.1], w=[0.5, 2.0, 0.0],
        y_truncated=[-0.5, 0.2, 0.9], n=[1, 3, 2],
        y_interval=[-0.2, 0.3, 0.0], y_upper=[0.1, 0.8, 0.4],
    )
    descriptor = brm_descriptor(bounded_builder, bounded_df; mod=@__MODULE__)
    code = brm_execute(descriptor, :transpile)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    problem = brm_execute(
        descriptor, :instantiate;
        path=joinpath(mktempdir(), string(descriptor.id) * ".stan"))
    @test LogDensityProblems.dimension(problem) == 0

    pointwise = brm_execute(
        descriptor, :pointwise_loglik; problem, draws=Float64[], seed=20261019)
    clamped = Normal(0.3, 1.2)
    @test pointwise.y_censored_likelihood ≈ bounded_df.w .* [
        y == lloq ? logcdf(clamped, lloq) : logpdf(clamped, y)
        for (y, lloq) in zip(bounded_df.y_censored, bounded_df.lloq)] atol=1e-10
    @test pointwise.y_truncated_likelihood ≈ bounded_df.n .*
        logpdf.(truncated(Normal(-0.2, 0.8), -1.0, 1.0), bounded_df.y_truncated) atol=1e-10
    interval = Normal(0.1, 0.5)
    @test pointwise.y_interval_likelihood ≈ bounded_df.w .*
        log.(cdf.(interval, bounded_df.y_upper) .- cdf.(interval, bounded_df.y_interval)) atol=1e-10

    predictions = brm_execute(
        descriptor, :predict; problem, draws=zeros(0, 4096), seed=20261019)
    for (j, lloq) in enumerate(bounded_df.lloq)
        draws = predictions.y_censored_gen[j, :]
        @test minimum(draws) == lloq
        @test mean(==(lloq), draws) ≈ cdf(clamped, lloq) atol=0.03
    end
    @test all(x -> -1.0 <= x <= 1.0, predictions.y_truncated_gen)
    @test mean(predictions.y_truncated_gen) ≈
        mean(truncated(Normal(-0.2, 0.8), -1.0, 1.0)) atol=0.02
    @test mean(predictions.y_interval_gen) ≈ 0.1 atol=0.02
    @test var(vec(predictions.y_interval_gen)) ≈ 0.25 atol=0.02
end
