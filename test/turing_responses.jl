using Test, BayesianRegressionModels, Distributions, Turing, Random, Statistics
using LogDensityProblems, LogExpFunctions
import StanBlocks
const BRM = BayesianRegressionModels
const BRMOrderedLogistic = BRM.OrderedLogistic

@testset "fitted categorical coding and generic class-logit expressions" begin
    data = (; x=[-0.8, 0.2, 0.7, 1.0], y=[10, 30, 20, 10])
    builder = @brm begin
        location ~ Normal(0, 1)
        eta ~ 0 + x
        y ~ CategoricalLogit(location + 0.2, eta)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; location=0.3, beta_pop=[0.4])
    expected = [logpdf(CategoricalLogit(0.5, 0.4x), y)
                for (x, y) in zip(data.x, [1, 3, 2, 1])]
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    replay = reprocess(backend, (; x=[0.3, -0.2], y=[30, 10]))
    @test replay.plan.response == [3, 1]
    @test replay.plan.response_fit.levels == backend.plan.response_fit.levels
    @test_throws "not a training level" reprocess(backend, (; x=[0.2], y=[40]))
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "implicit ordered thresholds are shared formula semantics" begin
    data = (; x=[-0.8, 0.2, 0.7, 1.0], y=[1, 3, 2, 1])
    builder = @brm begin
        eta ~ 0 + x
        y ~ BRMOrderedLogistic(eta)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; beta_pop=[0.4], y_cutpoints=[-0.5, 0.8])
    expected = [logpdf(BRMOrderedLogistic(0.4x, parameters.y_cutpoints), y)
                for (x, y) in zip(data.x, data.y)]
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(Normal(), 0.4) + sum(logpdf.(Normal(), parameters.y_cutpoints))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    @test all(in(1:3), turing_posterior_predictive(Xoshiro(42), backend, parameters).y)
    @test reprocess(backend, (; x=[0.1], y=[1])).plan.response_fit.n_levels == 3
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "typed ordinal links and threshold effects" begin
    data = (; x=[-0.8, 0.2, 0.7, 1.0], treat=[0.0, 1.0, 0.0, 1.0],
             y=[10, 30, 20, 10])
    for structure in (Cumulative(), StoppingRatio()), link in (LogitLink(), ProbitLink(), CloglogLink())
        S, L = typeof(structure), typeof(link)
        builder = @brm begin
            eta ~ 0 + x
            discrimination ~ Exponential(1)
            y ~ Ordinal(S(), L(), eta; discrimination)
        end
        backend = TuringBRMI(builder(data))
        parameters = (; beta_pop=[0.4], discrimination=1.2, y_thresholds=[-0.5, 0.8])
        expected = [logpdf(Ordinal(structure, link, 0.4x,
                            parameters.y_thresholds; discrimination=1.2), y)
                    for (x, y) in zip(data.x, [1, 3, 2, 1])]
        @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
        @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
        @test all(in(1:3), turing_posterior_predictive(Xoshiro(42), backend, parameters).y)
    end
    builder = @brm begin
        eta ~ 0 + x
        y ~ Ordinal(StoppingRatio(), ProbitLink(), eta; per_threshold=(treat,))
    end
    backend = TuringBRMI(builder(data))
    parameters = (; beta_pop=[0.4], y_thresholds=[-0.5, 0.8], y_threshold_beta=[0.3, -0.2])
    expected = [logpdf(Ordinal(StoppingRatio(), ProbitLink(),
                    0.4x .+ treat .* parameters.y_threshold_beta,
                    parameters.y_thresholds), y)
                for (x, treat, y) in zip(data.x, data.treat, [1, 3, 2, 1])]
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "one outcome level has zero-information likelihood" begin
    for distribution in (CategoricalLogit(), BRMOrderedLogistic(0.3, Float64[]),
                         Ordinal(Cumulative(), LogitLink(), 0.3, Float64[]),
                         Ordinal(StoppingRatio(), ProbitLink(), 0.3, Float64[]))
        @test logpdf(distribution, 1) == 0
        @test logpdf(distribution, 2) == -Inf
        @test Distributions.probs(distribution) == [1.0]
        @test rand(Xoshiro(42), distribution) == 1
    end
    data = (; y=[1, 1])
    backend = TuringBRMI((@brm begin
        y ~ BRMOrderedLogistic(0.0)
    end)(data))
    @test Turing.logjoint(backend.model, (; y_cutpoints=Float64[])) == 0
end

@testset "generic mixture likelihoods retain density, pointwise, and RNG" begin
    data = (; y=[-2.0, -1.8, 1.9, 2.2])
    builder = @brm begin
        mu1 ~ Normal(-2, 0.1)
        mu2 ~ Normal(2, 0.1)
        log(sigma) ~ 1
        y ~ MixtureModel([
            Normal(mu1, exp(log(sigma))),
            Normal(mu2, exp(log(sigma))),
        ], [0.4, 0.6])
    end
    backend = TuringBRMI(builder(data))
    parameters = (; mu1=-2.0, mu2=2.0, beta_pop=[log(0.3)])
    mixture = MixtureModel([Normal(-2.0, 0.3), Normal(2.0, 0.3)], [0.4, 0.6])
    expected = logpdf.(mixture, data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    @test length(turing_posterior_predictive(
        Xoshiro(42), backend, parameters).y) == length(data.y)
    code = stan_code(SBBRMI(builder(data)))
    @test occursin("log_sum_exp(terms)", code)
    @test occursin("normal_lpdf", code)
    @test occursin("categorical_rng", code)
    @test occursin("normal_rng", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "mixture lowering is exact on every Stan path" begin
    data = (; y=[-2.0, -1.8, 1.9, 2.2])
    builder = @brm begin
        mu1 ~ Normal(-2, 0.1)
        mu2 ~ Normal(2, 0.1)
        log(sigma) ~ 1
        y ~ MixtureModel([
            Normal(mu1, exp(log(sigma))),
            Normal(mu2, exp(log(sigma))),
        ], [0.4, 0.6])
    end
    descriptor = brm_descriptor(builder, data; mod=@__MODULE__, highlights=())
    @test StanBlocks.stanc_check(
        BRM.stan_code(descriptor.plan); warn_pedantic=false).ok
    cache = joinpath(tempdir(), "brm-mixture-response")
    isdir(cache) || mkpath(cache)
    problem = brm_execute(descriptor, :instantiate;
        path=joinpath(cache, string(descriptor.id) * ".stan"))
    @test LogDensityProblems.dimension(problem) == 3
    # Declaration order is the constrained-coordinate order.
    q = [-2.0, 2.0, log(0.3)]
    _, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
    reference(v) = sum(logpdf.(MixtureModel(
        [Normal(v[1], exp(v[3])), Normal(v[2], exp(v[3]))], [0.4, 0.6]), data.y)) +
        logpdf(Normal(-2, 0.1), v[1]) + logpdf(Normal(2, 0.1), v[2]) +
        logpdf(Normal(), v[3])
    h = 1e-6
    expected_gradient = [(reference(q .+ h .* basis) - reference(q .- h .* basis)) / 2h
                         for basis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])]
    @test gradient ≈ expected_gradient atol=1e-5
    mixture = MixtureModel([Normal(-2.0, 0.3), Normal(2.0, 0.3)], [0.4, 0.6])
    pointwise = brm_execute(descriptor, :pointwise_loglik;
        problem, draws=q, seed=20260913)
    @test pointwise.y_likelihood ≈ logpdf.(mixture, data.y) atol=1e-10
    predictions = brm_execute(descriptor, :predict;
        problem, draws=repeat(q, 1, 4096), seed=20260913)
    draws = vec(predictions.y_gen)
    @test length(draws) == 4 * 4096
    @test mean(draws) ≈ mean(mixture) atol=0.1

    plan = generative_plan(builder, data; mod=@__MODULE__)
    declaration = only(d for d in plan.declarations if d.target === :y)
    @test declaration.role === :observation
    @test declaration.draw === :y_gen
    @test declaration.family isa Function
    @test startswith(string(nameof(declaration.family)), "brm_mixture_")

    sb = SBBRMI(builder(data); mod=@__MODULE__)
    replayed = reprocess(sb, (; y=[0.1, 0.2]))
    @test replayed.data[:y_mixture_weights] == [0.4, 0.6]
    @test stan_code(replayed) == stan_code(sb)
end

@testset "discrete mixtures lower exactly on both backends" begin
    poisson_data = (; y=[0, 1, 3, 5, 2])
    poisson_builder = @brm begin
        lambda1 ~ Exponential(1)
        lambda2 ~ Exponential(1)
        y ~ MixtureModel([Poisson(lambda1), Poisson(lambda2)], [0.3, 0.7])
    end
    poisson_backend = TuringBRMI(poisson_builder(poisson_data))
    poisson_parameters = (; lambda1=1.5, lambda2=4.0)
    poisson_mixture = MixtureModel([Poisson(1.5), Poisson(4.0)], [0.3, 0.7])
    poisson_expected = logpdf.(poisson_mixture, poisson_data.y)
    @test Turing.loglikelihood(poisson_backend.model, poisson_parameters) ≈
        sum(poisson_expected)
    @test turing_pointwise_loglikelihoods(
        poisson_backend, poisson_parameters).y ≈ poisson_expected
    @test length(turing_posterior_predictive(
        Xoshiro(42), poisson_backend, poisson_parameters).y) == length(poisson_data.y)
    poisson_code = stan_code(SBBRMI(poisson_builder(poisson_data)))
    @test occursin("poisson_lpmf", poisson_code)
    @test occursin("poisson_rng", poisson_code)
    @test StanBlocks.stanc_check(poisson_code; warn_pedantic=false).ok
    poisson_descriptor = brm_descriptor(
        poisson_builder, poisson_data; mod=@__MODULE__, highlights=())
    cache = joinpath(tempdir(), "brm-mixture-response")
    isdir(cache) || mkpath(cache)
    poisson_problem = brm_execute(poisson_descriptor, :instantiate;
        path=joinpath(cache, string(poisson_descriptor.id) * ".stan"))
    @test LogDensityProblems.dimension(poisson_problem) == 2
    # BridgeStan draws are unconstrained: positive rates live in log space.
    poisson_draws_unc = log.([1.5, 4.0])
    poisson_pointwise = brm_execute(poisson_descriptor, :pointwise_loglik;
        problem=poisson_problem, draws=poisson_draws_unc, seed=20260913)
    @test poisson_pointwise.y_likelihood ≈ poisson_expected atol=1e-10
    poisson_predictions = brm_execute(poisson_descriptor, :predict;
        problem=poisson_problem, draws=repeat(poisson_draws_unc, 1, 4096),
        seed=20260913)
    poisson_draws = vec(poisson_predictions.y_gen)
    @test mean(poisson_draws) ≈ mean(poisson_mixture) atol=0.15
    @test all(>=(0), poisson_draws)

    binomial_data = (; y=[1, 8, 3, 9], n=[10, 10, 10, 10])
    binomial_builder = @brm begin
        p1 ~ Beta(2, 2)
        p2 ~ Beta(2, 2)
        y ~ MixtureModel([Binomial(n, p1), Binomial(n, p2)], [0.5, 0.5])
    end
    binomial_backend = TuringBRMI(binomial_builder(binomial_data))
    binomial_parameters = (; p1=0.2, p2=0.8)
    binomial_mixture = MixtureModel(
        [Binomial(10, 0.2), Binomial(10, 0.8)], [0.5, 0.5])
    binomial_expected = logpdf.(binomial_mixture, binomial_data.y)
    @test Turing.loglikelihood(binomial_backend.model, binomial_parameters) ≈
        sum(binomial_expected)
    @test turing_pointwise_loglikelihoods(
        binomial_backend, binomial_parameters).y ≈ binomial_expected
    binomial_code = stan_code(SBBRMI(binomial_builder(binomial_data)))
    @test occursin("binomial_lpmf", binomial_code)
    @test occursin("brm_mixture_rows_int", binomial_code)
    @test StanBlocks.stanc_check(binomial_code; warn_pedantic=false).ok
    binomial_descriptor = brm_descriptor(
        binomial_builder, binomial_data; mod=@__MODULE__, highlights=())
    binomial_problem = brm_execute(binomial_descriptor, :instantiate;
        path=joinpath(cache, string(binomial_descriptor.id) * ".stan"))
    @test LogDensityProblems.dimension(binomial_problem) == 2
    # BridgeStan draws are unconstrained: probabilities live in logit space.
    binomial_draws_unc = logit.([0.2, 0.8])
    binomial_pointwise = brm_execute(binomial_descriptor, :pointwise_loglik;
        problem=binomial_problem, draws=binomial_draws_unc, seed=20260913)
    @test binomial_pointwise.y_likelihood ≈ binomial_expected atol=1e-10
    binomial_predictions = brm_execute(binomial_descriptor, :predict;
        problem=binomial_problem, draws=repeat(binomial_draws_unc, 1, 4096),
        seed=20260913)
    @test mean(vec(binomial_predictions.y_gen)) ≈ mean(binomial_mixture) atol=0.2

    bernoulli_builder = @brm begin
        p1 ~ Beta(2, 2)
        p2 ~ Beta(2, 2)
        y ~ MixtureModel([Bernoulli(p1), Bernoulli(p2)], [0.5, 0.5])
    end
    bernoulli_data = (; y=[0, 1, 1, 0])
    bernoulli_backend = TuringBRMI(bernoulli_builder(bernoulli_data))
    bernoulli_parameters = (; p1=0.2, p2=0.9)
    bernoulli_mixture = MixtureModel([Bernoulli(0.2), Bernoulli(0.9)], [0.5, 0.5])
    bernoulli_expected = logpdf.(bernoulli_mixture, bernoulli_data.y)
    @test Turing.loglikelihood(bernoulli_backend.model, bernoulli_parameters) ≈
        sum(bernoulli_expected)
    @test turing_pointwise_loglikelihoods(
        bernoulli_backend, bernoulli_parameters).y ≈ bernoulli_expected
    bernoulli_code = stan_code(SBBRMI(bernoulli_builder(bernoulli_data)))
    @test occursin("bernoulli_lpmf", bernoulli_code)
    @test StanBlocks.stanc_check(bernoulli_code; warn_pedantic=false).ok
    # Float 0/1 outcomes coerce to the `int` response Stan needs.
    bernoulli_float_code = stan_code(
        SBBRMI(bernoulli_builder((; y=[0.0, 1.0, 1.0, 0.0]))))
    @test StanBlocks.stanc_check(bernoulli_float_code; warn_pedantic=false).ok
end

@testset "simplex-modeled mixture weights" begin
    data = (; y=[-1.2, 0.9, -0.8])
    builder = @brm begin
        w ~ Dirichlet(2, 1.0)
        y ~ MixtureModel([Normal(-1.0, 0.5), Normal(1.0, 0.5)], w)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; w=[0.4, 0.6])
    expected = logpdf.(MixtureModel(
        [Normal(-1.0, 0.5), Normal(1.0, 0.5)], [0.4, 0.6]), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test StanBlocks.stanc_check(
        stan_code(SBBRMI(builder(data))); warn_pedantic=false).ok
end

@testset "mixture contract rejects incompatible inputs loudly" begin
    @test_throws "must share one family" SBBRMI((@brm begin
        y ~ MixtureModel([Normal(0, 1), Cauchy(0, 1)], [0.5, 0.5])
    end)((; y=[0.1])))
    @test_throws "is not scalar" SBBRMI((@brm begin
        y ~ MixtureModel([MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0])], [1.0])
    end)((; y=[[0.1, 0.2]])))
    @test_throws "simplex" SBBRMI((@brm begin
        y ~ MixtureModel([Categorical([0.5, 0.5])], [1.0])
    end)((; y=[1, 2])))
    @test_throws "parameter-dependent" SBBRMI((@brm begin
        y ~ MixtureModel([Uniform(0, 1), Uniform(0, 1)], [0.5, 0.5])
    end)((; y=[0.5])))
    @test_throws "has no Stan translation" SBBRMI((@brm begin
        y ~ MixtureModel([ZeroInflatedPoisson(1.0, 0.2)], [1.0])
    end)((; y=[0, 1])))
    @test_throws "has no Stan translation" SBBRMI((@brm begin
        y ~ MixtureModel([LocationScale(0, 1, TDist(3))], [1.0])
    end)((; y=[0.1])))
    @test_throws "must sum to 1" SBBRMI((@brm begin
        y ~ MixtureModel([Normal(0, 1), Normal(1, 1)], [0.5, 0.6])
    end)((; y=[0.1])))
    @test_throws "2 components but 3 weights" SBBRMI((@brm begin
        y ~ MixtureModel([Normal(0, 1), Normal(1, 1)], [0.3, 0.3, 0.4])
    end)((; y=[0.1])))
    @test_throws "must be nonnegative" SBBRMI((@brm begin
        y ~ MixtureModel([Normal(0, 1), Normal(1, 1)], [-0.5, 1.5])
    end)((; y=[0.1])))
    @test_throws "at least one component" SBBRMI((@brm begin
        y ~ MixtureModel([], Float64[])
    end)((; y=[0.1])))
    @test_throws "identical trial-count" SBBRMI((@brm begin
        y ~ MixtureModel([Binomial(n1, 0.2), Binomial(n2, 0.8)], [0.5, 0.5])
    end)((; y=[1, 8], n1=[10, 10], n2=[10, 10])))
    @test_throws "only integer values" SBBRMI((@brm begin
        y ~ MixtureModel([Poisson(1.0)], [1.0])
    end)((; y=[1.5])))
    @test_throws "must be nonnegative" SBBRMI((@brm begin
        y ~ MixtureModel([Poisson(1.0)], [1.0])
    end)((; y=[-1])))
    @test_throws "Dirichlet" SBBRMI((@brm begin
        w ~ Normal(0, 1)
        y ~ MixtureModel([Normal(0, 1), Normal(1, 1)], w)
    end)((; y=[0.1])))
    @test_throws "AnalyticWeights" SBBRMI((@brm begin
        y ~ weighted(MixtureModel([Normal(0, 1), Normal(1, 1)], [0.5, 0.5]),
                     aweights(w))
    end)((; y=[0.1, 0.2], w=[1.0, 1.0])))
    # Degenerate single-component and objective-weighted mixtures transpile.
    k1 = @brm begin
        y ~ MixtureModel([Normal(0, 1)], [1.0])
    end
    @test StanBlocks.stanc_check(
        stan_code(SBBRMI(k1((; y=[0.1, -0.2])))); warn_pedantic=false).ok
    weighted_builder = @brm begin
        y ~ weighted(MixtureModel([Normal(0, 1), Normal(3, 1)], [0.5, 0.5]),
                     fweights(w))
    end
    @test StanBlocks.stanc_check(stan_code(
        SBBRMI(weighted_builder((; y=[0.1, 2.9], w=[1.0, 2.0]))));
        warn_pedantic=false).ok
end
