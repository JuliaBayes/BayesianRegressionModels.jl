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

    # p>1 per_threshold packs stage-major: flat[(k-1)*p+j] is the stage-k,
    # term-j coefficient, matching SB-Stan's `array[n_cut] vector[n_terms]`
    # runtime layout (snag brm-threshold-et-0f516c0c). Every flat entry is
    # distinct, so a term-major read cannot agree with this ref by accident.
    p2_data = (; x=data.x, z1=[0.0, 1.0, 0.5, -1.0],
        z2=[1.0, 0.0, -0.5, 2.0], y=data.y)
    p2_builder = @brm begin
        eta ~ 0 + x
        y ~ Ordinal(StoppingRatio(), ProbitLink(), eta; per_threshold=(z1, z2))
    end
    p2_backend = TuringBRMI(p2_builder(p2_data))
    p2_flat = [1.0, 2.0, 3.0, 4.0]
    p2_parameters = (; beta_pop=[0.4], y_thresholds=[-0.5, 0.8],
        y_threshold_beta=p2_flat)
    p2_stage_eta(x, a, b) =
        [0.4x + p2_flat[(k - 1) * 2 + 1] * a + p2_flat[(k - 1) * 2 + 2] * b
         for k in 1:2]
    p2_expected = [logpdf(Ordinal(StoppingRatio(), ProbitLink(),
                        p2_stage_eta(x, a, b), p2_parameters.y_thresholds), y)
                   for (x, a, b, y) in
                       zip(p2_data.x, p2_data.z1, p2_data.z2, [1, 3, 2, 1])]
    @test Turing.loglikelihood(p2_backend.model, p2_parameters) ≈ sum(p2_expected)
    @test turing_pointwise_loglikelihoods(p2_backend, p2_parameters).y ≈ p2_expected
    @test StanBlocks.stanc_check(stan_code(SBBRMI(p2_builder(p2_data)))).ok
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

@testset "gamma scale convention is exact Julia Distributions semantics" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[1.2, 0.8, 1.1])
    builder = @brm begin
        shape ~ LogNormal(0, 0.3)
        eta ~ 1 + x
        y ~ Gamma(shape, exp(eta))
    end
    backend = TuringBRMI(builder(data))
    parameters = (; shape=1.3, beta_pop=[0.1, 0.2])
    scales = exp.(0.1 .+ 0.2 .* data.x)
    expected = logpdf.(Gamma.(1.3, scales), data.y)
    # A rate reading of the second argument would give a different density;
    # pin the scale convention against the swapped-parameterization value.
    @test !(sum(expected) ≈ sum(logpdf.(Gamma.(1.3, inv.(scales)), data.y)))
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 1.3) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(>(0), predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[0.9, 1.4]))
    @test replayed.plan.response == [0.9, 1.4]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "weibull shape/scale order matches Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[0.2, 1.5, 3.8])
    builder = @brm begin
        shape ~ LogNormal(0, 0.3)
        eta ~ 1 + x
        y ~ Weibull(shape, exp(eta))
    end
    backend = TuringBRMI(builder(data))
    parameters = (; shape=1.7, beta_pop=[0.1, 0.2])
    scales = exp.(0.1 .+ 0.2 .* data.x)
    expected = logpdf.(Weibull.(1.7, scales), data.y)
    # A swapped shape/scale reading would give a different density; pin it.
    @test !(sum(expected) ≈ sum(logpdf.(Weibull.(scales, 1.7), data.y)))
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 1.7) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(>(0), predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[1.9, 0.4]))
    @test replayed.plan.response == [1.9, 0.4]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "wald mean/shape order matches Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[0.5, 1.2, 2.5])
    builder = @brm begin
        lambda ~ LogNormal(0, 0.3)
        eta ~ 1 + x
        y ~ InverseGaussian(exp(eta), lambda)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; lambda=3.0, beta_pop=[0.1, 0.2])
    mus = exp.(0.1 .+ 0.2 .* data.x)
    expected = logpdf.(InverseGaussian.(mus, 3.0), data.y)
    # A swapped mean/shape reading would give a different density; pin it.
    @test !(sum(expected) ≈ sum(logpdf.(InverseGaussian.(3.0, mus), data.y)))
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 3.0) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(>(0), predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[1.1, 0.7]))
    @test replayed.plan.response == [1.1, 0.7]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "inversegamma shape/scale order matches Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[0.3, 1.4, 5.0])
    builder = @brm begin
        shape ~ LogNormal(0, 0.3)
        eta ~ 1 + x
        y ~ InverseGamma(shape, exp(eta))
    end
    backend = TuringBRMI(builder(data))
    parameters = (; shape=3.2, beta_pop=[0.1, 0.2])
    scales = exp.(0.1 .+ 0.2 .* data.x)
    expected = logpdf.(InverseGamma.(3.2, scales), data.y)
    @test !(sum(expected) ≈ sum(logpdf.(InverseGamma.(scales, 3.2), data.y)))
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 3.2) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(>(0), predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[1.9, 0.4]))
    @test replayed.plan.response == [1.9, 0.4]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "pareto shape/scale order matches Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[1.0, 1.8, 4.5])
    builder = @brm begin
        shape ~ LogNormal(0, 0.3)
        eta ~ 1 + x
        y ~ Pareto(shape, exp(eta))
    end
    backend = TuringBRMI(builder(data))
    parameters = (; shape=2.2, beta_pop=[0.1, 0.2])
    scales = exp.(0.1 .+ 0.2 .* data.x)
    expected = logpdf.(Pareto.(2.2, scales), data.y)
    # Distributions takes (shape, scale); Stan takes (minimum, shape) — pin
    # the Julia order against the swapped reading.
    @test !(sum(expected) ≈ sum(logpdf.(Pareto.(scales, 2.2), data.y)))
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 2.2) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(>(0), predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[2.1, 1.2]))
    @test replayed.plan.response == [2.1, 1.2]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "gumbel location/scale matches Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[-0.8, 0.4, 2.3])
    builder = @brm begin
        sigma ~ LogNormal(0, 0.3)
        mu ~ 1 + x
        y ~ Gumbel(mu, sigma)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; sigma=1.1, beta_pop=[0.2, 0.3])
    locs = 0.2 .+ 0.3 .* data.x
    expected = logpdf.(Gumbel.(locs, 1.1), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 1.1) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(isfinite, predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[0.1, -0.3]))
    @test replayed.plan.response == [0.1, -0.3]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "chisq dof matches Julia Distributions exactly" begin
    data = (; y=[0.2, 1.3, 4.2])
    builder = @brm begin
        nu ~ LogNormal(0, 0.3)
        y ~ Chisq(nu)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; nu=3.5)
    expected = logpdf.(Chisq(3.5), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 3.5)
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(>(0), predicted)
    replayed = reprocess(backend, (; y=[1.1, 2.2]))
    @test replayed.plan.response == [1.1, 2.2]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "rayleigh scale matches Julia Distributions exactly" begin
    data = (; y=[0.2, 1.1, 3.2])
    builder = @brm begin
        sigma ~ LogNormal(0, 0.3)
        y ~ Rayleigh(sigma)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; sigma=1.5)
    expected = logpdf.(Rayleigh(1.5), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 1.5)
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(>(0), predicted)
    replayed = reprocess(backend, (; y=[0.9, 2.1]))
    @test replayed.plan.response == [0.9, 2.1]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "frechet shape/scale order matches Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[0.4, 1.5, 4.0])
    builder = @brm begin
        shape ~ LogNormal(0, 0.3)
        eta ~ 1 + x
        y ~ Frechet(shape, exp(eta))
    end
    backend = TuringBRMI(builder(data))
    parameters = (; shape=2.2, beta_pop=[0.1, 0.2])
    scales = exp.(0.1 .+ 0.2 .* data.x)
    expected = logpdf.(Frechet.(2.2, scales), data.y)
    @test !(sum(expected) ≈ sum(logpdf.(Frechet.(scales, 2.2), data.y)))
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 2.2) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(>(0), predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[1.9, 0.4]))
    @test replayed.plan.response == [1.9, 0.4]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "normalcanon natural parameters match Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[-0.8, 0.2, 1.4])
    builder = @brm begin
        lambda ~ LogNormal(0, 0.3)
        eta ~ 1 + x
        y ~ NormalCanon(eta, lambda)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; lambda=1.1, beta_pop=[0.2, 0.3])
    etas = 0.2 .+ 0.3 .* data.x
    expected = logpdf.(NormalCanon.(etas, 1.1), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 1.1) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(isfinite, predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[0.1, -0.3]))
    @test replayed.plan.response == [0.1, -0.3]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "skewnormal location/scale/shape match Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[-1.0, 0.3, 2.2])
    builder = @brm begin
        sigma ~ LogNormal(0, 0.3)
        alpha ~ Normal(0, 2)
        mu ~ 1 + x
        y ~ SkewNormal(mu, sigma, alpha)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; sigma=1.1, alpha=2.0, beta_pop=[0.2, 0.3])
    locs = 0.2 .+ 0.3 .* data.x
    expected = logpdf.(SkewNormal.(locs, 1.1, 2.0), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 1.1) + logpdf(Normal(0, 2), 2.0) +
        sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(isfinite, predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[0.1, -0.3]))
    @test replayed.plan.response == [0.1, -0.3]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "erlang integer shape with fitted scale is exact" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[0.3, 1.4, 4.2])
    builder = @brm begin
        eta ~ 1 + x
        y ~ Erlang(3, exp(eta))
    end
    backend = TuringBRMI(builder(data))
    parameters = (; beta_pop=[0.1, 0.2])
    scales = exp.(0.1 .+ 0.2 .* data.x)
    expected = logpdf.(Erlang.(3, scales), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(>(0), predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[1.9, 0.4]))
    @test replayed.plan.response == [1.9, 0.4]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "standard arcsine likelihood is exact beta(1/2, 1/2) density" begin
    data = (; y=[0.1, 0.5, 0.9])
    builder = @brm begin
        y ~ Arcsine()
    end
    backend = TuringBRMI(builder(data))
    parameters = (;)
    expected = logpdf.(Arcsine(), data.y)
    # The standard form is exactly beta(1/2, 1/2).
    @test expected ≈ logpdf.(Beta(0.5, 0.5), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(y -> 0 < y < 1, predicted)
    replayed = reprocess(backend, (; y=[0.2, 0.8]))
    @test replayed.plan.response == [0.2, 0.8]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "exponential scale likelihood is exact" begin
    data = (; y=[0.2, 1.3, 3.1])
    builder = @brm begin
        theta ~ LogNormal(0, 0.3)
        y ~ Exponential(theta)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; theta=2.5)
    expected = logpdf.(Exponential(2.5), data.y)
    # Distributions takes scale; Stan takes rate — pin the Julia order
    # against the swapped reading.
    @test !(sum(expected) ≈ sum(logpdf.(Exponential.(inv(2.5)), data.y)))
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 2.5)
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(>(0), predicted)
    replayed = reprocess(backend, (; y=[1.1, 2.2]))
    @test replayed.plan.response == [1.1, 2.2]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "truncated weibull matches Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[0.6, 1.5, 2.8])
    builder = @brm begin
        shape ~ LogNormal(0, 0.3)
        eta ~ 1 + x
        y ~ truncated(Weibull(shape, exp(eta)); lower=0.5, upper=3.0)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; shape=1.7, beta_pop=[0.1, 0.2])
    scales = exp.(0.1 .+ 0.2 .* data.x)
    expected = logpdf.(truncated.(Weibull.(1.7, scales), 0.5, 3.0), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 1.7) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(y -> 0.5 <= y <= 3.0, predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[1.9, 0.6]))
    @test replayed.plan.response == [1.9, 0.6]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "censored weibull matches Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[0.5, 1.5, 3.0])
    builder = @brm begin
        shape ~ LogNormal(0, 0.3)
        eta ~ 1 + x
        y ~ censored(Weibull(shape, exp(eta)); lower=0.5, upper=3.0)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; shape=1.7, beta_pop=[0.1, 0.2])
    scales = exp.(0.1 .+ 0.2 .* data.x)
    expected = logpdf.(censored.(Weibull.(1.7, scales), 0.5, 3.0), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 1.7) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(y -> 0.5 <= y <= 3.0, predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[3.0, 0.5]))
    @test replayed.plan.response == [3.0, 0.5]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "interval-censored weibull matches Julia CDF evidence exactly" begin
    # `upper` is a Stan reserved keyword; the endpoint column is `y_upper`.
    data = (; x=[-1.0, 0.0, 1.0], y=[0.4, 1.0, 2.0], y_upper=[0.9, 1.8, 3.2])
    builder = @brm begin
        shape ~ LogNormal(0, 0.3)
        eta ~ 1 + x
        y ~ interval_censored(Weibull(shape, exp(eta)); upper=y_upper)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; shape=1.7, beta_pop=[0.1, 0.2])
    scales = exp.(0.1 .+ 0.2 .* data.x)
    dists = Weibull.(1.7, scales)
    expected = log.(cdf.(dists, data.y_upper) .- cdf.(dists, data.y))
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 1.7) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(>(0), predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[0.6, 1.2], y_upper=[1.0, 2.0]))
    @test replayed.plan.response == [0.6, 1.2]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "truncated lognormal matches Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[0.6, 1.5, 2.8])
    builder = @brm begin
        sigma ~ LogNormal(0, 0.3)
        mu ~ 1 + x
        y ~ truncated(LogNormal(mu, sigma); lower=0.5, upper=3.0)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; sigma=0.8, beta_pop=[0.2, 0.3])
    locs = 0.2 .+ 0.3 .* data.x
    expected = logpdf.(truncated.(LogNormal.(locs, 0.8), 0.5, 3.0), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 0.8) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(y -> 0.5 <= y <= 3.0, predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[1.9, 0.6]))
    @test replayed.plan.response == [1.9, 0.6]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "censored lognormal matches Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[0.5, 1.5, 3.0])
    builder = @brm begin
        sigma ~ LogNormal(0, 0.3)
        mu ~ 1 + x
        y ~ censored(LogNormal(mu, sigma); lower=0.5, upper=3.0)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; sigma=0.8, beta_pop=[0.2, 0.3])
    locs = 0.2 .+ 0.3 .* data.x
    expected = logpdf.(censored.(LogNormal.(locs, 0.8), 0.5, 3.0), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 0.8) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(y -> 0.5 <= y <= 3.0, predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[3.0, 0.5]))
    @test replayed.plan.response == [3.0, 0.5]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "interval-censored lognormal matches Julia CDF evidence exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[0.4, 1.0, 2.0], y_upper=[0.9, 1.8, 3.2])
    builder = @brm begin
        sigma ~ LogNormal(0, 0.3)
        mu ~ 1 + x
        y ~ interval_censored(LogNormal(mu, sigma); upper=y_upper)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; sigma=0.8, beta_pop=[0.2, 0.3])
    locs = 0.2 .+ 0.3 .* data.x
    dists = LogNormal.(locs, 0.8)
    expected = log.(cdf.(dists, data.y_upper) .- cdf.(dists, data.y))
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 0.8) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(>(0), predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[0.6, 1.2], y_upper=[1.0, 2.0]))
    @test replayed.plan.response == [0.6, 1.2]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "truncated exponential matches Julia Distributions exactly" begin
    data = (; y=[0.6, 1.5, 2.8])
    builder = @brm begin
        theta ~ LogNormal(0, 0.3)
        y ~ truncated(Exponential(theta); lower=0.5, upper=3.0)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; theta=2.5)
    expected = logpdf.(truncated.(Exponential(2.5), 0.5, 3.0), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 2.5)
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(y -> 0.5 <= y <= 3.0, predicted)
    replayed = reprocess(backend, (; y=[1.9, 0.6]))
    @test replayed.plan.response == [1.9, 0.6]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "censored exponential matches Julia Distributions exactly" begin
    data = (; y=[0.5, 1.5, 3.0])
    builder = @brm begin
        theta ~ LogNormal(0, 0.3)
        y ~ censored(Exponential(theta); lower=0.5, upper=3.0)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; theta=2.5)
    expected = logpdf.(censored.(Exponential(2.5), 0.5, 3.0), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 2.5)
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(y -> 0.5 <= y <= 3.0, predicted)
    replayed = reprocess(backend, (; y=[3.0, 0.5]))
    @test replayed.plan.response == [3.0, 0.5]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "interval-censored exponential matches Julia CDF evidence exactly" begin
    data = (; y=[0.4, 1.0, 2.0], y_upper=[0.9, 1.8, 3.2])
    builder = @brm begin
        theta ~ LogNormal(0, 0.3)
        y ~ interval_censored(Exponential(theta); upper=y_upper)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; theta=2.5)
    expected = log.(cdf.(Exponential(2.5), data.y_upper) .- cdf.(Exponential(2.5), data.y))
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 2.5)
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(>(0), predicted)
    replayed = reprocess(backend, (; y=[0.6, 1.2], y_upper=[1.0, 2.0]))
    @test replayed.plan.response == [0.6, 1.2]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "beta likelihood matches Julia Distributions exactly" begin
    data = (; y=[0.1, 0.5, 0.9])
    builder = @brm begin
        a ~ LogNormal(0, 0.3)
        b ~ LogNormal(0, 0.3)
        y ~ Beta(a, b)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; a=2.0, b=4.0)
    expected = logpdf.(Beta(2.0, 4.0), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 2.0) + logpdf(LogNormal(0, 0.3), 4.0)
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(y -> 0 < y < 1, predicted)
    replayed = reprocess(backend, (; y=[0.2, 0.8]))
    @test replayed.plan.response == [0.2, 0.8]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "uniform likelihood matches Julia Distributions exactly" begin
    data = (; y=[-0.5, 0.25, 1.5])
    builder = @brm begin
        y ~ Uniform(-1.0, 2.0)
    end
    backend = TuringBRMI(builder(data))
    parameters = (;)
    expected = logpdf.(Uniform(-1.0, 2.0), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(y -> -1.0 <= y <= 2.0, predicted)
    replayed = reprocess(backend, (; y=[0.0, 1.0]))
    @test replayed.plan.response == [0.0, 1.0]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "lognormal likelihood matches Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[0.3, 1.2, 4.0])
    builder = @brm begin
        sigma ~ LogNormal(0, 0.3)
        mu ~ 1 + x
        y ~ LogNormal(mu, sigma)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; sigma=0.8, beta_pop=[0.2, 0.3])
    locs = 0.2 .+ 0.3 .* data.x
    expected = logpdf.(LogNormal.(locs, 0.8), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 0.8) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(>(0), predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[0.9, 1.4]))
    @test replayed.plan.response == [0.9, 1.4]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "laplace likelihood matches Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[-1.1, 0.4, 2.2])
    builder = @brm begin
        theta ~ LogNormal(0, 0.3)
        mu ~ 1 + x
        y ~ Laplace(mu, theta)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; theta=1.3, beta_pop=[-0.1, 0.25])
    locs = -0.1 .+ 0.25 .* data.x
    expected = logpdf.(Laplace.(locs, 1.3), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 1.3) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(isfinite, predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[0.1, -0.3]))
    @test replayed.plan.response == [0.1, -0.3]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "tdist likelihood matches Julia Distributions exactly" begin
    data = (; y=[-0.9, 0.2, 1.7])
    builder = @brm begin
        nu ~ LogNormal(0, 0.3)
        y ~ TDist(nu)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; nu=4.5)
    expected = logpdf.(TDist(4.5), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 4.5)
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(isfinite, predicted)
    replayed = reprocess(backend, (; y=[0.1, -0.3]))
    @test replayed.plan.response == [0.1, -0.3]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "locationscale student-t matches Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[-0.8, 0.2, 1.7])
    builder = @brm begin
        sigma ~ LogNormal(0, 0.3)
        nu ~ LogNormal(0, 0.3)
        mu ~ 1 + x
        y ~ LocationScale(mu, sigma, TDist(nu))
    end
    backend = TuringBRMI(builder(data))
    parameters = (; sigma=1.2, nu=4.5, beta_pop=[0.1, -0.2])
    locs = 0.1 .- 0.2 .* data.x
    expected = logpdf.(LocationScale.(locs, 1.2, TDist(4.5)), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 1.2) + logpdf(LogNormal(0, 0.3), 4.5) +
        sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(isfinite, predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[0.1, -0.3]))
    @test replayed.plan.response == [0.1, -0.3]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "logistic likelihood matches Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[-1.3, 0.2, 2.1])
    builder = @brm begin
        sigma ~ LogNormal(0, 0.3)
        mu ~ 1 + x
        y ~ Logistic(mu, sigma)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; sigma=1.1, beta_pop=[0.2, 0.3])
    locs = 0.2 .+ 0.3 .* data.x
    expected = logpdf.(Logistic.(locs, 1.1), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 1.1) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(isfinite, predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[0.1, -0.3]))
    @test replayed.plan.response == [0.1, -0.3]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "cauchy likelihood matches Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[-1.2, 0.3, 2.1])
    builder = @brm begin
        sigma ~ LogNormal(0, 0.3)
        mu ~ 1 + x
        y ~ Cauchy(mu, sigma)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; sigma=1.4, beta_pop=[-0.2, 0.35])
    locs = -0.2 .+ 0.35 .* data.x
    expected = logpdf.(Cauchy.(locs, 1.4), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 1.4) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(isfinite, predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[0.1, -0.3]))
    @test replayed.plan.response == [0.1, -0.3]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "betabinomial likelihood matches Julia Distributions exactly" begin
    data = (; y=[1, 3, 4])
    builder = @brm begin
        a ~ LogNormal(0, 0.3)
        b ~ LogNormal(0, 0.3)
        y ~ BetaBinomial(5, a, b)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; a=2.0, b=3.0)
    expected = logpdf.(BetaBinomial(5, 2.0, 3.0), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 2.0) + logpdf(LogNormal(0, 0.3), 3.0)
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(y -> 0 <= y <= 5, predicted)
    replayed = reprocess(backend, (; y=[2, 1]))
    @test replayed.plan.response == [2, 1]
    # Fitted-shape Stan GQ emission needs the scalar-trials
    # beta_binomial_rng companion (StanBlocks 5e6fc48; snag
    # beta-binomial-fi-dda1fdf2, resolved).
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "negativebinomial likelihood matches Julia Distributions exactly" begin
    data = (; y=[0, 3, 7])
    builder = @brm begin
        r ~ LogNormal(0, 0.3)
        p ~ Beta(2, 2)
        y ~ NegativeBinomial(r, p)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; r=2.5, p=0.4)
    expected = logpdf.(NegativeBinomial(2.5, 0.4), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 2.5) + logpdf(Beta(2, 2), 0.4)
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(>=(0), predicted)
    replayed = reprocess(backend, (; y=[1, 2]))
    @test replayed.plan.response == [1, 2]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "skewdoubleexponential quantile likelihood is exact" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[-1.1, -0.2, 0.1])
    builder = @brm begin
        sigma ~ LogNormal(0, 0.3)
        mu ~ 1 + x
        y ~ BRM.SkewDoubleExponential(mu, sigma, 0.25)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; sigma=1.4, beta_pop=[0.3, 0.2])
    locs = 0.3 .+ 0.2 .* data.x
    expected = logpdf.(BRM.SkewDoubleExponential.(locs, 1.4, 0.25), data.y)
    # tau = 0.5 is exactly Laplace(mu, sigma); pin the tau = 0.25 asymmetry.
    @test cdf(BRM.SkewDoubleExponential(0.0, 1.4, 0.25), 0.0) ≈ 0.25
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 1.4) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(isfinite, predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[0.1, -0.3]))
    @test replayed.plan.response == [0.1, -0.3]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "skewedexponentialpower shape-1 quantile likelihood is exact" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[-0.8, 0.4, 1.7])
    builder = @brm begin
        sigma_sepd ~ LogNormal(0, 0.3)
        mu ~ 1 + x
        y ~ SkewedExponentialPower(mu, sigma_sepd, 1, 0.7)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; sigma_sepd=0.8, beta_pop=[-0.1, 0.2])
    locs = -0.1 .+ 0.2 .* data.x
    expected = logpdf.(SkewedExponentialPower.(locs, 0.8, 1, 0.7), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 0.8) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(isfinite, predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[0.1, -0.3]))
    @test replayed.plan.response == [0.1, -0.3]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "hurdlepoisson likelihood matches Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.5, 1.5], y=[0, 1, 2])
    builder = @brm begin
        log_lambda ~ 1 + x
        y ~ BRM.HurdlePoisson(exp(log_lambda), 0.35)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; beta_pop=[0.5, 0.3])
    rates = exp.(0.5 .+ 0.3 .* data.x)
    expected = logpdf.(BRM.HurdlePoisson.(rates, 0.35), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(>=(0), predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[1, 0]))
    @test replayed.plan.response == [1, 0]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "vonmises moving support matches Julia Distributions exactly" begin
    data = (; x=[-1.0, 0.0, 1.0], y=[0.3 - pi, 0.3, 0.3 + pi])
    builder = @brm begin
        kappa ~ LogNormal(0, 0.3)
        mu ~ 1 + x
        y ~ VonMises(mu, kappa)
    end
    backend = TuringBRMI(builder(data))
    parameters = (; kappa=1.7, beta_pop=[0.3, 0.1])
    mus = 0.3 .+ 0.1 .* data.x
    expected = logpdf.(VonMises.(mus, 1.7), data.y)
    @test all(isfinite, expected)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 1.7) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(insupport.(VonMises.(mus, 1.7), predicted))
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[0.35, 0.25]))
    @test replayed.plan.response == [0.35, 0.25]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "circularvonmises fixed interval matches Julia contract exactly" begin
    lo, hi = 0.0, 6.283185307179586
    data = (; x=[-1.0, 0.0, 1.0], y=[0.0, 2.0, 6.0])
    builder = @brm begin
        kappa ~ LogNormal(0, 0.3)
        mu ~ 1 + x
        y ~ BRM.CircularVonMises(mu, kappa; interval=(0.0, 6.283185307179586))
    end
    backend = TuringBRMI(builder(data))
    parameters = (; kappa=2.1, beta_pop=[7.0, 0.0])
    mus = fill(7.0, 3)
    dists = BRM.CircularVonMises.(mus, 2.1; interval=(lo, hi))
    expected = logpdf.(dists, data.y)
    @test all(isfinite, expected)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test Turing.logprior(backend.model, parameters) ≈
        logpdf(LogNormal(0, 0.3), 2.1) + sum(logpdf.(Normal(), parameters.beta_pop))
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(y -> lo <= y < hi, predicted)
    replayed = reprocess(backend, (; x=[0.5, -0.5], y=[1.0, 5.0]))
    @test replayed.plan.response == [1.0, 5.0]
    @test StanBlocks.stanc_check(stan_code(SBBRMI(builder(data)))).ok
end

@testset "truncatednormal legacy marker matches censored normal exactly" begin
    data = (; y=[-0.5, 0.2, 1.1])
    builder = @brm begin
        y ~ BRM.TruncatedNormal(0.1, 1.2, -1.0, 2.0)
    end
    backend = TuringBRMI(builder(data))
    parameters = (;)
    expected = logpdf.(censored(Normal(0.1, 1.2), -1.0, 2.0), data.y)
    @test Turing.loglikelihood(backend.model, parameters) ≈ sum(expected)
    @test turing_pointwise_loglikelihoods(backend, parameters).y ≈ expected
    predicted = turing_posterior_predictive(Xoshiro(42), backend, parameters).y
    @test length(predicted) == length(data.y)
    @test all(y -> -1.0 <= y <= 2.0, predicted)
    replayed = reprocess(backend, (; y=[0.0, 1.0]))
    @test replayed.plan.response == [0.0, 1.0]
    # Stan-side TruncatedNormal is biomarker-context-only; outside that
    # context the boundary is loud, not a transpile. Pin it so silent drift
    # fails here. (StanBlocks 74ed796d surfaces this as a transpile error —
    # TruncatedNormal has no `lpxf_expr` — rather than the old BRM-side
    # `biomarker_idxs` indexing failure.)
    @test_throws "TruncatedNormal is missing" stan_code(SBBRMI(builder(data)))
end
