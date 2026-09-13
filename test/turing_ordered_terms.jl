using Test
using BayesianRegressionModels
using Distributions
using Turing

const BRM = BayesianRegressionModels

@testset "integrated monotonic and interval predictor terms" begin
    data = (;
        grade=[1, 2, 3, 4, 1, 2, 3, 4],
        x=[0.5, 0.8, 1.2, 0.4, 1.5, 0.7, 1.1, 0.6],
        limit=[0.5, 0.5, 0.5, 0.4, 0.5, 0.5, 0.5, 0.5],
        time=collect(1.0:8.0),
        y=zeros(8),
    )
    mo_builder = @brm begin
        simplex(mu, mo(grade)) ~ Dirichlet(2)
        mu ~ 1 + mo(grade)
        y ~ Normal(mu, 1)
    end
    mo1_builder = @brm begin
        simplex(mu, mo1(grade)) ~ Dirichlet([1.0, 2.0, 3.0])
        mu ~ 1 + mo1(grade)
        y ~ Normal(mu, 1)
    end
    interval_builder = @brm begin
        latent(mu, interval_censored(x)) ~ Normal(1, 2)
        mu ~ 1 + interval_censored(x; upper=limit, lower=0.0)
        y ~ Normal(mu, 1)
    end

    backends = map(builder -> TuringBRMI(builder(data)),
                   (mo_builder, mo1_builder, interval_builder))
    @test all(backend -> length(only(backend.plan.predictors).terms) == 1,
              backends)
    @test all(backend -> occursin("term_mu_1",
        string(BRM.turing_model_source(backend))), backends)

    mo_term = only(only(backends[1].plan.predictors).terms)
    mo1_term = only(only(backends[2].plan.predictors).terms)
    interval_term = only(only(backends[3].plan.predictors).terms)
    @test mo_term.state.alpha == fill(2.0, 3)
    @test mo1_term.state.alpha == [1.0, 2.0, 3.0]
    @test BRM.getargs(interval_term.state.latent_prior) == (1, 2)

    mo_values = (; simplex_incr=[0.2, 0.3, 0.5], beta=-0.7)
    mo_result = Turing.generated_quantities(
        BRM._brm_turing_term_model(mo_term, 8), mo_values)
    contrast = cumsum(vcat(0.0, mo_values.simplex_incr))[mo_term.state.idx]
    @test mo_result.effect ≈ mo_values.beta .* contrast
    @test Turing.logjoint(BRM._brm_turing_term_model(mo_term, 8), mo_values) ≈
        logpdf(Dirichlet(mo_term.state.alpha), mo_values.simplex_incr) +
        logpdf(Normal(), mo_values.beta)

    interval_values = (;
        x_interval=fill(0.25, length(interval_term.state.Jinterval)), beta=0.4)
    interval_model = BRM._brm_turing_term_model(interval_term, 8)
    interval_result = Turing.generated_quantities(interval_model, interval_values)
    @test interval_result.effect ≈ interval_values.beta .* interval_result.x_true
    interval_oracle = logpdf(Normal(), interval_values.beta) + sum(eachindex(
            interval_values.x_interval)) do i
        logpdf(truncated(Normal(1, 2);
            lower=interval_term.state.x_lower[i],
            upper=interval_term.state.x_upper[i]), interval_values.x_interval[i])
    end
    @test Turing.logjoint(interval_model, interval_values) ≈ interval_oracle

    replay_data = (;
        grade=[4, 1, 2, 4], x=[0.5, 0.9, 0.5, 1.1],
        limit=[0.5, 0.5, 0.5, 0.5], y=zeros(4))
    replayed = map(backend -> reprocess(backend, replay_data), backends)
    @test only(only(replayed[1].plan.predictors).terms).state.levels ==
          mo_term.state.levels
    @test only(only(replayed[2].plan.predictors).terms).state.levels ==
          mo1_term.state.levels
    @test only(only(replayed[3].plan.predictors).terms).state.nobs == 4
end


@testset "integrated AR and differenced-AR terms" begin
    data = (; time=collect(1.0:8.0), y=zeros(8))
    ar_builder = @brm begin
        mu ~ 1 + ar(time)
        y ~ Normal(mu, 1)
    end
    dar_builder = @brm begin
        ar(mu, dar(time)) ~ Uniform(0.1, 0.9)
        sd(mu, dar(time)) ~ Exponential(0.5)
        mu ~ 1 + dar(time)
        y ~ Normal(mu, 1)
    end
    ar_backend = TuringBRMI(ar_builder(data))
    dar_backend = TuringBRMI(dar_builder(data))
    ar_term = only(only(ar_backend.plan.predictors).terms)
    dar_term = only(only(dar_backend.plan.predictors).terms)
    @test BRM.getf(dar_term.state.ar_prior) <: Uniform
    @test BRM.getf(dar_term.state.sd_prior) <: Exponential

    ar_values = (; phi_raw=0.3, epsilon=collect(range(-0.2, 0.2; length=8)),
                 beta=0.4)
    ar_model = BRM._brm_turing_term_model(ar_term, 8)
    ar_result = Turing.generated_quantities(ar_model, ar_values)
    @test ar_result.effect ≈ ar_values.beta .* ar_result.path
    @test Turing.logjoint(ar_model, ar_values) ≈
        logpdf(Normal(), ar_values.phi_raw) +
        sum(logpdf.(Normal(), ar_values.epsilon)) + logpdf(Normal(), ar_values.beta)

    dar_values = (; beta=0.4, sigma=0.3, z=fill(0.2, 7))
    dar_model = BRM._brm_turing_term_model(dar_term, 8)
    dar_result = Turing.generated_quantities(dar_model, dar_values)
    @test dar_result.effect == dar_result.path
    @test Turing.logjoint(dar_model, dar_values) ≈
        logpdf(Uniform(0.1, 0.9), dar_values.beta) +
        logpdf(Exponential(0.5), dar_values.sigma) +
        sum(logpdf.(Normal(), dar_values.z))

    replayed = reprocess(dar_backend, (; time=collect(2.0:2.0:16.0), y=zeros(8)))
    @test only(only(replayed.plan.predictors).terms).state.time ==
          collect(2.0:2.0:16.0)
end
