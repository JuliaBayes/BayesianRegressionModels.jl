using Test
using BayesianRegressionModels
using Distributions
using LinearAlgebra
using Turing

const BRM = BayesianRegressionModels

@testset "integrated exact GP and HSGP variants" begin
    x = collect(range(-1.5, 1.5; length=12))
    z = sin.(x) .+ 0.1 .* x
    data = (; x, z, g=repeat(1:3; inner=4), y=zeros(12))
    gp_iso = @brm begin
        rho_rate ~ Exponential(1)
        sigma_loc ~ Normal(0, 1)
        length_scale(mu, gp(x)) ~ Exponential(rho_rate)
        sd(mu, gp(x)) ~ Normal(sigma_loc, 1)
        mu ~ 1 + gp(x; jitter=1e-8)
        y ~ Normal(mu, 1)
    end
    gp_aniso = @brm begin
        mu ~ 1 + gp(x, z; iso=false)
        y ~ Normal(mu, 1)
    end
    gp_periodic = @brm begin
        mu ~ 1 + gp(x; cov=:periodic, period=2.5)
        y ~ Normal(mu, 1)
    end
    hsgp_iso = @brm begin
        mu ~ 1 + hsgp(x; k=5, c=1.5)
        y ~ Normal(mu, 1)
    end
    hsgp_aniso = @brm begin
        mu ~ 1 + hsgp(x, z; k=(4, 3), c=(1.5, 1.7), iso=false)
        y ~ Normal(mu, 1)
    end
    hsgp_periodic = @brm begin
        mu ~ 1 + hsgp(x; k=4, cov=:periodic, period=2.5)
        y ~ Normal(mu, 1)
    end
    hsgp_by = @brm begin
        mu ~ 1 + hsgp(x; k=4, by=g)
        y ~ Normal(mu, 1)
    end
    builders = (gp_iso, gp_aniso, gp_periodic, hsgp_iso, hsgp_aniso,
                hsgp_periodic, hsgp_by)
    backends = map(builder -> TuringBRMI(builder(data)), builders)
    @test all(backend -> length(only(backend.plan.predictors).terms) == 1,
              backends)
    @test all(backend -> occursin("term_mu_1",
        string(BRM.turing_model_source(backend))), backends)
    @test occursin("rho_rate", string(BRM.turing_model_source(backends[1])))
    @test occursin("sigma_loc", string(BRM.turing_model_source(backends[1])))

    gp_term = only(only(backends[1].plan.predictors).terms)
    @test gp_term.state.cov === :exp_quad
    @test gp_term.state.iso
    @test BRM.getf(gp_term.state.rho_prior) <: Exponential
    gp_values = (; rho=0.8, sigma=0.7,
                 z=collect(range(-0.2, 0.2; length=12)))
    gp_model = BRM._brm_turing_term_model(gp_term, 12,
        (; rho=Exponential(2), sigma=Normal(-0.2, 1)))
    gp_result = Turing.generated_quantities(gp_model, gp_values)
    turing_ext = Base.get_extension(BRM, :BayesianRegressionModelsTuringExt)
    K = turing_ext._brm_gp_covariance(
        gp_term.state, gp_values.sigma, gp_values.rho)
    @test gp_result.effect ≈ cholesky(Symmetric(K)).L * gp_values.z
    @test Turing.logjoint(gp_model, gp_values) ≈
        logpdf(Exponential(2), gp_values.rho) +
        logpdf(Normal(-0.2, 1), gp_values.sigma) +
        sum(logpdf.(Normal(), gp_values.z))

    hsgp_term = only(only(backends[4].plan.predictors).terms)
    hvalues = (; rho=max(1.1, hsgp_term.state.rho_lower + 0.1), sigma=0.6,
               beta_raw=fill(0.15, size(hsgp_term.state.PHI, 2)))
    hmodel = BRM._brm_turing_term_model(hsgp_term, 12)
    hresult = Turing.generated_quantities(hmodel, hvalues)
    @test hresult.effect ≈ hsgp_term.state.PHI *
        (hresult.sqrt_spd .* hvalues.beta_raw)

    replay_data = (; x=data.x .+ 0.2, z=data.z .- 0.1,
                   g=data.g, y=zeros(12))
    replayed = map(backend -> reprocess(backend, replay_data), backends)
    @test only(only(replayed[4].plan.predictors).terms).state.fits ==
          hsgp_term.state.fits
    @test size(only(only(replayed[5].plan.predictors).terms).state.PHI) == (12, 12)
    @test only(only(replayed[7].plan.predictors).terms).state.by.levels == [1, 2, 3]
end

@testset "explicit HSGP replay rejects extrapolation" begin
    data = (; x=collect(range(-1, 1; length=12)), y=zeros(12))
    backend = TuringBRMI((@brm begin
        mu ~ 1 + hsgp(x; k=4, domain=(-2.0, 2.0))
        y ~ Normal(mu, 1)
    end)(data))
    @test_throws "outside its fixed domain" reprocess(
        backend, (; x=data.x .+ 4, y=data.y))
end

@testset "model-derived HSGP axis and frozen domain" begin
    data = (; t=collect(range(-1, 1; length=12)),
            c_obs=collect(range(-0.8, 0.8; length=12)), y=zeros(12))
    builder = @brm begin
        x ~ 1 + t
        c_obs ~ Normal(x, 0.2)
        mu ~ 1 + hsgp(x; k=4, domain=(-3.0, 3.0),
                      orthogonal_to=:linear)
        y ~ Normal(mu, 1)
    end
    backend = TuringBRMI(builder(data))
    plans = hasproperty(backend.plan, :plans) ? backend.plan.plans : (backend.plan,)
    terms = [term for plan in plans for predictor in plan.predictors
             for term in predictor.terms]
    term = only(filter(term -> term.callable === hsgp, terms))
    @test term.state.latent
    @test term.state.fits == ((0.0, 3.0),)
    @test term.dependencies == (:x,)
    latent_model = BRM._brm_turing_term_model(term, 12,
        (; rho=LogNormal(0, 1), sigma=LogNormal(0, 1)), (; x=data.c_obs))
    latent_values = (; rho=1.2, sigma=0.7, beta_raw=fill(0.1, 4))
    latent_result = Turing.generated_quantities(latent_model, latent_values)
    @test length(latent_result.effect) == 12
    @test abs(sum(latent_result.effect)) < 1e-10
    @test abs(dot(data.c_obs .- sum(data.c_obs) / 12,
                  latent_result.effect)) < 1e-10
    replay = reprocess(backend, merge(data, (; t=data.t .+ 0.1)))
    replay_plans = hasproperty(replay.plan, :plans) ? replay.plan.plans : (replay.plan,)
    replay_terms = [term for plan in replay_plans for predictor in plan.predictors
                    for term in predictor.terms]
    replay_term = only(filter(term -> term.callable === hsgp, replay_terms))
    @test replay_term.state.fits == term.state.fits
end
