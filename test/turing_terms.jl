using Test
using BayesianRegressionModels
using Distributions
using Turing

const BRM = BayesianRegressionModels

function prepared_term(builder, data, target, callable)
    brmi = builder(data)
    context = BRM._brm_backend_context(brmi)
    rhs = last(BRM.getargs(BRM.linear_predictor_op(brmi, target), 2))
    term = only(t for t in BRM._brm_additive_terms(rhs)
                if t isa BRM.ExprColumn && BRM.getf(t) === callable)
    BRM._brm_prepare_term(term, target, context), term
end

@testset "integrated TuringBRMI structured terms and replay" begin
    x = collect(range(-1.0, 1.0; length=12))
    z = x .^ 2 .+ 0.2 .* x
    data = (; x, z, y=zeros(12))
    s_builder = @brm begin
        sd(mu, s(x)) ~ Exponential(2)
        mu ~ 1 + s(x)
        y ~ Normal(mu, 1)
    end
    t2_builder = @brm begin
        sd(mu, t2(x, z), rr) ~ Exponential(3)
        mu ~ 1 + t2(x, z; k=(5, 5), basis=(:cr, :cr), full=false)
        y ~ Normal(mu, 1)
    end
    me_builder = @brm begin
        latent(mu, me(x)) ~ Normal(1, 3)
        mu ~ 1 + me(x, 0.2)
        y ~ Normal(mu, 1)
    end
    builders = (s_builder, t2_builder, me_builder)
    backends = map(builder -> TuringBRMI(builder(data)), builders)
    @test all(backend -> length(backend.plan.predictors) == 1, backends)
    @test all(backend -> length(only(backend.plan.predictors).terms) == 1,
              backends)
    @test all(backend -> !hasproperty(backend.model.args, :term_models), backends)
    @test all(backend -> occursin("term_mu_1", string(BRM.turing_model_source(backend))),
              backends)
    @test BRM.getf(only(only(backends[1].plan.predictors).terms).state.sd_prior) <:
          Exponential
    @test BRM.getf(only(only(backends[3].plan.predictors).terms).state.latent_prior) <:
          Normal

    replay_data = (; x=fill(0.25, 5), z=fill(-0.1, 5), y=zeros(5))
    replayed = map(backend -> reprocess(backend, replay_data), backends)
    @test all(backend -> length(backend.plan.response) == 5, replayed)
    @test size(only(only(replayed[1].plan.predictors).terms).state.Zpen, 1) == 5
    @test size(only(only(replayed[2].plan.predictors).terms).state.Zrr, 1) == 5
end

@testset "shared fitted s/t2/me preparation" begin
    x = collect(range(-1.0, 1.0; length=12))
    z = x .^ 2 .+ 0.2 .* x
    data = (; x, z, y=zeros(12))

    s_builder = @brm begin mu ~ 1 + s(x); y ~ Normal(mu, 1) end
    spline, spline_expr = prepared_term(s_builder, data, :mu, s)
    @test size(spline.state.Xnull) == (12, 2)
    @test size(spline.state.Zpen) == (12, 8)

    t2_builder = @brm begin
        mu ~ 1 + t2(x, z; k=(5, 5), basis=(:cr, :cr), full=false)
        y ~ Normal(mu, 1)
    end
    tensor, tensor_expr = prepared_term(t2_builder, data, :mu, t2)
    @test size(tensor.state.Xfixed) == (12, 3)
    @test size(tensor.state.Zrr, 1) == 12

    me_builder = @brm begin mu ~ 1 + me(x, 0.2); y ~ Normal(mu, 1) end
    measured, _ = prepared_term(me_builder, data, :mu, me)
    @test measured.state.x_obs ≈ data.x
    @test measured.state.sd_x == 0.2

    prior_builder = @brm begin
        sd(mu, s(x)) ~ Exponential(2)
        latent(mu, me(x)) ~ Normal(1, 3)
        mu ~ 1 + s(x) + me(x, 0.2)
        y ~ Normal(mu, 1)
    end
    prior_spline, _ = prepared_term(prior_builder, data, :mu, s)
    prior_measured, _ = prepared_term(prior_builder, data, :mu, me)
    @test BRM.getf(prior_spline.state.sd_prior) <: Exponential
    @test BRM.getargs(prior_spline.state.sd_prior) == (2,)
    @test BRM.getf(prior_measured.state.latent_prior) <: Normal
    @test BRM.getargs(prior_measured.state.latent_prior) == (1, 3)

    # Replay applies the fitted basis before considering the new data's
    # degeneracy. A fresh fit would reject each constant margin.
    replay_data = (; x=fill(0.25, 5), z=fill(-0.1, 5), y=zeros(5))
    replay_context = BRM._brm_backend_context(s_builder(replay_data))
    replayed = BRM._brm_replay_term(spline, spline_expr, replay_context)
    @test size(replayed.state.Xnull) == (5, 2)
    @test size(replayed.state.Zpen) == (5, 8)
    @test all(isfinite, replayed.state.Zpen)

    tensor_replay_context = BRM._brm_backend_context(t2_builder(replay_data))
    tensor_replayed = BRM._brm_replay_term(
        tensor, tensor_expr, tensor_replay_context)
    @test size(tensor_replayed.state.Xfixed) == (5, 3)
    @test all(isfinite, tensor_replayed.state.Zrr)
end

@testset "Turing term submodel effects match direct Julia algebra" begin
    x = collect(range(-1.0, 1.0; length=12))
    z = x .^ 2 .+ 0.2 .* x
    data = (; x, z, y=zeros(12))

    spline, _ = prepared_term(
        (@brm begin mu ~ 1 + s(x); y ~ Normal(mu, 1) end),
        data, :mu, s)
    smodel = BRM._brm_turing_term_model(spline, 12)
    svalues = (b_fixed=[0.3, -0.2], sd_pen=0.4,
               b_pen_raw=collect(range(-0.2, 0.2; length=8)))
    sresult = Turing.generated_quantities(smodel, svalues)
    @test sresult.effect ≈ spline.state.Xnull * svalues.b_fixed +
        spline.state.Zpen * (svalues.sd_pen .* svalues.b_pen_raw)
    @test Turing.logjoint(smodel, svalues) ≈
        logpdf(Normal(), svalues.sd_pen) +
        sum(logpdf.(Normal(), svalues.b_pen_raw)) atol=1e-12 rtol=1e-12

    custom_spline, _ = prepared_term((@brm begin
        sd(mu, s(x)) ~ Exponential(2)
        mu ~ 1 + s(x)
        y ~ Normal(mu, 1)
    end), data, :mu, s)
    custom_smodel = BRM._brm_turing_term_model(custom_spline, 12)
    @test Turing.logjoint(custom_smodel, svalues) ≈
        logpdf(Exponential(2), svalues.sd_pen) +
        sum(logpdf.(Normal(), svalues.b_pen_raw)) atol=1e-12 rtol=1e-12

    tensor, _ = prepared_term(
        (@brm begin
            mu ~ 1 + t2(x, z; k=(5, 5), basis=(:cr, :cr), full=false)
            y ~ Normal(mu, 1)
        end), data, :mu, t2)
    tmodel = BRM._brm_turing_term_model(tensor, 12)
    tvalues = (
        b_fixed=[0.1, -0.2, 0.3], sd_pen=[0.4, 0.5, 0.6],
        b_rr_raw=fill(0.1, size(tensor.state.Zrr, 2)),
        b_rn_raw=fill(-0.1, size(tensor.state.Zrn, 2)),
        b_nr_raw=fill(0.2, size(tensor.state.Znr, 2)),
    )
    tresult = Turing.generated_quantities(tmodel, tvalues)
    toracle = tensor.state.Xfixed * tvalues.b_fixed +
        tensor.state.Zrr * (tvalues.sd_pen[1] .* tvalues.b_rr_raw) +
        tensor.state.Zrn * (tvalues.sd_pen[2] .* tvalues.b_rn_raw) +
        tensor.state.Znr * (tvalues.sd_pen[3] .* tvalues.b_nr_raw)
    @test tresult.effect ≈ toracle
    @test Turing.logjoint(tmodel, tvalues) ≈
        sum(logpdf.(Normal(), tvalues.sd_pen)) +
        sum(logpdf.(Normal(), tvalues.b_rr_raw)) +
        sum(logpdf.(Normal(), tvalues.b_rn_raw)) +
        sum(logpdf.(Normal(), tvalues.b_nr_raw)) atol=1e-12 rtol=1e-12

    measured, _ = prepared_term(
        (@brm begin mu ~ 1 + me(x, 0.2); y ~ Normal(mu, 1) end),
        data, :mu, me)
    mmodel = BRM._brm_turing_term_model(measured, 12)
    mvalues = (; x_true=data.x .+ 0.05, beta=-0.7)
    mresult = Turing.generated_quantities(mmodel, mvalues)
    @test mresult.effect ≈ mvalues.beta .* mvalues.x_true
    me_oracle =
        logpdf(Normal(), mvalues.beta) +
        sum(logpdf.(Normal(), mvalues.x_true)) +
        sum(logpdf.(Normal.(mvalues.x_true, measured.state.sd_x), data.x))
    @test Turing.logjoint(mmodel, mvalues) ≈ me_oracle atol=1e-12 rtol=1e-12

    custom_measured, _ = prepared_term((@brm begin
        latent(mu, me(x)) ~ Normal(1, 3)
        mu ~ 1 + me(x, 0.2)
        y ~ Normal(mu, 1)
    end), data, :mu, me)
    custom_mmodel = BRM._brm_turing_term_model(custom_measured, 12)
    custom_me_oracle =
        logpdf(Normal(), mvalues.beta) +
        sum(logpdf.(Normal(1, 3), mvalues.x_true)) +
        sum(logpdf.(Normal.(mvalues.x_true, custom_measured.state.sd_x), data.x))
    @test Turing.logjoint(custom_mmodel, mvalues) ≈ custom_me_oracle atol=1e-12 rtol=1e-12
end
