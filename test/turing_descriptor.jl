using Test
using BayesianRegressionModels
using Turing
using Distributions
using Random: Xoshiro
const BRM = BayesianRegressionModels

@testset "Turing descriptor semantic outputs and execution" begin
    data = (; x=[-1.0, 0.5, 2.0], y=[0.2, 1.1, -0.4])
    backend = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end)(data))
    descriptor = brm_descriptor(backend; name=:native)
    @test descriptor.stan === nothing
    @test isempty(descriptor.highlights)
    @test descriptor.formula == sprint(show, backend.parent)
    @test Set(brm_columns(descriptor)) == Set((:x, :y))
    @test brm_output(descriptor, :mu; role=:linear_predictor).name == :mu
    @test brm_output(descriptor, :y; role=:posterior_predictive).name == :y
    @test brm_output(descriptor, :y; role=:pointwise_loglik).name == :y_loglik
    @test brm_output(descriptor, :sigma).kind == :parameter
    params = (; beta_pop=[0.25, -0.5], sigma=0.8)
    pointwise = brm_execute(descriptor, :pointwise_loglik, params)
    @test pointwise.y ≈ turing_pointwise_loglikelihoods(backend, params).y
    generated = brm_execute(descriptor, :generated_quantities, params)
    @test generated.mu == backend.plan.design.matrix * params.beta_pop
    predicted = brm_execute(descriptor, :predict, params; rng=Xoshiro(9))
    @test length(predicted.y) == length(data.y)
    replayed = brm_execute(descriptor, :reprocess,
        (; x=[0.0, 1.0], y=zeros(2)))
    @test replayed isa BRMDescriptor
    @test replayed.stan === nothing
end

@testset "Turing descriptor mirrors structural and R2D2 sites" begin
    grouped = TuringBRMI((@brm begin
        mu ~ 1 + x + (1 | g)
        y ~ Normal(mu, 1)
    end)((; x=[0.0, 1.0], g=["a", "b"], y=zeros(2))))
    grouped_descriptor = brm_descriptor(grouped)
    @test brm_output(grouped_descriptor, :g; role=:random_effect).name ==
          :group_1_1
    smooth = TuringBRMI((@brm begin
        mu ~ 1 + s(x)
        y ~ Normal(mu, 1)
    end)((; x=collect(range(0.0, 1.0; length=12)), y=zeros(12))))
    @test only(o for o in brm_descriptor(smooth).outputs
               if o.name == :term_mu_1).role == :parameter

    r2 = TuringBRMI((@brm begin
        mu ~ 1 + x
        effect(mu, :) ~ r2d2(R2=Beta(2, 3), tau_bsv=0.5)
        y ~ Normal(mu, 1)
    end)((; x=[0.0, 1.0], y=zeros(2))))
    descriptor = brm_descriptor(r2)
    @test brm_output(descriptor, :mu; role=:population_effect).kind ==
          :transformed_parameter
    @test only(o for o in descriptor.outputs if o.name == :r2d2_mu).kind ==
          :parameter

    matrix_prior = TuringBRMI((@brm begin
        L_res ~ LKJCovarianceFactor(2)
        y ~ Normal(0, 1)
    end)((; y=zeros(2))))
    @test brm_output(brm_descriptor(matrix_prior), :L_res).type == :matrix

    ordinal = TuringBRMI((@brm begin
        eta ~ 0 + x
        y ~ BRM.OrderedLogistic(eta)
    end)((; x=[-0.8, 0.2, 0.7], y=[1, 3, 2])))
    @test brm_output(brm_descriptor(ordinal), :y_cutpoints).type == :vector
end

@testset "Turing multi descriptor shares logical roles" begin
    data = (; x=[-1.0, 0.5, 2.0], y=[0.2, 1.1, -0.4], count=[0,2,4])
    backend = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        shifted = mu .+ 0.1
        y ~ Normal(shifted, sigma)
        log_rate ~ 1 + x
        count ~ Poisson(exp(log_rate))
    end)(data))
    descriptor = brm_descriptor(backend)
    @test Set(o.logical for o in brm_outputs(descriptor; role=:linear_predictor)) ==
          Set((:mu, :log_rate))
    @test Set(o.logical for o in brm_outputs(descriptor; role=:pointwise_loglik)) ==
          Set((:y, :count))
    @test only(o for o in descriptor.outputs if o.name == :shifted).kind ==
          :transformed_parameter
    @test descriptor.stan === nothing
    @test_throws ErrorException brm_descriptor(backend; highlights=(:foo,))
end

@testset "Stan and Turing descriptors share semantic schema and roles" begin
    builder = @brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    data = (; x=[-1.0, 0.5], y=[0.2, 1.1])
    turing = brm_descriptor(TuringBRMI(builder(data)))
    stan = brm_descriptor(builder, data; highlights=())
    @test Set(brm_columns(turing)) == Set(brm_columns(stan))
    @test Set(o.role for o in brm_outputs(turing; logical=:sigma)) ==
          Set(o.role for o in brm_outputs(stan; logical=:sigma)) == Set((:parameter,))
    @test brm_output(turing, :mu; role=:linear_predictor).role ==
          brm_output(stan, :mu; role=:linear_predictor).role
    @test Set(o.role for o in brm_outputs(turing; logical=:y)) ==
          Set(o.role for o in brm_outputs(stan; logical=:y)) ==
          Set((:posterior_predictive, :pointwise_loglik))
end
