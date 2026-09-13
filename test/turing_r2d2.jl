using Test
using BayesianRegressionModels
using Distributions: Beta, Dirichlet, Normal, logpdf
using LinearAlgebra: dot
using Random: Xoshiro
using Turing

const BRM_R2 = BayesianRegressionModels

@testset "Turing whole-predictor R2D2 variance decomposition" begin
    df = (; x=[-1.0, 0.0, 1.0, 2.0], y=[0.1, -0.2, 0.3, 0.4])
    backend = TuringBRMI((@brm begin
        mu ~ 1 + x
        effect(mu, :) ~ r2d2(R2=Beta(2, 3), tau_bsv=0.5, alpha=0.7)
        y ~ Normal(mu, 1)
    end)(df))
    plan = only(backend.plan.predictors).r2d2
    @test plan.share_indices == [0, 1]
    @test plan.alpha == 0.7
    @test plan.total_scale == 0.5

    params = (; r2d2_mu=(; R2=0.4, phi=[1.0], beta=[0.2, 0.3]))
    varx = sum(abs2, df.x .- sum(df.x) / length(df.x)) / (length(df.x) - 1)
    slope_scale = sqrt(0.4 * 0.5^2 / varx)
    expected_prior = logpdf(Beta(2, 3), 0.4) +
                     logpdf(Dirichlet([0.7]), [1.0]) +
                     logpdf(Normal(), 0.2) + logpdf(Normal(0, slope_scale), 0.3)
    @test Turing.logprior(backend.model, params) ≈ expected_prior atol=1e-10
    mu = 0.2 .+ 0.3 .* df.x
    @test Turing.loglikelihood(backend.model, params) ≈
          sum(logpdf.(Normal.(mu, 1), df.y)) atol=1e-10
    @test Turing.DynamicPPL.returned(backend.model, params).mu ≈ mu

    draw = rand(Xoshiro(22), backend.model)
    nested = draw.data.r2d2_mu.data
    @test 0 < nested.R2 < 1
    @test length(nested.beta) == 2
    @test all(isfinite(nested.beta[i]) for i in 1:length(nested.beta))

    sampled_total = TuringBRMI((@brm begin
        mu ~ 1 + x
        effect(mu, :) ~ r2d2(R2=Beta(2, 3), alpha=0.7)
        y ~ Normal(mu, 1)
    end)(df))
    sampled_params = (; r2d2_mu=(; R2=0.4, phi=[1.0], tau_bsv=0.5,
                                  beta=[0.2, 0.3]))
    sampled_slope_scale = sqrt(0.4 * 0.5^2 / varx)
    sampled_expected = logpdf(Beta(2, 3), 0.4) +
        logpdf(Dirichlet([0.7]), [1.0]) + logpdf(Normal(), 0.5) +
        logpdf(Normal(), 0.2) + logpdf(Normal(0, sampled_slope_scale), 0.3)
    # Positive geometry does not add the half-Normal normalization constant.
    @test Turing.logprior(sampled_total.model, sampled_params) ≈
          sampled_expected atol=1e-10

    grouped_df = merge(df, (; subject=[1, 1, 2, 2]))
    grouped = TuringBRMI((@brm begin
        mu ~ 1 + x + (1 | p | subject)
        effect(mu, :) ~ r2d2(R2=Beta(2, 3), tau_bsv=0.5, alpha=0.7)
        y ~ Normal(mu, 1)
    end)(grouped_df))
    grouped_draw = rand(Xoshiro(23), grouped.model)
    @test hasproperty(grouped_draw.data, :r2d2_mu)
    @test hasproperty(grouped_draw.data, :group_1_1)
    group_data = grouped_draw.data.group_1_1.data
    @test hasproperty(group_data, :z)
    @test !hasproperty(group_data, :tau)
end


@testset "Turing multi-response joint R2D2 has one allocation" begin
    df = (; x=[-1.0, 0.0, 1.0, 2.0], subject=[1, 1, 2, 2],
          y=zeros(4), y2=zeros(4))
    backend = TuringBRMI((@brm begin
        mu ~ 1 + x + (1 | p | subject)
        eta ~ 1 + x + (1 | p | subject)
        sd(:, p) ~ r2d2(R2=Beta(2, 2), alpha=0.7,
                         reference_scale=1.5, include=:population)
        y ~ Normal(mu, 1)
        y2 ~ Normal(eta, 1)
    end)(df))
    allocation = only(backend.plan.joint_r2d2)
    @test allocation.n_shares == 4
    @test [m.plan_index for m in allocation.predictors] == [1, 2]
    draw = rand(Xoshiro(25), backend.model)
    @test propertynames(draw.data) == (:r2d2_joint_1, :shared_group_1)
    @test isfinite(Turing.logprior(backend.model, draw.data))
end


@testset "joint R2D2 allocation accepts unequal predictor row axes" begin
    extension = Base.get_extension(BayesianRegressionModels,
                                   :BayesianRegressionModelsTuringExt)
    designs = ([ones(4) [-1.0, 0.0, 1.0, 2.0]],
               [ones(3) [-2.0, 0.0, 2.0]])
    model = extension._brm_r2d2_joint(
        designs, ([0, 3], [0, 4]), ([1], [2]), Beta(2, 2), 0.7, 1.5,
        ([Normal(), Normal()], [Normal(), Normal()]))
    draw = rand(Xoshiro(26), model)
    @test propertynames(draw.data) == (:R2, :phi, :beta)
    @test length(draw.data.phi) == 4
    @test isfinite(Turing.logprior(model, draw.data))
end


@testset "Turing joint population/group R2D2 preparation" begin
    df = (; x=[-1.0, 0.0, 1.0, 2.0], subject=[1, 1, 2, 2],
          y=[0.1, -0.2, 0.3, 0.4])
    backend = TuringBRMI((@brm begin
        mu ~ 1 + x + (1 | p | subject)
        sd(:, p) ~ r2d2(R2=Beta(2, 2), alpha=0.7,
                         reference_scale=1.5, include=:population)
        y ~ Normal(mu, 1)
    end)(df))
    allocation = only(backend.plan.joint_r2d2)
    @test allocation.id === :p
    @test allocation.n_shares == 2
    mapping = only(allocation.predictors)
    @test mapping.coefficient_shares == [0, 2]
    @test mapping.margin_shares == [1]

    draw = rand(Xoshiro(24), backend.model)
    joint = draw.data.r2d2_joint_1.data
    group = draw.data.group_1_1.data
    varx = sum(abs2, df.x .- sum(df.x) / length(df.x)) / (length(df.x)-1)
    slope_scale = 1.5 * sqrt(joint.phi[2] * joint.R2 /
                             ((1-joint.R2) * varx))
    expected = logpdf(Beta(2, 2), joint.R2) +
        logpdf(Dirichlet(fill(0.7, 2)), joint.phi) +
        logpdf(Normal(), joint.beta[1]) +
        logpdf(Normal(0, slope_scale), joint.beta[2]) +
        sum(logpdf.(Normal(), group.z))
    @test Turing.logprior(backend.model, draw.data) ≈ expected atol=1e-10
    @test !hasproperty(group, :tau)
    @test !hasproperty(group, :scale)
end
