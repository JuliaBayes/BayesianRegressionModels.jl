using Test
using BayesianRegressionModels
using Distributions
using LinearAlgebra
using Random: Xoshiro
using Turing

const BRM = BayesianRegressionModels

function shared_assignment_effect(blocks, parameters)
    L = Matrix(parameters.shared_group_1.L.L)
    coefficients = transpose(Diagonal(parameters.shared_group_1.tau) * L *
        reshape(parameters.shared_group_1.z_flat, length(parameters.shared_group_1.tau), :))
    offset = 0
    map(blocks) do block
        columns = (offset + 1):(offset + size(block.matrix, 2))
        offset += size(block.matrix, 2)
        vec(sum(block.matrix .* coefficients[block.indices, columns]; dims=2))
    end
end

@testset "response arguments do not alias persistent missing data" begin
    data = (; y=Union{Missing,Float64}[0.2, missing, -0.1])
    backend = TuringBRMI((@brm begin
        mu ~ Normal(0, 1)
        mi(y) ~ Normal(mu, 1)
    end)(data))
    original_response = copy(backend.plan.response)
    original_context = copy(backend.plan.context.data[:y])
    draw = rand(Xoshiro(91), backend.model)
    @test isequal(backend.plan.response, original_response)
    @test isequal(backend.plan.context.data[:y], original_context)
    pointwise = turing_pointwise_loglikelihoods(backend, draw.data).y
    @test ismissing(pointwise[2])
    @test pointwise[[1, 3]] ≈ logpdf.(Normal(draw.data.mu, 1), [0.2, -0.1])
end

@testset "shared-ID effects precede dependent distributional assignments" begin
    data = (; x=[-1.0, 0.5, 2.0, 0.25],
        subject=["b", "a", "b", "c"], y=[0.2, -0.1, 0.4, 0.3])
    backend = TuringBRMI((@brm begin
        mu ~ 1 + x + (1 | joint | subject)
        log_sigma ~ 1 + (1 | joint | subject)
        shifted = mu + 0.25
        doubled = 2 * shifted
        y ~ Normal(doubled, exp(log_sigma))
    end)(data))
    L = cholesky(Symmetric(Matrix{Float64}(I, 2, 2)))
    parameters = (; beta_pop=[0.1, -0.2], beta_pop_log_sigma=[-0.4],
        shared_group_1=(; L, tau=[0.5, 0.3],
            z_flat=[-0.2, 0.4, 0.3, -0.5, 0.1, 0.6]))
    blocks = map(component -> only(component.random_effects), backend.plan.predictors)
    group_effects = shared_assignment_effect(blocks, parameters)
    mu = backend.plan.predictors[1].design.matrix * parameters.beta_pop + group_effects[1]
    log_sigma = backend.plan.predictors[2].design.matrix *
        parameters.beta_pop_log_sigma + group_effects[2]
    shifted = mu .+ 0.25
    doubled = 2 .* shifted
    returned = Turing.DynamicPPL.returned(backend.model, parameters)
    @test returned.mu ≈ mu
    @test returned.shifted ≈ shifted
    @test returned.doubled ≈ doubled
    @test Turing.loglikelihood(backend.model, parameters) ≈
        sum(logpdf.(Normal.(doubled, exp.(log_sigma)), data.y))
end

@testset "shared-ID effects precede dependent multi-response assignments" begin
    data = (; x=[-1.0, 0.5, 2.0], subject=["a", "b", "a"],
        y=[0.2, -0.1, 0.4], count=[1, 0, 2])
    backend = TuringBRMI((@brm begin
        mu ~ 1 + x + (1 | joint | subject)
        log_rate ~ 1 + x + (1 | joint | subject)
        shifted = mu - 0.1
        rate = exp(log_rate)
        y ~ Normal(shifted, 1)
        count ~ Poisson(rate)
    end)(data))
    L = cholesky(Symmetric(Matrix{Float64}(I, 2, 2)))
    parameters = (; beta_pop_mu=[0.2, -0.1], beta_pop_log_rate=[-0.3, 0.15],
        shared_group_1=(; L, tau=[0.4, 0.25], z_flat=[-0.2, 0.5, 0.3, -0.4]))
    returned = Turing.DynamicPPL.returned(backend.model, parameters).responses
    @test returned[1].shifted ≈ returned[1].mu .- 0.1
    @test returned[2].rate ≈ exp.(returned[2].log_rate)
    @test Turing.loglikelihood(backend.model, parameters) ≈
        sum(logpdf.(Normal.(returned[1].shifted, 1), data.y)) +
        sum(logpdf.(Poisson.(returned[2].rate), data.count))
end
