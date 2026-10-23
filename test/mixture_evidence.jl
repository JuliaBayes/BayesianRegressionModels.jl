# Censored, truncated, interval-censored and weighted `MixtureModel` responses
# on SBBRMI (snag censored-finite-cbe5c30d), through StanBlocks' value-family
# combinator slots (StanBlocks 94e414bb). At the program's own constrained
# values each row's pointwise log-likelihood must equal Distributions.jl's
# wrapped `MixtureModel`, and each row's `y_membership` the posterior
# probability of each component given that row's evidence: the density for an
# exact row, the CDF mass beyond the limit for a censored one, the interval
# mass for an interval-censored one.
#
#     julia --project=test test/mixture_evidence.jl
using Test, BayesianRegressionModels, Distributions, BridgeStan, LogDensityProblems
using LogExpFunctions: logistic, logit
import StanBlocks
const BRM = BayesianRegressionModels
include("mixture_stan_values.jl")

membership_matrix(values, name, n, K) = [values["$name.$i.$k"] for i in 1:n, k in 1:K]
# Row-wise `softmax` of per-component log terms.
row_softmax(terms) = (m = exp.(terms .- maximum(terms; dims=2)); m ./ sum(m; dims=2))

function evidence_program(builder, data)
    descriptor = brm_descriptor(builder, data; mod=@__MODULE__, highlights=())
    @test StanBlocks.stanc_check(BRM.stan_code(descriptor.plan); warn_pedantic=false).ok
    output = brm_output(descriptor, :y; role=:membership)
    @test output.name === :y_membership
    @test output.dims == (length(data.y), 2)
    descriptor, mixture_problem(descriptor)
end

# Central finite differences of `f` at `q`.
fd_gradient(f, q; h=1e-6) = [(f(q .+ h .* (1:length(q) .== i)) -
                              f(q .- h .* (1:length(q) .== i))) / 2h for i in eachindex(q)]

@testset "censored Gaussian mixture below and above the limits" begin
    data = (; y=[-2.0, -1.7, 0.3, 2.2, 3.0, 3.0, -2.0],
              lloq=fill(-2.0, 7), uloq=fill(3.0, 7))
    builder = @brm begin
        mu1 ~ Normal(-1, 0.5)
        mu2 ~ Normal(2, 0.5)
        sigma ~ Exponential(1)
        w ~ Beta(2, 5)
        y ~ censored(MixtureModel([Normal(mu1, sigma), Normal(mu2, sigma)], [1 - w, w]);
                     lower=lloq, upper=uloq)
    end
    _, problem = evidence_program(builder, data)
    @test LogDensityProblems.dimension(problem) == 4
    q = [-0.8, 2.3, log(0.7), logit(0.3)]
    values = mixture_values(problem, q)
    components(v) = [Normal(v[1], exp(v[3])), Normal(v[2], exp(v[3]))]
    weights(v) = [1 - logistic(v[4]), logistic(v[4])]
    law(v) = censored(MixtureModel(components(v), weights(v)), -2.0, 3.0)
    @test [values["y_likelihood.$i"] for i in 1:7] ≈ logpdf.(law(q), data.y) atol=1e-10
    terms = [log(weights(q)[k]) +
             (y <= -2.0 ? logcdf(components(q)[k], -2.0) :
              y >= 3.0 ? logccdf(components(q)[k], 3.0) : logpdf(components(q)[k], y))
             for y in data.y, k in 1:2]
    membership = membership_matrix(values, "y_membership", 7, 2)
    @test membership ≈ row_softmax(terms) atol=1e-10
    @test sum(membership; dims=2) ≈ ones(7, 1)
    reference(v) = sum(logpdf.(law(v), data.y)) +
        logpdf(Normal(-1, 0.5), v[1]) + logpdf(Normal(2, 0.5), v[2]) +
        logpdf(Exponential(1), exp(v[3])) + v[3] +
        logpdf(Beta(2, 5), logistic(v[4])) + log(logistic(v[4]) * (1 - logistic(v[4])))
    _, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
    @test gradient ≈ fd_gradient(reference, q) atol=1e-5
end

@testset "truncated Gaussian mixture keeps the density's membership" begin
    data = (; y=[-2.4, -1.7, 0.3, 2.2, 2.9])
    builder = @brm begin
        mu1 ~ Normal(-1, 0.5)
        mu2 ~ Normal(2, 0.5)
        y ~ truncated(MixtureModel([Normal(mu1, 0.8), Normal(mu2, 0.8)], [0.4, 0.6]);
                      lower=-2.5)
    end
    _, problem = evidence_program(builder, data)
    q = [-0.7, 2.2]
    values = mixture_values(problem, q)
    components = [Normal(q[1], 0.8), Normal(q[2], 0.8)]
    mixture = MixtureModel(components, [0.4, 0.6])
    @test [values["y_likelihood.$i"] for i in 1:5] ≈
        logpdf.(truncated(mixture; lower=-2.5), data.y) atol=1e-10
    # The truncation normalizer divides every component of a row alike.
    terms = [log([0.4, 0.6][k]) + logpdf(components[k], y) for y in data.y, k in 1:2]
    @test membership_matrix(values, "y_membership", 5, 2) ≈ row_softmax(terms) atol=1e-10
end

@testset "interval-censored Gaussian mixture" begin
    data = (; y=[-2.4, -1.7, 0.3, 2.2], hi=[-1.9, -1.0, 1.0, 2.5])
    builder = @brm begin
        mu1 ~ Normal(-1, 0.5)
        mu2 ~ Normal(2, 0.5)
        y ~ interval_censored(MixtureModel([Normal(mu1, 0.8), Normal(mu2, 0.8)],
                                           [0.4, 0.6]); upper=hi)
    end
    _, problem = evidence_program(builder, data)
    q = [-0.7, 2.2]
    values = mixture_values(problem, q)
    components = [Normal(q[1], 0.8), Normal(q[2], 0.8)]
    mixture = MixtureModel(components, [0.4, 0.6])
    # Each row lies in (y, hi].
    @test [values["y_likelihood.$i"] for i in 1:4] ≈
        log.(cdf.(mixture, data.hi) .- cdf.(mixture, data.y)) atol=1e-10
    terms = [log([0.4, 0.6][k]) +
             log(cdf(components[k], data.hi[i]) - cdf(components[k], data.y[i]))
             for i in 1:4, k in 1:2]
    @test membership_matrix(values, "y_membership", 4, 2) ≈ row_softmax(terms) atol=1e-10
end

@testset "censored Poisson mixture: a row at the upper limit means at least it" begin
    data = (; y=[1, 2, 3, 6, 6, 1, 4], lo=fill(1, 7), hi=fill(6, 7))
    builder = @brm begin
        lambda1 ~ Exponential(1)
        lambda2 ~ Exponential(5)
        y ~ censored(MixtureModel([Poisson(lambda1), Poisson(lambda2)], [0.3, 0.7]);
                     lower=lo, upper=hi)
    end
    _, problem = evidence_program(builder, data)
    q = [log(1.2), log(4.5)]
    values = mixture_values(problem, q)
    components = [Poisson(exp(q[1])), Poisson(exp(q[2]))]
    mixture = MixtureModel(components, [0.3, 0.7])
    # Distributions' censored atoms: P(Y <= lo) at lo, P(Y >= hi) at hi.
    @test [values["y_likelihood.$i"] for i in 1:7] ≈
        logpdf.(censored(mixture, 1, 6), data.y) atol=1e-10
    terms = [log([0.3, 0.7][k]) +
             (y <= 1 ? logcdf(components[k], 1) :
              y >= 6 ? logccdf(components[k], 5) : logpdf(components[k], y))
             for y in data.y, k in 1:2]
    @test membership_matrix(values, "y_membership", 7, 2) ≈ row_softmax(terms) atol=1e-10
end

@testset "frequency-weighted mixture, alone and over a censored mixture" begin
    data = (; y=[-2.0, -1.7, 0.3, 2.2], f=[1.0, 2.0, 1.0, 3.0], lloq=fill(-2.0, 4))
    plain = @brm begin
        mu1 ~ Normal(-1, 0.5)
        mu2 ~ Normal(2, 0.5)
        y ~ weighted(MixtureModel([Normal(mu1, 0.8), Normal(mu2, 0.8)], [0.4, 0.6]),
                     fweights(f))
    end
    bounded = @brm begin
        mu1 ~ Normal(-1, 0.5)
        mu2 ~ Normal(2, 0.5)
        y ~ weighted(censored(MixtureModel([Normal(mu1, 0.8), Normal(mu2, 0.8)],
                                           [0.4, 0.6]); lower=lloq), fweights(f))
    end
    q = [-0.7, 2.2]
    components(v) = [Normal(v[1], 0.8), Normal(v[2], 0.8)]
    for (builder, law) in (
            (plain, v -> MixtureModel(components(v), [0.4, 0.6])),
            (bounded, v -> censored(MixtureModel(components(v), [0.4, 0.6]); lower=-2.0)))
        _, problem = evidence_program(builder, data)
        reference(v) = sum(data.f .* logpdf.(law(v), data.y)) +
            logpdf(Normal(-1, 0.5), v[1]) + logpdf(Normal(2, 0.5), v[2])
        _, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test gradient ≈ fd_gradient(reference, q) atol=1e-5
        # A row's frequency weight scales its likelihood, not its membership.
        terms = [log([0.4, 0.6][k]) +
                 (builder === bounded && y <= -2.0 ? logcdf(components(q)[k], -2.0) :
                  logpdf(components(q)[k], y)) for y in data.y, k in 1:2]
        @test membership_matrix(mixture_values(problem, q), "y_membership", 4, 2) ≈
            row_softmax(terms) atol=1e-10
    end
end
