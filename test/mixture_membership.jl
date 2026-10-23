# Per-row component-membership probabilities of a `MixtureModel` response on
# SBBRMI (snag censored-finite-cbe5c30d, decision 1l78mxi): every mixture
# response emits `<response>_membership`, an `n x K` generated quantity whose
# row `i` is the posterior probability that row `i` came from each component.
# At the program's own constrained values each row must equal Distributions.jl's
# `w_k f_k(y_i) / sum_j w_j f_j(y_i)`, and the descriptor must list the matrix
# under role `:membership`.
#
#     julia --project=test test/mixture_membership.jl
using Test, BayesianRegressionModels, Distributions, BridgeStan, LogDensityProblems
using LogExpFunctions: logit
import StanBlocks
const BRM = BayesianRegressionModels
include("mixture_stan_values.jl")

# Distributions' posterior component probabilities of one observation.
reference_membership(mixture, y) =
    exp.(log.(probs(mixture)) .+ logpdf.(components(mixture), y) .- logpdf(mixture, y))

membership_matrix(values, name, n, K) = [values["$name.$i.$k"] for i in 1:n, k in 1:K]

@testset "complementary weights from one sampled scalar" begin
    data = (; y=[-2.1, -1.9, 2.0, 2.2, 0.1])
    builder = @brm begin
        mu1 ~ Normal(-2, 0.1)
        mu2 ~ Normal(2, 0.1)
        w ~ Beta(2, 5)
        y ~ MixtureModel([Normal(mu1, 0.5), Normal(mu2, 0.5)], [1 - w, w])
    end
    descriptor = brm_descriptor(builder, data; mod=@__MODULE__, highlights=())
    @test StanBlocks.stanc_check(BRM.stan_code(descriptor.plan); warn_pedantic=false).ok
    output = brm_output(descriptor, :y; role=:membership)
    @test output.name === :y_membership
    @test output.kind === :generated_quantity
    @test output.source === :y
    @test output.labels == [:component_1, :component_2]
    @test output.dims == (5, 2)
    @test [o.name for o in brm_outputs(descriptor; role=:membership)] == [:y_membership]
    problem = mixture_problem(descriptor)
    # A generated quantity adds no sampled coordinate.
    @test LogDensityProblems.dimension(problem) == 3
    values = mixture_values(problem, [-1.8, 2.3, logit(0.3)])
    membership = membership_matrix(values, "y_membership", 5, 2)
    @test sum(membership; dims=2) ≈ ones(5, 1)
    mixture = MixtureModel([Normal(values["mu1"], 0.5), Normal(values["mu2"], 0.5)],
        [1 - values["w"], values["w"]])
    @test membership ≈ permutedims(reduce(hcat,
        [reference_membership(mixture, y) for y in data.y])) atol=1e-12
    # Read as draws through the descriptor (snag read-the-new-per-62e2b0bf):
    # the logical target is the response and `role` selects the carrier; the
    # columns follow the supplied names, Stan's column-major `y_membership.i.k`.
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    draws = permutedims([values[name] for name in names])
    membership_draws = brm_output_draws(descriptor, draws, names;
        logical=:y, role=:membership)
    @test size(membership_draws) == (1, 10)
    @test reshape(vec(membership_draws), 5, 2) == membership
    # refused: an emitted name is not a logical target, since the descriptor
    # resolves BRM meaning rather than emitter-owned names (`brm_output`); the
    # membership matrix's target is its response (decision 1l78mxi).
    @test_throws "pass `logical=:y, role=:membership`" brm_output_draws(
        descriptor, draws, names; logical=:y_membership)
end

@testset "per-row weights from a linked linear predictor" begin
    data = (; y=[-2.1, -1.9, 2.0, 2.2, 0.1, 1.5],
              x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    builder = @brm begin
        mu1 ~ Normal(-2, 0.1)
        mu2 ~ Normal(2, 0.1)
        sigma ~ Exponential(1)
        logit(w) ~ 1 + x
        y ~ MixtureModel([Normal(mu1, sigma), Normal(mu2, sigma)], [1 - w, w])
    end
    descriptor = brm_descriptor(builder, data; mod=@__MODULE__, highlights=())
    problem = mixture_problem(descriptor)
    D = LogDensityProblems.dimension(problem)
    values = mixture_values(problem, collect(range(-0.1, 0.4; length=D)))
    n = length(data.y)
    w = [values["w.$i"] for i in 1:n]
    @test !allequal(w)
    membership = membership_matrix(values, "y_membership", n, 2)
    expected = [reference_membership(MixtureModel(
                    [Normal(values["mu1"], values["sigma"]),
                     Normal(values["mu2"], values["sigma"])],
                    [1 - w[i], w[i]]), data.y[i])[k] for i in 1:n, k in 1:2]
    @test membership ≈ expected atol=1e-12
end

@testset "a discrete mixture with a simplex of three weights" begin
    data = (; y=[0, 1, 4, 9, 12, 3, 25])
    builder = @brm begin
        lambda1 ~ Gamma(2, 1)
        lambda2 ~ Gamma(5, 1)
        lambda3 ~ Gamma(20, 1)
        w ~ Dirichlet(3, 1.0)
        y ~ MixtureModel([Poisson(lambda1), Poisson(lambda2), Poisson(lambda3)], w)
    end
    descriptor = brm_descriptor(builder, data; mod=@__MODULE__, highlights=())
    @test StanBlocks.stanc_check(BRM.stan_code(descriptor.plan); warn_pedantic=false).ok
    output = brm_output(descriptor, :y; role=:membership)
    @test output.labels == [:component_1, :component_2, :component_3]
    @test output.dims == (7, 3)
    problem = mixture_problem(descriptor)
    D = LogDensityProblems.dimension(problem)
    values = mixture_values(problem, collect(range(-0.3, 0.6; length=D)))
    membership = membership_matrix(values, "y_membership", 7, 3)
    @test sum(membership; dims=2) ≈ ones(7, 1)
    mixture = MixtureModel([Poisson(values["lambda$k"]) for k in 1:3],
        [values["w.$k"] for k in 1:3])
    @test membership ≈ permutedims(reduce(hcat,
        [reference_membership(mixture, y) for y in data.y])) atol=1e-12
end

@testset "the membership name is reserved" begin
    data = (; y=[-2.1, -1.9, 2.0])
    # refused: `y_membership` is the output every mixture response emits
    # (decision 1l78mxi), so a data column or model value of that name would
    # be a second binding of one name.
    @test_throws "collides with a model/data column" SBBRMI((@brm begin
        mu ~ Normal(-2, 0.1)
        y ~ MixtureModel([Normal(mu + y_membership, 0.5), Normal(2, 0.5)], [0.5, 0.5])
    end)((; data..., y_membership=[0.1, 0.2, 0.3])); mod=@__MODULE__)
    @test_throws "reserved for the component-membership probabilities" SBBRMI((@brm begin
        mu ~ Normal(-2, 0.1)
        y ~ MixtureModel([Normal(mu, 0.5), Normal(2, 0.5)], [0.5, 0.5])
        y_membership = mu + 1
    end)(data); mod=@__MODULE__)
end
