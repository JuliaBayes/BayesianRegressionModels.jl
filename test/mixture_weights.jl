# `MixtureModel` weights that are model expressions on SBBRMI (snag
# censored-finite-cbe5c30d): complementary weights from one sampled scalar,
# `[1 - w, w]`, and per-row weights from a linked linear predictor,
# `logit(w) ~ 1 + x`. Each row's pointwise log-likelihood must equal
# Distributions.jl's `MixtureModel` log density at the program's own constrained
# values, and expression weights that are not a probability vector are rejected
# when the density is evaluated.
#
#     julia --project=test test/mixture_weights.jl
using Test, BayesianRegressionModels, Distributions, BridgeStan, LogDensityProblems
using LogExpFunctions: logit, logistic
import StanBlocks
const BRM = BayesianRegressionModels

const MW_CACHE = joinpath(tempdir(), "brm-mixture-weights")
isdir(MW_CACHE) || mkpath(MW_CACHE)

mw_problem(d) = brm_execute(d, :instantiate;
    path=joinpath(MW_CACHE, string(d.id) * ".stan"))

# Constrained values of every parameter, transformed parameter and generated
# quantity at the unconstrained point `q`, by Stan name.
function mw_values(problem, q)
    model = problem.model
    names = BridgeStan.param_names(model; include_tp=true, include_gq=true)
    values = BridgeStan.param_constrain(model, q; include_tp=true,
        include_gq=true, rng=BridgeStan.StanRNG(model, 20261023))
    Dict(zip(names, values))
end

@testset "complementary weights from one sampled scalar" begin
    data = (; y=[-2.1, -1.9, 2.0, 2.2, 0.1])
    builder = @brm begin
        mu1 ~ Normal(-2, 0.1)
        mu2 ~ Normal(2, 0.1)
        w ~ Beta(2, 5)
        y ~ MixtureModel([Normal(mu1, 0.5), Normal(mu2, 0.5)], [1 - w, w])
    end
    descriptor = brm_descriptor(builder, data; mod=@__MODULE__, highlights=())
    code = BRM.stan_code(descriptor.plan)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("brm_joint_mean_rows((1 - w), num_elements(y))", code)
    @test occursin("must be nonnegative and sum to 1", code)
    problem = mw_problem(descriptor)
    @test LogDensityProblems.dimension(problem) == 3
    q = [-1.8, 2.3, logit(0.3)]
    values = mw_values(problem, q)
    @test values["w"] ≈ 0.3
    mixture = MixtureModel([Normal(values["mu1"], 0.5), Normal(values["mu2"], 0.5)],
        [1 - values["w"], values["w"]])
    @test [values["y_likelihood.$i"] for i in eachindex(data.y)] ≈
        logpdf.(mixture, data.y) atol=1e-12
    # Declaration order is the unconstrained-coordinate order; `w` is
    # logit-transformed, so its density carries the logistic Jacobian.
    function reference(v)
        p = logistic(v[3])
        sum(logpdf.(MixtureModel([Normal(v[1], 0.5), Normal(v[2], 0.5)],
                [1 - p, p]), data.y)) +
            logpdf(Normal(-2, 0.1), v[1]) + logpdf(Normal(2, 0.1), v[2]) +
            logpdf(Beta(2, 5), p) + log(p * (1 - p))
    end
    _, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
    h = 1e-6
    basis(i) = [j == i ? 1.0 : 0.0 for j in 1:3]
    @test gradient ≈ [(reference(q .+ h .* basis(i)) - reference(q .- h .* basis(i))) / 2h
                      for i in 1:3] atol=1e-5
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
    @test StanBlocks.stanc_check(BRM.stan_code(descriptor.plan); warn_pedantic=false).ok
    problem = mw_problem(descriptor)
    D = LogDensityProblems.dimension(problem)
    values = mw_values(problem, collect(range(-0.1, 0.4; length=D)))
    n = length(data.y)
    w = [values["w.$i"] for i in 1:n]
    @test !allequal(w)
    expected = [logpdf(MixtureModel([Normal(values["mu1"], values["sigma"]),
                                     Normal(values["mu2"], values["sigma"])],
                                    [1 - w[i], w[i]]), data.y[i]) for i in 1:n]
    @test [values["y_likelihood.$i"] for i in 1:n] ≈ expected atol=1e-12
end

@testset "expression weights are checked when the density is evaluated" begin
    data = (; y=[-2.1, -1.9, 2.0, 2.2, 0.1])
    builder = @brm begin
        a ~ Uniform(0, 1)
        b ~ Uniform(0, 1)
        y ~ MixtureModel([Normal(-2, 0.5), Normal(2, 0.5)], [a, b])
    end
    descriptor = brm_descriptor(builder, data; mod=@__MODULE__, highlights=())
    problem = mw_problem(descriptor)
    # a = b = 0.5 is a probability vector; a = 0.5, b = logistic(1) is not,
    # and Distributions' `MixtureModel` refuses such weights as well.
    @test isfinite(LogDensityProblems.logdensity(problem, [0.0, 0.0]))
    @test_throws "must be nonnegative and sum to 1" BridgeStan.log_density(
        problem.model, [0.0, 1.0])
end

@testset "weights that are a probability vector by construction are not checked" begin
    data = (; y=[-1.2, 0.9, -0.8])
    for builder in (
            @brm(begin
                y ~ MixtureModel([Normal(-1.0, 0.5), Normal(1.0, 0.5)], [0.4, 0.6])
            end),
            @brm(begin
                w ~ Dirichlet(2, 1.0)
                y ~ MixtureModel([Normal(-1.0, 0.5), Normal(1.0, 0.5)], w)
            end))
        code = BRM.stan_code(SBBRMI(builder(data); mod=@__MODULE__))
        @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
        @test !occursin("must be nonnegative and sum to 1", code)
    end
end

@testset "malformed expression weights are refused when the model is built" begin
    data = (; y=[0.1, -0.2])
    # refused: a mixture weight is a probability, so a literal above 1 can
    # never be one (Distributions' `MixtureModel` rejects it too).
    @test_throws "outside [0, 1]" SBBRMI((@brm begin
        w ~ Beta(1, 1)
        y ~ MixtureModel([Normal(0, 1), Normal(1, 1)], [1.5, w])
    end)(data); mod=@__MODULE__)
    # refused: one weight per component (Distributions' `MixtureModel`
    # rejects a length mismatch too).
    @test_throws "2 components but 3 weights" SBBRMI((@brm begin
        w ~ Beta(1, 1)
        y ~ MixtureModel([Normal(0, 1), Normal(1, 1)], [1 - w, w, w])
    end)(data); mod=@__MODULE__)
end
