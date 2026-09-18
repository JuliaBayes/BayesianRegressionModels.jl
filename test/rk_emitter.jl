# test/rk_emitter.jl — RK backend structural plan + slice-1 admission.
#
# Run: julia --project=test test/rk_emitter.jl
#
# Covers `_brm_rk_plan` (core, no RK dependency): the structural plan shape
# for the four admitted (family, link, predictor-link) triples and the
# fail-closed battery for everything outside slice 1. Execution/parity
# against the thin layer lives in the later parity corpus, not here.

using Test
using BayesianRegressionModels
using Distributions: Bernoulli, Binomial, Cauchy, Exponential, Gamma,
                     Normal, Poisson, truncated
using LogExpFunctions: logistic, logit

const BRM = BayesianRegressionModels

df = (;
    x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    z=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    g=[1, 1, 2, 2, 3, 3],
    y=[0.5, -0.2, 0.1, 0.9, 1.4, 1.1],
    n=[1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
    b=[0, 1, 0, 1, 1, 0],
    c=[2, 1, 3, 2, 4, 3],
)

@testset "gaussian identity plan shape" begin
    brmi = @brm df begin
        mu ~ 1 + x + g + offset(z)
        effect(mu, x) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test plan.n_obs == 6
    @test length(plan.responses) == 1
    likelihood = only(plan.responses)
    @test likelihood.family === :gaussian
    @test likelihood.link === :identity
    @test likelihood.response === :y
    @test likelihood.predictor === :mu
    @test likelihood.scale === :s
    @test isnothing(likelihood.weights)
    @test likelihood.evidence.kind === :none
    @test length(plan.predictors) == 1
    predictor = only(plan.predictors)
    @test predictor.name === :mu
    @test predictor.link === :identity
    @test [t.kind for t in predictor.terms] ==
        [:intercept, :continuous, :factor, :offset]
    factor_term = predictor.terms[3]
    @test factor_term.columns == [:g]
    @test factor_term.options == (contrasts=:treatment, ref=1, levels=:observed)
    @test factor_term.addressee === :g
    @test plan.columns[:g] == [1, 1, 2, 2, 3, 3]
    @test sort!([p.addressee for p in plan.population_priors]) ==
        [:Intercept, :g, :x]
    x_prior = only(p for p in plan.population_priors if p.addressee === :x)
    @test (x_prior.location, x_prior.scale) == (0.0, 2.0)
    default_prior = only(
        p for p in plan.population_priors if p.addressee === :Intercept)
    @test (default_prior.location, default_prior.scale) == (0.0, 1.0)
    @test length(plan.parameters) == 1
    @test only(plan.parameters).name === :s
    @test only(plan.parameters).family === :Exponential
    @test only(plan.parameters).args == (1.0,)
    @test isempty(plan.assignments)
    backend = BRM.RKBRMI(brmi, plan, nothing)
    @test sprint(show, backend) ==
        "RKBRMI with 4 population coefficients and 6 observations"
end

@testset "factor ref translates to sort-order index" begin
    brmi = @brm df begin
        mu ~ 1 + factor(g; ref=3)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    factor_term = only(plan.predictors).terms[2]
    @test factor_term.kind === :factor
    @test factor_term.options == (contrasts=:treatment, ref=3, levels=:observed)
end

@testset "bernoulli-logit spellings" begin
    direct = @brm df begin
        eta ~ 1 + x
        b ~ BernoulliLogit(eta)
    end
    plan = BRM._brm_rk_plan(direct)
    likelihood = only(plan.responses)
    @test (likelihood.family, likelihood.link) === (:bernoulli_logit, :logit)
    @test only(plan.predictors).link === :identity
    @test isnothing(likelihood.scale)

    wrapped = @brm df begin
        eta ~ 1 + x
        b ~ Bernoulli(logistic(eta))
    end
    plan = BRM._brm_rk_plan(wrapped)
    likelihood = only(plan.responses)
    @test (likelihood.family, likelihood.link) === (:bernoulli_logit, :logit)

    linked = @brm df begin
        logit(p) ~ 1 + x
        b ~ Bernoulli(p)
    end
    plan = BRM._brm_rk_plan(linked)
    likelihood = only(plan.responses)
    @test (likelihood.family, likelihood.link) === (:bernoulli_logit, :logit)
    @test only(plan.predictors).link === :logit
end

@testset "poisson-log plan shape" begin
    brmi = @brm df begin
        log(mu) ~ 1 + x
        c ~ Poisson(mu)
    end
    plan = BRM._brm_rk_plan(brmi)
    likelihood = only(plan.responses)
    @test (likelihood.family, likelihood.link) === (:poisson_log, :log)
    @test only(plan.predictors).link === :log
end

@testset "weights, evidence, and multi-response" begin
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ weighted(Normal(mu, s), fweights(n))
    end
    plan = BRM._brm_rk_plan(brmi)
    @test only(plan.responses).weights === :n
    @test plan.columns[:n] == [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]

    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ truncated(Normal(mu, s); lower=0.0, upper=2.0)
    end
    plan = BRM._brm_rk_plan(brmi)
    evidence = only(plan.responses).evidence
    @test evidence.kind === :truncated
    @test (evidence.lower, evidence.upper) == (0.0, 2.0)

    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ Normal(mu, s)
        z ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test length(plan.responses) == 2
    @test length(plan.predictors) == 1 # shared predictor planned once
end

@testset "half-normal prior and assignment folding" begin
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ truncated(Normal(0, 1); lower=0)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    parameter = only(plan.parameters)
    @test parameter.family === :Normal
    @test parameter.support_override === :positive

    brmi = @brm df begin
        mu ~ 1 + x
        half = 0.5
        s ~ Exponential(half)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test only(plan.parameters).args == (0.5,)
    @test isempty(plan.assignments) # folded literal disappears
end

@testset "fail closed: scope" begin
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + s(x)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x & g
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + g
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        effect(mu, :) ~ r2d2(R2=Normal(0.5, 0.2), tau_bsv=0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
end

@testset "fail closed: response side" begin
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        y ~ Gamma(2, mu)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        eta ~ 1 + x
        b ~ Binomial(10, logistic(eta))
    end)
    logit_eta = @brm df begin
        logit(eta) ~ 1 + x
        b ~ BernoulliLogit(eta)
    end
    @test_throws ErrorException BRM._brm_rk_plan(logit_eta)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ truncated(BernoulliLogit(mu); lower=0, upper=1)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ weighted(Normal(mu, s), aweights(n))
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        s ~ 1 + x
        y ~ Normal(mu, s)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ Normal(0, s)
    end)
end

@testset "fail closed: priors, data, and cycles" begin
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        effect(mu, x) ~ Cauchy(0, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        s ~ truncated(Normal(0, 1), 1, 2)
        y ~ Normal(mu, s)
    end)
    missing_df = (; df..., y=[0.5, missing, 0.1, 0.9, 1.4, 1.1])
    @test_throws ErrorException BRM._brm_rk_plan(@brm missing_df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm missing_df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        mi(y) ~ Normal(mu, s)
    end)
    cyclic_a = BRM._RKSampledParameter(:a, :Normal, (:b,), nothing, :a)
    cyclic_b = BRM._RKSampledParameter(:b, :Normal, (:a,), nothing, :b)
    @test_throws ErrorException BRM._rk_gate_acyclic!([cyclic_a, cyclic_b], [])
end
