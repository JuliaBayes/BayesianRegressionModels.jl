# test/logit_link_guard.jl — logit-scale likelihoods reject linked predictors.
#
# Snag logit-link-predi-e5d55be7: `logit(p) ~ 1 + x` binds `p` on the
# inverse-link scale (`p = logistic(eta)`), so `y ~ BernoulliLogit(p)` applied
# the link twice — SBBRMI emitted `p = inv_logit(logit_p)` next to
# `y ~ bernoulli_logit(p)`, and TuringBRMI evaluated
# `BernoulliLogit(logistic.(eta))`, collapsing expressible probabilities to
# [0.5, 0.731] with no warning. Both backends now fail loudly through the
# shared `_brm_validate_logit_family_links`, while the sanctioned spellings —
# probability families over linked predictors, logit families over
# identity-link predictors, and explicit re-links — keep lowering.
# Transpile + stanc gate (mirrors test/sbbrmi_bernoulli.jl); no BridgeStan
# compile, no sampling.

using Test
using BayesianRegressionModels
using StanBlocks
using LogExpFunctions: logit, logistic
using Distributions: Normal, logpdf
using Turing
import StanBlocks.stan: transpiles

const BRM = BayesianRegressionModels

link_data() = (;
    y=[0, 1, 1, 0, 1, 0, 1, 1],
    x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
    trials=[4, 4, 4, 4, 6, 6, 6, 6],
    count=[1, 3, 2, 0, 5, 2, 4, 6],
    level=[1, 2, 3, 1, 2, 3, 1, 2])

link_code(brmi) = begin
    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BRM.stan_code(sb)
    @test transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    code
end

@testset "SBBRMI rejects logit families over linked predictors" begin
    df = link_data()
    @test_throws "double link" SBBRMI((@brm begin
        logit(p) ~ 1 + x
        y ~ BernoulliLogit(p)
    end)(df); mod=@__MODULE__)
    @test_throws "double link" SBBRMI((@brm begin
        logit(p) ~ 1 + x
        count ~ BRM.BinomialLogit(trials, p)
    end)(df); mod=@__MODULE__)
    @test_throws "double link" SBBRMI((@brm begin
        logit(p1) ~ 1 + x
        p2 ~ 1 + x
        level ~ CategoricalLogit(p1, p2)
    end)(df); mod=@__MODULE__)
    # The rule is about the scale, not the link name: any non-identity link
    # feeds a non-logit quantity to a logit-scale family.
    @test_throws "double link" SBBRMI((@brm begin
        log(p) ~ 1 + x
        y ~ BernoulliLogit(p)
    end)(df); mod=@__MODULE__)

    # The error names the probability-scale remedy, not just the refusal.
    try
        SBBRMI((@brm begin
            logit(p) ~ 1 + x
            y ~ BernoulliLogit(p)
        end)(df); mod=@__MODULE__)
        @test false
    catch err
        @test err isa ErrorException
        @test occursin("Bernoulli(p)", err.msg)
        @test occursin("p ~ ...", err.msg)
    end
end

@testset "SBBRMI sanctioned link/family pairings still lower" begin
    df = link_data()

    linked_prob = link_code((@brm begin
        logit(p) ~ 1 + x
        y ~ Bernoulli(p)
    end)(df))
    @test occursin("p = inv_logit(logit_p);", linked_prob)
    @test occursin("y ~ bernoulli(p);", linked_prob)
    @test !occursin("bernoulli_logit", linked_prob)

    linked_binom = link_code((@brm begin
        logit(p) ~ 1 + x
        count ~ Binomial(trials, p)
    end)(df))
    @test occursin("count ~ binomial(trials, p);", linked_binom)
    @test !occursin("binomial_logit", linked_binom)

    plain_logit = link_code((@brm begin
        eta ~ 1 + x
        y ~ BernoulliLogit(eta)
    end)(df))
    @test occursin("y ~ bernoulli_logit(eta);", plain_logit)
    @test !occursin("inv_logit", plain_logit)

    plain_binom_logit = link_code((@brm begin
        eta ~ 1 + x
        count ~ BRM.BinomialLogit(trials, eta)
    end)(df))
    # The density consumes linear-scale logits directly; the generated-quantities
    # RNG helper converts to probabilities at draw time, which is the correct
    # single-link shape, so only the model-block line is asserted here.
    @test occursin("count ~ binomial_logit(trials, eta);", plain_binom_logit)

    plain_cat_logit = link_code((@brm begin
        eta2 ~ 1 + x
        eta3 ~ 1 + x
        level ~ CategoricalLogit(eta2, eta3)
    end)(df))
    @test occursin("categorical_logit", plain_cat_logit)
    @test !occursin("inv_logit", plain_cat_logit)

    # Explicit re-link maps back to the linear scale: the documented brm-use
    # pattern (`y ~ Normal(log(Vc), sigma)`) must not trip the guard.
    relinked = link_code((@brm begin
        s ~ Exponential(1)
        log(v) ~ 1 + x
        y ~ Normal(log(v), s)
    end)(df))
    @test occursin("normal(log(v), s)", relinked)
end

@testset "TuringBRMI rejects logit families over linked predictors" begin
    df = link_data()
    @test_throws "double link" TuringBRMI((@brm begin
        logit(p) ~ 1 + x
        y ~ BernoulliLogit(p)
    end)(df))
    @test_throws "double link" TuringBRMI((@brm begin
        logit(p) ~ 1 + x
        count ~ BRM.BinomialLogit(trials, p)
    end)(df))

    # The sanctioned linked spelling keeps its single-link density.
    backend = TuringBRMI((@brm begin
        logit(p) ~ 1 + x
        y ~ Bernoulli(p)
    end)(df))
    params = (; beta_pop=[0.25, -0.5])
    eta = backend.plan.design.matrix * params.beta_pop
    prior = sum(logpdf.(Normal(), params.beta_pop))
    expected = prior + sum(logpdf.(BRM.BernoulliLogit.(eta), backend.plan.response))
    @test Turing.logjoint(backend.model, params) ≈ expected atol=1e-12 rtol=1e-12
    @test Turing.DynamicPPL.returned(backend.model, params).p == logistic.(eta)
end
