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
using Distributions: Bernoulli, Binomial, Categorical, Cauchy, Dirichlet,
                     Exponential, Gamma, Multinomial, Normal, Poisson,
                     truncated
using LogExpFunctions: logistic, logit
using Statistics: mean

const BRM = BayesianRegressionModels

df = (;
    x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    z=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    g=[1, 1, 2, 2, 3, 3],
    y=[0.5, -0.2, 0.1, 0.9, 1.4, 1.1],
    n=[1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
    b=[0, 1, 0, 1, 1, 0],
    c=[2, 1, 3, 2, 4, 3],
    gs=["a", "a", "b", "b", "c", "c"],
    h=[1, 2, 1, 2, 1, 2],
    bf=[0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
    cf=[2.0, 1.0, 3.0, 2.0, 4.0, 3.0],
    k1=[1, 1, 1, 1, 1, 1],
)

@testset "gaussian identity plan shape" begin
    brmi = @brm df begin
        mu ~ 0 + x + g + offset(z)
        effect(mu, x) ~ Normal(0, 2)
        effect(mu, g) ~ Normal(0, 2)
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
        [:continuous, :factor, :offset]
    factor_term = predictor.terms[2]
    @test factor_term.columns == [:g]
    @test factor_term.options == (coding=:fullrank, levels=:observed)
    @test factor_term.addressee === :g
    @test plan.columns[:g] == [1, 1, 2, 2, 3, 3]
    @test sort!([p.addressee for p in plan.population_priors]) ==
        [:g, :x]
    x_prior = only(p for p in plan.population_priors if p.addressee === :x)
    @test (x_prior.location, x_prior.scale) == (0.0, 2.0)
    g_prior = only(
        p for p in plan.population_priors if p.addressee === :g)
    @test (g_prior.location, g_prior.scale) == (0.0, 2.0)
    @test length(plan.parameters) == 1
    @test only(plan.parameters).name === :s
    @test only(plan.parameters).family === :Exponential
    @test only(plan.parameters).args == (1.0,)
    @test isempty(plan.assignments)
    backend = BRM.RKBRMI(brmi, plan, nothing)
    @test sprint(show, backend) ==
        "RKBRMI with 4 population coefficients and 6 observations"
    @test parent(backend) === brmi
    @test structure_of(backend) == structure_of(brmi)
    @test priors_of(backend) == priors_of(brmi)
end

@testset "factor subsets translate refs to sort-order drops" begin
    brmi = @brm df begin
        mu ~ 1 + factor(g; ref=3)
        effect(mu, g) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    factor_term = only(plan.predictors).terms[2]
    @test factor_term.kind === :factor
    @test factor_term.options == (coding=:subset, drop=3, levels=:observed)
    # String groupings code exactly like integer levels: sort(unique)
    # order, ref by level value.
    bare = BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + gs
        effect(mu, gs) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    bare_term = only(bare.predictors).terms[1]
    @test bare_term.kind === :factor
    @test bare_term.options == (coding=:fullrank, levels=:observed)
    @test bare_term.addressee === :gs
    @test bare.columns[:gs] == ["a", "a", "b", "b", "c", "c"]
    @test sort!([p.addressee for p in bare.population_priors]) ==
        [:gs]
    explicit = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + factor(gs; ref="b")
        effect(mu, gs) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    explicit_term = only(explicit.predictors).terms[2]
    @test explicit_term.kind === :factor
    @test explicit_term.options == (coding=:subset, drop=2, levels=:observed)
    # `cmc=false` without an intercept pins the reference at zero: a
    # subset with no intercept.
    pinned = BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + factor(g; ref=3, cmc=false)
        effect(mu, g) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    pinned_term = only(pinned.predictors).terms[1]
    @test pinned_term.kind === :factor
    @test pinned_term.options == (coding=:subset, drop=3, levels=:observed)
    @test sort!([p.addressee for p in pinned.population_priors]) == [:g]
    # A non-string non-integer ref still fails closed with attribution.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + factor(gs; ref=1.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # An explicit ref under `0 +` (cell means) is meaningless.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + factor(g; ref=3)
        effect(mu, g) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # A bare factor under an intercept is unidentified.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + g
        effect(mu, g) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Factor blocks need an explicit prior (no default).
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + g
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # `cmc` is inert under an intercept: still a reference subset.
    inert = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + factor(g; ref=3, cmc=false)
        effect(mu, g) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    inert_term = only(inert.predictors).terms[2]
    @test inert_term.options == (coding=:subset, drop=3, levels=:observed)
    # `factor()` without `ref` under `0 +` is full-rank like the bare column.
    noref = BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + factor(g)
        effect(mu, g) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test only(noref.predictors).terms[1].options ==
        (coding=:fullrank, levels=:observed)
    # A global population prior also satisfies the explicit-prior rule.
    global_prior = BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + x + g
        effect(mu, :) ~ Normal(0, 3)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test sort!([p.addressee for p in global_prior.population_priors]) ==
        [:g, :x]
    @test only(p for p in global_prior.population_priors
        if p.addressee === :g).scale == 3.0
    # Two reference subsets share one intercept (neither spans it).
    two = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + factor(g; ref=1) + factor(h; ref=1)
        effect(mu, g) ~ Normal(0, 2)
        effect(mu, h) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test [t.options for t in only(two.predictors).terms[2:3]] ==
        [(coding=:subset, drop=1, levels=:observed),
         (coding=:subset, drop=1, levels=:observed)]
    @test sort!([p.addressee for p in two.population_priors]) ==
        [:Intercept, :g, :h]
    # A single observed level subsets to nothing (fail closed), but
    # full-ranks to one cell mean.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + factor(k1; ref=1)
        effect(mu, k1) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    one = BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + k1
        effect(mu, k1) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test only(one.predictors).terms[1].options ==
        (coding=:fullrank, levels=:observed)
    @test sort!([p.addressee for p in one.population_priors]) == [:k1]
end

@testset "continuous interaction lowers to derived product" begin
    brmi = @brm df begin
        mu ~ 1 + x + x & z
        effect(mu, int_x_x_z) ~ Normal(0, 5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test length(plan.derived) == 1
    derived = only(plan.derived)
    @test derived.name === :int_x_x_z
    @test derived.expression == Expr(:call, :.*, :x, :z)
    @test [t.kind for t in only(plan.predictors).terms] ==
        [:intercept, :continuous, :continuous]
    interaction = only(plan.predictors).terms[3]
    @test interaction.columns == [:int_x_x_z]
    @test interaction.addressee === :int_x_x_z
    @test sort!([p.addressee for p in plan.population_priors]) ==
        [:Intercept, :int_x_x_z, :x]
    prior = only(
        p for p in plan.population_priors if p.addressee === :int_x_x_z)
    @test (prior.location, prior.scale) == (0.0, 5.0)
    @test sort!(collect(keys(plan.columns))) == [:x, :y, :z]
end

@testset "categorical interactions lower to comparison products" begin
    brmi = @brm df begin
        mu ~ 1 + x & g
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test [d.name for d in plan.derived] ==
        [:int_x_x_g_lvl_1, :int_x_x_g_lvl_2, :int_x_x_g_lvl_3]
    @test plan.derived[1].expression ==
        Expr(:call, :.*, :x, Expr(:call, :.==, :g, 1))
    @test plan.derived[2].expression ==
        Expr(:call, :.*, :x, Expr(:call, :.==, :g, 2))
    @test plan.derived[3].expression ==
        Expr(:call, :.*, :x, Expr(:call, :.==, :g, 3))
    @test length(only(plan.predictors).terms) == 4
    @test sort!([p.addressee for p in plan.population_priors]) ==
        [:Intercept, :int_x_x_g_lvl_1, :int_x_x_g_lvl_2, :int_x_x_g_lvl_3]
    # The reference level's dummy has no shared column (shared stays
    # treatment-coded), so it takes the emitter default.
    defaulted = only(p for p in plan.population_priors
        if p.addressee === :int_x_x_g_lvl_1)
    @test (defaulted.location, defaulted.scale) == (0.0, 1.0)
    # `factor()` is not admitted inside `&` operands (coding there is
    # always full-rank); the bare column spells it.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x & factor(g; ref=3)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Factor-factor crosses level comparisons; the full cross covers
    # every row, so it needs an intercept-free predictor.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + g & h
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    crossed = BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + g & h
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test [d.name for d in crossed.derived] ==
        [:int_g_lvl_1_x_h_lvl_1, :int_g_lvl_1_x_h_lvl_2,
         :int_g_lvl_2_x_h_lvl_1, :int_g_lvl_2_x_h_lvl_2,
         :int_g_lvl_3_x_h_lvl_1, :int_g_lvl_3_x_h_lvl_2]
    @test crossed.derived[1].expression == Expr(:call, :.*,
        Expr(:call, :.==, :g, 1), Expr(:call, :.==, :h, 1))
    @test sort!([p.addressee for p in crossed.population_priors]) ==
        sort!([d.name for d in crossed.derived])
    # String groupings in interactions fail closed: level codes cannot be
    # derived in-graph from raw strings.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x & gs
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
end

@testset "center/zscale lower to inline reductions" begin
    brmi = @brm df begin
        mu ~ 1 + center(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test [d.name for d in plan.derived] == [:center_x]
    @test only(plan.derived).expression ==
        Expr(:call, :.-, :x, Expr(:call, :mean, :x))
    @test only(plan.predictors).terms[2].addressee === :center_x
    scaled = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + zscale(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test [d.name for d in scaled.derived] == [:zscale_x]
    @test only(scaled.derived).expression == Expr(:call, :./,
        Expr(:call, :.-, :x, Expr(:call, :mean, :x)),
        Expr(:call, :std, :x))
    standardized = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + standardize(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test [d.name for d in standardized.derived] == [:standardize_x]
    @test only(standardized.derived).expression ==
        only(scaled.derived).expression
end

@testset "numeric data expressions lower to dotted forms" begin
    brmi = @brm df begin
        mu ~ 1 + log(z)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    # The shared protect label carries a process hash: assert structure,
    # not the exact name.
    @test length(plan.derived) == 1
    derived = only(plan.derived)
    @test derived.expression == Expr(:., :log, Expr(:tuple, :z))
    term = only(plan.predictors).terms[2]
    @test term.kind === :continuous
    @test term.columns == [derived.name]
    @test term.addressee === derived.name
    prior = only(
        p for p in plan.population_priors if p.addressee === derived.name)
    @test (prior.location, prior.scale) == (0.0, 1.0)
    @test sort!(collect(keys(plan.columns))) == [:y, :z]
    arithmetic = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x * 2
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test only(arithmetic.derived).expression ==
        Expr(:call, :.*, :x, 2)
    # Unknown functions fail closed with the admitted list named.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + sind(z)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Scalar-valued terms fail closed (predictors take vector terms).
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + mean(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Nested specials fail closed: shared materialization would crash.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + log(center(x))
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
end

@testset "offset of data expression lowers to derived offset" begin
    brmi = @brm df begin
        mu ~ 1 + x + offset(log(z))
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test [d.name for d in plan.derived] == [:rkd_offset_log_z]
    @test only(plan.derived).expression ==
        Expr(:., :log, Expr(:tuple, :z))
    off = only(plan.predictors).terms[3]
    @test off.kind === :offset
    @test off.columns == [:rkd_offset_log_z]
    @test sort!([p.addressee for p in plan.population_priors]) ==
        [:Intercept, :x]
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + offset(center(x))
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
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

@testset "slice-1 count/positive plan shapes" begin
    brmi = @brm df begin
        logit(p) ~ 1 + x
        b ~ Binomial(h, p)
    end
    plan = BRM._brm_rk_plan(brmi)
    likelihood = only(plan.responses)
    @test (likelihood.family, likelihood.link) === (:binomial_logit, :logit)
    @test likelihood.trials === :h
    @test plan.columns[:h] == [1, 2, 1, 2, 1, 2]
    brmi = @brm df begin
        logit(p) ~ 1 + x
        b ~ Binomial(2, p)
    end
    @test only(BRM._brm_rk_plan(brmi).responses).trials == 2
    # Relocated from fail-closed: the logistic-expression twin is admitted.
    brmi = @brm df begin
        eta ~ 1 + x
        b ~ Binomial(10, logistic(eta))
    end
    likelihood = only(BRM._brm_rk_plan(brmi).responses)
    @test (likelihood.family, likelihood.link) === (:binomial_logit, :logit)
    @test likelihood.trials == 10
    brmi = @brm df begin
        log(mu) ~ 1 + x
        phi ~ Exponential(1)
        c ~ NegativeBinomial2(mu, phi)
    end
    likelihood = only(BRM._brm_rk_plan(brmi).responses)
    @test (likelihood.family, likelihood.link) === (:nb2_log, :log)
    @test likelihood.scale === :phi
    brmi = @brm df begin
        log(mu) ~ 1 + x
        c ~ NegativeBinomial2(mu, 2.0)
    end
    @test only(BRM._brm_rk_plan(brmi).responses).scale == 2.0
    brmi = @brm df begin
        log(mu) ~ 1 + x
        alpha ~ Exponential(1)
        z ~ Gamma(alpha, mu / alpha)
    end
    likelihood = only(BRM._brm_rk_plan(brmi).responses)
    @test (likelihood.family, likelihood.link) === (:gamma_log, :log)
    @test likelihood.scale === :alpha
    brmi = @brm df begin
        log(mu) ~ 1 + x
        z ~ Gamma(2.0, mu / 2.0)
    end
    @test only(BRM._brm_rk_plan(brmi).responses).scale == 2.0
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

    # Positional form with Inf upper plans identically.
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ truncated(Normal(0, 1), 0, Inf)
        y ~ Normal(mu, s)
    end
    parameter = only(BRM._brm_rk_plan(brmi).parameters)
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

@testset "mains and crosses gate co-occurrence" begin
    # A full mixed cross sums exactly to its continuous leaf, so a main
    # effect on that leaf is structurally singular (intercept or not).
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + x & g
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + x + x & g
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Two crosses over the same leaf both sum to it.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x & g + x & h
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + x & g + x & h
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # An affine cousin (z-scored main) collides only with an intercept.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + zscale(x) + x & g
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    free = BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + zscale(x) + x & g
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test length(only(free.predictors).terms) == 4
    # Identical data-expression mains collide by expression equality,
    # while non-affine cousins (log) stay independent.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + log(z) + log(z) & g
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    nonaffine = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + z + log(z) & g
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test length(only(nonaffine.predictors).terms) == 5
    # Nested crosses splice their leaves: (x & g) & h still sums to x.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (x & g) & h
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Product mains meet product sums: (x & g) & z sums to x * z.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x * z + (x & g) & z
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Negated affine cousins collide with an intercept (1 - x).
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + (1 - x) + x & g
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
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
    # Tensor-product smooths stay closed with `s(x)` (thin-layer spline
    # contract pending): `t2(x, z)` must fail, not partially plan.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + t2(x, z)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end)
    # `&` interactions used to fail here; they are provisionally admitted
    # now (derived lowering, covered above). Monotonic effects stay closed.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + mo(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + g
        effect(mu, g) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Two full-cover groups without an intercept are mutually collinear.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + g + h
        effect(mu, g) ~ Normal(0, 2)
        effect(mu, h) ~ Normal(0, 2)
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

@testset "fail closed: SB long tail (mo1/me/ar/dar, simplex/LKJ/joint)" begin
    # Monotonic direct summand stays closed (`mo` is pinned above).
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + mo1(c)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Measurement-error latent predictor stays closed.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + me(x, 0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # AR(1) latent path stays closed.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + ar(x; p=1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Differenced-AR trajectory stays closed.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + dar(x; p=1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Simplex-valued parameter declaration stays closed.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        s ~ Dirichlet(3, 1.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end)
    # LKJ covariance-factor declaration stays closed.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Joint correlated-outcome response stays closed.
    dfj = (y1=[0.5, -0.2, 0.1, 0.9, 1.4, 1.1],
           y2=[0.1, 0.3, -0.4, 0.2, 0.8, -0.1],
           x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    @test_throws ErrorException BRM._brm_rk_plan(@brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end)
end

@testset "fail closed: exact gp awaits thin-layer dense cholesky" begin
    # `gp(...)` converges on SBBRMI only once ReactiveKernelsPPL grows the
    # latent non-centred construct SB emits (`_sb_gp`:
    # `cholesky_decompose(K) * z`, `z ~ std_normal`, `rho`/`sigma` lognormal).
    # Until then every spelling fails at the structured-term gate. When the
    # thin layer lands it, this testset flips to plan-shape assertions.
    @test_throws "structured term(s)" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + gp(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws "structured term(s)" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + gp(x, z; iso=false)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws "structured term(s)" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + gp(x; cov=:periodic, period=1.0)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
end

@testset "fail closed: response side" begin
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        y ~ Gamma(2, mu)
    end)
    # NOTE: `Binomial(10, logistic(eta))` lived here until slice 1 admitted
    # Binomial; it now plans in "slice-1 count/positive plan shapes".
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        logit(p) ~ 1 + x
        b ~ Binomial(n, p)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        logit(p) ~ 1 + x
        b ~ Binomial(2.5, p)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        logit(p) ~ 1 + x
        b ~ Binomial(-1, p)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        logit(p) ~ 1 + x
        c ~ Binomial(h, p)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        logit(p) ~ 1 + x
        y ~ Binomial(h, p)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        b ~ Binomial(h, mu)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        eta ~ 1 + x
        b ~ BinomialLogit(2, eta)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        log(mu) ~ 1 + x
        c ~ NegativeBinomial2(mu)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        log(mu) ~ 1 + x
        z ~ Gamma(2.0, mu / 3.0)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        log(mu) ~ 1 + x
        alpha ~ Exponential(1)
        z ~ Gamma(alpha, mu)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        log(mu) ~ 1 + x
        phi ~ Exponential(1)
        c ~ truncated(NegativeBinomial2(mu, phi); lower=0, upper=5)
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

@testset "thin-side validation mirrors" begin
    # Bare-column reduction crosses; nested reduction fails closed.
    brmi = @brm df begin
        mu ~ 1 + x
        m = sum(x)
        s ~ Exponential(m)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test length(plan.assignments) == 1
    @test only(plan.assignments).name === :m
    @test only(plan.parameters).args == (:m,)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        m = sum(log(x))
        s ~ Exponential(m)
        y ~ Normal(mu, s)
    end)
    # Response eltypes mirror the thin layer exactly.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        eta ~ 1 + x
        bf ~ BernoulliLogit(eta)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        log(mu) ~ 1 + x
        cf ~ Poisson(mu)
    end)
    # Interval evidence: upper required, lower forbidden, ordered values.
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ interval_censored(Normal(mu, s); upper=2.0)
    end
    evidence = only(BRM._brm_rk_plan(brmi).responses).evidence
    @test evidence.kind === :interval_censored
    @test (evidence.lower, evidence.upper) == (nothing, 2.0)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ interval_censored(Normal(mu, s); lower=0.0)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ interval_censored(Normal(mu, s); upper=1.0)
    end)
    # Inf bounds normalize to omission; NaN and reversals fail closed.
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ truncated(Normal(mu, s), 0, Inf)
    end
    evidence = only(BRM._brm_rk_plan(brmi).responses).evidence
    @test (evidence.lower, evidence.upper) == (0.0, nothing)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ truncated(Normal(mu, s), 0, NaN)
    end)
    # Poisson evidence bounds must be integer-valued (poisson.cdf(::Int)).
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        log(mu) ~ 1 + x
        c ~ truncated(Poisson(mu), 0, 6.5)
    end)
    brmi = @brm df begin
        log(mu) ~ 1 + x
        c ~ truncated(Poisson(mu), 0, 6)
    end
    @test only(BRM._brm_rk_plan(brmi).responses).evidence.kind ===
        :truncated
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ truncated(Normal(mu, s), 5.0, 1.0)
    end)
    # Folded constants substitute into scale and bounds.
    brmi = @brm df begin
        mu ~ 1 + x
        s0 = 0.5
        y ~ Normal(mu, s0)
    end
    @test only(BRM._brm_rk_plan(brmi).responses).scale == 0.5
    brmi = @brm df begin
        mu ~ 1 + x
        lo = 0.0
        s ~ Exponential(1)
        y ~ truncated(Normal(mu, s), lo, 2.0)
    end
    evidence = only(BRM._brm_rk_plan(brmi).responses).evidence
    @test (evidence.lower, evidence.upper) == (0.0, 2.0)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ truncated(Normal(mu, 1.0), s, 2.0)
    end)
    # Name hygiene mirrors the thin layer.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        mu_coef ~ Normal(0, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        _ppl_s ~ Exponential(1)
        y ~ Normal(mu, _ppl_s)
    end)
    # Half-normal location must be the literal 0 (:positive adds log(2)).
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        s ~ truncated(Normal(0.5, 1), 0, Inf)
        y ~ Normal(mu, s)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        m = mean(x)
        s ~ truncated(Normal(m, 1), 0, Inf)
        y ~ Normal(mu, s)
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

# Categorical/ordinal/multinomial admission points (decision 0w1i3qb):
# each family fails closed naming the missing thin-layer support, not the
# generic out-of-slice-1 spellings error.
function rk_plan_error(formula)
    try
        BRM._brm_rk_plan(formula)
        nothing
    catch error
        error
    end
end

@testset "fail closed: categorical/ordinal/multinomial attribution" begin
    ordered = rk_plan_error(@brm df begin
        eta ~ 1 + x
        c ~ OrderedLogistic(eta)
    end)
    @test ordered isa ErrorException
    @test occursin("OrderedLogistic", ordered.msg)
    @test occursin("ordered cutpoints", ordered.msg)
    ordinal = rk_plan_error(@brm df begin
        eta ~ 0 + x
        c ~ Ordinal(Cumulative(), LogitLink(), eta)
    end)
    @test ordinal isa ErrorException
    @test occursin("Ordinal", ordinal.msg)
    @test occursin("threshold", ordinal.msg)
    # The genuine two-predictor shape: the head-first gate fires, not the
    # generic several-predictors error.
    catlogit = rk_plan_error(@brm df begin
        eta1 ~ 1 + x
        eta2 ~ 1 + x
        c ~ CategoricalLogit(eta1, eta2)
    end)
    @test catlogit isa ErrorException
    @test occursin("CategoricalLogit", catlogit.msg)
    @test occursin("multi-predictor", catlogit.msg)
    categorical = rk_plan_error(@brm df begin
        mu ~ 1 + x
        c ~ Categorical(z)
    end)
    @test categorical isa ErrorException
    @test occursin("Categorical", categorical.msg)
    @test occursin("simplex", categorical.msg)
    multinomial = rk_plan_error(@brm df begin
        mu ~ 1 + x
        c ~ Multinomial(n, z)
    end)
    @test multinomial isa ErrorException
    @test occursin("Multinomial", multinomial.msg)
    @test occursin("simplex", multinomial.msg)
end

@testset "offset-only predictors plan with no priors" begin
    brmi = @brm df begin
        mu ~ 0 + offset(z)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    predictor = only(plan.predictors)
    @test predictor.name === :mu
    @test [t.kind for t in predictor.terms] == [:offset]
    @test predictor.terms[1].columns == [:z]
    @test isempty(plan.population_priors)
    @test plan.columns[:z] == df.z
    @test only(plan.responses).predictor === :mu
    # A derived offset-only predictor stages its definition.
    derived = BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + offset(log(z))
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test only(derived.predictors).terms[1].columns == [:rkd_offset_log_z]
    @test isempty(derived.population_priors)
    # Offsets take no priors: stated effect and r2d2 priors stay fail-closed.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + offset(z)
        effect(mu, z) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + offset(z)
        effect(mu, :) ~ r2d2(R2=Normal(0.5, 0.2), tau_bsv=0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
end
