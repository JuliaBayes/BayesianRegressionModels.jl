# test/rk_emitter.jl — RK backend structural plan + slice-1 admission.
#
# Run: julia --project=test test/rk_emitter.jl
#
# Covers `_brm_rk_plan` (core, no RK dependency): the structural plan shape
# for the four admitted (family, link, predictor-link) triples and the
# fail-closed battery for everything outside slice 1. Execution/parity
# against the thin layer lives in the parity corpus
# (test/rk_parity.jl carries the ranef slice), not here.

using Test
using BayesianRegressionModels
using CategoricalArrays: categorical
using Distributions: Bernoulli, Beta, Binomial, Categorical, Cauchy, Dirichlet,
                     Exponential, Gamma, InverseGaussian, LocationScale,
                     LogNormal, MixtureModel, Multinomial, MvNormal, Normal,
                     Poisson, TDist, Uniform, Weibull, truncated
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

# Slice-2 group-A link words (links.jl user-link path, minus the
# InverseFunctions inverses the RK planner never needs) + a unit-interval
# response column for Beta shapes.
probit(p) = quantile(Normal(), p)
cloglog(p) = log(-log1p(-p))
dfp = merge(df, (; prop=[0.2, 0.7, 0.4, 0.6, 0.3, 0.8]))

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

@testset "slice-2 group-A plan shapes" begin
    brmi = @brm df begin
        probit(p) ~ 1 + x
        b ~ Binomial(h, p)
    end
    likelihood = only(BRM._brm_rk_plan(brmi).responses)
    @test (likelihood.family, likelihood.link) === (:binomial_probit, :probit)
    @test likelihood.trials === :h
    brmi = @brm df begin
        cloglog(p) ~ 1 + x
        b ~ Binomial(2, p)
    end
    likelihood = only(BRM._brm_rk_plan(brmi).responses)
    @test (likelihood.family, likelihood.link) === (:binomial_cloglog, :cloglog)
    @test likelihood.trials == 2
    brmi = @brm df begin
        probit(p) ~ 1 + x
        b ~ Bernoulli(p)
    end
    likelihood = only(BRM._brm_rk_plan(brmi).responses)
    @test (likelihood.family, likelihood.link) === (:bernoulli_probit, :probit)
    brmi = @brm df begin
        cloglog(p) ~ 1 + x
        b ~ Bernoulli(p)
    end
    likelihood = only(BRM._brm_rk_plan(brmi).responses)
    @test (likelihood.family, likelihood.link) === (:bernoulli_cloglog, :cloglog)
    brmi = @brm dfp begin
        logit(mu) ~ 1 + x
        kappa ~ Gamma(2.0, 1000.0)
        prop ~ Beta(mu * kappa, (1 - mu) * kappa)
    end
    likelihood = only(BRM._brm_rk_plan(brmi).responses)
    @test (likelihood.family, likelihood.link) === (:beta_logit, :logit)
    @test likelihood.scale === :kappa
    # Either multiplication order admits; the plan normalizes.
    brmi = @brm dfp begin
        logit(mu) ~ 1 + x
        prop ~ Beta(10.0 * mu, (1 - mu) * 10.0)
    end
    @test only(BRM._brm_rk_plan(brmi).responses).scale == 10.0
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
    # `(1|g)` used to fail here; the draws regime admits it now (ranef
    # buckets, covered below). `s(x)`/`t2(x, z)` used to fail here too;
    # they plan now (thin-layer spline surface landed, covered below).
    # `&` interactions used to fail here; they are provisionally admitted
    # now (derived lowering, covered above). `mo(x)` used to fail here too;
    # it plans now (thin-layer monotonic surface landed, covered below).
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
    # `r2d2` used to fail here too; it plans now (thin-layer R2D2
    # surface landed, covered below).
end

@testset "spline plan shape" begin
    # `s(x)` needs 10 unique axis values for the default rank-10 basis,
    # so smooth tests carry their own 12-row frame, not the shared 6-row df.
    # The scale is `sigma`: a parameter named `s` would shadow the smooth
    # head at macro expansion (shared @brm resolution, not RK-specific).
    xs = collect(range(-2.0, 2.0, length=12))
    zs = collect(range(0.0, 3.0, length=12))
    sdf = (; x=xs, z=zs, y=sin.(xs), w=cos.(xs))
    brmi = @brm sdf begin
        mu ~ 1 + s(x)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    plan = BRM._brm_rk_plan(brmi)
    predictor = only(plan.predictors)
    @test [t.kind for t in predictor.terms] == [:intercept, :spline]
    term = only(t for t in predictor.terms if t.kind === :spline)
    @test term.columns == [:x]
    @test (term.options.id, term.options.kind, term.options.k) ==
        (:s_x, :tps, 10)
    @test plan.columns[:x] == sdf.x
    # Spline parameters are thin-layer-owned: nothing lands in
    # plan.parameters for the smooth itself.
    @test [p.name for p in plan.parameters] == [:sigma]
    # `t2(x, z)`: tensor declaration over both axes, default k=(5, 5).
    brmi = @brm sdf begin
        mu ~ 1 + t2(x, z)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    plan = BRM._brm_rk_plan(brmi)
    term = only(t for t in only(plan.predictors).terms if t.kind === :spline)
    @test term.columns == [:x, :z]
    @test (term.options.id, term.options.kind, term.options.k) ==
        (:t2_x_z, :t2, (5, 5))
    @test plan.columns[:z] == sdf.z
    # Explicit `k` rides the declaration.
    brmi = @brm sdf begin
        mu ~ 1 + t2(x, z; k=(4, 6))
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    plan = BRM._brm_rk_plan(brmi)
    term = only(t for t in only(plan.predictors).terms if t.kind === :spline)
    @test term.options.k == (4, 6)
    # One id per smooth occurrence: a second smooth in the same
    # predictor takes its own axis-derived id.
    brmi = @brm sdf begin
        mu ~ 1 + s(x) + s(z)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    plan = BRM._brm_rk_plan(brmi)
    ids = [t.options.id for t in only(plan.predictors).terms
        if t.kind === :spline]
    @test ids == [:s_x, :s_z]
    # ... and the same smooth in a second predictor serializes instead
    # of colliding (exactly-one-use linkage per declaration).
    brmi = @brm sdf begin
        mu ~ 1 + s(x)
        nu ~ 1 + s(x)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
        w ~ Normal(nu, sigma)
    end
    plan = BRM._brm_rk_plan(brmi)
    ids = Set(t.options.id for p in plan.predictors for t in p.terms
        if t.kind === :spline)
    @test ids == Set([:s_x, :s_x_2])
    # Generated ids disambiguate against user parameters.
    brmi = @brm sdf begin
        mu ~ 1 + s(x)
        s_x ~ Normal(0, 1)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    plan = BRM._brm_rk_plan(brmi)
    term = only(t for t in only(plan.predictors).terms if t.kind === :spline)
    @test term.options.id == :s_x_2
    # A smooth-only predictor plans (no ordinary terms required).
    brmi = @brm sdf begin
        mu ~ s(x)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test [t.kind for t in only(plan.predictors).terms] == [:spline]
end

@testset "fail closed: spline sequenced spellings" begin
    # `sd(...)` smoothing-scale overrides stay closed until the
    # thin-layer surface sequences them (default half-normal only).
    # The gate precedes the basis fit, so the shared 6-row df suffices.
    @test_throws "smoothing-scale priors" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + s(x)
        sd(mu, s(x)) ~ Exponential(3)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end)
    @test_throws "smoothing-scale priors" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + t2(x, z)
        sd(mu, t2(x, z), rr) ~ Exponential(3)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end)
end

@testset "monotonic plan shape" begin
    # `mo(c)`: free-beta column over bound `<c>_idx` codes + a
    # `:simplex_dirichlet` increments vector (thin-layer monotonic surface).
    brmi = @brm df begin
        mu ~ 1 + mo(c)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    predictor = only(plan.predictors)
    @test [t.kind for t in predictor.terms] == [:intercept, :monotonic]
    term = only(t for t in predictor.terms if t.kind === :monotonic)
    @test term.columns == [:c_idx]
    @test term.addressee == :c_idx
    @test (term.options.increments, term.options.source) ==
        (:mo_c_simplex_incr, :c)
    @test term.options.alpha == [1.0, 1.0, 1.0]
    @test plan.columns[:c_idx] == [2, 1, 3, 2, 4, 3]
    @test eltype(plan.columns[:c_idx]) <: Integer
    vec = only(plan.vector_parameters)
    @test (vec.name, vec.family, vec.size) ==
        (:mo_c_simplex_incr, :simplex_dirichlet, 3)
    @test only(vec.args) == [1.0, 1.0, 1.0]
    @test [(p.addressee, p.location, p.scale)
        for p in plan.population_priors] ==
        [(:Intercept, 0.0, 1.0), (:c_idx, 0.0, 1.0)]
    @test [p.name for p in plan.parameters] == [:s]
    @test BRM._rk_num_coefficients(plan) == 2
    # `mo1(c)`: beta-free direct summand — self-addressed, no beta prior.
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + mo1(c)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test [t.kind for t in only(plan.predictors).terms] ==
        [:intercept, :monotonic_summand]
    term = only(t for t in only(plan.predictors).terms
        if t.kind === :monotonic_summand)
    @test (term.columns, term.addressee) == ([:c_idx], term.label)
    @test term.options.increments == :mo1_c_simplex_incr
    @test [(p.addressee, p.location, p.scale)
        for p in plan.population_priors] == [(:Intercept, 0.0, 1.0)]
    @test only(plan.vector_parameters).name == :mo1_c_simplex_incr
    # Beta-free `mo1` plans without an intercept (coefficient-free LP).
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ mo1(c)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test [t.kind for t in only(plan.predictors).terms] ==
        [:monotonic_summand]
    @test isempty(plan.population_priors)
    # Dirichlet overrides ride the increments concentration.
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + mo(c)
        simplex(mu, mo(c)) ~ Dirichlet(1, 2, 3)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test only(plan.vector_parameters).args == ([1.0, 2.0, 3.0],)
    # Scalar Dirichlet expands over the increments (SB `rep_vector`).
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + mo(c)
        simplex(mu, mo(c)) ~ Dirichlet(2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test only(plan.vector_parameters).args == ([2.0, 2.0, 2.0],)
    # Non-Dirichlet simplex priors stay closed (thin-layer increments are
    # Dirichlet-sampled).
    @test_throws "is not Dirichlet" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + mo(c)
        simplex(mu, mo(c)) ~ MvNormal([0.0, 0.0, 0.0], 1.0)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # ... as do sampled (non-literal) concentrations.
    @test_throws "not a literal vector" BRM._brm_rk_plan(@brm df begin
        concentration ~ Exponential(1)
        mu ~ 1 + mo(c)
        simplex(mu, mo(c)) ~ Dirichlet(concentration)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # K=1 `mo` vanishes (SB: no column, no beta, no `simplex[0]`).
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + mo(k1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test [t.kind for t in only(plan.predictors).terms] == [:intercept]
    @test isempty(plan.vector_parameters)
    @test !haskey(plan.columns, :k1_idx)
    # ... and re-inflates to a zeros offset under `0 +` (SB scalar `0.0`).
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + mo(k1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test [t.kind for t in only(plan.predictors).terms] == [:offset]
    @test plan.columns[:mo_k1_zero] == zeros(6)
    # K=1 `mo1` contributes a zeros offset (SB scalar `0.0`, vector-shaped).
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + mo1(k1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test [t.kind for t in only(plan.predictors).terms] ==
        [:intercept, :offset]
    zero = only(t for t in only(plan.predictors).terms
        if t.kind === :offset)
    @test plan.columns[only(zero.columns)] == zeros(6)
    # RK mints one monotonic label per (predictor, source): a second `mo(c)`
    # on the SAME predictor fails closed.
    @test_throws "one `mo` increments vector per predictor" BRM._brm_rk_plan(
        @brm df begin
            mu ~ 1 + mo(c) + mo(c)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    # ... while the same column across predictors plans two independent
    # increments vectors (snag mo-term-in-sever-fe459870: SB suffixes `mo`
    # contrasts per occurrence).
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + mo(c)
        log(sigma) ~ 1 + mo(c)
        y ~ Normal(mu, sigma)
    end)
    @test [(v.name, v.family, v.size) for v in plan.vector_parameters] == [
        (:mo_c_simplex_incr, :simplex_dirichlet, 3),
        (:mo_c_simplex_incr_2, :simplex_dirichlet, 3),
    ]
    # ... with independent per-predictor concentrations.
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + mo(c)
        simplex(mu, mo(c)) ~ Dirichlet(1, 2, 3)
        log(sigma) ~ 1 + mo(c)
        simplex(sigma, mo(c)) ~ Dirichlet(4, 5, 6)
        s ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end)
    @test [only(v.args) for v in plan.vector_parameters] ==
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    # ... while `mo(c)` + `mo1(c)` coexist (separate SB contrasts).
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + mo(c) + mo1(c)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test [t.kind for t in only(plan.predictors).terms] ==
        [:intercept, :monotonic, :monotonic_summand]
    @test Set(v.name for v in plan.vector_parameters) ==
        Set([:mo_c_simplex_incr, :mo1_c_simplex_incr])
    # Whole-predictor `:` fans out onto the mo beta (SB covers it too).
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + mo(c)
        effect(mu, :) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test [(p.addressee, p.location, p.scale)
        for p in plan.population_priors] ==
        [(:Intercept, 0.0, 2.0), (:x, 0.0, 2.0), (:c_idx, 0.0, 2.0)]
    # ... but a column-specific claim leaves the mo beta at its default.
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + mo(c)
        effect(mu, x) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test [(p.addressee, p.location, p.scale)
        for p in plan.population_priors] ==
        [(:Intercept, 0.0, 1.0), (:x, 0.0, 2.0), (:c_idx, 0.0, 1.0)]
    # Generated-name `effect(mu, mo_c)` stays unaddressable
    # (interaction-label precedent): the colon is the mainline spelling.
    @test_throws "not a population coefficient" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + mo(c)
        effect(mu, mo_c) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # String ordinals code by sorted level (shared `_brm_fit_levels`).
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + mo(gs)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test plan.columns[:gs_idx] == [1, 1, 2, 2, 3, 3]
    @test only(plan.vector_parameters).size == 2
    # Increments disambiguate against user parameters (smooth-id precedent).
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + mo(c)
        mo_c_simplex_incr ~ Normal(0, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test only(plan.vector_parameters).name == :mo_c_simplex_incr_2
    # A raw `<c>_idx` column colliding with the codes fails loud: sharing
    # one prior across the continuous column and the mo beta would
    # silently misprice one of them.
    dfc = (; df..., c_idx=[0.5, -0.2, 0.1, 0.9, 1.4, 1.1])
    @test_throws "collides with a population column" BRM._brm_rk_plan(
        @brm dfc begin
            mu ~ 1 + c_idx + mo(c)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
end

@testset "r2d2 plan shape" begin
    # `effect(mu, :) ~ r2d2(...)`: no PopulationPrior rows — the prior
    # mass lives in the R2D2Prior (SB-named R2/phi/tau_bsv, minted).
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + z
        effect(mu, :) ~ r2d2(R2=Beta(2, 5), alpha=0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    rp = only(plan.r2d2_priors)
    @test (rp.predictor, rp.r2, rp.phi, rp.tau) ==
        (:mu, :r2d2_mu_R2, :r2d2_mu_phi, :r2d2_mu_tau_bsv)
    @test isempty(rp.overrides)
    @test isempty(plan.population_priors)
    @test [(p.name, p.family, p.args, p.support_override)
        for p in plan.parameters] == [
        (:s, :Exponential, (1.0,), nothing),
        (:r2d2_mu_R2, :Beta, (2.0, 5.0), nothing),
        (:r2d2_mu_tau_bsv, :Normal, (0.0, 1.0), :positive)]
    vec = only(plan.vector_parameters)
    @test (vec.name, vec.family, vec.size) ==
        (:r2d2_mu_phi, :simplex_dirichlet, 2)
    @test only(vec.args) == [0.5, 0.5]
    # A data `tau_bsv` inlines as a literal (no sampled tau); the R2
    # prior defaults to Beta(1, 1) (SB `_brm_r2d2_prior`).
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + z
        effect(mu, :) ~ r2d2(tau_bsv=2.0)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    rp = only(plan.r2d2_priors)
    @test rp.tau == 2.0
    @test [p.name for p in plan.parameters] == [:s, :r2d2_mu_R2]
    r2 = only(p for p in plan.parameters if p.name === :r2d2_mu_R2)
    @test (r2.family, r2.args) == (:Beta, (1.0, 1.0))
    # Explicit-Normal columns keep their own scale and leave the simplex.
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + z
        effect(mu, :) ~ r2d2()
        effect(mu, x) ~ Normal(0, 3)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    rp = only(plan.r2d2_priors)
    @test rp.overrides == Dict(:x => (0.0, 3.0))
    @test only(plan.vector_parameters).size == 1
    # ... the intercept too (unstated it rides the thin-layer default).
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        effect(mu, :) ~ r2d2()
        effect(mu, Intercept) ~ Normal(1, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test only(plan.r2d2_priors).overrides ==
        Dict(:Intercept => (1.0, 2.0))
    # A full-cover factor joins the simplex per dummy.
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + g
        effect(mu, :) ~ r2d2()
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test only(plan.vector_parameters).size == 3
    # A subset-coded block rides share 0 with an explicit Normal (the
    # subset survives, like the PopulationPrior path); the rest joins.
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + factor(g; ref=3)
        effect(mu, :) ~ r2d2()
        effect(mu, g) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    rp = only(plan.r2d2_priors)
    @test rp.overrides == Dict(:g => (0.0, 2.0))
    @test only(plan.vector_parameters).size == 1
    # Generated names disambiguate against user parameters.
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        effect(mu, :) ~ r2d2()
        r2d2_mu_R2 ~ Normal(0, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test only(plan.r2d2_priors).r2 == :r2d2_mu_R2_2
    # Non-Beta R2 stays closed (SB would need a Stan translation too).
    @test_throws "must be `Beta(a, b)`" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        effect(mu, :) ~ r2d2(R2=Normal(0.5, 0.2), tau_bsv=0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Non-Normal overrides stay closed (Normal-only slice-1 rule).
    @test_throws "must be `Normal(location, scale)`" BRM._brm_rk_plan(
        @brm df begin
            mu ~ 1 + x
            effect(mu, :) ~ r2d2()
            effect(mu, x) ~ Cauchy(0, 1)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    # Every column excluded: SB mirror (`_sb_r2d2_overrides` refuses it).
    @test_throws "has nothing to allocate" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        effect(mu, :) ~ r2d2()
        effect(mu, x) ~ Normal(0, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Intercept-only decomposes nothing (SB's tau-only no-op has no
    # thin-layer form).
    @test_throws "decomposes nothing" BRM._brm_rk_plan(@brm df begin
        mu ~ 1
        effect(mu, :) ~ r2d2()
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # The mo contrast is parameter-derived: no data variance exists.
    @test_throws "combines `mo` with `r2d2`" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + mo(c)
        effect(mu, :) ~ r2d2()
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # ... while beta-free `mo1` summands coexist (peer skips them).
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + mo1(c)
        effect(mu, :) ~ r2d2()
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test Set(v.name for v in plan.vector_parameters) ==
        Set([:r2d2_mu_phi, :mo1_c_simplex_incr])
    phi = only(v for v in plan.vector_parameters
        if v.name === :r2d2_mu_phi)
    @test phi.size == 1
    # gp latents have no R2D2 term rule.
    @test_throws "combines `gp` with `r2d2`" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + gp(x)
        effect(mu, :) ~ r2d2()
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # An unstated subset-coded factor would join under full cover,
    # changing the coding — it must ride share 0 or go full-rank.
    @test_throws "subset-coded but carries no explicit" BRM._brm_rk_plan(
        @brm df begin
            mu ~ 1 + factor(g; ref=3)
            effect(mu, :) ~ r2d2()
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
end

@testset "differenced-AR plan shape" begin
    # Own frame: `dar` needs a strictly increasing time axis.
    tdf = (; t=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        u=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        y=[0.5, -0.2, 0.1, 0.9, 1.4, 1.1])
    # `dar(t)`: beta-free trajectory summand over the bound axis + SB-named
    # persistence/scale scalars (thin-layer dar surface).
    plan = BRM._brm_rk_plan(@brm tdf begin
        mu ~ 1 + dar(t)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test [t.kind for t in only(plan.predictors).terms] ==
        [:intercept, :dar]
    term = only(t for t in only(plan.predictors).terms if t.kind === :dar)
    @test (term.columns, term.addressee) == (Symbol[], term.label)
    @test (term.options.beta, term.options.sigma, term.options.source) ==
        (:dar_mu_t_beta, :dar_mu_t_sigma, :t)
    @test (term.options.beta_param.family, term.options.beta_param.args,
        term.options.beta_param.support_override) ==
        (:Normal, (0.5, 0.2), :interval)
    @test (term.options.sigma_param.family, term.options.sigma_param.args,
        term.options.sigma_param.support_override) ==
        (:Normal, (0.0, 0.2), :positive)
    @test plan.columns[:t] == tdf.t
    @test [(p.addressee, p.location, p.scale)
        for p in plan.population_priors] == [(:Intercept, 0.0, 1.0)]
    @test BRM._rk_num_coefficients(plan) == 1
    # `ar(...)` overrides ride the persistence location/scale.
    plan = BRM._brm_rk_plan(@brm tdf begin
        mu ~ 1 + dar(t)
        ar(mu, dar(t)) ~ Normal(0.6, 0.1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    term = only(t for t in only(plan.predictors).terms if t.kind === :dar)
    @test term.options.beta_param.args == (0.6, 0.1)
    # `sd(...)` Normal overrides ride the half-normal scale ...
    plan = BRM._brm_rk_plan(@brm tdf begin
        mu ~ 1 + dar(t)
        sd(mu, dar(t)) ~ Normal(0, 0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    term = only(t for t in only(plan.predictors).terms if t.kind === :dar)
    @test (term.options.sigma_param.family, term.options.sigma_param.args,
        term.options.sigma_param.support_override) ==
        (:Normal, (0.0, 0.5), :positive)
    # Non-literal hyperparameters stay closed (hyperparameters ride the
    # AST as literals).
    @test_throws "must be finite literals" BRM._brm_rk_plan(@brm tdf begin
        a ~ Normal(0, 1)
        mu ~ 1 + dar(t)
        ar(mu, dar(t)) ~ Normal(a, 0.1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws "must be finite literals" BRM._brm_rk_plan(@brm tdf begin
        a ~ Exponential(1)
        mu ~ 1 + dar(t)
        sd(mu, dar(t)) ~ Normal(0, a)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Non-Normal persistence stays closed (thin-layer beta is
    # truncated-Normal on [0, 1]).
    @test_throws "out of slice 1" BRM._brm_rk_plan(@brm tdf begin
        mu ~ 1 + dar(t)
        ar(mu, dar(t)) ~ Beta(2, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Non-half-normal scales stay closed (thin-layer sigma is
    # HalfNormal/truncated-positive).
    @test_throws "out of slice 1" BRM._brm_rk_plan(@brm tdf begin
        mu ~ 1 + dar(t)
        sd(mu, dar(t)) ~ Exponential(1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws "must have location 0" BRM._brm_rk_plan(@brm tdf begin
        mu ~ 1 + dar(t)
        sd(mu, dar(t)) ~ Normal(0.1, 0.2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # One dar summand per predictor (thin-layer v1 state scoping).
    @test_throws "one dar summand per predictor" BRM._brm_rk_plan(
        @brm tdf begin
            mu ~ 1 + dar(t) + dar(u)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    # Coefficient-free dar predictors stay closed (the surface fails
    # latent-only shapes — dar needs a sibling coefficient).
    @test_throws "no estimated coefficients" BRM._brm_rk_plan(@brm tdf begin
        mu ~ dar(t)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws "no estimated coefficients" BRM._brm_rk_plan(@brm tdf begin
        mu ~ 0 + dar(t)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # T=1: SB's path is identically 0 — a zeros offset (mo1-K=1 shape).
    tdf1 = (; t=[1.0], y=[0.5])
    plan = BRM._brm_rk_plan(@brm tdf1 begin
        mu ~ 1 + dar(t)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test [t.kind for t in only(plan.predictors).terms] ==
        [:intercept, :offset]
    zero = only(t for t in only(plan.predictors).terms
        if t.kind === :offset)
    @test plan.columns[only(zero.columns)] == zeros(1)
    # ... while distinct predictors take distinct trajectories (SB scopes
    # contrasts per predictor — no mo-style dup gate).
    plan = BRM._brm_rk_plan(@brm tdf begin
        mu ~ 1 + dar(t)
        log(sigma) ~ 1 + dar(t)
        y ~ Normal(mu, sigma)
    end)
    @test [t.options.beta for p in plan.predictors for t in p.terms
        if t.kind === :dar] == [:dar_mu_t_beta, :dar_sigma_t_beta]
    # Trajectory scalars are sampled, not population: `effect()` cannot
    # address them (generated-name precedent).
    @test_throws "not a population coefficient" BRM._brm_rk_plan(
        @brm tdf begin
            mu ~ 1 + dar(t)
            effect(mu, dar_mu_t_beta) ~ Normal(0, 2)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    # Trajectory names disambiguate against user parameters.
    plan = BRM._brm_rk_plan(@brm tdf begin
        mu ~ 1 + dar(t)
        dar_mu_t_beta ~ Normal(0, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    term = only(t for t in only(plan.predictors).terms if t.kind === :dar)
    @test term.options.beta == :dar_mu_t_beta_2
end

@testset "fail closed: SB long tail (simplex)" begin
    # `mo1(c)` used to fail here; it plans now (thin-layer monotonic
    # surface landed, covered in "monotonic plan shape"). `dar(t)` used to
    # fail here too; it plans now (thin-layer dar surface, covered in
    # "differenced-AR plan shape"). `ar` plans now as well (thin-layer
    # scan-ar slice landed, covered in "ar plan shape"). The LKJ
    # declaration and joint response plan now too (thin-layer correlated
    # slice landed, covered in "LKJ factor + joint plan shape"). `me`
    # plans now as well (thin-layer plate-vector slice landed, covered
    # in "me plan shape").
    # Unreferenced simplex-valued parameter declaration stays closed
    # (response-linked simplexes are the categorical lane's open shape).
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        s ~ Dirichlet(3, 1.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end)
end

@testset "LKJ factor + joint plan shape" begin
    dfj = (y1=[0.5, -0.2, 0.1, 0.9, 1.4, 1.1],
           y2=[0.1, 0.3, -0.4, 0.2, 0.8, -0.1],
           x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    plan = BRM._brm_rk_plan(@brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end)
    spec = only(plan.responses)
    @test spec.family === :mvnormal_cholesky
    @test spec.link === :identity
    @test spec.response === :y1
    @test spec.predictor === :mu1
    @test spec.extra_predictors == [:mu2]
    @test spec.extra_responses == [:y2]
    @test spec.factor === :L_res
    @test spec.label === :brm_joint_y1__y2
    @test spec.scale === nothing
    @test spec.trials === nothing
    @test spec.weights === nothing
    stem = only(plan.parameters)
    @test stem.name === :L_res
    @test stem.family === :LKJCovarianceFactor
    @test stem.args == (2, 1.0, 1.0)
    @test stem.args[1] isa Int
    @test [p.name for p in plan.predictors] == [:mu1, :mu2]
    @test plan.n_obs == 6
    @test plan.columns[:y1] == dfj.y1
    @test plan.columns[:y2] == dfj.y2
    @test plan.columns[:x] == dfj.x
    # Omitted scale_prior defaults to Exponential(1); shape rides.
    plan = BRM._brm_rk_plan(@brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; shape=2)
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end)
    @test only(plan.parameters).args == (2, 1.0, 2.0)
    # Sampled scale hyperparameters ride the scalar-prior shape; folded
    # const assignments plan as numbers.
    plan = BRM._brm_rk_plan(@brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        tau ~ Exponential(1)
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(tau))
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end)
    @test only(p for p in plan.parameters if p.name === :L_res).args ==
        (2, :tau, 1.0)
    plan = BRM._brm_rk_plan(@brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        t = 2.0
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(t))
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end)
    @test only(p for p in plan.parameters if p.name === :L_res).args ==
        (2, 2.0, 1.0)
    # K=3: three outcomes, three means, width-3 stem.
    df3 = merge(dfj, (; y3=[-0.3, 0.7, 0.2, -0.1, 0.4, 0.6]))
    plan = BRM._brm_rk_plan(@brm df3 begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        mu3 ~ 1 + x
        L3 ~ LKJCovarianceFactor(3; scale_prior=Exponential(1))
        [y1, y2, y3] ~ MvNormalCholesky([mu1, mu2, mu3], L3)
    end)
    spec = only(plan.responses)
    @test (spec.response, spec.extra_responses) == (:y1, [:y2, :y3])
    @test (spec.predictor, spec.extra_predictors) == (:mu1, [:mu2, :mu3])
    @test spec.factor === :L3
    @test only(plan.parameters).args == (3, 1.0, 1.0)
    # Mixed joint + plain responses share the one observation axis.
    plan = BRM._brm_rk_plan(@brm df3 begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        mu3 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
        s ~ Exponential(1)
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
        y3 ~ Normal(mu3, s)
    end)
    @test [r.family for r in plan.responses] ==
        [:mvnormal_cholesky, :gaussian]
    @test plan.n_obs == 6
end

@testset "LKJ factor + joint fail-closed battery" begin
    dfj = (y1=[0.5, -0.2, 0.1, 0.9, 1.4, 1.1],
           y2=[0.1, 0.3, -0.4, 0.2, 0.8, -0.1],
           x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    # Non-Exponential scale priors stay closed (SB admits more; the RK
    # slice mirrors the thin-layer Exponential-only contract).
    @test_throws "scale prior is `Exponential" BRM._brm_rk_plan(@brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Normal(0, 1))
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end)
    @test_throws "must be finite and positive" BRM._brm_rk_plan(@brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(0))
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end)
    @test_throws "must be finite and positive" BRM._brm_rk_plan(@brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(-1))
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end)
    # Shape is a literal hyperparameter.
    @test_throws "must be finite and strictly positive" BRM._brm_rk_plan(
        @brm dfj begin
            mu1 ~ 1 + x
            mu2 ~ 1 + x
            L_res ~ LKJCovarianceFactor(2; shape=0)
            [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
        end)
    @test_throws "must be a finite positive literal" BRM._brm_rk_plan(
        @brm dfj begin
            mu1 ~ 1 + x
            mu2 ~ 1 + x
            eta ~ Exponential(1)
            L_res ~ LKJCovarianceFactor(2; shape=eta)
            [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
        end)
    # Unknown keywords and bad dimensions stay closed.
    @test_throws "accepts only" BRM._brm_rk_plan(@brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1), df=3)
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end)
    @test_throws "needs an integer dimension" BRM._brm_rk_plan(@brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(0; scale_prior=Exponential(1))
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end)
    # A stem no joint response uses fails (the old LKJ pin shape, now
    # with linkage attribution).
    @test_throws "no joint response uses" BRM._brm_rk_plan(@brm dfj begin
        mu ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
        s ~ Exponential(1)
        y1 ~ Normal(mu, s)
    end)
    # A K=1 stem can never link (`@brm` needs at least two outcomes).
    @test_throws "no joint response uses" BRM._brm_rk_plan(@brm dfj begin
        mu ~ 1 + x
        L1 ~ LKJCovarianceFactor(1; scale_prior=Exponential(1))
        s ~ Exponential(1)
        y1 ~ Normal(mu, s)
    end)
    # Factor linkage: unknown stem, scalar-backed stem, width mismatch.
    @test_throws "must be a sampled parameter" BRM._brm_rk_plan(@brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_missing)
    end)
    @test_throws "must name an `LKJCovarianceFactor` declaration" BRM._brm_rk_plan(
        @brm dfj begin
            mu1 ~ 1 + x
            mu2 ~ 1 + x
            s ~ Exponential(1)
            [y1, y2] ~ MvNormalCholesky([mu1, mu2], s)
        end)
    @test_throws "has 2 ordered outcomes but factor" BRM._brm_rk_plan(
        @brm dfj begin
            mu1 ~ 1 + x
            mu2 ~ 1 + x
            L3 ~ LKJCovarianceFactor(3; scale_prior=Exponential(1))
            [y1, y2] ~ MvNormalCholesky([mu1, mu2], L3)
        end)
    # One factor per joint response.
    df4 = merge(dfj, (; y3=[-0.3, 0.7, 0.2, -0.1, 0.4, 0.6],
        y4=[0.2, -0.5, 0.3, 0.1, -0.2, 0.9]))
    @test_throws "feeds two joint responses" BRM._brm_rk_plan(@brm df4 begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        mu3 ~ 1 + x
        mu4 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
        [y3, y4] ~ MvNormalCholesky([mu3, mu4], L_res)
    end)
    # Means: one declared identity-link predictor per outcome.
    @test_throws "received 1 means" BRM._brm_rk_plan(@brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
        [y1, y2] ~ MvNormalCholesky([mu1], L_res)
    end)
    @test_throws "repeat a predictor" BRM._brm_rk_plan(@brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
        [y1, y2] ~ MvNormalCholesky([mu1, mu1], L_res)
    end)
    @test_throws "not a declared linear predictor" BRM._brm_rk_plan(
        @brm dfj begin
            mu1 ~ 1 + x
            mu2 ~ 1 + x
            L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
            [y1, y2] ~ MvNormalCholesky([0.0, mu2], L_res)
        end)
    @test_throws "not a declared linear predictor" BRM._brm_rk_plan(
        @brm dfj begin
            mu1 ~ 1 + x
            mu2 ~ 1 + x
            L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
            [y1, y2] ~ MvNormalCholesky([x, mu2], L_res)
        end)
    @test_throws "must be identity-link" BRM._brm_rk_plan(@brm dfj begin
        mu1 ~ 1 + x
        log(mu2) ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end)
    # The joint family is joint-only both directions.
    @test_throws "joint-only" BRM._brm_rk_plan(@brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
        y1 ~ MvNormalCholesky([mu1, mu2], L_res)
    end)
    @test_throws "explicit joint family" BRM._brm_rk_plan(@brm dfj begin
        mu1 ~ 1 + x
        s ~ Exponential(1)
        [y1, y2] ~ Normal(mu1, s)
    end)
    # Row weights and bounded evidence stay closed on joint responses.
    dfw = merge(dfj, (; w=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    @test_throws "weights on a joint density" BRM._brm_rk_plan(@brm dfw begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
        [y1, y2] ~ weighted(MvNormalCholesky([mu1, mu2], L_res), w)
    end)
    # Complete aligned rows: missing and non-finite outcomes fail.
    dfm = merge(dfj, (; y2=[0.1, 0.3, missing, 0.2, 0.8, -0.1]))
    @test_throws "contains `missing`" BRM._brm_rk_plan(@brm dfm begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end)
    dfn = merge(dfj, (; y2=[0.1, 0.3, Inf, 0.2, 0.8, -0.1]))
    @test_throws "non-finite" BRM._brm_rk_plan(@brm dfn begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end)
    # Sampled scales must be scalar (vector-valued θ fails here, not
    # thin-side).
    @test_throws "not a scalar parameter" BRM._brm_rk_plan(@brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_other ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(L_other))
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end)
    # The stem reserves its two derived thin-layer bindings.
    @test_throws "reserves emitted binding" BRM._brm_rk_plan(@brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
        L_res_scales ~ Normal(0, 1)
        s ~ Exponential(1)
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end)
end

@testset "ar plan shape" begin
    brmi = @brm df begin
        mu ~ 1 + ar(x; p=1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    predictor = only(plan.predictors)
    @test [t.kind for t in predictor.terms] == [:intercept, :ar]
    term = only(t for t in predictor.terms if t.kind === :ar)
    @test term.columns == [:x]
    @test term.addressee === :ar_mu_x
    @test (term.options.state, term.options.phi, term.options.phi_raw,
        term.options.eps) ==
        (:ar_mu_x, :phi_ar_mu_x, :phi_raw_ar_mu_x, :eps_ar_mu_x)
    @test plan.columns[:x] == df.x
    # AR parameters are preamble-emitted (like gp hypers): nothing lands
    # in plan.parameters for the path itself.
    @test [p.name for p in plan.parameters] == [:s]
    # Default beta prior matches SB's popefs default.
    prior = only(p for p in plan.population_priors
        if p.addressee === :ar_mu_x)
    @test (prior.location, prior.scale) == (0.0, 1.0)
    # `:`-wide statements claim the latent beta exactly as SB does.
    brmi = @brm df begin
        mu ~ 1 + ar(x; p=1)
        effect(mu, :) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    got = Dict(p.addressee => (p.location, p.scale)
        for p in plan.population_priors)
    @test got == Dict(:Intercept => (0.0, 2.0), :ar_mu_x => (0.0, 2.0))
    # The predictor-wide default loses to the predictor-specific claim.
    brmi = @brm df begin
        mu ~ 1 + ar(x; p=1)
        effect(:, :) ~ Normal(1, 3)
        effect(mu, :) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    prior = only(p for p in plan.population_priors
        if p.addressee === :ar_mu_x)
    @test (prior.location, prior.scale) == (0.0, 2.0)
    # Explicit addresses on the latent column stay sequenced (SB's
    # `popcoefnames` spelling `ar_x`; the predictor-namespaced Stan
    # spelling is not a coefficient on either side).
    @test_throws "sequenced" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + ar(x; p=1)
        effect(mu, ar_x) ~ Normal(0, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws "sequenced" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + ar(x; p=1)
        effect(:, ar_x) ~ Normal(0, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws "not a population coefficient" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + ar(x; p=1)
        effect(mu, ar_mu_x) ~ Normal(0, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Two time axes mint two states.
    brmi = @brm df begin
        mu ~ 1 + ar(x; p=1) + ar(z; p=1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    states = [t.options.state for t in only(plan.predictors).terms
        if t.kind === :ar]
    @test states == [:ar_mu_x, :ar_mu_z]
    # ... and an exact duplicate fails closed exactly as SB does
    # (its deterministic names collide; mo precedent).
    @test_throws "already taken" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + ar(x; p=1) + ar(x; p=1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # The same axis on a second predictor namespaces by predictor.
    brmi = @brm df begin
        mu ~ 1 + ar(x; p=1)
        nu ~ 1 + ar(x; p=1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
        n ~ Normal(nu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    states = Set(t.options.state for p in plan.predictors for t in p.terms
        if t.kind === :ar)
    @test states == Set([:ar_mu_x, :ar_nu_x])
    # p > 1 stays closed (shared preparation admits p=1 only).
    @test_throws "p=1" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + ar(x; p=2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # A non-numeric time axis stays closed.
    dfb = merge(df, (; flag=[true, false, true, false, true, false]))
    @test_throws "plain numeric vector" BRM._brm_rk_plan(@brm dfb begin
        mu ~ 1 + ar(flag; p=1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # A scan summand needs a sibling coefficient.
    @test_throws "sibling population coefficient" BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + ar(x; p=1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws "sibling population coefficient" BRM._brm_rk_plan(@brm df begin
        mu ~ ar(x; p=1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Offsets are not coefficients.
    @test_throws "sibling population coefficient" BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + offset(z) + ar(x; p=1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # A data column holding the deterministic name fails closed too.
    dfc = merge(df, (; ar_mu_x=[0.5, -0.2, 0.1, 0.9, 1.4, 1.1]))
    @test_throws "already taken" BRM._brm_rk_plan(@brm dfc begin
        mu ~ 1 + ar_mu_x + ar(x; p=1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
end

@testset "me plan shape" begin
    brmi = @brm df begin
        mu ~ 1 + me(x, 0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    predictor = only(plan.predictors)
    @test [t.kind for t in predictor.terms] == [:intercept, :me]
    term = only(t for t in predictor.terms if t.kind === :me)
    @test term.columns == [:x]
    @test term.addressee === :me_x
    @test term.options.latent === :me_x
    @test (term.options.loc, term.options.scale) == (0.0, 1.0)
    @test term.options.sd == 0.5
    @test plan.columns[:x] == df.x
    # The latent is preamble-emitted (like gp/ar latents): nothing lands
    # in plan.parameters for the plate itself.
    @test [p.name for p in plan.parameters] == [:s]
    # The observation likelihood rides a synthetic gaussian-identity
    # response after the formula responses.
    @test length(plan.responses) == 2
    obs = plan.responses[2]
    @test obs.family === :gaussian
    @test obs.link === :identity
    @test obs.response === :x
    @test obs.predictor === :me_x
    @test obs.scale == 0.5
    @test obs.weights === nothing
    @test obs.evidence.kind === :none
    # Default beta prior matches SB's popefs default.
    prior = only(p for p in plan.population_priors
        if p.addressee === :me_x)
    @test (prior.predictor, prior.location, prior.scale) ===
        (:mu, 0.0, 1.0)
    # `:`-wide statements claim the latent beta exactly as SB does.
    brmi = @brm df begin
        mu ~ 1 + me(x, 0.5)
        effect(mu, :) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    got = Dict(p.addressee => (p.location, p.scale)
        for p in plan.population_priors)
    @test got == Dict(:Intercept => (0.0, 2.0), :me_x => (0.0, 2.0))
    # The predictor-wide default loses to the predictor-specific claim.
    brmi = @brm df begin
        mu ~ 1 + me(x, 0.5)
        effect(:, :) ~ Normal(1, 3)
        effect(mu, :) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    prior = only(p for p in plan.population_priors
        if p.addressee === :me_x)
    @test (prior.location, prior.scale) == (0.0, 2.0)
    # A `latent(...)` override rides the plate's shared-scalar args.
    brmi = @brm df begin
        mu ~ 1 + me(x, 0.5)
        latent(mu, me(x)) ~ Normal(0.5, 1.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    term = only(t for t in only(plan.predictors).terms if t.kind === :me)
    @test (term.options.loc, term.options.scale) == (0.5, 1.5)
    brmi = @brm df begin
        mu ~ 1 + me(x, 0.5)
        latent(:, me(x)) ~ Normal(1, 4)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    term = only(t for t in only(plan.predictors).terms if t.kind === :me)
    @test (term.options.loc, term.options.scale) == (1.0, 4.0)
    # ... while a non-Normal latent prior stays closed (SB's
    # arbitrary-prior merge is sequenced).
    @test_throws "latent prior must be `Normal(location, scale)`" BRM._brm_rk_plan(
        @brm df begin
            mu ~ 1 + me(x, 0.5)
            latent(mu, me(x)) ~ Cauchy(0, 1)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    # Explicit addresses on the latent column stay sequenced (SB's
    # `popcoefnames` spelling `me_x`).
    @test_throws "sequenced" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + me(x, 0.5)
        effect(mu, me_x) ~ Normal(0, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws "sequenced" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + me(x, 0.5)
        effect(:, me_x) ~ Normal(0, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Two error sizes mint two latents (and two observations).
    brmi = @brm df begin
        mu ~ 1 + me(x, 0.5) + me(z, 0.25)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    latents = [t.options.latent for t in only(plan.predictors).terms
        if t.kind === :me]
    @test latents == [:me_x, :me_z]
    @test [(r.response, r.predictor, r.scale) for r in plan.responses] ==
        [(:y, :mu, :s), (:x, :me_x, 0.5), (:z, :me_z, 0.25)]
    # ... but an exact duplicate fails closed (SB shares one latent per
    # model; the second would double-count the evidence).
    @test_throws "a second `me(x)` term" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + me(x, 0.5) + me(x, 0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # The same source on a second predictor fails closed too (SB shares
    # the one latent across predictors as well).
    @test_throws "a second `me(x)` term" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + me(x, 0.5)
        nu ~ 1 + me(x, 0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
        n ~ Normal(nu, s)
    end)
    # The latent needs no sibling coefficient: `0 + me(x)` is a valid
    # single-summand scaled design (`b .* me_x` classifies thin-layer
    # side, unlike the scan summand).
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + me(x, 0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test [t.kind for t in only(plan.predictors).terms] == [:me]
    # A non-positive sd stays closed (shared preparation).
    @test_throws "requires finite numeric sd" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + me(x, 0.0)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws "requires finite numeric sd" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + me(x, -0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # A non-numeric source column stays closed (shared preparation).
    dfg = merge(df, (; gi=["a", "a", "b", "b", "c", "c"]))
    @test_throws "numeric observations" BRM._brm_rk_plan(@brm dfg begin
        mu ~ 1 + me(gi, 0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # A sampled parameter holding the latent name fails closed.
    @test_throws "already taken" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + me(x, 0.5)
        me_x ~ Normal(0, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # The observed column doubles as an ordinary term (SB binds it once;
    # both the data column and the latent take betas).
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + me(x, 0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test [t.kind for t in only(plan.predictors).terms] ==
        [:intercept, :continuous, :me]
end

@testset "exact gp iso plan shape" begin
    brmi = @brm df begin
        mu ~ 1 + gp(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    predictor = only(plan.predictors)
    @test [t.kind for t in predictor.terms] == [:intercept, :gp]
    term = only(t for t in predictor.terms if t.kind === :gp)
    @test term.columns == [:x]
    @test (term.options.rho, term.options.sigma, term.options.z,
        term.options.f) == (:rho_gp, :sigma_gp, :z_gp, :f_gp)
    @test term.options.jitter == 1e-9
    @test term.options.rho_param.family === :LogNormal
    @test term.options.rho_param.args == (0.0, 1.0)
    @test term.options.sigma_param.family === :LogNormal
    @test term.options.sigma_param.args == (0.0, 1.0)
    @test plan.columns[:x] == df.x
    # Hypers ride the term, not plan.parameters (topo order: the AST
    # preamble emits them before the predictor affine).
    @test [p.name for p in plan.parameters] == [:s]
    # Explicit hyper priors lower onto the term's sampled params.
    brmi = @brm df begin
        mu ~ 1 + gp(x)
        length_scale(:, gp(x)) ~ Gamma(2, 1)
        sd(:, gp(x)) ~ Exponential(2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    term = only(t for t in only(plan.predictors).terms if t.kind === :gp)
    @test (term.options.rho_param.family, term.options.rho_param.args) ==
        (:Gamma, (2.0, 1.0))
    @test (term.options.sigma_param.family,
        term.options.sigma_param.args) == (:Exponential, (2.0,))
    # Generated names disambiguate against user parameters.
    brmi = @brm df begin
        mu ~ 1 + gp(x)
        rho_gp ~ Normal(0, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    term = only(t for t in only(plan.predictors).terms if t.kind === :gp)
    @test (term.options.rho, term.options.sigma, term.options.z,
        term.options.f) == (:rho_gp2, :sigma_gp2, :z_gp2, :f_gp2)
    # A gp-only predictor plans (no ordinary terms required).
    brmi = @brm df begin
        mu ~ gp(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test [t.kind for t in only(plan.predictors).terms] == [:gp]
end

@testset "fail closed: exact gp sequenced spellings" begin
    # Aniso, multi-axis, periodic, and Uniform hyper priors stay closed
    # until the thin-layer surface sequences them.
    @test_throws "anisotropic or multi-axis" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + gp(x, z; iso=false)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws "anisotropic or multi-axis" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + gp(x, z)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws "cov=:periodic" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + gp(x; cov=:periodic, period=1.0)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws "Uniform" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + gp(x)
        length_scale(:, gp(x)) ~ Uniform(0.5, 2.0)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # A gp-only sequenced spelling fails closed with RK attribution, not
    # the offset-only internal error.
    @test_throws "anisotropic or multi-axis" BRM._brm_rk_plan(@brm df begin
        mu ~ gp(x, z)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
end

@testset "hsgp plan shape" begin
    # The shared 6-row df suffices: no minimum-axis fit like `s(x)`.
    brmi = @brm df begin
        mu ~ 1 + hsgp(x; k=4)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    predictor = only(plan.predictors)
    @test [t.kind for t in predictor.terms] == [:intercept, :hsgp]
    term = only(t for t in predictor.terms if t.kind === :hsgp)
    @test term.columns == [:x]
    @test (term.options.id, term.options.k, term.options.c,
        term.options.iso) == (:hsgp_x, 4, 1.5, true)
    @test plan.columns[:x] == df.x
    # HSGP parameters are thin-layer-owned: nothing lands in
    # plan.parameters for the smooth itself.
    @test [p.name for p in plan.parameters] == [:s]
    # Defaults ride the declaration (k=20, c=1.5, iso=true).
    brmi = @brm df begin
        mu ~ 1 + hsgp(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    term = only(t for t in only(plan.predictors).terms if t.kind === :hsgp)
    @test (term.options.k, term.options.c, term.options.iso) ==
        (20, 1.5, true)
    # Aniso multi-axis: per-axis tuples over both axes.
    brmi = @brm df begin
        mu ~ 1 + hsgp(x, z; k=(4, 3), c=(1.5, 2.0), iso=false)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    term = only(t for t in only(plan.predictors).terms if t.kind === :hsgp)
    @test term.columns == [:x, :z]
    @test (term.options.id, term.options.k, term.options.c,
        term.options.iso) == (:hsgp_x_z, (4, 3), (1.5, 2.0), false)
    @test plan.columns[:z] == df.z
    # Iso multi-axis shares one length scale (scalar broadcasts).
    brmi = @brm df begin
        mu ~ 1 + hsgp(x, z; k=4)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    term = only(t for t in only(plan.predictors).terms if t.kind === :hsgp)
    @test (term.options.k, term.options.c, term.options.iso) ==
        ((4, 4), (1.5, 1.5), true)
    # One id per smooth occurrence: a second smooth in the same
    # predictor takes its own axis-derived id.
    brmi = @brm df begin
        mu ~ 1 + hsgp(x; k=4) + hsgp(z; k=3)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    ids = [t.options.id for t in only(plan.predictors).terms
        if t.kind === :hsgp]
    @test ids == [:hsgp_x, :hsgp_z]
    # ... and the same smooth in a second predictor serializes instead
    # of colliding (exactly-one-use linkage per declaration).
    brmi = @brm df begin
        mu ~ 1 + hsgp(x; k=4)
        nu ~ 1 + hsgp(x; k=4)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
        z ~ Normal(nu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    ids = Set(t.options.id for p in plan.predictors for t in p.terms
        if t.kind === :hsgp)
    @test ids == Set([:hsgp_x, :hsgp_x_2])
    # Generated ids disambiguate against user parameters.
    brmi = @brm df begin
        mu ~ 1 + hsgp(x; k=4)
        hsgp_x ~ Normal(0, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    term = only(t for t in only(plan.predictors).terms if t.kind === :hsgp)
    @test term.options.id == :hsgp_x_2
    # A smooth-only predictor plans (no ordinary terms required).
    brmi = @brm df begin
        mu ~ hsgp(x; k=4)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test [t.kind for t in only(plan.predictors).terms] == [:hsgp]
end

@testset "fail closed: hsgp sequenced spellings" begin
    # Hyper overrides stay closed until the thin-layer surface
    # sequences them (self-priored LogNormal(0, 1) defaults only).
    # The gate precedes the basis fit, so the shared 6-row df suffices.
    @test_throws "hyper priors" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + hsgp(x; k=4)
        length_scale(:, hsgp(x)) ~ Gamma(2, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test_throws "hyper priors" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + hsgp(x; k=4)
        sd(:, hsgp(x)) ~ Exponential(2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Grouped weights stay closed (ungrouped surface only).
    @test_throws "by=" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + hsgp(x; k=4, by=g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Periodic stays closed (exp_quad surface only).
    @test_throws "cov=:periodic" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + hsgp(x; cov=:periodic, period=1.0)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Partial centering stays closed (non-centered surface only).
    @test_throws "partially-centered" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + hsgp(x; k=4, centeredness=0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Explicit domains stay closed (the surface fits the boundary
    # from raw columns).
    @test_throws "domain=" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + hsgp(x; k=4, domain=(-2.0, 2.0))
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Orthogonalization stays closed (raw tensor-product basis only).
    @test_throws "orthogonal" BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + hsgp(x; k=4, orthogonal_to=:linear)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    # Latent axes stay closed (raw data columns only). The latent
    # axis needs its own observation, else the shared seam fails the
    # axis predictor's row axis before the RK gate is reached.
    ldf = merge(df, (; x_obs=[0.3, 0.5, 0.4, 0.6, 0.5, 0.7]))
    @test_throws "model-derived" BRM._brm_rk_plan(@brm ldf begin
        xlat ~ 1
        xob_sd ~ Exponential(1)
        x_obs ~ Normal(xlat, xob_sd)
        mu ~ 1 + hsgp(xlat; k=4, domain=(0.0, 2.0))
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
    # NOTE: `Normal(mu, s)` with `s ~ 1 + x` lived here until the
    # distributional lift admitted a scale predictor; it now plans in
    # "distributional scale/shape predictors".
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ Normal(0, s)
    end)
end

# Slice-2 demand battery: the docs/examples corpus families/links that
# neither group A nor the landed leveled slice admit. Each entry asserts
# fail-closed TODAY; later lands flip them one by one into plan-shape
# testsets above. Group labels match the demand inventory shared with
# rk:brm (A: links + Beta, now admitted — see "slice-2 group-A plan
# shapes"; B: robust/survival; C: multivariate/mixture — CategoricalLogit
# and Ordinal are admitted by the leveled slice, see "leveled plan
# shapes", so they are not listed here).
@testset "fail closed: slice-2 demand (not yet admitted)" begin
    # Group B: LocationScale-TDist (t_regression.jl).
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        nu ~ Gamma(2, 0.1)
        y ~ LocationScale(mu, s, TDist(nu))
    end)
    # Group B: censored Weibull/Exponential (surv_cont.jl, surv_model.jl).
    dfu = merge(df, (; u=[1.0, Inf, 0.8, Inf, 1.2, Inf]))
    @test_throws ErrorException BRM._brm_rk_plan(@brm dfu begin
        log(mu) ~ 1 + x
        alpha ~ Exponential(1)
        y ~ censored(Weibull(alpha, mu); upper=u)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm dfu begin
        log(mu) ~ 1 + x
        y ~ censored(Exponential(mu); upper=u)
    end)
    # Group C: hurdle / zero-inflated Poisson (hurdle_only.jl, zip.jl).
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        log(lambda) ~ 1 + x
        logit(p_zero) ~ 1 + x
        c ~ HurdlePoisson(lambda, p_zero)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        log(lambda) ~ 1 + x
        logit(zi) ~ 1 + x
        c ~ ZeroInflatedPoisson(lambda, zi)
    end)
    # Group C: InverseGaussian (wald_only.jl).
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        eta ~ 1 + x
        lam ~ LogNormal(-0.3, 1.0)
        y ~ InverseGaussian(exp(eta), lam)
    end)
    # Group C: SkewDoubleExponential (quantile.jl).
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        tau ~ Beta(2, 2)
        y ~ SkewDoubleExponential(mu, s, tau)
    end)
    # Group C: CircularVonMises (circular.jl).
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        log(kappa) ~ 1
        y ~ CircularVonMises(mu, kappa; interval=(-pi, pi))
    end)
end

# Slice-2 group-A scope edges (per the rk:brm scoping answer): no
# weights/evidence on the new triples, logit-only Beta mu, no inline
# cloglog, identical kappa in both Beta positions, and no
# identity-predictor `Bernoulli(probit(eta))` spelling (LHS-link form
# only — the thin-layer link words are peel-and-discard).
@testset "fail closed: group-A scope edges" begin
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        probit(p) ~ 1 + x
        b ~ weighted(Bernoulli(p), fweights(n))
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm dfp begin
        logit(mu) ~ 1 + x
        kappa ~ Gamma(2.0, 1000.0)
        prop ~ weighted(Beta(mu * kappa, (1 - mu) * kappa), fweights(n))
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        cloglog(p) ~ 1 + x
        b ~ truncated(Binomial(h, p); lower=0, upper=2)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm dfp begin
        probit(mu) ~ 1 + x
        kappa ~ Gamma(2.0, 1000.0)
        prop ~ Beta(mu * kappa, (1 - mu) * kappa)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        eta ~ 1 + x
        b ~ Bernoulli(1 - exp(-exp(eta)))
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm dfp begin
        logit(mu) ~ 1 + x
        kappa ~ Gamma(2.0, 1000.0)
        kappa2 ~ Gamma(2.0, 1000.0)
        prop ~ Beta(mu * kappa, (1 - mu) * kappa2)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm dfp begin
        logit(mu) ~ 1 + x
        kappa ~ Gamma(2.0, 1000.0)
        prop ~ Beta(mu + kappa, (1 - mu) * kappa)
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        eta ~ 1 + x
        b ~ Bernoulli(probit(eta))
    end)
    @test_throws ErrorException BRM._brm_rk_plan(@brm dfp begin
        logit(mu) ~ 1 + x
        kappa ~ Gamma(2.0, 1000.0)
        prop ~ Beta(logistic(mu) * kappa, (1 - logistic(mu)) * kappa)
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

@testset "distributional scale/shape predictors" begin
    # Log-link scale: the flagship `log(sigma) ~ ...` + bare `sigma` spelling.
    brmi = @brm df begin
        mu ~ 1 + x
        log(sigma) ~ 1 + z
        y ~ Normal(mu, sigma)
    end
    plan = BRM._brm_rk_plan(brmi)
    likelihood = only(plan.responses)
    @test (likelihood.family, likelihood.link) === (:gaussian, :identity)
    @test likelihood.predictor === :mu
    @test isnothing(likelihood.scale)
    @test likelihood.scale_predictor === :sigma
    @test [p.name for p in plan.predictors] == [:mu, :sigma]
    @test [p.link for p in plan.predictors] == [:identity, :log]
    @test isempty(plan.parameters)
    @test sort!([(p.predictor, p.addressee)
                 for p in plan.population_priors]) ==
        [(:mu, :Intercept), (:mu, :x), (:sigma, :Intercept), (:sigma, :z)]
    sigma_spec = only(p for p in plan.predictors if p.name === :sigma)
    @test [t.kind for t in sigma_spec.terms] == [:intercept, :continuous]

    # Identity-link scale is admitted (SB emits it raw).
    brmi = @brm df begin
        mu ~ 1 + x
        sigma ~ 1 + z
        y ~ Normal(mu, sigma)
    end
    likelihood = only(BRM._brm_rk_plan(brmi).responses)
    @test isnothing(likelihood.scale)
    @test likelihood.scale_predictor === :sigma

    # NB2 dispersion as a linear predictor.
    brmi = @brm df begin
        log(mu) ~ 1 + x
        log(phi) ~ 1 + z
        c ~ NegativeBinomial2(mu, phi)
    end
    likelihood = only(BRM._brm_rk_plan(brmi).responses)
    @test (likelihood.family, likelihood.link) === (:nb2_log, :log)
    @test likelihood.predictor === :mu
    @test isnothing(likelihood.scale)
    @test likelihood.scale_predictor === :phi

    # Gamma shape as a linear predictor (same predictor in both positions).
    brmi = @brm df begin
        log(mu) ~ 1 + x
        log(alpha) ~ 1 + x
        z ~ Gamma(alpha, mu / alpha)
    end
    likelihood = only(BRM._brm_rk_plan(brmi).responses)
    @test (likelihood.family, likelihood.link) === (:gamma_log, :log)
    @test likelihood.predictor === :mu
    @test isnothing(likelihood.scale)
    @test likelihood.scale_predictor === :alpha
end

@testset "distributional fail-closed" begin
    # The scale slot naming the location is degenerate.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        y ~ Normal(mu, mu)
    end)
    # Deterministic wrappers spell as an LP link, not at the use-site.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        log_sigma ~ 1 + z
        y ~ Normal(mu, exp(log_sigma))
    end)
    # A third predictor behind the two slots stays out.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + tau
        log(sigma) ~ 1 + z
        tau ~ 1 + x
        y ~ Normal(mu, sigma)
    end)
    # Binomial trials cannot be a predictor.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        logit(p) ~ 1 + x
        eta2 ~ 1 + z
        b ~ Binomial(eta2, p)
    end)
end

# Leveled emission (thin-layer contract 0178bfe2): categorical-logit,
# ordered-logit, ordinal, multinomial, and categorical plan shapes, plus
# the fail-closed battery for the new surface.
function rk_plan_error(formula)
    try
        BRM._brm_rk_plan(formula)
        nothing
    catch error
        error
    end
end

@testset "leveled plan shapes" begin
    # Reference-coded categorical: K−1 etas, class 1 implicit zero.
    # c = [2,1,3,2,4,3] has 4 levels, already 1..4 contiguous.
    plan = BRM._brm_rk_plan(@brm df begin
        eta1 ~ 1 + x
        eta2 ~ 1 + x
        eta3 ~ 1 + x
        c ~ CategoricalLogit(eta1, eta2, eta3)
    end)
    spec = only(plan.responses)
    @test (spec.family, spec.link) === (:categorical_logit, :logit)
    @test spec.predictor === :eta1
    @test spec.extra_predictors == [:eta2, :eta3]
    @test spec.n_levels == 4
    @test plan.columns[:c] == [2, 1, 3, 2, 4, 3]
    @test sort!([p.name for p in plan.predictors]) ==
        [:eta1, :eta2, :eta3]
    # Cumulative-logit ordinal: implicit ordered cutpoints, raw coding.
    plan = BRM._brm_rk_plan(@brm df begin
        eta ~ 1 + x
        c ~ OrderedLogistic(eta)
    end)
    spec = only(plan.responses)
    @test (spec.family, spec.link) === (:ordered_logit, :logit)
    @test spec.n_levels == 4
    @test spec.thresholds === :c_cutpoints
    cut = only(plan.vector_parameters)
    @test (cut.name, cut.family, cut.args, cut.size) ==
        (:c_cutpoints, :ordered_normal, (0.0, 1.0), 3)
    @test plan.columns[:c] == [2, 1, 3, 2, 4, 3]
    # General typed ordinal, plain (no extras).
    plan = BRM._brm_rk_plan(@brm df begin
        eta ~ 0 + x
        c ~ Ordinal(StoppingRatio(), ProbitLink(), eta)
    end)
    spec = only(plan.responses)
    @test (spec.family, spec.link) === (:ordinal, :probit)
    @test spec.ordinal_structure === :stopping
    @test spec.thresholds === :c_thresholds
    @test only(plan.vector_parameters).family === :vector_normal
    @test only(plan.vector_parameters).size == 3
    # Extras fields stay defaulted (any extras fail closed at plan —
    # thin-layer surface gap; see the fail-closed battery).
    @test spec.discrimination === nothing
    @test isempty(spec.threshold_columns)
    @test spec.threshold_coefs === nothing
    # Shared-simplex multinomial: lead + tail count columns.
    mdf = (;
        df...,
        obs=[3 1 0; 2 2 1; 0 0 5; 1 1 1; 4 0 0; 2 1 2],
        n=[4, 5, 5, 3, 4, 5],
    )
    plan = BRM._brm_rk_plan(@brm mdf begin
        s ~ Dirichlet(3, 1.0)
        obs ~ Multinomial(n, s)
    end)
    spec = only(plan.responses)
    @test (spec.family, spec.link) === (:multinomial, :identity)
    @test spec.predictor === :s
    @test spec.trials === :n
    @test spec.n_levels == 3
    @test spec.count_columns == [:obs_count_2, :obs_count_3]
    @test plan.columns[:obs] == [3, 2, 0, 1, 4, 2]
    @test plan.columns[:obs_count_2] == [1, 2, 0, 1, 0, 1]
    @test plan.columns[:obs_count_3] == [0, 1, 5, 1, 0, 2]
    simplex = only(plan.vector_parameters)
    @test (simplex.name, simplex.family, simplex.size) ==
        (:s, :simplex_dirichlet, 3)
    @test only(simplex.args) == [1.0, 1.0, 1.0]
    # Plain categorical over simplex probs; b = [0,1,...] recodes to 1..2.
    plan = BRM._brm_rk_plan(@brm df begin
        s ~ Dirichlet([2.0, 5.0])
        b ~ Categorical(s)
    end)
    spec = only(plan.responses)
    @test (spec.family, spec.link) === (:categorical, :identity)
    @test spec.predictor === :s
    @test spec.n_levels == 2
    @test plan.columns[:b] == [1, 2, 1, 2, 2, 1]
    @test only(only(plan.vector_parameters).args) == [2.0, 5.0]
end

@testset "ordinal extras plan shapes" begin
    # Literal discrimination (either structure).
    plan = BRM._brm_rk_plan(@brm df begin
        eta ~ 0 + x
        c ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=2.0)
    end)
    spec = only(plan.responses)
    @test (spec.family, spec.link) === (:ordinal, :logit)
    @test spec.ordinal_structure === :cumulative
    @test spec.discrimination == 2.0
    @test isempty(spec.threshold_columns)
    @test spec.threshold_coefs === nothing
    @test spec.thresholds === :c_thresholds
    @test length(plan.vector_parameters) == 1
    # Column discrimination crosses the column.
    plan = BRM._brm_rk_plan(@brm df begin
        eta ~ 0 + x
        c ~ Ordinal(Cumulative(), ProbitLink(), eta; discrimination=z)
    end)
    spec = only(plan.responses)
    @test spec.discrimination === :z
    @test plan.columns[:z] == df.z
    # Modeled discrimination: the log-link predictor plans with terms
    # and priors (the AST skips it; the extension translates it).
    plan = BRM._brm_rk_plan(@brm df begin
        eta ~ 0 + x
        log(disc) ~ 1 + x
        c ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=disc)
    end)
    spec = only(plan.responses)
    @test spec.discrimination === :disc
    @test sort!([p.name for p in plan.predictors]) == [:disc, :eta]
    dspec = only(p for p in plan.predictors if p.name === :disc)
    @test dspec.link === :log
    @test [(t.kind, t.addressee) for t in dspec.terms] ==
        [(:intercept, :Intercept), (:continuous, :x)]
    @test sort!([(p.predictor, p.addressee) for p in plan.population_priors
        if p.predictor === :disc]) ==
        [(:disc, :Intercept), (:disc, :x)]
    # Modeled grouping scale (the `log(disc) ~ group` recipe): factor
    # terms plan with an explicit block prior.
    plan = BRM._brm_rk_plan(@brm df begin
        eta ~ 0 + x
        log(disc) ~ 0 + g
        effect(disc, g) ~ Normal(0, 1)
        c ~ Ordinal(StoppingRatio(), CloglogLink(), eta; discrimination=disc)
    end)
    spec = only(plan.responses)
    @test spec.discrimination === :disc
    dspec = only(p for p in plan.predictors if p.name === :disc)
    @test [(t.kind, t.addressee) for t in dspec.terms] == [(:factor, :g)]
    @test dspec.terms[1].options.coding === :fullrank
    @test only([p for p in plan.population_priors
        if p.predictor === :disc]).addressee === :g
    # per_threshold p=1 and p=2 (stopping only): design columns cross
    # and the stage-major coef vector plans at (K-1)*p (c has K=4).
    plan = BRM._brm_rk_plan(@brm df begin
        eta ~ 0 + x
        c ~ Ordinal(StoppingRatio(), LogitLink(), eta; per_threshold=(z,))
    end)
    spec = only(plan.responses)
    @test spec.threshold_columns == [:z]
    @test spec.threshold_coefs === :c_threshold_beta
    @test plan.columns[:z] == df.z
    coef = only(v for v in plan.vector_parameters
        if v.name === :c_threshold_beta)
    @test (coef.family, coef.args, coef.size) ==
        (:vector_normal, (0.0, 1.0), 3)
    plan = BRM._brm_rk_plan(@brm df begin
        eta ~ 0 + x
        c ~ Ordinal(StoppingRatio(), ProbitLink(), eta;
            per_threshold=(z, x))
    end)
    spec = only(plan.responses)
    @test spec.threshold_columns == [:z, :x]
    coef = only(v for v in plan.vector_parameters
        if v.name === :c_threshold_beta)
    @test coef.size == 6
    # Full combination: stopping + modeled scale + per_threshold.
    plan = BRM._brm_rk_plan(@brm df begin
        eta ~ 0 + x
        log(disc) ~ 1 + x
        c ~ Ordinal(StoppingRatio(), LogitLink(), eta;
            discrimination=disc, per_threshold=(z,))
    end)
    spec = only(plan.responses)
    @test spec.discrimination === :disc
    @test spec.threshold_columns == [:z]
    @test spec.threshold_coefs === :c_threshold_beta
end

@testset "fail closed: leveled surfaces" begin
    # K=1 categorical is inexpressible (decision 0dteta6).
    single = rk_plan_error(@brm df begin
        k1 ~ CategoricalLogit()
    end)
    @test single isa ErrorException
    @test occursin("single-level", single.msg)
    @test occursin("inexpressible", single.msg)
    # Arity: b has 2 levels but 2 etas (expects K=3).
    arity = rk_plan_error(@brm df begin
        eta1 ~ 1 + x
        eta2 ~ 1 + x
        b ~ CategoricalLogit(eta1, eta2)
    end)
    @test arity isa ErrorException
    @test occursin("observed 2 outcome levels but received 2", arity.msg)
    # Zero-arg with K>=2 is an arity mismatch, not the K=1 case.
    noetas = rk_plan_error(@brm df begin
        c ~ CategoricalLogit()
    end)
    @test noetas isa ErrorException
    @test occursin("received 0 non-reference predictors", noetas.msg)
    # Repeated and non-predictor arguments.
    dup = rk_plan_error(@brm df begin
        eta1 ~ 1 + x
        eta2 ~ 1 + x
        c ~ CategoricalLogit(eta1, eta1, eta2)
    end)
    @test dup isa ErrorException
    @test occursin("repeats a predictor", dup.msg)
    datap = rk_plan_error(@brm df begin
        eta1 ~ 1 + x
        eta2 ~ 1 + x
        c ~ CategoricalLogit(eta1, eta2, x)
    end)
    @test datap isa ErrorException
    @test occursin("not a declared linear predictor", datap.msg)
    nonid = rk_plan_error(@brm df begin
        logit(p) ~ 1 + x
        b ~ CategoricalLogit(p)
    end)
    @test nonid isa ErrorException
    @test occursin("identity-link predictors", nonid.msg)
    # Ordered gaps and non-integers.
    gap = rk_plan_error(@brm (; df..., g3=[1, 1, 3, 3, 1, 3]) begin
        eta ~ 1 + x
        g3 ~ OrderedLogistic(eta)
    end)
    @test gap isa ErrorException
    @test occursin("non-contiguous", gap.msg)
    floaty = rk_plan_error(@brm df begin
        eta ~ 1 + x
        cf ~ OrderedLogistic(eta)
    end)
    @test floaty isa ErrorException
    @test occursin("expects integer outcome data", floaty.msg)
    # Ordinal shape errors.
    fixed = rk_plan_error(@brm df begin
        eta ~ 1 + x
        c ~ Ordinal(Cumulative(), LogitLink(), eta)
    end)
    @test fixed isa ErrorException
    @test occursin("cannot include a fixed intercept", fixed.msg)
    # Ordinal extras fail-closed battery (admitted shapes plan in
    # "ordinal extras plan shapes").
    cumstage = rk_plan_error(@brm df begin
        eta ~ 0 + x
        c ~ Ordinal(Cumulative(), LogitLink(), eta; per_threshold=(z,))
    end)
    @test cumstage isa ErrorException
    @test occursin("`StoppingRatio()` only", cumstage.msg)
    @test occursin("non-monotone", cumstage.msg)
    # Non-positive/non-finite literal scales: the helper pins each
    # spelling directly (zero, negative, infinite, NaN), and the surface
    # pins the zero literal end to end.
    for bad in (0.0, -1.0, Inf, NaN)
        literr = try
            BRM._rk_ordinal_discrimination(bad, :c, BRM._RKPredictorSpec[])
            nothing
        catch error
            error
        end
        @test literr isa ErrorException
        @test occursin("finite and strictly positive", literr.msg)
    end
    @test BRM._rk_ordinal_discrimination(2.0, :c,
        BRM._RKPredictorSpec[]) == (2.0, Symbol[])
    zerodisc = rk_plan_error(@brm df begin
        eta ~ 0 + x
        c ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=0.0)
    end)
    @test zerodisc isa ErrorException
    @test occursin("finite and strictly positive", zerodisc.msg)
    # x carries negatives — not a discrimination column.
    negcol = rk_plan_error(@brm df begin
        eta ~ 0 + x
        c ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=x)
    end)
    @test negcol isa ErrorException
    @test occursin("finite positive values", negcol.msg)
    # A sampled parameter is not a scale (SB takes it; the thin layer
    # takes literals, data columns, and log-link predictors only).
    sampled = rk_plan_error(@brm df begin
        eta ~ 0 + x
        d ~ Normal(0, 1)
        c ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=d)
    end)
    @test sampled isa ErrorException
    @test occursin("neither a declared `log()` linear predictor", sampled.msg)
    # An identity-link predictor is not a scale either.
    nonlog = rk_plan_error(@brm df begin
        eta ~ 0 + x
        disc ~ 0 + x
        c ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=disc)
    end)
    @test nonlog isa ErrorException
    @test occursin("must be a `log()` linear predictor", nonlog.msg)
    # Random effects are out of slice for a modeled scale.
    ranefscale = rk_plan_error(@brm df begin
        eta ~ 0 + x
        log(disc) ~ 0 + x + (1 | g)
        c ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=disc)
    end)
    @test ranefscale isa ErrorException
    @test occursin("admits population terms only", ranefscale.msg)
    # A modeled scale feeds no other response slot (here the Poisson
    # location legitimately takes the log-link predictor).
    shared = rk_plan_error(@brm (; df..., counts=[1, 2, 1, 3, 2, 1]) begin
        eta ~ 0 + x
        log(disc) ~ 0 + x
        c ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=disc)
        counts ~ Poisson(disc)
    end)
    @test shared isa ErrorException
    @test occursin("also feeds a location", shared.msg)
    # per_threshold shape errors.
    nontuple = rk_plan_error(@brm df begin
        eta ~ 0 + x
        c ~ Ordinal(StoppingRatio(), LogitLink(), eta; per_threshold=z)
    end)
    @test nontuple isa ErrorException
    @test occursin("expects a tuple", nontuple.msg)
    noncol = rk_plan_error(@brm df begin
        eta ~ 0 + x
        c ~ Ordinal(StoppingRatio(), LogitLink(), eta; per_threshold=(eta,))
    end)
    @test noncol isa ErrorException
    @test occursin("only raw numeric data columns", noncol.msg)
    dupes = rk_plan_error(@brm df begin
        eta ~ 0 + x
        c ~ Ordinal(StoppingRatio(), LogitLink(), eta; per_threshold=(z, z))
    end)
    @test dupes isa ErrorException
    @test occursin("repeat a column", dupes.msg)
    short = rk_plan_error(@brm (; df..., w=[0.5, 0.25]) begin
        eta ~ 0 + x
        c ~ Ordinal(StoppingRatio(), LogitLink(), eta; per_threshold=(w,))
    end)
    @test short isa ErrorException
    @test occursin("rows; outcome", short.msg)
    infty = rk_plan_error(@brm (; df..., w=[0.1, 0.2, Inf, 0.4, 0.5, 0.6]) begin
        eta ~ 0 + x
        c ~ Ordinal(StoppingRatio(), LogitLink(), eta; per_threshold=(w,))
    end)
    @test infty isa ErrorException
    @test occursin("non-finite", infty.msg)
    # Unknown Ordinal keywords die at `@brm` formula validation (before
    # any backend); the planner's own keyword check is defense-in-depth.
    badkw = try
        (@brm df begin
            eta ~ 0 + x
            c ~ Ordinal(Cumulative(), LogitLink(), eta; foo=1)
        end)
        nothing
    catch error
        error
    end
    @test badkw isa ErrorException
    @test occursin("unsupported keyword", badkw.msg)
    # Simplex source errors.
    nonsimp = rk_plan_error(@brm df begin
        s ~ Exponential(1)
        b ~ Categorical(s)
    end)
    @test nonsimp isa ErrorException
    @test occursin("must be a `Dirichlet`-sampled parameter", nonsimp.msg)
    datasimp = rk_plan_error(@brm df begin
        c ~ Categorical(z)
    end)
    @test datasimp isa ErrorException
    @test occursin("cannot be a data column", datasimp.msg)
    badalpha = rk_plan_error(@brm df begin
        s ~ Dirichlet([1.0, -2.0])
        b ~ Categorical(s)
    end)
    @test badalpha isa ErrorException
    @test occursin("strictly positive", badalpha.msg)
    badsym = rk_plan_error(@brm df begin
        s ~ Dirichlet(0, 1.0)
        b ~ Categorical(s)
    end)
    @test badsym isa ErrorException
    @test occursin("positive integer dimension", badsym.msg)
    sizemix = rk_plan_error(@brm df begin
        s ~ Dirichlet(2, 1.0)
        c ~ Categorical(s)
    end)
    @test sizemix isa ErrorException
    @test occursin("sizes must agree", sizemix.msg)
    unused = rk_plan_error(@brm df begin
        mu ~ 1 + x
        s ~ Dirichlet(2, 1.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end)
    @test unused isa ErrorException
    @test occursin("no multinomial/categorical response uses it", unused.msg)
    # Multinomial shape errors.
    badsums = rk_plan_error(@brm (;
        df...,
        obs=[3 1 0; 2 2 1; 0 0 5; 1 1 1; 4 0 0; 2 1 2],
        n=[4, 5, 5, 3, 4, 4],
    ) begin
        s ~ Dirichlet(3, 1.0)
        obs ~ Multinomial(n, s)
    end)
    @test badsums isa ErrorException
    @test occursin("must sum to their trials", badsums.msg)
    nomatrix = rk_plan_error(@brm df begin
        s ~ Dirichlet(3, 1.0)
        c ~ Multinomial(n, s)
    end)
    @test nomatrix isa ErrorException
    @test occursin("needs an n×K integer count matrix", nomatrix.msg)
    # Evidence stays Gaussian/Poisson-only.
    ev = rk_plan_error(@brm df begin
        eta1 ~ 1 + x
        eta2 ~ 1 + x
        eta3 ~ 1 + x
        c ~ truncated(CategoricalLogit(eta1, eta2, eta3), 1, 3)
    end)
    @test ev isa ErrorException
    @test occursin("evidence", ev.msg)
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
    @test_throws "decomposes nothing" BRM._brm_rk_plan(@brm df begin
        mu ~ 0 + offset(z)
        effect(mu, :) ~ r2d2(R2=Normal(0.5, 0.2), tau_bsv=0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
end

@testset "ranef ID bucket plan shape" begin
    brmi = @brm df begin
        mu ~ 1 + x + (1 + x | ID | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test length(plan.ranef_buckets) == 1
    bucket = only(plan.ranef_buckets)
    @test bucket.id === :ID
    @test bucket.group === :g
    @test bucket.kind === :correlated
    @test bucket.lkj_eta == 1.0
    @test bucket.label === :bucket_ID_g
    margins = [(m.predictor, m.coefficient) for m in bucket.margins]
    @test margins == [(:mu, :Intercept), (:mu, :x)]
    @test [m.z.kind for m in bucket.margins] == [:ones, :column]
    @test bucket.slices == [(:mu, 1:2)]
    # ranefcoefnames order agreement with SB.
    sb_margins = [(m.predictor, m.coefficient)
        for m in BRM.ranefcoefnames(brmi, :ID)]
    @test margins == sb_margins
    terms = only(plan.predictors).terms
    @test [t.kind for t in terms] == [:intercept, :continuous, :ranef_gather]
    gather = terms[3]
    @test gather.columns == [:g]
    @test gather.options == (bucket_id=:ID, bucket_group=:g)
    @test gather.addressee === :r_mu_ID_g
    @test gather.label === :r_mu_ID_g
    @test sort!([p.addressee for p in plan.population_priors]) ==
        [:Intercept, :x]
    @test plan.columns[:g] == [1, 1, 2, 2, 3, 3]
    backend = BRM.RKBRMI(brmi, plan, nothing)
    @test sprint(show, backend) ==
        "RKBRMI with 2 population coefficients and 6 observations"
end

@testset "ranef plain bucket kinds" begin
    # (1|g) — :intercept1, no eta. Draws-always per 0yl36fh (interim);
    # SB routes matchable lone intercepts through brm_total.
    intercept = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    ib = only(intercept.ranef_buckets)
    @test ib.id === nothing
    @test ib.kind === :intercept1
    @test ib.label === :bucket_g
    @test isnan(ib.lkj_eta)
    @test [(m.predictor, m.coefficient) for m in ib.margins] ==
        [(:mu, :Intercept)]
    @test ib.slices == [(:mu, 1:1)]
    gather = only(intercept.predictors).terms[end]
    @test gather.kind === :ranef_gather
    @test gather.addressee === :r_mu_g
    @test gather.label === :r_mu_g
    @test gather.options == (bucket_id=nothing, bucket_group=:g)
    # (0+x|g) — :slope1.
    slope = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (0 + x | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    sb = only(slope.ranef_buckets)
    @test sb.id === nothing
    @test sb.kind === :slope1
    @test isnan(sb.lkj_eta)
    @test [(m.predictor, m.coefficient) for m in sb.margins] == [(:mu, :x)]
    # (1+x|g) — :correlated with eta.
    correlated = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 + x | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    cb = only(correlated.ranef_buckets)
    @test cb.id === nothing
    @test cb.kind === :correlated
    @test cb.lkj_eta == 1.0
    @test cb.slices == [(:mu, 1:2)]
end

@testset "ranef categorical slope dummies" begin
    treatment = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 + c | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    tb = only(treatment.ranef_buckets)
    @test tb.kind === :correlated
    @test [m.coefficient for m in tb.margins] ==
        [:Intercept, :c_dummy_2, :c_dummy_3, :c_dummy_4]
    @test [m.z.level for m in tb.margins[2:4]] == [2, 3, 4]
    @test [m.z.column for m in tb.margins[2:4]] == [:c, :c, :c]
    @test treatment.columns[:c] == [2, 1, 3, 2, 4, 3]
    # Non-1..K codes prove value-based (not positional) labels.
    codedf = (; df..., c=[2, 4, 2, 6, 4, 6])
    coded = BRM._brm_rk_plan(@brm codedf begin
        mu ~ 1 + x + (1 + c | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    cb = only(coded.ranef_buckets)
    @test [m.coefficient for m in cb.margins] ==
        [:Intercept, :c_dummy_4, :c_dummy_6]
    @test [m.z.level for m in cb.margins[2:3]] == [4, 6]
    # Intercept-free LHS: per-level coding.
    cellmeans = BRM._brm_rk_plan(@brm codedf begin
        mu ~ 1 + x + (0 + c | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    mb = only(cellmeans.ranef_buckets)
    @test [m.coefficient for m in mb.margins] ==
        [:c_dummy_2, :c_dummy_4, :c_dummy_6]
    # String slopes ride the thin layer's exact-match dummies.
    strings = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 + gs | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    gb = only(strings.ranef_buckets)
    @test [m.coefficient for m in gb.margins] ==
        [:Intercept, :gs_dummy_b, :gs_dummy_c]
    @test [m.z.level for m in gb.margins[2:3]] == ["b", "c"]
end

@testset "ranef continuous interaction margins" begin
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 + x & z | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    bucket = only(plan.ranef_buckets)
    @test bucket.kind === :correlated
    @test [m.coefficient for m in bucket.margins] == [:Intercept, :int_x_x_z]
    cross = bucket.margins[2]
    @test cross.z.kind === :column
    @test cross.z.column === :int_x_x_z
    defs = filter(d -> d.name === :int_x_x_z, plan.derived)
    @test length(defs) == 1
    @test only(defs).expression == Expr(:call, :.*, :x, :z)
    @test only(defs).label === :int_x_x_z
    # Shooter shape: the same cross in population and ranef shares one def.
    both = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + z + x & z + (x + z + x & z | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test count(d -> d.name === :int_x_x_z, both.derived) == 1
    bb = only(both.ranef_buckets)
    @test bb.kind === :correlated
    @test [m.coefficient for m in bb.margins] == [:x, :z, :int_x_x_z]
    # Transformed operands mirror SB's data-materialized crosses.
    staged = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 + zscale(x) & z | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    zb = only(staged.ranef_buckets)
    @test length(zb.margins) == 2
    @test zb.margins[2].z.kind === :column
end

@testset "ranef custom-order categorical grouping fails closed" begin
    catdf = (; df...,
        g=categorical(["b", "b", "a", "a", "c", "c"]; levels=["b", "a", "c"]))
    err = rk_plan_error(@brm catdf begin
        mu ~ 1 + x + (1 + x | ID | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test err isa ErrorException
    @test occursin("custom-ordered", err.msg)
    @test occursin("15a8se2", err.msg)
end

@testset "ranef categorical grouping binds crossed strings" begin
    catdf = (; df...,
        g=categorical(["b", "b", "a", "a", "c", "c"]))
    plan = BRM._brm_rk_plan(@brm catdf begin
        mu ~ 1 + x + (1 + x | ID | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    bucket = only(plan.ranef_buckets)
    @test bucket.group === :g
    @test plan.columns[:g] == ["b", "b", "a", "a", "c", "c"]
    @test !haskey(plan.columns, :g_idx)
    @test bucket.label === :bucket_ID_g
    gather = only(plan.predictors).terms[end]
    @test gather.columns == [:g]
    @test gather.addressee === :r_mu_ID_g
    @test gather.label === :r_mu_ID_g
end

@testset "ranef multi-target ID slices" begin
    mdf = (; x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
             g=[1, 1, 2, 2, 3, 3],
             y1=[0.5, -0.2, 0.1, 0.9, 1.4, 1.1],
             y2=[1.5, 1.2, 1.1, 0.9, 0.4, 0.1])
    plan = BRM._brm_rk_plan(@brm mdf begin
        mu1 ~ 1 + x + (1 | ID | g)
        mu2 ~ 1 + x + (x | ID | g)
        s1 ~ Exponential(1)
        s2 ~ Exponential(1)
        y1 ~ Normal(mu1, s1)
        y2 ~ Normal(mu2, s2)
    end)
    @test length(plan.predictors) == 2
    bucket = only(plan.ranef_buckets)
    @test bucket.id === :ID
    @test bucket.kind === :correlated
    @test [(m.predictor, m.coefficient) for m in bucket.margins] ==
        [(:mu1, :Intercept), (:mu2, :x)]
    @test bucket.slices == [(:mu1, 1:1), (:mu2, 2:2)]
    for predictor in plan.predictors
        gather = predictor.terms[end]
        @test gather.kind === :ranef_gather
        @test gather.options.bucket_id === :ID
    end
end

# Fail-closed gates name the admitted spelling; assert the message, not
# just the throw, so a vacuous parser error cannot stand in for the gate.
function rk_throws_admission(f, needle::String)
    try
        f()
    catch err
        @test err isa ErrorException
        @test occursin("RK backend", sprint(showerror, err))
        @test occursin(needle, sprint(showerror, err))
        return nothing
    end
    @test false
end

@testset "ranef degenerate slopes" begin
    degdf = (; df..., k1=[1, 1, 1, 1, 1, 1])
    # A single-level treatment slope fails closed with guidance (SB drops
    # such slopes silently; the shared slope surface rejects them, so the
    # RK path cannot mirror the drop).
    rk_throws_admission("single observed level") do
        BRM._brm_rk_plan(@brm degdf begin
            mu ~ 1 + x + (1 + k1 | g)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    end
    # Intercept-free, the same column codes its one cell mean.
    cellmean1 = BRM._brm_rk_plan(@brm degdf begin
        mu ~ 1 + x + (0 + k1 | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    bucket = only(cellmean1.ranef_buckets)
    @test bucket.kind === :slope1
    @test [(m.predictor, m.coefficient) for m in bucket.margins] ==
        [(:mu, :k1_dummy_1)]
    # Empty LHS errors (mirrors SB).
    rk_throws_admission("no terms after dropping") do
        BRM._brm_rk_plan(@brm df begin
            mu ~ 1 + x + (0 | g)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    end
end

@testset "ranef K=1 ID stays correlated" begin
    # Unmatched-regime mirror of SB (vacuous 1x1 LKJ); 0yl36fh keeps the
    # matchable lone-intercept regime on draws too (interim).
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 | ID | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    bucket = only(plan.ranef_buckets)
    @test bucket.id === :ID
    @test bucket.kind === :correlated
    @test bucket.lkj_eta == 1.0
    @test [(m.predictor, m.coefficient) for m in bucket.margins] ==
        [(:mu, :Intercept)]
end

@testset "fail closed: ranef draws-regime battery" begin
    # `||` zerocorr is deferred.
    rk_throws_admission("||") do
        BRM._brm_rk_plan(@brm df begin
            mu ~ 1 + x + (0 + x || g)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    end
    # mm(...) is deferred.
    rk_throws_admission("mm(...)") do
        BRM._brm_rk_plan(@brm df begin
            mu ~ 1 + x + (1 | mm(g, h))
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    end
    # gr(...; by=...) is deferred.
    rk_throws_admission("by=") do
        BRM._brm_rk_plan(@brm df begin
            mu ~ 1 + x + (1 | gr(g, by=h))
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    end
    # Categorical `&` operands in a ranef LHS stay deferred: SB codes
    # ranef crosses treatment-coded while the shared recipe is full-rank.
    rk_throws_admission("categorical") do
        BRM._brm_rk_plan(@brm df begin
            mu ~ 1 + x + (1 + x & c | g)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    end
    rk_throws_admission("categorical") do
        BRM._brm_rk_plan(@brm df begin
            mu ~ 1 + x + (1 + c & h | g)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    end
    # The gate recurses through nested crosses.
    rk_throws_admission("categorical") do
        BRM._brm_rk_plan(@brm df begin
            mu ~ 1 + x + (1 + (x & z) & c | g)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    end
    # `factor()` inside ranef `&` names the ranef respell, not the bare
    # column the population path suggests (bare categoricals defer here).
    rk_throws_admission("inside `&` is not in the draws regime") do
        BRM._brm_rk_plan(@brm df begin
            mu ~ 1 + x + (1 + x & factor(h; ref=1) | g)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    end
    # offset() in a ranef LHS is refused (mirrors SB).
    rk_throws_admission("draws regime") do
        BRM._brm_rk_plan(@brm df begin
            mu ~ 1 + x + (1 + offset(z) | g)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    end
    # Transformed slopes are deferred.
    rk_throws_admission("draws regime") do
        BRM._brm_rk_plan(@brm df begin
            mu ~ 1 + x + (1 + zscale(x) | g)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    end
    # sd()/cor() statements are deferred (buckets take defaults).
    rk_throws_admission("sd(...)") do
        BRM._brm_rk_plan(@brm df begin
            mu ~ 1 + x + (1 + x | p | g)
            sd(:, p) ~ Exponential(0.3)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    end
    # Duplicate blocks mirror SB's rejection.
    rk_throws_admission("repeats") do
        BRM._brm_rk_plan(@brm df begin
            mu ~ 1 + x + (1 | g) + (x | g)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    end
    # One ID, two groups is ambiguous (mirrors SB).
    rk_throws_admission("conflicting grouping") do
        BRM._brm_rk_plan(@brm df begin
            mu ~ 1 + x + (1 | ID | g) + (1 | ID | h)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    end
    # factor() slopes are refused (shared slope surface takes bare
    # columns; SB recodes — narrowing, disclosed).
    rk_throws_admission("bare columns") do
        BRM._brm_rk_plan(@brm df begin
            mu ~ 1 + x + (1 + factor(c) | g)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    end
    # Non-1 integers are not ranef terms (mirrors SB).
    rk_throws_admission("unsupported integer") do
        BRM._brm_rk_plan(@brm df begin
            mu ~ 1 + x + (2 | g)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    end
    # Bool slopes need integer codes.
    boolf = (; df..., b=[true, false, true, false, true, false])
    rk_throws_admission("Bool column") do
        BRM._brm_rk_plan(@brm boolf begin
            mu ~ 1 + x + (1 + b | g)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    end
    # Ranef-only predictors stay out of scope.
    rk_throws_admission("has no terms") do
        BRM._brm_rk_plan(@brm df begin
            mu ~ (1 | g)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end)
    end
end

@testset "kernel(...) end-to-end via _brm_rk_plan" begin
    # Phase-1a: a panel kernel is ADMITTED — `_brm_rk_plan` returns a
    # `_RKKernelPlan` (globals as parameters + flattened per-subject columns +
    # the kernel spec) and `_rk_emit_ast` produces the globals plus the subject
    # plate. A grouped (LP-arg) kernel still fails closed (needs RK random
    # effects). See `BayesianRegressionModels:rk:kernel` todo `11b8mr9`.
    kdf = (;
        t=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        dose=[10.0, 20.0],
        obs=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    )
    brmi = @brm kdf begin
        sigma ~ Exponential(1)
        b0 ~ Normal(0, 1)
        pred ~ kernel(t, dose, obs) do ts, d, yy
            mu = b0 .* d .* ts
            yy ~ Normal(mu, sigma)
            mu
        end
    end
    plan = BRM._brm_rk_plan(brmi)
    @test plan isa BRM._RKKernelPlan
    @test plan.kernel.result === :pred
    @test Set(p.name for p in plan.parameters) == Set([:sigma, :b0])
    @test plan.obs.family === :gaussian
    # flattened bind layout: vector slices -> n_sub*T = 6; scalar -> n_sub = 2
    @test length(plan.columns[:t]) == 6
    @test length(plan.columns[:obs]) == 6
    @test length(plan.columns[:dose]) == 2
    @test plan.columns[:t] == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]   # flat T-blocked
    # full program: no submodel defs (a single plate carries no top-level
    # repeated structure); globals as top-level `~`, then the subject
    # plate as the last main-block stmt
    prog = BRM._rk_emit_ast(plan)
    @test prog isa BRM._RKEmittedProgram && isempty(prog.defs)
    @test Meta.isexpr(prog.main, :block)
    stmts = filter(s -> !(s isa LineNumberNode), prog.main.args)
    @test any(s -> Meta.isexpr(s, :call) && s.args[1] === :~ && s.args[2] === :sigma,
              stmts)
    @test any(s -> Meta.isexpr(s, :call) && s.args[1] === :~ && s.args[2] === :b0,
              stmts)
    plate_stmt = last(stmts)
    @test Meta.isexpr(plate_stmt, :call) && plate_stmt.args[1] === :~ &&
        plate_stmt.args[2] === :pred

    # grouped (LP-arg) kernel still fails closed via _brm_rk_plan
    gbrmi = @brm df begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + (1 | pk | g)
        pred ~ kernel(x, log_CL) do xs, lCL
            mu = exp(lCL) .* xs
            yy ~ Normal(mu, sigma)
            mu
        end
        y ~ Normal(pred, sigma)
    end
    @test_throws "per-subject data columns only" BRM._brm_rk_plan(gbrmi)
end

@testset "kernel(...) panel-mode structural extraction" begin
    # Phase-1a: parse a ranef-free panel kernel cell into a `_RKKernelSpec`
    # (params, per-subject data columns, subject count, cell assignments, the
    # in-cell observation, collected result). See `11b8mr9`. `_brm_rk_plan` still
    # fails closed on kernels (guard above); the extraction is tested directly.
    kdf = (;
        t=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        dose=[10.0, 20.0],
        obs=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    )
    brmi = @brm kdf begin
        sigma ~ Exponential(1)
        b0 ~ Normal(0, 1)
        pred ~ kernel(t, dose, obs) do ts, d, yy
            mu = b0 .* d .* ts
            yy ~ Normal(mu, sigma)
            mu
        end
    end
    ops = BRM._rk_kernel_ops(brmi)
    @test length(ops) == 1
    result, rhs = only(ops)
    @test result === :pred
    spec = BRM._rk_kernel_spec(brmi, result, rhs)
    @test spec.result === :pred
    @test spec.subject_count === :kernel_nsub_pred
    @test spec.n_subjects == 2
    @test spec.slice_params == [:ts, :d, :yy]
    @test spec.data_columns == [:t, :dose, :obs]
    @test [p.first for p in spec.assignments] == [:mu]
    @test spec.obs_response === :yy
    @test spec.collected === :mu
    # per-slice kind (vector T-blocked vs scalar-per-subject) + T
    @test spec.slice_kinds == [:vector, :scalar, :vector]  # t, dose, obs
    @test spec.n_timepoints == 3
    @test spec.timepoint_count === :kernel_T_pred

    # AST emission: `pred ~ plate(t, dose, obs; subjects=kernel_nsub_pred) do ...`
    ast = BRM._rk_emit_kernel_ast(spec)
    @test Meta.isexpr(ast, :call) && ast.args[1] === :~ && ast.args[2] === :pred
    do_expr = ast.args[3]
    @test Meta.isexpr(do_expr, :do)
    plate_call = do_expr.args[1]
    @test Meta.isexpr(plate_call, :call) && plate_call.args[1] === :plate
    @test plate_call.args[2] == Expr(:parameters,
        Expr(:kw, :subjects, :kernel_nsub_pred))
    @test collect(plate_call.args[3:end]) == [:t, :dose, :obs]
    lam = do_expr.args[2]
    @test lam.args[1] == Expr(:tuple, :ts, :d, :yy)
    cellbody = filter(s -> !(s isa LineNumberNode), lam.args[2].args)
    @test any(s -> Meta.isexpr(s, :(=)) && s.args[1] === :mu, cellbody)
    # vector obs is emitted DOTTED: `yy .~ Normal.(mu, sigma)`
    obs_stmt = only(s for s in cellbody
                    if Meta.isexpr(s, :call) && s.args[1] === :.~)
    @test obs_stmt.args[2] === :yy
    @test Meta.isexpr(obs_stmt.args[3], :.) && obs_stmt.args[3].args[1] === :Normal
    @test cellbody[end] === :mu

    # non-Gaussian in-cell family is a follow-up (fail closed)
    poisson_brmi = @brm kdf begin
        pred ~ kernel(t, dose, obs) do ts, d, yy
            mu = d .* ts
            yy ~ Poisson(mu)
            mu
        end
    end
    presult, prhs = only(BRM._rk_kernel_ops(poisson_brmi))
    pspec = BRM._rk_kernel_spec(poisson_brmi, presult, prhs)
    @test_throws "only `Normal(location, scale)`" BRM._rk_emit_kernel_ast(pspec)

    # a linear-predictor (grouped) arg is out of panel mode
    gbrmi = @brm df begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + (1 | pk | g)
        pred ~ kernel(x, log_CL) do xs, lCL
            mu = exp(lCL) .* xs
            yy = mu
            mu
        end
        y ~ Normal(pred, sigma)
    end
    gresult, grhs = only(BRM._rk_kernel_ops(gbrmi))
    @test_throws "per-subject data columns only" BRM._rk_kernel_spec(
        gbrmi, gresult, grhs)

    # varying per-subject vector length (ragged panel) fails closed until offsets
    rdf = (; t=[[0.0, 1.0], [0.0, 1.0, 2.0]], obs=[[0.1, 0.2], [0.3, 0.4, 0.5]])
    rbrmi = @brm rdf begin
        pred ~ kernel(t, obs) do ts, yy
            mu = ts .* 2.0
            yy ~ Normal(mu, 1.0)
            mu
        end
    end
    rresult, rrhs = only(BRM._rk_kernel_ops(rbrmi))
    @test_throws "varying timepoint counts" BRM._rk_kernel_spec(rbrmi, rresult, rrhs)

    # bind dims for the extension: subjects key always, timepoint key only
    # when vector slices name one (all-scalar plates omit it).
    @test BRM._rk_kernel_bind_dims(spec) ==
        Dict(:kernel_nsub_pred => 2, :kernel_T_pred => 3)
    scalar_brmi = @brm (; dose=[10.0, 20.0], obs=[0.1, 0.2]) begin
        pred ~ kernel(dose, obs) do dd, yy
            mu = dd .* 0.1
            yy ~ Normal(mu, 1.0)
            mu
        end
    end
    sresult, srhs = only(BRM._rk_kernel_ops(scalar_brmi))
    scalar_spec = BRM._rk_kernel_spec(scalar_brmi, sresult, srhs)
    @test BRM._rk_kernel_bind_dims(scalar_spec) == Dict(:kernel_nsub_pred => 2)
end

@testset "mixture plan shapes" begin
    # Gaussian driving case (docs shape, bare-sigma RK spelling): param
    # locations, one shared log-link scale predictor, literal weights.
    dfmix = (; y=[-2.0, -1.8, 1.9, 2.2])
    brmi = @brm dfmix begin
        mu1 ~ Normal(-2, 0.1)
        mu2 ~ Normal(2, 0.1)
        log(sigma) ~ 1
        y ~ MixtureModel([Normal(mu1, sigma), Normal(mu2, sigma)], [0.4, 0.6])
    end
    plan = BRM._brm_rk_plan(brmi)
    @test plan.n_obs == 4
    likelihood = only(plan.responses)
    @test likelihood.family === :mixture
    @test likelihood.link === :identity
    @test likelihood.response === :y
    # Anchor: no location predictor, so the scale predictor.
    @test likelihood.predictor === :sigma
    @test isnothing(likelihood.scale)
    @test isnothing(likelihood.scale_predictor)
    @test isnothing(likelihood.trials)
    @test isnothing(likelihood.weights)
    @test likelihood.evidence.kind === :none
    @test length(likelihood.mixture_components) == 2
    c1, c2 = likelihood.mixture_components
    @test (c1.family, c1.link) === (:gaussian, :identity)
    @test (c1.location, c1.location_kind) === (:mu1, :param)
    @test isnothing(c1.scale)
    @test c1.scale_predictor === :sigma
    @test (c2.location, c2.location_kind) === (:mu2, :param)
    @test c2.scale_predictor === :sigma
    @test likelihood.mixture_weights == [0.4, 0.6]
    @test [p.name for p in plan.predictors] == [:sigma]
    @test only(plan.predictors).link === :log
    @test sort!([p.name for p in plan.parameters]) == [:mu1, :mu2]
    @test plan.columns[:y] == dfmix.y

    # All-scalar Poisson mixture: zero predictors, param anchor.
    dfpois = (; y=[0, 1, 3, 5, 2])
    brmi = @brm dfpois begin
        lambda1 ~ Exponential(1)
        lambda2 ~ Exponential(1)
        y ~ MixtureModel([Poisson(lambda1), Poisson(lambda2)], [0.3, 0.7])
    end
    plan = BRM._brm_rk_plan(brmi)
    likelihood = only(plan.responses)
    @test likelihood.family === :mixture
    @test isempty(plan.predictors)
    @test likelihood.predictor === :lambda1
    @test [c.family for c in likelihood.mixture_components] ==
        [:poisson_log, :poisson_log]
    @test [c.location_kind for c in likelihood.mixture_components] ==
        [:param, :param]

    # Predictor locations + shared param scale + Dirichlet weights.
    dfw = (; x=[0.5, -1.0, 1.5, 0.0], y=[1.0, 2.0, 1.5, 2.5])
    brmi = @brm dfw begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        s ~ Exponential(1)
        w ~ Dirichlet(2, 1.0)
        y ~ MixtureModel([Normal(mu1, s), Normal(mu2, s)], w)
    end
    plan = BRM._brm_rk_plan(brmi)
    likelihood = only(plan.responses)
    @test likelihood.predictor === :mu1
    @test likelihood.mixture_weights === :w
    @test [c.location_kind for c in likelihood.mixture_components] ==
        [:predictor, :predictor]
    @test [c.scale for c in likelihood.mixture_components] == [:s, :s]
    wspec = only(v for v in plan.vector_parameters if v.name === :w)
    @test wspec.family === :simplex_dirichlet
    @test wspec.size == 2

    # K = 1 admits (the general form — no special-casing).
    brmi = @brm dfmix begin
        m ~ Normal(0, 1)
        y ~ MixtureModel([Normal(m, 1)], [1.0])
    end
    plan = BRM._brm_rk_plan(brmi)
    likelihood = only(plan.responses)
    @test length(likelihood.mixture_components) == 1
    @test likelihood.mixture_weights == [1.0]
    @test likelihood.predictor === :m

    # Binomial mixtures share one trials column at the response level.
    dfbin = (; y=[1, 8, 3, 9], n=[10, 10, 10, 10])
    brmi = @brm dfbin begin
        p1 ~ Beta(2, 2)
        p2 ~ Beta(2, 2)
        y ~ MixtureModel([Binomial(n, p1), Binomial(n, p2)], [0.5, 0.5])
    end
    plan = BRM._brm_rk_plan(brmi)
    likelihood = only(plan.responses)
    @test likelihood.trials === :n
    @test plan.columns[:n] == dfbin.n
    @test [c.family for c in likelihood.mixture_components] ==
        [:binomial_logit, :binomial_logit]

    # BernoulliLogit components need predictor locations (logit-scale
    # positions never ride bare); Bernoulli components take sampled
    # probabilities (the SB-tested shape).
    dfbern = (; x=[0.5, -1.0, 1.5, 0.0], y=[0, 1, 1, 0])
    brmi = @brm dfbern begin
        eta1 ~ 1 + x
        eta2 ~ 1 + x
        y ~ MixtureModel([BernoulliLogit(eta1), BernoulliLogit(eta2)],
            [0.5, 0.5])
    end
    plan = BRM._brm_rk_plan(brmi)
    likelihood = only(plan.responses)
    comps = likelihood.mixture_components
    @test [c.location_kind for c in comps] == [:predictor, :predictor]
    @test likelihood.predictor === :eta1
    brmi = @brm dfbern begin
        p1 ~ Beta(2, 2)
        p2 ~ Beta(2, 2)
        y ~ MixtureModel([Bernoulli(p1), Bernoulli(p2)], [0.5, 0.5])
    end
    plan = BRM._brm_rk_plan(brmi)
    likelihood = only(plan.responses)
    @test [c.location_kind for c in likelihood.mixture_components] ==
        [:param, :param]
    @test likelihood.predictor === :p1
end

@testset "fail closed: mixture" begin
    dfmix = (; y=[-2.0, -1.8, 1.9, 2.2])
    dfcount = (; y=[0, 1, 3, 5, 2])
    dfbin = (; y=[0, 1, 1, 0])
    # Heterogeneous families (SB's allequal rule, mirrored).
    @test_throws "must share one family" BRM._brm_rk_plan((@brm dfmix begin
        y ~ MixtureModel([Normal(0, 1), Cauchy(0, 1)], [0.5, 0.5])
    end))
    # BernoulliLogit/Bernoulli mix: one RK family, two Julia heads —
    # SB rejects it, so RK rejects it too.
    @test_throws "must share one family" BRM._brm_rk_plan((@brm dfbin begin
        eta ~ 1
        p ~ Beta(2, 2)
        y ~ MixtureModel([BernoulliLogit(eta), Bernoulli(p)], [0.5, 0.5])
    end))
    # Custom / parameter-support / simplex-parameter families.
    @test_throws "is not admitted" BRM._brm_rk_plan((@brm dfcount begin
        y ~ MixtureModel(
            [ZeroInflatedPoisson(1.0, 0.2), ZeroInflatedPoisson(2.0, 0.2)],
            [0.5, 0.5])
    end))
    @test_throws "is not admitted" BRM._brm_rk_plan((@brm dfmix begin
        y ~ MixtureModel([Uniform(0, 1), Uniform(0, 1)], [0.5, 0.5])
    end))
    @test_throws "is not admitted" BRM._brm_rk_plan((@brm dfbin begin
        s ~ Dirichlet(2, 1.0)
        y ~ MixtureModel([Categorical(s), Categorical(s)], [0.5, 0.5])
    end))
    # BinomialLogit head inherits the single-family rejection.
    @test_throws "out of mixture v1" BRM._brm_rk_plan((@brm dfbin begin
        y ~ MixtureModel(
            [BinomialLogit(10, 0.5), BinomialLogit(10, 0.5)], [0.5, 0.5])
    end))
    # Weights: sum, length, nonnegativity, data columns, simplex width.
    @test_throws "must sum to 1" BRM._brm_rk_plan((@brm dfmix begin
        m ~ Normal(0, 1)
        y ~ MixtureModel([Normal(m, 1), Normal(m, 1)], [0.5, 0.6])
    end))
    @test_throws "2 components but 3 weights" BRM._brm_rk_plan((@brm dfmix begin
        m ~ Normal(0, 1)
        y ~ MixtureModel([Normal(m, 1), Normal(m, 1)], [0.5, 0.3, 0.2])
    end))
    @test_throws "must be nonnegative" BRM._brm_rk_plan((@brm dfmix begin
        m ~ Normal(0, 1)
        y ~ MixtureModel([Normal(m, 1), Normal(m, 1)], [1.5, -0.5])
    end))
    @test_throws "data-column weights" BRM._brm_rk_plan(
        (@brm begin
            m ~ Normal(0, 1)
            y ~ MixtureModel([Normal(m, 1), Normal(m, 1)], w)
        end)((; y=[1.0, 2.0], w=[0.5, 0.5])))
    @test_throws "sizes must agree" BRM._brm_rk_plan((@brm dfmix begin
        w ~ Dirichlet(3, 1.0)
        m ~ Normal(0, 1)
        y ~ MixtureModel([Normal(m, 1), Normal(m, 1)], w)
    end))
    # An unused Dirichlet names mixture weights only when the model has
    # a mixture (the guidance stays context-sensitive).
    @test_throws "or mixture weights use it" BRM._brm_rk_plan((@brm dfmix begin
        w ~ Dirichlet(2, 1.0)
        m ~ Normal(0, 1)
        y ~ MixtureModel([Normal(m, 1), Normal(m, 1)], [0.5, 0.5])
    end))
    @test_throws "fully fixed" BRM._brm_rk_plan((@brm dfmix begin
        y ~ MixtureModel([Normal(-1.0, 0.5), Normal(1.0, 0.5)], [0.4, 0.6])
    end))
    # Deterministic wrappers spell as the LP link, like single-family.
    @test_throws "spell as an LP link" BRM._brm_rk_plan((@brm dfmix begin
        mu1 ~ Normal(-2, 0.1)
        mu2 ~ Normal(2, 0.1)
        log(sigma) ~ 1
        y ~ MixtureModel([Normal(mu1, exp(log(sigma))),
            Normal(mu2, exp(log(sigma)))], [0.4, 0.6])
    end))
    # Binomial components share one identical trials expression.
    @test_throws "share one identical" BRM._brm_rk_plan(
        (@brm begin
            p1 ~ Beta(2, 2)
            p2 ~ Beta(2, 2)
            y ~ MixtureModel([Binomial(n, p1), Binomial(10, p2)], [0.5, 0.5])
        end)((; y=[1, 8, 3, 9], n=[10, 10, 10, 10])))
    @test_throws "exceeds its trials" BRM._brm_rk_plan(
        (@brm begin
            p1 ~ Beta(2, 2)
            p2 ~ Beta(2, 2)
            y ~ MixtureModel([Binomial(n, p1), Binomial(n, p2)], [0.5, 0.5])
        end)((; y=[1, 80, 3, 9], n=[10, 10, 10, 10])))
    # Predictor-link rules mirror the single-family triples (mixture v1
    # admits logit only for Bernoulli/Binomial/Beta locations).
    @test_throws "logit-link predictor" BRM._brm_rk_plan(
        (@brm begin
            logit(p) ~ 1 + x
            s ~ Exponential(1)
            y ~ MixtureModel([Normal(p, s), Normal(p, s)], [0.5, 0.5])
        end)((; x=[0.5, -1.0, 1.5, 0.0], y=[1.0, 2.0, 1.5, 2.5])))
    @test_throws "mixture v1 admits" BRM._brm_rk_plan(
        (@brm begin
            probit(p) ~ 1 + x
            q ~ Beta(2, 2)
            y ~ MixtureModel([Bernoulli(p), Bernoulli(q)], [0.5, 0.5])
        end)((; x=[0.5, -1.0, 1.5, 0.0], y=[0, 1, 1, 0])))
    # Literal locations validate where the math is certain.
    @test_throws "must be finite and positive" BRM._brm_rk_plan(
        (@brm dfcount begin
            y ~ MixtureModel([Poisson(-1.0), Poisson(2.0)], [0.5, 0.5])
        end))
    @test_throws "must be a probability in [0, 1]" BRM._brm_rk_plan(
        (@brm dfbin begin
            y ~ MixtureModel([Bernoulli(0.2), Bernoulli(1.5)], [0.5, 0.5])
        end))
    # Logit-scale positions never ride bare: BernoulliLogit needs an
    # identity predictor (a bare parameter would read as a
    # probability, but SB computes bernoulli_logit_lpmf).
    @test_throws "needs an identity-link predictor" BRM._brm_rk_plan(
        (@brm begin
            e ~ Normal(0, 1)
            f ~ Normal(0, 1)
            y ~ MixtureModel([BernoulliLogit(e), BernoulliLogit(f)],
                [0.5, 0.5])
        end)((; y=[0, 1, 1, 0])))
    @test_throws "needs an identity-link predictor" BRM._brm_rk_plan(
        (@brm begin
            y ~ MixtureModel([BernoulliLogit(0.5), BernoulliLogit(-0.5)],
                [0.5, 0.5])
        end)((; y=[0, 1, 1, 0])))
    # Assignments are scale-only; data columns are never locations.
    @test_throws "assignments are scale-only" BRM._brm_rk_plan(
        (@brm begin
            m = 1.0 + 2.0
            y ~ MixtureModel([Normal(m, 1), Normal(m, 1)], [0.5, 0.5])
        end)((; y=[1.0, 2.0])))
    @test_throws "cannot be a data column" BRM._brm_rk_plan(
        (@brm begin
            s ~ Exponential(1)
            y ~ MixtureModel([Normal(x, s), Normal(x, s)], [0.5, 0.5])
        end)((; x=[0.5, -1.0], y=[1.0, 2.0])))
    # Response-value gating dispatches on the component family (SB
    # coerces float 0/1 — RK rejects, like single-family).
    @test_throws "Bool or 0/1 integers" BRM._brm_rk_plan(
        (@brm begin
            p1 ~ Beta(2, 2)
            p2 ~ Beta(2, 2)
            y ~ MixtureModel([Bernoulli(p1), Bernoulli(p2)], [0.5, 0.5])
        end)((; y=[0.0, 1.0, 1.0, 0.0])))
    # No response-level weights or bounded evidence in mixture v1.
    @test_throws "out of mixture v1" BRM._brm_rk_plan(
        (@brm begin
            m ~ Normal(0, 1)
            y ~ weighted(MixtureModel([Normal(m, 1), Normal(m, 1)],
                [0.5, 0.5]), fweights(w))
        end)((; y=[1.0, 2.0], w=[1.0, 1.0])))
    @test_throws "out of mixture v1" BRM._brm_rk_plan(
        (@brm begin
            m ~ Normal(0, 1)
            y ~ truncated(MixtureModel([Normal(m, 1), Normal(m, 1)],
                [0.5, 0.5]); lower=0.0)
        end)((; y=[1.0, 2.0])))
    # gp mixture predictors fail closed (single-predictor plate range).
    @test_throws "gp mixture predictors" BRM._brm_rk_plan((@brm df begin
        mu ~ 1 + gp(x)
        s ~ Exponential(1)
        y ~ MixtureModel([Normal(mu, s), Normal(mu, s)], [0.5, 0.5])
    end))
end
