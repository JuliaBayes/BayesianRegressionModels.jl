# Pure source goldens for flat coefficients and ordinary statistical bodies.
#
# Run: julia --project=test test/rk_ast.jl
#
# Response-family/evidence goldens are retained across migration. Builtin and
# lattice goldens are replaced by explicit-prior/algebra/source checks below;
# native density, gradient and printed-source acceptance lives in rk_plain.jl
# and rk_statistical_library.jl.

using Test
include(joinpath(@__DIR__, "testset_filter.jl"))
using BayesianRegressionModels
using CategoricalArrays: categorical
using Distributions: Bernoulli, Beta, Binomial, Categorical, Cauchy, Dirichlet,
                     Exponential, Gamma, InverseGaussian, Laplace,
                     LocationScale, Logistic, LogNormal, MixtureModel,
                     Multinomial, NegativeBinomial, Normal, Poisson, TDist,
                     Uniform, VonMises, Weibull, truncated
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
    obs=[3 1 1; 2 2 1; 0 0 5; 1 2 2; 4 0 1; 2 1 2],
)

probit(p) = quantile(Normal(), p)
cloglog(p) = log(-log1p(-p))
dfp = merge(df, (; prop=[0.2, 0.7, 0.4, 0.6, 0.3, 0.8]))

@stestset "link wrappers and triples" begin
    brmi = @brm df begin
        eta ~ 1 + x
        b ~ BernoulliLogit(eta)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :b,
        Expr(:., :Bernoulli, Expr(:tuple,
            Expr(:., :logistic, Expr(:tuple, :eta)))))
    # Triple 3 lowers to the T2 shape: the affine value feeds logistic.
    brmi = @brm df begin
        logit(p) ~ 1 + x
        b ~ Bernoulli(p)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :b,
        Expr(:., :Bernoulli, Expr(:tuple,
            Expr(:., :logistic, Expr(:tuple, :p)))))
    brmi = @brm df begin
        log(mu) ~ 1 + x
        c ~ Poisson(mu)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :c,
        Expr(:., :Poisson, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :mu)))))
end

@stestset "slice-1 response AST shapes" begin
    brmi = @brm df begin
        logit(p) ~ 1 + x
        b ~ Binomial(h, p)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :b,
        Expr(:., :Binomial, Expr(:tuple, :h,
            Expr(:., :logistic, Expr(:tuple, :p)))))
    brmi = @brm df begin
        log(mu) ~ 1 + x
        phi ~ Exponential(1)
        c ~ NegativeBinomial2(mu, phi)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :c,
        Expr(:., :NegativeBinomial2, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :mu)), :phi)))
    brmi = @brm df begin
        log(mu) ~ 1 + x
        alpha ~ Exponential(1)
        z ~ Gamma(alpha, mu / alpha)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :z,
        Expr(:., :Gamma, Expr(:tuple, :alpha,
            Expr(:call, :./, Expr(:., :exp, Expr(:tuple, :mu)),
                :alpha))))
end

@stestset "slice-2 group-A AST shapes" begin
    brmi = @brm df begin
        probit(p) ~ 1 + x
        b ~ Bernoulli(p)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :b,
        Expr(:., :Bernoulli, Expr(:tuple,
            Expr(:., :brm_invprobit, Expr(:tuple, :p)))))
    brmi = @brm df begin
        cloglog(p) ~ 1 + x
        b ~ Binomial(h, p)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :b,
        Expr(:., :Binomial, Expr(:tuple, :h,
            Expr(:., :brm_invcloglog, Expr(:tuple, :p)))))
    brmi = @brm dfp begin
        logit(mu) ~ 1 + x
        kappa ~ Gamma(2.0, 1000.0)
        prop ~ Beta(mu * kappa, (1 - mu) * kappa)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    mu_log = Expr(:., :logistic, Expr(:tuple, :mu))
    @test prog.main.args[end] == Expr(:call, :.~, :prop,
        Expr(:., :Beta, Expr(:tuple,
            Expr(:call, :.*, mu_log, :kappa),
            Expr(:call, :.*, Expr(:call, :.-, 1, mu_log), :kappa))))
end

@stestset "group-B student-t AST shape" begin
    # Dedicated single head (thin-layer decision, pair fam-student):
    # `LocationScale(mu, s, TDist(nu))` maps to `StudentT.(nu, mu, s)`
    # by arg reorder (Stan `student_t(nu, mu, sigma)` order).
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        nu ~ Gamma(2, 0.1)
        y ~ LocationScale(mu, s, TDist(nu))
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :y,
        Expr(:., :StudentT, Expr(:tuple, :nu, :mu, :s)))
    # Literals inline; the fused-heads flag changes nothing (one head
    # either way).
    brmi = @brm df begin
        mu ~ 1 + x
        y ~ LocationScale(mu, 2.0, TDist(4.0))
    end
    plan = BRM._brm_rk_plan(brmi)
    want = Expr(:call, :.~, :y,
        Expr(:., :StudentT, Expr(:tuple, 4.0, :mu, 2.0)))
    @test BRM._rk_emit_ast(plan, false).main.args[end] == want
    @test BRM._rk_emit_ast(plan, true).main.args[end] == want
    # Modeled nu (N1 probe shape): the `log(nu)` submodel rides under
    # `exp.`, the scale-predictor precedent.
    brmi = @brm df begin
        mu ~ 1 + x
        log(nu) ~ 1 + z
        y ~ LocationScale(mu, 2.0, TDist(nu))
    end
    plan = BRM._brm_rk_plan(brmi)
    want = Expr(:call, :.~, :y,
        Expr(:., :StudentT, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :nu)), :mu, 2.0)))
    @test BRM._rk_emit_ast(plan, false).main.args[end] == want
    @test BRM._rk_emit_ast(plan, true).main.args[end] == want
end

@stestset "group-C hurdle-poisson AST shape" begin
    # Twin head (thin-layer decision, pair fam-hurdle):
    # `HurdlePoisson(lambda, p_zero)` maps to
    # `HurdlePoisson.(exp.(lambda), logistic.(p_zero))` (NB2
    # precedent); no fused head.
    brmi = @brm df begin
        log(lambda) ~ 1 + x
        logit(p_zero) ~ 1 + x
        c ~ HurdlePoisson(lambda, p_zero)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :c,
        Expr(:., :HurdlePoisson, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :lambda)),
            Expr(:., :logistic, Expr(:tuple, :p_zero)))))
    # Scalar p_zero inlines bare; the fused-heads flag changes nothing
    # (one head either way).
    brmi = @brm df begin
        log(lambda) ~ 1 + x
        p0 ~ Beta(2, 2)
        c ~ HurdlePoisson(lambda, p0)
    end
    plan = BRM._brm_rk_plan(brmi)
    want = Expr(:call, :.~, :c,
        Expr(:., :HurdlePoisson, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :lambda)), :p0)))
    @test BRM._rk_emit_ast(plan, false).main.args[end] == want
    @test BRM._rk_emit_ast(plan, true).main.args[end] == want
end

@stestset "group-C ZIP AST shape" begin
    # Dedicated single head (thin-layer decision, pair fam-zip):
    # `ZeroInflatedPoisson(lambda, zi)` maps to
    # `ZeroInflatedPoisson.(exp.(lambda), zi)` (Julia/Stan
    # `(lambda, zi)` order).
    brmi = @brm df begin
        log(lambda) ~ 1 + x
        zi ~ Beta(2, 2)
        c ~ ZeroInflatedPoisson(lambda, zi)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :c,
        Expr(:., :ZeroInflatedPoisson, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :lambda)), :zi)))
    # Modeled zi (zip.jl model B): the `logit(zi)` submodel inverts
    # under `logistic.` (the hurdle hu precedent).
    brmi = @brm df begin
        log(lambda) ~ 1 + x
        logit(zi) ~ 1 + x
        c ~ ZeroInflatedPoisson(lambda, zi)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :c,
        Expr(:., :ZeroInflatedPoisson, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :lambda)),
            Expr(:., :logistic, Expr(:tuple, :zi)))))
    # Literals inline; the fused-heads flag changes nothing (one head
    # either way).
    brmi = @brm df begin
        log(lambda) ~ 1 + x
        c ~ ZeroInflatedPoisson(lambda, 0.25)
    end
    plan = BRM._brm_rk_plan(brmi)
    want = Expr(:call, :.~, :c,
        Expr(:., :ZeroInflatedPoisson, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :lambda)), 0.25)))
    @test BRM._rk_emit_ast(plan, false).main.args[end] == want
    @test BRM._rk_emit_ast(plan, true).main.args[end] == want
end

@stestset "group-C negative-binomial AST shape" begin
    # Twin head (thin-layer decision, pair fam-nb1):
    # `NegativeBinomial(r, p)` maps to
    # `NegativeBinomial.(exp.(r), p)` (NB2 precedent); no fused head.
    brmi = @brm df begin
        log(r) ~ 1 + x
        p ~ Beta(2, 2)
        c ~ NegativeBinomial(r, p)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :c,
        Expr(:., :NegativeBinomial, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :r)), :p)))
    # Literals inline; the fused-heads flag changes nothing (one head
    # either way).
    brmi = @brm df begin
        log(r) ~ 1 + x
        c ~ NegativeBinomial(r, 0.4)
    end
    plan = BRM._brm_rk_plan(brmi)
    want = Expr(:call, :.~, :c,
        Expr(:., :NegativeBinomial, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :r)), 0.4)))
    @test BRM._rk_emit_ast(plan, false).main.args[end] == want
    @test BRM._rk_emit_ast(plan, true).main.args[end] == want
    # Modeled p: the `logit(p)` submodel rides the scale slot under
    # `logistic.` (pair nuisance-nb1p; the hurdle twin precedent).
    brmi = @brm df begin
        log(r) ~ 1 + x
        logit(p) ~ 1 + x
        c ~ NegativeBinomial(r, p)
    end
    plan = BRM._brm_rk_plan(brmi)
    want = Expr(:call, :.~, :c,
        Expr(:., :NegativeBinomial, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :r)),
            Expr(:., :logistic, Expr(:tuple, :p)))))
    @test BRM._rk_emit_ast(plan, false).main.args[end] == want
    @test BRM._rk_emit_ast(plan, true).main.args[end] == want
end

@stestset "group-C weibull AST shape" begin
    # Twin head (thin-layer decision, pair fam-weibull):
    # `Weibull(k, theta)` maps to `Weibull.(k, exp.(theta))`
    # (Distributions `(shape, scale)` order, NB2 precedent); no
    # fused head.
    brmi = @brm df begin
        log(mu) ~ 1 + x
        k ~ LogNormal(0, 0.3)
        z ~ Weibull(k, mu)
    end
    plan = BRM._brm_rk_plan(brmi)
    want = Expr(:call, :.~, :z,
        Expr(:., :Weibull, Expr(:tuple, :k,
            Expr(:., :exp, Expr(:tuple, :mu)))))
    @test BRM._rk_emit_ast(plan, false).main.args[end] == want
    @test BRM._rk_emit_ast(plan, true).main.args[end] == want
    # Literal shape inlines; the fused-heads flag changes nothing
    # (one head either way).
    brmi = @brm df begin
        log(mu) ~ 1 + x
        z ~ Weibull(2.0, mu)
    end
    plan = BRM._brm_rk_plan(brmi)
    want = Expr(:call, :.~, :z,
        Expr(:., :Weibull, Expr(:tuple, 2.0,
            Expr(:., :exp, Expr(:tuple, :mu)))))
    @test BRM._rk_emit_ast(plan, false).main.args[end] == want
    @test BRM._rk_emit_ast(plan, true).main.args[end] == want
end

@stestset "group-C wald AST shape" begin
    # Twin head (thin-layer decision, pair fam-inversegaussian):
    # `InverseGaussian(mu, lam)` maps to
    # `InverseGaussian.(exp.(mu), lam)` (NB2 precedent); no fused head.
    brmi = @brm df begin
        log(mu) ~ 1 + x
        lam ~ LogNormal(-0.3, 1.0)
        z ~ InverseGaussian(mu, lam)
    end
    plan = BRM._brm_rk_plan(brmi)
    want = Expr(:call, :.~, :z,
        Expr(:., :InverseGaussian, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :mu)), :lam)))
    @test BRM._rk_emit_ast(plan, false).main.args[end] == want
    @test BRM._rk_emit_ast(plan, true).main.args[end] == want
    # Literal shape inlines; the fused-heads flag changes nothing
    # (one head either way).
    brmi = @brm df begin
        log(mu) ~ 1 + x
        z ~ InverseGaussian(mu, 2.0)
    end
    plan = BRM._brm_rk_plan(brmi)
    want = Expr(:call, :.~, :z,
        Expr(:., :InverseGaussian, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :mu)), 2.0)))
    @test BRM._rk_emit_ast(plan, false).main.args[end] == want
    @test BRM._rk_emit_ast(plan, true).main.args[end] == want
    # Modeled lambda (pair nuisance-lam): the `log(lam)` scale
    # predictor inverts under `exp.` like any scale predictor (the
    # von-Mises precedent); the fused-heads flag changes nothing (one
    # head either way).
    brmi = @brm df begin
        log(mu) ~ 1 + x
        log(lam) ~ 1 + x
        z ~ InverseGaussian(mu, lam)
    end
    plan = BRM._brm_rk_plan(brmi)
    want = Expr(:call, :.~, :z,
        Expr(:., :InverseGaussian, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :mu)),
            Expr(:., :exp, Expr(:tuple, :lam)))))
    @test BRM._rk_emit_ast(plan, false).main.args[end] == want
    @test BRM._rk_emit_ast(plan, true).main.args[end] == want
end

@stestset "group-C exponential AST shape" begin
    # Twin head (thin-layer decision, pair fam-exp):
    # `Exponential(mu)` maps to `Exponential.(exp.(mu))`
    # (Poisson-shaped single-arg twin); no fused head.
    brmi = @brm df begin
        log(mu) ~ 1 + x
        z ~ Exponential(mu)
    end
    plan = BRM._brm_rk_plan(brmi)
    want = Expr(:call, :.~, :z,
        Expr(:., :Exponential, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :mu)))))
    @test BRM._rk_emit_ast(plan, false).main.args[end] == want
    @test BRM._rk_emit_ast(plan, true).main.args[end] == want
end

@stestset "group-D beta-binomial AST shape" begin
    # Twin head (thin-layer decision, pair fam-betabinom):
    # `BetaBinomial2(n, mu, phi)` maps to
    # `BetaBinomial2.(n, logistic.(mu), phi)` (hurdle precedent);
    # no fused head.
    brmi = @brm df begin
        logit(mu) ~ 1 + x
        phi ~ Gamma(2, 0.1)
        b ~ BetaBinomial2(h, mu, phi)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :b,
        Expr(:., :BetaBinomial2, Expr(:tuple,
            :h,
            Expr(:., :logistic, Expr(:tuple, :mu)),
            :phi)))
    # Literal trials + literal precision inline bare; the fused-heads
    # flag changes nothing (one head either way).
    brmi = @brm df begin
        logit(mu) ~ 1 + x
        c ~ BetaBinomial2(10, mu, 5.0)
    end
    plan = BRM._brm_rk_plan(brmi)
    want = Expr(:call, :.~, :c,
        Expr(:., :BetaBinomial2, Expr(:tuple,
            10,
            Expr(:., :logistic, Expr(:tuple, :mu)),
            5.0)))
    @test BRM._rk_emit_ast(plan, false).main.args[end] == want
    @test BRM._rk_emit_ast(plan, true).main.args[end] == want
    # A `log(precision)` submodel inverts under `exp.` like any scale
    # predictor (pair nuisance-precision).
    brmi = @brm df begin
        logit(mu) ~ 1 + x
        log(precision) ~ 1 + z
        b ~ BetaBinomial2(h, mu, precision)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :b,
        Expr(:., :BetaBinomial2, Expr(:tuple,
            :h,
            Expr(:., :logistic, Expr(:tuple, :mu)),
            Expr(:., :exp, Expr(:tuple, :precision)))))
end

@stestset "group-C von-Mises AST shape" begin
    # Twin heads (thin-layer decision, pair fam-vonmises): exact
    # `VonMises(mu, kappa)` maps to `VonMises.(mu, kappa)`, and the
    # `log(kappa)` submodel inverts under `exp.` like any scale
    # predictor.
    brmi = @brm df begin
        mu ~ 1 + x
        log(kappa) ~ 1
        y ~ VonMises(mu, kappa)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :y,
        Expr(:., :VonMises, Expr(:tuple, :mu,
            Expr(:., :exp, Expr(:tuple, :kappa)))))
    # `CircularVonMises` appends the literal principal interval;
    # literals inline, and the fused-heads flag changes nothing (one
    # head either way).
    brmi = @brm df begin
        mu ~ 1 + x
        y ~ CircularVonMises(mu, 1.7; interval=(-pi, pi))
    end
    plan = BRM._brm_rk_plan(brmi)
    want = Expr(:call, :.~, :y,
        Expr(:., :CircularVonMises, Expr(:tuple, :mu, 1.7,
            -Float64(pi), Float64(pi))))
    @test BRM._rk_emit_ast(plan, false).main.args[end] == want
    @test BRM._rk_emit_ast(plan, true).main.args[end] == want
end

@stestset "group-C lognormal AST shape" begin
    # Single head (thin-layer decision, pair fam-lognormal):
    # `LogNormal(mu, sigma)` maps to `LogNormal.(mu, sigma)`
    # (Distributions order); no fused head.
    brmi = @brm df begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        z ~ LogNormal(mu, sigma)
    end
    plan = BRM._brm_rk_plan(brmi)
    want = Expr(:call, :.~, :z,
        Expr(:., :LogNormal, Expr(:tuple, :mu, :sigma)))
    @test BRM._rk_emit_ast(plan, false).main.args[end] == want
    @test BRM._rk_emit_ast(plan, true).main.args[end] == want
    # Literal scale inlines; the fused-heads flag changes nothing
    # (one head either way).
    brmi = @brm df begin
        mu ~ 1 + x
        z ~ LogNormal(mu, 0.5)
    end
    plan = BRM._brm_rk_plan(brmi)
    want = Expr(:call, :.~, :z,
        Expr(:., :LogNormal, Expr(:tuple, :mu, 0.5)))
    @test BRM._rk_emit_ast(plan, false).main.args[end] == want
    @test BRM._rk_emit_ast(plan, true).main.args[end] == want
end

@stestset "evidence and weights shapes" begin
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ weighted(truncated(Normal(mu, s), 0.0, 2.0), fweights(n))
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :y,
        Expr(:., :weighted, Expr(:tuple,
            Expr(:., :truncated, Expr(:tuple,
                Expr(:., :Normal, Expr(:tuple, :mu, :s)),
                0.0, 2.0)), :n)))
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ interval_censored(Normal(mu, s); upper=2.0)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :y,
        Expr(:., :interval_censored, Expr(:tuple,
            Expr(:., :Normal, Expr(:tuple, :mu, :s)), 2.0)))
    # Missing sides pass ∓Inf floats (normalized back at bind).
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ truncated(Normal(mu, s), 0, Inf)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :y,
        Expr(:., :truncated, Expr(:tuple,
            Expr(:., :Normal, Expr(:tuple, :mu, :s)),
            0.0, Inf)))
end

@stestset "offset-only predictors emit bare affines" begin
    brmi = @brm df begin
        mu ~ 0 + offset(z)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    # No predictor def either (offset-only has no scalar statements),
    # so the program carries no defs at all.
    @test isempty(prog.defs)
    @test prog.main == Expr(:block,
        Expr(:(=), :mu, :z),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :mu, :s))))
    # Several offsets sum; a derived offset stages its definition first.
    brmi = @brm df begin
        mu ~ 0 + offset(z) + offset(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    affine = only(a for a in prog.main.args if a isa Expr && a.head === :(=) &&
        a.args[1] === :mu)
    @test affine == Expr(:(=), :mu, Expr(:call, :.+, :z, :x))
    brmi = @brm df begin
        mu ~ 0 + offset(log(z))
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[1] == Expr(:(=), :rkd_offset_log_z,
        Expr(:., :log, Expr(:tuple, :z)))
    @test prog.main.args[2] == Expr(:(=), :mu, :rkd_offset_log_z)
end

@stestset "prior vocab v1: sampled splices" begin
    # New sampled heads splice generically; StudentT arrives in Stan
    # order; Uniform carries literal bounds. Each declaration feeds the
    # response: independent unused priors are generated, not fitted.
    brmi = @brm df begin
        mu ~ 1 + x
        a ~ Laplace(0, 2)
        t ~ LocationScale(0, 2, TDist(4))
        u ~ Uniform(0.5, 1.5)
        s ~ Exponential(1)
        fitted_location = mu + a + t + u
        y ~ Normal(fitted_location, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test Expr(:call, :~, :a, Expr(:call, :Laplace, 0.0, 2.0)) in
        prog.main.args
    @test Expr(:call, :~, :t, Expr(:call, :StudentT, 4.0, 0.0, 2.0)) in
        prog.main.args
    @test Expr(:call, :~, :u, Expr(:call, :Uniform, 0.5, 1.5)) in
        prog.main.args
    # New symmetric halves splice `truncated` verbatim; legacy halves
    # keep their `HalfNormal`/`HalfCauchy` heads byte-identically.
    brmi = @brm df begin
        mu ~ 1 + x
        h1 ~ truncated(Logistic(0, 1), 0, Inf)
        hn ~ truncated(Normal(0, 2), 0, Inf)
        hc ~ truncated(Cauchy(0, 2), 0, Inf)
        s ~ Exponential(1)
        fitted_location = mu + h1 + hn + hc
        y ~ Normal(fitted_location, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test Expr(:call, :~, :h1, Expr(:call, :truncated,
        Expr(:call, :Logistic, 0.0, 1.0), 0.0, Inf)) in prog.main.args
    @test Expr(:call, :~, :hn, Expr(:call, :HalfNormal, 2.0)) in
        prog.main.args
    @test Expr(:call, :~, :hc, Expr(:call, :HalfCauchy, 2.0)) in
        prog.main.args
end

strip_source_lines(x) = x
strip_source_lines(x::GlobalRef) =
    strip_source_lines(Meta.parse(sprint(Base.show_unquoted, x)))
strip_source_lines(x::AbstractFloat) = isinf(x) ? (signbit(x) ? Expr(:call, :-, :Inf) : :Inf) : x
strip_source_lines(x::Expr) = Expr(x.head,
    (strip_source_lines(a) for a in x.args if !(a isa LineNumberNode))...)

@stestset "plain named Gaussian coefficient source" begin
    model = @brm df begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    emitted = BRM._rk_emit_ast(BRM._brm_rk_plan(model))
    @test isempty(emitted.defs)
    @test strip_source_lines(emitted.main) == strip_source_lines(:(begin
        mu_Intercept ~ Normal(0.0, 1.0)
        mu_x ~ Normal(0.0, 1.0)
        mu_X = hcat(ones(length(x)), x)
        mu_coefficients = [mu_Intercept, mu_x]
        mu = mu_X * mu_coefficients
        sigma ~ Exponential(1.0)
        y .~ Normal.(mu, sigma)
    end))
end

@stestset "plain coefficient priors, shared predictors and derived design" begin
    model = @brm merge(df, (; y2=df.y)) begin
        mu ~ 1 + x + x & z
        effect(mu, x) ~ Laplace(0, 2)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
        y2 ~ Normal(mu, 1.0)
    end
    emitted = BRM._rk_emit_ast(BRM._brm_rk_plan(model))
    @test isempty(emitted.defs)
    @test Expr(:call, :~, :mu_x, Expr(:call, :Laplace, 0.0, 2.0)) in emitted.main.args
    @test count(a -> Meta.isexpr(a, :(=)) && a.args[1] === :mu, emitted.main.args) == 1
    @test emitted.main.args[1] == Expr(:(=), :int_x_x_z, Expr(:call, :.*, :x, :z))
end

@stestset "statistical bodies emit explicit priors and ordinary algebra" begin
    cases = (
        @brm(df, begin mu ~ 1 + (1 + x | g); sigma ~ Exponential(1); y ~ Normal(mu, sigma) end),
        @brm(df, begin mu ~ 1 + mo(c); sigma ~ Exponential(1); y ~ Normal(mu, sigma) end),
        @brm(df, begin mu ~ 1 + dar(x); sigma ~ Exponential(1); y ~ Normal(mu, sigma) end),
        @brm(df, begin mu ~ 1 + x + z; effect(mu, x) ~ Horseshoe(); effect(mu, z) ~ Horseshoe(); sigma ~ Exponential(1); y ~ Normal(mu, sigma) end),
        @brm(df, begin mu ~ 1 + x + z; effect(mu, :) ~ r2d2(); sigma ~ Exponential(1); y ~ Normal(mu, sigma) end),
        @brm(df, begin mu ~ 1 + hsgp(x; k=3); sigma ~ Exponential(1); y ~ Normal(mu, sigma) end),
        @brm(df, begin mu ~ 1 + gp(x); sigma ~ Exponential(1); y ~ Normal(mu, sigma) end),
        @brm(df, begin eta ~ 0 + x; c ~ OrderedLogistic(eta) end),
    )
    for model in cases
        emitted = BRM._rk_emit_ast(BRM._brm_rk_plan(model))
        source = sprint(Base.show_unquoted, emitted.main)
        @test BRM._rk_validate_source_definitions(emitted) === nothing
        for definition in emitted.defs
            @test strip_source_lines(Meta.parse(sprint(Base.show_unquoted, definition))) ==
                strip_source_lines(definition)
        end
        @test strip_source_lines(Meta.parse(source)) == strip_source_lines(emitted.main)
        @test !occursin("popefs", source)
        @test !occursin("varying_draws", source)
        @test !occursin("Horseshoe", source)
        @test !occursin("r2d2(", source)
        @test !occursin("hsgp_basis(:", source)
    end
    hs = sprint(Base.show_unquoted, BRM._rk_emit_ast(BRM._brm_rk_plan(cases[4])).main)
    @test count("mu_tau ~", hs) == 1
    @test occursin("HalfCauchy", hs)
    ordinal = sprint(Base.show_unquoted, BRM._rk_emit_ast(BRM._brm_rk_plan(last(cases))).main)
    @test occursin("Ordered(Normal", ordinal)
    @test occursin("Ref(c_cutpoints)", ordinal)
end
