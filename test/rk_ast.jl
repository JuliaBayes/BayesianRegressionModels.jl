# test/rk_ast.jl — BRM-side `@rkppl` program emission (submodel flip).
#
# Run: julia --project=test test/rk_ast.jl
#
# Pure-Julia checks (no ReactiveKernels dependency): exact `Expr` shapes
# over the whole slice-1 surface (`_rk_emit_ast` is total — the program
# is the sole emission path, no fallback). Each case pins the submodel
# `defs` (surface-spelling `sm(args...) = begin ... end` forms) and the
# `main` block separately. Lowerability through the real `lower_rkppl`
# is covered per-slice by the parity corpus (test/rk_parity.jl carries
# the ranef slice), which routes each case through the retargeted
# factory; the parity anchors are unchanged by the flip (expansion is
# transparent), so they are the independent oracle for these goldens.

using Test
using BayesianRegressionModels
using Distributions: Bernoulli, Beta, Binomial, Categorical, Dirichlet,
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
    obs=[3 1 1; 2 2 1; 0 0 5; 1 2 2; 4 0 1; 2 1 2],
)

probit(p) = quantile(Normal(), p)
cloglog(p) = log(-log1p(-p))
dfp = merge(df, (; prop=[0.2, 0.7, 0.4, 0.6, 0.3, 0.8]))

@testset "gaussian AST exact shape" begin
    brmi = @brm df begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog isa BRM._RKEmittedProgram
    @test prog.main isa Expr && prog.main.head === :block
    # The eligible canonical GLM replaces the predictor spine entirely:
    # no popefs def, a bare-name numeric X, alpha/beta priors, and the
    # object response.
    @test prog.defs == Expr[]
    @test prog.main == Expr(:block,
        Expr(:call, :~, :sigma, Expr(:call, :Exponential, 1.0)),
        Expr(:(=), :y_X, Expr(:call, :hcat, :x)),
        Expr(:call, :~, :mu_alpha, Expr(:call, :Normal, 0.0, 1.0)),
        Expr(:call, :.~, Expr(:ref, :mu_beta,
                Expr(:call, :axes, :y_X, 2)),
            Expr(:., :Normal, Expr(:tuple, 0.0, 1.0))),
        Expr(:call, :~, :y,
            Expr(:call, :NormalIDGLM, :y_X, :mu_alpha, :mu_beta, :sigma)))
end

@testset "GLM object eligibility and priors" begin
    # Per-column population priors map positionally onto beta through the
    # pinned literal-vector broadcast form.
    brmi = @brm df begin
        mu ~ 1 + x + z
        effect(mu, x) ~ Normal(1.0, 2.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :y,
        Expr(:call, :NormalIDGLM, :y_X, :mu_alpha, :mu_beta, :sigma))
    @test Expr(:call, :.~, Expr(:ref, :mu_beta,
            Expr(:call, :axes, :y_X, 2)),
        Expr(:., :Normal, Expr(:tuple,
            Expr(:vect, 1.0, 0.0), Expr(:vect, 2.0, 1.0)))) in prog.main.args

    # Factor, modeled-scale, and intercept-only shapes stay on the
    # decomposed path even with the flip on.
    brmi = @brm df begin
        mu ~ 1 + factor(g; ref=3)
        effect(mu, g) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test BRM._rk_emit_ast(plan, true).main.args[end] ==
        BRM._rk_emit_ast(plan, false).main.args[end]
    brmi = @brm df begin
        log(sigma) ~ 1 + x
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test BRM._rk_emit_ast(plan, true).main.args[end] ==
        BRM._rk_emit_ast(plan, false).main.args[end]
    brmi = @brm df begin
        mu ~ 1
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test BRM._rk_emit_ast(plan, true).main.args[end] ==
        BRM._rk_emit_ast(plan, false).main.args[end]
end

@testset "link wrappers and triples" begin
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

@testset "slice-1 response AST shapes" begin
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

@testset "slice-2 group-A AST shapes" begin
    brmi = @brm df begin
        probit(p) ~ 1 + x
        b ~ Bernoulli(p)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :b,
        Expr(:., :Bernoulli, Expr(:tuple,
            Expr(:., :probit, Expr(:tuple, :p)))))
    brmi = @brm df begin
        cloglog(p) ~ 1 + x
        b ~ Binomial(h, p)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :b,
        Expr(:., :Binomial, Expr(:tuple, :h,
            Expr(:., :cloglog, Expr(:tuple, :p)))))
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

@testset "evidence and weights shapes" begin
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

@testset "factors and ref gating" begin
    # Factor-only: no scalar coefficients, so no `popefs` def — the
    # broadcast prior and the affine stay top-level exactly as before.
    brmi = @brm df begin
        mu ~ 0 + g
        effect(mu, g) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog isa BRM._RKEmittedProgram
    @test Expr(:call, :.~,
        Expr(:ref, :mu_b1, Expr(:call, :levels, :g)),
        Expr(:., :Normal, Expr(:tuple, 0.0, 2.0))) in prog.main.args
    affine = only(a for a in prog.main.args if a isa Expr && a.head === :(=))
    @test affine == Expr(:(=), :mu, Expr(:ref, :mu_b1, :g))
    @test all(prog.defs) do d
        !startswith(string(d.args[1].args[1]), "popefs")
    end
    # Subsets under an intercept drop the reference position: edge drops
    # spell as literal ranges, middle drops as literal index lists. The
    # factor prior stays top-level; the intercept prior moves into the
    # shared `popefs_normal_i_f` def (factor column and coef ride
    # formals) and the affine becomes its return.
    brmi = @brm df begin
        mu ~ 1 + factor(g; ref=3)
        effect(mu, g) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test Expr(:call, :.~,
        Expr(:ref, :mu_b2, Expr(:ref, Expr(:call, :levels, :g),
            Expr(:call, :(:), 1, 2))),
        Expr(:., :Normal, Expr(:tuple, 0.0, 2.0))) in prog.main.args
    @test Expr(:call, :~, :mu, Expr(:call, :popefs_normal_i_f, :g,
        :mu_b2, 0.0, 1.0)) in prog.main.args
    @test Expr(:(=), Expr(:call, :popefs_normal_i_f, :x1, :f1, :loc1,
        :s1), Expr(:block,
        Expr(:call, :~, :b1, Expr(:call, :Normal, :loc1, :s1)),
        Expr(:call, :.+, :b1, Expr(:ref, :f1, :x1)))) in prog.defs
    brmi = @brm df begin
        mu ~ 1 + factor(g; ref=2)
        effect(mu, g) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test Expr(:call, :.~,
        Expr(:ref, :mu_b2, Expr(:ref, Expr(:call, :levels, :g),
            Expr(:vect, 1, 3))),
        Expr(:., :Normal, Expr(:tuple, 0.0, 2.0))) in prog.main.args
    # String refs lower to the same sort-order drop position.
    brmi = @brm df begin
        mu ~ 1 + factor(gs; ref="c")
        effect(mu, gs) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test Expr(:call, :.~,
        Expr(:ref, :mu_b2, Expr(:ref, Expr(:call, :levels, :gs),
            Expr(:call, :(:), 1, 2))),
        Expr(:., :Normal, Expr(:tuple, 0.0, 2.0))) in prog.main.args
    # A bare factor under an intercept never reaches the surface.
    @test_throws ErrorException BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + g
        effect(mu, g) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
end

@testset "derived definitions" begin
    brmi = @brm df begin
        mu ~ 1 + x + x & z
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[1] == Expr(:(=), :int_x_x_z,
        Expr(:call, :.*, :x, :z))
    @test Expr(:(=), Expr(:call, :popefs_normal_i_c_c, :x1, :x2,
        :loc1, :s1, :loc2, :s2, :loc3, :s3),
        Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, :loc1, :s1)),
            Expr(:call, :~, :b2, Expr(:call, :Normal, :loc2, :s2)),
            Expr(:call, :~, :b3, Expr(:call, :Normal, :loc3, :s3)),
            Expr(:call, :.+,
                :b1, Expr(:call, :.*, :b2, :x1),
                Expr(:call, :.*, :b3, :x2)))) in prog.defs
    @test Expr(:call, :~, :mu,
        Expr(:call, :popefs_normal_i_c_c, :x, :int_x_x_z,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0)) in prog.main.args
    # Transforms stage inline reductions in a single definition.
    brmi = @brm df begin
        mu ~ 1 + zscale(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[1] == Expr(:(=), :zscale_x, Expr(:call, :./,
        Expr(:call, :.-, :x, Expr(:call, :mean, :x)),
        Expr(:call, :std, :x)))
    # Mixed interactions compare against raw level values.
    brmi = @brm df begin
        mu ~ 1 + x & g
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[1] == Expr(:(=), :int_x_x_g_lvl_1,
        Expr(:call, :.*, :x, Expr(:call, :.==, :g, 1)))
    @test prog.main.args[2] == Expr(:(=), :int_x_x_g_lvl_2,
        Expr(:call, :.*, :x, Expr(:call, :.==, :g, 2)))
    @test prog.main.args[3] == Expr(:(=), :int_x_x_g_lvl_3,
        Expr(:call, :.*, :x, Expr(:call, :.==, :g, 3)))
end

@testset "sampled, assignments, collisions" begin
    brmi = @brm df begin
        mu ~ 1 + x
        m = sum(x)
        s ~ Gamma(m, 2.0)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    assign = only(a for a in prog.main.args if a isa Expr && a.head === :(=) &&
        a.args[1] === :m)
    @test assign == Expr(:(=), :m, Expr(:call, :sum, :x))
    @test Expr(:call, :~, :s, Expr(:call, :Gamma, :m, 2.0)) in prog.main.args
    # HalfNormal + Flat mappings.
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ truncated(Normal(0, 1), 0, Inf)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test Expr(:call, :~, :s, Expr(:call, :HalfNormal, 1.0)) in prog.main.args
    # Coefficient names disambiguate against user names: the user param
    # `mu_b1` occupies the intercept's natural expansion, so the
    # predictor LHS renames to `mu_` (locals stay canonical `b1, b2`,
    # expanding to `mu__b1, mu__b2` — the def stays shared).
    brmi = @brm df begin
        mu ~ 1 + x
        mu_b1 ~ Normal(0, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test Expr(:call, :~, :mu_, Expr(:call, :popefs_normal_i_c, :x,
        0.0, 1.0, 0.0, 1.0)) in prog.main.args
    @test Expr(:call, :.~, :y,
        Expr(:., :Normal, Expr(:tuple, :mu_, :s))) in prog.main.args
    def = only(d for d in prog.defs
        if d.args[1].args[1] === :popefs_normal_i_c)
    body = def.args[2]
    @test body.args[1] ==
        Expr(:call, :~, :b1, Expr(:call, :Normal, :loc1, :s1))
    @test body.args[2] ==
        Expr(:call, :~, :b2, Expr(:call, :Normal, :loc2, :s2))
    @test body.args[end] == Expr(:call, :.+,
        :b1, Expr(:call, :.*, :b2, :x1))
end

@testset "multi-response shares one affine" begin
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ Normal(mu, s)
        z ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    calls = [a for a in prog.main.args if a isa Expr && a.head === :call &&
        length(a.args) == 3 && a.args[1] === :~ && a.args[3] isa Expr &&
        a.args[3].head === :call && a.args[3].args[1] === :popefs_normal_i_c]
    responses = [a for a in prog.main.args if a isa Expr && a.head === :call &&
        length(a.args) == 3 && a.args[1] === :.~ && a.args[3] isa Expr &&
        a.args[3].head === :. && a.args[2] in (:y, :z)]
    @test length(calls) == 1
    @test length(responses) == 2
    # Both responses spell the same bare distribution inline (no
    # stream def mediates them).
    @test all(r -> r.args[3] ==
        Expr(:., :Normal, Expr(:tuple, :mu, :s)), responses)
    @test all(prog.defs) do d
        startswith(string(d.args[1].args[1]), "popefs")
    end
end

@testset "offset-only predictors emit bare affines" begin
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

@testset "predictor/data overlap alpha-renames" begin
    # Unreachable via `@brm` (observation discovery claims `n ~ …` as a
    # likelihood), so the plan is built by hand; the submodel LHS moves
    # to `n_` while the overlapping data column keeps `n` — definition
    # and data reference coexist. Expanded locals namespace under the
    # renamed LHS (`n__b1`), the one spelling the flip cannot preserve.
    plan = BRM._RKStructuralPlan(
        [BRM._RKLikelihoodSpec(:gaussian, :identity, :y, :n, :s, nothing,
            nothing, BRM._RKResponseEvidence(:none, nothing, nothing), :y,
            nothing, nothing, nothing, Symbol[], Symbol[], nothing, nothing,
            Symbol[], nothing, Symbol[], nothing)],
        [BRM._RKPredictorSpec(:n, :identity, BRM._RKTermSpec[
            BRM._RKTermSpec(:intercept, Symbol[], (;), :Intercept, :Intercept),
            BRM._RKTermSpec(:continuous, [:n], (;), :n, :n)], :n)],
        [BRM._RKPopulationPrior(:n, :Intercept, 0.0, 1.0),
            BRM._RKPopulationPrior(:n, :n, 0.0, 1.0)],
        [BRM._RKSampledParameter(:s, :Exponential, (1.0,), nothing, :s)],
        BRM._RKAssignmentSpec[],
        BRM._RKDerivedSpec[],
        Dict{Symbol,AbstractVector}(:y => df.y, :n => df.n),
        6,
        BRM._RKRanefBucket[],
        BRM._RKVectorParameter[],
        BRM._RKR2D2Prior[],
        BRM._RKHorseshoePrior[])
    prog = BRM._rk_emit_ast(plan, false)
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_normal_i_c, :x1, :loc1, :s1,
            :loc2, :s2), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, :loc1, :s1)),
            Expr(:call, :~, :b2, Expr(:call, :Normal, :loc2, :s2)),
            Expr(:call, :.+, :b1, Expr(:call, :.*, :b2, :x1)))),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :~, :n_, Expr(:call, :popefs_normal_i_c, :n,
            0.0, 1.0, 0.0, 1.0)),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :n_, :s))))
end

# Canonicalize for parser comparisons (drop line info; unwrap the
# toplevel block `Meta.parse` returns for a single expression).
function rk_strip_lines(x)
    x isa LineNumberNode && return nothing
    x isa Expr || return x
    args = Any[]
    for a in x.args
        a isa LineNumberNode && continue
        push!(args, rk_strip_lines(a))
    end
    Expr(x.head, args...)
end
function rk_parsed_surface(str::String)
    parsed = rk_strip_lines(Meta.parse(str))
    parsed.head === :block ? only(parsed.args) : parsed
end
rk_varying_stmts(main) = [a for a in main.args if a isa Expr &&
    a.head === :call && length(a.args) == 3 && a.args[1] === :~ &&
    a.args[3] isa Expr && a.args[3].args[1] in
    (:varying_draws, :varying_slice)]
# Submodel-def helpers: `rk_def_names` lists def names in order;
# `rk_def_body` fetches one def's body block.
rk_def_names(prog) = [d.args[1].args[1] for d in prog.defs]
rk_def_body(prog, name) =
    only(d.args[2] for d in prog.defs if d.args[1].args[1] === name)
# Shared-submodel helper: every leaf of type `T` in a def body (bodies
# must carry no baked values — every input rides a formal).
function rk_def_leaves(body, T::Type)
    found = Any[]
    walk(x) = begin
        x isa T && push!(found, x)
        x isa Expr && foreach(walk, x.args)
    end
    walk(body)
    found
end

@testset "ranef bucket AST matches surface" begin
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 + x | ID | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    prog = BRM._rk_emit_ast(plan, false)
    varying = rk_varying_stmts(prog.main)
    @test length(varying) == 2
    @test rk_strip_lines(varying[1]) == rk_parsed_surface(
        "ranef_draws_ID_g ~ varying_draws(g, [1, x]; eta = 1.0)")
    @test rk_strip_lines(varying[2]) == rk_parsed_surface(
        "ranef_mu_ID_g ~ varying_slice(ranef_draws_ID_g, 1:2)")
    ret = rk_def_body(prog, :popefs_normal_i_c_rid).args[end]
    @test :f1 in ret.args
    @test Expr(:call, :~, :mu, Expr(:call, :popefs_normal_i_c_rid, :x,
        :ranef_mu_ID_g, 0.0, 1.0, 0.0, 1.0)) in prog.main.args
end

@testset "ranef eta iff correlated" begin
    kinds = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 + x | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    correlated = rk_varying_stmts(BRM._rk_emit_ast(kinds, false).main)
    @test length(correlated) == 2
    @test rk_strip_lines(correlated[1]) == rk_parsed_surface(
        "ranef_draws_g ~ varying_draws(g, [1, x]; eta = 1.0)")
    @test rk_strip_lines(correlated[2]) == rk_parsed_surface(
        "ranef_mu_g ~ varying_slice(ranef_draws_g, 1:2)")
    ones = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    intercept1 = rk_varying_stmts(BRM._rk_emit_ast(ones, false).main)
    @test length(intercept1) == 2
    @test rk_strip_lines(intercept1[1]) ==
        rk_parsed_surface("ranef_draws_g ~ varying_draws(g, [1])")
    @test rk_strip_lines(intercept1[2]) ==
        rk_parsed_surface("ranef_mu_g ~ varying_slice(ranef_draws_g, 1)")
    slopes = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (0 + x | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    slope1 = rk_varying_stmts(BRM._rk_emit_ast(slopes, false).main)
    @test length(slope1) == 2
    @test rk_strip_lines(slope1[1]) ==
        rk_parsed_surface("ranef_draws_g ~ varying_draws(g, [x])")
    @test rk_strip_lines(slope1[2]) ==
        rk_parsed_surface("ranef_mu_g ~ varying_slice(ranef_draws_g, 1)")
    ret = rk_def_body(BRM._rk_emit_ast(ones, false), :popefs_normal_i_c_r).args[end]
    @test :f1 in ret.args
end

@testset "ranef dummy values in AST" begin
    codedf = (; df..., c=[2, 4, 2, 6, 4, 6])
    plan = BRM._brm_rk_plan(@brm codedf begin
        mu ~ 1 + x + (1 + c | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    varying = rk_varying_stmts(BRM._rk_emit_ast(plan, false).main)
    @test length(varying) == 2
    @test rk_strip_lines(varying[1]) == rk_parsed_surface(
        "ranef_draws_g ~ varying_draws(g, [1, dummy(c, 4), dummy(c, 6)]; eta = 1.0)")
    @test rk_strip_lines(varying[2]) == rk_parsed_surface(
        "ranef_mu_g ~ varying_slice(ranef_draws_g, 1:3)")
end

@testset "ranef interaction margin references derived def" begin
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 + x & z | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    prog = BRM._rk_emit_ast(plan, false)
    @test Expr(:(=), :int_x_x_z, Expr(:call, :.*, :x, :z)) in prog.main.args
    varying = rk_varying_stmts(prog.main)
    @test length(varying) == 2
    @test rk_strip_lines(varying[1]) == rk_parsed_surface(
        "ranef_draws_g ~ varying_draws(g, [1, int_x_x_z]; eta = 1.0)")
    @test rk_strip_lines(varying[2]) == rk_parsed_surface(
        "ranef_mu_g ~ varying_slice(ranef_draws_g, 1:2)")
end

@testset "ranef multi-target body order" begin
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
    varying = rk_varying_stmts(BRM._rk_emit_ast(plan, false).main)
    @test length(varying) == 3
    @test rk_strip_lines(varying[1]) == rk_parsed_surface(
        "ranef_draws_ID_g ~ varying_draws(g, [1, x]; eta = 1.0)")
    @test rk_strip_lines(varying[2]) == rk_parsed_surface(
        "ranef_mu1_ID_g ~ varying_slice(ranef_draws_ID_g, 1)")
    @test rk_strip_lines(varying[3]) == rk_parsed_surface(
        "ranef_mu2_ID_g ~ varying_slice(ranef_draws_ID_g, 2)")
    prog = BRM._rk_emit_ast(plan, false)
    # Both same-skeleton predictors share one latent def.
    ret = rk_def_body(prog, :popefs_normal_i_c_rid).args[end]
    @test :f1 in ret.args
    for (target, effect) in
            ((:mu1, :ranef_mu1_ID_g), (:mu2, :ranef_mu2_ID_g))
        @test Expr(:call, :~, target,
            Expr(:call, :popefs_normal_i_c_rid, :x, effect,
                0.0, 1.0, 0.0, 1.0)) in prog.main.args
    end
    # Both Gaussian responses emit bare `.~` statements (no stream
    # def); only the shared latent def remains.
    bare = [a for a in prog.main.args if a isa Expr && a.head === :call &&
        length(a.args) == 3 && a.args[1] === :.~]
    @test length(bare) == 2
    @test rk_def_names(prog) == [:popefs_normal_i_c_rid]
end

@testset "ranef bucket follows predictor rename" begin
    # Programmatic overlap (unreachable via @brm): the margin lines use the
    # renamed predictor, matching the renamed submodel LHS.
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 + x | ID | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    plan.columns[:mu] = plan.columns[:x]
    prog = BRM._rk_emit_ast(plan, false)
    # Effect names use the original predictor (rename-independent);
    # only the LHS and its call move to `mu_`.
    varying = rk_varying_stmts(prog.main)
    @test length(varying) == 2
    @test rk_strip_lines(varying[1]) == rk_parsed_surface(
        "ranef_draws_ID_g ~ varying_draws(g, [1, x]; eta = 1.0)")
    @test rk_strip_lines(varying[2]) == rk_parsed_surface(
        "ranef_mu_ID_g ~ varying_slice(ranef_draws_ID_g, 1:2)")
    @test Expr(:call, :~, :mu_,
        Expr(:call, :popefs_normal_i_c_rid, :x, :ranef_mu_ID_g,
            0.0, 1.0, 0.0, 1.0)) in prog.main.args
    ret = rk_def_body(prog, :popefs_normal_i_c_rid).args[end]
    @test :f1 in ret.args
end

@testset "leveled AST shapes" begin
    # Reference-coded categorical over K−1 etas: the lead plus its
    # tails spell inline in one bare `.~` response.
    brmi = @brm df begin
        eta1 ~ 1 + x
        eta2 ~ 1 + x
        eta3 ~ 1 + x
        c ~ CategoricalLogit(eta1, eta2, eta3)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :c,
        Expr(:., :CategoricalLogit, Expr(:tuple, :eta1, :eta2, :eta3)))
    # Ordered-logit: cutpoints implicit (no cutpoint statement).
    brmi = @brm df begin
        eta ~ 1 + x
        c ~ OrderedLogistic(eta)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :c,
        Expr(:., :OrderedLogistic, Expr(:tuple, :eta)))
    @test all(prog.main.args) do stmt
        !(stmt isa Expr && stmt.head === :call && length(stmt.args) >= 2 &&
            stmt.args[2] === :c_cutpoints)
    end
    @test all(prog.defs) do d
        all(d.args[2].args) do stmt
            !(stmt isa Expr && stmt.head === :call &&
                length(stmt.args) >= 2 && stmt.args[2] === :c_cutpoints)
        end
    end
    # Plain typed ordinal with tag calls; thresholds implicit.
    brmi = @brm df begin
        eta ~ 0 + x
        c ~ Ordinal(StoppingRatio(), ProbitLink(), eta)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :c,
        Expr(:., :Ordinal, Expr(:tuple,
            Expr(:call, :StoppingRatio), Expr(:call, :ProbitLink),
            :eta)))
    # Ordinal extras ride plan-level: the bare response carries no
    # extra statements or defs (thresholds and their coefs stay
    # implicit).
    brmi = @brm df begin
        eta ~ 0 + x
        c ~ Ordinal(StoppingRatio(), LogitLink(), eta;
            discrimination=2.0, per_threshold=(z,))
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :c,
        Expr(:., :Ordinal, Expr(:tuple,
            Expr(:call, :StoppingRatio), Expr(:call, :LogitLink),
            :eta)))
    @test length(prog.defs) == 1
    @test all(prog.main.args) do stmt
        !(stmt isa Expr && stmt.head === :call && length(stmt.args) >= 2 &&
            stmt.args[2] === :c_threshold_beta)
    end
    @test all(prog.defs) do d
        all(d.args[2].args) do stmt
            !(stmt isa Expr && stmt.head === :call &&
                length(stmt.args) >= 2 && stmt.args[2] === :c_threshold_beta)
        end
    end
    # A modeled scale skips the AST entirely (no affine, no priors, no
    # submodel def — the extension translates it plan-level); the
    # location side still emits.
    brmi = @brm df begin
        eta ~ 0 + x
        log(disc) ~ 0 + x
        c ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=disc)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :c,
        Expr(:., :Ordinal, Expr(:tuple,
            Expr(:call, :Cumulative), Expr(:call, :LogitLink),
            :eta)))
    @test length(prog.defs) == 1
    @test rk_def_names(prog) == [:popefs_normal_c]
    defined = Symbol[]
    for stmt in prog.main.args
        stmt isa Expr || continue
        if stmt.head === :(=) && stmt.args[1] isa Symbol
            push!(defined, stmt.args[1])
        elseif stmt.head === :call && !isempty(stmt.args) &&
                stmt.args[1] === :(~) && stmt.args[2] isa Symbol
            push!(defined, stmt.args[2])
        end
    end
    @test :eta in defined
    @test :disc ∉ defined
    # Shared-simplex multinomial + Dirichlet statement.
    brmi = @brm df begin
        s ~ Dirichlet(3, 1.0)
        obs ~ Multinomial(5, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test Expr(:call, :~, :s, Expr(:call, :Dirichlet,
        Expr(:vect, 1.0, 1.0, 1.0))) in prog.main.args
    @test prog.main.args[end] == Expr(:call, :.~, :obs,
        Expr(:., :Multinomial, Expr(:tuple, 5, :s, :obs_count_2,
            :obs_count_3)))
    # Plain categorical over simplex probs.
    brmi = @brm df begin
        s ~ Dirichlet([2.0, 5.0])
        b ~ Categorical(s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :b,
        Expr(:., :Categorical, Expr(:tuple, :s)))
end

@testset "spline AST shape" begin
    # Own 12-row frame: `s(x)` needs 10 unique axis values.
    xs = collect(range(-2.0, 2.0, length=12))
    zs = collect(range(0.0, 3.0, length=12))
    sdf = (; x=xs, z=zs, y=sin.(xs))
    brmi = @brm sdf begin
        mu ~ 1 + s(x)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_normal_i_s, :f1, :loc1, :s1),
            Expr(:block,
                Expr(:call, :~, :b1, Expr(:call, :Normal, :loc1, :s1)),
                Expr(:call, :.+,
                    :b1, Expr(:call, :spline, :f1)))),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :spline_basis,
            Expr(:parameters, Expr(:kw, :k, 10)),
            QuoteNode(:s_x), :x),
        Expr(:call, :~, :mu, Expr(:call, :popefs_normal_i_s,
            QuoteNode(:s_x), 0.0, 1.0)),
        Expr(:call, :~, :sigma, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :mu, :sigma))))
    # The declaration matches the parsed surface spelling exactly.
    @test prog.main.args[1] == Meta.parse("spline_basis(:s_x, x; k = 10)")
    # `t2(x, z)`: tuple-`k` declaration + inline summand.
    brmi = @brm sdf begin
        mu ~ 1 + t2(x, z)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_normal_i_s, :f1, :loc1, :s1),
            Expr(:block,
                Expr(:call, :~, :b1, Expr(:call, :Normal, :loc1, :s1)),
                Expr(:call, :.+,
                    :b1, Expr(:call, :spline, :f1)))),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :spline_basis,
            Expr(:parameters, Expr(:kw, :k, Expr(:tuple, 5, 5))),
            QuoteNode(:t2_x_z), :x, :z),
        Expr(:call, :~, :mu, Expr(:call, :popefs_normal_i_s,
            QuoteNode(:t2_x_z), 0.0, 1.0)),
        Expr(:call, :~, :sigma, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :mu, :sigma))))
    @test prog.main.args[1] ==
        Meta.parse("spline_basis(:t2_x_z, x, z; k = (5, 5))")
end

@testset "monotonic AST shape" begin
    brmi = @brm df begin
        mu ~ 1 + mo(c)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_normal_i_mo, :x1, :f1, :loc1,
            :s1, :loc2, :s2), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, :loc1, :s1)),
            Expr(:call, :~, :b2, Expr(:call, :Normal, :loc2, :s2)),
            Expr(:call, :.+,
                :b1, Expr(:call, :.*,
                    :b2, Expr(:call, :mo, :x1, :f1))))),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :~, :mu, Expr(:call, :popefs_normal_i_mo, :c_idx,
            :mo_c_simplex_incr, 0.0, 1.0, 0.0, 1.0)),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :~, :mo_c_simplex_incr,
            Expr(:call, :Dirichlet, Expr(:vect, 1.0, 1.0, 1.0))),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :mu, :s))))
    # The summand matches the parsed surface spelling exactly.
    ret = rk_def_body(prog, :popefs_normal_i_mo).args[end]
    @test rk_strip_lines(ret) ==
        rk_parsed_surface("b1 .+ b2 .* mo(x1, f1)")
    # `mo1(c)`: beta-free inline summand, no second coefficient.
    brmi = @brm df begin
        mu ~ 1 + mo1(c)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_normal_i_mo1, :x1, :f1, :loc1,
            :s1), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, :loc1, :s1)),
            Expr(:call, :.+,
                :b1, Expr(:call, :mo1, :x1, :f1)))),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :~, :mu, Expr(:call, :popefs_normal_i_mo1, :c_idx,
            :mo1_c_simplex_incr, 0.0, 1.0)),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :~, :mo1_c_simplex_incr,
            Expr(:call, :Dirichlet, Expr(:vect, 1.0, 1.0, 1.0))),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :mu, :s))))
    # Coefficient-free `mo1` predictor: no scalar terms, so no `popefs`
    # def — the affine stays inline exactly as before.
    brmi = @brm df begin
        mu ~ mo1(c)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test isempty(prog.defs)
    @test prog.main == Expr(:block,
        Expr(:(=), :mu,
            Expr(:call, :mo1, :c_idx, :mo1_c_simplex_incr)),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :~, :mo1_c_simplex_incr,
            Expr(:call, :Dirichlet, Expr(:vect, 1.0, 1.0, 1.0))),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :mu, :s))))
end

@testset "dar AST shape" begin
    # Own frame: `dar` needs a strictly increasing time axis.
    tdf = (; t=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        y=[0.5, -0.2, 0.1, 0.9, 1.4, 1.1])
    brmi = @brm tdf begin
        mu ~ 1 + dar(t)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    # The dar summand rides inside the shared submodel (the trajectory
    # scalars ride formals); their statements stay top-level.
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_normal_i_dar, :f1, :f2, :loc1,
            :s1), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, :loc1, :s1)),
            Expr(:call, :.+,
                :b1, Expr(:call, :dar, :f1, :f2)))),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :~, :dar_mu_t_beta,
            Expr(:call, :truncated,
                Expr(:call, :Normal, 0.5, 0.2), 0, 1)),
        Expr(:call, :~, :dar_mu_t_sigma,
            Expr(:call, :HalfNormal, 0.2)),
        Expr(:call, :~, :mu, Expr(:call, :popefs_normal_i_dar,
            :dar_mu_t_beta, :dar_mu_t_sigma, 0.0, 1.0)),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :mu, :s))))
    # Both dar spellings match the parsed surface exactly.
    @test rk_strip_lines(prog.main.args[1]) == rk_parsed_surface(
        "dar_mu_t_beta ~ truncated(Normal(0.5, 0.2), 0, 1)")
    ret = rk_def_body(prog, :popefs_normal_i_dar).args[end]
    @test rk_strip_lines(ret) ==
        rk_parsed_surface("b1 .+ dar(f1, f2)")
    # Prior overrides ride the preamble statements.
    brmi = @brm tdf begin
        mu ~ 1 + dar(t)
        ar(mu, dar(t)) ~ Normal(0.6, 0.1)
        sd(mu, dar(t)) ~ Normal(0, 0.3)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[1] == Expr(:call, :~, :dar_mu_t_beta,
        Expr(:call, :truncated, Expr(:call, :Normal, 0.6, 0.1), 0, 1))
    @test prog.main.args[2] == Expr(:call, :~, :dar_mu_t_sigma,
        Expr(:call, :HalfNormal, 0.3))
end

@testset "r2d2 AST shape" begin
    # Override-free: the affine inlines with program-global coefficient
    # names (no priors to state), followed by the bare `r2d2(...)`
    # declaration; R2/phi/tau sample top-level (SB spellings, minted).
    brmi = @brm df begin
        mu ~ 1 + x + z
        effect(mu, :) ~ r2d2(R2=Beta(2, 5), alpha=0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test isempty(prog.defs)
    @test prog.main == Expr(:block,
        Expr(:(=), :mu, Expr(:call, :.+,
            :mu_b1, Expr(:call, :.*, :mu_b2, :x),
            Expr(:call, :.*, :mu_b3, :z))),
        Expr(:call, :r2d2, :mu, :r2d2_mu_R2, :r2d2_mu_phi,
            :r2d2_mu_tau_bsv),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :~, :r2d2_mu_R2, Expr(:call, :Beta, 2.0, 5.0)),
        Expr(:call, :~, :r2d2_mu_tau_bsv, Expr(:call, :HalfNormal, 1.0)),
        Expr(:call, :~, :r2d2_mu_phi,
            Expr(:call, :Dirichlet, Expr(:vect, 0.5, 0.5))),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :mu, :s))))
    # The declaration matches the parsed surface exactly.
    @test rk_strip_lines(prog.main.args[2]) == rk_parsed_surface(
        "r2d2(mu, r2d2_mu_R2, r2d2_mu_phi, r2d2_mu_tau_bsv)")
    # An explicit Normal rides the submodel as a share-0 override
    # (simplex columns stay bare locals); a data `tau_bsv` inlines.
    brmi = @brm df begin
        mu ~ 1 + x + z
        effect(mu, :) ~ r2d2(tau_bsv=2.0)
        effect(mu, x) ~ Normal(0, 3)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.defs[1] == Expr(:(=),
        Expr(:call, :popefs_normal_i_c_c_s2, :x1, :x2, :loc2, :s2),
        Expr(:block,
            Expr(:call, :~, :b2, Expr(:call, :Normal, :loc2, :s2)),
            Expr(:call, :.+, :b1, Expr(:call, :.*, :b2, :x1),
                Expr(:call, :.*, :b3, :x2))))
    @test prog.main.args[1] ==
        Expr(:call, :~, :mu, Expr(:call, :popefs_normal_i_c_c_s2, :x,
            :z, 0.0, 3.0))
    @test prog.main.args[2] ==
        Expr(:call, :r2d2, :mu, :r2d2_mu_R2, :r2d2_mu_phi, 2.0)
    @test rk_strip_lines(prog.main.args[2]) ==
        rk_parsed_surface("r2d2(mu, r2d2_mu_R2, r2d2_mu_phi, 2.0)")
    @test prog.main.args[5] == Expr(:call, :~, :r2d2_mu_phi,
        Expr(:call, :Dirichlet, Expr(:vect, 1.0)))
    # Inline coefficients namespace against data (a `mu_b1` column
    # would otherwise merge with the minted name silently).
    dfb = (; df..., mu_b1=[0.2, -0.1, 0.4, 0.0, 0.3, -0.3])
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(@brm dfb begin
        mu ~ 1 + x + mu_b1
        effect(mu, :) ~ r2d2()
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end))
    @test prog.main.args[1] == Expr(:(=), :mu, Expr(:call, :.+,
        :mu_b1_, Expr(:call, :.*, :mu_b2, :x),
        Expr(:call, :.*, :mu_b3, :mu_b1)))
end

@testset "horseshoe AST shape" begin
    # Mixed Normal/Horseshoe: the Horseshoe slot states a literal-scale
    # `~ Horseshoe(...)` inside the shared submodel (no loc/s formals);
    # the hs slot pattern joins the def name (the thin surface takes
    # literals only, so scales are body identity).
    brmi = @brm df begin
        mu ~ 1 + x + z
        effect(mu, x) ~ Horseshoe(local_scale=0.5)
        effect(mu, z) ~ Normal(0, 3)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.defs[1] == Expr(:(=),
        Expr(:call, :popefs_normal_i_c_c_s1_3_hs2_0p5_1p0,
            :x1, :x2, :loc1, :s1, :loc3, :s3),
        Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, :loc1, :s1)),
            Expr(:call, :~, :b2, Expr(:call, :Horseshoe,
                Expr(:kw, :local_scale, 0.5),
                Expr(:kw, :global_scale, 1.0))),
            Expr(:call, :~, :b3, Expr(:call, :Normal, :loc3, :s3)),
            Expr(:call, :.+, :b1, Expr(:call, :.*, :b2, :x1),
                Expr(:call, :.*, :b3, :x2))))
    @test prog.main.args[1] ==
        Expr(:call, :~, :mu,
            Expr(:call, :popefs_normal_i_c_c_s1_3_hs2_0p5_1p0, :x,
                :z, 0.0, 1.0, 0.0, 3.0))
    # The statement matches the parsed corpus-55 surface spelling exactly.
    @test rk_strip_lines(prog.defs[1].args[2].args[2]) ==
        rk_parsed_surface("b2 ~ Horseshoe(local_scale=0.5, global_scale=1.0)")
    # Default scales emit the bare `Horseshoe()` call (corpus `b1` shape).
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x
        effect(mu, x) ~ Horseshoe()
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end), false)
    @test prog.defs[1] == Expr(:(=),
        Expr(:call, :popefs_normal_i_c_s1_hs2_1p0_1p0,
            :x1, :loc1, :s1),
        Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, :loc1, :s1)),
            Expr(:call, :~, :b2, Expr(:call, :Horseshoe)),
            Expr(:call, :.+, :b1, Expr(:call, :.*, :b2, :x1))))
    @test rk_strip_lines(prog.defs[1].args[2].args[2]) ==
        rk_parsed_surface("b2 ~ Horseshoe()")
    # Same horseshoe pattern shares one def; different scales split
    # (the scales are body identity, so they join the lattice name).
    dfj = (y1=[0.5, -0.2, 0.1], y2=[0.1, 0.3, -0.4], x=[-1.0, 0.0, 1.0])
    shared = @brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        effect(mu1, x) ~ Horseshoe(local_scale=0.5)
        effect(mu2, x) ~ Horseshoe(local_scale=0.5)
        s ~ Exponential(1)
        y1 ~ Normal(mu1, s)
        y2 ~ Normal(mu2, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(shared), false)
    latdefs = [d for d in prog.defs
               if startswith(string(d.args[1].args[1]), "popefs")]
    @test length(latdefs) == 1
    split = @brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        effect(mu1, x) ~ Horseshoe(local_scale=0.5)
        effect(mu2, x) ~ Horseshoe(local_scale=0.25)
        s ~ Exponential(1)
        y1 ~ Normal(mu1, s)
        y2 ~ Normal(mu2, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(split), false)
    latdefs = [d for d in prog.defs
               if startswith(string(d.args[1].args[1]), "popefs")]
    @test length(latdefs) == 2
    # A Horseshoe on a discrimination predictor fails closed (it skips
    # the AST, where the Horseshoe lowers).
    @test_throws "Horseshoe on discrimination predictors" BRM._rk_emit_ast(
        BRM._brm_rk_plan(@brm df begin
            eta ~ 0 + x
            log(disc) ~ 0 + x
            effect(disc, x) ~ Horseshoe()
            c ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=disc)
        end), false)
end

@testset "exact gp AST shape" begin
    brmi = @brm df begin
        mu ~ 1 + gp(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    plate = Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
        Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, :y)),
            Expr(:block, Expr(:call, :~,
                Expr(:ref, :z_gp, :i),
                Expr(:call, :Normal, 0.0, 1.0)))))
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_normal_i_gp, :f1, :loc1, :s1),
            Expr(:block,
                Expr(:call, :~, :b1, Expr(:call, :Normal, :loc1, :s1)),
                Expr(:call, :.+, :b1, :f1))),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :~, :rho_gp, Expr(:call, :LogNormal, 0.0, 1.0)),
        Expr(:call, :~, :sigma_gp, Expr(:call, :LogNormal, 0.0, 1.0)),
        plate,
        Expr(:(=), :f_gp, Expr(:call, :gp_chol_latent,
            Expr(:call, :gp_exp_quad_cov, :x, :sigma_gp, :rho_gp, 1e-9),
            :z_gp)),
        Expr(:call, :~, :mu, Expr(:call, :popefs_normal_i_gp, :f_gp,
            0.0, 1.0)),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :mu, :s))))
    # Hyper overrides ride the preamble with their families.
    brmi = @brm df begin
        mu ~ 1 + gp(x)
        length_scale(:, gp(x)) ~ Gamma(2, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test Expr(:call, :~, :rho_gp, Expr(:call, :Gamma, 2.0, 1.0)) in prog.main.args
    # Overlap alpha-renames the submodel LHS; the GP preamble is unaffected.
    # (Overlap is unconstructible from formulas — a data-named LHS
    # classifies as an observation — so force it by plan surgery.)
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + gp(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    plan.columns[:mu] = plan.columns[:y]
    prog = BRM._rk_emit_ast(plan, false)
    @test Expr(:call, :~, :mu_, Expr(:call, :popefs_normal_i_gp,
        :f_gp, 0.0, 1.0)) in prog.main.args
    @test Expr(:call, :.~, :y,
        Expr(:., :Normal, Expr(:tuple, :mu_, :s))) in prog.main.args
    @test Expr(:(=), :f_gp, Expr(:call, :gp_chol_latent,
        Expr(:call, :gp_exp_quad_cov, :x, :sigma_gp, :rho_gp, 1e-9),
        :z_gp)) in prog.main.args
end

@testset "hsgp AST shape" begin
    brmi = @brm df begin
        mu ~ 1 + hsgp(x; k=4)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_normal_i_h, :f1, :loc1, :s1),
            Expr(:block,
                Expr(:call, :~, :b1, Expr(:call, :Normal, :loc1, :s1)),
                Expr(:call, :.+,
                    :b1, Expr(:call, :hsgp, :f1)))),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :hsgp_basis,
            Expr(:parameters, Expr(:kw, :k, 4), Expr(:kw, :c, 1.5),
                Expr(:kw, :iso, true)),
            QuoteNode(:hsgp_x), :x),
        Expr(:call, :~, :mu, Expr(:call, :popefs_normal_i_h,
            QuoteNode(:hsgp_x), 0.0, 1.0)),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :mu, :s))))
    # The declaration matches the parsed surface spelling exactly.
    @test prog.main.args[1] ==
        Meta.parse("hsgp_basis(:hsgp_x, x; k = 4, c = 1.5, iso = true)")
    # Aniso multi-axis: tuple-`k`/`c` declaration + inline summand.
    brmi = @brm df begin
        mu ~ 1 + hsgp(x, z; k=(4, 3), c=(1.5, 2.0), iso=false)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_normal_i_h, :f1, :loc1, :s1),
            Expr(:block,
                Expr(:call, :~, :b1, Expr(:call, :Normal, :loc1, :s1)),
                Expr(:call, :.+,
                    :b1, Expr(:call, :hsgp, :f1)))),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :hsgp_basis,
            Expr(:parameters, Expr(:kw, :k, Expr(:tuple, 4, 3)),
                Expr(:kw, :c, Expr(:tuple, 1.5, 2.0)),
                Expr(:kw, :iso, false)),
            QuoteNode(:hsgp_x_z), :x, :z),
        Expr(:call, :~, :mu, Expr(:call, :popefs_normal_i_h,
            QuoteNode(:hsgp_x_z), 0.0, 1.0)),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :mu, :s))))
    @test prog.main.args[1] == Meta.parse("hsgp_basis(:hsgp_x_z, x, z; " *
        "k = (4, 3), c = (1.5, 2.0), iso = false)")
end

@testset "ar AST shape" begin
    brmi = @brm df begin
        mu ~ 1 + ar(x; p=1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog isa BRM._RKEmittedProgram
    scan = Expr(:macrocall, Symbol("@scan"), LineNumberNode(0),
        Expr(:block,
            Expr(:call, :~,
                Expr(:ref, :ar_mu_x, 1),
                Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:for, Expr(:(=), :t, Expr(:call, :(:), 2, :T)),
                Expr(:block,
                    Expr(:call, :~,
                        :eps_ar_mu_x,
                        Expr(:call, :Normal, 0.0, 1.0)),
                    Expr(:(=), Expr(:ref, :ar_mu_x, :t),
                        Expr(:call, :+,
                            Expr(:call, :*,
                                :phi_ar_mu_x,
                                Expr(:ref, :ar_mu_x,
                                    Expr(:call, :-, :t, 1))),
                            :eps_ar_mu_x))))))
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_normal_i_ar, :f1, :loc1, :s1,
            :loc2, :s2), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, :loc1, :s1)),
            Expr(:call, :~, :b2, Expr(:call, :Normal, :loc2, :s2)),
            Expr(:call, :.+, :b1, Expr(:call, :.*, :b2, :f1)))),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :~, :phi_raw_ar_mu_x, Expr(:call, :Normal, 0.0, 1.0)),
        scan,
        Expr(:(=), :phi_ar_mu_x,
            Expr(:call, :tanh, :phi_raw_ar_mu_x)),
        Expr(:call, :~, :mu, Expr(:call, :popefs_normal_i_ar, :ar_mu_x,
            0.0, 1.0, 0.0, 1.0)),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :mu, :s))))
    # The scan block matches the parsed surface spelling exactly.
    @test rk_strip_lines(prog.main.args[2]) == rk_parsed_surface(
        "@scan begin\n" *
        "ar_mu_x[1] ~ Normal(0.0, 1.0)\n" *
        "for t in 2:T\n" *
        "eps_ar_mu_x ~ Normal(0.0, 1.0)\n" *
        "ar_mu_x[t] = phi_ar_mu_x * ar_mu_x[t - 1] + eps_ar_mu_x\n" *
        "end\nend")
    # A `:`-wide prior rides the beta's submodel-local statement.
    brmi = @brm df begin
        mu ~ 1 + ar(x; p=1)
        effect(mu, :) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    popefs_body = prog.defs[1].args[2].args
    @test Expr(:call, :~, :b2, Expr(:call, :Normal, :loc2, :s2)) in popefs_body
    @test Expr(:call, :~, :mu, Expr(:call, :popefs_normal_i_ar, :ar_mu_x,
        0.0, 2.0, 0.0, 2.0)) in prog.main.args
end

@testset "me AST shape" begin
    brmi = @brm df begin
        mu ~ 1 + me(x, 0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog isa BRM._RKEmittedProgram
    plate = Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
        Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, :x)),
            Expr(:block,
                Expr(:call, :~,
                    Expr(:ref, :me_x, :i),
                    Expr(:call, :Normal, 0.0, 1.0)))))
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_normal_i_me, :f1, :loc1, :s1,
            :loc2, :s2), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, :loc1, :s1)),
            Expr(:call, :~, :b2, Expr(:call, :Normal, :loc2, :s2)),
            Expr(:call, :.+, :b1, Expr(:call, :.*, :b2, :f1)))),
    ]
    @test prog.main == Expr(:block,
        plate,
        Expr(:call, :~, :mu, Expr(:call, :popefs_normal_i_me, :me_x,
            0.0, 1.0, 0.0, 1.0)),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :mu, :s))),
        Expr(:call, :.~, :x,
            Expr(:., :Normal, Expr(:tuple, :me_x, 0.5))))
    # Both the main response and the `me` observation emit bare
    # (no stream def); only the latent def remains.
    @test rk_def_names(prog) == [:popefs_normal_i_me]
    # The plate block matches the parsed surface spelling exactly.
    @test rk_strip_lines(prog.main.args[1]) == rk_parsed_surface(
        "@plate for i in eachindex(x)\n" *
        "me_x[i] ~ Normal(0.0, 1.0)\n" *
        "end")
    # A `latent(...)` override rides the plate's shared-scalar args.
    brmi = @brm df begin
        mu ~ 1 + me(x, 0.5)
        latent(mu, me(x)) ~ Normal(0.5, 1.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[1] == Expr(:macrocall, Symbol("@plate"),
        LineNumberNode(0),
        Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, :x)),
            Expr(:block,
                Expr(:call, :~,
                    Expr(:ref, :me_x, :i),
                    Expr(:call, :Normal, 0.5, 1.5)))))
    # A `:`-wide prior rides the beta's submodel-local statement.
    brmi = @brm df begin
        mu ~ 1 + me(x, 0.5)
        effect(mu, :) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    popefs_body = prog.defs[1].args[2].args
    @test Expr(:call, :~, :b2, Expr(:call, :Normal, :loc2, :s2)) in popefs_body
    @test Expr(:call, :~, :mu, Expr(:call, :popefs_normal_i_me, :me_x,
        0.0, 2.0, 0.0, 2.0)) in prog.main.args
end

@testset "distributional scale AST" begin
    brmi = @brm df begin
        mu ~ 1 + x
        log(sigma) ~ 1 + z
        y ~ Normal(mu, sigma)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    # Both same-skeleton predictors share one latent def.
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_normal_i_c, :x1, :loc1, :s1,
            :loc2, :s2), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, :loc1, :s1)),
            Expr(:call, :~, :b2, Expr(:call, :Normal, :loc2, :s2)),
            Expr(:call, :.+, :b1, Expr(:call, :.*, :b2, :x1)))),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :~, :mu, Expr(:call, :popefs_normal_i_c, :x,
            0.0, 1.0, 0.0, 1.0)),
        Expr(:call, :~, :sigma, Expr(:call, :popefs_normal_i_c, :z,
            0.0, 1.0, 0.0, 1.0)),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple,
                :mu, Expr(:., :exp, Expr(:tuple, :sigma))))))
    # Identity-link scale reads the affine bare.
    brmi = @brm df begin
        mu ~ 1 + x
        sigma ~ 1 + z
        y ~ Normal(mu, sigma)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :y,
        Expr(:., :Normal, Expr(:tuple, :mu, :sigma)))
    # NB2 dispersion as a predictor.
    brmi = @brm df begin
        log(mu) ~ 1 + x
        log(phi) ~ 1 + z
        c ~ NegativeBinomial2(mu, phi)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :c,
        Expr(:., :NegativeBinomial2, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :mu)),
            Expr(:., :exp, Expr(:tuple, :phi)))))
    # Gamma shape inverts at both use positions.
    brmi = @brm df begin
        log(mu) ~ 1 + x
        log(alpha) ~ 1 + x
        z ~ Gamma(alpha, mu / alpha)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] == Expr(:call, :.~, :z,
        Expr(:., :Gamma, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :alpha)),
            Expr(:call, :./,
                Expr(:., :exp, Expr(:tuple, :mu)),
                Expr(:., :exp, Expr(:tuple, :alpha))))))
end

@testset "submodel defs resolve at every call" begin
    # Layer-3 shape contract: defs are surface-spelling
    # `name(args...) = begin ... end` forms; every main-block `~` call
    # whose head names a def resolves (arity matches); every def is
    # called at least once.
    models = [
        @brm(df, begin
            mu ~ 1 + x
            sigma ~ Exponential(1)
            y ~ Normal(mu, sigma)
        end),
        @brm(df, begin
            logit(p) ~ 1 + x
            b ~ Binomial(h, p)
        end),
        @brm(df, begin
            mu ~ 1 + x + (1 | g)
            s ~ Exponential(1)
            y ~ Normal(mu, s)
        end),
        @brm((y1=[0.5, -0.2, 0.1, 0.9, 1.4, 1.1],
                y2=[0.1, 0.3, -0.4, 0.2, 0.8, -0.1],
                x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]), begin
            mu1 ~ 1 + x
            mu2 ~ 1 + x
            L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
            [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
        end),
    ]
    for brmi in models
        prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
        @test !isempty(prog.defs)
        arities = Dict{Symbol,Int}()
        for d in prog.defs
            @test d.head === :(=) && d.args[1].head === :call
            @test d.args[2].head === :block
            # No duplicate def names.
            @test d.args[1].args[1] ∉ keys(arities)
            arities[d.args[1].args[1]] = length(d.args[1].args) - 1
        end
        calls = Any[]
        for stmt in prog.main.args
            stmt isa Expr && stmt.head === :call && length(stmt.args) == 3 &&
                stmt.args[1] === :~ && stmt.args[3] isa Expr &&
                stmt.args[3].head === :call || continue
            head = stmt.args[3].args[1]
            head in keys(arities) || continue
            push!(calls, stmt.args[3])
        end
        @test !isempty(calls)
        for c in calls
            @test length(c.args) - 1 == arities[c.args[1]]
        end
        @test Set(c.args[1] for c in calls) == Set(keys(arities))
    end
end

@testset "correlated AST shape" begin
    dfj = (y1=[0.5, -0.2, 0.1, 0.9, 1.4, 1.1],
        y2=[0.1, 0.3, -0.4, 0.2, 0.8, -0.1],
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    brmi = @brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1), shape=2)
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    # The joint response emits bare like every other response
    # (plain `~`, row-grouped) — defs hold the one predictor submodel
    # both same-shape predictors share.
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_normal_i_c, :x1, :loc1, :s1,
            :loc2, :s2), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, :loc1, :s1)),
            Expr(:call, :~, :b2, Expr(:call, :Normal, :loc2, :s2)),
            Expr(:call, :.+, :b1, Expr(:call, :.*, :b2, :x1)))),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :~, :mu1, Expr(:call, :popefs_normal_i_c, :x,
            0.0, 1.0, 0.0, 1.0)),
        Expr(:call, :~, :mu2, Expr(:call, :popefs_normal_i_c, :x,
            0.0, 1.0, 0.0, 1.0)),
        Expr(:call, :~, :L_res, Expr(:call, :LKJCovarianceFactor, 2,
            Expr(:call, :Exponential, 1.0), 2.0)),
        Expr(:call, :~, Expr(:vect, :y1, :y2),
            Expr(:call, :MvNormalCholesky, Expr(:vect, :mu1, :mu2),
                :L_res)))
    # Both joint spellings match the parsed surface exactly.
    @test rk_strip_lines(prog.main.args[3]) == rk_parsed_surface(
        "L_res ~ LKJCovarianceFactor(2, Exponential(1.0), 2.0)")
    @test rk_strip_lines(prog.main.args[4]) == rk_parsed_surface(
        "[y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)")
    # Sampled scale hyperparameters emit as bare names.
    brmi = @brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        tau ~ Exponential(1)
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(tau))
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[3] == Expr(:call, :~, :tau,
        Expr(:call, :Exponential, 1.0))
    @test prog.main.args[4] == Expr(:call, :~, :L_res,
        Expr(:call, :LKJCovarianceFactor, 2,
            Expr(:call, :Exponential, :tau), 1.0))
    @test rk_strip_lines(prog.main.args[4]) == rk_parsed_surface(
        "L_res ~ LKJCovarianceFactor(2, Exponential(tau), 1.0)")
    # K=3: three outcomes, three means, width-3 stem.
    df3 = merge(dfj, (; y3=[-0.3, 0.7, 0.2, -0.1, 0.4, 0.6]))
    brmi = @brm df3 begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        mu3 ~ 1 + x
        L3 ~ LKJCovarianceFactor(3; scale_prior=Exponential(1))
        [y1, y2, y3] ~ MvNormalCholesky([mu1, mu2, mu3], L3)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    @test prog.main.args[end] ==
        Expr(:call, :~, Expr(:vect, :y1, :y2, :y3),
            Expr(:call, :MvNormalCholesky, Expr(:vect, :mu1, :mu2, :mu3),
                :L3))
end

# Shared-submodel invariant battery (snag consume-post-fli-93fddce6): one
# model per def-shape class. The same-name = same-body invariant must
# hold across ALL of them jointly.
rk_shared_m1 = @brm df begin
    mu ~ 1 + x
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
rk_shared_m2 = @brm df begin
    mu ~ 1 + z
    effect(mu, z) ~ Normal(5.0, 2.0)
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
rk_shared_m3 = @brm df begin
    mu ~ 1
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
rk_shared_m4 = @brm df begin
    mu ~ 1 + factor(g; ref=3)
    effect(mu, g) ~ Normal(0, 2)
    s ~ Exponential(1)
    y ~ Normal(mu, s)
end
rk_shared_m5 = @brm df begin
    mu ~ 1 + mo(c)
    s ~ Exponential(1)
    y ~ Normal(mu, s)
end
rk_shared_sdf = (;
    x=collect(range(-2.0, 2.0, length=12)),
    z=collect(range(0.0, 3.0, length=12)),
    y=sin.(collect(range(-2.0, 2.0, length=12))))
rk_shared_m6 = @brm rk_shared_sdf begin
    mu ~ 1 + s(x)
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
rk_shared_m7 = @brm df begin
    mu ~ 1 + x + (1 + x | ID | g)
    s ~ Exponential(1)
    y ~ Normal(mu, s)
end
rk_shared_m8 = @brm df begin
    mu ~ 1 + x + (1 | g)
    s ~ Exponential(1)
    y ~ Normal(mu, s)
end
rk_shared_m9 = @brm df begin
    eta1 ~ 1 + x
    eta2 ~ 1 + x
    eta3 ~ 1 + x
    c ~ CategoricalLogit(eta1, eta2, eta3)
end
rk_shared_m10 = @brm df begin
    f1 ~ 1 + x
    f2 ~ 1 + x
    f3 ~ 1 + x
    c ~ CategoricalLogit(f1, f2, f3)
end
rk_shared_m14 = @brm df begin
    eta1 ~ 1 + x
    eta2 ~ 1 + x
    g ~ CategoricalLogit(eta1, eta2)
end
rk_shared_m15 = @brm df begin
    eta1 ~ 1 + x
    h ~ CategoricalLogit(eta1)
end
rk_shared_m11 = @brm df begin
    s ~ Dirichlet(3, 1.0)
    obs ~ Multinomial(5, s)
end
rk_shared_m12 = @brm df begin
    eta ~ 1 + x
    b ~ BernoulliLogit(eta)
end
rk_shared_m13 = @brm df begin
    mu ~ 1 + x + z
    effect(mu, :) ~ r2d2(R2=Beta(2, 5), alpha=0.5)
    s ~ Exponential(1)
    y ~ Normal(mu, s)
end
const RK_SHARED_BATTERY = Any[
    rk_shared_m1, rk_shared_m2, rk_shared_m3, rk_shared_m4,
    rk_shared_m5, rk_shared_m6, rk_shared_m7, rk_shared_m8,
    rk_shared_m9, rk_shared_m10, rk_shared_m11, rk_shared_m12,
    rk_shared_m13, rk_shared_m14, rk_shared_m15,
]

@testset "shared submodels: same name means same body" begin
    # Cross-model invariant: a def name denotes exactly one body across
    # every emitted program, so joint sessions share module bindings
    # safely. Fails while the emitter mints per-model defs.
    seen = Dict{Symbol,Expr}()
    for m in RK_SHARED_BATTERY
        prog = BRM._rk_emit_ast(BRM._brm_rk_plan(m), false)
        for d in prog.defs
            nm = d.args[1].args[1]
            if haskey(seen, nm)
                @test seen[nm] == d
            else
                seen[nm] = d
            end
        end
    end
end

@testset "shared submodels: same skeleton shares one def" begin
    # Same term skeleton, different columns and priors: one shared def,
    # while the use-sites carry the differing values.
    pa = BRM._rk_emit_ast(BRM._brm_rk_plan(rk_shared_m1), false)
    pb = BRM._rk_emit_ast(BRM._brm_rk_plan(rk_shared_m2), false)
    islatent(d) =
        startswith(string(d.args[1].args[1]), "popefs")
    la = only(d for d in pa.defs if islatent(d))
    lb = only(d for d in pb.defs if islatent(d))
    @test la == lb
    function latent_use(prog, defname)
        only(u for u in prog.main.args if u isa Expr && u.head === :call &&
            u.args[1] === :(~) && u.args[3] isa Expr &&
            u.args[3].args[1] === defname)
    end
    ua = latent_use(pa, la.args[1].args[1])
    ub = latent_use(pb, lb.args[1].args[1])
    @test ua != ub
    @test :z in ub.args[3].args
    @test 5.0 in ub.args[3].args && 2.0 in ub.args[3].args
end

@testset "shared submodels: def bodies carry no values" begin
    # Every body input rides a formal: no baked Float64 priors, no
    # quoted ids — each body is a pure function of its lattice name.
    for m in RK_SHARED_BATTERY
        prog = BRM._rk_emit_ast(BRM._brm_rk_plan(m), false)
        for d in prog.defs
            @test isempty(rk_def_leaves(d.args[2], Float64))
            @test isempty(rk_def_leaves(d.args[2], QuoteNode))
        end
    end
end

@testset "shared submodels: one def per skeleton in a program" begin
    # Two same-shape predictors in one program reuse one latent def
    # with two use-sites (intra-program sharing).
    dfj = (y1=[0.5, -0.2, 0.1, 0.9, 1.4, 1.1],
        y2=[0.1, 0.3, -0.4, 0.2, 0.8, -0.1],
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    brmi = @brm dfj begin
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1), shape=2)
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi), false)
    latdefs = [d for d in prog.defs
               if startswith(string(d.args[1].args[1]), "popefs")]
    @test length(latdefs) == 1
    uses = [u for u in prog.main.args if u isa Expr && u.head === :call &&
        u.args[1] === :(~) && u.args[3] isa Expr &&
        u.args[3].args[1] === latdefs[1].args[1].args[1]]
    @test length(uses) == 2
end

@testset "fused heads and GLM objects response shapes" begin
    # Fused-head emission is now default-on for the six families the
    # thin layer desugars pre-spine. Eligible canonical GLMs instead use
    # the stronger object spelling. Each fused spelling rewrites to
    # exactly the decomposed twin asserted beside it (same roles, same
    # order), and evidence/weights wrappers recurse, so the lowered plan
    # is identical by construction.
    brmi = @brm df begin
        eta ~ 1 + x
        b ~ BernoulliLogit(eta)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test BRM._rk_emit_ast(plan, true).main.args[end] ==
        Expr(:call, :~, :b,
            Expr(:call, :BernoulliLogitGLM, :b_X, :eta_alpha, :eta_beta))
    @test BRM._rk_emit_ast(plan, false).main.args[end] ==
        Expr(:call, :.~, :b,
            Expr(:., :Bernoulli, Expr(:tuple,
                Expr(:., :logistic, Expr(:tuple, :eta)))))
    brmi = @brm df begin
        log(mu) ~ 1 + x
        c ~ Poisson(mu)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test BRM._rk_emit_ast(plan, true).main.args[end] ==
        Expr(:call, :~, :c,
            Expr(:call, :PoissonLogGLM, :c_X, :mu_alpha, :mu_beta))
    @test BRM._rk_emit_ast(plan, false).main.args[end] ==
        Expr(:call, :.~, :c,
            Expr(:., :Poisson, Expr(:tuple,
                Expr(:., :exp, Expr(:tuple, :mu)))))
    brmi = @brm df begin
        logit(p) ~ 1 + x
        b ~ Binomial(h, p)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test BRM._rk_emit_ast(plan, true).main.args[end] ==
        Expr(:call, :.~, :b,
            Expr(:., :BinomialLogit, Expr(:tuple, :h, :p)))
    @test BRM._rk_emit_ast(plan, false).main.args[end] ==
        Expr(:call, :.~, :b,
            Expr(:., :Binomial, Expr(:tuple, :h,
                Expr(:., :logistic, Expr(:tuple, :p)))))
    brmi = @brm df begin
        log(mu) ~ 1 + x
        phi ~ Exponential(1)
        c ~ NegativeBinomial2(mu, phi)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test BRM._rk_emit_ast(plan, true).main.args[end] ==
        Expr(:call, :.~, :c,
            Expr(:., :NegativeBinomial2Log, Expr(:tuple, :mu, :phi)))
    @test BRM._rk_emit_ast(plan, false).main.args[end] ==
        Expr(:call, :.~, :c,
            Expr(:., :NegativeBinomial2, Expr(:tuple,
                Expr(:., :exp, Expr(:tuple, :mu)), :phi)))
    brmi = @brm df begin
        log(mu) ~ 1 + x
        alpha ~ Exponential(1)
        z ~ Gamma(alpha, mu / alpha)
    end
    plan = BRM._brm_rk_plan(brmi)
    @test BRM._rk_emit_ast(plan, true).main.args[end] ==
        Expr(:call, :.~, :z,
            Expr(:., :GammaLog, Expr(:tuple, :alpha, :mu)))
    @test BRM._rk_emit_ast(plan, false).main.args[end] ==
        Expr(:call, :.~, :z,
            Expr(:., :Gamma, Expr(:tuple, :alpha,
                Expr(:call, :./, Expr(:., :exp, Expr(:tuple, :mu)),
                    :alpha))))
    brmi = @brm dfp begin
        logit(mu) ~ 1 + x
        kappa ~ Gamma(2.0, 1000.0)
        prop ~ Beta(mu * kappa, (1 - mu) * kappa)
    end
    plan = BRM._brm_rk_plan(brmi)
    mu_log = Expr(:., :logistic, Expr(:tuple, :mu))
    @test BRM._rk_emit_ast(plan, true).main.args[end] ==
        Expr(:call, :.~, :prop,
            Expr(:., :BetaLogit, Expr(:tuple, :mu, :kappa)))
    @test BRM._rk_emit_ast(plan, false).main.args[end] ==
        Expr(:call, :.~, :prop,
            Expr(:., :Beta, Expr(:tuple,
                Expr(:call, :.*, mu_log, :kappa),
                Expr(:call, :.*, Expr(:call, :.-, 1, mu_log), :kappa))))
    # Fused heads ride inside evidence/weights wrappers (the desugar
    # recurses through all four dot-wrappers).
    brmi = @brm df begin
        eta ~ 1 + x
        b ~ weighted(BernoulliLogit(eta), fweights(n))
    end
    plan = BRM._brm_rk_plan(brmi)
    @test BRM._rk_emit_ast(plan, true).main.args[end] ==
        Expr(:call, :.~, :b,
            Expr(:., :weighted, Expr(:tuple,
                Expr(:., :BernoulliLogit, Expr(:tuple, :eta)), :n)))
    # A non-object canonical GLM now takes the object arm.
    brmi = @brm df begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    plan = BRM._brm_rk_plan(brmi)
    fused = BRM._rk_emit_ast(plan, true)
    plain = BRM._rk_emit_ast(plan, false)
    @test fused.main.args[end] == Expr(:call, :~, :y,
        Expr(:call, :NormalIDGLM, :y_X, :mu_alpha, :mu_beta, :sigma))
    @test plain.main.args[end] == Expr(:call, :.~, :y,
        Expr(:., :Normal, Expr(:tuple, :mu, :sigma)))
    brmi = @brm df begin
        probit(p) ~ 1 + x
        b ~ Bernoulli(p)
    end
    plan = BRM._brm_rk_plan(brmi)
    fused = BRM._rk_emit_ast(plan, true)
    plain = BRM._rk_emit_ast(plan, false)
    @test fused.main == plain.main && fused.defs == plain.defs
end
