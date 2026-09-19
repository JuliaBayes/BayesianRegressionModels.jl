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
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_mu, :x), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :~, :b2, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :.+, :b1, Expr(:call, :.*, :b2, :x)))),
        Expr(:(=), Expr(:call, :normal_id_glm, :eta, :sigma), Expr(:block,
            Expr(:call, :.~, :slot,
                Expr(:., :Normal, Expr(:tuple, :eta, :sigma))),
            :slot)),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :~, :mu, Expr(:call, :popefs_mu, :x)),
        Expr(:call, :~, :sigma, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :~, :y, Expr(:call, :normal_id_glm, :mu, :sigma)))
end

@testset "link wrappers and triples" begin
    brmi = @brm df begin
        eta ~ 1 + x
        b ~ BernoulliLogit(eta)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :b,
        Expr(:call, :bernoulli_logit_glm, :eta))
    @test Expr(:(=), Expr(:call, :bernoulli_logit_glm, :eta), Expr(:block,
        Expr(:call, :.~, :slot, Expr(:., :Bernoulli, Expr(:tuple,
            Expr(:., :logistic, Expr(:tuple, :eta))))),
        :slot)) in prog.defs
    # Triple 3 lowers to the T2 shape: the affine value feeds logistic.
    brmi = @brm df begin
        logit(p) ~ 1 + x
        b ~ Bernoulli(p)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :b,
        Expr(:call, :bernoulli_logit_glm, :p))
    @test Expr(:(=), Expr(:call, :bernoulli_logit_glm, :eta), Expr(:block,
        Expr(:call, :.~, :slot, Expr(:., :Bernoulli, Expr(:tuple,
            Expr(:., :logistic, Expr(:tuple, :eta))))),
        :slot)) in prog.defs
    brmi = @brm df begin
        log(mu) ~ 1 + x
        c ~ Poisson(mu)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :c,
        Expr(:call, :poisson_log_glm, :mu))
    @test Expr(:(=), Expr(:call, :poisson_log_glm, :eta), Expr(:block,
        Expr(:call, :.~, :slot, Expr(:., :Poisson, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :eta))))),
        :slot)) in prog.defs
end

@testset "slice-1 response AST shapes" begin
    brmi = @brm df begin
        logit(p) ~ 1 + x
        b ~ Binomial(h, p)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :b,
        Expr(:call, :binomial_logit_glm, :p, :h))
    @test Expr(:(=), Expr(:call, :binomial_logit_glm, :eta, :n),
        Expr(:block,
            Expr(:call, :.~, :slot, Expr(:., :Binomial, Expr(:tuple, :n,
                Expr(:., :logistic, Expr(:tuple, :eta))))),
            :slot)) in prog.defs
    brmi = @brm df begin
        log(mu) ~ 1 + x
        phi ~ Exponential(1)
        c ~ NegativeBinomial2(mu, phi)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :c,
        Expr(:call, :nb2_log_glm, :mu, :phi))
    @test Expr(:(=), Expr(:call, :nb2_log_glm, :eta, :phi),
        Expr(:block,
            Expr(:call, :.~, :slot,
                Expr(:., :NegativeBinomial2, Expr(:tuple,
                    Expr(:., :exp, Expr(:tuple, :eta)), :phi))),
            :slot)) in prog.defs
    brmi = @brm df begin
        log(mu) ~ 1 + x
        alpha ~ Exponential(1)
        z ~ Gamma(alpha, mu / alpha)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :z,
        Expr(:call, :gamma_log_glm, :mu, :alpha))
    @test Expr(:(=), Expr(:call, :gamma_log_glm, :eta, :alpha),
        Expr(:block,
            Expr(:call, :.~, :slot, Expr(:., :Gamma, Expr(:tuple, :alpha,
                Expr(:call, :./, Expr(:., :exp, Expr(:tuple, :eta)),
                    :alpha)))),
            :slot)) in prog.defs
end

@testset "slice-2 group-A AST shapes" begin
    brmi = @brm df begin
        probit(p) ~ 1 + x
        b ~ Bernoulli(p)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :b,
        Expr(:call, :bernoulli_probit_glm, :p))
    @test Expr(:(=), Expr(:call, :bernoulli_probit_glm, :eta), Expr(:block,
        Expr(:call, :.~, :slot, Expr(:., :Bernoulli, Expr(:tuple,
            Expr(:., :probit, Expr(:tuple, :eta))))),
        :slot)) in prog.defs
    brmi = @brm df begin
        cloglog(p) ~ 1 + x
        b ~ Binomial(h, p)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :b,
        Expr(:call, :binomial_cloglog_glm, :p, :h))
    @test Expr(:(=), Expr(:call, :binomial_cloglog_glm, :eta, :n),
        Expr(:block,
            Expr(:call, :.~, :slot, Expr(:., :Binomial, Expr(:tuple, :n,
                Expr(:., :cloglog, Expr(:tuple, :eta))))),
            :slot)) in prog.defs
    brmi = @brm dfp begin
        logit(mu) ~ 1 + x
        kappa ~ Gamma(2.0, 1000.0)
        prop ~ Beta(mu * kappa, (1 - mu) * kappa)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    eta_log = Expr(:., :logistic, Expr(:tuple, :eta))
    @test prog.main.args[end] == Expr(:call, :~, :prop,
        Expr(:call, :beta_logit_glm, :mu, :kappa))
    @test Expr(:(=), Expr(:call, :beta_logit_glm, :eta, :kappa),
        Expr(:block,
            Expr(:call, :.~, :slot, Expr(:., :Beta, Expr(:tuple,
                Expr(:call, :.*, eta_log, :kappa),
                Expr(:call, :.*, Expr(:call, :.-, 1, eta_log), :kappa)))),
            :slot)) in prog.defs
end

@testset "evidence and weights shapes" begin
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ weighted(truncated(Normal(mu, s), 0.0, 2.0), fweights(n))
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :y,
        Expr(:call, :gaussian_truncated_weighted_glm, :mu, :s, :n, 0.0, 2.0))
    @test Expr(:(=),
        Expr(:call, :gaussian_truncated_weighted_glm, :eta, :sigma,
            :weights, :lower, :upper),
        Expr(:block,
            Expr(:call, :.~, :slot, Expr(:., :weighted, Expr(:tuple,
                Expr(:., :truncated, Expr(:tuple,
                    Expr(:., :Normal, Expr(:tuple, :eta, :sigma)),
                    :lower, :upper)), :weights))),
            :slot)) in prog.defs
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ interval_censored(Normal(mu, s); upper=2.0)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :y,
        Expr(:call, :gaussian_interval_censored_glm, :mu, :s, 2.0))
    @test Expr(:(=),
        Expr(:call, :gaussian_interval_censored_glm, :eta, :sigma, :upper),
        Expr(:block,
            Expr(:call, :.~, :slot, Expr(:., :interval_censored, Expr(:tuple,
                Expr(:., :Normal, Expr(:tuple, :eta, :sigma)), :upper))),
            :slot)) in prog.defs
    # Missing sides pass ∓Inf floats (normalized back at bind).
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ truncated(Normal(mu, s), 0, Inf)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :y,
        Expr(:call, :gaussian_truncated_glm, :mu, :s, 0.0, Inf))
    @test Expr(:(=),
        Expr(:call, :gaussian_truncated_glm, :eta, :sigma, :lower, :upper),
        Expr(:block,
            Expr(:call, :.~, :slot, Expr(:., :truncated, Expr(:tuple,
                Expr(:., :Normal, Expr(:tuple, :eta, :sigma)),
                :lower, :upper))),
            :slot)) in prog.defs
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
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog isa BRM._RKEmittedProgram
    @test Expr(:call, :.~,
        Expr(:ref, :mu_b1, Expr(:call, :levels, :g)),
        Expr(:., :Normal, Expr(:tuple, 0.0, 2.0))) in prog.main.args
    affine = only(a for a in prog.main.args if a isa Expr && a.head === :(=))
    @test affine == Expr(:(=), :mu, Expr(:ref, :mu_b1, :g))
    @test all(prog.defs) do d
        d.args[1].args[1] !== :popefs_mu
    end
    # Subsets under an intercept drop the reference position: edge drops
    # spell as literal ranges, middle drops as literal index lists. The
    # factor prior stays top-level; the intercept prior moves into the
    # `popefs_mu` def and the affine becomes its return.
    brmi = @brm df begin
        mu ~ 1 + factor(g; ref=3)
        effect(mu, g) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test Expr(:call, :.~,
        Expr(:ref, :mu_b2, Expr(:ref, Expr(:call, :levels, :g),
            Expr(:call, :(:), 1, 2))),
        Expr(:., :Normal, Expr(:tuple, 0.0, 2.0))) in prog.main.args
    @test Expr(:call, :~, :mu, Expr(:call, :popefs_mu, :g)) in prog.main.args
    @test Expr(:(=), Expr(:call, :popefs_mu, :g), Expr(:block,
        Expr(:call, :~, :b1, Expr(:call, :Normal, 0.0, 1.0)),
        Expr(:call, :.+, :b1, Expr(:ref, :mu_b2, :g)))) in prog.defs
    brmi = @brm df begin
        mu ~ 1 + factor(g; ref=2)
        effect(mu, g) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
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
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
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
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[1] == Expr(:(=), :int_x_x_z,
        Expr(:call, :.*, :x, :z))
    @test Expr(:(=), Expr(:call, :popefs_mu, :int_x_x_z, :x),
        Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :~, :b2, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :~, :b3, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :.+,
                :b1, Expr(:call, :.*, :b2, :x),
                Expr(:call, :.*, :b3, :int_x_x_z)))) in prog.defs
    @test Expr(:call, :~, :mu,
        Expr(:call, :popefs_mu, :int_x_x_z, :x)) in prog.main.args
    # Transforms stage inline reductions in a single definition.
    brmi = @brm df begin
        mu ~ 1 + zscale(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[1] == Expr(:(=), :zscale_x, Expr(:call, :./,
        Expr(:call, :.-, :x, Expr(:call, :mean, :x)),
        Expr(:call, :std, :x)))
    # Mixed interactions compare against raw level values.
    brmi = @brm df begin
        mu ~ 1 + x & g
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
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
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
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
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test Expr(:call, :~, :s, Expr(:call, :HalfNormal, 1.0)) in prog.main.args
    # Coefficient names disambiguate against user names: the user param
    # `mu_b1` occupies the intercept's natural expansion, so the local
    # bumps to `b1_` (expanding to `mu_b1_`).
    brmi = @brm df begin
        mu ~ 1 + x
        mu_b1 ~ Normal(0, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    def = only(d for d in prog.defs if d.args[1].args[1] === :popefs_mu)
    body = def.args[2]
    @test body.args[1] ==
        Expr(:call, :~, :b1_, Expr(:call, :Normal, 0.0, 1.0))
    @test body.args[2] ==
        Expr(:call, :~, :b2, Expr(:call, :Normal, 0.0, 1.0))
    @test body.args[end] == Expr(:call, :.+,
        :b1_, Expr(:call, :.*, :b2, :x))
end

@testset "multi-response shares one affine" begin
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ Normal(mu, s)
        z ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    calls = [a for a in prog.main.args if a isa Expr && a.head === :call &&
        length(a.args) == 3 && a.args[1] === :~ && a.args[3] isa Expr &&
        a.args[3].head === :call && a.args[3].args[1] === :popefs_mu]
    responses = [a for a in prog.main.args if a isa Expr && a.head === :call &&
        length(a.args) == 3 && a.args[1] === :~ && a.args[3] isa Expr &&
        a.args[3].head === :call && a.args[3].args[1] === :normal_id_glm &&
        a.args[2] in (:y, :z)]
    @test length(calls) == 1
    @test length(responses) == 2
    # One shared stream def for both responses.
    @test count(d -> d.args[1].args[1] === :normal_id_glm, prog.defs) == 1
end

@testset "offset-only predictors emit bare affines" begin
    brmi = @brm df begin
        mu ~ 0 + offset(z)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :normal_id_glm, :eta, :sigma), Expr(:block,
            Expr(:call, :.~, :slot,
                Expr(:., :Normal, Expr(:tuple, :eta, :sigma))),
            :slot)),
    ]
    @test prog.main == Expr(:block,
        Expr(:(=), :mu, :z),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :~, :y, Expr(:call, :normal_id_glm, :mu, :s)))
    # Several offsets sum; a derived offset stages its definition first.
    brmi = @brm df begin
        mu ~ 0 + offset(z) + offset(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    affine = only(a for a in prog.main.args if a isa Expr && a.head === :(=) &&
        a.args[1] === :mu)
    @test affine == Expr(:(=), :mu, Expr(:call, :.+, :z, :x))
    brmi = @brm df begin
        mu ~ 0 + offset(log(z))
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
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
            Symbol[], nothing)],
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
        BRM._RKVectorParameter[])
    prog = BRM._rk_emit_ast(plan)
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_n, :n), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :~, :b2, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :.+, :b1, Expr(:call, :.*, :b2, :n)))),
        Expr(:(=), Expr(:call, :normal_id_glm, :eta, :sigma), Expr(:block,
            Expr(:call, :.~, :slot,
                Expr(:., :Normal, Expr(:tuple, :eta, :sigma))),
            :slot)),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :~, :n_, Expr(:call, :popefs_n, :n)),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :~, :y, Expr(:call, :normal_id_glm, :n_, :s)))
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
rk_bucket_stmts(main) =
    [a for a in main.args if a isa Expr && a.head === :do]
# Submodel-def helpers: `rk_def_names` lists def names in order;
# `rk_def_body` fetches one def's body block.
rk_def_names(prog) = [d.args[1].args[1] for d in prog.defs]
rk_def_body(prog, name) =
    only(d.args[2] for d in prog.defs if d.args[1].args[1] === name)

@testset "ranef bucket AST matches surface" begin
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 + x | ID | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    prog = BRM._rk_emit_ast(plan)
    @test length(rk_bucket_stmts(prog.main)) == 1
    @test rk_strip_lines(only(rk_bucket_stmts(prog.main))) ==
        rk_parsed_surface("ranef_bucket(:ID, g; eta = 1.0) do\n mu => [1, x]\nend")
    ret = rk_def_body(prog, :popefs_mu).args[end]
    @test Expr(:call, :ranef, QuoteNode(:ID), :g) in ret.args
end

@testset "ranef eta iff correlated" begin
    kinds = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 + x | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    correlated = only(rk_bucket_stmts(BRM._rk_emit_ast(kinds).main))
    @test rk_strip_lines(correlated) ==
        rk_parsed_surface("ranef_bucket(g; eta = 1.0) do\n mu => [1, x]\nend")
    ones = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    intercept1 = only(rk_bucket_stmts(BRM._rk_emit_ast(ones).main))
    @test rk_strip_lines(intercept1) ==
        rk_parsed_surface("ranef_bucket(g) do\n mu => [1]\nend")
    slopes = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (0 + x | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    slope1 = only(rk_bucket_stmts(BRM._rk_emit_ast(slopes).main))
    @test rk_strip_lines(slope1) ==
        rk_parsed_surface("ranef_bucket(g) do\n mu => [x]\nend")
    ret = rk_def_body(BRM._rk_emit_ast(ones), :popefs_mu).args[end]
    @test Expr(:call, :ranef, :g) in ret.args
end

@testset "ranef dummy values in AST" begin
    codedf = (; df..., c=[2, 4, 2, 6, 4, 6])
    plan = BRM._brm_rk_plan(@brm codedf begin
        mu ~ 1 + x + (1 + c | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test rk_strip_lines(only(rk_bucket_stmts(BRM._rk_emit_ast(plan).main))) ==
        rk_parsed_surface("ranef_bucket(g; eta = 1.0) do\n " *
            "mu => [1, dummy(c, 4), dummy(c, 6)]\nend")
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
    @test rk_strip_lines(only(rk_bucket_stmts(BRM._rk_emit_ast(plan).main))) ==
        rk_parsed_surface("ranef_bucket(:ID, g; eta = 1.0) do\n " *
            "mu1 => [1]; mu2 => [x]\nend")
    prog = BRM._rk_emit_ast(plan)
    for (target, defname) in ((:mu1, :popefs_mu1), (:mu2, :popefs_mu2))
        ret = rk_def_body(prog, defname).args[end]
        @test Expr(:call, :ranef, QuoteNode(:ID), :g) in ret.args
        @test Expr(:call, :~, target,
            Expr(:call, defname, :g, :x)) in prog.main.args
    end
    # Both Gaussian responses share one stream def.
    @test count(==(:normal_id_glm), rk_def_names(prog)) == 1
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
    prog = BRM._rk_emit_ast(plan)
    @test rk_strip_lines(only(rk_bucket_stmts(prog.main))) ==
        rk_parsed_surface("ranef_bucket(:ID, g; eta = 1.0) do\n mu_ => [1, x]\nend")
    @test Expr(:call, :~, :mu_,
        Expr(:call, :popefs_mu, :g, :x)) in prog.main.args
    ret = rk_def_body(prog, :popefs_mu).args[end]
    @test Expr(:call, :ranef, QuoteNode(:ID), :g) in ret.args
end

@testset "leveled AST shapes" begin
    # Reference-coded categorical over K−1 etas: per-response def (the
    # tail arity varies), free tail refs.
    brmi = @brm df begin
        eta1 ~ 1 + x
        eta2 ~ 1 + x
        eta3 ~ 1 + x
        c ~ CategoricalLogit(eta1, eta2, eta3)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :c,
        Expr(:call, :glm_c, :eta1))
    @test Expr(:(=), Expr(:call, :glm_c, :eta), Expr(:block,
        Expr(:call, :.~, :slot, Expr(:., :CategoricalLogit, Expr(:tuple,
            :eta, :eta2, :eta3))),
        :slot)) in prog.defs
    # Ordered-logit: cutpoints implicit (no cutpoint statement).
    brmi = @brm df begin
        eta ~ 1 + x
        c ~ OrderedLogistic(eta)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :c,
        Expr(:call, :ordered_logit_glm, :eta))
    @test Expr(:(=), Expr(:call, :ordered_logit_glm, :eta), Expr(:block,
        Expr(:call, :.~, :slot,
            Expr(:., :OrderedLogistic, Expr(:tuple, :eta))),
        :slot)) in prog.defs
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
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :c,
        Expr(:call, :ordinal_stopping_probit_glm, :eta))
    @test Expr(:(=), Expr(:call, :ordinal_stopping_probit_glm, :eta),
        Expr(:block,
            Expr(:call, :.~, :slot, Expr(:., :Ordinal, Expr(:tuple,
                Expr(:call, :StoppingRatio), Expr(:call, :ProbitLink),
                :eta))),
            :slot)) in prog.defs
    # Ordinal extras fail closed at plan (the AST lowering spells
    # `Ordinal.(structure, link, eta)` only — thin-layer surface gap).
    extras = try
        BRM._brm_rk_plan(@brm df begin
            eta ~ 0 + x
            c ~ Ordinal(StoppingRatio(), LogitLink(), eta;
                discrimination=2.0, per_threshold=(z,))
        end)
        nothing
    catch error
        error
    end
    @test extras isa ErrorException
    @test occursin("surface support", extras.msg)
    # Shared-simplex multinomial + Dirichlet statement.
    brmi = @brm df begin
        s ~ Dirichlet(3, 1.0)
        obs ~ Multinomial(5, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test Expr(:call, :~, :s, Expr(:call, :Dirichlet,
        Expr(:vect, 1.0, 1.0, 1.0))) in prog.main.args
    @test prog.main.args[end] == Expr(:call, :~, :obs,
        Expr(:call, :glm_obs, :s, 5))
    @test Expr(:(=), Expr(:call, :glm_obs, :p, :n), Expr(:block,
        Expr(:call, :.~, :slot, Expr(:., :Multinomial, Expr(:tuple, :n, :p,
            :obs_count_2, :obs_count_3))),
        :slot)) in prog.defs
    # Plain categorical over simplex probs.
    brmi = @brm df begin
        s ~ Dirichlet([2.0, 5.0])
        b ~ Categorical(s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :b,
        Expr(:call, :categorical_glm, :s))
    @test Expr(:(=), Expr(:call, :categorical_glm, :p), Expr(:block,
        Expr(:call, :.~, :slot,
            Expr(:., :Categorical, Expr(:tuple, :p))),
        :slot)) in prog.defs
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
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_mu), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :.+,
                :b1, Expr(:call, :spline, QuoteNode(:s_x))))),
        Expr(:(=), Expr(:call, :normal_id_glm, :eta, :sigma), Expr(:block,
            Expr(:call, :.~, :slot,
                Expr(:., :Normal, Expr(:tuple, :eta, :sigma))),
            :slot)),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :spline_basis,
            Expr(:parameters, Expr(:kw, :k, 10)),
            QuoteNode(:s_x), :x),
        Expr(:call, :~, :mu, Expr(:call, :popefs_mu)),
        Expr(:call, :~, :sigma, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :~, :y, Expr(:call, :normal_id_glm, :mu, :sigma)))
    # The declaration matches the parsed surface spelling exactly.
    @test prog.main.args[1] == Meta.parse("spline_basis(:s_x, x; k = 10)")
    # `t2(x, z)`: tuple-`k` declaration + inline summand.
    brmi = @brm sdf begin
        mu ~ 1 + t2(x, z)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_mu), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :.+,
                :b1, Expr(:call, :spline, QuoteNode(:t2_x_z))))),
        Expr(:(=), Expr(:call, :normal_id_glm, :eta, :sigma), Expr(:block,
            Expr(:call, :.~, :slot,
                Expr(:., :Normal, Expr(:tuple, :eta, :sigma))),
            :slot)),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :spline_basis,
            Expr(:parameters, Expr(:kw, :k, Expr(:tuple, 5, 5))),
            QuoteNode(:t2_x_z), :x, :z),
        Expr(:call, :~, :mu, Expr(:call, :popefs_mu)),
        Expr(:call, :~, :sigma, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :~, :y, Expr(:call, :normal_id_glm, :mu, :sigma)))
    @test prog.main.args[1] ==
        Meta.parse("spline_basis(:t2_x_z, x, z; k = (5, 5))")
end

@testset "monotonic AST shape" begin
    brmi = @brm df begin
        mu ~ 1 + mo(c)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_mu, :c_idx), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :~, :b2, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :.+,
                :b1, Expr(:call, :.*,
                    :b2, Expr(:call, :mo, :c_idx, :mo_c_simplex_incr))))),
        Expr(:(=), Expr(:call, :normal_id_glm, :eta, :sigma), Expr(:block,
            Expr(:call, :.~, :slot,
                Expr(:., :Normal, Expr(:tuple, :eta, :sigma))),
            :slot)),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :~, :mu, Expr(:call, :popefs_mu, :c_idx)),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :~, :mo_c_simplex_incr,
            Expr(:call, :Dirichlet, Expr(:vect, 1.0, 1.0, 1.0))),
        Expr(:call, :~, :y, Expr(:call, :normal_id_glm, :mu, :s)))
    # The summand matches the parsed surface spelling exactly.
    ret = rk_def_body(prog, :popefs_mu).args[end]
    @test rk_strip_lines(ret) ==
        rk_parsed_surface("b1 .+ b2 .* mo(c_idx, mo_c_simplex_incr)")
    # `mo1(c)`: beta-free inline summand, no second coefficient.
    brmi = @brm df begin
        mu ~ 1 + mo1(c)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_mu, :c_idx), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :.+,
                :b1, Expr(:call, :mo1, :c_idx, :mo1_c_simplex_incr)))),
        Expr(:(=), Expr(:call, :normal_id_glm, :eta, :sigma), Expr(:block,
            Expr(:call, :.~, :slot,
                Expr(:., :Normal, Expr(:tuple, :eta, :sigma))),
            :slot)),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :~, :mu, Expr(:call, :popefs_mu, :c_idx)),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :~, :mo1_c_simplex_incr,
            Expr(:call, :Dirichlet, Expr(:vect, 1.0, 1.0, 1.0))),
        Expr(:call, :~, :y, Expr(:call, :normal_id_glm, :mu, :s)))
    # Coefficient-free `mo1` predictor: no scalar terms, so no `popefs`
    # def — the affine stays inline exactly as before.
    brmi = @brm df begin
        mu ~ mo1(c)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :normal_id_glm, :eta, :sigma), Expr(:block,
            Expr(:call, :.~, :slot,
                Expr(:., :Normal, Expr(:tuple, :eta, :sigma))),
            :slot)),
    ]
    @test prog.main == Expr(:block,
        Expr(:(=), :mu,
            Expr(:call, :mo1, :c_idx, :mo1_c_simplex_incr)),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :~, :mo1_c_simplex_incr,
            Expr(:call, :Dirichlet, Expr(:vect, 1.0, 1.0, 1.0))),
        Expr(:call, :~, :y, Expr(:call, :normal_id_glm, :mu, :s)))
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
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast == Expr(:block,
        Expr(:call, :~, :mu_b1, Expr(:call, :Normal, 0.0, 1.0)),
        Expr(:call, :~, :dar_mu_t_beta,
            Expr(:call, :truncated,
                Expr(:call, :Normal, 0.5, 0.2), 0, 1)),
        Expr(:call, :~, :dar_mu_t_sigma,
            Expr(:call, :HalfNormal, 0.2)),
        Expr(:(=), :mu, Expr(:call, :.+,
            :mu_b1, Expr(:call, :dar, :dar_mu_t_beta, :dar_mu_t_sigma))),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :mu, :s))))
    # Both dar spellings match the parsed surface exactly.
    @test rk_strip_lines(ast.args[2]) == rk_parsed_surface(
        "dar_mu_t_beta ~ truncated(Normal(0.5, 0.2), 0, 1)")
    affine = only([a for a in ast.args if a isa Expr && a.head === :(=) &&
        a.args[1] === :mu])
    @test rk_strip_lines(affine.args[2]) ==
        rk_parsed_surface("mu_b1 .+ dar(dar_mu_t_beta, dar_mu_t_sigma)")
    # Prior overrides ride the preamble statements.
    brmi = @brm tdf begin
        mu ~ 1 + dar(t)
        ar(mu, dar(t)) ~ Normal(0.6, 0.1)
        sd(mu, dar(t)) ~ Normal(0, 0.3)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[2] == Expr(:call, :~, :dar_mu_t_beta,
        Expr(:call, :truncated, Expr(:call, :Normal, 0.6, 0.1), 0, 1))
    @test ast.args[3] == Expr(:call, :~, :dar_mu_t_sigma,
        Expr(:call, :HalfNormal, 0.3))
end

@testset "exact gp AST shape" begin
    brmi = @brm df begin
        mu ~ 1 + gp(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    plate = Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
        Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, :y)),
            Expr(:block, Expr(:call, :~,
                Expr(:ref, :z_gp, :i),
                Expr(:call, :Normal, 0.0, 1.0)))))
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_mu), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :.+, :b1, :f_gp))),
        Expr(:(=), Expr(:call, :normal_id_glm, :eta, :sigma), Expr(:block,
            Expr(:call, :.~, :slot,
                Expr(:., :Normal, Expr(:tuple, :eta, :sigma))),
            :slot)),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :~, :rho_gp, Expr(:call, :LogNormal, 0.0, 1.0)),
        Expr(:call, :~, :sigma_gp, Expr(:call, :LogNormal, 0.0, 1.0)),
        plate,
        Expr(:(=), :f_gp, Expr(:call, :gp_chol_latent,
            Expr(:call, :gp_exp_quad_cov, :x, :sigma_gp, :rho_gp, 1e-9),
            :z_gp)),
        Expr(:call, :~, :mu, Expr(:call, :popefs_mu)),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :~, :y, Expr(:call, :normal_id_glm, :mu, :s)))
    # Hyper overrides ride the preamble with their families.
    brmi = @brm df begin
        mu ~ 1 + gp(x)
        length_scale(:, gp(x)) ~ Gamma(2, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
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
    prog = BRM._rk_emit_ast(plan)
    @test Expr(:call, :~, :mu_, Expr(:call, :popefs_mu)) in prog.main.args
    @test Expr(:call, :~, :y,
        Expr(:call, :normal_id_glm, :mu_, :s)) in prog.main.args
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
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_mu), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :.+,
                :b1, Expr(:call, :hsgp, QuoteNode(:hsgp_x))))),
        Expr(:(=), Expr(:call, :normal_id_glm, :eta, :sigma), Expr(:block,
            Expr(:call, :.~, :slot,
                Expr(:., :Normal, Expr(:tuple, :eta, :sigma))),
            :slot)),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :hsgp_basis,
            Expr(:parameters, Expr(:kw, :k, 4), Expr(:kw, :c, 1.5),
                Expr(:kw, :iso, true)),
            QuoteNode(:hsgp_x), :x),
        Expr(:call, :~, :mu, Expr(:call, :popefs_mu)),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :~, :y, Expr(:call, :normal_id_glm, :mu, :s)))
    # The declaration matches the parsed surface spelling exactly.
    @test prog.main.args[1] ==
        Meta.parse("hsgp_basis(:hsgp_x, x; k = 4, c = 1.5, iso = true)")
    # Aniso multi-axis: tuple-`k`/`c` declaration + inline summand.
    brmi = @brm df begin
        mu ~ 1 + hsgp(x, z; k=(4, 3), c=(1.5, 2.0), iso=false)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_mu), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :.+,
                :b1, Expr(:call, :hsgp, QuoteNode(:hsgp_x_z))))),
        Expr(:(=), Expr(:call, :normal_id_glm, :eta, :sigma), Expr(:block,
            Expr(:call, :.~, :slot,
                Expr(:., :Normal, Expr(:tuple, :eta, :sigma))),
            :slot)),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :hsgp_basis,
            Expr(:parameters, Expr(:kw, :k, Expr(:tuple, 4, 3)),
                Expr(:kw, :c, Expr(:tuple, 1.5, 2.0)),
                Expr(:kw, :iso, false)),
            QuoteNode(:hsgp_x_z), :x, :z),
        Expr(:call, :~, :mu, Expr(:call, :popefs_mu)),
        Expr(:call, :~, :s, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :~, :y, Expr(:call, :normal_id_glm, :mu, :s)))
    @test prog.main.args[1] == Meta.parse("hsgp_basis(:hsgp_x_z, x, z; " *
        "k = (4, 3), c = (1.5, 2.0), iso = false)")
end

@testset "distributional scale AST" begin
    brmi = @brm df begin
        mu ~ 1 + x
        log(sigma) ~ 1 + z
        y ~ Normal(mu, sigma)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.defs == Expr[
        Expr(:(=), Expr(:call, :popefs_mu, :x), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :~, :b2, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :.+, :b1, Expr(:call, :.*, :b2, :x)))),
        Expr(:(=), Expr(:call, :popefs_sigma, :z), Expr(:block,
            Expr(:call, :~, :b1, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :~, :b2, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :.+, :b1, Expr(:call, :.*, :b2, :z)))),
        Expr(:(=), Expr(:call, :gaussian_dist_log_glm, :eta, :sigma),
            Expr(:block,
                Expr(:call, :.~, :slot, Expr(:., :Normal, Expr(:tuple,
                    :eta, Expr(:., :exp, Expr(:tuple, :sigma))))),
                :slot)),
    ]
    @test prog.main == Expr(:block,
        Expr(:call, :~, :mu, Expr(:call, :popefs_mu, :x)),
        Expr(:call, :~, :sigma, Expr(:call, :popefs_sigma, :z)),
        Expr(:call, :~, :y,
            Expr(:call, :gaussian_dist_log_glm, :mu, :sigma)))
    # Identity-link scale reads the affine bare.
    brmi = @brm df begin
        mu ~ 1 + x
        sigma ~ 1 + z
        y ~ Normal(mu, sigma)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :y,
        Expr(:call, :gaussian_dist_identity_glm, :mu, :sigma))
    @test Expr(:(=), Expr(:call, :gaussian_dist_identity_glm, :eta, :sigma),
        Expr(:block,
            Expr(:call, :.~, :slot,
                Expr(:., :Normal, Expr(:tuple, :eta, :sigma))),
            :slot)) in prog.defs
    # NB2 dispersion as a predictor.
    brmi = @brm df begin
        log(mu) ~ 1 + x
        log(phi) ~ 1 + z
        c ~ NegativeBinomial2(mu, phi)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :c,
        Expr(:call, :nb2_log_dist_log_glm, :mu, :phi))
    @test Expr(:(=), Expr(:call, :nb2_log_dist_log_glm, :eta, :phi),
        Expr(:block,
            Expr(:call, :.~, :slot,
                Expr(:., :NegativeBinomial2, Expr(:tuple,
                    Expr(:., :exp, Expr(:tuple, :eta)),
                    Expr(:., :exp, Expr(:tuple, :phi))))),
            :slot)) in prog.defs
    # Gamma shape inverts at both use positions.
    brmi = @brm df begin
        log(mu) ~ 1 + x
        log(alpha) ~ 1 + x
        z ~ Gamma(alpha, mu / alpha)
    end
    prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test prog.main.args[end] == Expr(:call, :~, :z,
        Expr(:call, :gamma_log_dist_log_glm, :mu, :alpha))
    @test Expr(:(=), Expr(:call, :gamma_log_dist_log_glm, :eta, :alpha),
        Expr(:block,
            Expr(:call, :.~, :slot, Expr(:., :Gamma, Expr(:tuple,
                Expr(:., :exp, Expr(:tuple, :alpha)),
                Expr(:call, :./,
                    Expr(:., :exp, Expr(:tuple, :eta)),
                    Expr(:., :exp, Expr(:tuple, :alpha)))))),
            :slot)) in prog.defs
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
    ]
    for brmi in models
        prog = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
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
