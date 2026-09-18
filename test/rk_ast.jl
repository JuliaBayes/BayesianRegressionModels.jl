# test/rk_ast.jl — BRM-side `@rkppl` AST emission (phase 2 U3 retarget).
#
# Run: julia --project=test test/rk_ast.jl
#
# Pure-Julia checks (no ReactiveKernels dependency): exact `Expr` shapes
# for the expressible subset, `nothing` for the inexpressible subset.
# Lowerability through the real `lower_rkppl` is covered by the scratch
# parity corpus, which routes every case through the retargeted factory.

using Test
using BayesianRegressionModels
using Distributions: Bernoulli, Exponential, Gamma, Normal, Poisson, truncated
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
)

@testset "gaussian AST exact shape" begin
    brmi = @brm df begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast isa Expr && ast.head === :block
    @test ast == Expr(:block,
        Expr(:call, :~, :mu_b1, Expr(:call, :Normal, 0.0, 1.0)),
        Expr(:call, :~, :mu_b2, Expr(:call, :Normal, 0.0, 1.0)),
        Expr(:(=), :mu, Expr(:call, :.+,
            :mu_b1, Expr(:call, :.*, :mu_b2, :x))),
        Expr(:call, :~, :sigma, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :mu, :sigma))))
end

@testset "link wrappers and triples" begin
    brmi = @brm df begin
        eta ~ 1 + x
        b ~ BernoulliLogit(eta)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :.~,
        :b, Expr(:., :Bernoulli, Expr(:tuple,
            Expr(:., :logistic, Expr(:tuple, :eta)))))
    # Triple 3 lowers to the T2 shape: the affine value feeds logistic.
    brmi = @brm df begin
        logit(p) ~ 1 + x
        b ~ Bernoulli(p)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :.~,
        :b, Expr(:., :Bernoulli, Expr(:tuple,
            Expr(:., :logistic, Expr(:tuple, :p)))))
    brmi = @brm df begin
        log(mu) ~ 1 + x
        c ~ Poisson(mu)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :.~,
        :c, Expr(:., :Poisson, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :mu)))))
end

@testset "evidence and weights shapes" begin
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ weighted(truncated(Normal(mu, s), 0.0, 2.0), fweights(n))
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :.~,
        :y, Expr(:., :weighted, Expr(:tuple,
            Expr(:., :truncated, Expr(:tuple,
                Expr(:., :Normal, Expr(:tuple, :mu, :s)), 0.0, 2.0)), :n)))
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ interval_censored(Normal(mu, s); upper=2.0)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :.~,
        :y, Expr(:., :interval_censored, Expr(:tuple,
            Expr(:., :Normal, Expr(:tuple, :mu, :s)), 2.0)))
    # Missing sides emit as ∓Inf floats (normalized back at bind).
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ truncated(Normal(mu, s), 0, Inf)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :.~,
        :y, Expr(:., :truncated, Expr(:tuple,
            Expr(:., :Normal, Expr(:tuple, :mu, :s)), 0.0, Inf)))
end

@testset "factors and ref gating" begin
    brmi = @brm df begin
        mu ~ 0 + g
        effect(mu, g) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast isa Expr
    @test Expr(:call, :.~,
        Expr(:ref, :mu_b1, Expr(:call, :levels, :g)),
        Expr(:., :Normal, Expr(:tuple, 0.0, 2.0))) in ast.args
    affine = only(a for a in ast.args if a isa Expr && a.head === :(=))
    @test affine == Expr(:(=), :mu, Expr(:ref, :mu_b1, :g))
    # Subsets under an intercept drop the reference position: edge drops
    # spell as literal ranges, middle drops as literal index lists.
    brmi = @brm df begin
        mu ~ 1 + factor(g; ref=3)
        effect(mu, g) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test Expr(:call, :.~,
        Expr(:ref, :mu_b2, Expr(:ref, Expr(:call, :levels, :g),
            Expr(:call, :(:), 1, 2))),
        Expr(:., :Normal, Expr(:tuple, 0.0, 2.0))) in ast.args
    affine = only(a for a in ast.args if a isa Expr && a.head === :(=))
    @test affine == Expr(:(=), :mu, Expr(:call, :.+,
        :mu_b1, Expr(:ref, :mu_b2, :g)))
    brmi = @brm df begin
        mu ~ 1 + factor(g; ref=2)
        effect(mu, g) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test Expr(:call, :.~,
        Expr(:ref, :mu_b2, Expr(:ref, Expr(:call, :levels, :g),
            Expr(:vect, 1, 3))),
        Expr(:., :Normal, Expr(:tuple, 0.0, 2.0))) in ast.args
    # String refs lower to the same sort-order drop position.
    brmi = @brm df begin
        mu ~ 1 + factor(gs; ref="c")
        effect(mu, gs) ~ Normal(0, 2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test Expr(:call, :.~,
        Expr(:ref, :mu_b2, Expr(:ref, Expr(:call, :levels, :gs),
            Expr(:call, :(:), 1, 2))),
        Expr(:., :Normal, Expr(:tuple, 0.0, 2.0))) in ast.args
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
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[1] == Expr(:(=), :int_x_x_z,
        Expr(:call, :.*, :x, :z))
    affine = only(a for a in ast.args if a isa Expr && a.head === :(=) &&
        a.args[1] === :mu)
    @test affine == Expr(:(=), :mu, Expr(:call, :.+,
        :mu_b1, Expr(:call, :.*, :mu_b2, :x),
        Expr(:call, :.*, :mu_b3, :int_x_x_z)))
    priors = [a for a in ast.args if a isa Expr && a.head === :call &&
        length(a.args) == 3 && a.args[1] === :~ &&
        a.args[2] in (:mu_b1, :mu_b2, :mu_b3)]
    @test length(priors) == 3
    # Transforms stage inline reductions in a single definition.
    brmi = @brm df begin
        mu ~ 1 + zscale(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[1] == Expr(:(=), :zscale_x, Expr(:call, :./,
        Expr(:call, :.-, :x, Expr(:call, :mean, :x)),
        Expr(:call, :std, :x)))
    # Mixed interactions compare against raw level values.
    brmi = @brm df begin
        mu ~ 1 + x & g
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[1] == Expr(:(=), :int_x_x_g_lvl_1,
        Expr(:call, :.*, :x, Expr(:call, :.==, :g, 1)))
    @test ast.args[2] == Expr(:(=), :int_x_x_g_lvl_2,
        Expr(:call, :.*, :x, Expr(:call, :.==, :g, 2)))
    @test ast.args[3] == Expr(:(=), :int_x_x_g_lvl_3,
        Expr(:call, :.*, :x, Expr(:call, :.==, :g, 3)))
end

@testset "sampled, assignments, collisions" begin
    brmi = @brm df begin
        mu ~ 1 + x
        m = sum(x)
        s ~ Gamma(m, 2.0)
        y ~ Normal(mu, s)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    assign = only(a for a in ast.args if a isa Expr && a.head === :(=) &&
        a.args[1] === :m)
    @test assign == Expr(:(=), :m, Expr(:call, :sum, :x))
    @test Expr(:call, :~, :s, Expr(:call, :Gamma, :m, 2.0)) in ast.args
    # HalfNormal + Flat mappings.
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ truncated(Normal(0, 1), 0, Inf)
        y ~ Normal(mu, s)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test Expr(:call, :~, :s, Expr(:call, :HalfNormal, 1.0)) in ast.args
    # Coefficient names disambiguate against user names.
    brmi = @brm df begin
        mu ~ 1 + x
        mu_b1 ~ Normal(0, 1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    coefs = Set{Symbol}()
    for stmt in ast.args
        stmt isa Expr && stmt.head === :call && length(stmt.args) == 3 &&
            stmt.args[1] === :~ && stmt.args[3] isa Expr &&
            stmt.args[3].args[1] === :Normal &&
            stmt.args[2] !== :mu_b1 && push!(coefs, stmt.args[2])
    end
    @test :mu_b1_ in coefs
end

@testset "multi-response shares one affine" begin
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ Normal(mu, s)
        z ~ Normal(mu, s)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    affines = [a for a in ast.args if a isa Expr && a.head === :(=) &&
        a.args[1] === :mu]
    responses = [a for a in ast.args if a isa Expr && a.head === :call &&
        length(a.args) == 3 && a.args[1] === :.~ &&
        a.args[2] in (:y, :z)]
    @test length(affines) == 1
    @test length(responses) == 2
end
