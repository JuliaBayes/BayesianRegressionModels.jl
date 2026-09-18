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
        Expr(:(=), :mu, Expr(:call, :+,
            :mu_b1, Expr(:call, :*, :mu_b2, :x))),
        Expr(:call, :~, :sigma, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :~, :y, Expr(:call, :Normal, :mu, :sigma)))
end

@testset "link wrappers and triples" begin
    brmi = @brm df begin
        eta ~ 1 + x
        b ~ BernoulliLogit(eta)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :~,
        :b, Expr(:call, :Bernoulli, Expr(:call, :logistic, :eta)))
    # Triple 3 lowers to the T2 shape: the affine value feeds logistic.
    brmi = @brm df begin
        logit(p) ~ 1 + x
        b ~ Bernoulli(p)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :~,
        :b, Expr(:call, :Bernoulli, Expr(:call, :logistic, :p)))
    brmi = @brm df begin
        log(mu) ~ 1 + x
        c ~ Poisson(mu)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :~,
        :c, Expr(:call, :Poisson, Expr(:call, :exp, :mu)))
end

@testset "evidence and weights shapes" begin
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ weighted(truncated(Normal(mu, s), 0.0, 2.0), fweights(n))
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :~,
        :y, Expr(:call, :weighted,
            Expr(:call, :truncated,
                Expr(:call, :Normal, :mu, :s), 0.0, 2.0), :n))
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ interval_censored(Normal(mu, s); upper=2.0)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :~,
        :y, Expr(:call, :interval_censored,
            Expr(:call, :Normal, :mu, :s), 2.0))
    # Missing sides emit as ∓Inf floats (normalized back at bind).
    brmi = @brm df begin
        mu ~ 1 + x
        s ~ Exponential(1)
        y ~ truncated(Normal(mu, s), 0, Inf)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :~,
        :y, Expr(:call, :truncated,
            Expr(:call, :Normal, :mu, :s), 0.0, Inf))
end

@testset "factors and ref gating" begin
    brmi = @brm df begin
        mu ~ 1 + factor(g; ref=1)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast isa Expr
    affine = only(a for a in ast.args if a isa Expr && a.head === :(=))
    @test affine == Expr(:(=), :mu, Expr(:call, :+,
        :mu_b1, Expr(:ref, :mu_b2, :g)))
    priors = [a for a in ast.args if a isa Expr && a.head === :call &&
        length(a.args) == 3 && a.args[1] === :~ &&
        a.args[2] in (:mu_b1, :mu_b2)]
    @test length(priors) == 2
    # Non-1 refs pin via treatment(g, ref).
    brmi = @brm df begin
        mu ~ 1 + factor(g; ref=3)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    affine = only(a for a in ast.args if a isa Expr && a.head === :(=))
    @test affine == Expr(:(=), :mu, Expr(:call, :+,
        :mu_b1, Expr(:ref, :mu_b2, Expr(:call, :treatment, :g, 3))))
    # String refs lower to the same sort-order treatment index.
    brmi = @brm df begin
        mu ~ 1 + factor(gs; ref="c")
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    affine = only(a for a in ast.args if a isa Expr && a.head === :(=))
    @test affine == Expr(:(=), :mu, Expr(:call, :+,
        :mu_b1, Expr(:ref, :mu_b2, Expr(:call, :treatment, :gs, 3))))
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
        length(a.args) == 3 && a.args[1] === :~ &&
        a.args[2] in (:y, :z)]
    @test length(affines) == 1
    @test length(responses) == 2
end
