# test/rk_ast.jl — BRM-side `@rkppl` AST emission (phase 2 U3 retarget).
#
# Run: julia --project=test test/rk_ast.jl
#
# Pure-Julia checks (no ReactiveKernels dependency): exact `Expr` shapes
# over the whole slice-1 surface (`_rk_emit_ast` is total — the AST is
# the sole emission path, no fallback). Lowerability through the real
# `lower_rkppl` is covered per-slice by the parity corpus
# (test/rk_parity.jl carries the ranef slice), which routes each case
# through the retargeted factory.

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

@testset "slice-1 response AST shapes" begin
    brmi = @brm df begin
        logit(p) ~ 1 + x
        b ~ Binomial(h, p)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :.~,
        :b, Expr(:., :Binomial, Expr(:tuple, :h,
            Expr(:., :logistic, Expr(:tuple, :p)))))
    brmi = @brm df begin
        log(mu) ~ 1 + x
        phi ~ Exponential(1)
        c ~ NegativeBinomial2(mu, phi)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :.~,
        :c, Expr(:., :NegativeBinomial2, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :mu)), :phi)))
    brmi = @brm df begin
        log(mu) ~ 1 + x
        alpha ~ Exponential(1)
        z ~ Gamma(alpha, mu / alpha)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :.~,
        :z, Expr(:., :Gamma, Expr(:tuple, :alpha,
            Expr(:call, :./, Expr(:., :exp, Expr(:tuple, :mu)), :alpha))))
end

@testset "slice-2 group-A AST shapes" begin
    brmi = @brm df begin
        probit(p) ~ 1 + x
        b ~ Bernoulli(p)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :.~,
        :b, Expr(:., :Bernoulli, Expr(:tuple,
            Expr(:., :probit, Expr(:tuple, :p)))))
    brmi = @brm df begin
        cloglog(p) ~ 1 + x
        b ~ Binomial(h, p)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :.~,
        :b, Expr(:., :Binomial, Expr(:tuple, :h,
            Expr(:., :cloglog, Expr(:tuple, :p)))))
    brmi = @brm dfp begin
        logit(mu) ~ 1 + x
        kappa ~ Gamma(2.0, 1000.0)
        prop ~ Beta(mu * kappa, (1 - mu) * kappa)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    mu_log = Expr(:., :logistic, Expr(:tuple, :mu))
    @test ast.args[end] == Expr(:call, :.~,
        :prop, Expr(:., :Beta, Expr(:tuple,
            Expr(:call, :.*, mu_log, :kappa),
            Expr(:call, :.*, Expr(:call, :.-, 1, mu_log), :kappa))))
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

@testset "offset-only predictors emit bare affines" begin
    brmi = @brm df begin
        mu ~ 0 + offset(z)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast == Expr(:block,
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
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    affine = only(a for a in ast.args if a isa Expr && a.head === :(=) &&
        a.args[1] === :mu)
    @test affine == Expr(:(=), :mu, Expr(:call, :.+, :z, :x))
    brmi = @brm df begin
        mu ~ 0 + offset(log(z))
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[1] == Expr(:(=), :rkd_offset_log_z,
        Expr(:., :log, Expr(:tuple, :z)))
    @test ast.args[2] == Expr(:(=), :mu, :rkd_offset_log_z)
end

@testset "predictor/data overlap alpha-renames" begin
    # Unreachable via `@brm` (observation discovery claims `n ~ …` as a
    # likelihood), so the plan is built by hand; the affine (definition
    # and response uses) moves to `n_` while the overlapping data column
    # keeps `n` — definition and data reference coexist.
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
    ast = BRM._rk_emit_ast(plan)
    @test ast == Expr(:block,
        Expr(:call, :~, :n_b1, Expr(:call, :Normal, 0.0, 1.0)),
        Expr(:call, :~, :n_b2, Expr(:call, :Normal, 0.0, 1.0)),
        Expr(:(=), :n_, Expr(:call, :.+,
            :n_b1, Expr(:call, :.*, :n_b2, :n))),
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
rk_bucket_stmts(ast) =
    [a for a in ast.args if a isa Expr && a.head === :do]

@testset "ranef bucket AST matches surface" begin
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 + x | ID | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    ast = BRM._rk_emit_ast(plan)
    @test length(rk_bucket_stmts(ast)) == 1
    @test rk_strip_lines(only(rk_bucket_stmts(ast))) ==
        rk_parsed_surface("ranef_bucket(:ID, g; eta = 1.0) do\n mu => [1, x]\nend")
    affine = only([a for a in ast.args if a isa Expr && a.head === :(=) &&
        a.args[1] === :mu])
    @test Expr(:call, :ranef, QuoteNode(:ID), :g) in affine.args[2].args
end

@testset "ranef eta iff correlated" begin
    kinds = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 + x | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    correlated = only(rk_bucket_stmts(BRM._rk_emit_ast(kinds)))
    @test rk_strip_lines(correlated) ==
        rk_parsed_surface("ranef_bucket(g; eta = 1.0) do\n mu => [1, x]\nend")
    ones = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    intercept1 = only(rk_bucket_stmts(BRM._rk_emit_ast(ones)))
    @test rk_strip_lines(intercept1) ==
        rk_parsed_surface("ranef_bucket(g) do\n mu => [1]\nend")
    slopes = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (0 + x | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    slope1 = only(rk_bucket_stmts(BRM._rk_emit_ast(slopes)))
    @test rk_strip_lines(slope1) ==
        rk_parsed_surface("ranef_bucket(g) do\n mu => [x]\nend")
    affine = only([a for a in BRM._rk_emit_ast(ones).args if a isa Expr &&
        a.head === :(=) && a.args[1] === :mu])
    @test Expr(:call, :ranef, :g) in affine.args[2].args
end

@testset "ranef dummy values in AST" begin
    codedf = (; df..., c=[2, 4, 2, 6, 4, 6])
    plan = BRM._brm_rk_plan(@brm codedf begin
        mu ~ 1 + x + (1 + c | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    @test rk_strip_lines(only(rk_bucket_stmts(BRM._rk_emit_ast(plan)))) ==
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
    @test rk_strip_lines(only(rk_bucket_stmts(BRM._rk_emit_ast(plan)))) ==
        rk_parsed_surface("ranef_bucket(:ID, g; eta = 1.0) do\n " *
            "mu1 => [1]; mu2 => [x]\nend")
    for target in (:mu1, :mu2)
        affine = only([a for a in BRM._rk_emit_ast(plan).args if a isa Expr &&
            a.head === :(=) && a.args[1] === target])
        @test Expr(:call, :ranef, QuoteNode(:ID), :g) in affine.args[2].args
    end
end

@testset "ranef bucket follows predictor rename" begin
    # Programmatic overlap (unreachable via @brm): the margin lines use the
    # renamed predictor, matching the renamed affine.
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + x + (1 + x | ID | g)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    plan.columns[:mu] = plan.columns[:x]
    ast = BRM._rk_emit_ast(plan)
    @test rk_strip_lines(only(rk_bucket_stmts(ast))) ==
        rk_parsed_surface("ranef_bucket(:ID, g; eta = 1.0) do\n mu_ => [1, x]\nend")
    affine = only([a for a in ast.args if a isa Expr && a.head === :(=) &&
        a.args[1] === :mu_])
    @test Expr(:call, :ranef, QuoteNode(:ID), :g) in affine.args[2].args
end

@testset "leveled AST shapes" begin
    # Reference-coded categorical over K−1 etas.
    brmi = @brm df begin
        eta1 ~ 1 + x
        eta2 ~ 1 + x
        eta3 ~ 1 + x
        c ~ CategoricalLogit(eta1, eta2, eta3)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :.~,
        :c, Expr(:., :CategoricalLogit, Expr(:tuple, :eta1, :eta2, :eta3)))
    # Ordered-logit: cutpoints implicit (no cutpoint statement).
    brmi = @brm df begin
        eta ~ 1 + x
        c ~ OrderedLogistic(eta)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :.~,
        :c, Expr(:., :OrderedLogistic, Expr(:tuple, :eta)))
    @test all(ast.args) do stmt
        !(stmt isa Expr && stmt.head === :call && length(stmt.args) >= 2 &&
            stmt.args[2] === :c_cutpoints)
    end
    # Plain typed ordinal with tag calls; thresholds implicit.
    brmi = @brm df begin
        eta ~ 0 + x
        c ~ Ordinal(StoppingRatio(), ProbitLink(), eta)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :.~,
        :c, Expr(:., :Ordinal, Expr(:tuple, Expr(:call, :StoppingRatio),
            Expr(:call, :ProbitLink), :eta)))
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
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    stmts = ast.args
    @test Expr(:call, :~, :s, Expr(:call, :Dirichlet,
        Expr(:vect, 1.0, 1.0, 1.0))) in stmts
    @test stmts[end] == Expr(:call, :.~,
        :obs, Expr(:., :Multinomial, Expr(:tuple, 5, :s,
            :obs_count_2, :obs_count_3)))
    # Plain categorical over simplex probs.
    brmi = @brm df begin
        s ~ Dirichlet([2.0, 5.0])
        b ~ Categorical(s)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :.~,
        :b, Expr(:., :Categorical, Expr(:tuple, :s)))
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
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast == Expr(:block,
        Expr(:call, :~, :mu_b1, Expr(:call, :Normal, 0.0, 1.0)),
        Expr(:call, :spline_basis,
            Expr(:parameters, Expr(:kw, :k, 10)),
            QuoteNode(:s_x), :x),
        Expr(:(=), :mu, Expr(:call, :.+,
            :mu_b1, Expr(:call, :spline, QuoteNode(:s_x)))),
        Expr(:call, :~, :sigma, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :mu, :sigma))))
    # The declaration matches the parsed surface spelling exactly.
    @test ast.args[2] == Meta.parse("spline_basis(:s_x, x; k = 10)")
    # `t2(x, z)`: tuple-`k` declaration + inline summand.
    brmi = @brm sdf begin
        mu ~ 1 + t2(x, z)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast == Expr(:block,
        Expr(:call, :~, :mu_b1, Expr(:call, :Normal, 0.0, 1.0)),
        Expr(:call, :spline_basis,
            Expr(:parameters, Expr(:kw, :k, Expr(:tuple, 5, 5))),
            QuoteNode(:t2_x_z), :x, :z),
        Expr(:(=), :mu, Expr(:call, :.+,
            :mu_b1, Expr(:call, :spline, QuoteNode(:t2_x_z)))),
        Expr(:call, :~, :sigma, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :mu, :sigma))))
    @test ast.args[2] ==
        Meta.parse("spline_basis(:t2_x_z, x, z; k = (5, 5))")
end

@testset "exact gp AST shape" begin
    brmi = @brm df begin
        mu ~ 1 + gp(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    plate = Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
        Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, :y)),
            Expr(:block, Expr(:call, :~,
                Expr(:ref, :z_gp, :i),
                Expr(:call, :Normal, 0.0, 1.0)))))
    @test ast == Expr(:block,
        Expr(:call, :~, :mu_b1, Expr(:call, :Normal, 0.0, 1.0)),
        Expr(:call, :~, :rho_gp, Expr(:call, :LogNormal, 0.0, 1.0)),
        Expr(:call, :~, :sigma_gp, Expr(:call, :LogNormal, 0.0, 1.0)),
        plate,
        Expr(:(=), :f_gp, Expr(:call, :gp_chol_latent,
            Expr(:call, :gp_exp_quad_cov, :x, :sigma_gp, :rho_gp, 1e-9),
            :z_gp)),
        Expr(:(=), :mu, Expr(:call, :.+, :mu_b1, :f_gp)),
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
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test Expr(:call, :~, :rho_gp, Expr(:call, :Gamma, 2.0, 1.0)) in ast.args
    # Overlap alpha-renames the affine; the GP preamble is unaffected.
    # (Overlap is unconstructible from formulas — a data-named LHS
    # classifies as an observation — so force it by plan surgery.)
    plan = BRM._brm_rk_plan(@brm df begin
        mu ~ 1 + gp(x)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end)
    plan.columns[:mu] = plan.columns[:y]
    ast = BRM._rk_emit_ast(plan)
    @test Expr(:(=), :mu_,
        Expr(:call, :.+, :mu_b1, :f_gp)) in ast.args
    @test Expr(:call, :.~, :y,
        Expr(:., :Normal, Expr(:tuple, :mu_, :s))) in ast.args
    @test Expr(:(=), :f_gp, Expr(:call, :gp_chol_latent,
        Expr(:call, :gp_exp_quad_cov, :x, :sigma_gp, :rho_gp, 1e-9),
        :z_gp)) in ast.args
end

@testset "distributional scale AST" begin
    brmi = @brm df begin
        mu ~ 1 + x
        log(sigma) ~ 1 + z
        y ~ Normal(mu, sigma)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast == Expr(:block,
        Expr(:call, :~, :mu_b1, Expr(:call, :Normal, 0.0, 1.0)),
        Expr(:call, :~, :mu_b2, Expr(:call, :Normal, 0.0, 1.0)),
        Expr(:(=), :mu, Expr(:call, :.+,
            :mu_b1, Expr(:call, :.*, :mu_b2, :x))),
        Expr(:call, :~, :sigma_b1, Expr(:call, :Normal, 0.0, 1.0)),
        Expr(:call, :~, :sigma_b2, Expr(:call, :Normal, 0.0, 1.0)),
        Expr(:(=), :sigma, Expr(:call, :.+,
            :sigma_b1, Expr(:call, :.*, :sigma_b2, :z))),
        Expr(:call, :.~, :y,
            Expr(:., :Normal, Expr(:tuple, :mu,
                Expr(:., :exp, Expr(:tuple, :sigma))))))
    # Identity-link scale reads the affine bare.
    brmi = @brm df begin
        mu ~ 1 + x
        sigma ~ 1 + z
        y ~ Normal(mu, sigma)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :.~, :y,
        Expr(:., :Normal, Expr(:tuple, :mu, :sigma)))
    # NB2 dispersion as a predictor.
    brmi = @brm df begin
        log(mu) ~ 1 + x
        log(phi) ~ 1 + z
        c ~ NegativeBinomial2(mu, phi)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :.~, :c,
        Expr(:., :NegativeBinomial2, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :mu)),
            Expr(:., :exp, Expr(:tuple, :phi)))))
    # Gamma shape inverts at both use positions.
    brmi = @brm df begin
        log(mu) ~ 1 + x
        log(alpha) ~ 1 + x
        z ~ Gamma(alpha, mu / alpha)
    end
    ast = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test ast.args[end] == Expr(:call, :.~, :z,
        Expr(:., :Gamma, Expr(:tuple,
            Expr(:., :exp, Expr(:tuple, :alpha)),
            Expr(:call, :./,
                Expr(:., :exp, Expr(:tuple, :mu)),
                Expr(:., :exp, Expr(:tuple, :alpha))))))
end
