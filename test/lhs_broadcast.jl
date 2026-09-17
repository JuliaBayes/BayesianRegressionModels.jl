# test/lhs_broadcast.jl — tuple LHS broadcast `(x, y, z) ~ rhs`.
#
# Run: julia --project=test test/lhs_broadcast.jl
#
# `(x, y, z) ~ rhs` is sugar for repeating the `~` row once per name.
# `[y1, y2] ~ ...` stays the joint multivariate response.

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

df = (;
    x=[-1.0, 0.0, 1.0],
    y1=[0.1, 0.2, -0.1],
    y2=[1.1, 0.9, 1.2],
    g=[1, 1, 2],
)

broadcast_outcomes = @brm df begin
    mu ~ 1 + x
    (y1, y2) ~ Normal(mu, 1)
end

repeated_outcomes = @brm df begin
    mu ~ 1 + x
    y1 ~ Normal(mu, 1)
    y2 ~ Normal(mu, 1)
end

@testset "broadcast over outcomes equals repeated rows" begin
    @test sprint(show, broadcast_outcomes) == sprint(show, repeated_outcomes)
    @test length(outcomes(broadcast_outcomes)) == 2
    @test [o.response for o in outcomes(broadcast_outcomes)] == [:y1, :y2]
    @test [o.family for o in outcomes(broadcast_outcomes)] == [Normal, Normal]
    @test outcomes(broadcast_outcomes) == outcomes(repeated_outcomes)
end

@testset "broadcast over linear predictors equals repeated rows" begin
    broadcast_lps = @brm df begin
        (a, b) ~ 1 + x
        y1 ~ Normal(a, 1)
        y2 ~ Normal(b, 1)
    end
    repeated_lps = @brm df begin
        a ~ 1 + x
        b ~ 1 + x
        y1 ~ Normal(a, 1)
        y2 ~ Normal(b, 1)
    end
    @test sprint(show, broadcast_lps) == sprint(show, repeated_lps)
    @test [lp.name for lp in linear_predictors(broadcast_lps)] == [:a, :b]
end

@testset "broadcast keeps LHS decorators and ranef IDs shielded" begin
    broadcast_mi = @brm df begin
        mu ~ 1 + x
        (mi(y1), mi(y2)) ~ Normal(mu, 1)
    end
    repeated_mi = @brm df begin
        mu ~ 1 + x
        mi(y1) ~ Normal(mu, 1)
        mi(y2) ~ Normal(mu, 1)
    end
    @test sprint(show, broadcast_mi) == sprint(show, repeated_mi)

    broadcast_ranef = @brm df begin
        (a, b) ~ 1 + x + (1 | p | g)
        y1 ~ Normal(a, 1)
        y2 ~ Normal(b, 1)
    end
    repeated_ranef = @brm df begin
        a ~ 1 + x + (1 | p | g)
        b ~ 1 + x + (1 | p | g)
        y1 ~ Normal(a, 1)
        y2 ~ Normal(b, 1)
    end
    # The brms `|p|` ID is structural, never a data column.
    @test :p ∉ data_columns(broadcast_ranef)
    @test Set(data_columns(broadcast_ranef)) == Set([:x, :g])
    @test data_columns(broadcast_ranef) == data_columns(repeated_ranef)
    @test sprint(show, broadcast_ranef) == sprint(show, repeated_ranef)
end

@testset "joint vector response still parses as one likelihood" begin
    joint = @brm df begin
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end
    outcome = only(outcomes(joint))
    @test outcome.response == (:y1, :y2)
    @test outcome.family === MvNormalCholesky
end

@testset "broadcast lowering matches repeated rows" begin
    stan_broadcast = BayesianRegressionModels.stan_code(
        SBBRMI(broadcast_outcomes; mod=@__MODULE__))
    stan_repeated = BayesianRegressionModels.stan_code(
        SBBRMI(repeated_outcomes; mod=@__MODULE__))
    @test stan_broadcast == stan_repeated
    checked = StanBlocks.stanc_check(stan_broadcast; warn_pedantic=false)
    checked.ok || @error "stanc rejected broadcast model" output=checked.output
    @test checked.ok
end

@testset "broadcast LHS rejects non-sampling shapes" begin
    @test_throws "needs at least one name" eval(quote
        @brm df begin
            () ~ Normal(0, 1)
        end
    end)
    @test_throws "not prior addresses" eval(quote
        @brm df begin
            (effect(mu, x), y1) ~ Normal(0, 1)
        end
    end)
    @test_throws "nested collections" eval(quote
        @brm df begin
            ([y1, y2], x) ~ Normal(0, 1)
        end
    end)
end
