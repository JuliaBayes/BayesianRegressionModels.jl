# test/horseshoe_priors.jl — configurable scalar Horseshoe scale contract.
#
# Run:
#   julia --project=test test/horseshoe_priors.jl

using Test
using BayesianRegressionModels
using Distributions: Normal
using StanBlocks

const BRM = BayesianRegressionModels
const HS_DF = (; y=[-0.4, 0.1, 0.7])

default_builder = @brm begin
    beta ~ Horseshoe()
    y ~ Normal(beta, 1)
end

scaled_builder = @brm begin
    beta ~ Horseshoe(local_scale=1 / 4, global_scale=0.1)
    y ~ Normal(beta, 1)
end

@testset "scalar Horseshoe scale keywords" begin
    default = SBBRMI(default_builder(HS_DF); mod=@__MODULE__)
    scaled = SBBRMI(scaled_builder(HS_DF); mod=@__MODULE__)
    default_code = BRM.stan_code(default)
    scaled_code = BRM.stan_code(scaled)

    # No keywords keep the historical sibling and unit half-Cauchy scales.
    default_decl = only(d for d in generative_plan(default).declarations
                        if d.target === :beta)
    @test default_decl.family === :_sb_horseshoe
    @test occursin("beta_lambda ~ cauchy(0.0, 1.0);", default_code)
    @test occursin("beta_tau ~ cauchy(0.0, 1.0);", default_code)
    @test occursin("beta_raw ~ std_normal();", default_code)

    # Configured calls take the sibling seam. Literal arithmetic is evaluated
    # with Julia semantics before becoming a Stan literal.
    scaled_decl = only(d for d in generative_plan(scaled).declarations
                       if d.target === :beta)
    @test scaled_decl.family === :_sb_horseshoe_scaled
    @test scaled_decl.keywords.local_scale == 0.25
    @test scaled_decl.keywords.global_scale == 0.1
    @test occursin("beta_lambda ~ cauchy(0.0, 0.25);", scaled_code)
    @test occursin("beta_tau ~ cauchy(0.0, 0.1);", scaled_code)
    @test occursin("beta_raw ~ std_normal();", scaled_code)

    @test StanBlocks.stan.transpiles(default.model)
    @test StanBlocks.stan.transpiles(scaled.model)
    @test StanBlocks.stanc_check(default_code; warn_pedantic=false).ok
    @test StanBlocks.stanc_check(scaled_code; warn_pedantic=false).ok
end

@testset "one scale defaults and invalid configurations refuse loudly" begin
    local_only = @brm HS_DF begin
        beta ~ Horseshoe(local_scale=0.3)
        y ~ Normal(beta, 1)
    end
    local_code = BRM.stan_code(SBBRMI(local_only; mod=@__MODULE__))
    @test occursin("beta_lambda ~ cauchy(0.0, 0.3);", local_code)
    @test occursin("beta_tau ~ cauchy(0.0, 1.0);", local_code)

    positional = @brm HS_DF begin
        beta ~ Horseshoe(0.5, 0.1)
        y ~ Normal(beta, 1)
    end
    @test_throws "accepts no positional arguments" SBBRMI(
        positional; mod=@__MODULE__)

    unknown = @brm HS_DF begin
        beta ~ Horseshoe(scale=0.5)
        y ~ Normal(beta, 1)
    end
    @test_throws "accepts only `local_scale` and `global_scale`" SBBRMI(
        unknown; mod=@__MODULE__)

    zero = @brm HS_DF begin
        beta ~ Horseshoe(local_scale=0.0)
        y ~ Normal(beta, 1)
    end
    @test_throws "finite and strictly positive" SBBRMI(zero; mod=@__MODULE__)

    negative = @brm HS_DF begin
        beta ~ Horseshoe(global_scale=-0.1)
        y ~ Normal(beta, 1)
    end
    @test_throws "finite and strictly positive" SBBRMI(
        negative; mod=@__MODULE__)

    nonfinite = @brm HS_DF begin
        beta ~ Horseshoe(global_scale=1.0 / 0.0)
        y ~ Normal(beta, 1)
    end
    @test_throws "finite and strictly positive" SBBRMI(
        nonfinite; mod=@__MODULE__)

    boolean = @brm HS_DF begin
        beta ~ Horseshoe(local_scale=true)
        y ~ Normal(beta, 1)
    end
    @test_throws "numeric formula constant" SBBRMI(boolean; mod=@__MODULE__)
end

const HS_COLS = (;
    y=[0.5, -0.3, 0.8, 0.1, -0.2],
    x=[1.0, 2.0, 3.0, 4.0, 5.0],
    z=[0.5, 1.5, 2.5, 3.5, 4.5],
    g=[1, 1, 2, 2, 1],
)

@testset "structured Horseshoe over population coefficients" begin
    brmi = @brm HS_COLS begin
        mu ~ 1 + x + z
        effect(mu, x) ~ Horseshoe()
        effect(mu, z) ~ Normal(0, 3)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BRM.stan_code(sb)
    # One bare-form triple for the Horseshoe column (column 2 of beta_pop),
    # scalar Normal statements for the intercept and the sibling column, and
    # a transformed beta_pop assembled in column order.
    @test occursin("hs_2_x_raw ~ std_normal();", code)
    @test occursin("hs_2_x_lambda ~ cauchy(0.0, 1.0);", code)
    @test occursin("hs_2_x_tau ~ cauchy(0.0, 1.0);", code)
    @test occursin("b_1_Intercept ~ normal(0.0, 1.0);", code)
    # Sibling literals keep their formula types (ints stay ints), exactly as
    # the generic vector-prior path emits them.
    @test occursin("b_3_z ~ normal(0, 3);", code)
    # SLIC prefixes submodel locals with the use-site target.
    @test occursin(
        "beta_pop = [pop_mu_b_1_Intercept, pop_mu_hs_2_x, pop_mu_b_3_z]';",
        code)
    @test occursin("pop_mu", code)
    pop_decls = [d for d in generative_plan(sb).declarations
                 if d.target === :pop_mu]
    @test length(pop_decls) == 1

    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "structured Horseshoe wildcard and scaled columns" begin
    brmi = @brm HS_COLS begin
        mu ~ 1 + x + z
        effect(mu, :) ~ Horseshoe(local_scale=0.5, global_scale=0.25)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    code = BRM.stan_code(SBBRMI(brmi; mod=@__MODULE__))
    # Every column (intercept included) gets the configured triple.
    @test occursin("hs_1_Intercept_lambda ~ cauchy(0.0, 0.5);", code)
    @test occursin("hs_1_Intercept_tau ~ cauchy(0.0, 0.25);", code)
    @test occursin("hs_2_x_lambda ~ cauchy(0.0, 0.5);", code)
    @test occursin("hs_3_z_tau ~ cauchy(0.0, 0.25);", code)
    @test occursin(
        "beta_pop = [pop_mu_hs_1_Intercept, pop_mu_hs_2_x, pop_mu_hs_3_z]';",
        code)

    single = @brm HS_COLS begin
        mu ~ 0 + x
        effect(mu, x) ~ Horseshoe()
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    single_code = BRM.stan_code(SBBRMI(single; mod=@__MODULE__))
    @test occursin("hs_1_x_raw ~ std_normal();", single_code)
    @test occursin("beta_pop = [pop_mu_hs_1_x]';", single_code)
    @test StanBlocks.stanc_check(single_code; warn_pedantic=false).ok
end

@testset "structured Horseshoe fail-closed battery" begin
    # Categorical contrast blocks are out of v1 scope.
    cat = @brm HS_COLS begin
        mu ~ 1 + x + factor(g)
        effect(mu, g) ~ Horseshoe()
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    @test_throws "categorical contrast blocks fail" SBBRMI(cat; mod=@__MODULE__)

    # Random-effect sd/cor addresses are out of v1 scope.
    raneff = @brm HS_COLS begin
        mu ~ 1 + x + (1 + x | p | g)
        sd(:, p) ~ Horseshoe()
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    @test_throws "population coefficients only" SBBRMI(raneff; mod=@__MODULE__)

    # One structured prior per predictor.
    combo = @brm HS_COLS begin
        mu ~ 1 + x + z
        effect(mu, :) ~ r2d2()
        effect(mu, x) ~ Horseshoe()
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    @test_throws "one structured prior" SBBRMI(combo; mod=@__MODULE__)

    # The shared scale validator applies through the effect spelling.
    zero = @brm HS_COLS begin
        mu ~ 1 + x
        effect(mu, x) ~ Horseshoe(global_scale=0.0)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    @test_throws "finite and strictly positive" SBBRMI(zero; mod=@__MODULE__)

    positional = @brm HS_COLS begin
        mu ~ 1 + x
        effect(mu, x) ~ Horseshoe(0.5)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    @test_throws "accepts no positional arguments" SBBRMI(
        positional; mod=@__MODULE__)

    # Structured scales bake as literals; model values refuse loudly.
    literal = @brm HS_COLS begin
        mu ~ 1 + x
        effect(mu, x) ~ Horseshoe(local_scale=s)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    @test_throws "must be a numeric constant" SBBRMI(literal; mod=@__MODULE__)

    # Non-scalar sibling families refuse loudly.
    lkj = @brm HS_COLS begin
        mu ~ 1 + x + z
        effect(mu, x) ~ Horseshoe()
        effect(mu, z) ~ LKJCovarianceFactor(2)
        s ~ Exponential(1)
        y ~ Normal(mu, s)
    end
    @test_throws "no scalar Stan translation" SBBRMI(lkj; mod=@__MODULE__)
end
