using Test
using BayesianRegressionModels
using Distributions
using StanBlocks

const BRM = BayesianRegressionModels

@testset "R2 prior call is independent of variance-allocation geometry" begin
    df = (; x=[-1.0, -0.3, 0.4, 1.1], y=[0.1, -0.2, 0.5, 0.8],
            g=[1, 1, 2, 2])
    whole = @brm begin
        prior_location ~ Normal(0.5, 0.1)
        mu ~ 1 + x + (1 | g)
        effect(mu, :) ~ r2d2(R2=Normal(prior_location, 0.2), tau_bsv=0.5)
        y ~ Normal(mu, 1.0)
    end
    sb = SBBRMI(whole(df); mod=@__MODULE__)
    code = BRM.stan_code(sb)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("r2d2_mu_R2 ~ normal(prior_location, 0.2);", code)
    @test occursin(r"real<lower\s*=\s*0.0,\s*upper\s*=\s*1.0> r2d2_mu_R2", code)
    @test !occursin("normal_lcdf", code)
    @test findfirst("prior_location ~", code) < findfirst("r2d2_mu_R2 ~", code)

    shared = @brm begin
        mu ~ 1 + x + (1 + x | p | g)
        sd(:, p) ~ r2d2(R2=Uniform(0.1, 0.8), reference_scale=1.0)
        y ~ Normal(mu, 1.0)
    end
    shared_code = BRM.stan_code(SBBRMI(shared(df); mod=@__MODULE__))
    @test StanBlocks.stanc_check(shared_code; warn_pedantic=false).ok
    @test occursin("uniform(0.1, 0.8)", shared_code)
    @test occursin(r"real<lower\s*=\s*0.1,\s*upper\s*=\s*0.8>", shared_code)

    impossible = @brm begin
        mu ~ 1 + x + (1 | g)
        effect(mu, :) ~ r2d2(R2=Uniform(2.0, 3.0), tau_bsv=0.5)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "empty" SBBRMI(impossible(df); mod=@__MODULE__)
end
