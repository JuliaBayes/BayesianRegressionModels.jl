# Distribution calls compose with observation structure and constrained priors.
using Test
using BayesianRegressionModels
using Distributions
using StanBlocks
using LogDensityProblems

const BRM = BayesianRegressionModels

@testset "missing response rewrites arbitrary continuous constructor arguments" begin
    df = (; x=[-0.6, 0.2, 0.7, 1.2],
            y=Union{Missing,Float64}[0.4, missing, 0.9, missing])
    normal = @brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        mi(y) ~ Normal(mu, sigma)
    end
    gamma = @brm begin
        log(rate) ~ 1 + x
        mi(y) ~ Gamma(2, 1 / rate)
    end
    student = @brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        mi(y) ~ LocationScale(mu + 0.5, sigma, TDist(4))
    end
    # The Student location-scale adapter is not the same call shape as Normal:
    # a generic missing-row rewrite must use its translated arguments.
    for builder in (normal, gamma, student)
        sb = SBBRMI(builder(df); mod=@__MODULE__)
        code = BRM.stan_code(sb)
        @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
        @test occursin("mi_merge(", code)
        @test sb.data[:Jobs_y] == [1, 3]
        @test sb.data[:Jmis_y] == [2, 4]
    end
    code = BRM.stan_code(SBBRMI(normal(df); mod=@__MODULE__))
    @test occursin("y_obs ~ normal(mu[Jobs_y], sigma);", code)
    @test !occursin("sigma[", code)
end

@testset "objective weights wrap the translated observation call" begin
    df = (; y=[-0.2, 0.5, 1.1], power=[0.5, 2.0, 1.25])
    builder = @brm begin
        y ~ weighted(LocationScale(0.3, 0.7, TDist(4)), weights(power))
    end
    descriptor = brm_descriptor(builder, df; mod=@__MODULE__, highlights=())
    code = BRM.stan_code(descriptor.plan)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("weighted_student_t_lpdf", code)
    cache = joinpath(tempdir(), "brm-generic-weighted-observation")
    mkpath(cache)
    problem = brm_execute(descriptor, :instantiate;
        path=joinpath(cache, string(descriptor.id) * ".stan"))
    @test LogDensityProblems.dimension(problem) == 0
    pointwise = brm_execute(descriptor, :pointwise_loglik;
        problem, draws=Float64[], seed=20260913)
    @test pointwise.y_likelihood ≈
          df.power .* logpdf.(LocationScale(0.3, 0.7, TDist(4)), df.y) atol=1e-10
end

@testset "covariance scales retain a general positive-constrained prior kernel" begin
    df = (; x=[-1.0, 0.0, 1.0], y1=[0.1, 0.2, -0.1], y2=[1.1, 0.9, 1.2])
    builder = @brm begin
        L_res ~ LKJCovarianceFactor(2; scale_prior=Normal(0.3, 0.7), shape=2)
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end
    brmi = builder(df)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BRM.stan_code(sb)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("L_res_scales ~ normal(0.3, 0.7);", code)
    @test occursin("vector<lower = 0.0>[L_res_n] L_res_scales;", code) ||
          occursin("vector<lower=0.0>[L_res_n] L_res_scales;", code)
    # Declaration bounds restrict the kernel; an explicit truncated prior would
    # carry a normalization correction, which must not be invented here.
    @test !occursin("normal_lccdf", code)
    @test !occursin("normal_lcdf", code)
end
