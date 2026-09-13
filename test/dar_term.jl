# test/dar_term.jl — semantic and executable contract for the direct
# differenced-AR(1) formula term.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Cauchy, Normal
import StanBlocks.stan: transpiles

const DAR_RUNTIME = get(ENV, "BRM_DAR_RUNTIME", "1") != "0"
const DAR_CACHE = joinpath(tempdir(), "brm-dar-term")
const BS = StanBlocks.BridgeStan

dar_df(n=4) = (; t=collect(1.0:n), y=zeros(n))

function dar_model(df=dar_df())
    @brm df begin
        mu ~ 1 + dar(t; p=1)
        y ~ Normal(mu, 1.0)
    end
end

stan(brmi) = StanBlocks.stan_code(SBBRMI(brmi; mod=@__MODULE__).model)

@testset "dar persistence accepts an arbitrary translated scalar prior" begin
    df = dar_df()
    configured = @brm df begin
        mu ~ 1 + dar(t)
        ar(:, dar(t)) ~ Cauchy(0.5, 0.2)
        y ~ Normal(mu, 1.0)
    end
    code = stan(configured)
    @test occursin("real<lower=0.0, upper=1.0> dar_mu_t_beta", code)
    @test occursin("dar_mu_t_beta ~ cauchy(0.5, 0.2)", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "dar is a direct differenced-AR trajectory, unlike ar" begin
    df = dar_df()
    ordinary = @brm df begin
        mu ~ 1 + ar(t; p=1)
        y ~ Normal(mu, 1.0)
    end
    differenced = dar_model(df)

    # The reported cost: ordinary `ar` occupies a population-design column and
    # therefore gets an extra beta. `dar` owns its complete model-scale path;
    # only the initial-level intercept remains in `beta_pop`.
    @test length(popcoefnames(ordinary, :mu)) == 2
    @test popcoefnames(differenced, :mu) == [:Intercept]

    code = stan(differenced)
    @test occursin("differenced_ar1_path", code)
    @test occursin("dar_mu_t_beta", code)
    @test occursin("dar_mu_t_sigma", code)
    @test occursin("dar_mu_t_z", code)
    @test transpiles(SBBRMI(differenced; mod=@__MODULE__).model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "dar term priors are inspectable and bounded" begin
    df = dar_df()
    configured = @brm df begin
        mu ~ 1 + dar(t)
        ar(:, dar(t)) ~ Normal(0.4, 0.1)
        sd(:, dar(t)) ~ Normal(0.0, 0.3)
        y ~ Normal(mu, 1.0)
    end
    specs = term_priors(configured)
    @test Set(s.class for s in specs) == Set((:term_ar, :term_sd))
    @test all(s.term === Symbol("dar(t)") for s in specs)

    code = stan(configured)
    @test occursin("real<lower=0.0, upper=1.0> dar_mu_t_beta", code)
    @test occursin("dar_mu_t_beta ~ normal(0.4, 0.1)", code)
    @test occursin("real<lower=0.0> dar_mu_t_sigma", code)
    @test occursin("dar_mu_t_sigma ~ normal(0.0, 0.3)", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "dar replay and descriptor contract" begin
    sb = SBBRMI(dar_model(); mod=@__MODULE__)
    replay = reprocess(sb, dar_df(7))
    @test replay.data[:t] == collect(1.0:7.0)
    @test StanBlocks.stan_code(replay.model) == StanBlocks.stan_code(sb.model)

    d = brm_descriptor(sb)
    @test any(o -> o.logical === :dar_mu_t, d.outputs)
    @test :reprocess in Symbol[op.name for op in d.operations]

    duplicate_time = (; t=[1.0, 2.0, 2.0, 4.0], y=zeros(4))
    @test_throws "strictly increasing" SBBRMI(dar_model(duplicate_time); mod=@__MODULE__)
end

@testset "dar BridgeStan recurrence, density, gradient, and coordinates" begin
    if DAR_RUNTIME
        sb = SBBRMI(dar_model(); mod=@__MODULE__)
        code = StanBlocks.stan_code(sb.model)
        isdir(DAR_CACHE) || mkpath(DAR_CACHE)
        problem = StanBlocks.stan_instantiate(
            sb.model; path=joinpath(DAR_CACHE, string(hash(code)) * ".stan"))
        sm = problem.model
        names = BS.param_names(sm)
        string_names = String.(names)
        q = zeros(LogDensityProblems.dimension(problem))

        beta_i = only(findall(==("dar_mu_t_beta"), string_names))
        sigma_i = only(findall(==("dar_mu_t_sigma"), string_names))
        z_i = findall(startswith("dar_mu_t_z."), string_names)
        @test length(z_i) == 3

        # On the unconstrained scale, beta=0 maps to 0.5 under [0,1], while
        # sigma=log(0.1) maps to 0.1. With z=[1,-2,0.5], the exact path is
        # [0, 0.1, -0.05, -0.075]. The population intercept stays at zero.
        q[beta_i] = 0.0
        q[sigma_i] = log(0.1)
        q[z_i] .= [1.0, -2.0, 0.5]
        constrained_names = BS.param_names(sm; include_tp=true, include_gq=false)
        constrained = BS.param_constrain(
            sm, q; include_tp=true, include_gq=false)
        mu = [v for (nm, v) in zip(constrained_names, constrained)
              if startswith(String(nm), "mu.")]
        @test mu ≈ [0.0, 0.1, -0.05, -0.075]

        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test length(gradient) == length(q)
        @test all(isfinite, gradient)

        descriptor = brm_descriptor(sb)
        @test length(brm_term_coordinates(
            descriptor, :mu, constrained_names;
            term=:dar_mu_t, parameter=:ar).coordinates) == 1
        @test length(brm_term_coordinates(
            descriptor, :mu, constrained_names;
            term=:dar_mu_t, parameter=:sd).coordinates) == 1
        @test length(brm_term_coordinates(
            descriptor, :mu, constrained_names;
            term=:dar_mu_t, parameter=:innovations).coordinates) == 3
    else
        @info "Skipping BridgeStan dar runtime gate (BRM_DAR_RUNTIME=0)"
        @test true
    end
end

println("dar_term.jl: all testsets passed")
