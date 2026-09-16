# test/rw_term.jl — semantic and executable contract for the random-walk
# formula term: `dar(time)` with the increments' persistence fixed at zero.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Normal
import StanBlocks.stan: transpiles

const RW_RUNTIME = get(ENV, "BRM_RW_RUNTIME", "1") != "0"
const RW_CACHE = joinpath(tempdir(), "brm-rw-term")
const BS = StanBlocks.BridgeStan

rw_df(n=4) = (; t=collect(1.0:n), y=zeros(n))

function rw_model(df=rw_df())
    @brm df begin
        mu ~ 1 + rw(t)
        y ~ Normal(mu, 1.0)
    end
end

stan(brmi) = StanBlocks.stan_code(SBBRMI(brmi; mod=@__MODULE__).model)

@testset "rw is a direct random-walk trajectory: one intercept, no persistence" begin
    df = rw_df()
    walk = rw_model(df)
    @test popcoefnames(walk, :mu) == [:Intercept]
    code = stan(walk)
    @test occursin("random_walk_path", code)
    @test occursin("rw_mu_t_sigma", code)
    @test occursin("rw_mu_t_z", code)
    @test !occursin("rw_mu_t_beta", code)
    @test transpiles(SBBRMI(walk; mod=@__MODULE__).model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "rw innovation scale is addressable; persistence is refused" begin
    df = rw_df()
    configured = @brm df begin
        mu ~ 1 + rw(t)
        sd(:, rw(t)) ~ Normal(0.0, 0.05)
        y ~ Normal(mu, 1.0)
    end
    specs = term_priors(configured)
    @test Set(s.class for s in specs) == Set((:term_sd,))
    @test all(s.term === Symbol("rw(t)") for s in specs)
    code = stan(configured)
    @test occursin("real<lower=0.0> rw_mu_t_sigma", code)
    @test occursin("rw_mu_t_sigma ~ normal(0.0, 0.05)", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    refused = @brm df begin
        mu ~ 1 + rw(t)
        ar(:, rw(t)) ~ Normal(0.4, 0.1)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "no bounded persistence coefficient" SBBRMI(refused; mod=@__MODULE__)
end

@testset "rw replay and descriptor contract" begin
    sb = SBBRMI(rw_model(); mod=@__MODULE__)
    @test sb.data[:rw_mu_t_n_steps] == 4
    @test sb.data[:rw_mu_t_time_idx] == [1, 2, 3, 4]
    replay = reprocess(sb, rw_df(7))                       # the grid grows: a forecast
    @test replay.data[:rw_mu_t_n_steps] == 7
    @test replay.data[:rw_mu_t_time_idx] == collect(1:7)
    @test StanBlocks.stan_code(replay.model) == StanBlocks.stan_code(sb.model)

    d = brm_descriptor(sb)
    @test any(o -> o.logical === :rw_mu_t, d.outputs)
    @test :reprocess in Symbol[op.name for op in d.operations]

    # rows sharing a time read one shared walk (a long frame with two groups per day)
    shared = SBBRMI(rw_model((; t=[1.0, 1.0, 2.0, 2.0, 3.0, 3.0], y=zeros(6))); mod=@__MODULE__)
    @test shared.data[:rw_mu_t_n_steps] == 3
    @test shared.data[:rw_mu_t_time_idx] == [1, 1, 2, 2, 3, 3]
    @test StanBlocks.stanc_check(StanBlocks.stan_code(shared.model); warn_pedantic=false).ok
end

@testset "rw BridgeStan path, density, gradient, and coordinates" begin
    if RW_RUNTIME
        sb = SBBRMI(rw_model(); mod=@__MODULE__)
        code = StanBlocks.stan_code(sb.model)
        isdir(RW_CACHE) || mkpath(RW_CACHE)
        problem = StanBlocks.stan_instantiate(
            sb.model; path=joinpath(RW_CACHE, string(hash(code)) * ".stan"))
        sm = problem.model
        string_names = String.(BS.param_names(sm))
        q = zeros(LogDensityProblems.dimension(problem))
        @test LogDensityProblems.dimension(problem) == 1 + 1 + 3   # intercept, sigma, z[1:3]

        sigma_i = only(findall(==("rw_mu_t_sigma"), string_names))
        z_i = findall(startswith("rw_mu_t_z."), string_names)
        @test length(z_i) == 3
        @test !any(==("rw_mu_t_beta"), string_names)

        # sigma = 0.1 (log-scale coordinate) and z = [1, -2, 0.5] give the exact
        # zero-started walk [0, 0.1, -0.1, -0.05]; the intercept stays at zero.
        q[sigma_i] = log(0.1)
        q[z_i] .= [1.0, -2.0, 0.5]
        constrained_names = BS.param_names(sm; include_tp=true, include_gq=false)
        constrained = BS.param_constrain(sm, q; include_tp=true, include_gq=false)
        mu = [v for (nm, v) in zip(constrained_names, constrained)
              if startswith(String(nm), "mu.")]
        @test mu ≈ [0.0, 0.1, -0.1, -0.05]

        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test all(isfinite, gradient)

        descriptor = brm_descriptor(sb)
        @test length(brm_term_coordinates(
            descriptor, :mu, constrained_names;
            term=:rw_mu_t, parameter=:sd).coordinates) == 1
        @test length(brm_term_coordinates(
            descriptor, :mu, constrained_names;
            term=:rw_mu_t, parameter=:innovations).coordinates) == 3
    else
        @info "Skipping BridgeStan rw runtime gate (BRM_RW_RUNTIME=0)"
        @test true
    end
end

println("rw_term.jl: all testsets passed")
