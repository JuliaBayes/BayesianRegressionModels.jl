# Focused regression for natural Stan emission from the StanBlocks backend.
#
# Run: julia --startup-file=no --project=test test/stanblocks_natural_emission.jl

using Test
using BayesianRegressionModels
using Distributions: Exponential, Normal
using LogDensityProblems
using Statistics
using StanBlocks

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan
const NATURAL_EMISSION_CACHE = joinpath(tempdir(), "brm-natural-emission")

natural_df = (;
    x = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    g = [1, 1, 2, 2, 3, 3],
    y = [-1.4, -0.8, -0.1, 0.4, 1.0, 1.7],
)

@testset "native matrix conversion and homogeneous ranef SD prior" begin
    homogeneous = @brm natural_df begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | g)
        sd(:, p) ~ Exponential(2)
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(homogeneous; mod=@__MODULE__)
    code = BRM.stan_code(sb)

    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test !occursin("matrix reshape(", code)
    @test !occursin(" = reshape(", code)
    @test occursin(
        "b_p_g_z = to_matrix(b_p_g_z_flat, n_terms_p_g, n_g);", code)
    @test occursin("vector<lower=0.0>[n_terms_p_g] b_p_g_tau;", code)
    @test occursin("b_p_g_tau ~ exponential((1.0 ./ 2));", code)
    @test !occursin(r"brm_vector_prior_[0-9a-f]+", code)

    block = only(ranef_blocks(sb))
    @test (block.binding, block.n_terms, block.n_groups) == (:b_p_g, 2, 3)

    heterogeneous = @brm natural_df begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | g)
        sd(:, p) ~ Exponential(2)
        sd(mu, p, x) ~ Normal(0, 0.5)
        y ~ Normal(mu, sigma)
    end
    heterogeneous_code = BRM.stan_code(SBBRMI(heterogeneous; mod=@__MODULE__))
    @test StanBlocks.stanc_check(heterogeneous_code; warn_pedantic=false).ok
    @test occursin(r"b_p_g_tau ~ brm_vector_prior_[0-9a-f]+", heterogeneous_code)
    @test occursin("exponential_lpdf(x[1]", heterogeneous_code)
    @test occursin("normal_lpdf(x[2]", heterogeneous_code)
end

# Reference family spelling the previous homogeneous lowering: one scalar
# Exponential kernel per vector coordinate. It is used only to compare the
# normalized density and unconstrained coordinate map against Stan's native
# vectorized Exponential statement.
function brm_test_iid_exponential end
function brm_test_iid_normal end
StanBlocks.@deffun begin
    @lhs @lpxf brm_test_iid_exponential_lpdf(
        x::vector[n], rate::real)::real = begin
        lp = 0.0
        for i in 1:n
            lp += exponential_lpdf(x[i], rate)::real
        end
        lp
    end
    @lhs @lpxf brm_test_iid_normal_lpdf(
        x::vector[n], location::real, scale::real)::real = begin
        lp = 0.0
        for i in 1:n
            lp += normal_lpdf(x[i], location, scale)::real
        end
        lp
    end
end

@testset "homogeneous vector density and prior RNG preserve semantics" begin
    direct = StanBlocks.@slic (; y=[0.2, -0.1]) begin
        theta ~ exponential(0.5; n=2, lower=0.0)
        y ~ normal(theta, 1.0)
    end
    coordinatewise = StanBlocks.@slic (; y=[0.2, -0.1]) begin
        theta ~ brm_test_iid_exponential(0.5; n=2, lower=0.0)
        y ~ normal(theta, 1.0)
    end
    direct_code = StanBlocks.stan_code(direct)
    coordinatewise_code = StanBlocks.stan_code(coordinatewise)
    @test StanBlocks.stanc_check(direct_code; warn_pedantic=false).ok
    @test StanBlocks.stanc_check(coordinatewise_code; warn_pedantic=false).ok

    mkpath(NATURAL_EMISSION_CACHE)
    direct_problem = StanBlocks.stan_instantiate(
        direct; path=joinpath(NATURAL_EMISSION_CACHE, "direct.stan"))
    coordinatewise_problem = StanBlocks.stan_instantiate(
        coordinatewise;
        path=joinpath(NATURAL_EMISSION_CACHE, "coordinatewise.stan"))
    @test LogDensityProblems.dimension(direct_problem) == 2
    @test LogDensityProblems.dimension(coordinatewise_problem) == 2
    @test BS.param_unc_names(direct_problem.model) ==
          BS.param_unc_names(coordinatewise_problem.model) ==
          ["theta.1", "theta.2"]
    for q in ([-0.7, 0.4], [0.0, 0.0], [0.8, -0.2])
        @test BS.log_density(direct_problem.model, q;
            propto=false, jacobian=true) ≈
              BS.log_density(coordinatewise_problem.model, q;
                  propto=false, jacobian=true) atol=1e-12
    end

    # A lower-constrained Normal sampling statement is the same unnormalised
    # positive-support kernel as the former coordinate-wise helper. In
    # particular, native emission must not turn it into a truncated density
    # with a parameter-dependent normalising constant.
    direct_normal = StanBlocks.@slic (; y=[0.2, -0.1]) begin
        theta ~ normal(0.1, 0.5; n=2, lower=0.0)
        y ~ normal(theta, 1.0)
    end
    coordinatewise_normal = StanBlocks.@slic (; y=[0.2, -0.1]) begin
        theta ~ brm_test_iid_normal(0.1, 0.5; n=2, lower=0.0)
        y ~ normal(theta, 1.0)
    end
    direct_normal_problem = StanBlocks.stan_instantiate(
        direct_normal; path=joinpath(NATURAL_EMISSION_CACHE, "direct-normal.stan"))
    coordinatewise_normal_problem = StanBlocks.stan_instantiate(
        coordinatewise_normal;
        path=joinpath(NATURAL_EMISSION_CACHE, "coordinatewise-normal.stan"))
    for q in ([-0.7, 0.4], [0.0, 0.0], [0.8, -0.2])
        @test BS.log_density(direct_normal_problem.model, q;
            propto=false, jacobian=true) ≈
              BS.log_density(coordinatewise_normal_problem.model, q;
                  propto=false, jacobian=true) atol=1e-12
    end

    prior = @brm (; x=natural_df.x, g=natural_df.g) begin
        mu ~ 1 + x + (1 + x | p | g)
        sd(:, p) ~ Exponential(2)
    end
    prior_sb = SBBRMI(prior; mod=@__MODULE__)
    prior_code = BRM.stan_code(prior_sb)
    @test StanBlocks.stanc_check(prior_code; warn_pedantic=false).ok
    @test occursin(
        "b_p_g_tau = exponential_vector_rng(n_terms_p_g, (1.0 ./ 2));",
        prior_code)
    @test occursin(r"parameters\s*\{\s*\}", prior_code)

    prior_problem = StanBlocks.stan_instantiate(
        prior_sb.model; path=joinpath(NATURAL_EMISSION_CACHE, "prior.stan"))
    @test LogDensityProblems.dimension(prior_problem) == 0
    names = BS.param_names(prior_problem.model; include_tp=true, include_gq=true)
    tau_indices = Int[findfirst(==("b_p_g_tau.$i"), names) for i in 1:2]
    draws = Matrix{Float64}(undef, 256, 2)
    for draw_index in axes(draws, 1)
        values = BS.param_constrain(
            prior_problem.model, Float64[];
            include_tp=true, include_gq=true,
            rng=BS.StanRNG(prior_problem.model, 20_000 + draw_index))
        draws[draw_index, :] = values[tau_indices]
    end
    @test all(draws .>= 0.0)
    @test vec(mean(draws; dims=1)) ≈ [2.0, 2.0] atol=0.3
end
