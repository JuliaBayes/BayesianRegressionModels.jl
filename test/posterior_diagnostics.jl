using Test, BayesianRegressionModels, BridgeStan, StanBlocks
using Random, LogDensityProblems, LinearAlgebra
using Distributions

const BRM = BayesianRegressionModels
const BS = BridgeStan
const D = (; x=collect(range(-1, 1; length=7)), y=[0.2, 0.1, -0.3, 0.4, 0.2, -0.2, 0.1],
           cm=[0.0, 0.4, 0.8], cs=[0.2, 0.6, 1.0])
const MODEL = @brm begin
    length_scale(:, hsgp(x)) ~ LogNormal(0, 4)
    sd(:, hsgp(x)) ~ LogNormal(0, 4)
    mu ~ hsgp(x; k=3, domain=(-1.5, 1.5), centeredness=cm)
    log(sigma) ~ hsgp(x; k=3, domain=(-1.5, 1.5), centeredness=cs)
    y ~ Normal(mu, sigma)
end

@testset "native posterior diagnostics use descriptor-owned values" begin
    sb = SBBRMI(MODEL(D); mod=@__MODULE__)
    descriptor = brm_descriptor(sb)
    scratch = mktempdir()
    problem = StanBlocks.stan_instantiate(sb.model;
        path=joinpath(scratch, "posterior-diagnostics.stan"))
    q = 0.1randn(Xoshiro(14), 5, LogDensityProblems.dimension(problem))
    names = BS.param_names(problem.model)
    draws = permutedims(reduce(hcat, [BS.param_constrain(problem.model, collect(row))
                                    for row in eachrow(q)]))
    full_names = BS.param_names(problem.model; include_tp=true, include_gq=false)
    full = permutedims(reduce(hcat, [BS.param_constrain(problem.model, collect(row);
                                   include_tp=true, include_gq=false) for row in eachrow(q)]))
    mu = brm_output_draws(descriptor, full, full_names; logical=:mu)
    sigma = brm_output_draws(descriptor, full, full_names; logical=:sigma)
    @test size(mu) == (5, 7)
    @test size(sigma) == (5, 7)
    @test all(>(0), sigma)
    @test_throws DimensionMismatch brm_output_draws(
        descriptor, permutedims(full), full_names; logical=:mu)
    @test_throws ErrorException brm_output_draws(
        descriptor, full, fill("duplicate", length(full_names)); logical=:mu)

    for (predictor, c, expected) in ((:mu, D.cm, mu), (:sigma, D.cs, log.(sigma)))
        term = :hsgp_x
        matched = brm_term_coordinates(descriptor, predictor, names;
            term, parameter=:basis_weights)
        gradients = permutedims(reduce(hcat, [last(LogDensityProblems.logdensity_and_gradient(problem, collect(row)))
                                              for row in eachrow(q)]))[:, matched.coordinates]
        original = hsgp_coordinate_draws(descriptor, draws, names;
            predictor, term, basis_gradients=gradients)
        ncp = hsgp_coordinate_draws(descriptor, draws, names;
            predictor, term, centeredness=0.0, basis_gradients=gradients)
        cp = hsgp_coordinate_draws(descriptor, draws, names;
            predictor, term, centeredness=1.0, basis_gradients=gradients)
        @test original.sampled_centeredness == c
        @test original.coordinates == draws[:, matched.coordinates]
        @test original.gradients == gradients
        @test ncp.coordinates ≈ original.unit_weights
        @test cp.coordinates ≈ original.physical_weights
        @test ncp.coordinates .* ncp.gradients ≈ original.coordinates .* gradients
        @test cp.coordinates .* cp.gradients ≈ original.coordinates .* gradients
        phi = [sin(pi / 3 * (x + 1.5) * j) / sqrt(1.5) for x in D.x, j in 1:3]
        @test original.physical_weights * phi' ≈ expected atol=2e-14 rtol=2e-14
        @test all(original.finite) && all(ncp.finite) && all(cp.finite)

        # Differentiate the actual compiled target plus its coordinate Jacobian,
        # with hyperparameters fixed. Do not use another copy of the gradient rule.
        for i in 1:2, b in 1:3
            u = cp.coordinates[i, b]
            log_s = cp.log_scales[i, b]
            index = matched.coordinates[b]
            density(u) = begin
                pos = copy(q[i, :])
                pos[index] = u * exp((c[b] - 1) * log_s)
                LogDensityProblems.logdensity(problem, pos) + (c[b] - 1) * log_s
            end
            h = 1e-6
            @test (density(u + h) - density(u - h)) / (2h) ≈ cp.gradients[i, b] atol=2e-6 rtol=2e-6
        end
    end

    pp = brm_predictive_draws(descriptor, q; problem, seed=42)
    @test keys(pp) == (:y,)
    @test size(pp.y) == (5, 7)
    @test all(isfinite, pp.y)
    @test pp == brm_predictive_draws(descriptor, q; problem, seed=42)
    @test pp != brm_predictive_draws(descriptor, q; problem, seed=43)
    @test pp.y != mu
    for i in axes(q, 1)
        exact = brm_execute(descriptor, :predict; problem, draws=q[i, :], seed=41+i)
        @test pp.y[i, :] == exact.y_gen
    end
end
