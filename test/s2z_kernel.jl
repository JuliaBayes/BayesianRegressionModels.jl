# S2Z geometry kernel: implicit Helmert, Sean's rho partial map and the
# per-cell Fisher reliability candidate. Standalone script; run with
# `julia --project=test test/s2z_kernel.jl` (stdlib-only, also runs on the
# root project).
using BayesianRegressionModels, LinearAlgebra, Random, Statistics, Test

const BRM = BayesianRegressionModels

# Dense Helmert oracle, matching Stan's `sum_to_zero_constrain` orientation.
function dense_helmert(J)
    Q = zeros(J, J - 1)
    for k in 1:J-1
        Q[1:k, k] .= inv(sqrt(k * (k + 1)))
        Q[k + 1, k] = -k / sqrt(k * (k + 1))
    end
    Q
end

rng = MersenneTwister(20260920)

@testset "implicit Helmert matches dense" begin
    for J in (2, 3, 7, 20)
        Q = dense_helmert(J)
        @test Q'Q ≈ I
        @test vec(Q'ones(J)) ≈ zeros(J - 1) atol = 1e-14
        v = randn(rng, J - 1)
        u = randn(rng, J)
        @test BRM._s2z_helmert_mul(v) ≈ Q * v atol = 1e-12
        @test BRM._s2z_helmert_transpose_mul(u) ≈ Q'u atol = 1e-12
        @test BRM._s2z_helmert_transpose_mul(BRM._s2z_helmert_mul(v)) ≈ v atol = 1e-12
        @test sum(BRM._s2z_helmert_mul(v)) ≈ 0.0 atol = 1e-12
    end
    @test_throws ArgumentError BRM._s2z_helmert_mul(Float64[])
    @test_throws ArgumentError BRM._s2z_helmert_transpose_mul([1.0])
end

@testset "rho partial map: roundtrip, zero-sum, endpoints" begin
    for J in (2, 5, 20), trial in 1:3
        Q = dense_helmert(J)
        tau = exp(randn(rng) * 1.5)
        rho = rand(rng, J)
        v = randn(rng, J - 1)
        u = Q * v
        delta = BRM._s2z_partial_forward(u, tau, rho)
        @test sum(delta) ≈ 0.0 atol = 1e-10
        @test BRM._s2z_partial_inverse(delta, tau, rho) ≈ u atol = 1e-10
        # Endpoints: rho = 0 scales, rho = 1 is the identity.
        @test BRM._s2z_partial_forward(u, tau, zeros(J)) ≈ tau .* u atol = 1e-12
        @test BRM._s2z_partial_forward(u, tau, ones(J)) ≈ u atol = 1e-10
        @test BRM._s2z_partial_logjac(tau, zeros(J)) ≈
            (J - 1) * log(tau) atol = 1e-12
        @test BRM._s2z_partial_logjac(tau, ones(J)) ≈ 0.0 atol = 1e-12
    end
    u = [0.5, -0.5]
    @test_throws ArgumentError BRM._s2z_partial_forward([0.0], 1.0, [0.5])
    @test_throws ArgumentError BRM._s2z_partial_forward(u, 0.0, [0.5, 0.5])
    @test_throws ArgumentError BRM._s2z_partial_forward(u, 1.0, [0.5, 2.0])
    @test_throws DimensionMismatch BRM._s2z_partial_forward(u, 1.0, [0.5])
    @test_throws ArgumentError BRM._s2z_partial_inverse([0.0], 1.0, [0.5])
    @test_throws ArgumentError BRM._s2z_partial_logjac(1.0, [0.5])
end

@testset "rho partial log-det vs dense and finite differences" begin
    for J in (2, 4, 9), trial in 1:3
        Q = dense_helmert(J)
        tau = exp(randn(rng))
        rho = rand(rng, J)
        d = 1 .- rho .+ rho .* tau
        analytic = BRM._s2z_partial_logjac(tau, rho)
        # Dense restricted determinant: Q' * tau * (I - 11'/J) * D^-1 * Q.
        T = tau .* (I - fill(inv(J), J, J)) * Diagonal(inv.(d))
        dense, sign = logabsdet(Symmetric(Q'T * Q))
        @test sign == 1
        @test dense ≈ analytic atol = 2e-11
        # Finite-difference Jacobian of the square v -> Q'delta map.
        square(v) = Q'BRM._s2z_partial_forward(Q * v, tau, rho)
        v = randn(rng, J - 1)
        h = 1e-6
        F = Matrix{Float64}(undef, J - 1, J - 1)
        for i in eachindex(v)
            vp, vm = copy(v), copy(v)
            vp[i] += h
            vm[i] -= h
            F[:, i] .= (square(vp) - square(vm)) / (2h)
        end
        fd, fsign = logabsdet(F)
        @test fsign == 1
        @test fd ≈ analytic atol = 1e-6
    end
end

@testset "Fisher candidate: endpoints, symmetry, closed form" begin
    # Closed form for M = 1 with equal group information I:
    # raw = K / (K + 1) with K = sd^2 * I, then the brms rho rescale.
    closed(J, I, sd) = begin
        raw = (sd^2 * I) / (sd^2 * I + 1)
        raw / (raw + (1 - raw) * sd)
    end
    for J in (2, 5, 20), (I, sd) in ((0.0, 1.7), (0.3, 0.4), (2.5, 1.7), (1e6, 0.9))
        infos = [fill(I, 1, 1) for _ in 1:J]
        rho = BRM._s2z_fisher_candidate(infos, [sd])
        @test size(rho) == (J, 1)
        @test all(0 .<= rho .<= 1)
        @test rho ≈ fill(closed(J, I, sd), J, 1) atol = 1e-10
    end
    # Zero information recovers NCP weights; overwhelming information CP weights.
    @test all(BRM._s2z_fisher_candidate(
        [zeros(2, 2) for _ in 1:4], [1.3, 0.7]) .< 1e-12)
    @test all(BRM._s2z_fisher_candidate(
        [fill(1e12, 2, 2) + 1e12 * I for _ in 1:4], [1.3, 0.7]) .> 1 - 1e-9)
    # Permuting groups permutes candidate rows; identical groups agree.
    J, sd = 6, [1.1, 0.6]
    ramp = [fill(0.1 * j, 2, 2) + 0.1 * j * I for j in 1:J]
    rho = BRM._s2z_fisher_candidate(ramp, sd)
    @test all(isfinite, rho)
    p = [4, 1, 6, 2, 5, 3]
    @test BRM._s2z_fisher_candidate(ramp[p], sd) ≈ rho[p, :] atol = 1e-12
    balanced = [Diagonal([0.5, 2.0]) for _ in 1:J]
    even = BRM._s2z_fisher_candidate(balanced, sd)
    @test even ≈ repeat(even[1:1, :], J, 1) atol = 1e-12
    @test_throws ArgumentError BRM._s2z_fisher_candidate([fill(1.0, 1, 1)], [1.0])
    @test_throws ArgumentError BRM._s2z_fisher_candidate(
        [fill(1.0, 1, 1) for _ in 1:2], [0.0])
    @test_throws DimensionMismatch BRM._s2z_fisher_candidate(
        [fill(1.0, 2, 2) for _ in 1:2], [1.0])
end
