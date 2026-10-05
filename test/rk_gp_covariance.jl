using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using LinearAlgebra, Enzyme
using DifferentiationInterface: AutoEnzyme
using Distributions
const BRM = BayesianRegressionModels
const Prep = BRM.StatisticalPreparation
const gp_chol_latent = Prep.gp_chol_latent
const exp_quad_graph = BRM.rk_model(:gp_exp_quad_cov)
const periodic_graph = BRM.rk_model(:gp_periodic_cov)

# Ordinary callbacks exercise the public vector wrappers as well as graph calls.
ordinary_exp_quad_cov(x, sigma, rho, jitter) =
    Prep.gp_exp_quad_cov(x, sigma, rho, jitter)
ordinary_periodic_cov(x, sigma, rho, period, jitter) =
    Prep.gp_periodic_cov(x, sigma, rho, period, jitter)

@kernel exp_quad_sum(q) = begin
    x = q[1:4]
    sigma = exp(q[5])
    rho = exp(q[6])
    K = exp_quad_graph(x, sigma, rho, 0.03)
    result = sum(K)
    return result
end

@kernel periodic_sum(q) = begin
    x = q[1:4]
    sigma = exp(q[5])
    rho = exp(q[6])
    period = exp(q[7])
    K = periodic_graph(x, sigma, rho, period, 0.03)
    result = sum(K)
    return result
end

@kernel ordinary_exp_quad_sum(q) = begin
    x = q[1:4]
    sigma = exp(q[5])
    rho = exp(q[6])
    K = ordinary_exp_quad_cov(x, sigma, rho, 0.03)
    result = sum(K)
    return result
end

@kernel ordinary_periodic_sum(q) = begin
    x = q[1:4]
    sigma = exp(q[5])
    rho = exp(q[6])
    period = exp(q[7])
    K = ordinary_periodic_cov(x, sigma, rho, period, 0.03)
    result = sum(K)
    return result
end

function covariance_reference(x, sigma, rho, jitter; period=nothing)
    X = x isa AbstractVector ? reshape(x, :, 1) : x
    R = rho isa Real ? fill(rho, size(X, 2)) : rho
    [sigma^2 * exp(period === nothing ?
        -sum(((X[i, d] - X[j, d]) / R[d])^2 for d in axes(X, 2)) / 2 :
        -2sin(pi * norm(X[i, :] - X[j, :]) / period)^2 / rho^2) +
        (i == j ? jitter : 0.0) for i in axes(X, 1), j in axes(X, 1)]
end

@testset "BRM GP covariance construction" begin
    x = [-0.7, 0.2, 0.2, 1.1]
    X = hcat(x, [0.1, -0.2, 0.5, 0.8])
    saved_x, saved_X = copy(x), copy(X)
    for sigma in (0.6, 1.4), rho in (0.3, 1.2), jitter in (0.0, 1e-7)
        @test Prep.gp_exp_quad_cov(x, sigma, rho, jitter) ≈
            covariance_reference(x, sigma, rho, jitter)
        @test Prep.gp_exp_quad_cov(X, sigma, rho, jitter) ≈
            covariance_reference(X, sigma, rho, jitter)
        rhos = [rho, 0.8rho]
        @test Prep.gp_exp_quad_cov(X, sigma, rhos, jitter) ≈
            covariance_reference(X, sigma, rhos, jitter)
        for period in (0.7, 2.1)
            @test Prep.gp_periodic_cov(x, sigma, rho, period, jitter) ≈
                covariance_reference(x, sigma, rho, jitter; period)
            @test Prep.gp_periodic_cov(X, sigma, rho, period, jitter) ≈
                covariance_reference(X, sigma, rho, jitter; period)
        end
    end
    @test x == saved_x && X == saved_X
    # A positional diagonal identifies observations, including duplicates.
    K = Prep.gp_exp_quad_cov(x, 1.2, 0.7, 0.03)
    @test K[2, 2] - K[2, 3] ≈ 0.03
    z = [0.2, -0.3, 0.4, -0.1]
    before_K, before_z = copy(K), copy(z)
    @test Prep.gp_chol_latent(K, z) ≈ cholesky(Symmetric(K, :L)).L * z
    upper = copy(K)
    upper[1, 3] = NaN
    @test Prep.gp_chol_latent(upper, z) == Prep.gp_chol_latent(K, z)
    @test K == before_K && z == before_z
    @test Prep.gp_chol_latent(zeros(0, 0), Float64[]) == Float64[]
    @test_throws ArgumentError("gp_chol_latent needs a square covariance, got (2, 3)") Prep.gp_chol_latent(zeros(2, 3), zeros(2))
    @test_throws ArgumentError("gp_chol_latent z has length 1 for a 2×2 covariance") Prep.gp_chol_latent(Matrix{Float64}(I, 2, 2), zeros(1))
    @test_throws ArgumentError("gp_chol_latent covariance is not positive definite (non-positive pivot at 2 — increase jitter)") Prep.gp_chol_latent([1.0 0.0; 0.0 0.0], zeros(2))
    @test_throws ArgumentError("gp_chol_latent covariance is not positive definite (non-positive pivot at 1 — increase jitter)") Prep.gp_chol_latent([NaN 0.0; 0.0 1.0], zeros(2))
    for name in (:gp_exp_quad_cov, :gp_periodic_cov)
        graph = BRM.rk_model(name)
        @test :plate in [e.kind for e in recipe_inventory(graph)]
        for locations in (x, collect(range(-1, 1; length=9)))
            values = name === :gp_exp_quad_cov ? (1.2, 0.7, 0.03) :
                (1.2, 0.7, 1.8, 0.03)
            kernel = prepare(graph; bound=(; x=locations))
            @test kernel(values...) ≈ covariance_reference(locations,
                values[1], values[2], last(values);
                period=name === :gp_periodic_cov ? values[3] : nothing)
        end
    end
    backend = AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const)
    for (graph, periodic) in ((exp_quad_sum, false), (periodic_sum, true),
                            (ordinary_exp_quad_sum, false),
                            (ordinary_periodic_sum, true))
        kernel = prepare(graph)
        point = vcat([-0.7, 0.2, 0.5, 1.1], log.([1.2, 0.7]),
            periodic ? [log(1.8)] : Float64[])
        prepared = prepare_ad(kernel, backend, point; active=:q)
        reference = p -> sum(covariance_reference(p[1:4], exp(p[5]), exp(p[6]),
            0.03; period=periodic ? exp(p[7]) : nothing))
        for shift in (0.0, 0.08, -0.04)
            q = point .+ shift
            saved = copy(q)
            value, gradient = ad_value_and_gradient!(prepared, similar(q), q)
            @test value ≈ reference(q)
            for j in eachindex(q)
                plus, minus = copy(q), copy(q)
                plus[j] += 1e-5; minus[j] -= 1e-5
                fd = (reference(plus) - reference(minus)) / 2e-5
                @test gradient[j] ≈ fd atol=2e-7 rtol=2e-6
            end
            @test q == saved
        end
    end
end

@testset "BRM-owned covariance in an explicit RKPPL GP" begin
    x = [-0.7, 0.2, 0.2, 1.1]
    y = [0.1, -0.2, 0.3, -0.1]
    model = @rkppl begin
        sigma ~ LogNormal(0, 1)
        rho ~ LogNormal(0, 1)
        z[1:4] .~ Normal.(0, 1)
        covariance = exp_quad_graph(x, sigma, rho, 0.03)
        latent = gp_chol_latent(covariance, z)
        mu = latent[oi]
        y .~ Normal.(mu, 0.8)
    end
    data = (; x, y, oi=collect(1:4))
    bound = bind_data(lower_rkppl(model, data; conditioned=(:y,)), data)
    built = build_kernel(bound)
    @test :plate in [e.kind for e in recipe_inventory(built.spec)]
    point = collect(range(-0.2, 0.15; length=built.layout.total))
    query = prepare_sampler(built, bound, point;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    function reference(q)
        nt = constrain(built.layout, q)
        K = covariance_reference(x, nt.sigma, nt.rho, 0.03)
        mu = cholesky(Symmetric(K, :L)).L * nt.z
        sum(logpdf.(Normal.(mu, 0.8), y)) +
            logpdf(LogNormal(0, 1), nt.sigma) +
            logpdf(LogNormal(0, 1), nt.rho) +
            sum(logpdf.(Normal(), nt.z)) + logjac(built.layout, q)
    end
    for shift in (0.0, 0.07, -0.05)
        q = point .+ shift
        saved = copy(q)
        value, gradient = sampler_value_and_gradient!(query, similar(q), q)
        @test value ≈ reference(q) atol=2e-11 rtol=2e-12
        for j in eachindex(q)
            plus, minus = copy(q), copy(q)
            plus[j] += 1e-5; minus[j] -= 1e-5
            fd = (reference(plus) - reference(minus)) / 2e-5
            @test gradient[j] ≈ fd atol=2e-7 rtol=2e-6
        end
        @test q == saved
    end
end
