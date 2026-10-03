# Actual BRM-emitted Gaussian source: analytic and mapped Stan acceptance,
# followed by allocation measurements of native density and Enzyme reverse.
using Test, BayesianRegressionModels, StanBlocks, Distributions
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
import BridgeStan
include(joinpath(@__DIR__, "rk_source_roundtrip.jl"))

function gaussian_emission_reference(data, q)
    a, b, logsigma = q
    sigma = exp(logsigma)
    residual = data.y .- a .- b .* data.x
    n = length(residual)
    ss = sum(abs2, residual) / sigma^2
    value = -0.5ss - n * logsigma - (n / 2 + 1) * log(2pi) -
        0.5(a^2 + b^2) - sigma + logsigma
    gradient = [sum(residual) / sigma^2 - a,
        sum(residual .* data.x) / sigma^2 - b, ss - n - sigma + 1]
    value, gradient
end

function gaussian_emission_allocations(query, u)
    gradient = similar(u)
    density() = query(u)
    reverse() = sampler_value_and_gradient!(query, gradient, u)
    for _ in 1:10
        density()
        reverse()
    end
    density_bytes = minimum([@allocated(density()) for _ in 1:5])
    reverse_bytes = minimum([@allocated(reverse()) for _ in 1:5])
    (; density_bytes, reverse_bytes)
end

function check_gaussian_emission(n)
    x = collect(range(-1.0, 1.0; length=n))
    data = (; x, y=0.3 .+ 0.7 .* x .+ 0.2 .* sin.(3 .* x))
    saved = deepcopy(data)
    brmi = @brm data begin
        mu ~ 1 + x
        effect(mu, :) ~ Normal(0, 1)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    backend = check_rk_source_roundtrip(RKBRMI(brmi))
    names = coordinate_names(backend.model.layout)
    @test names == [:mu_Intercept, :mu_x, :sigma]
    source = BayesianRegressionModels.emit_rk_artifact(brmi; case_id="gaussian-$n")
    @test isempty(source.defs)
    @test !occursin("NormalIDGLM", sprint(Base.show_unquoted, source.ast))
    cache = joinpath(tempdir(), "brm-rk-gaussian")
    mkpath(cache)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    stan = BayesianRegressionModels.stan_instantiate(sb;
        path=joinpath(cache, "gaussian.stan"))
    permutation = BayesianRegressionModels.resolve_sb_map([
        :mu_Intercept => "pop_mu_beta_pop.1",
        :mu_x => "pop_mu_beta_pop.2", :sigma => "sigma"],
        names, BridgeStan.param_unc_names(stan.model); case_id="gaussian-$n")
    problem = rk_logdensity_problem(backend;
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    points = ([0.0, 0.0, 0.0], [0.3, 0.7, log(0.4)], [-0.2, 1.1, log(1.3)])
    for u in points
        before = copy(u)
        expected, expected_gradient = gaussian_emission_reference(data, u)
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        stan_u = BayesianRegressionModels.apply_sb_map(u, permutation)
        stan_before = copy(stan_u)
        stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model,
            stan_u; propto=false, jacobian=true)
        mapped = BayesianRegressionModels.unmap_sb_grad(stan_gradient, permutation)
        @test value ≈ expected rtol=3e-12 atol=3e-12
        @test stan_value ≈ expected rtol=3e-12 atol=3e-12
        @test gradient ≈ expected_gradient rtol=3e-11 atol=3e-11
        @test mapped ≈ expected_gradient rtol=3e-11 atol=3e-11
        @test isequal(u, before) && isequal(stan_u, stan_before)
        @test isequal(data, saved)
    end
    measured = gaussian_emission_allocations(problem.query, points[2])
    println("GAUSSIAN_EMITTED rows=$n allocations=$measured")
    @test isequal(data, saved)
end

@testset "actual ordinary Gaussian emission: analytic and mapped Stan" begin
    for n in (300, 3000, 30000)
        check_gaussian_emission(n)
    end
end
