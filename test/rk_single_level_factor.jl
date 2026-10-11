# A single-level treatment-coded factor on the RK route: the ordinary K-1
# emission at K=1 (zero coefficients, SB's `vector[0]`), an exactly zero
# contribution, and same-BRMI Stan parity through the coordinate transport.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random: Xoshiro

module SingleLevelFactor
using BayesianRegressionModels, Distributions
using Random: Xoshiro
const N = 20
const BASE = let rng = Xoshiro(7)
    (; y=randn(rng, N), x=abs.(randn(rng, N)), z=randn(rng, N),
       g=repeat(1:4, 5), src=fill("a", N), isrc=fill(1, N))
end
const TWO = merge(BASE, (; src=repeat(["a", "b"], 10), isrc=repeat([1, 2], 10)))

# The snag shape: a factor-only predictor read as one summand.
summand(data) = @brm data begin
    mu_base ~ 1 + (1 | g)
    slope ~ 1 + (1 | g)
    shift ~ factor(src; cmc=false)
    mu = mu_base + slope * x + shift
    y ~ Normal(mu, 1.0)
end
summand_free(data) = @brm data begin
    mu_base ~ 1 + (1 | g)
    slope ~ 1 + (1 | g)
    mu = mu_base + slope * x
    y ~ Normal(mu, 1.0)
end
observed(data) = @brm data begin
    mu ~ factor(src; cmc=false)
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
observed_free(data) = @brm data begin
    sigma ~ Exponential(1)
    y ~ Normal(0.0, sigma)
end
beside(data) = @brm data begin
    mu ~ 1 + x + factor(isrc; cmc=false)
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
beside_free(data) = @brm data begin
    mu ~ 1 + x
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
budget(data) = @brm data begin
    mu ~ 1 + x + z + factor(src; cmc=false)
    effect(mu, :) ~ r2d2()
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
budget_free(data) = @brm data begin
    mu ~ 1 + x + z
    effect(mu, :) ~ r2d2()
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
shrunk(data) = @brm data begin
    mu ~ 1 + x + factor(src; cmc=false)
    effect(mu, x) ~ Horseshoe()
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
shrunk_free(data) = @brm data begin
    mu ~ 1 + x
    effect(mu, x) ~ Horseshoe()
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
summand_int(data) = @brm data begin
    mu_base ~ 1 + x
    shift ~ factor(isrc; cmc=false)
    mu = mu_base + shift
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
observed_int(data) = @brm data begin
    mu ~ factor(isrc; cmc=false)
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
end

const SLF = SingleLevelFactor
rk_source(brmi) = sprint(Base.show_unquoted,
    BRM.emit_rk_artifact(brmi; case_id="single_level_factor").ast)

@stestset "single-level factor keeps the ordinary K-1 emission" begin
    for build in (SLF.summand, SLF.observed, SLF.beside)
        one, two = rk_source(build(SLF.BASE)), rk_source(build(SLF.TWO))
        # Only the level list and the coefficient count differ from K=2.
        @test one == replace(two, "[\"a\", \"b\"]" => "[\"a\"]",
            "[1, 2]" => "[1]", "[1:1]" => "[1:0]")
        @test occursin("[1:0] .~ Normal.(0.0, 1.0)", one)
        @test occursin("(vcat(0.0, ", one)
    end
    backend = RKBRMI(SLF.summand(SLF.BASE))
    @test !any(name -> startswith(string(name), "shift"),
        coordinate_names(backend.model.layout))
end

@stestset "single-level factor contributes exactly zero" begin
    for (with, without) in ((SLF.summand, SLF.summand_free),
            (SLF.observed, SLF.observed_free), (SLF.beside, SLF.beside_free),
            (SLF.budget, SLF.budget_free), (SLF.shrunk, SLF.shrunk_free))
        (rk, problem), (rk_free, free) = consumer_problem(with(SLF.BASE)),
            consumer_problem(without(SLF.BASE))
        names = coordinate_names(rk.model.layout)
        @test names == coordinate_names(rk_free.model.layout)
        rng = Xoshiro(11)
        for u in (zeros(length(names)), randn(rng, length(names)), randn(rng, length(names)))
            saved = copy(u)
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            free_value, free_gradient = LogDensityProblems.logdensity_and_gradient(free, u)
            @test value == free_value
            @test gradient ≈ free_gradient atol=1e-12 rtol=1e-12
            @test isequal(u, saved)
        end
    end
end

@stestset "single-level factor matches Stan's vector[0]" begin
    for (label, build) in (("summand", SLF.summand_int), ("observed", SLF.observed_int),
            ("beside", SLF.beside))
        brmi = build(SLF.BASE)
        rk, problem = consumer_problem(brmi)
        stan = consumer_stan(brmi, "single_level_factor_" * label; mod=SLF)
        sb = SBBRMI(brmi; mod=SLF, total_groups=())
        transport = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model))
        @test length(transport) == length(coordinate_names(rk.model.layout))
        for u in (zeros(length(transport)),
                [0.37sin(i) for i in 1:length(transport)])
            stan_u = brm_rk_point_to_stan(transport, u)
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            stan_gradient = similar(stan_u)
            stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
                stan_gradient; propto=false, jacobian=true)
            @test value ≈ stan_value + transport.logdensity_offset atol=2e-11 rtol=2e-11
            @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-10 rtol=2e-10
        end
    end
end
