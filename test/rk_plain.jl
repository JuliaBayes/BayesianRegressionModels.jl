# BRM statistical bodies expressed as ordinary RKPPL declarations.
# Run: julia --project=test test/rk_plain.jl [testset substring ...]
using Test, BayesianRegressionModels, Distributions
using ReactiveKernels, ReactiveKernelsPPL
import BayesianRegressionModels: ProbitLink, LogitLink, Cumulative, StoppingRatio, Ordinal
using Enzyme, LogDensityProblems, LinearAlgebra
using DifferentiationInterface: AutoEnzyme
include(joinpath(@__DIR__, "testset_filter.jl"))
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "rk_source_roundtrip.jl"))

function check_printed_roundtrip(backend)
    check_rk_source_roundtrip(backend)
    emitted = BRM._rk_emit_ast(backend.plan)
    sprint(Base.show_unquoted, emitted.main)
end

function check_plain_gradient(backend)
    problem = rk_logdensity_problem(backend;
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    u = fill(0.13, LogDensityProblems.dimension(problem))
    value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
    @test isfinite(value)
    h = 1e-5
    fd = map(eachindex(u)) do j
        plus, minus = copy(u), copy(u)
        plus[j] += h
        minus[j] -= h
        (LogDensityProblems.logdensity(problem, plus) -
            LogDensityProblems.logdensity(problem, minus)) / (2h)
    end
    @test gradient ≈ fd atol=2e-5 rtol=2e-5
end

@stestset "BRM-owned smooth and GP statistical bodies" begin
    data = (; x=collect(range(-1, 1; length=12)),
        z=sin.(collect(range(-2, 2; length=12))), y=fill(0.3, 12))
    for (i, brmi) in enumerate((
        @brm(data, begin
            mu ~ 1 + s(x)
            sigma ~ Exponential(1)
            y ~ Normal(mu, sigma)
        end),
        @brm(data, begin
            mu ~ 1 + t2(x, z; k=(4, 4))
            sigma ~ Exponential(1)
            y ~ Normal(mu, sigma)
        end),
        @brm(data, begin
            mu ~ 1 + hsgp(x; k=4)
            sigma ~ Exponential(1)
            y ~ Normal(mu, sigma)
        end),
        @brm(data, begin
            mu ~ 1 + hsgp(x, z; k=(3, 3), iso=false)
            sigma ~ Exponential(1)
            y ~ Normal(mu, sigma)
        end),
        @brm(data, begin
            mu ~ 1 + hsgp(x; k=3, cov=:periodic, period=2.0)
            sigma ~ Exponential(1)
            y ~ Normal(mu, sigma)
        end),
        @brm(data, begin
            mu ~ 1 + gp(x)
            sigma ~ Exponential(1)
            y ~ Normal(mu, sigma)
        end)))
        @testset "smooth $i" begin
        backend = RKBRMI(brmi)
        source = check_printed_roundtrip(backend)
        @test !occursin("penalized_smooth", source)
        @test !occursin("hsgp_effect", source)
        @test occursin("brm_", source)
        check_plain_gradient(backend)
        end
    end
end

@stestset "flat named coefficient priors and source round trip" begin
    data = (; x=[-0.5, 0.2, 0.8], y=[0.4, -0.1, 0.6])
    backend = RKBRMI(@brm data begin
        mu ~ 1 + x
        effect(mu, Intercept) ~ Normal(0.3, 1.2)
        effect(mu, x) ~ Laplace(-0.1, 0.7)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end)
    emitted = BRM._rk_emit_ast(backend.plan)
    @test isempty(emitted.defs)
    source = check_printed_roundtrip(backend)
    @test occursin("mu_Intercept ~ Normal(0.3, 1.2)", source)
    @test occursin("mu_x ~ Laplace(-0.1, 0.7)", source)
    @test !occursin("popefs", source)
    u = fill(0.13, length(coordinate_names(backend.model.layout)))
    values = constrain(backend.model.layout, u)
    expected = sum(logpdf.(Normal.(values.mu_Intercept .+ values.mu_x .* data.x,
        values.sigma), data.y)) + logpdf(Normal(0.3, 1.2), values.mu_Intercept) +
        logpdf(Laplace(-0.1, 0.7), values.mu_x) +
        logpdf(Exponential(1), values.sigma) + logjac(backend.model.layout, u)
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    plan = ext._rk_translated_plan(backend.plan)
    actual = Base.invokelatest(prepare_query(backend.model, plan, :sampler), u)
    @test actual ≈ expected
end

@stestset "explicit varying shrinkage monotonic and DAR bodies" begin
    data = (; x=[-1.0, -0.4, 0.2, 0.6, 0.9, 1.3],
        z=[0.1, -0.3, 0.6, -0.2, 0.4, 0.8],
        g=["b", "a", "b", "c", "a", "c"],
        c=[1, 2, 3, 2, 1, 3], t=collect(1.0:6.0), y=fill(0.3, 6))
    for (i, brmi) in enumerate((
        @brm(data, begin
            mu ~ 1 + x + (1 + x | g)
            sigma ~ Exponential(1)
            y ~ Normal(mu, sigma)
        end),
        @brm(data, begin
            mu ~ 1 + x + z
            effect(mu, x) ~ Horseshoe()
            effect(mu, z) ~ Horseshoe(local_scale=0.5, global_scale=0.25)
            sigma ~ Exponential(1)
            y ~ Normal(mu, sigma)
        end),
        @brm(data, begin
            mu ~ 1 + x + z
            effect(mu, :) ~ r2d2(R2=Beta(2, 5), alpha=0.5)
            sigma ~ Exponential(1)
            y ~ Normal(mu, sigma)
        end),
        @brm(data, begin
            mu ~ 1 + mo(c)
            sigma ~ Exponential(1)
            y ~ Normal(mu, sigma)
        end),
        @brm(data, begin
            mu ~ 1 + dar(t)
            sigma ~ Exponential(1)
            y ~ Normal(mu, sigma)
        end)))
        @testset "structural $i" begin
            backend = RKBRMI(brmi)
            source = check_printed_roundtrip(backend)
            @test !occursin("varying_draws", source)
            @test !occursin("Horseshoe", source)
            @test !occursin("r2d2(", source)
            @test !occursin("mo(", source)
            @test !occursin("dar(", source)
            check_plain_gradient(backend)
        end
    end
end

@stestset "explicit ordinal and observed-row declarations" begin
    data = (; x=[-1.0, -0.4, 0.2, 0.6, 0.9, 1.3],
        z=[0.1, -0.3, 0.6, -0.2, 0.4, 0.8],
        y=[1, 2, 3, 2, 1, 3])
    for (i, brmi) in enumerate((
        @brm(data, begin
            eta ~ 0 + x
            y ~ OrderedLogistic(eta)
        end),
        @brm(data, begin
            eta ~ 0 + x
            log(disc) ~ 1 + z
            y ~ Ordinal(Cumulative(), ProbitLink(), eta; discrimination=disc)
        end),
        @brm(data, begin
            eta ~ 0 + x
            y ~ Ordinal(StoppingRatio(), LogitLink(), eta; per_threshold=(z,))
        end),
        @brm((; x=data.x, y=Union{Missing,Float64}[0.3, missing, 0.1, missing, 0.4, 0.2]), begin
            mu ~ 1 + x
            sigma ~ Exponential(1)
            mi(y) ~ Normal(mu, sigma)
        end)))
        @testset "ordinal/observed $i" begin
            backend = RKBRMI(brmi)
            source = check_printed_roundtrip(backend)
            @test !isempty(source)
            check_plain_gradient(backend)
        end
    end
end

public_prior_reader(a, b, row) = a[row] .+ b[row]

@stestset "shared random effect explicit scale and correlation priors" begin
    data = (; group=["b", "a"], row=[1, 2, 1], y=[0.2, -0.3, 0.5])
    control = @brm data begin
        a ~ 1 + (1 | shared | group)
        b ~ 1 + (1 | shared | group)
        reads = public_prior_reader(a, b, row)
        y ~ Normal(reads, 1.0)
    end
    explicit = @brm data begin
        a ~ 1 + (1 | shared | group)
        b ~ 1 + (1 | shared | group)
        sd(:, shared) ~ Exponential(0.7)
        cor(:, shared) ~ LKJCholesky(2, 3.0)
        reads = public_prior_reader(a, b, row)
        y ~ Normal(reads, 1.0)
    end
    default_backend, backend = RKBRMI(control), RKBRMI(explicit)
    check_printed_roundtrip(default_backend)
    source = check_printed_roundtrip(backend)
    @test occursin("Exponential.(0.7)", source)
    @test occursin("LKJCholesky(2, 3.0)", source)
    @test coordinate_names(backend.model.layout) == coordinate_names(default_backend.model.layout)
    check_plain_gradient(backend)
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    bound = ext._rk_translated_plan(backend.plan)
    default_bound = ext._rk_translated_plan(default_backend.plan)
    problem = rk_logdensity_problem(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for u in (fill(0.1, 9), collect(range(-0.3, 0.4; length=9)))
        saved = copy(u)
        nt = constrain(backend.model.layout, u)
        sd, z, L = nt.ranef_draws_shared_group.sd, nt.ranef_draws_shared_group.z, nt.ranef_draws_shared_group.L
        B = z * (Diagonal(sd) * L)'
        a = nt.a_Intercept .+ B[[2, 1], 1]
        b = nt.b_Intercept .+ B[[2, 1], 2]
        mu = [a[data.row[i]] + b[data.row[i]] for i in eachindex(data.y)]
        lk(eta) = logpdf(LKJCholesky(2, eta), LinearAlgebra.Cholesky(LinearAlgebra.LowerTriangular(L)))
        expected_prior = logpdf(Normal(), nt.a_Intercept) + logpdf(Normal(), nt.b_Intercept) +
            sum(logpdf.(Normal(), z)) + sum(logpdf.(Exponential(0.7), sd)) + lk(3)
        expected_ll = sum(logpdf.(Normal.(mu, 1), data.y))
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test value ≈ expected_prior + expected_ll + logjac(backend.model.layout, u)
        @test isequal(u, saved)
        @test all(isfinite, gradient)
        delta = sum(logpdf.(Exponential(0.7), sd) .- logpdf.(Normal(), sd)) + lk(3) - lk(1)
        @test Base.invokelatest(prepare_query(backend.model, bound, :sampler), u) -
            Base.invokelatest(prepare_query(default_backend.model, default_bound, :sampler), u) ≈ delta
    end
end

@stestset "ordinary indexed panel kernel source" begin
    for data in ((; x=[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]],
                    y=[[0.3, 0.1], [0.2, 0.4], [0.1, 0.5]]),
                 (; x=[0.1, 0.2, 0.3], y=[0.2, 0.4, 0.5]))
        model = @brm data begin
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            sigma ~ Exponential(1)
            pred ~ kernel(x, y) do xi, yi
                mu = a .+ b .* xi
                yi ~ Normal(mu, sigma)
                mu
            end
        end
        backend = RKBRMI(model)
        source = check_printed_roundtrip(backend)
        @test occursin("@plate", source)
        @test !occursin("subjects=", source)
        check_plain_gradient(backend)
    end
end
