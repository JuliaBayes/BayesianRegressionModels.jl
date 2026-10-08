using Test, BayesianRegressionModels, Distributions
using ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme
include(joinpath(@__DIR__, "testset_filter.jl"))

module BoundReaderCacheFixture
using BayesianRegressionModels, Distributions
const calls = Ref(0)
counted_positions(kinds) = (calls[] += 1; findall(isone, kinds))

function build(data)
    @brm data begin
        log_rate ~ 1 + (1 | p | subject)
        effect(log_rate, Intercept) ~ Normal(0, 1)
        sd(:, p) ~ Exponential(1)
        sigma ~ Exponential(1)
        loc ~ kernel(log_rate, kinds, picks) do rate, kinds, picks
            positions = counted_positions(kinds)
            selected = positions[picks]
            return exp(rate) .* selected
        end
        y ~ Normal(loc, sigma)
    end
end
end

@stestset "a mixed kernel reader uses a bound data domain" begin
    BRM = BayesianRegressionModels
    fixture = BoundReaderCacheFixture
    data = (; subject=["a", "b"], kinds=[[1, 2, 1], [1, 1, 2]],
        picks=[[1, 2], [2]], y=[[0.1, -0.2], [0.3]])
    saved = deepcopy(data)
    brmi = fixture.build(data)
    emitted = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    source = join((sprint(Base.show_unquoted, d) for d in emitted.defs), "\n")
    @test occursin("Base.eachindex", source)
    @test occursin("Ref(log_rate", source)

    backend = RKBRMI(brmi)
    bound = BRM.rk_translate_artifact(BRM.emit_rk_artifact(brmi;
        case_id="bound-reader-cache"))
    n = length(coordinate_names(backend.model.layout))
    u = zeros(n)
    fixture.calls[] = 0
    sampler = prepare_sampler(backend.model, bound, u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    # The checked-in test pin predates RK's array-valued plate cache. Exercise
    # the cache assertion when that capability is present (including b0f6fc08).
    array_cache = isdefined(ReactiveKernels, :_partial_plate_cacheable)
    if array_cache
        @test fixture.calls[] == length(data.kinds)
        @test !occursin("counted_positions", string(ReactiveKernels.readable_code(sampler.kernel)))
    end

    for point in (u, fill(0.13, n), collect(range(-0.2, 0.3; length=n)))
        gradient = similar(point)
        value, _ = sampler_value_and_gradient!(sampler, gradient, point)
        step = 1e-5
        finite_difference = map(eachindex(point)) do j
            plus, minus = copy(point), copy(point)
            plus[j] += step
            minus[j] -= step
            (sampler(plus) - sampler(minus)) / (2step)
        end
        @test isfinite(value)
        @test all(isfinite, gradient)
        @test gradient ≈ finite_difference atol=2e-7 rtol=2e-7
    end
    if array_cache
        @test fixture.calls[] == length(data.kinds)
    end
    @test isequal(data, saved)
end
