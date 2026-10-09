using Test, BayesianRegressionModels, Distributions
using ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme
include(joinpath(@__DIR__, "testset_filter.jl"))

module BoundReaderCacheFixture
using BayesianRegressionModels, Distributions, ReactiveKernels
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

# A recursion over each subject's ragged event schedule: kind 2 adds an input,
# kind 1 reads the decayed state. Three live per-subject predictors enter the
# child kernel; the reads each observation names are a data-only index chain.
@kernel event_subject(observed_reads, kinds, gaps, amounts, log_k, log_v, log_a) = begin
    k = exp(log_k)
    v = exp(log_v)
    a = exp(log_a)
    reads = scan(kinds, gaps, amounts, Ref(k), Ref(v), Ref(a); init=0.0) do previous, kind, gap, amount, kk, vv, aa
        decayed = previous * exp(-kk * gap)
        read = kind == 1 ? decayed / vv : 0.0
        next = kind == 1 ? decayed : decayed + aa * amount
        (next, read)
    end
    read_events = counted_positions(kinds)
    observed = read_events[observed_reads]
    locations = reads[observed]
    return locations
end

function build_events(data)
    @brm data begin
        log_k ~ 1 + (1 | p | subject)
        log_v ~ 1 + (1 | p | subject)
        log_a ~ 1 + (1 | p | subject)
        effect(log_k, Intercept) ~ Normal(0, 1)
        effect(log_v, Intercept) ~ Normal(0, 1)
        effect(log_a, Intercept) ~ Normal(0, 1)
        sd(:, p) ~ Exponential(1)
        sigma ~ Exponential(1)
        loc ~ kernel(log_k, log_v, log_a, observed_reads, kinds, gaps, amounts) do lk, lv, la, observed_reads, kinds, gaps, amounts
            return event_subject(observed_reads, kinds, gaps, amounts, lk, lv, la)
        end
        y ~ Normal(loc, sigma)
    end
end
end

function reader_source(brmi)
    emitted = BayesianRegressionModels._rk_emit_ast(BayesianRegressionModels._brm_rk_plan(brmi))
    join((sprint(Base.show_unquoted, d) for d in emitted.defs), "\n")
end

# Value and gradient against central differences at several points; the
# data-only index work runs once per subject at preparation when RK caches it.
function check_bound_reader(brmi, data, case_id)
    fixture = BoundReaderCacheFixture
    BRM = BayesianRegressionModels
    backend = RKBRMI(brmi)
    bound = BRM.rk_translate_artifact(BRM.emit_rk_artifact(brmi; case_id))
    n = length(coordinate_names(backend.model.layout))
    u = zeros(n)
    fixture.calls[] = 0
    sampler = prepare_sampler(backend.model, bound, u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    # The checked-in test pin predates RK's array-valued plate cache. Exercise
    # the cache assertion when that capability is present (including b0f6fc08).
    array_cache = isdefined(ReactiveKernels, :_partial_plate_cacheable)
    if array_cache
        @test fixture.calls[] == length(data.subject)
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
        @test fixture.calls[] == length(data.subject)
    end
end

# A predictor is a zipped plate operand whose declared rank lets the bound
# data operands fix the subject domain (snag brm-reader-bound-3a64804a). The
# earlier bound-axis reader instead indexed `Ref`'d predictors by an
# `eachindex` subject, which cost reverse passes more than the zip and broke
# native Reverse on one consumer model.
@stestset "a mixed kernel reader uses a bound data domain" begin
    data = (; subject=["a", "b"], kinds=[[1, 2, 1], [1, 1, 2]],
        picks=[[1, 2], [2]], y=[[0.1, -0.2], [0.3]])
    saved = deepcopy(data)
    brmi = BoundReaderCacheFixture.build(data)
    source = reader_source(brmi)
    @test occursin("loc_reader(log_rate::AbstractVector{Float64}, kinds, picks)", source)
    @test occursin("ReactiveKernels.plate(log_rate, kinds, picks) do rate, kinds, picks", source)
    for retired in ("Base.eachindex", "Ref(log_rate", "_subject")
        @test !occursin(retired, source)
    end
    check_bound_reader(brmi, data, "bound-reader-cache")
    @test isequal(data, saved)
end

@stestset "ragged event schedules with several live predictors keep exact gradients" begin
    groups = 3
    data = (; subject=string.(1:groups),
        kinds=[[2, 1, 1, 2, 1, 1, 1], [2, 1, 2, 1], [2, 2, 1, 1, 1]],
        gaps=[[0.0, 0.5, 1.0, 0.2, 0.7, 1.5, 2.0], [0.0, 0.4, 0.3, 1.1], [0.0, 0.6, 0.2, 0.9, 1.3]],
        amounts=[[1.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0], [1.5, 0.0, 0.5, 0.0], [0.7, 0.9, 0.0, 0.0, 0.0]],
        observed_reads=[[1, 2, 4, 5], [2], [1, 3]],
        y=[[0.3, 0.2, 0.4, 0.3], [0.5], [0.6, 0.3]])
    saved = deepcopy(data)
    brmi = BoundReaderCacheFixture.build_events(data)
    source = reader_source(brmi)
    @test occursin("log_k::AbstractVector{Float64}", source)
    @test occursin("log_v::AbstractVector{Float64}", source)
    @test occursin("log_a::AbstractVector{Float64}", source)
    @test !occursin("Base.eachindex", source)
    check_bound_reader(brmi, data, "bound-reader-events")
    @test isequal(data, saved)
end
