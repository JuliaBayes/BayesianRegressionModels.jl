# test/rk_batched_cell_flatten.jl — a `@plate` cell's collected value read
# through a position-batched borrowed reader. Emitted RK source flattens the
# cells with an emitted `@kernel` (`BRM._RK_FLATTEN_CELLS_KERNEL`) whose graph
# ReactiveKernels splices in, so it keeps the flat value in a lane slot instead
# of allocating it at every position, as it did behind the opaque
# `brm_flatten_cells` (snag `brm-flatten-cell-0dce6ee7`; RK `207b0e056`,
# reactivekernels-use §4c). Public synthetic models only.
#
# Run: julia --project=test test/rk_batched_cell_flatten.jl
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module FlattenModels
using BayesianRegressionModels, Distributions
vector_cells(d) = @brm d begin
    log_k ~ Normal(0, 1)
    sigma ~ Exponential(1)
    @plate for i in eachindex(t)
        loc[i] = dose[i] .* exp.(-exp(log_k) .* t[i])
    end
    y ~ Normal(loc, sigma)
end
number_cells(d) = @brm d begin
    log_k ~ Normal(0, 1)
    sigma ~ Exponential(1)
    @plate for i in eachindex(t)
        loc[i] = dose[i] * exp(-exp(log_k) * t[i])
    end
    y ~ Normal(loc, sigma)
end
end

module EmittedFlatten
using ReactiveKernels
using BayesianRegressionModels: brm_flatten_cells
end

@stestset "the emitted flatten kernel has the helper's values and types" begin
    defs = Any[]
    call = BRM._rk_flatten_cells!(defs, Set{Symbol}(), :cells)
    Core.eval(EmittedFlatten, only(defs))
    flatten = ReactiveKernels.prepare(getfield(EmittedFlatten, first(call.args)))
    for cells in (Vector{Float64}[], [[1.0, 2.0], Float64[], [3.0]], [2.5], Float64[],
            [[1, 2], [3]], [[0.5]], Any[[1.0], 2.0], Any[[1.0]], Any[2.0])
        want = BRM.brm_flatten_cells(cells)
        got = Base.invokelatest(flatten, copy(cells))
        @test isequal(got, want)
        @test typeof(got) == typeof(want)
    end
end

# Bytes of a warm call: the minimum over repeats, so a GC is not counted.
function bytes_per_call(reader, args...)
    reader(args...); reader(args...)
    minimum(@allocated(reader(args...)) for _ in 1:5)
end

# The cut's reader of `loc` at `P` positions of the unconstrained draw.
function batched_loc(spec, bound, n, P)
    cut = ReactiveKernels.prepare(spec; have=(:unconstrained, keys(bound)...), want=(:loc,),
        bound)
    reader = copy(ReactiveKernels.vectorize(cut; batched=:unconstrained, reuse=true))
    U = [0.1 * sin(j + 3k) for j in 1:n, k in 1:P]
    single(u) = (value = Base.invokelatest(cut, u); value isa Tuple ? only(value) : value)
    stacked = Base.invokelatest(reader, U)
    stacked = stacked isa Tuple ? only(stacked) : stacked
    at(k) = stacked isa AbstractMatrix ? stacked[:, k] : stacked[k]
    values_match = all(isequal(collect(at(k)), single(U[:, k])) for k in 1:P)
    bytes = Base.invokelatest(bytes_per_call, reader, U)
    (; values_match, bytes)
end

# The printed source, parsed back: a replay of printed source reads the same
# slot.
function printed_spec(backend)
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    emitted = BRM._rk_emit_ast(backend.plan)
    retyped = BRM._RKEmittedProgram(
        [Meta.parse(sprint(Base.show_unquoted, d)) for d in emitted.defs],
        Meta.parse(sprint(Base.show_unquoted, emitted.main)), emitted.bindings)
    translated = ext._rk_translate_from_emitted(backend.plan, retyped)
    Base.invokelatest(ReactiveKernelsPPL.build_kernel, translated).spec
end

@stestset "a batched reader of a cell's collected value allocates no flat value per position" begin
    S = 64
    lengths = [150 + 7 * (i % 11) for i in 1:S]           # ragged cells
    rows = sum(lengths)
    flat_bytes = sizeof(Float64) * rows
    t = [collect(range(0.1, 8.0; length=n)) for n in lengths]
    dose = [1.0 + 0.01i for i in 1:S]
    vector_data = (; t, dose, y=[0.3 .* exp.(-0.4 .* ti) for ti in t])
    number_data = (; t=[0.1 + 0.05i for i in 1:S], dose, y=[0.2 + 0.001i for i in 1:S])
    for (label, build, data, flat) in (
            ("vector cells", FlattenModels.vector_cells, vector_data, flat_bytes),
            ("number cells", FlattenModels.number_cells, number_data, sizeof(Float64) * S))
        backend = RKBRMI(build(data))
        main = sprint(Base.show_unquoted, BRM._rk_emit_ast(backend.plan).main)
        @test occursin("loc = flatten_cells(loc_cells)", main)
        n = backend.model.layout.total
        bound = rk_bound_data(backend)
        for (source, spec) in (("emitted", backend.model.spec), ("printed", printed_spec(backend)))
            @testset "$label, $source source" begin
                few, many = batched_loc(spec, bound, n, 8), batched_loc(spec, bound, n, 32)
                @test few.values_match
                @test many.values_match
                println("FLATTEN_BYTES ", label, " ", source, " P=8 ", few.bytes,
                    " P=32 ", many.bytes, " flat=", flat)
                # 24 more positions cost less than one flat value in all.
                @test many.bytes - few.bytes < flat
            end
        end
    end
end
