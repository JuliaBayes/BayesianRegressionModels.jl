# A ragged join's row partition — each cell's rows of the joined axis, by
# label — is computed in the RK graph on every call of a reader whose design is
# a runtime input (snag `rk-reader-recomp-638f7e98`). It finds each row's cell
# once, through a dense table for machine-integer labels over a compact range
# or a hash table keyed by the labels' element type, and sizes every cell
# before filling it. Oracle: the original partition, a `Dict{Any,Int}` lookup
# and `push!` per row, on every label kind; the same refusals.
#
# Run: julia --startup-file=no --project=test test/ragged_partition.jl
using Test
using BayesianRegressionModels
using CategoricalArrays
using Random: Xoshiro, shuffle
const BRM = BayesianRegressionModels

function reference_partition(ev_vals, subject_vals)
    pos = Dict{Any,Int}()
    for (i, v) in enumerate(subject_vals); pos[v] = i; end
    rows = [Int[] for _ in eachindex(subject_vals)]
    for (r, v) in enumerate(ev_vals)
        i = get(pos, v, 0)
        i == 0 && error("unknown label $v")
        push!(rows[i], r)
    end
    rows
end
partition(ev, labels) = BRM._brm_kernel_ragged_partition(:x, :g, ev, labels)

@testset "the partition of every label kind is the reference partition" begin
    rng = Xoshiro(20261022)
    labels = [
        collect(1:7),                              # dense integer table
        collect(101:108),                          # dense, offset
        shuffle(rng, collect(-3:3)),               # dense, negative, unordered
        Int32[5, 1, 9],                            # dense, narrow type
        [1, 10^9, 7],                              # sparse range: hash table
        [typemin(Int), 0, typemax(Int)],           # extreme range: hash table
        ["s1", "s2", "s10"],                       # strings
        [0.5, 1.5, -2.0],                          # floats
        Any[1, "a", :b],                           # mixed
    ]
    for cells in labels
        # Interleaved rows, and rows grouped in runs (each run reuses its lookup).
        for ev in (cells[rand(rng, eachindex(cells), 200)],
                cells[repeat(rand(rng, eachindex(cells), 20); inner=5)])
            @test partition(ev, cells) == reference_partition(ev, cells)
            @test partition(ev, cells) isa Vector{Vector{Int}}
        end
    end
    # Rows matched across numeric types by `isequal`, as the hash table does.
    @test partition([1.0, 3.0, 1.0], [1, 3]) == [[1, 3], [2]]
    @test partition(UInt8[3, 1, 3], [1, 2, 3]) == [[2], Int[], [1, 3]]
    @test partition([3, 1], UInt16[1, 3]) == [[2], [1]]
    # Categorical labels, unwrapped by the graph's own entry point.
    raw = categorical(["b", "a", "b", "c"])
    @test BRM.brm_ragged_rows([:x, :g, :c], raw, ["a", "b", "c"]) == [[2], [1, 3], [4]]
    # A zero-row axis, and cells with no rows.
    @test partition(Int[], [1, 2]) == [Int[], Int[]]
    @test partition([2, 2], [1, 2, 3]) == [Int[], [1, 2], Int[]]
    @test partition(Int[], Int[]) == Vector{Int}[]
    # A repeated label keeps its last cell, as before.
    @test partition([4, 4], [4, 9, 4]) == reference_partition([4, 4], [4, 9, 4])
end

@testset "an unknown or missing label is refused as before" begin
    @test_throws "label(s) Any[42] in `g` name no subject" partition([1, 42, 42], [1, 2])
    @test_throws "label(s) Any[\"zz\"] in `g` name no subject" partition(["a", "zz"], ["a"])
    @test_throws "label(s) Any[1] in `g` name no subject" partition([1], Int[])
    @test_throws "must be a column with no missing labels" partition([1, missing], [1])
    @test_throws "must be a column with no missing labels" partition(Any[missing], [1])
end
