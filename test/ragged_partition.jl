# A ragged join's row partition — each cell's rows of the joined axis, by
# label — is computed in the RK graph on every call of a reader whose design is
# a runtime input (snag `rk-reader-recomp-638f7e98`). It finds each row's cell
# once, through a dense table for machine-integer labels over a compact range
# or a hash table keyed by the labels' element type, and sizes every cell
# before filling it. Oracle: the original partition, a `Dict{Any,Int}` lookup
# and `push!` per row, on every label kind; the same refusals.
#
# The RK design binder validates each design's joins without building their
# partitions (snag `rk-design-binder-9d6b7561`): `_brm_check_ragged_rows`
# refuses exactly what `brm_ragged_rows` refuses, with the same message.
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

outcome(f) = try f(); :ok catch e; sprint(showerror, e) end

@testset "a join is validated as its partition validates it, without building it" begin
    rng = Xoshiro(9)
    # Every row's label names a cell exactly when its position is nonzero.
    for cells in (collect(1:7), [1, 10^9, 7], ["s1", "s2"], Any[1, "a"], Int[]),
            ev in (isempty(cells) ? Int[] : cells[rand(rng, eachindex(cells), 30)],
                [cells; 42], Any["zz"], Int[])
        @test BRM._brm_labels_name_cells(ev, cells) ==
            all(!iszero, BRM._brm_label_positions(ev, cells))
    end
    # brm_ragged_rows(names, groups, cells, axes...) against the binder's check,
    # on accepted joins and on each refusal.
    joins = [
        ([3, 1, 3, 2], [1, 2, 3], ([0.1, 0.2, 0.3, 0.4],)),          # accepted
        (["b", "a", "b"], ["a", "b"], ()),                            # accepted, no axis
        (Int[], [1, 2], (Float64[],)),                                # zero-row axis
        ([1, 42, 42], [1, 2], ([0.1, 0.2, 0.3],)),                    # unknown label
        (["a", "zz"], ["a"], ([0.1, 0.2],)),
        ([1, missing], [1, 2], ([0.1, 0.2],)),                        # missing label
        ([1, 2, 1], [1, 2], ([0.1, 0.2],)),                           # axis length
        ([1, 2], [1, 1], ([0.1, 0.2],)),                              # repeated cell label
        ([1, 2], [1, missing], ([0.1, 0.2],)),                        # missing cell label
    ]
    for (groups, cells, axes) in joins
        names = [:x, :g, :c, (:a for _ in axes)...]
        built = outcome(() -> BRM.brm_ragged_rows(names, groups, cells, axes...))
        @test built == outcome(() -> BRM._brm_check_ragged_rows(:x, :g, groups, :c, cells,
            names[4:end], axes))
        # The ordered join, with an order column of the right length, too short,
        # and holding a missing value.
        length(groups) < 2 && continue
        for times in ([reverse(eachindex(groups))...], Int[9], [missing; ones(length(groups) - 1)])
            ordered = outcome(() -> BRM.brm_ragged_rows_ordered([:x, :g, :c, :t, names[4:end]...],
                groups, cells, times, axes...))
            @test ordered == outcome(() -> BRM._brm_check_ragged_rows(:x, :g, groups, :c, cells,
                names[4:end], axes; order=:t, times))
        end
    end
    @test outcome(() -> BRM._brm_check_ragged_rows(:x, :g, [1, 42], :c, [1], [], ())) ==
        "RK backend: kernel(...) `ragged(x, g)`: label(s) Any[42] in `g` name no subject " *
        "in the kernel's per-subject frame. Every event row must belong to a subject this " *
        "kernel walks."
    # The check builds no partition: on a 24,000-row join it allocates less
    # than one integer per row (the cell labels and a lookup table).
    for (groups, cells) in ((fill(7, 24_000), [7]), (repeat(1:400; inner=60), collect(1:400)),
            (repeat(["n$(j)" for j in 1:400]; inner=60), ["n$(j)" for j in 1:400]))
        axis = (zeros(length(groups)),)
        check() = BRM._brm_check_ragged_rows(:x, :g, groups, :c, cells, [:a], axis)
        build() = BRM.brm_ragged_rows([:x, :g, :c, :a], groups, cells, axis...)
        check(); build()
        @test @allocated(check()) < 8 * length(groups) < @allocated(build())
    end
end
