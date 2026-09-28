# test/testset_filter_check.jl — self-check for test/testset_filter.jl.
#
# Run with no args (stdlib only; three short `julia` subprocesses):
#
#     julia --project=test test/testset_filter_check.jl
#
# The parent process checks the no-filter contract in-process, then spawns
# itself in `--check-subprocess <filter>...` mode (plus `BRM_TEST_FILTER=...`
# for the env case) to prove subset selection and the zero-match exit-1 guard
# in fresh processes — the same shape lanes use to chunk heavy suites.

using Test

const _CHECK_HELPER = joinpath(@__DIR__, "testset_filter.jl")

const _CHECK_SUBPROCESS = "--check-subprocess" in ARGS
if _CHECK_SUBPROCESS
    filter!(a -> a != "--check-subprocess", ARGS)
end

# Separate top-level statement: the `@stestset` uses below lower only after
# this `include` has executed and defined the macro.
include(_CHECK_HELPER)

if _CHECK_SUBPROCESS
    @stestset "filter-probe apple" begin
        @test true
    end
    @stestset "filter-probe orange" begin
        @test true
    end
    exit(0)
end

if !_CHECK_SUBPROCESS && !isempty(ARGS)
    println(stderr, "usage: julia --project=test test/testset_filter_check.jl (no args)")
    exit(2)
end

@testset "no-filter contract (in-process)" begin
    @test isempty(_STESTSET_FILTERS)
    @test _stestset_selected("anything at all")
    @test _stestset_selected("")
end

function _check_run(args::Vector{String}; env::Pair{String,String} = "" => "")
    exe = joinpath(Sys.BINDIR, "julia")
    self = joinpath(@__DIR__, "testset_filter_check.jl")
    cmd = `$exe --startup-file=no $self --check-subprocess $args`
    out = IOBuffer()
    ok = if isempty(first(env))
        success(pipeline(cmd; stdout = out, stderr = out))
    else
        withenv(first(env) => last(env)) do
            success(pipeline(cmd; stdout = out, stderr = out))
        end
    end
    return ok, String(take!(out))
end

@testset "ARGS subset selects in a fresh process" begin
    ok, s = _check_run(["apple"])
    @test ok
    @test occursin("test filter active", s)
    @test occursin("filter-probe apple", s)
    @test !occursin("filter-probe orange", s)
end

@testset "env filter selects in a fresh process" begin
    ok, s = _check_run(String[]; env = "BRM_TEST_FILTER" => "orange")
    @test ok
    @test occursin("test filter active", s)
    @test occursin("filter-probe orange", s)
    @test !occursin("filter-probe apple", s)
end

@testset "zero-match filter exits 1" begin
    ok, s = _check_run(["zzz-no-such-testset"])
    @test !ok
    @test occursin("matched zero testsets", s)
end

@testset "ARGS and env filters union" begin
    ok, s = _check_run(["apple"]; env = "BRM_TEST_FILTER" => "orange")
    @test ok
    @test occursin("filter-probe apple", s)
    @test occursin("filter-probe orange", s)
end
