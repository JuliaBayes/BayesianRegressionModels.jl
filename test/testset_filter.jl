# test/testset_filter.jl — substring subset contract for heavy standalone test files.
#
# Included by heavy `test/*.jl` scripts (currently `rk_parity.jl` and
# `rk_emitter.jl`) so lanes can run them in fresh-process chunks instead of one
# OOM-prone process. Usage in the including file, after `using Test`:
#
#     include(joinpath(@__DIR__, "testset_filter.jl"))
#
# and spell chunkable blocks `@stestset "name" begin ... end` instead of
# `@testset`. With no filter active `@stestset` is exactly `@testset` (plus a
# run counter); with a filter active only matching blocks run:
#
#     julia --project=test test/rk_emitter.jl "group-C" "fail closed: group-C"
#     BRM_TEST_FILTER="group-C,fail closed" julia --project=test test/rk_emitter.jl
#
# Contract:
# - Filters are ARGS entries plus comma-separated `BRM_TEST_FILTER` entries
#   (union; surrounding whitespace stripped; empties dropped).
# - A block runs when any filter is a substring of its name (`occursin`).
# - Empty filter set runs everything (default; CI/full runs unchanged).
# - A non-empty filter that matches NOTHING exits 1 at `atexit` time: a chunk
#   that silently runs zero tests would read as a green verification.
#
# Notes:
# - Testset names must be string literals (as in every current call site): the
#   name expression is evaluated twice (once for the match, once by `@testset`).
# - Standalone-script use only: the file is `include`d once per process. It
#   defines `const _STESTSET_FILTERS`, so a second `include` in one process
#   warns on the redefinition (and re-registers the `atexit` hook).

function _stestset_env_filters()
    raw = get(ENV, "BRM_TEST_FILTER", "")
    isempty(strip(raw)) && return String[]
    return [s for s in (strip(p) for p in split(raw, ",")) if !isempty(s)]
end

const _STESTSET_FILTERS = vcat(String.(ARGS), _stestset_env_filters())

const _stestset_count = Ref(0)

_stestset_selected(name::AbstractString) =
    isempty(_STESTSET_FILTERS) || any(f -> occursin(f, name), _STESTSET_FILTERS)

macro stestset(name, rest...)
    esc(quote
        if _stestset_selected($name)
            _stestset_count[] += 1
            @testset $name $(rest...)
        end
    end)
end

if !isempty(_STESTSET_FILTERS)
    println(
        "test filter active (",
        join(("$(repr(f))" for f in _STESTSET_FILTERS), ", "),
        "): only matching @stestset blocks run; ",
        "a filter matching nothing exits 1.",
    )
    atexit() do
        if _stestset_count[] == 0
            @error "test filter matched zero testsets — check the filter spelling" filters =
                _STESTSET_FILTERS
            exit(1)
        end
    end
end
