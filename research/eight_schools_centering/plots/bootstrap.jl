# Bootstrap the eight-schools plots environment.
#
#     julia --startup-file=no --project=test \
#       research/eight_schools_centering/plots/bootstrap.jl [PLOTSENV]
#
# `PLOTSENV` defaults to this directory. Every unregistered dependency is
# materialized at an exact pinned commit below the shared ignored
# `test/.bootstrap/` cache through the canonical resolver
# (`resolve_git_revision_checkout` in `test/dependency_floors.jl`), then all
# local checkouts enter ONE batched `Pkg.develop` call — the same shape as
# `test/setup_env.jl`, which stays the authority for the six shared pins.
#
# The six test-env pins are mirrored below so this entrypoint works
# standalone. They are cross-checked against `test/setup_env.jl` at runtime:
# a bump made there without syncing here (or vice versa) fails closed
# instead of resolving a drifted plots environment.
#
# Idempotent: re-running reuses the exact cached checkouts.
using Pkg

const PLOTSDIR = @__DIR__
const REPO = dirname(dirname(dirname(PLOTSDIR)))
const TESTENV = joinpath(REPO, "test")
const PLOTSENV = length(ARGS) >= 1 ? ARGS[1] : PLOTSDIR

include(joinpath(TESTENV, "dependency_floors.jl"))

const GITHUB = "https://github.com/nsiccha"

# name => (github url, pinned commit). Comments record the branch the commit
# was measured on; the pin itself is the full SHA and needs no branch.
const TEST_PINS_MIRROR = [
    ("MutatingFunctions", "$GITHUB/MutatingFunctions.jl.git", "4fc41b1c7b774133ceaacc4ff3c34c67b15b87b2"),  # main
    ("OutputSignatures", "$GITHUB/OutputSignatures.jl.git", "121de3194f02044e00bac0d11019a93458ddb63a"),  # main
    ("TreeArrays", "$GITHUB/TreeArrays.jl.git", "c317cc003fc41c2d933c27dc80799141eebd434e"),  # main
    ("StanBlocks", "$GITHUB/StanBlocks.jl.git", "bec23bc3c52303ebde60a026af48c435e4c81330"),  # devibe
    ("Treebars", "$GITHUB/Treebars.jl.git", "c02aa16ab1b08e4f5283597fe678a88e69555cd1"),  # dev
    ("WarmupHMC", "$GITHUB/WarmupHMC.jl.git", "deeea1d128d5235ad0ecb2fd911a6d881f1ac2c2"),  # dev
]

# Plotting-stack pins owned by this entrypoint: the exact working set that
# rendered the committed eight-schools figures (all verified loaded/rendered;
# DynamicObjects 3352e033 and Treebars c02aa16 are the inbox-confirmed
# source pins, not the unregistered 0.5.0 line).
const PLOT_STACK_PINS = [
    ("AlgebraOfVega", "$GITHUB/AlgebraOfVega.jl.git", "412422660d859a468b8357a11527ff0033e0881c"),  # kb-extdep
    ("DynamicObjects", "$GITHUB/DynamicObjects.jl.git", "3352e03368244c5f65280339fe036507818dff72"),  # kb-extdep
    ("HTMXObjects", "$GITHUB/HTMXObjects.jl.git", "a813640165d14cf9cb87502ecf632f0c42378a69"),  # kb-extdep
    ("HTMX", "$GITHUB/HTMX.jl.git", "d52ce5be0f42e3c375370c85499ba5c395f781a8"),  # kb-extdep
]

function read_test_pins()
    pins = Dict{String,String}()
    for line in eachline(joinpath(TESTENV, "setup_env.jl"))
        m = match(r"^\(\"([A-Za-z]+)\",\s*\"([^\"]+)\",\s*\"([0-9a-f]{40})\"\)", strip(line))
        m === nothing && continue
        pins[m.captures[1]] = m.captures[3]
    end
    return pins
end

function main()
    authoritative = read_test_pins()
    for (name, _url, rev) in TEST_PINS_MIRROR
        get(authoritative, name, nothing) == rev || error(
            "plots bootstrap mirror for $name ($rev) disagrees with " *
            "test/setup_env.jl ($(get(authoritative, name, "absent"))). " *
            "Bump both together; refusing to resolve a drifted plots env.")
    end

    cache_root = joinpath(TESTENV, ".bootstrap")
    paths = Dict("BayesianRegressionModels" => REPO)
    for (name, url, rev) in vcat(TEST_PINS_MIRROR, PLOT_STACK_PINS)
        paths[name] = resolve_git_revision_checkout(
            name, rev; cache_root, origin=url)
    end

    Pkg.activate(PLOTSENV)
    Pkg.develop(PackageSpec[
        PackageSpec(path=path) for (_name, path) in sort!(collect(paths))
    ])
    Pkg.instantiate(; allow_autoprecomp=false)
    Pkg.precompile()
    println("eight_schools_plots_bootstrap_complete\t", PLOTSENV)
    return nothing
end

main()
