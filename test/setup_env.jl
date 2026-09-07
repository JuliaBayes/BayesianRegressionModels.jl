# Rebuild the test environment.
#
#     julia --project=test test/setup_env.jl
#
# The test env has six external UNREGISTERED dependencies plus the unregistered
# BRM root itself. Each external package is materialized at a specific GitHub
# COMMIT under the ignored `test/.bootstrap/` cache; there is NO dependence on
# any shared `~/github/nsiccha/<pkg>` checkout. A full-SHA revision is
# branch-independent, so it does not matter that several of these live on a
# `dev`/`devibe` branch rather than `main`; the commit only has to be pushed to
# GitHub, which every pin below is.
#
# All seven paths enter ONE `Pkg.develop` call on EVERY Julia version we run.
# On 1.11+ the `[sources]` blocks in `test/Project.toml` (which mirror these
# revisions) would also resolve them; on **1.10, which is what this suite runs
# on, `[sources]` is IGNORED**, so a bare `Pkg.resolve()` fails with
#
#     ERROR: expected package `TreeArrays [5daaa025]` to be registered
#
# an error that looks like a missing registry rather than a version-gated
# feature. This script is the version-independent answer.
#
# Bumping a pin is a deliberate one-line edit here (+ the matching `[sources]`
# rev in test/Project.toml), reviewed like any other change — not a silent
# consequence of whatever a shared checkout drifted to.
#
# It is idempotent — re-running it validates/reuses the exact cached checkouts —
# so it is safe to run before any suite when unsure of the env's state.
using Pkg

include("dependency_floors.jl")

const REPO = dirname(@__DIR__)
const TESTENV = @__DIR__

# name => (github url, pinned commit)   — comment records the branch the commit
# is on, for humans; the pin itself is the full SHA and needs no branch.
const PINS = [
    # MutatingFunctions carries the SubArray-gather activity fix the julianic
    # 0-alloc surface depends on (main). Pinned at/after b353559 (2026-08-13),
    # where the LinearAlgebra/Random/Statistics integrations became strong
    # `[deps]` instead of package extensions. The pre-b353559 ext trio
    # (MutatingFunctions{LinearAlgebra,Random,Statistics}Ext) SELF-DEADLOCKS a
    # fresh parallel precompile under Pkg 1.10, so this pin must never regress
    # below b353559.
    ("MutatingFunctions", "https://github.com/nsiccha/MutatingFunctions.jl.git", "4fc41b1c7b774133ceaacc4ff3c34c67b15b87b2"),  # main
    ("OutputSignatures",  "https://github.com/nsiccha/OutputSignatures.jl.git",  "121de3194f02044e00bac0d11019a93458ddb63a"),  # main
    ("TreeArrays",        "https://github.com/nsiccha/TreeArrays.jl.git",        "c317cc003fc41c2d933c27dc80799141eebd434e"),  # main
    ("StanBlocks",        "https://github.com/nsiccha/StanBlocks.jl.git",        "bec23bc3c52303ebde60a026af48c435e4c81330"),  # devibe (registers log_modified_bessel_first_kind for the periodic hsgp basis; contains 66448de, the prior-predictive fix)
    ("Treebars",          "https://github.com/nsiccha/Treebars.jl.git",          "c02aa16ab1b08e4f5283597fe678a88e69555cd1"),  # dev
    ("WarmupHMC",         "https://github.com/nsiccha/WarmupHMC.jl.git",         "8fa829b39d6f519deaf13cd03ee17e8df6b7b9c2"),  # dev
]

function main()
    paths = Dict("BayesianRegressionModels" => REPO)
    for (name, url, rev) in PINS
        paths[name] = resolve_git_revision_checkout(
            name,
            rev;
            cache_root=joinpath(TESTENV, ".bootstrap"),
            origin=url,
        )
    end

    Pkg.activate(TESTENV)
    Pkg.develop(PackageSpec[
        PackageSpec(path=path) for (_name, path) in sort!(collect(paths))
    ])
    Pkg.instantiate()
    return nothing
end

main()
