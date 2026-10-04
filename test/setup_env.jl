# Rebuild the test environment.
#
#     julia --project=test test/setup_env.jl
#
# The test env has seven external UNREGISTERED dependencies plus the unregistered
# BRM root itself. Each external package is materialized at a specific GitHub
# COMMIT under the ignored `test/.bootstrap/` cache; there is NO dependence on
# any shared `~/github/nsiccha/<pkg>` checkout. A full-SHA revision is
# branch-independent, so it does not matter that several of these live on a
# `dev`/`devibe` branch rather than `main`; the commit only has to be pushed to
# GitHub, which every pin below is.
#
# ReactiveKernels contributes four developed paths from ONE pinned checkout
# (the monorepo root plus the nested `ReactiveKernelsDistributionKernels`,
# `ReactiveKernelsPPL`, and `ReactiveKernelsPPLExamples` packages, which have
# no standalone repos).
#
# All eleven paths enter ONE `Pkg.develop` call on EVERY Julia version we run.
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
    ("StanBlocks",        "https://github.com/nsiccha/StanBlocks.jl.git",        "65cabefda5a9345993c7e0343739e9f8e363611f"),  # devibe: published caller-dimension substitution for nested ragged helpers; retains mixed derived-record and empty-index fixes
    ("Treebars",          "https://github.com/nsiccha/Treebars.jl.git",          "c02aa16ab1b08e4f5283597fe678a88e69555cd1"),  # dev
    # 0194dce (2026-09-27, dev): WindowSelectionPlan (WarmupHMC-held evidence)
    # and the controls interface for custom reparametrizers,
    # which the per-window S2Z Fisher rule needs. Contains 7aed40b (active-state
    # preservation, nonfinite-transport rejection). Matches test/Project.toml.
    ("WarmupHMC",         "https://github.com/nsiccha/WarmupHMC.jl.git",         "0194dce08e986ff17fd5a788bb315c6431b7858e"),  # dev (contains exact sampling-counter floor 913da79)
    # Current plain-function/array PPL surface, with data-only declaration inputs
    # retained before predictor inlining (computed membership axes; 606d76d0),
    # plus published live matrices (1t1v8zo) and callable array cells (0mr4zu5).
    ("ReactiveKernels",   "https://github.com/nsiccha/ReactiveKernels.jl.git",   "d36a86c761dd9b4e9dd3cc360acfe62770c32697"),  # main: transparent module-resolved PPL graph composition; 03m2v2y
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
    # Nested monorepo packages develop from the pinned ReactiveKernels
    # checkout (same revision, no separate pins).
    rk_root = paths["ReactiveKernels"]
    paths["ReactiveKernelsDistributionKernels"] =
        joinpath(rk_root, "packages", "ReactiveKernelsDistributionKernels")
    paths["ReactiveKernelsPPL"] =
        joinpath(rk_root, "packages", "ReactiveKernelsPPL")
    # The v1 SB-parity sweep (test/sb_sweep_*.jl) consumes the inventory
    # models' exact data bindings from here — no transcription.
    paths["ReactiveKernelsPPLExamples"] =
        joinpath(rk_root, "packages", "ReactiveKernelsPPLExamples")

    Pkg.activate(TESTENV)
    Pkg.develop(PackageSpec[
        PackageSpec(path=path) for (_name, path) in sort!(collect(paths))
    ])
    # `Pkg.instantiate()` auto-precompiles the whole manifest in parallel, which
    # self-deadlocks under Pkg 1.10 on Pathfinder 0.10.7's sibling Turing
    # extensions (see test/README.md, "Pathfinder's Turing extension pair").
    # Serialize just that pair first, then let the parallel pass reuse the cache.
    Pkg.instantiate(; allow_autoprecomp=false)
    withenv("JULIA_NUM_PRECOMPILE_TASKS" => "1") do
        Pkg.precompile(["Pathfinder", "Turing"])
    end
    Pkg.precompile()
    return nothing
end

main()
