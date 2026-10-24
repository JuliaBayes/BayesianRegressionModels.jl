# Rebuild the test environment.
#
#     julia --project=test test/setup_env.jl
#
# The test env has five external UNREGISTERED dependencies plus the unregistered
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
# All nine paths enter ONE `Pkg.develop` call on EVERY Julia version we run.
# On 1.11+ the `[sources]` blocks in `test/Project.toml` (which mirror these
# revisions) would also resolve them; on **1.10, which is what this suite runs
# on, `[sources]` is IGNORED**, so a bare `Pkg.resolve()` fails with
#
#     ERROR: expected package `Treebars [e1e568c4]` to be registered
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
    ("StanBlocks",        "https://github.com/nsiccha/StanBlocks.jl.git",        "c7293b884363cf4d84da43bba7a775696ad81c70"),  # devibe: Stan's vectorised builtins (`fmax`, `fmin`, …) evaluate elementwise in the Julia layer, so a per-axis `fmax(0.0, lo)` bound reads one value per coordinate (c7293b88); descriptor inputs/outputs carry `constraint_values`, their bounds evaluated against the bound data (b233afbc); a `ValueFamily` as the base family of `weighted`/`truncated`/`censored`/`interval_censored` with `lcdf=`/`lccdf=` companions (94e414bb); descriptor outputs carry their value shape as `dims`; non-finite literals render as Stan infinity builtins; `weighted` over censored/truncated/interval_censored; an exchangeable ordered prior re-draws as the sorted iid draw; observed scans and per-cell scan lengths (82a954da), also through a do-cell parameter that shadows its data name (4ee6e0e8); integer step locals in nested scans (2f5b6203); a latent or observed `@scan` inside a called `@slic` submodel (e67bc314), observing a plate cell's group (d3483f11); `interval_censored`'s interval by keyword and unknown combinator keywords refused (897557c); typed `=` in `stan_descriptor` (204e5a7)
    ("Treebars",          "https://github.com/nsiccha/Treebars.jl.git",          "c02aa16ab1b08e4f5283597fe678a88e69555cd1"),  # dev
    # 0194dce (2026-09-27, dev): WindowSelectionPlan (WarmupHMC-held evidence)
    # and the controls interface for custom reparametrizers,
    # which the per-window S2Z Fisher rule needs. Contains 7aed40b (active-state
    # preservation, nonfinite-transport rejection). Matches test/Project.toml.
    ("WarmupHMC",         "https://github.com/nsiccha/WarmupHMC.jl.git",         "0194dce08e986ff17fd5a788bb315c6431b7858e"),  # dev (contains exact sampling-counter floor 913da79)
    # Current plain-function/array PPL surface, with data-only declaration inputs
    # retained before predictor inlining (computed membership axes; 606d76d0),
    # plus published live matrices (1t1v8zo) and callable array cells (0mr4zu5).
    # Contains db4f16aa (lower Cholesky + graph allocation + native callback AD
    # repairs), e0930dcd (a declared-array element is a scalar location summand),
    # 712c227b (vector-of-vectors nested-plate observations), 9e425eda
    # (LogDensity laws, link families, per-index arithmetic and derived responses
    # in nested plate cells), 628769da + ea596e50 (built graphs carry no bound
    # row count, so a graph built on one data set evaluates another), 683b8342
    # (one source always lowers to one program) and ed87ceef (a proven scalar
    # response location stays scalar; 0f7d2521 gives other locations their rows
    # once, without a ones vector). 83707d4d adds closure captures in plate
    # cells and scan steps (an enclosing name a do-block reads is shared whole,
    # never zipped), which BRM's emitted do-blocks use instead of Ref operands;
    # bed5d467 deprecates those Ref operands, and ecf99453 packs every hidden
    # bound operand (captures included) into one prepared-AD context, clear
    # of the 32-argument gradient cliff. 1603d87a adds
    # `prepare_sampler(...; retain=(:pointwise,))` and
    # `sampler_value_gradient_and_retained!` (pointwise densities from the
    # gradient's reverse sweep), which BRM's lockstep NLME gradient uses; it
    # contains ecf99453 and bed5d467, plus 3fc1f0c7, 54d4a7b9 and 46d5520c.
    # 5c2b80d5 keeps authored predictor names; 281933141 admits positional
    # level axes over pool definitions (`gg = vcat(g, h)`). f67bd182 adds
    # prepared HVPs, observation queries and pushforwards for Gaussian NLME.
    # 659a6e8b adds joint matrix columns and two-axis priors; cb3a0db7 reads
    # those columns in place during native reverse.
    # 7eafe9ba reads a positional level source as a whole pool, so per-group
    # tables need not have observation length. fbd0aef0 runs authored
    # `@plate for` cells over arrays per index; 2503ede6 supports undotted
    # per-index arithmetic. d702209e evaluates shared live cell outputs once
    # per subject and keeps weights based only on data independent of live
    # parameters. ddb68802 reports Gaussian observation arguments and colored
    # pushforwards for one observed array per plate index. fa23fe7f reads
    # columns of a row-gathered joint monotonic output in place during native
    # reverse.
    # 609c7a9d composes authored plate cells; a2268e47 adds compact
    # selected-coordinate cell gradients used by BRM's NLME adapter.
    # 68a77b03 binds a design without its conditioned response values to the
    # same program (snag rkppl-bind-a-des-e6cfca75), which `reprocess` uses
    # for prediction-only designs; dad59972 extends that to censored,
    # truncated and interval-censored responses (snag
    # rkppl-absent-cen-9af487f3). Both contain f13e68d4 (location-alias
    # elision renames cell parameters). ae731d3f + fe85d9f6 keep a location
    # over a computed design queryable by its authored name, e.g. `mu` over a
    # completed `mi` covariate (snag rkppl-location-a-2ec8954d); cc50dae0 is
    # the RK main containing them. It also observes a carried array that is
    # bound data in `@scan`, at top level and per plate cell (3581f5a8, snag
    # observed-scan-rk-f3f85355), and contains 60ff95f1 (sampled seeds and
    # steps in plate-cell scans). 6c7e18af adds a `@scan` in a per-cell
    # `@rkppl` submodel (c4008d62, snag per-cell-submode-cd87e071), and
    # contains ce48fc77 (`rk_ode_rk45_tol`), 9bef9c04 (per-row provider-law
    # subject cells) and fef26022 (queries skip unread plate outputs).
    # d92d6677 adds ac01d1f1d (`prepare_predictive`) and 2512d6b1c
    # (`predictive_node` / `predictive_targets` for a caller's own vectorized
    # cut), which test/rk_predictive.jl reads through `rk_bound_plan`.
    # 9af9b197 adds 276428589 (kernel construction for another package's
    # module inside a downstream package image, so a whole `RKBRMI` builds
    # in a precompile workload; snag rkppl-build-kern-4607ff90), and contains
    # a89e08381 (the `bind_data` structural cache) and 74e415976 (a computed
    # partition drives a per-cell loop). de89298a adds 7ba63ea6e
    # (`frozen_values`, `bind_data(…; frozen)`, `lower_rkppl(…; frozen)`: a
    # design binds its data-only definitions frozen by value, which
    # test/rk_fitted_state.jl replays without the training data) and
    # 28f8896d8 (a frozen design may omit an input of strings); snag
    # rkppl-bind-froze-da223711. 9f5d1e7de adds 6027d63bd (a position-batched
    # reader keeps each plate cell's result and the `reduce(vcat, cells)`
    # flatten in lane slots, which BRM's emitted cell flatten uses; snag
    # position-batched-cc11130b).
    ("ReactiveKernels",   "https://github.com/nsiccha/ReactiveKernels.jl.git",   "9f5d1e7de06510fe214fe522bc85a1375034337e"),  # main
    # ReactiveKernels' own `[sources]` revision of its unregistered
    # OutputSignatures dependency (RK compat 0.1.5); move it with the RK pin.
    ("OutputSignatures",  "https://github.com/nsiccha/OutputSignatures.jl.git",  "d16c4efb72b963f45de9d6f632c38d46706c23e1"),  # main
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
    # Older Julia 1.10 runtimes need a serial build of the sibling Turing
    # extensions before the normal parallel pass. Julia 1.10.12 includes the
    # loadable_exts fix and uses ordinary parallel precompilation directly.
    Pkg.instantiate(; allow_autoprecomp=false)
    if v"1.10" <= VERSION < v"1.10.12"
        withenv("JULIA_NUM_PRECOMPILE_TASKS" => "1") do
            Pkg.precompile(["Pathfinder", "Turing"])
        end
    end
    Pkg.precompile()
    return nothing
end

main()
