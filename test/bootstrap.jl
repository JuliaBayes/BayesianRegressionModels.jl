# Stand up the test environment described by `test/Project.toml`.
#
#     BRM_TEST_STANBLOCKS=/path/to/StanBlocks.jl \
#     BRM_TEST_TREEBARS=/path/to/Treebars.jl \
#     julia --project=test test/bootstrap.jl
#
# then run any test file with `julia --project=test test/<file>.jl`.
# See `test/README.md` for why each of the constraints below exists.

using Pkg
using TOML

include("dependency_floors.jl")

const REPO = dirname(@__DIR__)
const TESTENV = @__DIR__

# Seven packages must enter resolution as develop paths. The caller supplies
# StanBlocks and Treebars; WarmupHMC may be supplied or reproducibly materialized
# at its enforced floor. The three remaining source-only direct dependencies are
# materialized at the exact revisions committed in `test/Project.toml`.
# Absolute paths are machine-specific and deliberately never committed;
# `test/Manifest.toml` (where `Pkg.develop` records them) is covered by the root
# `.gitignore`.
function dev_path(name, var; required=true)
    raw = get(ENV, var, "")
    if isempty(raw)
        required && error("""
            $var is unset — $name has to be an explicit develop path.
            Point it at a local checkout: $var=/path/to/$name.jl
            """)
        return nothing
    end
    path = abspath(expanduser(raw))
    isdir(path) || error("$var=$raw does not name a directory ($path)")
    isfile(joinpath(path, "Project.toml")) ||
        error("$var=$path is not a Julia package (no Project.toml)")
    path
end

stanblocks = dev_path("StanBlocks", "BRM_TEST_STANBLOCKS")
configured_warmuphmc = dev_path(
    "WarmupHMC", "BRM_TEST_WARMUPHMC"; required=false)
warmuphmc = resolve_git_floor_checkout(
    "WarmupHMC",
    configured_warmuphmc,
    WARMUPHMC_TEST_MINIMUM;
    cache_root=joinpath(TESTENV, ".bootstrap"),
    mirror=get(ENV, "BRM_TEST_WARMUPHMC_MIRROR", joinpath(
        homedir(), ".local", "state", "kb-agents", "git-mirrors",
        "WarmupHMC.jl.git")),
    origin=get(ENV, "BRM_TEST_WARMUPHMC_ORIGIN",
        "https://github.com/nsiccha/WarmupHMC.jl.git"),
    branch="dev",
    reason="The test environment requires WarmupHMC's own-gradient " *
        "initialization and Pathfinder 0.10.7 compatibility with Turing 0.46.",
)
# Transitive: WarmupHMC depends on the unregistered Treebars and pins it with a
# `[sources]` entry, which Julia 1.10 ignores. `Pkg.develop` can only fix a path
# for a DIRECT dependency, so Treebars is also listed in `test/Project.toml`.
treebars = dev_path("Treebars", "BRM_TEST_TREEBARS")

project = TOML.parsefile(joinpath(TESTENV, "Project.toml"))
sources = project["sources"]
explicit_paths = Dict(
    "BayesianRegressionModels" => REPO,
    "StanBlocks" => stanblocks,
    "WarmupHMC" => warmuphmc,
    "Treebars" => treebars,
)
source_only = Dict(name => resolve_git_revision_checkout(
        name,
        source["rev"];
        cache_root=joinpath(TESTENV, ".bootstrap"),
        mirror=joinpath(
            homedir(), ".local", "state", "kb-agents", "git-mirrors",
            "$name.jl.git"),
        origin=source["url"],
    ) for (name, source) in sources if !haskey(explicit_paths, name))
merge!(explicit_paths, source_only)

@info "Developing into $TESTENV" paths = explicit_paths

Pkg.activate(TESTENV)

# ONE `develop` call for all seven. Resolution has to satisfy them together:
# omitting any unregistered direct dependency fails before the manifest can be
# written, while developing StanBlocks on its own also fails because BRM is
# unregistered and is not yet present in the manifest.
Pkg.develop(PackageSpec[
    PackageSpec(path=path) for (_name, path) in sort!(collect(explicit_paths))
])

# Every remaining dependency comes from the registry, per the committed
# `test/Project.toml` — the single source of truth for the dependency list and
# source revisions. This script only materializes paths that cannot be committed.
#
# `Pkg.instantiate()` auto-precompiles the whole manifest IN PARALLEL, and under
# Pkg 1.10 that parallel pass self-deadlocks on Pathfinder 0.10.7's two sibling
# Turing extensions (see test/README.md, "Pathfinder's Turing extension pair").
# So suppress the parallel auto-precompile, build just that pair serially first —
# one serial build lands a `.ji` valid for both siblings — then run the ordinary
# parallel pass, which finds them cached and never spawns the deadlocking workers.
# Both names are required: Pkg 1.10 keeps an extension in a named precompile only
# when its full trigger set is in the named closure, and Turing pulls them all in.
Pkg.instantiate(; allow_autoprecomp=false)
withenv("JULIA_NUM_PRECOMPILE_TASKS" => "1") do
    Pkg.precompile(["Pathfinder", "Turing"])
end
Pkg.precompile()

@info "Test environment ready" project = Base.active_project()
