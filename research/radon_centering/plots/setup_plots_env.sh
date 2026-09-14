#!/bin/bash
# Rebuild the radon plotting environment.
#
#     bash research/radon_centering/plots/setup_plots_env.sh
#
# The plots env has five unregistered dependencies plus the unregistered BRM
# root itself. Each one is materialized at a specific commit below the ignored
# `research/radon_centering/plots/.bootstrap/` cache — a host-mirror clone
# when the mirror carries the pin, otherwise the public GitHub origin — and
# then resolved with the ecosystem's canonical resolver (lib-resolve.sh),
# which handles the unregistered closure on Julia 1.10 where `[sources]` is
# ignored. A bare `Pkg.instantiate()` fails there with
#
#     ERROR: expected package `AlgebraOfVega [a2420894]` to be registered
#
# Bumping a pin is a deliberate edit of the table below, reviewed like any
# other change — not a silent consequence of whatever a shared checkout
# drifted to. It is idempotent: re-running reuses the cached checkouts.
set -euo pipefail

plots_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$plots_dir/../../.." && pwd)
cache_root="$plots_dir/.bootstrap"
mirror_root="${KB_GIT_MIRRORS:-/home/n/.local/state/kb-agents/git-mirrors}"

# name => "github-org/repo pin-commit" (comment: branch carrying the pin).
# DynamicObjects stays SOURCE-PINNED per user decision 2026-09-14 (no
# registry release): 1d268ea is the newer verified pin; 3352e03 remains the
# historical compatibility evidence (snag kb-extdep-htmxob-2b66fc5b).
# AlgebraOfVega 4124226 is the clean AoV composition source measured for the
# HSGP reference renders; AlgebraOfGraphics/CairoMakie/Makie resolve from the
# registry under the [compat] bounds in plots/Project.toml.
clone_pin() {
    local name="$1" repo="$2" rev="$3"
    local dest="$cache_root/$(echo "$name" | tr '[:upper:]' '[:lower:]')-${rev:0:12}"
    if [ -d "$dest" ]; then
        head=$(git -C "$dest" rev-parse HEAD)
        [ "$head" = "$rev" ] || {
            echo "stale $name checkout at $dest ($head != $rev); remove it and rerun" >&2
            exit 1
        }
        echo "reuse $name $rev ($dest)"
        return 0
    fi
    mkdir -p "$cache_root"
    local mirror="$mirror_root/$repo.git"
    if [ -d "$mirror" ] && git --git-dir="$mirror" cat-file -e "$rev^{commit}" 2>/dev/null; then
        git clone --quiet --no-checkout "$mirror" "$dest"
        echo "cloned $name $rev from host mirror"
    else
        git clone --quiet --no-checkout "https://github.com/nsiccha/$repo.git" "$dest"
        echo "cloned $name $rev from public origin"
    fi
    git -C "$dest" checkout --quiet --detach "$rev"
}

clone_pin AlgebraOfVega AlgebraOfVega.jl 412422660d859a468b8357a11527ff0033e0881c  # scratch composition source
clone_pin DynamicObjects DynamicObjects.jl 1d268ea6169f9152e5d14ac2c1464fd6a96e2793  # kb-impl/DynamicObjects merge
clone_pin HTMXObjects HTMXObjects.jl a813640165d14cf9cb87502ecf632f0c42378a69  # kb-impl/HTMXObjects-openapi
clone_pin HTMX HTMX.jl d52ce5be0f42e3c375370c85499ba5c395f781a8
clone_pin Treebars Treebars.jl c02aa16ab1b08e4f5283597fe678a88e69555cd1  # dev

lower() { echo "$1" | tr '[:upper:]' '[:lower:]'; }
overlay() { echo "$1=$cache_root/$(lower "$1")-${2:0:12}=$2"; }

JULIA_NUM_PRECOMPILE_TASKS=1 RESOLVE_ACCEPTANCE=1 bash -c '
set -euo pipefail
source /home/n/github/nsiccha/Claude/lib-repos.sh
source /home/n/github/nsiccha/Claude/lib-resolve.sh
plots_dir=$1
brm_dir=$2
shift 2
clean_stale_manifest "$plots_dir"
julia --startup-file=no --project="$plots_dir" -e "$(resolve_script "$plots_dir" BayesianRegressionModels "$brm_dir" "$@")"
' _ "$plots_dir" "$repo_root" \
    "$(overlay AlgebraOfVega 412422660d859a468b8357a11527ff0033e0881c)" \
    "$(overlay DynamicObjects 1d268ea6169f9152e5d14ac2c1464fd6a96e2793)" \
    "$(overlay HTMXObjects a813640165d14cf9cb87502ecf632f0c42378a69)" \
    "$(overlay HTMX d52ce5be0f42e3c375370c85499ba5c395f781a8)" \
    "$(overlay Treebars c02aa16ab1b08e4f5283597fe678a88e69555cd1)"
