const WARMUPHMC_TEST_MINIMUM =
    "9c642178720d5c294b9cead86fc8c82da5a5db09"

function _git_succeeds(arguments)
    try
        success(pipeline(Cmd(String.(arguments)); stdout=devnull, stderr=devnull))
    catch
        false
    end
end

function _git_output(arguments)
    readchomp(Cmd(String.(arguments)))
end

function _git_checkout_contains(path, minimum)
    path = abspath(path)
    _git_succeeds(["git", "-C", path, "rev-parse", "--git-dir"]) &&
        _git_succeeds(["git", "-C", path, "cat-file", "-e", "$minimum^{commit}"]) &&
        _git_succeeds([
            "git", "-C", path, "merge-base", "--is-ancestor", minimum, "HEAD",
        ])
end

function require_git_ancestor(name, path, minimum; reason)
    path = abspath(path)
    _git_succeeds(["git", "-C", path, "rev-parse", "--git-dir"]) || error("""
        $name at $path is not a readable Git checkout.
        $reason
        Point the test bootstrap at a checkout containing $minimum or later.
        """)

    head = readchomp(`git -C $path rev-parse HEAD`)
    _git_succeeds([
        "git", "-C", path, "merge-base", "--is-ancestor", minimum, head,
    ]) || error("""
        $name checkout $path is too old: HEAD is $head.
        $reason
        Required minimum: $minimum. Materialize that commit or a descendant
        with test/setup_env.jl.
        """)
    head
end

function _require_julia_package_checkout(name, path)
    isdir(path) || error("$name checkout does not exist: $path")
    isfile(joinpath(path, "Project.toml")) ||
        error("$name checkout at $path has no Project.toml")
    path
end

function _mirror_contains(mirror, revision)
    !isempty(mirror) && isdir(mirror) && _git_succeeds([
        "git", "--git-dir=$mirror", "cat-file", "-e", "$revision^{commit}",
    ])
end

function _clone_floor_checkout!(destination, minimum;
        mirror, origin, branch)
    branch_ref = "refs/heads/$branch"
    if _mirror_contains(mirror, minimum)
        if _mirror_contains(mirror, branch_ref) && _git_succeeds([
                "git", "--git-dir=$mirror", "merge-base", "--is-ancestor",
                minimum, branch_ref,
            ])
            run(Cmd([
                "git", "clone", "--quiet", "--branch", branch,
                "--single-branch", mirror, destination,
            ]))
            return "host mirror $branch_ref"
        end
        run(Cmd(["git", "clone", "--quiet", "--no-checkout", mirror, destination]))
        run(Cmd(["git", "-C", destination, "checkout", "--quiet", "--detach", minimum]))
        return "host mirror refs/kb-pins/$minimum"
    end

    run(Cmd([
        "git", "clone", "--quiet", "--branch", branch,
        "--single-branch", origin, destination,
    ]))
    "public $origin $branch"
end

function _clone_revision_checkout!(destination, revision; mirror, origin)
    source = if _mirror_contains(mirror, revision)
        run(Cmd(["git", "clone", "--quiet", "--no-checkout", mirror, destination]))
        "host mirror revision $revision"
    else
        run(Cmd(["git", "clone", "--quiet", "--no-checkout", origin, destination]))
        "public $origin revision $revision"
    end
    run(Cmd([
        "git", "-C", destination, "checkout", "--quiet", "--detach", revision,
    ]))
    source
end

function require_git_revision(name, path, revision)
    path = abspath(path)
    _require_julia_package_checkout(name, path)
    _git_succeeds(["git", "-C", path, "rev-parse", "--git-dir"]) ||
        error("$name at $path is not a readable Git checkout")
    head = _git_output(["git", "-C", path, "rev-parse", "HEAD"])
    head == revision || error("""
        $name checkout $path is at $head, not the pinned revision $revision.
        Remove the stale test bootstrap checkout and run test/setup_env.jl again.
        """)
    head
end

"""
    resolve_git_revision_checkout(name, revision; kwargs...)

Materialize and return an exact, persistent Git checkout below `cache_root`.
Prefer a host mirror containing `revision`, otherwise clone the public `origin`.
Unlike [`resolve_git_floor_checkout`](@ref), descendants do not satisfy an
exact source pin.
"""
function resolve_git_revision_checkout(name, revision;
        cache_root,
        mirror="",
        origin)
    cache_root = abspath(cache_root)
    cache = joinpath(cache_root, "$(lowercase(name))-$(revision[1:12])")
    if ispath(cache)
        require_git_revision(name, cache, revision)
        return cache
    end

    mkpath(cache_root)
    source = mktempdir(cache_root; prefix=".$(lowercase(name))-bootstrap-") do scratch
        source = _clone_revision_checkout!(
            scratch, revision; mirror=mirror, origin=origin)
        require_git_revision(name, scratch, revision)
        if ispath(cache)
            require_git_revision(name, cache, revision)
        else
            mv(scratch, cache)
        end
        source
    end
    @info "Materialized $name test dependency" path=cache source revision
    cache
end

"""
    resolve_git_floor_checkout(name, configured_path, minimum; kwargs...)

Return a Julia-package checkout whose `HEAD` contains `minimum`. A valid
caller-provided checkout wins. Otherwise materialize one persistent,
versioned checkout below `cache_root`, preferring the host mirror's immutable
pin/`branch` and falling back to the public `origin` branch. The shared
`~/github` checkout is deliberately never treated as evidence of what has
landed or been published: it may be stale when its worktree is dirty.
"""
function resolve_git_floor_checkout(name, configured_path, minimum;
        cache_root,
        mirror="",
        origin,
        branch="dev",
        reason)
    if !isnothing(configured_path)
        configured_path = abspath(configured_path)
        _require_julia_package_checkout(name, configured_path)
        if _git_checkout_contains(configured_path, minimum)
            return configured_path
        end
        head = _git_succeeds(["git", "-C", configured_path, "rev-parse", "HEAD"]) ?
            _git_output(["git", "-C", configured_path, "rev-parse", "HEAD"]) :
            "not a Git checkout"
        @warn "$name develop path is stale; materializing the required floor instead" path=configured_path head=head minimum=minimum
    end

    cache_root = abspath(cache_root)
    cache = joinpath(cache_root, "$(lowercase(name))-$(minimum[1:12])")
    if ispath(cache)
        _require_julia_package_checkout(name, cache)
        require_git_ancestor(name, cache, minimum; reason)
        return cache
    end

    mkpath(cache_root)
    source = mktempdir(cache_root; prefix=".$(lowercase(name))-bootstrap-") do scratch
        source = _clone_floor_checkout!(
            scratch, minimum; mirror=mirror, origin=origin, branch=branch)
        _require_julia_package_checkout(name, scratch)
        require_git_ancestor(name, scratch, minimum; reason)
        if ispath(cache)
            _require_julia_package_checkout(name, cache)
            require_git_ancestor(name, cache, minimum; reason)
        else
            mv(scratch, cache)
        end
        source
    end
    @info "Materialized $name test dependency" path=cache source minimum
    cache
end
