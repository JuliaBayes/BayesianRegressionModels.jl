# test/rk_append_worker.jl — BRM-side fusion-append worker (one case per process).
#
# Run: julia --project=test test/rk_append_worker.jl --spec <in.toml> --out <dir>
#        [--no-sb | --print-coords] [--no-token]
#
# The RK append driver shells out to this worker once per brm-probe case
# (pair closeout-appends; coordinator lock 14oeobv + rulings; twin
# worker-contract reply). The driver must not grow a BRM dependency, so
# everything BRM-side runs here, in the exact BRM test env:
# emit → translate → RK sections → SB legs → numbers, one process per case.
#
# SPEC (<in.toml>, driver-written TOML): id::String (case id, bare stable
# tag, e.g. "escs"), brm_inputs::String (absolute probe path), u_probes
# (absent = null→origin, expanded post-translate; or a list of
# equal-length float lists), manifest::String, pins (record-through,
# informational only).
#
# PROBE (probe.jl, fusion-authored from docs/examples builders + data):
# a script included into Main that MUST define
#   PROBE = (; builder, data, sb_map, oracle=nothing)
# where builder(data)::BRMI, data is the data table, sb_map is the
# explicit Vector{Pair{Symbol,String}} RK-name ⇒ Stan-unc-name map (see
# BRM.resolve_sb_map), and oracle is nothing or (u_rk -> value). The
# Layer-1 section is the probe text between the verbatim markers
#   # APPEND-LAYER1-BEGIN
#   # APPEND-LAYER1-END
# (exactly one pair, fail-closed otherwise): formula + data statements
# only, no maps/oracles/sampling tails.
#
# OUTPUTS (all into --out, all required; nonzero exit or missing outputs
# => driver ERROR row):
# - artifact.jls: v2 artifact (opaque to the driver).
# - sections.md: Layer 1, Layer 2, Layer 3, Boundary, Layer 4,
#   Verification, Layer SB bodies (spliced verbatim under the driver's
#   closeout header). RK sections render through the twin's
#   transpile_report_v2 (resolved dynamically from the loaded
#   ReactiveKernelsPPL report/; FAILS CLOSED with a loud error while the
#   twin's piece is unlanded — never a hand-rendered divergent copy).
# - numbers.toml: [[probe]] list parallel to u_probes (posterior
#   required; grad in PASS|FAIL|not run required; grad_maxdiff?,
#   sb_value?, sb_grad_maxdiff?, oracle?) + [pins] worker-env pins.
#
# --no-sb skips the Stan leg (RK sections + numbers without sb_*).
# --print-coords compiles the Stan model and prints RK layout names +
# Stan unconstrained names for sb_map authoring, then exits 0 writing no
# case outputs (Stan build files only). --no-token skips the worker's
# own token gating for driver-wrapped invocation (the driver holds the
# token across the worker run instead) — exactly one side gates, never
# both (double hold) and never neither. The heavy Stan legs (SB numbers,
# --print-coords) hold one of
# the 3 fleet compute tokens (self-gated; the driver/shell must NOT hold
# a token across worker invocation).
#
# Includable for unit tests: including this file defines the worker
# functions without running (main() runs only as a script).

using BayesianRegressionModels
using DifferentiationInterface: AutoEnzyme
using Enzyme
using LogDensityProblems: LogDensityProblems
using Pkg
using ReactiveKernels: prepare
using ReactiveKernelsPPL: build_kernel, coordinate_names
import ReactiveKernelsPPL # bind the module name for pathof (selective using does not)
using TOML
using UUIDs: UUID

const BRM = BayesianRegressionModels
const _BRM_ROOT = dirname(@__DIR__)
const _LAYER1_BEGIN = "# APPEND-LAYER1-BEGIN"
const _LAYER1_END = "# APPEND-LAYER1-END"
const _TOKEN_DIR = "/tmp/rk-compute-tokens"

# ------------------------------------------------------------------ argv/spec

function _usage()
    return """
usage: rk_append_worker.jl --spec <in.toml> --out <dir> [--no-sb | --print-coords] [--no-token]
"""
end

function _parse_argv(argv::Vector{String})
    spec = nothing
    out = nothing
    no_sb = false
    print_coords = false
    no_token = false
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--spec"
            i == length(argv) && error("worker: --spec needs a path\n$(_usage())")
            spec = argv[i + 1]
            i += 2
        elseif arg == "--out"
            i == length(argv) && error("worker: --out needs a dir\n$(_usage())")
            out = argv[i + 1]
            i += 2
        elseif arg == "--no-sb"
            no_sb = true
            i += 1
        elseif arg == "--print-coords"
            print_coords = true
            i += 1
        elseif arg == "--no-token"
            no_token = true
            i += 1
        else
            error("worker: unknown argument `$arg`\n$(_usage())")
        end
    end
    spec === nothing && error("worker: --spec is required\n$(_usage())")
    out === nothing && error("worker: --out is required\n$(_usage())")
    no_sb && print_coords && error(
        "worker: --no-sb and --print-coords are mutually exclusive")
    return (; spec, out, no_sb, print_coords, no_token)
end

function _read_spec(path::AbstractString)
    isfile(path) || error("worker: spec `$path` is not a file")
    spec = TOML.parsefile(path)
    id = get(spec, "id", nothing)
    id isa AbstractString && !isempty(id) ||
        error("worker: spec `$path` has no nonempty string id")
    probe = get(spec, "brm_inputs", nothing)
    probe isa AbstractString && !isempty(probe) ||
        error("worker: spec `$path` has no brm_inputs probe path")
    u_raw = get(spec, "u_probes", nothing)
    u_probes = _check_u_probes(u_raw, path)
    return (; id=String(id), probe=String(probe), u_probes,
        manifest=get(spec, "manifest", nothing))
end

function _check_u_probes(u_raw::Nothing, path)
    return nothing
end

function _check_u_probes(u_raw::AbstractVector, path)
    isempty(u_raw) && error(
        "worker: spec `$path`: u_probes is empty (omit it for origin)")
    out = Vector{Vector{Float64}}()
    for (i, u) in enumerate(u_raw)
        u isa AbstractVector || error(
            "worker: spec `$path`: u_probes[$i] is not a list")
        all(x -> x isa Number, u) || error(
            "worker: spec `$path`: u_probes[$i] holds non-numbers")
        push!(out, Float64.(u))
    end
    n = length(first(out))
    all(u -> length(u) == n, out) || error(
        "worker: spec `$path`: u_probes have unequal lengths")
    return out
end

function _check_u_probes(u_raw, path)
    error("worker: spec `$path`: u_probes must be a list of float lists")
end

# --------------------------------------------------------------------- probe

function _extract_layer1(probe_path::AbstractString)
    text = read(probe_path, String)
    lines = split(text, '\n')
    begins = findall(l -> strip(l) == _LAYER1_BEGIN, lines)
    ends = findall(l -> strip(l) == _LAYER1_END, lines)
    length(begins) == 1 && length(ends) == 1 || error(
        "worker: probe `$probe_path` must hold exactly one " *
        "$_LAYER1_BEGIN/$_LAYER1_END pair " *
        "(found $(length(begins))/$(length(ends)))")
    first(begins) < first(ends) || error(
        "worker: probe `$probe_path`: LAYER1 BEGIN follows END")
    body = join(lines[first(begins)+1:first(ends)-1], '\n')
    isempty(strip(body)) && error(
        "worker: probe `$probe_path`: LAYER1 body is empty")
    return strip(body) * "\n"
end

function _load_probe(probe_path::AbstractString)
    isfile(probe_path) || error("worker: probe `$probe_path` is not a file")
    layer1 = _extract_layer1(probe_path)
    include(probe_path)
    isdefined(Main, :PROBE) || error(
        "worker: probe `$probe_path` defines no PROBE")
    probe = Main.PROBE
    probe isa NamedTuple || error(
        "worker: probe `$probe_path`: PROBE is not a NamedTuple")
    for key in (:builder, :data, :sb_map)
        hasproperty(probe, key) || error(
            "worker: probe `$probe_path`: PROBE lacks `$key`")
    end
    # invokelatest: the probe was just include()d, so its builder method is
    # newer than this running function's world age (cf. generative_plan).
    Base.invokelatest(applicable, probe.builder, probe.data) || error(
        "worker: probe `$probe_path`: builder is not callable with data " *
        "(a `@brm data begin ... end` block binds immediately to a BRMI; " *
        "probes must define a data-free `@brm begin ... end` builder plus " *
        "separate data)")
    oracle = hasproperty(probe, :oracle) ? probe.oracle : nothing
    oracle === nothing || oracle isa Function || error(
        "worker: probe `$probe_path`: oracle must be nothing or a Function")
    probe.sb_map isa AbstractVector || error(
        "worker: probe `$probe_path`: sb_map must be a Vector of Pairs")
    return (; builder=probe.builder, data=probe.data,
        sb_map=probe.sb_map, oracle, layer1)
end

# ------------------------------------------------------------------ machinery

function _brm_pin()
    try
        return readchomp(`git -C $_BRM_ROOT rev-parse HEAD`)
    catch e
        error("worker: cannot resolve the BRM pin (`git -C $_BRM_ROOT " *
              "rev-parse HEAD` failed: $e)")
    end
end

function _worker_pins()
    pins = Dict{String,String}()
    try
        deps = Pkg.dependencies()
        for (name, uuid) in (
                ("ReactiveKernels", UUID("78e9f072-d36b-4c73-b7a3-751b0eb26cf9")),
                ("ReactiveKernelsPPL", UUID("e266ea2a-6817-46cd-8bec-6238f0035f44")),
                ("StanBlocks", UUID("2e771a56-c23a-4e0b-9282-20c2e37157e9")),
                ("BridgeStan", UUID("c88b6f0a-829e-4b0b-94b7-f06ab5908f5a")))
            info = get(deps, uuid, nothing)
            info === nothing && continue
            rev = info.git_revision
            if rev === nothing && hasproperty(info, :source)
                # Path/dev deps (the .bootstrap RK checkout) carry no
                # git_revision; resolve the clone's HEAD directly.
                try
                    rev = readchomp(`git -C $(info.source) rev-parse HEAD`)
                catch
                    rev = nothing
                end
            end
            pins[name] = rev === nothing ? "v$(info.version)" : string(rev)
        end
        pins["julia"] = string(VERSION)
        pins["brm"] = _brm_pin()
    catch e
        pins["error"] = "unresolved: $e"
    end
    return pins
end

# Fleet compute token (mirrors `kb-acquire-compute-token` rule v7):
# one fast non-blocking race over all tokens (a free token grants
# immediately), else a BLOCKING flock on the queue lock — the kernel
# grants blocked waiters FIFO, so the head scans all tokens and takes
# the next freed one whichever it is. Linux-only. Returns the held
# token fd; the caller closes it in a finally (close releases the
# lock). Never sleep-polls and never parks on one token (the v3 poll
# race and the v4 tok1-block both starved under load).
function _acquire_compute_token()
    Sys.islinux() || error(
        "worker: compute-token gating is Linux-only (flock)")
    for i in 1:3
        fd = _try_token(i)
        fd === nothing || begin
            println(stderr, "worker: holding compute token $i")
            return fd
        end
    end
    qpath = joinpath(_TOKEN_DIR, "queue.lock")
    isfile(qpath) || error("worker: queue lock `$qpath` missing")
    println(stderr, "worker: all tokens held; joining the queue lock...")
    qfd = ccall(:open, Cint, (Cstring, Cint), qpath, 0)
    qfd < 0 && error("worker: cannot open queue lock `$qpath`")
    if ccall(:flock, Cint, (Cint, Cint), qfd, 2) != 0
        err = Libc.errno()
        ccall(:close, Cint, (Cint,), qfd)
        error("worker: blocking flock on `$qpath` failed, errno $err")
    end
    try
        println(stderr, "worker: queue head; awaiting a free token...")
        while true
            for i in 1:3
                fd = _try_token(i)
                fd === nothing || begin
                    println(stderr, "worker: holding compute token $i")
                    return fd
                end
            end
            sleep(1)
        end
    finally
        ccall(:close, Cint, (Cint,), qfd)
    end
end

# One non-blocking attempt on token `i`: the held fd, or nothing when a
# peer holds it. Real flock errors fail closed (never mistaken for busy).
function _try_token(i::Int)
    path = joinpath(_TOKEN_DIR, "tok$i.lock")
    isfile(path) || error("worker: compute token `$path` missing")
    fd = ccall(:open, Cint, (Cstring, Cint), path, 0)
    fd < 0 && error("worker: cannot open token `$path`")
    if ccall(:flock, Cint, (Cint, Cint), fd, 2 | 4) == 0
        return fd
    end
    err = Libc.errno()
    ccall(:close, Cint, (Cint,), fd)
    err == 11 || error("worker: flock on `$path` failed, errno $err")
    return nothing
end

function _release_compute_token(fd::Cint)
    ccall(:close, Cint, (Cint,), fd)
    return nothing
end

# Twin reporter v2 seam: include the loaded ReactiveKernelsPPL's
# report/transpile_report.jl and resolve transpile_report_v2. Fails
# closed while the twin's piece is unlanded — the worker never
# hand-renders the RK sections.
function _resolve_reporter_v2()
    root = dirname(dirname(pathof(ReactiveKernelsPPL)))
    report = joinpath(root, "report", "transpile_report.jl")
    isfile(report) || error(
        "worker: RK reporter `$report` is missing (ReactiveKernelsPPL " *
        "checkout too old for the append sweep)")
    include(report)
    isdefined(Main, :transpile_report_v2) || error(
        "worker: loaded RK reporter has no transpile_report_v2 — the " *
        "twin's reporter-v2 piece is unlanded; cannot render the RK " *
        "sections (refusing a hand-rendered divergent copy)")
    return Main.transpile_report_v2
end

function _toml_str(s::AbstractString)
    return "\"" * replace(s, "\\" => "\\\\", "\"" => "\\\"") * "\""
end

function _toml_float(x::AbstractFloat)
    isfinite(x) || error("worker: refusing to write non-finite $x to TOML")
    return repr(x)
end

function _write_numbers(path::AbstractString, case_id::AbstractString,
        rows::AbstractVector, pins::AbstractDict)
    open(path, "w") do io
        println(io, "# machine-written by test/rk_append_worker.jl — case $case_id")
        for row in rows
            println(io, "[[probe]]")
            println(io, "posterior = $(_toml_float(row.posterior))")
            println(io, "grad = $(_toml_str(row.grad))")
            row.grad_maxdiff === nothing || println(io,
                "grad_maxdiff = $(_toml_float(row.grad_maxdiff))")
            row.sb_value === nothing || println(io,
                "sb_value = $(_toml_float(row.sb_value))")
            row.sb_grad_maxdiff === nothing || println(io,
                "sb_grad_maxdiff = $(_toml_float(row.sb_grad_maxdiff))")
            if row.oracle !== nothing
                println(io, "[probe.oracle]")
                println(io, "value = $(_toml_float(row.oracle.value))")
                println(io, "diff_vs_sb = $(_toml_float(row.oracle.diff_vs_sb))")
            end
        end
        println(io, "[pins]")
        for name in sort!(collect(keys(pins)))
            println(io, "$name = $(_toml_str(pins[name]))")
        end
    end
    return path
end

function _render_layer_sb(case_id::AbstractString, prepared, sb_map,
        rows::AbstractVector, pins::AbstractDict, oracle_present::Bool)
    lines = String[
        "## Layer SB — SBBRMI reference (machine)",
        "- stan: dim $(prepared.dim), $(length(rows)) probe(s); " *
        "StanBlocks $(get(pins, "StanBlocks", "unresolved")); " *
        "BridgeStan $(get(pins, "BridgeStan", "unresolved")); " *
        "full posterior (propto=false, jacobian=true)",
        "- map (RK ⇒ Stan): " * join(
            ["`$(p.first)`→`$(p.second)`" for p in sb_map], ", "),
    ]
    for (i, row) in enumerate(rows)
        push!(lines, "- probe $i: sb_value = $(repr(row.sb_value)), " *
            "|sb−rk| = $(repr(abs(row.sb_value - row.posterior))), " *
            "sb_grad_maxdiff_vs_rk = $(repr(row.sb_grad_maxdiff)), " *
            "sb_fd_maxdiff = $(repr(row.sb_fd_maxdiff))" *
            (oracle_present ?
                ", oracle_diff = $(repr(row.oracle.diff_vs_sb))" : ""))
    end
    push!(lines, "")
    return join(lines, "\n")
end

# ---------------------------------------------------------------------- main

function _run_case(spec_path::AbstractString, outdir::AbstractString;
        no_sb::Bool, print_coords::Bool, no_token::Bool=false)
    spec = _read_spec(spec_path)
    probe = _load_probe(spec.probe)
    # World age: the probe include() defined builder methods newer than this
    # running frame. Everything downstream (kernels, reporter, oracle) runs
    # at latest world so generated-model calls resolve.
    return Base.invokelatest(_run_case_loaded, spec, probe, outdir;
        no_sb, print_coords, no_token)
end

function _run_case_loaded(spec, probe, outdir::AbstractString;
        no_sb::Bool, print_coords::Bool, no_token::Bool)
    case_id = spec.id
    brmi = probe.builder(probe.data)
    brmi isa BRM.BRMI || error(
        "worker: case `$case_id`: builder(data) did not return a BRMI")
    artifact = BRM.emit_rk_artifact(brmi;
        case_id, provenance=spec.probe, brm_pin=_brm_pin())
    translated = BRM.rk_translate_artifact(artifact)
    # In-production skew tripwire: the artifact route must reproduce the
    # live RKBRMI route bit-equal at the origin.
    # invokelatest throughout: these lower/compile stages eval generated
    # model code and call it internally; only latest-at-call sees it.
    live = Base.invokelatest(BRM.RKBRMI, brmi)
    rt_model = Base.invokelatest(build_kernel, translated)
    origin = zeros(Float64, rt_model.layout.total)
    _assert_live_equal(case_id, live, rt_model, artifact.plan.columns, origin)
    rk_names = Vector{Symbol}(coordinate_names(rt_model.layout))
    dim = rt_model.layout.total
    u_probes = if spec.u_probes === nothing
        [zeros(Float64, dim)]
    else
        for (i, u) in enumerate(spec.u_probes)
            length(u) == dim || error(
                "worker: case `$case_id`: u_probes[$i] has $(length(u)) " *
                "entries for dim $dim")
            all(isfinite, u) || error(
                "worker: case `$case_id`: u_probes[$i] is not finite")
        end
        spec.u_probes
    end
    mkpath(outdir)
    if print_coords
        fd = no_token ? nothing : _acquire_compute_token()
        try
            prepared = Base.invokelatest(BRM.sb_prepare_model, brmi;
                mod=Main, case_id, stan_path=joinpath(outdir, "coords.stan"))
            println("rk_names = $(repr(rk_names))")
            println("stan_names = $(repr(prepared.stan_names))")
            println("dim = $dim")
            return 0
        finally
            fd === nothing || _release_compute_token(fd)
        end
    end
    BRM.write_rk_artifact(joinpath(outdir, "artifact.jls"), artifact)
    reporter_v2 = _resolve_reporter_v2()
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    # The v2 artifact object (not the .jls path) crosses to the
    # reporter (call shape verified against landed RK a715d41a).
    rep = Base.invokelatest(reporter_v2, artifact; u_probes, backend)
    rk_rows = _check_reporter_rows(case_id, rep, u_probes)
    sb_vec, prepared_sb = if no_sb
        ([(; sb_value=nothing, sb_grad_maxdiff=nothing,
            sb_fd_maxdiff=nothing, oracle=nothing) for _ in u_probes],
            nothing)
    else
        fd = no_token ? nothing : _acquire_compute_token()
        try
            sb_out = _run_sb(case_id, brmi, probe, rk_names, u_probes, outdir)
            (sb_out.rows, sb_out.prepared)
        finally
            fd === nothing || _release_compute_token(fd)
        end
    end
    # RK Enzyme gradients for the SB comparison come from BRM's own
    # LDP shim over the live backend (the reporter rows carry no
    # gradient vector); skipped entirely when the SB leg is off.
    rk_grads = if no_sb
        [nothing for _ in u_probes]
    else
        shim = Base.invokelatest(BRM.rk_logdensity_problem, live;
            ad_backend=backend, u0=zeros(Float64, dim))
        map(1:length(u_probes)) do i
            _, g = Base.invokelatest(
                LogDensityProblems.logdensity_and_gradient,
                shim, Vector{Float64}(u_probes[i]))
            all(isfinite, g) || error(
                "worker: case `$case_id`: probe $i RK gradient " *
                "is not finite")
            Vector{Float64}(g)
        end
    end
    rows = map(1:length(u_probes)) do i
        rk = rk_rows[i]
        sb = sb_vec[i]
        sb_grad_maxdiff = sb.sb_value === nothing ? nothing :
            maximum(abs.(sb.gradient .- rk_grads[i]))
        (; posterior=rk.posterior, grad=rk.grad,
            grad_maxdiff=rk.grad_maxdiff, sb_value=sb.sb_value,
            sb_grad_maxdiff, sb_fd_maxdiff=sb.sb_fd_maxdiff,
            oracle=sb.oracle)
    end
    pins = _worker_pins()
    sections = IOBuffer()
    println(sections, "## Layer 1 — BRM input (verbatim probe `$(spec.probe)`)")
    println(sections, "```julia")
    print(sections, probe.layer1)
    println(sections, "```")
    n_obs_str = artifact.plan isa BRM._RKStructuralPlan ?
        string(artifact.plan.n_obs) : "n/a (kernel)"
    println(sections, "data: n_obs = $n_obs_str, " *
        "columns = [$(join(sort!(string.(collect(keys(artifact.plan.columns)))), ", "))], " *
        "source = `$(spec.probe)`")
    println(sections, "")
    println(sections, "## Layer 2 — BRM emitter plan (machine)")
    println(sections, "```")
    print(sections, BRM.show_rk_plan(artifact.plan))
    println(sections, "```")
    println(sections, "")
    print(sections, rep.md)
    endswith(rep.md, "\n") || println(sections, "")
    println(sections, "")
    if !no_sb
        print(sections, _render_layer_sb(case_id, prepared_sb, probe.sb_map,
            rows, pins, probe.oracle !== nothing))
    end
    open(joinpath(outdir, "sections.md"), "w") do io
        print(io, String(take!(sections)))
    end
    _write_numbers(joinpath(outdir, "numbers.toml"), case_id, rows, pins)
    println(stderr, "worker: case `$case_id` done " *
        "($(length(u_probes)) probe(s), sb=$(!no_sb))")
    return 0
end

# Reporter-v2 return seam (verified against the landed RK pin
# a715d41a: `transpile_report_v2(artifact; u_probes, backend)` returns
# `(; md, probes, case_id)` with `probes[i] == (; u, val, grad_ok,
# grad_maxdiff)`; anything else fails closed HERE, never silently).
# `grad_ok` (Bool or nothing) maps to the numbers.toml verdict string.
function _check_reporter_rows(case_id, rep, u_probes)
    rep.md isa AbstractString || error(
        "worker: case `$case_id`: reporter v2 returned no md::String")
    length(rep.probes) == length(u_probes) || error(
        "worker: case `$case_id`: reporter v2 returned " *
        "$(length(rep.probes)) probe rows for $(length(u_probes)) probes")
    return map(1:length(u_probes)) do i
        p = rep.probes[i]
        p.val isa AbstractFloat || error(
            "worker: case `$case_id`: probe $i val is not a float")
        isfinite(p.val) || error(
            "worker: case `$case_id`: probe $i val is not finite")
        grad = p.grad_ok === nothing ? "not run" :
            p.grad_ok === true ? "PASS" :
            p.grad_ok === false ? "FAIL" : error(
            "worker: case `$case_id`: probe $i grad_ok " *
            "(`$(p.grad_ok)`) is not Bool or nothing")
        g = p.grad_maxdiff
        g === nothing || g isa AbstractFloat || error(
            "worker: case `$case_id`: probe $i grad_maxdiff is not numeric")
        (; posterior=Float64(p.val), grad,
            grad_maxdiff=g === nothing ? nothing : Float64(g))
    end
end

function _assert_live_equal(case_id, live, rt_model, columns, u)
    names = sort!(collect(keys(columns)))
    bound = NamedTuple{Tuple(names)}(Tuple(columns[k] for k in names))
    want = :posterior
    have = (:unconstrained, names...)
    klive = Base.invokelatest(prepare, live.model.spec; have, want, bound=bound)
    krt = Base.invokelatest(prepare, rt_model.spec; have, want, bound=bound)
    # Call-site invokelatest: prepare() eval'd these kernel methods after
    # this extent started, so only latest-at-call sees them.
    v_live = Base.invokelatest(klive, Vector{Float64}(u))
    v_rt = Base.invokelatest(krt, Vector{Float64}(u))
    v_rt == v_live || error(
        "worker: case `$case_id`: artifact route posterior $v_rt != " *
        "live RKBRMI route $v_live at the origin (emit/translate skew)")
    return nothing
end

function _run_sb(case_id, brmi, probe, rk_names, u_probes, outdir)
    prepared = Base.invokelatest(BRM.sb_prepare_model, brmi;
        mod=Main, case_id, stan_path=joinpath(outdir, "sb_model.stan"))
    out = map(1:length(u_probes)) do i
        u = Vector{Float64}(u_probes[i])
        nums = Base.invokelatest(BRM.sb_probe_numbers, prepared, u,
            probe.sb_map, rk_names; case_id, oracle=probe.oracle)
        oracle = nums.oracle_value === nothing ? nothing :
            (; value=nums.oracle_value, diff_vs_sb=nums.oracle_diff)
        (; sb_value=nums.value, gradient=nums.gradient,
            sb_fd_maxdiff=nums.fd_maxdiff, oracle)
    end
    return (; rows=out, prepared)
end

function main(argv::Vector{String}=ARGS)
    try
        opts = _parse_argv(argv)
        return _run_case(opts.spec, opts.out;
            no_sb=opts.no_sb, print_coords=opts.print_coords,
            no_token=opts.no_token)
    catch e
        println(stderr, "worker: ERROR: $(sprint(showerror, e))")
        Base.show_backtrace(stderr, catch_backtrace())
        println(stderr)
        return 1
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
