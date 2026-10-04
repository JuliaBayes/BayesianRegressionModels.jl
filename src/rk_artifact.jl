# src/rk_artifact.jl — fusion-append artifacts + SB reference numbers.
#
# The BRM-side library for the parity-closeout append sweep (pair
# closeout-appends; coordinator lock brief 14oeobv + rulings on peer todo
# 1womb1l + the twin's worker-contract reply): v3 RK artifacts
# (`emit_rk_artifact`: BRMI → `(; case_id, ast, defs, plan, meta)` → `.jls`)
# consumed by the RK append driver, and SBBRMI-side reference numbers
# (`sb_prepare_model` + `sb_probe_numbers`: full-posterior value +
# BridgeStan gradient + FD and oracle cross-checks) for the SB leg.
#
# RK-free: translation of an artifact to the bound thin-layer plan lives in
# `BayesianRegressionModelsReactiveKernelsExt` (`rk_translate_artifact`),
# which shares the emitted-source lowering with `_rk_translated_plan` — the
# single source of truth. The append worker (`test/rk_append_worker.jl`)
# orchestrates emit → translate → SB legs per case.

"""
    rk_artifact_version()

The append-artifact shape version this BRM reads and writes (`3`). The
September `(; ast, data, meta)` triple predates submodel `defs` and the
data crossings; v3 carries
`(; case_id, ast, defs, plan, meta)` with `data === plan.columns`.
"""
rk_artifact_version() = 3

# The plan kinds a v3 artifact may carry — the one predicate both the core
# read check (`_check_artifact`) and the extension translate check
# (`rk_translate_artifact`) consult. It currently equals the extension's
# `_RK_PLAN_TYPES`; it is kept as its own predicate so a future plan kind
# can join the RK backend without silently becoming an artifact kind.
const _RK_ARTIFACT_PLAN_TYPES = Union{_RKStructuralPlan,_RKKernelPlan,_RKValuePlan,_RKHeldOutPlan,_RKUnconditionedPlan}

"""
    emit_rk_artifact(brmi::BRMI; case_id, provenance=nothing, brm_pin=nothing, held_out=())

Lower `brmi` through the production RK route (`_brm_rk_plan` →
`_rk_emit_ast` with production defaults) and pack the v3 append artifact
`(; case_id, ast, defs, plan, meta)`. `meta` is
`(; case_id, provenance, brm_pin, generator_version, emitted_at,
julia_version)`; `emitted_at` is a unix-epoch-UTC `Int`
(`Dates` deliberately avoided). `provenance`/`brm_pin` are
caller-supplied (the worker resolves the pin from its checkout; nothing
here shells out). Fails closed on a malformed emission.
"""
function emit_rk_artifact(brmi::BRMI; case_id::AbstractString,
        provenance=nothing, brm_pin=nothing, held_out=())
    isempty(case_id) && error(
        "RK artifact: case_id must be non-empty")
    plan = _brm_rk_plan(brmi; held_out)
    emitted = _rk_emit_ast(plan)
    _check_emitted(emitted, case_id)
    columns = plan.columns
    isempty(columns) && !(plan isa _RKUnconditionedPlan) && error(
        "RK artifact: case `$(case_id)` planned zero data columns")
    meta = (;
        case_id=String(case_id),
        provenance,
        brm_pin,
        generator_version=rk_artifact_version(),
        emitted_at=Int(floor(time())),
        julia_version=string(VERSION))
    return (;
        case_id=String(case_id),
        ast=emitted.main,
        defs=emitted.defs,
        plan,
        meta)
end

function _check_emitted(emitted::_RKEmittedProgram, case_id)
    _rk_validate_source_definitions(emitted)
    main = emitted.main
    main isa Expr && main.head === :block || error(
        "RK artifact: case `$(case_id)` emitted a non-block main " *
        "($(typeof(main))) — refusing to pack an artifact the thin " *
        "layer cannot lower")
    return nothing
end

# Implemented only by the ReactiveKernels package extension
# (`rk_translate_artifact`), which shares the emitted-source lowering with
# `_rk_translated_plan`. Keeping the generic here lets the core emit
# artifacts without loading RK.
function rk_translate_artifact end

"""
    write_rk_artifact(path, artifact) -> path

`Serialization.serialize` an `emit_rk_artifact` artifact to `path`
(the `.jls` the RK driver treats as opaque). Filesystem layout is the
worker's job; this only serializes.
"""
function write_rk_artifact(path::AbstractString, artifact)
    open(path, "w") do io
        Serialization.serialize(io, artifact)
    end
    return path
end

"""
    read_rk_artifact(path) -> artifact

Deserialize a v3 artifact, failing closed on shape or version skew (a
`generator_version` other than `rk_artifact_version()` is a loud error,
never a silent read).
"""
function read_rk_artifact(path::AbstractString)
    artifact = open(path, "r") do io
        Serialization.deserialize(io)
    end
    _check_artifact(artifact, path)
    return artifact
end

function _check_artifact(artifact, path)
    keys(artifact) == (:case_id, :ast, :defs, :plan, :meta) || error(
        "RK artifact: `$(path)` does not hold a v3 artifact " *
        "(keys $(keys(artifact)))")
    artifact.meta.generator_version == rk_artifact_version() || error(
        "RK artifact: `$(path)` has generator_version " *
        "$(artifact.meta.generator_version); this BRM reads " *
        "$(rk_artifact_version())")
    artifact.plan isa _RK_ARTIFACT_PLAN_TYPES || error(
        "RK artifact: `$(path)` carries a $(typeof(artifact.plan)), " *
        "not an RK plan")
    _check_emitted(
        _RKEmittedProgram(artifact.defs, artifact.ast), artifact.case_id)
    return nothing
end

"""
    show_rk_plan(plan) -> String

Compact multi-line dump of an RK structural or kernel plan (the Layer-2
section body): one `key = value` line per plan component. Pure rendering;
the worker splices it verbatim.
"""
function show_rk_plan(plan::_RKStructuralPlan)
    rs = ["($(r.family) $(r.link), response $(r.response), " *
          "predictor $(r.predictor), scale $(_rk_show_scale(r.scale)))" *
          (r.weights === nothing ? "" : ", weights $(r.weights)") *
          ", evidence ($(r.evidence.kind))" for r in plan.responses]
    ps = ["($(p.name), $(p.link), terms [$(_rk_show_terms(p.terms))])"
        for p in plan.predictors]
    prs = ["($(p.predictor), $(p.addressee), $(p.family), " *
          "$(_rk_show_args(p.args)))" for p in plan.population_priors]
    smps = ["($(p.name), $(p.family), $(_rk_show_args(p.args)))"
        for p in plan.parameters]
    as = ["($(a.name) = $(_rk_show_expr(a.expr)))" for a in plan.assignments]
    ds = ["($(d.name) = $(_rk_show_expr(d.expr)))" for d in plan.derived]
    cols = sort!(string.(collect(keys(plan.columns))))
    lines = [
        "n_obs = $(plan.n_obs)",
        "responses   = [$(join(rs, ", "))]",
        "predictors  = [$(join(ps, ", "))]",
        "priors      = [$(join(prs, ", "))]",
        "parameters  = [$(join(smps, ", "))]",
        "assignments = [$(join(as, ", "))]",
        "derived     = [$(join(ds, ", "))]",
        "columns     = [$(join(cols, ", "))]",
    ]
    isempty(plan.ranef_buckets) || push!(lines,
        "ranef       = [$(join((string(b.label) for b in plan.ranef_buckets), ", "))]")
    isempty(plan.vector_parameters) || push!(lines,
        "vectors     = [$(join((string(v.name) for v in plan.vector_parameters), ", "))]")
    return join(lines, "\n") * "\n"
end

function show_rk_plan(plan::_RKKernelPlan)
    smps = ["($(p.name), $(p.family), $(_rk_show_args(p.args)))"
        for p in plan.parameters]
    as = ["($(a.name) = $(_rk_show_expr(a.expr)))" for a in plan.assignments]
    cols = sort!(string.(collect(keys(plan.columns))))
    return join([
        "kernel      = $(plan.kernel.n_subjects)-subject plate",
        "parameters  = [$(join(smps, ", "))]",
        "assignments = [$(join(as, ", "))]",
        "columns     = [$(join(cols, ", "))]",
    ], "\n") * "\n"
end

function show_rk_plan(plan::_RKValuePlan)
    "values      = ordinary callable assignments\n" *
        "observations = [$(join((string(o.name) for o in plan.observations), ", "))]\n" *
        show_rk_plan(plan.regression)
end

function show_rk_plan(plan::_RKHeldOutPlan)
    "held_out    = [$(join(sort!(string.(collect(plan.held_out))), ", "))]\n" *
        show_rk_plan(plan.parent)
end

function show_rk_plan(plan::_RKUnconditionedPlan)
    string(_rk_plan_summary(plan), "\n",
        "  data keys = ", sort!(collect(keys(plan.columns))), "\n",
        "  generative snapshot = retained SLIC model and bindings\n",
        "  native generated draws = unavailable\n")
end

function _rk_show_terms(terms)
    return join([string(t.kind) * " [" *
                 join(string.(t.columns), ", ") * "]" for t in terms], ", ")
end

_rk_show_scale(s::Symbol) = string(s)
_rk_show_scale(s) = repr(s)
_rk_show_args(args::Tuple) = "(" * join(repr.(args), ", ") * ")"
_rk_show_args(args) = repr(args)
_rk_show_expr(e::Expr) = sprint(Base.show_unquoted, e)
_rk_show_expr(e) = repr(e)

# ---------------------------------------------------------------- SB numbers
#
# The SB leg recipe (sibling SB briefs, e.g. fam-gaussian 1fl6esj):
# SBBRMI → stan_instantiate → direct BridgeStan calls with
# propto=false, jacobian=true (no StanProblem rewrap needed) → value +
# AD gradient, mapped back to the RK u-order via an explicit per-case
# map, cross-checked against central differences of the same compiled
# model and an optional caller oracle. The map is explicit because RK
# layout names and Stan unconstrained names live in different
# namespaces; guessing across them would risk silent wrong values, so
# resolution is mechanical and validated (bijection + full coverage),
# never inferred.

"""
    resolve_sb_map(sb_map, rk_names, stan_names; case_id) -> Vector{Int}

Resolve an explicit per-case SB map
(`sb_map::AbstractVector{<:Pair{Symbol,String}}`, RK coordinate name ⇒
Stan unconstrained name) to the Stan-order permutation
`stan_u[i] == rk_u[perm[i]]`. Fails closed on unknown names, duplicates,
or partial coverage — the map must be a bijection over BOTH name lists.
Pure (no Stan); the worker resolves after compiling.
"""
function resolve_sb_map(sb_map::AbstractVector, rk_names::AbstractVector,
        stan_names::AbstractVector; case_id::AbstractString="?")
    rk_pos = Dict{Symbol,Int}(n => i for (i, n) in enumerate(rk_names))
    stan_pos = Dict{String,Int}(n => i for (i, n) in enumerate(stan_names))
    length(rk_pos) == length(rk_names) || error(
        "SB numbers: case `$(case_id)`: RK coordinate names are not unique")
    length(stan_pos) == length(stan_names) || error(
        "SB numbers: case `$(case_id)`: Stan unconstrained names are not unique")
    length(sb_map) == length(rk_names) == length(stan_names) || error(
        "SB numbers: case `$(case_id)`: map covers $(length(sb_map)) " *
        "entries for $(length(rk_names)) RK coords / " *
        "$(length(stan_names)) Stan params — full bijection required")
    perm = Vector{Int}(undef, length(stan_names))
    seen_rk = Set{Symbol}()
    seen_stan = Set{String}()
    for (rk_name, stan_name) in sb_map
        rk_name isa Symbol || error(
            "SB numbers: case `$(case_id)`: RK name `$(rk_name)` is not a Symbol")
        stan_name isa AbstractString || error(
            "SB numbers: case `$(case_id)`: Stan name `$(rk_name)` " *
            "is not a String")
        rk_name in seen_rk && error(
            "SB numbers: case `$(case_id)`: RK name `$(rk_name)` mapped twice")
        stan_name in seen_stan && error(
            "SB numbers: case `$(case_id)`: Stan name `$(stan_name)` " *
            "mapped twice")
        haskey(rk_pos, rk_name) || error(
            "SB numbers: case `$(case_id)`: RK name `$(rk_name)` " *
            "matches no layout coordinate (have [$(join(rk_names, ", "))])")
        haskey(stan_pos, stan_name) || error(
            "SB numbers: case `$(case_id)`: Stan name `$(stan_name)` " *
            "matches no unconstrained parameter " *
            "(have [$(join(stan_names, ", "))])")
        push!(seen_rk, rk_name)
        push!(seen_stan, stan_name)
        perm[stan_pos[stan_name]] = rk_pos[rk_name]
    end
    return perm
end

"""
    apply_sb_map(u_rk, perm) -> u_stan
    unmap_sb_grad(g_stan, perm) -> g_rk

Apply a `resolve_sb_map` permutation (RK-order ⇒ Stan-order) and map a
Stan-order gradient back to RK u-order. Pure; bounds-checked by indexing.
"""
apply_sb_map(u_rk::AbstractVector, perm::AbstractVector{Int}) = u_rk[perm]

function unmap_sb_grad(g_stan::AbstractVector, perm::AbstractVector{Int})
    length(g_stan) == length(perm) || error(
        "SB numbers: Stan gradient has $(length(g_stan)) entries for " *
        "$(length(perm)) mapped coordinates")
    g_rk = similar(g_stan)
    for (stan_i, rk_i) in enumerate(perm)
        g_rk[rk_i] = g_stan[stan_i]
    end
    return g_rk
end

"""
    sb_prepare_model(brmi::BRMI; mod, case_id, stan_path)

Compile the SBBRMI Stan model once per case (`SBBRMI` → `stan_code` →
`stan_instantiate` to `stan_path`). Returns
`(; problem, model, stan_names, stan_code, dim)`. Heavy (Stan compile —
the worker holds a compute token); fail-closed (empty codegen is a
loud error). One preparation serves every probe via `sb_probe_numbers`.
"""
function sb_prepare_model(brmi::BRMI; mod::Module,
        case_id::AbstractString, stan_path::AbstractString)
    isempty(case_id) && error("SB numbers: case_id must be non-empty")
    sb = SBBRMI(brmi; mod)
    code = StanBlocks.stan_code(sb.model)
    code isa AbstractString && !isempty(code) || error(
        "SB numbers: case `$(case_id)`: SBBRMI emitted no Stan code")
    problem = StanBlocks.stan_instantiate(sb.model; path=stan_path)
    model = problem.model
    stan_names = Vector{String}(BridgeStan.param_unc_names(model))
    dim = LogDensityProblems.dimension(problem)
    return (;
        problem, model, stan_names, stan_code=String(code), dim)
end

"""
    sb_probe_numbers(prepared, u_rk::AbstractVector{Float64}, sb_map, rk_names;
        case_id, oracle=nothing, fd_step=1e-6, fd_tol=1e-4)

Full-posterior SB reference numbers at one RK-order probe `u_rk` against
a `sb_prepare_model` preparation: value + AD gradient (direct
BridgeStan, `propto=false, jacobian=true`), mapped through `sb_map`
(see `resolve_sb_map`; `stan_u[i] == u_rk[perm[i]]`), cross-checked
against central differences of the same compiled model plus
`oracle(u_rk)` when given. `rk_names` are the RK u-order coordinate
names from the translated layout (the worker supplies them; never
guessed here). Returns `(; value, gradient, stan_value, stan_gradient,
perm, fd_maxdiff, oracle_value, oracle_diff)`. Fail-closed throughout
(dimension mismatch, map skew, non-finite values, and FD disagreement
beyond `fd_tol` are loud errors, never `nothing`).
"""
function sb_probe_numbers(prepared, u_rk::AbstractVector{Float64}, sb_map,
        rk_names::AbstractVector; case_id::AbstractString,
        oracle=nothing, fd_step::Float64=1e-6, fd_tol::Float64=1e-4)
    isempty(case_id) && error("SB numbers: case_id must be non-empty")
    model = prepared.model
    stan_names = prepared.stan_names
    dim = prepared.dim
    length(u_rk) == length(rk_names) == dim || error(
        "SB numbers: case `$(case_id)`: probe has $(length(u_rk)) " *
        "entries / $(length(rk_names)) RK names for a $dim-dimensional " *
        "Stan model")
    perm = resolve_sb_map(sb_map, rk_names, stan_names; case_id)
    u_stan = Vector{Float64}(apply_sb_map(u_rk, perm))
    value = BridgeStan.log_density(model, u_stan;
        propto=false, jacobian=true)
    isfinite(value) || error(
        "SB numbers: case `$(case_id)`: Stan value is not finite ($value)")
    g_stan = Vector{Float64}(undef, dim)
    BridgeStan.log_density_gradient!(model, u_stan, g_stan;
        propto=false, jacobian=true)
    all(isfinite, g_stan) || error(
        "SB numbers: case `$(case_id)`: Stan gradient is not finite")
    fd = _sb_central_diff(u -> BridgeStan.log_density(model, u;
        propto=false, jacobian=true), u_stan, fd_step)
    fd_maxdiff = maximum(abs.(fd .- g_stan))
    fd_maxdiff <= fd_tol || error(
        "SB numbers: case `$(case_id)`: Stan AD gradient disagrees with " *
        "central differences (max|Δ| = $fd_maxdiff > $fd_tol)")
    gradient = unmap_sb_grad(g_stan, perm)
    oracle_value, oracle_diff = _sb_oracle(oracle, u_rk, value, case_id)
    return (;
        value, gradient, stan_value=value, stan_gradient=g_stan,
        perm, fd_maxdiff, oracle_value, oracle_diff)
end

function _sb_oracle(oracle::Nothing, u_rk, value, case_id)
    return nothing, nothing
end

function _sb_oracle(oracle, u_rk, value, case_id)
    oracle_value = Float64(oracle(u_rk))
    isfinite(oracle_value) || error(
        "SB numbers: case `$(case_id)`: oracle value is not finite")
    return oracle_value, abs(oracle_value - value)
end

function _sb_central_diff(f, u::Vector{Float64}, h::Float64)
    g = Vector{Float64}(undef, length(u))
    up = copy(u)
    um = copy(u)
    for i in eachindex(u)
        up[i] = u[i] + h
        um[i] = u[i] - h
        g[i] = (f(up) - f(um)) / (2h)
        up[i] = u[i]
        um[i] = u[i]
    end
    return g
end
