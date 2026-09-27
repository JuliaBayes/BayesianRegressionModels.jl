# test/sb_sweep_common.jl — shared helpers for the v1 SB-parity sweep.
#
# Included (not standalone) by test/sb_sweep_<family>.jl drivers. Each driver
# emits @brm/@slic counterparts for its inventory slice, evaluates the SB
# (BridgeStan) unconstrained posterior at sweep-defined parity points, and
# appends one JSONL record per (case, point). Records are the SB leg of the
# RK-vs-SB parity comparison; the RK half evaluates at the same published `q`
# (mapped by `stan_names`).
#
# Record schema (one JSON object per line):
#   case        inventory item, e.g. "eight_schools"
#   brm_tip     BRM worktree HEAD the numbers were produced at
#   stan_sha    sha256 of the emitted Stan program (first 16 hex)
#   data_hash   sha256 of the Stan data dict (first 16 hex)
#   stan_names  BridgeStan unconstrained coordinate names (declaration order)
#   q           parity point in stan_names order
#   lp          log_density(propto=false, jacobian=true)
#   label       "zeros" | "seeded" (which parity point)
#   offset      exact constant the SB side ADDS vs the RK model (0.0 = exact);
#               parity compares (lp - offset) against the RK posterior
#   offset_reason derivation of a nonzero offset, else ""

using Test
using BayesianRegressionModels
using StanBlocks
using BridgeStan
using JSON
using Random
using SHA

const BRM = BayesianRegressionModels
const BS = BridgeStan

const _SWEEP_CACHE = joinpath(tempdir(), "brm-sb-sweep")

# Two sweep-defined parity points per case (deterministic, documented):
# q1 = zeros(d) (always valid unconstrained), q2 = seeded randn (generic).
function sweep_points(case::AbstractString, d::Int)
    q1 = zeros(d)
    rng = Xoshiro(hash(String(case)))
    q2 = 0.25 .* randn(rng, d)
    return ("zeros" => q1, "seeded" => q2)
end

_short_sha(bytes) = bytes2hex(sha256(bytes))[1:16]

function _canonical_data(data::Dict)
    parts = String[]
    for k in sort!(collect(keys(data)))
        push!(parts, string(k, "=", repr(data[k])))
    end
    return join(parts, ";")
end

function _brm_tip()
    try
        return readchomp(`git -C $(dirname(@__DIR__)) rev-parse HEAD`)
    catch
        return "unknown"
    end
end

# Compile (cached by Stan hash) + evaluate the SBBRMI-emitted program at `q`.
function sweep_eval(sb::BRM.SBBRMI, q::AbstractVector; case::AbstractString)
    code = BRM.stan_code(sb)
    stan_sha = _short_sha(code)
    mkpath(_SWEEP_CACHE)
    path = joinpath(_SWEEP_CACHE, "$(case)-$(stan_sha).stan")
    isfile(path) || write(path, code)
    problem = Base.invokelatest(StanBlocks.stan_instantiate, sb.model; path=path)
    names = BS.param_unc_names(problem.model)
    @assert length(names) == length(q) "q length $(length(q)) != $(length(names)) for $case"
    lp, _ = BS.log_density_gradient!(problem.model, Vector{Float64}(q), zeros(length(q));
        propto=false, jacobian=true)
    data = BRM.stan_data(sb)
    return (; stan_sha, names, lp, data_hash=_short_sha(_canonical_data(data)))
end

function sweep_record(io::IO, case::AbstractString, sb::BRM.SBBRMI, label, q;
        offset::Float64=0.0, offset_reason::String="")
    ev = sweep_eval(sb, q; case=case)
    rec = Dict(
        "case" => String(case), "label" => String(label),
        "brm_tip" => _brm_tip(), "stan_sha" => ev.stan_sha,
        "data_hash" => ev.data_hash,
        "stan_names" => ev.names, "q" => Vector{Float64}(q), "lp" => ev.lp,
        # offset: exact constant ADDED by the SB counterpart vs the RK model
        # (e.g. explicit-Uniform -log(width) where the .stan is implicit).
        # Parity compares (lp - offset) against the RK posterior.
        "offset" => offset, "offset_reason" => offset_reason,
    )
    println(io, JSON.json(rec))
    return rec
end

# Emit + record both parity points for one case. `build()` returns the SBBRMI;
# `dim()` its unconstrained dimension (names come from the compiled model, so
# build first, then size the points from the evaluated names).
function sweep_case(io::IO, case::AbstractString, build::Function;
        offset::Float64=0.0, offset_reason::String="")
    sb = build()
    code = BRM.stan_code(sb)
    mkpath(_SWEEP_CACHE)
    path = joinpath(_SWEEP_CACHE, "$(case)-$(_short_sha(code)).stan")
    isfile(path) || write(path, code)
    problem = Base.invokelatest(StanBlocks.stan_instantiate, sb.model; path=path)
    names = BS.param_unc_names(problem.model)
    recs = []
    for (label, q) in sweep_points(case, length(names))
        push!(recs, sweep_record(io, case, sb, label, q;
            offset=offset, offset_reason=offset_reason))
    end
    return recs
end
