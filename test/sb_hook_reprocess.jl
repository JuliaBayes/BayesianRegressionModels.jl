# test/sb_hook_reprocess.jl — `reprocess` re-derives the data a custom
# `_sb_submodel_rhs!` term computes, on the replay DataFrame.
#
# The term below compresses an event schedule the way a segment-based
# superposition kernel does: it collapses the observation rows to unique
# (subject, time) pairs and lists every earlier event of each pair. Its Stan
# data are derived inside the hook: an observation → unique-pair index, the
# per-pair event rows, a structural count `n_unique` that sizes a vector inside
# the helper, and a prior location computed from the event amounts. None of
# them is a column of the data container.
#
# `reprocess(sb, new_df)` used to refuse such a model ("data key ... is a
# derived vector with no preprocessing record and is not a column of the new
# DataFrame"), with and without `resample_groups` (snag
# reprocess-fit-sb-b05fb211). It now calls the term again on the new frame and
# takes every key the term writes from that call:
#   - the replay's term data equal a fresh build on the replay frame, and the
#     Stan program is byte-identical to the fit's;
#   - a `name = constant(expr)` keyword keeps its fitted value
#     (`freeze_constants=false` re-evaluates it), while a value the hook
#     computes from the data is re-computed;
#   - a term whose emitted statement depends on the data (a data-derived
#     literal baked into the program) is refused instead of replayed;
#   - an artifact without the records (built before them) still refuses plain
#     replay with the carry-through hint, and still resamples.
#
# Run: julia --project=test test/sb_hook_reprocess.jl
# Set BRM_HOOK_REPROCESS_RUNTIME=0 to skip the compiled-model set.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems

const RUNTIME = get(ENV, "BRM_HOOK_REPROCESS_RUNTIME", "1") != "0"

module HookReplay
using BayesianRegressionModels, StanBlocks
import BayesianRegressionModels: _sb_submodel_rhs!
using BayesianRegressionModels: getargs, getkwargs, name

# Sum over every unique (subject, time) pair's earlier events of
# amount * exp(-rate * gap), broadcast back to the observation rows.
StanBlocks.@deffun segment_exposure(gather::int[n], seg_u::int[p], seg_gap::vector[p],
        seg_amp::vector[p], rate::real, n_unique::int)::vector[n] = begin
    acc = rep_vector(0, n_unique)
    for q in 1:p
        acc[seg_u[q]] += seg_amp[q] * exp(-rate * seg_gap[q])
    end
    acc[gather]
end

const exposure_cell = StanBlocks.@slic begin
    log_rate ~ normal(0, 1)
    log_potency ~ normal(amount_loc, amount_scale)
    return baseline + exp(log_potency) * segment_exposure(
        gather, seg_u, seg_gap, seg_amp, exp(log_rate), n_unique)
end
# The same cell with `n_unique` as a literal argument (see `exposure_literal`).
const literal_cell = StanBlocks.@slic begin
    log_rate ~ normal(0, 1)
    log_potency ~ normal(amount_loc, amount_scale)
    return baseline + exp(log_potency) * segment_exposure(
        gather, seg_u, seg_gap, seg_amp, exp(log_rate), n_unique)
end

# Julia side of the term: `subject` indexes the per-subject event vectors.
function segment_rows(subject, time, event_time, event_amount)
    pair = Dict{Tuple{Int,Float64},Int}()
    gather = Vector{Int}(undef, length(subject))
    us, ts = Int[], Float64[]
    for i in eachindex(subject)
        s, t = Int(subject[i]), Float64(time[i])
        gather[i] = get!(pair, (s, t)) do
            push!(us, s); push!(ts, t); length(us)
        end
    end
    seg_u, seg_gap, seg_amp = Int[], Float64[], Float64[]
    for u in eachindex(us), j in eachindex(event_time[us[u]])
        gap = ts[u] - event_time[us[u]][j]
        gap > 0 || continue
        push!(seg_u, u); push!(seg_gap, gap); push!(seg_amp, event_amount[us[u]][j])
    end
    (; gather, seg_u, seg_gap, seg_amp, n_unique=length(us))
end
amount_location(event_amount) =
    (logs = [log(a) for a in reduce(vcat, event_amount) if a > 0]; sum(logs) / length(logs))

function event_exposure end
function exposure_literal end

term_keys(target) = map(stem -> Symbol(stem, :_, target),
    (; gather=:gather, seg_u=:seg_u, seg_gap=:seg_gap, seg_amp=:seg_amp,
       n_unique=:n_unique, amount_loc=:amount_loc))

function _emit_exposure!(stmts, data, target, rhs; literal::Bool)
    subject, time, event_time, event_amount = map(c -> parent(parent(c)), getargs(rhs, 4))
    kw = getkwargs(rhs)
    seg = segment_rows(subject, time, event_time, event_amount)
    k = term_keys(target)
    data[k.gather] = seg.gather
    data[k.seg_u] = seg.seg_u
    data[k.seg_gap] = seg.seg_gap
    data[k.seg_amp] = seg.seg_amp
    data[k.amount_loc] = amount_location(event_amount)
    common = [Expr(:kw, key, value) for (key, value) in pairs((;
        gather=k.gather, seg_u=k.seg_u, seg_gap=k.seg_gap, seg_amp=k.seg_amp,
        amount_loc=k.amount_loc, amount_scale=name(kw[:amount_scale]),
        baseline=name(kw[:baseline])))]
    if literal
        # A data-derived count baked into the statement: the program itself
        # depends on the data, so a replay cannot reuse it.
        push!(stmts, :($target ~ $literal_cell(; $(common...), n_unique=$(seg.n_unique))))
    else
        data[k.n_unique] = seg.n_unique
        push!(stmts, :($target ~ $exposure_cell(; $(common...), n_unique=$(k.n_unique))))
    end
    :done
end
_sb_submodel_rhs!(stmts, data, target::Symbol, ::typeof(event_exposure), rhs) =
    _emit_exposure!(stmts, data, target, rhs; literal=false)
_sb_submodel_rhs!(stmts, data, target::Symbol, ::typeof(exposure_literal), rhs) =
    _emit_exposure!(stmts, data, target, rhs; literal=true)

builder = @brm begin
    sigma ~ Exponential(1)
    amount_scale = constant(maximum(reduce(vcat, event_amount)))
    baseline ~ 1 + (1 | subject)
    exposure ~ event_exposure(subject, time, event_time, event_amount;
                              amount_scale=amount_scale, baseline=baseline)
    y ~ Normal(exposure, sigma)
end

literal_builder = @brm begin
    sigma ~ Exponential(1)
    amount_scale = constant(maximum(reduce(vcat, event_amount)))
    baseline ~ 1 + (1 | subject)
    exposure ~ exposure_literal(subject, time, event_time, event_amount;
                                amount_scale=amount_scale, baseline=baseline)
    y ~ Normal(exposure, sigma)
end

# Three subjects; observations at irregular times, one duplicated pair.
const events = (;
    event_time=[[0.0, 2.0], [0.0], [0.0, 1.0, 3.0]],
    event_amount=[[1.0, 0.5], [2.0], [1.0, 1.0, 0.25]],
)
train = merge(events, (;
    subject=[1, 1, 1, 2, 2, 3, 3, 3],
    time=[0.5, 1.0, 2.5, 0.5, 1.5, 0.25, 1.5, 1.5],
    y=[0.8, 0.6, 0.7, 1.5, 1.0, 0.9, 0.5, 0.6],
))

# The same subjects on a dense grid: four times each.
const grid = [0.5, 1.5, 2.5, 3.5]
dense = merge(events, (;
    subject=repeat(1:3; inner=length(grid)),
    time=repeat(grid; outer=3),
    y=zeros(3 * length(grid)),
))

# A changed schedule: the amounts differ, so in-hook calibration changes.
rescheduled = merge(dense, (; event_amount=[[2.0, 2.0], [4.0], [1.0, 3.0, 0.5]]))

# Two new subjects (codes 1 and 2 of a new population).
future = (;
    event_time=[[0.0, 1.0], [0.5]],
    event_amount=[[1.5, 1.5], [3.0]],
    subject=[1, 1, 2, 2, 2],
    time=[0.75, 2.0, 1.0, 2.0, 3.0],
    y=zeros(5),
)

# Julia oracle for the term's value.
exposure(data, baseline, potency, rate) = let seg = segment_rows(
        data.subject, data.time, data.event_time, data.event_amount)
    acc = zeros(seg.n_unique)
    for q in eachindex(seg.seg_u)
        acc[seg.seg_u[q]] += seg.seg_amp[q] * exp(-rate * seg.seg_gap[q])
    end
    baseline .+ potency .* acc[seg.gather]
end
end

const HR = HookReplay

build(builder, data) = SBBRMI(builder(data); mod=HR, total_groups=())
stanc_ok(sb) = StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false)
gq_section(code) = code[first(findfirst("generated quantities {", code)):end]
const KEYS = HR.term_keys(:exposure)
const DERIVED = (KEYS.gather, KEYS.seg_u, KEYS.seg_gap, KEYS.seg_amp, KEYS.n_unique)

@testset "the term's data are derived inside the hook" begin
    sb = build(HR.builder, HR.train)
    seg = HR.segment_rows(HR.train.subject, HR.train.time, HR.train.event_time,
                          HR.train.event_amount)
    @test sb.data[KEYS.n_unique] == seg.n_unique == 7
    @test sb.data[KEYS.gather] == seg.gather
    @test all(k -> !hasproperty(HR.train, k), DERIVED)
    @test stanc_ok(sb).ok
end

@testset "reprocess re-derives the term's data on a dense grid" begin
    sb = build(HR.builder, HR.train)
    replay = reprocess(sb, HR.dense)
    fresh = build(HR.builder, HR.dense)
    for k in (DERIVED..., KEYS.amount_loc)
        @test replay.data[k] == fresh.data[k]
    end
    @test replay.data[KEYS.n_unique] == 12
    @test length(replay.data[:y]) == 12
    @test StanBlocks.stan_code(replay.model) == StanBlocks.stan_code(sb.model)
    @test stanc_ok(replay).ok
    # The carry-through spelling (the derived keys as replay columns) gives the
    # same data: re-deriving changes no value, only who computes it.
    carried = merge(HR.dense, NamedTuple{DERIVED}(map(k -> fresh.data[k], DERIVED)))
    by_hand = reprocess(sb, carried)
    @test all(k -> by_hand.data[k] == replay.data[k], keys(replay.data))
    # The replay is itself replayable.
    again = reprocess(replay, HR.train)
    @test all(k -> again.data[k] == sb.data[k], keys(sb.data))
end

@testset "a constant keyword stays fitted; the hook's own calibration does not" begin
    sb = build(HR.builder, HR.train)
    replay = reprocess(sb, HR.rescheduled)
    @test replay.data[:amount_scale] == sb.data[:amount_scale] == 2.0
    @test replay.data[KEYS.amount_loc] == HR.amount_location(HR.rescheduled.event_amount)
    @test replay.data[KEYS.amount_loc] != sb.data[KEYS.amount_loc]
    @test replay.data[KEYS.seg_amp] == build(HR.builder, HR.rescheduled).data[KEYS.seg_amp]
    refit = reprocess(sb, HR.rescheduled; freeze_constants=false)
    @test refit.data[:amount_scale] == 4.0
end

@testset "the term resamples new subjects through generated quantities" begin
    sb = build(HR.builder, HR.train)
    replay = reprocess(sb, HR.future; resample_groups=:subject)
    fresh = build(HR.builder, HR.future)
    @test replay.data[:n_subject] == 2
    for k in DERIVED
        @test replay.data[k] == fresh.data[k]
    end
    @test replay.data[:amount_scale] == sb.data[:amount_scale]
    @test occursin("segment_exposure(", gq_section(StanBlocks.stan_code(replay.model)))
    @test stanc_ok(replay).ok
end

@testset "a statement that depends on the data is refused, not replayed" begin
    sb = build(HR.literal_builder, HR.train)
    @test stanc_ok(sb).ok
    # Same pairs: the statement is unchanged, so the program applies.
    same = merge(HR.train, (; y=zeros(8)))
    @test StanBlocks.stan_code(reprocess(sb, same).model) == StanBlocks.stan_code(sb.model)
    err = try
        reprocess(sb, HR.dense); nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("emits different statements on the new DataFrame", err.msg)
    @test occursin("n_unique = 12", err.msg)
end

@testset "an artifact without the records keeps its old replay behaviour" begin
    sb = build(HR.builder, HR.train)
    record = BayesianRegressionModels._sb_submodel_hook_key(:exposure)
    @test sb.preproc[record].kind === :submodel_hook
    legacy = SBBRMI(sb.parent, sb.model, sb.data,
                    filter(p -> first(p) !== record, sb.preproc),
                    copy(sb.held_out), sb.bindings)
    @test_throws "rebuilt once" reprocess(legacy, HR.dense)
    # Resampling re-emits the model, which records the call on the new frame.
    replay = reprocess(legacy, HR.future; resample_groups=:subject)
    @test replay.data[KEYS.n_unique] == build(HR.builder, HR.future).data[KEYS.n_unique]
end

@testset "the description lists no replay record as a constant" begin
    d = brm_descriptor(HR.builder, HR.train; mod=HR, total_groups=())
    constants = BayesianRegressionModels._brmd_constants(d.plan)
    @test any(c -> c.kind === :constant && c.input === :amount_scale, constants)
    @test !any(c -> c.kind === :submodel_hook, constants)
end

@testset "the replayed program predicts the dense grid" begin
    if RUNTIME
        sb = build(HR.builder, HR.train)
        replay = reprocess(sb, HR.dense)
        fit = StanBlocks.stan_instantiate(sb.model)
        grid = StanBlocks.stan_instantiate(replay.model)
        dim = LogDensityProblems.dimension(fit)
        @test LogDensityProblems.dimension(grid) == dim
        names = StanBlocks.BridgeStan.param_names(grid.model; include_tp=true)
        pick(values, stem, n) = values[[findfirst(==("$stem.$i"), names) for i in 1:n]]
        scalar(values, stem) = values[findfirst(==(stem), names)]
        for q in ([0.1 * ((i % 5) - 2) for i in 1:dim], [0.3 * sin(i) for i in 1:dim])
            values = StanBlocks.BridgeStan.param_constrain(grid.model, q; include_tp=true)
            baseline = pick(values, "baseline", 12)
            @test pick(values, "exposure", 12) ≈ HR.exposure(HR.dense, baseline,
                exp(scalar(values, "exposure_log_potency")),
                exp(scalar(values, "exposure_log_rate"))) atol=1e-12
        end
    else
        @info "Skipping compiled-model set (BRM_HOOK_REPROCESS_RUNTIME=0)"
    end
end
