# test/sb_hook_unequal_columns.jl — a `_sb_submodel_rhs!` term over columns of
# unequal length, resampled to new groups, with its helper highlighted.
#
# One data container holds an 8-row observation table and a 5-row event table.
# The custom term reads both and computes, for every observation, the decayed
# amounts of its subject's earlier events. The brm-use reference cites this file
# as its example of these contracts:
#   - a custom term may consume flat unequal columns directly (kernel.md);
#   - a custom term resamples through `reprocess(...; resample_groups=...)`, and
#     a pure Stan helper it calls must not end in `_lp` (reprocessing.md; snag
#     brm-gq-consumer-55e40b4b);
#   - a helper the term calls can be selected with `highlights=` (descriptor.md),
#     and a `@deffun _helper` is selected by its emitted name `u_helper`.
#
# Run: julia --project=test test/sb_hook_unequal_columns.jl
# Set BRM_UNEQUAL_COLUMNS_RUNTIME=0 to skip the compiled-model set.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems

const RUNTIME = get(ENV, "BRM_UNEQUAL_COLUMNS_RUNTIME", "1") != "0"

module UnequalColumns
using BayesianRegressionModels, StanBlocks
import BayesianRegressionModels: _sb_submodel_rhs!
using BayesianRegressionModels: getargs, getkwargs, name

# Observation i sums event_amount[j] * exp(-rate_i * (t_i - t_j)) over the
# events j of its subject at or before t_i. A pure helper: it never touches
# `target`, so its name does not end in `_lp`.
StanBlocks.@deffun decayed_event_sum(obs_time::vector[n], obs_subject::int[n],
        event_time::vector[m], event_subject::int[m], event_amount::vector[m],
        log_rate::vector[n])::vector[n] = begin
    out::vector[n]
    for i in 1:n
        acc = 0.0
        for j in 1:m
            if event_subject[j] == obs_subject[i] && event_time[j] <= obs_time[i]
                acc = acc + event_amount[j] *
                      exp(-exp(log_rate[i]) * (obs_time[i] - event_time[j]))
            end
        end
        out[i] = acc
    end
    out
end
# The same helper under a name Stan reserves for functions that accumulate
# `target`; used only to pin the stanc rule below.
StanBlocks.@deffun decayed_event_sum_lp(args...) = decayed_event_sum(args...)

const decayed_cell = StanBlocks.@slic begin
    return decayed_event_sum(obs_time, obs_subject, event_time, event_subject,
                             event_amount, log_rate)
end
const misnamed_cell = StanBlocks.@slic begin
    return decayed_event_sum_lp(obs_time, obs_subject, event_time, event_subject,
                                event_amount, log_rate)
end

function decayed_events end
function decayed_events_misnamed end

# The term stores each column it reads under its own name, at its own length,
# and passes the per-observation linear predictor `log_rate` through.
function _emit_decayed!(stmts, data, target, cell, rhs)
    columns = getargs(rhs)
    for column in columns
        data[name(column)] = collect(parent(parent(column)))
    end
    obs_time, obs_subject, event_time, event_subject, event_amount = map(name, columns)
    log_rate = name(getkwargs(rhs)[:log_rate])
    push!(stmts, :($target ~ $cell(; obs_time=$obs_time, obs_subject=$obs_subject,
        event_time=$event_time, event_subject=$event_subject,
        event_amount=$event_amount, log_rate=$log_rate)))
    :done
end
_sb_submodel_rhs!(stmts, data, target::Symbol, ::typeof(decayed_events), rhs) =
    _emit_decayed!(stmts, data, target, decayed_cell, rhs)
_sb_submodel_rhs!(stmts, data, target::Symbol, ::typeof(decayed_events_misnamed), rhs) =
    _emit_decayed!(stmts, data, target, misnamed_cell, rhs)

builder = @brm begin
    sigma ~ Exponential(1)
    log_rate ~ 1 + (1 | subject)
    exposure ~ decayed_events(obs_time, subject, event_time, event_subject,
                              event_amount; log_rate=log_rate)
    y ~ Normal(exposure, sigma)
end

misnamed_builder = @brm begin
    sigma ~ Exponential(1)
    log_rate ~ 1 + (1 | subject)
    exposure ~ decayed_events_misnamed(obs_time, subject, event_time, event_subject,
                                       event_amount; log_rate=log_rate)
    y ~ Normal(exposure, sigma)
end

# Observation table (8 rows) and event table (5 rows) in one NamedTuple.
train = (;
    subject=[1, 1, 1, 2, 2, 3, 3, 3],
    obs_time=[0.5, 1.0, 2.0, 0.5, 1.5, 0.25, 1.0, 3.0],
    y=[0.8, 0.6, 0.7, 1.5, 1.0, 0.9, 0.5, 0.9],
    event_subject=[1, 1, 2, 3, 3],
    event_time=[0.0, 1.5, 0.0, 0.0, 2.0],
    event_amount=[1.0, 0.5, 2.0, 1.0, 1.0],
)

# New subjects: a 6-row observation table and a 4-row event table.
future = (;
    subject=[101, 101, 203, 203, 307, 307],
    obs_time=[0.5, 2.5, 1.0, 2.0, 0.5, 1.5],
    y=zeros(6),
    event_subject=[101, 203, 203, 307],
    event_time=[0.0, 0.0, 1.5, 0.25],
    event_amount=[1.0, 2.0, 1.0, 0.5],
)

# Julia oracle for the helper.
decayed(t, s, et, es, ea, lr) =
    [sum((ea[j] * exp(-exp(lr[i]) * (t[i] - et[j])) for j in eachindex(et)
          if es[j] == s[i] && et[j] <= t[i]); init=0.0) for i in eachindex(t)]
end

const UC = UnequalColumns

build(builder, data) = SBBRMI(builder(data); mod=UC, total_groups=())
stanc_ok(sb) = StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false)
gq_section(code) = code[first(findfirst("generated quantities {", code)):end]

@testset "a custom term reads columns of unequal length" begin
    sb = build(UC.builder, UC.train)
    @test length(sb.data[:obs_time]) == 8
    @test length(sb.data[:event_time]) == 5
    @test sb.data[:event_subject] == UC.train.event_subject
    code = StanBlocks.stan_code(sb.model)
    @test occursin("decayed_event_sum(", code)
    @test stanc_ok(sb).ok
end

@testset "the compiled term computes the schedule from both tables" begin
    if RUNTIME
        sb = build(UC.builder, UC.train)
        prob = StanBlocks.stan_instantiate(sb.model)
        dim = LogDensityProblems.dimension(prob)
        pnames = StanBlocks.BridgeStan.param_names(prob.model; include_tp=true)
        for q in ([0.1 * ((i % 5) - 2) for i in 1:dim], [0.3 * sin(i) for i in 1:dim])
            values = StanBlocks.BridgeStan.param_constrain(prob.model, q; include_tp=true)
            pick(stem) = values[[findfirst(==("$stem.$i"), pnames) for i in 1:8]]
            log_rate = pick("log_rate")
            @test length(unique(log_rate)) > 1
            t = UC.train
            @test pick("exposure") ≈ UC.decayed(t.obs_time, t.subject, t.event_time,
                t.event_subject, t.event_amount, log_rate) atol=1e-12
        end
    else
        @info "Skipping compiled-model set (BRM_UNEQUAL_COLUMNS_RUNTIME=0)"
    end
end

@testset "the term resamples to new subjects through generated quantities" begin
    sb = build(UC.builder, UC.train)
    replay = reprocess(sb, UC.future; resample_groups=:subject)
    @test replay.data[:n_subject] == 3
    @test length(replay.data[:obs_time]) == 6
    @test length(replay.data[:event_time]) == 4
    code = StanBlocks.stan_code(replay.model)
    @test occursin("decayed_event_sum(", gq_section(code))
    @test stanc_ok(replay).ok
end

@testset "a pure helper named *_lp breaks only the resample program" begin
    # Stan reserves the `_lp` suffix for functions that accumulate `target` and
    # refuses to call one from generated quantities, whatever its body does.
    # The fit program calls the helper from transformed parameters and passes;
    # the resample program moves the call to generated quantities and fails
    # (snag brm-gq-consumer-55e40b4b).
    sb = build(UC.misnamed_builder, UC.train)
    @test stanc_ok(sb).ok
    replay = reprocess(sb, UC.future; resample_groups=:subject)
    @test occursin("decayed_event_sum_lp(", gq_section(StanBlocks.stan_code(replay.model)))
    result = stanc_ok(replay)
    @test !result.ok
    @test occursin("Target can only be accessed", sprint(show, result))
end

@testset "the term's helper is a named highlight" begin
    d = brm_descriptor(UC.builder, UC.train; mod=UC, total_groups=(),
        highlights=(:decayed_event_sum => "Decayed event exposure",))
    h = only(d.highlights)
    @test h.name === :decayed_event_sum
    @test h.caption == "Decayed event exposure"
    @test occursin("event_amount", h.definition.source)
    @test [def.name for def in h.closure] == [:decayed_event_sum]
    s = only(d.submodels)
    @test s.target === :exposure
    @test s.trigger == "exposure ~ decayed_events(obs_time, subject, event_time, " *
                       "event_subject, event_amount; log_rate=log_rate)"
end

# Stan rejects identifiers that begin with `_`, so StanBlocks emits a
# `@deffun _helper` as `u_helper`; highlights name the emitted definition.
module UnderscoreHelper
using BayesianRegressionModels, StanBlocks
StanBlocks.@deffun _doubled(x::vector[n])::vector[n] = x .* 2.0
builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x
    y ~ Normal(_doubled(mu), sigma)
end
df = (; x=[0.0, 1.0, 2.0], y=[1.0, 2.0, 3.0])
end

@testset "a leading-underscore helper is highlighted by its emitted name" begin
    d = brm_descriptor(UnderscoreHelper.builder, UnderscoreHelper.df;
        mod=UnderscoreHelper, highlights=(:u_doubled,))
    h = only(d.highlights)
    @test h.name === :u_doubled
    @test h.definition.binding === :_doubled
    code = BayesianRegressionModels.stan_code(d.plan)
    @test occursin("u_doubled(", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end
