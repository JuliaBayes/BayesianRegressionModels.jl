# src/rk_reprocess.jl — frozen replay of a fitted RKBRMI on new data.
#
# RK preprocessing is in-graph: the emitted program computes standardization
# anchors, basis fits and level positions from its bound inputs. A replay binds
# new data to the SAME program, with the fitted values frozen at the training
# data's (user principle 077lr1b: preprocessing in-graph, frozen by partial
# evaluation):
#
# - every in-graph anchor reads a fit-role input (`_rk_fit_twin!`): the same
#   array at fit time, the training column at replay, so RKPPL's bind-time
#   evaluation computes the training anchors while the per-row transform reads
#   the new rows;
# - planning the new data runs under `_BRMFrozenReplay`, so every Julia-side
#   fit (level sets, the planner's own value checks) and every
#   `name = constant(expr)` takes its training value.
#
# The replay's emitted program must equal the fit's, statement for statement:
# anything a fit site still derived from the new data would show up as a
# different literal, and is refused rather than evaluated.
#
# A design may omit responses (prediction). Each omitted response plans as a
# `_BRMAbsentResponse`: the fit's route, no values. It is never bound, and
# RKPPL binds its observation without a value (rkppl-use §14), so values that
# do not read it evaluate and the density is refused, naming it.

"""
    reprocess(rk::RKBRMI, new_data; freeze_constants=true, resample_groups=(),
              held_out=<the fit's hold-outs>) -> RKBRMI

Bind `new_data` to the fitted RK model `rk` with the TRAINING fit's
preprocessing, reusing its executable graph (`rk.model`; no rebuild) and its
translated program: a replay only binds its own data. The
returned backend evaluates the same draws on the new rows: densities through
[`rk_logdensity_problem`](@ref), authored values through
`ReactiveKernels.prepare(rk2.model.spec; have, want, bound=rk_bound_data(rk2))`,
and RKPPL's queries over a binding, posterior-predictive draws included,
through [`rk_bound_plan`](@ref).

- `freeze_constants=true` (default; prediction and replay): `center` /
  `zscale` / `standardize` anchors, completed-covariate anchors, factor,
  `mo` and group level sets, spline fits, HSGP intervals and length-scale
  floors, R2D2 variances and `name = constant(expr)` values are the training
  fit's. A one-row design is fine. An unseen factor, `mo` or group level is
  an error.
- `freeze_constants=false`: fresh-fit semantics, equivalent to rebuilding on
  `new_data` (`RKBRMI` of the rebound model); its coordinates need not match
  `rk`'s.
- `resample_groups=[:g]`: group `g` takes its levels from `new_data` (new
  subjects). Its effects are not the fit's, so supply them in `have`
  (`have=(:unconstrained, :b_p_g, ports...)`); everything else stays frozen.
- `held_out`: the response hold-outs of the replay, as in `RKBRMI`.

`new_data` supplies every data column the model reads. With
`freeze_constants=true` it may omit responses: they are bound without values,
so authored values (`want=(:mu,)`, `@plate` cell outputs) evaluate on the new
rows, while the density and other queries that read an omitted response are
refused, naming it. A model whose planning reads the response values (for
example an `mi(y)` response or a bounded likelihood) needs them.
"""
function reprocess(rk::RKBRMI, new_data; freeze_constants::Bool=true,
        resample_groups=(), held_out=_rk_held_out_names(rk.plan))
    prefix = "RK reprocess"
    freeze_constants || return RKBRMI(_brm_rebind_brmi(parent(rk), new_data); held_out)
    # A data-free backend (`RKBRMI(brmi, state)`) replays from its fitted state.
    rk.fitted === nothing || return _rk_fitted_replay(brm_source(rk), new_data,
        rk.fitted; resample_groups, held_out, previous=rk, prefix)
    groups = _rk_resample_group_set(resample_groups; prefix)
    training = _rk_training_columns(parent(rk))
    unknown = setdiff(groups, keys(training))
    isempty(unknown) || error(
        "$prefix: `resample_groups` names $(sort!(collect(unknown))), which " *
        "is not a data column of the fitted model")
    absent = Dict{Symbol,Any}(name => _BRMAbsentResponse{eltype(training[name])}(name)
        for name in _rk_response_names(rk.plan)
        if haskey(training, name) && !_brm_df_has_column(new_data, name))
    rebound = _brm_rebind_brmi(parent(rk), _BRMReplayData(new_data, absent))
    _, fitted = _brm_resolve_constants(parent(rk); prefix)
    constants = Dict{Symbol,Any}(c.name => c.value for c in fitted)
    plan = _brm_with_frozen_replay(_BRMFrozenReplay(training, groups)) do
        _brm_rk_plan(rebound; held_out,
            fitted_constants=(name, _source) -> get(constants, name, nothing))
    end
    retained = _rk_retained_program(rk)
    replayed = _rk_emit_ast(plan)
    _rk_check_replay_program(retained, replayed; prefix)
    _rk_check_replay_terms(plan; prefix)
    _rk_check_replay_groupings(plan, groups; prefix)
    # The fit's program, translated once, binds the replay's data too, unless
    # a callable differs under the same name.
    program = _rk_same_callables(retained, replayed) ? rk.program : _RKOnce()
    replay = RKBRMI(parent(rk), plan, rk.model; program, source=brm_source(rk))
    # Same coordinates, so the fit's draws mean the same thing on the replay.
    # Resampled groups size their effects by the new data's levels.
    if isempty(groups)
        fitted_names = _rk_layout_coordinate_names(rk)
        replay_names = _rk_bound_coordinate_names(replay)
        fitted_names == replay_names || error(
            "$prefix: the new data changes the fitted coordinates " *
            "($(length(fitted_names)) fitted, $(length(replay_names)) on the " *
            "replay), so the fit's draws do not apply to it. A grouping whose " *
            "levels the replay does not freeze must be named in " *
            "`resample_groups`.")
    end
    replay
end

# The emitted program a backend's queries read, and the coordinate names its
# binding packs (ReactiveKernels package extension).
function _rk_retained_program end
function _rk_bound_coordinate_names end

_rk_same_callables(a::_RKEmittedProgram, b::_RKEmittedProgram) =
    length(a.bindings) == length(b.bindings) &&
    all(((x, y),) -> first(x) === first(y) && last(x) === last(y),
        zip(a.bindings, b.bindings))

_rk_replay_predictors(plan::_RKStructuralPlan) = plan.predictors
_rk_replay_predictors(plan::_RKValuePlan) = plan.regression.predictors
_rk_replay_predictors(plan::_RKHeldOutPlan) = _rk_replay_predictors(plan.parent)
_rk_replay_predictors(_plan) = ()

# Latent values that belong to the training rows themselves (an exact GP's
# values at its locations, an AR/DAR path over its time steps, measurement-
# error latents, structured group fields) have no meaning at other rows, and
# a replay with the same number of rows would silently reinterpret them.
const _RK_ROW_LATENT_TERMS = (gp="an exact `gp(...)`", ar="an `ar(...)`/`rw(...)` path",
    dar="a `dar(...)`/`cdar(...)` path", me="a measurement-error `me(...)`",
    structured="a structured group field")
function _rk_check_replay_terms(plan; prefix)
    for predictor in _rk_replay_predictors(plan), term in predictor.terms
        haskey(_RK_ROW_LATENT_TERMS, term.kind) || continue
        error("$prefix: $(_RK_ROW_LATENT_TERMS[term.kind]) term of `$(predictor.name)` " *
            "holds latent values of the training rows, which do not carry to new " *
            "rows; rebuild with `freeze_constants=false`")
    end
end

_rk_replay_buckets(plan::_RKStructuralPlan) = plan.ranef_buckets
_rk_replay_buckets(plan::_RKValuePlan) = plan.regression.ranef_buckets
_rk_replay_buckets(plan::_RKHeldOutPlan) = _rk_replay_buckets(plan.parent)
_rk_replay_buckets(_plan) = ()

# Plain blocks read their levels from a fit-role copy of the grouping column;
# `mm(...)` and `gr(...; by=...)` blocks still read the bound columns' levels,
# so a replay of their groups is refused unless they are resampled.
function _rk_check_replay_groupings(plan, groups::Set{Symbol}; prefix)
    for bucket in _rk_replay_buckets(plan)
        bucket.decomposition === nothing || error(
            "$prefix: a random-effect R2D2M2 budget (`sd(...) ~ r2d2(...)`) does " *
            "not freeze its column variances yet; rebuild with `freeze_constants=false`")
        grouping = bucket.grouping
        grouping.form === :plain && continue
        columns = Symbol[grouping.columns...]
        grouping.by === nothing || push!(columns, grouping.by)
        all(in(groups), columns) && continue
        error("$prefix: `$(grouping.form)(...)` random-effect blocks do not " *
            "freeze their group levels yet; name $(columns) in " *
            "`resample_groups` and supply their effects, or rebuild with " *
            "`freeze_constants=false`")
    end
end

# Replay data with each omitted response standing in as a `_BRMAbsentResponse`.
struct _BRMReplayData{D}
    data::D
    absent::Dict{Symbol,Any}
end
Base.hasproperty(d::_BRMReplayData, key::Symbol) =
    haskey(getfield(d, :absent), key) || hasproperty(getfield(d, :data), key)
Base.getproperty(d::_BRMReplayData, key::Symbol) = haskey(getfield(d, :absent), key) ?
    getfield(d, :absent)[key] : getproperty(getfield(d, :data), key)

# Every response the fit conditions on, held out or not.
_rk_response_names(plan) = _rk_observed_names(plan)
_rk_response_names(plan::_RKHeldOutPlan) = _rk_observed_names(plan.parent)

_rk_held_out_names(plan) = ()
_rk_held_out_names(plan::_RKHeldOutPlan) = Tuple(sort!(collect(plan.held_out)))

function _rk_resample_group_set(groups; prefix)
    groups === nothing && return Set{Symbol}()
    groups isa Symbol && return Set{Symbol}((groups,))
    names = collect(groups)
    all(g -> g isa Symbol, names) || error(
        "$prefix: `resample_groups` expects a Symbol or a collection of Symbols")
    Set{Symbol}(names)
end

# The training data a frozen fit reads, by column name: the materialized
# columns planning reads, and the raw value of any other data column.
function _rk_training_columns(brmi::BRMI)
    columns = Dict{Symbol,Any}()
    for op in _brm_operation_values(brmi)
        _rk_raw_data_columns!(columns, op)
    end
    merge!(columns, _brm_backend_context(brmi; retain_mm_sources=true).data)
end

_rk_raw_data_columns!(columns, _x) = columns
function _rk_raw_data_columns!(columns, x::NamedColumn)
    backing = parent(x)
    backing isa DataColumn && (columns[name(x)] = parent(backing))
    _rk_raw_data_columns!(columns, backing)
end
_rk_raw_data_columns!(columns, x::ExprColumn) =
    (_brm_foreach_child(a -> _rk_raw_data_columns!(columns, a), x); columns)

function _rk_program_statements(emitted::_RKEmittedProgram)
    text(x) = sprint(Base.show_unquoted, Base.remove_linenums!(deepcopy(x)))
    vcat(["definition: " * text(d) for d in emitted.defs],
        ["statement: " * text(s) for s in emitted.main.args if !(s isa LineNumberNode)],
        ["binding: " * string(first(b)) for b in emitted.bindings])
end

_rk_check_replay_program(retained::_RKEmittedProgram, replayed::_RKEmittedProgram;
    prefix) = _rk_check_replay_program(_rk_program_statements(retained),
        _rk_program_statements(replayed); prefix)
function _rk_check_replay_program(fitted::Vector{String}, replay::Vector{String}; prefix)
    fitted == replay && return nothing
    k = something(findfirst(i -> i > length(replay) || fitted[i] != replay[i],
        eachindex(fitted)), length(fitted) + 1)
    error("$prefix: the new data changes the fitted RK program, so its " *
        "preprocessing cannot be frozen. First difference:\n" *
        "  fitted: $(get(fitted, k, "(end of program)"))\n" *
        "  replay: $(get(replay, k, "(end of program)"))\n" *
        "Check that the new data keeps the fitted levels and column types, or " *
        "rebuild with `freeze_constants=false`.")
end

"""
    rk_bound_data(rk::RKBRMI) -> NamedTuple

The bound inputs of `rk`'s executable graph for its own data: pass them as
`bound=` to `ReactiveKernels.prepare(rk.model.spec; have=(:unconstrained,
keys(bound)...), want, bound)`. For [`reprocess`](@ref)'d backends these are
the new data's inputs with the training preprocessing frozen. They are the
columns of [`rk_bound_plan`](@ref)`(rk)`. Implemented by the ReactiveKernels
package extension.
"""
function rk_bound_data end

"""
    rk_bound_plan(rk::RKBRMI) -> ReactiveKernelsPPL.StructuralPlan

The RKPPL binding of `rk`'s program to its own data: the bound plan every query
of `rk` evaluates on `rk.model`. For a [`reprocess`](@ref)'d backend it binds the
new data, with the training preprocessing frozen and the responses the data
omit left absent. `(rk.model, rk_bound_plan(rk))` is the `(built, plan)` pair
RKPPL's queries take (rkppl-use §14), so each of them applies to `rk` as
documented there. Posterior-predictive draws on a new design, for example:

```julia
replay = reprocess(rk, design; resample_groups=[:subject])
targets = predictive_targets(replay.model, rk_bound_plan(replay))
# WANT predictive_node(replay.model) in a cut over replay.model.spec, then
draws = predictive_draws(value, targets, rng)
```

Without `resample_groups`, `prepare_predictive(replay.model, rk_bound_plan(replay))`
draws the whole model at a draw `u` of the fit. Every query of `rk` reads this
one plan: read it, never mutate it. Implemented by the ReactiveKernels package
extension.
"""
function rk_bound_plan end
