# src/rk_fitted_state.jl — a fitted RK model's frozen preprocessing, without
# its training data.
#
# A replay of a fitted RKBRMI (`reprocess`, `rk_design_binder`) binds the
# training columns as fit-role inputs (`_rk_fit_twin!`), and RKPPL's bind-time
# evaluation recomputes every fitted value from them (user principle 077lr1b:
# preprocessing in-graph, frozen by partial evaluation). The emitted program
# names each value it fits from those inputs (`_rk_ast_fitted_value!`:
# `age_fit_mean = brm_covariate_mean(age_fit)`), so the fitted values can stand
# in for the training columns. `brm_fitted_state` exports them (RKPPL's
# `frozen_values`), together with the fits the planner computes in Julia
# (`_brm_fitted`), the `constant(...)` values and the fit's coordinates, and
# `RKBRMI(brmi, state)` binds a design with them frozen by value (RKPPL's
# `bind_data(...; frozen)`): the fit's own program, with the fit-role inputs
# absent.

"""
    BRMFittedState

The frozen preprocessing of a fitted `RKBRMI`, without its training rows:
what [`brm_fitted_state`](@ref) returns. It holds

- `fitted`: the values the emitted program computes from its fit-role inputs
  (`center`/`zscale`/`standardize` means and SDs, group level sets, HSGP
  intervals, …), by the name the program reads them under, as RKPPL's
  `frozen_values` record (`fitted.values`). The record also holds the element
  type and size of each fit-role input, never its values;
- `fitted_reads` and `fit_twins`: the fit-role inputs each fitted value is
  fitted from, and the data column each fit-role input copies, so a replay that
  resamples a grouping refits that grouping's values from its own data;
- `fits`: the fits the planner computes in Julia (factor and `mo` level sets,
  spline fits), keyed by what they fit;
- `constants`: the `name = constant(expr)` values;
- `declaration`: the authored model (`show(brm_source(rk))`), which a replay's
  model must repeat;
- `program`: the fit's emitted program, which a replay must reproduce;
- `coordinates`: the names of the fit's unconstrained coordinates;
- `columns`: the element type of each data column of the fit (a scalar's type);
- `responses`: the responses the fit conditions on;
- `held_out`: the fit's response hold-outs.

Level sets include the group labels of the fitted subjects; means, SDs, level
sets and basis fits are summaries of the training data, never its rows. It
round-trips through `Serialization`.
"""
struct BRMFittedState
    format::Int
    fitted::Any
    fit_inputs::Vector{Symbol}
    fitted_reads::Dict{Symbol,Vector{Symbol}}
    fit_twins::Dict{Symbol,Symbol}
    fits::Dict{Tuple{Symbol,Symbol},Any}
    constants::Dict{Symbol,Any}
    declaration::String
    program::Vector{String}
    coordinates::Vector{Symbol}
    columns::Dict{Symbol,Any}
    responses::Vector{Symbol}
    held_out::Tuple{Vararg{Symbol}}
end

const _BRM_FITTED_STATE_FORMAT = 1

"""
    brm_fitted_state(rk::RKBRMI) -> BRMFittedState

Export the frozen preprocessing of the fitted model `rk` as a
[`BRMFittedState`](@ref) that holds no training rows. A model is exported
only when every value its program fits from the training data is a fitted
value the state holds. Otherwise the error names the statement that still
reads a training column, and that model is replayed with
[`reprocess`](@ref) on the training fit instead.
"""
function brm_fitted_state(rk::RKBRMI)
    prefix = "brm_fitted_state"
    _rk_design_checked(rk.plan) || error(
        "$prefix: a $(nameof(typeof(rk.plan))) program has no fitted-state export")
    for predictor in _rk_replay_predictors(rk.plan), term in predictor.terms
        reason = _rk_fitted_state_boundary(term)
        reason === nothing || error("$prefix: $reason; replay this model with " *
            "`reprocess(rk, new_data)` on the training fit")
    end
    emitted = _rk_retained_program(rk)
    fit_inputs = Set{Symbol}(keys(_rk_plan_fit_inputs(rk.plan)))
    names, reads = _rk_fitted_value_names(emitted, rk.plan, fit_inputs; prefix)
    BRMFittedState(_BRM_FITTED_STATE_FORMAT, _rk_frozen_values(rk, names),
        sort!(collect(fit_inputs)), reads,
        Dict{Symbol,Symbol}(_rk_plan_fit_inputs(rk.plan)),
        _rk_recorded_fits(rk), _rk_fitted_constants(rk), sprint(show, brm_source(rk)),
        _rk_program_statements(emitted), _rk_layout_coordinate_names(rk),
        Dict{Symbol,Any}(name => value isa AbstractArray ? eltype(value) : typeof(value)
            for (name, value) in _rk_training_columns(parent(rk))),
        sort!(collect(_rk_response_names(rk.plan))), Tuple(_rk_held_out_names(rk.plan)))
end

# A term whose fit no summary of the training data stands in for: a thin-plate
# basis keeps every training value of its axis as a knot (`_brm_fit_spline`).
_rk_fitted_state_boundary(term) = term.kind === :spline && term.options.kind === :tps ?
    "`s($(only(term.columns)))` builds its thin-plate basis from every training " *
    "value of `$(only(term.columns))` (its knots), so a fitted state would hold " *
    "that training column" :
    term.kind === :car ?
    "`car($(term.options.source))` reads its fitted adjacency graph as data columns, " *
    "which a fitted state does not export yet" : nothing

# The fitted values of an emitted program: its data-only definitions that read
# fit-role inputs (or other such definitions) and no other data, which a
# statement of another kind reads. Every other statement must read none of
# them, so the fitted values are all a replay needs from the training data.
# Returns their names, and for each the fit-role inputs it is fitted from.
function _rk_fitted_value_names(emitted, plan, fit_inputs::Set{Symbol}; prefix)
    data = Set{Symbol}(keys(plan.columns))
    for statement in emitted.main.args
        _rk_source_assignments!(data, statement)
        _rk_source_outputs!(data, statement)
    end
    fitted = Dict{Symbol,Set{Symbol}}()
    terminal = Symbol[]
    for statement in emitted.main.args
        statement isa Expr || continue
        reads = intersect(_rk_source_symbols!(Set{Symbol}(),
            Meta.isexpr(statement, :(=), 2) ? statement.args[2] : statement), data)
        if Meta.isexpr(statement, :(=), 2) && statement.args[1] isa Symbol &&
                !isempty(reads) && all(name -> name in fit_inputs || haskey(fitted, name), reads)
            fitted[statement.args[1]] = union(intersect(reads, fit_inputs),
                (fitted[name] for name in reads if haskey(fitted, name))...)
            continue
        end
        direct = intersect(reads, fit_inputs)
        isempty(direct) || error(
            "$prefix: `$(sprint(Base.show_unquoted, statement))` reads the " *
            "training data (`$(join(sort!(collect(direct)), "`, `"))`) beside " *
            "other values, so its fitted value cannot be exported yet; replay " *
            "this model with `reprocess(rk, new_data)` on the training fit")
        for name in reads
            haskey(fitted, name) && !(name in terminal) && push!(terminal, name)
        end
    end
    sort!(terminal), Dict{Symbol,Vector{Symbol}}(name => sort!(collect(fitted[name]))
        for name in terminal)
end

# The fits the planner computes in Julia, recorded while it plans the fit's own
# data as a frozen replay of itself (`_brm_fitted`).
function _rk_recorded_fits(rk::RKBRMI)
    training = _rk_training_columns(parent(rk))
    record = Dict{Tuple{Symbol,Symbol},Any}()
    constants = _rk_fitted_constants(rk)
    _brm_with_frozen_replay(_BRMFrozenReplay(training, Set{Symbol}(), nothing, record)) do
        _brm_rk_plan(parent(rk); held_out=_rk_held_out_names(rk.plan),
            fitted_constants=(name, _source) -> get(constants, name, nothing))
    end
    record
end

_rk_fitted_constants(rk::RKBRMI) = Dict{Symbol,Any}(c.name => c.value
    for c in last(_brm_resolve_constants(parent(rk); prefix="brm_fitted_state")))

# The plan of `new_data` under the exported fits alone, as `reprocess` plans it
# under the training data: every Julia-side fit and `constant(...)` value comes
# from `state`, the fit-role inputs bind the replay data's own values (a
# frozen binding replaces them with `state.fitted`), and the emitted program
# must be the fit's. `template` is the model on any data (`builder(new_data)`).
function _rk_fitted_replay_plan(template::BRMI, new_data, state::BRMFittedState;
        resample_groups=(), held_out=state.held_out, prefix="RK data-free replay")
    state.format == _BRM_FITTED_STATE_FORMAT || error(
        "$prefix: a format-$(state.format) fitted state; this BRM reads format " *
        "$_BRM_FITTED_STATE_FORMAT, so export it again from the fit")
    declaration = sprint(show, template)
    declaration == state.declaration || error(
        "$prefix: the fitted state was exported from another model:\n" *
        "  fitted: $(state.declaration)\n  replay: $declaration")
    groups = _rk_resample_group_set(resample_groups; prefix)
    unknown = setdiff(groups, keys(state.columns))
    isempty(unknown) || error(
        "$prefix: `resample_groups` names $(sort!(collect(unknown))), which " *
        "is not a data column of the fitted model")
    absent = Dict{Symbol,Any}(name => _BRMAbsentResponse{state.columns[name]}(name)
        for name in state.responses
        if haskey(state.columns, name) && !_brm_df_has_column(new_data, name))
    rebound = _brm_rebind_brmi(_brm_resolve_optional_priors(template),
        _BRMReplayData(new_data, absent))
    frozen = _BRMFrozenReplay(Dict{Symbol,Any}(), groups, state.fits, nothing)
    plan = _brm_with_frozen_replay(frozen) do
        _brm_rk_plan(rebound; held_out,
            fitted_constants=(name, _source) -> get(state.constants, name, nothing))
    end
    emitted = _rk_emit_ast(plan)
    _rk_check_replay_program(state.program, _rk_program_statements(emitted); prefix)
    _rk_check_replay_terms(plan; prefix)
    _rk_check_replay_groupings(plan, groups; prefix)
    plan, emitted
end

"""
    RKBRMI(brmi::BRMI, state::BRMFittedState; resample_groups=(), held_out=<the fit's>)

A fitted model's RK backend on the data of `brmi` (`builder(design)`), with the
fit's frozen preprocessing taken from `state` ([`brm_fitted_state`](@ref)) and
no training data. Its program and coordinates are the fit's, and its bound
inputs and graph values equal those of `reprocess(rk, design; resample_groups,
held_out)` on the training fit `rk`. The fit-role inputs (`x_fit`) are not
bound: the values the program fits from them are bound frozen in their place.
[`reprocess`](@ref) and [`rk_design_binder`](@ref) on the result replay further
designs from `state` as well. `brmi` must be the fit's model (the same
declaration); a design `reprocess` refuses is refused with the same error.
"""
RKBRMI(brmi::BRMI, state::BRMFittedState; resample_groups=(), held_out=state.held_out) =
    _rk_fitted_replay(brmi, _rk_brmi_data(brmi), state; resample_groups, held_out)

# The data columns of a model, as the replay data they rebind to.
function _rk_brmi_data(brmi::BRMI)
    columns = _rk_training_columns(brmi)
    names = Tuple(sort!(collect(keys(columns))))
    NamedTuple{names}(Tuple(columns[name] for name in names))
end

# What a data-free backend binds from its fitted state: the frozen values of the
# fit, less those a resampled grouping refits from the replay's own data
# (`frozen`, RKPPL's record), for the groupings it resamples.
struct _RKFittedReplay
    state::BRMFittedState
    frozen::Any
    resample::Tuple{Vararg{Symbol}}
end
_rk_fitted_state(rk::RKBRMI) = rk.fitted === nothing ? nothing : rk.fitted.state

# The fit-role inputs a resampled grouping's values are refitted from.
_rk_resampled_fit_inputs(state::BRMFittedState, groups) =
    Set{Symbol}(input for (input, column) in state.fit_twins if column in groups)

# A data-free replay of `new_data`: planned from `state` alone
# (`_rk_fitted_replay_plan`), bound with the fitted values frozen in place of the
# fit-role inputs, on the graph of `previous` (a data-free backend this replays)
# when its program's callables are the same.
function _rk_fitted_replay(template::BRMI, new_data, state::BRMFittedState;
        resample_groups=(), held_out=state.held_out, previous=nothing,
        prefix="RK data-free replay")
    groups = _rk_resample_group_set(resample_groups; prefix)
    plan, emitted = _rk_fitted_replay_plan(template, new_data, state;
        resample_groups=groups, held_out, prefix)
    groups = Tuple(sort!(collect(groups)))
    frozen = _rk_frozen_subset(state, _rk_resampled_fit_inputs(state, groups))
    program, bound, model = _rk_fitted_build(emitted, plan, frozen, previous)
    backend = RKBRMI(_brm_resolve_optional_priors(template), plan, model;
        program, bound, source=template, fitted=_RKFittedReplay(state, frozen, groups))
    if isempty(groups)
        names = _rk_layout_coordinate_names(backend)
        names == state.coordinates || error(
            "$prefix: the new data changes the fitted coordinates " *
            "($(length(state.coordinates)) fitted, $(length(names)) on the " *
            "replay), so the fit's draws do not apply to it. A grouping whose " *
            "levels the replay does not freeze must be named in " *
            "`resample_groups`.")
    end
    backend
end

# The frozen values of a fitted backend's named definitions, the translated
# program and binding of a data-free plan, and its graph (ReactiveKernels
# package extension).
function _rk_frozen_values end
function _rk_fitted_build end
function _rk_frozen_subset end

function Base.show(io::IO, state::BRMFittedState)
    print(io, "BRMFittedState(", length(state.fitted.values), " fitted value",
        length(state.fitted.values) == 1 ? "" : "s", ", ", length(state.fits), " planner fit",
        length(state.fits) == 1 ? "" : "s", ", ", length(state.constants), " constant",
        length(state.constants) == 1 ? "" : "s", ", ", length(state.coordinates),
        " coordinates)")
end
function Base.show(io::IO, ::MIME"text/plain", state::BRMFittedState)
    show(io, state)
    for (name, value) in pairs(state.fitted.values)
        print(io, "\n  ", name, " = ", value isa AbstractArray ?
            string(summary(value)) : repr(value))
    end
end
