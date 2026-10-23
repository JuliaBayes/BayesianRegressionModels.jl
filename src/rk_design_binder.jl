# src/rk_design_binder.jl — the graph inputs of new designs for a fitted
# RKBRMI, built once per fit.
#
# `rk_bound_data(reprocess(rk, design))` plans, emits and lowers the whole
# program again for every design, although none of that depends on the
# design's values: everything a design changes is its raw columns, which the
# fitted program reads directly (its preprocessing, level indices and ragged
# row partitions are graph values over them, with fitted constants read from
# fit-role inputs). A binder replays the first design of each kind in full —
# the ordinary `reprocess`, which validates it — and records where each graph
# input comes from: a column of the design, or a value no design changes.
# Every later design of that kind is checked as `reprocess` checks its values,
# and its raw columns are bound into the program lowered once for that kind,
# with the values fitted from fixed inputs alone frozen as that replay bound
# them.
#
# An input the planner derives from the design in Julia (a value BRM has not
# moved into the graph, or data a custom hook writes) cannot be recomputed
# without planning; designs of such a model take the planned replay, which
# `show(binder)` names.

"""
    rk_design_binder(rk::RKBRMI; resample_groups=(), held_out=<the fit's>) -> RKDesignBinder

The graph inputs of new designs (prediction scenarios) for the fitted model
`rk`, built once. `binder(design)` returns the same inputs as
`rk_bound_data(reprocess(rk, design; resample_groups, held_out))`, as a
`NamedTuple`, without planning, emitting or lowering the program again: the
first design of each kind (the same columns, column types, empty columns and
missing values) is replayed in full, and every later one is checked and bound
into that replay's lowered program.

- `binder.fitted` holds the inputs no design changes (the training columns
  behind fitted anchors, `constant(...)` values). Bind them once with
  `ReactiveKernels.prepare(spec; have, want, bound=binder.fitted)` and pass the
  rest per design; values that read only them are then computed once.
- What the program fits from those inputs alone and binding evaluates (spline
  and HSGP fits, fitted means and SDs, level sets) is taken from the first
  design of each kind and bound frozen into every later one (RKPPL's
  `bind_data(…; frozen)`): a later design computes only what its own columns
  determine, such as a basis at its rows.
- A design is refused exactly where `reprocess` refuses its values: unseen
  factor, `mo` or group levels, HSGP axis values past the fitted basis
  interval or fixed domain, ragged labels that name no cell, and per-cell
  inputs whose cell counts disagree.
- A model whose inputs include a value BRM derives from the design in Julia
  replays every design in full; `show(binder)` names those inputs.
"""
function rk_design_binder end

struct RKDesignBinder{B<:RKBRMI}
    rk::B
    resample_groups::Tuple{Vararg{Symbol}}
    held_out::Tuple{Vararg{Symbol}}
    # The fit's data columns: what a design's kind is read from.
    columns::Vector{Symbol}
    fitted::NamedTuple
    classes::Dict{Any,Any}
    lock::ReentrantLock
end

# Where a graph input of a design comes from.
struct _RKFixedInput
    value::Any
end
struct _RKDesignInput
    column::Symbol
end

# One kind of design: how its inputs are read, the program lowered for it, the
# value checks, and the definitions it binds frozen (RKPPL's `FrozenValues`, or
# `nothing`). `derived` names the inputs that force a planned replay.
struct _RKDesignClass
    sources::Vector{Pair{Symbol,Any}}
    lowering::Any
    checks::Vector{Any}
    derived::Vector{Symbol}
    frozen::Any
end
_RKDesignClass(derived::Vector{Symbol}) =
    _RKDesignClass(Pair{Symbol,Any}[], nothing, Any[], derived, nothing)
_rk_design_planned(class::_RKDesignClass) = !isempty(class.derived)

function rk_design_binder(rk::RKBRMI; resample_groups=(),
        held_out=_rk_held_out_names(rk.plan))
    prefix = "RK design binder"
    groups = _rk_resample_group_set(resample_groups; prefix)
    training = _rk_fit_column_names(rk)
    unknown = setdiff(groups, training)
    isempty(unknown) || error(
        "$prefix: `resample_groups` names $(sort!(collect(unknown))), which " *
        "is not a data column of the fitted model")
    groups = Tuple(sort!(collect(groups)))
    RKDesignBinder(rk, groups, Tuple(held_out), sort!(collect(training)),
        _rk_design_fitted(rk, groups), Dict{Any,Any}(), ReentrantLock())
end

# The data columns of the fit `rk` replays: its own, or on a data-free backend
# those of the fit its state was exported from.
_rk_fit_column_names(rk::RKBRMI) = rk.fitted === nothing ?
    Set{Symbol}(keys(_rk_training_columns(parent(rk)))) :
    Set{Symbol}(keys(_rk_fitted_state(rk).columns))
_rk_fit_constant_names(rk::RKBRMI) = rk.fitted === nothing ?
    Set{Symbol}(c.name for c in last(_brm_resolve_constants(parent(rk)))) :
    Set{Symbol}(keys(_rk_fitted_state(rk).constants))

(binder::RKDesignBinder)(design) = _rk_plan_columns(_rk_design_plan(binder, design))

# A design's RKPPL plan: bound into the lowered program of its kind, or the
# planned replay's own binding for the first design of a kind and for a kind
# whose inputs BRM derives in Julia.
function _rk_design_plan(binder::RKDesignBinder, design)
    key = _rk_design_key(binder, design)
    class = lock(() -> get(binder.classes, key, nothing), binder.lock)
    class === nothing && return _rk_design_first!(binder, key, design)
    _rk_design_planned(class) && return rk_bound_plan(_rk_design_replay(binder, design))
    for check in class.checks
        check(design)
    end
    columns = Dict{Symbol,Any}(name => _rk_design_input(source, design)
        for (name, source) in class.sources)
    _rk_bind_design_plan(class.lowering, columns, class.frozen)
end

_rk_design_replay(binder::RKDesignBinder, design) = reprocess(binder.rk, design;
    resample_groups=binder.resample_groups, held_out=binder.held_out)

_rk_design_input(source::_RKFixedInput, _design) = source.value
_rk_design_input(source::_RKDesignInput, design) = _brm_df_column(design, source.column)

# A design's kind: for each fitted data column, absent, or its type, whether it
# is empty, whether it holds missing values and whether it holds one array per
# index — every property of a value that lowering reads (rkppl-use §1) or that
# routes planning.
function _rk_design_key(binder::RKDesignBinder, design)
    Tuple(_brm_df_has_column(design, name) ?
        _rk_design_value_kind(name, _brm_df_column(design, name)) : (name, :absent)
        for name in binder.columns)
end
_rk_design_value_kind(name, value) = (name, typeof(value))
_rk_design_value_kind(name, value::AbstractArray) = (name, typeof(value), isempty(value),
    Missing <: eltype(value) && any(ismissing, value), _rk_design_holds_arrays(value))
_rk_design_holds_arrays(value::AbstractArray) = !(eltype(value) <: Number) &&
    (isempty(value) ? eltype(value) <: AbstractArray : all(x -> x isa AbstractArray, value))

function _rk_design_first!(binder::RKDesignBinder, key, design)
    replay = _rk_design_replay(binder, design)
    plan = rk_bound_plan(replay)
    class = _rk_design_class(binder, replay, design)
    lock(() -> get!(binder.classes, key, class), binder.lock)
    plan
end

# The fit-role inputs of a plan, by the column each one twins.
_rk_plan_fit_inputs(plan::_RKStructuralPlan) = _rk_fit_inputs(plan.derived)
_rk_plan_fit_inputs(plan::_RKValuePlan) = _rk_fit_inputs(plan.regression.derived)
_rk_plan_fit_inputs(plan::_RKHeldOutPlan) = _rk_plan_fit_inputs(plan.parent)
_rk_plan_fit_inputs(_plan) = Dict{Symbol,Symbol}()
function _rk_fit_inputs(derived)
    inputs = Dict{Symbol,Symbol}()
    for spec in derived
        Meta.isexpr(spec.expression, :_rk_fit_input) || continue
        twinned = _rk_fit_twinned(spec.label)
        twinned === nothing || (inputs[spec.name] = twinned)
    end
    inputs
end

_rk_plan_kernels(plan::_RKValuePlan) =
    Tuple(a for a in plan.assignments if a isa _RKPreparedKernelAssignment)
_rk_plan_kernels(plan::_RKHeldOutPlan) = _rk_plan_kernels(plan.parent)
_rk_plan_kernels(_plan) = ()

# The inputs no design changes: fit-role inputs of the columns that are not
# resampled, and fitted `constant(...)` values. A data-free backend binds no
# fit-role input: its frozen fitted values take their place.
function _rk_design_fitted(rk::RKBRMI, groups)
    columns = _rk_source_data_columns(rk.plan, _rk_retained_program(rk))
    twins = _rk_plan_fit_inputs(rk.plan)
    constants = _rk_fit_constant_names(rk)
    frozen = rk.fitted !== nothing
    names = sort!([name for name in keys(columns) if name in constants ||
        (!frozen && haskey(twins, name) && !(twins[name] in groups))])
    fitted = NamedTuple{Tuple(names)}(Tuple(columns[name] for name in names))
    frozen || return fitted
    state = _rk_fitted_state(rk)
    merge(fitted, _rk_frozen_subset(state, _rk_resampled_fit_inputs(state, groups)).values)
end

function _rk_design_class(binder::RKDesignBinder, replay::RKBRMI, design)
    rk = binder.rk
    # A provider that emits new callables per design binds its own program;
    # a plan whose value checks the binder cannot read replays in full.
    replay.program === rk.program || return _RKDesignClass([:callables])
    _rk_design_checked(replay.plan) || return _RKDesignClass([:plan])
    columns = _rk_source_data_columns(replay.plan, _rk_retained_program(rk))
    # A data-free replay binds its fitted values frozen, not the fit-role inputs
    # they are fitted from.
    replay.fitted === nothing ||
        foreach(name -> delete!(columns, name), keys(replay.fitted.frozen.inputs))
    twins = _rk_plan_fit_inputs(replay.plan)
    fitted = binder.fitted
    sources = Pair{Symbol,Any}[]
    derived = Symbol[]
    for (name, value) in columns
        # A fitted input reads the training values on every replay.
        if haskey(fitted, name) && isequal(value, fitted[name])
            push!(sources, name => _RKFixedInput(fitted[name]))
            continue
        end
        column = _rk_design_column_of(binder, design, name, get(twins, name, nothing), value)
        column === nothing ? push!(derived, name) :
            push!(sources, name => _RKDesignInput(column))
    end
    checks = _rk_design_checks(binder, replay.plan, design, derived)
    isempty(derived) || return _RKDesignClass(sort!(derived))
    # What the program fits from the fixed inputs alone is the same for every
    # design: bound frozen, it is computed once, here, not on every binding.
    fixed = Set{Symbol}(name for (name, source) in sources if source isa _RKFixedInput)
    _RKDesignClass(sources, _rk_design_lowering(replay), checks, Symbol[],
        _rk_design_frozen(replay, fixed))
end

_rk_design_checked(::Union{_RKStructuralPlan,_RKValuePlan}) = true
_rk_design_checked(plan::_RKHeldOutPlan) = _rk_design_checked(plan.parent)
_rk_design_checked(_plan) = false

# The design column an input is (by identity: the planner binds raw columns
# unchanged), or `nothing` when the planner derived it.
function _rk_design_column_of(binder, design, name, twin, value)
    for column in (something(twin, name), name, binder.columns...)
        _brm_df_has_column(design, column) || continue
        _brm_df_column(design, column) === value && return column
    end
    nothing
end

# The checks `reprocess` applies to a design's values, as closures over the
# planned terms. A term whose check cannot be read off the plan marks the kind
# as planned (`derived`).
function _rk_design_checks(binder::RKDesignBinder, plan, design, derived)
    checks = Any[]
    has(column) = _brm_df_has_column(design, column)
    for predictor in _rk_replay_predictors(plan), term in predictor.terms
        options = term.options
        if term.kind === :hsgp && get(options, :cov, :exp_quad) === :exp_quad &&
                !get(options, :latent, false) && get(options, :extrapolate, :error) === :error
            if get(options, :mask, nothing) !== nothing || !all(has, term.columns)
                push!(derived, Symbol(predictor.name, "#hsgp"))
                continue
            end
            axes, fits = Tuple(term.columns), options.fits
            fixed_domain = options.fixed_fits !== nothing
            push!(checks, design -> _brm_hsgp_replay_axes(
                Tuple(_brm_df_column(design, axis) for axis in axes), fits, nothing,
                :error; prefix="RK reprocess", names=axes, fixed_domain))
        elseif term.kind === :factor && haskey(options, :index_levels)
            source = only(term.columns)
            has(source) || continue
            levels = options.index_levels
            push!(checks, design -> _brm_apply_fitted_levels(levels,
                _brm_df_column(design, source); prefix="RK reprocess"))
        elseif term.kind === :monotonic && haskey(options, :levels)
            source = options.source
            has(source) || continue
            levels = options.levels
            push!(checks, design -> _brm_apply_fitted_levels(levels,
                _brm_df_column(design, source); prefix="RK reprocess"))
        end
    end
    for kernel in _rk_plan_kernels(plan)
        push!(checks, _rk_design_cell_check(kernel, design))
    end
    checks
end

# A kernel's per-cell inputs and its cell label column must agree on the cell
# count, its cell labels must be unique and each ragged join's labels and axis
# lengths must hold, as `_rk_prepare_kernel_value` requires: a plate iterating
# a shorter input would silently drop cells. Each join is checked as the graph
# checks it when it computes the partition, without building the partition,
# which binding never reads: the graph computes it once the design is bound.
function _rk_design_cell_check(kernel, design)
    has(column) = column !== nothing && _brm_df_has_column(design, column)
    per_cell = unique!(Symbol[input.column for input in kernel.inputs
        if input.kind === :element && has(input.column)])
    cells = has(kernel.cell_column) ? [kernel.cell_column] : Symbol[]
    joins = unique!([input.partition for input in kernel.inputs if input.kind === :gather &&
        get(input, :partition, nothing) !== nothing &&
        all(has, (input.partition.group, input.partition.cell, input.partition.axes...))])
    name = kernel.name
    function check(design)
        for cell in cells
            _brm_kernel_subject_values(_brm_df_column(design, cell), cell; prefix="RK backend")
        end
        lengths = Int[length(_brm_df_column(design, column)) for column in (cells..., per_cell...)]
        all(==(first(lengths)), lengths) || error(
            "RK backend: kernel `$name` positional inputs disagree on subject count")
        column(c) = _brm_df_column(design, c)
        for entry in joins
            _brm_check_ragged_rows(entry.x, entry.group, column(entry.group),
                entry.cell, column(entry.cell), entry.axes,
                Tuple(column(axis) for axis in entry.axes); entry.order,
                times=entry.order === nothing ? nothing : column(entry.order))
        end
        nothing
    end
    isempty(cells) && isempty(per_cell) && isempty(joins) ? (_design -> nothing) : check
end

# Lowering, freezing and binding live in the ReactiveKernels extension.
function _rk_design_lowering end
function _rk_design_frozen end
function _rk_bind_design_plan end
function _rk_plan_columns end

function Base.show(io::IO, binder::RKDesignBinder)
    classes = lock(() -> collect(values(binder.classes)), binder.lock)
    planned = filter(_rk_design_planned, classes)
    print(io, "RKDesignBinder(", length(classes), " design kind",
        length(classes) == 1 ? "" : "s", ", ", length(binder.fitted), " fitted input",
        length(binder.fitted) == 1 ? "" : "s")
    isempty(binder.resample_groups) ||
        print(io, ", resample_groups=", collect(binder.resample_groups))
    print(io, ")")
    isempty(planned) && return
    derived = sort!(unique!(reduce(vcat, (c.derived for c in planned))))
    print(io, "\n  replayed in full for each design: the planner derives ",
        join(("`$d`" for d in derived), ", "), " from the design")
end
