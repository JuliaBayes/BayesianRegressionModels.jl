# Group identity and event-row partitions shared by kernel emitters.
# This layer owns no SLIC statements or backend data cleanup.
_brm_group_values(raw::CA.CategoricalVector) = CA.unwrap.(raw)
_brm_group_values(raw::AbstractVector) = raw

_brm_collect_data_columns!(_acc, _x) = nothing
_brm_collect_data_columns!(acc, x::NamedColumn) = begin
    p = parent(x)
    p isa DataColumn ? _brm_push_data_column!(acc, name(x), parent(p)) :
        _brm_collect_data_columns!(acc, p)
    nothing
end
# A scalar data column — a scalar field of the container or a `constant(...)`
# declaration — broadcasts over whatever frame reads it, so it names no row axis.
_brm_push_data_column!(acc, column::Symbol, values) = push!(acc, (column, values))
_brm_push_data_column!(_acc, ::Symbol, ::Number) = nothing
_brm_collect_data_columns!(acc, x::ExprColumn) = begin
    getf(x) === (~) && return nothing
    for a in getargs(x); _brm_collect_data_columns!(acc, a); end
    for v in values(getkwargs(x)); _brm_collect_data_columns!(acc, v); end
    nothing
end
_brm_collect_data_columns!(acc, x::Union{Tuple,AbstractVector}) = begin
    for a in x; _brm_collect_data_columns!(acc, a); end
    nothing
end
# `(name, length)` of each row-wise data column `x` reads.
function _brm_collect_data_lengths!(acc, x)
    columns = Tuple{Symbol,Any}[]
    _brm_collect_data_columns!(columns, x)
    append!(acc, ((column, length(values)) for (column, values) in columns))
    nothing
end

# `(name, values)` of every row-wise raw data column a linear predictor's terms
# read; scalar columns are skipped.
function _brm_kernel_lp_data_columns(lp_col)
    columns = Tuple{Symbol,Any}[]
    _brm_collect_data_columns!(columns, getargs(parent(lp_col))[2])
    columns
end
_brm_kernel_lp_data_lengths(lp_col) =
    Tuple{Symbol,Int}[(column, length(values)) for (column, values) in _brm_kernel_lp_data_columns(lp_col)]

# A per-subject linear predictor without a random effect defines no grouping;
# it rides on the subject frame a ranef LP — or, in a no-random-effects panel,
# the pre-grouped columns — defines, sliced by the same row index. That is
# sound only when every data column its terms name has exactly one row per
# subject. An LP with no data column has no rows at all.
function _brm_kernel_check_subject_frame_lp(lp_col, nsub; prefix="BRM kernel preparation")
    lp = name(lp_col)
    lens = _brm_kernel_lp_data_lengths(lp_col)
    isempty(lens) && error(
        "$prefix: kernel(...) per-subject linear predictor `$lp` has no random-effect " *
        "term and no data column, so it has no per-subject rows. Declare a shared " *
        "scalar as `$lp ~ <distribution>` and read it in the cell without `[i]`.")
    for (column, n) in lens
        n == nsub || error(
            "$prefix: kernel(...) linear predictor `$lp` has no random-effect term, so " *
            "it is read per subject (one entry per subject), but its data vector " *
            "`$column` has $n rows for $nsub subjects. Read it at each subject's " *
            "indices instead, `$lp[rows[i]]` with `rows = groups(group => labels)`.")
    end
    nothing
end

# One non-missing unique label per row: what lets a grouping column label a
# plate's cells.
function _brm_kernel_labels_rows(raw::AbstractVector)
    values = _brm_group_values(raw)
    !isempty(values) && !any(ismissing, values) && allunique(values)
end
_brm_kernel_labels_rows(_raw) = false

function _brm_kernel_subject_values(raw, group::Symbol; prefix="BRM kernel preparation")
    raw isa AbstractVector || error(
        "$prefix: kernel(...) subject grouping column `$group` must be a vector, " *
        "got $(typeof(raw)).")
    values = collect(_brm_group_values(raw))
    _brm_kernel_labels_rows(values) || error(
        "$prefix: kernel(...) needs pre-grouped per-subject data — `$group` must " *
        "list one non-missing unique subject per row. Repeated levels indicate " *
        "long-format data. Got $(values).")
    values
end

# An empty `ev_vals` is a zero-row axis (a design with no rows on it, such as
# no placebo observations): every cell gets an empty slice. Each cell holds
# its rows ascending. The RK graph computes this on every call of a reader
# whose design is a runtime input, so it finds each row's cell once and sizes
# every cell before filling it.
function _brm_kernel_ragged_partition(a_name::Symbol, grp_name::Symbol,
                                     ev_vals::AbstractVector,
                                     subject_vals::AbstractVector; prefix="BRM kernel preparation")
    _brm_has_missing(ev_vals) && error(
        "$prefix: kernel(...) `ragged($a_name, $grp_name)`: `$grp_name` must be a " *
        "column with no missing labels.")
    cell = _brm_label_positions(ev_vals, subject_vals)
    counts = zeros(Int, length(subject_vals))
    for i in cell
        i == 0 || (counts[i] += 1)
    end
    if sum(counts) != length(cell)
        unknown = Any[v for (v, i) in zip(ev_vals, cell) if i == 0]
        error(
            "$prefix: kernel(...) `ragged($a_name, $grp_name)`: label(s) " *
            "$(unique(unknown)) in `$grp_name` name no subject in the kernel's " *
            "per-subject frame. Every event row must belong to a subject this " *
            "kernel walks.")
    end
    rows = [Vector{Int}(undef, n) for n in counts]
    fill!(counts, 0)
    for (r, i) in enumerate(cell)
        rows[i][counts[i] += 1] = r
    end
    rows
end

_brm_has_missing(values::AbstractVector) = Missing <: eltype(values) && any(ismissing, values)

# The position of each value's label in `labels` (`0` when none matches, the
# last match when a label repeats), compared by `isequal`. Machine-integer
# labels over a compact range index a dense table; any others a hash table
# keyed by the labels' own element type.
function _brm_label_positions(values::AbstractVector{<:Base.BitInteger},
                              labels::AbstractVector{<:Base.BitInteger})
    isempty(labels) && return zeros(Int, length(values))
    lo, hi = extrema(labels)
    span = Int128(hi) - Int128(lo) + 1
    span <= 4 * length(labels) + 64 || return _brm_label_positions_hashed(values, labels)
    table = zeros(Int, Int(span))
    for (i, v) in enumerate(labels)
        table[Int(Int128(v) - lo) + 1] = i
    end
    positions = Vector{Int}(undef, length(values))
    for (r, v) in enumerate(values)
        positions[r] = lo <= v <= hi ? table[Int(Int128(v) - lo) + 1] : 0
    end
    positions
end
_brm_label_positions(values::AbstractVector, labels::AbstractVector) =
    _brm_label_positions_hashed(values, labels)
# Rows usually arrive grouped by label, so a row repeating the previous row's
# label reuses its lookup.
function _brm_label_positions_hashed(values, labels)
    position = Dict{eltype(labels),Int}()
    for (i, v) in enumerate(labels)
        position[v] = i
    end
    positions = Vector{Int}(undef, length(values))
    isempty(values) && return positions
    previous = first(values)
    at = get(position, previous, 0)
    for (r, v) in enumerate(values)
        if !isequal(v, previous)
            previous = v
            at = get(position, v, 0)
        end
        positions[r] = at
    end
    positions
end

# `ragged(x, group)` or `ragged(x, group, cells)` — the kernel positional a
# model-scope `ragged(x, group => cells)` lowers to.
function _brm_kernel_ragged_args(c; prefix="BRM kernel preparation")
    args = getargs(c)
    length(args) in (2, 3) || error(
        "$prefix: kernel(...) `ragged(...)` takes the thing to group, its grouping ",
        "column and optionally the plate's cell label column — got ",
        "$(length(args)) positional args.")
    args
end

# The cell label column of a plate no random-effect linear predictor labels:
# the `cells` of every `ragged(x, group => cells)` input. Its rows are the
# cells, in row order, and each row of `x` joins its cell by its `group`
# label. Without it the only order left is position — the first appearance of
# `group`'s labels — which nothing can check against a per-cell column such
# as district totals, so a differently ordered frame would bind to the wrong
# cells silently (refused at the user's direction, 2026-10-19).
function _brm_kernel_cell_label_column(ragged_cols; prefix="BRM kernel preparation")
    unlabelled = [name(first(getargs(c))) for c in ragged_cols if length(getargs(c)) == 2]
    isempty(unlabelled) || error(
        "$prefix: kernel(...) `ragged(...)` input(s) ",
        join(("`$n`" for n in unlabelled), ", "), " have no cell labels to join ",
        "against: no per-cell linear predictor carries a random effect that would ",
        "label the plate's cells. Name the cells' label vector — `x[rows[i]]` with ",
        "`rows = groups(group => cells)` in a `@plate for` cell, or ",
        "`ragged(x, group, cells)` as a `kernel(...)` positional, where `cells` lists ",
        "each cell's label once — so every entry joins its cell by label. Matching ",
        "entries to cells by position is refused: a per-cell vector ordered ",
        "differently from `group`'s first appearance would bind to the wrong cell ",
        "without any error.")
    cells = [getargs(c)[3] for c in ragged_cols]
    names = unique(name.(cells))
    length(names) == 1 || error(
        "$prefix: kernel(...) `ragged(...)` inputs name different cell label ",
        "columns (", join(("`$n`" for n in names), ", "), "); one plate has one ",
        "set of cells.")
    column = first(cells)
    (column isa NamedColumn && parent(column) isa DataColumn) || error(
        "$prefix: kernel(...) the cell label column must be a raw data column; ",
        "got $(typeof(column)).")
    column
end

# The cells of a plate no random-effect linear predictor labels, as
# `(; column, values, distinct)`: one cell per row of the cell label column
# every `ragged(x, group => cells)` input names. A `@plate for p in groups(g)`
# header lowers each read to `ragged(x, g, g)`, its grouping column as its own
# cell labels: one cell per distinct label of `g`, sorted (`distinct`). Such a
# cell has no per-cell input and collects nothing (the header refuses it), so
# nothing is matched to a cell by position (decision `1pnob5t`); `@brm`
# refuses `ragged(x, g => g)` written at model scope.
function _brm_kernel_plate_cells(ragged_cols; prefix="BRM kernel preparation")
    column = _brm_kernel_cell_label_column(ragged_cols; prefix)
    distinct = all(c -> name(getargs(c)[2]) === name(column), ragged_cols)
    values = distinct ?
        _brm_kernel_distinct_labels(parent(parent(column)), name(column); prefix) :
        _brm_kernel_subject_values(parent(parent(column)), name(column); prefix)
    (; column, values, distinct)
end

# The distinct labels of a grouping column, sorted: the cells of a plate that
# only joins rows by that column.
function _brm_kernel_distinct_labels(raw, group::Symbol; prefix="BRM kernel preparation")
    raw isa AbstractVector || error(
        "$prefix: kernel(...) grouping column `$group` must be a vector, got $(typeof(raw)).")
    values = collect(_brm_group_values(raw))
    any(ismissing, values) && error(
        "$prefix: kernel(...) grouping column `$group` has missing labels.")
    sort!(unique(values))
end

# Under a random-effect linear predictor the cells are already labelled by its
# grouping; a `ragged(x, group => cells)` must then name that same column.
function _brm_kernel_check_cell_labels(ragged_cols, group::Symbol;
                                       prefix="BRM kernel preparation")
    for c in ragged_cols
        args = getargs(c)
        length(args) == 3 && name(args[3]) !== group || continue
        error(
            "$prefix: kernel(...) `ragged($(name(args[1])), $(name(args[2])) => ",
            "$(name(args[3])))` labels the cells by `$(name(args[3]))`, but the ",
            "plate's random-effect linear predictor already labels them by ",
            "`$group`. Name `$group`, or drop `=> $(name(args[3]))`.")
    end
end

# A raw column's rows, checked against its ragged group column. An absent
# replay response (`_BRMAbsentResponse`) lies on the group column's rows.
_brm_ragged_column_length!(lens, name, values) = push!(lens, (name, length(values)))
_brm_ragged_column_length!(lens, name, ::_BRMAbsentResponse) = lens

function _brm_kernel_ragged_rows(arg_col, grp_arg, g_vals;
                                 prefix="BRM kernel preparation",
                                 allow_unbound_observation=false, order=nothing)
    a_name = name(arg_col)
    decl = parent(arg_col)
    is_lp = decl isa ExprColumn && getf(decl) === (~)
    is_unbound = decl isa MissingColumn && allow_unbound_observation
    is_lp || decl isa DataColumn || is_unbound || error(
        "$prefix: kernel(...) positional `ragged($a_name, …)`: `$a_name` must be either a ",
        "linear predictor declared in this @brm block (`$a_name ~ <terms>`) or a raw ",
        "data column; got $(typeof(decl)).")
    (grp_arg isa NamedColumn && parent(grp_arg) isa DataColumn) || error(
        "$prefix: kernel(...) `ragged($a_name, …)` needs a raw data column naming the ",
        "subject of every row of `$a_name`'s row axis; got $(typeof(grp_arg)).")
    grp_name = name(grp_arg)
    ev_vals = collect(_brm_group_values(parent(parent(grp_arg))))
    n_ev = length(ev_vals)

    # The frame `$a_name` lives on must be the one `$grp_name` describes. For an
    # LP that means every data column its terms name; for a raw column, itself.
    lens = Tuple{Symbol,Int}[]
    if is_lp
        append!(lens, _brm_kernel_lp_data_lengths(arg_col))
    elseif decl isa DataColumn
        v = parent(decl)
        v isa AbstractVector{<:AbstractVector} && error(
            "$prefix: `$a_name` read through `$grp_name`: `$a_name` is ALREADY a ",
            "ragged per-subject column, so there is nothing to group. Read it per cell ",
            "directly, `$a_name[i]`.")
        _brm_ragged_column_length!(lens, a_name, v)
    end
    _brm_kernel_check_ragged_axis(a_name, grp_name, lens, n_ev; prefix)

    rows = _brm_kernel_ragged_partition(a_name, grp_name, ev_vals, g_vals; prefix)
    if !isnothing(order)
        (order isa NamedColumn && parent(order) isa DataColumn) || error(
            "$prefix: kernel(...) `ragged($a_name, $grp_name; order=…)`: `order=` ",
            "names a raw data column on `$grp_name`'s rows; got $(typeof(order)).")
        _brm_kernel_order_rows!(rows, parent(parent(order)), name(order), grp_name, n_ev;
            prefix)
    end

    (; rows, is_lp, is_unbound, group_name=grp_name, group_values=ev_vals)
end

function _brm_kernel_check_ragged_axis(a_name, grp_name, lens, n_ev;
                                       prefix="BRM kernel preparation")
    for (nm, L) in lens
        L == n_ev || error(
            "$prefix: `$a_name` read through `$grp_name`: `$a_name` has $L entries ",
            "(data vector `$nm`) but `$grp_name` has $n_ev. The grouping column must ",
            "name the subject of EVERY entry of `$a_name`.")
    end
    nothing
end

# The same partition computed in the RK graph from the bound columns (emitted
# by `_rk_emit_kernel_plate!`), so a design's row geometry is a graph value of
# its raw columns rather than a planned input. `names` is
# `[x, group, cell, axis...]` (plus the `order` column before the axes in the
# ordered form); `axes` are the row-wise data columns of `x`'s own frame, each
# checked against `groups` as `_brm_kernel_ragged_rows` checks them.
function brm_ragged_rows(names, groups, cells, axes...)
    x, group, cell = names[1], names[2], names[3]
    _brm_ragged_rows_graph(x, group, groups, cell, cells, names[4:end], axes)
end
function brm_ragged_rows_ordered(names, groups, cells, order, axes...)
    x, group, cell, order_name = names[1], names[2], names[3], names[4]
    rows = _brm_ragged_rows_graph(x, group, groups, cell, cells, names[5:end], axes)
    _brm_kernel_order_rows!(rows, order, order_name, group, length(groups);
        prefix="RK backend")
end
function _brm_ragged_rows_graph(x, group, groups, cell, cells, axis_names, axes)
    prefix = "RK backend"
    labels = _brm_kernel_subject_values(cells, cell; prefix)
    ev_vals = _brm_group_values(groups)
    _brm_kernel_check_ragged_axis(x, group,
        Tuple{Symbol,Int}[(nm, length(a)) for (nm, a) in zip(axis_names, axes)],
        length(ev_vals); prefix)
    _brm_kernel_ragged_partition(x, group, ev_vals, labels; prefix)
end

# `ragged(x, group; order=time)`: each cell's rows in ascending `time`, ties in
# row order.
function _brm_kernel_order_rows!(rows, times, order::Symbol, group::Symbol, nrows;
                                 prefix="BRM kernel preparation")
    got = times isa AbstractVector ? "$(length(times)) rows" : typeof(times)
    times isa AbstractVector && length(times) == nrows || error(
        "$prefix: kernel(...) `ragged(…, $group; order=$order)`: `$order` must be a ",
        "column on `$group`'s $nrows rows; got $got.")
    any(ismissing, times) && error(
        "$prefix: kernel(...) `ragged(…, $group; order=$order)`: `$order` has missing values.")
    for cell in rows
        sort!(cell; by=r -> times[r], alg=Base.Sort.DEFAULT_STABLE)
    end
    rows
end

# The `into=` target and `order=` column of a `ragged(...)` kernel positional,
# each `nothing` when absent.
function _brm_kernel_ragged_options(c; prefix="BRM kernel preparation")
    kwargs = getkwargs(c)
    unknown = [k for k in keys(kwargs) if !(k in (:into, :order))]
    isempty(unknown) || error(
        "$prefix: kernel(...) `ragged(...)` accepts only the `into=` and `order=` ",
        "keywords; got ", join(("`$k`" for k in unknown), ", "), ".")
    (; into=get(kwargs, :into, nothing), order=get(kwargs, :order, nothing))
end

# `ragged(idx, group; into=target)`: `idx` holds 1-based row numbers on the flat
# axis whose subjects `target` labels, with `0` meaning "no row". Each becomes
# its position among that subject's rows of `target`, in flat order — the order
# a `ragged(x, target)` gather builds — so a cell indexes its own subject's
# slice of the target axis. Flat in, flat out: the result is then gathered by
# `group` exactly like any other data column. An index into another subject's
# rows would read the wrong subject silently, so it is refused.
function _brm_kernel_ragged_index_values(arg_col, grp_arg, into_col;
                                         prefix="BRM kernel preparation")
    idx_name, grp_name = name(arg_col), name(grp_arg)
    parent(arg_col) isa DataColumn || error(
        "$prefix: kernel(...) `ragged($idx_name, $grp_name; into=…)`: `into=` ",
        "rebases a data column of row numbers; `$idx_name` is not a raw data column.")
    (into_col isa NamedColumn && parent(into_col) isa DataColumn) || error(
        "$prefix: kernel(...) `ragged($idx_name, $grp_name; into=…)`: `into=` must ",
        "name a raw data column labelling the subject of every row of the indexed ",
        "axis; got $(typeof(into_col)).")
    _brm_kernel_rebase_index(idx_name, parent(parent(arg_col)),
        grp_name, parent(parent(grp_arg)), name(into_col), parent(parent(into_col));
        prefix)
end

function _brm_kernel_rebase_index(idx_name::Symbol, idx, grp_name::Symbol, raw_groups,
                                  into_name::Symbol, raw_targets;
                                  prefix="BRM kernel preparation")
    decl = "`ragged($idx_name, $grp_name; into=$into_name)`"
    idx isa AbstractVector{<:Integer} && !(eltype(idx) <: Bool) || error(
        "$prefix: kernel(...) $decl: `$idx_name` must be a flat integer column of ",
        "row numbers; got $(typeof(idx)).")
    raw_targets isa AbstractVector || error(
        "$prefix: kernel(...) $decl: `$into_name` must be a vector; got ",
        "$(typeof(raw_targets)).")
    groups = collect(_brm_group_values(raw_groups))
    targets = collect(_brm_group_values(raw_targets))
    length(idx) == length(groups) || error(
        "$prefix: kernel(...) $decl: `$idx_name` has $(length(idx)) rows but ",
        "`$grp_name` has $(length(groups)).")
    any(ismissing, targets) && error(
        "$prefix: kernel(...) $decl: `$into_name` has missing labels.")
    seen = Dict{Any,Int}()
    position = [seen[label] = get(seen, label, 0) + 1 for label in targets]
    map(eachindex(idx)) do k
        r = idx[k]
        r == 0 && return 0
        1 <= r <= length(targets) || error(
            "$prefix: kernel(...) $decl: row $k of `$idx_name` is $r, outside the ",
            "$(length(targets)) rows of `$into_name`.")
        isequal(targets[r], groups[k]) || error(
            "$prefix: kernel(...) $decl: row $k of `$idx_name` belongs to subject ",
            "$(repr(groups[k])) but points at row $r of `$into_name`, which belongs ",
            "to subject $(repr(targets[r])). An index must stay inside its own subject.")
        position[r]
    end
end
