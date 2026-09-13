# Group identity and event-row partitions shared by kernel emitters.
# This layer owns no SLIC statements or backend data cleanup.
_brm_group_values(raw::CA.CategoricalVector) = CA.unwrap.(raw)
_brm_group_values(raw::AbstractVector) = raw

_brm_collect_data_lengths!(_acc, _x) = nothing
_brm_collect_data_lengths!(acc, x::NamedColumn) = begin
    p = parent(x)
    p isa DataColumn ? push!(acc, (name(x), length(parent(p)))) :
        _brm_collect_data_lengths!(acc, p)
    nothing
end
_brm_collect_data_lengths!(acc, x::ExprColumn) = begin
    getf(x) === (~) && return nothing
    for a in getargs(x); _brm_collect_data_lengths!(acc, a); end
    for v in values(getkwargs(x)); _brm_collect_data_lengths!(acc, v); end
    nothing
end
_brm_collect_data_lengths!(acc, x::Union{Tuple,AbstractVector}) = begin
    for a in x; _brm_collect_data_lengths!(acc, a); end
    nothing
end

function _brm_kernel_subject_values(raw, group::Symbol; prefix="BRM kernel preparation")
    raw isa AbstractVector || error(
        "$prefix: kernel(...) subject grouping column `$group` must be a vector, " *
        "got $(typeof(raw)).")
    values = collect(_brm_group_values(raw))
    (!isempty(values) && !any(ismissing, values) &&
     length(unique(values)) == length(values)) || error(
        "$prefix: kernel(...) needs pre-grouped per-subject data — `$group` must " *
        "list one non-missing unique subject per row. Repeated levels indicate " *
        "long-format data. Got $(values).")
    values
end

function _brm_kernel_ragged_partition(a_name::Symbol, grp_name::Symbol,
                                     ev_vals::AbstractVector,
                                     subject_vals::AbstractVector; prefix="BRM kernel preparation")
    (!isempty(ev_vals) && !any(ismissing, ev_vals)) || error(
        "$prefix: kernel(...) `ragged($a_name, $grp_name)`: `$grp_name` must be a " *
        "non-empty column with no missing labels.")
    pos = Dict{Any,Int}()
    for (i, v) in enumerate(subject_vals); pos[v] = i; end
    rows = [Int[] for _ in eachindex(subject_vals)]
    unknown = Any[]
    for (r, v) in enumerate(ev_vals)
        i = get(pos, v, 0)
        i == 0 ? push!(unknown, v) : push!(rows[i], r)
    end
    isempty(unknown) || error(
        "$prefix: kernel(...) `ragged($a_name, $grp_name)`: label(s) " *
        "$(unique(unknown)) in `$grp_name` name no subject in the kernel's " *
        "per-subject frame. Every event row must belong to a subject this " *
        "kernel walks.")
    rows
end

function _brm_kernel_ragged_rows(arg_col, grp_arg, g_vals; prefix="BRM kernel preparation")
    a_name = name(arg_col)
    decl = parent(arg_col)
    is_lp = decl isa ExprColumn && getf(decl) === (~)
    is_lp || decl isa DataColumn || error(
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
        _brm_collect_data_lengths!(lens, getargs(decl)[2])
    else
        v = parent(decl)
        v isa AbstractVector{<:AbstractVector} && error(
            "$prefix: kernel(...) `ragged($a_name, $grp_name)`: `$a_name` is ALREADY a ",
            "ragged per-subject column, so there is nothing to group. Pass it directly, ",
            "without `ragged(...)`.")
        push!(lens, (a_name, length(v)))
    end
    for (nm, L) in lens
        L == n_ev || error(
            "$prefix: kernel(...) `ragged($a_name, $grp_name)`: `$a_name` is declared ",
            "over a $L-row axis (data column `$nm`) but `$grp_name` has $n_ev rows. ",
            "The grouping column must name the subject of EVERY row of ",
            "`$a_name`'s own frame.")
    end

    rows = _brm_kernel_ragged_partition(a_name, grp_name, ev_vals, g_vals; prefix)

    (; rows, is_lp, group_name=grp_name, group_values=ev_vals)
end
