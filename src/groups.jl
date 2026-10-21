# ---- `groups`: the indices of each group --------------------------------------
#
# An ordinary function, and the whole meaning of a grouped `@plate` read
# (decision `1pnob5t`, todo `1nfukuv`, decision `0ivno7u`): a cell reads a
# vector at a group's indices, `x[p]` or `x[events[i]]`, exactly as the same
# Julia loop does. `@brm` lowers those reads to its kernel inputs; it does not
# give them another meaning.

"""
    groups(g; order = nothing)
    groups(g => labels; order = nothing)

The indices of each group of the vector `g`.

`groups(g)` has one group per distinct value of `g`, in sorted order of the
values. `groups(g => labels)` has one group per entry of `labels`, in that
order: the indices of `g` that hold that label (none, if no entry of `g`
does). Within a group the indices are in ascending `order` (a vector indexed
like `g`) when it is given, ties in index order, and in index order otherwise.

```julia
julia> groups(["b", "a", "b", "a"]; order = [4, 9, 1, 2])
2-element Vector{Vector{Int64}}:
 [4, 2]
 [3, 1]

julia> groups(["b", "a", "b", "a"] => ["b", "c", "a"])
3-element Vector{Vector{Int64}}:
 [1, 3]
 []
 [2, 4]
```

In a `@brm` model, `@plate for p in groups(probe; order = day)` has one cell per
group of the data vector `probe`, and `p` is that group's indices: a read
`x[p]` of a value indexed like `probe` is `x` at those indices in that order.
A plate over subjects reads another vector's entries per subject through a
model-scope binding, `events = groups(event_subject => subject)`, as
`x[events[i]]`. Either way the cell means what the same loop computes in
Julia. See [`kernel`](@ref).
"""
groups(g::AbstractVector; order::Union{Nothing,AbstractVector}=nothing) =
    _brm_groups(g, sort!(unique(_brm_groups_labels(g))), order)
groups(spec::Pair{<:AbstractVector,<:AbstractVector};
       order::Union{Nothing,AbstractVector}=nothing) =
    _brm_groups(spec.first, collect(_brm_group_values(spec.second)), order)

function _brm_groups_labels(g)
    labels = collect(_brm_group_values(g))
    any(ismissing, labels) && throw(ArgumentError("groups: `g` has missing values"))
    labels
end

function _brm_groups(g, cells, order)
    labels = _brm_groups_labels(g)
    position = Dict{Any,Int}()
    for (j, label) in enumerate(cells)
        haskey(position, label) && throw(ArgumentError(
            "groups: the label $(repr(label)) is listed twice"))
        position[label] = j
    end
    indices = [Int[] for _ in cells]
    for (r, label) in enumerate(labels)
        j = get(position, label, 0)
        j == 0 && throw(ArgumentError(
            "groups: `g` holds $(repr(label)), which is not among the labels"))
        push!(indices[j], r)
    end
    if order !== nothing
        length(order) == length(labels) || throw(DimensionMismatch(
            "groups: `order` has $(length(order)) values for $(length(labels)) in `g`"))
        any(ismissing, order) && throw(ArgumentError("groups: `order` has missing values"))
        foreach(cell -> sort!(cell; by=r -> order[r], alg=Base.Sort.DEFAULT_STABLE), indices)
    end
    indices
end

# Plain `indexin(idx[rows[i]], target[i])` has no position for an index `0`.
function _brm_check_indexin_zero(shown, idx::NamedColumn)
    values = parent(idx) isa DataColumn ? parent(parent(idx)) : nothing
    values isa AbstractVector || error(
        "@brm: `$shown` reads `$(name(idx))` as indices; it must be a data vector")
    at = findfirst(isequal(0), values)
    at === nothing || error(
        "@brm: `$shown`: `$(name(idx))[$at]` is 0, which is no index and has no " *
        "position; write `something.(indexin(…), 0)` to keep 0 as \"no index\"")
    nothing
end
