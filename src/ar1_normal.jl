# ---- `groups(g; order = t)` and `AR1Normal(...)` in a `@plate` cell ---------
#
# `groups` is an ordinary function: the indices of each group. In
# `@plate for p in groups(probe; order = day)` the cell's `p` is one group's
# indices, so a read `x[p]` means what the same Julia loop computes (decision
# `1pnob5t`, user 2026-10-21: no macro-only magic in the grouping).
#
# `y[p] ~ AR1Normal(mu[p], sigma[p], phi; time = day[p])` observes the cell's
# series under AR(1) residuals. Until a called submodel can hold the observed
# `@scan` on both backends (StanBlocks snag `scan-inside-call-2ea17cef`,
# ReactiveKernels snag `per-cell-submode-cd87e071`), `@brm` replaces the
# statement by the `@scan` it states — the temporary rewrite the user accepted
# (2026-10-21) while those land:
#
#     @scan begin
#         y[p][1] ~ Normal(mu[p][1], sigma[p][1] / sqrt(1 - phi ^ 2))
#         for t in 2:length(day[p])
#             k = day[p][t] - day[p][t - 1]
#             y[p][t] ~ Normal(mu[p][t] + phi ^ k * (y[p][t - 1] - mu[p][t - 1]),
#                              sigma[p][t] * sqrt((1 - phi ^ (2 * k)) / (1 - phi ^ 2)))
#         end
#     end
#
# `show` prints the statement as written; the backends and the descriptor see
# the expansion. `sigma` is the innovation SD and the first entry comes from
# the stationary law; a gap of `k` time steps uses `phi ^ k` with the matching
# innovation variance. That is the exact density of the AR(1) process observed
# on the recorded steps. Without `time` every step is one (`k = 1`): the plain
# recursion.
#
# An argument that reads the group's rows — `mu[p]`, `mu[p] .+ offset`, or a
# cell local computed from one — is indexed entry by entry; any other argument
# (`phi`, `exp(log_sigma)`, a number) is shared by every entry.

"""
    groups(g; order = nothing)

The indices of each group of `g`: for each distinct value of `g`, in sorted
order of the values, the indices of `g` that hold it. They are in ascending
`order` (a vector indexed like `g`) when it is given, ties in index order, and
in index order otherwise.

```julia
julia> groups(["b", "a", "b", "a"]; order = [4, 9, 1, 2])
2-element Vector{Vector{Int64}}:
 [4, 2]
 [3, 1]
```

In a `@brm` model, `@plate for p in groups(probe; order = day)` has one cell per
group of the data vector `probe`, and `p` is that group's indices: a read
`x[p]` of a value indexed like `probe` is `x` at those indices in that order,
as the same loop computes in Julia.
See [`kernel`](@ref) and [`AR1Normal`](@ref).
"""
function groups(g::AbstractVector; order::Union{Nothing,AbstractVector}=nothing)
    labels = collect(_brm_group_values(g))
    any(ismissing, labels) && throw(ArgumentError("groups: `g` has missing values"))
    cells = sort!(unique(labels))
    position = Dict(label => j for (j, label) in enumerate(cells))
    rows = [Int[] for _ in cells]
    for (r, label) in enumerate(labels)
        push!(rows[position[label]], r)
    end
    if order !== nothing
        length(order) == length(labels) || throw(DimensionMismatch(
            "groups: `order` has $(length(order)) values for $(length(labels)) in `g`"))
        any(ismissing, order) && throw(ArgumentError("groups: `order` has missing values"))
        foreach(cell -> sort!(cell; by=r -> order[r], alg=Base.Sort.DEFAULT_STABLE), rows)
    end
    rows
end

"""
    @plate for p in groups(g; order = t)
        y[p] ~ AR1Normal(mu[p], sigma, phi; time = t[p])
    end

Gaussian observations whose residuals follow an AR(1) process along each
group's series, in the order of the integer time column `t`:
`y[s] - mu[s] = phi^k * (y[s-k] - mu[s-k]) + e[s]`, where `k` is the number of
time steps since the previous entry. The plate is the grouping:
[`groups`](@ref)`(g; order = t)` gives one cell per group, with its rows in `t`
order.

- `sigma` is the innovation SD. The first entry comes from the stationary law
  `Normal(mu, sigma / sqrt(1 - phi^2))`, so `|phi| < 1` is required. This is
  the exact AR(1) likelihood (brms' `ar(cov = TRUE)`).
- A gap of `k` steps uses `phi^k` and innovation variance
  `sigma^2 (1 - phi^(2k)) / (1 - phi^2)`: the exact density of the process
  observed only on the recorded steps. Each series' times must be whole
  numbers, each at most once; the model refuses others when it is built,
  naming the group and value.
- Without `time`, consecutive rows are one step apart: the plain recursion.
- An argument that reads the group's rows (`mu[p]`, `sigma[p]`, or a cell local
  computed from one) varies along the series; any other argument is shared.
  `log(sigma) ~ 1 + (1 | g)` read as `sigma[p]`, or `atanh(phi) ~ 1 + (1 | g)`
  read as `phi[p]`, lets the scale or the persistence vary by group.

For now `@brm` evaluates the statement as the `@scan` recurrence it states, in
the cell, and the descriptor and draws show that recurrence: each entry's
pointwise log likelihood is its conditional density given the series' earlier
entries, and with `y` omitted the cell draws whole trajectories. `show` prints
the statement as written.

```julia
@scan begin
    y[p][1] ~ Normal(mu[p][1], sigma / sqrt(1 - phi ^ 2))
    for s in 2:length(t[p])
        k = t[p][s] - t[p][s - 1]
        y[p][s] ~ Normal(mu[p][s] + phi ^ k * (y[p][s - 1] - mu[p][s - 1]),
                         sigma * sqrt((1 - phi ^ (2 * k)) / (1 - phi ^ 2)))
    end
end
```

Order 1 only. For another innovation family, a higher order or a different
start or gap rule, write the `@scan` yourself (see [`kernel`](@ref)).
"""
function AR1Normal end

_is_brm_ar1_normal(x) = isxcall(x, :~) && isxcall(x.args[3], :AR1Normal)

# The `@plate` with every `AR1Normal` observation of its cell replaced by its
# `@scan`, and the build-time checks of their time columns.
function _brm_expand_cell_ar1_normal(plate)
    i, range, body = _brm_plate_loop(plate)
    stmts = Meta.isexpr(body, :block) ? body.args : Any[body]
    any(_is_brm_ar1_normal, stmts) || return plate, Any[]
    header = _brm_plate_header_grouping(range)
    # Every read in a `groups(...)` cell is a group's rows, so an argument's
    # spelling says whether it varies along the series.
    header === nothing && error(
        "@brm: `AR1Normal(...)` observes the rows of a `@plate for $i in groups(g; " *
        "order = t)` cell; in a `@plate for $i in $(sprint(Base.show_unquoted, range))` " *
        "cell write the `@scan` it stands for (see the `AR1Normal` docstring)")
    locals = Dict{Symbol,Any}(stmt.args[1] => stmt.args[2] for stmt in stmts
        if Meta.isexpr(stmt, :(=), 2) && stmt.args[1] isa Symbol)
    checks = Any[]
    expanded = map(stmts) do stmt
        _is_brm_ar1_normal(stmt) || return stmt
        scan, time = _brm_cell_ar1_normal_scan(stmt, i, locals, stmts, header)
        time === nothing || push!(checks,
            :($_brm_check_ar1_normal_time($(QuoteNode(stmt.args[2].args[1])),
                $time, $(header.group), $(something(header.order, :nothing)))))
        scan
    end
    loop = Expr(:for, plate.args[3].args[1], Expr(:block, expanded...))
    Expr(:macrocall, plate.args[1], plate.args[2], loop), checks
end

function _brm_cell_ar1_normal_scan(stmt, i, locals, stmts, header)
    lhs, rhs = stmt.args[2], stmt.args[3]
    source(x) = sprint(Base.show_unquoted, x)
    shown = source(stmt)
    _brm_plate_indexed(lhs, i) && lhs.args[1] isa Symbol || error(
        "@brm: `AR1Normal(...)` observes a response read as `y[$i]` in its cell; got `$shown`")
    positional, keywords = Any[], Dict{Symbol,Any}()
    for arg in rhs.args[2:end]
        for a in (Meta.isexpr(arg, :parameters) ? arg.args : Any[arg])
            Meta.isexpr(a, :kw, 2) ? (keywords[a.args[1]] = a.args[2]) : push!(positional, a)
        end
    end
    haskey(keywords, :by) && error(
        "@brm: `AR1Normal(...)` takes no `by=`: the plate is the grouping. Write " *
        "`@plate for $i in groups(g; order = t)` and observe `y[$i] ~ AR1Normal(...)` " *
        "in its cell; got `$shown`")
    length(positional) == 3 && all(==(:time), keys(keywords)) || error(
        "@brm: write `y[$i] ~ AR1Normal(mu, sigma, phi)` or `y[$i] ~ AR1Normal(mu, " *
        "sigma, phi; time = t[$i])`; got `$shown`")
    mu, sigma, phi = positional
    time = get(keywords, :time, nothing)
    time === nothing || _brm_plate_indexed(time, i) && time.args[1] isa Symbol || error(
        "@brm: `AR1Normal(...; time = t[$i])` reads the cell's time column; got `$shown`")

    used = Set{Symbol}(keys(locals))
    push!(used, i)
    _brm_cell_value_refs!(used, Expr(:block, stmts...))
    fresh(name) = (while name in used; name = Symbol(name, :_); end; push!(used, name); name)
    t, k = fresh(:t), fresh(:k)

    # Whether `value` reads a group's rows, directly or through cell locals.
    series(value, seen=Set{Symbol}()) =
        value isa Symbol ? (haskey(locals, value) && !(value in seen) &&
            series(locals[value], push!(seen, value))) :
        value isa Expr && (_brm_plate_indexed(value, i) ||
            any(arg -> series(arg, seen), value.args))
    reader(value) = !series(value) ? (_ -> value) :
        value isa Symbol || _brm_plate_indexed(value, i) ? (index -> :($value[$index])) :
        (index -> Expr(:ref, value, index))
    m, s, f = reader(mu), reader(sigma), reader(phi)
    obs(index) = :($lhs[$index])
    previous = :($t - 1)
    seed = :($(obs(1)) ~ Normal($(m(1)), $(s(1)) / sqrt(1 - $(f(1)) ^ 2)))
    # The series' length, read off data where the cell has it: with `y`
    # omitted, the cell draws the series and `y[p]` has no length to read.
    arguments = (mu, sigma, phi)
    along = findfirst(series, arguments)
    span = time !== nothing ? time :
        header.order !== nothing ? :($(header.order)[$i]) :
        along !== nothing ? arguments[along] : lhs
    steps = if time === nothing
        Expr(:for, :($t = 2:length($span)), Expr(:block,
            :($(obs(t)) ~ Normal($(m(t)) + $(f(t)) * ($(obs(previous)) - $(m(previous))),
                $(s(t))))))
    else
        Expr(:for, :($t = 2:length($time)), Expr(:block,
            :($k = $time[$t] - $time[$previous]),
            :($(obs(t)) ~ Normal(
                $(m(t)) + $(f(t)) ^ $k * ($(obs(previous)) - $(m(previous))),
                $(s(t)) * sqrt((1 - $(f(t)) ^ (2 * $k)) / (1 - $(f(t)) ^ 2))))))
    end
    scan = Expr(:macrocall, Symbol("@scan"), LineNumberNode(0), Expr(:block, seed, steps))
    scan, time === nothing ? nothing : time.args[1]
end

# The entries of each series are distinct whole time steps in increasing
# order, as the cell reads them: a repeated time has no AR(1) step, and the gap
# rule `phi ^ k` needs whole steps. The cell reads each group's rows of `t` in
# the order `groups` gives them.
function _brm_check_ar1_normal_time(response, time::NamedColumn, group::NamedColumn, order)
    times = parent(time)
    times isa DataColumn || error(
        "@brm: `$response ~ AR1Normal(...; time = $(name(time))[…])` needs `$(name(time))` " *
        "as a data column")
    values = parent(times)
    labels = collect(_brm_group_values(parent(parent(group))))
    rows = groups(labels; order=order isa NamedColumn ? parent(parent(order)) : nothing)
    for cell in rows
        label = labels[first(cell)]
        series = values[cell]
        for (a, b) in zip(series, Iterators.drop(series, 1))
            b isa Real && isinteger(b) || error(
                "@brm: `AR1Normal` time column `$(name(time))` holds $(repr(b)) in group " *
                "$(repr(label)); time advances in whole steps")
            b > a || error(
                "@brm: `AR1Normal` time column `$(name(time))` " *
                (b == a ? "repeats time $(repr(b))" : "goes back from $(repr(a)) to $(repr(b))") *
                " in group $(repr(label)); each group observes each time once, in order")
        end
        first(series) isa Real && isinteger(first(series)) || error(
            "@brm: `AR1Normal` time column `$(name(time))` holds $(repr(first(series))) in " *
            "group $(repr(label)); time advances in whole steps")
    end
    nothing
end
