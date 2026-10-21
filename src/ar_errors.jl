# ---- `ar_errors(...)`: AR(1) residuals of a cell's series, as a `@scan` -----
#
# Inside a `@plate` cell, `y[p] ~ ar_errors(Normal(mu[p], sigma), phi; time =
# day[p])` is shorthand for the recurrence it states (user decisions `0det9mo`,
# G+S, and `1pnob5t`: the grouping belongs to the plate, so the shorthand takes
# no `by=`). The cell rewrite replaces it by
#
#     @scan begin
#         y[p][1] ~ Normal(mu[p][1], sigma / sqrt(1 - phi ^ 2))
#         for t in 2:length(day[p])
#             k = day[p][t] - day[p][t - 1]
#             y[p][t] ~ Normal(mu[p][t] + phi ^ k * (y[p][t - 1] - mu[p][t - 1]),
#                              sigma * sqrt((1 - phi ^ (2 * k)) / (1 - phi ^ 2)))
#         end
#     end
#
# before anything else reads the cell, so `show` and the descriptor print it.
# `sigma` is the innovation SD and the first entry comes from the stationary
# law; a gap of `k` time steps uses `phi ^ k` with the matching innovation
# variance. That is the exact density of the AR(1) process observed on the
# recorded steps, and the plain recursion when no step is skipped.
#
# The cell's own spelling says what varies along the series. In a
# `groups(g; order = t)` cell every read `x[p]` is the group's rows of `x` (a
# response, a data column, a row-level predictor), so an argument that reads
# one — `mu[p]`, `mu[p] .+ offset`, or a cell local computed from one — is
# indexed entry by entry; any other argument (`sigma`, `exp(log_sigma)`, a
# number) is shared by every entry.

_is_brm_ar_errors(x) = isxcall(x, :~) && isxcall(x.args[3], :ar_errors)

# The `@plate` with every `ar_errors` observation of its cell replaced by its
# `@scan`, and the build-time checks of their time columns.
function _brm_expand_cell_ar_errors(plate)
    i, range, body = _brm_plate_loop(plate)
    stmts = Meta.isexpr(body, :block) ? body.args : Any[body]
    any(_is_brm_ar_errors, stmts) || return plate, Any[]
    header = _brm_plate_header_grouping(range)
    # The shorthand states the per-group recurrence over long-format rows
    # (decision `0det9mo`); a cell over pre-grouped data writes its `@scan`.
    header === nothing && error(
        "@brm: `ar_errors(...)` observes the rows of a `@plate for $i in groups(g; " *
        "order = t)` cell; in a `@plate for $i in $(sprint(Base.show_unquoted, range))` " *
        "cell write the `@scan` it stands for (see the `ar_errors` docstring)")
    locals = Dict{Symbol,Any}(stmt.args[1] => stmt.args[2] for stmt in stmts
        if Meta.isexpr(stmt, :(=), 2) && stmt.args[1] isa Symbol)
    checks = Any[]
    expanded = map(stmts) do stmt
        _is_brm_ar_errors(stmt) || return stmt
        scan, time = _brm_cell_ar_errors_scan(stmt, i, locals, stmts)
        push!(checks, :($_brm_check_ar_errors_time($(QuoteNode(stmt.args[2].args[1])),
            $time, $(header.group), $(something(header.order, :nothing)))))
        scan
    end
    loop = Expr(:for, plate.args[3].args[1], Expr(:block, expanded...))
    Expr(:macrocall, plate.args[1], plate.args[2], loop), checks
end

function _brm_cell_ar_errors_scan(stmt, i, locals, stmts)
    lhs, rhs = stmt.args[2], stmt.args[3]
    source(x) = sprint(Base.show_unquoted, x)
    shown = source(stmt)
    _brm_plate_indexed(lhs, i) && lhs.args[1] isa Symbol || error(
        "@brm: `ar_errors(...)` observes a response read as `y[$i]` in its cell; got `$shown`")
    positional, keywords = Any[], Dict{Symbol,Any}()
    for arg in rhs.args[2:end]
        for a in (Meta.isexpr(arg, :parameters) ? arg.args : Any[arg])
            Meta.isexpr(a, :kw, 2) ? (keywords[a.args[1]] = a.args[2]) : push!(positional, a)
        end
    end
    haskey(keywords, :by) && error(
        "@brm: `ar_errors(...)` takes no `by=`: the plate is the grouping. Write " *
        "`@plate for $i in groups(g; order = t)` and observe `y[$i] ~ ar_errors(...; " *
        "time = t[$i])` in its cell; got `$shown`")
    length(positional) == 2 && collect(keys(keywords)) == [:time] || error(
        "@brm: write `y[$i] ~ ar_errors(Normal(mu, sigma), phi; time = t[$i])`; got `$shown`")
    family, phi = positional
    time = keywords[:time]
    _brm_plate_indexed(time, i) && time.args[1] isa Symbol || error(
        "@brm: `ar_errors(...; time = t[$i])` reads the cell's time column; got `$shown`")
    isxcall(family, :Normal) && length(family.args) == 3 || error(
        "@brm: `ar_errors(...)` takes Gaussian innovations, `Normal(mu, sigma)`; got " *
        "`$(source(family))`. Write the `@scan` recurrence for another family " *
        "(see the `ar_errors` docstring).")
    mu, sigma = family.args[2], family.args[3]

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
    gap = :($k = $time[$t] - $time[$previous])
    step = :($(obs(t)) ~ Normal(
        $(m(t)) + $(f(t)) ^ $k * ($(obs(previous)) - $(m(previous))),
        $(s(t)) * sqrt((1 - $(f(t)) ^ (2 * $k)) / (1 - $(f(t)) ^ 2))))
    scan = Expr(:macrocall, Symbol("@scan"), LineNumberNode(0), Expr(:block,
        seed, Expr(:for, :($t = 2:length($time)), Expr(:block, gap, step))))
    scan, time.args[1]
end

# The entries of each series are distinct whole time steps in increasing
# order, as the cell reads them: a repeated time has no AR(1) step, and the gap
# rule `phi ^ k` needs whole steps. A `groups(g; order = t)` cell reads each
# group's rows of `t`, sorted by its `order` column when it has one.
function _brm_check_ar_errors_time(response, time::NamedColumn, args...)
    times = parent(time)
    times isa DataColumn || error(
        "@brm: `$response ~ ar_errors(...; time = $(name(time))[…])` needs `$(name(time))` " *
        "as a data column")
    series, labels = _brm_ar_errors_series(parent(times), args...)
    for (values, label) in zip(series, labels)
        for (a, b) in zip(values, Iterators.drop(values, 1))
            b isa Real && isinteger(b) || error(
                "@brm: `ar_errors` time column `$(name(time))` holds $(repr(b)) in group " *
                "$(repr(label)); time advances in whole steps")
            b > a || error(
                "@brm: `ar_errors` time column `$(name(time))` " *
                (b == a ? "repeats time $(repr(b))" : "goes back from $(repr(a)) to $(repr(b))") *
                " in group $(repr(label)); each group observes each time once, in order")
        end
        isempty(values) || (first(values) isa Real && isinteger(first(values))) || error(
            "@brm: `ar_errors` time column `$(name(time))` holds $(repr(first(values))) in " *
            "group $(repr(label)); time advances in whole steps")
    end
    nothing
end
# Each group's rows, sorted by the header's `order` column when it has one.
function _brm_ar_errors_series(times::AbstractVector, group::NamedColumn, order)
    labels = collect(_brm_group_values(parent(parent(group))))
    keys = order isa NamedColumn ? parent(parent(order)) : eachindex(times)
    cells = sort!(unique(labels))
    rows = [sort!(findall(isequal(c), labels); by=r -> keys[r], alg=Base.Sort.DEFAULT_STABLE)
        for c in cells]
    [times[r] for r in rows], cells
end
