# ---- `AR1Normal(...)` in a `@plate for p in groups(...)` cell ----------------
#
# `y[p] ~ AR1Normal(mu[p], sigma[p], phi; time = day[p])` observes the cell's
# series under AR(1) residuals. `@brm` keeps the statement as written and
# checks it; each backend calls its own BRM-owned submodel holding the
# observed `@scan` (`_sb_ar1_normal` in `sb_cell_julia.jl`,
# `_RK_AR1_NORMAL_SUBMODELS` in `rk_kernel_values.jl`), the same recurrence on
# both:
#
#     @scan begin
#         y[1] ~ Normal(mu[1], sigma[1] / sqrt(1 - phi[1] ^ 2))
#         for t in 2:length(time)
#             k = time[t] - time[t - 1]
#             y[t] ~ Normal(mu[t] + phi[t] ^ k * (y[t - 1] - mu[t - 1]),
#                           sigma[t] * sqrt((1 - phi[t] ^ (2 * k)) / (1 - phi[t] ^ 2)))
#         end
#     end
#
# `sigma` is the innovation SD and the first entry comes from the stationary
# law; a gap of `k` time steps uses `phi ^ k` with the matching innovation
# variance. That is the exact density of the AR(1) process observed on the
# recorded steps. Without `time` every step is one (`k = 1`): the plain
# recursion.
#
# An argument that reads the group's rows — `mu[p]`, `mu[p] .+ offset`, or a
# cell local computed from one — varies along the series; any other argument
# (`phi`, `exp(log_sigma)`, a number) is shared by every entry and reaches the
# submodel filled to the series' length.

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

`@brm` keeps the statement as written, and each backend calls a BRM-owned
submodel holding the recurrence (an `@slic` submodel on SBBRMI, an `@rkppl`
one on RKBRMI), the cell's series passed as its arguments:

```julia
@scan begin
    y[1] ~ Normal(mu[1], sigma[1] / sqrt(1 - phi[1] ^ 2))
    for s in 2:length(time)
        k = time[s] - time[s - 1]
        y[s] ~ Normal(mu[s] + phi[s] ^ k * (y[s - 1] - mu[s - 1]),
                      sigma[s] * sqrt((1 - phi[s] ^ (2 * k)) / (1 - phi[s] ^ 2)))
    end
end
```

A shared argument reaches it filled to the series' length. `show` and the
descriptor print the statement; the generative plan declares an `AR1Normal`
observation of `y`. Each entry's pointwise log likelihood is its conditional
density given the series' earlier entries, and with `y` omitted the cell draws
whole trajectories.

Order 1 only. For another innovation family, a higher order or a different
start or gap rule, write the `@scan` yourself (see [`kernel`](@ref)).
"""
function AR1Normal end

_is_brm_ar1_normal(x) = isxcall(x, :~) && isxcall(x.args[3], :AR1Normal)

# The positional and keyword arguments of an `AR1Normal(...)` call.
function _brm_ar1_normal_arguments(call)
    positional, keywords = Any[], Dict{Symbol,Any}()
    for arg in call.args[2:end]
        for a in (Meta.isexpr(arg, :parameters) ? arg.args : Any[arg])
            Meta.isexpr(a, :kw, 2) ? (keywords[a.args[1]] = a.args[2]) : push!(positional, a)
        end
    end
    positional, keywords
end

# The build-time checks of a `@plate`'s `AR1Normal` observations, which stay
# in the cell as written: one per statement with a time column.
function _brm_cell_ar1_normal_checks(plate)
    i, range, body = _brm_plate_loop(plate)
    stmts = Meta.isexpr(body, :block) ? body.args : Any[body]
    any(_is_brm_ar1_normal, stmts) || return Any[]
    header = _brm_plate_header_grouping(range)
    # Every read in a `groups(...)` cell is a group's rows, so an argument's
    # spelling says whether it varies along the series.
    header === nothing && error(
        "@brm: `AR1Normal(...)` observes the rows of a `@plate for $i in groups(g; " *
        "order = t)` cell; in a `@plate for $i in $(sprint(Base.show_unquoted, range))` " *
        "cell write the `@scan` it stands for (see the `AR1Normal` docstring)")
    checks = Any[]
    for stmt in stmts
        _is_brm_ar1_normal(stmt) || continue
        time = _brm_check_cell_ar1_normal(stmt, i)
        time === nothing || push!(checks,
            :($_brm_check_ar1_normal_time($(QuoteNode(stmt.args[2].args[1])),
                $time, $(header.group), $(something(header.order, :nothing)))))
    end
    checks
end

# Refuses an `AR1Normal` statement the backends cannot read and returns its
# time column, if any: it observes the cell's response `y[p]` with three
# positional arguments and an optional `time = t[p]`.
function _brm_check_cell_ar1_normal(stmt, i)
    lhs, rhs = stmt.args[2], stmt.args[3]
    shown = sprint(Base.show_unquoted, stmt)
    _brm_plate_indexed(lhs, i) && lhs.args[1] isa Symbol || error(
        "@brm: `AR1Normal(...)` observes a response read as `y[$i]` in its cell; got `$shown`")
    positional, keywords = _brm_ar1_normal_arguments(rhs)
    haskey(keywords, :by) && error(
        "@brm: `AR1Normal(...)` takes no `by=`: the plate is the grouping. Write " *
        "`@plate for $i in groups(g; order = t)` and observe `y[$i] ~ AR1Normal(...)` " *
        "in its cell; got `$shown`")
    length(positional) == 3 && all(==(:time), keys(keywords)) || error(
        "@brm: write `y[$i] ~ AR1Normal(mu, sigma, phi)` or `y[$i] ~ AR1Normal(mu, " *
        "sigma, phi; time = t[$i])`; got `$shown`")
    time = get(keywords, :time, nothing)
    time === nothing || _brm_plate_indexed(time, i) && time.args[1] isa Symbol || error(
        "@brm: `AR1Normal(...; time = t[$i])` reads the cell's time column; got `$shown`")
    time === nothing ? nothing : time.args[1]
end

# An `AR1Normal` observation in a kernel cell, once the plate's row reads are
# cell parameters: `y ~ AR1Normal(mu, sigma, phi; time = t)`. Returns the
# response, the three arguments with whether each varies along the series (it
# reads a parameter, directly or through a cell local), the time column and
# the series' length, read off data where the cell has it: with the response
# omitted, the cell draws the series.
function _brm_kernel_ar1_normal(statement, params, body)
    response = statement.args[2]
    positional, keywords = _brm_ar1_normal_arguments(statement.args[3])
    time = get(keywords, :time, nothing)
    locals = Dict{Symbol,Any}(s.args[1] => s.args[2] for s in body
        if Meta.isexpr(s, :(=), 2) && s.args[1] isa Symbol)
    series(value, seen=Set{Symbol}()) =
        value isa Symbol ? (value in params || haskey(locals, value) &&
            !(value in seen) && series(locals[value], push!(seen, value))) :
        value isa Expr && any(arg -> series(arg, seen), value.args)
    along = map(series, Tuple(positional))
    first_series = findfirst(along)
    span = time !== nothing ? time :
        first_series !== nothing ? positional[first_series] : response
    # A gathered read `x[rows]` is as long as its index, which is data.
    Meta.isexpr(span, :ref, 2) && span.args[2] isa Symbol && span.args[2] in params &&
        (span = span.args[2])
    (; response, arguments=Tuple(positional), along, time, length=:(length($span)))
end

# The arguments a backend's AR(1) submodel takes, one entry per step: a
# shared value is filled to the series' length with the backend's `fill`.
_brm_ar1_normal_series(spec, fill) = Tuple(along ? value : Expr(:call, fill, value, spec.length)
    for (value, along) in zip(spec.arguments, spec.along))

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
