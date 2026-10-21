# ---- `ar_errors(...)`: per-group AR(1) residuals as a plate + scan ----------
#
# `y ~ ar_errors(Normal(mu, sigma), phi; time = day, by = probe)` is a source
# rewrite done before anything else reads the `@brm` block (user decision
# `0det9mo`, option G+S): the shorthand IS the `@plate` + `@scan` recurrence
# below, which `show` and the descriptor print. Within each `by` group, in
# `time` order (decision `1pnob5t`: the grouping lives in the plate header),
#
#     @plate for p in groups(probe; order = day)
#         @scan begin
#             y[p][1] ~ Normal(mu[p][1], sigma / sqrt(1 - phi ^ 2))
#             for t in 2:length(day[p])
#                 k = day[p][t] - day[p][t - 1]
#                 y[p][t] ~ Normal(mu[p][t] + phi ^ k * (y[p][t - 1] - mu[p][t - 1]),
#                                  sigma * sqrt((1 - phi ^ (2 * k)) / (1 - phi ^ 2)))
#             end
#         end
#     end
#
# `sigma` is the innovation SD and the first row of a group comes from the
# stationary law; a gap of `k` time steps uses `phi ^ k` with the matching
# innovation variance. That is the exact density of the AR(1) process observed
# on the recorded steps, and the plain recursion when no step is skipped.
#
# How the cell reads each argument follows from the block's own statements: a
# name a formula declares (`mu ~ 1 + x`, `log(sigma) ~ 1 + (1 | g)`) or that no
# statement declares (a data column) holds one value per row and is read as
# `mu[p]`, the group's rows; a name with a distribution prior
# (`phi ~ Uniform(-1, 1)`) is shared by every row and read bare; a number is
# itself.

_is_brm_ar_errors(x) = isxcall(x, :~) && isxcall(x.args[3], :ar_errors)

# The block with every `ar_errors` statement replaced by its plate, and the
# build-time checks of its `time` columns (spliced into the builder body).
function _brm_expand_ar_errors(x)
    statements = _brm_plate_flat_statements(Any[x])
    any(_is_brm_ar_errors, statements) || return x, Any[]
    reads = _brm_declared_reads(statements)
    checks = Any[]
    function expand(stmt)
        Meta.isexpr(stmt, :block) && return Expr(:block, map(expand, stmt.args)...)
        _is_brm_ar_errors(stmt) || return stmt
        plate, check = _brm_ar_errors_plate(stmt, reads)
        push!(checks, check)
        plate
    end
    expand(x), checks
end

# How a cell reads each name the block declares: `:row` for a formula predictor
# (`mu ~ 1 + x`, the linked `log(sigma) ~ …`), `:shared` for a distribution
# prior (`phi ~ Uniform(-1, 1)`), `:assigned` for `name = …`.
function _brm_declared_reads(statements)
    reads = Dict{Symbol,Symbol}()
    for s in statements
        if isxcall(s, :~)
            lhs, rhs = s.args[2], s.args[3]
            if lhs isa Symbol
                reads[lhs] = _brm_formula_rhs(rhs) ? :row : :shared
            elseif Meta.isexpr(lhs, :call, 2) && lhs.args[2] isa Symbol &&
                    !_is_effect_lhs(lhs) && !_is_optional_prior_lhs(lhs) &&
                    !_is_brm_plate_grouping(lhs)
                reads[lhs.args[2]] = :row
            end
        elseif Meta.isexpr(s, :(=), 2) && s.args[1] isa Symbol
            reads[s.args[1]] = :assigned
        end
    end
    reads
end

# A formula right-hand side: an intercept, a column, a sum of terms, a
# grouping term or a smooth/structured term. Anything else is a distribution.
_brm_formula_rhs(rhs) = rhs isa Number || rhs isa Symbol ||
    (Meta.isexpr(rhs, :call) && !isempty(rhs.args) &&
     (rhs.args[1] in (:+, :-, :*, :|, :&) || rhs.args[1] in _TERM_HEADS))

function _brm_ar_errors_plate(stmt, reads)
    y, rhs = stmt.args[2], stmt.args[3]
    source(x) = sprint(Base.show_unquoted, x)
    shown = source(stmt)
    y isa Symbol || error(
        "@brm: `ar_errors(...)` observes one response column; got `$(source(y))` in `$shown`")
    positional, keywords = Any[], Dict{Symbol,Any}()
    for arg in rhs.args[2:end]
        for a in (Meta.isexpr(arg, :parameters) ? arg.args : Any[arg])
            Meta.isexpr(a, :kw, 2) ? (keywords[a.args[1]] = a.args[2]) : push!(positional, a)
        end
    end
    length(positional) == 2 && Set(keys(keywords)) == Set((:time, :by)) || error(
        "@brm: write `$y ~ ar_errors(Normal(mu, sigma), phi; time = t, by = g)`; got `$shown`")
    family, phi = positional
    time, by = keywords[:time], keywords[:by]
    time isa Symbol && by isa Symbol || error(
        "@brm: `ar_errors(...; time, by)` name a time column and a grouping column; got `$shown`")
    isxcall(family, :Normal) && length(family.args) == 3 || error(
        "@brm: `ar_errors(...)` takes Gaussian innovations, `Normal(mu, sigma)`; got " *
        "`$(source(family))`. Write the `@plate for` / `@scan` recurrence for another " *
        "family (see the `ar_errors` docstring).")
    mu, sigma = family.args[2], family.args[3]

    used = Set{Symbol}((y, time, by))
    foreach(v -> v isa Symbol && push!(used, v), (mu, sigma, phi))
    fresh(name) = (while name in used; name = Symbol(name, :_); end; push!(used, name); name)
    p, t, k = fresh(:p), fresh(:t), fresh(:k)

    function reader(value, role)
        value isa Real && return _ -> value
        value isa Symbol || error(
            "@brm: `ar_errors(...)` reads `$role` by name or as a number; bind " *
            "`$(source(value))` to a name first, in `$shown`")
        kind = get(reads, value, :row)
        kind === :assigned && error(
            "@brm: `ar_errors(...)` reads `$role` = `$value`, which the block assigns " *
            "with `=`; use the predictor, parameter or data column it is built from")
        kind === :shared && return _ -> value
        index -> :($value[$p][$index])
    end
    m, s, f = reader(mu, "mu"), reader(sigma, "sigma"), reader(phi, "phi")
    obs(index) = :($y[$p][$index])
    previous = :($t - 1)
    seed = :($(obs(1)) ~ Normal($(m(1)), $(s(1)) / sqrt(1 - $(f(1)) ^ 2)))
    gap = :($k = $time[$p][$t] - $time[$p][$previous])
    step = :($(obs(t)) ~ Normal(
        $(m(t)) + $(f(t)) ^ $k * ($(obs(previous)) - $(m(previous))),
        $(s(t)) * sqrt((1 - $(f(t)) ^ (2 * $k)) / (1 - $(f(t)) ^ 2))))
    scan = Expr(:macrocall, Symbol("@scan"), LineNumberNode(0), Expr(:block,
        seed,
        Expr(:for, :($t = 2:length($time[$p])), Expr(:block, gap, step))))
    plate = Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
        Expr(:for, :($p = groups($by; order = $time)), Expr(:block, scan)))
    plate, :($_brm_check_ar_errors_time($(QuoteNode(y)), $time, $by))
end

# Within each group the time steps are distinct integers: a repeated time has
# no AR(1) step, and the gap rule `phi ^ k` needs whole steps.
function _brm_check_ar_errors_time(response, time::NamedColumn, by::NamedColumn)
    times, groups = parent(time), parent(by)
    times isa DataColumn && groups isa DataColumn || error(
        "@brm: `$response ~ ar_errors(...; time = $(name(time)), by = $(name(by)))` " *
        "needs both as data columns")
    times, groups = parent(times), _brm_group_values(parent(groups))
    length(times) == length(groups) || error(
        "@brm: `ar_errors` time column `$(name(time))` has $(length(times)) rows but " *
        "grouping column `$(name(by))` has $(length(groups))")
    seen = Set{Tuple{Any,Any}}()
    for (value, group) in zip(times, groups)
        value isa Real && isinteger(value) || error(
            "@brm: `ar_errors` time column `$(name(time))` holds $(repr(value)) in group " *
            "$(repr(group)); time advances in whole steps")
        (group, value) in seen && error(
            "@brm: `ar_errors` time column `$(name(time))` repeats time $(repr(value)) " *
            "in group $(repr(group)); each group observes each time once")
        push!(seen, (group, value))
    end
    nothing
end
