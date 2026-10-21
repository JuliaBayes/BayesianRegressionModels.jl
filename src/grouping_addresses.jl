# ---- grouping-column random-effect addresses --------------------------------
#
# `sd`, `cor` and `ranef` statements (and `sd(...) = value`) address a shared
# `|ID|` block by its ID. A block written without one -- a plain `(… | g)` or an
# uncorrelated `(… || g)` -- is addressed by its grouping column instead, as in
# brms (`prior(..., class = sd, group = g, coef = x, dpar = sigma)` is
# `sd(sigma, g, x)`):
#
#     mu ~ 1 + c1 + (1 + c1 || probe)
#     log(sigma) ~ 1 + (1 | probe)
#     sd(:, probe)     ~ Exponential(50.0)   # every margin grouped by probe
#     sd(sigma, probe) ~ Exponential(1.0)    # more specific: sigma's block
#
# One address can reach several independent blocks (each predictor's plain
# block on `g`, and one block per `||` term), so `sd` addresses resolve margin
# by margin over all of them with the usual most-specific-wins rule. `cor(:, g)`
# names the one correlated block grouped by `g` (`cor(lp, g)` picks a
# predictor's when there are several), `ranef(<lp|:>, g)` the blocks of `lp`,
# or all of them.
#
# The resolution happens once, before any backend reads the model: every block
# a statement reaches is rewritten as a one-predictor `|ID|` block under a
# generated ID, and the statement as one statement per block it reaches. The
# backends then lower that block through the shared `|ID|` machinery (general
# scale families, known values, Student-t laws, exact totals, centered and cv
# emission, descriptors and start values), and keep the rewritten model as
# their `parent`. Generated IDs:
#
#     plain `(… | g)` of `lp`      -> `|lp__g|`
#     k-th `||` term on `g` of `lp` -> `|lp__g__nocor__k|`
#
# A block no statement reaches is left as written, so a model without
# grouping-column addresses is unchanged. A rewritten scalar intercept block
# keeps its plain default `LogNormal(0, 1)` scale when no `sd` statement
# reaches it (an `|ID|` margin defaults to half-normal). An `|ID|` keeps its ID
# address only; a name that is both an `|ID|` and the grouping column of an
# ID-less block is refused as ambiguous.

const _BRM_GROUPING_PREFIX = "BRM backend lowering"

"""
    _BRMGroupingBlock

One random-effect block written without an `|ID|`, as the backends lower it: a
predictor's plain terms on one grouping column coalesce into one correlated
block, and each `||` term is its own block. `declarations` are the indices (in
`_brm_group_declarations` order) of the formula terms that declare it, `id`
the generated `|ID|` it is rewritten under, and `margins` its ordered
`(predictor, coefficient)` addresses.
"""
struct _BRMGroupingBlock
    predictor::Symbol
    group::Symbol
    descriptor::Any
    zerocorr::Bool
    declarations::Vector{Int}
    effects::Vector{Any}
    id::Symbol
    margins::Vector{NamedTuple}
end

_brm_grouping_stratified(b::_BRMGroupingBlock) = b.descriptor isa Tuple
_brm_grouping_label(b::_BRMGroupingBlock) =
    b.zerocorr ? "the `||` block of `$(b.predictor)`'s term " *
                 "`$(sprint(show, only(b.effects)))` on `$(b.group)`" :
                 "the block of `$(b.predictor)` grouped by `$(b.group)`"

# The ID-less blocks grouped by the columns in `groups`, in declaration order.
function _brm_grouping_blocks(declarations, groups)
    blocks = _BRMGroupingBlock[]
    coalesced = Dict{Tuple{Symbol,Any},Int}()
    zerocorr_count = Dict{Tuple{Symbol,Symbol},Int}()
    for (index, d) in enumerate(declarations)
        isnothing(d.id) || continue
        g = _brm_grouping_column(d)
        g in groups || continue
        if d.uncorrelated
            for term in _sb_zerocorr_terms(first(getargs(d.expression)), g)
                k = zerocorr_count[(d.predictor, g)] =
                    get(zerocorr_count, (d.predictor, g), 0) + 1
                push!(blocks, _BRMGroupingBlock(d.predictor, g, d.descriptor,
                    true, [index], Any[term],
                    Symbol(d.predictor, :__, g, :__nocor__, k), NamedTuple[]))
            end
            continue
        end
        key = (d.predictor, _sb_group_key(d.descriptor))
        at = get(coalesced, key, nothing)
        if isnothing(at)
            id = d.descriptor isa Tuple ?
                Symbol(d.predictor, :__, g, :__by__, name(last(d.descriptor))) :
                Symbol(d.predictor, :__, g)
            push!(blocks, _BRMGroupingBlock(d.predictor, g, d.descriptor,
                false, [index], collect(Any, d.effects), id, NamedTuple[]))
            coalesced[key] = length(blocks)
        else
            push!(blocks[at].declarations, index)
            append!(blocks[at].effects, d.effects)
        end
    end
    map(blocks) do b
        margins = _sb_id_bucket_margins(
            (; group_desc=b.descriptor, per_target=[b.predictor => b.effects]))
        _BRMGroupingBlock(b.predictor, b.group, b.descriptor, b.zerocorr,
            b.declarations, b.effects, b.id, margins)
    end
end

_brm_margin_matches(spec, m) =
    (isnothing(spec.predictor) || m.predictor === spec.predictor) &&
    (isnothing(spec.coefficient) || m.coefficient === spec.coefficient)

_brm_grouping_correlated(b::_BRMGroupingBlock) =
    !b.zerocorr && length(b.margins) >= 2

# The blocks one grouping-column statement reaches, without validating it.
function _brm_grouping_candidates(spec, blocks)
    on = [i for (i, b) in enumerate(blocks) if b.group === spec.id &&
          (isnothing(spec.predictor) || b.predictor === spec.predictor)]
    spec.class === :sd && return [i for i in on
        if any(m -> _brm_margin_matches(spec, m), blocks[i].margins)]
    spec.class === :cor && return [i for i in on
        if _brm_grouping_correlated(blocks[i])]
    on
end

_brm_grouping_margin_list(blocks) = join(
    ("$(b.predictor): " * join((m.coefficient for m in b.margins), ", ")
     for b in blocks), "; ")

# The blocks a grouping-column statement reaches, refusing one that reaches
# nothing, reaches a block it cannot address, or is ambiguous.
function _brm_grouping_reach(spec, blocks; prefix=_BRM_GROUPING_PREFIX)
    spelling = _brm_ranef_spelling(spec)
    g = spec.id
    on = [b for b in blocks if b.group === g]
    reach = _brm_grouping_candidates(spec, blocks)
    if spec.class === :sd
        isempty(reach) && error(
            "$prefix: `$spelling` matches no random-effect margin of the " *
            "blocks grouped by `$g` without an `|ID|`; their margins are " *
            "$(_brm_grouping_margin_list(on)).")
        spec.family === r2d2 && length(reach) > 1 && error(
            "$prefix: `$spelling ~ r2d2(...)` reaches $(length(reach)) " *
            "independent random-effect blocks grouped by `$g` (" *
            join((_brm_grouping_label(blocks[i]) for i in reach), ", ") *
            "). An `r2d2` decomposition allocates the variance of one block; " *
            "address a single block, or give the blocks one shared `|ID|`.")
    elseif spec.class === :cor
        if isempty(reach)
            mine = [b for b in on
                    if isnothing(spec.predictor) || b.predictor === spec.predictor]
            error("$prefix: `$spelling` names no correlated random-effect " *
                  "block grouped by `$g` without an `|ID|`" *
                  (isempty(mine) ? "" : "; " * join((_brm_grouping_label(b) *
                   " has one margin" * (b.zerocorr ? " (`||` blocks are " *
                   "uncorrelated)" : "") for b in mine), ", ")) *
                  ", and so no correlation to set.")
        end
        length(reach) == 1 || error(
            "$prefix: `$spelling` is ambiguous: $(length(reach)) correlated " *
            "random-effect blocks are grouped by `$g` without an `|ID|`, one " *
            "per predictor. Write `cor(<linear_predictor>, $g)` for one of " *
            join(("`$(blocks[i].predictor)`" for i in reach), ", ") * ".")
    else
        isnothing(spec.coefficient) || error(
            "$prefix: a random-effect law is block-wide in this version -- " *
            "write `ranef(<linear_predictor|:>, $g)`. Per-coefficient laws " *
            "need one mixing weight per coefficient and are not implemented yet")
        isempty(reach) && error(
            "$prefix: `$spelling` matches no random-effect block grouped by " *
            "`$g` without an `|ID|`; those blocks belong to " *
            join(unique(("`$(b.predictor)`" for b in on)), ", ") * ".")
    end
    for i in reach
        _brm_grouping_stratified(blocks[i]) && error(
            "$prefix: `$spelling` reaches the stratified block " *
            "`(… | gr($g; by=$(name(last(blocks[i].descriptor)))))` of " *
            "`$(blocks[i].predictor)`; covariance priors for stratified " *
            "blocks are not supported yet.")
    end
    reach
end

# Refuse, in the statements' own spelling, everything the per-block `|ID|`
# resolution would refuse after the rewrite: a malformed scale prior or known
# value, a margin two statements set equally specifically, a correlation or law
# that does not fit its block.
function _brm_validate_grouping_specs(keys, specs, reaches, blocks;
                                      prefix=_BRM_GROUPING_PREFIX)
    for g in unique(spec.id for spec in specs)
        direct = [spec for spec in specs
                  if spec.id === g && spec.class === :sd && spec.family !== r2d2]
        isempty(direct) && continue
        margins = NamedTuple[m for b in blocks if b.group === g for m in b.margins]
        _brm_resolve_ranef_effect_overrides(direct, Dict((g, :grouping) => margins);
                                            prefix)
    end
    laws = Dict{Int,Tuple{Symbol,Any,Int}}()
    correlations = Dict{Int,Any}()
    for (key, spec, reach) in zip(keys, specs, reaches)
        spelling = _brm_ranef_spelling(spec)
        for i in reach
            block = blocks[i]
            if spec.class === :cor
                held = get(correlations, i, nothing)
                isnothing(held) || error(
                    "$prefix: `$(_brm_ranef_spelling(held))` and `$spelling` " *
                    "both set the correlation of $(_brm_grouping_label(block))")
                correlations[i] = spec
                n = length(block.margins)
                spec.family === _brm_known ?
                    _brm_known_ranef_cor(spec, n; prefix, spelling,
                                         block=_brm_grouping_label(block)) :
                    _brm_ranef_lkj_eta(spec, n; prefix)
            elseif spec.class === :ranef
                _brm_ranef_distribution(_as_distribution_type(spec.family), spec;
                                        prefix)
                rank = isnothing(spec.predictor) ? 0 : 1
                held = get(laws, i, nothing)
                !isnothing(held) && last(held) == rank && error(
                    "$prefix: `$(_brm_ranef_spelling(held[2]))` and " *
                    "`$spelling` are equally specific and both set the law of " *
                    "$(_brm_grouping_label(block))")
                (isnothing(held) || rank > last(held)) && (laws[i] = (key, spec, rank))
            end
        end
    end
    laws
end

"""
    _brm_resolve_grouping_addresses(brmi::BRMI) -> BRMI

Rewrite every random-effect block written without an `|ID|` that a
grouping-column `sd`/`cor`/`ranef` statement reaches as a one-predictor `|ID|`
block, and each such statement as one statement per block it reaches (see the
file header). Returns `brmi` itself when no statement addresses a grouping
column, so resolving twice is free.
"""
Base.@nospecializeinfer function _brm_resolve_grouping_addresses(@nospecialize(brmi::BRMI);
                                                                 prefix=_BRM_GROUPING_PREFIX)
    specs = ranef_effect_priors(brmi)
    isempty(specs) && return brmi
    declarations = _brm_group_declarations(brmi)
    ids = Set{Symbol}(d.id for d in declarations if !isnothing(d.id))
    idless = Set{Symbol}(_brm_grouping_column(d) for d in declarations
                         if isnothing(d.id) && !isnothing(_brm_grouping_column(d)))
    groups = Symbol[]
    for spec in specs
        spec.id in idless || continue
        spec.id in ids && error(
            "$prefix: `$(_brm_ranef_spelling(spec))` is ambiguous: `$(spec.id)` " *
            "is both a shared `|$(spec.id)|` random-effect ID and the grouping " *
            "column of a random-effect block written without an ID. Rename the " *
            "ID, or give that block an ID of its own.")
        spec.id in groups || push!(groups, spec.id)
    end
    isempty(groups) && return brmi

    blocks = _brm_grouping_blocks(declarations, groups)
    for b in blocks
        b.id in ids && error(
            "$prefix: $(_brm_grouping_label(b)) is addressed by its grouping " *
            "column and lowers as the generated `|$(b.id)|` block, but the " *
            "formula already uses that ID; rename it.")
    end
    entries = _brm_operation_entries(brmi)
    addressed_keys = Symbol[]
    addressed = NamedTuple[]
    for (key, op_nc) in entries
        spec = _brm_ranef_effect_prior(op_nc)
        (isnothing(spec) || !(spec.id in groups)) && continue
        push!(addressed_keys, key)
        push!(addressed, spec)
    end
    reaches = [_brm_grouping_reach(spec, blocks; prefix) for spec in addressed]
    laws = _brm_validate_grouping_specs(addressed_keys, addressed, reaches, blocks;
                                        prefix)

    rewritten = Set{Int}(d for reach in reaches for i in reach
                         for d in blocks[i].declarations)
    scaled = Set{Int}(i for (spec, reach) in zip(addressed, reaches)
                      if spec.class === :sd for i in reach)
    r2d2_predictors = [p.predictor for p in r2d2_priors(brmi)]
    replacement = Dict{Int,Vector{Any}}()
    defaults = Dict{Int,Vector{Pair{Symbol,Any}}}()
    for (i, b) in enumerate(blocks)
        all(in(rewritten), b.declarations) || continue
        for d in b.declarations
            term = declarations[d].expression
            replacement[d] = push!(get(replacement, d, Any[]),
                ExprColumn(|, b.zerocorr ? only(b.effects) : first(getargs(term)),
                           b.id, b.descriptor))
        end
        # The plain path samples a lone intercept's scale as `exp(log_scale)`
        # with `log_scale ~ std_normal()`; an `|ID|` margin is half-normal.
        intercept_only = length(b.effects) == 1 && only(b.effects) isa Integer &&
                         only(b.effects) == 1
        if intercept_only && !(i in scaled) &&
           !any(p -> isnothing(p) || p === b.predictor, r2d2_predictors)
            address = Any[:sd, b.id]
            push!(get!(defaults, first(b.declarations), Pair{Symbol,Any}[]),
                  _brm_prior_statement_key(address) => NamedColumn(
                      _brm_prior_statement_key(address),
                      ExprColumn(~, ExprColumn(effect, address...),
                                 ExprColumn(LogNormal, 0.0, 1.0))))
        end
    end

    kept = Pair{Symbol,Any}[]
    cursor = 0
    statement = Dict(zip(addressed_keys, zip(addressed, reaches)))
    for (key, op_nc) in entries
        expression = _brm_group_declaring_expression(op_nc)
        if haskey(statement, key)
            spec, reach = statement[key]
            address = collect(Any, getargs(first(getargs(_named_op(op_nc), 2))))
            rhs = last(getargs(_named_op(op_nc), 2))
            for i in reach
                spec.class === :ranef && first(laws[i]) !== key && continue
                block_address = spec.class === :sd ?
                    Any[address[1], blocks[i].id, address[3:end]...] :
                    Any[spec.class, blocks[i].id]
                block_key = _brm_prior_statement_key(block_address)
                push!(kept, block_key => NamedColumn(block_key,
                    ExprColumn(~, ExprColumn(effect, block_address...), rhs)))
            end
        elseif !isnothing(expression)
            lhs, rhs = getargs(expression, 2)
            first_declaration = cursor + 1
            rhs = _brm_map_grouped_terms(rhs) do term
                cursor += 1
                get(replacement, cursor, Any[term])
            end
            touched = any(d -> haskey(replacement, d), first_declaration:cursor)
            if touched
                op = ExprColumn(~, lhs, rhs; getkwargs(expression)...)
                push!(kept, key => (op_nc isa NamedColumn ? NamedColumn(name(op_nc), op) : op))
            else
                push!(kept, key => op_nc)
            end
            for d in first_declaration:cursor
                append!(kept, get(defaults, d, Pair{Symbol,Any}[]))
            end
        else
            push!(kept, key => op_nc)
        end
    end
    allunique(first.(kept)) || error(
        "$prefix: internal: rewriting grouping-column addresses produced " *
        "duplicate operation names")
    BRMI((; kept...))
end

# Rebuild an additive formula right-hand side, replacing every grouped term
# (in `_brm_additive_terms` order) by the terms `f(term)` returns. Every other
# leaf -- including an intercept-suppressing `0` -- is kept as written.
_brm_map_grouped_terms(f, @nospecialize(rhs)) = _brm_map_grouped_terms_item(f, rhs, true)
function _brm_map_grouped_terms_item(f, @nospecialize(x), top::Bool)
    if x isa ExprColumn && getf(x) === (+)
        items = Any[]
        for a in getargs(x)
            append!(items, _brm_map_grouped_terms_item(f, a, false))
        end
        rebuilt = ExprColumn(+, items...; getkwargs(x)...)
        return top ? rebuilt : Any[rebuilt]
    end
    _brm_is_grouped_term(x) || return top ? x : Any[x]
    terms = f(x)
    top || return terms
    length(terms) == 1 ? only(terms) : ExprColumn(+, terms...)
end

"""
    _brm_resolve_prior_addresses(brmi::BRMI) -> BRMI

Resolve the prior statements the formula grammar defines relative to the
model's structure, once, before a backend reads it: drop `optional(...)`
statements whose address the model lacks, then lower grouping-column
random-effect addresses onto generated `|ID|` blocks. Backends keep the result
as their `parent`.
"""
_brm_resolve_prior_addresses(brmi::BRMI) =
    _brm_resolve_grouping_addresses(_brm_resolve_optional_priors(brmi))

# Whether an address `[class, g, ...]` reaches an ID-less block grouped by `g`,
# or `nothing` when no such block exists. Never errors: `optional(...)` keeps a
# statement whenever it cannot tell.
Base.@nospecializeinfer function _brm_grouping_address_present(@nospecialize(brmi::BRMI),
                                                              address::Vector{Symbol})
    g = address[2]
    declarations = _brm_group_declarations(brmi)
    any(d -> isnothing(d.id) && _brm_grouping_column(d) === g, declarations) ||
        return nothing
    blocks = try
        _brm_grouping_blocks(declarations, (g,))
    catch
        return true
    end
    predictor = length(address) >= 3 && address[3] !== _EFFECT_COLON ? address[3] : nothing
    coefficient = length(address) == 4 ? address[4] : nothing
    spec = (; class=first(address), id=g, predictor, coefficient)
    !isempty(_brm_grouping_candidates(spec, blocks))
end
