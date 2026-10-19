# ---- optional prior addresses -----------------------------------------------
#
# `optional(<address>) ~ prior` is a prior that applies where its address
# exists and is dropped where the model provably lacks it, so one prior block
# serves a family of models whose covariates, predictors or random-effect
# blocks vary. An unmarked address still fails loudly at lowering when it
# matches nothing, so a typo in a required prior stays an error.
#
# The drop happens once, before any backend reads the model: `SBBRMI` and
# `RKBRMI` lower (and keep as their `parent`) the resolved BRMI, so
# descriptors, descriptions, start values and power scaling see the same
# statements. The matcher is deliberately permissive. It drops a statement only
# when its address names a linear predictor, coefficient, random-effect block
# or term the model does not contain, and keeps it whenever it cannot tell;
# the backend's own resolver then rejects a kept address it cannot place, so
# no prior is ever dropped silently on a guess.

_check_term_kwargs(::typeof(effect), kwargs) =
    all(==(:optional), keys(kwargs)) && all(v -> v isa Bool, values(kwargs)) ||
    error("@brm: a prior address takes no keywords other than the " *
          "`optional(...)` marker; got $(kwargs)")

_brm_is_optional_prior(@nospecialize(lhs)) =
    lhs isa ExprColumn && getf(lhs) === effect &&
    get(getkwargs(lhs), :optional, false) === true

# The `~` operation of an optional prior statement, else `nothing`.
function _brm_optional_prior_op(@nospecialize(op_nc))
    op = _named_op(op_nc)
    isnothing(op) && return nothing
    getf(op) === (~) || return nothing
    _brm_is_optional_prior(first(getargs(op, 2))) ? op : nothing
end

"""
    _brm_resolve_optional_priors(brmi::BRMI) -> BRMI

Drop every `optional(...)` prior statement whose address the model provably
lacks, and turn the kept ones into ordinary statements. Returns `brmi` itself
when it carries no optional statement, so resolving twice is free.
"""
Base.@nospecializeinfer function _brm_resolve_optional_priors(@nospecialize(brmi::BRMI))
    entries = _brm_operation_entries(brmi)
    any(entry -> !isnothing(_brm_optional_prior_op(last(entry))), entries) ||
        return brmi
    kept = Pair{Symbol,Any}[]
    for (key, op_nc) in entries
        op = _brm_optional_prior_op(op_nc)
        if isnothing(op)
            push!(kept, key => op_nc)
            continue
        end
        lhs, rhs = getargs(op, 2)
        address = collect(Symbol, getargs(lhs))
        _brm_prior_address_present(brmi, address) || continue
        push!(kept, key => NamedColumn(name(op_nc),
            ExprColumn(~, ExprColumn(effect, getargs(lhs)...), rhs)))
    end
    BRMI((; kept...))
end

# `false` only when the address provably reaches nothing in `brmi`.
Base.@nospecializeinfer function _brm_prior_address_present(@nospecialize(brmi::BRMI),
                                                            address::Vector{Symbol})
    class = first(address)
    class in (:sd, :ranef, :cor) && return _brm_ranef_address_present(brmi, address)
    class in _TERM_PRIOR_CLASSES && return _brm_term_address_present(brmi, address)
    _brm_population_address_present(brmi, address[1], address[2])
end

_brm_address_targets(brmi::BRMI, predictor::Symbol) =
    predictor === _EFFECT_COLON ?
        Symbol[p.name for p in linear_predictors(brmi)] : Symbol[predictor]

# `effect(<lp|:>, <coefficient|:>)`, including `r2d2(...)` on `effect(lp, :)`.
Base.@nospecializeinfer function _brm_population_address_present(
        @nospecialize(brmi::BRMI), predictor::Symbol, coefficient::Symbol)
    targets = filter(t -> !isnothing(linear_predictor_op(brmi, t)),
                     _brm_address_targets(brmi, predictor))
    isempty(targets) && return false
    coefficient === _EFFECT_COLON && return true
    any(t -> _brm_population_coefficient_present(brmi, t, coefficient), targets)
end

# A coefficient address is present when it names a column the predictor's
# population terms read, an interaction of two of them, or an emitted label
# built from one (`<c>_lvl_<k>`, `<c>__ref_<level>`, `int_<a>_x_<b>`,
# transformed-term keys): every population label embeds the name of a column
# its term reads.
Base.@nospecializeinfer function _brm_population_coefficient_present(
        @nospecialize(brmi::BRMI), target::Symbol, coefficient::Symbol)
    coefficient === :Intercept && return true
    op = linear_predictor_op(brmi, target)
    names = Set{Symbol}()
    for term in _brm_additive_terms(last(getargs(op, 2)))
        _brm_is_grouped_term(term) && continue
        _brm_operation_references!(names, term)
    end
    coefficient in names && return true
    _is_interaction_key(coefficient) &&
        return all(in(names), _interaction_key_operands(coefficient))
    label = String(coefficient)
    any(names) do n
        n = String(n)
        startswith(label, n * "_") || endswith(label, "_" * n) ||
            occursin("_" * n * "_", label)
    end
end

# `sd(<lp|:>, <ID>[, <coefficient>])`, `ranef(...)` and `cor(:, <ID>)`.
Base.@nospecializeinfer function _brm_ranef_address_present(@nospecialize(brmi::BRMI),
                                                           address::Vector{Symbol})
    class, id = address[1], address[2]
    margins = try
        ranefcoefnames(brmi, id)
    catch
        return true
    end
    isnothing(margins) && return _brm_grouping_name_present(brmi, id)
    class === :cor && return true
    predictor = length(address) >= 3 && address[3] !== _EFFECT_COLON ? address[3] : nothing
    coefficient = length(address) == 4 ? address[4] : nothing
    any(margins) do margin
        (isnothing(predictor) || margin.predictor === predictor) &&
            (isnothing(coefficient) || margin.coefficient === coefficient)
    end
end

# An address that names no `|ID|` block may still name a grouping factor some
# backend resolves on its own; keep it whenever any grouped term reads `id`.
Base.@nospecializeinfer function _brm_grouping_name_present(@nospecialize(brmi::BRMI),
                                                           id::Symbol)
    for predictor in linear_predictors(brmi)
        op = linear_predictor_op(brmi, predictor.name)
        isnothing(op) && continue
        for term in _brm_additive_terms(last(getargs(op, 2)))
            _brm_is_grouped_term(term) || continue
            id in _brm_operation_references!(Set{Symbol}(), term) && return true
        end
    end
    false
end

# `sd`/`ar`/`simplex`/`latent`/`length_scale` on a term: the same term map the
# backends' term-prior resolver matches against.
Base.@nospecializeinfer function _brm_term_address_present(@nospecialize(brmi::BRMI),
                                                          address::Vector{Symbol})
    term, predictor = address[2], address[3]
    any(t -> haskey(_brm_term_address_map(brmi, t), term),
        _brm_address_targets(brmi, predictor))
end
