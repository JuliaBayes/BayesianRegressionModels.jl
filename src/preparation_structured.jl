# Backend-neutral normalization and group resolution for structured term fields.

_brm_structured_fields(::Nothing, _f) = nothing
function _brm_structured_fields(decl::NamedTuple, f)
    haskey(decl, :fields) && return decl.fields
    group = haskey(decl, :group_fn) ?
        (; fn=decl.group_fn, fn_name=decl.group_fn_name) :
        (; arg_pos=get(decl, :group_arg_pos, 1))
    [(; name=nameof(f), n_per_group=decl.n_per_group, group,
       prior=:correlated_normal)]
end

function _brm_structured_group_column(gspec, term, data; prefix="BRM")
    haskey(gspec, :fn) && return gspec.fn(term, data)
    if haskey(gspec, :kwarg)
        kw = getkwargs(term)
        haskey(kw, gspec.kwarg) || error(
            "$prefix: structured-latent group kwarg `$(gspec.kwarg)=` missing in `$(nameof(getf(term)))` call")
        column = kw[gspec.kwarg]
        column isa NamedColumn || error(
            "$prefix: `$(gspec.kwarg)=` must be a NamedColumn group, got $(typeof(column))")
        return column
    end
    pos = gspec.arg_pos
    args = getargs(term)
    pos <= length(args) || error(
        "$prefix: structured-latent group_arg_pos=$pos but `$(nameof(getf(term)))` has $(length(args)) args")
    args[pos] isa NamedColumn || error(
        "$prefix: structured-latent group arg $pos must be a NamedColumn, got $(typeof(args[pos]))")
    args[pos]
end

_brm_structured_group_name(gspec, term) =
    haskey(gspec, :fn) ? gspec.fn_name :
    haskey(gspec, :kwarg) ? name(getkwargs(term)[gspec.kwarg]) :
                            name(getargs(term)[gspec.arg_pos])

function _brm_native_structured_effect end
function _brm_native_structured_effect(term, _block, _field)
    error("Turing backend: structured term `$(nameof(term.callable))` has no " *
          "native Julia implementation; define `_brm_native_structured_effect` " *
          "for this prepared callable")
end

function _brm_prepare_structured_term(term, target, context, declaration)
    specs = _brm_structured_fields(declaration, getf(term))
    fields = map(specs) do spec
        group = _brm_structured_group_column(spec.group, term, context.data)
        source = name(group)
        haskey(context.data, source) || error(
            "BRM structured term: group column `$source` is unavailable")
        raw = context.data[source]
        levels = _brm_fit_levels(raw)
        (; name=spec.name, n_per_group=spec.n_per_group, source, levels,
           idx=_brm_apply_levels(levels, raw), prior=spec.prior)
    end
    sources = Tuple(field.source for field in fields)
    _BRMPreparedTerm(getf(term), sources, (; target, fields=Tuple(fields)), sources)
end

function _brm_structured_prior_expression(field)
    field.prior === :correlated_normal && return (
        ExprColumn(Normal, 0, 1), ExprColumn(LKJCholesky, field.n_per_group, 1.0))
    field.prior === :iid_normal && return ExprColumn(Normal, 0, 1)
    prior = field.prior
    prior isa ExprColumn && return prior
    ExprColumn(prior.dist, get(prior, :args, ())...)
end

_brm_term_prior_expressions(term::_BRMPreparedTerm) = haskey(term.state, :fields) ?
    (; fields=Tuple(_brm_structured_prior_expression(field)
                    for field in term.state.fields)) : NamedTuple()
