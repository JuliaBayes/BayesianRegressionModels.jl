# Native structured-field draws. Term-specific effect assembly stays extensible.

Turing.@model function _brm_iid_structured_field(field, prior)
    lower = field.prior isa NamedTuple ? get(field.prior, :lower, nothing) : nothing
    upper = field.prior isa NamedTuple ? get(field.prior, :upper, nothing) : nothing
    distribution = isnothing(lower) && isnothing(upper) ? prior :
        _brm_constrained_kernel(prior; lower, upper)
    values ~ product_distribution(fill(distribution,
        length(field.levels) * field.n_per_group))
    block = transpose(reshape(values, field.n_per_group, length(field.levels)))
    (; block, values)
end

Turing.@model function _brm_correlated_structured_field(field, priors)
    scale_prior, correlation_prior = priors
    tau ~ product_distribution(fill(
        _brm_constrained_kernel(scale_prior; lower=0), field.n_per_group))
    L ~ correlation_prior
    z ~ product_distribution(fill(Normal(),
        length(field.levels) * field.n_per_group))
    block = transpose(Diagonal(tau) * Matrix(L.L) *
                      reshape(z, field.n_per_group, length(field.levels)))
    (; block, tau, L, z)
end

_brm_structured_field_model(field, prior) =
    field.prior === :correlated_normal ?
        _brm_correlated_structured_field(field, prior) :
        _brm_iid_structured_field(field, prior)

BRM._brm_native_structured_effect(
        ::BRM._BRMPreparedTerm{typeof(BRM.sb_group_demo)}, block, field) =
    [sum(@view block[field.idx[i], :]) for i in eachindex(field.idx)]
BRM._brm_native_structured_effect(
        ::BRM._BRMPreparedTerm{typeof(BRM.sb_group_clamped_demo)}, block, field) =
    [sum(@view block[field.idx[i], :]) for i in eachindex(field.idx)]

Turing.@model function _brm_turing_structured_term(term, prior)
    field = only(term.state.fields)
    draw ~ to_submodel(_brm_structured_field_model(field, prior))
    effect = BRM._brm_native_structured_effect(term, draw.block, field)
    (; effect, block=draw.block, draw)
end

# Row count for the same family: the validator calls this for every term,
# and the ext-local seam (like `_brm_turing_term_model` before it) is closed
# to downstream methods by namespace, so the generic lives here. Existing
# per-term methods stay more specific and keep winning.
function _brm_term_rows(term::BRM._BRMPreparedTerm)
    fields = hasproperty(term.state, :fields) ? term.state.fields : nothing
    isnothing(fields) && error(
        "Turing backend: no row count for `$(nameof(term.callable))`")
    maximum(length(field.idx) for field in fields)
end

# Generic single-field structured-latent path, downstream terms included:
# any prepared term whose state carries one structured field lowers through
# the shared field-draw machinery, with per-row assembly via the extensible
# `_brm_native_structured_effect` seam (which errors actionably for terms
# that do not define it). Existing per-term `_brm_turing_term_model` methods
# are more specific and keep winning; terms with no structured field still
# fail closed below.
function BRM._brm_turing_term_model(term::BRM._BRMPreparedTerm, nobs, priors)
    fields = hasproperty(term.state, :fields) ? term.state.fields : nothing
    isnothing(fields) && error(
        "Turing backend: no term model for `$(nameof(term.callable))`")
    length(fields) == 1 || error(
        "Turing backend: structured term `$(nameof(term.callable))` has " *
        "$(length(fields)) fields; multi-field terms need a term-specific model")
    length(only(fields).idx) == nobs || error(
        "Turing backend: structured term row count does not match response")
    _brm_turing_structured_term(term, only(priors.fields))
end
