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

function BRM._brm_turing_term_model(term::BRM._BRMPreparedTerm{F}, nobs,
                                    priors) where {F<:Union{
        typeof(BRM.sb_group_demo),typeof(BRM.sb_group_clamped_demo)}}
    length(term.state.fields) == 1 || error(
        "Turing backend: native structured term `$(nameof(term.callable))` " *
        "requires a term-specific multi-field model")
    length(only(term.state.fields).idx) == nobs || error(
        "Turing backend: structured term row count does not match response")
    _brm_turing_structured_term(term, only(priors.fields))
end
