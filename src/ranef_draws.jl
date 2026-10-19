# Random-effect summaries per posterior draw: each block's between-group SDs,
# its correlation matrix and its per-level effects, read from the carriers the
# compiled Stan program already computes. Nothing here re-derives an emitted
# transform: the effect matrix, `tau` and `L` are Stan's own values, located
# through the descriptor and the emission table (`_RANEF_FAMILIES`,
# src/prediction.jl), never by a consumer-built name.

"""
    brm_ranef_draws(d::BRMDescriptor, draws, constrained_names;
                    id=nothing, group=nothing) -> Vector{NamedTuple}

Every random-effect block's between-group standard deviations, correlation
matrix and per-level effects, per posterior draw — `lme4`'s `VarCorr` plus
`ranef`, for each draw.

`draws` is draws × constrained coordinates in `constrained_names` order, as
BridgeStan's `param_constrain` returns them with `include_tp=true`. Add
`include_gq=true` for an exact total-coefficient block (its deviations are
recovered in generated quantities) and for a block no observed likelihood
reaches (an omitted response, a prior program, a `cv_groups` or
`resample_groups` re-draw). The draws must be constrained from the compiled
model's own unconstrained frame. `adaptive_warmup_mcmc` returns adaptively
centred draws already in that frame. Centred and non-centred emissions are
read the same way.

`id = :p` selects the shared `(… | p | g)` block; `group = :g` selects every
block of grouping factor `g` (a multi-membership block matches any of its
member columns). With neither, every block is returned. A selection that
matches nothing is an error.

Each block is one named tuple:

- `binding`, `family`, `group`, `id` — the emitted block, its grouping factor
  and its `|ID|` (or `nothing`).
- `margins` — the ordered `(predictor, coefficient)` addresses of its terms:
  [`ranefcoefnames`](@ref) order for an `|ID|` block, the design columns of
  a plain one.
- `levels` — the group labels, in the order of the effects' level axis.
- `link`, `inverse_link` — per margin, the margin predictor's LHS link pair,
  the same pair [`brm_ranef_sd_coordinates`](@ref) returns.
- `sd` — draws × margins: each margin's between-group standard deviation
  `tau`, on its predictor's linear-predictor (link) scale. A Student-t block
  reports its scale `tau`; a known margin (`sd(...) = value`) its value.
- `cor` — draws × margins × margins: the correlation matrix `L L'`, the
  identity for a one-margin block.
- `effects` — draws × levels × margins: what each level adds to each margin's
  linear predictor, on the link scale. This is the emitted effect matrix:
  `diag(tau) L z` under the default emission, the sampled effects under
  `centered_groups`, the level's mixing weight included for Student-t, and
  Stan's recovered deviation `total − A β` of an exact total-coefficient
  block. A subject's own coefficient on the natural scale is
  `inverse_link(population + effect)`, with the population value from
  [`brm_population_effect_coordinates`](@ref).

An exact total-coefficient block reports one entry per absorbed column, each
with one margin, as the conventional `(… || g)` emission reports one block
per independent margin.

A stratified `gr(g, by=b)` block is refused: it has one `tau` and `L` per
stratum, so its SDs and correlations do not fit these per-block arrays.

```julia
sm = stan_instantiate(sb).model
names = BridgeStan.param_names(sm; include_tp=true, include_gq=true)
draws = reduce(vcat, permutedims(BridgeStan.param_constrain(sm, x;
    include_tp=true, include_gq=true, rng)) for x in unconstrained_draws)
p = only(brm_ranef_draws(brm_descriptor(sb), draws, names; id=:p))
p.margins                     # [(predictor=:mu, coefficient=:Intercept), …]
median(p.sd; dims=1)          # between-subject SDs
median(p.cor[:, 1, 2])        # intercept-slope correlation
p.effects[:, :, 1]            # draws × subjects: intercept deviations
```
"""
function brm_ranef_draws(d::BRMDescriptor, draws::AbstractMatrix, constrained_names;
                         id::Union{Nothing,Symbol}=nothing,
                         group::Union{Nothing,Symbol}=nothing)
    _brm_check_draw_names(draws, constrained_names)
    blocks = _brmd_ranef_metadata(d)
    selected = [m for m in blocks if _brm_ranef_draws_selects(m.block, id, group)]
    if isempty(selected)
        available = join(("(id=$(repr(m.block.id)), group=$(repr(m.block.group)))"
                          for m in blocks), ", ")
        error("brm_ranef_draws: no random-effect block matches id=$(repr(id)), " *
              "group=$(repr(group)). This model's blocks are: " *
              (isempty(available) ? "(none)" : available) * ".")
    end
    brmi = d.plan.parent
    map(selected) do m
        values = haskey(m, :total) ?
            _brm_ranef_total_draws(d, m, draws, constrained_names) :
            _brm_ranef_block_draws(d, m.block, draws, constrained_names)
        links = [_brm_margin_link(brmi, margin.predictor) for margin in m.margins]
        (; binding=m.block.binding, family=m.block.family, group=m.block.group,
           id=m.block.id, margins=collect(m.margins), levels=collect(m.block.levels),
           link=links, inverse_link=InverseFunctions.inverse.(links), values...)
    end
end

_brm_ranef_draws_selects(block, id, group) =
    (isnothing(id) || block.id === id) &&
    (isnothing(group) || group in _ranef_group_symbols(block.group))

# One conventional block: the effect matrix, `tau` and `L` it emitted.
function _brm_ranef_block_draws(d, block::RanefBlock, draws, names)
    isnothing(block.by) || error(
        "brm_ranef_draws: block `$(block.binding)` is stratified by `$(block.by)` " *
        "(`gr($(block.group), by=$(block.by))`); it has one `tau` and `L` per " *
        "stratum, so its SDs and correlations do not fit one per-block array.")
    spec = _RANEF_FAMILIES[block.family]
    keywords = only(x for x in d.plan.declarations
                    if x.role === :prior && x.target === block.binding).keywords
    n, K, G = size(draws, 1), block.n_terms, block.n_groups
    sd = _brm_ranef_scale_draws(d, block, spec, keywords, draws, names)
    cor = ones(n, 1, 1)
    if K > 1
        L_name = get(keywords, :L, Symbol(block.binding, :_L))
        L = reshape(_brm_ranef_carrier_draws(d, block, L_name, K * K, draws, names),
                    n, K, K)
        cor = similar(L)
        for i in 1:n
            Li = view(L, i, :, :)
            cor[i, :, :] = Li * Li'
        end
    end
    effects = if isnothing(spec.effects)
        K == 1 || error(
            "brm_ranef_draws: `$(block.family)` emits no effect carrier and has " *
            "$K terms; only a one-term block is its scale times its standardised draw.")
        z = _brm_ranef_carrier_draws(d, block, Symbol(block.binding, :_, spec.z), G,
                                     draws, names)
        reshape(sd[:, 1] .* z, n, G, 1)
    else
        carrier = spec.effects === :self ? block.binding :
            Symbol(block.binding, :_, spec.effects)
        reshape(_brm_ranef_carrier_draws(d, block, carrier, G * K, draws, names),
                n, G, K)
    end
    (; sd, cor, effects)
end

# Draws × margins of a block's between-group SDs: its per-margin `tau` vector,
# or the one scale of a block that emits none.
function _brm_ranef_scale_draws(d, block, spec, keywords, draws, names)
    K = block.n_terms
    if !isnothing(spec.tau)
        name = get(keywords, :tau, Symbol(block.binding, :_, spec.tau))
        return _brm_ranef_carrier_draws(d, block, name, K, draws, names)
    end
    scale = get(spec, :scale, nothing)
    scale === :log_scale && return exp.(_brm_ranef_carrier_draws(
        d, block, Symbol(block.binding, :_log_scale), 1, draws, names))
    scale === :keyword && return _brm_ranef_carrier_draws(
        d, block, keywords.scale, 1, draws, names)
    error("brm_ranef_draws: `$(block.family)` (block `$(block.binding)`) has no ",
          "tabled scale carrier; add one to `_RANEF_FAMILIES` (src/prediction.jl).")
end

# Draws × elements (column-major) of one carrier: its constrained coordinates,
# or, for a known value passed in as data (`sd(...) = value`,
# `cor(:, ID) = R`), that value in every draw.
function _brm_ranef_carrier_draws(d, block, name::Symbol, n_elements::Int, draws, names)
    outputs = [o for o in d.outputs if o.name === name]
    if isempty(outputs)
        haskey(d.plan.data, name) || error(
            "brm_ranef_draws: block `$(block.binding)` reads `$name`, which is ",
            "neither a model output nor model data. Re-reflect the model that ",
            "produced the posterior draws.")
        value = vec(collect(Float64, d.plan.data[name]))
        length(value) == n_elements || error(
            "brm_ranef_draws: data `$name` of block `$(block.binding)` has ",
            "$(length(value)) values; expected $n_elements.")
        return repeat(permutedims(value), size(draws, 1), 1)
    end
    output = only(outputs)
    coordinates = _brm_element_coordinates(output, names)
    length(coordinates) == n_elements || error(
        "brm_ranef_draws: carrier `$name` of block `$(block.binding)` has ",
        "$n_elements elements but resolves to $(length(coordinates)) of the ",
        "given names" * _brm_ranef_kind_hint(output) * ".")
    draws[:, coordinates]
end

_brm_ranef_kind_hint(output) =
    output.kind === :generated_quantity ?
        "; it is drawn in generated quantities, so pass names and draws that " *
        "include them (`include_gq=true`)" :
    output.kind === :transformed_parameter ?
        "; it is a transformed parameter, so pass names and draws that include " *
        "them (`include_tp=true`)" :
        ". Re-reflect the model that produced the posterior draws"

# One absorbed column of an exact total-coefficient block: its total scale and
# Stan's recovered deviations `total - A β`.
function _brm_ranef_total_draws(d, m, draws, names)
    total = m.total.binding.total
    k = m.total.column
    n, J, K = size(draws, 1), m.block.n_groups, length(total.columns)
    scales = [o for o in d.outputs if o.name === total.scales]
    length(scales) == 1 || error(
        "brm_ranef_draws: exact-total scale `$(total.scales)` owns ",
        "$(length(scales)) descriptor outputs; expected exactly one.")
    sd = draws[:, [_brm_total_scale_coordinate(
        (; block=total, column=k, output=only(scales)), names, "brm_ranef_draws")]]
    deviations = _brm_ranef_carrier_draws(d, m.block, total.deviations, J * K,
                                          draws, names)
    effects = reshape(deviations, n, J, K)[:, :, k:k]
    (; sd, cor=ones(n, 1, 1), effects)
end
