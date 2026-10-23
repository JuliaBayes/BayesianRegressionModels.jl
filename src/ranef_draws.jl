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
- `by`, `stratum` — a stratified `gr(g, by=b)` block's stratifying column and
  the label of the stratum this entry reports, `nothing` for any other block.
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
per independent margin. So does a sum-to-zero block (`s2z_groups`, family
`:brm_s2z`): each column's `sd` is the block's scale `tau`, its `effects` are
the recovered per-level effects `r + μ` (generated quantities, so pass
`include_gq=true` draws), and its `id` is the one-margin `|ID|` bucket the
block claimed, so `id = :p` selects it under the same address as without
`s2z_groups`.

A stratified `gr(g, by=b)` block has one `tau` and `L` per stratum (per level
of `b`), and every group belongs to one stratum. It reports one entry per
stratum, in the order of the stratum labels: that stratum's `sd` and `cor`,
and the `levels` and `effects` of its groups, in block order. `group = :g`
selects every stratum.

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
    blocks = (_brmd_ranef_metadata(d)..., _brm_s2z_ranef_metadata(d)...)
    selected = [m for m in blocks if _brm_ranef_draws_selects(m.block, id, group)]
    if isempty(selected)
        available = join(("(id=$(repr(m.block.id)), group=$(repr(m.block.group)))"
                          for m in blocks), ", ")
        error("brm_ranef_draws: no random-effect block matches id=$(repr(id)), " *
              "group=$(repr(group)). This model's blocks are: " *
              (isempty(available) ? "(none)" : available) * ".")
    end
    brmi = d.plan.parent
    result = NamedTuple[]
    for m in selected
        links = [_brm_margin_link(brmi, margin.predictor) for margin in m.margins]
        for entry in _brm_ranef_entries(d, m, draws, constrained_names)
            push!(result, (; binding=m.block.binding, family=m.block.family,
                group=m.block.group, id=m.block.id, by=m.block.by,
                stratum=entry.stratum, margins=collect(m.margins),
                levels=entry.levels, link=links,
                inverse_link=InverseFunctions.inverse.(links), sd=entry.sd,
                cor=entry.cor, effects=entry.effects))
        end
    end
    result
end

_brm_ranef_draws_selects(block, id, group) =
    (isnothing(id) || block.id === id) &&
    (isnothing(group) || group in _ranef_group_symbols(block.group))

# The entries of one block: one, or one per stratum of a stratified
# `gr(g, by=b)` block, holding that stratum's `tau`, `L`, levels and effects.
function _brm_ranef_entries(d, m, draws, names)
    if haskey(m, :total)
        total = m.total.binding.total
        return [(; stratum=nothing, levels=collect(m.block.levels),
            _brm_ranef_column_draws(d, total, m.total.column, total.deviations,
                                    draws, names)...)]
    end
    haskey(m, :s2z) && return [(; stratum=nothing, levels=collect(m.block.levels),
        _brm_ranef_column_draws(d, m.s2z.block, m.s2z.column, m.s2z.block.effects,
                                draws, names)...)]
    isnothing(m.block.by) && return [(; stratum=nothing,
        levels=collect(m.block.levels), _brm_ranef_block_draws(d, m.block, draws, names)...)]
    x = _brm_ranef_stratified_draws(d, m.block, draws, names)
    map(eachindex(x.strata)) do s
        members = findall(==(s), x.stratum)
        (; stratum=x.strata[s], levels=m.block.levels[members], sd=x.sd[:, s, :],
           cor=x.cor[:, s, :, :], effects=x.effects[:, members, :])
    end
end

# One conventional block: the effect matrix, `tau` and `L` it emitted.
function _brm_ranef_block_draws(d, block::RanefBlock, draws, names)
    spec = _RANEF_FAMILIES[block.family]
    keywords = _brm_ranef_keywords(d, block)
    n, K = size(draws, 1), block.n_terms
    sd = _brm_ranef_scale_draws(d, block, spec, keywords, draws, names)
    cor = K == 1 ? ones(n, 1, 1) : _brm_ranef_cor_draws(reshape(
        _brm_ranef_carrier_draws(d, block, get(keywords, :L, Symbol(block.binding, :_L)),
                                 K * K, draws, names), n, K, K))
    (; sd, cor, effects=_brm_ranef_effect_draws(d, block, spec, sd, draws, names))
end

# One stratified `gr(g, by=b)` block: the `tau` and `L` it emitted for each
# stratum `s` (`sd[:, s, :]`, `cor[:, s, :, :]`), the stratum labels in index
# order, the stratum index of each level, and the block's effect matrix.
function _brm_ranef_stratified_draws(d, block::RanefBlock, draws, names)
    spec = _RANEF_FAMILIES[block.family]
    keywords = _brm_ranef_keywords(d, block)
    n, K, G = size(draws, 1), block.n_terms, block.n_groups
    S = _ranef_data_int(d.plan.data, keywords.n_strata, block.binding, block.family)
    stratum = _ranef_data_vec(d.plan.data, keywords.stratum_idx, block.binding,
                              block.family)
    (stratum isa AbstractVector{<:Integer} && length(stratum) == G &&
     all(in(1:S), stratum)) || error(
        "brm_ranef_draws: stratum index `$(keywords.stratum_idx)` of block ",
        "`$(block.binding)` is not one stratum in 1:$S for each of its $G levels.")
    # The emitter's own coding of the `by` column (`_sb_ensure_group_data!`).
    strata = collect(_sb_fit_levels(column_data(d.plan.parent, block.by)))
    length(strata) == S || error(
        "brm_ranef_draws: `by` column `$(block.by)` of block `$(block.binding)` ",
        "has $(length(strata)) training levels but the block was emitted with ",
        "$S strata.")
    tau = reshape(_brm_ranef_carrier_draws(d, block,
        Symbol(block.binding, :_, spec.stratum_tau), K * S, draws, names), n, K, S)
    L = reshape(_brm_ranef_carrier_draws(d, block,
        Symbol(block.binding, :_, spec.stratum_L), S * K * K, draws, names), n, S, K, K)
    cor = similar(L)
    for s in 1:S
        cor[:, s, :, :] = _brm_ranef_cor_draws(L[:, s, :, :])
    end
    (; strata, stratum=collect(Int, stratum), sd=permutedims(tau, (1, 3, 2)), cor,
       effects=_brm_ranef_effect_draws(d, block, spec, nothing, draws, names))
end

_brm_ranef_keywords(d, block) =
    only(x for x in d.plan.declarations
         if x.role === :prior && x.target === block.binding).keywords

# Draws × margins × margins correlations `L L'` of draws × margins × margins
# Cholesky factors.
function _brm_ranef_cor_draws(L)
    cor = similar(L)
    for i in axes(L, 1)
        Li = view(L, i, :, :)
        cor[i, :, :] = Li * Li'
    end
    cor
end

# Draws × levels × margins: the block's emitted effect matrix, or, for a
# one-term family that emits none, its scale `sd` times its standardised draw.
function _brm_ranef_effect_draws(d, block, spec, sd, draws, names)
    n, K, G = size(draws, 1), block.n_terms, block.n_groups
    if isnothing(spec.effects)
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

# One column `k` of an exact total-coefficient or S2Z block: its scale and the
# per-level effects of that column in the block's `effects` carrier (a J × K
# matrix) — Stan's recovered deviations `total - A β` of a total block, the
# recovered effects `r + μ` of an S2Z block.
function _brm_ranef_column_draws(d, block, k, effects, draws, names)
    n, J, K = size(draws, 1), d.plan.data[block.group_count], length(block.columns)
    scales = [o for o in d.outputs if o.name === block.scales]
    length(scales) == 1 || error(
        "brm_ranef_draws: scale `$(block.scales)` of block `$(block.binding)` owns ",
        "$(length(scales)) descriptor outputs; expected exactly one.")
    sd = draws[:, [_brm_claimed_scale_coordinate(
        (; block, column=k, output=only(scales)), names, "brm_ranef_draws")]]
    values = _brm_ranef_carrier_draws(d, block, effects, J * K, draws, names)
    (; sd, cor=ones(n, 1, 1), effects=reshape(values, n, J, K)[:, :, k:k])
end

# Each column of every sum-to-zero block (`s2z_groups`) as one independent
# block, as `_brmd_ranef_metadata` lists exact totals: its `|ID|` (the bucket
# the block claimed, or `nothing`), margin and levels.
function _brm_s2z_ranef_metadata(d)
    result = NamedTuple[]
    for (; block, ids) in _brm_claimed_id_blocks(d)
        block isa S2ZEffectBlock || continue
        levels = d.plan.preproc[block.group_index].const_.levels
        for (k, label) in enumerate(block.columns)
            entry = (; binding=block.binding, family=:brm_s2z, group=block.group,
                id=ids[k], by=nothing, levels, n_terms=1,
                n_groups=d.plan.data[block.group_count], noncentered=false)
            push!(result, (; block=entry,
                margins=((; predictor=block.predictor, coefficient=label),),
                s2z=(; block, column=k)))
        end
    end
    result
end
