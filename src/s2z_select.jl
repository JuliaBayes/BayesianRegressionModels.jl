# Offline S2Z centering-weight selector: per-cell Fisher candidates from the
# proper posterior draws of a saved pilot (no new target evaluations), raw
# aggregation across draws, one rescale at the posterior-median scale, and
# refit wiring through the `s2z_rho` J-by-K matrix gate.

"""
    select_s2z_rho(s2z_model, pilot_model, draws, names; obs_prec, group,
                   predictor=nothing, aggregate=:median)

Select per-cell S2Z centering weights from a saved pilot fit. `draws`/`names`
are the pilot's posterior draws in the COMPILED (unconstrained) frame with one
column per name; `obs_prec` is the draws-by-observations expected observation
precision matrix (`obs_prec[d, n]`, caller-computed from the pilot draws, e.g.
`1 ./ sigma[d, n].^2` for Gaussian location). No density or gradient calls are
made.

For each draw and coefficient, group information matrices are rebuilt from
`obs_prec` and the S2Z block's retained design, per-draw group scales come
from the pilot's `<binding>_log_scale` / `<binding>_tau` coordinates, and raw
reliabilities come from [`_s2z_fisher_raw`](@ref). Cells aggregate across
draws (`aggregate`, `:median` or `:mean`) and rescale ONCE at the aggregated
scale — never median-of-rescaled-rho, which mixes the chart nonlinearity with
per-draw scale variation and is fragile under approximate pilots.

The pilot must cover the same grouping with identical levels and row order
(checked against the pilot's preprocessing record, loudly refused otherwise).
Returns `(; rho, raw, sd, carriers, n_draws, group, columns)`: `rho` feeds
`SBBRMI(...; s2z_rho=rho)` directly.
"""
function select_s2z_rho(s2z_model, pilot_model, draws::AbstractMatrix, names;
                        obs_prec::AbstractMatrix, group::Symbol,
                        predictor::Union{Symbol,Nothing}=nothing,
                        aggregate::Symbol=:median)
    aggregate in (:median, :mean) ||
        throw(ArgumentError("select_s2z_rho: aggregate must be :median or :mean"))
    blocks = filter(b -> b.group === group, s2z_effect_blocks(s2z_model))
    isempty(blocks) && throw(ArgumentError(
        "select_s2z_rho: S2Z model has no block for group `$group`"))
    if isnothing(predictor)
        length(blocks) == 1 || throw(ArgumentError(
            "select_s2z_rho: group `$group` has $(length(blocks)) S2Z blocks; " *
            "pass `predictor` to disambiguate"))
        block = only(blocks)
    else
        hits = filter(b -> b.predictor === predictor, blocks)
        length(hits) == 1 || throw(ArgumentError(
            "select_s2z_rho: group `$group` has $(length(hits)) S2Z blocks " *
            "for predictor `$predictor`"))
        block = only(hits)
    end
    size(draws, 2) == length(names) ||
        throw(DimensionMismatch("select_s2z_rho: draws columns must match names"))
    n_draws = size(draws, 1)
    n_draws >= 1 || throw(DimensionMismatch("select_s2z_rho: need at least one draw"))
    N = size(block.design, 1)
    K = size(block.design, 2)
    size(obs_prec) == (n_draws, N) ||
        throw(DimensionMismatch("select_s2z_rho: obs_prec must be draws-by-observations " *
            "($(n_draws)-by-$N), got $(size(obs_prec))"))
    all(p -> isfinite(p) && p >= 0, obs_prec) ||
        throw(ArgumentError("select_s2z_rho: obs_prec must be finite and nonnegative"))
    _s2z_check_pilot_rows(s2z_model, pilot_model, block, group)
    carriers = _s2z_pilot_carriers(pilot_model, group, K, block.design)
    pos = Dict(String(n) => i for (i, n) in enumerate(names))
    cols = map(carriers) do (carrier, kind)
        get(pos, carrier) do
            throw(ArgumentError("select_s2z_rho: pilot draws lack the scale " *
                "coordinate `$carrier` for group `$group`; is this a pilot fit " *
                "of the same formula in the unconstrained frame?"))
        end
    end
    aggregate_fn = aggregate === :median ? Statistics.median : Statistics.mean
    J = length(_s2z_block_levels(s2z_model, block))
    indices = Vector{Int}(vec(s2z_model.data[block.group_index]))
    masks = [findall(==(j), indices) for j in 1:J]
    any(isempty, masks) && throw(ArgumentError(
        "select_s2z_rho: group `$group` has an empty level in the S2Z model data"))
    Z = block.design
    sd = Vector{Float64}(undef, K)
    for k in 1:K
        sd[k] = aggregate_fn([exp(draws[d, cols[k]]) for d in 1:n_draws])
    end
    all(s -> isfinite(s) && s > 0, sd) ||
        throw(ArgumentError("select_s2z_rho: pilot scales for group `$group` " *
            "are not finite positive"))
    raw = Array{Float64}(undef, J, K, n_draws)
    ones_info = [zeros(1, 1) for _ in 1:J]
    for d in 1:n_draws, k in 1:K
        tau = exp(draws[d, cols[k]])
        (isfinite(tau) && tau > 0) || throw(ArgumentError(
            "select_s2z_rho: pilot scale draw $d for group `$group` coefficient " *
            "$(k) is not finite positive"))
        zk2 = Z[:, k] .^ 2
        for j in 1:J
            info = 0.0
            for n in masks[j]
                info += obs_prec[d, n] * zk2[n]
            end
            ones_info[j][1, 1] = info
        end
        raw[:, k, d] = vec(_s2z_fisher_raw(ones_info, [tau]))
    end
    raw_med = Matrix{Float64}(undef, J, K)
    for j in 1:J, k in 1:K
        raw_med[j, k] = aggregate_fn([raw[j, k, d] for d in 1:n_draws])
    end
    rho = Matrix{Float64}(undef, J, K)
    for j in 1:J, k in 1:K
        rho[j, k] = _s2z_rescale_rho(raw_med[j, k], sd[k])
    end
    (; rho, raw=raw_med, sd, carriers=first.(carriers), n_draws, group,
     columns=block.columns)
end

function _s2z_block_levels(s2z_model, block)
    entry = get(s2z_model.preproc, block.group_index, nothing)
    (isnothing(entry) || entry.kind !== :group_index ||
        !(entry.const_ isa NamedTuple) || !hasproperty(entry.const_, :levels)) &&
        throw(ArgumentError("select_s2z_rho: S2Z model lacks the group-index " *
            "preprocessing record for `$(block.group_index)`"))
    collect(entry.const_.levels)
end

# The pilot must cover the same grouping with identical levels in the same
# order and identical row indices; anything else is a wrong-pilot or
# row-reordered call that would silently mistarget every cell.
function _s2z_check_pilot_rows(s2z_model, pilot_model, block, group)
    levels = _s2z_block_levels(s2z_model, block)
    indices = Vector{Int}(vec(s2z_model.data[block.group_index]))
    matches = Symbol[]
    for (key, entry) in pilot_model.preproc
        entry.kind === :group_index || continue
        entry.const_ isa NamedTuple && hasproperty(entry.const_, :levels) || continue
        _ranef_levels_equal(entry.const_.levels, levels) || continue
        push!(matches, key)
    end
    isempty(matches) && throw(ArgumentError(
        "select_s2z_rho: pilot model has no grouping with the S2Z levels of " *
        "group `$group`; is this a pilot fit of the same data?"))
    for key in matches
        value = pilot_model.data[key]
        same = value isa AbstractVector && length(value) == length(indices) &&
            all(i -> value[i] == indices[i], eachindex(indices))
        same || throw(ArgumentError(
            "select_s2z_rho: pilot grouping `$key` row indices differ from the " *
            "S2Z model; the pilot must cover the same data in the same row order"))
    end
    nothing
end

# Per-coefficient pilot scale-carrier unconstrained names. A lone plain block
# maps directly; `||` terms map by exact `__nocor__k` binding suffix; anything
# else matches by carrier-kind consistency with the design (constant columns
# take scalar `log_scale` carriers, varying columns vector `tau` carriers),
# failing closed on ambiguity. `generated` (GQ-resampled) pilot blocks are
# fine: their hyperparameters stay sampled.
function _s2z_pilot_carriers(pilot_model, group, K, Z)
    blocks = filter(b -> b.group === group && isnothing(b.by),
        ranef_blocks(pilot_model))
    isempty(blocks) && throw(ArgumentError(
        "select_s2z_rho: pilot model has no random-effect block for group `$group`"))
    if length(blocks) == 1 && blocks[1].n_terms == K
        K == 1 || throw(ArgumentError(
            "select_s2z_rho: pilot block for group `$group` covers $K terms " *
            "in one correlated block, outside the scalar S2Z scope"))
        return [_s2z_carrier(blocks[1], group, 1)]
    end
    selected = Vector{RanefBlock}(undef, K)
    use_nocor = any(b -> occursin("__nocor__", String(b.binding)), blocks)
    if use_nocor
        for k in 1:K
            hits = filter(b -> endswith(String(b.binding), "__nocor__$(k)"), blocks)
            length(hits) == 1 || throw(ArgumentError(
                "select_s2z_rho: pilot model has $(length(hits)) blocks for " *
                "group `$group` coefficient $k (expected one `__nocor__$(k)` block)"))
            hits[1].n_terms == 1 || throw(ArgumentError(
                "select_s2z_rho: pilot block `$(hits[1].binding)` covers " *
                "$(hits[1].n_terms) terms, outside the scalar S2Z scope"))
            selected[k] = hits[1]
        end
    else
        remaining = collect(blocks)
        for k in 1:K
            want_scalar = all(==(1.0), view(Z, :, k))
            hits = filter(b -> b.n_terms == 1 &&
                _s2z_carrier_is_scalar(b, group) == want_scalar, remaining)
            length(hits) == 1 || throw(ArgumentError(
                "select_s2z_rho: cannot uniquely map group `$group` coefficient " *
                "$k to a pilot block ($(length(hits)) carrier-kind matches); " *
                "the pilot shape is outside the supported scope"))
            selected[k] = hits[1]
            filter!(b -> b !== hits[1], remaining)
        end
    end
    [_s2z_carrier(selected[k], group, k) for k in 1:K]
end

function _s2z_carrier(block, group, k)
    haskey(_RANEF_FAMILIES, block.family) || throw(ArgumentError(
        "select_s2z_rho: pilot block `$(block.binding)` family `$(block.family)` " *
        "is outside the supported scope"))
    suffix = _RANEF_FAMILIES[block.family].tau
    if isnothing(suffix)
        return ("$(block.binding)_log_scale", :scalar)
    end
    ("$(block.binding)_$(suffix).1", :vector)
end

function _s2z_carrier_is_scalar(block, group)
    haskey(_RANEF_FAMILIES, block.family) || throw(ArgumentError(
        "select_s2z_rho: pilot block `$(block.binding)` family `$(block.family)` " *
        "is outside the supported scope"))
    isnothing(_RANEF_FAMILIES[block.family].tau)
end
