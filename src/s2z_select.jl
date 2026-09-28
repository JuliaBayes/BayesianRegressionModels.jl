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
    block.coordinates === :groups && throw(ArgumentError(
        "select_s2z_rho: Fisher weights parameterize Sean's linear interpolation " *
        "of contrast coordinates, but group `$group` uses `s2z_coordinates=:groups`; " *
        "use select_s2z_centeredness instead"))
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

# Scalar S2Z centering cells. With `s2z_coordinates=:groups` every group
# coordinate is already an independent zero-location cell `tau^c * w_j`, at any
# compiled `c`. With contrast coordinates (design note §2) only an endpoint
# frame is scalar: `s2z_rho = 0` samples `z ~ N(0, 1)` (c = 0) and `s2z_rho = 1`
# samples `tau * z ~ N(0, tau^2)` (c = 1). Any power-interpolated source
# `tau^c * z` is then an exact coordinate change on the same collapsed target.
# Contrast controls are not group controls: contrast `r` puts squared weight
# `r/(r+1)` on group `r + 1` and `1/(r+1)` on groups `1:r` together.

"""
    _s2z_centering_cells(model, unc_names)

Scalar centering cells of every S2Z block, ordered by block, then coefficient,
then contrast or group: `indices` (sampled coordinates), `scales` (the
matching `log(tau_k)` coordinates), zero `locations` and the compiled frame
`targets`. Group coordinates accept any compiled centeredness. For contrast
coordinates, a coefficient whose compiled `s2z_rho` column is not uniformly 0
or 1 is refused: Sean's interior map `delta = tau * P * D^-1 * u` is not a
per-contrast power interpolation, so no scalar source frame reproduces it.
"""
function _s2z_centering_cells(model, names)
    indices, scales, targets = Int[], Int[], Float64[]
    for block in s2z_effect_blocks(model)
        coords = _s2z_coordinates(model, block, names)
        for k in axes(coords.contrasts, 2)
            weights = view(block.rho, :, k)
            groups = block.coordinates === :groups
            groups || all(iszero, weights) || all(isone, weights) ||
                throw(ArgumentError(
                    "S2Z block `$(block.group)` coefficient `$(block.columns[k])` " *
                    "was compiled with interior centering weights; contrast " *
                    "centering needs an endpoint frame, so compile it with " *
                    "`s2z_rho = 0` (noncentered) or `s2z_rho = 1` (centered), " *
                    "or use `s2z_coordinates = :groups`"))
            for r in axes(coords.contrasts, 1)
                push!(indices, coords.contrasts[r, k])
                push!(scales, coords.scales[k])
                push!(targets, groups ? weights[r] : first(weights))
            end
        end
    end
    (; indices, scales, locations=zeros(length(indices)), targets)
end

"""
    select_s2z_centeredness(model, draws, unc_names;
        criterion=:position, gradients=nothing, grid=0:0.1:1)

Select one centering per S2Z cell from a saved pilot. With
`s2z_coordinates=:groups` the cells are groups (any compiled centeredness);
with contrast coordinates they are the free Helmert contrasts of an endpoint
model (`s2z_rho=0` or `1`). `draws` and optional `gradients` are draws ×
coordinates in the COMPILED model frame; no density or gradient calls are
made. Each cell `tau_k^c * w` has location zero: `:position` minimizes log SD
minus mean log Jacobian, `:gradient` minimizes the position-gradient
correlation and needs matching exact gradients.

The returned `centeredness` vector is ordered by block, coefficient and then
group or contrast. Pass it to `adaptive_centering_problem(model, problem,
backend; centeredness)`, with `nonlinear_adapt=false` for a fixed post-hoc
refit. For group coordinates, `reshape(centeredness, J, K)` is also a valid
`s2z_rho` for a recompiled fixed model. When the model also has
total-coefficient blocks, their cells come first:
`vcat(select_total_centeredness(...).centeredness, s2z.centeredness)`.

[`select_s2z_rho`](@ref) instead gives Sean's Fisher-rule weights for the
linear interpolation of contrast coordinates.
"""
function select_s2z_centeredness(model, draws::AbstractMatrix, names;
        criterion=:position, gradients=nothing, grid=0.:0.1:1.)
    isempty(s2z_effect_blocks(model)) &&
        throw(ArgumentError("model has no S2Z blocks"))
    _select_scalar_centeredness(_s2z_centering_cells(model, names), draws, names;
        criterion, gradients, grid)
end

# ---- Sean's per-draw centering candidate, for per-window online selection ----
#
# brms PR #1919 (`stan_re_s2z_fisher_comp`) evaluates, at each precursor draw,
# the response-free expected information of every row on the linear-predictor
# scale at that draw's fitted values, accumulates it per group through the
# design, and maps it to a centering weight with `_s2z_fisher_candidate`
# (restricted reliability, then the per-draw chart rescale at that draw's
# scale). The plans below hold what that needs for one S2Z block; the WarmupHMC
# extension evaluates them at every window's evidence draws.

"""Per-row expected-information source for one S2Z block's likelihood."""
struct S2ZInformationPlan
    block::S2ZEffectBlock
    family::Symbol
    scale::Any
    trials::Any
    groups::Vector{Int}
end

_s2z_mentions(x, name::Symbol) = false
_s2z_mentions(x::_BRMPreparedRef, name::Symbol) = x.name === name
_s2z_mentions(x::_BRMPreparedExpr, name::Symbol) =
    any(a -> _s2z_mentions(a, name), x.args) ||
    any(a -> _s2z_mentions(a, name), values(x.kwargs))

_s2z_is_predictor(x, name) = x isa _BRMPreparedRef && x.name === name
_s2z_wraps(x, f, name) = x isa _BRMPreparedExpr && x.callable === f &&
    isempty(x.kwargs) && length(x.args) == 1 && _s2z_is_predictor(only(x.args), name)

const _S2Z_INFORMATION_FAMILIES =
    "`Normal(lp, sigma)`, `BernoulliLogit(lp)`, `Bernoulli(logistic(lp))`, " *
    "`BinomialLogit(n, lp)`, `Binomial(n, logistic(lp))` or `Poisson(exp(lp))`"

"""
    _s2z_information_plans(sb) -> Vector{S2ZInformationPlan}

Classify the one likelihood that consumes each S2Z block's linear predictor.
Fails closed for other families, weighted or modified likelihoods, several
consuming likelihoods, or a predictor that also enters the scale.
"""
function _s2z_information_plans(sb)
    blocks = s2z_effect_blocks(sb)
    isempty(blocks) && throw(ArgumentError("model has no S2Z blocks"))
    prepared = _brm_prepare_model(parent(sb))
    map(blocks) do block
        name = block.predictor
        hits = filter(o -> _s2z_mentions(o.distribution, name), prepared.observations)
        fail(why) = throw(ArgumentError(
            "S2Z Fisher selection for `$name` (group `$(block.group)`): $why; " *
            "supported likelihoods are " * _S2Z_INFORMATION_FAMILIES))
        length(hits) == 1 || fail("found $(length(hits)) likelihoods using the predictor")
        obs = only(hits)
        isnothing(obs.weight) && isnothing(obs.modifier) ||
            fail("weighted or modified likelihoods are not supported yet")
        d = obs.distribution
        f, args = d.callable, d.args
        isempty(d.kwargs) || fail("keyword arguments are not supported")
        family, scale, trials = if f === Normal && length(args) == 2 &&
                _s2z_is_predictor(args[1], name) && !_s2z_mentions(args[2], name)
            (:gaussian, args[2], nothing)
        elseif f === BernoulliLogit && length(args) == 1 && _s2z_is_predictor(args[1], name)
            (:bernoulli_logit, nothing, nothing)
        elseif f === Bernoulli && length(args) == 1 && _s2z_wraps(args[1], logistic, name)
            (:bernoulli_logit, nothing, nothing)
        elseif f === BinomialLogit && length(args) == 2 &&
                _s2z_is_predictor(args[2], name) && !_s2z_mentions(args[1], name)
            (:binomial_logit, nothing, args[1])
        elseif f === Binomial && length(args) == 2 &&
                _s2z_wraps(args[2], logistic, name) && !_s2z_mentions(args[1], name)
            (:binomial_logit, nothing, args[1])
        elseif f === Poisson && length(args) == 1 && _s2z_wraps(args[1], exp, name)
            (:poisson_log, nothing, nothing)
        else
            fail("`$(f)` with this argument pattern is not supported")
        end
        groups = Vector{Int}(vec(sb.data[block.group_index]))
        length(groups) == size(block.design, 1) ||
            fail("the predictor rows do not match the grouping rows")
        S2ZInformationPlan(block, family, scale, trials, groups)
    end
end

# Resolve a likelihood argument at one draw: a literal, compiled data, or a
# sampled / transformed parameter looked up by its Stan base name.
_s2z_argument(x::Real, lookup, N) = fill(Float64(x), N)
function _s2z_argument(x::_BRMPreparedRef, lookup, N)
    value = lookup(x.name)
    value isa Real && return fill(Float64(value), N)
    length(value) == N || throw(DimensionMismatch(
        "S2Z Fisher selection: `$(x.name)` has $(length(value)) values for $N rows"))
    Vector{Float64}(vec(value))
end
_s2z_argument(x, lookup, N) = throw(ArgumentError(
    "S2Z Fisher selection: likelihood argument $(x) is not a literal or a named value"))

"""
    _s2z_row_information(plan, lookup) -> Vector{Float64}

Expected information of every predictor row on the linear-predictor scale at
one draw, response free: `1/sigma^2` (Gaussian), `trials * p * (1 - p)`
(logit) or `exp(eta)` (Poisson log). `lookup(name)` returns a named value at
that draw.
"""
function _s2z_row_information(plan::S2ZInformationPlan, lookup)
    N = length(plan.groups)
    if plan.family === :gaussian
        return inv.(_s2z_argument(plan.scale, lookup, N) .^ 2)
    end
    eta = _s2z_argument(_BRMPreparedRef(plan.block.predictor, :observation), lookup, N)
    plan.family === :poisson_log && return exp.(eta)
    p = logistic.(eta)
    info = p .* (1 .- p)
    plan.family === :binomial_logit || return info
    info .* _s2z_argument(plan.trials, lookup, N)
end

"""
    _s2z_fisher_draw(plan, info, tau) -> Matrix

Sean's per-draw J-by-K centering weights from per-row information `info` and
the block scales `tau` of the same draw.
"""
function _s2z_fisher_draw(plan::S2ZInformationPlan, info::AbstractVector, tau::AbstractVector)
    design = plan.block.design
    J, K = size(plan.block.rho)
    infos = [zeros(K, K) for _ in 1:J]
    # Rank-1 accumulation without the per-row `z * z'` matrix (a `mul!` scratch
    # buffer is NOT used: on a row view it falls into an allocating generic
    # fallback, ~100x worse than the expression it replaced). Element order
    # matches the original `info[n] .* (z * z')` exactly.
    for n in eachindex(plan.groups)
        z = view(design, n, :)
        w = info[n]
        acc = infos[plan.groups[n]]
        @inbounds for a in 1:K, b in 1:K
            acc[a, b] += (z[a] * z[b]) * w
        end
    end
    _s2z_fisher_candidate(infos, tau)
end
