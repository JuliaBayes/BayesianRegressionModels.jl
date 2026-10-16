# Address-keyed starting values -> one complete unconstrained point of a backend.
#
# The caller names model quantities by the addresses priors use (a linear
# predictor's coefficients, an `|ID|` block's scales and correlation, a named
# parameter) instead of by emitted coordinate names. `_brm_start_targets`
# parses that grammar once against the BRMI. Each backend then writes the
# addressed CONSTRAINED values into its constrained view of a base point and
# inverts its own transforms: RK through its semantic coordinate records and the
# layout's `unconstrain`, Stan through the descriptor's resolvers and BridgeStan's
# `param_unconstrain`. Coordinates no address reaches keep the base point's
# constrained value, so standardized random-effect draws stay at the base.

"""
    brm_start(rk::RKBRMI, inits::NamedTuple; base) -> Vector{Float64}
    brm_start(sb::SBBRMI, stan_model, inits::NamedTuple; base) -> Vector{Float64}

Build a complete unconstrained starting point from initial values keyed by
the model's prior addresses. The RK method returns a packed point of `rk`'s
layout, the value `rk_logdensity_problem` and [`brm_nlme_start`](@ref) take.
The Stan method returns a point of the compiled `stan_model` (a
`BridgeStan.StanModel` compiled from `sb`, e.g. `stan_instantiate(sb; path).model`),
the value a sampler takes as its initial position.

`inits` keys are resolved against the model:

- `lp = (; Intercept = 1.2, x = 0.3)` sets population coefficients of linear
  predictor `lp`, on its link scale (`effect(lp, Intercept)`,
  `effect(lp, x)`). Use the public predictor name: `CL` for
  `log(CL) ~ 1 + x`. A categorical predictor column takes level-keyed pairs,
  `lp = (; arm = ["b" => 0.4, "c" => -0.1])`, one per non-reference level.
- `ID = Σ` sets the scales and correlation of the shared `(… | ID | g)` block
  from the covariance matrix `Σ` of its natural effects. Rows and columns
  follow `ranefcoefnames(brmi, ID)`. `ID = (; sd, cor)` sets either part:
  `sd` is one positive value per margin (or one for all) and `cor` a
  correlation matrix in the same order.
- `name = value` sets a scalar parameter (`sigma ~ Exponential(1)`) or a vector
  parameter (cutpoints, a `Dirichlet` simplex) on its constrained scale.

Every other coordinate keeps the constrained value it has at `base`, a full
unconstrained point of the same backend (default all zeros: coefficients 0,
scales 1, identity correlations, standardized random effects 0). A random
`base` gives Stan-style random starting values for the unaddressed rest. The
natural random effects are zero whenever their standardized draws are zero,
whatever scale and correlation are set.

On an `SBBRMI` with exact total-coefficient blocks, a coefficient the block
absorbed has no sampled coordinate: its value places every group's total at
the population value `A β` (zero deviations), with the block's unaddressed
absorbed coefficients at 0. An `|ID|` such a block absorbed keeps its address:
`ID = Σ` or `(; sd)` sets the block's total scale, the SD of each group's
total around `A β`, which is the margin's between-group SD.

An `|ID|` block whose margins are partly integrated out of the model (an
omitted response) samples only its active margins; their scales and
correlation are the corresponding sub-block of the given covariance.

Unknown addresses, ambiguous names, wrong shapes and values outside a
parameter's support are errors. The result is checked: constraining it
reproduces every addressed value. No input is mutated.
"""
function brm_start end

# One parsed address per target, independent of the backend.
#   (; kind=:population, predictor, coefficient, value)
#   (; kind=:categorical, predictor, coefficient, levels, values)
#   (; kind=:ranef, id, margins, sd, cor)
#   (; kind=:parameter, name, value)
function _brm_start_targets(brmi::BRMI, inits::NamedTuple)
    targets = NamedTuple[]
    for (key, value) in pairs(inits)
        margins = ranefcoefnames(brmi, key)
        if !isnothing(margins)
            isnothing(popcoefnames(brmi, key)) || error(
                "brm_start: `$key` names both an `|ID|` random-effect block and a " *
                "linear predictor; the address is ambiguous")
            push!(targets, _brm_start_ranef_target(key,
                Tuple{Symbol,Symbol}[(m.predictor, m.coefficient) for m in margins],
                value))
        elseif value isa NamedTuple
            for (coefficient, v) in pairs(value)
                push!(targets, _brm_start_population_target(key, coefficient, v))
            end
        else
            push!(targets, (; kind=:parameter, name=key,
                value=_brm_start_parameter_value(key, value)))
        end
    end
    targets
end

_brm_start_real(what, x::Real) = isfinite(x) ? Float64(x) :
    error("brm_start: $what must be finite, got $x")
_brm_start_real(what, x) = error("brm_start: $what must be a real number, got $(repr(x))")

_brm_start_population_target(predictor, coefficient, value::Real) =
    (; kind=:population, predictor, coefficient,
       value=_brm_start_real("effect($predictor, $coefficient)", value))
_brm_start_population_target(predictor, coefficient, value::AbstractDict) =
    _brm_start_population_target(predictor, coefficient, collect(pairs(value)))
function _brm_start_population_target(predictor, coefficient,
        value::AbstractVector{<:Pair})
    levels = _brm_transport_level.(first.(value))
    allunique(levels) || error(
        "brm_start: categorical `effect($predictor, $coefficient)` repeats a level")
    (; kind=:categorical, predictor, coefficient, levels,
       values=[_brm_start_real("effect($predictor, $coefficient) at level " *
           repr(level), v) for (level, v) in zip(levels, last.(value))])
end
_brm_start_population_target(predictor, coefficient, value) = error(
    "brm_start: `effect($predictor, $coefficient)` takes a real value, or " *
    "level => value pairs for a categorical column; got $(repr(value))")

_brm_start_parameter_value(name, value::Real) = _brm_start_real("`$name`", value)
_brm_start_parameter_value(name, value::AbstractVector{<:Real}) =
    [_brm_start_real("`$name`", x) for x in value]
_brm_start_parameter_value(name, value) = error(
    "brm_start: `$name` is neither a linear predictor (give its coefficients " *
    "as a NamedTuple) nor an `|ID|` block; as a named parameter it takes a " *
    "real value or a vector, got $(repr(value))")

function _brm_start_ranef_target(id, margins, Σ::AbstractMatrix)
    K = length(margins)
    size(Σ) == (K, K) || throw(DimensionMismatch(
        "brm_start: covariance of `|$id|` has size $(size(Σ)); its $K margins are " *
        "$(Tuple(margins)) (`ranefcoefnames` order)"))
    all(isfinite, Σ) && issymmetric(Σ) || error(
        "brm_start: covariance of `|$id|` must be finite and symmetric")
    isposdef(Symmetric(Matrix{Float64}(Σ))) || error(
        "brm_start: covariance of `|$id|` is not positive definite")
    sd = sqrt.(Float64.(diag(Σ)))
    cor = Matrix{Float64}(Σ) ./ (sd * sd')
    cor[diagind(cor)] .= 1
    (; kind=:ranef, id, margins, sd, cor=K == 1 ? nothing : cor)
end
function _brm_start_ranef_target(id, margins, value::NamedTuple)
    extra = setdiff(keys(value), (:sd, :cor))
    isempty(extra) || error(
        "brm_start: `|$id|` takes a covariance matrix or `(; sd, cor)`; unknown " *
        "field(s) $(Tuple(extra))")
    K = length(margins)
    sd = haskey(value, :sd) ? _brm_start_block_sd(id, K, value.sd) : nothing
    cor = haskey(value, :cor) ? _brm_start_block_cor(id, margins, value.cor) : nothing
    (; kind=:ranef, id, margins, sd, cor=K == 1 ? nothing : cor)
end
_brm_start_ranef_target(id, margins, value) = error(
    "brm_start: `|$id|` takes a covariance matrix or `(; sd, cor)`, got $(repr(value))")

_brm_start_block_sd(id, K, sd::Real) = _brm_start_block_sd(id, K, fill(sd, K))
function _brm_start_block_sd(id, K, sd::AbstractVector{<:Real})
    length(sd) == K || throw(DimensionMismatch(
        "brm_start: `|$id|` has $K margins, got $(length(sd)) scales"))
    all(x -> isfinite(x) && x > 0, sd) || error(
        "brm_start: scales of `|$id|` must be finite and positive, got $(sd)")
    Vector{Float64}(sd)
end
_brm_start_block_sd(id, K, sd) = error(
    "brm_start: `|$id|.sd` must be a positive number or vector, got $(repr(sd))")

function _brm_start_block_cor(id, margins, C::AbstractMatrix{<:Real})
    K = length(margins)
    size(C) == (K, K) || throw(DimensionMismatch(
        "brm_start: correlation of `|$id|` has size $(size(C)); its $K margins are " *
        "$(Tuple(margins))"))
    all(isfinite, C) && issymmetric(C) && all(isone, diag(C)) &&
        isposdef(Symmetric(Matrix{Float64}(C))) || error(
        "brm_start: correlation of `|$id|` must be symmetric, positive definite " *
        "and have a unit diagonal")
    Matrix{Float64}(C)
end
_brm_start_block_cor(id, margins, C) = error(
    "brm_start: `|$id|.cor` must be a correlation matrix, got $(repr(C))")

# The block's margins `ms` as positions in the target's `ranefcoefnames` order.
function _brm_start_margin_positions(t, ms)
    map(collect(ms)) do m
        j = findall(==(Tuple(m)), t.margins)
        length(j) == 1 || error(
            "brm_start: internal: margin $m of `|$(t.id)|` occurs $(length(j)) " *
            "times among $(Tuple(t.margins))")
        only(j)
    end
end

_brm_start_lower_factor(C) = Matrix{Float64}(cholesky(Symmetric(C)).L)

function _brm_start_base(base, n, what)
    length(base) == n || throw(DimensionMismatch(
        "brm_start: base has length $(length(base)), the $what has $n coordinates"))
    all(isfinite, base) || error("brm_start: base must be finite")
    Vector{Float64}(base)
end

function _brm_start_check(what, got, expected)
    isapprox(got, expected; atol=1e-9, rtol=1e-9) || error(
        "brm_start: $what is $(got) after unconstraining, expected $(expected)")
end

# ---------------------------------------------------------------------------
# ReactiveKernels: semantic coordinate records + the layout's inverse.

# Implemented by `BayesianRegressionModelsReactiveKernelsExt`: the packed
# unconstrained point of named constrained values.
function _rk_unconstrained_point end

function brm_start(rk::RKBRMI, inits::NamedTuple;
        base=zeros(Float64, length(_rk_layout_coordinate_names(rk))))
    u0 = _brm_start_base(base, length(_rk_layout_coordinate_names(rk)), "RK layout")
    records = _rk_coordinate_records(rk.plan)
    values = _rk_constrained_values(rk, u0)
    writes = Tuple{Symbol,Tuple,Any}[]
    for target in _brm_start_targets(parent(rk), inits)
        _brm_rk_start_writes!(writes, Val(target.kind), target, records, values)
    end
    seen = Set{Tuple{Symbol,Tuple}}()
    for (declaration, index, _) in writes
        (declaration, index) in seen && error(
            "brm_start: two addresses set RK value `$declaration`$(collect(index))")
        push!(seen, (declaration, index))
    end
    for (declaration, index, value) in writes
        values = _brm_rk_set_declaration_value(values, declaration, index, value)
    end
    u = Vector{Float64}(_rk_unconstrained_point(rk, values))
    all(isfinite, u) || error("brm_start: the RK point is not finite")
    check = _rk_constrained_values(rk, u)
    for (declaration, index, value) in writes
        got = _brm_rk_declaration_value(check, declaration)
        _brm_start_check("RK value `$declaration`$(collect(index))",
            isempty(index) ? got : got[index...], value)
    end
    u
end

function _brm_rk_start_writes!(writes, ::Val{:population}, t, records, values)
    found = Tuple{Symbol,Tuple}[]
    for r in records
        if r.kind === :population && r.predictor === t.predictor &&
                r.coefficient === t.coefficient
            push!(found, (r.declaration, get(r, :index, ())))
        elseif r.kind === :population_block && r.predictor === t.predictor
            for i in findall(==(t.coefficient), collect(r.labels))
                push!(found, (r.declaration, (i,)))
            end
        end
    end
    length(found) == 1 || _brm_start_unknown_population(t, records, length(found))
    push!(writes, (first(only(found)), last(only(found)), t.value))
end

function _brm_rk_start_writes!(writes, ::Val{:categorical}, t, records, values)
    blocks = [r for r in records if r.kind === :population_block &&
        r.predictor === t.predictor && r.coefficient === t.coefficient]
    length(blocks) == 1 || _brm_start_unknown_population(t, records, length(blocks))
    r = only(blocks)
    levels = _brm_transport_level.(collect(r.level_values))
    for (level, value) in zip(t.levels, t.values)
        i = findall(l -> isequal(l, level), levels)
        length(i) == 1 || error(
            "brm_start: `effect($(t.predictor), $(t.coefficient))` has no " *
            "coefficient for level $(repr(level)); its coefficient levels are " *
            "$(repr(levels))")
        push!(writes, (r.declaration, (only(i),), value))
    end
end

function _brm_start_unknown_population(t, records, n)
    n > 1 && error("brm_start: internal: `effect($(t.predictor), " *
        "$(t.coefficient))` matches $n RK coordinate records")
    available = [(r.predictor, r.coefficient) for r in records
        if r.kind === :population && r.predictor === t.predictor]
    append!(available, [(r.predictor, label) for r in records
        if r.kind === :population_block && r.predictor === t.predictor
        for label in (t.kind === :categorical ? (r.coefficient,) : r.labels)])
    error("brm_start: `effect($(t.predictor), $(t.coefficient))` is not a sampled " *
        "population coefficient of this model. Addressable coefficients of " *
        "`$(t.predictor)`: " *
        (isempty(available) ? "(none)" : join(("$c" for (_, c) in available), ", ")))
end

function _brm_rk_start_writes!(writes, ::Val{:parameter}, t, records, values)
    found = [r for r in records if r.kind in (:scalar, :vector) &&
        r.declaration === t.name]
    if isempty(found)
        available = [r.declaration for r in records if r.kind in (:scalar, :vector)]
        error("brm_start: `$(t.name)` is not a linear predictor, an `|ID|` block or " *
            "a named parameter of this model. Named parameters: " *
            (isempty(available) ? "(none)" : join(string.(available), ", ")))
    end
    current = _brm_rk_declaration_value(values, t.name)
    size(current) == size(t.value) || throw(DimensionMismatch(
        "brm_start: `$(t.name)` has size $(size(current)), got $(size(t.value))"))
    push!(writes, (t.name, (), t.value))
end

function _brm_rk_start_writes!(writes, ::Val{:ranef}, t, records, values)
    blocks = [r for r in records if r.kind === :ranef && isequal(r.id, t.id)]
    length(blocks) == 1 || error(
        "brm_start: `|$(t.id)|` has $(length(blocks)) RK random-effect records; " *
        "expected one plain block")
    r = only(blocks)
    P = _brm_start_margin_positions(t, r.margins)
    if !isnothing(t.sd)
        if isnothing(r.scales)
            current = _brm_rk_declaration_value(values, r.scale)
            if current isa Number
                length(P) == 1 || error(
                    "brm_start: internal: scalar scale `$(r.scale)` for $(length(P)) margins")
                push!(writes, (r.scale, (), t.sd[only(P)]))
            else
                for (k, j) in enumerate(P)
                    push!(writes, (r.scale, (k,), t.sd[j]))
                end
            end
        else
            for (k, j) in enumerate(P)
                push!(writes, (r.scales[k], (), t.sd[j]))
            end
        end
    end
    if !isnothing(t.cor) && !isnothing(r.L)
        push!(writes, (r.L, (), _brm_start_lower_factor(t.cor[P, P])))
    end
end

# Replace one addressed element (or the whole value) of a declaration in the
# nested constrained values, copying the leaf rather than mutating it.
function _brm_rk_set_declaration_value(values, declaration::Symbol, index::Tuple, value)
    path = hasproperty(values, declaration) ? (declaration,) :
        Tuple(Symbol.(split(String(declaration), '.')))
    _brm_rk_set_path(values, path, index, value)
end
function _brm_rk_set_path(values::NamedTuple, path::Tuple, index, value)
    head = first(path)
    hasproperty(values, head) || error(
        "brm_start: internal: RK constrained values carry no `$head`")
    leaf = getproperty(values, head)
    new = length(path) == 1 ? _brm_rk_set_leaf(leaf, index, value) :
        _brm_rk_set_path(leaf, Base.tail(path), index, value)
    merge(values, NamedTuple{(head,)}((new,)))
end
_brm_rk_set_leaf(leaf, ::Tuple{}, value) = value
function _brm_rk_set_leaf(leaf::AbstractArray, index::Tuple{Any,Vararg}, value)
    out = copy(leaf)
    out[index...] = value
    out
end

# ---------------------------------------------------------------------------
# StanBlocks: descriptor resolvers + BridgeStan's inverse transform.

function brm_start(sb::SBBRMI, stan_model, inits::NamedTuple;
        base=zeros(Float64, BridgeStan.param_unc_num(stan_model)))
    u0 = _brm_start_base(base, BridgeStan.param_unc_num(stan_model), "compiled Stan model")
    names = String.(BridgeStan.param_names(stan_model))
    d = brm_descriptor(sb)
    totals = total_effect_blocks(d.plan)
    writes = Tuple{Int,Float64}[]
    absorbed = Dict{Symbol,Vector{Pair{Symbol,Float64}}}()
    for target in _brm_start_targets(parent(sb), inits)
        _brm_sb_start_writes!(writes, absorbed, Val(target.kind), target, d,
            names, totals)
    end
    _brm_sb_start_totals!(writes, absorbed, sb, names, totals)
    seen = Set{Int}()
    for (i, _) in writes
        i in seen && error("brm_start: two addresses set Stan value `$(names[i])`")
        push!(seen, i)
    end
    θ = BridgeStan.param_constrain(stan_model, u0)
    for (i, value) in writes
        θ[i] = value
    end
    u = BridgeStan.param_unconstrain(stan_model, θ)
    all(isfinite, u) || error("brm_start: the Stan point is not finite")
    check = BridgeStan.param_constrain(stan_model, u)
    for (i, value) in writes
        _brm_start_check("Stan value `$(names[i])`", check[i], value)
    end
    u
end

_brm_sb_absorbing_block(totals, predictor, coefficient) =
    findfirst(b -> b.predictor === predictor && coefficient in b.population_columns,
        totals)

function _brm_sb_start_writes!(writes, absorbed, ::Val{:population}, t, d, names, totals)
    k = _brm_sb_absorbing_block(totals, t.predictor, t.coefficient)
    if !isnothing(k)
        push!(get!(absorbed, totals[k].predictor, Pair{Symbol,Float64}[]),
            t.coefficient => t.value)
        return
    end
    q = brm_population_effect_coordinates(d, t.predictor, names;
        coefficient=t.coefficient)
    hasproperty(q, :contrasts) && error(
        "brm_start: `effect($(t.predictor), $(t.coefficient))` is a categorical " *
        "column; give one value per non-reference level as level => value pairs")
    q.recovered && error(
        "brm_start: internal: `effect($(t.predictor), $(t.coefficient))` resolved " *
        "to a recovered carrier outside every exact-total block")
    push!(writes, (only(q.coordinates), t.value))
end

function _brm_sb_start_writes!(writes, absorbed, ::Val{:categorical}, t, d, names, totals)
    q = brm_population_effect_coordinates(d, t.predictor, names;
        coefficient=t.coefficient)
    hasproperty(q, :contrasts) || error(
        "brm_start: `effect($(t.predictor), $(t.coefficient))` is a numeric " *
        "coefficient; give it one real value")
    levels = [_brm_transport_level(c.nonreference_level) for c in q.contrasts]
    for (level, value) in zip(t.levels, t.values)
        i = findall(l -> isequal(l, level), levels)
        length(i) == 1 || error(
            "brm_start: `effect($(t.predictor), $(t.coefficient))` has no " *
            "coefficient for level $(repr(level)); its coefficient levels are " *
            "$(repr(levels))")
        push!(writes, (q.contrasts[only(i)].coordinate, value))
    end
end

function _brm_sb_start_writes!(writes, absorbed, ::Val{:parameter}, t, d, names, totals)
    isempty(brm_outputs(d; logical=t.name)) && error(
        "brm_start: `$(t.name)` is not a linear predictor, an `|ID|` block or a " *
        "named parameter of this model")
    order = _brm_element_coordinates(brm_output(d, t.name), names)
    isempty(order) && error(
        "brm_start: `$(t.name)` has no sampled coordinate in the compiled Stan model")
    length(order) == length(t.value) || throw(DimensionMismatch(
        "brm_start: `$(t.name)` has $(length(order)) Stan values, got $(length(t.value))"))
    for (i, value) in zip(order, t.value isa Number ? (t.value,) : t.value)
        push!(writes, (i, value))
    end
end

function _brm_sb_start_writes!(writes, absorbed, ::Val{:ranef}, t, d, names, totals)
    blocks = [b for b in ranef_blocks(d.plan) if isequal(b.id, t.id)]
    if isempty(blocks)
        scales = _brm_total_id_scales(d, t.id)
        isempty(scales) && error(
            "brm_start: `|$(t.id)|` has no sampled random-effect block in this " *
            "Stan model")
        return _brm_sb_start_total_scales!(writes, t, scales, names)
    end
    (; block, margins) = _brm_sb_ranef_block(d,
        (; group=only(unique(b.group for b in blocks)), id=t.id, margins=());
        caller="brm_start")
    block.generated && error(
        "brm_start: `|$(t.id)|` block `$(block.binding)` is drawn in generated " *
        "quantities and has no sampled coordinates")
    P = _brm_start_margin_positions(t, margins)
    pos = Dict(n => i for (i, n) in enumerate(names))
    lookup(name) = get(pos, name) do
        error("brm_start: `|$(t.id)|` expects Stan value `$name`, which the compiled " *
            "model does not declare")
    end
    K = length(P)
    if !isnothing(t.sd)
        tau = get(_RANEF_FAMILIES[block.family], :tau, nothing)
        if isnothing(tau)
            K == 1 || error("brm_start: `|$(t.id)|` block `$(block.binding)` " *
                "(`$(block.family)`) has no per-margin scale")
            push!(writes, (lookup(string(block.binding, "_log_scale")), log(t.sd[only(P)])))
        else
            carrier = string(block.binding, "_", tau)
            for (k, j) in enumerate(P)
                name = string(carrier, ".", k)
                push!(writes, (K == 1 && !haskey(pos, name) ? lookup(carrier) :
                    lookup(name), t.sd[j]))
            end
        end
    end
    if !isnothing(t.cor) && K > 1
        L = _brm_start_lower_factor(t.cor[P, P])
        carrier = _sb_ranef_correlation_carrier(block)
        for i in 1:K, j in 1:K
            push!(writes, (lookup(string(carrier, ".", i, ".", j)), L[i, j]))
        end
    end
end

# An `|ID|` an exact-total block absorbed: each margin's scale is the block's
# total scale `tau`, the SD of each group's total around `A β`. The absorbed
# margins are independent, so a correlation other than the identity has no
# coordinate.
function _brm_sb_start_total_scales!(writes, t, scales, names)
    P = _brm_start_margin_positions(t, [s.margin for s in scales])
    length(P) == length(t.margins) || error(
        "brm_start: internal: $(length(P)) of the $(length(t.margins)) margins " *
        "of `|$(t.id)|` are in exact total-coefficient blocks")
    isnothing(t.cor) || isapprox(t.cor, I; atol=1e-12) || error(
        "brm_start: the margins of `|$(t.id)|` are independent in this model's " *
        "exact total-coefficient blocks; its correlation must be the identity")
    isnothing(t.sd) && return
    for (s, j) in zip(scales, P)
        s.output.kind === :generated_quantity && error(
            "brm_start: `|$(t.id)|` total block `$(s.block.binding)` is drawn " *
            "in generated quantities and has no sampled coordinates")
        push!(writes, (_brm_total_scale_coordinate(s, names, "brm_start"), t.sd[j]))
    end
end

# An exact-total block samples each group's total `A β + deviation`; place every
# total at the population value `A β` the absorbed coefficients give.
function _brm_sb_start_totals!(writes, absorbed, sb, names, totals)
    for block in totals
        given = get(absorbed, block.predictor, nothing)
        isnothing(given) && continue
        β = zeros(Float64, length(block.population_columns))
        for (coefficient, value) in given
            β[only(findall(==(coefficient), collect(block.population_columns)))] = value
        end
        row = block.A * β
        coordinates = _total_coordinates(sb, block, names).totals
        size(coordinates, 2) == length(row) || error(
            "brm_start: internal: exact-total block of `$(block.predictor)` has " *
            "$(size(coordinates, 2)) totals per group for $(length(row)) columns")
        for g in axes(coordinates, 1), k in eachindex(row)
            push!(writes, (coordinates[g, k], row[k]))
        end
    end
end
