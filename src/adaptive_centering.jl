# Adaptive partial centering for ordinary and R2D2-scaled random-effect blocks,
# ungrouped and grouped squared-exponential HSGP basis weights, and ungrouped
# periodic HSGP cosine/sine weights.
#
# This file owns only BRM/Stan emission semantics: which unconstrained
# coordinates are one block's effects, optional LKJ-Cholesky free values, and
# positive scales, plus the pure-Julia reconstruction of C.  WarmupHMC
# owns the online candidate accumulators and window lifecycle through the
# optional extension method of `adaptive_centering_problem`.

function _adaptive_log_sample_std(values)
    n = length(values)
    n >= 2 || error("HSGP centeredness selection needs at least two pilot draws")
    mean_value = sum(values) / n
    variance = sum(abs2(value - mean_value) for value in values) / (n - 1)
    variance > 0 ? 0.5log(variance) : -Inf
end

"""
    select_hsgp_centeredness(unit_weights, log_scales;
                             candidates=0:0.1:1,
                             underflow_log=log(floatmin(Float64)))

Choose one fixed partial-centering exponent per HSGP basis weight from a pilot
fit. Rows are pilot draws and columns are basis frequencies. `unit_weights`
contains the noncentered standard-normal coordinates and `log_scales` contains
the corresponding log spectral standard deviations.

For candidate `c`, the loss is

```
log(std(unit_weights .* exp.(c .* log_scales))) - mean(c .* log_scales)
```

evaluated with a per-column exponent shift. A candidate is inadmissible when
its centered coordinate scale would fall below `underflow_log`; importantly,
the `c=0` endpoint is evaluated without multiplying by `log_scales`, so it
remains valid even when a physical high-frequency scale is `-Inf`.

The result is a named tuple with `centeredness`, the candidate `losses`, an
`admissible` mask, and the normalized `candidates`. This is an offline
pilot-then-refit rule. It does not update geometry during warmup.
"""
function select_hsgp_centeredness(unit_weights::AbstractMatrix,
                                  log_scales::AbstractMatrix;
                                  candidates=0:0.1:1,
                                  underflow_log=log(floatmin(Float64)))
    size(unit_weights) == size(log_scales) || throw(DimensionMismatch(
        "unit_weights and log_scales must have the same draws × basis shape"))
    size(unit_weights, 1) >= 2 || error(
        "HSGP centeredness selection needs at least two pilot draws")
    all(isfinite, unit_weights) || error(
        "HSGP centeredness selection needs finite pilot unit weights")
    underflow_log isa Real && isfinite(underflow_log) || error(
        "underflow_log must be finite")
    cs = Float64[candidates...]
    !isempty(cs) || error("HSGP centeredness candidates cannot be empty")
    all(c -> isfinite(c) && 0 <= c <= 1, cs) || error(
        "HSGP centeredness candidates must lie in [0, 1]")

    n_draws, n_basis = size(unit_weights)
    losses = fill(Inf, length(cs), n_basis)
    admissible = falses(length(cs), n_basis)
    for b in 1:n_basis, (ci, c) in enumerate(cs)
        centered_log_scale = if iszero(c)
            zeros(Float64, n_draws)
        else
            c .* @view(log_scales[:, b])
        end
        all(isfinite, centered_log_scale) || continue
        minimum(centered_log_scale) >= underflow_log || continue
        offset = maximum(centered_log_scale)
        transformed = @view(unit_weights[:, b]) .* exp.(centered_log_scale .- offset)
        all(isfinite, transformed) || continue
        losses[ci, b] = _adaptive_log_sample_std(transformed) + offset -
                        sum(centered_log_scale) / n_draws
        admissible[ci, b] = isfinite(losses[ci, b])
    end
    selected = Vector{Float64}(undef, n_basis)
    selected_index = Vector{Int}(undef, n_basis)
    for b in 1:n_basis
        any(@view admissible[:, b]) || error(
            "no numerically admissible centeredness candidate for basis weight $b")
        selected_index[b] = argmin(@view losses[:, b])
        selected[b] = cs[selected_index[b]]
    end
    (; centeredness=selected, selected_index, candidates=cs, losses, admissible,
       underflow_log=Float64(underflow_log))
end

"""
    select_ranef_centeredness(unit_effects, log_scales;
                              candidates=0:0.1:1,
                              underflow_log=log(floatmin(Float64)))

Apply the scalar offline partial-centering criterion to ordinary random-effect
coordinates. Rows are pilot draws; columns are one scalar `(term, group)` cell.
`unit_effects` holds noncentered standardized effects and `log_scales` the
matching log standard deviations. The criterion and stable shifted-exponent
evaluation are identical to [`select_hsgp_centeredness`](@ref); only the
coordinate interpretation differs. The result is a fixed-partial specification
for a fresh fit and does not adapt during warmup.
"""
select_ranef_centeredness(unit_effects::AbstractMatrix, log_scales::AbstractMatrix;
                          candidates=0:0.1:1,
                          underflow_log=log(floatmin(Float64))) =
    select_hsgp_centeredness(unit_effects, log_scales;
                             candidates, underflow_log)

const _ADAPTIVE_CORRELATED_FAMILIES = Set((
    :ranef_correlated,
    :ranef_slope,
    :ranef_correlated_draws,
    :ranef_correlated_draws_effect,
    :ranef_correlated_draws_generic,
    :ranef_correlated_centered,
    :ranef_correlated_draws_centered,
    :ranef_correlated_draws_centered_effect,
    :ranef_correlated_draws_centered_generic,
))

const _ADAPTIVE_INTERCEPT_FAMILIES = Set((
    :ranef_intercept,
    :ranef_intercept_centered,
    # Multi-membership intercepts sample `log_scale` exactly like an ordinary
    # `(1 | g)`; only the downstream gather differs, and it is linear in the
    # draws, so the scalar `u = s^c * z` path is exact here (snag
    # adaptive-centeri-953d87e0).
    :ranef_intercept_draws,
))

const _ADAPTIVE_STRATIFIED_FAMILIES = Set((
    :ranef_correlated_by,
    :ranef_correlated_by_draws,
))

# R2D2 blocks DERIVE their marginal scales (`tau[j] = reference_scale[j] *
# sqrt(phi[j] * R2 / (1 - R2))`) instead of sampling them, so the compiled
# model has no unconstrained `tau`/`log_scale` coordinates. They become
# `R2D2AdaptiveCenteringBlock`s whose scales are compiled from the emitted
# assignment (`_adaptive_r2d2_block`), never silently left fixed (snag
# adaptive-centeri-953d87e0).
const _ADAPTIVE_R2D2_FAMILIES = Set((
    :ranef_intercept_r2d2,
    :ranef_correlated_r2d2,
    :ranef_correlated_draws_r2d2,
))

"""
    AbstractAdaptiveCenteringBlock

Supertype of the random-effect blocks [`adaptive_centering_blocks`](@ref)
returns: [`AdaptiveCenteringBlock`](@ref) for a block whose marginal scales are
sampled coordinates, [`R2D2AdaptiveCenteringBlock`](@ref) for one whose scales
an R2D2 prior derives. Both carry `ranef`, `target_c`, `effects` and
`cholesky_free` with the same meaning, and
`_adaptive_block_cholesky(x, block)` rebuilds `diag(tau) * L` for either.
"""
abstract type AbstractAdaptiveCenteringBlock end

"""
    AdaptiveCenteringBlock

The unconstrained-coordinate description needed to adapt one ordinary
random-effect block without parsing generated Stan.

- `ranef` is BRM's emitted [`RanefBlock`](@ref).
- `target_c` is the parametrization of the compiled model: `0.0` for BRM's
  standardised `z` emission and `1.0` for a `centered_groups=` emission.
- `effects[k,g]` indexes term `k`, group `g` in the compiled model's
  unconstrained vector.
- `cholesky_free` indexes Stan's `K(K-1)/2` unconstrained values for a
  correlated block's `cholesky_factor_corr[K] L`; it is empty at `K=1`.
- `log_scales` indexes the unconstrained values underlying the positive scale:
  `tau` for a correlated emission, or `log_scale` for the `(1 | g)` fast path.

Construct with [`adaptive_centering_blocks`](@ref).  Indices are resolved by
name against the caller-supplied unconstrained-name vector; no ordering of the
whole parameter vector is assumed.
"""
struct AdaptiveCenteringBlock <: AbstractAdaptiveCenteringBlock
    ranef::RanefBlock
    target_c::Float64
    effects::Matrix{Int}
    cholesky_free::Vector{Int}
    log_scales::Vector{Int}
end

Base.show(io::IO, b::AdaptiveCenteringBlock) = print(
    io,
    "AdaptiveCenteringBlock(", b.ranef.binding,
    ", target_c=", b.target_c,
    ", ", b.ranef.n_terms, "×", b.ranef.n_groups, ")",
)

# ---- R2D2: derived marginal scales -----------------------------------------
#
# An R2D2 block's `tau[k]` is not a coordinate: the emitted model assigns it
# from other parameters (`ref * sqrt(phi[j] * R2 / (1 - R2))` for the
# `sd(...) ~ r2d2(...)` grammar, `sqrt((1 - R2) * tau_bsv^2)` for the
# whole-predictor form, a free half-normal for a margin a partial ICC statement
# leaves unaddressed). Every such assignment is a product of powers of positive
# quantities, so its LOG is linear in a handful of log-atoms of unconstrained
# coordinates. `_AdaptiveLogScale` is that normal form, compiled once from the
# emitted assignment and evaluated on every gradient call:
#
#     log tau = constant
#             + sum(w * x[i])                                  # log of a lower=0 parameter
#             + sum(a * log(inv_logit(x[i])) + b * log1m(inv_logit(x[i])))  # unit-interval parameter
#             + sum(w * log(simplex(x[coords])[j]))            # Dirichlet share
#
# Three loops over concrete vectors: no recursion, no closures, no per-call
# dispatch, so the generic Enzyme path differentiates it directly.

struct _AdaptiveSimplexLog
    weight::Float64
    # The simplex's `n - 1` unconstrained coordinates, in Stan order.
    coordinates::Vector{Int}
    entry::Int
end

struct _AdaptiveLogScale
    constant::Float64
    linear::Vector{Tuple{Float64,Int}}
    unit::Vector{Tuple{Float64,Float64,Int}}
    simplex::Vector{_AdaptiveSimplexLog}
end

"""
    R2D2AdaptiveCenteringBlock

An R2D2-scaled random-effect block (`ranef_intercept_r2d2`,
`ranef_correlated_r2d2`, `ranef_correlated_draws_r2d2`) described for adaptive
partial centering. `ranef`, `target_c`, `effects` and `cholesky_free` mean
exactly what they mean on [`AdaptiveCenteringBlock`](@ref). There is no
`log_scales` coordinate vector: the marginal scales are derived, so `scales[k]`
is the compiled log-scale of term `k` as a function of the unconstrained
vector — the model's own `R2`, Dirichlet share and reference/total scale
coordinates. Those coordinates are read, never transformed.

Construct with [`adaptive_centering_blocks`](@ref).
"""
struct R2D2AdaptiveCenteringBlock <: AbstractAdaptiveCenteringBlock
    ranef::RanefBlock
    target_c::Float64
    effects::Matrix{Int}
    cholesky_free::Vector{Int}
    scales::Vector{_AdaptiveLogScale}
end

Base.show(io::IO, b::R2D2AdaptiveCenteringBlock) = print(
    io,
    "R2D2AdaptiveCenteringBlock(", b.ranef.binding,
    ", target_c=", b.target_c,
    ", ", b.ranef.n_terms, "×", b.ranef.n_groups, ")",
)

# `log1p(exp(y))` without overflow on either side.
_adaptive_softplus(y) = y > zero(y) ? y + log1p(exp(-y)) : log1p(exp(y))

# `log(simplex_constrain(y)[entry])` for Stan's (>= 2.37) simplex transform:
# the isometric log-ratio sum-to-zero vector of `y`, then softmax. The loop is
# `stan::math::simplex_constrain` index for index.
function _adaptive_log_simplex(x::AbstractVector, coordinates, entry)
    T = eltype(x)
    N = length(coordinates)
    N == 0 && return zero(T)
    z = zeros(T, N + 1)
    sum_w = zero(T)
    for i in N:-1:1
        w = x[coordinates[i]] * inv(sqrt(T(i * (i + 1))))
        sum_w += w
        z[i] += sum_w
        z[i + 1] -= w * i
    end
    m = maximum(z)
    total = zero(T)
    for zi in z
        total += exp(zi - m)
    end
    z[entry] - (m + log(total))
end

function _adaptive_log_scale(x::AbstractVector, s::_AdaptiveLogScale)
    acc = convert(eltype(x), s.constant)
    for (w, i) in s.linear
        acc += w * x[i]
    end
    for (a, b, i) in s.unit
        # log(inv_logit(y)) = -softplus(-y); log(1 - inv_logit(y)) = -softplus(y)
        y = x[i]
        acc -= a * _adaptive_softplus(-y) + b * _adaptive_softplus(y)
    end
    for atom in s.simplex
        acc += atom.weight * _adaptive_log_simplex(x, atom.coordinates, atom.entry)
    end
    acc
end

# Every unconstrained coordinate a compiled scale reads.
function _adaptive_scale_coordinates(s::_AdaptiveLogScale)
    out = Int[]
    append!(out, (i for (_, i) in s.linear))
    append!(out, (i for (_, _, i) in s.unit))
    for atom in s.simplex
        append!(out, atom.coordinates)
    end
    out
end

# What the emitted model knows, gathered once per `adaptive_centering_blocks`
# call and only when an R2D2 block is present (every other model keeps its
# historical metadata path untouched). `parameters` are the descriptor's
# parameter outputs — their `type`/`constraints` are StanBlocks' declared Stan
# types, family-implied supports included — and `assignments` the emitted
# body's top-level `name = rhs` statements, where the derived `tau` lives.
function _adaptive_r2d2_context(model, pos)
    descriptor = brm_descriptor(model)
    plan = descriptor.plan
    parameters = Dict{Symbol,BRMOutput}(
        o.name => o for o in descriptor.outputs if o.kind === :parameter)
    assignments = Dict{Symbol,Any}()
    ambiguous = Set{Symbol}()
    body = plan.model.model
    body isa Expr && body.head === :block || error(
        "BRM adaptive centering: the emitted model body is not a statement " *
        "block; R2D2 scales cannot be resolved.")
    for st in body.args
        st isa Expr && st.head === :(=) && st.args[1] isa Symbol || continue
        lhs = st.args[1]
        haskey(assignments, lhs) && push!(ambiguous, lhs)
        assignments[lhs] = st.args[2]
    end
    for lhs in ambiguous
        delete!(assignments, lhs)
    end
    (; plan, data=plan.data, parameters, assignments, ambiguous, pos)
end

# Accumulates `w * log(expr)` while compiling; `_adaptive_log_scale_program`
# freezes it into the sorted, concrete `_AdaptiveLogScale`.
mutable struct _AdaptiveLogScaleBuilder
    constant::Float64
    linear::Dict{Int,Float64}
    unit::Dict{Int,Tuple{Float64,Float64}}
    simplex::Dict{Tuple{Vector{Int},Int},Float64}
end

_AdaptiveLogScaleBuilder() = _AdaptiveLogScaleBuilder(
    0.0, Dict{Int,Float64}(), Dict{Int,Tuple{Float64,Float64}}(),
    Dict{Tuple{Vector{Int},Int},Float64}())

function _adaptive_r2d2_refuse(owner, expr, why)
    error("BRM adaptive centering: R2D2 block `$owner` derives its scale as ",
          "`$expr`; $why. Only products, quotients, square roots and literal ",
          "powers of positive scalars, `1 - R2`, and Dirichlet shares compile ",
          "to an online log-scale.")
end

function _adaptive_unc_index(ctx, owner, name)
    i = get(ctx.pos, name, 0)
    i == 0 && error(
        "BRM adaptive centering: R2D2 block `$owner` reads the unconstrained ",
        "coordinate `$name`, which this compiled model does not have. The ",
        "model and `unc_names` do not describe the same emission.")
    i
end

function _adaptive_scalar_constraint(ctx, owner, o::BRMOutput)
    extra = setdiff(collect(keys(o.constraints)), [:lower, :upper])
    isempty(extra) || error(
        "BRM adaptive centering: R2D2 block `$owner` reads `$(o.name)`, whose ",
        "Stan type carries $(Tuple(extra)); only lower/upper bounds are ",
        "supported.")
    bound(key) = haskey(o.constraints, key) ?
        Float64(_adaptive_constraint_value(ctx.plan, o.constraints[key],
                                           owner, "R2D2 scale input")) : nothing
    (bound(:lower), bound(:upper))
end

# Add `w * log(sym)` for a scalar the scale reads by name.
function _adaptive_log_symbol!(acc, sym::Symbol, w, ctx, owner, depth)
    sym in ctx.ambiguous && _adaptive_r2d2_refuse(owner, sym,
        "`$sym` is assigned more than once in the emitted body")
    if haskey(ctx.data, sym)
        v = ctx.data[sym]
        v isa Real || _adaptive_r2d2_refuse(owner, sym,
            "data `$sym` is not a scalar")
        return _adaptive_log_form!(acc, v, w, ctx, owner, depth)
    end
    haskey(ctx.assignments, sym) &&
        return _adaptive_log_form!(acc, ctx.assignments[sym], w, ctx, owner, depth + 1)
    o = get(ctx.parameters, sym, nothing)
    isnothing(o) && _adaptive_r2d2_refuse(owner, sym,
        "`$sym` is neither data, a top-level assignment, nor a parameter")
    o.type === :real && isempty(o.size) || _adaptive_r2d2_refuse(owner, sym,
        "parameter `$sym` is a `$(o.type)`, not a scalar")
    lower, upper = _adaptive_scalar_constraint(ctx, owner, o)
    unit = lower == 0.0 && upper == 1.0
    lower == 0.0 && (isnothing(upper) || unit) || _adaptive_r2d2_refuse(
        owner, sym,
        "parameter `$sym` has bounds ($(something(lower, "none")), " *
        "$(something(upper, "none"))); only `lower=0` and the unit interval " *
        "are supported")
    i = _adaptive_unc_index(ctx, owner, String(sym))
    if unit
        a, b = get(acc.unit, i, (0.0, 0.0))
        acc.unit[i] = (a + w, b)
    else
        acc.linear[i] = get(acc.linear, i, 0.0) + w
    end
    acc
end

# Add `w * log(expr)` to `acc`.
function _adaptive_log_form!(acc, expr, w, ctx, owner, depth)
    depth <= 16 || _adaptive_r2d2_refuse(owner, expr,
        "its assignment chain is deeper than 16")
    expr = _adaptive_stan_expr_value(expr)
    if expr isa Real
        isfinite(expr) && expr > 0 || _adaptive_r2d2_refuse(owner, expr,
            "the constant $expr is not finite and positive")
        acc.constant += w * log(Float64(expr))
        return acc
    end
    expr isa Symbol && return _adaptive_log_symbol!(acc, expr, w, ctx, owner, depth)
    expr isa Expr || _adaptive_r2d2_refuse(owner, expr, "it is not an expression")
    if expr.head === :call
        f, args = expr.args[1], expr.args[2:end]
        if f === :* && !isempty(args)
            for a in args
                _adaptive_log_form!(acc, a, w, ctx, owner, depth)
            end
            return acc
        elseif f === :/ && length(args) == 2
            _adaptive_log_form!(acc, args[1], w, ctx, owner, depth)
            return _adaptive_log_form!(acc, args[2], -w, ctx, owner, depth)
        elseif f === :sqrt && length(args) == 1
            return _adaptive_log_form!(acc, args[1], w / 2, ctx, owner, depth)
        elseif f === :^ && length(args) == 2 && _adaptive_stan_expr_value(args[2]) isa Real
            p = Float64(_adaptive_stan_expr_value(args[2]))
            return _adaptive_log_form!(acc, args[1], w * p, ctx, owner, depth)
        elseif f === :- && length(args) == 2
            return _adaptive_log1m!(acc, args[1], args[2], w, ctx, owner)
        end
    elseif expr.head === :ref && length(expr.args) == 2
        return _adaptive_log_entry!(acc, expr.args[1], expr.args[2], w, ctx, owner)
    end
    _adaptive_r2d2_refuse(owner, expr, "this operation is not supported")
end

# `w * log(1 - s)` for a unit-interval parameter `s`: the residual share of R2.
function _adaptive_log1m!(acc, one_, s, w, ctx, owner)
    c = _adaptive_stan_expr_value(one_)
    s = _adaptive_stan_expr_value(s)
    expr = :($one_ - $s)
    c isa Real && c == 1 || _adaptive_r2d2_refuse(owner, expr,
        "only `1 - R2` subtracts")
    o = s isa Symbol ? get(ctx.parameters, s, nothing) : nothing
    (isnothing(o) || s in ctx.ambiguous) && _adaptive_r2d2_refuse(owner, expr,
        "`$s` is not a parameter")
    o.type === :real && isempty(o.size) &&
        _adaptive_scalar_constraint(ctx, owner, o) == (0.0, 1.0) ||
        _adaptive_r2d2_refuse(owner, expr, "`$s` is not a unit-interval scalar")
    i = _adaptive_unc_index(ctx, owner, String(s))
    a, b = get(acc.unit, i, (0.0, 0.0))
    acc.unit[i] = (a, b + w)
    acc
end

# `w * log(v[j])`: a Dirichlet share, an entry of a positive vector parameter,
# or a data entry.
function _adaptive_log_entry!(acc, v, j, w, ctx, owner)
    v = _adaptive_stan_expr_value(v)
    j = _adaptive_stan_expr_value(j)
    expr = :($v[$j])
    v isa Symbol && j isa Integer && j >= 1 || _adaptive_r2d2_refuse(owner, expr,
        "only a literal index into a named vector is supported")
    v in ctx.ambiguous && _adaptive_r2d2_refuse(owner, expr,
        "`$v` is assigned more than once in the emitted body")
    if haskey(ctx.data, v)
        d = ctx.data[v]
        d isa AbstractVector{<:Real} && j <= length(d) ||
            _adaptive_r2d2_refuse(owner, expr, "data `$v` has no real entry $j")
        return _adaptive_log_form!(acc, d[j], w, ctx, owner, 0)
    end
    o = get(ctx.parameters, v, nothing)
    isnothing(o) && _adaptive_r2d2_refuse(owner, expr, "`$v` is not a parameter")
    if o.type === :simplex
        n = _adaptive_simplex_length(ctx, owner, v)
        j <= n || _adaptive_r2d2_refuse(owner, expr, "`$v` has $n entries")
        coords = [_adaptive_unc_index(ctx, owner, "$v.$i") for i in 1:n-1]
        haskey(ctx.pos, "$v.$n") && _adaptive_r2d2_refuse(owner, expr,
            "simplex `$v` has more unconstrained coordinates than its $n-entry " *
            "Dirichlet declaration")
        key = (coords, Int(j))
        acc.simplex[key] = get(acc.simplex, key, 0.0) + w
    elseif o.type === :vector
        lower, upper = _adaptive_scalar_constraint(ctx, owner, o)
        lower == 0.0 && isnothing(upper) || _adaptive_r2d2_refuse(owner, expr,
            "vector `$v` is not `lower=0`")
        i = _adaptive_unc_index(ctx, owner, "$v.$j")
        acc.linear[i] = get(acc.linear, i, 0.0) + w
    else
        _adaptive_r2d2_refuse(owner, expr, "parameter `$v` is a `$(o.type)`")
    end
    acc
end

# A simplex's length, read from the Dirichlet declaration that samples it.
function _adaptive_simplex_length(ctx, owner, v)
    decls = [d for d in ctx.plan.declarations
             if d.role === :prior && d.target === v]
    length(decls) == 1 && decls[1].family === :dirichlet &&
        length(decls[1].arguments) == 1 || _adaptive_r2d2_refuse(owner, v,
        "simplex `$v` is not sampled by exactly one `dirichlet(alpha)` declaration")
    alpha = _adaptive_stan_expr_value(only(decls[1].arguments))
    alpha = alpha isa Symbol ? get(ctx.data, alpha, nothing) : alpha
    alpha isa AbstractVector{<:Real} || _adaptive_r2d2_refuse(owner, v,
        "the Dirichlet concentration of `$v` is not a data vector")
    length(alpha)
end

function _adaptive_log_scale_program(expr, ctx, owner)
    acc = _adaptive_log_form!(_AdaptiveLogScaleBuilder(), expr, 1.0, ctx, owner, 0)
    _AdaptiveLogScale(
        acc.constant,
        [(acc.linear[i], i) for i in sort!(collect(keys(acc.linear)))],
        [(acc.unit[i]..., i) for i in sort!(collect(keys(acc.unit)))],
        [_AdaptiveSimplexLog(acc.simplex[k], k[1], k[2])
         for k in sort!(collect(keys(acc.simplex)))],
    )
end

const _ADAPTIVE_R2D2_SCALE_KEYWORD = Dict(
    :ranef_intercept_r2d2 => :scale,
    :ranef_correlated_r2d2 => :tau,
    :ranef_correlated_draws_r2d2 => :tau,
)

function _adaptive_r2d2_block(ranef::RanefBlock, unc_names, pos, ctx)
    binding = ranef.binding
    decls = [d for d in ctx.plan.declarations
             if d.role === :prior && d.target === binding]
    length(decls) == 1 || error(
        "BRM adaptive centering: R2D2 block `$binding` has $(length(decls)) ",
        "prior declarations in the emitted body; expected exactly one.")
    key = _ADAPTIVE_R2D2_SCALE_KEYWORD[ranef.family]
    sym = get(only(decls).keywords, key, nothing)
    sym isa Symbol && haskey(ctx.assignments, sym) || error(
        "BRM adaptive centering: R2D2 block `$binding` passes `$key=$(repr(sym))`, ",
        "which is not a top-level assignment of the emitted body; its derived ",
        "scale cannot be resolved.")
    rhs = ctx.assignments[sym]
    margins = key === :scale ? Any[rhs] :
        (rhs isa Expr && rhs.head === :vect ? rhs.args : error(
            "BRM adaptive centering: R2D2 block `$binding` assigns `$sym = $rhs`; ",
            "expected one derived expression per margin."))
    K = ranef.n_terms
    length(margins) == K || error(
        "BRM adaptive centering: R2D2 block `$binding` has $K terms but `$sym` ",
        "derives $(length(margins)) scales.")
    scales = [_adaptive_log_scale_program(m, ctx, binding) for m in margins]
    cholesky_names = ranef.family === :ranef_intercept_r2d2 ? String[] :
        ["$(binding)_L.$i" for i in 1:(K * (K - 1) ÷ 2)]
    R2D2AdaptiveCenteringBlock(
        ranef,
        ranef.noncentered ? 0.0 : 1.0,
        ranef_coordinates(ranef, unc_names),
        _adaptive_named_indices(pos, cholesky_names, binding, "Cholesky"),
        scales,
    )
end

function _adaptive_named_indices(pos, names, binding, role)
    missing = String[]
    out = Vector{Int}(undef, length(names))
    for (i, name) in enumerate(names)
        idx = get(pos, name, 0)
        idx == 0 ? push!(missing, name) : (out[i] = idx)
    end
    isempty(missing) || error(
        "BRM adaptive centering: block `$binding` expects $role unconstrained ",
        "coordinates that this compiled model does not have: ",
        join(missing, ", "), ". The model and `unc_names` do not describe the ",
        "same emission.",
    )
    out
end

# One block's frame: the same name spells the whole-file loop uses, factored
# so centered replay (prediction.jl) resolves ONE block's hyperparameters
# without tripping over a sibling this contract skips or refuses. Returns
# `nothing` for a family outside every set below; stratified blocks raise
# rather than being silently left fixed. An R2D2 block needs the emitted
# model's scale context `r2d2` (`_adaptive_r2d2_context`), which only
# `adaptive_centering_blocks` builds: R2D2 blocks are never centered, so the
# replay caller never reaches one.
function _adaptive_block(ranef::RanefBlock, unc_names, pos; r2d2=nothing)
    ranef.family in _ADAPTIVE_STRATIFIED_FAMILIES && error(
        "BRM adaptive centering: stratified block `$(ranef.binding)` ",
        "(`$(ranef.family)`, group `$(ranef.group)`, by `$(ranef.by)`) is ",
        "not supported by the first correlated-block contract. It has one ",
        "Cholesky/scale frame per stratum and must not be treated as an ",
        "ordinary single-frame block.",
    )
    if ranef.family in _ADAPTIVE_R2D2_FAMILIES
        isnothing(r2d2) && error(
            "BRM adaptive centering: R2D2 block `$(ranef.binding)` derives its ",
            "scales from the emitted model; resolve it through ",
            "`adaptive_centering_blocks(model, unc_names)`.")
        return _adaptive_r2d2_block(ranef, unc_names, pos, r2d2)
    end
    is_intercept = ranef.family in _ADAPTIVE_INTERCEPT_FAMILIES
    is_correlated = ranef.family in _ADAPTIVE_CORRELATED_FAMILIES
    (is_intercept || is_correlated) || return nothing
    K = ranef.n_terms
    n_cholesky = K * (K - 1) ÷ 2
    binding = ranef.binding
    cholesky_names = is_intercept ? String[] :
        ["$(binding)_L.$i" for i in 1:n_cholesky]
    scale_names = is_intercept ? ["$(binding)_log_scale"] :
        ["$(binding)_tau.$i" for i in 1:K]
    target_c = ranef.noncentered ? 0.0 : 1.0
    AdaptiveCenteringBlock(
        ranef,
        target_c,
        ranef_coordinates(ranef, unc_names),
        _adaptive_named_indices(pos, cholesky_names, binding, "Cholesky"),
        _adaptive_named_indices(pos, scale_names, binding, "scale"),
    )
end

"""
    adaptive_centering_blocks(model, unc_names) -> Vector{AbstractAdaptiveCenteringBlock}

Describe every random-effect block in `model` for adaptive partial centering:
an [`AdaptiveCenteringBlock`](@ref) for each block whose marginal scales are
sampled, and an [`R2D2AdaptiveCenteringBlock`](@ref) for each block whose
scales an R2D2 prior derives. `model` is an [`SBBRMI`](@ref) or
[`GenerativePlan`](@ref), and `unc_names` is the compiled model's unconstrained
parameter-name vector (for example BridgeStan's `param_unc_names`).

The result supports both BRM endpoints: the default noncentered emission is the
compiled target `c=0`, while `SBBRMI(...; centered_groups=...)` is target `c=1`.
Every intermediate source uses the triangular term-wise map

```
u[k] = c[k] * sum(C[k,l] * z[l] for l < k) + C[k,k]^c[k] * z[k]
```

with `C = diag(tau) * L`. At `K=1`, this reduces to `u = s^c * z` with
`s = exp(log_scale)` and no Cholesky coordinates. Consequently `c=0` is the
literal standardised draw and `c=1` is the literal model-scale effect.

R2D2 blocks use the same map; only `tau` differs. It is not a coordinate but the
emitted model's own derived scale — `ref * sqrt(phi[j] * R2 / (1 - R2))` for
`sd(...) ~ r2d2(...)` (block-wide, per-margin ICC, or joint `include=`), with
a free half-normal for a margin a partial ICC statement leaves unaddressed, and
`sqrt((1 - R2) * tau_bsv^2)` for the whole-predictor `effect(lp, :) ~ r2d2(...)`
form — re-derived from the `R2`, Dirichlet share and reference/total scale
coordinates on every evaluation. Those coordinates are read, never transformed.
A reference scale must be a positive constant, data scalar, or `lower=0`
parameter (or a product, quotient, square root or literal power of them);
anything else raises naming the block and the expression.

Stratified `gr(g, by=b)` blocks currently raise rather than being silently
left fixed: they carry one `L,tau` frame per stratum and need a separate indexed
metadata contract. Intercept-only `(1 | mm(...))` blocks adapt through the
ordinary scalar path (their downstream gather is linear); an `mm` block with
any slope term, including a slope-only `(0 + x | mm(...))`, shares the ordinary
correlated emission and adapts with it.

Correlated `cdar(step; by=group, cor=C)` walks are not ordinary random-effect
blocks either; they have their own metadata contract in
[`_adaptive_cdar_centering_blocks`](@ref) and join the online plan through the
WarmupHMC extension exactly like the ungrouped-HSGP companion below.
"""
function adaptive_centering_blocks(model, unc_names)
    pos = _ranef_name_positions(unc_names)
    ranefs = ranef_blocks(model)
    r2d2 = any(r -> r.family in _ADAPTIVE_R2D2_FAMILIES, ranefs) ?
        _adaptive_r2d2_context(model, pos) : nothing
    out = AbstractAdaptiveCenteringBlock[]
    for ranef in ranefs
        blk = _adaptive_block(ranef, unc_names, pos; r2d2)
        isnothing(blk) || push!(out, blk)
    end

    claimed = Int[]
    for block in out
        append!(claimed, vec(block.effects))
        append!(claimed, block.cholesky_free)
        block isa AdaptiveCenteringBlock && append!(claimed, block.log_scales)
    end
    length(unique(claimed)) == length(claimed) || error(
        "BRM adaptive centering: emitted correlated blocks claim overlapping ",
        "unconstrained coordinates; refusing an ambiguous transform.",
    )
    # A derived scale must not read a coordinate the transform rewrites:
    # the map's triangular Jacobian assumes every block's scale is constant
    # in every block's effects. Scale inputs may be shared (one `R2` or
    # reference across margins), so they are checked against effects only.
    effects = Set{Int}(Iterators.flatten(vec(b.effects) for b in out))
    for block in out
        block isa R2D2AdaptiveCenteringBlock || continue
        for s in block.scales
            isdisjoint(_adaptive_scale_coordinates(s), effects) || error(
                "BRM adaptive centering: R2D2 block `$(block.ranef.binding)` ",
                "derives its scale from a random-effect coordinate this ",
                "transform rewrites; refusing an inexact transform.")
        end
    end
    out
end

# The HSGP path deliberately has its own metadata type rather than widening
# `AdaptiveCenteringBlock`.  The latter feeds a small-block Enzyme
# specialization whose exact dispatch and arithmetic are part of the ordinary
# random-effect contract.  Keeping the two paths separate makes adding HSGPs a
# zero-change operation for that hot path.
struct _HSGPAdaptiveCenteringBlock
    logical::Symbol
    term::Symbol
    # The compiled model may use a different fixed exponent for every basis.
    # Both the transform and retrospective scorer must start in that frame.
    target_c::Vector{Float64}
    effects::Vector{Int}
    length_scales::Vector{Int}
    length_scale_lower::Vector{Float64}
    sd::Int
    sd_lower::Float64
    omega2::Matrix{Float64}
    # Periodic spectral geometry: the compiler-owned harmonic index of each
    # basis column (`[1, 2, ..., K, 1, 2, ..., K]` for cosine/sine pairs).
    # Empty for squared-exponential blocks, whose geometry is `omega2` (which
    # is in turn empty for periodic blocks); the two are never populated
    # together, and `_adaptive_hsgp_log_scale` selects the spectrum on this
    # field.
    harmonics::Vector{Float64}
end

Base.show(io::IO, b::_HSGPAdaptiveCenteringBlock) = print(
    io,
    "HSGPAdaptiveCenteringBlock(", b.logical, ", ", b.term,
    ", target_c=", b.target_c, ", ", length(b.effects), " basis weights",
    isempty(b.harmonics) ? ")" : ", periodic)",
)

_adaptive_stan_expr_value(x::StanBlocks.StanExpr) =
    _adaptive_stan_expr_value(x.expr)
_adaptive_stan_expr_value(x) = x

function _adaptive_constraint_value(plan, raw, owner, role)
    value = _adaptive_stan_expr_value(raw)
    value isa Symbol && begin
        haskey(plan.data, value) || error(
            "BRM adaptive centering: HSGP `$owner` $role constraint refers to " *
            "compiler-owned data `$value`, but the generative plan does not " *
            "carry that value.",
        )
        value = plan.data[value]
    end
    value
end

function _adaptive_lower_bounds(plan, output::BRMOutput, n::Int, owner, role)
    constraints = output.constraints
    unsupported = setdiff(collect(keys(constraints)), [:lower])
    isempty(unsupported) || error(
        "BRM adaptive centering: HSGP `$owner` $role uses unsupported Stan " *
        "constraint(s) $(Tuple(unsupported)); this first contract supports a " *
        "finite lower bound and no upper/offset/multiplier transform.",
    )
    haskey(constraints, :lower) || error(
        "BRM adaptive centering: HSGP `$owner` $role is not lower-bounded; " *
        "the compiled unconstrained-to-physical transform is unsupported.",
    )
    raw = _adaptive_constraint_value(plan, constraints.lower, owner, role)
    values = raw isa Real ? fill(Float64(raw), n) : collect(Float64, raw)
    length(values) == n || error(
        "BRM adaptive centering: HSGP `$owner` $role has $(length(values)) " *
        "lower bounds for $n compiler-owned coordinates.",
    )
    all(isfinite, values) || error(
        "BRM adaptive centering: HSGP `$owner` $role lower bounds must be finite.",
    )
    values
end

function _adaptive_same_hsgp_owner(resolved, owner, role)
    declaration = resolved.output.declaration
    (!isnothing(declaration) && declaration.target === owner.target) || error(
        "BRM adaptive centering: HSGP `$(owner.target)` $role resolved to a " *
        "different compiler declaration; refusing crossed term metadata.",
    )
    resolved
end

"""
    _adaptive_hsgp_centering_blocks(model, unc_names)

Resolve the compiled coordinates and fixed spectral geometry for every
ungrouped and grouped squared-exponential HSGP and every ungrouped periodic
HSGP in an `SBBRMI` or `GenerativePlan`. A grouped term contributes one block
per group level; all of them share the term's spectral scales, so each
level's basis weights adapt as independent scalar cells around the same
per-basis frame. A periodic term contributes one block whose `harmonics`
carry the cosine/sine spectrum; its cells join the same HSGP pair order as
squared-exponential cells.

This is the backend-internal companion to [`adaptive_centering_blocks`](@ref).
It follows BRM's formula-term descriptor to declaration-owned parameter roles,
then reads the declaration's compiler-owned spectral data binding (`omega2`
for squared-exponential terms, `harmonics` for periodic ones).  It never
parses generated Stan or assumes a global parameter order.  The metadata is
intentionally fail-closed: unknown covariances, grouped periodic terms,
bounded transforms, and declaration/artifact coordinate drift raise before a
reparametrizer is built.
"""
function _adaptive_hsgp_centering_blocks(model, unc_names)
    descriptor = brm_descriptor(model)
    plan = descriptor.plan
    pos = _ranef_name_positions(unc_names)
    out = _HSGPAdaptiveCenteringBlock[]

    for predictor in linear_predictors(plan.parent)
        logical = predictor.name
        for entry in _brm_term_coordinate_entries(plan.parent, logical)
            getf(entry.value) === hsgp || continue
            kw = getkwargs(entry.value)
            covariance = _sb_gp_cov(kw, :hsgp)
            if covariance === :periodic
                _adaptive_periodic_hsgp_block!(
                    out, descriptor, plan, unc_names, logical, entry, kw)
                continue
            end
            covariance === :exp_quad || error(
                "BRM adaptive centering: HSGP `$(entry.term)` on predictor " *
                "`$logical` uses covariance `$covariance`; this contract " *
                "supports the squared-exponential and periodic geometries.",
            )
            if haskey(kw, :by)
                _adaptive_grouped_hsgp_blocks!(
                    out, descriptor, plan, pos, unc_names, logical, entry, kw)
                continue
            end

            weights = brm_term_coordinates(
                descriptor, logical, unc_names;
                term=entry.term, parameter=:basis_weights,
            )
            owner = weights.output.declaration
            isnothing(owner) && error(
                "BRM adaptive centering: HSGP `$(entry.term)` basis weights " *
                "have no compiler declaration owner.",
            )
            if owner.family isa Symbol
                owner.family in (
                    :_sb_hsgp, :_sb_hsgp_aniso,
                    :_sb_hsgp_partial, :_sb_hsgp_partial_aniso,
                    :_sb_hsgp_latent, :_sb_hsgp_latent_orthogonal,
                ) || error(
                    "BRM adaptive centering: HSGP `$(entry.term)` resolved to " *
                    "unsupported emitted family `$(owner.family)`.",
                )
            end
            isempty(weights.output.constraints) || error(
                "BRM adaptive centering: HSGP `$(entry.term)` basis weights " *
                "are constrained, so they are not the emitted standard-normal " *
                "`beta_raw` coordinates this transform requires.",
            )

            rho = _adaptive_same_hsgp_owner(brm_term_coordinates(
                descriptor, logical, unc_names;
                term=entry.term, parameter=:length_scale,
            ), owner, "length scale")
            sigma = _adaptive_same_hsgp_owner(brm_term_coordinates(
                descriptor, logical, unc_names;
                term=entry.term, parameter=:sd,
            ), owner, "marginal SD")
            length(rho.coordinates) >= 1 || error(
                "BRM adaptive centering: HSGP `$(entry.term)` has no length-scale coordinate.",
            )
            length(sigma.coordinates) == 1 || error(
                "BRM adaptive centering: HSGP `$(entry.term)` must have one marginal-SD coordinate.",
            )

            omega_key = get(owner.keywords, :omega2, nothing)
            omega_key isa Symbol || error(
                "BRM adaptive centering: HSGP `$(entry.term)` declaration does " *
                "not expose its compiler-owned `omega2` data binding.",
            )
            omega_raw = get(plan.data, omega_key, nothing)
            omega_raw isa AbstractMatrix{<:Real} || error(
                "BRM adaptive centering: HSGP `$(entry.term)` compiler data " *
                "`$omega_key` is not a real spectral-frequency matrix.",
            )
            omega2 = Matrix{Float64}(omega_raw)
            size(omega2) == (length(weights.coordinates), length(rho.coordinates)) ||
                error(
                    "BRM adaptive centering: HSGP `$(entry.term)` owns " *
                    "$(length(weights.coordinates)) basis weights and " *
                    "$(length(rho.coordinates)) length scales, but `$omega_key` " *
                    "has size $(size(omega2)).",
                )

            rho_lower = _adaptive_lower_bounds(
                plan, rho.output, length(rho.coordinates), entry.term,
                "length scale",
            )
            sigma_lower = only(_adaptive_lower_bounds(
                plan, sigma.output, 1, entry.term, "marginal SD",
            ))
            push!(out, _HSGPAdaptiveCenteringBlock(
                logical, entry.term,
                _brm_hsgp_centeredness(kw, length(weights.coordinates)),
                collect(weights.coordinates),
                collect(rho.coordinates), rho_lower, only(sigma.coordinates),
                sigma_lower, omega2, Float64[],
            ))
        end
    end

    claimed_effects = Int[]
    claimed_scales = Int[]
    seen_terms = Set{Tuple{Symbol,Symbol}}()
    for block in out
        append!(claimed_effects, block.effects)
        # One grouped term contributes one block per group level; all of them
        # share the term's length-scale and marginal-SD coordinates, so scales
        # are claimed once per (predictor, term), while every weight cell is
        # claimed exactly once.
        (block.logical, block.term) in seen_terms && continue
        push!(seen_terms, (block.logical, block.term))
        append!(claimed_scales, block.length_scales)
        push!(claimed_scales, block.sd)
    end
    length(unique(claimed_effects)) == length(claimed_effects) &&
        length(unique(claimed_scales)) == length(claimed_scales) || error(
        "BRM adaptive centering: emitted HSGP blocks claim overlapping " *
        "unconstrained coordinates; refusing an ambiguous transform.",
    )
    out
end

# Ungrouped periodic HSGP weights for online adaptation. The cosine/sine
# basis weights are conditionally independent given their spectral scales
# exactly like squared-exponential weights, so one term contributes one
# ordinary `_HSGPAdaptiveCenteringBlock` with `harmonics` populated and
# `omega2` empty; `_adaptive_hsgp_log_scale` selects the Bessel spectrum on
# that field. The compiled periodic emission is always fully non-centered
# (sbimpl refuses a periodic `centeredness`), so `target_c` is zeros by
# construction and checked here. Grouped periodic HSGP is not emitted by the
# compiler; a `by=` keyword reaching this path refuses rather than
# misrouting into the grouped squared-exponential layout.
function _adaptive_periodic_hsgp_block!(
        out, descriptor, plan, unc_names, logical, entry, kw)
    haskey(kw, :by) && error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` on predictor " *
        "`$logical` carries `by=`; the grouped periodic basis is not " *
        "implemented.",
    )

    weights = brm_term_coordinates(
        descriptor, logical, unc_names;
        term=entry.term, parameter=:basis_weights,
    )
    owner = weights.output.declaration
    isnothing(owner) && error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` basis weights " *
        "have no compiler declaration owner.",
    )
    if owner.family isa Symbol
        owner.family === :_sb_hsgp_periodic || error(
            "BRM adaptive centering: periodic HSGP `$(entry.term)` resolved " *
            "to unsupported emitted family `$(owner.family)`.",
        )
    end
    isempty(weights.output.constraints) || error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` basis weights " *
        "are constrained, so they are not the emitted standard-normal " *
        "`beta_raw` coordinates this transform requires.",
    )

    rho = _adaptive_same_hsgp_owner(brm_term_coordinates(
        descriptor, logical, unc_names;
        term=entry.term, parameter=:length_scale,
    ), owner, "length scale")
    sigma = _adaptive_same_hsgp_owner(brm_term_coordinates(
        descriptor, logical, unc_names;
        term=entry.term, parameter=:sd,
    ), owner, "marginal SD")
    length(rho.coordinates) == 1 || error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` owns " *
        "$(length(rho.coordinates)) length-scale coordinates; the cosine/sine " *
        "spectrum needs exactly one.",
    )
    length(sigma.coordinates) == 1 || error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` must have one " *
        "marginal-SD coordinate.",
    )

    harmonics_key = get(owner.keywords, :harmonics, nothing)
    harmonics_key isa Symbol || error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` declaration " *
        "does not expose its compiler-owned `harmonics` data binding.",
    )
    harmonics_raw = get(plan.data, harmonics_key, nothing)
    harmonics_raw isa AbstractVector{<:Real} || error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` compiler data " *
        "`$harmonics_key` is not a real harmonic-index vector.",
    )
    harmonics = Vector{Float64}(harmonics_raw)
    n_weights = length(weights.coordinates)
    length(harmonics) == n_weights || error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` owns " *
        "$n_weights basis weights, but `$harmonics_key` carries " *
        "$(length(harmonics)) harmonic indices.",
    )
    iseven(n_weights) || error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` owns " *
        "$n_weights basis weights; the cosine/sine basis needs an even count.",
    )
    harmonics == _brm_hsgp_periodic_harmonics(n_weights ÷ 2) || error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` compiler data " *
        "`$harmonics_key` is not the cosine/sine harmonic index vector; " *
        "refusing crossed term metadata.",
    )

    rho_lower = _adaptive_lower_bounds(
        plan, rho.output, length(rho.coordinates), entry.term,
        "length scale",
    )
    sigma_lower = only(_adaptive_lower_bounds(
        plan, sigma.output, 1, entry.term, "marginal SD",
    ))
    target_c = _brm_hsgp_centeredness(kw, n_weights)
    all(iszero, target_c) || error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` carries a " *
        "non-zero compiled centeredness; partial centering supports only " *
        "the exp_quad HSGP spectrum.",
    )
    push!(out, _HSGPAdaptiveCenteringBlock(
        logical, entry.term, target_c,
        collect(weights.coordinates),
        collect(rho.coordinates), rho_lower, only(sigma.coordinates),
        sigma_lower, Matrix{Float64}(undef, 0, 0), harmonics,
    ))
    out
end

# Grouped squared-exponential HSGP weights for online adaptation. Spectral
# hyperparameters stay shared across groups, so one term contributes one
# ordinary `_HSGPAdaptiveCenteringBlock` per group level: each block's
# `effects` are that level's basis-weight coordinates and every block shares
# the term's per-basis compiled frame, length scales, marginal SD, and
# spectral frequencies. The flat index rule mirrors the emitted
# `to_matrix(zflat, n_basis, n_groups)'` layout: flat position `(g-1)*B+b`
# addresses frequency `b` in level `g`.
function _adaptive_grouped_hsgp_blocks!(
        out, descriptor, plan, pos, unc_names, logical, entry, kw)
    args = getargs(entry.value)
    K, _ = _sb_hsgp_options(kw, length(args))
    B = prod(K)
    axnames = Tuple(name(_sb_named_inner(:hsgp, a)) for a in args)
    gname = name(kw[:by])
    n_key = Symbol(:n_, gname)
    G = get(plan.data, n_key, nothing)
    G isa Integer || error(
        "BRM adaptive centering: grouped HSGP `$(entry.term)` on predictor " *
        "`$logical` has no integer group count under `$n_key`; refusing " *
        "an ambiguous weight layout.",
    )
    field = Symbol(:hsgpw_, join(string.(axnames), "_"))
    flat_name = Symbol(:zflat_, field, :_, gname)
    flat_names = ["$(flat_name).$i" for i in 1:(G*B)]
    flat = _adaptive_named_indices(pos, flat_names, entry.term,
        "grouped HSGP basis weights")

    rho = brm_term_coordinates(
        descriptor, logical, unc_names;
        term=entry.term, parameter=:length_scale,
    )
    sigma = brm_term_coordinates(
        descriptor, logical, unc_names;
        term=entry.term, parameter=:sd,
    )
    owner = rho.output.declaration
    isnothing(owner) && error(
        "BRM adaptive centering: grouped HSGP `$(entry.term)` length scale " *
        "has no compiler declaration owner.",
    )
    if owner.family isa Symbol
        owner.family in (:_sb_hsgp_by, :_sb_hsgp_by_aniso) || error(
            "BRM adaptive centering: grouped HSGP `$(entry.term)` resolved to " *
            "unsupported emitted family `$(owner.family)`.",
        )
    end
    sigma = _adaptive_same_hsgp_owner(sigma, owner, "marginal SD")
    get(owner.keywords, :beta, nothing) === Symbol(:b_, field, :_, gname) || error(
        "BRM adaptive centering: grouped HSGP `$(entry.term)` declaration " *
        "does not carry the expected per-group weight block; refusing " *
        "crossed term metadata.",
    )
    length(rho.coordinates) >= 1 || error(
        "BRM adaptive centering: grouped HSGP `$(entry.term)` has no length-scale coordinate.",
    )
    length(sigma.coordinates) == 1 || error(
        "BRM adaptive centering: grouped HSGP `$(entry.term)` must have one marginal-SD coordinate.",
    )
    omega_key = get(owner.keywords, :omega2, nothing)
    omega_key isa Symbol || error(
        "BRM adaptive centering: grouped HSGP `$(entry.term)` declaration does " *
        "not expose its compiler-owned `omega2` data binding.",
    )
    omega_raw = get(plan.data, omega_key, nothing)
    omega_raw isa AbstractMatrix{<:Real} || error(
        "BRM adaptive centering: grouped HSGP `$(entry.term)` compiler data " *
        "`$omega_key` is not a real spectral-frequency matrix.",
    )
    omega2 = Matrix{Float64}(omega_raw)
    size(omega2) == (B, length(rho.coordinates)) || error(
        "BRM adaptive centering: grouped HSGP `$(entry.term)` owns " *
        "$B basis weights and $(length(rho.coordinates)) length scales, but " *
        "`$omega_key` has size $(size(omega2)).",
    )
    rho_lower = _adaptive_lower_bounds(
        plan, rho.output, length(rho.coordinates), entry.term,
        "length scale",
    )
    sigma_lower = only(_adaptive_lower_bounds(
        plan, sigma.output, 1, entry.term, "marginal SD",
    ))
    target_c = _brm_hsgp_centeredness(kw, B)
    rho_idx = collect(rho.coordinates)
    sd_idx = only(sigma.coordinates)
    for g in 1:G
        push!(out, _HSGPAdaptiveCenteringBlock(
            logical, entry.term, target_c,
            flat[(g-1)*B+1:g*B],
            rho_idx, rho_lower, sd_idx, sigma_lower, omega2, Float64[],
        ))
    end
    out
end

function _adaptive_hsgp_log_scale(x::AbstractVector,
                                  block::_HSGPAdaptiveCenteringBlock,
                                  basis::Int)
    1 <= basis <= length(block.effects) || throw(BoundsError(block.effects, basis))
    if !isempty(block.harmonics)
        length(block.length_scales) == 1 || error(
            "BRM adaptive centering: periodic HSGP `$(block.term)` owns " *
            "$(length(block.length_scales)) length scales; the cosine/sine " *
            "spectrum needs exactly one.",
        )
        sigma = block.sd_lower + exp(x[block.sd])
        rho = block.length_scale_lower[1] + exp(x[block.length_scales[1]])
        return _brm_hsgp_periodic_log_scale(block.harmonics[basis], sigma, rho)
    end
    sigma = block.sd_lower + exp(x[block.sd])
    value = log(sigma)
    for axis in eachindex(block.length_scales)
        rho = block.length_scale_lower[axis] + exp(x[block.length_scales[axis]])
        value += 0.5 * log(rho * 2.5066282746310002)
        value -= 0.25 * rho * rho * block.omega2[basis, axis]
    end
    value
end

# The cdar path has its own metadata type for the same reason the HSGP path
# does: `AdaptiveCenteringBlock` feeds the small-block Enzyme specialization
# for ordinary random effects, and walk cells must not perturb that dispatch.
# Cells are scalar and independent like HSGP basis weights — location zero and
# one marginal prior spread each — so per-pair transport stays O(1) and the
# whole walk stays linear, unlike a time-triangular map whose per-cell history
# reread would cost O((P*W)^2) per gradient with no legal Enzyme cache.
struct _CDARAdaptiveCenteringBlock
    logical::Symbol
    term::Symbol
    n_groups::Int
    n_steps::Int
    # Column-major `eta` unconstrained indices: position `p + (w - 1) * P`
    # addresses group `p`, step `w`, matching the emitted `eta.1`, ... order.
    effects::Vector{Int}
    sigma::Int
    sigma_lower::Float64
    rho::Int
    # Frozen per-group marginal variances `C[p, p]` of `C = L * L'`.
    cdiag::Vector{Float64}
end

Base.show(io::IO, b::_CDARAdaptiveCenteringBlock) = print(
    io,
    "CDARAdaptiveCenteringBlock(", b.logical, ", ", b.term,
    ", ", b.n_groups, "×", b.n_steps, " walk cells)",
)

function _adaptive_cdar_physical(block::_CDARAdaptiveCenteringBlock, x::AbstractVector)
    sigma = block.sigma_lower + exp(x[block.sigma])
    rho = 1 / (1 + exp(-x[block.rho]))
    sigma, rho
end

function _adaptive_cdar_log_scale(x::AbstractVector,
                                  block::_CDARAdaptiveCenteringBlock,
                                  pair::Int)
    1 <= pair <= length(block.effects) || throw(BoundsError(block.effects, pair))
    p = mod(pair - 1, block.n_groups) + 1
    w = div(pair - 1, block.n_groups) + 1
    sigma, rho = _adaptive_cdar_physical(block, x)
    isfinite(sigma) && sigma > 0 || error(
        "BRM adaptive centering: cdar `$(block.term)` marginal scale is not " *
        "finite-positive at this position.",
    )
    0 <= rho <= 1 || error(
        "BRM adaptive centering: cdar `$(block.term)` persistence left " *
        "[0, 1] at this position.",
    )
    one_minus_rho2 = 1 - rho * rho
    spread = if one_minus_rho2 <= 1e-12
        Float64(w)
    else
        exp(log1p(-rho^(2w)) - log(one_minus_rho2))
    end
    log(sigma) + 0.5 * log(block.cdiag[p]) + 0.5 * log(spread)
end

function _adaptive_cdar_rho_bounds(plan, output::BRMOutput, owner)
    constraints = output.constraints
    unsupported = setdiff(collect(keys(constraints)), (:lower, :upper))
    isempty(unsupported) || error(
        "BRM adaptive centering: cdar `$owner` persistence uses unsupported Stan " *
        "constraint(s) $(Tuple(unsupported)); this contract supports a " *
        "[0, 1] interval and no offset/multiplier transform.",
    )
    lower = _adaptive_constraint_value(plan, get(constraints, :lower, nothing), owner, "persistence")
    upper = _adaptive_constraint_value(plan, get(constraints, :upper, nothing), owner, "persistence")
    lower == 0.0 || error(
        "BRM adaptive centering: cdar `$owner` persistence lower bound is " *
        "$lower, not 0; the compiled logit transform is unsupported.",
    )
    upper == 1.0 || error(
        "BRM adaptive centering: cdar `$owner` persistence upper bound is " *
        "$upper, not 1; the compiled logit transform is unsupported.",
    )
    nothing
end

"""
    _adaptive_cdar_centering_blocks(model, unc_names)

Resolve the compiled coordinates and frozen walk geometry for every
`cdar(step; by=group, cor=C)` term in an `SBBRMI` or `GenerativePlan`.

This is the backend-internal companion to [`adaptive_centering_blocks`](@ref),
following the same descriptor-to-declaration route as
[`_adaptive_hsgp_centering_blocks`](@ref). Each of the `P * W` innovations is
one scalar cell with zero location and its marginal prior spread
`sigma * sqrt(C[p, p] * (1 - rho^(2w)) / (1 - rho^2))`; `c=0` is the emitted
`eta` frame. The metadata is intentionally fail-closed: a non-`[0, 1]`
persistence transform, a non-lower-bounded scale, a missing or misshapen
frozen factor, and declaration/artifact coordinate drift all raise before a
reparametrizer is built.
"""
function _adaptive_cdar_centering_blocks(model, unc_names)
    descriptor = brm_descriptor(model)
    plan = descriptor.plan
    out = _CDARAdaptiveCenteringBlock[]

    for predictor in linear_predictors(plan.parent)
        logical = predictor.name
        for entry in _brm_term_coordinate_entries(plan.parent, logical)
            getf(entry.value) === cdar || continue

            innovations = brm_term_coordinates(
                descriptor, logical, unc_names;
                term=entry.term, parameter=:innovations,
            )
            owner = innovations.output.declaration
            isnothing(owner) && error(
                "BRM adaptive centering: cdar `$(entry.term)` innovations " *
                "have no compiler declaration owner.",
            )
            if owner.family isa Symbol
                owner.family === :_sb_cdar || error(
                    "BRM adaptive centering: cdar `$(entry.term)` resolved to " *
                    "unsupported emitted family `$(owner.family)`.",
                )
            end
            isempty(innovations.output.constraints) || error(
                "BRM adaptive centering: cdar `$(entry.term)` innovations " *
                "are constrained, so they are not the emitted standard-normal " *
                "`eta` coordinates this transform requires.",
            )

            sigma = _adaptive_same_hsgp_owner(brm_term_coordinates(
                descriptor, logical, unc_names;
                term=entry.term, parameter=:sd,
            ), owner, "marginal SD")
            rho = _adaptive_same_hsgp_owner(brm_term_coordinates(
                descriptor, logical, unc_names;
                term=entry.term, parameter=:ar,
            ), owner, "persistence")
            length(sigma.coordinates) == 1 || error(
                "BRM adaptive centering: cdar `$(entry.term)` must have one marginal-SD coordinate.",
            )
            length(rho.coordinates) == 1 || error(
                "BRM adaptive centering: cdar `$(entry.term)` must have one persistence coordinate.",
            )

            l_key = get(owner.keywords, :L, nothing)
            l_key isa Symbol || error(
                "BRM adaptive centering: cdar `$(entry.term)` declaration does " *
                "not expose its frozen correlation-factor `L` data binding.",
            )
            l_raw = get(plan.data, l_key, nothing)
            l_raw isa AbstractMatrix{<:Real} || error(
                "BRM adaptive centering: cdar `$(entry.term)` compiler data " *
                "`$l_key` is not a real correlation-factor matrix.",
            )
            ng_key = get(owner.keywords, :n_groups, nothing)
            ns_key = get(owner.keywords, :n_steps, nothing)
            n_groups = ng_key isa Symbol ? get(plan.data, ng_key, nothing) : nothing
            n_steps = ns_key isa Symbol ? get(plan.data, ns_key, nothing) : nothing
            n_groups isa Integer && n_steps isa Integer || error(
                "BRM adaptive centering: cdar `$(entry.term)` declaration does " *
                "not expose integer group/step counts.",
            )
            P, W = Int(n_groups), Int(n_steps)
            L = Matrix{Float64}(l_raw)
            size(L) == (P, P) || error(
                "BRM adaptive centering: cdar `$(entry.term)` owns a " *
                "$(size(L)) factor for $P groups.",
            )
            length(innovations.coordinates) == P * W || error(
                "BRM adaptive centering: cdar `$(entry.term)` owns " *
                "$(length(innovations.coordinates)) innovations for a " *
                "$P×$W walk.",
            )
            cdiag = vec(sum(abs2, L; dims=2))
            all(c -> isfinite(c) && c > 0, cdiag) || error(
                "BRM adaptive centering: cdar `$(entry.term)` frozen factor " *
                "has a non-finite or non-positive marginal variance.",
            )

            sigma_lower = only(_adaptive_lower_bounds(
                plan, sigma.output, 1, entry.term, "marginal SD",
            ))
            _adaptive_cdar_rho_bounds(plan, rho.output, entry.term)
            push!(out, _CDARAdaptiveCenteringBlock(
                logical, entry.term, P, W,
                collect(innovations.coordinates),
                only(sigma.coordinates), sigma_lower, only(rho.coordinates),
                cdiag,
            ))
        end
    end

    claimed = Int[]
    for block in out
        append!(claimed, block.effects)
        push!(claimed, block.sigma)
        push!(claimed, block.rho)
    end
    length(unique(claimed)) == length(claimed) || error(
        "BRM adaptive centering: emitted cdar blocks claim overlapping " *
        "unconstrained coordinates; refusing an ambiguous transform.",
    )
    out
end

# Stan's native `cholesky_factor_corr[K]` constrain transform.  Its
# K(K-1)/2 free coordinates are row-major over the strict lower triangle; each
# is mapped with tanh and multiplied by the remaining row stick.  This exact
# spelling is checked against BridgeStan in the adaptive-centering acceptance
# probe, including a nonzero K=3 discriminator.
function _adaptive_cholesky_corr(raw::AbstractVector, K::Int)
    length(raw) == K * (K - 1) ÷ 2 || throw(DimensionMismatch(
        "cholesky_factor_corr[$K] needs $(K * (K - 1) ÷ 2) free values, got $(length(raw))",
    ))
    T = eltype(raw)
    L = zeros(T, K, K)
    L[1, 1] = one(T)
    p = 1
    for i in 2:K
        stick = one(T)
        for j in 1:i-1
            z = tanh(raw[p])
            p += 1
            L[i, j] = stick * z
            stick *= sqrt(one(T) - z * z)
        end
        L[i, i] = stick
    end
    L
end

_adaptive_block_taus(x::AbstractVector, block::AdaptiveCenteringBlock) =
    exp.(x[block.log_scales])
_adaptive_block_taus(x::AbstractVector, block::R2D2AdaptiveCenteringBlock) =
    [exp(_adaptive_log_scale(x, s)) for s in block.scales]

function _adaptive_block_cholesky(x::AbstractVector, block::AbstractAdaptiveCenteringBlock)
    K = block.ranef.n_terms
    L = _adaptive_cholesky_corr(x[block.cholesky_free], K)
    tau = _adaptive_block_taus(x, block)
    tau .* L
end

# The per-draw model-scale factor `C` with fresh effect `b = C * z`,
# reconstructed from one unconstrained draw with no BridgeStan call. Correlated
# blocks rebuild `tau .* L` above; a scalar `(1 | g)` intercept block has no
# Cholesky and its scale is `exp(log_scale)`, so it is the 1×1 factor. Shared
# by centered replay (prediction.jl); the WarmupHMC online path calls
# `_adaptive_block_cholesky` directly and never reaches the intercept branch.
function _adaptive_block_cov_chol(x::AbstractVector, block::AdaptiveCenteringBlock)
    if block.ranef.family in _ADAPTIVE_INTERCEPT_FAMILIES
        block.ranef.n_terms == 1 || error(
            "BRM adaptive centering: intercept block `$(block.ranef.binding)` ",
            "has $(block.ranef.n_terms) terms; the scalar scale path needs one.")
        return fill(exp(x[only(block.log_scales)]), 1, 1)
    end
    _adaptive_block_cholesky(x, block)
end

"""
    adaptive_centering_problem(model, problem, ad_backend; kwargs...)

Wrap a BRM `problem` in WarmupHMC's strictly-online adaptive-centering
transform. Compiled models use BRM's WarmupHMC extension. A `TuringBRMI` uses
the separate joint Turing+WarmupHMC extension and, for now, accepts only the
identity-link, fixed-scale Gaussian model with one default noncentered scalar
random-intercept geometry documented by that method.
"""
function adaptive_centering_problem end

function adaptive_centering_blocks(::TuringBRMI, _unc_names)
    error(
        "Turing backend: `adaptive_centering_blocks` describes compiled Stan " *
        "unconstrained coordinates and does not apply to DynamicPPL models. " *
        "Choose the executable endpoint with " *
        "`TuringBRMI(brmi; centered_groups=...)`.")
end

function adaptive_centering_problem(::TuringBRMI, _problem, _ad_backend;
                                    _kwargs...)
    error(
        "Turing backend: native online adaptive centering requires both " *
        "Turing and WarmupHMC, a DynamicPPL.LogDensityFunction, " *
        "and the supported fixed-scale Gaussian model with one default " *
        "noncentered `(1 | group)` geometry. " *
        "The compiled-Stan coordinate bridge does not apply to DynamicPPL models.")
end
