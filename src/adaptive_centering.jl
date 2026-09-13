# Adaptive partial centering for ordinary random-effect blocks.
#
# This file owns only BRM/Stan emission semantics: which unconstrained
# coordinates are one block's effects, optional LKJ-Cholesky free values, and
# positive scales, plus the pure-Julia reconstruction of C.  WarmupHMC
# owns the online candidate accumulators and window lifecycle through the
# optional extension method of `adaptive_centering_problem`.

const _ADAPTIVE_CORRELATED_FAMILIES = Set((
    :ranef_correlated,
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
))

const _ADAPTIVE_STRATIFIED_FAMILIES = Set((
    :ranef_correlated_by,
    :ranef_correlated_by_draws,
))

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
struct AdaptiveCenteringBlock
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

"""
    adaptive_centering_blocks(model, unc_names) -> Vector{AdaptiveCenteringBlock}

Describe every ordinary random-effect block in `model` for adaptive partial
centering. `model` is an [`SBBRMI`](@ref) or [`GenerativePlan`](@ref),
and `unc_names` is the compiled model's unconstrained parameter-name vector
(for example BridgeStan's `param_unc_names`).

The result supports both BRM endpoints: the default noncentered emission is the
compiled target `c=0`, while `SBBRMI(...; centered_groups=...)` is target `c=1`.
Every intermediate source uses the triangular term-wise map

```
u[k] = c[k] * sum(C[k,l] * z[l] for l < k) + C[k,k]^c[k] * z[k]
```

with `C = diag(tau) * L`. At `K=1`, this reduces to `u = s^c * z` with
`s = exp(log_scale)` and no Cholesky coordinates. Consequently `c=0` is the
literal standardised draw and `c=1` is the literal model-scale effect.

Stratified `gr(g, by=b)` blocks currently raise rather than being silently
left fixed: they carry one `L,tau` frame per stratum and need a separate indexed
metadata contract.
"""
function adaptive_centering_blocks(model, unc_names)
    pos = _ranef_name_positions(unc_names)
    out = AdaptiveCenteringBlock[]
    for ranef in ranef_blocks(model)
        ranef.family in _ADAPTIVE_STRATIFIED_FAMILIES && error(
            "BRM adaptive centering: stratified block `$(ranef.binding)` ",
            "(`$(ranef.family)`, group `$(ranef.group)`, by `$(ranef.by)`) is ",
            "not supported by the first correlated-block contract. It has one ",
            "Cholesky/scale frame per stratum and must not be treated as an ",
            "ordinary single-frame block.",
        )
        is_intercept = ranef.family in _ADAPTIVE_INTERCEPT_FAMILIES
        is_correlated = ranef.family in _ADAPTIVE_CORRELATED_FAMILIES
        (is_intercept || is_correlated) || continue
        K = ranef.n_terms
        n_cholesky = K * (K - 1) ÷ 2
        binding = ranef.binding
        cholesky_names = is_intercept ? String[] :
            ["$(binding)_L.$i" for i in 1:n_cholesky]
        scale_names = is_intercept ? ["$(binding)_log_scale"] :
            ["$(binding)_tau.$i" for i in 1:K]
        target_c = ranef.noncentered ? 0.0 : 1.0
        push!(out, AdaptiveCenteringBlock(
            ranef,
            target_c,
            ranef_coordinates(ranef, unc_names),
            _adaptive_named_indices(pos, cholesky_names, binding, "Cholesky"),
            _adaptive_named_indices(pos, scale_names, binding, "scale"),
        ))
    end

    claimed = Int[]
    for block in out
        append!(claimed, vec(block.effects))
        append!(claimed, block.cholesky_free)
        append!(claimed, block.log_scales)
    end
    length(unique(claimed)) == length(claimed) || error(
        "BRM adaptive centering: emitted correlated blocks claim overlapping ",
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

function _adaptive_block_cholesky(x::AbstractVector, block::AdaptiveCenteringBlock)
    K = block.ranef.n_terms
    L = _adaptive_cholesky_corr(x[block.cholesky_free], K)
    tau = exp.(x[block.log_scales])
    tau .* L
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
