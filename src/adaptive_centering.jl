# Adaptive partial centering for ordinary random-effect blocks and
# squared-exponential HSGP basis weights. Online HSGP adaptation is
# ungrouped-only; the offline pilot selector feeds grouped emission too.
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

The selected per-basis vector feeds `hsgp(...; centeredness=...)` directly —
both the ungrouped and the grouped (`by=`) emission, whose spectral scales
are shared across groups. Online adaptation during WarmupHMC
(`adaptive_centering_problem` HSGP blocks) remains ungrouped-only.
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
end

Base.show(io::IO, b::_HSGPAdaptiveCenteringBlock) = print(
    io,
    "HSGPAdaptiveCenteringBlock(", b.logical, ", ", b.term,
    ", target_c=", b.target_c, ", ", length(b.effects), " basis weights)",
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
ungrouped squared-exponential HSGP in an `SBBRMI` or `GenerativePlan`.

This is the backend-internal companion to [`adaptive_centering_blocks`](@ref).
It follows BRM's formula-term descriptor to declaration-owned parameter roles,
then reads the declaration's compiler-owned `omega2` data binding.  It never
parses generated Stan or assumes a global parameter order.  The metadata is
intentionally fail-closed: grouped and periodic HSGPs, bounded transforms, and
declaration/artifact coordinate drift raise before a reparametrizer is built.
"""
function _adaptive_hsgp_centering_blocks(model, unc_names)
    descriptor = brm_descriptor(model)
    plan = descriptor.plan
    out = _HSGPAdaptiveCenteringBlock[]

    for predictor in linear_predictors(plan.parent)
        logical = predictor.name
        for entry in _brm_term_coordinate_entries(plan.parent, logical)
            getf(entry.value) === hsgp || continue
            kw = getkwargs(entry.value)
            haskey(kw, :by) && error(
                "BRM adaptive centering: grouped HSGP `$(entry.term)` on " *
                "predictor `$logical` is unsupported; its basis weights live " *
                "in a separate per-group field rather than `beta_raw`.",
            )
            covariance = _sb_gp_cov(kw, :hsgp)
            covariance === :exp_quad || error(
                "BRM adaptive centering: HSGP `$(entry.term)` on predictor " *
                "`$logical` uses covariance `$covariance`; this first contract " *
                "supports only the non-periodic squared-exponential geometry.",
            )

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
                sigma_lower, omega2,
            ))
        end
    end

    claimed = Int[]
    for block in out
        append!(claimed, block.effects)
        append!(claimed, block.length_scales)
        push!(claimed, block.sd)
    end
    length(unique(claimed)) == length(claimed) || error(
        "BRM adaptive centering: emitted HSGP blocks claim overlapping " *
        "unconstrained coordinates; refusing an ambiguous transform.",
    )
    out
end

function _adaptive_hsgp_log_scale(x::AbstractVector,
                                  block::_HSGPAdaptiveCenteringBlock,
                                  basis::Int)
    1 <= basis <= length(block.effects) || throw(BoundsError(block.effects, basis))
    sigma = block.sd_lower + exp(x[block.sd])
    value = log(sigma)
    for axis in eachindex(block.length_scales)
        rho = block.length_scale_lower[axis] + exp(x[block.length_scales[axis]])
        value += 0.5 * log(rho * 2.5066282746310002)
        value -= 0.25 * rho * rho * block.omega2[basis, axis]
    end
    value
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
