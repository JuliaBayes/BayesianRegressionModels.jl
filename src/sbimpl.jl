using StanBlocks
import StanBlocks: RaggedVector


# ==============================================================================
# SlicModel helpers (ported verbatim from an external PKPD codebase's qt.jl,
# `popefs`/`ranefs`/`popranefs`/`cdirichlet` family, lines 303-344). Kept here
# as module-local bindings so the walker can emit calls to them by name without
# depending on that package. Duplication is intentional for now.
# ==============================================================================

# Formula-term stubs needed so the @brm macro can parse example formulas
# whose parser side is not yet owned by macro.jl. Empty function bindings only;
# the actual backend is implemented below (`_sb_predictor_term!`). These may
# migrate to macro.jl once the frontend agent adds shared stubs.

"""
    me(x_obs, sd_x)

brms-style measurement-error predictor marker. `me(x_obs, sd_x)` declares
that the observed `x_obs` has Gaussian measurement error with SD `sd_x`;
the backend allocates a latent `x_true` and emits the observation
likelihood `x_obs ~ Normal(x_true, sd_x)`. Dispatch tag — see `_sb_me`.
"""
function me end

"""
    s(x)

Add a penalized one-dimensional thin-plate regression spline to an `SBBRMI`
linear predictor. The fixed rank-10 basis contains the
unpenalized null space `{1, x}` and eight penalty-whitened range columns whose
shared smoothing standard deviation has a standard half-normal prior.

The public syntax is exactly one finite numeric predictor, `s(x)`. At least ten
unique training values are required; keyword arguments such as `k=` and
`knots=` are not supported. The term owns its complete smooth contribution and
does not receive an additional population coefficient.

Prediction and replay through [`reprocess`](@ref) or [`restan_data`](@ref) use
the frozen training basis by default. This marker is implemented only by the
StanBlocks backend; it is not a deterministic B-spline expansion and is not
available to `VBRMI`.

See [Formula terms](@ref) for an example and a comparison with
`bs(...)` formulas. Dispatch tag — see `_sb_s`.
"""
function s end

"""
    t2(x, z; k=(5, 5), basis=(:cr, :cr), full=false)

Tensor-product smooth marker. The StanBlocks backend builds cubic-regression-
spline margins, separates their null and penalized spaces, and gives the RR,
RN, and NR tensor blocks independent smoothing scales. The current surface is
two-dimensional, supports only `basis=(:cr, :cr)`, and requires `full=false`.
Prediction/replay freezes the training knots and penalty decomposition by
default. See [Formula terms](@ref) for the full contract. Dispatch tag — see
`_sb_t2`.
"""
function t2 end

"""
    ar(time; p=1)

Autoregressive-noise predictor marker. Adds an AR(p) noise process
ordered by `time`. Only `p=1` is supported in the current sbimpl
emitter. Dispatch tag — see `_sb_ar1`.
"""
function ar end

"""
    dar(time; p=1)

Differenced-autoregressive trajectory marker. The StanBlocks backend emits the
weekly path

`x[1] = 0`, `d[t] = beta * d[t-1] + sigma * z[t]`,
`x[t+1] = x[t] + d[t]`,

with `0 <= beta <= 1`, `sigma > 0`, and standardized innovations `z`. It is a
direct predictor summand, so a formula intercept supplies the initial level and
no additional population coefficient multiplies the path. The time values must
be finite and strictly increasing. Only `p=1` is supported. Dispatch tag — see
`_sb_dar1`.
"""
function dar end

"""
    Horseshoe

Carvalho-Polson-Scott horseshoe shrinkage prior marker. Use as a prior
on a coefficient:

```julia
coef ~ Horseshoe()
coef ~ Horseshoe(local_scale=0.5, global_scale=0.1)
```

`local_scale` and `global_scale` are positive finite formula constants and
default to one. sbimpl emits the standard reparameterised hierarchy
`beta = raw * lambda * tau`. Each scalar call owns its own `tau`; "global"
means global only within that call, not shared across several coefficients.
Marker struct only — the `@brm` parser never constructs an instance.
"""
struct Horseshoe end

_sb_interval_literal(x::Real) = Float64(x)
_sb_interval_literal(x::NamedColumn) = begin
    parent(x) isa MissingColumn && name(x) in (:pi, :π) || error(
        "CircularVonMises: `interval` must be a compile-time numeric pair; " *
        "got formula symbol `$(name(x))`")
    Float64(pi)
end
function _sb_interval_literal(x::ExprColumn)
    args = getargs(x)
    length(args) == 1 && getf(x) === (-) && return -_sb_interval_literal(args[1])
    length(args) == 1 && getf(x) === (+) && return _sb_interval_literal(args[1])
    error("CircularVonMises: `interval` must be a compile-time numeric pair; got $(x)")
end
_sb_interval_literal(x) = error(
    "CircularVonMises: `interval` must be a compile-time numeric pair; got $(typeof(x))")

function _sb_circular_interval(kwargs)
    interval = get(kwargs, :interval, (-Float64(pi), Float64(pi)))
    interval isa Tuple && length(interval) == 2 || error(
        "CircularVonMises: `interval` must be a 2-tuple `(lo, hi)`, got $(repr(interval))")
    lo, hi = map(_sb_interval_literal, interval)
    all(isfinite, (lo, hi)) && lo < hi || error(
        "CircularVonMises: `interval` endpoints must be finite with lo < hi, got $(repr((lo, hi)))")
    isapprox(hi - lo, 2 * Float64(pi); rtol=8eps(Float64), atol=8eps(Float64)) || error(
        "CircularVonMises: `interval` must have length 2pi, got $(hi - lo)")
    (lo, hi)
end

function _check_term_kwargs(::Type{<:CircularVonMises}, kwargs)
    unknown = filter(!=(:interval), keys(kwargs))
    isempty(unknown) || error(
        "CircularVonMises: unsupported keyword(s): $(join(unknown, ", ")); " *
        "the only supported keyword is `interval`")
    _sb_circular_interval(kwargs)
    nothing
end

function _check_term_kwargs(::Type{<:Ordinal}, kwargs)
    allowed = (:discrimination, :per_threshold)
    unknown = filter(k -> k ∉ allowed, keys(kwargs))
    isempty(unknown) || error(
        "Ordinal: unsupported keyword(s): $(join(unknown, ", ")); " *
        "supported keywords are `discrimination` and `per_threshold`")
    nothing
end

popefs = StanBlocks.@slic begin
    n_covariates = dims(X)[2]
    beta_pop ~ std_normal(; n=n_covariates)
    return X * beta_pop
end

# Population effects with coefficient-specific Normal priors. Kept as a
# sibling of `popefs` so the default submodel (and every downstream direct use
# of it) remains byte-for-byte unchanged. The caller supplies one location and
# scale per design column; StanBlocks keeps the same prefixed `beta_pop` vector
# parameter name as the default submodel.
_popefs_normal = StanBlocks.@slic begin
    n_covariates = dims(X)[2]
    beta_pop ~ normal(beta_loc, beta_scale; n=n_covariates)
    return X * beta_pop
end

# Coefficient-returning siblings of `popefs` / `_popefs_normal`. Identical
# parameters and identical priors; the ONLY difference is that they return
# `beta_pop` instead of `X * beta_pop`, which is what Stan's fused
# `normal_id_glm_lpdf` wants (it takes the design matrix and the coefficients
# separately and never materialises the product on the autodiff tape).
# Siblings rather than a flag on the originals so every unfused path keeps its
# emission byte for byte. Selected only by `_sb_fuse_normal_id_glm!` below.
#
# LOCKSTEP: `_brm_declaration_role` (`descriptor.jl`) maps population-block
# submodel names to the `:population_effect` role, which is what carries the
# `popcoefnames` labels onto the posterior columns. A name added here that is
# not added there silently drops those labels.
_popefs_coefs = StanBlocks.@slic begin
    n_covariates = dims(X)[2]
    beta_pop ~ std_normal(; n=n_covariates)
    return beta_pop
end

_popefs_normal_coefs = StanBlocks.@slic begin
    n_covariates = dims(X)[2]
    beta_pop ~ normal(beta_loc, beta_scale; n=n_covariates)
    return beta_pop
end

cdirichlet = StanBlocks.@slic begin
    increments ~ dirichlet(alpha)
    return cumulative_sum(increments)
end

c0dirichlet = StanBlocks.@slic begin
    increments ~ dirichlet(alpha)
    return cumulative_sum(increments) - increments[1]
end

c01dirichlet = StanBlocks.@slic begin
    increments ~ dirichlet(alpha)
    return append_row(0., cumulative_sum(increments))
end

# Monotonic effect contrast (Buerkner & Charpentier 2018). Returns the per-obs
# contrast vector; the walker hcat's it as one column of X_pop so popefs
# supplies the free beta (matches vimpl's free-beta `mo` variant, not `mo1`).
# Named `_sb_mo` to avoid clashing with vimpl's marker function `mo`.
# The Dirichlet concentration arrives as DATA rather than being built inline,
# so `simplex(<lp|:>, mo1(c)) ~ Dirichlet(...)` configures this ONE submodel
# instead of selecting a second copy of it. Julia supplies `rep_vector(1., K-1)`
# when the formula says nothing, which is the density the inline form had.
_sb_mo = StanBlocks.@slic begin
    simplex_incr ~ dirichlet(alpha)
    return cumulative_sum(append_row(0., simplex_incr))[x]
end

# (1 | g) random intercept. Mirrors vimpl's scalar grouped_normal + chol(n=1)
# collapse: Part{chol}(1x1) -> log_scale ~ N(0,1), L[1,1] = exp(log_scale);
# Part{grouped_normal}(n_groups, 1) -> xi ~ N(0,1), values = L[1,1] * xi;
# per-obs contribution is values[group_idx]. No LKJ needed at n=1.
#
# `n_groups` is an ordinary kwarg, so the CALLER picks the size expression and
# any cv taint rides in with it -- see the cv-contagion note below
# `ranef_correlated_draws`. That is why there is no `_cv` sibling.
ranef_intercept = StanBlocks.@slic begin
    log_scale ~ std_normal()
    xi ~ std_normal(; n=n_groups)
    return exp(log_scale) * xi[group_idx]
end

# Draw-returning sibling used by multi-membership intercepts. The caller keeps
# `group_idx=` in the declaration metadata (for ranef_blocks) but performs the
# many-to-one weighted gather with `multi_membership_intercept` below.
ranef_intercept_draws = StanBlocks.@slic begin
    log_scale ~ std_normal()
    xi ~ std_normal(; n=n_groups)
    return exp(log_scale) * xi
end

# Correlated random effects for K terms x G groups. brms-style (1 + x + y | g).
# Non-centered parameterization:
#   L      ~ lkj_corr_cholesky(1, K)         # K x K Cholesky factor
#   tau    ~ half-std_normal(; n=K)          # per-term marginal scales
#   z_flat ~ std_normal(; n=K*n_groups)      # one standardised vector
#   z      = reshape(z_flat, K, n_groups)
#   b      = (diag_pre_multiply(tau, L) * z)'   # n_groups x K correlated draws
# Per-row contribution = Z[i, :] . b[group_idx[i], :], returned as a length-n
# vector via rows_dot_product. Note: `(1 | g) + (0 + x | g)` and `(1 + x | g)`
# are equivalent -- the walker merges everything sharing a group symbol into
# one correlated block.
#
# FLAT, not a plate, and `n_groups` is an ordinary kwarg -- the same shape
# `ranef_correlated_draws` below uses, for the same two reasons (see the
# cv-contagion note there): the CALLER picks the size expression so one submodel
# serves both the ordinary and the cv-contagious case, and the flat spelling
# samples `z` in one vectorised `std_normal()` call rather than `n_groups`
# per-cell calls in a loop.
#
# This path WAS a plate (StanBlocks devibe 9210b05). It is not any more. The
# plate could not carry cv sizing -- StanBlocks' plate branch returns before the
# `cv ? :quantities : :parameter` decision, so a plate-internal fresh parameter
# never consults cv at all and fails SILENTLY into a model that still samples
# per-group effects nothing informs -- which forced a duplicate
# `ranef_correlated_cv` whose body differed from this one in exactly one size
# expression. Collapsing to flat deletes that duplicate. It also costs nothing:
# StanBlocks now hoists loop-invariants out of plate bodies itself (snag
# `benchmarked-brm-20aa0361` item 1, landed `94a71a0`), which closed about half
# of the measured plate gap, and the residual is the per-call Cholesky logdet
# that a plate cannot avoid at all (item 2, still open).
#
# EMITTED-STAN CHANGE: the sampled parameter is now `<binding>_z_flat`
# (`vector[K*n_groups]`) where it was `<binding>_b_cols_z`
# (`matrix[K, n_groups]`). Models on this path recompile, and their unconstrained
# coordinate NAMES change from `<p>.<t>.<g>` to `<p>.<i>`, i = t + (g-1)*K.
# Consumers that resolve coordinates through `ranef_blocks` / `ranef_coordinates`
# (src/prediction.jl) follow automatically; one that hardcoded the parameter name
# does not. The column-major layout is unchanged, so the VALUES are the same.
ranef_correlated = StanBlocks.@slic begin
    L      ~ lkj_corr_cholesky(1.; n=n_terms)
    tau    ~ std_normal(; n=n_terms, lower=0.)
    z_flat ~ std_normal(; n=n_terms * n_groups)
    z = reshape(z_flat, n_terms, n_groups)
    b = (diag_pre_multiply(tau, L) * z)'   # n_groups x n_terms
    return rows_dot_product(Z, b[group_idx, :])
end

# Cross-formula correlated ranef draws for brms-style `(e | ID | g)` buckets.
# Same parameterization as `ranef_correlated` but returns the raw per-group
# matrix `b` (n_groups x n_terms) so multiple sub-formulas can each slice out
# their own column(s) and apply their own Z separately.
#
# FLAT, not a plate, and there is deliberately NO `_cv` sibling: `n_groups` is
# an ordinary kwarg, so the CALLER picks the size expression and the cv taint
# rides in with it. `_sb_emit_id_bucket_sampling!` passes the data scalar
# `n_<g>` by default and `maximum(<g>_idx)` for a group in `cv_groups`; in the
# latter case a `maybecv(:<g>_idx)` mark reaches the declared size and the whole
# block flips to a generated-quantities re-draw. One submodel serves both.
#
# A plate CANNOT serve the cv case, which is why this one is flat. StanBlocks'
# plate branch (`forward.jl`, `forward!(::SamplingExpr{Symbol,<:StanExpr})`)
# returns before the `cv ? :quantities : :parameter` decision, so a plate-
# internal fresh parameter never consults cv -- even when the plate's OUTER size
# is itself cv-tainted the promoted parameter stays in `parameters` while the
# likelihood is dropped, i.e. it fails SILENTLY into a model that still samples
# per-group effects nothing informs. Measured, not assumed. Flat is also the
# faster Stan here: one vectorised `z_flat ~ std_normal()` instead of n_groups
# per-cell calls in a loop (1.6x fewer us/gradient at n_groups=200, 3.3x at
# n_groups=1000; identical log-density and gradients to ~1e-14).
ranef_correlated_draws = StanBlocks.@slic begin
    L      ~ lkj_corr_cholesky(1.; n=n_terms)
    tau    ~ std_normal(; n=n_terms, lower=0.)
    z_flat ~ std_normal(; n=n_terms * n_groups)
    z = reshape(z_flat, n_terms, n_groups)
    return (diag_pre_multiply(tau, L) * z)'   # n_groups x n_terms
end

# Heterogeneous marginal-SD prior used only when an `sd(...)`
# statement targets a shared `|ID|` bucket. `family[i] == 0` retains BRM's
# historical half-standard-normal density for that margin; `family[i] == 1`
# selects an Exponential whose `rate[i]` is already in Stan's rate
# parameterization; `family[i] == 2` selects a zero-centered Normal whose
# positive `rate[i]` entry is its scale (a half-Normal after the LHS bound).
# A single vector density lets block defaults and margin-specific overrides
# compose without double-prioring any element.
#
# The family switch is written as NESTED `if`/`else` rather than an
# `if`/`elseif`/`else` chain on purpose. StanBlocks renders `elseif` as Stan
# `else if` only from `86fce35cd1fc4d0767f12ca39cf9bb64551e94b1` (2026-08-13)
# on; on an older checkout — Bruno's `deps/StanBlocks.jl` pin `05835dbe`, one
# day earlier — the chain overflowed the stack inside `stan_code`, so every
# model with an explicit ranef `sd(...)` prior was unbuildable there. The two
# spellings emit the same program up to block nesting (snag
# `ranef-sd-lpdf-el-a190739d`); keep this shape until every consumer pin has
# passed that StanBlocks commit.
StanBlocks.@deffun begin
    @lhs @lpxf brm_ranef_sd_lpdf(tau::vector[n], family::vector[n],
                                  rate::vector[n])::real = begin
        rv = 0.
        for i in 1:n
            if family[i] == 0
                rv += std_normal_lpdf(tau[i])::real
            else
                if family[i] == 1
                    rv += exponential_lpdf(tau[i], rate[i])::real
                else
                    rv += normal_lpdf(tau[i], 0., rate[i])::real
                end
            end
        end
        rv
    end
    # Predictive companions for `brm_ranef_sd`. A `regime="prior"`
    # (likelihood-free) program lowers EVERY parameter to a generated-quantities
    # `_rng` re-draw (StanBlocks fixed_param path; stanblocks-use §8/§34), so a
    # `(1 | p | subject)` block's `tau ~ brm_ranef_sd(...; lower=0.)` needs a
    # companion or tracing fails loudly naming the signature to add.
    #
    # The load-bearing form is the SCALAR per-element draw. Because the `~`
    # carries `lower=0.`, StanBlocks wraps the re-draw in its `truncated` HOF
    # (`lower_conditioning_rng`), which rejection-samples THIS scalar per element
    # and keeps only draws >= 0 (the decision that landed the StanBlocks fix:
    # "redraw_rng_expr wraps a plain family carrying user lower/upper kwargs into
    # the truncated HOF's variant call"). So each branch returns the family's
    # UNTRUNCATED base draw -- the counterpart of the density switch above -- and
    # lets that wrapper enforce the bound: family 0 a std Normal (-> half-std-
    # Normal after truncation, matching `std_normal_lpdf` on a `>=0` parameter),
    # family 1 Exponential(rate) (already >= 0), else Normal(0, rate) (-> half-
    # Normal). Returning `abs(...)` here would PRE-truncate and be wrong for an
    # unbounded use of the family, whose density is then the full two-sided
    # Normal. The sized-token form is the §8/§30 protocol for an unbounded sized
    # re-draw and delegates to the scalar. Nested `if`/`else`, not `elseif`, for
    # the same StanBlocks-pin reason as the density above (snag
    # `ranef-sd-lpdf-el-a190739d`).
    brm_ranef_sd_rng(family::real, rate::real)::real = begin
        if family == 0
            normal_rng(0., 1.)
        else
            if family == 1
                exponential_rng(rate)
            else
                normal_rng(0., rate)
            end
        end
    end
    brm_ranef_sd_rng(vector[n], family::vector[n], rate::vector[n])::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = brm_ranef_sd_rng(family[i], rate[i])
        end
        rv
    end
end


ranef_correlated_draws_effect = StanBlocks.@slic begin
    L      ~ lkj_corr_cholesky(lkj_eta; n=n_terms)
    tau    ~ brm_ranef_sd(sd_family, sd_rate; n=n_terms, lower=0.)
    z_flat ~ std_normal(; n=n_terms * n_groups)
    z = reshape(z_flat, n_terms, n_groups)
    return (diag_pre_multiply(tau, L) * z)'
end

# ---- R2D2: derived random-effect scales -----------------------------------
#
# The `effect(..., :) ~ r2d2(...)` family DERIVES the marginal scale
# `sqrt((1 - R2) * tau_bsv^2)` instead of sampling it, so these siblings take
# `tau` (resp. `scale`) as an ordinary caller-supplied kwarg. Everything else --
# the LKJ factor, the standardised draws, the column-major layout -- is
# identical to the sampled-scale families above, which is what keeps the
# all-or-nothing rule of decision `1db6zkr` cheap: a bucket either passes a
# fully derived `tau` vector here, or keeps sampling it over there. `L` stays
# free and shared in both: R2D2 constrains marginal variances and says nothing
# about cross-predictor correlation.
ranef_correlated_draws_r2d2 = StanBlocks.@slic begin
    L      ~ lkj_corr_cholesky(lkj_eta; n=n_terms)
    z_flat ~ std_normal(; n=n_terms * n_groups)
    z = reshape(z_flat, n_terms, n_groups)
    return (diag_pre_multiply(tau, L) * z)'
end

# Plain `(1 | g)` with a derived scale. Sibling of `ranef_intercept`, whose
# `log_scale ~ std_normal()` is exactly the sampled degree of freedom R2D2
# replaces.
ranef_intercept_r2d2 = StanBlocks.@slic begin
    xi ~ std_normal(; n=n_groups)
    return scale * xi[group_idx]
end

# Plain `(1 + x | g)` with a derived `tau`. Sibling of `ranef_correlated`.
ranef_correlated_r2d2 = StanBlocks.@slic begin
    L      ~ lkj_corr_cholesky(lkj_eta; n=n_terms)
    z_flat ~ std_normal(; n=n_terms * n_groups)
    z = reshape(z_flat, n_terms, n_groups)
    b = (diag_pre_multiply(tau, L) * z)'
    return rows_dot_product(Z, b[group_idx, :])
end

# Per-column empirical variance of the population design matrix. R2D2 weights
# each coefficient's share by `Var(x_k)` so the decomposition is a statement
# about explained VARIANCE rather than about raw coefficient magnitudes, which
# is what lets un-standardised and Bernoulli columns enter without the user
# pre-scaling anything (decision `kx8wkd`). Emitted from `X` rather than
# precomputed in Julia because the design columns are built as StanBlocks
# expressions -- categorical contrasts, `zscale`, splines -- not materialised
# vectors. When every column is data this lands in transformed data and is
# computed once.
StanBlocks.@deffun begin
    brm_col_variances(X::matrix[m, n], m::int, n::int)::vector[n] = begin
        rv = rep_vector(1., n)
        for k in 1:n
            rv[k] = variance(col(X, k))
        end
        rv
    end
end

# Assemble the population prior-scale vector for one R2D2-scoped predictor.
# `share_idx[j] == 0` means column `j` is NOT part of the decomposition -- the
# intercept always, plus any column carrying its own `effect(lp, coef) ~
# Normal(...)` override -- and keeps `fallback[j]`. Otherwise column `j` takes
# the Dirichlet share `phi[share_idx[j]]` of the explained variance:
#
#     scale[j] = sqrt(phi[share_idx[j]] * R2 * tau_bsv^2 / varx[j])
#
# One function rather than a broadcast expression so the decomposed and
# non-decomposed columns compose in a single pass with no index arithmetic at
# the call site.
StanBlocks.@deffun begin
    brm_r2d2_scale(share_idx::int[n], fallback::vector[n], varx::vector[n],
                   phi::vector[k], R2::real, tau_bsv::real,
                   n::int, k::int)::vector[n] = begin
        rv = rep_vector(0., n)
        for j in 1:n
            if share_idx[j] == 0
                rv[j] = fallback[j]
            else
                rv[j] = sqrt(phi[share_idx[j]] * R2 * tau_bsv^2 / varx[j])
            end
        end
        rv
    end
end

# Multi-membership gathers. Membership indices and weights are row-major flat
# vectors: entries `(i-1)*n_memberships + m` describe observation `i`, member
# slot `m`. Keeping indices flat avoids relying on a Stan array-of-int matrix
# spelling while retaining a single validated preprocessing record in Julia.
StanBlocks.@deffun begin
    multi_membership_intercept(b::vector[n_groups],
                               group_idx::int[n_mm],
                               weights::vector[n_mm],
                               n_obs::int,
                               n_memberships::int)::vector[n_obs] = begin
        rv = rep_vector(0., n_obs)
        for i in 1:n_obs
            for m in 1:n_memberships
                j = (i - 1) * n_memberships + m
                rv[i] += weights[j] * b[group_idx[j]]
            end
        end
        rv
    end
    multi_membership_correlated(Z::matrix[n_obs, n_terms],
                                b::matrix[n_groups, n_terms],
                                group_idx::int[n_mm],
                                weights::vector[n_mm],
                                n_obs::int,
                                n_memberships::int)::vector[n_obs] = begin
        rv = rep_vector(0., n_obs)
        for i in 1:n_obs
            for m in 1:n_memberships
                j = (i - 1) * n_memberships + m
                for k in 1:n_terms
                    rv[i] += weights[j] * Z[i, k] * b[group_idx[j], k]
                end
            end
        end
        rv
    end
end

# ---- cv-contagious sizing (opt-in; for out-of-sample / CV models) ------------
#
# There are NO `_cv` submodels. There used to be three; all three are gone. The
# behaviour is not: it is now a property of the CALL SITE, because every
# non-centered submodel above takes `n_groups` as an ordinary kwarg.
#
# StanBlocks' cv-contagion (see StanBlocks forward.jl:321 + types.jl:215):
# a parameter taints to a `:quantities` (generated-quantities re-draw) qual iff
# its TYPE is cv, and a type is cv iff its own flag OR *any element of its size*
# is cv. So a random effect flips to a GQ population re-draw exactly when its
# SIZE traces from a cv-marked input.
#
# The emitters therefore pass, for a group `g`:
#
#   n_groups = n_<g>              (a standalone data scalar -- no taint; the RE
#                                  stays a fitted parameter: the ordinary build)
#   n_groups = maximum(<g>_idx)   (for `g in cv_groups`; `maximum` propagates the
#                                  taint from `maybecv(:<g>_idx)` into the
#                                  declared size via passes.jl:23, flipping the
#                                  draw to a generated-quantities re-draw)
#
# `L`/`tau` keep their `n_terms` (untainted) sizing either way, so under marking
# ONLY the standardised draw is re-drawn -- a leave-all-out population re-draw
# from the fitted covariance, the semantics confirmed for QT's `source` knob.
# It stays OPT-IN via `SBBRMI(...; cv_groups=...)`; a build without it emits the
# `n_<g>` form.
#
# WHY THE DUPLICATES EXISTED, AND WHY THEY DO NOT NOW. A `_cv` sibling was only
# ever needed where the size expression could not be supplied by the caller --
# i.e. where the submodel computed it internally because it was a plate. A plate
# cannot carry cv sizing for a sharper reason than "the taint does not arrive":
# StanBlocks' `forward!(::SamplingExpr{Symbol,<:StanExpr})` (forward.jl)
# dispatches to the plate promotion and RETURNS before reaching the
# `cv ? :quantities : :parameter` decision, so a plate-internal fresh parameter
# never consults cv AT ALL. Making the plate's OUTER size cv-tainted does not
# help: the promoted parameter stays in `parameters` while the likelihood is
# dropped -- a model that still samples per-group effects nothing informs, with
# no error. Measured, not inferred. Once the non-centered paths stopped being
# plates, each duplicate collapsed into its sibling.
#
# STILL UNSUPPORTED: the typed-LHS `_by`/stratified path
# (`z::vector[n_groups, n_terms] ~ multi_std_normal()`). StanBlocks' typed-LHS
# forward (forward.jl:355) derives cv from the RHS call-args, not the declared
# size, so a cv size there yields a cv *parameter*, not a `:quantities` re-draw.
# `gr(g, by=b)` in `cv_groups` is rejected loudly rather than emitted wrong.
# Centered emission is likewise not combinable with cv sizing -- see below.

# ---- centered ranef variants (opt-in; for strong per-group likelihoods) ------
#
# The default (and `_cv`) submodels above are NON-CENTERED: the sampled
# parameter is a standardised `z ~ std_normal()` and the scale is applied
# downstream (`diag_pre_multiply(tau, L) * z`). That geometry is the right
# default for weak per-group likelihoods (few observations per level), which is
# brms' default and the shape every existing BRM model was fitted under.
#
# It is the WRONG choice when the per-group likelihood is strong -- dense
# repeated measurement per level, e.g. a PK model with many samples per subject.
# There the non-centered funnel inverts and the centered parameterization, in
# which the per-group effect IS the sampled parameter with the covariance as its
# prior, samples better. These variants emit that form:
#
#   b_cols ~ plate(; outer=(n_groups,)) do g
#       bc::vector[n_terms] ~ multi_normal_cholesky(0, diag_pre_multiply(tau, L))
#       bc
#   end
#
# They are OPT-IN per grouping factor via `SBBRMI(...; centered_groups=[:g])`:
# a group not named here is emitted by the non-centered path above, untouched by
# anything in this section. The returned VALUE is identical in distribution to
# the non-centered sibling's -- only the parameterization (and hence the
# unconstrained coordinate system) differs, so fitted draws are NOT
# interchangeable between the two.
#
# Shape: the per-group effect is declared typed-LHS as `array[n_groups]
# vector[n_terms]` and sampled with ONE vectorised `multi_normal_cholesky` call,
# rather than as a plate over a per-cell `multi_normal_cholesky`. Both transpile
# and both are stanc-clean. The reason to prefer the typed-LHS spelling is the
# Cholesky LOG-DETERMINANT: Stan evaluates `log(diagonal(L))` once per
# `multi_normal_cholesky` CALL, so `n_groups` per-cell calls pay it `n_groups`
# times where one array-vectorised call pays it once. A plate cannot express a
# vectorised MVN, so this is a floor the plate spelling cannot reach.
#
# Measured (BridgeStan us/gradient, n_terms=5, identical posterior -- gradients
# agree to 2.6e-13, log-densities differ by exactly the dropped normalisation
# constant n_groups*K/2*log(2pi)):
#
#                                     n_groups=200   n_groups=1000
#   plate, cell body written inline    198.3          859.4
#   plate, invariants hoisted          138.7          537.4
#   typed-LHS (this file)               85.0          348.7
#
# Note what the middle row says: MOST of the naive plate's disadvantage was not
# the plate concept, it was that `diag_pre_multiply(tau, L)` and `rep_vector(0.,
# n_terms)` were emitted INSIDE the per-cell loop where they are loop-invariant.
# That is FIXED UPSTREAM as of StanBlocks `94a71a0` (snag
# `benchmarked-brm-20aa0361` item 1): the emitter now hoists loop-invariants out
# of plate bodies itself, so the top row collapses onto the middle one and the
# rows above are kept only as the record of how the gap was decomposed.
# StanBlocks re-measured it at n_groups=1000 as 728.8 -> 476.9 us/grad centered,
# 626.3 -> 308.4 non-centered, both matching a hand-hoisted control exactly.
#
# The residual 1.5-1.6x is the logdet, and it is the whole justification for
# spelling this one differently from the rest of the file. It is a real cost in
# idiom uniformity, accepted deliberately, not an oversight. Closing it needs
# StanBlocks to collapse a plate whose cell body is a single `~` over a
# multivariate distribution with cell-invariant parameters into one
# array-vectorised call -- item 2 of that same snag, still open and not yet
# filed as its own row. If it lands, this submodel can go back to a plate and
# this whole comment can go with it.
#
# Scope: centered emission is NOT combinable with cv-contagious sizing. A
# centered block's sampled parameter is the per-group effect itself, which under
# either available spelling (plate cell, or the typed-LHS form below) sits behind
# the same StanBlocks gap the `_cv` comment above describes -- so there is no
# centered shape whose size can carry a cv taint. `SBBRMI` rejects the
# combination explicitly rather than silently emitting an in-sample block.

# `@stanonly` because `vector[m, n]` (an `array[m] vector[n]`) has no Julia
# emission; both helpers exist only to be called from the SLIC bodies below,
# so they are never invoked from Julia. `multi_normal_cholesky0` is the zero-mean
# array-vectorised MVN-Cholesky the typed-LHS `~` routes to (a distinctly-named
# `@lhs @lpxf` UDF, exactly as `multi_std_normal` is for `ranef_correlated_by`);
# `ranef_b_matrix` rebuilds the `n_groups × n_terms` matrix the public contract
# promises. The loop is why it is a `@deffun` -- Stan's `to_matrix` has no
# `array[] vector` overload, and `@slic` bodies cannot contain control flow.
StanBlocks.@deffun begin
    @lhs @lpxf multi_normal_cholesky0_lpdf(x::vector[m, n], scale::matrix[n, n])::real = begin
        multi_normal_cholesky_lpdf(x, rep_vector(0., n), scale)
    end
    @stanonly ranef_b_matrix(b::vector[m, n])::matrix[m, n] = begin
        rv::matrix[m, n]
        for i in 1:m
            rv[i, :] = b[i]'
        end
        rv
    end
end

# Multinomial composition likelihood — a per-row K-category count response
# `obs::int[nrow, K]`. Two dispatch methods on `probs`: SHARED (one `vector[K]`
# simplex across all rows) delegates to StanBlocks' multi-row builtin; PER-ROW
# (`matrix[nrow, K]`, row i its own simplex — e.g. a scan carrier like the seal's
# per-year age composition) loops over rows. `Multinomial(N, probs)` routes here
# (`_sb_lik_family!(::Type{<:Multinomial})`); StanBlocks dispatches by the Stan type
# of `probs`. NOTE the `_rng` companions take a BARE type-TOKEN leading arg
# (`int[nrow, K]`, no name) as every sized `_rng` builtin does (multinomial_rng,
# builtin.jl) — a NAMED leading arg makes the auto-GQ `_gen` twin fail to type.
StanBlocks.@deffun begin
    @lhs @lpxf brm_multinomial_lpmf(obs::int[nrow, K], probs::matrix[nrow, K], N::int[nrow])::real = begin
        ll = 0.0
        for i in 1:nrow
            ll = ll + multinomial_lpmf(obs[i], to_vector(probs[i, :]), N[i])
        end
        ll
    end
    @lhs brm_multinomial_lpmf(obs::int[nrow, K], probs::vector[K], N::int[nrow])::real =
        multinomial_lpmf(obs, probs, N)
    # pointwise (per-row) log-likelihood twin for the `_likelihood` generated quantity.
    brm_multinomial_lpmfs(obs::int[nrow, K], probs::matrix[nrow, K], N::int[nrow])::vector[nrow] = begin
        lls::vector[nrow]
        for i in 1:nrow
            lls[i] = multinomial_lpmf(obs[i], to_vector(probs[i, :]), N[i])
        end
        lls
    end
    brm_multinomial_lpmfs(obs::int[nrow, K], probs::vector[K], N::int[nrow])::vector[nrow] = begin
        lls::vector[nrow]
        for i in 1:nrow
            lls[i] = multinomial_lpmf(obs[i], probs, N[i])
        end
        lls
    end
    # posterior-predictive (per-row) draw twin for the `_gen` generated quantity —
    # BARE type-token leading arg.
    brm_multinomial_rng(int[nrow, K], probs::matrix[nrow, K], N::int[nrow])::int[nrow, K] = begin
        rv::int[nrow, K]
        for i in 1:nrow
            rv[i, :] = multinomial_rng(to_vector(probs[i, :]), N[i])
        end
        rv
    end
    brm_multinomial_rng(int[nrow, K], probs::vector[K], N::int[nrow])::int[nrow, K] = begin
        rv::int[nrow, K]
        for i in 1:nrow
            rv[i, :] = multinomial_rng(probs, N[i])
        end
        rv
    end
end

ranef_intercept_centered = StanBlocks.@slic begin
    log_scale ~ std_normal()
    xi ~ normal(0., exp(log_scale); n=n_groups)
    return xi[group_idx]
end

ranef_correlated_centered = StanBlocks.@slic begin
    L   ~ lkj_corr_cholesky(1.; n=n_terms)
    tau ~ std_normal(; n=n_terms, lower=0.)
    b::vector[n_groups, n_terms] ~ multi_normal_cholesky0(diag_pre_multiply(tau, L))
    bm = ranef_b_matrix(b)   # n_groups x n_terms
    return rows_dot_product(Z, bm[group_idx, :])
end

# Centered sibling of `ranef_correlated_draws` for `(e | ID | g)` buckets.
ranef_correlated_draws_centered = StanBlocks.@slic begin
    L   ~ lkj_corr_cholesky(1.; n=n_terms)
    tau ~ std_normal(; n=n_terms, lower=0.)
    b::vector[n_groups, n_terms] ~ multi_normal_cholesky0(diag_pre_multiply(tau, L))
    return ranef_b_matrix(b)   # n_groups x n_terms
end

# Configured centered blocks use a plate of native MVN-Cholesky draws. With the
# heterogeneous custom `tau` density in the same submodel, the array-vector
# `multi_normal_cholesky0` tracetype cannot recover its free outer dimension;
# the plate carries that dimension explicitly and emits the model-scale effect
# parameter as `<binding>_b_cols_bc`. Keep prediction.jl's family table aligned.
ranef_correlated_draws_centered_effect = StanBlocks.@slic begin
    L   ~ lkj_corr_cholesky(lkj_eta; n=n_terms)
    tau ~ brm_ranef_sd(sd_family, sd_rate; n=n_terms, lower=0.)
    b_cols ~ plate(; outer=(n_groups,)) do g
        bc::vector[n_terms] ~ multi_normal_cholesky(
            rep_vector(0., n_terms), diag_pre_multiply(tau, L))
        bc
    end
    return b_cols'
end

# Stratified gather + Cholesky scale, kept as a Stan function (loops are not
# allowed in @slic bodies, but they are allowed in @deffun bodies). For each
# group g, pick the stratum s = stratum_idx[g] and compute
#   b[g, :] = (diag_pre_multiply(tau[s, :], L[s, :, :]) * z[g, :])'.

# ---- group-block toy demo term -----------------------------------------------
#
# sb_group_demo: demonstrates the inverted-control group-block mechanism.
# Allocates 2 correlated per-group params via ranef_correlated_draws (proper
# 2x2 LKJ block), receives them as `group_block` (n_groups x 2 matrix), and
# returns the per-obs sum of both per-group params.
# No docstring (docstring → @deffun AssertionError gotcha; see primer).
function sb_group_demo end

sb_group_demo_slic = StanBlocks.@slic begin
    return group_block[group_idx, 1] + group_block[group_idx, 2]
end

# sb_group_clamped_demo: proves the general structured-latent floor's
# clamped / non-normal element-wise prior path (decision 10uz10q, the obs_scale
# shape). Declares a `matrix<lower=0>[n_groups, 2]` latent with an element-wise
# Exponential prior — exactly varyingsource2's `matrix<lower=0>[n_assays,2]`
# obs_scale. Not a wired consumer; it exists so a transpile probe can confirm
# the floor emits a valid positively-constrained matrix param with a non-normal
# prior. Returns the per-obs sum of both clamped per-group params.
# No docstring (docstring → @deffun AssertionError gotcha; see primer).
function sb_group_clamped_demo end

sb_group_clamped_demo_slic = StanBlocks.@slic begin
    return group_block[group_idx, 1] + group_block[group_idx, 2]
end

# Zero-inflated Poisson lpmf (per-element + vectorised). Per element:
#   y == 0 -> log_sum_exp(log(zi),  log1m(zi) + poisson_lpmf(0  | lambda))
#   y >  0 ->                       log1m(zi) + poisson_lpmf(y  | lambda)
# Defined here (not in StanBlocks/builtin.jl) so the BRM frontend can ship
# zero-inflated Poisson without an upstream addition; @deffun bodies are
# allowed control flow, which the per-element conditional needs. The
# `@lpxf` annotation on the first definition wires the SLIC sampling
# dispatch (lpxf_expr / rng_expr / likelihood_expr hooks) so
# `y ~ zero_inflated_poisson(lambda, zi)` resolves to the lpmf.
StanBlocks.@deffun begin
    @lpxf zero_inflated_poisson_lpmf(y::int, lambda::real, zi::real)::real = begin
        if y == 0
            log_sum_exp(log(zi), log1m(zi) + poisson_lpmf(0::int, lambda))
        else
            log1m(zi) + poisson_lpmf(y, lambda)
        end
    end
    zero_inflated_poisson_lpmf(y::int[n], lambda::vector[n], zi::vector[n])::real = begin
        rv = 0.
        for i in 1:n
            rv += zero_inflated_poisson_lpmf(y[i], lambda[i], zi[i])::real
        end
        rv
    end
    zero_inflated_poisson_lpmf(y::int[n], lambda::vector[n], zi::real)::real = begin
        zero_inflated_poisson_lpmf(y, lambda, rep_vector(zi, n))
    end
    zero_inflated_poisson_lpmf(y::int[n], lambda::real, zi::vector[n])::real = begin
        zero_inflated_poisson_lpmf(y, rep_vector(lambda, n), zi)
    end
    zero_inflated_poisson_lpmf(y::int[n], lambda::real, zi::real)::real = begin
        zero_inflated_poisson_lpmf(y, rep_vector(lambda, n), rep_vector(zi, n))
    end
    # Hand-rolled per-element loop returning vector[n]. Mirrors the
    # `ordered_logistic_lpmfs` pattern in StanBlocks/builtin.jl --
    # avoids `jbroadcasted` which lives in `StanBlocks.builtin` and may
    # not be reachable from a user-side @deffun's symbol resolver.
    zero_inflated_poisson_lpmfs(args...) = begin
        zero_inflated_poisson_lpmf(args...)
    end
    zero_inflated_poisson_lpmfs(y::int[n], lambda::vector[n], zi::vector[n]) = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = zero_inflated_poisson_lpmf(y[i], lambda[i], zi[i])
        end
        rv
    end
    zero_inflated_poisson_lpmfs(y::int[n], lambda::vector[n], zi::real)::vector[n] = begin
        zero_inflated_poisson_lpmfs(y, lambda, rep_vector(zi, n))
    end
    zero_inflated_poisson_lpmfs(y::int[n], lambda::real, zi::vector[n])::vector[n] = begin
        zero_inflated_poisson_lpmfs(y, rep_vector(lambda, n), zi)
    end
    zero_inflated_poisson_lpmfs(y::int[n], lambda::real, zi::real)::vector[n] = begin
        zero_inflated_poisson_lpmfs(y, rep_vector(lambda, n), rep_vector(zi, n))
    end
    # Synthetic-data RNG used by SLIC's generated_quantities. Mirrors the
    # `binomial_logit_rng(int[n], …)` token-path pattern from
    # StanBlocks/builtin.jl: takes a sized int[n] token + the params,
    # writes per-element draws into the output vector.
    zero_inflated_poisson_rng(int[n], lambda::vector[n], zi::vector[n])::int[n] = begin
        rv::int[n]
        for i in 1:n
            if bernoulli_rng(zi[i]) == 1
                rv[i] = 0
            else
                rv[i] = poisson_rng(lambda[i])
            end
        end
        rv
    end
    zero_inflated_poisson_rng(int[n], lambda::vector[n], zi::real)::int[n] = begin
        rv::int[n]
        for i in 1:n
            if bernoulli_rng(zi) == 1
                rv[i] = 0
            else
                rv[i] = poisson_rng(lambda[i])
            end
        end
        rv
    end
    zero_inflated_poisson_rng(int[n], lambda::real, zi::vector[n])::int[n] = begin
        rv::int[n]
        for i in 1:n
            if bernoulli_rng(zi[i]) == 1
                rv[i] = 0
            else
                rv[i] = poisson_rng(lambda)
            end
        end
        rv
    end
    zero_inflated_poisson_rng(int[n], lambda::real, zi::real)::int[n] = begin
        rv::int[n]
        for i in 1:n
            if bernoulli_rng(zi) == 1
                rv[i] = 0
            else
                rv[i] = poisson_rng(lambda)
            end
        end
        rv
    end
end

# One internal Stan family serves the public typed ordinal composition. The
# integer tags are emitted compile-time literals (structure: cumulative=1,
# stopping-ratio=2; link: logit=1, probit=2, cloglog=3). Keeping the tags inside
# one private UDF avoids a Cartesian product of public pair names while still
# giving StanBlocks the complete lpmf / pointwise-lpmf / RNG triad.
#
# For cumulative-logit the scalar density delegates to Stan's native
# ordered_logistic_lpmf after applying discrimination to both eta and the
# cutpoints. Probit and cloglog use their exact native CDF/log-CDF primitives.
# Stopping-ratio has no native Stan distribution, so its sequential hazards are
# accumulated directly in log space.
StanBlocks.@deffun begin
    @stanonly brm_ordinal_logcdf(z::real, link::int)::real = begin
        if link == 1
            log_inv_logit(z)
        else
            if link == 2
                normal_lcdf(z, 0., 1.)
            else
                log1m_exp(-exp(z))
            end
        end
    end
    @stanonly brm_ordinal_logccdf(z::real, link::int)::real = begin
        if link == 1
            log_inv_logit(-z)
        else
            if link == 2
                normal_lccdf(z, 0., 1.)
            else
                -exp(z)
            end
        end
    end
    @stanonly brm_ordinal_cdf(z::real, link::int)::real = begin
        if link == 1
            inv_logit(z)
        else
            if link == 2
                Phi(z)
            else
                -expm1(-exp(z))
            end
        end
    end

    @stanonly @lpxf brm_ordinal_lpmf(y::int, eta::real, thresholds::vector[k],
                           discrimination::real, structure::int, link::int,
                           threshold_effect::vector[k])::real = begin
        K = k + 1
        if discrimination <= 0.
            negative_infinity()
        else
            if y < 1
                negative_infinity()
            else
                if y > K
                    negative_infinity()
                else
                    if structure == 1
                        if link == 1
                            ordered_logistic_lpmf(
                                y,
                                discrimination * eta,
                                discrimination .* thresholds,
                            )
                        else
                            if y == 1
                                z_first = discrimination * (thresholds[1] - eta)
                                brm_ordinal_logcdf(z_first, link)
                            else
                                if y == K
                                    z_last = discrimination * (thresholds[k] - eta)
                                    brm_ordinal_logccdf(z_last, link)
                                else
                                    z_hi = discrimination * (thresholds[y] - eta)
                                    z_lo = discrimination * (thresholds[y - 1] - eta)
                                    log_diff_exp(
                                        brm_ordinal_logcdf(z_hi, link),
                                        brm_ordinal_logcdf(z_lo, link),
                                    )
                                end
                            end
                        end
                    else
                        rv = 0.
                        for j in 1:k
                            z_stage = discrimination *
                                (thresholds[j] - eta - threshold_effect[j])
                            if j < y
                                rv += brm_ordinal_logccdf(z_stage, link)
                            else
                                if j == y
                                    rv += brm_ordinal_logcdf(z_stage, link)
                                end
                            end
                        end
                        rv
                    end
                end
            end
        end
    end

    @stanonly brm_ordinal_lpmf(y::int[n], eta::vector[n], thresholds::vector[k],
                     discrimination::vector[n], structure::int, link::int,
                     threshold_effect::matrix[n,k])::real = begin
        rv = 0.
        for i in 1:n
            rv += brm_ordinal_lpmf(
                y[i], eta[i], thresholds, discrimination[i], structure, link,
                to_vector(threshold_effect[i, :]),
            )
        end
        rv
    end
    @stanonly brm_ordinal_lpmf(y::int[n], eta::vector[n], thresholds::vector[k],
                     discrimination::real, structure::int, link::int,
                     threshold_effect::matrix[n,k])::real = begin
        brm_ordinal_lpmf(
            y, eta, thresholds, rep_vector(discrimination, n), structure, link,
            threshold_effect,
        )
    end
    @stanonly brm_ordinal_lpmf(y::int[n], eta::real, thresholds::vector[k],
                     discrimination::vector[n], structure::int, link::int,
                     threshold_effect::matrix[n,k])::real = begin
        brm_ordinal_lpmf(
            y, rep_vector(eta, n), thresholds, discrimination, structure, link,
            threshold_effect,
        )
    end
    @stanonly brm_ordinal_lpmf(y::int[n], eta::real, thresholds::vector[k],
                     discrimination::real, structure::int, link::int,
                     threshold_effect::matrix[n,k])::real = begin
        brm_ordinal_lpmf(
            y, rep_vector(eta, n), thresholds, rep_vector(discrimination, n),
            structure, link, threshold_effect,
        )
    end

    @stanonly brm_ordinal_lpmfs(args...) = begin
        brm_ordinal_lpmf(args...)
    end
    @stanonly brm_ordinal_lpmfs(y::int[n], eta::vector[n], thresholds::vector[k],
                      discrimination::vector[n], structure::int, link::int,
                      threshold_effect::matrix[n,k])::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = brm_ordinal_lpmf(
                y[i], eta[i], thresholds, discrimination[i], structure, link,
                to_vector(threshold_effect[i, :]),
            )
        end
        rv
    end
    @stanonly brm_ordinal_lpmfs(y::int[n], eta::vector[n], thresholds::vector[k],
                      discrimination::real, structure::int, link::int,
                      threshold_effect::matrix[n,k])::vector[n] = begin
        brm_ordinal_lpmfs(
            y, eta, thresholds, rep_vector(discrimination, n), structure, link,
            threshold_effect,
        )
    end
    @stanonly brm_ordinal_lpmfs(y::int[n], eta::real, thresholds::vector[k],
                      discrimination::vector[n], structure::int, link::int,
                      threshold_effect::matrix[n,k])::vector[n] = begin
        brm_ordinal_lpmfs(
            y, rep_vector(eta, n), thresholds, discrimination, structure, link,
            threshold_effect,
        )
    end
    @stanonly brm_ordinal_lpmfs(y::int[n], eta::real, thresholds::vector[k],
                      discrimination::real, structure::int, link::int,
                      threshold_effect::matrix[n,k])::vector[n] = begin
        brm_ordinal_lpmfs(
            y, rep_vector(eta, n), thresholds, rep_vector(discrimination, n),
            structure, link, threshold_effect,
        )
    end

    @stanonly brm_ordinal_rng(eta::real, thresholds::vector[k], discrimination::real,
                    structure::int, link::int,
                    threshold_effect::vector[k])::int = begin
        K = k + 1
        rv = K
        if structure == 1
            u = uniform_rng(0., 1.)
            for j in 1:k
                if rv == K
                    z_cumulative = discrimination * (thresholds[j] - eta)
                    if u <= brm_ordinal_cdf(z_cumulative, link)
                        rv += j - rv
                    end
                end
            end
        else
            for j in 1:k
                if rv == K
                    z_stopping = discrimination *
                        (thresholds[j] - eta - threshold_effect[j])
                    if bernoulli_rng(brm_ordinal_cdf(z_stopping, link)) == 1
                        rv += j - rv
                    end
                end
            end
        end
        rv
    end
    @stanonly brm_ordinal_rng(int[n], eta::vector[n], thresholds::vector[k],
                    discrimination::vector[n], structure::int, link::int,
                    threshold_effect::matrix[n,k])::int[n] = begin
        rv::int[n]
        for i in 1:n
            rv[i] = brm_ordinal_rng(
                eta[i], thresholds, discrimination[i], structure, link,
                to_vector(threshold_effect[i, :]),
            )
        end
        rv
    end
    @stanonly brm_ordinal_rng(int[n], eta::vector[n], thresholds::vector[k],
                    discrimination::real, structure::int, link::int,
                    threshold_effect::matrix[n,k])::int[n] = begin
        brm_ordinal_rng(
            int[n], eta, thresholds, rep_vector(discrimination, n), structure,
            link, threshold_effect,
        )
    end
    @stanonly brm_ordinal_rng(int[n], eta::real, thresholds::vector[k],
                    discrimination::vector[n], structure::int, link::int,
                    threshold_effect::matrix[n,k])::int[n] = begin
        brm_ordinal_rng(
            int[n], rep_vector(eta, n), thresholds, discrimination, structure,
            link, threshold_effect,
        )
    end
    @stanonly brm_ordinal_rng(int[n], eta::real, thresholds::vector[k],
                    discrimination::real, structure::int, link::int,
                    threshold_effect::matrix[n,k])::int[n] = begin
        brm_ordinal_rng(
            int[n], rep_vector(eta, n), thresholds,
            rep_vector(discrimination, n), structure, link, threshold_effect,
        )
    end
end

# Native-Stan von-Mises density with the two public BRM support contracts made
# explicit. `principal == 0` is Distributions.jl's `VonMises`: moving inclusive
# support `[mu - pi, mu + pi]`. `principal == 1` is `CircularVonMises`: fixed
# half-open support `[lo, hi)`, with `mu` and generated draws wrapped into it.
# The scalar density always delegates its in-support value to Stan's native
# `von_mises_lpdf`; the wrappers only add support/domain semantics.
StanBlocks.@deffun begin
    @lpxf brm_von_mises_lpdf(y::real, mu::real, kappa::real,
                             lo::real, hi::real, principal::int)::real = begin
        if kappa <= 0.
            negative_infinity()
        else
            if principal == 1
                if y < lo
                    negative_infinity()
                else
                    if y >= hi
                        negative_infinity()
                    else
                        wrapped_mu = lo + fmod(fmod(mu - lo, hi - lo) + hi - lo, hi - lo)
                        von_mises_lpdf(y, wrapped_mu, kappa)
                    end
                end
            else
                if y < mu - 3.141592653589793
                    negative_infinity()
                else
                    if y > mu + 3.141592653589793
                        negative_infinity()
                    else
                        von_mises_lpdf(y, mu, kappa)
                    end
                end
            end
        end
    end
    brm_von_mises_lpdf(y::vector[n], mu::vector[n], kappa::vector[n],
                       lo::real, hi::real, principal::int)::real = begin
        rv = 0.
        for i in 1:n
            rv += brm_von_mises_lpdf(y[i], mu[i], kappa[i], lo, hi, principal)::real
        end
        rv
    end
    brm_von_mises_lpdf(y::vector[n], mu::vector[n], kappa::real,
                       lo::real, hi::real, principal::int)::real = begin
        brm_von_mises_lpdf(y, mu, rep_vector(kappa, n), lo, hi, principal)
    end
    brm_von_mises_lpdf(y::vector[n], mu::real, kappa::vector[n],
                       lo::real, hi::real, principal::int)::real = begin
        brm_von_mises_lpdf(y, rep_vector(mu, n), kappa, lo, hi, principal)
    end
    brm_von_mises_lpdf(y::vector[n], mu::real, kappa::real,
                       lo::real, hi::real, principal::int)::real = begin
        brm_von_mises_lpdf(y, rep_vector(mu, n), rep_vector(kappa, n), lo, hi, principal)
    end

    brm_von_mises_lpdfs(args...) = begin
        brm_von_mises_lpdf(args...)
    end
    brm_von_mises_lpdfs(y::vector[n], mu::vector[n], kappa::vector[n],
                        lo::real, hi::real, principal::int)::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = brm_von_mises_lpdf(y[i], mu[i], kappa[i], lo, hi, principal)
        end
        rv
    end
    brm_von_mises_lpdfs(y::vector[n], mu::vector[n], kappa::real,
                        lo::real, hi::real, principal::int)::vector[n] = begin
        brm_von_mises_lpdfs(y, mu, rep_vector(kappa, n), lo, hi, principal)
    end
    brm_von_mises_lpdfs(y::vector[n], mu::real, kappa::vector[n],
                        lo::real, hi::real, principal::int)::vector[n] = begin
        brm_von_mises_lpdfs(y, rep_vector(mu, n), kappa, lo, hi, principal)
    end
    brm_von_mises_lpdfs(y::vector[n], mu::real, kappa::real,
                        lo::real, hi::real, principal::int)::vector[n] = begin
        brm_von_mises_lpdfs(y, rep_vector(mu, n), rep_vector(kappa, n), lo, hi, principal)
    end

    brm_von_mises_rng(mu::real, kappa::real,
                      lo::real, hi::real, principal::int)::real = begin
        if kappa <= 0.
            reject("brm_von_mises_rng: kappa must be strictly positive")
            0.
        else
            draw = von_mises_rng(mu, kappa)
            if principal == 1
                lo + fmod(fmod(draw - lo, hi - lo) + hi - lo, hi - lo)
            else
                support_lo = mu - 3.141592653589793
                support_lo + fmod(fmod(draw - support_lo, 6.283185307179586) +
                                  6.283185307179586, 6.283185307179586)
            end
        end
    end
    brm_von_mises_rng(vector[n], mu::vector[n], kappa::vector[n],
                      lo::real, hi::real, principal::int)::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = brm_von_mises_rng(mu[i], kappa[i], lo, hi, principal)
        end
        rv
    end
    brm_von_mises_rng(vector[n], mu::vector[n], kappa::real,
                      lo::real, hi::real, principal::int)::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = brm_von_mises_rng(mu[i], kappa, lo, hi, principal)
        end
        rv
    end
    brm_von_mises_rng(vector[n], mu::real, kappa::vector[n],
                      lo::real, hi::real, principal::int)::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = brm_von_mises_rng(mu, kappa[i], lo, hi, principal)
        end
        rv
    end
    brm_von_mises_rng(vector[n], mu::real, kappa::real,
                      lo::real, hi::real, principal::int)::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = brm_von_mises_rng(mu, kappa, lo, hi, principal)
        end
        rv
    end
end

function addprop end

StanBlocks.@deffun begin
    addprop(loc::vector[n], add::real, prop::real)::vector[n] = begin
        sqrt(add^2 .+ (loc .* prop).^2)
    end
    @inline addprop(loc::RaggedVector, add::real, prop::real) = begin
        RaggedVector(addprop(loc.mem, add, prop), loc.ends)
    end
end

StanBlocks.@deffun begin
    stratified_correlated_b(L, tau, z, stratum_idx::int[n_groups],
                            n_groups::int, n_terms::int) = begin
        b = rep_matrix(0., n_groups, n_terms)
        for g in 1:n_groups
            b[g, :] = (diag_pre_multiply(tau[stratum_idx[g], :],
                                         L[stratum_idx[g], :, :]) * z[g, :])'
        end
        b
    end
end

# `(expr | gr(g, by=b))` stratified random effects: independent LKJ-Cholesky +
# tau per level of `b`, so each stratum has its own full covariance structure.
# `stratum_idx[g]` maps each group-level to its stratum (walker pre-computes it
# and errors if any group straddles strata).
#   L   :: array[n_strata] cholesky_factor_corr[n_terms]
#   tau :: array[n_strata] vector<lower=0>[n_terms]
#   z   :: array[n_groups] vector[n_terms]
# Per-group contribution: b[g, :] = (diag_pre_multiply(tau[s], L[s]) * z[g])'
# where s = stratum_idx[g]. The per-group loop lives in
# `stratified_correlated_b` (a @deffun helper) because @slic bodies cannot
# contain control flow.
# Distinctly-named 2-arg `@lhs @lpxf` UDFs so typed-LHS sampling routes to a
# user-defined Stan function without clashing with Stan's scalar-only 
# built-ins. `@lpxf` creates the base stub + `lpxf_expr` hook so
# `L ~ multi_lkj_corr_cholesky(1.)` resolves; `@lhs` registers the base
# tracetype so the 2-arg call dispatches to this lpdf with `m, n` bound from
# the declared LHS shape.
StanBlocks.@deffun begin
    @lhs @lpxf multi_lkj_corr_cholesky_lpdf(L::cholesky_factor_corr[m, n], x::real)::real = begin
        rv = 0.
        for i in 1:m
            rv += lkj_corr_cholesky_lpdf(L[i, :, :], x)::real
        end
        rv
    end
    @lhs @lpxf multi_std_normal_lpdf(x::vector[m, n])::real = begin
        rv = 0.
        for i in 1:m
            rv += std_normal_lpdf(x[i, :])::real
        end
        rv
    end
end

ranef_correlated_by = StanBlocks.@slic begin
    L::cholesky_factor_corr[n_strata, n_terms] ~ multi_lkj_corr_cholesky(1.)
    tau::vector[n_strata, n_terms] ~ multi_std_normal(; lower=0.)
    # z is the standardised per-GROUP draw (n_groups rows), not per-stratum --
    # `stratified_correlated_b` indexes z[g, :] for g in 1:n_groups.
    z::vector[n_groups, n_terms] ~ multi_std_normal()
    b = stratified_correlated_b(L, tau, z, stratum_idx, n_groups, n_terms)
    return rows_dot_product(Z, b[group_idx, :])
end

# Cross-formula stratified correlated ranef draws for brms-style
# `(e | ID | gr(g, by=b))` buckets. Matrix-returning variant of
# `ranef_correlated_by` so each sub-formula can slice its own column(s).
ranef_correlated_by_draws = StanBlocks.@slic begin
    L::cholesky_factor_corr[n_strata, n_terms] ~ multi_lkj_corr_cholesky(1.)
    tau::vector[n_strata, n_terms] ~ multi_std_normal(; lower=0.)
    z::vector[n_groups, n_terms] ~ multi_std_normal()
    return stratified_correlated_b(L, tau, z, stratum_idx, n_groups, n_terms)
end

# Treatment-coded categorical predictor. Allocates K-1 free betas; reference
# level 1 contributes 0. Mirrors vimpl's `AbstractVector{<:Integer}` dispatch.
# `x` is the per-row 1-based level index, `n_levels = K`.
_sb_cat = StanBlocks.@slic begin
    beta ~ std_normal(; n=n_levels - 1)
    return append_row(0., beta)[x]
end

# Treatment-coded categorical predictor with a caller-supplied Normal prior on
# the K-1 contrasts. Kept as a SIBLING of `_sb_cat` (exactly as `_popefs_normal`
# is of `popefs`) so configured and unconfigured blocks share one output shape.
# One shared `(location, scale)` covers every contrast; the reference level
# still contributes 0. The sampled parameter keeps the same `beta` name, so the
# Stan parameter is `cat_<lp>_<c>_beta` for both sibling submodels.
_sb_cat_normal = StanBlocks.@slic begin
    beta ~ normal(beta_loc, beta_scale; n=n_levels - 1)
    return append_row(0., beta)[x]
end

# Minimal `ar(time, p=1)` autoregressive submodel. Adds an AR(1) noise process
# `u[t] = phi * u[t-1] + epsilon[t]` (with `u[1] = epsilon[1]`; no stationary
# init) to the linear predictor. `phi` is parameterized via `tanh(phi_raw)` so
# it stays in (-1, 1) under a `std_normal` prior on `phi_raw`. Rows are
# assumed to already be in time order; the `time` arg is used only as a
# length probe (explicit sort by time is a follow-up). Only `p=1` is
# supported for the first pass.
StanBlocks.@deffun begin
    ar1_recurse(phi::real, epsilon::vector[n], n::int)::vector[n] = begin
        u = rep_vector(0., n)
        u[1] = epsilon[1]
        for t in 2:n
            u[t] = phi * u[t-1] + epsilon[t]
        end
        u
    end
end

_sb_ar1 = StanBlocks.@slic begin
    n_obs = num_elements(time)
    phi_raw ~ std_normal()
    phi = tanh(phi_raw)
    epsilon ~ std_normal(; n=n_obs)
    return ar1_recurse(phi, epsilon, n_obs)
end

# Differenced AR(1) path. Starting at zero is load-bearing: `1 + dar(time)`
# leaves the formula intercept as the initial level instead of sampling a
# second, confounded location inside the term. The first difference has no
# inherited momentum (`d[0] = 0`), matching the CDC recurrence exactly.
StanBlocks.@deffun begin
    differenced_ar1_path(beta::real, sigma::real, z::vector[n])::vector[n + 1] = begin
        x = rep_vector(0., n + 1)
        increment = 0.
        if n > 0
            for t in 1:n
                increment = beta * increment + sigma * z[t]
                x[t + 1] = x[t] + increment
            end
        end
        x
    end
end

_sb_dar1 = StanBlocks.@slic begin
    n_innov = num_elements(time) - 1
    beta ~ normal(0.5, 0.2; lower=0., upper=1.)
    sigma ~ normal(0., 0.2; lower=0.)
    z ~ std_normal(; n=n_innov)
    return differenced_ar1_path(beta, sigma, z)
end

# Penalized 1-D thin-plate regression spline. `Xnull` contains the unpenalized
# polynomial null space {1, x}; `Zpen` is the range-space basis after the
# wiggliness penalty has been diagonalized and absorbed into the columns. The
# standardized range coefficients therefore have one iid Gaussian scale,
# matching the mixed-model parameterization used by mgcv/brms. The caller adds
# the resulting length-N contribution directly to the linear predictor.
#
# That scale is `sd_pen[1]` rather than a bare `sds` so `sd(<lp|:>, s(x)) ~
# Exponential(scale)` configures THIS submodel instead of selecting a second
# copy of it: `brm_ranef_sd` carries a family switch, `family == 0` being the
# half-standard-normal the formula gets when it says nothing. Only the scale is
# configurable — `b_pen_raw` stays standardized, because scaling it would
# duplicate the smoothing SD and change the advertised parameterization
# (decision `145tp0o`).
_sb_s = StanBlocks.@slic begin
    n_pen = dims(Zpen)[2]
    b_fixed::vector[2]
    sd_pen ~ brm_ranef_sd(sd_family, sd_rate; n=1, lower=0.)
    b_pen_raw ~ std_normal(; n=n_pen)
    b_pen = sd_pen[1] * b_pen_raw
    return Xnull * b_fixed + Zpen * b_pen
end

# Two-margin tensor-product smooth. With cubic-regression-spline margins each
# null space has dimension two. Removing the tensor intercept leaves three
# unpenalized NN columns. The remaining RR, RN, and NR blocks correspond to the
# three penalties used by mgcv/brms `t2(..., full=FALSE)` and deliberately get
# distinct smoothing scales.
#
# The three scales are one `vector[3]` in fixed (rr, rn, nr) order so a per-block
# `sd(<lp|:>, t2(x, z), <block>)` statement can configure any subset of them
# through `brm_ranef_sd`'s family switch, leaving the rest half-standard-normal.
# `_sb_t2_sd_index` owns the name -> index mapping.
_sb_t2 = StanBlocks.@slic begin
    n_rr = dims(Zrr)[2]
    n_rn = dims(Zrn)[2]
    n_nr = dims(Znr)[2]
    b_fixed::vector[3]
    sd_pen ~ brm_ranef_sd(sd_family, sd_rate; n=3, lower=0.)
    b_rr_raw ~ std_normal(; n=n_rr)
    b_rn_raw ~ std_normal(; n=n_rn)
    b_nr_raw ~ std_normal(; n=n_nr)
    b_rr = sd_pen[1] * b_rr_raw
    b_rn = sd_pen[2] * b_rn_raw
    b_nr = sd_pen[3] * b_nr_raw
    return Xfixed * b_fixed + Zrr * b_rr + Zrn * b_rn + Znr * b_nr
end

# Fit/apply split for Wood's rank-k thin-plate regression spline (d=1, m=2).
# The radial kernel is eta(r)=r^3/12 (Wood 2003, eq. 7). We keep the k largest-
# magnitude eigenvectors of the full kernel matrix, impose the T' * delta = 0
# side constraint, and diagonalize the resulting range-space penalty. Applying
# the fitted object to new x values needs only the frozen training centers,
# shift, and penalty-whitened range projection.
function _sb_tps_kernel(x::AbstractVector{<:Real}, centers::AbstractVector{<:Real})
    E = Matrix{Float64}(undef, length(x), length(centers))
    for j in eachindex(centers), i in eachindex(x)
        E[i, j] = abs(Float64(x[i]) - Float64(centers[j]))^3 / 12
    end
    E
end

function _sb_fit_spline(x::AbstractVector{<:Real}; k::Int=10)
    k > 2 || error("sbimpl: `s(x)` needs basis dimension k > 2 (got $k)")
    xs = collect(Float64, x)
    all(isfinite, xs) || error("sbimpl: `s(x)` requires finite numeric data")
    length(unique(xs)) >= k || error(
        "sbimpl: `s(x)` needs at least $k unique x values for the default ",
        "thin-plate basis (got $(length(unique(xs))))")

    shift = sum(xs) / length(xs)
    centers = xs .- shift
    E = _sb_tps_kernel(centers, centers)
    eig_E = eigen(Symmetric(E))
    keep = sortperm(abs.(eig_E.values); rev=true)[1:k]
    U = eig_E.vectors[:, keep]
    D = eig_E.values[keep]

    T = hcat(ones(Float64, length(xs)), centers)
    Z = nullspace(transpose(T) * U)
    size(Z, 2) == k - 2 || error(
        "sbimpl: `s(x)` could not isolate the two-dimensional TPS null space")

    S = Symmetric(transpose(Z) * Diagonal(D) * Z)
    eig_S = eigen(S)
    penalty_scale = maximum(abs, eig_S.values)
    penalty_scale > 0 || error("sbimpl: `s(x)` produced a zero range-space penalty")
    tol = penalty_scale * eps(Float64) * 100
    minimum(eig_S.values) >= -tol || error(
        "sbimpl: `s(x)` produced a non-positive range-space penalty")
    penalty_values = max.(eig_S.values, tol)
    penalty_whitener = eig_S.vectors * Diagonal(inv.(sqrt.(penalty_values)))
    range_projection = U * Z * penalty_whitener

    (; shift, centers, range_projection, k)
end

function _sb_apply_spline(fit, x::AbstractVector{<:Real})
    xs = collect(Float64, x)
    all(isfinite, xs) || error("sbimpl: `s(x)` requires finite numeric data")
    centered = xs .- fit.shift
    Xnull = hcat(ones(Float64, length(xs)), centered)
    Zpen = _sb_tps_kernel(centered, fit.centers) * fit.range_projection
    Xnull, Zpen
end

_sb_spline_basis_tps(x::AbstractVector{<:Real}; k::Int=10) =
    _sb_apply_spline(_sb_fit_spline(x; k), x)

# Cubic regression spline margin used by `t2`. Knots follow R's default
# quantile algorithm (type 7) over the sorted unique training values. `F` maps
# knot values to the natural cubic spline's second derivatives, while `S` is
# the integrated-squared-second-derivative penalty. The positive eigenspace of
# `S` is penalty-whitened without mixing in the null space, so tensoring the
# marginal range/null pieces preserves the three `t2(full=false)` penalties.
function _sb_type7_knots(x::AbstractVector{<:Real}, k::Int)
    values = sort!(unique(collect(Float64, x)))
    length(values) >= k || error(
        "sbimpl: `t2` margin needs at least $k unique values (got $(length(values)))")
    n = length(values)
    knots = Vector{Float64}(undef, k)
    for i in 1:k
        pos = 1 + (n - 1) * (i - 1) / (k - 1)
        lo = clamp(floor(Int, pos), 1, n)
        hi = clamp(ceil(Int, pos), 1, n)
        weight = pos - lo
        knots[i] = (1 - weight) * values[lo] + weight * values[hi]
    end
    all(diff(knots) .> 0) || error(
        "sbimpl: `t2` margin produced non-distinct cubic-regression-spline knots")
    knots
end

function _sb_cr_second_derivative_map(knots::AbstractVector{<:Real})
    k = length(knots)
    h = diff(knots)
    all(h .> 0) || error("sbimpl: `t2` cubic-regression-spline knots must increase")
    D = zeros(Float64, k - 2, k)
    B = zeros(Float64, k - 2, k - 2)
    for i in 1:(k - 2)
        D[i, i] = inv(h[i])
        D[i, i + 1] = -inv(h[i]) - inv(h[i + 1])
        D[i, i + 2] = inv(h[i + 1])
        B[i, i] = (h[i] + h[i + 1]) / 3
        if i < k - 2
            B[i, i + 1] = h[i + 1] / 6
            B[i + 1, i] = B[i, i + 1]
        end
    end
    interior = B \ D
    F = zeros(Float64, k, k)
    F[2:(k - 1), :] .= interior
    F, transpose(D) * interior
end

function _sb_cr_basis(knots, F, x::AbstractVector{<:Real})
    k = length(knots)
    X = zeros(Float64, length(x), k)
    for (i, raw_x) in enumerate(x)
        xi = Float64(raw_x)
        if xi < knots[1]
            h = knots[2] - knots[1]
            xik = xi - knots[1]
            cjm = -xik * h / 3
            cjp = -xik * h / 6
            for q in 1:k
                X[i, q] = cjm * F[1, q] + cjp * F[2, q]
            end
            X[i, 1] += 1 - xik / h
            X[i, 2] += xik / h
        elseif xi > knots[k]
            h = knots[k] - knots[k - 1]
            xik = xi - knots[k]
            cjm = xik * h / 6
            cjp = xik * h / 3
            for q in 1:k
                X[i, q] = cjm * F[k - 1, q] + cjp * F[k, q]
            end
            X[i, k - 1] -= xik / h
            X[i, k] += 1 + xik / h
        else
            j = clamp(searchsortedlast(knots, xi), 1, k - 1)
            h = knots[j + 1] - knots[j]
            ajm = knots[j + 1] - xi
            ajp = xi - knots[j]
            cjm = ajm * (ajm * ajm / h - h) / 6
            cjp = ajp * (ajp * ajp / h - h) / 6
            for q in 1:k
                X[i, q] = cjm * F[j, q] + cjp * F[j + 1, q]
            end
            X[i, j] += ajm / h
            X[i, j + 1] += ajp / h
        end
    end
    X
end

function _sb_fit_cr_spline(x::AbstractVector{<:Real}; k::Int=5)
    k > 2 || error("sbimpl: `t2` basis dimensions must be integers greater than 2 (got $k)")
    xs = collect(Float64, x)
    isempty(xs) && error("sbimpl: `t2` cannot use an empty margin")
    all(isfinite, xs) || error("sbimpl: `t2` margins require finite numeric data")
    shift = sum(xs) / length(xs)
    scale = maximum(xs) - minimum(xs)
    scale > 0 || error("sbimpl: `t2` margin is degenerate (all values equal)")
    normalized = (xs .- shift) ./ scale
    knots = _sb_type7_knots(normalized, k)
    F, penalty = _sb_cr_second_derivative_map(knots)

    eig_penalty = eigen(Symmetric(penalty))
    order = sortperm(eig_penalty.values; rev=true)
    keep = order[1:(k - 2)]
    penalty_scale = maximum(abs, eig_penalty.values)
    tol = penalty_scale * eps(Float64) * 100
    minimum(eig_penalty.values) >= -tol || error(
        "sbimpl: `t2` cubic-regression-spline penalty is not positive semidefinite")
    minimum(eig_penalty.values[keep]) > tol || error(
        "sbimpl: `t2` could not isolate the two-dimensional marginal null space")
    range_projection = eig_penalty.vectors[:, keep] *
                       Diagonal(inv.(sqrt.(eig_penalty.values[keep])))

    null_const_scale = inv(sqrt(length(xs)))
    slope_norm = norm(normalized)
    slope_norm > 0 || error("sbimpl: `t2` margin has a zero linear null-space norm")

    (; shift, scale, knots, F, range_projection, null_const_scale, slope_norm, k)
end

function _sb_apply_cr_spline(fit, x::AbstractVector{<:Real})
    xs = collect(Float64, x)
    all(isfinite, xs) || error("sbimpl: `t2` margins require finite numeric data")
    normalized = (xs .- fit.shift) ./ fit.scale
    Xnull = hcat(fill(fit.null_const_scale, length(xs)),
                 normalized ./ fit.slope_norm)
    range = _sb_cr_basis(fit.knots, fit.F, normalized) * fit.range_projection
    Xnull, range
end

function _sb_row_tensor(A::AbstractMatrix, B::AbstractMatrix)
    size(A, 1) == size(B, 1) || error(
        "sbimpl: `t2` marginal basis row counts differ ($(size(A, 1)) vs $(size(B, 1)))")
    out = Matrix{Float64}(undef, size(A, 1), size(A, 2) * size(B, 2))
    for i in axes(out, 1), a in axes(A, 2), b in axes(B, 2)
        out[i, (a - 1) * size(B, 2) + b] = A[i, a] * B[i, b]
    end
    out
end

function _sb_t2_raw_blocks(margins, x, z)
    N1, R1 = _sb_apply_cr_spline(margins[1], x)
    N2, R2 = _sb_apply_cr_spline(margins[2], z)
    NN = _sb_row_tensor(N1, N2)
    (fixed=Matrix(NN[:, 2:end]), rr=_sb_row_tensor(R1, R2),
     rn=_sb_row_tensor(R1, N2), nr=_sb_row_tensor(N1, R2))
end

_sb_block_center(A::AbstractMatrix) = vec(sum(A; dims=1)) ./ size(A, 1)
_sb_center_block(A::AbstractMatrix, center) = A .- reshape(center, 1, :)

function _sb_fit_t2(x::AbstractVector{<:Real}, z::AbstractVector{<:Real};
                    k::Tuple{Int,Int}=(5, 5))
    length(x) == length(z) || error(
        "sbimpl: `t2(x, z)` margins must have equal lengths ($(length(x)) vs $(length(z)))")
    margins = (_sb_fit_cr_spline(x; k=k[1]), _sb_fit_cr_spline(z; k=k[2]))
    raw = _sb_t2_raw_blocks(margins, x, z)
    fixed_center = _sb_block_center(raw.fixed)
    (; margins, fixed_center, k)
end

function _sb_apply_t2(fit, x::AbstractVector{<:Real}, z::AbstractVector{<:Real})
    length(x) == length(z) || error(
        "sbimpl: `t2(x, z)` margins must have equal lengths ($(length(x)) vs $(length(z)))")
    raw = _sb_t2_raw_blocks(fit.margins, x, z)
    (_sb_center_block(raw.fixed, fit.fixed_center), raw.rr, raw.rn, raw.nr)
end

# brms-style `me(x_obs, sd_x)` measurement-error predictor. The submodel
# allocates a length-N latent `x_true` vector with prior `std_normal` and
# emits the observation likelihood `x_obs ~ normal(x_true, sd_x)` directly.
# (Earlier StanBlocks versions silently dropped data-LHS `~` inside submodel
# bodies; that's fixed, so we keep the likelihood self-contained.)
# The linear predictor uses `x_true` via popefs's free beta, so `me` behaves
# like a regular continuous covariate except the predictor values themselves
# are parameters.
# `x_true` is a genuine model-scale quantity — the latent TRUE covariate, on
# the same scale as the observed one — so unlike a standardized innovation it
# takes a prior directly, via `latent(<lp|:>, me(x)) ~ Normal(loc, scale)`.
# Location/scale are data, defaulting to (0, 1): the standard normal the
# inline form had. The observation likelihood is never configurable.
_sb_me = StanBlocks.@slic begin
    x_true ~ normal(x_true_loc, x_true_scale; n=num_elements(x_obs))
    x_obs ~ normal(x_true, sd_x)
    return x_true
end

# BLOQ predictor. Quantified rows stay data; `x == lloq` rows allocate only the
# unknown coordinates, give them the term's Normal prior truncated to
# their row-wise bounds, then scatter exact and latent values back onto the
# formula's row axis. This is evidence about the predictor itself, not a
# response-likelihood wrapper and not an LLOQ/2 substitution.
_sb_interval_censored_predictor = StanBlocks.@slic begin
    n_interval = num_elements(Jinterval)
    x_interval :: vector[n_interval] ~ truncated(
        normal, x_true_loc, x_true_scale; lower=x_lower, upper=x_upper)
    return mi_merge(x_exact, x_interval, Jexact, Jinterval,
                    num_elements(Jexact) + n_interval)
end

# Carvalho-Polson-Scott horseshoe prior, scalar form. Standard
# reparameterisation: beta = raw * lambda * tau with raw ~ N(0,1) and
# half-Cauchy(0,1) local + global scales. Each `coef ~ Horseshoe()`
# call site gets its own (raw, lambda, tau) triple via SLIC's per-call
# scoping.
_sb_horseshoe = StanBlocks.@slic begin
    raw    ~ std_normal()
    lambda ~ cauchy(0., 1.; lower=0.)
    tau    ~ cauchy(0., 1.; lower=0.)
    return raw * lambda * tau
end

# Configured sibling. The no-keyword formula path deliberately keeps calling
# `_sb_horseshoe`, so an unconfigured model retains its historical SLIC body
# and emitted Stan byte for byte. The raw draw stays standardized: changing it
# would duplicate the local/global scales and alter the advertised hierarchy.
_sb_horseshoe_scaled = StanBlocks.@slic begin
    raw    ~ std_normal()
    lambda ~ cauchy(0., local_scale; lower=0.)
    tau    ~ cauchy(0., global_scale; lower=0.)
    return raw * lambda * tau
end

# Missing-data scatter. Builds a length-`n` vector by placing the observed
# values at positions `Jobs` and the imputed parameters at positions `Jmis`.
# Mutation is allowed inside `@deffun` bodies (top-level @slic blocks are
# single-assignment), which is why the merge lives here rather than inline.
StanBlocks.@deffun begin
    mi_merge(y_obs::vector[n_obs], y_mis::vector[n_mis],
             Jobs::int[n_obs], Jmis::int[n_mis], n::int)::vector[n] = begin
        rv = rep_vector(0., n)
        if n_obs > 0
            for i in 1:n_obs; rv[Jobs[i]] = y_obs[i]; end
        end
        if n_mis > 0
            for i in 1:n_mis; rv[Jmis[i]] = y_mis[i]; end
        end
        return rv
    end
end

# Missing-data response submodel for the Normal family. Caller passes
# `loc`, `scale`, `y_obs`, `Jobs`, `Jmis` as kwargs (all data-qualified
# in the SLIC sense -- caller-provided). The two `~` lines split the
# joint likelihood:
#   - `y_mis ~ Normal(loc[Jmis], scale[Jmis])` introduces y_mis as a
#     parameter (LHS not yet bound) and contributes the missing-row
#     log-density. The conditional shape is exactly the family at those
#     positions, so no informative prior bias is introduced.
#   - `y_obs ~ Normal(loc[Jobs], scale[Jobs])` is the observed-row
#     likelihood (y_obs is :data-qualified via the kwarg, so SLIC routes
#     this to the model block).
# `mi_merge` then assembles the merged response vector for cross-formula
# references (e.g. `loc2 = a + b * y` in another formula).
# Per-family submodels (one each for Normal / BinomialLogit / Poisson / ...)
# rather than HOF-generic because each family's arg list shape differs and
# needs to be sliced per-arg at `[Jobs]` / `[Jmis]`.
_sb_mi_normal = StanBlocks.@slic begin
    # Typed-LHS sampling form: explicit `vector[n_mis]` so SLIC declares
    # y_mis as a vector parameter rather than inferring scalar from the
    # bare `normal(...)` call (which has no size-bearing kwarg).
    #
    # `maybe_index` is StanBlocks' shape-robust distribution-arg slicer:
    # a rank-1 vector arg lowers to `arg[idx]`, a rank-0 scalar arg passes
    # through unchanged. So a SCALAR family arg -- e.g. `sigma` from
    # `sigma ~ Exponential(2)` in `mi(y) ~ Normal(mu, sigma)` -- broadcasts
    # rather than being indexed. Raw `scale[Jmis]` on a `real` scalar traces
    # to `anything` and breaks the StanBlocks tracer (snag
    # `sbimpl-mi-impute`); `maybe_index` is the same builtin StanBlocks' own
    # native missing-outcome auto-imputation uses, so no call-site lifting is
    # needed and every scalar/vector combination of `loc`/`scale` traces.
    n_mis = num_elements(Jmis)
    y_mis :: vector[n_mis] ~ normal(maybe_index(loc, Jmis), maybe_index(scale, Jmis))
    y_obs ~ normal(maybe_index(loc, Jobs), maybe_index(scale, Jobs))
    return mi_merge(y_obs, y_mis, Jobs, Jmis,
                    num_elements(Jobs) + n_mis)
end

# Squared-exponential GP helpers. The data-layout conversion loop lives in a
# Stan function because top-level @slic bodies are deliberately control-flow
# free.
#
# BRM records exact-GP inputs as an N x d matrix so replay and descriptors keep
# their ordinary dense-data shape. `brm_gp_locations` converts its rows to
# Stan's native `array[N] vector[d]` GP carrier once in transformed data.
# `brm_exp_quad_cov` then delegates the whole multidimensional covariance and
# diagonal jitter to Stan's `gp_exp_quad_cov` / `add_diag` built-ins.
#
# `brm_hsgp_sqrt_spd` evaluates the separable d-dimensional squared-exponential
# spectral density at every tensor-product HSGP frequency. `omega2[b, j]` is
# the squared angular frequency for basis row b and predictor axis j.
#
# `brm_periodic_cov` / `brm_hsgp_periodic_sqrt_spd` are the `cov=:periodic`
# siblings (Riutort-Mayol et al. 2023, "periodic kernel"). Stan's
# `gp_periodic_cov` is k(x, x') = sigma^2 exp(-2 sin^2(pi |x - x'| / period) /
# rho^2); with a = 1 / rho^2 and w0 = 2 pi / period its Fourier expansion is
# sigma^2 exp(-a) [I_0(a) + 2 sum_j I_j(a) cos(j w0 (x - x'))], so the
# Hilbert-space basis is cos(j w0 x) / sin(j w0 x) with spectral weight
# q_j = sigma sqrt(2 exp(-a) I_j(a)) on BOTH the cosine and the sine column of
# harmonic j (the constant I_0 harmonic is dropped -- the formula intercept
# owns it). `harmonics[b]` is the harmonic index j of basis column b, computed
# in log space through `log_modified_bessel_first_kind` so a small `rho`
# (large `a`) cannot overflow exp(a).
StanBlocks.@deffun begin
    @stanonly brm_gp_locations(X::matrix[n, d])::vector[n, d] = begin
        locations::vector[n, d]
        for i in 1:n
            locations[i] = to_vector(X[i, :])
        end
        return locations
    end

    @stanonly brm_exp_quad_cov(X::vector[n, d], sigma::real,
                               rho::real, jitter::real)::matrix[n, n] = begin
        return add_diag(gp_exp_quad_cov(X, sigma, rho), jitter)
    end

    @stanonly brm_exp_quad_cov(X::vector[n, d], sigma::real,
                               rho::vector[d], jitter::real)::matrix[n, n] = begin
        return add_diag(gp_exp_quad_cov(X, sigma, rho), jitter)
    end

    @stanonly brm_hsgp_sqrt_spd(omega2::matrix[m, d], sigma::real,
                                 rho::vector[d])::vector[m] = begin
        rv::vector[m]
        scale = sigma
        for axis in 1:d
            scale *= sqrt(rho[axis] * 2.5066282746310002)
        end
        for b in 1:m
            exponent = 0.
            for axis in 1:d
                exponent += rho[axis] * rho[axis] * omega2[b, axis]
            end
            rv[b] = scale * exp(-0.25 * exponent)
        end
        return rv
    end

    @stanonly brm_periodic_cov(X::matrix[n, 1], sigma::real, rho::real,
                               period::real, jitter::real)::matrix[n, n] = begin
        return add_diag(gp_periodic_cov(to_array_1d(col(X, 1)), sigma, rho, period),
                        jitter)
    end

    @stanonly brm_hsgp_periodic_sqrt_spd(harmonics::vector[m], sigma::real,
                                          rho::real)::vector[m] = begin
        rv::vector[m]
        a = 1. / (rho * rho)
        base = log(sigma) + 0.5 * (log(2.) - a)
        for b in 1:m
            rv[b] = exp(base + 0.5 * log_modified_bessel_first_kind(harmonics[b], a))
        end
        return rv
    end

    # A model-derived HSGP axis is not available while Julia materialises the
    # data dictionary, so its eigenfunctions have to be evaluated inside Stan.
    # `omega2`, `center`, and `L` still describe one FIXED approximation domain;
    # only the locations `x` vary with the sampled latent predictor.
    @stanonly brm_hsgp_basis_1d(x::vector[n], omega2::matrix[m, 1],
                                center::real, L::real)::matrix[n, m] = begin
        PHI::matrix[n, m]
        inv_sqrt_L = 1. / sqrt(L)
        for b in 1:m
            omega = sqrt(omega2[b, 1])
            for i in 1:n
                PHI[i, b] = inv_sqrt_L * sin(omega * (x[i] - center + L))
            end
        end
        return PHI
    end

    # Remove the intercept and linear-x directions from a one-dimensional
    # HSGP basis. At a degenerate all-equal x draw the linear direction is the
    # intercept, so centering alone is the well-defined limiting projection.
    # This keeps a separately declared population slope interpretable while
    # the HSGP carries only residual nonlinear shape.
    @stanonly brm_hsgp_orthogonalize_linear(PHI::matrix[n, m],
                                             x::vector[n])::matrix[n, m] = begin
        out::matrix[n, m]
        x_centered = x - mean(x)
        x_ss = dot_self(x_centered)
        for b in 1:m
            phi_centered = col(PHI, b) - mean(col(PHI, b))
            if x_ss > 1e-12
                phi_centered = phi_centered - x_centered *
                    (dot_product(x_centered, phi_centered) / x_ss)
            end
            for i in 1:n
                out[i, b] = phi_centered[i]
            end
        end
        return out
    end
end

# Exact latent squared-exponential GP. The non-centred draw keeps the geometry
# explicit: f = cholesky(K(X, X)) * z. These are direct predictor summands, so
# there is no redundant population beta multiplying the returned draw.
#
# LOCKSTEP: `length_scale(lp, gp(x))` / `sd(lp, gp(x))` override these two
# statements through `Base.merge`, which replaces a matching-named statement
# WHOLESALE -- so `_sb_gp_rho_lhs` reproduces each submodel's `rho` LHS
# character-for-character. Rename `rho`/`rho_iso`, or change its declared type,
# and that table has to change with it. `_sb_gp_submodel` likewise maps each
# submodel NAME back to the value below.
_sb_gp = StanBlocks.@slic begin
    n_obs = dims(X)[1]
    X_gp = brm_gp_locations(X)
    rho   ~ lognormal(0., 1.; lower=0.)
    sigma ~ lognormal(0., 1.; lower=0.)
    z     ~ std_normal(; n=n_obs)
    K = brm_exp_quad_cov(X_gp, sigma, rho, jitter)
    return cholesky_decompose(K) * z
end

_sb_gp_aniso = StanBlocks.@slic begin
    n_obs = dims(X)[1]
    n_axes = dims(X)[2]
    X_gp = brm_gp_locations(X)
    rho :: vector[n_axes] ~ lognormal(0., 1.; lower=0.)
    sigma ~ lognormal(0., 1.; lower=0.)
    z     ~ std_normal(; n=n_obs)
    K = brm_exp_quad_cov(X_gp, sigma, rho, jitter)
    return cholesky_decompose(K) * z
end

# Exact periodic GP (`gp(x; cov=:periodic, period=...)`): one axis, Stan's
# native `gp_periodic_cov`. `period` is a formula constant bound as data.
_sb_gp_periodic = StanBlocks.@slic begin
    n_obs = dims(X)[1]
    rho   ~ lognormal(0., 1.; lower=0.)
    sigma ~ lognormal(0., 1.; lower=0.)
    z     ~ std_normal(; n=n_obs)
    K = brm_periodic_cov(X, sigma, rho, period, jitter)
    return cholesky_decompose(K) * z
end

# Hilbert-space approximate GP (Riutort-Mayol et al. 2022). `PHI` and
# `omega2` are tensor-product basis data precomputed by Julia. Isotropic and
# anisotropic variants differ only in whether one or d log length scales are
# sampled. As with exact GP, the returned draw is a direct predictor summand.
#
# `rho_lower` is the approximation's validity floor (`_sb_hsgp_rho_lower`),
# supplied as data by the emitter -- scalar here, one entry per axis in the
# `_aniso` spellings. It is the DEFAULT bound only: `length_scale(lp, hsgp(x))`
# replaces this whole statement through `Base.merge`, so an explicit
# declaration sets its own support and this floor does not apply.
_sb_hsgp = StanBlocks.@slic begin
    n_basis = dims(omega2)[1]
    n_axes = dims(omega2)[2]
    rho_iso  ~ lognormal(0., 1.; lower=rho_lower)
    sigma    ~ lognormal(0., 1.; lower=0.)
    beta_raw ~ std_normal(; n=n_basis)
    rho = rep_vector(rho_iso, n_axes)
    sqrt_spd = brm_hsgp_sqrt_spd(omega2, sigma, rho)
    return PHI * (sqrt_spd .* beta_raw)
end

_sb_hsgp_aniso = StanBlocks.@slic begin
    n_basis = dims(omega2)[1]
    n_axes = dims(omega2)[2]
    rho :: vector[n_axes] ~ lognormal(0., 1.; lower=rho_lower)
    sigma    ~ lognormal(0., 1.; lower=0.)
    beta_raw ~ std_normal(; n=n_basis)
    sqrt_spd = brm_hsgp_sqrt_spd(omega2, sigma, rho)
    return PHI * (sqrt_spd .* beta_raw)
end

# Periodic Hilbert-space basis (`hsgp(x; k, cov=:periodic, period=...)`).
# `PHI` holds the `2k` cosine/sine columns precomputed by Julia and
# `harmonics` the harmonic index of each column; there is no boundary factor
# and no domain, so nothing here is data-derived except the axis itself. The
# parameter names deliberately match `_sb_hsgp` (`rho_iso`, `sigma`,
# `beta_raw`) so the term-prior addresses and descriptor roles are shared.
# `rho_lower` is the periodic validity floor (`_sb_hsgp_periodic_rho_lower`).
_sb_hsgp_periodic = StanBlocks.@slic begin
    n_basis = dims(harmonics)[1]
    rho_iso  ~ lognormal(0., 1.; lower=rho_lower)
    sigma    ~ lognormal(0., 1.; lower=0.)
    beta_raw ~ std_normal(; n=n_basis)
    sqrt_spd = brm_hsgp_periodic_sqrt_spd(harmonics, sigma, rho_iso)
    return PHI * (sqrt_spd .* beta_raw)
end

# One-dimensional HSGP over a model-derived vector. Unlike `_sb_hsgp`, PHI is
# evaluated at runtime because `x` may be a sampled linear predictor or an
# assignment such as `x = exp(log_x)`. The approximation domain and spectral
# frequencies remain formula constants/data, so the length-scale validity
# floor is still a legal static Stan parameter bound.
_sb_hsgp_latent = StanBlocks.@slic begin
    n_basis = dims(omega2)[1]
    PHI = brm_hsgp_basis_1d(x, omega2, center, L)
    rho_iso  ~ lognormal(0., 1.; lower=rho_lower)
    sigma    ~ lognormal(0., 1.; lower=0.)
    beta_raw ~ std_normal(; n=n_basis)
    rho = rep_vector(rho_iso, 1)
    sqrt_spd = brm_hsgp_sqrt_spd(omega2, sigma, rho)
    return PHI * (sqrt_spd .* beta_raw)
end

_sb_hsgp_latent_orthogonal = StanBlocks.@slic begin
    n_basis = dims(omega2)[1]
    PHI_raw = brm_hsgp_basis_1d(x, omega2, center, L)
    PHI = brm_hsgp_orthogonalize_linear(PHI_raw, x)
    rho_iso  ~ lognormal(0., 1.; lower=rho_lower)
    sigma    ~ lognormal(0., 1.; lower=0.)
    beta_raw ~ std_normal(; n=n_basis)
    rho = rep_vector(rho_iso, 1)
    sqrt_spd = brm_hsgp_sqrt_spd(omega2, sigma, rho)
    return PHI * (sqrt_spd .* beta_raw)
end

# Per-group HSGP. Length-scale/marginal-SD hyperparameters are shared across
# groups (decision 7p44fo); only tensor-basis weights vary by group.
_sb_hsgp_by = StanBlocks.@slic begin
    n_axes = dims(omega2)[2]
    rho_iso ~ lognormal(0., 1.; lower=rho_lower)
    sigma   ~ lognormal(0., 1.; lower=0.)
    rho = rep_vector(rho_iso, n_axes)
    sqrt_spd = brm_hsgp_sqrt_spd(omega2, sigma, rho)
    PHI_scaled = diag_post_multiply(PHI, sqrt_spd)
    return rows_dot_product(PHI_scaled, beta[group_idx, :])
end

_sb_hsgp_by_aniso = StanBlocks.@slic begin
    n_axes = dims(omega2)[2]
    rho :: vector[n_axes] ~ lognormal(0., 1.; lower=rho_lower)
    sigma ~ lognormal(0., 1.; lower=0.)
    sqrt_spd = brm_hsgp_sqrt_spd(omega2, sigma, rho)
    PHI_scaled = diag_post_multiply(PHI, sqrt_spd)
    return rows_dot_product(PHI_scaled, beta[group_idx, :])
end

# Categorical -> (n_levels::Int, per-row level index::Vector{Int}). Mirrors
# vimpl._level_index so the integer indices the walker stashes in `data`
# agree with what the cimpl-side uses.
_sb_level_index(raw::AbstractVector) = _brm_level_index(raw)

# Fit/apply split for categorical level coding (factor / mo). `_sb_fit_levels`
# returns the ordered level set (the frozen constant); `_sb_apply_levels` maps a
# raw column to 1-based codes against a (possibly frozen) level set, erroring on
# an unseen level (the dimension-coupled guard — brm-use §4 constraint 8). On
# the SAME training column these reproduce `_sb_level_index`'s codes exactly:
# for a CategoricalVector the level position == `CA.levelcode`; for a plain
# vector `sort(unique)` gives the same ordering. `_sb_level_index` (the
# construct-time entry) is unchanged.
_sb_fit_levels(raw::AbstractVector) = _brm_fit_levels(raw)
_sb_apply_levels(levels, raw::CA.CategoricalVector) = _sb_apply_levels(levels, CA.unwrap.(raw))
_sb_apply_levels(levels, raw::AbstractVector) = begin
    lm = Dict(l => i for (i, l) in enumerate(levels))
    idx = Vector{Int}(undef, length(raw))
    for (j, l) in enumerate(raw)
        haskey(lm, l) || error(
            "sbimpl: reprocess: value `$l` is not a training level for this factor ",
            "(training levels: $(collect(levels))). The trained model has no ",
            "parameter for an unseen level. Re-fit with `freeze_constants=false` to ",
            "re-derive levels from the new data, or drop unseen categories first.")
        idx[j] = lm[l]
    end
    idx
end

# Random-effect group coding has the same frozen-level geometry as `factor`,
# but deserves its own diagnostic: the missing coordinate is a fitted group
# effect, not a treatment contrast.  Keep this separate from `_sb_apply_levels`
# so an unseen group cannot be misreported as a factor-level problem.
_sb_group_values(raw::CA.CategoricalVector) = CA.unwrap.(raw)
_sb_group_values(raw::AbstractVector) = raw
function _sb_apply_group_levels(levels, raw::AbstractVector, group::Symbol)
    values = _sb_group_values(raw)
    lm = Dict(l => i for (i, l) in enumerate(levels))
    idx = Vector{Int}(undef, length(values))
    for (row, level) in enumerate(values)
        haskey(lm, level) || error(
            "sbimpl: reprocess: random-effects grouping column `$group` has " *
            "unseen level `$(level)` at row $row (training levels: " *
            "$(collect(levels))). The fitted model has no random-effect " *
            "coordinate for that level. Rebuild for a new population, or use " *
            "only fitted groups for frozen replay.")
        idx[row] = lm[level]
    end
    idx
end

# GP input helpers. Both public terms accept one-or-more raw real-valued axes
# and lower them to an N x d matrix. Keeping the raw column names in the
# preprocessing record lets `reprocess` rebuild that matrix on new data.
function _sb_gp_axes(label::Symbol, args::Tuple)
    isempty(args) && error("sbimpl: `$label(x...)` expects at least one positional axis")
    names = Symbol[]
    axes = Vector{Float64}[]
    for a in args
        n, raw = _sb_inner_data(label, a)
        v = collect(Float64, _sb_real_vec(label, n, raw))
        isempty(v) && error("sbimpl: `$label($n)` cannot use an empty axis")
        all(isfinite, v) || error("sbimpl: `$label($n)` requires finite values")
        push!(names, n)
        push!(axes, v)
    end
    n = length(first(axes))
    all(v -> length(v) == n, axes) || error(
        "sbimpl: `$label(x...)` axes must have equal lengths (got $(length.(axes)))")
    Tuple(names), Tuple(axes)
end

_sb_gp_matrix(axes::Tuple) = Matrix{Float64}(hcat(axes...))

function _sb_axis_option(label::Symbol, key::Symbol, value, n_axes::Int, pred, expectation::String)
    values = value isa Tuple || value isa AbstractVector ? Tuple(value) : ntuple(_ -> value, n_axes)
    length(values) == n_axes || error(
        "sbimpl: `$label(...; $key=...)` needs one value per axis ($n_axes), got $(length(values))")
    all(pred, values) || error(
        "sbimpl: `$label(...; $key=...)` expects $expectation, got $values")
    values
end

_sb_hsgp_options(kw, n_axes::Int) = begin
    K = _sb_axis_option(:hsgp, :k, get(kw, :k, 20), n_axes,
        x -> x isa Integer && !(x isa Bool) && x >= 1, "positive integers")
    c = _sb_axis_option(:hsgp, :c, get(kw, :c, 1.5), n_axes,
        x -> x isa Real && isfinite(x) && x > 1, "finite real values greater than 1")
    Tuple(Int(x) for x in K), Tuple(Float64(x) for x in c)
end

# `domain` is the actual compact interval used by the HSGP eigenfunctions,
# unlike `c`, which expands a domain inferred from raw training data. A latent
# axis has no Julia-time values from which to infer that interval, so it must
# supply one explicitly. Variadic raw HSGPs may supply one pair per axis.
function _sb_hsgp_domain_fits(kw, n_axes::Int; required::Bool=false)
    if !haskey(kw, :domain)
        required && error(
            "sbimpl: `hsgp(...)` over a model-derived axis requires an explicit " *
            "fixed `domain=(lower, upper)`; its sampled values do not exist " *
            "while the HSGP basis is being configured")
        return nothing
    end
    haskey(kw, :c) && error(
        "sbimpl: `hsgp(...; domain=...)` fixes the approximation boundary " *
        "directly and cannot also specify the data-derived expansion factor `c`")

    raw = kw[:domain]
    is_pair(x) = (x isa Tuple || x isa AbstractVector) && length(x) == 2 &&
                 all(v -> v isa Real && isfinite(v), x)
    pairs = if n_axes == 1 && is_pair(raw)
        (raw,)
    elseif (raw isa Tuple || raw isa AbstractVector) && length(raw) == n_axes &&
           all(is_pair, raw)
        Tuple(raw)
    else
        expectation = n_axes == 1 ? "`(lower, upper)`" :
            "one `(lower, upper)` pair per axis"
        error("sbimpl: `hsgp(...; domain=...)` expects $expectation, got $(repr(raw))")
    end

    ntuple(n_axes) do j
        lower, upper = Float64.(pairs[j])
        lower < upper || error(
            "sbimpl: `hsgp(...; domain=...)` needs lower < upper on axis $j, " *
            "got ($lower, $upper)")
        ((lower + upper) / 2, (upper - lower) / 2)
    end
end

function _sb_hsgp_orthogonal_to(kw, n_axes::Int)
    value = get(kw, :orthogonal_to, nothing)
    (isnothing(value) || value === :linear) || error(
        "sbimpl: `hsgp(...; orthogonal_to=...)` supports only `:linear`, got $(repr(value))")
    value === :linear && n_axes != 1 && error(
        "sbimpl: `hsgp(...; orthogonal_to=:linear)` currently supports exactly " *
        "one predictor axis, got $n_axes")
    value
end

_sb_gp_iso(kw, label::Symbol) = begin
    iso = get(kw, :iso, true)
    iso isa Bool || error("sbimpl: `$label(...; iso=...)` expects Bool, got $(typeof(iso))")
    iso
end

const _SB_GP_COVARIANCES = (:exp_quad, :periodic)

_sb_gp_cov(kw, label::Symbol) = begin
    cov = get(kw, :cov, :exp_quad)
    cov in _SB_GP_COVARIANCES || error(
        "sbimpl: `$label(...; cov=...)` supports " *
        join(("`$(repr(c))`" for c in _SB_GP_COVARIANCES), " and ") *
        ", got $(repr(cov))")
    cov
end

# `period` is the periodic kernel's formula constant: required with
# `cov=:periodic`, meaningless (and refused) otherwise.
function _sb_gp_period(kw, label::Symbol, cov::Symbol)
    if cov !== :periodic
        haskey(kw, :period) && error(
            "sbimpl: `$label(...; period=...)` is meaningful only with " *
            "`cov=:periodic` (got `cov=$(repr(cov))`)")
        return nothing
    end
    haskey(kw, :period) || error(
        "sbimpl: `$label(...; cov=:periodic)` requires a numeric `period=` " *
        "formula constant (the kernel's period on the axis's own scale)")
    period = kw[:period]
    (period isa Real && !(period isa Bool) && isfinite(period) && period > 0) || error(
        "sbimpl: `$label(...; period=...)` expects a finite positive numeric " *
        "formula constant, got $(repr(period))")
    Float64(period)
end

function _check_term_kwargs(::typeof(gp), kw)
    allowed = (:cov, :iso, :jitter, :period)
    unknown = filter(k -> k ∉ allowed, keys(kw))
    isempty(unknown) || error(
        "gp: exact GP accepts only `cov`, `iso`, `jitter`, and `period`; unsupported keyword(s): $(join(unknown, ", ")). " *
        "Use `hsgp(...; k=..., c=..., by=...)` for the Hilbert-space approximation.")
    cov = _sb_gp_cov(kw, :gp)
    _sb_gp_period(kw, :gp, cov)
    _sb_gp_iso(kw, :gp)
    jitter = get(kw, :jitter, 1e-9)
    jitter isa Real && isfinite(jitter) && jitter > 0 || error(
        "gp: `jitter` must be a finite positive real, got $(repr(jitter))")
    nothing
end

function _check_term_kwargs(::typeof(hsgp), kw)
    allowed = (:cov, :iso, :k, :c, :by, :domain, :orthogonal_to, :period)
    unknown = filter(k -> k ∉ allowed, keys(kw))
    isempty(unknown) || error(
        "hsgp: unsupported keyword(s): $(join(unknown, ", ")); " *
        "supported keywords are `cov`, `iso`, `k`, `c`, `by`, `domain`, " *
        "`orthogonal_to`, and `period`")
    cov = _sb_gp_cov(kw, :hsgp)
    _sb_gp_period(kw, :hsgp, cov)
    _sb_gp_iso(kw, :hsgp)
    nothing
end

function _sb_t2_options(kw)
    kval = get(kw, :k, (5, 5))
    kval isa Tuple && length(kval) == 2 || error(
        "t2: `k` must be a 2-tuple of integers greater than 2, got $(repr(kval))")
    all(x -> x isa Integer && !(x isa Bool) && x > 2, kval) || error(
        "t2: `k` must be a 2-tuple of integers greater than 2, got $(repr(kval))")

    basis = get(kw, :basis, (:cr, :cr))
    basis isa Tuple && length(basis) == 2 || error(
        "t2: `basis` must be a 2-tuple; only `(:cr, :cr)` is currently supported")
    basis == (:cr, :cr) || error(
        "t2: only cubic-regression-spline margins `basis=(:cr, :cr)` are currently supported, got $(repr(basis))")

    full = get(kw, :full, false)
    full isa Bool || error("t2: `full` must be Bool, got $(typeof(full))")
    full && error("t2: `full=true` is not supported yet; use `full=false`")
    (Tuple(Int(x) for x in kval), basis, full)
end

function _check_term_kwargs(::typeof(t2), kw)
    allowed = (:k, :basis, :full)
    unknown = filter(k -> k ∉ allowed, keys(kw))
    isempty(unknown) || error(
        "t2: unsupported keyword(s): $(join(unknown, ", ")); " *
        "supported keywords are `k`, `basis`, and `full`")
    _sb_t2_options(kw)
    nothing
end

function _sb_prepare_mm(raw_groups::Tuple, raw_weights, normalize::Bool;
                        levels=nothing,
                        group_names=ntuple(i -> Symbol(:group_, i), length(raw_groups)),
                        weight_names=nothing)
    _brm_prepare_mm(
        raw_groups, raw_weights, normalize;
        levels, group_names, weight_names, prefix="sbimpl")
end

# HSGP fit/apply split. The scalar methods reproduce the historical 1D basis;
# the tuple methods form its tensor product for variadic `hsgp(x...)`.
_sb_fit_hsgp(raw::AbstractVector{<:Real}, K::Integer, c::Real) = begin
    K >= 1 || error("hsgp: k must be >= 1 (got $K)")
    c > 1  || error("hsgp: c must be > 1 (got $c)")
    mu = sum(raw) / length(raw)
    L = c * maximum(abs, raw .- mu)
    L > 0 || error("hsgp: degenerate input (all x equal)")
    (mu, L)
end
_sb_apply_hsgp(c::Tuple, raw::AbstractVector{<:Real}, K::Integer) = begin
    mu, L = c
    x_c = raw .- mu
    lambda = [(k * pi / (2 * L))^2 for k in 1:K]
    PHI = zeros(length(raw), K)
    inv_sqrt_L = 1 / sqrt(L)
    for k in 1:K, i in eachindex(x_c)
        PHI[i, k] = inv_sqrt_L * sin(sqrt(lambda[k]) * (x_c[i] + L))
    end
    PHI, lambda
end

_sb_fit_hsgp(axes::Tuple, K::Tuple, c::Tuple) =
    ntuple(j -> _sb_fit_hsgp(axes[j], K[j], c[j]), length(axes))

# The weight threshold `w` in the validity bound below. 100 is the value the
# reference port uses; it is not reachable from the formula.
const _SB_HSGP_WEIGHT_THRESHOLD = 100.0

# Riutort-Mayol et al. (2022) bound where the Hilbert-space approximation stops
# representing the kernel: with `k` basis functions on a domain of half-width
# `L`, a length scale below
#
#     (4L/pi) * sqrt(log(w) / (k^2 - 1))
#
# is not approximated, and the model silently becomes a GP nobody asked for --
# it still transpiles, still samples, still returns finite draws. Both inputs
# are known here, so `hsgp` declares `rho` with this as its lower bound by
# DEFAULT (decision 13keyez).
#
# It is passed as DATA rather than baked into the emitted Stan because `L`
# comes from the covariate: a `reprocess` with `freeze_constants=false`
# re-fits the basis on new data, and a literal would leave the bound describing
# the OLD basis while `PHI`/`omega2` describe the new one.
#
# `k == 1` has no usable floor (`k^2 - 1 == 0` puts the bound at infinity), so
# that degenerate basis stays unbounded rather than emitting an
# impossible-to-satisfy declaration.
_sb_hsgp_rho_lower(fit::Tuple, K::Integer) = begin
    K > 1 || return 0.0
    _, L = fit
    (4 * L / pi) * sqrt(log(_SB_HSGP_WEIGHT_THRESHOLD) / (K^2 - 1))
end

# Per-axis bounds for the anisotropic spelling; the isotropic one shares a
# single `rho` across every axis, so it must satisfy the STRICTEST of them.
_sb_hsgp_rho_lowers(fits::Tuple, K::Tuple) =
    [_sb_hsgp_rho_lower(fits[j], K[j]) for j in eachindex(fits)]

_sb_hsgp_rho_lower_data(fits::Tuple, K::Tuple, iso::Bool) =
    iso ? maximum(_sb_hsgp_rho_lowers(fits, K)) : _sb_hsgp_rho_lowers(fits, K)

function _sb_apply_hsgp(fits::Tuple, axes::Tuple, K::Tuple)
    n_axes = length(axes)
    length(fits) == n_axes == length(K) || error("hsgp: internal axis-count mismatch")
    axis_basis = ntuple(j -> _sb_apply_hsgp(fits[j], axes[j], K[j]), n_axes)
    n_obs = length(first(axes))
    n_basis = prod(K)
    PHI = Matrix{Float64}(undef, n_obs, n_basis)
    omega2 = Matrix{Float64}(undef, n_basis, n_axes)
    for (b, I) in enumerate(CartesianIndices(K))
        for i in 1:n_obs
            value = 1.0
            for axis in 1:n_axes
                value *= axis_basis[axis][1][i, I[axis]]
            end
            PHI[i, b] = value
        end
        for axis in 1:n_axes
            omega2[b, axis] = axis_basis[axis][2][I[axis]]
        end
    end
    PHI, omega2
end

# Periodic Hilbert-space basis: `k` harmonics of the fundamental angular
# frequency `2pi / period`, `2k` columns -- cosines first, then sines -- with
# no centering, boundary factor, or domain. `harmonics` is the per-column
# harmonic index the Stan-side spectral weight reads.
function _sb_apply_hsgp_periodic(period::Real, raw::AbstractVector{<:Real},
                                 K::Integer)
    K >= 1 || error("hsgp: k must be >= 1 (got $K)")
    period > 0 || error("hsgp: period must be positive (got $period)")
    w0 = 2pi / period
    PHI = Matrix{Float64}(undef, length(raw), 2K)
    for j in 1:K, i in eachindex(raw)
        angle = w0 * j * raw[i]
        PHI[i, j] = cos(angle)
        PHI[i, K + j] = sin(angle)
    end
    PHI
end

_sb_hsgp_periodic_harmonics(K::Integer) =
    Float64[repeat(1:K, 2)...]

# The periodic analogue of `_sb_hsgp_rho_lower`, by the SAME amplitude-ratio
# rule: the exp-quad floor is the length scale at which the k-th basis
# function's spectral amplitude (sqrt of the spectral density) has fallen to
# 1/w of the first's -- `S(omega_k)/S(omega_1) = w^-2`, which is exactly what
# `(4L/pi) sqrt(log(w)/(k^2-1))` solves. The periodic basis has amplitude
# q_j = sigma sqrt(2 exp(-a) I_j(a)) with a = 1/rho^2, so the same rule reads
# `I_k(a)/I_1(a) = w^-2`. That ratio is monotone in `a` and has no closed
# form, so it is solved by bisection on the exponentially scaled Bessel
# functions (`besselix`, which never overflows). Unlike exp-quad the floor
# depends on `k` alone: there is no data-derived domain, so `reprocess` with
# either `freeze_constants` reproduces it exactly. `k == 1` stays unbounded
# for the same reason as the exp-quad case (no truncated harmonic to bound).
function _sb_hsgp_periodic_rho_lower(K::Integer)
    K > 1 || return 0.0
    target = _SB_HSGP_WEIGHT_THRESHOLD^-2
    ratio(loga) = let a = exp(loga)
        SpecialFunctions.besselix(K, a) / SpecialFunctions.besselix(1, a) - target
    end
    # AMOS refuses |z| beyond ~1e8 (argument-reduction accuracy), and the
    # solution sits at a ~ (k^2 - 1) / (2 log(w^2)) — far inside this bracket
    # for any usable k.
    lo, hi = log(1e-12), log(1e7)
    ratio(lo) < 0 < ratio(hi) || error(
        "hsgp: internal periodic validity-floor bracket failed for k=$K")
    for _ in 1:200
        mid = (lo + hi) / 2
        ratio(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    1 / sqrt(exp((lo + hi) / 2))
end

function _sb_hsgp_periodic_frozen_check(data, key, names, K::Integer,
                                        period::Real)
    frozen = _sb_frozen_preproc_entry(data, key, :hsgp, names)
    isnothing(frozen) && return nothing
    const_ = frozen.const_
    (get(const_, :cov, :exp_quad) === :periodic && const_.K == K &&
     const_.period == period) || error(
        "sbimpl: resample replay: fitted periodic HSGP configuration for " *
        "`$key` no longer matches the re-emitted formula")
    nothing
end

function _sb_orthogonalize_hsgp_linear(PHI::AbstractMatrix,
                                       x::AbstractVector{<:Real})
    size(PHI, 1) == length(x) || error(
        "hsgp: internal orthogonalization row-count mismatch")
    xc = collect(Float64, x)
    xc .-= sum(xc) / length(xc)
    ss = sum(abs2, xc)
    out = Matrix{Float64}(undef, size(PHI))
    for b in axes(PHI, 2)
        phi = collect(Float64, @view PHI[:, b])
        phi .-= sum(phi) / length(phi)
        ss > 0 && (phi .-= xc .* (dot(xc, phi) / ss))
        out[:, b] = phi
    end
    out
end


# ==============================================================================
# SBBRMI: walk a BRMI, emit a SlicModel whose body references raw data columns
# by name. Scope for phase 1: population fixed effects + Normal likelihood.
# Ranefs / categorical / non-Normal likelihoods error out clearly.
#
# Design contract: data columns are referenced in the emitted expression by
# their formula names. `hcat(rep_vector(1., n), a, c1)` builds the design
# matrix inside Stan at runtime, so changing the data size (= length of the
# vectors) does not require recompiling the Stan model. StanBlocks' activity
# analysis routes each statement to the right Stan block (data / transformed
# data / parameters / model).
# ==============================================================================

# ---- preprocessing-constant provenance (decision nr3v8n A) ------------------
# Each Category-A transform (zscale/standardize/center/factor/mo/s/t2/gp/hsgp),
# interval-censored predictor, and element-wise `protect`/implicit-fn fallback
# computes data-derived values in Julia at construct-time and lands only the
# transformed/split result in `data`. To support `reprocess`/`restan_data` on a
# new DataFrame we record, per EMITTED data key, how to regenerate it:
#   kind         -- which transform (:zscale/:standardize/:center/:factor/:mo/
#                   :spline/:tensor_spline/:gp/:hsgp/:protect/:interaction/
#                   :categorical_outcome/:ordinal_outcome/
#                   :ordinal_threshold_predictor/:interval_censored_predictor)
#   const_       -- the fitted constant: (μ,σ) / μ / level-vector / TPS basis /
#                   tensor-spline margins/centers / exact-GP axis metadata /
#                   HSGP (μ,L,K,c) / categorical
#                   outcome levels / nothing (protect)
#   raw_ref      -- the source: a column-node tree (zscale/center/standardize/
#                   protect, re-materialised via `_sb_rematerialize_vec`) or a
#                   column NAME Symbol (factor/mo/spline) or axis-name Tuple
#                   (gp/hsgp)
#   dim_coupled  -- true when a fitted level set drives parameter dimension
struct PreprocEntry
    kind::Symbol
    const_::Any
    raw_ref::Any
    dim_coupled::Bool
end

# During resample-group re-emission, the new-data BRMI still has to be lowered
# once to obtain the CV-contagious Stan body.  Frozen replay must make fitted
# transform constants available to that lowering pass: otherwise an eager
# constructor-time fit can reject perfectly valid prediction data before
# `reprocess` gets a chance to apply the training constants (notably an HSGP
# axis that is constant only on the future schedule).  Keep the frozen inputs
# separate from the entries recorded by this pass so the shape comparison in
# `_sb_resample_preproc` remains meaningful.
struct _SBPreprocContext
    recorded::Dict{Symbol,PreprocEntry}
    frozen::Any
end

# Reserved side-channel key: during construction the emitters record into
# `data[_SB_PREPROC_KEY]`; the constructor pops it BEFORE building the SlicModel
# so it never reaches Stan's data dict. Filtered in `_sb_any_data_symbol`'s
# last-resort fallback so a data-iterating helper can never mistake it for a
# column while present.
const _SB_PREPROC_KEY = :__preproc__

_sb_record_preproc!(data, key::Symbol, entry::PreprocEntry) = begin
    pp = get(data, _SB_PREPROC_KEY, nothing)
    pp === nothing && return nothing   # recording disabled (defensive)
    pp isa _SBPreprocContext && (pp = pp.recorded)
    pp[key] = entry
    nothing
end

function _sb_frozen_preproc_entry(data, key::Symbol, kind::Symbol, raw_ref)
    ctx = get(data, _SB_PREPROC_KEY, nothing)
    ctx isa _SBPreprocContext || return nothing
    entry = get(ctx.frozen, key, nothing)
    isnothing(entry) && return nothing
    (entry.kind === kind && isequal(entry.raw_ref, raw_ref)) || error(
        "sbimpl: resample replay: fitted preprocessing record `$key` no longer " *
        "matches the re-emitted `$kind` term")
    entry
end

function _sb_record_group_index!(data, idx_key::Symbol, n_groups_key::Symbol,
                                 group::Symbol, raw::AbstractVector)
    levels = _sb_fit_levels(raw)
    _sb_record_preproc!(data, idx_key, PreprocEntry(
        :group_index, (; levels, n_groups_key), group, true))
end

# Model-shape data that do not depend on dataframe rows (for example the
# number of columns in a shared `|ID|` bucket and its per-formula column
# selectors) must survive replay verbatim.  Recording them explicitly keeps
# `reprocess` fail-closed for every other unexplained derived datum.
_sb_record_static!(data, key::Symbol) =
    _sb_record_preproc!(data, key, PreprocEntry(:static, deepcopy(data[key]), nothing, true))

# Re-materialise a column-node tree (NamedColumn / ExprColumn) against a fresh
# DataFrame `df`, mirroring `_sb_materialize_vec` but resolving raw-data leaves
# from `df` instead of the BRMI's training-bound DataColumns. Used by `reprocess`
# for zscale/center/standardize inners and the protect/implicit-fn fallback.
_sb_rematerialize_vec(x::Number, _df) = x
_sb_rematerialize_vec(x::NamedColumn, df) = _sb_df_column(df, name(x))
_sb_rematerialize_vec(x::ExprColumn, df) =
    broadcast(getf(x), map(a -> _sb_rematerialize_vec(a, df), getargs(x))...)
_sb_rematerialize_vec(x, _df) = error(
    "sbimpl: reprocess: cannot re-materialise $(typeof(x)) against the new DataFrame")

# Fetch a raw column from a new df via the BRM `Data` wrapper (the same accessor
# the `@brm` builder uses), unwrapped to a plain vector. Errors if absent.
const _sb_df_column = _brm_df_column

function _sb_gp_axes_from_df(df, raw_ref, label::Symbol)
    names = raw_ref isa Symbol ? (raw_ref,) : Tuple(raw_ref)
    axes = ntuple(length(names)) do j
        v = collect(Float64, _sb_df_column(df, names[j]))
        isempty(v) && error("sbimpl: reprocess: `$label($(names[j]))` cannot use an empty axis")
        all(isfinite, v) || error(
            "sbimpl: reprocess: `$label($(names[j]))` requires finite values")
        v
    end
    n = length(first(axes))
    all(v -> length(v) == n, axes) || error(
        "sbimpl: reprocess: `$label(x...)` axes must have equal lengths (got $(length.(axes)))")
    axes
end

"""
    SBBRMI(brmi::BRMI; mod=@__MODULE__, cv_groups=Set{Symbol}(),
           centered_groups=Set{Symbol}(), held_out=()) -> SBBRMI

StanBlocks backend: walks `brmi`, emits a `StanBlocks.SlicModel`, and
materialises the data dict. Pass `mod` if you're constructing the model
from a module other than `BayesianRegressionModels` so SLIC's symbol
resolver finds your locally-defined submodels.

`cv_groups` is an opt-in set of grouping-factor names (e.g. `[:subject]`)
whose per-group random effect should be emitted with **cv-contagious
sizing** -- the std-normal draw is sized from `maximum(<g>_idx)` instead of
the standalone data scalar `n_<g>`, so marking the group index with
`maybecv(:<g>_idx)` at trace time flips that RE to a generated-quantities
population re-draw (leave-all-out / out-of-sample). This is used when
generating a CV model artifact; the default (empty `cv_groups`) sizes the
same submodel from `n_<g>` and leaves the RE a fitted parameter. There are no
separate `_cv` submodels — the size expression passed at the call site is the
entire difference. Plain `(… | g)` ranefs and cross-formula `(… |ID| g)`
buckets are both supported; stratified `gr(g, by=b)` errors if opted-in.

`centered_groups` is an opt-in set of grouping-factor names whose per-group
random effect should be emitted in the **centered** parameterization -- the
per-group effect itself is the sampled parameter, with the covariance as its
prior (`bc ~ multi_normal_cholesky(0, diag_pre_multiply(tau, L))`), instead of
the default non-centered standardised draw plus downstream scaling. Centered
is the better geometry when the per-group likelihood is strong (dense repeated
measurement per level, e.g. many PK samples per subject); non-centered stays
the default, and a group not named here is untouched by this kwarg. Plain
`(… | g)` ranefs and `(… |ID| g)` buckets are supported; stratified
`gr(g, by=b)` is not.

The two sets are **mutually exclusive per group**: a centered block's sampled
parameter is the per-group effect itself, and no available spelling of that
can carry a cv taint in its size, so `SBBRMI` rejects a group named in both
rather than silently emitting an in-sample block. Note also that centered and
non-centered emissions use different unconstrained coordinates, so fitted
draws are not interchangeable between them.

`held_out` names one response, a collection of responses, or `:all`. Each
named observation is emitted through StanBlocks' cv activity analysis: its
likelihood is removed while its predictive draw remains in generated
quantities. Other likelihoods remain active, so `held_out=:qt_y` fits the rest
of a joint model while drawing QT-only parameters from their priors;
`held_out=:all` produces the prior-predictive model. Names resolve against both
top-level responses and data-backed observations inside `kernel(...)` cells.

Formula statements `sd(:, ID) ~ Exponential(scale)` and
`cor(:, ID) ~ LKJCholesky(K, eta)` configure a shared `|ID|` block.
An SD statement can instead select one emitted margin with
`sd(predictor, ID, coefficient)`, or write `sd(predictor, ID)` when that
predictor contributes exactly one margin. See [`ranefcoefnames`](@ref)
for the authoritative ordered addresses. Omitted statements retain historical
defaults; Julia's Exponential scale is converted to Stan's rate.

Use [`stan_code`](@ref) to extract the transpiled Stan source. For
sampling, load `StanLogDensityProblems` + `BridgeStan` and wrap the
emitted `SlicModel` in a `StanProblem`.

```julia
brmi  = @brm df (y ~ 1 + a + (1|g))
sbbrmi = SBBRMI(brmi)
src   = stan_code(sbbrmi)
```
"""
struct SBBRMI{P<:BRMI, M, D<:AbstractDict, PP<:AbstractDict, HO<:AbstractSet{Symbol}}
    parent::P
    model::M
    data::D
    preproc::PP
    held_out::HO
end

# Preserve the historical positional constructor used by downstream code that
# rebuilds an SBBRMI around the same emitted body/data. Public hold-out state is
# attached by the keyword constructor below; an explicit four-argument rebuild
# retains the historical unmarked metadata contract.
SBBRMI(parent::BRMI, model, data::AbstractDict, preproc::AbstractDict) =
    SBBRMI(parent, model, data, preproc, Set{Symbol}())

"""
    parent(sb::SBBRMI) -> BRMI

Return the parsed [`BRMI`](@ref) that `sb` wraps. This is the public, stable
accessor for an `SBBRMI`'s underlying model; prefer it over the `sb.parent`
field, which is an implementation detail. Mirrors [`parent(::TuringBRMI)`](@ref).

The introspection accessors [`structure_of`](@ref) and [`priors_of`](@ref) also
accept an `SBBRMI` directly (unwrapping through this accessor), so a consumer
holding only the emitted model never has to reach for the wrapped `BRMI`.
"""
Base.parent(sb::SBBRMI) = sb.parent

# Accept the emitted `SBBRMI` wrapper directly, so a consumer holding only the
# emitted model (and not the bare `BRMI`) can split it into prior / model views
# without reaching into internals. Both unwrap through the `parent` accessor.
structure_of(sb::SBBRMI) = structure_of(parent(sb))
priors_of(sb::SBBRMI)    = priors_of(parent(sb))

# Resolve formula-level `effect(...) ~ Normal(...)` statements against the
# authoritative population-column labels before emission. The returned vector
# for each LP is aligned 1:1 with `popcoefnames`; `nothing` means retain the
# default Normal(0, 1) for that column.
#
# A `~` statement whose RHS is a distribution call is a SCALAR PARAMETER PRIOR
# (`log_F_bottle ~ Normal(0, 0.5)`) that `_sb_emit_prior!` claims at emission
# time -- it is NOT a population formula. `linear_predictors` reports it all the
# same (its LHS is a non-data `NamedColumn`), so effect-prior resolution has to
# recognise the shape itself: `popcoefnames` has no `beta_pop` columns to name
# there, and asking it anyway either throws or mislabels the distribution call
# as a population column.
function _sb_is_prior_declaration(brmi::BRMI, lp::Symbol)
    op = linear_predictor_op(brmi, lp)
    isnothing(op) && return false
    _, rhs = getargs(op, 2)
    rhs_e = _as_expr_column(rhs)
    isnothing(rhs_e) && return false
    f = getf(rhs_e)
    f === Horseshoe || f === LKJCovarianceFactor ||
        !isnothing(_as_distribution_type(f))
end

# Names of the predictors an `effect(...)` address may legitimately reach.
_sb_effect_available_predictors(lp_names, labels_of) =
    sort!(Symbol[lp for lp in lp_names if !isnothing(labels_of(lp))])

# Trailing hint for a `:`-predictor miss: a predictor whose labels could not be
# resolved was skipped rather than considered, so say so instead of leaving
# "matches no population coefficient" looking exhaustive. The explicit
# `effect(linear_predictor, coefficient)` form reports the underlying reason in
# full, so keep this note bounded to the names.
_sb_effect_unresolved_note(unresolved) =
    isempty(unresolved) ? "" :
    " Predictor(s) " *
    join(("`$lp`" for lp in sort!(collect(keys(unresolved)))), ", ") *
    " were skipped because `popcoefnames` cannot name their columns; address " *
    "such a coefficient explicitly with `effect(linear_predictor, coefficient)` " *
    "to see why."

function _sb_effect_prior_overrides(brmi::BRMI)
    specs = effect_priors(brmi)
    isempty(specs) && return Dict{Symbol,Any}()

    lp_names = Symbol[x.name for x in linear_predictors(brmi)]
    # LAZY, memoised label resolution: `popcoefnames` is only ever asked about a
    # predictor an `effect(...)` statement actually reaches, and a predictor it
    # cannot name is SKIPPED rather than fatal. Naming every `linear_predictors`
    # entry eagerly made one unrelated scalar prior (`log_F_bottle ~ Normal(0,
    # 0.5)`) fatal for the whole model as soon as any effect prior existed.
    resolved = Dict{Symbol,Union{Vector{Symbol},Nothing}}()
    unresolved = Dict{Symbol,String}()
    labels_of(lp::Symbol) = get!(resolved, lp) do
        # Not a population formula at all -- there is nothing to name.
        _sb_is_prior_declaration(brmi, lp) && return nothing
        try
            popcoefnames(brmi, lp)
        catch err
            # `popcoefnames` covers the standard fixed-effect surface and errors
            # on shapes that need full model context (`hsgp(x, by=g)`, ...).
            # That is no reason to reject an `effect(...)` addressed elsewhere;
            # keep the diagnostic for the messages that do need it.
            unresolved[lp] = sprint(showerror, err)
            nothing
        end
    end

    # Categorical addresses are resolved from the SAME lazy/memoised discipline:
    # a predictor is asked for its `cat_<lp>_<c>` blocks only when an `effect(...)`
    # statement could reach it, and a shape that cannot be walked is skipped
    # rather than made fatal for the whole model.
    resolved_cat = Dict{Symbol,Dict{Symbol,Symbol}}()
    cat_map_of(lp::Symbol) = get!(resolved_cat, lp) do
        _sb_is_prior_declaration(brmi, lp) && return Dict{Symbol,Symbol}()
        try
            _sb_cat_address_map(brmi, lp)
        catch
            Dict{Symbol,Symbol}()
        end
    end

    # Every slot is a name or `:`, and `:` means THE DEFAULT: a broader
    # statement is the base layer that a more specific one overrides. So a cell
    # carries the winning expression AND the specificity that won it, and
    # assignment is a comparison rather than a first-writer-wins store.
    # Specificity counts CONCRETE SLOTS, not how many parameters an address
    # happens to reach -- `effect(:, weight)` and `effect(mu, :)` are equally
    # specific and collide on `mu`'s `weight` column, which is the tie error.
    pop_overrides = Dict{Symbol,Vector{Any}}()
    cat_overrides = Dict{Symbol,Dict{Symbol,Any}}()
    _spelling(spec) = _brm_effect_spelling(spec)
    # `slot` is a 0-argument getter / 1-argument setter pair over whichever
    # container owns the cell, so pop columns and categorical blocks share one
    # precedence rule instead of two drifting copies.
    _claim!(get_cell, set_cell!, spec, what) =
        _brm_claim_effect_prior!(get_cell, set_cell!, spec, what; prefix="sbimpl")

    for spec in specs
        _brm_validate_population_effect_spec(spec; prefix="sbimpl")

        all_predictors = spec.predictor === _EFFECT_COLON
        all_coefficients = spec.coefficient === _EFFECT_COLON

        if all_predictors
            # `:` in the predictor slot fans out over every predictor this
            # address can legitimately reach -- tolerantly, skipping the
            # unnameable, since a default layer must not be made fatal by an
            # unrelated scalar prior declaration.
            targets = Symbol[lp for lp in lp_names
                             if all_coefficients ?
                                (!isnothing(labels_of(lp)) || !isempty(cat_map_of(lp))) :
                                (spec.coefficient in something(labels_of(lp), Symbol[]) ||
                                 haskey(cat_map_of(lp), spec.coefficient))]
            isempty(targets) && error(
                "sbimpl: `$(_spelling(spec))` matches no population coefficient " *
                "or categorical contrast block in any linear predictor. Inspect " *
                "`popcoefnames(brmi, lp)` for valid labels." *
                _sb_effect_unresolved_note(unresolved))
        else
            targets = Symbol[spec.predictor]
        end

        for target in targets
            cat_map = cat_map_of(target)
            labels = labels_of(target)

            if all_coefficients
                # The default layer for this predictor: every `beta_pop` column
                # and every categorical contrast block it owns.
                if !isnothing(labels) && !isempty(labels)
                    cells = get!(pop_overrides, target) do
                        Any[nothing for _ in labels]
                    end
                    for idx in eachindex(labels)
                        _claim!(() -> cells[idx], v -> (cells[idx] = v), spec,
                                "`$target`'s `$(labels[idx])` column")
                    end
                end
                for emitted in values(cat_map)
                    target_cat = get!(cat_overrides, target) do
                        Dict{Symbol,Any}()
                    end
                    _claim!(() -> get(target_cat, emitted, nothing),
                            v -> (target_cat[emitted] = v), spec,
                            "`$target`'s `$emitted` contrast block")
                end
                continue
            end

            # A categorical / integer-coded predictor owns its own K-1 contrast
            # block (`cat_<lp>_<c>_beta`) instead of `beta_pop` columns, so it is
            # structurally absent from `popcoefnames`. Resolve it FIRST: nothing
            # in `popcoefnames` can ever collide with a name only
            # `_sb_cat_address_map` knows, and this keeps the block reachable
            # even for an intercept-less `mu ~ 0 + factor(g)` whose `labels` are
            # legitimately empty.
            if haskey(cat_map, spec.coefficient)
                emitted = cat_map[spec.coefficient]
                target_cat = get!(cat_overrides, target) do
                    Dict{Symbol,Any}()
                end
                _claim!(() -> get(target_cat, emitted, nothing),
                        v -> (target_cat[emitted] = v), spec,
                        "`$target`'s `$emitted` contrast block")
                continue
            end

            if isnothing(labels)
                # An EXPLICITLY addressed target that cannot be named is a
                # genuine error -- just not one an unrelated predictor may raise
                # on its behalf.
                haskey(unresolved, target) && error(
                    "sbimpl: cannot resolve population-effect labels for predictor ",
                    "`$target` while applying `$(_spelling(spec))`: ",
                    unresolved[target])
                error("sbimpl: `$(_spelling(spec))` names no linear " *
                      "predictor with population coefficients. Available predictors: " *
                      "$(join(_sb_effect_available_predictors(lp_names, labels_of), ", ")).")
            end
            idx = findfirst(==(spec.coefficient), labels)
            isnothing(idx) && error(
                "sbimpl: `$(spec.coefficient)` is not a population coefficient of " *
                "`$target`. Available labels: $(join(labels, ", "))." *
                _sb_effect_cat_note(cat_map))
            cells = get!(pop_overrides, target) do
                Any[nothing for _ in labels]
            end
            _claim!(() -> cells[idx], v -> (cells[idx] = v), spec,
                    "`$target`'s `$(spec.coefficient)` column")
        end
    end

    # One entry per predictor that any `effect(...)` statement reached. `pop` is
    # `nothing` when only categorical blocks were addressed, so an untouched
    # `beta_pop` keeps the plain `popefs` emission byte for byte. The winning
    # expression is unwrapped here so every downstream consumer keeps seeing a
    # bare expression-or-`nothing`, unaware of the precedence bookkeeping.
    _unwrap(cell) = isnothing(cell) ? nothing : cell.expression
    out = Dict{Symbol,Any}()
    for lp in union(keys(pop_overrides), keys(cat_overrides))
        pop = get(pop_overrides, lp, nothing)
        cat = get(cat_overrides, lp, Dict{Symbol,Any}())
        out[lp] = (; pop = isnothing(pop) ? nothing : Any[_unwrap(c) for c in pop],
                     cat = Dict{Symbol,Any}(k => _unwrap(v) for (k, v) in cat))
    end
    out
end

# Trailing hint for a coefficient that IS in this predictor, just not as a
# `beta_pop` column. `cat_mu_g` (the emitted Stan parameter prefix) is the
# natural wrong guess once a user has read the transpiled code, so name the
# address that does work rather than leaving "Available labels" looking
# exhaustive.
function _sb_effect_cat_note(cat_map)
    isempty(cat_map) && return ""
    " Categorical contrast block(s) " *
    join(("`$a`" for a in sort!(collect(keys(cat_map)))), ", ") *
    " own their own `cat_<lp>_<c>_beta` parameters rather than `beta_pop` columns; " *
    "address one by that name to set its contrast prior."
end

# Readers for the `_sb_effect_prior_overrides` value shape. Kept as functions so
# every consumer of the threaded `effect_overrides` dict agrees on it, and so a
# default/empty dict (`Dict{Symbol,Any}()`) reads back as "no override" without
# each call site knowing the entry is a `(; pop, cat)` NamedTuple.
function _sb_pop_effect_overrides(effect_overrides, lp::Symbol)
    e = get(effect_overrides, lp, nothing)
    isnothing(e) ? nothing : e.pop
end
function _sb_cat_effect_overrides(effect_overrides, lp::Symbol)
    e = get(effect_overrides, lp, nothing)
    isnothing(e) ? Dict{Symbol,Any}() : e.cat
end
# `term` is absent from the record when no term-parameter statement exists, so
# the population/categorical assembly stays untouched and every consumer that
# never asks for it keeps its old value shape.
function _sb_term_effect_overrides(effect_overrides, lp::Symbol)
    e = get(effect_overrides, lp, nothing)
    (isnothing(e) || !hasproperty(e, :term)) ? Dict{Symbol,Any}() : e.term
end

function _sb_effect_normal_args(rhs::ExprColumn)
    map(_sb_effect_prior_arg,
        _brm_normal_effect_args(rhs; prefix="sbimpl"))
end

_sb_effect_prior_arg(x) = _sb_prior_arg(x)
function _sb_effect_prior_arg(x::ExprColumn)
    args = map(_sb_effect_prior_arg, getargs(x))
    # Formula parsing captures literal arithmetic as ExprColumns. Evaluate an
    # all-numeric effect-prior hyperparameter in Julia so `log(1 / 8)` retains
    # Julia's floating division semantics instead of becoming Stan integer
    # division. Symbol-bearing expressions stay as Stan expressions.
    if all(a -> a isa Real, args)
        value = try
            getf(x)(args...)
        catch
            nothing
        end
        value isa Real && return value
    end
    Expr(:call, getf(x), args...)
end

# StanBlocks' data phase (`forward!(::SlicModel)`) types EVERY key in the data
# dict via `stan_type`, so every value handed to the `SlicModel` must be Stan
# data. The shared `_brm_collect_data!` (backend_plan.jl) records the raw of
# every formula-referenced column as reprocess provenance — including raw
# `CategoricalVector`/string predictor columns whose Stan representation is a
# derived integer-code column (`<col>_idx`), NOT the raw column itself. The raw
# is never referenced by the emitted model, so it must be dropped before the
# SlicModel; otherwise `stan_type` errors on it at ALL K (not just K=1). Numeric
# arrays (including ragged vectors-of-vectors), scalars, functions, and tuples
# are all valid data and kept.
_sb_is_stan_data_value(::Any) = true
_sb_is_stan_data_value(v::AbstractArray) =
    eltype(v) <: Real || eltype(v) <: AbstractArray

# Stan reserved keywords. BRM names each emitted Stan `data` variable verbatim
# after its DataFrame column (`keys(data)` below), so a column named `lower`,
# `upper`, `real`, `int`, ... would be emitted as `vector[lower_n] lower;` — a
# program `transpiles()` accepts but `stanc` rejects with "Ill-formed
# identifier ... reserved keyword". The set is exactly the identifiers this
# `stanc` rejects, enumerated against the pinned compiler (snag
# `reserved-keyword`). It is deliberately NOT broadened to the full C++ keyword
# list: `stanc` accepts `double`/`float`/`class`/... as identifiers, so
# rejecting those here would break currently-compiling models.
const _SB_STAN_RESERVED_IDENTIFIERS = Set{Symbol}((
    :functions, :data, :parameters, :model, :transformed, :generated, :quantities,
    :return, :if, :else, :while, :for, :in, :break, :continue, :profile, :print,
    :reject, :target, :int, :real, :complex, :vector, :row_vector, :matrix, :array,
    :tuple, :void, :ordered, :positive_ordered, :simplex, :unit_vector,
    :sum_to_zero_vector, :cholesky_factor_corr, :cholesky_factor_cov, :corr_matrix,
    :cov_matrix, :complex_vector, :complex_row_vector, :complex_matrix, :lower,
    :upper, :offset, :multiplier, :var, :typedef, :struct, :auto, :export, :extern,
    :static,
    # `:true`/`:false` are the Bool literals in Julia, not Symbols — quote them.
    Symbol("true"), Symbol("false"),
))

SBBRMI(brmi::BRMI; mod::Module=@__MODULE__, cv_groups=Set{Symbol}(),
       centered_groups=Set{Symbol}(), held_out=(), _frozen_preproc=nothing) = begin
    cv_groups = cv_groups isa Set ? cv_groups : Set{Symbol}(cv_groups)
    centered_groups = centered_groups isa Set ? centered_groups : Set{Symbol}(centered_groups)
    both = intersect(cv_groups, centered_groups)
    isempty(both) || error(
        "sbimpl: group(s) $(join(sort!(collect(both)), ", ")) named in BOTH ",
        "`cv_groups` and `centered_groups`. A centered block's sampled ",
        "parameter is the per-group effect itself, whose size cannot carry a ",
        "cv taint (StanBlocks: cv does not reach a plate-internal fresh ",
        "parameter, and typed-LHS derives cv from RHS call-args rather than ",
        "the declared size). Emit the CV artifact non-centered, or drop the ",
        "group from `cv_groups`.")
    stmts = Any[]
    data = Dict{Symbol,Any}()
    # Side-channel: transform emitters record their fit-time constant + raw
    # reference here (via `_sb_record_preproc!`); popped before `SlicModel` below
    # so it never reaches Stan's data dict. See `PreprocEntry` / `reprocess`.
    data[_SB_PREPROC_KEY] = isnothing(_frozen_preproc) ?
        Dict{Symbol,PreprocEntry}() :
        _SBPreprocContext(Dict{Symbol,PreprocEntry}(), _frozen_preproc)
    _sb_validate_covariance_factor_names(brmi)
    # The shared, backend-neutral pass owns raw-data materialisation,
    # likelihood-decorator claims, and target -> observation row axes. Its
    # input `data` already carries the Stan preprocessing side-channel, which
    # the generic collector leaves untouched.
    context = _brm_backend_context(brmi; data)
    prepass = context.prepass
    effect_overrides = _sb_prior_overrides(brmi)
    # Prepass 2: collect brms-style `|ID|` ranef buckets across all sub-formulas,
    # emit one shared ranef_correlated_draws per bucket, and build a lookup
    # `(brmi_key, (id_sym, group_key)) => (bucket_name, col_range, idx_name, suffix)`
    # for per-sub-formula emission below.
    id_buckets = _sb_collect_id_buckets(brmi)
    ranef_effect_overrides = _sb_ranef_effect_overrides(brmi, id_buckets)
    ranef_r2d2_overrides = _sb_ranef_r2d2_overrides(brmi, id_buckets)
    # A sampled observation/reference scale must be declared before the bucket
    # prepass consumes it.  Opted-in R2D2 models alone move those scalar priors
    # forward; the ordinary statement order stays byte-identical.
    reference_prior_keys = Set{Symbol}(
        ref for decomposition in values(ranef_r2d2_overrides)
            for ref in decomposition.references)
    early_prior_keys = Set{Symbol}()
    if !isempty(reference_prior_keys)
        operation_keys = collect(keys(brmi.operations))
        positions = Dict(key => index for (index, key) in pairs(operation_keys))
        missing = setdiff(reference_prior_keys, Set(operation_keys))
        isempty(missing) || error(
            "sbimpl: R2D2 reference scale(s) $(join(sort!(collect(missing)), ", ")) " *
            "are not model declarations")
        last_reference = maximum(positions[key] for key in reference_prior_keys)
        # Move the scalar-prior prefix as a unit, in formula order. This keeps
        # a reference scale whose prior depends on an earlier sampled scalar
        # valid after both declarations move ahead of the ranef prepass.
        for key in operation_keys[1:last_reference]
            nc = _as_named_column(brmi.operations[key])
            isnothing(nc) && continue
            op = _as_expr_column(parent(nc))
            (!isnothing(op) && getf(op) === (~)) || continue
            lhs, rhs_raw = getargs(op, 2)
            lhs_nc = _as_named_column(lhs)
            rhs = _as_expr_column(rhs_raw)
            (!isnothing(lhs_nc) && name(lhs_nc) === key &&
             parent(lhs_nc) isa MissingColumn && !isnothing(rhs) &&
             _sb_is_scalar_prior_family(getf(rhs))) || continue
            push!(early_prior_keys, key)
        end
        unresolved = setdiff(reference_prior_keys, early_prior_keys)
        isempty(unresolved) || error(
            "sbimpl: R2D2 reference scale(s) " *
            "$(join(sort!(collect(unresolved)), ", ")) must be backed by " *
            "supported sampled scalar priors")
        for key in operation_keys
            key in early_prior_keys || continue
            nc = _as_named_column(brmi.operations[key])
            _sb_emit!(stmts, data, key, parent(nc))
        end
    end
    # Prepass 2a: whole-predictor R2D2 decompositions. Resolved and emitted
    # BEFORE the bucket statement below, which consumes the derived residual
    # scales. Empty unless the formula carries an `effect(..., :) ~ r2d2(...)`
    # statement, so every other model's emission is untouched.
    r2d2_overrides = _sb_r2d2_overrides(brmi, id_buckets, effect_overrides)
    r2d2_names = _sb_emit_r2d2_params!(stmts, data, r2d2_overrides)
    id_lookup = _sb_emit_id_buckets!(stmts, data, id_buckets;
        cv_groups, centered_groups, ranef_effect_overrides, r2d2_names,
        ranef_r2d2_overrides)
    # Prepass 2.5: group-block terms. For each `mu ~ f(...)` where f has a
    # _sb_term_group_block declaration, allocate one ranef_correlated_draws
    # block per (f, group-column) pair. The lookup is threaded into _sb_emit!
    # so _sb_sampling_backed! can route declaring terms to their emit hook
    # with the pre-allocated block name and group-index name in hand.
    gb_terms = _sb_collect_group_block_terms(brmi)
    group_block_lookup = _sb_emit_group_blocks!(stmts, data, gb_terms)
    # Prepass 3: target -> observation-source map. Lets purely-intercept
    # linear predictors (`loc ~ 1`, `log(y_scale) ~ 1`) borrow N from the
    # observed `~` target that consumes them, instead of the hash-order
    # `_sb_any_data_symbol(data)` fallback. The shared context discovers this
    # map before either backend emits or executes anything.
    target_obs = context.target_obs
    for (key, op) in pairs(brmi.operations)
        key in early_prior_keys && continue
        nc = _as_named_column(op)
        isnothing(nc) && error("sbimpl: top-level op `$key` is not a NamedColumn")
        obs_n = get(target_obs, key, nothing)
        _sb_emit!(stmts, data, key, parent(nc); id_lookup, obs_n, cv_groups,
                  centered_groups, group_block_lookup, effect_overrides,
                  r2d2=(; overrides=r2d2_overrides, names=r2d2_names))
    end
    # Post-pass: fuse pure-population Gaussian likelihoods into Stan's
    # `normal_id_glm_lpdf`. Whole-model, because its decisive guard is
    # "this linear predictor is read by exactly one likelihood". Every
    # non-matching model keeps its statements byte for byte.
    _sb_fuse_normal_id_glm!(stmts, data)
    # Pop the preproc side-channel BEFORE building the SlicModel so it never
    # pollutes Stan's data dict.
    preproc_ctx = pop!(data, _SB_PREPROC_KEY, Dict{Symbol,PreprocEntry}())
    preproc = preproc_ctx isa _SBPreprocContext ? preproc_ctx.recorded : preproc_ctx
    # Drop leaked non-Stan data (raw `CategoricalVector`/string predictor columns
    # collected as reprocess provenance). See `_sb_is_stan_data_value`. This also
    # keeps `reprocess`'s pass-through (step 2) from re-leaking the raw column
    # into `new_data`, since it only passes through keys present in `sb.data`.
    for k in collect(keys(data))
        _sb_is_stan_data_value(data[k]) || delete!(data, k)
    end
    # `keys(data)` is now exactly the set of Stan `data` identifiers this model
    # emits. Reject any that collide with a Stan reserved keyword here — with an
    # actionable BRM-level error — instead of returning invalid Stan that only
    # `stanc` catches (snag `reserved-keyword`).
    reserved_cols = sort!(Symbol[k for k in keys(data)
                                 if k in _SB_STAN_RESERVED_IDENTIFIERS])
    isempty(reserved_cols) || error(
        "sbimpl: column name(s) ", join(reserved_cols, ", "), " collide with Stan ",
        "reserved keyword(s). BRM emits each data column verbatim as a Stan `data` ",
        "identifier (e.g. `vector[", first(reserved_cols), "_n] ", first(reserved_cols),
        ";`), which `stanc` rejects with \"Ill-formed identifier\". Rename the ",
        "offending column(s) before building the model — e.g. `lower`/`upper` -> ",
        "`y_lower`/`y_upper` for interval-censored endpoints.")
    body = Expr(:block, stmts...)
    model = StanBlocks.SlicModel(body, data, mod)
    _sb_apply_held_out(SBBRMI(brmi, model, data, preproc), held_out)
end

_as_data_column(x::DataColumn) = x
_as_data_column(_) = nothing

_as_missing_column(x::MissingColumn) = x
_as_missing_column(_) = nothing

_as_symbol(s::Symbol) = s
_as_symbol(_) = nothing

_as_int_vec(v::AbstractVector{<:Integer}) = v
_as_int_vec(_) = nothing

_as_real_vec(v::AbstractVector{<:Real}) = v
_as_real_vec(_) = nothing

_as_integer(x::Integer) = x
_as_integer(_) = nothing

_as_real(x::Real) = x
_as_real(_) = nothing

_as_error_exception(e::ErrorException) = e
_as_error_exception(_) = nothing

"""
    stan_code(sb::SBBRMI) -> String

Return the transpiled Stan source generated from `sb.model`. Forwards
to `StanBlocks.stan_code`. Useful for inspecting what the sbimpl walker
emitted before compiling.
"""
stan_code(sb::SBBRMI) = StanBlocks.stan_code(sb.model)

Base.show(io::IO, sb::SBBRMI) = begin
    print(io, "SBBRMI with data keys = ", sort(collect(keys(sb.data))), "\n")
    print(io, "emitted @slic body:\n")
    print(io, sb.model.model)
end


# ---- declaration-driven generative plans -----------------------------------

"""
    GenerativeDeclaration

One emitted sampling declaration in a [`GenerativePlan`](@ref).

- `role` is `:prior` or `:observation`. A `:prior` includes structured latent
  submodels such as `popefs`, `ranef_correlated`, and `plate`, not only scalar
  distribution calls.
- `target` and `family` are the emitted SLIC binding and RHS head.
- `data_source` is the original data key for an observation (including a
  plate-local alias such as `kernel_y => dv`), otherwise `nothing`.
- `draw` is the canonical binding a generative executor should use for an
  observation, otherwise `nothing`. For top-level observations this matches
  StanBlocks' current posterior-predictive `*_gen` name. Nested plate
  observations are inventoried too, but consumers must discover their actual
  executable twins through `stan_descriptor`: the emitted names are based on
  `data_source`, not this plate-local target. A ragged base has a flat draw and
  a group-aggregate likelihood, both carrying the observed group boundaries.
- `context` names enclosing plate results, outermost first.
- `expression` is an exact snapshot of the emitted `~` expression.

The remaining fields decompose that RHS call so an executor never has to parse
the snapshot itself:

- `arguments` are the RHS call's positional arguments, in order (the rate of
  `exponential(1)`, the eta of `lkj_corr_cholesky(1.)`).
- `keywords` is a `NamedTuple` of every RHS keyword argument, verbatim.
- `annotation` is the LHS type annotation (`:(vector[3])` for
  `kernel_z::vector[3] ~ ...`), or `nothing` for a bare LHS.
- `dimension` normalises the *two* ways a declaration can spell its size into
  one tuple: the `m`/`n`/`o` (or `size`) keyword that StanBlocks' `autotype`
  reads, and the LHS `::T[s...]` annotation, which wins when both are present.
  `()` means the declaration spells no size of its own — it is a scalar, or its
  extent comes from the data or from a submodel's internals (`popefs`,
  `ranef_correlated`) rather than from this declaration.
- `constraints` is the subset of `keywords` StanBlocks folds into the declared
  type: `lower`, `upper`, `offset`, `multiplier`. **This is the field that makes
  a half-normal visible**: `std_normal(; n=3, lower=0.)` and `std_normal(; n=3)`
  differ only here.

Entries of `arguments`, `dimension`, and `constraints` are the emitted
expressions, not evaluated values: an entry may be a literal, or a `Symbol` that
is a key of the plan's `data`, or a larger `Expr`. Resolve symbolic sizes
against `plan.data`.

`constraints` reports only the constraints spelled **on this declaration**.
Families carry their own implied support (`exponential` is positive,
`beta` lives on `[0, 1]`) — that is a property of `family`, held in StanBlocks'
`autokwargs` table, and BRM deliberately does not duplicate it here.

The declaration is intentionally backend-level: it describes what BRM really
emitted after formula terms introduced their latent blocks, rather than a
second model-specific interpretation of the `@brm` source.
"""
struct GenerativeDeclaration
    role::Symbol
    target::Symbol
    family::Any
    data_source::Union{Nothing,Symbol}
    draw::Union{Nothing,Symbol}
    context::Tuple{Vararg{Symbol}}
    expression::Expr
    arguments::Tuple
    keywords::NamedTuple
    annotation::Union{Nothing,Symbol,Expr}
    dimension::Tuple
    constraints::NamedTuple
end

"""
    GenerativePlan

An introspectable, replayable snapshot of an `SBBRMI`'s actual emitted
declarations. `model`, `data`, and `preproc` are copied together at plan
construction, while `declarations` inventories every emitted sampling site,
including priors introduced inside `kernel(...)` plates and every observation
in a multi-output model.

Construct with [`generative_plan`](@ref). [`stan_code`](@ref) accepts a plan,
and [`reprocess`](@ref) preserves the existing SBBRMI replay contract (including
its correct-or-loud unsupported cases).
Plans constructed from a reusable `@brm` builder also retain that builder so
`generative_plan(plan, new_df)` can rebuild the same declarations for genuinely
new groups.

This is a declaration plan, not an RNG executor: it does not claim that a
component-wise consumer draw is prior-predictive. Its purpose is to expose the
one authoritative program and provenance an executor must consume.
"""
struct GenerativePlan{P,M,D,PP,DS,B,CV,HO}
    parent::P
    model::M
    data::D
    preproc::PP
    declarations::DS
    builder::B
    cv_groups::CV
    held_out::HO
end

# Keep the pre-held-out positional shape source-compatible. New plans derived
# through the public constructors always carry the explicit selection field.
GenerativePlan(parent, model, data, preproc, declarations, builder, cv_groups) =
    GenerativePlan(parent, model, data, preproc, declarations, builder,
                   cv_groups, Set{Symbol}())

_sb_plan_lhs_name(x::Symbol) = x
_sb_plan_lhs_name(x::Expr) =
    (x.head === :(::) || x.head === :ref) && !isempty(x.args) ?
        _sb_plan_lhs_name(x.args[1]) : nothing
_sb_plan_lhs_name(_) = nothing

_sb_plan_family(x::Expr) = begin
    if x.head === :call && !isempty(x.args)
        x.args[1]
    elseif x.head === :do && !isempty(x.args)
        _sb_plan_family(x.args[1])
    else
        x.head
    end
end
_sb_plan_family(x) = x

_sb_plan_generated(context, target) =
    Symbol(join((context..., target), "_"), "_gen")

# A configured term prior is spliced into the emitted body as a `SlicModel`
# value.  Its `mod` field is the stable namespace where the submodel was
# defined, not mutable model state, and Julia deliberately refuses to
# `deepcopy` a `Module`.  Copy the model payload while retaining that namespace
# by identity; ordinary emitted leaves keep the historical `deepcopy` path.
_sb_plan_copy(x) = deepcopy(x)
_sb_plan_copy(x::Module) = x
_sb_plan_copy(x::QuoteNode) = QuoteNode(_sb_plan_copy(x.value))
_sb_plan_copy(x::Expr) = Expr(x.head, map(_sb_plan_copy, x.args)...)
_sb_plan_copy(x::StanBlocks.SlicModel) = StanBlocks.SlicModel(
    _sb_plan_copy(x.model), deepcopy(x.data), x.mod)

# The LHS type annotation, if any: `z::vector[3] ~ rhs` -> `:(vector[3])`.
_sb_plan_annotation(x::Expr) =
    x.head === :(::) && length(x.args) == 2 ? _sb_plan_copy(x.args[2]) : nothing
_sb_plan_annotation(_) = nothing

# Split the emitted RHS call into positional args and keyword args. `do`-block
# RHSs (`plate(...) do ...`) split on the underlying call, matching `family`.
_sb_plan_as_kw(x::Expr) = x.head === :kw && length(x.args) == 2 ?
    (x.args[1]::Symbol => _sb_plan_copy(x.args[2])) : nothing
_sb_plan_as_kw(_) = nothing

_sb_plan_call_parts(x) = ((), NamedTuple())
function _sb_plan_call_parts(x::Expr)
    x.head === :do && !isempty(x.args) && return _sb_plan_call_parts(x.args[1])
    x.head === :call && !isempty(x.args) || return ((), NamedTuple())
    args, kws = Any[], Pair{Symbol,Any}[]
    for a in x.args[2:end]
        if a isa Expr && a.head === :parameters
            for p in a.args
                kw = _sb_plan_as_kw(p)
                isnothing(kw) || push!(kws, kw)
            end
            continue
        end
        kw = _sb_plan_as_kw(a)
        isnothing(kw) ? push!(args, _sb_plan_copy(a)) : push!(kws, kw)
    end
    (Tuple(args), (; kws...))
end

# StanBlocks' `autotype` reads sizes from `m`/`n`/`o` in that order, else `size`
# (functions.jl `autotype`), and folds these four keywords into the declared
# type's constraints (forward.jl typed-LHS path).
_sb_plan_size_kwargs() = (:m, :n, :o)
_sb_plan_constraint_kwargs() = (:lower, :upper, :offset, :multiplier)

# One tuple for both spellings of a declared size. The LHS `::T[s...]`
# annotation is the declared type's own size, so it wins over the keyword form.
function _sb_plan_dimension(annotation, kwargs)
    if annotation isa Expr && annotation.head === :ref && length(annotation.args) >= 2
        return Tuple(_sb_plan_copy.(annotation.args[2:end]))
    end
    sizes = Any[kwargs[k] for k in _sb_plan_size_kwargs() if haskey(kwargs, k)]
    isempty(sizes) || return Tuple(sizes)
    haskey(kwargs, :size) || return ()
    s = kwargs[:size]
    s isa Expr && s.head === :tuple ? Tuple(_sb_plan_copy.(s.args)) : (_sb_plan_copy(s),)
end

_sb_plan_constraints(kwargs) = (;
    (k => kwargs[k] for k in _sb_plan_constraint_kwargs() if haskey(kwargs, k))...)

_sb_plan_param_names(x::Symbol) = (x,)
_sb_plan_param_names(x::Expr) = if x.head === :tuple
    Tuple(filter(!isnothing, map(_sb_plan_lhs_name, x.args)))
else
    name = _sb_plan_lhs_name(x)
    isnothing(name) ? () : (name,)
end
_sb_plan_param_names(_) = ()

_sb_plan_plate_parts(x) = nothing
function _sb_plan_plate_parts(x::Expr)
    x.head === :do || return nothing
    length(x.args) == 2 || return nothing
    call, lambda = x.args
    call isa Expr && call.head === :call && !isempty(call.args) &&
        call.args[1] === :plate || return nothing
    lambda isa Expr && lambda.head === :-> && length(lambda.args) == 2 ||
        return nothing
    iterables = filter(a -> !(a isa Expr && a.head === :parameters), call.args[2:end])
    params = _sb_plan_param_names(lambda.args[1])
    (; iterables=Tuple(iterables), params, body=lambda.args[2])
end

function _sb_plan_collect!(declarations, x, data_scope, context=())
    x isa Expr || return nothing
    if x.head === :block
        foreach(stmt -> _sb_plan_collect!(declarations, stmt, data_scope, context), x.args)
        return nothing
    end
    if x.head === :call && length(x.args) >= 3 && x.args[1] === :~
        target = _sb_plan_lhs_name(x.args[2])
        isnothing(target) && error(
            "generative_plan: cannot identify emitted sampling LHS `$(x.args[2])`")
        rhs = x.args[3]
        data_source = get(data_scope, target, nothing)
        role = isnothing(data_source) ? :prior : :observation
        draw = role === :observation ? _sb_plan_generated(context, target) : nothing
        annotation = _sb_plan_annotation(x.args[2])
        arguments, keywords = _sb_plan_call_parts(rhs)
        push!(declarations, GenerativeDeclaration(
            role, target, _sb_plan_family(rhs), data_source, draw,
            Tuple(context), _sb_plan_copy(x), arguments, keywords, annotation,
            _sb_plan_dimension(annotation, keywords),
            _sb_plan_constraints(keywords)))

        plate = _sb_plan_plate_parts(rhs)
        if !isnothing(plate)
            nested_scope = copy(data_scope)
            for (param, iterable) in zip(plate.params, plate.iterables)
                iterable isa Symbol || continue
                source = get(data_scope, iterable, nothing)
                isnothing(source) || (nested_scope[param] = source)
            end
            _sb_plan_collect!(declarations, plate.body, nested_scope, (context..., target))
        end
        return nothing
    end
    nothing
end

function _generative_plan(sb::SBBRMI, builder, cv_groups)
    parent = deepcopy(sb.parent)
    data = deepcopy(sb.data)
    preproc = deepcopy(sb.preproc)
    body = _sb_plan_copy(sb.model.model)
    model = StanBlocks.SlicModel(body, data, sb.model.mod)
    declarations = GenerativeDeclaration[]
    data_scope = Dict{Symbol,Union{Nothing,Symbol}}(k => k for k in keys(data))
    _sb_plan_collect!(declarations, body, data_scope)
    GenerativePlan(parent, model, data, preproc, Tuple(declarations), builder,
                   copy(cv_groups), copy(sb.held_out))
end

function _sb_held_out_request(held_out)
    (held_out === nothing || held_out === ()) &&
        return (; all=false, names=Set{Symbol}())
    held_out === :all && return (; all=true, names=Set{Symbol}())
    held_out isa AbstractString && error(
        "sbimpl: `held_out` expects a response Symbol, a collection of response " *
        "Symbols, or `:all`; got $(repr(held_out))")
    values = held_out isa Symbol ? (held_out,) : try
        collect(held_out)
    catch
        error("sbimpl: `held_out` expects a response Symbol, a collection of " *
              "response Symbols, or `:all`; got $(repr(held_out))")
    end
    all(x -> x isa Symbol, values) || error(
        "sbimpl: every `held_out` response must be a Symbol; got $(repr(values))")
    names = Set{Symbol}(values)
    :all in names && error(
        "sbimpl: use `held_out=:all` by itself; do not mix `:all` with response names")
    (; all=false, names)
end

# Resolve public response names through the emitted declaration inventory. This
# is what makes a nested `qy ~ normal(...)` inside `kernel(..., qt_y, ...)`
# addressable as `held_out=:qt_y`: the declaration records `target=:qy` and
# `data_source=:qt_y`, while StanBlocks must receive the mark on the latter.
function _sb_apply_held_out(sb::SBBRMI, held_out)
    request = _sb_held_out_request(held_out)
    !request.all && isempty(request.names) && return sb

    plan = _generative_plan(sb, nothing, Set{Symbol}())
    aliases = Dict{Symbol,Set{Symbol}}()
    sources = Set{Symbol}()
    for declaration in plan.declarations
        declaration.role === :observation || continue
        source = declaration.data_source
        isnothing(source) && error(
            "sbimpl: observation `$(declaration.target)` has no data source; " *
            "cannot apply `held_out`")
        haskey(sb.data, source) || error(
            "sbimpl: observation `$(declaration.target)` resolves to absent Stan " *
            "data key `$source`; cannot apply `held_out`")
        push!(sources, source)
        for alias in (declaration.target, source)
            push!(get!(() -> Set{Symbol}(), aliases, alias), source)
        end
    end
    isempty(sources) && error(
        "sbimpl: `held_out` was requested, but this BRMI emits no observation likelihoods")

    selected = if request.all
        sources
    else
        unknown = sort!(collect(setdiff(request.names, Set(keys(aliases)))))
        isempty(unknown) || error(
            "sbimpl: `held_out` names unknown response(s) $(unknown). Available " *
            "responses: $(sort!(collect(keys(aliases)))).")
        ambiguous = sort!(Symbol[name for name in request.names
                                 if length(aliases[name]) > 1])
        isempty(ambiguous) || error(
            "sbimpl: `held_out` alias(es) $(ambiguous) each resolve to several " *
            "response data sources. Name the dataframe response column instead.")
        reduce(union, (aliases[name] for name in request.names);
               init=Set{Symbol}())
    end

    marked = Dict{Symbol,Any}(sb.data)
    for source in selected
        marked[source] = StanBlocks.stan.maybecv(source, marked[source])
    end
    model = StanBlocks.SlicModel(sb.model.model, marked, sb.model.mod)
    SBBRMI(sb.parent, model, marked, sb.preproc, selected)
end

"""
    generative_plan(sb::SBBRMI) -> GenerativePlan
    generative_plan(builder::Function, df; mod=@__MODULE__, cv_groups=Set(), held_out=()) -> GenerativePlan
    generative_plan(plan::GenerativePlan, new_df; cv_groups=Set(), held_out=()) -> GenerativePlan

Snapshot the declarations BRM actually emitted. The inventory is derived from
`sb.model.model`, so auto-introduced population coefficients, random-effect
blocks, named linear predictors consumed by `kernel(...)`, observation
families, and multiple outputs
cannot drift from the fitted model.

Use the reusable-builder form when future schedules may contain new groups:

```julia
builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject)
    y ~ Normal(mu, sigma)
end
plan = generative_plan(builder, schedule; mod=@__MODULE__)
new_population_plan = generative_plan(plan, new_schedule)
```

The `SBBRMI` form has no reusable formula builder to apply to genuinely new
groups; use [`reprocess`](@ref) on that plan for the existing frozen-constant
replay semantics instead.
"""
generative_plan(sb::SBBRMI) = _generative_plan(sb, nothing, Set{Symbol}())

function generative_plan(builder::Function, df;
                         mod::Module=@__MODULE__, cv_groups=Set{Symbol}(),
                         held_out=())
    brmi = Base.invokelatest(builder, df)
    brmi isa BRMI || error(
        "generative_plan: builder returned $(typeof(brmi)); expected a BRMI from `@brm begin ... end`")
    cv_groups = cv_groups isa Set ? cv_groups : Set{Symbol}(cv_groups)
    _generative_plan(SBBRMI(brmi; mod, cv_groups, held_out), builder, cv_groups)
end

function generative_plan(plan::GenerativePlan, new_df;
                         cv_groups=plan.cv_groups, held_out=plan.held_out)
    isnothing(plan.builder) && error(
        "generative_plan: this plan was built from an SBBRMI and has no reusable `@brm` builder. " *
        "Construct it with `generative_plan(builder, df)` to rebuild the same declarations for new groups.")
    generative_plan(plan.builder, new_df; mod=plan.model.mod, cv_groups, held_out)
end

stan_code(plan::GenerativePlan) = StanBlocks.stan_code(plan.model)

Base.show(io::IO, plan::GenerativePlan) = begin
    nprior = count(d -> d.role === :prior, plan.declarations)
    nobs = count(d -> d.role === :observation, plan.declarations)
    print(io, "GenerativePlan with $nprior prior/latent and $nobs observation declarations")
end


# ---- reprocess / restan_data (decision nr3v8n A) ----------------------------

const _sb_df_has_column = _brm_df_has_column
const _sb_rebind_brmi = _brm_rebind_brmi

function _sb_resample_group_set(groups)
    groups === nothing && return Set{Symbol}()
    values = groups isa Symbol ? (groups,) : groups
    values isa AbstractString && error(
        "sbimpl: `resample_groups` expects a Symbol or collection of Symbols, " *
        "got $(repr(values))")
    collected = try
        collect(values)
    catch
        error("sbimpl: `resample_groups` expects a Symbol or collection of " *
              "Symbols, got $(repr(values))")
    end
    all(v -> v isa Symbol, collected) || error(
        "sbimpl: `resample_groups` expects only Symbols, got $(repr(collected))")
    Set{Symbol}(collected)
end

_sb_same_raw_ref(a, b) = isequal(a, b)
_sb_same_raw_ref(::DataColumn, ::DataColumn) = true
_sb_same_raw_ref(a::NamedColumn, b::NamedColumn) =
    name(a) === name(b) && _sb_same_raw_ref(parent(a), parent(b))
_sb_same_raw_ref(a::ExprColumn, b::ExprColumn) =
    getf(a) === getf(b) && _sb_same_raw_ref(getargs(a), getargs(b)) &&
    _sb_same_raw_ref(getkwargs(a), getkwargs(b))
_sb_same_raw_ref(a::Tuple, b::Tuple) =
    length(a) == length(b) && all(ab -> _sb_same_raw_ref(ab...), zip(a, b))
_sb_same_raw_ref(a::NamedTuple, b::NamedTuple) =
    keys(a) == keys(b) && all(
        ab -> _sb_same_raw_ref(ab...), zip(values(a), values(b)))

function _sb_resample_preproc(training, fresh, groups)
    training_keys = Set(keys(training))
    fresh_keys = Set(keys(fresh))
    training_keys == fresh_keys || error(
        "sbimpl: resample replay: rebuilding on the new DataFrame changed the " *
        "preprocessed model shape (training-only keys: " *
        "$(sort!(collect(setdiff(training_keys, fresh_keys)))); new-only keys: " *
        "$(sort!(collect(setdiff(fresh_keys, training_keys))))). Preserve fitted " *
        "factor/design dimensions or build and fit a genuinely new model.")
    out = Dict{Symbol,PreprocEntry}()
    for key in training_keys
        old = training[key]
        new = fresh[key]
        (old.kind === new.kind && _sb_same_raw_ref(old.raw_ref, new.raw_ref)) || error(
            "sbimpl: resample replay: preprocessing record `$key` changed from " *
            "$(old.kind)/$(repr(old.raw_ref)) to " *
            "$(new.kind)/$(repr(new.raw_ref)); the rebuilt formula is not the " *
            "fitted model.")
        out[key] = old.kind === :group_index && old.raw_ref in groups ? new : old
    end
    out
end

function _sb_assert_cv_reemission(sb::SBBRMI, groups)
    plan = generative_plan(sb)
    for group in groups
        hits = GenerativeDeclaration[]
        for d in plan.declarations
            d.role === :prior || continue
            haskey(d.keywords, :group_idx) || continue
            idx = d.keywords.group_idx
            idx isa Symbol || continue
            entry = get(plan.preproc, idx, nothing)
            entry isa PreprocEntry || continue
            entry.kind === :group_index || continue
            entry.raw_ref === group || continue
            push!(hits, d)
        end
        isempty(hits) && error(
            "sbimpl: `resample_groups` names `$group`, but the model has no " *
            "ordinary random-effect block with a tracked group index on that " *
            "column.")
        for d in hits
            expected = Symbol(d.target, :_n_g)
            get(d.keywords, :n_groups, nothing) === expected || error(
                "sbimpl: `resample_groups=[:$group]` reaches random-effect " *
                "block `$(d.target)`, but that block has no cv-contagious size " *
                "local `$expected`. Stratified, multi-membership, grouped-HSGP, " *
                "and derived-scale R2D2 blocks are not supported by this " *
                "new-population emission.")
        end
    end
    nothing
end

function _sb_mark_resample_groups(sb::SBBRMI, groups)
    marked = Dict{Symbol,Any}(sb.data)
    seen = Set{Symbol}()
    for (key, entry) in sb.preproc
        entry.kind === :group_index || continue
        entry.raw_ref in groups || continue
        marked[key] = StanBlocks.stan.maybecv(key, marked[key])
        push!(seen, entry.raw_ref)
    end
    seen == groups || error(
        "sbimpl: resample replay: failed to mark group index provenance for " *
        "$(sort!(collect(setdiff(groups, seen))))")
    model = StanBlocks.SlicModel(sb.model.model, marked, sb.model.mod)
    SBBRMI(sb.parent, model, marked, sb.preproc, copy(sb.held_out))
end

function _sb_reprocess_resample(sb::SBBRMI, new_df, groups, freeze::Bool)
    # Re-emission cannot safely guess constructor-only geometry that SBBRMI did
    # not historically retain.  The public ergonomic path starts from the
    # ordinary non-centred fit; fail if the supplied artifact used a different
    # emission rather than silently changing it while adding CV sizing.
    baseline = SBBRMI(sb.parent; mod=sb.model.mod, held_out=sb.held_out)
    stan_code(baseline) == stan_code(sb) || error(
        "sbimpl: `resample_groups` requires an SBBRMI emitted with the default " *
        "non-centered, non-CV constructor. The supplied model used additional " *
        "constructor-time geometry (for example `centered_groups` or existing " *
        "`cv_groups`) that cannot be inferred from SBBRMI. Rebuild the ordinary " *
        "fit artifact, then request `resample_groups` from it.")

    rebound = _sb_rebind_brmi(sb.parent, new_df)
    cv_template = SBBRMI(
        rebound; mod=sb.model.mod, cv_groups=groups,
        held_out=sb.held_out,
        _frozen_preproc=freeze ? sb.preproc : nothing)
    _sb_assert_cv_reemission(cv_template, groups)
    preproc = _sb_resample_preproc(sb.preproc, cv_template.preproc, groups)
    hybrid = SBBRMI(cv_template.parent, cv_template.model,
                    cv_template.data, preproc, copy(cv_template.held_out))
    prepared = reprocess(hybrid, new_df; freeze_constants=freeze)
    _sb_mark_resample_groups(prepared, groups)
end

function _sb_reprocess_data_value(sb::SBBRMI, key::Symbol, value)
    key in sb.held_out || return value
    raw = StanBlocks.getvalue(value)
    ismissing(raw) && error(
        "sbimpl: held-out response `$key` carries no recoverable data value; " *
        "rebuild the SBBRMI from its dataframe before reprocessing")
    raw
end

# Recompute one transform-output data key against `df`. `freeze=true` applies
# the stored training constant (prediction-replay); `freeze=false` re-derives
# the constant from `df`, then applies (fresh-fit semantics). Writes the
# regenerated key(s) into `new_data`, the (possibly re-derived) record into
# `new_preproc`, and marks every key it owns in `handled`.
function _sb_reprocess_entry!(new_data, new_preproc, handled, key::Symbol, e::PreprocEntry, df, freeze::Bool)
    if e.kind === :static
        new_data[key] = deepcopy(e.const_)
        new_preproc[key] = e
    elseif e.kind === :zscale || e.kind === :standardize
        v = collect(Float64, _sb_rematerialize_vec(e.raw_ref, df))
        c = freeze ? e.const_ : _sb_fit_zscale(v)
        new_data[key] = _sb_apply_zscale(c, v)
        new_preproc[key] = PreprocEntry(e.kind, c, e.raw_ref, false)
    elseif e.kind === :center
        v = collect(Float64, _sb_rematerialize_vec(e.raw_ref, df))
        c = freeze ? e.const_ : _sb_fit_center(v)
        new_data[key] = _sb_apply_center(c, v)
        new_preproc[key] = PreprocEntry(:center, c, e.raw_ref, false)
    elseif e.kind === :protect
        # No fitted constant — re-materialise the same expr tree on `df`.
        new_data[key] = collect(Float64, _sb_rematerialize_vec(e.raw_ref, df))
        new_preproc[key] = e
    elseif e.kind === :interaction
        left_key, right_key = e.raw_ref
        haskey(new_data, left_key) || error(
            "sbimpl: reprocess: interaction `$key` is missing regenerated operand `$left_key`")
        haskey(new_data, right_key) || error(
            "sbimpl: reprocess: interaction `$key` is missing regenerated operand `$right_key`")
        left = new_data[left_key]
        right = new_data[right_key]
        length(left) == length(right) || error(
            "sbimpl: reprocess: interaction `$key` operand lengths mismatch ",
            "($(length(left)) vs $(length(right)))")
        new_data[key] = collect(Float64, left .* right)
        new_preproc[key] = e
    elseif e.kind === :population_factor_dummy
        raw = _sb_df_column(df, e.raw_ref)
        raw isa AbstractVector || error(
            "sbimpl: reprocess: categorical population predictor " *
            "`$(e.raw_ref)` must be a vector, got $(typeof(raw))")
        ref = e.const_.ref
        recoded = if ref == 1
            raw
        else
            raw isa AbstractVector{<:Integer} || error(
                "sbimpl: reprocess: `factor($(e.raw_ref); ref=$ref)` requires " *
                "integer-coded categorical data")
            Int[value == ref ? 1 : value == 1 ? ref : value for value in raw]
        end
        levels = freeze ? e.const_.levels : _sb_fit_levels(recoded)
        length(levels) == e.const_.n_levels || error(
            "sbimpl: reprocess: categorical population predictor " *
            "`$(e.raw_ref)` has $(length(levels)) levels, but the fitted " *
            "interaction design has $(e.const_.n_levels). Preserve the fitted " *
            "level count or rebuild the model.")
        idx = _sb_apply_levels(levels, recoded)
        new_data[key] = Float64[i == e.const_.level ? 1.0 : 0.0 for i in idx]
        new_preproc[key] = PreprocEntry(
            :population_factor_dummy,
            (; levels, level=e.const_.level, n_levels=e.const_.n_levels, ref),
            e.raw_ref, true)
    elseif e.kind === :group_index
        raw = _sb_df_column(df, e.raw_ref)
        raw isa AbstractVector || error(
            "sbimpl: reprocess: random-effects grouping column `$(e.raw_ref)` " *
            "must be a vector, got $(typeof(raw))")
        levels = freeze ? e.const_.levels : _sb_fit_levels(raw)
        new_data[key] = _sb_apply_group_levels(levels, raw, e.raw_ref)
        new_data[e.const_.n_groups_key] = length(levels)
        push!(handled, e.const_.n_groups_key)
        new_preproc[key] = PreprocEntry(
            :group_index, (; levels, n_groups_key=e.const_.n_groups_key),
            e.raw_ref, true)
    elseif e.kind === :ranef_factor_dummy
        raw = _sb_df_column(df, e.raw_ref)
        raw isa AbstractVector || error(
            "sbimpl: reprocess: categorical random-effect predictor `$(e.raw_ref)` " *
            "must be a vector, got $(typeof(raw))")
        levels = freeze ? e.const_.levels : _sb_fit_levels(raw)
        length(levels) == e.const_.n_levels || error(
            "sbimpl: reprocess: categorical random-effect predictor `$(e.raw_ref)` " *
            "has $(length(levels)) levels, but the fitted design has " *
            "$(e.const_.n_levels). Preserve the fitted level count or rebuild " *
            "the model.")
        idx = _sb_apply_levels(levels, raw)
        new_data[key] = Float64[i == e.const_.level ? 1.0 : 0.0 for i in idx]
        new_preproc[key] = PreprocEntry(
            :ranef_factor_dummy,
            (; levels, level=e.const_.level, n_levels=e.const_.n_levels),
            e.raw_ref, true)
    elseif e.kind === :joint_response
        names = Tuple(e.raw_ref)
        names == Tuple(e.const_.outcomes) || error(
            "sbimpl: reprocess: joint-response provenance for `$key` changed " *
            "outcome order")
        columns = map(names) do outcome
            NamedColumn(outcome, DataColumn(_sb_df_column(df, outcome)))
        end
        joint = JointResponseColumn(columns)
        values = _brm_joint_response_values(joint; prefix="sbimpl: reprocess")
        nobs = length(values)
        for source in e.const_.mean_sources
            source_values = _sb_df_column(df, source)
            source_values isa AbstractVector || error(
                "sbimpl: reprocess: joint-response mean source `$source` must " *
                "be a vector, got $(typeof(source_values))")
            length(source_values) == nobs || error(
                "sbimpl: reprocess: joint-response mean source `$source` has " *
                "$(length(source_values)) rows, but the aligned outcomes have " *
                "$nobs rows")
        end
        new_data[key] = values
        new_data[e.const_.n_key] = nobs
        push!(handled, e.const_.n_key)
        new_preproc[key] = e
    elseif e.kind === :kernel_subject_count
        subjects = _sb_kernel_subject_values(
            _sb_df_column(df, e.raw_ref), e.raw_ref)
        new_data[key] = length(subjects)
        new_preproc[key] = e
    elseif e.kind === :kernel_ragged
        arg_name, event_group, subject_group = e.raw_ref
        subjects = _sb_kernel_subject_values(
            _sb_df_column(df, subject_group), subject_group)
        event_groups = _sb_df_column(df, event_group)
        event_groups isa AbstractVector || error(
            "sbimpl: reprocess: kernel event grouping column `$event_group` " *
            "must be a vector, got $(typeof(event_groups))")
        rows = _sb_kernel_ragged_partition(
            arg_name, event_group,
            collect(_sb_group_values(event_groups)), subjects)
        if e.const_.is_lp
            new_data[key] = rows
        else
            flat = _sb_df_column(df, arg_name)
            flat isa AbstractVector || error(
                "sbimpl: reprocess: kernel ragged source `$arg_name` must be a " *
                "vector, got $(typeof(flat))")
            length(flat) == length(event_groups) || error(
                "sbimpl: reprocess: kernel ragged source `$arg_name` has " *
                "$(length(flat)) rows but grouping column `$event_group` has " *
                "$(length(event_groups)); they must describe the same event axis.")
            new_data[key] = [flat[r] for r in rows]
        end
        new_preproc[key] = e
    elseif e.kind === :multi_membership
        group_names = e.raw_ref.groups
        weight_names = e.raw_ref.weights
        raw_groups = Tuple(_sb_df_column(df, n) for n in group_names)
        raw_weights = isnothing(weight_names) ? nothing :
                      Tuple(_sb_df_column(df, n) for n in weight_names)
        levels = freeze ? e.const_.levels : nothing
        prepared = _sb_prepare_mm(raw_groups, raw_weights, e.const_.normalize;
                                  levels, group_names, weight_names)
        new_data[key] = prepared.group_idx
        new_data[e.const_.weight_key] = prepared.weights
        new_data[e.const_.n_groups_key] = length(prepared.levels)
        new_data[e.const_.n_obs_key] = prepared.n_obs
        new_data[e.const_.n_memberships_key] = prepared.n_memberships
        for owned in (e.const_.weight_key, e.const_.n_groups_key,
                      e.const_.n_obs_key, e.const_.n_memberships_key)
            push!(handled, owned)
        end
        const_ = merge(e.const_, (; levels=prepared.levels))
        new_preproc[key] = PreprocEntry(:multi_membership, const_, e.raw_ref, true)
    elseif e.kind === :spline
        v = collect(Float64, _sb_df_column(df, e.raw_ref))
        old_fit = e.const_.fit
        fit = freeze ? old_fit : _sb_fit_spline(v; k=old_fit.k)
        Xnull, Zpen = _sb_apply_spline(fit, v)
        new_data[key] = Xnull
        new_data[e.const_.zpen_key] = Zpen
        push!(handled, e.const_.zpen_key)
        new_preproc[key] = PreprocEntry(
            :spline, (; fit, zpen_key=e.const_.zpen_key), e.raw_ref, false)
    elseif e.kind === :tensor_spline
        axes = _sb_gp_axes_from_df(df, e.raw_ref, :t2)
        length(axes) == 2 || error("sbimpl: reprocess: `t2` needs exactly two margins")
        old_fit = e.const_.fit
        fit = freeze ? old_fit : _sb_fit_t2(axes[1], axes[2]; k=old_fit.k)
        Xfixed, Zrr, Zrn, Znr = _sb_apply_t2(fit, axes[1], axes[2])
        new_data[key] = Xfixed
        new_data[e.const_.zrr_key] = Zrr
        new_data[e.const_.zrn_key] = Zrn
        new_data[e.const_.znr_key] = Znr
        push!(handled, e.const_.zrr_key)
        push!(handled, e.const_.zrn_key)
        push!(handled, e.const_.znr_key)
        new_preproc[key] = PreprocEntry(:tensor_spline,
            (; fit, zrr_key=e.const_.zrr_key, zrn_key=e.const_.zrn_key,
             znr_key=e.const_.znr_key), e.raw_ref, false)
    elseif e.kind === :gp
        axes = _sb_gp_axes_from_df(df, e.raw_ref, :gp)
        new_data[key] = _sb_gp_matrix(axes)
        new_preproc[key] = PreprocEntry(:gp, e.const_, e.raw_ref, false)
    elseif e.kind === :hsgp && get(e.const_, :cov, :exp_quad) === :periodic
        axes = _sb_gp_axes_from_df(df, e.raw_ref, :hsgp)
        length(axes) == 1 || error(
            "sbimpl: reprocess: periodic `hsgp` expects exactly one axis")
        K = e.const_.K
        # Nothing here is fitted: the basis is a function of the formula
        # constants (period, k) and the new axis alone, so `freeze_constants`
        # cannot change it and the companion data reproduce byte-for-byte.
        new_data[key] = _sb_apply_hsgp_periodic(e.const_.period, only(axes), K)
        new_data[e.const_.harmonics_key] = _sb_hsgp_periodic_harmonics(K)
        push!(handled, e.const_.harmonics_key)
        new_data[e.const_.rho_lower_key] = _sb_hsgp_periodic_rho_lower(K)
        push!(handled, e.const_.rho_lower_key)
        new_preproc[key] = PreprocEntry(:hsgp, e.const_, e.raw_ref, false)
    elseif e.kind === :hsgp
        axes = _sb_gp_axes_from_df(df, e.raw_ref, :hsgp)
        K = e.const_.K
        domain_fits = get(e.const_, :domain_fits, nothing)
        _sb_hsgp_check_explicit_domain(domain_fits, axes)
        # An explicit domain is part of the formula, not a fitted constant: it
        # stays fixed even when other preprocessing constants are re-fitted.
        fits = !isnothing(domain_fits) ? domain_fits :
               freeze ? e.const_.fits : _sb_fit_hsgp(axes, K, e.const_.c)
        PHI, omega2 = _sb_apply_hsgp(fits, axes, K)
        orthogonal_to = get(e.const_, :orthogonal_to, nothing)
        orthogonal_to === :linear &&
            (PHI = _sb_orthogonalize_hsgp_linear(PHI, only(axes)))
        new_data[key] = PHI
        new_data[e.const_.omega2_key] = omega2
        push!(handled, e.const_.omega2_key)
        # The validity floor is a function of the fitted `L`, so it has to move
        # with the basis: a non-frozen re-fit that left it behind would bound
        # `rho` for a basis this data no longer has.
        iso = e.const_.iso
        new_data[e.const_.rho_lower_key] = _sb_hsgp_rho_lower_data(fits, K, iso)
        push!(handled, e.const_.rho_lower_key)
        new_preproc[key] = PreprocEntry(:hsgp,
            (; fits, K, c=e.const_.c, iso, domain_fits, orthogonal_to,
             omega2_key=e.const_.omega2_key,
             rho_lower_key=e.const_.rho_lower_key), e.raw_ref, false)
    elseif e.kind === :categorical_outcome
        v = _sb_df_column(df, e.raw_ref)
        fitted_levels = e.const_.levels
        expected_n_levels = e.const_.n_levels
        levels = freeze ? fitted_levels : _sb_fit_levels(v)
        length(levels) == expected_n_levels || error(
            "sbimpl: reprocess: categorical outcome `$key` has $(length(levels)) levels, " *
            "but the fitted `CategoricalLogit` has $expected_n_levels. " *
            "Preserve the fitted level set/order or rebuild the model with one " *
            "linear predictor per non-reference class.")
        new_data[key] = _sb_apply_levels(levels, v)
        new_preproc[key] = PreprocEntry(
            :categorical_outcome, (; levels, n_levels=expected_n_levels),
            e.raw_ref, true)
    elseif e.kind === :ordinal_outcome
        v = _sb_df_column(df, e.raw_ref)
        fitted_levels = e.const_.levels
        expected_n_levels = e.const_.n_levels
        levels = freeze ? fitted_levels : _sb_fit_levels(v)
        length(levels) == expected_n_levels || error(
            "sbimpl: reprocess: ordinal outcome `$key` has $(length(levels)) levels, " *
            "but the fitted `Ordinal` model has $expected_n_levels. Preserve the " *
            "fitted ordered level set or rebuild the model.")
        new_data[key] = _sb_apply_levels(levels, v)
        new_preproc[key] = PreprocEntry(
            :ordinal_outcome, (; levels, n_levels=expected_n_levels),
            e.raw_ref, true)
    elseif e.kind === :ordinal_threshold_predictor
        v = _sb_df_column(df, e.raw_ref)
        v isa AbstractVector{<:Real} || error(
            "sbimpl: reprocess: ordinal threshold predictor `$key` must be numeric, " *
            "got $(typeof(v))")
        all(isfinite, v) || error(
            "sbimpl: reprocess: ordinal threshold predictor `$key` contains " *
            "non-finite values")
        new_data[key] = collect(Float64, v)
        new_preproc[key] = e
    elseif e.kind === :observation_weight
        raw = _sb_df_column(df, e.raw_ref)
        response = _sb_df_column(df, e.const_.response)
        response isa AbstractVector || error(
            "sbimpl: reprocess: weighted response `$(e.const_.response)` must " *
            "be a vector, got $(typeof(response))")
        new_data[key] = _brm_prepare_observation_weight_values(
            e.const_.kind, raw, length(response), e.const_.response, e.raw_ref;
            prefix="sbimpl: reprocess")
        new_preproc[key] = e
    elseif e.kind === :interval_censored_predictor
        x_name, upper_name = e.raw_ref
        plan = _sb_interval_censored_predictor_plan(
            x_name, _sb_df_column(df, x_name),
            upper_name, _sb_df_column(df, upper_name), e.const_.lower)
        new_data[key] = plan.x_exact
        new_data[e.const_.lower_key] = plan.x_lower
        new_data[e.const_.upper_key] = plan.x_upper
        new_data[e.const_.exact_index_key] = plan.Jexact
        new_data[e.const_.interval_index_key] = plan.Jinterval
        for owned in (e.const_.lower_key, e.const_.upper_key,
                      e.const_.exact_index_key, e.const_.interval_index_key)
            push!(handled, owned)
        end
        new_preproc[key] = e
    elseif e.kind === :factor || e.kind === :mo
        v = _sb_df_column(df, e.raw_ref)
        levels = freeze ? e.const_ : _sb_fit_levels(v)
        new_data[key] = _sb_apply_levels(levels, v)   # errors loudly on an unseen level
        new_preproc[key] = PreprocEntry(e.kind, levels, e.raw_ref, true)
        if e.kind === :factor
            # The `<x>_n_levels` count key co-emitted by `_sb_emit_cat!`: frozen
            # training count under freeze=true, re-counted under freeze=false.
            n_name = Symbol(e.raw_ref, :_n_levels)
            new_data[n_name] = length(levels)
            push!(handled, n_name)
        end
    elseif e.kind === :ragged_gather
        # A gathered ragged response or its data-backed censoring/truncation
        # bound. Re-gather the flat source column into the kernel's per-subject
        # row order — the same operation `_sb_ragged_lhs_layout` performed at fit
        # time. No fitted constant, so `freeze` is irrelevant; the subject axis
        # comes from the new DataFrame's own kernel frame (which is how a
        # resample re-emission and a same-subject replay stay consistent).
        source = e.raw_ref
        group_col = e.const_.group_col
        subject_col = e.const_.subject_col
        raw = _sb_df_column(df, source)
        if raw isa AbstractVector{<:AbstractVector}
            new_data[key] = raw   # already per-subject ragged: pass through
        else
            subject_values = _sb_kernel_subject_values(
                _sb_df_column(df, subject_col), subject_col)
            group_values = collect(_sb_group_values(_sb_df_column(df, group_col)))
            length(group_values) == length(raw) || error(
                "sbimpl: reprocess: ragged source `$source` has $(length(raw)) " *
                "rows but grouping column `$group_col` has " *
                "$(length(group_values)); they must describe the same " *
                "observation axis.")
            rows = _sb_ragged_group_rows(key, group_col, group_values,
                                         subject_values)
            new_data[key] = [raw[r] for r in rows]
        end
        new_preproc[key] = e
    else
        error("sbimpl: reprocess: unknown preproc kind `$(e.kind)` for key `$key`")
    end
    push!(handled, key)
    nothing
end

"""
    reprocess(sb::SBBRMI, new_df; freeze_constants=true,
              resample_groups=()) -> SBBRMI

Re-materialise the SBBRMI's Stan data dict against `new_df`, **re-running** the
Julia-side preprocessing (decision nr3v8n A) — so the silent-stale-constant bug
of a naive per-column `sb.model(; col=…)` rebind is avoided. Returns a NEW
`SBBRMI` that REUSES the already-transpiled Stan model (byte-identical
`stan_code` when shapes are stable) with the new data dict.

- `freeze_constants=true` (default — prediction / replay): apply the **training**
  constant to `new_df` (z-score with training mean/sd, map factor codes via the
  training level set, rebuild the spline/HSGP basis on the training
  eigenbasis/(mean, L)). This is the operation the downstream PKPD app hand-rolls outside BRM.
- `freeze_constants=false` (fresh-fit semantics): re-derive each constant from
  `new_df`, then apply.
- `resample_groups=()` (default): retain the fitted random-effect coordinates.
  Naming one or more ordinary grouping factors (for example `[:subject]`)
  re-emits the BRMI with cv-contagious sizing, derives those groups' levels
  from `new_df`, and marks their indices so their standardized effects are
  re-drawn in generated quantities from the fitted covariance. Predictor
  constants remain frozen unless `freeze_constants=false` is also requested.
  This changes Stan source by construction; it is the new-population/CV twin
  of the default same-group replay.

Covered: the Julia-side transforms (`zscale`/`standardize`/`center`/`factor`/
`mo`/`s`/`t2`/`gp`/`hsgp`), interval-censored predictor splits,
`protect`/implicit-fn columns (re-materialised on `new_df`), typed `mm(...)`
and ordinary plain/`|ID|` random-effect group
indices, `kernel(...)` subject counts and `ragged(x, group)` columns,
formula-boundary `ragged(response, group)` observations and their data-backed
`censored`/`truncated`/`interval_censored` bounds (both re-gathered from the
flat source column into the kernel's per-subject row order), continuous
× continuous interaction columns, typed observation weights, categorical
outcomes, and pass-through raw columns (plain data, `me` obs values, `ar`
time). Frozen replay errors loudly on an unseen fitted level. Stratified
`gr(g, by=b)` group-index replay remains correct-or-loud unsupported rather
than silently copying stale structure.
"""
function reprocess(sb::SBBRMI, new_df; freeze_constants::Bool=true,
                   resample_groups=())
    groups = _sb_resample_group_set(resample_groups)
    isempty(groups) || return _sb_reprocess_resample(
        sb, new_df, groups, freeze_constants)
    # Current kernel(...) emitters record each gathered/index ragged input, but
    # an SBBRMI serialized by the short-lived fail-closed implementation may
    # still lack that provenance. Diagnose such a legacy artifact up front;
    # current models continue into the normal regeneration path below.
    kernel_ragged = sort!(Symbol[k for k in keys(sb.data)
        if k !== _SB_PREPROC_KEY && !haskey(sb.preproc, k) &&
           startswith(String(k), "kernel_") && endswith(String(k), "_ragged")])
    isempty(kernel_ragged) || error(
        "sbimpl: reprocess: this model has kernel(...) `ragged(x, group)` inputs ",
        "$(kernel_ragged) gathered per-group at build time with no preprocessing ",
        "record, so this artifact cannot regenerate them on a new DataFrame. ",
        "Rebuild the SBBRMI once with the current BayesianRegressionModels ",
        "version to enable frozen same-group replay. For genuinely new groups, ",
        "use `generative_plan(plan, new_schedule)` to rebuild from the formula.")
    new_data = Dict{Symbol,Any}()
    new_preproc = Dict{Symbol,PreprocEntry}()
    handled = Set{Symbol}()
    interaction_keys = Set(k for (k, e) in sb.preproc if e.kind === :interaction)
    # 1. Regenerate every independent transform-output key from its preproc
    #    record. Interactions wait until their raw/transformed operands exist.
    for (key, e) in sb.preproc
        e.kind === :interaction && continue
        _sb_reprocess_entry!(new_data, new_preproc, handled, key, e, new_df, freeze_constants)
    end
    # 2. Account for every remaining old data key: pass-through raw columns, or
    #    frozen structural scalars; ERROR on any unaccounted derived structure.
    for (k, stored) in sb.data
        v = _sb_reprocess_data_value(sb, k, stored)
        k in handled && continue
        k in interaction_keys && continue
        if v isa AbstractVector
            if _sb_df_has_column(new_df, k)
                new_data[k] = _sb_df_column(new_df, k)   # pass-through (plain / me obs / ar time)
            else
                error(
                    "sbimpl: reprocess: data key `$k` is a derived vector with no ",
                    "preprocessing record and is not a column of the new DataFrame. ",
                    "reprocess covers the Julia-side predictor transforms ",
                    "(zscale/standardize/center/factor/mo/s/t2/gp/hsgp), protect/implicit-fn ",
                    "columns, typed `mm(...)`, plain random-effects group indices, ",
                    "kernel ragged inputs, and pass-through raw columns; this derived ",
                    "structure is outside that replay surface — ",
                    "rebuild the SBBRMI from the new DataFrame instead.")
            end
        elseif v isa Number || v isa AbstractString || v isa Bool
            new_data[k] = v   # frozen structural scalar / formula literal (e.g. me `sd_<x>`)
        else
            error(
                "sbimpl: reprocess: data key `$k` (::$(typeof(v))) is a derived ",
                "structure with no preprocessing record. reprocess cannot safely ",
                "regenerate it on the new DataFrame — rebuild the SBBRMI instead.")
        end
    end
    # 3. Derived interactions depend on operands regenerated in steps 1-2.
    for (key, e) in sb.preproc
        e.kind === :interaction || continue
        _sb_reprocess_entry!(new_data, new_preproc, handled, key, e, new_df, freeze_constants)
    end
    new_model = StanBlocks.SlicModel(sb.model.model, new_data, sb.model.mod)
    _sb_apply_held_out(
        SBBRMI(sb.parent, new_model, new_data, new_preproc), sb.held_out)
end

function reprocess(plan::GenerativePlan, new_df; freeze_constants::Bool=true,
                   resample_groups=())
    sb = SBBRMI(plan.parent, plan.model, plan.data, plan.preproc,
                copy(plan.held_out))
    groups = _sb_resample_group_set(resample_groups)
    replayed = reprocess(sb, new_df; freeze_constants,
                         resample_groups=groups)
    _generative_plan(replayed, plan.builder,
                     isempty(groups) ? plan.cv_groups : groups)
end

"""
    restan_data(sb::SBBRMI, new_df; freeze_constants=true,
                resample_groups=()) -> Dict

Thin convenience over [`reprocess`](@ref): the prepared Stan **data dict** for
`new_df`, ready for a `param_constrain!` replay. Equivalent to
`StanBlocks.stan_data(reprocess(sb, new_df; freeze_constants,
resample_groups).model)`. Same
`freeze_constants` semantics (default `true` = training constants applied to new
data), and accepts the same `resample_groups` keyword. A non-empty
`resample_groups` also changes the Stan program; call `reprocess` when you need
the corresponding `SBBRMI`/source as well as this data-only convenience result.
See [`reprocess`](@ref) for the covered-terms list and error cases.
"""
restan_data(sb::SBBRMI, new_df; freeze_constants::Bool=true,
            resample_groups=()) =
    StanBlocks.stan_data(reprocess(
        sb, new_df; freeze_constants, resample_groups).model)

# ---- top-level op dispatch ---------------------------------------------------

_sb_emit!(stmts, data, key, op::ExprColumn; id_lookup=_sb_empty_id_lookup(), obs_n=nothing, cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(), group_block_lookup=Dict(), effect_overrides=Dict{Symbol,Any}(), r2d2=_sb_empty_r2d2()) =
    _sb_emit_expr!(stmts, data, key, getf(op), op; id_lookup, obs_n, cv_groups, centered_groups, group_block_lookup, effect_overrides, r2d2)
# Raw data / missing columns appear as top-level ops when the formula mentions
# them as bare references (e.g. `c2` in `loc ~ 1 + c2`). Nothing to emit — the
# prepass already stashed data columns in `data`.
_sb_emit!(stmts, data, key, ::DataColumn; kwargs...) = nothing
_sb_emit!(stmts, data, key, ::MissingColumn; kwargs...) = nothing
_sb_emit!(stmts, data, key, op; kwargs...) = error("sbimpl: top-level op for `$key` not an ExprColumn (got $(typeof(op)))")

_sb_emit_expr!(stmts, data, key, ::typeof(~), op; id_lookup=_sb_empty_id_lookup(), obs_n=nothing, cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(), group_block_lookup=Dict(), effect_overrides=Dict{Symbol,Any}(), r2d2=_sb_empty_r2d2()) = begin
    lhs, rhs = getargs(op, 2)
    _sb_sampling!(stmts, data, key, lhs, rhs; id_lookup, obs_n, cv_groups,
                  centered_groups, group_block_lookup, effect_overrides, r2d2)
end
_sb_emit_expr!(stmts, data, key, ::typeof(assign), op; id_lookup=_sb_empty_id_lookup(), kwargs...) = begin
    _, rhs = getargs(op, 2)
    target_expr = _sb_scalar_expr(rhs, data)
    push!(stmts, :($key = $target_expr))
end
_sb_emit_expr!(stmts, data, key, f, op; kwargs...) = error("sbimpl: unsupported top-level op `$f` for `$key`")

_sb_empty_id_lookup() = Dict{Tuple{Symbol,Tuple{Symbol,Any}}, Any}()


# ---- sampling: likelihood vs linear-predictor split --------------------------

"""
    _sb_submodel_rhs!(stmts, data, target, f, rhs)

sbimpl extension hook. Override (e.g. in a downstream `-ext.jl`) to route
`target ~ f(...)` where `f` is a known SLIC submodel family
(`logistic_dr`, `gamma_time`, …) straight to `target ~ <slic>(; kwargs)`,
bypassing the population-linear-predictor wrap (which would otherwise
multiply the submodel output by a fresh β).

Return anything non-`nothing` to claim the binding; return `nothing` to
fall through to the default linear-predictor path. The default method
(this one) returns `nothing`.

**Use `import BayesianRegressionModels: _sb_submodel_rhs!`** (not
`using`) when adding methods from a downstream module so the binding is
extended rather than shadowed.

**Reserved data keys — BRM owns raw grouping-column names.** Any raw
column used as a ranef grouping factor (the `g` in `(… | g)`,
`(… | ID | g)`, or `gr(g, by=…)`) is reclaimed by BRM's ranef pre-pass:
the raw labels are DELETED from `data` and replaced by a dense integer
index (`g_idx`) and level count (`n_g`), because Stan cannot consume raw
(possibly string) labels. So a hook must NOT stash its own vector under a
raw column name that the same model also uses as a grouping factor, nor
emit a reference to it — the reclaim will delete it out from under the
emitted statement, and the failure surfaces only much later, in
StanBlocks tracing, as an unresolvable-symbol error. Key consumer-owned
`data` PER TARGET instead, e.g. `Symbol(col_key, :_, target)`, which is
self-owned and order-independent. (`_sb_reclaim_group_col!` now turns a
collision into an immediate, correctly-attributed error at the delete
site rather than a late one in a third package.)
"""
_sb_submodel_rhs!(stmts, data, target, f, rhs) = nothing

# ============================================================================
"""
    pred ~ kernel(data..., per_subject_lps...) do slices..., lp_values...
        ...
    end

General group-local submodel term: broadcast an inline cell over the groups of
the linear predictors it is given.

The per-subject LP formulas own the population effects, covariates, links and
random-effect buckets — they are ordinary `@brm` formulas declared in the same
block. `kernel` derives ONE shared grouping from those LPs and evaluates the
`do` block once per group, with each positional passed in as that group's slice:

```julia
log_CL ~ 1 + weight + (1 | p | subject)
log_Vc ~ 1 + (1 | p | subject)

conc ~ kernel(t_obs, ragged(dose, dose_subject), log_CL, log_Vc) do ts, doses, lCL, lVc
    ...                                    # runs per subject, in Stan
end
```

Positionals split by kind. A raw data column on the kernel's own
one-row-per-subject frame is gathered into a per-group slice. A column living on
a DIFFERENT frame — a dose-event table, say — must declare its grouping with
[`ragged`](@ref). A linear predictor is passed as that group's scalar value.

Dispatch tag only — lowering lives in `_sb_kernel_doblock!` (sbimpl). The legacy
`model=` / `obs=` spelling and its anonymous `n_eta` block were removed by user
decision `130c904`; the `do`-block form above is the only one. A name such as
`eta_CL` is merely a user-chosen ordinary linear-predictor name—there is no
kernel-owned eta vector or positional eta-index contract in the current API.
"""
function kernel end

"""
    ragged(x, group)

Group a FLAT secondary row axis by a raw data column that names the subject of
every row. The marker has two formula positions:

- As a `kernel(...)` positional, `ragged(x, group)` gives the cell a ragged
  per-subject vector. `x` may be a flat data column or an event-axis linear
  predictor.
- As an observation LHS, `ragged(y, group) ~ Family(pred, ...)` groups a flat
  response before applying the top-level likelihood. The referenced
  `kernel(...)` result supplies the authoritative subject row order, so labels
  are joined rather than sorted or inferred from first occurrence. The emitted
  observation keeps the logical name `y` and therefore uses StanBlocks' normal
  top-level ragged outputs: flat `y_gen`, group-aggregate `y_likelihood`, and
  descriptor `segments`. This formula-boundary grouping is SBBRMI/sbimpl-only.

For the kernel-positional form, `x` lives on some frame other than the kernel's
one-row-per-subject frame — a dose-event table, say — and `group` names, for
every row of that frame, which subject it belongs to. The cell receives `x` as
a RAGGED per-subject vector.

`x` may be either a linear predictor declared in the same `@brm` block, or a raw
flat data column. It is the same grouping either way:

```julia
log_F  ~ 1 + vessel + mo(diet) + hsgp(log_dose)     # rows = dose events
log_CL ~ 1 + weight + (1 | p | subject)             # rows = subjects

pred ~ kernel(t_obs, dv,
              ragged(dose_amount, dose_subject),    # flat column -> grouped here
              ragged(log_F, dose_subject),          # predictor   -> indexed in Stan
              log_CL) do ts, yy, doses, lF, lCL
    effective_dose = doses .* exp(lF)
    ...
end
```

Only the REALIZATION differs, and the difference is forced: a linear predictor
is a Stan parameter, so it cannot be gathered on the Julia side — the plate
takes an index column and the cell fancy-indexes the unsliced predictor
(`log_F[rows]`). A data column is Julia data, so it is gathered directly into a
`Vector{Vector{T}}`, which StanBlocks ingests as a ragged column natively —
exactly what a hand-prepared per-subject view would have been. Wrapping a column
does not consume the flat original: a term that names it on its own axis
(`hsgp(log_dose)` above) still sees it.

Why the grouping is an explicit argument rather than derived: an ordinary
per-subject LP is grouped by the ranef bucket it already carries
(`_sb_kernel_lp_bucket`), but a secondary-axis population LP like the one above
has no random-effect term at all, and a raw column carries no grouping
whatsoever. The axis has to be declared, and `group` is that declaration.

ONE VALUE PER ROW OF `x`'s OWN FRAME — no expansion happens anywhere. If the
event table stores a compact schedule (one row per dose OP, carrying an interval
and a count) then `x` has one value per OP. Constructing `ragged(x, group)`
requires one `group` key per row of `x` and joins those keys to the kernel's
outer subject labels. That is local validation of the grouping operation, not
an inner-shape contract between cell arguments. Kernel compares no totals or
per-subject inner lengths across positionals; once each positional supplies one
outer cell per subject, relationships among the values inside a cell belong to
the cell body.

Dispatch tag only — lowering lives in `_sb_kernel_doblock!` (sbimpl).
Observation-LHS lowering lives in `_sb_sampling!`.
"""
ragged

_check_term_kwargs(::typeof(ragged), kwargs) = isempty(kwargs) || error(
    "@brm: ragged(...) takes no keywords, got $(keys(kwargs)). The spelling is ",
    "`ragged(<linear predictor or flat column>, <grouping column>)`.")

# Reject the retired keyword surface at CONSTRUCTION — i.e. at the `@brm` /
# `kernel(...)` call the consumer actually wrote — not merely when the model is
# lowered.
#
# The loud rejections further down (`_sb_kernel_doblock!`, `_sb_submodel_rhs!`)
# run inside `SBBRMI(...)`, and `@brm` is a pure parser that captured `by=` /
# `n_eta=` / `model=` / `obs=` into the term's `getkwargs()` without looking at
# them. So a model written in the retired v1 spelling BUILT cleanly and objected
# only once someone lowered it. A consumer whose compatibility gate stops at
# BRMI construction — a reasonable gate, since it needs no Stan toolchain — saw
# retired syntax keep passing, and carried `by=`/`n_eta=` across 13 executable
# kernel sites for eight days after the removal landed while `brm-use` told them
# the keywords were rejected loudly (snag `by-and-n-eta-are-3625f645`).
#
# Checked in the same order the lowering-time guards use, so the reported
# keyword does not change when several are present. Keys off the `kernel`
# function object, so `hsgp(x; by=g)` — where `by=` is live — is unaffected, and
# an aliased `kernel` is still caught. The lowering-time guards stay as the
# backstop for a BRMI assembled without the macro.
function _check_term_kwargs(::typeof(kernel), kwargs)
    for (k, replacement) in (
            (:by, "Grouping is DERIVED from the ranef bucket of the per-subject \
                   linear-predictor positional args."),
            (:n_eta, "Declare per-subject linear predictors with `(1 | ID | group)` \
                      terms and pass those LPs positionally."),
            (:model, "Write the per-subject cell INLINE as a `do`-block."),
            (:obs, "Write observation likelihoods as ordinary `~` statements inside \
                    the cell body."),
        )
        haskey(kwargs, k) && error(
            "@brm: kernel(...) no longer accepts `$k=`. ", replacement,
            "\nThe surface is formula linear predictors plus one inline do-block:\n",
            "    log_CL ~ 1 + weight + (1 | p | subject)\n",
            "    log_V  ~ 1 +          (1 | p | subject)\n",
            "    pred ~ kernel(t, dose, dv, log_CL, log_V) do ts, d, yy, lCL, lV\n",
            "        mu = <prediction from exp(lCL), exp(lV) over ts, d>\n",
            "        yy ~ normal(mu, sigma)\n",
            "        mu\n",
            "    end\n",
            "See the `brm-use` skill, `kernel(...)` section.")
    end

    # `kernel(...)` takes NO keywords: the cell is the do-block and everything it
    # consumes is positional. An unrecognised keyword is therefore a typo or
    # retired syntax, never a live option, and silently ignoring one is exactly
    # how the retired v1 spelling kept passing a consumer's construction-time
    # gate for eight days after its removal (snag `by-and-n-eta-are-3625f645`).
    # The four named guards above stay because they can say what to write
    # instead; this catches everything else.
    for k in keys(kwargs)
        error("@brm: kernel(...) does not accept `$k=` — it accepts no keywords. ",
              "The cell is the do-block and everything it consumes is positional; ",
              "named values the cell assigns are addressable from the descriptor ",
              "without any annotation (`brm_output(d, :$k)` if `$k` is one). ",
              "See the `brm-use` skill, `kernel(...)` section.")
    end
    nothing
end

# do-block kernel surface (decision z9vkkf, User chose B): the consumer writes the
# per-subject cell body INLINE as a plate-style do-block, with the obs likelihood as
# ordinary `~` statements in the body (no `obs=` family DSL). Per-subject LPs own
# their random-effect blocks; the kernel only broadcasts the cell over their shared
# grouping.
#
# Multi-output (e.g. joint PK-PD, two correlated LP buckets). Give EACH output its
# OWN ragged observation column + time grid and observe it as a WHOLE column:
#
#   log_CL   ~ 1 + (1 | p | subject)
#   log_V    ~ 1 + (1 | p | subject)
#   log_Emax ~ 1 + (1 | q | subject)
#   log_EC50 ~ 1 + (1 | q | subject)
#   pred ~ kernel(
#       t_pk, t_pd, dose, dv_pk, dv_pd, log_CL, log_V, log_Emax, log_EC50,
#   ) do tpk, tpd, d, ypk, ypd, lCL, lV, lEmax, lEC50
#       conc = <PK prediction from lCL/lV over tpk>
#       resp = <PD prediction from lEmax/lEC50 (+ conc at tpd) over tpd>
#       ypk ~ normal(conc, sd)   # obs are just statements, one per output
#       ypd ~ normal(resp, sd)
#       conc                      # last expr = collected result
#   end
#
# Do-params bind to sliced positional data/LP args one-for-one, in order. Outer
# params (sd, covariates) are reached by lexical scope. No `by=`/`n_eta=`/
# `model=`/`obs=`.
#
# A custom `@deffun`-backed distribution must match one of its declared
# signatures exactly. For `truncated_normal`, a vector observation and location
# therefore require vector scale/bounds too; scalar captures must be lifted:
#
#   y ~ truncated_normal(mu,
#       rep_vector(sigma, dims(mu)[1]),
#       rep_vector(lloq, dims(mu)[1]),
#       rep_vector(uloq, dims(mu)[1]))
#
# Keep the `dims(mu)[1]` query inline: assigning it to an untyped plate-local
# name currently loses the integer type needed by `rep_vector`.
#
# NOT YET SUPPORTED — in-cell cmt/compartment masking. Keying ONE interleaved obs
# column by a cmt column INSIDE the cell —
#     y[c .== 1] ~ normal(conc[c .== 1], sd)   # <- does NOT transpile
# — is blocked in the StanBlocks substrate two ways: (1) a real ragged column slices
# to a native Stan `vector`, and `.==` on a native vector yields a real vector, not
# the `array[] int` mask boolean-mask indexing needs; (2) with an integer cmt column
# the mask + `findall` work, but the plate cannot hold the resulting `array[] int`
# index as a per-cell local ("scalar or vector[K] only (MVP)", StanBlocks
# `_plate_cell_shape`). Until that substrate lands, use separate per-output columns
# (above), or sub-select OUTSIDE the plate with a top-level integer index column over
# the collected result (`conc_pred[obs_idx] ~ ...`). This is NOT pending work: decision
# z9vkkf ("what @brm surface for cmt-keyed PK-PD obs?") was RESOLVED 2026-07-22 as
# "None of the above" — the ByCmt/obs-family extension, the defer, and the coupled
# `driver=` terms were ALL declined. So no cmt-keyed obs surface is being built; the
# per-output-column do-block form above is the answer.
#
# Walk a per-subject linear predictor to the `(id, group)` ranef bucket that
# defines its grouping — what lets `kernel(...)` DERIVE its plate grouping from
# the LPs handed in instead of being told via `by=` (v2, GO `0dnesv9`). `by=`
# only ever contributed the subject COUNT, restating a fact the ranef already
# knows, and was checked against nothing.
#
# Returns `(id_sym, group_key, group_col)`.
function _sb_kernel_lp_bucket(lp_col)
    decl = parent(lp_col)
    (decl isa ExprColumn && getf(decl) === ~) || error(
        "sbimpl: kernel(...) positional arg `$(name(lp_col))` is not a formula ",
        "statement; expected `$(name(lp_col)) ~ <terms>` in this @brm block.")
    rhs = getargs(decl)[2]
    pop_terms, ran_terms, direct_terms = Any[], Any[], Any[]
    for t in _sb_terms(rhs)
        _sb_classify_term!(t, pop_terms, ran_terms, direct_terms)
    end
    isempty(ran_terms) && error(
        "sbimpl: kernel(...) per-subject linear predictor `$(name(lp_col))` has no ",
        "random-effect term, so it defines no grouping. Give it one, e.g. ",
        "`$(name(lp_col)) ~ 1 + (1 | p | subject)`.")
    parts = map(_sb_ranef_parts, ran_terms)
    buckets = unique([(p[1], _sb_group_key(p[3])) for p in parts])
    length(buckets) == 1 || error(
        "sbimpl: kernel(...) per-subject linear predictor `$(name(lp_col))` spans ",
        "several ranef buckets ($(buckets)); a kernel cell needs exactly one grouping.")
    id_sym, gkey = only(buckets)
    desc = parts[1][3]
    (id_sym, gkey, desc isa NamedColumn ? desc : first(desc))
end

# Collect `(name, length)` for every RAW data column reachable from a formula
# RHS. Used to check that a `ragged(x, group)` LP really is declared over the
# same row axis its grouping column describes. Descent stops at a nested `~`
# ExprColumn: that is a REFERENCE to another formula's value, not part of this
# one's design, so its data belongs to the other row axis.
_sb_collect_data_lengths!(_acc, _x) = nothing
_sb_collect_data_lengths!(acc, x::NamedColumn) = begin
    p = parent(x)
    p isa DataColumn ? push!(acc, (name(x), length(parent(p)))) :
        _sb_collect_data_lengths!(acc, p)
    nothing
end
_sb_collect_data_lengths!(acc, x::ExprColumn) = begin
    getf(x) === (~) && return nothing
    for a in getargs(x); _sb_collect_data_lengths!(acc, a); end
    for v in values(getkwargs(x)); _sb_collect_data_lengths!(acc, v); end
    nothing
end
_sb_collect_data_lengths!(acc, x::Union{Tuple,AbstractVector}) = begin
    for a in x; _sb_collect_data_lengths!(acc, a); end
    nothing
end

# Substitute a bare symbol inside a do-block cell body. `:kw` names and the
# field half of `a.b` are syntactic positions, never value references, so they
# are left alone; a `QuoteNode` is opaque.
_sb_subst_sym(x, from::Symbol, to) = x === from ? to : x
_sb_subst_sym(x::QuoteNode, ::Symbol, _to) = x
_sb_subst_sym(x::Expr, from::Symbol, to) =
    if x.head === :. && length(x.args) == 2
        Expr(:., _sb_subst_sym(x.args[1], from, to), x.args[2])
    elseif x.head === :kw && length(x.args) == 2
        Expr(:kw, x.args[1], _sb_subst_sym(x.args[2], from, to))
    else
        Expr(x.head, (_sb_subst_sym(a, from, to) for a in x.args)...)
    end

# `ragged(x, group)` does ONE thing, whatever `x` is: take a flat row axis and a
# key naming each of its rows' subject, and produce the ragged per-subject view.
# This helper computes that grouping — the row indices — and enforces the
# contract that makes it meaningful. Realizing the view is the caller's job,
# because a parameter and a data column can only be grouped in different places:
#
#   - a LINEAR PREDICTOR is a Stan parameter, so it cannot be grouped Julia-side
#     at all. The plate takes this index column and the cell fancy-indexes into
#     the (unsliced, event-length) predictor: `lp[rows]`.
#   - a RAW DATA column is Julia data, so it is simply gathered here into a
#     `Vector{Vector{T}}` and registered — StanBlocks ingests that as a ragged
#     column natively, exactly like a hand-prepared per-subject view.
#
# Same grouping either way; only the realization differs.
#
# `g_vals` is the kernel's per-subject label column IN ROW ORDER, because
# `_sb_kernel_doblock!` keeps cells in row order (see the ORDER note there).
# Cell `i` therefore gets the rows of the flat frame whose group label equals
# `g_vals[i]` — a LABEL join, not a level-index join, so it cannot silently
# disagree with the row-ordered linear predictors sharing the same plate.
# Nothing requires a subject's rows to be CONTIGUOUS, and nothing is reordered.
function _sb_kernel_subject_values(raw, group::Symbol)
    raw isa AbstractVector || error(
        "sbimpl: kernel(...) subject grouping column `$group` must be a vector, " *
        "got $(typeof(raw)).")
    values = collect(_sb_group_values(raw))
    (!isempty(values) && !any(ismissing, values) &&
     length(unique(values)) == length(values)) || error(
        "sbimpl: kernel(...) needs pre-grouped per-subject data — `$group` must " *
        "list one non-missing unique subject per row. Repeated levels indicate " *
        "long-format data. Got $(values).")
    values
end

function _sb_kernel_ragged_partition(a_name::Symbol, grp_name::Symbol,
                                     ev_vals::AbstractVector,
                                     subject_vals::AbstractVector)
    (!isempty(ev_vals) && !any(ismissing, ev_vals)) || error(
        "sbimpl: kernel(...) `ragged($a_name, $grp_name)`: `$grp_name` must be a " *
        "non-empty column with no missing labels.")
    pos = Dict{Any,Int}()
    for (i, v) in enumerate(subject_vals); pos[v] = i; end
    rows = [Int[] for _ in eachindex(subject_vals)]
    unknown = Any[]
    for (r, v) in enumerate(ev_vals)
        i = get(pos, v, 0)
        i == 0 ? push!(unknown, v) : push!(rows[i], r)
    end
    isempty(unknown) || error(
        "sbimpl: kernel(...) `ragged($a_name, $grp_name)`: label(s) " *
        "$(unique(unknown)) in `$grp_name` name no subject in the kernel's " *
        "per-subject frame. Every event row must belong to a subject this " *
        "kernel walks.")
    rows
end

function _sb_kernel_ragged_rows(data, arg_col, grp_arg, g_vals)
    a_name = name(arg_col)
    decl = parent(arg_col)
    is_lp = decl isa ExprColumn && getf(decl) === (~)
    is_lp || decl isa DataColumn || error(
        "sbimpl: kernel(...) positional `ragged($a_name, …)`: `$a_name` must be either a ",
        "linear predictor declared in this @brm block (`$a_name ~ <terms>`) or a raw ",
        "data column; got $(typeof(decl)).")
    (grp_arg isa NamedColumn && parent(grp_arg) isa DataColumn) || error(
        "sbimpl: kernel(...) `ragged($a_name, …)` needs a raw data column naming the ",
        "subject of every row of `$a_name`'s row axis; got $(typeof(grp_arg)).")
    grp_name = name(grp_arg)
    ev_vals = collect(_sb_group_values(parent(parent(grp_arg))))
    n_ev = length(ev_vals)

    # The frame `$a_name` lives on must be the one `$grp_name` describes. For an
    # LP that means every data column its terms name; for a raw column, itself.
    lens = Tuple{Symbol,Int}[]
    if is_lp
        _sb_collect_data_lengths!(lens, getargs(decl)[2])
    else
        v = parent(decl)
        v isa AbstractVector{<:AbstractVector} && error(
            "sbimpl: kernel(...) `ragged($a_name, $grp_name)`: `$a_name` is ALREADY a ",
            "ragged per-subject column, so there is nothing to group. Pass it directly, ",
            "without `ragged(...)`.")
        push!(lens, (a_name, length(v)))
    end
    for (nm, L) in lens
        L == n_ev || error(
            "sbimpl: kernel(...) `ragged($a_name, $grp_name)`: `$a_name` is declared ",
            "over a $L-row axis (data column `$nm`) but `$grp_name` has $n_ev rows. ",
            "The grouping column must name the subject of EVERY row of ",
            "`$a_name`'s own frame.")
    end

    rows = _sb_kernel_ragged_partition(a_name, grp_name, ev_vals, g_vals)

    # Same treatment the per-subject group column gets: the labels join rows on
    # the Julia side and never enter the emitted Stan program, so drop the raw
    # column when Stan cannot type it (`Vector{String}`). Numeric labels stay,
    # in case the consumer also handed that column to the cell as real data.
    all(v -> v isa Real, ev_vals) || pop!(data, grp_name, nothing)
    (rows, is_lp)
end

function _sb_kernel_doblock!(stmts, data, target::Symbol, dcols, kw)
    haskey(kw, :by) && error(
        "sbimpl: kernel(...) do-block form no longer accepts `by=`; grouping is ",
        "derived from its per-subject linear-predictor positional args.")
    haskey(kw, :n_eta) && error(
        "sbimpl: kernel(...) do-block form no longer accepts `n_eta=`; declare ",
        "per-subject linear predictors with `(1 | ID | group)` terms and pass ",
        "those LPs positionally.")
    haskey(kw, :model) &&
        error("sbimpl: kernel(...) do-block form takes an inline do-block, not `model=`")
    haskey(kw, :obs) &&
        error("sbimpl: kernel(...) do-block form takes ordinary `~` statements, not `obs=`")

    lam = first(dcols)
    ptuple = lam.args[1]
    params = ptuple isa Symbol ? Symbol[ptuple] :
        (Meta.isexpr(ptuple, :tuple) && all(p -> p isa Symbol, ptuple.args) ?
            Symbol[ptuple.args...] :
            error("sbimpl: kernel(...) do-block params must be plain names (no types/defaults)"))
    body = lam.args[2]
    body_stmts = Meta.isexpr(body, :block) ? body.args : Any[body]

    # positional args (everything after the do-block); sliced in the plate.
    # Three admissible kinds:
    #
    #   - a RAW DATA column   -> register its vector in `data`;
    #   - a LATENT per-subject LINEAR PREDICTOR declared by an earlier formula
    #     statement (`log_CL ~ 1 + weight + (1|p|subject)`) -> emit NOTHING here.
    #     `_sb_linear_predictor!` has already assigned that name in the SLIC body,
    #     so the plate can slice it by name. Registering it as data would shadow
    #     the parameter with a constant (v2, decision `0dnesv9`).
    #     Such an LP needs no reshaping: the kernel contract is one row per
    #     subject, so an ordinary LP over that frame is ALREADY length n_subjects.
    #   - `ragged(x, group)` -> a SECONDARY row axis: `x` lives on some other frame
    #     (one row per dose event / op), `group` names the subject of each of its
    #     rows. The first two kinds are both length n_subjects and slice to a
    #     SCALAR per cell; this one slices to a ragged VECTOR. `x` may be a linear
    #     predictor OR a raw flat data column — it is the same grouping either
    #     way, and only the realization differs (see `_sb_kernel_ragged_rows`):
    #     a predictor is a parameter, so the plate takes an index column and the
    #     cell parameter is rewritten to `x[<those rows>]`; a data column is
    #     gathered Julia-side into a ragged column and registered under a derived
    #     name, leaving the flat original in place for any term that still needs it.
    dcol_names    = Symbol[]
    lp_cols       = Any[]
    ragged_specs  = Any[]
    for (i, c) in enumerate(dcols[2:end])
        if c isa ExprColumn && getf(c) === ragged
            length(getargs(c)) == 2 || error(
                "sbimpl: kernel(...) `ragged(...)` takes exactly two positional args — ",
                "the thing to group and its grouping column — got ",
                "$(length(getargs(c))).")
            arg_col, grp_arg = getargs(c)
            arg_col isa NamedColumn || error(
                "sbimpl: kernel(...) `ragged(...)`: the first argument must name a ",
                "linear predictor declared in this @brm block, or a raw data column; ",
                "got a bare $(typeof(arg_col)).")
            gath_sym = Symbol("kernel_", target, "_", name(arg_col), "_ragged")
            push!(ragged_specs, (i, arg_col, grp_arg, gath_sym))
            push!(dcol_names, gath_sym)
            continue
        end
        c isa NamedColumn || error(
            "sbimpl: kernel(...) positional args (after the do-block) must be a data ",
            "column, a per-subject linear predictor declared in this @brm block, or ",
            "a secondary-axis predictor/column wrapped as `ragged(x, group)`; got a ",
            "bare $(typeof(c)).")
        k = name(c)
        if parent(c) isa DataColumn
            v = parent(parent(c))
            data[k] = v
        else
            push!(lp_cols, c)
        end
        push!(dcol_names, k)
    end
    ndata = length(dcol_names)

    length(params) == ndata || error(
        "sbimpl: kernel(...) do-block has $(length(params)) params but expects ",
        "$ndata — exactly one per positional data/LP arg.")
    slice_params = copy(params)

    # n_subjects + long-format guard (pre-grouped: one row per subject).
    #
    # v2 (`0xuaz0k`): the GROUPING IS DERIVED from the per-subject LPs' ranef
    # bucket. `by=` only ever restated a fact the ranef already knew (the subject
    # count), and a kernel with no per-subject LP now fails loudly because there
    # is no authoritative grouping to derive.
    #
    # ORDER: cells stay in ROW order, deliberately. `_sb_linear_predictor!`
    # returns `popefs(X) + rows_dot_product(Z, b[group_idx,:])`, i.e. a
    # ROW-ordered vector of length n_rows — it has already mapped level -> row.
    # Reordering cells to the ranef's LEVEL order (which `_sb_level_index` sorts)
    # would therefore MIS-align the LP against its own subjects whenever the
    # labels are not sorted. The only thing that must hold is the bijection
    # below: one level per row.
    isempty(lp_cols) && error(
        "sbimpl: kernel(...) needs at least one per-subject linear-predictor ",
        "positional arg with a random-effect term; without one there is no grouping ",
        "to derive.")
    lp_buckets = [_sb_kernel_lp_bucket(c) for c in lp_cols]
    groups = unique([b[2] for b in lp_buckets])
    length(groups) == 1 || error(
        "sbimpl: kernel(...) per-subject linear predictors disagree on their ",
        "grouping — got groups $(groups) across $(Tuple(name(c) for c in lp_cols)). ",
        "LPs may use distinct `|ID|` buckets, but every LP handed to one kernel ",
        "must describe the same subjects.")
    group_col = first(lp_buckets)[3]

    # The labels identify groups to Julia callers but never enter the emitted
    # Stan program: only their count and row order do. Accept arbitrary unique
    # labels so a reusable generative-plan builder can rebuild on genuinely new
    # subject ids without app-local recoding to 1:n.
    g_vals = _sb_kernel_subject_values(
        parent(parent(group_col)), name(group_col))
    nsub = length(g_vals)
    # Arbitrary labels identify rows on the Julia side; the emitted Stan program
    # consumes only their integer index/count. The generic data prepass has
    # already materialised the raw column, so discard it when Stan cannot type it
    # (e.g. `Vector{String}`). Numeric group labels remain available in case the
    # consumer also passed that column to the cell as ordinary numeric data.
    all(v -> v isa Real, g_vals) || pop!(data, name(group_col), nothing)
    nsub_sym = Symbol("kernel_nsub_", target); data[nsub_sym] = nsub
    _sb_record_preproc!(data, nsub_sym,
        PreprocEntry(:kernel_subject_count, nothing, name(group_col), false))

    # `ragged(x, group)` positionals, now that the subject row order is known.
    #
    # A PREDICTOR cannot be grouped Julia-side, so the plate takes a per-subject
    # INDEX column in the arg's place and the cell parameter the consumer wrote is
    # rewritten to the fancy-index `x[<rows>]` wherever it appears in the body —
    # `exp(lF)` then reads the secondary-axis parameter directly instead of a
    # redundant per-cell copy of it.
    #
    # A DATA column is just gathered here, and registered under the DERIVED name
    # rather than its own. Wrapping it for the cell must not claim the flat
    # column: something else may name it on its own axis — `hsgp(log_dose)` over
    # the event frame, say — and that term registers it through its own path.
    for (i, arg_col, grp_arg, gath_sym) in ragged_specs
        rows, is_lp = _sb_kernel_ragged_rows(data, arg_col, grp_arg, g_vals)
        if is_lp
            data[gath_sym] = rows
            rows_param = Symbol("kernel_rows_", params[i])
            slice_params[i] = rows_param
            sliced = :($(name(arg_col))[$rows_param])
            body_stmts = Any[_sb_subst_sym(s, params[i], sliced) for s in body_stmts]
        else
            flat = parent(parent(arg_col))
            gathered = [flat[r] for r in rows]
            data[gath_sym] = gathered
        end
        _sb_record_preproc!(data, gath_sym, PreprocEntry(
            :kernel_ragged, (; is_lp),
            (name(arg_col), name(grp_arg), name(group_col)), false))
    end

    # Per-subject plate: slice params bind the data columns and already-emitted LPs;
    # the user's inline body (obs `~` statements and all) runs inside; its last
    # expression is collected.
    plate_body = Expr(:block, body_stmts...)
    plate_call = Expr(:call, :plate,
        Expr(:parameters, Expr(:kw, :outer, Expr(:tuple, nsub_sym))),
        dcol_names...)
    plate_do = Expr(:do, plate_call,
        Expr(:->, Expr(:tuple, slice_params...), plate_body))
    push!(stmts, :($target ~ $plate_do))
    :done
end

function _sb_submodel_rhs!(stmts, data, target::Symbol, ::typeof(kernel), rhs)
    dcols = getargs(rhs)
    kw = getkwargs(rhs)
    # do-block form (decision z9vkkf, User chose B):
    # `kernel(datacols..., per_subject_lps...) do slices..., lps... <body> end`.
    # @brm captures the do-block as a verbatim lambda first-arg (macro.jl `_x`);
    # dispatch to the inline-body emitter.
    if !isempty(dcols) && first(dcols) isa Expr && first(dcols).head == :->
        return _sb_kernel_doblock!(stmts, data, target, dcols, kw)
    end
    error(
        "sbimpl: kernel(...) only accepts the inline do-block form; the legacy ",
        "`model=`/`obs=` surface was removed. Declare per-subject linear predictors, ",
        "pass them positionally, and write observation `~` statements in the cell.")
end

# ---- structured-latent (group-block) declaration + emit API -----------------
#
# This is BRM's GENERAL structured-latent floor (KB todo ez6anl; user resolved
# `1o7se36` to build the general mechanism, not a one-off). A term author
# declares one-or-more STRUCTURED latent fields — matrix-valued per-group blocks,
# each with its OWN prior (correlated-normal, iid-normal, or an element-wise
# possibly-clamped / non-normal distribution) — by defining a
# `_sb_term_group_block` method. Prepass 2.5 reads the declaration, allocates one
# block per field, and threads the un-expanded n_groups×K matrices into the term
# at emit time via `_sb_emit_group_block_term!`. Basis evaluation stays in-term.
#
# Three consumers shape this API: hsgp(x, by=g) (per-group iid-normal HSGP weights,
# the deliverable), obs_scale (matrix<lower=0> element-wise Exponential, decision
# 10uz10q — proven via `sb_group_clamped_demo`, not yet wired), and future
# splines-by-group (per-group normal spline coefficients).
#
# Declaration shape — the 2-arg `_sb_term_group_block(f, call)` form receives the
# term CALL so a term can declare fields conditional on its actual arguments
# (hsgp only declares a block when called with `by=`). It returns EITHER:
#   (a) legacy single correlated-normal block:
#       (; n_per_group::Int, group_arg_pos|group_fn, group_fn_name)
#   (b) general field list:
#       (; fields = [ (; name, n_per_group, group, prior), ... ])
# where, per field:
#   name         — Symbol; unique per term; names the emitted block `b_<name>_<g>`
#   n_per_group  — K columns per group (the matrix is n_groups × K)
#   group        — (; arg_pos=Int) | (; kwarg=Symbol) | (; fn, fn_name=Symbol)
#                    how to find the grouping column (positional arg / kwarg /
#                    a synthesised computed column)
#   prior        — :correlated_normal (LKJ across columns, std_normal across
#                    groups; via ranef_correlated_draws)
#                | :iid_normal        (iid std_normal weights; via std_normal)
#                | (; dist, args, [lower], [upper])  element-wise prior over the
#                    matrix, reusing the scalar-prior dist-name table
#                    (`_sb_stan_dist_name`); `lower`/`upper` clamp the matrix.
#
# Use `import BayesianRegressionModels: _sb_term_group_block, _sb_emit_group_block_term!`
# when adding methods from a downstream module so the binding is extended.

# Default: no group block. Override for declaring terms. The 2-arg form defaults
# to the legacy type-only 1-arg form so existing declarations keep working.
_sb_term_group_block(_) = nothing
_sb_term_group_block(f, _call) = _sb_term_group_block(f)

# Toy term declaration: 2 correlated params per group, first arg is the group.
_sb_term_group_block(::typeof(sb_group_demo)) = (; n_per_group=2, group_arg_pos=1)

# Clamped / non-normal demo (the obs_scale shape, decision 10uz10q): a
# matrix<lower=0>[n_groups, 2] field with an element-wise Exponential prior.
_sb_term_group_block(::typeof(sb_group_clamped_demo)) = (; fields=[
    (; name=:obs_scale_demo, n_per_group=2, group=(; arg_pos=1),
       prior=(; dist=Exponential, args=(1.0,), lower=0.)),
])

# Normalize a term's declaration (legacy single-block NT or general `fields` NT)
# into a uniform Vector of field specs, or `nothing` if the term declares none.
_sb_structured_fields(::Nothing, _f) = nothing
function _sb_structured_fields(decl::NamedTuple, f)
    haskey(decl, :fields) && return decl.fields
    # Legacy single correlated-normal block -> one normalized field. The field
    # name is `nameof(f)` so the emitted block name (`b_<name>_<g>`) is
    # byte-for-byte identical to the pre-generalization floor.
    group = haskey(decl, :group_fn) ?
        (; fn=decl.group_fn, fn_name=decl.group_fn_name) :
        (; arg_pos=get(decl, :group_arg_pos, 1))
    [(; name=nameof(f), n_per_group=decl.n_per_group, group, prior=:correlated_normal)]
end

# Emit hook for structured-latent terms. `block_info` carries a `fields` map
# (field-name => (; block_name, idx_name, n_per_group)); for single-field terms
# the lone field's keys are ALSO spliced at top level so legacy consumers that
# destructure `(; block_name, idx_name)` keep working unchanged.
# Default errors so a term with a declaration but no emit method is caught early.
_sb_emit_group_block_term!(stmts, data, target, f, rhs_e, block_info) =
    error("sbimpl: `$(nameof(f))` declared a structured-latent block but has no ",
          "`_sb_emit_group_block_term!` method — define one.")

# Toy term emit: thread group_block + group_idx into sb_group_demo_slic.
function _sb_emit_group_block_term!(stmts, data, target, ::typeof(sb_group_demo),
                                     rhs_e, block_info)
    (; block_name, idx_name) = block_info
    push!(stmts, :($target ~ sb_group_demo_slic(;
        group_block=$block_name, group_idx=$idx_name)))
end

# Clamped-demo emit: thread the matrix<lower=0> exponential block into the slic.
function _sb_emit_group_block_term!(stmts, data, target, ::typeof(sb_group_clamped_demo),
                                     rhs_e, block_info)
    info = block_info.fields[:obs_scale_demo]
    push!(stmts, :($target ~ sb_group_clamped_demo_slic(;
        group_block=$(info.block_name), group_idx=$(info.idx_name))))
end

# Built-in prior families on a missing-LHS sampling statement (e.g.
# `coef_a ~ Horseshoe()`). Returns `true` if it consumed the binding,
# `false` otherwise (then `_sb_linear_predictor!` runs).
const _SB_HORSESHOE_KWARGS = (:local_scale, :global_scale)

function _sb_horseshoe_scale(target, key::Symbol, value)
    resolved = _sb_effect_prior_arg(value)
    (resolved isa Real && !(resolved isa Bool)) || error(
        "sbimpl: `$target ~ Horseshoe($key=...)` must be a numeric formula " *
        "constant, got $(repr(resolved))")
    isfinite(resolved) && resolved > 0 || error(
        "sbimpl: `$target ~ Horseshoe($key=...)` must be finite and strictly " *
        "positive, got $resolved")
    Float64(resolved)
end

function _sb_emit_prior!(stmts, target, ::Type{<:Horseshoe}, op)
    args = getargs(op)
    isempty(args) || error(
        "sbimpl: `$target ~ Horseshoe(...)` accepts no positional arguments; " *
        "use `local_scale=` and/or `global_scale=`")
    kw = getkwargs(op)
    unknown = Symbol[k for k in keys(kw) if !(k in _SB_HORSESHOE_KWARGS)]
    isempty(unknown) || error(
        "sbimpl: `$target ~ Horseshoe(...)` accepts only `local_scale` and " *
        "`global_scale` keywords, got $unknown")
    if isempty(kw)
        push!(stmts, :($target ~ _sb_horseshoe()))
    else
        local_scale = _sb_horseshoe_scale(
            target, :local_scale, get(kw, :local_scale, 1.0))
        global_scale = _sb_horseshoe_scale(
            target, :global_scale, get(kw, :global_scale, 1.0))
        push!(stmts, :($target ~ _sb_horseshoe_scaled(;
            local_scale=$local_scale, global_scale=$global_scale)))
    end
    true
end

# A Distributions.jl `VonMises(mu, kappa)` prior has moving parameter support
# `[mu - pi, mu + pi]`. Stan cannot faithfully declare that support when `mu`
# is itself a parameter, so do not silently lower it to an unconstrained native
# `von_mises` prior. Observation likelihoods are handled separately below.
_sb_emit_prior!(_, target, ::Type{<:VonMises}, _) = error(
    "sbimpl: `VonMises` is supported as an observation likelihood, but not as " *
    "a parameter prior: preserving Distributions.jl's moving support " *
    "`[mu - pi, mu + pi]` requires a Stan parameterization chosen explicitly. " *
    "Use a real-valued latent parameter and transform/wrap it deliberately.")
# Generic scalar prior via a Distributions.jl constructor on the RHS
# (e.g. `coef_a ~ Normal(0, 0.1)`). Reuses the same family -> Stan-name
# table the likelihood path uses (`_sb_stan_dist_name`), so adding a
# new family extends both paths in one place. Args are literals or
# already-bound parameter symbols (no data materialisation in prior
# context), so we walk them directly without dragging in the full
# `_sb_scalar_expr` reducer.
function _sb_emit_prior!(stmts, target, ::Type{D}, op) where {D <: Distribution}
    stan_name = _sb_stan_dist_name(D)
    isnothing(stan_name) && return false
    # Lower Julia-side formula nodes first, then normalize constructor defaults
    # and parameterizations.  Some translations (scale -> inverse scale,
    # probability -> odds) create Stan expressions, so applying `_sb_prior_arg`
    # afterwards would mistake those already-lowered Exprs for formula nodes.
    prior_args = map(_sb_prior_arg, getargs(op))
    arg_exprs = _sb_stan_dist_args(D, prior_args)
    rhs = Expr(:call, stan_name, arg_exprs...)
    kwargs = getkwargs(op)
    if !isempty(kwargs)
        unknown = Symbol[k for k in keys(kwargs) if !(k in (:lower, :upper))]
        isempty(unknown) || error(
            "sbimpl: `$target ~ $(nameof(D))(...)` accepts only `lower` and " *
            "`upper` declaration-bound keywords, got $unknown")
        # A Stan declaration bound omits the truncation normalizer. That is
        # exact up to a parameter-independent constant only when the family
        # hyperparameters are fixed, so reject hierarchical bounds here.
        all(a -> a isa Real && !(a isa Bool), prior_args) || error(
            "sbimpl: bounded scalar prior `$target ~ $(nameof(D))(...)` requires " *
            "numeric formula-constant distribution arguments; a bound with " *
            "sampled hyperparameters needs an explicit normalized density")
        bounds = Dict{Symbol,Float64}()
        for key in (:lower, :upper)
            haskey(kwargs, key) || continue
            value = _sb_prior_arg(kwargs[key])
            (value isa Real && !(value isa Bool) && isfinite(value)) || error(
                "sbimpl: `$target ~ $(nameof(D))(...; $key=...)` requires a " *
                "finite numeric formula constant, got $(repr(value))")
            bounds[key] = Float64(value)
        end
        haskey(bounds, :lower) && haskey(bounds, :upper) &&
            bounds[:lower] >= bounds[:upper] && error(
                "sbimpl: bounded scalar prior `$target` requires `lower < upper`, " *
                "got $(bounds[:lower]) >= $(bounds[:upper])")
        insert!(rhs.args, 2, Expr(:parameters,
            (Expr(:kw, key, bounds[key]) for key in (:lower, :upper)
             if haskey(bounds, key))...))
    end
    push!(stmts, Expr(:call, :~, target, rhs))
    true
end
_sb_emit_prior!(_, _, _, _) = false

"""
    _sb_emit_vector_prior!(stmts, data, target, f, op)

sbimpl extension hook for a VECTOR-valued parameter prior on a non-data LHS
(`diet_share ~ Dirichlet(3, 1.0)`). Return `true` to claim the binding, `false`
to fall through to the scalar `_sb_emit_prior!` seam and then to the
linear-predictor path.

It exists alongside `_sb_emit_prior!` rather than inside it because a
multivariate family's *shape* comes from its hyperparameters: Stan sizes
`simplex[K]` from the Dirichlet concentration, so the concentration has to be
registered in `data` — which the four-argument scalar seam has no access to.
Keeping the two seams separate also leaves every existing downstream
`_sb_emit_prior!` method's arity untouched.

**Use `import BayesianRegressionModels: _sb_emit_vector_prior!`** (not `using`)
when adding methods from a downstream module.
"""
_sb_emit_vector_prior!(_stmts, _data, _target, _f, _op) = false

function _sb_lkj_covariance_factor_spec(target::Symbol, op::ExprColumn)
    args = getargs(op)
    length(args) == 1 || error(
        "sbimpl: `$target ~ LKJCovarianceFactor(K; ...)` needs exactly one " *
        "dimension argument, got $(length(args))")
    K = only(args)
    K isa Integer && !(K isa Bool) && K >= 1 || error(
        "sbimpl: `$target ~ LKJCovarianceFactor(K; ...)` needs an integer " *
        "dimension >= 1, got $(repr(K))")

    kw = getkwargs(op)
    unknown = Symbol[k for k in keys(kw) if !(k in (:scale_prior, :shape))]
    isempty(unknown) || error(
        "sbimpl: `$target ~ LKJCovarianceFactor(...)` accepts only " *
        "`scale_prior` and `shape`, got $unknown")

    scale_prior = get(kw, :scale_prior, ExprColumn(Exponential, 1.0))
    scale_prior isa ExprColumn || error(
        "sbimpl: `scale_prior` for `$target` must be a positive-support " *
        "distribution call, got $(typeof(scale_prior))")
    scale_family = _as_distribution_type(getf(scale_prior))
    !isnothing(scale_family) && scale_family <: Exponential || error(
        "sbimpl: `scale_prior` for `$target` currently supports " *
        "`Exponential(scale)` only, which guarantees positive residual " *
        "scales; got `$(getf(scale_prior))`")
    isempty(getkwargs(scale_prior)) || error(
        "sbimpl: `Exponential` scale_prior for `$target` accepts no keywords")
    scale_args = getargs(scale_prior)
    length(scale_args) == 1 || error(
        "sbimpl: `Exponential` scale_prior for `$target` needs exactly one " *
        "Julia scale argument")
    scale = only(scale_args)
    scale isa Real && !(scale isa Bool) && isfinite(scale) && scale > 0 || error(
        "sbimpl: `Exponential` scale_prior for `$target` needs a finite, " *
        "strictly positive scale, got $(repr(scale))")

    shape = get(kw, :shape, 1.0)
    shape isa Real && !(shape isa Bool) && isfinite(shape) && shape > 0 || error(
        "sbimpl: LKJ `shape` for `$target` must be finite and strictly " *
        "positive, got $(repr(shape))")
    (; K=Int(K), scale=Float64(scale), shape=Float64(shape))
end

function _sb_validate_covariance_factor_names(brmi::BRMI)
    operation_names = Set{Symbol}(keys(brmi.operations))
    for (target, value) in pairs(brmi.operations)
        column = _as_named_column(value)
        isnothing(column) && continue
        op = parent(column)
        op isa ExprColumn && getf(op) === (~) || continue
        _, rhs = getargs(op, 2)
        rhs_e = _as_expr_column(rhs)
        isnothing(rhs_e) && continue
        getf(rhs_e) === LKJCovarianceFactor || continue
        reserved = (
            Symbol(target, :_n),
            Symbol(target, :_scales),
            Symbol(target, :_L_corr),
        )
        collision = findfirst(in(operation_names), reserved)
        isnothing(collision) || error(
            "sbimpl: `$target ~ LKJCovarianceFactor(...)` reserves emitted " *
            "binding `$(reserved[collision])`, but the formula also declares " *
            "or references that name. Rename `$target` or the colliding name.")
    end
    nothing
end

function _sb_emit_vector_prior!(stmts, data, target,
                                ::typeof(LKJCovarianceFactor), op)
    spec = _sb_lkj_covariance_factor_spec(target, op)
    n_key = Symbol(target, :_n)
    scale_name = Symbol(target, :_scales)
    corr_name = Symbol(target, :_L_corr)
    haskey(data, n_key) && error(
        "sbimpl: `$target ~ LKJCovarianceFactor(...)` reserves data key " *
        "`$n_key`, but that name is already used")
    data[n_key] = spec.K

    # Julia's Exponential constructor takes SCALE; Stan's exponential family
    # takes RATE. Keep the same normalization used by ordinary BRM priors.
    rate = inv(spec.scale)
    push!(stmts, Expr(:call, :~, scale_name,
        Expr(:call, :exponential,
             Expr(:parameters, Expr(:kw, :n, n_key)), rate)))
    corr_lhs = Expr(:(::), corr_name,
                    Expr(:ref, :cholesky_factor_corr, n_key))
    push!(stmts, Expr(:call, :~, corr_lhs,
                      Expr(:call, :lkj_corr_cholesky, spec.shape)))
    push!(stmts, :($target = diag_pre_multiply($scale_name, $corr_name)))
    true
end

# `s ~ Dirichlet(alpha)` / `s ~ Dirichlet(K, a)` declares a SIMPLEX PARAMETER.
#
# This is the vector-valued sibling of the scalar-parameter prior path: the LHS
# is a non-data name, so the statement declares a new parameter rather than an
# observation. StanBlocks types the LHS of `~ dirichlet(alpha)` as
# `simplex[dims(alpha)[1]]` (`slic_stan/builtin.jl`, `dirichlet_lpdf(w::simplex[n],
# alpha::vector[n])`), so the declaration needs no size annotation of its own.
#
# The concentration goes into `data` under `<target>_alpha`, exactly as the R2D2
# prepass does for its explained-variance shares (`_sb_emit_r2d2_params!`). That
# keeps the emitted program's simplex size a data-block int rather than an
# inlined literal, and it is the shape whose stanc/BridgeStan acceptance R2D2
# already established.
#
# `Dirichlet` is deliberately NOT added to `_sb_stan_dist_name`: that table is
# shared with the observation-likelihood path, and a simplex-valued RESPONSE has
# no density/pointwise/predictive support here. A data-backed LHS therefore still
# reaches the loud "no `_sb_stan_dist_name` entry" family error.
function _sb_emit_vector_prior!(stmts, data, target, ::Type{<:Dirichlet}, op)
    alpha = _sb_dirichlet_alpha(target, getargs(op), getkwargs(op))
    alpha_name = Symbol(target, :_alpha)
    haskey(data, alpha_name) && error(
        "sbimpl: `$target ~ Dirichlet(...)` needs the data name `$alpha_name` for ",
        "its concentration, but that name is already taken. Rename the parameter.")
    data[alpha_name] = alpha
    push!(stmts, :($target ~ dirichlet($alpha_name)))
    true
end

# Both Distributions.jl constructor forms, and nothing invented on top of them:
# `Dirichlet(alpha::AbstractVector)` and the symmetric `Dirichlet(K::Int, a::Real)`.
# Concentrations are hyperparameters, so they must be literals here for the same
# reason `_sb_prior_arg` rejects data-backed scalars: a Dirichlet whose alpha is
# itself a parameter is a different model that needs its own emission.
function _sb_dirichlet_alpha(target, args, kwargs)
    isempty(kwargs) || error(
        "sbimpl: `$target ~ Dirichlet(...)` takes no keywords, got ",
        "$(collect(keys(kwargs))).")
    alpha = if length(args) == 1
        a = only(args)
        a isa AbstractVector{<:Real} || error(
            "sbimpl: `$target ~ Dirichlet(alpha)` needs a numeric concentration ",
            "vector literal, got $(typeof(a)). Concentrations are hyperparameters: ",
            "use `Dirichlet([a1, a2, ...])` or the symmetric `Dirichlet(K, a)`.",
            _sb_dirichlet_column_hint(a))
        collect(Float64, a)
    elseif length(args) == 2
        K, a = args
        (K isa Integer && a isa Real) || error(
            "sbimpl: symmetric `$target ~ Dirichlet(K, a)` needs an integer ",
            "dimension and a real concentration, got ($(typeof(K)), $(typeof(a))).",
            _sb_dirichlet_column_hint(K), _sb_dirichlet_column_hint(a))
        fill(Float64(a), K)
    else
        error("sbimpl: `$target ~ Dirichlet(...)` takes either a concentration ",
              "vector `Dirichlet(alpha)` or a symmetric `Dirichlet(K, a)`; got ",
              "$(length(args)) positional arguments.")
    end
    length(alpha) >= 1 || error(
        "sbimpl: `$target ~ Dirichlet(...)` needs dimension >= 1, got ",
        "$(length(alpha)). A zero-element Dirichlet is undefined (no simplex).")
    # A one-element simplex (`Dirichlet([a])` / `Dirichlet(1, a)`) is allowed and
    # emits `simplex[1] $target; $target ~ dirichlet(...)` — deterministically
    # `[1.0]` with zero sampler dimensions, so it costs nothing and keeps ONE
    # uniform emission (no `dimension == 1 ? constant : simplex` branch).
    all(x -> isfinite(x) && x > 0, alpha) || error(
        "sbimpl: `$target ~ Dirichlet(...)` concentrations must be finite and ",
        "strictly positive, got $alpha.")
    alpha
end

# `@brm` is a MACRO over the formula block, so a bare Julia symbol on the RHS is
# parsed as a formula LOCAL (`@getproperty` falls back to a `NamedColumn`) rather
# than interpolated -- and `$` cannot rescue it, since `$` outside a quote is a
# Julia syntax error the macro never gets to see. `Dirichlet(3, alpha)` with a
# captured `alpha` therefore arrives here as a column carrier, and the bare type
# name in the message above reads like a BRM bug instead of the spelling trap it
# is. Name the trap when, and only when, that is what happened.
_sb_dirichlet_column_hint(_) = ""
_sb_dirichlet_column_hint(x::AbstractColumn) = string(
    " `$(_sb_dirichlet_arg_label(x))` is a formula-local column here, not the ",
    "surrounding Julia value: `@brm` is a macro over the block, so a bare symbol ",
    "is never interpolated (and there is no `\$` escape). Spell the concentration ",
    "as a literal.")

_sb_dirichlet_arg_label(x::AbstractColumn) = x isa NamedColumn ? name(x) : "that argument"

# `LocationScale(mu, sigma, base)` is Distributions.jl's generic
# location-scale wrapper (alias for `AffineDistribution`). Useful
# specifically for distributions that don't take location/scale args
# natively -- in BRM today that's `TDist`. Compose:
#   coef ~ LocationScale(0, 0.1, TDist(3))  =>  coef ~ student_t(3, 0, 1) re-args'd to (3, 0, 0.1)
# i.e. the same effect as if Stan's `student_t` took (nu, mu, sigma).
# Unwraps the base distribution and reuses the existing per-family
# dispatch with the LocationScale's mu/sigma threaded in.
function _sb_emit_prior!(stmts, target, ::Type{<:LocationScale}, op)
    mu, sigma, base = getargs(op, 3)
    base_e = _as_expr_column(base)
    isnothing(base_e) && error(
        "sbimpl: `LocationScale(...)` third arg must be a distribution call, got $(typeof(base))")
    base_fam = getf(base_e)
    isnothing(_as_distribution_type(base_fam)) && error(
        "sbimpl: `LocationScale` base must be a Distribution type, got $(base_fam)")
    base = base_e
    stan_name = _sb_stan_dist_name(base_fam)
    isnothing(stan_name) && error(
        "sbimpl: `LocationScale` over `$(base_fam)` -- no Stan-name mapping ",
        "for the base. Add a `_sb_stan_dist_name(::Type{<:$(base_fam)})` entry.")
    base_args = _sb_stan_dist_args(base_fam, map(_sb_prior_arg, getargs(base)))
    # Stan's standard univariate dists put (location, scale) at
    # positions 2-3 (e.g. `student_t(nu, mu, sigma)`). The arg-shape
    # transform on the base already pads to that layout (TDist(nu) ->
    # (nu, 0, 1)). Replace those defaults with the LocationScale wrapper's
    # values; if the base call already supplied non-default location/scale,
    # composing them is the user's job (LocationScale(0.1, 2.0, Normal(5, 0.3))
    # would otherwise silently overwrite the inner 5 / 0.3).
    length(base_args) >= 3 || error(
        "sbimpl: `LocationScale` over `$(base_fam)`: base lowers to ",
        "$(length(base_args)) Stan args, need >= 3 for location/scale slots.")
    composed = (base_args[1], _sb_prior_arg(mu), _sb_prior_arg(sigma),
                base_args[4:end]...)
    arg_exprs = composed
    push!(stmts, Expr(:call, :~, target, Expr(:call, stan_name, arg_exprs...)))
    true
end

# Prior-arg lowering. Literals pass through; bare-Symbol references
# (already-bound parameter names) pass through; nested expressions
# recursively lower. Anything else (e.g. a data-column NamedColumn) is
# rejected -- prior args should be literal hyperparameters or
# already-declared scalar params.
_sb_prior_arg(x::Real) = x
_sb_prior_arg(x::Symbol) = x
_sb_prior_arg(x::NamedColumn) = _sb_prior_arg_named(x, parent(x))
_sb_prior_arg_named(x, ::MissingColumn) = name(x)
# After a formula-local scalar prior has been parsed, later references carry
# that declaration as their backing expression rather than a MissingColumn.
# Recognise only scalar-prior RHS families here; data and vector parameters keep
# the existing rejection path.
function _sb_prior_arg_named(x, op::ExprColumn{typeof(~)})
    lhs_raw, rhs_raw = getargs(op, 2)
    lhs = _as_named_column(lhs_raw)
    rhs = _as_expr_column(rhs_raw)
    if !isnothing(lhs) && parent(lhs) isa MissingColumn &&
       name(lhs) === name(x) && !isnothing(rhs) &&
       _sb_is_scalar_prior_family(getf(rhs))
        return name(x)
    end
    error(_sb_prior_arg_backing_error(x, op))
end
_sb_is_scalar_prior_family(::typeof(Horseshoe)) = true
_sb_is_scalar_prior_family(::Type{D}) where {D<:LocationScale} = true
_sb_is_scalar_prior_family(::Type{D}) where {D<:Distribution} =
    !(D <: VonMises) && !isnothing(_sb_stan_dist_name(D))
_sb_is_scalar_prior_family(_) = false
_sb_prior_arg_named(x, d) = error(
    _sb_prior_arg_backing_error(x, d))
_sb_prior_arg_backing_error(x, d) = string(
    "sbimpl: prior arg `$(name(x))` is backed by $(typeof(d)); ",
    "prior args must be literals or already-declared scalar parameters.")
_sb_prior_arg(x::ExprColumn) = Expr(:call, getf(x), map(_sb_prior_arg, getargs(x))...)
_sb_prior_arg(x) = error("sbimpl: unsupported prior-arg shape $(typeof(x))")

# LHS backed by real data => this is a likelihood. Record the observed values
# under the formula name in `data` and emit `key ~ dist(args...)`.
_sb_sampling!(stmts, data, key, lhs::NamedColumn, rhs; id_lookup=_sb_empty_id_lookup(), obs_n=nothing, cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(), group_block_lookup=Dict(), effect_overrides=Dict{Symbol,Any}(), r2d2=_sb_empty_r2d2()) =
    _sb_sampling_backed!(stmts, data, key, parent(lhs), rhs; id_lookup, obs_n,
                         cv_groups, centered_groups, group_block_lookup,
                         effect_overrides, r2d2)

function _sb_joint_factor_reference(target::Symbol, factor, K::Int)
    factor isa NamedColumn || error(
        "sbimpl: `MvNormalCholesky` factor for joint response `$target` must " *
        "name an earlier `LKJCovarianceFactor` declaration, got " *
        "$(typeof(factor))")
    declaration = parent(factor)
    declaration isa ExprColumn && getf(declaration) === (~) || error(
        "sbimpl: joint-response factor `$(name(factor))` must be declared " *
        "before `$target` with `~ LKJCovarianceFactor(...)`")
    lhs, rhs = getargs(declaration, 2)
    lhs isa NamedColumn && parent(lhs) isa MissingColumn || error(
        "sbimpl: joint-response factor `$(name(factor))` must be a sampled " *
        "parameter, not observed data")
    rhs isa ExprColumn && getf(rhs) === LKJCovarianceFactor || error(
        "sbimpl: joint-response factor `$(name(factor))` is not backed by " *
        "`LKJCovarianceFactor(...)`")
    spec = _sb_lkj_covariance_factor_spec(name(factor), rhs)
    spec.K == K || error(
        "sbimpl: joint response `$target` has $K ordered outcomes but factor " *
        "`$(name(factor))` has dimension $(spec.K)")
    name(factor)
end

function _sb_joint_mean_reference(target::Symbol, outcome::Symbol, mean_arg,
                                  data, nobs::Int)
    mean_arg isa NamedColumn || error(
        "sbimpl: mean for joint outcome `$outcome` in `$target` must be a " *
        "named, row-aligned predictor (for example `mu_$outcome ~ ...`), got " *
        "$(typeof(mean_arg))")
    backing = parent(mean_arg)
    if backing isa DataColumn
        values = parent(backing)
        values isa AbstractVector && length(values) == nobs || error(
            "sbimpl: data-backed mean `$(name(mean_arg))` for joint outcome " *
            "`$outcome` must have $nobs rows")
    elseif backing isa ExprColumn && getf(backing) === (~)
        lhs, rhs = getargs(backing, 2)
        rhs_e = _as_expr_column(rhs)
        if lhs isa NamedColumn && parent(lhs) isa MissingColumn &&
           !isnothing(rhs_e) &&
           (getf(rhs_e) === Horseshoe || getf(rhs_e) === LKJCovarianceFactor ||
            !isnothing(_as_distribution_type(getf(rhs_e))))
            error(
                "sbimpl: `$(name(mean_arg))` is a scalar prior declaration, " *
                "not a row-aligned mean for joint outcome `$outcome`. Declare " *
                "a predictor such as `$(name(mean_arg)) ~ 1 + ...` first.")
        end
    elseif !(backing isa ExprColumn)
        error(
            "sbimpl: mean `$(name(mean_arg))` for joint outcome `$outcome` " *
            "is not backed by data or a model declaration")
    end

    # Validate every ordinary raw row-axis source reachable from this mean.
    # `kernel(...)` is intentionally opaque here: its grouped/ragged inputs may
    # live on several axes, and StanBlocks owns the returned cell shape. Named
    # references to another declaration are opaque for the same reason.
    source_lengths = Pair{Symbol,Int}[]
    if backing isa DataColumn
        push!(source_lengths, name(mean_arg) => length(parent(backing)))
    elseif backing isa ExprColumn && getf(backing) in ((~), assign)
        _, mean_rhs = getargs(backing, 2)
        if !(mean_rhs isa ExprColumn && getf(mean_rhs) === kernel)
            found = Tuple{Symbol,Int}[]
            _sb_collect_data_lengths!(found, mean_rhs)
            append!(source_lengths, (source => len for (source, len) in found))
        end
    end
    unique!(source_lengths)
    for (source, len) in source_lengths
        len == nobs || error(
            "sbimpl: mean `$(name(mean_arg))` for joint outcome `$outcome` " *
            "uses row-axis source `$source` with $len rows, but the aligned " *
            "outcomes have $nobs rows")
    end
    (; expression=_sb_scalar_expr(mean_arg, data),
       sources=Tuple(first.(source_lengths)))
end

function _sb_sampling!(stmts, data, key, lhs::JointResponseColumn, rhs;
                       id_lookup=_sb_empty_id_lookup(), kwargs...)
    rhs isa ExprColumn && getf(rhs) === MvNormalCholesky || error(
        "sbimpl: vector response $(collect(joint_response_names(lhs))) supports " *
        "the explicit joint family `MvNormalCholesky(means, factor)`; got " *
        "$(rhs isa ExprColumn ? getf(rhs) : typeof(rhs))")
    isempty(getkwargs(rhs)) || error(
        "sbimpl: `MvNormalCholesky` accepts no keywords")
    args = getargs(rhs)
    length(args) == 2 || error(
        "sbimpl: `MvNormalCholesky(means, factor)` needs exactly two arguments")
    means, factor = args
    means isa AbstractVector || error(
        "sbimpl: `MvNormalCholesky` means must use vector syntax " *
        "`[mu1, mu2, ...]`, got $(typeof(means))")

    outcomes = joint_response_names(lhs)
    K = length(outcomes)
    length(means) == K || error(
        "sbimpl: joint response $(collect(outcomes)) has $K outcomes but " *
        "`MvNormalCholesky` received $(length(means)) means")
    data_key = _joint_response_data_key(lhs)
    n_key = _joint_response_n_key(lhs)
    observed = get(data, data_key, nothing)
    observed isa AbstractVector{<:AbstractVector{<:Real}} || error(
        "sbimpl: internal joint-response data `$data_key` was not packed " *
        "as one ordered vector per aligned row")
    all(row -> length(row) == K, observed) || error(
        "sbimpl: internal joint-response data `$data_key` has a row whose " *
        "outcome width differs from $K")
    nobs = length(observed)
    get(data, n_key, nothing) == nobs || error(
        "sbimpl: internal joint-response row count `$n_key` disagrees with data")

    factor_name = _sb_joint_factor_reference(key, factor, K)
    mean_specs = ntuple(i -> _sb_joint_mean_reference(
        key, outcomes[i], means[i], data, nobs), K)
    mean_exprs = map(spec -> spec.expression, mean_specs)
    mean_sources = Tuple(unique!(Symbol[
        source for spec in mean_specs for source in spec.sources]))

    # Record one row-grouped observation input with all source columns in stable
    # formula order. Replay regenerates the row vectors and their row count.
    _sb_record_preproc!(data, data_key, PreprocEntry(
        :joint_response, (; n_key, outcomes, mean_sources), outcomes, true))

    # Build a row-grouped mean collection first, then apply the likelihood at
    # top level. StanBlocks' ragged-observation path owns the full observation
    # triad: a model density, a flat predictive draw with row segments, and one
    # aggregate pointwise likelihood scalar per row. A dense observation slice
    # inside `plate` is intentionally predictive-only under its general contract.
    mean_key = Symbol(key, :_means)
    observed_cell = Symbol(key, :_observed_cell)
    mean_cells = ntuple(i -> Symbol(key, :_mean_cell_, i), K)
    mean_cell = Symbol(key, :_mean_vector)
    raw_mean_vector = Expr(:vect, mean_cells...)
    # Tie the result's dimension to the ragged observation cell. The zero term
    # is algebraically inert; its symbolic size is what makes the collected
    # plate result a RaggedVector rather than a dense matrix.
    sized_mean_vector = Expr(:call, :+,
        Expr(:call, :.*, 0.0, observed_cell), raw_mean_vector)
    cell_body = Expr(:block,
        Expr(:(=), mean_cell, sized_mean_vector),
        mean_cell)
    plate_call = Expr(:call, :plate,
        Expr(:parameters, Expr(:kw, :outer, Expr(:tuple, n_key))),
        data_key, mean_exprs...)
    plate_do = Expr(:do, plate_call,
        Expr(:->, Expr(:tuple, observed_cell, mean_cells...), cell_body))
    push!(stmts, Expr(:call, :~, mean_key, plate_do))
    push!(stmts, Expr(:call, :~, data_key,
        Expr(:call, :multi_normal_cholesky, mean_key, factor_name)))
    nothing
end

# `effect(...) ~ Distribution(...)` is metadata consumed by constructor prepasses;
# it deliberately emits no independent parameter or likelihood statement.
_sb_sampling!(_stmts, _data, _key, _lhs::ExprColumn{typeof(effect)}, _rhs;
              kwargs...) = nothing

_sb_sampling_backed!(stmts, data, key, backing::DataColumn, rhs; id_lookup, kwargs...) = begin
    data[key] = _brm_data_vec(key, parent(backing))
    _sb_likelihood!(stmts, key, rhs, data)
end

# Find the per-subject group column of every `kernel(...)` result referenced by
# a likelihood RHS. The formula node retains the producer declaration on the
# referenced NamedColumn, so the observation boundary can align a flat response
# to the kernel's ROW-ordered subject axis without guessing from first-seen or
# sorted labels.
_sb_ragged_rhs_kernel_groups!(_acc, _x) = nothing
function _sb_ragged_rhs_kernel_groups!(acc, x::NamedColumn)
    decl = parent(x)
    decl isa ExprColumn && getf(decl) === (~) || return nothing
    _, producer_rhs = getargs(decl, 2)
    producer_rhs isa ExprColumn && getf(producer_rhs) === kernel || return nothing

    buckets = Any[]
    for arg in getargs(producer_rhs)
        arg isa NamedColumn || continue
        arg_decl = parent(arg)
        arg_decl isa ExprColumn && getf(arg_decl) === (~) || continue
        push!(buckets, _sb_kernel_lp_bucket(arg))
    end
    isempty(buckets) && return nothing
    groups = unique(b[2] for b in buckets)
    length(groups) == 1 || error(
        "sbimpl: kernel result `$(name(x))` has no single subject grouping; " *
        "its per-subject predictors name groups $(collect(groups)).")
    push!(acc, (name(x), first(buckets)[3]))
    nothing
end
function _sb_ragged_rhs_kernel_groups!(acc, x::ExprColumn)
    foreach(a -> _sb_ragged_rhs_kernel_groups!(acc, a), getargs(x))
    foreach(v -> _sb_ragged_rhs_kernel_groups!(acc, v), values(getkwargs(x)))
    nothing
end

# Partition flat observation rows into per-subject groups using the kernel's
# ROW-ordered subject axis. Shared by `_sb_ragged_lhs_layout` (emission) and the
# `:ragged_gather` reprocess handler so a replayed frame is aligned to the
# kernel exactly as it was at fit time.
function _sb_ragged_group_rows(key::Symbol, group::Symbol,
                               group_values::AbstractVector,
                               subject_values::AbstractVector)
    positions = Dict{Any,Int}(v => i for (i, v) in enumerate(subject_values))
    rows = [Int[] for _ in subject_values]
    unknown = Any[]
    for (row, label) in enumerate(group_values)
        i = get(positions, label, 0)
        i == 0 ? push!(unknown, label) : push!(rows[i], row)
    end
    isempty(unknown) || error(
        "sbimpl: `ragged($key, $group)` contains label(s) " *
        "$(unique(unknown)) that name no subject in the referenced kernel.")
    rows
end

function _sb_ragged_lhs_layout(key::Symbol, lhs::ExprColumn, rhs)
    args = getargs(lhs)
    length(args) == 2 || error(
        "sbimpl: `ragged(...)` observation LHS takes exactly two arguments — " *
        "the flat response and its grouping column — got $(length(args)).")
    response, group = args
    response isa NamedColumn && parent(response) isa DataColumn || error(
        "sbimpl: `ragged(...)` observation LHS needs a flat data-backed response " *
        "as its first argument; got $(typeof(response)).")
    name(response) === key || error(
        "sbimpl: `ragged(...)` observation LHS is keyed as `$key` but names " *
        "response `$(name(response))`.")
    group isa NamedColumn && parent(group) isa DataColumn || error(
        "sbimpl: `ragged($key, ...)` observation LHS needs a raw data grouping " *
        "column as its second argument; got $(typeof(group)).")

    raw = _brm_data_vec(key, parent(parent(response)))
    raw isa AbstractVector{<:AbstractVector} && error(
        "sbimpl: `ragged($key, $(name(group)))` observation LHS received an " *
        "ALREADY-ragged response; write `$key ~ <family>(...)` directly.")
    group_values = collect(parent(parent(group)))
    length(group_values) == length(raw) || error(
        "sbimpl: `ragged($key, $(name(group)))` has $(length(raw)) response rows " *
        "but $(length(group_values)) grouping rows. The grouping column must name " *
        "the subject of every response row.")
    (!isempty(group_values) && !any(ismissing, group_values)) || error(
        "sbimpl: `ragged($key, $(name(group)))` needs a non-empty grouping column " *
        "with no missing labels.")

    producers = Tuple{Symbol,Any}[]
    _sb_ragged_rhs_kernel_groups!(producers, rhs)
    isempty(producers) && error(
        "sbimpl: `ragged($key, $(name(group))) ~ ...` needs a `kernel(...)` result " *
        "on the likelihood RHS so BRM can align groups to the kernel's subject " *
        "row order without guessing.")
    subject_values = collect(parent(parent(first(producers)[2])))
    for (producer, subject_col) in producers[2:end]
        candidate = collect(parent(parent(subject_col)))
        candidate == subject_values || error(
            "sbimpl: `ragged($key, $(name(group)))` combines kernel result " *
            "`$producer` with a different subject row order. Every kernel result " *
            "in one likelihood must describe the same subjects in the same order.")
    end
    (!isempty(subject_values) && !any(ismissing, subject_values) &&
     length(unique(subject_values)) == length(subject_values)) || error(
        "sbimpl: `ragged($key, $(name(group)))` needs the referenced kernel's " *
        "subject column to contain one non-missing unique label per row; got " *
        "$(subject_values).")

    rows = _sb_ragged_group_rows(key, name(group), group_values, subject_values)
    (; values=[raw[r] for r in rows], rows, nrows=length(raw),
       group_col=name(group), subject_col=name(first(producers)[2]))
end

# A data-backed bound on a ragged formula-LHS lives on the same flat observed
# frame as the response. Group it through the LHS's already-validated row map,
# but bind it under a likelihood-local derived key: the original flat column may
# still be used by another formula term on its native axis.
function _sb_ragged_bound(data, key::Symbol, label::Symbol, bound, layout)
    bound isa NamedColumn && parent(bound) isa DataColumn || return bound
    raw = _brm_data_vec(name(bound), parent(parent(bound)))
    grouped = if raw isa AbstractVector{<:AbstractVector}
        raw
    else
        length(raw) == layout.nrows || error(
            "sbimpl: `$(key)` $label bound `$(name(bound))` has $(length(raw)) " *
            "rows but the flat response has $(layout.nrows)")
        [raw[r] for r in layout.rows]
    end
    derived = Symbol(key, :_, label, :_, name(bound), :_ragged)
    if haskey(data, derived)
        data[derived] == grouped || error(
            "sbimpl: derived ragged-bound data key `$derived` collides with " *
            "different observed data")
    else
        data[derived] = grouped
        # Same provenance the gathered response carries (`_sb_sampling!`): a
        # data-backed censoring/truncation bound is re-gathered from its flat
        # column + the shared grouping on `reprocess`.
        _sb_record_preproc!(data, derived, PreprocEntry(
            :ragged_gather,
            (; group_col=layout.group_col, subject_col=layout.subject_col),
            name(bound), true))
    end
    NamedColumn(derived, DataColumn(grouped))
end

function _sb_ragged_likelihood_rhs(data, key::Symbol, rhs::ExprColumn, layout)
    f = getf(rhs)
    if f === truncated || f === censored
        lower, upper = _sb_wrapper_bounds(f, getargs(rhs), getkwargs(rhs))
        lower = _sb_ragged_bound(data, key, :lower, lower, layout)
        upper = _sb_ragged_bound(data, key, :upper, upper, layout)
        return ExprColumn(f, first(getargs(rhs)); lower, upper)
    elseif f === interval_censored && length(getargs(rhs)) == 1 &&
           keys(getkwargs(rhs)) == (:upper,)
        upper = _sb_ragged_bound(
            data, key, :upper, getkwargs(rhs).upper, layout)
        return ExprColumn(f, first(getargs(rhs)); upper)
    end
    rhs
end
_sb_ragged_likelihood_rhs(_data, _key, rhs, _layout) = rhs

# Formula-boundary grouping for a flat observed frame. The emitted likelihood
# keeps the logical response name (`key`), so StanBlocks' existing top-level
# RaggedVector path owns the flat predictive draw, group-aggregate likelihood,
# and descriptor `segments` exactly as it does for a pre-grouped response.
function _sb_sampling!(stmts, data, key,
                       lhs::ExprColumn{typeof(ragged)}, rhs;
                       id_lookup=_sb_empty_id_lookup(), kwargs...)
    layout = _sb_ragged_lhs_layout(key, lhs, rhs)
    data[key] = layout.values
    # The gathered ragged response has no raw column of its own on a new
    # DataFrame — record how to re-gather it from the flat response + grouping
    # so `reprocess` regenerates it rather than erroring on it or silently
    # keeping the stale/flat column.
    _sb_record_preproc!(data, key, PreprocEntry(
        :ragged_gather,
        (; group_col=layout.group_col, subject_col=layout.subject_col),
        key, true))
    grouped_rhs = _sb_ragged_likelihood_rhs(data, key, rhs, layout)
    _sb_likelihood!(stmts, key, grouped_rhs, data)
end

_sb_sampling_backed!(stmts, data, key, backing::MissingColumn, rhs;
                     id_lookup, obs_n=nothing, cv_groups=Set{Symbol}(),
                     centered_groups=Set{Symbol}(),
                     group_block_lookup=Dict(),
                     effect_overrides=Dict{Symbol,Any}(), r2d2=_sb_empty_r2d2()) = begin
    rhs_e = _as_expr_column(rhs)
    if !isnothing(rhs_e)
        f = getf(rhs_e)
        # Group-block terms are claimed before the generic submodel/prior hooks.
        if !isempty(group_block_lookup)
            block_info = _sb_find_group_block(f, rhs_e, group_block_lookup)
            if !isnothing(block_info)
                _sb_emit_group_block_term!(stmts, data, key, f, rhs_e, block_info)
                return
            end
        end
        _sb_submodel_rhs!(stmts, data, key, f, rhs_e) !== nothing && return
        # Vector-valued parameter priors first: they need `data` (a multivariate
        # family's hyperparameters carry the declared size), so they cannot ride
        # the four-argument scalar seam below.
        _sb_emit_vector_prior!(stmts, data, key, f, rhs_e) && return
        _sb_emit_prior!(stmts, key, f, rhs_e) && return
    end
    _sb_linear_predictor!(stmts, data, key, rhs; id_lookup, brmi_key=key, obs_n,
                          cv_groups, centered_groups, group_block_lookup,
                          effect_overrides, r2d2)
end

_sb_sampling_backed!(stmts, data, key, backing, rhs; id_lookup, kwargs...) =
    error("sbimpl: unsupported LHS backing for `$key` ($(typeof(backing)))")

# Link-transformed LHS: `log(err) ~ 1 + d`, `logit(p) ~ 1 + x`, etc.
# Sample the linear predictor on the linked scale, then invert to recover the
# response. Mirrors vimpl's `inverse(getf(lhs))` path — any link whose Julia
# `inverse` is a function with a Stan-known name (log/exp/logit/logistic/
# sqrt/square, ...) works; unknown links error at transpile time.
#
# Generic LHS-of-tilde call `f(NamedColumn) ~ rhs`: link-function path. `f`
# is treated as a link whose `InverseFunctions.inverse(f)` we can map to a
# Stan name. Per-`typeof(f)` overrides (`mi`, future `cens`/`trunc`, …) live
# as separate methods of `_sb_sampling!`, mirroring vimpl's
# `vbroadcasted(::ExprColumn{typeof(F)})` extension idiom.
_sb_sampling!(stmts, data, key, lhs::ExprColumn, rhs; id_lookup=_sb_empty_id_lookup(), obs_n=nothing, cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(), group_block_lookup=Dict(), effect_overrides=Dict{Symbol,Any}(), r2d2=_sb_empty_r2d2()) =
    _sb_sampling_through_link!(stmts, data, key, getf(lhs), only(getargs(lhs)), rhs;
                               id_lookup, obs_n, cv_groups, centered_groups,
                               group_block_lookup, effect_overrides, r2d2)

_sb_sampling_through_link!(stmts, data, key, f, inner, rhs; kwargs...) =
    error("sbimpl: expected NamedColumn inside link `$f(...)`, got $(typeof(inner))")

# The name a linear predictor's population design is EMITTED under. A bare
# `loc ~ 1 + x` emits under `loc`; an LHS link transformation `log(Vc) ~ 1 + x`
# emits the LINKED scale under `log_Vc` and then binds `Vc = exp(log_Vc)`. So
# every emitted artefact of the design carries the LINKED spelling
# (`X_log_Vc`, `pop_log_Vc`, `pop_log_Vc_beta_pop`, `Z_log_Vc_<id>_<g>`) while
# the PUBLIC address stays the bare `Vc` that `linear_predictors` reports and
# `popcoefnames` / `effect(...)` / `ranefcoefnames` take.
#
# Mapping the emitted name back to the public one is not derivable by string
# surgery (`pop_log_Vc` is equally `pop_` + an inert LP literally named
# `log_Vc`), so it is derived FORWARDS from the formula through this function.
# LOCKSTEP: `_sb_sampling_through_link!` below is the only emitter of the
# linked form, and `brm_descriptor`'s `pop_lp` map (`descriptor.jl`) is its
# only other caller — keep the three in step.
#
# `identity` is `linear_predictors`' marker for an UNWRAPPED LHS (`loc ~ 1 + x`),
# which emits under its own name. A literal `identity(loc) ~ 1 + x` reaches the
# link path instead and takes the same branch — harmless, because Stan has no
# `identity` function, so that spelling has never transpiled either way.
_sb_lp_emitted_name(lp_name::Symbol, link_lhs_fn) =
    _brm_lp_emitted_name(lp_name, link_lhs_fn)

function _sb_sampling_through_link!(stmts, data, key, f, inner::NamedColumn, rhs; id_lookup, obs_n=nothing, cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(), group_block_lookup=Dict(), effect_overrides=Dict{Symbol,Any}(), r2d2=_sb_empty_r2d2())
    inv_f = InverseFunctions.inverse(f)
    inner_name = name(inner)
    pre_name = _sb_lp_emitted_name(inner_name, f)
    _sb_linear_predictor!(stmts, data, pre_name, rhs; id_lookup, brmi_key=key,
                          obs_n, cv_groups, centered_groups, group_block_lookup,
                          effect_overrides, r2d2)
    push!(stmts, :($inner_name = $(_sb_julia_to_stan_fn(inv_f))($pre_name)))
end

# `mi(y) ~ <family>(args...)` — response with missing values modelled
# jointly. The inner symbol stays as the merged response so other formulas
# can reference it (e.g. `loc2 = ... + b * y`). One method on the same
# `_sb_sampling!` dispatch surface — same shape as `vbroadcasted(::ExprColumn{typeof(protect)})`
# and friends in vimpl.
_sb_sampling!(stmts, data, key, lhs::ExprColumn{typeof(mi)}, rhs; id_lookup=_sb_empty_id_lookup(), kwargs...) =
    _sb_emit_mi!(stmts, data, key, lhs, rhs)

# DRAFT: missing-data response handler. Triggered by `mi(y) ~ <family>(args)`.
# Splits the response into observed values (data) + missing parameters,
# routes both to the per-family `_sb_mi_<family>` submodel, and binds the
# merged vector back to the original symbol for cross-formula reference.
#
# The submodel does the actual SLIC work: declares y_mis via a real `~`,
# adds the observed-row likelihood, and returns the merged vector via the
# `mi_merge` UDF (mutation lives there since top-level slic is single-assign).
# Per-family rather than HOF-generic because family arg lists differ in
# shape and need per-arg `[Jobs]` / `[Jmis]` slicing.
#
# Today only the Normal family is wired (`_sb_mi_normal`); other families
# error with a clear "not yet implemented" message until their submodel
# lands. Predictor-side `mi(x)` (NAs in covariates needing a paired model
# formula) is also not drafted yet.
_SB_MI_FAMILIES = Dict{Any,Symbol}(
    Normal => :_sb_mi_normal,
)
function _sb_emit_mi!(stmts, data, key, lhs::ExprColumn, rhs)
    plan = _brm_missing_response_plan(lhs; prefix="sbimpl")
    isnothing(plan) && error("sbimpl: internal `mi(...)` response was not planned")
    inner_name = plan.source
    Jobs = plan.observed_indices
    Jmis = plan.missing_indices
    y_obs = plan.observed_values

    rhs_e = _as_expr_column(rhs)
    isnothing(rhs_e) && error("sbimpl: `mi($inner_name) ~ <family>(args)` requires the RHS to be a family call, got $(typeof(rhs))")
    fam = getf(rhs_e)
    rhs = rhs_e
    submodel = get(_SB_MI_FAMILIES, fam, nothing)
    isnothing(submodel) && error(
        "sbimpl: `mi(...)` for family `$fam` is not yet implemented. ",
        "Currently supported: $(sort(collect(keys(_SB_MI_FAMILIES)); by=string)). ",
        "Add a `_sb_mi_<family>` @slic submodel + a `_SB_MI_FAMILIES` entry.")

    # Split-data keys, scoped to the response name so two `mi()`-bearing
    # responses in the same model don't collide.
    obs_key  = Symbol(inner_name, :_obs)
    Jobs_key = Symbol(:Jobs_, inner_name)
    Jmis_key = Symbol(:Jmis_, inner_name)
    data[obs_key]  = y_obs
    data[Jobs_key] = Jobs
    data[Jmis_key] = Jmis

    # Family-arg kwargs: inspect the family RHS to pick out the per-arg
    # source bindings the submodel expects (e.g. Normal -> loc, scale).
    # Keep the kwarg names matching the submodel's body symbols.
    fam_kwargs = _sb_mi_family_kwargs(fam, rhs, data)

    # Single submodel call. SLIC binds `inner_name` to the merged vector
    # via the submodel's `return` (lowered to `<inner_name> = <return>`
    # in the parent model scope), so cross-formula references (e.g.
    # `loc2 = a + b * y` elsewhere) see the imputed-merged response.
    call_kwargs = Expr(:parameters,
        Expr(:kw, :y_obs, obs_key),
        Expr(:kw, :Jobs, Jobs_key),
        Expr(:kw, :Jmis, Jmis_key),
        fam_kwargs...)
    push!(stmts, Expr(:call, :~, inner_name, Expr(:call, submodel, call_kwargs)))
end

# Per-family unpacking of the family RHS into the kwargs the matching
# `_sb_mi_<family>` submodel body expects. Each family's arg list is
# fixed (Normal: loc, scale; later: BinomialLogit: n_trials, eta; etc.).
# Materialises any data columns referenced by family args into `data`.
function _sb_mi_family_kwargs(::Type{Normal}, rhs::ExprColumn, data)
    args = getargs(rhs, 2)
    loc, scale = args
    loc_v   = _sb_mi_kwarg_value(loc,   data)
    scale_v = _sb_mi_kwarg_value(scale, data)
    # The submodel body slices each arg with `maybe_index`, which broadcasts
    # a scalar and indexes a vector, so no call-site lifting is needed: pass
    # the resolved bindings straight through, scalar or full-length alike.
    [Expr(:kw, :loc,   loc_v),
     Expr(:kw, :scale, scale_v)]
end

# Resolve a family-arg into the symbol the submodel body should reference.
# Plain LP names (NamedColumn over an ExprColumn) pass through as the
# bound symbol. Data columns get materialised into `data` (if not already)
# and reference by name. Numeric literals turn into themselves.
_sb_mi_kwarg_value(x::Real, _) = x
_sb_mi_kwarg_value(x::NamedColumn, data) = begin
    _maybe_record_data!(data, x, parent(x))
    name(x)
end

_maybe_record_data!(data, x, d::DataColumn) =
    (data[name(x)] = _brm_data_vec(name(x), parent(d)); nothing)
_maybe_record_data!(args...) = nothing
_sb_mi_kwarg_value(x, _) = error("sbimpl: unsupported `mi(...)` family arg shape $(typeof(x))")

# Map a Julia function (typically the result of `InverseFunctions.inverse(...)`
# for a link transform) to the Stan-side function name. Stan ships
# `inv_logit` rather than `logistic`, so the bare `nameof` would emit a
# call stanc rejects. Add other rename cases here as they come up.
_sb_julia_to_stan_fn(f) = f === LogExpFunctions.logistic ? :inv_logit : Symbol(nameof(f))

# Inner-arg unwrap helpers shared by `_sb_predictor_term!` overloads (mo, me,
# s, t2, gp, hsgp, ar, mo1) and a few `mi`-style sites. Each step is a tiny dispatch
# pair so a wrong call shape errors with the wrapping function's name in the
# message rather than a generic `isa` failure.
_sb_named_inner(label::Symbol, x::NamedColumn) = x
_sb_named_inner(label::Symbol, x) =
    error("sbimpl: `$label(...)` expects a NamedColumn, got $(typeof(x))")

_sb_data_backing(label::Symbol, n::Symbol, d::DataColumn) = parent(d)
_sb_data_backing(label::Symbol, n::Symbol, d) =
    error("sbimpl: `$label($n)` expects a raw data column, got $(typeof(d))")

_sb_inner_data(label::Symbol, x) = let inner = _sb_named_inner(label, x)
    (name(inner), _sb_data_backing(label, name(inner), parent(inner)))
end

_sb_real_vec(label::Symbol, n::Symbol, v::AbstractVector{<:Real}) = v
_sb_real_vec(label::Symbol, n::Symbol, v) =
    error("sbimpl: `$label($n)` expects numeric data, got $(typeof(v))")


# ---- linear predictor: emit `X_<name> = hcat(...); <name> ~ popefs(; X=X_<name>)` --

_sb_classify_term!(t::ExprColumn, pop_terms, ran_terms, direct_terms) = begin
    f = getf(t)
    f === (|) && (push!(ran_terms, t); return)
    (f === offset || f === mo1 || f === s || f === t2 || f === gp ||
     f === hsgp || f === dar) &&
        (push!(direct_terms, t); return)
    push!(pop_terms, t)
end
_sb_classify_term!(t, pop_terms, ran_terms, direct_terms) =
    isnothing(_sb_cat_levels(t)) ? push!(pop_terms, t) : push!(direct_terms, t)

function _sb_shared_population_column!(data, column)
    isnothing(column.source) && return nothing
    p = column.preprocess
    if !isnothing(p)
        for dependency in p.dependencies
            _sb_shared_population_column!(data, dependency)
        end
        _sb_record_preproc!(data, column.label,
            PreprocEntry(p.kind, p.const_, p.raw_ref, false))
    end
    data[column.label] = column.values
    column.label
end

function _sb_shared_population_cols!(cols, data,
                                     design::_BRMPopulationDesign)
    for column in design.columns
        if isnothing(column.source)
            row_value = data[design.row_source]
            row_extent = row_value isa Integer && !(row_value isa Bool) ?
                design.row_source : :(num_elements($(design.row_source)))
            push!(cols, :(rep_vector(1.0, $row_extent)))
        else
            push!(cols, _sb_shared_population_column!(data, column))
        end
    end
    cols
end

function _sb_linear_predictor!(stmts, data, target::Symbol, rhs;
                                id_lookup=_sb_empty_id_lookup(),
                                brmi_key::Symbol=target,
                                obs_n::Union{Symbol,Nothing}=nothing,
                                cv_groups=Set{Symbol}(),
                                centered_groups=Set{Symbol}(),
                                group_block_lookup=Dict(),
                                effect_overrides=Dict{Symbol,Any}(), r2d2=_sb_empty_r2d2())
    terms = _sb_terms(rhs)
    pop_terms    = Any[]
    ran_terms    = Any[]  # `(expr | group)` -> collected per-group below
    direct_terms = Any[]  # e.g. `mo1(c)` / `s(x)` / `t2(x,z)` -> direct summand
    for t in terms
        _sb_classify_term!(t, pop_terms, ran_terms, direct_terms)
    end
    isempty(pop_terms) && isempty(ran_terms) && isempty(direct_terms) &&
        error("sbimpl: empty RHS for `$target` — no predictor terms")

    # Direct fixed-slope terms such as `offset(log(exposure))` may contribute
    # an expression rather than a named emitted column. Keep the summand
    # carrier expression-capable; sampled/direct submodels still push Symbols.
    summands = Any[]

    # Term-internal parameter priors are addressed per TERM, not per column, so
    # they ride down to the emitter that owns the term's submodel call rather
    # than being resolved into a per-column vector the way `pop` is.
    term_overrides = _sb_term_effect_overrides(effect_overrides, brmi_key)

    if !isempty(pop_terms)
        col_exprs = Any[]
        shared_design = isempty(ran_terms) && isempty(direct_terms) ?
            _brm_simple_population_design(target, rhs, data, obs_n) : nothing
        if isnothing(shared_design)
            for t in pop_terms
                # `direct_terms` / `ran_terms` ride along for the intercept's
                # tier-1d / tier-1c length probes: a categorical peer or a group
                # term names this formula's row axis, and one of them is the only
                # signal available when the intercept is the sole population term.
                _sb_pop_cols!(col_exprs, t, data, stmts, pop_terms;
                              obs_n, ran_terms, direct_terms, target,
                              group_block_lookup, term_overrides)
            end
        else
            _sb_shared_population_cols!(col_exprs, data, shared_design)
        end
        if isempty(col_exprs)
            # Every population term degenerated to zero columns (e.g.
            # `0 + mo(c)` with single-level `c`): the design spans no
            # coefficients, so `X * beta` is identically 0. Contribute a scalar
            # 0.0 (mirrors mo1 K=1) rather than an empty `hcat()` — which
            # StanBlocks cannot type — and skip the popefs entirely. Keeps the
            # summand list non-empty so the assembler never emits `+()`.
            push!(summands, 0.0)
        else
            X_name = Symbol(:X_, target)
            pop_name = Symbol(:pop_, target)
            # StanBlocks `hcat` promotes a lone vector to matrix[n,1] and folds to
            # append_col for two-or-more columns, so we can always just emit hcat.
            push!(stmts, :($X_name = $(Expr(:call, :hcat, col_exprs...))))
            overrides = _sb_pop_effect_overrides(effect_overrides, brmi_key)
            r2d2_spec = get(r2d2.overrides, brmi_key, nothing)
            if !isnothing(r2d2_spec) && r2d2_spec.n_shares > 0
                _sb_emit_r2d2_popefs!(stmts, data, brmi_key, X_name, pop_name,
                                      length(col_exprs), r2d2_spec,
                                      r2d2.names[brmi_key], overrides)
            elseif isnothing(overrides)
                push!(stmts, :($pop_name ~ popefs(; X=$X_name)))
            else
                length(overrides) == length(col_exprs) || error(
                    "sbimpl: internal effect-prior alignment error for `$brmi_key`: " *
                    "$(length(overrides)) priors for $(length(col_exprs)) columns")
                beta_loc = Any[0.0 for _ in overrides]
                beta_scale = Any[1.0 for _ in overrides]
                for i in eachindex(overrides)
                    isnothing(overrides[i]) && continue
                    beta_loc[i], beta_scale[i] = _sb_effect_normal_args(overrides[i])
                end
                loc_expr = Expr(:vect, beta_loc...)
                scale_expr = Expr(:vect, beta_scale...)
                push!(stmts, :($pop_name ~ _popefs_normal(;
                    X=$X_name, beta_loc=$loc_expr, beta_scale=$scale_expr)))
            end
            push!(summands, pop_name)
        end
    end

    cat_overrides = _sb_cat_effect_overrides(effect_overrides, brmi_key)
    for dt in direct_terms
        _sb_emit_direct!(stmts, data, target, dt, summands;
                         group_block_lookup, cat_overrides, term_overrides)
    end

    # A plain (un-`|ID|`'d) random effect under an `r2d2` decomposition IS the
    # residual: its scale is the derived `sqrt((1 - R2) * tau_bsv^2)` rather
    # than a sampled SD. `|ID|`'d terms get the same scale via the bucket
    # prepass, so this only reaches the plain path.
    r2d2_scale = haskey(r2d2.names, brmi_key) ?
        _sb_r2d2_resid_scale(r2d2.names[brmi_key]) : nothing
    _sb_emit_ranefs!(stmts, data, target, ran_terms, summands;
                     id_lookup, brmi_key, cv_groups, centered_groups, r2d2_scale,
                     term_overrides)

    if length(summands) == 1
        push!(stmts, :($target = $(only(summands))))
    else
        push!(stmts, :($target = $(Expr(:call, :+, summands...))))
    end
end

# ---- Tier-1 emission: fuse a pure-population Gaussian likelihood -----------
#
# Stan ships `normal_id_glm_lpdf(y | X, alpha, beta, sigma)`, whose gradient is
# written by hand against `X` and therefore never materialises the N-vector
# `X * beta` on the autodiff tape. BRM's ordinary emission builds that vector
# TWICE -- once as `pop_<lp>` (the submodel's return) and once as the linear
# predictor `<lp>` -- and both land in `transformed parameters`, i.e. on the
# gradient path. This post-pass rewrites the narrow case where the two are the
# same thing:
#
#   pop_mu ~ popefs(; X=X_mu)          =>  pop_mu ~ _popefs_coefs(; X=X_mu)
#   mu = pop_mu                            y ~ normal_id_glm(X_mu, 0.0, pop_mu, sigma)
#   y ~ normal(mu, sigma)                  mu = X_mu * pop_mu
#
# BOTH halves matter, and the second is not incidental: `mu` is now assigned
# AFTER the likelihood, so StanBlocks places it in `generated quantities`,
# off the gradient path entirely. The fused lpdf alone measures ~1.2x; the
# pair ~3.6x at N=2000, K=4. `alpha` is a literal `0.0` because BRM's design
# matrix already carries the intercept as a column of ones, so `beta_pop`
# covers every column and the parameter name `pop_<lp>_beta_pop` -- the
# `popcoefnames` public contract -- is untouched.
#
# This runs over the FINISHED statement list rather than inside
# `_sb_linear_predictor!` because the decisive guard ("`<lp>` is consumed by
# exactly one Gaussian likelihood and by nothing else in the model") is a
# whole-model property no per-formula emitter can see. Anything that misses
# even one guard falls through untouched, byte for byte.
_sb_mentions(::Any, ::Symbol) = false
_sb_mentions(e::Symbol, s::Symbol) = e === s
_sb_mentions(e::Expr, s::Symbol) = any(a -> _sb_mentions(a, s), e.args)

# Population submodel => its coefficient-returning sibling. The R2D2 popefs
# path routes through `_popefs_normal` too, but it emits extra statements that
# consume `pop_<lp>`, so the mention guards below decline it on their own.
const _SB_GLM_COEF_SUBMODEL = Dict{Symbol,Symbol}(
    :popefs => :_popefs_coefs, :_popefs_normal => :_popefs_normal_coefs)

function _sb_fuse_normal_id_glm!(stmts, data)
    n = length(stmts)
    # Statement INDICES mentioning `sym` anywhere in their AST. Deliberately
    # over-approximates (a kwarg NAME counts as a mention), which can only ever
    # decline a fusion, never license a wrong one.
    mentioning(sym) = findall(i -> _sb_mentions(stmts[i], sym), 1:n)
    plans = NamedTuple[]
    for i in 1:n
        st = stmts[i]
        # `y ~ normal(<lp>, <sigma>)` -- the plain two-arg Gaussian. Truncated,
        # censored, ordinal and every other family lower to a different Stan
        # name, so they cannot reach here.
        (st isa Expr && st.head === :call && length(st.args) == 3 &&
         st.args[1] === :~) || continue
        y, rhs = st.args[2], st.args[3]
        y isa Symbol || continue
        (rhs isa Expr && rhs.head === :call && length(rhs.args) == 3 &&
         rhs.args[1] === :normal) || continue
        lp, sigma = rhs.args[2], rhs.args[3]
        lp isa Symbol || continue
        # The response must be a materialised flat data vector of REALS, i.e.
        # something StanBlocks declares as `vector`. `mi(y)` leaves no data
        # entry, and a ragged/multi-column backing is a different Stan signature
        # entirely. The float check is not pedantry: an integer-valued Gaussian
        # response declares as `array[] int`, which plain `normal` accepts and
        # `normal_id_glm_lpdf` (whose `y` is a `vector`) rejects -- a program
        # `transpiles()` calls green and stanc calls ill-typed.
        yv = get(data, y, nothing)
        yv isa AbstractVector{<:AbstractFloat} || continue
        # `<lp>` must be produced by a single-summand assignment and consumed
        # by THIS likelihood -- nothing else in the model may read it. That is
        # what makes moving its assignment past the likelihood safe, and it is
        # also what rules out the LHS-link path (`log(Vc) ~ …` emits `log_Vc`,
        # which a following `Vc = exp(log_Vc)` reads) and every additive
        # random-effect or direct-term contribution.
        li = mentioning(lp)
        (length(li) == 2 && li[2] == i) || continue
        j = li[1]
        as = stmts[j]
        (as isa Expr && as.head === :(=) && as.args[1] === lp) || continue
        pop = as.args[2]
        pop isa Symbol || continue
        # ... and `pop_<lp>` in turn must come from one population submodel
        # call read only by that assignment.
        pj = mentioning(pop)
        (length(pj) == 2 && pj[2] == j) || continue
        k = pj[1]
        ps = stmts[k]
        (ps isa Expr && ps.head === :call && length(ps.args) == 3 &&
         ps.args[1] === :~ && ps.args[2] === pop) || continue
        call = ps.args[3]
        (call isa Expr && call.head === :call) || continue
        coef_f = get(_SB_GLM_COEF_SUBMODEL, call.args[1], nothing)
        isnothing(coef_f) && continue
        # The scale may be a scalar or an N-vector (a distributional-parameter
        # `log(sigma) ~ …`); both are valid `normal_id_glm` signatures. It must
        # not depend on the names we are about to move, though -- the mention
        # counts above tolerate a second occurrence inside this statement.
        (_sb_mentions(sigma, lp) || _sb_mentions(sigma, pop)) && continue
        # The design matrix must be a plain emitted name so it can be passed
        # positionally to the fused call.
        params = length(call.args) >= 2 ? call.args[2] : nothing
        (params isa Expr && params.head === :parameters) || continue
        X = nothing
        for kw in params.args
            (kw isa Expr && kw.head === :kw && kw.args[1] === :X) || continue
            X = kw.args[2]
        end
        X isa Symbol || continue
        push!(plans, (; k, j, i, pop, lp, X, y, sigma,
                      call=Expr(:call, coef_f, call.args[2:end]...)))
    end
    isempty(plans) && return stmts

    rewrite = Dict{Int,Any}()
    tail = Dict{Int,Any}()
    drop = Set{Int}()
    for p in plans
        rewrite[p.k] = Expr(:call, :~, p.pop, p.call)
        push!(drop, p.j)
        rewrite[p.i] = Expr(:call, :~, p.y,
                            Expr(:call, :normal_id_glm, p.X, 0.0, p.pop, p.sigma))
        # Re-emitted AFTER the likelihood => `generated quantities`.
        tail[p.i] = Expr(:(=), p.lp, Expr(:call, :*, p.X, p.pop))
    end
    out = Any[]
    for i in 1:n
        i in drop && continue
        push!(out, get(rewrite, i, stmts[i]))
        haskey(tail, i) && push!(out, tail[i])
    end
    empty!(stmts)
    append!(stmts, out)
    stmts
end

# ---- consumer helper: name the `beta_pop` columns -------------------------
#
# `popcoefnames(brmi, lhs)` lets a downstream consumer relabel raw
# `pop_<lhs>_beta_pop.N` posterior columns WITHOUT re-parsing the formula and
# WITHOUT re-deriving the (data-dependent) pop-vs-cat-vs-ranef classification.
# It mirrors the emitter EXACTLY by driving the SAME `_sb_classify_term!` +
# `_sb_pop_cols!` used to build the design matrix `X` — so the returned labels
# can never drift from what `popefs` actually multiplies. Standard fixed-effect
# terms only (covers the whole regression-covariate surface); group-structured
# terms whose columns need prepass context (e.g. `hsgp(x, by=g)`) are out of
# scope in v1 and raise a clear, actionable error rather than mis-counting.

"""
    popcoefnames(brmi::BRMI, lhs::Symbol) -> Union{Vector{Symbol}, Nothing}

Ordered labels of the population-level `beta_pop` coefficient columns for
linear predictor `lhs`, as emitted by the sbimpl backend (`SBBRMI`). The
k-th returned symbol labels the parameter column `pop_<lhs>_beta_pop.k`
1:1, so a consumer can turn raw `pop_<lhs>_beta_pop.N` posterior columns
into human-readable names without re-parsing the formula.

Only terms that sbimpl folds into the `popefs` design matrix appear, in
formula (left-to-right) order:

- the intercept `1` is INCLUDED (labelled `:Intercept`), at its formula
  position — conventionally first, `1 + …`; there is NO separate
  `pop_<lhs>_Intercept` parameter;
- plain continuous (`Real`, non-integer) predictors — one column each,
  labelled by the column name;
- single-`beta` wrapped terms (`mo`, `me`, `protect`,
  `log(x)`, `x^2`, …) — one column each, labelled by the emitted design key;
- an interaction `a & b` — one label per expanded treatment-contrast column.

EXCLUDED — emitted as their OWN parameters, NOT `beta_pop`, so they never
appear in `pop_<lhs>_beta_pop`:

- integer- or `CategoricalVector`-typed bare predictors → `cat_<lhs>_<name>`
  (K−1 treatment contrasts);
- `offset(x)` → `x` itself, with fixed coefficient one;
- `mo1(c)` → `mo1_<c>`; `s(x)` and `t2(x,z)` → their own fixed/range
  coefficients and smoothing scales;
- explicit-coefficient `coef * a` (own scalar);
- random-effects blocks `(… | g)` → per-group ranef parameters.

Needs a data-bound `BRMI` (any fitted model has one): the population-vs-
categorical split reads each predictor's element type. Returns `Symbol[]`
when `lhs` has no population columns (e.g. `loc ~ (1 | g)`), and `nothing`
when `lhs` is not a linear predictor of `brmi`.

These are the labels the `effect(lhs, label) ~ Normal(...)` prior address
resolves against — with ONE addition that is deliberately not listed here,
because it is not a `beta_pop` column: a categorical / integer-coded
predictor (bare, or wrapped in `factor(...)`) is addressable by its COLUMN
name, which sets one shared Normal prior over its K-1 treatment contrasts
(`cat_<lhs>_<c>_beta`). Everything else in the EXCLUDED list above still owns
parameters no `effect(...)` address reaches.
"""
function popcoefnames(brmi::BRMI, lhs::Symbol)
    op = linear_predictor_op(brmi, lhs)
    isnothing(op) && return nothing
    _, rhs = getargs(op, 2)
    pop_terms = Any[]; ran_terms = Any[]; direct_terms = Any[]
    for t in _sb_terms(rhs)
        _sb_classify_term!(t, pop_terms, ran_terms, direct_terms)
    end
    labels = Symbol[]
    # Scratch data/stmts: `_sb_pop_cols!` materialises columns into these as a
    # side effect; we only read back the pushed column reference(s) as labels.
    scratch = Dict{Symbol,Any}(); scratch_stmts = Any[]
    for t in pop_terms
        # The intercept is always exactly one all-ones column; skip the
        # emitter's length probe (irrelevant to naming) and label it directly.
        if t isa Integer
            push!(labels, :Intercept)
            continue
        end
        cols = Any[]
        try
            _sb_pop_cols!(cols, t, scratch, scratch_stmts, pop_terms)
        catch err
            error("popcoefnames: cannot resolve the `beta_pop` column(s) for term ",
                  "`$(_popcoef_show(t))` in predictor `$lhs` without full model context ",
                  "($(sprint(showerror, err))). This helper covers the standard ",
                  "fixed-effect terms; for group-structured terms (e.g. `hsgp(x, by=g)`) ",
                  "read the parameter names off the transpiled `SBBRMI` instead.")
        end
        for c in cols
            # Non-intercept pop columns are Symbol references; a stray Expr
            # (only the intercept probe emits one) maps back to :Intercept.
            label = if t isa ExprColumn && getf(t) === interval_censored
                name(_sb_named_inner(:interval_censored, only(getargs(t))))
            else
                c isa Symbol ? c : :Intercept
            end
            push!(labels, label)
        end
    end
    labels
end

# ---- consumer helper: name the `cat_<lp>_<c>_beta` contrast blocks ----------
#
# The categorical counterpart of `popcoefnames`. A categorical / integer-coded
# predictor (bare, or wrapped in `factor(...)`) is EXCLUDED from `beta_pop` and
# owns its own K-1 treatment-contrast block `cat_<lp>_<c>_beta`, so it can never
# appear in `popcoefnames`. This walker drives the SAME `_sb_terms` +
# `_sb_classify_term!` the emitter does, so the returned names can never drift
# from what `_sb_emit_cat!` actually emits.
#
# Returns the ordered EMITTED term names (`nothing` when `lhs` is not a linear
# predictor). `factor(c; ref=k)` recodes into a synthetic `c__ref_k` column, so
# the emitted name and the name the user wrote differ there; see
# `_sb_cat_addresses` for the address spellings that resolve onto it.
_sb_cat_block_name(lp::Symbol, term::Symbol) = Symbol(:cat_, lp, :_, term)

function _sb_cat_entries(brmi::BRMI, lhs::Symbol)
    op = linear_predictor_op(brmi, lhs)
    isnothing(op) && return nothing
    predictors = [lp for lp in linear_predictors(brmi) if lp.name === lhs]
    length(predictors) == 1 || error(
        "sbimpl: expected one linear predictor named `$lhs`, found $(length(predictors))")
    emitted_lp = _sb_lp_emitted_name(lhs, only(predictors).link_lhs_fn)
    _, rhs = getargs(op, 2)
    pop_terms = Any[]; ran_terms = Any[]; direct_terms = Any[]
    for t in _sb_terms(rhs)
        _sb_classify_term!(t, pop_terms, ran_terms, direct_terms)
    end
    [(; address=name(t), emitted=_sb_cat_block_name(emitted_lp, name(t)))
     for t in direct_terms
     if t isa NamedColumn && !isnothing(_sb_cat_levels(t))]
end

function _sb_cat_coefnames(brmi::BRMI, lhs::Symbol)
    entries = _sb_cat_entries(brmi, lhs)
    isnothing(entries) ? nothing : Symbol[e.emitted for e in entries]
end

# `factor(c; ref=k)` is re-encoded at term-collection time into a synthetic
# `c__ref_k` NamedColumn (see `_sb_collect_terms_expr!(::typeof(factor), ...)`),
# and that synthetic name is what reaches the emitted `cat_<lp>_c__ref_k_beta`.
# Requiring the user to spell the internal suffix would leak an implementation
# detail into a public prior address, so BOTH the emitted name and the column
# the formula names resolve onto the same block.
const _SB_REF_SUFFIX = r"__ref_[0-9]+$"
function _sb_cat_addresses(emitted::Symbol)
    s = String(emitted)
    m = match(_SB_REF_SUFFIX, s)
    isnothing(m) && return (emitted,)
    (emitted, Symbol(s[1:prevind(s, m.offset)]))
end

# address Symbol -> emitted `cat_<lp>_<name>` term name, for one linear predictor.
# An address that would name two different blocks (only reachable by writing
# both `factor(c)` and `factor(c; ref=k)` in one predictor) is dropped rather
# than silently resolving to the first, so `effect(...)` fails loudly instead.
# ---- term-parameter prior resolution ---------------------------------------
#
# `term_priors` yields statements addressed by TERM KEY — `Symbol("s(age)")`,
# the term as the formula spells it. Resolution walks each linear predictor's
# own terms, keys them the SAME way, and hands the winning statement to the
# emitter as a per-predictor `Dict{Symbol,Any}`, exactly like `cat_overrides`.

# Canonical key for a term as the BACKEND sees it. Must agree character for
# character with `_term_address_key` in `macro.jl`, which builds the same key
# from surface syntax — that agreement IS the address resolution. Numeric and
# keyword arguments are excluded on both sides, so `me(x, 0.5)` and `me(x)`
# name one term.
_sb_term_arg_name(x::NamedColumn) = name(x)
_sb_term_arg_name(_x) = nothing
_sb_term_key(t) = Symbol(nameof(getf(t)), "(",
    join((n for n in (_sb_term_arg_name(a) for a in getargs(t)) if !isnothing(n)), ","), ")")

# The three penalty blocks of a tensor smooth, in the order `_sb_t2` samples
# them. Fixed here so the public component name and the vector index cannot
# drift apart.
const _SB_T2_BLOCKS = (:rr, :rn, :nr)
_sb_t2_sd_index(c::Symbol) = findfirst(==(c), _SB_T2_BLOCKS)

# term key -> the terms carrying it, for one linear predictor. Only terms that
# own configurable parameters are listed; anything else is simply absent, so an
# address naming it fails with "matches no term" rather than resolving to
# something that has nothing to configure. The value is a VECTOR because two
# spellings of one key in one predictor (`s(x) + s(x)`) make the address
# ambiguous — the resolver reports that as its own error rather than silently
# configuring whichever copy the walker reached first.
const _SB_PRIOR_TERMS = (mo, mo1, me, interval_censored, s, t2, gp, hsgp, dar)
function _sb_term_address_map(brmi::BRMI, lhs::Symbol)
    out = Dict{Symbol,Vector{Any}}()
    op = linear_predictor_op(brmi, lhs)
    isnothing(op) && return out
    for t in _sb_terms(getargs(op, 2)[2])
        t isa ExprColumn || continue
        any(f -> getf(t) === f, _SB_PRIOR_TERMS) || continue
        push!(get!(() -> Any[], out, _sb_term_key(t)), t)
    end
    out
end

_sb_term_spelling(spec) = begin
    head = spec.class === :term_sd ? "sd" :
           spec.class === :term_ar ? "ar" :
           spec.class === :term_simplex ? "simplex" :
           spec.class === :term_length_scale ? "length_scale" : "latent"
    lp = isnothing(spec.predictor) ? ":" : string(spec.predictor)
    comp = isnothing(spec.component) ? "" : ", $(spec.component)"
    "$head($lp, $(spec.term)$comp)"
end

# One spec -> the emission-ready configuration for the term it reached, with
# every class/term/component mismatch refused by name. Returns a pair so the
# caller can key several statements onto ONE term (the three `t2` blocks) while
# keeping precedence per addressed parameter rather than per term.
function _sb_term_config(spec, t, spelling)
    f = getf(t)
    if spec.class === :term_sd
        if f === s
            isnothing(spec.component) || error(
                "sbimpl: `$spelling` names a component, but `s(x)` has exactly " *
                "one smoothing scale. Write `sd($(isnothing(spec.predictor) ? ":" : spec.predictor), $(spec.term))`.")
            return (:sd, (; rate=_sb_ranef_sd_rate(spec, spelling)))
        elseif f === t2
            isnothing(spec.component) && error(
                "sbimpl: `$spelling` is ambiguous — a tensor smooth has three " *
                "independent smoothing scales. Name one of " *
                join(("`$b`" for b in _SB_T2_BLOCKS), ", ") * ".")
            isnothing(_sb_t2_sd_index(spec.component)) && error(
                "sbimpl: `$spelling` names no penalty block of `$(spec.term)`; " *
                "valid blocks are " * join(("`$b`" for b in _SB_T2_BLOCKS), ", ") * ".")
            return (Symbol(:sd_, spec.component), (; rate=_sb_ranef_sd_rate(spec, spelling)))
        elseif f === gp || f === hsgp || f === dar
            # GP amplitude and differenced-AR innovation sigma are both positive
            # model-scale standard deviations, so they share the general
            # positive-scale family set.
            isnothing(spec.component) || error(
                "sbimpl: `$spelling` names a component, but `$(nameof(f))(...)` " *
                "has exactly one scale parameter.")
            cfg = f === dar ?
                _sb_gp_scale_prior(spec, spelling;
                    default="`Normal(0, 0.2)` truncated to be positive") :
                _sb_gp_scale_prior(spec, spelling)
            return (:sigma, cfg)
        end
        error("sbimpl: `$spelling` — `$(nameof(f))` has no scale to configure. " *
              "`sd(...)` on a term applies to `s(x)`, `t2(x, z)`, `gp(x...)`, " *
              "`hsgp(x...)`, and `dar(time)`.")
    elseif spec.class === :term_ar
        f === dar || error(
            "sbimpl: `$spelling` — `$(nameof(f))` has no bounded persistence " *
            "coefficient to configure. `ar(...)` on a term applies to `dar(time)`.")
        isnothing(spec.component) || error("sbimpl: `$spelling` takes no component slot")
        return (:ar, _sb_dar_ar_prior(spec, spelling))
    elseif spec.class === :term_length_scale
        (f === gp || f === hsgp) || error(
            "sbimpl: `$spelling` — `$(nameof(f))` has no length scale to " *
            "configure. `length_scale(...)` applies to `gp(x...)` and " *
            "`hsgp(x...)`.")
        isnothing(spec.component) || error("sbimpl: `$spelling` takes no component slot")
        return (:length_scale, _sb_gp_scale_prior(spec, spelling))
    elseif spec.class === :term_simplex
        (f === mo || f === mo1) || error(
            "sbimpl: `$spelling` — `$(nameof(f))` has no simplex to configure. " *
            "`simplex(...)` applies to `mo(c)` and `mo1(c)`.")
        isnothing(spec.component) || error("sbimpl: `$spelling` takes no component slot")
        T = _as_distribution_type(spec.family)
        (!isnothing(T) && T <: Dirichlet) || error(
            "sbimpl: `$spelling` expects `Dirichlet(...)`; got `$(spec.family)`")
        isempty(spec.keywords) || error(
            "sbimpl: `$spelling ~ Dirichlet(...)` does not accept keywords")
        return (:simplex, (; alpha=map(_sb_effect_prior_arg, spec.arguments)))
    end
    (f === me || f === interval_censored) || error(
        "sbimpl: `$spelling` — `$(nameof(f))` has no latent covariate to " *
        "configure. `latent(...)` applies to `me(x, sd)` and " *
        "`interval_censored(x; upper=lloq)` predictor terms.")
    isnothing(spec.component) || error("sbimpl: `$spelling` takes no component slot")
    T = _as_distribution_type(spec.family)
    (!isnothing(T) && T <: Normal) || error(
        "sbimpl: `$spelling` expects `Normal(location, scale)`; got `$(spec.family)`")
    isempty(spec.keywords) || error(
        "sbimpl: `$spelling ~ Normal(...)` does not accept keywords")
    loc, scale = _sb_effect_normal_args(spec.expression)
    (:latent, (; loc, scale))
end

# Resolve every term-parameter statement onto `lp -> term key -> config`.
# Empty when the formula has none, so an unconfigured model never pays for the
# walk and its emission is decided entirely by the Julia-side defaults.
function _sb_term_prior_overrides(brmi::BRMI)
    specs = term_priors(brmi)
    isempty(specs) && return Dict{Symbol,Dict{Symbol,Any}}()

    lp_names = Symbol[x.name for x in linear_predictors(brmi)]
    # Same lazy/memoised discipline as the population and categorical paths: a
    # predictor is walked only when a statement could reach it, and a shape
    # that cannot be walked is skipped rather than made fatal for the model.
    resolved = Dict{Symbol,Dict{Symbol,Vector{Any}}}()
    map_of(lp::Symbol) = get!(resolved, lp) do
        _sb_is_prior_declaration(brmi, lp) && return Dict{Symbol,Vector{Any}}()
        try
            _sb_term_address_map(brmi, lp)
        catch
            Dict{Symbol,Vector{Any}}()
        end
    end

    # `:` in the predictor slot is the DEFAULT layer, exactly as everywhere
    # else on this surface: rank 0, overridden by a named predictor at rank 1,
    # and an exact tie is an error rather than a silent winner.
    staged = Dict{Symbol,Dict{Symbol,Dict{Symbol,Any}}}()
    for spec in specs
        spelling = _sb_term_spelling(spec)
        rank = isnothing(spec.predictor) ? 0 : 1
        targets = if isnothing(spec.predictor)
            hits = Symbol[lp for lp in lp_names if haskey(map_of(lp), spec.term)]
            isempty(hits) && error(
                "sbimpl: `$spelling` matches no `$(spec.term)` term in any " *
                "linear predictor.")
            hits
        else
            haskey(map_of(spec.predictor), spec.term) || error(
                "sbimpl: `$spelling` matches no `$(spec.term)` term in `" *
                "$(spec.predictor)`. Terms carrying configurable parameters " *
                "there: " * (isempty(map_of(spec.predictor)) ? "(none)" :
                join(("`$k`" for k in sort!(collect(keys(map_of(spec.predictor))), by=string)), ", ")) * ".")
            Symbol[spec.predictor]
        end
        for lp in targets
            hits = map_of(lp)[spec.term]
            length(hits) == 1 || error(
                "sbimpl: `$spelling` is ambiguous — `$lp` carries $(length(hits)) " *
                "terms spelled `$(spec.term)`, and a prior address cannot tell " *
                "them apart. Give them distinguishable arguments, or drop the " *
                "statement.")
            slot, cfg = _sb_term_config(spec, only(hits), spelling)
            cells = get!(staged, lp) do
                Dict{Symbol,Dict{Symbol,Any}}()
            end
            per_term = get!(cells, spec.term) do
                Dict{Symbol,Any}()
            end
            held = get(per_term, slot, nothing)
            if isnothing(held) || rank > held.rank
                per_term[slot] = (; cfg, rank, spelling)
            elseif rank == held.rank
                error("sbimpl: `$spelling` and `$(held.spelling)` are equally " *
                      "specific and both set the same parameter of `$(spec.term)` " *
                      "in `$lp`. Neither wins — make one of them more specific, " *
                      "or drop it.")
            end
        end
    end

    out = Dict{Symbol,Dict{Symbol,Any}}()
    for (lp, cells) in staged
        out[lp] = Dict{Symbol,Any}(
            k => Dict{Symbol,Any}(slot => held.cfg for (slot, held) in per_term)
            for (k, per_term) in cells)
    end
    out
end

# ---- readers for the resolved term dict -------------------------------------
#
# One function per configurable parameter. Each returns the EMISSION expression
# for both cases, so the default an unconfigured formula gets is written down
# exactly once instead of once per call site.

_sb_term_cfg(term_overrides, t, slot) = begin
    per_term = get(term_overrides, _sb_term_key(t), nothing)
    isnothing(per_term) ? nothing : get(per_term, slot, nothing)
end

# `brm_ranef_sd`'s two data vectors for a smoothing term: family 0 is the
# half-standard-normal an unmentioned scale keeps, family 1 the exponential.
# `slots` fixes both the length and the block order, so the addressed component
# and the sampled vector index cannot drift apart.
_sb_term_sd_slots(::typeof(s)) = (:sd,)
_sb_term_sd_slots(::typeof(t2)) = map(c -> Symbol(:sd_, c), _SB_T2_BLOCKS)
function _sb_term_sd_args(term_overrides, t)
    slots = _sb_term_sd_slots(getf(t))
    family = Any[0 for _ in slots]
    rate = Any[1.0 for _ in slots]
    for (i, slot) in pairs(slots)
        cfg = _sb_term_cfg(term_overrides, t, slot)
        isnothing(cfg) && continue
        family[i] = 1
        rate[i] = cfg.rate
    end
    Expr(:vect, family...), Expr(:vect, rate...)
end

# Dirichlet concentration for a `mo`/`mo1` term with `n_levels` levels, hence a
# length `n_levels - 1` increment simplex. One argument is broadcast over the
# whole simplex; `n_levels - 1` of them set it elementwise.
function _sb_mo_alpha_expr(term_overrides, t, n_levels)
    k = n_levels - 1
    cfg = _sb_term_cfg(term_overrides, t, :simplex)
    isnothing(cfg) && return :(rep_vector(1., $k))
    a = cfg.alpha
    all(x -> x isa Real && isfinite(x) && x > 0, a) || error(
        "sbimpl: `simplex(..., $(_sb_term_key(t))) ~ Dirichlet(...)` expects " *
        "finite positive numeric concentrations, got $(repr(a))")
    length(a) == 1 && return :(rep_vector($(Float64(only(a))), $k))
    length(a) == k || error(
        "sbimpl: `simplex(..., $(_sb_term_key(t))) ~ Dirichlet(...)` expects " *
        "either one concentration or $k of them (one per increment of a " *
        "$n_levels-level monotonic effect), got $(length(a)).")
    Expr(:vect, map(Float64, a)...)
end

# Location/scale of a latent-covariate term. The (0, 1) default preserves the
# historical `me` contract and is also the interval-predictor default.
function _sb_me_latent_args(term_overrides, t)
    cfg = _sb_term_cfg(term_overrides, t, :latent)
    isnothing(cfg) ? (0.0, 1.0) : (cfg.loc, cfg.scale)
end

function _sb_interval_censored_predictor_plan(x_name::Symbol, x_raw,
                                               upper_name::Symbol, upper_raw,
                                               lower::Real=0.0)
    x = collect(Float64, _sb_real_vec(:interval_censored, x_name, x_raw))
    upper = collect(Float64,
        _sb_real_vec(:interval_censored, upper_name, upper_raw))
    isempty(x) && error(
        "sbimpl: interval-censored predictor `$x_name` cannot be empty")
    length(x) == length(upper) || error(
        "sbimpl: interval-censored predictor `$x_name` and LLOQ column " *
        "`$upper_name` must have equal lengths ($(length(x)) vs " *
        "$(length(upper)))")
    all(isfinite, x) || error(
        "sbimpl: interval-censored predictor `$x_name` must be finite")
    all(isfinite, upper) || error(
        "sbimpl: interval-censored predictor LLOQs `$upper_name` " *
        "must be finite")
    isfinite(lower) || error(
        "sbimpl: interval-censored predictor lower bound must be finite")
    invalid = findfirst(i -> x[i] < upper[i], eachindex(x))
    isnothing(invalid) || error(
        "sbimpl: interval-censored predictor row $invalid has `$x_name` = " *
        "$(x[invalid]) below its LLOQ $(upper[invalid]); under the `x == LLOQ` " *
        "BLOQ convention, quantified values must exceed LLOQ")
    Jinterval = findall(i -> x[i] == upper[i], eachindex(x))
    isempty(Jinterval) && error(
        "sbimpl: `interval_censored($x_name; upper=$upper_name)` has no BLOQ " *
        "rows (`$x_name == $upper_name`); use `$x_name` directly")
    invalid_bound = findfirst(i -> lower >= upper[i], Jinterval)
    isnothing(invalid_bound) || error(
        "sbimpl: interval-censored predictor row $(Jinterval[invalid_bound]) " *
        "requires lower < LLOQ, got $lower >= " *
        "$(upper[Jinterval[invalid_bound]])")
    Jexact = findall(i -> x[i] > upper[i], eachindex(x))
    (; x_exact=x[Jexact], x_lower=fill(Float64(lower), length(Jinterval)),
       x_upper=upper[Jinterval], Jexact=collect(Int, Jexact),
       Jinterval=collect(Int, Jinterval))
end

# ---- gp / hsgp length scale and marginal amplitude --------------------------
#
# `rho` and `sigma` are both strictly positive scales of the same latent
# function, so they share ONE family set rather than carrying a per-parameter
# one. The returned record keeps the DECLARATION bounds beside the density
# because an override must reproduce the whole base statement: `Base.merge`
# replaces a matching-named statement wholesale, so a dropped `lower=` leaves
# the parameter unconstrained and a `Uniform` density whose declaration does
# not match its support is -Inf everywhere the sampler starts.
const _SB_GP_SCALE_FAMILIES =
    (LogNormal, InverseGamma, Gamma, Exponential, Normal, Uniform)

# `_sb_stan_dist_args` maps Distributions' parameterisation onto Stan's. On this
# family set the only non-literal it can build from numeric inputs is the
# scale -> rate reciprocal, which folds back to a constant here so the emitted
# declaration bound and the density argument are both plain numbers.
_sb_gp_scale_const(x::Real) = Float64(x)
function _sb_gp_scale_const(x)
    (Meta.isexpr(x, :call) && length(x.args) == 3 && x.args[1] === Symbol("./") &&
     x.args[2] isa Real && x.args[3] isa Real) || error(
        "sbimpl: a positive-scale prior takes numeric formula constants, " *
        "got $(repr(x))")
    Float64(x.args[2] / x.args[3])
end

function _sb_gp_scale_prior(spec, spelling::AbstractString;
                            default="`LogNormal(0, 1)` truncated to be positive")
    T = _as_distribution_type(spec.family)
    (!isnothing(T) && any(F -> T <: F, _SB_GP_SCALE_FAMILIES)) || error(
        "sbimpl: `$spelling` supports " *
        join(("`$(nameof(F))`" for F in _SB_GP_SCALE_FAMILIES), ", ") *
        "; got `$(spec.family)`. An unmentioned scale keeps $default.")
    isempty(spec.keywords) || error(
        "sbimpl: `$spelling ~ $(spec.family)(...)` does not accept keywords; " *
        "the declaration bounds follow from the family.")
    args = map(_sb_effect_prior_arg, spec.arguments)
    all(a -> a isa Real && isfinite(a), args) || error(
        "sbimpl: `$spelling` hyperparameters must be finite numeric formula " *
        "constants, got $(repr(args))")
    stan_args = map(_sb_gp_scale_const, _sb_stan_dist_args(T, Tuple(args)))
    rhs = Expr(:call, _sb_stan_dist_name(T), stan_args...)
    T <: Uniform || return (; rhs, lower=0.0, upper=nothing)
    lower, upper = stan_args
    (lower >= 0 && upper > lower) || error(
        "sbimpl: `$spelling ~ Uniform($lower, $upper)` bounds a positive scale, " *
        "so it needs `0 <= lower < upper`.")
    (; rhs, lower, upper)
end

# A differenced-AR persistence is a unit-interval coefficient, not a positive
# scale. Normal/Beta/Uniform cover the useful bounded families without
# pretending the unconstrained `phi_raw` of the older `ar(...)` term is the
# same parameter. The default `_sb_dar1` statement is Normal(0.5, 0.2)
# truncated to [0, 1], matching the CDC model this term unblocks.
const _SB_DAR_AR_FAMILIES = (Normal, Beta, Uniform)
function _sb_dar_ar_prior(spec, spelling::AbstractString)
    T = _as_distribution_type(spec.family)
    (!isnothing(T) && any(F -> T <: F, _SB_DAR_AR_FAMILIES)) || error(
        "sbimpl: `$spelling` supports " *
        join(("`$(nameof(F))`" for F in _SB_DAR_AR_FAMILIES), ", ") *
        "; got `$(spec.family)`. An unmentioned coefficient keeps " *
        "`Normal(0.5, 0.2)` truncated to `[0, 1]`.")
    isempty(spec.keywords) || error(
        "sbimpl: `$spelling ~ $(spec.family)(...)` does not accept keywords; " *
        "the declaration is bounded by the differenced-AR contract.")
    args = map(_sb_effect_prior_arg, spec.arguments)
    all(a -> a isa Real && isfinite(a), args) || error(
        "sbimpl: `$spelling` hyperparameters must be finite numeric formula " *
        "constants, got $(repr(args))")
    try
        T(args...)
    catch err
        error("sbimpl: `$spelling` has invalid `$(spec.family)` hyperparameters " *
              "$(repr(args)): $(sprint(showerror, err))")
    end
    stan_args = map(_sb_gp_scale_const, _sb_stan_dist_args(T, Tuple(args)))
    rhs = Expr(:call, _sb_stan_dist_name(T), stan_args...)
    if T <: Uniform
        lower, upper = stan_args
        (0 <= lower < upper <= 1) || error(
            "sbimpl: `$spelling ~ Uniform($lower, $upper)` must stay inside " *
            "the differenced-AR persistence bounds `[0, 1]`.")
        return (; rhs, lower, upper)
    end
    (; rhs, lower=0.0, upper=1.0)
end

# The base SLIC behind each submodel name, and the exact LHS each declares `rho`
# with -- three of the six type it per-axis, three leave it a plain scalar. Val
# dispatch with NO fallback method makes a renamed or newly added submodel a
# MethodError at emission time rather than a silently unconfigured prior.
_sb_gp_submodel(::Val{:_sb_gp}) = _sb_gp
_sb_gp_submodel(::Val{:_sb_gp_aniso}) = _sb_gp_aniso
_sb_gp_submodel(::Val{:_sb_hsgp}) = _sb_hsgp
_sb_gp_submodel(::Val{:_sb_hsgp_aniso}) = _sb_hsgp_aniso
_sb_gp_submodel(::Val{:_sb_hsgp_by}) = _sb_hsgp_by
_sb_gp_submodel(::Val{:_sb_hsgp_by_aniso}) = _sb_hsgp_by_aniso
_sb_gp_submodel(::Val{:_sb_hsgp_latent}) = _sb_hsgp_latent
_sb_gp_submodel(::Val{:_sb_hsgp_latent_orthogonal}) = _sb_hsgp_latent_orthogonal
_sb_gp_submodel(::Val{:_sb_gp_periodic}) = _sb_gp_periodic
_sb_gp_submodel(::Val{:_sb_hsgp_periodic}) = _sb_hsgp_periodic

_sb_gp_rho_lhs(::Val{:_sb_gp}) = :rho
_sb_gp_rho_lhs(::Val{:_sb_gp_aniso}) = :(rho :: vector[n_axes])
_sb_gp_rho_lhs(::Val{:_sb_hsgp}) = :rho_iso
_sb_gp_rho_lhs(::Val{:_sb_hsgp_aniso}) = :(rho :: vector[n_axes])
_sb_gp_rho_lhs(::Val{:_sb_hsgp_by}) = :rho_iso
_sb_gp_rho_lhs(::Val{:_sb_hsgp_by_aniso}) = :(rho :: vector[n_axes])
_sb_gp_rho_lhs(::Val{:_sb_hsgp_latent}) = :rho_iso
_sb_gp_rho_lhs(::Val{:_sb_hsgp_latent_orthogonal}) = :rho_iso
_sb_gp_rho_lhs(::Val{:_sb_gp_periodic}) = :rho
_sb_gp_rho_lhs(::Val{:_sb_hsgp_periodic}) = :rho_iso

function _sb_gp_prior_stmt(lhs, cfg)
    rhs = copy(cfg.rhs)
    kws = Any[Expr(:kw, :lower, cfg.lower)]
    isnothing(cfg.upper) || push!(kws, Expr(:kw, :upper, cfg.upper))
    insert!(rhs.args, 2, Expr(:parameters, kws...))
    Expr(:call, :~, lhs, rhs)
end

# Symbol in, Symbol out when nothing is configured: an unconfigured formula keeps
# emitting the bare submodel name its transpile module resolves, and only a
# configured one pays for a merged `SlicModel` VALUE spliced into the generated
# call site.
function _sb_gp_submodel_expr(submodel::Symbol, term_overrides, t)
    rho_cfg = _sb_term_cfg(term_overrides, t, :length_scale)
    sigma_cfg = _sb_term_cfg(term_overrides, t, :sigma)
    (isnothing(rho_cfg) && isnothing(sigma_cfg)) && return submodel
    v = Val(submodel)
    stmts = Any[]
    isnothing(rho_cfg) || push!(stmts, _sb_gp_prior_stmt(_sb_gp_rho_lhs(v), rho_cfg))
    isnothing(sigma_cfg) || push!(stmts, _sb_gp_prior_stmt(:sigma, sigma_cfg))
    Base.merge(_sb_gp_submodel(v), stmts...)
end

function _sb_dar_submodel_expr(term_overrides, t)
    beta_cfg = _sb_term_cfg(term_overrides, t, :ar)
    sigma_cfg = _sb_term_cfg(term_overrides, t, :sigma)
    (isnothing(beta_cfg) && isnothing(sigma_cfg)) && return :_sb_dar1
    stmts = Any[]
    isnothing(beta_cfg) || push!(stmts, _sb_gp_prior_stmt(:beta, beta_cfg))
    isnothing(sigma_cfg) || push!(stmts, _sb_gp_prior_stmt(:sigma, sigma_cfg))
    Base.merge(_sb_dar1, stmts...)
end

# The single carrier every prior surface rides on. Folding the term dict into
# the record `_sb_effect_prior_overrides` already produces keeps the whole
# threading path — nine `_sb_emit!`/`_sb_sampling!` signatures deep — unchanged,
# and lets a formula that configures ONLY a term parameter still reach
# `_sb_linear_predictor!`.
function _sb_prior_overrides(brmi::BRMI)
    effects = _sb_effect_prior_overrides(brmi)
    terms = _sb_term_prior_overrides(brmi)
    isempty(terms) && return effects
    out = Dict{Symbol,Any}()
    for lp in union(keys(effects), keys(terms))
        e = get(effects, lp, nothing)
        out[lp] = (; pop = isnothing(e) ? nothing : e.pop,
                     cat = isnothing(e) ? Dict{Symbol,Any}() : e.cat,
                     term = get(terms, lp, Dict{Symbol,Any}()))
    end
    out
end

function _sb_cat_address_map(brmi::BRMI, lhs::Symbol)
    entries = _sb_cat_entries(brmi, lhs)
    out = Dict{Symbol,Symbol}()
    isnothing(entries) && return out
    # Exact emitted names bind first and are never displaced: a `mu ~ factor(g) +
    # factor(g; ref=3)` emits `g` AND `g__ref_3`, and the latter's stripped alias
    # `g` must not steal the former's own name. Aliases then fill in only where
    # they are unambiguous -- an alias wanted by two blocks binds to neither, so
    # such a model refuses the address instead of silently priming one block.
    exact = Set{Symbol}(e.address for e in entries)
    for e in entries
        out[e.address] = e.emitted
    end
    clashes = Set{Symbol}()
    for e in entries, a in _sb_cat_addresses(e.address)
        a in exact && continue
        haskey(out, a) && out[a] !== e.emitted && (push!(clashes, a); continue)
        out[a] = e.emitted
    end
    foreach(a -> delete!(out, a), clashes)
    out
end

_popcoef_show(t::ExprColumn) = string(getf(t))
_popcoef_show(t::NamedColumn) = string(name(t))
_popcoef_show(t) = string(t)

# Classify a predictor term as "direct" (allocates its own parameters, no
# popefs multiplication). Matches vimpl: integer-backed NamedColumns are
# treated as treatment-coded categoricals; floats go through popefs.
# Value-narrowing chain: returns the underlying integer/categorical level
# vector if `t` is a NamedColumn over a DataColumn over a categorical-shaped
# vector, else `nothing`. Replaces the old Bool-predicate trio
# (`_sb_is_categorical` / `_is_cat_data` / `_is_cat_vec`); call sites compose
# with `isnothing(_sb_cat_levels(t))` instead of carrying the predicate.
_sb_cat_levels(t::NamedColumn) = _sb_cat_levels_data(parent(t))
_sb_cat_levels(_t) = nothing

_sb_cat_levels_data(d::DataColumn) = _sb_cat_levels_vec(parent(d))
_sb_cat_levels_data(_d) = nothing

_sb_cat_levels_vec(v::AbstractVector{<:Integer}) = v
_sb_cat_levels_vec(v::CA.CategoricalVector) = v
_sb_cat_levels_vec(_v) = nothing

# Free-summand terms (no popefs beta): `mo1(c)`, `s(x)`, `t2(x,z)`, categoricals.
# Categoricals emit
# `cat_<lp>_<c> ~ _sb_cat(; x=<c>_idx, n_levels=<c>_n_levels)`.
# `mo1(c)` reuses `_sb_mo`; smooths own their complete fixed + penalized bases.
_sb_emit_direct!(stmts, data, target::Symbol, t::NamedColumn, summands;
                 cat_overrides=Dict{Symbol,Any}(), kwargs...) = begin
    block = _sb_cat_block_name(target, name(t))
    _sb_emit_cat!(stmts, data, target, t, summands;
                  prior=get(cat_overrides, block, nothing))
end
function _sb_emit_direct!(stmts, data, target::Symbol, t::ExprColumn, summands;
                          group_block_lookup=Dict(), cat_overrides=Dict{Symbol,Any}(),
                          term_overrides=Dict{Symbol,Any}())
    f = getf(t)
    if f === gp || f === hsgp
        push!(summands, _sb_predictor_term!(stmts, data, f, t;
                                            group_block_lookup, term_overrides))
        return
    end
    _sb_emit_direct_expr!(stmts, data, target, getf(t), t, summands; term_overrides)
end
function _sb_emit_direct_expr!(_stmts, data, _target::Symbol,
                               ::typeof(offset), t, summands; kwargs...)
    args = getargs(t)
    length(args) == 1 || error(
        "sbimpl: `offset(x)` expects exactly one positional argument, got $(length(args))")
    isempty(getkwargs(t)) || error("sbimpl: `offset(x)` does not accept keyword arguments")
    # Unlike `protect`, offset is a direct model expression: it may reference
    # raw data, an already-declared sampled scalar, or a composition of either.
    # `_sb_scalar_expr` preserves that provenance in the emitted Stan expression
    # and, critically, introduces no `popefs` coefficient.
    push!(summands, _sb_scalar_expr(only(args), data))
end
function _sb_emit_direct_expr!(stmts, data, target::Symbol, ::typeof(mo1), t, summands;
                               term_overrides=Dict{Symbol,Any}())
    inner_name, raw = _sb_inner_data(:mo1, only(getargs(t)))
    n_levels, idx = _sb_level_index(raw)
    col_name = Symbol(:mo1_, inner_name)
    if n_levels < 2
        # Single-level factor: 0 increments -> the monotonic effect is
        # identically 0. Contribute a scalar `0.0` summand and NEVER ask Sb for
        # the `simplex[0]` that `_sb_mo`'s dirichlet would declare -- Stan rejects
        # a zero-dim SUM-constrained type (simplex / sum_to_zero_vector /
        # unit_vector). A scalar keeps the summand list non-empty (so the
        # assembler never emits `+()`) and broadcasts away against the other
        # vector summands, WITHOUT leaking a derived `mo1_<c>` data column that
        # `reprocess` could not regenerate. `mu ~ 1 + mo1(c)` stays a vector via
        # the intercept; a bare `mu ~ 0 + mo1(c)` degenerates to the scalar 0.
        push!(summands, 0.0)
        return
    end
    idx_name = Symbol(inner_name, :_idx)
    data[idx_name] = idx
    alpha = _sb_mo_alpha_expr(term_overrides, t, n_levels)
    push!(stmts, :($col_name ~ _sb_mo(; x=$idx_name, alpha=$alpha)))
    push!(summands, col_name)
end
function _sb_emit_direct_expr!(stmts, data, target::Symbol, ::typeof(s), t, summands;
                               term_overrides=Dict{Symbol,Any}())
    push!(summands, _sb_predictor_term!(stmts, data, s, t; term_overrides))
end
function _sb_emit_direct_expr!(stmts, data, target::Symbol, ::typeof(t2), t, summands;
                               term_overrides=Dict{Symbol,Any}())
    push!(summands, _sb_predictor_term!(stmts, data, t2, t; target, term_overrides))
end
function _sb_emit_direct_expr!(stmts, data, target::Symbol, ::typeof(dar), t, summands;
                               term_overrides=Dict{Symbol,Any}())
    push!(summands, _sb_predictor_term!(stmts, data, dar, t; target, term_overrides))
end
_sb_emit_direct_expr!(_stmts, _data, _target::Symbol, f, _t, _summands; kwargs...) =
    error("sbimpl: unsupported direct-summand term `$f`")

# Categorical population-level predictor. Allocates K-1 betas via `_sb_cat`
# and pushes the per-row contribution column into `summands`. No data-shape
# branch: at K == 1 the usual `_sb_cat` path degenerates uniformly — `std_normal(;
# n=0)` -> `vector[0]`, `append_row(0., beta)[x]` -> an all-zero column (the lone
# level is absorbed by the intercept, or vanishes for an intercept-less
# predictor). The `PreprocEntry(:factor)` provenance is recorded for every K, so
# frozen reprocess / unseen-level fail-closed behaves identically at K == 1; and
# an `effect(...)` prior on a K == 1 block simply has zero contrasts to apply to.
function _sb_emit_cat!(stmts, data, target::Symbol, t::NamedColumn, summands;
                       prior=nothing)
    backing = parent(t)
    n_levels, idx = _sb_level_index(parent(backing))
    col_name = _sb_cat_block_name(target, name(t))
    idx_name = Symbol(name(t), :_idx)
    n_name   = Symbol(name(t), :_n_levels)
    data[idx_name] = idx
    data[n_name]   = n_levels
    # Frozen level set drives the K-1 treatment-contrast betas; reprocess
    # re-codes a new df against it and updates the `<x>_n_levels` count key
    # (derived from raw_ref). Dimension-coupled (unseen level / changed count).
    _sb_record_preproc!(data, idx_name,
        PreprocEntry(:factor, _sb_fit_levels(parent(backing)), name(t), true))
    if isnothing(prior)
        push!(stmts, :($col_name ~ _sb_cat(; x=$idx_name, n_levels=$n_name)))
    else
        loc, scale = _sb_effect_normal_args(prior)
        push!(stmts, :($col_name ~ _sb_cat_normal(;
            x=$idx_name, n_levels=$n_name, beta_loc=$loc, beta_scale=$scale)))
    end
    push!(summands, col_name)
end

# Expand a ranef LHS term into one-or-more design-matrix column references.
# Continuous/intercept terms produce a single column; categorical NamedColumns
# expand to K-1 treatment-coded dummy columns (level 1 is reference), matching
# the design-matrix that brms / lme4 build for `(1 + c | g)`.
# `gterms` is the full LHS-terms list of the current ranef block, threaded so
# an intercept term (`t === 1`) can probe peer terms in the same block for a
# deterministic length probe (analogous to how `pop_terms` is threaded on the
# population path; see `_sb_pop_cols!`).
# `group_idx` names this block's per-row grouping index. A Z column's row axis
# IS the grouping factor's by construction, so passing it settles the intercept
# length probe outright (tier 1c) instead of leaving it to guess — see
# `_sb_predictor_col(::Int, ...)`. Callers that have no flat per-row index in
# hand omit it; `mm(...)`'s `<mm>_idx` is an n_obs x n_memberships MATRIX, so
# `num_elements` would give it rows*cols and it is deliberately NOT threaded.
function _sb_ranef_cols!(cols, data, stmts, t, gterms=(); group_idx=nothing,
                         term_overrides=Dict{Symbol,Any}())
    _sb_ranef_cols_dispatch!(cols, data, stmts, t, _sb_cat_levels(t), gterms;
                             group_idx, term_overrides)
end
_sb_ranef_cols!(cols, data, stmts, t::ExprColumn{typeof(offset)}, gterms=(); kwargs...) =
    error("sbimpl: `offset(...)` is a population-level fixed contribution and cannot appear inside a random-effects term")
_sb_ranef_cols_dispatch!(cols, data, stmts, t, ::Nothing, gterms=();
                         group_idx=nothing, term_overrides=Dict{Symbol,Any}()) =
    _sb_maybe_push_col!(cols, _sb_predictor_col(
        t, data, stmts, gterms; group_idx, term_overrides))
function _sb_ranef_cols_dispatch!(cols, data, _stmts, t, levels, _gterms=();
                                  group_idx=nothing, term_overrides=nothing)
    n_levels, idx = _sb_level_index(levels)
    # Single-level factor: `2:n_levels` is empty, so this contributes 0 dummy
    # columns uniformly (no shape special-case) — a `(1 + c | g)` degenerates to
    # intercept-only, matching how the population path drops a K=1 factor.
    fitted_levels = _sb_fit_levels(levels)
    for lvl in 2:n_levels
        col_name = Symbol(name(t), :_dummy_, lvl)
        data[col_name] = Float64[l == lvl ? 1.0 : 0.0 for l in idx]
        _sb_record_preproc!(data, col_name, PreprocEntry(
            :ranef_factor_dummy,
            (; levels=fitted_levels, level=lvl, n_levels),
            name(t), true))
        push!(cols, col_name)
    end
end

# Collected ranef handling. Terms that share a grouping symbol are merged into
# a single correlated block (matches vimpl / brms: `(1 | g) + (x | g)` is the
# same as `(1 + x | g)`; `(1 | g) + (0 + x | g)` also collapses -- the two
# terms' LKJ and tau parameters are shared by design). Per-group we build:
#   Z_<target>_<g> = hcat(<col_1>, <col_2>, ...)   # n x K
# and emit one `~ ranef_correlated(...)` (K >= 2) or `~ ranef_intercept(...)`
# (K == 1, intercept-only) call. Scalar-slope-only blocks (K == 1 but not an
# intercept) also go through ranef_correlated -- the math degenerates gracefully.
# `(... | rhs)` -> walker-side group descriptor. Bare NamedColumn and `gr(g)`
# (no kwargs) both collapse to the inner NamedColumn (plain correlated block);
# `gr(g; by=b)` returns `(group, by)` so the emitter allocates the stratified
# `ranef_correlated_by` block. Mirrors vimpl's `_normalize_group`. Also extracts
# brms's `gr(g, id=<sym|str>)`, producing an id Symbol which the caller carries
# alongside the group descriptor to drive cross-formula bucket coalescing.
_sb_id_sym(::Nothing) = nothing
_sb_id_sym(s::Symbol) = s
_sb_id_sym(s::AbstractString) = Symbol(s)
_sb_id_sym(x) = error("sbimpl: `gr(...; id=...)` expects a Symbol or String, got $(typeof(x))")

_sb_normalize_group(g::NamedColumn) = (g, nothing)
_sb_normalize_group(g::MultiMembershipTerm) = (g, nothing)
_sb_normalize_group(g::ExprColumn) = begin
    getf(g) === gr || error("sbimpl: expected NamedColumn or `gr(...)` on RHS of `|`, got `$(getf(g))`")
    args = getargs(g); kw = getkwargs(g)
    length(args) == 1 || error("sbimpl: `gr(...)` expects exactly one positional group, got $(length(args))")
    group_raw = args[1]
    group = _as_named_column(group_raw)
    isnothing(group) && error("sbimpl: `gr(...)` expects a NamedColumn group, got $(typeof(group_raw))")
    by_raw = get(kw, :by, nothing)
    id_sym = _sb_id_sym(get(kw, :id, nothing))
    by_raw === nothing && return (group, id_sym)
    by = _as_named_column(by_raw)
    isnothing(by) && error("sbimpl: `gr(...; by=...)` expects a NamedColumn for `by`, got $(typeof(by_raw))")
    ((group, by), id_sym)
end
_sb_normalize_group(g) = error("sbimpl: expected NamedColumn or `gr(...)` on RHS of `|`, got $(typeof(g))")

# Walker-side key used to coalesce ran_terms. Plain group -> `Symbol`; stratified
# `gr(g, by=b)` -> `(Symbol, Symbol)` so the two don't accidentally merge.
_sb_group_key(g::NamedColumn) = name(g)
_sb_group_key(g::Tuple{NamedColumn,NamedColumn}) = (name(g[1]), name(g[2]))
_sb_group_key(g::MultiMembershipTerm) =
    (:mm, Tuple(name(x) for x in getargs(g)),
     isnothing(getfield(g, :weights)) ? nothing : Tuple(name(x) for x in getfield(g, :weights)),
     getfield(g, :normalize))

_sb_mm_group_names(g::MultiMembershipTerm) = _brm_mm_group_names(g)
_sb_mm_weight_names(g::MultiMembershipTerm) = _brm_mm_weight_names(g)
_sb_mm_suffix(g::MultiMembershipTerm) = _brm_mm_suffix(g)

# Normalize `(expr | group)` vs `(expr | id | group)` into (id_sym, lhs, descriptor)
# where id_sym === nothing signals the plain (non-ID'd) case. Surface-level
# `gr(g, id=...)` with a plain `|` likewise produces a non-nothing id_sym.
# `|` rewritten by macro.jl: ExprColumn{|}(lhs, id_sym::Symbol, group) for `|ID|`.
function _sb_ranef_parts(rt::ExprColumn)
    getf(rt) === (|) || error("sbimpl: expected `|` ExprColumn, got `$(getf(rt))`")
    args = getargs(rt)
    if length(args) == 2
        lhs, raw_group = args
        desc, id_sym = _sb_normalize_group(raw_group)
        desc isa MultiMembershipTerm && id_sym !== nothing && error(
            "sbimpl: `mm(...)` cannot be combined with `gr(...; id=...)`; ",
            "multi-membership terms already define one shared coefficient block")
        (id_sym, lhs, desc)
    elseif length(args) == 3
        lhs, id_sym_raw, raw_group = args
        id_sym = _as_symbol(id_sym_raw)
        isnothing(id_sym) && error("sbimpl: `(e | ID | g)` middle must be a Symbol, got $(typeof(id_sym_raw))")
        desc, gr_id_sym = _sb_normalize_group(raw_group)
        desc isa MultiMembershipTerm && error(
            "sbimpl: `(e | ID | mm(...))` is not supported; `mm(...)` already ",
            "defines one shared coefficient block across its membership columns")
        gr_id_sym === nothing ||
            error("sbimpl: `(e | ID | g)` cannot also carry `gr(g, id=...)` (got `$gr_id_sym`)")
        (id_sym, lhs, desc)
    else
        error("sbimpl: malformed ranef term, expected 2 or 3 args, got $(length(args))")
    end
end

# Per-group level of `g` -> stratum level of `by`. Errors if any group level
# straddles multiple strata. Ported from vimpl's `_stratum_idx`.
_sb_stratum_idx(g_idx::AbstractVector{Int}, b_idx::AbstractVector{Int},
                 gname, bname) =
    _brm_group_strata(g_idx, b_idx, maximum(g_idx), Symbol(gname), Symbol(bname))

function _sb_emit_ranefs!(stmts, data, target::Symbol, ran_terms, summands;
                           id_lookup=_sb_empty_id_lookup(),
                           brmi_key::Symbol=target,
                           cv_groups=Set{Symbol}(),
                           centered_groups=Set{Symbol}(),
                           r2d2_scale=nothing,
                           term_overrides=Dict{Symbol,Any}())
    isempty(ran_terms) && return
    # Partition: ID'd terms route to the pre-emitted shared bucket; plain terms
    # coalesce per-target via the existing `ranef_correlated` block. Bare-
    # NamedColumn and gr-by groups key differently so they never coalesce.
    plain_keys_seen = Any[]
    plain_by_group = Dict{Any, Vector{Any}}()
    plain_descs = Dict{Any, Any}()
    id_keys_seen = Any[]
    id_terms_by_bucket = Dict{Tuple{Symbol,Any}, Vector{Any}}()
    for rt in ran_terms
        id_sym, lhs, desc = _sb_ranef_parts(rt)
        if id_sym === nothing
            k = _sb_group_key(desc)
            haskey(plain_by_group, k) || (push!(plain_keys_seen, k); plain_by_group[k] = Any[]; plain_descs[k] = desc)
            append!(plain_by_group[k], _sb_terms(lhs))
        else
            k = (id_sym, _sb_group_key(desc))
            haskey(id_terms_by_bucket, k) || (push!(id_keys_seen, k); id_terms_by_bucket[k] = Any[])
            append!(id_terms_by_bucket[k], _sb_terms(lhs))
        end
    end
    for k in plain_keys_seen
        gterms = plain_by_group[k]
        desc = plain_descs[k]
        isempty(gterms) && error("sbimpl: ranef `(… | $k)` has no terms after dropping `0`")
        _sb_emit_ranef_block!(stmts, data, target, desc, gterms, summands;
                              cv_groups, centered_groups, r2d2_scale, term_overrides)
    end
    for k in id_keys_seen
        gterms = id_terms_by_bucket[k]
        isempty(gterms) && error("sbimpl: ranef `(… | $(k[1]) | $(k[2]))` has no terms after dropping `0`")
        # The `|ID|` bucket's shared draws block is emitted in prepass 2, which
        # receives `cv_groups` / `centered_groups` and picks the matching
        # `ranef_correlated_draws{,_cv,_centered}` variant there. Nothing to do
        # here beyond slicing it per sub-formula.
        info = get(id_lookup, (brmi_key, k), nothing)
        info === nothing && error("sbimpl: internal — no pre-emitted bucket for (target=$brmi_key, id=$(k[1]), group=$(k[2]))")
        _sb_emit_id_ranef_block!(stmts, data, target, info, gterms, summands;
                                 term_overrides)
    end
end

# Emit a single ranef block for one normalized group descriptor.
# `cv_groups`: groups whose RE should be sized cv-contagiously (opt-in, for CV
# model artifacts). There is no separate `_cv` submodel -- the SIZE EXPRESSION
# passed here is the whole mechanism: `maximum(<g>_idx)` instead of the data
# scalar `n_<g>` carries the taint from `maybecv(:<g>_idx)` into the declared
# size and flips the RE to a generated-quantities population re-draw. Empty by
# default. See the cv-contagion note above `ranef_intercept` in this file.
function _sb_emit_ranef_block!(stmts, data, target::Symbol, group::NamedColumn, gterms, summands;
                                cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(),
                                r2d2_scale=nothing,
                                term_overrides=Dict{Symbol,Any}())
    g_backing = _as_data_column(parent(group))
    isnothing(g_backing) && error("sbimpl: group `$(name(group))` must be a raw data column")
    g = name(group)
    is_cv = g in cv_groups
    is_centered = g in centered_groups
    n_levels, g_idx = _sb_level_index(parent(g_backing))
    idx_name = Symbol(g, :_idx)
    n_name   = Symbol(:n_, g)
    data[idx_name] = g_idx
    data[n_name]   = n_levels
    _sb_record_group_index!(data, idx_name, n_name, g, parent(g_backing))
    r_name = Symbol(:r_, target, :_, g)
    # The size expression the non-centered submodels are given. Bound to a named
    # local in the cv case so it appears once as `int <r>_n_g = max(<g>_idx);`
    # rather than being inlined into every declaration -- same shape as
    # `_sb_emit_id_bucket_sampling!`.
    n_groups_expr = n_name
    if is_cv
        n_groups_expr = Symbol(r_name, :_n_g)
        push!(stmts, :($n_groups_expr = maximum($idx_name)))
    end
    if !isnothing(r2d2_scale)
        # Derived residual scale. cv / centered are refused for the same reason
        # as in `_sb_emit_id_bucket_sampling!`: both interact with a derived
        # scale in ways nobody has designed, so they fail loudly rather than
        # silently sampling something else.
        is_cv && error(
            "sbimpl: group `$g` is in `cv_groups` and also carries an `r2d2` " *
            "decomposition; the cv re-draw path for a derived residual scale " *
            "is not implemented")
        is_centered && error(
            "sbimpl: group `$g` is in `centered_groups` and also carries an " *
            "`r2d2` decomposition; the centered path for a derived residual " *
            "scale is not implemented")
        length(gterms) == 1 && gterms[1] === 1 || error(
            "sbimpl: `r2d2` currently requires the random effect playing the " *
            "residual role to be a single intercept — `(1 | $g)`. Predictor " *
            "`$target` has $(length(gterms)) random-effect terms on `$g`, and " *
            "splitting the derived residual variance `(1 - R2) * tau_bsv^2` " *
            "among several margins needs a second simplex that the flat " *
            "decomposition does not build.")
        scale_name = Symbol(r_name, :_r2d2_scale)
        push!(stmts, :($scale_name = $r2d2_scale))
        push!(stmts, :($r_name ~ ranef_intercept_r2d2(;
            group_idx=$idx_name, n_groups=$n_groups_expr, scale=$scale_name)))
        push!(summands, r_name)
        return
    end
    if length(gterms) == 1 && gterms[1] === 1
        # (1 | g) fast/equivalent path -- stays on ranef_intercept so the
        # emitted Stan matches existing sb.3 smoke tests bit for bit.
        if is_centered
            push!(stmts, :($r_name ~ ranef_intercept_centered(; group_idx=$idx_name, n_groups=$n_name)))
        else
            push!(stmts, :($r_name ~ ranef_intercept(; group_idx=$idx_name, n_groups=$n_groups_expr)))
        end
    else
        col_exprs = Any[]
        for t in gterms
            _sb_ranef_cols!(col_exprs, data, stmts, t, gterms;
                            group_idx=idx_name, term_overrides)
        end
        if isempty(col_exprs)
            # Every slope term degenerated to zero columns (e.g. `(0 + c | g)`
            # with `c` single-level): the block spans no coefficients and is a
            # no-op. Emit no Z / correlated draw and add no summand — an empty
            # `hcat()` has no traceable shape (StanBlocks cannot type it), and a
            # zero-column random effect contributes nothing. One uniform
            # emission: the block degenerates rather than erroring.
            return
        end
        Z_name = Symbol(:Z_, target, :_, g)
        k_name = Symbol(:n_terms_, target, :_, g)
        data[k_name] = length(col_exprs)
        push!(stmts, :($Z_name = $(Expr(:call, :hcat, col_exprs...))))
        if is_centered
            push!(stmts, :($r_name ~ ranef_correlated_centered(;
                Z=$Z_name, group_idx=$idx_name,
                n_groups=$n_name, n_terms=$k_name)))
        else
            push!(stmts, :($r_name ~ ranef_correlated(;
                Z=$Z_name, group_idx=$idx_name,
                n_groups=$n_groups_expr, n_terms=$k_name)))
        end
    end
    push!(summands, r_name)
end

function _sb_emit_ranef_block!(stmts, data, target::Symbol,
                                term::MultiMembershipTerm, gterms, summands;
                                cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(),
                                r2d2_scale=nothing,
                                term_overrides=Dict{Symbol,Any}())
    isnothing(r2d2_scale) || error(
        "sbimpl: `r2d2` decompositions over typed `mm(...)` multi-membership " *
        "random effects are not yet supported")
    group_names = _sb_mm_group_names(term)
    weight_names = _sb_mm_weight_names(term)
    cv_hit = intersect(Set(group_names), cv_groups)
    isempty(cv_hit) || error(
        "sbimpl: cv-contagious sizing for `mm(...)` is not yet supported ",
        "(requested membership columns: $(collect(cv_hit)))")
    centered_hit = intersect(Set(group_names), centered_groups)
    isempty(centered_hit) || error(
        "sbimpl: centered parameterization for `mm(...)` is not supported ",
        "(requested membership columns: $(collect(centered_hit))); leave this ",
        "shared block non-centered")

    raw_groups = Tuple(begin
        backing = _as_data_column(parent(g))
        isnothing(backing) && error(
            "sbimpl: `mm(...)` group `$(name(g))` must be a raw data column")
        parent(backing)
    end for g in getargs(term))
    raw_weights = if isnothing(getfield(term, :weights))
        nothing
    else
        Tuple(begin
            backing = _as_data_column(parent(w))
            isnothing(backing) && error(
                "sbimpl: `mm(...)` weight `$(name(w))` must be a raw data column")
            parent(backing)
        end for w in getfield(term, :weights))
    end
    prepared = _sb_prepare_mm(raw_groups, raw_weights, getfield(term, :normalize);
                              group_names, weight_names)

    suffix = _sb_mm_suffix(term)
    idx_name = Symbol(suffix, :_idx)
    weight_name = Symbol(suffix, :_weights)
    n_name = Symbol(:n_, suffix)
    n_obs_name = Symbol(:n_obs_, suffix)
    n_memberships_name = Symbol(:n_memberships_, suffix)
    data[idx_name] = prepared.group_idx
    data[weight_name] = prepared.weights
    data[n_name] = length(prepared.levels)
    data[n_obs_name] = prepared.n_obs
    data[n_memberships_name] = prepared.n_memberships
    const_ = (; levels=prepared.levels, weight_key=weight_name,
              n_groups_key=n_name, n_obs_key=n_obs_name,
              n_memberships_key=n_memberships_name,
              normalize=getfield(term, :normalize))
    raw_ref = (; groups=group_names, weights=weight_names)
    _sb_record_preproc!(data, idx_name,
        PreprocEntry(:multi_membership, const_, raw_ref, true))

    b_name = Symbol(:b_, target, :_, suffix)
    r_name = Symbol(:r_, target, :_, suffix)
    if length(gterms) == 1 && gterms[1] === 1
        push!(stmts, :($b_name ~ ranef_intercept_draws(;
            group_idx=$idx_name, n_groups=$n_name)))
        push!(stmts, :($r_name = multi_membership_intercept(
            $b_name, $idx_name, $weight_name, $n_obs_name, $n_memberships_name)))
    else
        col_exprs = Any[]
        for t in gterms
            _sb_ranef_cols!(col_exprs, data, stmts, t, gterms; term_overrides)
        end
        Z_name = Symbol(:Z_, target, :_, suffix)
        k_name = Symbol(:n_terms_, target, :_, suffix)
        data[k_name] = length(col_exprs)
        push!(stmts, :($Z_name = $(Expr(:call, :hcat, col_exprs...))))
        push!(stmts, :($b_name ~ ranef_correlated_draws(;
            Z=$Z_name, group_idx=$idx_name,
            n_groups=$n_name, n_terms=$k_name)))
        push!(stmts, :($r_name = multi_membership_correlated(
            $Z_name, $b_name, $idx_name, $weight_name,
            $n_obs_name, $n_memberships_name)))
    end
    push!(summands, r_name)
end

function _sb_emit_ranef_block!(stmts, data, target::Symbol, group::Tuple{NamedColumn,NamedColumn}, gterms, summands;
                                cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(),
                                r2d2_scale=nothing,
                                term_overrides=Dict{Symbol,Any}())
    isnothing(r2d2_scale) || error(
        "sbimpl: `r2d2` decompositions over stratified `gr(g, by=b)` random " *
        "effects are not yet supported")
    gcol, bcol = group
    g_backing = _as_data_column(parent(gcol))
    b_backing = _as_data_column(parent(bcol))
    isnothing(g_backing) && error("sbimpl: group `$(name(gcol))` must be a raw data column")
    isnothing(b_backing) && error("sbimpl: `by=$(name(bcol))` must be a raw data column")
    g, b = name(gcol), name(bcol)
    g in cv_groups && error(
        "sbimpl: cv-contagious sizing requested for group `$g`, but `$g` is a ",
        "stratified `gr($g, by=$b)` ranef whose `z` is declared via typed-LHS. ",
        "StanBlocks' typed-LHS forward derives cv from RHS call-args, not the ",
        "declared size, so a cv size yields a cv *parameter*, not a generated-",
        "quantities re-draw. Making `by=` blocks cv-contagious needs a bare-`z` ",
        "rewrite or a StanBlocks typed-LHS fix -- not yet supported.")
    g in centered_groups && error(
        "sbimpl: centered parameterization requested for group `$g`, but `$g` is ",
        "a stratified `gr($g, by=$b)` ranef whose per-group draw goes through the ",
        "typed-LHS `ranef_correlated_by` path (one Cholesky per stratum). The ",
        "centered variants emit a plate over a single shared covariance and have ",
        "no per-stratum form -- not yet supported. Use a plain `(… | $g)` or ",
        "`(… |ID| $g)` ranef, or leave `$g` non-centered.")
    n_groups, g_idx = _sb_level_index(parent(g_backing))
    n_strata, b_idx = _sb_level_index(parent(b_backing))
    stratum_idx = _sb_stratum_idx(g_idx, b_idx, g, b)
    # Block-local names include both `g` and `b` so this block never clashes
    # with a plain `(… | g)` block against the same group column.
    suffix = Symbol(g, :__by__, b)
    idx_name     = Symbol(suffix, :_idx)
    n_name       = Symbol(:n_, suffix)
    s_idx_name   = Symbol(suffix, :_stratum_idx)
    n_strata_nm  = Symbol(:n_strata_, suffix)
    data[idx_name]    = g_idx
    data[n_name]      = n_groups
    data[s_idx_name]  = stratum_idx
    data[n_strata_nm] = n_strata
    r_name = Symbol(:r_, target, :_, suffix)
    col_exprs = Any[]
    for t in gterms
        _sb_ranef_cols!(col_exprs, data, stmts, t, gterms;
                        group_idx=idx_name, term_overrides)
    end
    Z_name = Symbol(:Z_, target, :_, suffix)
    k_name = Symbol(:n_terms_, target, :_, suffix)
    data[k_name] = length(col_exprs)
    push!(stmts, :($Z_name = $(Expr(:call, :hcat, col_exprs...))))
    push!(stmts, :($r_name ~ ranef_correlated_by(;
        Z=$Z_name, group_idx=$idx_name,
        n_groups=$n_name, n_terms=$k_name,
        stratum_idx=$s_idx_name, n_strata=$n_strata_nm)))
    push!(summands, r_name)
end

# Pre-pass: harvest every `|ID|` ranef across all sub-formulas and bucket by
# (id_sym, group_key). Returns an OrderedDict keyed by bucket, carrying
# `(group_desc, per_target::Vector{Pair{Symbol, Vector}})` in appearance order.
# Non-ID'd ranef terms are left alone for the existing per-target emitter.
function _sb_collect_id_buckets(brmi::BRMI)
    buckets = OrderedCollections.OrderedDict{Tuple{Symbol,Any}, Any}()
    for (brmi_key, op_nc) in pairs(brmi.operations)
        op = _as_expr_column(parent(op_nc)); isnothing(op) && continue
        getf(op) === (~) || continue
        _, rhs_raw = getargs(op, 2)
        rhs = _as_expr_column(rhs_raw); isnothing(rhs) && continue
        for t_raw in _sb_terms(rhs)
            t = _as_expr_column(t_raw); isnothing(t) && continue
            getf(t) === (|) || continue
            id_sym, lhs, desc = _sb_ranef_parts(t)
            id_sym === nothing && continue
            k = (id_sym, _sb_group_key(desc))
            if !haskey(buckets, k)
                buckets[k] = (group_desc=desc, per_target=Pair{Symbol,Vector{Any}}[])
            else
                # Consistency check: same id must always pair with the same group.
                _sb_group_desc_matches(buckets[k].group_desc, desc) ||
                    error("sbimpl: `|$id_sym|` sees conflicting grouping factors ($(buckets[k].group_desc) vs $desc)")
            end
            push!(buckets[k].per_target, brmi_key => _sb_terms(lhs))
        end
    end
    buckets
end

# Expand one collected `|ID|` bucket with the exact ranef-column emitter used
# by Stan lowering. This is the single source of truth for public margin
# addresses: categorical terms therefore expose their emitted dummy-column
# labels, and formula order is preserved across predictors and terms.
function _sb_id_bucket_margins(bucket)
    out = NamedTuple[]
    scratch_data = Dict{Symbol,Any}()
    scratch_stmts = Any[]
    for (predictor, terms) in bucket.per_target
        for t in terms
            if t isa Integer
                t == 1 || error(
                    "ranefcoefnames: unsupported integer random-effect term `$t`")
                push!(out, (; predictor, coefficient=:Intercept))
                continue
            end
            cols = Any[]
            try
                _sb_ranef_cols!(cols, scratch_data, scratch_stmts, t, terms)
            catch err
                error("ranefcoefnames: cannot resolve random-effect column(s) for " *
                      "predictor `$predictor`: $(sprint(showerror, err))")
            end
            for col in cols
                col isa Symbol || error(
                    "ranefcoefnames: random-effect column for predictor " *
                    "`$predictor` is not symbol-addressable: $(repr(col))")
                push!(out, (; predictor, coefficient=col))
            end
        end
    end
    out
end

"""
    ranefcoefnames(brmi::BRMI, id::Symbol) -> Union{Vector{NamedTuple},Nothing}

Ordered `(predictor, coefficient)` addresses of the marginal SDs in the
shared random-effect block selected by public `|ID|` symbol `id`. The k-th
entry labels the k-th `tau` element emitted by the SBBRMI backend. Categorical
random slopes use the exact treatment-contrast column symbols emitted into the
random-effect design matrix.

Returns `nothing` when `id` is absent. Reusing one ID with multiple grouping
factors is ambiguous on the public ID-only surface and raises.
"""
function ranefcoefnames(brmi::BRMI, id::Symbol)
    buckets = _sb_collect_id_buckets(brmi)
    matches = Pair[k => bucket for (k, bucket) in pairs(buckets) if first(k) === id]
    isempty(matches) && return nothing
    length(matches) == 1 || error(
        "ranefcoefnames: `|$id|` identifies $(length(matches)) random-effect " *
        "blocks with different grouping factors; the ID-only address is ambiguous")
    _sb_id_bucket_margins(last(only(matches)))
end

# Shared by the random-effect margins and the smoothing scales of `s`/`t2`:
# both sample their SD through `brm_ranef_sd`, so both accept exactly the same
# family. `spelling` is the address as the formula wrote it, so the message
# names the statement the user can actually edit.
function _sb_ranef_sd_rate(spec, spelling::AbstractString="sd(:, ...)")
    T = _as_distribution_type(spec.family)
    (!isnothing(T) && T <: Exponential) || error(
        "sbimpl: `$spelling` currently supports `Exponential(scale)`; " *
        "got `$(spec.family)`. An unmentioned scale keeps the half-standard-" *
        "normal prior.")
    isempty(spec.keywords) || error(
        "sbimpl: `$spelling ~ Exponential(...)` does not accept keywords")
    args = map(_sb_effect_prior_arg, spec.arguments)
    length(args) in (0, 1) || error(
        "sbimpl: `$spelling ~ Exponential` expects zero or one Julia scale argument")
    scale = isempty(args) ? 1.0 : only(args)
    scale isa Real || error(
        "sbimpl: `$spelling` scale must be a numeric formula constant, " *
        "got $(repr(scale))")
    isfinite(scale) && scale > 0 || error(
        "sbimpl: `$spelling` scale must be finite and strictly positive, " *
        "got $scale")
    # Distributions.Exponential uses scale; Stan's exponential_lpdf uses rate.
    Float64(1.0 / scale)
end

function _sb_ranef_lkj(spec, n_terms::Int)
    T = _as_distribution_type(spec.family)
    (!isnothing(T) && T <: LKJCholesky) || error(
        "sbimpl: `cor(:, ID)` expects `LKJCholesky(K, eta)`; " *
        "got `$(spec.family)`")
    isempty(spec.keywords) || error(
        "sbimpl: `cor(:, ID) ~ LKJCholesky(...)` does not accept keywords")
    args = map(_sb_effect_prior_arg, spec.arguments)
    length(args) == 2 || error(
        "sbimpl: `LKJCholesky` correlation priors require dimension K and eta")
    k, eta = args
    k isa Integer || error(
        "sbimpl: `LKJCholesky(K, eta)` dimension K must be an integer " *
        "formula constant, got $(repr(k))")
    k == n_terms || error(
        "sbimpl: `LKJCholesky($k, ...)` does not match the addressed " *
        "random-effect block width $n_terms")
    eta isa Real || error(
        "sbimpl: LKJ eta must be a numeric formula constant, got $(repr(eta))")
    isfinite(eta) && eta > 0 || error(
        "sbimpl: LKJ eta must be finite and strictly positive, got $eta")
    Float64(eta)
end

# Resolve one SD statement onto the margins it claims. `nothing` means the
# whole block -- the base layer every unclaimed margin falls through to.
# Otherwise the return is the claimed margin indices plus the statement's
# SPECIFICITY, counted in concrete (non-`:`) slots beyond the ID exactly as on
# the population surface, so a more specific statement can override a broader
# one and an exact tie is an error rather than last-writer-wins.
function _sb_ranef_margin_index(spec, margins)
    spelling = "sd($(isnothing(spec.predictor) ? ":" : spec.predictor), " *
               "$(spec.id)" *
               (isnothing(spec.coefficient) ? "" : ", $(spec.coefficient)") * ")"
    if isnothing(spec.predictor) && isnothing(spec.coefficient)
        return nothing
    elseif isnothing(spec.predictor)
        # `sd(:, ID, coefficient)` -- one margin across EVERY predictor that
        # slices this block. Only spellable since the head-position grammar.
        hits = findall(m -> m.coefficient === spec.coefficient, margins)
        isempty(hits) && error(
            "sbimpl: `$spelling` matches no random-effect margin. Inspect " *
            "`ranefcoefnames(brmi, :$(spec.id))` for valid addresses.")
        return (hits, 1)
    elseif isnothing(spec.coefficient)
        hits = findall(m -> m.predictor === spec.predictor, margins)
        isempty(hits) && error(
            "sbimpl: `$spelling` matches no " *
            "random-effect margin. Inspect `ranefcoefnames(brmi, :$(spec.id))`.")
        length(hits) == 1 || error(
            "sbimpl: `$spelling` is ambiguous " *
            "because that predictor contributes $(length(hits)) margins; use " *
            "`sd($(spec.predictor), $(spec.id), coefficient)`.")
        return (hits, 1)
    else
        hits = findall(m -> m.predictor === spec.predictor &&
                            m.coefficient === spec.coefficient, margins)
        isempty(hits) && error(
            "sbimpl: `$spelling` matches no random-effect margin. Inspect " *
            "`ranefcoefnames(brmi, :$(spec.id))` for valid addresses.")
        length(hits) == 1 || error(
            "sbimpl: `$spelling` matches $(length(hits)) margins and is " *
            "therefore ambiguous")
        return (hits, 2)
    end
end

# Resolve formula-level random-effect prior statements onto collected ID
# buckets. The result is keyed exactly like `id_buckets`; entries are present
# only for explicitly configured buckets, preserving default emission byte for
# byte when the formula contains no ranef effect statements.
function _sb_ranef_effect_overrides(brmi::BRMI, id_buckets)
    # `sd(...) ~ r2d2(...)` is a derived-scale prior, not a distribution on a
    # sampled `tau`.  Resolve it in `_sb_ranef_r2d2_overrides`; the ordinary
    # resolver still sees `cor(...)` and every direct-scale SD statement.
    specs = [spec for spec in ranef_effect_priors(brmi)
             if !(spec.class === :sd && spec.family === r2d2)]
    isempty(specs) && return Dict{Tuple{Symbol,Any},NamedTuple}()
    margins = Dict{Tuple{Symbol,Any},Any}()
    for (key, bucket) in id_buckets
        bucket.group_desc isa Tuple && any(spec -> spec.id === first(key), specs) &&
            error("sbimpl: covariance-prior effects for stratified " *
                  "`|$(first(key))| gr(..., by=...)` buckets are not yet supported")
        margins[key] = _sb_id_bucket_margins(bucket)
    end
    resolved = _brm_resolve_ranef_effect_overrides(
        specs, margins; prefix="sbimpl")
    Dict{Tuple{Symbol,Any},NamedTuple}(key => value
        for (key, value) in resolved)
end

# ---- random-effect R2D2M2 variance decomposition --------------------------
#
# `sd(...) ~ r2d2(...)` is the partial-R2D2M2 spelling for a shared random-
# effect covariance block.  A block-wide address creates one global R² and one
# Dirichlet allocation across every selected marginal scale.  More-specific
# statements may replace only a margin's observation/reference scale:
#
#   sd(:, p)       ~ r2d2(mean_R2=.5, prec_R2=2,
#                          reference_scale=sigma_pk)
#   sd(qt_base, p) ~ r2d2(reference_scale=sigma_qt)
#
# With no block-wide statement, each addressed subset is its own decomposition;
# a one-margin subset has `simplex[1]` and therefore reduces exactly to an ICC
# prior.  Unaddressed margins retain an independent half-standard-Normal scale.
# Correlation remains a separate `cor(:, p) ~ LKJCholesky(K, eta)` prior.

function _sb_ranef_r2d2_reference(spec, spelling)
    haskey(spec.keywords, :reference_scale) || error(
        "sbimpl: `$spelling ~ r2d2(...)` requires `reference_scale=`; " *
        "use the observation/residual SD whose variance defines R²")
    value = _sb_effect_prior_arg(spec.keywords.reference_scale)
    if value isa Real
        isfinite(value) && value > 0 || error(
            "sbimpl: `$spelling` reference_scale must be finite and strictly " *
            "positive, got $value")
        return Float64(value)
    end
    value isa Symbol || error(
        "sbimpl: `$spelling` reference_scale must be a positive numeric " *
        "formula constant or an already-declared sampled scalar parameter, " *
        "got $(repr(value))")
    value
end

function _sb_ranef_r2d2_config(spec, spelling; override_only=false)
    isempty(spec.arguments) || error(
        "sbimpl: `$spelling ~ r2d2(...)` takes keyword arguments only")
    known = (:R2, :mean_R2, :prec_R2, :alpha, :concentration,
             :reference_scale)
    for key in keys(spec.keywords)
        key in known || error(
            "sbimpl: unknown r2d2 keyword `$key` for `$spelling`; supported " *
            "keywords are $(join(known, ", "))")
    end
    reference = _sb_ranef_r2d2_reference(spec, spelling)
    if override_only
        extras = [key for key in keys(spec.keywords)
                  if key !== :reference_scale]
        isempty(extras) || error(
            "sbimpl: a margin-specific `$spelling ~ r2d2(...)` under a " *
            "block-wide decomposition may override only `reference_scale`; " *
            "the block owns one global R² and concentration")
        return (; reference)
    end

    has_beta = haskey(spec.keywords, :R2)
    has_moments = haskey(spec.keywords, :mean_R2) ||
                  haskey(spec.keywords, :prec_R2)
    has_beta && has_moments && error(
        "sbimpl: `$spelling ~ r2d2(...)` must use either `R2=Beta(a,b)` or " *
        "`mean_R2=`/`prec_R2=`, not both")
    if has_beta
        r2_a, r2_b = _sb_r2d2_beta(spec.keywords.R2, spelling)
    else
        mean_r2 = get(spec.keywords, :mean_R2, 0.5)
        prec_r2 = get(spec.keywords, :prec_R2, 2.0)
        mean_r2 isa Real && isfinite(mean_r2) && 0 < mean_r2 < 1 || error(
            "sbimpl: `$spelling` mean_R2 must be a finite number strictly " *
            "between zero and one, got $(repr(mean_r2))")
        prec_r2 isa Real && isfinite(prec_r2) && prec_r2 > 0 || error(
            "sbimpl: `$spelling` prec_R2 must be finite and strictly positive, " *
            "got $(repr(prec_r2))")
        r2_a = Float64(mean_r2 * prec_r2)
        r2_b = Float64((1 - mean_r2) * prec_r2)
    end
    haskey(spec.keywords, :alpha) &&
        haskey(spec.keywords, :concentration) && error(
            "sbimpl: `$spelling ~ r2d2(...)` accepts either `alpha=` or " *
            "`concentration=`, not both")
    raw_alpha = get(spec.keywords, :alpha,
                    get(spec.keywords, :concentration, 1.0))
    alpha = _sb_r2d2_positive(raw_alpha, "concentration", spelling)
    (; reference, r2_a, r2_b, alpha)
end

function _sb_ranef_r2d2_spelling(spec)
    "sd($(isnothing(spec.predictor) ? ":" : spec.predictor), $(spec.id)" *
    (isnothing(spec.coefficient) ? "" : ", $(spec.coefficient)") * ")"
end

function _sb_ranef_r2d2_overrides(brmi::BRMI, id_buckets)
    all_specs = ranef_effect_priors(brmi)
    specs = [spec for spec in all_specs
             if spec.class === :sd && spec.family === r2d2]
    isempty(specs) && return Dict{Tuple{Symbol,Any},NamedTuple}()

    out = Dict{Tuple{Symbol,Any},NamedTuple}()
    for id in unique(spec.id for spec in specs)
        matches = [key for key in keys(id_buckets) if first(key) === id]
        isempty(matches) && error(
            "sbimpl: `sd(:, $id) ~ r2d2(...)` matches no shared `|$id|` " *
            "random-effect block")
        length(matches) == 1 || error(
            "sbimpl: public `|$id|` addresses $(length(matches)) blocks with " *
            "different grouping factors; use a unique ID")
        key = only(matches)
        bucket = id_buckets[key]
        bucket.group_desc isa Tuple && error(
            "sbimpl: `sd(...) ~ r2d2(...)` for stratified `|$id| " *
            "gr(..., by=...)` buckets is not yet supported")
        margins = _sb_id_bucket_margins(bucket)
        id_specs = [spec for spec in specs if spec.id === id]
        direct_sd = [spec for spec in all_specs
                     if spec.id === id && spec.class === :sd &&
                        spec.family !== r2d2]
        isempty(direct_sd) || error(
            "sbimpl: `|$id|` mixes `sd(...) ~ r2d2(...)` with a direct-scale " *
            "SD prior. R2D2 derives the addressed marginal scales; keep direct " *
            "SD priors on a different block or leave unaddressed margins at " *
            "their default half-Normal prior.")

        block_specs = [spec for spec in id_specs
                       if isnothing(spec.predictor) &&
                          isnothing(spec.coefficient)]
        length(block_specs) <= 1 || error(
            "sbimpl: duplicate block-wide `sd(:, $id) ~ r2d2(...)` statements")
        groups = NamedTuple[]
        if !isempty(block_specs)
            base_spec = only(block_specs)
            base = _sb_ranef_r2d2_config(
                base_spec, _sb_ranef_r2d2_spelling(base_spec))
            references = fill(base.reference, length(margins))
            claims = Dict{Int,Int}()
            for spec in id_specs
                spec === base_spec && continue
                spelling = _sb_ranef_r2d2_spelling(spec)
                override = _sb_ranef_r2d2_config(
                    spec, spelling; override_only=true)
                claim = _sb_ranef_margin_index(spec, margins)
                isnothing(claim) && error(
                    "sbimpl: duplicate block-wide R2D2 statement for `|$id|`")
                indices, rank = claim
                for index in indices
                    held = get(claims, index, -1)
                    rank == held && error(
                        "sbimpl: two R2D2 reference-scale statements are equally " *
                        "specific for margin $(margins[index]) of `|$id|`")
                    if rank > held
                        references[index] = override.reference
                        claims[index] = rank
                    end
                end
            end
            push!(groups, (; indices=collect(eachindex(margins)), references,
                            r2_a=base.r2_a, r2_b=base.r2_b,
                            alpha=base.alpha))
        else
            claimed = Dict{Int,String}()
            for spec in id_specs
                spelling = _sb_ranef_r2d2_spelling(spec)
                config = _sb_ranef_r2d2_config(spec, spelling)
                claim = _sb_ranef_margin_index(spec, margins)
                isnothing(claim) && error(
                    "sbimpl: internal block-wide R2D2 resolution error for `$spelling`")
                indices, _ = claim
                for index in indices
                    haskey(claimed, index) && error(
                        "sbimpl: `$spelling` overlaps $(claimed[index]) on " *
                        "margin $(margins[index]) of `|$id|`")
                    claimed[index] = spelling
                end
                push!(groups, (; indices=collect(indices),
                                references=fill(config.reference,
                                                length(indices)),
                                r2_a=config.r2_a, r2_b=config.r2_b,
                                alpha=config.alpha))
            end
        end
        references = unique(Symbol[ref for group in groups
                                   for ref in group.references
                                   if ref isa Symbol])
        out[key] = (; margins, groups, references)
    end
    out
end

function _sb_emit_ranef_r2d2_tau!(stmts, data, bucket_name, n_terms,
                                   decomposition)
    tau = Any[nothing for _ in 1:n_terms]
    for (group_index, group) in enumerate(decomposition.groups)
        stem = Symbol(bucket_name, :_r2d2_, group_index)
        r2_name = Symbol(stem, :_R2)
        phi_name = Symbol(stem, :_phi)
        alpha_name = Symbol(stem, :_alpha)
        push!(stmts, :($r2_name ~ beta($(group.r2_a), $(group.r2_b))))
        data[alpha_name] = fill(group.alpha, length(group.indices))
        _sb_record_static!(data, alpha_name)
        push!(stmts, :($phi_name ~ dirichlet($alpha_name)))
        for (local_index, margin_index) in enumerate(group.indices)
            ref = group.references[local_index]
            tau[margin_index] = :($ref * sqrt(
                ($phi_name[$local_index] * $r2_name) / (1. - $r2_name)))
        end
    end
    # A partial per-margin ICC leaves every unaddressed margin on the ordinary
    # half-standard-Normal scale prior.  They are scalars here so the final
    # `tau` can be one fully specified transformed vector without nuisance
    # entries for derived margins.
    for margin_index in eachindex(tau)
        isnothing(tau[margin_index]) || continue
        free_name = Symbol(bucket_name, :_r2d2_free_tau_, margin_index)
        push!(stmts, :($free_name ~ std_normal(; lower=0.)))
        tau[margin_index] = free_name
    end
    Expr(:vect, tau...)
end

# ---- R2D2 whole-predictor variance decomposition ----------------------------
#
# `effect(lp, :) ~ r2d2(...)` puts ONE joint prior on a predictor's population
# columns and its random-effect margins, by splitting a total scale `tau_bsv`
# into an explained part (allocated across columns by a Dirichlet simplex) and
# a residual part (the random effect). Design record: decisions `kx8wkd`
# (nested R2-partition family), `x0ea1e` (`effect(...)` surface), `1bbq22v`
# (scope), `1db6zkr` (all-or-nothing per shared bucket).
#
# The INTERCEPT is deliberately never decomposed: it is a location, not a
# source of explained variance, and its design column is constant so `Var(x_k)`
# would be zero. It keeps the ordinary standard-Normal prior unless an explicit
# `effect(lp, Intercept) ~ Normal(...)` statement overrides it -- which composes,
# because a column carrying its own override is likewise excluded from the
# simplex rather than fought over.

_sb_r2d2_kwarg(::Nothing, _default) = _default
_sb_r2d2_kwarg(x, _default) = x

function _sb_r2d2_beta(kw, lp)
    isnothing(kw) && return (1.0, 1.0)
    e = _as_expr_column(kw)
    isnothing(e) && error(
        "sbimpl: `r2d2(R2 = ...)` for `$lp` expects a `Beta(a, b)` prior on the " *
        "explained fraction, got $(repr(kw))")
    T = _as_distribution_type(getf(e))
    (!isnothing(T) && T <: Beta) || error(
        "sbimpl: `r2d2(R2 = ...)` for `$lp` currently supports only `Beta(a, b)`; " *
        "got `$(getf(e))`")
    isempty(getkwargs(e)) || error(
        "sbimpl: `r2d2(R2 = Beta(...))` does not accept distribution keywords")
    args = map(_sb_effect_prior_arg, getargs(e))
    length(args) == 2 || error(
        "sbimpl: `r2d2(R2 = Beta(a, b))` requires both Beta shape parameters")
    all(a -> a isa Real && isfinite(a) && a > 0, args) || error(
        "sbimpl: `r2d2(R2 = Beta(a, b))` shape parameters must be finite, " *
        "strictly positive numeric formula constants, got $(repr(args))")
    (Float64(args[1]), Float64(args[2]))
end

function _sb_r2d2_positive(x, what, lp)
    x isa Real || error(
        "sbimpl: `r2d2($what = ...)` for `$lp` must be a numeric formula " *
        "constant, got $(repr(x))")
    isfinite(x) && x > 0 || error(
        "sbimpl: `r2d2($what = ...)` for `$lp` must be finite and strictly " *
        "positive, got $x")
    Float64(x)
end

# Resolve every `effect(..., :) ~ r2d2(...)` statement onto its linear
# predictor, decide which population columns take a Dirichlet share, and check
# the shared-bucket all-or-nothing rule. Returns a Dict keyed by predictor; an
# empty Dict when the formula contains no r2d2 statement, which keeps every
# other model's emission byte for byte unchanged.
function _sb_r2d2_overrides(brmi::BRMI, id_buckets, effect_overrides)
    specs = r2d2_priors(brmi)
    isempty(specs) && return Dict{Symbol,NamedTuple}()

    lp_names = Symbol[x.name for x in linear_predictors(brmi)]
    labels_of(lp) = _sb_is_prior_declaration(brmi, lp) ? nothing :
        try popcoefnames(brmi, lp) catch; nothing end

    out = Dict{Symbol,NamedTuple}()
    for spec in specs
        target = spec.predictor
        if isnothing(target)
            candidates = Symbol[lp for lp in lp_names
                                if !isnothing(labels_of(lp))]
            isempty(candidates) && error(
                "sbimpl: `effect(:, :) ~ r2d2(...)` matches no linear predictor " *
                "with population coefficients")
            length(candidates) == 1 || error(
                "sbimpl: `effect(:, :) ~ r2d2(...)` is ambiguous across linear " *
                "predictors $(join(candidates, ", ")); use " *
                "`effect(<linear_predictor>, :) ~ r2d2(...)`.")
            target = only(candidates)
        end
        haskey(out, target) && error(
            "sbimpl: duplicate `r2d2` statement for linear predictor `$target`")

        labels = labels_of(target)
        isnothing(labels) && error(
            "sbimpl: `effect($target, :) ~ r2d2(...)` names no linear predictor " *
            "with population coefficients. Available predictors: " *
            "$(join([lp for lp in lp_names if !isnothing(labels_of(lp))], ", ")).")

        getf(spec.expression) === r2d2 || error(
            "sbimpl: a `Colon` effect address currently supports only the " *
            "`r2d2(...)` family; got `$(spec.family)`")
        isempty(spec.arguments) || error(
            "sbimpl: `r2d2(...)` takes keyword arguments only " *
            "(`R2`, `tau_bsv`, `alpha`); got $(length(spec.arguments)) positional")
        kw = spec.keywords
        known = (:R2, :tau_bsv, :alpha)
        for k in keys(kw)
            k in known || error(
                "sbimpl: unknown `r2d2` keyword `$k` for `$target`; supported " *
                "keywords are $(join(known, ", "))")
        end
        r2_a, r2_b = _sb_r2d2_beta(get(kw, :R2, nothing), target)
        alpha = _sb_r2d2_positive(get(kw, :alpha, 1.0), "alpha", target)
        raw_tau = get(kw, :tau_bsv, nothing)
        tau_bsv = isnothing(raw_tau) ? nothing :
                  _sb_r2d2_positive(raw_tau, "tau_bsv", target)

        # Which columns enter the simplex. The intercept never does; neither
        # does a column that already carries its own `effect(lp, coef) ~
        # Normal(...)` statement -- that override wins and the column keeps its
        # own scale, rather than being double-prioried.
        col_overrides = _sb_pop_effect_overrides(effect_overrides, target)
        share_idx = zeros(Int, length(labels))
        n_shares = 0
        for (i, label) in pairs(labels)
            label === :Intercept && continue
            isnothing(col_overrides) || isnothing(col_overrides[i]) || continue
            n_shares += 1
            share_idx[i] = n_shares
        end
        out[target] = (; labels, share_idx, n_shares, alpha, r2_a, r2_b, tau_bsv)
    end

    _sb_r2d2_check_buckets(id_buckets, out)
    out
end

# Decision `1db6zkr`, all-or-nothing per bucket: within one shared brms `|ID|`
# block a margin's `tau` is either DERIVED for every margin or SAMPLED for every
# margin. A part-derived vector would mean one submodel whose scale is half
# transformed parameter and half free parameter, which is not built -- and a
# partly-decomposed correlated block is statistically odd anyway.
function _sb_r2d2_check_buckets(id_buckets, r2d2_overrides)
    isempty(r2d2_overrides) && return
    for (key, bucket) in id_buckets
        margins = _sb_id_bucket_margins(bucket)
        scoped = unique(Symbol[m.predictor for m in margins
                               if haskey(r2d2_overrides, m.predictor)])
        isempty(scoped) && continue
        missing_lps = unique(Symbol[m.predictor for m in margins
                                    if !haskey(r2d2_overrides, m.predictor)])
        isempty(missing_lps) || error(
            "sbimpl: `r2d2` scopes $(join(sort(scoped), ", ")) in the shared " *
            "`|$(first(key))|` random-effect block, but " *
            "$(join(sort(missing_lps), ", ")) also slice that block without an " *
            "`r2d2` statement. Within one shared bucket the decomposition is " *
            "all-or-nothing: give every linear predictor in the bucket its own " *
            "`effect(<lp>, :) ~ r2d2(...)`, or none of them.")
    end
end

# Emit the shared per-predictor R2D2 parameters. This runs BEFORE the `|ID|`
# bucket prepass because a bucket's derived `tau` references `R2` / `tau_bsv`.
# Returns a name table keyed by predictor; `r2_name === nothing` marks the
# degenerate no-covariate case, where there is nothing to allocate and the
# random effect simply keeps the free total scale (decision `1db6zkr`).
function _sb_emit_r2d2_params!(stmts, data, r2d2_overrides)
    names = Dict{Symbol,NamedTuple}()
    for target in sort!(collect(keys(r2d2_overrides)))
        spec = r2d2_overrides[target]
        r2_name  = Symbol(:r2d2_, target, :_R2)
        tau_name = Symbol(:r2d2_, target, :_tau_bsv)
        phi_name = Symbol(:r2d2_, target, :_phi)
        if isnothing(spec.tau_bsv)
            # No user anchor. A latent per-subject predictor has no observed
            # response to derive a total scale from, so the honest default is a
            # sampled half-standard-normal rather than a fabricated constant.
            push!(stmts, :($tau_name ~ std_normal(; lower=0.)))
        else
            data[tau_name] = spec.tau_bsv
            _sb_record_static!(data, tau_name)
        end
        if spec.n_shares == 0
            names[target] = (; r2_name=nothing, phi_name=nothing, tau_name)
            continue
        end
        push!(stmts, :($r2_name ~ beta($(spec.r2_a), $(spec.r2_b))))
        # No `n_shares == 1` special-case: a one-element simplex emits
        # `simplex[1] $phi_name; $phi_name ~ dirichlet($alpha_name)` uniformly.
        # It is deterministically [1.0] with zero sampler dimensions, so it costs
        # nothing and keeps ONE Sb emission (no count-conditional branch). The
        # `n_shares == 0` case above is a genuine no-op — zero non-intercept
        # columns means no variance to decompose — not a degenerate-shape branch.
        alpha_name = Symbol(:r2d2_, target, :_alpha)
        data[alpha_name] = fill(spec.alpha, spec.n_shares)
        _sb_record_static!(data, alpha_name)
        push!(stmts, :($phi_name ~ dirichlet($alpha_name)))
        names[target] = (; r2_name, phi_name, tau_name)
    end
    names
end

# The residual scale a predictor's random-effect margins take:
# `sqrt((1 - R2) * tau_bsv^2)`, or the bare total scale when the predictor has
# no covariates to explain anything (R2 unused).
_sb_r2d2_resid_scale(nm) = isnothing(nm.r2_name) ? nm.tau_name :
    :(sqrt((1. - $(nm.r2_name)) * $(nm.tau_name)^2))

# The neutral bundle threaded through every emission path. Both Dicts empty
# means "no r2d2 statement in this formula", which every call site tests with
# `haskey(r2d2.overrides, target)` before changing anything it emits.
_sb_empty_r2d2() = (; overrides=Dict{Symbol,NamedTuple}(),
                      names=Dict{Symbol,NamedTuple}())

# Population half of an R2D2-scoped predictor. Every column keeps its position
# in the same `beta_pop` vector the ordinary path emits -- only the SCALE
# changes, and only for columns the simplex covers. Columns it does not cover
# (the intercept; anything with its own `effect(lp, coef) ~ Normal(...)`) keep
# their loc/scale in `beta_loc` / the `fallback` vector, so the two prior
# surfaces compose in one emission instead of fighting over `beta_pop`.
function _sb_emit_r2d2_popefs!(stmts, data, target, X_name, pop_name,
                                n_cols, spec, names, overrides)
    n_cols == length(spec.labels) || error(
        "sbimpl: internal r2d2 alignment error for `$target`: " *
        "$(length(spec.labels)) population labels for $n_cols design columns")
    beta_loc = Float64[0.0 for _ in spec.labels]
    fallback = Float64[1.0 for _ in spec.labels]
    if !isnothing(overrides)
        for i in eachindex(overrides)
            isnothing(overrides[i]) && continue
            loc, scale = _sb_effect_normal_args(overrides[i])
            (loc isa Real && scale isa Real) || error(
                "sbimpl: `effect($target, $(spec.labels[i])) ~ Normal(...)` " *
                "combined with `effect($target, :) ~ r2d2(...)` requires " *
                "numeric location and scale constants, got " *
                "$(repr(loc)), $(repr(scale))")
            beta_loc[i] = Float64(loc)
            fallback[i] = Float64(scale)
        end
    end
    share_name = Symbol(:r2d2_, target, :_share_idx)
    fall_name  = Symbol(:r2d2_, target, :_fallback)
    loc_name   = Symbol(:r2d2_, target, :_beta_loc)
    varx_name  = Symbol(:r2d2_, target, :_varx)
    scale_name = Symbol(:r2d2_, target, :_beta_scale)
    data[share_name] = spec.share_idx
    data[fall_name]  = fallback
    data[loc_name]   = beta_loc
    _sb_record_static!(data, share_name)
    _sb_record_static!(data, fall_name)
    _sb_record_static!(data, loc_name)
    push!(stmts, :($varx_name = brm_col_variances(
        $X_name, dims($X_name)[1], dims($X_name)[2])))
    push!(stmts, :($scale_name = brm_r2d2_scale(
        $share_name, $fall_name, $varx_name, $(names.phi_name),
        $(names.r2_name), $(names.tau_name),
        dims($X_name)[2], $(spec.n_shares))))
    push!(stmts, :($pop_name ~ _popefs_normal(;
        X=$X_name, beta_loc=$loc_name, beta_scale=$scale_name)))
end

# ---- group-block prepass (Prepass 2.5) ---------------------------------------
#
# Scan brmi.operations for declaring terms anywhere in the model: a term `f`
# with a `_sb_term_group_block` declaration, appearing EITHER as a whole-RHS
# parameter submodel (`mu ~ f(...)`, like the biomarker family) OR as a predictor summand
# (`y ~ 1 + hsgp(t, by=g)`). We walk every `~` op's RHS summands (`_sb_terms`),
# which covers both: the biomarker family is a single summand of its RHS, hsgp is one of several.
# Every declaring summand is its own term INSTANCE — there is no `(key, f)`
# dedup, so N instances of the same term (e.g. `hsgp(t, by=g) + hsgp(s, by=g)`)
# each get collected and allocated their own block. Identical instances (same
# block name) are coalesced later, at emit time. A subsequent emit step
# allocates one block per declared field and builds the lookup.
function _sb_collect_group_block_terms(brmi::BRMI)
    result = Any[]
    for (key, op_nc) in pairs(brmi.operations)
        op = _as_expr_column(parent(op_nc)); isnothing(op) && continue
        getf(op) === (~) || continue
        _, rhs = getargs(op, 2)
        for t in _sb_terms(rhs)
            te = _as_expr_column(t); isnothing(te) && continue
            f = getf(te)
            fields = _sb_structured_fields(_sb_term_group_block(f, te), f)
            isnothing(fields) && continue
            push!(result, (; key, f, rhs_e=te, fields))
        end
    end
    result
end

# Resolve a field's grouping column from its group spec + the term call.
function _sb_resolve_group_col(gspec, rhs_e, data)
    haskey(gspec, :fn) && return gspec.fn(rhs_e, data)
    if haskey(gspec, :kwarg)
        kw = getkwargs(rhs_e)
        haskey(kw, gspec.kwarg) || error(
            "sbimpl: structured-latent group kwarg `$(gspec.kwarg)=` missing in `$(nameof(getf(rhs_e)))` call")
        nc = _as_named_column(kw[gspec.kwarg])
        isnothing(nc) && error(
            "sbimpl: `$(gspec.kwarg)=` must be a NamedColumn group, got $(typeof(kw[gspec.kwarg]))")
        return nc
    end
    pos = gspec.arg_pos
    args = getargs(rhs_e)
    pos <= length(args) || error(
        "sbimpl: structured-latent group_arg_pos=$pos but `$(nameof(getf(rhs_e)))` has $(length(args)) args")
    nc = _as_named_column(args[pos])
    isnothing(nc) && error(
        "sbimpl: structured-latent group arg $pos must be a NamedColumn, got $(typeof(args[pos]))")
    nc
end

# The grouping column's NAME, without materialising data. Used to build the
# disambiguating block name (`b_<field>_<gname>`) in BOTH the emit pass and the
# find pass so they agree on the per-instance block. For a `group_fn` spec this
# is `fn_name` by construction (the synthesised column is named `fn_name`), so
# `data` is never needed here — only `_sb_resolve_group_col` (emit) needs it.
_sb_group_name(gspec, rhs_e) =
    haskey(gspec, :fn)    ? gspec.fn_name :
    haskey(gspec, :kwarg) ? name(_as_named_column(getkwargs(rhs_e)[gspec.kwarg])) :
                            name(_as_named_column(getargs(rhs_e)[gspec.arg_pos]))

# Emit the per-group block draw for one field, dispatching on its prior spec.
# Each block is an n_groups × n_per_group matrix referenced by row = group.
_sb_emit_block_draw!(stmts, prior::Symbol, block_name, idx_name, n_name, n_terms_name, suffix) = begin
    if prior === :correlated_normal
        push!(stmts, :($block_name ~ ranef_correlated_draws(;
            group_idx=$idx_name, n_groups=$n_name, n_terms=$n_terms_name)))
    elseif prior === :iid_normal
        flat_name = Symbol(:zflat_, suffix)
        push!(stmts, :($flat_name ~ std_normal(; n=$n_name * $n_terms_name)))
        push!(stmts, :($block_name = reshape($flat_name, $n_terms_name, $n_name)'))
    else
        error("sbimpl: unknown structured-latent prior symbol `:$prior` ",
              "(expected :correlated_normal or :iid_normal)")
    end
end
# Element-wise prior over the matrix, reusing the scalar-prior dist-name table.
function _sb_emit_block_draw!(stmts, prior::NamedTuple, block_name, idx_name, n_name, n_terms_name, suffix)
    stan_name = _sb_stan_dist_name(prior.dist)
    isnothing(stan_name) && error(
        "sbimpl: structured-latent prior dist `$(prior.dist)` has no `_sb_stan_dist_name` mapping")
    pos_args = _sb_stan_dist_args(
        prior.dist, map(_sb_prior_arg, get(prior, :args, ())))
    flat_name = Symbol(:zflat_, suffix)
    kw = Any[Expr(:kw, :n, :($n_name * $n_terms_name))]
    haskey(prior, :lower) && push!(kw, Expr(:kw, :lower, prior.lower))
    haskey(prior, :upper) && push!(kw, Expr(:kw, :upper, prior.upper))
    rhs_call = Expr(:call, stan_name, pos_args..., Expr(:parameters, kw...))
    push!(stmts, Expr(:call, :~, flat_name, rhs_call))
    push!(stmts, :($block_name = reshape($flat_name, $n_terms_name, $n_name)'))
end

# Allocate one block per declared field and return a lookup
# Dict{block_name::Symbol => (; block_name, idx_name, n_per_group)}.
# Keyed by the fully-disambiguated block NAME (`b_<field>_<gname>`), so N
# distinct instances (different x and/or group) each get their own block and
# only genuinely identical instances (same block name) are coalesced.
function _sb_emit_group_blocks!(stmts, data, gb_terms)
    lookup = Dict{Symbol, NamedTuple}()
    for (; f, rhs_e, fields) in gb_terms
        for fld in fields
            group_col = _sb_resolve_group_col(fld.group, rhs_e, data)
            gname = name(group_col)
            # Block name `b_<field>_<gname>`; for legacy single-field terms
            # field == nameof(f), so this is byte-for-byte the old name. For hsgp
            # the field name embeds x, so two HSGPs on the same group never collide.
            suffix = Symbol(fld.name, :_, gname)
            block_name = Symbol(:b_, suffix)
            haskey(lookup, block_name) && continue   # identical instance already allocated
            n_terms_name = Symbol(:n_terms_, suffix)
            data[n_terms_name] = fld.n_per_group
            _sb_record_static!(data, n_terms_name)
            idx_name, n_name = _sb_ensure_group_data!(data, group_col)
            _sb_emit_block_draw!(stmts, fld.prior, block_name, idx_name, n_name, n_terms_name, suffix)
            lookup[block_name] = (; block_name, idx_name, n_per_group=fld.n_per_group)
        end
    end
    lookup
end

# Look up block_info for a declaring term call at emit time, or nothing. Rebuilds
# each field's block name (via `_sb_group_name`, matching the emit pass) and
# resolves it against the block-name-keyed lookup. Returns a NamedTuple carrying
# a `fields` map (field-name => per-field info); for single-field terms the lone
# field's keys are also spliced at top level so legacy consumers that destructure
# `(; block_name, idx_name)` keep working.
function _sb_find_group_block(f, rhs_e, group_block_lookup)
    fields = _sb_structured_fields(_sb_term_group_block(f, rhs_e), f)
    isnothing(fields) && return nothing
    fmap = Dict{Symbol,NamedTuple}()
    for fld in fields
        block_name = Symbol(:b_, fld.name, :_, _sb_group_name(fld.group, rhs_e))
        info = get(group_block_lookup, block_name, nothing)
        isnothing(info) && return nothing
        fmap[fld.name] = info
    end
    isempty(fmap) && return nothing
    first_info = fmap[first(fields).name]
    (; first_info..., fields=fmap)
end

_sb_group_desc_matches(a::NamedColumn, b::NamedColumn) = name(a) === name(b)
_sb_group_desc_matches(a::Tuple{NamedColumn,NamedColumn}, b::Tuple{NamedColumn,NamedColumn}) =
    name(a[1]) === name(b[1]) && name(a[2]) === name(b[2])
_sb_group_desc_matches(_, _) = false

# Column count for a ranef term (without emitting). `1` -> 1 (intercept);
# scalar NamedColumn -> 1; categorical NamedColumn -> (n_levels-1); ExprColumn
# submodel terms (mo/s/ar/me) -> 1.
_sb_ranef_term_ncols(t::Int, _) = t == 0 ? 0 : 1
_sb_ranef_term_ncols(t::NamedColumn, _data) = _sb_ranef_named_ncols(_sb_cat_levels(t))
_sb_ranef_named_ncols(::Nothing) = 1
_sb_ranef_named_ncols(levels) = _sb_level_index(levels)[1] - 1
_sb_ranef_term_ncols(::ExprColumn, _) = 1
_sb_ranef_term_ncols(t, _) = error("sbimpl: unsupported ranef term $(typeof(t)): $t")

# Emit one shared `b_<id>_<g> ~ ranef_correlated_draws(...)` per bucket, compute
# per-target column ranges, stash `group_idx` / `n_groups` / `n_terms_<id>_<g>`
# in `data`, and return the lookup table consumed by `_sb_emit_ranefs!`.
#
# `cv_groups` / `centered_groups` select the `_cv` / `_centered` draws variant
# for a bucket whose grouping factor is opted in. Both are known at `SBBRMI`
# construction, i.e. BEFORE this prepass runs, so there is no ordering obstacle
# to threading them here -- only the plain-group spelling has a variant, and the
# stratified `gr(g, by=b)` bucket still errors (see `_sb_emit_id_bucket_sampling!`).
function _sb_emit_id_buckets!(stmts, data, buckets;
                              cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(),
                              ranef_effect_overrides=Dict{Tuple{Symbol,Any},NamedTuple}(),
                              r2d2_names=Dict{Symbol,NamedTuple}(),
                              ranef_r2d2_overrides=Dict{Tuple{Symbol,Any},NamedTuple}())
    lookup = _sb_empty_id_lookup()
    for (k, bucket) in pairs(buckets)
        id_sym, _ = k
        desc = bucket.group_desc
        suffix = _sb_id_bucket_suffix(id_sym, desc)
        bucket_name = Symbol(:b_, suffix)
        n_terms_name = Symbol(:n_terms_, suffix)
        cursor = 0
        per_target_ranges = Pair{Symbol,UnitRange{Int}}[]
        for (brmi_key, terms) in bucket.per_target
            ncols = sum(_sb_ranef_term_ncols(t, data) for t in terms; init=0)
            ncols > 0 || error("sbimpl: `|$id_sym|` bucket sees empty term list for target `$brmi_key`")
            push!(per_target_ranges, brmi_key => (cursor+1):(cursor+ncols))
            cursor += ncols
        end
        n_terms_total = cursor
        n_terms_total >= 1 || error("sbimpl: `|$id_sym|` bucket has zero terms")
        data[n_terms_name] = n_terms_total
        _sb_record_static!(data, n_terms_name)
        ranef_effect = get(ranef_effect_overrides, k, nothing)
        # Derived-`tau` vector for an R2D2-scoped bucket. `_sb_r2d2_check_buckets`
        # has already guaranteed all-or-nothing, so either every margin resolves
        # here or none does.
        margins = _sb_id_bucket_margins(bucket)
        r2d2_tau = nothing
        bucket_r2d2 = get(ranef_r2d2_overrides, k, nothing)
        if !isnothing(bucket_r2d2)
            !isempty(r2d2_names) &&
                all(m -> haskey(r2d2_names, m.predictor), margins) && error(
                    "sbimpl: `|$id_sym|` carries both whole-predictor " *
                    "`effect(..., :) ~ r2d2(...)` and random-effect " *
                    "`sd(...) ~ r2d2(...)` decompositions; choose one " *
                    "variance allocation for the block")
            !isnothing(ranef_effect) && ranef_effect.has_sd && error(
                "sbimpl: `|$id_sym|` carries both an R2D2-derived scale and a " *
                "direct sampled SD prior; choose one scale prior")
            r2d2_tau = _sb_emit_ranef_r2d2_tau!(
                stmts, data, bucket_name, n_terms_total, bucket_r2d2)
        elseif !isempty(r2d2_names) &&
           all(m -> haskey(r2d2_names, m.predictor), margins)
            (!isnothing(ranef_effect) && ranef_effect.has_sd) && error(
                "sbimpl: `|$id_sym|` carries both an `r2d2` decomposition and an " *
                "`sd(...)` statement on `$id_sym`. The decomposition " *
                "DERIVES the block's marginal scales, so a sampled SD prior on " *
                "the same block has nothing to apply to; drop one of the two.")
            # One margin per predictor, for the same reason the plain path
            # insists on `(1 | g)`: each predictor contributes exactly one
            # derived residual scale `sqrt((1 - R2) * tau_bsv^2)`, and handing
            # that same scalar to two margins of one predictor would double its
            # random-effect variance instead of partitioning it.
            for m in margins
                count(x -> x.predictor === m.predictor, margins) == 1 || error(
                    "sbimpl: `|$id_sym|` gives predictor `$(m.predictor)` " *
                    "$(count(x -> x.predictor === m.predictor, margins)) " *
                    "random-effect margins, but its `r2d2` decomposition " *
                    "derives a single residual scale. Splitting " *
                    "`(1 - R2) * tau_bsv^2` among several margins needs a " *
                    "second simplex that the flat decomposition does not build.")
            end
            r2d2_tau = Expr(:vect,
                [_sb_r2d2_resid_scale(r2d2_names[m.predictor]) for m in margins]...)
        end
        idx_name = _sb_emit_id_bucket_sampling!(stmts, data, bucket_name, n_terms_name, desc;
                                                cv_groups, centered_groups, id_sym,
                                                ranef_effect, r2d2_tau,
                                                r2d2_lkj_eta=(isnothing(ranef_effect) ?
                                                    1.0 : ranef_effect.lkj_eta))
        for (brmi_key, cols) in per_target_ranges
            lookup[(brmi_key, k)] = (; bucket_name, cols, idx_name, suffix)
        end
    end
    lookup
end

_sb_id_bucket_suffix(id_sym, g::NamedColumn) = Symbol(id_sym, :_, name(g))
_sb_id_bucket_suffix(id_sym, g::Tuple{NamedColumn,NamedColumn}) =
    Symbol(id_sym, :_, name(g[1]), :__by__, name(g[2]))

# Emit the shared `b_<suffix> ~ …_draws(...)` statement for one ID bucket.
# Plain group -> `ranef_correlated_draws` (or its `_cv` / `_centered` variant
# when the group is opted in); `gr(g, by=b)` group -> stratified
# `ranef_correlated_by_draws`, which has neither variant. Returns the idx_name
# callers use to slice the draw matrix per sub-formula.
function _sb_emit_id_bucket_sampling!(stmts, data, bucket_name, n_terms_name, g::NamedColumn;
                                      cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(),
                                      id_sym=nothing, ranef_effect=nothing, r2d2_tau=nothing,
                                      r2d2_lkj_eta=1.0)
    idx_name, n_name = _sb_ensure_group_data!(data, g)
    gname = name(g)
    if !isnothing(r2d2_tau)
        # R2D2 bucket: the marginal scales are a transformed parameter, so the
        # block goes to the derived-`tau` sibling. Centered and cv variants are
        # deliberately not wired -- a derived scale interacts with both, and
        # neither has been designed, so they fail loudly rather than silently
        # sampling something else.
        gname in centered_groups && error(
            "sbimpl: group `$gname` is in `centered_groups` and also carries an " *
            "`r2d2` decomposition; the centered path for derived marginal " *
            "scales is not implemented")
        n_groups = n_name
        if gname in cv_groups
            # Match the ordinary non-centred CV path: taint the block through
            # its size expression so only `z_flat` moves to generated
            # quantities. R², phi, the reference scales and L stay fitted and
            # remain name-transportable.
            n_cv_name = Symbol(bucket_name, :_n_g)
            push!(stmts, :($n_cv_name = maximum($idx_name)))
            n_groups = n_cv_name
        end
        tau_name = Symbol(bucket_name, :_r2d2_tau)
        push!(stmts, :($tau_name = $r2d2_tau))
        push!(stmts, :($bucket_name ~ ranef_correlated_draws_r2d2(;
            group_idx=$idx_name, n_groups=$n_groups, n_terms=$n_terms_name,
            tau=$tau_name, lkj_eta=$r2d2_lkj_eta)))
        return idx_name
    end
    sd_family = isnothing(ranef_effect) ? nothing : Expr(:vect, ranef_effect.sd_family...)
    sd_rate = isnothing(ranef_effect) ? nothing : Expr(:vect, ranef_effect.sd_rate...)
    lkj_eta = isnothing(ranef_effect) ? nothing : ranef_effect.lkj_eta
    if gname in cv_groups
        # Same submodel as the default branch; only the SIZE EXPRESSION differs.
        # Tracing `maximum(<g>_idx)` at the CALL SITE carries the cv taint on
        # `<g>_idx` into the submodel's declared size, so a `maybecv(:<g>_idx)`
        # mark flips the whole block to a generated-quantities re-draw. Value and
        # column-major layout are unchanged. Bound to a named local first so the
        # size appears once as `int <b>_n_g = max(<g>_idx);` instead of being
        # inlined into every declaration.
        n_cv_name = Symbol(bucket_name, :_n_g)
        push!(stmts, :($n_cv_name = maximum($idx_name)))
        if isnothing(ranef_effect)
            push!(stmts, :($bucket_name ~ ranef_correlated_draws(;
                group_idx=$idx_name, n_groups=$n_cv_name, n_terms=$n_terms_name)))
        else
            push!(stmts, :($bucket_name ~ ranef_correlated_draws_effect(;
                group_idx=$idx_name, n_groups=$n_cv_name, n_terms=$n_terms_name,
                sd_family=$sd_family, sd_rate=$sd_rate, lkj_eta=$lkj_eta)))
        end
    elseif gname in centered_groups
        if isnothing(ranef_effect)
            push!(stmts, :($bucket_name ~ ranef_correlated_draws_centered(;
                group_idx=$idx_name, n_groups=$n_name, n_terms=$n_terms_name)))
        else
            push!(stmts, :($bucket_name ~ ranef_correlated_draws_centered_effect(;
                group_idx=$idx_name, n_groups=$n_name, n_terms=$n_terms_name,
                sd_family=$sd_family, sd_rate=$sd_rate, lkj_eta=$lkj_eta)))
        end
    else
        if isnothing(ranef_effect)
            push!(stmts, :($bucket_name ~ ranef_correlated_draws(;
                group_idx=$idx_name, n_groups=$n_name, n_terms=$n_terms_name)))
        else
            push!(stmts, :($bucket_name ~ ranef_correlated_draws_effect(;
                group_idx=$idx_name, n_groups=$n_name, n_terms=$n_terms_name,
                sd_family=$sd_family, sd_rate=$sd_rate, lkj_eta=$lkj_eta)))
        end
    end
    idx_name
end
function _sb_emit_id_bucket_sampling!(stmts, data, bucket_name, n_terms_name,
                                       g::Tuple{NamedColumn,NamedColumn};
                                       cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(),
                                       id_sym=nothing, ranef_effect=nothing, r2d2_tau=nothing,
                                       r2d2_lkj_eta=1.0)
    gname, bname = name(g[1]), name(g[2])
    id_str = isnothing(id_sym) ? "ID" : String(id_sym)
    isnothing(r2d2_tau) || error(
        "sbimpl: `r2d2` decompositions for stratified `|$id_str| " *
        "gr($gname, by=$bname)` buckets are not yet supported")
    isnothing(ranef_effect) || error(
        "sbimpl: covariance-prior effects for stratified `|$id_str| " *
        "gr($gname, by=$bname)` buckets are not yet supported")
    gname in cv_groups && error(
        "sbimpl: cv-contagious sizing requested for group `$gname`, but it ",
        "appears in a `(… |$id_str| gr($gname, by=$bname))` stratified ID bucket, ",
        "whose shared draws block goes through the typed-LHS ",
        "`ranef_correlated_by_draws` path. StanBlocks' typed-LHS forward derives ",
        "cv from RHS call-args, not the declared size, so a cv size there yields ",
        "a cv *parameter*, not a generated-quantities re-draw. Use a plain ",
        "`(… |$id_str| $gname)` bucket for the cv group -- `by=` is not yet supported.")
    gname in centered_groups && error(
        "sbimpl: centered parameterization requested for group `$gname`, but it ",
        "appears in a `(… |$id_str| gr($gname, by=$bname))` stratified ID bucket ",
        "(one Cholesky per stratum). The centered variants emit a plate over a ",
        "single shared covariance and have no per-stratum form -- not yet ",
        "supported. Use a plain `(… |$id_str| $gname)` bucket, or leave ",
        "`$gname` non-centered.")
    info = _sb_ensure_group_data!(data, g)
    push!(stmts, :($bucket_name ~ ranef_correlated_by_draws(;
        group_idx=$(info.idx_name), n_groups=$(info.n_name),
        n_terms=$n_terms_name,
        stratum_idx=$(info.s_idx_name), n_strata=$(info.n_strata_name))))
    info.idx_name
end

# Reclaim a raw grouping-column name for BRM's dense integer index. The generic
# data prepass sees grouping columns too, but Stan consumes only their dense
# integer code, never the raw labels (which may be strings and therefore are not
# valid Stan data at all) — so the raw column is deleted from `data` here. BRM
# OWNS the raw name of any column used as a ranef grouping factor and reclaims it
# in this pre-pass; see the `_sb_submodel_rhs!` docstring for the reserved-name
# contract this enforces.
#
# Guard against a silent clobber: a `_sb_submodel_rhs!` hook (or any other data-
# writing extension) that stashes its OWN vector under a raw column name the same
# model also uses as a grouping factor — and emits a reference to it — would have
# that vector deleted out from under the emitted statement, surfacing only much
# later as an unresolvable-symbol error deep in StanBlocks tracing that names
# neither the delete nor BRM. When the key no longer holds the raw column the
# data prepass wrote (i.e. something replaced it), fail loudly and attributed
# HERE, pointing at the per-target keying that fixes it.
function _sb_reclaim_group_col!(data, colname::Symbol, backing::DataColumn)
    if haskey(data, colname) && !isequal(data[colname], parent(backing))
        error(
            "sbimpl: `$colname` is used as a ranef grouping factor, so BRM ",
            "reclaims its raw column for a dense integer index (`$(colname)_idx` ",
            "/ `n_$colname`) and deletes the raw labels here — but `data[:$colname]` ",
            "currently holds a value the generic data prepass did NOT write. A ",
            "`_sb_submodel_rhs!` hook (or other extension) stashed its own vector ",
            "under this reserved name; it would be deleted out from under any ",
            "statement referencing it and fail later in StanBlocks tracing with an ",
            "unresolvable-symbol error naming a third package. BRM owns raw column ",
            "names used as grouping factors — key consumer-written data PER TARGET ",
            "instead, e.g. `_sb_kernel_key(col, target) = Symbol(col, :_, target)`, ",
            "which is self-owned and order-independent.")
    end
    delete!(data, colname)
    nothing
end

# Ensure `group_idx` / `n_groups` for a plain-group ranef descriptor are stashed
# in `data`. Idempotent — safe to call from both the ID pre-pass and the per-
# target plain-block emitter. Returns the (idx_name, n_name) pair used in stmts.
function _sb_ensure_group_data!(data, g::NamedColumn)
    g_backing = _as_data_column(parent(g))
    isnothing(g_backing) && error("sbimpl: group `$(name(g))` must be a raw data column")
    gname = name(g)
    idx_name = Symbol(gname, :_idx)
    n_name   = Symbol(:n_, gname)
    n_levels, g_idx = _sb_level_index(parent(g_backing))
    _sb_reclaim_group_col!(data, gname, g_backing)
    data[idx_name] = g_idx
    data[n_name]   = n_levels
    _sb_record_group_index!(data, idx_name, n_name, gname, parent(g_backing))
    idx_name, n_name
end

# Stratified variant for `gr(g, by=b)`: stashes g_idx / n_groups as well as
# stratum_idx / n_strata under block-local names (suffixed with `__by__`) so
# this block never clashes with a plain `(... | g)` bucket on the same group.
function _sb_ensure_group_data!(data, g::Tuple{NamedColumn,NamedColumn})
    gcol, bcol = g
    g_backing = _as_data_column(parent(gcol))
    b_backing = _as_data_column(parent(bcol))
    isnothing(g_backing) && error("sbimpl: group `$(name(gcol))` must be a raw data column")
    isnothing(b_backing) && error("sbimpl: `by=$(name(bcol))` must be a raw data column")
    gname, bname = name(gcol), name(bcol)
    n_groups, g_idx = _sb_level_index(parent(g_backing))
    n_strata, b_idx = _sb_level_index(parent(b_backing))
    _sb_reclaim_group_col!(data, gname, g_backing)
    _sb_reclaim_group_col!(data, bname, b_backing)
    stratum_idx = _sb_stratum_idx(g_idx, b_idx, gname, bname)
    suffix = Symbol(gname, :__by__, bname)
    idx_name       = Symbol(suffix, :_idx)
    n_name         = Symbol(:n_, suffix)
    s_idx_name     = Symbol(suffix, :_stratum_idx)
    n_strata_name  = Symbol(:n_strata_, suffix)
    data[idx_name]      = g_idx
    data[n_name]        = n_groups
    data[s_idx_name]    = stratum_idx
    data[n_strata_name] = n_strata
    (; idx_name, n_name, s_idx_name, n_strata_name)
end

# Emit the per-sub-formula reference to a pre-emitted ID bucket: slice the
# bucket's draw matrix at this target's column range, apply this target's Z,
# and append the resulting per-row contribution to `summands`.
function _sb_emit_id_ranef_block!(stmts, data, target::Symbol, info, gterms, summands;
                                  term_overrides=Dict{Symbol,Any}())
    (; bucket_name, cols, idx_name, suffix) = info
    r_name = Symbol(:r_, target, :_, suffix)
    col_exprs = Any[]
    for t in gterms
        _sb_ranef_cols!(col_exprs, data, stmts, t, gterms;
                        group_idx=idx_name, term_overrides)
    end
    length(col_exprs) == length(cols) ||
        error("sbimpl: id-bucket `$suffix` for target `$target`: expanded $(length(col_exprs)) columns but reserved $(length(cols)) — internal mismatch")
    if length(cols) == 1
        col_idx = first(cols)
        if length(gterms) == 1 && gterms[1] === 1
            # Intercept fast path: Z is all-ones, skip the elementwise multiply.
            push!(stmts, :($r_name = $bucket_name[$idx_name, $col_idx]))
        else
            push!(stmts, :($r_name = $(col_exprs[1]) .* $bucket_name[$idx_name, $col_idx]))
        end
    else
        Z_name       = Symbol(:Z_, target, :_, suffix)
        col_idx_name = Symbol(:col_idx_, target, :_, suffix)
        data[col_idx_name] = collect(Int, cols)
        _sb_record_static!(data, col_idx_name)
        push!(stmts, :($Z_name = $(Expr(:call, :hcat, col_exprs...))))
        push!(stmts, :($r_name = rows_dot_product($Z_name, $bucket_name[$idx_name, $col_idx_name])))
    end
    push!(summands, r_name)
end

# Extract pop terms from `1 + a + c1 [+ (...|g)]`. `0` is the standard
# formula-language drop-intercept marker (e.g. `loc ~ 0 + ftime`) and
# contributes no predictor column, so it's filtered out here.
_sb_terms(x) = (acc = Any[]; _sb_collect_terms!(acc, x); acc)
_sb_collect_terms!(acc, x::ExprColumn) = _sb_collect_terms_expr!(acc, getf(x), x)
_sb_collect_terms!(acc, x::Int) = x == 0 ? nothing : push!(acc, x)
_sb_collect_terms!(acc, x) = push!(acc, x)
_sb_collect_terms_expr!(acc, ::typeof(+), x) = foreach(a -> _sb_collect_terms!(acc, a), getargs(x))
# `(expr | group)` is kept as-is; `_sb_linear_predictor!` splits it off into
# the ranef side of the additive linear predictor.
_sb_collect_terms_expr!(acc, ::typeof(|), x) = push!(acc, x)
# `a * b` -- kept whole for the predictor-term layer. Under `~`, a `*` over raw
# data columns is a protect-style materialised term (popefs supplies its beta);
# a `*` that references a sampled coefficient is NOT a formula term at all and is
# rejected in `_sb_predictor_term!(::typeof(*))` with a pointer to the assignment
# (`=`) form. The old `scalar*data` LP escape hatch was dropped in 84434c7 so `~`
# stays formula-only and `coef * col` has exactly one spelling (`lp = coef * col`).
_sb_collect_terms_expr!(acc, ::typeof(*), x) = push!(acc, x)
# `factor(c, ref=k)`: configurable reference level for a categorical
# column. Re-encode at term-collection time (swap level k <-> level 1)
# and inject a synthetic NamedColumn with the recoded data, so the
# downstream categorical pipeline (treatment-coded dummies relative to
# level 1) reuses unchanged.
_sb_collect_terms_expr!(acc, ::typeof(factor), x::ExprColumn) = begin
    args = getargs(x); kw = getkwargs(x)
    length(args) == 1 || error("sbimpl: `factor(...)` expects 1 positional arg, got $(length(args))")
    inner_raw = only(args)
    inner = _as_named_column(inner_raw)
    isnothing(inner) && error("sbimpl: `factor(...)` expects a NamedColumn, got $(typeof(inner_raw))")
    backing = _as_data_column(parent(inner))
    isnothing(backing) && error("sbimpl: `factor($(name(inner)))` expects a raw data column")
    raw = _as_int_vec(parent(backing))
    isnothing(raw) && error("sbimpl: `factor($(name(inner)))` expects integer-coded categorical data, got $(typeof(parent(backing)))")
    ref_raw = get(kw, :ref, 1)
    ref = _as_integer(ref_raw)
    isnothing(ref) && error("sbimpl: `factor(...; ref=k)` expects an integer level, got $(typeof(ref_raw))")
    1 <= ref <= maximum(raw) || error("sbimpl: `factor($(name(inner)); ref=$ref)` ref out of range (max level $(maximum(raw)))")
    new_name = ref == 1 ? name(inner) : Symbol(name(inner), :__ref_, ref)
    new_backing = ref == 1 ? backing :
        DataColumn(Int[r == ref ? 1 : r == 1 ? ref : r for r in raw])
    push!(acc, NamedColumn(new_name, new_backing))
end
# `(t1 + t2 + ... || g)` zerocorr ranefs: independent variances per term,
# no shared correlation. Expand into N separate `(t_i | g_nocor_i)` ran
# terms with synthetic group names so the ran-term coalescer in
# `_sb_emit_ranefs!` keeps them as separate (degenerate K=1) blocks.
# Mirrors vimpl's `vmeta_sampling_rhs(::ExprColumn{typeof(doublepipe)})`.
#
# The intercept-suppressing `0` (`0 + x || g` == "uncorrelated slope x, no
# intercept") is the standard formula-language drop-intercept marker and
# contributes no term -- drop it BEFORE splitting into per-term nocor groups,
# exactly as `_sb_collect_terms!(::Int)` does on the correlated `|` path. Left
# in, `0` claimed its own `g__nocor__1` group that then had zero terms after
# `_sb_terms` dropped it in `_sb_emit_ranefs!`, so `(0 + x || g)` -- a single
# uncorrelated slope, and a documented feature-atlas example -- errored on "no
# terms" instead of emitting that lone slope as an independent scalar ranef.
_sb_collect_terms_expr!(acc, ::typeof(doublepipe), x) = begin
    args = getargs(x)
    length(args) == 2 || error("sbimpl: `||` zerocorr expects 2 args, got $(length(args))")
    lhs, rhs = args
    rhs_nc = _as_named_column(rhs)
    isnothing(rhs_nc) && error("sbimpl: `||` zerocorr RHS must be a NamedColumn group, got $(typeof(rhs))")
    inner = filter(!_sb_is_drop_intercept, _zerocorr_inner(lhs))
    isempty(inner) && error(
        "sbimpl: `(… || $(name(rhs_nc)))` has no terms after dropping `0`; an ",
        "uncorrelated block needs at least one slope or intercept term")
    for (i, term) in enumerate(inner)
        nocor = NamedColumn(Symbol(name(rhs_nc), :__nocor__, i), parent(rhs_nc))
        push!(acc, ExprColumn(|, term, nocor))
    end
end

# The `0` drop-intercept marker (`x::Int == 0`), matching `_sb_collect_terms!`.
_sb_is_drop_intercept(t) = t isa Int && t == 0
_zerocorr_inner(lhs::ExprColumn) = getf(lhs) === (+) ? collect(getargs(lhs)) : Any[lhs]
_zerocorr_inner(lhs) = Any[lhs]
# `a & b` is the interaction operator (parallels StatsModels.jl). `&` has
# higher precedence than `+` in Julia, so `1 + a + b + a&b` naturally parses
# as `+(1, a, b, a&b)` and the normal `+`-flatten path applies. We deliberately
# chose `&` over R's `:` because Julia parses `:` as lower-precedence than
# `+`, which forces a precedence-peel hack that breaks chained interactions.
_sb_collect_terms_expr!(acc, _, x) = push!(acc, x)

# Pop-term column accumulator. Most terms produce a single column via
# `_sb_predictor_col`; interactions (`a:b`) can produce multiple columns
# depending on operand types, so we push into a caller-owned vector.
# `pop_terms` is threaded so the intercept emitter can borrow N from a
# data-backed peer in the same formula (deterministic) rather than
# probing the shared `data` dict in hash order.
# A predictor term may legitimately contribute NO column: a `mo(c)` over a
# single-level factor has 0 increments, so its free-beta effect is identically 0
# and it vanishes rather than emitting a `simplex[0]` (which Stan rejects). Such
# terms return `nothing`; drop them instead of pushing a `nothing` into `cols`.
_sb_maybe_push_col!(cols, ::Nothing) = cols
_sb_maybe_push_col!(cols, c) = push!(cols, c)
_sb_pop_cols!(cols, t, data, stmts, pop_terms=(); obs_n=nothing, ran_terms=(), direct_terms=(), target=nothing, group_block_lookup=Dict(), term_overrides=Dict{Symbol,Any}()) =
    _sb_maybe_push_col!(cols, _sb_predictor_col(t, data, stmts, pop_terms; obs_n, ran_terms, direct_terms, target, group_block_lookup, term_overrides))
_sb_pop_cols!(cols, t::ExprColumn, data, stmts, pop_terms=(); obs_n=nothing, ran_terms=(), direct_terms=(), target=nothing, group_block_lookup=Dict(), term_overrides=Dict{Symbol,Any}()) =
    _sb_pop_cols_expr!(cols, getf(t), t, data, stmts, pop_terms; obs_n, ran_terms, direct_terms, target, group_block_lookup, term_overrides)
_sb_pop_cols_expr!(cols, ::Any, t, data, stmts, pop_terms=(); obs_n=nothing, ran_terms=(), direct_terms=(), target=nothing, group_block_lookup=Dict(), term_overrides=Dict{Symbol,Any}()) =
    _sb_maybe_push_col!(cols, _sb_predictor_col(t, data, stmts, pop_terms; obs_n, ran_terms, direct_terms, target, group_block_lookup, term_overrides))
_sb_pop_cols_expr!(cols, ::typeof(&), t, data, stmts, _pop_terms=(); kwargs...) =
    _sb_interaction_cols!(cols, t, data, stmts)

# `a & b` interaction expansion. Three supported operand-type combinations:
#   cont x cont -> 1 column (elementwise product)
#   cont x cat  -> K-1 columns (a .* (c == k ? 1 : 0) for k=2..K)
#   cat  x cat  -> (K1-1)*(K2-1) columns (product of level-k1, level-k2 dummies)
# Reference level is always 1 (treatment coding; matches brms / vimpl).
# Columns are materialised at walker time (indicator math only uses data
# columns), stashed in `data` under a `int_<name...>` key, and pushed as
# Symbol refs for hcat downstream.
function _sb_interaction_cols!(cols, t::ExprColumn, data, stmts)
    args = getargs(t)
    length(args) == 2 ||
        error("sbimpl: interaction `&` expects exactly 2 operands, got $(length(args))")
    l = _sb_interaction_operand(args[1], data, stmts)
    r = _sb_interaction_operand(args[2], data, stmts)
    _sb_interaction_expand!(cols, data, l, r)
end

_sb_interaction_operand(t::NamedColumn, _data, _stmts) = begin
    d_raw = parent(t)
    d = _as_data_column(d_raw)
    isnothing(d) && error(
        "sbimpl: interaction operand `$(name(t))` must be a raw data column, got $(typeof(d_raw))"
    )
    v = parent(d)
    _sb_interaction_operand_kind(t, v, _sb_cat_levels(t))
end
# A transformed raw-data term (for example `zscale(math)`) first uses the
# ordinary predictor-term lowering. Pure/data-derived terms leave a concrete
# vector in `data`; parameter-owning terms (`mo`, `me`, `s`, `gp`, `ar`, ...)
# return a Stan variable name instead and are rejected below. This keeps
# transformed interactions on the same fit/reprocess constants as their
# standalone term without ever snapshotting a latent parameter as data.
_sb_interaction_operand(t::ExprColumn, data, stmts) = begin
    col_name = _sb_predictor_col(t, data, stmts)
    haskey(data, col_name) || error(
        "sbimpl: interaction operand `$(getf(t))(...)` is parameter-owning, not a data-materialized transform; ",
        "supported transformed operands include `zscale`, `standardize`, `center`, `protect`, ",
        "and pure expressions in raw data columns"
    )
    v = data[col_name]
    rv = _as_real_vec(v)
    isnothing(rv) && error(
        "sbimpl: transformed interaction operand `$col_name` has unsupported eltype $(eltype(v))"
    )
    (; kind=:cont, name=col_name, vec=collect(Float64, rv))
end
_sb_interaction_operand_kind(t, v, levels) = begin
    n_levels, idx = _sb_level_index(levels)
    # Single-level factor: the expander's `2:n_levels` loops are empty, so this
    # contributes 0 interaction columns uniformly (no shape special-case).
    (; kind=:cat, name=name(t), n_levels, idx)
end
_sb_interaction_operand_kind(t, v, ::Nothing) = begin
    rv = _as_real_vec(v)
    isnothing(rv) && error("sbimpl: interaction operand `$(name(t))` has unsupported eltype $(eltype(v))")
    (; kind=:cont, name=name(t), vec=collect(Float64, rv))
end
_sb_interaction_operand(t, _data, _stmts) = error(
    "sbimpl: interaction operand must be a raw-data NamedColumn or data-materialized ExprColumn, got $(typeof(t)); ",
    "interactions with parameter-owning terms such as `mo` / `me` / `s` / `gp` / `ar` are not supported"
)

# cont x cont
_sb_interaction_expand!(cols, data, l::NamedTuple{<:Any,<:Tuple}, r::NamedTuple{<:Any,<:Tuple}) = begin
    if l.kind === :cont && r.kind === :cont
        col_name = Symbol(:int_, l.name, :_x_, r.name)
        data[col_name] = l.vec .* r.vec
        _sb_record_preproc!(data, col_name,
            PreprocEntry(:interaction, nothing, (l.name, r.name), false))
        push!(cols, col_name)
    elseif l.kind === :cont && r.kind === :cat
        for lvl in 2:r.n_levels
            col_name = Symbol(:int_, l.name, :_x_, r.name, :_lvl_, lvl)
            data[col_name] = Float64[l.vec[i] * (r.idx[i] == lvl ? 1.0 : 0.0) for i in eachindex(l.vec)]
            push!(cols, col_name)
        end
    elseif l.kind === :cat && r.kind === :cont
        # Symmetric: reuse the :cont × :cat branch with swapped operands so
        # column names consistently put the cont term first.
        _sb_interaction_expand!(cols, data, r, l)
    elseif l.kind === :cat && r.kind === :cat
        n = length(l.idx)
        length(r.idx) == n || error(
            "sbimpl: interaction `$(l.name):$(r.name)`: operand lengths mismatch ($n vs $(length(r.idx)))"
        )
        for lvl1 in 2:l.n_levels, lvl2 in 2:r.n_levels
            col_name = Symbol(:int_, l.name, :_lvl_, lvl1, :_x_, r.name, :_lvl_, lvl2)
            data[col_name] = Float64[(l.idx[i] == lvl1 && r.idx[i] == lvl2) ? 1.0 : 0.0 for i in 1:n]
            push!(cols, col_name)
        end
    else
        error("sbimpl: unsupported interaction operand combination (`$(l.kind)` x `$(r.kind)`)")
    end
end

# Predictor column emitter. `stmts` is threaded in so terms that need their own
# `~` statement (e.g. `mo(c)`) can push before returning their column symbol.
# Integer `1` -> intercept, NamedColumn -> reference by name, ExprColumn(mo, c)
# -> submodel-sampled contrast column.
_sb_predictor_col(t::Int, data, _stmts, pop_terms=(); obs_n::Union{Symbol,Nothing}=nothing, ran_terms=(), direct_terms=(), group_idx=nothing, target=nothing, kwargs...) = begin
    t == 1 || error("sbimpl: integer term must be `1` for intercept, got `$t`")
    # Five-tier length probe, in priority order:
    #   1. A data-backed peer in the same formula's terms (`_sb_n_obs_probe`).
    #      Deterministic for any mixed-intercept formula like `y ~ 1 + x`.
    #   1b. A data-backed column NESTED inside one of those terms
    #      (`_sb_n_obs_probe_deep`). A formula whose population terms are ALL
    #      wrapped — `loc ~ 1 + mo(diet) + hsgp(x)` — has no top-level peer, so
    #      tier 1 returns nothing even though the formula names its own row axis
    #      plainly. Tiers 2 and 3 then guess an axis, which is right only while
    #      every frame in the model has the same length: an intercept on a
    #      SECONDARY frame (`ragged(x, group)`) got `rep_vector(1., num_elements(weight))`
    #      — the SUBJECT axis — inside an `X` matrix sized by the event axis.
    #      stanc accepts that (both extents are runtime), so it fails as a
    #      dimension error at instantiation rather than at lowering.
    #   1d. A CATEGORICAL peer in the same formula (`direct_terms`). The term
    #      classifier (`_sb_classify_term!`) routes a bare integer-backed
    #      `NamedColumn` — `log(y_scale) ~ 1 + source` — into `direct_terms`,
    #      not `pop_terms`, because it expands to dummy columns. Tiers 1/1b see
    #      only `pop_terms`, so before this tier `pop_terms == [1]` and every
    #      deterministic probe came up empty even though the formula names its
    #      row axis in plain sight. On a multi-axis model that then hit tier 3
    #      and refused, telling the user to add a group term when `source` was
    #      already right there (reported against a downstream PKPD consumer).
    #      Uses the tier-1b probe rather than tier 1a's: a direct term may be
    #      backed by STRINGS, and `_sb_n_obs_probe` would hand back that raw
    #      name unguarded, emitting `num_elements(<string column>)` — which
    #      StanBlocks cannot type. `_sb_n_obs_probe_deep`'s live-numeric-vector
    #      guard admits the integer case and correctly skips the string one.
    #   1c. The formula's own GROUP term (`_sb_group_n_obs_probe`). A formula
    #      whose only population term IS the intercept — `log_ka ~ 1 + (1|p|subject)`
    #      — has no top-level peer for tier 1 and nothing wrapped for tier 1b,
    #      yet it still names its row axis unambiguously: the grouping factor
    #      IS the frame, so its per-row index column has exactly this formula's
    #      length. Deterministic, and it cannot pick a wrong frame the way
    #      tiers 2/3 can. This is the two-axis (`ragged(x, group)`) failure of
    #      tier 1b in the OPPOSITE direction: the per-SUBJECT `log_ka` was
    #      sized off an EVENT-axis column found in hash order, so
    #      `pop_log_ka + r_log_ka_p_subject` added an 11-vector to a 2-vector
    #      and every log-density evaluation threw (snag
    #      `two-axis-brm-an-9881c01b`, reported by a downstream PKPD consumer).
    #      Deliberately ranked BELOW tiers 1/1b rather than ahead of them: a
    #      population peer in the same formula is on that same row axis by
    #      construction, so promoting the group probe would rewrite the emitted
    #      extent for every ordinary mixed model (`y ~ 1 + x + (1|g)`) while
    #      fixing nothing.
    #      A caller EMITTING a ranef block's Z columns passes `group_idx`
    #      directly instead: it already holds the block's per-row index, and a Z
    #      column's row axis is the grouping factor's by construction. The two
    #      spellings are the same tier and never both apply — the population
    #      path has `ran_terms` and no `group_idx`, the ranef path the reverse.
    #   2. The observation column threaded from the likelihood walker
    #      (`obs_n`). Covers purely-intercept formulas like `loc ~ 1` whose
    #      length matches the observed `~` target consuming `loc`.
    #   3. Hash-order fallback (`_sb_any_data_symbol`). Last resort; lossy
    #      for composite models with multi-length data and reachable only
    #      when none of (1), (1b), (1d), (1c) or (2) yields a name — i.e. a
    #      formula whose ONLY term is the intercept (no covariate of any kind,
    #      no group term) and whose target no observed likelihood references.
    #      A formula naming any live numeric data column resolves above.
    probe = _sb_n_obs_probe(pop_terms)
    # Tier 1a is unguarded: keep it only when it named a live numeric datum,
    # else fall through to the numeric-guarded tiers (a raw categorical/string
    # peer must not become `num_elements(<non-numeric>)`).
    isnothing(probe) || _sb_probe_is_live_numeric(data, probe) || (probe = nothing)
    isnothing(probe) && (probe = _sb_n_obs_probe_deep(pop_terms, data))
    isnothing(probe) && (probe = _sb_n_obs_probe_deep(direct_terms, data))
    isnothing(probe) && (probe = group_idx)
    isnothing(probe) && (probe = _sb_group_n_obs_probe(data, ran_terms))
    isnothing(probe) && !isnothing(obs_n) && (probe = obs_n)
    isnothing(probe) && (probe = _sb_any_data_symbol(data, target))
    probe_value = get(data, probe, nothing)
    extent = probe_value isa Integer && !(probe_value isa Bool) ?
        probe : :(num_elements($probe))
    :(rep_vector(1., $extent))
end
_sb_predictor_col(t::NamedColumn, data, _stmts, _pop_terms=(); kwargs...) = _predictor_col_for(t, parent(t), data)

# If the named column was already bound earlier in the walker (e.g.
# `ftime ~ gamma_time(...)` emitted a `ftime ~ _sb_gamma_time(...)` stmt),
# its parent is the sampling ExprColumn rather than a raw data column --
# just reference the Stan variable by name.
_predictor_col_for(t, ::ExprColumn, _) = name(t)
function _predictor_col_for(t, d::DataColumn, data)
    v = parent(d)
    rv = _as_real_vec(v)
    isnothing(rv) && error("sbimpl: non-numeric predictor `$(name(t))` not supported yet (wrap in `categorical(…)` once we add it)")
    data[name(t)] = collect(Float64, rv)
    name(t)
end
_predictor_col_for(t, d, _) = error("sbimpl: expected data-backed NamedColumn for `$(name(t))`, got $(typeof(d))")
_sb_predictor_col(t::ExprColumn, data, stmts, _pop_terms=(); kwargs...) = _sb_predictor_term!(stmts, data, getf(t), t; kwargs...)
_sb_predictor_col(t, _data, _stmts, _pop_terms=(); kwargs...) = error("sbimpl: unsupported predictor term $(typeof(t)): $t")

# Monotonic-effect increment-simplex level set for emission. During a frozen
# resample re-emission the FITTED level set — recorded in `PreprocEntry(:mo)` —
# sizes the simplex, exactly as it freezes the `<c>_idx` codes in
# `_sb_reprocess_entry!`; a fresh fit or a `freeze_constants=false` re-emission
# derives it from `raw`. Mirrors `_sb_hsgp_fit_for_emission`.
function _sb_mo_levels_for_emission(data, idx_name::Symbol, inner_name, raw)
    frozen = _sb_frozen_preproc_entry(data, idx_name, :mo, inner_name)
    isnothing(frozen) ? _sb_fit_levels(raw) : frozen.const_
end

# Monotonic-effect predictor: emit `mo_<c> ~ _sb_mo(; x=<c>_idx)` and return
# `mo_<c>` as the column. Scope: single NamedColumn inner arg backed by raw
# data. Other wrapped terms dispatch to their own methods below.
_sb_predictor_term!(stmts, data, ::typeof(mo), t;
                    term_overrides=Dict{Symbol,Any}(), kwargs...) = begin
    inner_name, raw = _sb_inner_data(:mo, only(getargs(t)))
    idx_name = Symbol(inner_name, :_idx)
    # The FITTED level set drives the increment-simplex dimension. On a frozen
    # resample re-emission it comes from the recorded `PreprocEntry(:mo)`, so a
    # prediction frame carrying only a SUBSET of the training levels keeps the
    # fitted `simplex[n_levels-1]` rather than shrinking it and indexing a frozen
    # `<c>_idx` code past the simplex (snag reprocess-freeze-80ddd7e4).
    levels = _sb_mo_levels_for_emission(data, idx_name, inner_name, raw)
    n_levels = length(levels)
    col_name = Symbol(:mo_, inner_name)
    if n_levels < 2
        # Single-level factor: 0 increments -> the free-beta monotonic effect is
        # identically 0. Contribute NO column (returning `nothing`, which
        # `_sb_pop_cols!` / the ranef collector drop), so there is no free `beta`
        # and no `simplex[0]` (which Stan rejects). A K=1 `mo(c)` therefore
        # vanishes like a K=1 nominal factor: `mu ~ 1 + mo(c)` degenerates to
        # `mu ~ 1`. Sb is never asked to be clever about the degenerate simplex.
        return nothing
    end
    data[idx_name] = _sb_apply_levels(levels, raw)
    _sb_record_preproc!(data, idx_name, PreprocEntry(:mo, levels, inner_name, true))
    alpha = _sb_mo_alpha_expr(term_overrides, t, n_levels)
    push!(stmts, :($col_name ~ _sb_mo(; x=$idx_name, alpha=$alpha)))
    col_name
end
# Measurement-error predictor `me(x_obs, sd_x)`: emit a submodel that allocates
# a length-N latent `me_<x>` with prior std_normal and an observation
# likelihood `x_obs ~ normal(me_<x>, sd_x)`. Returns `me_<x>` as the predictor
# column so popefs supplies a free beta. `sd_x` must be a positive constant;
# per-row error sizes would require a vector kwarg and a tweaked submodel.
_sb_predictor_term!(stmts, data, ::typeof(me), t;
                    term_overrides=Dict{Symbol,Any}(), kwargs...) = begin
    args = getargs(t)
    length(args) == 2 || error("sbimpl: `me(x, sd)` expects 2 args, got $(length(args))")
    inner, sd_arg = args
    xname, raw = _sb_inner_data(:me, inner)
    v = _sb_real_vec(:me, xname, raw)
    sd_real = _as_real(sd_arg)
    isnothing(sd_real) && error("sbimpl: `me(x, sd)` expects a numeric constant `sd`, got $(typeof(sd_arg))")
    sd_real > 0 || error("sbimpl: `me(x, sd)` expects sd > 0 (got $sd_real)")
    sd_arg = sd_real
    data[xname] = collect(Float64, v)
    sd_name = Symbol(:sd_, xname)
    data[sd_name] = Float64(sd_arg)
    col_name = Symbol(:me_, xname)
    loc, scale = _sb_me_latent_args(term_overrides, t)
    stmt = :($col_name ~ _sb_me(; x_obs=$xname, sd_x=$sd_name,
                                  x_true_loc=$loc, x_true_scale=$scale))
    # One latent true covariate may feed several design collectors. In
    # particular, `me(x, sd)` can be both a population effect and a random
    # slope; those collectors share `stmts`, so reuse an exact earlier
    # submodel call instead of asking StanBlocks to bind the same return twice.
    stmt in stmts || push!(stmts, stmt)
    col_name
end


# Predictor-side `interval_censored(x; upper=lloq, lower=0)`. `x` is the
# observed covariate: `x == lloq` marks a BLOQ row and `x > lloq` is exact.
# BLOQ rows get one latent coordinate in `(lower, lloq)`; no flag is needed.
# The derived-data key is also the idempotence marker: a term used for both the
# population slope and `(… | group)` random slope must own ONE latent vector,
# with both design matrices referencing the same merged predictor.
_sb_predictor_term!(stmts, data, ::typeof(interval_censored), t;
                    term_overrides=Dict{Symbol,Any}(), kwargs...) = begin
    args = getargs(t)
    kw = getkwargs(t)
    length(args) == 1 || error(
        "sbimpl: predictor `interval_censored(x; upper=lloq)` expects one " *
        "positional observed-covariate column, got $(length(args))")
    (haskey(kw, :upper) && all(k -> k in (:lower, :upper), keys(kw))) || error(
        "sbimpl: predictor `interval_censored(x; upper=lloq)` requires `upper` " *
        "and accepts only the optional numeric `lower`; got " *
        "$(collect(keys(kw)))")
    x_name, x_raw = _sb_inner_data(:interval_censored, only(args))
    upper_name, upper_raw = _sb_inner_data(:interval_censored, kw.upper)
    lower = get(kw, :lower, 0.0)
    lower isa Real || error(
        "sbimpl: predictor `interval_censored($x_name; ...)` expects a numeric " *
        "constant for `lower`, got $(typeof(lower))")
    suffix = Symbol(x_name, :_, upper_name)
    exact_key = Symbol(:icp_exact_, suffix)
    lower_key = Symbol(:icp_lower_, suffix)
    upper_key = Symbol(:icp_upper_, suffix)
    exact_index_key = Symbol(:Jexact_, suffix)
    interval_index_key = Symbol(:Jinterval_, suffix)
    col_name = Symbol(:interval_censored_, suffix)

    # Population terms emit before random effects. Reusing the same term in a
    # random slope therefore returns the already-owned latent predictor rather
    # than duplicating its prior/submodel statement.
    haskey(data, exact_key) && return col_name

    plan = _sb_interval_censored_predictor_plan(
        x_name, x_raw, upper_name, upper_raw, lower)
    data[exact_key] = plan.x_exact
    data[lower_key] = plan.x_lower
    data[upper_key] = plan.x_upper
    data[exact_index_key] = plan.Jexact
    data[interval_index_key] = plan.Jinterval
    _sb_record_preproc!(data, exact_key, PreprocEntry(
        :interval_censored_predictor,
        (; lower=Float64(lower), lower_key, upper_key, exact_index_key,
           interval_index_key),
        (x_name, upper_name), true))

    loc, scale = _sb_me_latent_args(term_overrides, t)
    push!(stmts, :($col_name ~ _sb_interval_censored_predictor(;
        x_exact=$exact_key, x_lower=$lower_key, x_upper=$upper_key,
        Jexact=$exact_index_key, Jinterval=$interval_index_key,
        x_true_loc=$loc, x_true_scale=$scale)))
    col_name
end
# Penalized thin-plate predictor `s(x)`. Fits a frozen rank-10 TPS eigenbasis
# from the raw training column, then stashes its two-column null-space matrix
# and eight-column penalty-whitened range matrix as Stan data. `_sb_s` owns the
# flat null-space coefficients, penalized coefficients, and smoothing SD; the
# returned contribution is a direct summand (no extra `popefs` beta). Only
# the default basis is supported -- `bs` and `k=`/`knots=` are follow-ons.
_sb_predictor_term!(stmts, data, ::typeof(s), t;
                    term_overrides=Dict{Symbol,Any}(), kwargs...) = begin
    args = getargs(t)
    length(args) == 1 || error("sbimpl: `s(x)` expects 1 positional arg, got $(length(args))")
    isempty(getkwargs(t)) || error("sbimpl: `s(x)` does not support keyword arguments yet")
    xname, raw = _sb_inner_data(:s, only(args))
    v = _sb_real_vec(:s, xname, raw)
    fit = _sb_fit_spline(v)
    Xnull, Zpen = _sb_apply_spline(fit, v)
    Xnull_name = Symbol(:Xnull_, xname)
    Zpen_name = Symbol(:Zpen_, xname)
    data[Xnull_name] = Xnull
    data[Zpen_name] = Zpen
    # Frozen training centers/eigenbasis → fixed dimension. Reprocess evaluates
    # both matrices at new x values against these constants.
    _sb_record_preproc!(data, Xnull_name,
        PreprocEntry(:spline, (; fit, zpen_key=Zpen_name), xname, false))
    col_name = Symbol(:s_, xname)
    sd_family, sd_rate = _sb_term_sd_args(term_overrides, t)
    push!(stmts, :($col_name ~ _sb_s(; Xnull=$Xnull_name, Zpen=$Zpen_name,
                                       sd_family=$sd_family, sd_rate=$sd_rate)))
    col_name
end

# Two-margin tensor-product cubic-regression spline. The Julia-side fit records
# the marginal knot/penalty decomposition and training centering constants;
# `_sb_t2` owns the three unpenalized NN coefficients plus independent RR/RN/NR
# smoothing scales and standardized range coefficients. It is therefore a
# direct summand, never multiplied by an additional `popefs` beta.
_sb_predictor_term!(stmts, data, ::typeof(t2), t;
                    target::Union{Symbol,Nothing}=nothing,
                    term_overrides=Dict{Symbol,Any}(), kwargs...) = begin
    args = getargs(t)
    length(args) == 2 || error(
        "sbimpl: `t2(x, z)` expects exactly 2 positional margins, got $(length(args))")
    kw = getkwargs(t)
    _check_term_kwargs(t2, kw)
    k, _, _ = _sb_t2_options(kw)
    names, axes = _sb_gp_axes(:t2, args)
    fit = _sb_fit_t2(axes[1], axes[2]; k)
    Xfixed, Zrr, Zrn, Znr = _sb_apply_t2(fit, axes[1], axes[2])
    axes_suffix = join(string.(names), "_")
    suffix = isnothing(target) ? axes_suffix : string(target, "_", axes_suffix)
    Xfixed_name = Symbol(:Xfixed_t2_, suffix)
    Zrr_name = Symbol(:Zrr_t2_, suffix)
    Zrn_name = Symbol(:Zrn_t2_, suffix)
    Znr_name = Symbol(:Znr_t2_, suffix)
    data[Xfixed_name] = Xfixed
    data[Zrr_name] = Zrr
    data[Zrn_name] = Zrn
    data[Znr_name] = Znr
    _sb_record_preproc!(data, Xfixed_name, PreprocEntry(:tensor_spline,
        (; fit, zrr_key=Zrr_name, zrn_key=Zrn_name, znr_key=Znr_name),
        names, false))
    col_name = Symbol(:t2_, suffix)
    sd_family, sd_rate = _sb_term_sd_args(term_overrides, t)
    push!(stmts, :($col_name ~ _sb_t2(;
        Xfixed=$Xfixed_name, Zrr=$Zrr_name, Zrn=$Zrn_name, Znr=$Znr_name,
        sd_family=$sd_family, sd_rate=$sd_rate)))
    col_name
end

# `gp(x...)` is the exact GP term. It records an N x d predictor matrix and
# delegates covariance construction + non-centred sampling to `_sb_gp` (one
# shared length scale) or `_sb_gp_aniso` (one per axis).
_sb_predictor_term!(stmts, data, ::typeof(gp), t; group_block_lookup=Dict(),
                    term_overrides=Dict{Symbol,Any}(), kwargs...) = begin
    args = getargs(t); kw = getkwargs(t)
    _check_term_kwargs(gp, kw)
    names, axes = _sb_gp_axes(:gp, args)
    suffix = join(string.(names), "_")
    X_name = Symbol(:X_gp_, suffix)
    col_name = Symbol(:gp_, suffix)
    data[X_name] = _sb_gp_matrix(axes)
    _sb_record_preproc!(data, X_name, PreprocEntry(:gp, nothing, names, false))
    jitter = Float64(get(kw, :jitter, 1e-9))
    cov = _sb_gp_cov(kw, :gp)
    if cov === :periodic
        length(names) == 1 || error(
            "sbimpl: `gp(...; cov=:periodic)` supports exactly one axis, got " *
            "$(length(names))")
        _sb_gp_iso(kw, :gp) || error(
            "sbimpl: `gp(...; cov=:periodic)` has one axis and one length " *
            "scale; `iso=false` has no meaning here")
        period = _sb_gp_period(kw, :gp, cov)
        submodel = _sb_gp_submodel_expr(:_sb_gp_periodic, term_overrides, t)
        push!(stmts, :($col_name ~ $submodel(;
            X=$X_name, jitter=$jitter, period=$period)))
        return col_name
    end
    submodel = _sb_gp_submodel_expr(
        _sb_gp_iso(kw, :gp) ? :_sb_gp : :_sb_gp_aniso, term_overrides, t)
    push!(stmts, :($col_name ~ $submodel(; X=$X_name, jitter=$jitter)))
    col_name
end

# `hsgp(x...; k, c, by, iso, domain, orthogonal_to)` is the Hilbert-space
# approximation. Raw axes retain the historical Julia-precomputed basis.
# A one-dimensional model-derived axis instead evaluates that basis inside Stan
# and therefore requires an explicit fixed domain.
function _sb_hsgp_fit_for_emission(data, key, names, axes, K, c, iso,
                                   domain_fits, orthogonal_to)
    frozen = _sb_frozen_preproc_entry(data, key, :hsgp, names)
    # Fit the fresh basis LAZILY — only when there is no frozen entry to return.
    # A constant prediction axis (e.g. a fixed future dose) trips `_sb_fit_hsgp`'s
    # degeneracy check, so evaluating it eagerly crashed frozen CV-template
    # re-emission even though `const_.fits` is what gets returned. (`f5a177d`
    # had this lazy; `2e929f7` regressed it while threading in `domain_fits`.)
    isnothing(frozen) && return isnothing(domain_fits) ?
        _sb_fit_hsgp(axes, K, c) : domain_fits
    const_ = frozen.const_
    frozen_domain = get(const_, :domain_fits, nothing)
    frozen_orthogonal = get(const_, :orthogonal_to, nothing)
    (const_.K == K && const_.c == c && const_.iso == iso &&
     frozen_domain == domain_fits && frozen_orthogonal === orthogonal_to) || error(
        "sbimpl: resample replay: fitted HSGP configuration for `$key` no " *
        "longer matches the re-emitted formula")
    const_.fits
end

function _sb_hsgp_raw_axes(args)
    inners = Tuple(_sb_named_inner(:hsgp, a) for a in args)
    names = Tuple(name(i) for i in inners)
    backings = Tuple(parent(i) for i in inners)
    raw = map(backings) do backing
        backing isa DataColumn ? parent(backing) : nothing
    end
    names, backings, raw
end

function _sb_hsgp_check_explicit_domain(fits, axes)
    isnothing(fits) && return
    for j in eachindex(axes)
        center, L = fits[j]
        lower, upper = center - L, center + L
        all(x -> lower <= x <= upper, axes[j]) || error(
            "sbimpl: `hsgp(...; domain=...)` axis $j contains training values " *
            "outside its fixed domain ($lower, $upper)")
    end
end

_sb_predictor_term!(stmts, data, ::typeof(hsgp), t; group_block_lookup=Dict(),
                    term_overrides=Dict{Symbol,Any}(), kwargs...) = begin
    args = getargs(t); kw = getkwargs(t)
    _check_term_kwargs(hsgp, kw)
    isempty(args) && error("sbimpl: `hsgp(x...)` expects at least one positional axis")
    names, backings, raw = _sb_hsgp_raw_axes(args)
    raw_flags = map(x -> !isnothing(x), raw)
    (all(raw_flags) || all(!, raw_flags)) || error(
        "sbimpl: `hsgp(...)` cannot mix raw-data and model-derived axes")
    is_raw = all(raw_flags)
    n_axes = length(args)
    K, c = _sb_hsgp_options(kw, n_axes)
    cov = _sb_gp_cov(kw, :hsgp)
    period = _sb_gp_period(kw, :hsgp, cov)
    cov === :periodic && return _sb_hsgp_periodic_term!(
        stmts, data, t, names, raw, is_raw, K, kw, period, term_overrides)
    domain_fits = _sb_hsgp_domain_fits(kw, n_axes; required=!is_raw)
    orthogonal_to = _sb_hsgp_orthogonal_to(kw, n_axes)
    orthogonal_to === :linear && haskey(kw, :by) && error(
        "sbimpl: `hsgp(...; orthogonal_to=:linear)` is an ungrouped " *
        "population-shape constraint and cannot be combined with `by=`")
    suffix = join(string.(names), "_")
    iso = _sb_gp_iso(kw, :hsgp)

    if !is_raw
        n_axes == 1 || error(
            "sbimpl: model-derived `hsgp(...)` currently supports exactly one axis")
        iso || error(
            "sbimpl: one-dimensional model-derived `hsgp(...)` requires `iso=true`")
        haskey(kw, :by) && error(
            "sbimpl: model-derived `hsgp(...; by=...)` is not supported; " *
            "use an ungrouped population HSGP residual")
        all(b -> b isa ExprColumn, backings) || error(
            "sbimpl: model-derived `hsgp($(only(names)))` expects a sampled " *
            "linear predictor or assignment, got $(typeof(only(backings)))")

        x_name = only(names)
        center, L = only(domain_fits)
        # The spectral frequencies depend only on the fixed approximation
        # domain. Evaluate them with a one-row dummy axis; PHI itself is rebuilt
        # from the live sampled x inside `_sb_hsgp_latent*`.
        _, omega2 = _sb_apply_hsgp(domain_fits, ([center],), K)
        omega2_name = Symbol(:omega2_hsgp_, suffix)
        data[omega2_name] = omega2
        _sb_record_static!(data, omega2_name)
        rho_lower = _sb_hsgp_rho_lower_data(domain_fits, K, true)
        col_name = Symbol(:hsgp_, suffix)
        submodel_name = orthogonal_to === :linear ?
            :_sb_hsgp_latent_orthogonal : :_sb_hsgp_latent
        submodel = _sb_gp_submodel_expr(submodel_name, term_overrides, t)
        push!(stmts, :($col_name ~ $submodel(;
            x=$x_name, omega2=$omega2_name, center=$center, L=$L,
            rho_lower=$rho_lower)))
        return col_name
    end

    axes = ntuple(n_axes) do j
        v = collect(Float64, _sb_real_vec(:hsgp, names[j], raw[j]))
        isempty(v) && error("sbimpl: `hsgp($(names[j]))` cannot use an empty axis")
        all(isfinite, v) || error(
            "sbimpl: `hsgp($(names[j]))` requires finite values")
        v
    end
    n = length(first(axes))
    all(v -> length(v) == n, axes) || error(
        "sbimpl: `hsgp(x...)` axes must have equal lengths (got $(length.(axes)))")
    _sb_hsgp_check_explicit_domain(domain_fits, axes)

    if haskey(kw, :by)
        block_info = _sb_find_group_block(hsgp, t, group_block_lookup)
        isnothing(block_info) && error(
            "sbimpl: `hsgp($suffix, by=...)` found no allocated per-group weight ",
            "block — prepass 2.5 should have allocated it")
        info = block_info
        gname = name(_sb_resolve_group_col((; kwarg=:by), t, data))
        PHI_name = Symbol(:PHI_hsgp_, suffix, :_by_, gname)
        omega2_name = Symbol(:omega2_hsgp_, suffix, :_by_, gname)
        rho_lower_name = Symbol(:rho_lower_hsgp_, suffix, :_by_, gname)
        fits = _sb_hsgp_fit_for_emission(
            data, PHI_name, names, axes, K, c, iso,
            domain_fits, orthogonal_to)
        PHI, omega2 = _sb_apply_hsgp(fits, axes, K)
        orthogonal_to === :linear &&
            (PHI = _sb_orthogonalize_hsgp_linear(PHI, only(axes)))
        data[PHI_name] = PHI
        data[omega2_name] = omega2
        data[rho_lower_name] = _sb_hsgp_rho_lower_data(fits, K, iso)
        _sb_record_preproc!(data, PHI_name, PreprocEntry(:hsgp,
            (; fits, K, c, iso, domain_fits, orthogonal_to,
             omega2_key=omega2_name, rho_lower_key=rho_lower_name),
            names, false))
        col_name = Symbol(:hsgp_, suffix, :_by_, gname)
        submodel = _sb_gp_submodel_expr(
            iso ? :_sb_hsgp_by : :_sb_hsgp_by_aniso, term_overrides, t)
        push!(stmts, :($col_name ~ $submodel(; PHI=$PHI_name, omega2=$omega2_name,
            rho_lower=$rho_lower_name,
            beta=$(info.block_name), group_idx=$(info.idx_name))))
        return col_name
    end

    PHI_name = Symbol(:PHI_hsgp_, suffix)
    omega2_name = Symbol(:omega2_hsgp_, suffix)
    rho_lower_name = Symbol(:rho_lower_hsgp_, suffix)
    fits = _sb_hsgp_fit_for_emission(
        data, PHI_name, names, axes, K, c, iso,
        domain_fits, orthogonal_to)
    PHI, omega2 = _sb_apply_hsgp(fits, axes, K)
    orthogonal_to === :linear &&
        (PHI = _sb_orthogonalize_hsgp_linear(PHI, only(axes)))
    data[PHI_name] = PHI
    data[omega2_name] = omega2
    data[rho_lower_name] = _sb_hsgp_rho_lower_data(fits, K, iso)
    _sb_record_preproc!(data, PHI_name, PreprocEntry(:hsgp,
        (; fits, K, c, iso, domain_fits, orthogonal_to,
         omega2_key=omega2_name, rho_lower_key=rho_lower_name),
        names, false))
    col_name = Symbol(:hsgp_, suffix)
    submodel = _sb_gp_submodel_expr(
        iso ? :_sb_hsgp : :_sb_hsgp_aniso, term_overrides, t)
    push!(stmts, :($col_name ~ $submodel(; PHI=$PHI_name, omega2=$omega2_name,
        rho_lower=$rho_lower_name)))
    col_name
end

# `hsgp(x; cov=:periodic, period=...)`: one raw axis, `2k` cosine/sine
# columns, harmonic indices and the periodic validity floor as data. The
# periodic basis has no boundary factor, domain or projection, and its grouped
# spelling is not implemented, so every such keyword is refused by name rather
# than silently ignored.
function _sb_hsgp_periodic_term!(stmts, data, t, names, raw, is_raw, K, kw,
                                 period, term_overrides)
    n_axes = length(names)
    n_axes == 1 || error(
        "sbimpl: `hsgp(...; cov=:periodic)` supports exactly one axis, got $n_axes")
    is_raw || error(
        "sbimpl: `hsgp(...; cov=:periodic)` currently requires a raw-data axis; " *
        "a model-derived periodic axis is not supported")
    _sb_gp_iso(kw, :hsgp) || error(
        "sbimpl: `hsgp(...; cov=:periodic)` has one axis and one length scale; " *
        "`iso=false` has no meaning here")
    for key in (:c, :domain, :orthogonal_to, :by)
        haskey(kw, key) && error(
            "sbimpl: `hsgp(...; cov=:periodic)` does not accept `$key=`: the " *
            "periodic cosine/sine basis has no boundary factor and needs no " *
            "domain, and its grouped/projected spellings are not implemented")
    end
    K1 = only(K)
    x = only(names)
    axis = collect(Float64, _sb_real_vec(:hsgp, x, only(raw)))
    isempty(axis) && error("sbimpl: `hsgp($x)` cannot use an empty axis")
    all(isfinite, axis) || error("sbimpl: `hsgp($x)` requires finite values")

    PHI_name = Symbol(:PHI_hsgp_, x)
    harmonics_name = Symbol(:harmonics_hsgp_, x)
    rho_lower_name = Symbol(:rho_lower_hsgp_, x)
    _sb_hsgp_periodic_frozen_check(data, PHI_name, names, K1, period)
    data[PHI_name] = _sb_apply_hsgp_periodic(period, axis, K1)
    data[harmonics_name] = _sb_hsgp_periodic_harmonics(K1)
    data[rho_lower_name] = _sb_hsgp_periodic_rho_lower(K1)
    _sb_record_preproc!(data, PHI_name, PreprocEntry(:hsgp,
        (; cov=:periodic, period, K=K1, iso=true,
         harmonics_key=harmonics_name, rho_lower_key=rho_lower_name),
        names, false))
    col_name = Symbol(:hsgp_, x)
    submodel = _sb_gp_submodel_expr(:_sb_hsgp_periodic, term_overrides, t)
    push!(stmts, :($col_name ~ $submodel(; PHI=$PHI_name,
        harmonics=$harmonics_name, rho_lower=$rho_lower_name)))
    col_name
end

function _sb_term_group_block(::typeof(hsgp), call)
    kw = getkwargs(call)
    haskey(kw, :by) || return nothing
    _check_term_kwargs(hsgp, kw)
    _sb_gp_cov(kw, :hsgp) === :periodic && error(
        "sbimpl: `hsgp(...; cov=:periodic)` does not accept `by=`: the " *
        "grouped periodic basis is not implemented")
    args = getargs(call)
    isempty(args) && error("sbimpl: `hsgp(x...; by=...)` expects at least one positional axis")
    names = Tuple(name(_sb_named_inner(:hsgp, a)) for a in args)
    K, _ = _sb_hsgp_options(kw, length(args))
    fname = Symbol(:hsgpw_, join(string.(names), "_"))
    (; fields=[(; name=fname, n_per_group=prod(K), group=(; kwarg=:by), prior=:iid_normal)])
end

# `ar(time; p=1)` AR(1) residual submodel. Routes to `_sb_ar1`, which owns the
# phi / epsilon parameters and returns the per-row u[t] as a single length-N
# column. popefs multiplies by an overall beta -- harmless, but a direct-
# summand variant would skip it.
_sb_predictor_term!(stmts, data, ::typeof(ar), t; target=nothing, kwargs...) = begin
    args = getargs(t)
    kw = getkwargs(t)
    p = get(kw, :p, 1)
    p == 1 || error("sbimpl: `ar(time; p=$p)` only supports p=1 so far")
    length(args) == 1 || error("sbimpl: `ar(time; p=1)` expects 1 positional arg, got $(length(args))")
    xname, raw = _sb_inner_data(:ar, only(args))
    # Ensure the time column lands in `data`. The prepass already handles this
    # for named data columns, but be defensive -- the submodel uses it as a
    # length probe via `num_elements(time)`.
    data[xname] = collect(Float64, raw)
    # Namespace the AR(1) column (and thus the `_sb_ar1` phi/epsilon parameters it
    # owns) by the RESPONSE, not just the time column: two `ar(time; p=1)` terms
    # over the SAME time axis on different responses (e.g. a shared-Rt model with
    # `log_ru ~ 1 + ar(t)` AND `logit_ihr ~ 1 + ar(t)`) would otherwise both emit
    # `ar_<t> ~ _sb_ar1(...)` and collide (`name ∉ keys(info)`).
    col_name = isnothing(target) ? Symbol(:ar_, xname) : Symbol(:ar_, target, :_, xname)
    push!(stmts, :($col_name ~ _sb_ar1(; time=$xname)))
    col_name
end

# `dar(time; p=1)` is the direct differenced-AR trajectory. Unlike `ar`, the
# term owns the whole model-scale contribution: its beta/sigma/z parameters
# produce a zero-start path that is added to the formula intercept without a
# second population coefficient.
_sb_predictor_term!(stmts, data, ::typeof(dar), t;
                    target::Symbol, term_overrides=Dict{Symbol,Any}(), kwargs...) = begin
    args = getargs(t)
    kw = getkwargs(t)
    unknown = filter(k -> k !== :p, keys(kw))
    isempty(unknown) || error(
        "sbimpl: `dar(time; p=1)` accepts only `p`; unsupported keyword(s): " *
        join(unknown, ", "))
    p = get(kw, :p, 1)
    p == 1 || error("sbimpl: `dar(time; p=$p)` only supports p=1")
    length(args) == 1 || error(
        "sbimpl: `dar(time; p=1)` expects 1 positional arg, got $(length(args))")
    xname, raw = _sb_inner_data(:dar, only(args))
    v = collect(Float64, _sb_real_vec(:dar, xname, raw))
    isempty(v) && error("sbimpl: `dar($xname)` cannot use an empty time axis")
    all(isfinite, v) || error("sbimpl: `dar($xname)` requires finite time values")
    all(>(0), diff(v)) || error(
        "sbimpl: `dar($xname)` requires a strictly increasing, unique time axis")
    data[xname] = v
    col_name = Symbol(:dar_, target, :_, xname)
    submodel = _sb_dar_submodel_expr(term_overrides, t)
    push!(stmts, :($col_name ~ $submodel(; time=$xname)))
    col_name
end
# Vector-wise wrapper predictors: `zscale`, `standardize`, and `center`
# need the whole inner column to compute (mean / sd are not element-wise),
# so the generic broadcast-based fallback in `_sb_materialize_vec` won't
# do. Materialize the inner separately, apply the transform, stash. (The
# brms-style `protect(...)` no-op is handled by the generic fallback once
# `protect(x::Real) = x` is defined in macro.jl.)
for (fn, kind, fitf, applyf) in (
        (:zscale,      :zscale,      :_sb_fit_zscale, :_sb_apply_zscale),
        (:standardize, :standardize, :_sb_fit_zscale, :_sb_apply_zscale),
        (:center,      :center,      :_sb_fit_center, :_sb_apply_center),
    )
    # fit → apply keeps the construct-time column byte-identical to the old
    # one-pass while exposing the fitted constant `c` for the preproc record
    # (so `reprocess` can re-apply it, frozen, to a new df). `raw_ref` is the
    # inner column-node tree, re-materialised on the new df at reprocess time.
    @eval function _sb_predictor_term!(stmts, data, ::typeof($fn), t; kwargs...)
        inner = only(getargs(t))
        v = collect(Float64, _sb_materialize_vec(inner))
        c = $fitf(v)
        v_t = $applyf(c, v)
        cn = _sb_wrapper_col_name(Symbol($(QuoteNode(fn))), inner)
        data[cn] = v_t
        _sb_record_preproc!(data, cn, PreprocEntry($(QuoteNode(kind)), c, inner, false))
        cn
    end
end

# Fallback: a "plain" expression like `log(exposure)` or `a^2` reaching this
# point is treated as an implicit `protect(...)` -- materialize the whole
# subtree to a Stan data vector and let popefs supply the beta. Errors out
# if any leaf isn't a raw data column (e.g. references a sampled parameter
# directly), preserving the old "unsupported" diagnostic.
function _sb_materialize_protect_term!(stmts, data, f, t)
    try
        v = collect(Float64, _sb_materialize_vec(t))
        cn = _sb_wrapper_col_name(Symbol(f), t)
        data[cn] = v
        # Element-wise pure transform: no fitted constant, just re-materialise
        # the same expr tree against the new df (freeze-agnostic).
        _sb_record_preproc!(data, cn, PreprocEntry(:protect, nothing, t, false))
        return cn
    catch err
        _ee = _as_error_exception(err); isnothing(_ee) && rethrow()
        error("sbimpl: unsupported predictor-term function `$f` (supported: `mo`, `mo1`, `me`, `interval_censored`, `s`, `ar`, `dar`, `protect`, `zscale`, `center`, `standardize`, or any expression in raw data columns) -- materialization failed: $(_ee.msg)")
    end
end
_sb_predictor_term!(stmts, data, f::Function, t; kwargs...) =
    _sb_materialize_protect_term!(stmts, data, f, t)

# Does term `t` reference a sampled parameter (a NamedColumn NOT backed by a raw
# data column)? Mirrors `_materialize_named`'s DataColumn / not-DataColumn split.
_sb_term_refs_param(t::NamedColumn) = !(parent(t) isa DataColumn)
_sb_term_refs_param(t::ExprColumn) = any(_sb_term_refs_param, getargs(t))
_sb_term_refs_param(_) = false

# `coef * col` under `~`: a sampled coefficient times a data column is an
# ASSIGNMENT-path expression (`lp = coef * col`, emitted as `.*`), not a `~`
# formula summand -- the `scalar*data` LP escape hatch was dropped in 84434c7 so
# the product has exactly one spelling. Reject that shape with the exact remedy;
# a `*` over raw data columns only is still a valid protect-style materialised
# term and falls through to the generic path above.
function _sb_predictor_term!(stmts, data, ::typeof(*), t; target=nothing, kwargs...)
    if _sb_term_refs_param(t)
        lp = isnothing(target) ? "<lp>" : string(target)
        error("sbimpl: a `~` predictor RHS multiplies a sampled coefficient by a data " *
              "column (a `coef * col` term); that is an assignment, not a formula summand. " *
              "Write it with `=` instead of `~`: `$lp = <coef> * <col>` (emitted as `.*`), " *
              "not `$lp ~ 0 + <coef> * <col>`.")
    end
    _sb_materialize_protect_term!(stmts, data, *, t)
end

# Recursively materialize an ExprColumn / NamedColumn tree into a plain
# vector, walking only data-backed leaves. Used by the wrapper predictors
# (protect / zscale / etc) to compute their column at transpile time.
_sb_materialize_vec(x::Number) = x
_sb_materialize_vec(x::NamedColumn) = _materialize_named(x, parent(x))
_materialize_named(_, d::DataColumn) = parent(d)
_materialize_named(x, _) = error(
    "sbimpl: cannot materialize NamedColumn `$(name(x))` -- only raw data columns supported inside `protect` / `zscale` / `center` / `standardize`")
_sb_materialize_vec(x::ExprColumn) = broadcast(getf(x), map(_sb_materialize_vec, getargs(x))...)
_sb_materialize_vec(x) = error("sbimpl: cannot materialize $(typeof(x)) inside wrapper predictor")

# Fit/apply split for the vector-wise standardisers. `_sb_fit_*` computes the
# data-derived constant; `_sb_apply_*` applies a (possibly frozen) constant to
# a vector. The fused `_sb_zscale`/`_sb_center` keep construct-time behaviour
# byte-identical (fit∘apply == the old one-pass), and `reprocess` reuses the
# apply half with a frozen constant for prediction-replay (decision nr3v8n A).
_sb_fit_zscale(v::AbstractVector{<:Real}) = _brm_fit_zscale(v)
_sb_apply_zscale(c::Tuple, v::AbstractVector{<:Real}) =
    _brm_apply_zscale(c, v)
_sb_zscale(v::AbstractVector{<:Real}) = _sb_apply_zscale(_sb_fit_zscale(v), v)

_sb_fit_center(v::AbstractVector{<:Real}) = _brm_fit_center(v)
_sb_apply_center(mu::Real, v::AbstractVector{<:Real}) =
    _brm_apply_center(mu, v)
_sb_center(v::AbstractVector{<:Real}) = _sb_apply_center(_sb_fit_center(v), v)

# Stable, human-readable column name for a wrapped predictor. When the
# inner is a single NamedColumn we tag with its name; otherwise we hash
# the expr structure so multiple `protect(...)` summands don't collide.
_sb_wrapper_col_name(prefix::Symbol, inner) =
    _brm_wrapper_col_name(prefix, inner)

_n_obs_name(t::NamedColumn) = _n_obs_named_data(t, parent(t))
_n_obs_name(_) = nothing
_n_obs_named_data(t, ::DataColumn) = name(t)
_n_obs_named_data(args...) = nothing

_sb_n_obs_probe(terms) = begin
    for t in terms
        n = _n_obs_name(t)
        isnothing(n) || return n
    end
    nothing
end

# A probe name is usable in `num_elements(<name>)` only if it is a live numeric
# Stan datum. Tier 1a (`_sb_n_obs_probe`) is unguarded and can name a raw
# categorical/string column whose emitted Stan form is a derived label — the raw
# column is non-numeric and is dropped before the SlicModel — so the caller
# validates tier 1a's answer with this before using it, falling through to the
# numeric-guarded tiers otherwise.
_sb_probe_is_live_numeric(data, k::Symbol) =
    haskey(data, k) && data[k] isa AbstractVector{<:Real}

# Tier 1b of the intercept length probe (see `_sb_predictor_col(::Int, …)`):
# descend into WRAPPED population terms for a data-backed column of this
# formula's own row axis. Only consulted when the narrow probe above found
# nothing, so it can never displace an existing tier-1 answer — it only ever
# replaces a downstream GUESS (the consuming likelihood's N, or hash order)
# with a column the formula itself names.
#
# A candidate must be a flat real vector still present in `data`: a wrapper's
# inner column can be non-numeric (`factor(vessel)` over strings) or dropped by
# its own emitter, and `num_elements(...)` needs a live numeric Stan datum.
_sb_n_obs_probe_deep(terms, data) = begin
    for t in terms
        n = _n_obs_name_deep(t, data)
        isnothing(n) || return n
    end
    nothing
end
_n_obs_name_deep(_t, _data) = nothing
_n_obs_name_deep(t::NamedColumn, data) = begin
    parent(t) isa DataColumn || return nothing
    k = name(t)
    (haskey(data, k) && data[k] isa AbstractVector{<:Real}) ? k : nothing
end
_n_obs_name_deep(t::ExprColumn, data) = begin
    getf(t) === (~) && return nothing
    for a in getargs(t)
        n = _n_obs_name_deep(a, data)
        isnothing(n) || return n
    end
    for v in values(getkwargs(t))
        n = _n_obs_name_deep(v, data)
        isnothing(n) || return n
    end
    nothing
end

# Tier 1c of the intercept length probe (see `_sb_predictor_col(::Int, …)`):
# a `(… | g)` term names this formula's row axis outright. The grouping factor
# IS the frame, so `g`'s per-row index column — the same `<g>_idx` the ranef
# block itself is about to consume — has exactly the formula's length. That is
# the whole fix for an intercept-only per-subject formula in a two-axis model:
# the answer was already a local in `_sb_linear_predictor!`, one `ran_terms`
# away, while the probe fell through to guessing.
#
# Sizing goes through `_sb_ensure_group_data!`, NOT a re-derivation of the
# index name: that helper is the single source of truth both the ID prepass and
# the plain-block emitter already call, it is idempotent by contract, and
# routing through it is what keeps this probe from drifting out of lockstep
# with the name the ranef block actually declares.
#
# Group shapes with no single per-row index column fall through to the later
# tiers rather than guessing: `mm(...)` spreads each row across several
# memberships, so it has no `<g>_idx` of the formula's length to offer.
_sb_group_n_obs_probe(data, ran_terms) = begin
    for rt in ran_terms
        _, _, desc = _sb_ranef_parts(rt)
        n = _sb_group_row_idx(data, desc)
        isnothing(n) || return n
    end
    nothing
end
_sb_group_row_idx(data, g::NamedColumn) = first(_sb_ensure_group_data!(data, g))
_sb_group_row_idx(data, g::Tuple{NamedColumn,NamedColumn}) = _sb_ensure_group_data!(data, g).idx_name
_sb_group_row_idx(_data, _g) = nothing

_sb_any_data_symbol(data, target=nothing) = begin
    isempty(data) && error("sbimpl: can't emit `rep_vector(1., n)` — no data column seen yet. Make sure an observed `~` comes before the intercept-only predictor, or, if it is a single constant, declare it directly as a scalar parameter with its own prior (`x ~ <distribution>`).")
    # Prefer a flat length-N vector (numeric / integer) so `num_elements(...)` in
    # Stan resolves to an int. Skip ragged `Vector{<:AbstractVector}` layouts
    # (a downstream `-ext`'s `dose_times`) which StanBlocks serializes as a
    # `tuple(vector, array[] int)` that Stan's `num_elements` rejects.
    #
    # The pick is `Dict` HASH ORDER, so it is only meaningful when every
    # candidate has the same length — i.e. a single-frame model, where any
    # answer is right and real `~ 1` formulas depend on this tier. Once the
    # candidates span SEVERAL lengths the model has more than one row axis and
    # this is a coin flip: `num_elements(...)` is a runtime extent, so stanc
    # accepts the losing side and it dies as a dimension error on every
    # log-density evaluation instead of at lowering. Measured while handling
    # snag `two-axis-brm-an-9881c01b`: the same two-axis fixture picked the
    # RIGHT axis, then the WRONG one, after dropping a single unrelated column
    # from a NEIGHBOURING formula. So refuse rather than guess (decision
    # `0mt4q2s`) — the earlier tiers already resolve every case a formula can
    # state for itself, and this message names the frames it could not choose
    # between.
    first_hit = nothing
    # Keyed by length, so the message names each distinct row axis once. Hash
    # order does not leak into it: the entries are sorted by length below.
    by_len = Dict{Int,Symbol}()
    for (k, v) in data
        k === _SB_PREPROC_KEY && continue
        hit = _flat_vec_key(k, v)
        isnothing(hit) && continue
        isnothing(first_hit) && (first_hit = hit)
        get!(by_len, length(v), hit)
    end
    if length(by_len) > 1
        where_ = isnothing(target) ? "an intercept-only predictor" : "predictor `$target`"
        frames = join(("$n (e.g. `$k`)" for (n, k) in sort!(collect(by_len); by=first)), ", ")
        error(
            "sbimpl: cannot determine the row axis for $where_. The intercept is its ",
            "ONLY term — no covariate of any kind, no group term — and no observed ",
            "likelihood references its target, so there is nothing in the formula to ",
            "size the intercept from, and this model spans SEVERAL row axes, with ",
            "candidate lengths $frames. Picking one would be a guess that stanc accepts ",
            "and that then fails as a dimension error on every log-density evaluation. ",
            "An intercept-only predictor is a single constant, so the clearest fix is to ",
            "declare it directly as a scalar parameter with its own prior — ",
            "`$(isnothing(target) ? "loc" : target) ~ <distribution>` (e.g. `Normal(0, 1)`) — ",
            "which broadcasts wherever it is used and needs no row axis. If it is instead ",
            "meant to vary across a frame, name a column from that frame so its length is known.")
    end
    isnothing(first_hit) || return first_hit
    first(k for k in keys(data) if k !== _SB_PREPROC_KEY)
end

# Return `k` if `v` is a flat (non-ragged) vector, else `nothing` — replaces
# the old `_is_flat_vec` Bool predicate so the caller composes via the
# returned value rather than a branch on the test.
_flat_vec_key(k, ::AbstractVector{<:AbstractVector}) = nothing
_flat_vec_key(k, ::AbstractVector) = k
_flat_vec_key(_k, _v) = nothing


# ---- likelihood emitters: `y ~ Normal(loc, sigma)` etc. ----------------------

function _sb_likelihood!(stmts, target::Symbol, rhs::ExprColumn, data)
    f = getf(rhs)
    _sb_lik_family!(stmts, target, f, getargs(rhs), getkwargs(rhs), data)
end

_sb_weight_data_key(target::Symbol) = Symbol(:brm_weight_, target)

function _sb_weight_data!(data, target::Symbol,
                          plan::_BRMObservationWeightPlan)
    key = _sb_weight_data_key(target)
    haskey(data, key) && error(
        "sbimpl: reserved derived weight key `$key` collides with a model/data " *
        "column; rename that column")
    data[key] = plan.values
    _sb_record_preproc!(data, key, PreprocEntry(
        :observation_weight, (; kind=plan.kind, response=target),
        plan.source, false))
    key
end

function _sb_analytic_weighted_likelihood!(stmts, target::Symbol,
                                           distribution::ExprColumn,
                                           weight_key::Symbol, data)
    family = getf(distribution)
    family isa Type && family <: Normal || error(
        "sbimpl: `AnalyticWeights` currently support only `Normal` observations; " *
        "response `$target` uses `$family`")
    args = map(a -> _sb_scalar_expr(a, data), getargs(distribution))
    normal_args = _sb_stan_dist_args(family, args)
    length(normal_args) == 2 || error(
        "sbimpl: internal Normal lowering for analytic weights expected two " *
        "arguments, got $(length(normal_args))")
    location, scale = normal_args
    weighted_scale = Expr(:call, Symbol("./"), scale,
                          Expr(:call, :sqrt, weight_key))
    _sb_lik_stan_exprs!(stmts, target, :normal, (location, weighted_scale))
end

function _sb_objective_weighted_likelihood!(stmts, target::Symbol,
                                            distribution::ExprColumn,
                                            weight_key::Symbol, data)
    family = getf(distribution)
    family isa Type && family <: Distribution || error(
        "sbimpl: frequency/power weights currently require a Distributions.jl " *
        "family call for response `$target`, got `$family`")
    stan_name = _sb_stan_dist_name(family)
    isnothing(stan_name) && error(
        "sbimpl: weighted likelihood family `$family` is not supported yet")
    args = map(a -> _sb_scalar_expr(a, data), getargs(distribution))
    stan_args = _sb_stan_dist_args(family, args)
    weighted_rhs = Expr(:call, :weighted, stan_name, weight_key, stan_args...)
    push!(stmts, Expr(:call, :~, target, weighted_rhs))
end

function _sb_likelihood!(stmts, target::Symbol,
                         rhs::ExprColumn{typeof(weighted)}, data)
    response = get(data, target, nothing)
    response isa AbstractVector || error(
        "sbimpl: weighted response `$target` must be an observed vector, got " *
        "$(typeof(response))")
    plan = _brm_observation_weight_plan(
        rhs, target, response; prefix="sbimpl")
    isnothing(plan) && error("sbimpl: internal weighted response was not planned")
    weight_key = _sb_weight_data!(data, target, plan)
    if plan.kind === :analytic
        _sb_analytic_weighted_likelihood!(
            stmts, target, plan.distribution, weight_key, data)
    else
        _sb_objective_weighted_likelihood!(
            stmts, target, plan.distribution, weight_key, data)
    end
end
_sb_likelihood!(stmts, target, rhs, _) =
    error("sbimpl: likelihood RHS for `$target` must be an ExprColumn (got $(typeof(rhs)))")

# Existing ordinary families remain positional and retain their prior behavior.
# Wrapper families override this six-argument seam below so only the new public
# compositions interpret formula keywords.
_sb_lik_family!(stmts, target, fam, args, ::NamedTuple, data) =
    _sb_lik_family!(stmts, target, fam, args, data)

# One dispatch per likelihood family. Each method states the Stan name and
# implicitly the arity (by destructuring `args`). Julia constructor arguments
# are normalized centrally by `_sb_stan_dist_args` before they reach the Stan
# call, so model density, predictive RNG and pointwise log-likelihood all see
# the same exact parameterization.
_sb_lik_stan_exprs!(stmts, target, name::Symbol, arg_exprs) =
    push!(stmts, Expr(:call, :~, target, Expr(:call, name, arg_exprs...)))

_sb_lik_stan!(stmts, target, name::Symbol, args, data) =
    _sb_lik_stan_exprs!(
        stmts, target, name, map(a -> _sb_scalar_expr(a, data), args))

# `y ~ OrderedLogistic(eta)`: cumulative-link ordinal likelihood. Allocates
# `ordered[K-1]` cutpoints (K = max(y)) with a std_normal prior via typed-LHS,
# then emits `y ~ ordered_logistic(eta, cutpoints)` — dispatches against
# Stan's built-in `ordered_logistic_lpmf` (surfaced into SB's @builtin_module).
function _sb_lik_family!(stmts, target, ::Type{<:OrderedLogistic}, args::Tuple{Any}, data)
    y_raw = data[target]
    y = _as_int_vec(y_raw)
    isnothing(y) && error(
        "sbimpl: `OrderedLogistic` expects integer outcome data for `$target`, got $(typeof(y_raw))"
    )
    n_levels = maximum(y)
    # No `n_levels >= 2` guard: one uniform emission. At a single observed level
    # `n_cut == 0`, so this emits `ordered[0]` cutpoints and
    # `ordered_logistic(eta, cutpoints)` — both Stan-valid (`ordered[0]`
    # constructs; `ordered_logistic_lpmf(1 | eta, empty)` contributes 0). A
    # single-level outcome degenerates to a zero-information likelihood rather
    # than a shape-conditional error.
    n_cut = n_levels - 1
    cut_name = Symbol(target, :_cutpoints)
    push!(stmts, :($cut_name::ordered[$n_cut] ~ std_normal()))
    eta_expr = _sb_scalar_expr(args[1], data)
    push!(stmts, :($target ~ ordered_logistic($eta_expr, $cut_name)))
end

_sb_ordinal_structure_code(::Cumulative) = 1
_sb_ordinal_structure_code(::StoppingRatio) = 2
_sb_ordinal_structure_code(::Type{Cumulative}) = 1
_sb_ordinal_structure_code(::Type{StoppingRatio}) = 2
function _sb_ordinal_structure_code(x::ExprColumn)
    isempty(getargs(x)) && isempty(getkwargs(x)) || error(
        "sbimpl: ordinal structure tags take no arguments; use `Cumulative()` " *
        "or `StoppingRatio()`")
    f = getf(x)
    f === Cumulative && return 1
    f === StoppingRatio && return 2
    error("sbimpl: unsupported ordinal structure `$f`; use `Cumulative()` or " *
          "`StoppingRatio()`")
end
_sb_ordinal_structure_code(x) = error(
    "sbimpl: ordinal structure must be `Cumulative()` or `StoppingRatio()`, " *
    "got $(typeof(x))")

_sb_ordinal_link_code(::LogitLink) = 1
_sb_ordinal_link_code(::ProbitLink) = 2
_sb_ordinal_link_code(::CloglogLink) = 3
_sb_ordinal_link_code(::Type{LogitLink}) = 1
_sb_ordinal_link_code(::Type{ProbitLink}) = 2
_sb_ordinal_link_code(::Type{CloglogLink}) = 3
function _sb_ordinal_link_code(x::ExprColumn)
    isempty(getargs(x)) && isempty(getkwargs(x)) || error(
        "sbimpl: ordinal link tags take no arguments; use `LogitLink()`, " *
        "`ProbitLink()`, or `CloglogLink()`")
    f = getf(x)
    f === LogitLink && return 1
    f === ProbitLink && return 2
    f === CloglogLink && return 3
    error("sbimpl: unsupported ordinal link `$f`; use `LogitLink()`, " *
          "`ProbitLink()`, or `CloglogLink()`")
end
_sb_ordinal_link_code(x) = error(
    "sbimpl: ordinal link must be `LogitLink()`, `ProbitLink()`, or " *
    "`CloglogLink()`, got $(typeof(x))")

_sb_ordinal_has_fixed_intercept(x::Real) = !iszero(x)
_sb_ordinal_has_fixed_intercept(x::NamedColumn) =
    _sb_ordinal_has_fixed_intercept_parent(parent(x))
_sb_ordinal_has_fixed_intercept(_) = false
_sb_ordinal_has_fixed_intercept_parent(p::ExprColumn) = begin
    getf(p) === (~) || return false
    _, rhs = getargs(p, 2)
    any(t -> t isa Integer && t == 1, _sb_terms(rhs))
end
_sb_ordinal_has_fixed_intercept_parent(_) = false

function _sb_ordinal_threshold_predictors!(data, target, raw, n_obs)
    raw isa Tuple || error(
        "sbimpl: `Ordinal(...; per_threshold=...)` expects a tuple of raw " *
        "numeric columns, for example `per_threshold=(treat,)`")
    names = Symbol[]
    for term in raw
        term isa NamedColumn && parent(term) isa DataColumn || error(
            "sbimpl: `Ordinal(...; per_threshold=...)` currently accepts only " *
            "raw numeric data columns; got $(typeof(term))")
        key = name(term)
        values = parent(parent(term))
        values isa AbstractVector{<:Real} || error(
            "sbimpl: ordinal threshold predictor `$key` must be numeric, got " *
            "$(typeof(values))")
        length(values) == n_obs || error(
            "sbimpl: ordinal threshold predictor `$key` has $(length(values)) " *
            "rows; outcome `$target` has $n_obs")
        all(isfinite, values) || error(
            "sbimpl: ordinal threshold predictor `$key` contains non-finite values")
        data[key] = collect(Float64, values)
        _sb_record_preproc!(data, key, PreprocEntry(
            :ordinal_threshold_predictor, nothing, key, false))
        push!(names, key)
    end
    Tuple(names)
end

function _sb_ordinal_discrimination_expr(raw, data)
    if raw isa Real
        isfinite(raw) && raw > 0 || error(
            "sbimpl: ordinal discrimination must be finite and strictly " *
            "positive, got $(repr(raw))")
    elseif raw isa NamedColumn && parent(raw) isa DataColumn
        values = parent(parent(raw))
        values isa AbstractVector{<:Real} && all(x -> isfinite(x) && x > 0, values) ||
            error("sbimpl: ordinal discrimination data `$(name(raw))` must " *
                  "contain only finite positive values")
    end
    _sb_scalar_expr(raw, data)
end

# Typed structure/link composition. Cumulative thresholds are ordered;
# stopping-ratio stage intercepts are deliberately unconstrained by ordering.
# A fixed eta intercept is non-identifiable with either threshold vector, so
# the new surface rejects it (the legacy OrderedLogistic shorthand remains
# unchanged for compatibility).
function _sb_lik_family!(stmts, target, ::Type{<:Ordinal},
                         args::Tuple{Any,Any,Any}, kwargs::NamedTuple, data)
    structure_raw, link_raw, eta_raw = args
    structure = _sb_ordinal_structure_code(structure_raw)
    link = _sb_ordinal_link_code(link_raw)
    _sb_ordinal_has_fixed_intercept(eta_raw) && error(
        "sbimpl: `Ordinal($target)` cannot include a fixed intercept in `eta`; " *
        "the estimated thresholds already supply the location. Use `eta ~ 0 + ...`.")

    raw = get(data, target, nothing)
    raw isa AbstractVector || error(
        "sbimpl: `Ordinal` expects an observed outcome vector for `$target`, " *
        "got $(typeof(raw))")
    levels = _sb_fit_levels(raw)
    n_levels = length(levels)
    # No `n_levels >= 2` guard: one uniform emission. At a single observed level
    # `n_cut == 0` — `ordered[0]`/`vector[0]` thresholds and an empty
    # `rep_matrix(0., N, 0)` threshold effect, all Stan-valid, degenerating to a
    # zero-information likelihood rather than a shape-conditional error.
    n_cut = n_levels - 1
    data[target] = _sb_apply_levels(levels, raw)
    _sb_record_preproc!(data, target, PreprocEntry(
        :ordinal_outcome, (; levels, n_levels), target, true))

    cut_name = Symbol(target, :_thresholds)
    if structure == 1
        push!(stmts, :($cut_name::ordered[$n_cut] ~ std_normal()))
    else
        push!(stmts, :($cut_name::vector[$n_cut] ~ std_normal()))
    end

    per_threshold_raw = get(kwargs, :per_threshold, ())
    per_threshold = _sb_ordinal_threshold_predictors!(
        data, target, per_threshold_raw, length(raw))
    structure == 1 && !isempty(per_threshold) && error(
        "sbimpl: `per_threshold` is currently supported for " *
        "`StoppingRatio()` only; unrestricted cumulative category-specific " *
        "effects can make cumulative probabilities non-monotone")

    effect_name = Symbol(target, :_threshold_effect)
    if isempty(per_threshold)
        push!(stmts, :($effect_name = rep_matrix(
            0., num_elements($target), $n_cut)))
    else
        X_name = Symbol(target, :_threshold_X)
        beta_name = Symbol(target, :_threshold_beta)
        n_terms = length(per_threshold)
        push!(stmts, Expr(:(=), X_name, Expr(:call, :hcat, per_threshold...)))
        # Stan's std_normal_lpdf is not matrix-vectorised. Reuse BRM's
        # array-of-vectors standard-normal prior, then rebuild a K-1 by p matrix
        # through the measured ranef_b_matrix helper for the design multiply.
        push!(stmts, :($beta_name::vector[$n_cut,$n_terms] ~ multi_std_normal()))
        beta_matrix = Expr(:call, :adjoint,
            Expr(:call, :ranef_b_matrix, beta_name))
        push!(stmts, Expr(:(=), effect_name,
            Expr(:call, :*, X_name, beta_matrix)))
    end

    eta = _sb_scalar_expr(eta_raw, data)
    discrimination = _sb_ordinal_discrimination_expr(
        get(kwargs, :discrimination, 1.0), data)
    _sb_lik_stan_exprs!(stmts, target, :brm_ordinal,
        (eta, cut_name, discrimination, structure, link, effect_name))
end
_sb_lik_family!(_, target, ::Type{<:Ordinal}, args, ::NamedTuple, _) = error(
    "sbimpl: `Ordinal($target)` expects exactly three positional arguments " *
    "`(structure, link, eta)`, got $(length(args))")

_sb_lik_family!(stmts, target, ::Type{<:ZeroInflatedPoisson},
                args::Tuple{Any,Any}, data) =
    _sb_lik_stan!(stmts, target, :zero_inflated_poisson, args, data)

_sb_lik_family!(stmts, target, ::Type{<:NegativeBinomial2},
                args::Tuple{Any,Any}, data) =
    _sb_lik_stan!(stmts, target, :neg_binomial_2, args, data)

function _sb_lik_family!(stmts, target, ::Type{<:BinomialLogit},
                         args::Tuple{Any,Any}, data)
    response = get(data, target, nothing)
    response isa AbstractVector || error(
        "sbimpl: `BinomialLogit` expects an observed vector for `$target`")
    trials = _brm_materialize_count_argument(
        first(args), length(response), "BinomialLogit trial count";
        prefix="sbimpl")
    _brm_validate_binomial_response(response, trials, target; prefix="sbimpl")
    _sb_lik_stan!(stmts, target, :binomial_logit, args, data)
end

_sb_von_mises_observations(data, target) = begin
    raw = get(data, target, nothing)
    raw isa AbstractVector || error(
        "sbimpl: von-Mises likelihood expects an observed vector for `$target`, " *
        "got $(typeof(raw))")
    all(y -> y isa Real && isfinite(y), raw) || error(
        "sbimpl: von-Mises outcome `$target` must contain only finite real values")
    raw
end

function _sb_validate_von_mises!(data, target, mu, kappa;
                                 interval=nothing, principal=false)
    raw = _sb_von_mises_observations(data, target)
    if kappa isa Real
        isfinite(kappa) && kappa > 0 || error(
            "sbimpl: von-Mises concentration `kappa` must be finite and strictly " *
            "positive, got $(repr(kappa))")
    end
    if principal
        lo, hi = interval
        bad = findfirst(y -> !(lo <= y < hi), raw)
        isnothing(bad) || error(
            "sbimpl: `CircularVonMises($target)` observation $(repr(raw[bad])) " *
            "at index $bad is outside the half-open interval " *
            "[$lo, $hi)")
    elseif mu isa Real
        lo, hi = mu - Float64(pi), mu + Float64(pi)
        bad = findfirst(y -> !(lo <= y <= hi), raw)
        isnothing(bad) || error(
            "sbimpl: `VonMises($target)` observation $(repr(raw[bad])) at index " *
            "$bad is outside Distributions.jl support [$lo, $hi] for mu=$mu")
    end
    nothing
end

# Distributions.jl's exact constructor semantics. One positional argument is
# kappa (mu defaults to zero), while the two-argument form is `(mu, kappa)`.
# The custom lpxf adds the moving-support/strict-domain guards before calling
# native Stan `von_mises_lpdf`, and supplies matching pointwise/RNG hooks.
function _sb_lik_family!(stmts, target, ::Type{<:VonMises}, args, data)
    length(args) in (1, 2) || error(
        "sbimpl: `VonMises` expects `VonMises(kappa)` or " *
        "`VonMises(mu, kappa)`, got $(length(args)) positional arguments")
    arg_exprs = map(a -> _sb_scalar_expr(a, data), args)
    mu, kappa = _sb_stan_dist_args(VonMises, arg_exprs)
    _sb_validate_von_mises!(data, target, mu, kappa)
    _sb_lik_stan_exprs!(
        stmts, target, :brm_von_mises, (mu, kappa, 0.0, 0.0, 0))
end

function _sb_lik_family!(stmts, target, ::Type{<:CircularVonMises},
                         args::Tuple{Any,Any}, kwargs::NamedTuple, data)
    mu, kappa = map(a -> _sb_scalar_expr(a, data), args)
    interval = _sb_circular_interval(kwargs)
    _sb_validate_von_mises!(data, target, mu, kappa;
                            interval, principal=true)
    lo, hi = interval
    _sb_lik_stan_exprs!(
        stmts, target, :brm_von_mises, (mu, kappa, lo, hi, 1))
end
_sb_lik_family!(_, target, ::Type{<:CircularVonMises}, args, ::NamedTuple, _) = error(
    "sbimpl: `CircularVonMises($target)` expects exactly two positional " *
    "arguments `(mu, kappa)`, got $(length(args))")

# Reference-class categorical regression. The user supplies one named scalar
# LP per non-reference class; the fitted outcome level order determines which
# class each argument owns. A leading all-zero row fixes class 1 as the
# reference. StanBlocks' categorical-logit contract is matrix[K, N], one
# observation's K logits per column, so transpose the row-wise hcat carrier.
function _sb_lik_family!(stmts, target, ::Type{<:CategoricalLogit},
                         args::Tuple, data)
    # No `isempty(args)` guard: one uniform emission. A single-level outcome has
    # zero non-reference classes, so 0 predictors is the correct arity for it —
    # `hcat(zero_reference)` is a `matrix[1, N]` and `categorical_logit` over one
    # category is Stan-valid (contributes 0). The `expected_n_levels` check below
    # still rejects a genuine predictor/level mismatch at K >= 2.
    all(a -> a isa NamedColumn && !(parent(a) isa DataColumn), args) || error(
        "sbimpl: `CategoricalLogit($target)` expects one existing scalar linear " *
        "predictor per non-reference class, got $(args)")

    raw = get(data, target, nothing)
    raw isa AbstractVector || error(
        "sbimpl: `CategoricalLogit` expects an observed outcome vector for " *
        "`$target`, got $(typeof(raw))")
    levels = _sb_fit_levels(raw)
    n_levels = length(levels)
    # No `n_levels >= 2` guard: one uniform emission (see the arity note above).
    expected_n_levels = length(args) + 1
    n_levels == expected_n_levels || error(
        "sbimpl: `CategoricalLogit($target)` observed $n_levels outcome levels " *
        "but received $(length(args)) non-reference predictors; expected " *
        "$(n_levels - 1). Outcome level order is $(collect(levels)).")

    data[target] = _sb_apply_levels(levels, raw)
    _sb_record_preproc!(data, target, PreprocEntry(
        :categorical_outcome, (; levels, n_levels), target, true))

    logits_name = Symbol(target, :_categorical_logits)
    zero_reference = :(rep_vector(0., num_elements($target)))
    eta_exprs = map(a -> _sb_scalar_expr(a, data), args)
    row_logits = Expr(:call, :hcat, zero_reference, eta_exprs...)
    push!(stmts, Expr(:(=), logits_name, Expr(:call, :adjoint, row_logits)))
    push!(stmts, :($target ~ categorical_logit($logits_name)))
end

# Mean/precision Beta-binomial convenience surface. Shape lowering stays in
# the emitted expression so scalar, vector, and linked-predictor arguments all
# share one method and StanBlocks can synthesize matching lpmfs/RNG paths.
function _sb_lik_family!(stmts, target, ::Type{<:BetaBinomial2},
                         args::Tuple{Any,Any,Any}, data)
    trials, mean, precision = map(a -> _sb_scalar_expr(a, data), args)
    alpha = Expr(:call, Symbol(".*"), mean, precision)
    beta = Expr(:call, Symbol(".*"), Expr(:call, :-, 1, mean), precision)
    push!(stmts, Expr(:call, :~, target,
        Expr(:call, :beta_binomial, trials, alpha, beta)))
end

# Distributions.jl expresses a location-scale Student-t as
# `LocationScale(loc, scale, TDist(nu))`. The prior path already supports this
# composition; likelihoods use the same lowering to Stan's
# `student_t(nu, loc, scale)` rather than rejecting the wrapper family.
function _sb_lik_family!(stmts, target, ::Type{<:LocationScale},
                         args::Tuple{Any,Any,Any}, data)
    loc, scale, base = args
    base_e = _as_expr_column(base)
    isnothing(base_e) && error(
        "sbimpl: `LocationScale(...)` third arg must be a distribution call, got $(typeof(base))")
    base_fam = getf(base_e)
    isnothing(_as_distribution_type(base_fam)) && error(
        "sbimpl: `LocationScale` base must be a Distribution type, got $(base_fam)")
    stan_name = _sb_stan_dist_name(base_fam)
    isnothing(stan_name) && error(
        "sbimpl: `LocationScale` over `$(base_fam)` -- no Stan-name mapping for the base.")
    base_args = _sb_stan_dist_args(
        base_fam, map(a -> _sb_scalar_expr(a, data), getargs(base_e)))
    length(base_args) >= 3 || error(
        "sbimpl: `LocationScale` over `$(base_fam)`: base lowers to ",
        "$(length(base_args)) Stan args, need >= 3 for location/scale slots.")
    composed = (base_args[1], _sb_scalar_expr(loc, data),
                _sb_scalar_expr(scale, data), base_args[4:end]...)
    _sb_lik_stan_exprs!(stmts, target, stan_name, composed)
end
# Single source of truth: Julia Distribution type -> Stan distribution
# function name. Both the likelihood path (`_sb_lik_family!` below) and
# the scalar-prior path (`_sb_emit_prior!` above) consult this. Adding
# a new family adds one entry here and both routes pick it up.
_sb_stan_dist_name(::Type{<:Normal})              = :normal
_sb_stan_dist_name(::Type{<:Cauchy})              = :cauchy
# Distributions.jl's `TDist(nu)` is 1-arg (df only); Stan's `student_t`
# is 3-arg (nu, mu, sigma). The name maps directly; the arg-shape
# adjustment is the `_sb_stan_dist_args` override below.
_sb_stan_dist_name(::Type{<:TDist})               = :student_t
_sb_stan_dist_name(::Type{<:Exponential})         = :exponential
_sb_stan_dist_name(::Type{<:Gamma})               = :gamma
_sb_stan_dist_name(::Type{<:Beta})                = :beta
_sb_stan_dist_name(::Type{<:Uniform})             = :uniform
_sb_stan_dist_name(::Type{<:LogNormal})           = :lognormal
_sb_stan_dist_name(::Type{<:Laplace})             = :double_exponential
_sb_stan_dist_name(::Type{<:Logistic})            = :logistic
_sb_stan_dist_name(::Type{<:Gumbel})              = :gumbel
_sb_stan_dist_name(::Type{<:Chisq})               = :chi_square
_sb_stan_dist_name(::Type{<:Frechet})             = :frechet
_sb_stan_dist_name(::Type{<:Rayleigh})            = :rayleigh
_sb_stan_dist_name(::Type{<:SkewNormal})          = :skew_normal
_sb_stan_dist_name(::Type{<:Pareto})              = :pareto
_sb_stan_dist_name(::Type{<:Erlang})              = :gamma
_sb_stan_dist_name(::Type{<:Arcsine})             = :beta
_sb_stan_dist_name(::Type{<:NormalCanon})         = :normal
_sb_stan_dist_name(::Type{<:SkewDoubleExponential}) = :skew_double_exponential
_sb_stan_dist_name(::Type{<:SkewedExponentialPower}) = :skew_double_exponential
_sb_stan_dist_name(::Type{<:Weibull})             = :weibull
_sb_stan_dist_name(::Type{<:InverseGamma})        = :inv_gamma
_sb_stan_dist_name(::Type{<:Bernoulli})           = :bernoulli
_sb_stan_dist_name(::Type{<:BernoulliLogit})      = :bernoulli_logit
_sb_stan_dist_name(::Type{<:Binomial})            = :binomial
_sb_stan_dist_name(::Type{<:BinomialLogit})       = :binomial_logit
_sb_stan_dist_name(::Type{<:BetaBinomial})        = :beta_binomial
_sb_stan_dist_name(::Type{<:Poisson})             = :poisson
_sb_stan_dist_name(::Type{<:NegativeBinomial})    = :neg_binomial
# Composition families. Distributions `Multinomial(n, p)` -> Stan
# `multinomial(obs | probs, N)` (StanBlocks' 3-arg builtin; N explicit); the
# per-row `int[n,K]` response form shares `probs` across rows.
_sb_stan_dist_name(::Type{<:Multinomial})         = :multinomial
_sb_stan_dist_name(::Type{<:Categorical})         = :categorical
_sb_stan_dist_name(::Type) = nothing

# Per-family argument normalization between Julia constructors and native Stan
# distributions.  Inputs here are already-lowered Stan expressions.  Besides
# parameterization changes, preserve Distributions.jl's shorter constructor
# forms rather than emitting an invalid native-Stan arity.
_sb_stan_dist_args(::Type, args) = args

_sb_stan_reciprocal(x) = Expr(:call, Symbol("./"), 1.0, x)
_sb_stan_success_odds(p) =
    Expr(:call, Symbol("./"), p, Expr(:call, :-, 1.0, p))

_sb_stan_dist_args(::Type{<:Normal}, ::Tuple{}) = (0.0, 1.0)
_sb_stan_dist_args(::Type{<:Normal}, args::Tuple{Any}) = (args[1], 1.0)
_sb_stan_dist_args(::Type{<:Cauchy}, ::Tuple{}) = (0.0, 1.0)
_sb_stan_dist_args(::Type{<:Cauchy}, args::Tuple{Any}) = (args[1], 1.0)

# `TDist(nu)` is standard Student-t; Stan requires explicit location/scale.
_sb_stan_dist_args(::Type{<:TDist}, args::Tuple{Any}) = (args[1], 0, 1)

# Explicit capability gate for Julia distribution composition. A Stan density
# name alone is insufficient: generic truncation/censoring additionally needs
# lcdf/lccdf companions with the same parameterization. Keep this list honest
# and executable rather than implicitly advertising every name-table entry.
_sb_cdf_family_kind(::Type) = nothing
_sb_cdf_family_kind(::Type{<:Normal})       = :continuous
_sb_cdf_family_kind(::Type{<:Exponential})  = :continuous
_sb_cdf_family_kind(::Type{<:LogNormal})    = :continuous
_sb_cdf_family_kind(::Type{<:Weibull})      = :continuous
_sb_cdf_family_kind(::Type{<:Poisson})      = :discrete

function _sb_composed_family(wrapper, args)
    length(args) in (1, 3) || error(
        "sbimpl: `$wrapper` expects a base distribution and either keyword ",
        "bounds or positional `(lower, upper)` bounds, got $(length(args)) arguments")
    base = _as_expr_column(first(args))
    isnothing(base) && error(
        "sbimpl: `$wrapper` first argument must be a distribution call, got ",
        "$(typeof(first(args)))")
    isempty(getkwargs(base)) || error(
        "sbimpl: `$wrapper` base distribution `$(getf(base))` cannot use formula keywords")
    family = getf(base)
    D = _as_distribution_type(family)
    isnothing(D) && error(
        "sbimpl: `$wrapper` base must be a Distributions.jl distribution type, ",
        "got `$family`")
    stan_name = _sb_stan_dist_name(D)
    isnothing(stan_name) && error(
        "sbimpl: `$wrapper` base family `$D` has no Stan distribution-name mapping")
    kind = _sb_cdf_family_kind(D)
    isnothing(kind) && error(
        "sbimpl: `$wrapper` base family `$D` has no generic CDF/CCDF composition ",
        "capability; add and test its density, pointwise, predictive, lcdf and ",
        "lccdf paths before advertising it")
    (; family=D, stan_name, stan_args=getargs(base), kind)
end

function _sb_wrapper_bounds(wrapper, args, kwargs::NamedTuple)
    plan = _brm_response_modifier_plan(
        wrapper, args, kwargs; prefix="sbimpl")
    isnothing(plan) && error(
        "sbimpl: internal unsupported response modifier `$wrapper`")
    plan.lower, plan.upper
end

# Compatibility name for downstream/tests that inspect the established
# StanBlocks lowering helper; semantics now live in the shared core.
_sb_normalize_bound(x) = _brm_normalize_response_bound(x)

_sb_bound_data(x::Real, _data) = x
_sb_bound_data(x::AbstractVector{<:Real}, _data) = x
_sb_bound_data(x::AbstractVector{<:AbstractVector{<:Real}}, _data) = x
_sb_bound_data(x::NamedColumn, data) = _sb_bound_data_named(x, parent(x), data)
_sb_bound_data_named(_x, d::DataColumn, _data) = parent(d)
_sb_bound_data_named(x, backing, _data) = error(
    "sbimpl: bound `$(name(x))` must be backed by observed data, got $(typeof(backing))")
_sb_bound_data(x, _data) = error(
    "sbimpl: bounds must be numeric literals or observed data columns, got $(typeof(x))")

_sb_composed_values(x::AbstractVector{<:AbstractVector}) =
    collect(Iterators.flatten(x))
_sb_composed_values(x) = x

function _sb_validate_bound_segments(wrapper, target, label, y, b)
    y isa AbstractVector{<:AbstractVector} || return nothing
    b isa Real && return nothing
    b isa AbstractVector{<:AbstractVector} || error(
        "sbimpl: `$wrapper` $label bound for ragged response `$target` must " *
        "be scalar or have the same ragged grouping")
    length(b) == length(y) && length.(b) == length.(y) || error(
        "sbimpl: `$wrapper` $label bound for ragged response `$target` has " *
        "group lengths $(length.(b)); expected $(length.(y))")
    nothing
end

function _sb_validate_bounds(wrapper, target, lower, upper, data; check_order=true)
    raw_y = data[target]
    y = _sb_composed_values(raw_y)
    for (label, bound) in ((:lower, lower), (:upper, upper))
        isnothing(bound) && continue
        raw_b = _sb_bound_data(bound, data)
        _sb_validate_bound_segments(wrapper, target, label, raw_y, raw_b)
        b = _sb_composed_values(raw_b)
        b isa AbstractVector && length(b) != length(y) && error(
            "sbimpl: `$wrapper` $label bound has $(length(b)) rows but response ",
            "`$target` has $(length(y))")
    end
    if check_order && !isnothing(lower) && !isnothing(upper)
        lo = _sb_composed_values(_sb_bound_data(lower, data))
        hi = _sb_composed_values(_sb_bound_data(upper, data))
        ok = if lo isa AbstractVector || hi isa AbstractVector
            all(eachindex(y)) do i
                (lo isa AbstractVector ? lo[i] : lo) <=
                    (hi isa AbstractVector ? hi[i] : hi)
            end
        else
            lo <= hi
        end
        ok || error("sbimpl: `$wrapper` lower bounds must not exceed upper bounds")
    end
    nothing
end

function _sb_validate_composed_support(wrapper, target, lower, upper, kind, data)
    y = _sb_composed_values(data[target])
    lo = isnothing(lower) ? nothing :
        _sb_composed_values(_sb_bound_data(lower, data))
    hi = isnothing(upper) ? nothing :
        _sb_composed_values(_sb_bound_data(upper, data))
    if kind === :discrete
        (eltype(y) <: Integer && !(eltype(y) <: Bool)) || error(
            "sbimpl: `$wrapper` discrete base family requires an integer response, ",
            "got $(eltype(y)) for `$target`")
        for (label, bound) in ((:lower, lower), (:upper, upper))
            isnothing(bound) && continue
            b = _sb_composed_values(_sb_bound_data(bound, data))
            all(v -> v isa Integer && !(v isa Bool), b isa AbstractVector ? b : (b,)) ||
                error("sbimpl: `$wrapper` discrete $label bounds must be integers")
        end
    end
    all(eachindex(y)) do i
        lov = lo isa AbstractVector ? lo[i] : lo
        hiv = hi isa AbstractVector ? hi[i] : hi
        (isnothing(lov) || lov <= y[i]) && (isnothing(hiv) || y[i] <= hiv)
    end || error(
        "sbimpl: `$wrapper` response `$target` contains values outside its bounds")
    nothing
end

# StanBlocks decision 1wd43wt: one base-family token plus compile-time optional
# `lower` / `upper` kwargs. Spell only PRESENT bounds. This is semantically the
# same HOF call as an explicit `nothing`, and it matters for a ragged response:
# the producer groups every supplied kwarg before resolving the HOF variant, so
# asking it to group literal `nothing` has no Stan type and cannot transpile.
_sb_composed_stan_args(base, data) = _sb_stan_dist_args(
    base.family, map(a -> _sb_scalar_expr(a, data), base.stan_args))

function _sb_emit_optional_family!(stmts, target, producer, base, lower, upper, data)
    family_args = _sb_composed_stan_args(base, data)
    bound_kwargs = Any[]
    isnothing(lower) || push!(bound_kwargs,
        Expr(:kw, :lower, _sb_scalar_expr(lower, data)))
    isnothing(upper) || push!(bound_kwargs,
        Expr(:kw, :upper, _sb_scalar_expr(upper, data)))
    rhs = Expr(:call, producer,
        Expr(:parameters, bound_kwargs...),
        base.stan_name, family_args...)
    push!(stmts, Expr(:call, :~, target, rhs))
end

function _sb_lik_composed!(stmts, target, wrapper, producer,
                           args, kwargs::NamedTuple, data)
    base = _sb_composed_family(wrapper, args)
    lower, upper = _sb_wrapper_bounds(wrapper, args, kwargs)

    isnothing(lower) && isnothing(upper) && error(
        "sbimpl: `$wrapper` needs at least one non-`nothing` bound")

    _sb_validate_bounds(wrapper, target, lower, upper, data)
    _sb_validate_composed_support(wrapper, target, lower, upper, base.kind, data)
    _sb_emit_optional_family!(stmts, target, producer, base, lower, upper, data)
end

_sb_lik_family!(stmts, target, ::typeof(truncated), args, kwargs::NamedTuple, data) =
    _sb_lik_composed!(stmts, target, :truncated, :truncated, args, kwargs, data)

_sb_lik_family!(stmts, target, ::typeof(censored), args, kwargs::NamedTuple, data) =
    _sb_lik_composed!(stmts, target, :censored, :censored, args, kwargs, data)

# Genuine interval evidence uses the observed response as the lower endpoint.
# Its producer call has no optional-bound encoding, so it is independent of the
# one-sided truncation/censoring producer decision.
function _sb_lik_family!(stmts, target, ::typeof(interval_censored),
                         args, kwargs::NamedTuple, data)
    length(args) == 1 || error(
        "sbimpl: `interval_censored` expects one base distribution argument")
    keys(kwargs) == (:upper,) || error(
        "sbimpl: `interval_censored` requires exactly the `upper` keyword; ",
        "the response column is the interval lower endpoint")
    base = _sb_composed_family(:interval_censored, args)
    upper = kwargs.upper
    _sb_validate_bounds(:interval_censored, target, data[target], upper, data;
                        check_order=false)
    lo = _sb_composed_values(data[target])
    hi = _sb_composed_values(_sb_bound_data(upper, data))
    all(eachindex(lo)) do i
        lo[i] < (hi isa AbstractVector ? hi[i] : hi)
    end || error(
        "sbimpl: `interval_censored` lower endpoints must be strictly below upper endpoints")
    _sb_validate_composed_support(:interval_censored, target, data[target],
                                  upper, base.kind, data)
    family_args = _sb_composed_stan_args(base, data)
    upper_expr = _sb_scalar_expr(upper, data)
    _sb_lik_stan_exprs!(stmts, target, :interval_censored,
                        (base.stan_name, target, upper_expr, family_args...))
end

# Distributions.jl uses scale `theta`; Stan uses inverse scale (rate) `beta`.
_sb_stan_dist_args(::Type{<:Exponential}, ::Tuple{}) = (1.0,)
_sb_stan_dist_args(::Type{<:Exponential}, args::Tuple{Any}) =
    (_sb_stan_reciprocal(args[1]),)
_sb_stan_dist_args(::Type{<:Gamma}, ::Tuple{}) = (1.0, 1.0)
_sb_stan_dist_args(::Type{<:Gamma}, args::Tuple{Any}) = (args[1], 1.0)
_sb_stan_dist_args(::Type{<:Gamma}, args::Tuple{Any,Any}) =
    (args[1], _sb_stan_reciprocal(args[2]))

_sb_stan_dist_args(::Type{<:Beta}, ::Tuple{}) = (1.0, 1.0)
_sb_stan_dist_args(::Type{<:Beta}, args::Tuple{Any}) = (args[1], args[1])
_sb_stan_dist_args(::Type{<:Uniform}, ::Tuple{}) = (0.0, 1.0)
_sb_stan_dist_args(::Type{<:LogNormal}, ::Tuple{}) = (0.0, 1.0)
_sb_stan_dist_args(::Type{<:LogNormal}, args::Tuple{Any}) = (args[1], 1.0)
_sb_stan_dist_args(::Type{<:Laplace}, ::Tuple{}) = (0.0, 1.0)
_sb_stan_dist_args(::Type{<:Laplace}, args::Tuple{Any}) = (args[1], 1.0)
_sb_stan_dist_args(::Type{<:Logistic}, ::Tuple{}) = (0.0, 1.0)
_sb_stan_dist_args(::Type{<:Logistic}, args::Tuple{Any}) = (args[1], 1.0)
_sb_stan_dist_args(::Type{<:Gumbel}, ::Tuple{}) = (0.0, 1.0)
_sb_stan_dist_args(::Type{<:Gumbel}, args::Tuple{Any}) = (args[1], 1.0)
_sb_stan_dist_args(::Type{<:Frechet}, ::Tuple{}) = (1.0, 1.0)
_sb_stan_dist_args(::Type{<:Frechet}, args::Tuple{Any}) = (args[1], 1.0)
_sb_stan_dist_args(::Type{<:Rayleigh}, ::Tuple{}) = (1.0,)
_sb_stan_dist_args(::Type{<:SkewNormal}, ::Tuple{}) = (0.0, 1.0, 0.0)
_sb_stan_dist_args(::Type{<:SkewNormal}, args::Tuple{Any}) =
    (0.0, 1.0, args[1])

# Distributions.jl orders Pareto parameters as `(shape, scale)`, while Stan
# orders them as `(minimum, shape)`.
_sb_stan_dist_args(::Type{<:Pareto}, ::Tuple{}) = (1.0, 1.0)
_sb_stan_dist_args(::Type{<:Pareto}, args::Tuple{Any}) = (1.0, args[1])
_sb_stan_dist_args(::Type{<:Pareto}, args::Tuple{Any,Any}) =
    (args[2], args[1])

# Erlang is an integer-shape Gamma in Distributions.jl and uses the same scale
# convention, so it shares Gamma's scale-to-rate translation.
_sb_stan_dist_args(::Type{<:Erlang}, ::Tuple{}) = (1.0, 1.0)
_sb_stan_dist_args(::Type{<:Erlang}, args::Tuple{Any}) = (args[1], 1.0)
_sb_stan_dist_args(::Type{<:Erlang}, args::Tuple{Any,Any}) =
    (args[1], _sb_stan_reciprocal(args[2]))

# Only the standard [0, 1] Arcsine constructor is a native Beta(1/2, 1/2).
# Shifted/scaled constructors need a Jacobian-aware custom distribution triad.
_sb_stan_dist_args(::Type{<:Arcsine}, ::Tuple{}) = (0.5, 0.5)
_sb_stan_dist_args(::Type{<:Arcsine}, args) = throw(ArgumentError(
    "sbimpl: `Arcsine` is supported only as the standard `Arcsine()` on " *
    "[0, 1]; got $(length(args)) positional arguments"))

# NormalCanon stores natural parameters `(eta, lambda)`, where
# `mu = eta / lambda` and `sigma = inv(sqrt(lambda))`.
_sb_stan_dist_args(::Type{<:NormalCanon}, ::Tuple{}) = (0.0, 1.0)
_sb_stan_dist_args(::Type{<:NormalCanon}, args::Tuple{Any,Any}) = (
    Expr(:call, Symbol("./"), args[1], args[2]),
    Expr(:call, :inv, Expr(:call, :sqrt, args[2])),
)

# Distributions.jl's asymmetric-Laplace special case keeps its own scale.
# Only an explicit literal p=1 is accepted: the general SEPD has no faithful
# native Stan analogue. All three Stan paths consume this one translation.
function _sb_stan_dist_args(
    ::Type{<:SkewedExponentialPower},
    args::Tuple{Any,Any,Any,Any},
)
    mu, sigma_sepd, p, alpha = args
    p isa Real && p == one(p) || throw(ArgumentError(
        "sbimpl: `SkewedExponentialPower` is supported only with the explicit " *
        "literal shape `p = 1`; got $(repr(p))"))
    one_minus_alpha = Expr(:call, :-, 1.0, alpha)
    stan_scale = Expr(:call, Symbol(".*"),
        Expr(:call, Symbol(".*"),
            Expr(:call, Symbol(".*"), 4.0, sigma_sepd), alpha),
        one_minus_alpha)
    (mu, stan_scale, alpha)
end
_sb_stan_dist_args(::Type{<:SkewedExponentialPower}, args) =
    throw(ArgumentError(
        "sbimpl: `SkewedExponentialPower` requires four explicit arguments " *
        "`(mu, sigma, 1, alpha)`; got $(length(args))"))

_sb_stan_dist_args(::Type{<:Weibull}, ::Tuple{}) = (1.0, 1.0)
_sb_stan_dist_args(::Type{<:Weibull}, args::Tuple{Any}) = (args[1], 1.0)
_sb_stan_dist_args(::Type{<:InverseGamma}, ::Tuple{}) = (1.0, 1.0)
_sb_stan_dist_args(::Type{<:InverseGamma}, args::Tuple{Any}) = (args[1], 1.0)
_sb_stan_dist_args(::Type{<:Bernoulli}, ::Tuple{}) = (0.5,)
_sb_stan_dist_args(::Type{<:BernoulliLogit}, ::Tuple{}) = (0.0,)
_sb_stan_dist_args(::Type{<:Binomial}, ::Tuple{}) = (1, 0.5)
_sb_stan_dist_args(::Type{<:Binomial}, args::Tuple{Any}) = (args[1], 0.5)
_sb_stan_dist_args(::Type{<:Poisson}, ::Tuple{}) = (1.0,)

_sb_stan_dist_args(::Type{<:VonMises}, args::Tuple{Any}) = (0.0, args[1])

# Distributions `Multinomial(n, p)` -> StanBlocks `multinomial(obs | probs, N)`:
# reorder to (probs, N) so the emitted `obs ~ multinomial(probs, N)` matches the
# builtin `multinomial_lpmf(obs, probs, N)`.
_sb_stan_dist_args(::Type{<:Multinomial}, args::Tuple{Any,Any}) = (args[2], args[1])

# Distributions.jl `NegativeBinomial(r, p)` counts failures before `r`
# successes.  Native Stan `neg_binomial(alpha, beta)` uses shape and inverse
# scale, with the exact translation alpha=r, beta=p/(1-p).
_sb_stan_dist_args(::Type{<:NegativeBinomial}, ::Tuple{}) = (1.0, 1.0)
_sb_stan_dist_args(::Type{<:NegativeBinomial}, args::Tuple{Any}) =
    (args[1], 1.0)
_sb_stan_dist_args(::Type{<:NegativeBinomial}, args::Tuple{Any,Any}) =
    (args[1], _sb_stan_success_odds(args[2]))

# Multinomial composition likelihood. `Multinomial(N, probs)` -> the `brm_multinomial`
# density (above), which dispatches on the Stan type of `probs`: `vector[K]` (shared
# simplex) or `matrix[nrow, K]` (per-row simplex, e.g. a scan carrier). Response `obs`
# is a per-row `int[nrow, K]` count matrix.
function _sb_lik_family!(stmts, target, ::Type{<:Multinomial}, args::Tuple{Any,Any}, data)
    N_expr = _sb_scalar_expr(args[1], data)
    probs_expr = _sb_scalar_expr(args[2], data)
    _sb_lik_stan_exprs!(stmts, target, :brm_multinomial, (probs_expr, N_expr))
end

# Default: look up the Stan name from the table and emit
# `target ~ <stan-name>(<lowered-args>...)` via `_sb_lik_stan!`. Families
# that need bespoke handling (custom args, side effects, mangled
# arities) override on a more-specific signature -- see OrderedLogistic
# above.
_sb_lik_family!(stmts, target, ::Type{D}, args, data) where {D <: Distribution} =
    let stan_name = _sb_stan_dist_name(D)
        isnothing(stan_name) && error(
            "sbimpl: likelihood family `$(D)` not supported yet -- ",
            "no `_sb_stan_dist_name` entry. Add one (and a `_sb_stan_dist_args` ",
            "override if its args don't lower verbatim).")
        arg_exprs = map(a -> _sb_scalar_expr(a, data), args)
        _sb_lik_stan_exprs!(
            stmts, target, stan_name, _sb_stan_dist_args(D, arg_exprs))
    end

_sb_lik_family!(stmts, target, fam, args, _) =
    error("sbimpl: likelihood family `$fam` (arity $(length(args))) not supported yet")


# ---- scalar-expression reducer (unwraps NamedColumn references etc.) --------

_sb_scalar_expr(x::Symbol, _) = x
_sb_scalar_expr(x::Real, _) = x
_sb_scalar_expr(x::NamedColumn, data) = begin
    _record_scalar_data!(data, name(x), parent(x))
    name(x)
end
_record_scalar_data!(data, sym, d::DataColumn) = (data[sym] = parent(d); nothing)
_record_scalar_data!(args...) = nothing
# Formula arithmetic is element-wise by intent -- `loc = loc_loc + loc_slope * cdslope`
# on two length-n vectors means Stan's `.*`, not matrix/dot product. Translate `*`
# and `/` to their dotted variants so Stan's typechecker accepts vector-vector
# operands (and scalar operands broadcast correctly in either form). Addition /
# subtraction already element-wise-broadcast in Stan between vectors, no change
# needed there.
_sb_scalar_expr(x::ExprColumn, data) = begin
    f = getf(x)
    # Body-level `state.field` / `state[i]` on a model value (macro.jl getproperty /
    # getindex) render as Stan `.` / `[` so a `@deffun` struct/array return can be
    # consumed directly — `state = scan(...); y ~ f(state.population_total)`.
    if f === getproperty
        obj, field = getargs(x)      # field is a QuoteNode(:name)
        return Expr(:., _sb_scalar_expr(obj, data), field)
    elseif f === getindex
        obj = first(getargs(x))
        idxs = getargs(x)[2:end]
        return Expr(:ref, _sb_scalar_expr(obj, data),
                    (_sb_scalar_expr(a, data) for a in idxs)...)
    end
    op = f === (*) ? Symbol(".*") :
         f === (/) ? Symbol("./") :
         f
    Expr(:call, op, (_sb_scalar_expr(a, data) for a in getargs(x))...)
end
_sb_scalar_expr(x, _) = error("sbimpl: cannot lift to Stan expression: $(typeof(x)): $x")

# ==============================================================================
# Hierarchical biomarker family — BRM-side composition surface.
#
# Representative model formula (log_y MUST precede log_obs so sigma is in scope):
#
#   brmi = @brm df begin
#       log_y ~ biomarker_hierarchical_parametric(;
#           time, dose, person_idxs, biomarker_idxs,
#           sigma_rate, scale_rate, hierarchical_centeredness
#       )
#       log_obs ~ TruncatedNormal(log_y, sigma, lloq, uloq)
#   end
#
# Cross-tree contract (StanBlocks biomarker kernel builtins, decision 1lmystf):
#   - `truncated_normal` lpxf triad: censored normal (LLOQ→lcdf, ULOQ→lccdf)
#   - `biomarker_time_response(log_time, loc, log_slope, mag)::vector` bell kernel
#   - `biomarker_dose_response(log_dose, loc, log_slope)::vector` — returns LOG
#   - Transdata builtins: `linear_idxs`, `broadcasted_max`, `broadcasted_gt`
#   All registered in StanBlocks' builtin module; reachable with no import.
# ==============================================================================

"""
    biomarker_hierarchical_parametric

BRM formula marker for the hierarchical biomarker parametric mean submodel.
All data columns and hyperparameters are passed as keyword arguments.
Emits (in order): sizes (n_biomarkers, n_persons, n_series), sigma prior
(biomarker-level exponential), transdata (series_idxs, log_time, log_dose,
affectable), 6 per-series parameter priors, and log_y via StanBlocks
biomarker kernel builtins (biomarker_time_response + biomarker_dose_response).
"""
function biomarker_hierarchical_parametric end

# TruncatedNormal custom likelihood.
# Emits: target ~ truncated_normal(mean, sigma[biomarker_idxs], lloq[biomarker_idxs], uloq[biomarker_idxs])
# biomarker_idxs is in scope because biomarker_hierarchical_parametric emits it as data.
function _sb_lik_family!(stmts, target, ::typeof(TruncatedNormal), args, data)
    length(args) == 4 || error(
        "sbimpl: TruncatedNormal likelihood expects 4 positional args ",
        "(mean, sigma, lloq, uloq), got $(length(args))")
    mean_arg, sigma_arg, lloq_arg, uloq_arg = args
    mean_sym  = _sb_scalar_expr(mean_arg, data)
    sigma_sym = _sb_scalar_expr(sigma_arg, data)
    lloq_sym  = _sb_scalar_expr(lloq_arg, data)
    uloq_sym  = _sb_scalar_expr(uloq_arg, data)
    push!(stmts, :($target ~ truncated_normal(
        $mean_sym,
        $(sigma_sym)[biomarker_idxs],
        $(lloq_sym)[biomarker_idxs],
        $(uloq_sym)[biomarker_idxs])))
end

# Group-block declaration for biomarker_hierarchical_parametric.
# The 6 per-(biomarker×person) params (baseline, time_loc, time_log_slope,
# time_mag, dose_loc, dose_log_slope) are correlated normals via BRM's
# ranef_correlated_draws floor. The group is series_idxs = linear_idxs(bm, pn)
# — a computed column, so group_fn synthesises it from the kwargs in data.
_sb_term_group_block(::typeof(biomarker_hierarchical_parametric)) = (;
    n_per_group  = 6,
    group_fn     = (rhs_e, data) -> begin
        bm   = data[:biomarker_idxs]
        pn   = data[:person_idxs]
        n_bm = maximum(bm)
        series = bm .+ (pn .- 1) .* n_bm
        NamedColumn(:series_idxs, DataColumn(series))
    end,
    group_fn_name = :series_idxs,
)

# Emit hook: allocates sigma, transdata, and log_y from the group-block matrix.
# block_info = (; block_name, idx_name, n_per_group)
#   block_name  — n_series × 6 matrix from ranef_correlated_draws (col order
#                 matches the param order below)
#   idx_name    — per-obs integer index into block rows (= series_idxs_idx)
function _sb_emit_group_block_term!(stmts, data, target,
                                     ::typeof(biomarker_hierarchical_parametric),
                                     rhs_e, block_info)
    (; block_name, idx_name) = block_info
    # Sizes: deterministic transdata from int[] index arrays
    push!(stmts, :(n_biomarkers = max(biomarker_idxs)))
    push!(stmts, :(n_persons    = max(person_idxs)))
    push!(stmts, :(n_series     = n_biomarkers * n_persons))
    # sigma: biomarker-level exponential prior
    push!(stmts, :(sigma ~ exponential(sigma_rate; n=n_biomarkers, lower=0.)))
    # Derived transdata: obs-level index + input transforms
    push!(stmts, :(series_idxs = linear_idxs(biomarker_idxs, person_idxs)))
    push!(stmts, :(log_time    = log(broadcasted_max(time, 0.001))))
    push!(stmts, :(log_dose    = log(broadcasted_max(dose, 0.1))))
    push!(stmts, :(affectable  = broadcasted_gt(time .* dose, 0.0)))
    # Extract 6 per-series params from the group-block matrix (row = series, col = param)
    push!(stmts, :(baseline       = $(block_name)[$(idx_name), 1]))
    push!(stmts, :(time_loc       = $(block_name)[$(idx_name), 2]))
    push!(stmts, :(time_log_slope = $(block_name)[$(idx_name), 3]))
    push!(stmts, :(time_mag       = $(block_name)[$(idx_name), 4]))
    push!(stmts, :(dose_loc       = $(block_name)[$(idx_name), 5]))
    push!(stmts, :(dose_log_slope = $(block_name)[$(idx_name), 6]))
    # log_y via StanBlocks biomarker kernel builtins
    push!(stmts, :(time_response = biomarker_time_response(log_time, time_loc, time_log_slope, time_mag)))
    push!(stmts, :(dose_response = exp(biomarker_dose_response(log_dose, dose_loc, dose_log_slope))))
    push!(stmts, :($target = baseline + affectable .* time_response .* dose_response))
end
