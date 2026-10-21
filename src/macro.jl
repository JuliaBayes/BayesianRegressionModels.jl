using OrderedCollections

"""
    @brm formula_block
    @brm df formula_block

Parse a brms-style formula block into a [`BRMI`](@ref) (BRM Intermediate).

The one-argument form returns a function `model(df) -> BRMI` (the model
name is gensym-ed). The two-argument form bakes `df` in and returns the
`BRMI` directly. Inside `formula_block`, each line is one of:

- `lhs ~ rhs` — a sampling statement (likelihood or linear predictor).
- `(x, y, z) ~ rhs` — broadcast one RHS over several sampling LHSs, sugar
  for repeating the `~` row once per name. Each slot takes one ordinary
  sampling LHS; `[y1, y2] ~ ...` stays the joint multivariate response.
- `lhs = rhs` — a literal binding (named intermediate).
- `@plate for i in eachindex(x) ... end` — a group-local kernel cell. Reads
  such as `x[i]` are sliced per group, `result[i] = ...` declares a
  model-scope output, and bare assignments remain cell locals.
- `ragged(events, subject)` — a model-scope statement declaring that the flat
  data column or formula name `events` is grouped by `subject`; `@plate for`
  cells then read it as `events[i]`. `ragged(events, subject => cells)` also
  names the cell label column `cells` that lists the plate's cells when no
  random-effect predictor labels them.
- `ragged(y, group) ~ rhs` — group a flat observed response at the formula
  boundary, aligned to the `kernel(...)` result referenced by `rhs`.
- `effect(lp, coefficient) ~ Normal(location, scale)` — an SBBRMI
  population-coefficient prior override.
- `sd(lp, ID[, coefficient]) ~ Exponential(scale)` and
  `cor(:, ID) ~ LKJCholesky(K, eta)` — priors for a shared `|ID|`
  random-effect block; a block written without an ID is addressed by its
  grouping column (`sd(lp, g)`, `cor(lp, g)`, see [`effect`](@ref)).

Multiple `~` lines produce a multi-response / distributional model.
Inside an observed family's argument expression, nested `@brm(expr)` is the
explicit opt-in for a coefficient-bearing predictor formula. Unmarked family
arguments remain ordinary expressions.

```julia
brmi = @brm df begin
    y ~ Normal(loc, err)
    loc ~ 1 + age + (1 | subj)
    err ~ Exponential(1)
end
```

The resulting `BRMI` feeds either backend: `VBRMI(brmi)` (vectorised
Julia) or `SBBRMI(brmi)` (StanBlocks → Stan).
"""
macro brm(x)
    esc(_brm(x))
end
macro brm(df, x)
    esc(_brm(x; df=df))
end

"""
    @n lhs = rhs

Lower a parsed-formula LHS into a [`NamedColumn`](@ref) binding. Used
internally by [`@brm`](@ref); not normally called by hand.
"""
macro n(x)
    esc(_n(x))
end

"""
    @x expr

Lower a parsed-formula RHS into the column-type tree
([`ExprColumn`](@ref) / [`NamedColumn`](@ref)). Used internally by
[`@brm`](@ref); not normally called by hand.
"""
macro x(x)
    esc(_x(x))
end

"""
    @getproperty obj.field

Expands to `hasproperty(obj, :field) ? obj.field : :field` — used by the
`@brm` `df`-baked path to fall back from "column from the dataframe" to
"bare symbol" without erroring on missing columns.
"""
macro getproperty(x)
    esc(_getproperty(x))
end
_getproperty(x::Expr) = begin 
    @assert x.head == :(.)
    @assert length(x.args) == 2
    lhs, qrhs = x.args
    :(hasproperty($lhs, $qrhs) ? $x : $NamedColumn($(Meta.quot(qrhs.value)), $MissingColumn()))
end
begin
isxcall(x, f) = Meta.isexpr(x, :call) && x.args[1] == f
fixcall(x) = x
fixcall(x::Expr) = if Meta.isexpr(x, :call)
    f = x.args[1]
    pargs = []
    args = []
    for arg in fixcall.(x.args[2:end])
        if Meta.isexpr(arg, :parameters)
            append!(pargs, arg.args)
        else
            push!(args, arg)
        end
    end
    if length(pargs) > 0
        Expr(x.head, f, Expr(:parameters, pargs...), args...)
    else
        Expr(x.head, f, args...)
    end
else
    Expr(x.head, fixcall.(x.args)...)
end
"""Marker function for literal-binding (`lhs = rhs`) operations in a `@brm` block. Dispatch tag only — never called."""
function assign end

"""
    effect(linear_predictor, coefficient)
    sd(linear_predictor, id[, coefficient])
    cor(:, id)
    cor(linear_predictor, group)

Address a prior BRM owns, by naming the **parameter** as the head and the
target in the slots. Every slot is one name or `:`, and `:` means *the
default* — the base layer that a more specific statement overrides:

```julia
@brm begin
    log_ka ~ 1 + weight + (1 | pk | subject)
    effect(log_ka, Intercept) ~ Normal(log(1 / 8), 0.8)
    effect(log_ka, weight)    ~ Normal(0, 0.1)
    sd(:, pk)                 ~ Exponential(2 / 3)
    cor(:, pk)                ~ LKJCholesky(2, 2)
end
```

The grammar is `<quantity>(<linear predictor | :>, <target…>)`. `effect`
addresses a population coefficient, a categorical column's `K-1` treatment
contrasts, or — for the first categorical term of a predictor without an
intercept — its `K` cell means, each also addressable as `<column>_lvl_<k>`; `sd` and `cor` address a shared `|ID|` random-effect covariance
block. `sd`'s trailing coefficient slot may be omitted and means `:`, so
`sd(:, pk)` is the block-wide scale and `sd(log_ka, pk, weight)` is one
margin, while `sd(log_ka, pk)` sets every margin `log_ka` contributes.
Because `:` is positional rather than a trailing omission,
`sd(:, pk, weight)` addresses the `weight` margin across *every* predictor
slicing the block. Correlation priors on a shared `|ID|` are block-wide by
construction — one covariance spans every predictor that slices it — so
`cor(:, ID)` takes `:` in the predictor slot.

A random-effect block written without an ID — plain `(… | g)` or uncorrelated
`(… || g)` — is addressed by its grouping column, as brms' `group =` is:

```julia
@brm begin
    mu ~ 1 + c1 + (1 + c1 || probe)
    log(sigma) ~ 1 + (1 | probe)
    sd(:, probe)     ~ Exponential(50)   # every margin grouped by `probe`
    sd(sigma, probe) ~ Exponential(1)    # more specific: `sigma`'s block
    sd(mu, probe, c1) = 0.5              # a known value
end
```

One column can group several independent blocks — each predictor's block and
one block per `||` term — and an `sd` statement sets every margin of them it
matches, most specific first. `cor(:, g)` sets the correlation of the one
correlated block grouped by `g`; with one per predictor, write
`cor(<linear_predictor>, g)`. `ranef(<linear_predictor|:>, g) ~ TDist(nu)` sets
the law of the matching blocks. An addressed block lowers as a one-predictor
`|ID|` block under a generated ID (`|lp__g|`, the `k`-th `||` term
`|lp__g__nocor__k|`); a block no statement reaches is left as written. An `|ID|`
block keeps its ID address only, and a name that is both an `|ID|` and the
grouping column of a block without one is refused as ambiguous.

An interaction term is addressed the way the formula spells it:
`effect(mu, gen & mathz)` sets one shared `Normal` prior over every
`beta_pop` column the `gen & mathz` term emits — one column for a
continuous × continuous interaction, `K-1` for a continuous × categorical
one. Operand order does not matter (`a & b` and `b & a` are the same
address). Each emitted column keeps its own `int_…` label (see
[`popcoefnames`](@ref)), and addressing one of those labels directly refines
the whole-term address on that column alone.

There is deliberately no concise, predictor-inferring form: name the linear
predictor or write `:`. Two statements reaching the same parameter resolve
most-specific-wins — fewer `:` slots wins — and an exact tie is an error.

The whole-predictor `Colon` address is also how the joint
variance-decomposition family [`r2d2`](@ref) attaches: `effect(lp, :) ~
r2d2(...)` decomposes every population coefficient of `lp` at once, while
`effect(lp, :) ~ Normal(...)` is simply the default layer for those columns.

`effect` is an address marker, not a callable function; `sd` and `cor` are
rewritten onto it by the macro and are not exported. Use
[`effect_priors`](@ref), [`ranef_effect_priors`](@ref), [`r2d2_priors`](@ref),
and [`ranefcoefnames`](@ref) to inspect the captured statements and margin
labels.
"""
function effect end

"""
    r2d2(; R2=Beta(1, 1), tau_bsv=nothing, alpha=1)

R²-induced Dirichlet Decomposition prior for a whole linear predictor, applied
through a [`effect`](@ref) `Colon` address:

```julia
@brm begin
    log_CL ~ 1 + weight + age + (1 | p | subject)
    effect(log_CL, :) ~ r2d2(R2 = Beta(1, 1), tau_bsv = 0.5)
end
```

One total between-subject scale `tau_bsv` is split by `R2` into an explained
and a residual part. The explained part is allocated across the predictor's
population columns by a Dirichlet simplex `phi ~ Dirichlet(alpha)`, giving
column `k` the derived prior scale

```
beta_scale[k] = sqrt(phi[k] * R2 * tau_bsv^2 / Var(x_k))
```

while the predictor's random-effect margins take the residual scale
`sqrt((1 - R2) * tau_bsv^2)` — for a latent per-subject parameter the random
effect *is* the residual, which is the motivating PK reading (decision
`kx8wkd`).

Keywords:

- `R2` — a `Beta` prior on the explained fraction. Defaults to `Beta(1, 1)`.
- `tau_bsv` — the total scale. A positive number fixes it; omitted, it becomes
  a sampled half-standard-normal parameter, which is the only option for a
  latent predictor with no observed response to anchor it.
- `alpha` — the symmetric Dirichlet concentration over population columns.

Within a shared brms-style `|ID|` bucket the decomposition is all-or-nothing:
if any linear predictor slicing the bucket is `r2d2`-scoped, all of them must
be, so the bucket's `tau` is one wholly derived vector rather than a
part-sampled hybrid (decision `1db6zkr`). The correlation factor `L` stays free
and shared — `r2d2` constrains marginal variances and says nothing about
cross-predictor correlation.

`r2d2` is a formula marker, not a callable function; the SBBRMI backend lowers
it. Inspect captured statements with [`r2d2_priors`](@ref).

The same marker on a random-effect SD address is the partial-R2D2M2 form:

```julia
@brm begin
    sigma_pk ~ Exponential(1)
    sigma_qt ~ Exponential(1)
    log_CL   ~ 1 + (1 | p | subject)
    qt_base  ~ 1 + (1 | p | subject)

    sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=2,
                    concentration=1, reference_scale=sigma_pk)
    sd(qt_base, p) ~ r2d2(reference_scale=sigma_qt)
    cor(:, p) ~ LKJCholesky(2, 2)
end
```

This spelling samples one block-wide `R2`, allocates its odds
`R2 / (1 - R2)` across the addressed marginal variances with a Dirichlet
simplex, and derives margin `j` as
`tau[j] = reference_scale[j] * sqrt(phi[j] * R2 / (1 - R2))`. `R2=Beta(a,b)`
and the equivalent mean/precision spelling are both accepted; `alpha=` and
`concentration=` are aliases. A more-specific `sd(lp, id[, coefficient])`
statement under a block-wide decomposition may override only
`reference_scale`, which is how one correlated block can span observation
channels with different residual units. `cor(:, id)` remains an independent
LKJ prior and the non-centred draw geometry is unchanged.

Without a block-wide `sd(:, id)` statement, every R2D2 SD address owns its own
R2 and simplex. A one-margin address has a deterministic `simplex[1]` and is
therefore the per-margin ICC construction. Unaddressed margins retain their
ordinary half-standard-Normal scale prior. Random-effect R2D2 is SBBRMI-only;
centered and stratified buckets remain unsupported, while non-centred
`resample_groups` replay redraws only the standardised group coordinates and
transports the fitted R2, simplex, reference scales, and correlation factor.

Adding `include=` to the block-wide statement makes it the JOINT R2D2M2 budget:
the block's one `R2` and one Dirichlet simplex also allocate population
components of every linear predictor slicing the block —

```julia
@brm begin
    sigma_pk ~ Exponential(1)
    sigma_qt ~ Exponential(1)
    log_CL  ~ 1 + wt + indication + (1 | p | subject)
    qt_base ~ 1 + wt + indication + (1 | p | subject)

    sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=2, concentration=1,
                    reference_scale=sigma_pk,
                    include=(:population, :contrasts))
    sd(qt_base, p) ~ r2d2(reference_scale=sigma_qt)
    cor(:, p) ~ LKJCholesky(2, 2)
end
```

`:population` adds each predictor's non-intercept `beta_pop` columns,
`:contrasts` its categorical treatment-contrast coefficients; `:ranef` names
the margins, which are always allocated. A component of predictor `m` is scaled
by `m`'s own margin reference: a margin keeps
`tau[j] = reference_scale[m] * sqrt(phi[j] * R2 / (1 - R2))` and a coefficient
or contrast takes
`reference_scale[m] * sqrt(phi[k] * R2 / ((1 - R2) * Var(x_k)))`, the
design-column variance adjustment of the whole-predictor form (a contrast's
dummy variance is its level frequency's `p * (1 - p)`). Intercepts stay
outside. With `include=`, `reference_scale=` is optional: an omitted margin
reference is a sampled half-standard-normal, so the statement then allocates
LATENT between-group variation rather than an observed-outcome R². Per-column
`effect(lp, coef) ~ Normal(...)` and `effect(lp, categorical) ~ Normal(...)`
statements inside the scope are refused, as is a scoped predictor that also
carries `effect(lp, :) ~ r2d2(...)`; `include=` belongs on the block-wide
statement only. The emitted carriers (`pop_<lp>_beta_pop`,
`cat_<lp>_<col>_beta`, the block's derived `tau`) and every descriptor reader
are unchanged.
"""
function r2d2 end

"""Marker function for the brms `||` zero-correlation random-effects operator. Dispatch tag only — never called."""
function doublepipe end

"""
    weighted(distribution, weights)

Formula marker for a typed observation weight. The second argument is a
StatsBase weight constructor over a dataframe column, for example
`aweights(replicate_k)`, `fweights(repeats)`, or `weights(power)`.

The weight type is semantic. Analytic weights modify a supported observation
distribution's precision (currently `Normal`); frequency and generic weights
scale model and pointwise log-likelihood contributions while preserving the
base distribution's predictive RNG. Lowering is implemented by the StanBlocks
backend; this marker is never called directly.
"""
function weighted end

"""
    interval_censored(base; upper)
    interval_censored(x; upper, lower=0)

Formula marker for genuine interval evidence. On a response RHS, the observed
response column supplies each interval's lower endpoint and `upper` supplies
its upper endpoint:

```julia
y_lower ~ interval_censored(Normal(mu, sigma); upper=y_upper)
```

Each row contributes the base family's probability over `(y_lower, y_upper]`,
exactly `CDF(y_upper) - CDF(y_lower)`. The lower endpoint is open for discrete
families too; unlike inclusive truncation it receives no predecessor shift.
Posterior prediction remains on the uncoarsened base-response scale. This is
separate from Distributions.jl's `censored`, which is the distribution of
`clamp(X, lower, upper)` and therefore has atoms at its thresholds.

Inside a linear predictor, `x` is the observed covariate and `upper` is its
row-specific quantification limit:

```julia
mu ~ 1 + interval_censored(concentration; upper=lloq)
```

By convention, `concentration == lloq` marks a BLOQ row; values above `lloq`
are exact. BLOQ rows allocate latent predictor values between `lower` (zero by
default) and `lloq` with a truncated Normal prior; configure that
prior with `latent(<lp|:>, interval_censored(concentration)) ~ Normal(...)`.
This predictor form is implemented by `SBBRMI`.

The marker is intentionally formula-local: unlike `truncated` and `censored`,
there is no existing Distributions.jl value with these per-row evidence
semantics for BRM to construct outside `@brm`. It lives in the backend-neutral
formula layer so weak-dependency backends can inspect it without loading
StanBlocks.
"""
function interval_censored end

"""
    gr(group; by=strata)

brms-style stratified grouping marker. `(1 | gr(subj, by=diagnosis))`
gives each level of `by` its own random-effect covariance structure.
Dispatch tag only — backend interpretation lives in `vmeta_sampling_rhs`
(vimpl) / the sbimpl ranef walker.
"""
function gr end

"""
    mm(group1, group2, ...; weights=nothing, normalize=true)

Multi-membership grouping marker for random effects. Each observation belongs
to two or more grouping levels and receives the weighted sum of their shared
random-effect coefficients. `weights` is either omitted (equal `1/M` weights)
or a tuple of one data column per group. Supplied weights must be finite,
nonnegative, and have a positive row total; by default each row is normalized
to sum to one. Set `normalize=false` to preserve valid supplied magnitudes.

`@brm` lowers this surface immediately to a typed
`MultiMembershipTerm`; the Stan backend validates and materializes it, including
when data are replayed through [`reprocess`](@ref).
"""
function mm end

"""
    gp(x...; cov=:exp_quad, iso=true, jitter=1e-9)
    gp(x; cov=:periodic, period, jitter=1e-9)

Exact latent Gaussian-process predictor for the StanBlocks backend. Accepts
one-or-more real-valued axes, with an isotropic squared-exponential kernel by
default; set `iso=false` for one length scale per axis. `cov=:periodic`
selects Stan's periodic kernel `sigma^2 exp(-2 sin^2(pi |x - x'| / period) /
rho^2)` over exactly one axis and requires the numeric formula constant
`period`. `jitter` stabilizes the covariance Cholesky factor. Dispatch tag —
lowering lives in `_sb_gp` / `_sb_gp_aniso` / `_sb_gp_periodic` (sbimpl).
"""
function gp end

"""
    hsgp(x...; k=20, c=1.5, cov=:exp_quad, iso=true, by=nothing,
         domain=nothing, orthogonal_to=nothing)
    hsgp(x; k=20, cov=:periodic, period)

Hilbert-space approximate Gaussian-process predictor. The StanBlocks backend
supports variadic axes, per-axis `k`/`c` tuples, isotropic or anisotropic length
scales, and optional group-specific basis weights via `by=`. For raw data axes,
`domain=(lower, upper)` fixes the approximation interval instead of deriving it
through `c`. A one-dimensional model-derived axis, such as the sampled `x` from
`log(x) ~ ...`, requires that explicit domain and evaluates its basis inside
Stan. `orthogonal_to=:linear` removes the intercept and current linear-`x`
directions from a one-dimensional basis, so `x + hsgp(x; ...)` separates the
linear slope from residual nonlinear shape.

`cov=:periodic` selects the periodic-kernel approximation of Riutort-Mayol et
al. (2023) over exactly one raw-data axis: `k` harmonics of `2pi / period`
give `2k` cosine/sine basis functions whose spectral weights come from the
kernel's modified-Bessel expansion. It needs no boundary factor, so `c`,
`domain`, `by`, `orthogonal_to`, and `iso=false` are refused, and the numeric
formula constant `period` is required. In formulas this method is used only
as a dispatch tag.
"""
hsgp(args...; kwargs...) = error("hsgp is a formula marker and cannot be called directly")

"""
    offset(x)

Fixed-slope (no beta) contribution to a linear predictor. brms-style
`offset(...)` — adds `x` directly to the predictor without allocating
any parameters. Dispatch tag only.
"""
function offset end

"""Marker for `zscale(x)` — z-standardise `x` (subtract mean, divide by SD). Dispatch tag only."""
function zscale end

"""Marker for `center(x)` — mean-centre `x`. Dispatch tag only."""
function center end

"""Marker for `standardize(x)` — alias-style mean/SD standardisation. Dispatch tag only."""
function standardize end

"""
    name = constant(expr)

A scalar computed from the data when the model is built, such as a prior
hyperparameter calibrated on the observations:

```julia
@brm begin
    ec50_loc = constant(mean(log.(conc)))
    log_ec50 ~ Normal(ec50_loc, 1)
    ...
end
```

`expr` is ordinary Julia, evaluated once on whole columns of the data. A bare
name in it that is not a field of the data resolves in the builder's scope.
The result must be one finite real number. The constant is Stan data on
`SBBRMI` and a constant on `RKBRMI`, usable wherever a scalar data field is.

Replay keeps the fitted value, like `standardize`'s mean and SD: `reprocess`,
`reprocess(...; resample_groups)` and `generative_plan(plan, new_df)` reuse it
without the new data supplying it, and `reprocess(...; freeze_constants=false)`
evaluates `expr` again on the new data. Dispatch tag — see
`_brm_resolve_constants`.
"""
function constant end

"""
    protect(x)

brms `I(...)` analogue: literal-escape wrapper around an expression so
the formula parser doesn't try to re-interpret it. In formula-eval
contexts it's a no-op on `Real` — generic broadcast/materialise paths
handle `protect(expr)` without a dedicated dispatch.
"""
function protect end
protect(x::Real) = x

"""
    factor(x; ref=k, cmc=true)

Treat `x` as a categorical predictor with optional reference level
`ref`. Dispatch tag — backend buckets it into the categorical-predictor
path in `introspection.jl` and `_sb_cat` (sbimpl).

Under an intercept the term is K−1 treatment contrasts against `ref`. As the
first categorical term of a predictor WITHOUT an intercept it is cell-mean coded
(K coefficients, no reference) — brms / R `model.matrix` semantics. `cmc=false`
(brms' name, "cell-mean coding") switches that off and keeps the treatment
contrasts, e.g. to pin one group of an intercept-free predictor at zero.
"""
function factor end

"""
    mi(y)

Missing-data marker. Wrap a response LHS (`mi(y) ~ Normal(...)`) to opt
into the brms-style observed/imputed split: observed rows feed the
likelihood, missing rows become parameters drawn from the same family.
For a correlated continuous block, use
`mi([x, z]) ~ MvNormalCholesky([x_loc, z_loc], L)` with an earlier
`L ~ LKJCovarianceFactor(2; ...)` declaration. Rows may have different
missing patterns; later formulas read completed `x` and `z` columns.
See `_sb_emit_mi!` (sbimpl) for backend dispatch.
"""
function mi end

# Structural formula marker shared by the parser and sbimpl. Its full public
# contract is documented beside the lowering in `sbimpl.jl`.
function ragged end

"""
    LKJCovarianceFactor(K; scale_prior=Exponential(1), shape=1)

Formula-only composite prior for the Cholesky factor of a `K`-dimensional
covariance matrix. The StanBlocks backend samples positive marginal scales and
an LKJ correlation Cholesky factor, then binds
`diag_pre_multiply(scales, L_corr)` to the declaration's left-hand side.

Use the result with [`MvNormalCholesky`](@ref). This is a formula marker rather
than a Distributions.jl constructor.
"""
function LKJCovarianceFactor end

"""
    MvNormalCholesky(mean, factor)

Formula-only multivariate Normal family whose second argument is already the
lower Cholesky factor of the covariance. A vector response such as
`[y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)` is one row-wise joint
likelihood, not two conditionally independent scalar likelihoods.
Wrap the vector LHS in `mi(...)` for explicit joint missing-value completion.
"""
function MvNormalCholesky end

"""
    BRMBuilder

What `@brm begin … end` returns when no data is given: a model builder. Call it
on a data frame, `builder(df)`, to get the [`BRMI`](@ref); every API that takes
a builder (`brm_descriptor(builder, df)`, `generative_plan(builder, df)`, …)
accepts it unchanged. It keeps the authored source, so a model can be
inspected and composed before any data exists:

- [`brm_columns(builder)`](@ref brm_columns(::BRMBuilder)) lists the data
  columns the model reads;
- `merge(a, b, …)` concatenates the fragments' sources into one builder;
- `show(builder)` prints the source.

`Core.eval(mod, _brm(expr))` returns the same object for a source `Expr`, so a
model family can be assembled from interpolated `Expr` fragments.
"""
struct BRMBuilder{F} <: Function
    f::F
    source::Expr
    columns::Tuple{Vararg{Symbol}}
    scope::Module
end
# A merged builder's function is defined by `Core.eval`; calling it through
# `invokelatest` keeps it usable from code compiled before the merge.
(b::BRMBuilder)(df) = Base.invokelatest(b.f, df)

"""
    brm_columns(builder::BRMBuilder) -> Tuple{Vararg{Symbol}}

The data columns a model reads, sorted, without any data: predictors, grouping
factors, `@plate` cell reads and the columns they capture, `ragged(...)`
groupings, bound and weight columns, imputed covariates, the inputs of
`name = constant(expr)` that are not Julia values in the builder's scope, and
observed responses. A `~` statement whose name no other statement reads is an
observed response; one that another statement reads is a parameter or
predictor. So for a model that declares every parameter it reads
(`sigma ~ …`), the result equals `brm_columns(brm_descriptor(builder, df))`,
except that a column only a `constant(...)` reads is listed here but not in
the descriptor's replay schema, because replay keeps the fitted constant. A
name read but declared nowhere is listed as a column.
"""
brm_columns(b::BRMBuilder) = b.columns

"""
    merge(a::BRMBuilder, b::BRMBuilder, rest::BRMBuilder...) -> BRMBuilder

One builder whose source is the fragments' statements in order, exactly as if
they had been written in one `@brm` block: a fragment may read names another
defines, and a `@plate` cell may read a `ragged(...)` grouping declared in
another fragment. A statement repeated across fragments is a duplicate, as in
one block. To replace a built model's statements by address, `merge` the
built [`BRMI`](@ref)s instead.
"""
function Base.merge(a::BRMBuilder, b::BRMBuilder, rest::BRMBuilder...)
    builders = (a, b, rest...)
    scope = a.scope
    all(x -> x.scope === scope, builders) || error(
        "merge: the builders were defined in different modules " *
        "($(join(unique(string(x.scope) for x in builders), ", "))); names " *
        "their statements call would resolve in only one of them. Define the " *
        "fragments in one module.")
    source = Expr(:block, (stmt for x in builders for stmt in x.source.args)...)
    Core.eval(scope, _brm(source))
end

Base.show(io::IO, b::BRMBuilder) = begin
    for (i, stmt) in enumerate(b.source.args)
        i == 1 || print(io, "\n")
        _brm_print_source(io, stmt)
    end
end
Base.show(io::IO, ::MIME"text/plain", b::BRMBuilder) = show(io, b)

# The names `parselocals!` classifies while parsing one block, for the
# data-free column inventory (`brm_columns(::BRMBuilder)`):
# `reads` every name a right-hand side reads, `responses` every `~` left-hand
# name, `assigned` every `=` left-hand name, `data` names that are always data
# (an imputed `mi(z)`, joint responses), `latent` `~` targets that are model
# values whatever reads them (kernel cell outputs), `cell_bound` the parameters
# and locals of kernel cell bodies, `constant_reads` the inputs of
# `name = constant(expr)` statements.
_brm_inventory() = (; reads=Set{Symbol}(), responses=Set{Symbol}(),
                      assigned=Set{Symbol}(), data=Set{Symbol}(),
                      latent=Set{Symbol}(), cell_bound=Set{Symbol}(),
                      constant_reads=Set{Symbol}())

function _brm_inventory_statement!(inventory, lhs, rhs, legacy_kernel_doblock)
    isxcall(lhs, :mi) && length(lhs.args) == 2 && lhs.args[2] isa Symbol &&
        push!(inventory.data, lhs.args[2])
    (isxcall(rhs, :kernel) || legacy_kernel_doblock) && lhs isa Symbol &&
        push!(inventory.latent, lhs)
    inventory
end

function _brm_cell_bound_names!(acc, lambda::Expr)
    _brm_bound_symbols!(acc, lambda.args[1])
    body = lambda.args[2]
    for stmt in (Meta.isexpr(body, :block) ? body.args : Any[body])
        Meta.isexpr(stmt, :(=), 2) && _brm_bound_symbols!(acc, stmt.args[1])
    end
    _brm_scan_bound_names!(acc, body)
    acc
end

# The names a cell's `@scan` blocks bind: each loop variable, and every array or
# local a seed or step writes (`h[1] = …`, `h[t] ~ …`, `z ~ …`). They are scan
# locals, never data columns, even when a column happens to share the name.
function _brm_scan_bound_names!(acc, body)
    for stmt in (Meta.isexpr(body, :block) ? body.args : Any[body])
        _is_brm_scan(stmt) && _brm_scan_block_bound!(acc, stmt.args[end])
    end
    acc
end
function _brm_scan_block_bound!(acc, block)
    for stmt in (Meta.isexpr(block, :block) ? block.args : Any[block])
        if Meta.isexpr(stmt, :for)
            Meta.isexpr(stmt.args[1], :(=)) && _brm_bound_symbols!(acc, stmt.args[1].args[1])
            _brm_scan_block_bound!(acc, stmt.args[2])
        elseif Meta.isexpr(stmt, :(=), 2) || isxcall(stmt, :~)
            lhs = Meta.isexpr(stmt, :(=)) ? stmt.args[1] : stmt.args[2]
            Meta.isexpr(lhs, :ref) ? _brm_bound_symbols!(acc, lhs.args[1]) :
                _brm_bound_symbols!(acc, lhs)
        end
    end
    acc
end
_brm_bound_symbols!(acc, x::Symbol) = push!(acc, x)
_brm_bound_symbols!(acc, x::Expr) =
    Meta.isexpr(x, :tuple) ? (foreach(a -> _brm_bound_symbols!(acc, a), x.args); acc) : acc
_brm_bound_symbols!(acc, _) = acc

# A `~` name another statement (or a kernel cell) reads is a parameter or
# predictor; one nothing reads is an observed response.
function _brm_inventory_columns(inventory, captures)
    latent = union(inventory.assigned, inventory.latent,
                   intersect(inventory.responses, union(inventory.reads, captures)))
    names = union(inventory.reads, inventory.responses,
                  setdiff(captures, inventory.cell_bound))
    Tuple(sort!(collect(union(setdiff(names, latent), inventory.data))))
end

# The builder's column inventory as an expression. A `constant(expr)` input is
# a data field unless the builder's scope binds that name (`pi`, a captured
# local), exactly as `_brm_constant_input` resolves it, so that test runs
# where the builder is defined. The descriptor's `brm_columns` is the REPLAY
# schema and omits such an input, because replay keeps the fitted constant.
function _brm_inventory_columns_expr(inventory, captures, formula_names)
    columns = _brm_inventory_columns(inventory, captures)
    scoped = sort!([s for s in inventory.constant_reads
                    if !(s in columns) && !(s in formula_names)])
    static = Expr(:tuple, map(QuoteNode, columns)...)
    isempty(scoped) && return static
    Expr(:call, _brm_builder_columns, static,
         (:($(Expr(:macrocall, Symbol("@isdefined"), LineNumberNode(0), s)) ?
            () : ($(QuoteNode(s)),)) for s in scoped)...)
end
_brm_builder_columns(static, scoped...) =
    Tuple(sort!(unique!(Symbol[static..., (s for x in scoped for s in x)...])))

"""
    _brm(formula::Union{Expr,AbstractString}; df=nothing) -> Expr

The macro-free entry point behind [`@brm`](@ref). Parses a formula block
and returns either a function definition (`df=nothing`, default) or a
`let`-wrapped BRMI construction (`df` supplied). Useful when you want to
construct the formula expression programmatically rather than via the
macro.
"""
_brm(x::AbstractString; kwargs...) = _brm(Meta.parse("""
begin
    $x
end
"""); kwargs...)
_brm(x::Expr; df=nothing) = begin
    (x.head == :block || x.head == :(=) || isxcall(x, :~) ||
     _is_brm_plate(x)) || error(
        "@brm: a standalone predictor fragment such as `@brm(1 + x)` is " *
        "not implemented. Use nested `@brm(1 + x)` inside an outer `@brm` " *
        "likelihood, or write a top-level `lhs ~ rhs` formula.")
    named = x.head == :(=)
    lhs, x = named ? x.args : (:($(gensym("model"))(__df__)), x)
    source = Base.remove_linenums!(Meta.isexpr(x, :block) ?
        deepcopy(x) : Expr(:block, deepcopy(x)))
    alllocals = OrderedDict{Symbol,Symbol}()
    jointkeys = Set{Symbol}()
    captures = Set{Symbol}()
    groupings = _brm_plate_groupings(x)
    grouped_reads = Set{Symbol}()
    inventory = _brm_inventory()
    info = (;alllocals, jointkeys, captures, groupings, grouped_reads, inventory)
    x = parse!(x; info)
    # A builder fragment may declare groupings that a cell in another fragment
    # reads: `merge` re-parses the composed source, which checks them again.
    # The fragment itself refuses only when built on its own.
    fragment = isnothing(df) && !named
    unread = _brm_unread_plate_groupings(groupings, grouped_reads)
    fragment || isnothing(unread) || error(unread)
    nonlocals = [key for (key, value) in pairs(alllocals) if value == :nonlocal]
    maybelocals = [key for (key, value) in pairs(alllocals) if value == :maybelocal]
    # kernel(...) do-block captures: body free-symbols that are NOT already a
    # formula name. `capturedata` keeps only those present as df columns, so a
    # do-block param, cell-local, or typo that is not a column is silently
    # dropped (and a non-column typo still errors loudly at the backend/StanBlocks
    # trace, exactly as before this capture path existed).
    real_captures = sort!(collect(setdiff(captures, keys(alllocals))))
    finalize = :($_brm_finalize_operations(
        (;$(keys(alllocals)...)), __caps__))
    # ONE shared builder body, parameterised on the `__df__` symbol. Nonlocals
    # bind via the @getproperty (hasproperty→MissingColumn) fallback so a name
    # that isn't a df column (e.g. a multi-equation predictor/param like
    # `loc`/`err`) becomes a MissingColumn instead of erroring; `data` and
    # `maybedata` are interpolated as values so the generated code resolves in
    # ANY consumer scope (`data` is unexported). Both forms below reuse THIS
    # body, so the two builder forms cannot diverge.
    body = Expr(:block,
        (isnothing(unread) ? () : (:($error($unread)),))...,
        :(__ddf__ = $data(__df__)),
        [:($nonlocal = @getproperty __ddf__.$nonlocal) for nonlocal in nonlocals]...,
        :((;$(maybelocals...)) = $maybedata(__df__)),
        x,
        :(__caps__ = $capturedata(__df__, ($(map(QuoteNode, real_captures)...),))),
        finalize,
    )
    # Hygiene: the body assigns every formula name (nonlocal bindings, the
    # maybelocals destructuring, `@n` literal/response bindings) plus the
    # `__ddf__`/`__caps__` temporaries. Without an explicit local scope, a
    # builder defined in local scope (a function, loop, or let block) is a
    # closure that CAPTURES same-named enclosing bindings and overwrites them
    # when it runs — e.g. an in-loop `@brm` silently replaces the loop variable
    # with a NamedColumn (snag bambi-quantile-r-36bd8217). Declare every
    # assigned name `let`-local so the builder can neither read nor clobber
    # enclosing state; reads of never-assigned names (call heads, spliced
    # values) still resolve outward exactly as before.
    body = Expr(:let, Expr(:block, unique!([keys(alllocals)..., :__ddf__, :__caps__])...), body)
    if isnothing(df) && named
        # named no-df: the caller's `f(df) = body` function definition.
        Expr(:(=), lhs, body)
    elseif isnothing(df)
        # no-df: `gensym_model(__df__) = body`, a reusable `df -> BRMI` builder,
        # wrapped with its authored source and data-free column inventory.
        columns = _brm_inventory_columns_expr(inventory, captures, keys(alllocals))
        Expr(:block, Expr(:(=), lhs, body),
             Expr(:call, BRMBuilder, lhs.args[1], QuoteNode(source), columns,
                  Expr(:macrocall, Symbol("@__MODULE__"), LineNumberNode(0))))
    else
        # baked: the SAME body with `__df__` bound to the literal df.
        Expr(:let, Expr(:block, :(__df__ = $df)), body)
    end
end
brm(df, formula::AbstractString) = eval(_brm(formula; df))

"""
    parse!(x; info) -> Expr

Walk a formula expression once, classifying every bare `Symbol` into
`info.alllocals` (as `:nonlocal` for data columns, `:local` for literal
bindings, `:maybelocal` for response-side names that may or may not be
observed). Returns the rewritten expression with `~`/`=` statements
lowered into [`@n`](@ref)/[`@x`](@ref) calls. Used internally by
[`_brm`](@ref).
"""
parse!(x; info) = x

function _is_brm_legacy_kernel_doblock(x)
    Meta.isexpr(x, :do, 2) || return false
    call = x.args[1]
    Meta.isexpr(call, :call) && !isempty(call.args) || return false
    head = call.args[1]
    head === :kernel && return true
    head isa Expr && Meta.isexpr(head, :., 2) || return false
    name = head.args[2]
    name === :kernel || (name isa QuoteNode && name.value === :kernel)
end

function _brm_depwarn_kernel_doblock()
    Base.depwarn(
        "`kernel(...) do` inside `@brm` is deprecated; use an indexed " *
        "`@plate for i in eachindex(x) ... end` cell instead. The compatibility " *
        "spelling remains supported for now.",
        :_brm_depwarn_kernel_doblock,
    )
    nothing
end

function _brm_depwarn_plate_inbody_ragged()
    Base.depwarn(
        "`ragged(x, group)[i]` inside a `@plate for` body is deprecated; declare " *
        "the grouping once at model scope with a `ragged(x, group)` statement and " *
        "read `x[i]` in the cell. The in-body spelling remains supported for now.",
        :_brm_depwarn_plate_inbody_ragged,
    )
    nothing
end

parse!(x::Expr; info) = if x.head == :block
    Expr(:block, parse!.(x.args; info)...)
elseif _is_brm_plate(x)
    groupings = hasproperty(info, :groupings) ? info.groupings : Dict{Symbol,_BRMPlateGrouping}()
    grouped_reads = hasproperty(info, :grouped_reads) ? info.grouped_reads : Set{Symbol}()
    # `y[p] ~ AR1Normal(...)` stays in the cell as written, and each backend
    # calls its AR(1) submodel (`ar1_normal.jl`); its time checks run when the
    # model is built.
    ar1_normal_checks = _brm_cell_ar1_normal_checks(x)
    kernel, inbody_ragged = _brm_plate_kernel(x; groupings, grouped_reads)
    parsed = parse!(kernel; info)
    isempty(ar1_normal_checks) || (parsed = Expr(:block, parsed, ar1_normal_checks...))
    inbody_ragged ? Expr(:block, :($_brm_depwarn_plate_inbody_ragged()), parsed) : parsed
elseif _is_brm_ar1_normal(x)
    # The series and its grouping are the plate's (decision `1pnob5t`).
    error("@brm: `AR1Normal(...)` observes one series per `@plate` cell: write " *
        "`@plate for p in groups(g; order = t)` with " *
        "`y[p] ~ AR1Normal(mu[p], sigma, phi; time = t[p])` in its cell; got " *
        "`$(sprint(Base.show_unquoted, x))`")
elseif _is_brm_plate_grouping(x)
    # Collected by `_brm_plate_groupings` before parsing and consumed by the
    # `@plate for` rewrite; the declaration itself emits nothing.
    decl = _brm_plate_grouping_decl(x)
    hasproperty(info, :groupings) && decl !== nothing &&
        haskey(info.groupings, decl.name) || error(
        "@brm: `$(repr(x))` declares a plate grouping and must be a statement " *
        "of the `@brm` block")
    nothing
elseif x.head == :(=) && _is_optional_prior_lhs(x.args[1])
    lhs, rhs = x.args
    _parse_effect!(_optional_prior_address(lhs), rhs; info, known=true, optional=true)
elseif x.head == :(=) && _is_effect_lhs(x.args[1])
    lhs, rhs = x.args
    _parse_effect!(lhs, rhs; info, known=true)
elseif x.head == :(=) && isxcall(x.args[2], :constant)
    _parse_constant!(x.args...; info)
elseif x.head == :(=)
    lhs, rhs = x.args
    parselocals!(rhs; info, val=:nonlocal)
    parselocals!(lhs; info, val=:local)
    :(@n $lhs = @x $assign($(xname(lhs)), $rhs))
elseif isxcall(x, :~) && _is_optional_prior_lhs(x.args[2])
    _, lhs, rhs = x.args
    _parse_effect!(_optional_prior_address(lhs), rhs; info, optional=true)
elseif isxcall(x, :~) && _is_effect_lhs(x.args[2])
    _, lhs, rhs = x.args
    _parse_effect!(lhs, rhs; info)
elseif isxcall(x, :~) && Meta.isexpr(x.args[2], :vect)
    _, lhs, rhs = x.args
    _parse_joint_response!(lhs, rhs; info)
elseif isxcall(x, :~) && isxcall(x.args[2], :mi) &&
       length(x.args[2].args) == 2 && Meta.isexpr(x.args[2].args[2], :vect)
    _, lhs, rhs = x.args
    _parse_joint_response!(lhs.args[2], rhs; info, impute=true)
elseif isxcall(x, :~) && Meta.isexpr(x.args[2], :tuple)
    _, lhs, rhs = x.args
    _parse_broadcast_lhs!(lhs, rhs; info)
elseif isxcall(x, :~) && _is_hyper_lhs(x.args[2])
    _, lhs, rhs = x.args
    _parse_hyper!(lhs, rhs; info)
elseif isxcall(x, :~)
    _, lhs, rhs = x.args
    legacy_kernel_doblock = _is_brm_legacy_kernel_doblock(rhs)
    # Shield brms-style `(e | ID | g)` ranef IDs from parselocals! so the bare
    # ID symbol doesn't get registered as a data-column name. Left-associative
    # `|` lowers to `Expr(:call, :|, Expr(:call, :|, e, id), g)`; rewrite to a
    # three-arg `|` whose middle is a QuoteNode so parselocals! treats it as a
    # literal, not a bare Symbol. _x then sees the three-arg form and emits an
    # ExprColumn{|} with the id as a Symbol value.
    rhs = rewrite_ranef_ids(rhs)
    parselocals!(rhs; info, val=:nonlocal)
    parselocals!(lhs; info, val=:maybelocal)
    hasproperty(info, :inventory) &&
        _brm_inventory_statement!(info.inventory, lhs, rhs, legacy_kernel_doblock)
    parsed = :(@n $lhs = @x $(Expr(:call, :~, lhs, rhs)))
    legacy_kernel_doblock ?
        Expr(:block, :($_brm_depwarn_kernel_doblock()), parsed) : parsed
else
    dump(x)
    error("Don't know how to handle parse!($x)!")
end

# ---- `@plate for` kernel-cell surface --------------------------------------
#
# BRM's annotated plate is intentionally a surface rewrite, not a second
# implementation of grouped cells. It lowers to the already-authoritative
# `target ~ kernel(positionals...) do slices... ... end` IR before `_x` sees
# the formula. SBBRMI and RKBRMI therefore share every grouping, ragged-input,
# omitted-outcome, capture, and cell-observation rule with the legacy spelling.
#
# The accepted subset is the group-local kernel shape rather than StanBlocks'
# entire general annotated-loop language:
#
#     ragged(dose, dose_subject)
#     @plate for i in eachindex(t)
#         loc[i] = cell(t[i], dose[i], theta[i])
#         y[i] ~ normal(loc[i], sigma)
#     end
#
# An indexed read becomes one kernel positional and one cell parameter. An
# indexed assignment becomes a named cell value; the final indexed assignment
# is the collected kernel result. When that assignment is the cell's last
# statement its right-hand side is the cell's return value, as in a closure
# ending in an expression; otherwise the named value is returned. Bare
# assignments remain cell locals. This
# retains the StanBlocks/RKPPL distinction between model-scope indexed outputs
# and per-cell bare locals without asking either backend to learn new IR.
#
# A flat secondary row axis is declared once, outside every cell, by a
# model-scope `ragged(x, group)` statement. The rewrite then reads `x[i]` as the
# positional `ragged(x, group)`, so the lowered kernel — and with it the Stan
# program, the RK graph and the descriptor — is exactly the one the deprecated
# in-body `ragged(x, group)[i]` read produces.
#
# A cell whose only indexed statements are observations (`y[i] ~ …`) has no
# collected result. It lowers to a kernel whose closure ends in `nothing`, the
# closure spelling of "no value", under a compiler-owned name derived from the
# first observed response (`_brm_plate_effect_name`). Both backends then emit the
# cell's statements without a collected result, so a prediction the cell only
# observes stays a cell local instead of becoming a model-scope output.
_is_brm_plate(x) = Meta.isexpr(x, :macrocall) && !isempty(x.args) &&
                   x.args[1] === Symbol("@plate")

# A `@scan begin … end` statement inside an `@plate for` cell is one recurrence
# per cell, exactly as in StanBlocks and RKPPL: the seeds fill the first
# entries, the trailing `for` steps the rest. Its reads of sliced inputs
# (`y[i][t]`, `mu[i][t - 1]`) become reads of the cell's slice like any other
# cell read; the block itself is passed to the backend unchanged.
_is_brm_scan(x) = Meta.isexpr(x, :macrocall) && !isempty(x.args) &&
                  x.args[1] === Symbol("@scan")

# The cell's statements with every `@scan` block's seeds and step statements
# spliced in, for the checks that look at every observation of the cell.
function _brm_plate_flat_statements(stmts)
    flat = Any[]
    function visit(stmt)
        if _is_brm_scan(stmt)
            visit(stmt.args[end])
        elseif Meta.isexpr(stmt, :block)
            foreach(visit, stmt.args)
        elseif Meta.isexpr(stmt, :for)
            visit(stmt.args[2])
        elseif !(stmt isa LineNumberNode)
            push!(flat, stmt)
        end
    end
    foreach(visit, stmts)
    flat
end

_is_brm_plate_grouping(x) = isxcall(x, :ragged)

# Model-scope `ragged(x, group)` statements of one `@brm` block, keyed by `x`.
# Each value's `spec` is `(group,)`, or `(group, cells)` for
# `ragged(x, group => cells)`: the cell label column `cells` lists the plate's
# cells, one label per row of the cells' own frame, and `x`'s rows join to them
# by their `group` label. Its `into` is the `target` of
# `ragged(idx, group; into=target)`: `idx` holds row numbers on the flat axis
# `target` labels, which cells read as positions inside their own cell's slice
# of it. Collected before parsing so a declaration may follow the cells that
# read it.
const _BRMPlateGrouping =
    NamedTuple{(:spec, :into),Tuple{Tuple{Vararg{Symbol}},Union{Nothing,Symbol}}}

function _brm_plate_groupings(x)
    groupings = OrderedDict{Symbol,_BRMPlateGrouping}()
    function collect!(stmt)
        if Meta.isexpr(stmt, :block)
            foreach(collect!, stmt.args)
        elseif _is_brm_plate_grouping(stmt)
            decl = _brm_plate_grouping_decl(stmt)
            decl === nothing && error(
                "@brm: a model-scope plate grouping is `ragged(x, group)` or " *
                "`ragged(x, group => cells)` with a formula or data name `x`, a " *
                "grouping data column `group` and a cell label column `cells`, or " *
                "`ragged(idx, group; into=target)` for an index column into the " *
                "flat axis `target` labels; got `$(repr(stmt))`")
            (; name, spec, into, order) = decl
            # Row order belongs to the plate that reads the rows (decision
            # `1pnob5t`), not to another model-scope statement.
            order === nothing || error(
                "@brm: `$(sprint(Base.show_unquoted, stmt))`: a model-scope grouping " *
                "takes no `order=`; " *
                "state a per-group plate's grouping and row order in its header, " *
                "`@plate for p in groups($(spec[1]); order = $order)`")
            name in spec && error(
                "@brm: `$(repr(stmt))` groups `$name` by itself")
            length(spec) == 2 && spec[1] === spec[2] && error(
                "@brm: `$(repr(stmt))` matches `$(spec[1])` against itself; the " *
                "cell label column lists each cell once, on the cells' own frame")
            name === into && error(
                "@brm: `$(repr(stmt))` indexes `$name` into itself; `into=` names " *
                "the grouping column of the axis whose rows `$name` points at")
            haskey(groupings, name) && error(
                "@brm: the plate grouping of `$name` is declared twice " *
                "(`$(_brm_plate_grouping_repr(name, groupings[name]))` and " *
                "`$(repr(stmt))`)")
            groupings[name] = (; spec, into)
        end
    end
    collect!(x)
    groupings
end

# The positional arguments, `into=` target and `order=` column of a
# `ragged(...)` call, or `nothing` when it carries any other keyword.
function _brm_plate_grouping_args(stmt)
    positional, keywords = Any[], Any[]
    for arg in stmt.args[2:end]
        Meta.isexpr(arg, :parameters) ? append!(keywords, arg.args) :
        Meta.isexpr(arg, :kw) ? push!(keywords, arg) : push!(positional, arg)
    end
    options = Dict{Symbol,Symbol}()
    for keyword in keywords
        Meta.isexpr(keyword, :kw, 2) && keyword.args[1] in (:into, :order) &&
            keyword.args[2] isa Symbol && !haskey(options, keyword.args[1]) ||
            return nothing
        options[keyword.args[1]] = keyword.args[2]
    end
    (; positional, into=get(options, :into, nothing), order=get(options, :order, nothing))
end

function _brm_plate_grouping_decl(stmt)
    args = _brm_plate_grouping_args(stmt)
    args === nothing && return nothing
    (; positional, into, order) = args
    length(positional) == 2 && positional[1] isa Symbol || return nothing
    spec = _brm_plate_grouping_spec(positional[2])
    spec === nothing ? nothing : (; name=positional[1], spec, into, order)
end

_brm_plate_grouping_spec(group::Symbol) = (group,)
_brm_plate_grouping_spec(x) =
    Meta.isexpr(x, :call, 3) && x.args[1] === :(=>) &&
    x.args[2] isa Symbol && x.args[3] isa Symbol ? (x.args[2], x.args[3]) : nothing

function _brm_plate_grouping_repr(name, grouping::_BRMPlateGrouping)
    (; spec, into) = grouping
    group = length(spec) == 1 ? "$(spec[1])" : "$(spec[1]) => $(spec[2])"
    isnothing(into) ? "ragged($name, $group)" : "ragged($name, $group; into=$into)"
end

# The kernel positional a cell read of a declared name lowers to:
# `ragged(x, group[, cells][; into=target])`.
function _brm_plate_grouping_input(name, grouping::_BRMPlateGrouping)
    (; spec, into) = grouping
    call = Expr(:call, :ragged, name, spec...)
    isnothing(into) || insert!(call.args, 2, Expr(:parameters, Expr(:kw, :into, into)))
    call
end

# A declaration no cell reads would leave the author believing `x` is grouped
# somewhere it is not, for example in a top-level likelihood. Returns the
# refusal, or `nothing` when every declared grouping is read.
function _brm_unread_plate_groupings(groupings, grouped_reads)
    unread = [name for name in keys(groupings) if !(name in grouped_reads)]
    isempty(unread) && return nothing
    "@brm: model-scope `ragged(x, group)` declares how a `@plate for` cell " *
    "reads `x[i]`, but no cell reads " *
    join(("`$name[i]`" for name in unread), ", ") * ". Group a top-level " *
    "response with `ragged(y, group) ~ ...` instead"
end

function _brm_plate_loop(x::Expr)
    length(x.args) == 3 || error(
        "@brm: `@plate` takes exactly `@plate for i in eachindex(x) ... end`")
    loop = x.args[3]
    Meta.isexpr(loop, :for) || error(
        "@brm: `@plate` takes exactly `@plate for i in eachindex(x) ... end`")
    binding = loop.args[1]
    Meta.isexpr(binding, :(=), 2) && binding.args[1] isa Symbol || error(
        "@brm: a kernel `@plate` takes one loop variable, `for i in ...`")
    binding.args[1], binding.args[2], loop.args[2]
end

# `@plate for p in groups(g; order = t)`: one cell per distinct label of the
# data column `g`, and each read `x[p]` is `x`'s values on group `p`'s rows,
# ascending in `t` when `order` is given (decision `1pnob5t`). It lowers to the
# kernel positional `ragged(x, g, g; order = t)` for every name the cell reads.
# Returns `(; group, order)`, or `nothing` for an `eachindex`/`axes` range.
function _brm_plate_header_grouping(range)
    isxcall(range, :groups) || return nothing
    args = _brm_plate_grouping_args(range)
    args !== nothing && length(args.positional) == 1 && only(args.positional) isa Symbol &&
        args.into === nothing || error(
        "@brm: a grouped `@plate` range is `groups(g)` or `groups(g; order = t)` with a " *
        "grouping data column `g` and a time column `t`; got " *
        "`$(sprint(Base.show_unquoted, range))`")
    (; group=only(args.positional), args.order)
end
# Each read joins the header's grouping, which is also its own cell label
# column (`_brm_kernel_plate_cells`).
_brm_plate_header_input(name, (; group, order)) = order === nothing ?
    Expr(:call, :ragged, name, group, group) :
    Expr(:call, :ragged, Expr(:parameters, Expr(:kw, :order, order)), name, group, group)

function _brm_plate_range_input(range)
    Meta.isexpr(range, :call) || error(
        "@brm: a kernel `@plate` range must be `eachindex(x)`, `axes(x, 1)` or `groups(g)`")
    head = range.args[1]
    if head === :eachindex && length(range.args) == 2
        return range.args[2]
    elseif head === :axes && length(range.args) == 3 && range.args[3] == 1
        return range.args[2]
    end
    error("@brm: a kernel `@plate` range must be `eachindex(x)`, `axes(x, 1)` or `groups(g)`")
end

function _brm_plate_input_key(x)
    x isa Symbol && return (:column, x)
    # `ragged(x, group)`, or `ragged(x, group, cells)` lowered from a model-scope
    # `ragged(x, group => cells)`, either with an optional `into=target` and
    # `order=time`.
    args = isxcall(x, :ragged) ? _brm_plate_grouping_args(x) : nothing
    if args !== nothing && length(args.positional) in (2, 3) &&
       all(arg -> arg isa Symbol, args.positional)
        return (:ragged, args.positional..., args.into, args.order)
    end
    error(
        "@brm: a kernel `@plate` slices formula and data names (`x[i]`); declare " *
        "a flat secondary axis once at model scope with `ragged(x, group)`; " *
        "got `$(repr(x))`")
end

_brm_plate_param(x::Symbol) = x
_brm_plate_param(x::Expr) = _brm_plate_grouping_args(x).positional[1]::Symbol # validated by `_brm_plate_input_key`

_brm_plate_indexed(x, i) = Meta.isexpr(x, :ref, 2) && x.args[2] === i

# An observation samples data, so that omitting the data turns it into a prior
# draw; a name the cell defines by `=` has nothing to draw. StanBlocks refuses
# such a `~` inside every plate cell, but its message names BRM's lowered kernel
# formal (`qt_y = y`, "write `y ~ …`"), and `y ~ …` in an `@plate` cell reads
# the whole column. Refuse it here in the authored spelling instead.
function _brm_plate_check_observed(lhs, i, bare_bindings, output_set)
    if lhs isa Symbol && haskey(bare_bindings, lhs)
        rhs = bare_bindings[lhs]
        fix = _brm_plate_indexed(rhs, i) ? "write `$rhs ~ …` instead" :
              "observe the data it is computed from instead"
        error(
            "@brm: kernel `@plate` observation `$lhs ~ …` samples `$lhs`, which " *
            "this cell defines as `$lhs = $rhs`. An observation samples the data " *
            "itself, so that omitting the data turns it into a prior draw; a value " *
            "defined by `=` has nothing to draw — $fix")
    elseif _brm_plate_indexed(lhs, i) && lhs.args[1] in output_set
        error(
            "@brm: kernel `@plate` observation `$lhs ~ …` samples the plate output " *
            "`$lhs`, which this cell defines by `=`; observe the data itself")
    end
end

function _brm_plate_outputs(body, i)
    stmts = Meta.isexpr(body, :block) ? body.args : Any[body]
    outputs = Symbol[]
    for stmt in stmts
        stmt isa LineNumberNode && continue
        if Meta.isexpr(stmt, :(=), 2) && _brm_plate_indexed(stmt.args[1], i)
            lhs = stmt.args[1].args[1]
            lhs isa Symbol || error(
                "@brm: an indexed `@plate` output must be `name[$i] = ...`; " *
                "got `$(repr(stmt.args[1]))`")
            lhs in outputs && error(
                "@brm: kernel `@plate` output `$lhs[$i]` is assigned more than once")
            push!(outputs, lhs)
        end
    end
    isempty(outputs) && !any(stmt -> isxcall(stmt, :~), _brm_plate_flat_statements(stmts)) && error(
        "@brm: a kernel `@plate` cell needs an indexed output (`loc[$i] = ...`) " *
        "or an observation (`y[$i] ~ ...`); this one has neither")
    stmts, outputs
end

# The compiler-owned kernel name of an observation-only cell: its first observed
# response plus `_cell` (`y[i] ~ …` and `ragged(y, g)[i] ~ …` both give `y_cell`).
# Backends derive the cell's emitted names from it, as they do from an output's.
function _brm_plate_effect_name(stmts, i)
    for stmt in _brm_plate_flat_statements(stmts)
        isxcall(stmt, :~) || continue
        observed = stmt.args[2]
        # A scan step observes one entry of the cell's slice: `y[i][t] ~ …`.
        Meta.isexpr(observed, :ref) && _brm_plate_indexed(observed.args[1], i) &&
            (observed = observed.args[1])
        _brm_plate_indexed(observed, i) || continue
        source = observed.args[1]
        base = source isa Symbol ? source :
               isxcall(source, :ragged) && source.args[2] isa Symbol ? source.args[2] :
               nothing
        base === nothing || return Symbol(base, :_cell)
    end
    error("@brm: an observation-only kernel `@plate` cell observes a response " *
          "`y[$i] ~ …` or `ragged(y, group)[$i] ~ …`")
end

# A kernel closure ending in `nothing` has no collected result; backends emit its
# statements without one. `_brm_cell_without_result` drops that terminal marker.
function _brm_cell_has_no_result(statements)
    final = findlast(statement -> !(statement isa LineNumberNode), statements)
    final !== nothing && statements[final] === :nothing
end
function _brm_cell_without_result(statements)
    final = findlast(statement -> !(statement isa LineNumberNode), statements)
    Any[statements[1:final-1]; statements[final+1:end]]
end

function _brm_plate_kernel(x::Expr; groupings=Dict{Symbol,_BRMPlateGrouping}(),
                           grouped_reads=Set{Symbol}(), authored=x)
    i, range, body = _brm_plate_loop(x)
    stmts, outputs = _brm_plate_outputs(body, i)
    output_set = Set(outputs)

    positionals = Any[]
    params = Symbol[]
    keys = Any[]
    param_sources = Dict{Symbol,Any}()

    function add_input(source)
        key = _brm_plate_input_key(source)
        found = findfirst(isequal(key), keys)
        found === nothing || return params[found]
        param = _brm_plate_param(source)
        param in output_set && error(
            "@brm: kernel `@plate` output `$param[$i]` also names a sliced input; " *
            "use a distinct output name")
        if haskey(param_sources, param)
            error(
                "@brm: kernel `@plate` inputs `$(repr(param_sources[param]))` and " *
                "`$(repr(source))` would both bind cell name `$param`; rename one " *
                "source before the plate")
        end
        push!(keys, key)
        push!(positionals, source)
        push!(params, param)
        param_sources[param] = source
        param
    end

    # A name with a model-scope `ragged(x, group)` declaration is read through
    # that grouping. The deprecated in-body wrapper stays accepted for a name
    # without one; mixing both spellings for one name is refused. In a
    # `groups(...)` plate every read joins the header's grouping.
    inbody_ragged = false
    cell_groupings = Symbol[]
    header = _brm_plate_header_grouping(range)
    header_source = sprint(Base.show_unquoted, range)
    # Nothing outside the plate could tell which group a collected value
    # belongs to: it would be matched to the groups by position, the hazard
    # `ragged(x, group => cells)` exists to rule out (`3681f4a`).
    header === nothing || isempty(outputs) || error(
        "@brm: a `@plate for $i in $header_source` cell observes its group's rows " *
        "and collects no result; `$(first(outputs))[$i] = …` would leave a value " *
        "per group that nothing outside the plate can match to its group. Keep " *
        "the value a cell local (`$(first(outputs)) = …`) and observe in the cell.")
    function grouped(source)
        if header !== nothing
            source isa Symbol || error(
                "@brm: a `@plate for $i in $header_source` cell reads names as " *
                "`x[$i]`, each the rows of group `$i`; got " *
                "`$(sprint(Base.show_unquoted, source))`")
            haskey(groupings, source) && error(
                "@brm: `$source` is grouped at model scope by " *
                "`$(_brm_plate_grouping_repr(source, groupings[source]))`, but a " *
                "`@plate for $i in $header_source` cell groups every name it reads " *
                "by its header; drop the model-scope statement")
            return _brm_plate_header_input(source, header)
        elseif source isa Symbol && haskey(groupings, source)
            push!(grouped_reads, source)
            source in cell_groupings || push!(cell_groupings, source)
            return _brm_plate_grouping_input(source, groupings[source])
        elseif isxcall(source, :ragged)
            inbody_ragged = true
            name = length(source.args) >= 2 ? source.args[2] : nothing
            name isa Symbol && haskey(groupings, name) && error(
                "@brm: `$name` is grouped at model scope by " *
                "`$(_brm_plate_grouping_repr(name, groupings[name]))`; read it in the cell as " *
                "`$name[$i]`")
        end
        source
    end

    # The range anchor participates even when the body does not read it: it is
    # the subject-count witness in the no-random-effects panel form. A
    # `groups(...)` plate's cells are its grouping's labels instead.
    header === nothing && add_input(grouped(_brm_plate_range_input(range)))

    function rewrite(ex)
        ex === i && error(
            "@brm: a kernel `@plate` loop index is structural; use it only in " *
            "`x[$i]` reads")
        ex isa Expr || return ex
        if _brm_plate_indexed(ex, i)
            source = ex.args[1]
            if source isa Symbol && source in output_set
                return source
            end
            return add_input(grouped(source))
        end
        # A remaining reference with the loop variable as one of ITS OWN axes
        # (`x[i, j]`) is a different slicing contract, not a kernel positional.
        # Nested indexed lanes remain valid: `catalog[index[i]]` recursively
        # becomes `catalog[index]`, with `index` registered as the positional.
        if ex.head === :ref && any(a -> a === i, ex.args[2:end])
            error(
                "@brm: kernel `@plate` inputs use one cell slice `x[$i]`; got " *
                "`$(repr(ex))`")
        end
        Expr(ex.head, map(rewrite, ex.args)...)
    end

    rewritten = Any[]
    bare_bindings = Dict{Symbol,Any}() # cell local => authored right-hand side

    # A `@scan` block keeps its shape: seeds, then one `for` loop of steps. Its
    # slice reads are rewritten like every other cell read; its `=`/`~` targets
    # are the scan's carried arrays and step locals, or (`y[i][t] ~ …`) entries
    # of an observed slice.
    function rewrite_scan(block)
        Meta.isexpr(block, :block) || error(
            "@brm: a kernel `@plate` cell's `@scan` takes `@scan begin … end`")
        Expr(:block, map(rewrite_scan_statement, block.args)...)
    end
    function rewrite_scan_statement(stmt)
        stmt isa LineNumberNode && return stmt
        if Meta.isexpr(stmt, :for) && Meta.isexpr(stmt.args[1], :(=), 2) &&
                stmt.args[1].args[1] isa Symbol
            loop = stmt.args[1]
            return Expr(:for, Expr(:(=), loop.args[1], rewrite(loop.args[2])),
                rewrite_scan(stmt.args[2]))
        elseif Meta.isexpr(stmt, :(=), 2)
            return Expr(:(=), rewrite(stmt.args[1]), rewrite(stmt.args[2]))
        elseif isxcall(stmt, :~)
            _brm_plate_check_observed(stmt.args[2], i, bare_bindings, output_set)
            return Expr(:call, :~, rewrite(stmt.args[2]), rewrite(stmt.args[3]))
        end
        error(
            "@brm: a `@scan` block in a kernel `@plate` cell holds `=`/`~` seeds " *
            "and one `for t in …` loop of `=`/`~` steps; got `$(repr(stmt))`")
    end

    for stmt in stmts
        stmt isa LineNumberNode && (push!(rewritten, stmt); continue)
        if Meta.isexpr(stmt, :(=), 2)
            lhs, rhs = stmt.args
            if lhs isa Symbol
                bare_bindings[lhs] = rhs
                push!(rewritten, Expr(:(=), lhs, rewrite(rhs)))
            elseif _brm_plate_indexed(lhs, i) && lhs.args[1] isa Symbol
                push!(rewritten, Expr(:(=), lhs.args[1], rewrite(rhs)))
            else
                error(
                    "@brm: a kernel `@plate` assignment binds a bare cell local " *
                    "or `name[$i]`; got `$(repr(lhs))`")
            end
        elseif isxcall(stmt, :~)
            _brm_plate_check_observed(stmt.args[2], i, bare_bindings, output_set)
            push!(rewritten, Expr(:call, :~, rewrite(stmt.args[2]), rewrite(stmt.args[3])))
        elseif _is_brm_scan(stmt)
            push!(rewritten, Expr(:macrocall, stmt.args[1:end-1]..., rewrite_scan(stmt.args[end])))
        else
            error(
                "@brm: kernel `@plate` cells contain `=` assignments, `~` " *
                "statements and `@scan begin … end` blocks; got `$(repr(stmt))`")
        end
    end
    collision = findfirst(p -> haskey(bare_bindings, p), params)
    collision === nothing || error(
        "@brm: kernel `@plate` cell local `$(params[collision])` shadows a " *
        "sliced input of the same name")

    # A terminal `target[i] = expr` IS the cell's value: return `expr` directly,
    # exactly as a closure ending in `expr` does. Binding it to a cell local
    # first would make StanBlocks promote that local to a second, identical
    # posterior carrier and copy it into the collected result. An output that
    # later statements read stays a local and is returned by name. A cell with
    # no output ends in `nothing`: it has no collected result at all.
    if isempty(outputs)
        target = _brm_plate_effect_name(stmts, i)
        push!(rewritten, :nothing)
    else
        target = last(outputs)
        final = findlast(stmt -> !(stmt isa LineNumberNode), rewritten)
        if Meta.isexpr(stmts[final], :(=), 2) && _brm_plate_indexed(stmts[final].args[1], i) &&
           stmts[final].args[1].args[1] === target
            rewritten[final] = rewritten[final].args[2]
        else
            push!(rewritten, target)
        end
    end
    lambda = Expr(:->, Expr(:tuple, params...), Expr(:block, rewritten...),
                  _brm_plate_source(i, range, first(_brm_plate_outputs(_brm_plate_loop(authored)[3], i)),
                      cell_groupings, groupings))
    call = Expr(:call, :kernel, lambda, positionals...)
    Expr(:call, :~, target, call), inbody_ragged
end

# The cell as written, for `show`: the model-scope groupings it reads, then the
# loop with one statement per line.
function _brm_plate_source(i, range, stmts, cell_groupings, groupings)
    io = IOBuffer()
    for name in cell_groupings
        println(io, _brm_plate_grouping_repr(name, groupings[name]))
    end
    _brm_print_plate(io, i, range, stmts)
    String(take!(io))
end
function _brm_print_plate(io::IO, i, range, stmts)
    print(io, "@plate for ", i, " in ")
    _brm_print_source(io, range)
    println(io)
    _brm_print_cell_statements(io, stmts, "    ")
    print(io, "end")
end
# One line per statement; a `@scan` block and its `for` loop nest one level.
function _brm_print_cell_statements(io::IO, stmts, indent)
    for stmt in stmts
        stmt isa LineNumberNode && continue
        print(io, indent)
        if _is_brm_scan(stmt) || Meta.isexpr(stmt, :for)
            body = stmt.args[end]
            if _is_brm_scan(stmt)
                println(io, "@scan begin")
            else
                print(io, "for ", stmt.args[1].args[1], " in ")
                _brm_print_source(io, stmt.args[1].args[2])
                println(io)
            end
            _brm_print_cell_statements(io,
                Meta.isexpr(body, :block) ? body.args : Any[body], indent * "    ")
            print(io, indent, "end")
        else
            _brm_print_source(io, stmt)
        end
        println(io)
    end
end

# Formula source as written. Julia's own `Expr` printer renders a left-nested
# `(1 | p | g)` as `((1 | p) | g)`, `1e-3 * d` as `0.001d` and keywords as
# `ref = "b"`; this prints binary operators infix with the same precedence rule
# as `_brm_show_infix`, calls with `f(args; kw=value)`, and falls back to
# Julia's printer for any other syntax.
_brm_print_source(io::IO, x, outer::Int=0; left::Bool=false) = show(io, x)
_brm_print_source(io::IO, x::Symbol, outer::Int=0; left::Bool=false) = print(io, x)
# A quoted literal prints as written (`:clamp`); `show` would print the
# `QuoteNode` itself, `:(:clamp)`.
_brm_print_source(io::IO, x::QuoteNode, outer::Int=0; left::Bool=false) = print(io, x)
_brm_print_source(io::IO, x::LineNumberNode, outer::Int=0; left::Bool=false) = nothing
function _brm_print_source(io::IO, x::Expr, outer::Int=0; left::Bool=false)
    op = _brm_source_infix(x)
    if !isnothing(op)
        operands = x.head === :call ? x.args[2:end] : x.args
        precedence = Base.operator_precedence(op)
        parens = precedence < outer || (precedence == outer && !left)
        parens && print(io, "(")
        for (i, arg) in enumerate(operands)
            # A range is written `2:T`, as Julia prints it.
            i == 1 || print(io, op === :(:) ? ":" : " $op ")
            _brm_print_source(io, arg, precedence; left=i == 1)
        end
        parens && print(io, ")")
    elseif Meta.isexpr(x, :call, 2) && x.args[1] in (:-, :+, :!)
        print(io, x.args[1])
        _brm_print_source(io, x.args[2], Base.operator_precedence(:^))
    elseif Meta.isexpr(x, :call)
        _brm_print_source(io, x.args[1], typemax(Int))
        _brm_print_arguments(io, x.args[2:end], "(", ")")
    elseif _brm_is_broadcast_call(x)
        _brm_print_source(io, x.args[1], typemax(Int))
        _brm_print_arguments(io, x.args[2].args, ".(", ")")
    elseif Meta.isexpr(x, :ref)
        _brm_print_source(io, x.args[1], typemax(Int))
        _brm_print_arguments(io, x.args[2:end], "[", "]")
    elseif Meta.isexpr(x, :vect)
        _brm_print_arguments(io, x.args, "[", "]")
    elseif Meta.isexpr(x, :(=), 2)
        _brm_print_source(io, x.args[1])
        print(io, " = ")
        _brm_print_source(io, x.args[2])
    elseif Meta.isexpr(x, :block) && count(a -> !(a isa LineNumberNode), x.args) == 1
        # The value of a short-form `sd(:, p) = value` statement.
        _brm_print_source(io, only(filter(a -> !(a isa LineNumberNode), x.args)), outer; left)
    elseif _is_brm_plate(x)
        i, range, body = _brm_plate_loop(x)
        _brm_print_plate(io, i, range, Meta.isexpr(body, :block) ? body.args : Any[body])
    else
        print(io, x)
    end
end
function _brm_source_infix(x::Expr)
    x.head in (:||, :&&) && return x.head
    Meta.isexpr(x, :call) && length(x.args) >= 3 && x.args[1] isa Symbol &&
        !any(a -> Meta.isexpr(a, :parameters), x.args) &&
        Base.isbinaryoperator(x.args[1]) ? x.args[1] : nothing
end
function _brm_print_arguments(io::IO, args, open, close)
    print(io, open)
    positional = filter(a -> !Meta.isexpr(a, :parameters), args)
    keywords = Any[kw for a in args if Meta.isexpr(a, :parameters) for kw in a.args]
    append!(keywords, filter(a -> Meta.isexpr(a, :kw), positional))
    filter!(a -> !Meta.isexpr(a, :kw), positional)
    for (i, arg) in enumerate(positional)
        i == 1 || print(io, ", ")
        _brm_print_source(io, arg)
    end
    for (i, kw) in enumerate(keywords)
        print(io, i == 1 ? (open == "(" || open == ".(" ? "; " : ", ") : ", ")
        if Meta.isexpr(kw, :kw, 2)
            print(io, kw.args[1], "=")
            _brm_print_source(io, kw.args[2])
        else
            _brm_print_source(io, kw)
        end
    end
    print(io, close)
end

function _joint_response_symbols(lhs::Expr)
    Meta.isexpr(lhs, :vect) || error(
        "@brm: a joint response must use vector syntax `[y1, y2]`")
    length(lhs.args) >= 2 || error(
        "@brm: a joint response needs at least two outcome columns; got " *
        "$(length(lhs.args))")
    all(x -> x isa Symbol, lhs.args) || error(
        "@brm: every entry of a joint response must be a bare data-column " *
        "name, got `$lhs`")
    names = Tuple(lhs.args)
    length(unique(names)) == length(names) || error(
        "@brm: joint response columns must be unique and ordered, got `$lhs`")
    names
end

function _parse_joint_response!(lhs::Expr, rhs; info, impute=false)
    names = _joint_response_symbols(lhs)
    hasproperty(info, :inventory) && union!(info.inventory.data, names)
    parselocals!(rhs; info, val=:nonlocal)

    key = _joint_response_operation_key(names)
    reserved = _joint_response_reserved_keys(names; impute)
    collision = findfirst(k -> haskey(info.alllocals, k), reserved)
    isnothing(collision) || error(
        "@brm: generated joint-response binding `$(reserved[collision])` " *
        "collides with another formula binding; rename that formula name or " *
        "one of the outcome columns")
    info.alllocals[key] = :local
    hasproperty(info, :jointkeys) && union!(info.jointkeys, reserved)

    columns = Expr(:tuple, map(names) do n
        qn = QuoteNode(n)
        :($NamedColumn($qn,
            hasproperty(__df__, $qn) ?
                $DataColumn(getproperty(__df__, $qn)) : $MissingColumn()))
    end...)
    joint = :($JointResponseColumn($columns, $impute))
    parsed_rhs = _x(rhs)
    declaration = :($key = $NamedColumn($(QuoteNode(key)),
        $ExprColumn(~, $joint, $parsed_rhs)))
    impute || return declaration
    # Completed columns are deterministic projections of ONE joint draw.
    # Retain the joint declaration on each reference for dependency ordering,
    # observed-only transform anchors, and frozen replay.
    aliases = map(enumerate(names)) do (i, n)
        haskey(info.alllocals, n) && info.alllocals[n] !== :nonlocal && error(
            "@brm: imputed joint column `$n` already has a model declaration")
        info.alllocals[n] = :local
        :($n = $NamedColumn($(QuoteNode(n)), $ExprColumn($assign,
            $NamedColumn($(QuoteNode(n)), $MissingColumn()),
            $ExprColumn($brm_joint_column, $key, $i, $(length(names)),
                        $(QuoteNode(Symbol(key, :_n)))))))
    end
    Expr(:block, declaration, aliases...)
end

function _parse_broadcast_lhs!(lhs::Expr, rhs; info)
    Meta.isexpr(lhs, :tuple) || error(
        "@brm: a broadcast LHS must use tuple syntax `(x, y)`")
    isempty(lhs.args) && error(
        "@brm: a broadcast LHS `(x, y, ...) ~ rhs` needs at least one name; " *
        "got `() ~ ...`. Write `x ~ rhs` or `(x, y) ~ rhs`.")
    for elem in lhs.args
        (Meta.isexpr(elem, :tuple) || Meta.isexpr(elem, :vect)) && error(
            "@brm: a broadcast LHS `(x, y, ...) ~ rhs` takes one sampling LHS " *
            "per slot; nested collections such as `$elem` are not supported. " *
            "Write one statement per target.")
        _is_effect_lhs(elem) && error(
            "@brm: a broadcast LHS `(x, y, ...) ~ rhs` takes sampling LHSs, " *
            "not prior addresses; got `$elem`. Write one " *
            "`effect(...)/sd(...)/cor(...) ~ ...` statement per target.")
    end
    Expr(:block, map(lhs.args) do elem
        parse!(Expr(:call, :~, elem, deepcopy(rhs)); info)
    end...)
end

# Public prior-address heads. The user writes the PARAMETER as the head --
# `effect(mu, weight)`, `sd(:, p)`, `cor(:, p)` -- and every head normalises
# onto the ONE internal `effect(<role>, ...)` carrier built by
# `_prior_address`, so the introspection walkers and the SBBRMI resolver keep a
# single representation of an address.
#
# Keeping that carrier internal is also why `sd`/`cor` are NOT exported: the
# macro rewrites the head before `@x` ever evaluates it, so the user never
# needs them bound -- and exporting `cor` would collide with `Statistics.cor`
# for anyone doing `using BayesianRegressionModels, Statistics`.
const _PRIOR_HEADS = (:effect, :sd, :cor, :ranef, :ar, :simplex, :latent, :length_scale)

_is_effect_lhs(x) = any(h -> isxcall(x, h), _PRIOR_HEADS)

# `optional(<address>) ~ prior` (or `= value`) marks a prior that applies only
# where its address exists; `_brm_resolve_optional_priors` drops it before
# lowering when the model provably lacks the address. The marker wraps exactly
# one prior address and rides the `effect` carrier as `optional=true`.
_is_optional_prior_lhs(x) = isxcall(x, :optional)
function _optional_prior_address(x::Expr)
    length(x.args) == 2 && _is_effect_lhs(x.args[2]) || error(
        "@brm: `optional(...)` wraps exactly one prior address, as in " *
        "`optional(effect(mu, weight)) ~ Normal(0, 1)`; got `$x`.")
    x.args[2]
end
# A bare `:` reaches the macro as an ordinary Symbol, so the address vector
# stays homogeneous and no separate Colon carrier is needed. `:` means "the
# default" -- the base layer that a more specific statement overrides.
const _EFFECT_COLON = Symbol(":")
_effect_address_symbol(x::Symbol) = x
_effect_address_symbol(x::QuoteNode) = x.value isa Symbol ? x.value : error(
    "@brm: prior addresses must be symbols, got $(repr(x.value))")
# Collections in a slot are deliberately reserved, not merely unparseable:
# decision `06lrbib` deferred them ("we might want to add it at some later
# point"), so refuse them by NAME rather than letting them fall through to the
# generic bare-symbol error. Adding them later is then purely additive.
_effect_address_symbol(x::Expr) =
    (Meta.isexpr(x, :tuple) || Meta.isexpr(x, :vect)) ? error(
        "@brm: a prior address slot names ONE target or `:`; collections such " *
        "as `$x` are not supported yet. Write one statement per target, or " *
        "use `:` for the default that more specific statements override.") :
    error("@brm: prior addresses must be bare symbols, got $(repr(x))")
_effect_address_symbol(x) = error(
    "@brm: prior addresses must be bare symbols, got $(repr(x))")

# ---- term-address slots -----------------------------------------------------
#
# A prior on a term's OWN parameter names the term the way the formula spells
# it: `sd(log_ka, s(age))`, `simplex(mu, mo1(dose))`, `latent(mu, me(x))`. That
# slot is a call, not a bare symbol, so it gets its own tokenizer and is
# canonicalised to `Symbol("s(age)")` -- the address vector stays homogeneous
# (plain Symbols, one QuoteNode per slot) and the backend matches the key
# against the terms it walks without re-parsing an `Expr`.
#
# Only the SYMBOL arguments are part of the key: `me(x, 0.5)` and `me(x)` name
# the same term, because the measurement SD is a constant OF the term, not part
# of its identity. Keyword arguments are likewise not part of the identity --
# `hsgp(x; k=20)` and `hsgp(x)` are the same term with different tuning. Their
# literal values (symbols, numbers, numeric tuples) still ride in the key after
# a `;`, as QUALIFIERS: the address reaches only terms whose keywords agree, so
# when one predictor carries several terms over the same axes,
# `hsgp(x; cov=:periodic)` picks one of them (decision 1fk9b29).
const _TERM_HEADS =
    (:s, :t2, :mo, :mo1, :me, :interval_censored, :gp, :hsgp, :ar, :dar, :rw, :cdar)

_is_term_address(x) = Meta.isexpr(x, :call) && !isempty(x.args) &&
                      x.args[1] isa Symbol && x.args[1] in _TERM_HEADS

_term_address_literal(value::Union{Number,Bool}) = value
_term_address_literal(value::QuoteNode) =
    value.value isa Symbol ? value.value : nothing
_term_address_literal(value::Expr) =
    Meta.isexpr(value, :tuple) && all(v -> v isa Number, value.args) ?
        Tuple(value.args) : nothing
_term_address_literal(_value) = nothing

function _term_address_qualifiers(x::Expr)
    entries = Pair{Symbol,Any}[]
    for a in x.args[2:end]
        keywords = Meta.isexpr(a, :parameters) ? a.args :
                   Meta.isexpr(a, :kw) ? Any[a] : Any[]
        for kw in keywords
            Meta.isexpr(kw, :kw) && kw.args[1] isa Symbol || continue
            value = _term_address_literal(kw.args[2])
            isnothing(value) || push!(entries, kw.args[1] => value)
        end
    end
    sort!(entries; by=first)
    isempty(entries) ? "" : ";" * join(("$k=$(repr(v))" for (k, v) in entries), ",")
end

function _term_address_key(x::Expr)
    head = x.args[1]::Symbol
    names = Symbol[]
    for a in x.args[2:end]
        (Meta.isexpr(a, :parameters) || Meta.isexpr(a, :kw)) && continue
        a isa Number && continue
        if a isa Symbol
            push!(names, a)
        elseif a isa QuoteNode && a.value isa Symbol
            push!(names, a.value)
        else
            error("@brm: `$head(...)` in a prior address takes plain column " *
                  "names; got $(repr(a)). Address the term the way the formula " *
                  "spells it, minus any numeric or keyword arguments.")
        end
    end
    isempty(names) && error(
        "@brm: `$x` names no data column, so it does not identify a term to " *
        "put a prior on.")
    Symbol(head, "(", join(names, ","), _term_address_qualifiers(x), ")")
end

# Split a term key into the term's identity key and its keyword qualifiers.
function _brm_split_term_key(key::Symbol)
    text = string(key)
    i = findfirst(';', text)
    isnothing(i) && return key, (;)
    parsed = Meta.parse(text)
    params = only(a for a in parsed.args if Meta.isexpr(a, :parameters))
    entries = [kw.args[1]::Symbol => _term_address_literal(kw.args[2])
               for kw in params.args]
    Symbol(text[1:prevind(text, i)], ")"), NamedTuple(entries)
end

# A canonicalised term key is the one address symbol that can contain `(`, so
# it needs no separate carrier to be told apart from a coefficient or group id.
_is_term_key(s::Symbol) = occursin('(', string(s))

# ---- interaction-address slots ----------------------------------------------
#
# A prior on a whole interaction term names the term the way the formula
# spells it: `effect(mu, gen & mathz)`. That slot is a `&` call, not a bare
# symbol, so it gets its own tokenizer and is canonicalised to a sorted
# `&`-joined key -- the address vector stays homogeneous (plain Symbols, one
# QuoteNode per slot) and the backend matches the key against the `&` terms
# it walks without re-parsing an `Expr`.
#
# The operands are SORTED because the backends normalise emission order by
# operand type (a continuous × categorical interaction always emits the
# continuous operand first), which parse time cannot know. Sorting makes
# `a & b` and `b & a` the same key, so the address never depends on surface
# order. Only bare-symbol operands are accepted: a transformed operand such
# as `zscale(m) & g` is still addressable column by column through its
# emitted `int_…` labels (see `popcoefnames`).
_brm_interaction_key(a::Symbol, b::Symbol) =
    Symbol(join(sort!([a, b]), "&"))

_is_interaction_address(x) = Meta.isexpr(x, :call) && length(x.args) == 3 &&
                             x.args[1] === :&

function _interaction_address_key(x::Expr)
    _is_interaction_address(x) || error(
        "@brm: internal interaction-address error on `$x`")
    a, b = x.args[2], x.args[3]
    for operand in (a, b)
        operand isa Symbol || error(
            "@brm: an interaction prior address names two plain columns, " *
            "`effect(<linear_predictor|:>, <a & b>)`; got `$x`. Address a " *
            "transformed operand column by column through its emitted " *
            "`int_…` labels (`popcoefnames(brmi, lp)` lists them).")
    end
    _brm_interaction_key(a, b)
end

# A canonicalised interaction key is the one address symbol that can contain
# `&`: a bare `&` never survives Julia parsing as a Symbol (source `a&b` is
# always the call), so the marker is unambiguous.
_is_interaction_key(s::Symbol) = occursin('&', string(s))

# Split a canonicalised key back into its (sorted) operand pair.
_interaction_key_operands(s::Symbol) =
    Tuple(Symbol(p) for p in split(string(s), '&'))

_prior_slot(x) = _is_term_address(x) ? _term_address_key(x) :
                 _is_interaction_address(x) ? _interaction_address_key(x) :
                 _effect_address_symbol(x)

# Public head -> internal class for the term-parameter heads whose target slot
# is ALWAYS a term. `sd` is absent because it is shared with the random-effect
# surface and is classified by whether its target slot holds a term key.
const _TERM_PRIOR_HEAD_CLASS =
    (; ar=:term_ar, simplex=:term_simplex, latent=:term_latent,
       length_scale=:term_length_scale)

# One representative term per head, so a mis-slotted address gets told what a
# CORRECT `length_scale(...)` looks like rather than a `mo1(...)` example that
# its own head would refuse.
const _TERM_PRIOR_HEAD_EXAMPLE =
    (; ar="dar(time)", simplex="mo1(x)", latent="me(x)",
       length_scale="hsgp(x)")

# Public head + slots -> the internal `effect(...)` address vector.
#
#     effect(<lp|:>, <coefficient|:>)   -> [lp, coefficient]
#     sd(<lp|:>, <ID>[, <coefficient>]) -> [:sd, ID, lp[, coefficient]]
#     cor(:, <ID>)                      -> [:cor, ID]
#     cor(<lp>, <group>)                -> [:cor, group, lp]
#     sd(<lp|:>, <term>[, <block>])     -> [:term_sd, key, lp[, block]]
#     simplex(<lp|:>, <term>)           -> [:term_simplex, key, lp]
#     latent(<lp|:>, <term>)            -> [:term_latent, key, lp]
#     length_scale(<lp|:>, <term>)      -> [:term_length_scale, key, lp]
#     ar(<lp|:>, <term>)                -> [:term_ar, key, lp]
#
# Every slot is one name, one term, or `:`. A trailing `sd` slot may be omitted
# and means `:`; that is a deterministic DEFAULT, not the predictor INFERENCE
# removed with the concise one-slot form -- nothing is searched here and
# nothing can fail to resolve.
#
# `sd` is deliberately ONE head across both surfaces: the marginal scale of a
# random-effect block and the smoothing scale of `s(x)` are both standard
# deviations, and the target slot says which is meant. A group id is a bare
# symbol, a term is a call, so the two can never be confused.
function _prior_address(head::Symbol, args::Vector{Symbol})
    spelling = "$head($(join(args, ", ")))"
    term_slots = findall(_is_term_key, args)
    if !isempty(term_slots) && (head === :effect || head === :cor)
        error("@brm: `$head` addresses a coefficient or a grouping factor, not " *
              "a term's own parameters; got `$spelling`. Use `sd`, `ar`, " *
              "`simplex`, `latent` or `length_scale` with the term in the target slot.")
    elseif !isempty(term_slots) && term_slots != [2]
        error("@brm: a term belongs in the TARGET slot — " *
              "`$head(<linear_predictor|:>, <term>)`; got `$spelling`.")
    end

    # An interaction key addresses population columns, so only `effect` takes
    # one, and only in the coefficient slot.
    interaction_slots = findall(_is_interaction_key, args)
    if !isempty(interaction_slots) && head !== :effect
        error("@brm: `$head` addresses a covariance or a term's own " *
              "parameters, not population interaction columns; got " *
              "`$spelling`. Address the interaction through `effect`, as in " *
              "`effect(<linear_predictor|:>, a & b)`.")
    elseif !isempty(interaction_slots) && interaction_slots != [2]
        error("@brm: an interaction belongs in the COEFFICIENT slot — " *
              "`effect(<linear_predictor|:>, <a & b>)`; got `$spelling`.")
    end

    if haskey(_TERM_PRIOR_HEAD_CLASS, head)
        length(args) == 2 || error(
            "@brm: `$head` takes exactly `$head(<linear_predictor|:>, <term>)`; " *
            "got `$spelling`.")
        _is_term_key(args[2]) || error(
            "@brm: `$head` addresses a term's own parameter, so its target slot " *
            "must be a term as the formula spells it — `$head($(args[1]), " *
            "$(_TERM_PRIOR_HEAD_EXAMPLE[head]))`, not `$spelling`.")
        return Symbol[_TERM_PRIOR_HEAD_CLASS[head], args[2], args[1]]
    end

    if head === :effect
        length(args) == 2 || error(
            "@brm: `effect` takes exactly two slots — " *
            "`effect(<linear_predictor|:>, <coefficient|:>)`; got `$spelling`. " *
            "The concise predictor-inferring form was removed: name the linear " *
            "predictor explicitly, or write `:` for the default.")
        return copy(args)
    elseif head === :sd && length(args) >= 2 && _is_term_key(args[2])
        length(args) in (2, 3) || error(
            "@brm: `sd` on a term takes `sd(<linear_predictor|:>, <term>)` or " *
            "`sd(<linear_predictor|:>, <term>, <block>)`; got `$spelling`.")
        return length(args) == 3 ?
            Symbol[:term_sd, args[2], args[1], args[3]] :
            Symbol[:term_sd, args[2], args[1]]
    elseif head === :sd
        length(args) in (2, 3) || error(
            "@brm: `sd` takes `sd(<linear_predictor|:>, <ID>)` or " *
            "`sd(<linear_predictor|:>, <ID>, <coefficient>)`; got `$spelling`.")
        # Build the full internal 4-slot form, then strip trailing `:` so the
        # common block-wide and per-predictor spellings normalise onto the
        # address shapes the walkers and backend already understand.
        full = Symbol[:sd, args[2], args[1], length(args) == 3 ? args[3] : _EFFECT_COLON]
        while length(full) > 2 && last(full) === _EFFECT_COLON
            pop!(full)
        end
        return full
    elseif head === :ranef
        # The standardized law of a block's effects: `ranef(:, p) ~ TDist(nu)`.
        # Same slot grammar as `sd`, so a per-coefficient law can later reuse
        # the margin addresses; the backend currently accepts block-wide only.
        length(args) in (2, 3) || error(
            "@brm: `ranef` takes `ranef(<linear_predictor|:>, <ID>)` or " *
            "`ranef(<linear_predictor|:>, <ID>, <coefficient>)`; got `$spelling`.")
        full = Symbol[:ranef, args[2], args[1], length(args) == 3 ? args[3] : _EFFECT_COLON]
        while length(full) > 2 && last(full) === _EFFECT_COLON
            pop!(full)
        end
        return full
    elseif head === :cor
        # `cor(:, ID)`, or `cor(lp, g)` for the correlated block of one
        # predictor grouped by a column `g` without an `|ID|`. Whether the
        # target is an ID is known only once the formula is bound, so the
        # backend refuses a predictor on a shared `|ID|` block.
        length(args) == 2 || error(
            "@brm: `cor` takes exactly `cor(<linear_predictor|:>, <ID|group>)`; " *
            "got `$spelling`.")
        return args[1] === _EFFECT_COLON ? Symbol[:cor, args[2]] :
            Symbol[:cor, args[2], args[1]]
    end
    error("@brm: unknown prior-address head `$head`; valid heads are " *
          join(("`$h`" for h in _PRIOR_HEADS), ", ") * ".")
end

# ---- known-value statements -------------------------------------------------
#
# `sd(<lp|:>, <ID>[, <coefficient>]) = value` and `cor(:, <ID>) = R` hold a
# shared random-effect block's marginal scale or correlation matrix at a KNOWN
# value instead of sampling it (nlmixr2's `fix()` on an omega). `=` keeps BRM's
# split: `~` declares something stochastic, `=` a known model value. The
# address grammar is exactly the `~` one.
#
# A known value lowers onto the same `effect(...)` carrier and the same
# generated key as its `~` twin, with the value wrapped in the `_brm_known`
# marker. So it is a prior-address statement everywhere downstream
# (`priors_of`/`structure_of`, `ranef_effect_priors`), a `~` statement on the
# same address is a duplicate, and `merge(structure, priors)` swaps an estimated
# scale for a known one by address.
"""
    _brm_known(value)

Internal marker for a known-value prior-address statement,
`sd(lp, ID) = value` or `cor(:, ID) = R`. Never called.
"""
function _brm_known end

const _KNOWN_VALUE_HEADS = (:sd, :cor)

# The RHS of `f(args...) = value` reaches the macro as a short-form function
# body: `Expr(:block, LineNumberNode, value)`.
function _brm_known_value_expr(rhs, lhs)
    Meta.isexpr(rhs, :block) || return rhs
    values = [a for a in rhs.args if !(a isa LineNumberNode)]
    length(values) == 1 || error(
        "@brm: `$lhs = ...` takes exactly one known value; got " *
        "$(length(values)) expressions")
    only(values)
end

# The operation key of a prior statement with internal `effect(...)` address
# `address`: a generated LOCAL NAME. A term key carries `(`/`)`/`,`, which are
# legal inside a `Symbol` but not in emitted Julia source, so those three
# characters are folded out. An interaction key's `&` is folded the same way,
# to the `_x_` the emitted `int_a_x_b` columns already use. Nothing else is
# touched -- `:` already rode through this key before term addresses existed,
# and rewriting it would move the generated name of every ALREADY-configured
# model.
_brm_prior_statement_key(address) =
    Symbol("__effect__", replace(join(string.(address), "__"),
                                 "(" => "_", ")" => "", "," => "_",
                                 "&" => "_x_"))

function _parse_effect!(lhs::Expr, rhs; info, known::Bool=false, optional::Bool=false)
    head = lhs.args[1]::Symbol
    args = map(_prior_slot, lhs.args[2:end])
    if known
        head in _KNOWN_VALUE_HEADS || error(
            "@brm: only random-effect `sd(...)` and `cor(...)` addresses take a " *
            "known value with `=`; got `$lhs = ...`. Use `~` for a prior" *
            (head === :effect ? ", or `offset(...)` for a known population term." : "."))
        any(_is_term_key, args) && error(
            "@brm: `$lhs = ...` addresses a term's own scale; a known value is " *
            "supported for random-effect blocks only, " *
            "`sd(<linear_predictor|:>, <ID>[, <coefficient>]) = value`. Use `~` " *
            "for a prior on `$lhs`.")
        rhs = Expr(:call, _brm_known, _brm_known_value_expr(rhs, lhs))
    end
    address = _prior_address(head, args)
    # The ADDRESS itself is untouched, so diagnostics stay spelled the way
    # the user wrote them.
    key = _brm_prior_statement_key(address)
    haskey(info.alllocals, key) && error(
        "@brm: duplicate `$head($(join(args, ", ")))` prior statement")
    info.alllocals[key] = :local
    parselocals!(rhs; info, val=:nonlocal)
    quoted_lhs = optional ?
        Expr(:call, :effect, Expr(:parameters, Expr(:kw, :optional, true)),
             map(QuoteNode, address)...) :
        Expr(:call, :effect, map(QuoteNode, address)...)
    :(@n $key = @x $(Expr(:call, :~, quoted_lhs, rhs)))
end

# ---- hyper-predictor statements ---------------------------------------------
#
# `log(length_scale(hsgp(x))) ~ 1 + (1 | g)` defines a distributional linear
# predictor for a GP/HSGP hyperparameter. The `log` wrap is what tells it
# apart from a prior statement (whose RHS is a distribution) and from a
# linked predictor (whose wrap holds a bare name, not a hyper address).
# Only `log` is supported.
#
# The macro only normalises the address syntax here, reusing the prior
# machinery, so the operation keeps a `log(effect(...))` LHS the backend
# matches without re-parsing (same principle as term keys). Every semantic
# refusal (unmatched term, smooths, ungrouped ranef, unsupported term kind,
# distribution RHS, non-term-hyper address) fires at SBBRMI construction,
# where the term context exists and function-wrapped builders stay testable.
function _is_hyper_lhs(x)
    isxcall(x, :log) || return false
    length(x.args) == 2 || return false
    inner = x.args[2]
    Meta.isexpr(inner, :call) || return false
    length(inner.args) >= 2 || return false
    head = inner.args[1]
    head isa Symbol || return false
    head === :length_scale || head === :sd || return false
    length(inner.args) in (2, 3)
end

function _parse_hyper!(lhs, rhs; info)
    inner = lhs.args[2]
    head = inner.args[1]::Symbol
    slots = map(_prior_slot, inner.args[2:end])
    # A one-slot address names just the term (`length_scale(hsgp(x))`); expand
    # to the default-`:` two-slot form so it normalises exactly like its
    # prior-statement twin.
    length(slots) == 1 && (slots = [_EFFECT_COLON, first(slots)])
    address = _prior_address(head, slots)
    key = Symbol("__hyper__", replace(join(string.(address), "__"),
                                      "(" => "_", ")" => "", "," => "_"))
    haskey(info.alllocals, key) && error(
        "@brm: duplicate `$head($(join(string.(inner.args[2:end]), ", ")))` " *
        "hyper-predictor statement")
    info.alllocals[key] = :local
    rhs = rewrite_ranef_ids(rhs)
    parselocals!(rhs; info, val=:nonlocal)
    quoted_lhs = Expr(:call, :log, Expr(:call, :effect, map(QuoteNode, address)...))
    :(@n $key = @x $(Expr(:call, :~, quoted_lhs, rhs)))
end
rewrite_ranef_ids(x) = x
rewrite_ranef_ids(x::Expr) = if isxcall(x, :|) && length(x.args) == 3 &&
                                isxcall(x.args[2], :|) && length(x.args[2].args) == 3
    _quote_ranef_id(x.args[2].args[3], x)
else
    Expr(x.head, rewrite_ranef_ids.(x.args)...)
end
# Dispatches the brms ranef-ID rewrite on the type of the candidate ID.
# Symbol → wrap as QuoteNode and rebuild the three-arg `|`. Anything else
# (already a QuoteNode, an Expr, …) → fall through to the generic recursion.
_quote_ranef_id(id_sym::Symbol, x::Expr) = begin
    inner = x.args[2]
    lhs = rewrite_ranef_ids(inner.args[2])
    g = rewrite_ranef_ids(x.args[3])
    Expr(:call, :|, lhs, QuoteNode(id_sym), g)
end
_quote_ranef_id(_, x::Expr) = Expr(x.head, rewrite_ranef_ids.(x.args)...)

_is_nested_brm(x) = Meta.isexpr(x, :macrocall) &&
                    !isempty(x.args) && x.args[1] === Symbol("@brm")
_contains_nested_brm_syntax(x) = false
_contains_nested_brm_syntax(x::Expr) =
    _is_nested_brm(x) || any(_contains_nested_brm_syntax, x.args)
function _nested_brm_payload(x::Expr)
    _is_nested_brm(x) || error("@brm: expected a nested `@brm(expr)` marker")
    length(x.args) == 3 || error(
        "@brm: nested `@brm` accepts exactly one parenthesized predictor " *
        "expression, got $(max(length(x.args) - 2, 0)) payloads")
    payload = x.args[3]
    if payload isa Expr &&
       (payload.head == :block || payload.head == :(=) || isxcall(payload, :~))
        error("@brm: nested `@brm(...)` marks one predictor expression; " *
              "blocks, assignments, and `~` statements belong in the outer model")
    end
    _contains_nested_brm_syntax(payload) && error(
        "@brm: a nested predictor formula cannot contain another nested `@brm`")
    payload
end

parselocals!(x; kwargs...) = x
function parselocals!(x::Symbol; info, val)
    hasproperty(info, :jointkeys) && x in info.jointkeys && error(
        "@brm: formula name `$x` collides with a generated joint-response " *
        "binding; rename that formula name or one of the outcome columns")
    hasproperty(info, :inventory) && push!(
        val === :nonlocal ? info.inventory.reads :
        val === :maybelocal ? info.inventory.responses : info.inventory.assigned, x)
    get!(info.alllocals, x, val)
end
parselocals!(x::Expr; info, val) = if Meta.isexpr(x, :->)
    # Shield a `do`-block lambda (see `_x`): its params are bound inside the
    # block and its body is verbatim SLIC — registering those symbols as data
    # columns would be wrong. Genuine outer params are registered by their own
    # `~`/`=` lines.
    #
    # But a FREE reference in the cell body to a shared model-data vector (e.g. a
    # generation-interval PMF) IS a capture. Collect the candidate names; `_brm`
    # keeps only those that are real df columns AND are not formula names, then
    # registers them as shared data so the emitted plate cell resolves them the
    # way a `plate` captures a `@slic` data kwarg. Cell params/locals and typos
    # that are not columns fall away in `_brm`.
    hasproperty(info, :captures) && union!(info.captures, setdiff(
        _brm_cell_value_refs!(Set{Symbol}(), x.args[2]),
        _brm_scan_bound_names!(Set{Symbol}(), x.args[2])))
    hasproperty(info, :inventory) && _brm_cell_bound_names!(info.inventory.cell_bound, x)
    x
elseif _is_nested_brm(x)
    parselocals!(_nested_brm_payload(x); info, val)
elseif Meta.isexpr(x, (:call, :kw))
    parselocals!.(x.args[2:end]; info, val)
else
    parselocals!.(x.args; info, val)
end
_n(x::Expr) = begin 
    @assert x.head == :(=)
    lhs, rhs = x.args
    alhs = xassignable(lhs)
    nlhs = xname(alhs)
    :($alhs = $NamedColumn($nlhs, $rhs))
end
xassignable(x::Symbol) = x
xassignable(x::Expr) = if Meta.isexpr(x, (:tuple, :vect))
    Expr(x.head, xassignable.(x.args)...)
elseif isxcall(x, :ragged) && length(x.args) >= 2
    # `ragged(y, group) ~ family(...)` is an observation decorator, not a
    # two-argument assignment target. Keep the formula operation keyed by the
    # logical response (`y`); sbimpl groups its flat backing at the likelihood
    # boundary. This mirrors the one-argument link/decorator spelling below,
    # while retaining `group` on the parsed LHS for backend validation.
    # `into=` rebases an index column a plate cell reads; a response has none.
    any(a -> Meta.isexpr(a, (:parameters, :kw)), x.args) && error(
        "@brm: an observation `ragged(y, group) ~ …` takes no keywords; " *
        "`into=` belongs to a model-scope `ragged(idx, group; into=target)` " *
        "declaration of an index column; got `$(repr(x))`")
    xassignable(x.args[2])
elseif x.head == :call 
    if length(x.args) == 2
        xassignable(x.args[2])
    else
        @warn "Don't know how to handle xassignable($x)!"
        Symbol(x)
    end
else
    dump(x)
    error("Don't know how to handle xassignable($x)!")
end
xname(x::Symbol) = Meta.quot(x)
xname(x::Expr) = if Meta.isexpr(x, (:tuple, :vect))
    Expr(x.head, xname.(x.args)...)
else
    @warn "Don't know how to handle xassignable($x)!"
    Symbol(x)
    # dump(x)
    # error("Don't know how to handle xname($x)!")
end
# The inline kernel body remains ordinary source. Retain its lexical module
# alongside it so a native emitter resolves the same authored callables that
# the Stan emitter resolves, including definitions outside Main.
struct _BRMInlineBody
    expression::Expr
    scope::Module
    # The authored `@plate for` cell this lambda was rewritten from, rendered
    # with the model-scope `ragged(...)` groupings it reads, or `nothing`. Only
    # `show` reads it; every backend lowers `expression`.
    source::Union{String,Nothing}
end
# `_brm_plate_kernel` hands the rendered cell to `_x` as a third `->` argument,
# which `parselocals!` never reads; split it off here so emitters only ever see
# the two-argument lambda.
_BRMInlineBody(expression::Expr, scope::Module) =
    Meta.isexpr(expression, :->, 3) ?
        _BRMInlineBody(Expr(:->, expression.args[1], expression.args[2]), scope,
                       expression.args[3]) :
        _BRMInlineBody(expression, scope, nothing)
_brm_inline_expr(x) = x
_brm_inline_expr(x::_BRMInlineBody) = x.expression
_brm_inline_scope(x) = Main
_brm_inline_scope(x::_BRMInlineBody) = x.scope
Base.show(io::IO, x::_BRMInlineBody) =
    print(io, Base.remove_linenums!(deepcopy(x.expression)))
Base.deepcopy_internal(x::_BRMInlineBody, stack::IdDict) =
    _BRMInlineBody(Base.deepcopy_internal(x.expression, stack), x.scope, x.source)

# `name = constant(expr)` keeps `expr` unevaluated. The builder binds each name
# the expression reads, either to a field of the data or to a value in the
# builder's scope, and a backend evaluates it once or restores the fitted value
# on replay (`_brm_resolve_constants`). `evaluate` takes those inputs in order.
struct _BRMConstantSpec{F}
    expression::Any
    inputs::Tuple{Vararg{Symbol}}
    evaluate::F
end

# A constant input that is not a field of the data: a value of that name in
# the builder's scope (a Julia constant such as `pi`, or a captured local).
struct _BRMConstantScopeValue{T}
    value::T
end

# Which names does `constant(expr)` read? Every symbol in a value position,
# except the names an anonymous function or a generator binds. Call and
# broadcast heads are functions resolved in the builder's scope, and keyword
# names, comparison operators and `end` inside indexing are syntax.
function _brm_constant_inputs(expr)
    found = Symbol[]
    _brm_constant_inputs!(found, expr, ())
    unique!(found)
end
_brm_constant_inputs!(found, _x, _bound) = found
_brm_constant_inputs!(found, x::Symbol, bound) =
    (x in bound || x === :end || x === :begin || push!(found, x); found)
function _brm_constant_inputs!(found, x::Expr, bound)
    walk(args, bound=bound) = foreach(a -> _brm_constant_inputs!(found, a, bound), args)
    if x.head === :call
        walk(x.args[2:end])
    elseif x.head === :. && length(x.args) == 2 && Meta.isexpr(x.args[2], :tuple)
        walk(x.args[2].args)                       # `f.(args...)`
    elseif x.head === :.
        walk(x.args[1:1])                          # `a.b` reads `a`
    elseif x.head === :kw
        walk(x.args[2:2])
    elseif x.head === :comparison
        walk(x.args[1:2:end])
    elseif x.head === :macrocall
        walk(x.args[3:end])
    elseif x.head === :->
        params = Meta.isexpr(x.args[1], :tuple) ? x.args[1].args : (x.args[1],)
        walk(x.args[2:2], (bound..., (p for p in params if p isa Symbol)...))
    elseif x.head === :generator
        inner = bound
        for spec in x.args[2:end]
            cond = nothing
            if Meta.isexpr(spec, :filter)
                cond, spec = spec.args[1], spec.args[2:end]
            else
                spec = (spec,)
            end
            for s in spec
                Meta.isexpr(s, :(=)) || continue
                walk(s.args[2:2], inner)
                s.args[1] isa Symbol && (inner = (inner..., s.args[1]))
            end
            isnothing(cond) || walk((cond,), inner)
        end
        walk(x.args[1:1], inner)
    else
        walk(x.args)
    end
    found
end

# Bind one input of a constant when the builder runs: the data's field of that
# name, or else the builder's own binding of it. A data field missing from
# this frame stays unbound; only evaluating the constant needs it, and replay
# restores the fitted value instead.
function _brm_constant_input(df, key::Symbol, input::Symbol, scope)
    hasproperty(df, input) && return NamedColumn(input, DataColumn(getproperty(df, input)))
    value = try
        scope()
    catch err
        err isa UndefVarError && err.var === input || rethrow()
        return NamedColumn(input, MissingColumn())
    end
    value isa NamedColumn || return _BRMConstantScopeValue(value)
    parent(value) isa Union{DataColumn,MissingColumn} && return NamedColumn(input, parent(value))
    error("@brm: `$key = constant(...)` reads `$input`, a model value. A " *
          "constant reads only fields of the data and Julia values; it is " *
          "computed once, before anything is sampled.")
end

function _parse_constant!(lhs, rhs; info)
    lhs isa Symbol || error(
        "@brm: a constant is declared as `name = constant(expr)`; got " *
        "`$(lhs) = $(rhs)`")
    args = rhs.args[2:end]
    (length(args) == 1 && !Meta.isexpr(args[1], (:parameters, :kw))) || error(
        "@brm: `constant` takes one expression, as in " *
        "`$lhs = constant(mean(x))`; got `$rhs`")
    expr = only(args)
    inputs = _brm_constant_inputs(expr)
    lhs in inputs && error("@brm: `$lhs = constant(...)` reads `$lhs` itself")
    hasproperty(info, :inventory) && union!(info.inventory.constant_reads, inputs)
    parselocals!(lhs; info, val=:local)
    evaluate = Expr(:->, Expr(:tuple, inputs...), expr)
    bound = [:($_brm_constant_input(__df__, $(QuoteNode(lhs)), $(QuoteNode(s)), () -> $s))
             for s in inputs]
    spec = :($_BRMConstantSpec($(QuoteNode(expr)), ($(map(QuoteNode, inputs)...),), $evaluate))
    :(@n $lhs = $ExprColumn($assign, $(QuoteNode(lhs)), $ExprColumn($constant, $spec, $(bound...))))
end

_x(x) = x
_x(x::Symbol) = x
_x(x::Expr) = if x.head == :call
    Expr(:call, ExprColumn, _x.(x.args)...) |> fixcall
elseif _is_nested_brm(x)
    Expr(:call, NestedPredictorFormula, _x(_nested_brm_payload(x)))
elseif x.head ==  :||
    Expr(:call, ExprColumn, doublepipe, _x.(x.args)...)
elseif x.head == :do
    # At macro-expansion time `term(args...) do params ... end` is
    # `Expr(:do, term(args...), lambda)` (the lambda-first-arg lowering happens
    # LATER). Fold the lambda in as the term's FIRST positional arg, then process
    # the resulting call normally — so the term ExprColumn carries the do-block.
    call, lambda = x.args
    if Meta.isexpr(call, :call)
        newargs = copy(call.args)
        pos = (length(newargs) >= 2 && Meta.isexpr(newargs[2], :parameters)) ? 3 : 2
        insert!(newargs, pos, lambda)
        _x(Expr(:call, newargs...))
    else
        Expr(x.head, _x.(x.args)...)
    end
elseif x.head == :->
    # The folded-in do-block lambda: capture it VERBATIM as an Expr rather than
    # recursing — its body is SLIC to splice into the emitted plate, NOT more @brm
    # formula. No existing formula uses a bare lambda (regression-safe).
    Expr(:call, _BRMInlineBody, Meta.quot(x),
        Expr(:macrocall, Symbol("@__MODULE__"), LineNumberNode(0)))
else
    Expr(x.head, _x.(x.args)...)
end
"""
    Data(df)

Wrapper that exposes every column of the underlying DataFrame as a
[`NamedColumn`](@ref) over a [`DataColumn`](@ref). Used by [`@brm`](@ref)
to lift `df` columns into the formula's column-type universe.
"""
struct Data{P}
    parent::P
end
Base.parent(d::Data) = getfield(d, :parent)
Base.hasproperty(d::Data, x::Symbol) = hasproperty(parent(d), x)
Base.getproperty(d::Data, x::Symbol) = NamedColumn(x, DataColumn(getproperty(parent(d), x)))
data(x) = Data(x)

"""
    MaybeData(df)

Like [`Data`](@ref), but for columns that may not be present in the
dataframe. Missing columns surface as [`NamedColumn`](@ref) wrapping a
[`MissingColumn`](@ref) — i.e. a parameter to be sampled rather than
observed.
"""
struct MaybeData{P}
    parent::P
end
Base.parent(d::MaybeData) = getfield(d, :parent)
Base.hasproperty(d::MaybeData, x::Symbol) = hasproperty(parent(d), x)
Base.getproperty(d::MaybeData, x::Symbol) = NamedColumn(x, hasproperty(d, x) ? DataColumn(getproperty(parent(d), x)) : MissingColumn())

"""
    maybedata(df) -> MaybeData

Convenience constructor for [`MaybeData`](@ref). Used by [`@brm`](@ref)
to surface response-side names that may be data (observed) or
parameters (sampled).
"""
maybedata(x) = MaybeData(x)

"""
    capturedata(df, names::Tuple) -> NamedTuple

Bind the subset of `names` that ARE columns of `df` as [`NamedColumn`](@ref)s
over [`DataColumn`](@ref)s, skipping absent names. Used by [`@brm`](@ref) to
register a `kernel(...)` do-block's free references to SHARED model-data vectors
(e.g. a generation-interval / delay PMF) as data, so the emitted plate cell
captures them the same way a `plate` captures a `@slic` data kwarg. A name that
is not a column is dropped here and left for the backend to resolve (a builtin,
or a loud "not found" at trace) — so this never turns a typo or a cell-local
into silent data.
"""
capturedata(df, names::Tuple) = (;
    (n => NamedColumn(n, DataColumn(getproperty(df, n)))
     for n in names if hasproperty(df, n))...)

# Collect bare Symbols in VALUE positions inside a `do`-block cell body — the
# candidate names a kernel cell references freely. Call and macro heads (`exp`,
# `renewal`), `:kw` names and the field half of `a.b` are never values. A
# broadcast call `f.(args...)` shares the `:.` head with field access, but its
# second argument is a tuple of value expressions, visited exactly like an
# ordinary call's arguments; its callee is skipped like a call head. Callers
# keep only names they can resolve — `_brm` real df columns (via `capturedata`)
# that are not formula names, the RK kernel planner model names — so do-block
# params and cell-locals fall away there and no exclusion set is needed here.
_brm_is_broadcast_call(x) = Meta.isexpr(x, :., 2) && Meta.isexpr(x.args[2], :tuple)
_brm_cell_value_refs!(acc, x) = acc
_brm_cell_value_refs!(acc, x::Symbol) = (push!(acc, x); acc)
function _brm_cell_value_refs!(acc, x::Expr)
    values = if Meta.isexpr(x, :->)
        (x.args[2],)
    elseif Meta.isexpr(x, (:call, :macrocall, :kw))
        x.args[2:end]
    elseif _brm_is_broadcast_call(x)
        x.args[2].args
    elseif Meta.isexpr(x, :.)
        (x.args[1],)
    else
        x.args
    end
    foreach(value -> _brm_cell_value_refs!(acc, value), values)
    acc
end

"""
    AbstractColumn

Supertype of every column-shape token the `@brm` parser produces.
Subtypes include [`MissingColumn`](@ref), [`DataColumn`](@ref),
[`NamedColumn`](@ref), [`ExprColumn`](@ref), `MultiMembershipTerm`,
[`LikelihoodColumn`](@ref), and [`MaterializedColumn`](@ref).

Backend dispatchers (`vmeta_sampling_rhs`, `_sb_emit!`, the
introspection walkers) branch on this hierarchy rather than on
`Symbol` tags.
"""
abstract type AbstractColumn end

"""
    MissingColumn()

LHS marker for response columns that aren't in the dataframe — the
backend treats them as parameters rather than observations.
"""
struct MissingColumn <: AbstractColumn end

"""
    DataColumn(vec)

Wraps an observed vector from the dataframe. Surfaces as the leaf of a
[`NamedColumn`](@ref) for any data-backed predictor or response.
"""
struct DataColumn{P} <: AbstractColumn
    parent::P
end
Base.parent(d::DataColumn) = getfield(d, :parent)

"""
    NestedPredictorFormula(parent)

Internal parser node for an explicit nested `@brm(expr)` predictor formula.
The outer builder consumes every such node before returning a [`BRMI`](@ref),
replacing it with ordinary named `~` operations. It is intentionally not
exported and never reaches a backend.
"""
struct NestedPredictorFormula{P} <: AbstractColumn
    parent::P
end
Base.parent(x::NestedPredictorFormula) = getfield(x, :parent)

"""
    NamedColumn(name, parent)

Assigns a formula-local symbol `name` to its `parent` column. Every
`lhs ~ rhs` and `lhs = rhs` line produces a `NamedColumn` keyed by
`lhs`. Access via [`name`](@ref) / `parent`.
"""
struct NamedColumn{N,P} <: AbstractColumn
    name::N
    parent::P
end

"""
    name(x::NamedColumn) -> Symbol

Return the formula-local name assigned to `x` (i.e. the LHS of the
`~` / `=` line that produced it).
"""
name(x::NamedColumn) = getfield(x, :name)
Base.parent(x::NamedColumn) = getfield(x, :parent)

"""
    _check_term_kwargs(f, kwargs) -> nothing

Per-term keyword validation, run for EVERY [`ExprColumn`](@ref) as it is
constructed — i.e. while `@brm` builds the model, which is the first moment a
consumer can be told anything at all. The default method accepts everything;
a term that has retired part of its keyword surface adds a method next to the
term itself (see `kernel(...)` in `sbimpl.jl`).

Why here and not only in the backend: `@brm` is a pure parser, so a keyword it
does not recognise is captured into [`getkwargs`](@ref) unexamined and only the
backend emitter ever objects — which means a model written with retired syntax
constructs cleanly and complains solely once it is LOWERED (`SBBRMI(...)`).
A consumer whose compatibility gate stops at BRMI construction then sees
retired syntax pass silently.
"""
_check_term_kwargs(f, kwargs) = nothing

"""
    ExprColumn(f, args...; kwargs...)

Represents an `f(args...; kwargs...)` formula RHS — the leaves the
`@brm` parser builds for every `:call` Expr (e.g. `Normal(loc, err)`,
`1 + a + (1|g)`, `mo(c)`). Access via [`getf`](@ref) / [`getargs`](@ref)
/ [`getkwargs`](@ref) / [`getop`](@ref).

Construction runs `_check_term_kwargs` so a term can reject a retired
keyword at the `@brm` call site rather than in the backend.
"""
struct ExprColumn{F,A<:Tuple,K<:NamedTuple} <: AbstractColumn
    f::F
    args::A
    kwargs::K
    ExprColumn(f, args...; kwargs...) =
        (_check_term_kwargs(f, (;kwargs...));
         new{typeof(f),typeof(args),typeof((;kwargs...))}(f,args,(;kwargs...)))
    ExprColumn(f::Type, args...; kwargs...) =
        (_check_term_kwargs(f, (;kwargs...));
         new{Type{f},typeof(args),typeof((;kwargs...))}(f,args,(;kwargs...)))
end

"""
    MultiMembershipTerm

Typed internal column node produced by `mm(...)`. It deliberately does not use
the generic `ExprColumn` representation: grouping columns, optional weight
columns, and the normalization policy are structural parts of one random-effect
term and must survive preprocessing/replay together.
"""
struct MultiMembershipTerm{G<:Tuple,W} <: AbstractColumn
    groups::G
    weights::W
    normalize::Bool
end

"""
    JointResponseColumn(columns)

Internal ordered collection of the data-backed columns on a vector likelihood
LHS. It survives parsing and replay as one response node so a backend cannot
mistake the declaration for several independent scalar likelihoods.
"""
struct JointResponseColumn{C<:Tuple} <: AbstractColumn
    columns::C
    impute::Bool
    function JointResponseColumn(columns::C, impute::Bool=false) where {C<:Tuple}
        length(columns) >= 2 || throw(ArgumentError(
            "a joint response needs at least two columns"))
        all(c -> c isa NamedColumn, columns) || throw(ArgumentError(
            "every joint-response entry must be a named column"))
        names = map(name, columns)
        length(unique(names)) == length(names) || throw(ArgumentError(
            "joint-response columns must be unique"))
        new{C}(columns, impute)
    end
end

joint_response_columns(x::JointResponseColumn) = getfield(x, :columns)
joint_response_names(x::JointResponseColumn) = map(name, joint_response_columns(x))
_joint_response_operation_key(names) =
    Symbol("brm_joint_", join(string.(names), "__"))
function _joint_response_reserved_keys(names; impute=false)
    key = _joint_response_operation_key(names)
    base = (key, Symbol(key, :_observed), Symbol(key, :_n), Symbol(key, :_means))
    impute || return base
    (base..., Symbol(key, :_obs), Symbol(key, :_Jobs), Symbol(key, :_Jmis),
     Symbol(key, :_completed))
end
_joint_response_data_key(x::JointResponseColumn) =
    Symbol(_joint_response_operation_key(joint_response_names(x)), :_observed)
_joint_response_n_key(x::JointResponseColumn) =
    Symbol(_joint_response_operation_key(joint_response_names(x)), :_n)

function MultiMembershipTerm(groups...; weights=nothing, normalize=true)
    length(groups) >= 2 || throw(ArgumentError(
        "`mm(...)` requires at least two grouping columns; got $(length(groups))"))
    all(g -> g isa NamedColumn, groups) || throw(ArgumentError(
        "every positional argument to `mm(...)` must be a data-column name"))
    normalize isa Bool || throw(ArgumentError(
        "`mm(...; normalize=...)` expects `true` or `false`, got $(repr(normalize))"))
    if weights !== nothing
        weights isa Tuple || throw(ArgumentError(
            "`mm(...; weights=...)` expects a tuple with one weight column per group"))
        length(weights) == length(groups) || throw(ArgumentError(
            "`mm(...)` has $(length(groups)) groups but $(length(weights)) weight columns"))
        all(w -> w isa NamedColumn, weights) || throw(ArgumentError(
            "every entry of `mm(...; weights=(...))` must be a data-column name"))
    end
    MultiMembershipTerm(tuple(groups...), weights, normalize)
end

# Specialised outer constructor: `_x` continues to lower every call through
# `ExprColumn(...)`, while dispatch turns only `mm(...)` into the typed node.
ExprColumn(::typeof(mm), groups...; kwargs...) = MultiMembershipTerm(groups...; kwargs...)

getf(::MultiMembershipTerm) = mm
getargs(x::MultiMembershipTerm) = getfield(x, :groups)
getargs(x::MultiMembershipTerm, n) =
    (rv = getargs(x); @assert length(rv) == n; rv)
getkwargs(x::MultiMembershipTerm) =
    (; weights=getfield(x, :weights), normalize=getfield(x, :normalize))
getop(::MultiMembershipTerm) = mm

"""
    getf(x::ExprColumn) -> F

The callable head of an [`ExprColumn`](@ref) — e.g. `Normal`, `+`, `~`,
`assign`, `mo`. Sole dispatch tag for backend emitters.
"""
getf(x::ExprColumn) = getfield(x, :f)

"""
    getargs(x::ExprColumn) -> Tuple
    getargs(x::ExprColumn, n) -> Tuple

The positional args of an [`ExprColumn`](@ref). The two-arg form asserts
`length == n`. Plus helpers for `+`-flattening:
`getargs(+, x::ExprColumn{typeof(+)})` returns the summands;
`getargs(+, x)` wraps non-`+` shapes as a 1-tuple.
"""
getargs(x::ExprColumn) = getfield(x, :args)
getargs(x::ExprColumn, n) = (rv = getargs(x); @assert length(rv) == n; rv)
getargs(::typeof(+), x::ExprColumn{typeof(+)}) = getargs(x)
getargs(::typeof(+), x::ExprColumn) = (x,)
getargs(::typeof(+), x) = (x,)

"""
    getkwargs(x::ExprColumn) -> NamedTuple

The keyword args of an [`ExprColumn`](@ref). E.g. `hsgp(x; k=20, c=1.5)`
exposes `(; k=20, c=1.5)`.
"""
getkwargs(x::ExprColumn) = getfield(x, :kwargs)

# Syntax walkers visit an expression's positional, then keyword, children
# through these rather than `foreach`/`any` over its argument tuples. A node's
# type spells its whole subtree (a NamedColumn embeds its definition), so tuple
# higher-order functions and walker methods specialized on it compile again
# for every new model structure. With the walker methods themselves
# non-specializing, each walk compiles once.
Base.@nospecializeinfer function _brm_foreach_child(f, @nospecialize(node::ExprColumn))
    _brm_foreach_item(f, getargs(node))
    _brm_foreach_item(f, getkwargs(node))
end
Base.@nospecializeinfer function _brm_foreach_item(f, @nospecialize(items::Union{Tuple,NamedTuple}))
    for index in 1:nfields(items)
        f(getfield(items, index))
    end
    nothing
end
Base.@nospecializeinfer function _brm_any_child(f, @nospecialize(node::ExprColumn))
    args, kwargs = getargs(node), getkwargs(node)
    for index in 1:nfields(args)
        f(getfield(args, index)) && return true
    end
    for index in 1:nfields(kwargs)
        f(getfield(kwargs, index)) && return true
    end
    false
end

"""
    getop(x) -> Symbol_or_Function

Symbolic operator name for an [`ExprColumn`](@ref): `:||` for the
[`doublepipe`](@ref) marker, `:(=)` for [`assign`](@ref), otherwise
falls through to [`getf`](@ref). Used by the HTML renderer to print
formula RHS with the original operator surface.
"""
getop(x) = getf(x)
getop(::ExprColumn{typeof(doublepipe)}) = :||
getop(::ExprColumn{typeof(assign)}) = :(=)

# Body-level INDEX / FIELD access on a model value TRANSPILES to Stan.
#
# A body `lhs = rhs` is macro-processed, but `_x` leaves `.` / `[]` as ordinary
# runtime operations on the evaluated column object — it deliberately cannot tell
# `state.field` on a model value from `Module.func` on a module (both are
# `Expr(:., …)`); only the runtime RECEIVER type can. So the lowering lives here:
# a `getindex` / `getproperty` on a `NamedColumn` / `ExprColumn` (the two column
# types that carry model values) builds an `ExprColumn` that `_sb_scalar_expr`
# renders as Stan `x[i]` / `x.field`. StanBlocks already lowers `getindex` /
# `getproperty` on Stan values (arrays, `NamedTuple` struct returns), so this gives
# `@slic` PARITY for consuming a `@deffun` struct/array return directly —
# `state = scan(...); y ~ f(state.population_total)` — instead of hand-rolled
# accessor `@deffun`s. `getproperty` delegates the struct's OWN fields so internal
# access (which uses `getfield` / `name` / `parent` / `getf`, never dot) is
# untouched; every other name becomes a Stan field access.
Base.getindex(x::NamedColumn, i...) = ExprColumn(getindex, x, i...)
Base.getindex(x::ExprColumn, i...) = ExprColumn(getindex, x, i...)
function Base.getproperty(x::NamedColumn, f::Symbol)
    (f === :name || f === :parent) && return getfield(x, f)
    ExprColumn(getproperty, x, QuoteNode(f))
end
function Base.getproperty(x::ExprColumn, f::Symbol)
    (f === :f || f === :args || f === :kwargs) && return getfield(x, f)
    ExprColumn(getproperty, x, QuoteNode(f))
end

"""
    LikelihoodColumn(parent, rhs)

Pairs an observed-response `parent` ([`NamedColumn`](@ref) over a
[`DataColumn`](@ref)) with its distributional `rhs`. vimpl's
`llikelihood!` walks these.
"""
struct LikelihoodColumn{P,R} <: AbstractColumn
    parent::P
    rhs::R
end
Base.parent(d::LikelihoodColumn) = getfield(d, :parent)
rhs(d::LikelihoodColumn) = getfield(d, :rhs)
maybedists(lhs::AbstractColumn, x::AbstractColumn) = LikelihoodColumn(lhs, x)

"""
    BRMI(; ops...)
    BRMI(operations::NamedTuple)

BRM Intermediate — the parsed formula. `operations` is a NamedTuple
mapping each LHS name to its [`NamedColumn`](@ref). The two backends
[`VBRMI`](@ref) and [`SBBRMI`](@ref) walk this representation.

A `BRMI` is what [`@brm`](@ref) ultimately returns (either directly,
when `df` is baked in, or by calling the gensym'd model function on a
dataframe). Introspect via [`outcomes`](@ref) / [`linear_predictors`](@ref)
/ [`predictors`](@ref) / [`grouping_factors`](@ref) /
[`dependencies`](@ref) / [`data_columns`](@ref).
"""
struct BRMI{O<:NamedTuple}
    operations::O
end
BRMI(;kwargs...) = BRMI((;kwargs...))

# Planning reads a BRMI's operations as syntax. Their NamedTuple type spells
# every formula name and expression tree, so `values`, `pairs`, `keys`,
# indexing, tuple higher-order functions and closures capturing the BRMI each
# compile again for every model of a corpus. These accessors use builtins and
# the type as a value instead, so each compiles once for all models. Leaf
# dispatch still sees every operation's own value.
Base.@nospecializeinfer function _brm_operation_entries(@nospecialize(brmi::BRMI))
    operations = getfield(brmi, :operations)
    T = typeof(operations)
    entries = Vector{Pair{Symbol,Any}}(undef, nfields(operations))
    for index in eachindex(entries)
        entries[index] = Pair{Symbol,Any}(fieldname(T, index), getfield(operations, index))
    end
    entries
end
Base.@nospecializeinfer function _brm_operation_values(@nospecialize(brmi::BRMI))
    operations = getfield(brmi, :operations)
    values = Vector{Any}(undef, nfields(operations))
    for index in eachindex(values)
        values[index] = getfield(operations, index)
    end
    values
end
Base.@nospecializeinfer function _brm_operation_names(@nospecialize(brmi::BRMI))
    T = typeof(getfield(brmi, :operations))
    names = Vector{Symbol}(undef, fieldcount(T))
    for index in eachindex(names)
        names[index] = fieldname(T, index)
    end
    names
end
Base.@nospecializeinfer _brm_has_operation(@nospecialize(brmi::BRMI), key::Symbol) =
    hasfield(typeof(getfield(brmi, :operations)), key)
Base.@nospecializeinfer _brm_operation(@nospecialize(brmi::BRMI), key::Symbol) =
    getfield(getfield(brmi, :operations), key)

"""
    Base.merge(a::BRMI, b::BRMI, rest::BRMI...) -> BRMI

Compose two or more parsed models into one, later models overriding earlier
ones where they name the same operation. A `BRMI` is a formula-ordered
`NamedTuple` of operations keyed by LHS name — a linear predictor / parameter
name (`mu`, `sigma`), or a prior address such as `effect(...)` / `sd(...)` /
`cor(...)` (see [`@brm`](@ref)) — so merging is the corresponding `NamedTuple`
merge: `a`'s operations keep their order, `b`'s new operations are appended,
and any operation `b` shares with `a` (same key) takes `b`'s value.

This makes priors a first-class, separable fragment. Build the structural
model with one `@brm` block and its priors with another, then compose:

```julia
model  = @brm df begin
    log_ka ~ 1 + x + (1 | pk | subject)
    y ~ Normal(log_ka, 0.2)
end
priors = @brm df begin
    effect(log_ka, Intercept) ~ Normal(log(1 / 8), 0.8)
    sd(:, pk) ~ Exponential(1)
end
fitted = Base.merge(model, priors)
```

Because a prior address IS its operation key, a post-hoc prior overrides an
inline one at the same address (last-wins) rather than colliding the way two
same-address statements in ONE block do. `merge` is a pure structural
combine: an address is validated against the model it configures at backend
lowering (`SBBRMI(...)`), exactly as for an inline prior — never here.

Structural fragments compose the same way, in any order. A fragment that only
READS a name another fragment defines (`y ~ Normal(mu, sigma)` without a `mu`
formula) carries a bare placeholder for it, and a placeholder never overrides
a definition: `merge(lp_fragment, obs_fragment)` and
`merge(obs_fragment, lp_fragment)` both keep `mu`'s formula.
"""
Base.merge(a::BRMI, b::BRMI) = BRMI(Base.merge(a.operations,
    _brm_drop_shadowing_placeholders(b.operations, a.operations)))
Base.merge(a::BRMI, b::BRMI, rest::BRMI...) =
    foldl(Base.merge, rest; init=Base.merge(a, b))

# A builder binds every name it reads but does not define to a `MissingColumn`
# placeholder; such an entry is a reference, not a declaration.
_brm_is_placeholder(op) = op isa NamedColumn && parent(op) isa MissingColumn

# `incoming`'s operations minus the placeholders that would shadow a name
# `existing` already carries, in `incoming`'s order.
_brm_drop_shadowing_placeholders(incoming::NamedTuple, existing::NamedTuple) =
    (; (key => op for (key, op) in pairs(incoming)
        if !(_brm_is_placeholder(op) && haskey(existing, key)))...)

# Nested predictor formulas are normalised at builder execution time, after
# data columns have values (categorical K is therefore known) but before a BRMI
# becomes public. Downstream introspection, replay, descriptors, and both
# backends see only the existing explicit `eta ~ formula` representation.
_brm_fit_levels(raw::AbstractVector) = sort(unique(raw))

_nested_contains(@nospecialize(_value)) = false
_nested_contains(::NestedPredictorFormula) = true
_nested_contains(@nospecialize(x::NamedColumn)) = _nested_contains(parent(x))
_nested_contains(@nospecialize(x::ExprColumn)) = _brm_any_child(_nested_contains, x)

_nested_path_token(kind::Symbol, value) = Symbol(kind, value)
_nested_predictor_name(target::Symbol, path::Tuple) =
    Symbol(target, :_nested_, join(string.(path), "_"))

function _nested_predictor!(pending, occupied, target, path, formula)
    n = _nested_predictor_name(target, path)
    n in occupied && error(
        "@brm: generated nested-predictor name `$n` collides with an existing " *
        "formula/data name. Rename that binding or write the predictor as an " *
        "explicit top-level `eta ~ ...` statement.")
    lhs = NamedColumn(n, MissingColumn())
    op = NamedColumn(n, ExprColumn(~, lhs, formula))
    push!(pending, n => op)
    push!(occupied, n)
    op
end

function _nested_rewrite!(x::NestedPredictorFormula, target, path,
                          pending, occupied)
    _nested_predictor!(pending, occupied, target, path, parent(x))
end
function _nested_rewrite!(x::ExprColumn, target, path, pending, occupied)
    args = getargs(x)
    new_args = ntuple(length(args)) do i
        token = _nested_path_token(:arg, i)
        _nested_rewrite!(args[i], target, (path..., token), pending, occupied)
    end
    kwargs = getkwargs(x)
    kw_names = keys(kwargs)
    new_kw_values = ntuple(length(kw_names)) do i
        key = kw_names[i]
        token = _nested_path_token(:kw_, key)
        _nested_rewrite!(getfield(kwargs, key), target, (path..., token),
                         pending, occupied)
    end
    new_kwargs = NamedTuple{kw_names}(new_kw_values)
    ExprColumn(getf(x), new_args...; new_kwargs...)
end
_nested_rewrite!(x, _target, _path, _pending, _occupied) = x

_nested_observed_values(lhs::NamedColumn) =
    parent(lhs) isa DataColumn ? parent(parent(lhs)) : nothing
function _nested_observed_values(lhs::ExprColumn)
    args = getargs(lhs)
    length(args) == 1 ? _nested_observed_values(only(args)) : nothing
end
_nested_observed_values(lhs::ExprColumn{typeof(ragged)}) =
    _nested_observed_values(first(getargs(lhs)))
_nested_observed_values(_) = nothing

function _nested_rewrite_categorical!(target, lhs, rhs, pending, occupied)
    args = getargs(rhs)
    kwargs = getkwargs(rhs)
    has_marker = any(_nested_contains, args) || any(_nested_contains, values(kwargs))
    has_marker || return rhs
    (length(args) == 1 && only(args) isa NestedPredictorFormula && isempty(kwargs)) ||
        error("@brm: concise `CategoricalLogit` requires exactly one direct " *
              "marker: `CategoricalLogit(@brm(formula))`. Do not mix marked " *
              "and explicit class predictors.")

    raw = _nested_observed_values(lhs)
    raw isa AbstractVector || error(
        "@brm: `CategoricalLogit(@brm(...))` needs an observed outcome vector " *
        "to determine its non-reference classes; use explicit predictors when " *
        "the outcome schema is unavailable.")
    levels = _brm_fit_levels(raw)
    length(levels) >= 2 || error(
        "@brm: `CategoricalLogit(@brm(...))` needs at least two observed " *
        "outcome levels (got $(length(levels))).")

    formula = parent(only(args))
    refs = ntuple(length(levels) - 1) do j
        class_token = _nested_path_token(:class, j + 1)
        _nested_predictor!(pending, occupied, target,
                           (_nested_path_token(:arg, 1), class_token), formula)
    end
    ExprColumn(getf(rhs), refs...)
end

function _nested_rewrite_likelihood!(target, lhs, rhs::ExprColumn,
                                     pending, occupied)
    if isdefined(@__MODULE__, :CategoricalLogit) && getf(rhs) === CategoricalLogit
        return _nested_rewrite_categorical!(target, lhs, rhs, pending, occupied)
    end
    _nested_rewrite!(rhs, target, (), pending, occupied)
end
_nested_rewrite_likelihood!(target, _lhs, rhs, pending, occupied) =
    rhs isa NestedPredictorFormula ? error(
        "@brm: nested `@brm(...)` must be an argument of the observed " *
        "family, for example `y ~ Normal(@brm(1 + x), 1)`.") : rhs

function _expand_nested_top!(key, value, pending, occupied)
    value isa NamedColumn || return value
    op = parent(value)
    op isa ExprColumn || return value
    _nested_contains(op) || return value
    getf(op) === (~) || error(
        "@brm: nested `@brm(...)` is only valid inside an observed likelihood " *
        "argument, not inside `$key = ...`.")
    lhs, rhs = getargs(op, 2)
    isnothing(_nested_observed_values(lhs)) && error(
        "@brm: nested `@brm(...)` is only valid inside an observed likelihood " *
        "argument, not the linear predictor `$key ~ ...`.")
    new_rhs = _nested_rewrite_likelihood!(key, lhs, rhs, pending, occupied)
    NamedColumn(name(value), ExprColumn(~, lhs, new_rhs))
end

# Builder finalization runs once per model. Merging do-block captures and
# expanding nested formulas read the operations as syntax, so only constructing
# the model's own NamedTuple and BRMI specializes on it.
Base.@nospecializeinfer function _brm_finalize_operations(
        @nospecialize(operations::NamedTuple), @nospecialize(captures::NamedTuple))
    isempty(captures) || (operations = Base.merge(operations, captures))
    _expand_nested_predictor_formulas(operations)
end

Base.@nospecializeinfer function _expand_nested_predictor_formulas(
        @nospecialize(operations::NamedTuple))
    brmi = BRMI(operations)
    entries = _brm_operation_entries(brmi)
    any(entry -> _nested_contains(last(entry)), entries) || return brmi
    occupied = Set{Symbol}(first.(entries))
    expanded = Pair{Symbol,Any}[]
    for (key, value) in entries
        pending = Pair{Symbol,Any}[]
        new_value = _expand_nested_top!(key, value, pending, occupied)
        append!(expanded, pending)
        push!(expanded, key => new_value)
    end
    names = Tuple(first.(expanded))
    values_ = Tuple(last.(expanded))
    BRMI(NamedTuple{names}(values_))
end

# A BRMI prints as the model source: its statements, one per line, in formula
# order and in their authored spelling. Data columns and the placeholders a
# fragment binds for names it only reads are not statements and do not print,
# so `string(structure_of(m))` and `string(priors_of(m))` are the model's two
# source cards.
Base.show(io::IO, (;operations)::BRMI) = begin
    started = false
    for (key, value::NamedColumn) in pairs(operations)
        _brm_is_statement(parent(value)) || continue
        started && print(io, "\n")
        started = true
        _show_top(io, key, parent(value))
    end
end
_brm_is_statement(op) = false
_brm_is_statement(::ExprColumn{<:Union{typeof(~),typeof(assign)}}) = true
# Top-level entries in a BRMI listing are either ExprColumns whose own show
# already names the LHS (`loc ~ ...`, `(err = ...)`) or other column types
# (DataColumn, MaterializedColumn, ...) whose show prints just the value with
# no name. For the former we strip the outermost parens; for the latter we
# prefix with the operation key so the name doesn't get lost.
_show_top(io::IO, key, op) = print(io, key, ": ", op)
_show_top(io::IO, key, op::ExprColumn{<:Union{typeof(~),typeof(assign)}}) = begin
    args = getargs(op)
    # A known-value address statement prints the way it was written.
    if getf(op) === (~) && last(args) isa ExprColumn{typeof(_brm_known)}
        _show_bare(io, first(args))
        print(io, " = ")
        _show_bare(io, only(getargs(last(args))))
        return
    end
    # An `@plate for` cell prints as written.
    if getf(op) === (~) && last(args) isa ExprColumn{typeof(kernel)}
        body = first(getargs(last(args)))
        if body isa _BRMInlineBody && !isnothing(body.source)
            print(io, body.source)
            return
        end
    end
    _show_bare(io, first(args))
    for arg in Base.tail(args)
        print(io, " ", getop(op), " ")
        _show_bare(io, arg)
    end
end
# Print one statement side without the outermost grouping parentheses, so a
# top-level `loc ~ (1 + x)` reads `loc ~ 1 + x`. Only `*`/`+` sides are
# unwrapped: `~` binds tighter than `|`/`||`/`&&` in Julia, so unwrapping a
# top-level `y ~ (a | g)` would print the misleading `y ~ a | g`.
_show_bare(io::IO, x) = show(io, x)
# The name an `=` statement assigns, as written (`m = mu + b`, not `:m = …`).
_show_bare(io::IO, x::Symbol) = print(io, x)
_show_bare(io::IO, x::ExprColumn{<:Union{typeof(*),typeof(+)}}) =
    _brm_show_infix(io, x, _brm_infix_symbol(x), 0)
Base.show(io::IO, d::DataColumn) = begin
    print(io, "data (eltype=", eltype(parent(d)), ", n=", length(parent(d)), ")")
end
Base.show(io::IO, x::ExprColumn{<:Union{typeof(~),typeof(assign)}}) = begin
    print(io, "(", )
    join(io, getargs(x), " $(getop(x)) ")
    print(io, ")")
end
# Binary-operator terms print infix as written (`x & z`, `(1 + x | p | g)`),
# parenthesized only where Julia's precedence needs it: a child binding no
# tighter than its parent, except as the parent's left operand (operators
# here are left-associative). A term printed on its own is parenthesized.
_brm_infix_symbol(_) = nothing
function _brm_infix_symbol(x::ExprColumn)
    f = getf(x)
    f === doublepipe && return :||
    f isa Function && length(getargs(x)) >= 2 && isempty(getkwargs(x)) || return nothing
    op = nameof(f)
    Base.isbinaryoperator(op) ? op : nothing
end
function _brm_show_infix(io::IO, x, op::Symbol, outer::Int; left::Bool=false)
    precedence = Base.operator_precedence(op)
    parens = precedence < outer || (precedence == outer && !left)
    parens && print(io, "(")
    for (i, arg) in enumerate(getargs(x))
        i == 1 || print(io, " ", op, " ")
        _brm_show_operand(io, arg, precedence; left=i == 1)
    end
    parens && print(io, ")")
end
function _brm_show_operand(io::IO, x, outer::Int; left::Bool=false)
    op = _brm_infix_symbol(x)
    isnothing(op) ? show(io, x) : _brm_show_infix(io, x, op, outer; left)
end
# A bare symbol argument is a name as written (a `|ID|`, an address slot).
_brm_show_operand(io::IO, x::Symbol, _outer::Int; left::Bool=false) = print(io, x)
# A dot call in a formula, `f.(x)`, runs while the builder binds the data and
# would broadcast over the formula column itself, which has no elements.
Base.Broadcast.broadcastable(x::Union{NamedColumn,ExprColumn}) = error(
    "@brm: `.` broadcasting over the formula column `", x isa NamedColumn ? name(x) : x,
    "` while the model is built. A formula call already applies per row: write ",
    "`log(x)`, not `log.(x)`. A value computed once from whole columns, such as ",
    "a mean, is declared `name = constant(mean(log.(x)))`.")

# A constant prints as it was written; its bound inputs are an implementation
# detail of `constant(expr)`.
Base.show(io::IO, x::ExprColumn{typeof(constant)}) =
    print(io, "constant(", first(getargs(x)).expression, ")")
Base.show(io::IO, x::ExprColumn) = begin
    op = _brm_infix_symbol(x)
    isnothing(op) || return _brm_show_infix(io, x, op, typemax(Int))
    _show_call_head(io, getf(x))
    print(io, "(", )
    for (i, arg) in enumerate(getargs(x))
        i == 1 || print(io, ", ")
        _brm_show_operand(io, arg, 0)
    end
    for (i, (key, value)) in enumerate(pairs(getkwargs(x)))
        print(io, i == 1 ? "; " : ", ", key, "=")
        _brm_show_keyword_value(io, value)
    end
    print(io, ")")
end
# A keyword's Symbol value is a literal, as in `extrapolate=:clamp`: a name
# written as a keyword value is bound like any other read and is never stored
# as a Symbol, so the value keeps its colon and re-parses to the same keyword.
_brm_show_keyword_value(io::IO, x) = _brm_show_operand(io, x, 0)
_brm_show_keyword_value(io::IO, x::Symbol) = show(io, x)
# A resolved call head prints as the name the user wrote. Julia prints a
# type/callable binding module-qualified unless it is visible from the
# display context, so a docs build (whose `Main` never imports
# `Distributions`) would render `Distributions.Normal` next to authoring
# surface that says `Normal`. The binding is already resolved, so the module
# prefix is noise, not information — print the bare name.
_show_call_head(io::IO, f) = print(io, f)
_show_call_head(io::IO, f::Union{Function,DataType,UnionAll}) =
    print(io, Base.nameof(f))
_show_call_head(io::IO, g::GlobalRef) = print(io, g.name)
# A prior address prints the way it is written: the parameter head with its
# slots (`sd(:, p)`, `cor(:, p)`, `simplex(mu, mo1(x))`, `effect(mu, a & b)`),
# inverting `_prior_address`, and wrapped in `optional(...)` when marked.
Base.show(io::IO, x::ExprColumn{typeof(effect)}) = begin
    optional = get(getkwargs(x), :optional, false) === true
    optional && print(io, "optional(")
    head, slots = _prior_address_spelling(collect(Symbol, getargs(x)))
    print(io, head, "(")
    join(io, map(_prior_slot_spelling, slots), ", ")
    print(io, ")")
    optional && print(io, ")")
end
const _TERM_PRIOR_CLASS_HEAD = Dict(class => head for (head, class) in pairs(_TERM_PRIOR_HEAD_CLASS))
function _prior_address_spelling(address::Vector{Symbol})
    class = first(address)
    if class in (:sd, :ranef)
        predictor = length(address) >= 3 ? address[3] : _EFFECT_COLON
        return class, length(address) == 4 ?
            [predictor, address[2], address[4]] : [predictor, address[2]]
    elseif class === :cor
        return :cor, [length(address) >= 3 ? address[3] : _EFFECT_COLON, address[2]]
    elseif class === :term_sd
        return :sd, [address[3], address[2], address[4:end]...]
    elseif haskey(_TERM_PRIOR_CLASS_HEAD, class)
        return _TERM_PRIOR_CLASS_HEAD[class], [address[3], address[2]]
    end
    :effect, address
end
_prior_slot_spelling(slot::Symbol) =
    _is_interaction_key(slot) ? join(_interaction_key_operands(slot), " & ") :
    _is_term_key(slot) ? replace(string(slot), "," => ", ") : string(slot)

Base.show(io::IO, x::MultiMembershipTerm) = begin
    print(io, "mm(")
    join(io, getfield(x, :groups), ", ")
    weights = getfield(x, :weights)
    isnothing(weights) || (print(io, "; weights=("); join(io, weights, ", "); print(io, ")"))
    getfield(x, :normalize) || print(io, isnothing(weights) ? "; normalize=false" : ", normalize=false")
    print(io, ")")
end
Base.show(io::IO, x::NamedColumn) = print(io, name(x))
Base.show(io::IO, x::JointResponseColumn) = begin
    getfield(x, :impute) && print(io, "mi(")
    print(io, "[")
    for (i, n) in enumerate(joint_response_names(x))
        i == 1 || print(io, ", ")
        print(io, n)
    end
    print(io, "]")
    getfield(x, :impute) && print(io, ")")
end

end
