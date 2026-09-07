using OrderedCollections

"""
    @brm formula_block
    @brm df formula_block

Parse a brms-style formula block into a [`BRMI`](@ref) (BRM Intermediate).

The one-argument form returns a function `model(df) -> BRMI` (the model
name is gensym-ed). The two-argument form bakes `df` in and returns the
`BRMI` directly. Inside `formula_block`, each line is one of:

- `lhs ~ rhs` — a sampling statement (likelihood or linear predictor).
- `lhs = rhs` — a literal binding (named intermediate).
- `ragged(y, group) ~ rhs` — group a flat observed response at the formula
  boundary, aligned to the `kernel(...)` result referenced by `rhs`.
- `effect(lp, coefficient) ~ Normal(location, scale)` — an SBBRMI
  population-coefficient prior override.
- `sd(lp, ID[, coefficient]) ~ Exponential(scale)` and
  `cor(:, ID) ~ LKJCholesky(K, eta)` — SBBRMI priors for a shared
  `|ID|` random-effect block.

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
addresses a population coefficient or a categorical column's `K-1` treatment
contrasts; `sd` and `cor` address a shared `|ID|` random-effect covariance
block. `sd`'s trailing coefficient slot may be omitted and means `:`, so
`sd(:, pk)` is the block-wide scale and `sd(log_ka, pk, weight)` is one
margin. Because `:` is positional rather than a trailing omission,
`sd(:, pk, weight)` addresses the `weight` margin across *every* predictor
slicing the block. Correlation priors are block-wide by construction — one
shared `|ID|` covariance spans every predictor that slices it — so `cor`
takes `:` in the predictor slot.

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
    protect(x)

brms `I(...)` analogue: literal-escape wrapper around an expression so
the formula parser doesn't try to re-interpret it. In formula-eval
contexts it's a no-op on `Real` — generic broadcast/materialise paths
handle `protect(expr)` without a dedicated dispatch.
"""
function protect end
protect(x::Real) = x

"""
    factor(x; ref=k)

Treat `x` as a categorical predictor with optional reference level
`ref`. Dispatch tag — backend buckets it into the categorical-predictor
path in `introspection.jl` and `_sb_cat` (sbimpl).
"""
function factor end

"""
    mi(y)

Missing-data marker. Wrap a response LHS (`mi(y) ~ Normal(...)`) to opt
into the brms-style observed/imputed split: observed rows feed the
likelihood, missing rows become parameters drawn from the same family.
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
"""
function MvNormalCholesky end

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
    (x.head == :block || x.head == :(=) || isxcall(x, :~)) || error(
        "@brm: a standalone predictor fragment such as `@brm(1 + x)` is " *
        "not implemented. Use nested `@brm(1 + x)` inside an outer `@brm` " *
        "likelihood, or write a top-level `lhs ~ rhs` formula.")
    lhs, x = x.head == :(=) ? x.args : (:($(gensym("model"))(__df__)), x)
    alllocals = OrderedDict{Symbol,Symbol}()
    jointkeys = Set{Symbol}()
    captures = Set{Symbol}()
    info = (;alllocals, jointkeys, captures)
    x = parse!(x; info)
    nonlocals = [key for (key, value) in pairs(alllocals) if value == :nonlocal]
    maybelocals = [key for (key, value) in pairs(alllocals) if value == :maybelocal]
    # kernel(...) do-block captures: body free-symbols that are NOT already a
    # formula name. `capturedata` keeps only those present as df columns, so a
    # do-block param, cell-local, or typo that is not a column is silently
    # dropped (and a non-column typo still errors loudly at the backend/StanBlocks
    # trace, exactly as before this capture path existed).
    real_captures = sort!(collect(setdiff(captures, keys(alllocals))))
    finalize = :($_expand_nested_predictor_formulas(
        $Base.merge((;$(keys(alllocals)...)), __caps__)))
    # ONE shared builder body, parameterised on the `__df__` symbol. Nonlocals
    # bind via the @getproperty (hasproperty→MissingColumn) fallback so a name
    # that isn't a df column (e.g. a multi-equation predictor/param like
    # `loc`/`err`) becomes a MissingColumn instead of erroring; `data` and
    # `maybedata` are interpolated as values so the generated code resolves in
    # ANY consumer scope (`data` is unexported). Both forms below reuse THIS
    # body, so the two builder forms cannot diverge.
    body = Expr(:block,
        :(__ddf__ = $data(__df__)),
        [:($nonlocal = @getproperty __ddf__.$nonlocal) for nonlocal in nonlocals]...,
        :((;$(maybelocals...)) = $maybedata(__df__)),
        x,
        :(__caps__ = $capturedata(__df__, ($(map(QuoteNode, real_captures)...),))),
        finalize,
    )
    if isnothing(df)
        # no-df: `gensym_model(__df__) = body` — a reusable `df -> BRMI` builder.
        Expr(:(=), lhs, body)
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
parse!(x::Expr; info) = if x.head == :block
    Expr(:block, parse!.(x.args; info)...)
elseif x.head == :(=)
    lhs, rhs = x.args
    parselocals!(rhs; info, val=:nonlocal)
    parselocals!(lhs; info, val=:local)
    :(@n $lhs = @x $assign($(xname(lhs)), $rhs))
elseif isxcall(x, :~) && _is_effect_lhs(x.args[2])
    _, lhs, rhs = x.args
    _parse_effect!(lhs, rhs; info)
elseif isxcall(x, :~) && Meta.isexpr(x.args[2], :vect)
    _, lhs, rhs = x.args
    _parse_joint_response!(lhs, rhs; info)
elseif isxcall(x, :~)
    _, lhs, rhs = x.args
    # Shield brms-style `(e | ID | g)` ranef IDs from parselocals! so the bare
    # ID symbol doesn't get registered as a data-column name. Left-associative
    # `|` lowers to `Expr(:call, :|, Expr(:call, :|, e, id), g)`; rewrite to a
    # three-arg `|` whose middle is a QuoteNode so parselocals! treats it as a
    # literal, not a bare Symbol. _x then sees the three-arg form and emits an
    # ExprColumn{|} with the id as a Symbol value.
    rhs = rewrite_ranef_ids(rhs)
    parselocals!(rhs; info, val=:nonlocal)
    parselocals!(lhs; info, val=:maybelocal)
    :(@n $lhs = @x $(Expr(:call, :~, lhs, rhs)))
else
    dump(x)
    error("Don't know how to handle parse!($x)!")
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

function _parse_joint_response!(lhs::Expr, rhs; info)
    names = _joint_response_symbols(lhs)
    parselocals!(rhs; info, val=:nonlocal)

    key = _joint_response_operation_key(names)
    reserved = _joint_response_reserved_keys(names)
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
    joint = :($JointResponseColumn($columns))
    parsed_rhs = _x(rhs)
    :($key = $NamedColumn($(QuoteNode(key)),
        $ExprColumn(~, $joint, $parsed_rhs)))
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
const _PRIOR_HEADS = (:effect, :sd, :cor, :ar, :simplex, :latent, :length_scale)

_is_effect_lhs(x) = any(h -> isxcall(x, h), _PRIOR_HEADS)
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
# of its identity. Keyword arguments are likewise excluded -- `hsgp(x; k=20)`
# and `hsgp(x)` are the same term with different tuning.
const _TERM_HEADS =
    (:s, :t2, :mo, :mo1, :me, :interval_censored, :gp, :hsgp, :ar, :dar)

_is_term_address(x) = Meta.isexpr(x, :call) && !isempty(x.args) &&
                      x.args[1] isa Symbol && x.args[1] in _TERM_HEADS

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
    Symbol(head, "(", join(names, ","), ")")
end

# A canonicalised term key is the one address symbol that can contain `(`, so
# it needs no separate carrier to be told apart from a coefficient or group id.
_is_term_key(s::Symbol) = occursin('(', string(s))

_prior_slot(x) = _is_term_address(x) ? _term_address_key(x) : _effect_address_symbol(x)

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
    elseif head === :cor
        length(args) == 2 || error(
            "@brm: `cor` takes exactly `cor(:, <ID>)`; got `$spelling`.")
        args[1] === _EFFECT_COLON || error(
            "@brm: correlation priors are block-wide — one shared `|ID|` " *
            "covariance block spans every linear predictor that slices it, so " *
            "there is no per-predictor correlation to address. Write " *
            "`cor(:, $(args[2]))`, not `$spelling`.")
        return Symbol[:cor, args[2]]
    end
    error("@brm: unknown prior-address head `$head`; valid heads are " *
          join(("`$h`" for h in _PRIOR_HEADS), ", ") * ".")
end

function _parse_effect!(lhs::Expr, rhs; info)
    head = lhs.args[1]::Symbol
    args = map(_prior_slot, lhs.args[2:end])
    address = _prior_address(head, args)
    # The key becomes a generated LOCAL NAME. A term key carries `(`/`)`/`,`,
    # which are legal inside a `Symbol` but not in emitted Julia source, so
    # those three characters are folded out. Nothing else is touched -- `:`
    # already rode through this key before term addresses existed, and
    # rewriting it would move the generated name of every ALREADY-configured
    # model. The ADDRESS itself is untouched, so diagnostics stay spelled the
    # way the user wrote them.
    key = Symbol("__effect__", replace(join(string.(address), "__"),
                                       "(" => "_", ")" => "", "," => "_"))
    haskey(info.alllocals, key) && error(
        "@brm: duplicate `$head($(join(args, ", ")))` prior statement")
    info.alllocals[key] = :local
    parselocals!(rhs; info, val=:nonlocal)
    quoted_lhs = Expr(:call, :effect, map(QuoteNode, address)...)
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
    hasproperty(info, :captures) && _collect_capture_syms!(info.captures, x.args[2])
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
    Meta.quot(x)
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

# Collect bare Symbols in ARGUMENT positions (never call heads) inside a
# `do`-block cell body — the candidate names a kernel cell references freely.
# `_brm` keeps only those that are real df columns (via `capturedata`) and are not
# themselves formula names; do-block params and cell-locals that are not columns
# fall away there, so no exclusion set is needed here. Mirrors `parselocals!`'s
# call-head skipping so a function name (`exp`, `renewal`) is never captured.
_collect_capture_syms!(acc, x) = nothing
_collect_capture_syms!(acc, x::Symbol) = (push!(acc, x); nothing)
function _collect_capture_syms!(acc, x::Expr)
    if Meta.isexpr(x, :->)
        _collect_capture_syms!(acc, x.args[2])
    elseif Meta.isexpr(x, (:call, :kw))
        for a in x.args[2:end]
            _collect_capture_syms!(acc, a)
        end
    elseif Meta.isexpr(x, :.)
        _collect_capture_syms!(acc, x.args[1])
    else
        for a in x.args
            _collect_capture_syms!(acc, a)
        end
    end
    nothing
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
    function JointResponseColumn(columns::C) where {C<:Tuple}
        length(columns) >= 2 || throw(ArgumentError(
            "a joint response needs at least two columns"))
        all(c -> c isa NamedColumn, columns) || throw(ArgumentError(
            "every joint-response entry must be a named column"))
        names = map(name, columns)
        length(unique(names)) == length(names) || throw(ArgumentError(
            "joint-response columns must be unique"))
        new{C}(columns)
    end
end

joint_response_columns(x::JointResponseColumn) = getfield(x, :columns)
joint_response_names(x::JointResponseColumn) = map(name, joint_response_columns(x))
_joint_response_operation_key(names) =
    Symbol("brm_joint_", join(string.(names), "__"))
function _joint_response_reserved_keys(names)
    key = _joint_response_operation_key(names)
    (key, Symbol(key, :_observed), Symbol(key, :_n), Symbol(key, :_means))
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
"""
Base.merge(a::BRMI, b::BRMI) = BRMI(Base.merge(a.operations, b.operations))
Base.merge(a::BRMI, b::BRMI, rest::BRMI...) =
    foldl(Base.merge, rest; init=Base.merge(a, b))

# Nested predictor formulas are normalised at builder execution time, after
# data columns have values (categorical K is therefore known) but before a BRMI
# becomes public. Downstream introspection, replay, descriptors, and both
# backends see only the existing explicit `eta ~ formula` representation.
_brm_fit_levels(raw::AbstractVector) = sort(unique(raw))

_nested_contains(::Any) = false
_nested_contains(::NestedPredictorFormula) = true
_nested_contains(x::NamedColumn) = _nested_contains(parent(x))
_nested_contains(x::ExprColumn) =
    any(_nested_contains, getargs(x)) || any(_nested_contains, values(getkwargs(x)))

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

function _expand_nested_predictor_formulas(operations::NamedTuple)
    any(_nested_contains, values(operations)) || return BRMI(operations)
    occupied = Set{Symbol}(keys(operations))
    expanded = Pair{Symbol,Any}[]
    for (key, value) in pairs(operations)
        pending = Pair{Symbol,Any}[]
        new_value = _expand_nested_top!(key, value, pending, occupied)
        append!(expanded, pending)
        push!(expanded, key => new_value)
    end
    names = Tuple(first.(expanded))
    values_ = Tuple(last.(expanded))
    BRMI(NamedTuple{names}(values_))
end

Base.show(io::IO, (;operations)::BRMI) = begin
    print(io, "BRMI:\n")
    for (key, value::NamedColumn) in pairs(operations)
        print(io, "  ")
        _show_top(io, key, parent(value))
        print(io, "\n")
    end
end
# Top-level entries in a BRMI listing are either ExprColumns whose own show
# already names the LHS (`loc ~ ...`, `(err = ...)`) or other column types
# (DataColumn, MaterializedColumn, ...) whose show prints just the value with
# no name. For the former we strip the outermost parens; for the latter we
# prefix with the operation key so the name doesn't get lost.
_show_top(io::IO, key, op) = print(io, key, ": ", op)
_show_top(io::IO, key, op::ExprColumn{<:Union{typeof(~),typeof(assign)}}) =
    join(io, getargs(op), " $(getop(op)) ")
Base.show(io::IO, d::DataColumn) = begin
    print(io, "data (eltype=", eltype(parent(d)), ")")
end
Base.show(io::IO, x::ExprColumn{<:Union{typeof.((~,*,+,|,doublepipe,assign))...}}) = begin
    print(io, "(", )
    join(io, getargs(x), " $(getop(x)) ")
    print(io, ")")
end
nonemptyjoin(io::IO, iterator, args...; first) = if length(iterator) > 0
    print(io, first)
    join(io, iterator, args...)
end
Base.show(io::IO, x::ExprColumn) = begin
    print(io, getf(x), "(", )
    join(io, getargs(x), ", ")
    nonemptyjoin(io, ["$key=$value" for (key, value) in pairs(getkwargs(x))], ", "; first="; ")
    print(io, ")")
end
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
    print(io, "[")
    for (i, n) in enumerate(joint_response_names(x))
        i == 1 || print(io, ", ")
        print(io, n)
    end
    print(io, "]")
end

end
