# Formula terms

## Catalogue

Every term below lowers on the StanBlocks backend (`SBBRMI`); the
`VBRMI` column marks the ones the pure-Julia backend also implements. The pages
that follow document the terms whose behaviour is not obvious from the
signature — the rest are covered by their docstrings on the [API](@ref) page.

### Predictor terms

| Term | What it contributes | `VBRMI` |
| --- | --- | --- |
| `offset(x)` | `x` added with coefficient one — no `beta_pop` column | ✓ |
| `s(x)` | rank-10 penalized thin-plate regression spline | — |
| `t2(x, z)` | two-margin tensor-product smooth | — |
| `gp(x…; cov=:exp_quad, iso=true, jitter=1e-9)` | exact latent Gaussian process, noncentered Cholesky draw | — |
| `gp(x; cov=:periodic, period, jitter=1e-9)` | exact latent GP under Stan's periodic kernel, one axis | — |
| `hsgp(x…; k=20, c=1.5, iso=true, by=nothing, domain=nothing, orthogonal_to=nothing)` | Hilbert-space GP approximation over `prod(k)` basis functions | — |
| `hsgp(x; k=20, cov=:periodic, period)` | periodic Hilbert-space basis: `k` harmonics, `2k` cosine/sine functions | — |
| `ar(time; p=1)` | AR(p) noise process ordered by `time`; only `p=1` is emitted | — |
| `dar(time; p=1)` | direct differenced-AR(1) trajectory with bounded persistence and scaled innovations | — |
| `mo(c)`, `mo1(c)` | monotonic effect of an ordered factor via Dirichlet increments | — |
| `me(x, sd)` | measurement-error covariate — `x` is observed with known `sd` | — |
| `interval_censored(x; upper=lloq, lower=0)` | quantified/BLOQ covariate with bounded latent values on BLOQ rows | — |
| `factor(c; ref=k)` | treatment contrasts for a categorical column, reference level `k` | ✓ |
| `protect(x)` | materialize a raw data expression as one literal column | ✓ |

A plain RHS expression in raw data columns (`log(exposure)`, `x^2`) is treated
as an implicit `protect(...)` and materialized the same way.

### Differenced-AR(1) trajectories

`dar(time; p=1)` is not an alias for `ar(time; p=1)`. It emits the path

```
x[1] = 0
d[t] = beta * d[t-1] + sigma * z[t]    # d[0] = 0
x[t+1] = x[t] + d[t]
```

as a direct predictor summand. In `log_r_week ~ 1 + dar(week)`, the population
intercept is therefore the initial level `x0`; the trajectory receives no
second `beta_pop` multiplier. The term samples `beta` on `[0, 1]`, a positive
innovation scale `sigma`, and `length(week)-1` standardized innovations `z`.
Its defaults are the CDC wastewater priors `beta ~ Normal(0.5, 0.2)` truncated
to `[0, 1]` and `sigma ~ Normal(0, 0.2)` truncated at zero.

The time column must be nonempty, finite, strictly increasing, and unique.
Its values establish order; spacing does not rescale the recurrence. Expand a
weekly path onto a daily axis explicitly with a deterministic index operation
such as `log_r = weekly_expand(log_r_week, week_idx)`. `reprocess` may replace
the ordered grid, including its length, without changing the Stan source.

Configure the two model-scale priors by addressing the term:

```julia
ar(:, dar(week)) ~ Normal(0.4, 0.1)  # bounded persistence beta
sd(:, dar(week)) ~ Normal(0.0, 0.3)  # positive innovation sigma
```

`ar` accepts `Normal`, `Beta`, or an in-bounds `Uniform`; `sd` accepts the
positive-scale family set documented below. The standardized `z` innovations
are inspectable through the descriptor but deliberately have no prior override.

### HSGP over a model-derived predictor

A one-dimensional `hsgp` axis may be a sampled linear predictor or formula
assignment, not only a raw dataframe column. This makes a latent concentration
available to both a linear effect and a residual nonlinear effect in one joint
model:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
latent_hsgp_model = (@brm begin
    log(x) ~ 1 + factor(nominal_time) + (1 | assay | subject)
    sigma_assay_log ~ Exponential(0.5)
    c_obs ~ censored(LogNormal(log(x), sigma_assay_log); lower=lloq)

    mu ~ 1 + factor(nominal_time) + zbl + x +
         hsgp(x; k=5, domain=(0.01, 5.0), orthogonal_to=:linear) +
         (1 + x | qt | subject)
    sigma ~ Exponential(1)
    qtc ~ Normal(mu, sigma)
end)((;
    nominal_time=repeat([1, 2]; outer=4),
    subject=repeat(1:4; inner=2),
    zbl=[1., 1., 1., 1., 0., 0., 0., 0.],
    c_obs=[0.3, 0.3, 0.3, 0.3, 0.31, 0.37, 0.44, 0.54],
    lloq=fill(0.3, 8),
    qtc=[1.1, 1.2, 1.4, 1.5, 1.7, 1.8, 2.0, 2.1],
))
""", :latent_hsgp_model; title="Latent concentration with linear and HSGP effects")
```

The explicit `domain=(lower, upper)` is required because sampled `x` values do
not exist when Julia configures the basis. It is the actual compact HSGP
interval, so it cannot be combined with the raw-data expansion factor `c` and
remains fixed during `reprocess`, including with `freeze_constants=false`.
Choose it to cover scientifically plausible posterior support for every latent
`x`: the domain configures the approximation but does not truncate or otherwise
constrain `x` at runtime.

`orthogonal_to=:linear` centers every basis column and projects it off the
current sampled `x` direction at each draw. Use it when the formula also
contains `x`; the population coefficient then carries the linear association
and the HSGP carries only residual nonlinear shape. The option also works for a
one-dimensional raw axis. The orthogonality option is ungrouped: multiplying
the basis by group-specific weights would no longer preserve the global
projection. Model-derived HSGPs are currently one-dimensional, isotropic, and
ungrouped; raw-data HSGPs retain their variadic, anisotropic, and group-specific
forms when `orthogonal_to` is omitted.

After fitting, [`hsgp_population_curve`](@ref) evaluates the combined
population exposure contribution on a fixed grid without fabricating assay
rows or referring to emitted Stan names:

```julia
curve = hsgp_population_curve(
    descriptor, constrained_draws, constrained_names,
    collect(range(0.01, 5.0; length=100));
    predictor=:mu, coefficient=:x, term=:hsgp_x)

curve.linear  # beta_x * x, draws × grid
curve.hsgp    # orthogonal residual nonlinear contribution, draws × grid
curve.total   # the supported total exposure curve
```

`constrained_draws` and `constrained_names` must include transformed
parameters (`include_tp=true` in BridgeStan), because every posterior draw has
its own sampled training `x`. BRM uses those values to replay the exact fitted
intercept/linear projection on the new grid; re-orthogonalizing against the
grid would define a different curve. Evaluation outside the formula's fixed
`domain` is rejected. This is a population partial effect only: it deliberately
excludes the intercept, other covariates, and subject-specific random slopes.

### Interval-censored predictor

Use `interval_censored(x; upper=lloq)` when `x` is quantified on some rows and
BLOQ on others. Store the measured concentration on quantified rows and the
row-specific LLOQ on BLOQ rows; the convention `x == lloq` identifies BLOQ, so
no separate flag is needed. Each BLOQ row allocates one latent covariate value
between zero and its LLOQ, and the merged vector enters the ordinary fixed- or
random-effect design matrix:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
interval_censored_predictor_model = (@brm begin
    qtc ~ Normal(mu, sigma)
    mu ~ 1 + interval_censored(conc; upper=lloq)
    effect(mu, conc) ~ Normal(0, 2)
    latent(mu, interval_censored(conc)) ~ Normal(0, 5)
    sigma ~ Exponential(1)
end)((;
    qtc=[401.0, 408.0, 415.0],
    conc=[0.7, 0.25, 1.2],
    lloq=[0.25, 0.25, 0.4],
))
""", :interval_censored_predictor_model;
    title="Interval-censored concentration predictor")
```

The term's default latent prior is `Normal(0, 1)`; use
`latent(<lp|:>, interval_censored(x)) ~ Normal(location, scale)` to set it.
The population slope keeps the ordinary address `effect(<lp>, x)`. Reusing the
same term as a fixed effect and a random slope shares one latent covariate
vector. `reprocess` rebuilds the quantified/BLOQ split from new `x` and `lloq`
columns.

This is a continuous latent-covariate model, not an `LLOQ/2` substitution.
For a BLOQ assay row the lower bound defaults to zero and can be changed with a
numeric `lower=`. Values and LLOQs must be finite; `x` must equal its LLOQ on
BLOQ rows and exceed it on quantified rows, and at least one row must be BLOQ.
The predictor form is `SBBRMI`-only and is distinct from the response-likelihood
form documented under [Likelihoods](@ref).

`gp` and `hsgp` are distinct terms with no compatibility alias. Both are direct
predictor summands carrying their own latent draws and hyperparameters, so
neither contributes a `beta_pop` coefficient. `jitter` belongs only to `gp`;
`k`, `c` and `by` only to `hsgp`. Both support `cov=:exp_quad` (the default)
and `cov=:periodic`; see [Periodic covariance](@ref) for the latter's contract.

### Periodic covariance

`cov=:periodic` selects Stan's periodic kernel

```
k(x, x') = σ² exp(−2 sin²(π |x − x'| / period) / ρ²)
```

over exactly one axis. `period` is a required numeric formula constant on the
axis's own scale (hours, days, radians — whatever the column carries), and
`rho` / `sigma` keep their meaning and their `length_scale(…)` / `sd(…)`
addresses. It is an explicit assumption that the effect repeats with that
period; clock-time or dosing-phase evidence is what makes a 24-hour period on
elapsed time readable as shared diurnal variation.

```julia
@brm df begin
    y  ~ Normal(mu, sigma)
    mu ~ 1 + hsgp(hours_since_dose; k=6, cov=:periodic, period=24.0)
    length_scale(:, hsgp(hours_since_dose)) ~ LogNormal(0, 0.5)
    sd(:, hsgp(hours_since_dose))           ~ Normal(0, 0.3)
    sigma ~ Exponential(1)
end
```

- `gp(x; cov=:periodic, period=…)` is the exact kernel, lowered to Stan's
  native `gp_periodic_cov` plus `jitter`.
- `hsgp(x; k, cov=:periodic, period=…)` is the Hilbert-space approximation of
  Riutort-Mayol et al. (2023): with `w0 = 2π / period` the basis is
  `cos(j·w0·x)` and `sin(j·w0·x)` for `j = 1:k` — `2k` functions, emitted as
  data `PHI_hsgp_<x>` — and each harmonic's spectral weight is
  `σ·√(2·e^{−a}·I_j(a))` with `a = 1/ρ²`, computed inside Stan through
  `log_modified_bessel_first_kind` so a small length scale cannot overflow.
- **The term contains no constant.** The `j = 0` harmonic is excluded, so
  every basis function has zero mean over a period and the term behaves like
  a `0 +` term: a nonzero mean of the periodic function belongs to the
  formula intercept (or whichever declared quantity owns it), never to this
  term, and nothing is added to the consumer's intercept silently. Count
  `2k` functions against a basis budget (16 for `k = 8`).
- **Truncation is the only approximation, and it is measured.** The
  `k`-harmonic expansion is exactly periodic but approximates the
  exp-sine-squared kernel with a tail `σ²·2·e^{−a}·Σ_{j>k} I_j(a)` that
  shrinks as `rho` grows. Dropped fraction of the (non-constant) variance:

| `k` (harmonics) | basis functions | default floor on `rho` | Riutort-Mayol B.6 (`3.72/k`) | dropped variance fraction at `rho` = 0.3 / 0.47 / 0.6 / 0.8 / 1.0 / 1.5 |
|---|---|---|---|---|
| 4 | 8 | 1.931 | 0.930 | 2.0e-1 / 4.4e-2 / 1.3e-2 / 2.0e-3 / 4.1e-4 / 1.9e-5 |
| 8 | 16 | 0.614 | 0.465 | 1.3e-2 / 2.4e-4 / 1.2e-5 / 2.2e-7 / 8.0e-9 / 1.5e-11 |
| 16 | 32 | 0.278 | 0.233 | 3.1e-6 / 1.2e-10 / 1.5e-13 / 3.0e-17 / 3.0e-20 / 8.7e-26 |
| 32 | 64 | 0.135 | 0.116 | 4.3e-17 / 1.8e-27 / 1.0e-33 / 2.2e-41 / 1.9e-47 / 1.3e-58 |

  The default floor (next section) is the `rho` at which the `k`-th
  harmonic's amplitude is 1/100 of the first's; it is stricter than the
  paper's `k ≥ 3.72/rho` rule. At `k = 8` the maximum covariance error over a
  full period is `9e-6·σ²` at the floor and `1e-2·σ²` at `rho = 0.3`, where
  an explicit `length_scale` override would take you knowingly.
- `rho` is **dimensionless**: it is the kernel's length scale on the unit
  circle, in chord units (`2·sin(π·τ/period)` is the chord spanned by a lag
  `τ`), never in the axis's own units — `rho ≈ 1` means features of roughly
  one radian, i.e. `period/(2π)` on the axis. State `length_scale(…)` priors
  in those units. The raw axis values go straight into the cosines and sines,
  so a large elapsed time needs no consumer-side `mod(x, period)`.
- The periodic basis needs **no boundary factor and no domain**: `c`,
  `domain`, `orthogonal_to`, `by`, and `iso=false` are refused by name, and a
  model-derived axis is not supported. Cosines and sines already have zero
  mean over a period and are not collinear with a linear `x`.
- `reprocess` rebuilds the cosine/sine columns from the new axis. Nothing here
  is fitted, so `freeze_constants=true` and `false` are identical, a constant
  prediction axis is valid, and rows far outside the training range are the
  point rather than an extrapolation hazard.
- [`brm_term_coordinates`](@ref) resolves `:length_scale`, `:sd`, and
  `:basis_weights` exactly as for the exp-quad basis; `:basis_weights` has
  `2k` coordinates, and [`term_draws`](@ref) zeroes all of them.
- The validity floor below applies with the periodic weights substituted:
  `rho_lower_hsgp_<x>` is the length scale at which `I_k(a)/I_1(a) = 100⁻²`,
  a function of `k` alone.
- **StanBlocks floor.** The weight helper calls Stan's
  `log_modified_bessel_first_kind`, registered in StanBlocks from
  `bec23bc3c52303ebde60a026af48c435e4c81330` (`devibe`, 2026-09-07). On an older
  StanBlocks the periodic `hsgp` form fails at transpile with
  `Could not find log_modified_bessel_first_kind …`; the exact
  `gp(…; cov=:periodic)` form needs nothing new.

### Column transforms

Applied to a raw column before it enters the design matrix. The constants are
fitted on the training frame and frozen, so [`reprocess`](@ref) replays them
rather than re-deriving them — see the replay contract below.

| Term | What it does |
| --- | --- |
| `zscale(x)` | subtract the mean, divide by the SD |
| `center(x)` | subtract the mean |
| `standardize(x)` | mean/SD standardisation |

### Grouping factors

Used on the right of `|` in a random-effect block.

| Term | What it does |
| --- | --- |
| `gr(g)` | ordinary grouping factor / strata |
| `mm(g1, g2, …; weights=(w1, w2, …), normalize=true)` | multi-membership — one coefficient block shared across two or more levels per row |

`mm` needs at least two group columns; omitting `weights` gives exact equal
weights `1/M`. Supplied weights must be present, real, finite, nonnegative and
sum to something positive on every row. It does not combine with `|ID|`,
`cv_groups` or `centered_groups`, which error explicitly.

### Response-level wrappers

| Term | What it does |
| --- | --- |
| `mi(y)` | brms-style observed/imputed split — missing rows become parameters drawn from the same family |
| [`weighted(y, w)`](@ref) | typed observation weights on the likelihood |
| `truncated`, `censored`, [`interval_censored`](@ref) | truncation and censoring — see [Likelihoods](@ref) |

`mm`'s membership weights are part of the random-effect contribution and are a
different thing from `weighted(...)`, which scales the likelihood.

### Group-local kernels

| Term | What it does |
| --- | --- |
| `kernel(args…) do …` | a per-subject model cell — the PMX/PKPD surface |
| `ragged(x, group)` | group a flat secondary row axis into the ragged per-subject view a `kernel` cell slices |

`ragged`'s `x` may be a linear predictor declared in the same `@brm` block or a
raw flat data column; `group` names, for every row of that frame, which subject
the row belongs to. See the [multi-axis population PK kernel](@ref) for a
runnable example whose subject and observation columns have different lengths.

## Fixed-one contribution: `offset(x)`

[`offset`](@ref) adds `x` directly to a population-level linear predictor with
coefficient one. It never allocates a `beta_pop` column. The argument may be a
raw-data expression, as in an exposure offset, or an already-declared sampled
scalar:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
offset_model = (@brm begin
    log_ka_pop ~ Normal(-2.08, 1.0)
    sigma ~ Exponential(1.0)
    eta_ka ~ offset(log_ka_pop) + (1 | p | subject)
    concentration ~ Normal(exp(eta_ka), sigma)
end)((;
    subject=[1, 1, 2, 2],
    concentration=[0.2, 0.4, 0.7, 0.5],
))
""", :offset_model; title="Sampled-value offset")
```

Use [`protect`](@ref) for literal data transformations whose resulting column
should receive an estimated population coefficient. `protect(log_ka_pop)` is
therefore not an alternative spelling: `protect` materializes raw-data
expressions, while `offset` preserves a model-value reference and fixes its
coefficient to one. Offsets are population-level terms and are rejected inside
`(... | group)` random-effects terms.

## Penalized smooth: `s(x)`

[`s`](@ref) adds a penalized one-dimensional thin-plate regression spline to a
linear predictor. It is available only in the `SBBRMI` StanBlocks backend.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
x_smooth = collect(range(-2, 2; length=50))
smooth_model = (@brm begin
    y ~ Normal(mu, sigma)
    mu ~ s(x)
    sigma ~ Exponential(1)
end)((; x=x_smooth, y=sin.(x_smooth)))
""", :smooth_model; title="Penalized smooth")
```

The supported public call has exactly one numeric predictor and no keyword
arguments. Training values must be finite and contain at least 10 unique
values. Internally, `s(x)` uses a fixed rank-10 basis: two unpenalized
null-space columns for `{1, x}` and eight penalty-whitened range columns. The
range coefficients share a smoothing standard deviation whose default prior is
a standard half-normal; `sd(<lp|:>, s(x)) ~ Exponential(scale)` replaces it (see
[Term-internal priors](@ref)). The term contributes this complete smooth
directly to the linear predictor, so it does not receive an additional
population coefficient.

For prediction or posterior replay on new data, the default
`reprocess(sb, new_df)` and `restan_data(sb, new_df)` calls evaluate `x` against
the frozen training centers and basis. Passing `freeze_constants=false`
re-estimates the basis from the new data and therefore has fresh-fit rather
than prediction semantics.

### Difference from `bs(...)`

Bambi/Formulae formulas such as `bs(x, knots=knots)` create a deterministic
B-spline design matrix whose dimension and knots are controlled by the formula.
BayesianRegressionModels' `s(x)` instead represents a penalized thin-plate
smooth with the fixed rank and smoothing prior described above. They are not a
one-for-one syntax translation: `s(x; k=...)`, `s(x; knots=...)`, `bs(...)`, and
`t2(...)` are not alternative spellings of this term.

## Tensor-product smooth: `t2(x, z)`

[`t2`](@ref) adds a two-margin tensor-product smooth to an `SBBRMI` linear
predictor. Its public defaults follow the brms/mgcv `t2` catalogue surface,
with Julia tuples for per-margin options:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
t2_model = (@brm begin
    loc ~ 1 + t2(area, yearc;
                 k=(5, 5), basis=(:cr, :cr), full=false)
    rent ~ Normal(loc, sigma)
    sigma ~ Exponential(1)
end)((;
    area=repeat(collect(1.0:5.0); inner=5),
    yearc=repeat(collect(-2.0:2.0); outer=5),
    rent=collect(range(0.1, 2.5; length=25)),
))
""", :t2_model; title="Tensor-product smooth")
```

Both predictors must be finite numeric columns with at least the corresponding
number of unique values in `k`. Each `k` entry must be an integer greater than
2. The current implementation accepts only cubic-regression-spline margins
(`basis=(:cr, :cr)`) and `full=false`; unsupported values and unknown keywords
are rejected while the `BRMI` is built.

Each marginal basis is split into a two-dimensional null space and a
penalty-whitened range space. Their tensor product has three unpenalized
null×null columns after the intercept constraint, plus independently scaled
range×range, range×null, and null×range blocks. At the default `k=(5, 5)`,
those penalized blocks have 9, 6, and 6 coefficients. Each block has its own
smoothing scale, defaulting to a standard half-normal and addressable
individually as `sd(<lp|:>, t2(x, z), <block>)` with `<block>` one of `rr`,
`rn`, `nr` (see [Term-internal priors](@ref)). The complete smooth is added
directly without an extra population coefficient.

`reprocess(sb, new_df)` evaluates all four blocks against the frozen training
knots, penalty decomposition, and intercept constraint. Passing
`freeze_constants=false` re-estimates them from `new_df`. Like [`s`](@ref),
`t2` is implemented only by the StanBlocks backend and is not available to
`VBRMI`.

## Term-internal priors

Some terms own parameters that no coefficient address can reach. `s(x)`'s
smoothing scale, `mo(c)`'s Dirichlet increments, `me(x, sd)`'s latent true
covariate, `interval_censored(x; upper=lloq)`'s bounded latent values, and
a Gaussian process's length scale and amplitude all live inside the term's own
submodel. A `dar` trajectory likewise owns its persistence and innovation
scale. None is a `beta_pop` column or a grouping-factor margin. They are
addressed by naming the term itself in the target slot, in the same
head-position grammar the rest of the prior surface uses:

```
<quantity>(<linear predictor | :>, <term>[, <component>]) ~ <distribution>
```

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
term_prior_model = (@brm begin
    y ~ Normal(mu, sigma)
    mu ~ 1 + s(age) + t2(x, z) + mo(dose) + me(w_obs, 0.3) + hsgp(conc; k=5)
    sigma ~ Exponential(1)

    sd(:, s(age))               ~ Exponential(2)     # smoothing scale
    sd(mu, t2(x, z), rr)        ~ Exponential(3)     # one tensor penalty
    simplex(:, mo(dose))        ~ Dirichlet(2)       # monotonic increments
    latent(mu, me(w_obs))       ~ Normal(0, 5)       # latent true covariate
    length_scale(:, hsgp(conc)) ~ Uniform(0.84, 2)   # GP length scale
    sd(:, hsgp(conc))           ~ Normal(0, 0.5)     # GP marginal amplitude
end)((;
    age=collect(20.0:44.0),
    x=repeat(collect(1.0:5.0); inner=5),
    z=repeat(collect(-2.0:2.0); outer=5),
    dose=repeat(1:5; inner=5),
    w_obs=collect(range(1, 4; length=25)),
    conc=collect(range(-2, 2; length=25)),
    y=collect(range(-1, 1; length=25)),
))
""", :term_prior_model; title="Term-internal priors")
```

| head | term | parameter | default |
| --- | --- | --- | --- |
| `sd` | `s(x)` | smoothing SD | half-standard-normal |
| `sd` | `t2(x, z)` | one of the `rr` / `rn` / `nr` penalty SDs | half-standard-normal |
| `simplex` | `mo(c)`, `mo1(c)` | Dirichlet concentration | `Dirichlet(1)` |
| `latent` | `me(x, sd)` | latent true covariate | `Normal(0, 1)` |
| `latent` | `interval_censored(x; upper=lloq)` | latent covariate on BLOQ rows | truncated `Normal(0, 1)` |
| `length_scale` | `gp(x…)`, `hsgp(x…)` | GP length scale `rho` | `LogNormal(0, 1)` |
| `sd` | `gp(x…)`, `hsgp(x…)` | GP marginal amplitude `sigma` | `LogNormal(0, 1)` |
| `ar` | `dar(time)` | bounded persistence `beta` | truncated `Normal(0.5, 0.2)` on `[0, 1]` |
| `sd` | `dar(time)` | innovation scale `sigma` | half-`Normal(0, 0.2)` |

**Spell the term the way the formula does, minus numeric and keyword
arguments.** `me(w_obs, 0.3)` is addressed as `me(w_obs)` and `t2(x, z; k=(5,5))`
as `t2(x, z)` — the address names a term, not a call. A `:` in the predictor
slot is THE DEFAULT: it is the base layer that a statement naming a concrete
predictor overrides, exactly as on the coefficient and grouping-factor
surfaces. Specificity counts concrete slots, so an exact tie is an error rather
than a silent winner.

An address that cannot be honoured is refused by name while the model is
lowered — an unknown term, a term with no such parameter, a `t2` component slot
that is missing or names no penalty block, or two terms in one predictor
spelled identically. `term_priors(brmi)` returns the captured statements
(`class`, `term`, `predictor`, `component`, `family`, `arguments`, `keywords`,
`expression`) for formula-level provenance.

### Only model-scale quantities are exposed

The standardized raw innovations these submodels sample — `b_pen_raw`, `z`,
`beta_raw` — stay iid standard normal and are deliberately NOT configurable.
For a smooth, `Cov(f_pen | sd) = sd² Zpen Zpen'`, so a scale on the raw
coefficients would simply duplicate the smoothing SD and confound the two. The
configurable parameters are the ones that mean something on the model's own
scale.

### Current limits

- `sd` on `s(x)` / `t2(x, z)` accepts `Exponential(scale)` only. In contrast,
  an addressed shared grouping-factor SD also accepts the zero-centered
  half-Normal spelling `Normal(0, scale)`. `Distributions.Exponential` is
  scale-parameterized while Stan's `exponential_lpdf` takes a rate; BRM
  performs the conversion.
- `simplex` accepts `Dirichlet(a)` (one concentration, broadcast over every
  increment) or `Dirichlet(a₁, …, a_{K-1})` (one per increment of a `K`-level
  factor). The dimension comes from the data, so the
  `Dirichlet(dimension, concentration)` spelling used for a standalone simplex
  parameter is not accepted here.
- `latent` accepts `Normal(location, scale)`. The observation likelihood
  `x_obs ~ Normal(x_true, sd)` is never configurable.
- `length_scale` and `sd` on a `gp` / `hsgp` term accept `LogNormal`,
  `InverseGamma`, `Gamma`, `Exponential`, `Normal` and `Uniform`, with numeric
  hyperparameters. Both parameters are strictly positive, so every family
  except `Uniform` is emitted truncated at zero (`Normal(0, s)` is therefore a
  half-normal); `Uniform(a, b)` additionally declares `<lower=a, upper=b>` so
  the declaration and the density's support agree. An **anisotropic** term
  (`iso=false`) has one length scale per axis and the statement sets all of
  them; the isotropic form has a single shared one. With `by=g` the length
  scale and amplitude are shared across groups, so one statement configures the
  whole term.
- `ar(:, dar(time))` accepts `Normal`, `Beta`, or `Uniform`; every declaration
  stays within `[0, 1]`. `sd(:, dar(time))` accepts the same positive-scale
  families as a GP amplitude. The older `ar(time)` term's transformed
  autocorrelation still has no prior address.

The backend compatibility floor for configured `gp` / `hsgp` term priors is
StanBlocks `10529af04d42a330df383864059c2b61a11d9480`. These statements splice
a configured `SlicModel` value into BRM's generated model body. Earlier
StanBlocks revisions, including `c30d3a158ae6c996dee2423023ab6b35d2756fc9`,
trace that value before its `PHI` / `omega2` keyword data are bound and fail
with `Could not find omega2 ...`. Co-pin BRM revisions containing this surface
with `10529af` or later. Because StanBlocks is unregistered and both revisions
identify as version `0.1.5`, the commit SHA—not the package version—is the
effective compatibility check.

Omitting an explicit HSGP prior is not a model-preserving workaround. In
particular, removing `length_scale(:, hsgp(x)) ~ LogNormal(0, 1)` restores the
default approximation-validity floor described below, changing the parameter
support and the emitted Stan program.

### `hsgp` bounds its length scale by default

An HSGP with `k` basis functions over a domain of half-width
`L = c·max|x − mean(x)|` stops approximating the kernel it was asked for once
the length scale falls below

```
(4L/π)·√(log(100)/(k² − 1))
```

and the failure is **silent**: the model transpiles, passes `stanc`, samples,
and returns finite draws that simply are not that Gaussian process. On the
default `LogNormal(0, 1)` at `L = 1.5` that region holds 42.9 % of the prior
mass at `k = 5`, 18.8 % at `k = 10` and 5.7 % at `k = 20`.

Every `hsgp` term therefore declares `rho` with that floor as its lower bound.
The density is unchanged — only the support moves. Exact `gp` has no basis
truncation and is untouched.

The rule behind the number is that the `k`-th basis function's spectral
*amplitude* has fallen to 1/100 of the first's. A `cov=:periodic` term
applies the same rule to its own weights, `q_j ∝ √(e^{−a} I_j(a))` with
`a = 1/ρ²`: its floor is the length scale solving `I_k(a)/I_1(a) = 100⁻²`,
found by bisection on the scaled Bessel functions. It depends on `k` only —
there is no data-derived `L` — so every `reprocess` reproduces it exactly.

The floor is emitted as **data** (`rho_lower_hsgp_<axes>`), not as a literal,
because `L` comes from the covariate: `reprocess(sb, df2; freeze_constants=false)`
re-derives it alongside `PHI` / `omega2` while the Stan source stays
byte-identical. Isotropic spellings share one `rho` across axes and take the
strictest per-axis floor; `iso=false` bounds each axis separately. `k = 1` puts
the formula at infinity, so that degenerate basis stays unbounded.

**An explicit `length_scale` statement replaces the whole declaration, floor
included:**

```julia
mu ~ hsgp(x; k=5, c=1.5)                     # real<lower=rho_lower_hsgp_x>
length_scale(:, hsgp(x)) ~ LogNormal(0, 1)   # real<lower=0.0> — floor gone
```

That is deliberate. It is how a pre-floor posterior is reproduced, and it is
what keeps `Uniform(a, b)` self-consistent — an unconditional floor would
overwrite `lower=a` while the density stayed `uniform(a, b)`, leaving the
declaration and the support disagreeing. The consequence is that the guarantee
is **default-on, not absolute**: an override with mass below the floor restores
the silent approximation error, with no warning. Either keep the floor in the
overriding bounds (`Uniform(0.84, 2)`) or accept the error knowingly.

!!! warning "This changes the posterior of existing unedited `hsgp` models"
    Models that never named `length_scale` sampled an unbounded `rho` before
    this default landed. If you version models by their emitted Stan plus
    data, treat the change as a new model version rather than a refresh of the
    old one.

## R²-induced variance decomposition: `effect(lp, :) ~ r2d2(...)`

[`r2d2`](@ref) replaces the independent per-coefficient priors on a linear
predictor's population block with a *joint* prior induced by a prior on that
predictor's coefficient of determination. It is a separate statement in the
formula block — addressed with [`effect`](@ref), never with a per-column
`effect(lp, coef) ~ Normal(...)` — and is implemented only by the `SBBRMI`
StanBlocks backend.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
r2d2_model = (@brm begin
    log_CL ~ 1 + wt + age + (1 | p | subject)
    log_V  ~ 1 + wt + (1 | p | subject)

    effect(log_CL, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
    effect(log_V, :)  ~ r2d2(R2=Beta(2, 3), tau_bsv=0.25)

    conc ~ Normal(exp(log_CL - log_V) * time, 1)
end)((;
    wt=[55.0, 65.0, 75.0, 85.0], age=[21.0, 38.0, 55.0, 29.0],
    subject=[1, 1, 2, 2], time=[0.5, 1.0, 1.5, 2.0],
    conc=[0.2, 0.4, 0.3, 0.1],
))
""", :r2d2_model; title="R2D2 population and group decomposition")
```

`effect(lp, :)` addresses every population coefficient of `lp` at once, and
`effect(:, :)` addresses them across every predictor — which for an `r2d2`
decomposition resolves only when the model has exactly one population
predictor, since the decomposition is per-predictor. An `r2d2` statement is
deliberately invisible to [`effect_priors`](@ref) — it names no single labelled
column — and is read back with [`r2d2_priors`](@ref) instead.

### What it emits

Writing `tau_bsv` for the predictor's total scale, the decomposition is

```
R2               ~ Beta(a, b)                                 # parameter
phi              ~ Dirichlet(alpha)                           # simplex
beta_scale[k]     = sqrt(phi[k] * R2 * tau_bsv^2 / Var(x_k))  # transformed
tau_resid         = sqrt((1 - R2) * tau_bsv^2)                # transformed
```

`beta_scale` is injected into the shipped `_popefs_normal` seam, so the
population block still samples as `beta_pop ~ normal(beta_loc, beta_scale)`;
`tau_resid` becomes the random effect's standard deviation. The column
variances `Var(x_k)` depend on data alone and hoist to Stan's transformed-data
block.

The intercept is excluded from the simplex. Its design column is constant, so
`Var(x_k) = 0` there, and an intercept is a location rather than explained
variance. A predictor whose only non-intercept column count is one emits a
`simplex[1]`; it is deterministically `[1]` and adds no sampler dimensions.

### The random effect plays the residual role

The inciting shape is population PK: `log_CL` is a *latent* per-subject
parameter with no residual term of its own, so its subject random effect **is**
the unexplained half of `tau_bsv^2`. That is why `r2d2` derives the
random-effect SD rather than sampling it, and why an
`sd(...)` statement on the same block is rejected — the
decomposition already determines those scales.

### Keywords

| keyword | default | meaning |
|---|---|---|
| `R2` | required | `Beta(a, b)` prior on the coefficient of determination |
| `tau_bsv` | sampled half-standard-normal | the predictor's total scale; pass a number to fix it |
| `alpha` | `1` | Dirichlet concentration over the non-intercept columns |

`tau_bsv` has no data-derived default. A latent per-subject parameter has no
observed response to derive a scale from, so an omitted `tau_bsv` becomes a
sampled `real<lower=0>` with a half-standard-normal prior. Fix it whenever you
have a defensible scale — it is the quantity the whole decomposition is
relative to.

### Current limits

All of these fail loudly rather than silently sampling something else:

- The random effect playing the residual role must be a single intercept —
  `(1 | g)` or one margin per predictor inside a `(1 | ID | g)` bucket.
  Splitting `(1 - R2) * tau_bsv^2` over several margins needs a second simplex
  that this decomposition does not build.
- A shared `(… | ID | g)` bucket is all-or-nothing: either every margin's
  predictor carries an `r2d2` statement or none does.
- `cor(:, ID)` composes with a shared-ID decomposition and controls its LKJ
  `eta`; the marginal scales remain derived.
- Non-centred shared-ID `resample_groups` replay is supported. Adaptive
  centering (`centered_groups`), plain-group R2D2 resampling, stratified
  `gr(g, by=b)` groups, and `mm(...)` multi-membership terms remain unsupported.
- A column that also carries its own `effect(lp, coef) ~ Normal(loc, scale)`
  statement is dropped from the simplex and keeps that explicit prior.

## Random-effect R2D2M2 and per-margin ICC: `sd(...) ~ r2d2(...)`

Use the same marker on an `sd` address when the decomposition is over the
marginal variances of a shared random-effect block rather than over one linear
predictor's population coefficients:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
joint_ranef_r2d2 = (@brm begin
    sigma_pk ~ Exponential(1)
    sigma_qt ~ Exponential(1)

    log_Vc   ~ 1 + (1 | p | subject)
    log_k10  ~ 1 + (1 | p | subject)
    qt_base  ~ 1 + (1 | p | subject)
    qt_slope ~ 1 + (1 | p | subject)

    sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=2,
                    concentration=1, reference_scale=sigma_pk)
    sd(qt_base, p)  ~ r2d2(reference_scale=sigma_qt)
    sd(qt_slope, p) ~ r2d2(reference_scale=sigma_qt)
    cor(:, p) ~ LKJCholesky(4, 2)
end)((; subject=[1, 1, 2, 2]))
""", :joint_ranef_r2d2; title="Joint random-effect R2D2M2")
```

The block-wide statement samples one global R² and one Dirichlet simplex over
the four marginal variances. Margin `j` is reconstructed as

```
tau[j] = reference_scale[j] * sqrt(phi[j] * R2 / (1 - R2))
```

so the simplex allocates explained-variance odds in observation-scale units.
The two more-specific statements change only the QT margins' reference scale;
they do not create extra R² parameters. This is the explicit multi-response
form: BRM does not guess which observation scale belongs to a latent predictor.
The LKJ factor and the non-centred
`b = diag_pre_multiply(tau, L) * z` construction are unchanged.

`R2=Beta(a,b)` and `mean_R2=`/`prec_R2=` are equivalent spellings. `alpha=`
and `concentration=` are aliases for the symmetric Dirichlet concentration.
`reference_scale=` is required and may be a positive numeric formula constant
or an earlier sampled scalar parameter.

For independent per-margin ICC priors, omit the block-wide statement and
address margins separately:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
icc_ranef_r2d2 = (@brm begin
    sigma_pk ~ Exponential(1)
    sigma_qt ~ Exponential(1)

    log_Vc  ~ 1 + (1 | p | subject)
    qt_base ~ 1 + (1 | p | subject)

    sd(log_Vc, p)  ~ r2d2(R2=Beta(1, 1), reference_scale=sigma_pk)
    sd(qt_base, p) ~ r2d2(mean_R2=0.5, prec_R2=2,
                          reference_scale=sigma_qt)
    cor(:, p) ~ LKJCholesky(2, 2)
end)((; subject=[1, 1, 2, 2]))
""", :icc_ranef_r2d2; title="Independent per-margin ICC priors")
```

Each address then owns its own R². A one-margin group emits a deterministic
`simplex[1]`, so `R2 = tau² / (tau² + reference_scale²)` is exactly the ICC.
Margins not addressed by either statement keep their ordinary
half-standard-Normal scale prior. A block may not mix R2D2 addresses with
explicit direct-scale `sd(...)` priors; use separate blocks when both
constructions are required.

Random-effect R2D2 is SBBRMI-only. It supports ordinary non-centred shared-ID
blocks, `cor(:, ID)`, and `reprocess(...; resample_groups=[group])`. Centered,
stratified, and multi-membership blocks fail loudly.

## Bounded scalar parameter priors

A non-data scalar prior may add finite numeric `lower` and/or `upper`
declaration bounds. This is the direct spelling for a fitted positive scale
with the same fixed-hyperparameter Normal kernel as a Stan
`real<lower=0>` parameter:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
bounded_scalar = (@brm begin
    sigma ~ Normal(0, 2; lower=0.0)
    mu ~ 1 + x
    y ~ Normal(mu, sigma)
end)((;
    x=[-1.0, 0.5, 2.0],
    y=[0.2, 1.1, -0.4],
))
""", :bounded_scalar; title="Bounded scalar parameter prior")
```

Only `lower` and `upper` are accepted, both bounds and all distribution
arguments must be numeric formula constants, and two bounds must satisfy
`lower < upper`. BRM emits the bound on the Stan declaration and the ordinary
family kernel in the model block, exactly as in hand-written Stan. This
surface is SBBRMI-only; a bound with sampled hyperparameters needs an explicit
normalized parameterization and is rejected.

## Scalar horseshoe prior: `coef ~ Horseshoe(...)`

[`Horseshoe`](@ref) attaches a scalar Carvalho–Polson–Scott shrinkage
hierarchy to an explicitly named coefficient:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
horseshoe_model = (@brm begin
    beta_sparse ~ Horseshoe(local_scale=0.5, global_scale=0.1)
    sigma ~ Exponential(1)
    mu = beta_sparse * x
    y ~ Normal(mu, sigma)
end)((; x=[-1.0, 0.5, 2.0], y=[0.2, 1.1, -0.4]))
""", :horseshoe_model; title="Scalar horseshoe prior")
```

The emitted non-centered hierarchy is

```text
raw    ~ Normal(0, 1)
lambda ~ half-Cauchy(0, local_scale)
tau    ~ half-Cauchy(0, global_scale)
beta_sparse = raw * lambda * tau
```

A sampled coefficient enters the linear predictor through an **assignment**
(`mu = beta_sparse * x`), not a `~` formula term. A `~` right-hand side is
formula-style only — an intercept plus population/group summands whose
coefficients `SBBRMI` allocates — so `mu ~ 0 + beta_sparse * x` is rejected:
`beta_sparse` is your own sampled scalar, not a data column, and the `.*`
product is what you want to write directly. Use `=` for any linear predictor
that multiplies a sampled parameter by data.

`local_scale` and `global_scale` are optional positive finite formula
constants. Both default to `1.0`; the no-keyword spelling
`Horseshoe()` retains the historical emission byte for byte. Literal
arithmetic such as `local_scale=1/2` is evaluated with Julia semantics.
Positional arguments, unknown keywords, nonnumeric values, zero, negative,
and non-finite scales are rejected during `SBBRMI` construction.

The current surface is scalar and each `coef ~ Horseshoe(...)` call owns its
own `(raw, lambda, tau)` triple. Consequently `tau` is “global” only within
that scalar hierarchy; it is not shared across several coefficients. A
genuinely shared global scale requires a vector/group horseshoe declaration,
which this marker does not imply. The standardized `raw` draw is intentionally
not configurable because its scale would duplicate `lambda`/`tau`.

Like other sbimpl prior surfaces, this is implemented by `SBBRMI`; do not
assume `VBRMI` has the same marker-specific lowering.

## Simplex-valued parameter: `s ~ Dirichlet(...)`

A non-data left-hand side with a `Dirichlet` right-hand side declares a
**simplex-valued parameter** — Stan's `simplex[K]` — rather than a linear
predictor. Like every other scalar-parameter prior declaration it is addressable
by name anywhere later in the formula block, `kernel(...)` cells included. It is
implemented only by the `SBBRMI` StanBlocks backend.

The [multi-axis population PK kernel](@ref) shows the executable base shape:
ordinary formula parameters feed a `kernel(...)` cell, while `ragged(...)`
attaches a secondary observation frame to the subject axis.

The inciting shape is a per-**event** multiplier: `diet` scales dose
bioavailability inside the cell, so what the model needs is a `K`-simplex
parameter indexed by an ordinal level — not a per-observation design column.

### What it emits

For `diet_share ~ Dirichlet(3, 1.0)`, the generated Stan declares the
concentration vector as data, the parameter as `simplex[3]`, and the model
statement as `diet_share ~ dirichlet(diet_share_alpha)`. Complete emissions are
shown only through build-generated comparisons rather than copied Stan fences.

The concentration is registered as **data** under `<name>_alpha`, which is what
sizes the simplex; the name is reserved, so a collision is rejected rather than
overwritten. A `simplex[K]` costs `K - 1` unconstrained coordinates and reports
`K` constrained ones.

Because the concentration is data, Stan drops the Dirichlet log-normalizer — a
function of `alpha` alone — from `target`, exactly as it does for any other `~`
statement with data-only hyperparameters. The sampled distribution is unchanged.

### Accepted spellings

Only the two genuine `Distributions.Dirichlet` constructors, with Julia's
parameterization preserved:

| spelling | meaning |
|---|---|
| `Dirichlet(alpha)` | concentration vector literal, e.g. `Dirichlet([2.0, 1.5, 3.0])` |
| `Dirichlet(K, a)` | symmetric: `K` components, each concentration `a` |

`@brm` is a macro over the formula block, so a bare Julia symbol on the
right-hand side is parsed as a formula **local**, not interpolated —
`Dirichlet(3, alpha)` with a captured `alpha` reaches the backend as a column
carrier and is rejected by name. Spell the concentration as a literal; there is
no `$` escape (`$` outside a quote is a Julia syntax error, so the macro never
sees it). This matches the rest of the formula surface — `r2d2`'s `alpha=` and
`effect(:, x) ~ Normal(0, 0.25)` are literals for the same reason.

Concentrations must be finite and strictly positive, and `K >= 2`: a
one-element simplex is deterministically `[1.0]`, so there is no parameter to
sample.

### Difference from `mo(c)` / `mo1(c)`

[`mo`](@ref) and [`mo1`](@ref) are population linear-predictor terms. Their
Dirichlet increments live inside a submodel and are not addressable from the
formula; what a bare `share ~ mo1(c)` hands back is a per-**row** monotonic
contrast vector of length `nrow(df)`. Indexing that by an ordinal level is a
silent double indirection — it transpiles and `stanc`-checks, but computes
something else. Use `Dirichlet` when you want the simplex itself.

### Current limits

- The left-hand side must be a non-data name. A data-backed response with a
  `Dirichlet` right-hand side is a simplex-valued *likelihood*, which has no
  density/pointwise/predictive support and is rejected by the family path.
- The concentration is a constant, not a hyperprior: `Dirichlet(alpha)` with
  `alpha` a sampled parameter is not supported.
- `Dirichlet(K)` — no concentration — is not a `Distributions.jl` constructor
  and is rejected; write `Dirichlet(K, 1.0)` for the flat case.
- StanBlocks-only, like [`s`](@ref), [`t2`](@ref) and [`r2d2`](@ref); not
  available to `VBRMI`.
