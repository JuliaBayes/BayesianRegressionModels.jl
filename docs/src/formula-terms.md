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
| `hsgp(x…; k=20, c=1.5, iso=true, by=nothing, domain=nothing, orthogonal_to=nothing, centeredness=0)` | Hilbert-space GP approximation over `prod(k)` basis functions; optional fixed partial centering | — |
| `hsgp(x; k=20, cov=:periodic, period)` | periodic Hilbert-space basis: `k` harmonics, `2k` cosine/sine functions | — |
| `ar(time; p=1)` | AR(p) noise process ordered by `time`; only `p=1` is emitted | — |
| `dar(time; p=1)` | direct differenced-AR(1) trajectory with bounded persistence and scaled innovations | — |
| `rw(time)` | direct random-walk trajectory: `dar` with the persistence fixed at zero | — |
| `cdar(step; by=group, cor=C)` | per-group deviations following a damped walk over `step` with innovations correlated across groups by `C` | — |
| `mo(c)`, `mo1(c)` | monotonic effect of an ordered factor via Dirichlet increments | — |
| `me(x, sd)` | measurement-error covariate — `x` is observed with known `sd` | — |
| `interval_censored(x; upper=lloq, lower=0)` | quantified/BLOQ covariate with bounded latent values on BLOQ rows | — |
| `factor(c)`, a bare integer / `CategoricalArray` column | K−1 treatment contrasts under an intercept; K cell means as the first categorical term of a predictor without one ("Cell means" on the [overview page](index.md)) | ✓ |
| `factor(c; ref=k)` | reference level `k` for the term's treatment contrasts | ✓ |
| `factor(c; cmc=false)` | keep K−1 treatment contrasts even in a predictor without an intercept (brms' `cmc`, "cell-mean coding") | ✓ |
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

### Random-walk trajectories

`rw(time)` is `dar(time)` with the increments' persistence fixed at zero — the
path

```
x[1] = 0
x[t+1] = x[t] + sigma * z[t]
```

over the sorted distinct values of `time`, as a direct predictor summand: in
`log_R ~ 1 + rw(time)` the population intercept is the initial level and the
term samples a positive innovation scale `sigma` (default `Normal(0, 0.2)`
truncated at zero) and one standardized innovation `z` per step after the
first. Each row reads the point of its own time value, so a long frame whose
rows share times (several groups per day) gets one shared walk — with unique
times it is the plain path. Address the scale as `sd(:, rw(time))`; there is
no persistence to address, so `ar(:, rw(time))` is refused with a message
saying so. The time column must be nonempty and finite; on replay the grid may
gain new times (a forecast extends the walk with prior innovations). This is
the log-reproduction-number walk of a renewal model
([Epidemic renewal models](renewal.md)), stated as one formula line.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
random_walk_term = (@brm begin
    mu ~ 1 + rw(t)
    effect(mu, Intercept) ~ Normal(0.0, 1.0)
    sd(:, rw(t)) ~ Normal(0.0, 0.05)
    y ~ Normal(mu, 0.3)
end)((;
    t=collect(1.0:8),
    y=[0.4, 1.1, 0.9, 1.6, 1.2, 2.0, 1.8, 2.5],
))
""", :random_walk_term; title="Random-walk trajectory", require_stan=true)
```

### Grouped correlated damped walks

`cdar(step; by=group, cor=C)` gives every level of `group` its own deviation
path over the sorted distinct values of `step`, damped with a shared
persistence `rho` and driven by innovations that are correlated **across
groups** through the Cholesky factor `L` of `C` (`L L' = C`):

```
delta[:, 1] = sigma * L * eta[:, 1]
delta[:, w] = rho * delta[:, w-1] + sigma * sqrt(1 - rho^2) * L * eta[:, w]
```

Each row contributes `delta[group(row), step(row)]` as a direct summand, so
`log_R ~ 1 + rw(time) + cdar(week; by=patch, cor=C)` is a shared random walk
plus spatially correlated weekly patch deviations — the six-patch model of
[Epidemic renewal models](renewal.md), stated on the formula surface. `C` is a `P × P`
symmetric positive-definite matrix, `P` the number of group levels, supplied as
a data field (a matrix-valued field is accepted) or a literal; it is a fixed
hyperparameter, not a sampled covariance. The term samples `sigma > 0`
(default `Normal(0, 0.2)` truncated at zero), `rho` on `[0, 1]` (default
`Normal(0.5, 0.2)` truncated), and `P·W` standardized innovations `eta`;
address them as `sd(:, cdar(step))` and `ar(:, cdar(step))`. On replay the
group levels and `L` are frozen from the fit while the step grid may grow — a
forecast extends the walk with prior innovations — and an unseen group level is
refused.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
grouped_walk = (@brm begin
    mu ~ 1 + cdar(week; by=patch, cor=C)
    sd(:, cdar(week)) ~ Normal(0.0, 0.2)
    ar(:, cdar(week)) ~ Normal(0.8, 0.1)
    y ~ Normal(mu, 0.3)
end)((;
    week=[1, 1, 2, 2, 3, 3],
    patch=["a", "b", "a", "b", "a", "b"],
    y=[0.2, -0.1, 0.4, 0.0, 0.5, 0.1],
    C=[1.0 0.6; 0.6 1.0],
))
""", :grouped_walk; title="Grouped correlated damped walk", require_stan=true)
```

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

### Fixed partial centering of HSGP weights

For a raw, ungrouped squared-exponential HSGP, `centeredness` chooses the
coordinate of each basis weight without changing its physical prior. It may be
one scalar shared by all weights or a vector of length `prod(k)`, supplied
literally or through a data column. With spectral standard deviation `s`, unit
normal `z`, and centeredness `c` in `[0,1]`, BRM samples and reconstructs

```text
u ~ Normal(0, s^c)
w = s^(1-c) u = s z
```

Thus `c=0` is the default noncentered coordinate and `c=1` is centered;
intermediate values partially center that frequency. Both StanBlocks and
Turing use the same log-spectral-scale calculation and endpoint-safe `c=0`
path, so a numerically vanished high-frequency scale cannot turn `0 * -Inf`
into `NaN`.

[`select_hsgp_centeredness`](@ref) applies the pilot rule used in the
[adaptive HSGP case study](adaptive-centering.md): rows are pilot draws,
columns are basis frequencies, and candidates are fixed before the refit. The
selector is not an online warmup controller. Periodic, latent-input,
group-specific, and `orthogonal_to` HSGPs reject nonzero partial centering
until those geometries have their own verified coordinate contract.

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

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
periodic_time_model = (@brm begin
    mu ~ 1 + hsgp(hours_since_dose; k=8, cov=:periodic, period=24.0)
    length_scale(:, hsgp(hours_since_dose)) ~ LogNormal(0, 0.5)
    sd(:, hsgp(hours_since_dose))           ~ Normal(0, 0.3)
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end)((;
    hours_since_dose=[0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.5, 48.0, 72.0, 168.0],
    y=[0.2, 0.9, 1.1, 0.8, 0.4, -0.6, 0.3, 0.1, -0.2, 0.0],
))
""", :periodic_time_model; title="Periodic (24 h) time effect over hours since first dose")
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

### Exact total coefficients

The StanBlocks backend uses **total coefficients** by default for eligible
independent random effects. For example, in

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
builder = @brm begin
    mu ~ 1 + x + (1 | subject_effect | subject)
    effect(mu, Intercept) ~ Normal(0, 3)
    sd(:, subject_effect) ~ Normal(0, 1)
    y ~ Normal(mu, 1)
end
data = (; subject=[1, 1, 2, 2, 3, 3], x=[0., 1., 0., 1., 0., 1.],
          y=[0.2, 0.8, -0.1, 0.9, 0.4, 1.2])
total_intercept_model = builder(data)
""", :total_intercept_model; title="Exact total intercepts with a fixed slope")
```

the sampled group intercept is `total[j] = intercept + deviation[j]`.
The fixed slope of `x` remains explicit. BRM integrates out the shared intercept
using its original prior; this preserves the likelihood and posterior of the
totals and hyperparameters. All groups have the same status, and the sampler
uses one total per group without an extra mean direction.

With `intercept ~ Normal(m, s)` and independent `deviation[j] ~ Normal(0, tau)`,
the exact induced prior is

```math
\mathbf{total}\mid\tau \sim
\mathcal N\!\left(m\mathbf 1,\;\tau^2I+s^2\mathbf 1\mathbf 1^\mathsf T\right).
```

The density uses group sums and a small population-coefficient precision matrix,
so its work is linear in the number of groups for a fixed number of coefficients.
Student-t population priors (`TDist` or `LocationScale(..., TDist(...))`) use an
exact Gaussian scale mixture; `Flat()` denotes an improper uniform population
prior. Marginalization preserves the supported population prior specified in
the formula.

`total_effect_blocks(sb)` reports the selected blocks. `total_groups=()` opts out;
`total_groups=[:subject]` requires that group to be eligible. Explicit
`centered_groups=[:subject]` selects conventional centered deviations.

Automatic selection currently requires one grouping structure per predictor,
independent margins, supported population priors, and a known relation between
the population and random-effect columns. Raw numeric columns, pure data
expressions, and centered/scaled population columns with matching raw random
columns are supported. Correlated, crossed, stratified, multi-membership and
R2D2 blocks use the conventional representation. A request for totals on an
unsupported group raises an error.

Automatic selection retains the conventional representation if some ordinary
group blocks require it. A single adaptive wrapper currently uses one geometry
family: totals, ordinary random effects, or HSGP weights.

#### WarmupHMC and recovery

BRM discovers each total's scale and supplies independent centering controls:

```julia
using Random, Enzyme, WarmupHMC, BridgeStan, StanBlocks
using DifferentiationInterface: AutoEnzyme

sb = SBBRMI(total_intercept_model)
problem = StanBlocks.stan_instantiate(sb.model)
backend = AutoEnzyme(;
    mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation=Enzyme.Const)
adaptive = adaptive_centering_problem(sb, problem, backend;
                                     centeredness=0.0)
fit = WarmupHMC.adaptive_warmup_mcmc(Xoshiro(1), adaptive;
    n_draws=2000, nonlinear_adapt=true)
```

`centeredness=1` means model-scale totals; `0` means scaled totals. Intermediate
values may differ by group and coefficient. The marginal totals remain
correlated under their exact prior, including at the NCP endpoint. Online
selection uses WarmupHMC's default position-gradient loss. For a fixed fit, use
`nonlinear_adapt=false`.

For a post-hoc refit, call `select_total_centeredness(sb, draws, names)` on a
pilot in the compiled model frame, then pass its `centeredness` vector to
`adaptive_centering_problem`. The default `criterion=:position` needs positions;
`criterion=:gradient` additionally needs matching compiled-frame gradients.
Both inputs are draws × coordinates. Selection makes no additional gradient
calls; include the pilot's cost when comparing full workflows.

```julia
names = BridgeStan.param_unc_names(problem.model)
draws = permutedims(fit.posterior_position)
recovered = recover_population_draws(sb, draws, names; rng=Xoshiro(2))
recovered[:mu].population  # original shared intercept, conditionally recovered
recovered[:mu].totals      # sampled subject-specific intercepts
recovered[:mu].deviations  # totals minus the recovered shared intercept
```

The compiled model also exposes the recovered coefficients and deviations in
generated quantities. Recovery adds conditional randomness, so compare sampler
efficiency on consistent scientific quantities and distinguish recovered means
from sampled totals. The [pupil study](pupil-centering.md) explains this distinction.

`population_draws(...; groups=:subject, rng=...)` substitutes a recovered
population draw for each group's total. For new groups, use the reusable
`generative_plan(builder, data)` form, rebuild with `generative_plan(plan, new_data)`,
and call `transport_draws`. Existing groups retain their totals; new groups share
one recovered population draw per posterior draw. `resample=:subject` redraws
existing groups too. Total blocks require frozen preprocessing for replay and use
this transport route for resampling; `reprocess(...; resample_groups=...)` is not
supported for them. Improper population priors have no prior-predictive distribution.

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

### Downstream grouped terms: the reusable-term pattern

A parametric shape with group-varying parameters — a single-peak transient, a
saturating sigmoid — belongs in the downstream package whose science needs it,
not in BRM core. BRM core promotes such a term only once two or more
downstream packages have shipped the same curves; until then the package ships
its own term through the supported group-block seams, first-class rather than
as an escape hatch. The worked example is bordet's `transient` /
`saturating` pair (in the bordet tree, not here).

The recipe, for a term used in nested predictor position
(`mu ~ 1 + transient(logt; series)`), where it runs on both backends:

1. Declare the marker in the downstream module (`function transient end`).
2. Declare one structured-latent field per parameter group with the fields
   form of `_sb_term_group_block`. The `group` spec names the grouping
   column (`(; kwarg=:series)` reads it from the call's `series=` keyword);
   `prior=:correlated_normal` draws the per-group parameters through BRM's
   non-centered LKJ block on both backends.
3. Route the nested summand to your emitter:
   `_sb_is_direct_term(::typeof(transient)) = true`.
4. Prepare once, replay frozen: `_brm_prepares_term` is true;
   `_brm_prepare_term` delegates to `_brm_prepare_structured_term` and bakes
   the axis values the native effect needs into state; `_brm_replay_term`
   refreshes indices and axis values against the fitted levels.
5. Assemble per-row values natively for Turing in
   `_brm_native_structured_effect` — pure Julia math over the block row.
6. Emit the Stan contribution in `_sb_predictor_term!`, reusing StanBlocks
   builtins (e.g. `biomarker_time_response`) and threading the preallocated
   block via `_sb_find_group_block`, so the downstream module ships no
   custom Stan code. (Extend, never shadow:
   `import BayesianRegressionModels: _sb_term_group_block, _sb_is_direct_term,
   _brm_prepares_term, _brm_prepare_term, _brm_replay_term,
   _brm_native_structured_effect, _sb_predictor_term!`.)

```julia
module DownstreamTerms
using StanBlocks
import BayesianRegressionModels
const BRM = BayesianRegressionModels
import BayesianRegressionModels: _sb_term_group_block, _sb_is_direct_term,
    _brm_prepares_term, _brm_prepare_term, _brm_replay_term,
    _brm_native_structured_effect, _sb_predictor_term!

function transient end
_sb_is_direct_term(::typeof(transient)) = true
_sb_term_group_block(::typeof(transient)) = (; fields=[
    (; name=:transient, n_per_group=3, group=(; kwarg=:series),
       prior=:correlated_normal),
])

# Independent Julia reference math, shared by the native effect below.
bump_math(x, loc, ls, mag) = begin
    xi = (x - loc) * exp(ls)
    s = 1 / (1 + exp(-xi))
    sm = 1 / (1 + exp(xi))
    exp(log(s) + log(sm)) * mag
end

_brm_prepares_term(::BRM.ExprColumn{typeof(transient)}) = true
function _brm_prepare_term(term::BRM.ExprColumn{typeof(transient)}, target,
                           context)
    base = BRM._brm_prepare_structured_term(term, target, context,
        _sb_term_group_block(transient, term))
    xkey, xraw = BRM._brm_term_data(
        :transient, only(BRM.getargs(term)), context)
    BRM._BRMPreparedTerm(base.callable, base.source,
        merge(base.state, (; xname=xkey, x=collect(Float64, xraw))),
        base.dependencies)
end
function _brm_replay_term(::typeof(transient), training, fresh, context)
    fields = map(training.state.fields) do field
        raw = context.data[field.source]
        merge(field, (; idx=BRM._brm_apply_levels(field.levels, raw)))
    end
    xraw = context.data[training.state.xname]
    BRM._BRMPreparedTerm(training.callable, training.source,
        merge(training.state,
            (; fields=Tuple(fields), x=collect(Float64, xraw))),
        training.dependencies)
end

function _brm_native_structured_effect(
        term::BRM._BRMPreparedTerm{typeof(transient)}, block, field)
    st = term.state
    [bump_math(st.x[i], block[field.idx[i], 1], block[field.idx[i], 2],
               block[field.idx[i], 3]) for i in eachindex(field.idx)]
end

function _sb_predictor_term!(stmts, data, ::typeof(transient), t;
                             target::Symbol, group_block_lookup=Dict(),
                             kwargs...)
    xname, xraw = BRM._sb_inner_data(:transient, only(BRM.getargs(t)))
    data[xname] = collect(Float64, xraw)
    info = BRM._sb_find_group_block(transient, t, group_block_lookup)
    isnothing(info) && error("sbimpl: `transient` found no allocated block")
    (; block_name, idx_name) = info
    loc = Symbol(:transient_, target, :_loc)
    ls = Symbol(:transient_, target, :_ls)
    mag = Symbol(:transient_, target, :_mag)
    col = Symbol(:transient_, target, :_, xname)
    push!(stmts, :($loc = $(block_name)[$(idx_name), 1]))
    push!(stmts, :($ls = $(block_name)[$(idx_name), 2]))
    push!(stmts, :($mag = $(block_name)[$(idx_name), 3]))
    push!(stmts, :($col = biomarker_time_response($xname, $loc, $ls, $mag)))
    col
end
end
```

7. Fit through `SBBRMI(brmi; mod=DownstreamTerms)` or `TuringBRMI(brmi)`.
   The Stan path lowers to stanc-clean code; the Turing path draws the same
   non-centered LKJ block natively and assembles bit-exact effects —
   `test/downstream_group_block_term.jl` verifies both against independent
   Julia math.

A second shape follows the same seven steps. A saturating 0-to-1 dose
multiplier is two per-group parameters (`loc`, `log_slope`) whose Stan emit
exps the log-sigmoid builtin and whose native effect is one line of Julia:

```julia
function saturating end
_sb_is_direct_term(::typeof(saturating)) = true
_sb_term_group_block(::typeof(saturating)) = (; fields=[
    (; name=:saturating, n_per_group=2, group=(; kwarg=:series),
       prior=:correlated_normal),
])
sigmoid_math(x, loc, ls) = 1 / (1 + exp(-((x - loc) * exp(ls))))
# ... same prep/replay shape as above, with `saturating` for `transient` ...
function _brm_native_structured_effect(
        term::BRM._BRMPreparedTerm{typeof(saturating)}, block, field)
    st = term.state
    [sigmoid_math(st.x[i], block[field.idx[i], 1], block[field.idx[i], 2])
     for i in eachindex(field.idx)]
end
function _sb_predictor_term!(stmts, data, ::typeof(saturating), t;
                             target::Symbol, group_block_lookup=Dict(),
                             kwargs...)
    xname, xraw = BRM._sb_inner_data(:saturating, only(BRM.getargs(t)))
    data[xname] = collect(Float64, xraw)
    info = BRM._sb_find_group_block(saturating, t, group_block_lookup)
    isnothing(info) && error("sbimpl: `saturating` found no allocated block")
    (; block_name, idx_name) = info
    loc = Symbol(:saturating_, target, :_loc)
    ls = Symbol(:saturating_, target, :_ls)
    col = Symbol(:saturating_, target, :_, xname)
    push!(stmts, :($loc = $(block_name)[$(idx_name), 1]))
    push!(stmts, :($ls = $(block_name)[$(idx_name), 2]))
    push!(stmts, :($col = exp(biomarker_dose_response($xname, $loc, $ls))))
    col
end
```

Shapes compose two ways. Additively, as ordinary predictor summands:

```julia
mu ~ 1 + transient(logt; series) + saturating(logd; series)
```

Multiplicatively — baseline plus bump times response, the bordet mean shape —
through intercept-free named predictors combined by assignment (write `*`:
the formula layer is element-wise by intent and sbimpl dots it — a literal
`.*` is evaluated at parse time and fails):

```text
brmi = (@brm df begin
    sigma ~ Exponential(1)
    base ~ Normal(0, 1)
    bump ~ 0 + transient(logt; series)
    resp ~ 0 + saturating(logd; series)
    mu = base + bump * resp
    y ~ Normal(mu, sigma)
end)
sb = SBBRMI(brmi; mod=DownstreamTerms)
tb = TuringBRMI(brmi)
```

Each shape owns its own per-group hierarchy (two LKJ blocks here):
parameters correlate within a shape, not across shapes. A model that needs
the bump and the sigmoid parameters jointly correlated (all six in one
covariance) wants one joint term with `n_per_group=6` instead — formerly
bordet's `biomarker_hierarchical_parametric` hatch (shed as dead code; the
shared-bucket + `kernel(...)` composition in `test/kernel_biomarker_cell.jl`
is the current idiom), and what its `transient` / `saturating` worked example
will decide per fit. Term-internal
prior statements (`sd(mu, transient(logt))`) are not addressed yet: the
hierarchical scales keep their shared defaults, and naming a new term in a
prior address needs core registration alongside `_TERM_HEADS`.

Verified contract (`test/downstream_group_block_term.jl` pins all of it on
both backends): stanc-clean Stan lowering, Turing effects bit-exact versus
independent Julia math with finite joint densities, `brm_descriptor`, and
`reprocess` / `restan_data` on fitted group levels. Replaying a group level
the fit never saw stays refused by the shared ranef floor — the fitted model
has no coordinate for it, on any backend, for core grouped terms alike.

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
`t2` is implemented by both the StanBlocks and Turing backends; it remains
unavailable to `VBRMI`.

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
  them; the isotropic form has a single shared one. With `by=g` the term keeps
  one shared length scale and amplitude unless hyper-predictor statements
  predict them per group ([Hyper-predictors](@ref)).
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

### Hyper-predictors

A bare grouped term shares one length scale and one amplitude across all
groups — only the basis weights vary per group. When each group needs its own
smoothness or amplitude (per-biomarker hypers), predict the hypers with
hyper-predictor statements: `log(length_scale(hsgp(x))) ~ 1 + (1 | g)` and
`log(sd(hsgp(x))) ~ 1 + (1 | g)`. Both hypers use a log link, and each group
evaluates its own linear predictor to `eta_rho[g]` / `eta_sigma[g]`, consumed
as `rho_g = max(exp(eta_rho[g]), rho_lower)` and
`sigma_g = exp(eta_sigma[g])`.

- The hyper linear predictor accepts `1` and `(1 | g)` only. Smooths are
  refused, population slopes need level-grid covariate semantics that are not
  decided yet, and the random effect must be exactly `(1 | g)` over the
  addressed term's own `by=` grouping. An ungrouped term takes an
  intercept-only scalar (`~ 1`); a random effect without grouping is refused.
- Defaults reproduce today's predictive prior: the intercept carries
  `Normal(0, 1)` on the log scale (that is `LogNormal(0, 1)` on the natural
  scale), the random-effect SD carries half-`Normal(0, 1)`, and the group
  deviations are non-centered. An explicit `length_scale(:, hsgp(x)) ~ ...` /
  `sd(:, hsgp(x)) ~ ...` statement retargets from the shared scalar to the
  hyper-LP intercept and sets its own support, exactly as term priors do.
- The approximation-validity floor (next section) applies per group: each
  `rho_g` is maximised with the same domain-derived floor the bare term
  declares, so no group can silently leave the kernel the basis approximates.
- Scope is deliberately narrow: `hsgp` only (`gp` is refused by name),
  isotropic (`iso=true`) only, and neither periodic nor model-derived axes.
  Lowering is SBBRMI-only.
- Posterior addressing: the sampled hyper coefficients resolve through
  `brm_term_coordinates` under the `:length_scale_intercept` /
  `:length_scale_ranef_sd` / `:length_scale_ranef_z` roles (and the `:sd_*`
  mirrors) — one coordinate for the intercept and SD, one per group for the
  deviations. The `:length_scale` / `:sd` roles themselves name no sampled
  carrier on a predicted hyper and redirect to these roles; the per-group
  values are deterministic transforms with no role.
- After fitting, `hsgp_boundary_check` reports per-group
  posterior-mean-`rho` over the domain margin (flagged at 1.0) and the
  per-group probability the hyper value sits below the validity floor —
  the executable form of the sizing guidance below.

#### Hyper recovery needs a wider domain than latent recovery

A good latent fit does **not** imply the hypers recover: fitting the smooth
only needs the basis to span the data, while hyper posteriors need the HSGP
prior covariance to match the kernel on the data grid — a strictly stronger
requirement. The Dirichlet boundary pins the prior toward zero within roughly
one length scale of `±L`, so when the plausible `rho` approaches the domain
margin `(c-1)·max|x-mean(x)|` the prior variance collapses at the data edges
and the `rho` posterior drags low with tight, confident-wrong credible
intervals (snag `hyper-predictor-d453ecf9`: truth `rho=2.1` recovered as
`0.76±0.12` at `c=1.5, k=10` on a 6-point grid, while the latent RMSE was
0.07).

Size `(c, k)` for hyper recovery from the plausible hyper range before fitting:

- **Margin clears the largest plausible `rho`:**
  `(c-1)·max|x-mean(x)| ≳ 2·rho_max`. The default `c=1.5` leaves a margin of
  half the data half-range; once `rho_max` approaches that margin, widen `c`.
- **Floor clears the smallest plausible `rho`:** raising `c` raises `L` and the
  floor `(4L/π)·√(log(100)/(k²-1))` with it, so raise `k` alongside until the
  floor sits below `rho_min` with headroom (target `rho_min/2`).
- **Widen until the posterior stabilises:** intermediate domains can overshoot
  (same case: `c=2.5` recovers `3.35`, `c=3.0` recovers `3.86`, `c=4.0`
  recovers `2.75` against the exact-GP `2.71`). Do not stop at the first `c`
  that moves the posterior.

Two further posterior signatures to read correctly:

- Mass with `eta` below `log(rho_lower)` is likelihood-flat — the floor binds
  by `max()` inside the basis function, so `rho_vec = exp(eta)` reports prior
  mass there as posterior. Check `P(rho_g < rho_lower)`; if substantial, widen
  `k` to lower the floor and re-fit.
- With few groups the half-`Normal(0, 1)` hyper SD shrinks group hypers together
  (expected hierarchical pooling, not a bug). A chain stuck at huge `rho` with
  a vanished smooth is the `rho→∞` degenerate tail all HSGP hypers share —
  check `Rhat`: it is non-convergence, not information.

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
`effect(lp, coef) ~ Normal(...)` — and is implemented by both the `SBBRMI`
and `TuringBRMI` backends.

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
  statement is dropped from the simplex and keeps that explicit prior. The
  default-layer spelling `effect(:, coef) ~ Normal(...)` excludes that column
  in every predictor it reaches, exactly like the predictor-specific one; the
  remaining columns and the random-effect residual are still decomposed.
- Excluding **every** non-intercept column of an `r2d2`-scoped predictor this
  way is refused. There is then nothing left to allocate, and the only
  consistent emission would drop `R2`/`phi` and fix the random-effect scale at
  the bare `tau_bsv` with no prior — a silently different model. Either keep at
  least one column unaddressed, or move the decomposition onto the
  random-effect scale with `sd(lp, ID) ~ r2d2(reference_scale=...)` (the
  random-effect R2D2M2/ICC form below composes with per-column Normal priors;
  a shared bucket is all-or-nothing, so switch the whole bucket). A predictor
  with **no** non-intercept population column at all (`log_ka ~ 1 + (1 | p |
  g)`, forced into an `r2d2` statement by the all-or-nothing rule) is the one
  legitimate zero-share shape: nothing to explain, so the whole `tau_bsv` is
  its random-effect scale.

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

## Joint R2D2M2 budget over coefficients, contrasts and random effects: `include=`

When several linear predictors share one correlated block AND one covariate
right-hand side, brms' R2D2M2 puts ONE global R² and ONE Dirichlet over the
union of the population coefficients (continuous and categorical) and the
random-effect variances. Spell that by adding `include=` to the block-wide
statement:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
joint_budget_r2d2 = (@brm begin
    sigma_pk ~ Exponential(1)
    sigma_qt ~ Exponential(1)

    log_Vc  ~ 1 + wt + indication + (1 | p | subject)
    log_k10 ~ 1 + wt + indication + (1 | p | subject)
    qt_base ~ 1 + wt + indication + (1 | p | subject)

    sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=2, concentration=1,
                    reference_scale=sigma_pk,
                    include=(:population, :contrasts))
    sd(qt_base, p) ~ r2d2(reference_scale=sigma_qt)
    cor(:, p) ~ LKJCholesky(3, 2)
end)((;
    subject=[1, 1, 2, 2, 3, 3],
    wt=[-1.0, 0.5, 0.2, -0.3, 1.1, -0.6],
    indication=[1, 2, 1, 2, 2, 1],
))
""", :joint_budget_r2d2; title="Joint R2D2M2 budget with include=")
```

`include=` names which population components of every predictor slicing
`|p|` join the block's single R²/Dirichlet: `:population` is the non-intercept
continuous `beta_pop` columns, `:contrasts` the categorical treatment-contrast
coefficients (`cat_*` blocks), and `:ranef` the margins, which are always
allocated and may be listed for readability. The simplex is ordered margins
first, then each scoped predictor's population columns and contrast blocks in
formula order. A component of predictor `m` is measured in `m`'s own margin
reference — its `Intercept` margin, or its single margin — so a margin keeps

```
tau[j] = reference_scale[m] * sqrt(phi[j] * R2 / (1 - R2))
```

and a coefficient or contrast takes

```
beta_scale[k] = reference_scale[m] * sqrt(phi[k] * R2 / ((1 - R2) * Var(x_k)))
```

which keeps the whole-predictor form's design-column variance adjustment (a
contrast's dummy column has variance `p * (1 - p)` for level frequency `p`).
Intercepts stay outside and keep their ordinary or explicitly overridden prior.
Per-margin `reference_scale` overrides work exactly as above; the example gives
`qt_base`'s margin, coefficients, and contrast the QT residual scale.

With `include=`, `reference_scale=` becomes optional. An omitted margin
reference is a sampled half-standard-normal parameter, so the statement then
allocates **latent** between-subject variation: nothing observed anchors the
unit, `R2` is not an outcome R², and the sampled reference and `R2` are
identified by the data only through their product `reference_scale² * R2 /
(1 - R2)`. Prefer explicit references whenever a margin's scale is known.

Under a joint budget a per-column `effect(lp, coef) ~ Normal(...)` or
`effect(lp, categorical) ~ Normal(...)` statement inside the scope is refused:
it would silently pull that coefficient out of the simplex. A scoped predictor
may not also carry `effect(lp, :) ~ r2d2(...)`, and `include=` is accepted on
the block-wide statement only — not on a per-margin override, not on the ICC
form. `ranef_effect_priors` reports the joint statement with its `include`
keyword. The emitted carriers keep their names (`pop_<lp>_beta_pop`,
`cat_<lp>_<col>_beta`, the block's derived `tau`), so
`brm_population_effect_coordinates`, `brm_ranef_sd_coordinates`, and
`reprocess(...; resample_groups=[group])` are unchanged.

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
implemented by both `SBBRMI` and `TuringBRMI`.

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
- Not available to `VBRMI`. [`s`](@ref), [`t2`](@ref), [`r2d2`](@ref), and
  this simplex surface are supported by both `SBBRMI` and `TuringBRMI`.

## Vector-valued parameter: `x ~ MvNormal(...)`

A non-data left-hand side with an `MvNormal` right-hand side declares a
**vector-valued parameter** — Stan's `vector[n]` — rather than a linear
predictor. It is addressable by name anywhere later in the formula block: as an
argument of a `@deffun` in a top-level assignment, inside a custom family call,
or in a `kernel(...)` cell. `SBBRMI` emits Stan's vector-parameter form
(decision `187g4va`), while `TuringBRMI` lowers the retained generic callable
directly.

The inciting shape is a latent path whose innovations the formula wants to
state directly — the log-reproduction-number random walk of a renewal model
written with explicit innovations, `eps ~ MvNormal(zeros(T - 1), 1.0)`, or a
vector of per-group seeds, `log_I0 ~ MvNormal(seed_mean, 0.25 * I)`. (The
[Epidemic renewal models](renewal.md) page states the same walk with the
`rw(time)` term and the seeds as cell means.)
Before this seam the only way to get such a vector was a one-cell `kernel(...)`
with a dummy grouping random effect.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
using StanBlocks
StanBlocks.@deffun begin
    rw_path(init::real, sig::real, eps::vector[K])::vector[K + 1] =
        append_row(init, init + sig * cumulative_sum(eps))
end
random_walk = (@brm begin
    sig  ~ Normal(0.0, 0.05; lower=0.0)
    init ~ Normal(0.0, 1.0)
    eps  ~ MvNormal(zeros(length(time) - 1), 1.0)   # vector[T-1] of iid N(0, 1)
    walk = rw_path(init, sig, eps)
    y ~ Normal(walk, 0.3)
end)((;
    time=collect(1.0:8),
    y=[0.4, 1.1, 0.9, 1.6, 1.2, 2.0, 1.8, 2.5],
))
""", :random_walk; title="Random walk on a top-level vector parameter", require_stan=true)
```

### What it emits

For `eps ~ MvNormal(zeros(length(time) - 1), 1.0)` the generated Stan declares
the dimension as data (`<name>_n`), the mean as data (`<name>_mu`), the
parameter as `vector[eps_n] eps`, and the model statement as
`eps ~ normal(eps_mu, 1.0)` — Stan's vectorised univariate normal, no matrix.
A full covariance emits `multi_normal(<name>_mu, <name>_scale)`. The three data
names are reserved, so a collision is rejected rather than overwritten; a data
column used as the mean is referenced by its own name instead of being copied.

A vector nothing later reads is a generated quantity, not a sampler parameter:
StanBlocks' activity analysis lowers an unused declaration, exactly as it does
for an unused random effect. A top-level assignment's right-hand side is
Julia-evaluated by the macro, so Stan builtins (`append_row`, `cumulative_sum`)
reach it through a `@deffun`, as in the example.

### Accepted spellings

Only the genuine `Distributions.MvNormal` constructors, with Julia's
parameterization preserved (the `σ` forms are deprecated in Distributions.jl
but keep their Distributions meaning — a standard deviation):

| spelling | meaning | emitted |
|---|---|---|
| `MvNormal(mu, Σ::Matrix)` | mean and covariance | `multi_normal` |
| `MvNormal(mu, Diagonal(v))` | variances on the diagonal | vectorised `normal` with `sqrt.(v)` |
| `MvNormal(mu, λ * I)` | covariance `λ·I` | vectorised `normal` with `sqrt(λ)` |
| `MvNormal(Σ)` | zero mean | `multi_normal` |
| `MvNormal(mu, σ::Real)` | isotropic, standard deviation `σ` | vectorised `normal` |
| `MvNormal(mu, σ::Vector)` | standard deviations | vectorised `normal` |
| `MvNormal(n, σ::Real)` | zero mean, `n` components | vectorised `normal` |

The mean is a numeric vector literal, a **data column**, or a data-only
expression over data columns (`zeros(length(time) - 1)`, `fill(c, k)`) —
evaluated in Julia at build time and shipped as data. The scale may be a
literal, a data column, or a **sampled scalar parameter** (`MvNormal(zeros(k), sig)`
emits `normal(eps_mu, sig)`). The dimension comes from the mean, from a
vector/matrix scale, or from the integer form; when the mean itself involves a
parameter and the scale is a scalar, there is no data-determinable size and the
statement is rejected with a message saying so.

### Current limits

- The left-hand side must be a non-data name. A vector-valued *response* keeps
  its own spelling ([`MvNormalCholesky`](@ref), `[y1, y2] ~ ...`); `MvNormal`
  is not added to the family table.
- No keywords: bounds on a multivariate normal parameter are not supported.
- A parameter-bearing *covariance* (an `LKJCovarianceFactor` product, a sampled
  matrix) is not admitted here; use the scalar-scale form or compose the
  vector inside a `@deffun`.
- Not available to `VBRMI`. `TuringBRMI` retains the generic callable, so the
  same statement lowers there through its own path; [`Dirichlet`](@ref),
  [`s`](@ref), and [`r2d2`](@ref) are likewise supported by both executable
  backends.

## Function-valued arguments: lambdas and `do` blocks

A top-level assignment may hand a **function** to a higher-order `@deffun`:
written inline as the first argument, or as a trailing `do` block, which is the
same thing — `@brm` folds the block in as the first positional argument, as
`kernel(...) do … end` already does. Higher-order functions therefore take
their function argument **first**.

The body is Stan-side code, resolved by StanBlocks as a closure. It may read

- its own parameters (`l` below),
- **sampled model parameters** (`rate`) — they are passed into the generated
  Stan function as arguments, so they are estimated, not frozen in, and
- **shared data vectors** by name: a free name that is a data field and not a
  formula name is registered as Stan data, the way a `kernel(...)` cell body
  captures one.

The example is a distributed lag whose weights decay at a sampled rate — a
filter that a data vector of weights could not express, because the weights
depend on a parameter.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
using StanBlocks
StanBlocks.@deffun begin
    # Y[t] = sum over lags l = 0 .. L-1 of f(l) * x[t - l]
    lagged(f, x::vector[T], L::int)::vector[T] = begin
        Y::vector[T]
        for t in 1:T
            acc = 0.0
            for l in 0:(L - 1)
                if t - l >= 1
                    acc += f(l) * x[t - l]
                end
            end
            Y[t] = acc
        end
        Y
    end
end
decaying_lag = (@brm begin
    rate ~ Normal(1.0, 0.5; lower=0.0)
    log_mu ~ 1 + rw(time)
    Y = lagged(exp(log_mu), 3) do l
        exp(-rate * l)                 # reads the sampled `rate`
    end
    y ~ Poisson(Y)
end)((;
    time=collect(1.0:12),
    y=[3, 4, 6, 5, 8, 9, 12, 11, 15, 18, 17, 21],
))
""", :decaying_lag; title="A do block that reads a sampled parameter", require_stan=true)
```

In the generated Stan the closure is a function of its own (`lagged_closure_1`)
and the call site passes what the body captured:
`Y = lagged_closure_1(rate, exp(log_mu), 3)`. The inline spelling
`lagged(l -> exp(-rate * l), exp(log_mu), 3)` emits the same program.

The callee may live in another module — a package that ships `@deffun`
operators — as long as the calling module imports it and is passed as `mod=`.
This is a StanBlocks feature: `TuringBRMI` refuses the assignment by name, as
it does for every Stan-only callee.
