````@raw html
---
title: Adaptive HSGP centering
description: "Reproducing the motorcycle HSGP case study: a noncentered pilot, per-frequency partial centering, and a fresh posterior fit."
---
````

# Adaptive HSGP centering

The difficult geometry of a Gaussian process is not necessarily resolved by
choosing one centered or noncentered parameterization for every coefficient.
Low and high HSGP frequencies can need different coordinates. This case study
follows [Generable's motorcycle example](https://www.generable.com/post/hsgp-reparam):
fit the noncentered model, inspect its geometry, choose one centering per basis
weight, and fit the reparameterized model from scratch.

There are **two fits**, not three. The centered plots below transform the
noncentered pilot draws; they do not come from sampling a centered model.
Online adaptation is a separate extension of this workflow.

## The model

All 133 `MASS::mcycle` observations are used. A zero-mean squared-exponential
HSGP models the conditional mean, and another models the log conditional
standard deviation:

```text
mu(t)  = HSGP_mu(t)
eta(t) = HSGP_sigma(t)
y(t) ~ Normal(mu(t), exp(eta(t)))
```

Acceleration is divided by its sample standard deviation. Time is mapped to
`[-1,1]`. Each GP has **20** sine basis functions on `[-1.5,1.5]`, with the
source's `1/sqrt(1.5)` normalization. There is no additional population intercept.

![Low and high HSGP basis functions and their prior spectral scales](assets/adaptive-hsgp/hsgp_basis.png)

The left panel shows frequencies 1, 2, 19 and 20; the dotted vertical lines mark
the observed domain. The right panel shows how increasing the length scale
suppresses the high-frequency weights.

### Hyperpriors: verify the implementation, not just the notation

The source assigns independent `Normal(0,4)` priors to the **log** length scale
and **log** marginal standard deviation of each GP. BRM expresses these as
`LogNormal(0,4)` on the four positive parameters. The change of variables is
part of that equivalence:

```text
logpdf(LogNormal(0,4), exp(q)) + q = logpdf(Normal(0,4), q)
```

The `+q` is the unconstraining Jacobian. An additional lower bound on the
length scale would change the model; matching distribution names alone would
not detect that error.

`research/adaptive_centering/audit_source.jl` compiles the immutable original
Stan program and compares it with the actual BRM-generated Stan model. Its
26 noncentered-model checks cover the coordinate mapping, normalized target including the
Jacobian, and all 44 gradient components. Across six tested points the largest
absolute density and gradient differences were `5.7e-14` and `4.3e-14`.
The companion `audit_partial_source.jl` checks the fresh partial model against
the original adaptive Stan program at 16 saved posterior positions. All 51
checks pass; the largest density and gradient differences are `1.2e-13` and
`5.0e-12`. These are comparisons of the actual generated targets, including
the source hyperpriors, not just algebraic prior identities.

### One BRM formula and its generated backends

This executable example reads the same full dataset and uses the same
20-frequency model as the sampling run. The tabs expose the generated models;
generating a Turing model does not imply that it has passed the gradient
performance gate or has been sampled.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
using Statistics

function adaptive_motorcycle_model()
    csv = joinpath(dirname(pathof(BayesianRegressionModels)), "..",
                   "research", "adaptive_centering", "mcycle.csv")
    rows = split.(readlines(csv)[2:end], ',')
    times = parse.(Float64, getindex.(rows, 2))
    accel = parse.(Float64, getindex.(rows, 3))
    xmin, xmax = extrema(times)
    x = @. -1 + 2 * (times - xmin) / (xmax - xmin)
    y = accel ./ std(accel)
    (@brm begin
        length_scale(mu, hsgp(x)) ~ LogNormal(0, 4)
        sd(mu, hsgp(x)) ~ LogNormal(0, 4)
        length_scale(sigma, hsgp(x)) ~ LogNormal(0, 4)
        sd(sigma, hsgp(x)) ~ LogNormal(0, 4)
        mu ~ hsgp(x; k=20, domain=(-1.5, 1.5))
        log(sigma) ~ hsgp(x; k=20, domain=(-1.5, 1.5))
        y ~ Normal(mu, sigma)
    end)((; x, y))
end
""", :adaptive_motorcycle_model;
    title="The full motorcycle model", require_stan=true)
```

## 1. Fit the noncentered model

The source configuration is retained: one chain, `Xoshiro(1)`, 10,000 requested
draws, and ordinary WarmupHMC defaults including Pathfinder initialization.
The draw count is a floor, so the actual retained count is reported below.
No evaluation window, target acceptance rate, tree depth or transformation
setting has been changed to make the comparison look better.

```julia
pilot = WarmupHMC.adaptive_warmup_mcmc(
    Xoshiro(1), noncentered_problem; n_draws=10_000, monitor_ess=true)
```

`monitor_ess=true` preserves the diagnostic monitoring enabled by the source's
progress display; it does not retune the sampler.

![Noncentered pilot mean and conditional-noise posteriors](assets/adaptive-hsgp/noncentered_posterior.png)

The two panels show the conditional mean and conditional standard deviation,
not a posterior-predictive interval. Shading gives 90%, 80% and 50% central
credible intervals. Both use the source's standardized acceleration units;
the noise axis is logarithmic.

## 2. Inspect the geometry in different coordinates

For spectral standard deviation `s` and noncentered weight `z`, the physical
basis weight is `w = s*z`. In partial coordinates,

```text
u = s^c * z
u ~ Normal(0, s^c)
w = s^(1-c) * u
```

`c=0` is noncentered and `c=1` is centered. The physical prior and likelihood
stay the same; the sampler's coordinates and their matching Jacobian change.

![Noncentered pilot weights versus GP hyperparameters](assets/adaptive-hsgp/noncentered_scatter.png)

Rows show frequencies 1, 2, 19 and 20. Columns show the mean GP's marginal SD
and length scale, then the log-SD GP's marginal SD and length scale. The
hyperparameter axes are logarithmic. These panels contain the full pilot draw
set, not a small illustrative selection.

![Centered weights obtained by transforming the same pilot draws](assets/adaptive-hsgp/centered_scatter.png)

This is the **same pilot**, transformed to `w=s*z`. It reveals the centered
geometry without running another chain. A change that helps low frequencies
can make high frequencies much worse, which motivates choosing their
centeredness separately.

## 3. Select one centering per frequency

For each weight, the source searches `0:0.01:1` using

```text
loss(c) = log(std(z .* exp.(c .* log(s)))) - mean(c .* log(s))
```

BRM's `select_hsgp_centeredness` exposes this pilot-based selection. It uses
shifted exponents to evaluate the loss stably and reports candidates whose
partial-coordinate scales would underflow as inadmissible. The noncentered
endpoint remains available.

![Per-frequency loss profiles for mean and log-SD GPs](assets/adaptive-hsgp/loss_profiles.png)

Each curve is rescaled to `[0,1]` for display, as in the source. Only its
minimum matters; losses from different frequencies are not compared by their
plotted heights. Gaps represent explicitly inadmissible candidates.

![Selected centeredness across all 20 frequencies of both GPs](assets/adaptive-hsgp/selected_centeredness.png)

The selected vectors become ordinary model data through
`hsgp(...; centeredness=c_mu)` and `hsgp(...; centeredness=c_sigma)`.

![Pilot draws transformed into the selected partial coordinates](assets/adaptive-hsgp/optimal_scatter.png)

These panels apply the selected coordinates to the same pilot draws, so their
geometry can be compared directly with the noncentered and centered panels.
They are not the fresh refit's samples shown next.

## 4. Fit the selected partial model from scratch

The second fit starts fresh, again with `Xoshiro(1)`, 10,000 requested draws,
and ordinary WarmupHMC defaults. The pilot determines the coordinates, not
the posterior sample retained from the second fit.

```julia
refit = WarmupHMC.adaptive_warmup_mcmc(
    Xoshiro(1), selected_partial_problem; n_draws=10_000, monitor_ess=true)
```

![Fresh partially centered mean and conditional-noise posteriors](assets/adaptive-hsgp/partial_posterior.png)

![New partial-coordinate samples versus GP hyperparameters](assets/adaptive-hsgp/partial_scatter.png)

Both fits retained 10,000 draws with the configuration above:

| fit | max split R-hat | min bulk ESS | min tail ESS | divergences |
| --- | ---: | ---: | ---: | ---: |
| Noncentered pilot | 1.0012 | 2,514 | 2,886 | 34 (0.34%) |
| Selected-partial refit | 1.0007 | 2,533 | 1,797 | 16 (0.16%) |

The posterior curves agree visually and the low-frequency coordinate clouds
become less dependent on the hyperparameters. Divergences decrease, but do
not disappear. Bulk ESS is similar and the minimum tail ESS is lower in this
refit, so these results do **not** establish an across-the-board efficiency
improvement. They reproduce the source's geometry experiment, including its
limitations, rather than a claim that partial centering guarantees a clean fit.

An independent check applied the source's literal loss formula to every pilot
weight: all 40 selected values agree exactly with BRM's automated selection.
The full-fit receipts, candidate scores and figure checks pass 8,221 tests.

Rank-normalized split R-hat, bulk ESS and tail ESS are computed with
MCMCDiagnosticTools. As in the source's main example, each fit has one chain:
split R-hat is a within-chain diagnostic, not evidence that independent chains
agree. Divergences must remain visible in any interpretation of the geometry
comparison; changing the sampler settings to conceal them would answer a
different question.

ESS alone does not measure computational cost. The [cost comparison below](#compute-cost-and-ess-per-gradient)
reports the recoverable gradient counts for all three fits, including warmup.

## Online adaptive centering and the Turing gate

Online centering learns per-weight coordinates inside warmup instead of using
a separate pilot. BRM discovers the HSGP cells and constructs their transform:

```julia
online = adaptive_centering_problem(sb, stan_problem, enzyme_backend)
fit = WarmupHMC.adaptive_warmup_mcmc(
    Xoshiro(1), online; n_draws=10_000, monitor_ess=true)
learned = WarmupHMC.reparam_sources(online)
```

Returned draws are already back in the original model coordinates: do not
apply a second sampler-to-model back-transform. An explicit transformation
for a diagnostic plot is a separate operation, as shown below. The reproduction provides a separate
`run_online_stanblocks` entry point with the same full model and ordinary
sampler defaults; it is an extension, not one of the source's two fits.

![Online adaptive-centering posterior mean and conditional noise](assets/adaptive-hsgp/online_posterior.png)

The full online StanBlocks run retained 10,000 draws, with **zero divergences**,
maximum split R-hat `1.0018`, minimum bulk ESS `2,318`, and minimum tail ESS
`2,775`. This is encouraging evidence from one run, not a guarantee for other
data, seeds or models.

![Online warmup centering compared with the separate offline selection](assets/adaptive-hsgp/online_centeredness.png)

The online values come from warmup's `0:0.1:1` candidate grid; the offline
profile uses the source's finer `0:0.01:1` search over its separate pilot.
They are not expected to select identical values from different finite
trajectories and different objectives. Both preserve the same physical model
through their matching coordinate transform and Jacobian.

### Online pair plots

![Online posterior basis weights versus their hyperparameters, in the learned coordinates](assets/adaptive-hsgp/online_scatter.png)

These are the same pair diagnostics as for the pilot and partial refit, now
using all 10,000 online draws. BRM transforms each returned NCP weight into
its learned coordinate `u = s^c*z` exactly once for display. Length-scale and
marginal-SD axes are logarithmic; each panel has its own coordinate scale.
This transformation neither samples a new posterior nor changes the physical
weights represented by the saved draws.

### Which loss is online centering minimizing?

The two selectors use different criteria. The post-hoc selector above uses
the source's KL-derived log-scale proxy,

```text
L_offline(c) = log(std(s^c*z)) - mean(c*log(s)).
```

WarmupHMC's online selector uses **weighted position–gradient correlation**
at its default `w₁=0`:

```text
u_c = s^c*z
g_c = ∂ log p_c / ∂u_c
L_online(c) = Cor_weighted(u_c, g_c).
```

It minimizes that signed correlation on `0:0.1:1`; it is not minimizing an
absolute correlation or reusing the offline loss. For an independent Gaussian coordinate,
the log-density gradient is a decreasing affine function of position, giving
correlation `-1`. The optional Jacobian/log-variance part of WarmupHMC's
criterion has zero weight under these defaults.

![Native online correlation objective on a common pilot reference, faceted only by GP](assets/adaptive-hsgp/online_loss.png)

This plot evaluates WarmupHMC's native `candidate_scoring_losses` on the
**same 10,000 pilot draws** used by the offline diagnostic, with unit weights.
It facets only by GP: the coordinate frame in which the reference draws were
stored does not change the candidate loss landscape. A matched-draw audit
evaluated all 1,320 candidate comparisons across three representations of
those physical draws; their largest loss difference was `1.53e-15`.

Unlike the offline proxy plot, these are **raw, interpretable correlations**,
with fixed y-limits `[-1,0]` and no min–max scaling. Values near `-1` indicate
a nearly linear decreasing position–score relationship; values near zero
indicate a weak linear relationship. The exported scores remain unchanged.

This is a **retrospective objective diagnostic**, not a reconstruction of
the online run's warmup history. Actual online selection accumulates warmup
trajectory evidence and resets between adaptation windows. Saved posterior
draws do not retain that leaf stream, its weights, or those group boundaries,
so a posterior replay need not choose the exact centeredness learned online.

### Pair plots in gradient-loss-selected coordinates

![Pilot hyperparameter pair plots transformed using the gradient-loss minimizers](assets/adaptive-hsgp/gradient_selected_scatter.png)

For this third geometry view, each basis exponent is the minimizer of its
native gradient-loss curve evaluated on the common pilot, over `0:0.1:1`.
The familiar hyperparameter pair plots then transform all 10,000 pilot draws
to those coordinates. **There is no additional fit.** These values need not
equal warmup's final online selections: the criterion is the same, but the
posterior pilot and the online warmup stream supply different finite evidence.

### Position versus gradient, without hyperparameter axes

![Mean GP coordinate positions versus exact log-density gradients in three configurations](assets/adaptive-hsgp/gradient_mu.png)

![Log-SD GP coordinate positions versus exact log-density gradients in three configurations](assets/adaptive-hsgp/gradient_log_sigma.png)

Here the columns genuinely change the displayed coordinates: NCP pilot,
post-hoc partial refit, and online fit in its learned geometry. Each facet
shows **1,000 evenly selected saved draws**, with transparent points; the
underlying gradient evaluations and loss calculations still use all 10,000
draws per fit. There is no KDE, binning, smoothing, or fitted regression line.
Gradient axes are independent between facets, because reparameterization
changes their units as well as the coordinate units.

The gradients come from the actual BRM-generated Stan target. At fixed
hyperparameters, BRM transports them with `g_c = s^(-c)*g_z` (or the equivalent
source-to-target exponent difference). The change-of-coordinate Jacobian is
constant with respect to this basis coordinate; its hyperparameter derivatives
are not being plotted here. Seventy-two independent finite-difference checks
of the displayed gradients passed, with maximum scaled error `1.56e-8`.
These scatter plots visualize a component of the correlation criterion;
their appearance alone is not an ESS or convergence guarantee.

**Turing sampling is currently disabled.** The explicit length-scale
support mismatch found by the source audit has been corrected. The actual
DynamicPPL/Enzyme gradient path remains under correctness and runtime
verification on matched full-model coordinates. Turing must pass both numerical equality
and warmed gradient-runtime parity before it is used to sample this case
study. Finite gradients or a 20-draw execution check do not satisfy that gate.

## Compute cost and ESS per gradient

The final original checkpoints retain WarmupHMC's cumulative NUTS evaluation
counter. It includes step-size adaptation and all discarded restart epochs;
it is taken once from the final checkpoint, not summed over windows. It counts
DynamicHMC integration steps, **not** Pathfinder initialization or other setup
gradient calls. The numerator below is the minimum bulk ESS over the same 44
sampled model coordinates used in the fit-results table.

| Fit | Total NUTS gradient evaluations | Min bulk ESS / total gradients | Sampling-only gradient evaluations | Min bulk ESS / sampling gradients | Fit runtime |
| --- | ---: | ---: | --- | --- | --- |
| Noncentered pilot | 1,770,275 | 0.001420 | Not recorded | Not recoverable | Not recorded |
| Selected-partial refit | 904,842 | 0.002800 | Not recorded | Not recoverable | Not recorded |
| Online adaptive centering | 2,587,623 | 0.000896 | Not recorded | Not recoverable | Not recorded |

The refit alone obtains about twice the minimum bulk ESS per NUTS gradient of
the pilot. But obtaining its fixed centering required that pilot: charging
both runs costs **2,675,117** evaluations and gives **0.000947** refit bulk ESS
per total gradient. On this end-to-end accounting the pilot-then-refit workflow
does not beat the NCP run. The online run removes the observed divergences,
but uses more gradients and has lower minimum bulk ESS per total gradient;
it is not a computational-speedup result. The corresponding minimum tail-ESS
ratios are `0.001630`, `0.001985`, and `0.001072` for the individual runs.

The missing entries are measurement gaps, not zeros. These historical run
records did not save wall time. Their final retained epoch includes 50
step-size-adaptation transitions whose evaluation costs were not separately
stored, so subtracting the last restart's total does **not** give exact
sampling-only cost. Neither file timestamps nor an average steps-per-draw
estimate is substituted for those measurements. `report_costs.jl` reproduces
the exact available counts and ratios. New fresh-fit records preserve wall
time around the sampler call, the total counter, and, with WarmupHMC containing
`e376f8f`, the exact retained-sampling counter. The reporter computes both
ESS-per-gradient ratios when the fit record and final checkpoint agree;
it does not add measurements to these historical runs.

Finally, a gradient evaluation is not equally expensive in every coordinate
system: the online wrapper includes coordinate transport. Without measured
runtime, these counts cannot establish wall-clock efficiency. The 34/16/0
divergence counts and single-chain limitations still apply to the ESS figures.

## Native BRM diagnostics, rendered with AlgebraOfVega

The fits use BRM models and WarmupHMC sampling. BRM owns logical-output
extraction, posterior-predictive execution, HSGP coordinate transport, and
gradient transport; WarmupHMC owns the online candidate scores. The research
scripts assemble the comparison panels and retain the scientific provenance.
**All figures on this page are rendered in Julia with AlgebraOfVega; no R
renderer is used.**

Plotting is optional: `using BayesianRegressionModels, AlgebraOfVega` loads
BRM's plotting extension without adding plotting dependencies to fitting-only
workflows. Given a descriptor and matching saved draws, the reusable calls are:

```julia
using BayesianRegressionModels, AlgebraOfVega

# Constrained matrices have draws in rows and matching names in columns.
brm_posteriorplot(descriptor, constrained, names; logical=:mu, x=times)
brm_pairplot(descriptor, constrained, names;
    predictor=:mu, term=:hsgp_x, centeredness=learned_mu, bases=[1, 2, 19, 20])

# This simulates replicated observations through BRM's native :predict operation.
# It is different from plotting the latent conditional-mean ribbons above.
brm_ppcplot(descriptor, unconstrained; problem=stan_problem,
    response=:y, seed=1, x=times)

brm_centerednessplot(centering_rows; compare=true)
brm_centering_lossplot(online_loss_rows; ylimits=(-1, 0))
brm_centering_lossplot(offline_loss_rows; normalization=:minmax)
brm_gradientplot(gradient_rows; opacity=0.25, markersize=8)
```

`hsgp_coordinate_draws` prepares fitted or transformed coordinates and optional
basis gradients from descriptor-owned metadata. The HSGP diagnostics currently
support ungrouped squared-exponential terms and reject unsupported geometry.
WarmupHMC matrices use coordinates in rows: transpose them when calling these
BRM draw-table helpers. Conversely, `candidate_scoring_losses` expects
coordinates-by-draws matrices in its current source frame. The reproduction
scripts handle these boundaries explicitly; none substitutes an unrelated
hand-written sampler or gradient target.

## Reproduce and inspect the evidence

The script, plotting program, data and source audit are in
`research/adaptive_centering/`. The README documents the full commands.
`provenance.toml` and `packages.tsv` record the source/data hashes, exact code
and dependencies, seed, basis count and sampler configuration. Raw model-frame
draws are saved before plotting, and existing completed fits are never
silently overwritten.

Primary source boundaries:

- [Companion code at `0d00b853`](https://github.com/generable/public-materials/tree/0d00b8535e2c20c49017d03c7b060940eb8e7041/blog/hsgp-reparam).
- [Rdatasets at `1dcc2bf`](https://github.com/vincentarelbundock/Rdatasets/tree/1dcc2bf5f955cc1224a3e1307256e1fe86b68dae/csv/MASS).
- CSV SHA-256: `b89a1e4eb0391a982b32be3e378df00e8593ff9971e9425e9c5d7929b74f9801`.

The source plotting helper swaps the length-scale and marginal-SD labels;
these figures label the actual model quantities correctly. Different library
versions and parameter orderings can produce different trajectories at the
same seed. The source's numerical results are context, not numbers to copy
into a new run.
