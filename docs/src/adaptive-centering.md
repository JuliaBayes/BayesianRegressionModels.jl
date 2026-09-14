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

<!-- FULL_FIT_RESULTS -->
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
<!-- END_FULL_FIT_RESULTS -->

Rank-normalized split R-hat, bulk ESS and tail ESS are computed with
MCMCDiagnosticTools. As in the source's main example, each fit has one chain:
split R-hat is a within-chain diagnostic, not evidence that independent chains
agree. Divergences must remain visible in any interpretation of the geometry
comparison; changing the sampler settings to conceal them would answer a
different question.

## Online adaptive centering and the Turing gate

Online centering learns per-weight coordinates inside warmup instead of using
a separate pilot. BRM discovers the HSGP cells and constructs their transform:

```julia
online = adaptive_centering_problem(sb, stan_problem, enzyme_backend)
fit = WarmupHMC.adaptive_warmup_mcmc(
    Xoshiro(1), online; n_draws=10_000, monitor_ess=true)
learned = WarmupHMC.reparam_sources(online)
```

Returned draws are already back in the original model coordinates. They must
not be transformed a second time. The reproduction provides a separate
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
trajectories. Both preserve the same physical model through their matching
coordinate transform and Jacobian.

**Turing sampling is currently disabled.** A source audit found that its
explicit length-scale prior incorrectly retained the default HSGP lower
bound. That defect and the slow gradient path are being fixed and benchmarked
on matched full-model coordinates. Turing must pass both numerical equality
and warmed gradient-runtime parity before it is used to sample this case
study. Finite gradients or a 20-draw execution check do not satisfy that gate.

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
