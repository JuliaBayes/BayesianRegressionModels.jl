# Pupil: manual total-effect parametrization

This is a manual WarmupHMC experiment for [pupil post 3](https://discourse.mc-stan.org/t/help-testing-brms-pr-for-sum-to-zero-and-partial-centering/41542/3).
It uses all 2,228 observations from 20 subjects, in their original order.
The original data and a lossless CSV export are pinned in `reference/`.

The user requested two changes to the forum model:

- Independent random intercept and load effects, replacing the learned correlation.
- `Normal(5651.9, 2026.1)` for the mean population intercept, replacing its
  Student-t prior. These numbers match the original location and scale, not
  its variance.

The corresponding explicit brms model is:

```r
brm(bf(p_size ~ load + (load || subj), sigma ~ subj),
    data = df_pupil_complete,
    prior = prior(normal(5651.9, 2026.1), class = Intercept))
```

`subj` and `load` are numeric in the pinned data. Consequently `sigma ~ subj`
means a linear trend in **log residual scale versus numeric subject ID**.
This experiment preserves that literal post-3 specification. It does not
substitute categorical subject scales or post 4's random scale effects.

All other priors retain the generated brms semantics: flat population-load
and log-scale slope priors, half-Student-t(3,0,2026.1) group SD priors, and
Student-t(3,0,2.5) for the log-scale intercept. The flat population-load prior
is integrated, not replaced by independent flat group priors. The posterior
is preserved relative to the explicitly modified model above, not the original
correlated, Student-t-intercept forum fit.

## The integrated target

Let `xbar` be the observation-weighted mean load. The conventional mean is

```
beta0 + beta1 * (load - xbar) + a[subj] + b[subj] * load.
```

Sample total coefficients
`A[j] = beta0 - xbar*beta1 + a[j]` and `B[j] = beta1 + b[j]`.
The likelihood uses `A[j] + B[j]*load`. Both population coefficients are
integrated out. There are 44 sampled coordinates: 20 A, 20 B, two log group
SDs, and two residual-scale regression coefficients. There is no extra
population coefficient, selected last group, hard sum constraint, numerical
quadrature, or latent mixing variable.

Write `ma=mean(A)`, `mb=mean(B)`, `Qa=sum((A-ma)^2)`, `Qb=sum((B-mb)^2)` and

```
h = ma + xbar*mb - 5651.9
v = 2026.1^2 + (tau_a^2 + xbar^2*tau_b^2)/J.
```

The induced log prior, up to parameter-independent constants, is

```
-(J-1)*(log(tau_a)+log(tau_b))
- Qa/(2*tau_a^2) - Qb/(2*tau_b^2)
- log(v)/2 - h^2/(2*v).
```

The remaining common-slope prior is flat, as required by the original flat
population-load prior. The observed data identify the slope. Including this
mean factor and the `(J-1)` scale exponents is essential for equivalence.

`model.jl` implements this O(J) density and analytic gradient. Gaussian
likelihood sufficient statistics give an exact O(J) likelihood too. The
audit separately evaluates every original observation and integrates the
two population coefficients through an independent Gaussian conditioning
identity; it compares all coordinate gradients by finite differences.

## Comparison arms

The comparison baseline is the conventional brms NCP, retaining both population
mean coefficients. The integrated target has these three additional arms:

1. **Scaled total coefficients:** ordinary WarmupHMC with fixed centering
   controls `c=0`. This additional control still learns its linear transformation.
2. **Post-hoc partial refit:** choose each scalar control from that control fit's
   draws using the variance/Jacobian criterion on `0:0.01:1`, then start a
   fresh fit with those controls fixed.

Online adaptation was tested in the first Gaussian pilot and failed. It is
disabled for subsequent runs at the user's request. Its historical fit and
audit receipts remain available.

For intercepts the source coordinate is
`c*m + (A-m)*exp((c-1)*log_tau_a)`, with `m=5651.9`; for slopes `m=0`.
`c=1` gives the physical total coefficient. `c=0` scales the total coefficient;
it does **not** whiten the induced joint prior and is not an independent-Normal
NCP. The conventional baseline samples 46 parameters; integrating its two
population coefficients gives the same 44-dimensional marginal posterior
sampled by the Gaussian integrated fits.

Each fit uses seed 1, one chain, and a 2,000 retained-draw floor. All start
Pathfinder from the same physical within-subject regression estimates. Other
WarmupHMC settings are defaults. Returned positions are already in the model
frame; saved checkpoints are verified using their own source-to-model map.
Original population/deviation parameters are not reconstructed.

Run from the repository root using the existing test environment:

```sh
julia --project=test --startup-file=no research/pupil_total_effects/audit.jl
PUPIL_TOTAL_OUTPUT=/absolute/new/output/directory \
  kb-run-compact julia --project=test --startup-file=no research/pupil_total_effects/run.jl
PUPIL_BASELINE_OUTPUT=/absolute/new/baseline/directory \
  kb-run-compact julia --project=test --startup-file=no research/pupil_total_effects/brms_baseline.jl
```

An existing output directory is refused. A `STOP` file in the output directory
stops at a documented sampler boundary and reports the run as incomplete.

## Reporting

Each arm writes and prints its own diagnostics immediately: min bulk/tail ESS,
limiting coordinate, split R-hat, divergences, retained draws and exact gradient
counts. Primary ESS/gradient covers **all 44 model-frame coordinates**; a
secondary minimum covers the 40 total coefficients. Both use the same physical
quantities across fits, not each sampler's internal coordinates. The original
46-parameter NCP minimum is reported separately. Without conditional recovery
in the integrated fits, there is no comparison for their full original
population/deviation parameter set. With one chain, R-hat is only a within-chain
split diagnostic.

Sampling gradients count the retained sampling epoch; transition gradients
include discarded epochs and transition warmup. A separate instrumented target
counts every requested density-and-gradient evaluation, including initialization.
The offline workflow additionally reports pilot plus refit cost. Fit wall time
includes compilation and checkpoint I/O and is not a controlled timing benchmark.

### Gaussian comparison

Saved receipts are in `results/gaussian/`. All three fits retained 2,000 draws.

| Arm | Min bulk ESS / 1,000 sampling gradients, common 44 quantities | Divergences |
| --- | ---: | ---: |
| Conventional brms NCP | 0.918 | 50 |
| Fixed scaled-total control | 1.210 | 0 |
| Partial refit | 54.556 | 0 |
| Historical online failure | 1.120 (unusable) | 1,024 (51.2%) |

The partial refit has 59.4 times the baseline's common-coordinate sampling
efficiency. Counting its integrated pilot plus all initialization and warmup
gradient calls reduces that comparison to 15.8 times. Restricting the metric
to the 40 total coefficients gives 6.821 versus 54.556, about 8.0 times.
The NCP baseline's own 46-parameter minimum is 0.428, limited by `z_1.1.4`.
Its 2.5% divergences and the one-chain design limit the comparison.

The online fit failed: max split R-hat was 2.16 and tail ESS was nonfinite.
Its number is not evidence of usable posterior sampling. Its final source
gradients, checkpoint gradients and 440 retrospective native candidate-loss
values passed independent formula checks. The failure is not explained by
those coordinate/gradient checks. A possible implementation defect has not
been excluded; no cause is asserted and online runs are now disabled.

### Student-t mixture

The second comparison restores the source population-intercept prior through

```
lambda ~ Gamma(shape=3/2, rate=3/2)
beta0 | lambda ~ Normal(5651.9, 2026.1/sqrt(lambda)).
```

The integrated target samples `eta=log(lambda)`. Replace the first term in `v`
by `2026.1^2*exp(-eta)` and add `log p(lambda) + eta` to the density. This
adds one sampled mixture coordinate: 45 integrated coordinates versus 46 in
the conventional Student-t NCP. The 40 centering controls remain independent
scalar choices; the mixture precision is not reparametrized. Conditional
Gaussian integration and its gradient remain O(J). `audit_mixture.R` uses
quadrature solely as an independent check of the Student-t prior identity;
the sampler performs no numerical integration.

Set `PUPIL_INTERCEPT_PRIOR=student_mixture` for both Julia drivers. The
conventional baseline uses brms's direct Student-t prior, without a sampled
mixture variable. The common-quantity comparison excludes the integrated
precision; the full integrated minimum includes it.

| Arm | Min bulk ESS / 1,000 sampling gradients, common 44 quantities | Divergences |
| --- | ---: | ---: |
| Conventional brms Student-t NCP | 0.356 | 0 |
| Integrated scaled-total control | 0.757 | 0 |
| Integrated partial refit | 64.457 | 0 |

The common-quantity sampling ratio is 180.9 times. Charging the integrated
pilot plus refit gives 168,932 all-gradient calls, 11.395 minimum ESS per
1,000 calls, and a 37.3-times comparison against the conventional baseline's
all-gradient efficiency. This includes initialization and warmup in each arm.

Each retained 2,000 draws. The partial minimum is `total_load[702]`; including
the mixture precision does not change it. Its max split R-hat is 1.0063,
versus 1.0091 for NCP and 1.0300 for the integrated control. These remain
one-chain smoke results. Gaussian and Student-t comparisons concern different
priors and are kept separate.

### Saved-draw checks and figures

`compare_saved.jl INTEGRATED_DIR BASELINE_DIR` compares the means of the common
44 quantities, with mean MCSE, and writes sampling/workflow efficiency ratios.
This is a descriptive convergence check, not a proof of equivalence.
In these pilots no common mean differed by more than three estimated combined
MCSEs (maximum 2.96 for Gaussian and 2.66 for Student-t).

`prepare_pairs.jl INTEGRATED_DIR` exports every retained partial-refit draw in
three coordinate systems: centered total, scaled total, and selected partial.
Rows select distinct coordinates by minimum centeredness, nearest 0.5 among
remaining coordinates, and maximum among the rest. All columns use the same
draws. `plot_saved.jl` renders native AoV facets using the existing
`research/adaptive_centering/plots` environment. The centered total is the
visualization baseline. It is not the conventional deviation CP, and the
scaled total is not the conventional brms NCP.

Density audits, fitted sources, package provenance, learned controls, fit files
and checkpoints are preserved in the run directory. `export_reference.R`
exports original/modified brms source and standata without sampling in brms.
The committed `results/*/fits/` archives also preserve every completed fit
record, including the failed Gaussian online record, with compressed and
uncompressed SHA-256 receipts. Decompress these Julia 1.10.11 `.jls` files and
load `model.jl` before deserializing Student-mixture records. The conventional
records retain both native NCP draws and their named/common transformations;
integrated records retain physical draws, controls and exact cost counters.

## Conditional recovery and ESS sensitivity

The follow-up in `recovery.jl` now recovers the two population coefficients
and original group deviations from each saved marginal draw. In the Student-t
case it conditions on the saved mixture precision. It also derives conventional
NCP `z` coefficients and the two uncentered population/residual intercepts.
This requires no new HMC transitions. Twelve draws per fit pass an independent
precision-matrix calculation, conditional density-ratio check and exact
total-coefficient reconstruction checks. Twenty recovery seeds (101–120)
measure variation due solely to the added conditional random draws.

The original-physical scope contains the two population coefficients, four
scale-model quantities and 40 group deviations. Population and residual
intercepts refer to the centered predictor designs, as in the Stan parameter
block. A separate 48-quantity scope includes both raw-design intercepts; that
does not change the minima below. The original-NCP scope replaces deviations
by their SD-scaled `z` values, matching the conventional Stan parameters.

| Prior / fit | Original physical min ESS / 1,000 sampling gradients | Original NCP min ESS / 1,000 gradients |
| --- | ---: | ---: |
| Gaussian brms + WHMC, saved original draws | 0.470 | 0.428 |
| Same Gaussian brms + WHMC draws, population coefficients refreshed | 0.918 | 0.918 |
| Gaussian integrated partial, recovered | 60.093 [49.361, 61.650] | 59.435 [49.203, 63.666] |
| Student-t brms + WHMC, saved original draws | 0.356 | 0.356 |
| Student-t integrated partial, recovered | 63.678 [54.503, 65.849] | 63.607 [54.503, 68.509] |

Recovered entries show the median and range across 20 seeds, not independent
HMC replications or confidence intervals. The Gaussian refresh preserves the
baseline's marginal draws and gradient cost exactly. Its changed minimum
demonstrates that recovery alone can alter the comparison. The integrated
partial results remain close to their unrecovered common-quantity efficiencies
(54.556 Gaussian, 64.457 Student-t); their earlier gain was not created by
recovery noise.

### Why added recovery noise changes ESS

Write a recovered quantity as `X[t] = m(S[t]) + epsilon[t]`, where `m` is its
conditional mean given the sampled marginal state and each recovery noise has
conditional mean zero and is drawn independently across iterations. Then

```
Var(X) = Var(m(S)) + E[Var(X | S)]
Cov(X[t], X[t+k]) = Cov(m(S[t]), m(S[t+k]))   for k > 0.
```

The added variance dilutes positive autocorrelation, raising mean-ESS toward
the number of draws. With negative autocorrelation it can instead lower ESS.
The corresponding variance of the sample mean is

```
Var(mean(X)) = Var(mean(m(S))) + E[Var(X | S)] / N.
```

Thus recovery can raise ESS while making the mean estimate less precise than
the conditional-mean (Rao–Blackwell) estimate. These identities concern ordinary
covariance and mean-ESS; rank-normalized bulk ESS is measured separately. See
the [Stan definition of ESS and MCSE](https://mc-stan.org/docs/reference-manual/analysis.html#effective-sample-size).
The recovered values are valid posterior draws: this is a reporting/estimand
distinction, not evidence that conditional recovery is incorrect.

For the Student-t partial fit, 99.966% of population-intercept posterior
variance is conditional recovery variance. Its conditional-mean MCSE is 0.225;
the recovered MCSE is about 12.08 (median across seeds). Its conditional-mean
and recovered bulk ESS are both about 2,000. This is why MCSE and the shared
total-coefficient diagnostics accompany recovered-parameter ESS.
