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
3. **Online adaptation:** begin at `c=0` and use WarmupHMC's default weighted
   position-gradient-correlation criterion on `0:0.1:1`.

For intercepts the source coordinate is
`c*m + (A-m)*exp((c-1)*log_tau_a)`, with `m=5651.9`; for slopes `m=0`.
`c=1` gives the physical total coefficient. `c=0` scales the total coefficient;
it does **not** whiten the induced joint prior and is not an independent-Normal
NCP. All methods target the same 44-dimensional posterior.

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
```

An existing output directory is refused. A `STOP` file in the output directory
stops at a documented sampler boundary and reports the run as incomplete.

## Reporting

Each arm writes and prints its own diagnostics immediately: min bulk/tail ESS,
limiting coordinate, split R-hat, divergences, retained draws and exact gradient
counts. Primary ESS/gradient covers **all 44 model-frame coordinates**; a
secondary minimum covers the 40 total coefficients. With one chain, R-hat is
only a within-chain split diagnostic.

Sampling gradients count the retained sampling epoch; transition gradients
include discarded epochs and transition warmup. A separate instrumented target
counts every requested density-and-gradient evaluation, including initialization.
The offline workflow additionally reports pilot plus refit cost. Fit wall time
includes compilation and checkpoint I/O and is not a controlled timing benchmark.

### First Gaussian integrated-target run

Saved receipts are in `results/gaussian/`. All three fits retained 2,000 draws.

| Integrated arm | Min bulk ESS / 1,000 sampling gradients | Divergences |
| --- | ---: | ---: |
| Fixed scaled-total control | 1.210 | 0 |
| Partial refit | 54.556 | 0 |
| Online | 1.120 | 1,024 (51.2%) |

The online fit failed: max split R-hat was 2.16 and tail ESS was nonfinite.
Its number is not evidence of usable posterior sampling. Its final source
gradients, checkpoint gradients and 440 retrospective native candidate-loss
values passed independent formula checks. The failure is not explained by
those coordinate/gradient checks. The partial/control efficiency ratio measures
centering within the integrated target, not a gain against conventional brms NCP.

Density audits, fitted sources, package provenance, learned controls, fit files
and checkpoints are preserved in the run directory. `export_reference.R`
exports original/modified brms source and standata without sampling in brms.
