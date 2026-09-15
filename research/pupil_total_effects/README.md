# Pupil: manual total-effect parametrization

This is a manual WarmupHMC experiment for [pupil post 3](https://discourse.mc-stan.org/t/help-testing-brms-pr-for-sum-to-zero-and-partial-centering/41542/3).
It uses all 2,228 observations from 20 subjects, in their original order.
The original data and a lossless CSV export are pinned in `reference/`.

The first Gaussian sensitivity used two changes to the forum model:

- Independent random intercept and load effects, replacing the learned correlation.
- `Normal(5651.9, 2026.1)` for the mean population intercept, replacing its
  Student-t prior. These numbers match the original location and scale, not
  its variance.

The current comparison restores the Student-t intercept prior and retains
independent random effects. The earlier Gaussian sensitivity was:

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
mean coefficients. The integrated target has these two additional arms:

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
The original runs saved marginal draws; the recovery follow-up below now
reconstructs population/deviation parameters from those saved draws.

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
46-parameter NCP minimum is reported separately. Conditional recovery below
adds a separate comparison for the original population/deviation parameters.
With one chain, R-hat is only a within-chain split diagnostic.

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
three coordinate systems: CP, NCP, and offline adaptive partial centering (ACP),
all applied to total coefficients.
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

## Native Stan sensitivity

`native_stan.R` samples the exact exported ordinary brms target under
CmdStan 2.39.0, CmdStanR 0.9.0: one chain, seed 1, 1,000 warmup iterations,
2,000 retained draws, diagonal-metric NUTS, target acceptance 0.8, maximum
depth 10. It receives the same physical OLS initialization as WHMC; WHMC
additionally applies its usual Pathfinder and adaptive warmup policy.

| Intercept prior | Sampler | Common 44 min ESS / 1,000 sampling gradients | Sampling gradients | All gradients | Divergences |
| --- | --- | ---: | ---: | ---: | ---: |
| Gaussian | WHMC | 0.918 | 154,742 | 166,865 | 50 |
| Gaussian | Native Stan | 0.465 | 852,992 | 1,223,801 | 0 |
| Student-t | WHMC | 0.356 | 234,960 | 274,051 | 0 |
| Student-t | Native Stan | 0.472 | 682,496 | 1,063,829 | 0 |

The native Gaussian and Student-t fits hit depth 10 on 42 and 14 sampling
iterations respectively. Their common-coordinate minima are both the random
intercept SD (reported in log units); maximum within-chain split R-hat is
1.0065 and 1.0057. These are individual smoke fits, not repeated benchmarks.

The injected C++ counter increments on reverse-mode target evaluations and
returns exactly zero to the density. Every retained counter increment equals
that transition's leapfrog count plus one. Final process receipts verify all
calls, including initialization and warmup. Source instrumentation is reversible;
the original Stan source is preserved. `analyze_native.jl` uses named native
columns to construct the same common totals and original-parameter scopes as
the WHMC analysis. Full native CSVs, fit records, cost receipts and exact
executed source are archived under `results/*/native_ncp/`.

## Matched Student-t parametrization matrix

All rows below target the same independent-RE Student-t pupil posterior.
Primary ESS is the minimum across the same 46 scientific quantities: population
intercept at mean load, population slope, two group SDs, two residual-model
coefficients, and 40 subject-specific total coefficients. Population coefficients
are conditionally recovered for marginalized arms; deviations, standardized
coordinates and the mixture auxiliary are excluded from the minimum. Each fit
has one chain and 2,000 retained draws; WHMC nonlinear online adaptation is off.
Total gradients includes initialization, warmup and any required precursor or
offline pilot. The final two columns are relative to ordinary brms NCP + native
Stan on these same 46 quantities (baseline minimum ESS 170.298). Full numbers
and per-QOI ESS/MCSE are in `results/student_mixture/qoi46/`. Historical
diagnostic scopes are retained in `results/student_mixture/matrix/`.

| Model | Sampler | Total gradients | Relative sampling efficiency | Relative total efficiency |
|---|---|---:|---:|---:|
| brms NCP | Native Stan | 1063829 | 1× | 1× |
| brms NCP | WHMC | 274051 | 1.43× | 1.91× |
| brms CP | WHMC | 80373 | 2.54× | 3.68× |
| brms S2Z CP | Native Stan | 319693 | 135× | 40.0× |
| brms S2Z CP | WHMC | 188667 | 137× | 66.0× |
| brms S2Z NCP | Native Stan | 771035 | 1.17× | 1.22× |
| brms S2Z NCP | WHMC | 123267 | 3.19× | 4.48× |
| brms S2Z auto | Native Stan | 150012 | 187× | 62.6× |
| brms S2Z auto | WHMC | 64875 | 173× | 194× |
| total coefficients CP | WHMC | 54587 | 131× | 195× |
| total coefficients NCP | WHMC | 134652 | 3.03× | 4.46× |
| total coefficients ACP | WHMC | 168932 | 258× | 71.2× |

The partial refit has the highest sampling efficiency in this pilot. Once its
134,652-gradient pilot is charged, fixed total CP and brms S2Z auto under WHMC
are nearly tied for the best total efficiency. The poor S2Z NCP results show
that integrating out population means alone does not guarantee good geometry
in the chosen coordinates. Ordinary brms CP improves on NCP but its population
intercept still limits efficiency. All Student-t rows have zero sampling divergences,
but the total NCP control has within-chain split R-hat up to 1.030. This is
not a replicated ranking or a convergence certification.

### S2Z target and audit

`s2z_native.R` uses the pinned brms branch at
`73cf607889879cb2a55f50b88d8141d76ff43279` with
`gr(subj,cor=FALSE,s2z=TRUE,center=TRUE/FALSE/"auto")`. Native Stan fits each
resolved target. `s2z_whmc.jl` runs its identical clean source and resolved data
through BridgeStan and WarmupHMC. The brms auto precursor runs once; its 789
gradient calls are charged to each method using its fixed weights.

For each of the two coefficient blocks, a J-by-(J-1) orthonormal Helmert matrix
Q encodes zero-sum physical deviations. CP uses `r=Q*z`; NCP uses `r=tau*Q*z`.
For auto, with fixed group-labelled weights rho, set
`scale=1-rho+rho*tau`, `w=tau*(Q*z)/scale`, and `r=w-mean(w)`.
Two finite-population coefficients and the 38 contrasts replace the original
42 population/group coefficients after integrating two group means. A Student-t
mixing variable adds one coordinate, giving 45 total sampled dimensions.

The branch uses `u ~ inv_chi_square(3)` and intercept prior scale
`2026.1*sqrt(3*u)`; our precision is `lambda=1/(3*u)`. Mapping finite-population
coefficients theta and physical S2Z deviations to totals gives
`A=theta[1]-xbar*theta[2]+r_a`, `B=theta[2]+r_b`.
The source-to-total log determinant is `log(J)` for CP and
`log(J)+(J-1)*(log(tau_a)+log(tau_b))` for NCP. For auto each contrast block
adds `(J-1)*log(tau)-sum(log(scale))+log(mean(scale))` to `log(J)`.

Eight nontrivial points per target pass absolute density, finite-difference
gradient, inverse-transform and generated-quantity reconstruction checks.
Maximum absolute density discrepancy is below 4.8e-11; maximum normalized
coordinate-gradient discrepancy is below 2e-7. The same physical totals are
reconstructed before and after native brms conditional recovery.

The local BridgeStan wrapper rejects recognized numerical-domain exceptions
from invalid sampler/Pathfinder proposals, like Stan's native sampler. It
preserves the first complete exception and a rejection count; unrelated errors
propagate. This was needed when S2Z CP Pathfinder scored a nonfinite proposal.
No production package was changed. Failed pre-sampling attempts are recorded.

### Initialization and cost limits

Native/WHMC pairs use the same supplied physical initialization. Ordinary and
integrated fits use per-subject OLS coefficients; S2Z fits use pooled finite-
population coefficients, zero contrasts, and the same OLS scales. WHMC then
runs Pathfinder, whereas native Stan applies its usual NUTS warmup. Thus the
posterior and per-row inputs are matched, but initialization across different
parametrizations is not fully controlled. Total costs include that sensitivity.

Our manual Gaussian likelihood uses exact sufficient statistics, while the
generated brms likelihood evaluates the individual observations. Consequently
gradient count is an algorithmic cost proxy, not equal wall time per gradient.
No controlled wall-time ranking is claimed. The original brms CP branch also
uses its own mean-centering convention: its generated code centers the load
coefficient around the population slope but does not absorb the population
intercept. Its semantics are audited using actual generated quantities.

Full fit records, native CSVs, resolved auto weights, source capsules, exact
cost receipts and wrapper logs are archived by arm. `compare_matrix.jl`
recomputes common, recovered-physical and original-NCP scope comparisons from
saved draws. No extra HMC is required for that analysis.

`compare_qois.jl` produces the current 46-QOI table and per-quantity MCSE from
those same saved fits. `prepare_s2z_pairs.jl` audits all 2,000 saved auto-WHMC
draws against the actual source coordinates, then selects distinct minimum,
nearest-0.5 and maximum rho subject/term coordinates. Its CP/NCP/auto columns
show `r`, `r/tau` and `Q*z`, respectively. Each displayed vector has 20
subject-labelled entries but only 19 independent directions per term.
`plot_comparison.jl` renders this grid and the three-metric efficiency plot
through AoV, with native specifications and static PNG exports. The existing
total-coordinate CP/NCP/ACP grid is reused without refitting.

`build_brief.py` builds the single-scope KB brief with three AoV envelopes.
`share_sources.py` freezes the harness, exact generated Stan, resolved inputs
and diagnostics in a byte-hashed source snapshot and adds directly inspectable
KB file links. Neither script publishes a gist or runs posterior sampling.
