# Pupil model with subject-specific residual scales

This experiment follows [post 4 of the Stan discussion](https://discourse.mc-stan.org/t/help-testing-brms-pr-for-sum-to-zero-and-partial-centering/41542/4).
It keeps the data and priors, with the requested simplification to independent
mean random intercepts and slopes:

```r
bf(p_size ~ load + (load || subj), sigma ~ (1 | subj))
```

The data have 2,228 observations, 20 subjects (701–720), and numeric load 0–5.
Observation order is preserved. `capture.R` exports the exact generated targets
from brms commit `73cf607889879cb2a55f50b88d8141d76ff43279`. The source data are
pinned in the earlier study at bcogsci commit
`d90fc01e6f6fcdced7ee64c9d2ed607d212ec77c`.

## Posterior

For subject j and load x, with xbar the full-data average load:

```
y ~ Normal(beta0 + beta1*(x-xbar) + a[j] + b[j]*x, exp(gamma0 + v[j]))
beta0 ~ Student-t(3, 5651.9, 2026.1)
beta1 ~ Flat
gamma0 ~ Student-t(3, 0, 2.5)
a[j] ~ Normal(0, tau_a)
b[j] ~ Normal(0, tau_b)
v[j] ~ Normal(0, tau_v)
tau_a, tau_b, tau_v ~ half-Student-t(3, 0, 2026.1)
```

The very broad scale prior on `tau_v` is present in the generated brms model;
it is retained in every arm. Unlike the earlier post-3 model's numeric-subject
scale regression, this model has a distinct partially pooled residual SD for
each subject.

BRM automatically samples the total coefficients
`A[j]=beta0-xbar*beta1+a[j]`, `B[j]=beta1+b[j]`, and `C[j]=gamma0+v[j]`.
The likelihood is `Normal(A[j]+B[j]*x, exp(C[j]))`. Two exact Gaussian precision
mixtures represent the two Student-t population priors. The three shared
coefficients are integrated out during sampling and recovered jointly from
their conditional Gaussian posterior for each saved draw.

Ordinary brms samples 66 parameters. BRM totals and brms S2Z each sample 65:
the removed three population coefficients are offset by two mixture variables.
The marginalized joint priors are correlated even at the scaled/NCP endpoint.

## Matched comparison

- Ordinary brms NCP with native Stan and WHMC; ordinary brms CP with WHMC.
- brms S2Z CP, NCP and automatic centering, with native Stan and WHMC.
- Automatic BRM totals with WHMC: CP, NCP, post-hoc position loss, post-hoc
  position–gradient loss, online position loss and online position–gradient loss.

Every arm uses seed 1 and at least 2,000 retained draws in one chain. These are
exploratory comparisons, not precise estimates of method rankings. Native Stan
uses 1,000 warmup iterations, adapt_delta 0.8 and maximum tree depth 10. WHMC uses
its adaptive warmup policy, including Pathfinder initialization. All fits start
from common pooled total coefficients, zero deviations, scales estimated from
subject regressions, and mixture precisions equal to one. Native and WHMC warmup
policies differ; sampler comparisons must retain that distinction.

Post-hoc total selection reuses the saved NCP pilot. Its total gradient cost
includes that pilot. Position–gradient scoring transports the pilot's stored
gradients into the compiled model frame without evaluating the target again.
The three density-gradient calls verifying this transport are validation work,
not fit cost. brms automatic-centering pilot cost is included in both its native
and WHMC workflows when the resolved weights are reused.

## One common set of scientific quantities

The main table uses 66 quantities in every row:

- beta0 (at mean load), beta1, gamma0;
- tau_a, tau_b, tau_v;
- each subject's total intercept A, total load slope B, and residual SD exp(C).

Standardized deviations, raw sampling coordinates and mixture variables are
excluded. Marginalized methods use the same conditional-recovery implementation
and seed 404. Recovery randomness is checked separately; subject totals are
invariant to it. The table reports total gradients and minimum bulk ESS divided
by sampling gradients and by total gradients. Efficiency ratios use ordinary
brms NCP with native Stan as the denominator.

`audit.jl` checks the emitted BRM density, gradients, automatic coordinate
transforms and recovery. Its independent reference computes the Gaussian joint
prior divided by the normalized Gaussian conditional. BRM omits three constant
half-Student-t normalizers in the vector scale prior; adding `3*log(2)` matches
brms's normalized density. This constant does not affect the posterior or any
gradient. `s2z.jl` independently maps finite-population means and orthonormal
contrasts to totals, checks the full Jacobian, and verifies recovered totals.

## Harness

- `common.jl`: public BRM builder, data, independent density and shared QOIs.
- `audit.jl`: density, gradient, coordinate and recovery checks before fitting.
- `total_whmc.jl`: the six total-coefficient arms through BRM's automatic adapter.
- `native.R`: instrumented native Stan fits, including the brms auto pilot.
- `s2z.jl`: independent S2Z coordinate and density audit.
- `brms_whmc.jl`: ordinary/S2Z WHMC fits and native-fit scientific diagnostics.

The native gradient counter and recognized-numerical-proposal wrapper are
reused from `../pupil_total_effects/`. The counter contributes exactly zero to
the target; it counts reverse-mode target calls. Each retained transition's
counter increment must equal its leapfrog count plus one. Final process counts
also include initialization, warmup and any brms centering pilot.
