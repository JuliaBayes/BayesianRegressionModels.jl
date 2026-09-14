# Eight-schools adaptive centering

This is a standalone full case study for the standard eight-schools
meta-analysis model. It is independent of the motorcycle HSGP reproduction in
`research/adaptive_centering`; neither its settings nor its figures are shared.

The authoritative model and data are copied verbatim from Stan's public
`example-models` repository:

- model `misc/eight_schools/eight_schools.stan` at
  `a42b3da85b7dc38f2745dde4fca197425f18c516`
  (SHA-256 `1624c8770e8f08a90894f3417591eb45f93ccfb70cbe9870442ae2ab26343422`);
- data `misc/eight_schools/eight_schools.data.R` at
  `93b8b05cb7978952606f2043bec64d3b958b360c`
  (SHA-256 `fccd1624bd240b0c26a8b668f4d2181f0653000e4b169d393b3dc815a0b46952`).

The source model has improper flat priors on `mu` and positive `tau`; the
latter means a constant density on the declared positive support. The study
does not replace either with a convenient proper prior. Its one arm that is
reparameterized for sampling is the eight `theta[j]` coordinates. The physical
likelihood, data, and priors remain identical in every arm.

## Runs and entry points

All fits use one chain, `Xoshiro(1)`, 10,000 requested retained draws, ordinary
WarmupHMC initialization/adaptation defaults, and `monitor_ess=true`. Runtime
and setup are separated explicitly: Stan compilation occurs before the timed
sampler call, while each reported run-total gradient counter includes all MCMC
warmup/discarded epochs but excludes Pathfinder and setup. The retained
sampling counter is reported separately and never estimated.

```sh
# Compare actual density, support, and gradients with the immutable Stan source.
BRM_EIGHT_SCHOOLS_OUTPUT="$SCRATCH/eight-schools-audit" \
  julia --startup-file=no --project=test \
  research/eight_schools_centering/audit_source.jl

# Full NCP pilot, selected-partial refit, and online-adaptive fit.
BRM_EIGHT_SCHOOLS_RUNTIME=1 \
BRM_EIGHT_SCHOOLS_OUTPUT="$SCRATCH/eight-schools-fit" \
  julia --startup-file=no --project=test \
  research/eight_schools_centering/reproduce.jl

# Re-render figures from saved exports without invoking a sampler.
julia --startup-file=no --project=research/eight_schools_centering/plots \
  research/eight_schools_centering/plot_results.jl \
  "$SCRATCH/eight-schools-fit" "$SCRATCH/eight-schools-diagnostics"
```

There is deliberately no centered sampling arm, no R execution, and no Turing
sampling. The selected-partial arm samples fresh draws in selected source
coordinates and back-transforms them once through WarmupHMC's public
reparametrization API. Online adaptation uses BRM's `adaptive_centering_problem`
and WarmupHMC's actual default weighted position-gradient correlation objective
(`w1=0`), not an offline KL/log-scale proxy.

The full harness checks generated Stan with `stanc`, compares the generated
target to the immutable source at synthetic and saved posterior positions,
checks both returned and final-checkpoint counters, computes split R-hat,
bulk ESS, and tail ESS over the explicitly named ten-coordinate unconstrained
model scope, validates coordinate/Jacobian/gradient invariants, and emits
native AlgebraOfVega diagnostics.
