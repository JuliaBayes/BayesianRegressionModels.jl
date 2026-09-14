# Local AoV diagnostic preview

This is the in-progress Julia/AlgebraOfVega replacement for case-study plots.
It does **not yet replace every public figure or the old R entry point**.
Loading `BayesianRegressionModels, AlgebraOfVega` enables the optional BRM
plotting extension; fitting alone does not acquire a plotting dependency.

## Saved-fit coordinate–gradient preview

Resolve the repository test environment and this plotting environment using
the ecosystem's canonical resolver. From the repository root:

```sh
julia --startup-file=no --project=test \
  research/adaptive_centering/prepare_gradient_diagnostics.jl \
  /absolute/offline-fit-directory /absolute/online-fit-directory /absolute/diagnostics-directory

julia --startup-file=no --project=research/adaptive_centering/plots \
  research/adaptive_centering/plots/gradient_preview.jl /absolute/diagnostics-directory
```

No sampler is called. The first command checks the generated model against the
saved producer's Stan source, evaluates all 10,000 saved draws from each fit,
and checks selected transformed gradients against finite differences of the
compiled target. The output includes all 240,000 coordinate–gradient rows for
two GPs and four basis functions, plus the 72 numerical checks.

The second command generates AoV plots, PNGs, and two bounded `kb-aov/v1`
preview fences. Each preview has 50 evenly spaced saved draws per panel for
bases 1 and 20. This is display thinning, not a changed fitting budget; the
complete table retains every evaluated draw. Separate columns use separate
fits. Both axes are independent in every facet. Coordinates and gradients
are in the displayed geometry; online samples are returned in the original
NCP frame and then transformed to the learned frame exactly once.

These are **log-density-gradient geometry plots**, not scalar online-loss
curves, potential-energy gradients, or reconstructed warmup histories.

## Reusable native pieces

- `brm_output_draws`: descriptor-owned predictor/output extraction.
- `brm_predictive_draws`: native predictive replay with explicit per-draw seeds.
- `hsgp_coordinate_draws`: descriptor-owned HSGP geometry and coordinate/gradient transport.
- `hsgp_transform_draws`: the same frame transform for already-extracted matrices.
- AoV extension: `brm_posteriorplot`, `brm_ppcplot`, `brm_pairplot`,
  `brm_centerednessplot`, `brm_centering_lossplot`, `brm_gradientplot`.

Input matrices to these helpers are **draws × coordinates**, unlike WarmupHMC's
returned **coordinates × draws** matrices. Existing research rendering,
complete AoV export acceptance, and online-loss query integration remain work
in progress; this preview is not a new claim of case-study completion.
