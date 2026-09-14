# Local AoV diagnostic preview

The case-study renderer and entry point now use Julia/AlgebraOfVega, with no R
dependency. Public figure replacement is still pending visual acceptance and
documentation publication.
Loading `BayesianRegressionModels, AlgebraOfVega` enables the optional BRM
plotting extension; fitting alone does not acquire a plotting dependency.

## Full case-study figures

```sh
julia --startup-file=no --project=research/adaptive_centering/plots \
  research/adaptive_centering/plot_results.jl \
  /absolute/offline-fit-directory /absolute/online-fit-directory /absolute/figure-directory
```

This renders the eleven original scientific figures plus the online pair plot.
All 10,000 saved draws appear in each pair panel; the 90%, 80%, and 50% posterior
bands use the saved full-fit quantiles. Native `hsgp_transform_draws` supplies
every coordinate change. `sdraw!` composes the AoV panels; CairoMakie supplies
only layout and static file export. The saved fits and their tables are not
rewritten, and no sampling is invoked.

## Saved-fit coordinate–gradient preview

Resolve the repository test environment and this plotting environment using
the ecosystem's canonical resolver. From the repository root:

```sh
julia --startup-file=no --project=test \
  research/adaptive_centering/prepare_gradient_diagnostics.jl \
  /absolute/offline-fit-directory /absolute/online-fit-directory /absolute/diagnostics-directory

julia --startup-file=no --project=research/adaptive_centering/plots \
  research/adaptive_centering/plots/gradient_preview.jl /absolute/diagnostics-directory

julia --startup-file=no --project=research/adaptive_centering/plots \
  research/adaptive_centering/plots/online_loss_preview.jl /absolute/diagnostics-directory
```

No sampler is called. The first command checks the generated model against the
saved producer's Stan source, evaluates all 10,000 saved draws from each fit,
and checks selected transformed gradients against finite differences of the
compiled target. The output includes all 240,000 coordinate–gradient rows for
two GPs and four basis functions, plus the 72 numerical checks.

The second command generates AoV plots, PNGs, and `kb-aov/v1` preview fences.
The default display uses 1,000 evenly spaced saved draws per panel for bases
1, 2, 19 and 20: 12,000 transparent points per GP figure. This is the user's
chosen display subset, not a fitting or loss-calculation limit. No KDE,
binning, smoothing, or fitted regression line is applied. Calling
`gradient_preview(dir; draws_per_facet=nothing)` keeps every saved point.
It writes one fence per GP as well as a combined file;
delivery requires the KB's approved large-payload plot support (the old small
inline limits are not a scientific-data limit). Separate columns use separate
fits. Both axes are independent in every facet. Coordinates and gradients
are in the displayed geometry; online samples are returned in the original
NCP frame and then transformed to the learned frame exactly once.

These are **log-density-gradient geometry plots**, not scalar online-loss
curves, potential-energy gradients, or reconstructed warmup histories.
The first command also exports `retrospective_online_losses.tsv` through
WarmupHMC's public `candidate_scoring_losses` API. Its objective is the actual
default position–gradient correlation (`w₁=0`), replayed with unit weights on
the saved draws, not the original online trajectory stream. Missing/nonfinite
losses are retained in that table rather than replaced or labeled as history.
The loss preview uses `brm_centering_lossplot(...; normalization=:minmax)`:
each configuration/GP/basis curve is independently mapped to [0,1], matching
the offline figure. Its minimum is unchanged, but absolute correlation
magnitudes cannot be compared after this display normalization. Raw values
are retained in the input table and in the plot's `raw_loss` column;
`online_loss_preview(dir; normalization=:none)` shows the raw correlations.
Constant curves map to zero; missing/nonfinite entries remain gaps.

## Reusable native pieces

- `brm_output_draws`: descriptor-owned predictor/output extraction.
- `brm_predictive_draws`: native predictive replay with explicit per-draw seeds.
- `hsgp_coordinate_draws`: descriptor-owned HSGP geometry and coordinate/gradient transport.
- `hsgp_transform_draws`: the same frame transform for already-extracted matrices.
- AoV extension: `brm_posteriorplot`, `brm_ppcplot`, `brm_pairplot`,
  `brm_centerednessplot`, `brm_centering_lossplot`, `brm_gradientplot`.

Input matrices to these helpers are **draws × coordinates**, unlike WarmupHMC's
returned **coordinates × draws** matrices. Complete AoV export acceptance,
KB preview delivery, and public documentation integration remain work
in progress; this preview is not a new claim of case-study completion.
