# PosteriorDB Radon adaptive-centering study

This directory is a self-contained executable study of PosteriorDB posterior
`radon_all-radon_variable_intercept_slope_noncentered` at repository revision
`5545a1dd07ae297c36edecbcd82aa49097b4c385`. The immutable reference model,
metadata, data archive, and a complete inventory of the 25 radon posteriors at
that revision are under `reference/` and `variant_inventory.tsv`.

The selected model has two independently hierarchical vectors (386 county
intercepts and 386 county slopes), one likelihood covariate, and 777 scalar
parameters. It shares the structural maximum in the inventory with its centered
twin: 772 hierarchical effect cells and 777 total scalar parameters, more
cells than every other radon posterior at that revision except that twin.
Parameterization is not part of the complexity criterion; of the two
tied variants, the noncentered implementation is selected as the
adaptive-centering starting point.

The model uses named, independent intercept and slope blocks with explicit
half-normal scale priors:

```julia
sigma_y ~ Normal(0, 1; lower=0)
mu ~ 1 + floor_measure + (1 | county_intercept | county_idx) +
     (0 + floor_measure | county_slope | county_idx)
effect(mu, Intercept) ~ Normal(0, 10)
effect(mu, floor_measure) ~ Normal(0, 10)
sd(:, county_intercept) ~ Normal(0, 1)
sd(:, county_slope) ~ Normal(0, 1)
log_radon ~ Normal(mu, sigma_y)
```

The data passed to BRM contains the original `floor_measure`, `county_idx`
and `log_radon` columns.

## Reproduce

Resolve `test/Project.toml` with the repository's canonical environment script,
then run:

```sh
BRM_RADON_AUDIT_OUTPUT=/absolute/audit-directory \
  julia --startup-file=no --project=test \
  research/radon_centering/audit_source.jl

BRM_RADON_OUTPUT=/absolute/offline-directory \
BRM_RADON_RUNTIME=1 julia --startup-file=no --project=test \
  research/radon_centering/reproduce.jl

BRM_RADON_OUTPUT=/absolute/online-directory \
BRM_RADON_ONLINE=1 julia --startup-file=no --project=test \
  research/radon_centering/reproduce.jl

julia --startup-file=no --project=test \
  research/radon_centering/prepare_diagnostics.jl \
  /absolute/offline-directory /absolute/online-directory \
  /absolute/diagnostics-directory

julia --startup-file=no --project=test \
  research/radon_centering/report_costs.jl \
  /absolute/offline-directory /absolute/online-directory \
  /absolute/diagnostics-directory /absolute/cost-directory

bash research/radon_centering/plots/setup_plots_env.sh

julia --startup-file=no --project=research/radon_centering/plots \
  research/radon_centering/plots/plot_results.jl \
  /absolute/offline-directory /absolute/online-directory \
  /absolute/diagnostics-directory
```

The plotting environment is rebuilt from exact source pins by
`plots/setup_plots_env.sh`, which materializes clean detached clones below
the ignored `plots/.bootstrap/` cache (host mirror first, public GitHub
origin otherwise) and resolves with the canonical resolver. Pins:
AlgebraOfVega `4124226`, DynamicObjects `1d268ea` (source-pinned per user
decision),
HTMXObjects `a813640`, HTMX `d52ce5b`, Treebars `c02aa16`;
AlgebraOfGraphics 0.13.2, CairoMakie 0.15.14 and Makie 0.24.14 resolve from
the registry under the `[compat]` bounds in `plots/Project.toml`. Re-running
the script reuses the cached checkouts. Rendering records dependency identities
and verifies that they stay unchanged during the run.

Ten figures land in `<diagnostics>/figures/` with `figure_manifest.tsv`
hashes binding every PNG to its AlgebraOfVega specification: `data-ppc`,
`selected-centeredness`, `offline-loss-profiles`, `online-loss-profiles`,
four pair plots (`pair-pilot-intercept`, `pair-pilot-slope`,
`pair-fits-intercept`, `pair-fits-slope`) and two position–gradient scatters
(`position-gradient-intercept`, `position-gradient-slope`). The docs page
embeds copies under `docs/src/assets/adaptive-radon/`.

Scatter plots select three coordinates per role by the inferred offline
centeredness: minimum, nearest 0.5, maximum, with county-index tie-breaking.
The selected counties and controls are in `representative_coordinates.tsv`.
Every comparison starts with the NCP pilot transformed to centered coordinates.
Pilot comparisons show the same physical draws in each column; fresh-fit
comparisons show the separate selected and online draws beside that reference.
The sampling and cost baseline remains NCP.

The audit compares normalized BridgeStan densities and all 777 gradients against
the immutable PosteriorDB Stan model. The pilot is a full noncentered fit. Its
draws feed the offline scalar KL/log-scale proxy; those selected per-cell
centerings are frozen for a fresh fixed-partial WarmupHMC refit. A separate full
online run uses WarmupHMC's default position-gradient-correlation objective with
`w₁=0`. All fits use `Xoshiro(1)`, request 10,000 retained draws, set
`monitor_ess=true`, and otherwise keep normal WarmupHMC initialization and
adaptation. All three fits sample through BRM → StanBlocks/BridgeStan with
the model above.

The fixed-partial refit evaluates the same compiled BRM/Stan target through a
fixed WarmupHMC source transform (`nonlinear_adapt=false`). It is a fresh fit,
not a transformation of pilot draws. Pair plots distinguish transformed pilot
geometry from fresh-fit geometry. Saved results are never overwritten; a `STOP`
file in an output directory requests an early boundary and makes the run
ineligible for completion.

## Outputs and interpretation

- `provenance.toml` and `packages.tsv`: exact source/data/model hashes, code,
  dependency commits, sampler settings, timing/counter scope.
- `source_density_gradient_audit.tsv`: six full-vector density and gradient
  comparisons, plus the complete coordinate map.
- `selected_centeredness.tsv`, `offline_loss_profiles.tsv`: all 772 per-cell
  selections and all 101-point offline profiles.
- `retrospective_online_losses.tsv`: all 8,492 common-pilot candidate scores
  from WarmupHMC's public `candidate_scoring_losses`; this is retrospective
  scoring, not recorded warmup history.
- `online_centeredness.tsv`: independently learned values from the separate
  online fit.
- `diagnostics.tsv`, `online_diagnostics.tsv`, and `fit_costs.tsv`: split
  R-hat, bulk/tail ESS, divergences, retained draws, exact total and retained-
  sampling gradient counters, wall time, measured Julia compilation time, and ESS-per-gradient ratios.
- `representative_coordinates.tsv`: per-role display selection and inferred controls.
- `coordinate_pairs.tsv`, `coordinate_gradients.tsv`, and
  `density_jacobian_gradient_invariants.tsv`: all 10,000 draws for pair
  geometry, exactly 1,000 evenly selected saved draws per displayed gradient
  facet, and transformed density/Jacobian/gradient/roundtrip checks.
- `ppc_curves.tsv` (committed as `results/ppc_intervals.tsv`): native BRM
  predictive-replay quantiles for all 12,573 observations and 10,000 pilot draws.
- `figures/figure_manifest.tsv`: hashes binding every PNG to its AlgebraOfVega
  specification.

All three returned fit matrices hold model (NCP) coordinates, including the
fixed-centering refit. Final checkpoints hold sampler coordinates. Extraction
checks the independent scalar checkpoint-to-model map against the actual
returned positions, then verifies displayed coordinates, Gaussian gradients,
the transformed density and the physical linear predictor. Diagnostics use
model coordinates for every arm.

The PPC table preserves each original row index, county and floor code.
County facets show independent 50% and 90% predictive intervals with observed
values overlaid in original row order and coloured by floor code. Displayed
counties span the minimum, median and maximum sample sizes among counties
with at least five floor-0 and five floor-1 observations. All recorded floor
codes in those counties are displayed; the model and full PPC table include
every observation.

Delivered receipts committed under `results/`: the source audit and coordinate
map, `fit_costs.tsv` (both exact counters, ESS minima and workflow charge),
per-fit diagnostics, post-hoc and online centeredness selections, display
coordinate invariants, and per-run provenance plus dependency snapshots
(`offline_`/`online_provenance.toml`, `offline_`/`online_packages.tsv`).
Full `.jls` draws, checkpoints, compiled models and the large display tables
(loss profiles, retrospective scores, pairs and gradients) stay in
the run directories outside Git; the docs page embeds figure copies under
`docs/src/assets/adaptive-radon/`.

The displayed gradient facets retain independent axes, marker size 4, opacity
0.25, and no KDE, binning, smoothing, or fitted line. All 10,000 retained draws
enter each candidate-loss calculation; display thinning applies only to
gradient scatters. Missing or non-finite loss rows remain missing. ESS is a
single-chain diagnostic: rank-normalized split R-hat does not establish
agreement between independent chains.

Total NUTS counters include warmup and discarded restart epochs and exclude
Pathfinder/setup calls. Sampling counters include only retained appended
transitions. `fit_seconds` surrounds the sampler call only and excludes Stan
compilation/setup. The workflow cost row charges both the pilot and fresh
post-hoc refit when interpreting the latter.
