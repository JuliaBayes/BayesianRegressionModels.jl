# Adaptive HSGP centering: source reproduction

This reproduces the two fits and scientific figures in
[Generable's HSGP case study](https://www.generable.com/post/hsgp-reparam):
a noncentered pilot, then a **fresh** partially centered fit selected from
that pilot. The centered scatter panels transform the pilot draws; there is
no centered sampling arm.

An online adaptive-centering fit extends the comparison using the same model
and sampling configuration. Full-run diagnostics, exact gradient costs,
source audits and selected profiles are checked in under
`results/source-faithful/`; their figures are in
`docs/src/assets/adaptive-hsgp/`.

## Exact source contract

- [Companion source](https://github.com/generable/public-materials/tree/0d00b8535e2c20c49017d03c7b060940eb8e7041/blog/hsgp-reparam):
  `0d00b8535e2c20c49017d03c7b060940eb8e7041`.
- Data: all 133 `MASS::mcycle` observations. CSV SHA-256:
  `b89a1e4eb0391a982b32be3e378df00e8593ff9971e9425e9c5d7929b74f9801`;
  Rdatasets revision `1dcc2bf5f955cc1224a3e1307256e1fe86b68dae`.
- Acceleration is divided by its sample standard deviation. Time is mapped
  to `[-1,1]`. Both zero-mean HSGPs use 20 sine basis functions on
  `[-1.5,1.5]`, normalized by `sqrt(1.5)`.
- Source: `log(length_scale) ~ Normal(0,4)` and
  `log(marginal_sd) ~ Normal(0,4)`, independently for both GPs.
  BRM: `LogNormal(0,4)` on each positive scale. Equivalence includes the
  positive parameter's unconstraining Jacobian; it is tested against the
  source's actual normalized Stan density and all 44 gradients.
- Each fit uses `Xoshiro(1)`, `n_draws=10_000`, and WarmupHMC defaults,
  including Pathfinder initialization. No initializer, evaluation window,
  acceptance rate, tree depth or transformation setting is overridden.
  `monitor_ess=true` retains the diagnostic monitoring enabled by the
  source's progress display. Checkpoints and callbacks only record progress.
- Offline selection uses `0:0.01:1`. Candidates whose partial-coordinate
  scales underflow are reported as inadmissible; the noncentered endpoint
  remains available.

## Run

Use the repository's resolved `test` environment and resolve the separate
`research/adaptive_centering/plots` environment for Julia/AlgebraOfVega rendering.
Run from the repository root and put large outputs in a fresh disk-backed
scratch directory.

The scripts require WarmupHMC at or after `913da79d271276b2e6847b9699f5eb1957d050c7`,
which returns the exact retained-sampling gradient counter. The reproduction
requires that counter, saves it alongside the run-total counter and elapsed fit
time, and prints all three at completion. `report_costs.jl` checks the fit record
against the final checkpoint before computing either efficiency ratio.
Turing is not sampled in this reproduction.

```sh
# Compile the immutable original .stan file and compare its target with BRM.
BRM_ADAPTIVE_OUTPUT="$TMPDIR/hsgp-source-audit" \
  julia --startup-file=no --project=test research/adaptive_centering/audit_source.jl

# Two full original-configuration StanBlocks fits, then all source-style figures.
BRM_ADAPTIVE_RUNTIME=1 BRM_ADAPTIVE_OUTPUT="$TMPDIR/hsgp-source-fit" \
  julia --startup-file=no --project=test research/adaptive_centering/reproduce.jl

# Compare the original partial Stan target with the fresh refit's saved positions.
julia --startup-file=no --project=test research/adaptive_centering/audit_partial_source.jl \
  "$TMPDIR/hsgp-source-fit" "$TMPDIR/hsgp-partial-source-audit"

# Re-render saved tables without resampling (Julia/AlgebraOfVega).
julia --startup-file=no --project=research/adaptive_centering/plots \
  research/adaptive_centering/plot_results.jl "$TMPDIR/hsgp-source-fit"

# Check full draw receipts and all 40 selections against the literal source loss.
julia --startup-file=no --project=test research/adaptive_centering/validate_results.jl \
  "$TMPDIR/hsgp-source-fit" "$TMPDIR/hsgp-source-audit"

# Separate extension: online StanBlocks centering with the same full model
# and sampling defaults. This is additional to the original two-fit study.
BRM_ADAPTIVE_ONLINE=1 BRM_ADAPTIVE_OUTPUT="$TMPDIR/hsgp-online-fit" \
  julia --startup-file=no --project=test research/adaptive_centering/reproduce.jl

# Add online posterior, learned-coordinate pair plots and centering comparison.
julia --startup-file=no --project=research/adaptive_centering/plots \
  research/adaptive_centering/plot_results.jl \
  "$TMPDIR/hsgp-source-fit" "$TMPDIR/hsgp-online-fit"

# Evaluate the actual compiled BRM target at saved draws; this does not sample.
julia --startup-file=no --project=test \
  research/adaptive_centering/prepare_gradient_diagnostics.jl \
  "$TMPDIR/hsgp-source-fit" "$TMPDIR/hsgp-online-fit" "$TMPDIR/hsgp-diagnostics"

# Check that the common pilot has the same candidate loss landscape when
# stored in NCP, selected-partial, or online-selected coordinates. No sampling.
julia --startup-file=no --project=test \
  research/adaptive_centering/audit_loss_frames.jl \
  "$TMPDIR/hsgp-source-fit" "$TMPDIR/hsgp-online-fit" "$TMPDIR/hsgp-frame-audit"

# 1,000 transparent points per facet, with separate coordinate/gradient axes.
julia --startup-file=no --project=research/adaptive_centering/plots \
  research/adaptive_centering/plots/gradient_preview.jl "$TMPDIR/hsgp-diagnostics"

# One common pilot reference, GP-only facets, raw correlation with [-1,0] y-limits.
julia --startup-file=no --project=research/adaptive_centering/plots \
  research/adaptive_centering/plots/online_loss_preview.jl "$TMPDIR/hsgp-diagnostics"

# Include the pilot pair plots in gradient-loss-selected coordinates (no new fit).
julia --startup-file=no --project=research/adaptive_centering/plots \
  research/adaptive_centering/plot_results.jl "$TMPDIR/hsgp-source-fit" \
  "$TMPDIR/hsgp-online-fit" "$TMPDIR/hsgp-figures" "$TMPDIR/hsgp-diagnostics"

# Verify both NUTS counters and compute total/sampling ESS per gradient.
julia --startup-file=no --project=test research/adaptive_centering/report_costs.jl \
  "$TMPDIR/hsgp-source-fit" "$TMPDIR/hsgp-online-fit" "$TMPDIR/hsgp-costs"
```

Reduced-budget environment overrides fail instead of silently changing the
case study. A completed fit is saved before plotting and is never
automatically overwritten. A file named `STOP` in the output directory asks
the sampler to stop at its next boundary; an incomplete fit is labeled and
cannot report case-study completion.

## Outputs and interpretation

`provenance.toml` and `packages.tsv` record the exact source, data, code,
dependency versions and sampler settings. `noncentered.jls` and
`partial.jls` save the full returned model-coordinate draws without native
library handles. The plotted tables and `figures/` contain:

1. Basis functions 1, 2, 19 and 20, and prior spectral standard deviations.
2. Pilot mean and conditional-noise posteriors with 90%, 80% and 50% central
   credible intervals.
3. Pilot weight versus marginal-SD/length-scale scatter in noncentered,
   centered and selected-partial coordinates.
4. Per-basis centering-loss profiles and all 40 selected centering values.
5. The fresh partially centered fit's posterior and weight/hyperparameter
   scatter panels.
6. Online posterior, online-versus-offline centeredness, and pair plots in the
   online-selected coordinates (all saved draws, transformed from the returned
   NCP frame once).
7. Three-configuration coordinate–gradient diagnostics, using 1,000 evenly
   selected draws per facet only for display (all 10,000 are evaluated).
8. Native online position–gradient correlation curves at default `w₁=0`,
   faceted only by GP and replayed on the common 10,000-draw pilot reference.
9. Pilot hyperparameter pair plots in coordinates selected by that gradient
   objective, distinct from the online fit's learned-coordinate plots.

The loss export retains all three fit-specific replay tables for auditing;
the displayed global landscape uses only the common pilot reference, not
source-coordinate facets. This is retrospective scoring, not warmup history:
saved posterior matrices lack the original trajectory weights and boundaries.
Online curves retain raw correlations and use fixed `[-1,0]` y-limits.
The offline proxy alone uses per-curve min–max scaling. Gradient previews use marker size 8
and opacity 0.25, without KDE, binning, smoothing, or a regression overlay.

`report_costs.jl` reports both exact counters for each fit. Run-total
NUTS evaluations include warmup and discarded epochs but exclude Pathfinder/setup
gradients. Sampling evaluations count only appended transitions corresponding
to the final retained draws. Minimum bulk and tail ESS over the 44 sampled
coordinates are divided by each denominator separately. `sample_source_fit`
also records elapsed wall seconds around the sampler call, including initialization,
first-use compilation and checkpoint I/O, but excluding preceding Stan compilation
and subsequent plotting. These timings are not warmed gradient microbenchmarks.

Figures use the source's standardized response units; the noise and
hyperparameter axes are logarithmic. Pair plots label the GP length scale
and marginal standard deviation separately.

Diagnostics use MCMCDiagnosticTools' rank-normalized split R-hat, bulk ESS,
tail ESS and retained-draw divergence count/percent. As in the article,
each fit is one chain: split R-hat is a within-chain diagnostic, not proof
that independent chains agree. Different library versions and parameter
orderings can produce different trajectories at the same seed.
