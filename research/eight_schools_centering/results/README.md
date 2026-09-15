# Publication evidence

These tables describe four full PosteriorDB eight-schools fits: noncentered,
manually fully centered, offline-selected partial centering, and online
adaptation. Each retains 10,000 draws from `Xoshiro(1)`.

- `diagnostics.tsv`: split R-hat, minimum bulk/tail ESS, divergences and cost
  ratios on the same ten model coordinates in every fit.
- `fit_costs.tsv`, `workflow_costs.tsv`: exact total and retained-sampling
  NUTS counters, including the pilot cost for offline selection.
- `timing_repetitions.tsv`, `priming_costs.tsv`: complete first-use fits and
  three warmed repetitions, with measured Julia compilation and GC time.
- `centeredness.tsv`, `offline_centeredness.tsv`, `online_centeredness.tsv`,
  `offline_loss_profiles.tsv`: selected controls and the offline objective.
- `returned_checkpoint_frames.tsv`: independent scalar reconstruction of
  returned model draws from sampler-coordinate checkpoints.
- `gradient_checks.tsv`, `frame_invariants.tsv`,
  `retrospective_online_losses.tsv`: displayed-gradient checks and the online
  objective evaluated on a common pilot sample.
- `observations.tsv`, `ppc_intervals.tsv`: original school order, observations
  and posterior predictive interval summaries.
- `saved_source_density_gradient_audit.tsv`: PosteriorDB target equivalence
  at 16 retained noncentered draws.
- `provenance.toml`, `packages.tsv`, `diagnostics_provenance.toml`: source,
  data, producer, extraction and dependency identities.

The commands and coordinate contracts are in the [study README](../README.md).
The figures are in `docs/src/assets/adaptive-eight-schools/`.
