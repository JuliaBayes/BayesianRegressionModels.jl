# Result provenance

This directory contains the small publication evidence tables behind the
eight-schools case study: the noncentered pilot, selected-partial refit and
online adaptive-centering fit, each with 10,000 retained draws from
`Xoshiro(1)` under ordinary WarmupHMC defaults.

- `observations.tsv` — the eight reported estimates and standard errors.
- `saved_source_density_gradient_audit.tsv` — BridgeStan density/gradient
  identity of the BRM target against the immutable source at 16 saved pilot
  draws, with the 8-dimensional tau Jacobian made explicit.
- `diagnostics.tsv` — max split R-hat, min bulk/tail ESS, divergences and
  both exact gradient counters per fit.
- `fit_costs.tsv`, `workflow_costs.tsv` — total vs retained-sampling NUTS
  counters per fit and for the pilot-plus-refit workflow.
- `centeredness.tsv`, `offline_centeredness.tsv`,
  `online_centeredness.tsv`, `offline_loss_profiles.tsv` — the offline
  selection and the online learned coordinates.
- `retrospective_online_losses.tsv`, `gradient_checks.tsv`,
  `frame_invariants.tsv` — the common-pilot online objective replay, the 72
  finite-difference gradient checks, and the coordinate-frame invariants.
- `provenance.toml`, `packages.tsv`, `diagnostics_provenance.toml` — exact
  source revisions, script hashes, sampler configuration and dependency pins.

These are byte copies of the outputs produced by `reproduce.jl` and
`prepare_gradient_diagnostics.jl` (see the [parent README](../README.md)
for the exact commands); `validate_results.jl` checks the run directory,
not these copies. Full posterior draws (`.jls`), per-draw coordinate
tables, pair/gradient display tables and checkpoints stay in scratch —
they are tens of megabytes and are not needed to check any number on the
page. The corresponding figures are in
`docs/src/assets/adaptive-eight-schools/`.
