# Full source-configuration results (2026-09-14)

These fits record exact retained-sampling and run-total NUTS gradient
counters. All use 133 observations, 20 basis functions per GP, `Xoshiro(1)`,
10,000 requested draws, source-equivalent hyperpriors and ordinary WarmupHMC
initialization/adaptation. Each retained exactly 10,000 draws.

The source workflow has a noncentered pilot and a fresh selected-partial
refit. The online fit is a separate StanBlocks extension. No centered or
Turing sampling arm was run.

## Producing environment and measurements

`provenance.toml` and `online_provenance.toml` bind both runs to BRM
`6fa5ad1e191d1812b86a1c7b9eb9fa413c1d7bc4` and the exact reproduction-script
SHA-256. Their complete package snapshots were identical and are retained in
`packages.tsv`. Important pins are WarmupHMC
`deeea1d128d5235ad0ecb2fd911a6d881f1ac2c2`, StanBlocks
`c0b5b9197e2d06cf284f1990db024c37ed2b9d47`, and BridgeStan 2.9.0 under
Julia 1.10.11. Dependency snapshots were checked unchanged after each run.

The source audit, two offline fits, online fit and cost reporter completed
with process exit 0. `fit_costs.tsv` contains both exact gradient counters,
elapsed sampler-call time, minimum bulk/tail ESS over all 44 sampled model
coordinates, and both ESS-per-gradient ratios. The fit record and final
checkpoint agree on both counters. Run-total counts include NUTS warmup and
discarded epochs, but exclude Pathfinder/setup gradient calls. Sampling counts
include only transitions appended to the final retained sample.

The sampler calls ran sequentially, pinned to CPU 3 with one Julia and BLAS
thread, on the shared strato2 host. Elapsed times include initialization,
first-use Julia/AD compilation and checkpoint I/O, but exclude preceding Stan
compilation, post-fit extraction, plotting and offline selection. These are
observed call times, not warmed or replicated performance benchmarks.

## Verification

The actual original noncentered Stan source audit passed 26 checks: six
normalized-density/all-gradient comparisons, the 44-coordinate mapping, and
the four positive-scale prior/Jacobian identity. See
`source_density_gradient_audit.tsv` and `source_coordinate_map.tsv`.

The original adaptive Stan program was compared directly with the BRM partial
model at 16 saved refit positions: 51 checks passed, with maximum absolute
density/gradient differences `9.95e-14` / `5.59e-12`. Generated Stan source
must match the saved producer byte-for-byte before saved coordinates are used.
See `partial_source_density_gradient_audit.tsv` and
`partial_source_coordinate_map.tsv`.

Independent verification of complete fit records, every source-literal
pilot loss and all 40 selected centerings, and nine source-style figures passed
8,221 checks. Gradient diagnostics evaluated all 10,000 draws per fit, produced
1,320 finite native candidate scores, and passed 72 finite-difference checks
(maximum scaled error `3.95e-9`, in `gradient_checks.tsv`).
`audit_loss_frames.jl` repeated the common-pilot reference in three stored
coordinate frames: all 1,320 comparisons agree within `1.11e-15`
(`same_draw_loss_frame_comparison.tsv`). Neither audit invokes a sampler.

The observed divergence counts are 99 / 36 / 0 for NCP / selected-partial /
online. R-hat is rank-normalized and split within one chain, not a claim that
independent chains agree. Bulk and tail ESS come from MCMCDiagnosticTools.
The page and its Julia/AlgebraOfVega figures use these results.
