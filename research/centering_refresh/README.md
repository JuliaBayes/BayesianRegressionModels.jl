# Case-study refresh after the online state-transport repair

These runs replace the adaptive-centering, eight-schools and radon comparisons
produced before WarmupHMC's active-position transport fix. The repaired source
is `d9eeaac2c092b80aba3ef5d608261faea9b81265`, published on dev in
`6b377cb23934022af5879d199a7c57abfac54c70`.

The completed HSGP online-gradient arm additionally uses
`7aed40b18bd4cdabb75330f285d5c9b355575ab9`. It rejects a proposed centering
change when transporting the adaptation sample or active state would produce
nonfinite positions, gradients or density. Rejection restores the old controls,
scoring state, adaptation sample and active point atomically. All other 17
completed arms use the published active-position repair above. This package
difference is explicit in each arm's `packages.tsv` and archived environment.

The models and data still come from each existing case-study directory. All
fits use one chain, seed 1, 10,000 retained draws and the ordinary WarmupHMC
initialization/adaptation configuration. The harness compares fixed NCP/CP,
post-hoc position and gradient losses, and online position and gradient losses.
The position loss selects the existing internal criterion process-locally;
there is no new public loss-selection API.

`run.jl CASE ARM OUTPUT_ROOT` saves each arm under its own immutable directory.
Cases are `hsgp`, `eight`, `radon`; arms are `ncp`, `cp`, `posthoc_position`,
`posthoc_gradient`, `online_position`, `online_gradient`. Run NCP before either
post-hoc arm. Both selectors use its saved source gradients and positions.
Post-hoc selection uses `0:0.01:1`; online uses the native `0:0.1:1` grid.

The target wrapper counts **all** density-and-gradient calls, including
Pathfinder initialization and active-state reevaluations. The separate native
sampling counter records only transitions contributing retained draws. Each
post-hoc workflow includes the entire NCP pilot cost; the pilot's ESS is not
added to the refit's ESS. Selection uses stored gradients, so it introduces
no further fitted-target gradient calls. Density/coordinate audits and plot
preparation are recorded separately from fitting costs.

Every arm uses the same scientific quantities within its study:

- Eight schools: population mean, group SD, and eight school treatment effects.
- Radon: population intercept/slope, two group SDs, residual SD, and 386 county
  intercept totals and 386 county slope totals.
- HSGP: both length scales and marginal SDs, plus mean and conditional-noise
  functions at each distinct observed time.

Positive scales are stored in log coordinates when supplied that way by Stan;
rank-normalized ESS is invariant under that monotone change. Standardized
random-effect coordinates are excluded from the scientific minimum.

The main table reports total gradients and two efficiencies relative to the
same-study NCP baseline: minimum bulk ESS per sampling gradient and minimum
bulk ESS per full-workflow gradient. Divergences and within-chain split R-hat
are diagnostic qualifications, not extra competing efficiency tables.

The historical results are retained for inspection. Their online warmup
transitions may have displaced the physical state when centering changed;
they should not be used to rank the repaired adaptation implementation.

## Reproduction

Use the repository's Julia test environment with BridgeStan and Enzyme. The
root `test/Project.toml` pins the finite-transport implementation. Each archived
fit also contains the exact Project and Manifest used for that arm. A rerun
using the final pin for every arm is a new experiment; it does not replace the
recorded provenance of the existing 17 fits that used the first repair.

```sh
julia --project=test research/centering_refresh/run.jl hsgp ncp OUTPUT
julia --project=test research/centering_refresh/run.jl hsgp cp,posthoc_position,posthoc_gradient,online_position,online_gradient OUTPUT
julia --project=test research/centering_refresh/export.jl hsgp OUTPUT
python3 research/centering_refresh/summarize.py hsgp OUTPUT
julia --project=research/adaptive_centering/plots research/centering_refresh/plot.jl hsgp OUTPUT/export OUTPUT/figures
```

Substitute `eight` or `radon` for the other studies. The plot environment has its
own Project and Manifest. Copy the resulting PNGs into
`docs/src/assets/centering-refresh-CASE/`, then run `refresh_pages.py` to refresh
the numerical table and surrounding narrative. That generator retains the
audited model/example blocks from repository commit `b7fd3f5`.

`results/receipt.json` records wrapper logs and exact exits, source hashes,
package revisions, and the completed fit roster. `results/logs/` retains the
full relevant fitting, audit, export and plotting logs. These execution times
include process setup and are not presented as sampler timing comparisons.

## Scientific and coordinate checks

The original independent source-model density/gradient audits were rerun:
eight schools, radon (20 checks), and HSGP (26 checks) all passed. Export then
checked three saved points from every completed arm: physical/model mapping,
Jacobian-adjusted density and transformed gradients, giving 18 checks per
study. HSGP function summaries also agree with native BRM prediction at those
draws. `frame_checks.tsv` preserves the numerical errors.

The final transport implementation passed 152 active-state checks, nine
synthetic rejection checks, and the existing reparametrization and invariant
scoring test files. A captured HSGP adaptation pool reproduced the nonfinite
gradient transport; all eight rejection/metric checks passed with the guard.
See `results/transport_guard/` for the patch, analysis and captured fixture.
This is a targeted regression suite, not a claim that the entire package suite
was run.

## Raw-fit retention

`results/capsules.json` inventories 19 compressed archives: all 18 completed
fits and the original HSGP online-gradient failure under `6b377cb`. Each archive
contains the raw fit/checkpoint, generated Stan, run source, summaries and
environment. Large archives are split into 20 MiB KB uploads. Every uploaded
part was downloaded and checked against its SHA-256 hash before marking its
record complete. Raw fitted draws are retained outside Git.

For each record, download its `parts` in ascending `index` order from
`http://localhost:4200/code?path=URL_ENCODED_PART_PATH&raw=1`, verify the individual
hashes, concatenate, and verify the full archive hash before extracting into
an empty directory. `archive_fits.py` contains the upload and verification code.
The failed-arm key is `hsgp/online_gradient_failed_6b377cb`; its archive retains
the original internal directory name `online_gradient`. Keep it separate from
the completed final arm when extracting.

The captured numerical failure is a separate small fixture listed in
`results/transport_guard/capture.json`. After restoring its `transport-failure.jls`
to `OUTPUT/online_gradient/`, run:

```sh
julia --project=test research/centering_refresh/audit_rejection.jl hsgp online_gradient OUTPUT
```

The HSGP online-gradient rerun used the same seed and completed 10,000 retained
draws, with one rejected coordinate update and 12 sampling divergences. The
remaining divergences in this broad-prior model are reported alongside the
efficiency table; the numerical guard is not a general convergence guarantee.
