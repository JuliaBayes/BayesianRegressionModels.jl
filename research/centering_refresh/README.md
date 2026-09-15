# Case-study refresh after the online state-transport repair

These runs replace the adaptive-centering, eight-schools and radon comparisons
produced before WarmupHMC's active-position transport fix. The repaired source
is `d9eeaac2c092b80aba3ef5d608261faea9b81265`, published on dev in
`6b377cb23934022af5879d199a7c57abfac54c70`.

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
