# Adaptive HSGP centering: source reproduction

This reproduces the two fits and scientific figures in
[Generable's HSGP case study](https://www.generable.com/post/hsgp-reparam):
a noncentered pilot, then a **fresh** partially centered fit selected from
that pilot. The centered scatter panels transform the pilot draws; there is
no centered sampling arm.

The previous `results/` files are historical smoke-test artifacts, **not a
completed reproduction**. They reduced the basis and draw counts, bypassed
Pathfinder, and changed the sampler settings. Do not use them as posterior
estimates or backend performance evidence. The full replacement is run into
a separate output directory. The new full-run diagnostics, source audit and
selected profiles are checked in under `results/source-faithful/`; their
figures are in `docs/src/assets/adaptive-hsgp/`.

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
- Offline selection uses `0:0.01:1`. The selector reports inadmissible
  floating-point underflow candidates explicitly; these are not silently
  replaced by another prior or basis count.

## Run

Use the repository's resolved `test` environment. Run from the repository
root and put large outputs in a fresh disk-backed scratch directory.

```sh
# Compile the immutable original .stan file and compare its target with BRM.
BRM_ADAPTIVE_OUTPUT="$TMPDIR/hsgp-source-audit" \
  julia --startup-file=no --project=test research/adaptive_centering/audit_source.jl

# Two full original-configuration StanBlocks fits, then all source-style figures.
BRM_ADAPTIVE_RUNTIME=1 BRM_ADAPTIVE_OUTPUT="$TMPDIR/hsgp-source-fit" \
  julia --startup-file=no --project=test research/adaptive_centering/reproduce.jl

# Re-render saved tables without resampling (base R with cairo).
Rscript research/adaptive_centering/plot_results.R "$TMPDIR/hsgp-source-fit"

# Check full draw receipts and all 40 selections against the literal source loss.
julia --startup-file=no --project=test research/adaptive_centering/validate_results.jl \
  "$TMPDIR/hsgp-source-fit" "$TMPDIR/hsgp-source-audit"

# Separate extension: online StanBlocks centering with the same full model
# and sampling defaults. This is additional to the original two-fit study.
BRM_ADAPTIVE_ONLINE=1 BRM_ADAPTIVE_OUTPUT="$TMPDIR/hsgp-online-fit" \
  julia --startup-file=no --project=test research/adaptive_centering/reproduce.jl

# Add online posterior and online-versus-offline centering panels without resampling.
Rscript research/adaptive_centering/plot_results.R \
  "$TMPDIR/hsgp-source-fit" "$TMPDIR/hsgp-online-fit"
```

Reduced-budget environment overrides fail instead of silently changing the
case study. A completed fit is saved before plotting and is never
automatically overwritten. A file named `STOP` in the output directory asks
the sampler to stop at its next boundary; an incomplete fit is labeled and
cannot report case-study completion.

Turing sampling is disabled until its gradients have passed a matched-value
and warmed-runtime comparison with StanBlocks on this full model. A separate
backend optimization task owns that gate; finite gradients alone do not
satisfy it.

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

Figures use the source's standardized response units; the noise and
hyperparameter axes are logarithmic. Hyperparameter labels follow the
actual model semantics (length versus marginal SD), not the source plotting
helper's swapped labels.

Diagnostics use MCMCDiagnosticTools' rank-normalized split R-hat, bulk ESS,
tail ESS and retained-draw divergence count/percent. As in the article,
each fit is one chain: split R-hat is a within-chain diagnostic, not proof
that independent chains agree. New-version results must be reported as
observed; the source article's numerical results are not acceptance targets
to be manufactured by retuning the sampler.
