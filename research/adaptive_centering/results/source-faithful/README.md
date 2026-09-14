# Full original-configuration results (2026-09-14)

All fits use 133 observations, 20 basis functions per GP, `Xoshiro(1)`,
10,000 requested draws, source-equivalent hyperpriors and default WarmupHMC
initialization/adaptation. Each retained exactly 10,000 draws.

The offline study has only the noncentered pilot and fresh selected-partial
refit. The online fit is a separate StanBlocks extension. No Turing samples
were produced for this corrected study.

`provenance.toml` and `online_provenance.toml` record each runner's exact hash
and base commit. The only runner difference from checkpoint `09f3fed` was a
correction to the observational progress callback; the model and sampler
settings were unchanged. Raw fit records were saved before plotting.

The offline runner completed both fits but exited 1 in its R reader because
Julia's lowercase Boolean strings needed explicit conversion. Re-rendering
after the reader fix exited 0 without resampling. Independent verification
of the complete fit records, every source-literal pilot loss and selected
centering, and all nine figures passed 8,221 checks. The online runner exited
0. Its sampler-returned draws were used directly in the original target frame,
with no second coordinate transformation.

The actual original Stan source audit passed 26 checks: six normalized-density
and all-component gradient comparisons, the 44-coordinate mapping, and the
four positive-scale prior/Jacobian identity. See
`source_density_gradient_audit.tsv` and `source_coordinate_map.tsv`.

The diagnostics deliberately retain the observed limitations: the offline
pilot/refit had 34/16 divergences, respectively. The online fit had zero.
R-hat is rank-normalized and split within one chain, not a between-independent-
chains convergence claim. Bulk and tail ESS come from MCMCDiagnosticTools.
