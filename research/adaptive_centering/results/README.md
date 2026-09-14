# Result provenance

Use **`source-faithful/`** for the corrected 133-row, 20-basis, 10,000-draw
case study and its full-budget online StanBlocks extension. The corresponding
figures are in `docs/src/assets/adaptive-hsgp/`.

The TSV files directly in this directory are historical short-run artifacts.
They used a reduced basis, changed sampler configuration and inadequate draw
counts. The old native Turing path also retained a length-scale lower bound
that did not match the explicit source prior. These files are preserved for
audit, not accepted posterior estimates or backend-performance evidence.

The corrected full offline draws, plot tables and nine source-style figures
are archived centrally at `/home/niko/.local/state/kb-agents/uploads/eaaf3c4129b24f53.targz`.
The source audit and result validation commands are in the parent README.
