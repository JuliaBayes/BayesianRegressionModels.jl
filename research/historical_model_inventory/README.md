# Historical BRM model inventory

This directory recovers the removed 2026-05 historical catalogue, preserves
its claims and receipts, translates each row conservatively to current BRM
semantics, and records staged compiler/runtime evidence. Historical-source
fidelity and current-BRM capability are intentionally separate axes.

## Reproduce

Run from a checkout that contains the historical Git object and whose current
environment has been resolved with the ecosystem's canonical local-dependency
resolver.

```sh
julia --startup-file=no research/historical_model_inventory/extract.jl \
  05c3f465e7987e8d7caa7e214fedddd90415a922
julia --startup-file=no research/historical_model_inventory/translate.jl
julia --startup-file=no --project=web-macro \
  research/historical_model_inventory/probe.jl
julia --startup-file=no --project=web-macro \
  research/historical_model_inventory/probe.jl --runtime
julia --startup-file=no --project=web-macro \
  research/historical_model_inventory/surface_controls.jl
julia --startup-file=no --project=web-macro \
  research/historical_model_inventory/runtime_controls.jl
julia --startup-file=no --project=web-macro \
  research/historical_model_inventory/refresh_binomial_logit.jl <expected-brm-sha>
julia --startup-file=no --project=web-macro \
  research/historical_model_inventory/refresh_beta_binomial.jl <expected-brm-sha>
julia --startup-file=no --project=test \
  research/historical_model_inventory/refresh_student_zip.jl <expected-brm-sha>
julia --startup-file=no research/historical_model_inventory/assemble.jl
julia --startup-file=no --project=web-macro \
  research/historical_model_inventory/gallery/validate.jl
```

The static probe builds BRMI and SBBRMI objects, derives the executable
descriptor, generates Stan, and runs stanc. `--runtime` additionally compiles
each accepted unique program with BridgeStan and checks a deterministic density
and gradient. It does not sample every synthetic analogue. Sampling is reserved
for the real-data sleepstudy control and executable kernel/plate route controls
in `runtime_controls.jl`.

Receipt reproduction has separate instructions in `receipts/README.md` because
it performs timestamped network retrieval and requires an exact historical
source checkout.

## Evidence products

- `historical_catalog.tsv`: lossless source claims, source lines, formula
  return values, and hidden/deployed scope.
- `translations.tsv`: exact-metadata and inferred-family translations, each
  with provenance and an explicit ordinary/kernel/plate/unsupported route.
- `family_audit.tsv`: the second-pass authoritative family recovery for every
  one of the 122 rows left unresolved by the catalogue metadata/receipt pass,
  including searched surfaces and negative evidence.
- `brm_surface_audit.tsv` and `stanblocks_surface_audit.tsv`: source- and
  test-backed classification of every previously claimed missing component.
- `surface_controls.tsv`: executable representative pass/raise controls for
  the audited current BRM surface.
- `capability_results.tsv`: row-expanded stage evidence; inherited results are
  only from an identical normalized probe.
- `binomial_logit_refresh.tsv`: exact five-row before/after receipt for the
  StanBlocks `329a178...` → `277f233...` scalar-trials RNG trace repair,
  including prediction/generated quantities and pointwise likelihood.
- `beta_binomial_surface_control.tsv`: exact reviewed-SHA control for both
  shape and mean/precision Beta-binomial formula surfaces.
- `beta_binomial_refresh.tsv`: exact two-row receipts for the translated
  historical mean/precision formulas, including prediction/generated
  quantities and pointwise likelihood.
- `student_zip_refresh.tsv`: exact seven-row receipts for the recovered
  Student-t `nu` priors and paired ZIP joint declarations (six direct, one
  inherited-identical-probe), including prediction/generated quantities and
  pointwise likelihood.
- `runtime_controls.tsv`: real-data sleepstudy plus executable BRM-kernel and
  StanBlocks-plate controls.
- `receipts/`: model and dataset retrieval evidence, conservative source
  fidelity verdicts, and reproduction scripts.
- `model_matrix.tsv`: one row per deployed historical model, combining the
  independent fidelity, translation, compiler/runtime, and semantic-mount axes.
- `final_summary_by_source.tsv`: final manual-overridden fidelity and current
  capability counts for each of the 19 historical sources.
- `all_source_rows_matrix.tsv`: the same schema including the one hidden
  non-Bayesian baseline row.
- `PROVENANCE.md`: exact historical/current/SbPMX branch, commit, path, and blob
  provenance.
- `GAP_RANKING.md`: row-keyed leverage ranking, separated into quick defects,
  BRM adapters over existing substrate, true substrate work, and metadata work.
- `gallery/`: executable single-graph `semantic_app` gallery using `@options`
  and `option_domain`/`option_records`; it reads all 359 matrix rows at runtime.
- `REPORT.md`: findings, gaps, and interpretation boundaries.

## Interpretation boundaries

- An HTTP 200 is reachability, not semantic support.
- A generated program passing stanc is not a finite density or a correct model.
- A finite synthetic density is not a validated fit to the historical dataset.
- Family inference is reported beside, never in place of, exact metadata.
- Kernel/plate controls establish route capability only; they are not fanned
  onto historical rows without an exact row-specific implementation.
- The current SbPMX pattern is descriptor-driven, but the BRM-to-HTMXObjects
  mount adapter is not shipped. Mount projections here are experimental.
- `semantic-rewrite` means the current surface exists but a row-specific
  historical constant, prior, or declaration choice is still required; it is
  never displayed as an executed row.
