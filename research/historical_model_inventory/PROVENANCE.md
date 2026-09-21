# Provenance ledger

## Historical catalogue

- Source revision: BayesianRegressionModels.jl
  `05c3f465e7987e8d7caa7e214fedddd90415a922` (`origin/main`).
- Recovered paths: `scripts/examples/all.jl` and every
  `scripts/examples/*.jl` method named by its ordered `CATALOG`.
- Source census: 360 rows, of which `epinowcast/bnc_empirical` alone carries
  `hidden: true`; the historical site surface therefore contains 359 models.
- HTML control: snapshot retrieved 2026-07-28 21:00:07 UTC, 1,241,917 bytes,
  SHA-256
  `51e16039b24cb806cff516481ef812fa3c081511981c46c269d3f01ca93600e0`.
  It contains 359 card articles. The sole verified card is
  `<article class="card card-verified" id="bambi-escs" ...>`; an earlier peer
  parser requiring the exact class string `card` counted 358 and omitted it.
- `scripts/CatalogServer/src/CatalogServer.jl` hard-codes both historical
  parseability and sampleability checks to false. Those badges are not prior
  empirical results. Only one source row (`bambi/escs`) claims `verified: true`.
- The renderer substitutes `gaussian` whenever the `family:` field is absent.
  Only 21 deployed rows declare that field. The resulting display distribution
  is 350 Gaussian, four binomial, two Bernoulli, two Poisson, and one
  negative-binomial; the 338 substitutions are recorded as
  `CatalogServer_default_unsubstantiated` rather than evidence.

The old catalogue was removed from the active development line by `a660429`.
No current-site stamp or current compilation result is inferred from it.

Receipt retrieval ran from 2026-07-28 21:07:08 through 21:07:15 UTC with
redirects, a 45-second timeout, and an 8 MB body bound. The row-expanded
first-pass capture has SHA-256
`1ffacae3ea8ed304c9e6a41093a49a349e213ad54d05cfc4945a79335734ffd7`;
the committed LF-normalized TSV has SHA-256
`5268eb4a16cc97afea5cf6409f44f6023a59688f6bb1a703defba9fe04d403cb`.
The untouched 206-URL retrieval ledger has SHA-256
`8a2ca250c48d2adceda9eb424d8db2e0945f29482330c1608dfffa4172827523`;
the committed copy differs only by replacing its capture-directory prefix with
relative `bodies/` paths and normalizing line endings; it has SHA-256
`9de9c1b6940bc193e1b2377575d8ae0064865250ef573a94f684ec0a15cebbf6`.
All 44 deployed partial-support verdicts were then reviewed against the retained
bodies; the override TSV has SHA-256
`657a1f8186c86436a94aff8f8ab0cf4e9adcbd84aa31d87b7b60858d9fe723cd`.

## Current execution environment

- BayesianRegressionModels.jl:
  reviewed clean tree `a707af21d138b0019810f8dce9d655109dc97ff6`,
  landed as canonical `ns/devibe`
  `11031f2d3bbd0c9cad42bed53a4a8dd193ab9d2e`. The validation candidate
  `4d8f565b1cf4b347b3d0e93a0b6499c2561dc42d` has the same Git tree
  (`309028b731f07c9c8dd17a42e74022b66fc23f9e`) as the clean reviewed commit;
  the discarded VBRMI-mutating ancestry is not part of the reviewed or landed
  line.
- The Beta-binomial surface control is pinned to exact reviewed checkpoint
  `98d54fb413e3994eb3e4c9ea76d659cddce433c5`, imported intact onto this lane as
  `457199ddb76703685d98bd3bdc011f419e57f536`. Its combined control covers both
  `BetaBinomial(trials, alpha, beta)` and
  `BetaBinomial2(trials, mean, precision)`; the two historical rows have
  separate direct receipts in `beta_binomial_refresh.tsv`.
- The Student-t/ZIP refresh recovers the four kruschke `nu` priors from ASKurz
  `f90517976b2f52cc9b63c7c5e151e85b1d677e63` (fits 17.5/18.1/18.2/18.3) and
  pairs the split ZIP cards into joint declarations; six direct plus one
  inherited-identical-probe receipt live in `student_zip_refresh.tsv`.
- StanBlocks.jl:
  `329a178a7ad7877da0b58ad2c360d417ddd663f9` for the original exhaustive
  census (local active checkout selected by the canonical resolver). The five
  BinomialLogit rows were subsequently rerun only against exact canonical
  `277f23334fab9f2f88b53cd10f38f5d6bb1118c2`; their focused receipt preserves
  both SHAs and the old failure signature. The separately discovered
  `neg_binomial_2_log` trace defect was fixed after this pinned run in
  StanBlocks canonical `144188a808308177807ceb47f08749a335a0ef70`
  (`negative-binomia-fd397aa0`); the inventory's working non-log NB2 route is
  unaffected and the running exact-SHA census was not restarted.
  The Beta-binomial combined and row-specific controls use exact
  `b77d5b0172970e0eb414f59d746fc2f96b71d363`. The Student-t/ZIP row receipts
  use exact `9607eee6c7e839032f252b91d405007eea6a9021` (the test-env pin).
- WarmupHMC.jl:
  `b185eedbbeeef6fb3327afb30dc995c98591af02` for this rerun. The independent
  peer sleepstudy control used
  `38398527e0406ad31aeaec3efe24d581a18a269e`; its result is preserved as a
  comparison receipt, not represented as this run's package SHA.
- Julia 1.10.11; BridgeStan 2.9.0 at its default
  `~/.bridgestan/bridgestan-2.9.0`; `KB_HOST=strato2`.

The corpus probe records generated Stan and Stan-data hashes. A result is fanned
out only when the normalized current BRM body, declared data columns, grouping
columns, selected family, generated Stan, and data schema are identical.

## Family recovery audit

The 122 deployed rows left unresolved by the first renderer/receipt pass were
audited again on 2026-07-28 against the exact historical BRM source, the cited
project's repository/docs/code (including relevant historical revisions), and
dataset schema or values as supporting but never sole evidence. The row-keyed
ledger is `family_audit.tsv`: 116 explicit family recoveries, five defensible
semantic inferences, and one genuinely indeterminate row (`kruschke:calcium`).
Each row records the searched surfaces, exact revisions and path anchors,
retrieval status/time, positive evidence, dataset corroboration, and negative
evidence. No renderer default or response name was accepted as family evidence.

## SbPMX semantic HTMXO baseline

The selected design baseline is `origin/main` at
`a8fd3c02e1e2305683a56c91384701bfeba6b528`, observed from the remote on
2026-07-28. It is the newest coherent contract, not just the newest timestamp:

- `kb-design/SbPMX-inferrable` at `7fcd50cefc0fe4f4f85ad5bda2b038cd5ad957ee`
  is the merge base and is 79 commits behind main (0 ahead).
- Recorded deployed release
  `88c4fbff71228dfa9fca22504b87f016b76fa361` is three commits behind main
  (0 ahead).
- Main adds `81255eb` (code/target artifacts and disclosure), `10a819c`
  (routes return semantic nodes), and merge `a8fd3c0`.
- Other advertised semantic branches (`postprocessing`, `structure`,
  `synthetic-simulation`, and `v2`) are also contained in main.

Exact reviewed files and main-tree blob ids:

| path | blob |
|---|---|
| `design/ASKS.md` | `463d245aa4f9220611fd17092fb78c05b6bb3f06` |
| `design/SbPMX.jl` | `5889e49aa69386bfcc79515541d626017fe2a674` |
| `design/mock_runtime.jl` | `372f88a9b9f634909c161be858c54119a05bbe7c` |
| `compat/check-semantic-model-pipeline.jl` | `011d5ae50aa07118fdbdcaab186fb2fceae7ed1b` |

The contract uses `semantic_app`, semantic output nodes/cards, `@options`,
`option_domain`/`option_records`, descriptor operations, and ordinary GET
navigation. Retired `@semantic`/`dynamic_domain` vocabulary is not used.
`compat/check-semantic-model-pipeline.jl` is the existing 13-model
BRM→SBBRMI→SLIC→Stan reference gate and requires the complete card option
domain plus format-peer projections.

BRM's `brm_descriptor`/`brm_execute` graph is authoritative for executable
ordinary and kernel models. The adapter from `BRMDescriptor.operations` and
columns to an HTMXObjects `semantic_app` mount is not shipped. Consequently
every mount recommendation in `model_matrix.tsv` is explicitly experimental;
it proposes the current SbPMX semantic pattern rather than claiming an existing
authoritative application graph.

HTMXObjects semantic lock for the recorded SbPMX release:
`3a75633a5d9e50da844b1d553b037560ae1312f2`.

## Served gallery receipt

- Gallery source commit:
  `7b6a091ab6d240ec3fcfbd8f34248d2a78f3373b`; focused BinomialLogit evidence
  artifact `e8f1ab97df5435e0cf5d8b5605f1309ba5b28d7f`.
- Gallery source SHA-256:
  `40934aa56fc86b56684d8a8ebfa903b5117c44f2639f124308f86728d35663e3`.
- Matrix SHA-256:
  `929dbc0a131a8ae9fa108e287c9080574eab4784309c3607d0a219ada9b43ca7`.
- Deployment: `strato2`, `http://127.0.0.1:8129/`; unrelated listeners on
  8127--8128 were left untouched.
- Initial, post-restart, and focused pre-landing root checks returned HTTP 200
  with exactly 359 semantic cards. The focused artifact's confirmed-source,
  finite-BridgeStan, and unsupported-ordinary filters returned 154, 171, and 51
  cards respectively; the original 166-card finite receipt remains in the TSV.
  The refreshed offline matrix now contains 173 finite-BridgeStan cards and 48
  unsupported ordinary rows after adding the two direct Beta-binomial receipts;
  no new served-deployment receipt is inferred from the offline validation.
  After the seven Student-t/ZIP receipts the offline matrix contains 180
  finite-BridgeStan cards and 45 unsupported ordinary rows; again no served
  deployment is inferred.
- Row-by-row request times, paths, statuses, and counts:
  `gallery/served_smoke.tsv`.
