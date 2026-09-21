# Historical BRM site model inventory

## Outcome

This is the first empirical capability census of the removed historical BRM
catalogue. The old site did not test models: its parse/sample badges were
hard-coded placeholders. The recovered source contains 359 deployed model rows
plus one hidden non-Bayesian baseline.

The inventory keeps four questions separate for every deployed row:

1. What did the catalogue claim, exactly?
2. Does its cited source support that claim?
3. Is there a semantics-preserving translation to current BRM, a BRM kernel,
   or a StanBlocks plate?
4. How far did that exact translation run today?

Passing question 4 never repairs a failure in question 2. Likewise, an
authoritative citation does not imply current compiler support.

## Source and receipt fidelity

The 359 deployed rows contain 344 HTTP model-citation occurrences, 15
synthetic/uncited occurrences, and 125 distinct row-level HTTP URLs. Retrieval
on 2026-07-28 produced 121 HTTP 200 responses and four HTTP 404 responses. The
four dead URLs expand to 17 affected rows. Dataset receipts are a separate
surface and never validate a model formula or family.

Initial conservative row verdicts before the manual second pass were:

| verdict | rows |
|---|---:|
| confirmed by exact normalized formula text | 142 |
| adapted-but-defensible candidate | 44 |
| mismatch | 4 |
| unverifiable | 152 |
| dead-source | 17 |

The four direct mismatches are ActionModels
`pvl_igt_{lr,reward,loss,noise}`. Their cited code loads healthy controls and
declares independent population priors; it does not regress those parameters
on `clinical_group` as the catalogue claims. The matrix carries the correction
without pretending a different source was originally cited.

Exact formula occurrences are strong receipts. The 44 partial/adapted cases
received an additional manual body review rather than trusting token overlap:
12 were confirmed, 10 remained adapted-but-defensible, 21 became mismatches,
and one became unverifiable. Final deployed-row totals are therefore:

| verdict | rows |
|---|---:|
| confirmed | 154 |
| adapted-but-defensible | 10 |
| mismatch | 25 |
| unverifiable | 153 |
| dead-source | 17 |

The additional mismatches are substantive, not punctuation differences. They
include omitted or invented BRMS terms, four Epinowcast module compositions,
seven INLA dataset/formula substitutions, two mvgam portal state models, and
two rstanarm gasoline beta models. Every correction and bounded body anchor is
row-keyed in `receipts/manual_review.tsv` and expanded into the matrix.

## Family metadata defect

Only 21 deployed rows explicitly declare `family:`. The historical renderer
silently labels every absent field Gaussian, producing 350 Gaussian, four
binomial, two Bernoulli, two Poisson, and one negative-binomial card. This is a
real presentation defect, not a lowering convention.

The matrix therefore carries three independent fields:

- exact family metadata (possibly empty),
- the renderer claim and its `CatalogServer_default_unsubstantiated`
  provenance, and
- a separately inferred semantic family with its evidence class.

Exact-metadata and inferred-family translations were validated separately.
WarmupHMC's independent HTML pass found 33 formulas that visibly contradict a
Gaussian badge; that is a useful strict lower bound. The initial conservative
pass still left 122 deployed rows unresolved. Every one of those 122 was then
audited across the exact historical Git source, the cited project's
authoritative repository/docs/code and relevant historical revisions, and
dataset schema or values as supporting—but never sole—evidence. That second
pass recovered 116 explicit families and five defensible semantic inferences.
Only `kruschke:calcium` remains genuinely indeterminate: the historical row and
matching authoritative code supply no unique likelihood, its prose names two
incompatible alternatives, and the cited live dataset does not match the
claimed schema. `family_audit.tsv` records searched surfaces and negative
evidence for all 122 rows. The matrix preserves the renderer claim, exact
metadata, recovered family, recovery class, and evidence independently.

The defect is also an operational selection hazard, not only a display issue.
An independent WarmupHMC catalogue benchmark found that tag-based Gaussian
selection would incorrectly admit `cbpp` (binomial), `epilepsy` (count), and
`kidney` (censored survival), while explicitly tagged `contraception` and
`verbagg` remain correct. That independently localizes the defect to the
missing-metadata default rather than to every explicit tag.

The sole verified card, `bambi/escs`, also exposed a peer parser edge: its HTML
class is `card card-verified`, so an exact `class="card"` matcher counted 358.
A class-token matcher and the source both count 359.

## Translation to current semantics

The translation deliberately follows current BRM rather than mechanically
modernizing old brms syntax:

- `I(expr)` → `protect(expr)`;
- `scale(x)` → `zscale(x)`;
- simple source `a:b` → current raw-column `a & b`;
- simple source `a*b` → `a + b + (a & b)`;
- `S(x)`/`C(x)` → raw `x` plus an integer/categorical dtype requirement;
- source `||`/`zerocorr(...)` → current `||`;
- grouped/binomial responses → explicit `BinomialLogit`;
- every accepted likelihood is explicit, including its link and
  parameterization.

Transform interactions, chained interactions, composite group identifiers,
multi-formula models, and rich historical spline/HSGP signatures are not
papered over. They remain explicit gaps.

Route classification over the 359 rows is:

| route | rows | meaning |
|---|---:|---|
| `ordinary_brm` | 293 | ordinary current descriptor candidate |
| `brm_kernel` | 19 | faithful group-local/ragged/censoring cell requires a row-specific kernel |
| `stanblocks_plate` | 47 | structured latent/covariance/time-series cell requires a plate |

In the inferred-family variant, 175 rows are direct executable candidates, 69
are semantics-preserving rewrites that still require a row-specific historical
constant/prior/declaration choice, 66 are route-specific kernel/plate designs,
48 are unsupported, and one is family-unresolved. The independent surface audit
classifies 116 rows as expressible verbatim, 128 through a semantic rewrite, 26
as genuinely missing a BRM surface, none as a proven StanBlocks substrate gap
at the row level, and 89 as historically unresolved. These categories are
exhaustive and row-keyed in `model_matrix.tsv`.

This correction matters: the first translator blocklist was not trustworthy.
Current BRM already supports native linked predictors, independent
multi-response declarations, scalar `me`, Normal response `mi`, known Gaussian
response-SE rewrites, smooths, one-dimensional/grouped GPs, AR(1), categorical
predictors/interactions, `gr(; by/id)`, derived interaction/group/response
columns, and proportional-odds `OrderedLogistic`. Those are no longer reported
as absent. The remaining gaps are narrower and are ranked with exact row keys,
scope, dependency order, and reuse in `GAP_RANKING.md`.

## Current compiler and runtime evidence

The re-audit also produced and landed three quick SBBRMI-only family repairs at
canonical `11031f2d3bbd0c9cad42bed53a4a8dd193ab9d2e`:

- `ZeroInflatedPoisson` now has real likelihood/RNG dispatch;
- `NegativeBinomial2(mu, phi)` lowers to Stan's non-log
  `neg_binomial_2(mu, phi)` while native `log(mu) ~ ...` and
  `log(phi) ~ ...` linked predictors remain authoritative;
- `LocationScale(loc, scale, TDist(nu))` lowers as a Student-t likelihood.

The accepted control uses native linked-LHS syntax and passes stanc, BridgeStan
finite density/gradient, prediction, and pointwise log likelihood for all three
families. It was run at validation tip `4d8f565...`; that tip and clean reviewed
commit `a707af2...` have identical Git tree `309028b...`, and only the clean
commit was landed. No VBRMI change survived. This shared control is displayed
beside applicable rows but is never inherited as row validation. The three
historical ZIP cards gained paired joint declarations and the four kruschke
Student-t rows gained sourced `nu ~ Exponential(29)` priors (see the focused
receipt below); only `bambi:t_regression_t` still needs a sourced
degrees-of-freedom value. The working census uses the non-log route. The separate
`neg_binomial_2_log` trace defect found during the audit was subsequently fixed
in StanBlocks canonical `144188a808308177807ceb47f08749a335a0ef70`
(`negative-binomia-fd397aa0`); the corpus run remains pinned to `329a178...`
because its non-log path is unaffected.

The 182 executable inferred-family rows and 16 executable exact-metadata rows
collapse to 167 unique normalized bodies/data schemas. The static gate runs
BRMI evaluation, SBBRMI lowering, `brm_descriptor`,
`brm_execute(:transpile)`, and stanc. On the exact landed tree plus the focused
StanBlocks `277f233...`, Beta-binomial `98d54fb...`, and Student-t/ZIP refreshes
described below,
166 unique programs pass stanc. Every one of those 166 also instantiates under BridgeStan
2.9.0 and has finite log density and gradient at the synthetic-data zero point;
there are no post-stanc runtime failures. Byte-identical evidence fan-out maps
those programs onto 180 deployed rows. The only remaining executable failure is
one transformed-interaction program shared by two deployed rows. These results
are capability evidence, not posterior-correctness claims.

At the historical StanBlocks pin `329a178...`, five scalar-trials BinomialLogit
translations failed descriptor generation at
`binomial_logit_rng(::array[] tokenof, ::int, ::vector)`:
`mcelreath/{chimpanzees_intercept,chimpanzees_slopes,moralizing_gods}` and
`kruschke/{recall_conditions,recall_pooled}`. StanBlocks canonical
`277f23334fab9f2f88b53cd10f38f5d6bb1118c2` resolves that snag. A focused
Strato2 rerun of exactly those five probe ids now passes descriptor, stanc,
BridgeStan finite density/gradient, prediction/generated quantities, and
pointwise log likelihood. The focused receipt retains the old SHA and failure
signature in `binomial_logit_refresh.tsv`. That rerun also corrected a probe-data
bug: scalar one-trial rows must use outcomes in `0:1`, not the generic
multi-trial value `2`.

The sole remaining failed program,
`bambi:negative_binomial_interaction`, correctly fails earlier because current
BRM interactions require raw operands and do not accept `prog & zscale(math)`.

The exact Beta-binomial checkpoint `98d54fb...` passes a combined control for
both the shapes surface and the brms-aligned mean/precision surface, including
finite BridgeStan density/gradient, prediction/generated quantities, and
pointwise likelihood. Focused direct receipts then run
`brms:cbpp_beta_binomial` and `mcelreath:ucbadmit_beta_binomial` separately;
both pass all stages with 12 finite prediction and pointwise values.

The Student-t/ZIP refresh recovers the four kruschke Student-t rows' authoritative
`nu` prior from ASKurz `f90517976` (brms `exponential(rate=1/29)` renders as Julia
`Exponential(29)`; fits 17.5/18.1/18.2/18.3) with matching `effect()` coefficient
priors, and pairs the split `bambi:zip_mu`/`bambi:zip_psi` cards into one joint
`log(lambda)`/`logit(zi)` declaration (BRM `zi` = 1 − bambi `psi`, per the
`docs/examples/zip.jl` replication) plus the standalone `bambi:plot_comp_zip`
form. Six direct receipts pass descriptor, stanc, BridgeStan finite
density/gradient, 12 finite prediction and 12 finite pointwise values each; the
psi-submodel card inherits its partner's identical probe. Full per-row evidence
is in `student_zip_refresh.tsv`.

An earlier probe version bound vector-valued Beta shape expressions to
intermediate names and exposed a missing `lpxf_expr`. Re-checking the native
current idiom showed that keeping those expressions inline is accepted; the
translator was corrected and all six Beta rows now pass static validation.

These are not historical-source verdicts and are not extrapolated to merely
similar rows. Full BridgeStan and sampling results are recorded in
`capability_results.tsv`, `runtime_controls.tsv`, and the per-row matrix.

## Real-data and structured-route controls

The real-data control uses the historical sleepstudy model on the official
Rdatasets `lme4/sleepstudy` data (180 rows, 18 subjects). For direct comparison
with the peer control, `Reaction` is divided by 100 and raw subject ids are
densely recoded. The scaling is a capability-test choice, not a catalogue
claim or a validated historical fit.

On the clean reviewed BRM tree `a707af21d138b0019810f8dce9d655109dc97ff6`
(landed as canonical `11031f2d3bbd0c9cad42bed53a4a8dd193ab9d2e`), StanBlocks
`329a178a7ad7877da0b58ad2c360d417ddd663f9`, and current-host WarmupHMC
`b185eedbbeeef6fb3327afb30dc995c98591af02`, that control reproduces the peer
dimension (42) and zero-point log density (`-832.3603659550055`), has a finite
gradient, and completes 100 WarmupHMC draws with zero divergences. The peer's
stronger sampling receipt—500 draws across 12 seeds with zero divergences—used
WarmupHMC `38398527e0406ad31aeaec3efe24d581a18a269e` and remains a separate
control rather than evidence inherited by catalog rows.

Two additional controls are kept deliberately orthogonal to the row evidence:

- a BRM `kernel(...) do ... end` ragged group-local model derived from
  `test/descriptor.jl`;
- a fixed-width correlated StanBlocks `plate` model derived from
  `test/plate_stress.jl`.

They pass or fail as route controls only. No historical row inherits them
without an exact row-specific implementation. Both controls pass stanc,
BridgeStan instantiation, finite density/gradient, and WarmupHMC sampling: the
kernel is dimension 15 with 50 draws and zero divergences; the plate is
dimension 21 with 50 draws and zero divergences.

## Current SbPMX semantic mount

The coherent baseline is SbPMX `origin/main@a8fd3c0`, which contains every
advertised semantic branch and is three commits beyond the recorded deployed
release `88c4fbf`. Exact paths and blob ids are in `PROVENANCE.md`.

The translation target is not old page syntax. It is the descriptor-driven
pattern exercised by current `design/SbPMX.jl` and
`compat/check-semantic-model-pipeline.jl`:

- one `semantic_app` graph;
- `@options` backed by `option_domain`/`option_records`;
- semantic cards/nodes with HTML, Markdown, Hyperview, and text peers;
- descriptor operations projected as actions/forms;
- ordinary navigation URLs, artifacts, and disclosures.

`@semantic` and `dynamic_domain` are retired. The primer-recorded
`APPDATA.built(:entry)` 13-card finite density/gradient gate, reflected in the
current semantic pipeline compatibility check, is a reference control—not
evidence for these 359 translations.

BRM's descriptor is authoritative, but the adapter from
`BRMDescriptor.operations + columns` to the HTMXObjects mount is not shipped.
Every matrix mount recommendation is therefore marked experimental. A future
adapter should consume the descriptor; it should not re-parse formula text or
create a second declaration graph.

The updated gallery is executable rather than a mock. One mounted
`HistoricalInventory` graph owns the 359 runtime matrix rows, all four filter
parameters, their `@options` domains, filtered rows, semantic cards, and the GET
operation. Its thin page host calls `semantic_app` on that graph; it has no
manual route/form mirror, `AppData`/`AppContext`, or DynamicObjects/HTMXObjects
shadow model. Cards expose source fidelity, family provenance, translation
route, surface class, and the actual audited validation tier. Unsupported and
unresolved rows render `SemanticUnavailable`; historical badges are never
treated as current receipts.

The gallery source at `7b6a091ab6d240ec3fcfbd8f34248d2a78f3373b`
and focused BinomialLogit artifact at
`e8f1ab97df5435e0cf5d8b5605f1309ba5b28d7f` were served on Strato2 at
`http://127.0.0.1:8129/` (ports 8127--8128 were occupied by unrelated
deployments and were not touched). Root checks returned HTTP 200 with exactly
359 cards. The focused filters returned 154 confirmed-source cards, 171
finite-BridgeStan cards, and 51 ordinary unsupported cards, matching the matrix
at that receipt. The current offline matrix increases the finite count to 180
and reduces unsupported rows to 45 after the known-SE Student-t correction,
the two Beta-binomial receipts, and the seven Student-t/ZIP receipts. The listener contained one
`htmxo-semantic-app` and all four option
controls. Exact UTC times, source/matrix hashes, paths, statuses, and counts are
in `gallery/served_smoke.tsv`; the final landed listener receipt is also filed
durably after broker integration.

## What remains genuinely open

- Resolve or source-correct the one genuinely indeterminate family row,
  `kruschke:calcium`, and preserve the 121 recovered family receipts.
- Implement row-specific kernel/plate translations for the 66 structured-route
  candidates; representative substrate controls are not substitutes.
- Resolve the transformed-interaction failure and row-specific historical
  choices for the remaining Student-t degrees of freedom (`bambi:t_regression_t`
  only), ordinal links, binomial trial sizes, and old spline/GP configuration.
- Implement the genuinely missing BRM adapters and route declarations ranked in
  `GAP_RANKING.md`; do not convert route controls into row evidence.
- Ship the BRMDescriptor→semantic_app adapter, keeping the descriptor as the
  single authoritative graph.
