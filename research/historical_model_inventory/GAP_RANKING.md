# Historical catalogue gap ranking

This ranking is deliberately downstream of the current-source and executable-test audit. It does not repeat the historical translator's blocklist as fact. The evidence inputs are `translations.tsv`, `capability_results.tsv`, `brm_surface_audit.tsv`, `stanblocks_surface_audit.tsv`, and `surface_controls.tsv`. The family-surface baseline is canonical BayesianRegressionModels `11031f2d3bbd0c9cad42bed53a4a8dd193ab9d2e`: SBBRMI-only `ZeroInflatedPoisson`, `NegativeBinomial2(mu, phi)`, and `LocationScale(loc, scale, TDist(nu))` dispatch are landed there.

## How the score works

Each genuine implementation gap receives five 0--5 subscores. `R` is deployed-row coverage; `U` is the number of distinct normalized historical formulas or, where a current translation exists, distinct normalized programs; `X` is reuse outside this catalogue; `E` is implementation favourability (5 means small/low-risk, 1 means broad/high-risk); and `D` is dependency readiness (5 means the semantics and substrate are already established). The total is `R + U + X + E + D`, out of 25. Row and program counts are always given separately: two rows with the same formula do not become two independent program receipts.

For reproducibility, `R` bins coverage as 0 = none, 1 = one, 2 = two, 3 = three or four, 4 = five or six, and 5 = seven or more. `U` is capped at five: 0--4 distinct programs score their literal count and five or more score 5. `X`, `E`, and `D` are review judgments whose rationale is stated beside each item. A normalized historical formula collapses whitespace only; it does not treat semantically similar formulas as identical.

`R` counts potential coverage only where the deployed formula was recovered exactly from historical source; it never uses the site's renderer-default `family=gaussian` or a merely reachable citation. When family, parameterization, component pairing, censor code, basis, or structured-process meaning is still unresolved, the same exact keys appear in section D and their lower `D` score makes the coverage explicitly conditional rather than a claimed unlock. "Substrate exists" means the StanBlocks audit found the required primitive/building blocks; it is not a promise that the BRM adapter or generated quantities exist.

## A. Quick defects and completed family surfaces

These are kept separate from new modeling semantics.

### A1. Completed upstream: scalar-trials/vector-logits Binomial generated quantity -- 23/25

- Coverage: 5 rows, 5 distinct current programs (`R=4`, `U=5`).
- Exact rows: `mcelreath:chimpanzees_intercept`, `mcelreath:chimpanzees_slopes`, `mcelreath:moralizing_gods`, `kruschke:recall_conditions`, `kruschke:recall_pooled`.
- Historical failure at StanBlocks `329a178...`: generated-quantity trace inference had no signature for `binomial_logit_rng(::array[] tokenof, ::int, ::vector)`. The density-side model was not the defect.
- Reuse/scope/dependencies: common grouped-Binomial shape (`X=4`), narrow trace/signature repair (`E=5`), no semantic dependency (`D=5`).
- Resolution: StanBlocks canonical `277f23334fab9f2f88b53cd10f38f5d6bb1118c2` resolves [[StanBlocks:snag.binomial-logit-r-2d7c76b2]] (`binomial-logit-r-2d7c76b2`). The focused Strato2 receipt runs exactly the five probe ids through descriptor, stanc, BridgeStan finite density/gradient, prediction/generated quantities, and pointwise likelihood; all five pass. `binomial_logit_refresh.tsv` retains the `329a178...` failure signature rather than overwriting its provenance.

### A2. Remaining: transformed-column interaction lowering -- 17/25

- Coverage: 2 rows, 1 distinct current program (`R=2`, `U=1`).
- Exact rows: `bambi:negative_binomial_interaction`, `bambi:plot_pred_nb`.
- Observed failure: `prog & zscale(math)` reaches SBBRMI as an interaction whose second operand is an `ExprColumn`; current interaction lowering accepts raw-data `NamedColumn`s. This is independent of the now-landed NB2 likelihood.
- Reuse/scope/dependencies: useful for transformed interactions across families (`X=4`); a translator-side derived-column rewrite may be smaller than broadening every interaction operand (`E=5`); semantics are clear (`D=5`).

### A3. Completed upstream and not catalogue-blocking: `neg_binomial_2_log` trace/lpxf

- Direct catalogue increment: 0 rows because the landed non-log `NegativeBinomial2(mu, phi)` route expresses the seven ordinary NB2 rows. Relevance set: 7 rows, 5 distinct current programs (`R=0`, `U=5`).
- Exact relevance rows: `bambi:negative_binomial_main`, `bambi:negative_binomial_interaction`, `bambi:count_roaches_nb`, `bambi:plot_pred_nb`, `brms:distreg_nb`, `rstanarm:roaches_nb`, `glm_jl:quine_nb`.
- Historical audit evidence at StanBlocks `329a178a7ad7877da0b58ad2c360d417ddd663f9`: scalar and vector `neg_binomial_2_log` probes failed because `tracetype` was not defined for the non-GLM lpxf spelling. This is separate from the landed non-log BRM marker/lowering, and the current corpus intentionally still uses that non-log route.
- Resolution: StanBlocks canonical `144188a808308177807ceb47f08749a335a0ef70` (reviewed `74547458`) lands `neg_binomial_2_log_lpmf`, vector `lpmfs`, and scalar/vector RNG; its focused 21/21 tests, stanc, and adjacent checks are green. No catalogue rerun is required for that commit because the audited programs do not call the log family. A stock BRM `NegativeBinomial2Log` marker would still be separate adapter work if desired.

### A4. Completed at `11031f2`: NB2 marker and non-log lowering

- Surface coverage: 7 ordinary rows, 5 distinct current programs.
- Exact rows: `bambi:negative_binomial_main`, `bambi:negative_binomial_interaction`, `bambi:count_roaches_nb`, `bambi:plot_pred_nb`, `brms:distreg_nb`, `rstanarm:roaches_nb`, `glm_jl:quine_nb`.
- The native linked-LHS form is `log(mu) ~ ...`, `log(phi) ~ ...`, followed by `y ~ NegativeBinomial2(mu, phi)`; only `mu` is logged by Stan's `_2_log` family, so a hypothetical log form must not silently log `phi`.
- The combined family-surface control passes stanc, BridgeStan finite density/gradient, predict, and pointwise log-likelihood. Stable direct/inherited row-level evidence is: `bambi:negative_binomial_main`, `bambi:count_roaches_nb`, `brms:distreg_nb`, `rstanarm:roaches_nb`, and `glm_jl:quine_nb` pass stanc plus BridgeStan finite density/gradient (5 rows, 4 distinct programs); `bambi:negative_binomial_interaction` and `bambi:plot_pred_nb` share the 1 distinct program that fails A2. The shared surface control is not fanned onto row-level runtime badges.

### A5. Completed at `11031f2`: `LocationScale(..., TDist(nu))` dispatcher

- Relevance: 5 rows, 5 distinct historical formulas. Four are plain Student-t rows; the fifth uses the executable known-SE-plus-residual composition.
- Plain rows: `bambi:t_regression_t`, `kruschke:guber1999_base`, `kruschke:guber1999_complement`, `kruschke:guber1999_interaction`.
- Composed-SE row: `kruschke:income_famsize`.
- The surface control is green, and the row-varying known-SE plus fitted-residual composition for `kruschke:income_famsize` is also stanc- and BridgeStan-green. All five rows remain semantic rewrites rather than executable historical translations until their authoritative degrees-of-freedom value/prior is recovered.

### A6. Completed at `11031f2`: `ZeroInflatedPoisson` dispatcher

- Historical relevance: 3 rows, 2 distinct historical formulas.
- Exact rows: `bambi:zip_mu`, `bambi:zip_psi`, `bambi:plot_comp_zip`.
- The combined family-surface control passes, but these split catalogue records do not identify an authoritative mean/zero-inflation pairing. They therefore produce 0 row-level normalized programs and remain in D5. This is evidence-honest: the software defect is fixed; the historical source reconstruction is not.

After the focused `277f233...` refresh, the only failed normalized program is A2, shared by its two named rows.

## B. Existing-substrate BRM adapters

These are stock-BRM surface gaps for which the source audit found StanBlocks primitives or a faithful kernel/plate construction path. They are ranked by direct, defensibly recoverable coverage; source-semantic prerequisites are called out explicitly.

| Rank | Gap | Rows | Unique formulas | R | U | X | E | D | Total |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Generic family-aware censoring/truncation adapter | 9 | 9 | 5 | 5 | 5 | 3 | 3 | **21** |
| 2 | Correlated multivariate-outcome adapter | 8 | 8 | 5 | 5 | 5 | 2 | 3 | **20** |
| 3 | Fixed covariance / phylogenetic group adapter | 6 | 6 | 4 | 5 | 5 | 2 | 3 | **19** |
| 4 | Categorical-outcome family and vector predictor | 5 | 5 | 4 | 5 | 5 | 3 | 2 | **19** |
| 5 | Multi-axis/anisotropic HSGP term | 5 | 5 | 4 | 5 | 4 | 2 | 3 | **18** |
| 6 | Row-varying predictor measurement error | 2 | 2 | 3 | 3 | 5 | 2 | 3 | **16** |
| 7 | Multi-membership group term | 2 | 2 | 2 | 2 | 5 | 3 | 4 | **16** |
| 8 | Likelihood/frequency weights | 2 | 2 | 2 | 2 | 5 | 3 | 4 | **16** |
| 9 | Completed: Beta-binomial family adapter | 2 | 2 | 2 | 2 | 4 | 3 | 4 | **15** |
| 10 | Non-proportional/other-link ordinal surface | 4 | 3 | 3 | 3 | 4 | 2 | 2 | **14** |
| 11 | Tensor-product smooth term (`t2`) | 2 | 2 | 2 | 2 | 5 | 2 | 3 | **14** |
| 12 | Hurdle-Poisson family adapter | 2 | 2 | 2 | 2 | 4 | 3 | 3 | **14** |
| 13 | Von Mises outcome adapter | 1 | 1 | 1 | 1 | 3 | 4 | 4 | **13** |
| 14 | Asymmetric-Laplace/quantile family adapter | 1 immediate | 1 immediate | 1 | 1 | 4 | 3 | 3 | **12** |

The numerical total is a prioritization aid, not a substitute for dependency order. The `B1`--`B14` section identifiers follow the audit taxonomy; the table's `Rank` column is the score order. For example, B9 and B10 outscore B6 because their semantics and substrate path are clearer despite covering fewer rows. Exact row receipts follow.

The corrected inferred matrix's 26 rows classified `genuinely-missing-brm-surface` are exactly the row lists in B4, B5, B6, B7, B8, B9, B10, B11, the immediate row in B12, and B13. B1--B3 and B12's basis-dependent follow-ons are conditional leverage: their formulas are recovered, but the current matrix correctly keeps their historical semantics unresolved until the D-section prerequisites are settled.

The reuse/risk judgments are: B1 is reusable across survival and bounded-outcome families but needs family-specific normalizers and censor-code validation; B2 applies to multivariate regression generally but has high covariance/GQ risk; B3 generalizes to relationship-matrix random effects but needs robust level/matrix alignment; B4 is a broadly reusable nominal-response family with a new vector-predictor/GQ contract; B5 is reusable for spatial GP models but requires new multi-axis basis construction; B6 serves ordinal regression beyond proportional odds but is family/link-specific; B7 serves measurement-error models broadly but changes the latent-data and residual-scale contract; B8 is reusable across GAMs but basis construction is substantial; B9 and B10 are general-purpose terms with a clear plate/density substrate; B11, B12, B13 and B14 are bounded family adapters whose main risks are parameterization plus RNG/pointwise consistency.

### B1. Generic family-aware censoring/truncation adapter

Exact rows: `bambi:survival_intercept`, `bambi:survival_color`, `bambi:survival_cont_weibull`, `bambi:survival_cont_retention_fe`, `bambi:survival_cont_retention_re`, `kruschke:censored_lr`, `kruschke:censored_interval`, `burkner_papers:epilepsy_trunc`, `burkner_papers:kidney`.

The plate/kernel control proves a construction route, not these nine likelihood normalizers. Censor-code and interval-bound semantics must be recovered before implementing each branch. Truncation is mathematically distinct from the existing censored-LOQ helper.

### B2. Correlated multivariate-outcome adapter

Exact rows: `brms:btdata_compact`, `brms:btdata_explicit`, `brms:btdata_spline`, `mcelreath:mcelreath_btdata_intercept`, `mcelreath:mcelreath_btdata_full`, `mcelreath:mcelreath_btdata_interaction`, `mcmcglmm:btdata_bivariate`, `mcmcglmm:pbcseq_bivariate`.

Independent multi-response models are already expressible. This gap is specifically residual covariance/outcome packing, covariance parameterization, and generated quantities; replacing `set_rescor(TRUE)` or `us(trait):units` with independent Normals would change the model.

### B3. Fixed covariance / phylogenetic group adapter

Exact rows: `brms:phylo_simple_re`, `brms:phylo_simple_resid`, `brms:phylo_repeat_re`, `brms:phylo_effect`, `brms:phylo_pois`, `mcelreath:primates301_phylo`.

StanBlocks covariance/Cholesky primitives exist. BRM still needs the data-matrix, level-mapping, and group-effect adapter; `gr(..., cov=A)`/`fcor(R)` is not the current `gr(...; by=...)` surface.

### B4. Categorical outcome

Exact rows: `bambi:categorical_toy`, `bambi:categorical_iris`, `bambi:categorical_alligator`, `kruschke:softmax_categorical`, `kruschke:softmax_baseline`.

The missing layer is a categorical/categorical-logit marker, vector-valued predictor and generated-quantity lowering. StanBlocks already has the likelihood primitive. This is broadly reusable for nominal-outcome regression.

### B5. Multi-axis/anisotropic HSGP

Exact rows: `bambi:hsgp_2d_iso`, `bambi:hsgp_2d_by_group`, `bambi:hsgp_2d_nocov`, `bambi:hsgp_2d_aniso`, `bambi:hsgp_2d_poisson`.

Current `gp(...)` is a one-input surface; passing two axes raises. Basis construction and covariance sharing belong in BRM, while matrix/GP primitives exist below. The separate historical meaning of `centered` and `share_cov=false` remains D7.

### B6. Non-proportional/other-link ordinal models

Exact rows: `kruschke:condlog1_ordinal`, `kruschke:condlog2_ordinal`, `kruschke:ordinal_probit_disc`, `burkner_papers:inhaler`.

There are 4 rows but only 3 distinct formulas because the two `condlog` rows share `Y_ord ~ 1 + cs(X1) + cs(X2)`. Stock proportional-odds `OrderedLogistic` is already expressible; category-specific effects, discrimination, and other links need their own surface. Do not merge this implementation item with the evidence-only ordinal set in D3.

### B7. Measurement error and response uncertainty

Exact rows: `mcelreath:waffle_divorce_meas_err`, `vasishth:indiv_diff_me`.

Constant-scalar predictor `me`, known-SE Normal response rewriting, and the Student-t residual-plus-known-SE composition already work. These two rows still require predictor-side/row-varying uncertainty. The adapter must not silently replace posterior uncertainty with a fixed scale.

### B8. Tensor-product smooth

Exact rows: `burkner_papers:rent99_spline`, `burkner_papers:rent99_distr`.

Both use distinct `t2(area, yearc)` programs; linked distributional predictors already exist, so the second row does not require new declaration infrastructure. It depends on one faithful tensor-basis term. More complex `te`/dynamic mvgam rows remain structured-model reconstructions in D1.

### B9. Multi-membership

Exact rows: `burkner_papers:multi_member_equal`, `burkner_papers:multi_member_weighted`.

The StanBlocks plate substrate exists; stock BRM has no `mm` emitter. Implement equal membership first, then weight normalization and data validation for the weighted form.

### B10. Likelihood/frequency weights

Exact rows: `epidist:delay_naive`, `epidist:delay_marginal`.

This needs explicit power/frequency-likelihood semantics. Known response SE is a different supported rewrite; using it here would be wrong. The second row additionally carries `vreal(...)` payload semantics, so the first is the lower-risk initial target.

### B11. Hurdle Poisson

Exact rows: `bambi:hurdle_mu`, `bambi:hurdle_psi`.

The two split components are distinct formulas. A reviewed scalar density can be built on the existing substrate, but component pairing and RNG/log-likelihood behavior must be specified; this must not alias ZIP.

### B12. Asymmetric Laplace / quantile regression

Immediate row: `burkner_papers:hetero_jss_quantile` (1 row, 1 formula). Basis-dependent follow-ons: `bambi:quantile_p10`, `bambi:quantile_p50`, `bambi:quantile_p90` (3 rows, 1 shared formula).

The immediate score excludes the three Bambi rows because their `bs(...)` basis identity is unresolved (D2). The family adapter should make quantile an explicit parameter and cover generated quantities; it must not infer quantile from a title.

### B13. Von Mises outcome

Exact row: `bambi:circular_vonmises`.

Stan has the density; the missing work is BRM argument mapping, constraints, RNG and pointwise log-likelihood. This is one catalogue row but reusable for circular-response models and for later BMM kernels.

### B14. Completed at `98d54fb`: Beta-binomial outcome

Exact rows: `brms:cbpp_beta_binomial`, `mcelreath:ucbadmit_beta_binomial` (2 rows, 2 formulas).

The earlier translator false-positive is now closed by two explicit surfaces: Julia-native `BetaBinomial(trials, alpha, beta)` and brms-aligned `BetaBinomial2(trials, mean, precision)`. The exact `98d54fb` control passes stanc, finite BridgeStan density/gradient, prediction/generated quantities, and pointwise likelihood for both. Focused receipts separately run the two historical `vint(...)` formulas through the mean/precision surface; both pass all stages and appear as finite-BridgeStan gallery rows.

## C. True StanBlocks substrate gaps

The corrected inferred matrix classifies **0 deployed rows** as presently blocked solely by StanBlocks substrate. That zero has an exact row list: none. One narrow substrate shape remains; both concrete trace defects found by the audit have landed upstream.

1. **Completed: `neg_binomial_2_log` lpxf trace support.** Direct increment: 0 rows; relevance is the same 7 rows/5 programs listed in A3. The audit's failure is an exact receipt for StanBlocks `329a178a`, not current canonical: the repair landed at `144188a8`. Non-log NB2 at BRM `11031f2` remains the corpus route, and a BRM log-family marker is not implied by the substrate repair.
2. **Remaining: ragged plate in-cell outcome masking.** Direct deployed keys: none; distinct catalogue programs: 0. Separate-column ragged/multi-output plates are green, but the audited masked-within-cell form is unavailable. Reuse is high for PK/PD and structured multi-output kernels (`X=5`), while implementation and shape risk are high (`E=1`) and a concrete model contract is prerequisite (`D=2`), so its score is 8/25. It belongs after a real consumer supplies the exact mask and generated-quantity requirements.

3. **Completed: scalar-trials/vector-logits Binomial RNG trace support.** Direct increment: the five A1 rows and five programs. The old receipt is exact for StanBlocks `329a178...`; the focused all-stage pass is exact for canonical `277f233...`.

## D. Metadata and source-semantics work

These items must be resolved before an implementation score can honestly call the affected row "unlocked".

### D1. Route-specific structured models -- 64 rows, 64 formulas

These rows have a separately green kernel/plate control, but no ordinary-formula substitute and no per-row execution receipt. They are not one StanBlocks gap. Exact rows, grouped only for readability:

- BRM/multivariate/phylogenetic (8): `brms:btdata_compact`, `brms:btdata_explicit`, `brms:btdata_spline`, `brms:phylo_simple_re`, `brms:phylo_simple_resid`, `brms:phylo_repeat_re`, `brms:phylo_effect`, `brms:phylo_pois`.
- Bambi survival (5): `bambi:survival_intercept`, `bambi:survival_color`, `bambi:survival_cont_weibull`, `bambi:survival_cont_retention_fe`, `bambi:survival_cont_retention_re`.
- McElreath multivariate/phylogenetic (4): `mcelreath:primates301_phylo`, `mcelreath:mcelreath_btdata_intercept`, `mcelreath:mcelreath_btdata_full`, `mcelreath:mcelreath_btdata_interaction`.
- Kruschke censoring (2): `kruschke:censored_lr`, `kruschke:censored_interval`.
- Burkner truncation/survival (2): `burkner_papers:epilepsy_trunc`, `burkner_papers:kidney`.
- Action models (6): `action_models:rw_jget_lr`, `action_models:rw_jget_noise`, `action_models:pvl_igt_lr`, `action_models:pvl_igt_reward`, `action_models:pvl_igt_loss`, `action_models:pvl_igt_noise`.
- Epinowcast (6): `epinowcast:enw_basic`, `epinowcast:enw_report_dow`, `epinowcast:enw_np_reference`, `epinowcast:enw_age_reference`, `epinowcast:enw_age_week_reference`, `epinowcast:enw_rt_renewal`.
- BMM (4): `bmm:mixture2p_setsize`, `bmm:mixture3p_setsize`, `bmm:sdm_condition`, `bmm:imm_full_condition`.
- Flocker (6): `flocker:single_season_repvarying`, `flocker:single_season_repconstant`, `flocker:colex_explicit`, `flocker:colex_equilibrium`, `flocker:autologistic_equilibrium`, `flocker:augmented_multispecies`.
- Mvgam (6): `mvgam:portal_glm_re`, `mvgam:portal_ar_ndvi`, `mvgam:portal_shared_trend`, `mvgam:plankton_var`, `mvgam:salmon_beta_ar`, `mvgam:nmix_detection`.
- INLA (10): `inla:penicillin_iid`, `inla:sleepstudy_iid`, `inla:sleepstudy_slopes`, `inla:airp_rw1`, `inla:airp_rw2`, `inla:airp_ar1`, `inla:airp_seasonal`, `inla:lidar_rw2`, `inla:surg_binomial`, `inla:boston_besag`.
- MCMCglmm (3): `mcmcglmm:btdata_animal`, `mcmcglmm:btdata_bivariate`, `mcmcglmm:pbcseq_bivariate`.
- glmmTMB (2): `glmmtmb:ar1_covar`, `glmmtmb:volcano_spatial`.

The two multi-membership rows are excluded from this 64-row evidence backlog and scored in B9: `burkner_papers:multi_member_equal`, `burkner_papers:multi_member_weighted`. Together they account for all 66 `route-specific` translations.

### D2. Historical `bs(...)` basis identity -- 8 rows, 5 formulas

Exact rows: `bambi:cherry_blossoms_explicit`, `bambi:cherry_blossoms_absorbed`, `bambi:quantile_p10`, `bambi:quantile_p50`, `bambi:quantile_p90`, `bambi:quantile_gaussian`, `bambi:distributional_bikes`, `bambi:survival_disc_spline`.

Current `s(...)` is not a mechanical alias for historical `bs` knots, intercept, degree, or basis identity. Recover the design matrix/basis contract first. The three quantile rows share one formula, accounting for the difference between 8 rows and 5 formulas.

### D3. Ordinal link/threshold semantics -- 7 rows, 7 formulas

Exact rows: `bambi:ordinal_hr_years`, `mcelreath:trolley_intercept`, `mcelreath:trolley_effects`, `mcelreath:trolley_edu`, `kruschke:ordinal_probit_intercept`, `kruschke:happiness_assets`, `kruschke:movies`.

Recover link, threshold, discrimination and monotonic-predictor conventions. Stock proportional-odds `OrderedLogistic` passing is not evidence for these rows. The four rows needing known non-stock surfaces are separately scored in B6.

### D4. Student-t degrees of freedom and prior -- 5 rows, 5 programs, 4 closed

Exact rows: `bambi:t_regression_t`, `kruschke:income_famsize`, `kruschke:guber1999_base`, `kruschke:guber1999_complement`, `kruschke:guber1999_interaction`.

The four kruschke rows are closed: ASKurz `f90517976` fits 17.5/18.1/18.2/18.3 estimate `nu` with a brms `exponential(rate=1/29)` prior, rendered as Julia `nu ~ Exponential(29)` with matching `effect()` coefficient priors, and all four pass the full descriptor/stanc/BridgeStan/predict/pointwise gate. Only `bambi:t_regression_t` keeps a symbolic `nu`: the `LocationScale` dispatcher is executable, but an arbitrary `nu` inserted by the translator would change the historical model, so its authoritative bambi value is still required before direct row execution.

### D5. Closed: split ZIP component pairing -- 3 rows, 2 programs

Exact rows: `bambi:zip_mu`, `bambi:zip_psi`, `bambi:plot_comp_zip`.

Historical scripts prove `zip_mu` (count submodel) and `zip_psi` (psi submodel) are two halves of one joint bambi model on the fish data; both cards now carry the joint `log(lambda)`/`logit(zi)` declaration and share one probe, with BRM `zi` = 1 − bambi `psi` (sign-flipped submodel coefficients, verified in `docs/examples/zip.jl` model B). `plot_comp_zip` is the standalone count form with intercept-only `logit(zi) ~ 1`. All three pass the full gate; receipts are in `student_zip_refresh.tsv`.

### D6. Binomial trial denominator -- 3 rows, 3 formulas

Exact rows: `lme4:verbagg_crossed`, `lme4:contraception_urban`, `glm_jl:pima_logistic`.

The response name or a binary-looking dataset is supporting evidence, not authority. Recover Bernoulli versus grouped-Binomial semantics and, if grouped, the exact denominator.

### D7. HSGP option identity -- 2 rows, 2 formulas

Exact rows: `bambi:hsgp_1d_centered`, `bambi:hsgp_1d_nocov`.

One-dimensional and grouped current `gp` controls pass. That does not establish that historical `centered=True` or `share_cov=False` is equivalent to the current centering/covariance-sharing contract.

### D8. Wald/inverse-Gaussian parameterization -- 1 row, 1 formula

Exact row: `bambi:wald_gamma_wald`.

Resolve the authoritative distribution and parameterization before choosing a BRM marker or Stan density. Name similarity is not evidence that Gamma, inverse Gaussian, Wald and Wiener forms are interchangeable.

### D9. Genuinely indeterminate family -- 1 row, 1 formula

Exact row: `kruschke:calcium`.

This is the only family left genuinely indeterminate after the 122-row family audit. No software-gap score is defensible until primary source or equivalent authoritative historical code establishes the response distribution.

## Recommended dependency order

1. Preserve the completed A1 receipt and implement the translator-sized A2; rerun only A2's exact probe id plus generated quantities and pointwise log-likelihood.
2. Preserve the completed `11031f2` family controls and the closed D5 / near-closed D4 row receipts (`bambi:t_regression_t` is the one remaining symbolic-`nu` row).
3. Preserve the completed B14 receipts, then implement B9/B10 before the larger structured-model program: they have clear semantics, existing substrate and broad reuse.
4. Implement B4 and B5 as focused reusable terms/families; neither requires a new declaration graph.
5. Recover D2/D3 semantics before building smooth/ordinal surfaces, then implement B6/B8/B12 against those recovered contracts.
6. Treat B1--B3 and D1 as explicit kernel/plate projects with per-row generated-quantity and BridgeStan gates. Do not fan one analogue's badge across them.
7. Add a BRM log-NB2 marker only when a consumer needs it; the StanBlocks substrate is now green. Undertake ragged in-cell masking only with a concrete consumer and shape contract.
