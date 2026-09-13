# Turing backend

`TuringBRMI` is the direct-BRMI Turing backend. Loading Turing activates the
package extension and turns the backend-neutral plan owned by BRM into a
`DynamicPPL.Model`. Standard Turing inference APIs operate on its `model`
field.

Executable models do not live on this backend-specific page. They are presented
in the backend-neutral [BRM feature atlas](feature-atlas.md), where every example
shows BRM authoring, the emitted StanBlocks model, generated Stan, and the
generated Turing model through the same four-pane comparison.

## Architecture

The extension consumes `BRMI` analysis, materialized data, population designs,
priors, observation semantics, and group-effect plans directly. It does not
construct or inspect `SBBRMI`, `GenerativePlan`, a StanBlocks `SlicModel`, SLIC
IR, or generated Stan. This boundary lets either backend become a weak
dependency without changing BRM semantics.

`TuringBRMI.plan` combines shared preparation with Turing sample-site geometry;
`TuringBRMI.model` is the executable Turing model. Its generated body is available
through `turing_model_source(backend)`. Ordinary callable distributions and
their positional and keyword arguments survive lowering. Adding a likelihood
does not require another model template or an entry in a family list.
For a factory function used as a sampled declaration, define
`brm_distribution_type(::typeof(my_prior)) = Normal` (or its distribution type)
to describe the result shape while retaining the original callable.

The extension generates a body at construction, passes it through DynamicPPL's
model compiler, and stores an immediately callable evaluator. Density and
gradient evaluation do not interpret formula ASTs or use `invokelatest`.
An unsupported operation reports the missing capability. For example, a custom
distribution may supply density evaluation but lack predictive RNG or a latent
support transform; these are separate requirements.

## Parameterization

Population coefficients use the shared design matrix and labels, with Normal
defaults and arbitrary scalar priors selected through `effect(...)`. Sampled
declarations preserve their names, distribution calls, and dependencies, so a
scale can have a hierarchical prior and enter any later likelihood expression.
Simplex, covariance-factor, horseshoe, and R2D2 declarations use their own
parameter geometry. Declaration bounds retain the ordinary prior kernel;
`truncated(distribution; ...)` includes its truncation normalizer.

Group effects default to a noncentered parameterization: plain random intercepts
use a positive scale and standard-normal latent values, correlated slopes use marginal scales
plus an LKJ Cholesky factor, and `||` uses independent scales with no
correlation variable. `centered_groups` selects the corresponding centered
coefficient geometry for supported blocks; Stan-only adaptive/CV sizing controls
remain loud construction errors.

Multiple and crossed grouping factors remain separate blocks. `gr(..., by=)`
uses a separate scale/correlation frame per stratum. Matching `|ID|` terms in a
distributional mean and precision predictor instead share one joint scale
vector, LKJ factor, and group draw; each predictor consumes its own coefficient
slice from that covariance block. Addressed `sd(...)` and `cor(...)`
declarations reuse the same backend-neutral prior resolver as StanBlocks.
Weighted multi-membership intercepts and correlated slopes are supported with
strict all-source replay and resampling. Adaptive geometry remains fail-closed.

Canonical link declarations are lowered once in BRM and reused by the Turing
executor. Response modifiers likewise carry materialized bounds, interval
endpoints, and validation into the extension instead of being rediscovered from
backend code.

## Outputs and replay

`turing_pointwise_loglikelihoods` returns response-named, row-aligned
log-likelihood vectors; latent rows of a partly missing response remain
`missing`. `turing_generated_quantities` evaluates the model's deterministic
return value at one constrained draw. `turing_posterior_predictive` regenerates
every response row at one constrained draw, including rows that were latent in
the fitted model. For a fitted chain, `Turing.predict(backend, chain)` performs
the same response-latent exclusion before running DynamicPPL's chain-level
prediction.

`reprocess(backend, new_data)` rebuilds the direct BRMI plan on new rows while
reusing fitted centers, scales, categorical coordinates, interactions, offsets,
spline bases, HSGP domains and frequencies, and existing group coordinates.
`freeze_constants=false` explicitly refits
those preprocessing constants. Reusing existing groups is the fail-closed
default. `resample_groups=:group` explicitly derives that grouping coordinate
from `new_data`, keeps fitted population, scale, and correlation parameters,
and regenerates only the named groups' standardized effects. For a shared
`|ID|` block this redraw remains joint across its predictors. Unseen categorical
levels still fail closed.

## Parity contract

“Parity” means the same admitted BRMI has the same constrained prior and
likelihood semantics, coefficient addressing, preprocessing behavior, and
observable outputs in both backends. Producing a model that merely samples is
not sufficient.

| BRMI surface | Turing status | Current contract |
| --- | --- | --- |
| Backend boundary | **Supported** | Direct `BRMI` → backend-neutral plan → Turing extension; core loads without Turing |
| Population design | **Supported** | Intercepts, fitted numeric transforms, data expressions, offsets, contrasts, and interactions |
| Priors | **Supported** | Scalar callable priors, sampled hyperparameters, addressed coefficient/group/term priors, bounds, simplex and covariance factors, horseshoe and R2D2 geometry |
| Scalar likelihoods | **Generic** | Retained callable constructors and arguments; canonical logit/log links preserve stable numerical forms |
| Joint and ordinal responses | **Supported** | Vector-valued observations, Multinomial, covariance-factor joint normals, categorical logits, and typed ordinal structure/link composition |
| Group effects | **Partial** | Plain intercepts, correlated and exact-zero-correlation slopes, multiple/crossed factors, distributional cross-predictor `|ID|` covariance, fitted transformed/categorical slopes, stratified `gr(by=)`, explicit centering, `sd`/`cor` prior overrides, and weighted multi-membership intercepts/correlated slopes with replay/resampling; adaptive geometry remains pending |
| Response evidence | **Generic** | Truncated, censored, and interval evidence compose with the base distribution's required CDF/density operations |
| Missing responses | **Supported** | Missing rows use the same conditional family; only observed rows contribute pointwise likelihood |
| Multiple responses | **Supported** | Shared declarations are sampled once and responses can have distinct row axes; incompatible group schemas fail explicitly |
| Observation weights | **Supported subset** | Analytic Normal weights rescale sigma; frequency and power weights scale density while predictive draws retain the base distribution |
| Advanced terms | **Supported subsets** | `s`, `t2`, `mo`/`mo1`, `me`, interval predictors, `ar`/`dar`, exact GP, HSGP, structured fields, and Julia-callable kernel/ragged terms |
| Outputs | **Supported subset** | Row-aligned pointwise likelihoods, deterministic returned quantities, one-draw posterior prediction, and chain-level Turing prediction; fitted response latents are excluded before regeneration |
| Replay | **Supported subset** | Frozen preprocessing and existing-group coordinates replay on new rows; refitting constants and selective new-group resampling—including joint `|ID|` and stratified redraws—are explicit |

The matrix is intentionally an overview. The build-generated examples and
their current unsupported reasons are the executable source of truth.

Periodic HSGP currently uses one isotropic, ungrouped input axis. Latent HSGP
uses one nonperiodic, ungrouped axis with an explicit domain; fixed-input
HSGP additionally supports anisotropic and group-specific bases. A custom term
that supplies only a StanBlocks SLIC emitter keeps that Stan implementation and
needs a native Julia effect or submodel method for Turing. The native backend
does not interpret arbitrary SLIC bodies.

Missing responses currently do not combine with observation weights or response
modifiers. Analytic weights likewise do not combine with response modifiers;
frequency and power weights can compose with them. These combinations fail
explicitly during construction.
