# Running the tests

The files in this directory are standalone scripts, not a `Pkg.test` suite.
Run any one of them with:

```sh
julia --project=test test/adaptive_centering_bridgestan.jl
```

`test/Project.toml` is a superset of the root project: it carries every package
that any `test/*.jl` loads, so no test file fails at its own `using` line the way
three of them do under `--project=.` (see below).

One deliberate absence, and it is a standing rule rather than a gap:
**`ForwardDiff` is not in this environment and is not coming back.** BRM
differentiates with Enzyme only — every gradient in this suite goes through
`AutoEnzyme()` via `DifferentiationInterface`. No test file loads ForwardDiff,
so there is nothing here to work around; do not add it back to make a new
gradient site easier.

`kernel_ragged_broadcast.jl` checks Stan source emission and stanc acceptance
for modeled secondary-axis arguments in plain and nested dotted calls. It
covers interleaved rows, shared random effects, empty groups, raw-data controls
and preservation of field and keyword names. Run it with
`julia --project=. test/kernel_ragged_broadcast.jl` or the test environment above.

`plate_grouping.jl` checks model-scope `rows = groups(group => labels)`
bindings that `@plate for` cells read as `x[rows[i]]`: `groups` itself,
identical Stan source, descriptor outputs and RK log density to the legacy
`kernel(ragged(x, group, labels), …) do` spelling, the omitted-response path,
the boundary refusals (including the retired `ragged(...)` statement and
in-cell read), and a per-subject linear predictor without a random effect read
as `x[i]` (BridgeStan and RK density parity with the old one-element ragged
gather).

`plate_grouped_response.jl` checks a flat response observed in the cell through
a binding, `y[obs[i]] ~ …`: density and gradient equal to the response grouped
by hand, bounds read through the binding (censored, scalar, truncated),
invalid vectors, the refusal of the retired formula-level `ragged(y, group) ~ …`,
and replay/resample regathering a grouped response and its bound.

`fixed_ranef_scales.jl` checks random-effect scales and correlations held at
known values, `sd(lp, ID) = value` and `cor(:, ID) = R`. It covers:

- parsing, `priors_of`/`merge` composition and the refusals;
- SBBRMI emission and stanc acceptance for mixed, all-known, correlation-only,
  one-margin, Student-t, `cv_groups` and `@plate` models;
- BridgeStan densities: each known model equals its sampled twin evaluated at
  the known values, up to exactly the twin's prior density of those values;
- RK/Stan coordinate transport with density and gradient parity;
- `population_draws`, `reprocess(...; resample_groups=...)`, an
  omitted-response prior program, descriptors, descriptions, and the NLME and
  adaptive-centering refusals.

Set `BRM_FIXED_RANEF_RUNTIME=0` to skip the compiled-model sets.

`sd_prior_input_names.jl` checks that a scale prior's caller scalar may share a
name with a binding of the submodel the prior is merged into, such as `tau` in
`sd(:, ID) ~ Exponential(tau)`. It covers Gaussian, general-family,
per-margin, centered, `cv_groups` and Student-t blocks, smoothing scales, and
the completion submodel of a split block. Each spelling must emit exactly the
Stan program of a twin whose scalar is called `lam`; the twin must pass stanc.

`model_family_composition.jl` checks composing a family of related models
without string templating:

- `optional(<address>)` priors: every address class is dropped when the model
  lacks it, and the result is byte-identical to the plain twin on SBBRMI and RK;
- the typed builder: `brm_columns(builder)` equals the descriptor's columns, and
  `merge(builders...)`, `Expr` composition and `show(builder)` work;
- source views: `show(::BRMI)` prints the authored statements, and the printed
  source re-parses to the same Stan program;
- `String` categorical columns on both backends, with stanc acceptance and a
  BridgeStan density matching the integer-coded twin;
- order-free structural `merge`, and the refused `effect(<unknown predictor>, :)`.

`kernel_population_lp.jl` checks per-subject linear predictors without a random
effect (a typical value plus covariates) read as `x[i]`, beside a random-effect
predictor and in a no-random-effects panel, against an independent density
oracle with finite-difference RK gradients and matched BridgeStan coordinates.
It also covers a panel whose cell slices only such predictors and the refusal of
one declared on another row axis. A predictor whose only covariate is a string
categorical (`Vector{String}` or a string-leveled `CategoricalVector`) needs no
zero `offset(...)` witness: its Stan and RK densities equal the witnessed
twin's on both spellings.

`string_categorical_coordinates.jl` checks `brm_population_effect_coordinates`
for a predictor whose categorical column is a plain `Vector{String}`, in a flat
and an `@plate` model. `Intercept` and the column address resolve to the same
result shape as a `CategoricalVector` column. A `CategoricalVector` point
carried to the String model through the resolved coordinates gives the same
pointwise log likelihood, and the same point with two levels swapped does not.
It also covers exact totals, cell-mean coding, and `brm_start` against the RK
start.

`zero_element_coordinates.jl` checks `brm_output_coordinates` on carriers with
no elements: a one-level treatment factor's contrast vector and a one-level
ordinal outcome's thresholds return `Int[]` and carry `dims == (0,)`, every
unconstrained BridgeStan name has exactly one sampled owner, and a carrier
that owns elements but is absent from the names, or whose `dims` are
unresolved, still fails closed.

`output_predictors.jl` checks `BRMOutput.predictors`, `term` and the
population-level `component`s: every term family, a column used by two
predictors, a shared `|ID|` block listing both predictors, R2D2 and horseshoe
priors, exact totals, S2Z and stratified blocks, all found through
`brm_outputs(d; predictor=, term=, component=)`. Its compiled half resolves
every address back to its own output through
`brm_population_effect_coordinates` and `brm_term_coordinates`, and checks that
grouping by predictor reaches every sampled coordinate. Set
`BRM_OUTPUT_PREDICTORS_RUNTIME=0` to skip the compiled half.

`descriptor_trace_reuse.jl` checks that `ranef_blocks(d)`,
`brm_ranef_sd_coordinates` and `brm_ranef_draws` read the descriptor's own
traced program: a copy of the descriptor whose plan holds no traceable program
answers each query identically, in a fitted and in a prior program, while
`ranef_blocks` on that plan fails.

`transform_constants.jl` checks `brm_transform_constants`, the fitted
`center`/`zscale`/`standardize` constants. On SBBRMI each record reproduces the
model's transformed Stan data bit for bit, follows frozen and refitted
`reprocess`, and reports an `mi(...)` column's observed-only fit under the
coefficient's name. On RKBRMI the records equal the graph's own transformed
values, a frozen replay keeps them, and an `&` operand is refused by name.

`plate_index_into.jl` checks `indexin(idx[rows[i]], target[i])`, an index
vector into another grouping's entries: the per-subject positions equal
hand-rebased pre-grouped columns on BridgeStan and RK,
`something.(indexin(…), 0)` keeps `0` as "no index" while plain `indexin`
refuses it, `reprocess` rebases new data again, and an index that leaves its
subject or the target vector is refused, as are bindings with different labels
or an `order`.

`dose_axis_operation_gather.jl` checks a predictor read only at the doses of an
operation stream, declared over the dose vectors and placed onto the subject's
operations in the cell, `vcat(0.0, log_F[doses[i]])` read at
`something.(indexin(op_dose[events[i]], doses[i]), 0) .+ 1`. Its BridgeStan and
RK density and gradient equal the same model on the operation stream with its
read rows filled, with a subject that has no doses. A frozen `reprocess` onto a
new schedule gives the same density on RK and the same HSGP basis rows at the
doses on Stan, and both backends replay onto new subjects. Both models have
complete scientific descriptions.

`plate_empty_axis.jl` checks a `groups(group => labels)` binding over an empty
`group` vector: a formula LP, a data vector and an observed response (against a
cell expression or a collected output) each give every cell an empty slice. Each model's BridgeStan and
RK density and gradient equal a twin that never reads the empty axis.
`reprocess` moves the same Stan program onto data with rows and back,
recomputing the row count of a declared axis that sizes an intercept-only
predictor, and a missing group label is still refused.

`hsgp_empty_axis_gather.jl` checks an HSGP with an explicit `domain=` (and a
periodic one) over an axis with no rows, gathered per subject through
`ragged(drift, d_subject)` over an empty group: each model's BridgeStan and RK
density and gradient equal a twin that declares the same term but never reads
it, and `reprocess` moves the Stan program onto data with rows.

`hsgp_constant_axis.jl` checks that a fresh HSGP fit refuses a constant axis at
every row count, including those whose computed mean rounds an ulp away from
the axis value, on the shared fit, the vectorised backend's basis, a fresh
`SBBRMI` build and a `freeze_constants=false` reprocess. A frozen replay keeps
the fitted interval on that axis, and an explicit `domain=` fits it fresh.

`plate_ragged_omitted_response.jl` checks a censored ragged observation inside
`@plate` with its data column bound, omitted, and reprocessed. It requires the
StanBlocks revision pinned by `setup_env.jl` and checks the public flat
predictive segments, bound pointwise likelihood, and refusal of a missing
non-response ragged input.

`s2z_omitted_response.jl` checks `s2z_groups` when a response is omitted. A
predictor that no observed likelihood reaches keeps its block conventional, in
generated quantities. It covers an intercept-only predictor, a grouping that a
reached predictor shares, and an omitted `@plate` response. Each program
matches its conventional or renamed-grouping twin, and the sampled BridgeStan
density equals that of `prune_responses` of the same model.

`omitted_plate_observations.jl` checks the grouped `pk_conc[i]` spelling with
an omitted censored response and a joint QT likelihood. It checks the
predictive descriptor and a finite generated draw above the censoring floor.

`ragged_rowwise_arguments.jl` checks that SBBRMI evaluates a row-wise helper
(`addprop`) or elementwise arithmetic over a ragged plate or kernel result once
per subject when it is observed outside the cell. The observation never reads
the helper over the whole flattened result. It checks BridgeStan density and
gradient parity with the in-cell spelling. For a response observed in the cell
through a binding it also checks the omitted response, its replay, and the
predictive segments. For the
model-scope form over pre-grouped data (`y ~ censored(Normal(loc, addprop(loc,
…)); lower=lloq)` with `y` omitted, and its uncensored twin) it checks the same
observation statement, an argument plate over the plate result's own cells,
predictive segments before and after `reprocess`, a finite BridgeStan draw
above the censoring floor, and that a bound `qt` beside the omitted `y` has
exactly the `qt`-only model's coordinates and density.

`omitted_censored_response.jl` checks a flat `y ~ censored(Normal(mu, sigma);
lower=lloq)` with `y` omitted: the warning, the observation role, the
`y_gen`-only output, the bound twin's predictive law, BridgeStan draws clamped
at `lloq`, and a fitted response beside it whose density equals the
uncensored twin's.

`adaptive_multiaxis_hsgp_centering.jl` checks online adaptive centering of
two-axis squared-exponential HSGPs: isotropic (one length scale every axis
reads) and `iso=false`, ungrouped and grouped, beside a shared random-effect
block. Each cell's spectral scale is pinned against BridgeStan's constrained
hyperparameters, and the wrapped non-centered model at `c = 1` reproduces the
density and gradient of the compiled `centeredness=1.0` twin.
`hsgp_coordinate_draws` returns the same log scales on those draws.

`benchmark_emission_latency.jl` reports package loading, first model construction,
first RK artifact emission and repeated calls separately, including compiler time,
allocation and garbage collection. It uses independent synthetic regression and
ragged panel examples, with unchanged cold/warm artifact source checks. Run each
case in a fresh process; there is no timing threshold on a shared host:

```sh
julia --startup-file=no --project=test test/benchmark_emission_latency.jl regression
julia --startup-file=no --project=test test/benchmark_emission_latency.jl panel
```

A second argument saves the complete artifact and printed source for comparison
across revisions, plus a source copy with location nodes removed so checkout paths
do not affect the comparison. Dependency pins, Julia version and host load should
accompany timing reports; package-image rebuilding belongs to the separate
loading phase.

`benchmark_emission_breadth.jl` measures compiler work across many independently
named but structurally identical BRMs. The names deliberately give each model a
different `BRMI` carrier type while the statistical work stays fixed, exposing
planner or emitter specialization that grows with corpus breadth. It reports RK
artifact and StanBlocks construction separately, then repeats the first model as
a warm control. As above, compare cumulative compiler time and allocation from
fresh processes on the same host; the script has no shared-host timing threshold:

```sh
julia --startup-file=no --project=. test/benchmark_emission_breadth.jl 12 both
julia --startup-file=no --project=. test/benchmark_emission_breadth.jl 12 rk
julia --startup-file=no --project=. test/benchmark_emission_breadth.jl 12 sb
```

`rk_planning_specialization.jl` pins the same property without timing. After
one warm member of a value-route family (caller-owned submodel, ordinary
callable assignment, grouped, population and log-scale predictors), building
another member, emitting its RK artifact and building its kernel must not
compile any BRM method, or generic method over BRM values, whose signature
names that member's operations. Only constructing the model's own
`NamedTuple`, `BRMI` and `RKBRMI` may. It records compilation with
`jl_dump_compiles` and also checks that the reused compilation emits the same
program up to the names. A second testset builds a member whose submodel
callable differs, so its expression types are new throughout (a named column
embeds its definition): the source walkers that collect data, references,
target axes, roles and additive terms must still reuse their compilation, while
constructing the member's values, the caller-owned hook and the prepared
representation may specialize:

```sh
julia --startup-file=no --project=test test/rk_planning_specialization.jl
```

## NLME estimators

`nlme_joint_start.jl` checks a complete NLME estimate against the same
BRMI's compiled Stan model: non-unit covariance and natural subject effects
round-trip through Stan's unconstrained coordinates, fixed/free indices
partition the full vector, subject labels are checked, and a direct coordinate
copy is shown to be a different physical point.

`nlme_conditional.jl` conditions that compiled model, and the RK program of
the same BRMI, on a checked start for flat, fixed-shared-HSGP and nested
models: values, gradients and Hessians equal the full target's at the embedded
point restricted to the subject draws; the RK and Stan problems agree on free
gradients and log-density differences at the same physical points; fixed
coordinates survive every evaluation and short WarmupHMC runs on both
backends; provenance names each free draw's subject and nested unit; and
mismatched or tampered starts and foreign backends are refused. It runs in the
test environment above.

`nlme_view.jl` pins `brm_nlme_view`, lockstep evaluation
(`brm_nlme_loglikelihoods[_and_gradients]`), and single-subject cell
evaluation (`brm_nlme_loglikelihood[_and_gradient]`) against an independent
oracle with analytic η gradients. It also compares the deprecated
`kernel(...) do` spelling with the indexed cell's partition, likelihoods and
η gradients. The single-subject test checks that
value and gradient allocations stay flat when unrelated subjects grow from
3 to 9 to 27. A response observed in the cell through a `groups` binding,
including a left-censored one with a per-row limit and an `addprop` scale
beside another in-cell observation, has the same single-subject cell, checked
against independent oracles. Both joins keep one subject's work flat at 81 and
243 subjects; the censored join's `addprop` scale is written by the kernel's
plate, so its cell does not run the kernel for every subject. A completed
covariate's observed entries join their subject's cell.
Flat-row grouped models retain lockstep evaluation and refuse
the single-subject method by name. It runs in the test environment above.

`nlme_nested.jl` uses a public synthetic model with interleaved observations
and 1, 2, and 3 nested units per subject. It checks explicit ownership,
compact η likelihoods and gradients against an independent Normal oracle,
the ragged NLME protocol, posthoc/SAEM smoke paths, a full Stan-start
round trip with reordered subjects, and the end-to-end chain from an SAEM fit
through `brm_nlme_joint_start` to WarmupHMC on the conditional Stan and RK
subject-draw targets. Run it in the NLMEEstimation scratch
environment described below.

`nlme_fit.jl` checks the prior-median default start against independently
computed medians (authored, default, truncated, hierarchical and `Flat()`
laws), keeps random-effect blocks without an ID independent, reads named
natural-unit estimates of a hand-set point (and passes them back as
address-keyed `inits`), and runs `brm_nlme_fit` with `empirical_bayes`,
`saem` and `focei` on flat and nested models. Run it in the NLMEEstimation
scratch environment described below.

`nlme_start.jl` reads complete RK and Stan points as NLMEEstimation starting
values with `brm_nlme_start`. It covers a correlated population-PK model and a
nested model with a correlated subject family, nested units and a shared HSGP,
in estimated and fixed shared modes. Subject log-likelihoods sum to the RK
observation likelihood at the point, and `brm_nlme_joint_start` rebuilds the
same Stan point from the natural values. Initial estimates keyed by prior
address (`brm_nlme_start(m, inits)`) give the addressed θ, σ and Ω with zero
η, and `brm_start` on the joint map's Stan model gives the same point as
`brm_nlme_joint_start` of that start. Run it in the NLMEEstimation scratch
environment described below.

`start_values.jl` checks `brm_start`, initial values keyed by prior address,
on both backends. For a correlated population-PK model, a linked predictor,
a categorical predictor with a `Dirichlet` simplex, and an exact-total block,
it checks the addressed constrained values, the semantic transport's
agreement between the RK and Stan points, unaddressed coordinates kept at a
random base, and refusals of unknown addresses and invalid values. A model
with one lone `|ID|` per predictor checks that the default exact totals keep
each `|ID|`'s scale address, for `brm_start` and `brm_ranef_sd_coordinates`,
including the generated-quantities block of its prior program. It runs in
the test environment above.

`nlme_estimation_ext.jl` covers the NLMEEstimation.jl protocol extension:
batched and single-subject evaluation (including `check_protocol` and online
SAEM on ragged joins, and SAEM's batched population gradient for flat grouped
rows), subject-cell population gradients,
analytic η Hessian blocks, Gaussian observation moments
and colored η Jacobians, `check_protocol`, and Newton posthoc against
independent oracles. Its Gaussian checks include arrays observed within plate
cells and a modeled scale shared by the observations in each cell.
NLMEEstimation.jl is not part of `test/Project.toml`. Run the file from a
copy of this environment that develops a local NLMEEstimation checkout:

```sh
cp test/Project.toml test/Manifest.toml <scratch-env>/
julia --project=<scratch-env> -e 'using Pkg; Pkg.develop(path="<NLMEEstimation checkout>")'
julia --project=<scratch-env> test/nlme_estimation_ext.jl
```

Without NLMEEstimation the file stops with an error rather than skipping.
Shared fixtures live in `nlme_fixtures.jl`.

## Description PDF rendering

`description_tex.jl` checks Student-t sampling equations for standard, affine,
long-expression and embedded kernel models while preserving their source,
data, priors and model identity. It also runs under the root project. Add
`--pdf` to compile the complete public Markdown through Quarto and LuaLaTeX:

```sh
julia --project=. test/description_tex.jl --pdf
```

This explicit PDF mode requires Quarto and a working LuaLaTeX installation;
render failures propagate. Outputs use a temporary directory, or the directory
specified by `BRM_DESCRIPTION_TEX_OUTPUT`.

## Exact totals beside offsets

`total_effects_offset.jl` checks that `offset(...)` terms, of data and of a
model value, leave an otherwise eligible block on exact totals under `:auto`.
Its BridgeStan density equals the conventional program's marginal over the
integrated intercept at random points. It also checks that `reprocess` rebinds
the offset column, that `population_model` keeps the offset, and that the
description stays complete. An explicit `total_groups=[:g]` must raise an error
naming every predictor whose block on `g` would stay conventional:

```sh
julia --project=test test/total_effects_offset.jl
```

## Exact totals for some blocks of a group

`total_effects_selection.jl` checks `total_groups=[:g => [:b]]` on several
per-unit linear predictors that share one grouping and one `@plate` cell:

- `:auto` stays conventional;
- a whole-group request names each skipped predictor, why, and the eligible
  subset;
- the pair form integrates exactly the named blocks;
- the malformed or ineligible requests are refused.

Its BridgeStan density equals the conventional program's marginal over the
integrated intercept, also for a block whose scale prior is
`truncated(Normal(0, 1); lower=0)`. It checks that `generative_plan(plan,
new_df)` keeps the fitted selection, and that the joint WarmupHMC wrapper maps
the totals beside the group's conventional blocks:

```sh
julia --project=test test/total_effects_selection.jl
```

## Exact-total descriptions

`description_totals.jl` describes each synthetic model under the default
automatic exact totals and under `total_groups=()`. Shared logical prior
identities must carry identical families, arguments and support in both
programs, and the totals description must be complete. It also covers remaining
population columns, independent total columns, Student-t and flat population
priors, a kernel cell, coordinatewise generated vector priors with a consumer
family gap, and the retained sum-to-zero gap. It runs under the root project:

```sh
julia --startup-file=no --project=. test/description_totals.jl
```

## Group-level predictors in exact totals

`total_effects.jl` checks the exact-total Stan density, its gradient and
recovery against a Julia reference, with and without per-group design values
`W`. The reference subtracts beta's exact conditional density at two different
betas, which agree only for the true conditional moments.
`total_effects_group_predictors.jl` checks that a population column constant
within the block's groups, or equal to a random slope times a per-group value,
is integrated with the matched coefficients. Each totals program's log density
equals the conventional (`total_groups=()`) program's with those coefficients
integrated out. It also covers generated deviations, recovery,
`brm_population_effect_coordinates`, `population_draws`, `brm_start`,
centering locations, `population_model`, descriptions, frozen replay and rebuild
refusals for changed group-level values, transport for a new group, and the
legacy rebuild of a fit recorded without per-group values. Columns that vary
within a group, or that would make the totals' mean design rank-deficient,
stay sampled. `string_categorical_coordinates.jl` checks that a group-constant
`String` column stays sampled, like a `CategoricalVector` column.

```sh
julia --project=test test/total_effects.jl
julia --project=test test/total_effects_group_predictors.jl
```

## In-cell distribution combinators

`description_cell_combinators.jl` describes `@plate for` cells whose
observations use StanBlocks' `censored`, `truncated`, `weighted` and
`interval_censored` combinators. A call-form family is an argument child; a
token-form family is a `:family` child over the token's arguments, which a
family hook sees by its actual callable. Bounds are keyword-only, so a token's
arguments never render as bounds. The in-cell censoring prose matches the
formula wrapper's. A combinator token in `weighted`'s family position takes
`weighted`'s bounds, so `weighted(censored, w, normal, m, s; lower=lo)`
describes the same censoring as the call form
`weighted(censored(Normal(m, s); lower=lo), w)`. It runs under the test project:

```sh
julia --startup-file=no --project=test test/description_cell_combinators.jl
```

`description_concatenation.jl` describes `vcat` in its two roles. As a value in
a `@plate for` cell, such as the dose-axis gather's `vcat(0.0, log_F[doses[i]])`,
it is a covered built-in rendered as a stacked vector. As the distribution of
`~`, here in an included `@slic` submodel, it stays the block-wise prior family.
It runs under the test project:

```sh
julia --startup-file=no --project=test test/description_concatenation.jl
```

## ODE solvers in kernel cells

`rk_kernel_ode_cell.jl` runs the population-PK `@plate for` cell of
`nlme_fixtures.jl` with its closed-form concentration replaced by Stan's
`ode_rk45`, `ode_rk45_tol`, `ode_bdf` and `ode_bdf_tol`, spelled as
stanblocks-use §29 writes them (`rep_vector`, `to_array_1d`, `[:, 1]`,
`to_vector`) with a `@deffun @juliacompat` right-hand side. On RKBRMI each
solve is checked against the closed-form model's density and Reverse gradient
at the same coordinates, and the tolerance-controlled forms against the
compiled Stan density and gradient. It needs ReactiveKernels with
`rk_ode_rk45_tol` (`ce48fc77` or later) and the solver packages, which the
test environment does not carry. Run it from a copy of this environment that
develops such a ReactiveKernels checkout (its root and the three nested
packages, as `setup_env.jl` does) and adds the solvers:

```sh
cp test/Project.toml test/Manifest.toml <scratch-env>/
julia --project=<scratch-env> -e 'using Pkg; rk = "<ReactiveKernels checkout>"
    Pkg.develop([PackageSpec(path=rk),
        PackageSpec(path=joinpath(rk, "packages", "ReactiveKernelsPPL")),
        PackageSpec(path=joinpath(rk, "packages", "ReactiveKernelsDistributionKernels")),
        PackageSpec(path=joinpath(rk, "packages", "ReactiveKernelsPPLExamples"))])
    Pkg.add(["OrdinaryDiffEqLowOrderRK", "OrdinaryDiffEqBDF", "SciMLBase", "SciMLSensitivity"])'
julia --startup-file=no --project=<scratch-env> test/rk_kernel_ode_cell.jl
```

`kernel_ode_cell.jl` remains
the Stan-only gate for the closure-style `kernel(...) do` spelling.

## Native GP and dual HSGP adoption

`rk_gp_covariance.jl` checks BRM-owned squared-exponential and periodic
covariance construction, multivariate locations, duplicate-location jitter,
input ownership, graph composition, independent density and native Reverse
coordinates. `rk_dual_hsgp.jl` checks the exact public motorcycle dual HSGP
against independent normalized densities and all 44 gradients at three points
and four centeredness settings; it reuses the existing hashed public data.
Both load the real ReactiveKernels extension.

These focused scripts use RK's public `recipe_inventory` and
`rk_cholesky_lower` APIs. Run them
in an environment carrying RK `db4f16aa0f3d768e376269429ec613691c743b13`
or a descendant containing both APIs and the precompiled graph allocation
repair and native prepared-callback Reverse repair. The matching nested
packages share the same checkout. The standard
`test/setup_env.jl` pin and `test/Project.toml` source revision carry that floor:

```sh
julia --project=test test/setup_env.jl
julia --project=test test/rk_dual_hsgp.jl
julia --project=test test/rk_gp_covariance.jl
```

## Chunking heavy suites

`rk_retained_transport.jl` attaches `RKBRMI(brmi, artifact.plan, built)` to
the artifact route's existing build. It checks normalized compiled Stan
densities, every ordinary Reverse coordinate and the physical-check return
schema, with all responses and a held-out response. A centered SBBRMI maps
exactly onto the same build (densities and gradients included) while the
permutation-only helpers refuse it; invalid input and a corrupted permutation
remain errors. Public block metadata identifies known parameterization mismatches
before compiling Stan. `rk_coordinate_transport.jl` covers the broader semantic inventory,
including the typed unsupported-coverage diagnostic for HSGP internals.
Run either file with `julia --project=test test/<file>.jl`.

Files with dozens of testsets (`rk_parity.jl`, `rk_emitter.jl`) OOM a squeezed
host single-process. Those files spell their blocks `@stestset` (defined in
`testset_filter.jl`, covered by `testset_filter_check.jl`) instead of
`@testset`, so lanes can run them in fresh-process chunks: pass substring
filters matching testset names as trailing args, or comma-separated via
`BRM_TEST_FILTER` (union). An empty filter runs everything, exactly as before;
a non-empty filter that matches nothing exits 1 rather than reporting a
hollow green.

```sh
julia --project=test test/rk_emitter.jl "group-C" "fail closed: group-C"
BRM_TEST_FILTER="von-Mises,mixture" julia --project=test test/rk_parity.jl
```

A new heavy file adopts the same contract with one `include` plus the macro:
`include(joinpath(@__DIR__, "testset_filter.jl"))` after `using Test`, then
`@stestset "name" begin ... end` per chunkable block.

`hsgp_hyper_prior_support.jl` checks explicit priors on authored log HSGP
intercepts through both RK and Stan. The reported grouped model and an
ungrouped control retain unbounded Normal intercepts; bounded and normalized
truncated priors keep their own support and density constants. The checks
compare independent normalized densities, every ordinary Reverse coordinate,
compiled same-BRMI Stan, group-specific floors and full printed-source replay.

`missing_covariates.jl` checks joint continuous `mi(x)` predictors through
StanBlocks. It compiles Normal and LogNormal models and compares normalized
densities and all unconstrained gradients with an independent explicit
observed/missing split, then resolves completed covariates and derived values
through public logical output coordinates. Separate blocks cover complete
conditional observations, observed-only transform anchors and their degenerate
cases, frozen row masks, and retention
of fitted missing coordinates during subject kernel CV. It uses only root
dependencies, so `julia --project=. test/missing_covariates.jl` also works;
trailing filters such as `anchors` or `kernel CV` select focused blocks.

`rk_single_level_factor.jl` checks a single-level treatment-coded factor on the
RK route. Its K=1 source equals the K=2 source apart from the level list and the
zero-width coefficient declaration, as SBBRMI declares `vector[0]`. Densities and
every native Reverse coordinate equal the factor-free model, including R2D2 and
Horseshoe predictors, and same-BRMI compiled Stan agrees through the coordinate
transport. Run `julia --project=test test/rk_single_level_factor.jl`.

`rk_missing_value_predictors.jl` checks native completed covariates with fixed
observed-only standardization anchors, shared correlated subject effects and
subject-kernel composition. It compares full normalized densities and all
inference gradients with independent and compiled same-BRMI Stan controls,
and replays complete printed definitions/data. Observed-only declarations and
complete real columns retain their modeled observations without missing
inference coordinates. Run `julia --project=test
test/rk_missing_value_predictors.jl`; trailing filters select individual
blocks. These are public producer controls, not original application or
performance acceptance.

`rk_completed_covariate_axes.jl` checks a completed subject covariate feeding
ragged kernel cells with their own likelihood, scalar covariate hyperpriors,
and two correlated subject predictors. It compares normalized densities and
all native Reverse gradients with an independent oracle and compiled
same-BRMI Stan, including cell-row permutations and separate pointwise terms.
Printed source replay, completion graph ports, multiple missing entries and
callable name collisions are included. Run `julia --project=test
test/rk_completed_covariate_axes.jl`. The verified native closure is RK/PPL
`3bdbfbbbb1c82caf4acf7b6ecb5fa28e3a089c58`, StanBlocks
`e355994384d44b86108105264ae3e34e0f2a0625`, and Enzyme 0.13.209.

`rk_component_prior_inputs.jl` checks caller values used by group and HSGP
priors when their names collide with the component's own parameters. It uses
independent normalized laws, all-coordinate finite differences and full printed
source replay with native Reverse gradients. Run `julia --project=test
test/rk_component_prior_inputs.jl`.

`rk_modeled_transform_predictors.jl` checks named pointwise assignments from
modeled and completed columns as formula predictors. It retains the original
subject axis across ragged kernel likelihoods and checks independent normalized
densities, every ordinary Reverse coordinate, compiled same-model Stan, named
graph values, row permutations, bit-exact printed-source values, and
all-coordinate Reverse agreement (absolute/relative tolerance `2e-13`). A second model covers assignment chains, scalar sampled parents and
an explicit transformed-coefficient prior without completion. Run
`julia --project=test test/rk_modeled_transform_predictors.jl`; trailing filters
select the standalone blocks. It uses the same native closure listed above.

`rk_modeled_hsgp_axes.jl` checks HSGP terms over model-derived axes (an
assignment such as `x = exp(eta)` and a log-link predictor) with a fixed
`domain=`, and linear orthogonalization on model-derived and raw axes. The RK
graph evaluates the basis, centering and projection from the current axis
values. It checks the actual built graph intermediates, the emitted basis
law at fresh axes, independent normalized densities, every ordinary Reverse
coordinate, compiled same-model Stan, pointwise queries and their replay, row
permutations, artifact replay and complete printed-source replay. Run
`julia --project=test test/rk_modeled_hsgp_axes.jl`.

`missing_joint_covariates.jl` checks the native correlated
`mi([x, z]) ~ MvNormalCholesky(...)` block. Independent normalized density,
all-coordinate gradient and pointwise likelihood oracles cover all four
bivariate row-specific missing patterns, formula priors and frozen replay.
Other blocks check positive log-space assignments, same-named dataframe
columns, observed-only anchors, complete or entirely missing columns,
row-aligned vector locations, and fitted completion during group resampling.
Run `julia --project=. test/missing_joint_covariates.jl`; trailing filters
such as `independent` or `log-space` select focused blocks.

## Shared preparation and Turing lowering

`rk_addprop.jl` checks the shared additive/proportional scale law as visible
native graph source. Unequal grouped responses, empty subjects, callable
aliases, top-level assignments and in-cell calls retain normalized densities,
all ordinary Reverse gradients, censoring and row order against independent
oracles and compiled same-BRMI Stan. Per-location scales read in the cell,
including through a binding and a scale computed from parameters and per-row
data, have the same checks. Complete printed source and artifact re-entry are
included; a scalar native-location control is checked separately.

`rk_r2d2_priors.jl` checks shared-margin R2D2M2 and joint population/contrast
budgets through native RK execution. It compares absolute normalized densities,
every ordinary Reverse coordinate, pointwise responses and complete printed
source replay with independent oracles and compiled same-BRMI Stan at physical
simplex parameters. Controls cover unequal categorical frequencies, explicit
factor design columns, sampled latent references, partial per-margin ICCs and
conflicting allocation statements. The two original synthetic formulas retain
their separate active sampled reference scales.

`rk_coordinate_transport.jl` checks the public complete semantic RK/Stan
coordinate map: scalar and population coefficients, categorical reference and
cell-mean coding, grouping levels, random-effect scales and standardized draws,
and whole correlation factors, including independent predictor blocks on the
same grouping column, and centered SBBRMI blocks (intercept, correlated, single
slope, shared `|ID|` and configured-scale families) through their exact
per-point map. Distinct nonzero points compare constrained
values, normalized densities and every ordinary Reverse gradient with compiled
same-BRMI Stan. Point/gradient round trips, input preservation and incomplete
coverage diagnostics are included. HSGP internals remain an expected capability
gap. `rk_modeled_random_slopes.jl` additionally compares the generated map with
its independent hand-written mapping. Run either file through the test project;
trailing filters select individual blocks.

`spline_basis_signs.jl` checks canonical TPS/t2 projection signs, input
ownership, and frozen projections directly. The spline blocks in
`rk_parity.jl` compare RK densities and Enzyme gradients with independent
oracles; `spline_parity_models.jl` holds their shared BridgeStan fixtures.
`spline_sb_parity.jl` compiles those fixtures independently and checks the
full-posterior anchors and gradients, including the distributional spline toy.

The focused preparation gates are `preparation_program.jl`,
`preparation_replay.jl`, `preparation_assignments.jl`, and `backend_plan.jl`.
`rk_modeled_random_slopes.jl` keeps modeled formula values active in both
population and subject slope designs. It covers six rows from three subjects,
censored LogNormal and Normal observations, sampled covariance and authored
priors, row permutations, independent full-density/native Reverse oracles,
compiled same-BRMI Stan, and complete source/artifact replay. Direct formula
slopes and partially completed covariates also keep their original row axes.
`rk_declared_consumer_axes.jl` checks that an explicit ragged consumer supplies
an intercept predictor's row axis even when its endpoint data are omitted.
Its observed model compares complete printed RK replay, ordinary Reverse and
an independent normalized oracle with compiled same-BRMI Stan on distinct
subject, ECG-row and response axes. Omitted-data geometry checks do not claim
native prior-draw execution.
`rk_monotonic_consumer.jl` combines sampled monotonic contrasts and an HSGP
on operation rows consumed by a subject kernel. It checks complete source and
artifact replay, independent density/all-gradient oracles, compiled same-BRMI
Stan at physical simplex parameters, and an empty subject's retained prior.
The assignment gate checks dependency ordering and distinct response row axes.
`preparation_replay.jl` compares the shared fitted transform contract across
StanBlocks and Turing. Callable-specific frozen replay is covered by
`turing_terms.jl`, `turing_gp.jl`, `turing_structured.jl`, and the corresponding
StanBlocks term files; these tests guard callable-type dispatch, fitted spline
and HSGP geometry, categorical/group levels, and backend-owned data bindings.
`turing_generic.jl` exercises
callable likelihoods and priors, while `turing_backend.jl` retains the existing
grouping, conditioning, replay, prediction, and parameterization contracts.
`turing_world_age.jl` constructs and evaluates models inside compiled callers
and checks that generated-model caching distinguishes prior literals.
`model_source_ownership.jl` checks concurrent Turing construction, source-AST
isolation and stable shared-group source. Run it in a fresh process with `julia --threads=4 --project=test
test/model_source_ownership.jl` to exercise the concurrent paths.
`sbimpl_generation_concurrency.jl` checks concurrent SBBRMI construction and
immediate consumption in compiled callers, including cold and warm vector
priors, continuous and discrete mixtures, horseshoe labels and source ownership,
same-named custom modules, and valid construction after rejected input. It also
checks that generated families add no module bindings or support methods. Run
`julia --threads=4 --project=test test/sbimpl_generation_concurrency.jl` in a
fresh process.
`rk_construction_concurrency.jl` checks independent RK builds with distinct
data and layouts, then compares their first executions against analytic
densities. Run `test/setup_env.jl` to install the fixed RK pin, then
`julia --threads=4 --project=test test/rk_construction_concurrency.jl`.

`rk_bounded_source.jl` checks scalar declaration bounds and ordinary value
observations. `rk_bounded_runtime.jl` compares the original family kernels,
coordinate Jacobians, standard Enzyme gradients and full printed-source replay
with independent oracles and emitted Stan. These use the exact published
`restricted` capability pinned by `setup_env.jl`.

`rk_computed_predictor.jl` covers ordinary formula predictors consumed by a
whole-array gather, with and without an intercept and with equal or distinct
predictor/observation row axes. It checks full normalized densities, analytic
and finite-difference gradients, mapped compiled Stan, caller ownership and
complete printed-source replay on the published computed-matrix repair.

`rk_wildcard_ownership.jl` checks that wildcard coefficient defaults reach
owning predictors, skip nonowners, and retain unmatched-target and equal-
specificity errors. Its unchanged linked multi-axis model compares normalized
Normal/censored densities, standard native Enzyme gradients, mapped compiled
Stan and complete public printed-source replay.

`rk_held_out.jl` checks response-level likelihood withholding with shared and
separate parameter branches. It preserves every fitted prior and coordinate,
compares normalized native densities and reverse gradients against independent
oracles and compiled Stan, and replays the complete emitted source and artifact.

`held_out_rows.jl` checks that a per-row `held_out` mask is refused on both
backends and on every replay (a fit receives only the rows it conditions on,
user decision `1bwqnxs`), that a replay's `held_out=` replaces the fit's
whole-response hold-outs through `reprocess`, `restan_data`, `generative_plan`
and the descriptor, and that a per-subject window fitted on its kept rows
replays onto the full data, a denser grid and a re-drawn subject with its
frozen `zscale` constants.

`rk_callable_terms.jl` checks positional/keyword formula broadcasting, including
singleton input axes, against independent and compiled Stan targets.
`rk_kernel_values.jl` checks unequal and empty grouped cells, explicit ragged
joins, lexical native callables, distinct catalogue/response axes and in-cell
held-out aliases without rewriting the original BRMI.
`rk_grouped_bounded_response.jl` gathers observed rows before bounded-response
validation and keeps scalar or row-specific censoring, truncation and interval
bounds aligned with uneven/empty subject cells. Complete source replay,
ordinary Reverse, independent densities and gradients, compiled same-BRMI
Stan, invalid evidence and original bound-column ownership are covered.
`rk_submodel_values.jl`
checks paired source hooks for an empty marker with active predictor keywords,
no extra result coefficient, exact callable bindings and fresh-module replay;
an emitted entry and a bound leaf cannot share the same symbol. Two hook calls
appending one shared helper `@kernel` emit its definition once, with the
independent density/gradient oracle and artifact replay; a different definition
under that name fails before evaluation.

`rk_callable_source.jl` checks original nested callable identities supplied by
ordinary native source definitions. Ragged cells, live keyword arguments,
native Reverse, independent density/gradient oracles, ownership and complete
source replay are covered. Both ordinary assignments and uneven/empty ragged
nested helpers compare with compiled same-BRMI Stan on the published
StanBlocks caller-dimension fix `65cabef`, pinned by `setup_env.jl`.
These scoped checks do not certify complete application performance.

`rk_graph_source.jl` supplies explicit `@kernel` entries through the callable
source provider. It inspects the actual bound posterior for subject plates and
child scans, then checks complete printed and artifact replay, ordinary Reverse
and independent normalized oracles against compiled same-BRMI Stan. The
computed grouped predictor case retains an empty subject's prior and all five
coordinates while passing separately sliced inputs to the child object graph.
The pinned RK revision also supports those computed graph arguments; source
definitions or a separately prepared reader alone do not prove posterior
visibility.

`kernel_expr(bound, assign_layout(bound))` is pre-build replay text. Its calls
to qualified `KernelSpec` readers compose when `build_kernel` runs; textual
plate/scan counts therefore do not count the complete built graph. Use public
`kernel_graph(built.spec)`, `plate_body` and `scan_body` entry/body accessors.
The fixtures explicitly qualify their internal recipe predicates as frozen
revision diagnostics. The former `999451c` composition failure is repaired
on the pinned published RK `5184fe62`; provider-only controls still do not
certify the complete original application.

`rk_kernel_observation_families.jl` checks exact in-cell Student-t and
caller-owned scalar observation graphs through the original constructor
identity. Direct, grouped and separately bound child `KernelSpec` forms retain
two aligned vector arguments and a live shared scale, with normalized
independent values/gradients, ordinary Reverse, strict same-BRMI Stan,
complete source/data replay, original geometry/ownership and scalar algebra
inside the actual built observation plate. A Bernoulli-logit case preserves
integer response rows across uneven/empty cells.
`rk_grouped_observation_arguments.jl` checks the equivalent outside-cell law,
grouped argument collection, explicit ragged joins, empty cells, broadcasting,
elementwise argument expressions and unchanged data ports. Each case checks
normalized independent values and every ordinary Reverse coordinate, strict
same-model compiled Stan parity, printed source replay and actual graph gathers.
Run the standalone file with optional case filters such as `joined` or `integer`.
`rk_kernel_weighted_observations.jl` checks in-cell
`weighted(family, weight, args...)` power likelihoods: SLIC token and call
forms, a cell-expression weight and a subject-shared weight over a native
family, plus a caller-owned `_rk_observation_source!` law, each with uneven
and empty cells, independent normalized values and Reverse gradients, strict
same-BRMI Stan parity and printed source replay. The weight reader must read
only data. A weighted censored Normal, in the call form
`weighted(censored(Normal(m, s); lower=lo[i]), w[i])` and the token form
`weighted(censored, w[i], normal, m, s; lower=lo[i])`, is checked against an
independent weighted clamp density, Reverse gradients, pointwise rows and
strict same-BRMI Stan parity. In-cell `interval_censored` remains
`@test_broken` on RK.
`rk_incell_censored.jl` checks an in-cell `censored(Normal(...); lower=...)`
over a ragged join against Stan, pointwise rows and NLME subject
log-likelihoods, then the same cell under a row weight against the
unweighted model: density, Reverse gradient, pointwise rows and NLME
log-likelihoods all scale by the row weights, and its density and gradient
match Stan.
`rk_observed_graph_law.jl` checks the public inclusive threshold law's lower
and upper branches and normalized interior in that built graph, against
independent native values/gradients. The retained ordinary-constructor fixture
`rk_observed_callable_law.jl` has three strict Stan-gradient failures caused
by the independently attributed upstream `normal_lcdf` derivative
approximation (`inclusive-normal-eff38407`, StanBlocks issue #59); this
source/graph bridge does not repair or relax that scientific comparison.

`rk_plain.jl` and `rk_statistical_library.jl` exercise BRM-owned statistical
declarations through ordinary RKPPL lowering. `rk_statistical_source.jl`
compares emitted reusable block algebra with the former ordinary inline
source: every statistical submodel allocates its own parameters and returns
its computed value. Scoped names such as `b_g.tau.1` map bijectively
to the hand-inlined program's `b_g_tau.1`. Prior, likelihood and full
densities agree to rounding at three points, as do all standard Reverse
coordinates. Caller ownership is preserved and complete printed-source replay
is bit-identical. Optional testset filters select individual block cases.
Graph structure checks use RK's public `recipe_inventory`, including retained
child plates and scans.
`rk_prepared_data_source.jl` checks emitted categorical/ordinal preparation and
completed-covariate geometry, including declared level order, empty and
all-missing partitions, retained numerical recipes, printed replay and cached
columns that must not override source computations. Run it directly with
`julia --project=test test/rk_prepared_data_source.jl`.
`rk_factor_gather.jl` checks that categorical population effects gather their
coefficients from one level index per factor coding (treatment coding with a
leading reference zero, cell means directly, `ref=` listed first), that
predictors with identical design columns share one matrix, and the normalized
density and ordinary Reverse gradient against an independent oracle.
`rk_hsgp_domains.jl` checks fixed one-dimensional and tensor HSGP domains
against independent basis/frequency/floor calculations and normalized
same-BRMI compiled Stan values and every mapped ordinary Reverse coordinate.
It covers K=1, isotropic/anisotropic geometry, explicit normalized or bounded
hyperpriors, unchanged fixed boundaries under row rebinding, caller ownership,
actual built graph inspection, and complete printed-source fresh-module replay.
`rk_source_composition.jl` additionally combines those statistical definitions
with identity-resolved native entries and live keywords, retaining both kinds
of definitions in the ordinary source reference and complete printed replay.
`rk_leveled.jl` adds independent
categorical-reference and multinomial row-density, Jacobian and ordinary
reverse-gradient oracles. `rk_values.jl` checks whole
predictor arrays, callable readers and distinct predictor/observation axes.
`rk_source_roundtrip.jl` supplies the shared complete-program replay check:
definitions and model source are printed, reparsed and built afresh, then
compared bit-exactly for likelihood, prior and full density at three points.
`rk_parity.jl` adds independent statistical density and Jacobian references,
standard Enzyme reverse gradients and finite-difference checks.
`rk_stan_priors.jl` compares explicit shared scale/LKJ priors with emitted Stan
using a complete named coordinate map. `rk_gaussian.jl` checks the actual
ordinary Gaussian emission against analytic and mapped Stan densities and
gradients at three points and three data sizes, then reports warmed native
density and preallocated reverse allocations without machine-specific limits.

`turing_natural_emission.jl` checks direct observation ASTs and named model
inputs against an independently written Turing model. It executes the emitted
source again, checks input-name hygiene and closure captures, and verifies
bounds, analytic weights, prediction and replay after removing runtime
preparation-plan lookups from the generated body.

The `turing_terms.jl`, `turing_gp.jl`, `turing_structured.jl`,
`turing_r2d2.jl`, and `turing_responses.jl` scripts exercise the corresponding
native verticals. Prior support and density comparisons against BridgeStan live
in `truncated_priors.jl`, `hierarchical_prior_bounds.jl`,
`prior_expression_keywords.jl`, `term_prior_bounds.jl`, and
`von_mises_prior.jl`. `backend_comparisons.jl` executes the documentation's
four-pane helper, checks the emitted Stan, and scores the generated Turing
model.
`sbbrmi_display.jl` evaluates the displayed configured-submodel bindings and
body in a fresh module, requiring identical generated Stan and data. It covers
shared and distinct configurations, unchanged default calls, and replay.
`callable_priors.jl` checks distribution-factory shape registration and a
complete-call Stan AST translation against the original Julia factory's
density.
`term_prior_semantics.jl` checks that both backends consume every resolved term
prior address, reject unmatched or ambiguous addresses, and retain the exact
density, RNG, support, dimension, keywords, and simplex transform of a custom
multivariate simplex prior. `turing_descriptor.jl` checks the common
`brm_descriptor`/`brm_output`/`brm_outputs` schema and Turing-backed execution;
its descriptor intentionally has no Stan artifact or source highlights.
`stanblocks_term_prior_semantics.jl` checks custom simplex densities through
BridgeStan and prior-only RNG draws, alongside unchanged Dirichlet spellings.
`turing_shared_assignments.jl` checks that shared group effects reach dependent
assignments before their consumers are evaluated.

`adaptive_centering_public_api.jl` exercises ordinary scalar, shared-ID `K=1`,
and correlated `K=3` blocks through the public wrapper at both compiled endpoints.
It checks scalar/per-cell controls, independent BridgeStan scale/correlation
queries, map/Jacobian/inverse/density and all gradients, restoration/copy isolation,
and every cell's 11 candidate losses against an independent weighted-correlation
oracle at zero, mixed and one source controls. It tests the fixed-frame scoring
proxy; it does not measure online window winners or sampler convergence.

`adaptive_centering_learned_order.jl` covers `ranef_order=:learned`, where
each group's triangular order follows its controls (snag
`correlated-kreis-15564811`). Its synthetic half needs no Stan model: it checks
the map at both compiled endpoints against a dense `cholesky(P*Sigma*P')`
reference, plus the inverse, a finite-difference log-Jacobian and the Enzyme
gradient. With every group in declaration order, the map, Jacobian, inverse
and scoring frame must equal the per-pair joint wrapper (ordinary, R2D2, HSGP
and scalar cells). The window rule is checked against an independent
re-implementation. The compiled half checks a `K=3` block at both endpoints
against BridgeStan's own `diag(tau) * L`, the `adaptive_centering_controls`
report, copy isolation, the declared fallback of a model without a correlated
block, and a short WarmupHMC run that is checkpointed and resumed.

`adaptive_hsgp_centering.jl` checks pilot-selection under spectral underflow,
the partial-coordinate Jacobian, Turing/Enzyme versus StanBlocks/BridgeStan
density and gradient parity, physical constrained quantities, and the
distributional model's two distinct zero-mean HSGP bindings.

`adaptive_bounded_scales.jl` checks that online adaptive centering reads
`Uniform(a, b)`-bounded HSGP length scales and SDs, and cdar SDs and
persistences, through Stan's own `lb`/`lub` transform. It also reads the
`<lower=fmax(0.0, L)>` declared by a `truncated(D; lower=L, ...)` scale prior
whose `L` is `length_scale_floor` or a data scalar, per axis for an
anisotropic term. Because the wrapper's density identities hold for any
function of those coordinates, each compiled model that adapts first compares
the physical value every cell reads with BridgeStan's `param_constrain`, then
checks the wrapper identities. A bound-independent oracle checks that a
`truncated(LogNormal(0, 1); lower=length_scale_floor)` prior wraps exactly like
the default floor plus one constant. A bound written as arithmetic over data is
a `@test_broken` capability gap.

`adaptive_car_centering.jl` checks online adaptive centering of the areal
`car(area; edges, type=:icar | :bym2)` field: one triangular pair of cells per
area, its `phi` cell and (BYM2) its `theta` cell centred on the area's spatial
component. It resolves the cells against the compiled model, checks the
transform, Jacobian and inverse against an independent map written from the
emitted law, checks that `c=1` is BridgeStan's transformed `effects` (islands
and areas without rows included), checks candidate-frame invariance and Enzyme
and finite-difference gradients, checks car cells beside a correlated
random-effect block in the joint and learned-order wrappers (each family maps
as its own wrapper does), the controls report, the cdar refusal and the
generated-field skip, and runs one short online warmup on a data-rich lattice.

`adaptive_centering_threads.jl` differentiates the ordinary, joint (ordinary +
R2D2 + HSGP), learned-order (`ranef_order=:learned` over the joint cells),
S2Z, HSGP-only, cdar, scalar (total / S2Z cell) and car wrappers, and car
cells inside the joint and learned-order wrappers, on
three concurrent tasks while a fourth forces collections, in a child Julia
started with four threads; the child must exit cleanly and every gradient must
bitwise-equal its single-threaded reference. On Julia 1.10, Enzyme's reverse
pass corrupted Julia's GC whenever an accessor allocated a work vector, read a
struct holding vectors in its row loop, or reached two `_block_location`
instantiations: the process segfaulted in `getindex` or aborted with "GC error
(probable corruption)". Synthetic blocks only; no Stan toolchain. The stress
loop and child launch live in `threaded_gradient_stress.jl`, shared with the
gate below.

`rk_enzyme_threads.jl` runs the same stress, 15 seconds per case, over
native-Reverse RK gradients (`rk_logdensity_problem` for correlated random
effects, an HSGP with a modeled log-scale, a logit GLM and an `@plate for`
population-PK cell; the NLME batched, compact-cell and η-Hessian entry points).
It is known broken: Enzyme's reverse pass corrupts memory when another thread
runs GC, measured on Julia 1.10 and 1.12 — the same upstream defect, reproduced
without BRM by a plain `@rkppl` random-effects model (Enzyme snag
`julia-1-10-concu-152a4415`; `reactivekernels-use` §7ao). It then records one
`@test_broken` without stressing; `BRM_STRESS_KNOWN_BROKEN=1` runs the stress,
which is the BRM acceptance check for that upstream fix.

`stanblocks_preservation_corpus.jl` compares fourteen representative models'
emitted SLIC/Stan, prepared data, metadata, and frozen replay against an external
baseline artifact. Capture the artifact on the implementation base, then use
that same file when checking the refactor:

```sh
BRM_SB_PRESERVATION=write BRM_SB_PRESERVATION_FILE="$TMPDIR/brm-sb.bin" \
  julia --project=test test/stanblocks_preservation_corpus.jl
BRM_SB_PRESERVATION=check BRM_SB_PRESERVATION_FILE="$TMPDIR/brm-sb.bin" \
  julia --project=test test/stanblocks_preservation_corpus.jl
```

The artifact is intentionally untracked. The comparison normalizes only the
absolute test-source path embedded by StanBlocks; it retains model text and
parameter identities.

## One-time bootstrap

Use `test/setup_env.jl` as the single bootstrap entry:

```sh
julia --project=test test/setup_env.jl
```

The script materializes every external dependency under the ignored
`test/.bootstrap/` directory at an exact full-SHA revision. ReactiveKernels
contributes its monorepo root **and** the nested
`ReactiveKernelsDistributionKernels` and `ReactiveKernelsPPL` packages from
that same checkout; those nested packages have no standalone repositories. All
ten develop paths enter one resolve on Julia 1.10, where `[sources]` in
`test/Project.toml` is ignored. A dirty shared checkout is never used as the
source of truth: an exact revision is checked out deliberately, so the
environment does not drift with local branches.

That writes `test/Manifest.toml`, which is deliberately **not** committed (the
root `.gitignore` covers `Manifest*.toml`). Re-run `setup_env.jl` on a new
machine, after changing a pin, or when an existing ignored manifest still points
at an older dependency checkout.

`BridgeStan` needs the BridgeStan C++ sources in addition to the Julia package.
`BridgeStan.jl` finds them via `$BRIDGESTAN`, falling back to
`~/.bridgestan/bridgestan-<version>`, and downloads them if neither exists.

The grouped Turing parity benchmark compares the same BRMI and constrained
parameter point through StanBlocks/BridgeStan and Turing/Enzyme. Pass a case
name to keep a hardening run focused; the centered-geometry gate is:

```sh
julia --project=test test/benchmark_turing_backend_groups.jl \
  centered_correlated_poisson
```

It checks constrained density and mapped gradients before reporting warmed,
allocation-aware construction, density, and gradient timings. Gradient setup
is outside the hot loop: the harness prepares Enzyme once with
`DifferentiationInterface.prepare_gradient`, reuses a caller-owned gradient
buffer, and times only repeated `value_and_gradient!` calls. Sampling is omitted
because this gate targets the lowering and density kernels.

The multi-membership Turing parity benchmark exercises one weighted correlated
`mm(...)` block through the direct Turing plan and StanBlocks emission:

```sh
julia --project=test test/benchmark_turing_multi_membership.jl
```

It checks constrained density and mapped Enzyme/BridgeStan gradients before
reporting warmed, allocation-aware construction, density, and gradient timings.
Like the grouped and crossed-group harnesses, it prepares Enzyme once, reuses a
caller-owned gradient buffer, and measures the repeated hot call separately
from construction. Sampling is omitted because this focused gate targets
lowering and density kernels rather than sampler behavior.

The dual-HSGP gradient benchmark exercises the source-faithful 133-row
motorcycle model through generated Turing/DynamicPPL+Enzyme and compiled
StanBlocks/BridgeStan targets:

```sh
BRM_HSGP_BENCH_OUTPUT=/tmp/turing-hsgp-gradient.tsv \
  julia --project=test test/benchmark_turing_hsgp_gradients.jl
```

It prepares AD once, separates compilation/setup from hot calls, pins BLAS to
one thread, and interleaves 21 batches of 1,000 value-and-gradient calls. By
default it checks a deterministic physical point in fixed NCP, centered, and
per-basis partial coordinates plus all three online source frames. Set
`BRM_HSGP_BENCH_DRAW_DIR` to an extracted source-faithful posterior bundle to
also check columns 1, 2500, 5000, 7500, and 10000 from the original 10,000-draw
Stan fits. The TSV records normalized-density, absolute and scaled gradient,
finite-difference, allocation, runtime-ratio, model/draw hash, and exact
dependency provenance fields. Sampling is intentionally absent: this is the
numerical/runtime gate that must pass before native Turing sampling.
The committed source-faithful k=20 measurement is
`test/receipts/turing_hsgp_gradients.tsv`; its `brm_revision` and
`benchmark_sha256` columns bind every row to the exact measured implementation
and harness.
A committed `k=8` noncentered pilot is
`test/receipts/turing_hsgp_gradients_k8_noncentered.tsv`.

## Why each constraint exists

Every one of these was paid for by a failed resolve; none is stylistic.

- **`WarmupHMC` must be a develop path, never a registry resolve.** The
  registry only carries WarmupHMC 0.1.x, and the root `Project.toml` bounds it
  at `WarmupHMC = "0.2"` — so a registry resolve is unsatisfiable, not merely
  stale. `[sources]` cannot fix this either: it is a Julia 1.11+ feature and is
  silently ignored on 1.10, which is this package's compat floor. The pinned
  revision must contain WarmupHMC `9c642178720d5c294b9cead86fc8c82da5a5db09`:
  that floor retains Pathfinder's use of the target's own
  `logdensity_and_gradient` and admits Pathfinder 0.10.7, the first registered
  release compatible with the test environment's Turing 0.46. Older and newer
  checkouts all report version 0.2.1, so the version number cannot express
  this. `setup_env.jl` resolves an exact published revision rather than a
  shared checkout's possibly stale local branch.
- **`StanBlocks` must be a checkout, not a release.** BRM does not precompile
  against registered StanBlocks; it fails inside a `@deffun` in `src/sbimpl.jl`.
  Configured `gp` / `hsgp` term priors additionally require StanBlocks
  `10529af04d42a330df383864059c2b61a11d9480` or later. Older checkouts trace
  the spliced term model before its keyword data are bound and fail on an
  unresolved `omega2`. StanBlocks is unregistered and both sides of this floor
  report version `0.1.5`, so verify the checkout SHA rather than its version.
  `hsgp(x; cov=:periodic, period=…)` additionally needs StanBlocks
  `bec23bc3c52303ebde60a026af48c435e4c81330` or later, which registers the
  `log_modified_bessel_first_kind` builtin its spectral weights call (older
  checkouts fail at transpile with `Could not find
  log_modified_bessel_first_kind …`); `test/gp_hsgp_periodic.jl` is the gate.
  Sampled values used only in another parameter's support additionally require
  `67767349bb1e9b05a50fdd1c05e0a8cf878e43e5` or later (landed in
  `e2b2fa952ebbd550016ff6cb85d11b0f4482aba1`). Earlier activity analysis drops
  those dependencies; `test/hierarchical_prior_bounds.jl` exercises this case.
- **`Treebars` is here even though no test uses it.** It is an unregistered
  *transitive* dependency of WarmupHMC, which pins it with a `[sources]` entry —
  ignored on 1.10, same as above. Without a path the resolve fails outright with
  `Treebars [e1e568c4] has no known versions!`. `Pkg.develop` can only fix a path
  for a *direct* dependency, so Treebars has to be listed in `test/Project.toml`
  as well; that entry is transitive plumbing, not a test dependency.
- **`Enzyme_jll` 0.0.302 is excluded.** That binary, which Enzyme 0.13.211
  requires, can segfault (signal 11) in `getindex` inside the reverse pass on
  Julia 1.10 when a differentiated function returns a freshly allocated array,
  such as an index-vector gather
  ([EnzymeAD/Enzyme.jl#3776](https://github.com/EnzymeAD/Enzyme.jl/issues/3776);
  the fix ships in a later `Enzyme_jll`). Native Reverse gradients that read a
  grouped effect as `draws[indices, margin]` crash with it; measured on the
  public random-effect block of `rk_coordinate_transport.jl`,
  `rk_retained_transport.jl`, `rk_modeled_random_slopes.jl` and five
  `rk_values.jl` testsets, all of which pass on 0.0.301. `Enzyme_jll` is a
  direct dependency here only to carry `Enzyme_jll = "< 0.0.302, >= 0.0.303"`,
  which resolves Enzyme 0.13.210 with `Enzyme_jll` 0.0.301 and admits the next
  binary without another edit; like Treebars, it is resolution plumbing, not a
  test dependency. A manifest resolved before this entry still holds 0.0.302:
  re-run `setup_env.jl`.
- **All eight `develop` paths go in ONE `Pkg.develop` call.** Resolution has to
  satisfy them together: the unregistered BRM root, the four external
  source pins, and the three nested ReactiveKernels packages. Developing
  StanBlocks by itself fails with `expected package
  BayesianRegressionModels to be registered`, while omitting a source-only or
  nested package produces the same error for that dependency.
- **Serial extension precompilation is limited to Julia 1.10.0–1.10.11.**
  `setup_env.jl` builds Pathfinder's sibling Turing extensions once with
  `JULIA_NUM_PRECOMPILE_TASKS=1` on those runtimes, then runs the normal
  parallel pass. Julia 1.10.12 includes the Pkg `loadable_exts` fix and skips
  the serial phase, as do newer Julia versions. Setup suppresses instantiate's
  automatic precompile and explicitly runs `Pkg.precompile()` once after
  resolution. Host CPU and memory budgets still apply. See the `precompile`
  skill for the current runtime policy and independent stall diagnostics.

## Why not `Pkg.test`

`Pkg.test()` cannot work here, so there is deliberately no `test/runtests.jl`
and no `[extras]`/`[targets]` in the root `Project.toml`:

- `Enzyme` and `WarmupHMC` are root `[weakdeps]`. `Pkg.test`'s sandbox resolves
  them from the registry, which lands on WarmupHMC 0.1.x and violates the
  `"0.2"` bound above. The only fixes are a committed absolute develop path or
  `[sources]`, and neither is available.
- A `test/Project.toml` already takes precedence over `[extras]`/`[targets]` on
  every supported Julia version, so carrying both would be two declarations of
  one dependency list.

Six files are the reason this environment exists — they fail at their own
`using` line under `julia --project=.`, before any BRM code runs:

| file | needs beyond the root project |
| --- | --- |
| `test/benchmark_turing_multi_membership.jl` | `BridgeStan`, `StanBlocks`, `Enzyme`, `DifferentiationInterface` |
| `test/benchmark_turing_hsgp_gradients.jl` | `BridgeStan`, `StanBlocks`, `Turing`, `WarmupHMC`, `Enzyme`, `DifferentiationInterface` |
| `test/adaptive_centering_bridgestan.jl` | `WarmupHMC`, `Enzyme` |
| `test/adaptive_centering_warmuphmc.jl` | `WarmupHMC`, `Enzyme`, `DifferentiationInterface` |
| `test/turing_adaptive_centering_warmuphmc.jl` | `Turing`, `WarmupHMC`, `Enzyme`, `DifferentiationInterface` |
| `test/plate_stress.jl` | `BridgeStan` |

There is no CI workflow for these on purpose: they need `stanc` and a BridgeStan
toolchain, so a GitHub Actions job would be red by construction.
# Prior-regime fitted geometry: `rk_prior_regimes.jl` retains the authoritative
# generative snapshot for omitted responses while native fitted execution has
# zero coordinates. Bound baseline observations keep their inference ancestors.
# Controls cover normalized compiled same-BRMI Stan, ordinary native Reverse,
# full source/artifact replay and input ownership; native generated draws are
# unavailable.
