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

## Chunking heavy suites

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
oracles and compiled same-BRMI Stan. Complete printed source and artifact
re-entry are included; a scalar native-location control is checked separately.

`spline_basis_signs.jl` checks canonical TPS/t2 projection signs, input
ownership, and frozen projections directly. The spline blocks in
`rk_parity.jl` compare RK densities and Enzyme gradients with independent
oracles; `spline_parity_models.jl` holds their shared BridgeStan fixtures.
`spline_sb_parity.jl` compiles those fixtures independently and checks the
full-posterior anchors and gradients, including the distributional spline toy.

The focused preparation gates are `preparation_program.jl`,
`preparation_replay.jl`, `preparation_assignments.jl`, and `backend_plan.jl`.
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
isolation, stable shared-group source, and non-mutating Julianic lowering of
shared input syntax. Run it in a fresh process with `julia --threads=4 --project=test
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
an emitted entry and a bound leaf cannot share the same symbol.

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
source: sampled names, physical coordinates, prior/likelihood/full densities,
all standard Reverse coordinates, caller ownership and complete source replay
must agree bit-exactly. Priors remain explicit at their existing paths.
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

`adaptive_hsgp_centering.jl` checks pilot-selection under spectral underflow,
the partial-coordinate Jacobian, Turing/Enzyme versus StanBlocks/BridgeStan
density and gradient parity, physical constrained quantities, and the
distributional model's two distinct zero-mean HSGP bindings.

`adaptive_bounded_scales.jl` checks that online adaptive centering reads
`Uniform(a, b)`-bounded HSGP length scales and SDs, and cdar SDs and
persistences, through Stan's own `lb`/`lub` transform. Because the wrapper's
density identities hold for any function of those coordinates, each of its
four compiled models first compares the physical value every cell reads with
BridgeStan's `param_constrain`, then checks the wrapper identities.

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

The native-PPL milestone gates are separate on purpose:

```sh
julia --project=test test/native_ppl.jl
julia --project=test test/native_ppl_warmuphmc.jl
julia --project=test test/native_ppl_backend_parity.jl
```

The parity file compiles the unchanged substantive BRMI through `SBBRMI` and
also compiles `test/native_ppl_shared_distributional_mixed.stan`, its
hand-optimized minimal Stan analogue. It checks normalized density and mapped
gradients before printing warmed density/gradient allocation and timing rows;
those timings include the BridgeStan FFI crossing but exclude compilation and
model/data construction.

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
  silently ignored on 1.10, which is this package's compat floor. Native PPL
  sampling additionally requires WarmupHMC
  `9c642178720d5c294b9cead86fc8c82da5a5db09` or later. That floor retains
  Pathfinder's use of the target's own `logdensity_and_gradient` and admits
  Pathfinder 0.10.7, the first registered release compatible with the test
  environment's Turing 0.46. `setup_env.jl` and the focused sampler test
  enforce this ancestry because older and newer checkouts all report version
  0.2.1. `setup_env.jl` resolves an exact published revision rather than a
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
- **The other three `[sources]` packages must be develop paths too.**
  `MutatingFunctions`, `OutputSignatures`, and `TreeArrays` are unregistered
  direct dependencies. On Julia 1.10 their committed source pins are inert, so
  omitting their paths fails with `expected package ... to be registered`.
- **All ten `develop` paths go in ONE `Pkg.develop` call.** Resolution has to
  satisfy them together: the unregistered BRM root, the seven external
  source pins, and the two nested ReactiveKernels packages. Developing
  StanBlocks by itself fails with `expected package
  BayesianRegressionModels to be registered`, while omitting a source-only or
  nested package produces the same error for that dependency.
- **Pathfinder's Turing extension pair is precompiled serially first.** Both
  `setup_env.jl` suppresses the parallel auto-precompile that
  `Pkg.instantiate()` performs, build `Pkg.precompile(["Pathfinder", "Turing"])`
  once under `JULIA_NUM_PRECOMPILE_TASKS=1`, then run the ordinary parallel
  `Pkg.precompile()`. This is not stylistic — it is the same class of Pkg 1.10
  self-deadlock the `MutatingFunctions` pin note above avoids, for a pair we do
  not control. Pathfinder 0.10.7 ships two sibling Turing extensions,
  `PathfinderTuringExt` (triggers `AbstractMCMC`, `Accessors`, `DynamicPPL`,
  `Turing`) and `PathfinderTuringFlexiChainsExt` (triggers `FlexiChains`,
  `Turing`), and Turing 0.46 hard-depends on every one of those triggers. In a
  parallel precompile each extension gets its own `--output-ji` worker; each
  worker loads Turing, which loads the *other* extension's triggers, so each
  worker then blocks on the pidfile the other worker's driver holds. Pkg 1.10
  never forwards `loadable_exts` to the worker
  (`Pkg/src/precompilation.jl:869-871`, the kwarg is commented out of the
  `Base.compilecache` call), so nothing breaks the mutual wait and only the host
  reaper clears it, ~25 min later. Julia 1.12's `Base.Precompilation` forwards
  `loadable_exts`, so this is specific to 1.10 — this package's compat floor. One
  serial build lands a `.ji` valid for both siblings (the `PathfinderTuringExt`
  worker builds the FlexiChains sibling nested), after which the parallel pass
  finds them cached and never spawns those workers. Both names are required
  because Pkg 1.10 keeps an extension in a named precompile only when its full
  trigger set is inside the named closure (`precompilation.jl:610`), and naming
  `Turing` pulls in every trigger of both extensions.

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

Eight files are the reason this environment exists — they fail at their own
`using` line under `julia --project=.`, before any BRM code runs:

| file | needs beyond the root project |
| --- | --- |
| `test/benchmark_turing_multi_membership.jl` | `BridgeStan`, `StanBlocks`, `Enzyme`, `DifferentiationInterface` |
| `test/benchmark_turing_hsgp_gradients.jl` | `BridgeStan`, `StanBlocks`, `Turing`, `WarmupHMC`, `Enzyme`, `DifferentiationInterface` |
| `test/adaptive_centering_bridgestan.jl` | `WarmupHMC`, `Enzyme` |
| `test/adaptive_centering_warmuphmc.jl` | `WarmupHMC`, `Enzyme`, `DifferentiationInterface` |
| `test/turing_adaptive_centering_warmuphmc.jl` | `Turing`, `WarmupHMC`, `Enzyme`, `DifferentiationInterface` |
| `test/native_ppl_backend_parity.jl` | `BridgeStan`, `StanBlocks`, `Enzyme`, `DifferentiationInterface` |
| `test/native_ppl_warmuphmc.jl` | `WarmupHMC`, `Enzyme`, `DifferentiationInterface` |
| `test/plate_stress.jl` | `BridgeStan` |

There is no CI workflow for these on purpose: they need `stanc` and a BridgeStan
toolchain, so a GitHub Actions job would be red by construction.
# Prior-regime fitted geometry: `rk_prior_regimes.jl` retains the authoritative
# generative snapshot for omitted responses while native fitted execution has
# zero coordinates. Bound baseline observations keep their inference ancestors.
# Controls cover normalized compiled same-BRMI Stan, ordinary native Reverse,
# full source/artifact replay and input ownership; native generated draws are
# unavailable.
