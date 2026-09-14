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

## Shared preparation and Turing lowering

The focused preparation gates are `preparation_program.jl`,
`preparation_replay.jl`, `preparation_assignments.jl`, and `backend_plan.jl`.
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

`adaptive_hsgp_centering.jl` checks pilot-selection under spectral underflow,
the partial-coordinate Jacobian, Turing/Enzyme versus StanBlocks/BridgeStan
density and gradient parity, physical constrained quantities, and the
distributional model's two distinct zero-mean HSGP bindings.

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

Seven packages have to enter resolution as **develop paths**, and absolute
paths are machine-specific, so they are not committed. Supply StanBlocks and
Treebars once:

```sh
BRM_TEST_STANBLOCKS=/path/to/StanBlocks.jl \
BRM_TEST_TREEBARS=/path/to/Treebars.jl \
  julia --project=test test/bootstrap.jl
```

`BRM_TEST_WARMUPHMC=/path/to/WarmupHMC.jl` is an optional override. A supplied
checkout is used only when its `HEAD` contains the enforced NativePPL floor. If
it is unset or stale, `bootstrap.jl` materializes an ignored, versioned checkout
under `test/.bootstrap/`, preferring the host mirror's `dev`/immutable
`refs/kb-pins/<sha>` and otherwise cloning public `origin/dev`. This is
deliberate: a dirty shared `~/github/nsiccha/WarmupHMC.jl` checkout may be
hundreds of commits behind even though the floor is landed and published.
`BRM_TEST_WARMUPHMC_MIRROR` and `BRM_TEST_WARMUPHMC_ORIGIN` override those two
sources for an offline or nonstandard host.

The other three source-only direct dependencies — `MutatingFunctions`,
`OutputSignatures`, and `TreeArrays` — are materialized under the same ignored
directory at the exact full-SHA revisions in `test/Project.toml`. Julia 1.10
ignores those `[sources]` entries, so `bootstrap.jl` reads the committed table
itself and includes their paths in the single resolve.

That writes `test/Manifest.toml`, which is deliberately **not** committed (the
root `.gitignore` covers `Manifest*.toml`). Re-run `bootstrap.jl` after moving
a checkout, on a new machine, or when an existing ignored manifest still
points at an older dependency checkout.

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
  environment's Turing 0.46. `test/bootstrap.jl` and the focused sampler test
  enforce this ancestry because older and newer checkouts all report version
  0.2.1. Bootstrap resolves the floor from public `origin/dev` or the host
  mirror's immutable pin, not from a shared checkout's possibly stale local
  branch.
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
- **All seven `develop` paths go in ONE `Pkg.develop` call.** Resolution has to
  satisfy them together. Developing StanBlocks by itself fails with `expected
  package BayesianRegressionModels to be registered`, while omitting a
  source-only direct dependency produces the same error for that dependency.
- **Pathfinder's Turing extension pair is precompiled serially first.** Both
  `bootstrap.jl` and `setup_env.jl` suppress the parallel auto-precompile that
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
