# CDC `ww-inference-model`: an executable structural port

This page maps the four coupled components of CDC's
[`ww-inference-model`](https://github.com/CDCgov/ww-inference-model) onto the
`@brm` formula surface: infection renewal, subpopulation variation, wastewater
measurements, and interval-valued count streams. It is an executable **structural port**, not
a drop-in or numerically equivalent reproduction of the current CDC model. The
comparison below is verified through transpilation, `stanc`, and a finite
BridgeStan density/gradient by `test/cdc_ww_inference.jl`; those gates establish a
working model artifact, not posterior parity with CDC.

For a smaller introduction to the renewal and shedding pieces, start with the
[single-catchment wastewater example](wastewater.md). A parallel StanBlocks-native
case study is available in the
[StanBlocks documentation](https://nsiccha.github.io/StanBlocks.jl/dev/examples/case-studies/wastewater).

## The current model on the `@brm` formula surface

This is a genuinely multi-level, multi-stream renewal model. Its composition is on
the formula surface, while the sequential numerical kernels are ordinary
`@deffun`s. The seams are:

- **Shared reference log-Rᵘ** — `log_ru_week ~ 1 + dar(week_grid; p=1)` is
  expanded onto the daily latent axis and closed over inside each
  per-subpopulation cell.
- **Latent subpopulation hierarchy** — `I_mat ~ kernel(t_grid, is_reference,
  logit_I0, initial_growth) do … end` broadcasts only the renewal process. Each
  cell draws an AR(1) deviation and returns its infection trajectory; the uncovered
  reference cell suppresses that deviation.
- **Infection feedback and renewal** — `renewal_feedback` performs the carried-state
  scan inside each cell.
- **Sparse wastewater measurements** — `ww_expected_log` gathers arbitrary
  `(time, subpopulation, lab)` records from the collected latent matrix. The
  left-censored log-normal likelihood uses a separate lab hierarchy and one LOD per
  record, so a latent subpopulation may have no wastewater records and several labs
  may observe the same catchment/time pair.
- **Jurisdiction aggregation** — `wsum(I_mat, w)` forms a population-weighted
  infection trajectory.
- **Count streams** — `count_interval_mean` maps each stream to arbitrary latent
  subpopulation weights, a delay PMF, a population multiplier, and a stationary
  weekly logit-rate trajectory. Observation rows may be daily or inclusive
  multi-day intervals; a mean-one simplex weekday effect and negative-binomial
  likelihood finish the component.

The carried-state scans cannot be written as formula statements or `kernel`
control flow. `wsum` similarly owns the cross-cell matrix multiplication because
`*` at formula level is the Wilkinson interaction operator. These are the
interesting numerical kernels, so their exact checked-in Julia definitions are
shown rather than left as opaque calls:

```@eval
Main.BRMDocsComparisons.source_code_region(
    "research/wastewater/cdc_ww_inference.jl";
    starting_at="StanBlocks.@deffun begin",
    ending_before="\"\"\"\n    cdc_ww_inference_fixture",
)
```

The setup below evaluates those same definitions and the fixture before the
four-pane comparison is built.

```@eval
Main.BRMDocsComparisons.evaluate_source_prelude(
    Main.BRMDocsComparisons.example_module(:cdc_ww),
    "research/wastewater/cdc_ww_inference.jl";
    before=:cdc_ww_brm_model,
    starting_at="StanBlocks.@deffun begin",
)
nothing
```

The declaration is extracted verbatim from `cdc_ww_brm_model`; the StanBlocks and
Stan panes are generated from the resulting `BRMI` during the docs build.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:cdc_ww),
    Main.BRMDocsComparisons.source_function(
        "research/wastewater/cdc_ww_inference.jl", :cdc_ww_brm_model,
    ),
    :cdc_ww_brm_model;
    title="CDC ww-inference structural port (@brm formula surface)",
    require_stan=true,
)
```

The Turing pane is retained even though this structural kernel is outside the
current Turing executor. Its construction error documents that backend boundary.

## What matches, and what does not yet match

The example preserves the high-level causal order from CDC's
[`model_definition.md`](https://github.com/CDCgov/ww-inference-model/blob/main/model_definition.md):

- **Transmission:** unadjusted reproduction numbers drive feedback-adjusted renewal incidence.
- **Subpopulations:** trajectories vary around a shared process and aggregate by
  population weight.
- **Wastewater:** shedding-convolved subpopulation incidence drives censored wastewater
  measurements.
- **Counts:** delay-convolved mapped incidence drives weekday-adjusted interval counts.

The current fixture now includes the CDC model's distinct latent and observation
axes: a 50-day unobserved period, an uncovered reference population, sparse and
repeated site/lab/time records, record-specific detection limits, lab-level scale
and noise effects, a normalized inferred triangular shedding trajectory, hierarchical
first-observed incidence and growth with CDC's seeding back-calculation, bounded
stationary subpopulation AR deviations, the weekly differenced-AR reference process,
and a mean-one simplex weekday multiplier.

It does **not** yet reproduce these defining CDC details:

- CDC's exact mean-reverting IHR parameterization;
- the exact upstream hyperprior values supplied by CDC's R interface; or
- CDC's component switches.

The executable upstream
[`wwinference.stan`](https://github.com/CDCgov/ww-inference-model/blob/main/inst/stan/wwinference.stan)
is authoritative for those details. “Structural port” here means that the four
components are coupled in the same causal order; it does not mean that their
parameterizations, data axes, or priors are identical.

## Relation to the StanBlocks form

The reproduction file also carries a pure-StanBlocks `@slic` companion,
`cdc_ww_inference_model`. It uses the differenced-AR helper directly and broadcasts
subpopulation parameters with `plate`. The `@brm` `kernel` term serves the same
sampling role: parameters such as subpopulation innovations and initial incidence
are introduced per cell, while `@deffun` owns only deterministic recurrence.

Both forms use the same differenced-AR global R recurrence. The companion shares
the remaining simplifications listed above and is not presented as a numerical
reference implementation.

## BRM authoring gap removed by this port

This example exposed that the ordinary `ar` term could not represent CDC's
differenced-AR weekly process. BRM snag `brm-formula-diff-52808dea` added the direct
summand `dar(week_grid; p=1)`: the term owns bounded persistence, innovation scale,
and standardized innovations, while the formula intercept supplies the initial
level. Replay and descriptor semantics now derive from that term as they do for
other formula components. The example uses it directly; the remaining differences
above are model-port scope rather than this former authoring limitation.

## Composable counts and forecasts

The count likelihood is not tied to a column called `hosp` or to one row per day.
The fixture carries one row per observed interval:

- `count_start`, `count_stop`, and `count_obs_stream` define the record axis;
- `count_subpop_weights[:, stream]` maps the latent subpopulation matrix into that
  stream's catchment;
- `count_delay[:, stream]` supplies its infection-to-observation delay; and
- `count_population[stream]` supplies the absolute-count scale.

The default fixture deliberately crosses two shapes: daily jurisdiction hospital
admissions and weekly totals over only the sampled catchments. Adding another count
source is therefore a data extension—append a stream column and its records—rather
than a second hand-written likelihood.

Forecast mappings use the same deterministic functions but are not responses in the
calibration likelihood. The emitted Stan descriptor owns three named generated
quantities: `forecast_infections`, `forecast_count_mean`, and
`forecast_ww_log_mean`. The reproduction exposes both layers without a parallel
registry:

```julia
plan = cdc_ww_brm_plan(df)          # declarations and replay provenance
descriptor = cdc_ww_brm_descriptor(df)
```

The descriptor derives `fit`, `predict`, `pointwise_loglik`, `replay`, and
`reprocess` from the executable model. Consumers should resolve the named forecast
outputs from `descriptor.outputs`; they should not parse generated Stan identifiers
or infer them from declaration order.

## Provenance

Reproduction:
[`research/wastewater/cdc_ww_inference.jl`](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/wastewater/cdc_ww_inference.jl),
gated by `test/cdc_ww_inference.jl`. Both the `@slic` and `@brm` forms are checked
for transpilation and `stanc`; the default fixtures are also checked for a finite
BridgeStan density and gradient. The discretized kernels and priors remain
illustrative. The models are not sampled here and the gate is not a
posterior-equivalence test against CDC.

## Run the reproduction

After bootstrapping the repository's test environment:

```sh
julia --startup-file=no --project=test test/cdc_ww_inference.jl
```

Set `BRM_KERNEL_RUNTIME=0` to run lowering and `stanc` without BridgeStan
instantiation.
