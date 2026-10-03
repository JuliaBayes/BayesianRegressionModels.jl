# Scientific model descriptions

[`brm_description`](@ref) describes a selected, prepared StanBlocks model.
Reuse its [`BRMDescriptor`](@ref); the description reads its parsed formula,
prepared design, fitted preprocessing and generative declarations. It returns
structured prose, LaTeX, notation, effective priors and explicit coverage.
It does not fit, sample or instantiate an executable model.

This generic hierarchical model has a Gaussian observation distribution, a
standardized covariate and a shared intercept/slope covariance block:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
description_model = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + zscale(x) + (1 + x | shared | group)
    effect(mu, zscale_x) ~ Normal(0, 0.4)
    y ~ Normal(mu, sigma)
end)((;
    x=[-2.0, -0.5, 0.7, 2.5, 3.0, 4.2],
    group=[1, 1, 2, 2, 3, 3],
    y=[0.4, 0.1, 1.1, 1.7, 2.4, 2.8],
))
""", :description_model; title="Model to describe", require_stan=true, total_groups=())
```

```@eval
let d = brm_descriptor(SBBRMI(description_model; total_groups=())),
    anchors = Dict((:population, :mu, :zscale_x) => "#the-description-result")
    result = brm_description(d; prior_anchors=anchors)
    result.complete || error(join(result.diagnostics, "\n"))
    Markdown.parse(brm_description_markdown(result))
end
```

## The description result

```julia
d = brm_descriptor(prepared)
result = brm_description(d;
    labels=Dict(:mu => (; meaning="conditional mean", symbol="\\mu")),
    prior_anchors=Dict((:population, :mu, :x) => "#slope-prior"),
)
markdown = brm_description_markdown(result)
```

`result.model_id` is the selected descriptor's inference identity. Report labels,
units, symbols, hooks and links are presentation metadata. They do not alter
source, data, draws or inference caches. Units and biological meanings must be
supplied; a name such as `log_CL` does not establish a log transformation.

`components` is an ordered recursive inventory. Each component has an exact
logical tuple `id`, `kind`, actual `callable`, `arguments`, `keywords`, `children`,
`axes`, `outputs`, `priors`, `fitted_constants`, `bindings`, `notation` and
`provenance`. Use `brm_description_components(result)` to traverse it.
Arguments contain literals, [`BRMDescriptionReference`](@ref) values, tuples
and recursive components. Keyword arguments are a named tuple. Collections
and fitted values are read-only snapshots.
Vectors use tuples; matrices and higher-dimensional arrays use
`(size=..., values=...)`, with values in Julia's column-major order.

Output records expose `name`, `logical`, `role`, `kind` and `segments`.
`logical` is a symbol or `nothing`; component and prior IDs are tuples.
Ragged `segments` preserve the descriptor's inclusive grouping ends. Axes
and fitted constants retain their declared source and fitted level/basis data.
Conditioned, held-out and unconditioned observations remain distinct.

`priors` contains [`BRMPriorDescription`](@ref) records with `id`, `distribution`,
`support`, `source`, `anchor` and `provenance`. The distribution retains its actual
callable, arguments and keywords. Winning selector metadata comes from the
producer's precedence resolver. Generated latent priors, scalar hyperpriors,
positive scales and improper unpenalized coefficients are inventoried too.
Gaussian equations square the SD argument; Exponential scale and rate
parameterizations are stated explicitly.
Affine Student-t observations retain their location and scale: for degrees of
freedom ν > 2, their SD is scale × √(ν/(ν−2)). Scale is not their SD.
Declared support bounds describe the prepared parameter constraints; the
inventory retains the actual density factor on that support.

The Markdown renderer creates a stable target for each row of its complete
effective-prior table and links component prior references to those targets.
`brm_description_prior_anchor(result_or_context, prior.id)` returns the target ID;
`brm_description_prior_references(context)` returns the related logical prior IDs.
Targets are scoped to `model_id`. For repeated instances of one model artifact,
pass a unique `prefix` to both `brm_description_markdown` and the anchor helper.
This also applies when separate data or fitted constants share one source
identity; the adapter should give each model mount its own prefix.
The optional `prior_anchors` map adds outbound links to targets owned by the
calling report; it does not create those external targets.
Unobserved declarations that remain sampled parameters retain their actual
conditional density in this inventory, including parameter-dependent families.
Their contexts use `observation_role=:latent_parameter`; the prose identifies
the sampled quantity and its contribution to the joint model.

## Scientific extensions

An unknown callable leaves a coverage gap. Add a reusable
[`brm_describe_component`](@ref) method, or pass request-local hooks keyed by
actual callable identity:

```julia
hook = context -> BRMDescriptionFragment(
    prose=("The response is a quadratic function of its input.",),
    equations=("q=" * brm_description_math(context, only(context.arguments)) *
               "^2+0.6",),
    covers=(context.id,),
    notation=((; name=:q, meaning="quadratic response", axis=:observation),),
)
result = brm_description(d; hooks=(response_curve => hook,))
```

For this hook, the scientific callable's body is `response_curve(x) = x^2 + 0.6`.
Its actual model registration and units belong to the consumer. Equations are
LaTeX without display delimiters. Notation records have `name`, `meaning` and
optional `symbol`, `axis`, `unit`. Explicit symbols are trusted LaTeX; default
identifiers are escaped by [`brm_description_symbol`](@ref) and
[`brm_description_math`](@ref).

[`brm_description_binding`](@ref)`(context, :offset)` resolves a submodel's
authored internal parameter. Its record contains `name`, `role`, `value`,
`prior_ids` and, in an included submodel, `path`. Render `value` with
`brm_description_math`; use [`brm_description_prior`](@ref) on each exact prior
ID. Repeated nested names can be addressed by an authored tuple path.
Kernel lambda aliases point to actual model arguments. No generated prefix or
declaration-order assumption is needed.
Reference-valued keywords use the same math API, including identity aliases.
Kernel equations show the grouped input mapping; scientific child hooks provide
their recurrences without inserting lambda or block syntax into the equation.
Cell assignments emit separate indexing and arithmetic relations, and the
mapping connects to the cell's returned quantity. Cell-local names are scoped
to their kernel owner (for example, `pred.state`) unless notation overrides them.
The compact grouped map symbol refers to separately defined exact input
bindings. Each authored cell alias has a short symbol and its own binding
relation. Long record arguments receive short symbols with aligned definitions
of all their named fields. Long call assignments and additive population
relations use aligned lines. Wide transformed designs use named prepared coordinates, with their
fitted centering/scaling definitions and logical column mappings listed
separately; values and scientific child coverage are preserved.
Long likelihood arguments use compact intermediate symbols with separate exact
definitions and notation tied to their source component IDs. Distribution
conventions, input bindings and child coverage are preserved.
Long opaque observation-family relations place each actual argument on its own
aligned row, in authored order, including keyword bindings. The family still
needs independent scientific hook coverage.
Authored literal assignments have `role=:constant` and expose their value;
composed assignments have `role=:deterministic` and expose a public expression.
This includes assignments preceded by documentation strings in an included
SlicModel. Supplied data arguments retain their prepared bindings. A supplied
keyword wins over an authored fixed default in that same included scope,
including documented defaults; nested scopes remain distinct.
References to deterministic quantities render as symbols; render the binding's
`value` explicitly to expand its defining expression.
Actual higher-order callable arguments retain their identity and render through
the same escaped math helper.
Statically named records such as `(; lower, upper, weights=exp(offset))`
retain public named field values. Use
[`brm_description_record`](@ref)`(context, argument)` to resolve a record
argument through its authoritative local bindings. The returned NamedTuple
contains literal values, parameter/data references and expression components;
render individual fields with `brm_description_math`. Unresolved, cyclic or
non-record arguments fail rather than guessing a scope or evaluating code.

Included-model inputs backed by known prepared arrays retain
`BRMDescriptionReference` values rather than anonymous numeric tuples. Generated
prepared inputs have logical IDs `(:prepared_data, key)`; their math uses the
bound data symbol. This preserves observation, mask and transformed-input
identities without printing the dataset into an equation. Actual literal prior
shapes and limits still retain their numeric values. Static global callable
bindings retain their actual identity without traversing Julia's binding objects.
The lowered `dims` call describes the array shape; indexing selects an axis
extent. Lowered `max` and `min` describe vector reductions or scalar extrema.
Categorical fitted values retain their labels without pool or pointer state.
Embedded likelihood contexts include the response's actual descriptor outputs,
including its posterior-predictive quantity and selected conditioning role.
Conditioning follows prepared response provenance through ragged input aliases
and joint-completion carriers. A partially observed vector has
`observation_role=:partially_observed`, `observation_sources`, and exact
`observed_entries`/`missing_entries` counts; observed entries stay fixed.
Fresh modeled covariate replay uses `observation_role=:covariate_draw` with
`generated_entries` counting the newly drawn values. Every selected prediction
row is drawn conditional on retained model parameters; it adds no sampled
missing coordinates. Multivariate Gaussian Cholesky families retain the actual
lower covariance factor and describe covariance as its product with its transpose.
Joint covariance factors retain their actual marginal-scale and LKJ prior IDs.
The joint completion's allocation-only declaration has constant log density;
its substantive density comes from the declared multivariate model.
Array conversions, matrix replication, ranges, transposition and joint-column
selection retain their actual callable identities and explicit semantics.
Vector/matrix conversion order follows the
[Stan mixed operations reference](https://mc-stan.org/docs/functions-reference/mixed_operations.html).

Covariance blocks use compact, distinct numbers within each result. The notation
entries map each number to its full logical ID and ordered margins; predictor
designs and covariance factors use that same number. Logical IDs and prior
anchors keep their exact identities. The prior table uses short P1, P2 row keys
and names the distribution family. Complete logical IDs, distribution equations
and individual support facts follow in full-width definitions. Component prior
links use those same P keys; their C keys have separate complete logical paths.
Notation uses individual metadata entries, preserving supplied meanings and
units without repeating an unsupplied identifier as its meaning. The original
prior anchors remain in their table cells; exact logical IDs remain in the
public prior records. These presentation keys do not change logical identity.
GP axis and multiple-membership weight references use inline math in prose.

Included scientific SlicModels expose their parameter declarations,
deterministic expressions and nested calls as children. A top-level caption
does not cover those children. `covers` lists exact IDs in the hook's own
subtree; a hook must explicitly explain every scientific subtree it claims.
Claims outside that subtree or for nonexistent IDs raise an error.

`result.complete` certifies inventory coverage. The consumer remains responsible
for independent scientific review of its handwritten hooks. Unsupported calls
and syntax retain precise `coverage` records and `diagnostics`; the Markdown
renderer labels such a result **incomplete**.
Population Horseshoe allocation nodes currently retain explicit gaps. Their
Gaussian conditionals do not establish coverage of the allocation.

## R2D2 allocation conventions

Whole-predictor `effect(mu, :) ~ r2d2(...)` and shared-block
`sd(:, id) ~ r2d2(...)` use different scale conventions. Descriptions bind their
actual R² prior, simplex concentration, share indices and reference scales from
the prepared emitter; these quantities also appear in the effective prior
inventory.

For the whole-predictor form, T is the total latent scale, Vⱼ is the sample
variance of prepared design column j, and k(j) is its selected simplex share:

$$
s_j=T\sqrt{\phi_{k(j)}R^2/V_j},\qquad
\operatorname{SD}_{\mathrm{group}}=T\sqrt{1-R^2}.
$$

The intercept and population columns with their own explicit priors stay
outside the allocation. With no non-intercept columns, no R² or simplex is
introduced and the group SD is T. A one-component simplex equals [1].

The shared-block R2D2M2/ICC form uses each margin's reference scale rₘ:

$$
\operatorname{SD}_m=r_m\sqrt{\phi_{k(m)}R^2/(1-R^2)}.
$$

With `include=(:population, :contrasts)`, the same budget additionally allocates
population and contrast scales. Each predictor uses its own selected margin
reference; its total-scale expression is r/√(1−R²). Its population and contrast
scales therefore also contain R²/(1−R²), divided by their prepared design-column
variance. Treatment indicators use V=m(n−m)/(n(n−1)), for m selected rows among
n rows. Completed covariates stay inside their model-dependent variance.
Margins outside a selected budget retain their separately inventoried free
scale priors. Correlations keep their declared priors; allocated marginal SDs
do not receive an invented independent SD prior.
