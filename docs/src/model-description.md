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

The Markdown renderer creates a stable target for each row of its complete
effective-prior table and links component prior references to those targets.
`brm_description_prior_anchor(result_or_context, prior.id)` returns the target ID;
`brm_description_prior_references(context)` returns the related logical prior IDs.
Targets are scoped to `model_id`. For repeated instances of one model artifact,
pass a unique `prefix` to both `brm_description_markdown` and the anchor helper.
The optional `prior_anchors` map adds outbound links to targets owned by the
calling report; it does not create those external targets.

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
Authored literal assignments have `role=:constant` and expose their value;
composed assignments have `role=:deterministic` and expose a public expression.
Actual higher-order callable arguments retain their identity and render through
the same escaped math helper.

Included scientific SlicModels expose their parameter declarations,
deterministic expressions and nested calls as children. A top-level caption
does not cover those children. `covers` lists exact IDs in the hook's own
subtree; a hook must explicitly explain every scientific subtree it claims.
Claims outside that subtree or for nonexistent IDs raise an error.

`result.complete` certifies inventory coverage. The consumer remains responsible
for independent scientific review of its handwritten hooks. Unsupported calls
and syntax retain precise `coverage` records and `diagnostics`; the Markdown
renderer labels such a result **incomplete**.
