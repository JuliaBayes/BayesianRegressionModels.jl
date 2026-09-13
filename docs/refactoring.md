# Backend preparation architecture

BayesianRegressionModels lowers an `@brm` declaration in two stages. The first
stage builds a backend-neutral source graph. It records data columns, sampled
parameters, deterministic assignments, predictors, observations, dependencies,
and the row shape of each value. The second stage supplies backend geometry:
StanBlocks emits SLIC declarations and data, while the Turing extension builds
native Julia submodels and distributions.

This separation is deliberate. The source graph decides what an expression
means and when it can be evaluated. A backend decides how the corresponding
parameters are represented, constrained, and sampled. Backend code must not
select models through a list of admitted likelihood families or term names.
Ordinary callable expressions are retained in the prepared graph and invoked by
the generated backend program. Specialized geometry is selected by dispatch on
a prepared term or response type.

Both backends consume one ordered semantic source program for expression
identities, dependencies, response evidence, and resolved prior addresses.
Shared fit/apply helpers also define fitted transforms used by both paths. Each
backend still prepares the geometry its executor requires: StanBlocks owns SLIC
declarations, group prepasses, and named Stan data, while Turing builds
population, term, and group geometry for DynamicPPL sample sites.

The Turing emitter passes its generated body through DynamicPPL's own model
compiler, then stores the evaluator as a `RuntimeGeneratedFunction`. This small
compiler dependency makes a newly constructed model callable immediately from
compiled Julia functions. Density and gradient evaluation do not use
`invokelatest`. The public model source remains available through
`turing_model_source(backend)`.

## Prepared expressions and shapes

Prepared expressions contain callable objects rather than reconstructed names.
Their arguments contain constants, references to data or earlier operations,
and other prepared expressions. Dependency ordering is computed before backend
generation, including dependencies introduced by coefficient, group, and term
priors. This permits priors such as `Normal(sampled_hyperparameter, 1)` to be
constructed inside the generated Turing model, where the sampled value exists.

Every reference also carries its observation, event, group, or scalar axis.
Shape hooks normalize vector-valued observations and let arbitrary Julia
distribution constructors participate without identifying their family by
name. New callable distributions and deterministic functions generally need no
compiler change. A genuinely new parameter geometry should add a typed prepared
record and backend methods.

When a function constructs a distribution, declare its result shape with
`brm_distribution_type(::typeof(my_prior)) = Normal` (or the appropriate
distribution type). This distinguishes a sampled declaration from a predictor
formula without evaluating a factory whose arguments may be sampled values.
Turing retains the factory call. StanBlocks additionally needs its normal
distribution translation hook; the shape trait does not assert that arbitrary
Julia code has a Stan implementation.
Use `_sb_stan_distribution_call(constructor, args, kwargs)` when that translation
needs the complete call. It receives lowered argument expressions and returns
a Stan family-call AST, shared by priors, observations, and their composition.
This is where a Julia factory's keyword semantics can be rewritten to Stan
arguments; SLIC sampling keywords describe declarations rather than arbitrary
Julia constructor keywords.

## Fitted preparation and replay

Data-dependent transformations are split into fit and apply operations. Fitted
state includes categorical levels, centering and scaling constants, spline
bases, HSGP domains and frequencies, and structured-term group levels. Both
backends consume these records.

`reprocess` applies the training record to new data before any fresh fit can
replace it. This keeps spline knots, HSGP domains, categorical contrasts, and
group indices aligned with posterior parameters. Extrapolation is allowed when
the fitted transform defines it. Values outside an explicit HSGP domain and
previously unseen categorical or group levels fail with a diagnostic because
the fitted model has no corresponding basis or parameter. Rebuilding rather
than replaying is the operation that derives new fitted constants.

Term replay dispatches on the callable object itself (`typeof(s)`, `typeof(gp)`,
or an extension callable), rather than reconstructing a symbol with `nameof`.
The same dispatch contract handles replay from a fresh expression and from an
already prepared plan. This keeps extension methods open: a new term can own
fitted state and replay it by defining a method for its callable type. The
shared fit/apply operation produces transformed values and updated fitted state;
StanBlocks retains responsibility for binding those values to emitter-specific
data names.

## Term priors and constrained geometry

Term-prior statements are resolved once during backend-neutral preparation.
The resolver matches predictor and term addresses, applies the same precedence
rules for both backends, and rejects misspelled, unsupported, or ambiguous
addresses before emission. Backends consume the resolved semantic slots instead
of independently searching the formula.

Simplex-valued term parameters preserve the configured multivariate prior. The
Turing adapter supplies simplex constrained geometry through the ordinary
simplex bijector while delegating density, support, and random generation to the
original distribution. Consequently a custom continuous multivariate prior is
not approximated as a Dirichlet merely to obtain a transform. It must have the
declared simplex dimension and implement valid density, support, and sampling
behavior. StanBlocks support remains conditional on having a corresponding Stan
translation for that distribution call.

## Specialized terms

The shared preparation layer currently covers population transformations,
categorical and interaction columns, `s`, `t2`, `me`, `mo`, `mo1`, interval
censoring predictors, `ar`, `dar`, exact GP, HSGP, and structured group fields.
GP/HSGP preparation includes isotropic and anisotropic kernels, one-dimensional
isotropic periodic bases, group-specific HSGP weights, explicit domains,
model-derived axes, and frozen replay. Periodic HSGP terms do not currently
combine with group-specific weights. Model-derived HSGP axes require one
isotropic axis and an explicit domain; they do not currently support periodic
bases or grouping. Term-owned priors expose stable semantic roles and are
constructed by the generated backend program.

The Turing extension implements these prepared terms as native submodels. A
structured term declares one or more normalized fields and supplies a
callable-specific effect method; common samplers provide correlated normal, IID
normal, and ordinary callable field priors. This preserves the public
StanBlocks structured-field hooks for downstream SLIC emitters.

## Responses and backend boundaries

The generic response path supports scalar and vector-valued callable
distributions, multiple responses, observation modifiers, weights, missing-data
plans, and the repository's typed ordinal and joint-response compositions.
Special response shapes use dispatch and prepared shape metadata. They do not
form a family admission list.

Turing emits single- and multiple-response models through one response-graph
emitter. It schedules shared parameters, assignments, predictors, group blocks,
and observations once in dependency order, then attaches response-specific
likelihood, evidence, weight, missing-row, and predictive operations. This is
why a declaration shared by two responses is sampled once rather than copied
into two generated model bodies.

The descriptor adapter exposes this semantic plan through the same
`brm_descriptor`, `brm_output`, `brm_outputs`, and `brm_execute` APIs used for
Stan-backed descriptors. Turing descriptors report semantic parameters, linear
predictors, posterior-predictive values, pointwise log likelihoods, and replay
operations. Their `stan` field is `nothing`, and Stan source highlights are not
available because no Stan program exists.

SLIC bodies remain StanBlocks programs. The native backend does not interpret
arbitrary SLIC syntax as Julia. A term that only defines a StanBlocks emission
hook therefore continues to work with `SBBRMI` and produces an explicit
"no native Julia implementation" diagnostic with `TuringBRMI`. To support both
backends, keep the SLIC emitter and add a native effect or submodel method for
the same shared prepared record.
