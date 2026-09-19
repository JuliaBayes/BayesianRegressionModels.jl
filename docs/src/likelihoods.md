# Likelihoods

## Complete built-in catalogue

This is the exhaustive public likelihood catalogue for the default
[`SBBRMI`](@ref) Stan backend. A `Distributions.jl` subtype not named here is
not accepted merely because it is a distribution: BRM rejects it until its
Julia constructor has an explicit Stan mapping. The pure-Julia [`VBRMI`](@ref)
backend is independent and does not implement every specialized family or
response wrapper below.

### Direct `Distributions.jl` families

| Outcome | Accepted constructors |
| --- | --- |
| Continuous | `Normal`, `NormalCanon`, `Cauchy`, `TDist`, `Logistic`, `Gumbel`, `Chisq`, `Exponential`, `Gamma`, `Erlang`, `Beta`, `Uniform`, `LogNormal`, `Laplace`, `Frechet`, `Rayleigh`, `SkewNormal`, `Pareto`, `Weibull`, `InverseGamma`, `InverseGaussian`, `VonMises` |
| Continuous, restricted parameterization | `Arcsine()` — the standard `[0, 1]` form only; `SkewedExponentialPower(mu, sigma, 1, alpha)` — only the literal shape `1` |
| Discrete | `Bernoulli`, `BernoulliLogit`, `Binomial`, `BinomialLogit`, `BetaBinomial`, `Poisson`, `NegativeBinomial` |

BRM normalizes constructor conventions where Julia and Stan differ. In
particular, `Exponential`, `Gamma`, and `Erlang` use scale in
`Distributions.jl` but rate in Stan; `Pareto(shape, scale)` is reordered to
Stan's `(minimum, shape)` convention; `NormalCanon(eta, lambda)` becomes
`normal(eta / lambda, inv(sqrt(lambda)))`; `NegativeBinomial(r, p)` is
translated to Stan's shape/inverse-scale form; `TDist(nu)` becomes
`student_t(nu, 0, 1)`; and `Laplace` becomes Stan's `double_exponential`.
The standard `Arcsine()` is exactly `beta(1/2, 1/2)`. Shifted/scaled
`Arcsine(a, b)` constructors need a Jacobian-aware custom implementation and
are rejected rather than silently treated as a standard beta likelihood.

### BRM families and structured likelihoods

| Constructor | Meaning / boundary |
| --- | --- |
| `SkewDoubleExponential(mu, sigma, tau)` | Stan-native asymmetric-Laplace parameterization |
| `LocationScale(mu, sigma, TDist(nu))` | location-scale Student-t regression |
| `ZeroInflatedPoisson(lambda, zi)` | zero-inflated Poisson |
| `HurdlePoisson(lambda, p_zero)` | hurdle Poisson with zero-truncated positive component |
| `NegativeBinomial2(mu, phi)` | mean/precision negative binomial |
| `BetaBinomial2(n, mean, precision)` | mean/precision beta-binomial |
| `CategoricalLogit(eta2, eta3, ...)` or `CategoricalLogit(@brm(...))` | reference-class categorical logit |
| `OrderedLogistic(eta)` | legacy cumulative-logit ordinal model |
| `Ordinal(structure, link, eta; ...)` | `Cumulative()` or `StoppingRatio()` crossed with `LogitLink()`, `ProbitLink()`, or `CloglogLink()` |
| `CircularVonMises(mu, kappa; interval=(-pi, pi))` | von Mises on a fixed principal interval |
| `TruncatedNormal(mu, sigma, lower, upper)` | legacy censored-Normal marker; new models should use `censored` below |
| `[y1, y2, ...] ~ MvNormalCholesky([mu1, mu2, ...], L)` | row-wise correlated Gaussian outcomes using a declared `LKJCovarianceFactor` |
| `MixtureModel([D, ...], weights)` | finite mixture over K same-family scalar components |

### Response compositions and modifiers

| Form | Exact supported surface |
| --- | --- |
| `truncated(d; lower, upper)` | base family `Normal`, `LogNormal`, `Exponential`, `Weibull`, or `Poisson` |
| `censored(d; lower, upper)` | the same five base families |
| `interval_censored(d; upper)` | the same five base families; the response is the lower endpoint |
| `weighted(d, aweights(w))` | analytic/precision weights for `Normal` only |
| `weighted(d, fweights(w))` | frequency weights for direct mapped families in the first table |
| `weighted(d, weights(w))` | power-likelihood weights for direct mapped families in the first table |
| `mi(y) ~ d` | partly-missing continuous response imputation |

Every specialized family is expected to supply the fitted density, pointwise
log likelihood, and posterior-predictive RNG used by BRM's generated
quantities. Truncation and censoring additionally require matching CDF/CCDF
paths; ragged observations additionally require a sized RNG.

## Hurdle counts with `HurdlePoisson`

Use `HurdlePoisson(lambda, p_zero)` when zeros arise from a separate hurdle
and every count from the Poisson component is strictly positive:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
using LogExpFunctions: logit

hurdle_counts = (@brm begin
    log(lambda) ~ 1 + x
    logit(p_zero) ~ 1 + factor(group)
    y ~ HurdlePoisson(lambda, p_zero)
end)((;
    x=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
    group=[1, 1, 2, 2, 3, 3],
    y=[0, 1, 0, 3, 2, 5],
))
""", :hurdle_counts; title="Hurdle-Poisson counts")
```

Here `p_zero` is exactly ``P(Y=0)``. Conditional on crossing the hurdle,
positive observations follow a zero-truncated Poisson distribution:

```math
P(Y=y \mid Y>0) =
\frac{\operatorname{Poisson}(y \mid \lambda)}{1-e^{-\lambda}},
\qquad y=1,2,\ldots
```

This differs from `ZeroInflatedPoisson(lambda, zi)`: zero inflation mixes a
structural-zero component with an ordinary Poisson component, so both mixture
components can produce zeros. `HurdlePoisson` is also an executable
Distributions.jl distribution with matching `params`, `logpdf`, and `rand`
semantics outside a formula. It requires finite `lambda > 0` and
`0 <= p_zero <= 1`; BRM supplies no implicit link or prior.

## Wald regression with `InverseGaussian`

Use Distributions.jl's `InverseGaussian(mu, lambda)` for positive continuous
outcomes whose variance grows with the cube of the mean — the Wald GLM with a
log link:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
wald_costs = (@brm begin
    log(lambda) ~ 1
    eta ~ 1 + x
    y ~ InverseGaussian(exp(eta), lambda)
end)((;
    x=[-1.0, -0.5, 0.0, 0.5, 1.0],
    y=[1.2, 0.8, 1.1, 2.0, 1.6],
))
""", :wald_costs; title="Log-link Wald costs")
```

This preserves the constructor order `(mu, lambda)` — already Stan's order, so
no parameterization translation applies — the shorthand
`InverseGaussian(mu) == InverseGaussian(mu, 1)`, and the strict `y > 0`,
`mu > 0`, `lambda > 0` domain. Density, pointwise log likelihood, and
predictive RNG all use the same closed form; BRM supplies no implicit link or
prior.

## Correlated Gaussian outcomes

Use one ordered vector likelihood when several measurements from the same row
have experimental residual covariance that should be estimated rather than
treated as independent:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
correlated_measurements = (@brm begin
    L_res ~ LKJCovarianceFactor(
        2; scale_prior=Exponential(1), shape=2,
    )
    shared_log_rate ~ Normal(0, 1)
    concentration_mu = exp(-exp(shared_log_rate) * time)
    response_mu = 1.0 - concentration_mu
    [concentration, response] ~ MvNormalCholesky(
        [concentration_mu, response_mu], L_res)
end)((;
    time=[0.0, 1.0, 2.0, 4.0],
    concentration=[1.0, 0.72, 0.51, 0.27],
    response=[0.03, 0.18, 0.43, 0.79],
))
""", :correlated_measurements; title="Estimated experimental covariance")
```

`LKJCovarianceFactor(K; scale_prior=Exponential(1), shape=1)` samples `K`
positive marginal scales and an LKJ Cholesky correlation factor, then returns
`diag_pre_multiply(scales, L_corr)`. The likelihood lowers to Stan's native
`multi_normal_cholesky`: each aligned data row contributes one joint scalar
log likelihood and one ordered predictive vector. The example deliberately
uses one sampled rate in both mechanistic means; shared parameters need no
special syntax beyond ordinary formula-block assignments.

This first surface is deliberately strict. Every outcome and mean has the same
row axis, the number of means and factor dimension must match the left-hand
side, and outcome rows must be complete, finite, and nonempty. A missing value
is rejected; BRM never silently drops the row or replaces the joint density
with conditionally independent pieces. Correlated outcomes are currently an
[`SBBRMI`](@ref)-only feature.

## Finite mixtures with `MixtureModel`

Use Distributions.jl's `MixtureModel` when each observation comes from one of
`K` latent subpopulations with its own parameters:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
mixture_gaussians = (@brm begin
    mu1 ~ Normal(-2, 0.1)
    mu2 ~ Normal(2, 0.1)
    log(sigma) ~ 1
    y ~ MixtureModel([
        Normal(mu1, exp(log(sigma))),
        Normal(mu2, exp(log(sigma))),
    ], [0.4, 0.6])
end)((;
    y=[-2.0, -1.8, 1.9, 2.2],
))
""", :mixture_gaussians; title="Two-component Gaussian mixture")
```

Each row contributes `log_sum_exp(log(weights) + component_lpdf)`, with
matching pointwise log likelihoods and posterior-predictive draws (select a
component per row, then draw from it). The contract is deliberately narrow.
Every component must be a scalar (`Univariate`) call of ONE Julia family:
heterogeneous families are rejected because Stan's discrete densities throw
on out-of-support values where Turing returns `-Inf`, so mixed-support
mixtures would crash Stan where Turing stays finite. The family must be
directly Stan-mapped, or `NegativeBinomial2` / `BetaBinomial2` (which are
native translations); bespoke families, `Uniform` / `Pareto`
(parameter-dependent continuous support), and `Categorical` (simplex
parameters) are rejected. Binomial-family components must share one
identical trial-count expression. Weights are a numeric vector summing to 1,
a length-`K` numeric data column, or a Dirichlet-backed simplex parameter.

## Adding another likelihood

Yes—when Stan already has the distribution, adding it is usually small. BRM
needs an explicit Julia-type-to-Stan-name entry, plus an argument translation
when the two libraries use different parameterizations. It then inherits the
ordinary model, pointwise-log-likelihood, and predictive-RNG paths from
StanBlocks.

The work becomes larger when Stan has no native family: the implementation
must provide and test a density/mass function, a pointwise companion, and an
RNG. Supporting `truncated` or `censored` also needs the relevant CDF/CCDF
functions, and ragged responses need a vector-sized RNG. These are finite
implementation tasks, not an architectural prohibition; the explicit gates
prevent a family from appearing to fit while prediction or likelihood
diagnostics silently mean something else.

## Truncation and censoring

BRM preserves the standard Distributions.jl RHS composition for mathematical
truncation and threshold censoring:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
bounded_evidence = (@brm begin
    mu ~ 1 + x
    log(sigma) ~ 1

    y_truncated ~ truncated(Normal(mu, sigma); lower=0.0, upper=2.0)
    y_clamped ~ censored(LogNormal(mu, sigma); lower=0.25, upper=1.8)

    # Genuine interval evidence: y_lower stores the open lower endpoint and
    # y_upper stores the closed upper endpoint.
    y_lower ~ interval_censored(Normal(mu, sigma); upper=y_upper)
end)((;
    x=[-1.0, 0.0, 1.0],
    y_truncated=[0.2, 0.8, 1.4],
    y_clamped=[0.25, 0.9, 1.8],
    y_lower=[-0.4, 0.1, 0.8],
    y_upper=[-0.1, 0.4, 1.2],
))
""", :bounded_evidence; title="Truncated, censored, and interval evidence")
```

These are three different likelihood contracts:

- `truncated(d; lower, upper)` conditions `d` on the inclusive bounds and
  predicts from that conditional distribution;
- `censored(d; lower, upper)` is the distribution of
  `clamp(X, lower, upper)` and predicts clamped values;
- `interval_censored(d; upper)` contributes
  `log(CDF(upper) - CDF(response))` for the genuine interval observation
  `(response, upper]`, while prediction remains on the uncoarsened base scale.

The same marker has a separate predictor-side form,
`interval_censored(x; upper=lloq)`, for a quantified/BLOQ covariate. It
allocates bounded latent predictor values on rows where `x == lloq` rather than
changing a response likelihood; see [Interval-censored predictor](@ref).

Bounds may be numeric literals or observed row-wise columns. The initial
family-gated surface covers `Normal`, `LogNormal`, `Exponential`, `Weibull`,
and `Poisson`; BRM rejects other base families until their aggregate density,
pointwise likelihood, CDF/CCDF, generated prediction, and stanc paths are all
tested. BRM's eager two-sided bound check accepts `lower <= upper`, while the
StanBlocks producer requires a non-degenerate interval with `lower < upper`;
equal bounds are therefore rejected during Stan lowering.

This composition is implemented only by the `SBBRMI` Stan backend. **Do not
use `VBRMI` for these formulas:** it currently does not reject `truncated` or
`censored` at construction and can return log densities for a misinterpreted
model; `interval_censored` may fail only when the density is evaluated. A
`VBRMI` result is therefore not a valid cross-check of an `SBBRMI` fit. The
legacy `TruncatedNormal` marker remains a separate censored-Normal
compatibility surface.

The backend compatibility floor for this surface is StanBlocks
`0eaebfae904d3bffab150dfa2c59632ac783b992`, where the public distribution-HOF
tokens became `truncated`, `censored`, and `interval_censored` with no aliases.
BRM revisions containing this lowering must be co-pinned with that StanBlocks
commit or later; the preceding StanBlocks `9b879d5e` expects the older internal
token spellings and is intentionally incompatible.

## Concise categorical regression

`CategoricalLogit` accepts an explicit nested `@brm(...)` predictor formula:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
categorical_data = (;
    x = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    y = ["b", "a", "c", "b", "c", "a"],
)

categorical_model = @brm categorical_data begin
    y ~ CategoricalLogit(@brm(1 + x))
end
""", :categorical_model; title="Nested categorical-logit formula")
```

For an outcome with ``K`` levels, BRM expands the marked formula to ``K-1``
ordinary scalar linear predictors with distinct coefficients, then calls the
same reference-class categorical lowering as the fully explicit form. The
first fitted level has logit zero. Plain vectors use `sort(unique(y))` for the
fitted order; a `CategoricalVector` uses its declared level order. The latter is
the way to select a reference level deliberately.

For example, a three-level outcome above is equivalent in model structure to:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
explicit_data = (;
    x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    y=["b", "a", "c", "b", "c", "a"],
)
explicit_model = @brm explicit_data begin
    y_nested_arg1_class2 ~ 1 + x
    y_nested_arg1_class3 ~ 1 + x
    y ~ CategoricalLogit(y_nested_arg1_class2, y_nested_arg1_class3)
end
""", :explicit_model; title="Explicit categorical-logit predictors")
```

The generated names are deterministic implementation names; use the explicit
form when those predictor names are part of another formula. Only nested
`@brm(...)` opts into predictor-formula interpretation. Thus
`CategoricalLogit(1 + x)` remains an ordinary expression and is rejected by
the categorical backend, rather than silently acquiring coefficients.

The marker is not categorical-specific. It selects formula interpretation at
one family-argument position while surrounding expressions retain their usual
meaning. For example, a distributional Normal model can make both predictors
concise while keeping the positive scale link explicit:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
distributional_data = (; x=[-1.0, 0.0, 1.0], y=[-0.2, 0.3, 1.1])
distributional_model = @brm distributional_data begin
    y ~ Normal(@brm(1 + x), exp(@brm(1)))
end
""", :distributional_model; title="Nested distributional predictors")
```

This introduces distinct scalar predictors for location and log-scale, then
passes `exp(log_scale)` to `Normal`; nested `@brm` never inserts a link. A
standalone fragment such as `@brm(1 + x)` is not yet a first-class value and
produces a targeted error outside an enclosing model.

This is the same broad model structure expressed by a top-level categorical
formula in brms (`y ~ 1 + x`, `family = categorical(link = "logit")`) or Bambi
(`"y ~ 1 + x"`, `family="categorical"`). Defaults for priors, contrasts, and
reference-level selection are package-specific; BRM does not import those
defaults implicitly.

## Ordinal structure and link composition

BRM treats the ordinal probability construction and inverse link as separate
typed choices. A cumulative probit model is:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
ordinal_data = (; x=[-1.0, -0.5, 0.0, 0.5, 1.0], y=[1, 1, 2, 3, 3])
ordinal_model = @brm ordinal_data begin
    eta ~ 0 + x
    y ~ Ordinal(Cumulative(), ProbitLink(), eta)
end
""", :ordinal_model; title="Ordinal cumulative-probit model")
```

The accepted structures are `Cumulative()` and `StoppingRatio()`. Each composes
with `LogitLink()`, `ProbitLink()`, or `CloglogLink()`. This is intentionally a
Julia-native typed surface: BRM does not copy R formula helper names or encode
every structure/link pair in a new family type.

For `Cumulative()`, BRM estimates strictly ordered thresholds ``c_1 < \cdots <
c_{K-1}`` and uses

```math
P(Y \le k) = F\!\left(d(c_k - \eta)\right).
```

For `StoppingRatio()`, the estimated stage intercepts need not be ordered and

```math
q_k = P(Y=k \mid Y\ge k)
    = F\!\left(d(c_k - \eta_k)\right),\qquad
P(Y=k)=q_k\prod_{j<k}(1-q_j),
```

with the final category equal to the probability of continuing through every
stage. `F` is logistic, standard normal, or complementary-log-log according to
the link tag. Both threshold vectors currently receive element-wise standard
normal priors.

The thresholds already supply the model location, so the composed surface
requires an intercept-free common predictor (`eta ~ 0 + ...`). They count as
that predictor's intercept: a categorical term of `eta` keeps its K−1 treatment
contrasts rather than the cell means an intercept-free predictor otherwise gets
("Cell means" on the [overview page](index.md)). A positive discrimination parameter
is explicit, and its predictor is *not* threshold-located — switch its cell-mean
coding off with `cmc=false` (brms' name for it) so that one group's
discrimination stays pinned at `exp(0) = 1`, which is what identifies the scale
against the free thresholds:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
ordinal_disc_data = (;
    x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    group=[1, 2, 1, 2, 1, 2], y=[1, 1, 2, 2, 3, 3],
)
ordinal_disc = @brm ordinal_disc_data begin
    eta ~ 0 + x
    log(disc) ~ 0 + factor(group; cmc=false)
    y ~ Ordinal(Cumulative(), ProbitLink(), eta;
                discrimination=disc)
end
""", :ordinal_disc; title="Ordinal discrimination model")
```

`discrimination` defaults to one. Literal or data-supplied values are checked
for finiteness and strict positivity; a modeled value should use a
positive-support prior or a link such as `log(disc) ~ ...`.

Stopping-ratio models may add non-proportional effects with a tuple of raw
numeric predictors:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
sequential_data = (;
    period=[1, 2, 3, 1, 2, 3], carry=[0, 0, 1, 0, 1, 1],
    treat=[1, 2, 1, 2, 1, 2], y=[1, 1, 2, 2, 3, 3],
)
sequential = @brm sequential_data begin
    eta ~ 0 + period + carry
    y ~ Ordinal(StoppingRatio(), CloglogLink(), eta;
                per_threshold=(treat,))
end
""", :sequential; title="Sequential ordinal model")
```

BRM estimates one coefficient per predictor and non-terminal stage, so here
``\eta_k = \eta + \mathtt{treat}\,\beta_k``. `per_threshold` is deliberately
restricted to stopping-ratio models for now: unrestricted cumulative
category-specific effects can make cumulative probabilities non-monotone.
The predictors must currently be raw numeric data columns.

Outcome categories follow the declared order of a `CategoricalVector`; plain
vectors use sorted unique values. That fitted order is frozen for replay and
prediction. The legacy `OrderedLogistic(eta)` spelling remains supported and
continues to lower directly to Stan's native ordered-logistic distribution.
The composed cumulative-logit kernel also delegates its scalar density to that
native primitive; the other links use Stan's native stable CDF/log-CDF
functions. Stopping ratio has no native Stan distribution, so BRM supplies the
matching stable lpmf, pointwise log-likelihood, and RNG.

Outside `@brm`, `Ordinal(structure, link, eta, thresholds;
discrimination=1)` is an executable `DiscreteUnivariateDistribution` with
`params`, `probs`, `logpdf`, and `rand`. A stopping-ratio `eta` may be scalar or
a vector with one stage-specific value per threshold.

For neutral comparison, brms exposes the same statistical axes through
families such as cumulative-probit and stopping-ratio complementary-log-log,
and calls threshold-varying terms category-specific effects. BRM preserves
that statistical contract while using the typed composition and
`per_threshold=(...)` tuple above rather than importing brms's R formula
helpers.

## Median regression with `Laplace`

The StanBlocks backend accepts `Distributions.Laplace` as an ordinary
likelihood. For example, a robust regression for the conditional median can be
written as:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
laplace_data = (;
    x = [-1.0, -0.5, 0.0, 0.5, 1.0],
    y = [-1.1, -0.2, 0.1, 0.6, 1.4],
)

median_model = @brm laplace_data begin
    median_y ~ 1 + x
    log(laplace_scale) ~ 1
    y ~ Laplace(median_y, laplace_scale)
end
""", :median_model; title="Median regression with Laplace")
```

This lowers to Stan's native
`y ~ double_exponential(median_y, laplace_scale)`. The second argument is a
**Laplace scale**, not a standard deviation or rate:

```math
f(y \mid \mu, \theta)
= \frac{1}{2\theta}\exp\!\left(-\frac{|y-\mu|}{\theta}\right),
\qquad \theta = \mathtt{laplace\_scale}.
```

This symmetric likelihood is exactly the ``q = 0.5`` special case of the
asymmetric-Laplace likelihood used for quantile regression, after accounting
for parameterization:

- In the check-loss convention used by `brms`, with scale ``s`` and density
  ``q(1-q)s^{-1}\exp[-\rho_q((y-\mu)/s)]``, use
  ``\theta = 2s`` at ``q = 0.5``.
- In the Bambi/PyMC convention
  `AsymmetricLaplace(mu, b, kappa)`, where
  ``\kappa = \sqrt{q/(1-q)}``, use ``\kappa = 1`` and
  ``\theta = 1/b`` at ``q = 0.5``.

The `Laplace` spelling covers median regression only. It does **not** express
an asymmetric likelihood for ``q \ne 0.5``. It also does not translate
Bambi's historical `bs(age, knots=...)` term: that basis mapping is a separate
unresolved formula-semantic question. Thus the response-family component of
the catalogue's `quantile_p50` model is available, while the complete
historical model remains unsupported.

## Quantile regression with `SkewDoubleExponential`

For a non-median quantile, BRM exposes the executable distribution
`SkewDoubleExponential(mu, sigma, tau)`. Its arguments and scale exactly match
Stan's native `skew_double_exponential` family:

```math
f(y \mid \mu, \sigma, \tau)
= \frac{2\tau(1-\tau)}{\sigma}
  \exp\!\left[-\frac{2}{\sigma}
  \left((1-\tau)\mathbf{1}_{y<\mu}(\mu-y)
        +\tau\mathbf{1}_{y>\mu}(y-\mu)\right)\right].
```

Thus `cdf(SkewDoubleExponential(mu, sigma, tau), mu) == tau`, and
`SkewDoubleExponential(mu, sigma, 0.5)` is exactly `Laplace(mu, sigma)`.
There is no hidden brms-scale conversion on this primary Julia surface.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
quantile_data = (;
    x=[-1.0, -0.5, 0.0, 0.5, 1.0],
    y=[-1.1, -0.2, 0.1, 0.6, 1.4],
)
quantile_model = @brm quantile_data begin
    q25_y ~ 1 + x
    log(native_scale) ~ 1
    y ~ SkewDoubleExponential(q25_y, native_scale, 0.25)
end
""", :quantile_model; title="Non-median quantile regression")
```

The brms/check-loss scale ``s`` translates explicitly as
``\sigma = 2s``. Distributions.jl's existing exact special case also remains
available in formulas:

```julia
using Distributions: SkewedExponentialPower

y ~ SkewedExponentialPower(mu, sigma_sepd, 1, tau)
```

BRM lowers that spelling with
``\sigma = 4\,\mathtt{sigma\_sepd}\,\tau(1-\tau)``. The shape must be the
literal value `1`; other `SkewedExponentialPower` shapes are rejected because
Stan's asymmetric double-exponential family is not a native implementation of
the general SEPD. Density, pointwise log likelihood, and predictive RNG all
use the same translation.

## Circular regression with `VonMises`

BRM exposes two deliberately different von-Mises likelihoods. Use
Distributions.jl's `VonMises` when its exact Julia contract is intended:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
von_mises_data = (;
    x=[-1.0, -0.5, 0.0, 0.5, 1.0],
    direction=[-0.8, -0.2, 0.1, 0.5, 0.9],
)
exact_model = @brm von_mises_data begin
    mu ~ 1 + x
    log(kappa) ~ 1
    direction ~ VonMises(mu, kappa)
end
""", :exact_model; title="Moving-support von Mises")
```

This preserves the constructor order `(mu, kappa)`, the shorthand
`VonMises(kappa) == VonMises(0, kappa)`, strict `kappa > 0`, and the moving
closed support `[mu - pi, mu + pi]`. The backend adds those support/domain
guards around Stan's native `von_mises_lpdf`, and recenters native predictive
draws onto the same moving interval. Because the support moves with `mu`, this
surface is usually not the right choice for observations encoded once on a
fixed principal interval.

For conventional circular regression on a fixed interval, use the distinct
BRM distribution `CircularVonMises`:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
circular_data = (;
    x=[-1.0, -0.5, 0.0, 0.5, 1.0],
    direction=[-0.8, -0.2, 0.1, 0.5, 0.9],
)
circular_model = @brm circular_data begin
    mu ~ 1 + x
    log(kappa) ~ 1
    direction ~ CircularVonMises(mu, kappa; interval=(-pi, pi))
end
""", :circular_model; title="Fixed-interval circular regression")
```

`interval` is a compile-time pair of finite numbers with length `2pi`; it
defaults to `(-pi, pi)`. Observations must lie in the half-open interval
`[lo, hi)`. BRM wraps `mu` and generated draws into that interval, while the
density itself remains Stan's native `von_mises_lpdf`. Both arguments are
ordinary distributional parameters: BRM supplies no implicit link or prior.
Outside a formula, `CircularVonMises(mu, kappa; interval=...)` is an executable
`ContinuousUnivariateDistribution`: `params`, `logpdf`, and `rand` preserve the
same fixed-interval contract used by the Stan lowering.

For comparison, `brms` uses a fixed `(-pi, pi)` response convention and makes
both `mu` and `kappa` distributional, with default `tan_half` and `log` links.
Those are a useful neutral baseline, but BRM requires links and priors to be
written explicitly rather than silently changing Distributions.jl semantics.
In particular, `Distributions.Gamma` takes a **scale**, whereas Stan/brms gamma
syntax takes a **rate**: the brms prior `gamma(2, 0.01)` is spelled
`Gamma(2, 100.0)` in a BRM formula.
## Typed observation weights

Observation weights live in the `@brm` model beside the observation
distribution:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
weighted_model = (@brm begin
    y ~ weighted(Normal(mu, sigma), aweights(replicate_k))
    mu ~ 1 + x
    log(sigma) ~ 1
end)((;
    x=[-1.0, 0.0, 1.0], y=[-0.2, 0.3, 1.1],
    replicate_k=[1.0, 2.0, 4.0],
))
""", :weighted_model; title="Analytic observation weights")
```

The StatsBase constructor determines the statistical meaning:

- `aweights(k)` uses analytic/precision semantics. For a Normal response BRM
  emits `Normal(mu, sigma / sqrt(k))`; model density, pointwise likelihood, and
  predictive draws all use that adjusted scale.
- `fweights(n)` uses frequency/repeat semantics. BRM multiplies each model and
  pointwise log-likelihood contribution by `n`; predictive draws remain from
  the original distribution.
- `weights(w)` opts into a power likelihood with the same density/pointwise
  scaling and unchanged predictive distribution.

The current analytic-weight implementation supports Normal observations.
Frequency and power weights support likelihood families that lower through
BRM's native Distributions.jl-to-Stan family map. Probability weights and
unsupported family/type combinations error instead of silently changing
meaning. Weight columns are rebuilt from each dataframe by the reusable
`@brm` builder and by `reprocess`.
