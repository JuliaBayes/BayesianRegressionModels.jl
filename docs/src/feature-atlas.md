# BRM feature atlas

Every executable example in the user guide has the same four semantic views:
the BRM declaration, the StanBlocks model BRM emitted, the complete Stan source
emitted from that model, and the Turing model selected directly from the BRM.
Choose a tab for one view, or **Compare** and select any two or more views for a
side-by-side reading. If a backend does not support an example, its pane remains
present and shows the exact construction error produced by the current build.

Nothing below is copied output. The docs build evaluates the displayed source
and derives all three emission panes from it.

## Gaussian population model and effect priors

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
gaussian = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x
    effect(:, :) ~ Normal(0, 2)
    effect(mu, x) ~ Normal(0, 0.25)
    y ~ Normal(mu, sigma)
end)((; x=[-1.0, 0.5, 2.0], y=[0.2, 1.1, -0.4]))
""", :gaussian; title="Gaussian population model")
```

## Partly missing Gaussian response

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
missing_gaussian = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x
    mi(y) ~ Normal(mu, sigma)
end)((;
    x=[-1.0, 0.5, 2.0, 0.25],
    y=Union{Missing,Float64}[0.2, missing, -0.4, missing],
))
""", :missing_gaussian; title="Partly missing Gaussian response")
```

## Shared and independent multi-response model

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
multi_response = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x
    y ~ Normal(mu, sigma)

    log(rate) ~ 1 + z
    count ~ Poisson(rate)

    y_replicate ~ Normal(mu, sigma)
end)((;
    x=[-1.0, 0.5, 2.0],
    z=[0.25, -0.5, 1.0],
    y=[0.2, 1.1, -0.4],
    count=[0, 2, 4],
    y_replicate=[-0.1, 0.3, 0.7],
))
""", :multi_response; title="Shared and independent responses")
```

## Correlated Gaussian multi-response model

The vector response is one row-wise multivariate likelihood. Outcome order is
the order written on both sides, and `L_res` is the estimated residual
covariance Cholesky factor (marginal scales plus an LKJ correlation factor).

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
correlated_response = (@brm begin
    L_res ~ LKJCovarianceFactor(
        2; scale_prior=Exponential(1), shape=2,
    )
    shared_log_rate ~ Normal(0, 1)
    concentration_mu = exp(-exp(shared_log_rate) * time)
    effect_mu = 1.0 - concentration_mu
    [concentration, effect] ~ MvNormalCholesky(
        [concentration_mu, effect_mu], L_res)
end)((;
    time=[0.0, 1.0, 2.0, 4.0],
    concentration=[1.0, 0.72, 0.51, 0.27],
    effect=[0.03, 0.18, 0.43, 0.79],
))
""", :correlated_response; title="Correlated Gaussian responses")
```

## Canonical Binomial-logit model

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
using LogExpFunctions: logit

binomial = (@brm begin
    logit(p) ~ 1 + x
    successes ~ Binomial(trials, p)
end)((;
    x=[-1.0, 0.5, 2.0, 0.25],
    trials=[2, 4, 6, 3],
    successes=[0, 2, 5, 1],
))
""", :binomial; title="Binomial logit")
```

## Poisson model with transformed data and an offset

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
poisson_offset = (@brm begin
    log(rate) ~ 1 + center(x) + offset(log(exposure))
    counts ~ Poisson(rate)
end)((;
    x=[1.0, 2.0, 4.0, 8.0],
    exposure=[2.0, 4.0, 8.0, 16.0],
    counts=[0, 2, 5, 7],
))
""", :poisson_offset; title="Poisson exposure model")
```

## Distributional count model

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
negative_binomial = (@brm begin
    log(mu) ~ 1 + x
    log(phi) ~ 1
    y ~ NegativeBinomial2(mu, phi)
end)((; x=[-1.0, 0.5, 2.0, 0.25], y=[0, 2, 5, 1]))
""", :negative_binomial; title="Negative-binomial mean and precision")
```

## Distributional count model with independent group slopes

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
grouped_negative_binomial = (@brm begin
    log(mu) ~ 1 + x + (1 + x || subject)
    log(phi) ~ 1 + z + (0 + z || batch)
    y ~ NegativeBinomial2(mu, phi)
end)((;
    x=[-1.0, 0.5, 2.0, 0.25],
    z=[0.0, 1.0, -0.5, 0.75],
    subject=[2, 1, 2, 3],
    batch=[2, 1, 1, 2],
    y=[0, 2, 5, 1],
))
""", :grouped_negative_binomial;
title="Independent mean and precision group slopes")
```

## Shared distributional group covariance

The repeated `joint` ID makes the mean and precision slopes slices of one
four-dimensional covariance block for each subject. Removing the ID would make
the two predictor blocks independent.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
shared_group_negative_binomial = (@brm begin
    log(mu) ~ 1 + x + (1 + x | joint | subject)
    log(phi) ~ 1 + z + (1 + z | joint | subject)
    y ~ NegativeBinomial2(mu, phi)
end)((;
    x=[-1.0, 0.5, 2.0, 0.25, -0.75, 1.25],
    z=[0.0, 1.0, -0.5, 0.75, 0.25, -1.0],
    subject=[2, 1, 2, 3, 1, 3],
    y=[0, 2, 5, 1, 3, 4],
))
""", :shared_group_negative_binomial;
title="Shared mean and precision group covariance")
```

## Multi-axis population PK kernel

The subject frame has one row per person, while the observation frame has one
row per concentration measurement. Those axes deliberately have different
lengths and the observation rows are interleaved. `ragged(x, group)` joins the
flat observation columns to the subject axis; `kernel(...)` then evaluates one
structural-model cell per subject. In the generated StanBlocks pane, that
public BRM kernel lowers to a `plate`.

This is a deliberately small one-compartment IV-bolus model,
`C(t) = dose / V * exp(-(CL / V)t)`. The shared `pk` ID gives `CL` and `V` one
correlated between-subject variability block.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
population_pk = (@brm begin
    sigma ~ Exponential(1)
    log(CL) ~ 1 + (1 | pk | subject)
    log(V)  ~ 1 + (1 | pk | subject)

    predicted_concentration ~ kernel(
        ragged(time, obs_subject), dose, CL, V,
    ) do ts, d, cl, volume
        d / volume * exp((-cl / volume) * ts)
    end

    ragged(concentration, obs_subject) ~
        Normal(predicted_concentration, sigma)
end)((;
    # Subject axis: one row per subject.
    subject=["alice", "bob"],
    dose=[100.0, 80.0],

    # Observation axis: one row per sample, interleaved by subject.
    obs_subject=["alice", "bob", "alice", "bob", "alice"],
    time=[0.5, 0.25, 1.5, 1.0, 3.0],
    concentration=[8.1, 7.6, 5.2, 4.9, 2.1],
))
""", :population_pk; title="Multi-axis population PK kernel")
```

## Verified public Warfarin PK/PD reproduction

The repository includes a complete executable translation of Sebastian
Weber's public StanCon 2018 Warfarin programs: a first-stage one-compartment
oral PK model with lag, allometry, four independent subject effects, and
Gamma overdispersion; followed by a turnover PD ODE conditioned on the public
PK posterior medians, with three independent PD subject effects and the same
observation family.

The [reproduction script](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/warfarin/reproduce.jl)
contains the typed ODE and likelihood definitions, a public two-subject data
slice, both `@brm` declarations, `stanc` checks, and finite BridgeStan density
and gradient checks. The accompanying [audit notes](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/warfarin/README.md)
map every retained prior, structural equation, solver tolerance, and known
generated-quantity difference back to the public source.

The [Warfarin PK/PD page](warfarin.md) also renders the separately specified
joint model: one posterior in which shared subject-level PK effects feed both
the PK and PD likelihoods, so PK uncertainty propagates into PD and PD data can
update PK quantities.

```julia
include("research/warfarin/reproduce.jl")
models = warfarin_sbbrmis()
```

This is the strongest identifiable public match to the Warfarin model
mentioned in [brms issue #1509](https://github.com/paul-buerkner/brms/issues/1509#issuecomment-1598639613),
but the issue itself provides no equations or citation, so identity with the
commenter's private working model cannot be proved. The public StanCon model
is two separate scalar-Gamma stages; it does not need the correlated Gaussian
outcome surface above.

## Categorical population terms

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
categorical = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + factor(group; ref=3) + x + x & group
    effect(mu, group) ~ Normal(0, 0.5)
    y ~ Normal(mu, sigma)
end)((;
    group=[1, 2, 3, 1, 2, 3],
    x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5],
))
""", :categorical; title="Categorical population design")
```

## Correlated group effects

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
grouped = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x + (1 + x | subject)
    outcome ~ Normal(mu, sigma)
end)((;
    x=[-1.0, 0.5, 2.0, 0.25],
    subject=[2, 1, 2, 3],
    outcome=[0.2, 1.1, -0.4, 0.7],
))
""", :grouped; title="Correlated random slopes")
```

## Random-effect scale and correlation priors

The `p` identifier gives the group block a stable prior address. The shared
half-Normal sets both scales, then the more-specific Exponential override
replaces it only on the `x` margin.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
grouped_priors = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x + (1 + x | p | subject)
    sd(:, p) ~ Normal(0, 0.5)
    sd(mu, p, x) ~ Exponential(0.25)
    cor(:, p) ~ LKJCholesky(2, 2.5)
    outcome ~ Normal(mu, sigma)
end)((;
    x=[-1.0, 0.5, 2.0, 0.25],
    subject=[2, 1, 2, 3],
    outcome=[0.2, 1.1, -0.4, 0.7],
))
""", :grouped_priors; title="Addressed group scale and correlation priors")
```

## Stratified group covariance

`gr(subject, by=arm)` fits a separate scale/correlation frame in each arm while
retaining one pooled subject coordinate.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
stratified_groups = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x + (1 + x | gr(subject, by=arm))
    outcome ~ Normal(mu, sigma)
end)((;
    x=[-1.0, 0.5, 2.0, 0.25, -0.5, 1.25],
    subject=[1, 1, 2, 3, 3, 4],
    arm=[1, 1, 1, 2, 2, 2],
    outcome=[0.2, 1.1, -0.4, 0.7, -0.2, 0.5],
))
""", :stratified_groups; title="Group covariance stratified by arm")
```

## Multi-membership group effects

Each row belongs to two groups. The row weights are normalized before the two
group contributions are pooled; `normalize=false` is the explicit raw-weight
variant.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
multi_membership = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x + (1 + x | mm(g1, g2; weights=(w1, w2)))
    outcome ~ Normal(mu, sigma)
end)((;
    x=[-1.0, 0.5, 2.0, 0.25],
    g1=[1, 1, 2, 3],
    g2=[2, 3, 3, 1],
    w1=[2.0, 1.0, 0.5, 3.0],
    w2=[1.0, 1.0, 1.5, 1.0],
    outcome=[0.2, 1.1, -0.4, 0.7],
))
""", :multi_membership; title="Weighted multi-membership random slopes")
```

## Multiple crossed group effects

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
crossed_groups = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x + (1 + x | subject) + (1 | item) + (0 + x || site)
    outcome ~ Normal(mu, sigma)
end)((;
    x=[-1.0, 0.5, 2.0, 0.25],
    subject=[2, 1, 2, 3],
    item=[2, 1, 1, 2],
    site=[10, 10, 20, 20],
    outcome=[0.2, 1.1, -0.4, 0.7],
))
""", :crossed_groups;
title="Crossed correlated and independent group blocks")
```

## Interval-censored evidence

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
interval_normal = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x
    y_lower ~ interval_censored(Normal(mu, sigma); upper=y_upper)
end)((;
    x=[-1.0, 0.0, 1.0],
    y_lower=[-0.4, 0.1, 0.8],
    y_upper=[-0.1, 0.4, 1.2],
))
""", :interval_normal; title="Interval-censored Normal evidence")
```

## A StanBlocks-only surface

The Turing pane is deliberately not removed for unsupported features. This
smooth example exercises the same build path and preserves the current
fail-closed reason beside the successful StanBlocks and Stan emissions.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
smooth = (@brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + s(x)
    y ~ Normal(mu, sigma)
end)((;
    x=collect(range(-2, 2; length=20)),
    y=sin.(collect(range(-2, 2; length=20))),
))
""", :smooth; title="Spline term with an unsupported Turing executor")
```
