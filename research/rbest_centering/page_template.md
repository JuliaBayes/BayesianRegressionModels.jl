# RBesT meta-analytic-predictive priors: sum-to-zero and automatic totals

## Result

This study takes the meta-analytic-predictive (MAP) model of [RBesT](https://github.com/Novartis/RBesT), Novartis' evidence-synthesis package, and compares three ways of sampling the same posterior: RBesT's own program in its legacy and new sum-to-zero forms, BRM's conventional block under WarmupHMC, and BRM's exact total coefficients with post-hoc and online centering. It was prompted by [Sebastian Weber's report](https://discourse.mc-stan.org/t/help-testing-brms-pr-for-sum-to-zero-and-partial-centering/41542/32) of about 5× efficiency from sum-to-zero in RBesT and about 10× with auto-centering.

{{RESULT}}

These are one-chain exploratory comparisons on RBesT's shipped datasets. They do not establish a uniformly best parameterization, and RBesT's own auto-centering is not public, so it is not measured here.

## The gMAP model

RBesT's `gMAP` fits a random-effects meta-analysis on the link scale of `H` historical trials and predicts the effect of a new trial: that predictive distribution is the MAP prior. With one exchangeability group per trial,

```math
\theta_h=\beta+\varepsilon_h,\qquad \varepsilon_h\sim N(0,\tau^2),\qquad
\beta\sim N(0,s_\beta^2),\qquad \tau\sim N^+(0,s_\tau),
```

with the likelihood chosen by the family. Two shipped datasets and their documented calls are used:

```r
# AS: ankylosing spondylitis, ASAS20 responders at week 6 in 8 placebo arms (binomial, logit)
gMAP(cbind(r, n - r) ~ 1 | study, family = binomial, data = AS,
     tau.dist = "HalfNormal", tau.prior = 1, beta.prior = 2)

# crohn: CDAI change from baseline in 6 placebo arms (Gaussian with known SE 88/sqrt(n))
gMAP(cbind(y, y.se) ~ 1 | study, family = gaussian,
     data = transform(crohn, y.se = 88 / sqrt(n)), weights = n,
     tau.dist = "HalfNormal", tau.prior = 44, beta.prior = cbind(0, 88))
```

`AS` has `r ~ BinomialLogit(n, θ)` with `s_β = 2`, `s_τ = 1`; `crohn` has `y ~ N(θ, se)` with `s_β = 88`, `s_τ = 44`. The scientific quantities are the population mean `β`, the between-trial SD `τ`, and the `H` trial effects `θ_h`; RBesT's MAP prediction `θ_new ~ N(β, τ)` is a function of the first two.

## RBesT's parameterizations

RBesT 1.11 samples rescaled coordinates `beta_raw`, `tau_raw` (log scale) and one `xi_eta` per trial, with a global switch: non-centered (`θ_h = β + τ ξ_h`, `ξ ~ N(0,1)`, the default) or centered (`ξ` is the trial effect itself in the rescaled intercept frame). Pull request 64 adds a sum-to-zero form: the sampled intercept becomes `α = β + mean(ε)` with the widened prior `N(0, s_β^2 + τ^2/H)`, the `H − 1` contrast coordinates keep the centered/non-centered switch, and `β` is recovered as an exact conditional draw through one extra data-free standard normal. Its default target acceptance drops from 0.99 to 0.95.

BRM's exact totals do the same marginalization in different coordinates: the `H` totals `θ_h` are sampled under their exact joint prior `τ^2 I + s_β^2 11'`, `β` is integrated out and recovered conditionally, and each total has its own centeredness between the model-scale total (`c = 1`) and the total scaled about its prior location (`c = 0`), chosen per trial post hoc or online.

## BRM models

The authoring panes below read the exact fitted declarations. The backend views are regenerated during the docs build. Sampling uses StanBlocks/BridgeStan and WHMC; the generated Turing pane is for inspection only.

### AS

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__,
    Main.BRMCenteringExamples.authoring(:rbest_as),
    :rbest_as_brm_model;
    title="RBesT AS: binomial gMAP", require_stan=true)
```

### crohn

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__,
    Main.BRMCenteringExamples.authoring(:rbest_crohn),
    :rbest_crohn_brm_model;
    title="RBesT crohn: Gaussian gMAP with known SE", require_stan=true)
```

BRM's conventional `centered_groups` samples the trial deviation `ε_h` with its `N(0, τ)` prior and keeps `β` separate; that is the `c = 1` endpoint of the ordinary adaptive wrapper. RBesT's legacy centered form samples `θ_h` itself, which is the geometry of BRM's totals at `c = 1` with `β` integrated out. The two "CP" rows are therefore different parameterizations.

## Sampling and adaptation

```julia
using StanBlocks, BridgeStan, WarmupHMC, Enzyme, Random
using DifferentiationInterface: AutoEnzyme

sb = SBBRMI(rbest_as_brm_model(); mod=@__MODULE__)          # total_groups=:auto
problem = StanBlocks.stan_instantiate(sb.model)
names = BridgeStan.param_unc_names(problem.model)
adaptive = adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.0)
fit = adaptive_warmup_mcmc(Xoshiro(1), adaptive;
    n_draws=10_000, monitor_ess=true, nonlinear_adapt=true)

draws = permutedims(fit.posterior_position)
recovered = recover_population_draws(sb, draws, names; rng=Xoshiro(404))
```

Every arm retains 10,000 draws from one chain, seed 1, from the same empirical per-trial starting point mapped into each program's coordinates. RBesT's programs run under CmdStan 2.39 with a zero-contribution C++ gradient counter, once with RBesT's own control (target acceptance 0.99 legacy or 0.95 sum-to-zero, step size 0.01, tree depth 20, 2,000 warmup iterations) and once under CmdStan defaults (0.8, depth 10, 1,000 warmup). RBesT's default thinning is not applied, since it discards draws without saving gradients. WHMC arms use its adaptive warmup with Pathfinder initialization; post-hoc arms select their controls from the NCP pilot and are charged for it; online arms adapt during warmup with the position or the position–gradient loss.

## One scientific scope per dataset

Both efficiency columns are relative to RBesT 1.11's default non-centered fit under its own control: minimum bulk ESS over the scientific quantities divided by sampling gradients, and divided by total gradients including warmup and pilots.

### AS: 10 quantities

{{AS_TABLE}}

{{AS_EFFICIENCY}}

### crohn: 8 quantities

{{CROHN_TABLE}}

{{CROHN_EFFICIENCY}}

## Saved-draw geometry

Each figure reuses the same 10,000 post-hoc-position draws across its columns, with CP as the visualization baseline; rows select the trials with minimum, middle and maximum inferred centeredness. The interactive previews in the KB brief show every fifth draw.

### AS

{{AS_TOTAL_PAIRS}}

{{AS_ORDINARY_PAIRS}}

### crohn

{{CROHN_TOTAL_PAIRS}}

{{CROHN_ORDINARY_PAIRS}}

## Diagnostics, recovery and limits

{{VALIDATION}}

Both BRM targets and both wrappers passed their density, gradient, transport and finite-difference checks, and BRM's targets were evaluated against RBesT's own compiled programs at mapped points: the legacy program against the conventional block and the sum-to-zero program against the exact totals differ by a constant Jacobian only, with identical trial effects. BRM omits the constant half-Normal normalizer of the bounded SD declaration; adding `log 2` aligns the densities. Out of scope on this page: RBesT's other `tau` prior families, per-stratum `tau`, Student-t random effects, and RBesT's unpublished auto-centering.

## Inspect and reproduce

{{LINKS}}

RBesT: `3d5fa1dda74f9d68360094a11983f32dd1b61c10` (1.11-0) and pull request 64 head `a5acbbc39c2cd620f0549159aa7c1991d4c28d6a`; WHMC `7aed40b18bd4cdabb75330f285d5c9b355575ab9`; Julia 1.10.11, CmdStan 2.39.0, CmdStanR 0.9.0.
