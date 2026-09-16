# RBesT gMAP: primary-source model check

This study adds RBesT's meta-analytic-predictive (MAP) model as an adaptive-centering case study. Every statement below about what RBesT computes cites the package source at one pinned revision; nothing is taken from secondary summaries.

## Pinned source

- Repository: `https://github.com/Novartis/RBesT`, commit `3d5fa1dda74f9d68360094a11983f32dd1b61c10`, `DESCRIPTION` version `1.11-0`, license GPL (>= 3).
- Stan program: `inst/stan/gMAP.stan` (259 lines). R front end: `R/gMAP.R` (`gMAP <- function(...)` at line 360). Datasets: `data/{AS,colitis,crohn,transplant,asthma}.rda` documented in `R/{AS,colitis,crohn,transplant,asthma}.R`.

## The gMAP model, as sampled by RBesT

Data (`gMAP.stan` lines 5-60): `H` historical trials, one `link` in {1 normal, 2 binomial-logit, 3 Poisson-log}, family-specific responses (`y`, `y_se`; `r`, `r_n`; `count`, `log_offset`), an exchangeability cluster index `group_index` into `n_groups` clusters (one per trial unless the user groups trials), a `tau` stratum index with a prediction stratum, a design matrix `X` (`mX` columns; the intercept is column 1), Normal prior rows for `beta` and a two-parameter prior row per `tau` stratum, the `tau` prior family code, the random-effect family code (`re_dist` 0 Normal, 1 Student-t with `re_dist_t_df`), the `ncp` switch, and rescaling guesses for `beta` and `log tau`.

Parameters (lines 145-149): `beta_raw[mX]`, `tau_raw[n_tau_strata]`, `xi_eta[n_groups]`, all unconstrained.

Transformed parameters (lines 150-184):

- `beta = beta_raw_guess[1] + beta_raw_guess[2] .* beta_raw` (line 155).
- `tau = exp(tau_raw_guess[1] + tau_raw_guess[2] * tau_raw)` (line 161), with the log-scale Jacobian `target += tau_raw_guess[2] * tau_raw` added in the model block (line 231). `tau` is therefore positive by construction and every `tau` prior below is a truncated-at-zero density.
- Non-centered (`ncp == 1`, lines 164-177): `theta[h] = X[h] * beta + xi_eta[group_index[h]] * tau[stratum]`.
- Centered (`ncp == 0`, lines 130-141 and 178-183): the intercept column of `X` is zeroed (`X_param[i, 1] = 0`, and the program rejects a design whose first column is not all ones), and `theta[h] = X_param[h] * beta + beta_raw_guess[1, 1] + beta_raw_guess[2, 1] * xi_eta[group_index[h]]`, so `xi_eta` is the group-level intercept in the rescaled intercept frame.

Model block (lines 185-241):

- Random effects: NCP `xi_eta ~ normal(0, 1)` or `student_t(re_dist_t_df, 0, 1)` (lines 188-191); CP `xi_eta ~ normal((beta[1] - g1) / g2, tau[stratum] / g2)` or the Student-t analogue (lines 194-201), i.e. the centered group intercept has mean `beta[1]` and SD `tau` in the original frame.
- `beta ~ normal(beta_prior_stan[1], beta_prior_stan[2])` (line 205), independent Normals per column.
- `tau` prior by family code (lines 208-227): fixed (-1), half-Normal `normal(0, s)` (0), truncated Normal `normal(m, s)` (1), `uniform(a, b)` (2), `gamma(a, b)` (3), `inv_gamma(a, b)` (4), `lognormal(m, s)` (5), truncated Cauchy `cauchy(m, s)` (6), `exponential(rate)` (7).
- Likelihood unless `prior_PD` (lines 234-241): `y ~ normal(theta, y_se)`; `r ~ binomial_logit(r_n, theta)`; `count ~ poisson_log(log_offset + theta)`.

Generated quantities (lines 243-259): `theta_pred = normal_rng(beta[1], tau[tau_strata_pred])` (or the Student-t draw), the MAP prior for a new trial, and `theta_resp_pred` on the response scale (identity, `inv_logit`, `exp`).

## R-side choices that shape the fit (`R/gMAP.R`)

- `ncp <- getOption("RBesT.MC.ncp", 1)` (line 807): non-centered by default; `2` is an experimental auto mode (lines 813-822) that switches to centered only when `sqrt(tau_guess^2 / max(se^2)) > 20`. RBesT therefore fixes one parameterization per fit and exposes a manual global switch; there is no per-trial or online choice.
- Rescaling (`RBesT.MC.rescale`, default `TRUE`, lines 899-914): `beta_raw_guess` is the pooled GLM coefficient vector (lines 659-673) with SD guess `sigma_guess / sqrt(nInf)`, `nInf = 0.9 * (sigma_guess / tau_guess)^2` (line 829); `tau_raw_guess` is a log-Normal moment match to a square-root-Gamma spread of `tau_guess` (lines 836-848). `sigma_guess` and `tau_guess` are family-specific data summaries (lines 506-565). With `rescale = FALSE` both scale guesses become 1 and only the shifts remain.
- Sampler control (lines 900-905): `adapt_delta = 0.99`, `stepsize = 0.01`, `max_treedepth = 20`; defaults `iter = 6000`, `warmup = 2000`, `thin = 4`, `chains = 4`, `init = 1` (signature lines 385-390). These are far from CmdStan defaults and are part of what "the RBesT baseline" means.
- Priors: `beta.prior` as a scalar or vector is the SD of a zero-location Normal per column (lines 693-709); `tau.prior` is mandatory (line 723) and, for half-Normal, its single number is the SD with location forced to 0 (lines 760-765, 793-795). `REdist = c("normal", "t")`, `t.df = 5` (lines 382-383).
- Formula: `cbind(r, n - r) ~ 1 | study` (binomial), `cbind(y, y.se) ~ 1 | study` (Gaussian with known SE), `count ~ 1 + offset(log(exposure)) | study` (Poisson); the right of `|` is the exchangeability grouping, covariates go left of it. Gaussian `weights` enter only the `sigma_ref` guess (lines 522-530), never the likelihood.

## Shipped datasets and documented example calls

| dataset | family | H | columns | documented call |
| --- | --- | --- | --- | --- |
| `AS` (ankylosing spondylitis, ASAS20 at week 6) | binomial | 8 | `study, n, r` | `gMAP(cbind(r, n - r) ~ 1 \| study, family = binomial, data = AS, tau.dist = "HalfNormal", tau.prior = 1, beta.prior = 2)` (`R/AS.R`, `vignettes/introduction.Rmd` line 112) |
| `crohn` (CDAI change, placebo) | gaussian, `y.se = 88 / sqrt(n)` | 6 | `study, n, y` | `gMAP(cbind(y, y.se) ~ 1 \| study, family = gaussian, data = transform(crohn, y.se = 88 / sqrt(n)), weights = n, tau.dist = "HalfNormal", tau.prior = 44, beta.prior = cbind(0, 88))` (`R/crohn.R`) |
| `transplant` (treatment failure) | binomial | 11 | `study, n, r` | none documented |
| `colitis` (remission at week 8) | binomial | 4 | `study, n, r` | none documented |
| `asthma` (log exacerbation rate, phase III subset of 4) | gaussian on the log scale with `offset(log(d))` | 10 (4 used) | `study, d, n, log_mu_hat, se_log_mu_hat, ...` | `gMAP(cbind(log_mu_hat, se_log_mu_hat) ~ 1 + offset(log(d)) \| study, family = gaussian, data = asthma_ph3, tau.dist = "HalfNormal", tau.prior = 0.5, beta.prior = cbind(0, 2))` (`R/asthma.R`) |

No shipped dataset exercises the Poisson branch. The printed data frames are in `reference/datasets.tsv` once captured.

## BRM translation

The same posterior, on BRM's own coordinates:

```julia
AS_MODEL = @brm begin                      # binomial-logit gMAP, AS priors
    eta ~ 1 + (1 | map | study)
    effect(eta, Intercept) ~ Normal(0, 2)  # beta.prior = 2
    sd(:, map) ~ Normal(0, 1)              # tau.dist = "HalfNormal", tau.prior = 1
    r ~ BinomialLogit(n, eta)
end

CROHN_MODEL = @brm begin                   # Gaussian known-SE gMAP, crohn priors
    theta ~ 1 + (1 | map | study)
    effect(theta, Intercept) ~ Normal(0, 88)
    sd(:, map) ~ Normal(0, 44)
    y ~ Normal(theta, y_se)                # y_se = 88 / sqrt(n) as a data column
end
```

The Poisson branch is `log_rate ~ 1 + offset(log(exposure)) + (1 | map | study); count ~ Poisson(exp(log_rate))`. A gMAP covariate is an ordinary population term. `BinomialLogit(trials, logit)` is BRM's argument order (`src/sbimpl.jl:9758`).

Coordinates differ from RBesT's by affine maps only: BRM samples the intercept and `log tau` unshifted and unscaled, RBesT samples `(beta - guess_mean) / guess_sd` and `(log tau - g1) / g2`. Both use a standardized group innovation in the non-centered form; BRM's centered form (`centered_groups`) samples the group intercept itself in the original frame, RBesT's in the rescaled intercept frame. Densities agree up to these Jacobians and constants; the audit in the next leaf checks that numerically.

`eta ~ 1 + (1 | map | study)` with a Normal population prior is the eligible shape for BRM's automatic exact total coefficients (`total_groups=:auto`), so this page can compare RBesT's fixed NCP/CP switch, BRM's ordinary adaptive centering, and the exact-totals arms on the same posterior.

## Boundaries (verified against the source above)

- `tau` prior families. BRM's shared-block `sd(:, id)` address takes the zero-location half-Normal and Exponential forms, and the eight-schools study already used `sd(:, id) ~ Cauchy(0, 5)`, which covers RBesT's half-Normal, Exponential and location-zero truncated-Cauchy families. Truncated Normal with nonzero location, Uniform, Gamma, InvGamma and LogNormal are RBesT options that are not on that address today; they are documented as out of scope unless the custom-family hook is used. `Fixed` is not a prior.
- `tau.strata` (per-stratum `tau`) maps to BRM's `gr(study, by = stratum)`, which the adaptive-centering wrapper refuses (`test/adaptive_centering.jl`, "stratified correlated blocks refuse the single-frame contract") and which has no block-prior address. Out of scope for the page.
- `REdist = "t"` (Student-t random effects on `xi_eta`) is a different hierarchical distribution from BRM's Gaussian group block; BRM's exact totals marginalize a Student-t *population* prior, not a Student-t *random-effect* distribution. Out of scope for the page.
- `ncp = 2` auto-detection is a one-shot data heuristic, not a sampler adaptation; the page treats RBesT's two fixed parameterizations as the native baselines.
- The MAP prediction `theta_pred` is a fresh draw per iteration and carries no additional posterior information beyond `(beta[1], tau)`; the page reports `beta[1]`, `tau` and the trial totals as scientific quantities and derives the MAP predictive from them.

## Proposed page scope (decided in the next leaf)

Primary: `AS` (binomial, the package's canonical example). Secondary: `crohn` (Gaussian with known SE, the RBesT counterpart of eight schools with RBesT's own priors). `transplant` (binomial, H = 11) is a cheap third case if the protocol needs a larger trial count.
