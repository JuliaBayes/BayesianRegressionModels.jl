---
---

# RBesT meta-analytic-predictive priors: sum-to-zero and automatic totals {#RBesT-meta-analytic-predictive-priors:-sum-to-zero-and-automatic-totals}

## Result {#Result}

This study takes the meta-analytic-predictive (MAP) model of [RBesT](https://github.com/Novartis/RBesT), Novartis' evidence-synthesis package, and compares three ways of sampling the same posterior: RBesT's own program in its legacy and new sum-to-zero forms, BRM's conventional block under WarmupHMC, and BRM's exact total coefficients with post-hoc and online centering. It was prompted by [Sebastian Weber's report](https://discourse.mc-stan.org/t/help-testing-brms-pr-for-sum-to-zero-and-partial-centering/41542/32) of about 5× efficiency from sum-to-zero in RBesT and about 10× with auto-centering.

On **AS** (binomial, 8 trials), RBesT's sum-to-zero fit under its own control gave 1.68× the total-gradient efficiency of its legacy non-centered fit with no divergences; the same legacy program under CmdStan defaults already gave 2.69×, so much of the gap Weber measured against legacy is RBesT's conservative sampler control. BRM's conventional block under WarmupHMC with online position adaptation was the most efficient arm at 3.74×; BRM's exact totals reached 3.0× at full centering but with 61 divergences, and only 1.55× with online gradient adaptation. On **crohn** (Gaussian with known SE, 6 trials), sum-to-zero gave 2.29×, legacy under CmdStan defaults 2.0×, and BRM's exact totals with online gradient adaptation 7.65×, with 62 divergences; BRM's conventional block with online position adaptation gave 3.38× and was the only WarmupHMC arm with zero divergences there.

The totals arms tell one structural story on both datasets: every selector puts every trial at centeredness 0.8 or above, and every totals arm diverges (6 to 134 divergences, against 0 for RBesT's sum-to-zero non-centered fit). With few trials and a wide population prior, the common shift of the totals and their contrasts want different centering, and one scalar centeredness per trial cannot separate them; the sum-to-zero construction handles the shift direction analytically. RBesT's legacy centered form, which also samples the trial effects with the shift inside them, collapses on AS (533 divergences, 0.03×).

These are one-chain exploratory comparisons on RBesT's shipped datasets. They do not establish a uniformly best parameterization, and RBesT's own auto-centering is not public, so it is not measured here.

## The gMAP model {#The-gMAP-model}

RBesT's `gMAP` fits a random-effects meta-analysis on the link scale of `H` historical trials and predicts the effect of a new trial: that predictive distribution is the MAP prior. With one exchangeability group per trial,

$$\theta_h=\beta+\varepsilon_h,\qquad \varepsilon_h\sim N(0,\tau^2),\qquad
\beta\sim N(0,s_\beta^2),\qquad \tau\sim N^+(0,s_\tau),$$

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

## RBesT's parameterizations {#RBesT's-parameterizations}

RBesT 1.11 samples rescaled coordinates `beta_raw`, `tau_raw` (log scale) and one `xi_eta` per trial, with a global switch: non-centered (`θ_h = β + τ ξ_h`, `ξ ~ N(0,1)`, the default) or centered (`ξ` is the trial effect itself in the rescaled intercept frame). Pull request 64 adds a sum-to-zero form: the sampled intercept becomes `α = β + mean(ε)` with the widened prior `N(0, s_β^2 + τ^2/H)`, the `H − 1` contrast coordinates keep the centered/non-centered switch, and `β` is recovered as an exact conditional draw through one extra data-free standard normal. Its default target acceptance drops from 0.99 to 0.95.

BRM's exact totals do the same marginalization in different coordinates: the `H` totals `θ_h` are sampled under their exact joint prior `τ^2 I + s_β^2 11'`, `β` is integrated out and recovered conditionally, and each total has its own centeredness between the model-scale total (`c = 1`) and the total scaled about its prior location (`c = 0`), chosen per trial post hoc or online.

## BRM models {#BRM-models}

The authoring panes below read the exact fitted declarations. The backend views are regenerated during the docs build. Sampling uses StanBlocks/BridgeStan and WHMC; the generated Turing pane is for inspection only.

### AS {#AS}

```brm-comparison
RBesT AS: binomial gMAP
```


```julia
using BayesianRegressionModels, Distributions, JSON, DelimitedFiles

function rbest_as_brm_model()
    table, header = readdlm(joinpath(pkgdir(BayesianRegressionModels),
        "research", "rbest_centering", "reference", "datasets", "AS.tsv"), '\t'; header=true)
    column(name) = table[:, only(findall(==(name), vec(header)))]
    data = (;study=collect(1:size(table, 1)), n=Int.(column("n")), r=Int.(column("r")))
    
    builder = @brm begin
        eta ~ 1 + (1 | map | study)
        effect(eta, Intercept) ~ Normal(0, 2)
        sd(:, map) ~ Normal(0, 1)
        r ~ BinomialLogit(n, eta)
    end
    
    builder(data)
end
```


```julia
SBBRMI with data keys = [:n, :r, :study, :total_A_eta, :total_group_eta, :total_location_eta, :total_ng_eta, :total_nk_eta, :total_np_eta, :total_precision_eta]
configured submodels:
_brm_total_scales_configured_1 = Base.merge(BayesianRegressionModels._brm_total_scales, quote
            tau ~ (BayesianRegressionModels.brm_vector_prior_ca4b8a1c1bc116d6)(0.0, 1.0; n = 1, lower = 0.0)
        end)
emitted @slic body:
begin
    total_scale_eta ~ _brm_total_scales_configured_1(; n = total_nk_eta)
    total_eta::matrix[total_ng_eta, total_nk_eta] ~ brm_total(total_scale_eta, total_A_eta, total_location_eta, total_precision_eta)
    population_eta = brm_total_recover_rng(total_eta, total_scale_eta, total_A_eta, total_location_eta, total_precision_eta)
    deviation_eta = brm_total_deviations(total_eta, total_A_eta * population_eta)
    total_Z_eta = hcat(rep_vector(1.0, num_elements(total_group_eta)))
    eta = rows_dot_product(total_eta[total_group_eta, :], total_Z_eta)
    r ~ binomial_logit(n, eta)
end
```


```stan
functions {
real brm_vector_prior_ca4b8a1c1bc116d6_lpdf(
    vector x,
    real arg_1,
    real arg_2
) {
    if((x[1] < 0.0)) {
        return negative_infinity();
    }
    return normal_lpdf(x[1] | arg_1, arg_2);
}
real brm_total_lpdf(
    matrix total,
    vector tau,
    matrix A,
    vector location,
    vector precision
) {
    int j = dims(total)[1];
    int k = dims(total)[2];
    int p = dims(A)[2];
    if (dims(tau)[1] != k) reject("brm_total_lpdf: dim mismatch — `tau` dim 1 (= ", dims(tau)[1], ") does not match `k` (= ", k, "), inferred from `total` dim 2. `k` sizes: `total` dim 2 (= ", dims(total)[2], "), `tau` dim 1 (= ", dims(tau)[1], "), `A` dim 1 (= ", dims(A)[1], ").");
    if (dims(A)[1] != k) reject("brm_total_lpdf: dim mismatch — `A` dim 1 (= ", dims(A)[1], ") does not match `k` (= ", k, "), inferred from `total` dim 2. `k` sizes: `total` dim 2 (= ", dims(total)[2], "), `tau` dim 1 (= ", dims(tau)[1], "), `A` dim 1 (= ", dims(A)[1], ").");
    if (dims(location)[1] != p) reject("brm_total_lpdf: dim mismatch — `location` dim 1 (= ", dims(location)[1], ") does not match `p` (= ", p, "), inferred from `A` dim 2. `p` sizes: `A` dim 2 (= ", dims(A)[2], "), `location` dim 1 (= ", dims(location)[1], "), `precision` dim 1 (= ", dims(precision)[1], ").");
    if (dims(precision)[1] != p) reject("brm_total_lpdf: dim mismatch — `precision` dim 1 (= ", dims(precision)[1], ") does not match `p` (= ", p, "), inferred from `A` dim 2. `p` sizes: `A` dim 2 (= ", dims(A)[2], "), `location` dim 1 (= ", dims(location)[1], "), `precision` dim 1 (= ", dims(precision)[1], ").");
    matrix[dims(precision)[1], dims(precision)[1]] Q = brm_total_precision(tau, A, precision, j);
    vector[dims(total)[2]] average = brm_total_mean(total);
    vector[dims(precision)[1]] beta = brm_total_conditional_mean(total, tau, A, location, precision, Q);
    vector[dims(total)[2]] residual = (average - (A * beta));
    real quadratic = 0.0;
    real lp = ((-0.5 * ((j * k) - p) * 1.8378770664093453) - (j * sum(log(tau))));
    for(c in 1:k) {
        quadratic += ((j * square(residual[c])) / square(tau[c]));
        for(g in 1:j) {
            quadratic += (square((total[g, c] - average[c])) / square(tau[c]));
        }
    }
    for(a in 1:p) {
        if((precision[a] > 0.0)) {
            lp += (0.5 * (log(precision[a]) - 1.8378770664093453));
            quadratic += (precision[a] * square((beta[a] - location[a])));
        }
    }
    return (lp - (0.5 * (log_determinant(Q) + quadratic)));
}
matrix brm_total_precision(
    vector tau,
    matrix A,
    vector precision,
    int n_groups
) {
    int k = dims(tau)[1];
    int p = dims(A)[2];
    if (dims(A)[1] != k) reject("brm_total_precision: dim mismatch — `A` dim 1 (= ", dims(A)[1], ") does not match `k` (= ", k, "), inferred from `tau` dim 1. `k` sizes: `tau` dim 1 (= ", dims(tau)[1], "), `A` dim 1 (= ", dims(A)[1], ").");
    if (dims(precision)[1] != p) reject("brm_total_precision: dim mismatch — `precision` dim 1 (= ", dims(precision)[1], ") does not match `p` (= ", p, "), inferred from `A` dim 2. `p` sizes: `A` dim 2 (= ", dims(A)[2], "), `precision` dim 1 (= ", dims(precision)[1], ").");
    matrix[dims(precision)[1], dims(precision)[1]] out = diag_matrix(precision);
    for(a in 1:p) {
        for(b in 1:p) {
            for(c in 1:k) {
                out[a, b] += ((n_groups * A[c, a] * A[c, b]) / square(tau[c]));
            }
        }
    }
    return out;
}
vector brm_total_mean(
    matrix total
) {
    int j = dims(total)[1];
    int k = dims(total)[2];
    vector[k] out = rep_vector(0.0, k);
    for(c in 1:k) {
        out[c] = (sum(total[:, c]) / j);
    }
    return out;
}
vector brm_total_conditional_mean(
    matrix total,
    vector tau,
    matrix A,
    vector location,
    vector precision,
    matrix Q
) {
    int j = dims(total)[1];
    int k = dims(total)[2];
    int p = dims(A)[2];
    if (dims(tau)[1] != k) reject("brm_total_conditional_mean: dim mismatch — `tau` dim 1 (= ", dims(tau)[1], ") does not match `k` (= ", k, "), inferred from `total` dim 2. `k` sizes: `total` dim 2 (= ", dims(total)[2], "), `tau` dim 1 (= ", dims(tau)[1], "), `A` dim 1 (= ", dims(A)[1], ").");
    if (dims(A)[1] != k) reject("brm_total_conditional_mean: dim mismatch — `A` dim 1 (= ", dims(A)[1], ") does not match `k` (= ", k, "), inferred from `total` dim 2. `k` sizes: `total` dim 2 (= ", dims(total)[2], "), `tau` dim 1 (= ", dims(tau)[1], "), `A` dim 1 (= ", dims(A)[1], ").");
    if (dims(location)[1] != p) reject("brm_total_conditional_mean: dim mismatch — `location` dim 1 (= ", dims(location)[1], ") does not match `p` (= ", p, "), inferred from `A` dim 2. `p` sizes: `A` dim 2 (= ", dims(A)[2], "), `location` dim 1 (= ", dims(location)[1], "), `precision` dim 1 (= ", dims(precision)[1], "), `Q` dim 1 (= ", dims(Q)[1], "), `Q` dim 2 (= ", dims(Q)[2], ").");
    if (dims(precision)[1] != p) reject("brm_total_conditional_mean: dim mismatch — `precision` dim 1 (= ", dims(precision)[1], ") does not match `p` (= ", p, "), inferred from `A` dim 2. `p` sizes: `A` dim 2 (= ", dims(A)[2], "), `location` dim 1 (= ", dims(location)[1], "), `precision` dim 1 (= ", dims(precision)[1], "), `Q` dim 1 (= ", dims(Q)[1], "), `Q` dim 2 (= ", dims(Q)[2], ").");
    if (dims(Q)[1] != p) reject("brm_total_conditional_mean: dim mismatch — `Q` dim 1 (= ", dims(Q)[1], ") does not match `p` (= ", p, "), inferred from `A` dim 2. `p` sizes: `A` dim 2 (= ", dims(A)[2], "), `location` dim 1 (= ", dims(location)[1], "), `precision` dim 1 (= ", dims(precision)[1], "), `Q` dim 1 (= ", dims(Q)[1], "), `Q` dim 2 (= ", dims(Q)[2], ").");
    if (dims(Q)[2] != p) reject("brm_total_conditional_mean: dim mismatch — `Q` dim 2 (= ", dims(Q)[2], ") does not match `p` (= ", p, "), inferred from `A` dim 2. `p` sizes: `A` dim 2 (= ", dims(A)[2], "), `location` dim 1 (= ", dims(location)[1], "), `precision` dim 1 (= ", dims(precision)[1], "), `Q` dim 1 (= ", dims(Q)[1], "), `Q` dim 2 (= ", dims(Q)[2], ").");
    vector[dims(total)[2]] average = brm_total_mean(total);
    vector[dims(location)[1]] natural = (precision .* location);
    for(a in 1:p) {
        for(c in 1:k) {
            natural[a] += ((j * A[c, a] * average[c]) / square(tau[c]));
        }
    }
    return mdivide_left_spd(Q, natural);
}
vector brm_total_recover_rng(
    matrix total,
    vector tau,
    matrix A,
    vector location,
    vector precision
) {
    int j = dims(total)[1];
    int k = dims(total)[2];
    int p = dims(A)[2];
    if (dims(tau)[1] != k) reject("brm_total_recover_rng: dim mismatch — `tau` dim 1 (= ", dims(tau)[1], ") does not match `k` (= ", k, "), inferred from `total` dim 2. `k` sizes: `total` dim 2 (= ", dims(total)[2], "), `tau` dim 1 (= ", dims(tau)[1], "), `A` dim 1 (= ", dims(A)[1], ").");
    if (dims(A)[1] != k) reject("brm_total_recover_rng: dim mismatch — `A` dim 1 (= ", dims(A)[1], ") does not match `k` (= ", k, "), inferred from `total` dim 2. `k` sizes: `total` dim 2 (= ", dims(total)[2], "), `tau` dim 1 (= ", dims(tau)[1], "), `A` dim 1 (= ", dims(A)[1], ").");
    if (dims(location)[1] != p) reject("brm_total_recover_rng: dim mismatch — `location` dim 1 (= ", dims(location)[1], ") does not match `p` (= ", p, "), inferred from `A` dim 2. `p` sizes: `A` dim 2 (= ", dims(A)[2], "), `location` dim 1 (= ", dims(location)[1], "), `precision` dim 1 (= ", dims(precision)[1], ").");
    if (dims(precision)[1] != p) reject("brm_total_recover_rng: dim mismatch — `precision` dim 1 (= ", dims(precision)[1], ") does not match `p` (= ", p, "), inferred from `A` dim 2. `p` sizes: `A` dim 2 (= ", dims(A)[2], "), `location` dim 1 (= ", dims(location)[1], "), `precision` dim 1 (= ", dims(precision)[1], ").");
    matrix[dims(precision)[1], dims(precision)[1]] Q = brm_total_precision(tau, A, precision, j);
    vector[dims(precision)[1]] beta = brm_total_conditional_mean(total, tau, A, location, precision, Q);
    return multi_normal_rng(beta, inverse_spd(Q));
}
matrix brm_total_deviations(
    matrix total,
    vector mu
) {
    int j = dims(total)[1];
    int k = dims(total)[2];
    if (dims(mu)[1] != k) reject("brm_total_deviations: dim mismatch — `mu` dim 1 (= ", dims(mu)[1], ") does not match `k` (= ", k, "), inferred from `total` dim 2. `k` sizes: `total` dim 2 (= ", dims(total)[2], "), `mu` dim 1 (= ", dims(mu)[1], ").");
    matrix[dims(total)[1], dims(total)[2]] out = total;
    for(c in 1:k) {
        out[:, c] = (total[:, c] - rep_vector(mu[c], j));
    }
    return out;
}
matrix hcat(vector x) {
    int n = dims(x)[1];
    return to_matrix(x, n, 1);
}
vector binomial_logit_lpmfs(
    array[] int y,
    array[] int args1,
    vector args2
) {
    return jbroadcasted_binomial_logit_lpmfs(y, args1, args2);
}
vector jbroadcasted_binomial_logit_lpmfs(
    array[] int x1,
    array[] int x2,
    vector x3
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = binomial_logit_lpmfs(
            broadcasted_getindex(x1, i),
            broadcasted_getindex(x2, i),
            broadcasted_getindex(x3, i)
        );
    }
    return rv;
}
real binomial_logit_lpmfs(
    int args1,
    int args2,
    real args3
) {
    return binomial_logit_lpmf(args1 | args2, args3);
}
int broadcasted_getindex(array[] int x, int i) {
    return x[i];
}
real broadcasted_getindex(vector x, int i) {
    return x[i];
}
array[] int binomial_logit_int_rng(
    int anontok__1,
    array[] int N,
    vector eta
) {
    int n = anontok__1;
    if (dims(N)[1] != n) reject("binomial_logit_rng: dim mismatch — `N` dim 1 (= ", dims(N)[1], ") does not match `n` (= ", n, "), inferred from `anontok__1` dim 1. `n` sizes: `anontok__1` dim 1 (= ", anontok__1, "), `N` dim 1 (= ", dims(N)[1], ").");
    return binomial_rng(N, inv_logit(eta));
}
}
data {
    int total_ng_eta;
    int total_nk_eta;
    int total_A_eta_m;
    int total_A_eta_n;
    matrix[total_A_eta_m, total_A_eta_n] total_A_eta;
    int total_location_eta_n;
    vector[total_location_eta_n] total_location_eta;
    int total_precision_eta_n;
    vector[total_precision_eta_n] total_precision_eta;
    int total_group_eta_n;
    array[total_group_eta_n] int total_group_eta;
    int r_n;
    array[r_n] int r;
    int n_n;
    array[n_n] int n;
}
transformed data {
    matrix[num_elements(total_group_eta), 1] total_Z_eta = hcat(rep_vector(1.0, num_elements(total_group_eta)));
}
parameters {
    vector<lower=0.0>[1] total_scale_eta_tau;
    matrix[total_ng_eta, total_nk_eta] total_eta;
}
transformed parameters {
    vector<lower=0.0>[1] total_scale_eta = total_scale_eta_tau;
    vector[num_elements(total_group_eta)] eta = rows_dot_product(total_eta[total_group_eta, :], total_Z_eta);
}
model {
    total_scale_eta_tau ~ brm_vector_prior_ca4b8a1c1bc116d6(0.0, 1.0);
    total_eta ~ brm_total(total_scale_eta, total_A_eta, total_location_eta, total_precision_eta);
    r ~ binomial_logit(n, eta);
}
generated quantities {
    vector[total_precision_eta_n] population_eta = brm_total_recover_rng(
        total_eta,
        total_scale_eta,
        total_A_eta,
        total_location_eta,
        total_precision_eta
    );
    matrix[total_ng_eta, total_A_eta_m] deviation_eta = brm_total_deviations(total_eta, (total_A_eta * population_eta));
    vector[r_n] r_likelihood = binomial_logit_lpmfs(r, n, eta);
    array[n_n] int r_gen = binomial_logit_int_rng(r_n, n, eta);
}
```


```julia
#= line 0 =# Turing.@model(function brm_model(y, X_eta, random_effects_eta_1, n)
        beta_pop ~ Distributions.product_distribution([Distributions.Normal(0, 2)])
        eta_eta = X_eta * beta_pop
        group_effect_1 = Base.zeros(Base.length(y))
        group_1_1 ~ DynamicPPL.to_submodel(BayesianRegressionModelsTuringExt._brm_group_effect_model(random_effects_eta_1, (Distributions.Normal(0, 1),), nothing))
        group_effect_1 = group_effect_1 + group_1_1.effect
        eta_eta = eta_eta + group_effect_1
        eta = eta_eta
        begin
            for i = Base.eachindex(y)
                y[i] ~ BayesianRegressionModels.BinomialLogit(n[i], eta[i])
            end
        end
        (; eta = eta, response = y)
    end)
```


### crohn {#crohn}

```brm-comparison
RBesT crohn: Gaussian gMAP with known SE
```


```julia
using BayesianRegressionModels, Distributions, JSON, DelimitedFiles

function rbest_crohn_brm_model()
    table, header = readdlm(joinpath(pkgdir(BayesianRegressionModels),
        "research", "rbest_centering", "reference", "datasets", "crohn.tsv"), '\t'; header=true)
    column(name) = table[:, only(findall(==(name), vec(header)))]
    data = (;study=collect(1:size(table, 1)), y=Float64.(column("y")), y_se=88 ./ sqrt.(Float64.(column("n"))))
    
    builder = @brm begin
        theta ~ 1 + (1 | map | study)
        effect(theta, Intercept) ~ Normal(0, 88)
        sd(:, map) ~ Normal(0, 44)
        y ~ Normal(theta, y_se)
    end
    
    builder(data)
end
```


```julia
SBBRMI with data keys = [:study, :total_A_theta, :total_group_theta, :total_location_theta, :total_ng_theta, :total_nk_theta, :total_np_theta, :total_precision_theta, :y, :y_se]
configured submodels:
_brm_total_scales_configured_1 = Base.merge(BayesianRegressionModels._brm_total_scales, quote
            tau ~ (BayesianRegressionModels.brm_vector_prior_ca4b8a1c1bc116d6)(0.0, 44.0; n = 1, lower = 0.0)
        end)
emitted @slic body:
begin
    total_scale_theta ~ _brm_total_scales_configured_1(; n = total_nk_theta)
    total_theta::matrix[total_ng_theta, total_nk_theta] ~ brm_total(total_scale_theta, total_A_theta, total_location_theta, total_precision_theta)
    population_theta = brm_total_recover_rng(total_theta, total_scale_theta, total_A_theta, total_location_theta, total_precision_theta)
    deviation_theta = brm_total_deviations(total_theta, total_A_theta * population_theta)
    total_Z_theta = hcat(rep_vector(1.0, num_elements(total_group_theta)))
    theta = rows_dot_product(total_theta[total_group_theta, :], total_Z_theta)
    y ~ normal(theta, y_se)
end
```


```stan
functions {
real brm_vector_prior_ca4b8a1c1bc116d6_lpdf(
    vector x,
    real arg_1,
    real arg_2
) {
    if((x[1] < 0.0)) {
        return negative_infinity();
    }
    return normal_lpdf(x[1] | arg_1, arg_2);
}
real brm_total_lpdf(
    matrix total,
    vector tau,
    matrix A,
    vector location,
    vector precision
) {
    int j = dims(total)[1];
    int k = dims(total)[2];
    int p = dims(A)[2];
    if (dims(tau)[1] != k) reject("brm_total_lpdf: dim mismatch — `tau` dim 1 (= ", dims(tau)[1], ") does not match `k` (= ", k, "), inferred from `total` dim 2. `k` sizes: `total` dim 2 (= ", dims(total)[2], "), `tau` dim 1 (= ", dims(tau)[1], "), `A` dim 1 (= ", dims(A)[1], ").");
    if (dims(A)[1] != k) reject("brm_total_lpdf: dim mismatch — `A` dim 1 (= ", dims(A)[1], ") does not match `k` (= ", k, "), inferred from `total` dim 2. `k` sizes: `total` dim 2 (= ", dims(total)[2], "), `tau` dim 1 (= ", dims(tau)[1], "), `A` dim 1 (= ", dims(A)[1], ").");
    if (dims(location)[1] != p) reject("brm_total_lpdf: dim mismatch — `location` dim 1 (= ", dims(location)[1], ") does not match `p` (= ", p, "), inferred from `A` dim 2. `p` sizes: `A` dim 2 (= ", dims(A)[2], "), `location` dim 1 (= ", dims(location)[1], "), `precision` dim 1 (= ", dims(precision)[1], ").");
    if (dims(precision)[1] != p) reject("brm_total_lpdf: dim mismatch — `precision` dim 1 (= ", dims(precision)[1], ") does not match `p` (= ", p, "), inferred from `A` dim 2. `p` sizes: `A` dim 2 (= ", dims(A)[2], "), `location` dim 1 (= ", dims(location)[1], "), `precision` dim 1 (= ", dims(precision)[1], ").");
    matrix[dims(precision)[1], dims(precision)[1]] Q = brm_total_precision(tau, A, precision, j);
    vector[dims(total)[2]] average = brm_total_mean(total);
    vector[dims(precision)[1]] beta = brm_total_conditional_mean(total, tau, A, location, precision, Q);
    vector[dims(total)[2]] residual = (average - (A * beta));
    real quadratic = 0.0;
    real lp = ((-0.5 * ((j * k) - p) * 1.8378770664093453) - (j * sum(log(tau))));
    for(c in 1:k) {
        quadratic += ((j * square(residual[c])) / square(tau[c]));
        for(g in 1:j) {
            quadratic += (square((total[g, c] - average[c])) / square(tau[c]));
        }
    }
    for(a in 1:p) {
        if((precision[a] > 0.0)) {
            lp += (0.5 * (log(precision[a]) - 1.8378770664093453));
            quadratic += (precision[a] * square((beta[a] - location[a])));
        }
    }
    return (lp - (0.5 * (log_determinant(Q) + quadratic)));
}
matrix brm_total_precision(
    vector tau,
    matrix A,
    vector precision,
    int n_groups
) {
    int k = dims(tau)[1];
    int p = dims(A)[2];
    if (dims(A)[1] != k) reject("brm_total_precision: dim mismatch — `A` dim 1 (= ", dims(A)[1], ") does not match `k` (= ", k, "), inferred from `tau` dim 1. `k` sizes: `tau` dim 1 (= ", dims(tau)[1], "), `A` dim 1 (= ", dims(A)[1], ").");
    if (dims(precision)[1] != p) reject("brm_total_precision: dim mismatch — `precision` dim 1 (= ", dims(precision)[1], ") does not match `p` (= ", p, "), inferred from `A` dim 2. `p` sizes: `A` dim 2 (= ", dims(A)[2], "), `precision` dim 1 (= ", dims(precision)[1], ").");
    matrix[dims(precision)[1], dims(precision)[1]] out = diag_matrix(precision);
    for(a in 1:p) {
        for(b in 1:p) {
            for(c in 1:k) {
                out[a, b] += ((n_groups * A[c, a] * A[c, b]) / square(tau[c]));
            }
        }
    }
    return out;
}
vector brm_total_mean(
    matrix total
) {
    int j = dims(total)[1];
    int k = dims(total)[2];
    vector[k] out = rep_vector(0.0, k);
    for(c in 1:k) {
        out[c] = (sum(total[:, c]) / j);
    }
    return out;
}
vector brm_total_conditional_mean(
    matrix total,
    vector tau,
    matrix A,
    vector location,
    vector precision,
    matrix Q
) {
    int j = dims(total)[1];
    int k = dims(total)[2];
    int p = dims(A)[2];
    if (dims(tau)[1] != k) reject("brm_total_conditional_mean: dim mismatch — `tau` dim 1 (= ", dims(tau)[1], ") does not match `k` (= ", k, "), inferred from `total` dim 2. `k` sizes: `total` dim 2 (= ", dims(total)[2], "), `tau` dim 1 (= ", dims(tau)[1], "), `A` dim 1 (= ", dims(A)[1], ").");
    if (dims(A)[1] != k) reject("brm_total_conditional_mean: dim mismatch — `A` dim 1 (= ", dims(A)[1], ") does not match `k` (= ", k, "), inferred from `total` dim 2. `k` sizes: `total` dim 2 (= ", dims(total)[2], "), `tau` dim 1 (= ", dims(tau)[1], "), `A` dim 1 (= ", dims(A)[1], ").");
    if (dims(location)[1] != p) reject("brm_total_conditional_mean: dim mismatch — `location` dim 1 (= ", dims(location)[1], ") does not match `p` (= ", p, "), inferred from `A` dim 2. `p` sizes: `A` dim 2 (= ", dims(A)[2], "), `location` dim 1 (= ", dims(location)[1], "), `precision` dim 1 (= ", dims(precision)[1], "), `Q` dim 1 (= ", dims(Q)[1], "), `Q` dim 2 (= ", dims(Q)[2], ").");
    if (dims(precision)[1] != p) reject("brm_total_conditional_mean: dim mismatch — `precision` dim 1 (= ", dims(precision)[1], ") does not match `p` (= ", p, "), inferred from `A` dim 2. `p` sizes: `A` dim 2 (= ", dims(A)[2], "), `location` dim 1 (= ", dims(location)[1], "), `precision` dim 1 (= ", dims(precision)[1], "), `Q` dim 1 (= ", dims(Q)[1], "), `Q` dim 2 (= ", dims(Q)[2], ").");
    if (dims(Q)[1] != p) reject("brm_total_conditional_mean: dim mismatch — `Q` dim 1 (= ", dims(Q)[1], ") does not match `p` (= ", p, "), inferred from `A` dim 2. `p` sizes: `A` dim 2 (= ", dims(A)[2], "), `location` dim 1 (= ", dims(location)[1], "), `precision` dim 1 (= ", dims(precision)[1], "), `Q` dim 1 (= ", dims(Q)[1], "), `Q` dim 2 (= ", dims(Q)[2], ").");
    if (dims(Q)[2] != p) reject("brm_total_conditional_mean: dim mismatch — `Q` dim 2 (= ", dims(Q)[2], ") does not match `p` (= ", p, "), inferred from `A` dim 2. `p` sizes: `A` dim 2 (= ", dims(A)[2], "), `location` dim 1 (= ", dims(location)[1], "), `precision` dim 1 (= ", dims(precision)[1], "), `Q` dim 1 (= ", dims(Q)[1], "), `Q` dim 2 (= ", dims(Q)[2], ").");
    vector[dims(total)[2]] average = brm_total_mean(total);
    vector[dims(location)[1]] natural = (precision .* location);
    for(a in 1:p) {
        for(c in 1:k) {
            natural[a] += ((j * A[c, a] * average[c]) / square(tau[c]));
        }
    }
    return mdivide_left_spd(Q, natural);
}
vector brm_total_recover_rng(
    matrix total,
    vector tau,
    matrix A,
    vector location,
    vector precision
) {
    int j = dims(total)[1];
    int k = dims(total)[2];
    int p = dims(A)[2];
    if (dims(tau)[1] != k) reject("brm_total_recover_rng: dim mismatch — `tau` dim 1 (= ", dims(tau)[1], ") does not match `k` (= ", k, "), inferred from `total` dim 2. `k` sizes: `total` dim 2 (= ", dims(total)[2], "), `tau` dim 1 (= ", dims(tau)[1], "), `A` dim 1 (= ", dims(A)[1], ").");
    if (dims(A)[1] != k) reject("brm_total_recover_rng: dim mismatch — `A` dim 1 (= ", dims(A)[1], ") does not match `k` (= ", k, "), inferred from `total` dim 2. `k` sizes: `total` dim 2 (= ", dims(total)[2], "), `tau` dim 1 (= ", dims(tau)[1], "), `A` dim 1 (= ", dims(A)[1], ").");
    if (dims(location)[1] != p) reject("brm_total_recover_rng: dim mismatch — `location` dim 1 (= ", dims(location)[1], ") does not match `p` (= ", p, "), inferred from `A` dim 2. `p` sizes: `A` dim 2 (= ", dims(A)[2], "), `location` dim 1 (= ", dims(location)[1], "), `precision` dim 1 (= ", dims(precision)[1], ").");
    if (dims(precision)[1] != p) reject("brm_total_recover_rng: dim mismatch — `precision` dim 1 (= ", dims(precision)[1], ") does not match `p` (= ", p, "), inferred from `A` dim 2. `p` sizes: `A` dim 2 (= ", dims(A)[2], "), `location` dim 1 (= ", dims(location)[1], "), `precision` dim 1 (= ", dims(precision)[1], ").");
    matrix[dims(precision)[1], dims(precision)[1]] Q = brm_total_precision(tau, A, precision, j);
    vector[dims(precision)[1]] beta = brm_total_conditional_mean(total, tau, A, location, precision, Q);
    return multi_normal_rng(beta, inverse_spd(Q));
}
matrix brm_total_deviations(
    matrix total,
    vector mu
) {
    int j = dims(total)[1];
    int k = dims(total)[2];
    if (dims(mu)[1] != k) reject("brm_total_deviations: dim mismatch — `mu` dim 1 (= ", dims(mu)[1], ") does not match `k` (= ", k, "), inferred from `total` dim 2. `k` sizes: `total` dim 2 (= ", dims(total)[2], "), `mu` dim 1 (= ", dims(mu)[1], ").");
    matrix[dims(total)[1], dims(total)[2]] out = total;
    for(c in 1:k) {
        out[:, c] = (total[:, c] - rep_vector(mu[c], j));
    }
    return out;
}
matrix hcat(vector x) {
    int n = dims(x)[1];
    return to_matrix(x, n, 1);
}
vector normal_lpdfs(
    vector obs,
    vector loc,
    vector scale
) {
    return jbroadcasted_normal_lpdfs(obs, loc, scale);
}
vector jbroadcasted_normal_lpdfs(
    vector x1,
    vector x2,
    vector x3
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = normal_lpdfs(broadcasted_getindex(x1, i), broadcasted_getindex(x2, i), broadcasted_getindex(x3, i));
    }
    return rv;
}
real normal_lpdfs(
    real args1,
    real args2,
    real args3
) {
    return normal_lpdf(args1 | args2, args3);
}
real broadcasted_getindex(vector x, int i) {
    return x[i];
}
vector normal_vector_rng(
    int anontok__1,
    vector a,
    vector b
) {
    int n = anontok__1;
    return to_vector(normal_rng(a, b));
}
}
data {
    int total_ng_theta;
    int total_nk_theta;
    int total_A_theta_m;
    int total_A_theta_n;
    matrix[total_A_theta_m, total_A_theta_n] total_A_theta;
    int total_location_theta_n;
    vector[total_location_theta_n] total_location_theta;
    int total_precision_theta_n;
    vector[total_precision_theta_n] total_precision_theta;
    int total_group_theta_n;
    array[total_group_theta_n] int total_group_theta;
    int y_n;
    vector[y_n] y;
    int y_se_n;
    vector[y_se_n] y_se;
}
transformed data {
    matrix[num_elements(total_group_theta), 1] total_Z_theta = hcat(rep_vector(1.0, num_elements(total_group_theta)));
}
parameters {
    vector<lower=0.0>[1] total_scale_theta_tau;
    matrix[total_ng_theta, total_nk_theta] total_theta;
}
transformed parameters {
    vector<lower=0.0>[1] total_scale_theta = total_scale_theta_tau;
    vector[num_elements(total_group_theta)] theta = rows_dot_product(total_theta[total_group_theta, :], total_Z_theta);
}
model {
    total_scale_theta_tau ~ brm_vector_prior_ca4b8a1c1bc116d6(0.0, 44.0);
    total_theta ~ brm_total(total_scale_theta, total_A_theta, total_location_theta, total_precision_theta);
    y ~ normal(theta, y_se);
}
generated quantities {
    vector[total_precision_theta_n] population_theta = brm_total_recover_rng(
        total_theta,
        total_scale_theta,
        total_A_theta,
        total_location_theta,
        total_precision_theta
    );
    matrix[total_ng_theta, total_A_theta_m] deviation_theta = brm_total_deviations(total_theta, (total_A_theta * population_theta));
    vector[y_n] y_likelihood = normal_lpdfs(y, theta, y_se);
    vector[y_n] y_gen = normal_vector_rng(y_n, theta, y_se);
}
```


```julia
#= line 0 =# Turing.@model(function brm_model(y, X_theta, random_effects_theta_1, y_se)
        beta_pop ~ Distributions.product_distribution([Distributions.Normal(0, 88)])
        eta_theta = X_theta * beta_pop
        group_effect_1 = Base.zeros(Base.length(y))
        group_1_1 ~ DynamicPPL.to_submodel(BayesianRegressionModelsTuringExt._brm_group_effect_model(random_effects_theta_1, (Distributions.Normal(0, 44),), nothing))
        group_effect_1 = group_effect_1 + group_1_1.effect
        eta_theta = eta_theta + group_effect_1
        theta = eta_theta
        begin
            for i = Base.eachindex(y)
                y[i] ~ Distributions.Normal(theta[i], y_se[i])
            end
        end
        (; theta = theta, response = y)
    end)
```


BRM's conventional `centered_groups` samples the trial deviation `ε_h` with its `N(0, τ)` prior and keeps `β` separate; that is the `c = 1` endpoint of the ordinary adaptive wrapper. RBesT's legacy centered form samples `θ_h` itself, which is the geometry of BRM's totals at `c = 1` with `β` integrated out. The two "CP" rows are therefore different parameterizations.

## Sampling and adaptation {#Sampling-and-adaptation}

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

## One scientific scope per dataset {#One-scientific-scope-per-dataset}

Both efficiency columns are relative to RBesT 1.11's default non-centered fit under its own control: minimum bulk ESS over the scientific quantities divided by sampling gradients, and divided by total gradients including warmup and pilots.

### AS: 10 quantities {#AS:-10-quantities}

|                                       Method | Total gradients | Sampling efficiency | Total efficiency | Divergences |
| --------------------------------------------:| ---------------:| -------------------:| ----------------:| -----------:|
|   RBesT 1.11 NCP (its control) · Native Stan |         328,781 |                  1× |               1× |           0 |
|    RBesT 1.11 CP (its control) · Native Stan |         323,012 |             0.0336× |          0.0319× |         533 |
| RBesT 1.11 NCP (Stan defaults) · Native Stan |         145,801 |               2.44× |            2.69× |           4 |
|  RBesT 1.11 CP (Stan defaults) · Native Stan |         138,693 |             0.0627× |          0.0702× |         670 |
|             RBesT PR64 S2Z NCP · Native Stan |         262,110 |               1.63× |            1.68× |           0 |
|              RBesT PR64 S2Z CP · Native Stan |         182,173 |               1.75× |            1.73× |           9 |
|                      BRM ordinary NCP · WHMC |         128,794 |               2.16× |            2.59× |           1 |
|                       BRM ordinary CP · WHMC |          94,686 |               1.08× |            1.29× |           4 |
|        BRM ordinary post-hoc position · WHMC |         219,838 |               2.88× |            1.39× |           0 |
|        BRM ordinary post-hoc gradient · WHMC |         209,470 |               3.56× |            1.64× |           1 |
|          BRM ordinary online position · WHMC |          83,667 |               3.14× |            3.74× |          14 |
|          BRM ordinary online gradient · WHMC |         121,115 |               2.45× |            2.93× |           5 |
|                         BRM total NCP · WHMC |         133,444 |              0.324× |           0.383× |           6 |
|                          BRM total CP · WHMC |          87,205 |               2.49× |               3× |          61 |
|           BRM total post-hoc position · WHMC |         204,019 |                2.3× |           0.919× |          40 |
|           BRM total post-hoc gradient · WHMC |         238,614 |               2.34× |            1.23× |           7 |
|             BRM total online position · WHMC |          69,353 |              0.226× |           0.269× |          79 |
|             BRM total online gradient · WHMC |          89,144 |                1.3× |            1.55× |          19 |


[
![](assets/rbest-centering/AS-efficiency.png)
](assets/rbest-centering/AS-efficiency.png)

### crohn: 8 quantities {#crohn:-8-quantities}

|                                       Method | Total gradients | Sampling efficiency | Total efficiency | Divergences |
| --------------------------------------------:| ---------------:| -------------------:| ----------------:| -----------:|
|   RBesT 1.11 NCP (its control) · Native Stan |         466,164 |                  1× |               1× |           0 |
|    RBesT 1.11 CP (its control) · Native Stan |         446,598 |              0.377× |           0.376× |          53 |
| RBesT 1.11 NCP (Stan defaults) · Native Stan |         224,025 |               1.79× |               2× |           8 |
|  RBesT 1.11 CP (Stan defaults) · Native Stan |         163,427 |               1.73× |            1.92× |         124 |
|             RBesT PR64 S2Z NCP · Native Stan |         246,616 |               2.33× |            2.29× |           0 |
|              RBesT PR64 S2Z CP · Native Stan |         167,317 |               2.49× |            2.41× |          42 |
|                      BRM ordinary NCP · WHMC |         194,050 |               1.18× |            1.38× |           7 |
|                       BRM ordinary CP · WHMC |          97,820 |               1.46× |            1.63× |          44 |
|        BRM ordinary post-hoc position · WHMC |         315,028 |               3.55× |            1.63× |           9 |
|        BRM ordinary post-hoc gradient · WHMC |         292,381 |               3.48× |             1.4× |          12 |
|          BRM ordinary online position · WHMC |         138,160 |               2.82× |            3.38× |           0 |
|          BRM ordinary online gradient · WHMC |          99,443 |               3.55× |            4.25× |          47 |
|                         BRM total NCP · WHMC |         387,777 |             0.0844× |          0.0349× |         510 |
|                          BRM total CP · WHMC |          70,628 |                2.9× |            3.13× |         134 |
|           BRM total post-hoc position · WHMC |         460,830 |               3.88× |           0.731× |          42 |
|           BRM total post-hoc gradient · WHMC |         464,218 |               4.36× |           0.854× |          59 |
|             BRM total online position · WHMC |          69,343 |               3.59× |            4.28× |          38 |
|             BRM total online gradient · WHMC |          71,425 |               6.42× |            7.65× |          62 |


[
![](assets/rbest-centering/crohn-efficiency.png)
](assets/rbest-centering/crohn-efficiency.png)

## Saved-draw geometry {#Saved-draw-geometry}

Each figure reuses the same 10,000 post-hoc-position draws across its columns, with CP as the visualization baseline; rows select the trials with minimum, middle and maximum inferred centeredness. The interactive previews in the KB brief show every fifth draw.

### AS {#AS-2}

[
![](assets/rbest-centering/AS-total_pairs.png)
](assets/rbest-centering/AS-total_pairs.png)

[
![](assets/rbest-centering/AS-ordinary_pairs.png)
](assets/rbest-centering/AS-ordinary_pairs.png)

### crohn {#crohn-2}

[
![](assets/rbest-centering/crohn-total_pairs.png)
](assets/rbest-centering/crohn-total_pairs.png)

[
![](assets/rbest-centering/crohn-ordinary_pairs.png)
](assets/rbest-centering/crohn-ordinary_pairs.png)

## Diagnostics, recovery and limits {#Diagnostics,-recovery-and-limits}

Every arm is one chain of 10,000 draws; these are pilots, not replicated rankings. Divergences: on AS, 0 for RBesT legacy NCP and both S2Z NCP fits, 533 and 670 for legacy CP, 9 for S2Z CP, 1 to 14 for BRM's conventional arms, and 6 to 79 for the totals arms; on crohn, 0 for legacy NCP and S2Z NCP, 53 and 124 for legacy CP, 42 for S2Z CP, 0 to 47 for the conventional arms (online position 0), and 38 to 510 for the totals arms (NCP 510). The centered conventional AS arm also had 7 Stan-side numerical rejections, charged and rejected as native Stan does.

Posterior means were compared to the RBesT legacy NCP baseline in units of the combined MCSE over all quantities and arms. On AS every one of the 180 comparisons is below 3 (maximum 2.75). On crohn three arms exceed 3 on one trial total each: legacy CP under CmdStan defaults (3.09), BRM total CP (3.11) and BRM total online position (4.67), all divergent arms; the rest are below 3. Recovering the integrated population mean with ten different seeds leaves every totals arm's minimum ESS unchanged, because the between-trial SD is the limiting quantity in every arm; only the population mean's own ESS moves with the seed.

RBesT's default thinning (4) was not applied; its warmup and control were. Native counts come from the same zero-contribution C++ gradient counter as the other pages, checked against leapfrogs plus one per retained transition.

Both BRM targets and both wrappers passed their density, gradient, transport and finite-difference checks, and BRM's targets were evaluated against RBesT's own compiled programs at mapped points: the legacy program against the conventional block and the sum-to-zero program against the exact totals differ by a constant Jacobian only, with identical trial effects. BRM omits the constant half-Normal normalizer of the bounded SD declaration; adding `log 2` aligns the densities. Out of scope on this page: RBesT's other `tau` prior families, per-stratum `tau`, Student-t random effects, and RBesT's unpublished auto-centering.

## Inspect and reproduce {#Inspect-and-reproduce}
- [BRM models, priors and reference densities](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/model.jl).
    
- [Density, gradient, wrapper and RBesT cross-check audit](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/audit.jl).
    
- [Twelve WarmupHMC arms](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/run.jl).
    
- [RBesT data and program capture](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/capture.R).
    
- [Native RBesT arms under CmdStan](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/native.R).
    
- [Native-arm diagnostics](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/native_analysis.jl).
    
- [Recovery sensitivity and MCSE checks](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/complete.jl).
    
- [Saved-draw pairs](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/pairs.jl).
    
- [Full raw-fit archive manifest](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/results/fits/manifest.json).
    
- [Primary-source model check and reproduction](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/README.md).
    
- [AS: comparison table](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/results/AS/comparison.tsv).
    
- [AS: per-quantity MCSE](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/results/AS/per_quantity_mcse.tsv).
    
- [AS: recovery sensitivity](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/results/AS/recovery_seed_sensitivity.tsv).
    
- [AS: MCSE summary](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/results/AS/mcse_summary.tsv).
    
- [AS, RBesT legacy: Stan](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/AS/legacy/clean.stan).
    
- [AS, RBesT legacy: data (NCP)](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/AS/legacy/data-ncp.json).
    
- [AS, RBesT legacy: data (CP)](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/AS/legacy/data-cp.json).
    
- [AS, RBesT s2z: Stan](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/AS/s2z/clean.stan).
    
- [AS, RBesT s2z: data (NCP)](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/AS/s2z/data-ncp.json).
    
- [AS, RBesT s2z: data (CP)](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/AS/s2z/data-cp.json).
    
- [AS, rbest_ncp: gradient counts](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/AS/rbest_ncp/gradient_counts.tsv).
    
- [AS, rbest_cp: gradient counts](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/AS/rbest_cp/gradient_counts.tsv).
    
- [AS, s2z_ncp: gradient counts](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/AS/s2z_ncp/gradient_counts.tsv).
    
- [AS, s2z_cp: gradient counts](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/AS/s2z_cp/gradient_counts.tsv).
    
- [AS, stan_ncp: gradient counts](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/AS/stan_ncp/gradient_counts.tsv).
    
- [AS, stan_cp: gradient counts](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/AS/stan_cp/gradient_counts.tsv).
    
- [AS: BRM exact-totals Stan](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/automatic_totals/AS/automatic_totals.stan).
    
- [AS: BRM conventional Stan](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/automatic_totals/AS/ordinary_ncp.stan).
    
- [crohn: comparison table](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/results/crohn/comparison.tsv).
    
- [crohn: per-quantity MCSE](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/results/crohn/per_quantity_mcse.tsv).
    
- [crohn: recovery sensitivity](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/results/crohn/recovery_seed_sensitivity.tsv).
    
- [crohn: MCSE summary](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/results/crohn/mcse_summary.tsv).
    
- [crohn, RBesT legacy: Stan](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/crohn/legacy/clean.stan).
    
- [crohn, RBesT legacy: data (NCP)](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/crohn/legacy/data-ncp.json).
    
- [crohn, RBesT legacy: data (CP)](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/crohn/legacy/data-cp.json).
    
- [crohn, RBesT s2z: Stan](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/crohn/s2z/clean.stan).
    
- [crohn, RBesT s2z: data (NCP)](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/crohn/s2z/data-ncp.json).
    
- [crohn, RBesT s2z: data (CP)](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/crohn/s2z/data-cp.json).
    
- [crohn, rbest_ncp: gradient counts](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/crohn/rbest_ncp/gradient_counts.tsv).
    
- [crohn, rbest_cp: gradient counts](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/crohn/rbest_cp/gradient_counts.tsv).
    
- [crohn, s2z_ncp: gradient counts](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/crohn/s2z_ncp/gradient_counts.tsv).
    
- [crohn, s2z_cp: gradient counts](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/crohn/s2z_cp/gradient_counts.tsv).
    
- [crohn, stan_ncp: gradient counts](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/crohn/stan_ncp/gradient_counts.tsv).
    
- [crohn, stan_cp: gradient counts](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/native/crohn/stan_cp/gradient_counts.tsv).
    
- [crohn: BRM exact-totals Stan](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/automatic_totals/crohn/automatic_totals.stan).
    
- [crohn: BRM conventional Stan](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/reference/automatic_totals/crohn/ordinary_ncp.stan).
    

RBesT: `3d5fa1dda74f9d68360094a11983f32dd1b61c10` (1.11-0) and pull request 64 head `a5acbbc39c2cd620f0549159aa7c1991d4c28d6a`; WHMC `7aed40b18bd4cdabb75330f285d5c9b355575ab9`; Julia 1.10.11, CmdStan 2.39.0, CmdStanR 0.9.0.
