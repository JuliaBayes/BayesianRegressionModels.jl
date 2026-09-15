```brm-comparison
pupil numeric brm model
```

```julia
using BayesianRegressionModels, Distributions, JSON

function pupil_numeric_brm_model()
    reference = JSON.parsefile(joinpath(pkgdir(BayesianRegressionModels),
        "research", "pupil_scale_totals", "reference", "ordinary_ncp.json"))
    data = (;p_size=Float64.(reference["Y"]), load=Float64.(reference["Z_1_2"]),
             subj=Int.(reference["J_1"]))
    data = merge(data, (;subject_id=700 .+ data.subj))
    
    builder = @brm begin
        mu ~ 1 + center(load) + (1 | mean_intercept | subj) + (0 + load | mean_slope | subj)
        logsigma ~ 1 + center(subject_id)
        effect(mu, Intercept) ~ LocationScale(5651.9,2026.1,TDist(3))
        effect(mu, center_load) ~ Flat()
        effect(logsigma, Intercept) ~ LocationScale(0.,2.5,TDist(3))
        effect(logsigma, center_subject_id) ~ Flat()
        sd(:, mean_intercept) ~ LocationScale(0.,2026.1,TDist(3))
        sd(:, mean_slope) ~ LocationScale(0.,2026.1,TDist(3))
        p_size ~ Normal(mu,exp(logsigma))
    end
    
    builder(data)
end
```

```julia
SBBRMI with data keys = [:center_subject_id, :load, :p_size, :subj, :subject_id, :total_A_mu, :total_group_mu, :total_location_mu, :total_mixture_shape_mu, :total_ng_mu, :total_nk_mu, :total_nm_mu, :total_np_mu, :total_precision_mu]
configured submodels:
_brm_total_scales_configured_1 = Base.merge(BayesianRegressionModels._brm_total_scales, quote
            tau ~ (BayesianRegressionModels.brm_vector_prior_eecc99814bef47ac)(3.0, 0.0, 2026.1, 3.0, 0.0, 2026.1; n = 2, lower = 0.0)
        end)
_popefs_generic_configured_1 = Base.merge(BayesianRegressionModels._popefs_generic, quote
            beta_pop ~ (BayesianRegressionModels.brm_vector_prior_38a4c9e64638c44f)(3.0, 0.0, 2.5; n = 2)
        end)
emitted @slic body:
begin
    total_scale_mu ~ _brm_total_scales_configured_1(; n = total_nk_mu)
    total_mixture_mu::vector[total_nm_mu] ~ gamma(total_mixture_shape_mu, total_mixture_shape_mu; lower = 0.0)
    total_conditional_precision_mu = [total_precision_mu[1] * total_mixture_mu[1], total_precision_mu[2]]
    total_mu::matrix[total_ng_mu, total_nk_mu] ~ brm_total(total_scale_mu, total_A_mu, total_location_mu, total_conditional_precision_mu)
    population_mu = brm_total_recover_rng(total_mu, total_scale_mu, total_A_mu, total_location_mu, total_conditional_precision_mu)
    deviation_mu = brm_total_deviations(total_mu, total_A_mu * population_mu)
    total_Z_mu = hcat(rep_vector(1.0, num_elements(total_group_mu)), load)
    mu = rows_dot_product(total_mu[total_group_mu, :], total_Z_mu)
    X_logsigma = hcat(rep_vector(1.0, num_elements(subject_id)), center_subject_id)
    pop_logsigma ~ _popefs_generic_configured_1(; X = X_logsigma)
    logsigma = pop_logsigma
    p_size ~ normal(mu, (exp)(logsigma))
end
```

```stan
functions {
real brm_vector_prior_eecc99814bef47ac_lpdf(
    vector x,
    real arg_1,
    real arg_2,
    real arg_3,
    real arg_4,
    real arg_5,
    real arg_6
) {
    if((x[1] < 0.0)) {
        return negative_infinity();
    }
    if((x[2] < 0.0)) {
        return negative_infinity();
    }
    return (student_t_lpdf(x[1] | arg_1, arg_2, arg_3) + student_t_lpdf(x[2] | arg_4, arg_5, arg_6));
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
matrix hcat(
    vector x,
    vector y
) {
    int n = dims(x)[1];
    if (dims(y)[1] != n) reject("hcat: dim mismatch — `y` dim 1 (= ", dims(y)[1], ") does not match `n` (= ", n, "), inferred from `x` dim 1. `n` sizes: `x` dim 1 (= ", dims(x)[1], "), `y` dim 1 (= ", dims(y)[1], ").");
    return append_col(x, y);
}
real brm_vector_prior_38a4c9e64638c44f_lpdf(
    vector x,
    real arg_1,
    real arg_2,
    real arg_3
) {
    return (student_t_lpdf(x[1] | arg_1, arg_2, arg_3) + 0.0);
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
    int total_nm_mu;
    int total_mixture_shape_mu_n;
    vector[total_mixture_shape_mu_n] total_mixture_shape_mu;
    int total_precision_mu_n;
    vector[total_precision_mu_n] total_precision_mu;
    int total_ng_mu;
    int total_nk_mu;
    int total_A_mu_m;
    int total_A_mu_n;
    matrix[total_A_mu_m, total_A_mu_n] total_A_mu;
    int total_location_mu_n;
    vector[total_location_mu_n] total_location_mu;
    int load_n;
    int total_group_mu_n;
    array[total_group_mu_n] int total_group_mu;
    vector[load_n] load;
    int center_subject_id_n;
    int subject_id_n;
    array[subject_id_n] int subject_id;
    vector[center_subject_id_n] center_subject_id;
    int p_size_n;
    vector[p_size_n] p_size;
}
transformed data {
    matrix[load_n, 2] total_Z_mu = hcat(rep_vector(1.0, num_elements(total_group_mu)), load);
    matrix[center_subject_id_n, 2] X_logsigma = hcat(rep_vector(1.0, num_elements(subject_id)), center_subject_id);
    int pop_logsigma_n_covariates = 2;
}
parameters {
    vector<lower=0.0>[2] total_scale_mu_tau;
    vector<lower=0.0>[total_nm_mu] total_mixture_mu;
    matrix[total_ng_mu, total_nk_mu] total_mu;
    vector[pop_logsigma_n_covariates] pop_logsigma_beta_pop;
}
transformed parameters {
    vector<lower=0.0>[2] total_scale_mu = total_scale_mu_tau;
    vector[2] total_conditional_precision_mu = [(total_precision_mu[1] * total_mixture_mu[1]), total_precision_mu[2]]';
    vector[load_n] mu = rows_dot_product(total_mu[total_group_mu, :], total_Z_mu);
    vector[center_subject_id_n] pop_logsigma = (X_logsigma * pop_logsigma_beta_pop);
    vector[center_subject_id_n] logsigma = pop_logsigma;
}
model {
    total_scale_mu_tau ~ brm_vector_prior_eecc99814bef47ac(3.0, 0.0, 2026.1, 3.0, 0.0, 2026.1);
    total_mixture_mu ~ gamma(total_mixture_shape_mu, total_mixture_shape_mu);
    total_mu ~ brm_total(total_scale_mu, total_A_mu, total_location_mu, total_conditional_precision_mu);
    pop_logsigma_beta_pop ~ brm_vector_prior_38a4c9e64638c44f(3.0, 0.0, 2.5);
    p_size ~ normal(mu, exp(logsigma));
}
generated quantities {
    vector[2] population_mu = brm_total_recover_rng(
        total_mu,
        total_scale_mu,
        total_A_mu,
        total_location_mu,
        total_conditional_precision_mu
    );
    matrix[total_ng_mu, total_A_mu_m] deviation_mu = brm_total_deviations(total_mu, (total_A_mu * population_mu));
    vector[p_size_n] p_size_likelihood = normal_lpdfs(p_size, mu, exp(logsigma));
    vector[p_size_n] p_size_gen = normal_vector_rng(p_size_n, mu, exp(logsigma));
}
```

```julia
#= line 0 =# Turing.@model(function brm_model(y, callable_1, X_mu, random_effects_mu_1, callable_2, random_effects_mu_2, callable_3, callable_4, X_logsigma)
        beta_pop ~ Distributions.product_distribution([callable_1(5651.9, 2026.1, Distributions.TDist(3)), BayesianRegressionModels.Flat()])
        eta_mu = X_mu * beta_pop
        group_effect_1 = Base.zeros(Base.length(y))
        group_1_1 ~ DynamicPPL.to_submodel(BayesianRegressionModelsTuringExt._brm_group_effect_model(random_effects_mu_1, (callable_2(0.0, 2026.1, Distributions.TDist(3)),), nothing))
        group_effect_1 = group_effect_1 + group_1_1.effect
        group_1_2 ~ DynamicPPL.to_submodel(BayesianRegressionModelsTuringExt._brm_group_effect_model(random_effects_mu_2, (callable_3(0.0, 2026.1, Distributions.TDist(3)),), nothing))
        group_effect_1 = group_effect_1 + group_1_2.effect
        eta_mu = eta_mu + group_effect_1
        mu = eta_mu
        beta_pop_logsigma ~ Distributions.product_distribution([callable_4(0.0, 2.5, Distributions.TDist(3)), BayesianRegressionModels.Flat()])
        eta_logsigma = X_logsigma * beta_pop_logsigma
        logsigma = eta_logsigma
        begin
            for i = Base.eachindex(y)
                y[i] ~ Distributions.Normal(mu[i], Base.exp(logsigma[i]))
            end
        end
        (; mu = mu, logsigma = logsigma, response = y)
    end)
```
