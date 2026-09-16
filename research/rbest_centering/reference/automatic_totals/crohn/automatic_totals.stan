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