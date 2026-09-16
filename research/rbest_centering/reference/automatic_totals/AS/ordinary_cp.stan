functions {
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
    int n_terms_map_study;
    int n_study;
    int study_idx_n;
    array[study_idx_n] int study_idx;
    int r_n;
    array[r_n] int r;
    int n_n;
    array[n_n] int n;
}
transformed data {
    vector[n_terms_map_study] b_map_study_b_cols__pl_inv1_1 = rep_vector(0.0, n_terms_map_study);
    matrix[num_elements(study_idx), 1] X_eta = hcat(rep_vector(1.0, num_elements(study_idx)));
    int pop_eta_n_covariates = 1;
}
parameters {
    cholesky_factor_corr[n_terms_map_study] b_map_study_L;
    vector<lower=0.0>[n_terms_map_study] b_map_study_tau;
    array[n_study] vector[n_terms_map_study] b_map_study_b_cols_bc;
    vector[pop_eta_n_covariates] pop_eta_beta_pop;
}
transformed parameters {
    matrix[n_terms_map_study, n_terms_map_study] b_map_study_b_cols__pl_inv2_1 = diag_pre_multiply(b_map_study_tau, b_map_study_L);
    matrix[n_terms_map_study, n_study] b_map_study_b_cols;
    for(b_map_study_plate_i__pl_1 in 1:n_study) {
        b_map_study_b_cols[:, b_map_study_plate_i__pl_1] = b_map_study_b_cols_bc[b_map_study_plate_i__pl_1];
    }
    matrix[n_study, n_terms_map_study] b_map_study = (b_map_study_b_cols');
    vector[num_elements(study_idx)] pop_eta = (X_eta * pop_eta_beta_pop);
    vector[study_idx_n] r_eta_map_study = b_map_study[study_idx, 1];
    vector[num_elements(study_idx)] eta = (pop_eta + r_eta_map_study);
}
model {
    b_map_study_L ~ lkj_corr_cholesky(1.0);
    b_map_study_tau ~ normal(0, 1);
    b_map_study_b_cols_bc ~ multi_normal_cholesky(b_map_study_b_cols__pl_inv1_1, b_map_study_b_cols__pl_inv2_1);
    pop_eta_beta_pop ~ normal([0]', [2]');
    r ~ binomial_logit(n, eta);
}
generated quantities {
    vector[r_n] r_likelihood = binomial_logit_lpmfs(r, n, eta);
    array[n_n] int r_gen = binomial_logit_int_rng(r_n, n, eta);
}