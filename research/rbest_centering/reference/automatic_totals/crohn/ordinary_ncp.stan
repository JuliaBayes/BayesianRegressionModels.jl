functions {
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
    int n_terms_map_study;
    int n_study;
    int study_idx_n;
    array[study_idx_n] int study_idx;
    int y_n;
    vector[y_n] y;
    int y_se_n;
    vector[y_se_n] y_se;
}
transformed data {
    matrix[num_elements(study_idx), 1] X_theta = hcat(rep_vector(1.0, num_elements(study_idx)));
    int pop_theta_n_covariates = 1;
}
parameters {
    cholesky_factor_corr[n_terms_map_study] b_map_study_L;
    vector<lower=0.0>[n_terms_map_study] b_map_study_tau;
    vector[(n_terms_map_study * n_study)] b_map_study_z_flat;
    vector[pop_theta_n_covariates] pop_theta_beta_pop;
}
transformed parameters {
    matrix[n_terms_map_study, n_study] b_map_study_z = to_matrix(b_map_study_z_flat, n_terms_map_study, n_study);
    matrix[n_study, n_terms_map_study] b_map_study = ((diag_pre_multiply(b_map_study_tau, b_map_study_L) * b_map_study_z)');
    vector[num_elements(study_idx)] pop_theta = (X_theta * pop_theta_beta_pop);
    vector[study_idx_n] r_theta_map_study = b_map_study[study_idx, 1];
    vector[num_elements(study_idx)] theta = (pop_theta + r_theta_map_study);
}
model {
    b_map_study_L ~ lkj_corr_cholesky(1.0);
    b_map_study_tau ~ normal(0, 44);
    b_map_study_z_flat ~ std_normal();
    pop_theta_beta_pop ~ normal([0]', [88]');
    y ~ normal(theta, y_se);
}
generated quantities {
    vector[y_n] y_likelihood = normal_lpdfs(y, theta, y_se);
    vector[y_n] y_gen = normal_vector_rng(y_n, theta, y_se);
}