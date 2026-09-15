// generated with brms 2.23.1
functions {
  // Explicit substitution avoids general matrix solves for one or two rows.
  vector mdivide_left_tri_low_brms(matrix L, vector b) {
    int K = rows(L);
    if (cols(L) != K || num_elements(b) != K) {
      return mdivide_left_tri_low(L, b);
    }
    if (K == 1 && L[1, 1] != 0.0) {
      return b / L[1, 1];
    }
    if (K == 2 && L[1, 1] != 0.0 && L[2, 2] != 0.0) {
      vector[2] x;
      x[1] = b[1] / L[1, 1];
      x[2] = (b[2] - L[2, 1] * x[1]) / L[2, 2];
      return x;
    }
    return mdivide_left_tri_low(L, b);
  }

  matrix mdivide_left_tri_low_brms(matrix L, matrix B) {
    int K = rows(L);
    if (cols(L) != K || rows(B) != K) {
      return mdivide_left_tri_low(L, B);
    }
    if (K == 1 && L[1, 1] != 0.0) {
      return B / L[1, 1];
    }
    if (K == 2 && L[1, 1] != 0.0 && L[2, 2] != 0.0) {
      matrix[2, cols(B)] X;
      X[1] = B[1] / L[1, 1];
      X[2] = (B[2] - L[2, 1] * X[1]) / L[2, 2];
      return X;
    }
    return mdivide_left_tri_low(L, B);
  }

  row_vector mdivide_right_tri_low_brms(row_vector b, matrix L) {
    int K = rows(L);
    if (cols(L) != K || num_elements(b) != K) {
      return mdivide_right_tri_low(b, L);
    }
    if (K == 1 && L[1, 1] != 0.0) {
      return b / L[1, 1];
    }
    if (K == 2 && L[1, 1] != 0.0 && L[2, 2] != 0.0) {
      row_vector[2] x;
      x[2] = b[2] / L[2, 2];
      x[1] = (b[1] - x[2] * L[2, 1]) / L[1, 1];
      return x;
    }
    return mdivide_right_tri_low(b, L);
  }

  matrix cholesky_decompose_brms(matrix A) {
    int K = rows(A);
    if (cols(A) != K) {
      return cholesky_decompose(A);
    }
    if (K == 1 && A[1, 1] > 0.0 && !is_inf(A[1, 1])) {
      return rep_matrix(sqrt(A[1, 1]), 1, 1);
    }
    if (K == 2 && A[1, 2] == A[2, 1] && A[1, 1] > 0.0 &&
        !is_inf(A[1, 1]) && !is_inf(A[2, 1]) && !is_inf(A[2, 2])) {
      matrix[2, 2] L = rep_matrix(0.0, 2, 2);
      real pivot;
      L[1, 1] = sqrt(A[1, 1]);
      L[2, 1] = A[2, 1] / L[1, 1];
      pivot = A[2, 2] - square(L[2, 1]);
      if (pivot > 0.0) {
        L[2, 2] = sqrt(pivot);
        return L;
      }
    }
    // Preserve Stan's validation and treatment of numerical boundary cases.
    return cholesky_decompose(A);
  }

  matrix chol2inv_brms(matrix L) {
    int K = rows(L);
    if (cols(L) != K) {
      return chol2inv(L);
    }
    if (K == 1 && L[1, 1] != 0.0) {
      return rep_matrix(inv_square(L[1, 1]), 1, 1);
    }
    if (K == 2 && L[1, 2] == 0.0 &&
        L[1, 1] != 0.0 && L[2, 2] != 0.0) {
      matrix[2, 2] precision;
      real a = inv(L[1, 1]);
      real d = inv(L[2, 2]);
      real c = -L[2, 1] * a / L[2, 2];
      precision[1, 1] = square(a) + square(c);
      precision[1, 2] = c * d;
      precision[2, 1] = precision[1, 2];
      precision[2, 2] = square(d);
      return precision;
    }
    return chol2inv(L);
  }
  vector sum_to_zero_constrain_brms(vector y) {
    int N = num_elements(y);
    vector[N + 1] z = zeros_vector(N + 1);
    real sum_w = 0;
    for (ii in 1:N) {
      int i = N - ii + 1;
      real w = y[i] * inv_sqrt(i * (i + 1.0));
      sum_w += w;
      z[i] += sum_w;
      z[i + 1] -= i * w;
    }
    return z;
  }

  real s2z_require_finite_brms(real x) {
    if (is_nan(x) || is_inf(x)) {
      reject("S2Z population-prior locations must be finite.");
    }
    return x;
  }

  real s2z_require_positive_brms(real x) {
    if (is_nan(x) || is_inf(x) || x <= 0) {
      reject("S2Z population-prior scales and degrees of freedom must be ",
             "finite and strictly positive.");
    }
    return x;
  }

  real s2z_prior_coordinate_brms(real x, int index, int expected_size) {
    return x;
  }

  real s2z_prior_coordinate_brms(vector x, int index, int expected_size) {
    if (num_elements(x) != expected_size) {
      reject("An S2Z vector-valued population-prior argument must have one ",
             "entry per population-level coefficient.");
    }
    return x[index];
  }

  real s2z_prior_coordinate_brms(row_vector x, int index,
                                 int expected_size) {
    if (num_elements(x) != expected_size) {
      reject("An S2Z vector-valued population-prior argument must have one ",
             "entry per population-level coefficient.");
    }
    return x[index];
  }

  real s2z_prior_coordinate_brms(array[] real x, int index,
                                 int expected_size) {
    if (num_elements(x) != expected_size) {
      reject("An S2Z vector-valued population-prior argument must have one ",
             "entry per population-level coefficient.");
    }
    return x[index];
  }

  real s2z_prior_coordinate_brms(array[] int x, int index,
                                 int expected_size) {
    if (num_elements(x) != expected_size) {
      reject("An S2Z vector-valued population-prior argument must have one ",
             "entry per population-level coefficient.");
    }
    return x[index];
  }
  real pupil_record_gradient(real x);
real pupil_gradient_count();
}
data {
  int<lower=1> N;  // total number of observations
  vector[N] Y;  // response variable
  int<lower=1> K;  // number of population-level effects
  matrix[N, K] X;  // population-level design matrix
  int<lower=1> Kc;  // number of population-level effects after centering
  // data for group-level effects of ID 1
  int<lower=1> N_1;  // number of grouping levels
  int<lower=1> M_1;  // number of coefficients per level
  array[N] int<lower=1> J_1;  // grouping indicator per observation
  // group-level predictor values
  vector[N] Z_1_1;
  vector[N] Z_1_2;
  matrix<lower=0,upper=1>[N_1, M_1] rho_s2z_1;  // fixed precursor/final centering fractions
  int<lower=0,upper=1> compute_rho_center_candidate_1;  // evaluate the precursor proposal in generated quantities?
  // data for group-level effects of ID 2
  int<lower=1> N_2;  // number of grouping levels
  int<lower=1> M_2;  // number of coefficients per level
  array[N] int<lower=1> J_2;  // grouping indicator per observation
  // group-level predictor values
  vector[N] Z_2_sigma_1;
  matrix<lower=0,upper=1>[N_2, M_2] rho_s2z_2;  // fixed precursor/final centering fractions
  int<lower=0,upper=1> compute_rho_center_candidate_2;  // evaluate the precursor proposal in generated quantities?
  int prior_only;  // should the likelihood be ignored?
}
transformed data {
  matrix[N, Kc] Xc;  // centered version of X without an intercept
  vector[Kc] means_X;  // column means of X before centering
  vector[M_1] mean_rho_s2z_1;
  vector[M_1] intercept_map_s2z_1;
  vector[M_2] mean_rho_s2z_2;
  vector[1] H_s2z_2;
  for (i in 2:K) {
    means_X[i - 1] = mean(X[, i]);
    Xc[, i - 1] = X[, i] - means_X[i - 1];
  }
  for (k in 1:M_1) {
    mean_rho_s2z_1[k] = mean(rho_s2z_1[, k]);
  }
  intercept_map_s2z_1 = zeros_vector(M_1);
  intercept_map_s2z_1[1] = 1.0;
  intercept_map_s2z_1[2] = means_X[1];
  for (k in 1:M_2) {
    mean_rho_s2z_2[k] = mean(rho_s2z_2[, k]);
  }
  H_s2z_2 = zeros_vector(1);
  H_s2z_2[1] = 1.0;
}
parameters {
  vector[2] theta_s2z;  // finite-population coefficients for physical S2Z effects
  vector[1] theta_s2z_sigma;  // finite-population coefficients for physical S2Z effects
  vector<lower=0>[M_1] sd_1;  // group-level standard deviations
  vector[M_1 * (N_1 - 1)] z_s2z_1;  // partially centered orthonormal independent S2Z coordinates
  real<lower=0> udf_b_s2z_1;  // mixing variable for population coefficient 1
  vector<lower=0>[M_2] sd_2;  // group-level standard deviations
  vector[N_2 - 1] z_s2z_2;  // partially centered orthonormal scalar S2Z coordinates
  real<lower=0> udf_b_s2z_sigma_1;  // mixing variable for population coefficient 1
}
transformed parameters {
  // component-wise physical S2Z effects of ID 1
  vector[N_1] r_s2z_1_1;
  vector[N_1] r_s2z_1_2;
  vector[2] prior_mean_s2z_1;
  vector<lower=0>[2] prior_prec_s2z_1;
  vector<lower=0>[M_1] D_diag_s2z_1;
  real<lower=0> rank1_info_s2z_1;
  vector[M_1] mhat_s2z_1;
  vector[2] qhat_s2z_1;
  real<lower=0> group_quad_s2z_1;
  real log_det_partial_s2z_1;
  // specialized scalar physical S2Z effects of ID 2
  vector[N_2] r_s2z_2_sigma_1;
  vector[1] prior_mean_s2z_2;
  vector<lower=0>[1] prior_prec_s2z_2;
  real<lower=0> D_s2z_2;
  real<lower=0> sqrt_D_s2z_2;
  real mhat_s2z_2;
  vector[1] qhat_s2z_2;
  real<lower=0> group_quad_s2z_2;
  real log_det_partial_s2z_2;
  // prior contributions to the log posterior
  real lprior = 0;
  real pupil_counter_tick = pupil_record_gradient(theta_s2z[1]);
  log_det_partial_s2z_1 = 0.0;
  {
    vector[N_1] centered_partial_s2z = sum_to_zero_constrain_brms(segment(z_s2z_1, (1 - 1) * (N_1 - 1) + 1, N_1 - 1));
    vector[N_1] scale_partial_s2z = 1.0 - rho_s2z_1[, 1] + rho_s2z_1[, 1] * sd_1[1];
    centered_partial_s2z = sd_1[1] * centered_partial_s2z ./ scale_partial_s2z;
    r_s2z_1_1 = centered_partial_s2z - mean(centered_partial_s2z);
    log_det_partial_s2z_1 += -sum(log(scale_partial_s2z));
    log_det_partial_s2z_1 += log(
      (1.0 - mean_rho_s2z_1[1]) + mean_rho_s2z_1[1] * sd_1[1]
    );
  }
  {
    vector[N_1] centered_partial_s2z = sum_to_zero_constrain_brms(segment(z_s2z_1, (2 - 1) * (N_1 - 1) + 1, N_1 - 1));
    vector[N_1] scale_partial_s2z = 1.0 - rho_s2z_1[, 2] + rho_s2z_1[, 2] * sd_1[2];
    centered_partial_s2z = sd_1[2] * centered_partial_s2z ./ scale_partial_s2z;
    r_s2z_1_2 = centered_partial_s2z - mean(centered_partial_s2z);
    log_det_partial_s2z_1 += -sum(log(scale_partial_s2z));
    log_det_partial_s2z_1 += log(
      (1.0 - mean_rho_s2z_1[2]) + mean_rho_s2z_1[2] * sd_1[2]
    );
  }
  prior_mean_s2z_1[1] = 5651.8999999999996;
  prior_prec_s2z_1[1] = inv_square(2026.0999999999999 * sqrt(3 * udf_b_s2z_1));
  prior_mean_s2z_1[2] = 0;
  prior_prec_s2z_1[2] = 0.0;
  {
    vector[M_1] base_info_s2z = zeros_vector(M_1);
    vector[M_1] base_score_s2z = zeros_vector(M_1);
    vector[M_1] scaled_score_s2z;
    vector[M_1] independent_mode_s2z;
    real group_info_s2z = N_1;
    base_info_s2z[2] = prior_prec_s2z_1[2];
    base_score_s2z[2] = prior_prec_s2z_1[2] * (theta_s2z[2] - prior_mean_s2z_1[2]);
    D_diag_s2z_1 = group_info_s2z + square(sd_1) .* base_info_s2z;
    scaled_score_s2z[1] = square(sd_1[1]) * base_score_s2z[1];
    scaled_score_s2z[2] = square(sd_1[2]) * base_score_s2z[2];
    independent_mode_s2z = scaled_score_s2z ./ D_diag_s2z_1;
    rank1_info_s2z_1 = prior_prec_s2z_1[1] * dot_product(
      square(sd_1) .* square(intercept_map_s2z_1),
      1.0 ./ D_diag_s2z_1
    );
    mhat_s2z_1 = independent_mode_s2z +
      prior_prec_s2z_1[1] * square(sd_1) .* intercept_map_s2z_1 ./
      D_diag_s2z_1 * (theta_s2z[1] - prior_mean_s2z_1[1] -
      dot_product(intercept_map_s2z_1, independent_mode_s2z)) /
      (1.0 + rank1_info_s2z_1);
  }
  qhat_s2z_1 = theta_s2z;
  qhat_s2z_1[1] -= dot_product(intercept_map_s2z_1, mhat_s2z_1);
  qhat_s2z_1[2] -= mhat_s2z_1[2];
  group_quad_s2z_1 = 0.0;
  group_quad_s2z_1 += dot_self((r_s2z_1_1 + mhat_s2z_1[1]) / sd_1[1]);
  group_quad_s2z_1 += dot_self((r_s2z_1_2 + mhat_s2z_1[2]) / sd_1[2]);
  log_det_partial_s2z_2 = 0.0;
  {
    vector[N_2] centered_partial_s2z = sum_to_zero_constrain_brms(z_s2z_2);
    vector[N_2] scale_partial_s2z = 1.0 - rho_s2z_2[, 1] + rho_s2z_2[, 1] * sd_2[1];
    centered_partial_s2z = sd_2[1] * centered_partial_s2z ./ scale_partial_s2z;
    r_s2z_2_sigma_1 = centered_partial_s2z - mean(centered_partial_s2z);
    log_det_partial_s2z_2 += -sum(log(scale_partial_s2z));
    log_det_partial_s2z_2 += log(
      (1.0 - mean_rho_s2z_2[1]) + mean_rho_s2z_2[1] * sd_2[1]
    );
  }
  prior_mean_s2z_2[1] = 0;
  prior_prec_s2z_2[1] = inv_square(2.5 * sqrt(3 * udf_b_s2z_sigma_1));
  {
    real tau_sq_s2z = square(sd_2[1]);
    real prior_info_s2z = dot_product(prior_prec_s2z_2, square(H_s2z_2));
    real prior_score_s2z = dot_product(H_s2z_2, prior_prec_s2z_2 .* (theta_s2z_sigma - prior_mean_s2z_2));
    D_s2z_2 = tau_sq_s2z * prior_info_s2z + N_2;
    mhat_s2z_2 = tau_sq_s2z * prior_score_s2z / D_s2z_2;
  }
  sqrt_D_s2z_2 = sqrt(D_s2z_2);
  qhat_s2z_2 = theta_s2z_sigma - H_s2z_2 * mhat_s2z_2;
  {
    vector[N_2] white_s2z = (r_s2z_2_sigma_1 + mhat_s2z_2) / sd_2[1];
    group_quad_s2z_2 = dot_self(white_s2z);
  }
  lprior += student_t_lpdf(sd_1 | 3, 0, 2026.1)
    - 2 * student_t_lccdf(0 | 3, 0, 2026.1);
  lprior += inv_chi_square_lpdf(udf_b_s2z_1 | 3);
  lprior += normal_lpdf(qhat_s2z_1[1] | 5651.8999999999996, 2026.0999999999999 * sqrt(3 * udf_b_s2z_1));
  lprior += -0.5 * group_quad_s2z_1
    + log_det_partial_s2z_1
    - 0.5 * sum(log(D_diag_s2z_1))
    - 0.5 * log1p(rank1_info_s2z_1) - 0.5 * N_1 * M_1 * log(2 * pi()) + 0.5 * M_1 * log(2 * pi()) + 0.5 * M_1 * log(1.0 * N_1);
  lprior += student_t_lpdf(sd_2 | 3, 0, 2026.1)
    - 1 * student_t_lccdf(0 | 3, 0, 2026.1);
  lprior += inv_chi_square_lpdf(udf_b_s2z_sigma_1 | 3);
  lprior += normal_lpdf(qhat_s2z_2[1] | 0, 2.5 * sqrt(3 * udf_b_s2z_sigma_1));
  lprior += -0.5 * group_quad_s2z_2
    + log_det_partial_s2z_2
    - 0.5 * log(D_s2z_2) - 0.5 * N_2 * log(2 * pi()) + 0.5 * log(2 * pi()) + 0.5 * log(1.0 * N_2);
}
model {
  target += pupil_counter_tick;
  // likelihood including constants
  if (!prior_only) {
    // initialize linear predictor term
    vector[N] mu = rep_vector(0.0, N);
    // initialize linear predictor term
    vector[N] sigma = rep_vector(0.0, N);
    mu += theta_s2z[1] + Xc * tail(theta_s2z, 1);
    sigma += theta_s2z_sigma[1];
    for (n in 1:N) {
      // add more terms to the linear predictor
      mu[n] += r_s2z_1_1[J_1[n]] * Z_1_1[n] + r_s2z_1_2[J_1[n]] * Z_1_2[n];
    }
    for (n in 1:N) {
      // add more terms to the linear predictor
      sigma[n] += r_s2z_2_sigma_1[J_2[n]] * Z_2_sigma_1[n];
    }
    sigma = exp(sigma);
    target += normal_lpdf(Y | mu, sigma);
  }
  // priors including constants
  target += lprior;
}
generated quantities {
  matrix<lower=0,upper=1>[N_1, M_1] rho_center_candidate_1;
  vector<lower=0,upper=1>[M_1] mean_rho_center_candidate_1;
  vector[M_1] mean_r_s2z_1;
  vector[2] q_recovered_s2z_1;
  real Intercept;
  vector[Kc] b;
  real b_Intercept;
  vector[N_1] r_1_1;
  vector[N_1] r_1_2;
  matrix<lower=0,upper=1>[N_2, M_2] rho_center_candidate_2;
  vector<lower=0,upper=1>[M_2] mean_rho_center_candidate_2;
  real mean_r_s2z_2;
  vector[1] q_recovered_s2z_2;
  real Intercept_sigma;
  real b_sigma_Intercept;
  vector[N_2] r_2_sigma_1;
  if (compute_rho_center_candidate_1) {
  {
    array[N_1] matrix[M_1, M_1] info_fisher_s2z;
    array[N_1] matrix[M_1, M_1] white_post_cov_fisher_s2z;
    matrix[M_1, M_1] sum_white_post_cov_fisher_s2z = rep_matrix(0.0, M_1, M_1);
    matrix[M_1, M_1] L_sum_white_post_cov_fisher_s2z;
    real restricted_prior_fraction_fisher_s2z = 1.0 - inv(N_1);
    for (j in 1:N_1) {
      info_fisher_s2z[j] = rep_matrix(0.0, M_1, M_1);
    }
    {
      for (n in 1:N) {
        real eta_fisher_s2z_mu = (theta_s2z[1] + dot_product(Xc[n], tail(theta_s2z, 1)) + r_s2z_1_1[J_1[n]] * Z_1_1[n] + r_s2z_1_2[J_1[n]] * Z_1_2[n]);
        real value_fisher_s2z_mu = (eta_fisher_s2z_mu);
        real derivative_fisher_s2z_mu = 1.0;
        real eta_fisher_s2z_sigma = (theta_s2z_sigma[1] + r_s2z_2_sigma_1[J_2[n]] * Z_2_sigma_1[n]);
        real value_fisher_s2z_sigma = exp(eta_fisher_s2z_sigma);
        real derivative_fisher_s2z_sigma = value_fisher_s2z_sigma;
        real obs_prec_fisher_s2z = square(derivative_fisher_s2z_mu) * inv_square(value_fisher_s2z_sigma);
        vector[M_1] design_fisher_s2z = zeros_vector(M_1);
        design_fisher_s2z[1] = Z_1_1[n];
        design_fisher_s2z[2] = Z_1_2[n];
        info_fisher_s2z[J_1[n]] += obs_prec_fisher_s2z * design_fisher_s2z * design_fisher_s2z';
      }
    }
    for (j in 1:N_1) {
      matrix[M_1, M_1] K_fisher_s2z = 1.0 * quad_form_diag(info_fisher_s2z[j], sd_1);
      matrix[M_1, M_1] L_post_precision_fisher_s2z;
      matrix[M_1, M_1] white_factor_fisher_s2z;
      K_fisher_s2z = 0.5 * (K_fisher_s2z + K_fisher_s2z');
      L_post_precision_fisher_s2z = cholesky_decompose_brms(
        add_diag(K_fisher_s2z, 1.0)
      );
      white_factor_fisher_s2z = mdivide_left_tri_low_brms(
        L_post_precision_fisher_s2z, diag_matrix(rep_vector(1.0, M_1))
      );
      white_post_cov_fisher_s2z[j] = crossprod(white_factor_fisher_s2z);
      sum_white_post_cov_fisher_s2z += white_post_cov_fisher_s2z[j];
    }
    sum_white_post_cov_fisher_s2z = 0.5 * (sum_white_post_cov_fisher_s2z + sum_white_post_cov_fisher_s2z');
    L_sum_white_post_cov_fisher_s2z = cholesky_decompose_brms(sum_white_post_cov_fisher_s2z);
    for (j in 1:N_1) {
      matrix[M_1, M_1] constraint_factor_fisher_s2z = mdivide_left_tri_low_brms(
        L_sum_white_post_cov_fisher_s2z, white_post_cov_fisher_s2z[j]
      );
      matrix[M_1, M_1] restricted_white_post_cov_fisher_s2z = white_post_cov_fisher_s2z[j] - crossprod(constraint_factor_fisher_s2z);
      for (k in 1:M_1) {
        // Compare posterior and prior variances on the S2Z subspace.
        rho_center_candidate_1[j, k] = fmin(1.0, fmax(0.0, 1.0 -
          restricted_white_post_cov_fisher_s2z[k, k] / restricted_prior_fraction_fisher_s2z));
        rho_center_candidate_1[j, k] = rho_center_candidate_1[j, k] / (rho_center_candidate_1[j, k] + (1.0 - rho_center_candidate_1[j, k]) * (sd_1[k]));
      }
    }
    for (k in 1:M_1) {
      mean_rho_center_candidate_1[k] = mean(rho_center_candidate_1[, k]);
    }
  }
  } else {
    rho_center_candidate_1 = rho_s2z_1;
    mean_rho_center_candidate_1 = mean_rho_s2z_1;
  }
  {
    vector[M_1] independent_noise_s2z;
    real sqrt_rank1_s2z = sqrt(1.0 + rank1_info_s2z_1);
    real rank1_adjust_s2z;
    for (k in 1:M_1) {
      independent_noise_s2z[k] = sd_1[k] * std_normal_rng() / sqrt(D_diag_s2z_1[k]);
    }
    rank1_adjust_s2z = prior_prec_s2z_1[1] / (sqrt_rank1_s2z * (1.0 + sqrt_rank1_s2z)) *
      dot_product(intercept_map_s2z_1, independent_noise_s2z);
    mean_r_s2z_1 = mhat_s2z_1 + independent_noise_s2z -
      rank1_adjust_s2z * square(sd_1) .* intercept_map_s2z_1 ./ D_diag_s2z_1;
  }
  q_recovered_s2z_1 = theta_s2z;
  q_recovered_s2z_1[1] -= dot_product(intercept_map_s2z_1, mean_r_s2z_1);
  q_recovered_s2z_1[2] -= mean_r_s2z_1[2];
  r_1_1 = r_s2z_1_1 + mean_r_s2z_1[1];
  r_1_2 = r_s2z_1_2 + mean_r_s2z_1[2];
  Intercept = q_recovered_s2z_1[1];
  b = tail(q_recovered_s2z_1, Kc);
  b_Intercept = Intercept - dot_product(means_X, b);
  if (compute_rho_center_candidate_2) {
  {
    vector[N_2] info_fisher_s2z = zeros_vector(N_2);
    vector[N_2] relative_post_var_fisher_s2z;
    real restricted_prior_fraction_fisher_s2z = 1.0 - inv(N_2);
    real sum_relative_post_var_fisher_s2z = 0.0;
    {
      for (n in 1:N) {
        real eta_fisher_s2z_sigma = (theta_s2z_sigma[1] + r_s2z_2_sigma_1[J_2[n]] * Z_2_sigma_1[n]);
        real value_fisher_s2z_sigma = exp(eta_fisher_s2z_sigma);
        real derivative_fisher_s2z_sigma = value_fisher_s2z_sigma;
        real obs_prec_fisher_s2z = 2.0 * square(derivative_fisher_s2z_sigma) * inv_square(value_fisher_s2z_sigma);
        info_fisher_s2z[J_2[n]] += obs_prec_fisher_s2z * square(Z_2_sigma_1[n]);
      }
    }
    for (j in 1:N_2) {
      real scaled_info_fisher_s2z = square(sd_2[1]) * info_fisher_s2z[j];
      relative_post_var_fisher_s2z[j] = inv(1.0 + scaled_info_fisher_s2z);
      sum_relative_post_var_fisher_s2z += relative_post_var_fisher_s2z[j];
    }
    for (j in 1:N_2) {
      real restricted_relative_post_var_fisher_s2z = relative_post_var_fisher_s2z[j] - square(relative_post_var_fisher_s2z[j]) / sum_relative_post_var_fisher_s2z;
      // Compare posterior and prior variances on the S2Z subspace.
      rho_center_candidate_2[j, 1] = fmin(1.0, fmax(0.0, 1.0 - restricted_relative_post_var_fisher_s2z / restricted_prior_fraction_fisher_s2z));
      rho_center_candidate_2[j, 1] = rho_center_candidate_2[j, 1] / (rho_center_candidate_2[j, 1] + (1.0 - rho_center_candidate_2[j, 1]) * (sd_2[1]));
    }
    mean_rho_center_candidate_2[1] = mean(rho_center_candidate_2[, 1]);
  }
  } else {
    rho_center_candidate_2 = rho_s2z_2;
    mean_rho_center_candidate_2 = mean_rho_s2z_2;
  }
  mean_r_s2z_2 = mhat_s2z_2 + sd_2[1] * std_normal_rng() / sqrt_D_s2z_2;
  q_recovered_s2z_2 = theta_s2z_sigma - H_s2z_2 * mean_r_s2z_2;
  r_2_sigma_1 = r_s2z_2_sigma_1 + mean_r_s2z_2;
  Intercept_sigma = q_recovered_s2z_2[1];
  b_sigma_Intercept = Intercept_sigma;
  real pupil_gradients = pupil_gradient_count();
}

