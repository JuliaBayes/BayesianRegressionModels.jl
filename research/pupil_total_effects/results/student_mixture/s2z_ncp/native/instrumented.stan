// generated with brms 2.23.1
functions {
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
  int<lower=1> K_sigma;  // number of population-level effects
  matrix[N, K_sigma] X_sigma;  // population-level design matrix
  int<lower=1> Kc_sigma;  // number of population-level effects after centering
  // data for group-level effects of ID 1
  int<lower=1> N_1;  // number of grouping levels
  int<lower=1> M_1;  // number of coefficients per level
  array[N] int<lower=1> J_1;  // grouping indicator per observation
  // group-level predictor values
  vector[N] Z_1_1;
  vector[N] Z_1_2;
  int prior_only;  // should the likelihood be ignored?
}
transformed data {
  matrix[N, Kc] Xc;  // centered version of X without an intercept
  vector[Kc] means_X;  // column means of X before centering
  matrix[N, Kc_sigma] Xc_sigma;  // centered version of X_sigma without an intercept
  vector[Kc_sigma] means_X_sigma;  // column means of X_sigma before centering
  vector[M_1] intercept_map_s2z_1;
  for (i in 2:K) {
    means_X[i - 1] = mean(X[, i]);
    Xc[, i - 1] = X[, i] - means_X[i - 1];
  }
  for (i in 2:K_sigma) {
    means_X_sigma[i - 1] = mean(X_sigma[, i]);
    Xc_sigma[, i - 1] = X_sigma[, i] - means_X_sigma[i - 1];
  }
  intercept_map_s2z_1 = zeros_vector(M_1);
  intercept_map_s2z_1[1] = 1.0;
  intercept_map_s2z_1[2] = means_X[1];
}
parameters {
  vector[2] theta_s2z;  // finite-population coefficients for physical S2Z effects
  vector[Kc_sigma] b_sigma;  // regression coefficients
  real Intercept_sigma;  // temporary intercept for centered predictors
  vector<lower=0>[M_1] sd_1;  // group-level standard deviations
  vector[M_1 * (N_1 - 1)] z_s2z_1;  // standardized orthonormal independent S2Z coordinates
  real<lower=0> udf_b_s2z_1;  // mixing variable for population coefficient 1
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
  // prior contributions to the log posterior
  real lprior = 0;
  real pupil_counter_tick = pupil_record_gradient(theta_s2z[1]);
  r_s2z_1_1 = sum_to_zero_constrain_brms(sd_1[1] * segment(z_s2z_1, (1 - 1) * (N_1 - 1) + 1, N_1 - 1));
  r_s2z_1_2 = sum_to_zero_constrain_brms(sd_1[2] * segment(z_s2z_1, (2 - 1) * (N_1 - 1) + 1, N_1 - 1));
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
  lprior += student_t_lpdf(Intercept_sigma | 3, 0, 2.5);
  lprior += student_t_lpdf(sd_1 | 3, 0, 2026.1)
    - 2 * student_t_lccdf(0 | 3, 0, 2026.1);
  lprior += inv_chi_square_lpdf(udf_b_s2z_1 | 3);
  lprior += normal_lpdf(qhat_s2z_1[1] | 5651.8999999999996, 2026.0999999999999 * sqrt(3 * udf_b_s2z_1));
  lprior += -0.5 * group_quad_s2z_1
    - 0.5 * sum(log(D_diag_s2z_1))
    - 0.5 * log1p(rank1_info_s2z_1) - 0.5 * N_1 * M_1 * log(2 * pi()) + 0.5 * M_1 * log(2 * pi()) + 0.5 * M_1 * log(1.0 * N_1);
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
    sigma += Intercept_sigma + Xc_sigma * b_sigma;
    for (n in 1:N) {
      // add more terms to the linear predictor
      mu[n] += r_s2z_1_1[J_1[n]] * Z_1_1[n] + r_s2z_1_2[J_1[n]] * Z_1_2[n];
    }
    sigma = exp(sigma);
    target += normal_lpdf(Y | mu, sigma);
  }
  // priors including constants
  target += lprior;
}
generated quantities {
  // actual population-level intercept
  real b_sigma_Intercept = Intercept_sigma - dot_product(means_X_sigma, b_sigma);
  vector[M_1] mean_r_s2z_1;
  vector[2] q_recovered_s2z_1;
  real Intercept;
  vector[Kc] b;
  real b_Intercept;
  vector[N_1] r_1_1;
  vector[N_1] r_1_2;
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
  real pupil_gradients = pupil_gradient_count();
}

