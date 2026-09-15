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
  int prior_only;  // should the likelihood be ignored?
}
transformed data {
  matrix[N, Kc] Xc;  // centered version of X without an intercept
  vector[Kc] means_X;  // column means of X before centering
  vector[2] H_s2z_1;
  for (i in 2:K) {
    means_X[i - 1] = mean(X[, i]);
    Xc[, i - 1] = X[, i] - means_X[i - 1];
  }
  H_s2z_1 = zeros_vector(2);
  H_s2z_1[1] = 1.0;
}
parameters {
  vector[1] theta_s2z_active;  // S2Z-active finite-population coefficients
  vector[1] fixed_s2z;  // S2Z-inactive regression coefficients
  real<lower=0> sigma;  // dispersion parameter
  vector<lower=0>[M_1] sd_1;  // group-level standard deviations
  vector[N_1 - 1] z_s2z_1;  // standardized orthonormal scalar S2Z coordinates
  real<lower=0> udf_b_s2z_1;  // mixing variable for population coefficient 1
}
transformed parameters {
  vector[2] theta_s2z;  // assembled finite-population coefficients
  // specialized scalar physical S2Z effects of ID 1
  vector[N_1] r_s2z_1_1;
  vector[2] prior_mean_s2z_1;
  vector<lower=0>[2] prior_prec_s2z_1;
  real<lower=0> D_s2z_1;
  real<lower=0> sqrt_D_s2z_1;
  real mhat_s2z_1;
  vector[2] qhat_s2z_1;
  real<lower=0> group_quad_s2z_1;
  // prior contributions to the log posterior
  real lprior = 0;
  theta_s2z[1] = theta_s2z_active[1];
  theta_s2z[2] = fixed_s2z[1];
  r_s2z_1_1 = sum_to_zero_constrain_brms(sd_1[1] * z_s2z_1);
  prior_mean_s2z_1[1] = 2.7999999999999998;
  prior_prec_s2z_1[1] = inv_square(2.5 * sqrt(3 * udf_b_s2z_1));
  prior_mean_s2z_1[2] = 0;
  prior_prec_s2z_1[2] = 0.0;
  {
    real tau_sq_s2z = square(sd_1[1]);
    real prior_info_s2z = dot_product(prior_prec_s2z_1, square(H_s2z_1));
    real prior_score_s2z = dot_product(H_s2z_1, prior_prec_s2z_1 .* (theta_s2z - prior_mean_s2z_1));
    D_s2z_1 = tau_sq_s2z * prior_info_s2z + N_1;
    mhat_s2z_1 = tau_sq_s2z * prior_score_s2z / D_s2z_1;
  }
  sqrt_D_s2z_1 = sqrt(D_s2z_1);
  qhat_s2z_1 = theta_s2z - H_s2z_1 * mhat_s2z_1;
  {
    vector[N_1] white_s2z = (r_s2z_1_1 + mhat_s2z_1) / sd_1[1];
    group_quad_s2z_1 = dot_self(white_s2z);
  }
  lprior += student_t_lpdf(sigma | 3, 0, 2.5)
    - 1 * student_t_lccdf(0 | 3, 0, 2.5);
  lprior += student_t_lpdf(sd_1 | 3, 0, 2.5)
    - 1 * student_t_lccdf(0 | 3, 0, 2.5);
  lprior += inv_chi_square_lpdf(udf_b_s2z_1 | 3);
  lprior += normal_lpdf(qhat_s2z_1[1] | 2.7999999999999998, 2.5 * sqrt(3 * udf_b_s2z_1));
  lprior += -0.5 * group_quad_s2z_1
    - 0.5 * log(D_s2z_1) - 0.5 * N_1 * log(2 * pi()) + 0.5 * log(2 * pi()) + 0.5 * log(1.0 * N_1);
}
model {
  // likelihood including constants
  if (!prior_only) {
    // initialize linear predictor term
    vector[N] mu = rep_vector(0.0, N);
    mu += theta_s2z[1];
    for (n in 1:N) {
      // add more terms to the linear predictor
      mu[n] += r_s2z_1_1[J_1[n]] * Z_1_1[n];
    }
    target += normal_id_glm_lpdf(Y | Xc, mu, tail(theta_s2z, 1), sigma);
  }
  // priors including constants
  target += lprior;
}
generated quantities {
  real mean_r_s2z_1;
  vector[2] q_recovered_s2z_1;
  real Intercept;
  vector[Kc] b;
  real b_Intercept;
  vector[N_1] r_1_1;
  mean_r_s2z_1 = mhat_s2z_1 + sd_1[1] * std_normal_rng() / sqrt_D_s2z_1;
  q_recovered_s2z_1 = theta_s2z - H_s2z_1 * mean_r_s2z_1;
  r_1_1 = r_s2z_1_1 + mean_r_s2z_1;
  Intercept = q_recovered_s2z_1[1];
  b = tail(q_recovered_s2z_1, Kc);
  b_Intercept = Intercept - dot_product(means_X, b);
}

