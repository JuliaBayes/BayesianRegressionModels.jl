// generated with brms 2.23.1
functions {
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
  // data for group-level effects of ID 2
  int<lower=1> N_2;  // number of grouping levels
  int<lower=1> M_2;  // number of coefficients per level
  array[N] int<lower=1> J_2;  // grouping indicator per observation
  // group-level predictor values
  vector[N] Z_2_sigma_1;
  int prior_only;  // should the likelihood be ignored?
}
transformed data {
  matrix[N, Kc] Xc;  // centered version of X without an intercept
  vector[Kc] means_X;  // column means of X before centering
  for (i in 2:K) {
    means_X[i - 1] = mean(X[, i]);
    Xc[, i - 1] = X[, i] - means_X[i - 1];
  }
}
parameters {
  vector[Kc] b;  // regression coefficients
  real Intercept;  // temporary intercept for centered predictors
  real Intercept_sigma;  // temporary intercept for centered predictors
  vector<lower=0>[M_1] sd_1;  // group-level standard deviations
  array[M_1] vector[N_1] z_1;  // centered group-level coordinates
  vector<lower=0>[M_2] sd_2;  // group-level standard deviations
  array[M_2] vector[N_2] z_2;  // centered group-level coordinates
}
transformed parameters {
  vector[M_1] mean_center_re_1;  // matching population locations for group centering
  real log_jacobian_re_1;  // unrestricted centering log-Jacobian
  vector[N_1] r_1_1;  // actual group-level effects
  vector[N_1] r_1_2;  // actual group-level effects
  vector[M_2] mean_center_re_2;  // matching population locations for group centering
  real log_jacobian_re_2;  // unrestricted centering log-Jacobian
  vector[N_2] r_2_sigma_1;  // actual group-level effects
  // prior contributions to the log posterior
  real lprior = 0;
  mean_center_re_1 = zeros_vector(M_1);
  mean_center_re_1[1] = -5.6901591091737039e-14 * (b[1]);
  mean_center_re_1[2] = b[1];
  r_1_1 = z_1[1] - rep_vector(mean_center_re_1[1], N_1);
  r_1_2 = z_1[2] - rep_vector(mean_center_re_1[2], N_1);
  log_jacobian_re_1 = 0.0;
  mean_center_re_2 = zeros_vector(M_2);
  r_2_sigma_1 = z_2[1] - rep_vector(mean_center_re_2[1], N_2);
  log_jacobian_re_2 = 0.0;
  lprior += student_t_lpdf(Intercept | 3, 5651.9, 2026.1);
  lprior += student_t_lpdf(Intercept_sigma | 3, 0, 2.5);
  lprior += student_t_lpdf(sd_1 | 3, 0, 2026.1)
    - 2 * student_t_lccdf(0 | 3, 0, 2026.1);
  lprior += student_t_lpdf(sd_2 | 3, 0, 2026.1)
    - 1 * student_t_lccdf(0 | 3, 0, 2026.1);
}
model {
  // likelihood including constants
  if (!prior_only) {
    // initialize linear predictor term
    vector[N] mu = rep_vector(0.0, N);
    // initialize linear predictor term
    vector[N] sigma = rep_vector(0.0, N);
    mu += Intercept + Xc * b;
    sigma += Intercept_sigma;
    for (n in 1:N) {
      // add more terms to the linear predictor
      mu[n] += r_1_1[J_1[n]] * Z_1_1[n] + r_1_2[J_1[n]] * Z_1_2[n];
    }
    for (n in 1:N) {
      // add more terms to the linear predictor
      sigma[n] += r_2_sigma_1[J_2[n]] * Z_2_sigma_1[n];
    }
    sigma = exp(sigma);
    target += normal_lpdf(Y | mu, sigma);
  }
  // priors including constants
  target += lprior;
  target += normal_lpdf(r_1_1 | 0, sd_1[1]);
  target += normal_lpdf(r_1_2 | 0, sd_1[2]);
  target += log_jacobian_re_1;
  target += normal_lpdf(r_2_sigma_1 | 0, sd_2[1]);
  target += log_jacobian_re_2;
}
generated quantities {
  // actual population-level intercept
  real b_Intercept = Intercept - dot_product(means_X, b);
  // actual population-level intercept
  real b_sigma_Intercept = Intercept_sigma;
}

