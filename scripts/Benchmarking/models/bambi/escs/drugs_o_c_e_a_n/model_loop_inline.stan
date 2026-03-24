// Variant: vectorized normal_lpdf with matrix-vector product Xc * b.
// Unlike normal_id_glm_lpdf, the linear predictor is computed separately
// (not fused into the likelihood), but still uses Eigen's matrix-vector multiply.
data {
  int<lower=1> N;
  vector[N] Y;
  int<lower=1> K;
  matrix[N, K] X;
  int<lower=1> Kc;
  int prior_only;
}
transformed data {
  matrix[N, Kc] Xc;
  vector[Kc] means_X;
  for (i in 2:K) {
    means_X[i - 1] = mean(X[, i]);
    Xc[, i - 1] = X[, i] - means_X[i - 1];
  }
}
parameters {
  vector[Kc] b;
  real Intercept;
  real<lower=0> sigma;
}
transformed parameters {
  real lprior = 0;
  lprior += student_t_lpdf(Intercept | 3, 2.1, 2.5);
  lprior += student_t_lpdf(sigma | 3, 0, 2.5)
    - 1 * student_t_lccdf(0 | 3, 0, 2.5);
}
model {
  if (!prior_only) {
    target += normal_lpdf(Y | Intercept + Xc * b, sigma);
  }
  target += lprior;
}
generated quantities {
  real b_Intercept = Intercept - dot_product(means_X, b);
}
