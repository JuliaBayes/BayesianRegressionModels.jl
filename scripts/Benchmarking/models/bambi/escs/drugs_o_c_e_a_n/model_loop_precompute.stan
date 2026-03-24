// Variant: precompute mu in a loop, then pass to vectorized normal_lpdf.
// Separates the dot-product loop from the likelihood evaluation.
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
    vector[N] mu= Intercept + Xc * b;
    target += normal_lpdf(Y | mu, sigma);
  }
  target += lprior;
}
generated quantities {
  real b_Intercept = Intercept - dot_product(means_X, b);
}
