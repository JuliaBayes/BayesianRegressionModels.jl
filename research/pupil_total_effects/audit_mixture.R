# Independent numerical validation only; the sampler does not use quadrature.
m <- 5651.9
s <- 2026.1
nu <- 3
zs <- c(-12, -4, -1, 0, 1, 4, 12)
rows <- do.call(rbind, lapply(zs, function(z) {
  mixture <- integrate(function(lambda) {
    dnorm(m+s*z, m, s/sqrt(lambda)) * dgamma(lambda, shape=nu/2, rate=nu/2)
  }, 0, Inf, rel.tol=1e-10, abs.tol=1e-14)$value
  student <- dt(z, nu)/s
  data.frame(standardized_value=z, mixture=mixture, student=student,
             relative_error=abs(mixture-student)/student)
}))
stopifnot(max(rows$relative_error) < 1e-8)
write.table(rows, "research/pupil_total_effects/reference/student_mixture_marginal_audit.tsv",
            sep="\t", row.names=FALSE, quote=FALSE)
print(rows)
