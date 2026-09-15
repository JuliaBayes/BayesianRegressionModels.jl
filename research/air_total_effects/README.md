# AIR automatic-total follow-up

The follow-up preserves the population intercept and slope and the original
brms priors. The earlier deletion-of-population-effects smoke tests target
different posteriors and are not baselines for this comparison.

The original correlated regional intercept/slope model is outside the current
automatic-total planner. The supported comparisons are therefore explicitly
labelled simplifications: regional random intercepts alone, and independent
regional random intercepts and slopes. Within each simplification, ordinary
brms, S2Z and automatic BRM totals must target the same posterior.

`capture.R` captures both hierarchies for the three source grouping choices:
`cluster_region`, `cluster_log_region` and `super_region`. Begin the empirical
comparison with `cluster_region`, which has six groups with sizes 27, 3534,
1696, 418, 315 and 13. Keep the other captured models for the applicable-model
follow-up. The data have 6,003 original-order rows, pinned SHA-256
`8eed2b16c17fd1501d616dda0c983e57d44cc2f5552b21fbcad3c09e864507cd`.

The comparison protocol follows the post-4 pupil study: actual gradient costs,
both post-hoc and online losses, and a common scientific-QOI set including
population coefficients, group scales, residual scale, and regional total
coefficients. Parameterization-specific latent coordinates do not determine
the main efficiency minimum. Correlated models require a later exact extension;
silently replacing their covariance prior is not an exact reparameterization.
