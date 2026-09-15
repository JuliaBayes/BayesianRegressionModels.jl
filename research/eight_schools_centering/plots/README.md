# Eight-schools native AoV rendering

This environment loads BRM plus its optional AlgebraOfVega extension. Resolve it
with the reusable entrypoint, which pins every unregistered dependency at an
exact commit (the six `test` pins, cross-checked against `test/setup_env.jl`,
plus the four plotting-stack pins owned here) and develops them in one call:

```sh
julia --startup-file=no --project=test \
  research/eight_schools_centering/plots/bootstrap.jl
```

The committed project deliberately carries no Manifest; research rendering is
not a docs-CI dependency. Re-running the bootstrap reuses the exact cached
checkouts, so it is safe to run whenever the environment's state is in doubt.

```sh
julia --startup-file=no --project=research/eight_schools_centering/plots \
  research/eight_schools_centering/plot_results.jl \
  /absolute/full-fit-directory /absolute/diagnostics-directory \
  /absolute/figure-directory
```

Rendering consumes saved tables. Each paired geometry figure has one column
per configuration and one row per school. Pair panels use all 10,000 draws;
gradient panels use 1,000 evenly spaced draws per facet. Posterior and
predictive summaries use individual 50% and 90% intervals. Observed estimates
are overlaid in the original school order. Every school has a distinct colour
in objective plots. The renderer records figure hashes and dependency identities.
