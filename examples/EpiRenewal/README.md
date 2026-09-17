# EpiRenewal — an example package on top of BayesianRegressionModels.jl

BayesianRegressionModels knows nothing about epidemics. This package supplies the
mechanics of a renewal-equation model as **operators for `@brm` top-level
assignments**, so a model reads as formula lines for the statistics and one line
per mechanical step:

```julia
using BayesianRegressionModels, EpiRenewal, Distributions

model = @brm data begin
    log_I0  ~ Normal(log(50.0), 0.5)
    cluster ~ Normal(0.0, 0.1; lower=0.0)
    log_R   ~ 1 + rw(time)                                    # log R(t): a random walk
    effect(log_R, Intercept) ~ Normal(log(1.3), 0.1)
    sd(:, rw(time)) ~ Normal(0.0, 0.05)
    past    = seeded_history(log_I0, exp(log_R), gen_pmf, 14) # infections on days -13 … 0
    I       = renewal(gen_pmf, exp(log_R), past)              # I[t] = R[t] * Σ_s gen[s] I[t-s]
    Y       = delay(delay_pmf, I, past)                       # Y[t] = Σ_d pmf[d] I[t-d]
    cases   ~ nb_cases(Y, cluster, observed)                  # masked negative binomial
end
sb = SBBRMI(model; mod=@__MODULE__)                           # the operators are Stan functions
```

It is the package form of the documentation page
[Epidemic renewal models](https://nsiccha.github.io/BayesianRegressionModels.jl/dev/renewal),
whose hand-written Stan functions it replaces. The package's test checks that the
models above have **the same log density and gradient** as the page's, so the
page's fits and figures are this package's fits and figures.

## What it owns

| Piece | Spelling |
|---|---|
| infections before day 1, growing at the rate the first `R` implies | `seeded_history(log_I0, R, gen_pmf, H)` |
| renewal recursion, one population | `renewal(gen_pmf, R, past)` |
| renewal recursion, coupled patches | `renewal(gen_pmf, R, past, mixing)` |
| gravity mixing matrix | `gravity(pop, dist_flat, gamma)` |
| one value per patch from a per-row predictor | `per_patch(x, pop)` |
| reporting delay | `delay(pmf, I, past)` |
| negative-binomial counts with a row mask (a forecast is a mask) | `cases ~ nb_cases(Y, cluster, observed)` |
| linelist delays, doubly interval-censored and right-truncated | `delay ~ censored_delay(mu, sigma, window)` |
| daily masses of a continuous delay, in Julia | `censored_pmf(dist; D, drop_zero)` |
| sort and validate a frame | `epi_frame(rows; time, by)` |

## The kernel comes first, as a vector or as a function

Every operator takes its kernel as the **first** argument, so it can also be a
`do` block — and a function body may read **sampled parameters**, which a data
vector cannot. This estimates the reporting delay inside the renewal model:

```julia
mu_delay    ~ Normal(1.5, 0.2)
sigma_delay ~ Normal(0.5, 0.1; lower=0.0)
Y = delay(I, past, 15) do d                                   # d = 0 … 14
    censored_lognormal_mass(d, mu_delay, sigma_delay, 15)
end
```

`renewal(R, past, G) do s … end` does the same for the generation interval
(`s = 1 … G`).

## The row-order contract

The operators see plain vectors, so the frame must be laid out the way they read
it: **one row per (group, day), ordered by group and then by day, every group
covering the same equally spaced days.** With several groups a row vector is
read as a `T × P` matrix, patch `g` in column `g`. `epi_frame` sorts a table into
that order and refuses one that cannot satisfy it — a gap is an error, never
filled in silently:

```julia
rows = epi_frame((; day, patch, cases, observed, seed_mean); time=:day, by=:patch)
data = merge(rows, (; gen_pmf, delay_pmf, pop, dist_flat, C))   # shared fields go in after
```

## Scope

- Deterministic renewal only: infections are a function of `R` and the seeds.
  Latent (sampled) infections need a sequential sampling primitive
  (StanBlocks `@scan`) that the StanBlocks version pinned by this repository's
  test environment does not have.
- StanBlocks backend only. The operators are Stan functions; `TuringBRMI`
  refuses the assignments by name.
- The history before day 1 is the exponential seeding of the documentation page.
  A lag that reaches further back than `H` days counts as zero, so choose `H` at
  least as long as the longest kernel.

## Run the tests

From the repository root, with its test environment (no sampling):

```sh
julia --startup-file=no --project=test examples/EpiRenewal/test/runtests.jl
```
