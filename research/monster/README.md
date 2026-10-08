# The Monster PBPK model in BRM

`reproduce.jl` declares the Monster model (Gelman, Bois and Jiang 1996) with
`@brm` and StanBlocks. It is an exact reproduction of one program from
[`nsiccha/monster`](https://github.com/nsiccha/monster) at commit
`a362738efd9c83525a2edb6db3e3646fba1a293f`:

| Source | Copy here |
| --- | --- |
| `stan/unconstrained_monster.stan` | `source/unconstrained_monster.stan`, with only the syntax that current stanc rejects updated (see its header) |
| `cfg/nu=4/parallel_incremental_data.json` | `source/parallel_incremental_data.json`, verbatim |

`compare_to_source.jl` compiles both programs with BridgeStan and checks, at
25 random unconstrained points, that their normalized log densities (with
Jacobians) differ by one constant and that their gradients agree under the
affine map between the two parameterizations. Result on BridgeStan 2.9.0 /
Stan 2.39 (gordito, 2026-10-08):

- constant offset `-27.791920602356214`, equal to the analytic Jacobian
  constant `sum(log(log(eM_eS))) - 15 log(2)`; spread across points `6.4e-12`;
- maximum relative gradient error `2.1e-12`;
- both programs have 122 unconstrained coordinates and cost about 60 ms per
  gradient evaluation.

## Correspondence

| `unconstrained_monster.stan` | BRM |
| --- | --- |
| `unit_log_population_eM[j] ~ normal(0, 1)`, `population_eM = eM_eM .* eM_eS .^ u` | `effect(log_X, Intercept) ~ Normal(log(eM_eM), log(eM_eS))` |
| `log_population_eS = exp(log(log(eS_mu)) + u)` with `scaled_inv_chi_square(tau^2 \| nu, log(eS_mu)) + log(tau^2)` | `sd(:, X) ~ ScaledInvChiScale(nu, log(eS_mu))`, the induced density of `tau` |
| `unit_log_person_params ~ normal(0, 1)`, `person_params = population_eM .* population_eS .^ u` | `log_X ~ 1 + (1 \| X \| subject)` (non-centered) |
| normalization of `person_params[2:5]` and `[6:7]` | `softmax` in the `@plate` cell |
| `noise` with `target += -log(noise)` | `log_sigma_* ~ Flat()` |
| `weight * lognormal_lpdf(...)` with 0/1 weights | observed values plus an index of the sampled times |
| `simulate_person` with `no_sub_steps = 128` (Strang splitting) | `monster_experiment` with `n_substeps = 128` |

The population normalization of `population_eM[2:5]` and `[6:7]` in the source
cancels in the person-level normalization, so it does not enter the density.

## Not reproduced

- The BDF branch of `simulate_person` (`no_sub_steps <= 0`). The likelihood
  never uses it. The source's generated quantities call it with
  `no_sim_sub_steps = -12`, that is `ode_bdf_tol` with relative tolerance
  `1e-12`, to draw `predicted_states`. BRM's `*_gen` predictive draws reuse the
  128-step Strang solver instead. This changes generated quantities only.
- `flexible_monster.stan` with `enforce_constraints = 1`, the variant used for
  the fits stored under `cfg/` and reported in the source README. It replaces
  the soft identification of the flow and lean-volume fractions by a
  sum-to-zero constraint on their standardized deviations, which changes their
  prior. `unconstrained_monster.stan` ignores the `enforce_constraints` and
  `include_jacobian` keys of the data file and corresponds to
  `flexible_monster.stan` with `enforce_constraints = 0, include_jacobian = 1`.

## Run

```sh
julia --project=<env with BayesianRegressionModels, StanBlocks, BridgeStan, JSON> \
    research/monster/compare_to_source.jl
```

`write_monster_stan_files(dir)` writes the generated `monster.stan` and its
Stan JSON data `monster.data.json`; the documentation build publishes both.
