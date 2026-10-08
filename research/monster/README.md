# The Monster PBPK model in BRM

`reproduce.jl` declares the Monster model (Gelman, Bois and Jiang 1996) with
`@brm` and StanBlocks. It reproduces one program from
[`nsiccha/monster`](https://github.com/nsiccha/monster) at commit
`a362738efd9c83525a2edb6db3e3646fba1a293f` exactly, with two bugs of that
program corrected (decision `09w6xih`):

| Source | Copy here |
| --- | --- |
| `stan/unconstrained_monster.stan` | `source/unconstrained_monster.stan`, with only the syntax that current stanc rejects updated (see its header) |
| `cfg/nu=4/parallel_incremental_data.json` | `source/parallel_incremental_data.json`, verbatim |
| the same program with the two bugs fixed | `source/unconstrained_monster_corrected.stan`, a three-line diff from the copy above |

## The two corrections

- The non-fat organ volumes sum to 0.873 of lean body mass, not 0.837. Sources:
  Bois et al. 1996, Arch Toxicol 70:350 ("have to sum to 0.873 (the fraction of
  lean body weight not including bones)"); Gelman, Bois and Jiang 1996, JASA
  eq. (1); the prior means 0.28 + 0.56 + 0.033.
- Alveolar air is `(VPR*C_inh + C_ven)/(VPR + Pba)`, the lung mass balance of
  MCSim's `perc.model`. The source omits `VPR` in its output line only; it
  matters at the 240-minute end-of-exposure sample.

At 200 draws of the source's stored `cfg/nu=4` posterior (`flexible_monster.stan`),
the corrections change the log density by +2.2 on average (+2.9 for the
alveolar fix, −0.8 for 0.873). Predictions move by at most 1.1 noise SD, at the
240-minute exhaled-air samples.

`compare_to_source.jl` compiles the BRM model and the corrected source program
with BridgeStan. At 25 random unconstrained points it checks that their
normalized log densities (with Jacobians) differ by one constant, and that their
gradients agree under the affine map between the two parameterizations. Result on
BridgeStan 2.9.0 / Stan 2.39 (gordito, 2026-10-08):

- constant offset `-27.791920602356015`, equal to the analytic Jacobian
  constant `sum(log(log(eM_eS))) - 15 log(2)`; spread across points `9.4e-12`;
- maximum relative gradient error `9.0e-12`;
- negative control, `program="unconstrained_monster.stan"`, the uncorrected
  copy: offset spread `6.5`, gradient error `0.50`;
- both programs have 122 unconstrained coordinates and cost about 40–60 ms per
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
| `.837 - person_params[person,8]` | `0.873 - Vl` (corrected) |
| `(C_inh + C_ven)/(VPR + Pba)` | `(VPR*C_inh + C_ven)/(VPR + Pba)` (corrected) |

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
