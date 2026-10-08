# The Monster PBPK model

The Monster model is the hierarchical physiologically based pharmacokinetic
(PBPK) model of tetrachloroethylene uptake and elimination of
[Gelman, Bois and Jiang (1996)](http://www.stat.columbia.edu/~gelman/bayescomputation/GelmanBoisJIang1996.pdf),
named after A. C. Monster, first author of the
[exposure study](https://link.springer.com/content/pdf/10.1007/BF00377784.pdf)
behind its data. Six people each inhaled two concentrations of the solvent
for four hours. Their venous blood and exhaled air were then sampled for up to
a week.

This page declares the model with `@brm` and StanBlocks. The declaration is an
exact reproduction of one program from the Stan implementation in
[`nsiccha/monster`](https://github.com/nsiccha/monster), so it can serve as a
single, clearly specified model/data pair.

## What exactly is reproduced

| Source at commit `a362738` | Role |
| --- | --- |
| [`stan/unconstrained_monster.stan`](https://github.com/nsiccha/monster/blob/a362738efd9c83525a2edb6db3e3646fba1a293f/stan/unconstrained_monster.stan) | the Stan program |
| [`cfg/nu=4/parallel_incremental_data.json`](https://github.com/nsiccha/monster/blob/a362738efd9c83525a2edb6db3e3646fba1a293f/cfg/nu%3D4/parallel_incremental_data.json) | its data: all six people, both experiments, population GSD prior with `nu = 4`, 128 Strang substeps |

The two programs were compared with BridgeStan at 25 random unconstrained
points. Their normalized log densities, Jacobians included, differ by a single
constant (spread `6e-12`): the Jacobian of the affine map between the two
parameterizations. Their gradients agree to a relative error of `2e-12`. Both
have 122 unconstrained coordinates, and a gradient evaluation costs the same
(about 60 ms on the test machine). The source program needs small syntax
updates for current stanc; the
[reproduction directory](https://github.com/nsiccha/BayesianRegressionModels.jl/tree/ns/devibe/research/monster)
holds that updated copy, the verbatim data and the comparison script.

Downloads, generated from the declaration below while these docs were built:

- [`monster.stan`](downloads/monster.stan) — the generated Stan program;
- [`monster.data.json`](downloads/monster.data.json) — its data in Stan's JSON
  format.

The data use Stan tuples, which need Stan 2.33 or later; the pair was checked
with BridgeStan 2.9 (Stan 2.39).

## The model

**Physiology.** Four tissue compartments hold the solvent: well-perfused
tissue (`wp`), poorly perfused tissue (`pp`), fat (`f`) and liver (`l`).
Blood flow exchanges it between tissues and lungs. During the four-hour
exposure, inhalation adds a source; Michaelis–Menten metabolism in the liver
removes it. Each person has fifteen positive quantities:

| Quantity | Meaning |
| --- | --- |
| `VPR` | ventilation–perfusion ratio |
| `Fwp`, `Fpp`, `Ff`, `Fl` | fractions of blood flow to the four compartments (normalized to sum to one) |
| `Vwp`, `Vpp`, `Vl` | lean-body-mass fractions of the non-fat compartments (`Vwp + Vpp + Vl = 0.837`) |
| `Pba` | blood/air partition coefficient |
| `Pwp`, `Ppp`, `Pf`, `Pl` | tissue/blood partition coefficients |
| `VMI`, `KMI` | maximal metabolic rate (per kg^0.7 of lean body mass) and Michaelis constant |

Lean body mass, fat fraction and pulmonary flow were measured for each person
and enter as data. Fat volume is derived from them.

**Hierarchy.** On the log scale, each person's quantity is the population
geometric mean plus a deviation whose standard deviation is the log of the
population geometric standard deviation (GSD). The flow fractions and the
`Vwp`/`Vpp` split are hierarchical before normalization, so their overall
scale is identified only by the prior (see
[other variants](#other-variants-in-the-source-repository)). The population geometric means have lognormal priors. Each log GSD `tau` has
`tau^2 ~ Scaled-Inv-χ²(nu, s^2)` with `nu = 4`. All prior locations and scales
come from the source data file.

**Measurements.** Venous concentrations and exhaled-air concentrations are
lognormal around the predictions, with one scale each. Both scales have the
improper prior `p(sigma) ∝ 1/sigma`. Venous blood was not sampled at 245 and
270 minutes; those exhaled-air values are used and no venous value is
invented.

**What `nu` means.** It is the degrees of freedom of the prior on the squared
log GSDs. Gelman, Bois and Jiang used `nu = 2`. The source repository reports
that with `nu = 2` divergences could not be removed, and that `nu ≥ 3` fits
cleanly. Its `cfg/nu=4` and `cfg/nu=8` data files differ only in this value
(and in a warm-up diagnostic that the model does not read); `nu = 4` is the one
closer to the original.

## Solving the dynamics without an ODE solver

The likelihood never calls one of Stan's ODE solvers. With `no_sub_steps = 128`
the source program, and this reproduction, use a Strang splitting on a fixed
grid of `240 / 128 = 1.875` minutes. Each step applies half an exact
Michaelis–Menten step to the liver (a closed form through Lambert's W), then an
exact step of the linear tissue transport (a matrix exponential, with the
exposure source during the first four hours), then the other half
Michaelis–Menten step. The grid continues through washout. Each observation
time between two grid points takes a geometric interpolation of the two
states, and the trajectory is re-anchored exactly at the end of exposure.

```@eval
Main.BRMDocsComparisons.evaluate_source_prelude(
    Main.BRMDocsComparisons.example_module(:monster),
    "research/monster/reproduce.jl";
    before=:monster_brmi,
)
nothing
```

The simulator is ordinary StanBlocks code, a line-by-line port of the
source's `simulate_person`:

```@eval
Main.BRMDocsComparisons.source_code_region(
    "research/monster/reproduce.jl";
    starting_at="StanBlocks.@deffun begin\n    monster_min_concentration",
    ending_before="# Prior of a population geometric standard deviation",
)
```

The source's alternative branch (`no_sub_steps <= 0`) solves the same system
with `ode_bdf_tol`, at relative tolerance `10^no_sub_steps`. The likelihood
does not use it. The source's generated quantities do: `no_sim_sub_steps = -12`
draws `predicted_states` from BDF solutions at relative tolerance `1e-12`, for
each draw. That branch is not reproduced here. BRM's posterior-predictive
draws (`venous_72_gen`, …) reuse the 128-step solver, which changes generated
quantities only.

## The GSD prior as a BRM prior family

BRM's random-effect scale `tau` is the log GSD itself. Its prior is therefore
the density of `tau` induced by `tau^2 ~ Scaled-Inv-χ²(nu, s^2)`. It is
registered as a custom family: a Distributions.jl type for the BRM side, and
its Stan translation for SBBRMI.

```@eval
Main.BRMDocsComparisons.source_code_region(
    "research/monster/reproduce.jl";
    starting_at="# Prior of a population geometric standard deviation",
    ending_before="# The six persons and two exposure experiments",
)
```

## The BRM declaration

Each of the fifteen quantities is a named linear predictor with its own
non-centered subject effect, so their priors are addressed by name. The
`@plate` cell normalizes the fractions, runs both exposure experiments for one
person, and observes the two measurement channels. `monster_experiment`
returns venous predictions followed by exhaled-air predictions, and the
`*_index` columns select the sampled times. The data are the source's final
data update; `monster_data` in the reproduction file lists them in the
source's units.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:monster),
    Main.BRMDocsComparisons.source_function(
        "research/monster/reproduce.jl", :monster_brmi,
    ),
    :monster_brmi;
    title="Monster PBPK model",
    require_stan=true,
)
```

## Other variants in the source repository

The repository holds several related programs and data files. Three facts
matter when comparing results:

- **`flexible_monster.stan` produced every fit stored under `cfg/`** and the
  results in its README. Its data files set `enforce_constraints = 1`. The four
  flow fractions and the two lean-volume fractions then use standardized
  deviations constrained to sum to zero instead of the soft identification
  above, which changes their prior. `unconstrained_monster.stan` ignores the
  `enforce_constraints` and `include_jacobian` keys; it is the
  `enforce_constraints = 0` case of `flexible_monster.stan`.
- **The top-level data files are one dataset.** For a given `nu`,
  `parallel_incremental_data.json`, `serial_incremental_data.json` and
  `serial_regular_data.json` contain the same model data; only the fits,
  metrics and settings stored beside them differ by warm-up method. The
  exception is
  `cfg/nu=8/posterior_resampling_*`, which uses the 1996 posterior as its prior.
  The numbered files in the `parallel_incremental/` subdirectories are earlier
  warm-up stages: the prior alone, fewer observations or fewer substeps.
- **Changes relative to the 1996 model**, as the source README lists them:
  the hard bounds on the population means are removed; the
  `enforce_constraints = 1` variant also removes the over-parameterization of
  the softly enforced sum constraints; and the GSD prior is tightened from
  `nu = 2`, the only change needed to fit without divergences. The exposure
  concentrations use the parts-per-million conversion of the
  [MCSim example](https://www.gnu.org/software/mcsim/mcsim.html#perc_002emodel)
  rather than that of the papers.

## Reproduce the comparison

```sh
julia --project=<environment with BayesianRegressionModels, StanBlocks, BridgeStan, JSON> \
    research/monster/compare_to_source.jl
```
