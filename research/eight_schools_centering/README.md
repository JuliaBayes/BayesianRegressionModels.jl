# PosteriorDB eight schools: manual and adaptive centering

This study reproduces PosteriorDB posterior
`eight_schools-eight_schools_noncentered` at
`5545a1dd07ae297c36edecbcd82aa49097b4c385`. The immutable centered and
noncentered Stan programs, posterior metadata and data are in `reference/`.
The BRM formula uses `mu ~ Normal(0,5)` and positive `tau ~ Cauchy(0,5)`.

## Run

From the repository root, after the normal `test/setup_env.jl` bootstrap:

```sh
BRM_EIGHT_SCHOOLS_OUTPUT=/absolute/eight-audit \
  julia --startup-file=no --project=test research/eight_schools_centering/audit_source.jl

BRM_EIGHT_SCHOOLS_RUNTIME=1 BRM_EIGHT_SCHOOLS_OUTPUT=/absolute/eight-fit \
  julia --startup-file=no --project=test research/eight_schools_centering/reproduce.jl

julia --startup-file=no --project=test \
  research/eight_schools_centering/prepare_gradient_diagnostics.jl \
  /absolute/eight-fit /absolute/eight-diagnostics

julia --startup-file=no --project=test \
  research/eight_schools_centering/validate_results.jl \
  /absolute/eight-fit /absolute/eight-diagnostics

julia --startup-file=no --project=test research/eight_schools_centering/plots/bootstrap.jl
julia --startup-file=no --project=research/eight_schools_centering/plots \
  research/eight_schools_centering/plot_results.jl \
  /absolute/eight-fit /absolute/eight-diagnostics /absolute/eight-figures
```

Use fresh output directories. The driver runs complete 10,000-draw priming
fits for the plain, fixed-centering and online paths, then three complete
repetitions of four configurations. Each fit uses `Xoshiro(1)`, normal
WarmupHMC initialization/adaptation and checkpoint recording. The manual
centered arm fixes every coefficient at one; offline selection uses
`0:0.01:1`, and online adaptation uses `0:0.1:1`.

Every timed call records wall time, Julia compilation time and GC time.
The reported benchmark uses warmed repetitions, with the second repetition
reversing the fit order. Exact NUTS total and retained-sampling gradient
counters are checked against final checkpoints. The offline workflow charges
for both pilot and refit.

## Coordinates and validation

WarmupHMC `deeea1d` returns model coordinates for every arm. Source-coordinate
records come from final checkpoints. The driver verifies their transport to
the returned model draws; `extract_results.jl` independently verifies the
scalar map and can extract results from complete archived return values and
checkpoints into a fresh directory without sampling.

`noncentered.jls`, `centered.jls`, `partial.jls` and `online.jls` hold model
coordinates. Corresponding `*_source.jls` files hold sampler coordinates.
The `centered_target.jls` and `partial_target.jls` records also hold model
coordinates. School effects are always `theta = mu + tau*z` in model coordinates.

The source audit checks all ten gradient components against PosteriorDB and
checks the manually centered density, Jacobian and gradients. Saved-data
validation checks checkpoint/return coordinates, all pair rows, all displayed
gradients against an independent Gaussian derivative, the exact offline
criterion, repeated timing counters and recomputed model-frame diagnostics.

The published sampling/extraction directories and exact producer/extraction
hashes are in `results/provenance.toml`. Small evidence tables are committed
under `results/`; full draws and checkpoints stay in scratch. The figure
manifest records image hashes and rendering dependencies. All plots use native
AlgebraOfVega with CairoMakie. School summaries and predictive checks use
individual intervals in the original school order.

The scatter plots use the NCP pilot transformed to fully centered coordinates
as their first-column visual reference. The NCP comparison uses those same
physical draws; selected and online columns use their separate saved fits.
The manually centered fit remains in the posterior summaries and cost table.
