# Continuous-time state-space models in `@brm`

Sources behind the docs page `docs/src/state-space-models.md`. The models are the
demonstration models of [ctsem](https://github.com/cdriveraus/ctsem), Charles Driver's R
package for hierarchical continuous-time dynamic modelling; all data are simulated.

| file | what |
| --- | --- |
| `ema_sampled.jl` | hierarchical EMA model, latent states **sampled** (innovations are parameters) |
| `ema_kernel_kalman.jl` | linear-Gaussian special case, states integrated out **exactly** by a Kalman filter in the `kernel(...)` cell |
| `ema_kernel_marginalized.jl` | the full nonlinear / binary-indicator EMA model, states integrated out by an **extended Kalman filter** in the cell |
| `ema_state_dependent.jl` | drift and diffusion depend on the **latent state**; first-order or Gauss–Hermite moment-matched predict; filter precision (`nsub`, `gh`) is data |
| `ema_state_dependent_fit.jl` | WarmupHMC fit + recovery on a simulated 100 × 30 panel |
| `ema_state_dependent_psis.jl` | is the filter precise enough? — the importance-sampling reliability workflow of Timonen, Siccha, Bales, Lähdesmäki & Vehtari (arXiv:2205.09059) applied to the filter |
| `ema_state_dependent_replicate.jl` | refit on independently simulated panels: sampling variability vs. systematic error |
| `ema_state_dependent_summaries.jl`, `figures.jl` | tables in `results/` and the figures of the docs page |
| `ema_brm.jl` | incremental build-up: how far the pure formula surface carries, and where the kernel cell takes over |
| `kalman_marginalization.jl`, `ema_ekf.jl` | single-series building blocks: a dimension-generic Kalman `@lpxf` and a single-series EKF |

Every model file gates itself on the full pipeline — `@brm` → StanBlocks → `stanc` →
BridgeStan, finite log-density and gradient:

```sh
julia --project=test test/setup_env.jl
julia --project=test research/ema_ctsem/ema_state_dependent.jl
```
