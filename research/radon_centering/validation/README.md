# Independent saved-data acceptance

This read-only audit uses the recorded fits, immutable reference data and
derived diagnostic tables. It does not load BRM, use WarmupHMC's transport
helpers, run a sampler, recompile a target, or regenerate figures.

From the repository root, with Julia 1.10 and `unzip` on PATH:

```sh
julia --startup-file=no --project=research/radon_centering/validation \
  -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=research/radon_centering/validation \
  research/radon_centering/validation/audit_saved.jl \
  /absolute/offline-directory /absolute/online-directory \
  /absolute/diagnostics-directory /absolute/cost-directory
```

The validation project contains only registered packages. The audit checks:

- all 772 scalar source-to-model maps, plus unchanged hyperparameters;
- every pair-plot row against its named saved fit and displayed geometry;
- every coordinate–gradient row against the Gaussian likelihood/prior
  derivative computed independently from county sufficient statistics;
- model-frame R-hat and minimum bulk/tail ESS, both exact gradient counters,
  and the corresponding ESS-per-gradient ratios.

The expected full-data acceptance is 636,811 checks, covering 300,000 pair
rows and 18,000 coordinate–gradient rows. The audit verifies counters against
the saved returned records; returned-record versus final-checkpoint agreement
is separately enforced by `report_costs.jl`.
