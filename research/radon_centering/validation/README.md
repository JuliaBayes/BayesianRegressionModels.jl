# Independent saved-data acceptance

This read-only audit uses the recorded fits, immutable reference data and
derived diagnostic tables. It does not load BRM, use WarmupHMC's transport
helpers, run a sampler, recompile a target, or regenerate figures.

From the repository root, with Julia 1.10 and `unzip` on PATH:

```sh
julia --startup-file=no --project=test test/setup_env.jl
julia --startup-file=no --project=test \
  research/radon_centering/validation/audit_saved.jl \
  /absolute/offline-directory /absolute/online-directory \
  /absolute/diagnostics-directory /absolute/cost-directory
```

The test environment supplies the pinned WarmupHMC checkpoint types. The
audit loads those types for deserialization and checks:

- all 772 scalar source-to-model maps, plus unchanged hyperparameters;
- minimum/nearest-0.5/maximum coordinate selection and tie-breaking;
- all 12,573 PPC rows against original indices, observations, counties and floor codes;
- every pair-plot row against its named saved fit and displayed geometry;
- every coordinate–gradient row against the Gaussian likelihood/prior
  derivative computed independently from county sufficient statistics;
- model-frame R-hat and minimum bulk/tail ESS, both exact gradient counters,
  and the corresponding ESS-per-gradient ratios.

The full-data acceptance covers 300,000 pair rows and 18,000
coordinate–gradient rows. Returned model positions are checked against the
source-coordinate final checkpoint. The audit verifies counters against
the saved returned records; returned-record versus final-checkpoint agreement
is separately enforced by `report_costs.jl`.
