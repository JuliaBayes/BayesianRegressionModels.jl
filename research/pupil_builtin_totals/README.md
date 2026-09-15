# Pupil with a numeric residual-scale predictor: built-in BRM totals

This is the automatic-BRM version of the [post-3 pupil model](https://discourse.mc-stan.org/t/help-testing-brms-pr-for-sum-to-zero-and-partial-centering/41542/3):

```r
bf(p_size ~ load + (load || subj), sigma ~ subj)
```

`subj` is numeric (701–720), so its scale formula regresses log residual SD on numeric subject ID. The separate [`pupil_scale_totals`](../pupil_scale_totals/README.md) study fits hierarchical residual SDs. Both simplify the mean random effects to be independent. Every method within a comparison targets that same posterior.

## Model, quantities and completed fits

There are 2,228 observations and 20 subjects in original row order. Population load and numeric-ID predictors are observation-weight centered; random slopes use raw load. `common.jl` contains the actual BRM builder. BRM integrates the population mean intercept and slope, with one Gaussian precision mixture for the Student-t intercept. Both residual-model coefficients remain explicit. The retained flat slope prior requires BRM correction `f38bea8`.

The 46 scientific quantities are the two population mean coefficients, two group SDs, two residual-model coefficients, and 20 total intercepts and slopes. Population coefficients are recovered conditionally. Deviations, standardized coordinates and mixture variables do not determine the minimum ESS.

`results/comparison.tsv` and `table.md` contain 15 completed arms: nine unchanged ordinary/S2Z brms controls plus six new automatic-total WHMC fits (CP, NCP, both post-hoc losses, both online losses). Every arm has one chain and 2,000 retained draws. Both efficiency columns are relative to ordinary brms NCP + native Stan. Total costs include required pilots.

## Inspect without fitting

- `reference/automatic_totals.stan` and `.json`: exact exported target and data; `export-audit.json` records zero density and gradient differences.
- `results/per_quantity_mcse.tsv`: all scientific means, ESS and MCSE.
- `results/recovery_seed_sensitivity.tsv`: ten conditional-recovery seeds.
- `results/total_pairs*.tsv`: selected saved draws and CP/NCP/ACP transforms, including the source-frame audit.
- `results/fits/manifest.json`: both archive hashes and all member hashes, independently read back; 35,946,225 compressed bytes.
- `results/receipts/manifest.json`: exact process logs and environment.
- [`../pupil_total_effects/`](../pupil_total_effects/): original native/S2Z targets, counter implementation and reused raw fits.

The new-fit archive retains physical/source positions, gradients, checkpoints, selected controls and per-arm costs. The summary archive contains the 46 scientific arrays recomputed from the nine brms fits; `reused_brms_fits.tsv` maps these to the old fit paths. Nothing needs to be sampled to inspect the recorded comparison.

## Reproduce

Use Julia 1.10.11 and the Project/Manifest in `results/receipts`, replacing local path dependencies with recorded revisions. New fits used BRM's built-in planner (`54cbe3f`, landed as `5f30e53`) plus `f38bea8`, and repaired WHMC `7aed40b18bd4cdabb75330f285d5c9b355575ab9`. Data: bcogsci `d90fc01e6f6fcdced7ee64c9d2ed607d212ec77c`; brms `73cf607889879cb2a55f50b88d8141d76ff43279`; native controls: CmdStan 2.39.0, CmdStanR 0.9.0.

From the repository root, let `study_env` name that environment and `study_scratch` contain the archived run folders:

```sh
julia --project="$study_env" research/pupil_builtin_totals/audit.jl "$study_scratch/pupil3-builtin-audit-v2"
julia --project="$study_env" research/pupil_builtin_totals/total_whmc.jl "$study_scratch/pupil3-builtin-totals-v2" "$study_scratch/pupil3-builtin-audit-v2"
julia --project="$study_env" research/pupil_builtin_totals/reuse_brms.jl "$study_scratch" "$study_scratch/pupil3-builtin-brms-summary-v1"
python3 research/pupil_builtin_totals/assemble.py "$study_scratch" research/pupil_builtin_totals/results
julia --project="$study_env" research/pupil_builtin_totals/validate_saved.jl "$study_scratch" research/pupil_builtin_totals/results
julia --project="$study_env" research/pupil_builtin_totals/prepare_total_pairs.jl "$study_scratch/pupil3-builtin-totals-v2" research/pupil_builtin_totals/results
julia --project="$study_env" research/pupil_builtin_totals/plot.jl research/pupil_builtin_totals/results
```

Use fresh directories to refit. The audit passed 25 checks against the independent analytic target and analytic centering-gradient chain rule. The first attempt stopped at a numerical Gamma-domain proposal during post-hoc Pathfinder initialization. The recognized-domain rejection handler was corrected and the reported six-arm matrix was run afresh. Failed-attempt logs remain debugging evidence; their calls are not silently mixed into completed-workflow costs.

All six new arms have zero sampling divergences. The reused S2Z initialization differs from the ordinary/total initialization, as disclosed in the docs. These are single-chain exploratory results, not a replicated ranking or controlled wall-time benchmark. The docs build samples nothing.
