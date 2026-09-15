# AIR: automatic BRM totals and brms S2Z

The [source discussion](https://discourse.mc-stan.org/t/help-testing-brms-pr-for-sum-to-zero-and-partial-centering/41542) uses regional effects for log particulate concentration against a log satellite predictor. This study retains both population intercept and slope and matches the generated brms priors.

Two supported simplifications are fully measured, each with 14 arms:

```r
bf(log_pm25 ~ log_sat + (1 | region))
bf(log_pm25 ~ log_sat + (1 + log_sat || region))
```

The second deliberately removes random-effect correlations. The original correlated target is outside the current automatic planner. Earlier smoke tests deleting population effects target different posteriors and are not baselines here.

## Data and posterior

There are 6,003 original-order rows. The fitted grouping `cluster_region` has sizes 27, 3534, 1696, 418, 315, 13. Data revision: `2afb605f81cf0bdadc6476df865e0337aaacf183`; SHA-256: `8eed2b16c17fd1501d616dda0c983e57d44cc2f5552b21fbcad3c09e864507cd`. Captures for `cluster_log_region` and `super_region` are retained but were not fitted in this comparison.

The population intercept at centered log_sat has Student-t(3, 2.8, 2.5) prior; the population slope is flat. Group SDs and residual SD have half-Student-t(3, 0, 2.5) priors. Regional deviations are Gaussian. `model.jl` contains the BRM builders and an independent Gaussian-integration density.

The intercept-only comparison uses 10 scientific quantities: two population coefficients, group SD, residual SD, six total regional intercepts at raw predictor zero. The independent intercept/slope comparison uses 17: two population coefficients, two group SDs, residual SD, six total intercepts and six total slopes. Population coefficients are recovered conditionally for marginalized methods. Deviations and sampler-specific coordinates are excluded.

## Results and artifacts

Both `results/cluster-intercept` and `results/cluster-independent` contain one 14-row scientific-QOI table, an efficiency figure, total CP/NCP/ACP pairs, S2Z CP/NCP/auto pairs, per-quantity MCSE, saved-frame audits and ten-seed recovery sensitivity. Each arm has one chain with 2,000 retained draws. Both efficiency columns are relative to ordinary brms NCP + native Stan within that model. Total cost includes required pilots; native and WHMC warmup policies differ.

`reference/native/<case>/<arm>` retains clean and counter-instrumented Stan, resolved data/auto weights, initialization and gradient-count receipts. `reference/automatic_totals` retains the exact BRM exports and zero-error density/gradient export audits. `results/fits/manifest.json` lists 14 raw archives (51,400,524 bytes), with SHA-256 for every archive and member, independently read back. These include native CSV/RDS output, all final checkpoints, positions, gradients and controls. `results/receipts/manifest.json` records process completion and the environment.

Regional intercepts favor S2Z CP + WHMC in this pilot; independent intercepts/slopes favor BRM total CP. Weak NCP controls have divergences and some mean discrepancies. The docs retain these diagnostics; this is not a replicated method ranking.

## Reproduce

The fit environment is saved in `results/receipts`: Julia 1.10.11; BRM total implementation `54cbe3f` (landed as `5f30e53`) plus retained-flat-prior correction `f38bea8`; WHMC `7aed40b18bd4cdabb75330f285d5c9b355575ab9`; brms `73cf607889879cb2a55f50b88d8141d76ff43279`; CmdStan 2.39.0; CmdStanR 0.9.0. Replace machine-local manifest paths with those revisions. Set `PUPIL_BRMS_LIBRARY` to the pinned brms R library for native runs.

For `case=cluster-intercept` use `hierarchy=intercept_only` and `audit=air-totals-audit-intercept-v3`. For `case=cluster-independent` use `hierarchy=independent` and `audit=air-totals-audit-independent-v2`. From the repository root:

```sh
julia --project="$study_env" research/air_total_effects/audit.jl cluster_region "$hierarchy" "$study_scratch/$audit"
julia --project="$study_env" research/air_total_effects/run.jl cluster_region "$hierarchy" "$study_scratch/air-totals-$case-v1" "$study_scratch/$audit"
Rscript research/air_total_effects/native.R "$cmdstan" "$study_scratch/air-native-$case-v1/ordinary_ncp" cluster_region "$hierarchy" ordinary_ncp "$study_scratch/$audit/ordinary-init.json" "$study_scratch/$audit/audit.json"
python3 research/air_total_effects/native_matrix.py "$cmdstan" "$study_scratch/air-native-$case-v1" cluster_region "$hierarchy" "$study_scratch/$audit"
julia --project="$study_env" research/air_total_effects/complete.jl "$study_scratch" "$case" "research/air_total_effects/results/$case"
python3 research/air_total_effects/assemble.py "$study_scratch" "research/air_total_effects/results/$case" "$case"
julia --project="$study_env" research/air_total_effects/plot.jl "research/air_total_effects/results/$case" "$case"
python3 research/air_total_effects/build_pages.py /tmp/air-brief.md
```

`complete.jl` reuses completed fits, runs any missing S2Z WHMC arms, analyzes native draws, audits recovery and writes pair data. Use fresh output directories to refit. The native counter uses a zero-contribution C++ helper in the compiled model; the NUTS implementation is unchanged. It checks retained calls against leapfrogs plus one and records all precursor process counts.

Both automatic targets passed 25 checks. The independent post-hoc Pathfinder search encountered two ill-conditioned conditional precisions: higher-precision checks confirm mathematical positive definiteness while double precision fails. `target.jl` rejects this recognized numerical proposal error, and `numerical_audit.jl` records the checks in `results/cluster-independent/numerical_rejections.json`. Every attempted gradient call remains charged. Other exceptions propagate. Failed first-attempt logs are retained separately from the completed workflow costs. The documentation build samples nothing.
