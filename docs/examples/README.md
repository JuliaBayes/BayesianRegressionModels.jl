# Bambi notebook replications (SBBRMI)

One script per [bambi example notebook](https://bambinos.github.io/bambi/notebooks/),
each fitting the notebook's model(s) through BRM's SBBRMI backend and writing JSON
summaries + draws. Companion briefs with numbers and plots live on the KB agent
`BayesianRegressionModels:bambi`; probe/fix scaffolding from development is
deliberately not landed here.

## Run

```sh
cd docs/examples
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. sleepstudy_pilot.jl   # fast smoke test
julia --project=. sleepstudy.jl
```

Inputs come from `data/` (vendored). All outputs (JSON summaries, generated
`.stan`, compiled models) go under `.out/` (gitignored). Scripts are seeded and
purge their Stan tags before instantiating, so reruns are reproducible and never
reuse stale programs. No `Manifest.toml` is committed: resolve locally (above)
or in CI; the `[sources]` revs in `Project.toml` pin the unregistered deps.

`modelcomp_loo.jl` reads the saved fits from `model_comparison.jl` — run that
first. `mister_p.jl` unpacks its gzipped input to `.out/tmp/` on first run
(requires `gunzip`).

## Map

| script | bambi notebook | data |
|---|---|---|
| logistic.jl | logistic_regression | ANES_2016_pilot.csv |
| sleepstudy.jl, sleepstudy_pilot.jl | sleepstudy | sleepstudy.csv |
| polynomial.jl | polynomial_regression | simulated |
| ordinal.jl | ordinal_regression | Trolley.csv, hr_employee_attrition.tsv.txt |
| seq_only.jl | ordinal_regression (cumulative + stopping-ratio) | hr_employee_attrition.tsv.txt |
| categorical.jl | categorical_regression | iris.csv |
| counts.jl | count_roaches | roaches.csv, nb_data.dta |
| zip.jl, hurdle_only.jl | zero_inflated_regression | fish.csv |
| beta_only.jl, rate_beta.jl | beta_regression | Batting.csv, carclaims.csv |
| wald_only.jl, rate_beta.jl | wald_gamma_glm | carclaims.csv |
| t_regression.jl | t_regression | simulated |
| radon.jl | radon_example | radon.csv |
| escs.jl | ESCS_multiple_regression | ESCS.csv |
| cherry.jl | splines_cherry_blossoms | cherry_blossoms.csv |
| pigs.jl | multi-level_regression | dietox.csv |
| newgroups.jl, newgroups_predict.jl | predict_new_groups | pulmonary_fibrosis.csv |
| mundlak.jl | fixed_random | simulated |
| hsgp1d.jl, hsgp2d.jl | — (BRM extra, no bambi counterpart) | simulated |
| links.jl | alternative_links_binary | simulated |
| hierbinomial.jl | hierarchical_binomial_bambi | Batting.csv |
| orthogonal.jl | orthogonal_polynomial_reg | simulated |
| circular.jl | circular_regression | periwinkles.csv |
| surv_disc.jl | survival_discrete_time_notebook | child.csv |
| surv_cont.jl | survival_continuous_time_notebook | retention.csv |
| surv_model.jl | survival_model | AustinCats.csv |
| distributional.jl | distributional_models | bike_sharing.csv |
| distsleep.jl | per_parameter_noncentered | sleepstudy.csv |
| mister_p.jl | mister_p | mr_p_cces18_common_vv.csv.gz, mr_p_poststrat_df.csv, mr_p_statelevel_predictors.csv |
| model_comparison.jl, modelcomp_loo.jl | model_comparison | adults.csv |
| strack.jl | Strack_RRR_re_analysis | rrr_long.csv |
| shooter.jl | shooter_crossed_random_ANOVA | shooter.csv |
| quantile.jl | quantile_regression | bmi.csv |
| prior_sensitivity.jl | prior_sensitivity | body_fat.csv |
| alternative_samplers.jl | alternative_samplers | (simulated) |
| kulprit.jl | kulprit | (simulated) |
| plot_predictions.jl | plot_predictions | mtcars.csv, nb_data.dta, movies.csv.gz |
| plot_slopes.jl | plot_slopes | Wells.csv |
| plot_comparisons.jl | plot_comparisons | fish.csv |
