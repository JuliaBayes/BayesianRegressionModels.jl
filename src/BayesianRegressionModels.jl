module BayesianRegressionModels

# @brm macro — parses a formula block, produces a BRMI (BRM
# intermediate: expression tree + column metadata).
using OrderedCollections
include("macro.jl")

# Introspection helpers on a BRMI -- outcomes / linear predictors /
# grouping factors / continuous & categorical predictors / RE terms.
# Used by web-macro's auto-PPC detector (and any downstream code that
# wants the model shape without re-walking the operations dict).
include("introspection.jl")

# VBRMI — vectorized implementation. Materializes predictors and
# likelihood into a LogDensityProblems-compatible object.
using LogExpFunctions, InverseFunctions, Distributions, ElasticArrays,
      LogDensityProblems, LinearAlgebra, SpecialFunctions, Random
using StatsBase: AbstractWeights, AnalyticWeights, FrequencyWeights,
                 ProbabilityWeights, UnitWeights, Weights,
                 aweights, fweights, pweights, uweights, weights
import CategoricalArrays as CA
include("likelihood_distributions.jl")
include("backend_plan.jl")
include("turing_backend.jl")
include("native_ppl.jl")
include("vimpl.jl")

# SBBRMI — StanBlocks backend. Lowers a BRMI into a StanBlocks SlicModel
# so it can be compiled by BridgeStan / fit via Stan.
using StanBlocks
include("sbimpl.jl")

# BRMDescriptor — ONE authoritative executable semantic model descriptor.
# Collapses the GenerativePlan (what BRM emitted), introspection.jl (the
# formula shape), the preproc record (dataframe provenance) and StanBlocks'
# own `stan_descriptor` (the executable half) into one reflectable value with
# derived operations, so a consumer mounts a declaration without keeping any
# parallel registry of its own.
include("descriptor.jl")

# Post-fit prediction modes — population-level ("no random effects") and
# transported ("same draws, new covariates / new group levels") prediction.
# Both are unconstrained-draw-matrix operations whose correctness depends on
# the emitted random-effect parameterization, so they live next to the emitter
# rather than being re-derived (differently) in every consumer.
include("prediction.jl")
include("adaptive_centering.jl")

# Public surface. The macros and value types everything downstream
# (web-macro, downstream extensions, tests) reaches for.
export @brm, @n, @x, @getproperty
export assign, effect, r2d2, doublepipe, gr, mm, gp, offset, zscale, center, standardize, protect, factor
export LKJCovarianceFactor, MvNormalCholesky
export weighted, AbstractWeights, AnalyticWeights, FrequencyWeights,
       ProbabilityWeights, UnitWeights, Weights,
       aweights, fweights, pweights, uweights, weights
export me, mi, s, t2, ar, dar, mo, mo1, hsgp, OrderedLogistic, Ordinal,
       OrdinalStructure, Cumulative, StoppingRatio,
       OrdinalLink, LogitLink, ProbitLink, CloglogLink, Horseshoe,
       CategoricalLogit, ZeroInflatedPoisson, NegativeBinomial2,
       BetaBinomial, BetaBinomial2, CircularVonMises, SkewDoubleExponential,
       sb_group_demo, addprop
# Hierarchical biomarker family — formula-surface markers for custom likelihood + submodel
export TruncatedNormal, biomarker_hierarchical_parametric, kernel, ragged
export brm_multinomial, brm_multinomial_lpmf, brm_multinomial_lpmfs, brm_multinomial_rng
# Julia-native response-family composition. `truncated` and `censored` are the
# exact Distributions.jl functions; `interval_censored` is BRM's formula marker
# for row-wise interval evidence on either a response or a predictor.
export truncated, censored, interval_censored
# A julianic `@jmodel` body is ordinary Julia, so the distributions it names must
# be real `Distributions` objects resolved in the AUTHOR's scope. Re-export the
# ones a model body actually writes so `using BayesianRegressionModels` is the
# only import a model file needs — no `using Distributions` on top (todo
# `0n9bz7p`, decision `0w84aut`). These are `Distributions.*` verbatim, not
# wrappers. Note `NativePPL` exports its OWN `Exponential`/`StandardNormal` —
# the DECLARATIVE prior *declarations*, a different thing — so a caller who
# `using`s both modules must qualify that one name; `Normal` does not clash at
# all, because the declarative macro matches the bare symbol at expansion time
# rather than resolving a binding.
export Normal, Exponential, Poisson, Bernoulli, LKJCholesky
export product_distribution, logpdf
# Logit-form Bernoulli/Binomial -- prefer these over `Bernoulli(logistic(eta))`
# / `Binomial(N, logistic(eta))`. Both backends lower to a logit-native log-pmf
# (Stan's `bernoulli_logit_lpmf` / `binomial_logit_lpmf`; LogExpFunctions'
# `loglogistic` / `log1mlogistic` on the Julia side), avoiding the `inv_logit`
# round-trip and staying numerically stable for large |eta|. `BernoulliLogit`
# is re-exported from Distributions; `BinomialLogit` is defined alongside the
# other executable likelihood contracts in likelihood_distributions.jl.
export BernoulliLogit, BinomialLogit
# SLIC custom-family bindings must be visible in a caller-supplied model module
# (`brm_descriptor(...; mod=@__MODULE__)`), matching the existing exported
# zero-inflated-Poisson and von-Mises UDF triads.
export brm_ordinal, brm_ordinal_lpmf, brm_ordinal_lpmfs, brm_ordinal_rng,
       brm_ordinal_logcdf, brm_ordinal_logccdf, brm_ordinal_cdf,
       multi_std_normal, multi_std_normal_lpdf, ranef_b_matrix,
       brm_ranef_sd, brm_ranef_sd_lpdf
export Data, MaybeData, maybedata
export AbstractColumn, MissingColumn, DataColumn, NamedColumn,
       ExprColumn, LikelihoodColumn, MaterializedColumn
export BRMI, VBRMI, SBBRMI, TuringBRMI, GenerativeDeclaration, GenerativePlan,
       turing_pointwise_loglikelihoods, turing_predictive_model,
       turing_generated_quantities, turing_posterior_predictive
export NativePPL
export BRMDescriptor, BRMInput, BRMOutput, BRMOperation, BRMHighlight
export brm_descriptor, brm_output, brm_outputs, brm_output_coordinates,
       brm_population_effect_coordinates, brm_term_coordinates,
       brm_ranef_sd_coordinates,
       brm_operation, brm_execute, brm_columns, required_brm_inputs

# Accessor helpers for column types — used unqualified by html renderers,
# sbimpl dispatch logic, and downstream extension hooks.
export name, getf, getargs, getkwargs, getbroadcast, getop

# Macro plumbing + compiled-output accessors used by downstream code
# (web-macro's Formula struct, downstream direct pipeline calls).
export parse!, _brm, stan_code, reprocess, restan_data, generative_plan

# Post-fit prediction — the population-level ("nore") and transported
# ("recov") modes, plus the random-effect block description both stand on.
export RanefBlock, ranef_blocks, ranef_coordinates,
       population_draws, transport_draws, term_draws, hsgp_population_curve
export AdaptiveCenteringBlock, adaptive_centering_blocks,
       adaptive_centering_problem

# Introspection -- model-shape questions answered without re-walking
# the operations dict.
export outcomes, linear_predictor_op, linear_predictors, predictors,
       grouping_factors, column_data, data_columns, dependencies,
       hierarchical_outcomes,
       linear_predictor_args, data_args, primary_lp, popcoefnames, effect_priors,
       ranef_effect_priors, r2d2_priors, term_priors, ranefcoefnames,
       priors_of, structure_of

# Extension API. Downstream packages add their own formula terms
# by defining methods on `vmeta_sampling_rhs` + `_sb_submodel_rhs!` and
# pushing `Part`s via `push_parts!!`; `nparams` + `lprior!` complete the
# vimpl side. `vbroadcasted` is the materializer they call to resolve
# column args inside those method bodies.
export Part, push_parts!!, nparams, lprior!
export vbroadcasted, vmeta_sampling_rhs, _sb_submodel_rhs!
export _sb_term_group_block, _sb_emit_group_block_term!

# Internal @slic submodels from sbimpl.jl — exported so downstream
# modules (downstream extensions via web-macro) that build their own SBBRMI-style
# models with `StanBlocks.SlicModel(body, data, mod)` where `mod` is the
# caller's namespace can still resolve the BRM built-in submodel names.
export popefs, _popefs_normal, _popefs_coefs, _popefs_normal_coefs,
       cdirichlet, c0dirichlet, c01dirichlet,
       ranef_intercept, ranef_intercept_draws, ranef_correlated, ranef_correlated_by,
       ranef_correlated_draws, ranef_correlated_by_draws,
       ranef_intercept_centered, ranef_correlated_centered,
       ranef_correlated_draws_centered,
       ranef_correlated_draws_effect,
       ranef_correlated_draws_centered_effect,
       ranef_correlated_draws_r2d2, ranef_intercept_r2d2, ranef_correlated_r2d2,
       brm_col_variances, brm_r2d2_scale, brm_cat_variances,
       multi_membership_intercept, multi_membership_correlated,
       _sb_mo, _sb_cat, _sb_cat_normal, _sb_ar1, _sb_dar1, _sb_s, _sb_t2, _sb_me,
       _sb_interval_censored_predictor,
       _sb_gp, _sb_gp_aniso, _sb_hsgp, _sb_hsgp_aniso,
       _sb_hsgp_by, _sb_hsgp_by_aniso,
       _sb_hsgp_latent, _sb_hsgp_latent_orthogonal,
       _sb_gp_periodic, _sb_hsgp_periodic,
       brm_exp_quad_cov, brm_hsgp_sqrt_spd, brm_hsgp_basis_1d,
       brm_hsgp_orthogonalize_linear,
       brm_periodic_cov, brm_hsgp_periodic_sqrt_spd,
       _sb_horseshoe, _sb_horseshoe_scaled,
       _sb_mi_normal, mi_merge,
       zero_inflated_poisson, zero_inflated_poisson_lpmf,
       zero_inflated_poisson_lpmfs, zero_inflated_poisson_rng,
       brm_von_mises, brm_von_mises_lpdf,
       brm_von_mises_lpdfs, brm_von_mises_rng,
       sb_group_demo_slic, sb_group_clamped_demo, sb_group_clamped_demo_slic

end # module
