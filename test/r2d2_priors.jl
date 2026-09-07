# test/r2d2_priors.jl — R2D2 variance-decomposition priors (`effect(lp, :) ~ r2d2(...)`).
#
# Run: julia --project=. test/r2d2_priors.jl
# Set BRM_R2D2_RUNTIME=0 to skip the BridgeStan density/gradient probe.
#
# The inciting case is the population-PK shape: a latent per-subject parameter
# whose subject random effect PLAYS the residual role, so `R2` splits its total
# between-subject variance `tau_bsv^2` into a covariate-explained part (split
# further over the population columns by a Dirichlet simplex) and an
# unexplained part that becomes the random-effect SD.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Beta, Exponential, LKJCholesky, Normal

const R2D2_CACHE = joinpath(tempdir(), "brm-r2d2-priors")
const R2D2_RUNTIME = get(ENV, "BRM_R2D2_RUNTIME", "1") != "0"

df = (;
    conc    = [2.4, 2.2, 2.0, 1.8, 1.7, 1.5, 1.6, 1.9],
    time    = [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
    wt      = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -0.25, 0.75],
    age     = [0.2, -0.4, 1.1, -0.9, 0.3, 0.7, -1.2, 0.1],
    subject = [1, 1, 1, 1, 2, 2, 2, 2],
)

builder = @brm begin
    log_CL ~ 1 + wt + age + (1 | p | subject)
    log_V  ~ 1 + wt + (1 | p | subject)

    effect(log_CL, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
    effect(log_V, :)  ~ r2d2(R2=Beta(2, 3), tau_bsv=0.25, alpha=0.5)

    conc ~ Normal(exp(log_CL - log_V) * time, 1)
end

@testset "r2d2 capture, introspection, and lowering" begin
    brmi = builder(df)

    specs = r2d2_priors(brmi)
    @test [s.predictor for s in specs] == [:log_CL, :log_V]
    @test all(s -> s.family === r2d2, specs)
    @test all(s -> isempty(s.arguments), specs)
    # Keyword values stay as unevaluated column carriers -- the backend turns
    # them into Stan constants, so the formula surface never materializes a
    # Distributions object.
    @test getf(specs[1].keywords.R2) === Beta
    @test getargs(specs[1].keywords.R2) == (1, 1)
    @test specs[1].keywords.tau_bsv == 0.5
    @test getargs(specs[2].keywords.R2) == (2, 3)
    @test specs[2].keywords.alpha == 0.5

    # A whole-predictor `Colon` address belongs to `r2d2_priors` alone -- it
    # names every population coefficient at once, so it must not leak into the
    # per-column population API.
    @test isempty(effect_priors(brmi))
    @test isempty(ranef_effect_priors(brmi))

    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    # `R2` is the constrained unit-interval parameter; `phi` the simplex over
    # the NON-intercept population columns only (the intercept design column is
    # constant, so `Var(x_k) = 0` there and it carries no explained variance).
    @test occursin("real<lower=0, upper=1> r2d2_log_CL_R2;", code)
    @test occursin("simplex[r2d2_log_CL_alpha_n] r2d2_log_CL_phi;", code)
    @test occursin("r2d2_log_CL_R2 ~ beta(1.0, 1.0);", code)
    @test occursin("r2d2_log_V_R2 ~ beta(2.0, 3.0);", code)
    @test occursin("r2d2_log_CL_phi ~ dirichlet(r2d2_log_CL_alpha);", code)
    @test sb.data[:r2d2_log_CL_alpha] == [1.0, 1.0]
    @test sb.data[:r2d2_log_CL_share_idx] == [0, 1, 2]
    @test sb.data[:r2d2_log_CL_tau_bsv] == 0.5

    # A single share (`log_V` has exactly one non-intercept column) now emits a
    # `simplex[1]` uniformly -- no `n_shares == 1` special-case. simplex[1] is
    # deterministically [1.0] with zero sampler dimensions, so it costs nothing
    # and keeps ONE Sb emission. `phi` is a parameter now, not a data constant.
    @test occursin("simplex[r2d2_log_V_alpha_n] r2d2_log_V_phi;", code)
    @test occursin("r2d2_log_V_phi ~ dirichlet(r2d2_log_V_alpha);", code)
    @test sb.data[:r2d2_log_V_alpha] == [0.5]
    @test !haskey(sb.data, :r2d2_log_V_phi)

    # Column variances are data-only, so they hoist to transformed data.
    @test occursin("vector[(2 + 1)] r2d2_log_CL_varx = brm_col_variances(", code)
    # The derived scale is a transformed parameter feeding the shipped
    # `_popefs_normal` seam.
    @test occursin("r2d2_log_CL_beta_scale = brm_r2d2_scale(", code)
    @test occursin(
        "pop_log_CL_beta_pop ~ normal(r2d2_log_CL_beta_loc, r2d2_log_CL_beta_scale);",
        code)

    # The shared `|p|` block's marginal scales are the derived residual scales,
    # one per predictor, and LKJ eta stays free at 1.
    @test occursin("sqrt(((1.0 - r2d2_log_CL_R2) * (r2d2_log_CL_tau_bsv ^ 2)))", code)
    @test occursin("sqrt(((1.0 - r2d2_log_V_R2) * (r2d2_log_V_tau_bsv ^ 2)))", code)
    @test occursin("vector[2] b_p_subject_r2d2_tau = [", code)
    @test occursin("b_p_subject_L ~ lkj_corr_cholesky(1.0);", code)
    # No sampled SD prior survives -- the decomposition derives the scales.
    @test !occursin("brm_ranef_sd", code)

    # Formula-derived R2D2 constants are recorded as frozen preprocessing, so
    # a shared-ID resample target can rebuild them and move only z to GQ.
    replayed = reprocess(sb, df; resample_groups=[:subject])
    replayed_code = BayesianRegressionModels.stan_code(replayed)
    @test StanBlocks.stan.transpiles(replayed.model)
    @test StanBlocks.stanc_check(replayed_code; warn_pedantic=false).ok
    replayed_block = only(b for b in ranef_blocks(replayed) if b.id === :p)
    @test replayed_block.generated
    @test replayed_block.family === :ranef_correlated_draws_r2d2

    if R2D2_RUNTIME
        isdir(R2D2_CACHE) || mkpath(R2D2_CACHE)
        problem = StanBlocks.stan_instantiate(
            sb.model;
            path=joinpath(R2D2_CACHE, string(hash(code)) * ".stan"),
        )
        dimension = LogDensityProblems.dimension(problem)
        q = [0.03 * ((i % 7) - 3) for i in 1:dimension]
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test length(gradient) == dimension
        @test all(isfinite, gradient)
    else
        @info "Skipping r2d2 BridgeStan gate (BRM_R2D2_RUNTIME=0)"
    end
end

@testset "plain (un-ID'd) random effect and the sampled-tau default" begin
    # No `tau_bsv=` kwarg: the between-subject scale becomes a sampled
    # half-standard-normal. A latent per-subject parameter has no observed
    # response to derive a scale from, so there is nothing else to default to.
    plain_builder = @brm begin
        eta ~ 1 + wt + age + (1 | subject)
        effect(:, :) ~ r2d2(R2=Beta(1, 1))
        conc ~ Normal(eta, 1)
    end
    plain = SBBRMI(plain_builder(df); mod=@__MODULE__)
    plain_code = BayesianRegressionModels.stan_code(plain)
    @test StanBlocks.stan.transpiles(plain.model)
    @test StanBlocks.stanc_check(plain_code; warn_pedantic=false).ok
    @test occursin("real<lower=0.0> r2d2_eta_tau_bsv;", plain_code)
    @test occursin("r2d2_eta_tau_bsv ~ std_normal();", plain_code)
    @test occursin("r_eta_subject_r2d2_scale = sqrt(((1.0 - r2d2_eta_R2) * (r2d2_eta_tau_bsv ^ 2)))",
                   plain_code)
    @test !occursin("ranef_intercept(", plain_code)
end

@testset "byte-identical emission when no r2d2 statement is present" begin
    control_builder = @brm begin
        eta ~ 1 + wt + age + (1 | subject)
        conc ~ Normal(eta, 1)
    end
    control = SBBRMI(control_builder(df); mod=@__MODULE__)
    control_code = BayesianRegressionModels.stan_code(control)
    @test !occursin("r2d2", control_code)
    @test occursin("pop_eta_beta_pop ~ std_normal();", control_code)
    @test occursin("r_eta_subject", control_code)
end

@testset "R2D2M2 random-effect block and per-margin ICC spellings" begin
    # One global R² and one simplex across the shared covariance block.  The
    # third margin uses a different observation channel's residual scale while
    # retaining the block's single LKJ prior.
    joint_builder = @brm begin
        sigma_pk ~ Exponential(1)
        sigma_qt ~ Exponential(1)
        log_Vc   ~ 1 + (1 | p | subject)
        log_k10  ~ 1 + (1 | p | subject)
        log_k12  ~ 1 + (1 | p | subject)
        log_k21  ~ 1 + (1 | p | subject)
        log_ka   ~ 1 + (1 | p | subject)
        qt_base  ~ 1 + (1 | p | subject)
        qt_slope ~ 1 + (1 | p | subject)
        sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=4,
                        concentration=0.7, reference_scale=sigma_pk)
        sd(qt_base, p)  ~ r2d2(reference_scale=sigma_qt)
        sd(qt_slope, p) ~ r2d2(reference_scale=sigma_qt)
        cor(:, p) ~ LKJCholesky(7, 2)
        conc ~ Normal(log_Vc + log_k10 + log_k12 + log_k21 + log_ka +
                      qt_base + qt_slope, sigma_pk)
    end
    joint_brmi = joint_builder(df)
    joint_specs = [s for s in ranef_effect_priors(joint_brmi)
                   if s.class === :sd]
    @test length(joint_specs) == 3
    @test all(s -> s.family === r2d2, joint_specs)

    joint = SBBRMI(joint_brmi; mod=@__MODULE__)
    joint_code = BayesianRegressionModels.stan_code(joint)
    @test StanBlocks.stan.transpiles(joint.model)
    @test StanBlocks.stanc_check(joint_code; warn_pedantic=false).ok
    @test occursin("b_p_subject_r2d2_1_R2 ~ beta(2.0, 2.0);", joint_code)
    @test occursin("simplex[b_p_subject_r2d2_1_alpha_n] b_p_subject_r2d2_1_phi;",
                   joint_code)
    @test joint.data[:b_p_subject_r2d2_1_alpha] == fill(0.7, 7)
    @test occursin("b_p_subject_L ~ lkj_corr_cholesky(2.0);", joint_code)
    @test occursin("sigma_pk", joint_code)
    @test occursin("sigma_qt", joint_code)
    @test count("sigma_pk ~ exponential", joint_code) == 1
    @test count("sigma_qt ~ exponential", joint_code) == 1
    @test !occursin("b_p_subject_tau ~", joint_code)

    # The same non-centred geometry is replayable for new group levels: only
    # z moves to generated quantities; R², phi, L and both reference scales
    # remain fitted coordinates.
    joint_cv = reprocess(joint, df; resample_groups=[:subject])
    joint_cv_code = BayesianRegressionModels.stan_code(joint_cv)
    @test StanBlocks.stan.transpiles(joint_cv.model)
    @test StanBlocks.stanc_check(joint_cv_code; warn_pedantic=false).ok
    block = only(b for b in ranef_blocks(joint_cv) if b.id === :p)
    @test block.generated
    @test block.family === :ranef_correlated_draws_r2d2

    # The older whole-predictor spelling can now retain an explicit LKJ eta on
    # the same shared bucket as well.
    flat_cor_builder = @brm begin
        eta_a ~ 1 + wt + (1 | p | subject)
        eta_b ~ 1 + age + (1 | p | subject)
        effect(eta_a, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        effect(eta_b, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        cor(:, p) ~ LKJCholesky(2, 3)
        conc ~ Normal(eta_a + eta_b, 1)
    end
    flat_cor = SBBRMI(flat_cor_builder(df); mod=@__MODULE__)
    flat_cor_code = BayesianRegressionModels.stan_code(flat_cor)
    @test StanBlocks.stan.transpiles(flat_cor.model)
    @test StanBlocks.stanc_check(flat_cor_code; warn_pedantic=false).ok
    @test occursin("b_p_subject_L ~ lkj_corr_cholesky(3.0);", flat_cor_code)

    # Without a block-wide statement each address owns an independent R².
    # A one-margin simplex is deterministic, so each addressed margin is the
    # per-margin ICC construction; the unaddressed middle margin keeps a free
    # half-Normal scale.
    icc_builder = @brm begin
        sigma_pk ~ Exponential(1)
        sigma_qt ~ Exponential(1)
        eta_a ~ 1 + (1 | p | subject)
        eta_b ~ 1 + (1 | p | subject)
        eta_c ~ 1 + (1 | p | subject)
        sd(eta_a, p) ~ r2d2(R2=Beta(1, 1), reference_scale=sigma_pk)
        sd(eta_c, p) ~ r2d2(mean_R2=0.25, prec_R2=4,
                            reference_scale=sigma_qt)
        cor(:, p) ~ LKJCholesky(3, 2)
        conc ~ Normal(eta_a + eta_b + eta_c, sigma_pk)
    end
    icc = SBBRMI(icc_builder(df); mod=@__MODULE__)
    icc_code = BayesianRegressionModels.stan_code(icc)
    @test StanBlocks.stan.transpiles(icc.model)
    @test StanBlocks.stanc_check(icc_code; warn_pedantic=false).ok
    @test occursin("b_p_subject_r2d2_1_R2 ~ beta(1.0, 1.0);", icc_code)
    @test occursin("b_p_subject_r2d2_2_R2 ~ beta(1.0, 3.0);", icc_code)
    @test occursin("b_p_subject_r2d2_free_tau_2 ~ std_normal();", icc_code)
    @test occursin("b_p_subject_L ~ lkj_corr_cholesky(2.0);", icc_code)
end

@testset "explicit per-column Normal priors compose with r2d2 -- until they exclude every column" begin
    # Snag `sbimpl-r2d2-expl-33fca9c1` (reporter Bruno:brm). A population
    # column with its own `effect(...) ~ Normal(...)` keeps that loc/scale and
    # leaves the Dirichlet allocation; the REMAINING columns and the
    # random-effect residual are still decomposed. The default-layer
    # `effect(:, col)` spelling excludes the column in every predictor it
    # reaches, exactly like the predictor-specific one.
    partial_builder = @brm begin
        log_CL ~ 1 + wt + age + (1 | p | subject)
        log_V  ~ 1 + wt + age + (1 | p | subject)
        effect(log_CL, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        effect(log_V, :)  ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        effect(:, wt) ~ Normal(0.2, 0.1)
        conc ~ Normal(exp(log_CL - log_V) * time, 1)
    end
    partial = SBBRMI(partial_builder(df); mod=@__MODULE__)
    partial_code = BayesianRegressionModels.stan_code(partial)
    @test StanBlocks.stan.transpiles(partial.model)
    @test StanBlocks.stanc_check(partial_code; warn_pedantic=false).ok
    # `wt` is excluded (share 0) and keeps its own loc/scale via the fallback
    # vectors; `age` is the single remaining share in both predictors.
    @test partial.data[:r2d2_log_CL_share_idx] == [0, 0, 1]
    @test partial.data[:r2d2_log_V_share_idx] == [0, 0, 1]
    @test partial.data[:r2d2_log_CL_fallback] == [1.0, 0.1, 1.0]
    @test partial.data[:r2d2_log_CL_beta_loc] == [0.0, 0.2, 0.0]
    @test partial.data[:r2d2_log_CL_alpha] == [1.0]
    @test occursin("r2d2_log_CL_R2 ~ beta(1.0, 1.0);", partial_code)
    @test occursin("r2d2_log_V_R2 ~ beta(1.0, 1.0);", partial_code)
    @test occursin("r2d2_log_CL_phi ~ dirichlet(r2d2_log_CL_alpha);", partial_code)
    @test occursin("r2d2_log_CL_beta_scale = brm_r2d2_scale(", partial_code)
    @test occursin("sqrt(((1.0 - r2d2_log_CL_R2) * (r2d2_log_CL_tau_bsv ^ 2)))",
                   partial_code)
    @test occursin("sqrt(((1.0 - r2d2_log_V_R2) * (r2d2_log_V_tau_bsv ^ 2)))",
                   partial_code)

    # Excluding EVERY non-intercept column contradicts the decomposition: the
    # pre-fix emission silently dropped `R2`/`phi` and pinned the shared block's
    # scale to the DATA constant `tau_bsv` (transpiled and passed stanc). Now it
    # fails closed, naming the excluded columns and both escapes.
    all_excluded = @brm begin
        log_CL ~ 1 + wt + age + (1 | p | subject)
        log_V  ~ 1 + wt + (1 | p | subject)
        effect(log_CL, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        effect(log_V, :)  ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        effect(:, wt) ~ Normal(0, 0.1)
        effect(log_CL, age) ~ Normal(0, 0.1)
        conc ~ Normal(exp(log_CL - log_V) * time, 1)
    end
    @test_throws "nothing to allocate" SBBRMI(all_excluded(df); mod=@__MODULE__)
    @test_throws "reference_scale" SBBRMI(all_excluded(df); mod=@__MODULE__)
    # One fully-excluded predictor is enough, even when its bucket sibling still
    # has a share.
    one_excluded = @brm begin
        log_CL ~ 1 + wt + age + (1 | p | subject)
        log_V  ~ 1 + wt + (1 | p | subject)
        effect(log_CL, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        effect(log_V, :)  ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        effect(log_V, wt) ~ Normal(0, 0.1)
        conc ~ Normal(exp(log_CL - log_V) * time, 1)
    end
    @test_throws "log_V" SBBRMI(one_excluded(df); mod=@__MODULE__)

    # The genuinely covariate-free member stays the decided no-op (decision
    # `1db6zkr`): a predictor with NO non-intercept population column has
    # nothing to explain, so the whole `tau_bsv` is its random-effect scale and
    # no `R2`/`phi` is emitted for it. The all-or-nothing bucket rule is what
    # forces the `r2d2` statement onto it, so this shape must keep lowering.
    covariate_free = @brm begin
        log_CL ~ 1 + wt + (1 | p | subject)
        log_ka ~ 1 + (1 | p | subject)
        effect(log_CL, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        effect(log_ka, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.25)
        conc ~ Normal(exp(log_CL - log_ka) * time, 1)
    end
    noop = SBBRMI(covariate_free(df); mod=@__MODULE__)
    noop_code = BayesianRegressionModels.stan_code(noop)
    @test StanBlocks.stan.transpiles(noop.model)
    @test StanBlocks.stanc_check(noop_code; warn_pedantic=false).ok
    @test occursin("r2d2_log_CL_R2 ~ beta(1.0, 1.0);", noop_code)
    @test !occursin("r2d2_log_ka_R2", noop_code)
    @test !occursin("r2d2_log_ka_phi", noop_code)
    @test noop.data[:r2d2_log_ka_tau_bsv] == 0.25
    @test !haskey(noop.data, :r2d2_log_ka_share_idx)
    @test occursin(r"b_p_subject_r2d2_tau = \[.*?r2d2_log_ka_tau_bsv\s*\]'"s, noop_code)
    @test occursin("sqrt(((1.0 - r2d2_log_CL_R2) * (r2d2_log_CL_tau_bsv ^ 2)))",
                   noop_code)
end

@testset "rejected shapes error loudly" begin
    # `:` is positional in every slot now, so `effect(:, eta)` parses — but an
    # r2d2 decomposition is whole-predictor by construction, so its address
    # still has to END in `:`. The refusal moved from the parser to the walker.
    not_whole = @brm begin
        eta ~ 1 + wt
        effect(:, eta) ~ r2d2(R2=Beta(1, 1))
        conc ~ Normal(eta, 1)
    end
    @test_throws "must end in" SBBRMI(not_whole(df); mod=@__MODULE__)

    # `effect(:, :) ~ r2d2(...)` only resolves when there is exactly one
    # population predictor to decompose.
    ambiguous = @brm begin
        eta_a ~ 1 + wt
        eta_b ~ 1 + age
        effect(:, :) ~ r2d2(R2=Beta(1, 1))
        conc ~ Normal(eta_a + eta_b, 1)
    end
    @test_throws "linear predictor" SBBRMI(ambiguous(df); mod=@__MODULE__)

    unknown_kwarg = @brm begin
        eta ~ 1 + wt
        effect(eta, :) ~ r2d2(R2=Beta(1, 1), nope=1)
        conc ~ Normal(eta, 1)
    end
    @test_throws "nope" SBBRMI(unknown_kwarg(df); mod=@__MODULE__)

    # All-or-nothing per shared `|ID|` bucket (decision `1db6zkr`): a bucket
    # whose margins are only partly r2d2-scoped has no consistent tau vector.
    partial_bucket = @brm begin
        eta_a ~ 1 + wt + (1 | p | subject)
        eta_b ~ 1 + age + (1 | p | subject)
        effect(eta_a, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        conc ~ Normal(eta_a + eta_b, 1)
    end
    @test_throws "all-or-nothing" SBBRMI(partial_bucket(df); mod=@__MODULE__)

    # An `sd(:, ...)` prior on an r2d2 block has nothing to apply to.
    sd_conflict = @brm begin
        eta ~ 1 + wt + (1 | p | subject)
        effect(eta, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        sd(:, p) ~ Exponential(1)
        conc ~ Normal(eta, 1)
    end
    @test_throws "DERIVES" SBBRMI(sd_conflict(df); mod=@__MODULE__)

    missing_reference = @brm begin
        eta ~ 1 + (1 | p | subject)
        sd(:, p) ~ r2d2(R2=Beta(1, 1))
        conc ~ Normal(eta, 1)
    end
    @test_throws "reference_scale" SBBRMI(missing_reference(df); mod=@__MODULE__)

    competing_ranef_scales = @brm begin
        eta ~ 1 + (1 | p | subject)
        sd(:, p) ~ r2d2(R2=Beta(1, 1), reference_scale=1)
        sd(eta, p) ~ Exponential(1)
        conc ~ Normal(eta, 1)
    end
    @test_throws "direct-scale" SBBRMI(competing_ranef_scales(df); mod=@__MODULE__)

    second_global_r2 = @brm begin
        eta_a ~ 1 + (1 | p | subject)
        eta_b ~ 1 + (1 | p | subject)
        sd(:, p) ~ r2d2(R2=Beta(1, 1), reference_scale=1)
        sd(eta_a, p) ~ r2d2(R2=Beta(2, 2), reference_scale=2)
        conc ~ Normal(eta_a + eta_b, 1)
    end
    @test_throws "may override only" SBBRMI(second_global_r2(df); mod=@__MODULE__)

    # Splitting the derived residual variance across several RE margins needs a
    # second simplex the flat decomposition does not build.
    multi_margin = @brm begin
        eta ~ 1 + wt + (1 + wt | subject)
        effect(eta, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        conc ~ Normal(eta, 1)
    end
    @test_throws "single intercept" SBBRMI(multi_margin(df); mod=@__MODULE__)

    multi_margin_id = @brm begin
        eta ~ 1 + wt + (1 + wt | p | subject)
        effect(eta, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        conc ~ Normal(eta, 1)
    end
    @test_throws "second simplex" SBBRMI(multi_margin_id(df); mod=@__MODULE__)

    # Adaptive centering and cv-contagious sizing both interact with a derived
    # scale in ways nobody has designed.
    centered = @brm begin
        eta ~ 1 + wt + (1 | subject)
        effect(eta, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        conc ~ Normal(eta, 1)
    end
    @test_throws "centered" SBBRMI(centered(df); mod=@__MODULE__,
                                   centered_groups=[:subject])
    @test_throws "cv_groups" SBBRMI(centered(df); mod=@__MODULE__,
                                    cv_groups=[:subject])

    stratified_df = merge(df, (; stratum=[1, 1, 1, 1, 2, 2, 2, 2]))
    stratified = @brm begin
        eta ~ 1 + wt + (1 | p | gr(subject, by=stratum))
        effect(eta, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        conc ~ Normal(eta, 1)
    end
    @test_throws "stratified" SBBRMI(stratified(stratified_df); mod=@__MODULE__)
end
