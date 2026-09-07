# test/r2d2_joint_priors.jl — the JOINT R2D2M2 budget:
# `sd(:, ID) ~ r2d2(...; include=(:population, :contrasts))`.
#
# Run: julia --project=test test/r2d2_joint_priors.jl
# Set BRM_R2D2_RUNTIME=0 to skip the BridgeStan density/gradient probe.
#
# The inciting case (snag `request-joint-r2`, reporter Bruno:brm) is a joint
# PK/QT model whose seven subject-level parameters share one correlated `|p|`
# block and one covariate RHS with continuous AND categorical terms, and wants
# ONE global R² and ONE Dirichlet over the union of: every scoped predictor's
# non-intercept population columns, its categorical contrast coefficients, and
# the block's random-effect variances — each scaled by its margin's reference.
# Before this, `sd(:, p) ~ r2d2(...)` allocated the margins only and
# `effect(lp, :) ~ r2d2(...)` was per predictor and skipped `cat_*` blocks.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Beta, Exponential, LKJCholesky, Normal

const BS = StanBlocks.BridgeStan

const R2D2J_CACHE = joinpath(tempdir(), "brm-r2d2-joint-priors")
const R2D2J_RUNTIME = get(ENV, "BRM_R2D2_RUNTIME", "1") != "0"

df = (;
    conc       = [2.4, 2.2, 2.0, 1.8, 1.7, 1.5, 1.6, 1.9, 2.1, 1.4, 1.3, 1.8],
    wt         = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -0.25, 0.75, 0.3, -0.8, 1.2, -0.1],
    age        = [0.2, -0.4, 1.1, -0.9, 0.3, 0.7, -1.2, 0.1, 0.5, -0.6, 0.9, -0.3],
    indication = [1, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2],   # Int -> `cat_*` contrast block
    subject    = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
)

joint_builder = @brm begin
    sigma_pk ~ Exponential(1)
    sigma_qt ~ Exponential(1)

    log_Vc  ~ 1 + wt + indication + (1 | p | subject)
    log_k10 ~ 1 + wt + age + (1 | p | subject)
    qt_base ~ 1 + wt + indication + (1 | p | subject)

    sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=4, concentration=0.7,
                    reference_scale=sigma_pk,
                    include=(:population, :contrasts))
    sd(qt_base, p) ~ r2d2(reference_scale=sigma_qt)
    cor(:, p) ~ LKJCholesky(3, 2)

    conc ~ Normal(log_Vc + log_k10 + qt_base, sigma_pk)
end

@testset "joint R2D2M2 budget: capture, lowering, descriptor, replay" begin
    brmi = joint_builder(df)

    # Provenance: the joint statement is an ordinary block-wide `sd` statement
    # whose `include` keyword rides through `ranef_effect_priors` verbatim; the
    # whole-predictor walker stays empty because no `effect(lp, :)` exists.
    specs = [s for s in ranef_effect_priors(brmi) if s.class === :sd]
    @test length(specs) == 2
    @test all(s -> s.family === r2d2, specs)
    block = only(s for s in specs if isnothing(s.predictor))
    @test block.keywords.include == (:population, :contrasts)
    @test isempty(r2d2_priors(brmi))
    @test isempty(effect_priors(brmi))
    @test [m.predictor for m in ranefcoefnames(brmi, :p)] ==
          [:log_Vc, :log_k10, :qt_base]

    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    # ONE global R² and ONE simplex over 3 margins + 4 continuous columns
    # (wt | wt, age | wt) + 2 contrast blocks of one contrast each = 9.
    @test count("_R2 ~ beta(", code) == 1
    @test occursin("b_p_subject_r2d2_1_R2 ~ beta(2.0, 2.0);", code)
    @test count("~ dirichlet(", code) == 1
    @test occursin("simplex[b_p_subject_r2d2_1_alpha_n] b_p_subject_r2d2_1_phi;",
                   code)
    @test sb.data[:b_p_subject_r2d2_1_alpha] == fill(0.7, 9)

    # Margins first (1..3), then, per scoped predictor in formula order, its
    # population columns and then its contrast blocks; the intercept never
    # enters.
    @test sb.data[:r2d2_log_Vc_share_idx]  == [0, 4]
    @test sb.data[:cat_log_Vc_indication_r2d2_share_idx]  == [5]
    @test sb.data[:r2d2_log_k10_share_idx] == [0, 6, 7]
    @test sb.data[:r2d2_qt_base_share_idx] == [0, 8]
    @test sb.data[:cat_qt_base_indication_r2d2_share_idx] == [9]

    # Population columns keep the `_popefs_normal` seam and the SAME `beta_pop`
    # carrier; their scale is the R2D2M2 one, `ref * sqrt(phi R2 / ((1-R2) varx))`,
    # spelled through the shipped `brm_r2d2_scale` with `tau = ref / sqrt(1-R2)`.
    @test occursin("pop_log_Vc_beta_pop ~ normal(r2d2_log_Vc_beta_loc, r2d2_log_Vc_beta_scale);",
                   code)
    @test occursin("pop_log_k10_beta_pop ~ normal(r2d2_log_k10_beta_loc, r2d2_log_k10_beta_scale);",
                   code)
    @test occursin("r2d2_log_Vc_varx = brm_col_variances(", code)
    @test occursin("(sigma_pk / sqrt((1.0 - b_p_subject_r2d2_1_R2)))", code)
    # qt_base's population columns and contrasts take ITS margin reference.
    @test occursin("(sigma_qt / sqrt((1.0 - b_p_subject_r2d2_1_R2)))", code)

    # Categorical contrasts keep the `cat_<lp>_<col>_beta` carrier through the
    # `_sb_cat_normal` sibling, scaled by the dummy-column variance.
    @test occursin("cat_log_Vc_indication_r2d2_varx = brm_cat_variances(", code)
    @test occursin("cat_log_Vc_indication_r2d2_beta_scale = brm_r2d2_scale(", code)
    @test occursin("cat_log_Vc_indication_beta ~ normal(", code)
    @test occursin("cat_qt_base_indication_beta ~ normal(", code)
    @test !occursin("cat_log_Vc_indication_beta ~ std_normal", code)

    # The block's derived margins and free LKJ factor are the plain M2 ones.
    @test occursin("vector[3] b_p_subject_r2d2_tau = [", code)
    @test occursin("sigma_qt *", code)
    @test occursin("b_p_subject_L ~ lkj_corr_cholesky(2.0);", code)
    @test !occursin("brm_ranef_sd", code)
    @test !occursin("b_p_subject_tau ~", code)

    # Non-centred shared-ID resample replay: only z moves to GQ.
    replayed = reprocess(sb, df; resample_groups=[:subject])
    replayed_code = BayesianRegressionModels.stan_code(replayed)
    @test StanBlocks.stan.transpiles(replayed.model)
    @test StanBlocks.stanc_check(replayed_code; warn_pedantic=false).ok
    replayed_block = only(b for b in ranef_blocks(replayed) if b.id === :p)
    @test replayed_block.generated
    @test replayed_block.family === :ranef_correlated_draws_r2d2

    # Frozen prediction replay keeps the program byte-identical.
    @test BayesianRegressionModels.stan_code(reprocess(sb, df)) == code

    if R2D2J_RUNTIME
        isdir(R2D2J_CACHE) || mkpath(R2D2J_CACHE)
        d = brm_descriptor(joint_builder, df; mod=@__MODULE__)
        problem = brm_execute(d, :fit;
            path=joinpath(R2D2J_CACHE, string(hash(code)) * ".stan"))
        dimension = LogDensityProblems.dimension(problem)
        q = [0.03 * ((i % 7) - 3) for i in 1:dimension]
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test length(gradient) == dimension
        @test all(isfinite, gradient)

        # Consumer readers are unchanged: contrast semantics, the fitted
        # per-margin group SD (a transformed parameter), and the intercept.
        names = BS.param_names(
            problem.model; include_tp=true, include_gq=false)
        contrast = brm_population_effect_coordinates(
            d, :log_Vc, names; coefficient=:indication)
        @test contrast.reference_level == 1
        @test contrast.nonreference_levels == [2]
        @test length(contrast.coordinates) == 1
        @test names[only(contrast.coordinates)] == "cat_log_Vc_indication_beta.1"
        intercept = brm_population_effect_coordinates(
            d, :log_Vc, names; coefficient=:Intercept)
        @test names[only(intercept.coordinates)] == "pop_log_Vc_beta_pop.1"
        tau = brm_ranef_sd_coordinates(d, :qt_base, names; id=:p)
        @test names[only(tau.coordinates)] == "b_p_subject_r2d2_tau.3"
    else
        @info "Skipping joint r2d2 BridgeStan gate (BRM_R2D2_RUNTIME=0)"
    end
end

@testset "omitted reference scale is sampled; intercept overrides compose" begin
    latent_builder = @brm begin
        log_Vc  ~ 1 + wt + (1 | p | subject)
        log_k10 ~ 1 + wt + age + (1 | p | subject)
        effect(log_Vc, Intercept) ~ Normal(1, 2)
        sd(:, p) ~ r2d2(R2=Beta(1, 1), include=:population)
        conc ~ Normal(log_Vc + log_k10, 1)
    end
    latent = SBBRMI(latent_builder(df); mod=@__MODULE__)
    latent_code = BayesianRegressionModels.stan_code(latent)
    @test StanBlocks.stan.transpiles(latent.model)
    @test StanBlocks.stanc_check(latent_code; warn_pedantic=false).ok
    # One sampled half-standard-normal reference per margin, and every
    # component of that margin's predictor is measured in it.
    @test occursin("real<lower=0.0> b_p_subject_r2d2_1_ref_1;", latent_code)
    @test occursin("real<lower=0.0> b_p_subject_r2d2_1_ref_2;", latent_code)
    @test occursin("b_p_subject_r2d2_1_ref_1 ~ std_normal();", latent_code)
    @test occursin("(b_p_subject_r2d2_1_ref_2 / sqrt((1.0 - b_p_subject_r2d2_1_R2)))",
                   latent_code)
    @test latent.data[:b_p_subject_r2d2_1_alpha] == fill(1.0, 2 + 3)
    # The intercept stays outside and keeps its explicit prior.
    @test latent.data[:r2d2_log_Vc_beta_loc] == [1.0, 0.0]
    @test latent.data[:r2d2_log_Vc_fallback] == [2.0, 1.0]
    @test latent.data[:r2d2_log_Vc_share_idx] == [0, 3]
    @test !occursin("cat_", latent_code)
end

@testset "plain R2D2M2 emission is untouched without include=" begin
    plain_builder = @brm begin
        sigma_pk ~ Exponential(1)
        log_Vc  ~ 1 + wt + indication + (1 | p | subject)
        log_k10 ~ 1 + wt + age + (1 | p | subject)
        sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=4, reference_scale=sigma_pk)
        conc ~ Normal(log_Vc + log_k10, sigma_pk)
    end
    plain = SBBRMI(plain_builder(df); mod=@__MODULE__)
    plain_code = BayesianRegressionModels.stan_code(plain)
    @test StanBlocks.stanc_check(plain_code; warn_pedantic=false).ok
    @test plain.data[:b_p_subject_r2d2_1_alpha] == fill(1.0, 2)
    @test occursin("pop_log_Vc_beta_pop ~ std_normal();", plain_code)
    @test occursin("cat_log_Vc_indication_beta ~ std_normal();", plain_code)
    @test !haskey(plain.data, :r2d2_log_Vc_share_idx)
    @test !occursin("brm_cat_variances", plain_code)
    @test !occursin("_ref_", plain_code)
end

@testset "rejected shapes error loudly" begin
    # A per-column Normal override inside the scope would silently pull that
    # coefficient out of the simplex; the joint budget refuses it.
    column_override = @brm begin
        log_Vc  ~ 1 + wt + (1 | p | subject)
        log_k10 ~ 1 + age + (1 | p | subject)
        effect(log_Vc, wt) ~ Normal(0, 1)
        sd(:, p) ~ r2d2(reference_scale=1.0, include=:population)
        conc ~ Normal(log_Vc + log_k10, 1)
    end
    @test_throws "conflicts with the joint decomposition" SBBRMI(
        column_override(df); mod=@__MODULE__)

    contrast_override = @brm begin
        log_Vc  ~ 1 + indication + (1 | p | subject)
        log_k10 ~ 1 + age + (1 | p | subject)
        effect(log_Vc, indication) ~ Normal(0, 1)
        sd(:, p) ~ r2d2(reference_scale=1.0, include=(:population, :contrasts))
        conc ~ Normal(log_Vc + log_k10, 1)
    end
    @test_throws "conflicts with the joint decomposition" SBBRMI(
        contrast_override(df); mod=@__MODULE__)

    # A default-layer override reaches the scoped columns too.
    default_layer = @brm begin
        log_Vc  ~ 1 + wt + (1 | p | subject)
        log_k10 ~ 1 + age + (1 | p | subject)
        effect(:, :) ~ Normal(0, 1)
        sd(:, p) ~ r2d2(reference_scale=1.0, include=:population)
        conc ~ Normal(log_Vc + log_k10, 1)
    end
    @test_throws "conflicts with the joint decomposition" SBBRMI(
        default_layer(df); mod=@__MODULE__)

    # `include=` is a property of the block-wide statement.
    include_on_override = @brm begin
        log_Vc  ~ 1 + wt + (1 | p | subject)
        log_k10 ~ 1 + age + (1 | p | subject)
        sd(:, p) ~ r2d2(reference_scale=1.0, include=:population)
        sd(log_Vc, p) ~ r2d2(reference_scale=2.0, include=:population)
        conc ~ Normal(log_Vc + log_k10, 1)
    end
    @test_throws "may override only" SBBRMI(include_on_override(df); mod=@__MODULE__)

    include_on_icc = @brm begin
        log_Vc  ~ 1 + wt + (1 | p | subject)
        log_k10 ~ 1 + age + (1 | p | subject)
        sd(log_Vc, p) ~ r2d2(reference_scale=1.0, include=:population)
        conc ~ Normal(log_Vc + log_k10, 1)
    end
    @test_throws "requires the block-wide address" SBBRMI(
        include_on_icc(df); mod=@__MODULE__)

    unknown_member = @brm begin
        log_Vc  ~ 1 + wt + (1 | p | subject)
        log_k10 ~ 1 + age + (1 | p | subject)
        sd(:, p) ~ r2d2(reference_scale=1.0, include=:nope)
        conc ~ Normal(log_Vc + log_k10, 1)
    end
    @test_throws "unknown `include=` member" SBBRMI(
        unknown_member(df); mod=@__MODULE__)

    ranef_only = @brm begin
        log_Vc  ~ 1 + wt + (1 | p | subject)
        log_k10 ~ 1 + age + (1 | p | subject)
        sd(:, p) ~ r2d2(reference_scale=1.0, include=:ranef)
        conc ~ Normal(log_Vc + log_k10, 1)
    end
    @test_throws "names no population component" SBBRMI(
        ranef_only(df); mod=@__MODULE__)

    # A scoped predictor cannot also carry the whole-predictor decomposition.
    double_budget = @brm begin
        log_Vc  ~ 1 + wt + (1 | p | subject)
        log_k10 ~ 1 + age + (1 | p | subject)
        effect(log_Vc, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        sd(:, p) ~ r2d2(reference_scale=1.0, include=:population)
        conc ~ Normal(log_Vc + log_k10, 1)
    end
    @test_throws "choose one" SBBRMI(double_budget(df); mod=@__MODULE__)

    # The plain M2 form keeps its mandatory observation reference.
    missing_reference = @brm begin
        log_Vc  ~ 1 + wt + (1 | p | subject)
        sd(:, p) ~ r2d2(R2=Beta(1, 1))
        conc ~ Normal(log_Vc, 1)
    end
    @test_throws "reference_scale" SBBRMI(missing_reference(df); mod=@__MODULE__)
end
