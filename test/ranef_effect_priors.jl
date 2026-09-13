# test/ranef_effect_priors.jl — effect-DSL covariance priors for shared |ID| blocks.
#
# Run: julia --project=. test/ranef_effect_priors.jl
# Set BRM_RANEF_EFFECT_RUNTIME=0 to skip the BridgeStan density/gradient probe.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Cauchy, Exponential, LKJCholesky, LocationScale, Normal

const RANEF_EFFECT_CACHE = joinpath(tempdir(), "brm-ranef-effect-priors")
const RANEF_EFFECT_RUNTIME = get(ENV, "BRM_RANEF_EFFECT_RUNTIME", "1") != "0"

df = (;
    x = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    cat = [1, 2, 3, 1, 2, 3],
    subject = [1, 1, 2, 2, 3, 3],
    y = [-2.4, -2.2, -2.0, -1.8, -1.7, -1.5],
)

builder = @brm begin
    eta_CL ~ 1 + (1 | p | subject)
    eta_Vc ~ 1 + x + cat + (1 + x + cat | p | subject)
    eta_Q  ~ x + (0 + x | p | subject)

    effect(eta_Vc, x) ~ Normal(0, 0.25)
    sd(:, p) ~ Exponential(2 / 3)
    sd(eta_Vc, p, x) ~ Exponential(1 / 3)
    sd(eta_Vc, p, cat_dummy_3) ~ Exponential(1 / 5)
    sd(eta_Q, p) ~ Exponential(1 / 4)
    cor(:, p) ~ LKJCholesky(6, 2)

    y ~ Normal(eta_CL + eta_Vc + eta_Q, 1)
end

@testset "ranef effect capture, margin names, and lowering" begin
    brmi = builder(df)
    @test [(p.predictor, p.coefficient) for p in effect_priors(brmi)] ==
          [(:eta_Vc, :x)]

    specs = ranef_effect_priors(brmi)
    @test length(specs) == 5
    @test [s.class for s in specs] == [:sd, :sd, :sd, :sd, :cor]
    @test all(s -> s.id === :p, specs)
    @test [(s.predictor, s.coefficient) for s in specs] ==
          [(nothing, nothing), (:eta_Vc, :x),
           (:eta_Vc, :cat_dummy_3), (:eta_Q, nothing),
           (nothing, nothing)]

    margins = ranefcoefnames(brmi, :p)
    @test margins == [
        (; predictor=:eta_CL, coefficient=:Intercept),
        (; predictor=:eta_Vc, coefficient=:Intercept),
        (; predictor=:eta_Vc, coefficient=:x),
        (; predictor=:eta_Vc, coefficient=:cat_dummy_2),
        (; predictor=:eta_Vc, coefficient=:cat_dummy_3),
        (; predictor=:eta_Q, coefficient=:x),
    ]
    @test isnothing(ranefcoefnames(brmi, :absent))

    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    # A shared bucket keeps one tau vector. Its generated family sums one
    # concrete scalar kernel per margin and provides a matching sized RNG.
    @test occursin(r"real brm_vector_prior_[0-9a-f]+_lpdf", code)
    @test !occursin("brm_ranef_sd", code)
    @test occursin("b_p_subject_L ~ lkj_corr_cholesky(2.0);", code)
    @test occursin(r"b_p_subject_tau ~ brm_vector_prior_[0-9a-f]+", code)
    @test count("exponential_lpdf(x[", code) == 6

    plan = generative_plan(sb)
    decl = only(d for d in plan.declarations if d.target === :b_p_subject)
    @test decl.family isa StanBlocks.SlicModel
    @test decl.keywords.lkj_eta == 2.0

    block = only(ranef_blocks(sb))
    @test block.family === :ranef_correlated_draws_generic
    @test block.id === :p
    @test (block.n_terms, block.n_groups) == (6, 3)
    @test brm_descriptor(sb) isa BRMDescriptor

    n_l = block.n_terms * (block.n_terms - 1) ÷ 2
    unc_names = vcat(
        ["b_p_subject_L.$i" for i in 1:n_l],
        ["b_p_subject_tau.$i" for i in 1:block.n_terms],
        ["b_p_subject_z_flat.$i" for i in 1:(block.n_terms * block.n_groups)],
    )
    adaptive = only(adaptive_centering_blocks(sb, unc_names))
    @test adaptive.ranef.binding === block.binding
    @test adaptive.ranef.family === block.family
    @test adaptive.target_c == 0.0
    @test size(adaptive.effects) == (6, 3)
    @test length(adaptive.cholesky_free) == n_l
    @test length(adaptive.log_scales) == 6
    fake_draws = ones(2, length(unc_names))
    population = population_draws(sb, fake_draws, unc_names; groups=:subject)
    z_indices = vec(ranef_coordinates(block, unc_names))
    @test all(iszero, population[:, z_indices])
    @test all(isone, population[:, setdiff(eachindex(unc_names), z_indices)])

    if RANEF_EFFECT_RUNTIME
        isdir(RANEF_EFFECT_CACHE) || mkpath(RANEF_EFFECT_CACHE)
        problem = StanBlocks.stan_instantiate(
            sb.model;
            path=joinpath(RANEF_EFFECT_CACHE, string(hash(code)) * ".stan"),
        )
        dimension = LogDensityProblems.dimension(problem)
        q = [0.03 * ((i % 7) - 3) for i in 1:dimension]
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test length(gradient) == dimension
        @test all(isfinite, gradient)
    else
        @info "Skipping ranef-effect BridgeStan gate (BRM_RANEF_EFFECT_RUNTIME=0)"
    end
end

@testset "partial margins, CV, centered, and historical default" begin
    partial_builder = @brm begin
        eta_CL ~ 1 + (1 | p | subject)
        eta_Vc ~ 1 + x + (1 + x | p | subject)
        sd(eta_Vc, p, x) ~ Exponential(1 / 4)
        y ~ Normal(eta_CL + eta_Vc, 1)
    end
    partial_brmi = partial_builder(df)
    partial = SBBRMI(partial_brmi; mod=@__MODULE__)
    partial_code = BayesianRegressionModels.stan_code(partial)
    @test StanBlocks.stanc_check(partial_code; warn_pedantic=false).ok
    @test occursin(r"b_p_subject_tau ~ brm_vector_prior_[0-9a-f]+", partial_code)
    partial_decl = only(d for d in generative_plan(partial).declarations
                        if d.target === :b_p_subject)
    @test partial_decl.keywords.lkj_eta == 1.0

    centered = SBBRMI(partial_brmi; mod=@__MODULE__, centered_groups=[:subject])
    centered_code = BayesianRegressionModels.stan_code(centered)
    @test StanBlocks.stanc_check(centered_code; warn_pedantic=false).ok
    centered_block = only(ranef_blocks(centered))
    @test centered_block.family === :ranef_correlated_draws_centered_generic
    centered_unc_names = vcat(
        ["b_p_subject_L.$i" for i in 1:3],
        ["b_p_subject_tau.$i" for i in 1:3],
        ["b_p_subject_b_cols_bc.$g.$k" for g in 1:3 for k in 1:3],
    )
    centered_adaptive = only(adaptive_centering_blocks(centered, centered_unc_names))
    @test centered_adaptive.target_c == 1.0
    @test size(centered_adaptive.effects) == (3, 3)
    @test_throws "NON-CENTERED" population_draws(
        centered, ones(1, length(centered_unc_names)), centered_unc_names;
        groups=:subject,
    )
    if RANEF_EFFECT_RUNTIME
        isdir(RANEF_EFFECT_CACHE) || mkpath(RANEF_EFFECT_CACHE)
        centered_problem = StanBlocks.stan_instantiate(
            centered.model;
            path=joinpath(RANEF_EFFECT_CACHE, string(hash(centered_code)) * ".stan"),
        )
        centered_names = StanBlocks.BridgeStan.param_unc_names(centered_problem.model)
        compiled_adaptive = only(adaptive_centering_blocks(centered, centered_names))
        centered_q = zeros(LogDensityProblems.dimension(centered_problem))
        centered_lp, centered_gradient =
            LogDensityProblems.logdensity_and_gradient(centered_problem, centered_q)
        @test compiled_adaptive.target_c == 1.0
        @test size(compiled_adaptive.effects) == (3, 3)
        @test isfinite(centered_lp)
        @test all(isfinite, centered_gradient)
    end

    cv = SBBRMI(partial_brmi; mod=@__MODULE__, cv_groups=[:subject])
    cv_code = BayesianRegressionModels.stan_code(cv)
    @test StanBlocks.stanc_check(cv_code; warn_pedantic=false).ok
    cv_decl = only(d for d in generative_plan(cv).declarations
                   if d.target === :b_p_subject)
    @test cv_decl.family isa StanBlocks.SlicModel
    @test cv_decl.keywords.n_groups === :b_p_subject_n_g

    default_builder = @brm begin
        eta_CL ~ 1 + (1 | p | subject)
        eta_Vc ~ 1 + x + (1 + x | p | subject)
        y ~ Normal(eta_CL + eta_Vc, 1)
    end
    default_sb = SBBRMI(default_builder(df); mod=@__MODULE__)
    default_code = BayesianRegressionModels.stan_code(default_sb)
    default_decl = only(d for d in generative_plan(default_sb).declarations
                        if d.target === :b_p_subject)
    @test default_decl.family === :ranef_correlated_draws
    @test !occursin("brm_ranef_sd", default_code)
    @test occursin("b_p_subject_L ~ lkj_corr_cholesky(1.0);", default_code)
    @test occursin("b_p_subject_tau ~ std_normal();", default_code)

    new_df = merge(df, (; x=reverse(df.x), y=reverse(df.y)))
    replayed = reprocess(partial, new_df)
    @test replayed.data[:subject_idx] == partial.data[:subject_idx]
    @test replayed.data[:n_subject] == partial.data[:n_subject]
    @test BayesianRegressionModels.stan_code(replayed) ==
          BayesianRegressionModels.stan_code(partial)
    @test_throws "unseen level" reprocess(
        partial, merge(new_df, (; subject=[1, 1, 2, 2, 3, 9])))

    reusable = generative_plan(partial_builder, df; mod=@__MODULE__)
    rebuilt = generative_plan(reusable, new_df)
    rebuilt_decl = only(d for d in rebuilt.declarations if d.target === :b_p_subject)
    @test rebuilt_decl.family isa StanBlocks.SlicModel
end

@testset "ranef effect validation fails closed" begin
    wrong_dimension = @brm begin
        eta ~ 1 + (1 + x | p | subject)
        cor(:, p) ~ LKJCholesky(3, 2)
        y ~ Normal(eta, 1)
    end
    @test_throws "does not match" SBBRMI(wrong_dimension(df); mod=@__MODULE__)

    ambiguous_shorthand = @brm begin
        eta ~ 1 + (1 + x | p | subject)
        sd(eta, p) ~ Exponential(1)
        y ~ Normal(eta, 1)
    end
    @test_throws "is ambiguous" SBBRMI(ambiguous_shorthand(df); mod=@__MODULE__)

    # `sd(eta, p)` and `sd(eta, p, x)` reach the same single margin, but they
    # are NOT a duplicate: naming the coefficient is strictly more specific, so
    # the margin statement wins. (This was an error before `:` became the
    # default layer and specificity became the ordering rule.)
    layered_resolution = @brm begin
        eta ~ x + (0 + x | p | subject)
        sd(eta, p) ~ Exponential(1)
        sd(eta, p, x) ~ Exponential(2)
        y ~ Normal(eta, 1)
    end
    layered_code = BayesianRegressionModels.stan_code(
        SBBRMI(layered_resolution(df); mod=@__MODULE__))
    @test occursin(r"b_p_subject_tau ~ brm_vector_prior_[0-9a-f]+", layered_code)

    unknown_margin = @brm begin
        eta ~ 1 + (1 | p | subject)
        sd(eta, p, nope) ~ Exponential(1)
        y ~ Normal(eta, 1)
    end
    @test_throws "matches no random-effect margin" SBBRMI(unknown_margin(df); mod=@__MODULE__)

    unknown_id = @brm begin
        eta ~ 1 + (1 | p | subject)
        sd(:, q) ~ Exponential(1)
        y ~ Normal(eta, 1)
    end
    @test_throws "matches no shared" SBBRMI(unknown_id(df); mod=@__MODULE__)

    half_normal_sd = @brm begin
        eta ~ 1 + (1 | p | subject)
        sd(:, p) ~ Normal(0, 0.5)
        y ~ Normal(eta, 1)
    end
    half_normal_sb = SBBRMI(half_normal_sd(df); mod=@__MODULE__)
    half_normal_code = BayesianRegressionModels.stan_code(half_normal_sb)
    @test occursin(r"b_p_subject_tau ~ brm_vector_prior_[0-9a-f]+", half_normal_code)
    @test StanBlocks.stanc_check(half_normal_code; warn_pedantic=false).ok
    half_normal_decl = only(d for d in generative_plan(half_normal_sb).declarations
                            if d.target === :b_p_subject)
    @test half_normal_decl.family isa StanBlocks.SlicModel

    shifted_normal_sd = @brm begin
        eta ~ 1 + (1 | p | subject)
        sd(:, p) ~ Normal(0.1, 0.5)
        y ~ Normal(eta, 1)
    end
    shifted_code = BayesianRegressionModels.stan_code(
        SBBRMI(shifted_normal_sd(df); mod=@__MODULE__))
    @test occursin("vector<lower=0.0>[n_terms_p_subject] b_p_subject_tau;", shifted_code)
    @test occursin("normal_lpdf(x[1] | arg_1, arg_2)", shifted_code)
    @test StanBlocks.stanc_check(shifted_code; warn_pedantic=false).ok

    affine_sd = @brm begin
        eta ~ 1 + (1 | p | subject)
        sd(:, p) ~ LocationScale(0.1, 0.5, Normal())
        y ~ Normal(eta, 1)
    end
    affine_code = BayesianRegressionModels.stan_code(
        SBBRMI(affine_sd(df); mod=@__MODULE__))
    @test occursin("brm_affine_normal_lpdf(x[1] | arg_2, arg_3, arg_4, arg_5)",
                   affine_code)
    @test StanBlocks.stanc_check(affine_code; warn_pedantic=false).ok

    cauchy_sd = @brm begin
        eta ~ 1 + (1 | p | subject)
        sd(:, p) ~ Cauchy(0, 1)
        y ~ Normal(eta, 1)
    end
    cauchy_sb = SBBRMI(cauchy_sd(df); mod=@__MODULE__)
    cauchy_code = BayesianRegressionModels.stan_code(cauchy_sb)
    @test occursin("vector<lower=0.0>[n_terms_p_subject] b_p_subject_tau;", cauchy_code)
    @test occursin("cauchy_lpdf(x[1] | arg_1, arg_2)", cauchy_code)
    @test StanBlocks.stanc_check(cauchy_code; warn_pedantic=false).ok

    sampled_scale_sd = @brm begin
        log_scale ~ Normal(0, 1)
        eta ~ 1 + (1 | p | subject)
        sd(:, p) ~ Exponential(exp(log_scale))
        y ~ Normal(eta, 1)
    end
    sampled_scale_sb = SBBRMI(sampled_scale_sd(df); mod=@__MODULE__)
    sampled_scale_code = BayesianRegressionModels.stan_code(sampled_scale_sb)
    @test occursin("log_scale ~ normal(0, 1);", sampled_scale_code)
    @test occursin("exponential_lpdf(x[1] | arg_1)", sampled_scale_code)
    @test first(findfirst("log_scale ~ normal", sampled_scale_code)) <
          first(findfirst("b_p_subject_tau ~ brm_vector_prior", sampled_scale_code))
    @test StanBlocks.stanc_check(sampled_scale_code; warn_pedantic=false).ok
    if RANEF_EFFECT_RUNTIME
        problem = StanBlocks.stan_instantiate(
            sampled_scale_sb.model;
            path=joinpath(RANEF_EFFECT_CACHE, string(hash(sampled_scale_code)) * ".stan"))
        q = zeros(LogDensityProblems.dimension(problem))
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test all(isfinite, gradient)
    end
    cauchy_block = only(ranef_blocks(cauchy_sb))
    @test cauchy_block.binding === :b_p_subject
    @test cauchy_block.id === :p
    @test (cauchy_block.n_terms, cauchy_block.n_groups) == (1, 3)
    @test cauchy_block.noncentered
    cauchy_outputs = Dict(o.name => o for o in brm_descriptor(cauchy_sb).outputs)
    @test cauchy_outputs[:b_p_subject].role === :random_effect
    cauchy_replayed = reprocess(cauchy_sb, merge(df, (; y=reverse(df.y))))
    replayed_block = only(ranef_blocks(cauchy_replayed))
    @test (replayed_block.binding, replayed_block.family, replayed_block.group,
           replayed_block.id, replayed_block.n_terms, replayed_block.n_groups,
           replayed_block.z, replayed_block.noncentered) ==
          (cauchy_block.binding, cauchy_block.family, cauchy_block.group,
           cauchy_block.id, cauchy_block.n_terms, cauchy_block.n_groups,
           cauchy_block.z, cauchy_block.noncentered)
    @test BayesianRegressionModels.stan_code(cauchy_replayed) == cauchy_code

    bad_cor_family = @brm begin
        eta ~ 1 + (1 | p | subject)
        cor(:, p) ~ Cauchy(0, 1)
        y ~ Normal(eta, 1)
    end
    @test_throws "expects `LKJCholesky" SBBRMI(bad_cor_family(df); mod=@__MODULE__)

    ambiguous_id = @brm begin
        eta_a ~ 1 + (1 | p | subject)
        eta_b ~ 1 + (1 | p | cat)
        sd(:, p) ~ Exponential(1)
        y ~ Normal(eta_a + eta_b, 1)
    end
    @test_throws "addresses 2 blocks" SBBRMI(ambiguous_id(df); mod=@__MODULE__)
    @test_throws "ambiguous" ranefcoefnames(ambiguous_id(df), :p)

    stratified_df = merge(df, (; stratum=[1, 1, 1, 2, 2, 2]))
    stratified = @brm begin
        eta ~ 1 + (1 | p | gr(subject, by=stratum))
        sd(:, p) ~ Exponential(1)
        y ~ Normal(eta, 1)
    end
    @test_throws "stratified" SBBRMI(stratified(stratified_df); mod=@__MODULE__)

    @test_throws LoadError eval(quote
        @brm begin
            eta ~ 1
            cor(eta, p) ~ LKJCholesky(1, 1)
        end
    end)
end

# `sd(:, ID, coefficient)` — ONE margin across EVERY predictor that slices the
# block — is spellable only under the head-position grammar: the old
# `effect(sd, ID, lp, coef)` shape had no way to leave the predictor slot open
# while naming a coefficient. It sits between the block-wide default and a
# fully explicit margin address, and the same most-specific-wins rule as the
# population surface orders the three.
@testset "`:` predictor claims one margin across every predictor" begin
    margin_names(b) = ranefcoefnames(b, :p)
    rates(b) = BayesianRegressionModels.stan_code(SBBRMI(b; mod=@__MODULE__))

    cross = @brm begin
        eta_Vc ~ 1 + x + (1 + x | p | subject)
        eta_Q  ~ x + (0 + x | p | subject)
        sd(:, p) ~ Exponential(2)
        sd(:, p, x) ~ Exponential(1 / 3)
        y ~ Normal(eta_Vc + eta_Q, 1)
    end
    brmi = cross(df)
    @test margin_names(brmi) == [
        (; predictor=:eta_Vc, coefficient=:Intercept),
        (; predictor=:eta_Vc, coefficient=:x),
        (; predictor=:eta_Q, coefficient=:x),
    ]
    # Block default 1/2 on the intercept margin; BOTH `x` margins take 3.
    cross_code = rates(brmi)
    @test count("exponential_lpdf(x[", cross_code) == 3
    @test all(x -> occursin(string(x), cross_code), (0.5, 3.0))

    # A more specific address overrides the `:`-predictor one on its margin.
    refined = @brm begin
        eta_Vc ~ 1 + x + (1 + x | p | subject)
        eta_Q  ~ x + (0 + x | p | subject)
        sd(:, p) ~ Exponential(2)
        sd(:, p, x) ~ Exponential(1 / 3)
        sd(eta_Q, p, x) ~ Exponential(1 / 5)
        y ~ Normal(eta_Vc + eta_Q, 1)
    end
    refined_code = rates(refined(df))
    @test count("exponential_lpdf(x[", refined_code) == 3
    @test all(x -> occursin(string(x), refined_code), (0.5, 3.0, 5.0))

    # Two addresses of EQUAL specificity reaching one margin have no winner.
    tied = @brm begin
        eta_Vc ~ 1 + x + (1 + x | p | subject)
        eta_Q  ~ x + (0 + x | p | subject)
        sd(:, p, x) ~ Exponential(1 / 3)
        sd(eta_Q, p) ~ Exponential(1 / 5)
        y ~ Normal(eta_Vc + eta_Q, 1)
    end
    @test_throws "equally specific" SBBRMI(tied(df); mod=@__MODULE__)

    # An unmatched coefficient still fails loudly rather than falling back to
    # the block default.
    unknown = @brm begin
        eta_Vc ~ 1 + x + (1 + x | p | subject)
        sd(:, p, nope) ~ Exponential(1)
        y ~ Normal(eta_Vc, 1)
    end
    @test_throws "matches no random-effect margin" SBBRMI(unknown(df); mod=@__MODULE__)
end

# The observed-cQTc shape (Bruno:arv393, snag `ranef-sd-lpdf-el-a190739d`):
# one intercept-only random effect in a NAMED bucket with an explicit
# block-level SD prior. It is the smallest model that pins the generated
# whole-vector density/RNG family and the stable tau coordinate.
@testset "single-term named bucket `(1 | ri | subject)` + `sd(:, ri)` emits" begin
    single = @brm begin
        eta ~ 1 + (1 | ri | subject)
        sd(:, ri) ~ Exponential(10.0)
        y ~ Normal(eta, 1)
    end
    single_sb = SBBRMI(single(df); mod=@__MODULE__)
    single_code = BayesianRegressionModels.stan_code(single_sb)
    @test StanBlocks.stan.transpiles(single_sb.model)
    @test StanBlocks.stanc_check(single_code; warn_pedantic=false).ok
    @test occursin(r"real brm_vector_prior_[0-9a-f]+_lpdf", single_code)
    @test !occursin("brm_ranef_sd", single_code)
    @test occursin(r"b_ri_subject_tau ~ brm_vector_prior_[0-9a-f]+", single_code)
    @test brm_descriptor(single_sb) isa BRMDescriptor
end
