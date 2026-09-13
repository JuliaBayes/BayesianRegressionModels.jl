using Test
using BayesianRegressionModels
using Distributions: Cauchy, Exponential, Normal
using LogExpFunctions: logit

const BRM = BayesianRegressionModels

@testset "backend-neutral BRMI context and simple population design" begin
    df = (; x=Float32[0, 1, 2], y=[1.0, 1.5, 2.2])
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end)(df)

    context = BRM._brm_backend_context(brmi)
    @test context.parent === brmi
    @test context.target_obs == Dict(:mu => :y, :sigma => :y, :y => :y)
    @test context.data[:x] == df.x
    @test context.data[:y] == df.y

    _, rhs = getargs(linear_predictor_op(brmi, :mu), 2)
    design = BRM._brm_simple_population_design(
        :mu, rhs, context.data, context.target_obs[:mu]; required=true)
    @test Tuple(c.label for c in design.columns) == (:Intercept, :x)
    @test Tuple(c.source for c in design.columns) == (nothing, :x)
    @test design.row_source === :x
    @test design.matrix == [1.0 0.0; 1.0 1.0; 1.0 2.0]

    # SBBRMI consumes the same shared column plan, but retains its existing
    # backend-specific optimized program byte-for-byte at the AST boundary.
    emitted = deepcopy(SBBRMI(brmi).model.model)
    expected = quote
        sigma ~ exponential(1.0 ./ 2)
        X_mu = hcat(rep_vector(1.0, num_elements(x)), x)
        pop_mu ~ _popefs_coefs(; X=X_mu)
        y ~ normal_id_glm(X_mu, 0.0, pop_mu, sigma)
        mu = X_mu * pop_mu
    end
    @test Base.remove_linenums!(emitted) == Base.remove_linenums!(expected)
end


@testset "backend-neutral linked population predictor" begin
    df = (; x=[-1.0, 0.5, 2.0], y=[0, 2, 5])
    brmi = (@brm begin
        log(lambda) ~ 1 + x
        y ~ Poisson(lambda)
    end)(df)
    context = BRM._brm_backend_context(brmi)
    predictor = BRM._brm_simple_population_predictor(
        brmi, :lambda, context; required=true)

    @test predictor.name === :lambda
    @test predictor.link_lhs_fn === log
    @test predictor.emitted_name === :log_lambda
    @test predictor.design.target === :lambda
    @test predictor.design.matrix == hcat(ones(3), df.x)

    emitted = sprint(show, SBBRMI(brmi).model.model)
    @test occursin("X_log_lambda", emitted)
    @test occursin("lambda = exp(log_lambda)", emitted)
end


@testset "generic Turing plan reuses backend-neutral predictors" begin
    df = (;
        x=[-1.0, 0.5, 2.0], z=[0.0, 1.0, -0.5],
        trials=[4, 6, 5], y=[1, 4, 2])
    brmi = (@brm begin
        logit(mean) ~ 1 + x
        log(precision) ~ 1 + z
        y ~ BRM.BetaBinomial2(trials, mean, precision)
    end)(df)
    plan = BRM._brm_turing_plan(brmi)

    mean = only(filter(p -> p.predictor.name === :mean, plan.predictors))
    precision = only(filter(p -> p.predictor.name === :precision,
                            plan.predictors))
    @test mean.predictor.link_lhs_fn === logit
    @test precision.predictor.link_lhs_fn === log
    @test mean.design.matrix == hcat(ones(3), df.x)
    @test precision.design.matrix == hcat(ones(3), df.z)
    @test plan.distribution.callable === BRM.BetaBinomial2
    @test plan.distribution.args[1] == BRM._BRMPreparedRef(:trials, :observation)
    @test plan.response == df.y
end


@testset "backend-neutral plain random-intercept geometry" begin
    df = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        subject=["b", "a", "b", "c"],
        y=[0.2, 1.1, -0.4, 0.7],
    )
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x + (1 | subject)
        y ~ Normal(mu, sigma)
    end)(df)
    context = BRM._brm_backend_context(brmi)
    block = only(BRM._brm_simple_random_effect_plans(
        brmi, :mu, context; required=true))
    sb = SBBRMI(brmi)
    plan = BRM._brm_turing_plan(brmi)

    @test block.group === :subject
    @test block.levels == ["a", "b", "c"]
    @test block.indices == [2, 1, 2, 3]
    @test block.indices == sb.data[:subject_idx]
    @test length(block.levels) == sb.data[:n_subject]
    @test plan.design.matrix == hcat(ones(4), df.x)
    @test only(plan.random_effects).indices == block.indices

    sloped = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x + (1 + x | subject)
        y ~ Normal(mu, sigma)
    end)(df)
    sloped_context = BRM._brm_backend_context(sloped)
    sloped_block = only(BRM._brm_simple_random_effect_plans(
        sloped, :mu, sloped_context; required=true))
    @test !sloped_block.intercept_only
    @test Tuple(column.label for column in sloped_block.columns) ==
          (:Intercept, :x)
    @test sloped_block.matrix == hcat(ones(4), df.x)
    @test sloped_block.indices == block.indices

    zero_corr = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x + (1 + x || subject)
        y ~ Normal(mu, sigma)
    end)(df)
    zero_context = BRM._brm_backend_context(zero_corr)
    zero_block = only(BRM._brm_simple_random_effect_plans(
        zero_corr, :mu, zero_context; required=true))
    @test zero_block.zero_correlation
    @test zero_block.matrix == sloped_block.matrix
    @test zero_block.indices == block.indices
end


@testset "backend-neutral fitted population transforms" begin
    df = (;
        x=[1.0, 2.0, 4.0],
        w=[-2.0, 1.0, 5.0],
        q=[10.0, 14.0, 16.0],
        y=[0.2, 1.1, -0.4],
    )
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + zscale(x) + center(w) + standardize(q)
        effect(mu, zscale_x) ~ Normal(0, 0.25)
        y ~ Normal(mu, sigma)
    end)(df)
    context = BRM._brm_backend_context(brmi)
    _, rhs = getargs(linear_predictor_op(brmi, :mu), 2)
    design = BRM._brm_simple_population_design(
        :mu, rhs, context.data, context.target_obs[:mu]; required=true)

    x_mean = sum(df.x) / length(df.x)
    x_scale = sqrt(sum((df.x .- x_mean) .^ 2) / (length(df.x) - 1))
    w_mean = sum(df.w) / length(df.w)
    q_mean = sum(df.q) / length(df.q)
    q_scale = sqrt(sum((df.q .- q_mean) .^ 2) / (length(df.q) - 1))
    expected = hcat(
        ones(length(df.y)),
        (df.x .- x_mean) ./ x_scale,
        df.w .- w_mean,
        (df.q .- q_mean) ./ q_scale,
    )

    @test Tuple(c.label for c in design.columns) ==
          (:Intercept, :zscale_x, :center_w, :standardize_q)
    @test Tuple(c.source for c in design.columns) ==
          (nothing, :x, :w, :q)
    @test Tuple(isnothing(c.preprocess) ? nothing : c.preprocess.kind
                for c in design.columns) ==
          (nothing, :zscale, :center, :standardize)
    @test design.matrix ≈ expected

    sb = SBBRMI(brmi)
    @test sb.data[:zscale_x] ≈ expected[:, 2]
    @test sb.data[:center_w] ≈ expected[:, 3]
    @test sb.data[:standardize_q] ≈ expected[:, 4]
    @test all(isapprox.(sb.preproc[:zscale_x].const_, (x_mean, x_scale)))
    @test sb.preproc[:center_w].const_ == w_mean
    @test all(isapprox.(
        sb.preproc[:standardize_q].const_, (q_mean, q_scale)))

    future = (;
        x=[5.0, 7.0], w=[10.0, 12.0], q=[20.0, 24.0], y=zeros(2))
    replay = reprocess(sb, future)
    @test replay.preproc[:zscale_x].const_ == sb.preproc[:zscale_x].const_
    @test replay.preproc[:center_w].const_ == sb.preproc[:center_w].const_
    @test replay.data[:zscale_x] ≈ (future.x .- x_mean) ./ x_scale
    @test replay.data[:center_w] ≈ future.w .- w_mean
    @test replay.data[:standardize_q] ≈ (future.q .- q_mean) ./ q_scale
end


@testset "backend-neutral continuous interactions" begin
    df = (;
        x=[1.0, 2.0, 4.0],
        w=[-2.0, 1.0, 5.0],
        y=[0.2, 1.1, -0.4],
    )
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + zscale(x) & w
        effect(mu, int_zscale_x_x_w) ~ Normal(0.5, 0.25)
        y ~ Normal(mu, sigma)
    end)(df)
    context = BRM._brm_backend_context(brmi)
    _, rhs = getargs(linear_predictor_op(brmi, :mu), 2)
    design = BRM._brm_simple_population_design(
        :mu, rhs, context.data, context.target_obs[:mu]; required=true)

    fitted = BRM._brm_fit_zscale(df.x)
    scaled_x = BRM._brm_apply_zscale(fitted, df.x)
    expected_interaction = scaled_x .* df.w
    @test Tuple(c.label for c in design.columns) ==
          (:Intercept, :int_zscale_x_x_w)
    @test design.matrix ≈ hcat(ones(3), expected_interaction)
    interaction = design.columns[2].preprocess
    @test interaction.kind === :interaction
    @test interaction.raw_ref == (:zscale_x, :w)
    @test length(interaction.dependencies) == 1
    @test only(interaction.dependencies).label === :zscale_x

    sb = SBBRMI(brmi)
    @test sb.data[:zscale_x] ≈ scaled_x
    @test sb.data[:int_zscale_x_x_w] ≈ expected_interaction
    @test sb.preproc[:zscale_x].kind === :zscale
    @test sb.preproc[:int_zscale_x_x_w].kind === :interaction

    future = (; x=[5.0, 7.0], w=[10.0, 12.0], y=zeros(2))
    replay = reprocess(sb, future)
    future_scaled = BRM._brm_apply_zscale(fitted, future.x)
    @test replay.data[:zscale_x] ≈ future_scaled
    @test replay.data[:int_zscale_x_x_w] ≈ future_scaled .* future.w
end


@testset "backend-neutral categorical treatment contrasts" begin
    df = (;
        g=[1, 2, 3, 1, 2, 3],
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5],
    )
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + g + x
        effect(mu, g) ~ Normal(0.5, 0.25)
        y ~ Normal(mu, sigma)
    end)(df)
    context = BRM._brm_backend_context(brmi)
    _, rhs = getargs(linear_predictor_op(brmi, :mu), 2)
    design = BRM._brm_simple_population_design(
        :mu, rhs, context.data, context.target_obs[:mu]; required=true)

    @test Tuple(c.label for c in design.columns) ==
          (:Intercept, :g_lvl_2, :g_lvl_3, :x)
    @test Tuple(c.effect_addresses for c in design.columns) ==
          ((:Intercept,), (:g,), (:g,), (:x,))
    @test design.matrix == hcat(
        ones(6),
        Float64.(df.g .== 2),
        Float64.(df.g .== 3),
        df.x,
    )
    @test design.columns[2].preprocess.kind === :population_factor_dummy
    @test design.columns[2].preprocess.const_.levels == [1, 2, 3]

    overrides = BRM._brm_simple_population_effect_overrides(brmi, design)
    location, scale = BRM._brm_materialize_normal_effect_priors(
        overrides, length(design.columns))
    @test location == [0.0, 0.5, 0.5, 0.0]
    @test scale == [1.0, 0.25, 0.25, 1.0]

    # The StanBlocks backend owns a predictor-qualified `cat_mu_g_beta`
    # parameter block, but shares the same ordered level-coding primitive.
    sb = SBBRMI(brmi)
    @test sb.data[:g_idx] == df.g
    @test sb.preproc[:g_idx].const_ == [1, 2, 3]

    declared = BRM.CA.categorical(
        [10, 20, 30, 10, 20, 30]; levels=[30, 20, 10])
    declared_brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + g
        y ~ Normal(mu, sigma)
    end)((; g=declared, y=zeros(6)))
    declared_context = BRM._brm_backend_context(declared_brmi)
    _, declared_rhs = getargs(
        linear_predictor_op(declared_brmi, :mu), 2)
    declared_design = BRM._brm_simple_population_design(
        :mu, declared_rhs, declared_context.data,
        declared_context.target_obs[:mu]; required=true)
    @test declared_design.matrix == hcat(
        ones(6), Float64.(declared .== 20), Float64.(declared .== 10))
    @test declared_design.columns[2].preprocess.const_.levels == [30, 20, 10]

    declared_sb = SBBRMI(declared_brmi)
    @test declared_sb.data[:g_idx] == [3, 2, 1, 3, 2, 1]
    @test declared_sb.preproc[:g_idx].const_ == [30, 20, 10]

    reffed = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + factor(g; ref=3)
        effect(mu, g) ~ Normal(-0.5, 0.2)
        y ~ Normal(mu, sigma)
    end)(df)
    reffed_context = BRM._brm_backend_context(reffed)
    _, reffed_rhs = getargs(linear_predictor_op(reffed, :mu), 2)
    reffed_design = BRM._brm_simple_population_design(
        :mu, reffed_rhs, reffed_context.data,
        reffed_context.target_obs[:mu]; required=true)
    @test Tuple(c.label for c in reffed_design.columns) ==
          (:Intercept, :g__ref_3_lvl_2, :g__ref_3_lvl_3)
    @test reffed_design.matrix == hcat(
        ones(6), Float64.(df.g .== 2), Float64.(df.g .== 1))
    @test reffed_design.columns[2].effect_addresses == (:g__ref_3, :g)
    reffed_overrides = BRM._brm_simple_population_effect_overrides(
        reffed, reffed_design)
    @test BRM._brm_materialize_normal_effect_priors(
        reffed_overrides, 3) == ([0.0, -0.5, -0.5], [1.0, 0.2, 0.2])

    ambiguous = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + factor(g; ref=2) + factor(g; ref=3)
        effect(mu, g) ~ Normal(0, 0.5)
        y ~ Normal(mu, sigma)
    end)(df)
    ambiguous_context = BRM._brm_backend_context(ambiguous)
    _, ambiguous_rhs = getargs(linear_predictor_op(ambiguous, :mu), 2)
    ambiguous_design = BRM._brm_simple_population_design(
        :mu, ambiguous_rhs, ambiguous_context.data,
        ambiguous_context.target_obs[:mu]; required=true)
    @test_throws "ambiguously names categorical contrast blocks" begin
        BRM._brm_simple_population_effect_overrides(
            ambiguous, ambiguous_design)
    end
end


@testset "backend-neutral categorical interactions" begin
    df = (;
        x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        g=[1, 2, 3, 1, 2, 3],
        h=[1, 1, 2, 2, 1, 2],
        y=zeros(6),
    )
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x & g + g & h
        effect(mu, int_x_x_g_lvl_2) ~ Normal(0.5, 0.25)
        y ~ Normal(mu, sigma)
    end)(df)
    context = BRM._brm_backend_context(brmi)
    _, rhs = getargs(linear_predictor_op(brmi, :mu), 2)
    design = BRM._brm_simple_population_design(
        :mu, rhs, context.data, context.target_obs[:mu]; required=true)
    expected_labels = (
        :Intercept,
        :int_x_x_g_lvl_2,
        :int_x_x_g_lvl_3,
        :int_g_lvl_2_x_h_lvl_2,
        :int_g_lvl_3_x_h_lvl_2,
    )
    expected = hcat(
        ones(6),
        df.x .* (df.g .== 2),
        df.x .* (df.g .== 3),
        (df.g .== 2) .* (df.h .== 2),
        (df.g .== 3) .* (df.h .== 2),
    )
    @test Tuple(c.label for c in design.columns) == expected_labels
    @test design.matrix == expected

    sb = SBBRMI(brmi)
    @test sb.data[:int_x_x_g_lvl_2] == expected[:, 2]
    @test sb.data[:int_g_lvl_3_x_h_lvl_2] == expected[:, 5]
    @test sb.preproc[:g_lvl_2].kind === :population_factor_dummy
    @test sb.preproc[:int_x_x_g_lvl_2].kind === :interaction

    future = (;
        x=[10.0, 20.0, 30.0], g=[3, 2, 1], h=[2, 1, 2], y=zeros(3))
    replay = reprocess(sb, future)
    @test replay.data[:int_x_x_g_lvl_2] ==
          future.x .* (future.g .== 2)
    @test replay.data[:int_g_lvl_3_x_h_lvl_2] ==
          (future.g .== 3) .* (future.h .== 2)
end


@testset "backend-neutral pure expressions and fixed offsets" begin
    df = (;
        x=[1.0, 2.0, 4.0],
        exposure=[2.0, 4.0, 8.0],
        y=[0, 2, 5],
    )
    brmi = (@brm begin
        log_rate ~ 1 + log(x) + offset(log(exposure))
        y ~ Poisson(exp(log_rate))
    end)(df)
    context = BRM._brm_backend_context(brmi)
    _, rhs = getargs(linear_predictor_op(brmi, :log_rate), 2)
    design = BRM._brm_simple_population_design(
        :log_rate, rhs, context.data, context.target_obs[:log_rate];
        required=true)

    expression_label = design.columns[2].label
    @test startswith(String(expression_label), "log_expr_")
    @test Tuple(c.label for c in design.columns) ==
          (:Intercept, expression_label)
    @test design.matrix == hcat(ones(3), log.(df.x))
    @test length(design.fixed_terms) == 1
    @test only(design.fixed_terms).source === :exposure
    @test design.fixed == log.(df.exposure)

    # Pure coefficient-bearing data expressions use the exact same key and
    # replay record in StanBlocks. Offsets retain StanBlocks' direct
    # coefficient-one expression while Turing consumes the shared fixed vector.
    sb = SBBRMI(brmi)
    @test expression_label in keys(sb.data)
    @test sb.data[expression_label] == log.(df.x)
    @test sb.preproc[expression_label].kind === :protect
    @test popcoefnames(brmi, :log_rate) == [:Intercept, expression_label]

    future = (; x=[8.0, 16.0], exposure=[3.0, 9.0], y=zeros(Int, 2))
    replay = reprocess(sb, future)
    @test replay.data[expression_label] == log.(future.x)
    @test replay.data[:exposure] == future.exposure
end


@testset "backend-neutral population effect-prior plan" begin
    df = (; x=[-1.0, 0.0, 1.0], y=[0.1, 0.2, 0.3])
    brmi = (@brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        effect(:, :) ~ Normal(-1, 3)
        effect(mu, Intercept) ~ Normal(log(2), 0.5)
        effect(mu, x) ~ Normal(0, 0.25)
        y ~ Normal(mu, sigma)
    end)(df)
    context = BRM._brm_backend_context(brmi)
    _, rhs = getargs(linear_predictor_op(brmi, :mu), 2)
    design = BRM._brm_simple_population_design(
        :mu, rhs, context.data, context.target_obs[:mu]; required=true)

    overrides = BRM._brm_simple_population_effect_overrides(brmi, design)
    location, scale = BRM._brm_materialize_normal_effect_priors(
        overrides, length(design.columns))
    @test location == [log(2), 0.0]
    @test scale == [0.5, 0.25]
    @test BRM._brm_population_effect_operation_keys(brmi) == Set((
        Symbol("__effect__:__:"),
        Symbol("__effect__mu__Intercept"),
        Symbol("__effect__mu__x"),
    ))

    layered = (@brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        effect(:, x) ~ Normal(0, 0.4)
        effect(mu, x) ~ Normal(1, 0.2)
        y ~ Normal(mu, sigma)
    end)(df)
    layered_context = BRM._brm_backend_context(layered)
    _, layered_rhs = getargs(linear_predictor_op(layered, :mu), 2)
    layered_design = BRM._brm_simple_population_design(
        :mu, layered_rhs, layered_context.data,
        layered_context.target_obs[:mu]; required=true)
    layered_overrides = BRM._brm_simple_population_effect_overrides(
        layered, layered_design)
    @test BRM._brm_materialize_normal_effect_priors(
        layered_overrides, 2) == ([0.0, 1.0], [1.0, 0.2])

    tied = (@brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        effect(:, x) ~ Normal(0, 0.4)
        effect(mu, :) ~ Normal(0, 0.2)
        y ~ Normal(mu, sigma)
    end)(df)
    tied_context = BRM._brm_backend_context(tied)
    _, tied_rhs = getargs(linear_predictor_op(tied, :mu), 2)
    tied_design = BRM._brm_simple_population_design(
        :mu, tied_rhs, tied_context.data, tied_context.target_obs[:mu];
        required=true)
    @test_throws "equally specific" begin
        BRM._brm_simple_population_effect_overrides(tied, tied_design)
    end

    wrong_family = (@brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        effect(mu, x) ~ Cauchy(0, 1)
        y ~ Normal(mu, sigma)
    end)(df)
    wrong_context = BRM._brm_backend_context(wrong_family)
    _, wrong_rhs = getargs(linear_predictor_op(wrong_family, :mu), 2)
    wrong_design = BRM._brm_simple_population_design(
        :mu, wrong_rhs, wrong_context.data, wrong_context.target_obs[:mu];
        required=true)
    generic_priors = BRM._brm_simple_population_effect_overrides(
        wrong_family, wrong_design)
    @test isnothing(generic_priors[1])
    @test getf(generic_priors[2]) === Cauchy
end

@testset "shared population design is narrow and loud" begin
    intercept = (@brm begin
        sigma ~ Exponential(1)
        mu ~ 1
        y ~ Normal(mu, sigma)
    end)((; y=[0.0, 1.0, 2.0]))
    context = BRM._brm_backend_context(intercept)
    _, rhs = getargs(linear_predictor_op(intercept, :mu), 2)
    design = BRM._brm_simple_population_design(
        :mu, rhs, context.data, context.target_obs[:mu]; required=true)
    @test design.row_source === :y
    @test design.matrix == ones(3, 1)

    unsupported = (@brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + g
        y ~ Normal(mu, sigma)
    end)((; g=["a", "b", "a"], y=zeros(3)))
    cat_context = BRM._brm_backend_context(unsupported)
    _, cat_rhs = getargs(linear_predictor_op(unsupported, :mu), 2)
    @test isnothing(BRM._brm_simple_population_design(
        :mu, cat_rhs, cat_context.data, cat_context.target_obs[:mu]))
    @test_throws "supports `1`, continuous raw-data columns" begin
        BRM._brm_simple_population_design(
            :mu, cat_rhs, cat_context.data, cat_context.target_obs[:mu];
            required=true)
    end

    # A parameter-owning term head (`mo`, `me`, `s`, ...) is NOT a materialisable
    # data expression — the simple design must DECLINE it so the richer emitter
    # lowers it as a parameter, never `broadcast(mo, diet)` into a bare
    # `MethodError: no method matching mo(::Int64)` (snag `kernel-mo-term-i`).
    monotonic = (@brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + mo(diet)
        y ~ Normal(mu, sigma)
    end)((; diet=[1, 2, 1, 3], y=zeros(4)))
    mo_context = BRM._brm_backend_context(monotonic)
    _, mo_rhs = getargs(linear_predictor_op(monotonic, :mu), 2)
    @test isnothing(BRM._brm_simple_population_design(
        :mu, mo_rhs, mo_context.data, mo_context.target_obs[:mu]))
    @test_throws "supports `1`, continuous raw-data columns" begin
        BRM._brm_simple_population_design(
            :mu, mo_rhs, mo_context.data, mo_context.target_obs[:mu];
            required=true)
    end

    missing_x = Union{Missing,Float64}[1.0, missing]
    @test_throws "never silently drops rows" begin
        broken = (@brm begin
            sigma ~ Exponential(1)
            mu ~ 1 + x
            y ~ Normal(mu, sigma)
        end)((; x=missing_x, y=zeros(2)))
        BRM._brm_backend_context(broken)
    end
end
