# test/ranef_cv_centered.jl — cv-contagious `(… |ID| g)` buckets + centered
# ranef emission.
#
# Two opt-in emission modes on `SBBRMI`, both keyed by grouping-factor name:
#
#   cv_groups       -- size the per-group draw from `maximum(<g>_idx)` so a
#                      `maybecv(:<g>_idx)` mark flips the block to a
#                      generated-quantities population re-draw. Previously
#                      supported for plain `(… | g)` ranefs only; this suite
#                      pins the `(… |ID| g)` cross-formula bucket path.
#   centered_groups -- emit the centered parameterization (the per-group effect
#                      IS the sampled parameter, covariance as its prior)
#                      instead of the default non-centered standardised draw.
#
# The load-bearing invariant across both: an EMPTY opt-in is byte-for-byte the
# same emission as passing no kwarg at all, and naming a group the model does
# not have is a no-op. (That is a statement about the kwargs, not a promise that
# the default emission never moves — the plain `(… | g)` path was flattened off
# its plate deliberately, which does change it.)

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Normal

stanc_ok(code) = StanBlocks.stanc_check(code; warn_pedantic=false).ok
blocks(code) = code[first(findfirst("data {", code)):end]

# A five-sub-formula model whose random effects all share ONE correlated
# `|p|` bucket — the shape the plain `(… | g)` path cannot express, because
# dropping the ID would discard the cross-parameter RE correlation.
bucket_builder = @brm begin
    eta_CL ~ 0 + male + zage + diseased + (1 | p | subject)
    eta_Vc ~ 0 + male + zage + diseased + (1 | p | subject)
    eta_Q  ~ 0 + diseased + (1 | p | subject)
    eta_ka ~ 0 + (1 | p | subject)
    y ~ Normal(eta_CL + eta_Vc + eta_Q + eta_ka, 1.0)
end

bucket_df(; n_subj=6, per=4) = begin
    n = n_subj * per
    (; subject = repeat(1:n_subj, inner=per),
       male     = Float64.(repeat([0, 1], outer=n ÷ 2)),
       zage     = range(-1.0, 1.0, length=n) |> collect,
       diseased = Float64.(repeat([1, 0], outer=n ÷ 2)),
       y        = range(-0.5, 0.5, length=n) |> collect)
end

# Plain `(… | g)` counterpart, for the centered non-bucket path.
plain_builder = @brm begin
    mu ~ 1 + zage + (1 + zage | subject)
    y ~ Normal(mu, 1.0)
end
intercept_builder = @brm begin
    mu ~ 1 + zage + (1 | subject)
    y ~ Normal(mu, 1.0)
end

@testset "cv-contagious `(… |ID| g)` buckets" begin
    df = bucket_df()
    brmi = bucket_builder(df)

    base = StanBlocks.stan_code(SBBRMI(brmi; mod=@__MODULE__).model)
    sb   = SBBRMI(brmi; mod=@__MODULE__, cv_groups=[:subject])
    code = StanBlocks.stan_code(sb.model)

    @test stanc_ok(code)
    # Opting in changes the emission (it must, or the cv mark cannot bite) ...
    @test code != base
    # ... only in its SIZE EXPRESSION: the SAME `ranef_correlated_draws`
    # submodel is called, sized from `max(subject_idx)` rather than the
    # untainted standalone `n_subject` data scalar. There is no separate `_cv`
    # submodel to keep in step -- that duplication is the thing this path
    # deliberately does not have.
    @test occursin("b_p_subject_n_g = max(subject_idx)", code)
    @test occursin("b_p_subject_z_flat", code)
    # Default and cv emissions differ ONLY in the size expression: same
    # parameter names, same statement shapes.
    @test occursin("b_p_subject_z_flat ~ std_normal()", base)
    @test occursin("b_p_subject_z_flat ~ std_normal()", code)
    # The bucket is still ONE shared block sliced per sub-formula, so the
    # cross-formula correlation the `|p|` ID exists for is preserved.
    @test occursin("cholesky_factor_corr[n_terms_p_subject] b_p_subject_L", code)
    @test count(==("b_p_subject_z_flat"),
                [m.match for m in eachmatch(r"b_p_subject_z_flat", code)]) >= 1
    for t in (:eta_CL, :eta_Vc, :eta_Q, :eta_ka)
        @test occursin("r_$(t)_p_subject = b_p_subject[subject_idx,", code)
    end

    # Untouched groups keep the default sizing.
    @test !occursin("max(subject_idx)", base)
end

@testset "cv mark flips the bucket to a GQ population re-draw" begin
    df = bucket_df()
    sb = SBBRMI(bucket_builder(df); mod=@__MODULE__, cv_groups=[:subject])

    tainted = Dict{Symbol,Any}(sb.data)
    tainted[:subject_idx] = StanBlocks.stan.maybecv(:subject_idx, tainted[:subject_idx])
    code = StanBlocks.stan_code(
        StanBlocks.SlicModel(sb.model.model, tainted, sb.model.mod))

    @test stanc_ok(code)
    params = code[first(findfirst("parameters {", code)):first(findfirst("model {", code))]
    gq     = code[first(findfirst("generated quantities {", code)):end]

    # The whole per-subject block left the parameter space ...
    @test !occursin("b_p_subject_z_flat", params)
    # ... and is re-drawn in generated quantities from the FITTED covariance,
    # which itself stays a parameter. That is the leave-all-out semantics.
    @test occursin("b_p_subject_z_flat = std_normal_vector_rng", gq)
    @test occursin("diag_pre_multiply(b_p_subject_tau, b_p_subject_L)", gq)
    @test occursin("b_p_subject_L", params)
    @test occursin("b_p_subject_tau", params)
    # Population coefficients are unaffected — only the RE is held out.
    @test occursin("pop_eta_CL_beta_pop", params)
end

# The plain `(… | g)` path used to reach cv through its own `ranef_correlated_cv`
# / `ranef_intercept_cv` submodels, which computed `maximum(group_idx)` inside
# the body. Those are deleted; the plain path now takes the size from the call
# site exactly as the bucket path does. This pins that the BEHAVIOUR survived the
# deletion — same GQ flip, same leave-all-out semantics — on both plain shapes.
@testset "cv mark flips a plain `(… | g)` block to a GQ population re-draw" begin
    df = bucket_df()
    for (builder, zname) in ((plain_builder, "r_mu_subject_z_flat"),
                             (intercept_builder, "r_mu_subject_xi"))
        sb = SBBRMI(builder(df); mod=@__MODULE__, cv_groups=[:subject])
        code = StanBlocks.stan_code(sb.model)
        @test stanc_ok(code)
        @test occursin("r_mu_subject_n_g = max(subject_idx)", code)

        tainted = Dict{Symbol,Any}(sb.data)
        tainted[:subject_idx] = StanBlocks.stan.maybecv(:subject_idx, tainted[:subject_idx])
        tcode = StanBlocks.stan_code(
            StanBlocks.SlicModel(sb.model.model, tainted, sb.model.mod))
        @test stanc_ok(tcode)
        params = tcode[first(findfirst("parameters {", tcode)):first(findfirst("model {", tcode))]
        gq     = tcode[first(findfirst("generated quantities {", tcode)):end]
        @test !occursin(zname, params)          # left the parameter space ...
        @test occursin(zname, gq)               # ... re-drawn in GQ
    end
    # The scale hyperparameters stay FITTED parameters in the correlated case --
    # that is what makes the re-draw come from the fitted covariance rather than
    # from the prior. Checked on the correlated shape, which has both.
    sb = SBBRMI(plain_builder(df); mod=@__MODULE__, cv_groups=[:subject])
    tainted = Dict{Symbol,Any}(sb.data)
    tainted[:subject_idx] = StanBlocks.stan.maybecv(:subject_idx, tainted[:subject_idx])
    tcode = StanBlocks.stan_code(
        StanBlocks.SlicModel(sb.model.model, tainted, sb.model.mod))
    params = tcode[first(findfirst("parameters {", tcode)):first(findfirst("model {", tcode))]
    @test occursin("r_mu_subject_tau", params)
    @test occursin("r_mu_subject_L", params)
end

@testset "centered emission — plain `(… | g)`" begin
    df = bucket_df()
    brmi = plain_builder(df)

    base = StanBlocks.stan_code(SBBRMI(brmi; mod=@__MODULE__).model)
    code = StanBlocks.stan_code(
        SBBRMI(brmi; mod=@__MODULE__, centered_groups=[:subject]).model)

    @test stanc_ok(code)
    @test code != base
    # Default is non-centered: a standardised plate draw, scaled downstream.
    @test occursin("~ std_normal()", base)
    # Centered: the per-group effect itself is the parameter, declared as an
    # array of vectors and sampled with ONE vectorised MVN-Cholesky call.
    @test occursin("array[n_subject] vector[n_terms_mu_subject]", code)
    @test occursin("~ multi_normal_cholesky0(", code)
    # ... not a per-group loop of lpdf calls.
    @test !occursin("multi_normal_cholesky(rep_vector", code)
    # The public n_groups x n_terms contract is preserved.
    @test occursin("matrix[n_subject, n_terms_mu_subject]", code)
end

@testset "centered emission — `(1 | g)` intercept fast path" begin
    df = bucket_df()
    brmi = intercept_builder(df)

    # The contrast under test is centered vs the standardised-draw base, so
    # the base builds with totals off (the centered side already excludes
    # totals by design).
    base = StanBlocks.stan_code(SBBRMI(brmi; mod=@__MODULE__, total_groups=()).model)
    code = StanBlocks.stan_code(
        SBBRMI(brmi; mod=@__MODULE__, centered_groups=[:subject]).model)

    @test stanc_ok(code)
    @test code != base
    # Non-centered: xi ~ std_normal(), scaled by exp(log_scale) downstream.
    @test occursin("exp(", base)
    # Centered: the scale moves into the prior.
    @test occursin("normal(0.0, exp(", code)
end

@testset "centered emission — lone `(0 + x | g)` scalar slope" begin
    # Regression: the no-correlation scalar fast path is NCP-only. A centered
    # lone slope must keep the pre-existing `ranef_correlated_centered`
    # routing instead of erroring.
    slope_builder = @brm begin
        mu ~ 1 + zage + (0 + zage | subject)
        y ~ Normal(mu, 1.0)
    end
    df = bucket_df()
    base = StanBlocks.stan_code(SBBRMI(slope_builder(df); mod=@__MODULE__).model)
    code = StanBlocks.stan_code(
        SBBRMI(slope_builder(df); mod=@__MODULE__, centered_groups=[:subject]).model)

    @test stanc_ok(code)
    @test code != base
    # Non-centered: the scalar fast path samples tau/xi directly, no LKJ.
    @test !occursin("lkj_corr_cholesky", base)
    @test occursin("~ std_normal()", base)
    # Centered: the per-group slope itself is the parameter, drawn with ONE
    # vectorised MVN-Cholesky call — the correlated-centered body.
    @test occursin("~ multi_normal_cholesky0(", code)
    @test occursin("ranef_b_matrix", code)
end

@testset "centered emission — `(… |ID| g)` bucket" begin
    df = bucket_df()
    brmi = bucket_builder(df)

    base = StanBlocks.stan_code(SBBRMI(brmi; mod=@__MODULE__).model)
    code = StanBlocks.stan_code(
        SBBRMI(brmi; mod=@__MODULE__, centered_groups=[:subject]).model)

    @test stanc_ok(code)
    @test code != base
    @test occursin("array[n_subject] vector[n_terms_p_subject]", code)
    @test occursin("~ multi_normal_cholesky0(", code)
    # Still one shared bucket, still sliced per sub-formula.
    for t in (:eta_CL, :eta_Vc, :eta_Q, :eta_ka)
        @test occursin("r_$(t)_p_subject = b_p_subject[subject_idx,", code)
    end
end

@testset "default emission is unchanged by the new kwargs" begin
    df = bucket_df()
    for builder in (bucket_builder, plain_builder, intercept_builder)
        brmi = builder(df)
        a = StanBlocks.stan_code(SBBRMI(brmi; mod=@__MODULE__).model)
        b = StanBlocks.stan_code(
            SBBRMI(brmi; mod=@__MODULE__,
                   cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}()).model)
        # An empty opt-in is not merely equivalent — it is the same bytes.
        @test a == b
        # An opt-in naming a group this model does not have is also a no-op.
        @test a == StanBlocks.stan_code(
            SBBRMI(brmi; mod=@__MODULE__, cv_groups=[:nosuchgroup],
                   centered_groups=[:alsonone]).model)
    end
end

@testset "rejected combinations error loudly" begin
    df = bucket_df()
    brmi = bucket_builder(df)

    # cv and centered are mutually exclusive per group: a centered block's
    # sampled parameter cannot carry a cv taint in its size.
    err = try
        SBBRMI(brmi; mod=@__MODULE__, cv_groups=[:subject], centered_groups=[:subject])
        nothing
    catch e; e end
    @test err isa ErrorException
    @test occursin("BOTH", err.msg)
    @test occursin("cv_groups", err.msg)

    # Disjoint groups are fine even when both sets are non-empty.
    @test SBBRMI(brmi; mod=@__MODULE__, cv_groups=[:subject],
                 centered_groups=[:othergroup]) isa SBBRMI
end

@testset "stratified `gr(g, by=b)` supports cv, but not centering" begin
    n_subj, per = 6, 4
    n = n_subj * per
    df = (; subject = repeat(1:n_subj, inner=per),
            arm     = repeat([1, 2], inner=n ÷ 2),
            zage    = collect(range(-1.0, 1.0, length=n)),
            y       = collect(range(-0.5, 0.5, length=n)))
    by_builder = @brm begin
        mu ~ 1 + zage + (1 + zage | gr(subject, by = arm))
        y ~ Normal(mu, 1.0)
    end
    brmi = by_builder(df)

    # It still emits fine when not opted in.
    @test stanc_ok(StanBlocks.stan_code(SBBRMI(brmi; mod=@__MODULE__).model))

    by_sb = SBBRMI(brmi; mod=@__MODULE__, cv_groups=[:subject])
    code = StanBlocks.stan_code(by_sb.model)
    @test stanc_ok(code)
    @test occursin("subject__by__arm_n_g = max(subject__by__arm_idx)", code)
    @test occursin("r_mu_subject__by__arm_b_T_z_g", code)

    # The mark must leave the stratum-level covariance/scales fitted while the
    # group plate re-draws. This is exactly the composition the constrained-
    # matrix plate capability enables.
    tainted = Dict{Symbol,Any}(by_sb.data)
    tainted[:subject__by__arm_idx] =
        StanBlocks.stan.maybecv(:subject__by__arm_idx, tainted[:subject__by__arm_idx])
    marked_code = StanBlocks.stan_code(
        StanBlocks.SlicModel(by_sb.model.model, tainted, by_sb.model.mod))
    @test stanc_ok(marked_code)
    pstart = first(findfirst("parameters {", marked_code))
    params = marked_code[pstart:first(findfirst("model {", marked_code))]
    gq = marked_code[first(findfirst("generated quantities {", marked_code)):end]
    @test !occursin("r_mu_subject__by__arm_b_T_z_g", params)
    @test occursin("r_mu_subject__by__arm_b_T_z_g", gq)
    @test occursin("array[n_strata_subject__by__arm] cholesky_factor_corr", params)
    @test occursin("matrix<lower=0.0>[n_terms_mu_subject__by__arm", params)

    # Centering has no stratified form and no non-centered surface for the cv
    # artifact to ride on.
    for (kw, needle) in ((:centered_groups, "centered parameterization"),)
        err = try
            SBBRMI(brmi; mod=@__MODULE__, kw => [:subject])
            nothing
        catch e; e end
        @test err isa ErrorException
        @test occursin(needle, err.msg)
        @test occursin("gr(subject, by=arm)", err.msg)
    end
end

@testset "stratified `|ID| gr(g, by=b)` bucket supports cv" begin
    n_subj, per = 6, 4
    n = n_subj * per
    df = (; subject = repeat(1:n_subj, inner=per),
            arm     = repeat([1, 2], inner=n ÷ 2),
            zage    = collect(range(-1.0, 1.0, length=n)),
            y       = collect(range(-0.5, 0.5, length=n)))
    bucket_builder = @brm begin
        mu ~ 1 + zage + (1 + zage |p| gr(subject, by = arm))
        y ~ Normal(mu, 1.0)
    end
    brmi = bucket_builder(df)
    sb = SBBRMI(brmi; mod=@__MODULE__, cv_groups=[:subject])
    code = StanBlocks.stan_code(sb.model)
    @test stanc_ok(code)
    @test occursin("b_p_subject__by__arm_n_g = max(subject__by__arm_idx)", code)
    @test occursin("b_p_subject__by__arm_b_T_z_g", code)

    tainted = Dict{Symbol,Any}(sb.data)
    tainted[:subject__by__arm_idx] =
        StanBlocks.stan.maybecv(:subject__by__arm_idx, tainted[:subject__by__arm_idx])
    marked = SBBRMI(sb.parent,
                    StanBlocks.SlicModel(sb.model.model, tainted, sb.model.mod),
                    tainted, sb.preproc)
    marked_code = StanBlocks.stan_code(marked.model)
    @test stanc_ok(marked_code)
    pstart = first(findfirst("parameters {", marked_code))
    params = marked_code[pstart:first(findfirst("model {", marked_code))]
    gq = marked_code[first(findfirst("generated quantities {", marked_code)):end]
    @test !occursin("b_p_subject__by__arm_b_T_z_g", params)
    @test occursin("b_p_subject__by__arm_b_T_z_g", gq)
    @test occursin("array[n_strata_subject__by__arm] cholesky_factor_corr", params)
    @test occursin("matrix<lower=0.0>[n_terms_p_subject__by__arm", params)
end

@testset "runtime — stratified constrained-matrix plate" begin
    n_subj, per = 6, 4
    n = n_subj * per
    df = (; subject = repeat(1:n_subj, inner=per),
            arm     = repeat([1, 2], inner=n ÷ 2),
            zage    = collect(range(-1.0, 1.0, length=n)),
            y       = collect(range(-0.5, 0.5, length=n)))
    builder = @brm begin
        mu ~ 1 + zage + (1 + zage | gr(subject, by = arm))
        y ~ Normal(mu, 1.0)
    end
    brmi = builder(df)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    LDP = StanBlocks.LogDensityProblems
    prob = brm_execute(brm_descriptor(sb), :fit)
    theta = 0.1 .* collect(range(-1.0, 1.0, length=LDP.dimension(prob)))
    lp, grad = LDP.logdensity_and_gradient(prob, theta)
    @test isfinite(lp)
    @test all(isfinite, grad)

    block = only(ranef_blocks(sb))
    @test block.family === :ranef_correlated_by
    @test block.z === :r_mu_subject__by__arm_b_T_z_g

    cv_sb = SBBRMI(brmi; mod=@__MODULE__, cv_groups=[:subject])
    tainted = Dict{Symbol,Any}(cv_sb.data)
    tainted[:subject__by__arm_idx] =
        StanBlocks.stan.maybecv(:subject__by__arm_idx, tainted[:subject__by__arm_idx])
    held_sb = SBBRMI(cv_sb.parent,
                     StanBlocks.SlicModel(cv_sb.model.model, tainted, cv_sb.model.mod),
                     tainted, cv_sb.preproc)
    held = brm_descriptor(held_sb)
    offered = [op.name for op in held.operations]
    @test :predict in offered
    @test :pointwise_loglik in offered
    @test !(:fit in offered)
    outs = Dict(o.name => o for o in held.stan.outputs)
    @test outs[:r_mu_subject__by__arm_b_T_z_g].kind === :generated_quantity
    @test outs[:r_mu_subject__by__arm_L_s_L].kind === :parameter
    @test outs[:r_mu_subject__by__arm_tau_s_tau].kind === :parameter
    @test outs[:r_mu_subject__by__arm_L_s].kind === :transformed_parameter
    @test outs[:r_mu_subject__by__arm_tau_s].kind === :transformed_parameter
end

@testset "runtime — both emissions compile and differentiate" begin
    df = bucket_df()
    brmi = bucket_builder(df)
    LDP = StanBlocks.LogDensityProblems

    fitprob(sb) = brm_execute(brm_descriptor(sb), :fit)
    lp_and_grad(prob) = begin
        n = LDP.dimension(prob)
        theta = 0.1 .* collect(range(-1.0, 1.0, length=n))
        lp = LDP.logdensity(prob, theta)
        (n, lp)
    end

    base_prob = fitprob(SBBRMI(brmi; mod=@__MODULE__))
    n_base, lp_base = lp_and_grad(base_prob)
    @test isfinite(lp_base)

    # Centered is a REPARAMETERIZATION: same posterior, same coordinate count.
    cen_prob = fitprob(SBBRMI(brmi; mod=@__MODULE__, centered_groups=[:subject]))
    n_cen, lp_cen = lp_and_grad(cen_prob)
    @test isfinite(lp_cen)
    @test n_cen == n_base

    # The cv model, UNMARKED, is also just a reparameterization of the sizing —
    # same coordinate count, still a fittable posterior.
    cv_prob = fitprob(SBBRMI(brmi; mod=@__MODULE__, cv_groups=[:subject]))
    n_cv, lp_cv = lp_and_grad(cv_prob)
    @test isfinite(lp_cv)
    @test n_cv == n_base

    # MARKED, the per-subject block leaves the parameter space entirely — and
    # because EVERY observation is a held-out subject's, the likelihood goes
    # with it. So the marked artifact is a leave-ALL-out predictor: it offers
    # `:predict` / `:pointwise_loglik` and NOT `:fit`, exactly the operation
    # gating StanBlocks documents for a fully-held-out model. This is the
    # property the whole cv path exists to produce, and stanc cannot see it.
    sb = SBBRMI(brmi; mod=@__MODULE__, cv_groups=[:subject])
    tainted = Dict{Symbol,Any}(sb.data)
    tainted[:subject_idx] = StanBlocks.stan.maybecv(:subject_idx, tainted[:subject_idx])
    # NOTE: the tainted dict must go in BOTH slots — `generative_plan` rebuilds
    # the SlicModel from `sb.data`, so passing it only in the model is silently
    # discarded and the mark never reaches the trace.
    held_sb = SBBRMI(sb.parent,
                     StanBlocks.SlicModel(sb.model.model, tainted, sb.model.mod),
                     tainted, sb.preproc)
    held_d = brm_descriptor(held_sb)
    offered = [op.name for op in held_d.operations]
    @test :predict in offered
    @test :pointwise_loglik in offered
    @test !(:fit in offered)
    # The population parameters it conditions ON are still there to be fed in;
    # only the subject block became a generated quantity.
    outs = Dict(o.name => o for o in held_d.stan.outputs)
    @test outs[:b_p_subject].kind === :generated_quantity
    @test outs[:b_p_subject_L].kind === :parameter
    @test outs[:b_p_subject_tau].kind === :parameter
    # ... and the unmarked sibling keeps the very same block as a parameter,
    # so the difference is the mark and nothing else.
    unmarked_outs = Dict(o.name => o
                         for o in brm_descriptor(sb).stan.outputs)
    @test unmarked_outs[:b_p_subject].kind !== :generated_quantity
end

@testset "generative_plan snapshots a cv / centered model" begin
    df = bucket_df()
    brmi = bucket_builder(df)

    for kwargs in ((; cv_groups=[:subject]), (; centered_groups=[:subject]))
        sb = SBBRMI(brmi; mod=@__MODULE__, kwargs...)
        plan = generative_plan(sb)
        @test BayesianRegressionModels.stan_code(plan) ==
              BayesianRegressionModels.stan_code(sb)
        @test any(d -> d.role === :observation && d.target === :y, plan.declarations)
    end
end
