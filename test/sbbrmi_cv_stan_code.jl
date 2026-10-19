using Test, Distributions
using BayesianRegressionModels, StanBlocks
const BRM = BayesianRegressionModels

# Regression gate for snag brm-vector-prior-25ce8fc6.
#
# A leave-all-out CV artifact is the emitted `SBBRMI.model` called with a
# `maybecv`-marked group index. That value is a bare `StanBlocks.SlicModel`,
# so BRM provides a `stan_code` method for it as well as for `SBBRMI`.
#
# The snag was filed while lowering installed each generated
# `brm_vector_prior_*` family (here: a nine-margin correlated block with
# a composed scale margin) through `Core.eval`: a direct
# `StanBlocks.stan_code` in the constructing frame could not see the fresh
# methods and died with "... is missing `lpxf_expr`". Generated families are
# now `StanBlocks.ValueFamily` values and construction installs no methods
# (d10effe), so the direct trace and BRM's entry must both succeed in the
# constructing frame and agree with a top-level trace.

data = (; subject=[1, 1, 2, 2],
          y1=[-0.3, -0.1, 0.2, 0.4], y2=[-0.2, 0.0, 0.3, 0.5],
          y3=[-0.1, 0.1, 0.4, 0.6], y4=[0.0, 0.2, 0.5, 0.7],
          y5=[0.1, 0.3, 0.6, 0.8], y6=[0.2, 0.4, 0.7, 0.9],
          y7=[0.3, 0.5, 0.8, 1.0], y8=[0.4, 0.6, 0.9, 1.1],
          y9=[0.5, 0.7, 1.0, 1.2])

builder = @brm begin
    sd(:, p) ~ Exponential(1 / 3)
    sd(loc8, p) ~ Exponential(0.5)
    # One composed margin keeps the generated family: builtin margins alone,
    # even of differing families, lower to native statements.
    sd(loc9, p) ~ truncated(TDist(3); lower=0.0)
    cor(:, p) ~ LKJCholesky(9, 2.0)
    loc1 ~ 1 + (1 | p | subject)
    loc2 ~ 1 + (1 | p | subject)
    loc3 ~ 1 + (1 | p | subject)
    loc4 ~ 1 + (1 | p | subject)
    loc5 ~ 1 + (1 | p | subject)
    loc6 ~ 1 + (1 | p | subject)
    loc7 ~ 1 + (1 | p | subject)
    loc8 ~ 1 + (1 | p | subject)
    loc9 ~ 1 + (1 | p | subject)
    y1 ~ Normal(loc1, 1.0)
    y2 ~ Normal(loc2, 1.0)
    y3 ~ Normal(loc3, 1.0)
    y4 ~ Normal(loc4, 1.0)
    y5 ~ Normal(loc5, 1.0)
    y6 ~ Normal(loc6, 1.0)
    y7 ~ Normal(loc7, 1.0)
    y8 ~ Normal(loc8, 1.0)
    y9 ~ Normal(loc9, 1.0)
end

# Order is load-bearing: this must be the first trace of this family shape in
# the process. A prior top-level trace populates `_SB_VECTOR_PRIOR_CACHE` and
# would miss a cold-construction regression.
function build_and_trace_cv(builder, data)
    sb = SBBRMI(builder(data); mod=@__MODULE__, cv_groups=[:subject])
    marked = copy(sb.data)
    marked[:subject_idx] = StanBlocks.stan.maybecv(
        :subject_idx, marked[:subject_idx])
    cv_model = sb.model(; subject_idx=marked[:subject_idx])
    direct_source = StanBlocks.stan_code(cv_model)
    cv_source = BRM.stan_code(cv_model)
    (; cv_model, direct_source, cv_source)
end

@testset "call-site-independent SLIC-model trace" begin
    infunc = build_and_trace_cv(builder, data)
    @test infunc.cv_model isa StanBlocks.SlicModel
    @test infunc.direct_source == infunc.cv_source
    @test occursin("brm_vector_prior_", infunc.cv_source)
    @test occursin("subject_idx", infunc.cv_source)

    # A top-level trace of the same model is byte-identical.
    sb_top = SBBRMI(builder(data); mod=@__MODULE__, cv_groups=[:subject])
    marked_top = copy(sb_top.data)
    marked_top[:subject_idx] = StanBlocks.stan.maybecv(
        :subject_idx, marked_top[:subject_idx])
    cv_model_top = sb_top.model(; subject_idx=marked_top[:subject_idx])
    @test BRM.stan_code(cv_model_top) ==
          StanBlocks.stan_code(cv_model_top) ==
          infunc.cv_source
end
