using Test, Distributions
using BayesianRegressionModels, StanBlocks
const BRM = BayesianRegressionModels

# Regression gate for snag brm-slicmodel-st-cc033983.
#
# Same shape as test/sbbrmi_cv_stan_code.jl (snag brm-vector-prior-25ce8fc6),
# for the data half: a leave-all-out CV artifact is the emitted `SBBRMI.model`
# called with a `maybecv`-marked group index. That value is a bare
# `StanBlocks.SlicModel`, so consumers could not reach BRM's world-age-safe
# `stan_data` boundary (`stan_data(::SBBRMI)` only accepts the wrapper) and
# had to call `StanBlocks.stan_data` directly. If the model registers a
# generated `brm_vector_prior_*` family (here: a nine-margin correlated block
# with heterogeneous scale priors), that direct trace in the same compiled
# caller dies with "... is missing `lpxf_expr`" even though the hook exists.
# BRM's SLIC-model data method is the missing call-site-independent seam.

data = (; subject=[1, 1, 2, 2],
         y1=[-0.3, -0.1, 0.2, 0.4], y2=[-0.2, 0.0, 0.3, 0.5],
         y3=[-0.1, 0.1, 0.4, 0.6], y4=[0.0, 0.2, 0.5, 0.7],
         y5=[0.1, 0.3, 0.6, 0.8], y6=[0.2, 0.4, 0.7, 0.9],
         y7=[0.3, 0.5, 0.8, 1.0], y8=[0.4, 0.6, 0.9, 1.1],
         y9=[0.5, 0.7, 1.0, 1.2])

builder = @brm begin
    sd(:, p) ~ Exponential(1 / 3)
    sd(loc8, p) ~ Exponential(0.5)
    sd(loc9, p) ~ Exponential(0.5)
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
# the process, while the calling function's world age predates registration.
function build_and_trace_cv_data(builder, data)
    sb = SBBRMI(builder(data); mod=@__MODULE__, cv_groups=[:subject])
    marked = copy(sb.data)
    marked[:subject_idx] = StanBlocks.stan.maybecv(
        :subject_idx, marked[:subject_idx])
    cv_model = sb.model(; subject_idx=marked[:subject_idx])
    direct_error = try
        StanBlocks.stan_data(cv_model)
        nothing
    catch e
        sprint(showerror, e)
    end
    cv_data = BRM.stan_data(cv_model)
    (; cv_model, direct_error, cv_data)
end

@testset "world-age-safe SLIC-model data trace" begin
    infunc = build_and_trace_cv_data(builder, data)
    @test infunc.cv_model isa StanBlocks.SlicModel
    @test !isnothing(infunc.direct_error)
    @test occursin("is missing `lpxf_expr`", infunc.direct_error)
    @test haskey(infunc.cv_data, :subject_idx)
    @test infunc.cv_data[:subject_idx_n] == length(data.subject)

    # At top level (or in any frame compiled after registration), the same
    # direct StanBlocks trace succeeds and is equal to the BRM entry.
    sb_top = SBBRMI(builder(data); mod=@__MODULE__, cv_groups=[:subject])
    marked_top = copy(sb_top.data)
    marked_top[:subject_idx] = StanBlocks.stan.maybecv(
        :subject_idx, marked_top[:subject_idx])
    cv_model_top = sb_top.model(; subject_idx=marked_top[:subject_idx])
    @test BRM.stan_data(cv_model_top) ==
          StanBlocks.stan_data(cv_model_top) ==
          infunc.cv_data
end
