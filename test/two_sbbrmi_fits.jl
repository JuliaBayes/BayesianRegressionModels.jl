using Test, Distributions
using BayesianRegressionModels, StanBlocks
const BRM = BayesianRegressionModels

# Regression gate for snag two-sbbrmi-fits-f2beca06 (retitled during handling:
# NOT cross-fit — see below).
#
# Two SBBRMI fits in one session, same likelihood, different priors: fit B is
# all Normal/Exponential, fit A mixes a LocationScale-T Intercept and a
# truncated-T sigma with Normal commons. Fit A's population prior is not
# all-Normal, so lowering registers a composed `brm_vector_prior_*` family
# (`@deffun` triad + `lpxf_expr` hook) via `Core.eval` DURING `SBBRMI`
# construction, while fit B takes the direct `_popefs_normal` path and
# registers nothing.
#
# The reported "cross-fit collision" is refuted: the failure is world age,
# not shared state. Tracing fit A via direct StanBlocks entries
# (`StanBlocks.stan_code(sb.model)` / `StanBlocks.transpiles(sb.model)`) in
# the SAME compiled frame as its build resolves methods at that frame's
# world age, so the fresh hooks are invisible and tracing dies with
# "`brm_vector_prior_*` is missing `lpxf_expr`" — with or without fit B
# first, and an identical top-level trace succeeds. Fit B merely happened to
# need no new family, which is why it always passed and fit A always failed.
#
# The BRM trace entries (`stan_code`, `stan_model`, `stan_instantiate`, and
# the `transpiles`/`compiles` predicates) re-enter the compiler in the
# current world via `Base.invokelatest` and are call-site independent. This
# file pins the two-fit shape through them. Direct `StanBlocks.*(sb.model)`
# from inside such a frame remains a world-age error by Julia semantics (see
# test/totals_call_site.jl); it is documented, not asserted here.
#
# Order is load-bearing: the in-function fits below MUST run before any
# top-level trace of these shapes in this process. A prior top-level trace
# populates `_SB_VECTOR_PRIOR_CACHE`, so no new family is registered and the
# staleness the wrapped entries guard against never manifests.

df = (; x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5])

builderB = @brm begin
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, x) ~ Normal(0, 2)
    sigma ~ Exponential(1)
    mu ~ 1 + x
    y ~ Normal(mu, sigma)
end

builderA = @brm begin
    effect(mu, Intercept) ~ LocationScale(0, 5, TDist(3))
    effect(mu, x) ~ Normal(0, 2)
    sigma ~ truncated(TDist(3); lower=0.0)
    mu ~ 1 + x
    y ~ Normal(mu, sigma)
end

# Build + trace both fits entirely inside compiled frames: the FIRST lowering
# of these shapes in the process. Returns the traced sources for the
# call-site byte-identity check below.
function build_and_trace_two_fits(builderB, builderA, data)
    sbB = SBBRMI(builderB(data); mod=@__MODULE__)
    @test BRM.transpiles(sbB)
    srcB = BRM.stan_code(sbB)
    sbA = SBBRMI(builderA(data); mod=@__MODULE__)
    @test BRM.transpiles(sbA)
    srcA = BRM.stan_code(sbA)
    (; sbB, srcB, sbA, srcA)
end

@testset "two SBBRMI fits are call-site independent" begin
    infunc = build_and_trace_two_fits(builderB, builderA, df)
    @test infunc.srcB isa String
    @test !isempty(infunc.srcB)
    @test !occursin("brm_vector_prior_", infunc.srcB)
    @test infunc.srcA isa String
    @test !isempty(infunc.srcA)
    @test occursin("brm_vector_prior_", infunc.srcA)
    # Compile entry exists on the SBBRMI (same invokelatest pattern as the
    # trace entries; the C++ compile itself is covered per-shape by the
    # BridgeStan integration tests).
    @test applicable(BRM.compiles, infunc.sbA)

    # Same builders + same data at top level: identical Stan source.
    sbB_top = SBBRMI(builderB(df); mod=@__MODULE__)
    @test BRM.stan_code(sbB_top) == infunc.srcB
    sbA_top = SBBRMI(builderA(df); mod=@__MODULE__)
    @test BRM.stan_code(sbA_top) == infunc.srcA
end

# The reporter's exact custom-family shape: one nine-margin correlated random-
# effect block, with a common Exponential(1/3) scale prior and two public
# margin overrides at Exponential(0.5). The heterogeneous scale vector creates
# a fresh `brm_vector_prior_*` family while this compiled caller is already
# running.
heterogeneous_data = (;
    subject = [1, 1, 2, 2],
    y1 = [-0.3, -0.1, 0.2, 0.4],
    y2 = [-0.2,  0.0, 0.3, 0.5],
    y3 = [-0.1,  0.1, 0.4, 0.6],
    y4 = [ 0.0,  0.2, 0.5, 0.7],
    y5 = [ 0.1,  0.3, 0.6, 0.8],
    y6 = [ 0.2,  0.4, 0.7, 0.9],
    y7 = [ 0.3,  0.5, 0.8, 1.0],
    y8 = [ 0.4,  0.6, 0.9, 1.1],
    y9 = [ 0.5,  0.7, 1.0, 1.2],
)

heterogeneous_builder = @brm begin
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

function build_and_materialize_data(builder, data)
    sb = SBBRMI(builder(data); mod=@__MODULE__)
    src = BRM.stan_code(sb)
    direct_error = try
        StanBlocks.stan_data(sb.model)
        nothing
    catch e
        sprint(showerror, e)
    end
    prepared = BRM.stan_data(sb)
    (; sb, src, direct_error, prepared)
end

@testset "heterogeneous correlated-block data is call-site independent" begin
    infunc = build_and_materialize_data(heterogeneous_builder, heterogeneous_data)
    @test occursin("brm_vector_prior_", infunc.src)
    @test !isnothing(infunc.direct_error)
    @test occursin("is missing `lpxf_expr`", infunc.direct_error)
    @test infunc.prepared == StanBlocks.stan_data(infunc.sb.model)
    @test infunc.prepared == BRM.stan_data(infunc.sb)
end
