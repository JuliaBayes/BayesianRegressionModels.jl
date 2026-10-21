# test/backend_source.jl — `brm_source(backend)` is the BRMI the backend was
# built from, exactly as passed (snag `backend-parent-i-d62d9ac7`, user design
# decision `0qcv4yj`).
#
# `parent(backend)` is the model the backend lowered: the source after BRM
# resolved its relative prior addresses (`optional(...)` priors,
# grouping-column `sd`/`cor`/`ranef` addresses), so `parent(backend) === brmi`
# fails for exactly those models. `constant(...)` declarations never change
# `parent`. Two backends built from one BRMI have `===` parents.
#
# Run: julia --startup-file=no --project=test test/backend_source.jl

using Test
using BayesianRegressionModels, StanBlocks, ReactiveKernels, ReactiveKernelsPPL
using Distributions: Exponential, Normal
using Statistics: mean
const BRM = BayesianRegressionModels

const DF = (;
    x=[-1.0, -0.6, -0.2, 0.2, 0.6, 1.0, -0.9, -0.3, 0.1, 0.5, 0.8, 1.2],
    subject=repeat(1:4, inner=3),
    y=[0.3, 0.9, 1.6, 0.4, 1.2, 2.0, 0.1, 0.8, 1.7, 0.6, 1.1, 1.5],
)

# label => (BRMI, whether prior-address resolution leaves it unchanged)
const MODELS = [
    "plain" => (@brm(DF, begin
        mu ~ 1 + x + (1 | p | subject)
        effect(mu, Intercept) ~ Normal(0.0, 1.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), true),
    "constant" => (@brm(DF, begin
        loc = constant(mean(y))
        mu ~ 1 + x + (1 | p | subject)
        effect(mu, Intercept) ~ Normal(loc, 1.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), true),
    "optional" => (@brm(DF, begin
        mu ~ 1 + x + (1 | p | subject)
        effect(mu, Intercept) ~ Normal(0.0, 1.0)
        optional(effect(mu, weight)) ~ Normal(0.0, 0.1)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), false),
    "constant + optional" => (@brm(DF, begin
        loc = constant(mean(y))
        mu ~ 1 + x + (1 | p | subject)
        effect(mu, Intercept) ~ Normal(loc, 1.0)
        optional(effect(mu, weight)) ~ Normal(0.0, 0.1)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), false),
    "grouping-column sd" => (@brm(DF, begin
        mu ~ 1 + x + (1 | subject)
        effect(mu, Intercept) ~ Normal(0.0, 1.0)
        sd(mu, subject) ~ Exponential(1)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), false),
]

@testset "brm_source is exported" begin
    @test Base.isexported(BayesianRegressionModels, :brm_source)
end

@testset "brm_source is the passed BRMI: $label" for (label, (brmi, unchanged)) in MODELS
    sb = SBBRMI(brmi; mod=@__MODULE__)
    rk = RKBRMI(brmi)
    @test brm_source(sb) === brmi
    @test brm_source(rk) === brmi
    # `parent` is the lowered model; it equals the source only when nothing
    # needed resolving.
    @test (parent(sb) === brmi) == unchanged
    @test (parent(rk) === brmi) == unchanged
    @test parent(sb) === BRM._brm_resolve_prior_addresses(brmi)
    @test parent(rk) === parent(sb)
end

@testset "a retained RK build keeps the passed BRMI" begin
    brmi = first(MODELS[4].second)
    artifact = BRM.emit_rk_artifact(brmi; case_id="backend-source", held_out=())
    built = build_kernel(rk_translate_artifact(artifact))
    backend = RKBRMI(brmi, artifact.plan, built)
    @test brm_source(backend) === brmi
    @test parent(backend) === parent(SBBRMI(brmi; mod=@__MODULE__))
    @test backend.plan === artifact.plan && backend.model === built
end

@testset "derived backends keep the source with their parent" begin
    brmi = first(MODELS[4].second)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    # Frozen replays keep the fit's parent, so they keep its source.
    replay = reprocess(sb, DF)
    @test parent(replay) === parent(sb)
    @test brm_source(replay) === brmi
    rk = RKBRMI(brmi)
    rk_replay = reprocess(rk, DF)
    @test parent(rk_replay) === parent(rk)
    @test brm_source(rk_replay) === brmi
    # A fresh-constants RK replay rebuilds the model on the new data: its
    # source is that rebound model, whose resolution is its parent.
    fresh = reprocess(rk, DF; freeze_constants=false)
    @test brm_source(fresh) !== brmi
    @test parent(fresh) === BRM._brm_resolve_prior_addresses(brm_source(fresh))
end

@testset "an omitted response's inactive |ID| margins keep the source" begin
    data = (; DF.x, DF.subject, ya=DF.y)
    brmi = @brm data begin
        a ~ 1 + x + (1 | p | subject)
        b ~ 1 + (1 | p | subject)
        optional(effect(a, weight)) ~ Normal(0.0, 0.1)
        ya ~ Normal(a, 0.5)
        yb ~ Normal(b, 0.5)
    end
    sb = @test_logs (:warn, r"`yb` bind\(s\) no data column") match_mode=:any SBBRMI(
        brmi; mod=@__MODULE__, total_groups=())
    # Positive control: the constructor's integrated-margin rebuild ran.
    @test haskey(sb.data, :b_p_subject_inactive_margins)
    @test brm_source(sb) === brmi
    @test parent(sb) === BRM._brm_resolve_prior_addresses(brmi)
end
