# test/stanblocks_preservation_corpus.jl -- bounded StanBlocks preservation corpus.
#
# Capture the implementation-base artifact:
#   BRM_SB_PRESERVATION=write BRM_SB_PRESERVATION_FILE=/tmp/brm-sb.bin \
#     julia --startup-file=no --project=test test/stanblocks_preservation_corpus.jl
# Compare a later extraction against it:
#   BRM_SB_PRESERVATION=check BRM_SB_PRESERVATION_FILE=/tmp/brm-sb.bin \
#     julia --startup-file=no --project=test test/stanblocks_preservation_corpus.jl
#
# The binary is deliberately external. It contains exact emitted SLIC/Stan text,
# prepared data, and compact semantic identities; this file contains only the
# representative source fixtures that should remain easy to review and extend.

using Test
using Serialization
using BayesianRegressionModels
using Distributions
using StanBlocks

const BRM = BayesianRegressionModels

base_df() = (;
    x=[-1.0, -0.4, 0.1, 0.7, 1.2, 1.8],
    z=[0.2, -0.7, 1.1, 0.4, -0.3, 0.9],
    g=[1, 1, 2, 2, 3, 3], h=[1, 2, 1, 2, 1, 2],
    cat=[1, 2, 3, 1, 2, 3],
    y=[-0.8, -0.2, 0.1, 0.7, 1.0, 1.4],
    y2=[0.1, 0.3, -0.2, 0.8, 0.4, 1.1],
    count=[0, 1, 2, 1, 3, 2], trials=fill(4, 6),
    weight=[1.0, 2.0, 1.0, 0.5, 1.5, 1.0],
    y_upper=[-0.5, 0.0, 0.4, 1.0, 1.3, 1.8],
)

const BASIC_GAUSSIAN = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + z
    y ~ Normal(mu, sigma)
end

const TRANSFORMED_GROUPED = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + zscale(x) + center(z) + zscale(x) & center(z) +
          (1 + zscale(x) | g)
    y ~ Normal(mu, sigma)
end

const CATEGORICAL_ZEROCORR = @brm begin
    mu ~ 1 + factor(cat; ref=2) + (1 + factor(cat; ref=2) | g) +
         (0 + x || h)
    y ~ Normal(mu, 1.0)
end

const SHARED_ID = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | shared | g)
    log(rate) ~ 1 + z + (1 | shared | g)
    y ~ Normal(mu, sigma)
    count ~ Poisson(rate)
end

const MM_MODEL = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | mm(g1, g2; weights=(w1, w2)))
    y ~ Normal(mu, sigma)
end

mm_df() = (;
    x=[-1.0, -0.2, 0.4, 1.1], y=[0.1, -0.2, 0.5, 0.8],
    g1=["a", "a", "b", "c"], g2=["b", "c", "c", "a"],
    w1=[2.0, 1.0, 3.0, 1.0], w2=[1.0, 1.0, 1.0, 2.0],
)

const SIMPLE_PRIORS = @brm begin
    beta ~ Horseshoe(local_scale=0.25, global_scale=0.1)
    sigma ~ Exponential(2)
    effect(mu, x) ~ Normal(0, 0.4)
    mu ~ 1 + x
    y ~ Normal(mu + beta, sigma)
end

const R2D2_MODEL = @brm begin
    mu ~ 1 + x + z + (1 | g)
    effect(mu, :) ~ r2d2(R2=Beta(2, 3), tau_bsv=0.5, alpha=0.7)
    y ~ Normal(mu, 1.0)
end

const HSGP_MODEL = @brm begin
    mu ~ 1 + hsgp(x; k=4, c=1.5) + (1 | g)
    length_scale(:, hsgp(x)) ~ LogNormal(0, 0.5)
    sd(:, hsgp(x)) ~ Normal(0, 0.7)
    y ~ Normal(mu, 1.0)
end

const WEIGHTED_MODEL = @brm begin
    mu ~ 1 + x
    y ~ weighted(Normal(mu, 1.2), weights(weight))
end

const WRAPPED_MODEL = @brm begin
    mu ~ 1 + x
    y ~ interval_censored(Normal(mu, 1.0); upper=y_upper)
end

const JOINT_MODEL = @brm begin
    L ~ LKJCovarianceFactor(2; scale_prior=Exponential(1), shape=2)
    mu1 ~ 1 + x
    mu2 ~ 1 + z
    [y, y2] ~ MvNormalCholesky([mu1, mu2], L)
end

const ORDINAL_MODEL = @brm begin
    eta ~ 0 + x
    count ~ Ordinal(Cumulative(), ProbitLink(), eta)
end

ordinal_df() = (; x=base_df().x, count=[1, 2, 3, 1, 2, 3])

const RAGGED_KERNEL = @brm begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + (1 | pk | subject)
    pred ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
        exp(lCL) .+ 0.0 .* idxs
    end
    ragged(obs_y, obs_subject) ~ Normal(pred, sigma)
end

ragged_df() = (;
    subject=["s2", "s1", "s3"],
    obs_subject=["s1", "s2", "s3", "s2", "s3", "s3"],
    obs_idx=[1.0, 1.0, 1.0, 2.0, 2.0, 3.0],
    obs_y=[0.11, 0.21, 0.31, 0.22, 0.32, 0.33],
)

const REPLAY_MODEL = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + zscale(x) + factor(cat) + (1 | g)
    y ~ Normal(mu, sigma)
end

replay_df() = (;
    x=[-0.8, 0.0, 0.9, 1.6], cat=[3, 2, 1, 3],
    g=[1, 2, 3, 1], y=[-0.4, 0.2, 0.9, 1.3],
)

# Each entry is intentionally small; together these fixtures cross the major
# semantic boundaries that common-program extraction is expected to preserve.
fixtures() = [
    (; name=:basic_fused_gaussian, builder=BASIC_GAUSSIAN, data=base_df()),
    (; name=:transformed_grouped, builder=TRANSFORMED_GROUPED, data=base_df()),
    (; name=:categorical_zerocorr, builder=CATEGORICAL_ZEROCORR, data=base_df()),
    (; name=:shared_id_distributional, builder=SHARED_ID, data=base_df()),
    (; name=:multi_membership, builder=MM_MODEL, data=mm_df()),
    (; name=:simple_and_horseshoe_priors, builder=SIMPLE_PRIORS, data=base_df()),
    (; name=:r2d2, builder=R2D2_MODEL, data=base_df()),
    (; name=:hsgp, builder=HSGP_MODEL, data=base_df()),
    (; name=:weighted_observation, builder=WEIGHTED_MODEL, data=base_df()),
    (; name=:interval_observation, builder=WRAPPED_MODEL, data=base_df()),
    (; name=:joint_response, builder=JOINT_MODEL, data=base_df()),
    (; name=:ordinal, builder=ORDINAL_MODEL, data=ordinal_df()),
    (; name=:ragged_kernel, builder=RAGGED_KERNEL, data=ragged_df()),
    (; name=:frozen_replay, builder=REPLAY_MODEL, data=base_df(), replay=replay_df()),
]

stable_value(x::AbstractDict) = [string(k) => stable_value(x[k])
                                 for k in sort!(collect(keys(x)); by=string)]
stable_value(x::NamedTuple) = [(string(k), stable_value(v)) for (k, v) in pairs(x)]
stable_value(x::Tuple) = map(stable_value, x)
stable_value(x::AbstractArray) = (; size=size(x), values=map(stable_value, vec(x)))
stable_value(x::Set) = sort!(map(stable_value, collect(x)); by=repr)
stable_value(x::Module) = string(x)
stable_value(x::Function) = string(parentmodule(x), ".", nameof(x))
stable_value(x::Symbol) = string(x)
stable_value(x::Union{Nothing,Missing,Bool,Number,AbstractString}) = x
stable_value(x) = repr(x)

identity_record(x) = [(string(k), stable_value(getproperty(x, k)))
                      for k in propertynames(x)
                      if k in (:target, :family, :arguments, :keywords, :data_source,
                               :draw, :role, :context, :generated, :name, :logical,
                               :kind, :labels, :segments, :column, :transform)]

function capture_fixture(fixture)
    sb = SBBRMI(fixture.builder(fixture.data); mod=@__MODULE__)
    plan = generative_plan(sb)
    descriptor = brm_descriptor(sb; name=fixture.name, highlights=())
    replay = hasproperty(fixture, :replay) ?
        reprocess(sb, fixture.replay; freeze_constants=true) : nothing
    (;
        slic=sprint(show, sb.model.model),
        stan=BRM.stan_code(sb),
        data=stable_value(sb.data),
        preproc=stable_value(sb.preproc),
        declarations=map(identity_record, plan.declarations),
        descriptor=(;
            id=descriptor.id,
            columns=map(string, descriptor.columns),
            inputs=map(identity_record, descriptor.inputs),
            outputs=map(identity_record, descriptor.outputs),
            operations=map(x -> string(x.name), descriptor.operations),
            unpredictable=map(string, descriptor.unpredictable),
        ),
        replay=isnothing(replay) ? nothing : (;
            slic=sprint(show, replay.model.model),
            stan=BRM.stan_code(replay),
            data=stable_value(replay.data),
            preproc=stable_value(replay.preproc),
        ),
    )
end

capture_corpus() = Dict(f.name => capture_fixture(f) for f in fixtures())

# Inline kernel ASTs retain their fixture's absolute source path. The base and
# implementation intentionally run in different checkouts; retain the source
# filename and line while comparing all executable SLIC/Stan text exactly.
fixture_source_location(text::AbstractString) = replace(text,
    r"#= [^\n]*[/\\]test[/\\]stanblocks_preservation_corpus\.jl:(\d+) =#" =>
        s"#= test/stanblocks_preservation_corpus.jl:\1 =#")

function first_difference(expected, actual, path="corpus")
    typeof(expected) == typeof(actual) || return "$path: type $(typeof(expected)) != $(typeof(actual))"
    if expected isa AbstractString
        fixture_source_location(expected) == fixture_source_location(actual) && return nothing
    end
    expected == actual && return nothing
    if expected isa AbstractDict
        keys(expected) == keys(actual) || return "$path: keys differ"
        for k in sort!(collect(keys(expected)); by=string)
            diff = first_difference(expected[k], actual[k], "$path.$k")
            isnothing(diff) || return diff
        end
        return nothing
    elseif expected isa NamedTuple
        keys(expected) == keys(actual) || return "$path: fields differ"
        for k in keys(expected)
            diff = first_difference(getproperty(expected, k), getproperty(actual, k), "$path.$k")
            isnothing(diff) || return diff
        end
        return nothing
    elseif expected isa AbstractVector
        length(expected) == length(actual) || return "$path: lengths differ"
        for i in eachindex(expected)
            diff = first_difference(expected[i], actual[i], "$path[$i]")
            isnothing(diff) || return diff
        end
        return nothing
    end
    "$path: values differ\nexpected: $(repr(expected))\nactual:   $(repr(actual))"
end

function run_preservation_corpus()
    mode = get(ENV, "BRM_SB_PRESERVATION", "capture")
    path = get(ENV, "BRM_SB_PRESERVATION_FILE", "")
    mode in ("capture", "write", "check") ||
        error("BRM_SB_PRESERVATION must be capture, write, or check")
    mode == "capture" || !isempty(path) ||
        error("BRM_SB_PRESERVATION_FILE is required in $mode mode")

    actual = capture_corpus()
    @test length(actual) == 14
    @test all(!isempty(snapshot.stan) && !isempty(snapshot.slic) for snapshot in values(actual))

    if mode == "write"
        open(path, "w") do io
            serialize(io, actual)
        end
        @info "wrote StanBlocks preservation baseline" path fixtures=length(actual)
    elseif mode == "check"
        expected = open(deserialize, path)
        diff = first_difference(expected, actual)
        isnothing(diff) || error(diff)
        @test isnothing(diff)
    end
    actual
end

if abspath(PROGRAM_FILE) == @__FILE__
    @testset "StanBlocks preservation corpus" begin
        run_preservation_corpus()
    end
end
