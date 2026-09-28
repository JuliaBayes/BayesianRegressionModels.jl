# test/total_effects_population_coordinates.jl — `brm_population_effect_coordinates`
# resolves a coefficient absorbed by an exact total-coefficient block to its
# recovered generated carrier (snag population-effec-41c2d623).
#
# Run: julia --project=test test/total_effects_population_coordinates.jl
#
# Scope: one partially absorbed predictor (`mu ~ 1 + x + (1 | subject)`, whose
# slope stays sampled while its intercept is absorbed), one fully absorbed
# predictor with a linked LHS (`log(Vc) ~ 1 + (1 | subject)`), the
# conventional `total_groups=()` control, and the fail-closed cases (unknown
# coefficient, descriptor/artifact drift). Constrained names are
# BridgeStan-spelled (`stem.i` per labelled element): the resolvers match on
# those strings, so no compiled model is needed — the same precedent
# `test/prior_only_coordinates.jl` uses. The `stan_code` assertions below tie
# the resolved carriers to the independently emitted Stan program.
using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

const _MOD = @__MODULE__

df = (;
    x = [-1.2, -0.5, 0.1, 0.8, 1.5, -0.3, 0.4, 1.1],
    y1 = [0.1, -0.2, 0.4, 0.7, 1.1, -0.1, 0.3, 0.9],
    y2 = [0.3, 0.2, -0.4, 0.5, -1.1, 0.1, -0.3, 0.7],
    subject = [1, 1, 2, 2, 3, 3, 4, 4],
)
builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject)
    log(Vc) ~ 1 + (1 | subject)
    y1 ~ Normal(mu, sigma)
    y2 ~ Normal(log(Vc), sigma)
end

# BridgeStan-spelled constrained names from descriptor outputs. Labelled
# carriers (the resolver's targets) get exact `stem.i` container spelling;
# anything else is a distractor the resolver must not match, so its count is
# immaterial — only the stem spelling matters.
function _synth_names(d)
    names = String[]
    for o in d.outputs
        if !isnothing(o.labels)
            for i in eachindex(o.labels)
                push!(names, string(o.name, ".", i))
            end
        elseif isempty(o.size)
            push!(names, string(o.name))
        elseif all(x -> x isa Integer, o.size)
            for i in 1:prod(o.size)
                push!(names, string(o.name, ".", i))
            end
        else
            push!(names, string(o.name))
        end
    end
    names
end

@testset "absorbed coefficients resolve to the recovered carrier" begin
    sb = SBBRMI(builder(df); mod=_MOD)  # default total_groups=:auto
    blocks = total_effect_blocks(sb)
    @test Set(b.predictor for b in blocks) == Set((:mu, :Vc))
    @test all(b -> b.population_columns == (:Intercept,), blocks)

    d = brm_descriptor(sb; name=:totals_probe)
    names = _synth_names(d)

    # Partial absorption: the surviving slope stays sampled, the absorbed
    # intercept resolves to the recovered generated carrier.
    slope = brm_population_effect_coordinates(d, :mu, names; coefficient=:x)
    @test slope.output.name === :pop_mu_beta_pop
    @test slope.recovered === false
    @test names[slope.coordinates] == ["pop_mu_beta_pop.1"]

    intercept = brm_population_effect_coordinates(
        d, :mu, names; coefficient=:Intercept)
    @test intercept.logical === :mu
    @test intercept.coefficient === :Intercept
    @test intercept.output.name === :population_mu
    @test intercept.output.role === :population_effect
    @test intercept.recovered === true
    @test names[intercept.coordinates] == ["population_mu.1"]
    @test intercept.link === identity
    @test intercept.inverse_link === identity

    # Full absorption under a linked LHS: no conventional carrier exists, and
    # the recovered carrier keeps the predictor's link pair. The recovered
    # stem uses the PUBLIC predictor name (`population_Vc`, not
    # `population_log_Vc`) — the query address stays `:Vc` either way.
    linked = brm_population_effect_coordinates(
        d, :Vc, names; coefficient=:Intercept)
    @test linked.output.name === :population_Vc
    @test linked.recovered === true
    @test names[linked.coordinates] == ["population_Vc.1"]
    @test linked.link === log
    @test linked.inverse_link === exp

    # The two predictors do not leak into each other.
    @test intercept.coordinates != linked.coordinates

    # An unknown coefficient still fails closed, and the available labels name
    # the absorbed (recovered-addressable) label alongside the sampled one.
    @test_throws "available labels are (:x, :Intercept)" begin
        brm_population_effect_coordinates(d, :mu, names; coefficient=:Typo)
    end

    # A constrained-name vector that omits the generated-quantities axis
    # errors as descriptor/artifact drift rather than returning an empty slice.
    names_no_gq = filter(n -> !startswith(n, "population_"), names)
    @test_throws "Re-reflect" brm_population_effect_coordinates(
        d, :mu, names_no_gq; coefficient=:Intercept)

    # The resolved carriers are the independently emitted Stan program's own
    # recovered quantities, not descriptor-only metadata.
    code = BayesianRegressionModels.stan_code(sb)
    @test occursin("population_mu", code)
    @test occursin("population_Vc", code)
    @test occursin("brm_total_recover_rng", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "conventional rebuild keeps sampled resolutions" begin
    sb = SBBRMI(builder(df); mod=_MOD, total_groups=())
    @test isempty(total_effect_blocks(sb))
    d = brm_descriptor(sb; name=:conventional_probe)
    names = _synth_names(d)

    for (lp, carrier) in ((:mu, :pop_mu_beta_pop),
                          (:Vc, :pop_log_Vc_beta_pop))
        q = brm_population_effect_coordinates(d, lp, names; coefficient=:Intercept)
        @test q.output.name === carrier
        @test q.recovered === false
        @test names[q.coordinates] == [string(carrier, ".1")]
    end

    # Without a total block the failure text is byte-identical to before: no
    # recovered labels are appended to the available set.
    @test_throws "available labels are (:Intercept, :x)" begin
        brm_population_effect_coordinates(d, :mu, names; coefficient=:Typo)
    end
end
