# test/coordinates_axis_order.jl — REGRESSION for snag population-coeff-80eea661.
#
# The label- and margin-indexed coordinate resolvers
# (`brm_population_effect_coordinates`, `brm_term_coordinates`,
# `brm_ranef_sd_coordinates`, and the categorical-contrast path) must be
# axis-order free: each element resolves by its emitted `.i` suffix, so a
# reversed or permuted `constrained_names` returns the same elements a
# native-ordered axis does. They used to index axis-ordered matches by label
# position, so reversing the axis silently mapped `:Intercept` to the LAST
# beta element (`pop_loc_slope_beta_pop.5` instead of `.1` on Bruno's
# five-coefficient QT model). `brm_output_coordinates` is the deliberate
# exception: a whole-carrier slice preserves axis order by contract.
#
# Constrained names are BridgeStan-spelled (`stem.i` per element): the
# resolvers match on those strings, so no compiled model is needed — the same
# precedent `test/descriptor.jl` and
# `test/total_effects_population_coordinates.jl` use.
#
# Run: julia --project=. test/coordinates_axis_order.jl
using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

const _MOD = @__MODULE__

# Deterministic non-reverse permutation: a rotation by two. Fixed, so the
# suite is reproducible without an RNG.
_rotated(names) = circshift(names, 2)

@testset "population coefficients resolve under any axis order" begin
    # Bruno's QT shape: five numeric covariates plus a grouping factor, with
    # conventional emission (`total_groups=()`).
    df = (;
        zage=Float64[-1.2, -0.5, 0.1, 0.8, 1.5, -0.3, 0.4, 1.1],
        zweight=Float64[0.3, -0.7, 1.2, -0.1, 0.5, -1.0, 0.9, -0.4],
        male=Float64[0, 1, 0, 1, 0, 1, 0, 1],
        diseased=Float64[1, 0, 1, 0, 1, 0, 1, 0],
        y=Float64[0.1, -0.2, 0.4, 0.7, 1.1, -0.1, 0.3, 0.9],
        subjects=[1, 1, 2, 2, 3, 3, 4, 4],
    )
    builder = @brm begin
        sigma ~ Exponential(1)
        loc_slope ~ 1 + zage + zweight + male + diseased + (1 | subjects)
        y ~ Normal(loc_slope, sigma)
    end
    d = brm_descriptor(builder, df; total_groups=(), mod=_MOD, name=:axis_order)
    beta = only(filter(o -> o.name === :pop_loc_slope_beta_pop, d.outputs))
    @test beta.labels == [:Intercept, :zage, :zweight, :male, :diseased]

    native = ["pop_loc_slope_beta_pop.$i" for i in 1:5]
    for axis in (native, reverse(native), _rotated(native))
        for (k, coefficient) in enumerate(beta.labels)
            q = brm_population_effect_coordinates(
                d, :loc_slope, axis; coefficient=coefficient)
            @test q.output.name === :pop_loc_slope_beta_pop
            @test q.recovered === false
            @test axis[q.coordinates] == ["pop_loc_slope_beta_pop.$k"]
        end
    end

    # A strict subset still reports the long-standing count mismatch (the
    # element set `1..1` is complete, so the carrier check passes and the
    # label-count check fires exactly as before).
    @test_throws "5 coefficient labels but resolves to 1 constrained coordinates" begin
        brm_population_effect_coordinates(
            d, :loc_slope, ["pop_loc_slope_beta_pop.1"]; coefficient=:Intercept)
    end
end

@testset "single-coefficient carriers are trivially order-free" begin
    # Why Bruno's reversed `loc_loc` probe passed while `loc_slope` failed: a
    # one-element permutation is the identity.
    df = (;
        y=Float64[0.1, -0.2, 0.4, 0.7, 1.1, -0.1],
        subjects=[1, 1, 2, 2, 3, 3],
    )
    builder = @brm begin
        sigma ~ Exponential(1)
        loc ~ 1 + (1 | subjects)
        y ~ Normal(loc, sigma)
    end
    d = brm_descriptor(builder, df; total_groups=(), mod=_MOD, name=:axis_single)
    beta = only(filter(o -> o.name === :pop_loc_beta_pop, d.outputs))
    @test beta.labels == [:Intercept]
    axis = ["pop_loc_beta_pop.1"]
    for names in (axis, reverse(axis))
        q = brm_population_effect_coordinates(
            d, :loc, names; coefficient=:Intercept)
        @test names[q.coordinates] == ["pop_loc_beta_pop.1"]
    end
end

@testset "categorical contrasts pair levels to elements under any axis order" begin
    df = (;
        indication=[1, 2, 3, 1, 2, 3],
        y=zeros(6),
    )
    builder = @brm begin
        log(Vc) ~ 1 + indication
        y ~ Normal(log(Vc), 1.0)
    end
    d = brm_descriptor(builder, df; mod=_MOD, name=:axis_categorical)
    native = ["cat_log_Vc_indication_beta.1", "cat_log_Vc_indication_beta.2"]
    for names in (native, reverse(native))
        r = brm_population_effect_coordinates(
            d, :Vc, names; coefficient=:indication)
        @test r.reference_level == 1
        @test r.nonreference_levels == [2, 3]
        @test length(r.contrasts) == 2
        for (i, c) in enumerate(r.contrasts)
            @test c.nonreference_level == i + 1
            @test c.reference_level == 1
            @test names[c.coordinate] == "cat_log_Vc_indication_beta.$i"
        end
        @test names[r.coordinates] == native
    end
end

@testset "term parameters resolve in element order under any axis order" begin
    df = (;
        x=[0.0, 1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 3.5],
        g=[1, 2, 3, 4, 1, 2, 3, 4],
        ya=zeros(8),
    )
    builder = @brm begin
        a ~ 0 + x + mo(g)
        ya ~ Normal(a, 1.0)
    end
    d = brm_descriptor(builder, df; mod=_MOD, name=:axis_term)
    carrier = only(filter(
        o -> endswith(string(o.name), "_simplex_incr"), d.outputs))
    stem = string(carrier.name)
    native = ["$stem.$i" for i in 1:3]
    # The native axis establishes the three-element expectation (the resolver
    # errors unless the count matches the term's own expectation); the
    # permuted axes must return the same elements.
    r_native = brm_term_coordinates(d, :a, native; term=:mo_g, parameter=:simplex)
    @test r_native.output.name === carrier.name
    @test r_native.coordinates == [1, 2, 3]
    for names in (reverse(native), _rotated(native))
        r = brm_term_coordinates(d, :a, names; term=:mo_g, parameter=:simplex)
        @test r.output.name === carrier.name
        @test names[r.coordinates] == native
    end
end

@testset "ranef margins resolve under any axis order" begin
    df = (;
        x=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        y=[0.1, -0.2, 0.4, 0.7, 1.1, -0.1],
        subjects=[1, 1, 2, 2, 3, 3],
    )
    builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | subjects)
        y ~ Normal(mu, sigma)
    end
    brmi = builder(df)
    margins = ranefcoefnames(brmi, :p)
    @test length(margins) == 2
    d = brm_descriptor(builder, df; mod=_MOD, name=:axis_ranef)
    stem = string(only(filter(o -> o.role === :random_effect &&
                                  endswith(string(o.name), "_tau"), d.outputs)).name)
    native = ["$stem.1", "$stem.2"]
    # Two elements admit exactly one non-trivial permutation: reversal. (A
    # rotation by two is the identity here.)
    for names in (native, reverse(native))
        for (k, m) in enumerate(margins)
            r = brm_ranef_sd_coordinates(
                d, m.predictor, names; id=:p, coefficient=m.coefficient)
            @test names[r.coordinates] == ["$stem.$k"]
        end
    end
end

@testset "recovered coefficients resolve under a permuted axis" begin
    # Same partially absorbed shape as
    # `test/total_effects_population_coordinates.jl`: the surviving slope stays
    # sampled while the intercept resolves to the recovered carrier — here
    # with the full name axis reversed.
    df = (;
        x=[-1.2, -0.5, 0.1, 0.8, 1.5, -0.3, 0.4, 1.1],
        y1=[0.1, -0.2, 0.4, 0.7, 1.1, -0.1, 0.3, 0.9],
        subject=[1, 1, 2, 2, 3, 3, 4, 4],
    )
    builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | subject)
        y1 ~ Normal(mu, sigma)
    end
    sb = SBBRMI(builder(df); mod=_MOD)  # default total_groups=:auto
    blocks = total_effect_blocks(sb)
    @test length(blocks) == 1 && only(blocks).population_columns == (:Intercept,)
    d = brm_descriptor(sb; name=:axis_recovered)
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
    for axis in (names, reverse(names))
        slope = brm_population_effect_coordinates(d, :mu, axis; coefficient=:x)
        @test slope.output.name === :pop_mu_beta_pop
        @test slope.recovered === false
        @test axis[slope.coordinates] == ["pop_mu_beta_pop.1"]
        intercept = brm_population_effect_coordinates(
            d, :mu, axis; coefficient=:Intercept)
        @test intercept.output.name === :population_mu
        @test intercept.recovered === true
        @test axis[intercept.coordinates] == ["population_mu.1"]
    end
end

@testset "carrier element verification fails closed" begin
    df = (;
        weight=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5],
    )
    builder = @brm begin
        log(Vc) ~ 1 + weight
        y ~ Normal(log(Vc), 0.2)
    end
    d = brm_descriptor(builder, df; mod=_MOD, name=:axis_drift)
    stem = "pop_log_Vc_beta_pop"
    base = ["$stem.1", "$stem.2"]

    # A duplicated element is ambiguous, not resolvable.
    @test_throws "matches constrained element `$stem.2` more than once" begin
        brm_population_effect_coordinates(
            d, :Vc, [base; "$stem.2"]; coefficient=:Intercept)
    end
    # A substituted element (missing `.1`, extra `.3`) has the right count but
    # is not the carrier's element set.
    @test_throws "not the complete element set of one carrier" begin
        brm_population_effect_coordinates(
            d, :Vc, ["$stem.2", "$stem.3"]; coefficient=:Intercept)
    end
    # A non-integer suffix is not a container coordinate.
    @test_throws "whose container suffix is not a positive-integer coordinate" begin
        brm_population_effect_coordinates(
            d, :Vc, ["$stem.1", "$stem.2x"]; coefficient=:Intercept)
    end
    # The bare stem beside container elements mixes a scalar with a vector.
    @test_throws "including its bare scalar name" begin
        brm_population_effect_coordinates(
            d, :Vc, [stem; base]; coefficient=:Intercept)
    end
    # Mixed coordinate arity is not one carrier.
    @test_throws "mixed coordinate arity" begin
        brm_population_effect_coordinates(
            d, :Vc, ["$stem.1", "$stem.1.1"]; coefficient=:Intercept)
    end
end

@testset "whole-carrier slices keep axis order" begin
    # `brm_output_coordinates` is the deliberate exception to element order:
    # it preserves the supplied axis order by contract.
    df = (;
        zage=Float64[-1.2, -0.5, 0.1, 0.8],
        y=Float64[0.1, -0.2, 0.4, 0.7],
        subjects=[1, 1, 2, 2],
    )
    builder = @brm begin
        sigma ~ Exponential(1)
        loc_slope ~ 1 + zage + (1 | subjects)
        y ~ Normal(loc_slope, sigma)
    end
    d = brm_descriptor(builder, df; total_groups=(), mod=_MOD, name=:axis_slice)
    beta = only(filter(o -> o.name === :pop_loc_slope_beta_pop, d.outputs))
    native = ["pop_loc_slope_beta_pop.1", "pop_loc_slope_beta_pop.2"]
    @test brm_output_coordinates(beta, native) == [1, 2]
    rev = reverse(native)
    # Axis order, not element order: every match in axis position order, so
    # the element sequence follows the axis — the opposite of the
    # label-indexed resolvers above.
    @test brm_output_coordinates(beta, rev) == [1, 2]
    @test rev[brm_output_coordinates(beta, rev)] ==
        ["pop_loc_slope_beta_pop.2", "pop_loc_slope_beta_pop.1"]
end
