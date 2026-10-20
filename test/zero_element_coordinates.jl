# test/zero_element_coordinates.jl — `brm_output_coordinates` on carriers with
# no elements (snag `brm-output-coord-adf66ae9`).
#
# A one-level treatment factor's contrast vector is `vector[n_levels - 1]`, and
# a one-level ordinal outcome's thresholds are `ordered[0]`: valid, emitted, and
# listed by BridgeStan under no name at all. Mapping every sampled output of a
# descriptor to its posterior coordinates must therefore return `Int[]` for
# them instead of reporting descriptor/artifact drift, while a carrier that
# does own elements and is absent from the names still fails closed.
#
# Oracle: BridgeStan's own `param_unc_names` / `param_names` of the compiled
# model. Every unconstrained coordinate is owned by exactly one sampled output.
#
# Run: julia --project=test test/zero_element_coordinates.jl

using Test, BayesianRegressionModels
using Distributions: Exponential
import StanBlocks
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))

compiled(d) = brm_execute(d, :fit).model

output(d, name) = only(o for o in d.outputs if o.name === name)

# The consumer workflow from the snag: every sampled output, resolved through
# the discovered-carrier overload, owns each unconstrained coordinate once.
function owners(d, names)
    owner = Vector{Union{Nothing,Symbol}}(nothing, length(names))
    for o in d.outputs
        o.kind === :parameter || continue
        for c in brm_output_coordinates(o, names)
            @test isnothing(owner[c])
            owner[c] = o.name
        end
    end
    owner
end

x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
y = [1.0, 1.5, 2.0, 2.4, 3.1, 3.4]

factor_model = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + factor(src; cmc=false)
    y ~ Normal(mu, sigma)
end

@stestset "one-level treatment factor: empty contrast carrier" begin
    d = brm_descriptor(factor_model, (; src=fill(1, 6), x, y);
                       mod=@__MODULE__, name=:zero_element_factor)
    beta = output(d, :cat_mu_src_beta)
    @test beta.kind === :parameter
    @test beta.dims == (0,)
    @test output(d, :sigma).dims == ()
    @test output(d, :mu).dims == (6,)

    model = compiled(d)
    unc = StanBlocks.BridgeStan.param_unc_names(model)
    con = StanBlocks.BridgeStan.param_names(model; include_tp=true, include_gq=true)
    @test !any(n -> startswith(n, "cat_mu_src_beta"), con)

    @test brm_output_coordinates(beta, unc) == Int[]
    @test brm_output_coordinates(beta, con) == Int[]
    @test all(!isnothing, owners(d, unc))

    # The categorical query already answered this block as empty; the carrier
    # it names now agrees with it.
    result = brm_population_effect_coordinates(d, :mu, con; coefficient=:src)
    @test result.output.name === beta.name
    @test isempty(result.coordinates)
end

@stestset "two-level control: the same carrier owns its contrast" begin
    d = brm_descriptor(factor_model, (; src=[1, 2, 1, 2, 1, 2], x, y);
                       mod=@__MODULE__, name=:zero_element_control)
    beta = output(d, :cat_mu_src_beta)
    @test beta.dims == (1,)
    model = compiled(d)
    unc = StanBlocks.BridgeStan.param_unc_names(model)
    @test unc[brm_output_coordinates(beta, unc)] == ["cat_mu_src_beta.1"]
    @test all(!isnothing, owners(d, unc))
end

@stestset "drift still fails closed" begin
    d = brm_descriptor(factor_model, (; src=fill(1, 6), x, y);
                       mod=@__MODULE__, name=:zero_element_factor)
    beta = output(d, :cat_mu_src_beta)
    names = ["sigma", "pop_mu_beta_pop.1", "pop_mu_beta_pop.2"]
    # A carrier that owns elements and is absent from the names is drift.
    @test_throws "is absent from the supplied constrained names" begin
        brm_output_coordinates(output(d, :pop_mu_beta_pop), ["sigma"])
    end
    @test_throws "is absent from the supplied constrained names" begin
        brm_output_coordinates(d, :sigma, names[2:end])
    end
    # Without resolved dims nothing proves the carrier empty, so absence is
    # still drift (a hand-assembled output).
    unresolved = BRMOutput(beta.name, beta.kind, beta.type, beta.size,
                           beta.constraints, beta.generative, beta.source,
                           beta.role, beta.declaration, beta.logical,
                           beta.labels, beta.segments, beta.component)
    @test isnothing(unresolved.dims)
    @test_throws "is absent from the supplied constrained names" begin
        brm_output_coordinates(unresolved, names)
    end
end

@stestset "one-level cell-mean factor owns one coordinate" begin
    cells = @brm begin
        sigma ~ Exponential(1)
        mu ~ 0 + factor(src)
        y ~ Normal(mu, sigma)
    end
    d = brm_descriptor(cells, (; src=fill(1, 6), y);
                       mod=@__MODULE__, name=:zero_element_cells)
    beta = output(d, :cat_mu_src_beta)
    @test beta.dims == (1,)
    @test_throws "is absent from the supplied constrained names" begin
        brm_output_coordinates(beta, ["sigma"])
    end
end

@stestset "one-level ordinal outcome: empty thresholds" begin
    ordinal = @brm begin
        eta ~ 0 + x
        y ~ OrderedLogistic(eta)
    end
    d = brm_descriptor(ordinal, (; x, y=fill(1, 6));
                       mod=@__MODULE__, name=:zero_element_ordinal)
    cutpoints = output(d, :y_cutpoints)
    @test cutpoints.dims == (0,)
    model = compiled(d)
    unc = StanBlocks.BridgeStan.param_unc_names(model)
    @test brm_output_coordinates(cutpoints, unc) == Int[]
    @test all(!isnothing, owners(d, unc))
end
