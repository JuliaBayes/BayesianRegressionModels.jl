# test/descriptor_trace_reuse.jl — descriptor random-effect queries read the
# descriptor's own traced program (snag resolve-a-fitted-00d21a25).
#
# `brm_descriptor` traces the Stan program once and keeps it in `d.stan`.
# `ranef_blocks(d)` and the resolvers built on it (`brm_ranef_sd_coordinates`,
# `brm_ranef_draws`, `brm_start`, coordinate transport, descriptions) take each
# block's carrier placement from that trace instead of tracing `d.plan.model`
# again, which cost one full trace per call.
#
# The proof is structural rather than a timing: the same descriptor with an
# untraceable `plan.model` still answers every query identically, while
# `ranef_blocks(plan)` on that plan fails (the positive control that the probe
# would see a re-trace).
#
# Run: julia --project=test test/descriptor_trace_reuse.jl
using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

const _MOD = @__MODULE__
const BRM = BayesianRegressionModels

block_fields(blocks) =
    [ntuple(i -> getfield(b, i), fieldcount(RanefBlock)) for b in blocks]

# The same descriptor over a plan whose program cannot be traced.
function untraceable(d)
    p = d.plan
    plan = BRM.GenerativePlan(p.parent, nothing, p.data, p.preproc, p.declarations,
                              p.builder, p.cv_groups, p.held_out, p.bindings)
    BRMDescriptor(d.id, d.name, d.formula, plan, d.stan, d.highlights, d.inputs,
                  d.outputs, d.operations, d.columns, d.unpredictable,
                  d.submodels, d.highlight_specs)
end

# Every constrained element name (`stem.i.j`) of the outputs with resolved
# axis lengths; the resolvers match elements by suffix, in any order. A square
# matrix type declares one size, its order.
square(type) = type in (:cholesky_factor_corr, :cholesky_factor_cov,
                        :corr_matrix, :cov_matrix)
function element_names(d)
    names = String[]
    for o in d.outputs
        isnothing(o.dims) && continue
        dims = square(o.type) && length(o.dims) == 1 ? (o.dims..., o.dims...) : o.dims
        if isempty(dims)
            push!(names, String(o.name))
        else
            for I in CartesianIndices(dims)
                push!(names, join((o.name, Tuple(I)...), "."))
            end
        end
    end
    names
end

const DF = (;
    x=Float64[-1.2, -0.5, 0.1, 0.8, 1.5, -0.3, 0.4, 1.1, 0.6, -0.9],
    y=Float64[0.1, -0.2, 0.4, 0.7, 1.1, -0.1, 0.3, 0.9, 0.2, -0.4],
    z=Float64[1.0, 0.8, 1.3, 1.1, 0.7, 0.9, 1.2, 1.4, 1.0, 0.6],
    g=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    h=[1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
)

const BUILDER = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | g) + (1 | h)
    nu ~ 1 + (1 | p | g)
    y ~ Normal(mu, sigma)
    z ~ Normal(nu, sigma)
end

@testset "fitted program: descriptor queries need no re-trace" begin
    sb = SBBRMI(BUILDER(DF); mod=_MOD, total_groups=())
    d = brm_descriptor(sb)
    blocks = ranef_blocks(d)
    @test block_fields(blocks) == block_fields(ranef_blocks(sb))
    @test block_fields(blocks) == block_fields(ranef_blocks(d.plan))
    @test length(blocks) == 2
    @test !any(b -> b.generated, blocks)

    u = untraceable(d)
    @test_throws Exception ranef_blocks(u.plan)   # control: tracing would fail
    @test block_fields(ranef_blocks(u)) == block_fields(blocks)

    names = element_names(d)
    for (lp, coefficient) in ((:mu, :Intercept), (:mu, :x), (:nu, :Intercept))
        a = brm_ranef_sd_coordinates(d, lp, names; id=:p, coefficient)
        b = brm_ranef_sd_coordinates(u, lp, names; id=:p, coefficient)
        @test a.coordinates == b.coordinates
        @test a.output.name === b.output.name
    end

    draws = reshape(collect(range(0.1, 0.9; length=3 * length(names))), 3, :)
    for id in (:p,), (a, b) in zip(brm_ranef_draws(d, draws, names; id),
                                   brm_ranef_draws(u, draws, names; id))
        @test a.binding === b.binding
        @test a.sd == b.sd
        @test a.cor == b.cor
        @test a.effects == b.effects
    end
end

@testset "prior program: generated placement comes from the descriptor's trace" begin
    # Omitting both responses moves every random-effect carrier into generated
    # quantities; `ranef_blocks(d)` must read that from `d.stan`.
    df = Base.structdiff(DF, (; y=nothing, z=nothing))
    sb = SBBRMI(BUILDER(df); mod=_MOD, total_groups=())
    d = brm_descriptor(sb)
    blocks = ranef_blocks(d)
    @test block_fields(blocks) == block_fields(ranef_blocks(sb))
    @test all(b -> b.generated, blocks)
    @test block_fields(ranef_blocks(untraceable(d))) == block_fields(blocks)
end
