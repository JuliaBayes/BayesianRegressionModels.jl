# Exercise callable and statistical source together, including complete
# printed-source replay of the native density and every Reverse coordinate.
include(joinpath(@__DIR__, "rk_statistical_source.jl"))
using Distributions

module PublicComposedSource
import BayesianRegressionModels: _rk_callable_source!

function original_affine end
const original_alias = original_affine
native_scale(x, scale) = x .* scale

function _rk_callable_source!(definitions, bindings, entry,
        ::typeof(original_affine))
    leaf = Symbol(entry, :_leaf)
    push!(bindings, leaf => native_scale)
    push!(definitions, :(function $entry(x; scale, shift)
        return $leaf(x, scale) .+ shift
    end))
    :done
end
end

@testset "native callable sources compose with statistical definitions" begin
    data = (; x=[-1., -.3, .4, 1.], g=[1, 2, 1, 2], y=[.2, -.1, .3, -.2])
    cases = (
        "correlated" => @brm(data, begin
            eta ~ 1 + x + (1+x|block|g)
            sd(:,block) ~ Exponential(.7)
            cor(:,block) ~ LKJCholesky(2,3.)
            scale ~ Normal(0, .7)
            shift ~ Normal(0, .4)
            mu = PublicComposedSource.original_alias(eta; scale=scale, shift=shift)
            y ~ Normal(mu, .8)
        end),
        "HSGP" => @brm(data, begin
            eta ~ 1 + hsgp(x; k=3)
            scale ~ Normal(0, .7)
            shift ~ Normal(0, .4)
            mu = PublicComposedSource.original_alias(eta; scale=scale, shift=shift)
            y ~ Normal(mu, .8)
        end),
        "grouped HSGP" => @brm(data, begin
            eta ~ 1 + hsgp(x; k=3, by=g)
            scale ~ Normal(0, .7)
            shift ~ Normal(0, .4)
            mu = PublicComposedSource.original_alias(eta; scale=scale, shift=shift)
            y ~ Normal(mu, .8)
        end))
    for (label, brmi) in cases
        @testset "$label" begin
            saved = deepcopy(data)
            backend = check_rk_source_roundtrip(RKBRMI(brmi);
                ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            @test isequal(data, saved)
            emitted = BRM._rk_emit_ast(backend.plan)
            @test !any(d -> occursin("brm_design_product", sprint(show, d)), emitted.defs)
            @test any(d -> occursin("return X * beta_pop", sprint(Base.show_unquoted, d)),
                emitted.defs)
            @test any(d -> d.head === :function, emitted.defs)
            @test any(d -> d.head === :(=), emitted.defs)
            @test !any(p -> last(p) === PublicComposedSource.original_affine,
                emitted.bindings)
            @test any(p -> last(p) === PublicComposedSource.native_scale,
                emitted.bindings)
        end
    end
end
