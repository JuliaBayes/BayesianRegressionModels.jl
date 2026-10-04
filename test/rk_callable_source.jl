include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module PublicNestedSource
using BayesianRegressionModels, Distributions, StanBlocks
import BayesianRegressionModels: _rk_callable_source!

# The original callables have only their Stan declaration methods. A native
# implementation is supplied as ordinary source, without editing the model.
StanBlocks.@deffun original_shift(x::vector[n], displacement::real) = x + displacement
StanBlocks.@deffun original_scale(x::vector[n], scale::real) = x * scale
const original_alias = original_shift
native_leaf(x, value) = x .* value
function original_keyword end

function _rk_callable_source!(definitions, bindings, entry,
        ::typeof(original_keyword))
    push!(definitions, :(function $entry(x; shift)
        return x .+ shift
    end))
    :done
end

function _rk_callable_source!(definitions, bindings, entry,
        ::typeof(original_shift))
    push!(definitions, :(function $entry(x, offset)
        return x .+ offset
    end))
    :done
end
function _rk_callable_source!(definitions, bindings, entry,
        ::typeof(original_scale))
    leaf = Symbol(entry, :_leaf)
    push!(bindings, leaf => native_leaf)
    push!(definitions, :(function $entry(x, scale)
        return $leaf(x, scale)
    end))
    :done
end

function build(data, route)
    if route === :kernel
        @brm data begin
            a ~ Normal(0, 0.7)
            b ~ Normal(0, 0.4)
            loc ~ kernel(x) do xx
                shifted = original_alias(xx, a)
                original_scale(shifted, b)
            end
            y ~ Normal(loc, 0.8)
        end
    else
        @brm data begin
            a ~ Normal(0, 0.7)
            b ~ Normal(0, 0.4)
            loc = original_scale(original_shift(x, a), b)
            y ~ Normal(loc, 0.8)
        end
    end
end
end

@stestset "original nested callable identities resolve to complete native source" begin
    for route in (:kernel, :assignment)
        data = route === :kernel ?
            (; x=[[-0.4, 0.2], Float64[], [0.7]], y=[[0.1, -0.2], Float64[], [0.3]]) :
            (; x=[-0.4, 0.2, 0.7], y=[0.1, -0.2, 0.3])
        saved = deepcopy(data)
        brmi = PublicNestedSource.build(data, route)
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        @test Set(names) == Set((:a, :b))
        ia, ib = findfirst(==(:a), names), findfirst(==(:b), names)
        x = route === :kernel ? reduce(vcat, data.x) : data.x
        y = route === :kernel ? reduce(vcat, data.y) : data.y
        oracle(u) = logpdf(Normal(0, 0.7), u[ia]) + logpdf(Normal(0, 0.4), u[ib]) +
            sum(logpdf.(Normal.((x .+ u[ia]) .* u[ib], 0.8), y))
        # Published StanBlocks caller-dimension substitution preserves the
        # original nested helpers for both dense and uneven/empty ragged axes.
        stan = consumer_stan(brmi,
            "nested-native-source-$route"; mod=PublicNestedSource)
        for u in (zeros(2), [0.2, -0.3], [-0.4, 0.1])
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan,
                [:a => "a", :b => "b"], backend, u)
        end
        emitted = BRM._rk_emit_ast(backend.plan)
        @test !any(p -> last(p) === PublicNestedSource.original_shift ||
            last(p) === PublicNestedSource.original_scale, emitted.bindings)
        @test any(p -> last(p) === PublicNestedSource.native_leaf, emitted.bindings)
        @test count(d -> d.head === :function, emitted.defs) >= 2
        artifact = BRM.emit_rk_artifact(brmi; case_id="nested-native-source-$route")
        translated = BRM.rk_translate_artifact(artifact)
        rebuilt = build_kernel(translated)
        @test coordinate_names(rebuilt.layout) == names
        @test isequal(data, saved)
    end
end

@stestset "native callable source retains live original keyword arguments" begin
    data = (; x=[-0.4, 0.2, 0.7], y=[0.1, -0.2, 0.3])
    saved = deepcopy(data)
    brmi = @brm data begin
        a ~ Normal(0, 0.7)
        loc = PublicNestedSource.original_keyword(x; shift=a)
        y ~ Normal(loc, 0.8)
    end
    backend, problem = consumer_problem(brmi)
    @test coordinate_names(backend.model.layout) == [:a]
    oracle(u) = logpdf(Normal(0, 0.7), u[1]) +
        sum(logpdf.(Normal.(data.x .+ u[1], 0.8), data.y))
    for u in ([0.0], [0.2], [-0.3])
        check_consumer_point(problem, u, oracle)
    end
    emitted = BRM._rk_emit_ast(backend.plan)
    @test !any(p -> last(p) === PublicNestedSource.original_keyword, emitted.bindings)
    @test isequal(data, saved)
end

module PublicInvalidSource
import BayesianRegressionModels: _rk_callable_source!
function missing_entry end
function shadow_entry end
function recursive_entry end
function unclaimed_mutation end
_rk_callable_source!(definitions, bindings, entry, ::typeof(missing_entry)) = :done
function _rk_callable_source!(definitions, bindings, entry, ::typeof(shadow_entry))
    push!(definitions, :(function $entry(x); x; end))
    push!(bindings, entry => identity)
    :done
end
function _rk_callable_source!(definitions, bindings, entry, ::typeof(recursive_entry))
    push!(definitions, :(function $entry(x); x; end))
    push!(bindings, Symbol(entry, :_leaf) => recursive_entry)
    :done
end
function _rk_callable_source!(definitions, bindings, entry, ::typeof(unclaimed_mutation))
    push!(bindings, :unclaimed_leaf => identity)
    nothing
end
end

@stestset "callable source providers fail before evaluation on invalid ownership" begin
    program(f) = BRM._RKEmittedProgram(Expr[], :(begin result = source(x); end),
        Pair{Symbol,Any}[:source => f])
    @test_throws "did not define its entry" BRM._rk_resolve_callable_sources(program(PublicInvalidSource.missing_entry))
    @test_throws "both bound and defined" BRM._rk_resolve_callable_sources(program(PublicInvalidSource.shadow_entry))
    @test_throws "rebound its original callable" BRM._rk_resolve_callable_sources(program(PublicInvalidSource.recursive_entry))
    @test_throws "unclaimed provider" BRM._rk_resolve_callable_sources(program(PublicInvalidSource.unclaimed_mutation))
    duplicate = BRM._RKEmittedProgram([:(function source(x); x; end),
        :(function source(x); 2x; end)], Expr(:block), Pair{Symbol,Any}[])
    @test_throws "defined more than once" BRM._rk_validate_source_definitions(duplicate)
end
