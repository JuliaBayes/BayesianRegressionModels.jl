include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# Public downstream code supplies its own mathematics. The marker deliberately
# has no evaluation method; both backends consume one unchanged BRMI.
module PublicSourceExtension
using BayesianRegressionModels, Distributions, StanBlocks
import BayesianRegressionModels: _rk_submodel_rhs!, _sb_submodel_rhs!

function source_marker end
native_location(x, location, shift) = location .+ x .* shift
stan_location = StanBlocks.@slic begin
    return location + x * shift
end

function _sb_submodel_rhs!(statements, data, target::Symbol,
        ::typeof(source_marker), rhs)
    x = only(getargs(rhs))
    key = Symbol(target, :_source_x)
    data[key] = copy(parent(parent(x)))
    location, shift = name(getkwargs(rhs).location), name(getkwargs(rhs).shift)
    push!(statements, :($target ~ stan_location(; x=$key, location=$location, shift=$shift)))
    :done
end

function _rk_submodel_rhs!(definitions, statements, data, bindings,
        target::Symbol, ::typeof(source_marker), rhs)
    x = only(getargs(rhs))
    key = Symbol(target, :_source_x)
    data[key] = copy(parent(parent(x)))
    location, shift = name(getkwargs(rhs).location), name(getkwargs(rhs).shift)
    helper = Symbol(target, :_native_location)
    reader = Symbol(target, :_reader)
    push!(bindings, helper => native_location)
    push!(definitions, :(function $reader(x, location, shift)
        return $helper(x, location, shift)
    end))
    push!(statements, :($target = $reader($key, $location, $shift)))
    :done
end

function build(data)
    @brm data begin
        a ~ 0 + x
        effect(a, :) ~ Normal(0, 0.7)
        shift ~ Normal(0, 0.4)
        loc ~ source_marker(x; location=a, shift=shift)
        y ~ Normal(loc, 0.8)
    end
end
end

@stestset "original empty marker emits a native submodel result without a coefficient" begin
    data = (; x=[-0.4, 0.2, 0.7], y=[0.1, -0.2, 0.3])
    saved = deepcopy(data)
    brmi = PublicSourceExtension.build(data)
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    @test length(names) == 2
    @test !any(n -> startswith(string(n), "loc"), names)
    @test :shift in names
    ia = findfirst(==(:a_x), names)
    ishift = findfirst(==(:shift), names)
    @test ia !== nothing
    oracle(u) = logpdf(Normal(0, 0.7), u[ia]) +
        logpdf(Normal(0, 0.4), u[ishift]) +
        sum(logpdf.(Normal.(data.x .* (u[ia] + u[ishift]), 0.8), data.y))
    stan = consumer_stan(brmi, "original-native-source-hook"; mod=PublicSourceExtension)
    mapping = [names[ia] => "pop_a_beta_pop.1", names[ishift] => "shift"]
    for u in (zeros(2), [0.2, -0.3], [-0.4, 0.1])
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, mapping, backend, u)
    end
    emitted = BRM._rk_emit_ast(backend.plan)
    @test any(p -> last(p) === PublicSourceExtension.native_location, emitted.bindings)
    @test any(d -> d.head === :function, emitted.defs)
    definition_name = first(first(emitted.defs).args[1].args)
    collision = BRM._RKEmittedProgram(emitted.defs, emitted.main,
        [emitted.bindings; definition_name => PublicSourceExtension.native_location])
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    @test_throws "both bound and defined" ext._rk_emit_module(collision)
    artifact = BRM.emit_rk_artifact(brmi; case_id="original-native-source-hook")
    @test artifact.ast == emitted.main
    @test isequal(data, saved)
end
