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
    stan = consumer_stan(brmi, "original-native-source-hook"; mod=PublicSourceExtension)
    # Locate the coefficient and pair both backends through the semantic
    # transport rather than a spelled coordinate name.
    sb = SBBRMI(brmi; mod=PublicSourceExtension, total_groups=())
    transport = brm_coordinate_transport(backend, sb,
        BridgeStan.param_unc_names(stan.model))
    rk_index(address) =
        findfirst(==(only(p.rk for p in transport.pairs if p.address == address)), names)
    ia = rk_index((; kind=:population, predictor=:a, coefficient=:x))
    ishift = rk_index((; kind=:scalar, name=:shift))
    @test names[ishift] === :shift
    oracle(u) = logpdf(Normal(0, 0.7), u[ia]) +
        logpdf(Normal(0, 0.4), u[ishift]) +
        sum(logpdf.(Normal.(data.x .* (u[ia] + u[ishift]), 0.8), data.y))
    mapping = [p.rk => p.stan for p in transport.pairs]
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

# Two submodels calling one shared recurrence: each hook call appends the same
# helper definition, and the printed source defines it once.
module SharedHelperSource
using BayesianRegressionModels, Distributions
import BayesianRegressionModels: _rk_submodel_rhs!

function first_events end
function second_events end
const SECOND_FACTOR = Ref(1.0)

recurrence(factor) = :(ReactiveKernels.@kernel shared_response(x, gain) = begin
    values = ReactiveKernels.scan(x, Ref(gain); init=0.0) do carry, value, g
        next = carry + $factor * g * value
        (next, next)
    end
    return values
end)

function shared_source!(definitions, statements, data, target, rhs, factor)
    x = only(getargs(rhs))
    key = Symbol(target, :_source_x)
    data[key] = copy(parent(parent(x)))
    gain = name(getkwargs(rhs).gain)
    push!(definitions, recurrence(factor))
    push!(statements, :($target = shared_response($key, $gain)))
    :done
end
_rk_submodel_rhs!(definitions, statements, data, bindings, target::Symbol,
    ::typeof(first_events), rhs) =
    shared_source!(definitions, statements, data, target, rhs, 1.0)
_rk_submodel_rhs!(definitions, statements, data, bindings, target::Symbol,
    ::typeof(second_events), rhs) =
    shared_source!(definitions, statements, data, target, rhs, SECOND_FACTOR[])

function build(data)
    @brm data begin
        a ~ Normal(0, 0.7)
        b ~ Normal(0, 0.5)
        first_loc ~ first_events(x; gain=a)
        second_loc ~ second_events(z; gain=b)
        y ~ Normal(first_loc, 0.8)
        w ~ Normal(second_loc, 0.6)
    end
end
end

@stestset "submodel hooks sharing one helper emit its definition once" begin
    data = (; x=[-0.4, 0.2, 0.7, 0.1], z=[0.3, -0.5, 0.25, 0.6],
        y=[0.1, -0.2, 0.3, 0.05], w=[-0.1, 0.4, 0.2, -0.3])
    saved = deepcopy(data)
    brmi = SharedHelperSource.build(data)
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    ia, ib = findfirst(==(:a), names), findfirst(==(:b), names)
    oracle(u) = logpdf(Normal(0, 0.7), u[ia]) + logpdf(Normal(0, 0.5), u[ib]) +
        sum(logpdf.(Normal.(cumsum(data.x) .* u[ia], 0.8), data.y)) +
        sum(logpdf.(Normal.(cumsum(data.z) .* u[ib], 0.6), data.w))
    for u in ([0.0, 0.0], [0.2, -0.3], [-0.4, 0.1])
        check_consumer_point(problem, u, oracle)
    end
    emitted = BRM._rk_emit_ast(backend.plan)
    named(name) = count(d -> BRM._rk_source_definition(d).name === name, emitted.defs)
    @test named(:shared_response) == 1
    artifact = BRM.emit_rk_artifact(brmi; case_id="shared-helper-source")
    @test count(d -> BRM._rk_source_definition(d).name === :shared_response,
        artifact.defs) == 1
    rebuilt = build_kernel(BRM.rk_translate_artifact(artifact))
    @test coordinate_names(rebuilt.layout) == names
    # A different definition under the shared name still fails before evaluation.
    SharedHelperSource.SECOND_FACTOR[] = 2.0
    try
        @test_throws "with different definitions" BRM.emit_rk_artifact(
            SharedHelperSource.build(data); case_id="conflicting-helper-source")
    finally
        SharedHelperSource.SECOND_FACTOR[] = 1.0
    end
    @test isequal(data, saved)
end
