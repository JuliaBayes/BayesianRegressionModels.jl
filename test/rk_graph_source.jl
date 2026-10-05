include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# A public compiler regression, not a private application or benchmark. The
# original Stan recurrence and the native scan have independently stated math.
module PublicGraphSource
using BayesianRegressionModels, Distributions, StanBlocks
import BayesianRegressionModels: _rk_callable_source!

StanBlocks.@deffun original_scan(x::vector[n], gain::real)::vector[n] = begin
    values = rep_vector(0.0, n)
    carry = 0.0
    for i in 1:n
        carry = carry + gain * x[i]
        values[i] = carry
    end
    values
end
StanBlocks.@deffun original_object_scan(x::vector[n], gain::real)::vector[n] =
    original_scan(x, gain)

function _rk_callable_source!(definitions, bindings, entry,
        ::typeof(original_scan))
    push!(definitions, :(ReactiveKernels.@kernel $entry(x, gain) = begin
        values = ReactiveKernels.scan(x, Ref(gain); init=0.0) do carry, value, g
            next = carry + g * value
            (next, next)
        end
        return values
    end))
    :done
end

function _rk_callable_source!(definitions, bindings, entry,
        ::typeof(original_object_scan))
    subject = Symbol(entry, :_subject)
    push!(definitions, :(ReactiveKernels.@kernel $subject(x, gain) = begin
        values = ReactiveKernels.scan(x, Ref(gain); init=0.0) do carry, value, g
            next = carry + g * value
            (next, next)
        end
        locations() = values
    end))
    # The source-hook adapter is itself a graph. It exposes the object's
    # endpoint without introducing a PreparedKernel call or runtime planning.
    push!(definitions, :(ReactiveKernels.@kernel $entry(x, gain) = begin
        values = $subject(x, gain).locations()
        return values
    end))
    :done
end

function build(data, route, object)
    if route === :kernel
        if object
            @brm data begin
                a ~ Normal(0, 0.7)
                loc ~ kernel(x) do xx
                    original_object_scan(xx, a)
                end
                y ~ Normal(loc, 0.8)
            end
        else
            @brm data begin
                a ~ Normal(0, 0.7)
                loc ~ kernel(x) do xx
                    original_scan(xx, a)
                end
                y ~ Normal(loc, 0.8)
            end
        end
    elseif object
        @brm data begin
            a ~ Normal(0, 0.7)
            loc = original_object_scan(x, a)
            y ~ Normal(loc, 0.8)
        end
    else
        @brm data begin
            a ~ Normal(0, 0.7)
            loc = original_scan(x, a)
            y ~ Normal(loc, 0.8)
        end
    end
end
function build_grouped(data)
    @brm data begin
        theta ~ 1 + (1 | p | subject)
        effect(theta, Intercept) ~ Normal(0, 0.7)
        sd(:, p) ~ Exponential(0.9)
        pred ~ kernel(t, theta) do ts, a
            original_object_scan(ts, a)
        end
        ragged(y, event_subject) ~ Normal(pred, 0.8)
    end
end
end

function graph_source_inventory(graph; depth=0)
    [(entry.kind, depth + entry.depth) for entry in ReactiveKernels.recipe_inventory(graph)
        if entry.kind !== :ordinary]
end

@stestset "graph provider source and printed definitions retain child recipes" begin
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    for object in (false, true)
        data = (; x=[[-0.4, 0.2], Float64[], [0.7]],
            y=[[0.1, -0.2], Float64[], [0.3]])
        before = deepcopy(data)
        plan = BRM._brm_rk_plan(PublicGraphSource.build(data, :kernel, object))
        emitted = BRM._rk_emit_ast(plan)
        parsed = BRM._RKEmittedProgram(
            [Meta.parse(sprint(Base.show_unquoted, d)) for d in emitted.defs],
            Meta.parse(sprint(Base.show_unquoted, emitted.main)), emitted.bindings)
        source_module = ext._rk_emit_module(emitted)
        replay_module = ext._rk_emit_module(parsed)
        readers = [BRM._rk_source_definition(d).name for d in emitted.defs
            if endswith(string(BRM._rk_source_definition(d).name), "_reader")]
        @test length(readers) == 1
        reader = only(readers)
        a, b = getfield(source_module, reader), getfield(replay_module, reader)
        @test graph_source_inventory(a.graph) == [(:plate, 0), (:scan, 1)]
        @test graph_source_inventory(b.graph) == graph_source_inventory(a.graph)
        @test isequal(prepare(a)(3, data.x, 0.2),
            reduce(vcat, [cumsum(x) .* 0.2 for x in data.x]))
        @test isequal(prepare(b)(3, data.x, 0.2), prepare(a)(3, data.x, 0.2))
        @test isequal(data, before)
    end
end

@stestset "provider subject graphs execute standard native Reverse" begin
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    for object in (false, true)
        data = (; x=[[-0.4, 0.2], Float64[], [0.7]],
            y=[[0.1, -0.2], Float64[], [0.3]])
        before = deepcopy(data)
        plan = BRM._brm_rk_plan(PublicGraphSource.build(data, :kernel, object))
        emitted = BRM._rk_emit_ast(plan)
        source_module = ext._rk_emit_module(emitted)
        reader = only([BRM._rk_source_definition(d).name for d in emitted.defs
            if endswith(string(BRM._rk_source_definition(d).name), "_reader")])
        score = Core.eval(source_module,
            :(ReactiveKernels.@kernel provider_score(count, x, a) = begin
                values = $reader(count, x, a)
                total = sum(values)
                return total
            end))
        @test graph_source_inventory(score.graph) == [(:plate, 0), (:scan, 1)]
        prepared = prepare_ad(prepare(score), AutoEnzyme(; mode=Enzyme.Reverse),
            3, data.x, 0.2; active=:a)
        derivative = sum(reduce(vcat, [cumsum(x) for x in data.x]))
        for a in (0.0, 0.2, -0.3)
            value, gradient = ad_value_and_gradient(prepared, 3, data.x, a)
            @test value ≈ derivative * a atol=1e-14
            @test gradient ≈ derivative atol=1e-14
            @test isequal(data, before)
        end
    end
end

@stestset "graph recurrence preserves the same-BRMI Stan science" begin
    for route in (:kernel, :assignment), object in (false, true)
        data = route === :kernel ?
            (; x=[[-0.4, 0.2], Float64[], [0.7]],
                y=[[0.1, -0.2], Float64[], [0.3]]) :
            (; x=[-0.4, 0.2, 0.7], y=[0.1, -0.2, 0.3])
        before = deepcopy(data)
        brmi = PublicGraphSource.build(data, route, object)
        stan = consumer_stan(brmi, "graph-source-$route-$object"; mod=PublicGraphSource)
        @test BridgeStan.param_unc_names(stan.model) == ["a"]
        xs = route === :kernel ? data.x : [data.x]
        x = reduce(vcat, [cumsum(values) for values in xs])
        y = route === :kernel ? reduce(vcat, data.y) : data.y
        for a in (0.0, 0.2, -0.3)
            u = [a]
            gradient = similar(u)
            value, _ = BridgeStan.log_density_gradient!(stan.model, u, gradient;
                propto=false, jacobian=true)
            oracle = logpdf(Normal(0, 0.7), a) +
                sum(logpdf.(Normal.(x .* a, 0.8), y))
            derivative = -a / 0.7^2 + sum((y .- a .* x) .* x) / 0.8^2
            @test value ≈ oracle atol=2e-11 rtol=2e-11
            @test only(gradient) ≈ derivative atol=2e-11 rtol=2e-11
            @test isequal(u, [a])
        end
        @test isequal(data, before)
    end
end

@stestset "source providers retain authored scans and subject graph plates" begin
    for route in (:kernel, :assignment), object in (false, true)
        data = route === :kernel ?
            (; x=[[-0.4, 0.2], Float64[], [0.7]],
                y=[[0.1, -0.2], Float64[], [0.3]]) :
            (; x=[-0.4, 0.2, 0.7], y=[0.1, -0.2, 0.3])
        saved = deepcopy(data)
        brmi = PublicGraphSource.build(data, route, object)
        backend = RKBRMI(brmi)
        inventory = graph_source_inventory(backend.model.spec.graph)
        println("GRAPH_SOURCE_INVENTORY=", (; route, object, inventory)); flush(stdout)
        # These checks inspect the actual posterior graph, not macro spellings
        # or a separately prepared provider graph.
        @test count(item -> first(item) === :scan, inventory) == 1
        route === :kernel && @test any(item -> item == (:scan, 1), inventory)
        route === :kernel && @test count(item -> first(item) === :plate, inventory) >= 2
        backend = check_rk_source_roundtrip(backend)
        problem = rk_logdensity_problem(backend;
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        @test coordinate_names(backend.model.layout) == [:a]
        xs = route === :kernel ? data.x : [data.x]
        y = route === :kernel ? reduce(vcat, data.y) : data.y
        oracle(u) = logpdf(Normal(0, 0.7), u[1]) +
            sum(logpdf.(Normal.(reduce(vcat, [cumsum(x) .* u[1] for x in xs]), 0.8), y))
        stan = consumer_stan(brmi, "graph-source-$route-$object"; mod=PublicGraphSource)
        for u in ([0.0], [0.2], [-0.3])
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, [:a => "a"], backend, u)
        end
        emitted = BRM._rk_emit_ast(backend.plan)
        @test all(d -> BRM._rk_source_definition(d).kind === :kernel, emitted.defs)
        @test !any(binding -> last(binding) === PublicGraphSource.original_scan ||
            last(binding) === PublicGraphSource.original_object_scan, emitted.bindings)
        artifact = BRM.emit_rk_artifact(brmi; case_id="graph-source-$route-$object")
        rebuilt = build_kernel(BRM.rk_translate_artifact(artifact))
        @test graph_source_inventory(rebuilt.spec.graph) == inventory
        @test isequal(data, saved)
    end
end

@stestset "explicit source graph definitions validate before evaluation" begin
    valid = :(ReactiveKernels.@kernel cell(x) = begin result = x + 1; return result; end)
    @test BRM._rk_source_definition(valid) == (; name=:cell, kind=:kernel)
    parsed = Meta.parse(sprint(Base.show_unquoted, valid))
    @test BRM._rk_source_definition(parsed) == (; name=:cell, kind=:kernel)
    program(defs, bindings=Pair{Symbol,Any}[]) =
        BRM._RKEmittedProgram(defs, Expr(:block), bindings)
    @test_throws "defined more than once" BRM._rk_validate_source_definitions(program([valid, valid]))
    @test_throws "both bound and defined" BRM._rk_validate_source_definitions(program([valid], [:cell => identity]))
    @test_throws "explicit definitions" BRM._rk_source_definition(:(@eval cell(x) = x))
end

@stestset "computed grouped predictor ports retain the subject scan and empty-group prior" begin
    data = (; subject=["b", "empty", "a"],
        t=[[0.2, 0.5], Float64[], [0.7]],
        y=[0.1, 0.0, 0.4], event_subject=["a", "b", "b"])
    before = deepcopy(data)
    brmi = PublicGraphSource.build_grouped(data)
    backend, problem = consumer_problem(brmi)
    inventory = graph_source_inventory(ReactiveKernels.kernel_graph(backend.model.spec))
    @test (:scan, 1) in inventory
    @test count(item -> first(item) === :scan, inventory) == 1
    names = coordinate_names(backend.model.layout)
    @test length(names) == 5
    index(name) = only(findall(==(Symbol(name)), names))
    intercept = index("pop_theta.beta_pop.1")
    scale = index("b_p_subject.tau.1")
    levels = CategoricalArrays.levels(data.subject)
    innovations = [index("b_p_subject.z.$j.1") for j in eachindex(levels)]
    rows = [only(findall(==(subject), levels)) for subject in data.subject]
    grouped_y = reduce(vcat, [data.y[findall(==(subject), data.event_subject)]
        for subject in data.subject])
    oracle(u) = begin
        tau = exp(u[scale])
        theta = u[intercept] .+ tau .* u[innovations[rows]]
        locations = reduce(vcat, [cumsum(data.t[j]) .* theta[j] for j in 1:3])
        logpdf(Normal(0,0.7),u[intercept]) +
            logpdf(Exponential(0.9),tau) + u[scale] +
            sum(logpdf.(Normal(),u[innovations])) +
            sum(logpdf.(Normal.(locations,0.8),grouped_y))
    end
    stan = consumer_stan(brmi,"graph-grouped-predictor"; mod=PublicGraphSource)
    mapping = [names[intercept] => "pop_theta_beta_pop.1",
        names[scale] => "b_p_subject_tau.1"]
    append!(mapping,[names[innovations[j]] => "b_p_subject_z_flat.$j" for j in eachindex(levels)])
    empty_coordinate = innovations[only(findall(==("empty"), levels))]
    for u in (zeros(5),collect(range(-0.2,0.3;length=5)),fill(-0.1,5))
        _, gradient = check_consumer_point(problem,u,oracle)
        check_consumer_stan(problem,stan,mapping,backend,u)
        @test gradient[empty_coordinate] ≈ -u[empty_coordinate] atol=2e-11
    end
    artifact = BRM.emit_rk_artifact(brmi;case_id="graph-grouped-predictor")
    rebuilt = build_kernel(BRM.rk_translate_artifact(artifact))
    @test graph_source_inventory(ReactiveKernels.kernel_graph(rebuilt.spec)) == inventory
    @test isequal(data,before)
end
