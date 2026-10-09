include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# Emitted RK source names each opaque callable after the callable itself, so
# a reader sees `addprop` and `cell_decay` rather than a generic stem. A name
# the program already reads, or one with a fixed meaning in RK source, takes
# a fresh spelling instead (snag rk-emission-name-7b099eb9).
module PublicCallableNames
using BayesianRegressionModels, Distributions
const scale_alias = addprop
cell_decay(t, dose, k) = dose .* exp.(-k .* t)
# Caller-owned, unlike RKPPL's built-in reduction of the same spelling.
mean(v) = 2 .* v
const anonymous_shift = (v, s) -> v .+ s

reader(data) = @brm data begin
    log_k ~ Normal(0, 1)
    a ~ Exponential(1)
    b ~ Exponential(1)
    @plate for i in eachindex(t)
        loc[i] = cell_decay(t[i], dose[i], exp(log_k))
    end
    y ~ Normal(loc, scale_alias(loc, a, b))
end

fixed(data) = @brm data begin
    a ~ Normal(0, 1)
    reads = anonymous_shift(mean(x), a)
    y ~ Normal(reads, 0.7)
end
end

source_text(emitted) = join((sprint(Base.show_unquoted, d) for d in emitted.defs), "\n") *
    "\n" * sprint(Base.show_unquoted, emitted.main)

@stestset "authored callables keep their own names in emitted RK source" begin
    data = (; t=[[0.5, 1.5], Float64[], [0.7, 1.1, 2.0]], dose=[1.0, 2.0, 1.5],
        y=[[0.6, 0.2], Float64[], [1.0, 0.8, 0.3]])
    saved = deepcopy(data)
    brmi = PublicCallableNames.reader(data)
    backend, problem = consumer_problem(brmi)
    emitted = BRM._rk_emit_ast(backend.plan)
    source = source_text(emitted)
    @test !occursin("brm_value_function", source)
    @test !occursin("#brm_callable", source)
    # The BRM provider receives `addprop` as its entry and defines it.
    @test any(d -> BRM._rk_source_definition(d) == (; name=:addprop, kind=:kernel), emitted.defs)
    @test occursin("addprop(loc", source)
    # The cell's own callable is bound under its own name inside the reader.
    @test (:cell_decay => PublicCallableNames.cell_decay) in emitted.bindings
    @test occursin("cell_decay(t, dose, exp(log_k))", source)
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    @test getfield(ext._rk_emit_module(emitted), :cell_decay) === PublicCallableNames.cell_decay

    names = coordinate_names(backend.model.layout)
    ik, ia, ib = (findfirst(==(n), names) for n in (:log_k, :a, :b))
    @test length(names) == 3
    oracle(u) = begin
        k, a, b = exp(u[ik]), exp(u[ia]), exp(u[ib])
        logpdf(Normal(), u[ik]) + sum(logpdf.(Exponential(), (a, b))) + u[ia] + u[ib] +
            sum(sum(logpdf.(Normal.(m, sqrt.(a^2 .+ (m .* b).^2)), y); init=0.0)
                for (m, y) in ((dose .* exp.(-k .* t), y)
                    for (t, dose, y) in zip(data.t, data.dose, data.y)))
    end
    for u in (zeros(3), [0.3, -0.2, 0.1], [-0.4, 0.25, -0.3])
        check_consumer_point(problem, u, oracle)
    end
    artifact = BRM.emit_rk_artifact(brmi; case_id="callable-names-reader")
    @test artifact.bindings == emitted.bindings
    rebuilt = build_kernel(BRM.rk_translate_artifact(artifact))
    @test coordinate_names(rebuilt.layout) == names
    @test isequal(data, saved)
end

@stestset "fixed RK source names and anonymous callables take other names" begin
    data = (; x=[-0.4, 0.2, 0.9], y=[0.1, 0.4, -0.2])
    brmi = PublicCallableNames.fixed(data)
    backend, problem = consumer_problem(brmi)
    emitted = BRM._rk_emit_ast(backend.plan)
    # A caller's `mean` must not be read as RKPPL's built-in reduction.
    @test (:mean_ => PublicCallableNames.mean) in emitted.bindings
    @test (:brm_value_function => PublicCallableNames.anonymous_shift) in emitted.bindings
    @test !any(p -> first(p) === :mean, emitted.bindings)
    names = coordinate_names(backend.model.layout)
    @test names == [:a]
    oracle(u) = logpdf(Normal(), u[1]) + sum(logpdf.(Normal.(2 .* data.x .+ u[1], 0.7), data.y))
    for u in ([0.0], [0.35], [-0.6])
        check_consumer_point(problem, u, oracle)
    end
end

@stestset "callable naming reserves fixed RK source meanings" begin
    free(name, callable) = BRM._rk_callable_name_free(name, callable)
    named(callable) = BRM._rk_callable_own_name(callable)
    @test named(PublicCallableNames.cell_decay) === :cell_decay
    @test named(PublicCallableNames.scale_alias) === :addprop
    @test named(PublicCallableNames.anonymous_shift) === nothing
    let offset = 0.3
        @test named(x -> x + offset) === nothing
    end
    @test named(Base.Fix1(+, 1)) === nothing
    # A name with a fixed meaning names only that meaning.
    @test free(:logistic, BRM.logistic)
    @test !free(:logistic, PublicCallableNames.cell_decay)
    @test free(:cumsum, cumsum)
    @test !free(:cumsum, PublicCallableNames.cell_decay)
    @test !free(:mean, PublicCallableNames.mean)
    @test free(:cell_decay, PublicCallableNames.cell_decay)
    for imported in (:plate, :brm_flatten_cells, :brm_invprobit, :ReactiveKernels,
            :BayesianRegressionModels, :ReactiveKernelsPPL)
        @test !free(imported, BRM.brm_flatten_cells)
    end
    # RKPPL's built-in call vocabulary is reserved whatever binds it.
    vocabulary = Set{Symbol}(Symbol(name) for name in Iterators.flatten((
        ReactiveKernelsPPL.admitted_functions(),
        Iterators.flatten(ReactiveKernelsPPL.admitted_elementwise())))
        if !startswith(string(name), "."))
    for name in vocabulary
        @test !free(name, PublicCallableNames.cell_decay)
    end
end
