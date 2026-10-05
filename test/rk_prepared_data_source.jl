# Public synthetic preparation: fitted metadata fixes the coordinate geometry;
# executable indices, dummies and completion contributions live in source.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

@stestset "data preparation kernels retain named recipes and printed replay" begin
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    definitions, taken = Expr[], Set{Symbol}()
    for name in (:brm_prepared_indices, :brm_factor_dummy, :brm_covariate_geometry,
            :brm_covariate_observed, :brm_covariate_observed_rows, :brm_covariate_missing_rows,
            :brm_matrix_column, :brm_flatten_response, :brm_gather_response)
        BRM._rk_ast_statistical_call!(definitions, taken, name; kernel=true)
    end
    emitted = BRM._RKEmittedProgram(definitions, Expr(:block))
    parsed = BRM._RKEmittedProgram(
        [Meta.parse(sprint(Base.show_unquoted, d)) for d in definitions], Expr(:block))
    for mod in (ext._rk_emit_module(emitted), ext._rk_emit_module(parsed))
        indices = getfield(mod, :brm_prepared_indices)
        dummy = getfield(mod, :brm_factor_dummy)
        geometry = getfield(mod, :brm_covariate_geometry)
        @test [entry.kind for entry in recipe_inventory(indices.graph)] == [:plate]
        @test [entry.kind for entry in recipe_inventory(dummy.graph)] == [:plate]
        @test count(entry -> entry.kind === :plate, recipe_inventory(geometry.graph)) == 3
        @test Base.invokelatest(prepare(indices), ["a", "b", "a"],
            ["b", "unused", "a"]) == [3, 1, 3]
        @test Base.invokelatest(prepare(dummy), ["a", "b", "a"], "a") == [1., 0., 1.]
        query = prepare(geometry)
        raw = Union{Missing, Float64}[3., missing, 7., missing]
        @test Base.invokelatest(prepare(getfield(mod, :brm_covariate_observed)), raw) == [3., 7.]
        @test Base.invokelatest(prepare(getfield(mod, :brm_covariate_observed_rows)), raw) == [1, 3]
        @test Base.invokelatest(prepare(getfield(mod, :brm_covariate_missing_rows)), raw) == [2, 4]
        @test Base.invokelatest(prepare(getfield(mod, :brm_matrix_column)), [[1 2; 3 4]], 2) == [2, 4]
        @test Base.invokelatest(prepare(getfield(mod, :brm_flatten_response)),
            [[.2, -.3], Float64[], [.4]]) == [.2, -.3, .4]
        @test Base.invokelatest(prepare(getfield(mod, :brm_gather_response)),
            [.2, -.3, .4], [3, 1, 2]) == [.4, .2, -.3]
        for (observed, jobs, jmis, expected) in (
                ([3., 7.], [1, 3], [2, 4], ([3., 0., 7., 0.], [1, 1, 1, 2], [0., 1., 0., 1.])),
                ([3., 5., 7.], [1, 2, 3], Int[], ([3., 5., 7.], [1, 1, 1], zeros(3))),
                (Float64[], Int[], [1, 2], (zeros(2), [1, 2], ones(2))),
                (Float64[], Int[], Int[], (Float64[], Int[], Float64[])))
            saved = deepcopy((observed, jobs, jmis))
            @test Base.invokelatest(query, observed, jobs, jmis) == expected
            @test isequal((observed, jobs, jmis), saved)
        end
    end
end

@stestset "all-missing covariate source retains every latent law" begin
    data = (; x=Union{Missing, Float64}[missing, missing], y=[.2, -.3])
    saved = deepcopy(data)
    brmi = @brm data begin
        mi(x) ~ Normal(0, 1)
        mu ~ 1 + x
        y ~ Normal(mu, 1)
    end
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    index(n) = only(findall(==(Symbol(n)), names))
    a, b = index("mu_Intercept"), index("mu_x")
    latent = [index("x.y_mis.$j") for j in 1:2]
    @test length(names) == 4
    oracle(u) = sum(logpdf.(Normal(), u)) +
        sum(logpdf.(Normal.(u[a] .+ u[b] .* u[latent], 1), data.y))
    for u in (zeros(4), fill(.13, 4), collect(range(-.2, .3; length=4)))
        check_consumer_point(problem, u, oracle)
    end
    @test isequal(data, saved)
end

@stestset "weighted regression artifacts retain their normalized law" begin
    data = (; x=[-.4, .2, .7, 1.1], y=[.2, -.1, .4, .3], n=[1, 2, 1, 3])
    saved = deepcopy(data)
    brmi = @brm data begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        y ~ weighted(Normal(mu, sigma), fweights(n))
    end
    artifact = emit_rk_artifact(brmi; case_id="weighted-regression-preparation")
    translated = rk_translate_artifact(artifact)
    model = build_kernel(translated)
    @test coordinate_names(model.layout) == [:mu_Intercept, :mu_x, :sigma]
    oracle(u) = sum(data.n .* logpdf.(Normal.(u[1] .+ u[2] .* data.x, exp(u[3])), data.y)) +
        sum(logpdf.(Normal(), u[1:2])) + logpdf(Exponential(), exp(u[3])) + u[3]
    problem = prepare_sampler(model, translated, zeros(3);
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for u in (zeros(3), [.13, -.2, .3], [-.4, .2, -.1])
        gradient = similar(u)
        value, _ = sampler_value_and_gradient!(problem, gradient, u)
        @test value ≈ oracle(u) atol=2e-11 rtol=2e-11
        step = 1e-5
        independent = map(eachindex(u)) do j
            plus, minus = copy(u), copy(u)
            plus[j] += step
            minus[j] -= step
            (oracle(plus) - oracle(minus)) / (2step)
        end
        @test gradient ≈ independent atol=2e-8 rtol=2e-8
    end
    @test isequal(data, saved)
end

@stestset "cached preparation cannot override emitted categorical and ordinal values" begin
    groups = categorical(["b", "a", "b", "a"])
    levels!(groups, ["b", "unused", "a"])
    data = (; g=groups, rank=[1, 3, 2, 1], x=[-.7, -.2, .4, .9], y=[.2, -.1, .3, .4])
    saved = deepcopy(data)
    brmi = @brm data begin
        mu ~ 1 + factor(g; ref="a") + mo(rank) + hsgp(x; k=3, by=g)
        y ~ Normal(mu, 1)
    end
    backend, problem = consumer_problem(brmi)
    artifact = BRM.emit_rk_artifact(brmi; case_id="prepared-source-inputs")
    emitted = BRM._rk_emit_ast(backend.plan)
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    prepared = Symbol[]
    for term in only(backend.plan.predictors).terms
        term.kind === :factor && append!(prepared, term.options.design_columns)
        term.kind === :monotonic && append!(prepared, term.columns)
        term.kind === :hsgp && push!(prepared, term.options.group_index)
    end
    @test !isempty(prepared)
    bound_inputs = BRM._rk_source_data_columns(backend.plan, emitted)
    @test isequal(BRM.rk_artifact_inputs(artifact), bound_inputs)
    @test all(name -> !haskey(bound_inputs, name), prepared)
    for name in prepared
        backend.plan.columns[name] = ones(eltype(backend.plan.columns[name]), length(data.y))
    end
    translated = ext._rk_translate_from_emitted(backend.plan, emitted)
    model = Base.invokelatest(build_kernel, translated)
    @test coordinate_names(model.layout) == coordinate_names(backend.model.layout)
    n = model.layout.total
    replay = prepare_sampler(model, translated, zeros(n); backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for u in (zeros(n), fill(.13, n), collect(range(-.2, .3; length=n)))
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        rg = similar(u)
        rv, _ = sampler_value_and_gradient!(replay, rg, u)
        @test rv == value
        @test rg == gradient
    end
    @test isequal(data, saved)
end
