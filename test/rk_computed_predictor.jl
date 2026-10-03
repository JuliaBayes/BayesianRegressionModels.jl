# Ordinary formula design matrices retain their predictor axis before a
# whole-array reader gathers values onto the observation axis.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
StanBlocks.@deffun computed_public_gather(v::vector[n], indices::int[k]) = v[indices]
@inline computed_public_gather(v, indices) = v[indices]

function computed_predictor_model(data, intercept)
    if intercept
        @brm data begin
            a ~ 1 + x
            effect(a, :) ~ Normal(0, 1)
            mu = computed_public_gather(a, indices)
            y ~ Normal(mu, 1)
        end
    else
        @brm data begin
            a ~ 0 + x
            effect(a, :) ~ Normal(0, 1)
            mu = computed_public_gather(a, indices)
            y ~ Normal(mu, 1)
        end
    end
end

function computed_public_replay(backend)
    emitted = BRM._rk_emit_ast(backend.plan)
    mod = Module(gensym(:ComputedPredictorReplay))
    Core.eval(mod, :(using ReactiveKernelsPPL))
    for (name, callable) in emitted.bindings
        Core.eval(mod, Expr(:const, Expr(:(=), name, QuoteNode(callable))))
    end
    for definition in emitted.defs
        parsed = Meta.parse(sprint(Base.show_unquoted, definition))
        Core.eval(mod, Expr(:macrocall, Symbol("@rkppl"), LineNumberNode(0), parsed))
    end
    main = Meta.parse(sprint(Base.show_unquoted, emitted.main))
    data = backend.plan.columns
    plan = bind_data(lower_rkppl(main, data; mod, conditioned=(:y,)), data)
    built = build_kernel(plan)
    @test coordinate_names(built.layout) == coordinate_names(backend.model.layout)
    built, plan
end

@stestset "computed formula predictors and unchanged whole-array readers" begin
    for intercept in (false, true), same_axis in (true, false)
        data = same_axis ?
            (; x=[0.2, 0.5, -0.8], indices=[1, 2, 3], y=[0.3, 0.5, -0.2]) :
            (; x=[0.2, 0.5, -0.8], indices=[3, 1, 2, 3, 2], y=[0.3, 0.5, -0.2, 0.8, 0.1])
        before = deepcopy(data)
        brmi = computed_predictor_model(data, intercept)
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        @test Set(names) == Set(intercept ? [:a_Intercept, :a_x] : [:a_x])
        stan = consumer_stan(brmi, "computed-predictor-$intercept-$same_axis")
        # Stan stores the shared population block as a vector in its
        # Intercept/x design-column order; RK retains semantic scalar names.
        mapping = intercept ? [:a_Intercept => "pop_a_beta_pop.1",
            :a_x => "pop_a_beta_pop.2"] : [:a_x => "pop_a_beta_pop.1"]
        @test Set(BridgeStan.param_unc_names(stan.model)) == Set(last.(mapping))
        rebuilt, plan = computed_public_replay(backend)
        original = Base.get_extension(BRM,
            :BayesianRegressionModelsReactiveKernelsExt)._rk_translated_plan(backend.plan)
        function oracle(u)
            coefficients = Dict(name => u[j] for (j, name) in enumerate(names))
            mu = get(coefficients, :a_Intercept, 0.0) .+
                coefficients[:a_x] .* data.x[data.indices]
            residual = data.y .- mu
            -0.5 * (length(u) + length(data.y)) * log(2pi) -
                0.5sum(abs2, u) - 0.5sum(abs2, residual)
        end
        n = length(names)
        points = (zeros(n), fill(0.13, n), n == 1 ? [-0.2] : [-0.2, 0.3])
        for u in points
            value, gradient = check_consumer_point(problem, u, oracle)
            coefficients = Dict(name => u[j] for (j, name) in enumerate(names))
            mu = get(coefficients, :a_Intercept, 0.0) .+
                coefficients[:a_x] .* data.x[data.indices]
            residual = data.y .- mu
            analytic = [name == :a_x ? -coefficients[name] +
                sum(data.x[data.indices] .* residual) :
                -coefficients[name] + sum(residual) for name in names]
            @test gradient ≈ analytic atol=2e-12 rtol=2e-12
            check_consumer_stan(problem, stan, mapping, backend, u)
            for preset in (:sampler, :prior, :likelihood)
                a = Base.invokelatest(prepare_query(backend.model, original, preset), u)
                b = Base.invokelatest(prepare_query(rebuilt, plan, preset), u)
                @test isequal(a, b)
            end
        end
        @test isequal(data, before)
    end
end
