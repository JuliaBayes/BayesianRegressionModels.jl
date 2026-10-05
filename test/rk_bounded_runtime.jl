# Run after the generic restricted(D, lo, hi) producer capability lands.
# Keep all original declarations, family kernels and ordinary reverse AD.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
include(joinpath(@__DIR__, "rk_bounded_fixtures.jl"))

bounded_sigmoid(q) = 1 / (1 + exp(-q))
bounded_interval_jac(q, width) =
    log(width) + log(bounded_sigmoid(q)) + log1p(-bounded_sigmoid(q))
bounded_normal_oracle(u) = begin
    scale = exp(only(u))
    logpdf(Normal(), scale) + sum(logpdf.(Normal(0, scale), bounded_data.y)) + only(u)
end
bounded_df_oracle(u) = begin
    invdf = 0.5bounded_sigmoid(only(u))
    distribution = TDist(1 / invdf)
    logpdf(Exponential(0.125), invdf) +
        logcdf(distribution, -0.7) + logpdf(distribution, 0.2) +
        logccdf(distribution, 0.6) + bounded_interval_jac(only(u), 0.5)
end

function bounded_public_replay(backend)
    emitted = BRM._rk_emit_ast(backend.plan)
    mod = Module(gensym(:BoundedKernelReplay))
    Core.eval(mod, :(using ReactiveKernelsPPL))
    for (name, callable) in emitted.bindings
        Core.eval(mod, Expr(:const, Expr(:(=), name, QuoteNode(callable))))
    end
    Core.eval(mod, :(import ReactiveKernels))
    for definition in emitted.defs
        parsed = Meta.parse(sprint(Base.show_unquoted, definition))
        # Explicit `@kernel` graphs and functions keep their own kind; a bare
        # function-shaped definition is an `@rkppl` submodel.
        Core.eval(mod, BRM._rk_source_definition(definition).kind === :rkppl ?
            Expr(:macrocall, Symbol("@rkppl"), LineNumberNode(0), parsed) : parsed)
    end
    source = Meta.parse(sprint(Base.show_unquoted, emitted.main))
    data = backend.plan.columns
    plan = bind_data(lower_rkppl(source, data; mod, conditioned=(:y,)), data)
    built = build_kernel(plan)
    @test coordinate_names(built.layout) == coordinate_names(backend.model.layout)
    (built, plan)
end

function check_bounded_case(builder, data, label, oracle; offset=0.0)
    before = deepcopy(data)
    brmi = builder(data)
    backend, problem = consumer_problem(brmi)
    stan = consumer_stan(brmi, "bounded-" * label)
    names = coordinate_names(backend.model.layout)
    mapping = [name => string(name) for name in names]
    rebuilt, plan = bounded_public_replay(backend)
    original = Base.get_extension(BRM,
        :BayesianRegressionModelsReactiveKernelsExt)._rk_translated_plan(backend.plan)
    for q in (0.0, 0.3, -0.1)
        u = fill(q, length(names))
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, mapping, backend, u; density_offset=offset)
        for preset in (:sampler, :prior, :likelihood)
            a = Base.invokelatest(prepare_query(backend.model, original, preset), u)
            b = Base.invokelatest(prepare_query(rebuilt, plan, preset), u)
            @test isequal(a, b)
        end
    end
    @test isequal(data, before)
    backend, problem
end

@stestset "original bounded scalar priors and complete source replay" begin
    normal, normal_problem = check_bounded_case(bounded_normal_builder,
        bounded_data, "normal", bounded_normal_oracle)
    check_bounded_case(bounded_df_builder, bounded_data, "student-df", bounded_df_oracle)
    normalized = RKBRMI(bounded_normalized_builder(bounded_data))
    normalized_problem = rk_logdensity_problem(normalized;
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    @test coordinate_names(normal.model.layout) == coordinate_names(normalized.model.layout)
    for q in (0.0, 0.3, -0.1)
        v, g = LogDensityProblems.logdensity_and_gradient(normal_problem, [q])
        vn, gn = LogDensityProblems.logdensity_and_gradient(normalized_problem, [q])
        @test vn - v ≈ log(2) atol=2e-11 rtol=2e-11
        @test gn ≈ g atol=2e-10 rtol=2e-10
    end
end

@stestset "one-sided finite and observed scalar bounds" begin
    upper_oracle(u) = begin
        location = 0.5 - exp(only(u))
        logpdf(Normal(), location) +
            sum(logpdf.(Normal(location, 1), bounded_data.y)) + only(u)
    end
    check_bounded_case(bounded_upper_builder, bounded_data, "upper", upper_oracle)
    shifted_oracle(u) = begin
        value = 0.2 + exp(only(u))
        logpdf(Normal(0.4, 0.7), value) +
            sum(logpdf.(Normal(value, 1), bounded_data.y)) + only(u)
    end
    check_bounded_case(bounded_shifted_builder, bounded_data, "shifted", shifted_oracle)
    for (lo, hi) in ((0.0, 1.5), (0.2, 2.0))
        data = merge(bounded_data, (; lower_bound=lo, upper_bound=hi))
        oracle(u) = begin
            scale = lo + (hi-lo)bounded_sigmoid(only(u))
            logpdf(Normal(), scale) + sum(logpdf.(Normal(0, scale), data.y)) +
                bounded_interval_jac(only(u), hi-lo)
        end
        check_bounded_case(bounded_data_builder, data, "data-$lo", oracle)
    end
    for (builder, family, lo, hi, label) in (
            (bounded_student_prior_builder, TDist(6.0), -0.5, 1.5, "student-prior"),
            (bounded_beta_interval_builder, Beta(2.0, 3.0), 0.2, 0.8, "beta"))
        oracle(u) = begin
            value = lo + (hi-lo)bounded_sigmoid(only(u))
            logpdf(family, value) + sum(logpdf.(Normal(value, 1), bounded_data.y)) +
                bounded_interval_jac(only(u), hi-lo)
        end
        check_bounded_case(builder, bounded_data, label, oracle)
    end
end

@stestset "original scalar Student-t control" begin
    scalar_oracle(u) = begin
        distribution = LocationScale(only(u), 1, TDist(bounded_data.nu))
        logpdf(Normal(), only(u)) + logcdf(distribution, -0.7) +
            logpdf(distribution, 0.2) + logccdf(distribution, 0.6)
    end
    check_bounded_case(bounded_scalar_df_builder, bounded_data, "df-control", scalar_oracle)
end
