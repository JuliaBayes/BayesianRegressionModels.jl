include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module PublicAdditiveProportionalScale
using BayesianRegressionModels, Distributions
const scale_alias = addprop

function build(data, route)
    priors = "a ~ Exponential(1.0)\nb ~ Exponential(1.0)\n"
    if route === :scalar
        return Core.eval(@__MODULE__, BayesianRegressionModels._brm(priors * """
            theta ~ Normal(0.0, 1.0)
            reads = theta + x
            y ~ Normal(reads, scale_alias(reads, a, b))
            """; df=data))
    end
    tail = route === :assignment ? """
        scales = scale_alias(reads, a, b)
        y ~ censored(Normal(reads, scales); lower=lo)
        """ : "y ~ censored(Normal(reads, addprop(reads, a, b)); lower=lo)"
    cell = route === :cell ? "scale_alias(xs .+ t, a, b)" : "xs .+ t"
    route === :cell && (tail = "y ~ Normal(reads, 0.8)")
    Core.eval(@__MODULE__, BayesianRegressionModels._brm(priors * """
        theta ~ 1 + (1 | p | subject)
        effect(theta, Intercept) ~ Normal(0.0, 1.0)
        sd(:, p) ~ Exponential(1.0)
        reads ~ kernel(x, theta) do xs, t
            $cell
        end
        $tail
        """; df=data))
end
end

function scale_pointwise(means, a, b, observed, lower; censored_response=true)
    # Independent scalar law; no BRM helper or emitted body is called here.
    map(eachindex(observed)) do i
        sigma = sqrt(a * a + means[i] * means[i] * b * b)
        d = Normal(means[i], sigma)
        censored_response && observed[i] == lower[i] ?
            logcdf(d, lower[i]) : logpdf(d, observed[i])
    end
end

@stestset "addprop graph source preserves grouped scale activity and censoring" begin
    cases = (
        (:inline, (; subject=[1, 2, 3], x=[[0.1, 0.3], [0.2], [0.5]],
            y=[[0.0, 0.3], [0.2], [0.5]], lo=[[0.0, 0.0], [0.0], [0.0]])),
        (:assignment, (; subject=["b", "empty", "a"],
            x=[[0.2, 0.6], Float64[], [0.4]],
            y=[[-0.1, 0.5], Float64[], [0.2]],
            lo=[[-0.1, 0.0], Float64[], [0.0]])),
        (:cell, (; subject=["b", "empty", "a"],
            x=[[0.2, 0.6], Float64[], [0.4]],
            y=[[0.1, -0.2], Float64[], [0.3]])),
    )
    for (route, data) in cases
        saved = deepcopy(data)
        brmi = PublicAdditiveProportionalScale.build(data, route)
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        ia, ib = findfirst(==(:a), names), findfirst(==(:b), names)
        im = findfirst(==(Symbol("pop_theta.beta_pop.1")), names)
        it = findfirst(n -> occursin(".tau.", string(n)), names)
        iz = findall(n -> occursin(".z.", string(n)), names)
        @test length(names) == 7
        @test length(iz) == 3
        order = [findfirst(==(s), sort(unique(data.subject))) for s in data.subject]
        observed = reduce(vcat, data.y)
        lower = route === :cell ? zeros(length(observed)) : reduce(vcat, data.lo)
        function pointwise(u)
            a, b, tau = exp(u[ia]), exp(u[ib]), exp(u[it])
            theta = u[im] .+ tau .* u[iz][order]
            means = reduce(vcat, [data.x[j] .+ theta[j] for j in eachindex(data.x)])
            route === :cell ? logpdf.(Normal.(sqrt.(a*a .+ (means .* b).^2), 0.8), observed) :
                scale_pointwise(means, a, b, observed, lower)
        end
        oracle(u) = logpdf(Normal(), u[im]) + sum(logpdf.(Normal(), u[iz])) +
            sum(logpdf.(Exponential(), exp.(u[[ia, ib, it]]))) +
            sum(u[[ia, ib, it]]) + sum(pointwise(u))
        stan = consumer_stan(brmi, "addprop-$route"; mod=PublicAdditiveProportionalScale)
        mapping = vcat([names[ia] => "a", names[ib] => "b",
            names[im] => "pop_theta_beta_pop.1", names[it] => "b_p_subject_tau.1"],
            [names[iz[j]] => "b_p_subject_z_flat.$j" for j in eachindex(iz)])
        translated = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt).
            _rk_translated_plan(backend.plan)
        for u in (zeros(7), collect(range(-0.3, 0.4; length=7)), fill(-0.2, 7))
            _, gradient = check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, mapping, backend, u)
            actual = Base.invokelatest(prepare_query(backend.model, translated, :pointwise), u)
            @test actual.y ≈ pointwise(u) atol=2e-11 rtol=2e-11
            @test abs(gradient[ia]) > 1e-8
            @test abs(gradient[ib]) > 1e-8
        end
        emitted = BRM._rk_emit_ast(backend.plan)
        @test !any(p -> last(p) === addprop, emitted.bindings)
        @test any(d -> BRM._rk_source_definition(d).kind === :kernel &&
            isequal(last(d.args).args[2], Expr(:block, Expr(:return,
                BRM._rk_ast_dotted(:sqrt, BRM._BRM_ADDPROP_VARIANCE)))), emitted.defs)
        # Artifact translation resolves the same provider through its public seam.
        artifact = BRM.emit_rk_artifact(brmi; case_id="addprop-$route")
        rebuilt = build_kernel(BRM.rk_translate_artifact(artifact))
        @test coordinate_names(rebuilt.layout) == names
        @test isequal(saved, data)
    end
end

@stestset "addprop source handles a scalar location without changing reader geometry" begin
    data = (; x=0.3, y=[-0.2, 0.1, 0.4])
    saved = deepcopy(data)
    brmi = PublicAdditiveProportionalScale.build(data, :scalar)
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    ia, ib, im = (findfirst(==(n), names) for n in (:a, :b, :theta))
    @test length(names) == 3
    oracle(u) = begin
        a, b, mu = exp(u[ia]), exp(u[ib]), u[im] + data.x
        logpdf(Normal(), u[im]) - a - b + u[ia] + u[ib] +
            sum(logpdf.(Normal(mu, sqrt(a*a + mu*mu*b*b)), data.y))
    end
    # The scalar native entry is independent of Stan's vector-only helper
    # signature; the grouped cases above compile the original Stan contract.
    for u in (zeros(3), [0.2, -0.3, 0.4], [-0.1, 0.2, -0.3])
        check_consumer_point(problem, u, oracle)
    end
    @test isequal(saved, data)
end
