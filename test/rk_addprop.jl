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
        im = findfirst(==(Symbol("theta_Intercept")), names)
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
            # The cell route, and the inline route whose `addprop` scale is
            # row-local, observe their per-subject response per subject: one
            # array of densities per subject, in subject order. The assigned
            # scale is a whole value, so that route stays flattened.
            nested = route in (:cell, :inline)
            nested && @test length.(actual.y) == length.(data.y)
            flat = nested ? reduce(vcat, actual.y) : actual.y
            @test flat ≈ pointwise(u) atol=2e-11 rtol=2e-11
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

# Per-location scales: each observation selects its own additive and
# proportional scale (here by assay), written in the cell or joined per row.
module PublicPerLocationScale
using BayesianRegressionModels, Distributions
const data = (;
    subject=["b", "a", "c"],
    obs_subject=["a", "c", "b", "a", "b", "c"],
    obs_time=[0.5, 0.25, 1.5, 1.0, 3.0, 2.0],
    obs_assay=[0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    obs_sd=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    obs_limit=[0.05, 0.05, 0.1, 0.4, 0.05, 0.05],
    y=[0.81, 0.76, 0.52, 0.4, 0.21, 0.33],
)
const by_assay_add = "a1 .+ (a2 - a1) .* obs_assay[i]"
const by_assay_prop = "b1 .+ (b2 - b1) .* obs_assay[i]"

function build(route, add, prop)
    observation = route === :cell ? """
        @plate for i in eachindex(log_k)
            loc[i] = exp.(-exp(log_k[i]) .* obs_time[i])
            y[i] ~ censored(Normal(loc[i], addprop(loc[i], $add, $prop));
                lower=obs_limit[i])
        end
        """ : """
        @plate for i in eachindex(log_k)
            loc[i] = exp.(-exp(log_k[i]) .* obs_time[i])
        end
        ragged(y, obs_subject) ~ censored(Normal(loc, addprop(loc, $add, $prop));
            lower=obs_limit)
        """
    Core.eval(@__MODULE__, BayesianRegressionModels._brm("""
        a1 ~ Exponential(1.0)
        a2 ~ Exponential(1.0)
        b1 ~ Exponential(1.0)
        b2 ~ Exponential(1.0)
        log_k ~ 1 + (1 | p | subject)
        effect(log_k, Intercept) ~ Normal(0.0, 1.0)
        sd(:, p) ~ Exponential(1.0)
        ragged(obs_time, obs_subject)
        $(route === :cell ? """
        ragged(obs_assay, obs_subject)
        ragged(obs_limit, obs_subject)
        ragged(y, obs_subject)
        """ : "")
        $observation
        """; df=data))
end
end

@stestset "addprop takes per-location additive and proportional scales" begin
    P = PublicPerLocationScale
    data = P.data
    assay_add(s, j) = s.a1 + (s.a2 - s.a1) * data.obs_assay[j]
    assay_prop(s, j) = s.b1 + (s.b2 - s.b1) * data.obs_assay[j]
    cases = (
        # The cell route: both scales, either one, chosen per observation.
        (:cell, P.by_assay_add, P.by_assay_prop, assay_add, assay_prop, true),
        (:cell, P.by_assay_add, "b1", assay_add, (s, j) -> s.b1, true),
        (:cell, "a1", P.by_assay_prop, (s, j) -> s.a1, assay_prop, true),
        # A ragged join groups a per-row data scale with its rows.
        (:join, "obs_sd", "b1", (s, j) -> data.obs_sd[j], (s, j) -> s.b1, true),
        # RK also groups a joined scale computed from parameters and per-row
        # data; SBBRMI's twin is the `@test_broken` below.
        (:join, "a1 + (a2 - a1) * obs_assay", "b1", assay_add, (s, j) -> s.b1, false),
    )
    rows = [findall(==(s), data.obs_subject) for s in data.subject]
    order = [findfirst(==(s), sort(unique(data.subject))) for s in data.subject]
    for (route, add, prop, add_of, prop_of, stan_twin) in cases
        saved = deepcopy(data)
        brmi = P.build(route, add, prop)
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        # A scale no observation reads has no coordinate.
        scales = Tuple(filter(in(names), (:a1, :a2, :b1, :b2)))
        is = [findfirst(==(n), names) for n in scales]
        im = findfirst(==(:log_k_Intercept), names)
        it = findfirst(n -> occursin(".tau.", string(n)), names)
        iz = findall(n -> occursin(".z.", string(n)), names)
        N = length(names)
        @test N == length(scales) + 5 && length(iz) == 3
        function expected(u)
            s = NamedTuple{scales}(exp.(u[is]))
            map(eachindex(data.subject)) do i
                k = exp(u[im] + exp(u[it]) * u[iz][order[i]])
                map(rows[i]) do j
                    loc = exp(-k * data.obs_time[j])
                    # Independent scalar law; no BRM helper is called here.
                    a, b = add_of(s, j), prop_of(s, j)
                    law = Normal(loc, sqrt(a * a + loc * loc * b * b))
                    data.y[j] == data.obs_limit[j] ?
                        logcdf(law, data.obs_limit[j]) : logpdf(law, data.y[j])
                end
            end
        end
        positive = [is; it]
        oracle(u) = logpdf(Normal(), u[im]) + sum(logpdf.(Normal(), u[iz])) +
            sum(logpdf.(Exponential(), exp.(u[positive]))) + sum(u[positive]) +
            sum(sum, expected(u))
        if stan_twin
            sb = SBBRMI(brmi; mod=P, total_groups=())
            stan = BRM.stan_instantiate(sb;
                path=joinpath(mktempdir(), "addprop_per_location.stan"))
            transport = brm_coordinate_transport(backend, sb,
                BridgeStan.param_unc_names(stan.model))
            description = brm_description(sb)
            @test description.complete && isempty(description.diagnostics)
        end
        translated = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt).
            _rk_translated_plan(backend.plan)
        pointwise = prepare_query(backend.model, translated, :pointwise)
        for u in (zeros(N), collect(range(-0.3, 0.4; length=N)), fill(-0.2, N))
            value, gradient = check_consumer_point(problem, u, oracle)
            if stan_twin
                stan_u = brm_rk_point_to_stan(transport, u)
                stan_gradient = similar(stan_u)
                stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
                    stan_gradient; propto=false, jacobian=true)
                @test value ≈ stan_value + transport.logdensity_offset atol=2e-10 rtol=2e-10
                @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-8 rtol=2e-8
            end
            actual = Base.invokelatest(pointwise, u)
            @test length.(actual.y) == length.(rows)
            @test all(isapprox.(actual.y, expected(u); atol=2e-11, rtol=2e-11))
            # Every scale an observation reads stays active.
            @test all(abs.(gradient[is]) .> 1e-8)
        end
        model = brm_nlme_model(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        u = collect(range(-0.3, 0.4; length=N))
        eta = zeros(length(first(model.view.subject_coordinates)), length(data.subject))
        u[iz] .= 0
        by_level = [sum(expected(u)[only(findall(==(level), data.subject))])
            for level in model.view.levels]
        @test brm_nlme_loglikelihoods(model, u[model.theta], u[model.sigma], eta) ≈
            by_level atol=2e-11 rtol=2e-11
        @test isequal(saved, data)
    end
end

@stestset "addprop per-location scales outside a kernel" begin
    # A plain vector location takes per-row scales as they are.
    plain = @brm (; x=[0.1, 0.4, 0.9], w=[1.0, 2.0, 0.5], y=[0.2, 0.5, 1.1]) begin
        a ~ Exponential(1)
        b ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, addprop(mu, a * w, b * w))
    end
    code = BRM.stan_code(SBBRMI(plain; mod=@__MODULE__))
    @test occursin("addprop(mu, (a .* w), (b .* w))", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    # A joined scale computed from parameters and per-row data is grouped by
    # RK (above); StanBlocks has no arithmetic on a whole RaggedVector, so
    # SBBRMI needs the scale written in the cell (the `:cell` cases above).
    joined = PublicPerLocationScale.build(:join, "a1 + (a2 - a1) * obs_assay", "b1")
    @test_broken try
        StanBlocks.stanc_check(BRM.stan_code(SBBRMI(joined;
            mod=PublicPerLocationScale)); warn_pedantic=false).ok
    catch
        false
    end
end
