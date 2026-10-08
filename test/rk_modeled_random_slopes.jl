# Public synthetic modeled values in population and subject slope designs.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module PublicModeledRandomSlope
using BayesianRegressionModels, Distributions
function build(data)
    @brm data begin
        eta ~ 1 + (1 | location | subject)
        effect(eta, :) ~ Normal(-0.4, 0.7)
        sd(:, location) ~ Exponential(0.8)
        x = exp(eta)
        assay_scale ~ Exponential(0.6)
        assay ~ censored(LogNormal(eta, assay_scale); lower=limit)
        mu ~ 1 + x + (1 + x | effect | subject)
        effect(mu, :) ~ Normal(0.0, 1.2)
        sd(:, effect) ~ Exponential(0.9)
        cor(:, effect) ~ LKJCholesky(2, 2.5)
        sigma ~ Exponential(0.7)
        y ~ Normal(mu, sigma)
    end
end
end


@stestset "modeled random slopes preserve sampled design and conditional laws" begin
    original = (; subject=[1,1,2,2,3,3],
        assay=[0.2,0.7,0.5,0.2,0.8,0.9], limit=fill(0.2,6),
        y=[0.1,0.4,-0.2,0.3,0.7,0.5])
    for permutation in ([1,2,3,4,5,6], [2,3,5,1,6,4])
        data = map(column -> column[permutation], original)
        saved = deepcopy(data)
        brmi = PublicModeledRandomSlope.build(data)
        artifact = BRM.emit_rk_artifact(brmi; case_id="modeled-random-slope")
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        @test length(names) == 18
        @test !haskey(backend.plan.columns, :eta)
        @test !haskey(backend.plan.columns, :x)
        @test !haskey(backend.plan.columns, :mu)
        @test length(backend.plan.columns[:subject]) == 6
        @test length(backend.plan.regression.ranef_buckets) == 2
        effect = only(b for b in backend.plan.regression.ranef_buckets if b.id === :effect)
        @test [m.coefficient for m in effect.margins] == [:Intercept, :x]
        @test effect.margins[2].z.column === :x
        index(n) = only(findall(==(Symbol(n)), names))
        # The population component owns both ordinary design coefficients,
        # including the coefficient of the modeled column.
        beta = index.(["eta_Intercept", "pop_mu.beta_pop.1", "pop_mu.beta_pop.2"])
        location_scale = index("b_location_subject.tau.1")
        location_z = [index("b_location_subject.z.$j.1") for j in 1:3]
        scales = index.(["b_effect_subject.tau.1", "b_effect_subject.tau.2"])
        z = [index("b_effect_subject.z.$j.$k") for j in 1:3, k in 1:2]
        correlation = index("b_effect_subject.L.1")
        assay_scale, sigma = index.(["assay_scale", "sigma"])
        function components(u)
            rho, tau = tanh(u[correlation]), exp.(u[scales])
            L = [1.0 0.0; rho sqrt(1-rho^2)]
            random = u[z] * transpose(tau .* L)
            eta = u[beta[1]] .+ exp(u[location_scale]) .* u[location_z][data.subject]
            x = exp.(eta)
            mu = u[beta[2]] .+ u[beta[3]] .* x .+
                random[data.subject,1] .+ random[data.subject,2] .* x
            (; eta, x, mu, rho, tau,
                assay_scale=exp(u[assay_scale]), sigma=exp(u[sigma]))
        end
        function assay_parts(c)
            [data.assay[j] <= data.limit[j] ?
                logcdf(LogNormal(c.eta[j],c.assay_scale),data.limit[j]) :
                logpdf(LogNormal(c.eta[j],c.assay_scale),data.assay[j]) for j in 1:6]
        end
        function oracle(u)
            c = components(u)
            logpdf(Normal(-0.4,0.7),u[beta[1]]) +
                sum(logpdf.(Normal(0,1.2),u[beta[2:3]])) +
                sum(logpdf.(Normal(),u[location_z])) + sum(logpdf.(Normal(),u[z])) +
                logpdf(Exponential(0.8),exp(u[location_scale])) + u[location_scale] +
                sum(logpdf.(Exponential(0.9),c.tau) .+ u[scales]) +
                logpdf(Beta(2.5,2.5),(c.rho+1)/2) - log(2) + log1p(-c.rho^2) +
                logpdf(Exponential(0.6),c.assay_scale) + u[assay_scale] +
                logpdf(Exponential(0.7),c.sigma) + u[sigma] +
                sum(assay_parts(c)) + sum(logpdf.(Normal.(c.mu,c.sigma),data.y))
        end
        stan = consumer_stan(brmi,"modeled-random-slope-$(first(permutation))";
            mod=PublicModeledRandomSlope)
        mapping = [names[beta[1]]=>"pop_eta_beta_pop.1",
            names[beta[2]]=>"pop_mu_beta_pop.1", names[beta[3]]=>"pop_mu_beta_pop.2",
            names[location_scale]=>"b_location_subject_tau.1",
            names[correlation]=>"b_effect_subject_L.1",
            names[assay_scale]=>"assay_scale", names[sigma]=>"sigma"]
        append!(mapping,[names[location_z[j]]=>"b_location_subject_z_flat.$j" for j in 1:3])
        append!(mapping,[names[scales[k]]=>"b_effect_subject_tau.$k" for k in 1:2])
        append!(mapping,[names[z[j,k]]=>"b_effect_subject_z_flat.$(k+2*(j-1))" for j in 1:3 for k in 1:2])
        transport = brm_coordinate_transport(backend,
            SBBRMI(brmi; mod=PublicModeledRandomSlope, total_groups=()),
            BridgeStan.param_unc_names(stan.model))
        @test Dict(p.rk => p.stan for p in transport.pairs) == Dict(mapping)
        ext = Base.get_extension(BRM,:BayesianRegressionModelsReactiveKernelsExt)
        bound = ext._rk_translated_plan(backend.plan)
        pointwise = prepare_query(backend.model,bound,:pointwise)
        graph = ReactiveKernels.kernel_graph(backend.model.spec)
        println("VALUE_GRAPH_KINDS=", unique(entry.kind for entry in recipe_inventory(graph))); flush(stdout)
        translated = BRM.rk_translate_artifact(artifact)
        rebuilt = Base.invokelatest(build_kernel,translated)
        replay_pointwise = prepare_query(rebuilt, translated, :pointwise)
        replay = prepare_sampler(rebuilt,translated,zeros(18);
            backend=AutoEnzyme(; mode=Enzyme.Reverse))
        for u in (zeros(18),fill(0.13,18),collect(range(-0.2,0.3;length=18)))
            value, gradient = check_consumer_point(problem,u,oracle)
            check_consumer_stan(problem,stan,mapping,backend,u)
            c, parts = components(u), pointwise(u)
            @test parts.assay ≈ assay_parts(c)
            @test parts.y ≈ logpdf.(Normal.(c.mu,c.sigma),data.y)
            @test length.((parts.assay, parts.y)) == (6,6)
            @test isequal(parts, Base.invokelatest(replay_pointwise, u))
            replay_gradient = similar(u)
            replay_value, _ = sampler_value_and_gradient!(replay,replay_gradient,u)
            @test isequal(value,replay_value)
            @test gradient ≈ replay_gradient atol=2e-13 rtol=2e-13
        end
        @test isequal(data,saved)
    end
end

@stestset "direct formula and completed values remain active random slopes" begin
    data = (; subject=["b","a","b","c","a","c"],
        x=Union{Missing,Float64}[0.2,missing,-0.4,0.7,0.1,-0.2],
        y=[0.1,-0.2,0.5,0.2,-0.1,0.4])
    groups = [2,1,2,3,1,3]
    direct = @brm begin
        eta ~ 1 + (1 | location | subject)
        effect(eta, :) ~ Normal(-0.4,0.7)
        sd(:, location) ~ Exponential(0.8)
        mu ~ 0 + eta + (0 + eta | effect | subject)
        effect(mu, eta) ~ Normal(0,1.2)
        sd(:, effect) ~ Exponential(0.9)
        y ~ Normal(mu,1)
    end
    completed = @brm begin
        mi(x) ~ Normal(0,1)
        mu ~ 0 + x + (0 + x | effect | subject)
        effect(mu, x) ~ Normal(0,1.2)
        sd(:, effect) ~ Exponential(0.9)
        y ~ Normal(mu,1)
    end
    for (builder, slope) in ((direct,:eta), (completed,:x))
        saved = deepcopy(data)
        backend, problem = consumer_problem(builder(data))
        names = coordinate_names(backend.model.layout)
        index(n) = only(findall(==(Symbol(n)),names))
        beta = index("pop_mu.beta_pop.1")
        scale = index("b_effect_subject.tau.1")
        draws = [index("b_effect_subject.z.$j.1") for j in 1:3]
        @test !haskey(backend.plan.columns,slope)
        @test length(backend.plan.columns[:subject]) == 6
        @test backend.plan isa BRM._RKValuePlan
        function oracle(u)
            prior = logpdf(Normal(0,1.2),u[beta]) +
                logpdf(Exponential(0.9),exp(u[scale])) + u[scale] +
                sum(logpdf.(Normal(),u[draws]))
            value = if slope === :eta
                intercept = index("eta_Intercept")
                location_scale = index("b_location_subject.tau.1")
                location_z = [index("b_location_subject.z.$j.1") for j in 1:3]
                prior += logpdf(Normal(-0.4,0.7),u[intercept]) +
                    logpdf(Exponential(0.8),exp(u[location_scale])) + u[location_scale] +
                    sum(logpdf.(Normal(),u[location_z]))
                u[intercept] .+ exp(u[location_scale]) .* u[location_z][groups]
            else
                value = Float64[ismissing(data.x[j]) ? u[index("x.y_mis.1")] :
                    data.x[j] for j in 1:6]
                prior += sum(logpdf.(Normal(),value))
                value
            end
            mu = (u[beta] .+ exp(u[scale]) .* u[draws][groups]) .* value
            prior + sum(logpdf.(Normal.(mu,1),data.y))
        end
        for u in (zeros(length(names)),fill(0.13,length(names)),
                collect(range(-0.2,0.3;length=length(names))))
            check_consumer_point(problem,u,oracle)
        end
        @test isequal(data,saved)
    end
end
