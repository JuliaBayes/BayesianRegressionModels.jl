# Named pointwise modeled values retain formula geometry and emitted graph math.
# Run: julia --project=test test/rk_modeled_transform_predictors.jl [filter ...]
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module PublicModeledTransform
using BayesianRegressionModels, Distributions
function build(data)
    @brm data begin
        mu_w ~ Normal(3.0, 1.0)
        sigma_w ~ LogNormal(-1.0, 0.3)
        mi(w) ~ LogNormal(mu_w, sigma_w)
        mu_x ~ Normal(1.0, 1.0)
        sigma_x ~ LogNormal(-1.0, 0.3)
        raw_x ~ LogNormal(mu_x, sigma_x)
        transformed_x = log(raw_x * w)
        log(a) ~ 1 + transformed_x + (1 | p | subject)
        log(b) ~ 1 + (1 | p | subject)
        sd(:, p) ~ Exponential(1.0)
        cor(:, p) ~ LKJCholesky(2, 2.0)
        scale ~ Exponential(1.0)
        reads ~ kernel(rows, values, a, b) do indices, vv, aa, bb
            location = aa .+ bb .* vv
            vv ~ normal(location, scale)
            location[indices]
        end
        y ~ Normal(reads, scale)
    end
end
end

function modeled_location_in_graph(graph)
    any(recipe_inventory(graph)) do entry
        any(output -> occursin("location",string(output.name)),entry.recipe.outputs)
    end
end

@stestset "modeled transforms preserve subject geometry and all conditional laws" begin
    for rows in ([[1,2],[1],[1,2]], [[2,1],[1],[2,1]])
        data = (; subject=[1,2,3], w=Union{Missing,Float64}[10.,missing,30.],
            raw_x=[1.2,1.5,2.0], rows, values=[[0.1,0.2],[0.3],[0.4,0.5]],
            y=[[0.2,0.1],[0.0],[0.3,-0.1]])
        saved = deepcopy(data)
        brmi = PublicModeledTransform.build(data)
        artifact = BRM.emit_rk_artifact(brmi; case_id="modeled-transform")
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        @test length(names) == 18
        graph = ReactiveKernels.kernel_graph(backend.model.spec)
        @test any(recipe_inventory(graph)) do entry
            entry.kind === :plate && modeled_location_in_graph(plate_body(entry.recipe))
        end
        index(n) = only(findall(==(Symbol(n)), names))
        # The modeled column shares the population design with the intercept.
        beta = index.(["pop_log_a.beta_pop.1", "pop_log_a.beta_pop.2", "b_Intercept"])
        scales = index.(["b_p_subject.tau.1", "b_p_subject.tau.2"])
        z = [index("b_p_subject.z.$j.$k") for j in 1:3, k in 1:2]
        correlation = index("b_p_subject.L.1")
        iw, imw, isw, imx, isx, ie = index.([
            "w.y_mis.1", "mu_w", "sigma_w", "mu_x", "sigma_x", "scale"])
        function components(u)
            w = [10., exp(u[iw]), 30.]
            rho, tau = tanh(u[correlation]), exp.(u[scales])
            L = [1.0 0.0; rho sqrt(1-rho^2)]
            random = u[z] * transpose(tau .* L)
            transformed_x = log.(data.raw_x .* w)
            a = exp.(u[beta[1]] .+ u[beta[2]] .* transformed_x .+ random[:,1])
            b = exp.(u[beta[3]] .+ random[:,2])
            internal = [a[j] .+ b[j] .* data.values[j] for j in 1:3]
            reads = reduce(vcat, [internal[j][data.rows[j]] for j in 1:3])
            (; w, rho, tau, transformed_x, a, b,
                internal=reduce(vcat,internal), reads, mu_w=u[imw],
                sigma_w=exp(u[isw]), mu_x=u[imx], sigma_x=exp(u[isx]), scale=exp(u[ie]))
        end
        function oracle(u)
            c = components(u)
            sum(logpdf.(Normal(),u[beta])) + sum(logpdf.(Normal(),u[z])) +
                sum(logpdf.(Exponential(),c.tau) .+ u[scales]) +
                log(3/4) + 2log1p(-c.rho^2) +
                logpdf(Normal(3,1),c.mu_w) + logpdf(LogNormal(-1,0.3),c.sigma_w) + u[isw] +
                logpdf(Normal(1,1),c.mu_x) + logpdf(LogNormal(-1,0.3),c.sigma_x) + u[isx] +
                logpdf(Exponential(),c.scale) + u[ie] +
                logpdf(LogNormal(c.mu_w,c.sigma_w),c.w[2]) + u[iw] +
                sum(logpdf.(LogNormal(c.mu_w,c.sigma_w),[10.,30.])) +
                sum(logpdf.(LogNormal(c.mu_x,c.sigma_x),data.raw_x)) +
                sum(logpdf.(Normal.(c.reads,c.scale),reduce(vcat,data.y))) +
                sum(logpdf.(Normal.(c.internal,c.scale),reduce(vcat,data.values)))
        end
        stan = consumer_stan(brmi,"modeled-transform-$(first(rows[1]))"; mod=PublicModeledTransform)
        mapping = [names[correlation]=>"b_p_subject_L.1", names[iw]=>"w_y_mis.1",
            names[imw]=>"mu_w", names[isw]=>"sigma_w", names[imx]=>"mu_x",
            names[isx]=>"sigma_x", names[ie]=>"scale"]
        append!(mapping, [names[scales[j]]=>"b_p_subject_tau.$j" for j in 1:2])
        append!(mapping, [names[z[j,k]]=>"b_p_subject_z_flat.$(k+2*(j-1))" for j in 1:3 for k in 1:2])
        append!(mapping, [names[beta[j]]=>"pop_log_a_beta_pop.$j" for j in 1:2])
        push!(mapping,names[beta[3]]=>"pop_log_b_beta_pop.1")
        ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
        bound = ext._rk_translated_plan(backend.plan)
        pointwise = prepare_query(backend.model,bound,:pointwise)
        translated = BRM.rk_translate_artifact(artifact)
        rebuilt = Base.invokelatest(build_kernel,translated)
        replay = prepare_sampler(rebuilt,translated,zeros(18);
            backend=AutoEnzyme(; mode=Enzyme.Reverse))
        function named_query(model, translated_plan)
            spec = model.spec
            fixed_names = Tuple(n for n in spec.have_names if n !== :unconstrained)
            fixed = NamedTuple{fixed_names}(Tuple(translated_plan.columns[n] for n in fixed_names))
            Base.invokelatest(prepare,spec; have=spec.have_names,
                want=(:w,:transformed_x,:a,:b),bound=fixed)
        end
        named = named_query(backend.model,bound)
        replay_named = named_query(rebuilt,translated)
        @test !haskey(backend.plan.columns, :transformed_x)
        @test length(backend.plan.columns[:subject]) == 3
        @test length(backend.plan.columns[:raw_x]) == 3
        replay_max_delta = 0.0
        for u in (zeros(18),fill(0.13,18),collect(range(-0.2,0.3;length=18)))
            value, gradient = check_consumer_point(problem,u,oracle)
            check_consumer_stan(problem,stan,mapping,backend,u)
            c, parts = components(u), pointwise(u)
            @test parts.w_obs ≈ logpdf.(LogNormal(c.mu_w,c.sigma_w),[10.,30.])
            @test parts.raw_x ≈ logpdf.(LogNormal(c.mu_x,c.sigma_x),data.raw_x)
            # Per-subject responses give one array of densities per subject.
            @test length.(parts.y) == length.(data.y)
            @test reduce(vcat,parts.y) ≈ logpdf.(Normal.(c.reads,c.scale),reduce(vcat,data.y))
            @test length.(parts.values) == length.(data.values)
            @test reduce(vcat,parts.values) ≈ logpdf.(Normal.(c.internal,c.scale),reduce(vcat,data.values))
            before = copy(u)
            replay_gradient = similar(u)
            replay_value, _ = sampler_value_and_gradient!(replay,replay_gradient,u)
            @test isequal(value,replay_value)
            # Independent preparations can differ by rounding in reverse
            # accumulation; require every coordinate at near-machine precision.
            @test gradient ≈ replay_gradient atol=2e-13 rtol=2e-13
            replay_max_delta = max(replay_max_delta,maximum(abs.(gradient .- replay_gradient)))
            queried = Base.invokelatest(named,u)
            @test length.(queried) == (3,3,3,3)
            @test queried[1] ≈ c.w
            @test queried[2] ≈ c.transformed_x
            @test queried[3] ≈ c.a
            @test queried[4] ≈ c.b
            @test isequal(queried,Base.invokelatest(replay_named,u))
            @test isequal(u,before)
        end
        println("REPLAY_MAX_ABS_GRADIENT_DELTA=",replay_max_delta); flush(stdout)
        @test isequal(data,saved)
    end
end

@stestset "assignment chains and scalar sampled parents without completion" begin
    data = (; raw_x=[0.5,1.3,2.0], y=[0.2,-0.1,0.4])
    saved = deepcopy(data)
    brmi = @brm data begin
        raw_x ~ LogNormal(0,1)
        delta ~ Normal(0,1)
        x = log(raw_x)
        shifted = x + delta
        mu ~ 0 + shifted
        effect(mu, shifted) ~ Normal(0.2,0.8)
        y ~ Normal(mu,1)
    end
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    @test Set(names) == Set([:delta,Symbol("pop_mu.beta_pop.1")])
    function oracle(u)
        p = Dict(zip(names,u))
        mu = p[Symbol("pop_mu.beta_pop.1")] .* (log.(data.raw_x) .+ p[:delta])
        logpdf(Normal(),p[:delta]) + logpdf(Normal(0.2,0.8),p[Symbol("pop_mu.beta_pop.1")]) +
            sum(logpdf.(LogNormal(0,1),data.raw_x)) +
            sum(logpdf.(Normal.(mu,1),data.y))
    end
    for u in (zeros(2),[0.13,-0.2],[-0.2,0.3])
        check_consumer_point(problem,u,oracle)
    end
    @test !haskey(backend.plan.columns,:shifted)
    @test isequal(data,saved)
end
