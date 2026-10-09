include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Statistics: mean, std

# Public synthetic subject/measurement geometry, including a likelihood inside
# a cell. Completion must keep the subject axis when that likelihood is flat.
module PublicCompletedCovariateAxes
using BayesianRegressionModels, Distributions
function build(data)
    @brm data begin
        mu_age ~ Normal(3.0, 1.0)
        sigma_age ~ LogNormal(-1.0, 0.3)
        mi(age_yr) ~ LogNormal(mu_age, sigma_age)
        log(a) ~ 1 + standardize(age_yr) + (1 | p | subject)
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

@stestset "completed covariate graphs retain subject and kernel likelihood axes" begin
    # Permute the selected cell rows: the internal and external likelihoods
    # then have different locations despite their equal flattened lengths.
    data = (; subject=[1,2,3], age_yr=Union{Missing,Float64}[10.,missing,30.],
        rows=[[2,1],[1],[2,1]], values=[[0.1,0.2],[0.3],[0.4,0.5]],
        y=[[0.2,0.1],[0.0],[0.3,-0.1]])
    saved = deepcopy(data)
    brmi = PublicCompletedCovariateAxes.build(data)
    artifact = BRM.emit_rk_artifact(brmi; case_id="completed-covariate-axes")
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    @test length(names) == 16
    index(n) = only(findall(==(Symbol(n)), names))
    # The population component owns the intercept and completed-column slope.
    beta = index.(["pop_log_a.beta_pop.1", "pop_log_a.beta_pop.2", "b_Intercept"])
    scales = index.(["b_p_subject.tau.1", "b_p_subject.tau.2"])
    z = [index("b_p_subject.z.$j.$k") for j in 1:3, k in 1:2]
    correlation = index("b_p_subject.L.1")
    ia, im, is, ie = index.(["age_yr.y_mis.1", "mu_age", "sigma_age", "scale"])
    observed = collect(skipmissing(data.age_yr))
    anchor, spread = mean(observed), std(observed)
    function components(u)
        age = [10., exp(u[ia]), 30.]
        rho, tau = tanh(u[correlation]), exp.(u[scales])
        L = [1.0 0.0; rho sqrt(1-rho^2)]
        random = u[z] * transpose(tau .* L)
        a = exp.(u[beta[1]] .+ u[beta[2]] .* ((age .- anchor) ./ spread) .+ random[:,1])
        b = exp.(u[beta[3]] .+ random[:,2])
        internal = [a[j] .+ b[j] .* data.values[j] for j in 1:3]
        reads = reduce(vcat, [internal[j][data.rows[j]] for j in 1:3])
        (; age, rho, tau, internal=reduce(vcat,internal), reads,
            mu=u[im], sigma=exp(u[is]), scale=exp(u[ie]))
    end
    function oracle(u)
        c = components(u)
        sum(logpdf.(Normal(),u[beta])) + sum(logpdf.(Normal(),u[z])) +
            sum(logpdf.(Exponential(),c.tau) .+ u[scales]) +
            log(3/4) + 2log1p(-c.rho^2) +
            logpdf(Normal(3,1),c.mu) + logpdf(LogNormal(-1,0.3),c.sigma) + u[is] +
            logpdf(Exponential(),c.scale) + u[ie] +
            logpdf(LogNormal(c.mu,c.sigma),c.age[2]) + u[ia] +
            sum(logpdf.(LogNormal(c.mu,c.sigma),observed)) +
            sum(logpdf.(Normal.(c.reads,c.scale),reduce(vcat,data.y))) +
            sum(logpdf.(Normal.(c.internal,c.scale),reduce(vcat,data.values)))
    end
    stan = consumer_stan(brmi,"completed-covariate-axes"; mod=PublicCompletedCovariateAxes)
    mapping = [names[correlation]=>"b_p_subject_L.1", names[ia]=>"age_yr_y_mis.1",
        names[im]=>"mu_age", names[is]=>"sigma_age", names[ie]=>"scale"]
    append!(mapping, [names[scales[j]]=>"b_p_subject_tau.$j" for j in 1:2])
    append!(mapping, [names[z[j,k]]=>"b_p_subject_z_flat.$(k+2*(j-1))" for j in 1:3 for k in 1:2])
    append!(mapping, [names[beta[j]]=>"pop_log_a_beta_pop.$j" for j in 1:2])
    push!(mapping,names[beta[3]]=>"pop_log_b_beta_pop.1")
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    bound = ext._rk_translated_plan(backend.plan)
    pointwise = prepare_query(backend.model,bound,:pointwise)
    # Complete source replay is also checked by consumer_problem above.
    translated = BRM.rk_translate_artifact(artifact)
    replay = Base.invokelatest(prepare_query,build_kernel(translated),translated,:sampler)
    for u in (zeros(16),fill(0.13,16),collect(range(-0.2,0.3;length=16)))
        value, gradient = check_consumer_point(problem,u,oracle)
        check_consumer_stan(problem,stan,mapping,backend,u)
        c, parts = components(u), pointwise(u)
        @test parts.age_yr_obs ≈ logpdf.(LogNormal(c.mu,c.sigma),observed)
        # Per-subject responses give one array of densities per subject.
        @test length.(parts.y) == length.(data.y)
        @test reduce(vcat,parts.y) ≈ logpdf.(Normal.(c.reads,c.scale),reduce(vcat,data.y))
        @test length.(parts.values) == length.(data.values)
        @test reduce(vcat,parts.values) ≈ logpdf.(Normal.(c.internal,c.scale),reduce(vcat,data.values))
        @test value == Base.invokelatest(replay,u)
    end
    @test isequal(data,saved)
end

@stestset "completion graph remains addressable and avoids authored name collisions" begin
    data = (; subject=[1,2,3,4], x=Union{Missing,Float64}[1.,missing,3.,missing],
        w=Union{Missing,Float64}[missing,2.,missing,4.], y=zeros(4))
    brmi = @brm data begin
        brm_completed_covariate ~ Normal(0,1)
        mi(x) ~ Normal(0,1)
        mi(w) ~ Normal(0,1)
        mu ~ 1 + x + w + (1 | p | subject)
        y ~ Normal(mu,exp(brm_completed_covariate))
    end
    artifact = BRM.emit_rk_artifact(brmi; case_id="completion-name-collision")
    definition = only(filter(d -> Meta.isexpr(d, :macrocall) &&
        startswith(string(BRM._rk_source_definition(d).name), "brm_completed_covariate"),
        artifact.defs))
    name = BRM._rk_source_definition(definition).name
    @test name != :brm_completed_covariate
    mod = Module(gensym(:CompletionGraph))
    Core.eval(mod,:(using ReactiveKernels))
    graph = Core.eval(mod,definition)
    @test graph isa ReactiveKernels.KernelSpec
    # These are named graph ports, rather than algebra hidden in a callback.
    query = Base.invokelatest(prepare,graph;
        have=(:observed,:missing,:lookup,:mask),want=(:drawn,:completed))
    observed, missing_values = [1.,0.,3.,0.], [5.,7.]
    lookup, mask = [1,1,1,2], [0.,1.,0.,1.]
    before = deepcopy((observed,missing_values,lookup,mask))
    @test Base.invokelatest(query,observed,missing_values,lookup,mask) ==
        ([5.,5.,5.,7.],[1.,5.,3.,7.])
    @test isequal(before,(observed,missing_values,lookup,mask))
    backend = check_rk_source_roundtrip(RKBRMI(brmi))
    @test count(n->occursin(".y_mis.",string(n)),coordinate_names(backend.model.layout)) == 4
end
