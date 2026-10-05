include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Statistics: mean

# Generic public operation rows and a grouped consumer; no application model.
module PublicMonotonicConsumer
using BayesianRegressionModels, Distributions
function build(data)
    @brm data begin
        theta ~ 1 + (1 | p | subject)
        effect(theta, Intercept) ~ Normal(0, 0.7)
        sd(:, p) ~ Exponential(0.9)
        score ~ 1 + mo(rank) + hsgp(x; k=3)
        effect(score, Intercept) ~ Normal(0, 0.4)
        simplex(score, mo(rank)) ~ Dirichlet(1, 2)
        pred ~ kernel(t, theta, ragged(score, row_group)) do ts, a, scores
            a .+ ts .* sum(scores)
        end
        ragged(y, event_subject) ~ Normal(pred, 0.8)
    end
end
end

@stestset "monotonic and HSGP operation values retain law, gradients and source" begin
    data = (; subject=["b", "empty", "a"], t=[[0.2, 0.5], Float64[], [0.7]],
        row_group=["a", "b", "a", "b", "a"], rank=[1, 3, 2, 1, 3],
        x=[-0.8, -0.3, 0.1, 0.4, 0.9], y=[0.1, -0.2, 0.4],
        event_subject=["b", "b", "a"])
    saved = deepcopy(data)
    brmi = PublicMonotonicConsumer.build(data)
    # Capture before RKBRMI: this source-only API also works when RK admission
    # fails. The original fixture failed on an inline monotonic array gather.
    artifact = BRM.emit_rk_artifact(brmi; case_id="monotonic-hsgp-consumer")
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    index(n) = only(findall(==(Symbol(n)), names))
    # Components own their parameters: the monotonic effect its simplex and
    # coefficient, the HSGP its hyperparameters and basis weights.
    a, b = index("pop_score.beta_pop.1"), index("mo_rank.beta")
    intercept, sd = index("pop_theta.beta_pop.1"), index("b_p_subject.tau.1")
    levels = CategoricalArrays.levels(data.subject)
    z = [index("b_p_subject.z.$j.1") for j in eachindex(levels)]
    rows = [only(findall(==(subject), levels)) for subject in data.subject]
    rho, sigma = index("hsgp_x.rho_iso"), index("hsgp_x.sigma")
    weights = [index("hsgp_x.beta_raw.$j") for j in 1:3]
    simplex = index("mo_rank.simplex_incr.1")
    @test length(names) == 13
    # Independent sine basis, spectral weights and prior transform, rather
    # than using the emitter's fitted values or its density as an oracle.
    centered = data.x .- mean(data.x)
    L = 1.5maximum(abs, centered)
    frequencies = [(j*pi/(2L))^2 for j in 1:3]
    phi = [sin(sqrt(w)*(x+L))/sqrt(L) for x in centered, w in frequencies]
    floor = (4L/pi)*sqrt(log(100)/(3^2-1))
    function components(u)
        q = 1/(1+exp(-u[simplex]))
        increments = [q, 1-q]
        contrast = [sum(increments[1:(rank-1)]) for rank in data.rank]
        r, s, tau = floor+exp(u[rho]), exp(u[sigma]), exp(u[sd])
        smooth = phi * (s*sqrt(r*sqrt(2pi)) .* exp.(-r^2 .* frequencies ./ 4) .* u[weights])
        score = u[a] .+ u[b] .* contrast .+ smooth
        theta = u[intercept] .+ tau .* u[z[rows]]
        locations = reduce(vcat, [theta[j] .+ data.t[j] .*
            sum(score[findall(==(subject), data.row_group)])
            for (j, subject) in enumerate(data.subject)])
        jac = u[rho]+u[sigma]+u[sd]+log(q)+log1p(-q)
        # The default HSGP length-scale retains the base LogNormal kernel
        # above its fitted floor, exactly as stated by both emitted backends.
        prior = logpdf(Normal(0,0.4),u[a])+logpdf(Normal(),u[b])+
            logpdf(Normal(0,0.7),u[intercept])+logpdf(Exponential(0.9),tau)+
            sum(logpdf.(Normal(),u[z]))+logpdf(Dirichlet([1.,2.]),increments)+
            logpdf(LogNormal(),r)+logpdf(LogNormal(),s)+sum(logpdf.(Normal(),u[weights]))
        (; increments, r, s, tau, jac, prior, locations)
    end
    oracle(u) = begin
        c = components(u)
        c.prior+c.jac+sum(logpdf.(Normal.(c.locations,0.8),data.y))
    end
    stan = consumer_stan(brmi, "monotonic-hsgp-consumer"; mod=PublicMonotonicConsumer)
    stan_names = BridgeStan.param_names(stan.model)
    println("STAN_CONSTRAINED_NAMES=", stan_names); flush(stdout)
    physical(u) = begin
        c = components(u)
        values = Dict("pop_theta_beta_pop.1"=>u[intercept],
            "pop_score_beta_pop.1"=>u[a], "pop_score_beta_pop.2"=>u[b],
            "b_p_subject_L.1.1"=>1.0, "b_p_subject_tau.1"=>c.tau,
            "hsgp_x_rho_iso"=>c.r, "hsgp_x_sigma"=>c.s,
            "mo_rank_simplex_incr.1"=>c.increments[1],
            "mo_rank_simplex_incr.2"=>c.increments[2])
        for j in eachindex(z)
            values["b_p_subject_z_flat.$j"] = u[z[j]]
        end
        for j in eachindex(weights)
            values["hsgp_x_beta_raw.$j"] = u[weights[j]]
        end
        @test Set(keys(values)) == Set(stan_names)
        [values[n] for n in stan_names]
    end
    # Compare the same physical parameters: native stick-breaking and Stan's
    # simplex coordinates differ, so an unconstrained permutation is invalid.
    function stan_oracle(u)
        su = BridgeStan.param_unconstrain(stan.model, physical(u))
        sg = similar(su)
        value, _ = BridgeStan.log_density_gradient!(stan.model, su, sg;
            propto=false, jacobian=false)
        value+components(u).jac
    end
    translated = BRM.rk_translate_artifact(artifact)
    rebuilt = build_kernel(translated)
    @test coordinate_names(rebuilt.layout) == names
    replay = Base.invokelatest(prepare_query, rebuilt, translated, :sampler)
    empty = z[only(findall(==("empty"), levels))]
    for u in (zeros(13), fill(0.13,13), collect(range(-0.2,0.3; length=13)))
        value, gradient = check_consumer_point(problem,u,oracle)
        @test value ≈ stan_oracle(u) atol=2e-10 rtol=2e-10
        step = 1e-5
        stan_gradient = map(eachindex(u)) do j
            plus, minus = copy(u), copy(u)
            plus[j] += step; minus[j] -= step
            (stan_oracle(plus)-stan_oracle(minus))/(2step)
        end
        @test gradient ≈ stan_gradient atol=2e-8 rtol=2e-8
        @test gradient[empty] ≈ -u[empty] atol=2e-11
        @test value == Base.invokelatest(replay,u)
    end
    @test isequal(data,saved)
end

@stestset "monotonic components own their simplex without name collisions" begin
    data = (; rank=[1,3,2,1,3], x=[-0.8,-0.3,0.1,0.4,0.9], y=zeros(5))
    brmi = @brm data begin
        brm_monotonic_effect ~ Exponential(1)
        mu ~ 1 + mo(rank) + mo1(rank) + hsgp(x; k=3)
        y ~ Normal(mu,brm_monotonic_effect)
    end
    artifact = BRM.emit_rk_artifact(brmi; case_id="monotonic-name-collision")
    names = [first(first(d.args).args) for d in artifact.defs if Meta.isexpr(d, :(=))]
    # The authored parameter keeps its name; the component definition is renamed.
    @test :brm_monotonic_effect ∉ names
    @test any(n -> startswith(string(n), "brm_monotonic_effect"), names)
    @test :brm_monotonic_value in names
    @test !occursin("Dirichlet", sprint(Base.show_unquoted, artifact.ast))
    backend = check_rk_source_roundtrip(RKBRMI(brmi))
    # The authored name must be a fitted draw to reserve it for the collision.
    @test :brm_monotonic_effect in coordinate_names(backend.model.layout)
    @test backend.model.layout.total == 10
end
