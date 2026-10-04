include(joinpath(@__DIR__, "rk_consumer_support.jl"))

@stestset "unconditioned BRM retains generated roles and zero fitted coordinates" begin
    regression = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    direct = @brm begin
        a ~ Normal(0, 1)
        y ~ Normal(a, 1)
    end
    ragged_prior = @brm begin
        sigma_add ~ Exponential(1)
        sigma_prop ~ Exponential(1)
        log_CL ~ 1 + (1 | p | subject)
        loc ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
            exp(lCL) .+ 0.0 .* idxs
        end
        ragged(pk_conc, obs_subject) ~ censored(
            Normal(loc, addprop(loc, sigma_add, sigma_prop)); lower=pk_lloq)
    end
    ragged_data = (;
        subject=["s2", "s1"],
        obs_subject=["s1", "s2", "s1", "s2", "s2", "s1", "s2"],
        obs_idx=collect(1.0:7.0),
        pk_lloq=[0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16])
    for (label, factory, data) in (
        ("prior-regression", regression, (; x=[-0.2, 0.4, 0.8])),
        ("prior-direct-no-data", direct, (;)),
        ("prior-ragged-bounds", ragged_prior, ragged_data))
        before = deepcopy(data)
        brmi = factory(data)
        backend = check_rk_source_roundtrip(RKBRMI(brmi))
        @test backend.plan isa BRM._RKUnconditionedPlan
        @test isempty(coordinate_names(backend.model.layout))
        @test isempty(BRM._rk_emit_ast(backend.plan).main.args)
        @test isempty(BRM._rk_observed_names(backend.plan))
        @test any(d -> d.role === :observation && d.data_source === nothing,
            backend.plan.generative.declarations)
        @test occursin("native generated draws = unavailable", BRM.show_rk_plan(backend.plan))
        @test BRM.stan_code(backend.plan.generative) ==
            BRM.stan_code(SBBRMI(brmi; total_groups=()))
        if label == "prior-ragged-bounds"
            @test backend.plan.generative.data[:pk_conc_lower_pk_lloq_ragged] ==
                [[0.11, 0.13, 0.14, 0.16], [0.10, 0.12, 0.15]]
            @test !haskey(backend.plan.columns, :pk_conc)
        elseif label == "prior-regression"
            @test backend.plan.generative.data[:x_n] == 3
        end
        problem = rk_logdensity_problem(backend;
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, Float64[])
        @test value == 0.0
        @test isempty(gradient)
        stan = consumer_stan(brmi, label)
        @test BridgeStan.param_unc_num(stan.model) == 0
        sv, sg = BridgeStan.log_density_gradient(stan.model, Float64[];
            propto=false, jacobian=true)
        @test sv == value
        @test isempty(sg)
        artifact = BRM.emit_rk_artifact(brmi; case_id=label)
        rebuilt = build_kernel(BRM.rk_translate_artifact(artifact))
        @test isempty(coordinate_names(rebuilt.layout))
        @test isequal(data, before)
    end
end

@stestset "bound baseline retains inference while absent endpoint forward-simulates" begin
    data = (; baseline=[0.1, -0.3, 0.5], x=[-0.2, 0.4, 0.8])
    original = deepcopy(data)
    for value_route in (false, true)
        brmi = if value_route
            @brm data begin
                a ~ Normal(0, 1)
                unused ~ Exponential(2)
                loc = a + x
                baseline ~ Normal(a, 1)
                y ~ Normal(loc, 1)
            end
        else
            @brm data begin
                a ~ Normal(0, 1)
                unused ~ Exponential(2)
                mu ~ 1
                baseline ~ Normal(mu, 1)
                y ~ Normal(mu, 1)
            end
        end
        backend, problem = consumer_problem(brmi)
        @test !(backend.plan isa BRM._RKUnconditionedPlan)
        @test length(coordinate_names(backend.model.layout)) == 1
        @test BRM._rk_observed_names(backend.plan) == (:baseline,)
        stan = consumer_stan(brmi, "prior-baseline-$value_route")
        @test BridgeStan.param_unc_num(stan.model) == 1
        oracle(u) = logpdf(Normal(), u[1]) +
            sum(logpdf.(Normal(u[1], 1), data.baseline))
        for u in ([0.0], [0.23], [-0.17])
            value, gradient = check_consumer_point(problem, u, oracle)
            sv, sg = BridgeStan.log_density_gradient(stan.model, u;
                propto=false, jacobian=true)
            @test value ≈ sv atol=2e-12 rtol=2e-12
            @test gradient ≈ sg atol=2e-12 rtol=2e-12
        end
        @test isequal(data, original)
    end
end
