include(joinpath(@__DIR__, "rk_consumer_support.jl"))

@stestset "held-out likelihoods retain active target and shared ancestors" begin
    data = (; y=[0.2, -0.4, 0.7], z=[0.1, 0.3, -0.2], x=[-0.2, 0.4, 0.8])
    before = deepcopy(data)
    for separate in (false, true), hold in (:y, :z)
        brmi = if separate
            @brm data begin
                a ~ Normal(0, 1)
                b ~ Normal(0, 2)
                mu_y = a + x
                mu_z = b - x
                y ~ Normal(mu_y, 1)
                z ~ Normal(mu_z, 1)
            end
        else
            @brm data begin
                a ~ Normal(0, 1)
                mu = a + x
                y ~ Normal(mu, 1)
                z ~ Normal(mu, 1)
            end
        end
        backend = check_rk_source_roundtrip(RKBRMI(brmi; held_out=hold))
        names = coordinate_names(backend.model.layout)
        expected_names = separate ? [:a, :b] : [:a]
        @test names == expected_names
        active = hold === :y ? :z : :y
        ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
        emitted = BRM._rk_emit_ast(backend.plan)
        source = sprint(Base.show_unquoted, emitted.main)
        @test !occursin(string(hold) * " .~", source)
        @test occursin(string(active) * " .~", source)
        @test BRM._rk_observed_names(backend.plan) == (active,)
        @test occursin("held_out    = [$hold]", BRM.show_rk_plan(backend.plan))
        problem = rk_logdensity_problem(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=(), held_out=hold)
        stan = BRM.stan_instantiate(sb; path=joinpath(tempdir(), "brm-rk-consumer", "held-out-$separate-$hold.stan"))
        @test BridgeStan.param_unc_names(stan.model) == string.(expected_names)
        function oracle(u)
            mu = separate && active === :z ? u[2] .- data.x : u[1] .+ data.x
            logpdf(Normal(0, 1), u[1]) +
                (separate ? logpdf(Normal(0, 2), u[2]) : 0.0) +
                sum(logpdf.(Normal.(mu, 1), getproperty(data, active)))
        end
        points = separate ? ([0.0, 0.0], [0.23, -0.31], [-0.17, 0.29]) :
            ([0.0], [0.23], [-0.17])
        for u in points
            value, gradient = check_consumer_point(problem, u, oracle)
            sv, sg = BridgeStan.log_density_gradient(stan.model, u; propto=false, jacobian=true)
            @test value ≈ sv atol=2e-12 rtol=2e-12
            @test gradient ≈ sg atol=2e-12 rtol=2e-12
        end
        artifact = BRM.emit_rk_artifact(brmi; case_id="held-out-$separate-$hold", held_out=hold)
        translated = BRM.rk_translate_artifact(artifact)
        rebuilt = build_kernel(translated)
        @test coordinate_names(rebuilt.layout) == names
        for u in points[1:2], preset in (:sampler, :prior, :likelihood)
            @test isequal(Base.invokelatest(prepare_query(rebuilt, translated, preset), u),
                Base.invokelatest(prepare_query(backend.model, ext._rk_translated_plan(backend.plan), preset), u))
        end
        for selection in (:all, (:y, :z), :typo, "y", (1,))
            @test_throws ErrorException RKBRMI(brmi; held_out=selection)
            @test_throws ErrorException SBBRMI(brmi; mod=@__MODULE__, total_groups=(), held_out=selection)
        end
        @test isequal(data, before)
    end
end
