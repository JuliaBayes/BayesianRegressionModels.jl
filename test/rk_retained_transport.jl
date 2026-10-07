# Public controls for attaching transport to the artifact route's exact build.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

@stestset "parameterization metadata before Stan compilation" begin
    brmi = @brm (; subject=[1,2,1,2], y=[0.2,-0.1,0.4,0.3]) begin
        mu ~ 1 + (1 | subject)
        y ~ Normal(mu,1)
    end
    ordinary = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    centered = SBBRMI(brmi; mod=@__MODULE__, total_groups=(),
        centered_groups=[:subject])
    total = SBBRMI(brmi; mod=@__MODULE__)
    @test isempty(total_effect_blocks(ordinary))
    @test isempty(total_effect_blocks(centered))
    @test !isempty(total_effect_blocks(total))
    ordinary_blocks = ranef_blocks(ordinary)
    centered_blocks = ranef_blocks(centered)
    @test !isempty(ordinary_blocks)
    @test all(b -> b.noncentered && !b.generated, ordinary_blocks)
    @test length(centered_blocks) == 1
    @test only(centered_blocks).group === :subject
    @test !only(centered_blocks).noncentered
    @test !only(centered_blocks).generated
end

@stestset "retained build coordinate transport and held-out likelihood" begin
    data = (; subject=[2,1,3,2,1,3], x=[-0.7,0.2,0.6,-0.1,0.9,0.4],
        y=[0.2,-0.1,0.4,0.5,-0.3,0.1], y2=[-0.2,0.3,0.1,-0.4,0.6,0.2])
    brmi = @brm data begin
        mu ~ 1 + x + (1 + x | effect | subject)
        eta ~ 1 + (1 | subject)
        sigma ~ Exponential(0.7)
        y ~ Normal(mu, sigma)
        y2 ~ Normal(eta, sigma)
    end
    saved_data = deepcopy(data)
    for held_out in ((), :y2)
        println("Retained transport held_out=", repr(held_out)); flush(stdout)
        artifact = BRM.emit_rk_artifact(brmi; case_id="retained-transport", held_out)
        bound = rk_translate_artifact(artifact)
        built = build_kernel(bound)
        native = prepare_query(built, bound, :sampler)
        rk = RKBRMI(brmi, artifact.plan, built)
        @test parent(rk) === brmi
        @test rk.plan === artifact.plan
        @test rk.model === built
        sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=(), held_out)
        stan = BRM.stan_instantiate(sb; path=joinpath(tempdir(),
            "brm-coordinate-transport", "retained-$(held_out === () ? "all" : "y2").stan"))
        println("Retained RK and compiled Stan ready"); flush(stdout)
        transport = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model))
        @test sort(transport.permutation) == collect(1:built.layout.total)
        @test length(transport.correlations) == 1
        u0 = zeros(built.layout.total)
        query = prepare_sampler(built, bound, u0;
            backend=AutoEnzyme(; mode=Enzyme.Reverse))
        for u in (u0, collect(range(-0.43,0.51; length=length(u0))),
                [0.37sin(i) for i in eachindex(u0)])
            saved = copy(u)
            before = Base.invokelatest(native, u)
            checked = brm_check_coordinate_transport(transport, rk, stan.model, u)
            @test keys(checked) == (:pairs, :factors, :simplexes, :max_error)
            @test checked.pairs == count(p -> p.relation !== :cholesky, transport.pairs)
            @test checked.factors == 1
            @test checked.max_error <= 1e-12
            gradient = similar(u)
            value, _ = sampler_value_and_gradient!(query, gradient, u)
            stan_u = brm_rk_to_stan(transport, u)
            sv, sg = BridgeStan.log_density_gradient(stan.model, stan_u;
                propto=false, jacobian=true)
            @test value == before == Base.invokelatest(native, u)
            @test value ≈ sv atol=2e-11 rtol=2e-11
            @test gradient ≈ brm_stan_to_rk(transport, sg) atol=2e-10 rtol=2e-10
            @test isequal(u, saved)
            @test rk.model === built
        end
        # Different parameterizations cannot be compared using a permutation
        # (reporter's explicit permutation-only comparison requirement).
        centered = SBBRMI(brmi; mod=@__MODULE__, total_groups=(),
            centered_groups=[:subject], held_out)
        centered_stan = BRM.stan_instantiate(centered; path=joinpath(tempdir(),
            "brm-coordinate-transport", "retained-centered-$(held_out === () ? "all" : "y2").stan"))
        err = try
            brm_coordinate_transport(rk, centered,
                BridgeStan.param_unc_names(centered_stan.model))
        catch e
            e
        end
        @test err isa BRMCoordinateTransportError
        @test err.reason === :parameterization_mismatch
        @test occursin("is centered", sprint(showerror, err))

        # Input errors must not masquerade as unsupported model coverage.
        invalid = try
            brm_coordinate_transport(rk, sb, vcat(transport.stan_names,
                first(transport.stan_names)))
        catch e
            e
        end
        @test invalid isa ErrorException
        @test !(invalid isa BRMCoordinateTransportError)
        @test occursin("not unique", sprint(showerror, invalid))

        # A wrong permutation can retain valid lengths and names. The physical
        # check must detect it instead of returning a successful receipt.
        wrong = copy(transport.permutation)
        wrong[1], wrong[end] = wrong[end], wrong[1]
        corrupted = BRMCoordinateTransport(transport.rk_names, transport.stan_names,
            transport.pairs, wrong, transport.correlations)
        @test_throws "disagrees" brm_check_coordinate_transport(corrupted, rk,
            stan.model, collect(range(-0.43,0.51; length=built.layout.total)))
    end
    @test isequal(data, saved_data)
end

@stestset "exact-total coordinates require more than a permutation" begin
    brmi = @brm (; subject=[1,2,1,2], y=[0.2,-0.1,0.4,0.3]) begin
        mu ~ 1 + (1 | subject)
        y ~ Normal(mu,1)
    end
    artifact = BRM.emit_rk_artifact(brmi; case_id="retained-exact-total")
    built = build_kernel(rk_translate_artifact(artifact))
    rk = RKBRMI(brmi, artifact.plan, built)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    @test !isempty(total_effect_blocks(sb))
    stan = BRM.stan_instantiate(sb; path=joinpath(tempdir(),
        "brm-coordinate-transport", "retained-total.stan"))
    # Exact-total Stan and non-centered RK points are not a permutation
    # (reporter's explicit permutation-only comparison requirement).
    err = try
        brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model))
    catch e
        e
    end
    @test err isa BRMCoordinateTransportError
    @test err.reason === :parameterization_mismatch
    @test occursin("exact total-effect block", sprint(showerror, err))
end

@stestset "retained build pairs implicit OrderedLogistic cutpoints" begin
    # An authored location, as in a downstream retained artifact: the value
    # route emits `y_cutpoints ~ Ordered(Normal(0.0, 1.0), 3)` and SBBRMI
    # `ordered[3] y_cutpoints` (gappy raw codes, K = maximum(y) = 4).
    data = (; subject=[1,2,3,1,2,3,1,2], x=[-1.2,-0.4,0.1,0.5,0.9,1.4,-0.8,0.3],
        y=[1,2,2,4,4,4,1,2])
    brmi = @brm data begin
        slope ~ 0 + x + (1 | subject)
        bias ~ Normal(0, 1)
        loc = slope + bias
        y ~ OrderedLogistic(loc)
    end
    saved_data = deepcopy(data)
    artifact = BRM.emit_rk_artifact(brmi; case_id="retained-cutpoints")
    bound = rk_translate_artifact(artifact)
    built = build_kernel(bound)
    rk = RKBRMI(brmi, artifact.plan, built)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    stan = BRM.stan_instantiate(sb; path=joinpath(tempdir(),
        "brm-coordinate-transport", "retained-cutpoints.stan"))
    transport = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model))
    @test sort(transport.permutation) == collect(1:built.layout.total)
    @test [p.stan for p in transport.pairs if p.address.kind === :vector] ==
        ["y_cutpoints.$j" for j in 1:3]
    u0 = zeros(built.layout.total)
    query = prepare_sampler(built, bound, u0; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for u in (u0, collect(range(-0.43,0.51; length=length(u0))),
            [0.37sin(i) for i in eachindex(u0)])
        saved = copy(u)
        checked = brm_check_coordinate_transport(transport, rk, stan.model, u)
        @test checked.pairs == length(transport)
        @test checked.max_error <= 1e-12
        gradient = similar(u)
        value, _ = sampler_value_and_gradient!(query, gradient, u)
        sv, sg = BridgeStan.log_density_gradient(stan.model, brm_rk_to_stan(transport, u);
            propto=false, jacobian=true)
        @test value ≈ sv atol=2e-11 rtol=2e-11
        @test gradient ≈ brm_stan_to_rk(transport, sg) atol=2e-10 rtol=2e-10
        @test isequal(u, saved)
    end
    @test isequal(data, saved_data)
end
