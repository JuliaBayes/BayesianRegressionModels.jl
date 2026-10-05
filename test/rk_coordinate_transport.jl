# Public synthetic controls for complete semantic RK/Stan correspondence.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module PublicCoordinateTransport
using BayesianRegressionModels, Distributions

function correlated(data)
    @brm data begin
        eta ~ 1 + g + (1 | subject)
        mu ~ 1 + x + w + (1 + x + w | effect | subject)
        sd(mu, effect, x) ~ Exponential(0.8)
        sd(mu, effect, w) ~ Exponential(1.1)
        sigma ~ Exponential(0.7)
        y ~ Normal(mu, sigma)
        y2 ~ Normal(eta, sigma)
    end
end

function linked(data)
    @brm data begin
        log(rate) ~ 1 + x + g + (1 + x | subject)
        eta ~ 1 + (0 + x | site) + (1 + x | subject)
        sigma ~ Exponential(0.7)
        y ~ Poisson(rate)
        y2 ~ Normal(eta, sigma)
    end
end

function reference(data)
    @brm data begin
        mu ~ 1 + factor(g; ref=3)
        y ~ Normal(mu, 1)
    end
end

function cellmeans(data)
    @brm data begin
        mu ~ 0 + factor(g)
        y ~ Normal(mu, 1)
    end
end

function mixed(data)
    @brm data begin
        mu ~ 1 + x
        # Keep all comparison points away from the Laplace cusp.
        effect(mu, x) ~ Laplace(0.17, 0.8)
        y ~ Normal(mu, 1)
    end
end
end

function coordinate_transport_fixture(brmi, name)
    rk, problem = consumer_problem(brmi)
    sb = SBBRMI(brmi; mod=PublicCoordinateTransport, total_groups=())
    path = joinpath(tempdir(), "brm-coordinate-transport", name * ".stan")
    mkpath(dirname(path))
    stan = BRM.stan_instantiate(sb; path)
    stan_names = BridgeStan.param_unc_names(stan.model)
    saved_names = copy(stan_names)
    transport = brm_coordinate_transport(rk, sb, stan_names)
    @test isequal(stan_names, saved_names)
    @test length(transport) == length(coordinate_names(rk.model.layout)) == length(stan_names)
    @test Set(p.rk for p in transport.pairs) == Set(transport.rk_names)
    @test Set(p.stan for p in transport.pairs) == Set(transport.stan_names)
    @test length(unique(p.address for p in transport.pairs)) == length(transport)
    @test sort(transport.permutation) == collect(1:length(transport))
    (; rk, problem, sb, stan, transport)
end

function check_coordinate_transport(fixture)
    (; rk, problem, stan, transport) = fixture
    # Distinct nonzero values expose matrix flattening and factor packing bugs
    # that a zero point or a repeated scalar cannot detect.
    for u in (zeros(length(transport)),
            collect(range(-0.43, 0.51; length=length(transport))),
            [0.37sin(i) for i in 1:length(transport)])
        saved = copy(u)
        stan_u = brm_rk_to_stan(transport, u)
        @test isequal(brm_stan_to_rk(transport, stan_u), u)
        @test isequal(brm_rk_to_stan(transport, brm_stan_to_rk(transport, stan_u)), stan_u)
        @test isequal(u, saved)
        checked = brm_check_coordinate_transport(transport, rk, stan.model, u)
        @test checked.pairs == count(p -> p.relation !== :cholesky, transport.pairs)
        @test checked.factors == length(transport.correlations)
        @test checked.max_error <= 1e-12
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        stan_gradient = similar(stan_u)
        stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
            stan_gradient; propto=false, jacobian=true)
        @test value ≈ stan_value atol=2e-11 rtol=2e-11
        @test gradient ≈ brm_stan_to_rk(transport, stan_gradient) atol=2e-10 rtol=2e-10
        @test isequal(u, saved)
    end
end

@stestset "coordinate transport correlated factors and categorical pool levels" begin
    groups = categorical(["b","b","a","a","c","c","d","d"])
    levels!(groups, ["d","b","a","c"])
    data = (; subject=groups, g=categorical(["a","b","c","a","b","c","a","b"]),
        x=[0.1,0.4,-0.2,0.3,0.7,0.5,0.2,-0.1],
        w=[1.0,0.5,0.2,0.3,0.1,0.9,0.4,0.6],
        y=[0.1,0.4,-0.2,0.3,0.7,0.5,0.0,1.0],
        y2=[0.3,0.2,0.1,-0.4,0.6,0.8,0.1,0.0])
    saved = deepcopy(data)
    brmi = PublicCoordinateTransport.correlated(data)
    fixture = coordinate_transport_fixture(brmi, "correlated")
    @test only(fixture.transport.correlations).K == 3
    @test count(p -> p.relation === :exp, fixture.transport.pairs) == 1
    @test Set(p.address.level for p in fixture.transport.pairs
        if p.address.kind === :ranef_z) == Set(levels(groups))
    check_coordinate_transport(fixture)
    @test isequal(data, saved)

    # Refused inputs: correspondence requires one scientific BRMI and every
    # coordinate exactly once (snag brm-rk-stan-comp-6abf68f7's explicit contract).
    names = fixture.transport.stan_names
    @test_throws "same BRMI instance" brm_coordinate_transport(fixture.rk,
        SBBRMI(PublicCoordinateTransport.correlated(merge(data, (; x=data.x .+ 1)));
            mod=PublicCoordinateTransport,
            total_groups=()), names)
    @test_throws "not unique" brm_coordinate_transport(fixture.rk, fixture.sb,
        vcat(names, first(names)))
    @test_throws "no semantic RK counterpart" brm_coordinate_transport(fixture.rk,
        fixture.sb, vcat(names, "unclaimed"))
    @test_throws "absent" brm_coordinate_transport(fixture.rk, fixture.sb, names[2:end])
    @test_throws "values for" brm_rk_to_stan(fixture.transport, zeros(length(names)-1))
    @test_throws "values for" brm_stan_to_rk(fixture.transport, zeros(length(names)+1))
end

@stestset "coordinate transport linked shared-design factor and plain slope" begin
    data = (; subject=[1,1,2,2,3,3,4,4],
        site=["s1","s2","s1","s2","s3","s3","s1","s2"],
        g=["b","a","c","a","b","c","a","b"],
        x=[0.1,0.4,-0.2,0.3,0.7,0.5,0.2,-0.1],
        y=[1,4,0,3,7,5,0,2], y2=[0.3,0.2,0.1,-0.4,0.6,0.8,0.1,0.0])
    saved = deepcopy(data)
    fixture = coordinate_transport_fixture(PublicCoordinateTransport.linked(data), "linked")
    @test sort([c.K for c in fixture.transport.correlations]) == [2, 2]
    @test Set(p.address.margins for p in fixture.transport.pairs
        if p.address.kind === :ranef_correlation) ==
        Set([((:rate, :Intercept), (:rate, :x)), ((:eta, :Intercept), (:eta, :x))])
    @test count(p -> p.address.kind === :ranef_z && p.address.group === :site,
        fixture.transport.pairs) == 3
    check_coordinate_transport(fixture)
    @test isequal(data, saved)
end

@stestset "coordinate transport explicit reference and full-rank factors" begin
    data = (; g=[2,1,3,1,2,3], y=[0.1,-0.2,0.3,0.5,-0.4,0.2])
    for (build, name) in ((PublicCoordinateTransport.reference, "reference"),
            (PublicCoordinateTransport.cellmeans, "cellmeans"))
        fixture = coordinate_transport_fixture(build(data), name)
        check_coordinate_transport(fixture)
    end
end

@stestset "coordinate transport mixed-family population components" begin
    data = (; x=[-0.4,0.1,0.3,0.6], y=[0.2,-0.1,0.4,0.3])
    fixture = coordinate_transport_fixture(PublicCoordinateTransport.mixed(data), "mixed")
    check_coordinate_transport(fixture)
end

@stestset "coordinate transport capability gap for smooth internals" begin
    brmi = @brm (; x=[-0.7,0.0,0.6], y=[0.2,-0.1,0.4]) begin
        mu ~ 0 + hsgp(x; k=3, domain=(-2.0,2.0))
        y ~ Normal(mu,1)
    end
    rk = RKBRMI(brmi)
    sb = SBBRMI(brmi; total_groups=())
    stan = BRM.stan_instantiate(sb; path=joinpath(tempdir(), "brm-coordinate-transport", "hsgp.stan"))
    # A valid model outside the current semantic inventory is a capability gap,
    # not a rejected language shape (dev §2). It must never yield a partial map.
    result = try
        brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model))
    catch err
        err isa BRMCoordinateTransportError || rethrow()
        @test err.reason === :unsupported_coverage
        @test occursin("do not pair completely", sprint(showerror, err))
        @test occursin("hsgp", sprint(showerror, err))
        nothing
    end
    @test_broken result isa BRMCoordinateTransport
end
