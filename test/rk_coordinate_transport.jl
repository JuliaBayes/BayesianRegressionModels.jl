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

function smooth(data)
    @brm data begin
        mu ~ 0 + hsgp(x; k=3, domain=(-2.0, 2.0))
        y ~ Normal(mu, 1)
    end
end

function ordinal(data)
    @brm data begin
        mu ~ 1 + mo(r) + mo1(q) + hsgp(x; k=3) + hsgp(x, z; k=(2, 2), iso=false)
        simplex(mu, mo(r)) ~ Dirichlet(1, 2, 1.5)
        log(rate) ~ 1 + mo(r) + hsgp(x; k=2)
        sigma ~ Exponential(0.7)
        y ~ Normal(mu, sigma)
        n ~ Poisson(rate)
    end
end

# Response threshold vectors: implicit cutpoints for an authored and a
# formula location, cumulative and stopping-ratio thresholds.
function authored_cutpoints(data)
    @brm data begin
        slope ~ 0 + x
        bias ~ Normal(0, 1)
        loc = slope + bias
        y ~ OrderedLogistic(loc)
    end
end

function formula_cutpoints(data)
    @brm data begin
        eta ~ 0 + x + (1 | g)
        y ~ OrderedLogistic(eta)
    end
end

function cumulative_thresholds(data)
    @brm data begin
        slope ~ 0 + x
        bias ~ Normal(0, 1)
        loc = slope + bias
        # Logit: compiled Stan's `normal_lcdf` gradient is a known
        # StanBlocks boundary (issue 59) that a probit link would compare.
        y ~ Ordinal(Cumulative(), LogitLink(), loc; discrimination=2.0)
    end
end

function stopping_thresholds(data)
    @brm data begin
        eta ~ 0 + x
        y ~ Ordinal(StoppingRatio(), LogitLink(), eta)
    end
end

# Stopping-ratio per-threshold coefficients: several terms over several
# stages, on the formula and the authored-location routes.
function formula_threshold_effects(data)
    @brm data begin
        eta ~ 0 + x
        y ~ Ordinal(StoppingRatio(), LogitLink(), eta; per_threshold=(z, w, v))
    end
end

function authored_threshold_effects(data)
    @brm data begin
        slope ~ 0 + x
        bias ~ Normal(0, 1)
        loc = slope + bias
        y ~ Ordinal(StoppingRatio(), LogitLink(), loc; per_threshold=(w, z))
    end
end

# An authored simplex parameter read by an authored location.
function authored_simplex(data)
    @brm data begin
        s ~ Dirichlet([1.0, 2.0, 1.5])
        mu ~ 1 + x
        loc = mu + 0.8 * s[1] - s[3]
        y ~ Normal(loc, 1)
    end
end
end

# A downstream submodel provider supplying its own mathematics to both
# backends under the same declaration names.
module PublicSubmodelTransport
using BayesianRegressionModels, Distributions, StanBlocks
import BayesianRegressionModels: _rk_submodel_rhs!, _sb_submodel_rhs!

function shifted end
function renamed end
const ALPHA = [1.0, 2.0, 1.5, 0.8]

stan_inner = StanBlocks.@slic begin
    slope ~ normal(0, 1)
    return slope * x
end
stan_shifted = StanBlocks.@slic begin
    scale ~ lognormal(0, 1)
    offset ~ normal(0, 1)
    w::vector[3] ~ normal(0, 1)
    p::simplex[dims(alpha)[1]] ~ dirichlet(alpha)
    L::cholesky_factor_corr[3] ~ lkj_corr_cholesky(2.0)
    inner ~ stan_inner(; x=x)
    return location + offset + scale * x + inner +
        (w[1] + 2 * w[2] - w[3]) * p[1] + p[2] - p[4] + L[2, 1] + L[3, 2]
end

function stan_source!(statements, data, target, rhs)
    x = only(getargs(rhs))
    key, alpha = Symbol(target, :_source_x), Symbol(target, :_alpha)
    data[key] = copy(parent(parent(x)))
    data[alpha] = copy(ALPHA)
    location = name(getkwargs(rhs).location)
    push!(statements, :($target ~ stan_shifted(; x=$key, location=$location,
        alpha=$alpha)))
    :done
end
_sb_submodel_rhs!(statements, data, target::Symbol, ::typeof(shifted), rhs) =
    stan_source!(statements, data, target, rhs)
_sb_submodel_rhs!(statements, data, target::Symbol, ::typeof(renamed), rhs) =
    stan_source!(statements, data, target, rhs)

function native_source!(definitions, statements, data, target, rhs, offset)
    x = only(getargs(rhs))
    key, alpha = Symbol(target, :_source_x), Symbol(target, :_alpha)
    data[key] = copy(parent(parent(x)))
    data[alpha] = copy(ALPHA)
    location = name(getkwargs(rhs).location)
    inner, outer = Symbol(target, :_inner), Symbol(target, :_shifted)
    push!(definitions, :($inner(x) = begin
        slope ~ Normal(0, 1)
        return slope .* x
    end))
    push!(definitions, :($outer(x, location, alpha) = begin
        scale ~ LogNormal(0, 1)
        $offset ~ Normal(0, 1)
        w[1:3] .~ Normal(0, 1)
        p ~ Dirichlet(alpha)
        L ~ LKJCholesky(3, 2.0)
        inner ~ $inner(x)
        return location .+ $offset .+ scale .* x .+ inner .+
            (w[1] + 2 * w[2] - w[3]) * p[1] .+ p[2] .- p[4] .+ L[2, 1] .+ L[3, 2]
    end))
    push!(statements, :($target ~ $outer($key, $location, $alpha)))
    :done
end
_rk_submodel_rhs!(definitions, statements, data, bindings, target::Symbol,
    ::typeof(shifted), rhs) =
    native_source!(definitions, statements, data, target, rhs, :offset)
_rk_submodel_rhs!(definitions, statements, data, bindings, target::Symbol,
    ::typeof(renamed), rhs) =
    native_source!(definitions, statements, data, target, rhs, :shift)

build(data) = @brm data begin
    a ~ 0 + x
    effect(a, :) ~ Normal(0, 0.7)
    loc ~ shifted(x; location=a)
    y ~ Normal(loc, 0.8)
end
build_renamed(data) = @brm data begin
    a ~ 0 + x
    effect(a, :) ~ Normal(0, 0.7)
    loc ~ renamed(x; location=a)
    y ~ Normal(loc, 0.8)
end
end

function coordinate_transport_fixture(brmi, name; mod=PublicCoordinateTransport)
    rk, problem = consumer_problem(brmi)
    sb = SBBRMI(brmi; mod, total_groups=())
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
        stan_u = brm_rk_point_to_stan(transport, u)
        back = brm_stan_point_to_rk(transport, stan_u)
        if isempty(transport.simplexes)
            @test transport.logdensity_offset == 0
            @test isequal(brm_rk_to_stan(transport, u), stan_u)
            @test isequal(brm_stan_to_rk(transport, stan_u), u)
            @test isequal(back, u)
            @test isequal(brm_rk_to_stan(transport, brm_stan_to_rk(transport, stan_u)), stan_u)
        else
            @test back ≈ u atol=1e-12 rtol=1e-12
            @test brm_rk_point_to_stan(transport, back) ≈ stan_u atol=1e-12 rtol=1e-12
            # refused: a simplex block's free coordinates are a nonlinear map,
            # so the point permutation would silently mis-map a gradient.
            @test_throws "simplex block" brm_rk_to_stan(transport, u)
            @test_throws "simplex block" brm_stan_to_rk(transport, stan_u)
        end
        @test isequal(u, saved)
        checked = brm_check_coordinate_transport(transport, rk, stan.model, u)
        @test keys(checked) == (:pairs, :factors, :simplexes, :max_error)
        @test checked.pairs == count(p -> p.relation ∉ (:cholesky, :simplex), transport.pairs)
        @test checked.factors == length(transport.correlations)
        @test checked.simplexes == length(transport.simplexes)
        @test checked.max_error <= 1e-12
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        stan_gradient = similar(stan_u)
        stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
            stan_gradient; propto=false, jacobian=true)
        @test value ≈ stan_value + transport.logdensity_offset atol=2e-11 rtol=2e-11
        @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-10 rtol=2e-10
        @test isequal(u, saved)
    end
end

@stestset "coordinate records retain fresh block-local declarations" begin
    definitions, records, taken = Expr[], Any[], Set{Symbol}()
    address = (; kind=:ranef, group=:subject, id=nothing, bucket_kind=:intercept1,
        margins=((:mu, :Intercept),))
    BRM._rk_ast_varying_draws!(definitions, taken, 1, 1.0, nothing;
        group=:subject, scale_priors=:(Exponential(z)), coordinates=records,
        coordinate_record=address, scope=:draws)
    record = only(records)
    @test record.z !== Symbol("draws.z")
    definition = sprint(Base.show_unquoted, only(definitions))
    local_z = last(split(String(record.z), '.'))
    @test occursin("$(local_z)[levels(g), 1:1]", definition)
    nested = (; draws=NamedTuple{(Symbol(local_z),)}((reshape([.2, .3], 2, 1),)))
    @test BRM._brm_rk_declaration_value(nested, record.z) == reshape([.2, .3], 2, 1)
end

@stestset "allocating blocks retain caller-side hyperparameter priors" begin
    definitions, records, taken = Expr[], Any[], Set{Symbol}([:z])
    address = (; kind=:ranef, group=:subject, id=nothing, bucket_kind=:intercept1,
        margins=((:mu, :Intercept),))
    call = BRM._rk_ast_varying_draws!(definitions, taken, 1, 1.0, nothing;
        group=:subject, scale_priors=:(Exponential(z)), coordinates=records,
        coordinate_record=address, scope=:draws)
    emitted = BRM._rk_fitted_source(BRM._RKEmittedProgram(definitions, quote
        z ~ Exponential(.8)
        draws ~ $call
        mu = draws[subject, 1]
        y .~ Normal.(mu, 1)
    end), (:y,))
    parsed = BRM._RKEmittedProgram(
        [Meta.parse(sprint(Base.show_unquoted, definition)) for definition in emitted.defs],
        Meta.parse(sprint(Base.show_unquoted, emitted.main)))
    data = Dict(:subject => [1, 2], :y => [.2, -.3])
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    record = only(records)
    for program in (emitted, parsed)
        translated = bind_data(lower_rkppl(program.main, data;
            mod=ext._rk_emit_module(program), conditioned=(:y,)), data)
        model = Base.invokelatest(build_kernel, translated)
        names = coordinate_names(model.layout)
        index(name) = only(findall(==(Symbol(name)), names))
        hyper = index(:z)
        scale = index(string(record.scale, ".1"))
        innovations = [index(string(record.z, ".", j, ".1")) for j in 1:2]
        @test length(names) == 4
        query = prepare_sampler(model, translated, zeros(4);
            backend=AutoEnzyme(; mode=Enzyme.Reverse))
        for u in (zeros(4), fill(.13, 4), collect(range(-.2, .3; length=4)))
            saved = copy(u)
            z, sd, v = exp(u[hyper]), exp(u[scale]), u[innovations]
            residual = data[:y] .- sd .* v
            oracle = logpdf(Exponential(.8), z) + u[hyper] +
                logpdf(Exponential(z), sd) + u[scale] +
                sum(logpdf.(Normal(), v)) + sum(logpdf.(Normal.(sd .* v, 1), data[:y]))
            expected = zeros(4)
            expected[hyper] = -z/.8 + sd/z
            expected[scale] = 1 - sd/z + sum(residual .* sd .* v)
            expected[innovations] = -v .+ sd .* residual
            gradient = similar(u)
            value, _ = sampler_value_and_gradient!(query, gradient, u)
            @test value ≈ oracle atol=2e-11 rtol=2e-11
            @test gradient ≈ expected atol=2e-10 rtol=2e-10
            @test isequal(u, saved)
        end
    end
end

@stestset "public random-effect scale hyperprior remains a fitted draw" begin
    data = (; g=[1, 1, 2, 2], y=[.2, -.1, .4, .3])
    brmi = @brm data begin
        tau ~ Exponential(1)
        mu ~ 1 + (1 | p | g)
        sd(:, p) ~ Exponential(tau)
        y ~ Normal(mu, 1)
    end
    # consumer_problem independently reparses every emitted definition and
    # the complete main program before preparing the ordinary Reverse query.
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    index(name) = only(findall(==(Symbol(name)), names))
    record = only(filter(r -> r.kind === :ranef,
        BRM._rk_coordinate_records(backend.plan)))
    hyper, scale, population = index(:tau), index(string(record.scale, ".1")),
        index("pop_mu.beta_pop.1")
    innovations = [index(string(record.z, ".", j, ".1")) for j in 1:2]
    @test length(names) == 5
    for u in (zeros(5), fill(.13, 5), collect(range(-.2, .3; length=5)))
        saved = copy(u)
        tau, sd, beta, z = exp(u[hyper]), exp(u[scale]), u[population], u[innovations]
        means = beta .+ sd .* z[data.g]
        residual = data.y .- means
        oracle = logpdf(Exponential(1), tau) + u[hyper] +
            logpdf(Exponential(tau), sd) + u[scale] + logpdf(Normal(), beta) +
            sum(logpdf.(Normal(), z)) + sum(logpdf.(Normal.(means, 1), data.y))
        expected = zeros(5)
        expected[hyper] = -tau + sd / tau
        expected[scale] = 1 - sd / tau + sum(residual .* sd .* z[data.g])
        expected[population] = -beta + sum(residual)
        expected[innovations] = [-z[j] + sd * sum(residual[data.g .== j]) for j in 1:2]
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test value ≈ oracle atol=2e-11 rtol=2e-11
        @test gradient ≈ expected atol=2e-10 rtol=2e-10
        @test isequal(u, saved)
    end
end

@stestset "overridden scale hyperprior remains outside fitted coordinates" begin
    data = (; g=[1, 1, 2, 2], y=[.2, -.1, .4, .3])
    brmi = @brm data begin
        unused ~ Exponential(1)
        mu ~ 1 + (1 | p | g)
        sd(:, p) ~ Exponential(unused)
        sd(mu, p, Intercept) ~ Exponential(.7)
        y ~ Normal(mu, 1)
    end
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    index(name) = only(findall(==(Symbol(name)), names))
    record = only(filter(r -> r.kind === :ranef,
        BRM._rk_coordinate_records(backend.plan)))
    scale, population = index(string(record.scale, ".1")), index("pop_mu.beta_pop.1")
    innovations = [index(string(record.z, ".", j, ".1")) for j in 1:2]
    @test length(names) == 4
    @test :unused ∉ names
    for u in (zeros(4), fill(.13, 4), collect(range(-.2, .3; length=4)))
        saved = copy(u)
        sd, beta, z = exp(u[scale]), u[population], u[innovations]
        means = beta .+ sd .* z[data.g]
        residual = data.y .- means
        oracle = logpdf(Exponential(.7), sd) + u[scale] +
            logpdf(Normal(), beta) + sum(logpdf.(Normal(), z)) +
            sum(logpdf.(Normal.(means, 1), data.y))
        expected = zeros(4)
        expected[scale] = 1 - sd / .7 + sum(residual .* sd .* z[data.g])
        expected[population] = -beta + sum(residual)
        expected[innovations] = [-z[j] + sd * sum(residual[data.g .== j]) for j in 1:2]
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test value ≈ oracle atol=2e-11 rtol=2e-11
        @test gradient ≈ expected atol=2e-10 rtol=2e-10
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

@stestset "coordinate transport fixed-domain HSGP internals" begin
    data = (; x=[-0.7,0.0,0.6], y=[0.2,-0.1,0.4])
    fixture = coordinate_transport_fixture(PublicCoordinateTransport.smooth(data), "hsgp")
    @test Set(p.address.parameter for p in fixture.transport.pairs) ==
        Set([:length_scale, :sd, :basis_weights])
    @test count(p -> p.address.parameter === :basis_weights, fixture.transport.pairs) == 3
    check_coordinate_transport(fixture)
end

@stestset "coordinate transport monotonic and HSGP terms across linked predictors" begin
    data = (; r=[1,4,2,3,1,4,2,3], q=[2,1,3,3,1,2,2,1],
        x=[-0.7,0.0,0.6,0.9,0.2,-0.4,0.5,-0.1], z=[0.3,-0.2,0.5,0.1,-0.4,0.6,0.0,0.2],
        y=[0.2,-0.1,0.4,0.3,0.1,-0.3,0.5,0.0], n=[1,4,0,3,2,5,1,2])
    saved = deepcopy(data)
    fixture = coordinate_transport_fixture(PublicCoordinateTransport.ordinal(data), "ordinal")
    (; transport) = fixture
    # Three monotonic simplexes: `mo(r)` on each predictor (4 levels, 3
    # increments) and `mo1(q)` (3 levels, 2 increments).
    @test sort([b.K for b in transport.simplexes]) == [2, 3, 3]
    @test transport.logdensity_offset ≈ -(log(2) + 2log(3)) / 2
    @test Set(b.address.predictor for b in transport.simplexes) == Set([:mu, :rate])
    @test count(p -> p.address.kind === :population &&
        p.address.coefficient === :mo_r, transport.pairs) == 2
    # The repeated terms keep their semantic addresses whatever each
    # backend's collision-free carrier names are.
    hsgp_pairs = filter(p -> p.address.kind === :hsgp, transport.pairs)
    @test Set((p.address.predictor, p.address.term) for p in hsgp_pairs) ==
        Set([(:mu, :hsgp_x), (:mu, :hsgp_x_z), (:rate, :hsgp_x)])
    @test count(p -> p.address.term === :hsgp_x_z &&
        p.address.parameter === :length_scale, hsgp_pairs) == 2
    check_coordinate_transport(fixture)
    @test isequal(data, saved)
end

@stestset "coordinate transport custom submodel provider declarations" begin
    data = (; x=[-0.4, 0.2, 0.7, 0.1], y=[0.1, -0.2, 0.3, 0.4])
    saved = deepcopy(data)
    fixture = coordinate_transport_fixture(PublicSubmodelTransport.build(data),
        "provider"; mod=PublicSubmodelTransport)
    (; transport) = fixture
    provider = filter(p -> p.address.kind === :submodel, transport.pairs)
    @test Set(p.address.declaration for p in provider) == Set(Symbol.(
        ["loc.scale", "loc.offset", "loc.w", "loc.p", "loc.L", "loc.inner.slope"]))
    @test only(transport.simplexes).K == 4
    @test only(transport.correlations).K == 3
    @test only(p.stan for p in provider
        if p.address.declaration === Symbol("loc.inner.slope")) == "loc_inner_slope"
    check_coordinate_transport(fixture)
    @test isequal(data, saved)

    # A provider pair that spells one declaration differently on the two
    # backends has no shared address: both coordinates are named, no map.
    brmi = PublicSubmodelTransport.build_renamed(data)
    rk = RKBRMI(brmi)
    sb = SBBRMI(brmi; mod=PublicSubmodelTransport, total_groups=())
    stan = BRM.stan_instantiate(sb; path=joinpath(tempdir(),
        "brm-coordinate-transport", "provider-renamed.stan"))
    err = try
        brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model))
    catch e
        e
    end
    @test err isa BRMCoordinateTransportError
    @test err.reason === :unsupported_coverage
    @test occursin("loc.shift", sprint(showerror, err))
    @test occursin("loc_offset", sprint(showerror, err))
end

@stestset "simplex maps match both backends' transforms exactly" begin
    for K in (2, 3, 5)
        for u in (zeros(K - 1), [-0.8 + 0.47j for j in 1:(K - 1)],
                [0.6cos(3j) for j in 1:(K - 1)])
            saved = copy(u)
            logx, z = BRM._brm_rk_simplex_log(u)
            @test exp.(logx) ≈ ReactiveKernelsPPL.simplex_constrain(u) atol=1e-14
            @test BRM._brm_rk_simplex_free(logx) ≈ u atol=1e-12
            y = BRM._brm_stan_simplex_free(logx)
            @test BRM._brm_stan_simplex_log(y) ≈ logx atol=1e-12
            # Both log-Jacobians reduce to `sum(log.(x))`, up to Stan's
            # constant `log(K) / 2`: the transport's density offset.
            @test ReactiveKernelsPPL.simplex_logjac(u) ≈ sum(logx) atol=1e-12
            # Independent central differences of the composed map.
            g = [0.3, -1.2, 0.7, 0.25][1:(K - 1)]
            step = 1e-6
            jacobian = reduce(hcat, map(1:(K - 1)) do i
                plus, minus = copy(u), copy(u)
                plus[i] += step; minus[i] -= step
                (BRM._brm_stan_simplex_free(first(BRM._brm_rk_simplex_log(plus))) .-
                 BRM._brm_stan_simplex_free(first(BRM._brm_rk_simplex_log(minus)))) ./ (2step)
            end)
            @test BRM._brm_simplex_pullback(u, g) ≈ transpose(jacobian) * g atol=1e-8
            @test isequal(u, saved)
        end
    end
end

@stestset "coordinate transport response threshold vectors" begin
    # Gappy raw codes for OrderedLogistic on an authored location (K = max = 4,
    # as SBBRMI); contiguous codes on the formula route; fitted 1:3 levels for
    # Ordinal.
    data = (; x=[-1.2, -0.4, 0.1, 0.5, 0.9, 1.4, -0.8, 0.3],
        g=[1, 1, 2, 2, 3, 3, 1, 2],
        y=[1, 2, 2, 4, 4, 4, 1, 2])
    contiguous = merge(data, (; y=[1, 2, 2, 3, 3, 3, 1, 2]))
    saved = deepcopy(data)
    cases = (
        (PublicCoordinateTransport.authored_cutpoints, data, :y_cutpoints, 3, :ordered),
        (PublicCoordinateTransport.formula_cutpoints, contiguous, :y_cutpoints, 2, :ordered),
        (PublicCoordinateTransport.cumulative_thresholds, data, :y_thresholds, 2, :ordered),
        (PublicCoordinateTransport.stopping_thresholds, data, :y_thresholds, 2, :identity))
    for (build, input, declaration, n, transform) in cases
        @testset "$(nameof(build))" begin
            fixture = coordinate_transport_fixture(build(input), String(nameof(build)))
            (; rk, transport) = fixture
            vector = filter(p -> p.address.kind === :vector, transport.pairs)
            @test [p.address for p in vector] ==
                [(; kind=:vector, declaration, index=j) for j in 1:n]
            @test [p.stan for p in vector] == ["$declaration.$j" for j in 1:n]
            @test all(p -> p.relation === :identity, vector)
            positions = [only(findall(==(p.rk), transport.rk_names)) for p in vector]
            @test all(==(transform), BRM._rk_layout_coordinate_transforms(rk)[positions])
            @test isempty(transport.simplexes)
            check_coordinate_transport(fixture)
        end
    end
    @test isequal(data, saved)
end

@stestset "coordinate transport authored simplex parameter" begin
    data = (; x=[-0.4, 0.1, 0.3, 0.6, -0.2], y=[0.2, -0.1, 0.4, 0.3, 0.0])
    saved = deepcopy(data)
    fixture = coordinate_transport_fixture(
        PublicCoordinateTransport.authored_simplex(data), "authored-simplex")
    (; transport) = fixture
    block = only(transport.simplexes)
    @test block.address == (; kind=:vector, declaration=:s)
    @test block.K == 3
    @test transport.logdensity_offset ≈ -log(3) / 2
    check_coordinate_transport(fixture)
    @test isequal(data, saved)
end

@stestset "coordinate transport per-threshold ordinal coefficients" begin
    data = (; x=[-1.2, -0.4, 0.1, 0.5, 0.9, 1.4, -0.8, 0.3],
        z=[0.3, -0.2, 0.5, 0.1, -0.7, 0.9, 0.2, -0.4],
        w=[1.0, 0.5, -0.3, 0.2, 0.8, -1.1, 0.4, 0.0],
        v=[0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8],
        y=[1, 2, 2, 4, 4, 4, 1, 2])
    saved = deepcopy(data)
    for (build, terms) in ((PublicCoordinateTransport.formula_threshold_effects, (:z, :w, :v)),
            (PublicCoordinateTransport.authored_threshold_effects, (:w, :z)))
        @testset "$(nameof(build))" begin
            fixture = coordinate_transport_fixture(build(data), String(nameof(build)))
            (; transport) = fixture
            coefficients = filter(p -> p.address.kind === :threshold_coefficient,
                transport.pairs)
            # Fitted levels 1:3 give two stages; RK's `terms x stages` matrix
            # element (t, s) is Stan's `y_threshold_beta[s][t]`.
            @test Set((p.address.stage, p.address.term) for p in coefficients) ==
                Set((s, t) for s in 1:2 for t in terms)
            for p in coefficients
                t = findfirst(==(p.address.term), terms)
                @test p.address.response === :y
                @test p.rk === Symbol("y_threshold_beta.$t.$(p.address.stage)")
                @test p.rk_index == (t, p.address.stage)
                @test p.stan_value == "y_threshold_beta.$(p.address.stage).$t"
            end
            # BridgeStan lists this array of vectors stage-fastest, so with
            # several terms some names differ from the element they hold.
            @test any(p -> p.stan != p.stan_value, coefficients)
            check_coordinate_transport(fixture)
        end
    end
    @test isequal(data, saved)
end
