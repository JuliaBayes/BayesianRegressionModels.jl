using Test, BayesianRegressionModels, Distributions, Turing
import StanBlocks
import LogDensityProblems

const BRM = BayesianRegressionModels

plain_factory(location, scale) = Normal(location, scale)
keyword_factory(location, scale; shift=0.0, blend=0.0) =
    Normal(location + shift + blend, scale)
vector_factory() = Dirichlet([1.0, 1.0])

BRM.brm_distribution_type(::typeof(plain_factory)) = Normal
BRM.brm_distribution_type(::typeof(keyword_factory)) = Normal
BRM.brm_distribution_type(::typeof(vector_factory)) = Dirichlet
BRM._sb_stan_dist_name(::typeof(plain_factory)) = :registered_plain
function BRM._sb_stan_distribution_call(::typeof(keyword_factory), args, kwargs)
    location = :($(args[1]) + $(kwargs.shift) + $(kwargs.blend))
    Expr(:call, :registered_plain, location, args[2])
end

StanBlocks.@deffun begin
    @lpxf registered_plain_lpdf(y::real, location::real, scale::real)::real =
        normal_lpdf(y, location, scale)
    registered_plain_rng(location::real, scale::real)::real =
        normal_rng(location, scale)
    registered_plain_lpdf(y::vector[n], location::real,
                          scale::real)::real = normal_lpdf(y, location, scale)
    registered_plain_lpdfs(y::vector[n], location::real,
                           scale::real)::vector[n] = normal_lpdfs(y, location, scale)
    registered_plain_rng(vector[n], location::real,
                         scale::real)::vector[n] = begin
        out::vector[n]
        for i in 1:n
            out[i] = normal_rng(location, scale)
        end
        out
    end
end

const CALLABLE_CACHE = joinpath(tempdir(), "brm-callable-priors")

@testset "callable scalar shape admission in structured models" begin
    r2_builder = @brm begin
        reference_scale ~ plain_factory(0.7, 0.2; lower=0)
        mu ~ 1 + (1 | p | subject)
        sd(:, p) ~ r2d2(R2=Beta(2, 2), reference_scale=reference_scale)
        y ~ Normal(mu, 1)
    end
    r2_data = (; subject=["a", "a", "b"], y=[0.1, -0.2, 0.3])
    r2_code = BRM.stan_code(SBBRMI(r2_builder(r2_data); mod=@__MODULE__))
    @test occursin("reference_scale ~ registered_plain(0.7, 0.2);", r2_code)
    @test count("reference_scale ~ registered_plain", r2_code) == 1
    @test StanBlocks.stanc_check(r2_code; warn_pedantic=false).ok

    joint_data = (; y1=[0.1, 0.2], y2=[1.1, 0.9])
    joint_builder = @brm begin
        shared_mean ~ plain_factory(0.0, 1.0)
        L_res ~ LKJCovarianceFactor(2)
        [y1, y2] ~ MvNormalCholesky([shared_mean, 0.0], L_res)
    end
    joint = SBBRMI(joint_builder(joint_data); mod=@__MODULE__)
    @test joint isa SBBRMI

    @test !BRM._sb_is_scalar_prior(BRM.ExprColumn(vector_factory))

    lkj_mean = @brm begin
        bad_mean ~ LKJCovarianceFactor(2)
        L_res ~ LKJCovarianceFactor(2)
        [y1, y2] ~ MvNormalCholesky([bad_mean, 0.0], L_res)
    end
    @test_throws "must be scalar" SBBRMI(lkj_mean(joint_data); mod=@__MODULE__)
end

@testset "callable homogeneous ranef scale uses its model-module family" begin
    # A one-margin shared-ID block is homogeneous but cannot lower to a native
    # vectorized call when the family itself is callable. Resolve the registered
    # consumer-module function for the generated vector family's RNG rather
    # than assuming every distribution token lives in StanBlocks.
    builder = @brm begin
        eta ~ 1 + (1 | ri | subject)
        sd(:, ri) ~ plain_factory(0.0, 2.0)
        y ~ Normal(eta, 1)
    end
    data = (; subject=["a", "a", "b"], y=[0.1, -0.2, 0.3])
    sb = SBBRMI(builder(data); mod=@__MODULE__)
    code = BRM.stan_code(sb)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin(r"brm_vector_prior_[0-9a-f]+_lpdf", code)
    @test occursin("registered_plain_lpdf(x[1]", code)
end

@testset "registered callable prior factory" begin
    builder = @brm begin
        location ~ Normal(0, 1)
        scale ~ Exponential(2)
        theta ~ plain_factory(location, scale; lower=-1, upper=1)
        y ~ Normal(theta, 1)
    end
    data = (; y=[0.1, -0.2])
    descriptor = brm_descriptor(builder, data; mod=@__MODULE__, highlights=())
    code = BRM.stan_code(descriptor.plan)
    @test occursin("real<lower=-1.0, upper=1.0> theta;", code)
    @test occursin("theta ~ registered_plain(location, scale);", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    mkpath(CALLABLE_CACHE)
    problem = brm_execute(descriptor, :instantiate;
        path=joinpath(CALLABLE_CACHE, string(descriptor.id) * ".stan"))
    raw = zeros(3)
    names = StanBlocks.BridgeStan.param_names(problem.model)
    values = StanBlocks.BridgeStan.param_constrain(problem.model, raw)
    physical = Dict(zip(names, values))
    params = (; location=physical["location"], scale=physical["scale"],
               theta=physical["theta"])
    expected = logpdf(Normal(), params.location) +
        logpdf(Exponential(2), params.scale) +
        logpdf(Normal(params.location, params.scale), params.theta) +
        sum(logpdf.(Normal(params.theta, 1), data.y))
    @test StanBlocks.BridgeStan.log_density(problem.model, raw;
        propto=false, jacobian=false) ≈ expected
    backend = TuringBRMI(builder(data))
    @test Turing.logjoint(backend.model, params) ≈ expected

    draw = StanBlocks.BridgeStan.param_constrain(problem.model, zeros(3);
        rng=StanBlocks.BridgeStan.StanRNG(problem.model, 912))
    @test all(isfinite, draw)
end

@testset "callable AST translation retains two keywords for prior and observation" begin
    builder = @brm begin
        location ~ Normal(0, 1)
        scale ~ Exponential(2)
        theta ~ keyword_factory(location, scale;
            shift=0.2, blend=0.3, lower=-1, upper=1)
        y ~ keyword_factory(theta, 1; shift=0.1, blend=-0.1)
    end
    data = (; y=[0.1, -0.2])

    backend = TuringBRMI(builder(data))
    params = (; location=0.1, scale=0.7, theta=0.25)
    expected = logpdf(Normal(), params.location) +
        logpdf(Exponential(2), params.scale) +
        logpdf(keyword_factory(params.location, params.scale;
            shift=0.2, blend=0.3), params.theta) +
        sum(logpdf.(keyword_factory(params.theta, 1;
            shift=0.1, blend=-0.1), data.y))
    @test Turing.logjoint(backend.model, params) ≈ expected

    sb = SBBRMI(builder(data); mod=@__MODULE__)
    code = BRM.stan_code(sb)
    @test occursin("theta ~ registered_plain((location + 0.2 + 0.3), scale);", code)
    @test occursin("y ~ registered_plain((theta + 0.1 + -0.1), 1);", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    problem = StanBlocks.stan_instantiate(sb.model;
        path=joinpath(CALLABLE_CACHE, string(hash(code)) * ".stan"))
    raw = zeros(LogDensityProblems.dimension(problem))
    names = StanBlocks.BridgeStan.param_names(problem.model)
    values = StanBlocks.BridgeStan.param_constrain(problem.model, raw)
    physical = Dict(zip(names, values))
    sb_expected = logpdf(Normal(), physical["location"]) +
        logpdf(Exponential(2), physical["scale"]) +
        logpdf(keyword_factory(physical["location"], physical["scale"];
            shift=0.2, blend=0.3), physical["theta"]) +
        sum(logpdf.(keyword_factory(physical["theta"], 1;
            shift=0.1, blend=-0.1), data.y))
    @test StanBlocks.BridgeStan.log_density(problem.model, raw;
        propto=false, jacobian=false) ≈ sb_expected
end
