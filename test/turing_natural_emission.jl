using Test, BayesianRegressionModels, Turing, Distributions, Random

const BRM = BayesianRegressionModels

Turing.@model function expert_gaussian(y, X)
    sigma ~ Exponential(1)
    beta_pop ~ product_distribution(fill(Normal(), size(X, 2)))
    mu = X * beta_pop
    for i in eachindex(y)
        y[i] ~ Normal(mu[i], sigma)
    end
    (; mu, sigma, response=y)
end

function observation_rhs(node, response=:y)
    node isa Expr || return nothing
    if Meta.isexpr(node, :call) && first(node.args) === (~)
        node.args[2] == :($response[i]) && return node.args[3]
    elseif Meta.isexpr(node, :call) && first(node.args) === :~
        node.args[2] == :($response[i]) && return node.args[3]
    end
    for child in node.args
        found = observation_rhs(child, response)
        isnothing(found) || return found
    end
    nothing
end

function assert_natural_inputs(backend)
    source = string(turing_model_source(backend))
    @test !occursin("_brm_generic_observation", source)
    @test !occursin("multi.plans", source)
    @test !occursin("response_modifier", source)
    @test !occursin("observation_weight", source)
    @test !hasproperty(backend.model.args, :group_models)
    @test !hasproperty(backend.model.args, :term_models)
    source
end

@testset "ordinary emitted Gaussian is an ordinary Turing model" begin
    data = (; x=[-1., 0.5, 2.], y=[-0.2, 0.4, 1.2])
    backend = TuringBRMI((@brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end)(data))
    source = assert_natural_inputs(backend)
    @test !occursin("identity.", source)
    @test !occursin("offset_mu", source)
    @test keys(backend.model.args) == (:y, :X_mu)
    rhs = observation_rhs(turing_model_source(backend))
    @test rhs.args[1] == GlobalRef(Distributions, :Normal)
    @test rhs.args[2:end] == Any[:(mu[i]), :sigma]
    parameters = (; sigma=0.7, beta_pop=[0.2, 0.3])
    expert = expert_gaussian(data.y, backend.plan.design.matrix)
    @test Turing.logprior(backend.model, parameters) ≈ Turing.logprior(expert, parameters)
    @test Turing.loglikelihood(backend.model, parameters) ≈
        Turing.loglikelihood(expert, parameters)
    @test Turing.logjoint(backend.model, parameters) ≈ Turing.logjoint(expert, parameters)
    @test turing_generated_quantities(backend, parameters) ==
        Turing.DynamicPPL.returned(expert, parameters)
    # The displayed source is the executable model, not a prettified surrogate.
    constructor = Core.eval(@__MODULE__, deepcopy(turing_model_source(backend)))
    roundtrip = Base.invokelatest(constructor, values(backend.model.args)...)
    @test Turing.logjoint(roundtrip, parameters) ≈ Turing.logjoint(expert, parameters)
    @test turing_posterior_predictive(Xoshiro(31), backend, parameters).y ==
        rand.(Ref(Xoshiro(31)), Normal.(data.x .* 0.3 .+ 0.2, 0.7))
end

@testset "generated input names do not capture model bindings" begin
    backend = TuringBRMI((@brm begin
        X_mu ~ Normal()
        mu ~ 1 + x
        y ~ Normal(mu + X_mu, 1)
    end)((; x=[-1., 2.], y=[0.2, 0.6])))
    @test !hasproperty(backend.model.args, :X_mu)
    parameters = (; X_mu=0.3, beta_pop=[0.1, 0.2])
    expected = sum(logpdf.(Normal.([0.2, 0.8], 1), [0.2, 0.6]))
    @test Turing.loglikelihood(backend.model, parameters) ≈ expected
end

@testset "data bindings cannot capture generated calls, loops or latent sites" begin
    response_collision = TuringBRMI((@brm begin
        mu ~ 1 + y
        z ~ Normal(mu, 1)
    end)((; y=[1., 2.], z=[0.1, 0.2])))
    parameters = (; beta_pop=[0., 1.])
    expected = sum(logpdf.(Normal.([1., 2.], 1), [0.1, 0.2]))
    @test Turing.loglikelihood(response_collision.model, parameters) ≈ expected
    @test length(turing_posterior_predictive(Xoshiro(33), response_collision, parameters).z) == 2

    function_collision = TuringBRMI((@brm begin
        mu ~ 1 + fill
        y ~ Normal(mu, 1)
    end)((; fill=[1., 2.], y=[0.1, 0.2])))
    @test !hasproperty(function_collision.model.args, :fill)
    @test Turing.loglikelihood(function_collision.model, parameters) ≈ expected

    loop_collision = TuringBRMI((@brm begin
        y ~ Normal(i, 1)
    end)((; i=[1., 2.], y=[0.1, 0.2])))
    @test Turing.loglikelihood(loop_collision.model, (;)) ≈ expected

    design_collision = TuringBRMI((@brm begin
        mu ~ 1 + beta_pop
        y ~ Normal(mu, 1)
    end)((; beta_pop=[1., 2.], y=[0.1, 0.2])))
    @test !hasproperty(design_collision.model.args, :beta_pop)
    @test Turing.logjoint(design_collision.model, parameters) ≈
        expected + sum(logpdf.(Normal(), parameters.beta_pop))

    used_collision = TuringBRMI((@brm begin
        mu ~ 1
        y ~ Normal(mu + beta_pop, 1)
    end)((; beta_pop=[1., 2.], y=[0.1, 0.2])))
    @test !hasproperty(used_collision.model.args, :beta_pop)
    @test Turing.loglikelihood(used_collision.model, (; beta_pop=[0.])) ≈ expected

    preparation_names = TuringBRMI((@brm begin
        y ~ Normal(multi + callables, 1)
    end)((; multi=[0.5, 1.], callables=[0.5, 1.], y=[0.1, 0.2])))
    @test Turing.loglikelihood(preparation_names.model, (;)) ≈ expected
end

@testset "bounds compose directly and fitted inputs remain live in replay" begin
    data = (; x=[-1., 0.5, 2.], y=[-0.2, 0.4, 1.2],
             lo=[-1., -0.5, 0.], hi=[1., 1.5, 2.])
    builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ truncated(Normal(mu, sigma); lower=lo, upper=hi)
    end
    backend = TuringBRMI(builder(data))
    source = assert_natural_inputs(backend)
    rhs = observation_rhs(turing_model_source(backend))
    @test rhs.args[1] == GlobalRef(Distributions, :truncated)
    @test occursin("lower_y[i]", source)
    @test occursin("upper_y[i]", source)
    parameters = (; sigma=0.7, beta_pop=[0.2, 0.3])
    for current in (backend, reprocess(backend, merge(data, (; hi=data.hi .+ 0.5))))
        mu = current.plan.design.matrix * parameters.beta_pop
        modifier = current.plan.response_modifier
        distributions = [truncated(Normal(mu[i], parameters.sigma);
            lower=modifier.lower[i], upper=modifier.upper[i]) for i in eachindex(mu)]
        expected = logpdf.(distributions, data.y)
        @test Turing.loglikelihood(current.model, parameters) ≈ sum(expected)
        @test turing_pointwise_loglikelihoods(current, parameters).y ≈ expected
        @test turing_posterior_predictive(Xoshiro(32), current, parameters).y ==
            rand.(Ref(Xoshiro(32)), distributions)
    end
end

natural_normal_factory(mu; width) = Normal(mu, width)

@testset "precision AST keeps direct Normal and arbitrary factory calls" begin
    data = (; y=[0.2, -0.3], w=[0.5, 2.])
    direct = TuringBRMI((@brm begin
        sigma ~ Exponential(1)
        y ~ weighted(Normal(0, sigma), aweights(w))
    end)(data))
    factory = TuringBRMI((@brm begin
        sigma ~ Exponential(1)
        y ~ weighted(natural_normal_factory(0; width=sigma), aweights(w))
    end)(data))
    for backend in (direct, factory)
        assert_natural_inputs(backend)
        expected = logpdf.(Normal.(0, 0.7 ./ sqrt.(data.w)), data.y)
        @test Turing.loglikelihood(backend.model, (; sigma=0.7)) ≈ sum(expected)
        @test turing_pointwise_loglikelihoods(backend, (; sigma=0.7)).y ≈ expected
    end
    @test observation_rhs(turing_model_source(direct)).args[3] ==
        Expr(:call, GlobalRef(Base, :/), :sigma,
             Expr(:call, GlobalRef(Base, :sqrt), :(weights_y[i])))
    @test occursin("natural_normal_factory", string(turing_model_source(factory)))
end

@testset "callable captures are explicit arguments and retain cache identity" begin
    function build_captured(shift)
        family = mu -> Normal(mu + shift, 0.8)
        TuringBRMI((@brm begin
            location ~ Normal()
            y ~ family(location)
        end)((; y=[0.1, 0.4])))
    end
    first_backend, second_backend = build_captured(0.2), build_captured(1.3)
    for (backend, shift) in ((first_backend, 0.2), (second_backend, 1.3))
        assert_natural_inputs(backend)
        @test !hasproperty(backend.model.args, :callables)
        @test Turing.loglikelihood(backend.model, (; location=0.3)) ≈
            sum(logpdf.(Normal(0.3 + shift, 0.8), [0.1, 0.4]))
    end
end
