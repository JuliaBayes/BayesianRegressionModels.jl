using Test, BayesianRegressionModels, Distributions, Turing
import StanBlocks

StanBlocks.@deffun StanBlocks.@juliacompat prior_scale(
    value::real; extra=0.0, gain=1.0)::real = gain * value + extra

@testset "prior and observation argument rewrites retain callable keywords" begin
    builder = @brm begin
        sigma ~ LogNormal(0, 0.3)
        location ~ Normal(0, prior_scale(sigma; extra=0.2, gain=1.5))
        y ~ Normal(location, prior_scale(sigma; extra=0.1, gain=1.1))
    end
    data = (; y=[0.1, -0.2])
    descriptor = brm_descriptor(builder, data; mod=@__MODULE__, highlights=())
    code = BayesianRegressionModels.stan_code(descriptor.plan)
    @test StanBlocks.stanc_check(code).ok
    cache = joinpath(tempdir(), "brm-prior-expression-keywords")
    mkpath(cache)
    problem = brm_execute(descriptor, :instantiate;
        path=joinpath(cache, string(descriptor.id) * ".stan"))
    raw = zeros(2)
    names = StanBlocks.BridgeStan.param_names(problem.model)
    values = StanBlocks.BridgeStan.param_constrain(problem.model, raw)
    physical = Dict(zip(names, values))
    params = (; sigma=physical["sigma"], location=physical["location"])
    expected = logpdf(LogNormal(0, 0.3), params.sigma) +
        logpdf(Normal(0, prior_scale(params.sigma; extra=0.2, gain=1.5)), params.location) +
        sum(logpdf.(Normal(params.location,
            prior_scale(params.sigma; extra=0.1, gain=1.1)), data.y))
    @test StanBlocks.BridgeStan.log_density(problem.model, raw;
        propto=false, jacobian=false) ≈ expected
    backend = TuringBRMI(builder(data))
    @test Turing.logjoint(backend.model, params) ≈ expected
end
