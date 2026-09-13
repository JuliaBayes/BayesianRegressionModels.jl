using Test, BayesianRegressionModels, Distributions, Turing, LogDensityProblems
import StanBlocks
const BRM = BayesianRegressionModels
const BridgeStan = StanBlocks.BridgeStan

@testset "parameter-dependent scalar declaration bounds retain their kernel" begin
    dependent_location = @brm begin
        location ~ Normal(0, 1)
        bounded ~ Normal(0.2, 0.7; lower=location, upper=location + 2)
        y ~ Normal(bounded + location, 0.8)
    end
    constraint_only = @brm begin
        location ~ Normal(0, 1)
        bounded ~ Normal(0.2, 0.7; lower=location, upper=location + 2)
        y ~ Normal(bounded, 0.8)
    end
    data = (; y=[0.1, 0.5])
    for (builder, use_location) in ((dependent_location, true), (constraint_only, false))
        descriptor = brm_descriptor(builder, data; mod=@__MODULE__, highlights=())
        @test StanBlocks.stanc_check(BRM.stan_code(descriptor.plan)).ok
        cache = joinpath(tempdir(), "brm-hierarchical-bounds")
        mkpath(cache)
        problem = brm_execute(descriptor, :instantiate;
            path=joinpath(cache, string(descriptor.id) * ".stan"))
        backend = TuringBRMI(builder(data))
        @test LogDensityProblems.dimension(problem) == 2
        for raw in ([-0.3, 0.1], [0.4, -0.5])
            names = BridgeStan.param_names(problem.model)
            values = BridgeStan.param_constrain(problem.model, raw)
            physical = Dict(zip(names, values))
            parameters = (; location=physical["location"], bounded=physical["bounded"])
            @test parameters.location < parameters.bounded < parameters.location + 2
            mean = parameters.bounded + (use_location ? parameters.location : 0)
            expected = logpdf(Normal(), parameters.location) +
                logpdf(Normal(0.2, 0.7), parameters.bounded) +
                sum(logpdf.(Normal(mean, 0.8), data.y))
            @test Turing.logjoint(backend.model, parameters) ≈ expected
            @test BridgeStan.log_density(problem.model, raw; propto=false, jacobian=false) ≈ expected
        end
    end
end
