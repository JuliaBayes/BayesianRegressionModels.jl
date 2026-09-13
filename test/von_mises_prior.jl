using Test, BayesianRegressionModels, Distributions, Turing, LogDensityProblems
import StanBlocks
const BRM = BayesianRegressionModels
const BridgeStan = StanBlocks.BridgeStan

@testset "circular prior preserves moving support and density" begin
    ordinary = @brm begin
        location ~ Normal(7.0, 1.0)
        theta ~ VonMises(location, 1.7)
        y ~ Normal(theta, 0.8)
    end
    bounded = @brm begin
        location ~ Normal(7.0, 1.0)
        theta ~ VonMises(location, 1.7; lower=location - 1, upper=location + 1)
        y ~ Normal(theta, 0.8)
    end
    data = (; y=[6.8, 7.4])
    for builder in (ordinary, bounded)
        descriptor = brm_descriptor(builder, data; mod=@__MODULE__, highlights=())
        checked = StanBlocks.stanc_check(BRM.stan_code(descriptor.plan))
        checked.ok || @error "VonMises prior stanc" output=checked.output
        @test checked.ok
        cache = joinpath(tempdir(), "brm-von-mises-prior")
        mkpath(cache)
        problem = brm_execute(descriptor, :instantiate;
            path=joinpath(cache, string(descriptor.id) * ".stan"))
        backend = TuringBRMI(builder(data))
        @test LogDensityProblems.dimension(problem) == 2
        for raw in ([7.0, 0.0], [6.8, -0.6])
            names = Symbol.(BridgeStan.param_names(problem.model))
            values = BridgeStan.param_constrain(problem.model, raw)
            parameters = NamedTuple{Tuple(names)}(Tuple(values))
            @test parameters.location - pi <= parameters.theta <= parameters.location + pi
            @test BridgeStan.log_density(problem.model, raw; propto=false, jacobian=false) ≈
                Turing.logjoint(backend.model, parameters) atol=1e-10
        end
    end
end
