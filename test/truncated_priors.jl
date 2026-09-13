using Test, BayesianRegressionModels, Distributions, Turing, LogDensityProblems
import StanBlocks
const BRM = BayesianRegressionModels
const BridgeStan = StanBlocks.BridgeStan

@testset "normalized prior composition retains family arguments and sampled bounds" begin
    builder = @brm begin
        location ~ Normal(0, 1)
        bounded ~ truncated(Cauchy(0.2, 0.7); lower=location, upper=location + 2)
        positive ~ truncated(Gamma(2.0, 0.6); lower=0.2)
        affine ~ truncated(LocationScale(0.3, 1.2, Laplace(0.1, 0.8)); upper=1.5)
        y ~ Normal(bounded + positive + affine, 0.8)
    end
    data = (; y=[0.1, 0.5])
    descriptor = brm_descriptor(builder, data; mod=@__MODULE__, highlights=())
    checked = StanBlocks.stanc_check(BRM.stan_code(descriptor.plan))
    checked.ok || @error "truncated-prior stanc" output=checked.output
    @test checked.ok
    cache = joinpath(tempdir(), "brm-truncated-priors")
    mkpath(cache)
    problem = brm_execute(descriptor, :instantiate;
        path=joinpath(cache, string(descriptor.id) * ".stan"))
    backend = TuringBRMI(builder(data))
    @test LogDensityProblems.dimension(problem) == 4
    for raw in (zeros(4), [-0.3, 0.1, 0.4, -0.5])
        names = Symbol.(BridgeStan.param_names(problem.model))
        values = BridgeStan.param_constrain(problem.model, raw)
        parameters = NamedTuple{Tuple(names)}(Tuple(values))
        @test parameters.location <= parameters.bounded <= parameters.location + 2
        @test parameters.positive >= 0.2
        @test parameters.affine <= 1.5
        @test BridgeStan.log_density(problem.model, raw; propto=false, jacobian=false) ≈
            Turing.logjoint(backend.model, parameters) atol=1e-10
    end
end
