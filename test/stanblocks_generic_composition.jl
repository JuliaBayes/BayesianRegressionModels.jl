using Test, BayesianRegressionModels, Distributions, LogDensityProblems
import StanBlocks
const BRM = BayesianRegressionModels

@testset "CDF composition follows translated distributions" begin
    data = (; y_gamma=[0.5, 1.0, 2.0], y_student=[-0.5, 0.2, 1.0],
             y_affine=[0.2, 0.6, 1.3])
    builder = @brm begin
        y_gamma ~ truncated(Gamma(2.0, 0.7); lower=0.3, upper=2.5)
        y_student ~ censored(TDist(4.0); lower=-0.5, upper=1.0)
        y_affine ~ truncated(LocationScale(0.2, 1.3, Laplace(0.3, 0.6));
                             lower=0.0, upper=1.5)
    end
    descriptor = brm_descriptor(builder, data; mod=@__MODULE__, highlights=())
    code = BRM.stan_code(descriptor.plan)
    checked = StanBlocks.stanc_check(code; warn_pedantic=false)
    checked.ok || @error "composition stanc" output=checked.output
    @test checked.ok
    cache = joinpath(tempdir(), "brm-generic-composition")
    mkpath(cache)
    problem = brm_execute(descriptor, :instantiate;
        path=joinpath(cache, string(descriptor.id) * ".stan"))
    @test LogDensityProblems.dimension(problem) == 0
    pointwise = brm_execute(descriptor, :pointwise_loglik;
        problem, draws=Float64[], seed=20260913)
    distributions = (;
        y_gamma=truncated(Gamma(2.0, 0.7); lower=0.3, upper=2.5),
        y_student=censored(TDist(4.0); lower=-0.5, upper=1.0),
        y_affine=truncated(LocationScale(0.2, 1.3, Laplace(0.3, 0.6));
                           lower=0.0, upper=1.5))
    for (name, distribution) in pairs(distributions)
        @test getproperty(pointwise, Symbol(name, :_likelihood)) ≈
            logpdf.(Ref(distribution), getproperty(data, name)) atol=1e-10
    end
    draws = brm_execute(descriptor, :predict;
        problem, draws=zeros(0, 40), seed=20260913)
    @test all(x -> 0.3 <= x <= 2.5, draws.y_gamma_gen)
    @test all(x -> -0.5 <= x <= 1.0, draws.y_student_gen)
    @test all(x -> 0.0 <= x <= 1.5, draws.y_affine_gen)
end
