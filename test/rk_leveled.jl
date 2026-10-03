# Independent native checks for the two leveled families beside ordinals.
using Test, BayesianRegressionModels, ReactiveKernelsPPL, Distributions
using Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "rk_source_roundtrip.jl"))

function check_leveled(brmi, oracle)
    backend = check_rk_source_roundtrip(BRM.RKBRMI(brmi))
    N = backend.model.layout.total
    for u in (zeros(N), collect(range(-0.2, 0.3; length=N)))
        problem = BRM.rk_logdensity_problem(backend;
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), u0=u)
        saved = copy(u)
        lp, g = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test lp ≈ oracle(u) atol=1e-11
        h = 1e-5
        fd = map(eachindex(u)) do i
            plus, minus = copy(u), copy(u)
            plus[i] += h
            minus[i] -= h
            (oracle(plus) - oracle(minus)) / (2h)
        end
        @test g ≈ fd atol=1e-6 rtol=1e-5
        @test isequal(u, saved)
    end
end

@testset "ordinary categorical reference probabilities" begin
    data = (; x=[-1.0, 0.2, 0.5, -0.4, 1.1, 0.0], y=[2, 1, 3, 2, 1, 3])
    model = @brm data begin
        a ~ 1 + x
        b ~ 1 + x
        y ~ CategoricalLogit(a, b)
    end
    # Four named, unconstrained Normal coefficients in authored order.
    function oracle(u)
        a = u[1] .+ u[2] .* data.x
        b = u[3] .+ u[4] .* data.x
        likelihood = sum(eachindex(data.y)) do i
            p = exp.([0.0, a[i], b[i]])
            logpdf(Categorical(p / sum(p)), data.y[i])
        end
        likelihood + sum(logpdf.(Normal(), u))
    end
    check_leveled(model, oracle)
end

@testset "ordinary multinomial row observations" begin
    data = (; y=[3 1 0; 2 2 1; 0 0 5; 1 1 1; 4 0 0; 2 1 2],
        n=[4, 5, 5, 3, 4, 5])
    model = @brm data begin
        p ~ Dirichlet(3, 1.0)
        y ~ Multinomial(n, p)
    end
    function oracle(u)
        # Independent stick breaking in the existing RK coordinate frame,
        # including its positive log(K-i) offset and full Jacobian.
        v1 = 1 / (1 + exp(-(u[1] + log(2))))
        v2 = 1 / (1 + exp(-u[2]))
        p = [v1, (1-v1)*v2, (1-v1)*(1-v2)]
        jac = log(v1) + 2log1p(-v1) + log(v2) + log1p(-v2)
        sum(logpdf(Multinomial(data.n[i], p), vec(data.y[i, :]))
            for i in axes(data.y, 1)) + logpdf(Dirichlet(3, 1.0), p) + jac
    end
    check_leveled(model, oracle)
end
