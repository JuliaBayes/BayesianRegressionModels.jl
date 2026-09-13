using Test, BayesianRegressionModels, Distributions, LogDensityProblems
import StanBlocks

@testset "structural scalar support intersects the retained prior bounds" begin
    data = (; x=[-1.0, -0.3, 0.4, 1.1], y=[0.1, -0.2, 0.5, 0.8])
    gp_builder = @brm begin
        rho ~ Uniform(0.1, 0.2)
        mu ~ gp(x)
        length_scale(mu, gp(x)) ~ Uniform(rho, rho + 1)
        sd(mu, gp(x)) ~ Normal(0, 1; lower=0.2, upper=0.8)
        y ~ Normal(mu, 1)
    end
    r2_builder = @brm begin
        hyper ~ Uniform(0.1, 0.2)
        mu ~ 1 + x
        effect(mu, :) ~ r2d2(R2=Normal(0.5, 0.2; lower=hyper, upper=0.9), tau_bsv=0.5)
        y ~ Normal(mu, 1)
    end
    cache = joinpath(tempdir(), "brm-term-prior-bounds")
    mkpath(cache)
    for (builder, geometry) in ((gp_builder, :gp), (r2_builder, :r2))
        descriptor = brm_descriptor(builder, data; mod=@__MODULE__, highlights=())
        code = BayesianRegressionModels.stan_code(descriptor.plan)
        checked = StanBlocks.stanc_check(code)
        checked.ok || @error "term-prior stanc" output=checked.output
        @test checked.ok
        problem = brm_execute(descriptor, :instantiate;
            path=joinpath(cache, string(descriptor.id) * ".stan"))
        raw = zeros(LogDensityProblems.dimension(problem))
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, raw)
        @test isfinite(lp)
        @test all(isfinite, gradient)
        names = StanBlocks.BridgeStan.param_names(problem.model)
        values = StanBlocks.BridgeStan.param_constrain(problem.model, raw)
        physical = Dict(zip(names, values))
        if geometry === :gp
            rho = only(value for (name, value) in physical if endswith(name, "_rho"))
            sigma = only(value for (name, value) in physical if endswith(name, "_sigma"))
            @test physical["rho"] < rho < physical["rho"] + 1
            @test 0.2 < sigma < 0.8
        else
            @test physical["hyper"] < physical["r2d2_mu_R2"] < 0.9
        end
    end
end
