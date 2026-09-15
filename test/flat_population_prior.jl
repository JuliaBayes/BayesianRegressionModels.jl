using Test, BayesianRegressionModels, StanBlocks, BridgeStan, Distributions
const BRM=BayesianRegressionModels
data=(;x=[-1.,0.,2.],y=[.2,-.4,1.3])
mixed=@brm begin
    mu ~ 1 + x
    effect(mu,Intercept) ~ Normal(0,2)
    effect(mu,x) ~ Flat()
    y ~ Normal(mu,1)
end
allflat=@brm begin
    mu ~ 1 + x
    effect(mu,Intercept) ~ Flat()
    effect(mu,x) ~ Flat()
    y ~ Normal(mu,1)
end
@testset "Retained flat population coefficients" begin
    for (builder,proper_intercept) in ((mixed,true),(allflat,false))
        sb=SBBRMI(builder(data);mod=@__MODULE__)
        code=BRM.stan_code(sb)
        @test !occursin("flat_lpdf",code)
        problem=Base.invokelatest(StanBlocks.stan_instantiate,sb.model;path=joinpath(mktempdir(),"flat.stan"))
        @test length(BridgeStan.param_unc_names(problem.model))==2
        for q in ([.3,-.2],[-1.2,.7])
            residual=data.y.-q[1].-q[2].*data.x
            expected=sum(logpdf.(Normal(),residual))+(proper_intercept ? logpdf(Normal(0,2),q[1]) : 0.)
            gradient=[sum(residual)-(proper_intercept ? q[1]/4 : 0.),sum(residual.*data.x)]
            lp,g=BridgeStan.log_density_gradient(problem.model,q;propto=false)
            @test lp≈expected
            @test g≈gradient
        end
    end
end
