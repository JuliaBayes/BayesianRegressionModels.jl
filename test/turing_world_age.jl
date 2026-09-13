using Test, BayesianRegressionModels, Turing, Distributions, LogDensityProblems
const DP = Turing.DynamicPPL

# Construction and the first execution happen inside the same compiled caller.
# A Core.eval-generated @model method fails here despite working at top level.
function construct_and_evaluate(builder, data, parameters)
    backend = TuringBRMI(builder(data))
    varinfo = DP.VarInfo(backend.model, DP.InitFromParams(parameters), DP.UnlinkAll())
    density = DP.LogDensityFunction(backend.model, DP.getlogjoint_internal, varinfo)
    position = collect(DP.get_sample_input_vector(density))
    (; density=LogDensityProblems.logdensity(density, position),
       joint=Turing.logjoint(backend.model, parameters),
       pointwise=turing_pointwise_loglikelihoods(backend, parameters))
end

@testset "compiled-body cache preserves changed group-prior literals" begin
    data = (; g=["a", "b", "a"], y=[0.1, -0.2, 0.4])
    first_builder = @brm begin
        mu ~ 1 + (1 | p | g)
        sd(mu, p) ~ Normal(0, 1)
        y ~ Normal(mu, 0.7)
    end
    second_builder = @brm begin
        mu ~ 1 + (1 | p | g)
        sd(mu, p) ~ Normal(2, 0.4)
        y ~ Normal(mu, 0.7)
    end
    parameters = (; beta_pop=[0.3], group_1_1=(; scale=0.8, z=[0.1, -0.1]))
    first_result = construct_and_evaluate(first_builder, data, parameters)
    second_result = construct_and_evaluate(second_builder, data, parameters)
    expected_difference = logpdf(Normal(2, 0.4), 0.8) - logpdf(Normal(0, 1), 0.8)
    @test second_result.joint - first_result.joint ≈ expected_difference
    @test second_result.density - first_result.density ≈ expected_difference
end

@testset "first native model execution works inside compiled callers" begin
    builder = @brm begin
        beta ~ Normal(0.2, 0.7)
        y ~ Normal(beta, 0.5)
    end
    data = (; y=[0.1, 0.5])
    expected = logpdf(Normal(0.2, 0.7), 0.3) + sum(logpdf.(Normal(0.3, 0.5), data.y))
    result = construct_and_evaluate(builder, data, (; beta=0.3))
    @test result.density ≈ expected
    @test result.joint ≈ expected
    @test result.pointwise.y ≈ logpdf.(Normal(0.3, 0.5), data.y)

    joint_builder = @brm begin
        beta ~ Normal(0.2, 0.7)
        y ~ Normal(beta, 0.5)
        z ~ Cauchy(beta + 0.1, 0.8)
    end
    joint_data = (; data..., z=[0.2, -0.1, 0.4])
    expected_joint = expected + sum(logpdf.(Cauchy(0.4, 0.8), joint_data.z))
    result_joint = construct_and_evaluate(joint_builder, joint_data, (; beta=0.3))
    @test result_joint.density ≈ expected_joint
    @test result_joint.joint ≈ expected_joint
    @test result_joint.pointwise.z ≈ logpdf.(Cauchy(0.4, 0.8), joint_data.z)
end
