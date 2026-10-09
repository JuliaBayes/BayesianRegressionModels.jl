include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module PublicDeclaredConsumerAxis
using BayesianRegressionModels, Distributions
function build(data)
    @brm data begin
        theta ~ 1 + (1 | p | subject)
        effect(theta, Intercept) ~ Normal(0, 0.7)
        sd(:, p) ~ Exponential(0.9)
        log(qt_scale) ~ 1
        effect(qt_scale, Intercept) ~ Normal(0, 0.4)
        pred ~ kernel(t, theta, ragged(qt_scale, ecg_subject)) do ts, a, scales
            a + sum(scales)
        end
        y ~ Normal(pred, 0.8)
    end
end
end

@stestset "declared gather supplies the predictor axis independently of response rows" begin
    data = (; subject=["b", "a"], t=[[0.1, 0.3], [0.2, 0.4]],
        ecg_subject=["a", "b", "a", "b", "a"], y=[0.1, -0.2])
    saved = deepcopy(data)
    brmi = PublicDeclaredConsumerAxis.build(data)
    context = BRM._brm_backend_context(brmi)
    geometry = BRM._brm_prepare_predictor_geometry(brmi, context, :qt_scale;
        available_predictors=(:theta, :qt_scale))
    @test context.target_axes[:qt_scale] == [:ecg_subject]
    @test geometry.design.row_source === :ecg_subject
    @test size(geometry.design.matrix) == (5, 1)
    @test geometry.design.matrix == ones(5, 1)
    # Omitting endpoint values does not remove declared consumer geometry.
    prior_brmi = PublicDeclaredConsumerAxis.build(Base.structdiff(data, (; y=data.y)))
    prior_context = BRM._brm_backend_context(prior_brmi)
    prior_geometry = BRM._brm_prepare_predictor_geometry(prior_brmi, prior_context, :qt_scale;
        available_predictors=(:theta, :qt_scale))
    @test prior_geometry.design.row_source === :ecg_subject
    @test size(prior_geometry.design.matrix) == (5, 1)
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    ia = findfirst(==(Symbol("theta_Intercept")), names)
    iq = findfirst(==(Symbol("qt_scale_Intercept")), names)
    it = findfirst(n -> occursin(".tau.", string(n)), names)
    iz = findall(n -> occursin(".z.", string(n)), names)
    @test length(names) == 5
    @test length(iz) == 2
    oracle(u) = begin
        tau = exp(u[it])
        theta = u[ia] .+ tau .* u[iz][[2, 1]]
        means = theta .+ [2, 3] .* exp(u[iq])
        logpdf(Normal(0, 0.7), u[ia]) + logpdf(Normal(0, 0.4), u[iq]) +
            sum(logpdf.(Normal(), u[iz])) + logpdf(Exponential(0.9), tau) + u[it] +
            sum(logpdf.(Normal.(means, 0.8), data.y))
    end
    stan = consumer_stan(brmi, "declared-consumer-axis"; mod=PublicDeclaredConsumerAxis)
    mapping = [names[ia] => "pop_theta_beta_pop.1", names[iq] => "pop_log_qt_scale_beta_pop.1",
        names[it] => "b_p_subject_tau.1",
        names[iz[1]] => "b_p_subject_z_flat.1", names[iz[2]] => "b_p_subject_z_flat.2"]
    for u in (zeros(5), collect(range(-0.2, 0.3; length=5)), fill(-0.1, 5))
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, mapping, backend, u)
    end
    @test isequal(data, saved)
end
