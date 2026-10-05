include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Statistics: mean, std

module PublicMissingValuePredictors
using BayesianRegressionModels, Distributions
function build(data)
    @brm data begin
        mi(age) ~ LogNormal(1.4, 0.4)
        mi(weight) ~ LogNormal(2.0, 0.3)
        log(Vc) ~ 1 + standardize(age) + standardize(weight) + (1 | p | subject)
        log(k10) ~ 1 + standardize(age) + (1 | p | subject)
        effect(Vc, :) ~ Normal(0, 0.4)
        effect(k10, :) ~ Normal(0, 0.4)
        sd(:, p) ~ Exponential(0.9)
        y ~ Normal(Vc, 0.8)
        z ~ Normal(k10, 0.7)
    end
end
function build_kernel(data)
    @brm data begin
        mi(age) ~ LogNormal(1.4, 0.4)
        mi(weight) ~ LogNormal(2.0, 0.3)
        log(Vc) ~ 1 + standardize(age) + standardize(weight) + (1 | p | subject)
        log(k10) ~ 1 + standardize(age) + (1 | p | subject)
        effect(Vc, :) ~ Normal(0, 0.4)
        effect(k10, :) ~ Normal(0, 0.4)
        sd(:, p) ~ Exponential(0.9)
        locations ~ kernel(Vc, k10) do vc, k
            vc
        end
        y ~ Normal(locations, 0.8)
        z ~ Normal(k10, 0.7)
    end
end
function build_observed_only_mi(data)
    @brm data begin
        mi(age) ~ LogNormal(1.4, 0.4)
        mu ~ Normal(0, 1)
        y ~ Normal(mu, 0.8)
    end
end
end

@stestset "completed covariates retain fixed observed anchors and shared subject hierarchy" begin
    data = (; subject=[1, 2, 3],
        age=Union{Missing,Float64}[3.2, missing, 5.1],
        weight=Union{Missing,Float64}[7.4, 8.3, missing],
        y=[1.1, 1.3, 0.9], z=[1.0, 0.8, 1.2])
    saved = deepcopy(data)
    brmi = PublicMissingValuePredictors.build(data)
    backend, problem = consumer_problem(brmi)
    @test length(backend.plan.completions) == 2
    @test !haskey(backend.plan.columns, :age)
    @test !haskey(backend.plan.columns, :weight)
    @test length(backend.plan.regression.ranef_buckets) == 1
    @test length(only(backend.plan.regression.ranef_buckets).margins) == 2
    names = coordinate_names(backend.model.layout)
    N = length(names)
    u = collect(range(-0.2, 0.3; length=N))
    value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
    @test isfinite(value)
    @test all(isfinite, gradient)
    @test isequal(data, saved)
    stan = consumer_stan(brmi, "missing-value-predictors"; mod=PublicMissingValuePredictors)
    @test N == 16
    index(name) = only(findall(==(Symbol(name)), names))
    coefficients = index.(["Vc_Intercept", "Vc_standardize_age", "Vc_standardize_weight",
        "k10_Intercept", "k10_standardize_age"])
    scales = index.(["ranef_draws_p_subject.sd.1", "ranef_draws_p_subject.sd.2"])
    innovations = [index("ranef_draws_p_subject.z.$row.$margin")
        for row in 1:3, margin in 1:2]
    correlation = index("ranef_draws_p_subject.L.1")
    ia, iw = index("age.y_mis.1"), index("weight.y_mis.1")
    observed_age = collect(skipmissing(data.age))
    observed_weight = collect(skipmissing(data.weight))
    am, as = mean(observed_age), std(observed_age)
    wm, ws = mean(observed_weight), std(observed_weight)
    expected_values(u) = begin
        age = [data.age[1], exp(u[ia]), data.age[3]]
        weight = [data.weight[1], data.weight[2], exp(u[iw])]
        xa, xw = (age .- am) ./ as, (weight .- wm) ./ ws
        rho = tanh(u[correlation])
        factor = [1.0 0.0; rho sqrt(1-rho^2)]
        tau = exp.(u[scales])
        random = u[innovations] * transpose(tau .* factor)
        beta = u[coefficients]
        vc = exp.(beta[1] .+ beta[2] .* xa .+ beta[3] .* xw .+ random[:,1])
        k10 = exp.(beta[4] .+ beta[5] .* xa .+ random[:,2])
        (; age, weight, vc, k10, rho, tau)
    end
    oracle(u) = begin
        v = expected_values(u)
        sum(logpdf.(Normal(0,0.4), u[coefficients])) +
            sum(logpdf.(Normal(), u[innovations])) +
            sum(logpdf.(Exponential(0.9), v.tau) .+ u[scales]) +
            -log(2) + log1p(-v.rho^2) +
            logpdf(Normal(1.4,0.4), u[ia]) + logpdf(Normal(2.0,0.3), u[iw]) +
            sum(logpdf.(LogNormal(1.4,0.4), observed_age)) +
            sum(logpdf.(LogNormal(2.0,0.3), observed_weight)) +
            sum(logpdf.(Normal.(v.vc,0.8), data.y)) +
            sum(logpdf.(Normal.(v.k10,0.7), data.z))
    end
    mapping = [names[correlation] => "b_p_subject_L.1",
        names[ia] => "age_y_mis.1", names[iw] => "weight_y_mis.1"]
    append!(mapping, [names[scales[j]] => "b_p_subject_tau.$j" for j in 1:2])
    append!(mapping, [names[innovations[row,margin]] =>
        "b_p_subject_z_flat.$(margin+2*(row-1))" for row in 1:3 for margin in 1:2])
    append!(mapping, [names[coefficients[j]] => "pop_log_Vc_beta_pop.$j" for j in 1:3])
    append!(mapping, [names[coefficients[j+3]] => "pop_log_k10_beta_pop.$j" for j in 1:2])
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    bound = ext._rk_translated_plan(backend.plan)
    pointwise = prepare_query(backend.model, bound, :pointwise)
    for point in (zeros(N), collect(range(-0.2,0.3;length=N)), fill(-0.1,N))
        check_consumer_point(problem, point, oracle)
        check_consumer_stan(problem, stan, mapping, backend, point)
        parts = pointwise(point)
        @test parts.age_obs ≈ logpdf.(LogNormal(1.4,0.4), observed_age)
        @test parts.weight_obs ≈ logpdf.(LogNormal(2.0,0.3), observed_weight)
        expected = expected_values(point)
        @test parts.y ≈ logpdf.(Normal.(expected.vc,0.8), data.y)
        @test parts.z ≈ logpdf.(Normal.(expected.k10,0.7), data.z)
    end
    @test isequal(data, saved)
end

@stestset "kernel composition retains completed predictor values and missing priors" begin
    data = (; subject=[1, 2, 3],
        age=Union{Missing,Float64}[3.2, missing, 5.1],
        weight=Union{Missing,Float64}[7.4, 8.3, missing],
        y=[1.1, 1.3, 0.9], z=[1.0, 0.8, 1.2])
    saved = deepcopy(data)
    direct, direct_problem = consumer_problem(PublicMissingValuePredictors.build(data))
    composed, composed_problem = consumer_problem(PublicMissingValuePredictors.build_kernel(data))
    @test length(composed.plan.completions) == 2
    @test coordinate_names(composed.model.layout) == coordinate_names(direct.model.layout)
    N = LogDensityProblems.dimension(direct_problem)
    for u in (zeros(N), collect(range(-0.2,0.3;length=N)), fill(-0.1,N))
        before = copy(u)
        direct_value, direct_gradient = LogDensityProblems.logdensity_and_gradient(direct_problem,u)
        value, gradient = LogDensityProblems.logdensity_and_gradient(composed_problem,u)
        @test value ≈ direct_value atol=2e-11 rtol=2e-11
        @test gradient ≈ direct_gradient atol=2e-10 rtol=2e-10
        @test u == before
    end
    @test isequal(data, saved)
end

@stestset "observed-only mi likelihood does not create missing inference coordinates" begin
    data = (;age=Union{Missing,Float64}[3.2,missing,5.1], y=[1.1,1.3,0.9])
    saved = deepcopy(data)
    brmi = PublicMissingValuePredictors.build_observed_only_mi(data)
    backend,problem = consumer_problem(brmi)
    @test isempty(backend.plan.completions)
    @test coordinate_names(backend.model.layout) == [:mu]
    @test !haskey(backend.plan.columns,:age)
    observed = collect(skipmissing(data.age))
    oracle(u) = logpdf(Normal(),u[1]) +
        sum(logpdf.(LogNormal(1.4,0.4),observed)) +
        sum(logpdf.(Normal(u[1],0.8),data.y))
    stan = consumer_stan(brmi,"observed-only-mi";mod=PublicMissingValuePredictors)
    @test BridgeStan.param_unc_names(stan.model) == ["mu"]
    for u in ([0.0],[0.2],[-0.1])
        check_consumer_point(problem,u,oracle)
        check_consumer_stan(problem,stan,[:mu=>"mu"],backend,u)
    end
    @test isequal(data,saved)
end

@stestset "complete modeled covariates retain their row values without empty latent ports" begin
    data = (;subject=[1,2,3],age=[3.2,4.1,5.1],weight=[7.4,8.3,9.2],
        y=[1.1,1.3,0.9],z=[1.0,0.8,1.2])
    saved = deepcopy(data)
    direct,direct_problem = consumer_problem(PublicMissingValuePredictors.build(data))
    composed,composed_problem = consumer_problem(PublicMissingValuePredictors.build_kernel(data))
    @test length(direct.plan.completions) == 2
    @test all(completion->completion.nmissing==0,direct.plan.completions)
    @test all(completion->isempty(direct.plan.columns[completion.missing_rows]),direct.plan.completions)
    names = coordinate_names(direct.model.layout)
    @test length(names) == 14
    @test !any(name->occursin("y_mis",string(name)),names)
    @test coordinate_names(composed.model.layout) == names
    for u in (zeros(14),collect(range(-0.2,0.3;length=14)),fill(-0.1,14))
        v,g = LogDensityProblems.logdensity_and_gradient(direct_problem,u)
        vc,gc = LogDensityProblems.logdensity_and_gradient(composed_problem,u)
        @test isfinite(v) && all(isfinite,g)
        @test vc ≈ v atol=2e-11 rtol=2e-11
        @test gc ≈ g atol=2e-10 rtol=2e-10
    end
    @test isequal(data,saved)
end
