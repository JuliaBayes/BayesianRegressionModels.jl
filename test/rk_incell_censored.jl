# Run: julia --startup-file=no --project=test test/rk_incell_censored.jl
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# Interleaved rows and row-specific limits exercise the original subject plate,
# its ragged inputs, and the bounded observation on the same BRMI.
const CENSORED_CELL_DATA = (;
    subject=[:b, :a, :c],
    row_subject=[:a, :b, :a, :c, :b],
    time=[0.2, 0.4, 0.8, 0.5, 1.1],
    y=[0.1, 0.6, 1.0, 0.3, 1.2],
    limit=[0.1, 0.2, 0.15, 0.3, 0.25],
)

const CENSORED_CELL = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    location ~ 1 + (1 | panel | subject)
    @plate for i in eachindex(location)
        mean[i] = location[i] .+ ragged(time, row_subject)[i]
        ragged(y, row_subject)[i] ~ censored(
            Normal(mean[i], addprop(mean[i], sigma_add, sigma_prop));
            lower=ragged(limit, row_subject)[i])
    end
end

@stestset "censored Normal inside a ragged cell keeps RK and Stan densities" begin
    brmi = CENSORED_CELL(CENSORED_CELL_DATA)
    backend, problem = consumer_problem(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    stan = BRM.stan_instantiate(sb; path=joinpath(mktempdir(), "incell_censored.stan"))
    transport = brm_coordinate_transport(backend, sb,
        BridgeStan.param_unc_names(stan.model))
    translated = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt).
        _rk_translated_plan(backend.plan)
    pointwise = prepare_query(backend.model, translated, :pointwise)
    data = CENSORED_CELL_DATA
    for u in (zeros(length(transport.rk_names)),
              [0.15 * sin(i) for i in eachindex(transport.rk_names)])
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        stan_u = brm_rk_point_to_stan(transport, u)
        stan_gradient = similar(stan_u)
        stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
            stan_gradient; propto=false, jacobian=true)
        @test value ≈ stan_value + transport.logdensity_offset atol=2e-10 rtol=2e-10
        @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-8 rtol=2e-8
    end

    u = zeros(length(transport.rk_names))
    actual = Base.invokelatest(pointwise, u)
    @test length.(actual.y) == [2, 2, 1]
    expected = map(data.subject) do subject
        map(findall(==(subject), data.row_subject)) do j
            mean = data.time[j]
            scale = sqrt(1 + mean^2)
            law = Normal(mean, scale)
            data.y[j] == data.limit[j] ? logcdf(law, data.limit[j]) :
                logpdf(law, data.y[j])
        end
    end
    @test all(isapprox(actual.y[i], expected[i]; atol=2e-11, rtol=2e-11)
        for i in eachindex(expected))

    model = brm_nlme_model(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    eta = zeros(length(first(model.view.subject_coordinates)), length(data.subject))
    expected_by_level = [sum(expected[only(findall(==(level), data.subject))])
        for level in model.view.levels]
    @test brm_nlme_loglikelihoods(model, u[model.theta], u[model.sigma], eta) ≈
        expected_by_level atol=2e-11 rtol=2e-11
end

const WEIGHTED_CENSORED_CELL_DATA = merge(CENSORED_CELL_DATA, (;
    w=[1.0, 0.5, 0.0, 2.0, 0.75]))

const WEIGHTED_CENSORED_CELL = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    location ~ 1 + (1 | panel | subject)
    @plate for i in eachindex(location)
        mean[i] = location[i] .+ ragged(time, row_subject)[i]
        ragged(y, row_subject)[i] ~ weighted(censored(
            Normal(mean[i], addprop(mean[i], sigma_add, sigma_prop));
            lower=ragged(limit, row_subject)[i]), ragged(w, row_subject)[i])
    end
end

# The unweighted cell above is verified against Stan; weighting must scale
# each of its row log densities, everywhere, by that row's weight.
@stestset "weighted censored Normal inside a ragged cell scales every row density" begin
    data = WEIGHTED_CENSORED_CELL_DATA
    backend, problem = consumer_problem(WEIGHTED_CENSORED_CELL(data))
    base, base_problem = consumer_problem(CENSORED_CELL(CENSORED_CELL_DATA))
    @test coordinate_names(backend.model.layout) == coordinate_names(base.model.layout)
    extension = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    query(b) = prepare_query(b.model, extension._rk_translated_plan(b.plan), :pointwise)
    pointwise, base_pointwise = query(backend), query(base)
    weights = [data.w[findall(==(subject), data.row_subject)] for subject in data.subject]
    oracle(u) = LogDensityProblems.logdensity(base_problem, u) +
        sum(sum((w .- 1) .* row) for (w, row) in
            zip(weights, Base.invokelatest(base_pointwise, u).y))
    for u in (zeros(LogDensityProblems.dimension(problem)),
              [0.15 * sin(i) for i in 1:LogDensityProblems.dimension(problem)])
        check_consumer_point(problem, u, oracle)
        actual = Base.invokelatest(pointwise, u).y
        expected = map((w, row) -> w .* row, weights, Base.invokelatest(base_pointwise, u).y)
        @test all(isapprox(actual[i], expected[i]; atol=2e-11, rtol=2e-11)
            for i in eachindex(expected))
    end

    model = brm_nlme_model(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    u = zeros(LogDensityProblems.dimension(problem))
    eta = zeros(length(first(model.view.subject_coordinates)), length(data.subject))
    rows = Base.invokelatest(pointwise, u).y
    expected_by_level = [sum(rows[only(findall(==(level), data.subject))])
        for level in model.view.levels]
    @test brm_nlme_loglikelihoods(model, u[model.theta], u[model.sigma], eta) ≈
        expected_by_level atol=2e-11 rtol=2e-11
    # Capability gap, not a wrong shape: StanBlocks' `weighted` does not yet
    # take a censoring combinator as its family (StanBlocks snag
    # `weighted-over-ce-a55869f3`), so SBBRMI cannot transpile this cell.
    @test_broken (BRM.stan_code(SBBRMI(WEIGHTED_CENSORED_CELL(data);
        mod=@__MODULE__, total_groups=())); true)
end
