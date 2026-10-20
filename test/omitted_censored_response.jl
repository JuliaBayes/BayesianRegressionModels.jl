# A censored observation whose response column is omitted.
#
#   julia --startup-file=no --project=test test/omitted_censored_response.jl

using Test
using BayesianRegressionModels, StanBlocks
using BridgeStan, LogDensityProblems
using Distributions: Exponential, Normal

const BRM = BayesianRegressionModels

omit(data, keys::Symbol...) = Base.structdiff(data, NamedTuple{keys})
stanc_ok(sb) = StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok

const flat = (; x=[1.0, 2.0, 0.5, 1.5], y=[0.4, 0.25, 0.3, 0.5],
    z=[0.0, 0.4, 0.2, 0.9], lloq=fill(0.25, 4))

const censored_builder = @brm begin
    mu ~ 1 + x
    sigma ~ Exponential(1)
    y ~ censored(Normal(mu, sigma); lower=lloq)
end

@testset "an omitted censored response forward-simulates its fitted predictive law" begin
    omitted = @test_logs (:warn, r"`y` bind\(s\) no data column") SBBRMI(
        censored_builder(omit(flat, :y, :z)); mod=@__MODULE__)
    bound = SBBRMI(censored_builder(omit(flat, :z)); mod=@__MODULE__)
    @test stanc_ok(omitted)
    # The fitted program draws its `y_gen` from that same law; only the size
    # it reads differs (`y_n` there, `lloq_n` here).
    law = r"lower_clamping_vector_normal_rng\((y|lloq)_n, lloq, mu, sigma\)"
    @test occursin(law, BRM.stan_code(omitted))
    @test occursin(law, BRM.stan_code(bound))
    @test !haskey(omitted.data, :y)
    @test omitted.model.observations == (:y,)
    y = only(d for d in generative_plan(omitted).declarations if d.target === :y)
    @test y.role === :observation
    @test isnothing(y.data_source)
    @test y.family === :censored
    descriptor = brm_descriptor(omitted)
    output = brm_output(descriptor, :y; role=:posterior_predictive)
    @test output.name === :y_gen
    @test [o.name for o in descriptor.outputs if o.logical === :y] == [:y_gen]

    problem = StanBlocks.stan_instantiate(omitted.model)
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    coordinates = brm_output_coordinates(output, names)
    @test length(coordinates) == 4
    rng = BridgeStan.StanRNG(problem.model, 3)
    u = zeros(LogDensityProblems.dimension(problem))
    draws = reduce(hcat, (BridgeStan.param_constrain(problem.model, u;
        include_tp=true, include_gq=true, rng)[coordinates] for _ in 1:200))
    # Censored at `lloq`: draws below it sit exactly on it.
    @test all(>=(0.25), draws)
    @test any(==(0.25), draws)
    @test any(>(0.25), draws)
end

@testset "a fitted response beside an omitted censored one" begin
    mixed = @brm begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        nu ~ 1 + x
        z ~ Normal(nu, 1.0)
        y ~ censored(Normal(mu, sigma); lower=lloq)
    end
    plain = @brm begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        nu ~ 1 + x
        z ~ Normal(nu, 1.0)
        y ~ Normal(mu, sigma)
    end
    sb = @test_logs (:warn, r"`y` bind\(s\) no data column") SBBRMI(
        mixed(omit(flat, :y)); mod=@__MODULE__)
    @test stanc_ok(sb)
    @test sb.model.observations == (:y, :z)
    # Only `z` conditions the fit, exactly as with an uncensored `y`.
    reference = SBBRMI(plain(omit(flat, :y)); mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model)
    expected = StanBlocks.stan_instantiate(reference.model)
    @test BridgeStan.param_unc_names(problem.model) ==
          BridgeStan.param_unc_names(expected.model)
    x = collect(range(-0.3, 0.4; length=LogDensityProblems.dimension(problem)))
    @test LogDensityProblems.logdensity(problem, x) ≈
          LogDensityProblems.logdensity(expected, x) rtol=1e-12
    @test brm_output(brm_descriptor(sb), :y; role=:posterior_predictive).name === :y_gen
end
