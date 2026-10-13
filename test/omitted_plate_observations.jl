# Run: julia --startup-file=no --project=test test/omitted_plate_observations.jl
using Test
using BayesianRegressionModels
using StanBlocks
using BridgeStan, LogDensityProblems
using Distributions: Exponential, Normal

const plate_data = (;
    subject=[1, 2], obs_subject=[1, 2, 1],
    time=[0.2, 0.3, 0.8], lloq=[0.1, 0.1, 0.1],
    qt_y=[0.5, 0.6],
)

const plate_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    eta ~ 1 + (1 | p | subject)
    @plate for i in eachindex(subject)
        pk_loc[i] = exp(eta[i]) .+ ragged(time, obs_subject)[i]
        pk_conc[i] ~ censored(Normal(pk_loc[i],
            addprop(pk_loc[i], sigma_add, sigma_prop));
            lower=ragged(lloq, obs_subject)[i])
        qt_y[i] ~ Normal(eta[i], 1.0)
    end
end

@testset "omitted censored plate observation keeps its predictive carrier" begin
    omitted = SBBRMI(plate_builder(plate_data); mod=@__MODULE__)
    pk = only(d for d in generative_plan(omitted).declarations
        if d.target === :pk_conc)
    @test pk.role === :observation
    @test isnothing(pk.data_source)
    @test !isempty(pk.context)
    @test Symbol(join((pk.context..., pk.target), "_")) in omitted.model.observations
    @test :qt_y in omitted.model.observations
    @test :sigma_add ∉ omitted.model.observations
    @test :sigma_prop ∉ omitted.model.observations

    code = BayesianRegressionModels.stan_code(omitted)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    descriptor = brm_descriptor(omitted)
    output = brm_output(descriptor, :pk_conc; role=:posterior_predictive)
    @test endswith(string(output.name), "_gen")
    @test output.segments == [2, 3]
    @test :predict in getproperty.(descriptor.operations, :name)
    @test all(o.logical !== :pk_conc for o in
        brm_outputs(descriptor; role=:pointwise_loglik))
    problem = StanBlocks.stan_instantiate(omitted.model)
    dimension = LogDensityProblems.dimension(problem)
    @test dimension > 0
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    coordinates = brm_output_coordinates(output, names)
    @test length(coordinates) == 3
    draw = BridgeStan.param_constrain(problem.model, zeros(dimension);
        include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(problem.model, 1))[coordinates]
    @test all(isfinite, draw)
    @test all(draw .>= 0.1)

    bound = SBBRMI(plate_builder(merge(plate_data,
        (; pk_conc=[[0.4, 1.2], [0.6]]))); mod=@__MODULE__)
    @test :pk_conc in bound.model.observations
    @test StanBlocks.stanc_check(BayesianRegressionModels.stan_code(bound);
        warn_pedantic=false).ok
    bound_output = brm_output(brm_descriptor(bound), :pk_conc;
        role=:posterior_predictive)
    @test bound_output.segments == [2, 3]
end
