# Run: julia --startup-file=no --project=test test/plate_ragged_omitted_response.jl
using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

const PANEL = (;
    subject=[:b, :a, :c],
    x=[0.2, -0.1, 0.3],
    event_subject=[:a, :b, :a, :c, :b],
    event_time=[0.2, 0.3, 0.8, 0.4, 0.9],
    limit=fill(0.05, 5),
    y=[0.3, 0.4, 0.8, 0.5, 1.0],
)

const RAGGED_CELL = @brm begin
    sigma ~ Exponential(1)
    log_rate ~ 1 + x + (1 | panel | subject)
    events = groups(event_subject => subject)
    @plate for i in eachindex(log_rate)
        location[i] = exp(log_rate[i]) .* event_time[events[i]]
        y[events[i]] ~ censored(
            Normal(location[i], sigma); lower=limit[events[i]])
    end
end

drop_column(data, column) = NamedTuple(k => v for (k, v) in pairs(data) if k !== column)

@testset "unbound ragged observation in an indexed plate" begin
    bound = SBBRMI(RAGGED_CELL(PANEL); mod=@__MODULE__)
    omitted = @test_logs (:warn, r"`y` bind\(s\) no data column") SBBRMI(
        RAGGED_CELL(drop_column(PANEL, :y)); mod=@__MODULE__)

    @test !haskey(omitted.data, :y)
    @test !haskey(omitted.preproc, :y)
    @test all(!occursin("_y_ragged", String(key)) for key in keys(omitted.data))
    @test omitted.model.observations == (:location_y,)
    @test StanBlocks.stanc_check(BayesianRegressionModels.stan_code(bound);
                                 warn_pedantic=false).ok

    bound_descriptor = brm_descriptor(bound)
    @test brm_output(bound_descriptor, :y; role=:posterior_predictive) !== nothing
    @test brm_output(bound_descriptor, :y; role=:pointwise_loglik) !== nothing

    replayed = reprocess(omitted, drop_column(merge(PANEL, (;
        event_subject=[:c, :a, :b, :a, :b],
        event_time=[0.1, 0.4, 0.6, 0.8, 1.1],
    )), :y))
    @test !haskey(replayed.data, :y)
    @test replayed.model.observations == (:location_y,)

    for (label, model) in (("omitted", omitted), ("replayed", replayed))
        @testset "$label compiler and public outputs" begin
            code = BayesianRegressionModels.stan_code(model)
            @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
            descriptor = brm_descriptor(model)
            output = brm_output(descriptor, :y; role=:posterior_predictive)
            @test output !== nothing
            if output !== nothing
                @test output.logical === :y
                @test output.segments == [2, 4, 5]
                @test output.name in brm_operation(descriptor, :predict).outputs
            end
            @test isempty(brm_outputs(descriptor; role=:pointwise_loglik))
        end
    end
end

@testset "a missing ragged input still fails" begin
    @test_throws "positional `ragged(event_time, …)`" SBBRMI(
        RAGGED_CELL(drop_column(PANEL, :event_time)); mod=@__MODULE__)
end
