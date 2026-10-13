# Run: julia --startup-file=no --project=test test/ragged_plate_descriptor_source.jl
using Test
using BayesianRegressionModels
using Distributions: Exponential, Normal

const PANEL = (;
    subject=[:b, :a, :c], x=[0.2, -0.1, 0.3],
    event_subject=[:a, :b, :a, :c, :b],
    event_time=[0.2, 0.3, 0.8, 0.4, 0.9],
    limit=fill(0.05, 5), y=[0.3, 0.4, 0.8, 0.5, 1.0],
)

const INCELL = @brm begin
    sigma ~ Exponential(1)
    log_rate ~ 1 + x + (1 | panel | subject)
    @plate for i in eachindex(log_rate)
        location[i] = exp(log_rate[i]) .* ragged(event_time, event_subject)[i]
        ragged(y, event_subject)[i] ~ censored(
            Normal(location[i], sigma); lower=ragged(limit, event_subject)[i])
    end
end

const TOPLEVEL = @brm begin
    sigma ~ Exponential(1)
    log_rate ~ 1 + x + (1 | panel | subject)
    @plate for i in eachindex(log_rate)
        location[i] = exp(log_rate[i]) .* ragged(event_time, event_subject)[i]
    end
    ragged(y, event_subject) ~ censored(Normal(location, sigma); lower=limit)
end

without_y(data) = NamedTuple(k => v for (k, v) in pairs(data) if k !== :y)

@testset "ragged plate observation source resolves to flat response" begin
    inside = brm_descriptor(SBBRMI(INCELL(PANEL); mod=@__MODULE__))
    outside = brm_descriptor(SBBRMI(TOPLEVEL(PANEL); mod=@__MODULE__))
    omitted = @test_logs (:warn, r"`y` bind\(s\) no data column") brm_descriptor(
        SBBRMI(TOPLEVEL(without_y(PANEL)); mod=@__MODULE__))

    for role in (:posterior_predictive, :pointwise_loglik)
        nested = brm_output(inside, :y; role)
        flat = brm_output(outside, :y; role)
        @test nested.logical === :y && flat.logical === :y
        @test nested.segments == [2, 4, 5]
        @test flat.segments == nested.segments
        n = role === :posterior_predictive ? 5 : 3
        @test brm_output_coordinates(nested, ["$(nested.name).$i" for i in 1:n]) ==
              collect(1:n)
    end

    draw = brm_output(omitted, :y; role=:posterior_predictive)
    @test draw.logical === :y
    @test draw.segments == [2, 4, 5]
    @test isempty(brm_outputs(omitted; role=:pointwise_loglik))
end
