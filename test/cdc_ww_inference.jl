# test/cdc_ww_inference.jl — gate for the CDC ww-inference structural port in
# research/wastewater/cdc_ww_inference.jl (decision `2026-08-26T16-44-33-638-1yz7ybe`,
# option B). Gates semantic fixtures derived from upstream SHA
# 3e92fea29a0b96212e1690dbdcaeb8abfe19cc4f plus the model artifact:
# transpile + stanc + finite BridgeStan density/gradient. Not a fit or a
# posterior-equivalence test.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
import StanBlocks.stan: transpiles, compiles

include(joinpath(@__DIR__, "..", "research", "wastewater", "cdc_ww_inference.jl"))

const CDC_RUN_BRIDGESTAN = get(ENV, "BRM_KERNEL_RUNTIME", "1") != "0"
const CDC_CACHE = joinpath(tempdir(), "brm-cdc-ww-inference")

function cdc_bridgestan_finite(model)
    isdir(CDC_CACHE) || mkpath(CDC_CACHE)
    code = StanBlocks.stan_code(model)
    path = joinpath(CDC_CACHE, string(hash(code)) * ".stan")
    prob = StanBlocks.stan_instantiate(model; path)
    dim = LogDensityProblems.dimension(prob)
    lp, grad = LogDensityProblems.logdensity_and_gradient(prob, zeros(dim))
    isfinite(lp) && all(isfinite, grad) && length(grad) == dim
end

cdc_stanc_ok(model) =
    StanBlocks.stanc_check(StanBlocks.stan_code(model); warn_pedantic = false).ok

# Direct Julia translations of the small upstream recurrences used as numerical
# fixtures. These deliberately do not call the BRM helpers: agreement therefore
# checks the mathematical contract rather than a helper calling itself.
function upstream_diff_ar1(x0, ar, sigma, z)
    differences = similar(z)
    differences[1] = sigma * z[1]
    for i in 2:length(z)
        differences[i] = ar * differences[i - 1] + sigma * z[i]
    end
    vcat(x0, x0 .+ cumsum(differences))
end

function upstream_stationary_ar1(ar, sigma, z)
    out = similar(z)
    out[1] = sigma * z[1] / sqrt(1 - ar^2)
    for i in 2:length(z)
        out[i] = ar * out[i - 1] + sigma * z[i]
    end
    out
end

function upstream_shedding(t_peak, viral_peak, duration, n)
    growth = viral_peak / t_peak
    wane = viral_peak / (duration - t_peak)
    loads = [10.0^(t <= t_peak ? growth * t :
                   max(0.0, viral_peak + wane * t_peak - wane * t)) for t in 1:n]
    loads ./ sum(loads)
end

function reference_count_interval_mean(I_mat, subpop_weights, delay, logit_rate,
                                       starts, stops, streams, dow, dow_effect,
                                       population)
    out = zeros(length(starts))
    for record in eachindex(starts)
        stream = streams[record]
        for outcome_time in starts[record]:stops[record]
            delayed = 0.0
            for lag in axes(delay, 1)
                infection_time = outcome_time - lag + 1
                infection_time >= 1 || continue
                incidence = sum(I_mat[infection_time, :] .* subpop_weights[:, stream])
                rate = inv(1 + exp(-logit_rate[infection_time, stream]))
                delayed += delay[lag, stream] * rate * incidence
            end
            out[record] += population[stream] * dow_effect[dow[outcome_time]] * delayed
        end
    end
    out .+ 1e-8
end

@testset "CDC upstream numerical recurrences" begin
    @test upstream_diff_ar1(0.2, 0.5, 0.1, [1.0, -2.0, 0.5]) ≈
          [0.2, 0.3, 0.15, 0.125]
    @test upstream_stationary_ar1(0.6, 0.2, [0.4, -0.5, 1.0]) ≈
          [0.1, -0.04, 0.176]

    shedding = upstream_shedding(2.0, 4.0, 6.0, 6)
    @test shedding ≈ [100.0, 10000.0, 1000.0, 100.0, 10.0, 1.0] ./ 11211.0
    @test sum(shedding) ≈ 1.0

    i_first_obs, growth, uot = 0.002, 0.03, 50
    seed_start = exp(log(i_first_obs) - uot * growth)
    @test seed_start ≈ i_first_obs * exp(-1.5)
    @test seed_start * exp(growth * (uot - 1)) ≈ i_first_obs * exp(-growth)
end

@testset "CDC multi-frame fixture invariants" begin
    df = cdc_ww_brm_fixture()
    @test df.subpopulation[1] == "reference-uncovered"
    @test 1 ∉ df.ww_subpop
    @test length(unique(zip(df.ww_subpop, df.ww_time))) < length(df.ww_time)
    @test length(unique(df.lab_to_subpop)) < df.n_labs
    @test length(unique(df.ww_lod)) > 1
    @test any(df.ww_log_conc .== df.ww_lod)

    @test size(df.count_subpop_weights) == (df.K, df.n_count_streams)
    @test all(isapprox.(vec(sum(df.count_subpop_weights; dims=1)), 1.0))
    @test all(isapprox.(vec(sum(df.count_delay; dims=1)), 1.0))
    @test all(df.count_start .<= df.count_stop)
    @test all(df.count_start[df.count_obs_stream .== 1] .==
              df.count_stop[df.count_obs_stream .== 1])
    @test any(df.count_start[df.count_obs_stream .== 2] .<
              df.count_stop[df.count_obs_stream .== 2])
    @test all(df.forecast_count_start .> df.uot + df.nt)
    @test all(df.forecast_count_stop .<= df.n_total)
    @test all(df.forecast_ww_time .> df.uot + df.nt)
    @test all(df.forecast_ww_time .<= df.n_total)

    # One interval and the matching collection of daily rows have the same mean.
    incidence = reshape(collect(1.0:20.0), 10, 2) ./ 10_000
    weights = [0.75 0.0; 0.25 1.0]
    delay = [0.7 0.4; 0.3 0.6]
    rates = fill(-4.0, 10, 2)
    dow = [mod(t - 1, 7) + 1 for t in 1:10]
    effect = ones(7)
    population = [100_000.0, 40_000.0]
    weekly = reference_count_interval_mean(
        incidence, weights, delay, rates, [4], [10], [2], dow, effect, population)
    daily = reference_count_interval_mean(
        incidence, weights, delay, rates, collect(4:10), collect(4:10),
        fill(2, 7), dow, effect, population)
    @test only(weekly) ≈ sum(daily .- 1e-8) + 1e-8
end

@testset "CDC ww-inference — structural @slic companion" begin
    m = cdc_ww_inference_model()
    @test transpiles(m)
    @test compiles(m)
    @test cdc_stanc_ok(m)
    code = StanBlocks.stan_code(m)
    @test occursin("dar_logru", code)
    @test occursin("vector[n_innov] eps_r", code)
    CDC_RUN_BRIDGESTAN && @test cdc_bridgestan_finite(m)
end

@testset "CDC ww-inference — re-bind new data (different K / horizon)" begin
    m = cdc_ww_inference_model(cdc_ww_inference_fixture(; K = 2, nt = 56, n_weeks = 8))
    @test transpiles(m)
    @test cdc_stanc_ok(m)
end

# The four-component model expressed on the @brm formula surface: latent
# subpopulation renewal, sparse lab/site wastewater, generic interval count streams,
# and named forecast generated quantities. The global weekly log-R process uses
# BRM's direct differenced-AR term rather than an ordinary-AR substitution.
@testset "CDC ww-inference — @brm structural port" begin
    df = cdc_ww_brm_fixture()
    sb = SBBRMI(cdc_ww_brm_model(df))
    m = sb.model
    @test transpiles(m)
    @test compiles(m)
    @test cdc_stanc_ok(m)
    code = StanBlocks.stan_code(m)
    @test occursin("differenced_ar1_path", code)
    @test occursin("dar_log_ru_week_week_grid_beta", code)
    @test occursin("renewal_from_first_observed", code)
    @test occursin("viral_shedding_trajectory", code)
    @test occursin("count_interval_mean", code)
    @test occursin("forecast_infections", code)
    @test occursin("forecast_count_mean", code)
    @test occursin("forecast_ww_log_mean", code)
    @test occursin(r"simplex\[[^]]+\] dow_share", code)
    @test occursin("real<lower=0, upper=1> phi_delta", code)

    plan = cdc_ww_brm_plan(df)
    @test Set(d.data_source for d in plan.declarations if d.role === :observation) ==
          Set([:ww_log_conc, :count])
    descriptor = cdc_ww_brm_descriptor(df)
    output_names = Set(o.name for o in descriptor.outputs)
    @test Set([:forecast_infections, :forecast_count_mean, :forecast_ww_log_mean]) ⊆
          output_names
    operation_names = Set(op.name for op in descriptor.operations)
    @test Set([:fit, :predict, :pointwise_loglik, :replay, :reprocess]) ⊆
          operation_names
    CDC_RUN_BRIDGESTAN && @test cdc_bridgestan_finite(m)
end

@testset "CDC ww-inference — @brm form re-binds new data (different K / horizon)" begin
    m = SBBRMI(cdc_ww_brm_model(cdc_ww_brm_fixture(; K = 2, nt = 56))).model
    @test transpiles(m)
    @test cdc_stanc_ok(m)
end
