# Run: julia --startup-file=no --project=test test/ragged_omitted_response.jl
#
# An omitted response observed per subject in the cell, through a `groups`
# binding, with a data-backed censoring bound: the observation stays, the bound
# is regathered on replay, and prior draws respect it.
using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal
using BridgeStan, LogDensityProblems

omitted_ragged_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i]) .+ 0.0 .* obs_idx[obs[i]]
        pk_conc[obs[i]] ~ censored(
            Normal(loc[i], addprop(loc[i], sigma_add, sigma_prop)); lower=pk_lloq[obs[i]])
    end
end

# The data the cell reads for `name`, whatever its prepared key.
cell_key(sb, name) = only(k for k in keys(sb.data)
    if occursin(string(name), string(k)) && endswith(string(k), "_ragged"))

omitted_ragged_data = (;
    subject=["s2", "s1"],
    obs_subject=["s1", "s2", "s1", "s2", "s2", "s1", "s2"],
    obs_idx=collect(1.0:7.0),
    pk_lloq=[0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
)

@testset "omitted grouped response retains observation and layout" begin
    brmi = omitted_ragged_builder(omitted_ragged_data)
    sb = @test_logs (:warn, r"`pk_conc` bind\(s\) no data column") SBBRMI(
        brmi; mod=@__MODULE__)
    @test !haskey(sb.data, :pk_conc)
    @test !haskey(sb.preproc, :pk_conc)
    # The omitted response is observed by its cell, under the cell's name.
    observations = sb.model.observations
    @test observations == (:loc_pk_conc,)
    bound_key = cell_key(sb, :pk_lloq)
    @test sb.data[bound_key] == [[0.11, 0.13, 0.14, 0.16], [0.10, 0.12, 0.15]]

    # The very same likelihood remains, with the same model-derived location
    # and scale. Omission changes the data binding, never the observation body.
    observed = SBBRMI(omitted_ragged_builder(merge(omitted_ragged_data,
        (; pk_conc=fill(0.2, 7)))); mod=@__MODULE__)
    @test BayesianRegressionModels.stan_code(sb) isa String
    # Bound, the observation reads the response gathered per subject.
    @test observed.model.observations == (:kernel_loc_pk_conc_ragged,)
    @test sb.data[bound_key] == observed.data[bound_key]

    # Frozen data replay regathers retained bounds from the new observation
    # design without demanding or manufacturing response values.
    replay_data = merge(omitted_ragged_data, (;
        obs_subject=["s2", "s1", "s2", "s1", "s2"],
        obs_idx=collect(1.0:5.0), pk_lloq=[0.21, 0.22, 0.23, 0.24, 0.25]))
    replayed = reprocess(sb, replay_data)
    @test !haskey(replayed.data, :pk_conc)
    @test !haskey(replayed.preproc, :pk_conc)
    @test replayed.model.observations == observations
    @test replayed.data[bound_key] == [[0.21, 0.23, 0.25], [0.22, 0.24]]

    @test_throws "grouping column must name" SBBRMI(
        omitted_ragged_builder(merge(omitted_ragged_data,
            (; pk_lloq=fill(0.1, 6)))); mod=@__MODULE__)
    @test_throws "ALREADY a ragged per-subject column" SBBRMI(
        omitted_ragged_builder(merge(omitted_ragged_data,
            (; pk_lloq=[fill(0.1, 3), fill(0.1, 4)]))); mod=@__MODULE__)
    @test_throws "name no subject" SBBRMI(
        omitted_ragged_builder(merge(omitted_ragged_data,
            (; obs_subject=fill("unknown", 7)))); mod=@__MODULE__)
end

@testset "unbound response bounds still validate" begin
    BRM = BayesianRegressionModels
    data = Dict{Symbol,Any}()
    @test isnothing(BRM._sb_validate_bounds(:truncated, :y, 0.0, 1.0, data))
    @test_throws "lower bounds must not exceed" BRM._sb_validate_bounds(
        :truncated, :y, [0.0, 2.0], [1.0, 1.0], data)
    @test_throws "bounds must be numeric" BRM._sb_validate_bounds(
        :censored, :y, "bad", nothing, data)
    @test_throws "bounds must be integers" BRM._sb_validate_composed_support(
        :censored, :y, 0.1, nothing, :discrete, data)
end

@testset "omitted grouped censored response draws and replay" begin
    sb = SBBRMI(omitted_ragged_builder(omitted_ragged_data); mod=@__MODULE__)
    replay_data = merge(omitted_ragged_data, (;
        obs_subject=["s2", "s1", "s2", "s1", "s2"],
        obs_idx=collect(1.0:5.0), pk_lloq=[0.21, 0.22, 0.23, 0.24, 0.25]))
    replayed = reprocess(sb, replay_data)
    for (label, model, segments, bounds) in (
        ("original", sb, [4, 7], [0.11, 0.13, 0.14, 0.16, 0.10, 0.12, 0.15]),
        ("changed layout", replayed, [3, 5], [0.21, 0.23, 0.25, 0.22, 0.24]),
    )
        @testset "$label" begin
            code = BayesianRegressionModels.stan_code(model)
            @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
            descriptor = brm_descriptor(model)
            output = brm_output(descriptor, :pk_conc; role=:posterior_predictive)
            @test output.logical === :pk_conc
            @test output.segments == segments
            @test isempty(brm_outputs(descriptor; role=:pointwise_loglik))
            @test :predict in [op.name for op in descriptor.operations]
            @test :fit ∉ [op.name for op in descriptor.operations]
            problem = brm_execute(descriptor, :instantiate)
            @test LogDensityProblems.dimension(problem) == 0
            names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
            coords = brm_output_coordinates(output, names)
            @test length(coords) == last(segments)
            draws = map(1:4) do seed
                constrained = BridgeStan.param_constrain(problem.model, Float64[];
                    include_tp=true, include_gq=true,
                    rng=BridgeStan.StanRNG(problem.model, seed))
                constrained[coords]
            end
            @test all(all(isfinite, draw) && all(draw .>= bounds) for draw in draws)
            @test any(draw != first(draws) for draw in draws[2:end])
            predicted = brm_execute(descriptor, :predict;
                problem, draws=Float64[], seed=1)
            @test getproperty(predicted, output.name) == first(draws)
        end
    end
end
