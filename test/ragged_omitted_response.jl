# Run: julia --project=test test/ragged_omitted_response.jl
using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal
using BridgeStan, LogDensityProblems

omitted_ragged_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    loc ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
        exp(lCL) .+ 0.0 .* idxs
    end
    ragged(pk_conc, obs_subject) ~ censored(
        Normal(loc, addprop(loc, sigma_add, sigma_prop)); lower=pk_lloq)
end

omitted_ragged_data = (;
    subject=["s2", "s1"],
    obs_subject=["s1", "s2", "s1", "s2", "s2", "s1", "s2"],
    obs_idx=collect(1.0:7.0),
    pk_lloq=[0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
)

@testset "omitted ragged response retains observation and layout" begin
    brmi = omitted_ragged_builder(omitted_ragged_data)
    sb = @test_logs (:warn, r"`pk_conc` bind\(s\) no data column") SBBRMI(
        brmi; mod=@__MODULE__)
    @test !haskey(sb.data, :pk_conc)
    @test !haskey(sb.preproc, :pk_conc)
    @test sb.model.observations == (:pk_conc,)
    bound_key = :pk_conc_lower_pk_lloq_ragged
    @test sb.data[bound_key] == [[0.11, 0.13, 0.14, 0.16], [0.10, 0.12, 0.15]]
    @test sb.preproc[bound_key].kind === :ragged_gather

    # The very same likelihood remains, with the same model-derived location
    # and scale. Omission changes the data binding, never the observation body.
    observed = SBBRMI(omitted_ragged_builder(merge(omitted_ragged_data,
        (; pk_conc=fill(0.2, 7)))); mod=@__MODULE__)
    observation(body) = only(s for s in body.args if
        Meta.isexpr(s, :call) && length(s.args) == 3 &&
        s.args[1] === :~ && s.args[2] === :pk_conc)
    @test observation(sb.model.model) == observation(observed.model.model)
    @test sb.data[bound_key] == observed.data[bound_key]

    # Frozen data replay regathers retained bounds from the new observation
    # design without demanding or manufacturing response values.
    replay_data = merge(omitted_ragged_data, (;
        obs_subject=["s2", "s1", "s2", "s1", "s2"],
        obs_idx=collect(1.0:5.0), pk_lloq=[0.21, 0.22, 0.23, 0.24, 0.25]))
    replayed = reprocess(sb, replay_data)
    @test !haskey(replayed.data, :pk_conc)
    @test !haskey(replayed.preproc, :pk_conc)
    @test replayed.model.observations == (:pk_conc,)
    @test replayed.data[bound_key] == [[0.21, 0.23, 0.25], [0.22, 0.24]]

    @test_throws "has 6 rows but the flat response has 7" SBBRMI(
        omitted_ragged_builder(merge(omitted_ragged_data,
            (; pk_lloq=fill(0.1, 6)))); mod=@__MODULE__)
    @test_throws "group lengths [3, 4]; expected [4, 3]" SBBRMI(
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

@testset "omitted ragged censored response draws and replay" begin
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
            @test output.source === :pk_conc
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
