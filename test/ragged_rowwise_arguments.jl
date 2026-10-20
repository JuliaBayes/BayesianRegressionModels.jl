# Run: julia --startup-file=no --project=test test/ragged_rowwise_arguments.jl
using Test
using BayesianRegressionModels
using StanBlocks
using BridgeStan, LogDensityProblems
using Distributions: Exponential, Normal

# A row-wise scale helper over a ragged location, observed outside the cell
# that computes the location. `addprop` must run on each subject's rows once,
# like the location itself, not on the whole flattened location once per
# subject.
const model_scope = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_k ~ 1 + (1 | p | subject)
    @plate for i in eachindex(log_k)
        loc[i] = exp.(-exp(log_k[i]) .* t[i])
    end
    y ~ censored(Normal(loc, addprop(loc, sigma_add, sigma_prop)); lower=lloq)
end

# The same law observed inside the cell: the reference density.
const in_cell = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_k ~ 1 + (1 | p | subject)
    @plate for i in eachindex(log_k)
        loc[i] = exp.(-exp(log_k[i]) .* t[i])
        y[i] ~ censored(Normal(loc[i], addprop(loc[i], sigma_add, sigma_prop));
            lower=lloq[i])
    end
end

const uncensored_model_scope = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_k ~ 1 + (1 | p | subject)
    @plate for i in eachindex(log_k)
        loc[i] = exp.(-exp(log_k[i]) .* t[i])
    end
    y ~ Normal(loc, addprop(loc, sigma_add, sigma_prop))
end

# The censored model observed beside a per-subject response `qt`, and the
# same model without `y`: with `y` omitted, only `qt` conditions the fit.
const with_qt = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_k ~ 1 + (1 | p | subject)
    @plate for i in eachindex(log_k)
        loc[i] = exp.(-exp(log_k[i]) .* t[i])
    end
    y ~ censored(Normal(loc, addprop(loc, sigma_add, sigma_prop)); lower=lloq)
    qt ~ Normal(log_k, 1.0)
end
const qt_only = @brm begin
    log_k ~ 1 + (1 | p | subject)
    qt ~ Normal(log_k, 1.0)
end

const joined = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_k ~ 1 + (1 | p | subject)
    loc ~ kernel(ragged(t, obs_subject), log_k) do ts, lk
        exp.(-exp(lk) .* ts)
    end
    ragged(y, obs_subject) ~ censored(
        Normal(loc, addprop(loc, sigma_add, sigma_prop)); lower=lloq)
end

const joined_scaled = @brm begin
    sigma ~ Exponential(1)
    log_k ~ 1 + (1 | p | subject)
    loc ~ kernel(ragged(t, obs_subject), log_k) do ts, lk
        exp.(-exp(lk) .* ts)
    end
    ragged(y, obs_subject) ~ Normal(2 * loc, sigma)
end

const grouped_data = (;
    subject=["a", "b", "c"],
    t=[[0.5, 1.0, 2.0], [0.5], [1.0, 2.0, 3.0, 6.0]],
    y=[[0.8, 0.6, 0.3], [0.9], [0.5, 0.3, 0.1, 0.05]],
    lloq=[fill(0.05, 3), [0.05], fill(0.05, 4)],
)
const joined_data = (;
    subject=["b", "a"],
    obs_subject=["a", "b", "a", "b", "b", "a", "b"],
    t=[0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0],
    y=[0.8, 0.9, 0.6, 0.7, 0.4, 0.3, 0.1],
    lloq=fill(0.1, 7),
)
drop_response(data) = Base.structdiff(data, (; y=0))

observation(sb) = only(s for s in sb.model.model.args if
    Meta.isexpr(s, :call) && length(s.args) == 3 && s.args[1] === :~ &&
    s.args[2] === :y)
argument_plates(sb) = [s for s in sb.model.model.args if
    Meta.isexpr(s, :call) && length(s.args) == 3 && s.args[1] === :~ &&
    s.args[2] isa Symbol && startswith(String(s.args[2]), "y_argument_")]
# The reported shape: `addprop` applied to a plate result's whole flat memory,
# then sliced to one subject inside the per-subject observation loop.
whole_memory_addprop(code) = occursin(r"addprop\(\w+__pl_mem_\d+,", code)
@test whole_memory_addprop(
    "addprop(loc__pl_mem_1, sigma_add, sigma_prop)[ragged_start(loc__pl_end_1, g)")

@testset "row-wise scale over a plate result runs per subject" begin
    sb = SBBRMI(model_scope(grouped_data); mod=@__MODULE__)
    plate = only(argument_plates(sb))
    @test occursin("addprop", sprint(show, plate))
    @test plate.args[2] in observation(sb).args[3].args
    code = BayesianRegressionModels.stan_code(sb)
    @test !whole_memory_addprop(code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    reference = SBBRMI(in_cell(grouped_data); mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model)
    expected = StanBlocks.stan_instantiate(reference.model)
    dimension = LogDensityProblems.dimension(problem)
    @test LogDensityProblems.dimension(expected) == dimension
    for x in (zeros(dimension), collect(range(-0.4, 0.3; length=dimension)),
              fill(0.2, dimension))
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, x)
        reference_value, reference_gradient =
            LogDensityProblems.logdensity_and_gradient(expected, x)
        @test isfinite(value)
        @test value ≈ reference_value rtol=1e-12
        @test gradient ≈ reference_gradient rtol=1e-12
    end
end

@testset "an omitted ragged join reads the same grouped argument" begin
    replay_data = drop_response(merge(joined_data, (;
        obs_subject=["b", "a", "b", "a", "b"], t=[0.5, 0.5, 1.0, 2.0, 2.0],
        lloq=fill(0.1, 5))))
    for builder in (joined, joined_scaled)
        bound = SBBRMI(builder(joined_data); mod=@__MODULE__)
        omitted = @test_logs (:warn, r"`y` bind\(s\) no data column") SBBRMI(
            builder(drop_response(joined_data)); mod=@__MODULE__)
        @test observation(omitted) == observation(bound)
        @test length(argument_plates(bound)) == 1
        @test length(argument_plates(omitted)) == 1
        # Without values, the argument plate iterates the join's row
        # partition: kernel subject order, flat row numbers.
        @test !haskey(bound.data, :y_rows)
        @test omitted.data[:y_rows] == [[2, 4, 5, 7], [1, 3, 6]]
        @test omitted.preproc[:y_rows].kind === :ragged_rows
        replayed = reprocess(omitted, replay_data)
        @test replayed.data[:y_rows] == [[1, 3, 5], [2, 4]]
        for (model, segments) in ((bound, nothing), (omitted, [4, 7]),
                                  (replayed, [3, 5]))
            code = BayesianRegressionModels.stan_code(model)
            @test !whole_memory_addprop(code)
            @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
            segments === nothing && continue
            output = brm_output(brm_descriptor(model), :y;
                role=:posterior_predictive)
            @test output.segments == segments
        end
    end
end

# The plate's iterables, in order: what each argument plate's cells walk over.
plate_iterables(plate) = plate.args[3].args[1].args[2:end]

@testset "an omitted model-scope response reads the same grouped argument" begin
    replay_data = (; subject=["a", "b", "c"], t=[[0.5], [0.5, 1.0], [1.0, 2.0]],
        lloq=[[0.05], fill(0.05, 2), fill(0.05, 2)])
    for builder in (model_scope, uncensored_model_scope)
        bound = SBBRMI(builder(grouped_data); mod=@__MODULE__)
        omitted = @test_logs (:warn, r"`y` bind\(s\) no data column") SBBRMI(
            builder(drop_response(grouped_data)); mod=@__MODULE__)
        @test observation(omitted) == observation(bound)
        @test !haskey(omitted.data, :y)
        @test omitted.model.observations == (:y,)
        y = only(d for d in generative_plan(omitted).declarations if d.target === :y)
        @test y.role === :observation
        @test isnothing(y.data_source)
        # Bound, the argument plate walks the response's groups beside the
        # plate result; omitted, it walks the plate result's own cells.
        @test plate_iterables(only(argument_plates(bound))) == [:y, :loc]
        plate = only(argument_plates(omitted))
        @test plate_iterables(plate) == [:loc]
        @test occursin("addprop", sprint(show, plate))
        replayed = reprocess(omitted, replay_data)
        for (model, segments) in ((omitted, [3, 4, 8]), (replayed, [1, 3, 5]))
            code = BayesianRegressionModels.stan_code(model)
            @test !whole_memory_addprop(code)
            @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
            output = brm_output(brm_descriptor(model), :y;
                role=:posterior_predictive)
            @test output.segments == segments
        end
        problem = StanBlocks.stan_instantiate(omitted.model)
        @test LogDensityProblems.dimension(problem) == 0
        names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
        output = brm_output(brm_descriptor(omitted), :y; role=:posterior_predictive)
        coordinates = brm_output_coordinates(output, names)
        @test length(coordinates) == 8
        draw = BridgeStan.param_constrain(problem.model, Float64[];
            include_tp=true, include_gq=true,
            rng=BridgeStan.StanRNG(problem.model, 1))[coordinates]
        @test all(isfinite, draw)
        builder === model_scope && @test all(draw .>= 0.05)
    end

    # Beside a bound response, the omitted `y` adds no density: the fit is the
    # `qt`-only model's, coordinate for coordinate.
    qt = [0.1, -0.2, 0.3]
    mixed = @test_logs (:warn, r"`y` bind\(s\) no data column") SBBRMI(
        with_qt(merge(drop_response(grouped_data), (; qt))); mod=@__MODULE__)
    @test mixed.model.observations == (:qt, :y)
    @test !whole_memory_addprop(BayesianRegressionModels.stan_code(mixed))
    reference = SBBRMI(qt_only((; subject=grouped_data.subject, qt));
        mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(mixed.model)
    expected = StanBlocks.stan_instantiate(reference.model)
    dimension = LogDensityProblems.dimension(problem)
    @test dimension == LogDensityProblems.dimension(expected) > 0
    @test BridgeStan.param_unc_names(problem.model) ==
          BridgeStan.param_unc_names(expected.model)
    for x in (zeros(dimension), collect(range(-0.4, 0.3; length=dimension)))
        @test LogDensityProblems.logdensity(problem, x) ≈
              LogDensityProblems.logdensity(expected, x) rtol=1e-12
    end
end
