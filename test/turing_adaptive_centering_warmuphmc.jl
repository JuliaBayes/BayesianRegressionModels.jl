using Test
using BayesianRegressionModels
import DifferentiationInterface as DI
using Distributions: Exponential, Laplace, Normal
import Enzyme
using LogDensityProblems
using Random: Xoshiro
using Statistics: std
using Turing
using WarmupHMC

const BRM = BayesianRegressionModels
const DP = Turing.DynamicPPL
const TURING_AC_EXT = Base.get_extension(
    BRM, :BayesianRegressionModelsTuringWarmupHMCExt,
)
const ENZYME_BACKEND = DI.AutoEnzyme()

function random_intercept_backend(; centered_groups=())
    data = (;
        x=collect(range(-1.5, 1.5; length=12)),
        subject=repeat(["a", "b", "c"]; inner=4),
    )
    group_effect = Dict("a" => -0.35, "b" => 0.1, "c" => 0.45)
    residual = repeat([-0.08, 0.03, 0.06, -0.02], 3)
    data = merge(data, (;
        y=[0.25 + 0.7data.x[i] + group_effect[data.subject[i]] + residual[i]
           for i in eachindex(data.x)],
    ))
    brmi = (@brm begin
        mu ~ 1 + x + (1 | subject)
        y ~ Normal(mu, 0.6)
    end)(data)
    TuringBRMI(brmi; centered_groups)
end

function random_intercept_problem(backend)
    parameters = (;
        beta_pop=[0.2, 0.65],
        group_1_1=(; log_scale=log(0.7), z=[0.0, -0.4, 0.6]),
    )
    vi = DP.VarInfo(
        backend.model, DP.InitFromParams(parameters), DP.UnlinkAll(),
    )
    problem = DP.LogDensityFunction(backend.model, DP.getlogjoint_internal, vi)
    problem, collect(DP.get_sample_input_vector(problem))
end

function set_turing_sources!(problem, controls)
    ir = WarmupHMC.reparametrizer(problem)
    length(ir.pairs) == length(controls) || throw(DimensionMismatch())
    ir.pairs .= map(ir.pairs, controls) do (idx, value), c
        idx => WarmupHMC.Reparametrization(
            value.target, WarmupHMC.PartiallyCentered(c), value.args...,
        )
    end
    problem.scoring_plan.synchronize!(ir)
    problem
end

function manual_turing_map(x, log_scale_index, effect_indices, controls)
    y = copy(x)
    log_scale = x[log_scale_index]
    ljac = zero(eltype(x))
    for (idx, c) in zip(effect_indices, controls)
        y[idx] = x[idx] * exp(-c * log_scale)
        ljac -= c * log_scale
    end
    ljac, y
end

function central_difference(f, x; step=1e-5)
    gradient = similar(x)
    for i in eachindex(x)
        left = copy(x)
        right = copy(x)
        left[i] -= step
        right[i] += step
        gradient[i] = (f(right) - f(left)) / (2step)
    end
    gradient
end

@testset "Turing native scalar adaptive-centering transform is exact" begin
    @test !isnothing(TURING_AC_EXT)
    backend = random_intercept_backend()
    density, q = random_intercept_problem(backend)
    problem = adaptive_centering_problem(backend, density, ENZYME_BACKEND)
    @test LogDensityProblems.capabilities(typeof(problem)) isa
          LogDensityProblems.LogDensityOrder{1}
    ir = WarmupHMC.reparametrizer(problem)
    ranges = DP.get_all_ranges_and_transforms(density)
    log_scale_index = only(ranges[DP.@varname(group_1_1.log_scale)].range)
    effect_indices = collect(ranges[DP.@varname(group_1_1.z)].range)

    @test first.(ir.pairs) == effect_indices
    @test all(value.target.c == 0.0 && value.source.c == 0.0
              for (_idx, value) in ir.pairs)

    controls = [0.2, 0.7, 1.0]
    set_turing_sources!(problem, controls)
    source = copy(q)
    source[log_scale_index] = log(1.8)
    source[effect_indices] .= [0.0, -0.25, 1.1]
    expected_ljac, expected = manual_turing_map(
        source, log_scale_index, effect_indices, controls,
    )
    ljac, mapped = ir(source)
    @test ljac ≈ expected_ljac atol=1e-15 rtol=0
    @test mapped ≈ expected atol=1e-15 rtol=0

    centered_source = copy(q)
    centered_source[effect_indices] .=
        exp(q[log_scale_index]) .* q[effect_indices]
    set_turing_sources!(problem, ones(length(effect_indices)))
    endpoint_ljac, endpoint = ir(centered_source)
    @test endpoint ≈ q atol=1e-15 rtol=0
    @test endpoint_ljac ≈
          -length(effect_indices) * q[log_scale_index] atol=1e-15 rtol=0

    set_turing_sources!(problem, controls)
    expected_density =
        expected_ljac + LogDensityProblems.logdensity(density, expected)
    inner_density, inner_gradient =
        LogDensityProblems.logdensity_and_gradient(problem.problem, expected)
    reference_inner_gradient = central_difference(
        x -> LogDensityProblems.logdensity(density, x), expected,
    )
    actual_density, gradient =
        LogDensityProblems.logdensity_and_gradient(problem, source)
    reference_gradient = central_difference(
        x -> LogDensityProblems.logdensity(problem, x), source,
    )
    @test isapprox(
        inner_density,
        LogDensityProblems.logdensity(density, expected);
        atol=5e-12,
        rtol=5e-12,
    )
    @test inner_gradient ≈ reference_inner_gradient atol=2e-5 rtol=2e-5
    @test actual_density ≈ expected_density atol=5e-12 rtol=5e-12
    @test all(isfinite, gradient)
    @test gradient ≈ reference_gradient atol=2e-5 rtol=2e-5
end

@testset "Turing scalar candidate scores are source invariant" begin
    backend = random_intercept_backend()
    density, q = random_intercept_problem(backend)
    problem = adaptive_centering_problem(backend, density, ENZYME_BACKEND)
    ir = WarmupHMC.reparametrizer(problem)
    ranges = DP.get_all_ranges_and_transforms(density)
    log_scale_index = only(ranges[DP.@varname(group_1_1.log_scale)].range)
    effect_indices = collect(ranges[DP.@varname(group_1_1.z)].range)
    log_scale = log(1.6)
    scale = exp(log_scale)
    innovations = [0.0, -0.45, 0.8]
    invariant_gradient = [-0.7, 0.25, 0.9]

    frames = map((zeros(3), [0.2, 0.6, 1.0])) do controls
        set_turing_sources!(problem, controls)
        position = copy(q)
        gradient = zeros(length(q))
        position[log_scale_index] = log_scale
        for pair_number in eachindex(effect_indices)
            idx = effect_indices[pair_number]
            source_scale = scale^controls[pair_number]
            position[idx] = source_scale * innovations[pair_number]
            gradient[idx] = invariant_gradient[pair_number] / source_scale
        end
        problem.scoring_plan.prepare(ir, position, gradient)
    end
    for frame in frames
        @test frame.innovation ≈ innovations atol=2e-15 rtol=0
        @test frame.invariant_gradient ≈ invariant_gradient atol=2e-15 rtol=0
    end
    for pair_number in eachindex(ir.pairs), c in 0.0:0.1:1.0
        candidate = WarmupHMC.PartiallyCentered(c)
        observations = [problem.scoring_plan.score(
            frame, pair_number, first(ir.pairs[pair_number]),
            last(ir.pairs[pair_number]), candidate,
        ) for frame in frames]
        @test all(isapprox(obs[2], observations[1][2]; atol=2e-15, rtol=0)
                  for obs in observations)
        @test all(isapprox(obs[3], observations[1][3]; atol=2e-15, rtol=0)
                  for obs in observations)
    end
end

@testset "Turing native adaptive centering rejects unsupported geometry" begin
    backend = random_intercept_backend()
    density, _ = random_intercept_problem(backend)

    data_with_scale = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        subject=["b", "a", "b", "c"],
        y=[0.2, 1.1, -0.4, 0.7],
    )
    free_scale = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x + (1 | subject)
        y ~ Normal(mu, sigma)
    end)(data_with_scale))
    free_scale_error = try
        adaptive_centering_problem(free_scale, density, ENZYME_BACKEND)
        nothing
    catch error
        error
    end
    @test free_scale_error isa ErrorException
    @test occursin("free distribution parameters", free_scale_error.msg)

    non_gaussian = TuringBRMI((@brm begin
        mu ~ 1 + x + (1 | subject)
        y ~ Laplace(mu, 0.6)
    end)(data_with_scale))
    non_gaussian_density, _ = random_intercept_problem(non_gaussian)
    non_gaussian_error = try
        adaptive_centering_problem(
            non_gaussian, non_gaussian_density, ENZYME_BACKEND,
        )
        nothing
    catch error
        error
    end
    @test non_gaussian_error isa ErrorException
    @test occursin("not `Normal`", non_gaussian_error.msg)

    other_backend = random_intercept_backend()
    mismatch_error = try
        adaptive_centering_problem(other_backend, density, ENZYME_BACKEND)
        nothing
    catch error
        error
    end
    @test mismatch_error isa ErrorException
    @test occursin("exact model", mismatch_error.msg)

    centered = random_intercept_backend(centered_groups=:subject)
    centered_error = try
        adaptive_centering_problem(centered, density, ENZYME_BACKEND)
        nothing
    catch error
        error
    end
    @test centered_error isa ErrorException
    @test occursin("generated model is centered", centered_error.msg)

    data = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        subject=["b", "a", "b", "c"],
        y=[0.2, 1.1, -0.4, 0.7],
    )
    correlated = TuringBRMI((@brm begin
        mu ~ 1 + x + (1 + x | subject)
        y ~ Normal(mu, 0.6)
    end)(data))
    correlated_error = try
        adaptive_centering_problem(correlated, density, ENZYME_BACKEND)
        nothing
    catch error
        error
    end
    @test correlated_error isa ErrorException
    @test occursin("slope blocks", correlated_error.msg)

    stan_bridge_error = try
        adaptive_centering_blocks(backend, String[])
        nothing
    catch error
        error
    end
    @test stan_bridge_error isa ErrorException
    @test occursin("compiled Stan", stan_bridge_error.msg)
end

@testset "Turing scalar adaptive centering samples end to end" begin
    backend = random_intercept_backend()
    density, q = random_intercept_problem(backend)
    problem = adaptive_centering_problem(backend, density, ENZYME_BACKEND)
    result = WarmupHMC.adaptive_warmup_mcmc(
        Xoshiro(0x20260913),
        problem;
        n_draws=40,
        n_evaluations=500,
        stepsize_adaptation_limit=100,
        target_acceptance_rate=0.95,
        max_tree_depth=10,
        progress=nothing,
        monitor_ess=false,
        init=q,
    )

    @test size(result.posterior_position) ==
          (LogDensityProblems.dimension(density), 40)
    @test all(isfinite, result.posterior_position)
    @test all(>(1e-7), vec(std(result.posterior_position; dims=2)))
    @test result.n_divergent_samples == 0
    sources = [value.source.c for (_idx, value) in
               WarmupHMC.reparametrizer(problem).pairs]
    @test all(c -> c in 0.0:0.1:1.0, sources)
end
