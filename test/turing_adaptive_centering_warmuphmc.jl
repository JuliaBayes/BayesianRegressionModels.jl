using Test
using BayesianRegressionModels
import DifferentiationInterface as DI
using Distributions: Exponential, Laplace, LogNormal, Normal
import Enzyme
using LogDensityProblems
using Random: Xoshiro
using Serialization: deserialize
using Statistics: std
using Turing
using WarmupHMC

const BRM = BayesianRegressionModels
const DP = Turing.DynamicPPL
const TURING_AC_EXT = Base.get_extension(
    BRM, :BayesianRegressionModelsTuringWarmupHMCExt,
)
const ENZYME_BACKEND = DI.AutoEnzyme(;
    mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation=Enzyme.Const)

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

function same_axis_hsgp_backend()
    x = collect(range(-1.2, 1.2; length=12))
    mu = 0.25 .+ 0.35sin.(2.2x)
    sigma = exp.(-0.45 .+ 0.18cos.(1.7x))
    residual = repeat([-0.35, 0.12, 0.28, -0.08], 3)
    data = (; x, y=mu .+ sigma .* residual)
    builder = @brm begin
        mu ~ 1 + hsgp(x; k=3)
        log(sigma) ~ 1 + hsgp(x; k=2)
        y ~ Normal(mu, sigma)
    end
    TuringBRMI(builder(data))
end

function same_axis_hsgp_problem(backend; linked=true)
    mu_term = only(backend.plan.predictors[1].terms)
    sigma_term = only(backend.plan.predictors[2].terms)
    parameters = (;
        beta_pop=[0.15],
        term_mu_1=(;
            rho=mu_term.state.rho_lower + 0.55,
            sigma=0.65,
            beta_raw=[0.15, -0.2, 0.3],
        ),
        beta_pop_sigma=[-0.35],
        term_sigma_1=(;
            rho=sigma_term.state.rho_lower + 0.45,
            sigma=0.4,
            beta_raw=[0.2, -0.1],
        ),
    )
    strategy = linked ? DP.LinkAll() : DP.UnlinkAll()
    vi = DP.VarInfo(backend.model, DP.InitFromParams(parameters), strategy)
    problem = DP.LogDensityFunction(
        backend.model, DP.getlogjoint_internal, vi)
    problem, collect(DP.get_sample_input_vector(problem))
end

function motorcycle_hsgp_backend()
    lines = readlines(joinpath(
        @__DIR__, "..", "research", "adaptive_centering", "mcycle.csv"))
    first(lines) == "rownames,times,accel" || error("unexpected mcycle header")
    rows = split.(lines[2:end], ',')
    times = parse.(Float64, getindex.(rows, 2))
    accel = parse.(Float64, getindex.(rows, 3))
    length(times) == 133 || error("expected 133 motorcycle observations")
    xmin, xmax = extrema(times)
    x = @. -1 + 2 * (times - xmin) / (xmax - xmin)
    data = (; x, y=accel ./ std(accel))
    builder = @brm begin
        length_scale(mu, hsgp(x)) ~ LogNormal(0, 4)
        sd(mu, hsgp(x)) ~ LogNormal(0, 4)
        length_scale(sigma, hsgp(x)) ~ LogNormal(0, 4)
        sd(sigma, hsgp(x)) ~ LogNormal(0, 4)
        mu ~ hsgp(x; k=8, domain=(-1.5, 1.5))
        log(sigma) ~ hsgp(x; k=8, domain=(-1.5, 1.5))
        y ~ Normal(mu, sigma)
    end
    TuringBRMI(builder(data))
end

function motorcycle_hsgp_problem(backend)
    mu_term = only(backend.plan.predictors[1].terms)
    sigma_term = only(backend.plan.predictors[2].terms)
    parameters = (;
        term_mu_1=(;
            rho=mu_term.state.rho_lower + 0.55,
            sigma=0.65,
            beta_raw=collect(range(-0.16, 0.18; length=8)),
        ),
        term_sigma_1=(;
            rho=sigma_term.state.rho_lower + 0.45,
            sigma=0.4,
            beta_raw=collect(range(0.12, -0.1; length=8)),
        ),
    )
    vi = DP.VarInfo(
        backend.model, DP.InitFromParams(parameters), DP.LinkAll(),
    )
    problem = DP.LogDensityFunction(
        backend.model, DP.getlogjoint_internal, vi)
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

function manual_turing_hsgp_map(x, blocks, controls)
    y = copy(x)
    ljac = zero(eltype(x))
    pair = 0
    for block in blocks, basis in eachindex(block.effects)
        pair += 1
        log_scale = BRM._adaptive_hsgp_log_scale(x, block, basis)
        y[block.effects[basis]] = x[block.effects[basis]] *
                                  exp(-controls[pair] * log_scale)
        ljac -= controls[pair] * log_scale
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

@testset "Turing motorcycle HSGP metadata and scale-4 priors are exact" begin
    backend = motorcycle_hsgp_backend()
    density, q = motorcycle_hsgp_problem(backend)
    problem = adaptive_centering_problem(backend, density, ENZYME_BACKEND)
    inner = problem.problem
    contract = TURING_AC_EXT._two_hsgp_contract(backend)
    blocks, mu, log_sigma = TURING_AC_EXT._two_hsgp_geometry(
        backend, density, contract)
    components = [mu, log_sigma]
    @test blocks == first.(components)
    @test inner.model === density.model

    @test LogDensityProblems.dimension(density) == 20
    @test all(isempty(component[2]) for component in components)
    # `_hsgp_gradient_component` returns plain `(block, beta_indices)` tuples
    # since the native-gradient perf pass; the prior scales it still parses
    # (and refuses loudly) are pinned through the same parser here.
    @test TURING_AC_EXT._hsgp_prior_scale(
        contract.mu_term.state.rho_prior, :mu, "length-scale") == 4.0
    @test TURING_AC_EXT._hsgp_prior_scale(
        contract.sigma_term.state.rho_prior, :sigma, "length-scale") == 4.0
    @test TURING_AC_EXT._hsgp_prior_scale(
        contract.mu_term.state.sigma_prior, :mu, "marginal-SD") == 4.0
    @test TURING_AC_EXT._hsgp_prior_scale(
        contract.sigma_term.state.sigma_prior, :sigma, "marginal-SD") == 4.0
    @test getfield.(blocks, :length_scale_lower) == [[0.0], [0.0]]
    @test blocks[1].length_scales == [1]
    @test blocks[1].sd == 2
    @test blocks[1].effects == collect(3:10)
    @test blocks[2].length_scales == [11]
    @test blocks[2].sd == 12
    @test blocks[2].effects == collect(13:20)

    inner_density, inner_gradient =
        LogDensityProblems.logdensity_and_gradient(inner, q)
    reference_inner_gradient = central_difference(
        x -> LogDensityProblems.logdensity(density, x), q)
    @test inner_density == LogDensityProblems.logdensity(density, q)
    @test inner_gradient ≈ reference_inner_gradient atol=5e-5 rtol=5e-5

    controls = collect(range(0.1, 1.0; length=16))
    set_turing_sources!(problem, controls)
    expected_ljac, expected = manual_turing_hsgp_map(q, blocks, controls)
    actual_ljac, actual = WarmupHMC.reparametrizer(problem)(q)
    @test actual_ljac ≈ expected_ljac atol=3e-14 rtol=0
    @test actual ≈ expected atol=3e-14 rtol=0
    density_value, gradient =
        LogDensityProblems.logdensity_and_gradient(problem, q)
    reference_gradient = central_difference(
        x -> LogDensityProblems.logdensity(problem, x), q)
    @test density_value ≈ expected_ljac +
          LogDensityProblems.logdensity(density, expected) atol=1e-11 rtol=1e-11
    @test gradient ≈ reference_gradient atol=7e-5 rtol=7e-5
end

@testset "Turing same-axis HSGP metadata and transform are exact" begin
    backend = same_axis_hsgp_backend()
    density, q = same_axis_hsgp_problem(backend)
    problem = adaptive_centering_problem(backend, density, ENZYME_BACKEND)
    @test LogDensityProblems.capabilities(typeof(problem)) isa
          LogDensityProblems.LogDensityOrder{1}
    inner = problem.problem
    contract = TURING_AC_EXT._two_hsgp_contract(backend)
    blocks, _mu, _log_sigma = TURING_AC_EXT._two_hsgp_geometry(
        backend, density, contract)
    @test inner.model === density.model
    @test getfield.(blocks, :logical) == [:mu, :sigma]
    @test getfield.(blocks, :term) == [:hsgp_x, :hsgp_x]
    @test blocks[1].effects == [4, 5, 6]
    @test blocks[2].effects == [10, 11]
    @test blocks[1].length_scales == [2]
    @test blocks[2].length_scales == [8]
    @test blocks[1].sd == 3
    @test blocks[2].sd == 9
    @test isempty(intersect(blocks[1].effects, blocks[2].effects))
    @test blocks[1].omega2 == backend.plan.predictors[1].terms[1].state.omega2
    @test blocks[2].omega2 == backend.plan.predictors[2].terms[1].state.omega2

    ir = WarmupHMC.reparametrizer(problem)
    @test first.(ir.pairs) == [4, 5, 6, 10, 11]
    @test all(value.target.c == 0.0 && value.source.c == 0.0
              for (_idx, value) in ir.pairs)
    controls = [0.2, 0.5, 0.8, 0.4, 1.0]
    set_turing_sources!(problem, controls)
    source = copy(q)
    source[blocks[1].effects] .= [0.3, -0.25, 0.45]
    source[blocks[2].effects] .= [-0.2, 0.35]
    expected_ljac, expected = manual_turing_hsgp_map(
        source, blocks, controls)
    ljac, mapped = ir(source)
    @test ljac ≈ expected_ljac atol=2e-14 rtol=0
    @test mapped ≈ expected atol=2e-14 rtol=0

    inner_density, inner_gradient =
        LogDensityProblems.logdensity_and_gradient(inner, expected)
    reference_inner_gradient = central_difference(
        x -> LogDensityProblems.logdensity(density, x), expected)
    expected_density = expected_ljac +
        LogDensityProblems.logdensity(density, expected)
    actual_density, gradient =
        LogDensityProblems.logdensity_and_gradient(problem, source)
    reference_gradient = central_difference(
        x -> LogDensityProblems.logdensity(problem, x), source)
    @test isapprox(
        inner_density,
        LogDensityProblems.logdensity(density, expected);
        atol=5e-12,
        rtol=5e-12,
    )
    @test inner_gradient ≈ reference_inner_gradient atol=3e-5 rtol=3e-5
    @test actual_density ≈ expected_density atol=5e-12 rtol=5e-12
    @test all(isfinite, gradient)
    @test gradient ≈ reference_gradient atol=3e-5 rtol=3e-5

    centered_source = copy(q)
    endpoint_jacobian = 0.0
    for block in blocks, basis in eachindex(block.effects)
        log_scale = BRM._adaptive_hsgp_log_scale(q, block, basis)
        centered_source[block.effects[basis]] *= exp(log_scale)
        endpoint_jacobian -= log_scale
    end
    set_turing_sources!(problem, ones(length(ir.pairs)))
    endpoint_ljac, endpoint = ir(centered_source)
    @test endpoint ≈ q atol=2e-14 rtol=0
    @test endpoint_ljac ≈ endpoint_jacobian atol=2e-14 rtol=0
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

    hsgp_backend = same_axis_hsgp_backend()
    hsgp_density, _ = same_axis_hsgp_problem(hsgp_backend)
    hsgp_unlinked, _ = same_axis_hsgp_problem(hsgp_backend; linked=false)
    unlinked_error = try
        adaptive_centering_problem(
            hsgp_backend, hsgp_unlinked, ENZYME_BACKEND)
        nothing
    catch error
        error
    end
    @test unlinked_error isa ErrorException
    @test occursin("not linked", unlinked_error.msg)

    hsgp_data = (;
        x=collect(range(-1.0, 1.0; length=8)),
        z=collect(range(1.0, -1.0; length=8)),
        g=repeat(1:2; inner=4),
        y=zeros(8),
    )
    different_axes = TuringBRMI((@brm begin
        mu ~ 1 + hsgp(x; k=3)
        log(sigma) ~ 1 + hsgp(z; k=2)
        y ~ Normal(mu, sigma)
    end)(hsgp_data))
    axes_error = try
        adaptive_centering_problem(
            different_axes, hsgp_density, ENZYME_BACKEND)
        nothing
    catch error
        error
    end
    @test axes_error isa ErrorException
    @test occursin("do not share", axes_error.msg)

    grouped_hsgp = TuringBRMI((@brm begin
        mu ~ 1 + hsgp(x; k=3, by=g)
        log(sigma) ~ 1 + hsgp(x; k=2)
        y ~ Normal(mu, sigma)
    end)(hsgp_data))
    grouped_error = try
        adaptive_centering_problem(
            grouped_hsgp, hsgp_density, ENZYME_BACKEND)
        nothing
    catch error
        error
    end
    @test grouped_error isa ErrorException
    @test occursin("grouped HSGP", grouped_error.msg)

    non_lognormal = TuringBRMI((@brm begin
        length_scale(mu, hsgp(x)) ~ Normal(1, 0.5)
        mu ~ hsgp(x; k=3)
        log(sigma) ~ hsgp(x; k=2)
        y ~ Normal(mu, sigma)
    end)(hsgp_data))
    non_lognormal_error = try
        adaptive_centering_problem(
            non_lognormal, hsgp_density, ENZYME_BACKEND)
        nothing
    catch error
        error
    end
    @test non_lognormal_error isa ErrorException
    @test occursin("non-LogNormal", non_lognormal_error.msg)

    shifted_lognormal = TuringBRMI((@brm begin
        length_scale(mu, hsgp(x)) ~ LogNormal(1, 4)
        mu ~ hsgp(x; k=3)
        log(sigma) ~ hsgp(x; k=2)
        y ~ Normal(mu, sigma)
    end)(hsgp_data))
    shifted_lognormal_error = try
        adaptive_centering_problem(
            shifted_lognormal, hsgp_density, ENZYME_BACKEND)
        nothing
    catch error
        error
    end
    @test shifted_lognormal_error isa ErrorException
    @test occursin("zero-location", shifted_lognormal_error.msg)
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

@testset "Turing same-axis HSGP adapts online with checkpoint support" begin
    backend = motorcycle_hsgp_backend()
    density, q = motorcycle_hsgp_problem(backend)
    problem = adaptive_centering_problem(backend, density, ENZYME_BACKEND)
    result = mktempdir() do checkpoint_dir
        sampled = WarmupHMC.adaptive_warmup_mcmc(
            Xoshiro(0x48534750),
            problem;
            n_draws=20,
            n_evaluations=500,
            stepsize_adaptation_limit=100,
            target_acceptance_rate=0.95,
            max_tree_depth=10,
            checkpoint_dir,
            progress=nothing,
            monitor_ess=false,
            init=q,
        )
        checkpoint = joinpath(checkpoint_dir, "cp_latest.jls")
        @test isfile(checkpoint)
        payload = deserialize(checkpoint)
        @test first.(payload.reparam_sources) ==
              first.(WarmupHMC.reparametrizer(problem).pairs)
        @test all(last(pair).c in 0.0:0.1:1.0
                  for pair in payload.reparam_sources)
        raw = isempty(payload.posterior_position) ?
            payload.dropped_posterior_position : payload.posterior_position
        @test size(raw, 1) == LogDensityProblems.dimension(density)
        @test all(isfinite, WarmupHMC.back_transform(payload, problem, raw))
        sampled
    end

    @test size(result.posterior_position) ==
          (LogDensityProblems.dimension(density), 20)
    @test all(isfinite, result.posterior_position)
    @test all(>(1e-7), vec(std(result.posterior_position; dims=2)))
    @test result.n_divergent_samples == 0
    sources = [value.source.c for (_idx, value) in
               WarmupHMC.reparametrizer(problem).pairs]
    @test all(c -> c in 0.0:0.1:1.0, sources)
    @test any(!iszero, sources)
end
