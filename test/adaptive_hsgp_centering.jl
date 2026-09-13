using Test
using BayesianRegressionModels
using DifferentiationInterface: AutoEnzyme, Constant
import DifferentiationInterface
using Distributions: Normal, Uniform
using Enzyme
using LinearAlgebra
using LogDensityProblems
using Random: Xoshiro
using StanBlocks
using WarmupHMC

const BRM = BayesianRegressionModels
const HSGP_AC_EXT = Base.get_extension(
    BayesianRegressionModels, :BayesianRegressionModelsWarmupHMCExt,
)

const HSGP_BUILDER = @brm begin
    mu ~ 1 + hsgp(time; k=3)
    log_sigma ~ 1 + hsgp(time_noise; k=2)
    y ~ Normal(mu, exp(log_sigma))
end

const HSGP_TIME = collect(range(-1.0, 1.0; length=18))
const HSGP_DATA = (;
    time=HSGP_TIME,
    time_noise=copy(HSGP_TIME),
    y=[0.3 + 0.7 * sinpi(t) + exp(-1.2 + 0.2cospi(t)) *
       0.08sinpi(7t) for t in HSGP_TIME],
)

function hsgp_fake_unc_names()
    vcat(
        ["hsgp_time_rho_iso", "hsgp_time_sigma"],
        ["hsgp_time_beta_raw.$basis" for basis in 1:3],
        ["hsgp_time_noise_rho_iso", "hsgp_time_noise_sigma"],
        ["hsgp_time_noise_beta_raw.$basis" for basis in 1:2],
    )
end

function set_hsgp_sources!(state, ir, controls)
    length(controls) == length(ir.pairs) || throw(DimensionMismatch())
    ir.pairs .= map(ir.pairs, controls) do (idx, value), c
        idx => WarmupHMC.Reparametrization(
            value.target, WarmupHMC.PartiallyCentered(c), value.args...,
        )
    end
    HSGP_AC_EXT._sync_sources!(state, ir)
end

function manual_hsgp_map(x, blocks, controls)
    y = copy(x)
    ljac = zero(eltype(x))
    p = 1
    for block in blocks, basis in eachindex(block.effects)
        log_scale = BRM._adaptive_hsgp_log_scale(x, block, basis)
        c = controls[p]
        y[block.effects[basis]] = x[block.effects[basis]] * exp(-c * log_scale)
        ljac -= c * log_scale
        p += 1
    end
    ljac, y
end

struct HSGPQuadraticTarget
    dimension::Int
end

LogDensityProblems.dimension(target::HSGPQuadraticTarget) = target.dimension
LogDensityProblems.capabilities(::Type{HSGPQuadraticTarget}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::HSGPQuadraticTarget, x) = -sum(abs2, x) / 2
LogDensityProblems.logdensity_and_gradient(target::HSGPQuadraticTarget, x) =
    (LogDensityProblems.logdensity(target, x), -x)

@testset "HSGP adaptive metadata is semantic and fail-closed" begin
    sb = SBBRMI(HSGP_BUILDER(HSGP_DATA); mod=@__MODULE__)
    names = hsgp_fake_unc_names()
    blocks = BRM._adaptive_hsgp_centering_blocks(sb, names)
    @test length(blocks) == 2
    @test getfield.(blocks, :logical) == [:mu, :log_sigma]
    @test getfield.(blocks, :term) == [:hsgp_time, :hsgp_time_noise]
    @test getfield.(blocks, :target_c) == [0.0, 0.0]
    @test blocks[1].effects == [3, 4, 5]
    @test blocks[2].effects == [8, 9]
    @test blocks[1].length_scales == [1]
    @test blocks[2].length_scales == [6]
    @test blocks[1].sd == 2
    @test blocks[2].sd == 7
    @test size(blocks[1].omega2) == (3, 1)
    @test size(blocks[2].omega2) == (2, 1)
    @test blocks[1].omega2[:, 1] == sb.data[:omega2_hsgp_time][:, 1]
    @test blocks[2].omega2[:, 1] == sb.data[:omega2_hsgp_time_noise][:, 1]
    @test blocks[1].length_scale_lower == [sb.data[:rho_lower_hsgp_time]]
    @test blocks[2].length_scale_lower == [sb.data[:rho_lower_hsgp_time_noise]]

    missing = filter(!=("hsgp_time_beta_raw.2"), names)
    @test_throws "basis_weights" BRM._adaptive_hsgp_centering_blocks(sb, missing)

    periodic = @brm HSGP_DATA begin
        mu ~ 1 + hsgp(time; k=3, cov=:periodic, period=2.5)
        y ~ Normal(mu, 1)
    end
    periodic_sb = SBBRMI(periodic; mod=@__MODULE__)
    @test_throws "covariance `periodic`" BRM._adaptive_hsgp_centering_blocks(
        periodic_sb, String[],
    )

    grouped_data = merge(HSGP_DATA, (; group=repeat([:a, :b, :c], inner=6)))
    grouped = @brm grouped_data begin
        mu ~ 1 + hsgp(time; k=3, by=group)
        y ~ Normal(mu, 1)
    end
    grouped_sb = SBBRMI(grouped; mod=@__MODULE__)
    @test_throws "grouped HSGP" BRM._adaptive_hsgp_centering_blocks(
        grouped_sb, String[],
    )

    bounded = @brm HSGP_DATA begin
        mu ~ 1 + hsgp(time; k=3)
        length_scale(mu, hsgp(time)) ~ Uniform(0.5, 2.0)
        y ~ Normal(mu, 1)
    end
    bounded_sb = SBBRMI(bounded; mod=@__MODULE__)
    bounded_names = vcat(
        ["hsgp_time_rho_iso", "hsgp_time_sigma"],
        ["hsgp_time_beta_raw.$basis" for basis in 1:3],
    )
    @test_throws "unsupported Stan constraint" BRM._adaptive_hsgp_centering_blocks(
        bounded_sb, bounded_names,
    )
end

@testset "per-basis HSGP transform, Jacobian, scores, and Enzyme gradient" begin
    sb = SBBRMI(HSGP_BUILDER(HSGP_DATA); mod=@__MODULE__)
    names = hsgp_fake_unc_names()
    blocks = BRM._adaptive_hsgp_centering_blocks(sb, names)
    state, ir = HSGP_AC_EXT._adaptive_hsgp_centering_reparametrizer(blocks)
    @test first.(ir.pairs) == vcat(blocks[1].effects, blocks[2].effects)

    x = collect(range(-0.45, 0.65; length=length(names)))
    x[blocks[1].length_scales] .= -0.2
    x[blocks[1].sd] = -0.4
    x[blocks[2].length_scales] .= 0.15
    x[blocks[2].sd] = -0.7
    log_scales = [
        BRM._adaptive_hsgp_log_scale(x, block, basis)
        for block in blocks for basis in eachindex(block.effects)
    ]
    @test length(unique(log_scales)) == length(log_scales)

    zeros_c = zeros(length(ir.pairs))
    set_hsgp_sources!(state, ir, zeros_c)
    @test isequal(ir(x), (0.0, x))

    controls = [0.2, 0.5, 1.0, 0.7, 0.3]
    set_hsgp_sources!(state, ir, controls)
    expected_ljac, expected = manual_hsgp_map(x, blocks, controls)
    ljac, mapped = ir(x)
    @test ljac ≈ expected_ljac atol=2e-14
    @test mapped ≈ expected atol=2e-14
    inverse_ljac, roundtrip = WarmupHMC._inverse_with_logabsdet_jacobian(ir, mapped)
    @test inverse_ljac ≈ -ljac atol=2e-14
    @test roundtrip ≈ x atol=2e-14

    ones_c = ones(length(ir.pairs))
    set_hsgp_sources!(state, ir, ones_c)
    centered_ljac, centered = ir(x)
    expected_centered_ljac, expected_centered = manual_hsgp_map(x, blocks, ones_c)
    @test centered_ljac ≈ expected_centered_ljac atol=2e-14
    @test centered ≈ expected_centered atol=2e-14

    set_hsgp_sources!(state, ir, controls)
    weight = collect(range(0.3, 1.4; length=length(x)))
    objective(v, transform, w) = ((j, q) = transform(v); j + dot(w, q))
    ad_gradient = DifferentiationInterface.gradient(
        objective, AutoEnzyme(), x, Constant(ir), Constant(weight),
    )
    step = 1e-6
    finite_difference = [begin
        plus, minus = copy(x), copy(x)
        plus[i] += step
        minus[i] -= step
        (objective(plus, ir, weight) - objective(minus, ir, weight)) / (2step)
    end for i in eachindex(x)]
    @test ad_gradient ≈ finite_difference atol=3e-8 rtol=3e-8

    innovations = collect(range(-0.8, 0.9; length=length(ir.pairs)))
    invariant_gradient = collect(range(0.7, -0.6; length=length(ir.pairs)))
    frames = map((zeros_c, controls, ones_c)) do source_controls
        set_hsgp_sources!(state, ir, source_controls)
        position = copy(x)
        gradient = zeros(length(x))
        p = 1
        for block in blocks, basis in eachindex(block.effects)
            scale = exp(BRM._adaptive_hsgp_log_scale(position, block, basis))
            c = source_controls[p]
            idx = block.effects[basis]
            position[idx] = scale^c * innovations[p]
            gradient[idx] = invariant_gradient[p] / scale^c
            p += 1
        end
        HSGP_AC_EXT._prepare_frame(state, ir, position, gradient)
    end
    for frame in frames
        @test frame.location == zeros(length(ir.pairs))
        @test frame.innovation ≈ innovations atol=2e-14
        @test frame.invariant_gradient ≈ invariant_gradient atol=2e-14
    end
    for p in eachindex(ir.pairs), candidate_c in 0.0:0.1:1.0
        candidate = WarmupHMC.PartiallyCentered(candidate_c)
        observations = [HSGP_AC_EXT._score_candidate(
            frame, p, first(ir.pairs[p]), last(ir.pairs[p]), candidate,
        ) for frame in frames]
        @test all(isapprox(obs[2], observations[1][2]; atol=2e-14)
                  for obs in observations)
        @test all(isapprox(obs[3], observations[1][3]; atol=2e-14)
                  for obs in observations)
    end

    target = HSGPQuadraticTarget(length(x))
    wrapped = WarmupHMC.ReparametrizedProblem(ir, target, AutoEnzyme())
    density, gradient = LogDensityProblems.logdensity_and_gradient(wrapped, x)
    density_fd = [begin
        plus, minus = copy(x), copy(x)
        plus[i] += 1e-5
        minus[i] -= 1e-5
        (LogDensityProblems.logdensity(wrapped, plus) -
         LogDensityProblems.logdensity(wrapped, minus)) / 2e-5
    end for i in eachindex(x)]
    @test isfinite(density)
    @test gradient ≈ density_fd atol=3e-5 rtol=3e-5
end

@testset "two-HSGP BridgeStan density, gradient, and online warmup" begin
    sb = SBBRMI(HSGP_BUILDER(HSGP_DATA); mod=@__MODULE__)
    cache = joinpath(tempdir(), "brm-adaptive-hsgp-centering")
    mkpath(cache)
    problem = StanBlocks.stan_instantiate(
        sb.model; path=joinpath(cache, "two_hsgp.stan"),
    )
    unc_names = StanBlocks.BridgeStan.param_unc_names(problem.model)
    blocks = BRM._adaptive_hsgp_centering_blocks(sb, unc_names)
    @test length(blocks) == 2
    @test length.(getfield.(blocks, :effects)) == [3, 2]

    backend = AutoEnzyme()
    wrapped = adaptive_centering_problem(sb, problem, backend)
    ir = WarmupHMC.reparametrizer(wrapped)
    state = ir.pairs[1][2].args[1].state
    initial = zeros(length(unc_names))
    plain = LogDensityProblems.logdensity_and_gradient(problem, initial)
    adaptive = LogDensityProblems.logdensity_and_gradient(wrapped, initial)
    @test adaptive[1] ≈ plain[1] atol=2e-11
    @test adaptive[2] ≈ plain[2] atol=2e-11
    @test WarmupHMC.candidate_scoring_plan(wrapped) isa
          WarmupHMC.CandidateScoringPlan

    controls = [0.2, 0.5, 1.0, 0.7, 0.3]
    set_hsgp_sources!(state, ir, controls)
    position = collect(range(-0.2, 0.25; length=length(unc_names)))
    ljac, model_position = ir(position)
    wrapped_lp, wrapped_gradient =
        LogDensityProblems.logdensity_and_gradient(wrapped, position)
    @test wrapped_lp ≈ ljac + LogDensityProblems.logdensity(problem, model_position) atol=2e-11
    @test all(isfinite, wrapped_gradient)
    step = 1e-5
    finite_difference = [begin
        plus, minus = copy(position), copy(position)
        plus[i] += step
        minus[i] -= step
        (LogDensityProblems.logdensity(wrapped, plus) -
         LogDensityProblems.logdensity(wrapped, minus)) / (2step)
    end for i in eachindex(position)]
    @test wrapped_gradient ≈ finite_difference atol=3e-5 rtol=3e-5

    online = adaptive_centering_problem(sb, problem, backend)
    result = WarmupHMC.adaptive_warmup_mcmc(
        Xoshiro(0x20260913), online;
        n_draws=20,
        n_evaluations=120,
        stepsize_adaptation_limit=20,
        max_tree_depth=7,
        progress=nothing,
        monitor_ess=false,
    )
    adapted = [value.c for (_, value) in WarmupHMC.reparam_sources(online)]
    @test length(adapted) == 5
    @test all(c -> c in 0.0:0.1:1.0, adapted)
    @test any(!=(0.0), adapted)
    @test size(result.posterior_position, 1) == length(unc_names)
    @test size(result.posterior_position, 2) >= 20
    @test all(isfinite, result.posterior_position)
end
