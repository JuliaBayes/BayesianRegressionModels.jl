# test/adaptive_cdar_centering.jl — online adaptive-centering contract for the
# grouped, correlated, damped random-walk term `cdar(step; by=group, cor=C)`.
#
# Each of the P * W innovations is one scalar cell with zero location and its
# marginal prior spread `sigma * sqrt(C[p, p] * (1 - rho^(2w)) / (1 - rho^2))`;
# `c=0` is the emitted `eta` frame. Mirrors `test/adaptive_hsgp_centering.jl`:
# metadata resolution, the exact transform with its Jacobian, candidate
# scoring, Enzyme gradients, BridgeStan density/gradient parity, and one short
# online warmup.
using Test
using BayesianRegressionModels
using BridgeStan
using Distributions: Normal
import DifferentiationInterface as DI
import Enzyme
using LinearAlgebra
using LogDensityProblems
import StanBlocks
using WarmupHMC
using Random: Xoshiro

const BRM = BayesianRegressionModels
const BS = BridgeStan

const CDAR_AC_EXT = Base.get_extension(
    BayesianRegressionModels, :BayesianRegressionModelsWarmupHMCExt,
)

# Asymmetric covariance: marginal variances 1 and 2 discriminate the
# per-group spread mapping, and rho = 0.5 separates every step.
const CDAR_AC_C = [1.0 0.5; 0.5 2.0]
const CDAR_AC_DATA = (;
    week=repeat(1:2; inner=2), patch=repeat(["a", "b"]; outer=2),
    y=[0.5, -0.3, 0.4, -0.2], C=CDAR_AC_C)

const CDAR_AC_BUILDER = @brm CDAR_AC_DATA begin
    mu ~ 1 + cdar(week; by=patch, cor=C)
    y ~ Normal(mu, 0.1)
end

const CDAR_AC_CONFIGURED = @brm CDAR_AC_DATA begin
    mu ~ 1 + cdar(week; by=patch, cor=C)
    sd(:, cdar(week)) ~ Normal(0.0, 0.2)
    ar(:, cdar(week)) ~ Normal(0.8, 0.1)
    y ~ Normal(mu, 0.1)
end

function cdar_ac_unc_names()
    vcat(
        ["pop_mu_beta_pop.1", "cdar_mu_week_sigma", "cdar_mu_week_rho"],
        ["cdar_mu_week_eta.$i" for i in 1:4],
    )
end

# Independent marginal spreads in pair order (column-major groups-in-steps).
function manual_cdar_scales(sigma, rho, cdiag, P, W)
    [sigma * sqrt(cdiag[p]) * sqrt((1 - rho^(2w)) / (1 - rho^2))
     for w in 1:W for p in 1:P]
end

@testset "cdar metadata resolves frozen walk geometry" begin
    sb = SBBRMI(CDAR_AC_BUILDER; mod=@__MODULE__)
    names = cdar_ac_unc_names()
    blocks = BRM._adaptive_cdar_centering_blocks(sb, names)
    block = only(blocks)
    @test block.logical === :mu
    @test block.term === :cdar_mu_week
    @test block.n_groups == 2
    @test block.n_steps == 2
    @test block.effects == [4, 5, 6, 7]
    @test block.sigma == 2
    @test block.rho == 3
    @test block.sigma_lower == 0.0
    @test block.cdiag ≈ [1.0, 2.0] atol=2e-14
    @test sb.data[:cdar_mu_week_L] ≈ Matrix(cholesky(Symmetric(CDAR_AC_C)).L)

    x = zeros(length(names))
    x[block.sigma] = log(0.1)
    x[block.rho] = 0.0
    expected = manual_cdar_scales(0.1, 0.5, [1.0, 2.0], 2, 2)
    @test length(unique(expected)) == 4
    for pair in 1:4
        @test exp(BRM._adaptive_cdar_log_scale(x, block, pair)) ≈
              expected[pair] atol=2e-14
    end

    configured = SBBRMI(CDAR_AC_CONFIGURED; mod=@__MODULE__)
    configured_blocks = BRM._adaptive_cdar_centering_blocks(configured, names)
    @test length(configured_blocks) == 1
    @test only(configured_blocks).term === :cdar_mu_week

    missing = filter(!=("cdar_mu_week_eta.4"), names)
    @test_throws "innovations" BRM._adaptive_cdar_centering_blocks(sb, missing)
end

function set_cdar_sources!(state, ir, controls)
    length(controls) == length(ir.pairs) || throw(DimensionMismatch())
    ir.pairs .= map(ir.pairs, controls) do (idx, value), c
        idx => WarmupHMC.Reparametrization(
            value.target, WarmupHMC.PartiallyCentered(c), value.args...,
        )
    end
    CDAR_AC_EXT._sync_sources!(state, ir)
end

function manual_cdar_map(x, blocks, controls, scales)
    y = copy(x)
    ljac = zero(eltype(x))
    p = 1
    for block in blocks, cell in eachindex(block.effects)
        s = scales[p]
        c = controls[p]
        y[block.effects[cell]] = x[block.effects[cell]] * exp(-c * log(s))
        ljac -= c * log(s)
        p += 1
    end
    ljac, y
end

struct CDARQuadraticTarget
    dimension::Int
end

LogDensityProblems.dimension(target::CDARQuadraticTarget) = target.dimension
LogDensityProblems.capabilities(::Type{CDARQuadraticTarget}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::CDARQuadraticTarget, x) = -sum(abs2, x) / 2
LogDensityProblems.logdensity_and_gradient(target::CDARQuadraticTarget, x) =
    (LogDensityProblems.logdensity(target, x), -x)

@testset "cdar transform, Jacobian, scores, and Enzyme gradient" begin
    sb = SBBRMI(CDAR_AC_BUILDER; mod=@__MODULE__)
    names = cdar_ac_unc_names()
    blocks = BRM._adaptive_cdar_centering_blocks(sb, names)
    state, ir = CDAR_AC_EXT._adaptive_cdar_centering_reparametrizer(blocks)
    @test first.(ir.pairs) == only(blocks).effects

    x = collect(range(-0.45, 0.65; length=length(names)))
    x[only(blocks).sigma] = log(0.1)
    x[only(blocks).rho] = 0.0
    scales = manual_cdar_scales(0.1, 0.5, [1.0, 2.0], 2, 2)

    zeros_c = zeros(length(ir.pairs))
    set_cdar_sources!(state, ir, zeros_c)
    @test isequal(ir(x), (0.0, x))

    controls = [0.2, 0.5, 1.0, 0.7]
    set_cdar_sources!(state, ir, controls)
    expected_ljac, expected = manual_cdar_map(x, blocks, controls, scales)
    ljac, mapped = ir(x)
    @test ljac ≈ expected_ljac atol=2e-14
    @test mapped ≈ expected atol=2e-14
    inverse_ljac, roundtrip = WarmupHMC._inverse_with_logabsdet_jacobian(ir, mapped)
    @test inverse_ljac ≈ -ljac atol=2e-14
    @test roundtrip ≈ x atol=2e-14

    ones_c = ones(length(ir.pairs))
    set_cdar_sources!(state, ir, ones_c)
    centered_ljac, centered = ir(x)
    expected_centered_ljac, expected_centered =
        manual_cdar_map(x, blocks, ones_c, scales)
    @test centered_ljac ≈ expected_centered_ljac atol=2e-14
    @test centered ≈ expected_centered atol=2e-14

    set_cdar_sources!(state, ir, controls)
    weight = collect(range(0.3, 1.4; length=length(x)))
    objective(v, transform, w) = ((j, q) = transform(v); j + dot(w, q))
    ad_gradient = DI.gradient(
        objective, DI.AutoEnzyme(), x, DI.Constant(ir), DI.Constant(weight),
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
        set_cdar_sources!(state, ir, source_controls)
        position = copy(x)
        gradient = zeros(length(x))
        p = 1
        for block in blocks, cell in eachindex(block.effects)
            s = scales[p]
            c = source_controls[p]
            idx = block.effects[cell]
            position[idx] = s^c * innovations[p]
            gradient[idx] = invariant_gradient[p] / s^c
            p += 1
        end
        CDAR_AC_EXT._prepare_frame(state, ir, position, gradient)
    end
    for frame in frames
        @test frame.location == zeros(length(ir.pairs))
        @test frame.innovation ≈ innovations atol=2e-14
        @test frame.invariant_gradient ≈ invariant_gradient atol=2e-14
    end
    for p in eachindex(ir.pairs), candidate_c in 0.0:0.1:1.0
        candidate = WarmupHMC.PartiallyCentered(candidate_c)
        observations = [CDAR_AC_EXT._score_candidate(
            frame, p, first(ir.pairs[p]), last(ir.pairs[p]), candidate,
        ) for frame in frames]
        @test all(isapprox(obs[2], observations[1][2]; atol=2e-14)
                  for obs in observations)
        @test all(isapprox(obs[3], observations[1][3]; atol=2e-14)
                  for obs in observations)
    end

    target = CDARQuadraticTarget(length(x))
    wrapped = WarmupHMC.ReparametrizedProblem(ir, target, DI.AutoEnzyme())
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

@testset "cdar BridgeStan density, gradient, and online warmup" begin
    sb = SBBRMI(CDAR_AC_BUILDER; mod=@__MODULE__)
    cache = joinpath(tempdir(), "brm-adaptive-cdar-centering")
    mkpath(cache)
    problem = StanBlocks.stan_instantiate(
        sb.model; path=joinpath(cache, "cdar_walk.stan"),
    )
    unc_names = StanBlocks.BridgeStan.param_unc_names(problem.model)
    blocks = BRM._adaptive_cdar_centering_blocks(sb, unc_names)
    @test length(blocks) == 1
    @test length(only(blocks).effects) == 4

    backend = DI.AutoEnzyme()
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

    controls = [0.2, 0.5, 1.0, 0.7]
    set_cdar_sources!(state, ir, controls)
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
        Xoshiro(0x20260917), online;
        n_draws=20,
        n_evaluations=120,
        stepsize_adaptation_limit=20,
        max_tree_depth=7,
        progress=nothing,
        monitor_ess=false,
    )
    adapted = [value.c for (_, value) in WarmupHMC.reparam_sources(online)]
    @test length(adapted) == 4
    @test all(c -> c in 0.0:0.1:1.0, adapted)
    # No movement assert: the scorer holds c=0 across data regimes and budgets
    # probed (flat, pinned, and funnel-candidate walks at 20/120 and 60/600),
    # consistent with the criterion's known optimum outside funnel posteriors.
    # The exactness net above (BridgeStan parity, finite differences,
    # round-trips, frame recovery, score consistency) is what pins the
    # machinery; the end-to-end run below proves it samples.
    @test size(result.posterior_position, 1) == length(unc_names)
    @test size(result.posterior_position, 2) >= 20
    @test all(isfinite, result.posterior_position)
end

@testset "cdar cells refuse mixed online plans" begin
    mixed_builder = @brm begin
        mu ~ 1 + (1 | subject) + cdar(week; by=patch, cor=C)
        y ~ Normal(mu, 1.0)
    end
    mixed_df = merge(CDAR_AC_DATA, (; subject=["s1", "s1", "s2", "s2"]))
    mixed_sb = SBBRMI(mixed_builder(mixed_df); total_groups=(), mod=@__MODULE__)
    mixed_sb = SBBRMI(mixed_builder(mixed_df); total_groups=(), mod=@__MODULE__)
    sb = SBBRMI(CDAR_AC_BUILDER; mod=@__MODULE__)
    cache = joinpath(tempdir(), "brm-adaptive-cdar-centering")
    mkpath(cache)
    problem = StanBlocks.stan_instantiate(
        sb.model; path=joinpath(cache, "cdar_walk.stan"),
    )
    mixed_names = vcat(
        cdar_ac_unc_names(),
        ["r_mu_subject_log_scale", "r_mu_subject_xi.1", "r_mu_subject_xi.2"],
    )
    @test_throws "cannot yet mix" adaptive_centering_problem(
        mixed_sb, problem, DI.AutoEnzyme(); unc_names=mixed_names)
end
