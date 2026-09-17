include(joinpath(@__DIR__, "adaptive_centering_warmuphmc.jl"))

using Enzyme
using DifferentiationInterface: AutoEnzyme
using LogDensityProblems
using StanBlocks

function enzyme_gradient_stress(problem, initial; n=5_000)
    x = copy(initial)
    checksum = zero(eltype(x))
    for i in 1:n
        x[1] = initial[1] + 1e-6 * (i % 11)
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, x)
        checksum += lp + gradient[1]
    end
    checksum
end

struct AdaptiveCenteringQuadraticTarget
    dimension::Int
end

LogDensityProblems.dimension(target::AdaptiveCenteringQuadraticTarget) =
    target.dimension
LogDensityProblems.capabilities(::Type{AdaptiveCenteringQuadraticTarget}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::AdaptiveCenteringQuadraticTarget, x) =
    -sum(abs2, x) / 2
LogDensityProblems.logdensity_and_gradient(
    target::AdaptiveCenteringQuadraticTarget, x,
) = (LogDensityProblems.logdensity(target, x), -x)

function synthetic_adaptive_block(K, G)
    n_cholesky = K * (K - 1) ÷ 2
    ranef = BayesianRegressionModels.RanefBlock(
        :r_mu_subject, :ranef_correlated, :subject, nothing, nothing,
        string.(1:G), K, G, :r_mu_subject_z_flat, true, false,
    )
    AdaptiveCenteringBlock(
        ranef,
        0.0,
        reshape((n_cholesky + K + 1):(n_cholesky + K + K * G), K, G),
        collect(1:n_cholesky),
        collect((n_cholesky + 1):(n_cholesky + K)),
    )
end

# Accessor contract across the whole supported K range.
#
# Two DIFFERENT bars are asserted here, and the split is deliberate:
#
#   * the FORWARD map (`ir(x)`, every location/log_scale accessor value) is
#     bit-exact against the materialized semantic reference at every K.  The
#     arbitrary-K accessor walks Stan's stick recursion inline rather than
#     materializing `L` and `C`, but it preserves that graph's multiplication
#     association exactly, so no primal value moves.
#
#   * the GRADIENT is bit-exact only for K<=2, which is the hand-rolled Enzyme
#     reverse pass this change does not touch.  At K>=3 it agrees to ~1 ulp.
#     The materialized reference accumulates every `dC[i,j]` into an adjoint
#     MATRIX and only then reduces it into `dtau`/`dL` in one fixed pass; an
#     accessor that never builds `C` has no such buffer to defer into, so
#     Enzyme associates the same terms in a different order.  Float addition is
#     not associative, so this is structural, not a spelling defect: a per-row
#     `C`-buffer restructuring, a `tau` broadcast hoist, and a phased
#     `L`-then-`tau` build were all measured and all reproduce the same 1-ulp
#     difference.  Preserving the reverse order exactly REQUIRES materializing
#     the matrix, which is precisely the cost this accessor removes.
const ADAPTIVE_GRADIENT_ATOL = 1e-10

@testset "adaptive-centering accessors are exact, bounded, and GC-stable" begin
    @test !isnothing(Base.get_extension(
        BayesianRegressionModels,
        :BayesianRegressionModelsWarmupHMCEnzymeExt,
    ))
    backend = AutoEnzyme()
    for (K, G, allocation_limit) in (
        (1, 6, 4_096), (2, 18, 16_384), (3, 18, 96_000),
        (4, 12, 96_000), (8, 6, 128_000),
    )
        block = synthetic_adaptive_block(K, G)
        state, ir = AC_EXT._adaptive_centering_reparametrizer([block])
        # The K<=2 hand-rolled Enzyme reverse pass stays gated on this type
        # parameter; wider blocks keep the generic DifferentiationInterface path.
        @test (state isa AC_EXT.BRMAdaptiveCenteringState{true}) == (K <= 2)
        controls = collect(range(0.17, 0.83, length=K * G))
        set_adaptive_sources!(state, ir, controls)
        reference_ir = materialized_reparametrizer(state, ir)
        x = collect(range(-0.63, 0.81, length=maximum(vec(block.effects))))
        n_free = length(block.cholesky_free)
        if n_free > 0
            # Distinct per-coordinate values: an all-equal fill cannot separate a
            # transposed or mis-offset row walk from a correct one.  The K=2
            # spelling is unchanged (one free coordinate, still -0.27).
            x[block.cholesky_free] .=
                [-0.27 + 0.31 * ((i - 1) % 5) for i in 1:n_free]
        end
        x[block.log_scales] .=
            K == 1 ? [-0.31] : collect(range(-0.31, 0.44, length=K))
        target = AdaptiveCenteringQuadraticTarget(length(x))
        wrapped = WarmupHMC.ReparametrizedProblem(ir, target, backend)
        reference = WarmupHMC.ReparametrizedProblem(reference_ir, target, backend)

        # Forward map: bit-exact at every K, including the log-Jacobian.
        ljac, y = ir(x)
        reference_ljac, reference_y = reference_ir(x)
        @test isequal(ljac, reference_ljac)
        @test isequal(y, reference_y)
        for p in eachindex(ir.pairs), a in 1:2
            @test isequal(
                ir.pairs[p].second.args[a](x),
                reference_ir.pairs[p].second.args[a](x),
            )
        end

        actual = LogDensityProblems.logdensity_and_gradient(wrapped, x)
        expected = LogDensityProblems.logdensity_and_gradient(reference, x)
        @test isequal(actual[1], expected[1])
        if K <= 2
            @test isequal(actual[2], expected[2])
        else
            @test actual[2] ≈ expected[2] atol=ADAPTIVE_GRADIENT_ATOL rtol=ADAPTIVE_GRADIENT_ATOL
        end

        allocated = @allocated LogDensityProblems.logdensity_and_gradient(wrapped, x)
        reference_allocated =
            @allocated LogDensityProblems.logdensity_and_gradient(reference, x)
        @test allocated <= allocation_limit
        @test allocated < reference_allocated
        @test isfinite(enzyme_gradient_stress(wrapped, x))
    end
end

@testset "two-term BridgeStan target is bit-exact" begin
    sb = SBBRMI(slope_builder(df); total_groups=(), mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model)
    unc_names = StanBlocks.BridgeStan.param_unc_names(problem.model)
    block = only(adaptive_centering_blocks(sb, unc_names))
    @test block.ranef.n_terms == 2

    state, ir = AC_EXT._adaptive_centering_reparametrizer([block])
    controls = collect(range(0.17, 0.83, length=length(block.effects)))
    set_adaptive_sources!(state, ir, controls)
    materialized_ir = materialized_reparametrizer(state, ir)

    x = zeros(length(unc_names))
    x[only(block.cholesky_free)] = -0.27
    x[block.log_scales] .= [-0.31, 0.44]
    x[vec(block.effects)] .=
        collect(range(-0.63, 0.81, length=length(block.effects)))
    backend = AutoEnzyme()
    wrapped = WarmupHMC.ReparametrizedProblem(ir, problem, backend)
    materialized = WarmupHMC.ReparametrizedProblem(
        materialized_ir, problem, backend,
    )

    actual = LogDensityProblems.logdensity_and_gradient(wrapped, x)
    expected = LogDensityProblems.logdensity_and_gradient(materialized, x)
    @test isequal(actual, expected)
end

@testset "BridgeStan target differentiates through the mixed correlated frame" begin
    sb = SBBRMI(builder(df); total_groups=(), mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model)
    unc_names = StanBlocks.BridgeStan.param_unc_names(problem.model)
    block = only(adaptive_centering_blocks(sb, unc_names))
    state, ir = AC_EXT._adaptive_centering_reparametrizer([block])
    controls = [0.2, 0.6, 0.9, 0.8, 0.1, 0.5]
    set_adaptive_sources!(state, ir, controls)

    x = zeros(length(unc_names))
    x[block.cholesky_free] .= [0.2, -0.4, 0.7]
    x[block.log_scales] .= log.([0.5, 1.5, 3.0])
    x[vec(block.effects)] .=
        collect(range(-0.8, 0.9, length=length(block.effects)))
    backend = AutoEnzyme()
    reparametrized = WarmupHMC.ReparametrizedProblem(ir, problem, backend)
    materialized_ir = materialized_reparametrizer(state, ir)
    materialized = WarmupHMC.ReparametrizedProblem(
        materialized_ir, problem, backend,
    )

    lp, gradient = LogDensityProblems.logdensity_and_gradient(reparametrized, x)
    materialized_lp, materialized_gradient =
        LogDensityProblems.logdensity_and_gradient(materialized, x)
    @test isequal(lp, materialized_lp)
    # K=3 is on the arbitrary-K accessor, so the gradient bar here is ~1 ulp, not
    # `isequal` — see the accessor contract above the synthetic testset.  The
    # forward map below is still asserted bit-exact.
    @test gradient ≈ materialized_gradient atol=ADAPTIVE_GRADIENT_ATOL rtol=ADAPTIVE_GRADIENT_ATOL
    ljac, model_position = ir(x)
    materialized_ljac, materialized_position = materialized_ir(x)
    @test isequal(ljac, materialized_ljac)
    @test isequal(model_position, materialized_position)
    inner_lp, _ = LogDensityProblems.logdensity_and_gradient(problem, model_position)
    @test lp ≈ ljac + inner_lp atol=2e-12
    @test isfinite(lp)
    @test all(isfinite, gradient)

    step = 1e-5
    finite_difference = [begin
        plus, minus = copy(x), copy(x)
        plus[i] += step
        minus[i] -= step
        (LogDensityProblems.logdensity(reparametrized, plus) -
         LogDensityProblems.logdensity(reparametrized, minus)) / (2step)
    end for i in eachindex(x)]
    @test gradient ≈ finite_difference atol=2e-5 rtol=2e-5

    # One Enzyme gradient stayed green after bd9ca2d, but sustained calls through
    # its recursive K<=4 accessor corrupted Julia's GC after about 1,750 calls.
    # Keep this above the measured threshold so that process-level failure cannot
    # hide behind the otherwise-correct single-gradient assertion.
    @test isfinite(enzyme_gradient_stress(reparametrized, x))

    wrapped = adaptive_centering_problem(sb, problem, backend)
    wrapped_ir = WarmupHMC.reparametrizer(wrapped)
    wrapped_state = wrapped_ir.pairs[1][2].args[1].state
    plain = LogDensityProblems.logdensity_and_gradient(problem, zeros(length(x)))
    adaptive =
        LogDensityProblems.logdensity_and_gradient(wrapped, zeros(length(x)))
    @test adaptive[1] ≈ plain[1] atol=2e-12
    @test adaptive[2] ≈ plain[2] atol=2e-12
    @test WarmupHMC.candidate_scoring_plan(wrapped) isa
          WarmupHMC.CandidateScoringPlan

    restored = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    wrapped_state.sources .= -1.0
    WarmupHMC.restore_reparam_sources!(
        wrapped,
        [idx => WarmupHMC.PartiallyCentered(c)
         for ((idx, _), c) in zip(wrapped_ir.pairs, restored)],
    )
    @test [value.source.c for (_, value) in wrapped_ir.pairs] == restored
    @test wrapped_state.sources == restored

    copied = deepcopy(wrapped)
    copied_ir = WarmupHMC.reparametrizer(copied)
    copied_state = copied_ir.pairs[1][2].args[1].state
    @test copied_state !== wrapped_state
    copied_state.sources .= -1.0
    WarmupHMC.restore_reparam_sources!(
        copied,
        [idx => WarmupHMC.PartiallyCentered(0.0) for (idx, _) in copied_ir.pairs],
    )
    @test copied_state.sources == zeros(length(restored))
    @test wrapped_state.sources == restored
end

@testset "BridgeStan scalar random intercept adapts through the public wrapper" begin
    sb = SBBRMI(intercept_builder(df); total_groups=(), mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model)
    unc_names = StanBlocks.BridgeStan.param_unc_names(problem.model)
    block = only(adaptive_centering_blocks(sb, unc_names))
    @test block.ranef.n_terms == 1
    @test isempty(block.cholesky_free)

    backend = AutoEnzyme()
    wrapped = adaptive_centering_problem(sb, problem, backend)
    ir = WarmupHMC.reparametrizer(wrapped)
    state = ir.pairs[1][2].args[1].state
    controls = collect(range(0.2, 0.8, length=block.ranef.n_groups))
    set_adaptive_sources!(state, ir, controls)

    x = zeros(length(unc_names))
    x[only(block.log_scales)] = log(1.6)
    x[vec(block.effects)] .= collect(range(-0.5, 0.7, length=length(block.effects)))
    lp, gradient = LogDensityProblems.logdensity_and_gradient(wrapped, x)
    @test isfinite(lp)
    @test all(isfinite, gradient)
    @test isfinite(enzyme_gradient_stress(wrapped, x))

    step = 1e-5
    finite_difference = [begin
        plus, minus = copy(x), copy(x)
        plus[i] += step
        minus[i] -= step
        (LogDensityProblems.logdensity(wrapped, plus) -
         LogDensityProblems.logdensity(wrapped, minus)) / (2step)
    end for i in eachindex(x)]
    @test gradient ≈ finite_difference atol=2e-5 rtol=2e-5

    ljac, model_position = ir(x)
    inverse_ljac, roundtrip = WarmupHMC._inverse_with_logabsdet_jacobian(
        ir, model_position,
    )
    @test inverse_ljac ≈ -ljac atol=2e-12
    @test roundtrip ≈ x atol=2e-12
end

const JOINT_HSGP_BUILDER = @brm begin
    loc ~ 1 + x + hsgp(x; k=3, by=g) + (1 + x | subject)
    y ~ Normal(loc, 1)
end

function joint_hsgp_df()
    x = collect(range(-1.0, 1.0; length=8))
    y = sin.(x)
    g = repeat(["a", "b"], inner=4)
    subject = repeat([11, 12], inner=4)
    (; x, y, g, subject)
end

@testset "joint ordinary+HSGP wrapper adapts a mixed BridgeStan model" begin
    sb = SBBRMI(JOINT_HSGP_BUILDER(joint_hsgp_df()); total_groups=(),
                mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model)
    unc_names = StanBlocks.BridgeStan.param_unc_names(problem.model)
    blocks = adaptive_centering_blocks(sb, unc_names)
    hsgp_blocks = BRM._adaptive_hsgp_centering_blocks(sb, unc_names)
    @test length(blocks) == 1
    @test length(hsgp_blocks) == 2
    flat_pos = [findfirst(==("zflat_hsgpw_x_g.$i"), unc_names) for i in 1:6]
    @test all(!isnothing, flat_pos)
    @test hsgp_blocks[1].effects == flat_pos[1:3]
    @test hsgp_blocks[2].effects == flat_pos[4:6]
    @test hsgp_blocks[1].length_scales == hsgp_blocks[2].length_scales
    @test hsgp_blocks[1].sd == hsgp_blocks[2].sd

    # The HSGP-only reparametrizer enumerates every group level's cells.
    _, hir = AC_EXT._adaptive_hsgp_centering_reparametrizer(hsgp_blocks)
    @test first.(hir.pairs) == flat_pos

    # The public joint wrapper (previously: mixing error) orders ordinary
    # cells first, then HSGP cells.
    backend = AutoEnzyme()
    wrapped = adaptive_centering_problem(sb, problem, backend)
    wir = WarmupHMC.reparametrizer(wrapped)
    ranef_cells = vcat(vec(blocks[1].effects), blocks[1].cholesky_free,
                       blocks[1].log_scales)
    @test length(wir.pairs) == length(ranef_cells) + 6
    @test first.(wir.pairs) == vcat(ranef_cells, flat_pos)
    @test WarmupHMC.candidate_scoring_plan(wrapped) isa
          WarmupHMC.CandidateScoringPlan

    x = zeros(length(unc_names))
    x[blocks[1].cholesky_free] .= -0.27
    x[blocks[1].log_scales] .= [-0.31, 0.44]
    x[vec(blocks[1].effects)] .= collect(range(-0.63, 0.81, length=4))
    x[hsgp_blocks[1].length_scales] .= -0.2
    x[hsgp_blocks[1].sd] = -0.4
    x[flat_pos] .= collect(range(-0.5, 0.7, length=6))
    plain = LogDensityProblems.logdensity_and_gradient(problem, x)
    adaptive = LogDensityProblems.logdensity_and_gradient(wrapped, x)
    @test adaptive[1] ≈ plain[1] atol=2e-12
    @test adaptive[2] ≈ plain[2] atol=2e-12

    controls = collect(range(0.15, 0.85, length=length(wir.pairs)))
    wir.pairs .= map(wir.pairs, controls) do (idx, value), c
        idx => WarmupHMC.Reparametrization(
            value.target, WarmupHMC.PartiallyCentered(c), value.args...)
    end
    wstate = wir.pairs[1][2].args[1].state
    AC_EXT._sync_sources!(wstate, wir)
    @test vcat(wstate.ranef.sources, wstate.hsgp.sources) == controls
    lp, gradient = LogDensityProblems.logdensity_and_gradient(wrapped, x)
    @test isfinite(lp)
    @test all(isfinite, gradient)
    ljac, model_position = wir(x)
    inner_lp, _ = LogDensityProblems.logdensity_and_gradient(
        problem, model_position)
    @test lp ≈ ljac + inner_lp atol=2e-12

    step = 1e-5
    finite_difference = [begin
        plus, minus = copy(x), copy(x)
        plus[i] += step
        minus[i] -= step
        (LogDensityProblems.logdensity(wrapped, plus) -
         LogDensityProblems.logdensity(wrapped, minus)) / (2step)
    end for i in eachindex(x)]
    @test gradient ≈ finite_difference atol=2e-5 rtol=2e-5

    restored = collect(range(0.1, 0.9, length=length(wir.pairs)))
    wstate.ranef.sources .= -1.0
    wstate.hsgp.sources .= -1.0
    WarmupHMC.restore_reparam_sources!(
        wrapped,
        [idx => WarmupHMC.PartiallyCentered(c)
         for ((idx, _), c) in zip(wir.pairs, restored)],
    )
    @test [value.source.c for (_, value) in wir.pairs] == restored
    @test vcat(wstate.ranef.sources, wstate.hsgp.sources) == restored

    # The joint accessors are new differentiated code: stress them past the
    # historical GC-corruption threshold like the family accessors.
    @test isfinite(enzyme_gradient_stress(wrapped, x; n=2000))
end
