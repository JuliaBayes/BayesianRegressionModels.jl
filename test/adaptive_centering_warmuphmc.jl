include(joinpath(@__DIR__, "adaptive_centering.jl"))

using WarmupHMC
using Enzyme
using DifferentiationInterface: AutoEnzyme, Constant
import DifferentiationInterface

const AC_EXT = Base.get_extension(
    BayesianRegressionModels, :BayesianRegressionModelsWarmupHMCExt,
)

accessor_allocations(f::F, x::X) where {F,X} = @allocated f(x)

function set_adaptive_sources!(state, ir, controls)
    length(controls) == length(ir.pairs) || throw(DimensionMismatch())
    ir.pairs .= map(ir.pairs, controls) do (idx, value), c
        idx => WarmupHMC.Reparametrization(
            value.target, WarmupHMC.PartiallyCentered(c), value.args...,
        )
    end
    AC_EXT._sync_sources!(state, ir)
end

function manual_adaptive_map(x, block, controls)
    y = copy(x)
    C = BayesianRegressionModels._adaptive_block_cholesky(x, block)
    ljac = zero(eltype(x))
    p = 1
    for g in 1:block.ranef.n_groups
        z = similar(C, block.ranef.n_terms)
        for k in 1:block.ranef.n_terms
            c = controls[p]
            m = sum(C[k, l] * z[l] for l in 1:k-1; init=zero(eltype(x)))
            s = C[k, k]
            u = x[block.effects[k, g]]
            z[k] = (u - c * m) / s^c
            y[block.effects[k, g]] =
                block.target_c * m + s^block.target_c * z[k]
            ljac += (block.target_c - c) * log(s)
            p += 1
        end
    end
    ljac, y
end

# Frozen reference for the pre-bd9ca2d accessor.  Keep this materialized form
# in the test: the optimized hot path must preserve its floating-point operation
# order, not merely agree to a tolerance.  WarmupHMC selects a discrete winner
# after every window, so a one-ulp accessor drift can redirect the rest of an
# otherwise deterministic run.
function materialized_block_location(x, state, pair_number)
    bi, k, g = AC_EXT._pair_location(state, pair_number)
    block = state.blocks[bi]
    C = BayesianRegressionModels._adaptive_block_cholesky(x, block)
    z = Vector{eltype(C)}(undef, k)
    m = zero(eltype(C))
    for i in 1:k
        m = zero(eltype(C))
        for l in 1:i-1
            m += C[i, l] * z[l]
        end
        if i < k
            p = state.pair_lookup[bi][i, g]
            c = state.sources[p]
            z[i] = (x[block.effects[i, g]] - c * m) / C[i, i]^c
        end
    end
    m
end

struct MaterializedAdaptiveCenteringArgument{S} <: Function
    state::S
    pair_number::Int
    kind::Symbol
end

function (arg::MaterializedAdaptiveCenteringArgument)(x)
    arg.kind === :location &&
        return materialized_block_location(x, arg.state, arg.pair_number)
    if arg.kind === :log_scale
        bi, k, _ = AC_EXT._pair_location(arg.state, arg.pair_number)
        C = BayesianRegressionModels._adaptive_block_cholesky(
            x, arg.state.blocks[bi],
        )
        return log(C[k, k])
    end
    error("unknown materialized adaptive-centering argument kind $(arg.kind)")
end

function materialized_reparametrizer(state, ir)
    pairs = map(enumerate(ir.pairs)) do (p, (idx, value))
        idx => WarmupHMC.Reparametrization(
            value.target,
            value.source,
            MaterializedAdaptiveCenteringArgument(state, p, :location),
            MaterializedAdaptiveCenteringArgument(state, p, :log_scale),
        )
    end
    WarmupHMC.IndexedReparametrization(pairs)
end

function source_observation(x, block, controls, innovations, invariant_gradient)
    position = copy(x)
    gradient = zeros(eltype(x), length(x))
    C = BayesianRegressionModels._adaptive_block_cholesky(x, block)
    p = 1
    for g in 1:block.ranef.n_groups
        A = zeros(eltype(x), block.ranef.n_terms, block.ranef.n_terms)
        for k in 1:block.ranef.n_terms
            c = controls[p]
            m = sum(C[k, l] * innovations[l, g] for l in 1:k-1;
                    init=zero(eltype(x)))
            s = C[k, k]
            position[block.effects[k, g]] = c * m + s^c * innovations[k, g]
            A[k, k] = s^c
            for l in 1:k-1
                A[k, l] = c * C[k, l]
            end
            p += 1
        end
        gradient[block.effects[:, g]] .=
            transpose(A) \ invariant_gradient[:, g]
    end
    position, gradient
end

@testset "scalar random intercept transform and scores are source invariant" begin
    names = [
        "r_mu_subject_log_scale",
        "r_mu_subject_xi.1",
        "r_mu_subject_xi.2",
    ]
    sb = SBBRMI(intercept_builder(df); total_groups=(), mod=@__MODULE__)
    block = only(adaptive_centering_blocks(sb, names))
    state, ir = AC_EXT._adaptive_centering_reparametrizer([block])
    @test first.(ir.pairs) == [2, 3]

    controls = [0.2, 0.8]
    set_adaptive_sources!(state, ir, controls)
    x = [log(1.7), -0.4, 0.9]
    expected_ljac, expected = manual_adaptive_map(x, block, controls)
    ljac, mapped = ir(x)
    @test ljac ≈ expected_ljac atol=1e-15
    @test mapped ≈ expected atol=1e-15
    inverse_ljac, roundtrip = WarmupHMC._inverse_with_logabsdet_jacobian(ir, mapped)
    @test inverse_ljac ≈ -ljac atol=1e-15
    @test roundtrip ≈ x atol=1e-15

    centered = SBBRMI(
        intercept_builder(df); total_groups=(), mod=@__MODULE__, centered_groups=[:subject],
    )
    centered_block = only(adaptive_centering_blocks(centered, names))
    centered_state, centered_ir =
        AC_EXT._adaptive_centering_reparametrizer([centered_block])
    set_adaptive_sources!(centered_state, centered_ir, controls)
    expected_centered_ljac, expected_centered =
        manual_adaptive_map(x, centered_block, controls)
    centered_ljac, centered_mapped = centered_ir(x)
    @test centered_ljac ≈ expected_centered_ljac atol=1e-15
    @test centered_mapped ≈ expected_centered atol=1e-15
    centered_inverse_ljac, centered_roundtrip =
        WarmupHMC._inverse_with_logabsdet_jacobian(centered_ir, centered_mapped)
    @test centered_inverse_ljac ≈ -centered_ljac atol=1e-15
    @test centered_roundtrip ≈ x atol=1e-15

    innovations = reshape([0.3, -0.8], 1, 2)
    invariant_gradient = reshape([-0.4, 0.9], 1, 2)
    frames = map((zeros(2), controls, ones(2))) do source_controls
        set_adaptive_sources!(state, ir, source_controls)
        position, gradient = source_observation(
            x, block, source_controls, innovations, invariant_gradient,
        )
        AC_EXT._prepare_frame(state, ir, position, gradient)
    end
    for frame in frames
        @test frame.location == zeros(2)
        @test frame.innovation ≈ vec(innovations) atol=2e-15
        @test frame.invariant_gradient ≈ vec(invariant_gradient) atol=2e-15
    end
    for p in eachindex(ir.pairs), t in 0.0:0.1:1.0
        candidate = WarmupHMC.PartiallyCentered(t)
        observations = [AC_EXT._score_candidate(
            frame, p, first(ir.pairs[p]), last(ir.pairs[p]), candidate,
        ) for frame in frames]
        @test all(isapprox(obs[2], observations[1][2]; atol=2e-15)
                  for obs in observations)
        @test all(isapprox(obs[3], observations[1][3]; atol=2e-15)
                  for obs in observations)
    end
end

@testset "WarmupHMC correlated transform is exact at both BRM endpoints" begin
    @test !isnothing(AC_EXT)
    sb = SBBRMI(builder(df); total_groups=(), mod=@__MODULE__)
    block = only(adaptive_centering_blocks(sb, UNC_NONCENTERED))
    state, ir = AC_EXT._adaptive_centering_reparametrizer([block])
    @test first.(ir.pairs) == vec(block.effects)

    controls = [0.2, 0.6, 0.9, 0.8, 0.1, 0.5]
    set_adaptive_sources!(state, ir, controls)
    x = collect(range(-0.7, 0.9, length=length(UNC_NONCENTERED)))
    x[block.cholesky_free] .= [0.2, -0.4, 0.7]
    x[block.log_scales] .= log.([0.5, 1.5, 3.0])

    # The optimized path keeps the legacy matrix graph but elides its indexed
    # input copies and temporary scale vector.
    location = ir.pairs[end][2].args[1]
    log_scale = ir.pairs[end][2].args[2]
    reference_ir = materialized_reparametrizer(state, ir)
    reference_location = reference_ir.pairs[end][2].args[1]
    reference_log_scale = reference_ir.pairs[end][2].args[2]
    location(x)
    log_scale(x)
    reference_location(x)
    reference_log_scale(x)
    @test accessor_allocations(location, x) <
          accessor_allocations(reference_location, x)
    @test accessor_allocations(log_scale, x) <
          accessor_allocations(reference_log_scale, x)

    expected_ljac, expected = manual_adaptive_map(x, block, controls)
    ljac, mapped = ir(x)
    @test ljac ≈ expected_ljac atol=1e-13
    @test mapped ≈ expected atol=1e-13
    inverse_ljac, roundtrip = WarmupHMC._inverse_with_logabsdet_jacobian(ir, mapped)
    @test inverse_ljac ≈ -ljac atol=1e-13
    @test roundtrip ≈ x atol=1e-13

    # Differentiate with the backend BRM actually ships, against a reference that
    # cannot be wrong in the same direction: central differences, not a second AD
    # engine.  `ir` and `weight` ride as `Constant` contexts rather than captures
    # because `ir` holds a mutable `BRMAdaptiveCenteringState`; closing over it is
    # what raised `EnzymeMutabilityException`, and contexts are the fix — not a
    # whole-backend `function_annotation=Enzyme.Const`.  The tolerance answers to
    # the finite-difference floor (~eps/step at step=1e-6), not to Enzyme, so a
    # trip here is a real gradient error: Richardson-extrapolate, do not widen it.
    weight = collect(range(0.3, 1.7, length=length(x)))
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
    @test ad_gradient ≈ finite_difference atol=2e-8 rtol=2e-8

    centered = SBBRMI(builder(df); total_groups=(), mod=@__MODULE__, centered_groups=[:subject])
    unc_centered = vcat(
        UNC_NONCENTERED[1:9],
        ["r_mu_subject_b.$g.$k" for g in 1:2 for k in 1:3],
    )
    centered_block = only(adaptive_centering_blocks(centered, unc_centered))
    centered_state, centered_ir =
        AC_EXT._adaptive_centering_reparametrizer([centered_block])
    set_adaptive_sources!(centered_state, centered_ir, controls)
    centered_ljac, centered_map = centered_ir(x)
    expected_centered_ljac, expected_centered =
        manual_adaptive_map(x, centered_block, controls)
    @test centered_ljac ≈ expected_centered_ljac atol=1e-13
    @test centered_map ≈ expected_centered atol=1e-13

    copied = deepcopy(ir)
    copied_state = copied.pairs[1][2].args[1].state
    @test copied_state !== state
    @test all(pair[2].args[1].state === copied_state for pair in copied.pairs)
    set_adaptive_sources!(copied_state, copied, zeros(length(controls)))
    @test state.sources == controls
    @test copied_state.sources == zeros(length(controls))
end

@testset "adaptive accessors preserve legacy K=1:4 arithmetic exactly" begin
    for K in 1:4
        G = 2
        n_cholesky = K * (K - 1) ÷ 2
        ranef = BayesianRegressionModels.RanefBlock(
            :r_mu_subject, :ranef_correlated, :subject, nothing, nothing,
            string.(1:G), K, G, :r_mu_subject_z_flat, true, false,
        )
        block = AdaptiveCenteringBlock(
            ranef,
            0.0,
            reshape(
                (n_cholesky + K + 1):(n_cholesky + K + K * G), K, G,
            ),
            collect(1:n_cholesky),
            collect((n_cholesky + 1):(n_cholesky + K)),
        )
        state, ir = AC_EXT._adaptive_centering_reparametrizer([block])
        x = collect(range(-0.6, 0.8, length=n_cholesky + K + K * G))
        source_controls = (
            zeros(K * G),
            collect(range(0.1, 0.9, length=K * G)),
            ones(K * G),
        )
        for controls in source_controls
            set_adaptive_sources!(state, ir, controls)
            reference_ir = materialized_reparametrizer(state, ir)

            for p in eachindex(ir.pairs)
                for arg in 1:2
                    actual = ir.pairs[p][2].args[arg](x)
                    expected = reference_ir.pairs[p][2].args[arg](x)
                    @test isequal(actual, expected)
                end
            end

            @test isequal(ir(x), reference_ir(x))

            # Those two `isequal`s are the whole invariant, and they are load
            # bearing: every accessor individually, then the composed transform,
            # bit-identical to the materialized reference.  Do not weaken either to
            # `≈`, and do not restore a gradient-vs-gradient comparison here — equal
            # primals from equal operation orders is strictly stronger than equal
            # derivatives, and reverse-mode accuracy is gated separately in
            # `adaptive_centering_bridgestan.jl` (exact for K<=2, ADAPTIVE_GRADIENT_
            # ATOL for K>=3).  An AD-vs-AD check here would only add a weaker
            # duplicate: this loop runs K in 1:4, where Enzyme reassociates.
        end
    end
end

@testset "four-term optimized and wide fallback accessors are exact" begin
    for K in (4, 5)
        G = 2
        n_cholesky = K * (K - 1) ÷ 2
        ranef = BayesianRegressionModels.RanefBlock(
            :r_mu_subject, :ranef_correlated, :subject, nothing, nothing,
            ["a", "b"], K, G, :r_mu_subject_z_flat, true, false,
        )
        block = AdaptiveCenteringBlock(
            ranef,
            0.0,
            reshape((n_cholesky + K + 1):(n_cholesky + K + K * G), K, G),
            collect(1:n_cholesky),
            collect((n_cholesky + 1):(n_cholesky + K)),
        )
        state, ir = AC_EXT._adaptive_centering_reparametrizer([block])
        controls = collect(range(0.1, 0.9, length=K * G))
        set_adaptive_sources!(state, ir, controls)
        x = collect(range(-0.6, 0.8, length=n_cholesky + K + K * G))

        expected_ljac, expected = manual_adaptive_map(x, block, controls)
        ljac, mapped = ir(x)
        @test ljac ≈ expected_ljac atol=2e-13
        @test mapped ≈ expected atol=2e-13
        inverse_ljac, roundtrip =
            WarmupHMC._inverse_with_logabsdet_jacobian(ir, mapped)
        @test inverse_ljac ≈ -ljac atol=2e-13
        @test roundtrip ≈ x atol=2e-13

        if K == 4
            location = ir.pairs[K][2].args[1]
            reference_ir = materialized_reparametrizer(state, ir)
            reference_location = reference_ir.pairs[K][2].args[1]
            location(x)
            reference_location(x)
            @test accessor_allocations(location, x) <
                  accessor_allocations(reference_location, x)
        end

    end
end

@testset "candidate scores remove the installed source controls" begin
    sb = SBBRMI(builder(df); total_groups=(), mod=@__MODULE__)
    block = only(adaptive_centering_blocks(sb, UNC_NONCENTERED))
    state, ir = AC_EXT._adaptive_centering_reparametrizer([block])
    x = collect(range(-0.7, 0.9, length=length(UNC_NONCENTERED)))
    x[block.cholesky_free] .= [0.2, -0.4, 0.7]
    x[block.log_scales] .= log.([0.5, 1.5, 3.0])
    innovations = [0.3 -0.8; 1.1 0.4; -0.6 1.3]
    invariant_gradient = [-0.4 0.9; 0.7 -1.2; 1.4 0.2]
    source_controls = (
        zeros(6),
        [0.2, 0.6, 0.9, 0.8, 0.1, 0.5],
        ones(6),
    )

    frames = map(source_controls) do controls
        set_adaptive_sources!(state, ir, controls)
        position, gradient = source_observation(
            x, block, controls, innovations, invariant_gradient,
        )
        AC_EXT._prepare_frame(state, ir, position, gradient)
    end
    for frame in frames
        @test frame.innovation ≈ vec(innovations) atol=2e-15
        @test frame.invariant_gradient ≈ vec(invariant_gradient) atol=2e-15
    end

    for p in eachindex(ir.pairs), t in 0.0:0.1:1.0
        candidate = WarmupHMC.PartiallyCentered(t)
        observations = [AC_EXT._score_candidate(
            frame, p, first(ir.pairs[p]), last(ir.pairs[p]), candidate,
        ) for frame in frames]
        @test all(isapprox(obs[2], observations[1][2]; atol=2e-15)
                  for obs in observations)
        @test all(isapprox(obs[3], observations[1][3]; atol=2e-15)
                  for obs in observations)
        # The source-dependent Jacobian offset is constant across candidates;
        # its difference from t=0 is therefore source invariant too.
        baselines = [AC_EXT._score_candidate(
            frame, p, first(ir.pairs[p]), last(ir.pairs[p]),
            WarmupHMC.PartiallyCentered(0.0),
        )[1] for frame in frames]
        @test all(isapprox(
            observations[i][1] - baselines[i],
            observations[1][1] - baselines[1]; atol=2e-15,
        ) for i in eachindex(observations))
    end
end
