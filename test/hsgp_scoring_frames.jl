using Test, BayesianRegressionModels, WarmupHMC, Statistics
import LogDensityProblems
import DifferentiationInterface as DI

const FRAME_BRM = BayesianRegressionModels
const FRAME_MODEL = @brm begin
    mu ~ hsgp(x; k=3, centeredness=c_mu)
    log(sigma) ~ hsgp(x; k=3, centeredness=c_sigma)
    y ~ Normal(mu, sigma)
end

struct FrameDimensionOnly
    dimension::Int
end
LogDensityProblems.dimension(p::FrameDimensionOnly) = p.dimension
LogDensityProblems.capabilities(::Type{FrameDimensionOnly}) =
    LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.logdensity(::FrameDimensionOnly, q) = -sum(abs2, q) / 2

@testset "HSGP candidate replay preserves the compiled source frame" begin
    n = 80
    x = collect(range(-1, 1; length=8))
    innovations = [sin(0.31i + b) + 0.02i for b in 1:6, i in 1:n]
    invariant_gradient = [-innovations[b, i] + 0.2cos(0.17i * b)
                          for b in 1:6, i in 1:n]
    baseline = nothing
    for (cm, cs) in ((0.0, 0.0), (1.0, 1.0),
                     ([0.0, 0.35, 1.0], [0.7, 0.2, 0.9]))
        sb = SBBRMI(FRAME_MODEL((; x, y=sin.(x), c_mu=cm, c_sigma=cs));
                    mod=@__MODULE__)
        wanted = vcat(FRAME_BRM._brm_hsgp_centeredness((; centeredness=cm), 3),
                      FRAME_BRM._brm_hsgp_centeredness((; centeredness=cs), 3))
        mu_field = all(iszero, wanted[1:3]) ? "beta_raw" : "beta_partial"
        sigma_field = all(iszero, wanted[4:6]) ? "beta_raw" : "beta_partial"
        names = vcat(
            ["hsgp_x_rho_iso", "hsgp_x_sigma"],
            ["hsgp_x_$mu_field.$b" for b in 1:3],
            ["hsgp_log_sigma_x_rho_iso", "hsgp_log_sigma_x_sigma"],
            ["hsgp_log_sigma_x_$sigma_field.$b" for b in 1:3])
        blocks = FRAME_BRM._adaptive_hsgp_centering_blocks(sb, names)
        @test vcat(getfield.(blocks, :target_c)...) == wanted
        adaptive = adaptive_centering_problem(
            sb, FrameDimensionOnly(length(names)), DI.AutoEnzyme(); unc_names=names)
        ir = WarmupHMC.reparametrizer(adaptive)
        @test [value.target.c for (_, value) in ir.pairs] == wanted
        @test [value.source.c for (_, value) in ir.pairs] == wanted
        positions, gradients = zeros(length(names), n), zeros(length(names), n)
        log_scales = zeros(6, n)
        for i in 1:n
            for (bi, block) in enumerate(blocks)
                positions[only(block.length_scales), i] = -1.5 + 0.1sin(0.3i + bi)
                positions[block.sd, i] = -0.4 + 0.2cos(0.21i + bi)
                for b in eachindex(block.effects)
                    p = 3(bi - 1) + b
                    log_scales[p, i] = FRAME_BRM._adaptive_hsgp_log_scale(
                        view(positions, :, i), block, b)
                    scale = exp(wanted[p] * log_scales[p, i])
                    positions[block.effects[b], i] = scale * innovations[p, i]
                    gradients[block.effects[b], i] = invariant_gradient[p, i] / scale
                end
            end
        end
        jac, mapped = ir(positions[:, 1])
        @test jac ≈ 0 atol=1e-14
        @test mapped ≈ positions[:, 1] atol=1e-14
        scored = candidate_scoring_losses(adaptive, positions, gradients)
        @test length(scored) == 66
        @test all(s -> s.groups == n && s.effective_n == n, scored)
        for s in scored
            p, c = s.pair_number, s.candidate
            scale = exp.(c .* log_scales[p, :])
            expected = cor(scale .* innovations[p, :], invariant_gradient[p, :] ./ scale)
            @test s.loss ≈ expected atol=2e-13 rtol=2e-13
        end
        if isnothing(baseline)
            baseline = getproperty.(scored, :loss)
        else
            @test getproperty.(scored, :loss) ≈ baseline atol=2e-13 rtol=2e-13
        end
    end
end
