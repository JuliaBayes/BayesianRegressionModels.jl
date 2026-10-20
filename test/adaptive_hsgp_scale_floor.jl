# test/adaptive_hsgp_scale_floor.jl — HSGP cells read a floored log-scale
# (snag adaptive-hsgp-ce-024ae486).
#
# Run: julia --project=test test/adaptive_hsgp_scale_floor.jl
#
# A long length scale drives a high-frequency basis weight's exact log-scale to
# `-rho^2 * omega2 / 4`; a reported two-axis field measured -473 at `rho = 11`.
# Every candidate frame `s^t * z` of such a cell then underflowed: from the
# first `t` whose variance left Float64's range the candidate losses were NaN,
# and a cell centred there had a source coordinate whose spread was exactly
# zero, so the adapted diagonal metric was singular. A cell now reads its exact
# log-scale above `exp(-110)` and a smooth floor at `exp(-150)` below, which is
# a change of coordinates only: the wrapper stays exact.

using Test
using BayesianRegressionModels
using Statistics
import DifferentiationInterface as DI
import Enzyme
using LogDensityProblems
using WarmupHMC

const FLOOR_BRM = BayesianRegressionModels
const FLOOR_AC_EXT = Base.get_extension(
    BayesianRegressionModels, :BayesianRegressionModelsWarmupHMCExt,
)

# A standard-normal inner density in the compiled (non-centred) frame. The
# wrapper's map and scoring read only the coordinates, so no Stan model is
# needed.
struct FloorGaussian
    dimension::Int
end
LogDensityProblems.dimension(p::FloorGaussian) = p.dimension
LogDensityProblems.capabilities(::Type{FloorGaussian}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::FloorGaussian, q) = -sum(abs2, q) / 2
LogDensityProblems.logdensity_and_gradient(p::FloorGaussian, q) =
    (LogDensityProblems.logdensity(p, q), -collect(q))

const FLOOR_MODEL = @brm begin
    mu ~ hsgp(x1, x2; k=6)
    y ~ Normal(mu, 1)
end

const FLOOR_K = 36
const FLOOR_NAMES = vcat(["hsgp_x1_x2_rho_iso", "hsgp_x1_x2_sigma"],
                         ["hsgp_x1_x2_beta_raw.$b" for b in 1:FLOOR_K])

# The exact separable squared-exponential log spectral weight of one basis row
# with the one shared (isotropic) length scale.
floor_exact_log_scale(sigma, rho, omega2_row) =
    log(sigma) + length(omega2_row) * 0.5 * log(rho * sqrt(2pi)) -
    0.25 * rho^2 * sum(omega2_row)

function floor_set_controls!(adaptive, c)
    ir = WarmupHMC.reparametrizer(adaptive)
    ir.pairs .= map(ir.pairs) do (idx, value)
        idx => WarmupHMC.Reparametrization(
            value.target, WarmupHMC.PartiallyCentered(c), value.args...)
    end
    FLOOR_AC_EXT._sync_sources!(ir.pairs[1][2].args[1].state, ir)
    ir
end

function floor_problem()
    n = 30
    x1 = collect(range(-1.0, 1.0; length=n))
    x2 = [sinpi(0.6t) for t in x1]
    sb = SBBRMI(FLOOR_MODEL((; x1, x2, y=0.3 .* x1)); mod=@__MODULE__)
    block = only(FLOOR_BRM._adaptive_hsgp_centering_blocks(sb, FLOOR_NAMES))
    adaptive = adaptive_centering_problem(
        sb, FloorGaussian(length(FLOOR_NAMES)), DI.AutoEnzyme();
        unc_names=FLOOR_NAMES)
    sb, block, adaptive
end

# Stan's inverse `lb_constrain` / `lub_constrain`.
floor_unconstrain(v, lower, upper) = isfinite(upper) ?
    log((v - lower) / (upper - v)) : log(v - lower)

# Unconstrained position with physical length scale `rho` and SD `sigma`.
function floor_position(block, rho, sigma, weights)
    x = zeros(length(FLOOR_NAMES))
    x[only(block.length_scales)] = floor_unconstrain(
        rho, only(block.length_scale_lower), only(block.length_scale_upper))
    x[block.sd] = floor_unconstrain(sigma, block.sd_lower, block.sd_upper)
    x[block.effects] .= weights
    x
end

@testset "the floor is the identity above exp(-110) and smooth below" begin
    f = FLOOR_BRM._adaptive_hsgp_floor
    for v in (3.0, 0.0, -5.0, -36.0, -109.5, nextfloat(-110.0))
        @test f(v) === v
    end
    @test f(-110.0) == -110.0
    grid = collect(range(-2000.0, -100.0; length=4001))
    values = f.(grid)
    @test all(>=(-150.0), values)
    @test issorted(values)
    @test f(-473.0) == -150.0
    @test f(-130.0) ≈ -150.0 + log1p(exp(20.0)) rtol=1e-15
    # Its derivative is the logistic of `v + 150` below `-110` and 1 above, so
    # the cell's map has a continuous gradient across both bounds.
    for v in (-400.0, -160.0, -150.0, -140.0, -120.0, -110.0, -100.0, 0.0)
        expected = v > -110 ? 1.0 : 1 / (1 + exp(-(v + 150)))
        @test DI.derivative(f, DI.AutoEnzyme(), v) ≈ expected atol=1e-15
    end
end

@testset "a long length scale keeps every HSGP candidate frame finite" begin
    _, block, adaptive = floor_problem()
    @test size(block.omega2) == (FLOOR_K, 2)
    sigma, rho = 0.8, 8.0
    exact = [floor_exact_log_scale(sigma, rho, block.omega2[b, :]) for b in 1:FLOOR_K]
    x0 = floor_position(block, rho, sigma, zeros(FLOOR_K))
    cell = [FLOOR_BRM._adaptive_hsgp_log_scale(x0, block, b) for b in 1:FLOOR_K]
    # The point spans every regime: exact, softened and floored weights.
    @test any(>(-110), exact) && any(e -> -150 < e < -110, exact) &&
          any(<(-745), exact)
    healthy = exact .> -110
    @test cell[healthy] ≈ exact[healthy] atol=1e-12
    @test all(>=(-150.0), cell)
    @test cell[.!healthy] ≈ FLOOR_BRM._adaptive_hsgp_floor.(exact[.!healthy]) atol=1e-12

    n = 80
    innovations = [sin(0.31i + 1.7b) + 0.02i for b in 1:FLOOR_K, i in 1:n]
    invariant_gradient = [-innovations[b, i] + 0.2cos(0.17i * b)
                          for b in 1:FLOOR_K, i in 1:n]
    positions = zeros(length(FLOOR_NAMES), n)
    gradients = zeros(length(FLOOR_NAMES), n)
    log_scales = zeros(FLOOR_K, n)
    for i in 1:n
        positions[only(block.length_scales), i] = x0[only(block.length_scales)] + 0.05sin(0.3i)
        positions[block.sd, i] = x0[block.sd] + 0.2cos(0.21i)
        for b in 1:FLOOR_K
            log_scales[b, i] = FLOOR_BRM._adaptive_hsgp_log_scale(
                view(positions, :, i), block, b)
        end
        # The compiled frame is non-centred (c = 0): source = innovation.
        positions[block.effects, i] .= innovations[:, i]
        gradients[block.effects, i] .= invariant_gradient[:, i]
    end
    scored = candidate_scoring_losses(adaptive, positions, gradients)
    @test length(scored) == 11 * FLOOR_K
    # Before the floor, a weight whose exact log-scale is below -745 had
    # `exp(log_scale) == 0`, and its losses were NaN from the first candidate.
    @test all(s -> s.loss isa Float64 && isfinite(s.loss), scored)
    for s in scored
        b, t = s.pair_number, s.candidate
        scale = exp.(t .* log_scales[b, :])
        expected = cor(scale .* innovations[b, :], invariant_gradient[b, :] ./ scale)
        @test s.loss ≈ expected atol=2e-13 rtol=2e-13
    end
end

@testset "partially centred cells on a floored scale keep a finite transform" begin
    _, block, adaptive = floor_problem()
    ir = floor_set_controls!(adaptive, 0.9)
    sigma, rho = 0.8, 11.0
    x0 = floor_position(block, rho, sigma, zeros(FLOOR_K))
    log_scales = [FLOOR_BRM._adaptive_hsgp_log_scale(x0, block, b) for b in 1:FLOOR_K]
    exact = [floor_exact_log_scale(sigma, rho, block.omega2[b, :]) for b in 1:FLOOR_K]
    @test any(<(-1000), exact)

    # Model-frame innovations z map to source coordinates s^0.9 * z. With the
    # exact scale those were exactly zero for every z, so their spread was the
    # singular metric entry; floored, they spread like z times a fixed factor.
    zs = [sin(0.7i + 0.3b) for b in 1:FLOOR_K, i in 1:50]
    sources = map(eachcol(zs)) do z
        y = floor_position(block, rho, sigma, z)
        ljac, x = WarmupHMC._inverse_with_logabsdet_jacobian(ir, y)
        @test isfinite(ljac)
        @test all(isfinite, x)
        x[block.effects]
    end
    spread = vec(var(reduce(hcat, sources); dims=2))
    @test all(>(0), spread)
    @test spread ≈ exp.(1.8 .* log_scales) .* vec(var(zs; dims=2)) rtol=1e-12

    # The wrapper is exact and differentiable there.
    x = WarmupHMC._inverse_with_logabsdet_jacobian(
        ir, floor_position(block, rho, sigma, zs[:, 1]))[2]
    lp, gradient = LogDensityProblems.logdensity_and_gradient(adaptive, x)
    @test isfinite(lp)
    @test all(isfinite, gradient)
    ljac, y = ir(x)
    @test lp ≈ ljac + LogDensityProblems.logdensity(FloorGaussian(length(x)), y) rtol=1e-12
    @test y[block.effects] ≈ zs[:, 1] rtol=1e-12
    step = 1e-6
    finite_difference = [begin
        plus, minus = copy(x), copy(x)
        h = step * max(1, abs(x[i]))
        plus[i] += h
        minus[i] -= h
        (LogDensityProblems.logdensity(adaptive, plus) -
         LogDensityProblems.logdensity(adaptive, minus)) / 2h
    end for i in (only(block.length_scales), block.sd)]
    @test gradient[[only(block.length_scales), block.sd]] ≈ finite_difference rtol=1e-5
end
