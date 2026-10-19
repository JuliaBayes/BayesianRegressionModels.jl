# test/adaptive_multiaxis_hsgp_centering.jl — online adaptive centering of
# multi-axis squared-exponential HSGPs (snag adaptive-centeri-b777132d).
#
# Run: julia --project=test test/adaptive_multiaxis_hsgp_centering.jl
#
# `hsgp(x1, x2; k)` owns `prod(k)` basis weights and an `omega2` matrix with
# one squared-frequency column per axis. An isotropic term samples ONE length
# scale that the compiled spectrum broadcasts over every axis
# (`rep_vector(rho_iso, n_axes)`); `iso=false` samples one per axis. The
# adaptive wrapper used to pair length scales with `omega2` columns one-to-one,
# so an isotropic two-axis term refused the whole `adaptive_centering_problem`
# (and `hsgp_coordinate_draws` threw a `DimensionMismatch` on the same term).
#
# The wrapper's density identities hold for any deterministic scale, so they
# cannot show the cells read the right spectrum. Each case therefore pins the
# per-basis scale against BridgeStan's constrained hyperparameters, and the
# ungrouped cases compare the fully centered frame with the compiled
# `centeredness=1` twin: wrapping the non-centered model at c=1 must reproduce
# that model's density and gradient exactly when the cells' scales equal the
# compiled spectral weights.

using Test
using BayesianRegressionModels
using BridgeStan
using Distributions: Normal
import DifferentiationInterface as DI
import Enzyme
using LogDensityProblems
import StanBlocks
using WarmupHMC

const BRM = BayesianRegressionModels
const MULTIAXIS_AC_EXT = Base.get_extension(
    BayesianRegressionModels, :BayesianRegressionModelsWarmupHMCExt,
)
const MULTIAXIS_CACHE = mkpath(joinpath(tempdir(), "brm-adaptive-multiaxis-hsgp"))

multiaxis_instantiate(sb, name) = StanBlocks.stan_instantiate(
    sb.model; path=joinpath(MULTIAXIS_CACHE, name * ".stan"),
)

function multiaxis_constrained(problem, x, name)
    names = BridgeStan.param_names(problem.model)
    values = BridgeStan.param_constrain(problem.model, x)
    values[only(findall(==(name), names))]
end

# Separable squared-exponential log spectral weight of one basis row.
multiaxis_log_scale(sigma, rhos, omega2_row) =
    log(sigma) + sum(0.5 * log(rho * sqrt(2pi)) for rho in rhos) -
    0.25 * sum(rhos .^ 2 .* omega2_row)

function set_multiaxis_controls!(wrapped, controls)
    ir = WarmupHMC.reparametrizer(wrapped)
    length(controls) == length(ir.pairs) || throw(DimensionMismatch())
    ir.pairs .= map(ir.pairs, controls) do (idx, value), c
        idx => WarmupHMC.Reparametrization(
            value.target, WarmupHMC.PartiallyCentered(c), value.args...)
    end
    MULTIAXIS_AC_EXT._sync_sources!(ir.pairs[1][2].args[1].state, ir)
    ir
end

function check_multiaxis_identities(problem, wrapped, x)
    plain = LogDensityProblems.logdensity_and_gradient(problem, x)
    adaptive = LogDensityProblems.logdensity_and_gradient(wrapped, x)
    @test adaptive[1] ≈ plain[1] atol=2e-11
    @test adaptive[2] ≈ plain[2] atol=2e-11

    n = length(WarmupHMC.reparametrizer(wrapped).pairs)
    ir = set_multiaxis_controls!(wrapped, collect(range(0.15, 0.85; length=n)))
    lp, gradient = LogDensityProblems.logdensity_and_gradient(wrapped, x)
    @test isfinite(lp)
    @test all(isfinite, gradient)
    ljac, model_position = ir(x)
    @test lp ≈ ljac + LogDensityProblems.logdensity(problem, model_position) atol=2e-11
    step = 1e-5
    finite_difference = [begin
        plus, minus = copy(x), copy(x)
        plus[i] += step
        minus[i] -= step
        (LogDensityProblems.logdensity(wrapped, plus) -
         LogDensityProblems.logdensity(wrapped, minus)) / (2step)
    end for i in eachindex(x)]
    @test gradient ≈ finite_difference atol=3e-5 rtol=3e-5
    inverse_ljac, roundtrip = WarmupHMC._inverse_with_logabsdet_jacobian(
        ir, model_position)
    @test inverse_ljac ≈ -ljac atol=2e-12
    @test roundtrip ≈ x atol=2e-12
end

function multiaxis_df(; n=40, groups=8)
    x1 = collect(range(-1.0, 1.0; length=n))
    x2 = [sinpi(0.7t) + 0.3t^2 for t in x1]
    region = repeat(1:groups, outer=cld(n, groups))[1:n]
    z = collect(range(-0.5, 0.5; length=n))
    y = [0.4sinpi(a) + 0.3cospi(b) + 0.1c for (a, b, c) in zip(x1, x2, z)]
    g = repeat(["a", "b"], inner=cld(n, 2))[1:n]
    (; x1, x2, region, z, y, g)
end

# Two predictors that each carry an isotropic two-axis HSGP beside a shared
# correlated random-effect block: the reported model shape.
const MULTIAXIS_JOINT_BUILDER = @brm begin
    mu ~ 1 + z + hsgp(x1, x2; k=4) + (1 | blk | region)
    log(sigma) ~ 1 + hsgp(x1, x2; k=3) + (1 | blk | region)
    y ~ Normal(mu, sigma)
end

@testset "isotropic two-axis HSGPs adapt beside a shared block" begin
    sb = SBBRMI(MULTIAXIS_JOINT_BUILDER(multiaxis_df()); total_groups=(),
                mod=@__MODULE__)
    problem = multiaxis_instantiate(sb, "joint_iso")
    unc_names = BridgeStan.param_unc_names(problem.model)
    blocks = BRM._adaptive_hsgp_centering_blocks(sb, unc_names)
    @test length(blocks) == 2
    by_logical = Dict(b.logical => b for b in blocks)
    mu, sigma = by_logical[:mu], by_logical[:sigma]
    @test length(mu.effects) == 16
    @test length(sigma.effects) == 9
    @test length(mu.length_scales) == 1
    @test length(sigma.length_scales) == 1
    @test size(mu.omega2) == (16, 2)
    @test size(sigma.omega2) == (9, 2)
    @test mu.omega2 == sb.data[:omega2_hsgp_x1_x2]
    @test sigma.omega2 == sb.data[:omega2_hsgp_log_sigma_x1_x2]
    @test unc_names[only(mu.length_scales)] == "hsgp_x1_x2_rho_iso"
    @test unc_names[only(sigma.length_scales)] == "hsgp_log_sigma_x1_x2_rho_iso"

    x = collect(range(-0.6, 0.7; length=length(unc_names)))
    for (block, prefix) in ((mu, "hsgp_x1_x2"), (sigma, "hsgp_log_sigma_x1_x2"))
        rho = multiaxis_constrained(problem, x, prefix * "_rho_iso")
        sd = multiaxis_constrained(problem, x, prefix * "_sigma")
        for basis in eachindex(block.effects)
            @test BRM._adaptive_hsgp_log_scale(x, block, basis) ≈
                multiaxis_log_scale(sd, (rho, rho), block.omega2[basis, :]) atol=1e-12
        end
    end

    ranef = adaptive_centering_blocks(sb, unc_names)
    @test length(ranef) == 1
    wrapped = adaptive_centering_problem(sb, problem, DI.AutoEnzyme())
    ir = WarmupHMC.reparametrizer(wrapped)
    @test length(ir.pairs) == length(vec(only(ranef).effects)) + 16 + 9
    check_multiaxis_identities(problem, wrapped, x)
end

const MULTIAXIS_NCP = Dict(
    :iso => @brm(begin
        mu ~ 1 + hsgp(x1, x2; k=(3, 4))
        y ~ Normal(mu, 0.5)
    end),
    :aniso => @brm(begin
        mu ~ 1 + hsgp(x1, x2; k=(3, 4), iso=false)
        y ~ Normal(mu, 0.5)
    end),
)
const MULTIAXIS_CP = Dict(
    :iso => @brm(begin
        mu ~ 1 + hsgp(x1, x2; k=(3, 4), centeredness=1.0)
        y ~ Normal(mu, 0.5)
    end),
    :aniso => @brm(begin
        mu ~ 1 + hsgp(x1, x2; k=(3, 4), iso=false, centeredness=1.0)
        y ~ Normal(mu, 0.5)
    end),
)

@testset "the centered frame of a two-axis HSGP is the compiled c=1 model ($kind)" for kind in (:iso, :aniso)
    df = multiaxis_df()
    ncp_sb = SBBRMI(MULTIAXIS_NCP[kind](df); mod=@__MODULE__)
    cp_sb = SBBRMI(MULTIAXIS_CP[kind](df); mod=@__MODULE__)
    ncp = multiaxis_instantiate(ncp_sb, "ncp_$kind")
    cp = multiaxis_instantiate(cp_sb, "cp_$kind")
    ncp_names = BridgeStan.param_unc_names(ncp.model)
    cp_names = BridgeStan.param_unc_names(cp.model)
    block = only(BRM._adaptive_hsgp_centering_blocks(ncp_sb, ncp_names))
    @test size(block.omega2) == (12, 2)
    @test length(block.length_scales) == (kind === :iso ? 1 : 2)
    @test all(iszero, block.target_c)

    # The two programs differ only in the weight coordinates' names.
    cp_pos = Dict(n => i for (i, n) in enumerate(cp_names))
    perm = [cp_pos[replace(n, "beta_raw" => "beta_partial")] for n in ncp_names]
    @test sort(perm) == eachindex(cp_names)

    points = [collect(range(-0.6, 0.7; length=length(ncp_names))),
              collect(range(0.8, -0.4; length=length(ncp_names)))]
    wrapped = adaptive_centering_problem(ncp_sb, ncp, DI.AutoEnzyme())
    set_multiaxis_controls!(wrapped, ones(length(block.effects)))
    for x in points
        lp, gradient = LogDensityProblems.logdensity_and_gradient(wrapped, x)
        cp_x = similar(x)
        cp_x[perm] = x
        cp_lp, cp_gradient = LogDensityProblems.logdensity_and_gradient(cp, cp_x)
        @test lp ≈ cp_lp atol=1e-10 rtol=1e-12
        @test gradient ≈ cp_gradient[perm] atol=1e-9 rtol=1e-9
    end

    # The posterior HSGP coordinate reader shares the spectrum: one length-scale
    # column per sampled length scale, log scales equal to the adaptive cells'.
    constrained_names = BridgeStan.param_names(ncp.model)
    draws = permutedims(reduce(hcat,
        [BridgeStan.param_constrain(ncp.model, x) for x in points]))
    gp = hsgp_coordinate_draws(brm_descriptor(ncp_sb), draws, constrained_names;
                               predictor=:mu, term=block.term)
    @test size(gp.length_scales) == (2, length(block.length_scales))
    for (i, x) in enumerate(points), basis in eachindex(block.effects)
        @test gp.log_scales[i, basis] ≈
            BRM._adaptive_hsgp_log_scale(x, block, basis) atol=1e-12
    end
end

const MULTIAXIS_GROUPED_BUILDER = @brm begin
    loc ~ 1 + hsgp(x1, x2; k=3, by=g)
    y ~ Normal(loc, 0.5)
end

@testset "grouped isotropic two-axis HSGP cells share one length scale" begin
    sb = SBBRMI(MULTIAXIS_GROUPED_BUILDER(multiaxis_df()); mod=@__MODULE__)
    problem = multiaxis_instantiate(sb, "grouped_iso")
    unc_names = BridgeStan.param_unc_names(problem.model)
    blocks = BRM._adaptive_hsgp_centering_blocks(sb, unc_names)
    @test length(blocks) == 2
    @test all(b -> length(b.effects) == 9, blocks)
    @test all(b -> size(b.omega2) == (9, 2), blocks)
    @test blocks[1].length_scales == blocks[2].length_scales
    @test unc_names[only(blocks[1].length_scales)] == "hsgp_x1_x2_by_g_rho_iso"
    @test isempty(intersect(blocks[1].effects, blocks[2].effects))

    x = collect(range(-0.5, 0.6; length=length(unc_names)))
    rho = multiaxis_constrained(problem, x, "hsgp_x1_x2_by_g_rho_iso")
    sd = multiaxis_constrained(problem, x, "hsgp_x1_x2_by_g_sigma")
    for block in blocks, basis in eachindex(block.effects)
        @test BRM._adaptive_hsgp_log_scale(x, block, basis) ≈
            multiaxis_log_scale(sd, (rho, rho), block.omega2[basis, :]) atol=1e-12
    end

    wrapped = adaptive_centering_problem(sb, problem, DI.AutoEnzyme())
    @test length(WarmupHMC.reparametrizer(wrapped).pairs) == 18
    check_multiaxis_identities(problem, wrapped, x)
end
