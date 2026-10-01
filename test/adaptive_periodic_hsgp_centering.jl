# test/adaptive_periodic_hsgp_centering.jl — periodic HSGP cells in online
# adaptive centering (snag adaptive-centeri-96a06b1f).
#
# Run: julia --project=test test/adaptive_periodic_hsgp_centering.jl
# (root-env deps only, so `julia --project=.` runs it too, minus the
# Turing-ext parity testset, which skips without the Turing extension.)
#
# A model mixing an ordinary random-effect block with a periodic HSGP used to
# refuse the whole `adaptive_centering_problem` wrapper: the HSGP resolver
# raised on any periodic term even when ordinary cells could adapt. Periodic
# terms now contribute one scalar cell per cosine/sine weight with the known
# per-harmonic Bessel spectrum, alongside ordinary and squared-exponential
# cells.

using Test
using BayesianRegressionModels
using Distributions: Exponential, Normal
using SpecialFunctions: besselix

const BRM = BayesianRegressionModels

periodic_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject) + hsgp(t; k=2, cov=:periodic, period=24.0)
    y ~ Normal(mu, sigma)
end

mixed_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject) + hsgp(t; k=2, cov=:periodic, period=24.0) +
        hsgp(x; k=3, c=1.5)
    y ~ Normal(mu, sigma)
end

periodic_df = (;
    subject=repeat([11, 12], inner=6),
    x=collect(range(-1.0, 1.0; length=12)),
    t=collect(range(0.0, 20.0; length=12)),
    y=zeros(12),
)

const UNC_PERIODIC = [
    "sigma",
    "pop_mu_beta_pop.1", "pop_mu_beta_pop.2",
    "r_mu_subject_log_scale",
    "r_mu_subject_xi.1", "r_mu_subject_xi.2",
    "hsgp_t_rho_iso", "hsgp_t_sigma",
    "hsgp_t_beta_raw.1", "hsgp_t_beta_raw.2",
    "hsgp_t_beta_raw.3", "hsgp_t_beta_raw.4",
]

const UNC_MIXED = [
    "sigma",
    "pop_mu_beta_pop.1", "pop_mu_beta_pop.2",
    "hsgp_t_rho_iso", "hsgp_t_sigma",
    "hsgp_t_beta_raw.1", "hsgp_t_beta_raw.2",
    "hsgp_t_beta_raw.3", "hsgp_t_beta_raw.4",
    "hsgp_x_rho_iso", "hsgp_x_sigma",
    "hsgp_x_beta_raw.1", "hsgp_x_beta_raw.2", "hsgp_x_beta_raw.3",
    "r_mu_subject_log_scale",
    "r_mu_subject_xi.1", "r_mu_subject_xi.2",
]

@testset "periodic HSGP block metadata resolves" begin
    sb = SBBRMI(periodic_builder(periodic_df); total_groups=(), mod=@__MODULE__)
    block = only(BRM._adaptive_hsgp_centering_blocks(sb, UNC_PERIODIC))
    @test block.logical === :mu
    @test block.term === :hsgp_t
    # The compiled periodic emission is always fully non-centered.
    @test block.target_c == zeros(4)
    @test block.effects == [9, 10, 11, 12]
    @test block.length_scales == [7]
    @test block.length_scale_lower == [sb.data[:rho_lower_hsgp_t]]
    @test block.sd == 8
    @test block.sd_lower == 0.0
    @test block.harmonics == [1.0, 2.0, 1.0, 2.0]
    @test isempty(block.omega2)
    @test occursin("periodic", sprint(show, block))
end

@testset "periodic log scales match the Bessel spectrum" begin
    sb = SBBRMI(periodic_builder(periodic_df); total_groups=(), mod=@__MODULE__)
    block = only(BRM._adaptive_hsgp_centering_blocks(sb, UNC_PERIODIC))
    rho_lower = only(block.length_scale_lower)
    # Fixed unconstrained draw with known physical scales.
    x = zeros(length(UNC_PERIODIC))
    x[block.sd] = log(0.7)
    x[only(block.length_scales)] = log(51.0 - rho_lower)
    sigma = 0.7
    rho = 51.0
    a = inv(rho^2)
    for (basis, j) in enumerate(block.harmonics)
        # Independent rebuild of `s_j = sigma * sqrt(2 * exp(-a) * I_j(a))`:
        # `besselix` is the scaled `exp(-a) * I_j(a)`, so the Stan emission's
        # `-a/2 + log(I_j(a))/2` pair collapses to `log(besselix)/2`.
        expected = log(sigma) + (log(2) + log(besselix(j, a))) / 2
        @test BRM._adaptive_hsgp_log_scale(x, block, basis) ≈ expected atol=1e-12
    end
    # Cosine/sine pairs share one harmonic, so their scales agree.
    @test BRM._adaptive_hsgp_log_scale(x, block, 1) ==
        BRM._adaptive_hsgp_log_scale(x, block, 3)
    @test BRM._adaptive_hsgp_log_scale(x, block, 2) ==
        BRM._adaptive_hsgp_log_scale(x, block, 4)
end

@testset "mixed ordinary, exp-quad, and periodic cells resolve jointly" begin
    sb = SBBRMI(mixed_builder(periodic_df); total_groups=(), mod=@__MODULE__)
    # The snag's shape: this mixed model refused the whole wrapper before the
    # fix. All three resolvers now return their cells together.
    ordinary = adaptive_centering_blocks(sb, UNC_MIXED)
    @test length(ordinary) == 1
    hsgp = BRM._adaptive_hsgp_centering_blocks(sb, UNC_MIXED)
    @test length(hsgp) == 2
    by_term = Dict(b.term => b for b in hsgp)
    @test sort!(collect(keys(by_term))) == [:hsgp_t, :hsgp_x]
    periodic, exp_quad = by_term[:hsgp_t], by_term[:hsgp_x]
    @test periodic.harmonics == [1.0, 2.0, 1.0, 2.0]
    @test isempty(periodic.omega2)
    @test isempty(exp_quad.harmonics)
    @test size(exp_quad.omega2) == (3, 1)
    @test periodic.target_c == zeros(4)
    # No coordinate is claimed twice across the three families.
    claimed = vcat(
        vec(only(ordinary).effects), periodic.effects, exp_quad.effects,
    )
    @test length(unique(claimed)) == length(claimed)
end

@testset "shared periodic log-scale helper matches the kernel reference" begin
    # Pins `_brm_hsgp_periodic_log_scale` itself: `exp(helper)` is the
    # Riutort-Mayol weight `sigma * sqrt(2 * exp(-a) * I_j(a))`, the same
    # reference `test/gp_hsgp_periodic.jl` rebuilds from `besselix`.
    for (j, sigma, rho) in ((1, 0.7, 0.8), (3, 1.3, 52.0), (8, 0.5, 120.0))
        a = inv(rho^2)
        expected = sigma * sqrt(2 * besselix(j, a))
        @test exp(BRM._brm_hsgp_periodic_log_scale(j, sigma, rho)) ≈ expected rtol=1e-14
    end
end

@testset "Turing ext log spectrum delegates to the shared helper" begin
    ext = Base.get_extension(BRM, :BayesianRegressionModelsTuringExt)
    if isnothing(ext)
        @test_skip "Turing extension not loaded (run under --project=test)"
    else
        state = (; cov=:periodic, harmonics=[1.0, 2.0, 1.0, 2.0])
        got = ext._brm_hsgp_log_sqrt_spd(state, 0.7, 52.0)
        expected = [BRM._brm_hsgp_periodic_log_scale(j, 0.7, 52.0)
                    for j in state.harmonics]
        @test got ≈ expected atol=0
    end
end
