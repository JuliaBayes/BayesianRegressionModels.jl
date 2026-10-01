# test/adaptive_bounded_scales.jl — online adaptive centering with
# `Uniform(a, b)`-bounded HSGP and cdar hyperparameters (snag
# adaptive-centeri-fadd03fd).
#
# Run: julia --project=test test/adaptive_bounded_scales.jl
#
# An explicit `length_scale(:, hsgp(...)) ~ Uniform(a, b)`, `sd(...) ~
# Uniform(a, b)` or cdar `ar(...) ~ Uniform(a, b)` declares the parameter
# `<lower=a, upper=b>`. The adaptive resolvers used to read only `<lower=a>`
# and refused the whole `adaptive_centering_problem` wrapper. They now read
# both declaration shapes and map the unconstrained coordinate with Stan's
# `lb_constrain` / `lub_constrain`.
#
# The wrapper's density identities hold for ANY deterministic function of the
# untouched hyperparameter coordinates, so they cannot show that the cells read
# the right length scale. Each BridgeStan testset therefore first pins the
# physical value every cell reads against BridgeStan's own `param_constrain`,
# then checks the usual identities.

using Test
using BayesianRegressionModels
using BridgeStan
using Distributions: Exponential, LogNormal, Normal, Uniform, censored
import DifferentiationInterface as DI
import Enzyme
using LogDensityProblems
using LogExpFunctions: logistic
import StanBlocks
using WarmupHMC

const BRM = BayesianRegressionModels
const BOUNDED_AC_EXT = Base.get_extension(
    BayesianRegressionModels, :BayesianRegressionModelsWarmupHMCExt,
)
const BOUNDED_CACHE = mkpath(joinpath(tempdir(), "brm-adaptive-bounded-scales"))

bounded_instantiate(sb, name) = StanBlocks.stan_instantiate(
    sb.model; path=joinpath(BOUNDED_CACHE, name * ".stan"),
)

# The physical value BridgeStan constrains unconstrained `x` to for the
# scalar parameter `name`.
function stan_constrained(problem, x, name)
    names = BridgeStan.param_names(problem.model)
    values = BridgeStan.param_constrain(problem.model, x)
    values[only(findall(==(name), names))]
end

# The wrapper keeps the compiled density at its starting frame, preserves it
# through the Jacobian at arbitrary controls, differentiates it correctly, and
# inverts exactly.
function check_wrapper_identities(problem, wrapped, x)
    plain = LogDensityProblems.logdensity_and_gradient(problem, x)
    adaptive = LogDensityProblems.logdensity_and_gradient(wrapped, x)
    @test adaptive[1] ≈ plain[1] atol=2e-11
    @test adaptive[2] ≈ plain[2] atol=2e-11

    ir = WarmupHMC.reparametrizer(wrapped)
    controls = collect(range(0.15, 0.85; length=length(ir.pairs)))
    ir.pairs .= map(ir.pairs, controls) do (idx, value), c
        idx => WarmupHMC.Reparametrization(
            value.target, WarmupHMC.PartiallyCentered(c), value.args...)
    end
    BOUNDED_AC_EXT._sync_sources!(ir.pairs[1][2].args[1].state, ir)
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

# Independent squared-exponential HSGP log spectral weight per basis.
exp_quad_log_scale(sigma, rho, omega2) =
    log(sigma) + 0.5 * log(rho * sqrt(2pi)) - 0.25 * rho^2 * omega2

@testset "adaptive bounds mirror Stan's scalar transforms" begin
    for x in (-40.0, -37.0, -36.0, -5.0, -1e-3, 0.0, 1e-3, 5.0, 40.0)
        @test BRM._adaptive_constrain(x, 0.3, Inf) == exp(x) + 0.3
        @test BRM._adaptive_constrain(x, 1.2, 3.5) ==
            2.3 * BRM._adaptive_inv_logit(x) + 1.2
        @test BRM._adaptive_inv_logit(x) ≈ logistic(x) rtol=4eps()
    end
    # Below log(eps), Stan's inv_logit returns exp(x) itself.
    @test BRM._adaptive_inv_logit(-40.0) == exp(-40.0)
    @test BRM._adaptive_inv_logit(0.0) == 0.5
    # Any finite unconstrained value lands inside the declared interval.
    @test all(x -> 1.2 < BRM._adaptive_constrain(x, 1.2, 3.5) < 3.5,
              range(-30.0, 30.0; length=61))
end

const BOUNDED_JOINT_BUILDER = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject) + hsgp(t; k=2, cov=:periodic, period=24.0) +
        hsgp(x; k=3, c=1.5)
    length_scale(mu, hsgp(x)) ~ Uniform(0.5, 2.0)
    sd(mu, hsgp(x)) ~ Uniform(0.1, 3.0)
    length_scale(mu, hsgp(t)) ~ Uniform(0.8, 5.0)
    y ~ Normal(mu, sigma)
end

function bounded_joint_df()
    x = collect(range(-1.0, 1.0; length=8))
    t = collect(range(0.0, 20.0; length=8))
    y = sin.(range(0.0, 1.0; length=8))
    subject = repeat([11, 12], inner=4)
    (; x, y, subject, t)
end

@testset "bounded exp-quad and periodic HSGPs adapt beside ordinary cells" begin
    sb = SBBRMI(BOUNDED_JOINT_BUILDER(bounded_joint_df()); total_groups=(),
                mod=@__MODULE__)
    problem = bounded_instantiate(sb, "bounded_joint")
    unc_names = BridgeStan.param_unc_names(problem.model)
    blocks = adaptive_centering_blocks(sb, unc_names)
    hsgp_blocks = BRM._adaptive_hsgp_centering_blocks(sb, unc_names)
    @test length(blocks) == 1
    by_term = Dict(b.term => b for b in hsgp_blocks)
    @test sort!(collect(keys(by_term))) == [:hsgp_t, :hsgp_x]
    periodic, exp_quad = by_term[:hsgp_t], by_term[:hsgp_x]
    @test exp_quad.length_scale_lower == [0.5]
    @test exp_quad.length_scale_upper == [2.0]
    @test (exp_quad.sd_lower, exp_quad.sd_upper) == (0.1, 3.0)
    @test periodic.length_scale_lower == [0.8]
    @test periodic.length_scale_upper == [5.0]
    # The periodic SD keeps its default `<lower=0>` declaration.
    @test (periodic.sd_lower, periodic.sd_upper) == (0.0, Inf)

    x = collect(range(-0.6, 0.7; length=length(unc_names)))
    x[blocks[1].log_scales] .= -0.31
    # Every hyperparameter a cell reads equals BridgeStan's constrained value.
    for (block, prefix) in ((exp_quad, "hsgp_x"), (periodic, "hsgp_t"))
        rho = stan_constrained(problem, x, prefix * "_rho_iso")
        sigma = stan_constrained(problem, x, prefix * "_sigma")
        @test BRM._adaptive_constrain(x[only(block.length_scales)],
            only(block.length_scale_lower), only(block.length_scale_upper)) ≈
            rho rtol=4eps()
        @test BRM._adaptive_constrain(x[block.sd], block.sd_lower,
            block.sd_upper) ≈ sigma rtol=4eps()
        @test only(block.length_scale_lower) < rho < only(block.length_scale_upper)
    end
    rho = stan_constrained(problem, x, "hsgp_x_rho_iso")
    sigma = stan_constrained(problem, x, "hsgp_x_sigma")
    for basis in eachindex(exp_quad.effects)
        @test BRM._adaptive_hsgp_log_scale(x, exp_quad, basis) ≈
            exp_quad_log_scale(sigma, rho, exp_quad.omega2[basis, 1]) atol=1e-12
    end
    rho_t = stan_constrained(problem, x, "hsgp_t_rho_iso")
    sigma_t = stan_constrained(problem, x, "hsgp_t_sigma")
    for (basis, j) in enumerate(periodic.harmonics)
        @test BRM._adaptive_hsgp_log_scale(x, periodic, basis) ≈
            BRM._brm_hsgp_periodic_log_scale(j, sigma_t, rho_t) atol=1e-12
    end

    wrapped = adaptive_centering_problem(sb, problem, DI.AutoEnzyme())
    ir = WarmupHMC.reparametrizer(wrapped)
    @test length(ir.pairs) == length(vec(blocks[1].effects)) +
        length(periodic.effects) + length(exp_quad.effects)
    check_wrapper_identities(problem, wrapped, x)
end

const BOUNDED_GROUPED_BUILDER = @brm begin
    loc ~ 1 + x + hsgp(x; k=3, by=g)
    length_scale(loc, hsgp(x)) ~ Uniform(0.4, 3.0)
    y ~ Normal(loc, 1)
end

function bounded_grouped_df()
    x = collect(range(-1.0, 1.0; length=8))
    (; x, y=sin.(x), g=repeat(["a", "b"], inner=4))
end

@testset "bounded grouped HSGP cells share the declared interval" begin
    sb = SBBRMI(BOUNDED_GROUPED_BUILDER(bounded_grouped_df()); total_groups=(),
                mod=@__MODULE__)
    problem = bounded_instantiate(sb, "bounded_grouped")
    unc_names = BridgeStan.param_unc_names(problem.model)
    hsgp_blocks = BRM._adaptive_hsgp_centering_blocks(sb, unc_names)
    @test length(hsgp_blocks) == 2
    @test all(b -> b.length_scale_lower == [0.4], hsgp_blocks)
    @test all(b -> b.length_scale_upper == [3.0], hsgp_blocks)
    @test all(b -> (b.sd_lower, b.sd_upper) == (0.0, Inf), hsgp_blocks)

    x = collect(range(-0.5, 0.6; length=length(unc_names)))
    block = first(hsgp_blocks)
    rho_name = unc_names[only(block.length_scales)]
    @test BRM._adaptive_constrain(x[only(block.length_scales)], 0.4, 3.0) ≈
        stan_constrained(problem, x, rho_name) rtol=4eps()

    wrapped = adaptive_centering_problem(sb, problem, DI.AutoEnzyme())
    @test length(WarmupHMC.reparametrizer(wrapped).pairs) == 6
    check_wrapper_identities(problem, wrapped, x)
end

# An HSGP over a model-derived (sampled) axis, orthogonal to that axis's
# linear term, with a bounded Uniform length scale. Toy data and values.
function bounded_latent_df()
    n = 12
    u = collect(range(-1.0, 1.0; length=n))
    driver = exp.(-1.1 .+ 0.6 .* u)
    z_floor = fill(0.35, n)
    z_obs = max.(driver, z_floor)
    flag = Float64.(driver .<= z_floor)
    period = repeat([1, 2], outer=n ÷ 2)
    unit = repeat(1:(n ÷ 2), inner=2)
    y = 0.3 .* driver
    (; u, period, unit, flag, z_obs, z_floor, y)
end

const BOUNDED_LATENT_BUILDER = @brm begin
    log(x) ~ 1 + factor(period) + (1 | a | unit)
    x_sd ~ Exponential(1)
    z_obs ~ censored(LogNormal(log(x), x_sd); lower=z_floor)
    mu ~ 1 + factor(period) + flag + x +
         hsgp(x; k=4, domain=(0.02, 4.0), orthogonal_to=:linear) +
         (1 + x | b | unit)
    length_scale(:, hsgp(x)) ~ Uniform(1.2, 3.5)
    sd(:, hsgp(x)) ~ Normal(0, 0.8)
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end

@testset "bounded HSGP over a model-derived orthogonal axis adapts" begin
    sb = SBBRMI(BOUNDED_LATENT_BUILDER(bounded_latent_df()); total_groups=(),
                mod=@__MODULE__)
    problem = bounded_instantiate(sb, "bounded_latent")
    unc_names = BridgeStan.param_unc_names(problem.model)
    block = only(BRM._adaptive_hsgp_centering_blocks(sb, unc_names))
    @test block.term === :hsgp_x
    @test block.length_scale_lower == [1.2]
    @test block.length_scale_upper == [3.5]
    @test (block.sd_lower, block.sd_upper) == (0.0, Inf)
    @test !isempty(adaptive_centering_blocks(sb, unc_names))

    x = collect(range(-0.4, 0.5; length=length(unc_names)))
    @test BRM._adaptive_constrain(x[only(block.length_scales)], 1.2, 3.5) ≈
        stan_constrained(problem, x, "hsgp_x_rho_iso") rtol=4eps()

    wrapped = adaptive_centering_problem(sb, problem, DI.AutoEnzyme())
    check_wrapper_identities(problem, wrapped, x)
end

const BOUNDED_CDAR_DATA = (;
    week=repeat(1:2; inner=2), patch=repeat(["a", "b"]; outer=2),
    y=[0.5, -0.3, 0.4, -0.2], C=[1.0 0.5; 0.5 2.0])

const BOUNDED_CDAR_BUILDER = @brm BOUNDED_CDAR_DATA begin
    mu ~ 1 + cdar(week; by=patch, cor=C)
    sd(:, cdar(week)) ~ Uniform(0.05, 2.0)
    ar(:, cdar(week)) ~ Uniform(0.2, 0.9)
    y ~ Normal(mu, 0.1)
end

@testset "bounded cdar scale and persistence adapt" begin
    sb = SBBRMI(BOUNDED_CDAR_BUILDER; mod=@__MODULE__)
    problem = bounded_instantiate(sb, "bounded_cdar")
    unc_names = BridgeStan.param_unc_names(problem.model)
    block = only(BRM._adaptive_cdar_centering_blocks(sb, unc_names))
    @test (block.sigma_lower, block.sigma_upper) == (0.05, 2.0)
    @test (block.rho_lower, block.rho_upper) == (0.2, 0.9)

    x = collect(range(-0.2, 0.25; length=length(unc_names)))
    sigma, rho = BRM._adaptive_cdar_physical(block, x)
    @test sigma ≈ stan_constrained(problem, x, unc_names[block.sigma]) rtol=4eps()
    @test rho ≈ stan_constrained(problem, x, unc_names[block.rho]) rtol=4eps()
    @test 0.2 < rho < 0.9

    wrapped = adaptive_centering_problem(sb, problem, DI.AutoEnzyme())
    @test length(WarmupHMC.reparametrizer(wrapped).pairs) == 4
    check_wrapper_identities(problem, wrapped, x)
end

@testset "bounded declarations outside the supported shapes still raise" begin
    # A cdar persistence must stay a stationary interval.
    @test_throws "outside the stationary interval" BRM._adaptive_cdar_rho_bounds(
        (; data=Dict{Symbol,Any}()),
        BRM.BRMOutput(:rho, :parameter, :real, (), (; lower=0.0, upper=1.5),
            :sampled, nothing, :parameter, nothing, nothing),
        :cdar_mu_week,
    )
end
