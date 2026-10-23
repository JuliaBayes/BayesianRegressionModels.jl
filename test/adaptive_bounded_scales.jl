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
# `lb_constrain` / `lub_constrain`. A `truncated(D; lower=L, ...)` prior whose
# `L` is `length_scale_floor`, a data scalar or a constant declares
# `<lower=fmax(0.0, L)>`; the resolvers evaluate that intersection over the
# bound data (snag adaptive-centeri-9426bb39).
#
# The wrapper's density identities hold for ANY deterministic function of the
# untouched hyperparameter coordinates, so they cannot show that the cells read
# the right length scale. Each BridgeStan testset therefore first pins the
# physical value every cell reads against BridgeStan's own `param_constrain`,
# then checks the usual identities.

using Test
using BayesianRegressionModels
using BridgeStan
using Distributions: Exponential, LogNormal, Normal, Uniform, censored, truncated
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

# A positive scale whose `truncated(D; lower=L, ...)` prior names a non-literal
# `L` — the floor-keeping `length_scale_floor` (brm-use hsgp-priors), a data
# scalar or a constant — declares `<lower=fmax(0.0, L)>`: sbimpl intersects
# the explicit bound with the scale's structural `lower=0`
# (`_sb_bound_intersection`). The resolvers evaluate that intersection over the
# bound data (snag adaptive-centeri-9426bb39). Toy heteroscedastic data.
function bounded_floor_df()
    x = collect(range(-1.0, 1.0; length=24))
    (; x, y=sin.(3 .* x) .+ 0.1 .* cos.(7 .* x), sd_floor=0.05)
end

const BOUNDED_FLOOR_BUILDER = @brm begin
    mu ~ 0 + hsgp(x; k=6, domain=(-1.5, 1.5))
    log(sigma) ~ 0 + hsgp(x; k=4, domain=(-1.5, 1.5))
    length_scale(mu, hsgp(x)) ~ truncated(LogNormal(0, 4); lower=length_scale_floor)
    length_scale(sigma, hsgp(x)) ~
        truncated(LogNormal(0, 4); lower=length_scale_floor, upper=24.0)
    sd(mu, hsgp(x)) ~ truncated(LogNormal(0, 1); lower=sd_floor)
    y ~ Normal(mu, sigma)
end

@testset "floor-keeping and data-valued truncated scale bounds adapt" begin
    sb = SBBRMI(BOUNDED_FLOOR_BUILDER(bounded_floor_df()); mod=@__MODULE__)
    code = BRM.stan_code(sb)
    # The shape under test: every bound here is an intersection, not a symbol.
    @test count("<lower=fmax(0.0, ", code) == 3
    problem = bounded_instantiate(sb, "bounded_floor")
    unc_names = BridgeStan.param_unc_names(problem.model)
    by_lp = Dict(b.logical => b for b in BRM._adaptive_hsgp_centering_blocks(sb, unc_names))
    @test sort!(collect(keys(by_lp))) == [:mu, :sigma]
    mu_block, sigma_block = by_lp[:mu], by_lp[:sigma]
    d = brm_descriptor(sb)
    mu_floor = brm_length_scale_floor(d, :mu; term=:hsgp_x)
    sigma_floor = brm_length_scale_floor(d, :sigma; term=:hsgp_x)
    @test mu_floor > 0 && sigma_floor > 0
    @test mu_block.length_scale_lower == [mu_floor]
    @test mu_block.length_scale_upper == [Inf]
    @test (mu_block.sd_lower, mu_block.sd_upper) == (0.05, Inf)
    @test sigma_block.length_scale_lower == [sigma_floor]
    @test sigma_block.length_scale_upper == [24.0]
    @test (sigma_block.sd_lower, sigma_block.sd_upper) == (0.0, Inf)

    x = collect(range(-0.7, 0.6; length=length(unc_names)))
    # Every hyperparameter a cell reads equals BridgeStan's constrained value.
    for block in (mu_block, sigma_block)
        rho_name = unc_names[only(block.length_scales)]
        sigma_name = unc_names[block.sd]
        rho = stan_constrained(problem, x, rho_name)
        sigma = stan_constrained(problem, x, sigma_name)
        @test BRM._adaptive_constrain(x[only(block.length_scales)],
            only(block.length_scale_lower), only(block.length_scale_upper)) ≈
            rho rtol=4eps()
        @test BRM._adaptive_constrain(x[block.sd], block.sd_lower,
            block.sd_upper) ≈ sigma rtol=4eps()
        @test only(block.length_scale_lower) < rho < only(block.length_scale_upper)
        # Above the floor (`exp(-110)`) a cell's scale is its exact spectral
        # weight; the floor itself is pinned by adaptive_hsgp_scale_floor.jl.
        exact = [exp_quad_log_scale(sigma, rho, block.omega2[basis, 1])
                 for basis in eachindex(block.effects)]
        @test count(>(-110), exact) >= 2
        for basis in eachindex(block.effects)
            @test BRM._adaptive_hsgp_log_scale(x, block, basis) ≈
                BRM._adaptive_hsgp_floor(exact[basis]) atol=1e-12
        end
    end

    wrapped = adaptive_centering_problem(sb, problem, DI.AutoEnzyme())
    @test length(WarmupHMC.reparametrizer(wrapped).pairs) ==
        length(mu_block.effects) + length(sigma_block.effects)
    check_wrapper_identities(problem, wrapped, x)
end

# `truncated(LogNormal(0, 1); lower=length_scale_floor)` is the default
# length-scale prior (LogNormal(0, 1) above the floor, `<lower=rho_lower_...>`)
# up to its normalizer, so its wrapper is the default wrapper plus a constant:
# an oracle that does not depend on how the bounds are read.
const BOUNDED_DEFAULT_FLOOR_BUILDER = @brm begin
    mu ~ 0 + hsgp(x; k=6, domain=(-1.5, 1.5))
    log(sigma) ~ 0 + hsgp(x; k=4, domain=(-1.5, 1.5))
    y ~ Normal(mu, sigma)
end
const BOUNDED_TRUNCATED_DEFAULT_BUILDER = @brm begin
    mu ~ 0 + hsgp(x; k=6, domain=(-1.5, 1.5))
    log(sigma) ~ 0 + hsgp(x; k=4, domain=(-1.5, 1.5))
    length_scale(mu, hsgp(x)) ~ truncated(LogNormal(0, 1); lower=length_scale_floor)
    length_scale(sigma, hsgp(x)) ~ truncated(LogNormal(0, 1); lower=length_scale_floor)
    y ~ Normal(mu, sigma)
end

@testset "a floor-truncated default prior adapts exactly like the default" begin
    wrap(builder, name) = begin
        sb = SBBRMI(builder(bounded_floor_df()); mod=@__MODULE__)
        problem = bounded_instantiate(sb, name)
        (; problem, names=BridgeStan.param_unc_names(problem.model),
           wrapped=adaptive_centering_problem(sb, problem, DI.AutoEnzyme()))
    end
    default = wrap(BOUNDED_DEFAULT_FLOOR_BUILDER, "bounded_default_floor")
    truncated_default = wrap(BOUNDED_TRUNCATED_DEFAULT_BUILDER, "bounded_truncated_default")
    @test default.names == truncated_default.names
    plain_shift, wrapped_shift = Float64[], Float64[]
    for x in (collect(range(-0.7, 0.6; length=length(default.names))),
              collect(range(0.5, -0.4; length=length(default.names))),
              0.3 .* sin.(1:length(default.names)))
        push!(plain_shift, LogDensityProblems.logdensity(truncated_default.problem, x) -
                           LogDensityProblems.logdensity(default.problem, x))
        lp_default, g_default = LogDensityProblems.logdensity_and_gradient(default.wrapped, x)
        lp_truncated, g_truncated = LogDensityProblems.logdensity_and_gradient(
            truncated_default.wrapped, x)
        push!(wrapped_shift, lp_truncated - lp_default)
        @test g_truncated ≈ g_default rtol=1e-10 atol=1e-10
    end
    # One constant: minus the two truncation normalizers.
    @test all(s -> isapprox(s, first(plain_shift); atol=1e-10), plain_shift)
    @test all(s -> isapprox(s, first(plain_shift); atol=1e-10), wrapped_shift)
    @test first(plain_shift) > 0
end

const BOUNDED_ANISO_FLOOR_BUILDER = @brm begin
    mu ~ 0 + hsgp(x1, x2; k=(3, 3), iso=false)
    length_scale(mu, hsgp(x1, x2)) ~ truncated(LogNormal(0, 4); lower=length_scale_floor)
    y ~ Normal(mu, 0.5)
end

function bounded_aniso_df()
    x1 = collect(range(-1.0, 1.0; length=12))
    x2 = collect(range(0.0, 3.0; length=12)) .^ 2
    (; x1, x2, y=sin.(x1) .+ 0.1 .* x2)
end

@testset "anisotropic floor intersections are read per axis" begin
    sb = SBBRMI(BOUNDED_ANISO_FLOOR_BUILDER(bounded_aniso_df()); mod=@__MODULE__)
    # Stan's `fmax(real, vector)` is elementwise: one floor per axis.
    @test occursin("vector<lower=fmax(0.0, ", BRM.stan_code(sb))
    problem = bounded_instantiate(sb, "bounded_aniso_floor")
    unc_names = BridgeStan.param_unc_names(problem.model)
    block = only(BRM._adaptive_hsgp_centering_blocks(sb, unc_names))
    floors = brm_length_scale_floor(brm_descriptor(sb), :mu; term=:hsgp_x1_x2)
    @test length(floors) == 2 && floors[1] != floors[2]
    @test block.length_scale_lower == floors
    @test block.length_scale_upper == [Inf, Inf]

    x = collect(range(-0.5, 0.4; length=length(unc_names)))
    for (axis, i) in enumerate(block.length_scales)
        @test BRM._adaptive_constrain(x[i], block.length_scale_lower[axis],
            block.length_scale_upper[axis]) ≈
            stan_constrained(problem, x, unc_names[i]) rtol=4eps()
    end
    check_wrapper_identities(problem,
        adaptive_centering_problem(sb, problem, DI.AutoEnzyme()), x)
end

@testset "bounded declarations outside the supported shapes still raise" begin
    # Capability gap, not a refusal: a bound written as arithmetic over data
    # (`lower = 2 * length_scale_floor`) declares `fmax(0.0, (2 * floor))`,
    # which the resolvers do not evaluate yet; it raises a named error until
    # then (todo BayesianRegressionModels/todos/2026-10-23T01-46-40-802-18ug4zr).
    arithmetic = SBBRMI((@brm begin
        mu ~ 0 + hsgp(x; k=6, domain=(-1.5, 1.5))
        length_scale(mu, hsgp(x)) ~
            truncated(LogNormal(0, 4); lower=2 * length_scale_floor)
        y ~ Normal(mu, 0.5)
    end)(bounded_floor_df()); mod=@__MODULE__)
    arithmetic_names = BridgeStan.param_unc_names(
        bounded_instantiate(arithmetic, "bounded_arithmetic").model)
    @test_broken try
        !isempty(BRM._adaptive_hsgp_centering_blocks(arithmetic, arithmetic_names))
    catch
        false
    end

    # A cdar persistence must stay a stationary interval.
    @test_throws "outside the stationary interval" BRM._adaptive_cdar_rho_bounds(
        (; data=Dict{Symbol,Any}()),
        BRM.BRMOutput(:rho, :parameter, :real, (), (; lower=0.0, upper=1.5),
            :sampled, nothing, :parameter, nothing, nothing),
        :cdar_mu_week,
    )
end
