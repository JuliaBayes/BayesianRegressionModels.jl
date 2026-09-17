using Test
using BayesianRegressionModels
using BridgeStan
import DifferentiationInterface as DI
import Enzyme
using LogDensityProblems
import StanBlocks
using Turing

const BRM = BayesianRegressionModels
const BS = BridgeStan
const DP = Turing.DynamicPPL
const ENZYME_BACKEND = DI.AutoEnzyme(;
    mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation=Enzyme.Const)

turing_logdensity_kernel(x, target) = LogDensityProblems.logdensity(target, x)

function turing_density(backend, params)
    vi = DP.VarInfo(backend.model, DP.InitFromParams(params), DP.UnlinkAll())
    ldf = DP.LogDensityFunction(backend.model, DP.getlogjoint_internal, vi)
    q = collect(DP.get_sample_input_vector(ldf))
    preparation = DI.prepare_gradient(
        turing_logdensity_kernel,
        ENZYME_BACKEND, q, DI.Constant(ldf))
    gradient = similar(q)
    (; ldf, q, preparation, gradient)
end

function turing_value_and_gradient(td)
    DI.value_and_gradient!(
        turing_logdensity_kernel,
        td.gradient, td.preparation, ENZYME_BACKEND,
        td.q, DI.Constant(td.ldf))
end

@testset "pilot HSGP centeredness is stable at spectral underflow" begin
    unit = reshape(collect(range(-1.2, 1.3; length=60)), 20, 3)
    logs = hcat(fill(-Inf, 20), collect(range(-3.0, -1.0; length=20)),
                 collect(range(-9.0, -4.0; length=20)))
    selection = select_hsgp_centeredness(
        unit, logs; candidates=0:0.25:1)
    @test selection.centeredness[1] == 0
    @test selection.admissible[:, 1] == [true, false, false, false, false]
    @test all(0 .<= selection.centeredness .<= 1)
    @test all(isfinite, selection.losses[selection.admissible])
end

const HSGP_DATA = (;
    x=collect(range(-1.2, 1.2; length=10)),
    y=[0.15sin(2x) - 0.05cos(3x) for x in range(-1.2, 1.2; length=10)],
    hsgp_c=[0.0, 0.35, 0.75, 1.0],
)
const HSGP_C = [0.0, 0.35, 0.75, 1.0]

const HSGP_PARTIAL = @brm begin
    mu ~ hsgp(x; k=4, c=1.5, centeredness=hsgp_c)
    y ~ Normal(mu, 1)
end

const HSGP_NCP = @brm begin
    mu ~ hsgp(x; k=4, c=1.5)
    y ~ Normal(mu, 1)
end

@testset "partial coordinate preserves physical HSGP density" begin
    @test !isnothing(Base.get_extension(BRM, :BayesianRegressionModelsTuringExt))
    partial = TuringBRMI(HSGP_PARTIAL(HSGP_DATA))
    ncp = TuringBRMI(HSGP_NCP(HSGP_DATA))
    partial_term = only(only(partial.plan.predictors).terms)
    ncp_term = only(only(ncp.plan.predictors).terms)
    @test partial_term.state.PHI == ncp_term.state.PHI
    @test partial_term.state.omega2 == ncp_term.state.omega2
    @test partial_term.state.fits == ncp_term.state.fits

    rho = partial_term.state.rho_lower + 0.45
    sigma = 0.8
    z = [0.2, -0.35, 0.1, 0.4]
    ext = Base.get_extension(BRM, :BayesianRegressionModelsTuringExt)
    log_scale = ext._brm_hsgp_log_sqrt_spd(
        partial_term.state, sigma, rho)
    beta_partial = exp.(HSGP_C .* log_scale) .* z
    partial_model = BRM._brm_turing_term_model(partial_term, length(HSGP_DATA.y))
    ncp_model = BRM._brm_turing_term_model(ncp_term, length(HSGP_DATA.y))
    partial_values = (; rho, sigma, beta_partial)
    ncp_values = (; rho, sigma, beta_raw=z)
    generated = Turing.generated_quantities(partial_model, partial_values)
    expected_weights = exp.(log_scale) .* z
    @test generated.weights ≈ expected_weights rtol=2e-14 atol=2e-14
    @test generated.effect ≈ partial_term.state.PHI * expected_weights

    # u = exp(c log(s))z changes the coordinate density by
    # -sum(c log(s)); both backends obtain this Jacobian from Normal(0, s^c).
    partial_lp = Turing.logjoint(partial_model, partial_values)
    ncp_lp = Turing.logjoint(ncp_model, ncp_values)
    @test partial_lp ≈ ncp_lp - sum(HSGP_C .* log_scale) rtol=2e-13
end

@testset "StanBlocks and Turing share the partial HSGP model" begin
    turing = TuringBRMI(HSGP_PARTIAL(HSGP_DATA))
    term = only(only(turing.plan.predictors).terms)
    rho = term.state.rho_lower + 0.45
    sigma = 0.8
    z = [0.2, -0.35, 0.1, 0.4]
    ext = Base.get_extension(BRM, :BayesianRegressionModelsTuringExt)
    log_scale = ext._brm_hsgp_log_sqrt_spd(term.state, sigma, rho)
    beta_partial = exp.(HSGP_C .* log_scale) .* z
    td = turing_density(turing, (;
        term_mu_1=(; rho, sigma, beta_partial)))
    turing_lp, turing_gradient = turing_value_and_gradient(td)
    @test turing_lp == LogDensityProblems.logdensity(td.ldf, td.q)
    @test all(isfinite, turing_gradient)

    sb = SBBRMI(HSGP_PARTIAL(HSGP_DATA); mod=@__MODULE__)
    code = BRM.stan_code(sb)
    checked = StanBlocks.stanc_check(code)
    checked.ok || @error "partial HSGP stanc" output=checked.output
    @test checked.ok
    path = joinpath(tempdir(), "brm-adaptive-hsgp.stan")
    problem = StanBlocks.stan_instantiate(sb.model; path)
    json = "{\"hsgp_x_rho_iso\":$rho," *
           "\"hsgp_x_sigma\":$sigma," *
           "\"hsgp_x_beta_partial\":[$(join(beta_partial, ','))]}"
    stan_q = BS.param_unconstrain_json(problem.model, json)
    stan_gradient = zeros(length(stan_q))
    stan_lp, _ = BS.log_density_gradient!(
        problem.model, stan_q, stan_gradient;
        propto=false, jacobian=false)
    @test turing_lp ≈ stan_lp rtol=5e-11 atol=5e-9

    stan_names = BS.param_unc_names(problem.model)
    stan_by_name = Dict(stan_names .=> stan_gradient)
    projected = vcat(
        stan_by_name["hsgp_x_rho_iso"] / (rho - term.state.rho_lower),
        stan_by_name["hsgp_x_sigma"] / sigma,
        [stan_by_name["hsgp_x_beta_partial.$i"] for i in eachindex(beta_partial)],
    )
    @test turing_gradient ≈ projected rtol=3e-8 atol=3e-7

    physical = Dict(BS.param_names(problem.model) .=>
                    BS.param_constrain(problem.model, stan_q))
    @test physical["hsgp_x_rho_iso"] ≈ rho
    @test physical["hsgp_x_sigma"] ≈ sigma
end

@testset "two zero-mean HSGPs have distinct backend bindings" begin
    builder = @brm begin
        mu ~ hsgp(x; k=4, centeredness=hsgp_c)
        log(sigma) ~ hsgp(x; k=4, centeredness=hsgp_c)
        y ~ Normal(mu, sigma)
    end
    brmi = builder(HSGP_DATA)
    turing = TuringBRMI(brmi)
    @test length(turing.plan.predictors) == 2
    @test all(p -> size(p.design.matrix, 2) == 0, turing.plan.predictors)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BRM.stan_code(sb)
    @test occursin("hsgp_x_beta_partial", code)
    @test occursin("hsgp_log_sigma_x_beta_partial", code)
    @test haskey(sb.data, :PHI_hsgp_x)
    @test haskey(sb.data, :PHI_hsgp_log_sigma_x)
    checked = StanBlocks.stanc_check(code)
    checked.ok || @error "dual HSGP stanc" output=checked.output
    @test checked.ok
end

using Distributions: Uniform
using LinearAlgebra
using Random: Xoshiro
using WarmupHMC

const HSGP_AC_EXT = Base.get_extension(
    BayesianRegressionModels, :BayesianRegressionModelsWarmupHMCExt,
)

const HSGP_BUILDER = @brm begin
    mu ~ 1 + hsgp(time; k=3)
    log_sigma ~ 1 + hsgp(time_noise; k=2)
    y ~ Normal(mu, exp(log_sigma))
end

const HSGP_SAME_AXIS_BUILDER = @brm begin
    mu ~ 1 + hsgp(time; k=3)
    log(sigma) ~ 1 + hsgp(time; k=2)
    y ~ Normal(mu, sigma)
end

const HSGP_OWNER_ALIAS_BUILDER = @brm begin
    mu ~ 1 + hsgp(log_sigma_time; k=3)
    log(sigma) ~ 1 + hsgp(time; k=2)
    y ~ Normal(mu, sigma)
end

const HSGP_TIME = collect(range(-1.0, 1.0; length=18))
const HSGP_ONLINE_DATA = (;
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

function hsgp_same_axis_fake_unc_names()
    vcat(
        ["hsgp_time_rho_iso", "hsgp_time_sigma"],
        ["hsgp_time_beta_raw.$basis" for basis in 1:3],
        ["hsgp_log_sigma_time_rho_iso", "hsgp_log_sigma_time_sigma"],
        ["hsgp_log_sigma_time_beta_raw.$basis" for basis in 1:2],
    )
end

function hsgp_owner_alias_fake_unc_names()
    vcat(
        ["hsgp_log_sigma_time_rho_iso", "hsgp_log_sigma_time_sigma"],
        ["hsgp_log_sigma_time_beta_raw.$basis" for basis in 1:3],
        ["hsgp_time_rho_iso", "hsgp_time_sigma"],
        ["hsgp_time_beta_raw.$basis" for basis in 1:2],
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

@testset "same-axis HSGPs resolve target-scoped coordinates" begin
    data = (; time=HSGP_ONLINE_DATA.time, y=HSGP_ONLINE_DATA.y)
    sb = SBBRMI(HSGP_SAME_AXIS_BUILDER(data); mod=@__MODULE__)
    names = hsgp_same_axis_fake_unc_names()
    descriptor = brm_descriptor(sb)

    mu_weights = brm_term_coordinates(
        descriptor, :mu, names; term=:hsgp_time, parameter=:basis_weights)
    sigma_weights = brm_term_coordinates(
        descriptor, :sigma, names; term=:hsgp_time, parameter=:basis_weights)
    @test mu_weights.output.name === :hsgp_time_beta_raw
    @test sigma_weights.output.name === :hsgp_log_sigma_time_beta_raw
    @test mu_weights.coordinates == [3, 4, 5]
    @test sigma_weights.coordinates == [8, 9]

    blocks = BRM._adaptive_hsgp_centering_blocks(sb, names)
    @test length(blocks) == 2
    @test getfield.(blocks, :logical) == [:mu, :sigma]
    @test getfield.(blocks, :term) == [:hsgp_time, :hsgp_time]
    @test blocks[1].effects == mu_weights.coordinates
    @test blocks[2].effects == sigma_weights.coordinates
    @test isempty(intersect(blocks[1].effects, blocks[2].effects))
    @test blocks[1].omega2 == sb.data[:omega2_hsgp_time]
    @test blocks[2].omega2 == sb.data[:omega2_hsgp_log_sigma_time]

    alias_data = (;
        time=HSGP_ONLINE_DATA.time,
        log_sigma_time=HSGP_ONLINE_DATA.time_noise,
        y=HSGP_ONLINE_DATA.y,
    )
    alias_sb = SBBRMI(HSGP_OWNER_ALIAS_BUILDER(alias_data); mod=@__MODULE__)
    alias_names = hsgp_owner_alias_fake_unc_names()
    alias_descriptor = brm_descriptor(alias_sb)
    alias_sigma = brm_term_coordinates(
        alias_descriptor, :sigma, alias_names;
        term=:hsgp_time, parameter=:basis_weights)
    @test alias_sigma.output.name === :hsgp_time_beta_raw
    @test alias_sigma.coordinates == [8, 9]
end

const GROUPED_HSGP_ONLINE_BUILDER = @brm begin
    loc ~ 1 + hsgp(x; k=4, by=g)
    y ~ Normal(loc, 1)
end

const GROUPED_HSGP_ONLINE_ANISO_BUILDER = @brm begin
    loc ~ 1 + hsgp(x, x2; k=(3, 4), c=(1.5, 2.0), iso=false, by=g)
    y ~ Normal(loc, 1)
end

const GROUPED_HSGP_ONLINE_JOINT_BUILDER = @brm begin
    loc ~ 1 + x + hsgp(x; k=4, by=g) + (1 + x | subject)
    y ~ Normal(loc, 1)
end

function grouped_hsgp_online_df()
    x = collect(range(-1.0, 1.0; length=8))
    x2 = x .^ 2 .+ 0.1 .* x
    y = sin.(x)
    g = repeat(["a", "b"], inner=4)
    subject = repeat([11, 12], inner=4)
    (; x, x2, y, g, subject)
end

function grouped_hsgp_fake_unc_names()
    vcat(
        ["pop_loc_beta_pop.1"],
        ["hsgp_x_by_g_rho_iso", "hsgp_x_by_g_sigma"],
        ["zflat_hsgpw_x_g.$i" for i in 1:8],
    )
end

@testset "grouped HSGPs resolve one online block per group level" begin
    df = grouped_hsgp_online_df()
    sb = SBBRMI(GROUPED_HSGP_ONLINE_BUILDER(df); mod=@__MODULE__)
    names = grouped_hsgp_fake_unc_names()
    blocks = BRM._adaptive_hsgp_centering_blocks(sb, names)
    @test length(blocks) == 2
    @test getfield.(blocks, :logical) == [:loc, :loc]
    @test getfield.(blocks, :term) == [:hsgp_x_by_g, :hsgp_x_by_g]
    @test blocks[1].effects == [4, 5, 6, 7]
    @test blocks[2].effects == [8, 9, 10, 11]
    @test blocks[1].target_c == zeros(4)
    @test blocks[1].length_scales == [2]
    @test blocks[2].length_scales == [2]
    @test blocks[1].sd == 3
    @test blocks[2].sd == 3
    @test blocks[1].omega2 == blocks[2].omega2 == sb.data[:omega2_hsgp_x_by_g]
    @test isempty(intersect(blocks[1].effects, blocks[2].effects))

    # A missing flat coordinate fails closed naming the coordinate.
    @test_throws "zflat_hsgpw_x_g.8" BRM._adaptive_hsgp_centering_blocks(
        sb, names[1:end-1])

    # Ordinary random-effect metadata ignores the grouped HSGP field.
    @test isempty(BRM.adaptive_centering_blocks(sb, names))

    # Anisotropic grouped terms share one vector length scale per axis.
    aniso_sb = SBBRMI(GROUPED_HSGP_ONLINE_ANISO_BUILDER(df); mod=@__MODULE__)
    aniso_names = vcat(
        ["pop_loc_beta_pop.1"],
        ["hsgp_x_x2_by_g_rho.1", "hsgp_x_x2_by_g_rho.2",
         "hsgp_x_x2_by_g_sigma"],
        ["zflat_hsgpw_x_x2_g.$i" for i in 1:24],
    )
    aniso_blocks = BRM._adaptive_hsgp_centering_blocks(aniso_sb, aniso_names)
    @test length(aniso_blocks) == 2
    @test length(aniso_blocks[1].effects) == 12
    @test aniso_blocks[1].length_scales == [2, 3]
    @test size(aniso_blocks[1].omega2) == (12, 2)

    # A joint model resolves both families with disjoint cells.
    joint_sb = SBBRMI(GROUPED_HSGP_ONLINE_JOINT_BUILDER(df); mod=@__MODULE__)
    joint_names = vcat(
        ["pop_loc_beta_pop.1", "pop_loc_beta_pop.2"],
        ["hsgp_x_by_g_rho_iso", "hsgp_x_by_g_sigma"],
        ["zflat_hsgpw_x_g.$i" for i in 1:8],
        ["r_loc_subject_L.1", "r_loc_subject_tau.1", "r_loc_subject_tau.2",
         "r_loc_subject_z_flat.1", "r_loc_subject_z_flat.2",
         "r_loc_subject_z_flat.3", "r_loc_subject_z_flat.4"],
    )
    joint_hsgp = BRM._adaptive_hsgp_centering_blocks(joint_sb, joint_names)
    joint_ranef = BRM.adaptive_centering_blocks(joint_sb, joint_names)
    @test length(joint_hsgp) == 2
    @test length(joint_ranef) == 1
    hsgp_cells = vcat(vec.(getfield.(joint_hsgp, :effects))...)
    ranef_cells = vcat(
        vec(joint_ranef[1].effects), joint_ranef[1].cholesky_free,
        joint_ranef[1].log_scales)
    @test length(hsgp_cells) == 8
    @test length(ranef_cells) == 4 + 1 + 2
    @test isempty(intersect(hsgp_cells, ranef_cells))
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
    sb = SBBRMI(HSGP_BUILDER(HSGP_ONLINE_DATA); mod=@__MODULE__)
    names = hsgp_fake_unc_names()
    blocks = BRM._adaptive_hsgp_centering_blocks(sb, names)
    @test length(blocks) == 2
    @test getfield.(blocks, :logical) == [:mu, :log_sigma]
    @test getfield.(blocks, :term) == [:hsgp_time, :hsgp_time_noise]
    @test getfield.(blocks, :target_c) == [zeros(3), zeros(2)]
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

    periodic = @brm HSGP_ONLINE_DATA begin
        mu ~ 1 + hsgp(time; k=3, cov=:periodic, period=2.5)
        y ~ Normal(mu, 1)
    end
    periodic_sb = SBBRMI(periodic; mod=@__MODULE__)
    @test_throws "covariance `periodic`" BRM._adaptive_hsgp_centering_blocks(
        periodic_sb, String[],
    )

    grouped_data = merge(HSGP_ONLINE_DATA, (; group=repeat([:a, :b, :c], inner=6)))
    grouped = @brm grouped_data begin
        mu ~ 1 + hsgp(time; k=3, by=group)
        y ~ Normal(mu, 1)
    end
    grouped_sb = SBBRMI(grouped; mod=@__MODULE__)
    grouped_names = vcat(
        ["pop_mu_beta_pop.1"],
        ["hsgp_time_by_group_rho_iso", "hsgp_time_by_group_sigma"],
        ["zflat_hsgpw_time_group.$i" for i in 1:9],
    )
    grouped_blocks = BRM._adaptive_hsgp_centering_blocks(
        grouped_sb, grouped_names)
    @test length(grouped_blocks) == 3
    @test getfield.(grouped_blocks, :term) == fill(:hsgp_time_by_group, 3)
    @test grouped_blocks[1].effects == [4, 5, 6]
    @test grouped_blocks[2].effects == [7, 8, 9]
    @test grouped_blocks[3].effects == [10, 11, 12]
    @test grouped_blocks[1].length_scales == [2]
    @test grouped_blocks[1].sd == 3
    @test grouped_blocks[1].omega2 == grouped_sb.data[:omega2_hsgp_time_by_group]

    bounded = @brm HSGP_ONLINE_DATA begin
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
    sb = SBBRMI(HSGP_BUILDER(HSGP_ONLINE_DATA); mod=@__MODULE__)
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

@testset "same-axis two-HSGP BridgeStan density, gradient, and online warmup" begin
    data = (; time=HSGP_ONLINE_DATA.time, y=HSGP_ONLINE_DATA.y)
    sb = SBBRMI(HSGP_SAME_AXIS_BUILDER(data); mod=@__MODULE__)
    cache = joinpath(tempdir(), "brm-adaptive-hsgp-centering")
    mkpath(cache)
    problem = StanBlocks.stan_instantiate(
        sb.model; path=joinpath(cache, "same_axis_two_hsgp.stan"),
    )
    unc_names = StanBlocks.BridgeStan.param_unc_names(problem.model)
    blocks = BRM._adaptive_hsgp_centering_blocks(sb, unc_names)
    @test length(blocks) == 2
    @test length.(getfield.(blocks, :effects)) == [3, 2]

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
