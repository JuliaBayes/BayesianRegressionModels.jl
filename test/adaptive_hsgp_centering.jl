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
    mode=Enzyme.set_runtime_activity(Enzyme.Reverse))

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
