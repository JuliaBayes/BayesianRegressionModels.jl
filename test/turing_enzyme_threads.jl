# Concurrent Enzyme gradients through the Turing backend under GC pressure:
# DynamicPPL's native AD gradient of BRM-generated models (a random intercept;
# a same-axis location/log-scale HSGP pair) and the Turing adaptive-centering
# wrappers around them. WarmupHMC's multi-chain and completion samplers, and
# Turing's `MCMCThreads`, differentiate one copy of the density per chain on
# separate threads; `threaded_gradient_stress.jl` runs the stress in a child
# Julia and checks it.
#
#     julia --project=test test/turing_enzyme_threads.jl
using BayesianRegressionModels, Distributions, LogDensityProblems, Random
using Turing, WarmupHMC
import Enzyme
using DifferentiationInterface: AutoEnzyme
include("threaded_gradient_stress.jl")

const BRM = BayesianRegressionModels
const DP = Turing.DynamicPPL
# Seconds per case: these paths corrupt rarely enough that a fixed count misses it.
const _SECONDS_PER_CASE = 15.0
const _AD = AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation=Enzyme.Const)

function _random_intercept()
    x = collect(range(-1.5, 1.5; length=24))
    data = (; x, subject=repeat(["a", "b", "c", "d", "e", "f"]; inner=4),
        y=0.25 .+ 0.7 .* x .+ 0.1 .* sin.(1:24))
    backend = TuringBRMI((@brm begin
        mu ~ 1 + x + (1 | subject)
        y ~ Normal(mu, 0.6)
    end)(data))
    parameters = (; beta_pop=[0.2, 0.65],
        group_1_1=(; log_scale=log(0.7), z=collect(range(-0.4, 0.6; length=6))))
    backend, DP.VarInfo(backend.model, DP.InitFromParams(parameters), DP.UnlinkAll())
end

function _same_axis_hsgp()
    x = collect(range(-1.2, 1.2; length=24))
    data = (; x, y=0.25 .+ 0.35 .* sin.(2.2 .* x) .+ 0.1 .* cos.(5 .* x))
    backend = TuringBRMI((@brm begin
        mu ~ 1 + hsgp(x; k=6)
        log(sigma) ~ 1 + hsgp(x; k=4)
        y ~ Normal(mu, sigma)
    end)(data))
    mu_term = only(backend.plan.predictors[1].terms)
    sigma_term = only(backend.plan.predictors[2].terms)
    parameters = (; beta_pop=[0.15],
        term_mu_1=(; rho=mu_term.state.rho_lower + 0.55, sigma=0.65,
            beta_raw=collect(range(-0.2, 0.3; length=6))),
        beta_pop_sigma=[-0.35],
        term_sigma_1=(; rho=sigma_term.state.rho_lower + 0.45, sigma=0.4,
            beta_raw=collect(range(0.2, -0.1; length=4))))
    backend, DP.VarInfo(backend.model, DP.InitFromParams(parameters), DP.LinkAll())
end

# Prepared as the Turing adaptive-centering wrapper prepares its inner
# gradient; without fixed transforms Enzyme rejects the HSGP model's type
# analysis.
function _native_problem(model)
    backend, vi = model()
    problem = DP.LogDensityFunction(backend.model, DP.getlogjoint_internal, vi;
        adtype=_AD, fix_transforms=true)
    problem, LogDensityProblems.dimension(problem)
end

# Random centering controls, so both the transform and the inner gradient work.
function _centering_problem(model)
    backend, vi = model()
    density = DP.LogDensityFunction(backend.model, DP.getlogjoint_internal, vi)
    problem = adaptive_centering_problem(backend, density, _AD)
    ir = WarmupHMC.reparametrizer(problem)
    controls = rand(Xoshiro(1), length(ir.pairs))
    ir.pairs .= map(ir.pairs, controls) do (idx, value), c
        idx => WarmupHMC.Reparametrization(
            value.target, WarmupHMC.PartiallyCentered(c), value.args...)
    end
    problem.scoring_plan.synchronize!(ir)
    problem, LogDensityProblems.dimension(problem)
end

const _CASES = (
    "native_random_intercept" => () -> _native_problem(_random_intercept),
    "native_hsgp" => () -> _native_problem(_same_axis_hsgp),
    "centering_random_intercept" => () -> _centering_problem(_random_intercept),
    "centering_hsgp" => () -> _centering_problem(_same_axis_hsgp))

run_threaded_gradient_gate(@__FILE__, _CASES, _SECONDS_PER_CASE;
    broken_on_julia_1_10=ENZYME_JULIA_1_10_GC)
