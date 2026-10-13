# Concurrent native-Reverse RK gradients under GC pressure: the sampler query
# behind `rk_logdensity_problem` (correlated random effects, an HSGP with a
# modeled log-scale, a logit GLM, an `@plate for` population-PK cell) and the
# NLME entry points over the same RK model (batched lockstep gradients, one
# subject's compact cell gradient, reverse-over-forward η Hessian blocks).
# WarmupHMC's multi-chain and completion samplers differentiate one deepcopy of
# the density per chain on separate threads; `threaded_gradient_stress.jl` runs
# the stress in a child Julia and checks it.
#
#     julia --project=test test/rk_enzyme_threads.jl
using BayesianRegressionModels, Distributions, LogDensityProblems, Random
using ReactiveKernels, ReactiveKernelsPPL
import Enzyme
using DifferentiationInterface: AutoEnzyme
include("threaded_gradient_stress.jl")

const BRM = BayesianRegressionModels
# Seconds per case: these paths corrupt rarely enough that a fixed count misses it.
const _SECONDS_PER_CASE = 15.0
const _AD = AutoEnzyme(; mode=Enzyme.Reverse)

function _sampler_problem(brmi)
    problem = BRM.rk_logdensity_problem(BRM.RKBRMI(brmi); ad_backend=_AD)
    problem, LogDensityProblems.dimension(problem)
end

function _grouped_data(; G=12, n=6)
    rng = Xoshiro(3)
    g = repeat(["g$i" for i in 1:G]; inner=n)
    x = randn(rng, G * n)
    z = randn(rng, G * n)
    y = 0.3 .+ 0.5 .* x .- 0.2 .* z .+ 0.3 .* randn(rng, G * n)
    (; g, x, z, y, yb=Int.(y .> 0.3))
end

_ranef_problem() = _sampler_problem(@brm _grouped_data() begin
    mu ~ 1 + x + z + (1 + x + z | ID | g)
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end)

_hsgp_problem() = _sampler_problem(@brm (; x=collect(range(-1, 1; length=40)),
        y=sin.(3 .* range(-1, 1; length=40))) begin
    mu ~ 1 + hsgp(x; k=8)
    log(sigma) ~ 1 + x
    y ~ Normal(mu, sigma)
end)

_logit_problem() = _sampler_problem(@brm _grouped_data() begin
    eta ~ 1 + x + z + (1 | g)
    yb ~ BernoulliLogit(eta)
end)

const _PK_DATA = (; subject=["a", "b", "c", "d"], dose=[100.0, 80.0, 120.0, 90.0],
    weight=[0.1, -0.2, 0.3, 0.0],
    t=[[0.5, 1.0, 2.0], [0.5, 3.0], [1.0, 2.0, 4.0, 8.0], [1.0, 6.0]],
    dv=[[18.0, 15.0, 11.0], [20.0, 8.0], [16.0, 12.0, 7.0, 3.0], [17.0, 6.0]])

_plate_pk() = @brm _PK_DATA begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | pk | subject)
    log_V ~ 1 + (1 | pk | subject)
    @plate for i in eachindex(log_CL)
        CL = exp(log_CL[i])
        Vc = exp(log_V[i])
        pred[i] = dose[i] / Vc .* exp.(-(CL / Vc) .* t[i])
        dv[i] ~ normal(pred[i], sigma)
    end
end

_plate_problem() = _sampler_problem(_plate_pk())

# The NLME entry points are not log densities; this adapter returns whatever
# one returns, which the stress compares bitwise against its reference.
struct _EntryPoint{F}
    f::F
    d::Int
end
LogDensityProblems.dimension(p::_EntryPoint) = p.d
LogDensityProblems.capabilities(::Type{<:_EntryPoint}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity_and_gradient(p::_EntryPoint, x) = p.f(x)

# Packed `x = [θ; σ; vec(H)]`, θ shifted to a plausible CL/V scale. An
# `RKBRMI` (and so a `BRMNLMEModel`) holds its plan's Module, which cannot be
# deep-copied, so every worker builds its own model.
function _nlme_entry_point(evaluate)
    m = BRM.brm_nlme_model(BRM.RKBRMI(_plate_pk()); ad_backend=_AD)
    nθ, nσ = length(m.theta), length(m.sigma)
    nη, n = sum(m.eta_blocks), length(m.view.levels)
    shift = [3.0; zeros(nθ - 2); 4.0]
    arguments(x) = (x[1:nθ] .+ shift, x[nθ+1:nθ+nσ], reshape(x[nθ+nσ+1:end], nη, n))
    _EntryPoint(x -> evaluate(m, arguments(x)...), nθ + nσ + nη * n)
end

function _nlme_problem(evaluate)
    problem = _nlme_entry_point(evaluate)
    problem, problem.d, _ -> _nlme_entry_point(evaluate)
end

_nlme_batched_problem() = _nlme_problem(BRM.brm_nlme_loglikelihoods_and_gradients)
_nlme_cell_problem() = _nlme_problem((m, θ, σ, H) ->
    [BRM.brm_nlme_loglikelihood_and_gradient(m, i, θ, σ, H[:, i]) for i in axes(H, 2)])
_nlme_hessian_problem() = _nlme_problem(BRM._brm_nlme_hessian_blocks)

const _CASES = ("ranef" => _ranef_problem, "hsgp" => _hsgp_problem,
                "logit" => _logit_problem, "plate" => _plate_problem,
                "nlme_batched" => _nlme_batched_problem,
                "nlme_cell" => _nlme_cell_problem,
                "nlme_hessian" => _nlme_hessian_problem)

run_threaded_gradient_gate(@__FILE__, _CASES, _SECONDS_PER_CASE;
    broken_on_julia_1_10=ENZYME_JULIA_1_10_GC)
