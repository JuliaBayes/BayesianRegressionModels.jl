# Public consumer semantics: full normalized density, ordinary reverse AD,
# independent oracles, actual emitted Stan, and complete printed RK replay.
using Test, BayesianRegressionModels, Distributions, StanBlocks
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
import BridgeStan
using CategoricalArrays
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))
include(joinpath(@__DIR__, "rk_source_roundtrip.jl"))

function consumer_problem(brmi)
    backend = check_rk_source_roundtrip(RKBRMI(brmi))
    println("RK_NAMES=", coordinate_names(backend.model.layout)); flush(stdout)
    backend, rk_logdensity_problem(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
end

function consumer_stan(brmi, name)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    folder = joinpath(tempdir(), "brm-rk-consumer")
    mkpath(folder)
    problem = BRM.stan_instantiate(sb; path=joinpath(folder, name * ".stan"))
    println("STAN_NAMES=", BridgeStan.param_unc_names(problem.model)); flush(stdout)
    problem
end

function check_consumer_point(problem, u, oracle)
    before = copy(u)
    value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
    @test value ≈ oracle(u) atol=2e-11 rtol=2e-11
    step = 1e-5
    independent_gradient = map(eachindex(u)) do j
        plus, minus = copy(u), copy(u)
        plus[j] += step; minus[j] -= step
        (oracle(plus) - oracle(minus)) / (2step)
    end
    @test gradient ≈ independent_gradient atol=2e-8 rtol=2e-8
    @test isequal(u, before)
    value, gradient
end

function check_consumer_stan(problem, stan, mapping, backend, u)
    permutation = BRM.resolve_sb_map(mapping, coordinate_names(backend.model.layout),
        BridgeStan.param_unc_names(stan.model); case_id="public-consumer")
    stan_u = BRM.apply_sb_map(u, permutation)
    gradient = similar(stan_u)
    value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u, gradient;
        propto=false, jacobian=true)
    rk_value, rk_gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
    @test rk_value ≈ value atol=2e-11 rtol=2e-11
    @test rk_gradient ≈ BRM.unmap_sb_grad(gradient, permutation) atol=2e-10 rtol=2e-10
end

