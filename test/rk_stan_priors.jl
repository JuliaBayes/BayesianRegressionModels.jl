# Public shared-block prior acceptance against actual emitted Stan. The two
# readers state the same ordinary indexed algebra in their native languages.
# Run: julia --project=test test/rk_stan_priors.jl
using Test, BayesianRegressionModels, StanBlocks
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
import BridgeStan
include(joinpath(@__DIR__, "rk_source_roundtrip.jl"))

shared_prior_reader(a, b, row) = a[row] .+ b[row]

module SharedPriorStanOracle
using StanBlocks, BayesianRegressionModels
@deffun begin
    shared_prior_reader(a::vector[na], b::vector[nb], row::int[m])::vector[m] =
        a[row] + b[row]
end
const builder = @brm begin
    a ~ 1 + (1 | shared | group)
    b ~ 1 + (1 | shared | group)
    sd(:, shared) ~ Exponential(0.7)
    cor(:, shared) ~ LKJCholesky(2, 3.0)
    reads = shared_prior_reader(a, b, row)
    y ~ Normal(reads, 1.0)
end
end

@testset "explicit shared priors: printed RK and mapped emitted Stan" begin
    data = (; group=["b", "a"], row=[1, 2, 1], y=[0.2, -0.3, 0.5])
    brmi = @brm data begin
        a ~ 1 + (1 | shared | group)
        b ~ 1 + (1 | shared | group)
        sd(:, shared) ~ Exponential(0.7)
        cor(:, shared) ~ LKJCholesky(2, 3.0)
        reads = shared_prior_reader(a, b, row)
        y ~ Normal(reads, 1.0)
    end
    backend = check_rk_source_roundtrip(RKBRMI(brmi))
    sb = SBBRMI(SharedPriorStanOracle.builder(data);
        mod=SharedPriorStanOracle, total_groups=())
    @test BayesianRegressionModels.transpiles(sb)
    cache = joinpath(tempdir(), "brm-rk-shared-priors")
    mkpath(cache)
    problem = BayesianRegressionModels.stan_instantiate(sb;
        path=joinpath(cache, "shared-priors.stan"))
    stan_names = BridgeStan.param_unc_names(problem.model)
    rk_names = coordinate_names(backend.model.layout)
    mapping = [
        :a_Intercept => "pop_a_beta_pop.1",
        :b_Intercept => "pop_b_beta_pop.1",
        Symbol("ranef_draws_shared_group.sd.1") => "b_shared_group_tau.1",
        Symbol("ranef_draws_shared_group.sd.2") => "b_shared_group_tau.2",
        Symbol("ranef_draws_shared_group.z.1.1") => "b_shared_group_z_flat.1",
        Symbol("ranef_draws_shared_group.z.2.1") => "b_shared_group_z_flat.3",
        Symbol("ranef_draws_shared_group.z.1.2") => "b_shared_group_z_flat.2",
        Symbol("ranef_draws_shared_group.z.2.2") => "b_shared_group_z_flat.4",
        Symbol("ranef_draws_shared_group.L.1") => "b_shared_group_L.1"]
    permutation = BayesianRegressionModels.resolve_sb_map(
        mapping, rk_names, stan_names; case_id="shared-exp-lkj")
    rk_problem = rk_logdensity_problem(backend;
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for u in (zeros(9), collect(range(-0.2, 0.3; length=9)), fill(0.13, 9))
        before = copy(u)
        rk_value, rk_gradient = LogDensityProblems.logdensity_and_gradient(rk_problem, u)
        stan_u = BayesianRegressionModels.apply_sb_map(u, permutation)
        stan_before = copy(stan_u)
        stan_gradient = similar(stan_u)
        stan_value, _ = BridgeStan.log_density_gradient!(problem.model,
            stan_u, stan_gradient; propto=false, jacobian=true)
        mapped_gradient = BayesianRegressionModels.unmap_sb_grad(stan_gradient, permutation)
        @test rk_value ≈ stan_value atol=2e-11 rtol=2e-11
        @test rk_gradient ≈ mapped_gradient atol=2e-10 rtol=2e-10
        @test isequal(u, before)
        @test isequal(stan_u, stan_before)
    end
end
