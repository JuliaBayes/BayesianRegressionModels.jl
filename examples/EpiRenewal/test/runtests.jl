# Run from the repository root with BRM's test environment; no sampling:
#   julia --startup-file=no --project=test examples/EpiRenewal/test/runtests.jl
#
# The regression target is the documentation page "Epidemic renewal models": its three models are
# written by hand as Stan functions in research/epi_renewal/renewal.jl and were fitted and checked
# against simulated truth there. The same models spelled with this package's operators must be the
# SAME posterior: same dimension, and the same log density and gradient at arbitrary points. A model
# with the same density needs no second fit.

using Test, Random
using BayesianRegressionModels, StanBlocks
using Distributions: Normal, LogNormal, Gamma
import LogDensityProblems, BridgeStan

include(joinpath(@__DIR__, "..", "src", "EpiRenewal.jl"))
using .EpiRenewal

# the page's source: data, hand-written models and its `build`
module Page
include(joinpath(@__DIR__, "..", "..", "..", "research", "epi_renewal", "renewal.jl"))
end

build(brmi; held_out=()) = StanBlocks.stan_instantiate(SBBRMI(brmi; mod=@__MODULE__, held_out).model)
function same_posterior(package_problem, page_problem; seeds=1:3, rtol=1e-8)
    dimension = LogDensityProblems.dimension(page_problem)
    @test LogDensityProblems.dimension(package_problem) == dimension
    for seed in seeds
        q = 0.1 .* randn(Xoshiro(seed), dimension)
        lp_page, grad_page = LogDensityProblems.logdensity_and_gradient(page_problem, q)
        lp, grad = LogDensityProblems.logdensity_and_gradient(package_problem, q)
        @test isfinite(lp) && all(isfinite, grad)
        @test isapprox(lp, lp_page; rtol)
        @test isapprox(grad, grad_page; rtol=1e-6)
    end
end

# ── the page's models in the package's spelling (also shown on the page) ───────
const page = Page
include(joinpath(@__DIR__, "..", "models.jl"))

@testset "censored_pmf" begin
    pmf = censored_pmf(LogNormal(1.5, 0.5); D=15)
    @test length(pmf) == 15 && sum(pmf) ≈ 1
    @test pmf == Page.reporting_pmf()
    gen = censored_pmf(Gamma(6.5, 0.62); D=14, drop_zero=true)
    @test length(gen) == 13 && sum(gen) ≈ 1
    @test gen == Page.generation_pmf()
    @test_throws ArgumentError censored_pmf(LogNormal(1.5, 0.5); D=0)
end

@testset "epi_frame: the row-order contract" begin
    shuffled = (; day=[2, 1, 2, 1, 3, 3], patch=["b", "b", "a", "a", "a", "b"], cases=[22, 21, 12, 11, 13, 23])
    frame = epi_frame(shuffled; time=:day, by=:patch)
    @test frame.patch == ["a", "a", "a", "b", "b", "b"]
    @test frame.day == [1, 2, 3, 1, 2, 3]
    @test frame.cases == [11, 12, 13, 21, 22, 23]
    @test epi_frame((; day=[3, 1, 2], cases=[3, 1, 2]); time=:day).cases == [1, 2, 3]
    @test_throws "more than one row" epi_frame((; day=[1, 1, 2], patch=["a", "a", "a"]); time=:day, by=:patch)
    @test_throws "does not cover the same" epi_frame((; day=[1, 2, 3, 1, 2], patch=["a", "a", "a", "b", "b"]); time=:day, by=:patch)
    @test_throws "not equally spaced" epi_frame((; day=[1, 2, 4]); time=:day)
    @test_throws "one entry per row" epi_frame((; day=[1, 2, 3], gen_pmf=[0.5, 0.5]); time=:day)
    @test_throws "no column `week`" epi_frame((; day=[1, 2, 3]); time=:week)
    # the page's six-patch frame already satisfies the contract: sorting it changes nothing
    patches = Page.renewal_patch_data()
    rows = (; time=patches.time, week=patches.week, patch=patches.patch, cases=patches.cases,
              observed=patches.observed, seed_mean=patches.seed_mean)
    @test epi_frame(rows; time=:time, by=:patch) == rows
end

@testset "reporting delay: the package family is the page's" begin
    data = Page.reporting_delay_data()
    same_posterior(build(epirenewal_delay_model(data)), Page.build(Page.reporting_delay_model(data)).problem)
end

@testset "one population: renewal + delay are the page's hand-written recursion" begin
    data = Page.renewal_single_data()
    page_problem = Page.build(Page.renewal_single_model(data)).problem
    same_posterior(build(epirenewal_single_model(data)), page_problem)
    # kernels given as functions of the lag (do blocks reading data vectors): the same posterior again
    same_posterior(build(epirenewal_single_model_functions(data)), page_problem)
    # a forecast is a row mask, exactly as on the page
    masked = Page.renewal_single_data(; observed_through=42)
    same_posterior(build(epirenewal_single_model(masked)), Page.build(Page.renewal_single_model(masked)).problem)
    # prior-only build
    @test LogDensityProblems.dimension(build(epirenewal_single_model(data); held_out=:all)) == 59
end

@testset "six coupled patches: gravity mixing and per-patch seeds" begin
    data = Page.renewal_patch_data()
    same_posterior(build(epirenewal_patch_model(data)), Page.build(Page.renewal_patch_model(data)).problem; seeds=1:2)
end

@testset "a delay distribution estimated inside the renewal model" begin
    data = Page.renewal_single_data()
    problem = build(epirenewal_estimated_delay_model(data))
    @test LogDensityProblems.dimension(problem) == 61               # the page's 59 + mu_delay + sigma_delay
    names = BridgeStan.param_unc_names(problem.model)
    q = 0.1 .* randn(Xoshiro(5), 61)
    q[findfirst(==("mu_delay"), names)] = 1.5                        # unconstrained
    q[findfirst(==("sigma_delay"), names)] = log(0.5)                # lower = 0: log transform
    lp, grad = LogDensityProblems.logdensity_and_gradient(problem, q)
    @test isfinite(lp) && all(isfinite, grad)
    @test grad[findfirst(==("mu_delay"), names)] != 0               # the counts inform the delay
    # at the true delay parameters the function form reproduces the data-vector model's expected cases
    # (closed-form masses in Stan vs 400-point quadrature in Julia: equal to quadrature accuracy)
    reference = build(epirenewal_single_model(data))
    reference_names = BridgeStan.param_unc_names(reference.model)
    q_reference = [q[findfirst(==(name), names)] for name in reference_names]
    expected(problem_, q_) = begin
        all_names = BridgeStan.param_names(problem_.model; include_tp=true)
        values = BridgeStan.param_constrain(problem_.model, q_; include_tp=true)
        values[[startswith(name, "Y.") for name in all_names]]
    end
    Y_function, Y_vector = expected(problem, q), expected(reference, q_reference)
    @test length(Y_function) == 56
    @test isapprox(Y_function, Y_vector; rtol=1e-4)
end
