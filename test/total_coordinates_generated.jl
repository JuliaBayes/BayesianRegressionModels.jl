# test/total_coordinates_generated.jl — every `_total_coordinates` caller on a
# model whose exact total-coefficient block is emitted in GENERATED QUANTITIES.
#
# The generic toy joint model of `population_draws_generated.jl`, built here
# from its own copy: one observed and one omitted response, default
# `total_groups = :auto`. No likelihood reaches the omitted response, so its
# totals block `total_nu` (and its scales) is drawn in generated quantities and
# has no unconstrained coordinate, while `total_mu` is sampled.
#
# A caller skips a generated block where it has nothing to contribute, and
# refuses, naming the block, where it would need coordinates the draws do not
# carry; none surfaces `_total_coordinates`' raw "expects unconstrained
# coordinate" error (todo `BayesianRegressionModels/todos/2026-10-15T22-51-11-856-1vd49r3`).
#
# Run: julia --project=test test/total_coordinates_generated.jl

using Test
using BayesianRegressionModels
using StanBlocks
using Random
using LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme
using Distributions: Exponential, Normal

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan

totals_builder = @brm begin
    sigma  ~ Exponential(1)
    sigma2 ~ Exponential(1)
    mu ~ 1 + x + (1 | subject)
    nu ~ 1 + (1 | subject)
    y ~ Normal(mu, sigma)
    z ~ Normal(nu, sigma2)
end

rng = MersenneTwister(3)
n = 24
covariates = (; x = randn(rng, n), subject = repeat(11:16; inner = 4))
observed = merge(covariates, (; y = randn(rng, n)))   # `z` omitted: unobserved
complete = merge(observed, (; z = randn(rng, n)))

totals_sb = SBBRMI(totals_builder(observed); mod = @__MODULE__)
complete_sb = SBBRMI(totals_builder(complete); mod = @__MODULE__)
prior_sb = SBBRMI(totals_builder(covariates); mod = @__MODULE__)

# `BRM.stan_instantiate` takes an `SBBRMI`; a `GenerativePlan` goes through its
# traced program, in the current world for the same lowering-time reason.
instantiate(model) = Base.invokelatest(StanBlocks.stan_instantiate, model.model)
unc_names(model) = BS.param_unc_names(instantiate(model).model)
block_of(model, predictor) = only(b for b in total_effect_blocks(model) if b.predictor === predictor)

const PROBLEM = instantiate(totals_sb)
const UNC = BS.param_unc_names(PROBLEM.model)
const UNC_COMPLETE = unc_names(complete_sb)
const MU = block_of(totals_sb, :mu)
const NU = block_of(totals_sb, :nu)
const MU_COORDS = BRM._total_coordinates(totals_sb, MU, UNC)

@testset "placement: total_mu sampled, total_nu generated with its scales" begin
    bindings(blocks) = [b.binding for b in blocks]
    placement = BRM._total_blocks_by_placement(totals_sb)
    @test bindings(placement.sampled) == [:total_mu]
    @test bindings(placement.generated) == [:total_nu]
    @test !any(n -> startswith(n, "total_nu.") || startswith(n, "total_scale_nu"), UNC)
    @test count(startswith("total_mu."), UNC) == 6
    @test bindings(BRM._total_blocks_by_placement(complete_sb).sampled) == [:total_mu, :total_nu]
    @test isempty(BRM._total_blocks_by_placement(prior_sb).sampled)
    @test bindings(BRM._total_blocks_by_placement(prior_sb).generated) == [:total_mu, :total_nu]
    # The low-level lookup has nothing to resolve for a generated block; the
    # callers below must never reach it with one.
    @test_throws "expects unconstrained coordinate `total_nu.1.1`" BRM._total_coordinates(
        totals_sb, NU, UNC)
end

@testset "recover_population_draws: sampled blocks recovered, generated ones skipped" begin
    draws = randn(MersenneTwister(5), 4, length(UNC))
    recovered = recover_population_draws(totals_sb, draws, UNC; rng = MersenneTwister(6))
    @test collect(keys(recovered)) == [:mu]
    mu = recovered[:mu]
    @test size(mu.population) == (4, length(MU.population_columns))
    @test all(i -> mu.totals[i, :, :] == draws[i, MU_COORDS.totals], 1:4)
    @test all(isfinite, mu.population)
    # Both blocks sampled: both recovered, as before.
    complete_draws = randn(MersenneTwister(5), 4, length(UNC_COMPLETE))
    @test Set(keys(recover_population_draws(complete_sb, complete_draws, UNC_COMPLETE))) ==
          Set([:mu, :nu])
    # A conventional model has no total blocks and nothing to recover.
    conventional = SBBRMI(totals_builder(observed); mod = @__MODULE__, total_groups = ())
    @test isempty(recover_population_draws(conventional, zeros(2, 0), String[]))
end

# The generated block's recovered population coefficient is still there: Stan
# draws it in generated quantities, where the logical query resolves it.
@testset "a generated block's population coefficient is a generated quantity" begin
    sm = PROBLEM.model
    names = BS.param_names(sm; include_tp = true, include_gq = true)
    query = brm_population_effect_coordinates(brm_descriptor(totals_sb), :nu, names;
                                              coefficient = :Intercept)
    @test query.recovered
    values = BS.param_constrain(sm, zeros(length(UNC)); include_tp = true,
                                include_gq = true, rng = BS.StanRNG(sm, 9))
    @test all(isfinite, values[query.coordinates])
end

# refused: every block generated means the draws condition no total at all;
# an empty result would read as "no exact-total blocks" (fail closed, BRM
# primer invariant: unsupported semantic coordinate transport fails closed).
@testset "a model whose every total block is generated is refused" begin
    @test_throws "recover_population_draws: every exact total-coefficient block (total_mu, total_nu)" recover_population_draws(
        prior_sb, zeros(3, 0), String[])
    @test_throws "select_total_centeredness: every exact total-coefficient block (total_mu, total_nu)" select_total_centeredness(
        prior_sb, zeros(3, 0), String[])
    @test isempty(BRM._total_centering_cells(prior_sb, String[]).indices)
end

@testset "total centering: cells and selection cover the sampled block only" begin
    cells = BRM._total_centering_cells(totals_sb, UNC)
    @test cells.indices == vec(MU_COORDS.totals)
    @test cells.scales == fill(only(MU_COORDS.scales), 6)
    @test cells.targets == ones(6)
    sm = PROBLEM.model
    pilot = 0.3 .* randn(MersenneTwister(9), 40, length(UNC))
    gradients = permutedims(hcat([last(BS.log_density_gradient(sm, collect(row)))
                                  for row in eachrow(pilot)]...))
    for criterion in (:position, :gradient)
        selected = select_total_centeredness(totals_sb, pilot, UNC; criterion, gradients)
        @test selected.indices == vec(MU_COORDS.totals)
        @test length(selected.centeredness) == 6
        @test all(0 .<= selected.centeredness .<= 1)
    end
end

@testset "transport_draws: generated target blocks skipped, generated sources refused" begin
    draws = randn(MersenneTwister(11), 3, length(UNC))
    copied(moved, from_draws, from_names, to_names) = all(eachindex(to_names)) do j
        moved[:, j] == from_draws[:, findfirst(==(to_names[j]), from_names)]
    end
    # Same declaration replayed on a new covariate grid: `total_nu` generated
    # on both sides, every remaining coordinate aligned by name.
    regrid = reprocess(totals_sb, merge(observed, (; x = randn(MersenneTwister(4), n))))
    unc_regrid = unc_names(regrid)
    @test copied(transport_draws(totals_sb, regrid, draws, UNC, unc_regrid), draws, UNC, unc_regrid)
    # New subjects: kept levels copy their totals, new levels draw from the
    # recovered population; the generated block is the target's own re-draw.
    plan = generative_plan(totals_builder, observed)
    future = (; x = randn(MersenneTwister(12), 8), subject = repeat(15:18; inner = 2),
              y = randn(MersenneTwister(13), 8))
    newplan = generative_plan(plan, future)
    @test [b.binding for b in BRM._total_blocks_by_placement(newplan).generated] == [:total_nu]
    unc_new = unc_names(newplan)
    moved = transport_draws(plan, newplan, draws, unc_names(plan), unc_new;
                            rng = MersenneTwister(14))
    new_coords = BRM._total_coordinates(newplan, block_of(newplan, :mu), unc_new)
    @test moved[:, new_coords.totals[1:2, :]] == draws[:, MU_COORDS.totals[5:6, :]]
    @test all(isfinite, moved[:, new_coords.totals[3:4, :]])
    # Fitted with both responses, replayed with `z` omitted: the target
    # generates `total_nu`, so its fitted totals and scales are dropped.
    complete_draws = randn(MersenneTwister(15), 3, length(UNC_COMPLETE))
    @test copied(transport_draws(complete_sb, totals_sb, complete_draws, UNC_COMPLETE, UNC),
                 complete_draws, UNC_COMPLETE, UNC)
    # refused: the source never sampled `total_nu`, so the fitted draws carry
    # no totals or scales for a target that samples it; inventing them would be
    # unsupported coordinate transport (BRM primer invariant: fail it closed).
    @test_throws "the source emits it in generated quantities" transport_draws(
        totals_sb, complete_sb, draws, UNC, UNC_COMPLETE)
    # All generated on both sides: nothing to transport, nothing refused.
    @test size(transport_draws(prior_sb, prior_sb, zeros(2, 0), String[], String[])) == (2, 0)
end

@testset "adaptive_centering_problem: the generated block has no cell" begin
    problem = PROBLEM
    physical = 0.2 .* randn(MersenneTwister(16), length(UNC))
    for c in (0.0, 1.0)
        rp = adaptive_centering_problem(totals_sb, problem, AutoEnzyme(); centeredness = c)
        ir = WarmupHMC.reparametrizer(rp)
        @test [index for (index, _) in ir.pairs] == vec(MU_COORDS.totals)
        _, source = WarmupHMC._inverse_with_logabsdet_jacobian(ir, physical)
        jac, mapped = ir(source)
        @test mapped ≈ physical atol = 1e-12
        @test LogDensityProblems.logdensity(rp, source) ≈
              LogDensityProblems.logdensity(problem, physical) + jac atol = 1e-10
    end
    # A generated total block does not claim the totals wrapper: with an
    # ordinary sampled block beside it, the ordinary wrapper is chosen.
    mixed_builder = @brm begin
        sigma  ~ Exponential(1)
        sigma2 ~ Exponential(1)
        mu ~ 1 + x + (1 + x | site)
        nu ~ 1 + (1 | subject)
        y ~ Normal(mu, sigma)
        z ~ Normal(nu, sigma2)
    end
    mixed = SBBRMI(mixed_builder(merge(observed, (; site = repeat(1:4, 6))));
                   mod = @__MODULE__, total_groups = [:subject])
    @test isempty(BRM._total_blocks_by_placement(mixed).sampled)
    mixed_problem = instantiate(mixed)
    mixed_names = BS.param_unc_names(mixed_problem.model)
    rp = adaptive_centering_problem(mixed, mixed_problem, AutoEnzyme())
    ordinary = only(adaptive_centering_blocks(mixed, mixed_names))
    @test Set(index for (index, _) in WarmupHMC.reparametrizer(rp).pairs) ⊆
          Set(vec(ordinary.effects))
end
