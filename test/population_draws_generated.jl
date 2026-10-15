# test/population_draws_generated.jl — `population_draws` on a model whose
# selected grouping factor has a block emitted in GENERATED QUANTITIES.
#
# A joint model with one observed and one unobserved response: no likelihood
# reaches the unobserved response's random-effect block, so the traced program
# draws it in generated quantities and it has no unconstrained coordinate.
# `population_draws` zeroes the sampled blocks of the factor and skips the
# generated one, as `transport_draws` does (snag `population-draws-fb937ebb`).
# The load-bearing checks read the model's own transformed parameters back out
# of BridgeStan: the sampled block's per-row contribution is exactly zero.
#
# Run: julia --project=<worktree> test/population_draws_generated.jl
# (needs BridgeStan reachable through StanBlocks; set BRM_PRED_RUNTIME=0 to run
# only the declaration-level half.)

using Test
using BayesianRegressionModels
using StanBlocks
using Random
using Distributions: Exponential, Normal

const PRED_RUNTIME = get(ENV, "BRM_PRED_RUNTIME", "1") != "0"
const BS = StanBlocks.BridgeStan

joint_builder = @brm begin
    sigma  ~ Exponential(1)
    sigma2 ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    nu ~ 1 + (1 | q | subject)
    y ~ Normal(mu, sigma)
    z ~ Normal(nu, sigma2)
end

rng = MersenneTwister(3)
n = 24
covariates = (; x = randn(rng, n), subject = repeat(11:16; inner = 4))
observed = merge(covariates, (; y = randn(rng, n)))   # `z` omitted: unobserved

# Conventional blocks (the correlated `|p|` / `|q|` buckets), then the exact
# total-coefficient emission the default `total_groups=:auto` picks for the
# same shape without the slope.
conv_sb = SBBRMI(joint_builder(observed); mod = @__MODULE__, total_groups = ())
totals_builder = @brm begin
    sigma  ~ Exponential(1)
    sigma2 ~ Exponential(1)
    mu ~ 1 + x + (1 | subject)
    nu ~ 1 + (1 | subject)
    y ~ Normal(mu, sigma)
    z ~ Normal(nu, sigma2)
end
totals_sb = SBBRMI(totals_builder(observed); mod = @__MODULE__)

@testset "the fixtures carry a sampled and a generated block of one factor" begin
    blocks = ranef_blocks(conv_sb)
    @test [(b.binding, b.generated) for b in blocks] ==
        [(:b_p_subject, false), (:b_q_subject, true)]
    @test isempty(ranef_blocks(totals_sb))
    @test Set(b.binding for b in total_effect_blocks(totals_sb)) ==
        Set([:total_mu, :total_nu])
    @test BayesianRegressionModels._total_blocks_generated(
        totals_sb, total_effect_blocks(totals_sb)) == [false, true]   # sorted: mu, nu
end

# refused: with every block of the factor generated, `draws` has no coordinate
# of it to zero, and returning the draws unchanged would report the generated
# effects as population-level (fail closed, BRM primer: unsupported semantic
# coordinate transport).
@testset "a factor whose every block is generated is refused, not a no-op" begin
    for builder in (joint_builder, totals_builder)
        kwargs = builder === joint_builder ? (; total_groups = ()) : (;)
        prior_sb = SBBRMI(builder(covariates); mod = @__MODULE__, kwargs...)
        @test all(b -> b.generated, ranef_blocks(prior_sb))
        @test all(BayesianRegressionModels._total_blocks_generated(
            prior_sb, total_effect_blocks(prior_sb)))
        @test_throws "every random-effect block of grouping factor `subject`" population_draws(
            prior_sb, zeros(2, 0), String[]; groups = :subject)
    end
end

if !PRED_RUNTIME
    @info "BRM_PRED_RUNTIME=0 — skipping the compiled-model half"
else

function constrained_by_name(sm, theta; gq_seed = nothing)
    include_gq = !isnothing(gq_seed)
    names = BS.param_names(sm; include_tp = true, include_gq)
    vals = BS.param_constrain(sm, theta; include_tp = true, include_gq,
        rng = include_gq ? BS.StanRNG(sm, gq_seed) : nothing)
    Dict(n => v for (n, v) in zip(names, vals))
end
entries(d, prefix) = [d[k] for k in sort!([k for k in keys(d) if startswith(k, prefix)];
                                          by = k -> parse(Int, split(k, '.')[end]))]

@testset "conventional: sampled block zeroed, generated block skipped" begin
    sm = BayesianRegressionModels.stan_instantiate(conv_sb).model
    unc = BS.param_unc_names(sm)
    draws = randn(MersenneTwister(5), 4, length(unc))
    pop = population_draws(conv_sb, draws, unc; groups = :subject)

    sampled = only(filter(b -> !b.generated, ranef_blocks(conv_sb)))
    coords = vec(ranef_coordinates(sampled, unc))
    @test all(iszero, pop[:, coords])
    keep = setdiff(eachindex(unc), coords)
    @test pop[:, keep] == draws[:, keep]

    # The model's own answer: the sampled block contributes exactly zero, so
    # the observed response's location is its population part.
    for i in axes(pop, 1)
        cons = constrained_by_name(sm, pop[i, :])
        @test all(iszero, entries(cons, "r_mu_p_subject."))
        @test entries(cons, "mu.") == entries(cons, "pop_mu.")
    end
    # The generated block keeps the program's own draws (documented: a
    # quantity reading it retains its between-group variation).
    gq = constrained_by_name(sm, pop[1, :]; gq_seed = 9)
    @test any(!iszero, entries(gq, "r_nu_q_subject."))
end

@testset "exact totals: sampled totals recovered, generated totals skipped" begin
    sm = BayesianRegressionModels.stan_instantiate(totals_sb).model
    unc = BS.param_unc_names(sm)
    @test !any(startswith("total_nu."), unc)
    draws = randn(MersenneTwister(6), 4, length(unc))
    pop = population_draws(totals_sb, draws, unc; groups = :subject,
                           rng = MersenneTwister(7))
    totals = findall(startswith("total_mu."), unc)
    @test length(totals) == 6
    # Every group total of a draw is replaced by that draw's population mean.
    @test all(i -> allequal(pop[i, totals]), axes(pop, 1))
    @test pop[:, totals] != draws[:, totals]
    keep = setdiff(eachindex(unc), totals)
    @test pop[:, keep] == draws[:, keep]
end

end # PRED_RUNTIME
