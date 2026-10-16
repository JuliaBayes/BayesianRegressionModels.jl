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
# `population_model` is the exact Stan-side counterpart: a program in which
# every block of the factor, sampled or generated, contributes exactly zero,
# with the fitted program's coordinates so the same draws evaluate on it. Its
# exact-total blocks read their population mean `A * beta` instead, with `beta`
# recovered in generated quantities.
#
# Run: julia --project=<worktree> test/population_draws_generated.jl
# (needs BridgeStan reachable through StanBlocks; set BRM_PRED_RUNTIME=0 to run
# only the declaration-level half.)

using Test
using BayesianRegressionModels
using StanBlocks
using Random
using Statistics: mean, std
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

# The population program describes the same blocks: fitted family, effect
# carrier and placement, flagged `population`.
block_shape(b) = (b.binding, b.family, b.group, b.id, b.n_terms, b.n_groups, b.z,
                  b.noncentered, b.generated)

@testset "population_model: same blocks, flagged population" begin
    pop = population_model(conv_sb; groups = :subject)
    @test block_shape.(ranef_blocks(pop)) == block_shape.(ranef_blocks(conv_sb))
    @test all(b -> b.population, ranef_blocks(pop))
    @test !any(b -> b.population, ranef_blocks(conv_sb))
    @test BayesianRegressionModels.stan_code(pop) != BayesianRegressionModels.stan_code(conv_sb)
    # Selecting an already-population factor again changes nothing.
    @test BayesianRegressionModels.stan_code(population_model(pop; groups = :subject)) ==
          BayesianRegressionModels.stan_code(pop)
    fitted_d = brm_descriptor(conv_sb)
    pop_d = brm_descriptor(pop)
    @test [(o.name, o.kind, o.role) for o in pop_d.outputs] ==
          [(o.name, o.kind, o.role) for o in fitted_d.outputs]
    # Replay on a new grid of the same groups, in either order.
    regrid = merge(observed, (; x = randn(MersenneTwister(4), n)))
    replayed = reprocess(pop, regrid)
    @test all(b -> b.population, ranef_blocks(replayed))
    @test BayesianRegressionModels.stan_code(replayed) ==
          BayesianRegressionModels.stan_code(
              population_model(reprocess(conv_sb, regrid); groups = :subject))
    @test BayesianRegressionModels.stan_code(replayed) == BayesianRegressionModels.stan_code(pop)
    # refused: a typo'd factor must not silently return the fitted program
    # (fail closed, as `population_draws` does).
    @test_throws "no random-effect block on grouping factor `subjct`" population_model(
        conv_sb; groups = :subjct)
end

# An exact-total block's population mean is `A * beta`, with `beta` the
# population coefficient the program recovers in generated quantities. Its
# predictor reads that mean, so it is a generated quantity and the observed
# response is held out; the totals keep their placement.
@testset "population_model: exact totals read their population mean" begin
    pop = population_model(totals_sb; groups = :subject)
    code = BayesianRegressionModels.stan_code(pop)
    @test code != BayesianRegressionModels.stan_code(totals_sb)
    @test isempty(totals_sb.held_out)
    @test pop.held_out == Set([:y])
    @test BayesianRegressionModels._total_blocks_generated(
        pop, total_effect_blocks(pop)) == [false, true]
    @test isempty(ranef_blocks(pop))
    # Selecting an already-population factor again changes nothing.
    again = population_model(pop; groups = :subject)
    @test BayesianRegressionModels.stan_code(again) == code
    @test again.held_out == pop.held_out
    ops = Set(o.name for o in brm_descriptor(pop).operations)
    @test :predict in ops && :pointwise_loglik in ops && !(:fit in ops)
    # Replay on a new grid of the same groups, in either order.
    regrid = merge(observed, (; x = randn(MersenneTwister(4), n)))
    replayed = reprocess(pop, regrid)
    @test replayed.held_out == Set([:y])
    @test BayesianRegressionModels.stan_code(replayed) == BayesianRegressionModels.stan_code(
        population_model(reprocess(totals_sb, regrid); groups = :subject))
    # A response the caller already held out is not marked twice: the program
    # equals the one holding out both responses itself.
    both = merge(observed, (; z = randn(MersenneTwister(5), n)))
    held = SBBRMI(totals_builder(both); mod = @__MODULE__, held_out = :z)
    pop_held = population_model(held; groups = :subject)
    @test pop_held.held_out == Set([:y, :z])
    @test BayesianRegressionModels.stan_code(pop_held) == BayesianRegressionModels.stan_code(
        population_model(SBBRMI(totals_builder(both); mod = @__MODULE__); groups = :subject))
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

@testset "population_model: exact totals at the population mean" begin
    pop = population_model(totals_sb; groups = :subject)
    sm = BayesianRegressionModels.stan_instantiate(totals_sb).model
    pop_sm = BayesianRegressionModels.stan_instantiate(pop).model
    unc = BS.param_unc_names(sm)
    # The fitted coordinates, so fitted draws evaluate on it unchanged.
    @test BS.param_unc_names(pop_sm) == unc
    draws = randn(MersenneTwister(6), 4, length(unc))
    for i in axes(draws, 1)
        gq = constrained_by_name(pop_sm, draws[i, :]; gq_seed = 9)
        fitted = constrained_by_name(sm, draws[i, :]; gq_seed = 9)
        # Each row's total contribution is the recovered population intercept,
        # for the sampled block (`mu`) and the generated one (`nu`).
        @test entries(gq, "mu.") == entries(gq, "pop_mu.") .+ only(entries(gq, "population_mu."))
        @test entries(gq, "nu.") == fill(only(entries(gq, "population_nu.")), n)
        # The fitted totals and population part are reported unchanged.
        for prefix in ("total_mu.", "total_scale_mu.", "pop_mu.")
            names = filter(startswith(prefix), collect(keys(fitted)))
            @test !isempty(names) && all(k -> gq[k] == fitted[k], names)
        end
        @test length(entries(gq, "y_gen.")) == n && length(entries(gq, "z_gen.")) == n
        @test all(isfinite, entries(gq, "y_likelihood."))
    end
    # The fitted program's generated block varies between groups.
    @test !allequal(entries(constrained_by_name(sm, draws[1, :]; gq_seed = 9), "nu."))
    # In law: the recovered intercept the observed location reads, and the one
    # `population_draws` substitutes, are both the exact conditional given a draw.
    block = only(filter(b -> b.binding === :total_mu, total_effect_blocks(totals_sb)))
    coords = BayesianRegressionModels._total_coordinates(totals_sb, block, unc)
    conditional = BayesianRegressionModels._total_conditional(block, coords, draws[1, :])
    m, s = only(conditional.mean), sqrt(only(inv(conditional.factor)))
    S = 4000
    program = [only(entries(constrained_by_name(pop_sm, draws[1, :]; gq_seed = seed),
                            "population_mu.")) for seed in 1:S]
    julia = population_draws(totals_sb, repeat(draws[1:1, :], S), unc; groups = :subject,
                             rng = MersenneTwister(8))[:, coords.totals[1, 1]]
    for sample in (program, julia)
        @test abs(mean(sample) - m) < 5 * s / sqrt(S)
        @test isapprox(std(sample), s; rtol = 0.1)
    end
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

# One factor, a sampled `|p|` block and a generated `|q|` block, plus the
# centered emission, a shared `|p|` block split into its sampled active margins
# and generated completion, and a prior program in which every block is
# generated.
shared_builder = @brm begin
    sigma  ~ Exponential(1)
    sigma2 ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    nu ~ 1 + (1 | p | subject)
    y ~ Normal(mu, sigma)
    z ~ Normal(nu, sigma2)
end
population_cases = [
    "conventional" => (conv_sb, ("r_mu_p_subject.", "r_nu_q_subject.")),
    "centered" => (SBBRMI(joint_builder(observed); mod = @__MODULE__, total_groups = (),
                          centered_groups = [:subject]),
                   ("r_mu_p_subject.", "r_nu_q_subject.")),
    "marginal" => (SBBRMI(shared_builder(observed); mod = @__MODULE__, total_groups = ()),
                   ("r_mu_p_subject.", "r_nu_p_subject.")),
    "prior" => (SBBRMI(joint_builder(covariates); mod = @__MODULE__, total_groups = ()),
                ("r_mu_p_subject.", "r_nu_q_subject.")),
]

@testset "population_model: every block contributes exactly zero ($label)" for
        (label, (sb, gathers)) in population_cases
    pop = population_model(sb; groups = :subject)
    sm = BayesianRegressionModels.stan_instantiate(sb).model
    pop_sm = BayesianRegressionModels.stan_instantiate(pop).model
    unc = BS.param_unc_names(sm)
    # The fitted coordinates, so fitted draws evaluate on it unchanged.
    @test BS.param_unc_names(pop_sm) == unc
    @test BS.param_unc_num(pop_sm) == BS.param_unc_num(sm)
    # The prior program samples nothing (its name list is BridgeStan's `[""]`).
    @test (BS.param_unc_num(sm) == 0) == (label == "prior")
    draws = randn(MersenneTwister(5), 4, BS.param_unc_num(sm))
    blocks = ranef_blocks(sb)
    zeroed = all(b -> b.generated, blocks) ? nothing :
        population_draws(sb, draws, unc; groups = :subject)
    for i in axes(draws, 1)
        gq = constrained_by_name(pop_sm, draws[i, :]; gq_seed = 9)
        for g in gathers
            @test !isempty(entries(gq, g))
            @test all(iszero, entries(gq, g))
        end
        # Both responses' locations are their population parts.
        @test entries(gq, "mu.") == entries(gq, "pop_mu.")
        @test entries(gq, "nu.") == entries(gq, "pop_nu.")
        isnothing(zeroed) && continue
        # On the sampled likelihood it is exactly `population_draws`.
        @test entries(gq, "mu.") ==
              entries(constrained_by_name(sm, zeroed[i, :]), "mu.")
    end
    # The descriptor mounts it: `:predict` takes the same draws.
    if label == "conventional"
        predicted = brm_execute(brm_descriptor(pop), :predict; draws = draws[1, :], seed = 3)
        @test length(predicted.z_gen) == n && all(isfinite, predicted.z_gen)
    end
    # The fitted program's generated block is not at the population mean.
    any(b -> b.generated, blocks) && @test any(!iszero,
        entries(constrained_by_name(sm, draws[1, :]; gq_seed = 9), last(gathers)))
end

end # PRED_RUNTIME
