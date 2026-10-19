# test/split_models.jl — `priors_of` / `structure_of` model partition.
#
# Run: julia --project=test test/split_models.jl
#
# The two accessors split ONE BRMI into its prior fragment and its structural
# fragment (each a BRMI). They are the inverse of `Base.merge(::BRMI, ::BRMI)`:
# `merge(structure_of(b), priors_of(b))` must recompose an equivalent model.
# `show(::BRMI)` prints the model's statements as authored (decision `0eqw9xp`),
# so each fragment prints as its own source card; `d.formula` and the web BRMI
# view print the same source.

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

df = (; x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        subject=[1, 1, 2, 2, 3, 3],
        y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5])

# A full model: a linear predictor, a SCALAR parameter declaration
# (`sigma ~ Exponential(1)` — this counts as MODEL, not a prior), a likelihood,
# and the prior-address block (`effect(...)` / `sd(...)`).
inline = @brm df begin
    log_ka ~ 1 + x + (1 | pk | subject)
    sigma ~ Exponential(1)
    y ~ Normal(log_ka, sigma)
    effect(log_ka, Intercept) ~ Normal(log(1 / 8), 0.8)
    effect(log_ka, x) ~ Normal(0, 0.1)
    sd(:, pk) ~ Exponential(1)
end

stan(brmi) = BayesianRegressionModels.stan_code(SBBRMI(brmi; mod=@__MODULE__))
opkeys(brmi) = collect(keys(brmi.operations))

@testset "priors_of / structure_of partition the operations" begin
    priors = priors_of(inline)
    structure = structure_of(inline)
    @test priors isa BayesianRegressionModels.BRMI
    @test structure isa BayesianRegressionModels.BRMI

    # Disjoint, exhaustive cover: every op lands in exactly one fragment, and
    # each fragment keeps the original relative order of its ops.
    @test isempty(intersect(opkeys(priors), opkeys(structure)))
    @test sort(vcat(opkeys(priors), opkeys(structure))) == sort(opkeys(inline))
    @test opkeys(priors) == filter(in(Set(opkeys(priors))), opkeys(inline))
    @test opkeys(structure) == filter(in(Set(opkeys(structure))), opkeys(inline))

    # priors_of holds exactly the prior-address statements.
    @test [(p.predictor, p.coefficient, p.family, p.arguments)
           for p in effect_priors(priors)] ==
          [(p.predictor, p.coefficient, p.family, p.arguments)
           for p in effect_priors(inline)]
    @test length(ranef_effect_priors(priors)) == 1

    # structure_of holds NO prior-address statements...
    @test isempty(effect_priors(structure))
    @test isempty(ranef_effect_priors(structure))
    # ...but KEEPS the scalar `sigma ~ Exponential(1)` (declares a param, not a
    # prior). Its key is the bare `sigma`, present in structure, absent in priors.
    @test :sigma in opkeys(structure)
    @test :sigma ∉ opkeys(priors)
    @test :log_ka in opkeys(structure)   # the linear predictor stays with the model
end

@testset "print views: suppressed vs only" begin
    # structure_of prints the model WITHOUT the prior block ("suppressed").
    structure_txt = sprint(show, structure_of(inline))
    @test occursin("log_ka", structure_txt)
    @test occursin("sigma", structure_txt)
    @test !occursin("effect(", structure_txt)
    @test !occursin("sd(", structure_txt)

    # priors_of prints ONLY the prior block ("set to the only thing"). Every
    # prior statement prints the way it was written: the `sd(:, pk)` prior as
    # `sd(:, pk)`, never as its internal `effect(sd, pk)` carrier.
    priors_txt = sprint(show, priors_of(inline))
    @test occursin("effect(", priors_txt)
    @test occursin("sd(:, pk) ~ ", priors_txt)
    @test !occursin("effect(sd", priors_txt)
    @test !occursin("Normal(log_ka", priors_txt)  # no likelihood
    @test !occursin("1 + x", priors_txt)          # no linear predictor

    # The full listing interleaves both.
    full_txt = sprint(show, inline)
    @test occursin("effect(", full_txt)
    @test occursin("log_ka", full_txt)
end

@testset "split is the inverse of merge" begin
    # merge(structure, priors) recomposes an equivalent model: same Stan.
    recomposed = Base.merge(structure_of(inline), priors_of(inline))
    @test StanBlocks.stan.transpiles(SBBRMI(recomposed; mod=@__MODULE__).model)
    @test stan(recomposed) == stan(inline)

    # Non-vacuous: structure alone (priors defaulted) lowers differently.
    @test stan(structure_of(inline)) != stan(inline)

    # Idempotent / disjoint: re-splitting a fragment is a no-op, and the prior
    # fragment has no structural part (and vice versa).
    @test opkeys(priors_of(priors_of(inline))) == opkeys(priors_of(inline))
    @test opkeys(structure_of(structure_of(inline))) == opkeys(structure_of(inline))
    @test isempty(opkeys(priors_of(structure_of(inline))))
    @test isempty(opkeys(structure_of(priors_of(inline))))
end

println("split_models.jl: all testsets passed")
