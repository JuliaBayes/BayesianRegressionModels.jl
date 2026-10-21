# test/rk_predictive.jl — posterior-predictive draws of an RKBRMI through
# RKPPL's public predictive API (rkppl-use §14), reached with `rk_bound_plan`
# (snag `no-public-route-7bb7ad36`; the user deferred design decision
# `0ctkcdm` to its recommended option).
#
# `rk_bound_plan(rk)` is the RKPPL binding every query of `rk` evaluates on
# `rk.model`; for a `reprocess` replay it binds the design. Oracles: each law's
# location is the authored graph value the response reads (`conc`, `mu`) at
# the same draw, its scale is the constrained `sigma`, and the draws take the
# design's shapes. Public synthetic models only.
#
# Run: julia --project=test test/rk_predictive.jl
using Test
using BayesianRegressionModels, Distributions, Random
using ReactiveKernels, ReactiveKernelsPPL

module PredictiveModels
using BayesianRegressionModels, Distributions
pk(d) = @brm d begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + zscale(weight) + (1 | p | subject)
    effect(log_CL, Intercept) ~ Normal(0, 1)
    sd(:, p) ~ Exponential(1)
    @plate for i in eachindex(log_CL)
        conc[i] = dose[i] .* exp.(-exp(log_CL[i]) .* t[i])
        y[i] ~ normal(conc[i], sigma)
    end
end
grouped(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + zscale(x) + (1 | g)
    y ~ Normal(mu, sigma)
end
end
const PM = PredictiveModels

const PK = (; subject=["s1", "s2", "s3", "s4", "s5"],
    weight=[60.0, 72.0, 81.0, 55.0, 90.0], dose=[100.0, 120.0, 80.0, 100.0, 150.0],
    t=[[0.5, 1.0, 2.0], [0.25, 1.5], [0.5, 1.0, 2.0, 4.0], [1.0], [0.5, 2.0]],
    y=[[1.0, 0.8, 0.5], [1.2, 0.7], [0.9, 0.7, 0.4, 0.2], [0.6], [1.3, 0.5]])
const TRAIN = (; x=collect(range(0.3, 4.1; length=9)), g=["a", "b", "c"][mod1.(1:9, 3)],
    y=sin.(1:9))

point(rk) = collect(range(-0.6, 0.8; length=rk.model.layout.total))
flat(v) = eltype(v) <: AbstractVector ? reduce(vcat, v) : v
law_means(laws) = flat([mean.(l) for l in laws])
law_scales(laws) = flat([std.(l) for l in laws])
first_value(value) = value isa Tuple ? first(value) : value

# The authored value `want` of `rk`'s graph at `u`, on its own bound data.
function graph_value(rk, want, u)
    bound = rk_bound_data(rk)
    query = ReactiveKernels.prepare(rk.model.spec;
        have=(:unconstrained, keys(bound)...), want=(want,), bound)
    first_value(Base.invokelatest(query, u))
end

@testset "the fit's binding gives its responses' shapes and laws" begin
    rk = RKBRMI(PM.pk(PK))
    u = point(rk)
    plan = rk_bound_plan(rk)
    @test rk_bound_plan(rk) === plan      # the one binding every query reads
    bound = rk_bound_data(rk)
    @test keys(bound) == Tuple(keys(plan.columns))
    @test all(bound[k] === plan.columns[k] for k in keys(bound))
    targets = predictive_targets(rk.model, plan)
    @test targets.y.kind === :cells
    @test targets.y.dims == [size(y) for y in PK.y]
    pred = prepare_predictive(rk.model, plan)
    laws = predictive_laws(pred, u)
    @test [size(l) for l in laws.y] == [size(y) for y in PK.y]
    @test law_means(laws.y) ≈ flat(graph_value(rk, :conc, u)) atol=1e-12 rtol=1e-12
    @test all(≈(constrain(rk.model.layout, u).sigma), law_scales(laws.y))
    draws = pred(u, Xoshiro(1))
    @test length.(draws.y) == length.(PK.y)
    @test draws == pred(u, Xoshiro(1))
end

@testset "a replay of new subjects draws in the caller's own vectorized cut" begin
    rk = RKBRMI(PM.pk(PK))
    u = point(rk)
    design = (; subject=["n1", "n2"], weight=[70.0, 64.0], dose=[110.0, 90.0],
        t=[[0.5, 1.0, 2.0, 6.0, 8.0], [0.25]])       # no `y`: prediction only
    replay = reprocess(rk, design; resample_groups=[:subject])
    plan = rk_bound_plan(replay)
    @test replay.model === rk.model
    targets = predictive_targets(replay.model, plan)
    @test keys(targets) == (:y,)
    @test targets.y.kind === :cells
    bound = rk_bound_data(replay)
    ports = Tuple(k for k in keys(bound) if !haskey(targets, k))
    args = Tuple(bound[k] for k in ports)
    cut = ReactiveKernels.prepare(replay.model.spec;
        have=(:unconstrained, :b_p_subject, ports...),
        want=(predictive_node(replay.model), :conc))
    B = [0.1; -0.2;;]                                  # the new subjects' effects
    value, conc = Base.invokelatest(cut, u, B, args...)
    laws = predictive_laws(value, targets)
    @test [length(l) for l in laws.y] == length.(design.t)
    @test law_means(laws.y) ≈ flat(conc) atol=1e-12 rtol=1e-12
    @test all(≈(constrain(replay.model.layout, u).sigma), law_scales(laws.y))
    @test length.(predictive_draws(value, targets, Xoshiro(2)).y) == length.(design.t)

    # Vectorized over draws: position k reads draw k's laws.
    U = hcat(u, u .+ 0.1, u .- 0.1)
    BB = cat(B, B .+ 0.05, B .- 0.05; dims=3)
    reader = ReactiveKernels.vectorize(cut; batched=(:unconstrained, :b_p_subject))
    stacked, _ = Base.invokelatest(reader, U, BB, args...)
    for k in 1:3
        _, conc_k = Base.invokelatest(cut, U[:, k], BB[:, :, k], args...)
        laws_k = predictive_laws(stacked, targets; position=k)
        @test law_means(laws_k.y) ≈ flat(conc_k) atol=1e-12 rtol=1e-12
        @test length.(predictive_draws(stacked, targets, Xoshiro(k); position=k).y) ==
            length.(design.t)
    end

    # refused: a resampled replay's subject effects are not coordinates of the
    # fit's draw, so the whole-model query (HAVE `:unconstrained` alone) has no
    # value for them; its density is refused the same way (test/rk_reprocess.jl).
    @test_throws "different program" prepare_predictive(replay.model, plan)
end

@testset "a replay of fitted subjects draws the whole model" begin
    rk = RKBRMI(PM.pk(PK))
    u = point(rk)
    design = (; subject=["s2", "s4"], weight=[72.0, 55.0], dose=[120.0, 100.0],
        t=[[0.25, 1.5, 3.0], [1.0, 2.0]])
    replay = reprocess(rk, design)
    pred = prepare_predictive(replay.model, rk_bound_plan(replay))
    laws = predictive_laws(pred, u)
    @test [length(l) for l in laws.y] == length.(design.t)
    @test law_means(laws.y) ≈ flat(graph_value(replay, :conc, u)) atol=1e-12 rtol=1e-12
    @test length.(pred(u, Xoshiro(3)).y) == length.(design.t)
end

@testset "a flat response omitted from the design takes the design's rows" begin
    rk = RKBRMI(PM.grouped(TRAIN))
    u = point(rk)
    replay = reprocess(rk, (; x=[1.0, 2.0, 3.0, 4.0], g=["a", "c", "c", "b"]))
    plan = rk_bound_plan(replay)
    @test predictive_targets(replay.model, plan).y == (kind=:array, dims=(4,), eltype=Float64)
    pred = prepare_predictive(replay.model, plan)
    laws = predictive_laws(pred, u)
    @test size(laws.y) == (4,)
    @test mean.(laws.y) ≈ graph_value(replay, :mu, u) atol=1e-12 rtol=1e-12
    @test all(≈(constrain(replay.model.layout, u).sigma), std.(laws.y))
    @test size(pred(u, Xoshiro(4)).y) == (4,)
end
