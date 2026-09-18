# test/powerscale.jl — power-scaling sensitivity acceptance.
#
# Covers the four things `src/powerscale.jl` must promise a consumer:
#   1. PSIS weights match PSIS.jl (the test env already carries it), with
#      loud errors for non-finite input and too-few draws.
#   2. CJS matches its published calibration (EABM: 0.05 ~= 0.3-sd mean shift
#      on a standard Normal) and agrees between its weighted-ECDF and binned
#      paths; validation errors are loud.
#   3. The sensitivity summary recovers the analytic Normal perturbation
#      (independent direct-sampling oracle) and flags the conflict direction.
#   4. The SBBRMI assembly is SOUND: the Julia-side joint prior plus the Stan
#      twins reconstruct the BridgeStan target up to an additive constant on
#      every admitted shape (the identity battery), the coverage gate names
#      everything outside, and missing generated quantities error loudly.
#
# Run: julia --project=test test/powerscale.jl

using Test
using BayesianRegressionModels
using BridgeStan
using StanBlocks
using Distributions
using PSIS
using Random
using Statistics
using Logging
using LinearAlgebra
using LogDensityProblems

const BRM = BayesianRegressionModels
const BS = BridgeStan

# PSIS.jl warns on bad shapes; the cross-check pins VALUES, not warnings.
function ref_psis(logr)
    with_logger(NullLogger()) do
        PSIS.psis(Vector{Float64}(logr))
    end
end

@testset "psis weights match PSIS.jl" begin
    rng = Xoshiro(20260918)
    cases = Dict(
        "mild" => randn(rng, 500),
        "skewed" => rand(rng, LogNormal(0.0, 1.0), 500) .- 1.5,
        "heavy tail" => vcat(randn(rng, 490), 8 .+ randn(rng, 10)),
    )
    for (label, logr) in cases
        mine = brm_psis_weights(logr)
        ref = ref_psis(logr)
        @test mine.weights ≈ ref.weights rtol = 1e-8
        @test mine.pareto_k ≈ PSIS.pareto_shape(ref) rtol = 1e-8
        @test mine.tail_length == ref.tail_length
        @test sum(mine.weights) ≈ 1.0
        @test all(>(0), mine.weights)
    end
    # ESS is the textbook L2 form on the SMOOTHED weights.
    logr = cases["mild"]
    mine = brm_psis_weights(logr)
    @test mine.ess ≈ 1 / sum(abs2, mine.weights)
    reffed = brm_psis_weights(logr; reff=0.5)
    @test reffed.ess ≈ 0.5 / sum(abs2, reffed.weights)

    @test_throws ErrorException brm_psis_weights(fill(0.1, 20))
    bad = randn(rng, 100)
    bad[7] = Inf
    @test_throws ErrorException brm_psis_weights(bad)
    bad[7] = NaN
    @test_throws ErrorException brm_psis_weights(bad)
    @test_throws ErrorException brm_psis_weights(randn(rng, 100); reff=0.0)
    @test_throws ErrorException brm_psis_weights(randn(rng, 100); reff=NaN)
end

@testset "cjs matches its published calibration" begin
    S = 200_000
    x = randn(Xoshiro(11), S)
    # EABM sensitivity chapter: CJS ~= 0.05 corresponds to a mean shift of
    # ~= 0.3 standard deviations on a standard Normal. The book's figure is
    # approximate, so this pins the ORDER, not the digit.
    shifted = 0.3 .+ randn(Xoshiro(12), S)
    c = brm_cjs_dist(x, shifted)
    @test 0.02 < c < 0.10
    # Same distribution, independent samples: near zero.
    same = brm_cjs_dist(x, randn(Xoshiro(13), S))
    @test same < 0.02
    # Monotone in the shift and symmetric.
    further = brm_cjs_dist(x, 0.6 .+ randn(Xoshiro(14), S))
    @test further > c
    @test brm_cjs_dist(x, shifted) ≈ brm_cjs_dist(shifted, x)
    # Identical draws: exactly zero, weights or not.
    @test brm_cjs_dist(x, x) == 0.0
    # Differently-normalised uniform weights agree up to float noise.
    @test brm_cjs_dist(x, x; y_weights=fill(2.0, S)) ≈ 0.0 atol = 1e-9
    # A point mass against itself is 0 (R priorsense NaNs here).
    @test brm_cjs_dist(fill(1.5, 50), fill(1.5, 50)) == 0.0
    # Unsigned: invariant to a joint sign flip.
    @test brm_cjs_dist(x, shifted) ≈ brm_cjs_dist(-x, -shifted)

    # Weighted-ECDF path versus binned path on the SAME perturbation: smooth
    # weights tilting N(0,1) toward N(0.4,1). Both estimate one truth.
    Sb = 20_000
    xb = randn(Xoshiro(21), Sb)
    tilt = exp.(0.4 .* xb .- 0.08)
    weighted = brm_cjs_dist(xb, xb; y_weights=tilt)
    resampled =
        xb[rand(Xoshiro(22), Distributions.Categorical(tilt ./ sum(tilt)), Sb)]
    binned = brm_cjs_dist(xb, resampled)
    @test abs(weighted - binned) < 0.03

    @test_throws ErrorException brm_cjs_dist([1.0], [1.0, 2.0])
    bad = [1.0, 2.0, 3.0]
    bad[2] = NaN
    @test_throws ErrorException brm_cjs_dist(bad, [1.0, 2.0, 3.0])
    @test_throws ErrorException brm_cjs_dist(
        [1.0, 2.0], [1.0, 2.0]; y_weights=[0.5, -0.5])
    @test_throws ErrorException brm_cjs_dist(
        [1.0, 2.0], [1.0, 2.0]; y_weights=[0.0, 0.0])
    @test_throws DimensionMismatch brm_cjs_dist(
        [1.0, 2.0], [1.0, 2.0]; y_weights=[1.0])
end

@testset "powerscale weights follow alpha" begin
    rng = Xoshiro(31)
    S = 2_000
    draws = randn(rng, S)
    ℓ = logpdf.(Normal(0.0, 1.0), draws)
    # alpha == 1 is the base posterior, exactly as priorsense.
    base = brm_powerscale_weights(ℓ; alpha=1)
    @test base.weights ≈ fill(1 / S, S)
    @test base.pareto_k == -Inf
    # Strengthening a standard-Normal prior compresses toward 0 ...
    up = brm_powerscale_weights(ℓ; alpha=1.25)
    @test sum(up.weights .* abs2.(draws)) < sum(abs2.(draws)) / S
    # ... and weakening stretches away from it.
    down = brm_powerscale_weights(ℓ; alpha=0.8)
    @test sum(down.weights .* abs2.(draws)) > sum(abs2.(draws)) / S
    @test_throws ErrorException brm_powerscale_weights(fill(3.0, S); alpha=1.1)
    bad = copy(ℓ)
    bad[5] = -Inf
    @test_throws ErrorException brm_powerscale_weights(bad; alpha=1.1)
    @test_throws ErrorException brm_powerscale_weights(ℓ; alpha=-0.5)
end

@testset "sensitivity recovers the analytic Normal perturbation" begin
    # Consistent triple: prior == likelihood == N(0, sqrt(2)), so the
    # posterior is N(0, 1). Power-scaling the prior by alpha gives
    # N(0, sqrt(2 / (1 + alpha))) in closed form — the INDEPENDENT oracle is
    # direct sampling of that Normal, scored through the binned CJS path.
    S = 20_000
    draws = randn(Xoshiro(41), S)
    ℓ = logpdf.(Normal(0.0, sqrt(2.0)), draws)
    α = 1.01
    w_hi = brm_powerscale_weights(ℓ; alpha=α).weights
    weighted = brm_cjs_dist(draws, draws; y_weights=w_hi)
    oracle = sqrt(2 / (1 + α)) .* randn(Xoshiro(42), S)
    direct = brm_cjs_dist(draws, oracle)
    @test abs(weighted - direct) < 0.02

    # draws × coordinates orientation is enforced, like every diagnostic:
    # one row against S-length densities is a transposed caller matrix.
    @test_throws DimensionMismatch brm_powerscale_sensitivity(
        reshape(draws, 1, S), ℓ, ℓ, ["v"])
    names = ["v"]
    s = brm_powerscale_sensitivity(reshape(draws, S, 1), ℓ, ℓ, names)
    @test s.variables == [:v]
    @test all(isfinite, s.prior) && all(isfinite, s.likelihood)
    @test length(s.diagnosis) == 1
    @test s.threshold == 0.05
    # The v1 summary always scales the joint densities it is given.
    @test sprint(show, MIME("text/plain"), s) isa String
end

@testset "sensitivity flags prior-data conflict" begin
    # Steep prior far from the posterior mass + steep likelihood: both
    # perturbations move the marginal a lot.
    S = 20_000
    draws = 1.0 .+ 0.5 .* randn(Xoshiro(51), S)
    ℓp = logpdf.(Normal(0.0, 1.0), draws)
    ℓl = logpdf.(Normal(1.2, 0.4), draws)
    s = brm_powerscale_sensitivity(
        reshape(draws, S, 1), ℓp, ℓl, ["beta"])
    @test s.prior[1] > 0.05 && s.likelihood[1] > 0.05
    @test s.diagnosis == [:prior_data_conflict]
    # The threshold is honoured, not decorative.
    calm = brm_powerscale_sensitivity(
        reshape(draws, S, 1), ℓp, ℓl, ["beta"]; threshold=1e9)
    @test calm.diagnosis == [:none]
    # Robust direction: the conflict triple moves strictly more than a calm
    # triple (flat prior, likelihood matching the posterior).
    ℓflat = logpdf.(Normal(0.0, 100.0), draws)
    ℓmatch = logpdf.(Normal(1.0, 0.5), draws)
    t = brm_powerscale_sensitivity(
        reshape(draws, S, 1), ℓflat, ℓmatch, ["beta"])
    @test s.prior[1] > 5 * t.prior[1]
    # Selection, validation, and the multi-variable path.
    two = hcat(draws, draws)
    s2 = brm_powerscale_sensitivity(two, ℓp, ℓl, ["a", "b"])
    @test s2.variables == [:a, :b]
    @test s2.prior[1] ≈ s.prior[1]
    sel = brm_powerscale_sensitivity(two, ℓp, ℓl, ["a", "b"]; variables=[:b])
    @test sel.variables == [:b]
    @test_throws ErrorException brm_powerscale_sensitivity(
        two, ℓp, ℓl, ["a", "b"]; variables=[:nope])
    @test_throws DimensionMismatch brm_powerscale_sensitivity(
        two, ℓp[1:(S - 1)], ℓl, ["a", "b"])
    @test_throws ErrorException brm_powerscale_sensitivity(
        two, ℓp, ℓl, ["a", "b"]; lower_alpha=1.5)
end

# The gate runs before any draw is touched, so rejected shapes are provoked
# with dummy draws — no compile, no sampling.
function gate_message(builder, df)
    sb = SBBRMI(builder(df); mod=@__MODULE__)
    try
        brm_powerscale_inputs(sb, zeros(2, 2), ["a", "b"])
    catch err
        return sprint(showerror, err)
    end
    error("gate test: expected brm_powerscale_inputs to throw")
end

@testset "coverage gate names everything outside" begin
    df = (; x=[0.0, 1.0, 2.0, 3.0], g=[1, 1, 2, 2],
           y=[0.2, 0.4, 0.6, 0.8], n=[10, 10, 10, 10])

    ranef = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | g)
        y ~ Normal(mu, sigma)
    end
    @test occursin("`|`", gate_message(ranef, df))

    hsgp_builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + hsgp(x)
        y ~ Normal(mu, sigma)
    end
    @test occursin("hsgp", gate_message(hsgp_builder, df))

    r2d2_builder = @brm begin
        effect(mu, :) ~ r2d2(R2=Beta(1, 1))
        mu ~ 1 + x
        y ~ Normal(mu, 1.0)
    end
    @test occursin("r2d2", gate_message(r2d2_builder, df))

    shared = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | p | g)
        sd(:, p) ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    @test occursin("sd", gate_message(shared, df))

    simplex = @brm begin
        s ~ Dirichlet(3, 1.0)
        mu ~ 1 + x
        y ~ Normal(mu, s[1])
    end
    @test occursin("Dirichlet", gate_message(simplex, df))

    gamma_prior = @brm begin
        tau ~ Gamma(2.0, 1.0)
        mu ~ 1 + x
        y ~ Normal(mu, tau)
    end
    @test occursin("Gamma", gate_message(gamma_prior, df))

    # Poisson is a LIKELIHOOD rejection (v1 admits Normal/BernoulliLogit):
    # the boundary is deliberate and this pins it until it grows.
    poisson = @brm begin
        mu ~ 1 + x
        n ~ Poisson(mu)
    end
    @test occursin("Poisson", gate_message(poisson, df))

    hierarchical = @brm begin
        tau ~ Exponential(1)
        sigma ~ Exponential(tau)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    @test occursin("non-constant", gate_message(hierarchical, df))

    gp_builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + gp(x)
        y ~ Normal(mu, sigma)
    end
    @test occursin("gp", gate_message(gp_builder, df))
end

# ---- BridgeStan identity battery -------------------------------------------
#
# For each admitted shape: the Julia-side joint prior plus the Stan twins
# must reconstruct the BridgeStan target, up to ONE additive constant (draw
# independent by construction — normalisation constants cancel in importance
# weights, but a MISSING prior term would show up as draw-varying residual).
# Hand unconstrained draws keep this deterministic; no MCMC is involved.
function battery_identity(sb, df; sigma_unc::Union{Symbol,Nothing})
    d = brm_descriptor(sb)
    @test :pointwise_loglik in Symbol[op.name for op in d.operations]
    prob = brm_execute(d, :fit)
    n = LogDensityProblems.dimension(prob)
    S = 25
    q = 0.3 .* randn(Xoshiro(61), S, n)
    full_names = BS.param_names(
        prob.model; include_tp=true, include_gq=true)
    constrained = permutedims(reduce(hcat, [
        BS.param_constrain(prob.model, collect(row);
                           include_tp=true, include_gq=true,
                           rng=BS.new_rng(prob.model, 10_000 + i))
        for (i, row) in enumerate(eachrow(q))]))
    inputs = brm_powerscale_inputs(sb, constrained, full_names)
    @test all(isfinite, inputs.log_prior)
    @test all(isfinite, inputs.log_lik)
    target = [LogDensityProblems.logdensity(prob, collect(row))
              for row in eachrow(q)]
    jac = if isnothing(sigma_unc)
        zeros(S)
    else
        unc_names = BS.param_unc_names(prob.model)
        j = findfirst(==(string(sigma_unc)), unc_names)
        @test !isnothing(j)
        # `<lower=0>` sigma: q = log(sigma), so log|J| = q.
        vec(q[:, j])
    end
    julia = inputs.log_prior .+ inputs.log_lik
    stan = target .- jac
    @test maximum(abs, (julia .- mean(julia)) .- (stan .- mean(stan))) < 1e-6
    return d, constrained, full_names, inputs
end

@testset "identity battery — gaussian with overrides and cell means" begin
    df = (; x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, 0.25],
            g=[1, 1, 2, 2, 3, 3, 1, 2],
            h=[1, 2, 1, 2, 1, 2, 1, 2],
            y=[0.2, 0.1, -0.3, 0.4, 0.2, -0.2, 0.1, 0.0],
            z=[1.2, 1.1, 0.7, 1.4, 1.2, 0.8, 1.1, 1.0])
    builder = @brm begin
        sigma ~ Exponential(1)
        effect(mu, x) ~ Normal(0, 2)
        effect(mu, g) ~ Normal(0.5, 1.5)
        effect(mu2, h_lvl_2) ~ Normal(1.0, 0.5)
        mu ~ 1 + x + g
        mu2 ~ 0 + h
        y ~ Normal(mu, sigma)
        z ~ Normal(mu2, sigma)
    end
    sb = SBBRMI(builder(df); mod=@__MODULE__)
    d, constrained, full_names, inputs =
        battery_identity(sb, df; sigma_unc=:sigma)
    # The h block really is cell-mean coded — otherwise this test would prove
    # the wrong coding.
    h_res = brm_population_effect_coordinates(d, :mu2, full_names; coefficient=:h)
    @test h_res.coding === :cellmeans
    @test length(h_res.coordinates) == 2
    # Default variables: parameters in, twins and sampler diagnostics out.
    @test :sigma in inputs.variables
    @test !any(v -> endswith(string(v), "__"), inputs.variables)
    @test !any(v -> occursin("likelihood", string(v)), inputs.variables)
    @test !any(v -> occursin("_gen", string(v)), inputs.variables)
    # The summary runs end to end on the assembled triple.
    sens = brm_powerscale_sensitivity(constrained, inputs.log_prior,
                                   inputs.log_lik, full_names;
                                   variables=inputs.variables)
    @test sens.variables == inputs.variables
    @test all(isfinite, sens.prior) && all(isfinite, sens.likelihood)
    @test all(isfinite, (sens.pareto_k.prior_lower, sens.pareto_k.prior_upper,
                         sens.pareto_k.likelihood_lower,
                         sens.pareto_k.likelihood_upper))
    rendered = sprint(show, MIME("text/plain"), sens)
    @test occursin("Power-scaling sensitivity", rendered)
    @test occursin("Pareto-k", rendered)
    @test inputs.descriptor isa BRMDescriptor
end

@testset "identity battery — bernoullilogit, jacobian-free" begin
    df = (; x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, 0.25],
            y=[0, 1, 0, 1, 1, 0, 1, 0])
    builder = @brm begin
        effect(mu, Intercept) ~ Normal(0, 5)
        mu ~ 1 + x
        y ~ BernoulliLogit(mu)
    end
    sb = SBBRMI(builder(df); mod=@__MODULE__)
    battery_identity(sb, df; sigma_unc=nothing)
end

@testset "missing generated quantities error loudly" begin
    df = (; x=[0.0, 1.0, 2.0, 3.0], y=[0.2, 0.4, 0.6, 0.8])
    builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(builder(df); mod=@__MODULE__)
    d = brm_descriptor(sb)
    # This is exactly the Tier-1 GLM-fused shape (pure-population Gaussian):
    # fusion must preserve the pointwise twins power-scaling assembles.
    @test :pointwise_loglik in Symbol[op.name for op in d.operations]
    prob = brm_execute(d, :fit)
    n = LogDensityProblems.dimension(prob)
    q = 0.3 .* randn(Xoshiro(71), 5, n)
    bare_names = BS.param_names(prob.model; include_tp=true, include_gq=false)
    bare = permutedims(reduce(hcat, [
        BS.param_constrain(prob.model, collect(row);
                           include_tp=true, include_gq=false)
        for row in eachrow(q)]))
    msg = try
        brm_powerscale_inputs(sb, bare, bare_names)
        "NO-THROW"
    catch err
        sprint(showerror, err)
    end
    @test occursin("generated quantities", msg)
end
