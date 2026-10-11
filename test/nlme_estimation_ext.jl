# The NLMEEstimation.jl protocol for RK-lowered BRM models
# (ext/BayesianRegressionModelsNLMEEstimationExt.jl): batched evaluation and the
# Newton posthoc estimator, checked against the analytic oracle of
# nlme_fixtures.jl.
#
# NLMEEstimation.jl is not published yet, so it is not part of test/Project.toml.
# Run this file from an environment that adds it to the test environment; see the
# "NLME estimators" section of test/README.md. It refuses to run without
# NLMEEstimation rather than skipping silently.
Base.find_package("NLMEEstimation") === nothing && error(
    "test/nlme_estimation_ext.jl needs NLMEEstimation.jl in the active environment; " *
    "see the header of this file")

using NLMEEstimation
using LinearAlgebra
using Random
include(joinpath(@__DIR__, "nlme_fixtures.jl"))

const NLMEEXT = Base.get_extension(BRM, :BayesianRegressionModelsNLMEEstimationExt)

# A realistic population point for NLME_PLATE_DATA: V ≈ 5, CL ≈ 1.2.
function plate_pk_point(m)
    named = Dict(Symbol("pop_log_CL.beta_pop.1") => 0.2,
        Symbol("pop_log_CL.beta_pop.2") => 0.1, :log_V_Intercept => 1.6)
    θ = [named[c] for c in m.view.coordinates[m.theta]]
    θ, [0.0]
end

# Subject columns placed into the all-subject η matrix the oracle takes.
function full_eta(m, subjects, H)
    full = zeros(size(H, 1), length(m.view.levels))
    full[:, subjects] .= H
    full
end

@stestset "the extension loads with NLMEEstimation" begin
    @test NLMEEXT !== nothing
end

@stestset "batched protocol methods match the analytic oracle" begin
    m = brm_nlme_model(RKBRMI(nlme_plate_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    n = nsubjects(m)
    @test n == 3
    @test nlme_layout(m) == NLMELayout(; ntheta=3, nsigma=1, eta_blocks=[2])
    @test evaluation_style(m) isa BatchedEvaluation
    @test provides_eta_hessian(m)
    names = parameter_names(m)
    @test length(names.theta) == 3 && length(names.sigma) == 1 && length(names.eta) == 2
    θ, σ = plate_pk_point(m)
    for subjects in ([1, 2, 3], [3, 1], [2])
        H = 0.1 .* reshape(collect(1.0:(2length(subjects))), 2, :) .- 0.2
        full = full_eta(m, subjects, H)
        values = conditional_loglikelihoods(m, subjects, θ, σ, H)
        @test values ≈ plate_pk_oracle(m, θ, σ, full)[subjects] rtol=1e-12
        ℓ, G = conditional_loglikelihoods_and_gradients(m, subjects, θ, σ, H)
        @test ℓ == values
        @test G ≈ plate_pk_oracle_gradients(m, θ, σ, full)[:, subjects] rtol=1e-10
        blocks = conditional_loglikelihood_hessian_blocks(m, subjects, θ, σ, H)
        oracle = plate_pk_oracle_hessians(m, θ, σ, full)[subjects]
        @test length(blocks) == length(subjects)
        @test all(isapprox.(blocks, oracle; rtol=1e-9, atol=1e-9))
    end
    # The per-subject entry points use one plate cell, not a batch of one.
    η = [0.05, -0.1]
    oracle_H = full_eta(m, [2], reshape(η, 2, 1))
    @test conditional_loglikelihood(m, 2, θ, σ, η) ≈
        plate_pk_oracle(m, θ, σ, oracle_H)[2] rtol=1e-12
    value, gradient = conditional_loglikelihood_and_gradient(m, 2, θ, σ, η)
    @test value ≈ plate_pk_oracle(m, θ, σ, oracle_H)[2] rtol=1e-12
    @test gradient ≈ plate_pk_oracle_gradients(m, θ, σ, oracle_H)[:, 2] rtol=1e-10
    @test_throws ArgumentError conditional_loglikelihoods(m, [1, 1], θ, σ, zeros(2, 2))
    @test_throws ArgumentError conditional_loglikelihoods(m, [n + 1], θ, σ, zeros(2, 1))
    @test_throws DimensionMismatch conditional_loglikelihoods(m, [1, 2], θ, σ, zeros(2, 1))
end

@stestset "a singleton protocol batch uses subject-local RK work" begin
    measured = map((3, 9, 27)) do n
        d = NLME_PLATE_DATA
        data = (; subject=["s$j" for j in 1:n],
            dose=[d.dose[mod1(j, 3)] for j in 1:n],
            weight=[d.weight[mod1(j, 3)] for j in 1:n],
            t=[d.t[mod1(j, 3)] for j in 1:n],
            dv=[d.dv[mod1(j, 3)] for j in 1:n])
        m = brm_nlme_model(RKBRMI(nlme_plate_pk(data)); ad_backend=NLME_AD)
        θ, σ = plate_pk_point(m)
        i = findfirst(==("s2"), m.view.levels)
        H = reshape([0.05, -0.1], 2, 1)
        values() = conditional_loglikelihoods(m, [i], θ, σ, H)
        gradients() = conditional_loglikelihoods_and_gradients(m, [i], θ, σ, H)
        values(); gradients()
        primal_bytes = minimum(@allocated(values()) for _ in 1:3)
        gradient_bytes = minimum(@allocated(gradients()) for _ in 1:3)
        (; result=gradients(), primal_bytes, gradient_bytes)
    end
    @test all(x -> isapprox(x.result[1], measured[1].result[1]; rtol=1e-12), measured)
    @test all(x -> isapprox(x.result[2], measured[1].result[2]; rtol=1e-10), measured)
    @test maximum(x.primal_bytes for x in measured) <=
        1.2minimum(x.primal_bytes for x in measured)
    @test maximum(x.gradient_bytes for x in measured) <=
        1.2minimum(x.gradient_bytes for x in measured)
end

@stestset "declared mu-referencing holds on the density" begin
    m = brm_nlme_model(RKBRMI(nlme_plate_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    mr = mu_referencing(m)
    @test mr isa MuReferencing
    @test sort(parameter_names(m).theta[mr.theta_indices]) ==
        ["log_V_Intercept", "pop_log_CL.beta_pop.1", "pop_log_CL.beta_pop.2"]
    cl_margin = findfirst(b -> b.predictor === :log_CL, m.view.block_margins)
    weight_column = findfirst(==("pop_log_CL.beta_pop.2"),
        parameter_names(m).theta[mr.theta_indices])
    @test [X[cl_margin, weight_column] for X in mr.design] ==
        NLME_PLATE_DATA.weight
    θ, σ = plate_pk_point(m)
    δ = [0.3, -0.2, 0.4]
    θs = copy(θ); θs[mr.theta_indices] .+= δ
    subjects = collect(1:nsubjects(m))
    H = repeat([0.1, 0.2], 1, length(subjects))
    Hs = H .- reduce(hcat, [mr.design[i] * δ for i in subjects])
    @test conditional_loglikelihoods(m, subjects, θs, σ, Hs) ≈
        conditional_loglikelihoods(m, subjects, θ, σ, H) rtol=1e-12
    for j in eachindex(δ)
        shift = zeros(length(δ)); shift[j] = δ[j]
        shifted = copy(θ); shifted[mr.theta_indices] .+= shift
        effects = H .- reduce(hcat, [mr.design[i] * shift for i in subjects])
        @test conditional_loglikelihoods(m, subjects, shifted, σ, effects) ≈
            conditional_loglikelihoods(m, subjects, θ, σ, H) rtol=1e-12
    end
end

@stestset "check_protocol accepts the model" begin
    m = brm_nlme_model(RKBRMI(nlme_plate_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    θ, σ = plate_pk_point(m)
    ω = omega_parameters(nlme_layout(m), [0.09 0.0; 0.0 0.04])
    @test check_protocol(m, θ, σ, ω; η=[0.1 -0.05 0.2; -0.1 0.15 0.05])
end

@stestset "SAEM differentiates the batched conditional density" begin
    m = brm_nlme_model(RKBRMI(nlme_plate_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    θ, σ = plate_pk_point(m)
    ω = omega_parameters(nlme_layout(m), [0.09 0.0; 0.0 0.04])
    @test NLMEEstimation.ad_backend(m) isa AutoEnzyme
    # Three subjects now identify three mu-referenced coefficients, so this
    # tiny AD smoke test needs a covariance floor in its Gaussian M-step.
    fit = saem(m, θ, σ, ω; rng=Random.Xoshiro(2026),
        options=SAEMOptions(iterations=3, burnin=1, mcmc_burnin=0,
            adapt_iterations=0, numeric_maxiter=2, mstep_interval=3,
            omega_floor=0.05))
    @test fit.kernel == :standard_mh
    @test all(isfinite, (fit.theta..., fit.sigma..., fit.omega...))
    @test size(fit.eta) == (2, nsubjects(m))
end

@stestset "posthoc finds the oracle's complete-data modes" begin
    m = brm_nlme_model(RKBRMI(nlme_plate_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    θ, σ = plate_pk_point(m)
    ω = omega_parameters(nlme_layout(m), [0.09 0.02; 0.02 0.04])
    prior = re_prior(m, ω)
    result = empirical_bayes(m, θ, σ, ω)
    @test result.method isa NewtonEBE
    @test all(result.converged)
    @test result.hessians !== nothing
    @test length(ebe_covariances(result)) == nsubjects(m)
    H = result.modes
    quasi = empirical_bayes(m, θ, σ, ω; method=QuasiNewtonEBE())
    @test all(quasi.converged)
    @test H ≈ quasi.modes rtol=1e-5 atol=1e-7
    @test result.conditional_loglikelihood ≈ plate_pk_oracle(m, θ, σ, H) rtol=1e-12
    oracle_hessians = plate_pk_oracle_hessians(m, θ, σ, H)
    @test all(isapprox.(result.hessians,
        [B - prior.precision for B in oracle_hessians]; rtol=1e-8, atol=1e-8))
    @test all(isapprox.(ebe_covariances(result),
        [inv(-B + prior.precision) for B in oracle_hessians]; rtol=1e-8, atol=1e-8))
    # Stationary for the oracle's complete-data density: analytic likelihood
    # gradient plus the Gaussian prior's, standardized as the solver measures it.
    G = plate_pk_oracle_gradients(m, θ, σ, H) .- prior.precision * H
    @test maximum(abs, prior.cholesky_factor' * G) <= 1e-6
    # A maximum, not merely stationary: every nearby point is lower.
    f(Hc, i) = plate_pk_oracle(m, θ, σ, Hc)[i] + re_logprior(prior, Hc[:, i])
    for i in axes(H, 2), d in ([1e-3, 0.0], [0.0, 1e-3], [-1e-3, 1e-3])
        moved = copy(H); moved[:, i] .+= d
        @test f(moved, i) < f(H, i)
    end
    # A subset of subjects gives the same modes for those subjects.
    subset = empirical_bayes(m, θ, σ, ω; subjects=[3, 1])
    @test subset.modes ≈ H[:, [3, 1]] rtol=1e-6 atol=1e-8
end

@stestset "batched Gaussian observations and colored η Jacobians" begin
    d = NLME_RAGGED_DATA
    m = brm_nlme_model(RKBRMI(nlme_ragged_pk(d)); ad_backend=NLME_AD)
    @test observation_model(m) isa GaussianObservations
    θ = [0.4, 2.1]
    σ = [log(0.8)]
    H = [0.15 -0.2 0.05; -0.1 0.25 0.3]
    population(pred) = θ[findfirst(==(only(r.population for r in m.view.mu_references
        if r.predictor === pred)), m.theta)]
    margin(pred) = findfirst(b -> b.predictor === pred, m.view.block_margins)
    kcl, kv = margin(:CL), margin(:V)
    function oracle(i)
        level = string(m.view.levels[i])
        r = only(findall(==(level), d.subject))
        rows = findall(==(level), d.obs_subject)
        CL = exp(population(:CL) + H[kcl, i])
        V = exp(population(:V) + H[kv, i])
        kt = (CL / V) .* d.t[rows]
        μ = d.dose[r] / V .* exp.(-kt)
        v = fill(exp(2 * only(σ)), length(rows))
        Jμ = zeros(length(rows), size(H, 1))
        Jμ[:, kcl] .= μ .* -kt
        Jμ[:, kv] .= μ .* (kt .- 1)
        (d.y[rows], μ, v, Jμ, zeros(size(Jμ)))
    end
    for subjects in ([1, 2, 3], [3, 1], [2])
        observed = observations(m, subjects)
        moments = observation_moments(m, subjects, θ, σ, H[:, subjects])
        with_jacobians = observation_moments_and_jacobians(
            m, subjects, θ, σ, H[:, subjects])
        densities = conditional_loglikelihoods(m, subjects, θ, σ, H[:, subjects])
        for (k, i) in enumerate(subjects)
            y, μ, v, Jμ, Jv = oracle(i)
            @test observed[k] == y
            @test moments[k][1] ≈ μ rtol=1e-12
            @test moments[k][2] ≈ v rtol=1e-12
            @test with_jacobians[k][1] ≈ μ rtol=1e-12
            @test with_jacobians[k][2] ≈ v rtol=1e-12
            @test with_jacobians[k][3] ≈ Jμ rtol=1e-10 atol=1e-12
            @test with_jacobians[k][4] ≈ Jv atol=1e-12
            @test sum(logpdf.(Normal.(μ, sqrt.(v)), y)) ≈ densities[k] rtol=1e-12
        end
    end
    ω = omega_parameters(nlme_layout(m), [0.09 0.0; 0.0 0.04])
    # Decision 110evqv: the batch Gaussian route remains valid, but conformance
    # also probes the single-subject route. A ragged join has no subject cell.
    err = try check_protocol(m, θ, σ, ω; η=H); nothing catch e; e end
    @test err isa BRMNLMEViewError
    @test occursin("ragged-join", err.message)
end

@stestset "Gaussian moments and colored η Jacobians for array-valued plate cells" begin
    d = NLME_PLATE_DATA
    m = brm_nlme_model(RKBRMI(nlme_plate_pk(d)); ad_backend=NLME_AD)
    @test observation_model(m) isa GaussianObservations
    θ, σ = plate_pk_point(m)
    H = [0.15 -0.2 0.05; -0.1 0.25 0.3]
    θn = Dict(zip(m.view.coordinates[m.theta], θ))
    kcl = findfirst(b -> b.predictor === :log_CL, m.view.block_margins)
    kv = findfirst(b -> b.predictor === :log_V, m.view.block_margins)
    function oracle(i)
        r = only(m.view.rows[i])
        CL = exp(θn[Symbol("pop_log_CL.beta_pop.1")] +
            θn[Symbol("pop_log_CL.beta_pop.2")] * d.weight[r] + H[kcl, i])
        V = exp(θn[:log_V_Intercept] + H[kv, i])
        kt = (CL / V) .* d.t[r]
        μ = d.dose[r] / V .* exp.(-kt)
        v = fill(exp(2 * only(σ)), length(μ))
        Jμ = zeros(length(μ), size(H, 1))
        Jμ[:, kcl] .= -kt .* μ
        Jμ[:, kv] .= (kt .- 1) .* μ
        (d.dv[r], μ, v, Jμ, zeros(size(Jμ)))
    end
    for subjects in ([1, 2, 3], [3, 1], [2])
        observed = observations(m, subjects)
        moments = observation_moments(m, subjects, θ, σ, H[:, subjects])
        with_jacobians = observation_moments_and_jacobians(
            m, subjects, θ, σ, H[:, subjects])
        densities = conditional_loglikelihoods(m, subjects, θ, σ, H[:, subjects])
        for (k, i) in enumerate(subjects)
            y, μ, v, Jμ, Jv = oracle(i)
            @test observed[k] == y
            @test moments[k][1] ≈ μ rtol=1e-12
            @test moments[k][2] ≈ v rtol=1e-12
            @test with_jacobians[k][1] ≈ μ rtol=1e-12
            @test with_jacobians[k][2] ≈ v rtol=1e-12
            @test with_jacobians[k][3] ≈ Jμ rtol=1e-10 atol=1e-12
            @test with_jacobians[k][4] ≈ Jv atol=1e-12
            @test sum(logpdf.(Normal.(μ, sqrt.(v)), y)) ≈ densities[k] rtol=1e-12
        end
    end
end

@stestset "per-cell Gaussian scales expand over their observations" begin
    d = (; subject=["a", "b", "c"],
        y=[[1.0, 2.0], [3.0], [1.5, 0.4, -0.2]])
    brmi = @brm d begin
        mu ~ 1 + (1 | subject)
        log(sigma) ~ 1 + (1 | subject)
        @plate for i in eachindex(mu)
            pred[i] = mu[i]
            y[i] ~ normal(pred[i], sigma[i])
        end
    end
    m = brm_nlme_model(RKBRMI(brmi); ad_backend=NLME_AD)
    @test observation_model(m) isa GaussianObservations
    θ = [name === :mu_Intercept ? 0.5 : log(0.8)
        for name in m.view.coordinates[m.theta]]
    H = [0.1 -0.2 0.3; -0.05 0.2 -0.1]
    kmu = findfirst(b -> b.predictor === :mu, m.view.block_margins)
    ksigma = findfirst(b -> b.predictor === :sigma, m.view.block_margins)
    for subjects in ([1, 2, 3], [3, 1], [2])
        y = observations(m, subjects)
        result = observation_moments_and_jacobians(m, subjects, θ, Float64[], H[:, subjects])
        densities = conditional_loglikelihoods(m, subjects, θ, Float64[], H[:, subjects])
        for (k, i) in enumerate(subjects)
            r = only(m.view.rows[i])
            μ = fill(0.5 + H[kmu, i], length(d.y[r]))
            v = fill(exp(2 * (log(0.8) + H[ksigma, i])), length(d.y[r]))
            Jμ = zeros(length(μ), 2); Jμ[:, kmu] .= 1
            Jv = zeros(length(v), 2); Jv[:, ksigma] .= 2 .* v
            @test y[k] == d.y[r]
            @test result[k][1] ≈ μ rtol=1e-12
            @test result[k][2] ≈ v rtol=1e-12
            @test result[k][3] ≈ Jμ rtol=1e-10 atol=1e-12
            @test result[k][4] ≈ Jv rtol=1e-10 atol=1e-12
            @test sum(logpdf.(Normal.(μ, sqrt.(v)), y[k])) ≈ densities[k] rtol=1e-12
        end
    end
end

@stestset "a modeled Gaussian scale has a nonzero variance Jacobian" begin
    d = (; subject=["a", "b", "a", "c", "b"], y=[1.0, 2.2, 0.6, -0.4, 1.7])
    brmi = @brm d begin
        mu ~ 1 + (1 | subject)
        log(sigma) ~ 1 + (1 | subject)
        y ~ Normal(mu, sigma)
    end
    m = brm_nlme_model(RKBRMI(brmi); ad_backend=NLME_AD)
    @test observation_model(m) isa GaussianObservations
    @test isempty(m.sigma)
    θ = [name === :mu_Intercept ? 0.5 : log(0.8)
        for name in m.view.coordinates[m.theta]]
    H = [0.1 -0.2 0.3; -0.05 0.2 -0.1]
    kmu = findfirst(b -> b.predictor === :mu, m.view.block_margins)
    ksigma = findfirst(b -> b.predictor === :sigma, m.view.block_margins)
    for subjects in ([1, 2, 3], [3, 1])
        y = observations(m, subjects)
        result = observation_moments_and_jacobians(m, subjects, θ, Float64[], H[:, subjects])
        for (k, i) in enumerate(subjects)
            rows = findall(==(string(m.view.levels[i])), d.subject)
            μ = fill(0.5 + H[kmu, i], length(rows))
            v = fill(exp(2 * (log(0.8) + H[ksigma, i])), length(rows))
            Jμ = zeros(length(rows), 2); Jμ[:, kmu] .= 1
            Jv = zeros(length(rows), 2); Jv[:, ksigma] .= 2 .* v
            @test y[k] == d.y[rows]
            @test result[k][1] ≈ μ rtol=1e-12
            @test result[k][2] ≈ v rtol=1e-12
            @test result[k][3] ≈ Jμ rtol=1e-10 atol=1e-12
            @test result[k][4] ≈ Jv rtol=1e-10 atol=1e-12
        end
    end
end
