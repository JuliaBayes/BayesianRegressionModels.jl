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

@stestset "population priors use the authored laws in NLME coordinates" begin
    data = NLME_PLATE_DATA
    brmi = @brm data begin
        sigma ~ Exponential(2)
        log_CL ~ 1 + weight + (1 | pk | subject)
        log_V ~ 1 + (1 | pk | subject)
        effect(log_CL, Intercept) ~ Normal(0.3, 0.8)
        effect(log_CL, weight) ~ Normal(-0.1, 0.4)
        effect(log_V, Intercept) ~ Normal(1.5, 0.6)
        sd(:, pk) ~ Exponential(0.7)
        cor(:, pk) ~ LKJCholesky(2, 2)
        @plate for i in eachindex(log_CL)
            CL = exp(log_CL[i])
            Vc = exp(log_V[i])
            pred[i] = dose[i] / Vc .* exp.(-(CL / Vc) .* t[i])
            dv[i] ~ normal(pred[i], sigma)
        end
    end
    m = brm_nlme_model(RKBRMI(brmi); ad_backend=NLME_AD)
    named = Dict(Symbol("pop_log_CL.beta_pop.1") => 0.25,
        Symbol("pop_log_CL.beta_pop.2") => -0.15, :log_V_Intercept => 1.4)
    θ = [named[name] for name in m.view.coordinates[m.theta]]
    σ = [log(0.8)]
    ω = [log(0.7), 0.15, log(0.6)]
    a, b, c = exp(ω[1]), ω[2], exp(ω[3])
    τ = hypot(b, c)
    ρ = b / τ
    # K=2 LKJ(η=2) has density 3/4*(1-ρ²) on its free Cholesky entry.
    # The change from NLME's log-Cholesky ω to (τ₁, τ₂, ρ) has Jacobian
    # log(a) + 2log(c/τ). BRM's standardized z prior belongs to NLME instead.
    oracle = logpdf(Normal(0.3, 0.8), named[Symbol("pop_log_CL.beta_pop.1")]) +
        logpdf(Normal(-0.1, 0.4), named[Symbol("pop_log_CL.beta_pop.2")]) +
        logpdf(Normal(1.5, 0.6), named[:log_V_Intercept]) +
        logpdf(Exponential(2), exp(only(σ))) + only(σ) +
        logpdf(Exponential(0.7), a) + logpdf(Exponential(0.7), τ) +
        log(3 / 4) + log1p(-ρ^2) + log(a) + 2log(c / τ)
    value = population_logprior(m, θ, σ, ω)
    @test value ≈ oracle atol=1e-10
    v, gθ, gσ, gω = population_logprior_and_gradient(m, θ, σ, ω)
    @test v ≈ value atol=1e-10
    θgradient = Dict(Symbol("pop_log_CL.beta_pop.1") =>
            -(named[Symbol("pop_log_CL.beta_pop.1")] - 0.3) / 0.8^2,
        Symbol("pop_log_CL.beta_pop.2") =>
            -(named[Symbol("pop_log_CL.beta_pop.2")] + 0.1) / 0.4^2,
        :log_V_Intercept => -(named[:log_V_Intercept] - 1.5) / 0.6^2)
    @test gθ ≈ [θgradient[name] for name in m.view.coordinates[m.theta]] atol=1e-9
    @test gσ ≈ [1 - exp(only(σ)) / 2] atol=1e-9
    @test gω ≈ [1 - a / 0.7,
        -b / (0.7τ) - 4b / τ^2,
        -c^2 / (0.7τ) + 4b^2 / τ^2] atol=1e-9
    provenance = population_prior_provenance(m)
    @test provenance.source === :BRM_RK_authored_priors
    @test provenance.theta == m.view.coordinates[m.theta]
    @test provenance.omega === :NLME_log_Cholesky
end

@stestset "three-margin covariance prior gradient follows NLME coordinates" begin
    data = (; subject=["a", "b", "c"],
        ya=[0.2, -0.1, 0.5], yb=[-0.3, 0.4, 0.1], yc=[0.1, 0.0, -0.2])
    brmi = @brm data begin
        sigma ~ Exponential(1)
        a ~ 1 + (1 | pk | subject)
        b ~ 1 + (1 | pk | subject)
        c ~ 1 + (1 | pk | subject)
        sd(:, pk) ~ Exponential(0.7)
        cor(:, pk) ~ LKJCholesky(3, 2)
        ya ~ Normal(a, sigma)
        yb ~ Normal(b, sigma)
        yc ~ Normal(c, sigma)
    end
    m = brm_nlme_model(RKBRMI(brmi); ad_backend=NLME_AD)
    θ = fill(0.1, length(m.theta))
    σ = [log(0.8)]
    ω = [log(0.7), 0.12, log(0.8), -0.08, 0.16, log(0.9)]
    value, _, _, gω = population_logprior_and_gradient(m, θ, σ, ω)
    @test value ≈ population_logprior(m, θ, σ, ω) atol=1e-10
    @test length(gω) == 6
    for j in eachindex(ω)
        plus, minus = copy(ω), copy(ω)
        plus[j] += 1e-5
        minus[j] -= 1e-5
        numerical = (population_logprior(m, θ, σ, plus) -
            population_logprior(m, θ, σ, minus)) / 2e-5
        @test gω[j] ≈ numerical atol=1e-6
    end
end

# An SAEM MAP M-step can try a finite ω whose Cholesky row is dominated by an
# off-diagonal entry, so that row's partial correlation rounds to ±1. The prior
# is then -Inf, which the line search rejects, instead of an LKJ refusal that
# aborts the fit (snag saem-map-mstep-l-82de0b0a).
@stestset "a covariance prior at a correlation that rounds to ±1 is -Inf" begin
    data = (; subject=["a", "b", "c"],
        ya=[0.2, -0.1, 0.5], yb=[-0.3, 0.4, 0.1], yc=[0.1, 0.0, -0.2])
    brmi = @brm data begin
        sigma ~ Exponential(1)
        a ~ 1 + (1 | pk | subject)
        b ~ 1 + (1 | pk | subject)
        c ~ 1 + (1 | pk | subject)
        sd(:, pk) ~ Exponential(0.7)
        cor(:, pk) ~ LKJCholesky(3, 2)
        ya ~ Normal(a, sigma)
        yb ~ Normal(b, sigma)
        yc ~ Normal(c, sigma)
    end
    m = brm_nlme_model(RKBRMI(brmi); ad_backend=NLME_AD)
    layout = NLMEEstimation.nlme_layout(m)
    θ = fill(0.1, length(m.theta))
    σ = [log(0.8)]
    # Row 2's partial correlation C[2,1]/‖C[2,1:2]‖, then row 3's
    # C[3,2]/‖C[3,2:3]‖, rounds to exactly 1.
    for (ω, row, col) in (([log(0.7), 0.5, log(1e-10), -0.08, 0.16, log(0.9)], 2, 1),
            ([log(0.7), 0.12, log(0.8), -0.08, 0.5, log(1e-10)], 3, 2))
        C = Matrix(NLMEEstimation.omega_cholesky(layout, ω))
        @test C[row, col] / norm(C[row, col:row]) == 1
        @test population_logprior(m, θ, σ, ω) == -Inf
        value, gθ, gσ, gω = population_logprior_and_gradient(m, θ, σ, ω)
        @test value == -Inf
        @test all(isnan, gθ) && all(isnan, gσ) && all(isnan, gω)
    end
    # Approaching that boundary, the prior stays finite and differentiable.
    ω = [log(0.7), 0.12, log(0.8), -0.08, 0.5, log(1e-4)]
    value, _, _, gω = population_logprior_and_gradient(m, θ, σ, ω)
    @test isfinite(value) && all(isfinite, gω)
    @test value ≈ population_logprior(m, θ, σ, ω) atol=1e-10
    for j in eachindex(ω)
        plus, minus = copy(ω), copy(ω)
        plus[j] += 1e-5
        minus[j] -= 1e-5
        numerical = (population_logprior(m, θ, σ, plus) -
            population_logprior(m, θ, σ, minus)) / 2e-5
        @test gω[j] ≈ numerical rtol=1e-4 atol=1e-4
    end
end

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

@stestset "subject-cell population gradient includes estimated HSGP coordinates" begin
    data = (; subject=["a", "b", "c"], t=[-0.8, 0.1, 0.7],
        y=[0.2, -0.1, 0.5])
    brmi = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + hsgp(t; k=3, domain=(-1.0, 1.0)) + (1 | subject)
        @plate for i in eachindex(mu)
            loc[i] = mu[i]
            y[i] ~ normal(loc[i], sigma)
        end
    end
    m = brm_nlme_model(RKBRMI(brmi); ad_backend=NLME_AD)
    shared = findall(==(:shared_latent), m.view.role)
    @test !isempty(shared) && all(i -> i in m.theta, shared)
    @test parameter_names(m).theta == string.(m.view.coordinates[m.theta])
    θ = fill(0.2, length(m.theta))
    σ = [-0.3]
    η = [0.1]
    value, gθ, gσ = conditional_loglikelihood_and_population_gradient(
        m, 2, θ, σ, η)
    @test value ≈ conditional_loglikelihood(m, 2, θ, σ, η) rtol=1e-12
    h = 1e-5
    for j in eachindex(θ)
        plus = copy(θ); plus[j] += h
        minus = copy(θ); minus[j] -= h
        finite = (conditional_loglikelihood(m, 2, plus, σ, η) -
            conditional_loglikelihood(m, 2, minus, σ, η)) / (2h)
        @test gθ[j] ≈ finite rtol=1e-5 atol=1e-6
    end
    @test length(gσ) == 1 && isfinite(only(gσ))
    ω = [log(0.7)]
    prior, pgθ, _, _ = population_logprior_and_gradient(m, θ, σ, ω)
    @test prior ≈ population_logprior(m, θ, σ, ω) atol=1e-10
    @test population_prior_provenance(m).theta == m.view.coordinates[m.theta]
    for j in eachindex(θ)
        plus = copy(θ); plus[j] += h
        minus = copy(θ); minus[j] -= h
        finite = (population_logprior(m, plus, σ, ω) -
            population_logprior(m, minus, σ, ω)) / (2h)
        @test pgθ[j] ≈ finite rtol=1e-5 atol=1e-6
    end
end

@stestset "batched protocol methods match the analytic oracle" begin
    m = brm_nlme_model(RKBRMI(nlme_plate_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    n = nsubjects(m)
    @test n == 3
    @test nlme_layout(m) == NLMELayout(; ntheta=3, nsigma=1, eta_blocks=[2])
    @test evaluation_style(m) isa BatchedEvaluation
    @test saem_population_gradient_style(m) isa PerSubjectEvaluation
    @test saem_population_gradient_style(m, Val(:online)) isa PerSubjectEvaluation
    @test saem_population_gradient_style(m, Val(:mixture)) isa BatchedEvaluation
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
    for subjects in ([3, 1], [1, 2, 3])
        H = 0.1 .* reshape(collect(1.0:(2length(subjects))), 2, :) .- 0.2
        value, gradient = NLMEEstimation._saem_population_value_gradient(
            BatchedEvaluation(), m, subjects, θ, σ, H)
        cell_value, cell_gradient = NLMEEstimation._saem_population_value_gradient(
            PerSubjectEvaluation(), m, subjects, θ, σ, H)
        @test value ≈ cell_value rtol=1e-11 atol=1e-11
        @test gradient ≈ cell_gradient rtol=1e-9 atol=1e-9
    end
    # The per-subject entry points use one plate cell, not a batch of one.
    η = [0.05, -0.1]
    oracle_H = full_eta(m, [2], reshape(η, 2, 1))
    @test conditional_loglikelihood(m, 2, θ, σ, η) ≈
        plate_pk_oracle(m, θ, σ, oracle_H)[2] rtol=1e-12
    value, gradient = conditional_loglikelihood_and_gradient(m, 2, θ, σ, η)
    @test value ≈ plate_pk_oracle(m, θ, σ, oracle_H)[2] rtol=1e-12
    @test gradient ≈ plate_pk_oracle_gradients(m, θ, σ, oracle_H)[:, 2] rtol=1e-10
    for i in (3, 1, 2)
        ηi = [0.05 * i, -0.1 * i]
        Hi = full_eta(m, [i], reshape(ηi, :, 1))
        oracle = _plate_pk_oracle_terms(m, θ, σ, Hi)[i]
        ℓ, gθ, gσ = conditional_loglikelihood_and_population_gradient(m, i, θ, σ, ηi)
        @test ℓ ≈ oracle.value rtol=1e-12
        @test gθ ≈ oracle.theta_gradient rtol=1e-10
        @test gσ ≈ oracle.sigma_gradient rtol=1e-10
    end
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
    # Conformance also probes the single-subject routes, which a ragged join
    # serves from its per-subject cell.
    @test check_protocol(m, θ, σ, ω; η=H, population_gradient=true)
end

@stestset "a censored ragged join beside an in-cell observation meets the protocol" begin
    d = NLME_CENSORED_JOIN_DATA
    backend = RKBRMI(nlme_censored_join(d))
    m = brm_nlme_model(backend; ad_backend=NLME_AD)
    @test saem_population_gradient_style(m) isa PerSubjectEvaluation
    θ = [0.3, 2.0]
    σ = [-0.4, -1.0, -0.7]
    H = [0.15 -0.2 0.05 0.1; -0.1 0.25 0.3 -0.2]
    ω = omega_parameters(nlme_layout(m), [0.09 0.0; 0.0 0.04])
    @test check_protocol(m, θ, σ, ω; η=H, population_gradient=true)
    oracle = censored_join_oracle(m, backend, θ, σ, H, d)
    @test conditional_loglikelihoods(m, [4, 2], θ, σ, H[:, [4, 2]]) ≈ oracle[[4, 2]] rtol=1e-12
    for i in eachindex(m.view.levels)
        @test conditional_loglikelihood(m, i, θ, σ, H[:, i]) ≈ oracle[i] rtol=1e-12
    end
    fit = saem(m, θ, σ, ω; rng=Random.Xoshiro(2026),
        options=SAEMOptions(iterations=3, burnin=1, mcmc_burnin=0,
            adapt_iterations=0, numeric_maxiter=2, mstep_interval=1,
            numeric_mstep=:online, omega_floor=0.05))
    @test all(isfinite, (fit.theta..., fit.sigma..., fit.omega...))
end

@stestset "flat grouped rows take SAEM's batched population gradient" begin
    # Only a prepared subject cell differentiates one subject's population
    # coordinates; flat grouped rows have none, so SAEM uses the lockstep batch.
    m = brm_nlme_model(RKBRMI(nlme_pk(NLME_DATA)); ad_backend=NLME_AD)
    @test m.cell_gradient === nothing
    for mode in (Val(:online), Val(:mixture))
        @test saem_population_gradient_style(m, mode) isa BatchedEvaluation
    end
    θ = fill(0.1, length(m.theta))
    σ = [log(2.0)]
    fit = saem(m, θ, σ, zeros(3); rng=Random.Xoshiro(1),
        options=SAEMOptions(iterations=3, burnin=1, mcmc_burnin=0,
            adapt_iterations=0, numeric_maxiter=2, mstep_interval=1,
            numeric_mstep=:online, omega_floor=0.05))
    @test all(isfinite, (fit.theta..., fit.sigma..., fit.omega...))
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

# A program whose reverse gradient needs Enzyme runtime activity needs it in
# the η Hessian's reverse-over-forward and the Gaussian pushforward too, so
# both take the runtime activity of the caller's `ad_backend` (BRM e6a72328).
# The model that first raised `EnzymeRuntimeActivityError` there (a derived
# per-row `map` over a closure that no likelihood reads) differentiates under
# static activity from ReactiveKernels fef260229, which drops unread plate
# outputs from the query, so the modes are checked directly.
@stestset "η Hessian blocks take the caller's runtime activity" begin
    ad = AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
    for mode in (Enzyme.Reverse, Enzyme.Forward)
        @test Enzyme.runtime_activity(BRM._brm_enzyme_mode(mode, ad))
        @test !Enzyme.runtime_activity(BRM._brm_enzyme_mode(mode, NLME_AD))
    end
    @test Enzyme.runtime_activity(BRM._brm_nlme_forward_backend(ad).mode)
    @test !Enzyme.runtime_activity(BRM._brm_nlme_forward_backend(NLME_AD).mode)
    m = brm_nlme_model(RKBRMI(nlme_plate_pk(NLME_PLATE_DATA)); ad_backend=ad)
    θ, σ = plate_pk_point(m)
    H = 0.1 .* reshape(collect(1.0:6.0), 2, :) .- 0.2
    blocks = conditional_loglikelihood_hessian_blocks(m, [1, 2, 3], θ, σ, H)
    @test all(isapprox.(blocks, plate_pk_oracle_hessians(m, θ, σ, H); rtol=1e-9, atol=1e-9))
    ω = omega_parameters(nlme_layout(m), [0.09 0.02; 0.02 0.04])
    result = empirical_bayes(m, θ, σ, ω)
    @test result.method isa NewtonEBE
    @test all(result.converged)
    plain = brm_nlme_model(RKBRMI(nlme_plate_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    @test result.modes ≈ empirical_bayes(plain, θ, σ, ω).modes rtol=1e-10 atol=1e-12
end
