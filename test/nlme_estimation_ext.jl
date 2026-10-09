# The NLMEEstimation.jl protocol for RK-lowered BRM models
# (ext/BayesianRegressionModelsNLMEEstimationExt.jl): batched evaluation and the
# gradient-only posthoc estimator, checked against the analytic oracle of
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
    @test !provides_eta_hessian(m)
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
    end
    # The per-subject entry points derive from a batch of one.
    η = [0.05, -0.1]
    @test conditional_loglikelihood(m, 2, θ, σ, η) ≈
        plate_pk_oracle(m, θ, σ, full_eta(m, [2], reshape(η, 2, 1)))[2] rtol=1e-12
    @test_throws ArgumentError conditional_loglikelihoods(m, [1, 1], θ, σ, zeros(2, 2))
    @test_throws ArgumentError conditional_loglikelihoods(m, [n + 1], θ, σ, zeros(2, 1))
    @test_throws DimensionMismatch conditional_loglikelihoods(m, [1, 2], θ, σ, zeros(2, 1))
end

@stestset "declared mu-referencing holds on the density" begin
    m = brm_nlme_model(RKBRMI(nlme_plate_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    mr = mu_referencing(m)
    @test mr isa MuReferencing
    @test sort(parameter_names(m).theta[mr.theta_indices]) ==
        ["log_V_Intercept", "pop_log_CL.beta_pop.1"]
    θ, σ = plate_pk_point(m)
    δ = [0.3, -0.2]
    θs = copy(θ); θs[mr.theta_indices] .+= δ
    subjects = collect(1:nsubjects(m))
    H = repeat([0.1, 0.2], 1, length(subjects))
    Hs = H .- reduce(hcat, [mr.design[i] * δ for i in subjects])
    @test conditional_loglikelihoods(m, subjects, θs, σ, Hs) ≈
        conditional_loglikelihoods(m, subjects, θ, σ, H) rtol=1e-12
end

@stestset "check_protocol accepts the model" begin
    m = brm_nlme_model(RKBRMI(nlme_plate_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    θ, σ = plate_pk_point(m)
    ω = omega_parameters(nlme_layout(m), [0.09 0.0; 0.0 0.04])
    @test check_protocol(m, θ, σ, ω; η=[0.1 -0.05 0.2; -0.1 0.15 0.05])
end

@stestset "posthoc finds the oracle's complete-data modes" begin
    m = brm_nlme_model(RKBRMI(nlme_plate_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    θ, σ = plate_pk_point(m)
    ω = omega_parameters(nlme_layout(m), [0.09 0.02; 0.02 0.04])
    prior = re_prior(m, ω)
    result = empirical_bayes(m, θ, σ, ω)
    @test result.method isa QuasiNewtonEBE
    @test all(result.converged)
    @test result.hessians === nothing
    H = result.modes
    @test result.conditional_loglikelihood ≈ plate_pk_oracle(m, θ, σ, H) rtol=1e-12
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
