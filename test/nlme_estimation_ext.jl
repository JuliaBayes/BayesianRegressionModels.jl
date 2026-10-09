# The NLMEEstimation.jl protocol for RK-lowered BRM models
# (ext/BayesianRegressionModelsNLMEEstimationExt.jl).
#
# NLMEEstimation.jl is not published yet, so it is not part of test/Project.toml.
# Run this file from an environment that adds it to the test environment, e.g.
# a scratch copy of test/Project.toml + test/Manifest.toml with
# `Pkg.develop(path=<NLMEEstimation checkout>)`:
#
#   julia --project=<scratch env> test/nlme_estimation_ext.jl
#
# It refuses to run without NLMEEstimation rather than skipping silently.
Base.find_package("NLMEEstimation") === nothing && error(
    "test/nlme_estimation_ext.jl needs NLMEEstimation.jl in the active environment; " *
    "see the header of this file")

using NLMEEstimation
include(joinpath(@__DIR__, "nlme_fixtures.jl"))

const NLMEEXT = Base.get_extension(BRM, :BayesianRegressionModelsNLMEEstimationExt)

@stestset "the extension loads with NLMEEstimation" begin
    @test NLMEEXT !== nothing
end

@stestset "protocol methods agree with the lockstep evaluation and the oracle" begin
    m = brm_nlme_model(RKBRMI(nlme_plate_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    n = length(m.view.levels)
    @test nsubjects(m) == n
    layout = nlme_layout(m)
    @test layout == NLMELayout(; ntheta=3, nsigma=1, eta_blocks=[2])
    names = parameter_names(m)
    @test length(names.theta) == 3 && length(names.sigma) == 1 && length(names.eta) == 2
    θ = [0.4, -0.7, 0.2][sortperm(m.view.coordinates[m.theta])]
    σ = [log(0.8)]
    for i in 1:n
        η = [0.1i, -0.05i]
        H = zeros(2, n); H[:, i] .= η
        ℓ = conditional_loglikelihood(m, i, θ, σ, η)
        @test ℓ ≈ plate_pk_oracle(m, θ, σ, H)[i] rtol=1e-12
        ℓg, g = conditional_loglikelihood_and_gradient(m, i, θ, σ, η)
        @test ℓg == ℓ
        step = 1e-6
        reference = map(1:2) do k
            plus, minus = copy(H), copy(H)
            plus[k, i] += step; minus[k, i] -= step
            (plate_pk_oracle(m, θ, σ, plus)[i] - plate_pk_oracle(m, θ, σ, minus)[i]) / 2step
        end
        @test g ≈ reference rtol=1e-6 atol=1e-8
    end
    @test_throws ArgumentError conditional_loglikelihood(m, n + 1, θ, σ, zeros(2))
end

@stestset "declared mu-referencing holds on the density" begin
    m = brm_nlme_model(RKBRMI(nlme_plate_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    mr = mu_referencing(m)
    @test mr isa MuReferencing
    @test length(mr.theta_indices) == 2
    @test all(names -> occursin("Intercept", names) || occursin("beta_pop.1", names),
        parameter_names(m).theta[mr.theta_indices])
    θ = [0.4, -0.7, 0.2][sortperm(m.view.coordinates[m.theta])]
    σ = [log(0.8)]
    δ = [0.3, -0.2]
    θs = copy(θ); θs[mr.theta_indices] .+= δ
    for i in 1:nsubjects(m)
        η = [0.1, 0.2]
        ηs = η - mr.design[i] * δ
        @test conditional_loglikelihood(m, i, θs, σ, ηs) ≈
            conditional_loglikelihood(m, i, θ, σ, η) rtol=1e-12
    end
end

# Capability gap, not a refusal: the package's conformance check and its
# Newton EBE need an η Hessian, and RK exposes first-order AD only
# (ReactiveKernels snag rkppl-second-ord-ba21b602; gradient-only EBE proposed
# on NLMEEstimation todo 2026-10-09T11-54-51-737-1rrm3pc).
@stestset "Hessian-based conformance and posthoc await second-order AD" begin
    m = brm_nlme_model(RKBRMI(nlme_plate_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    θ = [0.4, -0.7, 0.2][sortperm(m.view.coordinates[m.theta])]
    σ = [log(0.8)]
    ω = omega_parameters(nlme_layout(m), [0.09 0.0; 0.0 0.04])
    @test_broken check_protocol(m, θ, σ, ω)
    @test_broken empirical_bayes(m, θ, σ, ω) isa EBEResult
end
