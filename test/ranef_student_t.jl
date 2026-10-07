# test/ranef_student_t.jl — Student-t random-effect blocks, `ranef(:, ID) ~ TDist(nu)`.
#
# Run: julia --project=test test/ranef_student_t.jl
# Set BRM_RANEF_STUDENT_T_RUNTIME=0 to skip the BridgeStan density probes.
#
# The block law is the Gaussian scale mixture brms emits for
# `gr(..., dist = "student")`: one weight `w[j] ~ InverseGamma(nu/2, nu/2)` per
# group level scales that level's Gaussian deviation vector by `sqrt(w[j])`,
# so the deviations are multivariate t with the Gaussian block's scale matrix
# (user decision `BayesianRegressionModels:studentt-res/decisions/0d59ktf`).
using Test
using Random
using Statistics
using LinearAlgebra
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Turing: Turing   # loads the direct Turing backend extension
using Distributions: Exponential, Gamma, InverseGamma, LKJCholesky, Normal, TDist,
    Cauchy, logpdf, quantile, cdf

const BRM = BayesianRegressionModels
const STUDENT_T_CACHE = joinpath(tempdir(), "brm-ranef-student-t")
const STUDENT_T_RUNTIME = get(ENV, "BRM_RANEF_STUDENT_T_RUNTIME", "1") != "0"

df = (;
    x = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.2, 0.3, 0.8, -0.4],
    subject = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    y = [-2.4, -2.2, -2.0, -1.8, -1.7, -1.5, -2.6, -1.1, -0.9, -2.1],
)

fixed_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    ranef(:, p) ~ TDist(4)
    y ~ Normal(mu, sigma)
end

estimated_builder = @brm begin
    sigma ~ Exponential(1)
    nu ~ Gamma(2, 10; lower=1.0)
    mu ~ 1 + x + (1 + x | p | subject)
    sd(:, p) ~ Exponential(1)
    cor(:, p) ~ LKJCholesky(2, 2)
    ranef(:, p) ~ TDist(nu)
    y ~ Normal(mu, sigma)
end

gaussian_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    y ~ Normal(mu, sigma)
end

stan_code(sb) = BRM.stan_code(sb)
build(builder, data=df; kwargs...) =
    SBBRMI(builder(data); mod=@__MODULE__, total_groups=(), kwargs...)

function instantiate(sb)
    isdir(STUDENT_T_CACHE) || mkpath(STUDENT_T_CACHE)
    code = stan_code(sb)
    StanBlocks.stan_instantiate(sb.model;
        path=joinpath(STUDENT_T_CACHE, string(hash(code)) * ".stan"))
end

# Constrained values of one named parameter, in Stan's (column-major) order.
function constrained(bs, theta, prefix)
    names = StanBlocks.BridgeStan.param_names(bs)
    values = StanBlocks.BridgeStan.param_constrain(bs, theta)
    hits = [i for (i, n) in enumerate(names) if n == prefix || startswith(n, prefix * ".")]
    isempty(hits) && error("no constrained parameter `$prefix`")
    values[hits]
end

@testset "address capture and the population-effect boundary" begin
    brmi = estimated_builder(df)
    specs = ranef_effect_priors(brmi)
    @test [s.class for s in specs] == [:sd, :cor, :ranef]
    law = only(s for s in specs if s.class === :ranef)
    @test law.id === :p
    @test isnothing(law.predictor) && isnothing(law.coefficient)
    @test law.family === TDist
    # A block law is not a population-coefficient prior.
    @test isempty(effect_priors(brmi))
    # It is a prior statement, not model structure.
    @test length(ranef_effect_priors(priors_of(brmi))) == 3
    @test isempty(ranef_effect_priors(structure_of(brmi)))
end

@testset "fixed df: emission, family and stanc" begin
    sb = build(fixed_builder)
    code = stan_code(sb)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("vector<lower=0.0>[n_subject] b_p_subject_w;", code)
    @test occursin("b_p_subject_w ~ inv_gamma((0.5 * 4.0), (0.5 * 4.0));", code)
    @test occursin("sqrt(b_p_subject_w)", code)
    # The Gaussian part is the default draws submodel, unchanged.
    @test occursin("b_p_subject_L ~ lkj_corr_cholesky(1.0);", code)
    @test occursin("b_p_subject_tau ~ std_normal();", code)
    block = only(ranef_blocks(sb))
    @test block.family === :ranef_correlated_draws_student_t
    @test block.z === :b_p_subject_z_flat
    @test block.noncentered
end

@testset "estimated df with configured sd/cor" begin
    sb = build(estimated_builder)
    code = stan_code(sb)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("b_p_subject_w ~ inv_gamma((0.5 * nu), (0.5 * nu));", code)
    @test occursin("b_p_subject_L ~ lkj_corr_cholesky(2.0);", code)
    @test occursin("b_p_subject_tau ~ exponential(", code)
    # The sampled df is declared before the block that reads it.
    @test findfirst("nu ~ gamma(", code).start < findfirst("b_p_subject_w ~", code).start
    @test only(ranef_blocks(sb)).family === :ranef_correlated_draws_generic_student_t
end

@testset "explicit Normal() is the default law, byte for byte" begin
    explicit = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | subject)
        ranef(:, p) ~ Normal()
        y ~ Normal(mu, sigma)
    end
    @test stan_code(build(explicit)) == stan_code(build(gaussian_builder))
end

@testset "scale mixture is the Student-t law" begin
    # Integrating w ~ InverseGamma(nu/2, nu/2) out of sqrt(w) * N(0, 1) gives
    # TDist(nu); a wrong parameterization (for example rate instead of scale)
    # moves these quantiles visibly.
    rng = Xoshiro(20261007)
    nu = 3.5
    n = 400_000
    draws = sqrt.(rand(rng, InverseGamma(nu / 2, nu / 2), n)) .* randn(rng, n)
    for p in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        @test isapprox(quantile(draws, p), quantile(TDist(nu), p); atol=0.03)
    end
end

@testset "BridgeStan density equals the intended joint" begin
    if STUDENT_T_RUNTIME
        for (label, builder, nu_of) in (
                ("fixed", fixed_builder, (_, _) -> 4.0),
                ("estimated", estimated_builder, (bs, th) -> only(constrained(bs, th, "nu"))))
            sb = build(builder)
            problem = instantiate(sb)
            bs = problem.model
            dim = LogDensityProblems.dimension(problem)
            K, G = 2, 5
            rng = Xoshiro(hash(label))
            for _ in 1:3
                theta = 0.4 .* randn(rng, dim)
                L = reshape(constrained(bs, theta, "b_p_subject_L"), K, K)
                tau = constrained(bs, theta, "b_p_subject_tau")
                z = constrained(bs, theta, "b_p_subject_z_flat")
                w = constrained(bs, theta, "b_p_subject_w")
                sigma = only(constrained(bs, theta, "sigma"))
                beta = constrained(bs, theta, "pop_mu_beta_pop")
                nu = nu_of(bs, theta)
                b = Diagonal(sqrt.(w)) * (Diagonal(tau) * L * reshape(z, K, G))'
                mu = beta[1] .+ beta[2] .* df.x .+
                     [b[df.subject[i], 1] + b[df.subject[i], 2] * df.x[i] for i in eachindex(df.x)]
                lkj, tau_prior = label == "fixed" ?
                    (LKJCholesky(K, 1.0), Normal()) : (LKJCholesky(K, 2.0), Exponential(1.0))
                reference = logpdf(lkj, Cholesky(LowerTriangular(L))) +
                    sum(logpdf.(tau_prior, tau)) +
                    sum(logpdf.(Normal(), z)) +
                    sum(logpdf.(InverseGamma(nu / 2, nu / 2), w)) +
                    logpdf(Exponential(1.0), sigma) +
                    sum(logpdf.(Normal(), beta)) +
                    sum(logpdf.(Normal.(mu, sigma), df.y)) +
                    (label == "fixed" ? 0.0 : logpdf(Gamma(2, 10), nu))
                emitted = StanBlocks.BridgeStan.log_density(bs, theta;
                    propto=false, jacobian=false)
                @test emitted ≈ reference rtol=1e-10
            end
            lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, zeros(dim))
            @test isfinite(lp) && all(isfinite, gradient)
        end
    else
        @info "Skipping Student-t BridgeStan density gate (BRM_RANEF_STUDENT_T_RUNTIME=0)"
    end
end

@testset "cv sizing and resampling re-draw the mixing weights" begin
    # cv sizing gives the weights the standardized draws' size expression, so
    # they move to generated quantities together when the group index is marked.
    cv = build(estimated_builder; cv_groups=[:subject])
    cv_code = stan_code(cv)
    @test StanBlocks.stanc_check(cv_code; warn_pedantic=false).ok
    @test occursin("vector<lower=0.0>[b_p_subject_n_g] b_p_subject_w;", cv_code)
    @test only(d for d in generative_plan(cv).declarations
               if d.target === :b_p_subject).keywords.n_groups === :b_p_subject_n_g
    resampled = reprocess(build(fixed_builder), df; resample_groups=[:subject])
    re_code = stan_code(resampled)
    @test StanBlocks.stanc_check(re_code; warn_pedantic=false).ok
    @test occursin("inv_gamma_rng", re_code)
    @test only(ranef_blocks(resampled)).generated
end

@testset "exact totals keep a Student-t block conventional" begin
    intercept_t = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + (1 | p | subject)
        ranef(:, p) ~ TDist(5)
        y ~ Normal(mu, sigma)
    end
    intercept_gaussian = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + (1 | p | subject)
        y ~ Normal(mu, sigma)
    end
    # Positive control: the Gaussian twin is selected automatically.
    @test !isempty(total_effect_blocks(SBBRMI(intercept_gaussian(df); mod=@__MODULE__)))
    auto = SBBRMI(intercept_t(df); mod=@__MODULE__)
    @test isempty(total_effect_blocks(auto))
    @test only(ranef_blocks(auto)).family === :ranef_correlated_draws_student_t
    @test_throws "Gaussian grouping structure" SBBRMI(intercept_t(df);
        mod=@__MODULE__, total_groups=[:subject])
end

@testset "prediction carries mixing weights" begin
    sb = build(fixed_builder)
    block = only(ranef_blocks(sb))
    names(G) = vcat(["b_p_subject_L.1"], ["b_p_subject_tau.$k" for k in 1:2],
        ["b_p_subject_z_flat.$i" for i in 1:(2G)], ["b_p_subject_w.$g" for g in 1:G],
        ["sigma"], ["pop_mu_beta_pop.$k" for k in 1:2])
    unc = names(5)
    if STUDENT_T_RUNTIME
        @test StanBlocks.BridgeStan.param_unc_names(instantiate(sb).model) == unc
    end
    @test BRM._ranef_mixing_coordinates(block, unc) ==
          [findfirst(==("b_p_subject_w.$g"), unc) for g in 1:5]
    draws = reshape(collect(1.0:(3 * length(unc))), 3, length(unc))
    population = population_draws(sb, draws, unc; groups=:subject)
    z = vec(ranef_coordinates(block, unc))
    @test all(iszero, population[:, z])
    @test population[:, setdiff(eachindex(unc), z)] == draws[:, setdiff(eachindex(unc), z)]

    # Retained levels carry their own weight by label, not by position.
    subset = (; x=df.x[5:10], subject=df.subject[5:10], y=df.y[5:10])   # levels 3, 4, 5
    target = build(fixed_builder, subset)
    target_unc = names(3)
    moved = transport_draws(sb, target, draws, unc, target_unc)
    for (g, level) in enumerate((3, 4, 5))
        @test moved[:, findfirst(==("b_p_subject_w.$g"), target_unc)] ==
              draws[:, findfirst(==("b_p_subject_w.$level"), unc)]
    end
    # Fresh levels re-draw Stan-side, never as Julia-side Normal draws.
    @test_throws "resample_groups" transport_draws(sb, target, draws, unc, target_unc;
        resample=:subject)
end

@testset "scientific description names the Student-t law" begin
    description = brm_description(brm_descriptor(build(estimated_builder); name=:student_t))
    @test description.complete
    @test any(contains("multivariate Student-t"), description.prose)
    @test any(contains("t_{\\mathrm{nu},2}"), description.equations)
    @test any(contains("\\operatorname{InvGamma}"), description.equations)
    weights = only(p for p in description.priors
                   if p.id == (:random_effect, :p, :subject, :mixing_weights))
    @test weights.distribution.callable === InverseGamma
end

@testset "invalid block laws are refused" begin
    # refused: a block law is standardized; its scale is `sd(...)` (decision 0d59ktf).
    scaled = @brm begin
        mu ~ 1 + (1 | p | subject)
        ranef(:, p) ~ Normal(0, 2)
        y ~ Normal(mu, 1)
    end
    @test_throws "without arguments" build(scaled)
    # refused: the supported block laws are Normal() and TDist(nu) (decision 0d59ktf).
    cauchy = @brm begin
        mu ~ 1 + (1 | p | subject)
        ranef(:, p) ~ Cauchy(0, 1)
        y ~ Normal(mu, 1)
    end
    @test_throws "support the standardized laws" build(cauchy)
    # refused: degrees of freedom must be strictly positive.
    zero_df = @brm begin
        mu ~ 1 + (1 | p | subject)
        ranef(:, p) ~ TDist(0)
        y ~ Normal(mu, 1)
    end
    @test_throws "strictly positive" build(zero_df)
    # refused: one law per block, caught when the macro expands.
    @test_throws "duplicate `ranef(:, p)`" macroexpand(@__MODULE__, :(@brm begin
        mu ~ 1 + (1 | p | subject)
        ranef(:, p) ~ TDist(3)
        ranef(:, p) ~ TDist(5)
        y ~ Normal(mu, 1)
    end))
    unknown_id = @brm begin
        mu ~ 1 + (1 | p | subject)
        ranef(:, q) ~ TDist(3)
        y ~ Normal(mu, 1)
    end
    @test_throws "matches no shared" build(unknown_id)
end

@testset "capability gaps stay explicit" begin
    # Not built yet (tracked under todo 2026-10-07T10-23-31-772-1pdw1mw); each
    # must refuse rather than silently fit a Gaussian block.
    per_predictor = @brm begin
        mu ~ 1 + (1 | p | subject)
        ranef(mu, p) ~ TDist(3)
        y ~ Normal(mu, 1)
    end
    @test_throws "block-wide in this version" build(per_predictor)
    @test_throws "centered Student-t emission is not implemented" build(
        fixed_builder; centered_groups=[:subject])
    @test_throws "does not implement Student-t" TuringBRMI(fixed_builder(df))
    stratified = @brm begin
        mu ~ 1 + (1 | p | gr(subject, by=arm))
        ranef(:, p) ~ TDist(3)
        y ~ Normal(mu, 1)
    end
    @test_throws "stratified" build(stratified, merge(df, (; arm=repeat([1, 2], 5))))
end
