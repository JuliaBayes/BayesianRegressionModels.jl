# A complete point of the same model -> checked NLMEEstimation starting values.
# RUN: julia --project=<env with NLMEEstimation> test/nlme_start.jl
using BridgeStan, NLMEEstimation, Random, LinearAlgebra
include(joinpath(@__DIR__, "nlme_fixtures.jl"))

# A deterministic point with non-unit scales, a non-identity correlation factor
# and non-zero standardized draws.
spread_point(n; scale=0.35) = [scale * sin(1.7k + 0.3) for k in 1:n]

# Three subjects with one, two and three nested units, interleaved
# observations, a correlated subject block and a shared HSGP over a
# subject-level covariate.
const START_DATA = (;
    subject=["s1", "s2", "s3"],
    x=[-0.6, 0.1, 0.7],
    obs_subject=["s2", "s1", "s3", "s2", "s3", "s3", "s1", "s2", "s3", "s1"],
    unit=["u2", "u1", "u4", "u3", "u5", "u6", "u1", "u2", "u4", "u1"],
    t=[0.5, 0.2, 0.9, 1.4, 0.3, 1.1, 1.6, 2.0, 0.4, 0.8],
    y=[0.4, 0.2, -0.1, 0.9, 0.2, -0.3, 0.6, 1.1, 0.0, 0.3],
)

function start_model(data)
    @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + hsgp(x; k=3, domain=(-1.0, 1.0)) + (1 | p | subject)
        rate ~ 1 + (1 | p | subject)
        unit_shift ~ 0 + (1 | u | unit)
        @plate for i in eachindex(mu)
            pred[i] = mu[i] .+ rate[i] .* ragged(t, obs_subject)[i] .+
                ragged(unit_shift, obs_subject)[i]
        end
        ragged(y, obs_subject) ~ Normal(pred, sigma)
    end
end

const START_GROUPS = (; subject_group=:subject, nested_groups=(unit=:obs_subject,))

@stestset "an RK point of a correlated PK model becomes NLME starting values" begin
    brmi = nlme_plate_pk(NLME_PLATE_DATA)
    backend = RKBRMI(brmi)
    m = brm_nlme_model(backend; ad_backend=NLME_AD)
    view = m.view
    u = spread_point(length(view.coordinates))
    saved = copy(u)
    start = brm_nlme_start(m, u)
    @test u == saved
    @test start.theta == u[m.theta]
    @test start.sigma == u[m.sigma]
    @test start.subjects == view.levels
    @test start.names == parameter_names(m)
    @test all(==(:population), start.theta_roles)
    @test isempty(start.shared.names) && start.shared.estimated
    layout = nlme_layout(m)
    @test size(start.eta) == (neta(layout), nsubjects(m))
    @test length(start.omega) == nomega(layout)
    @test omega_matrix(layout, start.omega) ≈ start.omega_matrix rtol=1e-12
    @test isposdef(Symmetric(start.omega_matrix))
    @test !(start.omega_matrix ≈ I) # the point's covariance, not the unit point's
    @test start.max_error <= 1e-10
    # The natural effects reproduce the physical observation likelihood at u.
    @test sum(start.loglikelihoods) ≈
        Base.invokelatest(likelihood_query(backend), u) rtol=1e-10
    @test start.loglikelihoods ==
        brm_nlme_loglikelihoods(m, start.theta, start.sigma, start.eta)
    @test start.population_logprior ==
        population_logprior(m, start.theta, start.sigma, start.omega)
    @test check_protocol(m, start.theta, start.sigma, start.omega; η=start.eta)

    # Independent inverse: Stan's own transforms rebuild the same point from
    # the natural covariance and effects.
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    map = brm_nlme_joint_map(backend, sb;
        path=joinpath(mktempdir(), "nlme_start_pk.stan"))
    stan_u = brm_rk_point_to_stan(map.transport, u)
    forward = brm_nlme_joint_start(map; theta=start.theta, sigma=start.sigma,
        omega=start.omega_matrix, eta=start.eta, subjects=start.subjects)
    @test forward.stan ≈ stan_u rtol=1e-9 atol=1e-10
    from_stan = brm_nlme_start(m, map, stan_u)
    @test from_stan.theta ≈ start.theta rtol=1e-12
    @test from_stan.sigma ≈ start.sigma rtol=1e-12
    @test from_stan.omega ≈ start.omega rtol=1e-10
    @test from_stan.eta ≈ start.eta rtol=1e-10

    # refused: the start must be read from this model's own packed layout.
    @test_throws DimensionMismatch brm_nlme_start(m, u[1:end-1])
    # refused: NLMEEstimation requires finite starting coordinates.
    bad = copy(u); bad[first(m.theta)] = NaN
    @test_throws BRMNLMEViewError brm_nlme_start(m, bad)
    # refused: a zero natural scale has no log-Cholesky coordinate.
    collapsed = copy(u)
    collapsed[findfirst(==(:subject_scale), view.role)] = -800.0
    @test_throws BRMNLMEViewError brm_nlme_start(m, collapsed)
    # refused: a Stan point is only meaningful for a map of the same BRMI
    # instance; an equal-looking rebuild could differ in data or source.
    other = BRMNLMEJointMap(RKBRMI(nlme_plate_pk(NLME_PLATE_DATA)), map.sb,
        map.stan_problem, map.stan_model, map.view, map.transport, map.blocks)
    @test_throws BRMNLMEViewError brm_nlme_start(m, other, stan_u)
end

@stestset "a nested model with a shared HSGP starts in both shared modes" begin
    brmi = start_model(START_DATA)
    backend = RKBRMI(brmi)
    m = brm_nlme_model(backend; ad_backend=NLME_AD, START_GROUPS...)
    view = m.view
    layout = nlme_layout(m)
    @test layout isa RaggedNLMELayout
    @test view.family_sizes == [2, 1]
    shared = findall(==(:shared_latent), view.role)
    @test !isempty(shared)
    u = spread_point(length(view.coordinates))
    start = brm_nlme_start(m, u)
    @test start.theta == u[m.theta]
    @test start.shared.estimated
    @test start.shared.names == view.coordinates[shared]
    @test start.shared.values == u[shared]
    @test count(==(:shared_latent), start.theta_roles) == length(shared)
    @test [start.theta[findfirst(==(i), m.theta)] for i in shared] == u[shared]
    @test length.(start.eta) == [neta(layout, i) for i in 1:nsubjects(m)]
    @test length.(start.names.eta) == length.(start.eta)
    @test length(start.omega_matrix) == 2
    @test size.(start.omega_matrix) == [(2, 2), (1, 1)]
    @test all(isposdef(Symmetric(Ω)) for Ω in start.omega_matrix)
    @test !isdiag(start.omega_matrix[1])
    @test all(omega_matrix(layout, start.omega) .≈ start.omega_matrix)
    @test start.max_error <= 1e-10
    @test sum(start.loglikelihoods) ≈
        Base.invokelatest(likelihood_query(backend), u) rtol=1e-10
    @test isfinite(start.population_logprior)
    @test check_protocol(m, start.theta, start.sigma, start.omega; η=start.eta)
    fit = saem(m, start.theta, start.sigma, start.omega; η0=start.eta,
        rng=Random.Xoshiro(2026),
        options=SAEMOptions(iterations=2, burnin=1, mcmc_burnin=0,
            adapt_iterations=0, numeric_maxiter=2, mstep_interval=2,
            omega_floor=0.05))
    @test fit isa RaggedSAEMResult
    @test length.(fit.eta) == length.(start.eta)

    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    map = brm_nlme_joint_map(backend, sb;
        path=joinpath(mktempdir(), "nlme_start_nested.stan"), START_GROUPS...)
    stan_u = brm_rk_point_to_stan(map.transport, u)
    order = [2, 3, 1]
    forward = brm_nlme_joint_start(map; theta=start.theta, sigma=start.sigma,
        omega=start.omega_matrix, eta=start.eta[order],
        subjects=start.subjects[order])
    @test forward.stan ≈ stan_u rtol=1e-9 atol=1e-10
    from_stan = brm_nlme_start(m, map, stan_u)
    @test from_stan.theta ≈ start.theta rtol=1e-12
    @test from_stan.omega ≈ start.omega rtol=1e-10
    @test all(from_stan.eta .≈ start.eta)

    # Fixed shared coordinates: the same point conditions the model.
    fixed = brm_nlme_model(backend; ad_backend=NLME_AD, shared_point=u,
        START_GROUPS...)
    held = brm_nlme_start(fixed, u)
    @test !held.shared.estimated
    @test held.shared.values == u[shared]
    @test held.theta == u[fixed.theta]
    @test isempty(intersect(shared, fixed.theta))
    @test all(==(:population), held.theta_roles)
    @test held.omega ≈ start.omega rtol=1e-12
    @test all(held.eta .≈ start.eta)
    @test held.loglikelihoods ≈ start.loglikelihoods rtol=1e-12
    fixed_forward = brm_nlme_joint_start(map; theta=held.theta,
        sigma=held.sigma, omega=held.omega_matrix, eta=held.eta,
        subjects=held.subjects, shared_point=u)
    @test fixed_forward.stan ≈ stan_u rtol=1e-9 atol=1e-10
    # refused: a fixed-shared model's start must sit at its fixed shared
    # values; another point would silently start a different conditional target.
    moved = copy(u); moved[first(shared)] += 0.25
    @test_throws BRMNLMEViewError brm_nlme_start(fixed, moved)
end

@stestset "address-keyed initial estimates become NLME and full Stan starts" begin
    brmi = nlme_plate_pk(NLME_PLATE_DATA)
    backend = RKBRMI(brmi)
    m = brm_nlme_model(backend; ad_backend=NLME_AD)
    view = m.view
    Ω = [0.09 0.018; 0.018 0.16]           # ranefcoefnames(brmi, :pk) order
    inits = (; log_CL=(; Intercept=log(2.7), weight=0.75),
        log_V=(; Intercept=log(31.5)), pk=Ω, sigma=0.1)
    start = brm_nlme_start(m, inits)
    @test start == brm_nlme_start(m, brm_start(backend, inits; base=m.unit))
    theta = Dict(zip(view.coordinates[m.theta], start.theta))
    @test theta[Symbol("pop_log_CL.beta_pop.1")] ≈ log(2.7) rtol=1e-12
    @test theta[Symbol("pop_log_CL.beta_pop.2")] ≈ 0.75 rtol=1e-12
    @test theta[:log_V_Intercept] ≈ log(31.5) rtol=1e-12
    @test start.sigma ≈ [log(0.1)] rtol=1e-12
    order = [findfirst(b -> b.predictor === lp, view.block_margins)
        for lp in (:log_CL, :log_V)]
    @test start.omega_matrix[order, order] ≈ Ω rtol=1e-10
    @test omega_matrix(nlme_layout(m), start.omega) ≈ start.omega_matrix rtol=1e-12
    @test all(iszero, start.eta)
    @test check_protocol(m, start.theta, start.sigma, start.omega; η=start.eta)

    # The full Stan target starts from the same addresses at the same point.
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    map = brm_nlme_joint_map(backend, sb;
        path=joinpath(mktempdir(), "nlme_start_inits.stan"))
    joint = brm_nlme_joint_start(map; theta=start.theta, sigma=start.sigma,
        omega=start.omega_matrix, eta=start.eta, subjects=start.subjects)
    @test brm_start(map.sb, map.stan_model, inits) ≈ joint.stan rtol=1e-9 atol=1e-10

    # refused: the base is a full packed point of this model's own layout.
    @test_throws DimensionMismatch brm_nlme_start(m, inits; base=zeros(2))
end
