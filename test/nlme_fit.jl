# Prior-median starting values, NLME estimates in the model's own names and
# natural units, and the one-call fit through NLMEEstimation's estimators.
# RUN: julia --project=<env with NLMEEstimation> test/nlme_fit.jl
using NLMEEstimation, Random, LinearAlgebra
using Treebars: with_progress, with_prepared_phases, render_text, request_interrupt!,
    ProgressInterrupted
include(joinpath(@__DIR__, "nlme_fixtures.jl"))

# The plate population-PK fixture with an authored prior on every estimated
# population, scalar and scale coordinate.
function authored_pk(data)
    @brm data begin
        sigma ~ Exponential(2)
        log_CL ~ 1 + weight + (1 | pk | subject)
        log_V ~ 1 + (1 | pk | subject)
        effect(log_CL, Intercept) ~ Normal(0.3, 0.8)
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
end

# Independent random intercepts without a block ID, a categorical covariate, a
# flat coefficient, and a scale whose prior reads another sampled scalar.
const UNNAMED_DATA = (; NLME_PLATE_DATA..., sex=["f", "m", "f"])
function unnamed_pk(data)
    @brm data begin
        s0 ~ Exponential(1)
        sigma ~ Exponential(s0)
        log_CL ~ 1 + weight + sex + (1 | subject)
        log_V ~ 1 + (1 | subject)
        effect(log_CL, weight) ~ Flat()
        @plate for i in eachindex(log_CL)
            CL = exp(log_CL[i])
            Vc = exp(log_V[i])
            pred[i] = dose[i] / Vc .* exp.(-(CL / Vc) .* t[i])
            dv[i] ~ normal(pred[i], sigma)
        end
    end
end

named_theta(m, θ) = Dict(zip(m.view.coordinates[m.theta], θ))
scale_positions(m) = findall(==(:subject_scale), m.view.role)

@stestset "the default start is the declared priors' medians" begin
    m = brm_nlme_model(RKBRMI(authored_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    start = brm_nlme_start(m)
    θ = named_theta(m, start.theta)
    @test θ[Symbol("pop_log_CL.beta_pop.1")] == 0.3
    @test θ[Symbol("pop_log_CL.beta_pop.2")] == 0.0 # the default Normal(0, 1)
    @test θ[:log_V_Intercept] == 1.5
    @test exp(only(start.sigma)) ≈ 2log(2) rtol=1e-12 # median of Exponential(2)
    τ = 0.7log(2)                                      # median of Exponential(0.7)
    @test start.omega_matrix ≈ [τ^2 0; 0 τ^2] rtol=1e-12
    @test all(iszero, start.eta)
    @test isempty(start.start_source.zero)
    @test Set(start.start_source.prior_median) ==
        Set(m.view.coordinates[vcat(m.theta, m.sigma, scale_positions(m))])
    @test all(isfinite, start.loglikelihoods) && isfinite(start.population_logprior)
    # The emitted defaults: Normal(0, 1) coefficients, Exponential(1) for
    # sigma and half-normal scales of a shared block.
    m = brm_nlme_model(RKBRMI(nlme_plate_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    start = brm_nlme_start(m)
    @test all(iszero, start.theta)
    @test exp(only(start.sigma)) ≈ log(2) rtol=1e-12
    @test start.omega_matrix ≈ quantile(Normal(), 0.75)^2 * I(2) rtol=1e-12
end

@stestset "independent blocks without an ID stay independent NLME blocks" begin
    m = brm_nlme_model(RKBRMI(unnamed_pk(UNNAMED_DATA)); ad_backend=NLME_AD)
    blocks = unique(b.block for b in m.view.block_margins)
    @test length(blocks) == 2
    @test [b.predictor for b in blocks] == [:log_CL, :log_V]
    @test m.eta_blocks == [1, 1]
    @test nomega(nlme_layout(m)) == 2
    # A point of the model reads back as a diagonal covariance.
    u = [0.1 * sin(1.3k) for k in eachindex(m.view.coordinates)]
    from_point = brm_nlme_start(m, u)
    @test isdiag(from_point.omega_matrix)
    start = brm_nlme_start(m)
    @test start.omega_matrix ≈ I(2) rtol=1e-12 # LogNormal(0, 1) default scales
    σ = Dict(zip(m.view.coordinates[m.sigma], exp.(start.sigma)))
    @test σ[:s0] ≈ log(2) rtol=1e-12
    @test σ[:sigma] ≈ log(2)^2 rtol=1e-12 # Exponential(s0) at s0's median
    flat = Symbol("pop_log_CL.beta_pop_2")
    @test start.start_source.zero == [flat]
    @test named_theta(m, start.theta)[flat] == 0.0
    @test isfinite(start.population_logprior)
    e = brm_nlme_estimates(m, start.theta, start.sigma, start.omega_matrix)
    @test keys(e.point) == (:log_CL, :log_V, :s0, :sigma,
        Symbol("log_CL|subject"), Symbol("log_V|subject"), :other)
    @test keys(e.point.log_CL) == (:Intercept, :weight, :sex) # formula order
    @test e.point.log_CL.sex == ["m" => 0.0] # level pairs, as brm_start takes them
    @test e.point[Symbol("log_V|subject")].margins ==
        [(; predictor=:log_V, coefficient=:Intercept)]
    @test e.point[Symbol("log_CL|subject")].sd ≈ [1.0] rtol=1e-12
end

@stestset "estimates name a point in natural units" begin
    m = brm_nlme_model(RKBRMI(authored_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    names = m.view.coordinates[m.theta]
    values = Dict(Symbol("pop_log_CL.beta_pop.1") => 0.25,
        Symbol("pop_log_CL.beta_pop.2") => -0.15, :log_V_Intercept => 1.4)
    θ = [values[n] for n in names]
    σ = [log(0.8)]
    Ω = [0.09 0.012; 0.012 0.04]
    inputs = deepcopy((θ, σ, Ω))
    e = brm_nlme_estimates(m, θ, σ, Ω)
    @test (θ, σ, Ω) == inputs
    @test keys(e.point) == (:log_CL, :log_V, :sigma, :pk, :other)
    @test e.point.log_CL == (; Intercept=0.25, weight=-0.15)
    @test e.point.log_V == (; Intercept=1.4)
    @test e.point.sigma ≈ 0.8 rtol=1e-12
    pk = e.point.pk
    @test pk.group === :subject
    @test pk.margins == [(; predictor=:log_CL, coefficient=:Intercept),
        (; predictor=:log_V, coefficient=:Intercept)]
    @test pk.covariance == Ω
    @test pk.sd ≈ [0.3, 0.2] rtol=1e-12
    @test pk.cor ≈ [1 0.2; 0.2 1] rtol=1e-12
    @test e.point.other == (;)
    @test e.eta === nothing
    ω = omega_parameters(nlme_layout(m), Ω)
    ebes = empirical_bayes(m, θ, σ, ω)
    e = brm_nlme_estimates(m, θ, σ, Ω, ebes)
    @test e.eta.subjects == ["a", "b", "c"]
    @test e.eta.names == ["log_CL:Intercept", "log_V:Intercept"]
    @test e.eta.values == ebes.modes
    # The named point is what address-keyed starts take: its entries give
    # back the same θ, σ and Ω.
    back = brm_nlme_start(m, (; log_CL=e.point.log_CL, log_V=e.point.log_V,
        sigma=e.point.sigma, pk=(; sd=e.point.pk.sd, cor=e.point.pk.cor)))
    @test back.theta ≈ θ rtol=1e-10
    @test back.sigma ≈ σ rtol=1e-10
    @test back.omega_matrix ≈ Ω rtol=1e-10
    @test_throws DimensionMismatch brm_nlme_estimates(m, θ[1:2], σ, Ω)
    @test_throws DimensionMismatch brm_nlme_estimates(m, θ, σ, Ω[1:1, 1:1])
    # refused: predictors, scalars and blocks share the result's one namespace
    # (keyed by the model's identifiers, decision 10kdjsv), so a scalar `pk`
    # beside block `|pk|` has no unambiguous name.
    clash = @brm NLME_PLATE_DATA begin
        pk ~ Exponential(1)
        log_CL ~ 1 + (1 | pk | subject)
        log_V ~ 1 + (1 | pk | subject)
        @plate for i in eachindex(log_CL)
            CL = exp(log_CL[i])
            Vc = exp(log_V[i])
            pred[i] = dose[i] / Vc .* exp.(-(CL / Vc) .* t[i])
            dv[i] ~ normal(pred[i], pk)
        end
    end
    mc = brm_nlme_model(RKBRMI(clash); ad_backend=NLME_AD)
    s = brm_nlme_start(mc)
    @test_throws BRMNLMEViewError brm_nlme_estimates(mc, s.theta, s.sigma, s.omega_matrix)
end

@stestset "brm_nlme_fit runs NLMEEstimation's estimators from the prior medians" begin
    brmi = authored_pk(NLME_PLATE_DATA)
    m = brm_nlme_model(RKBRMI(brmi); ad_backend=NLME_AD)
    start = brm_nlme_start(m)
    # Posthoc EBEs at the start's population, from any of the three inputs.
    direct = empirical_bayes(m, start.theta, start.sigma, start.omega; η0=start.eta)
    for x in (brmi, RKBRMI(brmi), m)
        f = brm_nlme_fit(x, empirical_bayes)
        @test f.start.theta == start.theta && f.start.sigma == start.sigma
        @test f.ebes === f.result
        @test f.estimates.point == brm_nlme_estimates(m, start.theta, start.sigma,
            start.omega_matrix).point
        @test f.estimates.eta.values ≈ direct.modes rtol=1e-8 atol=1e-10
    end
    @test brm_nlme_fit(m, posthoc).estimates.eta.values ≈ direct.modes rtol=1e-8 atol=1e-10

    f = brm_nlme_fit(m, saem; rng=Xoshiro(1), options=SAEMOptions(iterations=10, burnin=5))
    r = f.result
    @test r isa SAEMResult && r.objective === :MAP
    @test f.estimates.point.sigma ≈ exp(only(r.sigma)) rtol=1e-12
    @test f.estimates.point.pk.covariance == r.omega_matrix
    @test named_theta(m, r.theta)[:log_V_Intercept] == f.estimates.point.log_V.Intercept
    # SAEM's own eta are chain states; the EBEs come from a posthoc pass at the
    # fitted population point.
    ebes = empirical_bayes(m, r.theta, r.sigma, r.omega; subjects=r.subjects, η0=r.eta)
    @test f.ebes.modes ≈ ebes.modes rtol=1e-8 atol=1e-10
    @test f.estimates.eta.values == f.ebes.modes
    g = brm_nlme_fit(m, saem; ebes=false, rng=Xoshiro(1),
        options=SAEMOptions(iterations=10, burnin=5))
    @test g.ebes === nothing && g.estimates.eta === nothing
    @test g.result.theta == r.theta

    f = brm_nlme_fit(m, focei; options=FOCEOptions(maxeval=30))
    @test f.result isa FOCEResult && f.result.objective === :MAP
    @test f.ebes === f.result.ebes
    @test f.estimates.point.pk.covariance == f.result.omega_matrix
    @test f.estimates.eta.values == f.result.ebes.modes

    # Caller-supplied starts override the prior medians; estimator keywords
    # pass through.
    u = [0.2 * cos(0.7k) for k in eachindex(m.view.coordinates)]
    f = brm_nlme_fit(m, empirical_bayes; start=u, method=QuasiNewtonEBE())
    @test f.start.theta == u[m.theta] && f.start.sigma == u[m.sigma]
    @test f.result.method isa QuasiNewtonEBE
    @test brm_nlme_fit(m, empirical_bayes; start=f.start,
        method=QuasiNewtonEBE()).start === f.start
    g = brm_nlme_fit(m, saem; start=u, ebes=false, rng=Xoshiro(3),
        options=SAEMOptions(iterations=4, burnin=2))
    @test g.start.theta == u[m.theta]
    # Address-keyed values override the prior medians where they address.
    f = brm_nlme_fit(m, empirical_bayes; inits=(; log_V=(; Intercept=1.2), sigma=0.5))
    θ = named_theta(m, f.start.theta)
    @test θ[:log_V_Intercept] ≈ 1.2 rtol=1e-12
    @test θ[Symbol("pop_log_CL.beta_pop.1")] == 0.3 # still its prior median
    @test exp(only(f.start.sigma)) ≈ 0.5 rtol=1e-12
    @test f.start.omega_matrix ≈ start.omega_matrix rtol=1e-12
    # refused: a packed point and address-keyed values would compete.
    @test_throws ArgumentError brm_nlme_fit(m, empirical_bayes; start=u,
        inits=(; sigma=0.5))
    # The default NewtonEBE from this start: its first trials all lack a step
    # (NLMEEstimation snag newtonebe-trial-97825f7d, fixed in 1dd8b86).
    f = brm_nlme_fit(m, empirical_bayes; start=u)
    q = empirical_bayes(m, f.start.theta, f.start.sigma, f.start.omega;
        η0=f.start.eta, method=QuasiNewtonEBE())
    @test f.estimates.eta.values ≈ q.modes rtol=1e-6 atol=1e-8
    # refused: a prepared model already fixes its grouping and AD backend, so
    # model-building keywords given with it would otherwise be ignored silently.
    @test_throws ArgumentError brm_nlme_fit(m, empirical_bayes; subject_group=:subject)
    # refused: the estimator is one of NLMEEstimation's own functions.
    @test_throws ArgumentError brm_nlme_fit(m, sum)
    # refused: a start needs a whole population point (theta, sigma, omega).
    @test_throws ArgumentError brm_nlme_fit(m, empirical_bayes; start=(; theta=start.theta))
end

@stestset "brm_nlme_fit starts from NLMEEstimation's own starts and results" begin
    m = brm_nlme_model(RKBRMI(authored_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    start = brm_nlme_start(m)
    θ, σ, ω = start.theta, start.sigma, start.omega
    # θ, σ and ω alone: Ω is derived, and the subject effects start at zero.
    f = brm_nlme_fit(m, empirical_bayes; start=(; theta=θ, sigma=σ, omega=ω))
    @test f.start.omega_matrix ≈ start.omega_matrix rtol=1e-12
    @test f.result.modes == empirical_bayes(m, θ, σ, ω).modes
    @test f.estimates.point == brm_nlme_estimates(m, θ, σ, start.omega_matrix).point
    a = brm_nlme_fit(m, saem; start=(; theta=θ, sigma=σ, omega=ω), ebes=false,
        rng=Xoshiro(1), options=SAEMOptions(iterations=4, burnin=2))
    b = brm_nlme_fit(m, saem; ebes=false, rng=Xoshiro(1),
        options=SAEMOptions(iterations=4, burnin=2))
    @test a.result.theta == b.result.theta && a.result.eta == b.result.eta
    # Subject effects are matched to the estimator's subjects by subject. One
    # EBE iteration from each start tells the warm starts apart.
    H = [0.3 -0.2 0.1; -0.1 0.25 0.2]
    one_step = (; options=EBEOptions(maxiter=1), require_convergence=false)
    s = (; theta=θ, sigma=σ, omega=ω, eta=H[:, [3, 1]], subjects=["c", "a"])
    f = brm_nlme_fit(m, empirical_bayes; start=s, subjects=[1, 3], one_step...)
    expected = empirical_bayes(m, θ, σ, ω; subjects=[1, 3], η0=H[:, [1, 3]], one_step...)
    @test f.result.modes == expected.modes
    @test f.result.modes != empirical_bayes(m, θ, σ, ω; subjects=[1, 3],
        η0=H[:, [3, 1]], one_step...).modes
    # refused: a start names its subjects by grouping level, as brm_nlme_start
    # does; an index would be ambiguous with an integer-coded grouping column.
    @test_throws ArgumentError brm_nlme_fit(m, empirical_bayes;
        start=(; s..., subjects=[3, 1]), subjects=[3])
    @test_throws ArgumentError brm_nlme_fit(m, empirical_bayes;
        start=(; s..., subjects=["c", "z"]), subjects=[3])
    # Without `subjects` of its own, a start's η belongs to every subject.
    f = brm_nlme_fit(m, empirical_bayes; start=(; theta=θ, sigma=σ, omega=ω, eta=H),
        subjects=[2, 3], one_step...)
    @test f.result.modes == empirical_bayes(m, θ, σ, ω; subjects=[2, 3],
        η0=H[:, [2, 3]], one_step...).modes
    # The caller's η0 replaces the start's effects, and is the only way to
    # start a subject the start has none for.
    z = zeros(2, 1)
    f = brm_nlme_fit(m, empirical_bayes; start=s, subjects=[2], η0=z, one_step...)
    @test f.result.modes == empirical_bayes(m, θ, σ, ω; subjects=[2], η0=z,
        one_step...).modes
    # refused: subject 2 has no effects in the start and no η0 was given;
    # starting it elsewhere would be a silent default (dev §1).
    @test_throws ArgumentError brm_nlme_fit(m, empirical_bayes; start=s, subjects=[2])
    @test_throws ArgumentError brm_nlme_fit(m, empirical_bayes; start=s)
    # refused: an Ω that is not ω's covariance would name the estimates wrongly.
    @test_throws ArgumentError brm_nlme_fit(m, empirical_bayes;
        start=(; theta=θ, sigma=σ, omega=ω, omega_matrix=2start.omega_matrix))

    # An SAEM result over some subjects: its population point, warm-started from
    # those subjects' final chain states, as a later EBE pass needs it.
    options = SAEMOptions(iterations=6, burnin=3)
    fit = brm_nlme_fit(m, saem; subjects=[1, 3], rng=Xoshiro(4), options)
    r = fit.result
    @test r.subjects == [1, 3] && size(r.eta) == (2, 2)
    e = brm_nlme_fit(m, empirical_bayes; start=r, subjects=r.subjects)
    @test e.result.modes == fit.ebes.modes # the pass `ebes=true` ran
    @test e.start.omega_matrix == r.omega_matrix
    @test e.start.subjects == ["a", "c"]
    @test e.estimates.point == fit.estimates.point
    @test e.estimates.eta == fit.estimates.eta
    tuned = (; options=EBEOptions(maxiter=20), require_convergence=false)
    e = brm_nlme_fit(m, empirical_bayes; start=r, subjects=[3, 1], tuned...)
    @test e.result.modes == empirical_bayes(m, r.theta, r.sigma, r.omega;
        subjects=[3, 1], η0=r.eta[:, [2, 1]], tuned...).modes
    @test_throws ArgumentError brm_nlme_fit(m, empirical_bayes; start=r)
    # `ebes` keywords tune saem's own follow-up pass in the same call.
    t = brm_nlme_fit(m, saem; subjects=[1, 3], rng=Xoshiro(4), options, ebes=one_step)
    @test t.result.theta == r.theta && t.result.eta == r.eta
    @test t.ebes.modes == empirical_bayes(m, r.theta, r.sigma, r.omega;
        subjects=r.subjects, η0=r.eta, one_step...).modes
    @test t.ebes.modes != fit.ebes.modes
    @test t.estimates.eta.values == t.ebes.modes
    # refused: the SAEM result fixes the pass's subjects and warm start.
    @test_throws ArgumentError brm_nlme_fit(m, saem; ebes=(; subjects=[1]))
    @test_throws ArgumentError brm_nlme_fit(m, saem; ebes=(; η0=r.eta))
    @test_throws ArgumentError brm_nlme_fit(m, saem; ebes=:defaults)
    # refused: focei and empirical_bayes take EBE settings as their own keywords,
    # and silently ignoring them here would drop the caller's settings (dev §1).
    @test_throws ArgumentError brm_nlme_fit(m, focei; ebes=one_step)
    @test_throws ArgumentError brm_nlme_fit(m, empirical_bayes; ebes=one_step)
    # SAEM continues from an SAEM result.
    g = brm_nlme_fit(m, saem; start=r, subjects=r.subjects, ebes=false,
        rng=Xoshiro(5), options)
    @test g.result.theta == saem(m, r.theta, r.sigma, r.omega; subjects=r.subjects,
        η0=r.eta, rng=Xoshiro(5), options).theta
    # A FOCE(I) result warm-starts from its EBEs.
    fo = brm_nlme_fit(m, focei; options=FOCEOptions(maxeval=20)).result
    e = brm_nlme_fit(m, empirical_bayes; start=fo, one_step...)
    @test e.result.modes == empirical_bayes(m, fo.theta, fo.sigma, fo.omega;
        η0=fo.ebes.modes, one_step...).modes
end

@stestset "a nested model fits with one covariance per family and compact EBEs" begin
    groups = (; subject_group=:subject, nested_groups=(unit=:obs_subject,))
    f = brm_nlme_fit(nested_model(NESTED_DATA), saem; groups...,
        rng=Xoshiro(2), options=SAEMOptions(iterations=8, burnin=4))
    @test f.result isa RaggedSAEMResult
    point = f.estimates.point
    @test keys(point) == (:mu, :sigma, :p, :u, :other)
    @test point.p.group === :subject && point.u.group === :unit
    @test point.p.covariance == f.result.omega_matrix[1]
    @test point.u.covariance == f.result.omega_matrix[2]
    eta = f.estimates.eta
    @test eta.subjects == ["s1", "s2", "s3"]
    @test length.(eta.values) == [2, 3, 4]
    @test eta.names[2] == ["subject[s2]:mu:Intercept",
        "unit[u2]:unit_shift:Intercept", "unit[u3]:unit_shift:Intercept"]
    # The ragged SAEM result starts a later EBE pass with compact effects.
    e = brm_nlme_fit(f.model, empirical_bayes; start=f.result, subjects=[3, 1])
    @test e.result.modes == empirical_bayes(f.model, f.result.theta, f.result.sigma,
        f.result.omega; subjects=[3, 1], η0=f.result.eta[[3, 1]]).modes
    @test length.(e.result.modes) == [4, 2]
end

# The state glyph of every row of a rendered progress tree, keyed by the row's
# label (the text after the glyph, without counter, message or duration).
function progress_states(node)
    rows = Dict{String,Char}()
    for line in split(render_text(node), '\n')
        m = match(r"^[│├└─\s]*(\S) (.+?)(?: \(\d+/\d+\))?(?: — .*)?(?: \[[^\]]*\])?(?: · .*)?$", line)
        m === nothing || (rows[m[2]] = only(m[1]))
    end
    rows
end

@stestset "brm_nlme_fit reports one progress tree" begin
    brmi = authored_pk(NLME_PLATE_DATA)
    options = SAEMOptions(iterations=10, burnin=5)
    tree = Ref{Any}(nothing)
    f = with_progress(:state; description="fit") do p
        tree[] = p
        brm_nlme_fit(brmi, saem; progress=p, rng=Xoshiro(1), options)
    end
    rows = progress_states(tree[])
    # BRM's preparation, the estimator's own node and the EBE pass's, under one
    # node, all finished.
    for label in ("BRM NLME fit", "Lower to ReactiveKernels", "Prepare NLME model",
            "Subject partition", "Pointwise sampler", "Observation attribution",
            "η Hessian-vector product", "Gaussian observations", "Subject cells",
            "Population prior", "Starting point", "Estimate (saem)", "SAEM",
            "MCMC burn-in", "Stochastic approximation", "Posthoc EBEs",
            "Empirical Bayes (Newton)")
        @test get(rows, label, nothing) == '✓'
    end
    @test all(==('✓'), values(rows))
    # Reporting changes no number.
    g = brm_nlme_fit(f.model, saem; rng=Xoshiro(1), options)
    @test f.result.theta == g.result.theta && f.result.omega == g.result.omega
    @test f.ebes.modes == g.ebes.modes

    # A prepared model reuses both preparation stages; the EBE pass only
    # follows saem.
    e = with_progress(:state; description="posthoc") do p
        tree[] = p
        brm_nlme_fit(f.model, empirical_bayes; progress=p)
    end
    rows = progress_states(tree[])
    @test rows["Lower to ReactiveKernels"] == '↺' && rows["Prepare NLME model"] == '↺'
    @test rows["Starting point"] == '✓' && rows["Estimate (empirical_bayes)"] == '✓'
    @test !haskey(rows, "Posthoc EBEs") && !haskey(rows, "Subject partition")
    # A backend symbol starts the fit's own root.
    @test brm_nlme_fit(f.model, empirical_bayes; progress=:state).ebes.modes ==
        e.ebes.modes
    @test_throws ArgumentError brm_nlme_fit(f.model, empirical_bayes; progress="tree")

    # A step a model does not need stays skipped: a nested model has no
    # Gaussian observation protocol.
    with_progress(:state; description="nested") do p
        tree[] = p
        brm_nlme_model(RKBRMI(nested_model(NESTED_DATA)); ad_backend=NLME_AD,
            subject_group=:subject, nested_groups=(unit=:obs_subject,), progress=p)
    end
    rows = progress_states(tree[])
    @test rows["Prepare NLME model"] == '✓' && rows["Subject partition"] == '✓'
    @test rows["Gaussian observations"] == '⊘'
end

@stestset "an interrupt stops brm_nlme_fit and keeps saem's result" begin
    brmi = authored_pk(NLME_PLATE_DATA)
    tree = Ref{Any}(nothing)
    # Before any result exists, the next preparation phase raises.
    with_progress(:state; description="stopped") do p
        tree[] = p
        request_interrupt!(p)
        @test_throws ProgressInterrupted brm_nlme_fit(brmi, saem; progress=p)
    end
    rows = progress_states(tree[])
    @test rows["Lower to ReactiveKernels"] == '✗' && rows["Prepare NLME model"] == '⊘'
    @test rows["Estimate (saem)"] == '⊘' && rows["Posthoc EBEs"] == '⊘'

    # Once saem runs it honours the interrupt: an interrupt it sees at its
    # first iteration boundary returns the start with `interrupted = true`, and
    # the EBE pass, which would raise over that result, is skipped. The fit's
    # own estimator step runs here with the request already made, so the test
    # needs no second thread (a native Enzyme gradient beside an allocating
    # thread is unsafe, reactivekernels-use §7ao).
    m = brm_nlme_model(RKBRMI(brmi); ad_backend=NLME_AD)
    start = brm_nlme_start(m)
    fit_ext = Base.get_extension(BRM, :BayesianRegressionModelsNLMEEstimationExt)
    result, ebe, estimates = with_progress(:state; description="interrupted") do p
        tree[] = p
        with_prepared_phases(p, fit_ext._fit_phases(:saem, (;))) do phases
            request_interrupt!(p)
            fit_ext._fit_run(saem, m, start, (;), phases; rng=Xoshiro(1),
                options=SAEMOptions(iterations=10, burnin=5))
        end
    end
    @test result.interrupted && result.iterations == 0
    @test result.theta == start.theta && result.sigma == start.sigma
    @test ebe === nothing && estimates.eta === nothing
    rows = progress_states(tree[])
    @test rows["Estimate (saem)"] == '✓' && rows["SAEM"] == '✓'
    @test rows["Posthoc EBEs"] == '⊘'
end
