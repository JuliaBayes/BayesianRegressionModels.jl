# Public synthetic values crossing formula row axes. No consumer model or
# private parameter packing is reproduced here.
# Run: julia --project=test test/rk_values.jl [testset substring ...]
using Test, BayesianRegressionModels, Distributions, Enzyme, LogDensityProblems
using ReactiveKernels, ReactiveKernelsPPL
using DifferentiationInterface: AutoEnzyme
import LogDensityProblems: dimension, logdensity, logdensity_and_gradient
using LinearAlgebra: cholesky, Symmetric, diagind
using Statistics: mean
using LogExpFunctions: logit
include(joinpath(@__DIR__, "testset_filter.jl"))
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "rk_source_roundtrip.jl"))

function value_query(backend, name, u)
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    translated = ext._rk_translated_plan(backend.plan)
    Base.invokelatest(prepare_query(backend.model, translated, name), u)
end

function check_value_gradient(backend)
    check_rk_source_roundtrip(backend)
    problem = rk_logdensity_problem(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    N = dimension(problem)
    u = N <= 1 ? fill(0.13, N) : collect(range(-0.18, 0.22; length=N))
    value, gradient = logdensity_and_gradient(problem, u)
    @test isfinite(value)
    @test all(isfinite, gradient)
    h = 1e-5
    fd = map(eachindex(u)) do j
        plus, minus = copy(u), copy(u)
        plus[j] += h
        minus[j] -= h
        (logdensity(problem, plus) - logdensity(problem, minus)) / (2h)
    end
    @test gradient ≈ fd atol=2e-5 rtol=2e-5
    u
end

# Plain Julia reader receives whole named predictor arrays. Its keyword and
# gathered row indices are part of the authored call, not a panel convention.
function synthetic_reader(t, a, b, c, subject_row, secondary_row; multiplier=1.0)
    result = Vector{Float64}(undef, length(t))
    for j in eachindex(t)
        result[j] = multiplier * (a[subject_row[j]] + b[subject_row[j]] * t[j]) +
            c[secondary_row[j]]
    end
    result
end

const mixed_axes = (;
    subject=["c", "a", "b"], x=[0.2, 0.5, -0.8],
    w=[-0.7, 0.4, 0.9, 1.3],
    t=[0.1, 0.4, 0.8, 1.2, 1.8],
    subject_row=[2, 1, 3, 2, 1], secondary_row=[4, 2, 1, 3, 2],
    y=[0.3, 0.5, -0.2, 0.8, 0.1])

@stestset "callable readers of correlated subject and secondary predictors" begin
    brmi = @brm mixed_axes begin
        a ~ 1 + x + (1 | p | subject)
        log(b) ~ 1 + (1 | p | subject)
        c ~ 1 + w
        multiplier ~ Normal(0, 1)
        reads = synthetic_reader(t, a, b, c, subject_row, secondary_row; multiplier)
        sigma ~ Exponential(1)
        y ~ Normal(reads, sigma)
    end
    backend = RKBRMI(brmi)
    @test backend.plan isa BRM._RKValuePlan
    @test length(backend.plan.columns[:subject]) == 3
    @test length(backend.plan.columns[:w]) == 4
    @test length(backend.plan.columns[:y]) == 5
    @test any(pair -> last(pair) === synthetic_reader, BRM._rk_emit_ast(backend.plan).bindings)
    u = check_value_gradient(backend)
    nt = constrain(backend.model.layout, u)
    # Independent population and correlation reconstruction, including the
    # sorted label map (input rows are c,a,b) and b's public inverse link.
    group = nt.b_p_subject
    C = group.z * (group.tau .* group.L)'
    a = nt.pop_a.beta_pop[1] .+ nt.pop_a.beta_pop[2] .* mixed_axes.x .+ C[[3, 1, 2], 1]
    b = exp.(only(nt.pop_log_b.beta_pop) .+ C[[3, 1, 2], 2])
    c = nt.pop_c.beta_pop[1] .+ nt.pop_c.beta_pop[2] .* mixed_axes.w
    expected = [nt.multiplier * (a[mixed_axes.subject_row[j]] +
        b[mixed_axes.subject_row[j]] * mixed_axes.t[j]) +
        c[mixed_axes.secondary_row[j]] for j in eachindex(mixed_axes.y)]
    pointwise = value_query(backend, :pointwise, u)
    expected_ll = logpdf.(Normal.(expected, nt.sigma), mixed_axes.y)
    @test pointwise.y ≈ expected_ll
    @test value_query(backend, :likelihood, u) ≈ sum(expected_ll)
    normal_draws = [nt.pop_a.beta_pop; nt.pop_log_b.beta_pop; nt.pop_c.beta_pop; nt.multiplier]
    prior = sum(logpdf.(Normal(), normal_draws)) +
        sum(logpdf.(Normal(), group.z)) +
        # Default shared scales restrict the Normal kernel to positive values;
        # explicit normalized half-Normal priors alone add log(2).
        sum(logpdf.(Normal(), group.tau)) -
        nt.sigma - log(2) # Exponential(1), then LKJCholesky(2,1)
    @test value_query(backend, :prior, u) ≈ prior
    @test value_query(backend, :sampler, u) ≈
        prior + sum(expected_ll) + logjac(backend.model.layout, u)
    artifact = BRM.emit_rk_artifact(brmi; case_id="synthetic-mixed-axes")
    rebuilt = build_kernel(BRM.rk_translate_artifact(artifact))
    @test coordinate_names(rebuilt.layout) == coordinate_names(backend.model.layout)
    @test length(artifact.plan.regression.ranef_buckets) == 1
end

function shared_reader(t, h, f, theta; direction=1.0)
    result = Vector{Float64}(undef, length(t))
    for j in eachindex(t)
        result[j] = theta * h[j] + direction * f[j] + t[j]
    end
    result
end

@stestset "shared HSGP and exact GP values across readers" begin
    df = (; t=[-0.8, -0.2, 0.3, 0.7, 1.2],
        y=[0.2, 0.4, -0.1, 0.7, 0.9], y2=[-0.3, 0.1, 0.5, 0.2, -0.4])
    backend = RKBRMI(@brm df begin
        h ~ hsgp(t; k=3)
        f ~ gp(t)
        theta ~ Normal(0, 1)
        reads = shared_reader(t, h, f, theta)
        reads2 = shared_reader(t, h, f, theta; direction=-1.0)
        sigma ~ Exponential(1)
        y ~ Normal(reads, sigma)
        y2 ~ Normal(reads2, sigma)
    end)
    u = check_value_gradient(backend)
    nt = constrain(backend.model.layout, u)
    # An independent exact covariance and Cholesky calculation pins the GP
    # value. Both readers consume the same sampled hyperparameters/latents.
    gp_term = only(t for p in backend.plan.regression.predictors for t in p.terms if t.kind === :gp)
    opts = gp_term.options
    gp_draws = getproperty(nt, opts.f)
    rho, amplitude, z = gp_draws.rho, gp_draws.sigma, gp_draws.z
    covariance = [amplitude^2 * exp(-0.5 * ((x-y)/rho)^2) for x in df.t, y in df.t]
    covariance[diagind(covariance)] .+= opts.jitter
    f = cholesky(Symmetric(covariance)).L * z
    # Independent one-axis Hilbert basis and exp-quad spectral weights.
    hs = (; sigma=nt.hsgp_t.sigma, rho=nt.hsgp_t.rho_iso, z=nt.hsgp_t.beta_raw)
    centered = df.t .- mean(df.t)
    L = 1.5maximum(abs, centered)
    omega = (1:3) .* (pi / (2L))
    PHI = [sin(w * (x + L)) / sqrt(L) for x in centered, w in omega]
    weights = sqrt.(hs.sigma^2 * sqrt(2pi) * hs.rho .* exp.(-0.5 .* hs.rho^2 .* omega.^2))
    h = PHI * (weights .* hs.z)
    r1 = nt.theta .* h .+ f .+ df.t
    r2 = nt.theta .* h .- f .+ df.t
    pointwise = value_query(backend, :pointwise, u)
    @test pointwise.y ≈ logpdf.(Normal.(r1, nt.sigma), df.y)
    @test pointwise.y2 ≈ logpdf.(Normal.(r2, nt.sigma), df.y2)
    @test value_query(backend, :likelihood, u) ≈ sum(pointwise.y) + sum(pointwise.y2)
    @test count(n -> occursin("rho", string(n)), coordinate_names(backend.model.layout)) == 2
end

module ReaderOne
    readout(t, a) = t .* a
end
module ReaderTwo
    readout(t, a) = -t .* a
end

@stestset "exact callable bindings and ordinary data definitions" begin
    schedule(t) = reverse(t)
    fixed_reader = let multiplier=2.5
        (t, a) -> multiplier .* t .* a
    end
    df = (; t=[0.2, 0.6, 1.1], y=[0.1, -0.3, 0.7],
        y2=[-0.1, 0.2, 0.5], y3=[0.6, -0.2, 0.4])
    backend = RKBRMI(@brm df begin
        a ~ Normal(0, 1)
        times = schedule(t)
        r1 = ReaderOne.readout(times, a)
        r2 = ReaderTwo.readout(times, a)
        r3 = fixed_reader(times, a)
        y ~ Normal(r1, 1)
        y2 ~ Normal(r2, 1)
        y3 ~ Normal(r3, 1)
    end)
    u = check_value_gradient(backend)
    expected = reverse(df.t) .* only(u)
    pointwise = value_query(backend, :pointwise, u)
    @test pointwise.y ≈ logpdf.(Normal.(expected, 1), df.y)
    @test pointwise.y2 ≈ logpdf.(Normal.(-expected, 1), df.y2)
    @test pointwise.y3 ≈ logpdf.(Normal.(2.5expected, 1), df.y3)
    @test length(coordinate_names(backend.model.layout)) == 1
end

read_rows(value, rows) = value[rows]

@stestset "ordinary AR and DAR arrays on a different observation axis" begin
    df = (; t=collect(1.0:5.0), rows=[5, 2, 4], y=[0.4, -0.2, 0.7])
    saved = deepcopy(df)
    for kind in (:ar, :dar)
        brmi = kind === :ar ? (@brm df begin
            mu ~ 1 + ar(t; p=1)
            reads = read_rows(mu, rows)
            y ~ Normal(reads, 1)
        end) : (@brm df begin
            mu ~ 1 + dar(t)
            reads = read_rows(mu, rows)
            y ~ Normal(reads, 1)
        end)
        backend = RKBRMI(brmi)
        u = check_value_gradient(backend)
        nt = constrain(backend.model.layout, u)
        path = zeros(5)
        prior, jac = if kind === :ar
            z = getproperty(nt.ar_mu_t, :_ppl_scan_z_state)
            path[1] = z[1]
            for i in 2:5
                path[i] = tanh(nt.ar_mu_t.phi_raw) * path[i - 1] + z[i]
            end
            path .*= nt.mu_ar_mu_t
            @test length(u) == 8
            (logpdf(Normal(), only(nt.pop_mu.beta_pop)) + logpdf(Normal(), nt.mu_ar_mu_t) +
                logpdf(Normal(), nt.ar_mu_t.phi_raw) + sum(logpdf.(Normal(), z)), 0.0)
        else
            z = getproperty(nt.dar_mu_t_level, :_ppl_scan_z_level)
            increment = 0.0
            for i in 2:5
                increment = nt.dar_mu_t_level.beta * increment + nt.dar_mu_t_level.sigma * z[i - 1]
                path[i] = path[i - 1] + increment
            end
            @test length(u) == 7
            (logpdf(Normal(), only(nt.pop_mu.beta_pop)) +
                logpdf(truncated(Normal(0.5, 0.2), 0, 1), nt.dar_mu_t_level.beta) +
                logpdf(truncated(Normal(0, 0.2), 0, Inf), nt.dar_mu_t_level.sigma) +
                sum(logpdf.(Normal(), z)),
                log(nt.dar_mu_t_level.beta) + log1p(-nt.dar_mu_t_level.beta) + log(nt.dar_mu_t_level.sigma))
        end
        likelihood = sum(logpdf.(Normal.(only(nt.pop_mu.beta_pop) .+ path[df.rows], 1), df.y))
        @test value_query(backend, :likelihood, u) ≈ likelihood
        @test value_query(backend, :prior, u) ≈ prior
        @test value_query(backend, :sampler, u) ≈ likelihood + prior + jac
        @test isequal(df, saved)
    end
end

@stestset "ordinary smooth arrays on a different observation axis" begin
    df = (; x=collect(range(-1.3, 1.5; length=12)), z=sin.((1:12) ./ 2),
        rows=[5, 2, 12, 2], y=[0.4, -0.2, 0.7, 0.1])
    backend = RKBRMI(@brm df begin
        mu ~ s(x) + t2(x, z; k=(4, 4)) + hsgp(x, z; k=(2, 2), iso=false)
        reads = read_rows(mu, rows)
        y ~ Normal(reads, 1)
    end)
    u = check_value_gradient(backend)
    pointwise = value_query(backend, :pointwise, u)
    @test length(pointwise.y) == 4
    @test value_query(backend, :likelihood, u) ≈ sum(pointwise.y)
end

@stestset "ordinary multi-membership and stratified arrays" begin
    df = (; g=["a", "a", "b", "c", "c"], h=["c", "b", "a", "a", "b"],
        stratum=["A", "A", "A", "B", "B"], x=[-0.5, 0.1, 0.8, 1.2, -0.3],
        rows=collect(1:5), y=[0.1, -0.2, 0.3, 0.7, 0.4])
    for grouped in (:membership, :stratified)
        brmi = grouped === :membership ? (@brm df begin
            mu ~ 1 + (1 | mm(g, h))
            reads = read_rows(mu, rows)
            y ~ Normal(reads, 1)
        end) : (@brm df begin
            mu ~ 1 + (1 + x | gr(g; by=stratum))
            reads = read_rows(mu, rows)
            y ~ Normal(reads, 1)
        end)
        backend = RKBRMI(brmi)
        u = check_value_gradient(backend)
        nt = constrain(backend.model.layout, u)
        # The group component owns its scales, factors and standardized draws.
        group = getproperty(nt, only(filter(n -> startswith(string(n), "b_"), propertynames(nt))))
        scale, raw = group.tau, group.z
        intercept = only(nt.pop_mu.beta_pop)
        gi = [1, 1, 2, 3, 3]
        expected = if grouped === :membership
            hi = [3, 2, 1, 1, 2]
            effects = only(scale) .* vec(raw)
            intercept .+ (effects[gi] .+ effects[hi]) ./ 2
        else
            factors = group.L
            si = [1, 1, 1, 2, 2]
            map(eachindex(df.y)) do j
                effects = (scale[si[j], :] .* factors[:, :, si[j]]) * raw[gi[j], :]
                intercept + effects[1] + df.x[j] * effects[2]
            end
        end
        pointwise = value_query(backend, :pointwise, u)
        @test pointwise.y ≈ logpdf.(Normal.(expected, 1), df.y)
    end
end

probit(p) = quantile(Normal(), p)
cloglog(p) = log(-log1p(-p))
blend_values(p, q, r, rows) = p[rows] .+ q[rows] .+ r[rows]

@stestset "ordinary readers receive inverse links" begin
    df = (; x=[-0.4, 0.2, 0.9], rows=[3, 1, 2, 1], y=[0.1, 0.4, -0.2, 0.8])
    backend = RKBRMI(@brm df begin
        logit(p) ~ 1 + x
        probit(q) ~ 1 + x
        cloglog(r) ~ 1 + x
        reads = blend_values(p, q, r, rows)
        y ~ Normal(reads, 1)
    end)
    u = check_value_gradient(backend)
    nt = constrain(backend.model.layout, u)
    # Population components are named after each linked target (SBBRMI's names).
    linear(beta) = beta[1] .+ beta[2] .* df.x
    p = 1 ./ (1 .+ exp.(-linear(nt.pop_logit_p.beta_pop)))
    q = cdf.(Normal(), linear(nt.pop_probit_q.beta_pop))
    r = -expm1.(-exp.(linear(nt.pop_cloglog_r.beta_pop)))
    expected = p[df.rows] .+ q[df.rows] .+ r[df.rows]
    @test value_query(backend, :pointwise, u).y ≈ logpdf.(Normal.(expected, 1), df.y)
end

@stestset "ordinary categorical random slopes" begin
    df = (; g=[2, 1, 3, 1, 2], c=[2, 4, 2, 6, 4],
        rows=[1, 5, 2, 4], y=[0.1, -0.2, 0.4, 0.3])
    backend = RKBRMI(@brm df begin
        mu ~ 1 + (1 + c | g)
        reads = read_rows(mu, rows)
        y ~ Normal(reads, 1)
    end)
    u = check_value_gradient(backend)
    nt = constrain(backend.model.layout, u)
    C = nt.b_g.z * (nt.b_g.tau .* nt.b_g.L)'
    gi = [2, 1, 3, 1, 2]
    mu = only(nt.pop_mu.beta_pop) .+ C[gi, 1] .+ (df.c .== 4) .* C[gi, 2] .+ (df.c .== 6) .* C[gi, 3]
    @test value_query(backend, :pointwise, u).y ≈
        logpdf.(Normal.(mu[df.rows], 1), df.y)
end
