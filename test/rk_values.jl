# Public synthetic values crossing formula row axes. No consumer model or
# private parameter packing is reproduced here.
# Run: julia --project=test test/rk_values.jl [testset substring ...]
using Test, BayesianRegressionModels, Distributions, Enzyme, LogDensityProblems
using ReactiveKernels, ReactiveKernelsPPL
using DifferentiationInterface: AutoEnzyme
import LogDensityProblems: dimension, logdensity, logdensity_and_gradient
using LinearAlgebra: cholesky, Symmetric, diagind
using Statistics: mean
using LogExpFunctions: logit, logistic
using BayesianRegressionModels: Cumulative, StoppingRatio, LogitLink, ProbitLink
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
    # Formula values and authored assignments are plain source assignments,
    # as in the StanBlocks emission: no generated one-line definition wraps
    # them (snag rk-emission-wrap-3b8da520).
    emitted = BRM._rk_emit_ast(backend.plan)
    kernels = Set(d.name for d in map(BRM._rk_source_definition, emitted.defs)
        if d.kind === :kernel)
    checked = Symbol[]
    for statement in emitted.main.args
        (Meta.isexpr(statement, :(=), 2) && statement.args[1] isa Symbol) || continue
        push!(checked, statement.args[1])
        value = statement.args[2]
        @test !(Meta.isexpr(value, :call) && first(value.args) in kernels)
    end
    @test issubset((:a, :b, :c, :reads), checked)
    u = check_value_gradient(backend)
    nt = constrain(backend.model.layout, u)
    # Independent population and correlation reconstruction, including the
    # sorted label map (input rows are c,a,b) and b's public inverse link.
    group = nt.b_p_subject
    C = group.z * (group.tau .* group.L)'
    a = nt.pop_a.beta_pop[1] .+ nt.pop_a.beta_pop[2] .* mixed_axes.x .+ C[[3, 1, 2], 1]
    b = exp.(nt.b_Intercept .+ C[[3, 1, 2], 2])
    c = nt.pop_c.beta_pop[1] .+ nt.pop_c.beta_pop[2] .* mixed_axes.w
    expected = [nt.multiplier * (a[mixed_axes.subject_row[j]] +
        b[mixed_axes.subject_row[j]] * mixed_axes.t[j]) +
        c[mixed_axes.secondary_row[j]] for j in eachindex(mixed_axes.y)]
    pointwise = value_query(backend, :pointwise, u)
    expected_ll = logpdf.(Normal.(expected, nt.sigma), mixed_axes.y)
    @test pointwise.y ≈ expected_ll
    @test value_query(backend, :likelihood, u) ≈ sum(expected_ll)
    normal_draws = [nt.pop_a.beta_pop; nt.b_Intercept; nt.pop_c.beta_pop; nt.multiplier]
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
            (logpdf(Normal(), nt.mu_Intercept) + logpdf(Normal(), nt.mu_ar_mu_t) +
                logpdf(Normal(), nt.ar_mu_t.phi_raw) + sum(logpdf.(Normal(), z)), 0.0)
        else
            z = getproperty(nt.dar_mu_t_level, :_ppl_scan_z_level)
            increment = 0.0
            for i in 2:5
                increment = nt.dar_mu_t_level.beta * increment + nt.dar_mu_t_level.sigma * z[i - 1]
                path[i] = path[i - 1] + increment
            end
            @test length(u) == 7
            (logpdf(Normal(), nt.mu_Intercept) +
                logpdf(truncated(Normal(0.5, 0.2), 0, 1), nt.dar_mu_t_level.beta) +
                logpdf(truncated(Normal(0, 0.2), 0, Inf), nt.dar_mu_t_level.sigma) +
                sum(logpdf.(Normal(), z)),
                log(nt.dar_mu_t_level.beta) + log1p(-nt.dar_mu_t_level.beta) + log(nt.dar_mu_t_level.sigma))
        end
        likelihood = sum(logpdf.(Normal.(nt.mu_Intercept .+ path[df.rows], 1), df.y))
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
        intercept = nt.mu_Intercept
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

# Caller data occupy both the elementary function name and the generated
# callable stem. The emitted inverse link must retain its exact callable.
inverse_link_collision_reader(p, q, r, rows, logistic, brm_value_function) =
    p[rows] .+ q[rows] .+ r[rows] .+ logistic .+ brm_value_function

@stestset "inverse-link source bindings preserve names and normalized law" begin
    base = (; x=[-0.4, 0.2, 0.9], rows=[3, 1, 2, 1], y=[0.1, 0.4, -0.2, 0.8])
    collision = merge(base, (; logistic=[.1, -.2, .3, -.1],
        brm_value_function=[-.2, .3, -.1, .2]))
    models = (
        "ordinary source namespace" => (base, @brm(base, begin
            logit(p) ~ 1 + x
            probit(q) ~ 1 + x
            cloglog(r) ~ 1 + x
            reads = blend_values(p, q, r, rows)
            y ~ Normal(reads, 1)
        end)),
        "caller name collisions" => (collision, @brm(collision, begin
            logit(p) ~ 1 + x
            probit(q) ~ 1 + x
            cloglog(r) ~ 1 + x
            reads = inverse_link_collision_reader(p, q, r, rows, logistic, brm_value_function)
            y ~ Normal(reads, 1)
        end)))
    for (label, (data, brmi)) in models
        @testset "$label" begin
            before = deepcopy(data)
            backend = check_rk_source_roundtrip(RKBRMI(brmi);
                ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            emitted = BRM._rk_emit_ast(backend.plan)
            binding = only(filter(pair -> last(pair) === BRM.logistic, emitted.bindings))
            @test first(binding) ∉ keys(data)
            ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
            mod = ext._rk_emit_module(emitted)
            @test getfield(mod, first(binding)) === BRM.logistic
            names = coordinate_names(backend.model.layout)
            @test sort(names) == sort([Symbol("pop_$(target).beta_pop.$j")
                for target in (:logit_p, :probit_q, :cloglog_r) for j in 1:2])
            index(target, j) = only(findall(==(Symbol("pop_$(target).beta_pop.$j")), names))
            linear(u, target) = u[index(target, 1)] .+ u[index(target, 2)] .* data.x
            function oracle(u)
                p = 1 ./ (1 .+ exp.(-linear(u, :logit_p)))
                q = cdf.(Normal(), linear(u, :probit_q))
                r = -expm1.(-exp.(linear(u, :cloglog_r)))
                expected = p[data.rows] .+ q[data.rows] .+ r[data.rows]
                haskey(data, :logistic) &&
                    (expected = expected .+ data.logistic .+ data.brm_value_function)
                sum(logpdf.(Normal(), u)) + sum(logpdf.(Normal.(expected, 1), data.y))
            end
            problem = rk_logdensity_problem(backend;
                ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            for u in (zeros(6), fill(.13, 6), collect(range(-.2, .3; length=6)))
                saved = copy(u)
                value, gradient = logdensity_and_gradient(problem, u)
                @test value ≈ oracle(u) atol=2e-11 rtol=2e-11
                step = 1e-5
                independent = map(eachindex(u)) do j
                    plus, minus = copy(u), copy(u)
                    plus[j] += step
                    minus[j] -= step
                    (oracle(plus) - oracle(minus)) / (2step)
                end
                @test gradient ≈ independent atol=2e-5 rtol=2e-5
                @test all(isfinite, gradient)
                @test isequal(u, saved)
            end
            @test isequal(data, before)
        end
    end
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
    mu = nt.mu_Intercept .+ C[gi, 1] .+ (df.c .== 4) .* C[gi, 2] .+ (df.c .== 6) .* C[gi, 3]
    @test value_query(backend, :pointwise, u).y ≈
        logpdf.(Normal.(mu[df.rows], 1), df.y)
end

# P(y = k) = logistic(c[k] - eta) - logistic(c[k - 1] - eta), with
# c[0] = -Inf and c[K] = Inf: Stan's ordered_logistic on the 1:K scale.
function ordered_logistic_reference(eta, c, k)
    upper = k > length(c) ? 1.0 : logistic(c[k] - eta)
    lower = k == 1 ? 0.0 : logistic(c[k - 1] - eta)
    log(upper - lower)
end

@stestset "legacy OrderedLogistic reads an authored location" begin
    # Gappy codes: K = maximum(y) = 4 with level 3 unobserved, as SBBRMI.
    df = (; x=[-1.2, -0.4, 0.1, 0.5, 0.9, 1.4, -0.8, 0.3],
        y=[1, 2, 2, 4, 4, 4, 1, 2])
    saved = deepcopy(df)
    brmi = @brm df begin
        slope ~ 0 + x
        bias ~ Normal(0, 1)
        loc = slope + bias
        y ~ OrderedLogistic(loc)
    end
    backend = RKBRMI(brmi)
    @test backend.plan isa BRM._RKValuePlan
    source = string(Base.remove_linenums!(deepcopy(BRM._rk_emit_ast(backend.plan).main)))
    @test occursin("y_cutpoints ~ Ordered(Normal(0.0, 1.0), 3)", source)
    @test occursin("y .~ OrderedLogistic.(loc, Ref(y_cutpoints))", source)
    u = check_value_gradient(backend)
    nt = constrain(backend.model.layout, u)
    c = nt.y_cutpoints
    @test length(c) == 3 && issorted(c)
    eta = only(nt.pop_slope.beta_pop) .* df.x .+ nt.bias
    expected_ll = ordered_logistic_reference.(eta, Ref(c), df.y)
    @test value_query(backend, :pointwise, u).y ≈ expected_ll
    @test value_query(backend, :likelihood, u) ≈ sum(expected_ll)
    prior = logpdf(Normal(), only(nt.pop_slope.beta_pop)) +
        logpdf(Normal(), nt.bias) + sum(logpdf.(Normal(), c))
    @test value_query(backend, :prior, u) ≈ prior
    # First cutpoint free, later cutpoints through log increments.
    @test value_query(backend, :sampler, u) ≈ prior + sum(expected_ll) + sum(log.(diff(c)))
    artifact = BRM.emit_rk_artifact(brmi; case_id="authored-ordered-logistic")
    rebuilt = build_kernel(BRM.rk_translate_artifact(artifact))
    @test coordinate_names(rebuilt.layout) == coordinate_names(backend.model.layout)
    @test isequal(df, saved)
end

ordinal_reference_cdf(::BRM.LogitLink, z) = logistic(z)
ordinal_reference_cdf(::BRM.ProbitLink, z) = cdf(Normal(), z)

# Cumulative: P(y <= k) = F(d * (c[k] - eta)). Stopping ratio: stage j stops
# with F(d * (c[j] - eta - effect[j])), and continues otherwise.
function ordinal_reference(::BRM.Cumulative, link, eta, c, d, effect, k)
    upper = k > length(c) ? 1.0 : ordinal_reference_cdf(link, d * (c[k] - eta))
    lower = k == 1 ? 0.0 : ordinal_reference_cdf(link, d * (c[k - 1] - eta))
    log(upper - lower)
end
function ordinal_reference(::BRM.StoppingRatio, link, eta, c, d, effect, k)
    q(j) = ordinal_reference_cdf(link, d * (c[j] - eta - effect[j]))
    sum(log1p(-q(j)) for j in 1:k-1; init=0.0) + (k > length(c) ? 0.0 : log(q(k)))
end

@stestset "ordinal value responses read authored locations" begin
    # Observed codes 1, 2, 4: fitted levels recode them to 1:3 (two thresholds).
    df = (; x=[-1.2, -0.4, 0.1, 0.5, 0.9, 1.4, -0.8, 0.3],
        z=[0.3, -0.2, 0.5, 0.1, -0.7, 0.9, 0.2, -0.4],
        y=[1, 2, 2, 4, 4, 4, 1, 2])
    coded = [1, 2, 2, 3, 3, 3, 1, 2]
    saved = deepcopy(df)
    cases = (
        ("cumulative logit", BRM.Cumulative(), BRM.LogitLink(), 1.0, false, @brm(df, begin
            slope ~ 0 + x
            bias ~ Normal(0, 1)
            loc = slope + bias
            y ~ Ordinal(Cumulative(), LogitLink(), loc)
        end)),
        ("cumulative probit discrimination", BRM.Cumulative(), BRM.ProbitLink(), 2.0, false, @brm(df, begin
            slope ~ 0 + x
            bias ~ Normal(0, 1)
            loc = slope + bias
            y ~ Ordinal(Cumulative(), ProbitLink(), loc; discrimination=2.0)
        end)),
        ("stopping ratio threshold effects", BRM.StoppingRatio(), BRM.LogitLink(), 1.0, true, @brm(df, begin
            slope ~ 0 + x
            bias ~ Normal(0, 1)
            loc = slope + bias
            y ~ Ordinal(StoppingRatio(), LogitLink(), loc; per_threshold=(z,))
        end)))
    for (label, structure, link, d, effects, brmi) in cases
        @testset "$label" begin
            backend = RKBRMI(brmi)
            @test backend.plan isa BRM._RKValuePlan
            u = check_value_gradient(backend)
            nt = constrain(backend.model.layout, u)
            c = nt.y_thresholds
            @test length(c) == 2
            beta = effects ? nt.y_threshold_beta : zeros(1, 2)
            @test size(beta) == (1, 2)
            eta = only(nt.pop_slope.beta_pop) .* df.x .+ nt.bias
            effect = effects ? df.z * beta : zeros(length(df.y), 2)
            expected_ll = [ordinal_reference(structure, link, eta[i], c, d, effect[i, :], coded[i])
                for i in eachindex(coded)]
            @test value_query(backend, :pointwise, u).y ≈ expected_ll
            prior = logpdf(Normal(), only(nt.pop_slope.beta_pop)) +
                logpdf(Normal(), nt.bias) + sum(logpdf.(Normal(), c)) +
                sum(logpdf.(Normal(), beta); init=0.0) * effects
            @test value_query(backend, :prior, u) ≈ prior
            jac = structure isa BRM.Cumulative ? sum(log.(diff(c))) : 0.0
            @test value_query(backend, :sampler, u) ≈ prior + sum(expected_ll) + jac
            artifact = BRM.emit_rk_artifact(brmi; case_id="authored-ordinal")
            rebuilt = build_kernel(BRM.rk_translate_artifact(artifact))
            @test coordinate_names(rebuilt.layout) == coordinate_names(backend.model.layout)
        end
    end
    @test isequal(df, saved)
end

@stestset "categorical logit value response codes fitted labels" begin
    df = (; x=[-1.2, -0.4, 0.1, 0.5, 0.9, 1.4, -0.8, 0.3],
        c=["b", "a", "c", "a", "b", "c", "c", "a"])
    saved = deepcopy(df)
    brmi = @brm df begin
        s2 ~ 0 + x
        b2 ~ Normal(0, 1)
        e2 = s2 + b2
        e3 = 0.5 * s2
        c ~ CategoricalLogit(e2, e3)
    end
    backend = RKBRMI(brmi)
    @test backend.plan isa BRM._RKValuePlan
    u = check_value_gradient(backend)
    nt = constrain(backend.model.layout, u)
    s2 = only(nt.pop_s2.beta_pop) .* df.x
    logits = [zeros(length(df.x)) s2 .+ nt.b2 0.5 .* s2]  # reference level "a"
    code = Dict("a" => 1, "b" => 2, "c" => 3)
    expected_ll = [logits[i, code[df.c[i]]] - log(sum(exp, logits[i, :]))
        for i in eachindex(df.c)]
    @test value_query(backend, :pointwise, u).c ≈ expected_ll
    artifact = BRM.emit_rk_artifact(brmi; case_id="authored-categorical-logit")
    rebuilt = build_kernel(BRM.rk_translate_artifact(artifact))
    @test coordinate_names(rebuilt.layout) == coordinate_names(backend.model.layout)
    @test isequal(df, saved)
end
