# Run: julia --startup-file=no --project=test test/rk_prior_spellings.jl
#
# Prior spellings SBBRMI accepts build on RKBRMI with the same density:
# hyperparameter expressions, `truncated` over any scalar prior family with
# frozen bounds (`Inf` an open end), and declaration bounds on random-effect
# scale priors. Each case checks an independent oracle where one is written,
# the emitted Stan program through BridgeStan, ordinary Reverse gradients and
# the complete printed RK source.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

const SPELLING_DATA = (; g=[1, 1, 2, 2, 3, 3], h=[1, 2, 1, 2, 1, 2],
    x=[-0.4, 0.1, 0.3, 0.8, -0.2, 0.5], y=[0.2, -0.1, 0.4, 0.9, 0.1, 0.6])

interval_jacobian(value, lo, hi) = log(value - lo) + log(hi - value) - log(hi - lo)

# `law(p)` is the posterior log density at the constrained values `p`,
# including each coordinate's log-Jacobian.
function check_prior_spelling(brmi, label; law=nothing)
    before = deepcopy(SPELLING_DATA)
    backend, problem = consumer_problem(brmi)
    layout = backend.model.layout
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    stan = BRM.stan_instantiate(sb;
        path=joinpath(mktempdir(), "prior_spelling_$label.stan"))
    transport = brm_coordinate_transport(backend, sb,
        BridgeStan.param_unc_names(stan.model))
    n = layout.total
    for u in (zeros(n), [0.15 * sin(i) for i in 1:n], [-0.1 * cos(i) for i in 1:n])
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        stan_u = brm_rk_point_to_stan(transport, u)
        stan_gradient = similar(stan_u)
        stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
            stan_gradient; propto=false, jacobian=true)
        @test value ≈ stan_value + transport.logdensity_offset atol=2e-10 rtol=2e-10
        @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-8 rtol=2e-8
        law === nothing || check_consumer_point(problem, u, v -> law(constrain(layout, v)))
    end
    @test isequal(SPELLING_DATA, before)
    backend
end

# The main block and the submodel definitions it calls (random-effect blocks
# state their scale priors in their own definition).
emitted_source(backend) = let emitted = BRM._rk_emit_ast(backend.plan)
    join((sprint(Base.show_unquoted, x) for x in (emitted.defs..., emitted.main)), "\n")
end

@stestset "prior hyperparameters are Julia expressions" begin
    brmi = @brm SPELLING_DATA begin
        m ~ Normal(0.0, 1.0)
        scale ~ LogNormal(log(0.7), 0.3)
        loc ~ Normal(m + 0.5, exp(-1.0))
        y ~ Normal(loc, scale)
    end
    law(p) = logpdf(Normal(), p.m) +
        logpdf(LogNormal(log(0.7), 0.3), p.scale) + log(p.scale) +
        logpdf(Normal(p.m + 0.5, exp(-1.0)), p.loc) +
        sum(logpdf.(Normal(p.loc, p.scale), SPELLING_DATA.y))
    backend = check_prior_spelling(brmi, "expressions"; law)
    source = emitted_source(backend)
    # A constant expression is its number; a live one stays in the graph.
    @test occursin("scale ~ LogNormal($(log(0.7)), 0.3)", source)
    @test occursin("loc ~ Normal(m + 0.5, $(exp(-1.0)))", source)
end

@stestset "truncated priors over any scalar family" begin
    brmi = @brm SPELLING_DATA begin
        p ~ truncated(Beta(1.0, 100.0), 0.0, 0.2)
        d ~ truncated(LogNormal(-1.0, 0.7), 0.01, Inf)
        s ~ truncated(Normal(1.0, 1.0), 0.0, Inf)
        half ~ truncated(Normal(0.0, 1.0), 0.0, Inf)
        loc = p + d + half
        y ~ Normal(loc, s)
    end
    law(q) = logpdf(truncated(Beta(1.0, 100.0), 0.0, 0.2), q.p) +
        interval_jacobian(q.p, 0.0, 0.2) +
        logpdf(truncated(LogNormal(-1.0, 0.7), 0.01, Inf), q.d) + log(q.d - 0.01) +
        logpdf(truncated(Normal(1.0, 1.0), 0.0, Inf), q.s) + log(q.s) +
        logpdf(truncated(Normal(0.0, 1.0), 0.0, Inf), q.half) + log(q.half) +
        sum(logpdf.(Normal(q.p + q.d + q.half, q.s), SPELLING_DATA.y))
    backend = check_prior_spelling(brmi, "truncated"; law)
    source = emitted_source(backend)
    @test occursin("p ~ truncated(Beta(1.0, 100.0), 0.0, 0.2)", source)
    @test occursin("d ~ truncated(LogNormal(-1.0, 0.7), 0.01, Inf)", source)
    # A zero-location half keeps its HalfNormal spelling.
    @test occursin("half ~ HalfNormal(1.0)", source)
    @test_throws "lower < upper" RKBRMI(@brm SPELLING_DATA begin
        p ~ truncated(Beta(1.0, 2.0), 0.5, 0.2)
        y ~ Normal(p, 1.0)
    end)

    # Sampled bounds stay live, as SBBRMI's `truncated` keeps them
    # (test/truncated_priors.jl).
    live = @brm SPELLING_DATA begin
        location ~ Normal(0.0, 1.0)
        bounded ~ truncated(Cauchy(0.2, 0.7); lower=location, upper=location + 2)
        y ~ Normal(bounded, 1.0)
    end
    live_law(q) = logpdf(Normal(), q.location) +
        logpdf(truncated(Cauchy(0.2, 0.7), q.location, q.location + 2), q.bounded) +
        interval_jacobian(q.bounded, q.location, q.location + 2) +
        sum(logpdf.(Normal(q.bounded, 1.0), SPELLING_DATA.y))
    source = emitted_source(check_prior_spelling(live, "truncated-live"; law=live_law))
    @test occursin("bounded ~ truncated(Cauchy(0.2, 0.7), location, location + 2", source)
end

@stestset "random-effect scale priors take declaration bounds" begin
    brmi = @brm SPELLING_DATA begin
        sigma ~ Exponential(1.0)
        mu ~ 1 + (1 | a | g) + (1 | b | h)
        sd(:, a) ~ Normal(0.0, 1.0; lower=0.0)
        sd(:, b) ~ Exponential(0.5; upper=2.0)
        y ~ Normal(mu, sigma)
    end
    law(p) = begin
        ba, bb = p.b_a_g, p.b_b_h
        eta = p.mu_Intercept .+ (ba.z .* ba.tau[1])[SPELLING_DATA.g] .+
            (bb.z .* bb.tau[1])[SPELLING_DATA.h]
        logpdf(Exponential(), p.sigma) + log(p.sigma) + logpdf(Normal(), p.mu_Intercept) +
            logpdf(Normal(), only(ba.tau)) + log(only(ba.tau)) +
            logpdf(Exponential(0.5), only(bb.tau)) + interval_jacobian(only(bb.tau), 0.0, 2.0) +
            sum(logpdf.(Normal(), ba.z)) + sum(logpdf.(Normal(), bb.z)) +
            sum(logpdf.(Normal.(eta, p.sigma), SPELLING_DATA.y))
    end
    backend = check_prior_spelling(brmi, "scale-bounds"; law)
    source = emitted_source(backend)
    # Declaration bounds keep the family kernel (no truncation normalizer).
    @test occursin("tau[1:1] .~ restricted.(Normal(0.0, 1.0), 0.0, Inf)", source)
    @test occursin("tau[1:1] .~ restricted.(Exponential(0.5), 0.0, 2.0)", source)

    truncated_brmi = @brm SPELLING_DATA begin
        sigma ~ Exponential(1.0)
        mu ~ 1 + (1 | a | g)
        sd(:, a) ~ truncated(Normal(0.0, 1.0), 0.0, Inf)
        y ~ Normal(mu, sigma)
    end
    truncated_law(p) = begin
        ba = p.b_a_g
        eta = p.mu_Intercept .+ (ba.z .* ba.tau[1])[SPELLING_DATA.g]
        logpdf(Exponential(), p.sigma) + log(p.sigma) + logpdf(Normal(), p.mu_Intercept) +
            logpdf(truncated(Normal(), 0.0, Inf), only(ba.tau)) + log(only(ba.tau)) +
            sum(logpdf.(Normal(), ba.z)) +
            sum(logpdf.(Normal.(eta, p.sigma), SPELLING_DATA.y))
    end
    check_prior_spelling(truncated_brmi, "scale-truncated"; law=truncated_law)

    # Per-margin bounds of one correlated block: Stan is the reference.
    margins = @brm SPELLING_DATA begin
        sigma ~ Exponential(1.0)
        mu ~ 1 + x + (1 + x | a | g)
        sd(mu, a, Intercept) ~ Normal(0.0, 1.0; lower=0.0)
        sd(mu, a, x) ~ Normal(0.0, 0.5; upper=1.0)
        y ~ Normal(mu, sigma)
    end
    source = emitted_source(check_prior_spelling(margins, "scale-margins"))
    @test occursin("restricted(Normal(0.0, 0.5), 0.0, 1.0)", source)
end

const SPELLING_CELL_DATA = (; subject=[:b, :a, :c],
    row_subject=[:a, :b, :a, :c, :b], t=[0.2, 0.4, 0.8, 0.5, 1.1],
    y=[0.1, 0.6, 1.0, 0.3, 1.2])

const SPELLING_CELL = @brm begin
    rho ~ LogNormal(log(0.7), 0.3)
    p ~ truncated(Beta(1.0, 100.0), 0.0, 0.2)
    d ~ truncated(LogNormal(-1.0, 0.7), 0.01, Inf)
    base ~ 1 + (1 | lv | subject)
    amp ~ 1 + (1 | am | subject)
    sd(:, lv) ~ truncated(Normal(0.0, 1.0), 0.0, Inf)
    sd(:, am) ~ Normal(0.0, 1.0; lower=0.0)
    rows = groups(row_subject => subject)
    @plate for i in eachindex(base)
        m[i] = base[i] .+ amp[i] .* t[rows[i]] .+ p .+ d
        y[rows[i]] ~ Normal(m[i], rho)
    end
end

@stestset "the same priors inside a plate-cell model" begin
    brmi = SPELLING_CELL(SPELLING_CELL_DATA)
    before = deepcopy(SPELLING_CELL_DATA)
    backend, problem = consumer_problem(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    stan = BRM.stan_instantiate(sb; path=joinpath(mktempdir(), "prior_spelling_cell.stan"))
    transport = brm_coordinate_transport(backend, sb,
        BridgeStan.param_unc_names(stan.model))
    n = length(transport.rk_names)
    for u in (zeros(n), [0.15 * sin(i) for i in 1:n])
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        stan_u = brm_rk_point_to_stan(transport, u)
        stan_gradient = similar(stan_u)
        stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
            stan_gradient; propto=false, jacobian=true)
        @test value ≈ stan_value + transport.logdensity_offset atol=2e-10 rtol=2e-10
        @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-8 rtol=2e-8
    end
    @test isequal(SPELLING_CELL_DATA, before)
end
