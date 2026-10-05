# Statistical components are self-contained RKPPL submodels: each allocates
# its own parameters and returns its post-processed value, as SBBRMI's
# components do. Single-expression algebra stays inline. The printed program
# must also reproduce an independent normalized density and its gradient.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Distributions
using Enzyme, LinearAlgebra
using DifferentiationInterface: AutoEnzyme
include(joinpath(@__DIR__, "testset_filter.jl"))
include(joinpath(@__DIR__, "rk_source_roundtrip.jl"))
const BRM = BayesianRegressionModels
const EXT = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)

strip_lines(x) = x isa Expr ? Expr(x.head,
    (strip_lines(a) for a in x.args if !(a isa LineNumberNode))...) : x

declares(x) = x isa Expr && ((x.head === :call && first(x.args) in (:~, :.~)) ||
    any(declares, x.args))

# Submodel definitions are bare function-shaped definitions; explicit
# `@kernel` graphs and ordinary functions are numerical code, not components.
submodels(emitted) = [d for d in emitted.defs if Meta.isexpr(d, :(=))]
submodel_name(d) = first(first(d.args).args)

const DATA = (; x=[-1.0, -0.3, 0.4, 1.0, 0.2, -0.6], w=[0.2, -0.5, 0.6, 0.1, -0.2, 0.4],
    g=[1, 2, 1, 2, 3, 3], h=[1, 1, 2, 2, 1, 2], s=[1, 1, 1, 1, 2, 2],
    membership=[3, 3, 1, 1, 2, 2],
    c=[1, 3, 2, 4, 2, 3], y=[0.2, -0.1, 0.3, -0.2, 0.5, 0.1],
    brm_population_effects=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

const CASES = (
    "population, correlated group, monotonic" => () -> @brm(DATA, begin
        mu ~ 1 + x + mo(c) + (1 + x | p | g)
        sd(:, p) ~ Exponential(0.7)
        cor(:, p) ~ LKJCholesky(2, 3.0)
        sigma ~ Exponential(1.0)
        y ~ Normal(mu, sigma)
    end),
    "two scalar groups share one definition" => () -> @brm(DATA, begin
        mu ~ 1 + (1 | g) + (1 | h)
        y ~ Normal(mu, 1.0)
    end),
    "mixed population families" => () -> @brm(DATA, begin
        mu ~ 1 + x + w
        effect(mu, x) ~ Laplace(0, 2)
        y ~ Normal(mu, 1.0)
    end),
    "component name collision" => () -> @brm(DATA, begin
        mu ~ 1 + x + brm_population_effects + (1 | g)
        y ~ Normal(mu, 1.0)
    end),
    "multi-membership" => () -> @brm(DATA, begin
        mu ~ 1 + (1 + x | mm(g, membership))
        y ~ Normal(mu, 1.0)
    end),
    "two HSGPs share one definition" => () -> @brm(DATA, begin
        mu ~ 1 + hsgp(x; k=3) + hsgp(w; k=3)
        y ~ Normal(mu, 1.0)
    end),
    "anisotropic and periodic HSGP" => () -> @brm(DATA, begin
        mu ~ 1 + hsgp(x, w; k=(2, 2), iso=false) + hsgp(w; k=2, cov=:periodic, period=2.0)
        y ~ Normal(mu, 1.0)
    end),
    "stratified group" => () -> @brm(DATA, begin
        mu ~ 1 + (1 + x | gr(g; by=s))
        y ~ Normal(mu, 1.0)
    end),
)

@stestset "every emitted submodel allocates its own parameters" begin
    for (label, build) in CASES
        @testset "$label" begin
            saved = deepcopy(DATA)
            backend = check_rk_source_roundtrip(RKBRMI(build()))
            emitted = BRM._rk_emit_ast(backend.plan)
            @test !isempty(submodels(emitted))
            for definition in submodels(emitted)
                @test declares(last(definition.args))
            end
            source = sprint(Base.show_unquoted, emitted.main)
            for retired in ("brm_correlated_random_coefficients", "brm_scaled_random_coefficients",
                    "brm_hsgp_summand", "brm_monotonic_contrast", "brm_logdensity_value")
                @test !occursin(retired, source)
            end
            @test isequal(DATA, saved)
        end
    end
end

@stestset "component definitions are shared and renamed on collision" begin
    emitted(i) = BRM._rk_emit_ast(BRM._brm_rk_plan(last(CASES[i])()))
    names(i) = Set(submodel_name.(submodels(emitted(i))))
    @test names(2) == Set([:brm_group_effects, :brm_population_effects])
    groups = filter(s -> Meta.isexpr(s, :call) && s.args[1] === :~ &&
        Meta.isexpr(s.args[3], :call) && s.args[3].args[1] === :brm_group_effects,
        emitted(2).main.args)
    @test length(groups) == 2
    @test names(3) == Set([:brm_mixed_population_effects])
    collision = names(4)
    @test :brm_population_effects ∉ collision
    @test any(n -> startswith(string(n), "brm_population_effects"), collision)
    hsgp = emitted(6)
    @test names(6) == Set([:brm_hsgp_effect, :brm_population_effects])
    spectral = [d for d in hsgp.defs if Meta.isexpr(d, :macrocall) &&
        occursin("brm_hsgp_spectral_graph", sprint(show, d))]
    @test length(spectral) == 1
end

@stestset "main block calls components like SBBRMI" begin
    emitted = BRM._rk_emit_ast(BRM._brm_rk_plan(last(CASES[1])()))
    main = strip_lines(emitted.main)
    @test main.args[1] == :(b_p_g ~ brm_correlated_group_effects(g, 2, 3.0))
    @test :(mo_c ~ brm_monotonic_effect(c_idx, [1.0, 1.0, 1.0], 0.0, 1.0)) in main.args
    @test :(X_mu = hcat(ones(length(c)), x)) in main.args
    @test :(pop_mu ~ brm_population_effects(X_mu, 2, 0.0, 1.0)) in main.args
    definitions = Dict(submodel_name(d) => strip_lines(d) for d in submodels(emitted))
    @test definitions[:brm_correlated_group_effects] == strip_lines(:(
        brm_correlated_group_effects(g, K, eta) = begin
            tau[1:K] .~ Exponential.(0.7)
            L ~ LKJCholesky(K, eta)
            z[levels(g), 1:K] .~ Normal.(0, 1)
            return z * transpose(tau .* L)
        end))
    @test definitions[:brm_monotonic_effect] == strip_lines(:(
        brm_monotonic_effect(c, alpha, loc, scale) = begin
            simplex_incr ~ Dirichlet(alpha)
            beta ~ Normal(loc, scale)
            return beta .* (cumsum(vcat(0.0, simplex_incr)))[c]
        end))
    @test definitions[:brm_population_effects] == strip_lines(:(
        brm_population_effects(X, ncoef, loc, scale) = begin
            beta_pop[1:ncoef] .~ Normal.(loc, scale)
            return X * beta_pop
        end))
end

@stestset "component program matches an independent normalized density" begin
    model = last(CASES[1])()
    backend = RKBRMI(model)
    bound = EXT._rk_translated_plan(backend.plan)
    layout = backend.model.layout
    names = coordinate_names(layout)
    @test Symbol("pop_mu.beta_pop.1") in names
    @test Symbol("b_p_g.tau.2") in names
    @test Symbol("mo_c.simplex_incr.2") in names
    @test :sigma in names
    prior = prepare_query(backend.model, bound, :prior)
    likelihood = prepare_query(backend.model, bound, :likelihood)
    levels = sort(unique(DATA.g))
    group = [findfirst(==(v), levels) for v in DATA.g]
    for u in (fill(0.13, layout.total), collect(range(-0.4, 0.3; length=layout.total)))
        saved = copy(u)
        p = constrain(layout, u)
        b = p.b_p_g
        effects = b.z * transpose(b.tau .* b.L)
        contrast = cumsum(vcat(0.0, p.mo_c.simplex_incr))[DATA.c]
        mu = p.pop_mu.beta_pop[1] .+ p.pop_mu.beta_pop[2] .* DATA.x .+
            p.mo_c.beta .* contrast .+ effects[group, 1] .+ effects[group, 2] .* DATA.x
        expected_prior = sum(logpdf.(Normal(0, 1), p.pop_mu.beta_pop)) +
            sum(logpdf.(Exponential(0.7), b.tau)) +
            logpdf(LKJCholesky(2, 3.0), Cholesky(b.L, 'L', 0)) +
            sum(logpdf.(Normal(0, 1), b.z)) +
            logpdf(Dirichlet(ones(3)), p.mo_c.simplex_incr) +
            logpdf(Normal(0, 1), p.mo_c.beta) + logpdf(Exponential(1.0), p.sigma)
        @test Base.invokelatest(prior, u) ≈ expected_prior atol=1e-10 rtol=1e-12
        @test Base.invokelatest(likelihood, u) ≈
            sum(logpdf.(Normal.(mu, p.sigma), DATA.y)) atol=1e-10 rtol=1e-12
        query = prepare_sampler(backend.model, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
        gradient = similar(u)
        value, _ = sampler_value_and_gradient!(query, gradient, u)
        step = 1e-6
        finite = map(eachindex(u)) do j
            plus, minus = copy(u), copy(u)
            plus[j] += step; minus[j] -= step
            (query(plus) - query(minus)) / 2step
        end
        @test gradient ≈ finite atol=1e-6 rtol=1e-6
        @test isequal(u, saved)
    end
end
