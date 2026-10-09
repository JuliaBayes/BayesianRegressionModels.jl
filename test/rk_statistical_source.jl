# Statistical blocks are self-contained submodels, as in StanBlocks: each
# `lhs ~ block(...)` allocates its parameters with their priors and returns
# its post-processed value (USER guiding principles, RKPPLBench review
# `1nhjhdl`). A block must mean exactly its hand-inlined flat program: the
# same priors, density, gradient and coordinates, with `lhs.local` for the
# inlined `lhs_local`.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Distributions
using Enzyme, LogDensityProblems, LinearAlgebra
using DifferentiationInterface: AutoEnzyme
include(joinpath(@__DIR__, "testset_filter.jl"))
include(joinpath(@__DIR__, "rk_source_roundtrip.jl"))
const BRM = BayesianRegressionModels
const EXT = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)

block_definitions(emitted) = Dict(BRM._rk_source_definition(d).name => d
    for d in emitted.defs if BRM._rk_source_definition(d).kind === :rkppl)

declared!(names, _) = names
declared!(names, x::Symbol) = push!(names, x)
function declared!(names, x::Expr)
    x.head === :ref && return declared!(names, first(x.args))
    x.head === :tuple && foreach(arg -> declared!(names, arg), x.args)
    names
end

# Every name a block body binds: assignment and sampling targets, loop indices.
function block_locals!(names, x)
    x isa Expr || return names
    if x.head === :(=)
        declared!(names, first(x.args))
    elseif x.head === :call && length(x.args) == 3 && first(x.args) in (:~, :.~)
        declared!(names, x.args[2])
    end
    foreach(arg -> block_locals!(names, arg), x.args)
    names
end

substitute(x, replacements) = x isa Symbol ? get(replacements, x, x) :
    x isa Expr ? Expr(x.head, map(arg -> substitute(arg, replacements), x.args)...) : x

# The flat program a block call stands for: arguments substituted by the call's
# expressions, locals spelled `lhs_local`, and `lhs` bound to the return value.
function inline_block(statement, definitions)
    Meta.isexpr(statement, :call, 3) && first(statement.args) === :~ || return Any[statement]
    lhs, call = statement.args[2], statement.args[3]
    Meta.isexpr(call, :call) && haskey(definitions, first(call.args)) || return Any[statement]
    signature, body = definitions[first(call.args)].args
    replacements = Dict{Symbol,Any}(zip(signature.args[2:end], call.args[2:end]))
    for name in block_locals!(Set{Symbol}(), body)
        replacements[name] = Symbol(lhs, "_", name)
    end
    map(filter(x -> !(x isa LineNumberNode), body.args)) do x
        Meta.isexpr(x, :return) ? Expr(:(=), lhs, substitute(only(x.args), replacements)) :
            substitute(x, replacements)
    end
end

function inline_blocks(emitted)
    definitions = block_definitions(emitted)
    main = Expr(:block, Iterators.flatten(inline_block(x, definitions)
        for x in emitted.main.args)...)
    retained = filter(d -> BRM._rk_source_definition(d).kind !== :rkppl, emitted.defs)
    BRM._RKEmittedProgram(retained, main, emitted.bindings)
end

# `lhs.local...` is the inlined `lhs_local...`; a block scan packs its
# innovations as `lhs._ppl_scan_z_<carried>` for `_ppl_scan_z_lhs_<carried>`.
function inlined_name(name, names)
    name in names && return name
    scan = match(r"^(\w+)\._ppl_scan_z_(.+)$", name)
    scan === nothing || return "_ppl_scan_z_$(scan[1])_$(scan[2])"
    replace(name, '.' => '_'; count=1)
end

function check_statistical_source(brmi, data; repeated=false, collision=false)
    before = deepcopy(data)
    backend = check_rk_source_roundtrip(RKBRMI(brmi))
    emitted = BRM._rk_emit_ast(backend.plan)
    definitions = block_definitions(emitted)
    @test !isempty(definitions)
    source = sprint(Base.show_unquoted, emitted.main)
    @test occursin("~ brm_", source)
    # A block allocates: no submodel is a single-line helper.
    for (name, definition) in definitions
        body = last(definition.args)
        @test any(x -> Meta.isexpr(x, :call) && length(x.args) == 3 &&
            first(x.args) in (:~, :.~), body.args) || any(x -> Meta.isexpr(x, :macrocall), body.args)
    end
    repeated && @test count(n -> startswith(string(n), "brm_group_effects"), keys(definitions)) == 1
    if collision
        @test occursin("~ brm_correlated_group_effects_(", source)
        @test haskey(backend.plan.columns, :brm_correlated_group_effects)
    end
    # The reference is the ordinary printed flat program. It goes through
    # public lowering and kernel creation; no lowered plan, names, priors or
    # AD activity are patched.
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    reference = inline_blocks(emitted)
    original = ext._rk_translated_plan(backend.plan)
    inlined = ext._rk_translate_from_emitted(backend.plan, reference)
    baseline = Base.invokelatest(build_kernel, inlined)
    names = String.(coordinate_names(backend.model.layout))
    flat = String.(coordinate_names(baseline.layout))
    mapped = [inlined_name(name, flat) for name in names]
    @test sort(mapped) == sort(flat)
    index = Dict(name => i for (i, name) in enumerate(flat))
    perm = [index[name] for name in mapped]
    n = backend.model.layout.total
    for u in (zeros(n), fill(.13, n), n <= 1 ? fill(-.2, n) :
            collect(range(-.2, .3; length=n)))
        saved = copy(u)
        v = zeros(n)
        v[perm] = u
        for preset in (:sampler, :prior, :likelihood)
            actual = Base.invokelatest(prepare_query(backend.model, original, preset), u)
            expected = Base.invokelatest(prepare_query(baseline, inlined, preset), v)
            @test actual ≈ expected rtol=1e-12 atol=1e-12
        end
        qa = prepare_sampler(backend.model, original, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
        qb = prepare_sampler(baseline, inlined, v; backend=AutoEnzyme(; mode=Enzyme.Reverse))
        ga, gb = similar(u), similar(v)
        va, _ = sampler_value_and_gradient!(qa, ga, u)
        vb, _ = sampler_value_and_gradient!(qb, gb, v)
        @test va ≈ vb rtol=1e-12 atol=1e-12
        @test ga ≈ gb[perm] rtol=1e-10 atol=1e-10
        @test all(isfinite, ga)
        @test isequal(u, saved)
    end
    @test isequal(data, before)
    backend
end

@testset "BRM statistical blocks are self-contained submodels" begin
    data=(;x=[-1.,-.3,.4,1.],w=[.2,-.5,.6,.1],g=[1,2,1,2],h=[1,1,2,2],
        membership=[3,3,1,1],w1=[.4,.5,.6,.7],w2=[.6,.5,.4,.3],
        c=[1,2,3,2],t=[1.,2.,3.,4.],
        brm_correlated_group_effects=[.1,.2,.3,.4],y=[.2,-.1,.3,-.2])
    smooth_data = (; x=collect(range(-1., 1.; length=12)), y=fill(.2, 12))
    cases=(
        ("one margin", @brm(data,begin mu ~ 0 + (1|g); y ~ Normal(mu,1.) end)),
        ("reused definition", @brm(data,begin mu ~ 1 + (1|g) + (1|h); y ~ Normal(mu,1.) end)),
        ("correlated", @brm(data,begin mu ~ 1 + x + (1+x|block|g)
            sd(:,block) ~ Exponential(.7); cor(:,block) ~ LKJCholesky(2,3.)
            y ~ Normal(mu,1.) end)),
        ("callable name collision", @brm(data,begin
            mu ~ 1 + brm_correlated_group_effects + (1+x|g)
            y ~ Normal(mu,1.) end)),
        ("multi-membership", @brm(data,begin
            mu ~ 1 + (1|mm(g,membership)); y ~ Normal(mu,1.) end)),
        ("correlated weighted multi-membership", @brm(data,begin
            mu ~ 1 + (1+x|mm(g,membership;weights=(w1,w2)))
            y ~ Normal(mu,1.) end)),
        ("HSGP", @brm(data,begin mu ~ 1 + hsgp(x;k=3); y ~ Normal(mu,1.) end)),
        ("anisotropic HSGP", @brm(data,begin mu ~ 1 + hsgp(x,w;k=(2,2),iso=false)
            y ~ Normal(mu,1.) end)),
        ("periodic HSGP", @brm(data,begin mu ~ 1 + hsgp(x;k=3,cov=:periodic,period=2.)
            y ~ Normal(mu,1.) end)),
        ("grouped HSGP", @brm(data,begin mu ~ 1 + hsgp(x;k=3,by=g); y ~ Normal(mu,1.) end)),
        ("smooth", @brm(smooth_data,begin mu ~ 1 + s(x); y ~ Normal(mu,1.) end)),
        ("monotonic", @brm(data,begin mu ~ 1 + mo(c); y ~ Normal(mu,1.) end)),
        ("GP", @brm(data,begin mu ~ 1 + gp(x); y ~ Normal(mu,1.) end)),
        ("AR(1)", @brm(data,begin mu ~ 1 + ar(t; p=1); y ~ Normal(mu,1.) end)),
        ("differenced AR(1)", @brm(data,begin mu ~ 1 + dar(t); y ~ Normal(mu,1.) end)))
    for (label,model) in cases
        @stestset "$label" begin
            println("checking statistical block: ", label)
            flush(stdout)
            check_statistical_source(model,label=="smooth" ? smooth_data : data;
                repeated=label=="reused definition",
                collision=label=="callable name collision")
        end
    end
end

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
    @test names(2) == Set([:brm_group_effects])
    groups = filter(s -> Meta.isexpr(s, :call) && s.args[1] === :~ &&
        Meta.isexpr(s.args[3], :call) && s.args[3].args[1] === :brm_group_effects,
        emitted(2).main.args)
    @test length(groups) == 2
    @test names(3) == Set([:brm_mixed_population_effects])
    collision = names(4)
    @test :brm_population_effects ∉ collision
    @test any(n -> startswith(string(n), "brm_population_effects"), collision)
    hsgp = emitted(6)
    @test names(6) == Set([:brm_hsgp_effect])
    spectral = [d for d in hsgp.defs if Meta.isexpr(d, :macrocall) &&
        occursin("brm_hsgp_spectral_graph", sprint(show, d))]
    @test length(spectral) == 1
end

@stestset "main block calls components like SBBRMI" begin
    emitted = BRM._rk_emit_ast(BRM._brm_rk_plan(last(CASES[1])()))
    main = strip_lines(emitted.main)
    # Executable preparation precedes allocating calls in the main graph.
    @test :(b_p_g ~ brm_correlated_group_effects(g, 2, 3.0)) in main.args
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
