# Statistical blocks are self-contained submodels, as in StanBlocks: each
# `lhs ~ block(...)` allocates its parameters with their priors and returns
# its post-processed value (USER guiding principles, RKPPLBench review
# `1nhjhdl`). A block must mean exactly its hand-inlined flat program: the
# same priors, density, gradient and coordinates, with `lhs.local` for the
# inlined `lhs_local`.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
include(joinpath(@__DIR__, "rk_source_roundtrip.jl"))
include(joinpath(@__DIR__, "testset_filter.jl"))
const BRM = BayesianRegressionModels

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
    repeated && @test length(definitions) == 1
    if collision
        @test occursin("~ brm_correlated_draws_(", source)
        @test haskey(backend.plan.columns, :brm_correlated_draws)
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
        brm_correlated_draws=[.1,.2,.3,.4],y=[.2,-.1,.3,-.2])
    smooth_data = (; x=collect(range(-1., 1.; length=12)), y=fill(.2, 12))
    cases=(
        ("one margin", @brm(data,begin mu ~ 0 + (1|g); y ~ Normal(mu,1.) end)),
        ("reused definition", @brm(data,begin mu ~ 1 + (1|g) + (1|h); y ~ Normal(mu,1.) end)),
        ("correlated", @brm(data,begin mu ~ 1 + x + (1+x|block|g)
            sd(:,block) ~ Exponential(.7); cor(:,block) ~ LKJCholesky(2,3.)
            y ~ Normal(mu,1.) end)),
        ("callable name collision", @brm(data,begin
            mu ~ 1 + brm_correlated_draws + (1+x|g)
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
