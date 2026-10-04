# Readable statistical submodels must preserve the existing explicit sampled
# coordinates and the executable meaning of their former inline source.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
include(joinpath(@__DIR__, "rk_source_roundtrip.jl"))
const BRM = BayesianRegressionModels

function inline_statistical_source(emitted)
    definitions = Dict(first(first(d.args).args) => d for d in emitted.defs
        if Meta.isexpr(d, :(=), 2))
    substitute(node, replacements) = node isa Symbol ? get(replacements, node, node) :
        node isa Expr ? Expr(node.head, map(x -> substitute(x, replacements), node.args)...) : node
    function inline(statement)
        Meta.isexpr(statement, :call, 3) && first(statement.args) === :~ || return statement
        call = last(statement.args)
        Meta.isexpr(call, :call) && haskey(definitions, first(call.args)) || return statement
        definition = definitions[first(call.args)]
        parameters = first(definition.args).args[2:end]
        expressions = filter(x -> !(x isa LineNumberNode), last(definition.args).args)
        length(expressions) == 1 || error("reference expects a single statistical value expression")
        replacements = Dict(zip(parameters, call.args[2:end]))
        Expr(:(=), statement.args[2], substitute(only(expressions), replacements))
    end
    BRM._RKEmittedProgram(Expr[], Expr(:block, map(inline, emitted.main.args)...), emitted.bindings)
end

function check_statistical_source(brmi, data; repeated=false, collision=false)
    before = deepcopy(data)
    backend = check_rk_source_roundtrip(RKBRMI(brmi))
    emitted = BRM._rk_emit_ast(backend.plan)
    @test !isempty(emitted.defs)
    source = sprint(Base.show_unquoted, emitted.main)
    @test occursin("~ brm_", source)
    repeated && @test length(emitted.defs) == 1
    if collision
        @test occursin("~ brm_correlated_random_coefficients_(", source)
        @test haskey(backend.plan.columns, :brm_correlated_random_coefficients)
    end
    # This reference is an ordinary printed program with the exact previous
    # value expressions. It goes through public lowering and kernel creation;
    # no lowered plan, names, priors or AD activity are patched.
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    reference = inline_statistical_source(emitted)
    original = ext._rk_translated_plan(backend.plan)
    inlined = ext._rk_translate_from_emitted(backend.plan, reference)
    baseline = Base.invokelatest(build_kernel, inlined)
    @test coordinate_names(backend.model.layout) == coordinate_names(baseline.layout)
    n = backend.model.layout.total
    for u in (zeros(n), fill(.13,n), collect(range(-.2,.3;length=n)))
        saved = copy(u)
        @test isequal(constrain(backend.model.layout,u), constrain(baseline.layout,u))
        for preset in (:sampler,:prior,:likelihood)
            actual = Base.invokelatest(prepare_query(backend.model,original,preset),u)
            expected = Base.invokelatest(prepare_query(baseline,inlined,preset),u)
            @test isequal(actual,expected)
        end
        qa = prepare_sampler(backend.model,original,u;backend=AutoEnzyme(;mode=Enzyme.Reverse))
        qb = prepare_sampler(baseline,inlined,u;backend=AutoEnzyme(;mode=Enzyme.Reverse))
        ga,gb = similar(u),similar(u)
        va,_ = sampler_value_and_gradient!(qa,ga,u)
        vb,_ = sampler_value_and_gradient!(qb,gb,u)
        @test isequal(va,vb)
        @test isequal(ga,gb)
        @test all(isfinite,ga)
        @test isequal(u,saved)
    end
    @test isequal(data,before)
    backend
end

@testset "BRM statistical source preserves explicit coordinates" begin
    data=(;x=[-1.,-.3,.4,1.],w=[.2,-.5,.6,.1],g=[1,2,1,2],h=[1,1,2,2],
        membership=[3,3,1,1],w1=[.4,.5,.6,.7],w2=[.6,.5,.4,.3],
        brm_correlated_random_coefficients=[.1,.2,.3,.4],y=[.2,-.1,.3,-.2])
    cases=(
        ("one margin", @brm(data,begin mu ~ 0 + (1|g); y ~ Normal(mu,1.) end)),
        ("reused definition", @brm(data,begin mu ~ 1 + (1|g) + (1|h); y ~ Normal(mu,1.) end)),
        ("correlated", @brm(data,begin mu ~ 1 + x + (1+x|block|g)
            sd(:,block) ~ Exponential(.7); cor(:,block) ~ LKJCholesky(2,3.)
            y ~ Normal(mu,1.) end)),
        ("callable name collision", @brm(data,begin
            mu ~ 1 + brm_correlated_random_coefficients + (1+x|g)
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
        ("grouped HSGP", @brm(data,begin mu ~ 1 + hsgp(x;k=3,by=g); y ~ Normal(mu,1.) end)))
    for (label,model) in cases
        @testset "$label" begin
            check_statistical_source(model,data;repeated=label=="reused definition",
                collision=label=="callable name collision")
        end
    end
end
