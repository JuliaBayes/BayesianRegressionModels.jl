# Independent public multi-axis fixture: wildcard defaults belong only to
# predictors that own the addressed coefficient.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
StanBlocks.@deffun begin
    consumer_subject_operation_reads(v::vector[ns], f::vector[nop], ends::int[ns])::vector[ns] = begin
        result = rep_vector(0., ns)
        first = 1
        for s in 1:ns
            result[s] = v[s]
            for op in first:ends[s]
                result[s] += f[op]
            end
            first = ends[s] + 1
        end
        result
    end
    consumer_gather(values::vector[n], rows::int[m])::vector[m] = values[rows]
end
function consumer_subject_operation_reads(v, f, ends)
    result = copy(v); first = 1
    for s in eachindex(ends)
        for op in first:ends[s]
            result[s] += f[op]
        end
        first = ends[s] + 1
    end
    result
end
consumer_gather(values, rows) = values[rows]
const WILDCARD_DATA = (; x=[-.4,.1,.7], op_x=[-.2,.3,.5,.8,1.1,1.4],
    ends=[2,3,6], rows=[1,3,2,1,2], y=[-.5,.4,-.1,.3,.7], lo=fill(-.5,5))
const WILDCARD_BODY = """
log(v) ~ 1 + x
f ~ 0 + op_x
effect(v, Intercept) ~ Normal(0., 1.)
effect(f, op_x) ~ Normal(0., .4)
reads = consumer_subject_operation_reads(v,f,ends)
effect(:, x) ~ Normal(0.,.3)
mu = consumer_gather(reads,rows)
y ~ Normal(mu,1.)
"""
wildcard_model(body=WILDCARD_BODY) = Core.eval(@__MODULE__, BRM._brm(body; df=WILDCARD_DATA))

function wildcard_public_replay(backend)
    emitted = BRM._rk_emit_ast(backend.plan)
    mod = Module(gensym(:WildcardReplay))
    Core.eval(mod, :(using ReactiveKernelsPPL))
    for (name, callable) in emitted.bindings
        Core.eval(mod, Expr(:const, Expr(:(=), name, QuoteNode(callable))))
    end
    for definition in emitted.defs
        parsed = Meta.parse(sprint(Base.show_unquoted, definition))
        Core.eval(mod, Expr(:macrocall, Symbol("@rkppl"), LineNumberNode(0), parsed))
    end
    parsed = Meta.parse(sprint(Base.show_unquoted, emitted.main))
    data = backend.plan.columns
    plan = bind_data(lower_rkppl(parsed, data; mod, conditioned=(:y,)), data)
    build_kernel(plan), plan
end

function wildcard_reference(u, names; censored=false)
    q = Dict(name => u[j] for (j, name) in enumerate(names))
    alpha, beta, gamma = q[:v_Intercept], q[:v_x], q[:f_op_x]
    subject = exp.(alpha .+ beta .* WILDCARD_DATA.x)
    operations = [sum(WILDCARD_DATA.op_x[1:2]), WILDCARD_DATA.op_x[3],
        sum(WILDCARD_DATA.op_x[4:6])]
    mu = (subject .+ gamma .* operations)[WILDCARD_DATA.rows]
    prior = logpdf(Normal(0,1), alpha) + logpdf(Normal(0,.3), beta) +
        logpdf(Normal(0,.4), gamma)
    likelihood = sum(eachindex(mu)) do i
        d = Normal(mu[i],1)
        censored && WILDCARD_DATA.y[i] == WILDCARD_DATA.lo[i] ?
            logcdf(d,WILDCARD_DATA.lo[i]) : logpdf(d,WILDCARD_DATA.y[i])
    end
    # The exp link transforms a predictor, not sampled coefficients; no
    # additional coefficient-coordinate Jacobian belongs in this density.
    prior + likelihood
end

@stestset "wildcard ownership: unchanged linked multi-axis model" begin
    before = deepcopy(WILDCARD_DATA)
    for evidence in (false, true)
        body = evidence ? replace(WILDCARD_BODY, "y ~ Normal(mu,1.)" =>
            "y ~ censored(Normal(mu,1.);lower=lo)") : WILDCARD_BODY
        brmi = wildcard_model(body)
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        @test Set(names) == Set([:v_Intercept, :v_x, :f_op_x])
        @test BRM.popcoefnames(brmi, :v) == [:Intercept, :x]
        @test BRM.popcoefnames(brmi, :f) == [:op_x]
        explicit = wildcard_model(replace(body, "effect(:, x)" => "effect(v, x)"))
        @test BRM.stan_code(SBBRMI(brmi; mod=@__MODULE__, total_groups=())) ==
            BRM.stan_code(SBBRMI(explicit; mod=@__MODULE__, total_groups=()))
        stan = consumer_stan(brmi, "wildcard-ownership-$evidence")
        mapping = [:v_Intercept => "pop_log_v_beta_pop.1",
            :v_x => "pop_log_v_beta_pop.2", :f_op_x => "pop_f_beta_pop.1"]
        @test Set(BridgeStan.param_unc_names(stan.model)) == Set(last.(mapping))
        rebuilt, plan = wildcard_public_replay(backend)
        @test coordinate_names(rebuilt.layout) == names
        original = Base.get_extension(BRM,
            :BayesianRegressionModelsReactiveKernelsExt)._rk_translated_plan(backend.plan)
        for u in (zeros(3), fill(.13,3), [-.2,.05,.3])
            oracle(q) = wildcard_reference(q, names; censored=evidence)
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, mapping, backend, u)
            for preset in (:sampler, :prior, :likelihood)
                a = Base.invokelatest(prepare_query(backend.model, original, preset), u)
                b = Base.invokelatest(prepare_query(rebuilt, plan, preset), u)
                @test isequal(a,b)
            end
        end
    end
    @test isequal(before, WILDCARD_DATA)
end

@stestset "wildcard owners, precedence and unmatched targets" begin
    # Two owners and a third nonowner on the same axis. Scalar priors are
    # independent declarations, not population coefficient owners.
    data = (; x=[-.2,.3,.7], z=[.5,.4,.2], y=[.1,.2,.3])
    base = """
    a ~ 0 + x
    b ~ 0 + x
    c ~ 0 + z
    extra ~ Normal(0,1)
    effect(:, x) ~ Normal(0,.3)
    mu = a + b + c + extra
    y ~ Normal(mu,1)
    """
    make(body) = Core.eval(@__MODULE__, BRM._brm(body; df=data))
    for specific_first in (false, true)
        clauses = specific_first ?
            "effect(a, x) ~ Normal(.2,.7)\neffect(:, x) ~ Normal(0,.3)" :
            "effect(:, x) ~ Normal(0,.3)\neffect(a, x) ~ Normal(.2,.7)"
        brmi = make(replace(base,"effect(:, x) ~ Normal(0,.3)" => clauses))
        backend = RKBRMI(brmi)
        regression = backend.plan.regression
        actual = Dict((p.predictor,p.addressee) => p.args for p in regression.population_priors)
        @test actual[(:a,:x)] == (.2,.7)
        @test actual[(:b,:x)] == (0.,.3)
        @test actual[(:c,:z)] == (0.,1.)
        @test !haskey(actual,(:c,:x))
    end
    unmatched = make(replace(base,"effect(:, x)" => "effect(:, typo)"))
    @test_throws "matches no population coefficient" RKBRMI(unmatched)
    @test_throws "matches no population coefficient" SBBRMI(unmatched; mod=@__MODULE__,total_groups=())
    wrong_owner = make(replace(base,"effect(:, x)" => "effect(c, x)"))
    @test_throws "is not a population coefficient of `c`" RKBRMI(wrong_owner)
    tie = make(replace(base,"effect(:, x) ~ Normal(0,.3)" =>
        "effect(:, x) ~ Normal(0,.3)\neffect(a, :) ~ Normal(0,.7)"))
    @test_throws "equally specific" RKBRMI(tie)
    @test_throws "equally specific" SBBRMI(tie; mod=@__MODULE__,total_groups=())
end
