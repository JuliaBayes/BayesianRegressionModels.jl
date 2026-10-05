# Keep the ordinary-constructor strict Stan-gradient receipt in its original
# fixture. Reuse its exact public scientific family and observed data here.
const law_fixture = joinpath(@__DIR__, "rk_observed_callable_law.jl")
include_string(@__MODULE__, first(split(read(law_fixture, String),
    "\n@stestset"; limit=2)), law_fixture)
import SpecialFunctions
import BayesianRegressionModels: _rk_observation_source!

function _rk_observation_source!(definitions, bindings, entry,
        ::typeof(PublicObservedCallableLaw.tail_normal))
    logerfc = Symbol(entry, :_logerfc)
    push!(bindings, logerfc => SpecialFunctions.logerfc)
    push!(definitions, :(ReactiveKernels.@kernel $entry(value, mu, sigma, lower, upper) = begin
        tail_logdensity = if value <= lower
            $logerfc(-(lower - mu) / (sigma * sqrt(2.0))) - log(2.0)
        elseif value >= upper
            $logerfc((upper - mu) / (sigma * sqrt(2.0))) - log(2.0)
        else
            -0.5 * log(2 * pi) - log(sigma) -
                0.5 * ((value - mu) / sigma) * ((value - mu) / sigma)
        end
        return tail_logdensity
    end))
    :done
end

function graph_law_recipes(graph; depth=0)
    [(; depth=depth + entry.depth, source=entry.recipe.source,
        outputs=entry.recipe.outputs) for entry in recipe_inventory(graph)]
end

@stestset "inclusive observed law composes normalized threshold branches inside the plate" begin
    data = (;y=[-2.0,-1.0,0.2,1.0,2.0],lo=-1.0,hi=1.0)
    saved = deepcopy(data)
    brmi = PublicObservedCallableLaw.build(data)
    backend, problem = consumer_problem(brmi)
    @test isempty(backend.plan.regression.predictors)
    @test coordinate_names(backend.model.layout) == [:mu,:sigma]
    oracle(u) = begin
        mu,sigma = u[1],exp(u[2])
        d = Normal(mu,sigma)
        logpdf(Normal(),mu) + logpdf(Exponential(),sigma) + u[2] +
            sum(y<=data.lo ? logcdf(d,data.lo) :
                y>=data.hi ? logccdf(d,data.hi) : logpdf(d,y) for y in data.y)
    end
    for u in ([0.,0.],[0.17,-0.21],[-0.1,0.2])
        check_consumer_point(problem,u,oracle)
    end
    recipes = graph_law_recipes(kernel_graph(backend.model.spec))
    branches = filter(record -> record.depth==1 && record.source isa Expr &&
        record.source.head===:if, recipes)
    @test length(branches)==1
    dump = sprint(Base.show_unquoted,only(branches).source)
    @test occursin("value <= lower",dump)
    @test occursin("value >= upper",dump)
    @test occursin("log(sigma)",dump)
    @test occursin("logerfc",dump)
    @test isequal(data,saved)
end
