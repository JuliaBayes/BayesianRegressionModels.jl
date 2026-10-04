include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module PublicKernelObservationFamilies
using BayesianRegressionModels, StanBlocks, Distributions
import BayesianRegressionModels: _rk_observation_source!

# A public vector law with two aligned row arguments and one live shared
# scale. Its native scalar graph uses the exact same normalized law.
StanBlocks.@deffun begin
    @lhs @lpxf relative_normal_lpdf(y::vector[n], location::vector[n],
            reference::vector[n], scale::real)::real = begin
        result::real = 0.0
        for i in 1:n
            result += normal_lpdf(y[i], location[i] - reference[i], scale)
        end
        return result
    end
    relative_normal_lpdfs(y::vector[n], location::vector[n],
            reference::vector[n], scale::real)::vector[n] = begin
        result::vector[n]
        for i in 1:n
            result[i] = normal_lpdf(y[i], location[i] - reference[i], scale)
        end
        return result
    end
    relative_normal_rng(vector[n], location::vector[n],
            reference::vector[n], scale::real)::vector[n] = begin
        result::vector[n]
        for i in 1:n
            result[i] = normal_rng(location[i] - reference[i], scale)
        end
        return result
    end
end

function _rk_observation_source!(definitions, bindings, entry,
        ::typeof(relative_normal))
    push!(definitions, :(ReactiveKernels.@kernel $entry(value, location, reference, scale) = begin
        relative_residual = (value - location + reference) / scale
        relative_log_scale = log(scale)
        relative_logdensity = -0.5 * log(2 * pi) - relative_log_scale -
            0.5 * relative_residual * relative_residual
        return relative_logdensity
    end))
    :done
end

function build(data)
    @brm data begin
        a ~ Normal(0, 0.7)
        sigma ~ Exponential(0.9)
        nu ~ Exponential(0.2)
        locations ~ kernel(x, y) do xs, ys
            loc = xs * a
            ys ~ student_t(nu, loc, sigma)
            loc
        end
    end
end

function build_relative(data)
    @brm data begin
        a ~ Normal(0, 0.7)
        sigma ~ Exponential(0.9)
        locations ~ kernel(x, reference, y) do xs, refs, ys
            loc = xs * a
            retained_reference = refs * (1 + a)
            ys ~ relative_normal(loc, retained_reference, sigma)
            loc
        end
    end
end

function build_relative_direct(data)
    @brm data begin
        a ~ Normal(0, 0.7)
        sigma ~ Exponential(0.9)
        loc = x * a
        retained_reference = reference * (1 + a)
        y ~ relative_normal(loc, retained_reference, sigma)
    end
end
end

@stestset "kernel cell-law retains exact Student-t constructor and observation axes" begin
    data = (;x=[[0.2,0.5],Float64[],[0.7]],
        y=[[0.1,0.4],Float64[],[-0.2]])
    saved = deepcopy(data)
    brmi = PublicKernelObservationFamilies.build(data)
    backend, problem = consumer_problem(brmi)
    @test coordinate_names(backend.model.layout) == [:a,:sigma,:nu]
    @test backend.plan.columns[:y] == [0.1,0.4,-0.2]
    oracle(u) = begin
        a,sigma,nu = u[1],exp(u[2]),exp(u[3])
        logpdf(Normal(0,0.7),a) +
            logpdf(Exponential(0.9),sigma) + u[2] +
            logpdf(Exponential(0.2),nu) + u[3] +
            sum(logpdf(LocationScale(x*a,sigma,TDist(nu)),y)
                for (xs,ys) in zip(data.x,data.y) for (x,y) in zip(xs,ys))
    end
    stan = consumer_stan(brmi,"kernel-student-t-law";mod=PublicKernelObservationFamilies)
    for u in ([0.0,0.0,0.0],[0.17,-0.21,0.3],[-0.1,0.2,-0.2])
        check_consumer_point(problem,u,oracle)
        check_consumer_stan(problem,stan,[:a=>"a",:sigma=>"sigma",:nu=>"nu"],backend,u)
    end
    emitted = BRM._rk_emit_ast(backend.plan)
    @test occursin("StudentT",sprint(Base.show_unquoted,emitted.main))
    @test isequal(data,saved)
end

# Frozen518 diagnostic fields/classifier, with public entry/body accessors.
# kernel_expr is a pre-build replay and can retain KernelSpec reader calls;
# the built numerical graph is the place to verify composed scalar recipes.
function observation_graph_recipes(graph;depth=0)
    records = NamedTuple[]
    for recipe in graph.recipes
        push!(records,(;depth,outputs=sprint(show,recipe.outputs),source=recipe.source))
        if recipe.op isa ReactiveKernels._AuthoredPlateOp
            append!(records,observation_graph_recipes(plate_body(recipe);depth=depth+1))
        elseif recipe.op isa ReactiveKernels._AuthoredScanOp
            append!(records,observation_graph_recipes(scan_body(recipe);depth=depth+1))
        end
    end
    records
end

@stestset "caller observation graph retains vector arguments and visible scalar law" begin
  for route in (:kernel, :direct)
    data = (;x=[[0.2,0.5],Float64[],[0.7]],
        reference=[[0.1,0.4],Float64[],[-0.2]],
        y=[[0.1,0.4],Float64[],[-0.2]])
    if route === :direct
        data = map(values -> reduce(vcat,values),data)
    end
    saved = deepcopy(data)
    brmi = route === :kernel ? PublicKernelObservationFamilies.build_relative(data) :
        PublicKernelObservationFamilies.build_relative_direct(data)
    backend, problem = consumer_problem(brmi)
    @test coordinate_names(backend.model.layout) == [:a,:sigma]
    @test backend.plan.columns[:y] == [0.1,0.4,-0.2]
    oracle(u) = begin
        a,sigma = u[1],exp(u[2])
        rows = route === :kernel ? zip(data.x,data.reference,data.y) :
            [(data.x,data.reference,data.y)]
        logpdf(Normal(0,0.7),a) + logpdf(Exponential(0.9),sigma) + u[2] +
            sum(logpdf(Normal(x*a-r*(1+a),sigma),y)
                for (xs,rs,ys) in rows
                for (x,r,y) in zip(xs,rs,ys))
    end
    stan = consumer_stan(brmi,"relative-normal-graph-$route";
        mod=PublicKernelObservationFamilies)
    for u in ([0.0,0.0],[0.17,-0.21],[-0.1,0.2])
        check_consumer_point(problem,u,oracle)
        check_consumer_stan(problem,stan,[:a=>"a",:sigma=>"sigma"],backend,u)
    end
    bound = BRM.rk_translate_artifact(BRM.emit_rk_artifact(brmi;
        case_id="relative-normal-graph-$route"))
    expression = kernel_expr(bound, assign_layout(bound))
    dump = sprint(Base.show_unquoted, expression)
    records = observation_graph_recipes(kernel_graph(build_kernel(bound).spec))
    scalar = filter(record->record.depth==1,records)
    println("OBSERVATION_BUILT_SCALAR_RECIPES=",scalar)
    @test any(record->occursin("relative_residual",record.outputs) &&
        isequal(record.source,:(((value-location)+reference)/scale)),scalar)
    @test any(record->occursin("relative_log_scale",record.outputs) &&
        isequal(record.source,:(log(scale))),scalar)
    @test any(record->occursin("relative_logdensity",record.outputs) &&
        isequal(record.source,:((-0.5*log(2*pi)-relative_log_scale)-
            0.5*relative_residual*relative_residual)),scalar)
    @test !occursin("y_scalar_logdensity(",dump)
    @test isequal(data,saved)
  end
end
