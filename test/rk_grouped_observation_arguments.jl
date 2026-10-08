# Reuse the public scalar/vector law and source hook; its original fixture
# retains the in-cell constructor and strict Stan acceptance independently.
const family_fixture = joinpath(@__DIR__, "rk_kernel_observation_families.jl")
include_string(@__MODULE__, first(split(read(family_fixture, String),
    "\n@stestset"; limit=2)), family_fixture)

@eval PublicKernelObservationFamilies begin
# The integer control supplies the corresponding caller-owned Stan methods;
# the original law's vector signatures only accept real-valued reference rows.
StanBlocks.@deffun begin
    @lhs @lpxf relative_normal_lpdf(y::vector[n], location::vector[n],
            reference::int[n], scale::real)::real = begin
        result::real = 0.0
        for i in 1:n
            result += normal_lpdf(y[i], location[i] - reference[i], scale)
        end
        return result
    end
    relative_normal_lpdfs(y::vector[n], location::vector[n],
            reference::int[n], scale::real)::vector[n] = begin
        result::vector[n]
        for i in 1:n
            result[i] = normal_lpdf(y[i], location[i] - reference[i], scale)
        end
        return result
    end
    relative_normal_rng(vector[n], location::vector[n],
            reference::int[n], scale::real)::vector[n] = begin
        result::vector[n]
        for i in 1:n
            result[i] = normal_rng(location[i] - reference[i], scale)
        end
        return result
    end
end

function grouped_outside(data; arithmetic=false, ordinary=false)
    if ordinary
        return @brm data begin
            alpha ~ 1 + (1 | p | subject)
            effect(alpha, Intercept) ~ Normal(0, 0.7)
            sd(:, p) ~ Exponential(0.9)
            sigma ~ Exponential(0.9)
            loc ~ kernel(x, alpha) do xs, a
                xs * a
            end
            y ~ Normal(loc - reference, sigma)
        end
    elseif arithmetic
        return @brm data begin
            alpha ~ 1 + (1 | p | subject)
            effect(alpha, Intercept) ~ Normal(0, 0.7)
            sd(:, p) ~ Exponential(0.9)
            sigma ~ Exponential(0.9)
            loc ~ kernel(x, alpha) do xs, a
                xs * a
            end
            y ~ relative_normal(loc, reference * sigma, sigma)
        end
    end
    @brm data begin
        alpha ~ 1 + (1 | p | subject)
        effect(alpha, Intercept) ~ Normal(0, 0.7)
        sd(:, p) ~ Exponential(0.9)
        sigma ~ Exponential(0.9)
        loc ~ kernel(x, alpha) do xs, a
            xs * a
        end
        y ~ relative_normal(loc, reference, sigma)
    end
end

function grouped_inside(data)
    @brm data begin
        alpha ~ 1 + (1 | p | subject)
        effect(alpha, Intercept) ~ Normal(0, 0.7)
        sd(:, p) ~ Exponential(0.9)
        sigma ~ Exponential(0.9)
        loc ~ kernel(x, reference, y, alpha) do xs, refs, ys, a
            mu = xs * a
            ys ~ relative_normal(mu, refs, sigma)
            mu
        end
    end
end

function grouped_join(data)
    @brm data begin
        alpha ~ 1 + (1 | p | subject)
        effect(alpha, Intercept) ~ Normal(0, 0.7)
        sd(:, p) ~ Exponential(0.9)
        sigma ~ Exponential(0.9)
        loc ~ kernel(x, alpha) do xs, a
            xs * a
        end
        ragged(y, event_subject) ~ relative_normal(loc, reference, sigma)
        z ~ Normal(reference, 1)
    end
end
end

# Inspect the actual built graph, including retained child recipes.
function argument_graph_sources(graph)
    [entry.recipe.source for entry in recipe_inventory(graph)]
end

argument_callee_is(callee, name) = callee === name ||
    (callee isa GlobalRef && callee.name === name) ||
    (Meta.isexpr(callee, :.) && last(callee.args) == QuoteNode(name))

nested = (;subject=["b", "empty", "a"],
    x=[[0.0, 0.5, 1.0], Float64[], [0.0, 0.3, 0.8, 1.4]],
    y=[[0.1, 0.2, 0.4], Float64[], [0.0, 0.1, 0.3, 0.4]],
    reference=[[0.01, 0.02, 0.03], Float64[], [0.01, 0.02, 0.03, 0.04]])
permutation = [2, 3, 5, 1, 4, 6, 7]
joined = (;subject=nested.subject, x=nested.x,
    event_subject=["a", "b", "b", "a", "b", "a", "a"],
    y=[0.0, 0.1, 0.2, 0.1, 0.4, 0.3, 0.4],
    reference=[0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04],
    z=[0.2, -0.1, 0.3, -0.2, 0.1, 0.0, 0.5])
integer = merge(nested, (;reference=[[1, 2, 3], Int[], [4, 5, 6, 7]]))
singleton = merge(nested, (;reference=[[0.01], [0.02], [0.03]]))

function check_grouped_arguments(label, data)
    saved = deepcopy(data)
    brmi = label === :joined ? PublicKernelObservationFamilies.grouped_join(data) :
        PublicKernelObservationFamilies.grouped_outside(data;
            arithmetic=label === :arithmetic, ordinary=label === :ordinary)
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    ia = findfirst(==(Symbol("alpha_Intercept")), names)
    it = findfirst(n -> occursin(".tau.", string(n)), names)
    iz = findall(n -> occursin(".z.", string(n)), names)
    is = findfirst(==(:sigma), names)
    @test length(names) == 6
    @test length(iz) == 3
    @test backend.plan.columns[:reference] == data.reference
    observed = label === :joined ? data.y[permutation] : reduce(vcat, data.y)
    reference = label === :joined ? data.reference[permutation] :
        reduce(vcat, [ones(length(xs)) .* refs for (xs, refs) in zip(data.x, data.reference)])
    @test backend.plan.columns[:y] == observed
    levels = sort(unique(data.subject))
    subject_order = [findfirst(==(s), levels) for s in data.subject]
    oracle(u) = begin
        tau, sigma = exp(u[it]), exp(u[is])
        alpha = u[ia] .+ tau .* u[iz][subject_order]
        means = reduce(vcat, [data.x[i] .* alpha[i] for i in eachindex(data.x)])
        refs = label === :arithmetic ? reference .* sigma : reference
        prior = logpdf(Normal(0, 0.7), u[ia]) + sum(logpdf.(Normal(), u[iz])) +
            logpdf(Exponential(0.9), tau) + u[it] +
            logpdf(Exponential(0.9), sigma) + u[is]
        likelihood = sum(logpdf.(Normal.(means .- refs, sigma), observed))
        extra = label === :joined ? sum(logpdf.(Normal.(data.reference, 1), data.z)) : 0.0
        prior + likelihood + extra
    end
    stan = consumer_stan(brmi, "grouped-argument-$label"; mod=PublicKernelObservationFamilies)
    mapping = [names[ia] => "pop_alpha_beta_pop.1";
        names[it] => "b_p_subject_tau.1"; :sigma => "sigma";
        [names[iz[i]] => "b_p_subject_z_flat.$i" for i in 1:3]]
    inside = label === :nested ? last(consumer_problem(
        PublicKernelObservationFamilies.grouped_inside(data))) : nothing
    for u in (zeros(6), collect(range(-0.2, 0.3; length=6)), fill(-0.1, 6))
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, mapping, backend, u)
        if inside !== nothing
            inside_value, inside_gradient = LogDensityProblems.logdensity_and_gradient(inside, u)
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            @test value ≈ inside_value atol=2e-11 rtol=2e-11
            @test gradient ≈ inside_gradient atol=2e-11 rtol=2e-11
        end
    end
    main = sprint(Base.show_unquoted, BRM._rk_emit_ast(backend.plan).main)
    definitions = sprint(show, BRM._rk_emit_ast(backend.plan).defs)
    graph_sources = argument_graph_sources(kernel_graph(backend.model.spec))
    sources = sprint(show, graph_sources)
    if label === :joined
        # The join's per-subject row partition is a bound port read by the
        # emitted argument reader, never a literal in source.
        partition = [[2, 3, 5], Int[], [1, 4, 6, 7]]
        port = only(key for (key, value) in backend.plan.columns if isequal(value, partition))
        @test occursin(string(port), main)
        @test !occursin("[2, 3, 5, 1, 4, 6, 7]", definitions * main)
        @test :(raw[brm_flatten_cells(groups)]) in graph_sources
    elseif label === :singleton
        # Each subject's response cell length comes from the bound response.
        @test !occursin("[3, 0, 4]", definitions * main)
        @test occursin("ones(length(rows))", sources)
    else
        @test occursin("y_observation_argument_1(loc, reference, sigma)", main)
        @test any(source -> Meta.isexpr(source, :call) &&
            argument_callee_is(first(source.args), :brm_flatten_cells) &&
            :raw in source.args, graph_sources)
    end
    if label !== :ordinary
        @test occursin("relative_residual", sources)
    end
    artifact = BRM.emit_rk_artifact(brmi; case_id="grouped-argument-$label")
    rebuilt = build_kernel(BRM.rk_translate_artifact(artifact))
    @test coordinate_names(rebuilt.layout) == names
    @test isequal(data, saved)
end

@stestset "grouped outside arguments: nested" begin
    check_grouped_arguments(:nested, nested)
end

@stestset "grouped outside arguments: joined" begin
    check_grouped_arguments(:joined, joined)
end

@stestset "grouped outside arguments: arithmetic" begin
    check_grouped_arguments(:arithmetic, nested)
end

@stestset "grouped outside arguments: ordinary" begin
    check_grouped_arguments(:ordinary, nested)
end

@stestset "grouped outside arguments: integer" begin
    check_grouped_arguments(:integer, integer)
end

@stestset "grouped outside arguments: singleton" begin
    check_grouped_arguments(:singleton, singleton)
end

@stestset "grouped argument partitions must match the observed partitions" begin
    data = (;subject=["b", "a"], x=[[0.0, 0.5, 1.0], [0.0, 1.0]],
        y=[[0.1, 0.2, 0.4], [0.0, 0.4]], reference=[[0.01, 0.02], [0.01, 0.02, 0.03]])
    saved = deepcopy(data)
    # Refused: five values on differently partitioned subject rows are not
    # aligned observation arguments (brm-use, Exact observation constructors).
    error = try
        RKBRMI(PublicKernelObservationFamilies.grouped_outside(data))
        nothing
    catch ex
        ex
    end
    @test error isa ErrorException
    @test occursin("has group lengths [2, 3]; expected [3, 2]", sprint(showerror, error))
    @test isequal(data, saved)
end
