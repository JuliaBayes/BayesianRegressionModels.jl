include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# HSGP options of decision 1fk9b29 on RKBRMI: the emitted RK graph evaluates
# an anchored or row-masked basis, an empty axis and a floor-bounded explicit
# length-scale prior with the same normalized density and gradient as the
# emitted Stan program. The Stan side is `test/hsgp_options.jl`.
const RK_HSGP_OPTION_DATA = (; t=[0.5, 1.0, 2.0, 3.5], carried=[true, true, false, true],
    p=Float64[], y=[0.2, -0.1, 0.4, 0.3])

function rk_hsgp_option_mapping(names, ids)
    mapping = Pair{Symbol,String}[]
    for name in names
        text = string(name)
        id = findfirst(i -> startswith(text, i * "."), ids)
        id === nothing && continue
        rest = text[length(ids[id]) + 2:end]
        push!(mapping, name => ids[id] * "_" * rest)
    end
    mapping
end

@stestset "HSGP options keep RK and Stan densities equal" begin
    data = RK_HSGP_OPTION_DATA
    before = deepcopy(data)
    cases = (
        ("anchor", ["hsgp_t"], @brm(begin
            location ~ 0 + hsgp(t; k=3, anchor=0.0)
            y ~ Normal(location, 1.0)
        end)),
        ("rows", ["hsgp_t"], @brm(begin
            location ~ 0 + hsgp(t; k=3, rows=carried)
            y ~ Normal(location, 1.0)
        end)),
        ("anchor-rows", ["hsgp_t"], @brm(begin
            location ~ 0 + hsgp(t; k=3, rows=carried, anchor=0.0)
            y ~ Normal(location, 1.0)
        end)),
        ("floor-prior", ["hsgp_t"], @brm(begin
            location ~ 0 + hsgp(t; k=3)
            length_scale(:, hsgp(t)) ~ Uniform(length_scale_floor, 5.0)
            y ~ Normal(location, 1.0)
        end)),
        ("floor-truncated", ["hsgp_t"], @brm(begin
            location ~ 0 + hsgp(t; k=3)
            length_scale(:, hsgp(t)) ~ truncated(LogNormal(0.0, 1.0); lower=length_scale_floor, upper=5.0)
            y ~ Normal(location, 1.0)
        end)),
        ("periodic-anchor-rows", ["hsgp_t"], @brm(begin
            location ~ 0 + hsgp(t; k=2, cov=:periodic, period=4.0, anchor=0.0, rows=carried)
            y ~ Normal(location, 1.0)
        end)),
        ("empty-domain", ["hsgp_t", "hsgp_p"], @brm(begin
            location ~ 0 + hsgp(t; k=3)
            placebo ~ 0 + hsgp(p; k=2, domain=(-1.5, 1.5))
            y ~ Normal(location, 1.0)
        end)),
    )
    for (label, ids, builder) in cases
        @testset "$label" begin
            brmi = builder(data)
            backend, problem = consumer_problem(brmi)
            names = coordinate_names(backend.model.layout)
            stan = consumer_stan(brmi, "hsgp-option-" * label)
            mapping = rk_hsgp_option_mapping(names, ids)
            @test length(mapping) == length(names)
            for u in (zeros(length(names)), fill(0.13, length(names)),
                      collect(range(-0.2, 0.3; length=length(names))))
                check_consumer_stan(problem, stan, mapping, backend, u)
            end
        end
    end
    @test isequal(data, before)
end
