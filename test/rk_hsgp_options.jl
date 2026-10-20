include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# HSGP options of decision 1fk9b29 on RKBRMI: the emitted RK graph evaluates
# an anchored or row-masked basis, an empty axis and a floor-bounded explicit
# length-scale prior with the same normalized density and gradient as the
# emitted Stan program. The Stan side is `test/hsgp_options.jl`.
const RK_HSGP_OPTION_DATA = (; t=[0.5, 1.0, 2.0, 3.5], carried=[true, true, false, true],
    p=Float64[], y=[0.2, -0.1, 0.4, 0.3])

# RK coordinate names onto the emitted Stan parameter names: `<id>.<rest>` is
# `<id>_<rest>`, with an anisotropic `rho_<j>` as `rho.<j>`; a `by=` term's
# scales live under `<id>_by_<group>_` and its weights `z.<g>.<b>` in the
# flattened `zflat_hsgpw_<axes>_<group>` vector.
function rk_hsgp_option_mapping(names, ids; by=nothing)
    weights = [match(r"\.z\.(\d+)\.(\d+)$", string(name)) for name in names]
    B = maximum((parse(Int, w[2]) for w in weights if w !== nothing); init=0)
    mapping = Pair{Symbol,String}[]
    for (name, w) in zip(names, weights)
        text = string(name)
        id = findfirst(i -> startswith(text, i * "."), ids)
        id === nothing && continue
        stem, rest = ids[id], text[length(ids[id]) + 2:end]
        anisotropic = match(r"^rho_(\d+)$", rest)
        target = if w !== nothing
            g, b = parse(Int, w[1]), parse(Int, w[2])
            "zflat_hsgpw_$(chopprefix(stem, "hsgp_"))_$by.$((g - 1) * B + b)"
        elseif by !== nothing
            "$(stem)_by_$(by)_" * (rest == "rho" ? "rho_iso" : rest)
        elseif anisotropic !== nothing
            "$(stem)_rho.$(anisotropic[1])"
        else
            "$(stem)_$rest"
        end
        push!(mapping, name => target)
    end
    mapping
end

@stestset "HSGP options keep RK and Stan densities equal" begin
    data = RK_HSGP_OPTION_DATA
    before = deepcopy(data)
    # `@brm(...)` expands to an assignment, so each builder is bound before
    # it joins the case tuple.
    anchor_model = @brm begin
        location ~ 0 + hsgp(t; k=3, anchor=0.0)
        y ~ Normal(location, 1.0)
    end
    rows_model = @brm begin
        location ~ 0 + hsgp(t; k=3, rows=carried)
        y ~ Normal(location, 1.0)
    end
    anchor_rows_model = @brm begin
        location ~ 0 + hsgp(t; k=3, rows=carried, anchor=0.0)
        y ~ Normal(location, 1.0)
    end
    floor_prior_model = @brm begin
        location ~ 0 + hsgp(t; k=3)
        length_scale(:, hsgp(t)) ~ Uniform(length_scale_floor, 5.0)
        y ~ Normal(location, 1.0)
    end
    floor_truncated_model = @brm begin
        location ~ 0 + hsgp(t; k=3)
        length_scale(:, hsgp(t)) ~ truncated(LogNormal(0.0, 1.0); lower=length_scale_floor, upper=5.0)
        y ~ Normal(location, 1.0)
    end
    floor_anisotropic_model = @brm begin
        location ~ 0 + hsgp(t, u; k=2, iso=false)
        length_scale(:, hsgp(t, u)) ~ Uniform(length_scale_floor, 20.0)
        y ~ Normal(location, 1.0)
    end
    floor_grouped_model = @brm begin
        location ~ 0 + hsgp(t; k=3, by=g)
        length_scale(:, hsgp(t)) ~ Uniform(length_scale_floor, 5.0)
        y ~ Normal(location, 1.0)
    end
    periodic_anchor_rows_model = @brm begin
        location ~ 0 + hsgp(t; k=2, cov=:periodic, period=4.0, anchor=0.0, rows=carried)
        y ~ Normal(location, 1.0)
    end
    empty_domain_model = @brm begin
        location ~ 0 + hsgp(t; k=3)
        drift ~ 0 + hsgp(p; k=2, domain=(-1.5, 1.5))
        y ~ Normal(location, 1.0)
    end
    with_u = merge(data, (; u=data.t .^ 2))
    with_g = merge(data, (; g=["a", "b", "a", "b"]))
    # Rows outside the mask may be missing.
    with_missing = merge(data, (; t=[0.5, 1.0, missing, 3.5]))
    cases = (
        ("anchor", ["hsgp_t"], anchor_model, data, nothing),
        ("rows", ["hsgp_t"], rows_model, data, nothing),
        ("rows-missing", ["hsgp_t"], rows_model, with_missing, nothing),
        ("anchor-rows", ["hsgp_t"], anchor_rows_model, data, nothing),
        ("floor-prior", ["hsgp_t"], floor_prior_model, data, nothing),
        ("floor-truncated", ["hsgp_t"], floor_truncated_model, data, nothing),
        ("floor-anisotropic", ["hsgp_t_u"], floor_anisotropic_model, with_u, nothing),
        ("floor-grouped", ["hsgp_t"], floor_grouped_model, with_g, "g"),
        ("periodic-anchor-rows", ["hsgp_t"], periodic_anchor_rows_model, data, nothing),
        ("empty-domain", ["hsgp_t", "hsgp_p"], empty_domain_model, data, nothing),
    )
    for (label, ids, builder, frame, by) in cases
        @testset "$label" begin
            brmi = builder(frame)
            backend, problem = consumer_problem(brmi)
            names = coordinate_names(backend.model.layout)
            stan = consumer_stan(brmi, "hsgp-option-" * label)
            mapping = rk_hsgp_option_mapping(names, ids; by)
            @test length(mapping) == length(names)
            for u in (zeros(length(names)), fill(0.13, length(names)),
                      collect(range(-0.2, 0.3; length=length(names))))
                check_consumer_stan(problem, stan, mapping, backend, u)
            end
        end
    end
    @test isequal(data, before)
end

@stestset "HSGP floor at or above the explicit upper bound is refused" begin
    # `t` spans [0.5, 3.5]: a k=3 basis has L = 1.5 * 1.75 = 2.625 and a
    # floor of about 2.54.
    # refused: the declared length-scale support is empty, so the density
    # could never be evaluated (decision 0ddmtf8)
    for prior in (:(Uniform(length_scale_floor, 1.0)),
                  :(truncated(LogNormal(0.0, 1.0); lower=length_scale_floor, upper=1.0)))
        builder = eval(:(@brm begin
            location ~ 0 + hsgp(t; k=3)
            length_scale(:, hsgp(t)) ~ $prior
            y ~ Normal(location, 1.0)
        end))
        @test_throws "(k=3, c=1.5, L=2.625) is not below the upper bound 1.0" RKBRMI(
            builder(RK_HSGP_OPTION_DATA))
    end
end
