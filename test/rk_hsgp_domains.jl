include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# Independent tensor eigenbasis and normalized density. Formula boundaries
# stay fixed when rows change; this oracle never calls BRM basis helpers.
function fixed_hsgp_oracle(axes, K, domains)
    widths = [(hi-lo)/2 for (lo,hi) in domains]
    centers = [(lo+hi)/2 for (lo,hi) in domains]
    modes = collect(CartesianIndices(K))[:]
    omega2 = [(I[j]*pi/(2widths[j]))^2 for I in modes, j in eachindex(K)]
    PHI = [prod(sin(sqrt(omega2[b,j])*(axes[j][i]-centers[j]+widths[j])) /
        sqrt(widths[j]) for j in eachindex(K)) for i in eachindex(first(axes)), b in eachindex(modes)]
    floors = [K[j] == 1 ? 0.0 : (4widths[j]/pi)*sqrt(log(100)/(K[j]^2-1))
        for j in eachindex(K)]
    PHI, omega2, floors
end

# Public graph/body accessors, with recipe classification explicitly frozen
# to the tested RK revision (a stable public classifier remains separate).
function hsgp_plate_depths(graph; depth=0)
    result = Int[]
    for recipe in graph.recipes
        if recipe.op isa ReactiveKernels._AuthoredPlateOp
            push!(result, depth)
            append!(result, hsgp_plate_depths(plate_body(recipe); depth=depth+1))
        elseif recipe.op isa ReactiveKernels._AuthoredScanOp
            append!(result, hsgp_plate_depths(scan_body(recipe); depth=depth+1))
        end
    end
    result
end

@stestset "fixed HSGP domain graphs retain basis, priors and normalized Stan density" begin
    data = (; x=[-0.7,0.0,0.6], w=[0.4,-0.2,0.7], y=[0.2,-0.1,0.4])
    cases = (
        ("original", (3,), ((-2.0,2.0),), true, @brm(data, begin
            location ~ 0 + hsgp(x;k=3,domain=(-2.0,2.0))
            y ~ Normal(location,1.0)
        end)),
        ("tensor-isotropic", (2,3), ((-2.0,2.0),(-1.5,2.5)), true, @brm(data, begin
            location ~ 0 + hsgp(x,w;k=(2,3),domain=((-2.0,2.0),(-1.5,2.5)))
            y ~ Normal(location,1.0)
        end)),
        ("tensor-anisotropic", (2,3), ((-2.0,2.0),(-1.5,2.5)), false, @brm(data, begin
            location ~ 0 + hsgp(x,w;k=(2,3),iso=false,domain=((-2.0,2.0),(-1.5,2.5)))
            y ~ Normal(location,1.0)
        end)))
    original_data = deepcopy(data)
    for (label,K,domains,iso,brmi) in cases
        backend, problem = consumer_problem(brmi)
        emitted = BRM._rk_emit_ast(backend.plan)
        text = join(sprint(Base.show_unquoted,d) for d in emitted.defs)
        @test !occursin("brm_hsgp_basis(",text)
        @test !occursin("brm_hsgp_sqrt_spd(",text)
        inventory = hsgp_plate_depths(kernel_graph(backend.model.spec))
        @test !isempty(inventory)
        @test any(>(0),inventory)
        PHI, omega2, floors = fixed_hsgp_oracle((data.x,data.w)[1:length(K)],K,domains)
        ext = Base.get_extension(BRM,:BayesianRegressionModelsReactiveKernelsExt)
        mod = ext._rk_emit_module(emitted)
        definition = only(filter(d -> BRM._rk_source_definition(d).kind === :kernel &&
            endswith(string(BRM._rk_source_definition(d).name),"_basis_graph"),emitted.defs))
        owner = getfield(mod,BRM._rk_source_definition(definition).name)
        have = Tuple(Symbol(:axis_,j) for j in eachindex(K))
        for (method, expected) in ((:basis_matrix,PHI),(:squared_frequencies,omega2),
                (:length_scale_floor,iso ? maximum(floors) : floors))
            reader = Base.invokelatest(prepare,getproperty(owner,method);have,want=method)
            @test Base.invokelatest(reader,(data.x,data.w)[1:length(K)]...) ≈ expected atol=2e-15 rtol=2e-15
            if method === :basis_matrix
                new_axes = ([0.9,-0.8,0.1,0.3],[-0.7,0.2,0.4,-0.3])[1:length(K)]
                new_expected = first(fixed_hsgp_oracle(new_axes,K,domains))
                @test Base.invokelatest(reader,new_axes...) ≈ new_expected atol=2e-15 rtol=2e-15
                @test Base.invokelatest(reader,map(x->view(x,:),new_axes)...) ≈ new_expected atol=2e-15 rtol=2e-15
            end
        end
        names = coordinate_names(backend.model.layout)
        id = length(K)==1 ? "hsgp_x" : "hsgp_x_w"
        position(name) = only(findall(==(Symbol(name)),names))
        rpos = iso ? [position(id*"_rho")] : [position(id*"_rho$j") for j in eachindex(K)]
        spos = position(id*"_sigma")
        zpos = [position(id*"_z.$b") for b in 1:prod(K)]
        function oracle(u)
            lower = iso ? [maximum(floors)] : floors
            rhos = lower .+ exp.(u[rpos])
            sigma = exp(u[spos]); z = u[zpos]
            spectral = [sigma*prod(sqrt((iso ? only(rhos) : rhos[j])*sqrt(2pi)) for j in eachindex(K)) *
                exp(-0.25sum((iso ? only(rhos) : rhos[j])^2*omega2[b,j] for j in eachindex(K))) for b in 1:prod(K)]
            locations = PHI*(spectral.*z)
            sum(logpdf.(LogNormal(0,1),rhos)) + logpdf(LogNormal(0,1),sigma) +
                sum(logpdf.(Normal(),z)) + sum(logpdf.(Normal.(locations,1),data.y)) +
                sum(u[rpos]) + u[spos]
        end
        sb = SBBRMI(brmi)
        stan = BRM.stan_instantiate(sb;path=joinpath(tempdir(),"brm-rk-consumer","domain-"*label*".stan"))
        mapping = Pair{Symbol,String}[names[spos]=>id*"_sigma"]
        append!(mapping,[names[zpos[b]]=>id*"_beta_raw.$b" for b in eachindex(zpos)])
        append!(mapping,[names[rpos[j]]=>(iso ? id*"_rho_iso" : id*"_rho.$j") for j in eachindex(rpos)])
        for u in (zeros(length(names)),fill(0.13,length(names)),collect(range(-0.2,0.3;length=length(names))))
            check_consumer_point(problem,u,oracle)
            check_consumer_stan(problem,stan,mapping,backend,u)
        end
        artifact = BRM.emit_rk_artifact(brmi;case_id=label)
        rebuilt = build_kernel(BRM.rk_translate_artifact(artifact))
        @test hsgp_plate_depths(kernel_graph(rebuilt.spec)) == inventory
        @test isequal(data,original_data)
    end
end
