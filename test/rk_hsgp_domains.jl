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

# Replay the complete printed definitions and body, preserving explicit
# bindings and observation metadata, in a fresh defining namespace.
function printed_hsgp_replay(backend)
    emitted=BRM._rk_emit_ast(backend.plan)
    namespace=Module(gensym(:PrintedHSGP))
    Core.eval(namespace,:(using ReactiveKernelsPPL))
    Core.eval(namespace,:(import ReactiveKernels))
    for (name,value) in emitted.bindings
        Core.eval(namespace,Expr(:const,Expr(:(=),name,QuoteNode(value))))
    end
    definitions=join(map(emitted.defs) do definition
        prefix=BRM._rk_source_definition(definition).kind === :rkppl ? "@rkppl " : ""
        prefix*sprint(Base.show_unquoted,definition)
    end,"\n")
    Core.eval(namespace,Meta.parseall(definitions))
    body=Meta.parse(sprint(Base.show_unquoted,emitted.main))
    bound=bind_data(lower_rkppl(body,backend.plan.columns;mod=namespace,
        conditioned=BRM._rk_observed_names(backend.plan)),backend.plan.columns)
    built=build_kernel(bound)
    sampler=prepare_sampler(built,bound,zeros(built.layout.total);
        backend=AutoEnzyme(;mode=Enzyme.Reverse))
    built,sampler
end

@stestset "fixed HSGP domain graphs retain basis, priors and normalized Stan density" begin
    data = (; x=[-0.7,0.0,0.6], w=[0.4,-0.2,0.7], y=[0.2,-0.1,0.4])
    cases = (
        ("original", (3,), ((-2.0,2.0),), true, @brm(data, begin
            location ~ 0 + hsgp(x;k=3,domain=(-2.0,2.0))
            y ~ Normal(location,1.0)
        end)),
        ("single-mode", (1,), ((-2.0,2.0),), true, @brm(data, begin
            location ~ 0 + hsgp(x;k=1,domain=(-2.0,2.0))
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
        replayed,sampler=printed_hsgp_replay(backend)
        @test coordinate_names(replayed.layout)==names
        @test hsgp_plate_depths(kernel_graph(replayed.spec))==inventory
        for u in (zeros(length(names)),fill(0.13,length(names)),collect(range(-0.2,0.3;length=length(names))))
            check_consumer_point(problem,u,oracle)
            check_consumer_stan(problem,stan,mapping,backend,u)
            gradient=zeros(length(u))
            value,_=sampler_value_and_gradient!(sampler,gradient,u)
            expected,expected_gradient=LogDensityProblems.logdensity_and_gradient(problem,u)
            @test value≈expected atol=2e-12 rtol=2e-12
            @test gradient≈expected_gradient atol=2e-10 rtol=2e-10
        end
        artifact = BRM.emit_rk_artifact(brmi;case_id=label)
        rebuilt = build_kernel(BRM.rk_translate_artifact(artifact))
        @test hsgp_plate_depths(kernel_graph(rebuilt.spec)) == inventory
        @test isequal(data,original_data)
    end
end

@stestset "fixed HSGP explicit hyperpriors preserve stated support and Jacobians" begin
    data=(;x=[-.5,-.1,.4,.9],y=[.2,-.1,.4,.3])
    before=deepcopy(data)
    builders=(
        "lognormal"=>@brm(begin
            location ~ 0+hsgp(x;k=3,domain=(-2.,2.))
            length_scale(:,hsgp(x)) ~ LogNormal(0.,1.)
            sd(:,hsgp(x)) ~ LogNormal(0.,1.)
            y ~ Normal(location,1.)
        end),
        "bounded"=>@brm(begin
            location ~ 0+hsgp(x;k=3,domain=(-2.,2.))
            length_scale(:,hsgp(x)) ~ Uniform(.2,2.)
            sd(:,hsgp(x)) ~ truncated(Normal(0.,1.);lower=0.)
            y ~ Normal(location,1.)
        end))
    PHI,omega2,_=fixed_hsgp_oracle((data.x,),(3,),((-2.,2.),))
    for (label,builder) in builders
        brmi=builder(data)
        backend,problem=consumer_problem(brmi)
        names=coordinate_names(backend.model.layout)
        index(name)=only(findall(==(Symbol(name)),names))
        r=index("hsgp_x_rho");s=index("hsgp_x_sigma")
        z=[index("hsgp_x_z.$b") for b in 1:3]
        term=only(filter(t->t.kind===:hsgp,only(backend.plan.predictors).terms))
        @test !term.options.rho_truncated
        @test term.options.rho_stated && term.options.sigma_stated
        function oracle(u)
            p=1/(1+exp(-u[r]))
            rho=label=="bounded" ? .2+1.8p : exp(u[r])
            sigma=exp(u[s])
            rho_law=label=="bounded" ? Uniform(.2,2.) : LogNormal()
            sigma_law=label=="bounded" ? truncated(Normal(),0,Inf) : LogNormal()
            jac=label=="bounded" ? log(1.8)+log(p)+log1p(-p) : u[r]
            weights=[sigma*sqrt(rho*sqrt(2pi))*exp(-rho^2*omega2[b,1]/4) for b in 1:3]
            locations=PHI*(weights.*u[z])
            logpdf(rho_law,rho)+logpdf(sigma_law,sigma)+jac+u[s]+
                sum(logpdf.(Normal(),u[z]))+sum(logpdf.(Normal.(locations,1.),data.y))
        end
        stan=consumer_stan(brmi,"fixed-hsgp-"*label)
        mapping=Pair{Symbol,String}[:hsgp_x_rho=>"hsgp_x_rho_iso",
            :hsgp_x_sigma=>"hsgp_x_sigma"]
        append!(mapping,[names[z[b]]=>"hsgp_x_beta_raw.$b" for b in 1:3])
        replayed,sampler=printed_hsgp_replay(backend)
        @test coordinate_names(replayed.layout)==names
        for u in (zeros(length(names)),fill(.13,length(names)),
                collect(range(-.2,.3;length=length(names))))
            check_consumer_point(problem,u,oracle)
            check_consumer_stan(problem,stan,mapping,backend,u)
            gradient=zeros(length(u))
            value,_=sampler_value_and_gradient!(sampler,gradient,u)
            expected,expected_gradient=LogDensityProblems.logdensity_and_gradient(problem,u)
            @test value≈expected atol=2e-12 rtol=2e-12
            @test gradient≈expected_gradient atol=2e-10 rtol=2e-10
        end
    end
    @test isequal(data,before)
end
