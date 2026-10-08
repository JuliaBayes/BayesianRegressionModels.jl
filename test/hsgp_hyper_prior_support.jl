include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# This public specimen is the unchanged reported grouped HSGP model. The
# override addresses its authored log intercept, rather than a positive scale.
const explicit_log_hyper_model = @brm begin
    mu ~ 1+hsgp(x;k=3,by=g)
    log(length_scale(hsgp(x))) ~ 1+(1|g)
    log(sd(hsgp(x))) ~ 1+(1|g)
    length_scale(:,hsgp(x)) ~ Normal(.2,.7)
    sd(:,hsgp(x)) ~ Normal(-.1,.8)
    y ~ Normal(mu,1)
end

const ungrouped_log_hyper_model = @brm begin
    mu ~ 1+hsgp(x;k=3)
    log(length_scale(hsgp(x))) ~ 1
    log(sd(hsgp(x))) ~ 1
    length_scale(:,hsgp(x)) ~ Normal(.2,.7)
    sd(:,hsgp(x)) ~ Normal(-.1,.8)
    y ~ Normal(mu,1)
end

const bounded_log_hyper_model = @brm begin
    mu ~ 1+hsgp(x;k=3,by=g)
    log(length_scale(hsgp(x))) ~ 1+(1|g)
    log(sd(hsgp(x))) ~ 1+(1|g)
    length_scale(:,hsgp(x)) ~ Uniform(-1.2,1.4)
    sd(:,hsgp(x)) ~ Normal(-.1,.8;lower=-.8,upper=1.6)
    y ~ Normal(mu,1)
end

const truncated_log_hyper_model = @brm begin
    mu ~ 1+hsgp(x;k=3,by=g)
    log(length_scale(hsgp(x))) ~ 1+(1|g)
    log(sd(hsgp(x))) ~ 1+(1|g)
    length_scale(:,hsgp(x)) ~ truncated(Normal(.2,.7);lower=-1.,upper=.9)
    sd(:,hsgp(x)) ~ LogNormal(0.,.3)
    y ~ Normal(mu,1)
end

function log_hyper_basis(x,k,c)
    center=sum(x)/length(x)
    width=c*maximum(abs.(x.-center))
    frequencies=[(j*pi/(2width))^2 for j in 1:k]
    phi=[sin(sqrt(frequencies[j])*(v-center+width))/sqrt(width) for v in x,j in 1:k]
    floor=4width/pi*sqrt(log(100)/(k^2-1))
    phi,frequencies,floor
end

function log_hyper_coordinate(u,lo,hi)
    if isnothing(lo) && isnothing(hi)
        return u,0.
    elseif isnothing(hi)
        return lo+exp(u),u
    end
    probability=1/(1+exp(-u))
    lo+(hi-lo)*probability,log(hi-lo)-log1p(exp(-u))-log1p(exp(u))
end

function log_hyper_printed_sampler(backend)
    emitted=BRM._rk_emit_ast(backend.plan)
    namespace=Module(gensym(:PrintedLogHyper))
    Core.eval(namespace,:(using ReactiveKernelsPPL))
    Core.eval(namespace,:(import ReactiveKernels))
    for (name,value) in emitted.bindings
        Core.eval(namespace,Expr(:const,Expr(:(=),name,QuoteNode(value))))
    end
    definitions=join(map(emitted.defs) do definition
        prefix=BRM._rk_source_definition(definition).kind===:rkppl ? "@rkppl " : ""
        prefix*sprint(Base.show_unquoted,definition)
    end,"\n")
    Core.eval(namespace,Meta.parseall(definitions))
    body=Meta.parse(sprint(Base.show_unquoted,emitted.main))
    # Emitted assignments own prepared indices retained in the plan cache.
    inputs=BRM._rk_source_data_columns(backend.plan,emitted)
    bound=bind_data(lower_rkppl(body,inputs;mod=namespace,
        conditioned=BRM._rk_observed_names(backend.plan)),inputs)
    built=build_kernel(bound)
    @test coordinate_names(built.layout)==coordinate_names(backend.model.layout)
    built,prepare_sampler(built,bound,zeros(built.layout.total);
        backend=AutoEnzyme(;mode=Enzyme.Reverse))
end

@stestset "explicit log HSGP intercept priors preserve support density and source" begin
    data=(;x=[-.5,0.,.5,1.],g=[1,1,2,2],y=[-1.,.1,1.,.4])
    original=deepcopy(data)
    cases=(
        ("original",explicit_log_hyper_model,true,Normal(.2,.7),Normal(-.1,.8),
            (nothing,nothing),(nothing,nothing)),
        ("ungrouped",ungrouped_log_hyper_model,false,Normal(.2,.7),Normal(-.1,.8),
            (nothing,nothing),(nothing,nothing)),
        ("declaration-bounds",bounded_log_hyper_model,true,Uniform(-1.2,1.4),Normal(-.1,.8),
            (-1.2,1.4),(-.8,1.6)),
        ("normalized-truncation",truncated_log_hyper_model,true,
            truncated(Normal(.2,.7);lower=-1.,upper=.9),LogNormal(0.,.3),
            (-1.,.9),(0.,nothing)))
    for (label,builder,grouped,rho_law,sigma_law,rho_bounds,sigma_bounds) in cases
        brmi=builder(data)
        backend,problem=consumer_problem(brmi)
        names=coordinate_names(backend.model.layout)
        position(name)=only(findall(==(Symbol(name)),names))
        intercept=position("pop_mu.beta_pop.1")
        G=grouped ? 2 : 1
        weights=grouped ? [position("hsgp_x.z.$g.$k") for g in 1:G,k in 1:3] :
            reshape([position("hsgp_x.z.$k") for k in 1:3],1,3)
        term=only(filter(t->t.kind===:hsgp,only(backend.plan.predictors).terms))
        phi,frequencies,floor=log_hyper_basis(data.x,3,term.options.c)
        mapping=Pair{Symbol,String}[Symbol("pop_mu.beta_pop.1")=>"pop_mu_beta_pop.1"]
        for g in 1:G,k in 1:3
            target=grouped ? "zflat_hsgpw_x_g.$((g-1)*3+k)" : "hsgp_x_beta_raw.$k"
            push!(mapping,names[weights[g,k]]=>target)
        end
        for stem in ("rho","sigma")
            prefix=grouped ? "hsgp_x_by_g" : "hsgp_x"
            push!(mapping,Symbol("hsgp_x.$(stem)_Intercept")=>"$(prefix)_beta0_$stem")
            if grouped
                push!(mapping,Symbol("hsgp_x.$(stem)_sd")=>"$(prefix)_sd_$stem")
                for g in 1:G
                    push!(mapping,Symbol("hsgp_x.$(stem)_z.$g")=>"$(prefix)_z_$stem.$g")
                end
            end
        end
        function physical(u)
            prior=logpdf(Normal(),u[intercept])+sum(logpdf.(Normal(),u[weights]))
            hypers=map((("rho",rho_law,rho_bounds),("sigma",sigma_law,sigma_bounds))) do (stem,law,bounds)
                beta,jac=log_hyper_coordinate(u[position("hsgp_x.$(stem)_Intercept")],bounds...)
                prior+=logpdf(law,beta)+jac
                eta=fill(beta,G)
                if grouped
                    sd=position("hsgp_x.$(stem)_sd")
                    zs=[position("hsgp_x.$(stem)_z.$g") for g in 1:G]
                    prior+=logpdf(Normal(),exp(u[sd]))+u[sd]+sum(logpdf.(Normal(),u[zs]))
                    eta+=exp(u[sd]).*u[zs]
                end
                value=exp.(eta)
                stem=="rho" ? max.(value,floor) : value
            end
            rho,sigma=hypers
            group_index=grouped ? data.g : ones(Int,length(data.y))
            mu=[u[intercept]+sum(phi[i,k]*u[weights[group_index[i],k]]*
                sigma[group_index[i]]*sqrt(rho[group_index[i]]*sqrt(2pi))*
                exp(-rho[group_index[i]]^2*frequencies[k]/4) for k in 1:3)
                for i in eachindex(data.y)]
            prior+sum(logpdf.(Normal.(mu,1),data.y)),rho,sigma
        end
        oracle(u)=first(physical(u))
        sb=SBBRMI(brmi)
        code=BRM.stan_code(sb)
        if label in ("original","ungrouped")
            prefix=grouped ? "hsgp_x_by_g" : "hsgp_x"
            for stem in ("rho","sigma")
                @test occursin("real $(prefix)_beta0_$stem",code)
                @test !occursin("real<lower=0.0> $(prefix)_beta0_$stem",code)
            end
        end
        stan=consumer_stan(brmi,"log-hyper-support-"*label)
        built,replay=log_hyper_printed_sampler(backend)
        replay_gradient=zeros(length(names))
        for delta in (-.6,.15,-.35)
            u=collect(range(-.4,.4;length=length(names)))
            u[position("hsgp_x.rho_Intercept")]=delta
            u[position("hsgp_x.sigma_Intercept")]=delta/2
            if grouped
                u[position("hsgp_x.rho_sd")]=-.7
                u[position("hsgp_x.rho_z.1")]=-.8
                u[position("hsgp_x.rho_z.2")]=3.2
                _,rho,sigma=physical(u)
                @test rho[1]==floor
                @test rho[2]>floor
                @test sigma[1]!=sigma[2]
            end
            value,gradient=check_consumer_point(problem,u,oracle)
            check_consumer_stan(problem,stan,mapping,backend,u)
            before=copy(u)
            replay_value,_=Base.invokelatest(sampler_value_and_gradient!,replay,replay_gradient,u)
            @test replay_value==value
            @test replay_gradient==gradient
            @test isequal(u,before)
        end
        @test isequal(data,original)
    end
end
