include(joinpath(@__DIR__, "rk_consumer_support.jl"))

@stestset "unchanged scalar vector assignment full density and mapped Stan" begin
    data=(;x=[-.4,.2,.7],y=[.1,-.2,.3]); before=deepcopy(data)
    model=@brm data begin
        a ~ Normal(0,1)
        b ~ Normal(0,1)
        sigma ~ Exponential(1)
        mu=a+b*x
        y ~ Normal(mu,sigma)
    end
    backend,problem=consumer_problem(model)
    stan=consumer_stan(model,"value-arithmetic-original")
    names=coordinate_names(backend.model.layout)
    ia,ib,is=Int.(indexin([:a,:b,:sigma],names))
    oracle(u)=logpdf(Normal(),u[ia])+logpdf(Normal(),u[ib])+
        logpdf(Exponential(1),exp(u[is]))+u[is]+
        sum(logpdf.(Normal.(u[ia].+u[ib].*data.x,exp(u[is])),data.y))
    for (a,b,s) in ((0.,0.,0.),(.3,-.2,.1),(-.4,.7,-.3))
        u=zeros(3);u[ia]=a;u[ib]=b;u[is]=s
        check_consumer_point(problem,u,oracle)
        check_consumer_stan(problem,stan,[:a=>"a",:b=>"b",:sigma=>"sigma"],backend,u)
    end
    @test isequal(data,before)
end

value_array_kw(x,a;offset=0.)=x .+ a .+ offset

@stestset "prepared assignment arithmetic math reductions and callable semantics" begin
    data=(;x=[-.4,.2,.7],offset=[.2,.4,-.1],y=[.1,-.2,.3])
    cases=(
        ("nested",@brm(data,begin
            a ~ Normal(0,1)
            q=a+x*offset/(1+x^2)
            mu=log1p(exp(q))/(sqrt(2)+1)
            y ~ Normal(mu,1)
        end), a->log1p.(exp.(a .+ data.x.*data.offset./(1 .+ data.x.^2)))./(sqrt(2)+1)),
        ("reduction",@brm(data,begin
            a ~ Normal(0,1)
            mu=a+sum(x)
            y ~ Normal(mu,1)
        end), a->a+sum(data.x)),
        ("vector-index",@brm(data,begin
            a ~ Normal(0,1)
            v=[a,2]
            mu=v[1]+v[2]*x
            y ~ Normal(mu,1)
        end), a->a .+ 2data.x),
        ("whole-call",@brm(data,begin
            a ~ Normal(0,1)
            mu=value_array_kw(x,a;offset=.3)
            y ~ Normal(mu,1)
        end), a->data.x .+ a .+ .3),
        ("scalar-math",@brm(data,begin
            a ~ Normal(0,1)
            q=exp(a)/(1+exp(a))
            mu=log(q)
            y ~ Normal(mu,1)
        end), a->log(exp(a)/(1+exp(a))))
    )
    for (label,model,location) in cases
      @testset "$label" begin
        before=deepcopy(data)
        backend,problem=consumer_problem(model)
        @test coordinate_names(backend.model.layout)==[:a]
        oracle(u)=logpdf(Normal(),u[1])+sum(logpdf.(Normal.(location(u[1]),1),data.y))
        for u in ([.13],[-.2],[.3])
            check_consumer_point(problem,u,oracle)
        end
        @test isequal(data,before)
      end
    end
    scalar=(;x=.7,y=[.1,-.2,.3])
    model=@brm scalar begin
        a ~ Normal(0,1)
        mu=a+x/2
        y ~ Normal(mu,1)
    end
    _,problem=consumer_problem(model)
    check_consumer_point(problem,[.13],u->logpdf(Normal(),u[1])+
        sum(logpdf.(Normal(u[1]+.35,1),scalar.y)))
end
