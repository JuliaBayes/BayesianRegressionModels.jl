include(joinpath(@__DIR__, "rk_consumer_support.jl"))

@stestset "unchanged categorical formula and shared default prior" begin
    data = (; g=[1,1,2,2], y=[0.1,0.2,0.7,0.8])
    original = deepcopy(data)
    brmi = @brm data begin
        mu ~ 1 + g
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    backend, problem = consumer_problem(brmi)
    @test LogDensityProblems.dimension(problem) == 3
    stan = consumer_stan(brmi, "categorical-default")
    mapping = [Symbol("pop_mu.beta_pop.1") => "pop_mu_beta_pop.1",
        Symbol("mu_g.1") => "cat_mu_g_beta.1", :sigma => "sigma"]
    names = coordinate_names(backend.model.layout)
    ia, ib, is = indexin([Symbol("pop_mu.beta_pop.1"), Symbol("mu_g.1"), :sigma], names)
    oracle(u) = begin
        a, b, logs = u[ia], u[ib], u[is]; sigma = exp(logs)
        sum(logpdf.(Normal.(a .+ b .* (data.g .== 2), sigma), data.y)) +
            logpdf(Normal(), a) + logpdf(Normal(), b) +
            logpdf(Exponential(1), sigma) + logs
    end
    for (a,b,logs) in ((0.,0.,0.), (.3,-.2,.1), (-.4,.7,-.3))
        u = zeros(3); u[ia]=a; u[ib]=b; u[is]=logs
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, mapping, backend, u)
    end
    @test isequal(data, original)
end

@stestset "categorical levels reference overrides and cell means" begin
    pool = categorical(["b","a","b","a"])
    levels!(pool, ["b","unused","a"])
    # Each expected design is independently specified in declared/sorted order.
    for (i, (g, rows)) in enumerate(((fill(1,4), zeros(4,0)),
            ([2,5,9,5], Float64[0 0; 1 0; 0 1; 1 0]),
            (pool, Float64[0 0; 0 1; 0 0; 0 1])))
        data=(;g, y=[0.1,0.2,0.7,0.8]); before=deepcopy(data)
        model=@brm data begin
            mu ~ 1 + g
            y ~ Normal(mu,1)
        end
        backend, problem=consumer_problem(model)
        @test LogDensityProblems.dimension(problem) == 1 + size(rows,2)
        names=coordinate_names(backend.model.layout)
        ia=findfirst(==(Symbol("pop_mu.beta_pop.1")),names)
        ib=findall(n->startswith(string(n),"mu_g."),names)
        oracle(u)=sum(logpdf.(Normal.(u[ia] .+ rows*u[ib],1),data.y)) +
            sum(logpdf.(Normal(),u))
        for u in (zeros(length(names)), length(names)==1 ? [-.2] :
                collect(range(-0.2,0.3;length=length(names))))
            check_consumer_point(problem,u,oracle)
        end
        @test isequal(data,before)
    end
    data=(;g=[1,2,3,2], h=[1,1,2,2], y=[0.1,0.2,0.7,0.8])
    for (label, model, design, priors) in (
            ("reference", @brm(data, begin
                mu ~ 1 + factor(g;ref=3)
                effect(mu,g) ~ Normal(0.2,0.7)
                y ~ Normal(mu,1)
            end), Float64[1 0 1; 1 1 0; 1 0 0; 1 1 0],
                (Normal(),Normal(.2,.7),Normal(.2,.7))),
            ("cell-means", @brm(data, begin
                mu ~ 0 + g + h
                y ~ Normal(mu,1)
            end), Float64[1 0 0 0; 0 1 0 0; 0 0 1 1; 0 1 0 1],
                (Normal(),Normal(),Normal(),Normal())))
        backend, problem=consumer_problem(model)
        names=coordinate_names(backend.model.layout)
        ordered=label=="reference" ? [Symbol("pop_mu.beta_pop.1"),Symbol("mu_g.1"),Symbol("mu_g.2")] :
            [Symbol("mu_g.$i") for i in 1:3] ∪ [Symbol("mu_h.1")]
        indices=Int.(indexin(ordered,names))
        @test length(indices)==length(names)
        oracle(u)=sum(logpdf.(Normal.(design*u[indices],1),data.y)) +
            sum(logpdf(priors[j],u[indices[j]]) for j in eachindex(indices))
        for u in (zeros(length(names)), collect(range(-.2,.3;length=length(names))))
            check_consumer_point(problem,u,oracle)
        end
    end
end

@stestset "scoped ordinary scalar intercept lognormal prior" begin
    data=(;g=[1,1,2,2],y=[-1.,.1,1.,.4]); before=deepcopy(data)
    brmi=@brm data begin
        mu ~ 1 + (1|g)
        y ~ Normal(mu,1)
    end
    backend, problem=consumer_problem(brmi)
    stan=consumer_stan(brmi,"scalar-intercept")
    names=coordinate_names(backend.model.layout)
    ia=findfirst(==(Symbol("pop_mu.beta_pop.1")),names)
    it=findfirst(==(Symbol("b_g.tau.1")),names)
    iz=Int.(indexin([Symbol("b_g.z.$i.1") for i in 1:2],names))
    oracle(u)=sum(logpdf.(Normal.(u[ia] .+ exp(u[it]).*u[iz][data.g],1),data.y)) +
        logpdf(Normal(),u[ia]) + logpdf(Normal(),u[it]) + sum(logpdf.(Normal(),u[iz]))
    mapping=[Symbol("pop_mu.beta_pop.1")=>"pop_mu_beta_pop.1",
        Symbol("b_g.tau.1")=>"r_mu_g_log_scale",
        Symbol("b_g.z.1.1")=>"r_mu_g_xi.1",
        Symbol("b_g.z.2.1")=>"r_mu_g_xi.2"]
    for u in (zeros(4),fill(.3,4),fill(-.1,4),[-.3,.2,.1,-.4])
        check_consumer_point(problem,u,oracle)
        check_consumer_stan(problem,stan,mapping,backend,u)
    end
    @test isequal(data,before)
end

@stestset "random-effect-only predictors retain zero population terms" begin
    data=(;subject=[1,2],x=[-.2,.2],y=[.1,.3])
    for (label, model) in (
            ("named",@brm(data,begin
                eta ~ 0 + (1|p|subject)
                y ~ Normal(eta,1)
            end)),
            ("unnamed",@brm(data,begin
                eta ~ 0 + (1|subject)
                y ~ Normal(eta,1)
            end)),
            ("slope",@brm(data,begin
                eta ~ 0 + (0+x|subject)
                y ~ Normal(eta,1)
            end)))
        @test isempty(popcoefnames(model,:eta))
        backend, problem=consumer_problem(model)
        @test BRM._rk_num_coefficients(backend.plan)==0
        bucket=only(backend.plan.ranef_buckets)
        names=coordinate_names(backend.model.layout)
        scale=findfirst(n->occursin(".tau.",string(n)),names)
        z=findall(n->occursin(".z.",string(n)),names)
        @test length(z)==2
        oracle(u)=begin
            t=exp(u[scale]); mu=t.*u[z]
            label=="slope" && (mu=mu.*data.x)
            scale_prior=label=="unnamed" ? logpdf(Normal(),u[scale]) :
                logpdf(Normal(),t)+u[scale]
            sum(logpdf.(Normal.(mu,1),data.y))+sum(logpdf.(Normal(),u[z]))+scale_prior
        end
        stan=consumer_stan(model,"random-only-"*label)
        prefix=label=="named" ? "b_p_subject" : "r_eta_subject"
        stan_scale=label=="unnamed" ? "$(prefix)_log_scale" : "$(prefix)_tau.1"
        stan_z=label=="named" ? "$(prefix)_z_flat" : "$(prefix)_xi"
        mapping=[names[scale]=>stan_scale; [names[z[i]]=>"$stan_z.$i" for i in 1:2]]
        for u in (zeros(length(names)),collect(range(-.2,.3;length=length(names))))
            check_consumer_point(problem,u,oracle)
            check_consumer_stan(problem,stan,mapping,backend,u)
        end
        @test bucket.kind == (label=="named" ? :correlated : label=="slope" ? :slope1 : :intercept1)
    end
end

consumer_shift(x,beta)=x .+ beta
const censored_lower_builder = @brm begin
    beta ~ Normal(0,1)
    mu=consumer_shift(x,beta)
    y ~ censored(Normal(mu,1);lower=lo)
end
const censored_both_builder = @brm begin
    beta ~ Normal(0,1)
    mu=consumer_shift(x,beta)
    y ~ censored(Normal(mu,1);lower=lo,upper=hi)
end
const censored_upper_builder = @brm begin
    beta ~ Normal(0,1)
    mu=consumer_shift(x,beta)
    y ~ censored(Normal(mu,1);upper=hi)
end
const censored_assignment_builder = @brm begin
    beta ~ Normal(0,1)
    mu=consumer_shift(x,beta)
    lower=consumer_shift(lo,0.0)
    y ~ censored(Normal(mu,1);lower=lower,upper=hi)
end
const truncated_reader_builder=@brm begin
    beta ~ Normal(0,1)
    mu=consumer_shift(x,beta)
    y ~ truncated(Normal(mu,1);lower=lo,upper=hi)
end
const interval_reader_builder=@brm begin
    beta ~ Normal(0,1)
    mu=consumer_shift(x,beta)
    y ~ interval_censored(Normal(mu,1);upper=hi)
end

module ConsumerCensorStan
using BayesianRegressionModels,StanBlocks
@deffun consumer_shift(x::vector[n],beta::real)::vector[n]=x+beta
const lower=@brm begin
    beta ~ Normal(0,1)
    mu=consumer_shift(x,beta)
    y ~ censored(Normal(mu,1);lower=lo)
end
const upper=@brm begin
    beta ~ Normal(0,1)
    mu=consumer_shift(x,beta)
    y ~ censored(Normal(mu,1);upper=hi)
end
const both=@brm begin
    beta ~ Normal(0,1)
    mu=consumer_shift(x,beta)
    y ~ censored(Normal(mu,1);lower=lo,upper=hi)
end
const assigned=@brm begin
    beta ~ Normal(0,1)
    mu=consumer_shift(x,beta)
    lower=consumer_shift(lo,0.0)
    y ~ censored(Normal(mu,1);lower=lower,upper=hi)
end
const truncated_reader=@brm begin
    beta ~ Normal(0,1)
    mu=consumer_shift(x,beta)
    y ~ truncated(Normal(mu,1);lower=lo,upper=hi)
end
const interval_reader=@brm begin
    beta ~ Normal(0,1)
    mu=consumer_shift(x,beta)
    y ~ interval_censored(Normal(mu,1);upper=hi)
end
end

@stestset "reader censoring carries threshold masses and bound dependencies" begin
    for (label, data, model) in (
            ("original",(;x=[-.2,.2],y=[.1,.3],lo=[0.,0.]),censored_lower_builder),
            ("scalar-lower",(;x=[-.2,.2],y=[0.,.3],lo=0.),censored_lower_builder),
            ("scalar-upper",(;x=[-.2,.2],y=[.1,.5],hi=.5),censored_upper_builder),
            ("row-upper",(;x=[-.2,.2],y=[.1,.7],hi=[.5,.7]),censored_upper_builder),
            ("mixed",(;x=[-.2,.2,.4],y=[0.,1.,.3],lo=zeros(3),hi=ones(3)),censored_both_builder),
            ("assigned-bound",(;x=[-.2,.2,.4],y=[0.,1.,.3],lo=zeros(3),hi=ones(3)),censored_assignment_builder),
            ("truncated",(;x=[-.2,.2,.4],y=[.1,.3,.5],lo=zeros(3),hi=ones(3)),truncated_reader_builder),
            ("interval",(;x=[-.2,.2,.4],y=[0.,.2,.4],hi=[.5,.5,.8]),interval_reader_builder))
        before=deepcopy(data)
        backend, problem=consumer_problem(model(data))
        @test LogDensityProblems.dimension(problem)==1
        # The independent Stan oracle uses the original raw bounds. BRM's
        # Stan bound validator presently accepts data-backed bounds only;
        # the native named bound assignment is the exact identity lo+0.
        stan_builder=label=="truncated" ? ConsumerCensorStan.truncated_reader :
            label=="interval" ? ConsumerCensorStan.interval_reader :
            label in ("assigned-bound","mixed") ? ConsumerCensorStan.both :
            occursin("upper",label) ? ConsumerCensorStan.upper : ConsumerCensorStan.lower
        stan=consumer_stan(stan_builder(data),"reader-censor-"*label;mod=ConsumerCensorStan)
        oracle(u)=logpdf(Normal(),u[1])+sum(eachindex(data.y)) do i
            mu=data.x[i]+u[1]
            lo=hasproperty(data,:lo) ? (data.lo isa Number ? data.lo : data.lo[i]) : -Inf
            hi=hasproperty(data,:hi) ? (data.hi isa Number ? data.hi : data.hi[i]) : Inf
            label=="truncated" ? logpdf(truncated(Normal(mu,1),lo,hi),data.y[i]) :
                label=="interval" ? log(cdf(Normal(mu,1),hi)-cdf(Normal(mu,1),data.y[i])) :
                data.y[i]==lo ? logcdf(Normal(mu,1),lo) :
                data.y[i]==hi ? logccdf(Normal(mu,1),hi) : logpdf(Normal(mu,1),data.y[i])
        end
        for u in ([0.],[-.5],[.5])
            check_consumer_point(problem,u,oracle)
            check_consumer_stan(problem,stan,[:beta=>"beta"],backend,u)
        end
        @test isequal(data,before)
    end
end

const censored_multiple_builder=@brm begin
    beta ~ Normal(0,1)
    a=consumer_shift(x,beta)
    b=consumer_shift(x2,beta)
    y ~ censored(Normal(a,1);lower=lo)
    y2 ~ censored(Normal(b,1);upper=hi)
end

@stestset "reader evidence independent row axes and attributed invalid data" begin
    data=(;x=[-.2,.2],y=[0.,.3],lo=0.,x2=[-.1,.3,.5],y2=[.1,.7,.2],hi=.7)
    before=deepcopy(data)
    backend,problem=consumer_problem(censored_multiple_builder(data))
    oracle(u)=logpdf(Normal(),u[1])+
        sum(y==0 ? logcdf(Normal(x+u[1],1),0) : logpdf(Normal(x+u[1],1),y)
            for (x,y) in zip(data.x,data.y))+
        sum(y==.7 ? logccdf(Normal(x+u[1],1),.7) : logpdf(Normal(x+u[1],1),y)
            for (x,y) in zip(data.x2,data.y2))
    for u in ([0.],[-.5],[.5])
        check_consumer_point(problem,u,oracle)
    end
    @test isequal(data,before)
    for invalid in ((;x=[-.2,.2],y=[-.1,.3],lo=zeros(2),hi=ones(2)),
            (;x=[-.2,.2],y=[.1,.3],lo=[0.,.8],hi=[1.,.2]))
        saved=deepcopy(invalid)
        try
            RKBRMI(censored_both_builder(invalid))
            @test false
        catch err
            message=sprint(showerror,err)
            @test occursin("y",message)
            @test occursin("row",message)
        end
        @test isequal(invalid,saved)
    end
end
