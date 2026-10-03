# Public consumer semantics: full normalized density, ordinary reverse AD,
# independent oracles, actual emitted Stan, and complete printed RK replay.
using Test, BayesianRegressionModels, Distributions, StanBlocks
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
import BridgeStan
using CategoricalArrays
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))
include(joinpath(@__DIR__, "rk_source_roundtrip.jl"))

function consumer_problem(brmi)
    backend = check_rk_source_roundtrip(RKBRMI(brmi))
    println("RK_NAMES=", coordinate_names(backend.model.layout)); flush(stdout)
    backend, rk_logdensity_problem(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
end

function consumer_stan(brmi, name)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    folder = joinpath(tempdir(), "brm-rk-consumer")
    mkpath(folder)
    problem = BRM.stan_instantiate(sb; path=joinpath(folder, name * ".stan"))
    println("STAN_NAMES=", BridgeStan.param_unc_names(problem.model)); flush(stdout)
    problem
end

function check_consumer_point(problem, u, oracle)
    before = copy(u)
    value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
    @test value ≈ oracle(u) atol=2e-11 rtol=2e-11
    step = 1e-5
    independent_gradient = map(eachindex(u)) do j
        plus, minus = copy(u), copy(u)
        plus[j] += step; minus[j] -= step
        (oracle(plus) - oracle(minus)) / (2step)
    end
    @test gradient ≈ independent_gradient atol=2e-8 rtol=2e-8
    @test isequal(u, before)
    value, gradient
end

function check_consumer_stan(problem, stan, mapping, backend, u)
    permutation = BRM.resolve_sb_map(mapping, coordinate_names(backend.model.layout),
        BridgeStan.param_unc_names(stan.model); case_id="public-consumer")
    stan_u = BRM.apply_sb_map(u, permutation)
    gradient = similar(stan_u)
    value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u, gradient;
        propto=false, jacobian=true)
    rk_value, rk_gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
    @test rk_value ≈ value atol=2e-11 rtol=2e-11
    @test rk_gradient ≈ BRM.unmap_sb_grad(gradient, permutation) atol=2e-10 rtol=2e-10
end

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
    mapping = [:mu_Intercept => "pop_mu_beta_pop.1",
        Symbol("mu_g.1") => "cat_mu_g_beta.1", :sigma => "sigma"]
    names = coordinate_names(backend.model.layout)
    ia, ib, is = indexin([:mu_Intercept, Symbol("mu_g.1"), :sigma], names)
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
        ia=findfirst(==(:mu_Intercept),names)
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
        ordered=label=="reference" ? [:mu_Intercept,Symbol("mu_g.1"),Symbol("mu_g.2")] :
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
    ia=findfirst(==(:mu_Intercept),names)
    it=findfirst(==(Symbol("ranef_draws_g_sd.1")),names)
    iz=Int.(indexin([Symbol("ranef_draws_g_z.$i.1") for i in 1:2],names))
    oracle(u)=sum(logpdf.(Normal.(u[ia] .+ exp(u[it]).*u[iz][data.g],1),data.y)) +
        logpdf(Normal(),u[ia]) + logpdf(Normal(),u[it]) + sum(logpdf.(Normal(),u[iz]))
    mapping=[:mu_Intercept=>"pop_mu_beta_pop.1",
        Symbol("ranef_draws_g_sd.1")=>"r_mu_g_log_scale",
        Symbol("ranef_draws_g_z.1.1")=>"r_mu_g_xi.1",
        Symbol("ranef_draws_g_z.2.1")=>"r_mu_g_xi.2"]
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
        scale=findfirst(n->occursin("_sd.",string(n)),names)
        z=findall(n->occursin("_z.",string(n)),names)
        @test length(z)==2
        oracle(u)=begin
            t=exp(u[scale]); mu=t.*u[z]
            label=="slope" && (mu=mu.*data.x)
            scale_prior=label=="unnamed" ? logpdf(Normal(),u[scale]) :
                logpdf(Normal(),t)+log(2)+u[scale]
            sum(logpdf.(Normal.(mu,1),data.y))+sum(logpdf.(Normal(),u[z]))+scale_prior
        end
        for u in (zeros(length(names)),collect(range(-.2,.3;length=length(names))))
            check_consumer_point(problem,u,oracle)
        end
        @test bucket.kind == (label=="named" ? :correlated : label=="slope" ? :slope1 : :intercept1)
    end
end

consumer_shift(x,beta)=x .+ beta

@stestset "reader censoring carries threshold masses and bound dependencies" begin
    for (label, data, model) in (
            ("original",(;x=[-.2,.2],y=[.1,.3],lo=[0.,0.]), @brm(begin
                beta ~ Normal(0,1)
                mu=consumer_shift(x,beta)
                y ~ censored(Normal(mu,1);lower=lo)
            end)),
            ("mixed",(;x=[-.2,.2,.4],y=[0.,1.,.3],lo=zeros(3),hi=ones(3)), @brm(begin
                beta ~ Normal(0,1)
                mu=consumer_shift(x,beta)
                y ~ censored(Normal(mu,1);lower=lo,upper=hi)
            end)),
            ("assigned-bound",(;x=[-.2,.2,.4],y=[0.,1.,.3],lo=zeros(3),hi=ones(3)), @brm(begin
                beta ~ Normal(0,1)
                mu=consumer_shift(x,beta)
                lower=lo+0.0
                y ~ censored(Normal(mu,1);lower,upper=hi)
            end)))
        before=deepcopy(data)
        backend, problem=consumer_problem(model(data))
        @test LogDensityProblems.dimension(problem)==1
        oracle(u)=logpdf(Normal(),u[1])+sum(eachindex(data.y)) do i
            mu=data.x[i]+u[1]; lo=data.lo[i]; hi=hasproperty(data,:hi) ? data.hi[i] : Inf
            data.y[i]==lo ? logcdf(Normal(mu,1),lo) :
                data.y[i]==hi ? logccdf(Normal(mu,1),hi) : logpdf(Normal(mu,1),data.y[i])
        end
        for u in ([0.],[-.5],[.5])
            check_consumer_point(problem,u,oracle)
        end
        @test isequal(data,before)
    end
end
