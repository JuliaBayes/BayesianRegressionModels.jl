include(joinpath(@__DIR__,"rk_consumer_support.jl"))

# Public constructor reduction supplied by the Bordet consumer. The scientific
# Stan family and its native scalar counterpart retain both inclusive atoms.
module PublicObservedCallableLaw
using BayesianRegressionModels, StanBlocks, Distributions, ReactiveKernelsPPL
import BayesianRegressionModels: _rk_callable_source!
StanBlocks.@deffun begin
    @lhs @lpxf tail_normal_lpdf(y::vector[n],mu::real,sigma::real,lo::real,hi::real)::real = begin
        result::real=0.0
        for i in 1:n
            if y[i]<=lo
                result+=normal_lcdf(lo,mu,sigma)
            elseif y[i]>=hi
                result+=normal_lccdf(hi,mu,sigma)
            else
                result+=normal_lpdf(y[i],mu,sigma)
            end
        end
        return result
    end
    tail_normal_lpdfs(y::vector[n],mu::real,sigma::real,lo::real,hi::real)::vector[n] = begin
        result::vector[n]
        for i in 1:n
            if y[i]<=lo
                result[i]=normal_lcdf(lo,mu,sigma)
            elseif y[i]>=hi
                result[i]=normal_lccdf(hi,mu,sigma)
            else
                result[i]=normal_lpdf(y[i],mu,sigma)
            end
        end
        return result
    end
    tail_normal_rng(vector[n],mu::real,sigma::real,lo::real,hi::real)::vector[n] = begin
        result::vector[n]
        for i in 1:n
            result[i]=fmin(hi,fmax(lo,normal_rng(mu,sigma)))
        end
        return result
    end
end

function _rk_callable_source!(definitions,bindings,entry,::typeof(tail_normal))
    scalar = Symbol(entry,:_scalar_lpdf)
    normal,cdf,ccdf,pdf = Symbol(entry,:_normal),Symbol(entry,:_cdf),
        Symbol(entry,:_ccdf),Symbol(entry,:_pdf)
    append!(bindings,[normal=>Distributions.Normal,cdf=>Distributions.logcdf,
        ccdf=>Distributions.logccdf,pdf=>Distributions.logpdf])
    push!(definitions,:(function $scalar(y,mu,sigma,lo,hi)
        d=$normal(mu,sigma)
        y<=lo && return $cdf(d,lo)
        y>=hi && return $ccdf(d,hi)
        return $pdf(d,y)
    end))
    push!(definitions,:(function $entry(mu,sigma,lo,hi)
        return ReactiveKernelsPPL.LogDensity($scalar,mu,sigma,lo,hi)
    end))
    :done
end

function build(data)
    @brm data begin
        mu ~ Normal(0,1)
        sigma ~ Exponential(1)
        y ~ tail_normal(mu,sigma,lo,hi)
    end
end
end

@stestset "exact observed callable constructor retains scalar cells without a formula predictor" begin
    data=(;y=[-2.0,-1.0,0.2,1.0,2.0],lo=-1.0,hi=1.0)
    saved=deepcopy(data)
    brmi=PublicObservedCallableLaw.build(data)
    backend,problem=consumer_problem(brmi)
    @test isempty(backend.plan.regression.predictors)
    @test coordinate_names(backend.model.layout)==[:mu,:sigma]
    oracle(u)=begin
        mu,sigma=u[1],exp(u[2])
        d=Normal(mu,sigma)
        logpdf(Normal(),mu)+logpdf(Exponential(),sigma)+u[2]+
            sum(y<=data.lo ? logcdf(d,data.lo) :
                y>=data.hi ? logccdf(d,data.hi) : logpdf(d,y) for y in data.y)
    end
    stan=consumer_stan(brmi,"observed-callable-law";mod=PublicObservedCallableLaw)
    for u in ([0.0,0.0],[0.17,-0.21],[-0.1,0.2])
        check_consumer_point(problem,u,oracle)
        check_consumer_stan(problem,stan,[:mu=>"mu",:sigma=>"sigma"],backend,u)
    end
    artifact=BRM.emit_rk_artifact(brmi;case_id="observed-callable-law")
    rebuilt=build_kernel(BRM.rk_translate_artifact(artifact))
    @test coordinate_names(rebuilt.layout)==[:mu,:sigma]
    @test isequal(data,saved)
end
