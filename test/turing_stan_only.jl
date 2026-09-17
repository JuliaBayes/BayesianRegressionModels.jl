using Test
using BayesianRegressionModels
using Distributions
using StanBlocks
using Turing

const BRM = BayesianRegressionModels

module TuringStanOnlyFixtures
using StanBlocks
StanBlocks.@deffun begin
    stan_vec(log_R::vector[T], log_I0::real, gen_pmf::vector[G])::vector[T] = begin
        out::vector[T]
        for t in 1:T
            out[t] = log_R[t] + log_I0 + gen_pmf[1]
        end
        out
    end
    @lhs @lpxf stanfam_lpmf(cases::int[N], Y::vector[N], cluster::real)::real = begin
        lp = 0.0
        for t in 1:N
            lp += normal_lpdf(Y[t], 0.0, 1.0)
        end
        lp
    end
    stanfam_lpmfs(cases::int[N], Y::vector[N], cluster::real)::vector[N] = begin
        lp::vector[N]
        for t in 1:N
            lp[t] = normal_lpdf(Y[t], 0.0, 1.0)
        end
        lp
    end
    stanfam_rng(int[N], Y::vector[N], cluster::real)::int[N] = begin
        out::int[N]
        for t in 1:N
            out[t] = 0
        end
        out
    end
end
end

juliafam(m, s) = Normal(m, s)

@testset "Stan-only assignment callee fails closed at construction" begin
    data = (; x=[-1.0, 0.5, 2.0, 0.25], y=[0.2, -0.1, 0.4, 0.3],
        gen_pmf=[0.5, 0.3, 0.2])
    brmi = (@brm begin
        log_I0 ~ Normal(log(50.0), 0.5)
        mu ~ 1 + x
        Y = TuringStanOnlyFixtures.stan_vec(mu, log_I0, gen_pmf)
        y ~ Normal(Y, 1)
    end)(data)
    err = try
        TuringBRMI(brmi)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("assignment `Y`", err.msg)
    @test occursin("stan_vec", err.msg)
end

@testset "Stan-only observation family fails closed at construction" begin
    data = (; x=[-1.0, 0.5, 2.0, 0.25], cases=[1, 0, 2, 1])
    brmi = (@brm begin
        cluster ~ Normal(0.0, 0.1; lower=0.0)
        mu ~ 1 + x
        cases ~ TuringStanOnlyFixtures.stanfam(mu, cluster)
    end)(data)
    err = try
        TuringBRMI(brmi)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("observation `cases`", err.msg)
    @test occursin("stanfam", err.msg)
end

@testset "ordinary Julia assignments and custom callables still lower" begin
    data = (; x=[-1.0, 0.5, 2.0, 0.25], y=[0.2, -0.1, 0.4, 0.3])
    backend = TuringBRMI((@brm begin
        mu ~ 1 + x
        shifted = mu + 0.25
        y ~ juliafam(shifted, 1)
    end)(data))
    @test Turing.loglikelihood(backend.model, (; beta_pop=[0.1, -0.2])) ≈
        sum(logpdf.(Normal.(data.x .* -0.2 .+ 0.1 .+ 0.25, 1), data.y))
end
