using Test, Random, LinearAlgebra, Distributions
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
const BRM = BayesianRegressionModels
using BayesianRegressionModels: brm_total, brm_total_recover_rng

@testset "Exact total-coefficient marginal density" begin
    rng = Xoshiro(3487)
    J, K = 5, 2
    tau = [0.8, 1.3]
    A = [1.0 -2.4; 0.0 1.0]
    location = [2.1, -0.5]
    for precision in ([0.25, 0.7], [0.25, 0.0])
        model = StanBlocks.@slic (; J, K, A, location, precision, y=0.3) begin
            log_tau::vector[K] ~ std_normal()
            log_mixture ~ std_normal()
            tau = exp(log_tau)
            conditional_precision = exp(log_mixture) * precision
            total::matrix[J,K] ~ brm_total(tau, A, location, conditional_precision)
            y ~ normal(total[1,1], 1.)
            beta = brm_total_recover_rng(total, tau, A, location, conditional_precision)
        end
        problem = StanBlocks.stan_instantiate(model;
            path=joinpath(mktempdir(), "totals.stan"))
        @test LogDensityProblems.dimension(problem) == J*K+K+1
        function reference(x)
            totals = reshape(x[K+2:end], J, K)
            D = Diagonal(exp.(2x[1:K]))
            P = precision .* exp(x[K+1])
            Q = J*A'*(D\A) + Diagonal(P)
            V = inv(Symmetric(Q))
            h = A'*(D\vec(sum(totals; dims=1))) + P.*location
            b = V*h .+ [0.2,-0.1]
            joint = sum(logpdf(MvNormal(A*b,D), collect(row)) for row in eachrow(totals))
            joint += sum(P[i] > 0 ? logpdf(Normal(location[i], inv(sqrt(P[i]))), b[i]) : 0.0 for i in 1:K)
            joint - logpdf(MvNormal(V*h,V),b) + logpdf(Normal(totals[1,1],1),0.3) + sum(logpdf.(Normal(),x[1:K+1]))
        end
        for trial in 1:6
            x = vcat(log.(tau) .+ 0.2randn(rng,K), 0.2randn(rng), randn(rng,J*K) .* repeat(tau; inner=J))
            lp, grad = BridgeStan.log_density_gradient(problem.model,x; propto=false)
            @test lp ≈ reference(x) atol=2e-12 rtol=2e-12
            fd = map(eachindex(x)) do i
                delta = zeros(length(x)); delta[i] = 1e-5
                (reference(x+delta)-reference(x-delta))/2e-5
            end
            @test grad ≈ fd atol=3e-8 rtol=3e-8
            permutation = [3,5,1,4,2]
            reordered = vcat(x[1:K+1],vec(reshape(x[K+2:end],J,K)[permutation,:]))
            lp_without_y = lp - logpdf(Normal(x[K+2],1),0.3)
            permuted_lp = BridgeStan.log_density(problem.model,reordered; propto=false) - logpdf(Normal(reordered[K+2],1),0.3)
            @test lp_without_y ≈ permuted_lp atol=2e-12 rtol=2e-12
        end
    end
end
