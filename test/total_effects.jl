using Test, Random, LinearAlgebra, Distributions
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
const BRM = BayesianRegressionModels
using BayesianRegressionModels: brm_total, brm_total_recover_rng

# `W` holds each group's value of every absorbed coefficient's column: all ones
# for coefficients matched to a random column, per-group values for
# group-level predictors (here two, folded into the intercept and slope).
@testset "Exact total-coefficient marginal density" begin
    rng = Xoshiro(3487)
    J, K = 5, 2
    tau = [0.8, 1.3]
    shapes = (
        (A=[1.0 -2.4; 0.0 1.0], W=ones(J,2), location=[2.1, -0.5],
         precisions=([0.25, 0.7], [0.25, 0.0])),
        (A=[1.0 -2.4 1.0 0.0; 0.0 1.0 0.0 1.0],
         W=hcat(ones(J,2), [0.4, -1.3, 0.0, 2.2, 0.9], [1.0, 0.0, 0.0, 1.0, 1.0]),
         location=[2.1, -0.5, 0.3, -0.2],
         precisions=([0.25, 0.7, 1.1, 0.4], [0.25, 0.0, 1.1, 0.0])))
    for (; A, W, location, precisions) in shapes, precision in precisions
        P = size(A,2)
        model = StanBlocks.@slic (; J, K, A, W, location, precision, y=0.3) begin
            log_tau::vector[K] ~ std_normal()
            log_mixture ~ std_normal()
            tau = exp(log_tau)
            conditional_precision = exp(log_mixture) * precision
            total::matrix[J,K] ~ brm_total(tau, A, W, location, conditional_precision)
            y ~ normal(total[1,1], 1.)
            beta = brm_total_recover_rng(total, tau, A, W, location, conditional_precision)
        end
        problem = StanBlocks.stan_instantiate(model;
            path=joinpath(mktempdir(), "totals.stan"))
        @test LogDensityProblems.dimension(problem) == J*K+K+1
        # Joint density at an arbitrary beta minus beta's exact conditional
        # density is the marginal, whatever beta is; the trials below check
        # that independence, which only the true conditional moments give.
        function reference(x; shift=0.2)
            totals = reshape(x[K+2:end], J, K)
            D = Diagonal(exp.(2x[1:K]))
            Pr = precision .* exp(x[K+1])
            Q = sum((A[c,:]*A[c,:]') .* (W'W) ./ D[c,c] for c in 1:K) + Diagonal(Pr)
            V = inv(Symmetric(Q))
            h = sum(A[c,:] .* (W'totals)[:,c] ./ D[c,c] for c in 1:K) + Pr.*location
            b = V*h .+ shift .* (-1) .^ (1:P)
            joint = sum(logpdf(MvNormal(A*(W[g,:] .* b),D), totals[g,:]) for g in 1:J)
            joint += sum(Pr[i] > 0 ? logpdf(Normal(location[i], inv(sqrt(Pr[i]))), b[i]) : 0.0 for i in 1:P)
            joint - logpdf(MvNormal(V*h,V),b) + logpdf(Normal(totals[1,1],1),0.3) + sum(logpdf.(Normal(),x[1:K+1]))
        end
        for trial in 1:6
            x = vcat(log.(tau) .+ 0.2randn(rng,K), 0.2randn(rng), randn(rng,J*K) .* repeat(tau; inner=J))
            lp, grad = BridgeStan.log_density_gradient(problem.model,x; propto=false)
            @test lp ≈ reference(x) atol=2e-12 rtol=2e-12
            @test reference(x) ≈ reference(x; shift=-0.7) atol=1e-10 rtol=1e-10
            fd = map(eachindex(x)) do i
                delta = zeros(length(x)); delta[i] = 1e-5
                (reference(x+delta)-reference(x-delta))/2e-5
            end
            @test grad ≈ fd atol=3e-8 rtol=3e-8
            # Groups are exchangeable only without group-level predictors:
            # otherwise each group's W row moves with it.
            allequal(eachrow(W)) || continue
            permutation = [3,5,1,4,2]
            reordered = vcat(x[1:K+1],vec(reshape(x[K+2:end],J,K)[permutation,:]))
            lp_without_y = lp - logpdf(Normal(x[K+2],1),0.3)
            permuted_lp = BridgeStan.log_density(problem.model,reordered; propto=false) - logpdf(Normal(reordered[K+2],1),0.3)
            @test lp_without_y ≈ permuted_lp atol=2e-12 rtol=2e-12
        end
    end
end
