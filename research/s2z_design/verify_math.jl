using LinearAlgebra, Random, Statistics, Test, Printf, TOML

# Independent finite-dimensional checks of the proposed mathematics. This file
# does not exercise a BRM S2Z implementation: no such integration exists yet.
function helmert(J)
    Q = zeros(J, J - 1)
    for k in 1:J-1
        Q[1:k, k] .= 1 / sqrt(k * (k + 1))
        Q[k + 1, k] = -k / sqrt(k * (k + 1))
    end
    Q
end

function partial_matrices(L, c)
    K, J = size(c)
    [begin
        A = zeros(K, K)
        for k in 1:K
            A[k,k] = L[k,k]^c[k,j]
            for l in 1:k-1
                A[k,l] = c[k,j] * L[k,l]
            end
        end
        A
    end for j in 1:J]
end

function forward(v, L, c, Q)
    K, J = size(c)
    u = reshape(kron(Q, Matrix{Float64}(I, K, K)) * v, K, J)
    A = partial_matrices(L, c)
    delta = hcat((L * (A[j] \ u[:,j]) for j in 1:J)...)
    delta .-= mean(delta; dims=2)
    delta
end

function inverse(delta, L, c, Q)
    K, J = size(c)
    A = partial_matrices(L, c)
    B = [A[j] / L for j in 1:J]
    shift = sum(B) \ sum(B[j] * delta[:,j] for j in 1:J)
    u = hcat((B[j] * (delta[:,j] - shift) for j in 1:J)...)
    kron(Q, Matrix{Float64}(I, K, K))' * vec(u)
end

function restricted_logjac(L, c)
    K, J = size(c)
    result = 0.0
    for k in 1:K
        h = log(L[k,k])
        a = c[k,:] .* h
        shift = maximum(a)
        result += (J-1)*h - sum(a) + shift + log(mean(exp.(a .- shift)))
    end
    result
end

function fd_jacobian(f, x; step=1e-5)
    hcat((begin
        h = step * max(1.0, abs(x[i]))
        xp, xm = copy(x), copy(x)
        xp[i] += h
        xm[i] -= h
        (f(xp) - f(xm)) / (2h)
    end for i in eachindex(x))...)
end
fd_gradient(f, x) = vec(fd_jacobian(z -> [f(z)], x))

function lognormal(x, m, V)
    z = x - m
    -length(x)/2*log(2pi) - logdet(Symmetric(V))/2 - dot(z, V \ z)/2
end

rng = MersenneTwister(582309)
metrics = Dict{String,Any}("seed" => 582309, "scope" => "math only; no BRM implementation or sampling benchmark")
max_inverse = 0.0
max_logjac = 0.0
max_full_jac = 0.0
cases = 0

@testset "Heterogeneous S2Z transform, inverse, restricted Jacobian" begin
    for J in (2, 3, 7), K in (1, 2, 3), trial in 1:4
        Q = helmert(J)
        H = kron(Q, Matrix{Float64}(I, K, K))
        @test Q'Q ≈ I
        @test Q'ones(J) ≈ zeros(J-1) atol=1e-14
        L = tril(0.3randn(rng, K, K))
        for k in 1:K
            L[k,k] = exp(randn(rng) * 1.2)
        end
        c = rand(rng, K, J)
        v = randn(rng, K*(J-1))
        delta = forward(v, L, c, Q)
        vinv = inverse(delta, L, c, Q)
        err = maximum(abs.(vinv-v))
        global max_inverse = max(max_inverse, err)
        @test err < 2e-10
        @test maximum(abs.(sum(delta; dims=2))) < 2e-12
        A = partial_matrices(L, c)
        T = zeros(J*K, J*K)
        for j in 1:J
            idx = (j-1)*K+1:j*K
            T[idx,idx] = L / A[j]
        end
        M = H'T*H
        dense_lj, sign_lj = logabsdet(M)
        errj = abs(dense_lj - restricted_logjac(L,c))
        global max_logjac = max(max_logjac, errj)
        @test sign_lj == 1
        @test errj < 2e-11
        @test H'vec(delta) ≈ M*v atol=2e-12
        for endpoint in (0.0, 1.0)
            ce = fill(endpoint, K, J)
            de = forward(v, L, ce, Q)
            expected = endpoint == 0 ? L * reshape(H*v,K,J) : reshape(H*v,K,J)
            @test de ≈ expected atol=2e-12
            @test restricted_logjac(L, ce) ≈ (1-endpoint)*(J-1)*sum(log,diag(L)) atol=2e-12
        end
        # Full map keeps log-diagonal parameters. Hyperparameter derivatives
        # occupy the off-diagonal Jacobian block and must not alter its determinant.
        x = vcat(v, log.(diag(L)))
        function full_map(x)
            LL = copy(L)
            for k in 1:K
                LL[k,k] = exp(x[length(v)+k])
            end
            vcat(H'vec(forward(x[1:length(v)],LL,c,Q)), x[length(v)+1:end])
        end
        full_lj = first(logabsdet(fd_jacobian(full_map, x)))
        errf = abs(full_lj - restricted_logjac(L,c))
        global max_full_jac = max(max_full_jac, errf)
        @test errf < 1e-7
        global cases += 1
    end
end
metrics["transform_cases"] = cases
metrics["max_inverse_absolute_error"] = max_inverse
metrics["max_restricted_logjac_absolute_error"] = max_logjac
metrics["max_full_finite_difference_logjac_absolute_error"] = max_full_jac

max_gradient = 0.0
@testset "Scalar transformed target: all coordinate and scale gradients" begin
    for J in (2, 4, 9), logtau in (-4.0, 0.0, 3.0)
        Q = helmert(J)
        c = rand(rng, J)
        y = randn(rng, J)
        mu, prior_variance, noise_variance = 0.4, 1.3, 0.8
        v = randn(rng, J-1)
        x = vcat(v, logtau, 0.7)
        function lp_and_grad(x)
            v, h, alpha = x[1:J-1], x[J], x[J+1]
            tau = exp(h)
            u = Q*v
            t = exp.((1 .- c).*h)
            e = Q'*(t.*u)
            delta = Q*e
            r = y .- alpha .- delta
            S = tau^2/J
            V = prior_variance + S
            lj = (J-1)*h - sum(c)*h + log(mean(exp.(c.*h)))
            lp = -sum(abs2,r)/(2noise_variance) - sum(abs2,e)/(2tau^2) - (J-1)*h - (alpha-mu)^2/(2V) - log(V)/2 - h^2/2 + lj
            ge = Q'r/noise_variance - e/tau^2
            gv = (Q'*Diagonal(t)*Q)'*ge
            gh = sum(abs2,e)/tau^2 - (J-1) + S*((alpha-mu)^2/V^2 - 1/V) - h
            gh += dot(ge, Q'*((1 .-c).*t.*u))
            gh += (J-1) - sum(c) + dot(c, exp.(c.*h))/sum(exp.(c.*h))
            ga = sum(r)/noise_variance - (alpha-mu)/V
            lp, vcat(gv,gh,ga)
        end
        _, g = lp_and_grad(x)
        fd = fd_gradient(z -> first(lp_and_grad(z)),x)
        err = maximum(abs.(g-fd)./max.(1.0,abs.(g)))
        global max_gradient = max(max_gradient,err)
        @test err < 2e-7
    end
end
metrics["max_transformed_gradient_scaled_error"] = max_gradient

max_density = 0.0
max_covariance = 0.0
@testset "Joint omitted means: exact Gaussian marginalization and recovery" begin
    for p in (1, 3), d in (1, 2, 5), trial in 1:4
        A = randn(rng,p,d)
        R = randn(rng,p,p)
        V = R*R' + I
        T = randn(rng,d,d)
        S = (T*T' + I)/5
        C = V + A*S*A'
        mu, alpha, m = randn(rng,p), randn(rng,p), randn(rng,d)
        gain = S*A'/C
        mc = gain*(alpha-mu)
        Sc = S - gain*A*S
        left = lognormal(alpha-A*m,mu,V) + lognormal(m,zeros(d),S) - lognormal(alpha,mu,C)
        right = lognormal(m,mc,Sc)
        err = abs(left-right)
        global max_density = max(max_density,err)
        @test err < 2e-10
        @test Sc ≈ inv(inv(S) + A'*inv(V)*A) atol=2e-12
        # Recovery must reproduce original independent beta and m priors.
        cov_am = C*gain'
        cov_m = gain*C*gain' + Sc
        cov_beta = C + A*cov_m*A' - cov_am*A' - A*cov_am'
        errc = maximum(abs.(cov_beta - V))
        global max_covariance = max(max_covariance,errc)
        @test errc < 2e-11
        @test cov_m ≈ S atol=2e-12
        @test cov_am - A*cov_m ≈ zeros(p,d) atol=2e-12
    end
    for J in (2, 7), K in (1, 3)
        Q = helmert(J)
        R = randn(rng,K,K)
        Sigma = R*R' + I
        # No sqrt(J/(J-1)) adjustment: contrasts + omitted mean recover iid b.
        @test kron(Q*Q',Sigma) + kron(ones(J,J)/J,Sigma) ≈ kron(Matrix{Float64}(I,J,J),Sigma) atol=2e-12
    end
end
metrics["max_marginalization_logdensity_absolute_error"] = max_density
metrics["max_recovered_covariance_absolute_error"] = max_covariance
open(joinpath(@__DIR__, "math_receipt.toml"), "w") do io
    TOML.print(io,metrics; sorted=true)
end
println("MATH_CHECKS_COMPLETE")
for k in sort(collect(keys(metrics)))
    println(k, " = ", metrics[k])
end
