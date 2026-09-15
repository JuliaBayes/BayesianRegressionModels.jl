using Distributions, LinearAlgebra, Random, Statistics, Test
using LogDensityProblems
include("model.jl")
using .PupilTotalEffects
const PTE = PupilTotalEffects

# Independent Gaussian integration: divide the full centered joint density by
# its two-dimensional conditional population-coefficient density at fixed beta.
# This is an audit identity, not reconstruction of fitted original parameters.
function reference_density(p, q)
    d = p.data
    J = length(d.ids)
    A, B = q[5:4+J], q[5+J:4+2J]
    tau = exp.(q[1:2])
    D = Diagonal(tau.^2)
    T = [1.0 -d.xbar; 0.0 1.0]
    precision = Diagonal([inv(PTE.INTERCEPT_SD^2), 0.0]) + J*T'*(D\T)
    covariance = inv(Symmetric(precision))
    natural = [PTE.INTERCEPT_MEAN/PTE.INTERCEPT_SD^2, 0.0] + T'*(D\[sum(A),sum(B)])
    conditional_mean = covariance * natural
    beta = [5700.0, 40.0]
    lp = logpdf(Normal(PTE.INTERCEPT_MEAN, PTE.INTERCEPT_SD), beta[1])
    for j in 1:J
        lp += logpdf(MvNormal(T*beta, D), [A[j], B[j]])
    end
    lp -= logpdf(MvNormal(conditional_mean, covariance), beta)
    for t in tau
        lp += logpdf(truncated(LocationScale(0.0, PTE.GROUP_SCALE, TDist(3)), 0, Inf), t) + log(t)
    end
    lp += logpdf(LocationScale(0.0, PTE.LOG_SIGMA_SCALE, TDist(3)), q[3])
    for i in eachindex(d.y)
        j = d.group[i]
        sigma = exp(q[3] + q[4]*(d.ids[j]-d.idbar))
        lp += logpdf(Normal(A[j]+B[j]*d.x[i], sigma), d.y[i])
    end
    lp
end

function numerical_gradient(f, q)
    map(eachindex(q)) do j
        h = 1e-5 * max(1.0, abs(q[j]))
        p2, p1, m1, m2 = copy(q), copy(q), copy(q), copy(q)
        p2[j] += 2h; p1[j] += h; m1[j] -= h; m2[j] -= 2h
        (-f(p2)+8f(p1)-8f(m1)+f(m2))/(12h)
    end
end

function audit_model(p)
    rng = Xoshiro(9241)
    initial = PTE.initial_position(p.data)
    rows = NamedTuple[]
    @testset "Pupil induced target and independent gradients" begin
        @test LogDensityProblems.dimension(p) == 44
        for k in 1:12
            q = copy(initial)
            q[1:2] .+= 0.7randn(rng, 2)
            q[3] += 0.1randn(rng)
            q[4] = 0.01randn(rng)
            q[5:24] .+= 80randn(rng, 20)
            q[25:44] .+= 8randn(rng, 20)
            value, g = PTE.evaluate(p, q)
            reference = reference_density(p, q)
            fd = numerical_gradient(x -> reference_density(p,x), q)
            density_error = abs(value-reference)
            gradient_error = maximum(abs.(g-fd) ./ (1 .+ abs.(g)))
            @test density_error < 1e-7
            @test gradient_error < 2e-5
            controls = rand(rng, 40)
            source = PTE.to_source(q, controls, p.data)
            @test PTE.to_model(source, controls, p.data) ≈ q
            push!(rows, (; point=k, density_error, gradient_error))
        end
    end
    rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    rows = audit_model(PTE.PupilProblem())
    println("maximum_density_error=", maximum(r.density_error for r in rows))
    println("maximum_relative_gradient_error=", maximum(r.gradient_error for r in rows))
end
