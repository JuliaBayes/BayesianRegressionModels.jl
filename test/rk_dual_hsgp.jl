# Scientific acceptance adopted from public RK test_brm_hsgp{,_reactant}.jl.
# The reference evaluates independent scalar distributions, not the graph.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Distributions, Statistics, SHA, Enzyme
using DifferentiationInterface: AutoEnzyme
const Prep = BayesianRegressionModels.StatisticalPreparation

function motorcycle_data()
    path = joinpath(@__DIR__, "..", "research", "adaptive_centering", "mcycle.csv")
    @test bytes2hex(sha256(read(path))) ==
        "b89a1e4eb0391a982b32be3e378df00e8593ff9971e9425e9c5d7929b74f9801"
    rows = split.(readlines(path)[2:end], ',')
    times = parse.(Float64, getindex.(rows, 2))
    accel = parse.(Float64, getindex.(rows, 3))
    lo, hi = extrema(times)
    (; x=@.(-1 + 2 * (times - lo) / (hi - lo)), y=accel ./ std(accel))
end

function motorcycle_reference(q, c, data)
    lp = sum(logpdf(Normal(0, 4), q[i]) for i in (1, 2, 23, 24))
    curves = [zeros(length(data.y)), zeros(length(data.y))]
    for (offset, coffset, curve) in ((0, 0, curves[1]), (22, 20, curves[2]))
        rho, sd = exp(q[offset + 1]), exp(q[offset + 2])
        for j in 1:20
            omega = j * pi / 3
            logs = log(sd) + log(sqrt(2pi) * rho) / 2 - rho^2 * omega^2 / 4
            v = q[offset + 2 + j]
            lp += logpdf(Normal(0, exp(c[coffset + j] * logs)), v)
            weight = v * exp((1 - c[coffset + j]) * logs)
            for i in eachindex(curve)
                curve[i] += sin(omega * (data.x[i] + 1.5)) / sqrt(1.5) * weight
            end
        end
    end
    lp + sum(logpdf(Normal(curves[1][i], exp(curves[2][i])), data.y[i])
             for i in eachindex(data.y))
end

@testset "BRM-owned exact dual HSGP" begin
    @test Base.get_extension(BayesianRegressionModels,
        :BayesianRegressionModelsReactiveKernelsExt) !== nothing
    data = motorcycle_data()
    saved = deepcopy(data)
    kernel = Prep.prepare_dual_hsgp(data)
    @test Tuple(p.name for p in inputs(kernel)) == (:q, :c)
    @test :plate in [e.kind for e in recipe_inventory(kernel)]
    backend = AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const)
    q = 0.04sin.(collect(1.0:44.0))
    q[[1, 23]] .= -2.0
    prepared = prepare_ad(kernel, backend, q, zeros(40); active=:q)
    for shift in (0.0, 0.09, -0.06)
        point = q .+ shift
        for c in (zeros(40), ones(40), collect(range(0, 1; length=40)),
                  repeat([0.0, 0.25, 0.6, 1.0], 10))
            before_q, before_c = copy(point), copy(c)
            value, gradient = ad_value_and_gradient!(prepared, similar(point), point, c)
            @test value ≈ motorcycle_reference(point, c, data) atol=2e-10 rtol=2e-12
            for j in eachindex(point)
                plus, minus = copy(point), copy(point)
                plus[j] += 1e-5; minus[j] -= 1e-5
                fd = (motorcycle_reference(plus, c, data) -
                    motorcycle_reference(minus, c, data)) / 2e-5
                @test gradient[j] ≈ fd atol=3e-7 rtol=2e-6
            end
            @test point == before_q && c == before_c
        end
    end
    basis_kernel = Prep.prepare_dual_hsgp(data; want=:basis)
    @test !occursin("sin", string(code_expr(basis_kernel)))
    @test size(basis_kernel(q, zeros(40))) == (133, 20)
    @test basis_kernel(q, zeros(40))[71, 13] ≈
        sin(13pi / 3 * (data.x[71] + 1.5)) / sqrt(1.5)
    @test data == saved
end
