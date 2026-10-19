# Per-row response hold-out: `held_out=(; y = mask)`.
#
#   julia --startup-file=no --project=test test/held_out_rows.jl

using Test
using BayesianRegressionModels, StanBlocks, Distributions
using LogDensityProblems
using ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels

rk_problem(rk) = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
pointwise(rk, u) = BRM._rk_pointwise_values(BRM._rk_pointwise_query(rk), u)

const flat = (; x=[1.0, 2.0, 0.5, 1.5, -0.3], y=[0.1, 0.2, 0.3, 0.5, -0.4],
    z=[0.0, 0.4, 0.2, 0.9, 0.1])
const late = [false, false, false, true, true]

const single = @brm begin
    mu ~ 1 + x
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
const joint = @brm begin
    mu ~ 1 + x
    sigma ~ Exponential(1)
    nu ~ 1 + x
    z ~ Normal(nu, 1.0)
    y ~ Normal(mu, sigma)
end

@testset "held_out row masks are validated against the response" begin
    brmi = single(flat)
    @test_throws r"length 3, but the response has 5" RKBRMI(brmi; held_out=(; y=late[1:3]))
    @test_throws r"must hold `Bool` values" RKBRMI(brmi; held_out=(; y=[0, 0, 0, 1, 1]))
    @test_throws r"unknown response" RKBRMI(brmi; held_out=(; w=late))
    @test_throws r"both as a whole response and with a row mask" RKBRMI(joint(flat);
        held_out=[:y, :y => late])
    # refused: every row of the only response is not a fit (held-out-and-prior-draws)
    @test_throws r"covers every observation" RKBRMI(brmi; held_out=(; y=trues(5)))
end

@testset "RKBRMI: held-out rows leave the fitted density, coordinates stay" begin
    for brmi in (single(flat), joint(flat))
        full = RKBRMI(brmi)
        masked = RKBRMI(brmi; held_out=(; y=late))
        @test occursin("withheld rows: y 2 of 5", sprint(show, masked))
        @test coordinate_names(masked.model.layout) == coordinate_names(full.model.layout)
        n = length(coordinate_names(full.model.layout))
        for u in (fill(0.1, n), collect(range(-0.4, 0.3; length=n)))
            rows = pointwise(full, u)
            held = rows.y[late]
            @test LogDensityProblems.logdensity(rk_problem(masked), u) ≈
                  LogDensityProblems.logdensity(rk_problem(full), u) - sum(held) atol=1e-10
            @test pointwise(masked, u).y[.!late] ≈ rows.y[.!late] atol=1e-12
            @test all(iszero, pointwise(masked, u).y[late])
        end
    end
    # An all-false mask holds nothing out; an all-true one holds the response out.
    @test sprint(show, RKBRMI(joint(flat); held_out=(; y=falses(5)))) ==
          sprint(show, RKBRMI(joint(flat)))
    @test sprint(show, RKBRMI(joint(flat); held_out=(; y=trues(5)))) ==
          sprint(show, RKBRMI(joint(flat); held_out=:y))
end

@testset "per-row hold-outs not built yet" begin
    # not built yet: SBBRMI needs StanBlocks to hold out single elements of an
    # observation (StanBlocks snag hold-out-a-subse-b48c943c)
    @test_broken (SBBRMI(single(flat); mod=@__MODULE__, held_out=(; y=late)); true)
    # not built yet: RKPPL does not skip `missing` entries inside per-subject arrays
    cell = (; x=[[1.0, 2.0], [0.5, 1.5, -0.4]], y=[[0.1, 0.2], [0.3, 0.5, -0.1]])
    incell = @brm begin
        s ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @plate for i in eachindex(x)
            m = s .* x[i]
            y[i] ~ Normal(m, sigma)
        end
    end
    @test_broken (RKBRMI(incell(cell); held_out=(; y=[[false, true], [false, false, true]])); true)
end
