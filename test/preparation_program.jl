using Test
using BayesianRegressionModels
using Distributions

const BRM = BayesianRegressionModels

@testset "shared source program preserves typed calls and declaration dependencies" begin
    df = (; x=[-1.0, 0.0, 1.0], y=[0.1, -0.2, 0.3])
    brmi = (@brm begin
        location ~ Normal(0, 1)
        spread ~ LogNormal(location, 0.3)
        mean ~ 1 + x
        shifted = mean + location
        effect(mean, x) ~ Cauchy(0, 2)
        y ~ Normal(shifted, spread)
    end)(df)
    program = BRM._brm_prepare_program(brmi)
    byname = Dict(op.name => op for op in program.operations)
    @test Set(keys(byname)) == Set(keys(brmi.operations))
    @test byname[:location].role === :parameter
    @test byname[:mean].role === :predictor
    @test byname[:shifted].role === :assignment
    @test byname[:y].role === :observation
    @test byname[:spread].dependencies == (:location,)
    @test Set(byname[:y].dependencies) == Set((:shifted, :spread))
    positions = Dict(name => i for (i, name) in pairs(program.order))
    @test positions[:location] < positions[:spread] < positions[:y]
    @test positions[:mean] < positions[:shifted] < positions[:y]
    @test program.context.data[:x] == df.x
    @test program.context.target_obs[:mean] === :y
    @test getf(last(getargs(byname[:spread].expression))) === LogNormal
end

@testset "resolved prior dependencies order their owning sample sites" begin
    brmi = (@brm begin
        mu ~ 1 + x
        hyper ~ Normal(0, 1)
        y ~ Normal(mu, 1)
    end)((; x=[-1.0, 0.0, 1.0], y=[0.1, 0.2, 0.3]))
    program = BRM._brm_prepare_program(brmi)
    prior = ExprColumn(Normal, brmi.operations.hyper, 1.0)
    prepared = BRM._brm_with_prior_dependencies(program, (:mu => (prior,),))
    @test findfirst(==(:hyper), prepared.order) < findfirst(==(:mu), prepared.order)
    @test prepared.context === program.context
    @test length(prepared.operations) == length(program.operations)
    self_prior = ExprColumn(Normal, brmi.operations.mu, 1.0)
    @test_throws "cyclic model declarations" BRM._brm_with_prior_dependencies(
        program, (:mu => (self_prior,),))
    @test BRM._brm_prior_value_dependencies(program, (:hyper,)) == (:hyper,)
    @test BRM._brm_prior_value_dependencies(program, (:x,)) == ()
    @test_throws "model-level prior references" BRM._brm_prior_value_dependencies(
        program, (:mu,))
    @test_throws "unknown model value" BRM._brm_prior_value_dependencies(
        program, (:absent,))
    refs = BRM._brm_operation_references!(Set{Symbol}(), Dict(:scale => (; prior)))
    @test refs == Set((:hyper,))
end

@testset "stable dependency order and cycle diagnostics" begin
    op(name, deps) = BRM._BRMPreparedOperation(name, :assignment, nothing, deps)
    @test BRM._brm_operation_order((op(:c, (:b,)), op(:a, ()), op(:b, (:a,)))) ==
          (:a, :b, :c)
    @test BRM._brm_operation_order((op(:a, ()), op(:b, ()))) == (:a, :b)
    @test_throws "cyclic model declarations" BRM._brm_operation_order((
        op(:a, (:b,)), op(:b, (:a,))))
end
