# Preparation/source checks can run before the producer authoring form lands.
# Runtime density/gradient/replay acceptance is in rk_bounded_runtime.jl.
using Test, BayesianRegressionModels, Distributions
include(joinpath(@__DIR__, "rk_bounded_fixtures.jl"))
const BRM = BayesianRegressionModels

function bounded_emission(builder, data=bounded_data)
    before = deepcopy(data)
    emitted = BRM._rk_emit_ast(BRM._brm_rk_plan(builder(data)))
    @test isequal(data, before)
    source = sprint(Base.show_unquoted, emitted.main)
    @test Meta.parse(source) isa Expr
    source
end

@testset "declaration support retains visible family kernels" begin
    @test occursin("scale ~ restricted(Normal(0.0, 1.0), 0.0, Inf)",
        bounded_emission(bounded_normal_builder))
    @test occursin("invdf ~ restricted(Exponential(0.125), 0.0, 0.5)",
        bounded_emission(bounded_df_builder))
    @test occursin("location ~ restricted(Normal(0.0, 1.0), -Inf, 0.5)",
        bounded_emission(bounded_upper_builder))
    @test occursin("value ~ restricted(StudentT(6.0, 0.0, 1.0), -0.5, 1.5)",
        bounded_emission(bounded_student_prior_builder))
    @test occursin("value ~ restricted(Beta(2.0, 3.0), -1.0, 0.8)",
        bounded_emission(bounded_beta_builder))
    @test_throws ErrorException BRM._brm_rk_plan(bounded_live_builder(bounded_data))
    for builder in (bounded_scalar_array_builder, bounded_scalar_df_builder)
        @test !occursin("restricted", bounded_emission(builder))
    end
    normalized = bounded_emission(bounded_normalized_builder)
    @test occursin("HalfNormal", normalized)
    @test !occursin("restricted", normalized)
end

@testset "observed scalar bounds and attributed rejection" begin
    for (lo, hi) in ((0.0, 1.5), (0.2, 2.0), (-Inf, Inf))
        data = merge(bounded_data, (; lower_bound=lo, upper_bound=hi))
        source = bounded_emission(bounded_data_builder, data)
        @test occursin("scale ~ restricted(Normal(0.0, 1.0), $lo, $hi)", source)
    end
    for (lo, hi) in ((0.5, 0.5), (1.0, 0.5), (NaN, 1.0),
            (Inf, Inf), (0.0, -Inf), ([0.0, 0.1], 1.0), (false, 1.0))
        data = merge(bounded_data, (; lower_bound=lo, upper_bound=hi))
        before = deepcopy(data)
        message = try
            BRM._brm_rk_plan(bounded_data_builder(data))
            ""
        catch error
            sprint(showerror, error)
        end
        @test occursin("parameter `scale`", message)
        @test isequal(data, before)
    end
    invalid = @brm bounded_data begin
        scale ~ Normal(0.0, 1.0; lower=0.0, unrelated=2.0)
        y ~ Normal(0.0, scale)
    end
    @test_throws ErrorException BRM._brm_rk_plan(invalid)
end
