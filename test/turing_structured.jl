using Test
using BayesianRegressionModels
using Distributions
using Turing
using LinearAlgebra
using Random

const BRM = BayesianRegressionModels

@testset "shared structured-field normalization and grouping" begin
    legacy = (; n_per_group=2, group_arg_pos=1)
    fields = BRM._brm_structured_fields(legacy, BRM.sb_group_demo)
    @test only(fields).name === :sb_group_demo
    @test only(fields).prior === :correlated_normal
    @test only(fields).group == (; arg_pos=1)

    data = (; g=["a", "b", "a"], x=[1.0, 2.0, 3.0])
    brmi = @brm data begin
        mu ~ sb_group_demo(g)
        x ~ Normal(mu, 1)
    end
    call = BRM.getargs(BRM.parent(brmi.operations.mu), 2)[2]
    column = BRM._brm_structured_group_column(
        only(fields).group, call, Dict(:g => data.g))
    @test BRM.name(column) === :g
    @test BRM._brm_structured_group_name(only(fields).group, call) === :g
    sb = SBBRMI(brmi; mod=@__MODULE__)
    @test haskey(sb.data, :n_terms_sb_group_demo_g)
    @test haskey(sb.data, :g_idx)
end

@testset "typed structured field priors preserve bounds and emission" begin
    typed = BRM.ExprColumn(Exponential, 2.0)
    field = (; name=:typed, n_per_group=1, source=:g,
             levels=["a", "b"], idx=[1, 2], prior=typed)
    @test BRM._brm_structured_prior_expression(field) === typed
    statements = Any[]
    BRM._sb_emit_block_draw!(
        statements, typed, :b_typed_g, :g_idx, :n_g, :n_terms_typed_g,
        :typed_g)
    @test occursin("exponential", lowercase(string(first(statements))))

    ext = Base.get_extension(BRM, :BayesianRegressionModelsTuringExt)
    bounded = merge(field,
        (; prior=(; dist=Normal, args=(0.0, 1.0), lower=0.0)))
    model = ext._brm_iid_structured_field(bounded, Normal())
    @test isfinite(Turing.logjoint(model, (; values=[0.2, 0.4])))
    @test Turing.logjoint(model, (; values=[-0.2, 0.4])) == -Inf
end


ragged_mvnormal(xs, location) =
    MvNormal([location, location + sum(xs)], [0.4 0.0; 0.0 0.4])

@testset "ordinary callable consumes ragged vector rows" begin
    data = (; x=[-1.0, -0.2, 0.5, 1.0],
            pieces=[[0.1], [0.2, -0.1], Float64[], [0.3, 0.2, -0.2]],
            y=[[0.0, 0.1], [-0.2, 0.0], [0.3, 0.4], [0.5, 0.8]])
    backend = TuringBRMI((@brm begin
        location ~ 1 + x
        y ~ ragged_mvnormal(pieces, location)
    end)(data))
    parameters = (; beta_pop=[0.15, -0.25])
    locations = backend.plan.design.matrix * parameters.beta_pop
    expected = sum(logpdf.(Normal(), parameters.beta_pop)) + sum(eachindex(locations)) do i
        logpdf(ragged_mvnormal(data.pieces[i], locations[i]), data.y[i])
    end
    @test Turing.logjoint(backend.model, parameters) ≈ expected
    @test backend.plan.distribution.callable === ragged_mvnormal
end

@testset "native structured demo terms and frozen groups" begin
    data = (; g=["a", "b", "a", "b"], y=zeros(4))
    correlated_builder = @brm begin
        mu ~ 1 + sb_group_demo(g)
        y ~ Normal(mu, 1)
    end
    clamped_builder = @brm begin
        mu ~ 1 + sb_group_clamped_demo(g)
        y ~ Normal(mu, 1)
    end
    builders = (correlated_builder, clamped_builder)
    backends = map(builder -> TuringBRMI(builder(data)), builders)
    @test all(backend -> length(only(backend.plan.predictors).terms) == 1,
              backends)
    @test all(backend -> occursin("term_mu_1",
        string(BRM.turing_model_source(backend))), backends)
    @test all(backend -> !isempty(keys(rand(MersenneTwister(11), backend.model).data)),
              backends)

    fields = map(backends) do backend
        only(only(only(backend.plan.predictors).terms).state.fields)
    end
    ext = Base.get_extension(BRM, :BayesianRegressionModelsTuringExt)
    iid_model = ext._brm_iid_structured_field(fields[2], Exponential(1))
    iid_values = (; values=[0.3, 0.5, 0.7, 0.9])
    iid_result = Turing.generated_quantities(iid_model, iid_values)
    @test iid_result.block == [0.3 0.5; 0.7 0.9]
    @test Turing.logjoint(iid_model, iid_values) ≈
        sum(logpdf.(Exponential(1), iid_values.values))

    corr_priors = (Normal(), LKJCholesky(2, 1.0))
    corr_model = ext._brm_correlated_structured_field(fields[1], corr_priors)
    corr_values = (; tau=[0.6, 0.9], L=cholesky(Symmetric(Matrix{Float64}(I, 2, 2))),
                   z=[-0.2, 0.1, 0.3, -0.4])
    corr_result = Turing.generated_quantities(corr_model, corr_values)
    @test corr_result.block ≈ [-0.12 0.09; 0.18 -0.36]
    @test Turing.logjoint(corr_model, corr_values) ≈
        sum(logpdf.(Normal(), corr_values.tau)) +
        logpdf(corr_priors[2], corr_values.L) +
        sum(logpdf.(Normal(), corr_values.z))

    replay = reprocess(backends[2], (; g=["b", "a", "b"], y=zeros(3)))
    replay_field = only(only(only(replay.plan.predictors).terms).state.fields)
    @test replay_field.levels == fields[2].levels
    @test replay_field.idx == [2, 1, 2]
    @test_throws "not a training level" reprocess(
        backends[2], (; g=["a", "new"], y=zeros(2)))
end

function native_missing_structured end

@testset "SLIC-only structured terms fail explicitly in native execution" begin
    term = BRM._BRMPreparedTerm(
        native_missing_structured, (:g,), (;), (:g,))
    @test_throws "has no native Julia implementation" BRM._brm_native_structured_effect(
        term, zeros(1, 1), (; idx=[1]))
end
