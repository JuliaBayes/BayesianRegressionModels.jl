# test/correlated_outcomes.jl — correlated row-wise multivariate responses.
#
# Run on a capable host:
#   julia --startup-file=no --project=test test/correlated_outcomes.jl

using Test
using BayesianRegressionModels
using Distributions: Exponential, Normal
using LogDensityProblems
using StanBlocks

const JOINT_DATA_KEY = :brm_joint_y1__y2_observed
const JOINT_N_KEY = :brm_joint_y1__y2_n

joint_df = (;
    x=[-1.0, 0.0, 1.0],
    y1=[0.1, 0.2, -0.1],
    y2=[1.1, 0.9, 1.2],
)

joint_builder = @brm begin
    L_res ~ LKJCovarianceFactor(
        2; scale_prior=Exponential(1), shape=2,
    )
    mu1 ~ 1 + x
    mu2 ~ 1 + x
    [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
end

@testset "joint response parses as one ordered likelihood" begin
    brmi = joint_builder(joint_df)
    outcome = only(outcomes(brmi))

    @test outcome.response == (:y1, :y2)
    @test outcome.family === MvNormalCholesky
    @test outcome.args == [
        (; role=:joint_means, names=(:mu1, :mu2)),
        (; role=:covariance_factor, name=:L_res),
    ]
    @test Set(data_columns(brmi)) == Set((:x, :y1, :y2))
    @test Set(dependencies(brmi, :y1).data) == Set((:x,))
    @test Set(dependencies(brmi, :y1).intermediates) ==
          Set((:mu1, :mu2, :L_res))
    @test dependencies(brmi, :y1) == dependencies(brmi, :y2)
    @test occursin("[y1, y2] ~ MvNormalCholesky", sprint(show, brmi))

    @test_throws "at least two outcome columns" eval(quote
        @brm begin
            [y1] ~ MvNormalCholesky([mu1], L_res)
        end
    end)
    @test_throws "must be a bare data-column" eval(quote
        @brm begin
            [log(y1), y2] ~ MvNormalCholesky([mu1, mu2], L_res)
        end
    end)
    @test_throws "must be unique and ordered" eval(quote
        @brm begin
            [y1, y1] ~ MvNormalCholesky([mu1, mu1], L_res)
        end
    end)
    @test_throws "collides with a generated joint-response binding" eval(quote
        @brm begin
            [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
            brm_joint_y1__y2 ~ 1
        end
    end)
    @test_throws "collides with a generated joint-response binding" eval(quote
        @brm begin
            [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
            brm_joint_y1__y2_means ~ 1
        end
    end)
end

@testset "StanBlocks lowering preserves outcome and row axes" begin
    brmi = joint_builder(joint_df)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)

    @test sb.data[JOINT_DATA_KEY] ==
          [[0.1, 1.1], [0.2, 0.9], [-0.1, 1.2]]
    @test length.(sb.data[JOINT_DATA_KEY]) == fill(2, 3)
    @test sb.data[JOINT_N_KEY] == 3
    @test sb.preproc[JOINT_DATA_KEY].kind === :joint_response
    @test sb.preproc[JOINT_DATA_KEY].raw_ref == (:y1, :y2)
    @test sb.preproc[JOINT_DATA_KEY].const_.mean_sources == (:x,)
    @test occursin("plate(", sprint(show, sb))

    @test occursin("brm_joint_y1__y2_observed_mem", code)
    @test occursin("brm_joint_y1__y2_observed_ends", code)
    @test occursin("L_res_L_corr ~ lkj_corr_cholesky(2.0);", code)
    @test occursin("L_res_scales ~ exponential(1.0);", code)
    @test occursin("L_res = diag_pre_multiply(L_res_scales, L_res_L_corr);", code)
    @test occursin("multi_normal_cholesky", code)
    @test occursin("getindex_RaggedVector(brm_joint_y1__y2_observed", code)
    @test occursin("brm_joint_y1__y2_observed_gen", code)

    checked = StanBlocks.stanc_check(code; warn_pedantic=false)
    checked.ok || @error "stanc rejected correlated-outcome model" output=checked.output
    @test checked.ok

    plan = generative_plan(sb)
    observation = only(d for d in plan.declarations if d.role === :observation)
    @test observation.data_source === JOINT_DATA_KEY
    @test observation.context == ()
    @test endswith(string(observation.draw), "brm_joint_y1__y2_observed_gen")

    descriptor = brm_descriptor(sb; highlights=())
    @test Set(descriptor.columns) == Set((:x, :y1, :y2))
    byname = Dict(output.name => output for output in descriptor.outputs)
    @test byname[:L_res].role === :parameter
    @test byname[:brm_joint_y1__y2_observed_gen].role === :posterior_predictive
    @test byname[:brm_joint_y1__y2_observed_gen].logical === JOINT_DATA_KEY
    @test byname[:brm_joint_y1__y2_observed_gen].segments == [2, 4, 6]

end

@testset "intercept-only joint means borrow the joint observation axis" begin
    two_axis = @brm begin
        sigma ~ Exponential(1)
        other_mu ~ 1 + z
        other ~ Normal(other_mu, sigma)

        L_res ~ LKJCovarianceFactor(2)
        mu1 ~ 1
        mu2 ~ 1
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end
    data = merge(joint_df, (;
        z=[-1.0, -0.5, 0.0, 0.5, 1.0],
        other=zeros(5),
    ))
    code = BayesianRegressionModels.stan_code(
        SBBRMI(two_axis(data); mod=@__MODULE__))

    @test occursin("vector[brm_joint_y1__y2_n] pop_mu1", code)
    @test occursin("vector[brm_joint_y1__y2_n] pop_mu2", code)
    @test occursin("vector[z_n] other_mu =", code)
end

@testset "linked mechanistic means may share sampled parameters" begin
    linked = @brm joint_df begin
        shared_log_rate ~ Normal(0, 1)
        L_res ~ LKJCovarianceFactor(2)
        concentration_mu = exp(-exp(shared_log_rate) * x)
        effect_mu = 1.0 - concentration_mu
        [y1, y2] ~ MvNormalCholesky(
            [concentration_mu, effect_mu], L_res)
    end
    linked_sb = SBBRMI(linked; mod=@__MODULE__)
    linked_code = BayesianRegressionModels.stan_code(linked_sb)

    @test occursin("shared_log_rate ~ normal(0, 1);", linked_code)
    @test occursin(
        "vector[x_n] concentration_mu = exp(((-exp(shared_log_rate)) .* x));",
        linked_code)
    @test occursin(
        "vector[x_n] effect_mu = (1.0 - concentration_mu);", linked_code)
    @test StanBlocks.stanc_check(linked_code; warn_pedantic=false).ok
end

@testset "joint response replay rebuilds complete aligned row vectors" begin
    sb = SBBRMI(joint_builder(joint_df); mod=@__MODULE__)
    new_df = (;
        x=[-0.5, 0.5],
        y1=[10.0, 20.0],
        y2=[30.0, 40.0],
    )
    replayed = reprocess(sb, new_df)

    @test replayed.data[JOINT_DATA_KEY] == [[10.0, 30.0], [20.0, 40.0]]
    @test replayed.data[JOINT_N_KEY] == 2
    @test BayesianRegressionModels.stan_code(replayed) ==
          BayesianRegressionModels.stan_code(sb)
    @test restan_data(sb, new_df)[JOINT_DATA_KEY] ==
          (mem=[10.0, 30.0, 20.0, 40.0], ends=[2, 4])
end

@testset "joint response validation is strict and backend boundaries are loud" begin
    @test_throws "contains `missing`" joint_builder((;
        x=[-1.0, 0.0, 1.0],
        y1=Union{Missing,Float64}[0.1, missing, -0.1],
        y2=joint_df.y2,
    )) |> SBBRMI
    @test_throws "equal lengths" joint_builder((;
        x=joint_df.x, y1=joint_df.y1, y2=joint_df.y2[1:2],
    )) |> SBBRMI
    @test_throws "non-finite" joint_builder((;
        x=joint_df.x, y1=[0.1, NaN, -0.1], y2=joint_df.y2,
    )) |> SBBRMI
    @test_throws "uses row-axis source `x` with 2 rows" joint_builder((;
        x=joint_df.x[1:2], y1=joint_df.y1, y2=joint_df.y2,
    )) |> SBBRMI

    wrong_dimension = @brm joint_df begin
        L_res ~ LKJCovarianceFactor(3; scale_prior=Exponential(1), shape=2)
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end
    @test_throws "has dimension 3" SBBRMI(wrong_dimension; mod=@__MODULE__)

    wrong_means = @brm joint_df begin
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1), shape=2)
        mu1 ~ 1 + x
        [y1, y2] ~ MvNormalCholesky([mu1], L_res)
    end
    @test_throws "received 1 means" SBBRMI(wrong_means; mod=@__MODULE__)

    bad_shape = @brm joint_df begin
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1), shape=0)
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end
    @test_throws "must be finite and strictly positive" SBBRMI(bad_shape; mod=@__MODULE__)

    factor_collision = @brm joint_df begin
        L_res ~ LKJCovarianceFactor(2)
        L_res_scales ~ 1 + x
        [y1, y2] ~ MvNormalCholesky([L_res_scales, L_res_scales], L_res)
    end
    @test_throws "reserves emitted binding `L_res_scales`" SBBRMI(
        factor_collision; mod=@__MODULE__)

    brmi = joint_builder(joint_df)
    @test_throws "supported by the StanBlocks backend only" VBRMI(brmi)
    @test BayesianRegressionModels._brm_turing_plan(brmi).response ==
          [[joint_df.y1[i], joint_df.y2[i]] for i in eachindex(joint_df.y1)]
    @test_throws "supported by the StanBlocks backend only" BayesianRegressionModels.NativePPL.lower(brmi)

    trained = SBBRMI(brmi; mod=@__MODULE__)
    @test_throws "mean source `x` has 1 rows" reprocess(trained, (;
        x=[0.0], y1=[0.1, 0.2], y2=[1.0, 1.1],
    ))
end

@testset "joint density exposes one pointwise contribution per row" begin
    sb = SBBRMI(joint_builder(joint_df); mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)
    descriptor = brm_descriptor(sb; highlights=())
    byname = Dict(output.name => output for output in descriptor.outputs)

    # The grouped observation is one multivariate density per aligned row, not
    # K independent scalar contributions and not one density for the full data.
    likelihood_name = :brm_joint_y1__y2_observed_likelihood
    likelihood_ready = occursin(string(likelihood_name), code) &&
        haskey(byname, likelihood_name) &&
        byname[likelihood_name].role === :pointwise_loglik &&
        (:pointwise_loglik in
            (operation.name for operation in descriptor.operations))
    @test likelihood_ready
end

@testset "joint density has a finite BridgeStan gradient and executable twins" begin
    sb = SBBRMI(joint_builder(joint_df); mod=@__MODULE__)
    descriptor = brm_descriptor(sb; highlights=())
    cache = joinpath(tempdir(), "brm-correlated-outcomes")
    isdir(cache) || mkpath(cache)
    problem = brm_execute(
        descriptor, :instantiate;
        path=joinpath(cache, string(descriptor.id) * ".stan"))
    dimension = LogDensityProblems.dimension(problem)
    q = [0.05 * ((i % 7) - 3) for i in 1:dimension]
    lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)

    @test dimension > 0
    @test isfinite(lp)
    @test length(gradient) == dimension
    @test all(isfinite, gradient)

    pointwise = brm_execute(
        descriptor, :pointwise_loglik;
        problem, draws=q, seed=20260820)
    likelihood = pointwise.brm_joint_y1__y2_observed_likelihood
    @test length(likelihood) == length(joint_df.y1)
    @test all(isfinite, likelihood)

    predictions = brm_execute(
        descriptor, :predict;
        problem, draws=q, seed=20260820)
    generated = predictions.brm_joint_y1__y2_observed_gen
    @test length(generated) == 2 * length(joint_df.y1)
    @test all(isfinite, generated)
end
