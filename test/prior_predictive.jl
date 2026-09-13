# test/prior_predictive.jl — response-level likelihood hold-out / prior sampling.
#
# Run: julia --project=test test/prior_predictive.jl

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, LKJCholesky, Normal

joint_builder = @brm begin
    sigma_y ~ Exponential(1)
    sigma_z ~ Exponential(1)
    mu ~ 1 + x
    y ~ Normal(mu, sigma_y)
    z ~ Normal(mu, sigma_z)
end

joint_df = (;
    x=[-1.0, 0.0, 1.0, 2.0],
    y=[0.1, 0.8, 1.7, 2.6],
    z=[-0.2, 0.4, 1.2, 2.1],
)

operation_names(d) = Set(op.name for op in d.operations)
input_by_name(d) = Dict(input.name => input for input in d.inputs)

# The pre-feature workaround: reach into the emitted data dict, cv-mark the
# response by hand, and rebuild the wrapper in both model/data slots. The new
# public keyword must trace to the exact same program.
function manual_hold_out(sb, responses)
    marked = Dict{Symbol,Any}(sb.data)
    for response in responses
        marked[response] = StanBlocks.stan.maybecv(response, marked[response])
    end
    SBBRMI(
        parent(sb),
        StanBlocks.SlicModel(sb.model.model, marked, sb.model.mod),
        marked,
        sb.preproc,
    )
end

@testset "top-level response hold-out" begin
    brmi = joint_builder(joint_df)
    ordinary = SBBRMI(brmi; mod=@__MODULE__)
    explicit_default = SBBRMI(brmi; mod=@__MODULE__, held_out=())
    partial = SBBRMI(brmi; mod=@__MODULE__, held_out=:z)
    prior = SBBRMI(brmi; mod=@__MODULE__, held_out=:all)

    # The default remains byte-stable. Both public modes are exactly the
    # StanBlocks activity-analysis programs the manual workaround produced.
    @test BayesianRegressionModels.stan_code(explicit_default) ==
          BayesianRegressionModels.stan_code(ordinary)
    @test BayesianRegressionModels.stan_code(partial) ==
          BayesianRegressionModels.stan_code(manual_hold_out(ordinary, (:z,)))
    @test BayesianRegressionModels.stan_code(prior) ==
          BayesianRegressionModels.stan_code(manual_hold_out(ordinary, (:y, :z)))

    full_d = brm_descriptor(ordinary)
    partial_d = brm_descriptor(partial)
    prior_d = brm_descriptor(prior)

    full_inputs = input_by_name(full_d)
    partial_inputs = input_by_name(partial_d)
    prior_inputs = input_by_name(prior_d)
    @test !full_inputs[:y].held_out && !full_inputs[:z].held_out
    @test !partial_inputs[:y].held_out && partial_inputs[:z].held_out
    @test prior_inputs[:y].held_out && prior_inputs[:z].held_out

    @test :fit in operation_names(full_d)
    @test :fit in operation_names(partial_d)
    @test :prior_predictive ∉ operation_names(full_d)
    @test :prior_predictive ∉ operation_names(partial_d)
    @test :fit ∉ operation_names(prior_d)
    @test :prior_predictive in operation_names(prior_d)
    @test brm_operation(prior_d, :prior_predictive).origin === :stan
    @test brm_operation(prior_d, :prior_predictive).outputs ==
          Tuple(o.name for o in prior_d.outputs if o.kind === :parameter)

    # The semantic operation is executable, not presentation-only: it prepares
    # the same BridgeStan log-density problem a sampler consumes for a fit,
    # now with no observation likelihood in that density.
    prior_problem = brm_execute(prior_d, :prior_predictive)
    prior_dimension = StanBlocks.LogDensityProblems.dimension(prior_problem)
    @test prior_dimension > 0
    @test isfinite(StanBlocks.LogDensityProblems.logdensity(
        prior_problem, fill(0.1, prior_dimension)))

    # Both descriptor replay paths preserve the response selection.
    shifted = (; x=joint_df.x .+ 0.25, y=joint_df.y, z=joint_df.z)
    replayed = brm_descriptor(joint_builder, joint_df;
                              mod=@__MODULE__, held_out=:all)
    replayed = brm_execute(replayed, :replay, shifted)
    @test :prior_predictive in operation_names(replayed)
    reprocessed = brm_execute(partial_d, :reprocess, shifted)
    @test input_by_name(reprocessed)[:z].held_out
    @test !input_by_name(reprocessed)[:y].held_out

    @test_throws "unknown response" SBBRMI(
        brmi; mod=@__MODULE__, held_out=:missing_response)
    @test_throws "use `held_out=:all` by itself" SBBRMI(
        brmi; mod=@__MODULE__, held_out=(:all, :z))
end

# The motivating joint PK/QT shape puts both likelihoods inside a kernel cell.
# Its local aliases (`pk_obs`, `qt_obs`) are not the public dataframe response
# names, so this specifically guards the data-source resolution seam.
kernel_builder = @brm begin
    sigma_pk ~ Exponential(1)
    sigma_qt ~ Exponential(1)
    log_scale ~ 1 + (1 | p | subject)
    pred ~ kernel(dose, pk_y, qt_y, log_scale) do dd, pk_obs, qt_obs, ls
        location = (dd / 10.0) * exp(ls)
        pk_obs ~ normal(location, sigma_pk)
        qt_obs ~ normal(location, sigma_qt)
        location
    end
end

kernel_df = (;
    dose=[100.0, 100.0, 100.0],
    pk_y=[1.0, 1.5, 2.0],
    qt_y=[0.8, 1.2, 1.7],
    subject=[:a, :b, :c],
)

@testset "kernel-nested response hold-out" begin
    brmi = kernel_builder(kernel_df)
    ordinary = SBBRMI(brmi; mod=@__MODULE__)
    pk_only = SBBRMI(brmi; mod=@__MODULE__, held_out=:qt_y)
    prior = SBBRMI(brmi; mod=@__MODULE__, held_out=:all)

    @test BayesianRegressionModels.stan_code(pk_only) ==
          BayesianRegressionModels.stan_code(manual_hold_out(ordinary, (:qt_y,)))
    @test BayesianRegressionModels.stan_code(prior) ==
          BayesianRegressionModels.stan_code(
              manual_hold_out(ordinary, (:pk_y, :qt_y)))

    pk_only_d = brm_descriptor(pk_only)
    prior_d = brm_descriptor(prior)
    # Activity analysis removes a held-out plate response from the executable
    # data block (only `qt_y_n` remains for its generated draw), while the plan
    # retains the public selection used by replay and descriptor gating.
    @test pk_only_d.plan.held_out == Set([:qt_y])
    @test :qt_y ∉ keys(input_by_name(pk_only_d))
    @test :qt_y_n in keys(input_by_name(pk_only_d))
    @test !input_by_name(pk_only_d)[:pk_y].held_out
    @test :fit in operation_names(pk_only_d)
    @test :prior_predictive in operation_names(prior_d)

    shifted = merge(kernel_df, (; qt_y=kernel_df.qt_y .+ 0.1))
    reprocessed = brm_execute(pk_only_d, :reprocess, shifted)
    @test reprocessed.plan.held_out == Set([:qt_y])
    @test :qt_y ∉ keys(input_by_name(reprocessed))

    # The cell-local alias is accepted too, but the recorded public state is
    # the actual Stan/dataframe source that reprocess must mark again.
    alias = SBBRMI(brmi; mod=@__MODULE__, held_out=:qt_obs)
    @test alias.held_out == Set([:qt_y])
    @test BayesianRegressionModels.stan_code(alias) ==
          BayesianRegressionModels.stan_code(pk_only)
end

# A shared `|ID|` block whose marginal scales carry an explicit `sd(...)` prior
# samples `tau` through the custom `@lpxf brm_ranef_sd` family. When the `@brm`
# formula carries NO observation `~` statement (a `regime="prior"` program:
# nothing likelihood-shaped is in the trace) StanBlocks lowers the whole program
# to generated quantities (fixed_param) and RE-DRAWS every `tau` from
# `brm_ranef_sd_rng`. Without that predictive companion, trace fails loudly at
# `stan_code` naming exactly this signature to add (stanblocks-use §8/§34); this
# pins that it is present and the program empties its `parameters` block.
#
# This is NOT `held_out=:all`: that marks the response with `maybecv`, which
# drops the likelihood but keeps the parameters SAMPLED from their priors under
# NUTS (dimension > 0, covered by the top-level testset above). The fixed_param
# prior is the response-free formula, exactly as a consumer (Bruno's
# `regime="prior"`) builds it by dropping the observation statements.
#
# The whole-vector prior retains distinct half-Normal, Exponential, and
# half-Normal(scale=2) coordinates. Its sized RNG draws the same vector in GQ.
@testset "regime=prior shared |ID| sd() retains joint vector draws" begin
    ranef_sd_df = (;
        x = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        z = [0.3, -0.2, 0.5, -0.4, 0.1, 0.0],
        subject = [1, 1, 2, 2, 3, 3],
    )

    # Fitted twin (a response likelihood present): `tau` stays a SAMPLED
    # parameter scored by the custom family's density in the model block.
    fitted = @brm begin
        eta ~ 1 + x + z + (1 + x + z | p | subject)
        sd(eta, p, x) ~ Exponential(1 / 3)
        sd(eta, p, z) ~ Normal(0, 2)
        cor(:, p) ~ LKJCholesky(3, 2)
        y ~ Normal(eta, 1)
    end
    fitted_df = (; ranef_sd_df..., y = [-2.4, -2.2, -2.0, -1.8, -1.7, -1.5])
    fitted_code = BayesianRegressionModels.stan_code(SBBRMI(fitted(fitted_df); mod=@__MODULE__))
    @test occursin(r"brm_vector_prior_[0-9a-f]+_lpdf", fitted_code)
    @test occursin(r"~\s*brm_vector_prior_[0-9a-f]+\(", fitted_code)

    # Prior program: same block, NO `y ~ ...`.
    prior = @brm begin
        eta ~ 1 + x + z + (1 + x + z | p | subject)
        sd(eta, p, x) ~ Exponential(1 / 3)
        sd(eta, p, z) ~ Normal(0, 2)
        cor(:, p) ~ LKJCholesky(3, 2)
    end
    sb = SBBRMI(prior(ranef_sd_df); mod=@__MODULE__)
    @test StanBlocks.stan.transpiles(sb.model)
    prior_code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stanc_check(prior_code; warn_pedantic=false).ok
    @test occursin(r"brm_vector_prior_[0-9a-f]+_vector_rng", prior_code)
    @test !occursin(r"~\s*brm_vector_prior_[0-9a-f]+\(", prior_code)
    # Every parameter, `tau` included, is a GQ draw -> the `parameters` block is
    # empty (the fixed_param program has zero sampler dimensions).
    @test occursin(r"parameters\s*\{\s*\}", prior_code)

    prior_problem = StanBlocks.stan_instantiate(sb.model)
    @test StanBlocks.LogDensityProblems.dimension(prior_problem) == 0
    @test isfinite(StanBlocks.LogDensityProblems.logdensity(prior_problem, Float64[]))

    # Bruno ARV-393 exact shape: a block-wide `sd(:, p) ~ Exponential(2/3)`
    # emits Exponential with rate = 1.5 (the Distributions
    # `Exponential(scale=2/3)` -> Stan rate-1.5 conversion) for every margin.
    # Each is re-drawn per element as `exponential(rate[i])`, matching the
    # density's own `exponential_lpdf(tau[i], rate[i])` -- the per-family
    # rng<->lpdf agreement the consumer requires.
    bruno = @brm begin
        eta ~ 1 + x + z + (1 + x + z | p | subject)
        sd(:, p) ~ Exponential(2 / 3)
    end
    bruno_code = BayesianRegressionModels.stan_code(SBBRMI(bruno(ranef_sd_df); mod=@__MODULE__))
    @test StanBlocks.stanc_check(bruno_code; warn_pedantic=false).ok
    @test occursin(r"brm_vector_prior_[0-9a-f]+_vector_rng", bruno_code)
    @test !occursin(r"~\s*brm_vector_prior_[0-9a-f]+\(", bruno_code)
    @test occursin("exponential", bruno_code)
    @test occursin(r"brm_vector_prior_[0-9a-f]+_vector_rng\([^;]*1\.5,\s*1\.5,\s*1\.5\)", bruno_code)
end
