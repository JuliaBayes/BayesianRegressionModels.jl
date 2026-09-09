# test/prior_only_coordinates.jl — descriptor coordinate resolvers are
# generated-aware: a response-free (regime="prior") `@brm` program moves every
# population/categorical/term/shared-|ID|-scale carrier from `parameters` into
# generated quantities (StanBlocks' prior-predictive `_rng` lowering), and the
# public resolvers must still address them against BridgeStan CONSTRAINED names.
#
# Run: julia --project=test test/prior_only_coordinates.jl
#
# Scope (one of each, both fitted and prior constrained axes, plus a fail-closed
# ambiguity/missing-axis case): one numeric population coefficient, one
# categorical contrast block, one HSGP term-internal carrier, one shared-|ID|
# per-margin `tau`.
using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, LKJCholesky, Normal

const _MOD = @__MODULE__
const BRM = BayesianRegressionModels

# --- unit coverage of the carrier-selection rule --------------------------
# A real BRM descriptor never emits both a sampled `:parameter` and a
# `:generated_quantity` twin for one declaration, so the prefer-sampled and
# sampled-ambiguity behaviors of `_brm_carrier_indices` are exercised directly
# on synthetic `BRMOutput`s. Only `.kind` matters to the helper.
_mk(name, kind) = BRM.BRMOutput(
    name, kind, :real, (), NamedTuple(), :none, nothing,
    :population_effect, nothing, nothing, nothing)

@testset "carrier selection: prefer sampled, fail-closed on sampled ambiguity" begin
    ci = BRM._brm_carrier_indices
    # A sampled parameter beside a GQ twin: the parameter wins, GQ ignored.
    @test ci([_mk(:a, :parameter), _mk(:b, :generated_quantity)], _ -> true) == [1]
    # A prior program has only the GQ carrier: it is selected.
    @test ci([_mk(:g, :generated_quantity)], _ -> true) == [1]
    # Two sampled parameters: BOTH returned (length 2), so the resolver's
    # `length(...) == 1` check fails closed; the GQ fallback fires only on an
    # EMPTY parameter match and cannot mask this ambiguity.
    @test ci([_mk(:a, :parameter), _mk(:b, :parameter), _mk(:c, :generated_quantity)],
             _ -> true) == [1, 2]
    # A transformed parameter is not a draw carrier here.
    @test ci([_mk(:t, :transformed_parameter)], _ -> true) == Int[]
    # The predicate still filters: a non-matching carrier is not selected.
    @test ci([_mk(:a, :parameter)], o -> o.name === :b) == Int[]
end

@testset "prior-only joint R2D2 categorical coordinates select beta, not scale" begin
    # Inciting Bruno regime: with no observations, both the categorical beta
    # and its joint-R2D2 derived scale move to generated quantities under the
    # same declaration. Declaration ownership alone therefore finds two GQ
    # internals; the public address must follow the categorical `beta` binding.
    df = (;
        weight=[-1.2, -0.5, 0.1, 0.8, 1.5, -0.3, 0.4, 1.1],
        arm=[1, 2, 3, 1, 2, 3, 1, 2],
        subject=[1, 1, 2, 2, 3, 3, 4, 4],
    )
    builder = @brm begin
        log_CL ~ 1 + weight + factor(arm) + (1 | p | subject)
        log_V  ~ 1 + weight + (1 | p | subject)
        sd(:, p) ~ r2d2(reference_scale=1.0,
                        include=(:population, :contrasts))
        cor(:, p) ~ LKJCholesky(2, 2)
    end
    d = brm_descriptor(SBBRMI(builder(df); mod=_MOD))
    block = :cat_log_CL_arm
    owned = [o for o in d.outputs
             if !isnothing(o.declaration) && o.declaration.target === block]
    @test Set(o.name for o in owned) == Set((
        :cat_log_CL_arm_beta,
        :cat_log_CL_arm_r2d2_beta_scale,
        :cat_log_CL_arm,
    ))
    @test all(o -> o.kind === :generated_quantity, owned)

    names = ["cat_log_CL_arm_beta.1", "cat_log_CL_arm_beta.2"]
    contrast = brm_population_effect_coordinates(
        d, :log_CL, names; coefficient=:arm)
    @test contrast.output.name === :cat_log_CL_arm_beta
    @test contrast.coordinates == [1, 2]
    @test contrast.reference_level == 1
    @test contrast.nonreference_levels == [2, 3]
end

df = (;
    weight  = [-1.2, -0.5, 0.1, 0.8, 1.5, -0.3, 0.4, 1.1],
    arm     = [1, 2, 3, 1, 2, 3, 1, 2],
    conc    = [0.2, 0.5, 0.9, 1.3, 1.8, 2.4, 3.1, 3.9],
    subject = [1, 1, 2, 2, 3, 3, 4, 4],
)
fitted_df = merge(df, (; y = [0.1, -0.2, 0.4, 0.7, 1.1, -0.1, 0.3, 0.9]))

# Fitted: an observation is present, so every carrier is a sampled parameter.
fitted = @brm begin
    sigma  ~ Exponential(1)
    log_CL ~ 1 + weight + factor(arm) + hsgp(conc; k=5) + (1 + weight | p | subject)
    sd(:, p)  ~ Exponential(2/3)
    cor(:, p) ~ LKJCholesky(2, 2)
    y ~ Normal(log_CL, sigma)
end

# Prior: SAME structure, NO observation `~` — `parameters {}` empty, all in GQ.
prior = @brm begin
    log_CL ~ 1 + weight + factor(arm) + hsgp(conc; k=5) + (1 + weight | p | subject)
    sd(:, p)  ~ Exponential(2/3)
    cor(:, p) ~ LKJCholesky(2, 2)
end

@testset "prior-only coordinate resolution (generated-aware)" begin
    fitted_sb = SBBRMI(fitted(fitted_df); mod=_MOD)
    prior_sb  = SBBRMI(prior(df); mod=_MOD)

    # The prior program is genuinely the fixed_param / GQ regime.
    prior_code = BayesianRegressionModels.stan_code(prior_sb)
    @test StanBlocks.stanc_check(prior_code; warn_pedantic=false).ok
    @test occursin(r"parameters\s*\{\s*\}", prior_code)

    fitted_d = brm_descriptor(fitted_sb)
    prior_d  = brm_descriptor(prior_sb)

    fitted_prob = StanBlocks.stan_instantiate(fitted_sb.model)
    prior_prob  = StanBlocks.stan_instantiate(prior_sb.model)
    @test StanBlocks.LogDensityProblems.dimension(prior_prob) == 0

    fitted_names = StanBlocks.BridgeStan.param_names(
        fitted_prob.model; include_tp=true, include_gq=false)
    # The prior carriers live in generated quantities.
    prior_names = StanBlocks.BridgeStan.param_names(
        prior_prob.model; include_tp=true, include_gq=true)

    # 1. Numeric population coefficient (weight).
    for (d, names) in ((fitted_d, fitted_names), (prior_d, prior_names))
        r = brm_population_effect_coordinates(d, :log_CL, names; coefficient=:weight)
        @test length(r.coordinates) == 1
        @test all(startswith(string(r.output.name) * "."),
                  names[r.coordinates])
    end

    # 2. Categorical contrast block (arm -> K-1 = 2 treatment contrasts).
    for (d, names) in ((fitted_d, fitted_names), (prior_d, prior_names))
        r = brm_population_effect_coordinates(d, :log_CL, names; coefficient=:arm)
        @test length(r.coordinates) == 2
        @test length(r.contrasts) == 2
        @test all(startswith(string(r.output.name) * "."),
                  names[r.coordinates])
    end

    # 3. HSGP term-internal carrier (basis weights).
    for (d, names) in ((fitted_d, fitted_names), (prior_d, prior_names))
        r = brm_term_coordinates(d, :log_CL, names;
                                 term=:hsgp_conc, parameter=:basis_weights)
        @test length(r.coordinates) == 5
        @test all(startswith(string(r.output.name) * "."),
                  names[r.coordinates])
    end

    # 4. Shared-|ID| per-margin tau (Intercept margin).
    for (d, names) in ((fitted_d, fitted_names), (prior_d, prior_names))
        r = brm_ranef_sd_coordinates(d, :log_CL, names; id=:p, coefficient=:Intercept)
        @test length(r.coordinates) == 1
        @test startswith(names[only(r.coordinates)], string(r.output.name) * ".")
    end

    # The exact empirical gap: on the all-GQ prior program the block RESULT
    # assignment (`pop_log_CL` / `cat_log_CL_arm`, name === declaration.target)
    # is ALSO emitted as a generated quantity under the same target, beside the
    # coefficient/contrast vector. The resolver must select the INTERNAL vector
    # (name !== target) and leave the block-result twin present-but-excluded.
    for coefficient in (:weight, :arm)
        r = brm_population_effect_coordinates(prior_d, :log_CL, prior_names;
                                              coefficient=coefficient)
        block = r.output.declaration.target
        @test r.output.name !== block                      # selected the INTERNAL vector
        twins = [o for o in prior_d.outputs
                 if !isnothing(o.declaration) && o.declaration.target === block]
        @test any(o -> o.name === block, twins)            # block-result twin IS present
        @test all(o -> o.kind === :generated_quantity, twins)  # both are GQ in a prior program
        @test count(o -> o.name !== block, twins) == 1     # exactly one internal carrier
    end

    # 5. Fail-closed under the prior axis: an unknown coefficient errors, and a
    # constrained-name vector that OMITS the generated-quantities axis
    # (`include_gq=false`, so the GQ carrier is absent) errors rather than
    # returning an empty slice.
    prior_names_no_gq = StanBlocks.BridgeStan.param_names(
        prior_prob.model; include_tp=true, include_gq=false)
    @test_throws "occurs 0 times" brm_population_effect_coordinates(
        prior_d, :log_CL, prior_names; coefficient=:nonexistent)
    @test_throws "Re-reflect" brm_population_effect_coordinates(
        prior_d, :log_CL, prior_names_no_gq; coefficient=:weight)
end
